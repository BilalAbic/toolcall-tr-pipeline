"""Strict, frozen Pydantic contracts for the canonical pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, cast

from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from toolcall_tr.constants import CANONICAL_SCHEMA_VERSION, PIPELINE_VERSION
from toolcall_tr.hashing import JsonValue

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
EpisodeId = Annotated[str, StringConstraints(pattern=r"^ep_[0-9a-f]{64}$")]
ToolId = Annotated[str, StringConstraints(pattern=r"^tool_[0-9a-f]{64}$")]
OccurrenceId = Annotated[str, StringConstraints(pattern=r"^occ_[0-9a-f]{64}$")]
SnapshotId = Annotated[str, StringConstraints(pattern=r"^snap_[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Role(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class DecisionAction(StrEnum):
    TOOL_CALL = "tool_call"
    CLARIFICATION = "clarification"
    TOOL_UNAVAILABLE = "tool_unavailable"
    DIRECT_ANSWER = "direct_answer"
    FINAL_ANSWER = "final_answer"


class ImageRef(StrictModel):
    id: NonEmptyStr
    uri: NonEmptyStr
    mime_type: NonEmptyStr
    sha256: Sha256
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    anchor: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def reject_unsafe_uri(self) -> ImageRef:
        uri = self.uri.replace("\\", "/")
        if uri.startswith(("/", "//")) or "://" in uri or ".." in uri.split("/"):
            raise ValueError("image uri must be repository-relative or content-addressed")
        return self


class FunctionCall(StrictModel):
    name: NonEmptyStr
    arguments: dict[str, JsonValue]


class ToolCall(StrictModel):
    id: NonEmptyStr
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(StrictModel):
    role: Role
    content: str | None
    reasoning_content: str | None
    thinking: str | None
    tool_calls: list[ToolCall] | None
    images: list[ImageRef] | None
    name: str | None
    tool_call_id: str | None

    @model_validator(mode="after")
    def validate_role_matrix(self) -> Message:
        if self.role in {Role.SYSTEM, Role.DEVELOPER, Role.USER}:
            if (
                self.tool_calls is not None
                or self.tool_call_id is not None
                or self.name is not None
            ):
                raise ValueError(f"{self.role} cannot carry tool call metadata")
        elif self.role is Role.ASSISTANT:
            if self.tool_call_id is not None or self.name is not None:
                raise ValueError("assistant cannot carry tool result metadata")
            has_payload = any(
                value is not None and value != []
                for value in (self.content, self.reasoning_content, self.thinking, self.tool_calls)
            )
            if not has_payload:
                raise ValueError("assistant must contain text, reasoning, thinking, or tool calls")
        elif self.role is Role.TOOL:
            if self.content is None or not self.tool_call_id:
                raise ValueError("tool message requires content and tool_call_id")
            if (
                self.tool_calls is not None
                or self.reasoning_content is not None
                or self.thinking is not None
            ):
                raise ValueError("tool message cannot contain assistant-only fields")
        return self


class ToolFunction(StrictModel):
    name: NonEmptyStr
    description: str | None
    parameters: dict[str, JsonValue]
    strict: bool | None


class CanonicalTool(StrictModel):
    tool_id: ToolId
    raw_schema_hash: Sha256
    semantic_schema_hash: Sha256
    documentation_hash: Sha256
    normalizer_version: Literal["0.1.0"] = "0.1.0"
    type: Literal["function"] = "function"
    function: ToolFunction


class SourceProvenance(StrictModel):
    dataset_namespace: NonEmptyStr
    snapshot_id: SnapshotId
    source_occurrence_id: OccurrenceId
    source_sequence: Annotated[int, Field(gt=0)]
    source_native_id: str | None
    raw_record_sha256: Sha256
    observed_paths: list[str]


class PipelineProvenance(StrictModel):
    version: Literal["0.1.0"] = PIPELINE_VERSION
    run_event_id: NonEmptyStr


class Transformation(StrictModel):
    transformation_id: NonEmptyStr
    version: NonEmptyStr
    input_pointers: list[str]
    output_pointer: str


class Provenance(StrictModel):
    sources: Annotated[list[SourceProvenance], Field(min_length=1)]
    pipeline: PipelineProvenance
    transformations: list[Transformation]


class Decision(StrictModel):
    action: DecisionAction
    call_shape: Literal["single", "multi_same_turn"] | None
    call_ids: list[str]
    resolved_tool_ids: list[ToolId]
    missing_required_parameters: list[str]
    evidence_status: Literal["source_explicit", "source_derived", "human_adjudicated", "unknown"]

    @model_validator(mode="after")
    def validate_action_shape(self) -> Decision:
        if self.action is DecisionAction.TOOL_CALL:
            if self.call_shape is None or not self.call_ids or not self.resolved_tool_ids:
                raise ValueError("tool_call requires shape, call IDs, and resolved tool IDs")
            expected = "single" if len(self.call_ids) == 1 else "multi_same_turn"
            if self.call_shape != expected or len(self.call_ids) != len(self.resolved_tool_ids):
                raise ValueError("tool_call shape/ID cardinality mismatch")
        elif self.call_shape is not None or self.call_ids or self.resolved_tool_ids:
            raise ValueError("non-tool decisions cannot contain call metadata")
        return self


class Annotations(StrictModel):
    source_conversation_id: NonEmptyStr
    target_message_index: Annotated[int, Field(ge=0)]
    parent_episode_id: EpisodeId | None
    decision: Decision
    trajectory_state: Literal["complete", "awaiting_tool", "truncated", "failed"]
    execution_topology: Literal["unknown", "sequential", "parallel"]


class Quality(StrictModel):
    state: Literal[
        "unreviewed",
        "deterministic_failed",
        "model_review",
        "human_accepted",
        "human_rejected",
        "quarantined",
    ]
    flags: list[str]


class CanonicalEpisode(StrictModel):
    schema_version: Literal["0.1.0"] = CANONICAL_SCHEMA_VERSION
    episode_id: EpisodeId
    source_episode_fingerprint: Sha256
    variant_id: Sha256
    parent_variant_id: Sha256 | None
    conversation: Annotated[list[Message], Field(min_length=1)]
    tools: list[CanonicalTool]
    provenance: Provenance
    annotations: Annotations
    quality: Quality

    @model_validator(mode="after")
    def validate_episode_state_machine(self) -> CanonicalEpisode:
        target_index = self.annotations.target_message_index
        if target_index != len(self.conversation) - 1:
            raise ValueError("canonical episode must end at its target message")
        target = self.conversation[target_index]
        if target.role is not Role.ASSISTANT:
            raise ValueError("target message must be assistant")

        tool_by_name: dict[str, CanonicalTool] = {}
        duplicate_names: set[str] = set()
        for tool in self.tools:
            name = tool.function.name
            if name in tool_by_name:
                duplicate_names.add(name)
            tool_by_name[name] = tool

        open_calls: dict[str, str] = {}
        seen_call_ids: set[str] = set()
        for message in self.conversation:
            if message.role is Role.ASSISTANT and message.tool_calls:
                for call in message.tool_calls:
                    if call.id in seen_call_ids:
                        raise ValueError("tool call IDs must be unique within an episode")
                    seen_call_ids.add(call.id)
                    if (
                        call.function.name in duplicate_names
                        or call.function.name not in tool_by_name
                    ):
                        raise ValueError("tool call name must resolve exactly once")
                    tool = tool_by_name[call.function.name]
                    validator = cast(Validator, Draft202012Validator(tool.function.parameters))
                    errors = list(validator.iter_errors(call.function.arguments))
                    if errors:
                        raise ValueError(f"tool arguments violate schema: {errors[0].message}")
                    open_calls[call.id] = call.function.name
            elif message.role is Role.TOOL:
                call_id = message.tool_call_id
                if call_id not in open_calls:
                    raise ValueError("orphan or repeated tool result")
                if message.name is not None and message.name != open_calls[call_id]:
                    raise ValueError("tool result name does not match opened call")
                del open_calls[call_id]

        decision = self.annotations.decision
        if decision.action is DecisionAction.TOOL_CALL:
            target_ids = [call.id for call in target.tool_calls or []]
            if target.content is not None or target_ids != decision.call_ids:
                raise ValueError("tool-call target and decision metadata differ")
            if self.annotations.trajectory_state != "awaiting_tool" or not open_calls:
                raise ValueError("tool-call-only target must be awaiting_tool")
        else:
            if target.tool_calls not in (None, []) or target.content is None:
                raise ValueError("text decision requires assistant content and no calls")
            if self.annotations.trajectory_state != "complete" or open_calls:
                raise ValueError("text/final decision must be complete with no open calls")
        return self


class RawToolDefinition(StrictModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class AdaptedConversation(StrictModel):
    source_conversation_id: NonEmptyStr
    conversation: Annotated[list[Message], Field(min_length=1)]
    tools: list[RawToolDefinition]
    target_message_index: Annotated[int, Field(ge=0)]
    decision_action: DecisionAction
    observed_paths: list[str]
