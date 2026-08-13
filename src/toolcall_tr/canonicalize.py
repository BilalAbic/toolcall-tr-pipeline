"""Convert adapted records into strict canonical target episodes."""

from __future__ import annotations

from typing import cast

from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator

from toolcall_tr.hashing import JsonValue, sha256_jcs
from toolcall_tr.ids import episode_id
from toolcall_tr.models import (
    AdaptedConversation,
    Annotations,
    CanonicalEpisode,
    Decision,
    DecisionAction,
    PipelineProvenance,
    Provenance,
    Quality,
    SourceProvenance,
)
from toolcall_tr.source import BronzeRecord
from toolcall_tr.tool_registry import normalize_tool


class CanonicalizationError(ValueError):
    """A canonicalization rejection with a stable catalog code and pointer."""

    def __init__(self, code: str, message: str, pointer: str) -> None:
        super().__init__(message)
        self.code = code
        self.pointer = pointer


def canonicalize(
    bronze: BronzeRecord,
    adapted: AdaptedConversation,
    *,
    run_event_id: str,
    parent_episode_id: str | None = None,
    parent_variant_id: str | None = None,
) -> CanonicalEpisode:
    if bronze.status != "valid" or bronze.parsed_record is None:
        raise ValueError("only valid bronze records can be canonicalized")
    canonical_tools = [normalize_tool(tool).canonical for tool in adapted.tools]
    identifier = episode_id(
        bronze.source_occurrence_id,
        adapted.source_conversation_id,
        adapted.target_message_index,
    )
    target = adapted.conversation[adapted.target_message_index]
    target_calls = target.tool_calls or []
    tool_by_name: dict[str, list[str]] = {}
    for tool in canonical_tools:
        tool_by_name.setdefault(tool.function.name, []).append(tool.tool_id)
    resolved_tool_ids: list[str] = []
    for call_index, call in enumerate(target_calls):
        candidates = tool_by_name.get(call.function.name, [])
        if len(candidates) != 1:
            raise CanonicalizationError(
                "TOOL_NAME_UNRESOLVED",
                "Tool call name does not resolve exactly once in the presented tools.",
                f"/conversation/{adapted.target_message_index}/tool_calls/{call_index}/function/name",
            )
        resolved_tool_ids.append(candidates[0])
        tool = next(tool for tool in canonical_tools if tool.tool_id == candidates[0])
        validator = cast(Validator, Draft202012Validator(tool.function.parameters))
        errors = list(validator.iter_errors(call.function.arguments))
        if errors:
            raise CanonicalizationError(
                "TOOL_ARGUMENT_SCHEMA_INVALID",
                "Tool arguments violate the resolved parameter schema.",
                f"/conversation/{adapted.target_message_index}/tool_calls/{call_index}/function/arguments",
            )
    call_shape = None
    if adapted.decision_action is DecisionAction.TOOL_CALL:
        call_shape = "single" if len(target_calls) == 1 else "multi_same_turn"
    decision = Decision(
        action=adapted.decision_action,
        call_shape=call_shape,
        call_ids=[call.id for call in target_calls],
        resolved_tool_ids=resolved_tool_ids,
        missing_required_parameters=[],
        evidence_status="source_explicit",
    )
    annotations = Annotations(
        source_conversation_id=adapted.source_conversation_id,
        target_message_index=adapted.target_message_index,
        parent_episode_id=parent_episode_id,
        decision=decision,
        trajectory_state=(
            "awaiting_tool" if adapted.decision_action is DecisionAction.TOOL_CALL else "complete"
        ),
        execution_topology="unknown",
    )
    source_fingerprint_payload = {
        "source_language_conversation": [
            message.model_dump(mode="json", exclude_none=False) for message in adapted.conversation
        ],
        "presented_tools_in_source_order": [
            tool.model_dump(mode="json", exclude_none=False) for tool in canonical_tools
        ],
        "target_decision_and_output": {
            "decision": decision.model_dump(mode="json", exclude_none=False),
            "target": target.model_dump(mode="json", exclude_none=False),
        },
    }
    source_fingerprint = sha256_jcs(source_fingerprint_payload)
    provenance = Provenance(
        sources=[
            SourceProvenance(
                dataset_namespace=bronze.dataset_namespace,
                snapshot_id=bronze.snapshot_id,
                source_occurrence_id=bronze.source_occurrence_id,
                source_sequence=bronze.source_sequence,
                source_native_id=bronze.source_native_id,
                raw_record_sha256=bronze.raw_record_sha256,
                observed_paths=adapted.observed_paths,
            )
        ],
        pipeline=PipelineProvenance(run_event_id=run_event_id),
        transformations=[],
    )
    variant_payload: dict[str, JsonValue] = {
        "episode_id": identifier,
        "conversation": cast(
            list[JsonValue],
            [
                message.model_dump(mode="json", exclude_none=False)
                for message in adapted.conversation
            ],
        ),
        "tools": cast(
            list[JsonValue],
            [tool.model_dump(mode="json", exclude_none=False) for tool in canonical_tools],
        ),
        "annotations": cast(
            dict[str, JsonValue], annotations.model_dump(mode="json", exclude_none=False)
        ),
    }
    variant_id = sha256_jcs(variant_payload)
    return CanonicalEpisode(
        episode_id=identifier,
        source_episode_fingerprint=source_fingerprint,
        variant_id=variant_id,
        parent_variant_id=parent_variant_id,
        conversation=adapted.conversation,
        tools=canonical_tools,
        provenance=provenance,
        annotations=annotations,
        quality=Quality(state="unreviewed", flags=[]),
    )
