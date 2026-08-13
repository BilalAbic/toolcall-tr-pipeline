"""Deterministic source-semantic Pass 1 evidence without model inference."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from toolcall_tr.diagnostics import Diagnostic, diagnostic
from toolcall_tr.hashing import JsonValue
from toolcall_tr.models import (
    CanonicalEpisode,
    DecisionAction,
    EpisodeId,
    NonEmptyStr,
    Role,
    StrictModel,
    ToolId,
)

type ArgumentOrigin = Literal[
    "explicit_user",
    "prior_turn",
    "tool_result",
    "system_context",
    "deterministic_default",
    "derived",
    "must_not_infer",
    "unknown",
]


class ArgumentEvidenceInput(StrictModel):
    """Explicit evidence supplied by an adapter or human; never inferred from text."""

    call_id: NonEmptyStr
    argument_pointer: Annotated[str, Field(pattern=r"^/(?:[^/~]|~[01])+(?:/[^/]*)*$")]
    origin: ArgumentOrigin
    evidence_pointers: list[str]
    transformation_id: str | None = None
    input_pointers: list[str]

    @model_validator(mode="after")
    def validate_origin_shape(self) -> ArgumentEvidenceInput:
        if self.origin in {"unknown", "must_not_infer"}:
            if self.evidence_pointers or self.transformation_id or self.input_pointers:
                raise ValueError(f"{self.origin} cannot claim evidence")
        elif self.origin == "derived":
            if not self.transformation_id or not self.input_pointers:
                raise ValueError("derived evidence requires a transformation and inputs")
        elif not self.evidence_pointers:
            raise ValueError(f"{self.origin} requires at least one evidence pointer")
        return self


class SourceEvidenceRequest(StrictModel):
    """One explicitly accounted Pass 1 request for a canonical episode."""

    schema_version: Literal["source-evidence-request-0.1.0"] = "source-evidence-request-0.1.0"
    episode_id: EpisodeId
    argument_evidence: list[ArgumentEvidenceInput]


class DeterministicCheck(StrictModel):
    check_id: NonEmptyStr
    status: Literal["passed", "failed", "review"]
    json_pointer: str | None
    message: NonEmptyStr


class ArgumentProvenance(StrictModel):
    call_id: NonEmptyStr
    argument_pointer: NonEmptyStr
    origin: ArgumentOrigin
    evidence_pointers: list[str]
    transformation_id: str | None
    input_pointers: list[str]


class AcceptableBehavior(StrictModel):
    action: DecisionAction
    tool_ids: list[ToolId]
    authority: Literal["source_explicit", "human_adjudicated"]


class EvidenceClaim(StrictModel):
    kind: Literal["argument_grounding"]
    status: Literal["supported", "unsupported", "unresolved"]
    source_pointers: list[str]
    target_pointer: NonEmptyStr


class SourceEvidence(StrictModel):
    schema_version: Literal["source-evidence-0.1.0"] = "source-evidence-0.1.0"
    episode_id: EpisodeId
    deterministic_checks: Annotated[list[DeterministicCheck], Field(min_length=1)]
    argument_provenance: list[ArgumentProvenance]
    acceptable_behaviors: Annotated[list[AcceptableBehavior], Field(min_length=1)]
    forbidden_behaviors: list[NonEmptyStr]
    claims: list[EvidenceClaim]
    diagnostics: list[Diagnostic]
    pass1_result: Literal["deterministic_pass", "deterministic_fail", "needs_semantic_review"]
    judge_verdict: Literal["not_run"] = "not_run"
    human_verdict: Literal["source_review"] = "source_review"
    review_event_id: None = None

    @model_validator(mode="after")
    def validate_result(self) -> SourceEvidence:
        statuses = {check.status for check in self.deterministic_checks}
        origins = {item.origin for item in self.argument_provenance}
        if "failed" in statuses and self.pass1_result != "deterministic_fail":
            raise ValueError("failed deterministic checks require deterministic_fail")
        if self.pass1_result == "deterministic_pass" and (
            statuses != {"passed"} or origins.intersection({"unknown", "must_not_infer"})
        ):
            raise ValueError("deterministic_pass cannot contain unresolved grounding")
        if any(
            item.authority not in {"source_explicit", "human_adjudicated"}
            for item in self.acceptable_behaviors
        ):
            raise ValueError("only source or human authority may define acceptable behavior")
        return self


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _resolve_pointer(document: JsonValue, pointer: str) -> JsonValue:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError("JSON Pointer must be absolute")
    current = document
    for encoded in pointer[1:].split("/"):
        token = _decode_pointer_token(encoded)
        if isinstance(current, dict):
            if token not in current:
                raise ValueError(f"JSON Pointer does not resolve: {pointer}")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdecimal() or int(token) >= len(current):
                raise ValueError(f"JSON Pointer does not resolve: {pointer}")
            current = current[int(token)]
        else:
            raise ValueError(f"JSON Pointer does not resolve: {pointer}")
    return current


def _leaf_pointers(value: JsonValue, pointer: str = "") -> list[str]:
    if isinstance(value, dict) and value:
        return [
            child
            for key in sorted(value)
            for child in _leaf_pointers(value[key], f"{pointer}/{_escape_pointer_token(key)}")
        ]
    if isinstance(value, list) and value:
        return [
            child
            for index, item in enumerate(value)
            for child in _leaf_pointers(item, f"{pointer}/{index}")
        ]
    return [pointer or "/"]


def _message_role_for_pointer(episode: CanonicalEpisode, pointer: str) -> Role | None:
    tokens = pointer.split("/")
    if len(tokens) < 4 or tokens[1] != "conversation" or not tokens[2].isdecimal():
        return None
    index = int(tokens[2])
    if index >= episode.annotations.target_message_index or index >= len(episode.conversation):
        return None
    return episode.conversation[index].role


def _evidence_pointer_allowed(
    episode: CanonicalEpisode,
    item: ArgumentEvidenceInput,
    episode_json: JsonValue,
) -> tuple[bool, str]:
    pointers = item.input_pointers if item.origin == "derived" else item.evidence_pointers
    try:
        for pointer in pointers:
            _resolve_pointer(episode_json, pointer)
    except ValueError as exc:
        return False, str(exc)

    roles = {_message_role_for_pointer(episode, pointer) for pointer in pointers}
    if item.origin == "explicit_user" and roles != {Role.USER}:
        return False, "explicit_user evidence must point only to prior user messages"
    if item.origin == "tool_result" and roles != {Role.TOOL}:
        return False, "tool_result evidence must point only to prior tool messages"
    if item.origin == "system_context" and not roles.issubset({Role.SYSTEM, Role.DEVELOPER}):
        return False, "system_context evidence must point to system/developer messages"
    if item.origin == "prior_turn" and (None in roles or not roles):
        return False, "prior_turn evidence must point inside the conversation prefix"
    if item.origin == "deterministic_default" and any(
        not pointer.startswith("/tools/") for pointer in pointers
    ):
        return False, "deterministic_default evidence must point into presented tools"
    if item.origin == "derived" and (None in roles or not roles):
        return False, "derived inputs must point inside the conversation prefix"
    return True, "evidence pointers resolve with the declared origin"


def build_source_evidence(
    episode: CanonicalEpisode,
    evidence: Iterable[ArgumentEvidenceInput],
) -> SourceEvidence:
    """Build Pass 1 evidence without searching or interpreting natural language."""
    episode_json = cast(JsonValue, episode.model_dump(mode="json", exclude_none=False))
    target = episode.conversation[episode.annotations.target_message_index]
    calls = target.tool_calls or []
    expected: dict[tuple[str, str], str] = {}
    for call_index, call in enumerate(calls):
        for argument_pointer in _leaf_pointers(call.function.arguments):
            pointer_suffix = argument_pointer if argument_pointer != "/" else ""
            expected[(call.id, argument_pointer)] = (
                f"/conversation/{episode.annotations.target_message_index}/tool_calls/"
                f"{call_index}/function/arguments{pointer_suffix}"
            )

    supplied: dict[tuple[str, str], ArgumentEvidenceInput] = {}
    for item in evidence:
        key = (item.call_id, item.argument_pointer)
        if key in supplied:
            raise ValueError(f"duplicate argument evidence: {item.call_id}{item.argument_pointer}")
        if key not in expected:
            argument_identity = f"{item.call_id}{item.argument_pointer}"
            raise ValueError(f"evidence does not match an expected argument: {argument_identity}")
        supplied[key] = item

    checks: list[DeterministicCheck] = [
        DeterministicCheck(
            check_id="target_assistant_decision",
            status="passed",
            json_pointer=f"/conversation/{episode.annotations.target_message_index}",
            message="Canonical target is the final assistant decision.",
        ),
        DeterministicCheck(
            check_id="presented_tool_resolution",
            status="passed",
            json_pointer="/tools",
            message="Every call resolves exactly once in the presented tool list.",
        ),
        DeterministicCheck(
            check_id="tool_argument_schema",
            status="passed",
            json_pointer=f"/conversation/{episode.annotations.target_message_index}/tool_calls",
            message="Canonical validation proved argument schema compatibility.",
        ),
    ]
    provenance: list[ArgumentProvenance] = []
    claims: list[EvidenceClaim] = []
    diagnostics: list[Diagnostic] = []
    has_review = False
    has_failure = False
    source = episode.provenance.sources[0]

    for key, target_pointer in sorted(expected.items()):
        item = supplied.get(key)
        if item is None:
            item = ArgumentEvidenceInput(
                call_id=key[0],
                argument_pointer=key[1],
                origin="unknown",
                evidence_pointers=[],
                transformation_id=None,
                input_pointers=[],
            )
        valid_pointer, message = _evidence_pointer_allowed(episode, item, episode_json)
        origin: ArgumentOrigin = item.origin if valid_pointer else "unknown"
        if not valid_pointer:
            has_failure = True
            status: Literal["supported", "unsupported", "unresolved"] = "unsupported"
        elif origin == "must_not_infer":
            has_failure = True
            status = "unsupported"
            message = "Argument exists even though policy forbids inferring it."
        elif origin in {"unknown", "derived"}:
            has_review = True
            status = "unresolved"
            message = "Argument requires semantic or transformation review."
        else:
            status = "supported"

        if status != "supported":
            diagnostics.append(
                diagnostic(
                    "SOURCE_ARG_NOT_GROUNDED",
                    message,
                    source_occurrence_id=source.source_occurrence_id,
                    episode_id=episode.episode_id,
                    json_pointer=target_pointer,
                    source_line=None,
                    evidence_refs=item.evidence_pointers or item.input_pointers,
                )
            )
        provenance.append(
            ArgumentProvenance(
                call_id=item.call_id,
                argument_pointer=item.argument_pointer,
                origin=origin,
                evidence_pointers=item.evidence_pointers if valid_pointer else [],
                transformation_id=item.transformation_id if valid_pointer else None,
                input_pointers=item.input_pointers if valid_pointer else [],
            )
        )
        claims.append(
            EvidenceClaim(
                kind="argument_grounding",
                status=status,
                source_pointers=(item.evidence_pointers or item.input_pointers)
                if valid_pointer
                else [],
                target_pointer=target_pointer,
            )
        )

    coverage_status: Literal["passed", "failed", "review"]
    if has_failure:
        coverage_status = "failed"
    elif has_review:
        coverage_status = "review"
    else:
        coverage_status = "passed"
    checks.append(
        DeterministicCheck(
            check_id="argument_provenance_coverage",
            status=coverage_status,
            json_pointer=f"/conversation/{episode.annotations.target_message_index}/tool_calls",
            message=f"Accounted for {len(provenance)} expected argument values.",
        )
    )

    if has_failure:
        result = "deterministic_fail"
    elif has_review:
        result = "needs_semantic_review"
    else:
        result = "deterministic_pass"
    behavior = AcceptableBehavior(
        action=episode.annotations.decision.action,
        tool_ids=list(episode.annotations.decision.resolved_tool_ids),
        authority="source_explicit",
    )
    return SourceEvidence(
        episode_id=episode.episode_id,
        deterministic_checks=checks,
        argument_provenance=provenance,
        acceptable_behaviors=[behavior],
        forbidden_behaviors=["guess_missing_parameter"],
        claims=claims,
        diagnostics=diagnostics,
        pass1_result=result,
    )
