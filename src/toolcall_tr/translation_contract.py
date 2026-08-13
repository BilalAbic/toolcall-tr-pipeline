"""Strict, provider-free request/response contracts for leaf-only translation.

This module deliberately contains no client, retry loop, environment lookup, or
network operation.  It defines the only leaf-segment payload that a later,
separately approved provider adapter may receive and validates its response
before any host-side merge.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.hashing import stable_id
from toolcall_tr.models import EpisodeId, NonEmptyStr, Sha256, StrictModel

SegmentId = Annotated[str, Field(pattern=r"^seg_[0-9a-f]{64}$")]
RequestId = Annotated[str, Field(pattern=r"^trq_[0-9a-f]{64}$")]
JsonPointer = Annotated[str, Field(pattern=r"^(?:/(?:[^/~]|~[01])*)*$")]
_SENTINEL_PATTERN = re.compile(r"⟪S[1-9][0-9]*_P[1-9][0-9]*⟫")


class TranslationContractError(ValueError):
    """Raised when a provider-shaped response breaks the local contract."""


class ProtectedToken(StrictModel):
    """One occurrence-specific sentinel already substituted into a source segment."""

    token: Annotated[str, Field(pattern=r"^⟪S[1-9][0-9]*_P[1-9][0-9]*⟫$")]
    occurrence: Annotated[int, Field(gt=0)]


class TranslationSegment(StrictModel):
    """A single leaf that may be translated without reconstructing source JSON."""

    segment_id: SegmentId
    path: JsonPointer
    source_text: NonEmptyStr
    protected_tokens: list[ProtectedToken]

    @model_validator(mode="after")
    def validate_tokens(self) -> TranslationSegment:
        tokens = [item.token for item in self.protected_tokens]
        occurrences = [item.occurrence for item in self.protected_tokens]
        if tokens != sorted(set(tokens)):
            raise ValueError("protected tokens must be unique and ordered")
        if occurrences != list(range(1, len(occurrences) + 1)):
            raise ValueError("protected token occurrences must be contiguous")
        if any(self.source_text.count(token) != 1 for token in tokens):
            raise ValueError("each protected token must occur once in source_text")
        observed = _SENTINEL_PATTERN.findall(self.source_text)
        if observed != tokens:
            raise ValueError("source_text sentinel sequence differs from protected_tokens")
        return self


class TranslationRequest(StrictModel):
    """Immutable request body for a provider adapter that is not implemented here."""

    schema_version: Literal["translation-request-0.1.0"] = "translation-request-0.1.0"
    request_id: RequestId
    episode_id: EpisodeId
    input_variant_id: Sha256
    source_language: Literal["en"] = "en"
    target_language: Literal["tr"] = "tr"
    field_policy_version: NonEmptyStr
    segments: Annotated[list[TranslationSegment], Field(min_length=1)]
    terminology_evidence: list[dict[str, str]]

    @model_validator(mode="after")
    def validate_identity_and_segments(self) -> TranslationRequest:
        segment_ids = [segment.segment_id for segment in self.segments]
        paths = [segment.path for segment in self.segments]
        if segment_ids != sorted(set(segment_ids)):
            raise ValueError("request segment IDs must be unique and sorted")
        if paths != sorted(set(paths)):
            raise ValueError("request segment paths must be unique and sorted")
        body = self.model_dump(mode="json", exclude={"request_id"})
        if self.request_id != stable_id("trq", body):
            raise ValueError("translation request ID does not match deterministic content")
        return self


class SegmentTranslation(StrictModel):
    segment_id: SegmentId
    target_text: NonEmptyStr
    research_needed: bool
    uncertainty_tags: list[NonEmptyStr]


class TranslationResponse(StrictModel):
    """A provider-shaped result; valid JSON alone never establishes acceptance."""

    schema_version: Literal["translation-response-0.1.0"] = "translation-response-0.1.0"
    request_id: RequestId
    status: Literal["translated", "research_needed"]
    segments: Annotated[list[SegmentTranslation], Field(min_length=1)]
    term_queries: list[NonEmptyStr]

    @model_validator(mode="after")
    def validate_status(self) -> TranslationResponse:
        segment_ids = [segment.segment_id for segment in self.segments]
        if segment_ids != sorted(set(segment_ids)):
            raise ValueError("response segment IDs must be unique and sorted")
        needs_research = any(segment.research_needed for segment in self.segments)
        if self.status == "translated" and needs_research:
            raise ValueError("translated response cannot mark a segment research_needed")
        if self.status == "research_needed" and not needs_research:
            raise ValueError("research_needed response must mark at least one segment")
        return self


def build_translation_request(
    *,
    episode_id: str,
    input_variant_id: str,
    field_policy_version: str,
    segments: list[TranslationSegment],
    terminology_evidence: list[dict[str, str]] | None = None,
) -> TranslationRequest:
    """Construct a content-addressed request after a local extraction step."""
    evidence = terminology_evidence or []
    body: dict[str, object] = {
        "schema_version": "translation-request-0.1.0",
        "episode_id": episode_id,
        "input_variant_id": input_variant_id,
        "source_language": "en",
        "target_language": "tr",
        "field_policy_version": field_policy_version,
        "segments": [segment.model_dump(mode="json") for segment in segments],
        "terminology_evidence": evidence,
    }
    return TranslationRequest(
        request_id=stable_id("trq", body),
        episode_id=episode_id,
        input_variant_id=input_variant_id,
        field_policy_version=field_policy_version,
        segments=segments,
        terminology_evidence=evidence,
    )


def validate_translation_response(
    request: TranslationRequest, response: TranslationResponse
) -> None:
    """Fail closed unless response coverage, NFC, and sentinels are exact.

    A successful return only means the response preserves the leaf-only wire
    contract. It is not a semantic quality or human-acceptance verdict.
    """
    if response.request_id != request.request_id:
        raise TranslationContractError("response request_id does not match request")
    source_by_id = {segment.segment_id: segment for segment in request.segments}
    result_by_id = {segment.segment_id: segment for segment in response.segments}
    if set(source_by_id) != set(result_by_id):
        missing = sorted(set(source_by_id) - set(result_by_id))
        extra = sorted(set(result_by_id) - set(source_by_id))
        raise TranslationContractError(
            f"response segment coverage mismatch: missing={missing} extra={extra}"
        )

    for segment_id in sorted(source_by_id):
        source = source_by_id[segment_id]
        result = result_by_id[segment_id]
        if unicodedata.normalize("NFC", result.target_text) != result.target_text:
            raise TranslationContractError(f"target text must be NFC: {segment_id}")
        expected_tokens = [token.token for token in source.protected_tokens]
        observed_tokens = _SENTINEL_PATTERN.findall(result.target_text)
        if observed_tokens != expected_tokens or any(
            result.target_text.count(token) != 1 for token in expected_tokens
        ):
            raise TranslationContractError(f"protected token mismatch: {segment_id}")
