"""Fail-closed, resumable live translation for explicit canonical JSONL input.

The provider sees one already policy-approved natural-language leaf at a time.
Source JSON is never edited: accepted text is merged by the host and every
intermediate checkpoint is immutable.  A persisted attempt without its
validated leaf result is deliberately unrecoverable in place, preventing a
possibly duplicated paid request after an interrupted run.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.artifacts import PublishError, publish_bytes_atomic, publish_jsonl_artifact
from toolcall_tr.config import PipelineConfig
from toolcall_tr.deepseek_adapter import (
    DeepSeekTranslationAdapter,
    serialize_deepseek_translation_request,
)
from toolcall_tr.field_policy import (
    FieldPolicy,
    Segment,
    SegmentExtraction,
    SegmentTranslation,
    extract_leaf_segments,
    merge_translated_segments,
)
from toolcall_tr.hashing import canonical_bytes, sha256_bytes, sha256_jcs, stable_id
from toolcall_tr.jsonio import iter_jsonl
from toolcall_tr.models import CanonicalEpisode, Sha256, StrictModel
from toolcall_tr.prompt_contract import PromptBundle
from toolcall_tr.provider_adapter import ResponsesTransport
from toolcall_tr.provider_provenance import (
    ProviderAttemptRecord,
    ProviderOperation,
)
from toolcall_tr.provider_usage import ProviderUsageRecord
from toolcall_tr.translation_contract import (
    ProtectedToken,
    TranslationRequest,
    TranslationSegment,
    build_translation_request,
)

_SENTINEL_PATTERN = re.compile(r"⟪S[1-9][0-9]*_P[1-9][0-9]*⟫")


class OperationalTranslationError(RuntimeError):
    """Raised before a source-mutating or automatic-retry path can occur."""


class LeafTranslationRecord(StrictModel):
    """Validated, host-owned outcome of exactly one leaf request."""

    schema_version: Literal["leaf-translation-record-0.1.0"] = "leaf-translation-record-0.1.0"
    leaf_result_id: Annotated[str, Field(pattern=r"^leaftr_[0-9a-f]{64}$")]
    request_id: Annotated[str, Field(pattern=r"^trq_[0-9a-f]{64}$")]
    segment_id: Annotated[str, Field(pattern=r"^seg_[0-9a-f]{64}$")]
    source_sha256: Sha256
    provider_attempt_id: Annotated[str, Field(pattern=r"^pvattempt_[0-9a-f]{64}$")]
    status: Literal["translated", "research_needed"]
    target_text: str | None

    @model_validator(mode="after")
    def validate_identity_and_state(self) -> LeafTranslationRecord:
        if (self.status == "translated") != (self.target_text is not None):
            raise ValueError(
                "translated leaf state must carry target text and research state must not"
            )
        body = self.model_dump(mode="json", exclude={"leaf_result_id"})
        if self.leaf_result_id != stable_id("leaftr", body):
            raise ValueError("leaf result ID does not match deterministic content")
        return self


class OperationalTranslationResult(StrictModel):
    """One canonical input episode, never a Gold or release decision."""

    schema_version: Literal["operational-translation-result-0.1.0"] = (
        "operational-translation-result-0.1.0"
    )
    result_id: Annotated[str, Field(pattern=r"^trresult_[0-9a-f]{64}$")]
    episode_id: Annotated[str, Field(pattern=r"^ep_[0-9a-f]{64}$")]
    input_variant_id: Sha256
    field_policy_sha256: Sha256
    prompt_id: Annotated[str, Field(pattern=r"^prompt_[0-9a-f]{64}$")]
    status: Literal["translated", "research_needed", "no_translatable_segments"]
    leaf_result_ids: list[Annotated[str, Field(pattern=r"^leaftr_[0-9a-f]{64}$")]]
    translated_episode: CanonicalEpisode | None
    promotion: Literal["not_eligible"] = "not_eligible"

    @model_validator(mode="after")
    def validate_result(self) -> OperationalTranslationResult:
        if self.leaf_result_ids != sorted(set(self.leaf_result_ids)):
            raise ValueError("leaf result IDs must be unique and sorted")
        if self.status == "translated":
            if (
                self.translated_episode is None
                or self.translated_episode.episode_id != self.episode_id
            ):
                raise ValueError("translated result must carry the same canonical episode")
        elif self.translated_episode is not None:
            raise ValueError("non-translated result cannot carry a canonical episode")
        body = self.model_dump(mode="json", exclude={"result_id"})
        if self.result_id != stable_id("trresult", body):
            raise ValueError("translation result ID does not match deterministic content")
        return self


class OperationalTranslationReport(StrictModel):
    """Terminal local receipt for a bounded batch that cannot promote data."""

    schema_version: Literal["operational-translation-0.1.0"] = "operational-translation-0.1.0"
    batch_id: Annotated[str, Field(pattern=r"^trbatch_[0-9a-f]{64}$")]
    input_file_sha256: Sha256
    field_policy_sha256: Sha256
    prompt_id: Annotated[str, Field(pattern=r"^prompt_[0-9a-f]{64}$")]
    source_records: Annotated[int, Field(ge=0)]
    translated_records: Annotated[int, Field(ge=0)]
    research_required_records: Annotated[int, Field(ge=0)]
    no_translatable_records: Annotated[int, Field(ge=0)]
    provider_attempts: Annotated[int, Field(ge=0)]
    result_manifest_id: Annotated[str, Field(pattern=r"^manifest_[0-9a-f]{64}$")]
    promotion: Literal["not_eligible"] = "not_eligible"

    @model_validator(mode="after")
    def validate_report(self) -> OperationalTranslationReport:
        if self.source_records != (
            self.translated_records + self.research_required_records + self.no_translatable_records
        ):
            raise ValueError("translation report row accounting must balance")
        body = self.model_dump(mode="json", exclude={"batch_id"})
        if self.batch_id != stable_id("trbatch", body):
            raise ValueError("translation batch ID does not match deterministic content")
        return self


@dataclass(frozen=True)
class _LeafJob:
    source_segment: Segment
    request: TranslationRequest
    request_body: bytes
    attempt_id: str


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_boundaries(input_jsonl: Path, output_root: Path) -> tuple[Path, Path]:
    input_path = input_jsonl.resolve(strict=True)
    if not input_path.is_file() or input_path.suffix.lower() != ".jsonl":
        raise OperationalTranslationError("translation input must be an existing .jsonl file")
    if output_root.exists() and not output_root.is_dir():
        raise OperationalTranslationError("translation output root must be a directory")
    resolved_output = output_root.resolve(strict=False)
    source_root = input_path.parent
    if _is_within(resolved_output, source_root) or _is_within(source_root, resolved_output):
        raise OperationalTranslationError(
            "translation output root must be disjoint from the canonical input root"
        )
    return input_path, resolved_output


def _load_episodes(input_path: Path) -> list[CanonicalEpisode]:
    episodes = [
        CanonicalEpisode.model_validate_json(canonical_bytes(record), strict=True)
        for record in iter_jsonl(input_path)
    ]
    identities = [(item.episode_id, item.variant_id) for item in episodes]
    if len(identities) != len(set(identities)):
        raise OperationalTranslationError(
            "canonical input contains duplicate episode/variant identities"
        )
    return sorted(episodes, key=lambda item: (item.episode_id, item.variant_id))


def _protected_tokens(source_text: str) -> list[ProtectedToken]:
    observed = _SENTINEL_PATTERN.findall(source_text)
    if not observed:
        return []
    if observed != sorted(set(observed)):
        raise OperationalTranslationError(
            "source sentinel tokens must be unique and in canonical order"
        )
    return [
        ProtectedToken(token=token, occurrence=index) for index, token in enumerate(observed, 1)
    ]


def _leaf_job(
    *,
    extraction: SegmentExtraction,
    segment: Segment,
    config: PipelineConfig,
    prompt: PromptBundle,
) -> _LeafJob:
    request = build_translation_request(
        episode_id=extraction.episode_id,
        input_variant_id=extraction.input_variant_id,
        field_policy_version=extraction.field_policy_version,
        segments=[
            TranslationSegment(
                segment_id=segment.segment_id,
                path=segment.json_pointer,
                source_text=segment.source_text,
                protected_tokens=_protected_tokens(segment.source_text),
            )
        ],
    )
    role = config.providers.translator
    request_body = serialize_deepseek_translation_request(
        request=request,
        prompt=prompt,
        model=role.model,
        temperature=role.temperature,
        thinking=role.thinking,
        max_output_tokens=1_024,
    )
    attempt_id = stable_id(
        "pvattempt",
        {
            "schema_version": "provider-attempt-0.1.0",
            "operation": ProviderOperation.TRANSLATION,
            "provider": role.provider,
            "model": role.model,
            "endpoint": role.endpoint or "",
            "request_sha256": sha256_bytes(request_body),
        },
    )
    return _LeafJob(
        source_segment=segment,
        request=request,
        request_body=request_body,
        attempt_id=attempt_id,
    )


def _read_model(path: Path, model: type[StrictModel]) -> StrictModel:
    return model.model_validate_json(path.read_text(encoding="utf-8"), strict=True)


def _leaf_record(
    *,
    root: Path,
    job: _LeafJob,
    config: PipelineConfig,
    prompt: PromptBundle,
    transport: ResponsesTransport,
) -> LeafTranslationRecord:
    leaf_path = root / "leaf-results" / f"{job.request.request_id}.json"
    if leaf_path.exists():
        existing = _read_model(leaf_path, LeafTranslationRecord)
        if not isinstance(existing, LeafTranslationRecord):  # pragma: no cover - static guard
            raise OperationalTranslationError("invalid persisted leaf result")
        if (
            existing.request_id != job.request.request_id
            or existing.segment_id != job.source_segment.segment_id
            or existing.source_sha256 != job.source_segment.source_sha256
            or existing.provider_attempt_id != job.attempt_id
        ):
            raise OperationalTranslationError(
                "persisted leaf result conflicts with requested leaf identity"
            )
        return existing

    attempt_path = root / "provider-attempts" / f"{job.attempt_id}.json"
    claim_path = root / "provider-claims" / f"{job.attempt_id}.json"
    if attempt_path.exists() or claim_path.exists():
        raise OperationalTranslationError(
            "a prior provider attempt lacks its validated leaf result; manual recovery required"
        )
    claim = {
        "schema_version": "provider-claim-0.1.0",
        "attempt_id": job.attempt_id,
        "request_sha256": sha256_bytes(job.request_body),
    }
    try:
        publish_bytes_atomic(claim_path, canonical_bytes(claim) + b"\n")
    except PublishError as exc:
        raise OperationalTranslationError(
            "provider attempt claim collision; no request was sent"
        ) from exc

    def persist_attempt(record: ProviderAttemptRecord) -> None:
        if record.attempt_id != job.attempt_id:
            raise OperationalTranslationError(
                "adapter emitted a provider attempt with unexpected identity"
            )
        publish_bytes_atomic(
            root / "provider-attempts" / f"{record.attempt_id}.json",
            canonical_bytes(record) + b"\n",
        )

    def persist_usage(record: ProviderUsageRecord) -> None:
        if record.attempt_id != job.attempt_id:
            raise OperationalTranslationError(
                "adapter emitted provider usage with unexpected attempt identity"
            )
        publish_bytes_atomic(
            root / "provider-usage" / f"{record.usage_id}.json",
            canonical_bytes(record) + b"\n",
        )

    response = DeepSeekTranslationAdapter(
        config=config,
        transport=transport,
        max_output_tokens=1_024,
        attempt_sink=persist_attempt,
        usage_sink=persist_usage,
    ).translate(request=job.request, prompt=prompt)
    result = response.segments[0]
    status: Literal["translated", "research_needed"] = (
        "research_needed" if response.status == "research_needed" else "translated"
    )
    body: dict[str, object] = {
        "schema_version": "leaf-translation-record-0.1.0",
        "request_id": job.request.request_id,
        "segment_id": job.source_segment.segment_id,
        "source_sha256": job.source_segment.source_sha256,
        "provider_attempt_id": job.attempt_id,
        "status": status,
        "target_text": result.target_text if status == "translated" else None,
    }
    leaf = LeafTranslationRecord(
        leaf_result_id=stable_id("leaftr", body),
        request_id=job.request.request_id,
        segment_id=job.source_segment.segment_id,
        source_sha256=job.source_segment.source_sha256,
        provider_attempt_id=job.attempt_id,
        status=status,
        target_text=result.target_text if status == "translated" else None,
    )
    publish_bytes_atomic(leaf_path, canonical_bytes(leaf) + b"\n")
    return leaf


def _result(
    *,
    episode: CanonicalEpisode,
    policy: FieldPolicy,
    prompt: PromptBundle,
    extraction: SegmentExtraction,
    leaf_records: Iterable[LeafTranslationRecord],
) -> OperationalTranslationResult:
    records = sorted(leaf_records, key=lambda item: item.leaf_result_id)
    field_policy_sha = sha256_jcs(policy)
    if not extraction.segments:
        status: Literal["translated", "research_needed", "no_translatable_segments"] = (
            "no_translatable_segments"
        )
        translated: CanonicalEpisode | None = None
    elif any(item.status == "research_needed" for item in records):
        status = "research_needed"
        translated = None
    else:
        status = "translated"
        translated = merge_translated_segments(
            episode,
            policy,
            extraction,
            [
                SegmentTranslation(segment_id=item.segment_id, target_text=item.target_text or "")
                for item in records
            ],
        )
    body: dict[str, object] = {
        "schema_version": "operational-translation-result-0.1.0",
        "episode_id": episode.episode_id,
        "input_variant_id": episode.variant_id,
        "field_policy_sha256": field_policy_sha,
        "prompt_id": prompt.prompt_id,
        "status": status,
        "leaf_result_ids": [item.leaf_result_id for item in records],
        "translated_episode": (
            translated.model_dump(mode="json", exclude_none=False)
            if translated is not None
            else None
        ),
        "promotion": "not_eligible",
    }
    return OperationalTranslationResult(
        result_id=stable_id("trresult", body),
        episode_id=episode.episode_id,
        input_variant_id=episode.variant_id,
        field_policy_sha256=field_policy_sha,
        prompt_id=prompt.prompt_id,
        status=status,
        leaf_result_ids=[item.leaf_result_id for item in records],
        translated_episode=translated,
    )


def run_operational_translation(
    input_jsonl: Path,
    output_root: Path,
    *,
    config: PipelineConfig,
    field_policy: FieldPolicy,
    prompt: PromptBundle,
    transport: ResponsesTransport,
) -> OperationalTranslationReport:
    """Translate explicit canonical JSONL through a single-shot injected transport.

    ``config`` must have both live gates set.  This function never reads a key,
    never retries, and never writes into the input tree.  It purposefully emits
    only intermediate translation evidence; review, Gold, and release remain
    separate human-gated stages.
    """
    if not config.providers.enabled or not config.providers.network_egress_enabled:
        raise OperationalTranslationError("translation requires both live provider gates")
    role = config.providers.translator
    if role.provider != "deepseek" or role.endpoint is None:
        raise OperationalTranslationError(
            "translation requires an explicit DeepSeek translator endpoint"
        )
    input_path, root = _validate_boundaries(input_jsonl, output_root)
    input_digest = sha256_bytes(input_path.read_bytes())
    episodes = _load_episodes(input_path)
    results: list[OperationalTranslationResult] = []
    for episode in episodes:
        extraction = extract_leaf_segments(episode, field_policy)
        leaf_records = [
            _leaf_record(
                root=root,
                job=_leaf_job(
                    extraction=extraction,
                    segment=segment,
                    config=config,
                    prompt=prompt,
                ),
                config=config,
                prompt=prompt,
                transport=transport,
            )
            for segment in extraction.segments
        ]
        result = _result(
            episode=episode,
            policy=field_policy,
            prompt=prompt,
            extraction=extraction,
            leaf_records=leaf_records,
        )
        publish_bytes_atomic(
            root / "episode-results" / f"{result.result_id}.json",
            canonical_bytes(result) + b"\n",
        )
        results.append(result)

    if sha256_bytes(input_path.read_bytes()) != input_digest:
        raise OperationalTranslationError(
            "canonical input changed during translation; no batch manifest emitted"
        )
    ordered_results = sorted(results, key=lambda item: (item.episode_id, item.input_variant_id))
    manifest = publish_jsonl_artifact(
        root / "translation-results",
        logical_name="operational-translation-results",
        schema_version="operational-translation-result-0.1.0",
        stage="operational-live-translation",
        records=[item.model_dump(mode="json", exclude_none=False) for item in ordered_results],
        contract_hashes={
            "field_policy": sha256_jcs(field_policy),
            "prompt_bundle": sha256_jcs(prompt),
            "input_canonical_jsonl": input_digest,
        },
    )
    translated = sum(item.status == "translated" for item in ordered_results)
    research = sum(item.status == "research_needed" for item in ordered_results)
    no_segments = sum(item.status == "no_translatable_segments" for item in ordered_results)
    # Every leaf record is independently checkpointed; report only the attempts
    # that belong to this input, not unrelated immutable files in a reused root.
    provider_attempts = sum(len(item.leaf_result_ids) for item in ordered_results)
    body: dict[str, object] = {
        "schema_version": "operational-translation-0.1.0",
        "input_file_sha256": input_digest,
        "field_policy_sha256": sha256_jcs(field_policy),
        "prompt_id": prompt.prompt_id,
        "source_records": len(ordered_results),
        "translated_records": translated,
        "research_required_records": research,
        "no_translatable_records": no_segments,
        "provider_attempts": provider_attempts,
        "result_manifest_id": manifest.manifest_id,
        "promotion": "not_eligible",
    }
    report = OperationalTranslationReport(
        batch_id=stable_id("trbatch", body),
        input_file_sha256=input_digest,
        field_policy_sha256=sha256_jcs(field_policy),
        prompt_id=prompt.prompt_id,
        source_records=len(ordered_results),
        translated_records=translated,
        research_required_records=research,
        no_translatable_records=no_segments,
        provider_attempts=provider_attempts,
        result_manifest_id=manifest.manifest_id,
    )
    publish_bytes_atomic(
        root / "translation-batches" / f"{report.batch_id}.json", canonical_bytes(report) + b"\n"
    )
    return report
