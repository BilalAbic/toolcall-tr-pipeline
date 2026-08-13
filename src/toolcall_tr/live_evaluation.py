"""Bounded operational execution for explicit OpenAI evaluation JSONL.

The live provider adapter remains separate from translation.  This module only
accepts complete, content-addressed source/target pairs, persists hash-only
provider attempts, and writes immutable result artifacts.  Model output is
triage evidence; it cannot produce a Gold acceptance or a human-review event.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.artifacts import ContentManifest, publish_bytes_atomic, publish_jsonl_artifact
from toolcall_tr.config import PipelineConfig
from toolcall_tr.eval_contract import EvaluationUnit, ModelEvaluationVerdict, SegmentPathEvidence
from toolcall_tr.hashing import canonical_bytes, sha256_bytes, stable_id
from toolcall_tr.jsonio import iter_jsonl
from toolcall_tr.models import NonEmptyStr, Sha256, StrictModel
from toolcall_tr.openai_judge import OpenAIResponsesJudge, validate_openai_endpoint
from toolcall_tr.provider_adapter import ProviderConfigurationError
from toolcall_tr.provider_provenance import (
    ProviderAttemptRecord,
    ProviderAttemptSink,
    ProviderOperation,
)

LiveEvaluationInputId = Annotated[str, Field(pattern=r"^liveevalinput_[0-9a-f]{64}$")]
LiveEvaluationResultId = Annotated[str, Field(pattern=r"^liveevalresult_[0-9a-f]{64}$")]
LiveEvaluationRunId = Annotated[str, Field(pattern=r"^liveevalrun_[0-9a-f]{64}$")]
LiveJudgeRole = Literal["mini_verifier", "strong_judge"]


class LiveEvaluationConfigurationError(ValueError):
    """Raised before execution when an input, config, or output boundary is unsafe."""


class LiveEvaluationRuntimeError(RuntimeError):
    """Raised when an adapter fails to produce one auditable terminal attempt."""


class LiveEvaluationInput(StrictModel):
    """One strict, full-leaf source/target pair prepared for one judge role.

    ``SegmentPathEvidence`` names its texts "excerpts" for generic evaluation
    use.  The operational runner deliberately strengthens that contract: both
    values must be the *complete* leaf whose hashes are in ``evaluation_unit``.
    This prevents an operator from silently pairing evidence with a different
    immutable evaluation unit.
    """

    schema_version: Literal["live-evaluation-input-0.1.0"] = "live-evaluation-input-0.1.0"
    input_id: LiveEvaluationInputId
    evaluation_unit: EvaluationUnit
    evidence: SegmentPathEvidence

    @model_validator(mode="after")
    def validate_pair(self) -> LiveEvaluationInput:
        if (
            self.evidence.segment_id != self.evaluation_unit.segment_id
            or self.evidence.path != self.evaluation_unit.path
        ):
            raise ValueError("evaluation evidence must match the unit segment and path")
        if sha256_bytes(self.evidence.source_excerpt.encode("utf-8")) != (
            self.evaluation_unit.source_text_sha256
        ):
            raise ValueError("evaluation source evidence must match the unit content hash")
        if sha256_bytes(self.evidence.target_excerpt.encode("utf-8")) != (
            self.evaluation_unit.target_text_sha256
        ):
            raise ValueError("evaluation target evidence must match the unit content hash")
        body = self.model_dump(mode="json", exclude={"input_id"})
        if self.input_id != stable_id("liveevalinput", body):
            raise ValueError("live evaluation input ID does not match its content")
        return self


class LiveEvaluationResult(StrictModel):
    """One terminal model-triage result, with no Gold-acceptance capability."""

    schema_version: Literal["live-evaluation-result-0.1.0"] = "live-evaluation-result-0.1.0"
    result_id: LiveEvaluationResultId
    input_id: LiveEvaluationInputId
    evaluation_unit: EvaluationUnit
    attempt: ProviderAttemptRecord
    verdict: ModelEvaluationVerdict | None
    gold_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_terminal_result(self) -> LiveEvaluationResult:
        if self.attempt.operation is not ProviderOperation.JUDGE:
            raise ValueError("live evaluation result must retain a judge attempt")
        if self.attempt.outcome.value == "succeeded":
            if self.verdict is None:
                raise ValueError("successful judge attempt requires a local verdict")
        elif self.verdict is not None:
            raise ValueError("unsuccessful judge attempt cannot retain a model verdict")
        if self.verdict is not None and self.verdict.evaluation_unit != self.evaluation_unit:
            raise ValueError("model verdict must refer to the result evaluation unit")
        body = self.model_dump(mode="json", exclude={"result_id"})
        if self.result_id != stable_id("liveevalresult", body):
            raise ValueError("live evaluation result ID does not match its content")
        return self


class LiveEvaluationRunReport(StrictModel):
    """Content-addressed receipt for an immutable live evaluation output set."""

    schema_version: Literal["live-evaluation-run-0.1.0"] = "live-evaluation-run-0.1.0"
    run_id: NonEmptyStr
    report_id: LiveEvaluationRunId
    input_file_sha256: Sha256
    config_sha256: Sha256
    role: LiveJudgeRole
    input_rows: Annotated[int, Field(gt=0)]
    succeeded_rows: Annotated[int, Field(ge=0)]
    failed_rows: Annotated[int, Field(ge=0)]
    results_manifest_id: Annotated[str, Field(pattern=r"^manifest_[0-9a-f]{64}$")]
    attempts_manifest_id: Annotated[str, Field(pattern=r"^manifest_[0-9a-f]{64}$")]
    gold_release_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> LiveEvaluationRunReport:
        if self.input_rows != self.succeeded_rows + self.failed_rows:
            raise ValueError("live evaluation row accounting must balance")
        body = self.model_dump(mode="json", exclude={"report_id"})
        if self.report_id != stable_id("liveevalrun", body):
            raise ValueError("live evaluation run ID does not match its content")
        return self


@dataclass(frozen=True, slots=True)
class LiveEvaluationRunArtifacts:
    """Published immutable artifact references from one completed invocation."""

    report: LiveEvaluationRunReport
    results_manifest: ContentManifest
    attempts_manifest: ContentManifest


JudgeFactory = Callable[[ProviderAttemptSink], OpenAIResponsesJudge]


def build_live_evaluation_input(
    *, evaluation_unit: EvaluationUnit, evidence: SegmentPathEvidence
) -> LiveEvaluationInput:
    """Build the only accepted live-evaluation input shape from an exact pair."""
    body = {
        "schema_version": "live-evaluation-input-0.1.0",
        "evaluation_unit": evaluation_unit.model_dump(mode="json"),
        "evidence": evidence.model_dump(mode="json"),
    }
    return LiveEvaluationInput(
        input_id=stable_id("liveevalinput", body),
        evaluation_unit=evaluation_unit,
        evidence=evidence,
    )


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_boundaries(input_jsonl: Path, output_root: Path) -> tuple[Path, Path]:
    input_path = input_jsonl.resolve(strict=True)
    if not input_path.is_file() or input_path.suffix.lower() != ".jsonl":
        raise LiveEvaluationConfigurationError("evaluation input must be an existing .jsonl file")
    resolved_output = output_root.resolve(strict=False)
    if output_root.exists() and not output_root.is_dir():
        raise LiveEvaluationConfigurationError("evaluation output root must be a directory")
    if _is_within(resolved_output, input_path.parent) or _is_within(
        input_path.parent, resolved_output
    ):
        raise LiveEvaluationConfigurationError(
            "evaluation output root must be disjoint from the input source root"
        )
    return input_path, resolved_output


def _read_inputs(input_path: Path) -> list[LiveEvaluationInput]:
    inputs = [
        LiveEvaluationInput.model_validate(record, strict=True) for record in iter_jsonl(input_path)
    ]
    if not inputs:
        raise LiveEvaluationConfigurationError(
            "evaluation input JSONL must contain at least one row"
        )
    input_ids = [item.input_id for item in inputs]
    if len(input_ids) != len(set(input_ids)):
        raise LiveEvaluationConfigurationError(
            "evaluation input JSONL contains duplicate input IDs"
        )
    unit_ids = [item.evaluation_unit.unit_id for item in inputs]
    if len(unit_ids) != len(set(unit_ids)):
        raise LiveEvaluationConfigurationError(
            "evaluation input JSONL contains more than one row for an evaluation unit"
        )
    return inputs


def _validate_live_judge_config(config: PipelineConfig, role_name: str) -> LiveJudgeRole:
    if role_name == "mini_verifier":
        role: LiveJudgeRole = "mini_verifier"
    elif role_name == "strong_judge":
        role = "strong_judge"
    else:
        raise LiveEvaluationConfigurationError(
            "live evaluation role must be mini_verifier or strong_judge"
        )
    if not config.providers.enabled or not config.providers.network_egress_enabled:
        raise LiveEvaluationConfigurationError(
            "live evaluation requires both provider and network egress gates"
        )
    provider_role = getattr(config.providers, role)
    if provider_role.provider != "openai" or provider_role.endpoint is None:
        raise LiveEvaluationConfigurationError(
            "live evaluation requires an explicit OpenAI judge role"
        )
    try:
        validate_openai_endpoint(provider_role.endpoint)
    except ProviderConfigurationError as exc:
        raise LiveEvaluationConfigurationError(
            "live evaluation has an unapproved OpenAI endpoint"
        ) from exc
    if provider_role.model not in {"gpt-5.4", "gpt-5.4-mini"}:
        raise LiveEvaluationConfigurationError(
            "live evaluation has an unapproved OpenAI judge model"
        )
    return role


def _build_result(
    *,
    item: LiveEvaluationInput,
    attempt: ProviderAttemptRecord,
    verdict: ModelEvaluationVerdict | None,
) -> LiveEvaluationResult:
    body = {
        "schema_version": "live-evaluation-result-0.1.0",
        "input_id": item.input_id,
        "evaluation_unit": item.evaluation_unit.model_dump(mode="json"),
        "attempt": attempt.model_dump(mode="json"),
        "verdict": verdict.model_dump(mode="json") if verdict is not None else None,
        "gold_eligible": False,
    }
    return LiveEvaluationResult(
        result_id=stable_id("liveevalresult", body),
        input_id=item.input_id,
        evaluation_unit=item.evaluation_unit,
        attempt=attempt,
        verdict=verdict,
    )


def _validate_results_artifact(path: Path) -> None:
    records = [
        LiveEvaluationResult.model_validate_json(canonical_bytes(record), strict=True)
        for record in iter_jsonl(path)
    ]
    result_ids = [record.result_id for record in records]
    if result_ids != sorted(result_ids) or len(result_ids) != len(set(result_ids)):
        raise LiveEvaluationRuntimeError("evaluation result artifact must be unique and sorted")


def _validate_attempts_artifact(path: Path) -> None:
    records = [
        ProviderAttemptRecord.model_validate_json(canonical_bytes(record), strict=True)
        for record in iter_jsonl(path)
    ]
    attempt_ids = [record.attempt_id for record in records]
    if attempt_ids != sorted(attempt_ids) or len(attempt_ids) != len(set(attempt_ids)):
        raise LiveEvaluationRuntimeError("evaluation attempt artifact must be unique and sorted")


def run_live_evaluation(
    input_jsonl: Path,
    output_root: Path,
    *,
    config: PipelineConfig,
    role_name: str,
    run_id: str,
    judge_factory: JudgeFactory,
) -> LiveEvaluationRunArtifacts:
    """Judge every strict input row once and publish only immutable receipts.

    A provider failure is represented by its hash-only terminal attempt and does
    not stop unrelated rows.  A malformed input/configuration or missing
    attempt receipt fails before any output is published.  The input file is
    rehashed immediately before publication so a changed source cannot be
    represented by a completed evaluation run.
    """
    if not run_id.strip():
        raise LiveEvaluationConfigurationError("live evaluation run_id must be non-empty")
    role = _validate_live_judge_config(config, role_name)
    input_path, resolved_output = _validate_boundaries(input_jsonl, output_root)
    input_bytes = input_path.read_bytes()
    input_sha256 = sha256_bytes(input_bytes)
    inputs = _read_inputs(input_path)

    results: list[LiveEvaluationResult] = []
    attempts: list[ProviderAttemptRecord] = []
    for item in inputs:
        row_attempts: list[ProviderAttemptRecord] = []
        judge = judge_factory(row_attempts.append)
        verdict: ModelEvaluationVerdict | None = None
        try:
            verdict = judge.judge(
                evaluation_unit=item.evaluation_unit,
                evidence=item.evidence,
            )
        except Exception:
            # The adapter emits a redaction-safe terminal ProviderAttemptRecord
            # for every transport/preflight/local-contract error.
            verdict = None
        if len(row_attempts) != 1:
            raise LiveEvaluationRuntimeError(
                "live judge did not emit exactly one terminal attempt record"
            )
        attempt = row_attempts[0]
        results.append(_build_result(item=item, attempt=attempt, verdict=verdict))
        attempts.append(attempt)

    result_ids = [result.result_id for result in results]
    if len(result_ids) != len(set(result_ids)):
        raise LiveEvaluationRuntimeError("live judge yielded duplicate result records")
    attempt_ids = [attempt.attempt_id for attempt in attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise LiveEvaluationRuntimeError("live judge yielded duplicate attempt records")
    if sha256_bytes(input_path.read_bytes()) != input_sha256:
        raise LiveEvaluationRuntimeError("evaluation input JSONL changed during execution")

    ordered_results = sorted(results, key=lambda item: item.result_id)
    ordered_attempts = sorted(attempts, key=lambda item: item.attempt_id)
    config_sha256 = sha256_bytes(canonical_bytes(config.model_dump(mode="json")))
    attempts_manifest = publish_jsonl_artifact(
        resolved_output / "attempts",
        logical_name="provider-attempts",
        schema_version="provider-attempt-0.1.0",
        stage="live-evaluation-attempts",
        records=[item.model_dump(mode="json") for item in ordered_attempts],
        contract_hashes={"pipeline_config": config_sha256},
        validator=_validate_attempts_artifact,
    )
    results_manifest = publish_jsonl_artifact(
        resolved_output / "results",
        logical_name="live-evaluation-results",
        schema_version="live-evaluation-result-0.1.0",
        stage="live-evaluation-results",
        records=[item.model_dump(mode="json") for item in ordered_results],
        input_manifest_ids=[attempts_manifest.manifest_id],
        contract_hashes={"pipeline_config": config_sha256},
        validator=_validate_results_artifact,
    )
    succeeded_rows = sum(item.verdict is not None for item in ordered_results)
    body = {
        "schema_version": "live-evaluation-run-0.1.0",
        "run_id": run_id,
        "input_file_sha256": input_sha256,
        "config_sha256": config_sha256,
        "role": role,
        "input_rows": len(inputs),
        "succeeded_rows": succeeded_rows,
        "failed_rows": len(inputs) - succeeded_rows,
        "results_manifest_id": results_manifest.manifest_id,
        "attempts_manifest_id": attempts_manifest.manifest_id,
        "gold_release_allowed": False,
    }
    report = LiveEvaluationRunReport(
        report_id=stable_id("liveevalrun", body),
        run_id=run_id,
        input_file_sha256=input_sha256,
        config_sha256=config_sha256,
        role=role,
        input_rows=len(inputs),
        succeeded_rows=succeeded_rows,
        failed_rows=len(inputs) - succeeded_rows,
        results_manifest_id=results_manifest.manifest_id,
        attempts_manifest_id=attempts_manifest.manifest_id,
    )
    publish_bytes_atomic(
        resolved_output / "runs" / f"{report.report_id}.json",
        canonical_bytes(report) + b"\n",
    )
    return LiveEvaluationRunArtifacts(
        report=report,
        results_manifest=results_manifest,
        attempts_manifest=attempts_manifest,
    )
