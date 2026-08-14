"""Explicit, append-only recovery for a completed automation run.

The normal automation runner never retries a terminal provider result in place:
the original attempt might have reached the provider and a blind resend could
create a duplicate charge.  This module provides the deliberately separate
operator path for a known billing/quota recovery.  It reads a completed run,
retries only explicitly authorised HTTP statuses in a *new sibling root*, and
publishes an effective overlay.  The parent run is never modified.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from toolcall_tr.artifacts import ContentManifest, publish_bytes_atomic, publish_jsonl_artifact
from toolcall_tr.autonomous_pipeline import (
    AutomationCandidateManifest,
    AutomationEpisodeResult,
    AutomationTranslationReport,
    build_automation_evaluation_inputs,
    build_hierarchical_consensus,
    build_huggingface_review_package,
    prepare_strong_escalation_inputs,
    read_automation_results,
    run_automation_translation,
)
from toolcall_tr.config import PipelineConfig
from toolcall_tr.field_policy import FieldPolicy
from toolcall_tr.hashing import canonical_bytes, sha256_bytes, stable_id
from toolcall_tr.jsonio import iter_jsonl
from toolcall_tr.live_evaluation import (
    JudgeFactory,
    LiveEvaluationInput,
    LiveEvaluationResult,
    LiveEvaluationRunReport,
    run_live_evaluation,
)
from toolcall_tr.models import CanonicalEpisode, Sha256, StrictModel
from toolcall_tr.prompt_contract import PromptBundle, require_validated_prompt
from toolcall_tr.provider_adapter import ResponsesTransport
from toolcall_tr.provider_provenance import ProviderAttemptOutcome, ProviderAttemptRecord

AutomationRecoveryId = Annotated[str, Field(pattern=r"^autorecovery_[0-9a-f]{64}$")]
_SUPPORTED_RETRY_HTTP_STATUSES = frozenset({402, 429})


class AutomationRecoveryError(RuntimeError):
    """Raised when an operator recovery would mutate or ambiguously replace evidence."""


class AutomationRecoveryPlan(StrictModel):
    """Read-only paid-retry preview; it cannot authorize provider egress."""

    schema_version: Literal["automation-recovery-plan-0.1.0"] = "automation-recovery-plan-0.1.0"
    plan_id: Annotated[str, Field(pattern=r"^autorecoveryplan_[0-9a-f]{64}$")]
    parent_candidate_id: Annotated[str, Field(pattern=r"^autocand_[0-9a-f]{64}$")]
    retry_http_statuses: Annotated[list[int], Field(min_length=1)]
    parent_candidate_sha256: Sha256
    parent_translation_sha256: Sha256
    parent_mini_results_sha256: Sha256
    parent_strong_results_sha256: Sha256 | None
    translation_retry_episodes: Annotated[int, Field(ge=0)]
    mini_retry_units: Annotated[int, Field(ge=0)]
    strong_retry_units: Annotated[int, Field(ge=0)]
    provider_egress_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_identity(self) -> AutomationRecoveryPlan:
        if self.retry_http_statuses != sorted(set(self.retry_http_statuses)):
            raise ValueError("retry HTTP statuses must be unique and sorted")
        if not set(self.retry_http_statuses).issubset(_SUPPORTED_RETRY_HTTP_STATUSES):
            raise ValueError("recovery permits only explicit payment/quota HTTP statuses")
        body = self.model_dump(mode="json", exclude={"plan_id"})
        if self.plan_id != stable_id("autorecoveryplan", body):
            raise ValueError("recovery plan ID does not match deterministic content")
        return self


class AutomationRecoveryReport(StrictModel):
    """Immutable, non-promoting receipt for a paid retry overlay."""

    schema_version: Literal["automation-recovery-0.1.0"] = "automation-recovery-0.1.0"
    recovery_id: AutomationRecoveryId
    parent_candidate_id: Annotated[str, Field(pattern=r"^autocand_[0-9a-f]{64}$")]
    retry_http_statuses: Annotated[list[int], Field(min_length=1)]
    parent_candidate_sha256: Sha256
    parent_translation_sha256: Sha256
    parent_mini_results_sha256: Sha256
    parent_strong_results_sha256: Sha256 | None
    retried_translation_episodes: Annotated[int, Field(ge=0)]
    recovered_translation_episodes: Annotated[int, Field(ge=0)]
    retried_mini_units: Annotated[int, Field(ge=0)]
    recovered_mini_units: Annotated[int, Field(ge=0)]
    retried_strong_units: Annotated[int, Field(ge=0)]
    recovered_strong_units: Annotated[int, Field(ge=0)]
    effective_translation_manifest_id: Annotated[str, Field(pattern=r"^manifest_[0-9a-f]{64}$")]
    effective_mini_manifest_id: Annotated[str, Field(pattern=r"^manifest_[0-9a-f]{64}$")]
    effective_strong_manifest_id: Annotated[str, Field(pattern=r"^manifest_[0-9a-f]{64}$")] | None
    consensus_report_id: Annotated[str, Field(pattern=r"^autohconsreport_[0-9a-f]{64}$")]
    hf_package_id: Annotated[str, Field(pattern=r"^hfpackage_[0-9a-f]{64}$")]
    promotion: Literal["not_eligible"] = "not_eligible"
    publish_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_identity(self) -> AutomationRecoveryReport:
        if self.retry_http_statuses != sorted(set(self.retry_http_statuses)):
            raise ValueError("retry HTTP statuses must be unique and sorted")
        if not set(self.retry_http_statuses).issubset(_SUPPORTED_RETRY_HTTP_STATUSES):
            raise ValueError("recovery permits only explicit payment/quota HTTP statuses")
        body = self.model_dump(mode="json", exclude={"recovery_id"})
        if self.recovery_id != stable_id("autorecovery", body):
            raise ValueError("recovery ID does not match deterministic content")
        return self


class _ParentRun:
    """Validated parent paths; every JSONL is content-addressed by its manifest."""

    def __init__(
        self,
        *,
        root: Path,
        candidate: AutomationCandidateManifest,
        candidate_jsonl: Path,
        translation_jsonl: Path,
        evaluation_inputs_jsonl: Path,
        mini_results_jsonl: Path,
        strong_results_jsonl: Path | None,
    ) -> None:
        self.root = root
        self.candidate = candidate
        self.candidate_jsonl = candidate_jsonl
        self.translation_jsonl = translation_jsonl
        self.evaluation_inputs_jsonl = evaluation_inputs_jsonl
        self.mini_results_jsonl = mini_results_jsonl
        self.strong_results_jsonl = strong_results_jsonl


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _read_exact_model[ModelT: StrictModel](directory: Path, model: type[ModelT]) -> ModelT:
    paths = sorted(directory.glob("*.json"))
    if len(paths) != 1:
        raise AutomationRecoveryError(
            f"parent recovery evidence is ambiguous or missing: {directory}"
        )
    try:
        return model.model_validate_json(paths[0].read_text(encoding="utf-8"), strict=True)
    except (OSError, ValidationError, ValueError) as exc:
        raise AutomationRecoveryError("parent recovery evidence is invalid") from exc


def _manifest_jsonl(root: Path, manifest_id: str) -> Path:
    descriptor = root / f"{manifest_id}.json"
    try:
        manifest = ContentManifest.model_validate_json(
            descriptor.read_text(encoding="utf-8"), strict=True
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise AutomationRecoveryError("parent content manifest is missing or invalid") from exc
    if len(manifest.artifacts) != 1:
        raise AutomationRecoveryError("recovery requires exactly one JSONL artifact per stage")
    path = root / manifest.artifacts[0].relative_path
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        raise AutomationRecoveryError("parent content manifest does not resolve to JSONL")
    return path


def _parent_run(parent_root: Path) -> _ParentRun:
    root = parent_root.resolve(strict=True)
    if not root.is_dir():
        raise AutomationRecoveryError("parent automation root must be a directory")
    candidate = _read_exact_model(root / "candidate" / "candidates", AutomationCandidateManifest)
    translation = _read_exact_model(
        root / "translation" / "batches", AutomationTranslationReport
    )
    mini = _read_exact_model(root / "mini-judge" / "runs", LiveEvaluationRunReport)
    strong_runs = sorted((root / "strong-judge" / "runs").glob("*.json"))
    strong: LiveEvaluationRunReport | None
    if not strong_runs:
        strong = None
    elif len(strong_runs) == 1:
        try:
            strong = LiveEvaluationRunReport.model_validate_json(
                strong_runs[0].read_text(encoding="utf-8"), strict=True
            )
        except (OSError, ValidationError, ValueError) as exc:
            raise AutomationRecoveryError("parent strong-judge receipt is invalid") from exc
    else:
        raise AutomationRecoveryError("parent strong-judge receipts are ambiguous")
    input_candidates = sorted((root / "evaluation-inputs").glob("*.jsonl"))
    if len(input_candidates) != 1:
        raise AutomationRecoveryError("parent evaluation inputs are ambiguous or missing")
    return _ParentRun(
        root=root,
        candidate=candidate,
        candidate_jsonl=_manifest_jsonl(
            root / "candidate" / "canonical", candidate.canonical_manifest_id
        ),
        translation_jsonl=_manifest_jsonl(
            root / "translation" / "translation-results", translation.result_manifest_id
        ),
        evaluation_inputs_jsonl=input_candidates[0],
        mini_results_jsonl=_manifest_jsonl(
            root / "mini-judge" / "results", mini.results_manifest_id
        ),
        strong_results_jsonl=(
            _manifest_jsonl(root / "strong-judge" / "results", strong.results_manifest_id)
            if strong is not None
            else None
        ),
    )


def _read_episodes(path: Path) -> list[CanonicalEpisode]:
    return [
        CanonicalEpisode.model_validate_json(canonical_bytes(row), strict=True)
        for row in iter_jsonl(path)
    ]


def _read_inputs(path: Path) -> list[LiveEvaluationInput]:
    return [
        LiveEvaluationInput.model_validate_json(canonical_bytes(row), strict=True)
        for row in iter_jsonl(path)
    ]


def _read_results(path: Path) -> list[LiveEvaluationResult]:
    results = [
        LiveEvaluationResult.model_validate_json(canonical_bytes(row), strict=True)
        for row in iter_jsonl(path)
    ]
    unit_ids = [item.evaluation_unit.unit_id for item in results]
    if len(unit_ids) != len(set(unit_ids)):
        raise AutomationRecoveryError("judge result evidence has duplicate evaluation units")
    return results


def _route_attempts(
    parent: _ParentRun, episode_id: str, route: str
) -> list[ProviderAttemptRecord]:
    paths = sorted(
        (parent.root / "translation" / "routes" / episode_id / route / "provider-attempts").glob(
            "*.json"
        )
    )
    try:
        return [
            ProviderAttemptRecord.model_validate_json(path.read_text(encoding="utf-8"), strict=True)
            for path in paths
        ]
    except (OSError, ValidationError, ValueError) as exc:
        raise AutomationRecoveryError("translation provider attempt evidence is invalid") from exc


def _retryable_attempt(attempt: ProviderAttemptRecord, statuses: frozenset[int]) -> bool:
    return (
        attempt.outcome is ProviderAttemptOutcome.FAILED
        and attempt.http_status is not None
        and attempt.http_status in statuses
    )


def _retryable_translation_ids(
    parent: _ParentRun,
    results: Iterable[AutomationEpisodeResult],
    statuses: frozenset[int],
) -> set[str]:
    selected: set[str] = set()
    for result in results:
        if result.status != "needs_review":
            continue
        terminal = result.routes[-1]
        if terminal.status != "failed":
            continue
        attempts = _route_attempts(parent, result.episode_id, terminal.route)
        if any(_retryable_attempt(attempt, statuses) for attempt in attempts):
            selected.add(result.episode_id)
    return selected


def _artifact_jsonl(
    root: Path,
    *,
    logical_name: str,
    schema_version: str,
    stage: str,
    records: Iterable[StrictModel],
    contract_hashes: dict[str, Sha256],
) -> tuple[ContentManifest, Path]:
    manifest = publish_jsonl_artifact(
        root,
        logical_name=logical_name,
        schema_version=schema_version,
        stage=stage,
        records=[record.model_dump(mode="json", exclude_none=False) for record in records],
        contract_hashes=contract_hashes,
    )
    if len(manifest.artifacts) != 1:
        raise AutomationRecoveryError("recovery stage did not publish exactly one artifact")
    return manifest, root / manifest.artifacts[0].relative_path


def _combined_by_unit(
    parent_results: Iterable[LiveEvaluationResult],
    recovered: Iterable[LiveEvaluationResult],
    *,
    allowed_parent_replacements: set[str],
    additional_unit_ids: set[str],
) -> list[LiveEvaluationResult]:
    parent_list = list(parent_results)
    combined = {item.evaluation_unit.unit_id: item for item in parent_list}
    if len(combined) != len(parent_list):
        raise AutomationRecoveryError("parent judge results contain duplicate units")
    for item in recovered:
        unit_id = item.evaluation_unit.unit_id
        if unit_id in combined and unit_id not in allowed_parent_replacements:
            raise AutomationRecoveryError(
                "recovery attempted to replace a non-selected judge result"
            )
        if unit_id not in combined and unit_id not in additional_unit_ids:
            raise AutomationRecoveryError("recovery judge result is not tied to selected evidence")
        combined[unit_id] = item
    return [combined[unit_id] for unit_id in sorted(combined)]


def _exact_result_path(root: Path, report: LiveEvaluationRunReport) -> Path:
    return _manifest_jsonl(root / "results", report.results_manifest_id)


def _effective_translation(
    parent_results: list[AutomationEpisodeResult],
    recovered_results: list[AutomationEpisodeResult],
    retry_ids: set[str],
) -> list[AutomationEpisodeResult]:
    combined = {item.episode_id: item for item in parent_results}
    if len(combined) != len(parent_results):
        raise AutomationRecoveryError("parent translation results contain duplicate episodes")
    for item in recovered_results:
        if item.episode_id not in retry_ids:
            raise AutomationRecoveryError(
                "recovery translation result is not in the selected retry set"
            )
        if item.status == "translated":
            combined[item.episode_id] = item
    return [combined[episode_id] for episode_id in sorted(combined)]


def _recovery_id(
    *,
    parent: _ParentRun,
    statuses: frozenset[int],
) -> str:
    body = {
        "candidate_id": parent.candidate.candidate_id,
        "candidate_sha256": sha256_bytes(parent.candidate_jsonl.read_bytes()),
        "translation_sha256": sha256_bytes(parent.translation_jsonl.read_bytes()),
        "mini_sha256": sha256_bytes(parent.mini_results_jsonl.read_bytes()),
        "strong_sha256": (
            sha256_bytes(parent.strong_results_jsonl.read_bytes())
            if parent.strong_results_jsonl is not None
            else None
        ),
        "retry_http_statuses": sorted(statuses),
    }
    return stable_id("autorecovery", body)


def inspect_automation_recovery(
    parent_root: Path,
    *,
    retry_http_statuses: Iterable[int],
) -> AutomationRecoveryPlan:
    """Preview the exact currently retryable billing/quota failures without I/O writes."""
    statuses = frozenset(retry_http_statuses)
    if not statuses or not statuses.issubset(_SUPPORTED_RETRY_HTTP_STATUSES):
        raise AutomationRecoveryError(
            "recovery requires one or more explicit supported HTTP statuses: 402, 429"
        )
    parent = _parent_run(parent_root)
    parent_translation = read_automation_results(parent.translation_jsonl)
    translation_retry_ids = _retryable_translation_ids(parent, parent_translation, statuses)
    parent_inputs = _read_inputs(parent.evaluation_inputs_jsonl)
    parent_mini = _read_results(parent.mini_results_jsonl)
    mini_by_input_id = {item.input_id: item for item in parent_mini}
    if set(mini_by_input_id) != {item.input_id for item in parent_inputs}:
        raise AutomationRecoveryError("parent mini results do not cover parent evaluation inputs")
    mini_retry_units = sum(
        _retryable_attempt(mini_by_input_id[item.input_id].attempt, statuses)
        for item in parent_inputs
    )
    strong_retry_units = (
        sum(
            _retryable_attempt(item.attempt, statuses)
            for item in _read_results(parent.strong_results_jsonl)
        )
        if parent.strong_results_jsonl is not None
        else 0
    )
    body = {
        "schema_version": "automation-recovery-plan-0.1.0",
        "parent_candidate_id": parent.candidate.candidate_id,
        "retry_http_statuses": sorted(statuses),
        "parent_candidate_sha256": sha256_bytes(parent.candidate_jsonl.read_bytes()),
        "parent_translation_sha256": sha256_bytes(parent.translation_jsonl.read_bytes()),
        "parent_mini_results_sha256": sha256_bytes(parent.mini_results_jsonl.read_bytes()),
        "parent_strong_results_sha256": (
            sha256_bytes(parent.strong_results_jsonl.read_bytes())
            if parent.strong_results_jsonl is not None
            else None
        ),
        "translation_retry_episodes": len(translation_retry_ids),
        "mini_retry_units": mini_retry_units,
        "strong_retry_units": strong_retry_units,
        "provider_egress_allowed": False,
    }
    return AutomationRecoveryPlan(
        plan_id=stable_id("autorecoveryplan", body),
        parent_candidate_id=parent.candidate.candidate_id,
        retry_http_statuses=sorted(statuses),
        parent_candidate_sha256=sha256_bytes(parent.candidate_jsonl.read_bytes()),
        parent_translation_sha256=sha256_bytes(parent.translation_jsonl.read_bytes()),
        parent_mini_results_sha256=sha256_bytes(parent.mini_results_jsonl.read_bytes()),
        parent_strong_results_sha256=(
            sha256_bytes(parent.strong_results_jsonl.read_bytes())
            if parent.strong_results_jsonl is not None
            else None
        ),
        translation_retry_episodes=len(translation_retry_ids),
        mini_retry_units=mini_retry_units,
        strong_retry_units=strong_retry_units,
    )


def run_automation_recovery(
    parent_root: Path,
    output_root: Path,
    *,
    config: PipelineConfig,
    field_policy: FieldPolicy,
    prompt: PromptBundle,
    translation_transport: ResponsesTransport,
    mini_judge_factory: JudgeFactory,
    strong_judge_factory: JudgeFactory,
    retry_http_statuses: Iterable[int],
    translation_workers: int = 1,
    mini_workers: int = 1,
    strong_workers: int = 1,
    strong_pass_sample_basis_points: int = 200,
) -> AutomationRecoveryReport:
    """Retry only explicit payment/quota failures and publish a new effective overlay.

    The caller must create a new, currently absent ``output_root`` and pass an
    explicit approved HTTP status allowlist (normally 402 and/or 429).  Existing
    source, attempt, result, consensus, and HF artifacts remain byte-for-byte
    unchanged in ``parent_root``.
    """
    statuses = frozenset(retry_http_statuses)
    if not statuses or not statuses.issubset(_SUPPORTED_RETRY_HTTP_STATUSES):
        raise AutomationRecoveryError(
            "recovery requires one or more explicit supported HTTP statuses: 402, 429"
        )
    if not all(
        1 <= workers <= 16 for workers in (translation_workers, mini_workers, strong_workers)
    ):
        raise AutomationRecoveryError("recovery worker limits must be between 1 and 16")
    try:
        require_validated_prompt(prompt)
    except Exception as exc:
        raise AutomationRecoveryError("recovery prompt contract is not validated") from exc
    if not config.providers.enabled or not config.providers.network_egress_enabled:
        raise AutomationRecoveryError("recovery requires both live provider gates")
    parent = _parent_run(parent_root)
    root = output_root.resolve(strict=False)
    if output_root.exists():
        raise AutomationRecoveryError("recovery output root must be new and absent")
    if _is_within(root, parent.root) or _is_within(parent.root, root):
        raise AutomationRecoveryError("recovery output must be a disjoint sibling root")

    episodes = _read_episodes(parent.candidate_jsonl)
    episodes_by_id = {episode.episode_id: episode for episode in episodes}
    if len(episodes_by_id) != len(episodes):
        raise AutomationRecoveryError("parent candidate contains duplicate episodes")
    parent_translation = read_automation_results(parent.translation_jsonl)
    translation_retry_ids = _retryable_translation_ids(parent, parent_translation, statuses)
    parent_inputs = _read_inputs(parent.evaluation_inputs_jsonl)
    parent_mini = _read_results(parent.mini_results_jsonl)
    mini_by_input_id = {item.input_id: item for item in parent_mini}
    if set(mini_by_input_id) != {item.input_id for item in parent_inputs}:
        raise AutomationRecoveryError("parent mini results do not cover parent evaluation inputs")
    mini_retry_inputs = [
        item
        for item in parent_inputs
        if _retryable_attempt(mini_by_input_id[item.input_id].attempt, statuses)
    ]
    parent_strong = (
        _read_results(parent.strong_results_jsonl)
        if parent.strong_results_jsonl is not None
        else []
    )

    if not translation_retry_ids and not mini_retry_inputs and not any(
        _retryable_attempt(item.attempt, statuses) for item in parent_strong
    ):
        raise AutomationRecoveryError("parent run has no selected payment/quota retry candidates")

    recovery_id = _recovery_id(parent=parent, statuses=statuses)
    recovered_translation: list[AutomationEpisodeResult] = []
    if translation_retry_ids:
        retry_episodes = [
            episodes_by_id[episode_id] for episode_id in sorted(translation_retry_ids)
        ]
        _, retry_candidate_jsonl = _artifact_jsonl(
            root / "translation-retry-inputs",
            logical_name="automation-recovery-translation-inputs",
            schema_version="0.1.0",
            stage="automation-recovery-translation-inputs",
            records=retry_episodes,
            contract_hashes={"parent_candidate": sha256_bytes(parent.candidate_jsonl.read_bytes())},
        )
        retry_translation = run_automation_translation(
            retry_candidate_jsonl,
            root / "translation-retry",
            config=config,
            field_policy=field_policy,
            prompt=prompt,
            transport=translation_transport,
            max_workers=translation_workers,
        )
        recovered_translation = read_automation_results(
            _manifest_jsonl(
                root / "translation-retry" / "translation-results",
                retry_translation.result_manifest_id,
            )
        )

    effective_translation = _effective_translation(
        parent_translation, recovered_translation, translation_retry_ids
    )
    effective_translation_manifest, effective_translation_jsonl = _artifact_jsonl(
        root / "effective" / "translation-results",
        logical_name="automation-recovery-effective-translation-results",
        schema_version="autonomous-translation-result-0.1.0",
        stage="automation-recovery-effective-translation",
        records=effective_translation,
        contract_hashes={
            "parent_translation": sha256_bytes(parent.translation_jsonl.read_bytes()),
            "recovery_translation": sha256_bytes(
                canonical_bytes([item.model_dump(mode="json") for item in recovered_translation])
            ),
        },
    )

    recovered_translation_ids = {
        item.episode_id for item in recovered_translation if item.status == "translated"
    }
    recovery_input_episodes = [episodes_by_id[item] for item in sorted(recovered_translation_ids)]
    recovery_input_results = [
        item for item in recovered_translation if item.episode_id in recovered_translation_ids
    ]
    newly_translated_inputs = build_automation_evaluation_inputs(
        recovery_input_episodes,
        recovery_input_results,
        field_policy=field_policy,
    ) if recovery_input_episodes else []
    mini_recovery_by_id = {item.input_id: item for item in mini_retry_inputs}
    for item in newly_translated_inputs:
        if item.input_id in mini_recovery_by_id:
            raise AutomationRecoveryError("newly translated input collides with parent evidence")
        mini_recovery_by_id[item.input_id] = item
    mini_retry_payload = [mini_recovery_by_id[input_id] for input_id in sorted(mini_recovery_by_id)]
    recovered_mini: list[LiveEvaluationResult] = []
    if mini_retry_payload:
        _, mini_retry_jsonl = _artifact_jsonl(
            root / "mini-retry-inputs",
            logical_name="automation-recovery-mini-inputs",
            schema_version="live-evaluation-input-0.1.0",
            stage="automation-recovery-mini-inputs",
            records=mini_retry_payload,
            contract_hashes={
                "parent_inputs": sha256_bytes(parent.evaluation_inputs_jsonl.read_bytes()),
                "parent_mini": sha256_bytes(parent.mini_results_jsonl.read_bytes()),
            },
        )
        mini_run = run_live_evaluation(
            mini_retry_jsonl,
            root / "mini-retry",
            config=config,
            role_name="mini_verifier",
            run_id=f"{recovery_id}-mini",
            judge_factory=mini_judge_factory,
            max_workers=mini_workers,
        )
        recovered_mini = _read_results(_exact_result_path(root / "mini-retry", mini_run.report))

    original_mini_retry_units = {
        item.evaluation_unit.unit_id for item in mini_retry_inputs
    }
    new_mini_units = {item.evaluation_unit.unit_id for item in newly_translated_inputs}
    effective_mini = _combined_by_unit(
        parent_mini,
        recovered_mini,
        allowed_parent_replacements=original_mini_retry_units,
        additional_unit_ids=new_mini_units,
    )
    all_inputs_by_id = {item.input_id: item for item in parent_inputs}
    for item in newly_translated_inputs:
        if item.input_id in all_inputs_by_id:
            raise AutomationRecoveryError("effective evaluation inputs contain duplicate input IDs")
        all_inputs_by_id[item.input_id] = item
    all_inputs = [all_inputs_by_id[input_id] for input_id in sorted(all_inputs_by_id)]
    _, effective_inputs_jsonl = _artifact_jsonl(
        root / "effective" / "evaluation-inputs",
        logical_name="automation-recovery-effective-evaluation-inputs",
        schema_version="live-evaluation-input-0.1.0",
        stage="automation-recovery-effective-evaluation-inputs",
        records=all_inputs,
        contract_hashes={
            "parent_inputs": sha256_bytes(parent.evaluation_inputs_jsonl.read_bytes()),
            "new_translation_inputs": sha256_bytes(
                canonical_bytes([item.model_dump(mode="json") for item in newly_translated_inputs])
            ),
        },
    )
    effective_mini_manifest, effective_mini_jsonl = _artifact_jsonl(
        root / "effective" / "mini-results",
        logical_name="automation-recovery-effective-mini-results",
        schema_version="live-evaluation-result-0.1.0",
        stage="automation-recovery-effective-mini-results",
        records=effective_mini,
        contract_hashes={
            "parent_mini": sha256_bytes(parent.mini_results_jsonl.read_bytes()),
            "recovery_mini": sha256_bytes(
                canonical_bytes([item.model_dump(mode="json") for item in recovered_mini])
            ),
        },
    )

    escalation = prepare_strong_escalation_inputs(
        effective_inputs_jsonl,
        effective_mini_jsonl,
        root / "strong-selection",
        pass_sample_basis_points=strong_pass_sample_basis_points,
    )
    effective_strong: list[LiveEvaluationResult] = []
    effective_strong_manifest: ContentManifest | None = None
    if escalation is not None:
        escalation_jsonl = root / "strong-selection" / escalation.artifacts[0].relative_path
        selected_inputs = _read_inputs(escalation_jsonl)
        parent_strong_by_unit = {item.evaluation_unit.unit_id: item for item in parent_strong}
        retry_strong_inputs: list[LiveEvaluationInput] = []
        for item in selected_inputs:
            existing = parent_strong_by_unit.get(item.evaluation_unit.unit_id)
            if existing is None or _retryable_attempt(existing.attempt, statuses):
                retry_strong_inputs.append(item)
        recovered_strong: list[LiveEvaluationResult] = []
        if retry_strong_inputs:
            _, strong_retry_jsonl = _artifact_jsonl(
                root / "strong-retry-inputs",
                logical_name="automation-recovery-strong-inputs",
                schema_version="live-evaluation-input-0.1.0",
                stage="automation-recovery-strong-inputs",
                records=retry_strong_inputs,
                contract_hashes={
                    "effective_inputs": sha256_bytes(effective_inputs_jsonl.read_bytes()),
                    "effective_mini": sha256_bytes(effective_mini_jsonl.read_bytes()),
                },
            )
            strong_run = run_live_evaluation(
                strong_retry_jsonl,
                root / "strong-retry",
                config=config,
                role_name="strong_judge",
                run_id=f"{recovery_id}-strong",
                judge_factory=strong_judge_factory,
                max_workers=strong_workers,
            )
            recovered_strong = _read_results(
                _exact_result_path(root / "strong-retry", strong_run.report)
            )
        replacement_units = {
            item.evaluation_unit.unit_id
            for item in selected_inputs
            if item.evaluation_unit.unit_id in parent_strong_by_unit
            and _retryable_attempt(
                parent_strong_by_unit[item.evaluation_unit.unit_id].attempt, statuses
            )
        }
        additional_strong_units = {
            item.evaluation_unit.unit_id
            for item in selected_inputs
            if item.evaluation_unit.unit_id not in parent_strong_by_unit
        }
        selected_parent_strong = [
            parent_strong_by_unit[item.evaluation_unit.unit_id]
            for item in selected_inputs
            if item.evaluation_unit.unit_id in parent_strong_by_unit
        ]
        effective_strong = _combined_by_unit(
            selected_parent_strong,
            recovered_strong,
            allowed_parent_replacements=replacement_units,
            additional_unit_ids=additional_strong_units,
        )
        if {item.evaluation_unit.unit_id for item in effective_strong} != {
            item.evaluation_unit.unit_id for item in selected_inputs
        }:
            raise AutomationRecoveryError(
                "effective strong results do not cover selected escalation"
            )
        effective_strong_manifest, effective_strong_jsonl = _artifact_jsonl(
            root / "effective" / "strong-results",
            logical_name="automation-recovery-effective-strong-results",
            schema_version="live-evaluation-result-0.1.0",
            stage="automation-recovery-effective-strong-results",
            records=effective_strong,
            contract_hashes={
                "parent_strong": sha256_bytes(parent.strong_results_jsonl.read_bytes())
                if parent.strong_results_jsonl is not None
                else sha256_bytes(b"no-parent-strong-results"),
                "recovery_strong": sha256_bytes(
                    canonical_bytes([item.model_dump(mode="json") for item in recovered_strong])
                ),
            },
        )
    else:
        effective_strong_jsonl = None
        recovered_strong = []
        retry_strong_inputs = []

    consensus = build_hierarchical_consensus(
        effective_mini_jsonl,
        effective_strong_jsonl,
        root / "consensus",
        pass_sample_basis_points=strong_pass_sample_basis_points,
    )
    consensus_jsonl = _manifest_jsonl(
        root / "consensus" / "consensus", consensus.consensus_manifest_id
    )
    package = build_huggingface_review_package(
        parent.candidate_jsonl,
        effective_translation_jsonl,
        consensus_jsonl,
        root / "hf-review-package",
        field_policy=field_policy,
    )
    report_body = {
        "schema_version": "automation-recovery-0.1.0",
        "parent_candidate_id": parent.candidate.candidate_id,
        "retry_http_statuses": sorted(statuses),
        "parent_candidate_sha256": sha256_bytes(parent.candidate_jsonl.read_bytes()),
        "parent_translation_sha256": sha256_bytes(parent.translation_jsonl.read_bytes()),
        "parent_mini_results_sha256": sha256_bytes(parent.mini_results_jsonl.read_bytes()),
        "parent_strong_results_sha256": (
            sha256_bytes(parent.strong_results_jsonl.read_bytes())
            if parent.strong_results_jsonl is not None
            else None
        ),
        "retried_translation_episodes": len(translation_retry_ids),
        "recovered_translation_episodes": len(recovered_translation_ids),
        "retried_mini_units": len(mini_retry_payload),
        "recovered_mini_units": sum(item.verdict is not None for item in recovered_mini),
        "retried_strong_units": len(retry_strong_inputs),
        "recovered_strong_units": sum(item.verdict is not None for item in recovered_strong),
        "effective_translation_manifest_id": effective_translation_manifest.manifest_id,
        "effective_mini_manifest_id": effective_mini_manifest.manifest_id,
        "effective_strong_manifest_id": (
            effective_strong_manifest.manifest_id if effective_strong_manifest is not None else None
        ),
        "consensus_report_id": consensus.report_id,
        "hf_package_id": package.package_id,
        "promotion": "not_eligible",
        "publish_allowed": False,
    }
    report = AutomationRecoveryReport(
        recovery_id=stable_id("autorecovery", report_body),
        parent_candidate_id=parent.candidate.candidate_id,
        retry_http_statuses=sorted(statuses),
        parent_candidate_sha256=sha256_bytes(parent.candidate_jsonl.read_bytes()),
        parent_translation_sha256=sha256_bytes(parent.translation_jsonl.read_bytes()),
        parent_mini_results_sha256=sha256_bytes(parent.mini_results_jsonl.read_bytes()),
        parent_strong_results_sha256=(
            sha256_bytes(parent.strong_results_jsonl.read_bytes())
            if parent.strong_results_jsonl is not None
            else None
        ),
        retried_translation_episodes=len(translation_retry_ids),
        recovered_translation_episodes=len(recovered_translation_ids),
        retried_mini_units=len(mini_retry_payload),
        recovered_mini_units=sum(item.verdict is not None for item in recovered_mini),
        retried_strong_units=len(retry_strong_inputs),
        recovered_strong_units=sum(item.verdict is not None for item in recovered_strong),
        effective_translation_manifest_id=effective_translation_manifest.manifest_id,
        effective_mini_manifest_id=effective_mini_manifest.manifest_id,
        effective_strong_manifest_id=(
            effective_strong_manifest.manifest_id if effective_strong_manifest is not None else None
        ),
        consensus_report_id=consensus.report_id,
        hf_package_id=package.package_id,
    )
    publish_bytes_atomic(
        root / "reports" / f"{report.recovery_id}.json",
        canonical_bytes(report) + b"\n",
    )
    return report
