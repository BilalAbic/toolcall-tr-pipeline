"""Bounded, resumable automation for review-ready translation candidates.

This module deliberately sits beside the original single-shot operational
translation contract.  It never rewrites its immutable evidence.  Instead it
selects a deterministic, non-promoting cohort and routes each episode through
a primary DeepSeek model and, when that produces a *known safe* terminal
failure or ``research_needed``, an approved fallback model.  An uncertain
network delivery is never resent automatically: the affected episode is
retained as a review item while unrelated work continues.

The output is still only a candidate set.  Model consensus and the explicit
human publication approval are downstream stages.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, ValidationError, model_validator

from toolcall_tr.artifacts import ContentManifest, publish_bytes_atomic, publish_jsonl_artifact
from toolcall_tr.audit import AuditId, ExactConflictAudit
from toolcall_tr.config import PipelineConfig, ProviderConfig, ProviderRole
from toolcall_tr.eval_contract import EvaluationUnit, SegmentPathEvidence, build_evaluation_unit
from toolcall_tr.field_policy import FieldPolicy, extract_leaf_segments
from toolcall_tr.hashing import JsonValue, canonical_bytes, sha256_bytes, sha256_jcs, stable_id
from toolcall_tr.jsonio import iter_jsonl, loads_strict_bytes
from toolcall_tr.live_evaluation import (
    LiveEvaluationInput,
    LiveEvaluationResult,
    build_live_evaluation_input,
)
from toolcall_tr.live_preflight import LivePreflightBlockedError
from toolcall_tr.models import (
    CanonicalEpisode,
    CanonicalTool,
    EpisodeId,
    Message,
    Sha256,
    StrictModel,
)
from toolcall_tr.operational_translation import (
    OperationalTranslationError,
    OperationalTranslationResult,
    run_operational_translation,
)
from toolcall_tr.prompt_contract import PromptBundle, PromptContractError, require_validated_prompt
from toolcall_tr.provider_adapter import ProviderAdapterError, ResponsesTransport
from toolcall_tr.provider_provenance import ProviderAttemptRecord, ProviderFailureCode
from toolcall_tr.provider_usage import ProviderUsageSinkError
from toolcall_tr.secure_transport import SecureTransportError

AUTOMATION_POLICY_VERSION = "autonomous-candidate-0.1.1"
AutomationCandidateId = Annotated[str, Field(pattern=r"^autocand_[0-9a-f]{64}$")]
AutomationRouteId = Annotated[str, Field(pattern=r"^autoroute_[0-9a-f]{64}$")]
AutomationResultId = Annotated[str, Field(pattern=r"^autotr_[0-9a-f]{64}$")]
AutomationBatchId = Annotated[str, Field(pattern=r"^autobatch_[0-9a-f]{64}$")]


class AutonomousPipelineError(RuntimeError):
    """Raised when an automation boundary or immutable receipt is unsafe."""


class AutomationCandidateMember(StrictModel):
    """One deterministic, non-promoting source episode selected for automation."""

    rank: Annotated[int, Field(gt=0)]
    episode_id: EpisodeId
    input_variant_id: Sha256
    dataset_namespace: str
    rank_key: Sha256
    translatable_segments: Annotated[int, Field(gt=0)]


class AutomationCandidateManifest(StrictModel):
    """A bounded and stratified review-candidate cohort, never Gold membership."""

    schema_version: Literal["autonomous-candidate-0.1.1"] = AUTOMATION_POLICY_VERSION
    candidate_id: AutomationCandidateId
    input_file_sha256s: Annotated[list[Sha256], Field(min_length=1)]
    audit_ids: Annotated[list[AuditId], Field(min_length=1)]
    field_policy_sha256: Sha256
    requested_episode_count: Annotated[int, Field(ge=1, le=1_000)]
    candidate_offset: Annotated[int, Field(ge=0)]
    max_translatable_segments: Annotated[int, Field(ge=1)]
    source_row_cap: Annotated[int, Field(ge=1)] | None
    members: Annotated[list[AutomationCandidateMember], Field(min_length=1, max_length=1_000)]
    total_translatable_segments: Annotated[int, Field(gt=0)]
    canonical_manifest_id: Annotated[str, Field(pattern=r"^manifest_[0-9a-f]{64}$")]
    promotion: Literal["not_eligible"] = "not_eligible"
    human_review: Literal["required_before_publish"] = "required_before_publish"

    @model_validator(mode="after")
    def validate_manifest(self) -> AutomationCandidateManifest:
        if self.input_file_sha256s != sorted(set(self.input_file_sha256s)):
            raise ValueError("candidate input file hashes must be unique and sorted")
        if self.audit_ids != sorted(set(self.audit_ids)):
            raise ValueError("candidate audit IDs must be unique and sorted")
        if len(self.members) != self.requested_episode_count:
            raise ValueError("candidate must contain the requested number of episodes")
        if [member.rank for member in self.members] != list(range(1, len(self.members) + 1)):
            raise ValueError("candidate member ranks must be contiguous")
        if len({member.episode_id for member in self.members}) != len(self.members):
            raise ValueError("candidate episode IDs must be unique")
        if self.total_translatable_segments != sum(
            member.translatable_segments for member in self.members
        ):
            raise ValueError("candidate segment total must equal its member rows")
        if self.total_translatable_segments > self.max_translatable_segments:
            raise ValueError("candidate exceeds its declared segment budget")
        body = self.model_dump(mode="json", exclude={"candidate_id"})
        if self.candidate_id != stable_id("autocand", body):
            raise ValueError("candidate ID does not match deterministic content")
        return self


class AutomationRoute(StrictModel):
    """One immutable model route for an episode.

    A route result embeds the regular operational translation result so its
    normal source rehash and host-merge guarantees remain visible.  A failed
    route carries only a stable failure code, not remote error text.
    """

    schema_version: Literal["autonomous-route-0.1.0"] = "autonomous-route-0.1.0"
    route_id: AutomationRouteId
    episode_id: EpisodeId
    route: Literal["primary", "fallback"]
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"]
    status: Literal["translated", "research_needed", "no_translatable_segments", "failed"]
    translation_report_id: Annotated[str, Field(pattern=r"^trbatch_[0-9a-f]{64}$")] | None
    translation_result: OperationalTranslationResult | None
    failure_code: ProviderFailureCode | None

    @model_validator(mode="after")
    def validate_route(self) -> AutomationRoute:
        succeeded = self.status != "failed"
        if succeeded != (self.translation_result is not None):
            raise ValueError("successful route state must carry one translation result")
        if succeeded != (self.translation_report_id is not None):
            raise ValueError("successful route state must carry one translation report")
        if self.status == "failed":
            if self.failure_code is None:
                raise ValueError("failed route requires a stable failure code")
        elif self.failure_code is not None:
            raise ValueError("successful route cannot carry a failure code")
        if self.translation_result is not None and (
            self.translation_result.episode_id != self.episode_id
            or self.translation_result.status != self.status
        ):
            raise ValueError("route result must match its episode and status")
        body = self.model_dump(mode="json", exclude={"route_id"})
        if self.route_id != stable_id("autoroute", body):
            raise ValueError("route ID does not match deterministic content")
        return self


class AutomationEpisodeResult(StrictModel):
    """One terminal translation decision; failed records do not stop a batch."""

    schema_version: Literal["autonomous-translation-result-0.1.0"] = (
        "autonomous-translation-result-0.1.0"
    )
    result_id: AutomationResultId
    episode_id: EpisodeId
    input_variant_id: Sha256
    routes: Annotated[list[AutomationRoute], Field(min_length=1, max_length=2)]
    selected_route_id: AutomationRouteId | None
    status: Literal["translated", "needs_review"]
    translation_result: OperationalTranslationResult | None
    promotion: Literal["not_eligible"] = "not_eligible"

    @model_validator(mode="after")
    def validate_result(self) -> AutomationEpisodeResult:
        route_ids = [route.route_id for route in self.routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("automation routes must be unique")
        if [route.route for route in self.routes] not in (["primary"], ["primary", "fallback"]):
            raise ValueError("automation routes must be primary then optional fallback")
        selected = next(
            (route for route in self.routes if route.route_id == self.selected_route_id), None
        )
        if self.status == "translated":
            if (
                selected is None
                or selected.status != "translated"
                or self.translation_result != selected.translation_result
            ):
                raise ValueError("translated automation result must select a translated route")
        elif self.selected_route_id is not None or self.translation_result is not None:
            raise ValueError("review-needed automation result cannot carry selected translation")
        body = self.model_dump(mode="json", exclude={"result_id"})
        if self.result_id != stable_id("autotr", body):
            raise ValueError("automation result ID does not match deterministic content")
        return self


class AutomationTranslationReport(StrictModel):
    """Aggregate receipt for a batch that can continue after per-episode errors."""

    schema_version: Literal["autonomous-translation-0.1.0"] = "autonomous-translation-0.1.0"
    batch_id: AutomationBatchId
    input_file_sha256: Sha256
    field_policy_sha256: Sha256
    prompt_id: Annotated[str, Field(pattern=r"^prompt_[0-9a-f]{64}$")]
    source_records: Annotated[int, Field(gt=0)]
    translated_records: Annotated[int, Field(ge=0)]
    needs_review_records: Annotated[int, Field(ge=0)]
    primary_routes: Annotated[int, Field(ge=0)]
    fallback_routes: Annotated[int, Field(ge=0)]
    result_manifest_id: Annotated[str, Field(pattern=r"^manifest_[0-9a-f]{64}$")]
    promotion: Literal["not_eligible"] = "not_eligible"

    @model_validator(mode="after")
    def validate_report(self) -> AutomationTranslationReport:
        if self.source_records != self.translated_records + self.needs_review_records:
            raise ValueError("automation translation row accounting must balance")
        if self.primary_routes != self.source_records:
            raise ValueError("every source record must receive exactly one primary route")
        if self.fallback_routes > self.source_records:
            raise ValueError("fallback routes cannot exceed source records")
        body = self.model_dump(mode="json", exclude={"batch_id"})
        if self.batch_id != stable_id("autobatch", body):
            raise ValueError("automation batch ID does not match deterministic content")
        return self


class AutomationConsensus(StrictModel):
    """Independent mini/strong judge agreement for one exact translated leaf."""

    schema_version: Literal["autonomous-consensus-0.1.0"] = "autonomous-consensus-0.1.0"
    consensus_id: Annotated[str, Field(pattern=r"^autoconsensus_[0-9a-f]{64}$")]
    evaluation_unit: EvaluationUnit
    mini_result_id: Annotated[str, Field(pattern=r"^liveevalresult_[0-9a-f]{64}$")]
    mini_verdict_id: Annotated[str, Field(pattern=r"^evalverdict_[0-9a-f]{64}$")] | None
    mini_conclusion: Literal["pass", "needs_human_review", "fail", "unavailable"]
    strong_result_id: Annotated[str, Field(pattern=r"^liveevalresult_[0-9a-f]{64}$")]
    strong_verdict_id: Annotated[str, Field(pattern=r"^evalverdict_[0-9a-f]{64}$")] | None
    strong_conclusion: Literal["pass", "needs_human_review", "fail", "unavailable"]
    status: Literal["accepted_for_review_package", "needs_review"]
    gold_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_consensus(self) -> AutomationConsensus:
        mini_available = self.mini_conclusion != "unavailable"
        strong_available = self.strong_conclusion != "unavailable"
        if mini_available != (self.mini_verdict_id is not None):
            raise ValueError("mini verdict ID must match mini availability")
        if strong_available != (self.strong_verdict_id is not None):
            raise ValueError("strong verdict ID must match strong availability")
        both_pass = self.mini_conclusion == "pass" and self.strong_conclusion == "pass"
        if (self.status == "accepted_for_review_package") != both_pass:
            raise ValueError("only two independent pass verdicts may enter the review package")
        body = self.model_dump(mode="json", exclude={"consensus_id"})
        if self.consensus_id != stable_id("autoconsensus", body):
            raise ValueError("consensus ID does not match deterministic content")
        return self


class AutomationConsensusReport(StrictModel):
    """Aggregate receipt for two-model quality consensus, never Gold acceptance."""

    schema_version: Literal["autonomous-consensus-report-0.1.0"] = (
        "autonomous-consensus-report-0.1.0"
    )
    report_id: Annotated[str, Field(pattern=r"^autoconsreport_[0-9a-f]{64}$")]
    mini_results_sha256: Sha256
    strong_results_sha256: Sha256
    requested_units: Annotated[int, Field(gt=0)]
    accepted_units: Annotated[int, Field(ge=0)]
    needs_review_units: Annotated[int, Field(ge=0)]
    consensus_manifest_id: Annotated[str, Field(pattern=r"^manifest_[0-9a-f]{64}$")]
    gold_release_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> AutomationConsensusReport:
        if self.requested_units != self.accepted_units + self.needs_review_units:
            raise ValueError("consensus row accounting must balance")
        body = self.model_dump(mode="json", exclude={"report_id"})
        if self.report_id != stable_id("autoconsreport", body):
            raise ValueError("consensus report ID does not match deterministic content")
        return self


class HierarchicalConsensus(StrictModel):
    """Final decision after mini-first review and optional strong escalation."""

    schema_version: Literal["hierarchical-consensus-0.1.0"] = "hierarchical-consensus-0.1.0"
    consensus_id: Annotated[str, Field(pattern=r"^autohconsensus_[0-9a-f]{64}$")]
    evaluation_unit: EvaluationUnit
    mini_result_id: Annotated[str, Field(pattern=r"^liveevalresult_[0-9a-f]{64}$")]
    mini_verdict_id: Annotated[str, Field(pattern=r"^evalverdict_[0-9a-f]{64}$")] | None
    mini_conclusion: Literal["pass", "needs_human_review", "fail", "unavailable"]
    escalation_reason: Literal["mini_non_pass", "pass_sample"] | None
    strong_result_id: Annotated[str, Field(pattern=r"^liveevalresult_[0-9a-f]{64}$")] | None
    strong_verdict_id: Annotated[str, Field(pattern=r"^evalverdict_[0-9a-f]{64}$")] | None
    strong_conclusion: Literal["pass", "needs_human_review", "fail", "unavailable"] | None
    final_decider: Literal["mini", "strong"]
    final_conclusion: Literal["pass", "needs_human_review", "fail", "unavailable"]
    status: Literal["accepted_for_review_package", "needs_review"]
    gold_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_hierarchy(self) -> HierarchicalConsensus:
        mini_available = self.mini_conclusion != "unavailable"
        strong_available = self.strong_conclusion not in {None, "unavailable"}
        if mini_available != (self.mini_verdict_id is not None):
            raise ValueError("mini verdict ID must match mini availability")
        if self.escalation_reason is None:
            if (
                self.mini_conclusion != "pass"
                or self.strong_result_id is not None
                or self.strong_verdict_id is not None
                or self.strong_conclusion is not None
                or self.final_decider != "mini"
                or self.final_conclusion != "pass"
                or self.status != "accepted_for_review_package"
            ):
                raise ValueError("only an unsampled mini pass may finish at the mini stage")
        else:
            if self.strong_result_id is None or self.strong_conclusion is None:
                raise ValueError("escalation requires one strong terminal result")
            if strong_available != (self.strong_verdict_id is not None):
                raise ValueError("strong verdict ID must match strong availability")
            if self.final_decider != "strong" or self.final_conclusion != self.strong_conclusion:
                raise ValueError("escalated decision must be decided by the strong result")
        if (self.status == "accepted_for_review_package") != (self.final_conclusion == "pass"):
            raise ValueError("only a final pass may enter the review package")
        body = self.model_dump(mode="json", exclude={"consensus_id"})
        if self.consensus_id != stable_id("autohconsensus", body):
            raise ValueError("hierarchical consensus ID does not match deterministic content")
        return self


class HierarchicalConsensusReport(StrictModel):
    """Receipt for one mini-first, selectively escalated quality decision set."""

    schema_version: Literal["hierarchical-consensus-report-0.1.0"] = (
        "hierarchical-consensus-report-0.1.0"
    )
    report_id: Annotated[str, Field(pattern=r"^autohconsreport_[0-9a-f]{64}$")]
    mini_results_sha256: Sha256
    strong_results_sha256: Sha256 | None
    pass_sample_basis_points: Annotated[int, Field(ge=0, le=10_000)]
    requested_units: Annotated[int, Field(gt=0)]
    strong_escalated_units: Annotated[int, Field(ge=0)]
    accepted_units: Annotated[int, Field(ge=0)]
    needs_review_units: Annotated[int, Field(ge=0)]
    consensus_manifest_id: Annotated[str, Field(pattern=r"^manifest_[0-9a-f]{64}$")]
    gold_release_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> HierarchicalConsensusReport:
        if self.requested_units != self.accepted_units + self.needs_review_units:
            raise ValueError("hierarchical consensus row accounting must balance")
        if self.strong_escalated_units > self.requested_units:
            raise ValueError("strong escalation cannot exceed requested units")
        body = self.model_dump(mode="json", exclude={"report_id"})
        if self.report_id != stable_id("autohconsreport", body):
            raise ValueError(
                "hierarchical consensus report ID does not match deterministic content"
            )
        return self


class HuggingFaceDatasetRow(StrictModel):
    """One JSONL row directly loadable as a nested tool-calling dataset."""

    schema_version: Literal["hf-tool-calling-row-0.1.0"] = "hf-tool-calling-row-0.1.0"
    id: EpisodeId
    messages: list[Message]
    tools: list[CanonicalTool]
    source_dataset_namespace: str
    source_snapshot_ids: list[str]
    quality_tier: Literal["silver_candidate"] = "silver_candidate"
    consensus_status: Literal["two_judge_pass"] = "two_judge_pass"
    human_approval_required: Literal[True] = True


class HierarchicalHuggingFaceDatasetRow(StrictModel):
    """One current JSONL row directly loadable as a nested tool-calling dataset."""

    schema_version: Literal["hf-tool-calling-row-0.1.1"] = "hf-tool-calling-row-0.1.1"
    id: EpisodeId
    messages: list[Message]
    tools: list[CanonicalTool]
    source_dataset_namespace: str
    source_snapshot_ids: list[str]
    quality_tier: Literal["silver_candidate"] = "silver_candidate"
    consensus_status: Literal["accepted_for_review_package"] = "accepted_for_review_package"
    human_approval_required: Literal[True] = True


class HuggingFaceReviewPackage(StrictModel):
    """Content-addressed upload directory, deliberately pending human approval."""

    schema_version: Literal["hf-review-package-0.1.0"] = "hf-review-package-0.1.0"
    package_id: Annotated[str, Field(pattern=r"^hfpackage_[0-9a-f]{64}$")]
    candidate_jsonl_sha256: Sha256
    automation_results_sha256: Sha256
    consensus_jsonl_sha256: Sha256
    train_jsonl_sha256: Sha256
    source_records: Annotated[int, Field(gt=0)]
    review_ready_records: Annotated[int, Field(ge=0)]
    needs_review_records: Annotated[int, Field(ge=0)]
    upload_path: Literal["data/train.jsonl"] = "data/train.jsonl"
    status: Literal["pending_human_approval"] = "pending_human_approval"
    publish_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_package(self) -> HuggingFaceReviewPackage:
        if self.source_records != self.review_ready_records + self.needs_review_records:
            raise ValueError("HF review package row accounting must balance")
        body = self.model_dump(mode="json", exclude={"package_id"})
        if self.package_id != stable_id("hfpackage", body):
            raise ValueError("HF package ID does not match deterministic content")
        return self


def _within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _candidate_rank_key(episode: CanonicalEpisode) -> str:
    return sha256_jcs(
        {
            "candidate_policy_version": AUTOMATION_POLICY_VERSION,
            "episode_id": episode.episode_id,
            "input_variant_id": episode.variant_id,
        }
    )


def _conflicted_ids(audits: Iterable[ExactConflictAudit]) -> set[str]:
    return {
        episode_id
        for audit in audits
        for candidate in audit.conflict_candidates
        for episode_id in [
            *candidate.left_member_episode_ids,
            *candidate.right_member_episode_ids,
        ]
    }


def _alias_ids(audits: Iterable[ExactConflictAudit]) -> set[str]:
    return {
        episode_id
        for audit in audits
        for group in audit.duplicate_groups
        for episode_id in group.alias_episode_ids
    }


def _dataset_namespace(episode: CanonicalEpisode) -> str:
    namespaces = {source.dataset_namespace for source in episode.provenance.sources}
    if len(namespaces) != 1:
        raise AutonomousPipelineError("candidate episode must have exactly one dataset namespace")
    return next(iter(namespaces))


def select_automation_candidates(
    episodes: Iterable[CanonicalEpisode],
    audits: Iterable[ExactConflictAudit],
    *,
    field_policy: FieldPolicy,
    requested_episode_count: int,
    max_translatable_segments: int,
    candidate_offset: int = 0,
) -> tuple[list[AutomationCandidateMember], list[CanonicalEpisode]]:
    """Choose a deterministic, source-stratified cohort without deleting records."""
    if not 1 <= requested_episode_count <= 1_000:
        raise AutonomousPipelineError("candidate size must be between 1 and 1000")
    if max_translatable_segments < 1:
        raise AutonomousPipelineError("candidate segment budget must be positive")
    if candidate_offset < 0:
        raise AutonomousPipelineError("candidate offset must not be negative")
    episode_by_id: dict[str, CanonicalEpisode] = {}
    for episode in episodes:
        if episode.episode_id in episode_by_id:
            raise AutonomousPipelineError(f"duplicate canonical episode ID: {episode.episode_id}")
        episode_by_id[episode.episode_id] = episode
    audit_list = list(audits)
    excluded = _conflicted_ids(audit_list) | _alias_ids(audit_list)
    strata: dict[str, deque[CanonicalEpisode]] = defaultdict(deque)
    for episode in episode_by_id.values():
        if (
            episode.quality.state != "unreviewed"
            or episode.annotations.decision.evidence_status != "source_explicit"
            or episode.episode_id in excluded
        ):
            continue
        strata[_dataset_namespace(episode)].append(episode)
    for namespace, values in strata.items():
        strata[namespace] = deque(
            sorted(values, key=lambda item: (_candidate_rank_key(item), item.episode_id))
        )
    selected: list[tuple[CanonicalEpisode, int]] = []
    skipped_candidates = 0
    segment_total = 0
    namespaces = sorted(strata)
    # Round-robin retains source diversity.  A row that cannot fit the current
    # budget is skipped, never mutated or removed from its source artifact.
    while namespaces and len(selected) < requested_episode_count:
        made_progress = False
        remaining: list[str] = []
        for index, namespace in enumerate(namespaces):
            queue = strata[namespace]
            while queue:
                episode = queue.popleft()
                segment_count = len(extract_leaf_segments(episode, field_policy).segments)
                if not segment_count:
                    continue
                if skipped_candidates < candidate_offset:
                    skipped_candidates += 1
                    made_progress = True
                    # Preserve one-candidate-per-stratum round-robin order while
                    # consuming the offset stream for a disjoint later batch.
                    break
                if segment_total + segment_count > max_translatable_segments:
                    raise AutonomousPipelineError(
                        "candidate offset window exceeds the requested segment budget"
                    )
                selected.append((episode, segment_count))
                segment_total += segment_count
                made_progress = True
                break
            if queue:
                remaining.append(namespace)
            if len(selected) == requested_episode_count:
                remaining.extend(name for name in namespaces[index + 1 :] if strata[name])
                break
        namespaces = sorted(set(remaining))
        if not made_progress:
            break
    if len(selected) != requested_episode_count:
        raise AutonomousPipelineError(
            "insufficient conflict-free, policy-covered episodes after the candidate offset"
        )
    members = [
        AutomationCandidateMember(
            rank=rank,
            episode_id=episode.episode_id,
            input_variant_id=episode.variant_id,
            dataset_namespace=_dataset_namespace(episode),
            rank_key=_candidate_rank_key(episode),
            translatable_segments=segment_count,
        )
        for rank, (episode, segment_count) in enumerate(selected, start=1)
    ]
    return members, [episode for episode, _ in selected]


def _read_audit(path: Path) -> ExactConflictAudit:
    parsed = loads_strict_bytes(path.read_bytes())
    if not isinstance(parsed, dict):
        raise AutonomousPipelineError("conflict audit must be a JSON object")
    return ExactConflictAudit.model_validate(parsed, strict=True)


def prepare_automation_candidates(
    canonical_jsonl_paths: Iterable[Path],
    audit_paths: Iterable[Path],
    output_root: Path,
    *,
    field_policy: FieldPolicy,
    requested_episode_count: int,
    max_translatable_segments: int,
    source_row_cap: int | None = None,
    candidate_offset: int = 0,
) -> AutomationCandidateManifest:
    """Publish a bounded candidate canonical JSONL in a root disjoint from inputs."""
    canonical_paths = [path.resolve(strict=True) for path in canonical_jsonl_paths]
    audits_paths = [path.resolve(strict=True) for path in audit_paths]
    if not canonical_paths or not audits_paths:
        raise AutonomousPipelineError(
            "candidate preparation requires canonical JSONL and audit evidence"
        )
    if any(not path.is_file() or path.suffix.lower() != ".jsonl" for path in canonical_paths):
        raise AutonomousPipelineError("candidate canonical inputs must be existing JSONL files")
    if any(not path.is_file() or path.suffix.lower() != ".json" for path in audits_paths):
        raise AutonomousPipelineError("candidate audit inputs must be existing JSON files")
    if source_row_cap is not None and source_row_cap < 1:
        raise AutonomousPipelineError("candidate source row cap must be positive when specified")
    if candidate_offset < 0:
        raise AutonomousPipelineError("candidate offset must not be negative")
    root = output_root.resolve(strict=False)
    if output_root.exists() and not output_root.is_dir():
        raise AutonomousPipelineError("candidate output root must be a directory")
    for path in [*canonical_paths, *audits_paths]:
        if _within(path, root) or _within(root, path.parent):
            raise AutonomousPipelineError("candidate output root must be disjoint from inputs")
    input_hashes = {path: sha256_bytes(path.read_bytes()) for path in canonical_paths}
    audit_hashes = {path: sha256_bytes(path.read_bytes()) for path in audits_paths}
    episodes = [
        CanonicalEpisode.model_validate_json(canonical_bytes(record), strict=True)
        for path in canonical_paths
        for index, record in enumerate(iter_jsonl(path), start=1)
        if source_row_cap is None or index <= source_row_cap
    ]
    audits = [_read_audit(path) for path in audits_paths]
    members, selected = select_automation_candidates(
        episodes,
        audits,
        field_policy=field_policy,
        requested_episode_count=requested_episode_count,
        max_translatable_segments=max_translatable_segments,
        candidate_offset=candidate_offset,
    )
    if any(sha256_bytes(path.read_bytes()) != digest for path, digest in input_hashes.items()):
        raise AutonomousPipelineError("canonical input changed during candidate preparation")
    if any(sha256_bytes(path.read_bytes()) != digest for path, digest in audit_hashes.items()):
        raise AutonomousPipelineError("conflict audit changed during candidate preparation")
    canonical_manifest = publish_jsonl_artifact(
        root / "canonical",
        logical_name="autonomous-candidate-canonical",
        schema_version="0.1.0",
        stage="autonomous-candidate",
        records=[episode.model_dump(mode="json", exclude_none=False) for episode in selected],
        contract_hashes={
            "field_policy": sha256_jcs(field_policy),
            "input_canonical_jsonl": sha256_jcs(sorted(input_hashes.values())),
        },
    )
    body: dict[str, object] = {
        "schema_version": AUTOMATION_POLICY_VERSION,
        "input_file_sha256s": sorted(set(input_hashes.values())),
        "audit_ids": sorted(audit.audit_id for audit in audits),
        "field_policy_sha256": sha256_jcs(field_policy),
        "requested_episode_count": requested_episode_count,
        "candidate_offset": candidate_offset,
        "max_translatable_segments": max_translatable_segments,
        "source_row_cap": source_row_cap,
        "members": [member.model_dump(mode="json") for member in members],
        "total_translatable_segments": sum(member.translatable_segments for member in members),
        "canonical_manifest_id": canonical_manifest.manifest_id,
        "promotion": "not_eligible",
        "human_review": "required_before_publish",
    }
    manifest = AutomationCandidateManifest(
        candidate_id=stable_id("autocand", body),
        input_file_sha256s=sorted(set(input_hashes.values())),
        audit_ids=sorted(audit.audit_id for audit in audits),
        field_policy_sha256=sha256_jcs(field_policy),
        requested_episode_count=requested_episode_count,
        candidate_offset=candidate_offset,
        max_translatable_segments=max_translatable_segments,
        source_row_cap=source_row_cap,
        members=members,
        total_translatable_segments=sum(member.translatable_segments for member in members),
        canonical_manifest_id=canonical_manifest.manifest_id,
    )
    publish_bytes_atomic(
        root / "candidates" / f"{manifest.candidate_id}.json", canonical_bytes(manifest) + b"\n"
    )
    return manifest


def _alternate_translator_model(config: PipelineConfig, model: str) -> PipelineConfig:
    role = config.providers.translator
    translator = ProviderRole(
        provider=role.provider,
        model=model,
        api_key_env=role.api_key_env,
        endpoint=role.endpoint,
        temperature=role.temperature,
        thinking=role.thinking,
        max_workers=role.max_workers,
        daily_token_budget=role.daily_token_budget,
    )
    return PipelineConfig(
        schema_version=config.schema_version,
        canonical_schema_version=config.canonical_schema_version,
        diagnostic_catalog_version=config.diagnostic_catalog_version,
        normalizer_version=config.normalizer_version,
        max_record_bytes=config.max_record_bytes,
        jsonl_shard_rows=config.jsonl_shard_rows,
        providers=ProviderConfig(
            enabled=config.providers.enabled,
            network_egress_enabled=config.providers.network_egress_enabled,
            translator=translator,
            strong_judge=config.providers.strong_judge,
            mini_verifier=config.providers.mini_verifier,
        ),
    )


def _read_single_operational_result(root: Path) -> OperationalTranslationResult:
    files = sorted((root / "translation-results").glob("*.jsonl"))
    if len(files) != 1:
        raise AutonomousPipelineError("successful route must publish exactly one result JSONL")
    rows = list(iter_jsonl(files[0]))
    if len(rows) != 1:
        raise AutonomousPipelineError("one-episode route must publish exactly one result row")
    return OperationalTranslationResult.model_validate_json(canonical_bytes(rows[0]), strict=True)


def _route_failure(root: Path) -> ProviderFailureCode:
    records: list[ProviderAttemptRecord] = []
    for path in sorted((root / "provider-attempts").glob("*.json")):
        records.append(
            ProviderAttemptRecord.model_validate_json(path.read_text(encoding="utf-8"), strict=True)
        )
    failed = [record for record in records if record.failure_code is not None]
    if not failed:
        return ProviderFailureCode.UNKNOWN
    return failed[-1].failure_code or ProviderFailureCode.UNKNOWN


def _route_receipt_path(root: Path, episode_id: str, route: str) -> Path:
    return root / "route-receipts" / f"{episode_id}-{route}.json"


def _load_route_receipt(path: Path) -> AutomationRoute:
    return AutomationRoute.model_validate_json(path.read_text(encoding="utf-8"), strict=True)


def _run_route(
    *,
    root: Path,
    episode: CanonicalEpisode,
    route: Literal["primary", "fallback"],
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"],
    config: PipelineConfig,
    field_policy: FieldPolicy,
    prompt: PromptBundle,
    transport: ResponsesTransport,
) -> AutomationRoute:
    receipt_path = _route_receipt_path(root, episode.episode_id, route)
    if receipt_path.exists():
        existing = _load_route_receipt(receipt_path)
        if (
            existing.episode_id != episode.episode_id
            or existing.model != model
            or existing.route != route
        ):
            raise AutonomousPipelineError("persisted route receipt conflicts with requested route")
        return existing
    input_path = root / "episode-inputs" / f"{episode.episode_id}.jsonl"
    publish_bytes_atomic(input_path, canonical_bytes(episode) + b"\n")
    route_root = root / "routes" / episode.episode_id / route
    try:
        report = run_operational_translation(
            input_path,
            route_root,
            config=_alternate_translator_model(config, model),
            field_policy=field_policy,
            prompt=prompt,
            transport=transport,
        )
        result = _read_single_operational_result(route_root)
        status = result.status
        body: dict[str, object] = {
            "schema_version": "autonomous-route-0.1.0",
            "episode_id": episode.episode_id,
            "route": route,
            "model": model,
            "status": status,
            "translation_report_id": report.batch_id,
            "translation_result": result.model_dump(mode="json", exclude_none=False),
            "failure_code": None,
        }
        receipt = AutomationRoute(
            route_id=stable_id("autoroute", body),
            episode_id=episode.episode_id,
            route=route,
            model=model,
            status=status,
            translation_report_id=report.batch_id,
            translation_result=result,
            failure_code=None,
        )
    except (
        LivePreflightBlockedError,
        OperationalTranslationError,
        ProviderAdapterError,
        ProviderUsageSinkError,
        SecureTransportError,
        ValueError,
    ):
        failure = _route_failure(route_root)
        body = {
            "schema_version": "autonomous-route-0.1.0",
            "episode_id": episode.episode_id,
            "route": route,
            "model": model,
            "status": "failed",
            "translation_report_id": None,
            "translation_result": None,
            "failure_code": failure,
        }
        receipt = AutomationRoute(
            route_id=stable_id("autoroute", body),
            episode_id=episode.episode_id,
            route=route,
            model=model,
            status="failed",
            translation_report_id=None,
            translation_result=None,
            failure_code=failure,
        )
    publish_bytes_atomic(receipt_path, canonical_bytes(receipt) + b"\n")
    return receipt


def _should_fallback(route: AutomationRoute) -> bool:
    if route.status == "research_needed":
        return True
    if route.status != "failed":
        return False
    # A network delivery may have reached the provider.  Do not risk a second
    # paid request automatically; preserve the episode as a review item.
    return route.failure_code in {
        ProviderFailureCode.HTTP_TRANSIENT,
        ProviderFailureCode.MALFORMED_RESPONSE,
        ProviderFailureCode.PROVIDER_RESPONSE_INVALID,
        ProviderFailureCode.RESPONSE_TOO_LARGE,
    }


def run_automation_translation(
    candidate_jsonl: Path,
    output_root: Path,
    *,
    config: PipelineConfig,
    field_policy: FieldPolicy,
    prompt: PromptBundle,
    transport: ResponsesTransport,
    max_workers: int = 1,
) -> AutomationTranslationReport:
    """Translate every candidate, continuing after safe failures with one fallback."""
    try:
        require_validated_prompt(prompt)
    except PromptContractError as exc:
        raise AutonomousPipelineError(str(exc)) from exc
    if not config.providers.enabled or not config.providers.network_egress_enabled:
        raise AutonomousPipelineError("automation translation requires both live provider gates")
    if not 1 <= max_workers <= 16:
        raise AutonomousPipelineError(
            "automation translation worker count must be between 1 and 16"
        )
    if (
        config.providers.translator.provider != "deepseek"
        or config.providers.translator.endpoint is None
    ):
        raise AutonomousPipelineError("automation translation requires an explicit DeepSeek role")
    input_path = candidate_jsonl.resolve(strict=True)
    root = output_root.resolve(strict=False)
    if (
        not input_path.is_file()
        or input_path.suffix.lower() != ".jsonl"
        or (output_root.exists() and not output_root.is_dir())
    ):
        raise AutonomousPipelineError("automation input/output paths are invalid")
    if _within(root, input_path.parent) or _within(input_path, root):
        raise AutonomousPipelineError(
            "automation output root must be disjoint from candidate input"
        )
    input_digest = sha256_bytes(input_path.read_bytes())
    episodes = [
        CanonicalEpisode.model_validate_json(canonical_bytes(record), strict=True)
        for record in iter_jsonl(input_path)
    ]
    if not episodes:
        raise AutonomousPipelineError("automation candidate input must not be empty")
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise AutonomousPipelineError("automation candidate input contains duplicate episode IDs")

    def process_episode(episode: CanonicalEpisode) -> AutomationEpisodeResult:
        primary = _run_route(
            root=root,
            episode=episode,
            route="primary",
            model="deepseek-v4-flash",
            config=config,
            field_policy=field_policy,
            prompt=prompt,
            transport=transport,
        )
        routes = [primary]
        selected: AutomationRoute | None = primary if primary.status == "translated" else None
        if selected is None and _should_fallback(primary):
            fallback = _run_route(
                root=root,
                episode=episode,
                route="fallback",
                model="deepseek-v4-pro",
                config=config,
                field_policy=field_policy,
                prompt=prompt,
                transport=transport,
            )
            routes.append(fallback)
            if fallback.status == "translated":
                selected = fallback
        body: dict[str, object] = {
            "schema_version": "autonomous-translation-result-0.1.0",
            "episode_id": episode.episode_id,
            "input_variant_id": episode.variant_id,
            "routes": [route.model_dump(mode="json", exclude_none=False) for route in routes],
            "selected_route_id": selected.route_id if selected is not None else None,
            "status": "translated" if selected is not None else "needs_review",
            "translation_result": (
                selected.translation_result.model_dump(mode="json", exclude_none=False)
                if selected is not None and selected.translation_result is not None
                else None
            ),
            "promotion": "not_eligible",
        }
        result = AutomationEpisodeResult(
            result_id=stable_id("autotr", body),
            episode_id=episode.episode_id,
            input_variant_id=episode.variant_id,
            routes=routes,
            selected_route_id=selected.route_id if selected is not None else None,
            status="translated" if selected is not None else "needs_review",
            translation_result=(selected.translation_result if selected is not None else None),
        )
        publish_bytes_atomic(
            root / "episode-results" / f"{result.result_id}.json", canonical_bytes(result) + b"\n"
        )
        return result

    ordered_episodes = sorted(episodes, key=lambda item: (item.episode_id, item.variant_id))
    if max_workers == 1:
        results = [process_episode(episode) for episode in ordered_episodes]
    else:
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tcdata-translate"
        ) as pool:
            results = list(pool.map(process_episode, ordered_episodes))
    if sha256_bytes(input_path.read_bytes()) != input_digest:
        raise AutonomousPipelineError("automation candidate input changed during translation")
    ordered = sorted(results, key=lambda item: (item.episode_id, item.input_variant_id))
    manifest = publish_jsonl_artifact(
        root / "translation-results",
        logical_name="autonomous-translation-results",
        schema_version="autonomous-translation-result-0.1.0",
        stage="autonomous-translation",
        records=[item.model_dump(mode="json", exclude_none=False) for item in ordered],
        contract_hashes={
            "field_policy": sha256_jcs(field_policy),
            "prompt_bundle": sha256_jcs(prompt),
            "candidate_jsonl": input_digest,
        },
    )
    translated = sum(item.status == "translated" for item in ordered)
    body = {
        "schema_version": "autonomous-translation-0.1.0",
        "input_file_sha256": input_digest,
        "field_policy_sha256": sha256_jcs(field_policy),
        "prompt_id": prompt.prompt_id,
        "source_records": len(ordered),
        "translated_records": translated,
        "needs_review_records": len(ordered) - translated,
        "primary_routes": len(ordered),
        "fallback_routes": sum(len(item.routes) == 2 for item in ordered),
        "result_manifest_id": manifest.manifest_id,
        "promotion": "not_eligible",
    }
    report = AutomationTranslationReport(
        batch_id=stable_id("autobatch", body),
        input_file_sha256=input_digest,
        field_policy_sha256=sha256_jcs(field_policy),
        prompt_id=prompt.prompt_id,
        source_records=len(ordered),
        translated_records=translated,
        needs_review_records=len(ordered) - translated,
        primary_routes=len(ordered),
        fallback_routes=sum(len(item.routes) == 2 for item in ordered),
        result_manifest_id=manifest.manifest_id,
    )
    publish_bytes_atomic(
        root / "batches" / f"{report.batch_id}.json", canonical_bytes(report) + b"\n"
    )
    return report


def translated_results_for_consensus(
    results: Iterable[AutomationEpisodeResult],
) -> list[OperationalTranslationResult]:
    """Return only fully host-merged translations; review records remain separate."""
    translated = [item.translation_result for item in results if item.status == "translated"]
    if any(item is None for item in translated):
        raise AutonomousPipelineError("translated automation result is missing its translation")
    return sorted(
        cast(list[OperationalTranslationResult], translated),
        key=lambda item: (item.episode_id, item.input_variant_id),
    )


def read_automation_results(path: Path) -> list[AutomationEpisodeResult]:
    """Read one strict automation result JSONL with unique episode coverage."""
    results = [
        AutomationEpisodeResult.model_validate_json(canonical_bytes(record), strict=True)
        for record in iter_jsonl(path)
    ]
    episode_ids = [result.episode_id for result in results]
    if episode_ids != sorted(set(episode_ids)):
        raise AutonomousPipelineError("automation result JSONL must be unique and sorted")
    return results


def _pointer_text(document: JsonValue, pointer: str) -> str:
    if not pointer.startswith("/"):
        raise AutonomousPipelineError("evaluation segment pointer must be absolute")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdecimal() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise AutonomousPipelineError(
                f"translated segment pointer no longer resolves: {pointer}"
            )
    if not isinstance(current, str):
        raise AutonomousPipelineError(f"translated segment is not textual: {pointer}")
    return current


def build_automation_evaluation_inputs(
    episodes: Iterable[CanonicalEpisode],
    results: Iterable[AutomationEpisodeResult],
    *,
    field_policy: FieldPolicy,
) -> list[LiveEvaluationInput]:
    """Build full-leaf judge inputs only for complete host-merged translations."""
    episode_list = list(episodes)
    result_list = list(results)
    episodes_by_id = {episode.episode_id: episode for episode in episode_list}
    if len(episodes_by_id) != len(episode_list):
        # The iterable is normally a list; this protects callers that supply a
        # duplicate sequence without allowing silent key replacement.
        raise AutonomousPipelineError("automation candidate episodes must be unique")
    results_by_id = {result.episode_id: result for result in result_list}
    if len(results_by_id) != len(result_list):
        raise AutonomousPipelineError("automation results must be unique")
    if set(results_by_id) != set(episodes_by_id):
        raise AutonomousPipelineError(
            "automation results must cover exactly the candidate episodes"
        )
    inputs: list[LiveEvaluationInput] = []
    policy_sha = sha256_jcs(field_policy)
    for episode_id in sorted(episodes_by_id):
        episode = episodes_by_id[episode_id]
        result = results_by_id[episode_id]
        if result.status != "translated":
            continue
        translated = result.translation_result
        if (
            translated is None
            or translated.status != "translated"
            or translated.translated_episode is None
            or translated.episode_id != episode.episode_id
            or translated.input_variant_id != episode.variant_id
            or translated.field_policy_sha256 != policy_sha
        ):
            raise AutonomousPipelineError(
                "translated automation result is not exact host-merged output"
            )
        document = translated.translated_episode.model_dump(mode="json", exclude_none=False)
        for segment in extract_leaf_segments(episode, field_policy).segments:
            target_text = _pointer_text(document, segment.json_pointer)
            unit = build_evaluation_unit(
                episode_id=episode.episode_id,
                segment_id=segment.segment_id,
                path=segment.json_pointer,
                source_text_sha256=sha256_bytes(segment.source_text.encode("utf-8")),
                target_text_sha256=sha256_bytes(target_text.encode("utf-8")),
            )
            inputs.append(
                build_live_evaluation_input(
                    evaluation_unit=unit,
                    evidence=SegmentPathEvidence(
                        segment_id=segment.segment_id,
                        path=segment.json_pointer,
                        source_excerpt=segment.source_text,
                        target_excerpt=target_text,
                    ),
                )
            )
    input_ids = [item.input_id for item in inputs]
    if len(input_ids) != len(set(input_ids)):
        raise AutonomousPipelineError("automation evaluation inputs must be unique")
    return sorted(inputs, key=lambda item: item.input_id)


def prepare_automation_evaluation_inputs(
    candidate_jsonl: Path,
    results_jsonl: Path,
    output_root: Path,
    *,
    field_policy: FieldPolicy,
) -> ContentManifest:
    """Publish full-leaf consensus inputs in a directory disjoint from evidence."""
    candidate_path = candidate_jsonl.resolve(strict=True)
    results_path = results_jsonl.resolve(strict=True)
    root = output_root.resolve(strict=False)
    if (
        not candidate_path.is_file()
        or candidate_path.suffix.lower() != ".jsonl"
        or not results_path.is_file()
        or results_path.suffix.lower() != ".jsonl"
        or (output_root.exists() and not output_root.is_dir())
    ):
        raise AutonomousPipelineError("automation evaluation input/output paths are invalid")
    if any(
        _within(path, root) or _within(root, path.parent) for path in (candidate_path, results_path)
    ):
        raise AutonomousPipelineError(
            "automation evaluation output root must be disjoint from inputs"
        )
    candidate_sha = sha256_bytes(candidate_path.read_bytes())
    results_sha = sha256_bytes(results_path.read_bytes())
    episodes = [
        CanonicalEpisode.model_validate_json(canonical_bytes(record), strict=True)
        for record in iter_jsonl(candidate_path)
    ]
    results = read_automation_results(results_path)
    inputs = build_automation_evaluation_inputs(episodes, results, field_policy=field_policy)
    if not inputs:
        raise AutonomousPipelineError(
            "no complete translations are available for consensus evaluation"
        )
    if (
        sha256_bytes(candidate_path.read_bytes()) != candidate_sha
        or sha256_bytes(results_path.read_bytes()) != results_sha
    ):
        raise AutonomousPipelineError("automation evaluation input changed during preparation")
    return publish_jsonl_artifact(
        root,
        logical_name="autonomous-live-evaluation-inputs",
        schema_version="live-evaluation-input-0.1.0",
        stage="autonomous-evaluation-inputs",
        records=[item.model_dump(mode="json", exclude_none=False) for item in inputs],
        contract_hashes={
            "candidate_jsonl": candidate_sha,
            "automation_results": results_sha,
            "field_policy": sha256_jcs(field_policy),
        },
    )


def _read_live_results(path: Path) -> list[LiveEvaluationResult]:
    results = [
        LiveEvaluationResult.model_validate_json(canonical_bytes(record), strict=True)
        for record in iter_jsonl(path)
    ]
    unit_ids = [result.evaluation_unit.unit_id for result in results]
    if len(unit_ids) != len(set(unit_ids)):
        raise AutonomousPipelineError("judge result JSONL must have unique evaluation units")
    return results


def _conclusion(
    result: LiveEvaluationResult,
) -> Literal["pass", "needs_human_review", "fail", "unavailable"]:
    return result.verdict.conclusion if result.verdict is not None else "unavailable"


def _is_sampled_pass(unit_id: str, pass_sample_basis_points: int) -> bool:
    if not 0 <= pass_sample_basis_points <= 10_000:
        raise AutonomousPipelineError("pass sample basis points must be between 0 and 10000")
    bucket = int(sha256_bytes(unit_id.encode("utf-8"))[7:15], 16) % 10_000
    return bucket < pass_sample_basis_points


def _escalation_reason(
    result: LiveEvaluationResult,
    *,
    pass_sample_basis_points: int,
) -> Literal["mini_non_pass", "pass_sample"] | None:
    conclusion = _conclusion(result)
    if conclusion != "pass":
        return "mini_non_pass"
    return (
        "pass_sample"
        if _is_sampled_pass(result.evaluation_unit.unit_id, pass_sample_basis_points)
        else None
    )


def prepare_strong_escalation_inputs(
    evaluation_inputs_jsonl: Path,
    mini_results_jsonl: Path,
    output_root: Path,
    *,
    pass_sample_basis_points: int,
) -> ContentManifest | None:
    """Select mini non-passes and a deterministic sample of mini passes for GPT.

    A mini ``pass`` outside the declared sample proceeds without a second model
    call.  This is intentional cost control, recorded by the eventual
    hierarchical consensus receipt rather than inferred from missing rows.
    """
    input_path = evaluation_inputs_jsonl.resolve(strict=True)
    mini_path = mini_results_jsonl.resolve(strict=True)
    root = output_root.resolve(strict=False)
    if (
        not input_path.is_file()
        or input_path.suffix.lower() != ".jsonl"
        or not mini_path.is_file()
        or mini_path.suffix.lower() != ".jsonl"
        or (output_root.exists() and not output_root.is_dir())
    ):
        raise AutonomousPipelineError("strong escalation input/output paths are invalid")
    if _within(root, input_path.parent) or _within(root, mini_path.parent):
        raise AutonomousPipelineError("strong escalation output root must be disjoint from inputs")
    input_sha = sha256_bytes(input_path.read_bytes())
    mini_sha = sha256_bytes(mini_path.read_bytes())
    inputs = [
        LiveEvaluationInput.model_validate_json(canonical_bytes(record), strict=True)
        for record in iter_jsonl(input_path)
    ]
    inputs_by_unit = {item.evaluation_unit.unit_id: item for item in inputs}
    mini_by_unit = {
        result.evaluation_unit.unit_id: result for result in _read_live_results(mini_path)
    }
    if not inputs_by_unit or set(inputs_by_unit) != set(mini_by_unit):
        raise AutonomousPipelineError("mini results must cover every evaluation input exactly")
    selected = [
        inputs_by_unit[unit_id]
        for unit_id in sorted(inputs_by_unit)
        if _escalation_reason(
            mini_by_unit[unit_id], pass_sample_basis_points=pass_sample_basis_points
        )
        is not None
    ]
    if not selected:
        return None
    if (
        sha256_bytes(input_path.read_bytes()) != input_sha
        or sha256_bytes(mini_path.read_bytes()) != mini_sha
    ):
        raise AutonomousPipelineError("strong escalation input changed during preparation")
    return publish_jsonl_artifact(
        root,
        logical_name="autonomous-strong-escalation-inputs",
        schema_version="live-evaluation-input-0.1.0",
        stage="autonomous-strong-escalation-inputs",
        records=[item.model_dump(mode="json", exclude_none=False) for item in selected],
        contract_hashes={
            "evaluation_inputs": input_sha,
            "mini_results": mini_sha,
            "pass_sample_basis_points": sha256_bytes(f"{pass_sample_basis_points:04d}".encode()),
        },
    )


def build_hierarchical_consensus(
    mini_results_jsonl: Path,
    strong_results_jsonl: Path | None,
    output_root: Path,
    *,
    pass_sample_basis_points: int,
) -> HierarchicalConsensusReport:
    """Finalise mini-first decisions, using GPT only for declared escalations."""
    mini_path = mini_results_jsonl.resolve(strict=True)
    root = output_root.resolve(strict=False)
    if (
        not mini_path.is_file()
        or mini_path.suffix.lower() != ".jsonl"
        or (
            strong_results_jsonl is not None
            and not strong_results_jsonl.resolve(strict=True).is_file()
        )
        or (output_root.exists() and not output_root.is_dir())
    ):
        raise AutonomousPipelineError("hierarchical consensus input/output paths are invalid")
    strong_path = strong_results_jsonl.resolve(strict=True) if strong_results_jsonl else None
    paths = (mini_path,) if strong_path is None else (mini_path, strong_path)
    if any(_within(root, path.parent) for path in paths):
        raise AutonomousPipelineError(
            "hierarchical consensus output root must be disjoint from inputs"
        )
    mini_sha = sha256_bytes(mini_path.read_bytes())
    strong_sha = sha256_bytes(strong_path.read_bytes()) if strong_path is not None else None
    mini_by_unit = {
        result.evaluation_unit.unit_id: result for result in _read_live_results(mini_path)
    }
    if not mini_by_unit:
        raise AutonomousPipelineError("hierarchical consensus requires mini results")
    expected_escalations = {
        unit_id
        for unit_id, result in mini_by_unit.items()
        if _escalation_reason(result, pass_sample_basis_points=pass_sample_basis_points) is not None
    }
    strong_by_unit = (
        {result.evaluation_unit.unit_id: result for result in _read_live_results(strong_path)}
        if strong_path is not None
        else {}
    )
    if set(strong_by_unit) != expected_escalations:
        raise AutonomousPipelineError("strong results must cover exactly the selected escalations")
    decisions: list[HierarchicalConsensus] = []
    for unit_id in sorted(mini_by_unit):
        mini = mini_by_unit[unit_id]
        reason = _escalation_reason(mini, pass_sample_basis_points=pass_sample_basis_points)
        mini_conclusion = _conclusion(mini)
        strong = strong_by_unit.get(unit_id)
        if reason is None:
            strong_conclusion = None
            final_decider: Literal["mini", "strong"] = "mini"
            final_conclusion: Literal["pass", "needs_human_review", "fail", "unavailable"] = "pass"
            status: Literal["accepted_for_review_package", "needs_review"] = (
                "accepted_for_review_package"
            )
        else:
            if strong is None or strong.evaluation_unit != mini.evaluation_unit:
                raise AutonomousPipelineError(
                    "strong escalation result lost its exact evaluation unit"
                )
            strong_conclusion = _conclusion(strong)
            final_decider = "strong"
            final_conclusion = strong_conclusion
            status = "accepted_for_review_package" if final_conclusion == "pass" else "needs_review"
        body: dict[str, object] = {
            "schema_version": "hierarchical-consensus-0.1.0",
            "evaluation_unit": mini.evaluation_unit.model_dump(mode="json"),
            "mini_result_id": mini.result_id,
            "mini_verdict_id": mini.verdict.verdict_id if mini.verdict is not None else None,
            "mini_conclusion": mini_conclusion,
            "escalation_reason": reason,
            "strong_result_id": strong.result_id if strong is not None else None,
            "strong_verdict_id": strong.verdict.verdict_id
            if strong is not None and strong.verdict is not None
            else None,
            "strong_conclusion": strong_conclusion,
            "final_decider": final_decider,
            "final_conclusion": final_conclusion,
            "status": status,
            "gold_eligible": False,
        }
        decisions.append(
            HierarchicalConsensus(
                consensus_id=stable_id("autohconsensus", body),
                evaluation_unit=mini.evaluation_unit,
                mini_result_id=mini.result_id,
                mini_verdict_id=mini.verdict.verdict_id if mini.verdict is not None else None,
                mini_conclusion=mini_conclusion,
                escalation_reason=reason,
                strong_result_id=strong.result_id if strong is not None else None,
                strong_verdict_id=(
                    strong.verdict.verdict_id
                    if strong is not None and strong.verdict is not None
                    else None
                ),
                strong_conclusion=strong_conclusion,
                final_decider=final_decider,
                final_conclusion=final_conclusion,
                status=status,
            )
        )
    if sha256_bytes(mini_path.read_bytes()) != mini_sha or (
        strong_path is not None
        and strong_sha is not None
        and sha256_bytes(strong_path.read_bytes()) != strong_sha
    ):
        raise AutonomousPipelineError("hierarchical consensus input changed during execution")
    manifest = publish_jsonl_artifact(
        root / "consensus",
        logical_name="autonomous-hierarchical-consensus",
        schema_version="hierarchical-consensus-0.1.0",
        stage="autonomous-hierarchical-consensus",
        records=[item.model_dump(mode="json", exclude_none=False) for item in decisions],
        contract_hashes={
            "mini_results": mini_sha,
            "strong_results": strong_sha or sha256_bytes(b"no-strong-escalations"),
            "pass_sample_basis_points": sha256_bytes(f"{pass_sample_basis_points:04d}".encode()),
        },
    )
    accepted = sum(item.status == "accepted_for_review_package" for item in decisions)
    body = {
        "schema_version": "hierarchical-consensus-report-0.1.0",
        "mini_results_sha256": mini_sha,
        "strong_results_sha256": strong_sha,
        "pass_sample_basis_points": pass_sample_basis_points,
        "requested_units": len(decisions),
        "strong_escalated_units": len(expected_escalations),
        "accepted_units": accepted,
        "needs_review_units": len(decisions) - accepted,
        "consensus_manifest_id": manifest.manifest_id,
        "gold_release_allowed": False,
    }
    report = HierarchicalConsensusReport(
        report_id=stable_id("autohconsreport", body),
        mini_results_sha256=mini_sha,
        strong_results_sha256=strong_sha,
        pass_sample_basis_points=pass_sample_basis_points,
        requested_units=len(decisions),
        strong_escalated_units=len(expected_escalations),
        accepted_units=accepted,
        needs_review_units=len(decisions) - accepted,
        consensus_manifest_id=manifest.manifest_id,
    )
    publish_bytes_atomic(
        root / "reports" / f"{report.report_id}.json", canonical_bytes(report) + b"\n"
    )
    return report


def build_automation_consensus(
    mini_results_jsonl: Path,
    strong_results_jsonl: Path,
    output_root: Path,
) -> AutomationConsensusReport:
    """Require independent mini and strong passes for each review-package leaf."""
    mini_path = mini_results_jsonl.resolve(strict=True)
    strong_path = strong_results_jsonl.resolve(strict=True)
    root = output_root.resolve(strict=False)
    if (
        not mini_path.is_file()
        or mini_path.suffix.lower() != ".jsonl"
        or not strong_path.is_file()
        or strong_path.suffix.lower() != ".jsonl"
        or (output_root.exists() and not output_root.is_dir())
    ):
        raise AutonomousPipelineError("consensus input/output paths are invalid")
    if any(_within(path, root) or _within(root, path.parent) for path in (mini_path, strong_path)):
        raise AutonomousPipelineError("consensus output root must be disjoint from judge inputs")
    mini_sha = sha256_bytes(mini_path.read_bytes())
    strong_sha = sha256_bytes(strong_path.read_bytes())
    mini_by_unit = {
        result.evaluation_unit.unit_id: result for result in _read_live_results(mini_path)
    }
    strong_by_unit = {
        result.evaluation_unit.unit_id: result for result in _read_live_results(strong_path)
    }
    if not mini_by_unit or set(mini_by_unit) != set(strong_by_unit):
        raise AutonomousPipelineError("mini and strong judge inputs must have exact unit coverage")
    decisions: list[AutomationConsensus] = []
    for unit_id in sorted(mini_by_unit):
        mini = mini_by_unit[unit_id]
        strong = strong_by_unit[unit_id]
        if mini.evaluation_unit != strong.evaluation_unit:
            raise AutonomousPipelineError(
                "judge results disagree on immutable evaluation unit identity"
            )
        mini_verdict = mini.verdict
        strong_verdict = strong.verdict
        mini_conclusion: Literal["pass", "needs_human_review", "fail", "unavailable"] = (
            mini_verdict.conclusion if mini_verdict is not None else "unavailable"
        )
        strong_conclusion: Literal["pass", "needs_human_review", "fail", "unavailable"] = (
            strong_verdict.conclusion if strong_verdict is not None else "unavailable"
        )
        status: Literal["accepted_for_review_package", "needs_review"] = (
            "accepted_for_review_package"
            if mini_conclusion == "pass" and strong_conclusion == "pass"
            else "needs_review"
        )
        body: dict[str, object] = {
            "schema_version": "autonomous-consensus-0.1.0",
            "evaluation_unit": mini.evaluation_unit.model_dump(mode="json"),
            "mini_result_id": mini.result_id,
            "mini_verdict_id": mini_verdict.verdict_id if mini_verdict is not None else None,
            "mini_conclusion": mini_conclusion,
            "strong_result_id": strong.result_id,
            "strong_verdict_id": strong_verdict.verdict_id if strong_verdict is not None else None,
            "strong_conclusion": strong_conclusion,
            "status": status,
            "gold_eligible": False,
        }
        decisions.append(
            AutomationConsensus(
                consensus_id=stable_id("autoconsensus", body),
                evaluation_unit=mini.evaluation_unit,
                mini_result_id=mini.result_id,
                mini_verdict_id=mini_verdict.verdict_id if mini_verdict is not None else None,
                mini_conclusion=mini_conclusion,
                strong_result_id=strong.result_id,
                strong_verdict_id=strong_verdict.verdict_id if strong_verdict is not None else None,
                strong_conclusion=strong_conclusion,
                status=status,
            )
        )
    if (
        sha256_bytes(mini_path.read_bytes()) != mini_sha
        or sha256_bytes(strong_path.read_bytes()) != strong_sha
    ):
        raise AutonomousPipelineError("judge results changed during consensus")
    manifest = publish_jsonl_artifact(
        root / "consensus",
        logical_name="autonomous-consensus",
        schema_version="autonomous-consensus-0.1.0",
        stage="autonomous-consensus",
        records=[item.model_dump(mode="json", exclude_none=False) for item in decisions],
        contract_hashes={"mini_results": mini_sha, "strong_results": strong_sha},
    )
    accepted = sum(item.status == "accepted_for_review_package" for item in decisions)
    body = {
        "schema_version": "autonomous-consensus-report-0.1.0",
        "mini_results_sha256": mini_sha,
        "strong_results_sha256": strong_sha,
        "requested_units": len(decisions),
        "accepted_units": accepted,
        "needs_review_units": len(decisions) - accepted,
        "consensus_manifest_id": manifest.manifest_id,
        "gold_release_allowed": False,
    }
    report = AutomationConsensusReport(
        report_id=stable_id("autoconsreport", body),
        mini_results_sha256=mini_sha,
        strong_results_sha256=strong_sha,
        requested_units=len(decisions),
        accepted_units=accepted,
        needs_review_units=len(decisions) - accepted,
        consensus_manifest_id=manifest.manifest_id,
    )
    publish_bytes_atomic(
        root / "reports" / f"{report.report_id}.json", canonical_bytes(report) + b"\n"
    )
    return report


def _accepted_consensus_unit_ids(path: Path) -> set[str]:
    """Read either legacy two-judge or current hierarchical consensus safely."""
    records = list(iter_jsonl(path))
    try:
        decisions: list[AutomationConsensus | HierarchicalConsensus] = [
            AutomationConsensus.model_validate_json(canonical_bytes(record), strict=True)
            for record in records
        ]
    except ValidationError as legacy_error:
        try:
            decisions = [
                HierarchicalConsensus.model_validate_json(canonical_bytes(record), strict=True)
                for record in records
            ]
        except ValidationError as hierarchy_error:
            raise AutonomousPipelineError(
                "consensus JSONL must use a supported strict consensus schema"
            ) from hierarchy_error
        if not decisions and records:
            raise AutonomousPipelineError(
                "hierarchical consensus JSONL is invalid"
            ) from legacy_error
    unit_ids = [item.evaluation_unit.unit_id for item in decisions]
    if unit_ids != sorted(set(unit_ids)):
        raise AutonomousPipelineError(
            "consensus JSONL must be unique and sorted by evaluation unit"
        )
    return {
        item.evaluation_unit.unit_id
        for item in decisions
        if item.status == "accepted_for_review_package"
    }


def _hf_row(episode: CanonicalEpisode) -> HierarchicalHuggingFaceDatasetRow:
    snapshots = sorted({source.snapshot_id for source in episode.provenance.sources})
    return HierarchicalHuggingFaceDatasetRow(
        id=episode.episode_id,
        messages=episode.conversation,
        tools=episode.tools,
        source_dataset_namespace=_dataset_namespace(episode),
        source_snapshot_ids=snapshots,
    )


def _dataset_card(*, package: HuggingFaceReviewPackage, namespaces: list[str]) -> bytes:
    sources = "\n".join(f"- `{name}`" for name in namespaces)
    text = (
        "---\n"
        "language:\n- tr\n"
        "license: cc-by-4.0\n"
        "task_categories:\n- text-generation\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: data/train.jsonl\n"
        "---\n\n"
        "# Turkish Tool Calling review candidate\n\n"
        "This package is a local, quality-evaluated silver candidate and remains "
        "pending explicit human approval. It must not be published as Gold without "
        "that approval.\n\n"
        f"- Review-ready records: {package.review_ready_records}\n"
        f"- Records retained for review: {package.needs_review_records}\n"
        f"- Package ID: `{package.package_id}`\n\n"
        "## Upstream sources\n\n"
        f"{sources}\n"
    )
    return text.encode("utf-8")


def _validate_huggingface_rows(rows: Iterable[HierarchicalHuggingFaceDatasetRow]) -> None:
    """Round-trip each rendered JSONL row through the strict public wire contract."""
    for row in rows:
        HierarchicalHuggingFaceDatasetRow.model_validate_json(canonical_bytes(row), strict=True)


def build_huggingface_review_package(
    candidate_jsonl: Path,
    automation_results_jsonl: Path,
    consensus_jsonl: Path,
    output_root: Path,
    *,
    field_policy: FieldPolicy,
) -> HuggingFaceReviewPackage:
    """Render an upload-ready JSONL directory while retaining the human publish gate."""
    candidate_path = candidate_jsonl.resolve(strict=True)
    results_path = automation_results_jsonl.resolve(strict=True)
    consensus_path = consensus_jsonl.resolve(strict=True)
    root = output_root.resolve(strict=False)
    paths = (candidate_path, results_path, consensus_path)
    if any(not path.is_file() or path.suffix.lower() != ".jsonl" for path in paths) or (
        output_root.exists() and not output_root.is_dir()
    ):
        raise AutonomousPipelineError("HF package input/output paths are invalid")
    if any(_within(path, root) or _within(root, path.parent) for path in paths):
        raise AutonomousPipelineError("HF package output root must be disjoint from inputs")
    candidate_sha = sha256_bytes(candidate_path.read_bytes())
    results_sha = sha256_bytes(results_path.read_bytes())
    consensus_sha = sha256_bytes(consensus_path.read_bytes())
    episodes = [
        CanonicalEpisode.model_validate_json(canonical_bytes(record), strict=True)
        for record in iter_jsonl(candidate_path)
    ]
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    if len(episode_by_id) != len(episodes):
        raise AutonomousPipelineError("HF package candidate episodes must be unique")
    results_by_id = {result.episode_id: result for result in read_automation_results(results_path)}
    if set(results_by_id) != set(episode_by_id):
        raise AutonomousPipelineError(
            "HF package results must cover exactly the candidate episodes"
        )
    accepted_consensus_unit_ids = _accepted_consensus_unit_ids(consensus_path)
    accepted: list[HierarchicalHuggingFaceDatasetRow] = []
    for episode_id in sorted(episode_by_id):
        episode = episode_by_id[episode_id]
        result = results_by_id[episode_id]
        if result.status != "translated" or result.translation_result is None:
            continue
        translated = result.translation_result.translated_episode
        if translated is None:
            raise AutonomousPipelineError(
                "translated automation result is missing its target episode"
            )
        expected_units: list[str] = []
        translated_document = translated.model_dump(mode="json", exclude_none=False)
        for segment in extract_leaf_segments(episode, field_policy).segments:
            target_text = _pointer_text(translated_document, segment.json_pointer)
            unit = build_evaluation_unit(
                episode_id=episode.episode_id,
                segment_id=segment.segment_id,
                path=segment.json_pointer,
                source_text_sha256=sha256_bytes(segment.source_text.encode("utf-8")),
                target_text_sha256=sha256_bytes(target_text.encode("utf-8")),
            )
            expected_units.append(unit.unit_id)
        if not expected_units:
            raise AutonomousPipelineError("review package episode has no translatable segments")
        if all(unit_id in accepted_consensus_unit_ids for unit_id in expected_units):
            accepted.append(_hf_row(translated))
    ordered_rows = sorted(accepted, key=lambda item: item.id)
    _validate_huggingface_rows(ordered_rows)
    payload = b"".join(canonical_bytes(row) + b"\n" for row in ordered_rows)
    train_sha = sha256_bytes(payload)
    body = {
        "schema_version": "hf-review-package-0.1.0",
        "candidate_jsonl_sha256": candidate_sha,
        "automation_results_sha256": results_sha,
        "consensus_jsonl_sha256": consensus_sha,
        "train_jsonl_sha256": train_sha,
        "source_records": len(episodes),
        "review_ready_records": len(ordered_rows),
        "needs_review_records": len(episodes) - len(ordered_rows),
        "upload_path": "data/train.jsonl",
        "status": "pending_human_approval",
        "publish_allowed": False,
    }
    package = HuggingFaceReviewPackage(
        package_id=stable_id("hfpackage", body),
        candidate_jsonl_sha256=candidate_sha,
        automation_results_sha256=results_sha,
        consensus_jsonl_sha256=consensus_sha,
        train_jsonl_sha256=train_sha,
        source_records=len(episodes),
        review_ready_records=len(ordered_rows),
        needs_review_records=len(episodes) - len(ordered_rows),
    )
    target = root / package.package_id
    namespaces = sorted({_dataset_namespace(episode) for episode in episode_by_id.values()})
    dataset_info = {
        "schema_version": "hf-dataset-info-0.1.0",
        "package_id": package.package_id,
        "format": "openai-tool-calling-jsonl",
        "splits": {"train": package.review_ready_records},
        "status": package.status,
        "publish_allowed": False,
    }
    if (
        sha256_bytes(candidate_path.read_bytes()) != candidate_sha
        or sha256_bytes(results_path.read_bytes()) != results_sha
        or sha256_bytes(consensus_path.read_bytes()) != consensus_sha
    ):
        raise AutonomousPipelineError("HF package inputs changed during rendering")
    publish_bytes_atomic(target / "data" / "train.jsonl", payload)
    publish_bytes_atomic(target / "dataset_info.json", canonical_bytes(dataset_info) + b"\n")
    publish_bytes_atomic(
        target / "README.md", _dataset_card(package=package, namespaces=namespaces)
    )
    publish_bytes_atomic(target / "manifest.json", canonical_bytes(package) + b"\n")
    return package
