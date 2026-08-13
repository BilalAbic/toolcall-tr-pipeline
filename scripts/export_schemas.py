"""Export immutable Draft 2020-12 contracts from their Pydantic source."""

from __future__ import annotations

import json
from pathlib import Path

from toolcall_tr.adjudication import ConflictAdjudication
from toolcall_tr.artifacts import ContentManifest
from toolcall_tr.audit import ConflictCandidate, ExactConflictAudit, ExactDuplicateGroup
from toolcall_tr.constants import JSON_SCHEMA_DIALECT
from toolcall_tr.diagnostics import Diagnostic, DiagnosticCatalog
from toolcall_tr.egress_guard import EgressRequest, EgressViolation, PreEgressDecision
from toolcall_tr.eval_contract import (
    CoverageSummary,
    EvaluationReport,
    EvaluationUnit,
    FindingCount,
    GoldAcceptance,
    HumanEvaluationReview,
    ModelEvaluationVerdict,
    MqmFinding,
    OutcomeSummary,
    SegmentPathEvidence,
    WilsonConfidenceInterval,
)
from toolcall_tr.events import RunEvent
from toolcall_tr.field_policy import (
    ArgumentPathPolicy,
    FieldPolicy,
    SegmentExtraction,
)
from toolcall_tr.field_policy import (
    Segment as FieldPolicySegment,
)
from toolcall_tr.field_policy import (
    SegmentTranslation as FieldPolicySegmentTranslation,
)
from toolcall_tr.hashing import canonical_bytes
from toolcall_tr.human_review_log import HumanEvaluationReviewEntry
from toolcall_tr.live_evaluation import (
    LiveEvaluationInput,
    LiveEvaluationResult,
    LiveEvaluationRunReport,
)
from toolcall_tr.live_preflight import LivePreflightDecision
from toolcall_tr.models import CanonicalEpisode
from toolcall_tr.openai_judge import JudgeFindingOutput, JudgeOutput
from toolcall_tr.operational_translation import (
    LeafTranslationRecord,
    OperationalTranslationReport,
    OperationalTranslationResult,
)
from toolcall_tr.phase4_config import Phase4Config
from toolcall_tr.pilot import CanonicalQuarantineRecord, PilotRunReport, TolerantPilotRunReport
from toolcall_tr.prompt_contract import PromptBundle, PromptLayer
from toolcall_tr.provider_provenance import (
    ProviderAttemptRecord,
    RetryBudgetClassification,
)
from toolcall_tr.release_contract import (
    ReleaseDatasetFile,
    ReleaseGoldMember,
    ReleaseManifest,
)
from toolcall_tr.render_contract import (
    CharacterRange,
    LossMask,
    RenderArtifact,
    RenderCandidate,
    RenderConfig,
    SupervisedRender,
    TargetPayloadRange,
    TokenizedText,
    TokenRange,
)
from toolcall_tr.research_policy import (
    ResearchBudget,
    ResearchCandidate,
    ResearchRequest,
    ResearchResolution,
    TerminologyInput,
    TerminologyRisk,
)
from toolcall_tr.review_queue import ReviewTask
from toolcall_tr.selection import SelectionCandidate, SelectionManifest
from toolcall_tr.similarity import NearDuplicateCandidate, SimilarityDocument
from toolcall_tr.source import BronzeRecord, SourceSnapshot
from toolcall_tr.source_array import JsonArrayConversionReport
from toolcall_tr.source_evidence import (
    ArgumentEvidenceInput,
    SourceEvidence,
    SourceEvidenceRequest,
)
from toolcall_tr.split_guard import SplitLeakage
from toolcall_tr.translation_contract import (
    ProtectedToken,
    TranslationRequest,
    TranslationResponse,
    TranslationSegment,
)
from toolcall_tr.translation_contract import (
    SegmentTranslation as ContractSegmentTranslation,
)
from toolcall_tr.translation_memory import MemoryLookupKey, SegmentMemoryEntry

MODELS = {
    "bronze-record.schema.json": BronzeRecord,
    "canonical-episode.schema.json": CanonicalEpisode,
    "argument-path-policy.schema.json": ArgumentPathPolicy,
    "conflict-adjudication.schema.json": ConflictAdjudication,
    "conflict-candidate.schema.json": ConflictCandidate,
    "content-manifest.schema.json": ContentManifest,
    "diagnostic-catalog.schema.json": DiagnosticCatalog,
    "diagnostic.schema.json": Diagnostic,
    "egress-request.schema.json": EgressRequest,
    "egress-violation.schema.json": EgressViolation,
    "evaluation-report.schema.json": EvaluationReport,
    "evaluation-unit.schema.json": EvaluationUnit,
    "exact-conflict-audit.schema.json": ExactConflictAudit,
    "exact-duplicate-group.schema.json": ExactDuplicateGroup,
    "field-policy.schema.json": FieldPolicy,
    "field-policy-segment.schema.json": FieldPolicySegment,
    "finding-count.schema.json": FindingCount,
    "gold-acceptance.schema.json": GoldAcceptance,
    "human-evaluation-review.schema.json": HumanEvaluationReview,
    "human-evaluation-review-entry.schema.json": HumanEvaluationReviewEntry,
    "model-evaluation-verdict.schema.json": ModelEvaluationVerdict,
    "mqm-finding.schema.json": MqmFinding,
    "outcome-summary.schema.json": OutcomeSummary,
    "segment-extraction.schema.json": SegmentExtraction,
    "segment-translation.schema.json": FieldPolicySegmentTranslation,
    "near-duplicate-candidate.schema.json": NearDuplicateCandidate,
    "phase4-config.schema.json": Phase4Config,
    "operational-pilot.schema.json": PilotRunReport,
    "operational-pilot-tolerant.schema.json": TolerantPilotRunReport,
    "canonical-quarantine.schema.json": CanonicalQuarantineRecord,
    "operational-translation.schema.json": OperationalTranslationReport,
    "operational-translation-result.schema.json": OperationalTranslationResult,
    "leaf-translation-record.schema.json": LeafTranslationRecord,
    "prompt-bundle.schema.json": PromptBundle,
    "prompt-layer.schema.json": PromptLayer,
    "pre-egress-decision.schema.json": PreEgressDecision,
    "protected-token.schema.json": ProtectedToken,
    "research-budget.schema.json": ResearchBudget,
    "research-candidate.schema.json": ResearchCandidate,
    "research-request.schema.json": ResearchRequest,
    "research-resolution.schema.json": ResearchResolution,
    "release-dataset-file.schema.json": ReleaseDatasetFile,
    "release-gold-member.schema.json": ReleaseGoldMember,
    "release-manifest.schema.json": ReleaseManifest,
    "review-task.schema.json": ReviewTask,
    "render-artifact.schema.json": RenderArtifact,
    "render-candidate.schema.json": RenderCandidate,
    "render-character-range.schema.json": CharacterRange,
    "render-config.schema.json": RenderConfig,
    "render-loss-mask.schema.json": LossMask,
    "render-supervised.schema.json": SupervisedRender,
    "render-target-payload-range.schema.json": TargetPayloadRange,
    "render-token-range.schema.json": TokenRange,
    "tokenized-text.schema.json": TokenizedText,
    "run-event.schema.json": RunEvent,
    "selection-candidate.schema.json": SelectionCandidate,
    "selection-manifest.schema.json": SelectionManifest,
    "segment-path-evidence.schema.json": SegmentPathEvidence,
    "similarity-document.schema.json": SimilarityDocument,
    "source-snapshot.schema.json": SourceSnapshot,
    "json-array-conversion.schema.json": JsonArrayConversionReport,
    "source-evidence-input.schema.json": ArgumentEvidenceInput,
    "source-evidence-request.schema.json": SourceEvidenceRequest,
    "source-evidence.schema.json": SourceEvidence,
    "split-leakage.schema.json": SplitLeakage,
    "translation-request.schema.json": TranslationRequest,
    "translation-response.schema.json": TranslationResponse,
    "translation-segment.schema.json": TranslationSegment,
    "translation-segment-result.schema.json": ContractSegmentTranslation,
    "segment-memory-entry.schema.json": SegmentMemoryEntry,
    "memory-lookup-key.schema.json": MemoryLookupKey,
    "live-preflight-decision.schema.json": LivePreflightDecision,
    "live-evaluation-input.schema.json": LiveEvaluationInput,
    "live-evaluation-result.schema.json": LiveEvaluationResult,
    "live-evaluation-run.schema.json": LiveEvaluationRunReport,
    "openai-judge-finding-output.schema.json": JudgeFindingOutput,
    "openai-judge-output.schema.json": JudgeOutput,
    "provider-attempt-record.schema.json": ProviderAttemptRecord,
    "retry-budget-classification.schema.json": RetryBudgetClassification,
    "terminology-input.schema.json": TerminologyInput,
    "terminology-risk.schema.json": TerminologyRisk,
    "wilson-confidence-interval.schema.json": WilsonConfidenceInterval,
    "coverage-summary.schema.json": CoverageSummary,
}


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "schemas" / "0.1.0"
    root.mkdir(parents=True, exist_ok=True)
    for filename, model in MODELS.items():
        schema = model.model_json_schema(mode="validation")
        schema["$schema"] = JSON_SCHEMA_DIALECT
        # Pretty artifacts are reviewable; semantic hashes always use JCS at runtime.
        payload = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        target = root / filename
        if target.exists():
            if target.read_text(encoding="utf-8") != payload:
                raise FileExistsError(
                    f"immutable schema artifact differs; export a new version: {target}"
                )
        else:
            target.write_text(payload, encoding="utf-8", newline="\n")
        canonical_bytes(schema)  # Assert JCS representability as part of export.


if __name__ == "__main__":
    main()
