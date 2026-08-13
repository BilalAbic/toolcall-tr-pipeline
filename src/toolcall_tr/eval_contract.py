"""Offline, fail-closed quality-evaluation contracts.

This module defines auditable inputs and summaries for MQM-style evaluation.  It
does not contain a provider client, environment lookup, retry loop, or network
operation.  A model verdict is only triage: it can never make an item gold.
Only an explicit, content-addressed human acceptance can make that transition.
"""

from __future__ import annotations

from collections import Counter
from math import isclose, sqrt
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from toolcall_tr.hashing import stable_id
from toolcall_tr.models import EpisodeId, NonEmptyStr, Sha256, StrictModel
from toolcall_tr.translation_contract import JsonPointer, SegmentId

type MqmCategory = Literal[
    "accuracy.mistranslation",
    "accuracy.omission",
    "accuracy.addition",
    "accuracy.untranslated",
    "terminology",
    "fluency.grammar",
    "fluency.spelling",
    "fluency.punctuation",
    "locale_convention",
    "style.register",
    "tool_semantics",
    "protected_content",
    "research_provenance",
]
type MqmSeverity = Literal["minor", "major", "critical"]
type ModelConclusion = Literal["pass", "needs_human_review", "fail"]

EvaluationUnitId = Annotated[str, StringConstraints(pattern=r"^evalunit_[0-9a-f]{64}$")]
MqmFindingId = Annotated[str, StringConstraints(pattern=r"^mqm_[0-9a-f]{64}$")]
ModelVerdictId = Annotated[str, StringConstraints(pattern=r"^evalverdict_[0-9a-f]{64}$")]
HumanReviewId = Annotated[str, StringConstraints(pattern=r"^humanreview_[0-9a-f]{64}$")]
EvaluationReportId = Annotated[str, StringConstraints(pattern=r"^evalreport_[0-9a-f]{64}$")]

WILSON_95_Z = 1.959963984540054


class EvaluationUnit(StrictModel):
    """One immutable source/target leaf that may be evaluated once per run."""

    schema_version: Literal["evaluation-unit-0.1.0"] = "evaluation-unit-0.1.0"
    unit_id: EvaluationUnitId
    episode_id: EpisodeId
    segment_id: SegmentId
    path: JsonPointer
    source_text_sha256: Sha256
    target_text_sha256: Sha256

    @model_validator(mode="after")
    def validate_content_id(self) -> EvaluationUnit:
        body = self.model_dump(mode="json", exclude={"unit_id"})
        if self.unit_id != stable_id("evalunit", body):
            raise ValueError("evaluation unit ID does not match deterministic content")
        return self


class SegmentPathEvidence(StrictModel):
    """Localized, bidirectional evidence for exactly one evaluation leaf."""

    segment_id: SegmentId
    path: JsonPointer
    source_excerpt: NonEmptyStr
    target_excerpt: NonEmptyStr


class MqmFinding(StrictModel):
    """One atomic MQM-style issue; compound findings must be split by callers."""

    schema_version: Literal["mqm-finding-0.1.0"] = "mqm-finding-0.1.0"
    finding_id: MqmFindingId
    category: MqmCategory
    severity: MqmSeverity
    evidence: SegmentPathEvidence
    rationale: NonEmptyStr

    @model_validator(mode="after")
    def validate_content_id(self) -> MqmFinding:
        body = self.model_dump(mode="json", exclude={"finding_id"})
        if self.finding_id != stable_id("mqm", body):
            raise ValueError("MQM finding ID does not match deterministic content")
        return self


class ModelEvaluationVerdict(StrictModel):
    """Provider-shaped triage record, intentionally unable to express acceptance."""

    schema_version: Literal["model-evaluation-verdict-0.1.0"] = "model-evaluation-verdict-0.1.0"
    verdict_id: ModelVerdictId
    evaluation_unit: EvaluationUnit
    evaluator_label: NonEmptyStr
    conclusion: ModelConclusion
    findings: list[MqmFinding]
    unresolved_reasons: list[NonEmptyStr]

    @model_validator(mode="after")
    def validate_verdict(self) -> ModelEvaluationVerdict:
        finding_ids = [finding.finding_id for finding in self.findings]
        if finding_ids != sorted(set(finding_ids)):
            raise ValueError("model finding IDs must be unique and sorted")
        if any(
            finding.evidence.segment_id != self.evaluation_unit.segment_id
            or finding.evidence.path != self.evaluation_unit.path
            for finding in self.findings
        ):
            raise ValueError("finding evidence must match the evaluation unit segment and path")

        if self.unresolved_reasons != sorted(set(self.unresolved_reasons)):
            raise ValueError("unresolved reasons must be unique and sorted")
        if self.conclusion == "pass" and (self.findings or self.unresolved_reasons):
            raise ValueError("pass verdict cannot contain findings or unresolved reasons")
        if self.conclusion == "fail" and (not self.findings or self.unresolved_reasons):
            raise ValueError("fail verdict requires findings and cannot be unresolved")
        if self.conclusion == "needs_human_review" and not self.unresolved_reasons:
            raise ValueError("needs_human_review verdict requires unresolved reasons")

        body = self.model_dump(mode="json", exclude={"verdict_id"})
        if self.verdict_id != stable_id("evalverdict", body):
            raise ValueError("model verdict ID does not match deterministic content")
        return self


class WilsonConfidenceInterval(StrictModel):
    """A standard-library Wilson 95% confidence interval for a binomial rate."""

    method: Literal["wilson-95"] = "wilson-95"
    successes: Annotated[int, Field(ge=0)]
    trials: Annotated[int, Field(gt=0)]
    estimate: Annotated[float, Field(ge=0.0, le=1.0)]
    lower: Annotated[float, Field(ge=0.0, le=1.0)]
    upper: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_interval(self) -> WilsonConfidenceInterval:
        if self.successes > self.trials:
            raise ValueError("Wilson successes cannot exceed trials")
        expected = _wilson_bounds(self.successes, self.trials)
        if not (
            isclose(self.estimate, self.successes / self.trials, rel_tol=0.0, abs_tol=1e-15)
            and isclose(self.lower, expected[0], rel_tol=0.0, abs_tol=1e-15)
            and isclose(self.upper, expected[1], rel_tol=0.0, abs_tol=1e-15)
        ):
            raise ValueError("Wilson interval values must match the standard calculation")
        return self


class CoverageSummary(StrictModel):
    requested_units: Annotated[int, Field(gt=0)]
    verdict_units: Annotated[int, Field(ge=0)]
    uncovered_unit_ids: list[EvaluationUnitId]
    coverage: WilsonConfidenceInterval

    @model_validator(mode="after")
    def validate_coverage(self) -> CoverageSummary:
        if self.verdict_units > self.requested_units:
            raise ValueError("verdict unit count cannot exceed requested units")
        if self.uncovered_unit_ids != sorted(set(self.uncovered_unit_ids)):
            raise ValueError("uncovered unit IDs must be unique and sorted")
        if len(self.uncovered_unit_ids) != self.requested_units - self.verdict_units:
            raise ValueError("uncovered unit count does not match coverage counts")
        if (
            self.coverage.successes != self.verdict_units
            or self.coverage.trials != self.requested_units
        ):
            raise ValueError("coverage interval counts do not match coverage summary")
        return self


class OutcomeSummary(StrictModel):
    pass_count: Annotated[int, Field(ge=0)]
    needs_human_review_count: Annotated[int, Field(ge=0)]
    fail_count: Annotated[int, Field(ge=0)]


class FindingCount(StrictModel):
    category: MqmCategory
    severity: MqmSeverity
    count: Annotated[int, Field(gt=0)]


class EvaluationReport(StrictModel):
    """Deterministic report over a declared leaf-level evaluation sample."""

    schema_version: Literal["evaluation-report-0.1.0"] = "evaluation-report-0.1.0"
    report_id: EvaluationReportId
    requested_units: Annotated[list[EvaluationUnit], Field(min_length=1)]
    model_verdicts: list[ModelEvaluationVerdict]
    coverage: CoverageSummary
    outcomes: OutcomeSummary
    finding_counts: list[FindingCount]
    gold_release_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> EvaluationReport:
        requested_ids = [unit.unit_id for unit in self.requested_units]
        if requested_ids != sorted(set(requested_ids)):
            raise ValueError("requested unit IDs must be unique and sorted")
        verdict_ids = [verdict.verdict_id for verdict in self.model_verdicts]
        if len(verdict_ids) != len(set(verdict_ids)):
            raise ValueError("model verdict IDs must be unique")
        verdict_unit_ids = [verdict.evaluation_unit.unit_id for verdict in self.model_verdicts]
        if verdict_unit_ids != sorted(set(verdict_unit_ids)):
            raise ValueError("model verdict units must be unique and sorted")

        requested_by_id = {unit.unit_id: unit for unit in self.requested_units}
        for verdict in self.model_verdicts:
            expected = requested_by_id.get(verdict.evaluation_unit.unit_id)
            if expected != verdict.evaluation_unit:
                raise ValueError("model verdict must refer to one exact requested unit")

        expected_uncovered = sorted(set(requested_ids) - set(verdict_unit_ids))
        if self.coverage.uncovered_unit_ids != expected_uncovered:
            raise ValueError("coverage uncovered IDs do not match requested verdict coverage")
        if (
            self.coverage.requested_units != len(self.requested_units)
            or self.coverage.verdict_units != len(self.model_verdicts)
        ):
            raise ValueError("coverage counts do not match report memberships")

        observed_outcomes = Counter(verdict.conclusion for verdict in self.model_verdicts)
        if self.outcomes != OutcomeSummary(
            pass_count=observed_outcomes["pass"],
            needs_human_review_count=observed_outcomes["needs_human_review"],
            fail_count=observed_outcomes["fail"],
        ):
            raise ValueError("outcome counts do not match model verdicts")

        observed_finding_counts: Counter[tuple[MqmCategory, MqmSeverity]] = Counter(
            (finding.category, finding.severity)
            for verdict in self.model_verdicts
            for finding in verdict.findings
        )
        expected_finding_counts = [
            FindingCount(category=category, severity=severity, count=count)
            for (category, severity), count in sorted(observed_finding_counts.items())
        ]
        if self.finding_counts != expected_finding_counts:
            raise ValueError("finding counts do not match model verdicts")

        body = self.model_dump(mode="json", exclude={"report_id"})
        if self.report_id != stable_id("evalreport", body):
            raise ValueError("evaluation report ID does not match deterministic content")
        return self


class HumanEvaluationReview(StrictModel):
    """Explicit human sign-off record; this is the sole gold-eligibility authority."""

    schema_version: Literal["human-evaluation-review-0.1.0"] = "human-evaluation-review-0.1.0"
    review_id: HumanReviewId
    verdict_id: ModelVerdictId
    reviewer_authority: Literal["human"]
    reviewer_id: NonEmptyStr
    decision: Literal["accept_for_gold", "reject"]
    reviewed_finding_ids: list[MqmFindingId]
    rationale: NonEmptyStr

    @model_validator(mode="after")
    def validate_review(self) -> HumanEvaluationReview:
        if self.reviewed_finding_ids != sorted(set(self.reviewed_finding_ids)):
            raise ValueError("reviewed finding IDs must be unique and sorted")
        body = self.model_dump(mode="json", exclude={"review_id"})
        if self.review_id != stable_id("humanreview", body):
            raise ValueError("human review ID does not match deterministic content")
        return self


class GoldAcceptance(StrictModel):
    """A policy result.  A model-only verdict is always pending and non-gold."""

    schema_version: Literal["gold-acceptance-0.1.0"] = "gold-acceptance-0.1.0"
    verdict_id: ModelVerdictId
    human_review_id: HumanReviewId | None
    status: Literal["pending_human_review", "human_rejected", "human_accepted"]
    gold_eligible: bool
    acceptance_authority: Literal["human"] | None

    @model_validator(mode="after")
    def validate_fail_closed_state(self) -> GoldAcceptance:
        accepted = self.status == "human_accepted"
        if accepted:
            if (
                not self.gold_eligible
                or self.human_review_id is None
                or self.acceptance_authority != "human"
            ):
                raise ValueError("gold acceptance requires an explicit human review")
        elif self.gold_eligible or self.acceptance_authority is not None:
            raise ValueError("only a human_accepted result can be gold eligible")
        elif self.status == "pending_human_review" and self.human_review_id is not None:
            raise ValueError("pending acceptance cannot contain a human review")
        elif self.status == "human_rejected" and self.human_review_id is None:
            raise ValueError("human rejection requires an explicit human review")
        return self


def build_evaluation_unit(
    *,
    episode_id: str,
    segment_id: str,
    path: str,
    source_text_sha256: str,
    target_text_sha256: str,
) -> EvaluationUnit:
    """Build a content-addressed evaluation unit without reading its source text."""
    body = {
        "schema_version": "evaluation-unit-0.1.0",
        "episode_id": episode_id,
        "segment_id": segment_id,
        "path": path,
        "source_text_sha256": source_text_sha256,
        "target_text_sha256": target_text_sha256,
    }
    return EvaluationUnit(
        unit_id=stable_id("evalunit", body),
        episode_id=episode_id,
        segment_id=segment_id,
        path=path,
        source_text_sha256=source_text_sha256,
        target_text_sha256=target_text_sha256,
    )


def build_mqm_finding(
    *,
    category: MqmCategory,
    severity: MqmSeverity,
    evidence: SegmentPathEvidence,
    rationale: str,
) -> MqmFinding:
    """Build one content-addressed atomic finding."""
    body = {
        "schema_version": "mqm-finding-0.1.0",
        "category": category,
        "severity": severity,
        "evidence": evidence.model_dump(mode="json"),
        "rationale": rationale,
    }
    return MqmFinding(
        finding_id=stable_id("mqm", body),
        category=category,
        severity=severity,
        evidence=evidence,
        rationale=rationale,
    )


def build_model_verdict(
    *,
    evaluation_unit: EvaluationUnit,
    evaluator_label: str,
    conclusion: ModelConclusion,
    findings: list[MqmFinding] | None = None,
    unresolved_reasons: list[str] | None = None,
) -> ModelEvaluationVerdict:
    """Build a model-triage record.  It deliberately has no acceptance state."""
    ordered_findings = sorted(findings or [], key=lambda item: item.finding_id)
    ordered_reasons = sorted(set(unresolved_reasons or []))
    body = {
        "schema_version": "model-evaluation-verdict-0.1.0",
        "evaluation_unit": evaluation_unit.model_dump(mode="json"),
        "evaluator_label": evaluator_label,
        "conclusion": conclusion,
        "findings": [finding.model_dump(mode="json") for finding in ordered_findings],
        "unresolved_reasons": ordered_reasons,
    }
    return ModelEvaluationVerdict(
        verdict_id=stable_id("evalverdict", body),
        evaluation_unit=evaluation_unit,
        evaluator_label=evaluator_label,
        conclusion=conclusion,
        findings=ordered_findings,
        unresolved_reasons=ordered_reasons,
    )


def calculate_wilson_interval(*, successes: int, trials: int) -> WilsonConfidenceInterval:
    """Calculate a deterministic 95% Wilson interval using only ``math``."""
    if trials <= 0:
        raise ValueError("Wilson trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("Wilson successes must be between zero and trials")
    lower, upper = _wilson_bounds(successes, trials)
    return WilsonConfidenceInterval(
        successes=successes,
        trials=trials,
        estimate=successes / trials,
        lower=lower,
        upper=upper,
    )


def build_evaluation_report(
    *,
    requested_units: list[EvaluationUnit],
    model_verdicts: list[ModelEvaluationVerdict],
) -> EvaluationReport:
    """Compute coverage and finding summaries from declared IDs, fail-closed by design."""
    unit_ids = [unit.unit_id for unit in requested_units]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("requested evaluation unit IDs must be unique")
    verdict_ids = [verdict.verdict_id for verdict in model_verdicts]
    if len(verdict_ids) != len(set(verdict_ids)):
        raise ValueError("model verdict IDs must be unique")
    verdict_unit_ids = [verdict.evaluation_unit.unit_id for verdict in model_verdicts]
    if len(verdict_unit_ids) != len(set(verdict_unit_ids)):
        raise ValueError("model verdict units must be unique")

    ordered_units = sorted(requested_units, key=lambda unit: unit.unit_id)
    requested_by_id = {unit.unit_id: unit for unit in ordered_units}
    for verdict in model_verdicts:
        if requested_by_id.get(verdict.evaluation_unit.unit_id) != verdict.evaluation_unit:
            raise ValueError("model verdict must refer to one exact requested unit")
    ordered_verdicts = sorted(model_verdicts, key=lambda verdict: verdict.evaluation_unit.unit_id)

    uncovered = sorted(set(requested_by_id) - set(verdict_unit_ids))
    coverage = CoverageSummary(
        requested_units=len(ordered_units),
        verdict_units=len(ordered_verdicts),
        uncovered_unit_ids=uncovered,
        coverage=calculate_wilson_interval(
            successes=len(ordered_verdicts), trials=len(ordered_units)
        ),
    )
    outcome_counts = Counter(verdict.conclusion for verdict in ordered_verdicts)
    outcomes = OutcomeSummary(
        pass_count=outcome_counts["pass"],
        needs_human_review_count=outcome_counts["needs_human_review"],
        fail_count=outcome_counts["fail"],
    )
    observed_finding_counts: Counter[tuple[MqmCategory, MqmSeverity]] = Counter(
        (finding.category, finding.severity)
        for verdict in ordered_verdicts
        for finding in verdict.findings
    )
    finding_counts = [
        FindingCount(category=category, severity=severity, count=count)
        for (category, severity), count in sorted(observed_finding_counts.items())
    ]
    body = {
        "schema_version": "evaluation-report-0.1.0",
        "requested_units": [unit.model_dump(mode="json") for unit in ordered_units],
        "model_verdicts": [verdict.model_dump(mode="json") for verdict in ordered_verdicts],
        "coverage": coverage.model_dump(mode="json"),
        "outcomes": outcomes.model_dump(mode="json"),
        "finding_counts": [item.model_dump(mode="json") for item in finding_counts],
        "gold_release_allowed": False,
    }
    return EvaluationReport(
        report_id=stable_id("evalreport", body),
        requested_units=ordered_units,
        model_verdicts=ordered_verdicts,
        coverage=coverage,
        outcomes=outcomes,
        finding_counts=finding_counts,
    )


def build_human_evaluation_review(
    *,
    verdict_id: str,
    reviewer_id: str,
    decision: Literal["accept_for_gold", "reject"],
    reviewed_finding_ids: list[str],
    rationale: str,
) -> HumanEvaluationReview:
    """Build an explicit human review record; no model identity is accepted here."""
    ordered_finding_ids = sorted(set(reviewed_finding_ids))
    body = {
        "schema_version": "human-evaluation-review-0.1.0",
        "verdict_id": verdict_id,
        "reviewer_authority": "human",
        "reviewer_id": reviewer_id,
        "decision": decision,
        "reviewed_finding_ids": ordered_finding_ids,
        "rationale": rationale,
    }
    return HumanEvaluationReview(
        review_id=stable_id("humanreview", body),
        verdict_id=verdict_id,
        reviewer_authority="human",
        reviewer_id=reviewer_id,
        decision=decision,
        reviewed_finding_ids=ordered_finding_ids,
        rationale=rationale,
    )


def decide_gold_acceptance(
    *,
    model_verdict: ModelEvaluationVerdict,
    human_review: HumanEvaluationReview | None = None,
) -> GoldAcceptance:
    """Apply the sole acceptance policy: no human review means no gold eligibility."""
    if human_review is None:
        return GoldAcceptance(
            verdict_id=model_verdict.verdict_id,
            human_review_id=None,
            status="pending_human_review",
            gold_eligible=False,
            acceptance_authority=None,
        )
    if human_review.verdict_id != model_verdict.verdict_id:
        raise ValueError("human review does not refer to the supplied model verdict")

    model_finding_ids = {finding.finding_id for finding in model_verdict.findings}
    reviewed_finding_ids = set(human_review.reviewed_finding_ids)
    if not reviewed_finding_ids.issubset(model_finding_ids):
        raise ValueError("human review refers to a finding absent from the model verdict")
    if (
        human_review.decision == "accept_for_gold"
        and reviewed_finding_ids != model_finding_ids
    ):
        raise ValueError("human gold acceptance must explicitly review every model finding")

    if human_review.decision == "accept_for_gold":
        return GoldAcceptance(
            verdict_id=model_verdict.verdict_id,
            human_review_id=human_review.review_id,
            status="human_accepted",
            gold_eligible=True,
            acceptance_authority="human",
        )
    return GoldAcceptance(
        verdict_id=model_verdict.verdict_id,
        human_review_id=human_review.review_id,
        status="human_rejected",
        gold_eligible=False,
        acceptance_authority=None,
    )


def _wilson_bounds(successes: int, trials: int) -> tuple[float, float]:
    proportion = successes / trials
    z_squared = WILSON_95_Z**2
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        WILSON_95_Z
        * sqrt((proportion * (1.0 - proportion) + z_squared / (4.0 * trials)) / trials)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
