from __future__ import annotations

import pytest
from pydantic import ValidationError

from toolcall_tr.eval_contract import (
    GoldAcceptance,
    SegmentPathEvidence,
    build_evaluation_report,
    build_evaluation_unit,
    build_human_evaluation_review,
    build_model_verdict,
    build_mqm_finding,
    calculate_wilson_interval,
    decide_gold_acceptance,
)

EPISODE_ID = f"ep_{'1' * 64}"
SEGMENT_ID = f"seg_{'2' * 64}"
SOURCE_HASH = f"sha256:{'3' * 64}"
TARGET_HASH = f"sha256:{'4' * 64}"


def make_unit(*, target_hash: str = TARGET_HASH):
    return build_evaluation_unit(
        episode_id=EPISODE_ID,
        segment_id=SEGMENT_ID,
        path="/conversation/0/content",
        source_text_sha256=SOURCE_HASH,
        target_text_sha256=target_hash,
    )


def make_finding():
    return build_mqm_finding(
        category="accuracy.mistranslation",
        severity="major",
        evidence=SegmentPathEvidence(
            segment_id=SEGMENT_ID,
            path="/conversation/0/content",
            source_excerpt="Book a train on Friday.",
            target_excerpt="Perşembe tren ay\u0131rt.",
        ),
        rationale="The weekday changes the requested action.",
    )


def test_atomic_mqm_finding_is_content_addressed_and_requires_local_evidence() -> None:
    first = make_finding()
    second = make_finding()
    assert first == second
    with pytest.raises(ValidationError):
        SegmentPathEvidence(
            segment_id=SEGMENT_ID,
            path="not-a-pointer",
            source_excerpt="source",
            target_excerpt="target",
        )
    with pytest.raises(ValidationError):
        build_mqm_finding(
            category="not-a-category",  # type: ignore[arg-type]
            severity="major",
            evidence=first.evidence,
            rationale="invalid category is forbidden",
        )


def test_model_verdict_fails_closed_on_invalid_evidence_or_pass_findings() -> None:
    unit = make_unit()
    finding = make_finding()
    with pytest.raises(ValidationError, match="match the evaluation unit"):
        build_model_verdict(
            evaluation_unit=unit,
            evaluator_label="offline-test-evaluator",
            conclusion="fail",
            findings=[
                build_mqm_finding(
                    category="accuracy.mistranslation",
                    severity="major",
                    evidence=SegmentPathEvidence(
                        segment_id=SEGMENT_ID,
                        path="/conversation/1/content",
                        source_excerpt="Friday",
                        target_excerpt="Perşembe",
                    ),
                    rationale="wrong path",
                )
            ],
        )
    with pytest.raises(ValidationError, match="pass verdict"):
        build_model_verdict(
            evaluation_unit=unit,
            evaluator_label="offline-test-evaluator",
            conclusion="pass",
            findings=[finding],
        )
    with pytest.raises(ValidationError, match="unresolved reasons"):
        build_model_verdict(
            evaluation_unit=unit,
            evaluator_label="offline-test-evaluator",
            conclusion="needs_human_review",
        )


def test_wilson_interval_is_standard_library_deterministic_and_validated() -> None:
    interval = calculate_wilson_interval(successes=5, trials=10)
    assert interval.estimate == 0.5
    assert abs(interval.lower - 0.236593) < 1e-6
    assert abs(interval.upper - 0.763407) < 1e-6
    with pytest.raises(ValueError, match="between zero"):
        calculate_wilson_interval(successes=11, trials=10)
    with pytest.raises(ValueError, match="positive"):
        calculate_wilson_interval(successes=0, trials=0)


def test_report_is_deterministic_has_exact_coverage_and_cannot_release_gold() -> None:
    unit_one = make_unit()
    unit_two = make_unit(target_hash=f"sha256:{'5' * 64}")
    verdict = build_model_verdict(
        evaluation_unit=unit_one,
        evaluator_label="offline-test-evaluator",
        conclusion="fail",
        findings=[make_finding()],
    )
    first = build_evaluation_report(
        requested_units=[unit_two, unit_one], model_verdicts=[verdict]
    )
    second = build_evaluation_report(
        requested_units=[unit_one, unit_two], model_verdicts=[verdict]
    )
    assert first == second
    assert first.coverage.requested_units == 2
    assert first.coverage.verdict_units == 1
    assert first.coverage.coverage.estimate == 0.5
    assert first.coverage.uncovered_unit_ids == [unit_two.unit_id]
    assert first.outcomes.fail_count == 1
    assert first.finding_counts[0].category == "accuracy.mistranslation"
    assert first.gold_release_allowed is False


def test_report_rejects_verdict_for_a_nonrequested_or_mutated_unit() -> None:
    requested = make_unit()
    different = make_unit(target_hash=f"sha256:{'5' * 64}")
    verdict = build_model_verdict(
        evaluation_unit=different,
        evaluator_label="offline-test-evaluator",
        conclusion="pass",
    )
    with pytest.raises(ValueError, match="exact requested unit"):
        build_evaluation_report(requested_units=[requested], model_verdicts=[verdict])


def test_model_pass_never_auto_golds_but_explicit_human_acceptance_can() -> None:
    verdict = build_model_verdict(
        evaluation_unit=make_unit(),
        evaluator_label="offline-test-evaluator",
        conclusion="pass",
    )
    pending = decide_gold_acceptance(model_verdict=verdict)
    assert pending.status == "pending_human_review"
    assert pending.gold_eligible is False

    review = build_human_evaluation_review(
        verdict_id=verdict.verdict_id,
        reviewer_id="reviewer-17",
        decision="accept_for_gold",
        reviewed_finding_ids=[],
        rationale="Checked source, target, and tool behavior manually.",
    )
    accepted = decide_gold_acceptance(model_verdict=verdict, human_review=review)
    assert accepted.status == "human_accepted"
    assert accepted.gold_eligible is True
    assert accepted.acceptance_authority == "human"


def test_human_acceptance_must_link_the_verdict_and_all_its_findings() -> None:
    finding = make_finding()
    verdict = build_model_verdict(
        evaluation_unit=make_unit(),
        evaluator_label="offline-test-evaluator",
        conclusion="fail",
        findings=[finding],
    )
    incomplete = build_human_evaluation_review(
        verdict_id=verdict.verdict_id,
        reviewer_id="reviewer-17",
        decision="accept_for_gold",
        reviewed_finding_ids=[],
        rationale="Not enough traceability.",
    )
    with pytest.raises(ValueError, match="every model finding"):
        decide_gold_acceptance(model_verdict=verdict, human_review=incomplete)

    wrong_link = build_human_evaluation_review(
        verdict_id=f"evalverdict_{'f' * 64}",
        reviewer_id="reviewer-17",
        decision="reject",
        reviewed_finding_ids=[],
        rationale="Wrong verdict on purpose.",
    )
    with pytest.raises(ValueError, match="does not refer"):
        decide_gold_acceptance(model_verdict=verdict, human_review=wrong_link)


def test_gold_acceptance_contract_cannot_represent_model_only_gold() -> None:
    with pytest.raises(ValidationError, match="explicit human review"):
        GoldAcceptance(
            verdict_id=f"evalverdict_{'6' * 64}",
            human_review_id=None,
            status="human_accepted",
            gold_eligible=True,
            acceptance_authority=None,
        )
