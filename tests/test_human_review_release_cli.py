from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from toolcall_tr.adjudication import ConflictAdjudication, ConflictAdjudicationLog
from toolcall_tr.cli import app
from toolcall_tr.eval_contract import (
    HumanEvaluationReview,
    ModelEvaluationVerdict,
    build_evaluation_unit,
    build_human_evaluation_review,
    build_model_verdict,
)
from toolcall_tr.human_review_log import HumanEvaluationReviewLog
from toolcall_tr.jsonio import write_jsonl
from toolcall_tr.release_contract import ReleaseGoldMember, read_release_manifest

RUNNER = CliRunner()


def _accepted_review_evidence(
    character: str,
) -> tuple[ModelEvaluationVerdict, HumanEvaluationReview, ReleaseGoldMember]:
    episode_id = f"ep_{character * 64}"
    unit = build_evaluation_unit(
        episode_id=episode_id,
        segment_id=f"seg_{character * 64}",
        path="/conversation/0/content",
        source_text_sha256=f"sha256:{'a' * 64}",
        target_text_sha256=f"sha256:{'b' * 64}",
    )
    verdict = build_model_verdict(
        evaluation_unit=unit,
        evaluator_label="offline-cli-test",
        conclusion="pass",
    )
    review = build_human_evaluation_review(
        verdict_id=verdict.verdict_id,
        reviewer_id="reviewer-17",
        decision="accept_for_gold",
        reviewed_finding_ids=[],
        rationale="Reviewer checked this fixture manually before release.",
    )
    return (
        verdict,
        review,
        ReleaseGoldMember(
            episode_id=episode_id,
            verdict_id=verdict.verdict_id,
            human_review_id=review.review_id,
        ),
    )


def test_evaluation_review_cli_appends_only_a_single_valid_human_decision(tmp_path: Path) -> None:
    verdict, review, _ = _accepted_review_evidence("1")
    decision_jsonl = tmp_path / "review.jsonl"
    verdicts_jsonl = tmp_path / "verdicts.jsonl"
    events_root = tmp_path / "events"
    write_jsonl(decision_jsonl, [review.model_dump(mode="json")])
    write_jsonl(verdicts_jsonl, [verdict.model_dump(mode="json")])

    result = RUNNER.invoke(
        app,
        [
            "review",
            "submit-evaluation",
            str(decision_jsonl),
            "--run-id",
            "fixture-human-review",
            "--model-verdicts-jsonl",
            str(verdicts_jsonl),
            "--events-root",
            str(events_root),
            "--timestamp-utc",
            "2026-08-13T00:00:00.000000Z",
        ],
    )
    assert result.exit_code == 0, result.output
    entries = HumanEvaluationReviewLog(events_root).read_verified()
    assert len(entries) == 1
    assert entries[0].review == review
    assert entries[0].review.verdict_id == verdict.verdict_id

    duplicate = RUNNER.invoke(
        app,
        [
            "review",
            "submit-evaluation",
            str(decision_jsonl),
            "--run-id",
            "fixture-human-review",
            "--model-verdicts-jsonl",
            str(verdicts_jsonl),
            "--events-root",
            str(events_root),
        ],
    )
    assert duplicate.exit_code == 1
    assert len(HumanEvaluationReviewLog(events_root).read_verified()) == 1


def test_review_cli_rejects_multiple_jsonl_decisions_before_creating_an_event(
    tmp_path: Path,
) -> None:
    verdict, review, _ = _accepted_review_evidence("2")
    decision_jsonl = tmp_path / "multiple-reviews.jsonl"
    verdicts_jsonl = tmp_path / "verdicts.jsonl"
    events_root = tmp_path / "events"
    write_jsonl(
        decision_jsonl,
        [review.model_dump(mode="json"), review.model_dump(mode="json")],
    )
    write_jsonl(verdicts_jsonl, [verdict.model_dump(mode="json")])

    result = RUNNER.invoke(
        app,
        [
            "review",
            "submit-evaluation",
            str(decision_jsonl),
            "--run-id",
            "fixture-human-review",
            "--model-verdicts-jsonl",
            str(verdicts_jsonl),
            "--events-root",
            str(events_root),
        ],
    )
    assert result.exit_code == 1
    assert not events_root.exists()


def test_review_cli_requires_the_referenced_local_model_verdict(tmp_path: Path) -> None:
    _, review, _ = _accepted_review_evidence("4")
    unrelated_verdict, _, _ = _accepted_review_evidence("5")
    decision_jsonl = tmp_path / "review.jsonl"
    verdicts_jsonl = tmp_path / "unrelated-verdicts.jsonl"
    events_root = tmp_path / "events"
    write_jsonl(decision_jsonl, [review.model_dump(mode="json")])
    write_jsonl(verdicts_jsonl, [unrelated_verdict.model_dump(mode="json")])

    result = RUNNER.invoke(
        app,
        [
            "review",
            "submit-evaluation",
            str(decision_jsonl),
            "--run-id",
            "fixture-human-review",
            "--model-verdicts-jsonl",
            str(verdicts_jsonl),
            "--events-root",
            str(events_root),
        ],
    )
    assert result.exit_code == 1
    assert not events_root.exists()


def test_conflict_review_cli_appends_only_reviewer_provided_decision(tmp_path: Path) -> None:
    decision = ConflictAdjudication(
        conflict_id=f"conf_{'a' * 64}",
        left_episode_id=f"ep_{'1' * 64}",
        right_episode_id=f"ep_{'2' * 64}",
        decision="defer",
        reviewer_id="reviewer-17",
        reviewer_authority="human",
        rubric_version="fixture-rubric-0.1.0",
        rationale="The fixture lacks enough evidence for a final choice.",
        supersedes_event_id=None,
    )
    decision_jsonl = tmp_path / "conflict.jsonl"
    events_root = tmp_path / "conflict-events"
    write_jsonl(decision_jsonl, [decision.model_dump(mode="json")])

    result = RUNNER.invoke(
        app,
        [
            "review",
            "submit-conflict",
            str(decision_jsonl),
            "--run-id",
            "fixture-conflict-review",
            "--events-root",
            str(events_root),
            "--timestamp-utc",
            "2026-08-13T00:00:00.000000Z",
        ],
    )
    assert result.exit_code == 0, result.output
    assert ConflictAdjudicationLog(events_root).current()[0].adjudication == decision


def test_release_cli_requires_logged_human_gold_review_then_revalidates(tmp_path: Path) -> None:
    verdict, review, member = _accepted_review_evidence("3")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    write_jsonl(dataset_root / "gold.jsonl", [{"episode_id": member.episode_id}])
    members_jsonl = tmp_path / "members.jsonl"
    verdicts_jsonl = tmp_path / "verdicts.jsonl"
    decision_jsonl = tmp_path / "review.jsonl"
    events_root = tmp_path / "events"
    manifest_jsonl = tmp_path / "manifest.jsonl"
    write_jsonl(members_jsonl, [member.model_dump(mode="json")])
    write_jsonl(verdicts_jsonl, [verdict.model_dump(mode="json")])
    write_jsonl(decision_jsonl, [review.model_dump(mode="json")])

    blocked = RUNNER.invoke(
        app,
        [
            "release",
            "build",
            str(dataset_root),
            str(members_jsonl),
            str(verdicts_jsonl),
            "--file",
            "gold.jsonl",
            "--review-events-root",
            str(events_root),
            "--output",
            str(manifest_jsonl),
        ],
    )
    assert blocked.exit_code == 1
    assert not manifest_jsonl.exists()

    submitted = RUNNER.invoke(
        app,
        [
            "review",
            "submit-evaluation",
            str(decision_jsonl),
            "--run-id",
            "fixture-gold-review",
            "--model-verdicts-jsonl",
            str(verdicts_jsonl),
            "--events-root",
            str(events_root),
            "--timestamp-utc",
            "2026-08-13T00:00:00.000000Z",
        ],
    )
    assert submitted.exit_code == 0, submitted.output

    built = RUNNER.invoke(
        app,
        [
            "release",
            "build",
            str(dataset_root),
            str(members_jsonl),
            str(verdicts_jsonl),
            "--file",
            "gold.jsonl",
            "--review-events-root",
            str(events_root),
            "--output",
            str(manifest_jsonl),
        ],
    )
    assert built.exit_code == 0, built.output
    manifest = read_release_manifest(manifest_jsonl)
    assert manifest.gold_members == [member]

    validated = RUNNER.invoke(
        app,
        [
            "release",
            "validate",
            str(dataset_root),
            str(manifest_jsonl),
            str(verdicts_jsonl),
            "--review-events-root",
            str(events_root),
        ],
    )
    assert validated.exit_code == 0, validated.output
