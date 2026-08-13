from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from toolcall_tr.audit import ConflictCandidate, ExactConflictAudit, conflict_id
from toolcall_tr.cli import app
from toolcall_tr.diagnostics import diagnostic
from toolcall_tr.fingerprints import BehaviorComparison
from toolcall_tr.hashing import canonical_bytes, stable_id
from toolcall_tr.jsonio import iter_jsonl, write_jsonl
from toolcall_tr.pilot import CanonicalQuarantineRecord
from toolcall_tr.review_queue import ReviewQueueInputError, ReviewTask, build_review_tasks

RUNNER = CliRunner()


def _quarantine(index: int) -> CanonicalQuarantineRecord:
    occurrence_id = f"occ_{index:064x}"
    record_body = {
        "schema_version": "canonical-quarantine-0.1.0",
        "source_occurrence_id": occurrence_id,
        "raw_record_sha256": f"sha256:{index:064x}",
        "source_line": index + 1,
        "diagnostic": diagnostic(
            "SOURCE_ADAPTER_INVALID_FIELD",
            "Fixture adapter field is invalid.",
            source_occurrence_id=occurrence_id,
            source_line=index + 1,
            json_pointer="/fixture",
        ).model_dump(mode="json", exclude_none=False),
    }
    return CanonicalQuarantineRecord(
        quarantine_id=stable_id("canonq", record_body),
        source_occurrence_id=occurrence_id,
        raw_record_sha256=f"sha256:{index:064x}",
        source_line=index + 1,
        diagnostic=diagnostic(
            "SOURCE_ADAPTER_INVALID_FIELD",
            "Fixture adapter field is invalid.",
            source_occurrence_id=occurrence_id,
            source_line=index + 1,
            json_pointer="/fixture",
        ),
    )


def _candidate(*, extra_right_member: bool = False) -> ConflictCandidate:
    context = f"sha256:{'a' * 64}"
    left = f"sha256:{'b' * 64}"
    right = f"sha256:{'c' * 64}"
    right_members = [f"ep_{'2' * 64}"]
    if extra_right_member:
        right_members.append(f"ep_{'3' * 64}")
    return ConflictCandidate(
        conflict_id=conflict_id(context, left, right),
        comparison=BehaviorComparison.HARD_CONFLICT,
        presented_context_fingerprint=context,
        left_ordered_behavior_fingerprint=left,
        right_ordered_behavior_fingerprint=right,
        left_owner_episode_id=f"ep_{'1' * 64}",
        right_owner_episode_id=f"ep_{'2' * 64}",
        left_member_episode_ids=[f"ep_{'1' * 64}"],
        right_member_episode_ids=right_members,
        call_multiset_same=False,
    )


def _audit(candidate: ConflictCandidate, salt: str) -> ExactConflictAudit:
    episode_ids = sorted(
        set(candidate.left_member_episode_ids + candidate.right_member_episode_ids)
    )
    body = {
        "schema_version": "exact-conflict-audit-0.1.0",
        "episode_ids": episode_ids,
        "duplicate_groups": [],
        "conflict_candidates": [candidate.model_dump(mode="json", exclude_none=False)],
        "automatic_drop_episode_ids": [],
    }
    return ExactConflictAudit(
        audit_id=stable_id("audit", body),
        episode_ids=episode_ids,
        duplicate_groups=[],
        conflict_candidates=[candidate],
        automatic_drop_episode_ids=[],
    )


def test_review_tasks_are_deterministic_and_non_decisional() -> None:
    first = _quarantine(1)
    second = _quarantine(2)
    audit = _audit(_candidate(), "one")

    tasks = build_review_tasks([second, first, first], [audit, audit])
    reversed_tasks = build_review_tasks([first, second], [audit])

    assert tasks == reversed_tasks
    assert [task.task_id for task in tasks] == sorted(task.task_id for task in tasks)
    assert {task.task_kind for task in tasks} == {
        "canonical_quarantine",
        "conflict_adjudication",
    }
    assert all(task.reviewer_authority == "human" for task in tasks)
    assert all(task.task_id.startswith("reviewtask_") for task in tasks)
    conflict_task = next(task for task in tasks if task.task_kind == "conflict_adjudication")
    assert conflict_task.action == "submit_conflict_adjudication"
    assert conflict_task.audit_ids == [audit.audit_id]


def test_review_queue_rejects_conflicting_evidence_for_one_conflict() -> None:
    first_audit = _audit(_candidate(), "one")
    second_audit = _audit(_candidate(extra_right_member=True), "two")

    with pytest.raises(ReviewQueueInputError, match="conflicting conflict candidate evidence"):
        build_review_tasks([], [first_audit, second_audit])


def test_review_prepare_cli_publishes_an_idempotent_worklist(tmp_path: Path) -> None:
    quarantine_path = tmp_path / "canonical-quarantine.jsonl"
    audit_path = tmp_path / "audit.json"
    output = tmp_path / "review-output"
    quarantine = _quarantine(4)
    audit = _audit(_candidate(), "one")
    write_jsonl(quarantine_path, [quarantine.model_dump(mode="json", exclude_none=False)])
    audit_path.write_bytes(canonical_bytes(audit) + b"\n")

    arguments = [
        "review",
        "prepare",
        "--canonical-quarantine",
        str(quarantine_path),
        "--conflict-audit",
        str(audit_path),
        "--output",
        str(output),
    ]
    first = RUNNER.invoke(app, arguments)
    second = RUNNER.invoke(app, arguments)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    task_path = next(output.glob("human-review-tasks-*.jsonl"))
    tasks = [ReviewTask.model_validate(record, strict=True) for record in iter_jsonl(task_path)]
    assert len(tasks) == 2
    assert {task.task_kind for task in tasks} == {
        "canonical_quarantine",
        "conflict_adjudication",
    }
    assert len(list(output.glob("manifest_*.json"))) == 1
