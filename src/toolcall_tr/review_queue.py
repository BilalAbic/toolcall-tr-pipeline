"""Deterministic, non-decisional worklists for human review gates.

The queue turns existing canonical quarantine records and exact conflict audits
into stable, content-addressed task rows. It never changes a source record,
chooses a disposition, or writes a reviewer decision. Conflict tasks point to
the existing ``review submit-conflict`` flow; quarantine tasks are explicit
remediation work items until a human-approved adapter or policy change creates
a new immutable pilot.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.audit import AuditId, ConflictCandidate, ExactConflictAudit
from toolcall_tr.hashing import stable_id
from toolcall_tr.models import StrictModel
from toolcall_tr.pilot import CanonicalQuarantineRecord

ReviewTaskId = Annotated[str, Field(pattern=r"^reviewtask_[0-9a-f]{64}$")]
ReviewTaskKind = Literal["canonical_quarantine", "conflict_adjudication"]
ReviewAction = Literal["remediate_canonical_quarantine", "submit_conflict_adjudication"]


class ReviewQueueInputError(ValueError):
    """Raised when a review queue would hide contradictory input evidence."""


class ReviewTask(StrictModel):
    """One open human-only task derived from immutable upstream evidence."""

    schema_version: Literal["review-task-0.1.0"] = "review-task-0.1.0"
    task_id: ReviewTaskId
    task_kind: ReviewTaskKind
    reviewer_authority: Literal["human"] = "human"
    action: ReviewAction
    canonical_quarantine: CanonicalQuarantineRecord | None
    conflict_candidate: ConflictCandidate | None
    audit_ids: list[AuditId]

    @model_validator(mode="after")
    def validate_task(self) -> ReviewTask:
        is_quarantine = self.task_kind == "canonical_quarantine"
        if is_quarantine:
            if (
                self.action != "remediate_canonical_quarantine"
                or self.canonical_quarantine is None
                or self.conflict_candidate is not None
                or self.audit_ids
            ):
                raise ValueError(
                    "canonical quarantine task must contain only one quarantine record"
                )
        elif (
            self.action != "submit_conflict_adjudication"
            or self.canonical_quarantine is not None
            or self.conflict_candidate is None
            or self.audit_ids != sorted(set(self.audit_ids))
            or not self.audit_ids
        ):
            raise ValueError("conflict task must contain one candidate and sorted audit IDs")
        body = self.model_dump(mode="json", exclude={"task_id"})
        if self.task_id != stable_id("reviewtask", body):
            raise ValueError("review task ID does not match deterministic task body")
        return self


def _quarantine_task(record: CanonicalQuarantineRecord) -> ReviewTask:
    body: dict[str, object] = {
        "schema_version": "review-task-0.1.0",
        "task_kind": "canonical_quarantine",
        "reviewer_authority": "human",
        "action": "remediate_canonical_quarantine",
        "canonical_quarantine": record.model_dump(mode="json", exclude_none=False),
        "conflict_candidate": None,
        "audit_ids": [],
    }
    return ReviewTask(
        task_id=stable_id("reviewtask", body),
        task_kind="canonical_quarantine",
        action="remediate_canonical_quarantine",
        canonical_quarantine=record,
        conflict_candidate=None,
        audit_ids=[],
    )


def _conflict_task(candidate: ConflictCandidate, audit_ids: list[str]) -> ReviewTask:
    sorted_audit_ids = sorted(set(audit_ids))
    body: dict[str, object] = {
        "schema_version": "review-task-0.1.0",
        "task_kind": "conflict_adjudication",
        "reviewer_authority": "human",
        "action": "submit_conflict_adjudication",
        "canonical_quarantine": None,
        "conflict_candidate": candidate.model_dump(mode="json", exclude_none=False),
        "audit_ids": sorted_audit_ids,
    }
    return ReviewTask(
        task_id=stable_id("reviewtask", body),
        task_kind="conflict_adjudication",
        action="submit_conflict_adjudication",
        canonical_quarantine=None,
        conflict_candidate=candidate,
        audit_ids=sorted_audit_ids,
    )


def build_review_tasks(
    quarantines: Iterable[CanonicalQuarantineRecord], audits: Iterable[ExactConflictAudit]
) -> list[ReviewTask]:
    """Build an ordered, deduplicated worklist without choosing any outcome.

    Identical repeated evidence is collapsed. The same deterministic identity
    with divergent content is rejected, because choosing one would discard
    review-relevant evidence.
    """
    quarantine_by_id: dict[str, CanonicalQuarantineRecord] = {}
    for record in quarantines:
        existing = quarantine_by_id.get(record.quarantine_id)
        if existing is not None and existing != record:
            raise ReviewQueueInputError(
                f"conflicting canonical quarantine evidence: {record.quarantine_id}"
            )
        quarantine_by_id[record.quarantine_id] = record

    candidate_by_id: dict[str, ConflictCandidate] = {}
    audit_ids_by_conflict_id: dict[str, list[str]] = {}
    for audit in audits:
        for candidate in audit.conflict_candidates:
            existing = candidate_by_id.get(candidate.conflict_id)
            if existing is not None and existing != candidate:
                raise ReviewQueueInputError(
                    f"conflicting conflict candidate evidence: {candidate.conflict_id}"
                )
            candidate_by_id[candidate.conflict_id] = candidate
            audit_ids_by_conflict_id.setdefault(candidate.conflict_id, []).append(audit.audit_id)

    tasks = [
        _quarantine_task(quarantine_by_id[quarantine_id])
        for quarantine_id in sorted(quarantine_by_id)
    ]
    tasks.extend(
        _conflict_task(
            candidate_by_id[conflict_id],
            audit_ids_by_conflict_id[conflict_id],
        )
        for conflict_id in sorted(candidate_by_id)
    )
    return sorted(tasks, key=lambda task: task.task_id)
