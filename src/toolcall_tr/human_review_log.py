"""Append-only storage for externally authored human evaluation decisions.

The log accepts a fully formed :class:`HumanEvaluationReview` only.  It never
constructs a decision, derives a reviewer identity, or treats model triage as
human authorization.  The generic event hash chain gives every accepted local
JSONL decision an immutable operational receipt.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic import ValidationError

from toolcall_tr.eval_contract import HumanEvaluationReview
from toolcall_tr.events import EventChainError, EventLog, RunEvent
from toolcall_tr.hashing import JsonValue
from toolcall_tr.models import StrictModel

HUMAN_EVALUATION_REVIEW_STAGE = "human_evaluation_review"
HUMAN_EVALUATION_REVIEW_EVENT_TYPE = "human_evaluation_reviewed"


class HumanEvaluationReviewEntry(StrictModel):
    """A verified chain event and the reviewer decision it preserves."""

    event: RunEvent
    review: HumanEvaluationReview


class HumanEvaluationReviewChainError(EventChainError):
    """Raised when a review log is malformed, ambiguous, or mixed with another log."""


class HumanEvaluationReviewLog:
    """Verify and append one immutable human review per model verdict.

    Corrections intentionally require a new model verdict.  This avoids an
    implicit overwrite or an ambiguous set of reviewer decisions for a single
    verdict at release time.
    """

    def __init__(self, root: Path) -> None:
        self._events = EventLog(root)

    def read_verified(self) -> list[HumanEvaluationReviewEntry]:
        entries: list[HumanEvaluationReviewEntry] = []
        review_ids: set[str] = set()
        verdict_ids: set[str] = set()
        for event in self._events.read_verified():
            if (
                event.stage != HUMAN_EVALUATION_REVIEW_STAGE
                or event.event_type != HUMAN_EVALUATION_REVIEW_EVENT_TYPE
            ):
                raise HumanEvaluationReviewChainError(
                    f"unexpected event kind in human review chain: {event.event_id}"
                )
            try:
                review = HumanEvaluationReview.model_validate(event.details, strict=True)
            except ValidationError as exc:
                raise HumanEvaluationReviewChainError(
                    f"invalid human review details at {event.event_id}: {exc}"
                ) from exc
            if review.review_id in review_ids:
                raise HumanEvaluationReviewChainError(
                    f"duplicate human review ID in chain: {review.review_id}"
                )
            if review.verdict_id in verdict_ids:
                raise HumanEvaluationReviewChainError(
                    f"multiple human reviews for model verdict: {review.verdict_id}"
                )
            review_ids.add(review.review_id)
            verdict_ids.add(review.verdict_id)
            entries.append(HumanEvaluationReviewEntry(event=event, review=review))
        return entries

    def append(
        self,
        *,
        run_id: str,
        review: HumanEvaluationReview,
        parent_manifest_id: str | None = None,
        timestamp_utc: str | None = None,
    ) -> HumanEvaluationReviewEntry:
        """Append a pre-validated reviewer decision without changing its contents."""
        existing = self.read_verified()
        if any(entry.review.review_id == review.review_id for entry in existing):
            raise HumanEvaluationReviewChainError(
                f"human review is already appended: {review.review_id}"
            )
        if any(entry.review.verdict_id == review.verdict_id for entry in existing):
            raise HumanEvaluationReviewChainError(
                f"model verdict already has a human review: {review.verdict_id}"
            )
        details = cast(dict[str, JsonValue], review.model_dump(mode="json", exclude_none=False))
        event = self._events.append(
            run_id=run_id,
            stage=HUMAN_EVALUATION_REVIEW_STAGE,
            event_type=HUMAN_EVALUATION_REVIEW_EVENT_TYPE,
            details=details,
            parent_manifest_id=parent_manifest_id,
            timestamp_utc=timestamp_utc,
        )
        return HumanEvaluationReviewEntry(event=event, review=review)
