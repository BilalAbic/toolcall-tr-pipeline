"""Strict append-only human adjudication over exact conflict candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, ValidationError, model_validator

from toolcall_tr.audit import ConflictId
from toolcall_tr.events import EventChainError, EventLog, RunEvent
from toolcall_tr.hashing import JsonValue
from toolcall_tr.models import EpisodeId, NonEmptyStr, StrictModel

ADJUDICATION_STAGE = "conflict_adjudication"
ADJUDICATION_EVENT_TYPE = "conflict_adjudicated"

type AdjudicationDecision = Literal[
    "keep_left",
    "keep_right",
    "keep_both_context_insufficient",
    "source_error",
    "policy_variant",
    "defer",
]
EventId = Annotated[str, Field(pattern=r"^evt_[0-9a-f]{64}$")]


class ConflictAdjudication(StrictModel):
    """Human-authored decision payload stored inside a generic chained run event."""

    schema_version: Literal["conflict-adjudication-0.1.0"] = "conflict-adjudication-0.1.0"
    conflict_id: ConflictId
    left_episode_id: EpisodeId
    right_episode_id: EpisodeId
    decision: AdjudicationDecision
    reviewer_id: NonEmptyStr
    reviewer_authority: Literal["human"]
    rubric_version: NonEmptyStr
    rationale: NonEmptyStr
    supersedes_event_id: EventId | None

    @model_validator(mode="after")
    def validate_pair(self) -> ConflictAdjudication:
        if self.left_episode_id == self.right_episode_id:
            raise ValueError("an adjudication must reference two distinct episodes")
        return self


class AdjudicationEntry(StrictModel):
    """Typed projection of a verified run event and its adjudication details."""

    event: RunEvent
    adjudication: ConflictAdjudication


class AdjudicationChainError(EventChainError):
    """Raised when typed adjudication semantics break the append-only chain."""


class ConflictAdjudicationLog:
    """Append and rebuild strict adjudications using the shared event hash chain."""

    def __init__(self, root: Path) -> None:
        self._events = EventLog(root)

    def read_verified(self) -> list[AdjudicationEntry]:
        entries: list[AdjudicationEntry] = []
        heads: dict[str, AdjudicationEntry] = {}
        for event in self._events.read_verified():
            if event.stage != ADJUDICATION_STAGE or event.event_type != ADJUDICATION_EVENT_TYPE:
                raise AdjudicationChainError(
                    f"unexpected event kind in adjudication chain: {event.event_id}"
                )
            try:
                details = ConflictAdjudication.model_validate(event.details, strict=True)
            except ValidationError as exc:
                raise AdjudicationChainError(
                    f"invalid adjudication details at {event.event_id}: {exc}"
                ) from exc
            previous = heads.get(details.conflict_id)
            expected_supersedes = previous.event.event_id if previous is not None else None
            if details.supersedes_event_id != expected_supersedes:
                raise AdjudicationChainError(
                    f"invalid supersedes link for conflict {details.conflict_id}"
                )
            entry = AdjudicationEntry(event=event, adjudication=details)
            entries.append(entry)
            heads[details.conflict_id] = entry
        return entries

    def current(self) -> list[AdjudicationEntry]:
        """Rebuild the latest decision per conflict without mutating event history."""
        heads: dict[str, AdjudicationEntry] = {}
        for entry in self.read_verified():
            heads[entry.adjudication.conflict_id] = entry
        return [heads[conflict_id] for conflict_id in sorted(heads)]

    def append(
        self,
        *,
        run_id: str,
        conflict_id: str,
        left_episode_id: str,
        right_episode_id: str,
        decision: AdjudicationDecision,
        reviewer_id: str,
        reviewer_authority: Literal["human"],
        rubric_version: str,
        rationale: str,
        supersedes_event_id: str | None,
        parent_manifest_id: str | None = None,
        timestamp_utc: str | None = None,
    ) -> AdjudicationEntry:
        existing = self.read_verified()
        prior = next(
            (
                entry
                for entry in reversed(existing)
                if entry.adjudication.conflict_id == conflict_id
            ),
            None,
        )
        expected_supersedes = prior.event.event_id if prior is not None else None
        if supersedes_event_id != expected_supersedes:
            raise AdjudicationChainError(
                "a decision must explicitly supersede the current decision for its conflict"
            )
        details = ConflictAdjudication(
            conflict_id=conflict_id,
            left_episode_id=left_episode_id,
            right_episode_id=right_episode_id,
            decision=decision,
            reviewer_id=reviewer_id,
            reviewer_authority=reviewer_authority,
            rubric_version=rubric_version,
            rationale=rationale,
            supersedes_event_id=supersedes_event_id,
        )
        event_details = cast(
            dict[str, JsonValue], details.model_dump(mode="json", exclude_none=False)
        )
        event = self._events.append(
            run_id=run_id,
            stage=ADJUDICATION_STAGE,
            event_type=ADJUDICATION_EVENT_TYPE,
            details=event_details,
            parent_manifest_id=parent_manifest_id,
            timestamp_utc=timestamp_utc,
        )
        return AdjudicationEntry(event=event, adjudication=details)
