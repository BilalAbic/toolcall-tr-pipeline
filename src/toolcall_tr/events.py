"""Append-only one-event-per-shard JSONL hash chain."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.artifacts import PublishError, publish_bytes_atomic
from toolcall_tr.hashing import JsonValue, canonical_bytes, sha256_jcs
from toolcall_tr.jsonio import iter_jsonl
from toolcall_tr.models import NonEmptyStr, Sha256, StrictModel


class RunEvent(StrictModel):
    schema_version: Literal["run-event-0.1.0"] = "run-event-0.1.0"
    event_id: Annotated[str, Field(pattern=r"^evt_[0-9a-f]{64}$")]
    event_hash: Sha256
    previous_event_hash: Sha256 | None
    sequence: Annotated[int, Field(gt=0)]
    timestamp_utc: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T.*Z$")]
    run_id: NonEmptyStr
    stage: NonEmptyStr
    event_type: NonEmptyStr
    parent_manifest_id: str | None
    details: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_hash(self) -> RunEvent:
        payload = self.model_dump(mode="json", exclude={"event_id", "event_hash"})
        expected = sha256_jcs(payload)
        if (
            self.event_hash != expected
            or self.event_id != f"evt_{expected.removeprefix('sha256:')}"
        ):
            raise ValueError("event ID/hash mismatch")
        return self


class EventChainError(ValueError):
    pass


def event_payload(
    *,
    previous_event_hash: str | None,
    sequence: int,
    timestamp_utc: str,
    run_id: str,
    stage: str,
    event_type: str,
    parent_manifest_id: str | None,
    details: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        "schema_version": "run-event-0.1.0",
        "previous_event_hash": previous_event_hash,
        "sequence": sequence,
        "timestamp_utc": timestamp_utc,
        "run_id": run_id,
        "stage": stage,
        "event_type": event_type,
        "parent_manifest_id": parent_manifest_id,
        "details": details,
    }


class EventLog:
    def __init__(self, root: Path) -> None:
        self.root = root

    def read_verified(self) -> list[RunEvent]:
        if not self.root.exists():
            return []
        events: list[RunEvent] = []
        for path in sorted(self.root.glob("*.jsonl")):
            rows = list(iter_jsonl(path))
            if len(rows) != 1:
                raise EventChainError(f"event shard must contain exactly one row: {path}")
            events.append(RunEvent.model_validate(rows[0], strict=True))
        expected_previous: str | None = None
        for expected_sequence, event in enumerate(events, start=1):
            if (
                event.sequence != expected_sequence
                or event.previous_event_hash != expected_previous
            ):
                raise EventChainError(f"invalid event chain at sequence {expected_sequence}")
            expected_previous = event.event_hash
        return events

    def append(
        self,
        *,
        run_id: str,
        stage: str,
        event_type: str,
        details: dict[str, JsonValue],
        parent_manifest_id: str | None = None,
        timestamp_utc: str | None = None,
    ) -> RunEvent:
        events = self.read_verified()
        timestamp = timestamp_utc or datetime.now(UTC).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )
        payload = event_payload(
            previous_event_hash=events[-1].event_hash if events else None,
            sequence=len(events) + 1,
            timestamp_utc=timestamp,
            run_id=run_id,
            stage=stage,
            event_type=event_type,
            parent_manifest_id=parent_manifest_id,
            details=details,
        )
        event_hash = sha256_jcs(payload)
        event = RunEvent(
            event_id=f"evt_{event_hash.removeprefix('sha256:')}",
            event_hash=event_hash,
            previous_event_hash=events[-1].event_hash if events else None,
            sequence=len(events) + 1,
            timestamp_utc=timestamp,
            run_id=run_id,
            stage=stage,
            event_type=event_type,
            parent_manifest_id=parent_manifest_id,
            details=details,
        )
        target = self.root / f"{event.sequence:012d}-{event.event_id}.jsonl"
        try:
            publish_bytes_atomic(target, canonical_bytes(event) + b"\n")
        except PublishError as exc:
            raise EventChainError(str(exc)) from exc
        self.read_verified()
        return event
