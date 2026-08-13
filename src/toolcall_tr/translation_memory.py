"""Exact, human-promoted segment memory with no database or fuzzy lookup."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.hashing import sha256_bytes, stable_id
from toolcall_tr.models import DecisionAction, NonEmptyStr, Sha256, StrictModel, ToolId

MemoryEntryId = Annotated[str, Field(pattern=r"^mem_[0-9a-f]{64}$")]


class MemoryConflictError(ValueError):
    """An exact lookup has multiple active targets and may not choose one."""


class SegmentMemoryEntry(StrictModel):
    """A human-promoted full segment, immutable once emitted as JSONL."""

    schema_version: Literal["segment-memory-0.1.0"] = "segment-memory-0.1.0"
    memory_entry_id: MemoryEntryId
    segment_source_sha256: Sha256
    source_text: NonEmptyStr
    target_text_sha256: Sha256
    target_text: NonEmptyStr
    field_policy_version: NonEmptyStr
    argument_policy_version: NonEmptyStr
    locale_policy_version: NonEmptyStr
    presented_context_fingerprint: Sha256
    tool_scope: list[ToolId]
    decision_action: DecisionAction
    origin_translation_config_id: NonEmptyStr
    human_review_event_id: NonEmptyStr
    state: Literal["active", "superseded", "revoked"]

    @model_validator(mode="after")
    def validate_entry(self) -> SegmentMemoryEntry:
        if self.segment_source_sha256 != sha256_bytes(self.source_text.encode("utf-8")):
            raise ValueError("segment source hash does not match exact UTF-8 text")
        if self.target_text_sha256 != sha256_bytes(self.target_text.encode("utf-8")):
            raise ValueError("target text hash does not match exact UTF-8 text")
        if self.tool_scope != sorted(set(self.tool_scope)):
            raise ValueError("tool scope must be unique and sorted")
        body = self.model_dump(mode="json", exclude={"memory_entry_id"})
        if self.memory_entry_id != stable_id("mem", body):
            raise ValueError("memory entry ID does not match deterministic content")
        return self


class MemoryLookupKey(StrictModel):
    """The complete exact key; normalized and partial matching are prohibited."""

    segment_source_sha256: Sha256
    field_policy_version: NonEmptyStr
    argument_policy_version: NonEmptyStr
    locale_policy_version: NonEmptyStr
    presented_context_fingerprint: Sha256
    tool_scope: list[ToolId]
    decision_action: DecisionAction

    @model_validator(mode="after")
    def validate_scope(self) -> MemoryLookupKey:
        if self.tool_scope != sorted(set(self.tool_scope)):
            raise ValueError("lookup tool scope must be unique and sorted")
        return self


def build_memory_entry(
    *,
    source_text: str,
    target_text: str,
    field_policy_version: str,
    argument_policy_version: str,
    locale_policy_version: str,
    presented_context_fingerprint: str,
    tool_scope: list[str],
    decision_action: DecisionAction,
    origin_translation_config_id: str,
    human_review_event_id: str,
    state: Literal["active", "superseded", "revoked"] = "active",
) -> SegmentMemoryEntry:
    """Create a content-addressed entry after a separate human promotion event."""
    body = {
        "schema_version": "segment-memory-0.1.0",
        "segment_source_sha256": sha256_bytes(source_text.encode("utf-8")),
        "source_text": source_text,
        "target_text_sha256": sha256_bytes(target_text.encode("utf-8")),
        "target_text": target_text,
        "field_policy_version": field_policy_version,
        "argument_policy_version": argument_policy_version,
        "locale_policy_version": locale_policy_version,
        "presented_context_fingerprint": presented_context_fingerprint,
        "tool_scope": sorted(set(tool_scope)),
        "decision_action": decision_action,
        "origin_translation_config_id": origin_translation_config_id,
        "human_review_event_id": human_review_event_id,
        "state": state,
    }
    return SegmentMemoryEntry(
        memory_entry_id=stable_id("mem", body),
        segment_source_sha256=sha256_bytes(source_text.encode("utf-8")),
        source_text=source_text,
        target_text_sha256=sha256_bytes(target_text.encode("utf-8")),
        target_text=target_text,
        field_policy_version=field_policy_version,
        argument_policy_version=argument_policy_version,
        locale_policy_version=locale_policy_version,
        presented_context_fingerprint=presented_context_fingerprint,
        tool_scope=sorted(set(tool_scope)),
        decision_action=decision_action,
        origin_translation_config_id=origin_translation_config_id,
        human_review_event_id=human_review_event_id,
        state=state,
    )


def lookup_exact(
    entries: Iterable[SegmentMemoryEntry], key: MemoryLookupKey
) -> SegmentMemoryEntry | None:
    """Return one active exact target or fail closed on competing active targets."""
    by_id: dict[str, SegmentMemoryEntry] = {}
    for entry in entries:
        if entry.memory_entry_id in by_id:
            raise ValueError(f"duplicate memory entry ID: {entry.memory_entry_id}")
        by_id[entry.memory_entry_id] = entry
    matches = [
        entry
        for entry in by_id.values()
        if entry.state == "active"
        and entry.segment_source_sha256 == key.segment_source_sha256
        and entry.field_policy_version == key.field_policy_version
        and entry.argument_policy_version == key.argument_policy_version
        and entry.locale_policy_version == key.locale_policy_version
        and entry.presented_context_fingerprint == key.presented_context_fingerprint
        and entry.tool_scope == key.tool_scope
        and entry.decision_action is key.decision_action
    ]
    targets = {entry.target_text_sha256 for entry in matches}
    if len(targets) > 1:
        raise MemoryConflictError("multiple active targets share one exact memory lookup key")
    return min(matches, key=lambda entry: entry.memory_entry_id) if matches else None
