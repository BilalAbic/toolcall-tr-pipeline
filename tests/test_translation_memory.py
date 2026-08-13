from __future__ import annotations

from typing import Literal

import pytest

from toolcall_tr.hashing import canonical_bytes
from toolcall_tr.models import DecisionAction
from toolcall_tr.translation_memory import (
    MemoryConflictError,
    MemoryLookupKey,
    SegmentMemoryEntry,
    build_memory_entry,
    lookup_exact,
)

CONTEXT = f"sha256:{'a' * 64}"
TOOL = f"tool_{'b' * 64}"


def entry(
    *,
    target_text: str = "Merhaba",
    state: Literal["active", "superseded", "revoked"] = "active",
):
    return build_memory_entry(
        source_text="Hello",
        target_text=target_text,
        field_policy_version="field-policy-0.1.0",
        argument_policy_version="argument-policy-0.1.0",
        locale_policy_version="locale-policy-0.1.0",
        presented_context_fingerprint=CONTEXT,
        tool_scope=[TOOL],
        decision_action=DecisionAction.DIRECT_ANSWER,
        origin_translation_config_id="translation-config-1",
        human_review_event_id="evt_human_001",
        state=state,
    )


def key() -> MemoryLookupKey:
    source = entry()
    return MemoryLookupKey(
        segment_source_sha256=source.segment_source_sha256,
        field_policy_version=source.field_policy_version,
        argument_policy_version=source.argument_policy_version,
        locale_policy_version=source.locale_policy_version,
        presented_context_fingerprint=source.presented_context_fingerprint,
        tool_scope=source.tool_scope,
        decision_action=source.decision_action,
    )


def test_exact_lookup_is_content_addressed_and_ignores_nonactive_entries() -> None:
    active = entry()
    revoked = entry(target_text="Selam", state="revoked")
    assert lookup_exact([revoked, active], key()) == active
    assert entry() == active


def test_lookup_never_fuzzily_matches_context_or_policy() -> None:
    active = entry()
    mismatch = key().model_copy(update={"field_policy_version": "field-policy-other"})
    assert lookup_exact([active], mismatch) is None


def test_multiple_active_targets_are_a_fail_closed_conflict() -> None:
    with pytest.raises(MemoryConflictError, match="multiple active targets"):
        lookup_exact([entry(), entry(target_text="Selam")], key())


def test_strict_entry_rejects_noncanonical_tool_scope() -> None:
    payload = entry().model_dump(mode="json")
    payload["tool_scope"] = [TOOL, TOOL]
    with pytest.raises(ValueError, match="tool scope"):
        SegmentMemoryEntry.model_validate_json(
            canonical_bytes(payload),
            strict=True,
        )
