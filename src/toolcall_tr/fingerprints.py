"""Exact context/behavior fingerprints kept separate from membership identity."""

from __future__ import annotations

from enum import StrEnum

from toolcall_tr.hashing import JsonValue, canonical_bytes, sha256_jcs
from toolcall_tr.models import CanonicalEpisode, ToolCall


def _call_behavior(call: ToolCall) -> dict[str, JsonValue]:
    return {"name": call.function.name, "arguments": call.function.arguments}


def structural_context_fingerprint(episode: CanonicalEpisode) -> str:
    target_index = episode.annotations.target_message_index
    return sha256_jcs(
        {
            "conversation_prefix": [
                message.model_dump(mode="json", exclude_none=False)
                for message in episode.conversation[:target_index]
            ],
            "tool_ids": sorted(tool.tool_id for tool in episode.tools),
        }
    )


def presented_context_fingerprint(episode: CanonicalEpisode) -> str:
    target_index = episode.annotations.target_message_index
    return sha256_jcs(
        {
            "conversation_prefix": [
                message.model_dump(mode="json", exclude_none=False)
                for message in episode.conversation[:target_index]
            ],
            "presented_tools": [
                {"tool_id": tool.tool_id, "documentation_hash": tool.documentation_hash}
                for tool in episode.tools
            ],
        }
    )


def ordered_behavior_fingerprint(episode: CanonicalEpisode) -> str:
    target = episode.conversation[episode.annotations.target_message_index]
    return sha256_jcs(
        {
            "action": episode.annotations.decision.action,
            "calls": [_call_behavior(call) for call in target.tool_calls or []],
            "content": target.content,
        }
    )


def call_multiset_fingerprint(episode: CanonicalEpisode) -> str:
    target = episode.conversation[episode.annotations.target_message_index]
    calls = [_call_behavior(call) for call in target.tool_calls or []]
    calls.sort(key=canonical_bytes)
    return sha256_jcs(calls)


class BehaviorComparison(StrEnum):
    DIFFERENT_CONTEXT = "different_context"
    EXACT_DUPLICATE = "exact_duplicate"
    ORDER_AMBIGUITY_REVIEW = "order_ambiguity_review"
    HARD_CONFLICT = "hard_conflict"


def compare_behavior(left: CanonicalEpisode, right: CanonicalEpisode) -> BehaviorComparison:
    if presented_context_fingerprint(left) != presented_context_fingerprint(right):
        return BehaviorComparison.DIFFERENT_CONTEXT
    if ordered_behavior_fingerprint(left) == ordered_behavior_fingerprint(right):
        return BehaviorComparison.EXACT_DUPLICATE
    if (
        left.annotations.execution_topology == "unknown"
        and right.annotations.execution_topology == "unknown"
        and call_multiset_fingerprint(left) == call_multiset_fingerprint(right)
    ):
        return BehaviorComparison.ORDER_AMBIGUITY_REVIEW
    return BehaviorComparison.HARD_CONFLICT
