"""Deterministic exact-duplicate and behavior-conflict audit views.

The audit is deliberately non-destructive: aliases identify a canonical owner for
reporting, but every input episode remains present and every behavior difference is
routed to review.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.fingerprints import (
    BehaviorComparison,
    call_multiset_fingerprint,
    compare_behavior,
    ordered_behavior_fingerprint,
    presented_context_fingerprint,
)
from toolcall_tr.hashing import sha256_jcs, stable_id
from toolcall_tr.models import CanonicalEpisode, DecisionAction, EpisodeId, Sha256, StrictModel

ConflictId = Annotated[str, Field(pattern=r"^conf_[0-9a-f]{64}$")]
AuditId = Annotated[str, Field(pattern=r"^audit_[0-9a-f]{64}$")]


class AuditInputError(ValueError):
    """Raised when an audit input cannot have a unique deterministic identity."""


class ExactDuplicateGroup(StrictModel):
    """One exact context/behavior group with a deterministic reporting owner."""

    schema_version: Literal["exact-duplicate-0.1.0"] = "exact-duplicate-0.1.0"
    presented_context_fingerprint: Sha256
    ordered_behavior_fingerprint: Sha256
    owner_episode_id: EpisodeId
    alias_episode_ids: Annotated[list[EpisodeId], Field(min_length=1)]
    member_episode_ids: Annotated[list[EpisodeId], Field(min_length=2)]
    automatic_drop: Literal[False] = False

    @model_validator(mode="after")
    def validate_members(self) -> ExactDuplicateGroup:
        expected_members = sorted([self.owner_episode_id, *self.alias_episode_ids])
        if self.alias_episode_ids != sorted(set(self.alias_episode_ids)):
            raise ValueError("duplicate aliases must be unique and sorted")
        if self.owner_episode_id in self.alias_episode_ids:
            raise ValueError("duplicate owner cannot also be an alias")
        if self.member_episode_ids != expected_members:
            raise ValueError("duplicate members must be the sorted owner/alias union")
        if self.owner_episode_id != self.member_episode_ids[0]:
            raise ValueError("duplicate owner must be the smallest stable episode ID")
        return self


class ConflictCandidate(StrictModel):
    """A behavior pair that requires review and carries no automatic disposition."""

    schema_version: Literal["conflict-candidate-0.1.0"] = "conflict-candidate-0.1.0"
    conflict_id: ConflictId
    comparison: Literal[
        BehaviorComparison.HARD_CONFLICT,
        BehaviorComparison.ORDER_AMBIGUITY_REVIEW,
    ]
    presented_context_fingerprint: Sha256
    left_ordered_behavior_fingerprint: Sha256
    right_ordered_behavior_fingerprint: Sha256
    left_owner_episode_id: EpisodeId
    right_owner_episode_id: EpisodeId
    left_member_episode_ids: Annotated[list[EpisodeId], Field(min_length=1)]
    right_member_episode_ids: Annotated[list[EpisodeId], Field(min_length=1)]
    call_multiset_same: bool
    review_required: Literal[True] = True
    resolution: Literal["unresolved"] = "unresolved"
    automatic_drop: Literal[False] = False

    @model_validator(mode="after")
    def validate_identity_and_order(self) -> ConflictCandidate:
        if self.left_ordered_behavior_fingerprint >= self.right_ordered_behavior_fingerprint:
            raise ValueError("conflict behaviors must be in strict fingerprint order")
        if self.left_member_episode_ids != sorted(set(self.left_member_episode_ids)):
            raise ValueError("left conflict members must be unique and sorted")
        if self.right_member_episode_ids != sorted(set(self.right_member_episode_ids)):
            raise ValueError("right conflict members must be unique and sorted")
        if self.left_owner_episode_id != self.left_member_episode_ids[0]:
            raise ValueError("left owner must be the smallest stable episode ID")
        if self.right_owner_episode_id != self.right_member_episode_ids[0]:
            raise ValueError("right owner must be the smallest stable episode ID")
        expected_id = conflict_id(
            self.presented_context_fingerprint,
            self.left_ordered_behavior_fingerprint,
            self.right_ordered_behavior_fingerprint,
        )
        if self.conflict_id != expected_id:
            raise ValueError("conflict ID does not match context/behavior identity")
        if (
            self.comparison is BehaviorComparison.ORDER_AMBIGUITY_REVIEW
            and not self.call_multiset_same
        ):
            raise ValueError("order ambiguity requires the same call multiset")
        return self


class ExactConflictAudit(StrictModel):
    """Stable derived view over all audited episode identities."""

    schema_version: Literal["exact-conflict-audit-0.1.0"] = "exact-conflict-audit-0.1.0"
    audit_id: AuditId
    episode_ids: list[EpisodeId]
    duplicate_groups: list[ExactDuplicateGroup]
    conflict_candidates: list[ConflictCandidate]
    automatic_drop_episode_ids: list[EpisodeId]

    @model_validator(mode="after")
    def validate_report(self) -> ExactConflictAudit:
        if self.episode_ids != sorted(set(self.episode_ids)):
            raise ValueError("audit episode IDs must be unique and sorted")
        if self.automatic_drop_episode_ids:
            raise ValueError("exact/conflict audit must never automatically drop episodes")
        expected = _audit_id_payload(
            self.episode_ids,
            self.duplicate_groups,
            self.conflict_candidates,
        )
        if self.audit_id != stable_id("audit", expected):
            raise ValueError("audit ID does not match report content")
        return self


def conflict_id(context_fingerprint: str, left_behavior: str, right_behavior: str) -> str:
    """Return an alias-independent identity for a pair of behaviors in one context."""
    behaviors = sorted((left_behavior, right_behavior))
    return stable_id(
        "conf",
        {
            "presented_context_fingerprint": context_fingerprint,
            "ordered_behavior_fingerprints": behaviors,
        },
    )


def _audit_id_payload(
    episode_ids: list[str],
    duplicate_groups: list[ExactDuplicateGroup],
    conflict_candidates: list[ConflictCandidate],
) -> dict[str, object]:
    return {
        "schema_version": "exact-conflict-audit-0.1.0",
        "episode_ids": episode_ids,
        "duplicate_groups": [
            group.model_dump(mode="json", exclude_none=False) for group in duplicate_groups
        ],
        "conflict_candidates": [
            candidate.model_dump(mode="json", exclude_none=False)
            for candidate in conflict_candidates
        ],
        "automatic_drop_episode_ids": [],
    }


def audit_exact_conflicts(episodes: Iterable[CanonicalEpisode]) -> ExactConflictAudit:
    """Classify exact aliases and behavior conflicts independent of input order."""
    by_episode_id: dict[str, CanonicalEpisode] = {}
    for episode in episodes:
        if episode.episode_id in by_episode_id:
            raise AuditInputError(f"duplicate episode ID in audit input: {episode.episode_id}")
        by_episode_id[episode.episode_id] = episode

    context_groups: dict[str, dict[str, list[CanonicalEpisode]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for episode in by_episode_id.values():
        context_groups[presented_context_fingerprint(episode)][
            ordered_behavior_fingerprint(episode)
        ].append(episode)

    duplicate_groups: list[ExactDuplicateGroup] = []
    conflict_candidates: list[ConflictCandidate] = []
    for context_fingerprint in sorted(context_groups):
        behavior_groups = context_groups[context_fingerprint]
        for behavior_fingerprint in sorted(behavior_groups):
            members = sorted(
                behavior_groups[behavior_fingerprint], key=lambda episode: episode.episode_id
            )
            if len(members) > 1:
                member_ids = [member.episode_id for member in members]
                duplicate_groups.append(
                    ExactDuplicateGroup(
                        presented_context_fingerprint=context_fingerprint,
                        ordered_behavior_fingerprint=behavior_fingerprint,
                        owner_episode_id=member_ids[0],
                        alias_episode_ids=member_ids[1:],
                        member_episode_ids=member_ids,
                    )
                )

        for left_behavior, right_behavior in combinations(sorted(behavior_groups), 2):
            left_members = sorted(
                behavior_groups[left_behavior], key=lambda episode: episode.episode_id
            )
            right_members = sorted(
                behavior_groups[right_behavior], key=lambda episode: episode.episode_id
            )
            left = left_members[0]
            right = right_members[0]
            comparison = compare_behavior(left, right)
            same_multiset = call_multiset_fingerprint(left) == call_multiset_fingerprint(right)
            all_topologies_unknown = all(
                episode.annotations.execution_topology == "unknown"
                for episode in [*left_members, *right_members]
            )
            if not (
                comparison is BehaviorComparison.ORDER_AMBIGUITY_REVIEW
                and left.annotations.decision.action is DecisionAction.TOOL_CALL
                and right.annotations.decision.action is DecisionAction.TOOL_CALL
                and same_multiset
                and all_topologies_unknown
            ):
                comparison = BehaviorComparison.HARD_CONFLICT
            conflict_candidates.append(
                ConflictCandidate(
                    conflict_id=conflict_id(
                        context_fingerprint,
                        left_behavior,
                        right_behavior,
                    ),
                    comparison=comparison,
                    presented_context_fingerprint=context_fingerprint,
                    left_ordered_behavior_fingerprint=left_behavior,
                    right_ordered_behavior_fingerprint=right_behavior,
                    left_owner_episode_id=left.episode_id,
                    right_owner_episode_id=right.episode_id,
                    left_member_episode_ids=[episode.episode_id for episode in left_members],
                    right_member_episode_ids=[episode.episode_id for episode in right_members],
                    call_multiset_same=same_multiset,
                )
            )

    duplicate_groups.sort(
        key=lambda group: (
            group.presented_context_fingerprint,
            group.ordered_behavior_fingerprint,
        )
    )
    conflict_candidates.sort(key=lambda candidate: candidate.conflict_id)
    episode_ids = sorted(by_episode_id)
    audit_payload = _audit_id_payload(episode_ids, duplicate_groups, conflict_candidates)
    return ExactConflictAudit(
        audit_id=stable_id("audit", audit_payload),
        episode_ids=episode_ids,
        duplicate_groups=duplicate_groups,
        conflict_candidates=conflict_candidates,
        automatic_drop_episode_ids=[],
    )


def exact_alias_key(episode: CanonicalEpisode) -> str:
    """Expose the exact alias key without conflating it with source membership."""
    return sha256_jcs(
        {
            "presented_context_fingerprint": presented_context_fingerprint(episode),
            "ordered_behavior_fingerprint": ordered_behavior_fingerprint(episode),
        }
    )
