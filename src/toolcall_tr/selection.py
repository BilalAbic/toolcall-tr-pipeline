"""Deterministic stratified reserve ranking and immutable S400 selection freeze."""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.hashing import sha256_jcs, stable_id
from toolcall_tr.models import EpisodeId, NonEmptyStr, StrictModel

SELECTION_POLICY_VERSION = "selection-policy-0.1.0"
type TierName = Literal["S30", "S100", "S250", "S400"]
PRODUCTION_TIERS: tuple[tuple[TierName, int], ...] = (
    ("S30", 30),
    ("S100", 100),
    ("S250", 250),
    ("S400", 400),
)

type SourceVerdict = Literal["source_valid", "source_review", "source_invalid"]
type GroundingOrigin = Literal[
    "explicit_user",
    "prior_turn",
    "tool_result",
    "system_context",
    "deterministic_default",
    "derived",
    "must_not_infer",
    "unknown",
]


class SelectionStratum(StrictModel):
    dataset_namespace: NonEmptyStr
    action: NonEmptyStr
    call_shape: NonEmptyStr
    tool_family: NonEmptyStr
    domain: NonEmptyStr
    length_bucket: NonEmptyStr
    tool_count: Annotated[int, Field(ge=0)]


class SelectionCandidate(StrictModel):
    episode_id: EpisodeId
    stratum: SelectionStratum
    source_verdict: SourceVerdict
    human_adjudicated: bool
    argument_grounding: list[GroundingOrigin]
    unresolved_hard_conflict: bool


class ReserveQueueEntry(StrictModel):
    rank: Annotated[int, Field(gt=0)]
    episode_id: EpisodeId
    stratum: SelectionStratum
    rank_key: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class RankedMembership(StrictModel):
    rank: Annotated[int, Field(gt=0)]
    reserve_rank: Annotated[int, Field(gt=0)]
    episode_id: EpisodeId


class TierMembership(StrictModel):
    tier: TierName
    episode_ids: list[EpisodeId]


class SelectionManifest(StrictModel):
    schema_version: Literal["selection-manifest-0.1.0"] = "selection-manifest-0.1.0"
    selection_policy_version: Literal["selection-policy-0.1.0"] = SELECTION_POLICY_VERSION
    selection_manifest_id: Annotated[str, Field(pattern=r"^selection_[0-9a-f]{64}$")]
    ranked_reserve_queue: Annotated[list[ReserveQueueEntry], Field(min_length=400)]
    master_membership: Annotated[list[RankedMembership], Field(min_length=400, max_length=400)]
    tiers: Annotated[list[TierMembership], Field(min_length=4, max_length=4)]

    @model_validator(mode="after")
    def validate_frozen_membership(self) -> SelectionManifest:
        if [entry.rank for entry in self.ranked_reserve_queue] != list(
            range(1, len(self.ranked_reserve_queue) + 1)
        ):
            raise ValueError("reserve queue ranks must be contiguous")
        reserve_ids = [entry.episode_id for entry in self.ranked_reserve_queue]
        if len(reserve_ids) != len(set(reserve_ids)):
            raise ValueError("reserve queue episode IDs must be unique")

        master_ids = [entry.episode_id for entry in self.master_membership]
        if [entry.rank for entry in self.master_membership] != list(range(1, 401)):
            raise ValueError("S400 master ranks must be 1..400")
        if len(master_ids) != len(set(master_ids)):
            raise ValueError("S400 master membership IDs must be unique")
        reserve_by_rank = {entry.rank: entry.episode_id for entry in self.ranked_reserve_queue}
        if any(
            reserve_by_rank.get(entry.reserve_rank) != entry.episode_id
            for entry in self.master_membership
        ):
            raise ValueError("master membership must reference explicit reserve ranks")

        expected_tiers = [name for name, _ in PRODUCTION_TIERS]
        if [tier.tier for tier in self.tiers] != expected_tiers:
            raise ValueError("selection tiers must be ordered S30/S100/S250/S400")
        for tier, (name, size) in zip(self.tiers, PRODUCTION_TIERS, strict=True):
            if tier.tier != name or tier.episode_ids != master_ids[:size]:
                raise ValueError(f"{name} must be the exact S400 prefix of size {size}")

        body = self.model_dump(mode="json", exclude={"selection_manifest_id"})
        if self.selection_manifest_id != stable_id("selection", body):
            raise ValueError("selection manifest ID does not match deterministic content")
        return self


def _stratum_key(stratum: SelectionStratum) -> tuple[str, str]:
    body = stratum.model_dump(mode="json")
    return sha256_jcs({"policy": SELECTION_POLICY_VERSION, "stratum": body}), str(body)


def _candidate_rank_key(candidate: SelectionCandidate) -> str:
    return sha256_jcs(
        {
            "policy": SELECTION_POLICY_VERSION,
            "episode_id": candidate.episode_id,
            "stratum": candidate.stratum.model_dump(mode="json"),
        }
    )


def build_ranked_reserve_queue(
    candidates: list[SelectionCandidate],
) -> list[ReserveQueueEntry]:
    """Round-robin deterministic strata after excluding unresolved hard conflicts."""
    candidate_ids = [candidate.episode_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("selection candidate episode IDs must be unique")

    buckets: dict[SelectionStratum, list[SelectionCandidate]] = defaultdict(list)
    for candidate in candidates:
        if not candidate.unresolved_hard_conflict:
            buckets[candidate.stratum].append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (_candidate_rank_key(item), item.episode_id))

    ordered_strata = sorted(buckets, key=_stratum_key)
    ordered_candidates: list[SelectionCandidate] = []
    depth = 0
    while True:
        appended = False
        for stratum in ordered_strata:
            bucket = buckets[stratum]
            if depth < len(bucket):
                ordered_candidates.append(bucket[depth])
                appended = True
        if not appended:
            break
        depth += 1

    return [
        ReserveQueueEntry(
            rank=rank,
            episode_id=candidate.episode_id,
            stratum=candidate.stratum,
            rank_key=_candidate_rank_key(candidate),
        )
        for rank, candidate in enumerate(ordered_candidates, start=1)
    ]


def freeze_exclusion_reasons(candidate: SelectionCandidate) -> tuple[str, ...]:
    reasons: list[str] = []
    if candidate.source_verdict != "source_valid":
        reasons.append("source_not_valid")
    if not candidate.human_adjudicated:
        reasons.append("not_human_adjudicated")
    if candidate.unresolved_hard_conflict:
        reasons.append("unresolved_hard_conflict")
    if "unknown" in candidate.argument_grounding:
        reasons.append("unknown_grounding")
    if "must_not_infer" in candidate.argument_grounding:
        reasons.append("must_not_infer_grounding")
    return tuple(reasons)


def freeze_s400(candidates: list[SelectionCandidate]) -> SelectionManifest:
    """Freeze production memberships by scanning the ranked reserve queue once."""
    queue = build_ranked_reserve_queue(candidates)
    by_id = {candidate.episode_id: candidate for candidate in candidates}
    selected_entries = [
        entry for entry in queue if not freeze_exclusion_reasons(by_id[entry.episode_id])
    ][:400]
    if len(selected_entries) < 400:
        raise ValueError(
            "S400 freeze requires at least 400 human-adjudicated source_valid candidates "
            "without unresolved conflict or grounding"
        )

    master = [
        RankedMembership(
            rank=rank,
            reserve_rank=entry.rank,
            episode_id=entry.episode_id,
        )
        for rank, entry in enumerate(selected_entries, start=1)
    ]
    master_ids = [entry.episode_id for entry in master]
    tiers = [
        TierMembership(tier=name, episode_ids=master_ids[:size]) for name, size in PRODUCTION_TIERS
    ]
    body = {
        "schema_version": "selection-manifest-0.1.0",
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "ranked_reserve_queue": [entry.model_dump(mode="json") for entry in queue],
        "master_membership": [entry.model_dump(mode="json") for entry in master],
        "tiers": [tier.model_dump(mode="json") for tier in tiers],
    }
    return SelectionManifest(
        selection_manifest_id=stable_id("selection", body),
        ranked_reserve_queue=queue,
        master_membership=master,
        tiers=tiers,
    )
