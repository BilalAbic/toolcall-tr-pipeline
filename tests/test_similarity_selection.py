from __future__ import annotations

import json

import pytest

from toolcall_tr.selection import (
    SelectionCandidate,
    SelectionManifest,
    SelectionStratum,
    build_ranked_reserve_queue,
    freeze_exclusion_reasons,
    freeze_s400,
)
from toolcall_tr.similarity import connected_components, retrieve_near_duplicate_candidates
from toolcall_tr.split_guard import SplitGuardError, assert_components_in_one_split


def episode_id(index: int) -> str:
    return f"ep_{index:064x}"


def candidate(index: int, **changes: object) -> SelectionCandidate:
    values: dict[str, object] = {
        "episode_id": episode_id(index),
        "stratum": SelectionStratum(
            dataset_namespace=f"source-{index % 2}",
            action=("tool_call", "clarification", "direct_answer")[index % 3],
            call_shape="single" if index % 3 == 0 else "none",
            tool_family=f"family-{index % 5}",
            domain=f"domain-{index % 4}",
            length_bucket=("short", "medium", "long")[index % 3],
            tool_count=index % 4,
        ),
        "source_verdict": "source_valid",
        "human_adjudicated": True,
        "argument_grounding": ["explicit_user"] if index % 3 == 0 else [],
        "unresolved_hard_conflict": False,
    }
    values.update(changes)
    return SelectionCandidate.model_validate(values, strict=True)


def test_similarity_candidates_never_authorize_automatic_drop() -> None:
    candidates = retrieve_near_duplicate_candidates(
        {
            episode_id(1): "Book a flight from Ankara to Istanbul tomorrow",
            episode_id(2): "BOOK  a flight from Ankara to Istanbul tomorrow!",
            episode_id(3): "Convert 25 Celsius to Fahrenheit",
        },
        threshold=0.75,
        ngram_size=3,
    )
    assert [(item.left_episode_id, item.right_episode_id) for item in candidates] == [
        (episode_id(1), episode_id(2))
    ]
    assert candidates[0].disposition == "human_review"
    assert candidates[0].automatic_drop is False


def test_connected_components_are_deterministic_and_guard_splits() -> None:
    identifiers = [episode_id(index) for index in range(1, 6)]
    edges = [
        (episode_id(2), episode_id(3)),
        (episode_id(1), episode_id(2)),
        (episode_id(5), episode_id(4)),
    ]
    components = connected_components(reversed(identifiers), reversed(edges))
    assert components == [identifiers[:3], identifiers[3:]]
    assignments = {identifier: "train" for identifier in identifiers}
    assignments[episode_id(4)] = assignments[episode_id(5)] = "test"
    assert_components_in_one_split(components, assignments)
    assignments[episode_id(3)] = "validation"
    with pytest.raises(SplitGuardError, match="crosses splits"):
        assert_components_in_one_split(components, assignments)


def test_ranked_queue_excludes_unresolved_hard_conflicts_and_is_order_independent() -> None:
    candidates = [candidate(index) for index in range(20)]
    conflicted = candidate(100, unresolved_hard_conflict=True)
    all_candidates = [*candidates, conflicted]
    left = build_ranked_reserve_queue(all_candidates)
    right = build_ranked_reserve_queue(list(reversed(all_candidates)))
    assert left == right
    assert conflicted.episode_id not in {entry.episode_id for entry in left}
    assert [entry.rank for entry in left] == list(range(1, 21))


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"source_verdict": "source_review"}, "source_not_valid"),
        ({"human_adjudicated": False}, "not_human_adjudicated"),
        ({"unresolved_hard_conflict": True}, "unresolved_hard_conflict"),
        ({"argument_grounding": ["unknown"]}, "unknown_grounding"),
        ({"argument_grounding": ["must_not_infer"]}, "must_not_infer_grounding"),
    ],
)
def test_freeze_gate_is_fail_closed(changes: dict[str, object], reason: str) -> None:
    assert reason in freeze_exclusion_reasons(candidate(1, **changes))


def test_s400_tiers_are_strict_prefixes_and_rerun_is_byte_identical() -> None:
    candidates = [candidate(index) for index in range(400)]
    left = freeze_s400(candidates)
    right = freeze_s400(list(reversed(candidates)))
    left_bytes = json.dumps(
        left.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    right_bytes = json.dumps(
        right.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert left_bytes == right_bytes
    assert [tier.tier for tier in left.tiers] == ["S30", "S100", "S250", "S400"]
    master = [entry.episode_id for entry in left.master_membership]
    assert [len(tier.episode_ids) for tier in left.tiers] == [30, 100, 250, 400]
    assert all(tier.episode_ids == master[: len(tier.episode_ids)] for tier in left.tiers)
    assert set(SelectionManifest.model_fields) == {
        "schema_version",
        "selection_policy_version",
        "selection_manifest_id",
        "ranked_reserve_queue",
        "master_membership",
        "tiers",
    }


def test_s400_scans_ranked_reserves_and_rejects_shortfall() -> None:
    candidates = [candidate(index) for index in range(400)]
    candidates[0] = candidate(0, argument_grounding=["unknown"])
    with pytest.raises(ValueError, match="requires at least 400"):
        freeze_s400(candidates)

    replacement = candidate(1000)
    manifest = freeze_s400([*candidates, replacement])
    selected = {entry.episode_id for entry in manifest.master_membership}
    assert candidates[0].episode_id not in selected
    assert replacement.episode_id in selected
