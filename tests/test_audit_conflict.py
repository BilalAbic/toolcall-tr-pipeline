from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal, cast

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from tests.helpers import canonical_fixture
from toolcall_tr.adapters import get_adapter
from toolcall_tr.adjudication import (
    AdjudicationChainError,
    ConflictAdjudication,
    ConflictAdjudicationLog,
)
from toolcall_tr.audit import AuditInputError, audit_exact_conflicts
from toolcall_tr.canonicalize import canonicalize
from toolcall_tr.fingerprints import BehaviorComparison
from toolcall_tr.hashing import canonical_bytes
from toolcall_tr.models import CanonicalEpisode
from toolcall_tr.source import ingest_snapshot, register_source


def _episode_copy(
    episode: CanonicalEpisode,
    id_digit: str,
    *,
    target_content: str | None = None,
    topology: str | None = None,
) -> CanonicalEpisode:
    payload = episode.model_dump(mode="json", exclude_none=False)
    payload["episode_id"] = f"ep_{id_digit * 64}"
    if target_content is not None:
        conversation = payload["conversation"]
        assert isinstance(conversation, list) and isinstance(conversation[-1], dict)
        conversation[-1]["content"] = target_content
    if topology is not None:
        annotations = payload["annotations"]
        assert isinstance(annotations, dict)
        annotations["execution_topology"] = topology
    return CanonicalEpisode.model_validate_json(canonical_bytes(payload), strict=True)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(order=st.permutations((0, 1, 2)))
def test_exact_alias_owner_and_audit_are_input_order_invariant(
    fixture_root: Path, order: list[int]
) -> None:
    base = canonical_fixture(fixture_root / "no_tool", "no_tool", 2)
    episodes = [
        _episode_copy(base, "b"),
        _episode_copy(base, "a"),
        _episode_copy(base, "c", target_content="Hi!"),
    ]
    expected = audit_exact_conflicts(episodes)
    actual = audit_exact_conflicts(episodes[index] for index in order)

    assert canonical_bytes(actual) == canonical_bytes(expected)
    assert len(actual.duplicate_groups) == 1
    duplicate = actual.duplicate_groups[0]
    assert duplicate.owner_episode_id == f"ep_{'a' * 64}"
    assert duplicate.alias_episode_ids == [f"ep_{'b' * 64}"]
    assert duplicate.automatic_drop is False
    assert actual.automatic_drop_episode_ids == []
    assert len(actual.conflict_candidates) == 1
    assert actual.conflict_candidates[0].comparison is BehaviorComparison.HARD_CONFLICT


def test_duplicate_episode_identity_fails_closed(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "no_tool", "no_tool", 2)
    with pytest.raises(AuditInputError, match="duplicate episode ID"):
        audit_exact_conflicts([episode, episode])


def test_adjudication_payload_forbids_extra_fields_and_same_side() -> None:
    payload = {
        "conflict_id": f"conf_{'a' * 64}",
        "left_episode_id": f"ep_{'a' * 64}",
        "right_episode_id": f"ep_{'a' * 64}",
        "decision": "defer",
        "reviewer_id": "reviewer-1",
        "reviewer_authority": "human",
        "rubric_version": "source-conflict-rubric-0.1.0",
        "rationale": "Fixture rationale.",
        "supersedes_event_id": None,
        "unexpected": True,
    }
    with pytest.raises(ValidationError):
        ConflictAdjudication.model_validate(payload, strict=True)
    del payload["unexpected"]
    with pytest.raises(ValidationError, match="two distinct episodes"):
        ConflictAdjudication.model_validate(payload, strict=True)


def test_unknown_topology_same_call_multiset_routes_to_order_review(
    fixture_root: Path,
) -> None:
    root = fixture_root / "xlam"
    snapshot = register_source(
        root,
        dataset_namespace="fixture-xlam",
        source_revision="fixture-v1",
        license_id="test-only",
    )
    bronze = list(ingest_snapshot(snapshot, root))[1]
    assert bronze.parsed_record is not None
    left_adapted = get_adapter("xlam").adapt(bronze.parsed_record)
    reversed_record = deepcopy(bronze.parsed_record)
    calls = reversed_record["tool_calls"]
    assert isinstance(calls, list)
    calls.reverse()
    right_adapted = get_adapter("xlam").adapt(reversed_record)
    left = _episode_copy(canonicalize(bronze, left_adapted, run_event_id="run_fixture"), "a")
    right = _episode_copy(canonicalize(bronze, right_adapted, run_event_id="run_fixture"), "b")

    candidate = audit_exact_conflicts([right, left]).conflict_candidates[0]
    assert candidate.comparison is BehaviorComparison.ORDER_AMBIGUITY_REVIEW
    assert candidate.call_multiset_same is True
    assert candidate.review_required is True
    assert candidate.automatic_drop is False

    known_order = _episode_copy(right, "b", topology="sequential")
    known_candidate = audit_exact_conflicts([left, known_order]).conflict_candidates[0]
    assert known_candidate.comparison is BehaviorComparison.HARD_CONFLICT


def test_adjudication_chain_requires_human_authority_and_explicit_supersedes(
    fixture_root: Path, tmp_path: Path
) -> None:
    base = canonical_fixture(fixture_root / "no_tool", "no_tool", 2)
    left = _episode_copy(base, "a")
    right = _episode_copy(base, "b", target_content="Hi!")
    candidate = audit_exact_conflicts([left, right]).conflict_candidates[0]
    log = ConflictAdjudicationLog(tmp_path / "adjudications")

    model_authority = cast(Literal["human"], "model")
    with pytest.raises(ValidationError, match="reviewer_authority"):
        log.append(
            run_id="run_fixture",
            conflict_id=candidate.conflict_id,
            left_episode_id=candidate.left_owner_episode_id,
            right_episode_id=candidate.right_owner_episode_id,
            decision="defer",
            reviewer_id="reviewer-1",
            reviewer_authority=model_authority,
            rubric_version="source-conflict-rubric-0.1.0",
            rationale="Insufficient evidence for a final decision.",
            supersedes_event_id=None,
            timestamp_utc="2026-08-12T00:00:00.000000Z",
        )
    assert not list((tmp_path / "adjudications").glob("*.jsonl"))

    first = log.append(
        run_id="run_fixture",
        conflict_id=candidate.conflict_id,
        left_episode_id=candidate.left_owner_episode_id,
        right_episode_id=candidate.right_owner_episode_id,
        decision="defer",
        reviewer_id="reviewer-1",
        reviewer_authority="human",
        rubric_version="source-conflict-rubric-0.1.0",
        rationale="Insufficient evidence for a final decision.",
        supersedes_event_id=None,
        timestamp_utc="2026-08-12T00:00:00.000000Z",
    )
    with pytest.raises(AdjudicationChainError, match="explicitly supersede"):
        log.append(
            run_id="run_fixture",
            conflict_id=candidate.conflict_id,
            left_episode_id=candidate.left_owner_episode_id,
            right_episode_id=candidate.right_owner_episode_id,
            decision="keep_left",
            reviewer_id="reviewer-2",
            reviewer_authority="human",
            rubric_version="source-conflict-rubric-0.1.0",
            rationale="The left behavior is explicitly supported.",
            supersedes_event_id=None,
        )
    second = log.append(
        run_id="run_fixture",
        conflict_id=candidate.conflict_id,
        left_episode_id=candidate.left_owner_episode_id,
        right_episode_id=candidate.right_owner_episode_id,
        decision="keep_left",
        reviewer_id="reviewer-2",
        reviewer_authority="human",
        rubric_version="source-conflict-rubric-0.1.0",
        rationale="The left behavior is explicitly supported.",
        supersedes_event_id=first.event.event_id,
        timestamp_utc="2026-08-12T00:00:01.000000Z",
    )

    verified = log.read_verified()
    assert [entry.adjudication.decision for entry in verified] == ["defer", "keep_left"]
    assert verified[1].adjudication.supersedes_event_id == first.event.event_id
    assert log.current() == [second]


@pytest.mark.parametrize(
    "decision",
    [
        "keep_left",
        "keep_right",
        "keep_both_context_insufficient",
        "source_error",
        "policy_variant",
        "defer",
    ],
)
def test_all_adjudication_decisions_are_strictly_supported(
    fixture_root: Path, tmp_path: Path, decision: str
) -> None:
    base = canonical_fixture(fixture_root / "no_tool", "no_tool", 2)
    left = _episode_copy(base, "a")
    right = _episode_copy(base, "b", target_content="Hi!")
    candidate = audit_exact_conflicts([left, right]).conflict_candidates[0]
    log = ConflictAdjudicationLog(tmp_path / decision)
    typed_decision = cast(
        Literal[
            "keep_left",
            "keep_right",
            "keep_both_context_insufficient",
            "source_error",
            "policy_variant",
            "defer",
        ],
        decision,
    )
    entry = log.append(
        run_id="run_fixture",
        conflict_id=candidate.conflict_id,
        left_episode_id=candidate.left_owner_episode_id,
        right_episode_id=candidate.right_owner_episode_id,
        decision=typed_decision,
        reviewer_id="reviewer-1",
        reviewer_authority="human",
        rubric_version="source-conflict-rubric-0.1.0",
        rationale="Fixture adjudication.",
        supersedes_event_id=None,
        timestamp_utc="2026-08-12T00:00:00.000000Z",
    )
    assert entry.adjudication.decision == decision
