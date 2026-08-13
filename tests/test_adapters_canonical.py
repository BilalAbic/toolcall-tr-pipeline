from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator
from pydantic import ValidationError

from tests.helpers import canonical_fixture
from toolcall_tr.adapters import get_adapter
from toolcall_tr.canonicalize import CanonicalizationError, canonicalize
from toolcall_tr.fingerprints import (
    BehaviorComparison,
    compare_behavior,
    ordered_behavior_fingerprint,
)
from toolcall_tr.models import CanonicalEpisode, DecisionAction
from toolcall_tr.source import ingest_snapshot, register_source


def test_xlam_tool_call_only_stays_awaiting_without_synthetic_result(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam")
    assert episode.annotations.decision.action is DecisionAction.TOOL_CALL
    assert episode.annotations.trajectory_state == "awaiting_tool"
    assert episode.conversation[-1].content is None
    assert all(message.role != "tool" for message in episode.conversation)
    assert len(episode.conversation) == 2


@pytest.mark.parametrize(
    ("index", "action"),
    [
        (0, DecisionAction.CLARIFICATION),
        (1, DecisionAction.TOOL_UNAVAILABLE),
        (2, DecisionAction.DIRECT_ANSWER),
    ],
)
def test_no_tool_actions_remain_source_explicit(
    fixture_root: Path, index: int, action: DecisionAction
) -> None:
    episode = canonical_fixture(fixture_root / "no_tool", "no_tool", index)
    assert episode.annotations.decision.action is action
    assert episode.annotations.trajectory_state == "complete"
    assert episode.conversation[-1].content is not None
    assert episode.conversation[-1].tool_calls is None


def test_strict_model_and_exported_schema_round_trip(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam")
    payload = episode.model_dump(mode="json", exclude_none=False)
    round_trip = CanonicalEpisode.model_validate_json(episode.model_dump_json(), strict=True)
    assert round_trip == episode
    schema = CanonicalEpisode.model_json_schema(mode="validation")
    Draft202012Validator.check_schema(schema)
    validator = cast(Validator, Draft202012Validator(schema))
    validator.validate(payload)


def test_extra_fields_are_forbidden(fixture_root: Path) -> None:
    payload = canonical_fixture(fixture_root / "xlam", "xlam").model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        CanonicalEpisode.model_validate(payload, strict=True)


def test_call_order_changes_behavior_and_routes_unknown_topology_to_review(
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
    left = canonicalize(bronze, left_adapted, run_event_id="run_fixture")
    right = canonicalize(bronze, right_adapted, run_event_id="run_fixture")
    assert ordered_behavior_fingerprint(left) != ordered_behavior_fingerprint(right)
    assert compare_behavior(left, right) is BehaviorComparison.ORDER_AMBIGUITY_REVIEW


def test_invalid_arguments_fail_canonical_state_machine(fixture_root: Path) -> None:
    root = fixture_root / "xlam"
    snapshot = register_source(
        root,
        dataset_namespace="fixture-xlam",
        source_revision="fixture-v1",
        license_id="test-only",
    )
    bronze = next(iter(ingest_snapshot(snapshot, root)))
    assert bronze.parsed_record is not None
    changed = deepcopy(bronze.parsed_record)
    calls = changed["tool_calls"]
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    calls[0]["arguments"] = {"city": 123}
    adapted = get_adapter("xlam").adapt(changed)
    with pytest.raises(CanonicalizationError, match="Tool arguments violate"):
        canonicalize(bronze, adapted, run_event_id="run_fixture")
