from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from toolcall_tr.phase4_config import Phase4Config, load_phase4_config


def valid_payload() -> dict[str, object]:
    return {
        "schema_version": "phase4-config-0.1.0",
        "ngram_size": 3,
        "near_duplicate_candidate_threshold": 0.82,
        "automatic_similarity_drop": False,
        "required_source_valid_membership": 400,
        "selection_tiers": [30, 100, 250, 400],
        "semantic_judge_enabled": False,
        "human_adjudication_required": True,
    }


def test_checked_in_phase4_config_is_offline_and_human_gated() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_phase4_config(root / "configs" / "phase4.toml")
    assert config.selection_tiers == [30, 100, 250, 400]
    assert config.automatic_similarity_drop is False
    assert config.semantic_judge_enabled is False
    assert config.human_adjudication_required is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("automatic_similarity_drop", True),
        ("semantic_judge_enabled", True),
        ("human_adjudication_required", False),
        ("selection_tiers", [30, 100, 400]),
        ("required_source_valid_membership", 401),
    ],
)
def test_unsafe_phase4_policy_fails_closed(field: str, value: object) -> None:
    payload = valid_payload()
    payload[field] = value
    with pytest.raises(ValidationError):
        Phase4Config.model_validate(payload, strict=True)
