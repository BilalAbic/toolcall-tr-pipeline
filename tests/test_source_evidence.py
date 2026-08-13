from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.helpers import canonical_fixture
from toolcall_tr.source_evidence import (
    AcceptableBehavior,
    ArgumentEvidenceInput,
    build_source_evidence,
)


def explicit_city(call_id: str = "call_001") -> ArgumentEvidenceInput:
    return ArgumentEvidenceInput(
        call_id=call_id,
        argument_pointer="/city",
        origin="explicit_user",
        evidence_pointers=["/conversation/0/content"],
        transformation_id=None,
        input_pointers=[],
    )


def test_exact_explicit_evidence_passes_without_text_inference(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam", 0)
    evidence = build_source_evidence(episode, [explicit_city()])
    assert evidence.pass1_result == "deterministic_pass"
    assert len(evidence.argument_provenance) == 1
    assert evidence.argument_provenance[0].origin == "explicit_user"
    assert evidence.claims[0].status == "supported"
    assert evidence.acceptable_behaviors[0].authority == "source_explicit"
    assert evidence.judge_verdict == "not_run"
    assert evidence.human_verdict == "source_review"


def test_missing_evidence_is_accounted_as_unknown_and_never_auto_valid(
    fixture_root: Path,
) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam", 1)
    evidence = build_source_evidence(episode, [explicit_city("call_001")])
    assert evidence.pass1_result == "needs_semantic_review"
    assert [item.origin for item in evidence.argument_provenance] == [
        "explicit_user",
        "unknown",
    ]
    assert evidence.diagnostics[0].code == "SOURCE_ARG_NOT_GROUNDED"


def test_must_not_infer_is_a_deterministic_failure(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam", 0)
    forbidden = ArgumentEvidenceInput(
        call_id="call_001",
        argument_pointer="/city",
        origin="must_not_infer",
        evidence_pointers=[],
        transformation_id=None,
        input_pointers=[],
    )
    evidence = build_source_evidence(episode, [forbidden])
    assert evidence.pass1_result == "deterministic_fail"
    assert evidence.claims[0].status == "unsupported"


def test_wrong_role_pointer_fails_closed(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam", 0)
    invalid = explicit_city()
    payload = invalid.model_dump(mode="json")
    payload["evidence_pointers"] = ["/tools/0/function/name"]
    evidence = build_source_evidence(
        episode, [ArgumentEvidenceInput.model_validate(payload, strict=True)]
    )
    assert evidence.pass1_result == "deterministic_fail"
    assert evidence.argument_provenance[0].origin == "unknown"


def test_extra_or_duplicate_evidence_is_rejected(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam", 0)
    with pytest.raises(ValueError, match="does not match"):
        build_source_evidence(
            episode,
            [
                ArgumentEvidenceInput(
                    call_id="call_999",
                    argument_pointer="/city",
                    origin="unknown",
                    evidence_pointers=[],
                    transformation_id=None,
                    input_pointers=[],
                )
            ],
        )
    with pytest.raises(ValueError, match="duplicate argument evidence"):
        build_source_evidence(episode, [explicit_city(), explicit_city()])


def test_model_cannot_authorize_an_acceptable_behavior() -> None:
    with pytest.raises(ValidationError):
        AcceptableBehavior.model_validate(
            {"action": "tool_call", "tool_ids": [], "authority": "model_judge"},
            strict=True,
        )


def test_no_tool_episode_has_zero_argument_provenance(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "no_tool", "no_tool", 0)
    evidence = build_source_evidence(episode, [])
    assert evidence.pass1_result == "deterministic_pass"
    assert evidence.argument_provenance == []
    assert evidence.acceptable_behaviors[0].tool_ids == []
