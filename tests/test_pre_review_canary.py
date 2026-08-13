from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from tests.helpers import canonical_fixture
from toolcall_tr.audit import ExactConflictAudit, audit_exact_conflicts
from toolcall_tr.cli import app
from toolcall_tr.field_policy import (
    SegmentTranslation,
    extract_leaf_segments,
    load_field_policy,
    merge_translated_segments,
)
from toolcall_tr.hashing import JsonValue, canonical_bytes, sha256_jcs, stable_id
from toolcall_tr.jsonio import iter_jsonl, write_jsonl
from toolcall_tr.live_evaluation import LiveEvaluationInput
from toolcall_tr.models import CanonicalEpisode
from toolcall_tr.operational_translation import OperationalTranslationResult
from toolcall_tr.pre_review_canary import (
    PreReviewCanaryError,
    PreReviewCanaryManifest,
    prepare_pre_review_canary,
    prepare_pre_review_evaluation_inputs,
    select_pre_review_canary,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


def _episode(
    base: CanonicalEpisode,
    identity: str,
    *,
    user_text: str,
    target_text: str,
) -> CanonicalEpisode:
    payload = cast(dict[str, JsonValue], base.model_dump(mode="json", exclude_none=False))
    payload["episode_id"] = f"ep_{identity * 64}"
    conversation = payload["conversation"]
    assert isinstance(conversation, list)
    assert isinstance(conversation[0], dict)
    assert isinstance(conversation[-1], dict)
    conversation[0]["content"] = user_text
    conversation[-1]["content"] = target_text
    return CanonicalEpisode.model_validate_json(canonical_bytes(payload), strict=True)


def _audit_path(path: Path, audit: ExactConflictAudit) -> Path:
    path.write_bytes(canonical_bytes(audit) + b"\n")
    return path


def _translation_result(
    episode: CanonicalEpisode,
    *,
    policy_path: Path,
) -> OperationalTranslationResult:
    policy = load_field_policy(policy_path)
    extraction = extract_leaf_segments(episode, policy)
    translated = merge_translated_segments(
        episode,
        policy,
        extraction,
        [
            SegmentTranslation(segment_id=segment.segment_id, target_text=f"Türkçe {index}")
            for index, segment in enumerate(extraction.segments, start=1)
        ],
    )
    leaf_ids = [f"leaftr_{index:064x}" for index in range(1, len(extraction.segments) + 1)]
    body = {
        "schema_version": "operational-translation-result-0.1.0",
        "episode_id": episode.episode_id,
        "input_variant_id": episode.variant_id,
        "field_policy_sha256": sha256_jcs(policy),
        "prompt_id": f"prompt_{'a' * 64}",
        "status": "translated",
        "leaf_result_ids": leaf_ids,
        "translated_episode": translated.model_dump(mode="json", exclude_none=False),
        "promotion": "not_eligible",
    }
    return OperationalTranslationResult(
        result_id=stable_id("trresult", body),
        episode_id=episode.episode_id,
        input_variant_id=episode.variant_id,
        field_policy_sha256=sha256_jcs(policy),
        prompt_id=f"prompt_{'a' * 64}",
        status="translated",
        leaf_result_ids=leaf_ids,
        translated_episode=translated,
    )


def test_pre_review_canary_is_deterministic_and_keeps_human_as_final_gate(
    fixture_root: Path,
) -> None:
    base = canonical_fixture(fixture_root / "no_tool", "no_tool", 2)
    episodes = [
        _episode(base, "1", user_text="Need a short weather summary.", target_text="Clear."),
        _episode(base, "2", user_text="Need a short train summary.", target_text="On time."),
        _episode(base, "3", user_text="Need a short hotel summary.", target_text="Available."),
    ]
    policy = load_field_policy(ROOT / "configs" / "field_policy.toml")
    audit = audit_exact_conflicts(episodes)

    first_members, first_episodes = select_pre_review_canary(
        episodes,
        [audit],
        field_policy=policy,
        requested_episode_count=3,
        max_translatable_segments=6,
    )
    second_members, second_episodes = select_pre_review_canary(
        reversed(episodes),
        [audit],
        field_policy=policy,
        requested_episode_count=3,
        max_translatable_segments=6,
    )

    assert first_members == second_members
    assert first_episodes == second_episodes
    assert all(member.translatable_segments == 2 for member in first_members)
    assert [member.rank for member in first_members] == [1, 2, 3]


def test_pre_review_canary_excludes_every_exact_conflict_member(fixture_root: Path) -> None:
    base = canonical_fixture(fixture_root / "no_tool", "no_tool", 2)
    conflicting_left = _episode(
        base,
        "4",
        user_text="Same context must remain for the conflict.",
        target_text="First answer.",
    )
    conflicting_right = _episode(
        base,
        "5",
        user_text="Same context must remain for the conflict.",
        target_text="Second answer.",
    )
    safe = _episode(
        base,
        "6",
        user_text="A separate safe canary context.",
        target_text="Safe answer.",
    )
    policy = load_field_policy(ROOT / "configs" / "field_policy.toml")
    audit = audit_exact_conflicts([conflicting_left, conflicting_right, safe])

    members, selected = select_pre_review_canary(
        [conflicting_left, conflicting_right, safe],
        [audit],
        field_policy=policy,
        requested_episode_count=1,
        max_translatable_segments=2,
    )

    assert [member.episode_id for member in members] == [safe.episode_id]
    assert selected == [safe]
    with pytest.raises(PreReviewCanaryError, match="insufficient conflict-free"):
        select_pre_review_canary(
            [conflicting_left, conflicting_right, safe],
            [audit],
            field_policy=policy,
            requested_episode_count=2,
            max_translatable_segments=4,
        )


def test_canary_prepare_cli_publishes_idempotent_non_promoting_artifacts(
    fixture_root: Path, tmp_path: Path
) -> None:
    base = canonical_fixture(fixture_root / "no_tool", "no_tool", 2)
    episodes = [
        _episode(base, "7", user_text="Canary user one.", target_text="Canary reply one."),
        _episode(base, "8", user_text="Canary user two.", target_text="Canary reply two."),
    ]
    input_root = tmp_path / "input"
    input_root.mkdir()
    input_path = input_root / "canonical.jsonl"
    audit_path = _audit_path(input_root / "audit.json", audit_exact_conflicts(episodes))
    output = tmp_path / "canary-output"
    write_jsonl(
        input_path,
        [episode.model_dump(mode="json", exclude_none=False) for episode in episodes],
    )
    arguments = [
        "canary",
        "prepare",
        str(input_path),
        "--conflict-audit",
        str(audit_path),
        "--field-policy",
        str(ROOT / "configs" / "field_policy.toml"),
        "--output",
        str(output),
        "--episodes",
        "2",
        "--max-segments",
        "4",
    ]

    first = RUNNER.invoke(app, arguments)
    second = RUNNER.invoke(app, arguments)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    canary_path = next((output / "canaries").glob("canary_*.json"))
    manifest = PreReviewCanaryManifest.model_validate_json(
        canary_path.read_text(encoding="utf-8"), strict=True
    )
    assert manifest.promotion == "not_eligible"
    assert manifest.human_review == "required_for_final_acceptance"
    rows = list(iter_jsonl(next((output / "canonical").glob("*.jsonl"))))
    assert len(rows) == 2


def test_canary_rejects_an_output_root_inside_any_input_tree(
    fixture_root: Path, tmp_path: Path
) -> None:
    episode = _episode(
        canonical_fixture(fixture_root / "no_tool", "no_tool", 2),
        "a",
        user_text="Boundary test source.",
        target_text="Boundary test target.",
    )
    input_root = tmp_path / "input"
    input_root.mkdir()
    input_path = input_root / "canonical.jsonl"
    audit_path = _audit_path(input_root / "audit.json", audit_exact_conflicts([episode]))
    write_jsonl(input_path, [episode.model_dump(mode="json", exclude_none=False)])

    with pytest.raises(PreReviewCanaryError, match="disjoint"):
        prepare_pre_review_canary(
            [input_path],
            [audit_path],
            input_root / "derived",
            field_policy=load_field_policy(ROOT / "configs" / "field_policy.toml"),
            requested_episode_count=1,
            max_translatable_segments=2,
        )


def test_canary_evaluation_inputs_require_exact_host_merged_translation(
    fixture_root: Path, tmp_path: Path
) -> None:
    episode = _episode(
        canonical_fixture(fixture_root / "no_tool", "no_tool", 2),
        "9",
        user_text="Evaluate the pre-review source leaf.",
        target_text="Evaluate the pre-review target leaf.",
    )
    input_root = tmp_path / "input"
    input_root.mkdir()
    canonical_path = input_root / "canonical.jsonl"
    results_path = input_root / "results.jsonl"
    output = tmp_path / "evaluation-inputs"
    policy_path = ROOT / "configs" / "field_policy.toml"
    result = _translation_result(episode, policy_path=policy_path)
    write_jsonl(canonical_path, [episode.model_dump(mode="json", exclude_none=False)])
    write_jsonl(results_path, [result.model_dump(mode="json", exclude_none=False)])

    manifest = prepare_pre_review_evaluation_inputs(
        canonical_path,
        results_path,
        output,
        field_policy=load_field_policy(policy_path),
    )

    rows = [
        LiveEvaluationInput.model_validate(row, strict=True)
        for row in iter_jsonl(output / manifest.artifacts[0].relative_path)
    ]
    assert len(rows) == 2
    assert {row.evidence.target_excerpt for row in rows} == {"Türkçe 1", "Türkçe 2"}
