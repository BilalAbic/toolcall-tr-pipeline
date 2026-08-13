from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from tests.helpers import canonical_fixture
from toolcall_tr.field_policy import (
    ArgumentPathPolicy,
    FieldAction,
    FieldPolicy,
    FieldPolicyError,
    SegmentTranslation,
    extract_leaf_segments,
    load_field_policy,
    merge_translated_segments,
)
from toolcall_tr.hashing import JsonValue, canonical_bytes, sha256_jcs


def policy(*, descriptions: FieldAction = FieldAction.COPY_EXACT) -> FieldPolicy:
    return FieldPolicy(
        policy_version="field-policy-test-0.1.0",
        tool_description_action=descriptions,
        parameter_description_action=descriptions,
        argument_policies=[
            ArgumentPathPolicy(
                tool_name="get_weather",
                argument_pointer="/city",
                action=FieldAction.COPY_EXACT,
            )
        ],
    )


def test_checked_in_field_policy_translates_documentation_but_copies_all_arguments() -> None:
    root = Path(__file__).resolve().parents[1]
    field_policy = load_field_policy(root / "configs" / "field_policy.toml")
    assert field_policy.tool_description_action is FieldAction.TRANSLATE
    assert field_policy.parameter_description_action is FieldAction.TRANSLATE
    assert field_policy.argument_action("qrcode", "/data") is FieldAction.COPY_EXACT
    assert field_policy.argument_action("reverse_input", "/input_value") is FieldAction.COPY_EXACT
    assert field_policy.argument_action("any_tool", "/any/nested/path") is FieldAction.COPY_EXACT


def test_extracts_only_user_and_assistant_content_for_no_tool_episode(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "no_tool", "no_tool", index=0)
    extraction = extract_leaf_segments(episode, policy())
    assert [segment.json_pointer for segment in extraction.segments] == [
        "/conversation/0/content",
        "/conversation/1/content",
    ]
    assert extraction.input_variant_id == episode.variant_id
    assert all(segment.segment_id.startswith("seg_") for segment in extraction.segments)


def test_unresolved_documentation_policy_fails_closed(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam")
    with pytest.raises(FieldPolicyError, match="translate_if_natural_language"):
        extract_leaf_segments(
            episode,
            policy(descriptions=FieldAction.TRANSLATE_IF_NATURAL_LANGUAGE),
        )


def test_uncovered_textual_argument_fails_closed(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam")
    with pytest.raises(FieldPolicyError, match="uncovered textual argument path"):
        extract_leaf_segments(
            episode,
            FieldPolicy(
                policy_version="field-policy-test-0.1.0",
                tool_description_action=FieldAction.COPY_EXACT,
                parameter_description_action=FieldAction.COPY_EXACT,
                argument_policies=[],
            ),
        )


def test_global_copy_fallback_unlocks_tool_documentation_without_exposing_arguments(
    fixture_root: Path,
) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam")
    extraction = extract_leaf_segments(
        episode,
        FieldPolicy(
            policy_version="field-policy-test-0.2.0",
            tool_description_action=FieldAction.TRANSLATE,
            parameter_description_action=FieldAction.TRANSLATE,
            argument_policies=[
                ArgumentPathPolicy(
                    tool_name="*",
                    argument_pointer="/*",
                    action=FieldAction.COPY_EXACT,
                )
            ],
        ),
    )

    pointers = [segment.json_pointer for segment in extraction.segments]
    assert "/conversation/1/tool_calls/0/function/arguments/city" not in pointers
    assert "/tools/0/function/description" in pointers
    assert "/tools/0/function/parameters/properties/city/description" in pointers


def test_global_argument_fallback_cannot_translate_or_partially_match() -> None:
    with pytest.raises(ValueError, match="must copy_exact"):
        FieldPolicy(
            policy_version="field-policy-test-0.2.0",
            argument_policies=[
                ArgumentPathPolicy(
                    tool_name="*",
                    argument_pointer="/*",
                    action=FieldAction.TRANSLATE,
                )
            ],
        )
    with pytest.raises(ValueError, match="must be exactly"):
        FieldPolicy(
            policy_version="field-policy-test-0.2.0",
            argument_policies=[
                ArgumentPathPolicy(
                    tool_name="*",
                    argument_pointer="/city",
                    action=FieldAction.COPY_EXACT,
                )
            ],
        )


def test_unsafe_argument_cannot_be_translated_even_with_explicit_policy(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam")
    unsafe = FieldPolicy(
        policy_version="field-policy-test-0.1.0",
        tool_description_action=FieldAction.COPY_EXACT,
        parameter_description_action=FieldAction.COPY_EXACT,
        argument_policies=[
            ArgumentPathPolicy(
                tool_name="get_weather",
                argument_pointer="/city",
                action=FieldAction.TRANSLATE,
            )
        ],
    )
    # A city has no format/pattern constraint in this fixture and the explicit
    # policy may legitimately treat it as free text.
    extraction = extract_leaf_segments(episode, unsafe)
    assert any(segment.json_pointer.endswith("/arguments/city") for segment in extraction.segments)

    payload = cast(
        dict[str, JsonValue], deepcopy(episode.model_dump(mode="json", exclude_none=False))
    )
    conversation = payload["conversation"]
    assert isinstance(conversation, list)
    final_message = cast(JsonValue, conversation[-1])
    assert isinstance(final_message, dict)
    calls = cast(JsonValue, final_message["tool_calls"])
    assert isinstance(calls, list)
    first_call = cast(JsonValue, calls[0])
    assert isinstance(first_call, dict)
    function = cast(JsonValue, first_call["function"])
    assert isinstance(function, dict)
    arguments = cast(JsonValue, function["arguments"])
    assert isinstance(arguments, dict)
    arguments["city"] = "https://private.invalid/path"
    payload["variant_id"] = sha256_jcs(
        {
            "episode_id": payload["episode_id"],
            "conversation": payload["conversation"],
            "tools": payload["tools"],
            "annotations": payload["annotations"],
        }
    )
    from toolcall_tr.models import CanonicalEpisode

    changed = CanonicalEpisode.model_validate_json(canonical_bytes(payload), strict=True)
    with pytest.raises(FieldPolicyError, match="URL or path"):
        extract_leaf_segments(changed, unsafe)


def test_host_merge_changes_only_segments_and_derived_hashes(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "no_tool", "no_tool", index=0)
    field_policy = policy()
    extraction = extract_leaf_segments(episode, field_policy)
    merged = merge_translated_segments(
        episode,
        field_policy,
        extraction,
        [
            SegmentTranslation(
                segment_id=segment.segment_id,
                target_text=f"TR: {segment.source_text}",
            )
            for segment in extraction.segments
        ],
    )
    assert merged.episode_id == episode.episode_id
    assert merged.source_episode_fingerprint == episode.source_episode_fingerprint
    assert merged.parent_variant_id == episode.variant_id
    assert merged.variant_id != episode.variant_id
    assert merged.conversation[0].content == "TR: Book me a flight."
    assert merged.conversation[1].content == "TR: Which destination and date should I use?"
    assert merged.provenance == episode.provenance
    assert merged.annotations == episode.annotations


def test_merge_requires_exact_coverage_and_is_idempotent_for_same_text(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "no_tool", "no_tool", index=2)
    field_policy = policy()
    extraction = extract_leaf_segments(episode, field_policy)
    with pytest.raises(FieldPolicyError, match="cover exactly"):
        merge_translated_segments(episode, field_policy, extraction, [])
    same = merge_translated_segments(
        episode,
        field_policy,
        extraction,
        [
            SegmentTranslation(segment_id=item.segment_id, target_text=item.source_text)
            for item in extraction.segments
        ],
    )
    assert same == episode
