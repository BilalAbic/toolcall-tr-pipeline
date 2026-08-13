from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.helpers import canonical_fixture
from toolcall_tr.hashing import sha256_bytes, sha256_jcs
from toolcall_tr.models import CanonicalEpisode
from toolcall_tr.render_contract import (
    CharacterRange,
    RenderCandidate,
    RenderConfig,
    RenderContractError,
    TargetPayloadRange,
    TokenizedText,
    build_render_config,
    final_assistant_payload,
    render_with_loss_mask,
    technical_structure_sha256,
)


def config() -> RenderConfig:
    return build_render_config(
        renderer_id="test-renderer",
        renderer_revision="test-revision-1",
        chat_template_id="test-template",
        chat_template_sha256=f"sha256:{'a' * 64}",
        tokenizer_id="test-tokenizer",
        tokenizer_revision="test-revision-1",
        max_rendered_characters=10_000,
        max_tokens=10_000,
    )


def candidate_for(episode: CanonicalEpisode, render_config: RenderConfig) -> RenderCandidate:
    target = episode.conversation[-1]
    payload = final_assistant_payload(target)
    rendered = f"<history>{payload}</history>"
    start = rendered.index(payload)
    payload_range = TargetPayloadRange(
        character_range=CharacterRange(start=start, end=start + len(payload)),
        rendered_payload_sha256=sha256_bytes(payload.encode("utf-8")),
        assistant_message_sha256=episode_target_hash(episode),
    )
    return RenderCandidate(
        render_config_id=render_config.render_config_id,
        episode_id=episode.episode_id,
        variant_id=episode.variant_id,
        technical_structure_sha256=technical_structure_sha256(
            conversation=episode.conversation,
            tools=episode.tools,
            target_message_index=episode.annotations.target_message_index,
        ),
        rendered_text=rendered,
        rendered_conversation=episode.conversation,
        rendered_tools=episode.tools,
        target_message_index=episode.annotations.target_message_index,
        target_payload_ranges=[payload_range],
        truncated=False,
    )


def episode_target_hash(episode: CanonicalEpisode) -> str:
    return sha256_jcs(episode.conversation[-1].model_dump(mode="json", exclude_none=False))


@dataclass
class StaticRenderer:
    candidate: RenderCandidate

    def render(
        self, *, episode: CanonicalEpisode, config: RenderConfig
    ) -> RenderCandidate:
        return self.candidate


class CharacterTokenizer:
    def encode(self, text: str) -> TokenizedText:
        return TokenizedText(
            token_ids=[ord(character) for character in text],
            offsets=[CharacterRange(start=index, end=index + 1) for index in range(len(text))],
        )


def test_render_masks_only_the_final_assistant_payload(
    fixture_root: Path,
) -> None:
    episode = canonical_fixture(fixture_root / "no_tool", "no_tool")
    render_config = config()
    candidate = candidate_for(episode, render_config)

    result = render_with_loss_mask(
        episode=episode,
        config=render_config,
        renderer=StaticRenderer(candidate),
        tokenizer=CharacterTokenizer(),
    )

    assert result.render.render_config_id == render_config.render_config_id
    assert result.render.target_payload_ranges == candidate.target_payload_ranges
    target = result.loss_mask.target_token_ranges[0]
    assert result.loss_mask.labels[: target.start_token] == [-100] * target.start_token
    assert (
        result.loss_mask.labels[target.start_token : target.end_token]
        == result.loss_mask.input_ids[target.start_token : target.end_token]
    )
    assert result.loss_mask.labels[target.end_token :] == [-100] * (
        len(result.loss_mask.labels) - target.end_token
    )


@pytest.mark.parametrize("mutation", ["absent", "multiple", "truncated"])
def test_render_fails_closed_without_one_complete_target_range(
    fixture_root: Path, mutation: str
) -> None:
    episode = canonical_fixture(fixture_root / "no_tool", "no_tool")
    render_config = config()
    candidate = candidate_for(episode, render_config)
    if mutation == "absent":
        candidate = candidate.model_copy(update={"target_payload_ranges": []})
    elif mutation == "multiple":
        candidate = candidate.model_copy(
            update={"target_payload_ranges": candidate.target_payload_ranges * 2}
        )
    else:
        candidate = candidate.model_copy(update={"truncated": True})

    with pytest.raises(RenderContractError, match=r"target.*payload range|truncated"):
        render_with_loss_mask(
            episode=episode,
            config=render_config,
            renderer=StaticRenderer(candidate),
            tokenizer=CharacterTokenizer(),
        )


def test_render_rejects_technical_structure_mutation(fixture_root: Path) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam")
    render_config = config()
    candidate = candidate_for(episode, render_config)
    changed_target = episode.conversation[-1].model_copy(update={"tool_calls": []})
    changed_conversation = [*episode.conversation[:-1], changed_target]
    changed = candidate.model_copy(
        update={
            "rendered_conversation": changed_conversation,
            "technical_structure_sha256": technical_structure_sha256(
                conversation=changed_conversation,
                tools=episode.tools,
                target_message_index=episode.annotations.target_message_index,
            ),
        }
    )

    with pytest.raises(RenderContractError, match="mutated conversation structure"):
        render_with_loss_mask(
            episode=episode,
            config=render_config,
            renderer=StaticRenderer(changed),
            tokenizer=CharacterTokenizer(),
        )


def test_render_rejects_target_boundary_that_would_cut_a_token(
    fixture_root: Path,
) -> None:
    episode = canonical_fixture(fixture_root / "no_tool", "no_tool")
    render_config = config()
    candidate = candidate_for(episode, render_config)

    class WholeTextTokenizer:
        def encode(self, text: str) -> TokenizedText:
            return TokenizedText(
                token_ids=[1], offsets=[CharacterRange(start=0, end=len(text))]
            )

    with pytest.raises(RenderContractError, match="token boundaries"):
        render_with_loss_mask(
            episode=episode,
            config=render_config,
            renderer=StaticRenderer(candidate),
            tokenizer=WholeTextTokenizer(),
        )
