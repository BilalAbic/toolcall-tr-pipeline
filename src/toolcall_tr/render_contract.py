"""Offline render and loss-mask contract for final assistant supervision.

This module deliberately knows nothing about a model hub, a tokenizer package,
or a network provider.  A caller injects a renderer and a tokenizer through
small protocols.  Their output is accepted only after this host-side contract
proves that the exact canonical episode, its final assistant target, and one
untruncated character/token range still agree.
"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from toolcall_tr.hashing import canonical_bytes, sha256_bytes, sha256_jcs, stable_id
from toolcall_tr.models import (
    CanonicalEpisode,
    CanonicalTool,
    EpisodeId,
    Message,
    NonEmptyStr,
    Role,
    Sha256,
    StrictModel,
)

RenderConfigId = Annotated[str, Field(pattern=r"^rendercfg_[0-9a-f]{64}$")]
RenderId = Annotated[str, Field(pattern=r"^render_[0-9a-f]{64}$")]
LossMaskId = Annotated[str, Field(pattern=r"^lossmask_[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]


class RenderContractError(ValueError):
    """Raised when an injected renderer or tokenizer breaks this local contract."""


class RenderConfig(StrictModel):
    """Pinned, content-addressed identity for one render implementation.

    The renderer and tokenizer are intentionally named and revision-pinned here
    rather than being discovered from a local cache or a remote model registry.
    ``truncation_policy`` has only a rejecting mode: a partial target must never
    become a trainable example.  The final payload is the JCS representation of
    the final canonical assistant message, which gives every implementation one
    unambiguous generic completion representation.
    """

    schema_version: Literal["render-config-0.1.0"] = "render-config-0.1.0"
    render_config_id: RenderConfigId
    renderer_id: NonEmptyStr
    renderer_revision: NonEmptyStr
    chat_template_id: NonEmptyStr
    chat_template_sha256: Sha256
    tokenizer_id: NonEmptyStr
    tokenizer_revision: NonEmptyStr
    max_rendered_characters: PositiveInt
    max_tokens: PositiveInt
    add_generation_prompt: Literal[False] = False
    truncation_policy: Literal["reject"] = "reject"

    @model_validator(mode="after")
    def validate_identity_and_limits(self) -> RenderConfig:
        body = self.model_dump(mode="json", exclude={"render_config_id"})
        if self.render_config_id != stable_id("rendercfg", body):
            raise ValueError("render config ID does not match deterministic content")
        return self


class CharacterRange(StrictModel):
    """A half-open Python Unicode code-point range in the rendered text."""

    start: NonNegativeInt
    end: PositiveInt

    @model_validator(mode="after")
    def validate_nonempty(self) -> CharacterRange:
        if self.end <= self.start:
            raise ValueError("character range must be nonempty and half-open")
        return self


class TargetPayloadRange(StrictModel):
    """One renderer-declared span containing the final assistant payload."""

    character_range: CharacterRange
    rendered_payload_sha256: Sha256
    assistant_message_sha256: Sha256


class RenderCandidate(StrictModel):
    """Untrusted, in-memory response supplied by an injected chat renderer.

    ``rendered_conversation`` and ``rendered_tools`` are deliberate echoes of
    the technical input used by the renderer.  They allow the host to reject a
    renderer that drops, reorders, or rewrites tool-call structure before any
    loss labels are emitted.
    """

    schema_version: Literal["render-candidate-0.1.0"] = "render-candidate-0.1.0"
    render_config_id: RenderConfigId
    episode_id: EpisodeId
    variant_id: Sha256
    technical_structure_sha256: Sha256
    rendered_text: str
    rendered_conversation: Annotated[list[Message], Field(min_length=1)]
    rendered_tools: list[CanonicalTool]
    target_message_index: NonNegativeInt
    target_payload_ranges: list[TargetPayloadRange]
    truncated: bool


class TokenizedText(StrictModel):
    """Tokenizer output with one positive-width offset for every input token.

    Offsets use the same Python Unicode code-point coordinates as
    :class:`CharacterRange`.  Special tokens with no source-text offset are
    intentionally outside this protocol: their placement would make precise
    target-only supervision ambiguous.
    """

    schema_version: Literal["tokenized-text-0.1.0"] = "tokenized-text-0.1.0"
    token_ids: Annotated[list[NonNegativeInt], Field(min_length=1)]
    offsets: Annotated[list[CharacterRange], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_cardinality(self) -> TokenizedText:
        if len(self.token_ids) != len(self.offsets):
            raise ValueError("token IDs and offsets must have identical cardinality")
        return self


class TokenRange(StrictModel):
    """A half-open token range paired with its exact character range."""

    start_token: NonNegativeInt
    end_token: PositiveInt
    character_range: CharacterRange

    @model_validator(mode="after")
    def validate_nonempty(self) -> TokenRange:
        if self.end_token <= self.start_token:
            raise ValueError("token range must be nonempty and half-open")
        return self


class RenderArtifact(StrictModel):
    """Verified, content-addressed rendering before labels are attached."""

    schema_version: Literal["render-artifact-0.1.0"] = "render-artifact-0.1.0"
    render_id: RenderId
    render_config_id: RenderConfigId
    episode_id: EpisodeId
    variant_id: Sha256
    technical_structure_sha256: Sha256
    rendered_text: str
    rendered_text_sha256: Sha256
    target_payload_ranges: Annotated[list[TargetPayloadRange], Field(min_length=1, max_length=1)]
    token_count: PositiveInt

    @model_validator(mode="after")
    def validate_identity_and_text_hash(self) -> RenderArtifact:
        if self.rendered_text_sha256 != sha256_bytes(self.rendered_text.encode("utf-8")):
            raise ValueError("rendered text hash does not match exact UTF-8 bytes")
        body = self.model_dump(mode="json", exclude={"render_id"})
        if self.render_id != stable_id("render", body):
            raise ValueError("render ID does not match deterministic content")
        return self


class LossMask(StrictModel):
    """Target-only labels for one verified render; ``-100`` means ignore."""

    schema_version: Literal["loss-mask-0.1.0"] = "loss-mask-0.1.0"
    loss_mask_id: LossMaskId
    render_id: RenderId
    input_ids: Annotated[list[NonNegativeInt], Field(min_length=1)]
    labels: Annotated[list[int], Field(min_length=1)]
    ignored_label: Literal[-100] = -100
    target_token_ranges: Annotated[list[TokenRange], Field(min_length=1, max_length=1)]

    @model_validator(mode="after")
    def validate_identity_and_labels(self) -> LossMask:
        if len(self.input_ids) != len(self.labels):
            raise ValueError("input IDs and labels must have identical cardinality")
        target_indexes = {
            index
            for item in self.target_token_ranges
            for index in range(item.start_token, item.end_token)
        }
        if not target_indexes or max(target_indexes) >= len(self.input_ids):
            raise ValueError("target token range is outside input IDs")
        for index, (input_id, label) in enumerate(zip(self.input_ids, self.labels, strict=True)):
            if index in target_indexes:
                if label != input_id:
                    raise ValueError("target token labels must equal their input IDs")
            elif label != self.ignored_label:
                raise ValueError("non-target token labels must use the ignored label")
        body = self.model_dump(mode="json", exclude={"loss_mask_id"})
        if self.loss_mask_id != stable_id("lossmask", body):
            raise ValueError("loss mask ID does not match deterministic content")
        return self


class SupervisedRender(StrictModel):
    """The bounded offline product of a renderer and a tokenizer."""

    schema_version: Literal["supervised-render-0.1.0"] = "supervised-render-0.1.0"
    render: RenderArtifact
    loss_mask: LossMask

    @model_validator(mode="after")
    def validate_links(self) -> SupervisedRender:
        if self.loss_mask.render_id != self.render.render_id:
            raise ValueError("loss mask must belong to the rendered artifact")
        if len(self.loss_mask.input_ids) != self.render.token_count:
            raise ValueError("loss mask token count must match the rendered artifact")
        return self


class RendererProtocol(Protocol):
    """Generic, injected rendering boundary with no model-loading surface."""

    def render(self, *, episode: CanonicalEpisode, config: RenderConfig) -> RenderCandidate: ...


class TokenizerProtocol(Protocol):
    """Generic, injected tokenizer boundary with exact source-text offsets."""

    def encode(self, text: str) -> TokenizedText: ...


def build_render_config(
    *,
    renderer_id: str,
    renderer_revision: str,
    chat_template_id: str,
    chat_template_sha256: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    max_rendered_characters: int,
    max_tokens: int,
) -> RenderConfig:
    """Build a strict render configuration without discovering any model assets."""
    body = {
        "schema_version": "render-config-0.1.0",
        "renderer_id": renderer_id,
        "renderer_revision": renderer_revision,
        "chat_template_id": chat_template_id,
        "chat_template_sha256": chat_template_sha256,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "max_rendered_characters": max_rendered_characters,
        "max_tokens": max_tokens,
        "add_generation_prompt": False,
        "truncation_policy": "reject",
    }
    return RenderConfig(
        render_config_id=stable_id("rendercfg", body),
        renderer_id=renderer_id,
        renderer_revision=renderer_revision,
        chat_template_id=chat_template_id,
        chat_template_sha256=chat_template_sha256,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        max_rendered_characters=max_rendered_characters,
        max_tokens=max_tokens,
    )


def technical_structure_sha256(
    *,
    conversation: list[Message],
    tools: list[CanonicalTool],
    target_message_index: int,
) -> str:
    """Fingerprint non-linguistic conversation and tool structure exactly."""
    return sha256_jcs(
        {
            "conversation": [
                {
                    "role": message.role.value,
                    "tool_calls": (
                        None
                        if message.tool_calls is None
                        else [
                            call.model_dump(mode="json", exclude_none=False)
                            for call in message.tool_calls
                        ]
                    ),
                    "images": (
                        None
                        if message.images is None
                        else [
                            image.model_dump(mode="json", exclude_none=False)
                            for image in message.images
                        ]
                    ),
                    "name": message.name,
                    "tool_call_id": message.tool_call_id,
                }
                for message in conversation
            ],
            "tools": [tool.model_dump(mode="json", exclude_none=False) for tool in tools],
            "target_message_index": target_message_index,
        }
    )


def _assistant_message_sha256(message: Message) -> str:
    return sha256_jcs(message.model_dump(mode="json", exclude_none=False))


def final_assistant_payload(message: Message) -> str:
    """Return the one generic serialisation that can carry every target shape.

    A text-only target and a tool-call target both remain lossless because this
    is the RFC 8785 JSON representation of the complete assistant message, not
    merely its optional ``content`` field.
    """
    if message.role is not Role.ASSISTANT:
        raise RenderContractError("only an assistant message can be a target payload")
    return canonical_bytes(message.model_dump(mode="json", exclude_none=False)).decode("utf-8")


def _validate_candidate(
    *, episode: CanonicalEpisode, config: RenderConfig, candidate: RenderCandidate
) -> TargetPayloadRange:
    target_index = episode.annotations.target_message_index
    target = episode.conversation[target_index]
    if target_index != len(episode.conversation) - 1 or target.role is not Role.ASSISTANT:
        raise RenderContractError("canonical target must be the final assistant message")
    if candidate.render_config_id != config.render_config_id:
        raise RenderContractError("renderer returned a different render config ID")
    if candidate.episode_id != episode.episode_id or candidate.variant_id != episode.variant_id:
        raise RenderContractError("renderer returned a different canonical episode identity")
    if candidate.truncated:
        raise RenderContractError("truncated renderer output cannot be supervised")
    if not candidate.rendered_text:
        raise RenderContractError("rendered text cannot be empty")
    if len(candidate.rendered_text) > config.max_rendered_characters:
        raise RenderContractError("rendered text exceeds max_rendered_characters")
    if candidate.target_message_index != target_index:
        raise RenderContractError("renderer changed the final assistant target index")
    if candidate.rendered_conversation != episode.conversation:
        raise RenderContractError("renderer mutated conversation structure or payload")
    if candidate.rendered_tools != episode.tools:
        raise RenderContractError("renderer mutated technical tool structure")

    expected_structure = technical_structure_sha256(
        conversation=episode.conversation,
        tools=episode.tools,
        target_message_index=target_index,
    )
    observed_structure = technical_structure_sha256(
        conversation=candidate.rendered_conversation,
        tools=candidate.rendered_tools,
        target_message_index=candidate.target_message_index,
    )
    if candidate.technical_structure_sha256 != observed_structure:
        raise RenderContractError("renderer technical structure fingerprint is inconsistent")
    if observed_structure != expected_structure:
        raise RenderContractError("renderer mutated technical structure")

    if len(candidate.target_payload_ranges) != 1:
        raise RenderContractError("exactly one final assistant target payload range is required")
    payload_range = candidate.target_payload_ranges[0]
    span = payload_range.character_range
    if span.end > len(candidate.rendered_text):
        raise RenderContractError("target payload range is outside rendered text")
    payload = candidate.rendered_text[span.start : span.end]
    expected_payload = final_assistant_payload(target)
    if payload != expected_payload:
        raise RenderContractError(
            "target payload range must contain the exact final assistant payload"
        )
    if payload_range.rendered_payload_sha256 != sha256_bytes(payload.encode("utf-8")):
        raise RenderContractError("target payload range hash does not match rendered text")
    if payload_range.assistant_message_sha256 != _assistant_message_sha256(target):
        raise RenderContractError(
            "target payload range does not identify the final assistant payload"
        )
    return payload_range


def _validate_token_offsets(text: str, tokenized: TokenizedText) -> None:
    """Require a complete, non-overlapping mapping from text to tokens."""
    expected_start = 0
    for offset in tokenized.offsets:
        if offset.start != expected_start or offset.end > len(text):
            raise RenderContractError(
                "token offsets must exactly and contiguously cover rendered text"
            )
        expected_start = offset.end
    if expected_start != len(text):
        raise RenderContractError("token offsets must cover all rendered text")


def _target_token_range(
    *, target: CharacterRange, tokenized: TokenizedText
) -> TokenRange:
    starts = [offset.start for offset in tokenized.offsets]
    ends = [offset.end for offset in tokenized.offsets]
    if target.start not in starts or target.end not in ends:
        raise RenderContractError("target payload range must align with complete token boundaries")
    start_token = starts.index(target.start)
    end_token = ends.index(target.end) + 1
    selected = tokenized.offsets[start_token:end_token]
    if not selected or selected[0].start != target.start or selected[-1].end != target.end:
        raise RenderContractError("target payload range is truncated by token boundaries")
    if any(
        offset.start < target.start or offset.end > target.end
        for offset in selected
    ):
        raise RenderContractError("target payload range is truncated by token boundaries")
    return TokenRange(
        start_token=start_token,
        end_token=end_token,
        character_range=target,
    )


def render_with_loss_mask(
    *,
    episode: CanonicalEpisode,
    config: RenderConfig,
    renderer: RendererProtocol,
    tokenizer: TokenizerProtocol,
) -> SupervisedRender:
    """Render one canonical episode and label only its final assistant payload.

    This function is intentionally bounded by ``config`` and uses neither a
    model identifier resolver nor an HTTP/client library.  Callers must bring
    their own already-local renderer and tokenizer implementations.
    """
    candidate = renderer.render(episode=episode, config=config)
    payload_range = _validate_candidate(episode=episode, config=config, candidate=candidate)

    tokenized = tokenizer.encode(candidate.rendered_text)
    if len(tokenized.token_ids) > config.max_tokens:
        raise RenderContractError("tokenized render exceeds max_tokens")
    _validate_token_offsets(candidate.rendered_text, tokenized)
    target_range = _target_token_range(
        target=payload_range.character_range,
        tokenized=tokenized,
    )

    render_body = {
        "schema_version": "render-artifact-0.1.0",
        "render_config_id": config.render_config_id,
        "episode_id": episode.episode_id,
        "variant_id": episode.variant_id,
        "technical_structure_sha256": candidate.technical_structure_sha256,
        "rendered_text": candidate.rendered_text,
        "rendered_text_sha256": sha256_bytes(candidate.rendered_text.encode("utf-8")),
        "target_payload_ranges": [payload_range.model_dump(mode="json")],
        "token_count": len(tokenized.token_ids),
    }
    render = RenderArtifact(
        render_id=stable_id("render", render_body),
        render_config_id=config.render_config_id,
        episode_id=episode.episode_id,
        variant_id=episode.variant_id,
        technical_structure_sha256=candidate.technical_structure_sha256,
        rendered_text=candidate.rendered_text,
        rendered_text_sha256=sha256_bytes(candidate.rendered_text.encode("utf-8")),
        target_payload_ranges=[payload_range],
        token_count=len(tokenized.token_ids),
    )
    labels = [
        token_id if target_range.start_token <= index < target_range.end_token else -100
        for index, token_id in enumerate(tokenized.token_ids)
    ]
    mask_body = {
        "schema_version": "loss-mask-0.1.0",
        "render_id": render.render_id,
        "input_ids": tokenized.token_ids,
        "labels": labels,
        "ignored_label": -100,
        "target_token_ranges": [target_range.model_dump(mode="json")],
    }
    loss_mask = LossMask(
        loss_mask_id=stable_id("lossmask", mask_body),
        render_id=render.render_id,
        input_ids=tokenized.token_ids,
        labels=labels,
        target_token_ranges=[target_range],
    )
    return SupervisedRender(render=render, loss_mask=loss_mask)


def build_supervised_render(
    *,
    episode: CanonicalEpisode,
    config: RenderConfig,
    renderer: RendererProtocol,
    tokenizer: TokenizerProtocol,
) -> SupervisedRender:
    """Named alias for :func:`render_with_loss_mask` for pipeline callers."""
    return render_with_loss_mask(
        episode=episode,
        config=config,
        renderer=renderer,
        tokenizer=tokenizer,
    )
