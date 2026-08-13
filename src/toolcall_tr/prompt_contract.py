"""Immutable, local prompt-bundle contract without any provider integration."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from toolcall_tr.hashing import sha256_bytes, stable_id
from toolcall_tr.models import NonEmptyStr, StrictModel

PromptLayerName = Literal[
    "core_contract",
    "field_policy",
    "fidelity_contract",
    "protected_span_contract",
    "terminology_protocol",
    "output_contract",
    "contrastive_examples",
]
PromptId = Annotated[str, Field(pattern=r"^prompt_[0-9a-f]{64}$")]

_REQUIRED_LAYERS: tuple[PromptLayerName, ...] = (
    "core_contract",
    "field_policy",
    "fidelity_contract",
    "protected_span_contract",
    "terminology_protocol",
    "output_contract",
)


class PromptContractError(ValueError):
    """Raised for malformed immutable prompt bundles."""


class PromptLayer(StrictModel):
    name: PromptLayerName
    version: NonEmptyStr
    content: NonEmptyStr
    content_sha256: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def validate_hash(self) -> PromptLayer:
        if self.content_sha256 != sha256_bytes(self.content.encode("utf-8")):
            raise ValueError("prompt layer content hash does not match exact UTF-8 bytes")
        return self


class PromptBundle(StrictModel):
    """Fixed-order, content-addressed system-prompt layers for a future adapter."""

    schema_version: Literal["prompt-bundle-0.1.0"] = "prompt-bundle-0.1.0"
    prompt_id: PromptId
    prompt_version: NonEmptyStr
    output_schema_version: Literal["translation-response-0.1.0"]
    layers: list[PromptLayer]

    @model_validator(mode="after")
    def validate_bundle(self) -> PromptBundle:
        names = [layer.name for layer in self.layers]
        if names[: len(_REQUIRED_LAYERS)] != list(_REQUIRED_LAYERS):
            raise ValueError("prompt layers must begin with the required fixed order")
        if not (
            len(names) == len(_REQUIRED_LAYERS)
            or names == [*_REQUIRED_LAYERS, "contrastive_examples"]
        ):
            raise ValueError(
                "only one optional contrastive_examples layer may follow required layers"
            )
        body = self.model_dump(mode="json", exclude={"prompt_id"})
        if self.prompt_id != stable_id("prompt", body):
            raise ValueError("prompt ID does not match deterministic bundle content")
        return self

    @property
    def system_text(self) -> str:
        """Return exact compiled text; it is data only and is never sent here."""
        return "\n\n".join(layer.content for layer in self.layers)


def make_prompt_layer(*, name: PromptLayerName, version: str, content: str) -> PromptLayer:
    return PromptLayer(
        name=name,
        version=version,
        content=content,
        content_sha256=sha256_bytes(content.encode("utf-8")),
    )


def build_prompt_bundle(
    *, prompt_version: str, layers: list[PromptLayer]
) -> PromptBundle:
    body = {
        "schema_version": "prompt-bundle-0.1.0",
        "prompt_version": prompt_version,
        "output_schema_version": "translation-response-0.1.0",
        "layers": [layer.model_dump(mode="json") for layer in layers],
    }
    return PromptBundle(
        prompt_id=stable_id("prompt", body),
        prompt_version=prompt_version,
        output_schema_version="translation-response-0.1.0",
        layers=layers,
    )


def load_prompt_bundle(path: Path) -> PromptBundle:
    """Load TOML through JCS so strict literals do not accept plain strings."""
    with path.open("rb") as handle:
        parsed = cast(dict[str, object], tomllib.load(handle))
    raw_layers = parsed.pop("layers", None)
    if not isinstance(raw_layers, Mapping):
        raise PromptContractError("prompt bundle requires a [layers] TOML table")
    layer_table = cast(Mapping[str, object], raw_layers)
    if set(layer_table) - set(_REQUIRED_LAYERS) - {"contrastive_examples"}:
        raise PromptContractError("prompt bundle contains an unknown layer")
    layers: list[PromptLayer] = []
    ordered_names: tuple[PromptLayerName, ...] = (*_REQUIRED_LAYERS, "contrastive_examples")
    for name in ordered_names:
        raw = layer_table.get(name)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise PromptContractError(f"prompt layer {name} must be a TOML table")
        layer = cast(Mapping[str, object], raw)
        version = layer.get("version")
        content = layer.get("content")
        if not isinstance(version, str) or not isinstance(content, str):
            raise PromptContractError(f"prompt layer {name} requires string version and content")
        layers.append(make_prompt_layer(name=name, version=version, content=content))
    prompt_version = parsed.get("prompt_version")
    if not isinstance(prompt_version, str):
        raise PromptContractError("prompt bundle requires prompt_version")
    return build_prompt_bundle(prompt_version=prompt_version, layers=layers)
