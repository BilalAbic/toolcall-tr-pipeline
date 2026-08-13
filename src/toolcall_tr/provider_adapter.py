"""Bounded, transport-injected adapter for Responses-style translation calls.

This module owns deterministic serialization and local response validation only.
It deliberately has no HTTP client, SDK import, retry policy, environment lookup,
or credential handling.  A caller must provide a transport implementation; tests
use a recording fake transport.  Consequently, adding this module alone cannot
make a network or model request.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol, cast

from toolcall_tr.config import PipelineConfig
from toolcall_tr.hashing import JsonValue, canonical_bytes, to_json_value
from toolcall_tr.prompt_contract import PromptBundle
from toolcall_tr.translation_contract import (
    TranslationContractError,
    TranslationRequest,
    TranslationResponse,
    validate_translation_response,
)


class ResponsesTransport(Protocol):
    """A deliberately credential-free boundary for a Responses-compatible client.

    Implementations receive a complete, canonical request body and return the raw
    Responses API JSON body.  This protocol does not prescribe HTTP: a production
    implementation, if separately approved, is responsible for credentials and
    endpoint-specific delivery outside this module.
    """

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes: ...


class ProviderAdapterError(RuntimeError):
    """Base error for provider-adapter configuration or wire-contract failures."""


class ProviderGateError(ProviderAdapterError):
    """Raised before transport use when a required egress gate is disabled."""


class ProviderConfigurationError(ProviderAdapterError):
    """Raised when the selected role does not have an endpoint."""


class ProviderResponseError(ProviderAdapterError):
    """Raised for malformed Responses envelopes or invalid structured output."""


def _response_schema() -> dict[str, JsonValue]:
    """Build a JSON-only schema for the strict local translation response model."""
    return cast(dict[str, JsonValue], to_json_value(TranslationResponse.model_json_schema()))


def serialize_responses_request(
    *, request: TranslationRequest, prompt: PromptBundle, model: str, temperature: float | None
) -> bytes:
    """Serialize the only Responses request shape accepted by this adapter.

    The leaf-only translation request is canonical JSON text rather than
    reconstructed source data.  The compiled immutable prompt is carried in a
    system message, and the local strict response schema is requested through
    the Responses ``text.format`` structured-output field.
    """
    body: dict[str, JsonValue] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": prompt.system_text}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": canonical_bytes(request).decode("utf-8"),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "translation_response",
                "strict": True,
                "schema": _response_schema(),
            }
        },
        "metadata": {"request_id": request.request_id, "prompt_id": prompt.prompt_id},
        "store": False,
    }
    if temperature is not None:
        body["temperature"] = temperature
    return canonical_bytes(body)


def _structured_output_text(raw_response: bytes) -> str:
    """Extract exactly one output-text item without exposing remote content in errors."""
    try:
        parsed = cast(object, json.loads(raw_response))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderResponseError("Responses transport returned invalid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ProviderResponseError("Responses transport returned a non-object envelope")
    envelope = cast(Mapping[str, object], parsed)
    output = envelope.get("output")
    if not isinstance(output, list):
        raise ProviderResponseError("Responses envelope is missing an output list")
    output_items = cast(list[object], output)

    texts: list[str] = []
    for raw_item in output_items:
        if not isinstance(raw_item, Mapping):
            continue
        item = cast(Mapping[str, object], raw_item)
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        content_parts = cast(list[object], content)
        for raw_part in content_parts:
            if not isinstance(raw_part, Mapping):
                continue
            part = cast(Mapping[str, object], raw_part)
            if part.get("type") != "output_text":
                continue
            text = part.get("text")
            if not isinstance(text, str):
                raise ProviderResponseError("Responses output_text must contain a string")
            texts.append(text)
    if len(texts) != 1:
        raise ProviderResponseError("Responses envelope must contain exactly one output_text")
    return texts[0]


class ResponsesTranslationAdapter:
    """Translate through an injected transport after closed-by-default gate checks."""

    def __init__(self, *, config: PipelineConfig, transport: ResponsesTransport) -> None:
        self._config = config
        self._transport = transport

    def translate(
        self, *, request: TranslationRequest, prompt: PromptBundle
    ) -> TranslationResponse:
        """Make one bounded transport invocation and validate the returned leaf response."""
        provider = self._config.providers
        if not provider.enabled:
            raise ProviderGateError("provider execution is disabled by configuration")
        if not provider.network_egress_enabled:
            raise ProviderGateError("network egress is disabled by configuration")
        role = provider.translator
        if role.endpoint is None:
            raise ProviderConfigurationError("translator endpoint must be configured")

        request_body = serialize_responses_request(
            request=request,
            prompt=prompt,
            model=role.model,
            temperature=role.temperature,
        )
        raw_response = self._transport.create_response(
            endpoint=role.endpoint,
            request_body=request_body,
        )
        try:
            response = TranslationResponse.model_validate_json(
                _structured_output_text(raw_response), strict=True
            )
            validate_translation_response(request, response)
        except (TranslationContractError, ValueError) as exc:
            raise ProviderResponseError(
                "structured output violates the local translation contract"
            ) from exc
        return response
