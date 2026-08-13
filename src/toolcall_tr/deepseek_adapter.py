"""Strict DeepSeek Chat Completions adapter for leaf-only translation.

This is deliberately distinct from the OpenAI Responses adapter: DeepSeek
accepts ``messages`` at ``/chat/completions`` and returns JSON text under one
chat choice.  The host still validates the entire local response contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast
from urllib.parse import urlsplit

from toolcall_tr.config import PipelineConfig
from toolcall_tr.hashing import JsonValue, canonical_bytes
from toolcall_tr.live_preflight import (
    LivePreflightBlockedError,
    LivePreflightDecision,
    preflight_live_request,
)
from toolcall_tr.prompt_contract import PromptBundle
from toolcall_tr.provider_adapter import (
    ProviderConfigurationError,
    ProviderGateError,
    ProviderResponseError,
    ResponsesTransport,
)
from toolcall_tr.provider_provenance import (
    ProviderAttemptSink,
    ProviderOperation,
    build_provider_attempt_record,
    emit_provider_attempt,
)
from toolcall_tr.provider_usage import (
    ProviderUsageSink,
    emit_response_usage,
)
from toolcall_tr.translation_contract import (
    TranslationContractError,
    TranslationRequest,
    TranslationResponse,
    validate_translation_response,
)

_DEEPSEEK_HOST = "api.deepseek.com"
_DEEPSEEK_PATH = "/chat/completions"


def validate_deepseek_endpoint(endpoint: str) -> None:
    """Allow only the documented public DeepSeek Chat Completions endpoint."""
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _DEEPSEEK_HOST
        or parsed.path.rstrip("/") != _DEEPSEEK_PATH
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ProviderConfigurationError(
            "translator endpoint is not the approved DeepSeek endpoint"
        )


def serialize_deepseek_translation_request(
    *,
    request: TranslationRequest,
    prompt: PromptBundle,
    model: str,
    temperature: float | None,
    thinking: bool | None,
    max_output_tokens: int,
) -> bytes:
    """Encode the documented JSON-output Chat Completions request body."""
    if model not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        raise ProviderConfigurationError("translator model is not an approved DeepSeek V4 model")
    if not 1 <= max_output_tokens <= 4_096:
        raise ProviderConfigurationError("translator max_output_tokens must be between 1 and 4096")
    body: dict[str, JsonValue] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"{prompt.system_text}\n\n"
                    "Return exactly one JSON object, not the request object, with this exact "
                    'top-level shape: {"schema_version":"translation-response-0.1.0",'
                    '"request_id":"the exact request_id from the input",'
                    '"status":"translated" or "research_needed",'
                    '"segments":[{"segment_id":"the exact input segment_id",'
                    '"target_text":"Turkey Turkish translation preserving sentinels",'
                    '"research_needed":false,"uncertainty_tags":[]}],'
                    '"term_queries":[]}. Include no input-only fields, markdown, '
                    "explanatory text, or additional keys."
                ),
            },
            {
                "role": "user",
                "content": canonical_bytes(request).decode("utf-8"),
            },
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "max_tokens": max_output_tokens,
        "thinking": {"type": "enabled" if thinking else "disabled"},
    }
    if temperature is not None:
        body["temperature"] = temperature
    return canonical_bytes(body)


def _deepseek_content(raw_response: bytes) -> str:
    """Extract exactly one natural stop text without exposing remote body data."""
    try:
        parsed = cast(object, json.loads(raw_response))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderResponseError("DeepSeek transport returned invalid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ProviderResponseError("DeepSeek transport returned a non-object envelope")
    envelope = cast(Mapping[str, object], parsed)
    raw_choices = envelope.get("choices")
    if not isinstance(raw_choices, list):
        raise ProviderResponseError("DeepSeek envelope must contain exactly one choice")
    choices = cast(list[object], raw_choices)
    if len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ProviderResponseError("DeepSeek envelope must contain exactly one choice")
    choice = cast(Mapping[str, object], choices[0])
    if choice.get("finish_reason") != "stop":
        raise ProviderResponseError("DeepSeek completion did not reach a natural stop")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ProviderResponseError("DeepSeek choice is missing a message")
    content = cast(Mapping[str, object], message).get("content")
    if not isinstance(content, str) or not content:
        raise ProviderResponseError("DeepSeek choice is missing JSON content")
    return content


class DeepSeekTranslationAdapter:
    """Execute one gated DeepSeek translation and validate it locally."""

    def __init__(
        self,
        *,
        config: PipelineConfig,
        transport: ResponsesTransport,
        max_output_tokens: int = 1_024,
        attempt_sink: ProviderAttemptSink | None = None,
        usage_sink: ProviderUsageSink | None = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._max_output_tokens = max_output_tokens
        self._attempt_sink = attempt_sink
        self._usage_sink = usage_sink

    def _emit_terminal(
        self,
        *,
        request_body: bytes,
        preflight: LivePreflightDecision,
        provider: str,
        model: str,
        endpoint: str,
        error: BaseException | None = None,
        response_body: bytes | None = None,
    ) -> None:
        """Persist provenance before optional token counters for one terminal path."""
        record = build_provider_attempt_record(
            operation=ProviderOperation.TRANSLATION,
            provider=provider,
            model=model,
            endpoint=endpoint,
            request_body=request_body,
            preflight=preflight,
            error=error,
            response_body=response_body,
        )
        emit_provider_attempt(self._attempt_sink, record)
        emit_response_usage(
            self._usage_sink,
            attempt=record,
            response_body=response_body,
        )

    def translate(
        self, *, request: TranslationRequest, prompt: PromptBundle
    ) -> TranslationResponse:
        provider = self._config.providers
        if not provider.enabled:
            raise ProviderGateError("provider execution is disabled by configuration")
        if not provider.network_egress_enabled:
            raise ProviderGateError("network egress is disabled by configuration")
        role = provider.translator
        if role.provider != "deepseek" or role.endpoint is None:
            raise ProviderConfigurationError(
                "translator must be an explicitly configured DeepSeek role"
            )
        validate_deepseek_endpoint(role.endpoint)
        request_body = serialize_deepseek_translation_request(
            request=request,
            prompt=prompt,
            model=role.model,
            temperature=role.temperature,
            thinking=role.thinking,
            max_output_tokens=self._max_output_tokens,
        )
        preflight = preflight_live_request(
            config=self._config,
            provider=role.provider,
            endpoint=role.endpoint,
            payload=canonical_bytes(
                {
                    "segments": [
                        {"source_text": segment.source_text} for segment in request.segments
                    ],
                    "terminology_evidence": request.terminology_evidence,
                }
            ),
        )
        if not preflight.allowed:
            self._emit_terminal(
                request_body=request_body,
                preflight=preflight,
                provider=role.provider,
                model=role.model,
                endpoint=role.endpoint,
            )
            raise LivePreflightBlockedError(preflight)

        raw_response: bytes | None = None
        try:
            raw_response = self._transport.create_response(
                endpoint=role.endpoint,
                request_body=request_body,
            )
            response = TranslationResponse.model_validate_json(
                _deepseek_content(raw_response), strict=True
            )
            validate_translation_response(request, response)
        except (TranslationContractError, ValueError) as exc:
            error = ProviderResponseError(
                "DeepSeek JSON output violates the local translation contract"
            )
            self._emit_terminal(
                request_body=request_body,
                preflight=preflight,
                provider=role.provider,
                model=role.model,
                endpoint=role.endpoint,
                error=error,
                response_body=raw_response,
            )
            raise error from exc
        except Exception as exc:
            self._emit_terminal(
                request_body=request_body,
                preflight=preflight,
                provider=role.provider,
                model=role.model,
                endpoint=role.endpoint,
                error=exc,
                response_body=raw_response,
            )
            raise
        self._emit_terminal(
            request_body=request_body,
            preflight=preflight,
            provider=role.provider,
            model=role.model,
            endpoint=role.endpoint,
            response_body=raw_response,
        )
        return response
