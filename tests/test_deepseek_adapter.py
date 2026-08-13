from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from tests.test_provider_adapter import config, prompt
from tests.test_translation_contract import request, response
from toolcall_tr.config import PipelineConfig, ProviderConfig, ProviderRole
from toolcall_tr.deepseek_adapter import (
    DeepSeekTranslationAdapter,
    serialize_deepseek_translation_request,
    validate_deepseek_endpoint,
)
from toolcall_tr.hashing import canonical_bytes
from toolcall_tr.provider_adapter import ProviderConfigurationError, ProviderResponseError
from toolcall_tr.translation_contract import TranslationRequest


def _calls() -> list[tuple[str, bytes]]:
    return []


@dataclass
class RecordingTransport:
    body: bytes
    calls: list[tuple[str, bytes]] = field(default_factory=_calls)

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        self.calls.append((endpoint, request_body))
        return self.body


def live_deepseek_config() -> PipelineConfig:
    base = config(providers_enabled=True, egress_enabled=True)
    return PipelineConfig(
        schema_version=base.schema_version,
        canonical_schema_version=base.canonical_schema_version,
        diagnostic_catalog_version=base.diagnostic_catalog_version,
        normalizer_version=base.normalizer_version,
        max_record_bytes=base.max_record_bytes,
        jsonl_shard_rows=base.jsonl_shard_rows,
        providers=ProviderConfig(
            enabled=True,
            network_egress_enabled=True,
            translator=ProviderRole(
                provider="deepseek",
                model="deepseek-v4-flash",
                api_key_env="DEEPSEEK_API_KEY",
                endpoint="https://api.deepseek.com/chat/completions",
                temperature=0.0,
                thinking=False,
            ),
            strong_judge=base.providers.strong_judge,
            mini_verifier=base.providers.mini_verifier,
        ),
    )


def _envelope(translation_request: TranslationRequest) -> bytes:
    return canonical_bytes(
        {
            "id": "chatcmpl_test",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": canonical_bytes(
                            response(translation_request.request_id)
                        ).decode("utf-8"),
                    },
                }
            ],
        }
    )


def test_deepseek_adapter_encodes_chat_completions_and_validates_local_contract() -> None:
    translation_request = request()
    transport = RecordingTransport(_envelope(translation_request))
    adapter = DeepSeekTranslationAdapter(config=live_deepseek_config(), transport=transport)

    translated = adapter.translate(request=translation_request, prompt=prompt())

    assert translated.request_id == translation_request.request_id
    endpoint, encoded = transport.calls[0]
    assert endpoint == "https://api.deepseek.com/chat/completions"
    body = json.loads(encoded)
    assert body["model"] == "deepseek-v4-flash"
    assert body["response_format"] == {"type": "json_object"}
    assert body["stream"] is False
    assert body["thinking"] == {"type": "disabled"}
    assert body["messages"][1]["content"] == canonical_bytes(translation_request).decode("utf-8")
    assert "Return exactly one JSON object" in body["messages"][0]["content"]
    assert "not the request object" in body["messages"][0]["content"]
    assert '"target_text"' in body["messages"][0]["content"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1/chat/completions",
        "https://other.example/chat/completions",
        "http://api.deepseek.com/chat/completions",
        "https://api.deepseek.com/chat/completions?unsafe=yes",
    ],
)
def test_deepseek_endpoint_is_exactly_allowlisted(endpoint: str) -> None:
    with pytest.raises(ProviderConfigurationError, match="approved DeepSeek endpoint"):
        validate_deepseek_endpoint(endpoint)


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
def test_deepseek_adapter_rejects_nonstop_or_contract_breaking_content(finish_reason: str) -> None:
    translation_request = request()
    invalid = canonical_bytes(
        {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"content": "this must never reach contract validation"},
                }
            ]
        }
    )
    adapter = DeepSeekTranslationAdapter(
        config=live_deepseek_config(), transport=RecordingTransport(invalid)
    )
    with pytest.raises(ProviderResponseError, match="natural stop"):
        adapter.translate(request=translation_request, prompt=prompt())


def test_serializer_rejects_retired_model_without_transport_call() -> None:
    with pytest.raises(ProviderConfigurationError, match="approved DeepSeek V4"):
        serialize_deepseek_translation_request(
            request=request(),
            prompt=prompt(),
            model="deepseek-chat",
            temperature=0.0,
            thinking=False,
            max_output_tokens=128,
        )
