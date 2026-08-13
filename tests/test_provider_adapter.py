from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from tests.test_translation_contract import request, response
from toolcall_tr.config import PipelineConfig
from toolcall_tr.hashing import canonical_bytes
from toolcall_tr.prompt_contract import PromptBundle, build_prompt_bundle, make_prompt_layer
from toolcall_tr.provider_adapter import (
    ProviderGateError,
    ProviderResponseError,
    ResponsesTranslationAdapter,
)


def _empty_calls() -> list[tuple[str, bytes]]:
    return []


@dataclass
class RecordingTransport:
    response_body: bytes
    calls: list[tuple[str, bytes]] = field(default_factory=_empty_calls)

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        self.calls.append((endpoint, request_body))
        return self.response_body


def prompt() -> PromptBundle:
    layer_names = (
        "core_contract",
        "field_policy",
        "fidelity_contract",
        "protected_span_contract",
        "terminology_protocol",
        "output_contract",
    )
    return build_prompt_bundle(
        prompt_version="test-prompt-0.1.0",
        layers=[
            make_prompt_layer(name=name, version="1", content=f"{name} instructions")
            for name in layer_names
        ],
    )


def config(*, providers_enabled: bool, egress_enabled: bool) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "schema_version": "pipeline-config-0.1.0",
            "canonical_schema_version": "0.1.0",
            "diagnostic_catalog_version": "0.1.0",
            "normalizer_version": "0.1.0",
            "max_record_bytes": 1,
            "jsonl_shard_rows": 1,
            "providers": {
                "enabled": providers_enabled,
                "network_egress_enabled": egress_enabled,
                "translator": {
                    "provider": "test-provider",
                    "model": "test-model",
                    "api_key_env": "UNREAD_TEST_API_KEY",
                    "endpoint": "https://provider.invalid/v1/responses",
                    "temperature": 0.0,
                    "thinking": False,
                },
                "strong_judge": {
                    "provider": "test-provider",
                    "model": "judge-model",
                    "api_key_env": "UNREAD_TEST_API_KEY",
                },
                "mini_verifier": {
                    "provider": "test-provider",
                    "model": "verifier-model",
                    "api_key_env": "UNREAD_TEST_API_KEY",
                },
            },
        },
        strict=True,
    )


def responses_envelope(output: bytes) -> bytes:
    return canonical_bytes(
        {
            "id": "resp_test",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": output.decode("utf-8")}
                    ],
                }
            ],
        }
    )


def test_adapter_serializes_a_strict_responses_request_and_validates_output() -> None:
    translation_request = request()
    prompt_bundle = prompt()
    transport = RecordingTransport(
        responses_envelope(canonical_bytes(response(translation_request.request_id)))
    )
    adapter = ResponsesTranslationAdapter(
        config=config(providers_enabled=True, egress_enabled=True),
        transport=transport,
    )

    translated = adapter.translate(request=translation_request, prompt=prompt_bundle)

    assert translated.request_id == translation_request.request_id
    assert len(transport.calls) == 1
    endpoint, raw_request = transport.calls[0]
    assert endpoint == "https://provider.invalid/v1/responses"
    body = json.loads(raw_request)
    assert body["model"] == "test-model"
    assert body["store"] is False
    assert body["temperature"] == 0
    assert body["metadata"] == {
        "request_id": translation_request.request_id,
        "prompt_id": prompt_bundle.prompt_id,
    }
    assert body["input"][0]["content"][0]["text"] == prompt_bundle.system_text
    assert body["input"][1]["content"][0]["text"] == canonical_bytes(
        translation_request
    ).decode("utf-8")
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["name"] == "translation_response"
    assert body["text"]["format"]["strict"] is True
    assert body["text"]["format"]["schema"]["additionalProperties"] is False


@pytest.mark.parametrize(
    ("providers_enabled", "egress_enabled", "message"),
    [
        (False, False, "provider execution is disabled"),
        (True, False, "network egress is disabled"),
    ],
)
def test_disabled_config_gates_fail_closed_without_transport_use(
    providers_enabled: bool, egress_enabled: bool, message: str
) -> None:
    translation_request = request()
    transport = RecordingTransport(b"{}")
    adapter = ResponsesTranslationAdapter(
        config=config(providers_enabled=providers_enabled, egress_enabled=egress_enabled),
        transport=transport,
    )

    with pytest.raises(ProviderGateError, match=message):
        adapter.translate(request=translation_request, prompt=prompt())

    assert transport.calls == []


def test_adapter_rejects_missing_or_invalid_structured_output_without_leaking_it() -> None:
    translation_request = request()
    missing_output = RecordingTransport(canonical_bytes({"output": []}))
    adapter = ResponsesTranslationAdapter(
        config=config(providers_enabled=True, egress_enabled=True),
        transport=missing_output,
    )
    with pytest.raises(ProviderResponseError, match="exactly one output_text"):
        adapter.translate(request=translation_request, prompt=prompt())

    unsafe_response = response(translation_request.request_id, "protected values removed")
    invalid_output = RecordingTransport(responses_envelope(canonical_bytes(unsafe_response)))
    adapter = ResponsesTranslationAdapter(
        config=config(providers_enabled=True, egress_enabled=True),
        transport=invalid_output,
    )
    with pytest.raises(ProviderResponseError, match="violates"):
        adapter.translate(request=translation_request, prompt=prompt())
