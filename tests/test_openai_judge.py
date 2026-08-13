from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from tests.test_provider_adapter import config
from toolcall_tr.config import PipelineConfig, ProviderConfig, ProviderRole
from toolcall_tr.eval_contract import SegmentPathEvidence, build_evaluation_unit
from toolcall_tr.hashing import canonical_bytes
from toolcall_tr.openai_judge import (
    OpenAIResponsesJudge,
    serialize_openai_judge_request,
    validate_openai_endpoint,
)
from toolcall_tr.provider_adapter import ProviderConfigurationError, ProviderResponseError


def _calls() -> list[tuple[str, bytes]]:
    return []


@dataclass
class RecordingTransport:
    body: bytes
    calls: list[tuple[str, bytes]] = field(default_factory=_calls)

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        self.calls.append((endpoint, request_body))
        return self.body


def _unit_and_evidence():
    unit = build_evaluation_unit(
        episode_id="ep_" + "1" * 64,
        segment_id="seg_" + "2" * 64,
        path="/conversation/0/content",
        source_text_sha256="sha256:" + "3" * 64,
        target_text_sha256="sha256:" + "4" * 64,
    )
    evidence = SegmentPathEvidence(
        segment_id=unit.segment_id,
        path=unit.path,
        source_excerpt="Keep ⟪S1_P1⟫ unchanged.",
        target_excerpt="⟪S1_P1⟫ öğesini değiştirmeyin.",
    )
    return unit, evidence


def _openai_config() -> PipelineConfig:
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
            translator=base.providers.translator,
            strong_judge=ProviderRole(
                provider="openai",
                model="gpt-5.4",
                api_key_env="OPENAI_API_KEY",
                endpoint="https://api.openai.com/v1/responses",
            ),
            mini_verifier=ProviderRole(
                provider="openai",
                model="gpt-5.4-mini",
                api_key_env="OPENAI_API_KEY",
                endpoint="https://api.openai.com/v1/responses",
            ),
        ),
    )


def _envelope(output: object) -> bytes:
    return canonical_bytes(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": canonical_bytes(output).decode("utf-8")}
                    ],
                }
            ],
        }
    )


def test_openai_judge_uses_all_required_strict_schema_and_maps_a_pass() -> None:
    unit, evidence = _unit_and_evidence()
    transport = RecordingTransport(
        _envelope({"conclusion": "pass", "findings": [], "unresolved_reasons": []})
    )
    verdict = OpenAIResponsesJudge(
        config=_openai_config(), role_name="mini_verifier", transport=transport
    ).judge(evaluation_unit=unit, evidence=evidence)

    assert verdict.conclusion == "pass"
    endpoint, raw = transport.calls[0]
    assert endpoint == "https://api.openai.com/v1/responses"
    body = json.loads(raw)
    assert body["store"] is False
    assert body["text"]["format"]["strict"] is True
    system_text = body["input"][0]["content"][0]["text"]
    assert "A pass requires findings=[]" in system_text
    assert "contiguous source and target excerpts verbatim" in system_text
    assert "Do not invent excerpts" in system_text
    schema = body["text"]["format"]["schema"]
    assert set(schema["required"]) == set(schema["properties"])
    finding = schema["$defs"]["JudgeFindingOutput"]
    assert set(finding["required"]) == set(finding["properties"])


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.openai.com/v1/chat/completions",
        "https://other.example/v1/responses",
        "http://api.openai.com/v1/responses",
        "https://api.openai.com/v1/responses?unsafe=yes",
    ],
)
def test_openai_endpoint_is_exactly_allowlisted(endpoint: str) -> None:
    with pytest.raises(ProviderConfigurationError, match="approved OpenAI"):
        validate_openai_endpoint(endpoint)


def test_openai_judge_rejects_refusal_and_ungrounded_finding() -> None:
    unit, evidence = _unit_and_evidence()
    refused = canonical_bytes({"status": "completed", "output": [{"type": "refusal"}]})
    judge = OpenAIResponsesJudge(
        config=_openai_config(), role_name="mini_verifier", transport=RecordingTransport(refused)
    )
    with pytest.raises(ProviderResponseError, match="refused"):
        judge.judge(evaluation_unit=unit, evidence=evidence)

    ungrounded = _envelope(
        {
            "conclusion": "fail",
            "findings": [
                {
                    "category": "accuracy.mistranslation",
                    "severity": "major",
                    "source_excerpt": "not in source",
                    "target_excerpt": "not in target",
                    "rationale": "ungrounded evidence",
                }
            ],
            "unresolved_reasons": [],
        }
    )
    judge = OpenAIResponsesJudge(
        config=_openai_config(), role_name="mini_verifier", transport=RecordingTransport(ungrounded)
    )
    with pytest.raises(ProviderResponseError, match="local judge contract"):
        judge.judge(evaluation_unit=unit, evidence=evidence)


def test_serializer_rejects_unapproved_model() -> None:
    unit, evidence = _unit_and_evidence()
    role = _openai_config().providers.mini_verifier.model_copy(update={"model": "unknown"})
    with pytest.raises(ProviderConfigurationError, match="approved OpenAI"):
        serialize_openai_judge_request(
            role=role, evaluation_unit=unit, evidence=evidence, max_output_tokens=128
        )
