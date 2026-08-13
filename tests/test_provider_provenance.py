"""Contract tests for hash-only live-provider attempt provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from tests.test_deepseek_adapter import live_deepseek_config
from tests.test_provider_adapter import config as base_config
from tests.test_provider_adapter import prompt
from tests.test_translation_contract import request, response
from toolcall_tr.config import PipelineConfig, ProviderConfig, ProviderRole
from toolcall_tr.deepseek_adapter import DeepSeekTranslationAdapter
from toolcall_tr.eval_contract import EvaluationUnit, SegmentPathEvidence, build_evaluation_unit
from toolcall_tr.hashing import canonical_bytes
from toolcall_tr.live_preflight import (
    LivePreflightBlockedError,
    LivePreflightDecision,
    preflight_live_request,
)
from toolcall_tr.openai_judge import OpenAIResponsesJudge
from toolcall_tr.provider_provenance import (
    ProviderAttemptOutcome,
    ProviderAttemptRecord,
    ProviderAttemptSinkError,
    ProviderFailureCode,
    ProviderOperation,
    RetryDisposition,
    build_provider_attempt_record,
    classify_failure,
)
from toolcall_tr.secure_transport import TransportHttpError
from toolcall_tr.translation_contract import build_translation_request


def _calls() -> list[tuple[str, bytes]]:
    return []


@dataclass
class RecordingTransport:
    body: bytes
    calls: list[tuple[str, bytes]] = field(default_factory=_calls)

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        self.calls.append((endpoint, request_body))
        return self.body


@dataclass
class RaisingTransport:
    error: Exception
    calls: list[tuple[str, bytes]] = field(default_factory=_calls)

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        self.calls.append((endpoint, request_body))
        raise self.error


def _clean_preflight() -> LivePreflightDecision:
    return preflight_live_request(
        config=live_deepseek_config(),
        provider="deepseek",
        endpoint="https://api.deepseek.com/chat/completions",
        payload=b'{"synthetic":true}',
    )


def _deepseek_envelope(request_id: str) -> bytes:
    return canonical_bytes(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": canonical_bytes(response(request_id)).decode("utf-8")
                    },
                }
            ]
        }
    )


def _openai_config() -> PipelineConfig:
    base = base_config(providers_enabled=True, egress_enabled=True)
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


def _unit_and_evidence() -> tuple[EvaluationUnit, SegmentPathEvidence]:
    unit = build_evaluation_unit(
        episode_id="ep_" + "1" * 64,
        segment_id="seg_" + "2" * 64,
        path="/conversation/0/content",
        source_text_sha256="sha256:" + "3" * 64,
        target_text_sha256="sha256:" + "4" * 64,
    )
    return (
        unit,
        SegmentPathEvidence(
            segment_id=unit.segment_id,
            path=unit.path,
            source_excerpt="Keep protected text unchanged.",
            target_excerpt="Korunan metni değiştirmeyin.",
        ),
    )


def test_attempt_record_is_deterministic_and_retains_hashes_not_sensitive_bodies() -> None:
    request_body = b'{"source":"private request detail","api_key":"secret-value"}'
    response_body = b'{"output":"private response detail"}'
    first = build_provider_attempt_record(
        operation=ProviderOperation.TRANSLATION,
        provider="deepseek",
        model="deepseek-v4-flash",
        endpoint="https://api.deepseek.com/chat/completions",
        request_body=request_body,
        response_body=response_body,
        preflight=_clean_preflight(),
    )
    second = build_provider_attempt_record(
        operation=ProviderOperation.TRANSLATION,
        provider="deepseek",
        model="deepseek-v4-flash",
        endpoint="https://api.deepseek.com/chat/completions",
        request_body=request_body,
        response_body=response_body,
        preflight=_clean_preflight(),
    )

    serialized = first.model_dump_json()
    assert first == second
    assert first.outcome is ProviderAttemptOutcome.SUCCEEDED
    assert first.retry.automatic_retry_budget == 0
    assert first.retry.disposition is RetryDisposition.NOT_APPLICABLE
    assert "private request detail" not in serialized
    assert "private response detail" not in serialized
    assert "secret-value" not in serialized


@pytest.mark.parametrize(
    ("status", "failure_code", "disposition"),
    [
        (429, ProviderFailureCode.HTTP_TRANSIENT, RetryDisposition.MANUAL_RETRY_CANDIDATE),
        (503, ProviderFailureCode.HTTP_TRANSIENT, RetryDisposition.MANUAL_RETRY_CANDIDATE),
        (401, ProviderFailureCode.HTTP_PERMANENT, RetryDisposition.DO_NOT_RETRY),
    ],
)
def test_http_retry_classification_is_manual_only_and_has_zero_automatic_budget(
    status: int,
    failure_code: ProviderFailureCode,
    disposition: RetryDisposition,
) -> None:
    actual_failure, http_status, budget = classify_failure(TransportHttpError(status))

    assert actual_failure is failure_code
    assert http_status == status
    assert budget.automatic_retry_budget == 0
    assert budget.disposition is disposition


def test_deepseek_adapter_emits_one_hash_only_success_record() -> None:
    translation_request = request()
    records: list[ProviderAttemptRecord] = []
    adapter = DeepSeekTranslationAdapter(
        config=live_deepseek_config(),
        transport=RecordingTransport(_deepseek_envelope(translation_request.request_id)),
        attempt_sink=records.append,
    )

    adapter.translate(request=translation_request, prompt=prompt())

    assert len(records) == 1
    record = records[0]
    assert record.operation is ProviderOperation.TRANSLATION
    assert record.outcome is ProviderAttemptOutcome.SUCCEEDED
    assert record.response_sha256 is not None
    assert translation_request.segments[0].source_text not in record.model_dump_json()


def test_deepseek_adapter_records_preflight_block_without_transport_delivery() -> None:
    source_request = request()
    blocked_request = build_translation_request(
        episode_id=source_request.episode_id,
        input_variant_id=source_request.input_variant_id,
        field_policy_version=source_request.field_policy_version,
        segments=source_request.segments,
        terminology_evidence=[{"term": "person@example.com"}],
    )
    records: list[ProviderAttemptRecord] = []
    transport = RecordingTransport(b'{"this":"must not be returned"}')
    adapter = DeepSeekTranslationAdapter(
        config=live_deepseek_config(),
        transport=transport,
        attempt_sink=records.append,
    )

    with pytest.raises(LivePreflightBlockedError, match=r"pii\.email"):
        adapter.translate(request=blocked_request, prompt=prompt())

    assert transport.calls == []
    assert len(records) == 1
    record = records[0]
    assert record.outcome is ProviderAttemptOutcome.PREFLIGHT_BLOCKED
    assert record.failure_code is ProviderFailureCode.PREFLIGHT_BLOCKED
    assert record.response_sha256 is None
    assert record.retry.automatic_retry_budget == 0
    assert "person@example.com" not in record.model_dump_json()


def test_openai_judge_emits_transient_failure_record_without_response_body() -> None:
    unit, evidence = _unit_and_evidence()
    records: list[ProviderAttemptRecord] = []
    judge = OpenAIResponsesJudge(
        config=_openai_config(),
        role_name="mini_verifier",
        transport=RaisingTransport(TransportHttpError(429)),
        attempt_sink=records.append,
    )

    with pytest.raises(TransportHttpError, match="status 429"):
        judge.judge(evaluation_unit=unit, evidence=evidence)

    assert len(records) == 1
    record = records[0]
    assert record.operation is ProviderOperation.JUDGE
    assert record.outcome is ProviderAttemptOutcome.FAILED
    assert record.failure_code is ProviderFailureCode.HTTP_TRANSIENT
    assert record.http_status == 429
    assert record.response_sha256 is None
    assert record.retry.automatic_retry_budget == 0
    assert record.retry.disposition is RetryDisposition.MANUAL_RETRY_CANDIDATE
    assert evidence.source_excerpt not in record.model_dump_json()
    assert json.loads(record.model_dump_json()) == record.model_dump(mode="json")


def test_sink_failure_fails_closed_after_one_request_without_leaking_provider_data() -> None:
    translation_request = request()
    transport = RecordingTransport(_deepseek_envelope(translation_request.request_id))

    def rejecting_sink(record: ProviderAttemptRecord) -> None:
        assert record.outcome is ProviderAttemptOutcome.SUCCEEDED
        raise RuntimeError("sensitive sink failure detail")

    adapter = DeepSeekTranslationAdapter(
        config=live_deepseek_config(),
        transport=transport,
        attempt_sink=rejecting_sink,
    )
    with pytest.raises(ProviderAttemptSinkError, match="audit sink failed") as raised:
        adapter.translate(request=translation_request, prompt=prompt())

    assert len(transport.calls) == 1
    assert translation_request.segments[0].source_text not in str(raised.value)
    assert "sensitive sink failure detail" not in str(raised.value)
