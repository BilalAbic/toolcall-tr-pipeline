from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from tests.test_deepseek_adapter import live_deepseek_config
from toolcall_tr.live_preflight import preflight_live_request
from toolcall_tr.provider_provenance import (
    ProviderAttemptRecord,
    ProviderOperation,
    build_provider_attempt_record,
)
from toolcall_tr.provider_usage import (
    ProviderUsageSinkError,
    cost_lines,
    emit_provider_usage,
    estimated_total_usd,
    provider_usage_from_response,
)


def _attempt(provider: str, model: str, operation: ProviderOperation) -> ProviderAttemptRecord:
    return build_provider_attempt_record(
        operation=operation,
        provider=provider,
        model=model,
        endpoint=(
            "https://api.deepseek.com/chat/completions"
            if provider == "deepseek"
            else "https://api.openai.com/v1/responses"
        ),
        request_body=f'{{"request":"{provider}-{model}"}}'.encode(),
        response_body=b"{}",
        preflight=preflight_live_request(
            config=live_deepseek_config(),
            provider=provider,
            endpoint=(
                "https://api.deepseek.com/chat/completions"
                if provider == "deepseek"
                else "https://api.openai.com/v1/responses"
            ),
            payload=b'{"synthetic":true}',
        ),
    )


def test_usage_parser_is_hash_linked_and_never_retains_response_text() -> None:
    attempt = _attempt("deepseek", "deepseek-v4-flash", ProviderOperation.TRANSLATION)
    usage = provider_usage_from_response(
        attempt=attempt,
        response_body=(
            b'{"usage":{"prompt_tokens":1000,"prompt_cache_hit_tokens":250,'
            b'"completion_tokens":500},"private":"must-not-persist"}'
        ),
    )

    assert usage is not None
    assert usage.attempt_id == attempt.attempt_id
    assert usage.input_tokens == 1000
    assert usage.cached_input_tokens == 250
    assert usage.output_tokens == 500
    assert "must-not-persist" not in usage.model_dump_json()


def test_cost_dashboard_uses_provider_reported_tokens_and_price_cards(tmp_path: Path) -> None:
    deepseek_attempt = _attempt("deepseek", "deepseek-v4-flash", ProviderOperation.TRANSLATION)
    openai_attempt = _attempt("openai", "gpt-5.4", ProviderOperation.JUDGE)
    deepseek_usage = provider_usage_from_response(
        attempt=deepseek_attempt,
        response_body=b'{"usage":{"prompt_tokens":1000000,"prompt_cache_hit_tokens":500000,"completion_tokens":1000000}}',
    )
    openai_usage = provider_usage_from_response(
        attempt=openai_attempt,
        response_body=b'{"usage":{"input_tokens":1000000,"input_tokens_details":{"cached_tokens":200000},"output_tokens":1000000}}',
    )
    assert deepseek_usage is not None and openai_usage is not None
    (tmp_path / "translation" / "provider-attempts").mkdir(parents=True)
    (tmp_path / "judge" / "provider-attempts").mkdir(parents=True)
    (tmp_path / "translation" / "provider-usage").mkdir(parents=True)
    (tmp_path / "judge" / "provider-usage").mkdir(parents=True)
    (tmp_path / "translation" / "provider-attempts" / "deepseek.json").write_text(
        deepseek_attempt.model_dump_json(), encoding="utf-8"
    )
    (tmp_path / "judge" / "provider-attempts" / "openai.json").write_text(
        openai_attempt.model_dump_json(), encoding="utf-8"
    )
    (tmp_path / "translation" / "provider-usage" / "deepseek.json").write_text(
        deepseek_usage.model_dump_json(), encoding="utf-8"
    )
    (tmp_path / "judge" / "provider-usage" / "openai.json").write_text(
        openai_usage.model_dump_json(), encoding="utf-8"
    )

    lines = cost_lines(tmp_path)

    assert [(line.provider, line.model, line.requests) for line in lines] == [
        ("deepseek", "deepseek-v4-flash", 1),
        ("openai", "gpt-5.4", 1),
    ]
    assert lines[0].estimated_usd == Decimal("0.3514")
    assert lines[1].estimated_usd == Decimal("17.05")
    assert estimated_total_usd(lines) == Decimal("17.4014")


def test_usage_sink_failure_does_not_expose_cause() -> None:
    usage = provider_usage_from_response(
        attempt=_attempt("openai", "gpt-5.4-mini", ProviderOperation.JUDGE),
        response_body=b'{"usage":{"input_tokens":1,"output_tokens":1}}',
    )
    assert usage is not None

    def reject(_record: object) -> None:
        raise RuntimeError("private persistence failure")

    with pytest.raises(ProviderUsageSinkError, match="usage audit sink failed") as raised:
        emit_provider_usage(reject, usage)
    assert "private persistence failure" not in str(raised.value)
