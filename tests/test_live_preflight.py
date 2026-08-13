from __future__ import annotations

import pytest

from tests.test_deepseek_adapter import live_deepseek_config
from toolcall_tr.live_preflight import preflight_live_request


def test_live_preflight_allows_clean_explicitly_enabled_synthetic_payload() -> None:
    config = live_deepseek_config()
    decision = preflight_live_request(
        config=config,
        provider="deepseek",
        endpoint="https://api.deepseek.com/chat/completions",
        payload=b'{"synthetic":true}',
    )
    assert decision.allowed is True
    assert decision.violations == []
    assert decision.payload_sha256.startswith("sha256:")


@pytest.mark.parametrize(
    ("payload", "rule"),
    [
        (b'{"text":"person@example.com"}', "pii.email"),
        (b'{"text":"C:\\\\secret\\\\file.txt"}', "path.windows_drive"),
        (b'{"api_key":"sk-1234567890abcdefghijkl"}', "secret.openai_style"),
    ],
)
def test_live_preflight_blocks_sensitive_payload_without_retaining_it(
    payload: bytes, rule: str
) -> None:
    decision = preflight_live_request(
        config=live_deepseek_config(),
        provider="deepseek",
        endpoint="https://api.deepseek.com/chat/completions",
        payload=payload,
    )
    assert decision.allowed is False
    assert rule in {item.rule_id for item in decision.violations}
    assert payload.decode("utf-8") not in decision.model_dump_json()


def test_live_preflight_blocks_offline_config_even_for_a_clean_payload() -> None:
    base = live_deepseek_config()
    config = base.model_copy(
        update={"providers": base.providers.model_copy(update={"enabled": False})}
    )
    decision = preflight_live_request(
        config=config,
        provider="deepseek",
        endpoint="https://api.deepseek.com/chat/completions",
        payload=b'{"synthetic":true}',
    )
    assert decision.allowed is False
    assert "policy.providers_disabled" in {item.rule_id for item in decision.violations}
