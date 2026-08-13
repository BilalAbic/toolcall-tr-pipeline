from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from toolcall_tr.config import PipelineConfig, load_config
from toolcall_tr.egress_guard import (
    EgressBlockedError,
    EgressRequest,
    EgressViolationCategory,
    preflight_egress,
    require_preflight_clear,
)


@pytest.fixture
def offline_config() -> PipelineConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "pipeline.toml")


def _request(payload: str, endpoint: str = "https://api.example.com/v1") -> EgressRequest:
    return EgressRequest(provider="fixture", endpoint=endpoint, payload=payload)


@pytest.mark.parametrize(
    ("payload", "rule_id", "redaction"),
    [
        ("api_key=sk-1234567890abcdefghijkl", "secret.assignment", "[REDACTED:SECRET]"),
        ("mail: deniz@example.com", "pii.email", "[REDACTED:PII]"),
        ("telefon +90 532 123 45 67", "pii.phone", "[REDACTED:PII]"),
        ("TCKN 10000000146", "pii.tckn", "[REDACTED:PII]"),
        (r"log C:\\Users\\bilal\\token.txt", "path.windows_drive", "[REDACTED:LOCAL_PATH]"),
        (r"share \\server\share\input.json", "path.unc", "[REDACTED:LOCAL_PATH]"),
    ],
)
def test_sensitive_payloads_are_redacted_and_blocked(
    offline_config: PipelineConfig, payload: str, rule_id: str, redaction: str
) -> None:
    decision = preflight_egress(_request(payload), offline_config)

    assert decision.allowed is False
    assert rule_id in {item.rule_id for item in decision.violations}
    assert redaction in decision.sanitized_payload
    assert "sk-1234567890abcdefghijkl" not in decision.sanitized_payload


@pytest.mark.parametrize(
    ("endpoint", "rule_id"),
    [
        ("https://127.0.0.1:8443", "endpoint.private_address"),
        ("https://[::1]/v1", "endpoint.private_address"),
        ("https://localhost/v1", "endpoint.private_hostname"),
        ("https://service.internal/v1", "endpoint.private_hostname"),
        ("http://api.example.com/v1", "endpoint.non_https"),
        ("https://api-key:secret@api.example.com/v1", "endpoint.embedded_credentials"),
    ],
)
def test_private_or_unsafe_destinations_are_blocked(
    offline_config: PipelineConfig, endpoint: str, rule_id: str
) -> None:
    decision = preflight_egress(_request("clean minimal context", endpoint), offline_config)

    assert decision.allowed is False
    assert rule_id in {item.rule_id for item in decision.violations}


def test_private_url_in_payload_is_non_redactable_and_blocked(
    offline_config: PipelineConfig,
) -> None:
    decision = preflight_egress(
        _request("inspect http://127.0.0.1:8080/admin"), offline_config
    )

    violation = next(
        item for item in decision.violations if item.rule_id == "endpoint.private_address"
    )
    assert violation.category is EgressViolationCategory.PRIVATE_ENDPOINT
    assert violation.redactable is False


def test_clean_request_still_fails_closed_for_offline_config(
    offline_config: PipelineConfig,
) -> None:
    request = _request("only minimal, safe context")
    decision = preflight_egress(request, offline_config)

    assert decision.allowed is False
    assert decision.sanitized_payload == request.payload
    assert [item.rule_id for item in decision.violations] == ["policy.offline_configuration"]
    with pytest.raises(EgressBlockedError, match=r"policy\.offline_configuration"):
        require_preflight_clear(request, offline_config)


def test_contracts_are_strict_and_do_not_log_matched_values() -> None:
    with pytest.raises(ValidationError):
        EgressRequest.model_validate(
            {
                "provider": "fixture",
                "endpoint": "https://api.example.com",
                "payload": "x",
                "unexpected": True,
            },
            strict=True,
        )
