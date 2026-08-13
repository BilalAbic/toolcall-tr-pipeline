"""Approved live egress preflight that retains the offline guard's scanners."""

from __future__ import annotations

from typing import Literal

from toolcall_tr.config import PipelineConfig
from toolcall_tr.egress_guard import (
    EgressRequest,
    EgressViolation,
    EgressViolationCategory,
    security_violations,
)
from toolcall_tr.hashing import sha256_bytes
from toolcall_tr.models import NonEmptyStr, Sha256, StrictModel


class LivePreflightDecision(StrictModel):
    """A redaction-safe security decision; it never stores the request payload."""

    schema_version: Literal["live-preflight-0.1.0"] = "live-preflight-0.1.0"
    provider: NonEmptyStr
    endpoint: NonEmptyStr
    payload_sha256: Sha256
    allowed: bool
    violations: list[EgressViolation]


class LivePreflightBlockedError(RuntimeError):
    """Raised without sensitive payload or credential details."""

    def __init__(self, decision: LivePreflightDecision) -> None:
        self.decision = decision
        rules = ",".join(item.rule_id for item in decision.violations)
        super().__init__(f"live preflight blocked request ({rules})")


def preflight_live_request(
    *, config: PipelineConfig, provider: str, endpoint: str, payload: bytes
) -> LivePreflightDecision:
    """Scan one exact request and refuse it unless both explicit gates are true."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LivePreflightBlockedError(
            LivePreflightDecision(
                provider=provider,
                endpoint=endpoint,
                payload_sha256=sha256_bytes(payload),
                allowed=False,
                violations=[],
            )
        ) from exc
    violations = security_violations(
        EgressRequest(provider=provider, endpoint=endpoint, payload=text)
    )
    if not config.providers.enabled:
        violations.append(
            EgressViolation(
                rule_id="policy.providers_disabled",
                category=EgressViolationCategory.CONFIGURATION,
                start=0,
                end=1,
                redactable=False,
            )
        )
    if not config.providers.network_egress_enabled:
        violations.append(
            EgressViolation(
                rule_id="policy.network_egress_disabled",
                category=EgressViolationCategory.CONFIGURATION,
                start=0,
                end=1,
                redactable=False,
            )
        )
    return LivePreflightDecision(
        provider=provider,
        endpoint=endpoint,
        payload_sha256=sha256_bytes(payload),
        allowed=not violations,
        violations=violations,
    )


def require_live_preflight(
    *, config: PipelineConfig, provider: str, endpoint: str, payload: bytes
) -> LivePreflightDecision:
    """Raise before transport use whenever a live request is not safe to send."""
    decision = preflight_live_request(
        config=config, provider=provider, endpoint=endpoint, payload=payload
    )
    if not decision.allowed:
        raise LivePreflightBlockedError(decision)
    return decision
