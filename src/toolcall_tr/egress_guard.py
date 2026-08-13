"""Deterministic, fail-closed pre-egress scanning.

This module intentionally has no HTTP client, DNS lookup, provider SDK, or model
integration.  It gives later phases a narrow contract: inspect the exact text and
destination first, record redaction-safe findings, and reject the egress attempt.
The checked-in pipeline configuration is offline, so even an otherwise clean
request is blocked until a separately reviewed policy replaces this guard.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from toolcall_tr.config import PipelineConfig
from toolcall_tr.models import NonEmptyStr, StrictModel


class EgressViolationCategory(StrEnum):
    """Classes of content that must not leave the local pipeline."""

    CONFIGURATION = "configuration"
    SECRET = "secret"
    PII = "pii"
    LOCAL_PATH = "local_path"
    PRIVATE_ENDPOINT = "private_endpoint"
    ENDPOINT = "endpoint"


class EgressViolation(StrictModel):
    """A finding without the matched sensitive value."""

    rule_id: NonEmptyStr
    category: EgressViolationCategory
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]
    redactable: bool

    @model_validator(mode="after")
    def has_nonempty_span(self) -> EgressViolation:
        if self.end <= self.start:
            raise ValueError("violation end must be greater than start")
        return self


class EgressRequest(StrictModel):
    """The complete, local-only input to a proposed external egress."""

    schema_version: Literal["egress-request-0.1.0"] = "egress-request-0.1.0"
    provider: NonEmptyStr
    endpoint: NonEmptyStr
    payload: str


class PreEgressDecision(StrictModel):
    """A safe-to-log decision; it never contains the original sensitive payload."""

    schema_version: Literal["pre-egress-decision-0.1.0"] = "pre-egress-decision-0.1.0"
    allowed: Literal[False] = False
    sanitized_payload: str
    violations: list[EgressViolation]
    providers_enabled: bool
    network_egress_enabled: bool


class EgressBlockedError(RuntimeError):
    """Raised by callers that require a permitted provider request."""

    def __init__(self, decision: PreEgressDecision) -> None:
        self.decision = decision
        rules = ", ".join(item.rule_id for item in decision.violations)
        super().__init__(f"pre-egress policy blocked request ({rules})")


@dataclass(frozen=True, slots=True)
class _Match:
    rule_id: str
    category: EgressViolationCategory
    start: int
    end: int
    redactable: bool


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "secret.private_key",
        re.compile(
            r"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----[\s\S]*?"
            r"-----END(?: [A-Z]+)? PRIVATE KEY-----",
            re.IGNORECASE,
        ),
    ),
    (
        "secret.assignment",
        re.compile(
            r"\b(?:api[_-]?key|access[_-]?token|auth(?:orization)?|secret|"
            r"password|passwd|token)\b\s*[:=]\s*(?:[\"']?)[^\s\"']{4,}",
            re.IGNORECASE,
        ),
    ),
    ("secret.bearer", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)),
    ("secret.openai_style", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("secret.google_api_key", re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b")),
    ("secret.aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
)

_EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
)
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?90[\s.-]?)?(?:0?5\d{2}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2})(?!\w)"
)
_TCKN_PATTERN = re.compile(r"(?<!\d)\d{11}(?!\d)")
_PATH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("path.file_uri", re.compile(r"\bfile://[^\s\"']+", re.IGNORECASE)),
    ("path.windows_drive", re.compile(r"(?<![\w:])[A-Za-z]:[\\/][^\s\"']*")),
    ("path.unc", re.compile(r"(?<!:)\\\\[^\s\"']+|(?<!:)//[^\s\"']+")),
    ("path.unix_absolute", re.compile(r"(?<![\w:])/(?:[^/\s\"']+/?)+")),
    ("path.home", re.compile(r"(?<!\w)~/[^\s\"']+")),
)
_URL_PATTERN = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)


def _is_tckn(value: str) -> bool:
    """Return whether a numeric value satisfies the Turkish identity checksum."""
    if value[0] == "0":
        return False
    digits = [int(character) for character in value]
    tenth = ((sum(digits[0:9:2]) * 7) - sum(digits[1:8:2])) % 10
    eleventh = sum(digits[:10]) % 10
    return digits[9] == tenth and digits[10] == eleventh


def _endpoint_rule(endpoint: str, *, require_https: bool) -> str | None:
    """Classify unsafe destinations without resolving DNS or contacting a host."""
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "endpoint.invalid_url"
    if parsed.username is not None or parsed.password is not None:
        return "endpoint.embedded_credentials"
    if require_https and parsed.scheme != "https":
        return "endpoint.non_https"
    host = parsed.hostname.rstrip(".").casefold()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(
        (".localhost", ".local", ".internal")
    ):
        return "endpoint.private_hostname"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
        or address.is_multicast
    ):
        return "endpoint.private_address"
    return None


def _matches(payload: str) -> list[_Match]:
    found: list[_Match] = []
    for rule_id, pattern in _SECRET_PATTERNS:
        found.extend(
            _Match(rule_id, EgressViolationCategory.SECRET, item.start(), item.end(), True)
            for item in pattern.finditer(payload)
        )
    found.extend(
        _Match("pii.email", EgressViolationCategory.PII, item.start(), item.end(), True)
        for item in _EMAIL_PATTERN.finditer(payload)
    )
    found.extend(
        _Match("pii.phone", EgressViolationCategory.PII, item.start(), item.end(), True)
        for item in _PHONE_PATTERN.finditer(payload)
    )
    found.extend(
        _Match("pii.tckn", EgressViolationCategory.PII, item.start(), item.end(), True)
        for item in _TCKN_PATTERN.finditer(payload)
        if _is_tckn(item.group())
    )
    for rule_id, pattern in _PATH_PATTERNS:
        found.extend(
            _Match(rule_id, EgressViolationCategory.LOCAL_PATH, item.start(), item.end(), True)
            for item in pattern.finditer(payload)
        )
    for item in _URL_PATTERN.finditer(payload):
        if (rule_id := _endpoint_rule(item.group(), require_https=False)) is not None:
            found.append(
                _Match(
                    rule_id,
                    EgressViolationCategory.PRIVATE_ENDPOINT,
                    item.start(),
                    item.end(),
                    False,
                )
            )
    return _non_overlapping(found)


def _non_overlapping(found: list[_Match]) -> list[_Match]:
    """Choose deterministic spans so the redacted payload cannot expose overlap."""
    selected: list[_Match] = []
    ordered = sorted(
        found,
        key=lambda match: (match.start, -(match.end - match.start), match.rule_id),
    )
    for item in ordered:
        if not selected or item.start >= selected[-1].end:
            selected.append(item)
    return selected


def _redact(payload: str, matches: list[_Match]) -> str:
    pieces: list[str] = []
    cursor = 0
    for item in matches:
        pieces.append(payload[cursor : item.start])
        pieces.append(f"[REDACTED:{item.category.value.upper()}]")
        cursor = item.end
    pieces.append(payload[cursor:])
    return "".join(pieces)


def security_violations(request: EgressRequest) -> list[EgressViolation]:
    """Return secret/PII/path/endpoint findings without granting egress permission.

    The offline guard below intentionally adds a configuration violation and
    always rejects.  A separately approved live policy can reuse these exact
    deterministic security findings but must make its own authorization decision.
    """
    matches = _matches(request.payload)
    violations = [
        EgressViolation(
            rule_id=item.rule_id,
            category=item.category,
            start=item.start,
            end=item.end,
            redactable=item.redactable,
        )
        for item in matches
    ]
    if (endpoint_rule := _endpoint_rule(request.endpoint, require_https=True)) is not None:
        violations.append(
            EgressViolation(
                rule_id=endpoint_rule,
                category=(
                    EgressViolationCategory.PRIVATE_ENDPOINT
                    if endpoint_rule.startswith("endpoint.private")
                    else EgressViolationCategory.ENDPOINT
                ),
                start=0,
                end=len(request.endpoint),
                redactable=False,
            )
        )
    return violations


def preflight_egress(request: EgressRequest, config: PipelineConfig) -> PreEgressDecision:
    """Scan and reject an attempted egress without issuing any network request.

    `allowed` is deliberately fixed to ``False`` in this offline phase.  A future
    egress-capable phase must introduce a separate approved policy; it may not
    reinterpret a clean result from this function as permission to call a host.
    """
    matches = _matches(request.payload)
    violations = security_violations(request)
    violations.append(
        EgressViolation(
            rule_id="policy.offline_configuration",
            category=EgressViolationCategory.CONFIGURATION,
            start=0,
            end=1,
            redactable=False,
        )
    )
    return PreEgressDecision(
        sanitized_payload=_redact(request.payload, matches),
        violations=violations,
        providers_enabled=config.providers.enabled,
        network_egress_enabled=config.providers.network_egress_enabled,
    )


def require_preflight_clear(request: EgressRequest, config: PipelineConfig) -> None:
    """Fail closed for any accidental call path in the offline implementation."""
    raise EgressBlockedError(preflight_egress(request, config))
