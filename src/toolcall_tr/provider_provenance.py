"""Secret-safe, deterministic provenance for one live provider attempt.

The live adapters remain deliberately single-shot: this module records only
hashes of serialized request/response bytes and explicit policy outcomes.  It
never stores raw payloads, model output, headers, credentials, or exception
text.  A caller may inject a sink that appends these records to its own audited
event stream without giving the adapters filesystem or logging dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.hashing import sha256_bytes, stable_id
from toolcall_tr.live_preflight import LivePreflightDecision
from toolcall_tr.models import NonEmptyStr, Sha256, StrictModel
from toolcall_tr.provider_adapter import (
    ProviderAdapterError,
    ProviderConfigurationError,
    ProviderGateError,
    ProviderResponseError,
)
from toolcall_tr.secure_transport import (
    CredentialUnavailableError,
    EndpointRejectedError,
    MalformedJsonResponseError,
    ResponseTooLargeError,
    TransportConfigurationError,
    TransportHttpError,
    TransportNetworkError,
)


class ProviderOperation(StrEnum):
    """The only live provider operations implemented by this project."""

    TRANSLATION = "translation"
    JUDGE = "judge"


class ProviderAttemptOutcome(StrEnum):
    """Terminal result of one adapter call or a preflight rejection."""

    PREFLIGHT_BLOCKED = "preflight_blocked"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ProviderFailureCode(StrEnum):
    """Stable, redaction-safe reason codes; remote error text is intentionally absent."""

    PREFLIGHT_BLOCKED = "preflight_blocked"
    CREDENTIAL_UNAVAILABLE = "credential_unavailable"
    ENDPOINT_REJECTED = "endpoint_rejected"
    TRANSPORT_CONFIGURATION = "transport_configuration"
    HTTP_TRANSIENT = "http_transient"
    HTTP_PERMANENT = "http_permanent"
    NETWORK_DELIVERY_UNKNOWN = "network_delivery_unknown"
    RESPONSE_TOO_LARGE = "response_too_large"
    MALFORMED_RESPONSE = "malformed_response"
    PROVIDER_RESPONSE_INVALID = "provider_response_invalid"
    PROVIDER_CONFIGURATION = "provider_configuration"
    PROVIDER_GATE = "provider_gate"
    UNKNOWN = "unknown"


class RetryDisposition(StrEnum):
    """A classification for an operator; the adapters never retry automatically."""

    NOT_APPLICABLE = "not_applicable"
    DO_NOT_RETRY = "do_not_retry"
    MANUAL_RETRY_CANDIDATE = "manual_retry_candidate"


class RetryBudgetClassification(StrictModel):
    """The fixed zero automatic-retry budget prevents accidental duplicate charges."""

    automatic_retry_budget: Literal[0] = 0
    disposition: RetryDisposition
    reason: NonEmptyStr


class ProviderAttemptRecord(StrictModel):
    """Content-addressed, safe-to-log evidence for a single provider operation.

    ``attempt_id`` deliberately identifies the same provider/model/endpoint/body
    combination deterministically.  A surrounding append-only event stream is
    responsible for distinguishing repeated operator-initiated executions.
    """

    schema_version: Literal["provider-attempt-0.1.0"] = "provider-attempt-0.1.0"
    attempt_id: Annotated[str, Field(pattern=r"^pvattempt_[0-9a-f]{64}$")]
    operation: ProviderOperation
    provider: NonEmptyStr
    model: NonEmptyStr
    endpoint: NonEmptyStr
    request_sha256: Sha256
    preflight: LivePreflightDecision
    outcome: ProviderAttemptOutcome
    response_sha256: Sha256 | None
    failure_code: ProviderFailureCode | None
    http_status: Annotated[int, Field(ge=100, le=599)] | None
    retry: RetryBudgetClassification

    @model_validator(mode="after")
    def validate_state(self) -> ProviderAttemptRecord:
        identity = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "request_sha256": self.request_sha256,
        }
        if self.attempt_id != stable_id("pvattempt", identity):
            raise ValueError("provider attempt ID does not match deterministic identity")
        if self.preflight.provider != self.provider or self.preflight.endpoint != self.endpoint:
            raise ValueError("provider attempt must retain its exact preflight decision")
        if self.outcome is ProviderAttemptOutcome.PREFLIGHT_BLOCKED:
            if self.preflight.allowed or self.response_sha256 is not None:
                raise ValueError("blocked preflight cannot have a response")
            if self.failure_code is not ProviderFailureCode.PREFLIGHT_BLOCKED:
                raise ValueError("blocked preflight requires its stable failure code")
        elif self.outcome is ProviderAttemptOutcome.SUCCEEDED:
            if (
                not self.preflight.allowed
                or self.response_sha256 is None
                or self.failure_code is not None
                or self.http_status is not None
                or self.retry.disposition is not RetryDisposition.NOT_APPLICABLE
            ):
                raise ValueError("successful attempt has inconsistent provenance state")
        elif self.response_sha256 is not None and not self.preflight.allowed:
            raise ValueError("blocked preflight cannot have a response hash")
        if self.outcome is ProviderAttemptOutcome.FAILED and self.failure_code is None:
            raise ValueError("failed attempt requires a stable failure code")
        if self.http_status is not None and self.failure_code not in {
            ProviderFailureCode.HTTP_TRANSIENT,
            ProviderFailureCode.HTTP_PERMANENT,
        }:
            raise ValueError("only HTTP failures may retain an HTTP status")
        return self


ProviderAttemptSink = Callable[[ProviderAttemptRecord], None]


class ProviderAttemptSinkError(RuntimeError):
    """Raised without record contents if an injected audit sink cannot persist a record."""


def _http_failure_code(status_code: int) -> ProviderFailureCode:
    if status_code in {408, 409, 425, 429} or 500 <= status_code <= 599:
        return ProviderFailureCode.HTTP_TRANSIENT
    return ProviderFailureCode.HTTP_PERMANENT


def classify_failure(
    error: BaseException | None,
) -> tuple[ProviderFailureCode | None, int | None, RetryBudgetClassification]:
    """Classify without retaining exception text, provider bodies, or credentials."""
    if error is None:
        return (
            None,
            None,
            RetryBudgetClassification(
                disposition=RetryDisposition.NOT_APPLICABLE,
                reason="completed",
            ),
        )
    if isinstance(error, TransportHttpError):
        code = _http_failure_code(error.status_code)
        return (
            code,
            error.status_code,
            RetryBudgetClassification(
                disposition=(
                    RetryDisposition.MANUAL_RETRY_CANDIDATE
                    if code is ProviderFailureCode.HTTP_TRANSIENT
                    else RetryDisposition.DO_NOT_RETRY
                ),
                reason=code.value,
            ),
        )
    if isinstance(error, TransportNetworkError):
        return (
            ProviderFailureCode.NETWORK_DELIVERY_UNKNOWN,
            None,
            RetryBudgetClassification(
                disposition=RetryDisposition.MANUAL_RETRY_CANDIDATE,
                reason=ProviderFailureCode.NETWORK_DELIVERY_UNKNOWN.value,
            ),
        )
    if isinstance(error, CredentialUnavailableError):
        code = ProviderFailureCode.CREDENTIAL_UNAVAILABLE
    elif isinstance(error, EndpointRejectedError):
        code = ProviderFailureCode.ENDPOINT_REJECTED
    elif isinstance(error, TransportConfigurationError):
        code = ProviderFailureCode.TRANSPORT_CONFIGURATION
    elif isinstance(error, ResponseTooLargeError):
        code = ProviderFailureCode.RESPONSE_TOO_LARGE
    elif isinstance(error, MalformedJsonResponseError):
        code = ProviderFailureCode.MALFORMED_RESPONSE
    elif isinstance(error, ProviderResponseError):
        code = ProviderFailureCode.PROVIDER_RESPONSE_INVALID
    elif isinstance(error, ProviderConfigurationError):
        code = ProviderFailureCode.PROVIDER_CONFIGURATION
    elif isinstance(error, ProviderGateError):
        code = ProviderFailureCode.PROVIDER_GATE
    elif isinstance(error, ProviderAdapterError):
        code = ProviderFailureCode.UNKNOWN
    else:
        code = ProviderFailureCode.UNKNOWN
    return (
        code,
        None,
        RetryBudgetClassification(
            disposition=RetryDisposition.DO_NOT_RETRY,
            reason=code.value,
        ),
    )


def build_provider_attempt_record(
    *,
    operation: ProviderOperation,
    provider: str,
    model: str,
    endpoint: str,
    request_body: bytes,
    preflight: LivePreflightDecision,
    error: BaseException | None = None,
    response_body: bytes | None = None,
) -> ProviderAttemptRecord:
    """Build a record using hashes only; callers must never supply text payload fields."""
    failure_code, http_status, retry = classify_failure(error)
    if not preflight.allowed:
        outcome = ProviderAttemptOutcome.PREFLIGHT_BLOCKED
        failure_code = ProviderFailureCode.PREFLIGHT_BLOCKED
        http_status = None
        retry = RetryBudgetClassification(
            disposition=RetryDisposition.DO_NOT_RETRY,
            reason=ProviderFailureCode.PREFLIGHT_BLOCKED.value,
        )
        response_body = None
    elif error is None:
        outcome = ProviderAttemptOutcome.SUCCEEDED
    else:
        outcome = ProviderAttemptOutcome.FAILED
    request_sha256 = sha256_bytes(request_body)
    identity = {
        "schema_version": "provider-attempt-0.1.0",
        "operation": operation,
        "provider": provider,
        "model": model,
        "endpoint": endpoint,
        "request_sha256": request_sha256,
    }
    return ProviderAttemptRecord(
        attempt_id=stable_id("pvattempt", identity),
        operation=operation,
        provider=provider,
        model=model,
        endpoint=endpoint,
        request_sha256=request_sha256,
        preflight=preflight,
        outcome=outcome,
        response_sha256=sha256_bytes(response_body) if response_body is not None else None,
        failure_code=failure_code,
        http_status=http_status,
        retry=retry,
    )


def emit_provider_attempt(
    sink: ProviderAttemptSink | None, record: ProviderAttemptRecord
) -> None:
    """Send a safe record to an injected sink without granting I/O capabilities here."""
    if sink is None:
        return
    try:
        sink(record)
    except Exception as exc:
        raise ProviderAttemptSinkError("provider attempt audit sink failed") from exc
