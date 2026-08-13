"""Offline tests for the standard-library HTTPS JSON transport boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.error import URLError
from urllib.request import Request

import pytest

from toolcall_tr.secure_transport import (
    MAX_RESPONSE_BYTES,
    MAX_TIMEOUT_SECONDS,
    CredentialUnavailableError,
    EndpointRejectedError,
    MalformedJsonResponseError,
    ResponseTooLargeError,
    StdlibJsonTransport,
    TransportConfigurationError,
    TransportHttpError,
    TransportLimits,
    TransportNetworkError,
)


def _empty_ints() -> list[int]:
    return []


def _empty_requests() -> list[Request]:
    return []


def _empty_floats() -> list[float]:
    return []


@dataclass
class FakeResponse:
    status: int
    body: bytes
    closed: bool = False
    requested_amounts: list[int] = field(default_factory=_empty_ints)
    cursor: int = 0

    def read(self, amount: int = -1) -> bytes:
        self.requested_amounts.append(amount)
        if amount < 0:
            chunk = self.body[self.cursor :]
        else:
            chunk = self.body[self.cursor : self.cursor + amount]
        self.cursor += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


@dataclass
class ChunkedResponse(FakeResponse):
    chunk_bytes: int = 1

    def read(self, amount: int = -1) -> bytes:
        self.requested_amounts.append(amount)
        upper_bound = self.cursor + min(amount, self.chunk_bytes)
        chunk = self.body[self.cursor : upper_bound]
        self.cursor += len(chunk)
        return chunk


@dataclass
class FakeOpener:
    response: FakeResponse | None = None
    exception: Exception | None = None
    requests: list[Request] = field(default_factory=_empty_requests)
    timeouts: list[float] = field(default_factory=_empty_floats)

    def open(self, request: Request, timeout: float) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self.exception is not None:
            raise self.exception
        assert self.response is not None
        return self.response


def _lookup(name: str) -> str | None:
    assert name == "TEST_API_KEY"
    return "test-secret"


def _transport(opener: FakeOpener, *, limits: TransportLimits | None = None) -> StdlibJsonTransport:
    return StdlibJsonTransport(
        credential_name="TEST_API_KEY", secret_lookup=_lookup, opener=opener, limits=limits
    )


def test_transport_posts_json_with_bearer_auth_and_finite_limits() -> None:
    response = FakeResponse(200, b'{"output":[]}')
    opener = FakeOpener(response=response)
    limits = TransportLimits(timeout_seconds=2.5, max_response_bytes=64)
    transport = _transport(opener, limits=limits)

    actual = transport.create_response(
        endpoint="https://api.example.com/v1/responses", request_body=b'{"model":"test"}'
    )

    assert actual == response.body
    assert response.closed is True
    assert response.requested_amounts == [65, 52]
    assert opener.timeouts == [2.5]
    assert len(opener.requests) == 1
    request = opener.requests[0]
    assert request.get_method() == "POST"
    assert request.data == b'{"model":"test"}'
    headers = {name.casefold(): value for name, value in request.header_items()}
    assert headers == {
        "accept": "application/json",
        "authorization": "Bearer test-secret",
        "content-type": "application/json",
    }


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://api.example.com/v1",
        "https://127.0.0.1/v1",
        "https://[::1]/v1",
        "https://localhost/v1",
        "https://service.internal/v1",
        "https://user:password@api.example.com/v1",
        "https://api.example.com/v1#fragment",
    ],
)
def test_transport_rejects_nonpublic_or_nonhttps_endpoints_before_lookup_or_open(
    endpoint: str,
) -> None:
    opener = FakeOpener(response=FakeResponse(200, b"{}"))
    calls: list[str] = []

    def lookup(name: str) -> str | None:
        calls.append(name)
        return "test-secret"

    transport = StdlibJsonTransport(
        credential_name="TEST_API_KEY", secret_lookup=lookup, opener=opener
    )

    with pytest.raises(EndpointRejectedError, match="permitted HTTPS URL"):
        transport.create_response(endpoint=endpoint, request_body=b"{}")

    assert calls == []
    assert opener.requests == []


@pytest.mark.parametrize(
    ("status", "body", "error_type", "message"),
    [
        (500, b'{"secret":"must-not-leak"}', TransportHttpError, "status 500"),
        (200, b"not-json", MalformedJsonResponseError, "not valid JSON"),
    ],
)
def test_transport_rejects_bad_responses_without_leaking_body(
    status: int, body: bytes, error_type: type[Exception], message: str
) -> None:
    response = FakeResponse(status, body)
    opener = FakeOpener(response=response)

    with pytest.raises(error_type, match=message) as raised:
        _transport(opener).create_response(
            endpoint="https://api.example.com/v1/responses", request_body=b"{}"
        )

    assert "must-not-leak" not in str(raised.value)
    assert response.closed is True


def test_transport_rejects_oversized_response_without_retaining_its_body() -> None:
    response = FakeResponse(200, b"x" * (MAX_RESPONSE_BYTES + 1))
    opener = FakeOpener(response=response)

    with pytest.raises(ResponseTooLargeError, match="byte limit"):
        _transport(opener).create_response(
            endpoint="https://api.example.com/v1/responses", request_body=b"{}"
        )

    assert response.closed is True


def test_transport_detects_oversized_partial_response_chunks() -> None:
    response = ChunkedResponse(200, b"x" * 65, chunk_bytes=7)
    opener = FakeOpener(response=response)

    with pytest.raises(ResponseTooLargeError, match="byte limit"):
        _transport(opener, limits=TransportLimits(max_response_bytes=64)).create_response(
            endpoint="https://api.example.com/v1/responses", request_body=b"{}"
        )

    assert response.closed is True


def test_transport_redacts_credential_lookup_and_network_failures() -> None:
    opener = FakeOpener(exception=URLError("network detail test-secret"))
    transport = StdlibJsonTransport(
        credential_name="TEST_API_KEY",
        secret_lookup=lambda _: (_ for _ in ()).throw(RuntimeError("test-secret")),
        opener=opener,
    )

    with pytest.raises(CredentialUnavailableError, match="credential is unavailable") as raised:
        transport.create_response(endpoint="https://api.example.com", request_body=b"{}")
    assert "test-secret" not in str(raised.value)
    assert opener.requests == []

    with pytest.raises(TransportNetworkError, match="HTTPS request failed") as raised:
        _transport(opener).create_response(endpoint="https://api.example.com", request_body=b"{}")
    assert "test-secret" not in str(raised.value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"timeout_seconds": MAX_TIMEOUT_SECONDS + 1}, "timeout_seconds"),
        ({"max_response_bytes": 0}, "max_response_bytes"),
        ({"max_response_bytes": MAX_RESPONSE_BYTES + 1}, "max_response_bytes"),
    ],
)
def test_limits_must_be_finite_and_bounded(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(TransportConfigurationError, match=message):
        TransportLimits(**kwargs)


def test_invalid_limits_and_request_body_are_rejected_before_open() -> None:
    with pytest.raises(TransportConfigurationError, match="timeout_seconds"):
        TransportLimits(timeout_seconds=0)
    with pytest.raises(TransportConfigurationError, match="max_response_bytes"):
        TransportLimits(max_response_bytes=0)

    opener = FakeOpener(response=FakeResponse(200, b"{}"))
    with pytest.raises(TransportConfigurationError, match="valid JSON"):
        _transport(opener).create_response(
            endpoint="https://api.example.com", request_body=b"not-json"
        )
    assert opener.requests == []
