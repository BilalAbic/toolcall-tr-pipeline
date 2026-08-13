"""A bounded, standard-library HTTPS JSON transport boundary.

The surrounding pipeline deliberately keeps this class separate from policy and
configuration.  It does not read environment variables, emit logs, perform
retries, or contain a provider-specific API key name.  The caller explicitly
supplies both the credential lookup function and (for tests) the opener.

No request is made merely by importing or constructing this transport.  A
network request can occur only when :meth:`StdlibJsonTransport.create_response`
is invoked after the higher-level gates have approved it.
"""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RESPONSE_BYTES = 1_048_576
MAX_TIMEOUT_SECONDS = 120.0
MAX_RESPONSE_BYTES = 8_388_608


class SecureTransportError(RuntimeError):
    """Base class for redaction-safe HTTP transport failures."""


class TransportConfigurationError(SecureTransportError):
    """Raised for invalid local transport options or outbound request JSON."""


class EndpointRejectedError(SecureTransportError):
    """Raised before an unsafe endpoint can be handed to an HTTP implementation."""


class CredentialUnavailableError(SecureTransportError):
    """Raised without exposing a credential value or lookup implementation error."""


class TransportHttpError(SecureTransportError):
    """Raised for non-successful HTTP status codes without exposing response text."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTPS request failed with status {status_code}")


class TransportNetworkError(SecureTransportError):
    """Raised for connection, TLS, read, or opener errors without remote details."""


class ResponseTooLargeError(SecureTransportError):
    """Raised when the response would exceed the configured byte ceiling."""


class MalformedJsonResponseError(SecureTransportError):
    """Raised when a successful response is not valid JSON."""


class HttpResponse(Protocol):
    """The small response surface required from an injected opener."""

    status: int

    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...


class HttpOpener(Protocol):
    """Minimal injectable boundary around ``urllib``'s opener object."""

    def open(self, request: Request, timeout: float) -> HttpResponse: ...


SecretLookup = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class TransportLimits:
    """Finite HTTP limits that make accidental long-running calls fail safely."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES

    def __post_init__(self) -> None:
        if not _valid_timeout(self.timeout_seconds):
            raise TransportConfigurationError(
                "timeout_seconds must be between zero and the transport maximum"
            )
        if not _valid_response_limit(self.max_response_bytes):
            raise TransportConfigurationError(
                "max_response_bytes must be between one and the transport maximum"
            )


def _valid_timeout(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0 < value <= MAX_TIMEOUT_SECONDS
    )


def _valid_response_limit(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and 0 < value <= MAX_RESPONSE_BYTES
    )


class _NoRedirect(HTTPRedirectHandler):
    """Reject redirects so validated HTTPS endpoints cannot redirect elsewhere."""

    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


def _stdlib_opener() -> HttpOpener:
    """Build an opener that neither inherits proxies nor follows redirects."""
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    return cast(HttpOpener, opener)


def validate_https_endpoint(endpoint: str) -> None:
    """Validate a public HTTPS destination without DNS resolution or network access."""
    try:
        parsed = urlsplit(endpoint)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise EndpointRejectedError("endpoint is not a permitted HTTPS URL") from exc
    if (
        parsed.scheme != "https"
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise EndpointRejectedError("endpoint is not a permitted HTTPS URL")

    host = hostname.rstrip(".").casefold()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(
        (".localhost", ".local", ".internal")
    ):
        raise EndpointRejectedError("endpoint is not a permitted HTTPS URL")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
        or address.is_multicast
    ):
        raise EndpointRejectedError("endpoint is not a permitted HTTPS URL")


def _require_json_object(request_body: bytes) -> None:
    """Reject invalid outbound JSON before the request reaches an opener."""
    try:
        value = json.loads(request_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransportConfigurationError("request body must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise TransportConfigurationError("request body must be a JSON object")


def _read_bounded(response: HttpResponse, max_response_bytes: int) -> bytes:
    """Read one byte beyond the cap, including when a stream returns partial chunks."""
    chunks: list[bytes] = []
    remaining = max_response_bytes + 1
    try:
        while remaining:
            chunk = response.read(remaining)
            if not chunk:
                break
            if len(chunk) > remaining:
                raise ResponseTooLargeError("HTTPS response exceeds the configured byte limit")
            chunks.append(chunk)
            remaining -= len(chunk)
    except Exception as exc:
        if isinstance(exc, ResponseTooLargeError):
            raise
        raise TransportNetworkError("HTTPS response could not be read") from exc
    body = b"".join(chunks)
    if len(body) > max_response_bytes:
        raise ResponseTooLargeError("HTTPS response exceeds the configured byte limit")
    return body


class StdlibJsonTransport:
    """POST JSON over a single validated HTTPS request, using injected credentials."""

    def __init__(
        self,
        *,
        credential_name: str,
        secret_lookup: SecretLookup,
        opener: HttpOpener | None = None,
        limits: TransportLimits | None = None,
    ) -> None:
        if not credential_name:
            raise TransportConfigurationError("credential_name must not be empty")
        self._credential_name = credential_name
        self._secret_lookup = secret_lookup
        self._opener = opener if opener is not None else _stdlib_opener()
        self._limits = limits if limits is not None else TransportLimits()

    def _credential(self) -> str:
        try:
            secret = self._secret_lookup(self._credential_name)
        except Exception as exc:
            raise CredentialUnavailableError("API credential is unavailable") from exc
        if (
            not isinstance(secret, str)
            or not secret
            or any(character in secret for character in "\r\n")
        ):
            raise CredentialUnavailableError("API credential is unavailable")
        return secret

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        """Issue exactly one bounded POST request and return validated raw JSON bytes."""
        validate_https_endpoint(endpoint)
        _require_json_object(request_body)
        credential = self._credential()
        request = Request(
            endpoint,
            data=request_body,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {credential}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            response = self._opener.open(request, timeout=self._limits.timeout_seconds)
        except HTTPError as exc:
            raise TransportHttpError(exc.code) from exc
        except URLError as exc:
            raise TransportNetworkError("HTTPS request failed") from exc
        except Exception as exc:
            raise TransportNetworkError("HTTPS request failed") from exc

        try:
            if not 200 <= response.status < 300:
                raise TransportHttpError(response.status)
            body = _read_bounded(response, self._limits.max_response_bytes)
        finally:
            with suppress(Exception):
                response.close()

        try:
            json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MalformedJsonResponseError("HTTPS response was not valid JSON") from exc
        return body
