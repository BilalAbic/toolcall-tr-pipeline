"""Recorded-response provider test double; it can never access a network.

The class is intentionally the only provider-shaped execution surface in this
phase.  A future client must pass the same request/response validation but may
not replace this test double by silently opening an HTTP path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from toolcall_tr.translation_contract import (
    TranslationContractError,
    TranslationRequest,
    TranslationResponse,
    validate_translation_response,
)


class ProviderProtocol(Protocol):
    """Small interface deliberately free of credentials and transport details."""

    def translate(self, request: TranslationRequest) -> TranslationResponse: ...


class RecordedResponseError(ValueError):
    """A recorded response is missing, malformed, or fails local validation."""


class RecordedResponseProvider:
    """Replay strict JSON payloads by request ID without filesystem or network I/O."""

    def __init__(self, responses_by_request_id: Mapping[str, bytes]) -> None:
        self._responses_by_request_id = dict(responses_by_request_id)
        self.calls: list[str] = []

    def translate(self, request: TranslationRequest) -> TranslationResponse:
        self.calls.append(request.request_id)
        payload = self._responses_by_request_id.get(request.request_id)
        if payload is None:
            raise RecordedResponseError(f"no recorded response for request {request.request_id}")
        try:
            response = TranslationResponse.model_validate_json(payload, strict=True)
            validate_translation_response(request, response)
        except (TranslationContractError, ValueError) as exc:
            raise RecordedResponseError(
                f"recorded response violates the local translation contract: {exc}"
            ) from exc
        return response
