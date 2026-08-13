from __future__ import annotations

import pytest

from tests.test_translation_contract import request
from toolcall_tr.fake_provider import RecordedResponseError, RecordedResponseProvider
from toolcall_tr.hashing import canonical_bytes
from toolcall_tr.translation_contract import (
    SegmentTranslation,
    TranslationResponse,
)


def recorded_response_payload(request_id: str) -> bytes:
    response = TranslationResponse(
        request_id=request_id,
        status="translated",
        segments=[
            SegmentTranslation(
                segment_id=f"seg_{'3' * 64}",
                target_text="⟪S1_P1⟫ ile ⟪S1_P2⟫ kullanin.",
                research_needed=False,
                uncertainty_tags=[],
            )
        ],
        term_queries=[],
    )
    return canonical_bytes(response)


def test_recorded_provider_replays_only_a_locally_valid_response() -> None:
    translation_request = request()
    provider = RecordedResponseProvider(
        {translation_request.request_id: recorded_response_payload(translation_request.request_id)}
    )

    response = provider.translate(translation_request)

    assert response.request_id == translation_request.request_id
    assert provider.calls == [translation_request.request_id]


def test_recorded_provider_rejects_missing_malformed_and_contract_breaking_payloads() -> None:
    translation_request = request()
    missing = RecordedResponseProvider({})
    with pytest.raises(RecordedResponseError, match="no recorded response"):
        missing.translate(translation_request)

    malformed = RecordedResponseProvider({translation_request.request_id: b"not json"})
    with pytest.raises(RecordedResponseError, match="violates"):
        malformed.translate(translation_request)

    unsafe = TranslationResponse(
        request_id=translation_request.request_id,
        status="translated",
        segments=[
            SegmentTranslation(
                segment_id=f"seg_{'3' * 64}",
                target_text="sentinels removed",
                research_needed=False,
                uncertainty_tags=[],
            )
        ],
        term_queries=[],
    )
    contract_breaking = RecordedResponseProvider(
        {translation_request.request_id: canonical_bytes(unsafe)}
    )
    with pytest.raises(RecordedResponseError, match="protected token mismatch"):
        contract_breaking.translate(translation_request)
