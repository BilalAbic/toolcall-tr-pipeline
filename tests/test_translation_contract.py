from __future__ import annotations

import pytest
from pydantic import ValidationError

from toolcall_tr.translation_contract import (
    ProtectedToken,
    SegmentTranslation,
    TranslationContractError,
    TranslationRequest,
    TranslationResponse,
    TranslationSegment,
    build_translation_request,
    validate_translation_response,
)

EPISODE_ID = f"ep_{'1' * 64}"
VARIANT_ID = f"sha256:{'2' * 64}"
SEGMENT_ID = f"seg_{'3' * 64}"


def request() -> TranslationRequest:
    return build_translation_request(
        episode_id=EPISODE_ID,
        input_variant_id=VARIANT_ID,
        field_policy_version="field-policy-0.1.0",
        segments=[
            TranslationSegment(
                segment_id=SEGMENT_ID,
                path="/conversation/0/content",
                source_text="Use ⟪S1_P1⟫ with ⟪S1_P2⟫.",
                protected_tokens=[
                    ProtectedToken(token="⟪S1_P1⟫", occurrence=1),
                    ProtectedToken(token="⟪S1_P2⟫", occurrence=2),
                ],
            )
        ],
    )


def response(request_id: str, text: str = "⟪S1_P1⟫ ile ⟪S1_P2⟫ kullanin.") -> TranslationResponse:
    return TranslationResponse(
        request_id=request_id,
        status="translated",
        segments=[
            SegmentTranslation(
                segment_id=SEGMENT_ID,
                target_text=text,
                research_needed=False,
                uncertainty_tags=[],
            )
        ],
        term_queries=[],
    )


def test_request_id_is_content_derived_and_response_contract_accepts_exact_coverage() -> None:
    first = request()
    second = request()
    assert first == second
    result = response(first.request_id)
    validate_translation_response(first, result)


def test_response_rejects_sentinel_reordering_or_coverage_changes() -> None:
    source = request()
    with pytest.raises(TranslationContractError, match="protected token mismatch"):
        validate_translation_response(source, response(source.request_id, "⟪S1_P2⟫ sonra ⟪S1_P1⟫"))
    missing = TranslationResponse(
        request_id=source.request_id,
        status="translated",
        segments=[
            SegmentTranslation(
                segment_id=f"seg_{'4' * 64}",
                target_text="metin",
                research_needed=False,
                uncertainty_tags=[],
            )
        ],
        term_queries=[],
    )
    with pytest.raises(TranslationContractError, match="coverage mismatch"):
        validate_translation_response(source, missing)


def test_contract_rejects_non_nfc_and_invalid_statuses() -> None:
    source = request()
    with pytest.raises(TranslationContractError, match="NFC"):
        validate_translation_response(source, response(source.request_id, "I\u0307⟪S1_P1⟫ ⟪S1_P2⟫"))
    with pytest.raises(ValidationError, match="research_needed"):
        TranslationResponse(
            request_id=source.request_id,
            status="research_needed",
            segments=[
                SegmentTranslation(
                    segment_id=SEGMENT_ID,
                    target_text="⟪S1_P1⟫ ve ⟪S1_P2⟫",
                    research_needed=False,
                    uncertainty_tags=[],
                )
            ],
            term_queries=[],
        )


def test_segment_rejects_missing_or_unlisted_sentinels() -> None:
    with pytest.raises(ValidationError, match="occur once"):
        TranslationSegment(
            segment_id=SEGMENT_ID,
            path="/conversation/0/content",
            source_text="no sentinel",
            protected_tokens=[ProtectedToken(token="⟪S1_P1⟫", occurrence=1)],
        )
