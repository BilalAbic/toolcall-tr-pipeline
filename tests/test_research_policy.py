from __future__ import annotations

import pytest

from toolcall_tr.research_policy import (
    ResearchBudget,
    ResearchCandidate,
    ResearchResolution,
    TerminologyInput,
    build_research_request,
    build_research_resolution,
    route_terminology_risks,
)

SEGMENT = f"seg_{'1' * 64}"


def risks():
    return route_terminology_risks(
        [
            TerminologyInput(
                segment_id=SEGMENT,
                path="/conversation/0/content",
                text="Use ISO API data.",
                policy_state="resolved",
            ),
            TerminologyInput(
                segment_id=f"seg_{'2' * 64}",
                path="/conversation/1/content",
                text="free text",
                policy_state="manual_policy_required",
            ),
        ]
    )


def test_router_is_deterministic_and_routes_only_mechanical_risks() -> None:
    first = risks()
    assert first == risks()
    assert {risk.trigger for risk in first} == {
        "uppercase_abbreviation",
        "field_policy_unresolved",
    }
    abbreviations = {risk.term for risk in first if risk.trigger == "uppercase_abbreviation"}
    assert abbreviations == {"API", "ISO"}


def test_research_request_is_not_sent_and_enforces_budget_and_public_https() -> None:
    request = build_research_request(
        risk=risks()[0],
        endpoint="https://docs.example.com/term",
        source_type="vendor_docs",
        query_count_for_episode=1,
        source_count_for_episode=1,
        elapsed_seconds_for_episode=0,
    )
    assert request.status == "not_sent"
    with pytest.raises(ValueError, match="private"):
        build_research_request(
            risk=risks()[0],
            endpoint="https://127.0.0.1/term",
            source_type="vendor_docs",
            query_count_for_episode=1,
            source_count_for_episode=1,
            elapsed_seconds_for_episode=0,
        )
    with pytest.raises(ValueError, match="query budget"):
        build_research_request(
            risk=risks()[0],
            endpoint="https://docs.example.com/term",
            source_type="vendor_docs",
            query_count_for_episode=4,
            source_count_for_episode=1,
            elapsed_seconds_for_episode=0,
            budget=ResearchBudget(),
        )


def test_resolution_requires_explicit_evidence_or_unresolved_empty_state() -> None:
    request = build_research_request(
        risk=risks()[0],
        endpoint="https://docs.example.com/term",
        source_type="vendor_docs",
        query_count_for_episode=1,
        source_count_for_episode=1,
        elapsed_seconds_for_episode=0,
    )
    candidate = ResearchCandidate(
        target="API",
        source_url="https://docs.example.com/api",
        source_type="vendor_docs",
        evidence_span="Official API name.",
    )
    assert build_research_resolution(
        request_id=request.request_id, status="resolved", candidates=[candidate]
    ).status == "resolved"
    with pytest.raises(ValueError, match="unresolved"):
        build_research_resolution(
            request_id=request.request_id, status="unresolved", candidates=[candidate]
        )


def test_research_resolution_never_fetches_or_accepts_private_candidate_urls() -> None:
    with pytest.raises(ValueError, match="private"):
        ResearchCandidate(
            target="term",
            source_url="https://localhost/private",
            source_type="vendor_docs",
            evidence_span="not safe",
        )
    with pytest.raises(ValueError, match="two distinct"):
        ResearchResolution(
            resolution_id=f"researchres_{'f' * 64}",
            request_id=f"research_{'e' * 64}",
            status="conflicting",
            candidates=[],
        )
