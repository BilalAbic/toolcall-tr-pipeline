"""Deterministic terminology routing and research metadata, with no fetching."""

from __future__ import annotations

import ipaddress
import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from toolcall_tr.hashing import stable_id
from toolcall_tr.models import NonEmptyStr, StrictModel
from toolcall_tr.translation_contract import JsonPointer, SegmentId

ResearchRiskId = Annotated[str, Field(pattern=r"^risk_[0-9a-f]{64}$")]
ResearchRequestId = Annotated[str, Field(pattern=r"^research_[0-9a-f]{64}$")]
ResearchResolutionId = Annotated[str, Field(pattern=r"^researchres_[0-9a-f]{64}$")]
SourceType = Literal["vendor_docs", "standards_body", "official_institution"]
_ABBREVIATION = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")


class ResearchPolicyError(ValueError):
    """A proposed research request or evidence cannot safely leave the host."""


class TerminologyInput(StrictModel):
    segment_id: SegmentId
    path: JsonPointer
    text: NonEmptyStr
    policy_state: Literal["resolved", "manual_policy_required"]


class TerminologyRisk(StrictModel):
    schema_version: Literal["terminology-risk-0.1.0"] = "terminology-risk-0.1.0"
    risk_id: ResearchRiskId
    segment_id: SegmentId
    path: JsonPointer
    term: NonEmptyStr
    trigger: Literal["uppercase_abbreviation", "field_policy_unresolved"]
    disposition: Literal["needs_research"] = "needs_research"

    @model_validator(mode="after")
    def validate_identity(self) -> TerminologyRisk:
        body = self.model_dump(mode="json", exclude={"risk_id"})
        if self.risk_id != stable_id("risk", body):
            raise ValueError("terminology risk ID does not match deterministic content")
        return self


class ResearchBudget(StrictModel):
    schema_version: Literal["research-budget-0.1.0"] = "research-budget-0.1.0"
    max_queries_per_episode: Literal[3] = 3
    max_sources_per_episode: Literal[5] = 5
    max_elapsed_seconds_per_episode: Literal[60] = 60


class ResearchRequest(StrictModel):
    """A not-sent request. Another phase must perform any retrieval explicitly."""

    schema_version: Literal["research-request-0.1.0"] = "research-request-0.1.0"
    request_id: ResearchRequestId
    risk: TerminologyRisk
    endpoint: NonEmptyStr
    source_type: SourceType
    query_count_for_episode: Annotated[int, Field(ge=1)]
    source_count_for_episode: Annotated[int, Field(ge=1)]
    elapsed_seconds_for_episode: Annotated[int, Field(ge=0)]
    budget: ResearchBudget
    status: Literal["not_sent"] = "not_sent"

    @model_validator(mode="after")
    def validate_request(self) -> ResearchRequest:
        _validate_public_https_url(self.endpoint)
        if self.query_count_for_episode > self.budget.max_queries_per_episode:
            raise ValueError("research query budget exceeded")
        if self.source_count_for_episode > self.budget.max_sources_per_episode:
            raise ValueError("research source budget exceeded")
        if self.elapsed_seconds_for_episode > self.budget.max_elapsed_seconds_per_episode:
            raise ValueError("research elapsed-time budget exceeded")
        body = self.model_dump(mode="json", exclude={"request_id"})
        if self.request_id != stable_id("research", body):
            raise ValueError("research request ID does not match deterministic content")
        return self


class ResearchCandidate(StrictModel):
    target: NonEmptyStr
    source_url: NonEmptyStr
    source_type: SourceType
    evidence_span: NonEmptyStr

    @model_validator(mode="after")
    def validate_safe_source(self) -> ResearchCandidate:
        _validate_public_https_url(self.source_url)
        return self


class ResearchResolution(StrictModel):
    """Evidence sidecar only; it never edits a canonical episode or tool field."""

    schema_version: Literal["research-resolution-0.1.0"] = "research-resolution-0.1.0"
    resolution_id: ResearchResolutionId
    request_id: ResearchRequestId
    status: Literal["resolved", "conflicting", "unresolved"]
    candidates: list[ResearchCandidate]

    @model_validator(mode="after")
    def validate_resolution(self) -> ResearchResolution:
        targets = {candidate.target for candidate in self.candidates}
        if self.status == "resolved" and len(self.candidates) != 1:
            raise ValueError("resolved research requires exactly one evidence candidate")
        if self.status == "conflicting" and len(targets) < 2:
            raise ValueError("conflicting research requires two distinct targets")
        if self.status == "unresolved" and self.candidates:
            raise ValueError("unresolved research cannot claim candidates")
        body = self.model_dump(mode="json", exclude={"resolution_id"})
        if self.resolution_id != stable_id("researchres", body):
            raise ValueError("research resolution ID does not match deterministic content")
        return self


def route_terminology_risks(inputs: list[TerminologyInput]) -> list[TerminologyRisk]:
    """Route only mechanically observable ambiguity; it does not translate terms."""
    risks: list[TerminologyRisk] = []
    for item in inputs:
        if item.policy_state == "manual_policy_required":
            risks.append(_risk(item, item.text, "field_policy_unresolved"))
        for term in sorted(set(_ABBREVIATION.findall(item.text))):
            risks.append(_risk(item, term, "uppercase_abbreviation"))
    return sorted(risks, key=lambda risk: risk.risk_id)


def build_research_request(
    *,
    risk: TerminologyRisk,
    endpoint: str,
    source_type: SourceType,
    query_count_for_episode: int,
    source_count_for_episode: int,
    elapsed_seconds_for_episode: int,
    budget: ResearchBudget | None = None,
) -> ResearchRequest:
    selected_budget = budget or ResearchBudget()
    body = {
        "schema_version": "research-request-0.1.0",
        "risk": risk.model_dump(mode="json"),
        "endpoint": endpoint,
        "source_type": source_type,
        "query_count_for_episode": query_count_for_episode,
        "source_count_for_episode": source_count_for_episode,
        "elapsed_seconds_for_episode": elapsed_seconds_for_episode,
        "budget": selected_budget.model_dump(mode="json"),
        "status": "not_sent",
    }
    return ResearchRequest(
        request_id=stable_id("research", body),
        risk=risk,
        endpoint=endpoint,
        source_type=source_type,
        query_count_for_episode=query_count_for_episode,
        source_count_for_episode=source_count_for_episode,
        elapsed_seconds_for_episode=elapsed_seconds_for_episode,
        budget=selected_budget,
    )


def build_research_resolution(
    *,
    request_id: str,
    status: Literal["resolved", "conflicting", "unresolved"],
    candidates: list[ResearchCandidate],
) -> ResearchResolution:
    body = {
        "schema_version": "research-resolution-0.1.0",
        "request_id": request_id,
        "status": status,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    return ResearchResolution(
        resolution_id=stable_id("researchres", body),
        request_id=request_id,
        status=status,
        candidates=candidates,
    )


def _risk(
    item: TerminologyInput,
    term: str,
    trigger: Literal["uppercase_abbreviation", "field_policy_unresolved"],
) -> TerminologyRisk:
    body = {
        "schema_version": "terminology-risk-0.1.0",
        "segment_id": item.segment_id,
        "path": item.path,
        "term": term,
        "trigger": trigger,
        "disposition": "needs_research",
    }
    return TerminologyRisk(
        risk_id=stable_id("risk", body),
        segment_id=item.segment_id,
        path=item.path,
        term=term,
        trigger=trigger,
    )


def _validate_public_https_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ResearchPolicyError("research endpoint must be credential-free HTTPS")
    host = parsed.hostname.rstrip(".").casefold()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(
        (".localhost", ".local", ".internal")
    ):
        raise ResearchPolicyError("research endpoint must not be a private hostname")
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
        raise ResearchPolicyError("research endpoint must not be a private address")
