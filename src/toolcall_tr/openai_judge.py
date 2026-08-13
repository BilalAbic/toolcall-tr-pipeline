"""Strict OpenAI Responses adapter for model-only evaluation triage.

This module maps a structured provider response into the local MQM contract.
It cannot produce a Gold acceptance; that remains an explicit human action in
``eval_contract``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast
from urllib.parse import urlsplit

from pydantic import model_validator

from toolcall_tr.config import PipelineConfig, ProviderRole
from toolcall_tr.eval_contract import (
    EvaluationUnit,
    ModelConclusion,
    ModelEvaluationVerdict,
    MqmCategory,
    MqmSeverity,
    SegmentPathEvidence,
    build_model_verdict,
    build_mqm_finding,
)
from toolcall_tr.hashing import JsonValue, canonical_bytes, to_json_value
from toolcall_tr.live_preflight import (
    LivePreflightBlockedError,
    LivePreflightDecision,
    preflight_live_request,
)
from toolcall_tr.models import NonEmptyStr, StrictModel
from toolcall_tr.provider_adapter import (
    ProviderConfigurationError,
    ProviderGateError,
    ProviderResponseError,
    ResponsesTransport,
)
from toolcall_tr.provider_provenance import (
    ProviderAttemptSink,
    ProviderOperation,
    build_provider_attempt_record,
    emit_provider_attempt,
)
from toolcall_tr.provider_usage import ProviderUsageSink, emit_response_usage

_OPENAI_HOST = "api.openai.com"
_OPENAI_PATH = "/v1/responses"


class JudgeFindingOutput(StrictModel):
    """Provider-returned atomic finding before host-side content addressing."""

    category: MqmCategory
    severity: MqmSeverity
    source_excerpt: NonEmptyStr
    target_excerpt: NonEmptyStr
    rationale: NonEmptyStr


class JudgeOutput(StrictModel):
    """All fields are required so the strict Responses schema is provider-valid."""

    conclusion: ModelConclusion
    findings: list[JudgeFindingOutput]
    unresolved_reasons: list[NonEmptyStr]

    @model_validator(mode="after")
    def validate_conclusion(self) -> JudgeOutput:
        if self.conclusion == "pass" and (self.findings or self.unresolved_reasons):
            raise ValueError("pass output cannot contain findings or unresolved reasons")
        if self.conclusion == "fail" and (not self.findings or self.unresolved_reasons):
            raise ValueError("fail output requires findings and no unresolved reasons")
        if self.conclusion == "needs_human_review" and not self.unresolved_reasons:
            raise ValueError("needs_human_review output requires unresolved reasons")
        return self


def validate_openai_endpoint(endpoint: str) -> None:
    """Allow only the public OpenAI Responses endpoint for this judge."""
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _OPENAI_HOST
        or parsed.path.rstrip("/") != _OPENAI_PATH
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ProviderConfigurationError(
            "judge endpoint is not the approved OpenAI Responses endpoint"
        )


def _judge_output_schema() -> dict[str, JsonValue]:
    """Return a strict structured-output schema with required fields at every object."""
    schema = cast(dict[str, JsonValue], to_json_value(JudgeOutput.model_json_schema()))
    _require_all_properties(schema)
    return schema


def _require_all_properties(value: object) -> None:
    """Assert (rather than mutate) OpenAI strict-schema required-key invariants."""
    if not isinstance(value, Mapping):
        return
    mapping = cast(Mapping[str, object], value)
    properties = mapping.get("properties")
    if isinstance(properties, Mapping):
        keys = list(cast(Mapping[str, object], properties))
        required = mapping.get("required")
        if not isinstance(required, list):
            raise ProviderConfigurationError(
                "OpenAI strict structured-output schema must require every property"
            )
        required_names = cast(list[object], required)
        if not all(isinstance(item, str) for item in required_names) or {
            cast(str, item) for item in required_names
        } != set(keys):
            raise ProviderConfigurationError(
                "OpenAI strict structured-output schema must require every property"
            )
    for item in mapping.values():
        if isinstance(item, Mapping):
            _require_all_properties(cast(Mapping[str, object], item))
        elif isinstance(item, list):
            for child in cast(list[object], item):
                _require_all_properties(child)


def serialize_openai_judge_request(
    *,
    role: ProviderRole,
    evaluation_unit: EvaluationUnit,
    evidence: SegmentPathEvidence,
    max_output_tokens: int,
) -> bytes:
    """Create one bounded, strict-schema Responses request for blind triage."""
    if role.model not in {"gpt-5.4", "gpt-5.4-mini"}:
        raise ProviderConfigurationError("judge model is not an approved OpenAI judge model")
    if not 1 <= max_output_tokens <= 2_048:
        raise ProviderConfigurationError("judge max_output_tokens must be between 1 and 2048")
    request_input: dict[str, JsonValue] = {
        "evaluation_unit": evaluation_unit.model_dump(mode="json"),
        "evidence": evidence.model_dump(mode="json"),
    }
    body: dict[str, JsonValue] = {
        "model": role.model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are a blind Turkish translation triage judge. Evaluate only "
                            "the supplied source and target excerpts. Do not accept a record "
                            "into Gold. Return exactly the required JSON object; the schema is "
                            "the only output contract. A pass requires findings=[] and "
                            "unresolved_reasons=[]; do not use pass when a material issue is "
                            "present. A fail requires one or more grounded findings and "
                            "unresolved_reasons=[]. Use needs_human_review only when you cannot "
                            "reach a supported decision, and then provide one or more concise "
                            "unresolved_reasons. For every finding, quote non-empty, contiguous "
                            "source and target excerpts verbatim from the supplied input. Do not "
                            "invent excerpts, use Markdown, add prose, or refer to information "
                            "outside the supplied unit."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": canonical_bytes(request_input).decode("utf-8"),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "translation_triage",
                "strict": True,
                "schema": _judge_output_schema(),
            }
        },
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    return canonical_bytes(body)


def _response_output_text(raw_response: bytes) -> str:
    """Extract one completed output_text and reject refusal/incomplete envelopes."""
    try:
        parsed = cast(object, json.loads(raw_response))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderResponseError("OpenAI transport returned invalid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ProviderResponseError("OpenAI transport returned a non-object envelope")
    envelope = cast(Mapping[str, object], parsed)
    if envelope.get("status") not in {None, "completed"}:
        raise ProviderResponseError("OpenAI response did not complete")
    output = envelope.get("output")
    if not isinstance(output, list):
        raise ProviderResponseError("OpenAI response is missing an output list")
    texts: list[str] = []
    for raw_item in cast(list[object], output):
        if not isinstance(raw_item, Mapping):
            continue
        item = cast(Mapping[str, object], raw_item)
        if item.get("type") == "refusal":
            raise ProviderResponseError("OpenAI response was refused")
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for raw_part in cast(list[object], content):
            if not isinstance(raw_part, Mapping):
                continue
            part = cast(Mapping[str, object], raw_part)
            if part.get("type") == "refusal":
                raise ProviderResponseError("OpenAI response was refused")
            if part.get("type") == "output_text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ProviderResponseError("OpenAI output_text must contain a string")
                texts.append(text)
    if len(texts) != 1:
        raise ProviderResponseError("OpenAI response must contain exactly one output_text")
    return texts[0]


class OpenAIResponsesJudge:
    """Run model-only triage through a gated, injected OpenAI transport."""

    def __init__(
        self,
        *,
        config: PipelineConfig,
        role_name: str,
        transport: ResponsesTransport,
        max_output_tokens: int = 512,
        attempt_sink: ProviderAttemptSink | None = None,
        usage_sink: ProviderUsageSink | None = None,
    ) -> None:
        if role_name not in {"strong_judge", "mini_verifier"}:
            raise ValueError("OpenAI judge role must be strong_judge or mini_verifier")
        self._config = config
        self._role_name = role_name
        self._transport = transport
        self._max_output_tokens = max_output_tokens
        self._attempt_sink = attempt_sink
        self._usage_sink = usage_sink

    def _emit_terminal(
        self,
        *,
        request_body: bytes,
        preflight: LivePreflightDecision,
        provider: str,
        model: str,
        endpoint: str,
        error: BaseException | None = None,
        response_body: bytes | None = None,
    ) -> None:
        """Persist safe provenance and provider-reported token counters once."""
        record = build_provider_attempt_record(
            operation=ProviderOperation.JUDGE,
            provider=provider,
            model=model,
            endpoint=endpoint,
            request_body=request_body,
            preflight=preflight,
            error=error,
            response_body=response_body,
        )
        emit_provider_attempt(self._attempt_sink, record)
        emit_response_usage(
            self._usage_sink,
            attempt=record,
            response_body=response_body,
        )

    def judge(
        self,
        *,
        evaluation_unit: EvaluationUnit,
        evidence: SegmentPathEvidence,
    ) -> ModelEvaluationVerdict:
        provider = self._config.providers
        if not provider.enabled:
            raise ProviderGateError("provider execution is disabled by configuration")
        if not provider.network_egress_enabled:
            raise ProviderGateError("network egress is disabled by configuration")
        role = getattr(provider, self._role_name)
        if role.provider != "openai" or role.endpoint is None:
            raise ProviderConfigurationError("judge must be an explicitly configured OpenAI role")
        validate_openai_endpoint(role.endpoint)
        if (
            evidence.segment_id != evaluation_unit.segment_id
            or evidence.path != evaluation_unit.path
        ):
            raise ValueError("judge evidence must match the evaluation unit")
        request_body = serialize_openai_judge_request(
            role=role,
            evaluation_unit=evaluation_unit,
            evidence=evidence,
            max_output_tokens=self._max_output_tokens,
        )
        preflight = preflight_live_request(
            config=self._config,
            provider=role.provider,
            endpoint=role.endpoint,
            payload=canonical_bytes(
                {
                    "source_excerpt": evidence.source_excerpt,
                    "target_excerpt": evidence.target_excerpt,
                }
            ),
        )
        if not preflight.allowed:
            self._emit_terminal(
                request_body=request_body,
                preflight=preflight,
                provider=role.provider,
                model=role.model,
                endpoint=role.endpoint,
            )
            raise LivePreflightBlockedError(preflight)

        raw_response: bytes | None = None
        try:
            raw_response = self._transport.create_response(
                endpoint=role.endpoint,
                request_body=request_body,
            )
            output = JudgeOutput.model_validate_json(
                _response_output_text(raw_response), strict=True
            )
            findings = [
                _to_local_finding(
                    output=item,
                    evaluation_unit=evaluation_unit,
                    evidence=evidence,
                )
                for item in output.findings
            ]
            verdict = build_model_verdict(
                evaluation_unit=evaluation_unit,
                evaluator_label=f"openai:{self._role_name}:{role.model}",
                conclusion=output.conclusion,
                findings=findings,
                unresolved_reasons=sorted(set(output.unresolved_reasons)),
            )
        except (ValueError, TypeError) as exc:
            error = ProviderResponseError("OpenAI output violates the local judge contract")
            self._emit_terminal(
                request_body=request_body,
                preflight=preflight,
                provider=role.provider,
                model=role.model,
                endpoint=role.endpoint,
                error=error,
                response_body=raw_response,
            )
            raise error from exc
        except Exception as exc:
            self._emit_terminal(
                request_body=request_body,
                preflight=preflight,
                provider=role.provider,
                model=role.model,
                endpoint=role.endpoint,
                error=exc,
                response_body=raw_response,
            )
            raise
        self._emit_terminal(
            request_body=request_body,
            preflight=preflight,
            provider=role.provider,
            model=role.model,
            endpoint=role.endpoint,
            response_body=raw_response,
        )
        return verdict


def _to_local_finding(
    *,
    output: JudgeFindingOutput,
    evaluation_unit: EvaluationUnit,
    evidence: SegmentPathEvidence,
):
    if (
        output.source_excerpt not in evidence.source_excerpt
        or output.target_excerpt not in evidence.target_excerpt
    ):
        raise ValueError("judge finding excerpts must be grounded in supplied evidence")
    return build_mqm_finding(
        category=output.category,
        severity=output.severity,
        evidence=SegmentPathEvidence(
            segment_id=evaluation_unit.segment_id,
            path=evaluation_unit.path,
            source_excerpt=output.source_excerpt,
            target_excerpt=output.target_excerpt,
        ),
        rationale=output.rationale,
    )
