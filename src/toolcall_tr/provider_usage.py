"""Redaction-safe provider token usage and deterministic USD estimates.

The provider adapters retain only counters reported by the provider response.
They never persist request or response text, credentials, headers, or remote
error messages.  Price cards are intentionally versioned in code: they are
estimates for operator visibility, not a replacement for a provider invoice.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from toolcall_tr.hashing import canonical_bytes, stable_id
from toolcall_tr.jsonio import iter_jsonl
from toolcall_tr.models import NonEmptyStr, StrictModel
from toolcall_tr.provider_provenance import (
    ProviderAttemptOutcome,
    ProviderAttemptRecord,
)

ProviderUsageId = Annotated[str, Field(pattern=r"^pvusage_[0-9a-f]{64}$")]


class ProviderUsageRecord(StrictModel):
    """Hash-linked token counts for one provider response.

    ``input_tokens`` includes cached input tokens when a provider reports
    them.  ``cached_input_tokens`` is therefore a priced subset, never an
    additional token count.
    """

    schema_version: Literal["provider-usage-0.1.0"] = "provider-usage-0.1.0"
    usage_id: ProviderUsageId
    attempt_id: Annotated[str, Field(pattern=r"^pvattempt_[0-9a-f]{64}$")]
    provider: NonEmptyStr
    model: NonEmptyStr
    operation: Literal["translation", "judge"]
    input_tokens: Annotated[int, Field(ge=0)]
    cached_input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    source: Literal["provider_response"] = "provider_response"

    @model_validator(mode="after")
    def validate_identity(self) -> ProviderUsageRecord:
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input tokens cannot exceed total input tokens")
        body = self.model_dump(mode="json", exclude={"usage_id"})
        if self.usage_id != stable_id("pvusage", body):
            raise ValueError("provider usage ID does not match deterministic identity")
        return self


ProviderUsageSink = Callable[[ProviderUsageRecord], None]


class ProviderUsageSinkError(RuntimeError):
    """Raised if a usage sidecar cannot be persisted after a live request."""


def _usage_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def provider_usage_from_response(
    *, attempt: ProviderAttemptRecord, response_body: bytes | None
) -> ProviderUsageRecord | None:
    """Extract token counters from an opaque provider response without retaining it."""
    if response_body is None or not attempt.preflight.allowed:
        return None
    try:
        parsed = cast(object, json.loads(response_body))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, Mapping):
        return None
    raw_usage = cast(Mapping[str, object], parsed).get("usage")
    if not isinstance(raw_usage, Mapping):
        return None
    usage = cast(Mapping[str, object], raw_usage)
    if attempt.provider == "deepseek":
        input_tokens = _usage_integer(usage.get("prompt_tokens"))
        output_tokens = _usage_integer(usage.get("completion_tokens"))
        cached = _usage_integer(usage.get("prompt_cache_hit_tokens")) or 0
    elif attempt.provider == "openai":
        input_tokens = _usage_integer(usage.get("input_tokens"))
        output_tokens = _usage_integer(usage.get("output_tokens"))
        details = usage.get("input_tokens_details")
        cached = (
            _usage_integer(cast(Mapping[str, object], details).get("cached_tokens"))
            if isinstance(details, Mapping)
            else 0
        ) or 0
    else:
        return None
    if input_tokens is None or output_tokens is None or cached > input_tokens:
        return None
    body = {
        "schema_version": "provider-usage-0.1.0",
        "attempt_id": attempt.attempt_id,
        "provider": attempt.provider,
        "model": attempt.model,
        "operation": attempt.operation.value,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output_tokens,
        "source": "provider_response",
    }
    return ProviderUsageRecord(
        usage_id=stable_id("pvusage", body),
        attempt_id=attempt.attempt_id,
        provider=attempt.provider,
        model=attempt.model,
        operation=attempt.operation.value,
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
    )


def emit_provider_usage(sink: ProviderUsageSink | None, record: ProviderUsageRecord) -> None:
    """Emit one safe token sidecar; a broken sink fails closed."""
    if sink is None:
        return
    try:
        sink(record)
    except Exception as exc:
        raise ProviderUsageSinkError("provider usage audit sink failed") from exc


def emit_response_usage(
    sink: ProviderUsageSink | None,
    *,
    attempt: ProviderAttemptRecord,
    response_body: bytes | None,
) -> None:
    """Parse and emit a response usage record when counters are available."""
    record = provider_usage_from_response(attempt=attempt, response_body=response_body)
    if record is not None:
        emit_provider_usage(sink, record)


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD price card per one million tokens, fixed at the noted effective date."""

    input_usd_per_million: Decimal
    cached_input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    effective_from: str


# Direct API standard rates verified on 2026-08-13.  DeepSeek changes on
# 2026-08-16 are intentionally not applied before their published effective date.
_PRICE_CARDS: dict[str, ModelPrice] = {
    "deepseek-v4-flash": ModelPrice(
        Decimal("0.14"), Decimal("0.0028"), Decimal("0.28"), "2026-04-24"
    ),
    "deepseek-v4-pro": ModelPrice(
        Decimal("0.435"), Decimal("0.003625"), Decimal("0.87"), "2026-04-24"
    ),
    "gpt-5.4": ModelPrice(Decimal("2.50"), Decimal("0.25"), Decimal("15.00"), "2026-03-05"),
    "gpt-5.4-mini": ModelPrice(Decimal("0.75"), Decimal("0.075"), Decimal("4.50"), "2026-03-17"),
}


@dataclass(frozen=True, slots=True)
class CostLine:
    provider: str
    model: str
    requests: int
    completed_responses: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    priced_responses: int
    estimated_usd: Decimal | None


def _read_attempts(root: Path) -> dict[str, ProviderAttemptRecord]:
    records: dict[str, ProviderAttemptRecord] = {}
    for path in sorted(root.rglob("*.json")):
        if "provider-attempt" not in path.as_posix():
            continue
        try:
            record = ProviderAttemptRecord.model_validate_json(
                path.read_text(encoding="utf-8"), strict=True
            )
        except (OSError, ValueError):
            continue
        records[record.attempt_id] = record
    for path in sorted(root.rglob("*.jsonl")):
        if "attempt" not in path.as_posix():
            continue
        try:
            for value in iter_jsonl(path):
                record = ProviderAttemptRecord.model_validate_json(
                    canonical_bytes(value), strict=True
                )
                records[record.attempt_id] = record
        except (OSError, ValueError):
            continue
    return records


def _read_usage(root: Path) -> dict[str, ProviderUsageRecord]:
    records: dict[str, ProviderUsageRecord] = {}
    for path in sorted(root.rglob("*.json")):
        if "provider-usage" not in path.as_posix():
            continue
        try:
            record = ProviderUsageRecord.model_validate_json(
                path.read_text(encoding="utf-8"), strict=True
            )
        except (OSError, ValueError):
            continue
        records[record.usage_id] = record
    return records


def cost_lines(root: Path) -> list[CostLine]:
    """Aggregate one output tree into model-specific actual-token estimates."""
    attempts = _read_attempts(root)
    usage = _read_usage(root)
    grouped_attempts: dict[tuple[str, str], list[ProviderAttemptRecord]] = defaultdict(list)
    for attempt in attempts.values():
        grouped_attempts[(attempt.provider, attempt.model)].append(attempt)
    grouped_usage: dict[tuple[str, str], list[ProviderUsageRecord]] = defaultdict(list)
    for record in usage.values():
        grouped_usage[(record.provider, record.model)].append(record)
    lines: list[CostLine] = []
    for provider, model in sorted(set(grouped_attempts) | set(grouped_usage)):
        model_attempts = grouped_attempts[(provider, model)]
        model_usage = grouped_usage[(provider, model)]
        input_tokens = sum(item.input_tokens for item in model_usage)
        cached = sum(item.cached_input_tokens for item in model_usage)
        output_tokens = sum(item.output_tokens for item in model_usage)
        card = _PRICE_CARDS.get(model)
        estimated: Decimal | None = None
        if card is not None and model_usage:
            uncached = input_tokens - cached
            estimated = (
                Decimal(uncached) * card.input_usd_per_million
                + Decimal(cached) * card.cached_input_usd_per_million
                + Decimal(output_tokens) * card.output_usd_per_million
            ) / Decimal(1_000_000)
        lines.append(
            CostLine(
                provider=provider,
                model=model,
                requests=sum(
                    attempt.outcome is not ProviderAttemptOutcome.PREFLIGHT_BLOCKED
                    for attempt in model_attempts
                ),
                completed_responses=len(model_usage),
                input_tokens=input_tokens,
                cached_input_tokens=cached,
                output_tokens=output_tokens,
                priced_responses=len(model_usage),
                estimated_usd=estimated,
            )
        )
    return lines


def estimated_total_usd(lines: Iterable[CostLine]) -> Decimal:
    return sum((line.estimated_usd or Decimal(0) for line in lines), Decimal(0))


def format_usd(value: Decimal | None) -> str:
    """Render a compact, stable operator-facing USD amount."""
    if value is None:
        return "n/a (usage unavailable)"
    return f"${value.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP):f}"
