"""Strict JSON/JSONL parsing and streaming utilities."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TypeVar, cast

from toolcall_tr.hashing import JsonValue, canonical_bytes

T = TypeVar("T")


class StrictJsonError(ValueError):
    """Strict JSON failure with a stable diagnostic code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _reject_constant(value: str) -> None:
    raise StrictJsonError("PARSE_NON_FINITE_NUMBER", f"Non-finite number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError("PARSE_DUPLICATE_KEY", f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_surrogates(value: object, pointer: str = "") -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise StrictJsonError(
                "PARSE_INVALID_UNICODE",
                f"Unpaired Unicode surrogate at {pointer or '/'}",
            )
    elif isinstance(value, list):
        items = cast(list[object], value)
        for index, item in enumerate(items):
            _reject_surrogates(item, f"{pointer}/{index}")
    elif isinstance(value, dict):
        items = cast(dict[str, object], value)
        for key, item in items.items():
            _reject_surrogates(key, f"{pointer}/<key>")
            escaped = key.replace("~", "~0").replace("/", "~1")
            _reject_surrogates(item, f"{pointer}/{escaped}")


def _reject_non_finite_numbers(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise StrictJsonError("PARSE_NON_FINITE_NUMBER", f"Non-finite number is forbidden: {value}")
    if isinstance(value, list):
        for item in cast(list[object], value):
            _reject_non_finite_numbers(item)
    elif isinstance(value, dict):
        for item in cast(dict[str, object], value).values():
            _reject_non_finite_numbers(item)


def _strip_line_terminator(physical_line: bytes) -> bytes:
    if not physical_line.endswith(b"\n"):
        return physical_line
    without_lf = physical_line[:-1]
    return without_lf[:-1] if without_lf.endswith(b"\r") else without_lf


def loads_strict_bytes(raw: bytes) -> JsonValue:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StrictJsonError("PARSE_INVALID_UTF8", str(exc)) from exc
    try:
        value = cast(
            object,
            json.loads(
                text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            ),
        )
    except StrictJsonError:
        raise
    except json.JSONDecodeError as exc:
        raise StrictJsonError("PARSE_INVALID_JSON", str(exc)) from exc
    _reject_non_finite_numbers(value)
    _reject_surrogates(value)
    return cast(JsonValue, value)


def iter_jsonl[T](
    path: Path, parser: Callable[[JsonValue], T] | None = None
) -> Iterator[T | JsonValue]:
    with path.open("rb") as handle:
        for line_number, physical_line in enumerate(handle, start=1):
            raw = _strip_line_terminator(physical_line)
            if not raw:
                raise StrictJsonError(
                    "PARSE_EMPTY_RECORD", f"Empty JSONL record at line {line_number}"
                )
            value = loads_strict_bytes(raw)
            yield parser(value) if parser else value


def write_jsonl(path: Path, records: Iterable[object]) -> tuple[int, int]:
    """Write canonical JSONL to a new file and fsync it; never overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    count = 0
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                payload = canonical_bytes(record) + b"\n"
                handle.write(payload)
                count += 1
                size += len(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return count, size
