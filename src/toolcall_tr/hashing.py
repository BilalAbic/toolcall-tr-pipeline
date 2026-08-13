"""RFC 8785/JCS hashing and stable identifier helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Protocol, cast

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class _Rfc8785Module(Protocol):
    def dumps(self, value: JsonValue, /) -> bytes: ...


class _ModelDump(Protocol):
    def model_dump(self, *, mode: str, exclude_none: bool) -> object: ...


_rfc8785 = cast(_Rfc8785Module, import_module("rfc8785"))


class CanonicalizationError(ValueError):
    """Raised when a value cannot be represented by RFC 8785."""


def to_json_value(value: object) -> JsonValue:
    """Convert Pydantic-like values into the closed JSON value algebra."""
    if hasattr(value, "model_dump"):
        value = cast(_ModelDump, value).model_dump(mode="json", exclude_none=False)
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        converted: dict[str, JsonValue] = {}
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise CanonicalizationError("JCS object keys must be strings")
            converted[key] = to_json_value(item)
        return converted
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        sequence = cast(Sequence[object], value)
        return [to_json_value(item) for item in sequence]
    raise CanonicalizationError(f"Unsupported canonical JSON value: {type(value).__name__}")


def canonical_bytes(value: object) -> bytes:
    """Return the RFC 8785 canonical UTF-8 representation."""
    try:
        return _rfc8785.dumps(to_json_value(value))
    except (ValueError, OverflowError) as exc:
        raise CanonicalizationError(str(exc)) from exc


def sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_jcs(value: object) -> str:
    return sha256_bytes(canonical_bytes(value))


def stable_id(prefix: str, value: object) -> str:
    """Build a readable ID without weakening the underlying SHA-256 identity."""
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("ID prefix must be non-empty and alphanumeric/underscore")
    return f"{prefix}_{hashlib.sha256(canonical_bytes(value)).hexdigest()}"
