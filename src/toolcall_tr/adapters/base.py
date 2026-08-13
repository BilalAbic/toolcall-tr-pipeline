"""Source adapter interface and strict extraction helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from toolcall_tr.hashing import JsonValue
from toolcall_tr.models import AdaptedConversation, RawToolDefinition, ToolFunction


class AdapterError(ValueError):
    def __init__(self, code: str, message: str, pointer: str) -> None:
        super().__init__(message)
        self.code = code
        self.pointer = pointer


def require_string(record: dict[str, JsonValue], key: str) -> str:
    if key not in record:
        raise AdapterError(
            "SOURCE_ADAPTER_MISSING_FIELD", f"missing required field: {key}", f"/{key}"
        )
    value = record[key]
    if not isinstance(value, str) or not value:
        raise AdapterError(
            "SOURCE_ADAPTER_INVALID_FIELD", f"{key} must be a non-empty string", f"/{key}"
        )
    return value


def optional_native_id(record: dict[str, JsonValue]) -> str:
    value = record.get("id")
    if isinstance(value, str | int) and str(value):
        return str(value)
    raise AdapterError("SOURCE_ADAPTER_MISSING_FIELD", "missing source conversation ID", "/id")


def parse_tools(value: JsonValue, pointer: str = "/tools") -> list[RawToolDefinition]:
    if not isinstance(value, list):
        raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", "tools must be an array", pointer)
    definitions: list[RawToolDefinition] = []
    for index, item in enumerate(value):
        item_pointer = f"{pointer}/{index}"
        if not isinstance(item, dict):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", "tool must be an object", item_pointer
            )
        # Source fixture shape is intentionally simple and unambiguous.
        if "function" in item:
            raw_function = item.get("function")
            raw_type = item.get("type", "function")
        else:
            raw_function = item
            raw_type = "function"
        if raw_type != "function" or not isinstance(raw_function, dict):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", "only function tools are supported", item_pointer
            )
        name = raw_function.get("name")
        description = raw_function.get("description")
        parameters = raw_function.get("parameters")
        strict = raw_function.get("strict")
        if not isinstance(name, str) or not name or not isinstance(parameters, dict):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD",
                "tool name and parameter schema are required",
                item_pointer,
            )
        if description is not None and not isinstance(description, str):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", "description must be text or null", item_pointer
            )
        if strict is not None and not isinstance(strict, bool):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", "strict must be boolean or null", item_pointer
            )
        definitions.append(
            RawToolDefinition(
                type="function",
                function=ToolFunction(
                    name=name,
                    description=description,
                    parameters=parameters,
                    strict=strict,
                ),
            )
        )
    return definitions


class SourceAdapter(ABC):
    name: str

    @abstractmethod
    def adapt(self, record: dict[str, JsonValue]) -> AdaptedConversation:
        """Map source fields without repairing or inventing source behavior."""
