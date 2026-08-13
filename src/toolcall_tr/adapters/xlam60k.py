"""Adapter for Salesforce xLAM 60k's embedded-JSON APIGen records.

The source defines function parameters as a map of APIGen/Python-like type
descriptions, not as JSON Schema.  This adapter deterministically maps only
the documented, JSON-representable subset to Draft 2020-12.  It never repairs
an answer, adds a tool result, or guesses an unsupported type.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from toolcall_tr.adapters.base import (
    AdapterError,
    SourceAdapter,
    optional_native_id,
    require_string,
)
from toolcall_tr.hashing import JsonValue
from toolcall_tr.jsonio import StrictJsonError, loads_strict_bytes
from toolcall_tr.models import (
    AdaptedConversation,
    DecisionAction,
    FunctionCall,
    Message,
    RawToolDefinition,
    Role,
    ToolCall,
    ToolFunction,
)

_PRIMITIVE_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "none": "null",
}
_CONTAINER_NAMES = {"dict", "Dict", "list", "List", "set", "Set", "tuple", "Tuple"}


def _parse_embedded_json(value: JsonValue, pointer: str) -> JsonValue:
    if not isinstance(value, str):
        raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", "embedded JSON must be text", pointer)
    try:
        return loads_strict_bytes(value.encode("utf-8"))
    except StrictJsonError as exc:
        raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", exc.code, pointer) from exc


def _top_level_items(value: str) -> list[str]:
    """Split a generic argument list without interpreting nested subexpressions."""
    items: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(value):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced brackets")
        elif char == "," and depth == 0:
            item = value[start:index].strip()
            if not item:
                raise ValueError("empty generic argument")
            items.append(item)
            start = index + 1
    if depth != 0:
        raise ValueError("unbalanced brackets")
    final = value[start:].strip()
    if not final:
        raise ValueError("empty generic argument")
    items.append(final)
    return items


def _strip_qualifiers(source_type: str) -> tuple[str, bool]:
    """Remove top-level optional/default annotations from one APIGen type string."""
    depth = 0
    for index, char in enumerate(source_type):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced brackets")
        elif char == "," and depth == 0:
            suffix = source_type[index + 1 :].strip().lower()
            if suffix.startswith("optional") or suffix.startswith("default"):
                return source_type[:index].strip(), suffix.startswith("optional")
    if depth != 0:
        raise ValueError("unbalanced brackets")
    return source_type.strip(), False


def _generic_expression(value: str) -> tuple[str, list[str]] | None:
    if "[" not in value:
        return None
    if not value.endswith("]"):
        raise ValueError("generic type must end with a closing bracket")
    name, inner = value.split("[", maxsplit=1)
    return name.strip(), _top_level_items(inner[:-1])


def _type_schema(source_type: str, pointer: str) -> dict[str, JsonValue]:
    try:
        expression, _ = _strip_qualifiers(source_type)
        primitive = _PRIMITIVE_TYPES.get(expression.lower())
        if primitive is not None:
            return {"type": primitive}
        if expression in {"list", "List"}:
            return {"type": "array"}
        if expression in {"dict", "Dict"}:
            return {"type": "object"}
        if expression in {"set", "Set"}:
            return {"type": "array", "uniqueItems": True}
        if expression in {"tuple", "Tuple"}:
            return {"type": "array"}

        generic = _generic_expression(expression)
        if generic is None:
            raise ValueError("unknown type")
        name, parts = generic
        if name in {"list", "List"}:
            if len(parts) != 1:
                raise ValueError("List requires one item type")
            return {"type": "array", "items": _type_schema(parts[0], pointer)}
        if name in {"set", "Set"}:
            if len(parts) != 1:
                raise ValueError("Set requires one item type")
            return {
                "type": "array",
                "items": _type_schema(parts[0], pointer),
                "uniqueItems": True,
            }
        if name in {"dict", "Dict"}:
            if len(parts) not in {1, 2}:
                raise ValueError("Dict requires one or two type arguments")
            schema: dict[str, JsonValue] = {"type": "object"}
            if len(parts) == 2:
                schema["additionalProperties"] = _type_schema(parts[1], pointer)
            return schema
        if name in {"tuple", "Tuple"}:
            if not parts:
                raise ValueError("Tuple requires at least one item type")
            return {
                "type": "array",
                "prefixItems": [_type_schema(part, pointer) for part in parts],
                "items": False,
            }
        if name == "Union":
            if len(parts) < 2:
                raise ValueError("Union requires at least two type arguments")
            return {"anyOf": [_type_schema(part, pointer) for part in parts]}
        if name == "Optional":
            if len(parts) != 1:
                raise ValueError("Optional requires one type argument")
            return {"anyOf": [_type_schema(parts[0], pointer), {"type": "null"}]}
        raise ValueError("unknown generic type")
    except ValueError as exc:
        raise AdapterError(
            "SOURCE_ADAPTER_INVALID_FIELD",
            f"unsupported xLAM parameter type at {pointer}",
            pointer,
        ) from exc


def _property_schema(
    value: JsonValue,
    *,
    pointer: str,
) -> tuple[dict[str, JsonValue], bool]:
    if not isinstance(value, dict):
        raise AdapterError(
            "SOURCE_ADAPTER_INVALID_FIELD", "parameter definition must be an object", pointer
        )
    source_type = value.get("type")
    description = value.get("description")
    if not isinstance(source_type, str) or not source_type.strip():
        raise AdapterError(
            "SOURCE_ADAPTER_INVALID_FIELD", "parameter type must be non-empty text", pointer
        )
    if description is not None and not isinstance(description, str):
        raise AdapterError(
            "SOURCE_ADAPTER_INVALID_FIELD", "parameter description must be text", pointer
        )
    schema = _type_schema(source_type, pointer)
    if description is not None:
        schema["description"] = description
    default = value.get("default")
    if default is not None:
        schema["default"] = default
    _, optional = _strip_qualifiers(source_type)
    return schema, optional or "default" in value


def _tools(value: JsonValue) -> list[RawToolDefinition]:
    parsed = _parse_embedded_json(value, "/tools")
    if not isinstance(parsed, list):
        raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", "tools must be an array", "/tools")
    definitions: list[RawToolDefinition] = []
    for index, raw_tool in enumerate(parsed):
        pointer = f"/tools/{index}"
        if not isinstance(raw_tool, dict):
            raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", "tool must be an object", pointer)
        name = raw_tool.get("name")
        description = raw_tool.get("description")
        parameters = raw_tool.get("parameters")
        if not isinstance(name, str) or not name or not isinstance(parameters, dict):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", "tool name and parameter map are required", pointer
            )
        if description is not None and not isinstance(description, str):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", "tool description must be text", pointer
            )
        properties: dict[str, JsonValue] = {}
        required: list[str] = []
        for parameter_name, parameter_value in parameters.items():
            if not parameter_name:
                raise AdapterError(
                    "SOURCE_ADAPTER_INVALID_FIELD", "parameter names must be non-empty", pointer
                )
            property_schema, is_optional = _property_schema(
                parameter_value,
                pointer=f"{pointer}/parameters/{parameter_name}",
            )
            properties[parameter_name] = property_schema
            if not is_optional:
                required.append(parameter_name)
        schema: dict[str, JsonValue] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = cast(list[JsonValue], required)
        definitions.append(
            RawToolDefinition(
                type="function",
                function=ToolFunction(
                    name=name,
                    description=description,
                    parameters=schema,
                    strict=None,
                ),
            )
        )
    return definitions


def _calls(
    value: JsonValue,
    conversation_id: str,
    tools: Iterable[RawToolDefinition],
) -> list[ToolCall]:
    parsed = _parse_embedded_json(value, "/answers")
    if not isinstance(parsed, list) or not parsed:
        raise AdapterError(
            "SOURCE_ADAPTER_INVALID_FIELD", "answers must be a non-empty array", "/answers"
        )
    tool_names = {tool.function.name for tool in tools}
    calls: list[ToolCall] = []
    for index, raw_call in enumerate(parsed):
        pointer = f"/answers/{index}"
        if not isinstance(raw_call, dict):
            raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", "answer must be an object", pointer)
        name = raw_call.get("name")
        arguments = raw_call.get("arguments")
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", "answer requires name and object arguments", pointer
            )
        if name not in tool_names:
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD",
                "answer tool name is not presented",
                f"{pointer}/name",
            )
        calls.append(
            ToolCall(
                id=f"call_{conversation_id}_{index}",
                type="function",
                function=FunctionCall(name=name, arguments=arguments),
            )
        )
    return calls


class Xlam60kAdapter(SourceAdapter):
    """Map xLAM 60k's ordered answer list to one source-explicit tool-call turn."""

    name = "xlam60k"

    def adapt(self, record: dict[str, JsonValue]) -> AdaptedConversation:
        conversation_id = optional_native_id(record)
        query = require_string(record, "query")
        tools = _tools(record.get("tools"))
        calls = _calls(record.get("answers"), conversation_id, tools)
        return AdaptedConversation(
            source_conversation_id=conversation_id,
            conversation=[
                Message(
                    role=Role.USER,
                    content=query,
                    reasoning_content=None,
                    thinking=None,
                    tool_calls=None,
                    images=None,
                    name=None,
                    tool_call_id=None,
                ),
                Message(
                    role=Role.ASSISTANT,
                    content=None,
                    reasoning_content=None,
                    thinking=None,
                    tool_calls=calls,
                    images=None,
                    name=None,
                    tool_call_id=None,
                ),
            ],
            tools=tools,
            target_message_index=1,
            decision_action=DecisionAction.TOOL_CALL,
            observed_paths=["/id", "/query", "/tools", "/answers"],
        )
