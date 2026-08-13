"""Adapter for NVIDIA When2Call's source-explicit decision benchmark.

The upstream JSONL stores tools and the selected tool call as JSON strings.
The adapter parses those strings strictly, converts only documented APIGen type
aliases to Draft 2020-12 primitives, and preserves the selected outcome
without inventing a tool result or final answer.
"""

from __future__ import annotations

import re
from typing import cast

from toolcall_tr.adapters.base import AdapterError, SourceAdapter, require_string
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

_DECISIONS = {
    "tool_call": DecisionAction.TOOL_CALL,
    "request_for_info": DecisionAction.CLARIFICATION,
    "cannot_answer": DecisionAction.TOOL_UNAVAILABLE,
}
_TYPE_ALIASES = {
    "array": "array",
    "boolean": "boolean",
    "bool": "boolean",
    "dict": "object",
    "dictionary": "object",
    "float": "number",
    "integer": "integer",
    "int": "integer",
    "number": "number",
    "object": "object",
    "str": "string",
    "string": "string",
}
_GENERIC_ARRAY_TYPE = re.compile(r"^(?:list|set|tuple)(?:\[.*\])?$", re.IGNORECASE)
_GENERIC_OBJECT_TYPE = re.compile(r"^(?:dict|mapping)(?:\[.*\])?$", re.IGNORECASE)


def parse_embedded_json(value: JsonValue, pointer: str) -> JsonValue:
    if not isinstance(value, str):
        raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", "embedded JSON must be text", pointer)
    try:
        return loads_strict_bytes(value.encode("utf-8"))
    except StrictJsonError as exc:
        raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", exc.code, pointer) from exc


def _convert_schema(value: JsonValue, pointer: str) -> JsonValue:
    if isinstance(value, bool):
        return value
    if not isinstance(value, dict):
        raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", "schema must be an object", pointer)
    result = dict(value)
    if "type" in result:
        source_type = result["type"]
        if not isinstance(source_type, str):
            raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", "schema type must be text", pointer)
        normalized_type = source_type.strip().split(",", maxsplit=1)[0].strip()
        normalized_key = normalized_type.casefold()
        if normalized_key == "any":
            del result["type"]
        elif normalized_key in _TYPE_ALIASES:
            result["type"] = _TYPE_ALIASES[normalized_key]
        elif _GENERIC_ARRAY_TYPE.fullmatch(normalized_type):
            # APIGen's Python collection annotations do not always include an
            # item schema.  ``array`` is the source-supported JSON shape; we
            # deliberately do not invent an item constraint.
            result["type"] = "array"
        elif _GENERIC_OBJECT_TYPE.fullmatch(normalized_type):
            result["type"] = "object"
        else:
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", f"unsupported APIGen type: {source_type}", pointer
            )
    properties = result.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", "properties must be an object", pointer
            )
        result["properties"] = {
            name: _convert_schema(schema, f"{pointer}/properties/{name}")
            for name, schema in properties.items()
        }
    items = result.get("items")
    if items is not None:
        result["items"] = _convert_schema(items, f"{pointer}/items")
    return cast(JsonValue, result)


def parse_apigen_tools(value: JsonValue) -> list[RawToolDefinition]:
    if not isinstance(value, list):
        raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", "tools must be an array", "/tools")
    tools: list[RawToolDefinition] = []
    for index, raw_tool in enumerate(value):
        pointer = f"/tools/{index}"
        parsed = parse_embedded_json(raw_tool, pointer)
        if not isinstance(parsed, dict):
            raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", "tool must be an object", pointer)
        name = parsed.get("name")
        description = parsed.get("description")
        parameters = parsed.get("parameters")
        if not isinstance(name, str) or not name or not isinstance(parameters, dict):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", "tool name and parameters are required", pointer
            )
        if description is not None and not isinstance(description, str):
            raise AdapterError("SOURCE_ADAPTER_INVALID_FIELD", "description must be text", pointer)
        converted = _convert_schema(parameters, f"{pointer}/parameters")
        if not isinstance(converted, dict):  # pragma: no cover - object input guarantees this
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", "parameters must be an object", pointer
            )
        tools.append(
            RawToolDefinition(
                type="function",
                function=ToolFunction(
                    name=name,
                    description=description,
                    parameters=converted,
                    strict=None,
                ),
            )
        )
    return tools


class When2CallAdapter(SourceAdapter):
    """Map each selected When2Call decision to one canonical target turn."""

    name = "when2call"

    def adapt(self, record: dict[str, JsonValue]) -> AdaptedConversation:
        source_id = require_string(record, "uuid")
        question = require_string(record, "question")
        decision_label = require_string(record, "correct_answer")
        try:
            action = _DECISIONS[decision_label]
        except KeyError as exc:
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD",
                f"unsupported When2Call decision: {decision_label}",
                "/correct_answer",
            ) from exc
        answers = record.get("answers")
        if not isinstance(answers, dict):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", "answers must be an object", "/answers"
            )
        selected = answers.get(decision_label)
        if not isinstance(selected, str) or not selected:
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD",
                "selected answer must be non-empty text",
                f"/answers/{decision_label}",
            )
        tools = parse_apigen_tools(record.get("tools"))
        user = Message(
            role=Role.USER,
            content=question,
            reasoning_content=None,
            thinking=None,
            tool_calls=None,
            images=None,
            name=None,
            tool_call_id=None,
        )
        if action is DecisionAction.TOOL_CALL:
            call = parse_embedded_json(selected, "/answers/tool_call")
            if not isinstance(call, dict):
                raise AdapterError(
                    "SOURCE_ADAPTER_INVALID_FIELD",
                    "tool call must be an object",
                    "/answers/tool_call",
                )
            name = call.get("name")
            arguments = call.get("arguments")
            if not isinstance(name, str) or not name or not isinstance(arguments, dict):
                raise AdapterError(
                    "SOURCE_ADAPTER_INVALID_FIELD",
                    "tool call requires name and object arguments",
                    "/answers/tool_call",
                )
            target = Message(
                role=Role.ASSISTANT,
                content=None,
                reasoning_content=None,
                thinking=None,
                tool_calls=[
                    ToolCall(
                        id=f"call_{source_id}",
                        type="function",
                        function=FunctionCall(name=name, arguments=arguments),
                    )
                ],
                images=None,
                name=None,
                tool_call_id=None,
            )
        else:
            target = Message(
                role=Role.ASSISTANT,
                content=selected,
                reasoning_content=None,
                thinking=None,
                tool_calls=None,
                images=None,
                name=None,
                tool_call_id=None,
            )
        return AdaptedConversation(
            source_conversation_id=source_id,
            conversation=[user, target],
            tools=tools,
            target_message_index=1,
            decision_action=action,
            observed_paths=["/uuid", "/question", "/correct_answer", "/answers", "/tools"],
        )
