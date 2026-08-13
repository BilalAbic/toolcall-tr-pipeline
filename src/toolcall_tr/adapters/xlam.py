"""Adapter for xLAM-style tool-call-only source fixtures."""

from __future__ import annotations

from toolcall_tr.adapters.base import (
    AdapterError,
    SourceAdapter,
    optional_native_id,
    parse_tools,
    require_string,
)
from toolcall_tr.hashing import JsonValue
from toolcall_tr.models import (
    AdaptedConversation,
    DecisionAction,
    FunctionCall,
    Message,
    Role,
    ToolCall,
)


class XlamAdapter(SourceAdapter):
    name = "xlam"

    def adapt(self, record: dict[str, JsonValue]) -> AdaptedConversation:
        conversation_id = optional_native_id(record)
        query = require_string(record, "query")
        tools = parse_tools(record.get("tools"))
        raw_calls = record.get("tool_calls")
        if not isinstance(raw_calls, list) or not raw_calls:
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD",
                "xLAM record requires a non-empty tool_calls array",
                "/tool_calls",
            )
        calls: list[ToolCall] = []
        for index, raw_call in enumerate(raw_calls):
            pointer = f"/tool_calls/{index}"
            if not isinstance(raw_call, dict):
                raise AdapterError(
                    "SOURCE_ADAPTER_INVALID_FIELD", "call must be an object", pointer
                )
            call_id = raw_call.get("id")
            name = raw_call.get("name")
            arguments = raw_call.get("arguments")
            if (
                not isinstance(call_id, str)
                or not call_id
                or not isinstance(name, str)
                or not name
                or not isinstance(arguments, dict)
            ):
                raise AdapterError(
                    "SOURCE_ADAPTER_INVALID_FIELD",
                    "call requires id, name, and object arguments",
                    pointer,
                )
            calls.append(
                ToolCall(
                    id=call_id,
                    type="function",
                    function=FunctionCall(name=name, arguments=arguments),
                )
            )
        conversation = [
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
        ]
        return AdaptedConversation(
            source_conversation_id=conversation_id,
            conversation=conversation,
            tools=tools,
            target_message_index=1,
            decision_action=DecisionAction.TOOL_CALL,
            observed_paths=["/id", "/query", "/tools", "/tool_calls"],
        )
