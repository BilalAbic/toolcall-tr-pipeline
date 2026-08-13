"""Adapter for source-explicit clarification/unavailable/direct-answer fixtures."""

from __future__ import annotations

from toolcall_tr.adapters.base import (
    AdapterError,
    SourceAdapter,
    optional_native_id,
    parse_tools,
    require_string,
)
from toolcall_tr.hashing import JsonValue
from toolcall_tr.models import AdaptedConversation, DecisionAction, Message, Role

ALLOWED_ACTIONS = {
    "clarification": DecisionAction.CLARIFICATION,
    "tool_unavailable": DecisionAction.TOOL_UNAVAILABLE,
    "direct_answer": DecisionAction.DIRECT_ANSWER,
}


class NoToolAdapter(SourceAdapter):
    name = "no_tool"

    def adapt(self, record: dict[str, JsonValue]) -> AdaptedConversation:
        conversation_id = optional_native_id(record)
        query = require_string(record, "query")
        response = require_string(record, "response")
        raw_action = record.get("action")
        if not isinstance(raw_action, str) or raw_action not in ALLOWED_ACTIONS:
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD",
                f"action must be one of {sorted(ALLOWED_ACTIONS)}",
                "/action",
            )
        raw_tools = record.get("tools", [])
        tools = parse_tools(raw_tools)
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
                content=response,
                reasoning_content=None,
                thinking=None,
                tool_calls=None,
                images=None,
                name=None,
                tool_call_id=None,
            ),
        ]
        observed = ["/id", "/query", "/response", "/action"]
        if "tools" in record:
            observed.append("/tools")
        return AdaptedConversation(
            source_conversation_id=conversation_id,
            conversation=conversation,
            tools=tools,
            target_message_index=1,
            decision_action=ALLOWED_ACTIONS[raw_action],
            observed_paths=observed,
        )
