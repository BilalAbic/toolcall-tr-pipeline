"""Adapter for the explicit assistant targets in When2Call training files.

The training variants contain no separate ``correct_answer`` field, but do
contain the selected assistant output: SFT stores it in the final assistant
message and preference data stores it in ``chosen_response``.  A full
``<TOOLCALL>`` wrapper is source-explicit.  Text-only targets are admitted
only when they explicitly ask the user for missing information or explicitly
state that the assistant cannot perform the task; all other text remains
quarantined rather than being guessed as a direct answer.
"""

from __future__ import annotations

import re

from toolcall_tr.adapters.base import AdapterError, SourceAdapter
from toolcall_tr.adapters.when2call import parse_apigen_tools, parse_embedded_json
from toolcall_tr.hashing import JsonValue, sha256_jcs
from toolcall_tr.models import (
    AdaptedConversation,
    DecisionAction,
    FunctionCall,
    Message,
    Role,
    ToolCall,
)

_TOOLCALL_WRAPPER = re.compile(r"^\s*<TOOLCALL>(.*?)</TOOLCALL>\s*$", re.DOTALL)
_CLARIFICATION_PATTERNS = (
    re.compile(
        r"\b(?:could|can|would|will|may) you (?:please )?(?:provide|specify|share|"
        r"tell|give|confirm|clarify|supply)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:please|kindly) (?:provide|specify|share|tell|give|confirm)\b", re.IGNORECASE),
    re.compile(r"\b(?:which|what|where|when|who|how many|how much)\b.*\?", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'ll| will)|i) need (?:to know|more|the following)\b", re.IGNORECASE),
    re.compile(r"\b(?:missing|additional|more) (?:information|details?)\b", re.IGNORECASE),
)
_UNAVAILABLE_PATTERN = re.compile(
    r"\b(?:apolog(?:y|ies|ize)|sorry|unable|cannot|can't|do not have|don't have|"
    r"not (?:able|capable)|no (?:access|capability)|cannot perform)\b",
    re.IGNORECASE,
)


def _message(role: Role, content: str) -> Message:
    return Message(
        role=role,
        content=content,
        reasoning_content=None,
        thinking=None,
        tool_calls=None,
        images=None,
        name=None,
        tool_call_id=None,
    )


def _parse_messages(value: JsonValue) -> list[Message]:
    if not isinstance(value, list) or not value:
        raise AdapterError(
            "SOURCE_ADAPTER_INVALID_FIELD",
            "When2Call training messages must be a non-empty array",
            "/messages",
        )
    messages: list[Message] = []
    for index, raw in enumerate(value):
        pointer = f"/messages/{index}"
        if not isinstance(raw, dict):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD",
                "When2Call training message must be an object",
                pointer,
            )
        role_value = raw.get("role")
        content = raw.get("content")
        if not isinstance(role_value, str) or not isinstance(content, str) or not content:
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD",
                "When2Call training message requires non-empty text role and content",
                pointer,
            )
        try:
            role = Role(role_value)
        except ValueError as exc:
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD",
                f"unsupported When2Call training role: {role_value}",
                f"{pointer}/role",
            ) from exc
        if role is Role.TOOL:
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD",
                "When2Call training messages cannot contain an unbound tool result",
                pointer,
            )
        messages.append(_message(role, content))
    return messages


def _selected_preference_response(value: JsonValue) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AdapterError(
            "SOURCE_ADAPTER_INVALID_FIELD",
            "chosen_response must be an assistant message object",
            "/chosen_response",
        )
    content = value.get("content")
    if value.get("role") != "assistant" or not isinstance(content, str):
        raise AdapterError(
            "SOURCE_ADAPTER_INVALID_FIELD",
            "chosen_response requires assistant role and text content",
            "/chosen_response",
        )
    if not content:
        raise AdapterError(
            "SOURCE_ADAPTER_INVALID_FIELD",
            "chosen_response content must be non-empty",
            "/chosen_response/content",
        )
    return content


def _text_decision(content: str, pointer: str) -> DecisionAction:
    # Asking the user for missing information takes precedence when an
    # otherwise apologetic response contains both signals: its actionable
    # selected behavior is the request, not an invented tool-unavailable call.
    if any(pattern.search(content) for pattern in _CLARIFICATION_PATTERNS):
        return DecisionAction.CLARIFICATION
    if _UNAVAILABLE_PATTERN.search(content):
        return DecisionAction.TOOL_UNAVAILABLE
    raise AdapterError(
        "SOURCE_ADAPTER_INVALID_FIELD",
        "text target does not explicitly declare clarification or tool unavailability",
        pointer,
    )


def _tool_call_target(content: str, source_id: str, pointer: str) -> Message | None:
    match = _TOOLCALL_WRAPPER.fullmatch(content)
    if match is None:
        return None
    parsed = parse_embedded_json(match.group(1), pointer)
    if not isinstance(parsed, list) or not parsed:
        raise AdapterError(
            "SOURCE_ADAPTER_INVALID_FIELD",
            "TOOLCALL payload must be a non-empty array",
            pointer,
        )
    calls: list[ToolCall] = []
    for index, raw_call in enumerate(parsed):
        call_pointer = f"{pointer}/{index}"
        if not isinstance(raw_call, dict):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD", "tool call must be an object", call_pointer
            )
        name = raw_call.get("name")
        arguments = raw_call.get("arguments")
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD",
                "tool call requires a non-empty name and object arguments",
                call_pointer,
            )
        calls.append(
            ToolCall(
                id=f"call_{source_id}_{index}",
                type="function",
                function=FunctionCall(name=name, arguments=arguments),
            )
        )
    return Message(
        role=Role.ASSISTANT,
        content=None,
        reasoning_content=None,
        thinking=None,
        tool_calls=calls,
        images=None,
        name=None,
        tool_call_id=None,
    )


class When2CallTrainingAdapter(SourceAdapter):
    """Map selected SFT/preference assistant targets without source repair."""

    name = "when2call_training"

    def adapt(self, record: dict[str, JsonValue]) -> AdaptedConversation:
        conversation = _parse_messages(record.get("messages"))
        tools = parse_apigen_tools(record.get("tools"))
        selected = _selected_preference_response(record.get("chosen_response"))

        source_id = f"when2call-train-{sha256_jcs(record).removeprefix('sha256:')}"
        target_pointer = "/chosen_response/content"
        if selected is None:
            if conversation[-1].role is not Role.ASSISTANT:
                raise AdapterError(
                    "SOURCE_ADAPTER_INVALID_FIELD",
                    "SFT training row must end with a selected assistant message",
                    "/messages",
                )
            target = conversation[-1]
            conversation = conversation[:-1]
            target_pointer = "/messages/-/content"
        else:
            target = _message(Role.ASSISTANT, selected)

        if not conversation or conversation[-1].role is Role.ASSISTANT:
            raise AdapterError(
                "SOURCE_ADAPTER_INVALID_FIELD",
                "training context must end before the selected assistant target",
                "/messages",
            )
        tool_target = _tool_call_target(target.content or "", source_id, target_pointer)
        if tool_target is None:
            action = _text_decision(target.content or "", target_pointer)
        else:
            action = DecisionAction.TOOL_CALL
            target = tool_target
        conversation.append(target)
        observed_paths = ["/messages", "/tools"]
        if selected is not None:
            observed_paths.append("/chosen_response")
        return AdaptedConversation(
            source_conversation_id=source_id,
            conversation=conversation,
            tools=tools,
            target_message_index=len(conversation) - 1,
            decision_action=action,
            observed_paths=observed_paths,
        )
