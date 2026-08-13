from __future__ import annotations

import json

import pytest

from toolcall_tr.adapters import get_adapter
from toolcall_tr.adapters.base import AdapterError
from toolcall_tr.canonicalize import canonicalize
from toolcall_tr.models import DecisionAction
from toolcall_tr.source import BronzeRecord


def _tool() -> str:
    return json.dumps(
        {
            "name": "weather.lookup",
            "description": "Look up weather.",
            "parameters": {"type": "dict", "properties": {"city": {"type": "str"}}},
        }
    )


def _bronze(record: dict[str, object]) -> BronzeRecord:
    return BronzeRecord(
        dataset_namespace="fixture-when2call-training",
        snapshot_id="snap_" + "1" * 64,
        source_occurrence_id="occ_" + "2" * 64,
        source_sequence=1,
        relative_file_path="source.jsonl",
        byte_offset=0,
        byte_length=1,
        source_line=1,
        raw_record_sha256="sha256:" + "3" * 64,
        raw_record_utf8="fixture",
        parsed_record=record,  # type: ignore[arg-type]
        source_native_id=None,
        observed_paths=["/messages"],
        status="valid",
        diagnostics=[],
    )


@pytest.mark.parametrize(
    ("content", "action"),
    [
        ("Could you please provide the city?", DecisionAction.CLARIFICATION),
        ("Apologies, but I am unable to perform that task.", DecisionAction.TOOL_UNAVAILABLE),
    ],
)
def test_when2call_sft_text_target_preserves_explicit_no_tool_behavior(
    content: str, action: DecisionAction
) -> None:
    record: dict[str, object] = {
        "messages": [
            {"role": "user", "content": "Do something."},
            {"role": "assistant", "content": content},
        ],
        "tools": [_tool()],
    }

    adapted = get_adapter("when2call_training").adapt(record)  # type: ignore[arg-type]

    assert adapted.decision_action is action
    assert adapted.conversation[-1].content == content
    assert adapted.tools[0].function.parameters["type"] == "object"
    episode = canonicalize(_bronze(record), adapted, run_event_id="fixture")
    assert episode.annotations.decision.action is action


def test_when2call_preference_target_parses_source_explicit_tool_call() -> None:
    record: dict[str, object] = {
        "messages": [{"role": "user", "content": "Check Ankara weather."}],
        "tools": [_tool()],
        "chosen_response": {
            "role": "assistant",
            "content": (
                '<TOOLCALL>[{"name":"weather.lookup",'
                '"arguments":{"city":"Ankara"}}]</TOOLCALL>'
            ),
        },
        "rejected_response": {"role": "assistant", "content": "I cannot help."},
    }

    adapted = get_adapter("when2call_training").adapt(record)  # type: ignore[arg-type]

    assert adapted.decision_action is DecisionAction.TOOL_CALL
    assert adapted.conversation[-1].tool_calls is not None
    assert adapted.conversation[-1].tool_calls[0].function.name == "weather.lookup"
    episode = canonicalize(_bronze(record), adapted, run_event_id="fixture")
    assert episode.annotations.decision.action is DecisionAction.TOOL_CALL


def test_when2call_training_adapter_quarantines_ambiguous_text_without_guessing() -> None:
    record: dict[str, object] = {
        "messages": [{"role": "user", "content": "An instruction."}],
        "tools": [],
        "chosen_response": {"role": "assistant", "content": "A response."},
    }

    with pytest.raises(AdapterError, match="does not explicitly declare"):
        get_adapter("when2call_training").adapt(record)  # type: ignore[arg-type]


def test_when2call_training_adapter_rejects_invalid_message_envelope() -> None:
    record: dict[str, object] = {"messages": [{"role": "user"}], "tools": []}

    with pytest.raises(AdapterError, match="requires non-empty text role and content"):
        get_adapter("when2call_training").adapt(record)  # type: ignore[arg-type]
