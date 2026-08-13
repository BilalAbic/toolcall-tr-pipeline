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
            "description": "Look up current weather.",
            "parameters": {
                "type": "dict",
                "required": ["city"],
                "properties": {
                    "city": {"type": "string", "description": "City name."},
                    "days": {"type": "integer", "description": "Number of days."},
                },
            },
        }
    )


def _record(decision: str) -> dict[str, object]:
    answers = {
        "tool_call": json.dumps({"name": "weather.lookup", "arguments": {"city": "Ankara"}}),
        "request_for_info": "Which city should I check?",
        "cannot_answer": "I cannot answer with the supplied tools.",
    }
    return {
        "uuid": "when2call-fixture-001",
        "question": "What is the weather?",
        "correct_answer": decision,
        "answers": answers,
        "tools": [_tool()],
    }


@pytest.mark.parametrize(
    ("label", "action"),
    [
        ("tool_call", DecisionAction.TOOL_CALL),
        ("request_for_info", DecisionAction.CLARIFICATION),
        ("cannot_answer", DecisionAction.TOOL_UNAVAILABLE),
    ],
)
def test_when2call_adapter_maps_source_explicit_decisions(
    label: str, action: DecisionAction
) -> None:
    adapted = get_adapter("when2call").adapt(_record(label))  # type: ignore[arg-type]

    assert adapted.decision_action is action
    assert adapted.source_conversation_id == "when2call-fixture-001"
    assert adapted.tools[0].function.parameters["type"] == "object"
    if action is DecisionAction.TOOL_CALL:
        assert adapted.conversation[-1].tool_calls is not None
        assert adapted.conversation[-1].tool_calls[0].function.name == "weather.lookup"
    else:
        assert adapted.conversation[-1].content is not None


def test_when2call_tool_call_canonicalizes_after_type_alias_conversion() -> None:
    parsed = _record("tool_call")
    bronze = BronzeRecord(
        dataset_namespace="fixture-when2call",
        snapshot_id="snap_" + "1" * 64,
        source_occurrence_id="occ_" + "2" * 64,
        source_sequence=1,
        relative_file_path="source.jsonl",
        byte_offset=0,
        byte_length=1,
        source_line=1,
        raw_record_sha256="sha256:" + "3" * 64,
        raw_record_utf8="fixture",
        parsed_record=parsed,  # type: ignore[arg-type]
        source_native_id="when2call-fixture-001",
        observed_paths=["/uuid"],
        status="valid",
        diagnostics=[],
    )
    episode = canonicalize(bronze, get_adapter("when2call").adapt(parsed), run_event_id="fixture")  # type: ignore[arg-type]

    assert episode.annotations.decision.action is DecisionAction.TOOL_CALL
    assert episode.tools[0].function.parameters["type"] == "object"


def test_when2call_training_apigen_optional_and_collection_types_are_json_shapes() -> None:
    record = _record("request_for_info")
    tool = json.loads(str(record["tools"][0]))  # type: ignore[index]
    tool["parameters"]["properties"] = {
        "city": {"type": "str, optional"},
        "coordinates": {"type": "List[float]"},
        "metadata": {"type": "Dict"},
    }
    record["tools"] = [json.dumps(tool)]

    adapted = get_adapter("when2call").adapt(record)  # type: ignore[arg-type]

    properties = adapted.tools[0].function.parameters["properties"]
    assert isinstance(properties, dict)
    assert properties["city"] == {"type": "string"}
    assert properties["coordinates"] == {"type": "array"}
    assert properties["metadata"] == {"type": "object"}


def test_when2call_rejects_unknown_apigen_schema_type() -> None:
    record = _record("tool_call")
    tool = json.loads(str(record["tools"][0]))  # type: ignore[index]
    tool["parameters"]["properties"]["city"]["type"] = "decimal128"
    record["tools"] = [json.dumps(tool)]

    with pytest.raises(AdapterError, match="unsupported APIGen type"):
        get_adapter("when2call").adapt(record)  # type: ignore[arg-type]
