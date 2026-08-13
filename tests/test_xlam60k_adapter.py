from __future__ import annotations

import json

import pytest

from toolcall_tr.adapters import get_adapter
from toolcall_tr.adapters.base import AdapterError
from toolcall_tr.canonicalize import canonicalize
from toolcall_tr.models import DecisionAction
from toolcall_tr.source import BronzeRecord


def _record() -> dict[str, object]:
    return {
        "id": 17,
        "query": "Find the temperature for Ankara.",
        "tools": json.dumps(
            [
                {
                    "name": "weather.lookup",
                    "description": "Look up weather.",
                    "parameters": {
                        "city": {"type": "str", "description": "A city.", "required": True},
                        "days": {"type": "List[int], optional", "description": "Forecast days."},
                        "labels": {"type": "set", "description": "Optional labels.", "default": []},
                    },
                }
            ]
        ),
        "answers": json.dumps(
            [{"name": "weather.lookup", "arguments": {"city": "Ankara", "days": [1, 2]}}]
        ),
    }


def test_xlam60k_adapter_decodes_embedded_records_and_derives_call_ids() -> None:
    adapted = get_adapter("xlam60k").adapt(_record())  # type: ignore[arg-type]

    assert adapted.decision_action is DecisionAction.TOOL_CALL
    assert adapted.source_conversation_id == "17"
    assert adapted.conversation[-1].tool_calls is not None
    assert adapted.conversation[-1].tool_calls[0].id == "call_17_0"
    schema = adapted.tools[0].function.parameters
    assert schema["type"] == "object"
    assert schema["required"] == ["city"]
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert isinstance(properties["days"], dict)
    assert properties["days"]["type"] == "array"
    assert properties["days"]["items"] == {"type": "integer"}
    assert isinstance(properties["labels"], dict)
    assert properties["labels"]["type"] == "array"
    assert properties["labels"]["uniqueItems"] is True
    assert properties["labels"]["default"] == []


def test_xlam60k_adapter_canonicalizes_multiple_source_ordered_calls() -> None:
    record = _record()
    record["answers"] = json.dumps(
        [
            {"name": "weather.lookup", "arguments": {"city": "Ankara"}},
            {"name": "weather.lookup", "arguments": {"city": "Istanbul"}},
        ]
    )
    bronze = BronzeRecord(
        dataset_namespace="fixture-xlam60k",
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
        source_native_id="17",
        observed_paths=["/id"],
        status="valid",
        diagnostics=[],
    )

    episode = canonicalize(
        bronze,
        get_adapter("xlam60k").adapt(record),  # type: ignore[arg-type]
        run_event_id="fixture",
    )

    assert episode.annotations.decision.call_shape == "multi_same_turn"
    assert episode.annotations.decision.call_ids == ["call_17_0", "call_17_1"]


@pytest.mark.parametrize("source_type", ["Callable[[float], float]", "UnknownType"])
def test_xlam60k_adapter_quarantines_unsupported_parameter_types(source_type: str) -> None:
    record = _record()
    tools = json.loads(str(record["tools"]))
    tools[0]["parameters"]["city"]["type"] = source_type
    record["tools"] = json.dumps(tools)

    with pytest.raises(AdapterError, match="unsupported xLAM parameter type"):
        get_adapter("xlam60k").adapt(record)  # type: ignore[arg-type]


def test_xlam60k_adapter_rejects_unpresented_answer_tool() -> None:
    record = _record()
    record["answers"] = json.dumps([{"name": "not-presented", "arguments": {}}])

    with pytest.raises(AdapterError, match="not presented"):
        get_adapter("xlam60k").adapt(record)  # type: ignore[arg-type]
