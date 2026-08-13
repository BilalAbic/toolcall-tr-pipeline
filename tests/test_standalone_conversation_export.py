from __future__ import annotations

import json
from pathlib import Path

import pytest

from standalone_tools.export_conversation_json import (
    ConversationExportError,
    export_conversations,
)


def test_export_conversations_keeps_only_target_message_fields(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "source-only",
                "messages": [
                    {
                        "role": "system",
                        "content": "",
                        "reasoning_content": None,
                        "thinking": None,
                        "tool_calls": None,
                        "images": None,
                        "name": None,
                        "tool_call_id": None,
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": None,
                        "thinking": None,
                        "tool_calls": [{"id": "call_1", "type": "function"}],
                        "images": None,
                        "name": "assistant-name",
                        "tool_call_id": "source-only-id",
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "output.json"

    assert export_conversations(source, destination) == 1
    exported = json.loads(destination.read_text(encoding="utf-8"))
    assert exported == [
        {
            "conversation": [
                {
                    "role": "system",
                    "content": "",
                    "reasoning_content": None,
                    "thinking": None,
                    "tool_calls": None,
                    "images": None,
                },
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": None,
                    "thinking": None,
                    "tool_calls": [{"id": "call_1", "type": "function"}],
                    "images": None,
                },
            ]
        }
    ]

    with pytest.raises(ConversationExportError, match="refusing to overwrite"):
        export_conversations(source, destination)
