"""Export pipeline review JSONL as plain conversation JSON.

This is a standalone utility.  It does not import, configure, or mutate the
pipeline; it only reads an existing HF-review JSONL file and creates a new JSON
array in the requested conversation-only format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

MESSAGE_FIELDS = (
    "role",
    "content",
    "reasoning_content",
    "thinking",
    "tool_calls",
    "images",
)
ALLOWED_ROLES: set[str] = {"system", "user", "assistant", "tool"}


class ConversationExportError(ValueError):
    """Raised when the input cannot be represented in the target format."""


def _normalise_message(
    raw: object, *, line_number: int, message_index: int
) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ConversationExportError(
            f"line {line_number}, message {message_index}: message must be an object"
        )
    message = cast(dict[str, object], raw)
    role = message.get("role")
    content = message.get("content")
    if role not in ALLOWED_ROLES:
        raise ConversationExportError(
            f"line {line_number}, message {message_index}: unsupported role {role!r}"
        )
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ConversationExportError(
            f"line {line_number}, message {message_index}: content must be a string or null"
        )
    normalised = {field: message.get(field) for field in MESSAGE_FIELDS}
    normalised["content"] = content
    return normalised


def export_conversations(input_path: Path, output_path: Path) -> int:
    """Create one strict JSON array without modifying the source JSONL file.

    Each source row must contain ``messages``.  Source-only metadata such as
    dataset IDs, quality tiers, message names, and tool-call IDs is deliberately
    omitted. Assistant tool-call messages with null content are normalized to
    an empty string; the ``tool_calls`` payload itself is preserved unchanged.
    """
    source = input_path.resolve(strict=True)
    destination = output_path.resolve(strict=False)
    if not source.is_file() or source.suffix.lower() != ".jsonl":
        raise ConversationExportError("input must be an existing .jsonl file")
    if destination.suffix.lower() != ".json":
        raise ConversationExportError("output must use the .json suffix")
    if source == destination:
        raise ConversationExportError("input and output must be different files")
    if destination.exists():
        raise ConversationExportError(f"refusing to overwrite existing output: {destination}")

    conversations: list[dict[str, list[dict[str, object]]]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise ConversationExportError(f"line {line_number}: empty JSONL line")
        try:
            row: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConversationExportError(f"line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ConversationExportError(f"line {line_number}: row must be an object")
        source_row = cast(dict[str, object], row)
        messages = source_row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ConversationExportError(f"line {line_number}: messages must be a non-empty array")
        source_messages = cast(list[object], messages)
        conversations.append(
            {
                "conversation": [
                    _normalise_message(message, line_number=line_number, message_index=index)
                    for index, message in enumerate(source_messages)
                ]
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(conversations, ensure_ascii=False, indent=2) + "\n"
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return len(conversations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export review JSONL to conversation-only JSON.")
    parser.add_argument("input", type=Path, help="source HF-review .jsonl file")
    parser.add_argument("output", type=Path, help="new conversation-only .json file")
    args = parser.parse_args()
    try:
        count = export_conversations(args.input, args.output)
    except (ConversationExportError, OSError) as exc:
        parser.error(str(exc))
    print(f"exported {count} conversations to {args.output}")


if __name__ == "__main__":
    main()
