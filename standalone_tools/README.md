# Standalone conversation JSON exporter

This folder is intentionally outside the pipeline. The exporter reads an
existing HF-review JSONL artifact and creates a new, conversation-only JSON
array without changing the source file or calling any model/API.

```powershell
uv run python standalone_tools/export_conversation_json.py `
  artifacts/automation/when2call-xlam-50-prompt-v3-workers6-20260813/hf-review-package/hfpackage_fe1966778e00f1c0a16fd0b955ed369f302f2c90c671bddfeb2226038637a50e/data/train.jsonl `
  exports/conversation-json/when2call-xlam-50-prompt-v3-workers6-20260813/train.json
```

The output is a UTF-8 `.json` array. Each row has only a `conversation` field;
each message contains exactly `role`, `content`, `reasoning_content`,
`thinking`, `tool_calls`, and `images`. Source-only metadata is omitted while
the original `tool_calls` payload is retained. An assistant tool-call message
whose source `content` is `null` is written with `content: ""`.

The command refuses to overwrite an existing output file.
