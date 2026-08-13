from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from toolcall_tr.cli import app
from toolcall_tr.source_array import SourceArrayConversionError, convert_json_array_to_jsonl


def _source_json(tmp_path: Path, payload: str) -> Path:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "records.json"
    source.write_text(payload, encoding="utf-8")
    return source


def test_json_array_conversion_is_immutable_and_idempotent(tmp_path: Path) -> None:
    source = _source_json(tmp_path, '[{"z":1,"id":"two"},{"id":"one","a":[true,null]}]')
    output = tmp_path / "derived"
    before = source.read_bytes()

    first = convert_json_array_to_jsonl(source, output)
    second = convert_json_array_to_jsonl(source, output)

    assert first == second
    assert first.input_record_count == 2
    assert first.output_record_count == 2
    target = output / first.output_relative_path
    assert target.read_text(encoding="utf-8") == (
        '{"id":"two","z":1}\n{"a":[true,null],"id":"one"}\n'
    )
    assert source.read_bytes() == before
    assert (output / "reports" / f"{first.conversion_id}.json").is_file()


@pytest.mark.parametrize(
    "payload",
    [
        '{"not":"an array"}',
        '[{"id":"valid"},42]',
        '[{"id":"one","id":"two"}]',
    ],
)
def test_json_array_conversion_rejects_invalid_source_before_writing(
    tmp_path: Path, payload: str
) -> None:
    source = _source_json(tmp_path, payload)
    output = tmp_path / "derived"

    with pytest.raises(ValueError):
        convert_json_array_to_jsonl(source, output)

    assert not output.exists()


def test_json_array_conversion_rejects_output_inside_source_root(tmp_path: Path) -> None:
    source = _source_json(tmp_path, '[{"id":"one"}]')

    with pytest.raises(SourceArrayConversionError, match="disjoint"):
        convert_json_array_to_jsonl(source, source.parent / "derived")


def test_json_array_conversion_cli_requires_output_and_does_not_print_data(tmp_path: Path) -> None:
    source = _source_json(tmp_path, json.dumps([{"id": "one", "query": "private payload"}]))
    runner = CliRunner()

    missing = runner.invoke(app, ["source", "json-array-to-jsonl", str(source)])
    completed = runner.invoke(
        app,
        ["source", "json-array-to-jsonl", str(source), "--output", str(tmp_path / "derived")],
    )

    assert missing.exit_code == 2
    assert completed.exit_code == 0
    assert "private payload" not in completed.stdout
