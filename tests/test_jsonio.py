from __future__ import annotations

from pathlib import Path

import pytest

from toolcall_tr.jsonio import StrictJsonError, iter_jsonl, loads_strict_bytes, write_jsonl


def test_streaming_jsonl_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    count, size = write_jsonl(path, [{"b": 2, "a": 1}, {"value": "İstanbul"}])
    assert count == 2
    assert size == path.stat().st_size
    assert list(iter_jsonl(path)) == [{"a": 1, "b": 2}, {"value": "İstanbul"}]
    with pytest.raises(FileExistsError):
        write_jsonl(path, [])


def test_empty_physical_record_fails_streaming_scan(tmp_path: Path) -> None:
    for payload in (b'{"ok":true}\n\n', b'{"ok":true}\r\n\r\n'):
        path = tmp_path / f"records-{len(payload)}.jsonl"
        path.write_bytes(payload)
        with pytest.raises(StrictJsonError) as error:
            list(iter_jsonl(path))
        assert error.value.code == "PARSE_EMPTY_RECORD"


def test_numeric_overflow_is_rejected_as_non_finite() -> None:
    with pytest.raises(StrictJsonError) as error:
        loads_strict_bytes(b"1e400")
    assert error.value.code == "PARSE_NON_FINITE_NUMBER"
