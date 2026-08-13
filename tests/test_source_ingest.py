from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from toolcall_tr.source import SourceMutationError, SourceSnapshot, ingest_snapshot, register_source


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_source(root: Path, payload: bytes) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "source.jsonl"
    path.write_bytes(payload)
    return path


def snapshot(root: Path) -> SourceSnapshot:
    return register_source(
        root,
        dataset_namespace="test",
        source_revision="r1",
        license_id="test-only",
    )


def test_occurrence_identity_uses_physical_byte_offset_before_parse(tmp_path: Path) -> None:
    path = write_source(tmp_path, b'{"id":"a"}\n{"id":"b"}\n')
    registered = snapshot(tmp_path)
    before = file_hash(path)
    rows = list(ingest_snapshot(registered, tmp_path))
    assert [row.byte_offset for row in rows] == [0, len(b'{"id":"a"}\n')]
    assert [row.source_sequence for row in rows] == [1, 2]
    assert rows[0].source_occurrence_id != rows[1].source_occurrence_id
    assert file_hash(path) == before


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=15, unique=True))
def test_same_snapshot_always_yields_same_order_and_ids(tmp_path: Path, values: list[int]) -> None:
    payload = b"".join(f'{{"id":{value}}}\n'.encode() for value in values)
    write_source(tmp_path, payload)
    registered = snapshot(tmp_path)
    left = list(ingest_snapshot(registered, tmp_path))
    right = list(ingest_snapshot(registered, tmp_path))
    assert [(r.source_sequence, r.source_occurrence_id) for r in left] == [
        (r.source_sequence, r.source_occurrence_id) for r in right
    ]


def test_mutation_creates_new_snapshot_and_invalidates_old_one(tmp_path: Path) -> None:
    path = write_source(tmp_path, b'{"id":"a"}\n')
    old = snapshot(tmp_path)
    path.write_bytes(path.read_bytes() + b'{"id":"b"}\n')
    new = snapshot(tmp_path)
    assert old.snapshot_id != new.snapshot_id
    with pytest.raises(SourceMutationError):
        list(ingest_snapshot(old, tmp_path))


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b'{"id":"a","id":"b"}\n', "PARSE_DUPLICATE_KEY"),
        (b'{"value":NaN}\n', "PARSE_NON_FINITE_NUMBER"),
        (b"\xff\n", "PARSE_INVALID_UTF8"),
        (b"[]\n", "SCHEMA_EXPECTED_OBJECT"),
        (b"\n", "PARSE_EMPTY_RECORD"),
        (b"{broken}\n", "PARSE_INVALID_JSON"),
    ],
)
def test_strict_parse_failures_are_accounted_in_quarantine(
    tmp_path: Path, payload: bytes, code: str
) -> None:
    write_source(tmp_path, payload)
    row = next(iter(ingest_snapshot(snapshot(tmp_path), tmp_path)))
    assert row.status == "quarantined"
    assert row.source_occurrence_id.startswith("occ_")
    assert row.raw_record_sha256.startswith("sha256:")
    assert [item.code for item in row.diagnostics] == [code]


def test_record_size_limit_quarantines_without_parsing(tmp_path: Path) -> None:
    write_source(tmp_path, b'{"id":"large","payload":"abcdef"}\n')
    row = next(iter(ingest_snapshot(snapshot(tmp_path), tmp_path, max_record_bytes=8)))
    assert row.status == "quarantined"
    assert row.diagnostics[0].code == "PARSE_RECORD_TOO_LARGE"


def test_ingest_removes_only_one_physical_record_terminator(tmp_path: Path) -> None:
    write_source(tmp_path, b'{"id":"a"}\r\r\n')
    row = next(iter(ingest_snapshot(snapshot(tmp_path), tmp_path)))
    assert row.status == "valid"
    assert row.raw_record_utf8 == '{"id":"a"}\r'
    assert row.raw_record_sha256 == "sha256:" + hashlib.sha256(b'{"id":"a"}\r').hexdigest()
