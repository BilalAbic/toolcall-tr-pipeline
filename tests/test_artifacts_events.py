from __future__ import annotations

from pathlib import Path

import pytest

from toolcall_tr.artifacts import PublishError, publish_bytes_atomic, publish_jsonl_artifact
from toolcall_tr.events import EventChainError, EventLog
from toolcall_tr.shards import publish_jsonl_shards


def test_manifest_is_byte_identical_and_publish_is_idempotent(tmp_path: Path) -> None:
    left = publish_jsonl_artifact(
        tmp_path,
        logical_name="records",
        schema_version="fixture-0.1.0",
        stage="test",
        records=[{"b": 2, "a": 1}],
        contract_hashes={"schema": "sha256:" + "0" * 64},
    )
    before = (tmp_path / f"{left.manifest_id}.json").read_bytes()
    right = publish_jsonl_artifact(
        tmp_path,
        logical_name="records",
        schema_version="fixture-0.1.0",
        stage="test",
        records=[{"a": 1, "b": 2}],
        contract_hashes={"schema": "sha256:" + "0" * 64},
    )
    after = (tmp_path / f"{right.manifest_id}.json").read_bytes()
    assert left == right
    assert before == after


def test_atomic_publish_never_overwrites_different_bytes(tmp_path: Path) -> None:
    target = tmp_path / "fixed.jsonl"
    assert publish_bytes_atomic(target, b"first\n") is True
    assert publish_bytes_atomic(target, b"first\n") is False
    with pytest.raises(PublishError):
        publish_bytes_atomic(target, b"different\n")
    assert target.read_bytes() == b"first\n"


def test_validation_failure_publishes_no_shard(tmp_path: Path) -> None:
    def fail(_: Path) -> None:
        raise ValueError("invalid shard")

    with pytest.raises(ValueError, match="invalid shard"):
        publish_jsonl_shards(
            tmp_path,
            logical_name="records",
            schema_version="fixture-0.1.0",
            stage="test",
            records=[{"id": 1}, {"id": 2}],
            shard_rows=1,
            validator=fail,
        )
    assert list(tmp_path.iterdir()) == []


def test_fixed_row_shards_and_manifest_account_for_all_rows(tmp_path: Path) -> None:
    manifest = publish_jsonl_shards(
        tmp_path,
        logical_name="records",
        schema_version="fixture-0.1.0",
        stage="test",
        records=[{"id": index} for index in range(5)],
        shard_rows=2,
    )
    assert [item.row_count for item in manifest.artifacts] == [2, 2, 1]
    assert manifest.row_accounting.accepted == 5


def test_event_log_hash_chain_and_tamper_detection(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    first = log.append(
        run_id="run-1",
        stage="ingest",
        event_type="started",
        details={},
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    second = log.append(
        run_id="run-1",
        stage="ingest",
        event_type="completed",
        details={"rows": 2},
        timestamp_utc="2026-01-01T00:00:01Z",
    )
    assert second.previous_event_hash == first.event_hash
    assert log.read_verified() == [first, second]
    second_path = sorted(tmp_path.glob("*.jsonl"))[1]
    second_path.write_bytes(second_path.read_bytes().replace(b'"rows":2', b'"rows":3'))
    with pytest.raises((EventChainError, ValueError)):
        log.read_verified()
