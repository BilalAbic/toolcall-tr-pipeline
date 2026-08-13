from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from toolcall_tr.artifacts import PublishError
from toolcall_tr.cli import app
from toolcall_tr.pilot import PilotConfigurationError, run_operational_pilot
from toolcall_tr.source import register_source


def _copy_fixture(tmp_path: Path, name: str) -> Path:
    fixture = Path(__file__).parent / "fixtures" / name / "source.jsonl"
    source_root = tmp_path / "source"
    source_root.mkdir()
    target = source_root / "pilot-input.jsonl"
    target.write_bytes(fixture.read_bytes())
    return target


def _run_xlam(input_jsonl: Path, output_root: Path):  # type: ignore[no-untyped-def]
    return run_operational_pilot(
        input_jsonl,
        output_root,
        dataset_namespace="pilot-fixture",
        source_revision="fixture-v1",
        license_id="test-only",
        adapter_name="xlam",
        run_event_id="run_pilot_fixture",
    )


def _run_no_tool(input_jsonl: Path, output_root: Path):  # type: ignore[no-untyped-def]
    return run_operational_pilot(
        input_jsonl,
        output_root,
        dataset_namespace="pilot-fixture",
        source_revision="fixture-v1",
        license_id="test-only",
        adapter_name="no_tool",
        run_event_id="run_pilot_fixture",
    )


def test_pilot_is_deterministic_and_does_not_modify_explicit_input(tmp_path: Path) -> None:
    input_jsonl = _copy_fixture(tmp_path, "xlam")
    output_root = tmp_path / "pilot-output"
    before = hashlib.sha256(input_jsonl.read_bytes()).hexdigest()

    first = _run_xlam(input_jsonl, output_root)
    second = _run_xlam(input_jsonl, output_root)

    assert first == second
    assert first.status == "passed"
    assert first.source_records == 2
    assert first.canonical_records == 2
    assert first.block_reasons == []
    assert hashlib.sha256(input_jsonl.read_bytes()).hexdigest() == before
    assert (output_root / "snapshots" / f"{first.source_snapshot_id}.json").is_file()
    assert (output_root / "bronze" / f"{first.bronze_manifest_id}.json").is_file()
    assert first.canonical_manifest_id is not None
    assert (output_root / "canonical" / f"{first.canonical_manifest_id}.json").is_file()
    assert first.audit_id is not None
    assert (output_root / "audit" / f"{first.audit_id}.json").is_file()
    assert (output_root / "pilot" / f"{first.pilot_id}.json").is_file()


def test_pilot_records_ingest_and_canonical_quarantines_without_partial_canonical_output(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    input_jsonl = source_root / "input.jsonl"
    input_jsonl.write_bytes(b'{"id":"fine"}\n{broken}\n')
    before = input_jsonl.read_bytes()
    output_root = tmp_path / "pilot-output"

    report = _run_xlam(input_jsonl, output_root)

    assert report.status == "blocked"
    assert report.block_reasons == [
        "adapter.source_adapter_missing_field",
        "canonical.quarantined_records",
        "ingest.quarantined_records",
    ]
    assert report.valid_records == 1
    assert report.quarantined_records == 1
    assert report.canonical_quarantined_records == 1
    assert report.canonical_manifest_id is None
    assert report.audit_id is None
    assert input_jsonl.read_bytes() == before
    assert report.quarantine_manifest_id is not None
    assert report.canonical_quarantine_manifest_id is not None
    assert not (output_root / "canonical").exists()
    assert not (output_root / "audit").exists()


def test_pilot_blocks_adapter_error_without_partial_canonical_output(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    input_jsonl = source_root / "input.jsonl"
    input_jsonl.write_text('{"id":"missing-xlam-fields"}\n', encoding="utf-8")
    output_root = tmp_path / "pilot-output"

    report = _run_xlam(input_jsonl, output_root)

    assert report.status == "blocked"
    assert report.block_reasons == [
        "adapter.source_adapter_missing_field",
        "canonical.quarantined_records",
    ]
    assert report.canonical_records == 0
    assert report.canonical_quarantined_records == 1
    assert report.canonical_manifest_id is None
    assert report.canonical_quarantine_manifest_id is not None
    assert not (output_root / "canonical").exists()


def test_pilot_preserves_canonical_survivors_while_quarantining_invalid_rows(
    tmp_path: Path,
) -> None:
    input_jsonl = _copy_fixture(tmp_path, "xlam")
    rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines()]
    rows[1]["tool_calls"][0]["arguments"] = {}
    input_jsonl.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_root = tmp_path / "pilot-output"

    report = _run_xlam(input_jsonl, output_root)

    assert report.status == "blocked"
    assert report.block_reasons == [
        "canonical.quarantined_records",
        "canonical.tool_argument_schema_invalid",
    ]
    assert report.canonical_records == 1
    assert report.canonical_quarantined_records == 1
    assert report.canonical_manifest_id is not None
    assert report.canonical_quarantine_manifest_id is not None
    assert report.audit_id is not None


def test_pilot_blocks_unresolved_exact_conflicts_but_preserves_audit_evidence(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    input_jsonl = source_root / "input.jsonl"
    input_jsonl.write_text(
        "\n".join(
            [
                '{"id":"one","query":"same question","response":"one","action":"direct_answer"}',
                '{"id":"two","query":"same question","response":"two","action":"direct_answer"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "pilot-output"

    report = _run_no_tool(input_jsonl, output_root)

    assert report.status == "blocked"
    assert report.block_reasons == ["audit.review_required_conflicts"]
    assert report.canonical_records == 2
    assert report.review_required_conflicts == 1
    assert report.canonical_manifest_id is not None
    assert report.audit_id is not None
    assert (output_root / "audit" / f"{report.audit_id}.json").is_file()


def test_pilot_blocks_empty_jsonl_without_canonical_output(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    input_jsonl = source_root / "input.jsonl"
    input_jsonl.write_bytes(b"")
    output_root = tmp_path / "pilot-output"

    report = _run_xlam(input_jsonl, output_root)

    assert report.status == "blocked"
    assert report.block_reasons == ["ingest.empty_source"]
    assert report.canonical_manifest_id is None
    assert report.audit_id is None


def test_pilot_refuses_collision_without_overwriting_existing_output(tmp_path: Path) -> None:
    input_jsonl = _copy_fixture(tmp_path, "xlam")
    output_root = tmp_path / "pilot-output"
    snapshot = register_source(
        input_jsonl.parent,
        dataset_namespace="pilot-fixture",
        source_revision="fixture-v1",
        license_id="test-only",
        relative_files=[input_jsonl.name],
    )
    collision = output_root / "snapshots" / f"{snapshot.snapshot_id}.json"
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"different immutable content\n")

    with pytest.raises(PublishError):
        _run_xlam(input_jsonl, output_root)

    assert collision.read_bytes() == b"different immutable content\n"


def test_pilot_rejects_output_root_that_intersects_source_root(tmp_path: Path) -> None:
    input_jsonl = _copy_fixture(tmp_path, "xlam")

    with pytest.raises(PilotConfigurationError, match="disjoint"):
        _run_xlam(input_jsonl, input_jsonl.parent / "derived")

    assert not (input_jsonl.parent / "derived").exists()


def test_pilot_rejects_unknown_adapter_before_writing_output(tmp_path: Path) -> None:
    input_jsonl = _copy_fixture(tmp_path, "xlam")
    output_root = tmp_path / "pilot-output"

    with pytest.raises(PilotConfigurationError, match="unknown adapter"):
        run_operational_pilot(
            input_jsonl,
            output_root,
            dataset_namespace="pilot-fixture",
            source_revision="fixture-v1",
            license_id="test-only",
            adapter_name="not-installed",
            run_event_id="run_pilot_fixture",
        )

    assert not output_root.exists()


def test_pilot_cli_requires_explicit_output_and_returns_blocked_exit(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    input_jsonl = source_root / "input.jsonl"
    input_jsonl.write_text("{broken}\n", encoding="utf-8")
    runner = CliRunner()

    missing_output = runner.invoke(
        app,
        [
            "pilot",
            "run",
            str(input_jsonl),
            "--dataset",
            "pilot",
            "--revision",
            "r1",
            "--license",
            "test-only",
            "--adapter",
            "xlam",
            "--run-event-id",
            "run_pilot_cli",
        ],
    )
    blocked = runner.invoke(
        app,
        [
            "pilot",
            "run",
            str(input_jsonl),
            "--output",
            str(tmp_path / "output"),
            "--dataset",
            "pilot",
            "--revision",
            "r1",
            "--license",
            "test-only",
            "--adapter",
            "xlam",
            "--run-event-id",
            "run_pilot_cli",
        ],
    )

    assert missing_output.exit_code == 2
    assert blocked.exit_code == 2
    assert "pilot blocked" in blocked.stdout
