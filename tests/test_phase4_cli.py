from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tests.helpers import canonical_fixture
from toolcall_tr.cli import app
from toolcall_tr.hashing import canonical_bytes
from toolcall_tr.jsonio import iter_jsonl, write_jsonl
from toolcall_tr.selection import SelectionCandidate, SelectionStratum
from toolcall_tr.similarity import NearDuplicateCandidate, SimilarityDocument
from toolcall_tr.source_evidence import ArgumentEvidenceInput, SourceEvidenceRequest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


def write_records(path: Path, records: list[object]) -> None:
    write_jsonl(path, records)


def selection_candidate(index: int) -> SelectionCandidate:
    return SelectionCandidate(
        episode_id=f"ep_{index:064x}",
        stratum=SelectionStratum(
            dataset_namespace=f"fixture-{index % 2}",
            action="tool_call",
            call_shape="single",
            tool_family="weather",
            domain="fixture",
            length_bucket="short",
            tool_count=1,
        ),
        source_verdict="source_valid",
        human_adjudicated=True,
        argument_grounding=["explicit_user"],
        unresolved_hard_conflict=False,
    )


def test_source_evidence_cli_requires_complete_explicit_request_set(
    fixture_root: Path, tmp_path: Path
) -> None:
    episode = canonical_fixture(fixture_root / "xlam", "xlam", 0)
    canonical_path = tmp_path / "canonical.jsonl"
    requests_path = tmp_path / "requests.jsonl"
    output = tmp_path / "output"
    write_records(canonical_path, [episode.model_dump(mode="json", exclude_none=False)])
    request = SourceEvidenceRequest(
        episode_id=episode.episode_id,
        argument_evidence=[
            ArgumentEvidenceInput(
                call_id="call_001",
                argument_pointer="/city",
                origin="explicit_user",
                evidence_pointers=["/conversation/0/content"],
                input_pointers=[],
            )
        ],
    )
    write_records(requests_path, [request.model_dump(mode="json", exclude_none=False)])

    result = RUNNER.invoke(
        app,
        ["source", "evidence", str(canonical_path), str(requests_path), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    artifact = next(output.glob("source-evidence-*.jsonl"))
    records = list(iter_jsonl(artifact))
    assert len(records) == 1
    assert isinstance(records[0], dict)
    assert records[0]["pass1_result"] == "deterministic_pass"


def test_audit_commands_are_offline_and_non_destructive(fixture_root: Path, tmp_path: Path) -> None:
    episode = canonical_fixture(fixture_root / "no_tool", "no_tool", 2)
    first = episode.model_dump(mode="json", exclude_none=False)
    second = episode.model_dump(mode="json", exclude_none=False)
    first["episode_id"] = f"ep_{'a' * 64}"
    second["episode_id"] = f"ep_{'b' * 64}"
    canonical_path = tmp_path / "canonical.jsonl"
    write_records(canonical_path, [first, second])
    audit_output = tmp_path / "audit"
    result = RUNNER.invoke(
        app, ["audit", "duplicates", str(canonical_path), "--output", str(audit_output)]
    )
    assert result.exit_code == 0, result.output
    audit = json.loads(next(audit_output.glob("audit_*.json")).read_text(encoding="utf-8"))
    assert audit["automatic_drop_episode_ids"] == []
    assert audit["duplicate_groups"][0]["automatic_drop"] is False

    documents_path = tmp_path / "documents.jsonl"
    documents = [
        SimilarityDocument(episode_id=f"ep_{'1' * 64}", text="Book flight Ankara Istanbul"),
        SimilarityDocument(episode_id=f"ep_{'2' * 64}", text="BOOK flight Ankara Istanbul!"),
    ]
    write_records(documents_path, [document.model_dump(mode="json") for document in documents])
    near_output = tmp_path / "near"
    result = RUNNER.invoke(
        app,
        [
            "audit",
            "near-duplicates",
            str(documents_path),
            "--output",
            str(near_output),
            "--config",
            str(ROOT / "configs" / "phase4.toml"),
        ],
    )
    assert result.exit_code == 0, result.output
    candidate_file = next(near_output.glob("near-duplicate-candidates-*.jsonl"))
    candidate = NearDuplicateCandidate.model_validate_json(
        candidate_file.read_text(encoding="utf-8"), strict=True
    )
    assert candidate.automatic_drop is False
    assert candidate.disposition == "human_review"


def test_selection_freeze_cli_only_writes_complete_human_adjudicated_s400(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    output = tmp_path / "selection"
    candidates = [selection_candidate(index) for index in range(400)]
    write_records(candidates_path, [candidate.model_dump(mode="json") for candidate in candidates])
    result = RUNNER.invoke(
        app,
        [
            "select",
            "freeze",
            str(candidates_path),
            "--output",
            str(output),
            "--config",
            str(ROOT / "configs" / "phase4.toml"),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads(next(output.glob("selection_*.json")).read_text(encoding="utf-8"))
    assert [len(tier["episode_ids"]) for tier in manifest["tiers"]] == [30, 100, 250, 400]
    assert "timestamp" not in canonical_bytes(manifest).decode("utf-8")
    assert "run_id" not in canonical_bytes(manifest).decode("utf-8")
