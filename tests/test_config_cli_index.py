from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.helpers import canonical_fixture
from toolcall_tr.cli import app
from toolcall_tr.config import load_config
from toolcall_tr.indexes import rebuild_membership_index


def test_checked_in_config_is_offline() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.toml")
    assert config.providers.enabled is False
    assert config.providers.network_egress_enabled is False


def test_enabling_any_provider_gate_is_rejected(tmp_path: Path) -> None:
    source = (Path(__file__).resolve().parents[1] / "configs" / "pipeline.toml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "unsafe.toml"
    path.write_text(source.replace("enabled = false", "enabled = true", 1), encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires providers"):
        load_config(path)


def test_cli_smoke_and_translation_requires_explicit_live_inputs() -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "source" in help_result.stdout
    assert "canonicalize" in help_result.stdout
    translate_help = runner.invoke(app, ["translate", "--help"])
    assert translate_help.exit_code == 0
    assert "--live" in translate_help.stdout
    assert "--field-policy" in translate_help.stdout
    index_blocked = runner.invoke(app, ["index", "rebuild"])
    assert index_blocked.exit_code == 2
    assert "intentionally unavailable" in index_blocked.stdout


def test_membership_index_is_rebuildable_and_deterministic(fixture_root: Path) -> None:
    episodes = [
        canonical_fixture(fixture_root / "xlam", "xlam", 0),
        canonical_fixture(fixture_root / "xlam", "xlam", 1),
    ]
    left = rebuild_membership_index(iter(episodes))
    right = rebuild_membership_index(iter(reversed(episodes)))
    assert left == right
    assert {
        identity_type for row in left if isinstance((identity_type := row["identity_type"]), str)
    } == {
        "source_occurrence_id",
        "source_native_id",
        "raw_record_sha256",
        "source_episode_fingerprint",
    }
