from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from toolcall_tr.cli import app


def _write_config(root: Path, *, name: str, providers_enabled: bool) -> Path:
    source = (Path(__file__).resolve().parents[1] / "configs" / "pipeline.toml").read_text(
        encoding="utf-8"
    )
    contents = source.replace(
        "enabled = false",
        f"enabled = {str(providers_enabled).lower()}",
        1,
    ).replace(
        "network_egress_enabled = false",
        f"network_egress_enabled = {str(providers_enabled).lower()}",
        1,
    )
    config_path = root / "configs" / name
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(contents, encoding="utf-8")
    return config_path


def test_provider_smoke_defaults_to_offline_dry_run_without_reading_env_values(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _write_config(tmp_path, name="pipeline.toml", providers_enabled=False)
    secret = "never-print-this-provider-secret"
    (tmp_path / ".env").write_text(f"DEEPSEEK_API_KEY={secret}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["provider", "smoke"])

    assert result.exit_code == 0
    assert "dry-run" in result.stdout
    assert "providers.enabled=False" in result.stdout
    assert ".env: present" in result.stdout
    assert "contents and credential values not read" in result.stdout
    assert "live API: disabled" in result.stdout
    assert "zero network requests" in result.stdout
    assert secret not in result.stdout


def test_provider_smoke_live_requires_a_non_default_config(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _write_config(tmp_path, name="pipeline.toml", providers_enabled=False)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["provider", "smoke", "--live"])

    assert result.exit_code == 2
    assert "explicit non-default --config" in result.stdout
    assert "no provider request was made" in " ".join(result.stdout.split())


def test_provider_smoke_live_only_validates_a_custom_config_without_network(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _write_config(tmp_path, name="pipeline.toml", providers_enabled=False)
    _write_config(tmp_path, name="live.toml", providers_enabled=True)
    secret = "never-print-this-other-secret"
    (tmp_path / ".env").write_text(f"OPENAI_API_KEY={secret}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        ["provider", "smoke", "--live", "--config", "configs/live.toml"],
    )

    assert result.exit_code == 0
    assert "live prerequisite inspection only" in result.stdout
    assert "live prerequisites are structurally present" in result.stdout
    assert "live API remains disabled" in result.stdout
    assert "zero network requests" in result.stdout
    assert secret not in result.stdout


def test_provider_smoke_live_reports_missing_structural_prerequisites(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _write_config(tmp_path, name="pipeline.toml", providers_enabled=False)
    _write_config(tmp_path, name="offline-custom.toml", providers_enabled=False)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        ["provider", "smoke", "--live", "--config", "configs/offline-custom.toml"],
    )

    assert result.exit_code == 2
    assert "providers.enabled must be true" in result.stdout
    assert "providers.network_egress_enabled must be true" in result.stdout
    assert ".env file is missing" in result.stdout
    assert "zero network requests" in result.stdout
