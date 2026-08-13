from __future__ import annotations

from pathlib import Path

import pytest

from toolcall_tr.credentials import AllowListedSecretResolver, CredentialResolutionError


def test_resolver_reads_only_an_allowlisted_dotenv_key_without_mutating_environment(
    tmp_path: Path,
) -> None:
    secret = "never-display-this-secret"
    path = tmp_path / ".env"
    path.write_text(f"OTHER_KEY=ignored\nDEEPSEEK_API_KEY='{secret}'\n", encoding="utf-8")
    resolver = AllowListedSecretResolver(
        allowed_names=frozenset({"DEEPSEEK_API_KEY"}), env_file=path, environment={}
    )

    assert resolver.resolve("DEEPSEEK_API_KEY") == secret
    with pytest.raises(CredentialResolutionError, match="not approved") as raised:
        resolver.resolve("OTHER_KEY")
    assert secret not in str(raised.value)


def test_resolver_prefers_explicit_process_value_and_redacts_unavailable_errors(
    tmp_path: Path,
) -> None:
    secret = "never-display-this-second-secret"
    resolver = AllowListedSecretResolver(
        allowed_names=frozenset({"OPENAI_API_KEY"}),
        env_file=tmp_path / "missing.env",
        environment={"OPENAI_API_KEY": secret},
    )
    assert resolver.resolve("OPENAI_API_KEY") == secret

    missing = AllowListedSecretResolver(
        allowed_names=frozenset({"OPENAI_API_KEY"}),
        env_file=tmp_path / "missing.env",
        environment={},
    )
    with pytest.raises(CredentialResolutionError, match="unavailable") as raised:
        missing.resolve("OPENAI_API_KEY")
    assert secret not in str(raised.value)
