"""Narrow, secret-safe credential resolution for explicitly approved live calls.

The resolver is intentionally separate from config parsing and from every
offline workflow.  It reads only an allow-listed provider key, never mutates
the process environment, and its exceptions never include a secret value.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path


class CredentialResolutionError(RuntimeError):
    """Raised without revealing credential contents or dotenv details."""


_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


class AllowListedSecretResolver:
    """Resolve one explicitly allowed secret from env or a local dotenv file."""

    def __init__(
        self,
        *,
        allowed_names: frozenset[str],
        env_file: Path = Path(".env"),
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not allowed_names or any(_ENV_NAME.fullmatch(name) is None for name in allowed_names):
            raise ValueError("allowed secret names must be non-empty uppercase environment names")
        self._allowed_names = allowed_names
        self._env_file = env_file
        self._environment = environment if environment is not None else os.environ

    def resolve(self, name: str) -> str:
        """Return a non-empty allowed secret without emitting it anywhere."""
        if name not in self._allowed_names:
            raise CredentialResolutionError("credential name is not approved for this operation")
        value = self._environment.get(name)
        if value is None:
            value = self._dotenv_values().get(name)
        if not isinstance(value, str) or not value or "\r" in value or "\n" in value:
            raise CredentialResolutionError("approved API credential is unavailable")
        return value

    def _dotenv_values(self) -> dict[str, str]:
        """Parse only allow-listed KEY=VALUE entries; unknown keys are ignored."""
        if not self._env_file.is_file():
            return {}
        try:
            lines = self._env_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise CredentialResolutionError("approved API credential is unavailable") from exc

        values: dict[str, str] = {}
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line.removeprefix("export ").lstrip()
            key, separator, raw_value = line.partition("=")
            if not separator:
                continue
            key = key.strip()
            if key not in self._allowed_names:
                continue
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            values[key] = value
        return values
