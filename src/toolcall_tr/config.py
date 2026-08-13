"""Fail-closed local configuration; this phase has no provider implementation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from toolcall_tr.models import NonEmptyStr, StrictModel


class ProviderRole(StrictModel):
    provider: NonEmptyStr
    model: NonEmptyStr
    api_key_env: NonEmptyStr
    endpoint: str | None = None
    temperature: float | None = None
    thinking: bool | None = None
    max_workers: int = Field(default=4, ge=1, le=16)
    daily_token_budget: int | None = Field(default=None, ge=1)


class ProviderConfig(StrictModel):
    enabled: bool
    network_egress_enabled: bool
    translator: ProviderRole
    strong_judge: ProviderRole
    mini_verifier: ProviderRole


class PipelineConfig(StrictModel):
    schema_version: Literal["pipeline-config-0.1.0"]
    canonical_schema_version: Literal["0.1.0"]
    diagnostic_catalog_version: Literal["0.1.0"]
    normalizer_version: Literal["0.1.0"]
    max_record_bytes: int
    jsonl_shard_rows: int
    providers: ProviderConfig

    @model_validator(mode="after")
    def positive_limits(self) -> PipelineConfig:
        if self.max_record_bytes < 1 or self.jsonl_shard_rows < 1:
            raise ValueError("pipeline size limits must be positive")
        return self

    def require_offline_phase(self) -> None:
        if self.providers.enabled or self.providers.network_egress_enabled:
            raise RuntimeError(
                "offline core requires providers and network egress to stay disabled"
            )


def inspect_config(path: Path) -> PipelineConfig:
    """Strictly parse a config without granting permission to execute it."""
    with path.open("rb") as handle:
        return PipelineConfig.model_validate(tomllib.load(handle), strict=True)


def load_config(path: Path) -> PipelineConfig:
    config = inspect_config(path)
    config.require_offline_phase()
    return config


def load_live_config(path: Path) -> PipelineConfig:
    """Load an explicitly named live config; callers still own execution policy."""
    config = inspect_config(path)
    if not config.providers.enabled or not config.providers.network_egress_enabled:
        raise RuntimeError("live execution requires both provider and network egress gates")
    return config
