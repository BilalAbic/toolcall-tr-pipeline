"""Fail-closed Phase 4 duplicate, conflict, and selection policy."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from toolcall_tr.models import StrictModel


class Phase4Config(StrictModel):
    schema_version: Literal["phase4-config-0.1.0"]
    ngram_size: int = Field(ge=1, le=8)
    near_duplicate_candidate_threshold: float = Field(ge=0.0, le=1.0)
    automatic_similarity_drop: bool
    required_source_valid_membership: int = Field(ge=400)
    selection_tiers: list[int]
    semantic_judge_enabled: bool
    human_adjudication_required: bool

    @model_validator(mode="after")
    def enforce_offline_human_gate(self) -> Phase4Config:
        if self.automatic_similarity_drop:
            raise ValueError("near similarity may only create review candidates")
        if self.semantic_judge_enabled:
            raise ValueError("offline Phase 4 cannot enable a model judge")
        if not self.human_adjudication_required:
            raise ValueError("source-valid selection requires human adjudication")
        if self.selection_tiers != [30, 100, 250, 400]:
            raise ValueError("Phase 4 selection tiers must be strict S400 prefixes")
        if self.required_source_valid_membership != self.selection_tiers[-1]:
            raise ValueError("required membership must equal the S400 tier")
        return self


def load_phase4_config(path: Path) -> Phase4Config:
    with path.open("rb") as handle:
        return Phase4Config.model_validate(tomllib.load(handle), strict=True)
