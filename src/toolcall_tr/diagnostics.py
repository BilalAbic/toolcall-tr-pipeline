"""Versioned diagnostic catalog and fail-closed envelopes."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.constants import DIAGNOSTIC_CATALOG_VERSION, DIAGNOSTIC_SCHEMA_VERSION
from toolcall_tr.models import EpisodeId, NonEmptyStr, OccurrenceId, StrictModel


class CatalogEntry(StrictModel):
    code: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]+$")]
    stage: NonEmptyStr
    severity: Literal["info", "warning", "error"]
    retryable: bool
    meaning: NonEmptyStr


class DiagnosticCatalog(StrictModel):
    schema_version: Literal["diagnostic-catalog-0.1.0"]
    catalog_version: Literal["0.1.0"]
    entries: Annotated[list[CatalogEntry], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_codes(self) -> DiagnosticCatalog:
        codes = [entry.code for entry in self.entries]
        if len(codes) != len(set(codes)):
            raise ValueError("diagnostic catalog codes must be unique")
        return self

    def get(self, code: str) -> CatalogEntry:
        for entry in self.entries:
            if entry.code == code:
                return entry
        raise UnknownDiagnosticCode(code)


class UnknownDiagnosticCode(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(f"Unknown diagnostic code (fail-closed): {code}")
        self.code = code


def load_catalog() -> DiagnosticCatalog:
    resource = files("toolcall_tr.data").joinpath("diagnostic_catalog.json")
    return DiagnosticCatalog.model_validate_json(resource.read_text(encoding="utf-8"), strict=True)


CATALOG = load_catalog()


class Diagnostic(StrictModel):
    schema_version: Literal["diagnostic-0.1.0"] = DIAGNOSTIC_SCHEMA_VERSION
    diagnostic_catalog_version: Literal["0.1.0"] = DIAGNOSTIC_CATALOG_VERSION
    code: NonEmptyStr
    stage: NonEmptyStr
    severity: Literal["info", "warning", "error"]
    source_occurrence_id: OccurrenceId | None
    episode_id: EpisodeId | None
    json_pointer: str | None
    source_line: int | None
    message: NonEmptyStr
    retryable: bool
    evidence_refs: list[str]

    @model_validator(mode="after")
    def validate_against_catalog(self) -> Diagnostic:
        entry = CATALOG.get(self.code)
        if (self.stage, self.severity, self.retryable) != (
            entry.stage,
            entry.severity,
            entry.retryable,
        ):
            raise ValueError("diagnostic envelope conflicts with catalog semantics")
        if self.source_line is not None and self.source_line < 1:
            raise ValueError("source_line is one-based")
        return self


def diagnostic(
    code: str,
    message: str,
    *,
    source_occurrence_id: str | None = None,
    episode_id: str | None = None,
    json_pointer: str | None = None,
    source_line: int | None = None,
    evidence_refs: list[str] | None = None,
) -> Diagnostic:
    entry = CATALOG.get(code)
    return Diagnostic(
        code=entry.code,
        stage=entry.stage,
        severity=entry.severity,
        source_occurrence_id=source_occurrence_id,
        episode_id=episode_id,
        json_pointer=json_pointer,
        source_line=source_line,
        message=message,
        retryable=entry.retryable,
        evidence_refs=evidence_refs or [],
    )


def catalog_as_pretty_json() -> str:
    return json.dumps(CATALOG.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
