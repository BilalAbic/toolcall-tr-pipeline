"""Fail-closed, provider-free operational pilot orchestration.

The pilot accepts one explicit JSONL file and writes immutable derived artifacts
only below an explicit, disjoint output root.  It deliberately stops before any
translation, model, review, or release action.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.adapters import get_adapter
from toolcall_tr.adapters.base import AdapterError
from toolcall_tr.artifacts import ContentManifest, publish_bytes_atomic, publish_jsonl_artifact
from toolcall_tr.audit import ExactConflictAudit, audit_exact_conflicts
from toolcall_tr.canonicalize import CanonicalizationError, canonicalize
from toolcall_tr.diagnostics import Diagnostic, diagnostic
from toolcall_tr.hashing import canonical_bytes, stable_id
from toolcall_tr.models import CanonicalEpisode, NonEmptyStr, OccurrenceId, Sha256, StrictModel
from toolcall_tr.source import BronzeRecord, SourceSnapshot, ingest_snapshot, register_source
from toolcall_tr.tool_registry import ToolNormalizationError


class PilotConfigurationError(ValueError):
    """Raised before any output is written when pilot boundaries are unsafe."""


class PilotRunReport(StrictModel):
    """Deterministic report for a bounded, non-promoting source pilot."""

    schema_version: Literal["operational-pilot-0.1.0"] = "operational-pilot-0.1.0"
    pilot_id: Annotated[str, Field(pattern=r"^pilot_[0-9a-f]{64}$")]
    source_snapshot_id: Annotated[str, Field(pattern=r"^snap_[0-9a-f]{64}$")]
    input_file_sha256: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    adapter: NonEmptyStr
    run_event_id: NonEmptyStr
    status: Literal["passed", "blocked"]
    source_records: Annotated[int, Field(ge=0)]
    valid_records: Annotated[int, Field(ge=0)]
    quarantined_records: Annotated[int, Field(ge=0)]
    canonical_records: Annotated[int, Field(ge=0)]
    review_required_conflicts: Annotated[int, Field(ge=0)]
    bronze_manifest_id: NonEmptyStr
    quarantine_manifest_id: str | None
    canonical_manifest_id: str | None
    audit_id: str | None
    block_reasons: list[NonEmptyStr]

    @model_validator(mode="after")
    def validate_report(self) -> PilotRunReport:
        if self.source_records != self.valid_records + self.quarantined_records:
            raise ValueError("pilot source row accounting must balance")
        if self.block_reasons != sorted(set(self.block_reasons)):
            raise ValueError("pilot block reasons must be unique and sorted")
        if self.status == "passed":
            if self.block_reasons:
                raise ValueError("passed pilot cannot carry block reasons")
            if self.quarantined_records or self.canonical_records != self.valid_records:
                raise ValueError("passed pilot requires every source row to canonicalize")
            if self.canonical_manifest_id is None or self.audit_id is None:
                raise ValueError("passed pilot requires canonical and audit outputs")
        elif not self.block_reasons:
            raise ValueError("blocked pilot requires at least one stable block reason")
        payload = self.model_dump(mode="json", exclude={"pilot_id"})
        if self.pilot_id != stable_id("pilot", payload):
            raise ValueError("pilot ID does not match deterministic report body")
        return self


class CanonicalQuarantineRecord(StrictModel):
    """A source-valid row rejected by strict canonicalization without repair."""

    schema_version: Literal["canonical-quarantine-0.1.0"] = "canonical-quarantine-0.1.0"
    quarantine_id: Annotated[str, Field(pattern=r"^canonq_[0-9a-f]{64}$")]
    source_occurrence_id: OccurrenceId
    raw_record_sha256: Sha256
    source_line: Annotated[int, Field(gt=0)]
    diagnostic: Diagnostic

    @model_validator(mode="after")
    def validate_identity_and_evidence(self) -> CanonicalQuarantineRecord:
        if self.diagnostic.source_occurrence_id != self.source_occurrence_id:
            raise ValueError("canonical quarantine diagnostic occurrence does not match record")
        if self.diagnostic.source_line != self.source_line:
            raise ValueError("canonical quarantine diagnostic line does not match record")
        payload = self.model_dump(mode="json", exclude={"quarantine_id"})
        if self.quarantine_id != stable_id("canonq", payload):
            raise ValueError("canonical quarantine ID does not match deterministic body")
        return self


class TolerantPilotRunReport(StrictModel):
    """Additive pilot contract that preserves canonical survivors as evidence.

    ``operational-pilot-0.1.0`` remains immutable.  This distinct contract is
    used where source-valid rows must be quarantined individually after strict
    canonicalization rather than collapsing the full pilot result.
    """

    schema_version: Literal["operational-pilot-tolerant-0.1.0"] = (
        "operational-pilot-tolerant-0.1.0"
    )
    pilot_id: Annotated[str, Field(pattern=r"^pilot_[0-9a-f]{64}$")]
    source_snapshot_id: Annotated[str, Field(pattern=r"^snap_[0-9a-f]{64}$")]
    input_file_sha256: Sha256
    adapter: NonEmptyStr
    run_event_id: NonEmptyStr
    status: Literal["passed", "blocked"]
    source_records: Annotated[int, Field(ge=0)]
    valid_records: Annotated[int, Field(ge=0)]
    quarantined_records: Annotated[int, Field(ge=0)]
    canonical_records: Annotated[int, Field(ge=0)]
    canonical_quarantined_records: Annotated[int, Field(ge=0)]
    review_required_conflicts: Annotated[int, Field(ge=0)]
    bronze_manifest_id: NonEmptyStr
    quarantine_manifest_id: str | None
    canonical_manifest_id: str | None
    canonical_quarantine_manifest_id: str | None
    audit_id: str | None
    block_reasons: list[NonEmptyStr]

    @model_validator(mode="after")
    def validate_report(self) -> TolerantPilotRunReport:
        if self.source_records != self.valid_records + self.quarantined_records:
            raise ValueError("pilot source row accounting must balance")
        if self.valid_records != self.canonical_records + self.canonical_quarantined_records:
            raise ValueError("pilot canonical row accounting must balance")
        if self.block_reasons != sorted(set(self.block_reasons)):
            raise ValueError("pilot block reasons must be unique and sorted")
        if self.canonical_quarantined_records == 0:
            if self.canonical_quarantine_manifest_id is not None:
                raise ValueError("empty canonical quarantine cannot have a manifest")
        elif self.canonical_quarantine_manifest_id is None:
            raise ValueError("canonical quarantine requires a manifest")
        if self.status == "passed":
            if self.block_reasons:
                raise ValueError("passed pilot cannot carry block reasons")
            if self.quarantined_records or self.canonical_quarantined_records:
                raise ValueError("passed pilot cannot contain quarantined records")
            if self.canonical_records != self.valid_records:
                raise ValueError("passed pilot requires every valid row to canonicalize")
            if self.canonical_manifest_id is None or self.audit_id is None:
                raise ValueError("passed pilot requires canonical and audit outputs")
        elif not self.block_reasons:
            raise ValueError("blocked pilot requires at least one stable block reason")
        payload = self.model_dump(mode="json", exclude={"pilot_id"})
        if self.pilot_id != stable_id("pilot", payload):
            raise ValueError("pilot ID does not match deterministic report body")
        return self


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_boundaries(input_jsonl: Path, output_root: Path) -> tuple[Path, Path, Path]:
    input_path = input_jsonl.resolve(strict=True)
    if not input_path.is_file() or input_path.suffix.lower() != ".jsonl":
        raise PilotConfigurationError("pilot input must be an existing .jsonl file")
    source_root = input_path.parent
    resolved_output = output_root.resolve(strict=False)
    if output_root.exists() and not output_root.is_dir():
        raise PilotConfigurationError("pilot output root must be a directory")
    if _is_within(resolved_output, source_root) or _is_within(source_root, resolved_output):
        raise PilotConfigurationError(
            "pilot output root must be disjoint from the input source root"
        )
    return input_path, source_root, resolved_output


def _canonical_quarantine(
    row: BronzeRecord,
    *,
    code: str,
    message: str,
    json_pointer: str | None,
) -> CanonicalQuarantineRecord:
    envelope = diagnostic(
        code,
        message,
        source_occurrence_id=row.source_occurrence_id,
        source_line=row.source_line,
        json_pointer=json_pointer,
    )
    body: dict[str, object] = {
        "schema_version": "canonical-quarantine-0.1.0",
        "source_occurrence_id": row.source_occurrence_id,
        "raw_record_sha256": row.raw_record_sha256,
        "source_line": row.source_line,
        "diagnostic": envelope.model_dump(mode="json", exclude_none=False),
    }
    return CanonicalQuarantineRecord(
        quarantine_id=stable_id("canonq", body),
        source_occurrence_id=row.source_occurrence_id,
        raw_record_sha256=row.raw_record_sha256,
        source_line=row.source_line,
        diagnostic=envelope,
    )


def _new_tolerant_report(
    *,
    snapshot: SourceSnapshot,
    adapter: str,
    run_event_id: str,
    source_records: int,
    valid_records: int,
    quarantined_records: int,
    canonical_records: int,
    canonical_quarantined_records: int,
    review_required_conflicts: int,
    bronze_manifest: ContentManifest,
    quarantine_manifest: ContentManifest | None,
    canonical_manifest: ContentManifest | None,
    canonical_quarantine_manifest: ContentManifest | None,
    audit: ExactConflictAudit | None,
    block_reasons: list[str],
) -> TolerantPilotRunReport:
    reasons = sorted(set(block_reasons))
    status: Literal["passed", "blocked"] = "blocked" if reasons else "passed"
    body: dict[str, object] = {
        "schema_version": "operational-pilot-tolerant-0.1.0",
        "source_snapshot_id": snapshot.snapshot_id,
        "input_file_sha256": snapshot.files[0].sha256,
        "adapter": adapter,
        "run_event_id": run_event_id,
        "status": status,
        "source_records": source_records,
        "valid_records": valid_records,
        "quarantined_records": quarantined_records,
        "canonical_records": canonical_records,
        "canonical_quarantined_records": canonical_quarantined_records,
        "review_required_conflicts": review_required_conflicts,
        "bronze_manifest_id": bronze_manifest.manifest_id,
        "quarantine_manifest_id": (
            quarantine_manifest.manifest_id if quarantine_manifest is not None else None
        ),
        "canonical_manifest_id": (
            canonical_manifest.manifest_id if canonical_manifest is not None else None
        ),
        "canonical_quarantine_manifest_id": (
            canonical_quarantine_manifest.manifest_id
            if canonical_quarantine_manifest is not None
            else None
        ),
        "audit_id": audit.audit_id if audit is not None else None,
        "block_reasons": reasons,
    }
    return TolerantPilotRunReport(
        pilot_id=stable_id("pilot", body),
        source_snapshot_id=snapshot.snapshot_id,
        input_file_sha256=snapshot.files[0].sha256,
        adapter=adapter,
        run_event_id=run_event_id,
        status=status,
        source_records=source_records,
        valid_records=valid_records,
        quarantined_records=quarantined_records,
        canonical_records=canonical_records,
        canonical_quarantined_records=canonical_quarantined_records,
        review_required_conflicts=review_required_conflicts,
        bronze_manifest_id=bronze_manifest.manifest_id,
        quarantine_manifest_id=(
            quarantine_manifest.manifest_id if quarantine_manifest is not None else None
        ),
        canonical_manifest_id=(
            canonical_manifest.manifest_id if canonical_manifest is not None else None
        ),
        canonical_quarantine_manifest_id=(
            canonical_quarantine_manifest.manifest_id
            if canonical_quarantine_manifest is not None
            else None
        ),
        audit_id=audit.audit_id if audit is not None else None,
        block_reasons=reasons,
    )


def run_operational_pilot(
    input_jsonl: Path,
    output_root: Path,
    *,
    dataset_namespace: str,
    source_revision: str,
    license_id: str,
    adapter_name: str,
    run_event_id: str,
    max_record_bytes: int = 8 * 1024 * 1024,
    source_config: str | None = None,
    source_split: str | None = None,
    license_url: str | None = None,
) -> TolerantPilotRunReport:
    """Run ingest, canonicalization, and exact audit without provider access.

    All stages are evaluated in memory before publication.  Malformed source
    records and source-valid records that fail strict adaptation/canonicalization
    each remain explicit quarantine evidence; canonical survivors are retained
    for audit but the pilot is blocked.  The input snapshot is re-registered
    immediately before output publication, so a changed source fails before it
    can be represented as a completed pilot.
    """
    if not run_event_id.strip():
        raise PilotConfigurationError("run_event_id must be non-empty")
    if max_record_bytes < 1:
        raise PilotConfigurationError("max_record_bytes must be positive")
    input_path, source_root, resolved_output = _validate_boundaries(input_jsonl, output_root)
    try:
        adapter = get_adapter(adapter_name)
    except ValueError as exc:
        raise PilotConfigurationError(str(exc)) from exc
    snapshot = register_source(
        source_root,
        dataset_namespace=dataset_namespace,
        source_revision=source_revision,
        license_id=license_id,
        relative_files=[input_path.name],
        source_config=source_config,
        source_split=source_split,
        license_url=license_url,
    )
    rows = list(ingest_snapshot(snapshot, source_root, max_record_bytes=max_record_bytes))
    valid_rows = [row for row in rows if row.status == "valid"]
    quarantined_rows = [row for row in rows if row.status == "quarantined"]
    canonical: list[CanonicalEpisode] = []
    canonical_quarantines: list[CanonicalQuarantineRecord] = []
    block_reasons: list[str] = []
    audit: ExactConflictAudit | None = None

    if not rows:
        block_reasons.append("ingest.empty_source")
    if quarantined_rows:
        block_reasons.append("ingest.quarantined_records")
    for row in valid_rows:
        if row.parsed_record is None:
            canonical_quarantines.append(
                _canonical_quarantine(
                    row,
                    code="SCHEMA_CANONICAL_INVALID",
                    message="Canonical episode failed strict schema/state validation.",
                    json_pointer=None,
                )
            )
            block_reasons.append("canonical.validation_error")
            continue
        try:
            canonical.append(
                canonicalize(
                    row,
                    adapter.adapt(row.parsed_record),
                    run_event_id=run_event_id,
                )
            )
        except AdapterError as exc:
            canonical_quarantines.append(
                _canonical_quarantine(
                    row,
                    code=exc.code,
                    message="Source adapter rejected a required source field.",
                    json_pointer=exc.pointer,
                )
            )
            block_reasons.append(f"adapter.{exc.code.lower()}")
        except ToolNormalizationError as exc:
            canonical_quarantines.append(
                _canonical_quarantine(
                    row,
                    code=exc.code,
                    message="Tool definition failed strict normalization.",
                    json_pointer=f"/tools{exc.pointer}",
                )
            )
            block_reasons.append(f"canonical.{exc.code.lower()}")
        except CanonicalizationError as exc:
            canonical_quarantines.append(
                _canonical_quarantine(
                    row,
                    code=exc.code,
                    message="Canonical tool call failed strict validation.",
                    json_pointer=exc.pointer,
                )
            )
            block_reasons.append(f"canonical.{exc.code.lower()}")
        except ValueError:
            canonical_quarantines.append(
                _canonical_quarantine(
                    row,
                    code="SCHEMA_CANONICAL_INVALID",
                    message="Canonical episode failed strict schema/state validation.",
                    json_pointer=None,
                )
            )
            block_reasons.append("canonical.validation_error")
    if canonical_quarantines:
        block_reasons.append("canonical.quarantined_records")
    if canonical:
        audit = audit_exact_conflicts(canonical)
        if audit.conflict_candidates:
            block_reasons.append("audit.review_required_conflicts")

    # Rehash exactly the selected source file before any artifact is published.
    verified_snapshot = register_source(
        source_root,
        dataset_namespace=dataset_namespace,
        source_revision=source_revision,
        license_id=license_id,
        relative_files=[input_path.name],
        source_config=source_config,
        source_split=source_split,
        license_url=license_url,
    )
    if verified_snapshot != snapshot:
        raise PilotConfigurationError("input JSONL changed during pilot execution")

    publish_bytes_atomic(
        resolved_output / "snapshots" / f"{snapshot.snapshot_id}.json",
        canonical_bytes(snapshot) + b"\n",
    )
    bronze_manifest = publish_jsonl_artifact(
        resolved_output / "bronze",
        logical_name="bronze",
        schema_version="bronze-record-0.1.0",
        stage="pilot-ingest",
        records=[row.model_dump(mode="json", exclude_none=False) for row in valid_rows],
        input_manifest_ids=[snapshot.snapshot_id],
        quarantined_rows=len(quarantined_rows),
    )
    quarantine_manifest: ContentManifest | None = None
    if quarantined_rows:
        quarantine_manifest = publish_jsonl_artifact(
            resolved_output / "quarantine",
            logical_name="ingest-quarantine",
            schema_version="bronze-record-0.1.0",
            stage="pilot-ingest-quarantine",
            records=[row.model_dump(mode="json", exclude_none=False) for row in quarantined_rows],
            input_manifest_ids=[snapshot.snapshot_id],
        )

    canonical_manifest: ContentManifest | None = None
    if audit is not None:
        canonical_manifest = publish_jsonl_artifact(
            resolved_output / "canonical",
            logical_name="canonical",
            schema_version="0.1.0",
            stage="pilot-canonicalize",
            records=[episode.model_dump(mode="json", exclude_none=False) for episode in canonical],
            input_manifest_ids=[bronze_manifest.manifest_id],
        )
        publish_bytes_atomic(
            resolved_output / "audit" / f"{audit.audit_id}.json",
            canonical_bytes(audit) + b"\n",
        )

    canonical_quarantine_manifest: ContentManifest | None = None
    if canonical_quarantines:
        canonical_quarantine_manifest = publish_jsonl_artifact(
            resolved_output / "canonical-quarantine",
            logical_name="canonical-quarantine",
            schema_version="canonical-quarantine-0.1.0",
            stage="pilot-canonicalize-quarantine",
            records=[
                record.model_dump(mode="json", exclude_none=False)
                for record in canonical_quarantines
            ],
            input_manifest_ids=[bronze_manifest.manifest_id],
        )

    report = _new_tolerant_report(
        snapshot=snapshot,
        adapter=adapter_name,
        run_event_id=run_event_id,
        source_records=len(rows),
        valid_records=len(valid_rows),
        quarantined_records=len(quarantined_rows),
        canonical_records=len(canonical),
        canonical_quarantined_records=len(canonical_quarantines),
        review_required_conflicts=len(audit.conflict_candidates) if audit is not None else 0,
        bronze_manifest=bronze_manifest,
        quarantine_manifest=quarantine_manifest,
        canonical_manifest=canonical_manifest,
        canonical_quarantine_manifest=canonical_quarantine_manifest,
        audit=audit,
        block_reasons=block_reasons,
    )
    publish_bytes_atomic(
        resolved_output / "pilot" / f"{report.pilot_id}.json",
        canonical_bytes(report) + b"\n",
    )
    return report
