"""Immutable source registration and occurrence-ID-first bronze ingest."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from toolcall_tr.diagnostics import Diagnostic, diagnostic
from toolcall_tr.hashing import JsonValue, sha256_bytes
from toolcall_tr.ids import occurrence_id, snapshot_id
from toolcall_tr.jsonio import StrictJsonError, loads_strict_bytes
from toolcall_tr.models import NonEmptyStr, OccurrenceId, Sha256, SnapshotId, StrictModel


class SnapshotFile(StrictModel):
    relative_path: NonEmptyStr
    sha256: Sha256
    size: Annotated[int, Field(ge=0)]
    record_count: Annotated[int, Field(ge=0)]


class SourceSnapshot(StrictModel):
    schema_version: Literal["source-snapshot-0.1.0"] = "source-snapshot-0.1.0"
    id_version: Literal[1] = 1
    snapshot_id: SnapshotId
    dataset_namespace: NonEmptyStr
    source_revision: NonEmptyStr
    source_config: str | None
    source_split: str | None
    license_id: NonEmptyStr
    license_url: str | None
    files: Annotated[list[SnapshotFile], Field(min_length=1)]
    total_size: Annotated[int, Field(ge=0)]
    total_record_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_identity_and_totals(self) -> SourceSnapshot:
        if [item.relative_path for item in self.files] != sorted(
            item.relative_path for item in self.files
        ):
            raise ValueError("snapshot files must be sorted by relative path")
        identity_files = [
            {"relative_path": item.relative_path, "sha256": item.sha256, "size": item.size}
            for item in self.files
        ]
        expected = snapshot_id(self.dataset_namespace, self.source_revision, identity_files)
        if self.snapshot_id != expected:
            raise ValueError("snapshot_id does not match immutable file identity")
        if self.total_size != sum(item.size for item in self.files):
            raise ValueError("snapshot total_size mismatch")
        if self.total_record_count != sum(item.record_count for item in self.files):
            raise ValueError("snapshot total_record_count mismatch")
        return self


class BronzeRecord(StrictModel):
    schema_version: Literal["bronze-record-0.1.0"] = "bronze-record-0.1.0"
    dataset_namespace: NonEmptyStr
    snapshot_id: SnapshotId
    source_occurrence_id: OccurrenceId
    source_sequence: Annotated[int, Field(gt=0)]
    relative_file_path: NonEmptyStr
    byte_offset: Annotated[int, Field(ge=0)]
    byte_length: Annotated[int, Field(ge=0)]
    source_line: Annotated[int, Field(gt=0)]
    raw_record_sha256: Sha256
    raw_record_utf8: str | None
    parsed_record: dict[str, JsonValue] | None
    source_native_id: str | None
    observed_paths: list[str]
    status: Literal["valid", "quarantined"]
    diagnostics: list[Diagnostic]

    @model_validator(mode="after")
    def validate_status(self) -> BronzeRecord:
        if self.status == "valid":
            if self.parsed_record is None or self.diagnostics:
                raise ValueError("valid bronze record requires parsed object and no diagnostics")
        elif not self.diagnostics:
            raise ValueError("quarantined bronze record requires diagnostics")
        return self


class SourceMutationError(RuntimeError):
    pass


def _strip_record_terminator(physical: bytes) -> bytes:
    """Remove exactly one LF or CRLF physical-record terminator."""
    if not physical.endswith(b"\n"):
        return physical
    without_lf = physical[:-1]
    return without_lf[:-1] if without_lf.endswith(b"\r") else without_lf


def _validate_relative_path(relative: str) -> str:
    normalized = PurePosixPath(relative.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts or str(normalized) in {"", "."}:
        raise ValueError(f"unsafe source-relative path: {relative}")
    return normalized.as_posix()


def _hash_and_count(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    size = 0
    records = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    with path.open("rb") as handle:
        while handle.readline():
            records += 1
    return f"sha256:{digest.hexdigest()}", size, records


def register_source(
    root: Path,
    *,
    dataset_namespace: str,
    source_revision: str,
    license_id: str,
    relative_files: list[str] | None = None,
    source_config: str | None = None,
    source_split: str | None = None,
    license_url: str | None = None,
) -> SourceSnapshot:
    resolved_root = root.resolve(strict=True)
    if relative_files is None:
        relative_files = [
            path.relative_to(resolved_root).as_posix()
            for path in resolved_root.rglob("*.jsonl")
            if path.is_file()
        ]
    normalized_files = sorted({_validate_relative_path(path) for path in relative_files})
    if not normalized_files:
        raise ValueError("source snapshot requires at least one JSONL file")
    files: list[SnapshotFile] = []
    for relative in normalized_files:
        candidate = (resolved_root / Path(relative)).resolve(strict=True)
        try:
            candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"source file escapes registered root: {relative}") from exc
        if not candidate.is_file():
            raise ValueError(f"source path is not a file: {relative}")
        digest, size, count = _hash_and_count(candidate)
        files.append(
            SnapshotFile(relative_path=relative, sha256=digest, size=size, record_count=count)
        )
    identity_files = [
        {"relative_path": item.relative_path, "sha256": item.sha256, "size": item.size}
        for item in files
    ]
    identifier = snapshot_id(dataset_namespace, source_revision, identity_files)
    return SourceSnapshot(
        snapshot_id=identifier,
        dataset_namespace=dataset_namespace,
        source_revision=source_revision,
        source_config=source_config,
        source_split=source_split,
        license_id=license_id,
        license_url=license_url,
        files=files,
        total_size=sum(item.size for item in files),
        total_record_count=sum(item.record_count for item in files),
    )


def _verify_file(path: Path, expected: SnapshotFile) -> None:
    digest, size, count = _hash_and_count(path)
    if (digest, size, count) != (expected.sha256, expected.size, expected.record_count):
        raise SourceMutationError(f"registered source file changed: {expected.relative_path}")


def ingest_snapshot(
    snapshot: SourceSnapshot,
    root: Path,
    *,
    max_record_bytes: int = 8 * 1024 * 1024,
    native_id_field: str = "id",
) -> Iterator[BronzeRecord]:
    """Assign physical identity before strict parse; never writes to source root."""
    if max_record_bytes < 1:
        raise ValueError("max_record_bytes must be positive")
    resolved_root = root.resolve(strict=True)
    sequence = 0
    for expected in snapshot.files:
        path = (resolved_root / Path(expected.relative_path)).resolve(strict=True)
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise SourceMutationError("snapshot path escapes registered source root") from exc
        _verify_file(path, expected)
        with path.open("rb") as handle:
            line_number = 0
            while physical := handle.readline():
                offset = handle.tell() - len(physical)
                line_number += 1
                sequence += 1
                raw = _strip_record_terminator(physical)
                identifier = occurrence_id(snapshot.snapshot_id, expected.relative_path, offset)
                raw_hash = sha256_bytes(raw)
                record_diagnostics: list[Diagnostic] = []
                parsed: dict[str, JsonValue] | None = None
                raw_text: str | None = None
                native_id: str | None = None
                observed_paths: list[str] = []
                if len(raw) > max_record_bytes:
                    record_diagnostics.append(
                        diagnostic(
                            "PARSE_RECORD_TOO_LARGE",
                            f"Record is {len(raw)} bytes; limit is {max_record_bytes}",
                            source_occurrence_id=identifier,
                            source_line=line_number,
                        )
                    )
                elif not raw:
                    record_diagnostics.append(
                        diagnostic(
                            "PARSE_EMPTY_RECORD",
                            "Physical JSONL record is empty",
                            source_occurrence_id=identifier,
                            source_line=line_number,
                        )
                    )
                else:
                    try:
                        raw_text = raw.decode("utf-8", errors="strict")
                        value = loads_strict_bytes(raw)
                        if not isinstance(value, dict):
                            record_diagnostics.append(
                                diagnostic(
                                    "SCHEMA_EXPECTED_OBJECT",
                                    "Top-level source record must be an object",
                                    source_occurrence_id=identifier,
                                    source_line=line_number,
                                )
                            )
                        else:
                            parsed = cast(dict[str, JsonValue], value)
                            observed_paths = [
                                f"/{key.replace('~', '~0').replace('/', '~1')}" for key in value
                            ]
                            raw_native_id = value.get(native_id_field)
                            if isinstance(raw_native_id, str | int):
                                native_id = str(raw_native_id)
                    except UnicodeDecodeError as exc:
                        record_diagnostics.append(
                            diagnostic(
                                "PARSE_INVALID_UTF8",
                                str(exc),
                                source_occurrence_id=identifier,
                                source_line=line_number,
                            )
                        )
                    except StrictJsonError as exc:
                        record_diagnostics.append(
                            diagnostic(
                                exc.code,
                                str(exc),
                                source_occurrence_id=identifier,
                                source_line=line_number,
                            )
                        )
                yield BronzeRecord(
                    dataset_namespace=snapshot.dataset_namespace,
                    snapshot_id=snapshot.snapshot_id,
                    source_occurrence_id=identifier,
                    source_sequence=sequence,
                    relative_file_path=expected.relative_path,
                    byte_offset=offset,
                    byte_length=len(raw),
                    source_line=line_number,
                    raw_record_sha256=raw_hash,
                    raw_record_utf8=raw_text,
                    parsed_record=parsed,
                    source_native_id=native_id,
                    observed_paths=observed_paths,
                    status="quarantined" if record_diagnostics else "valid",
                    diagnostics=record_diagnostics,
                )
        _verify_file(path, expected)
    if sequence != snapshot.total_record_count:
        raise SourceMutationError("snapshot row accounting changed during ingest")
