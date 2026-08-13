"""Immutable conversion of one strict JSON array source into canonical JSONL."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from toolcall_tr.artifacts import publish_bytes_atomic
from toolcall_tr.hashing import JsonValue, canonical_bytes, sha256_bytes, stable_id
from toolcall_tr.jsonio import loads_strict_bytes, write_jsonl
from toolcall_tr.models import NonEmptyStr, Sha256, StrictModel


class SourceArrayConversionError(ValueError):
    """Raised before output publication for an unsafe source-array conversion."""


class JsonArrayConversionReport(StrictModel):
    """Content-derived evidence connecting an immutable JSON source to JSONL."""

    schema_version: Literal["json-array-conversion-0.1.0"] = "json-array-conversion-0.1.0"
    conversion_id: Annotated[str, Field(pattern=r"^convert_[0-9a-f]{64}$")]
    input_file_sha256: Sha256
    input_byte_size: Annotated[int, Field(ge=0)]
    input_record_count: Annotated[int, Field(ge=0)]
    output_file_sha256: Sha256
    output_byte_size: Annotated[int, Field(ge=0)]
    output_record_count: Annotated[int, Field(ge=0)]
    output_relative_path: NonEmptyStr

    @model_validator(mode="after")
    def validate_identity_and_accounting(self) -> JsonArrayConversionReport:
        if self.input_record_count != self.output_record_count:
            raise ValueError("JSON array conversion must retain every source record")
        payload = self.model_dump(mode="json", exclude={"conversion_id"})
        if self.conversion_id != stable_id("convert", payload):
            raise ValueError("conversion ID does not match deterministic report body")
        return self


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _jsonl_identity(records: list[dict[str, JsonValue]]) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for record in records:
        line = canonical_bytes(record) + b"\n"
        digest.update(line)
        size += len(line)
    return f"sha256:{digest.hexdigest()}", size


def _file_identity(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    size = 0
    record_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    with path.open("rb") as handle:
        while handle.readline():
            record_count += 1
    return f"sha256:{digest.hexdigest()}", size, record_count


def convert_json_array_to_jsonl(input_json: Path, output_root: Path) -> JsonArrayConversionReport:
    """Convert a strict object-array JSON file into an immutable JSONL artifact.

    The input bytes remain untouched.  Parsing, root/row validation, and the
    target content identity are all completed before the JSONL file is created.
    The output root must be disjoint from the input file's parent directory.
    """
    input_path = input_json.resolve(strict=True)
    if not input_path.is_file() or input_path.suffix.lower() != ".json":
        raise SourceArrayConversionError("input must be an existing .json file")
    resolved_output = output_root.resolve(strict=False)
    source_root = input_path.parent
    if _is_within(resolved_output, source_root) or _is_within(source_root, resolved_output):
        raise SourceArrayConversionError("conversion output root must be disjoint from source root")

    source_bytes = input_path.read_bytes()
    parsed = loads_strict_bytes(source_bytes)
    if not isinstance(parsed, list):
        raise SourceArrayConversionError("source JSON root must be an array")
    if not all(isinstance(record, dict) for record in parsed):
        raise SourceArrayConversionError("source JSON array must contain only objects")
    records = cast(list[dict[str, JsonValue]], parsed)

    output_hash, output_size = _jsonl_identity(records)
    output_suffix = output_hash.removeprefix("sha256:")
    relative_path = f"jsonl/source-array-{output_suffix}.jsonl"
    target = resolved_output / Path(relative_path)
    input_hash = sha256_bytes(source_bytes)

    if target.exists():
        existing_hash, existing_size, existing_count = _file_identity(target)
        if (existing_hash, existing_size, existing_count) != (
            output_hash,
            output_size,
            len(records),
        ):
            raise SourceArrayConversionError("immutable JSONL target exists with different content")
    else:
        count, size = write_jsonl(target, records)
        if (count, size) != (len(records), output_size):  # pragma: no cover - write_jsonl contract
            raise RuntimeError("JSONL writer accounting mismatch")

    body: dict[str, object] = {
        "schema_version": "json-array-conversion-0.1.0",
        "input_file_sha256": input_hash,
        "input_byte_size": len(source_bytes),
        "input_record_count": len(records),
        "output_file_sha256": output_hash,
        "output_byte_size": output_size,
        "output_record_count": len(records),
        "output_relative_path": relative_path,
    }
    report = JsonArrayConversionReport(
        conversion_id=stable_id("convert", body),
        input_file_sha256=input_hash,
        input_byte_size=len(source_bytes),
        input_record_count=len(records),
        output_file_sha256=output_hash,
        output_byte_size=output_size,
        output_record_count=len(records),
        output_relative_path=relative_path,
    )
    publish_bytes_atomic(
        resolved_output / "reports" / f"{report.conversion_id}.json",
        canonical_bytes(report) + b"\n",
    )
    return report
