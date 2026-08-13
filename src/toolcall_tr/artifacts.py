"""Deterministic manifests and atomic content-addressed artifact publication."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.constants import PIPELINE_VERSION
from toolcall_tr.hashing import canonical_bytes, sha256_bytes, stable_id
from toolcall_tr.models import NonEmptyStr, Sha256, StrictModel


class ArtifactDescriptor(StrictModel):
    logical_name: NonEmptyStr
    relative_path: NonEmptyStr
    sha256: Sha256
    size: Annotated[int, Field(ge=0)]
    row_count: Annotated[int, Field(ge=0)]
    schema_version: NonEmptyStr


class RowAccounting(StrictModel):
    input: Annotated[int, Field(ge=0)]
    accepted: Annotated[int, Field(ge=0)]
    quarantined: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def balanced(self) -> RowAccounting:
        if self.input != self.accepted + self.quarantined:
            raise ValueError("row accounting must balance")
        return self


class ContentManifest(StrictModel):
    schema_version: Literal["content-manifest-0.1.0"] = "content-manifest-0.1.0"
    manifest_id: Annotated[str, Field(pattern=r"^manifest_[0-9a-f]{64}$")]
    stage: NonEmptyStr
    pipeline_version: Literal["0.1.0"] = PIPELINE_VERSION
    input_manifest_ids: list[str]
    artifacts: Annotated[list[ArtifactDescriptor], Field(min_length=1)]
    row_accounting: RowAccounting
    contract_hashes: dict[str, Sha256]

    @model_validator(mode="after")
    def validate_identity(self) -> ContentManifest:
        body = self.model_dump(mode="json", exclude={"manifest_id"})
        expected = stable_id("manifest", body)
        if self.manifest_id != expected:
            raise ValueError("manifest_id does not match deterministic manifest body")
        if self.input_manifest_ids != sorted(self.input_manifest_ids):
            raise ValueError("input manifest IDs must be sorted")
        if [item.logical_name for item in self.artifacts] != sorted(
            item.logical_name for item in self.artifacts
        ):
            raise ValueError("artifacts must be sorted by logical name")
        return self


class PublishError(RuntimeError):
    pass


def publish_bytes_atomic(target: Path, payload: bytes) -> bool:
    """Publish without overwrite. Identical existing content is an idempotent resume."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() == payload:
            return False
        raise PublishError(f"immutable artifact target exists with different bytes: {target}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A hard-link creation is atomic and, unlike Path.rename on POSIX,
            # cannot replace a target created by a concurrent publisher.
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise PublishError(f"publish race produced conflicting target: {target}") from None
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def publish_jsonl_artifact(
    root: Path,
    *,
    logical_name: str,
    schema_version: str,
    stage: str,
    records: Iterable[object],
    input_manifest_ids: list[str] | None = None,
    quarantined_rows: int = 0,
    contract_hashes: dict[str, str] | None = None,
    validator: Callable[[Path], None] | None = None,
) -> ContentManifest:
    """Render temp bytes, validate, then publish artifact and manifest by content hash."""
    root.mkdir(parents=True, exist_ok=True)
    lines: list[bytes] = []
    for record in records:
        lines.append(canonical_bytes(record) + b"\n")
    payload = b"".join(lines)
    digest = sha256_bytes(payload)
    suffix = digest.removeprefix("sha256:")
    artifact_name = f"{logical_name}-{suffix}.jsonl"
    artifact_target = root / artifact_name

    if validator is not None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".validate-", dir=root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            validator(temporary)
        finally:
            temporary.unlink(missing_ok=True)

    descriptor_record = ArtifactDescriptor(
        logical_name=logical_name,
        relative_path=artifact_name,
        sha256=digest,
        size=len(payload),
        row_count=len(lines),
        schema_version=schema_version,
    )
    accounting = RowAccounting(
        input=len(lines) + quarantined_rows,
        accepted=len(lines),
        quarantined=quarantined_rows,
    )
    sorted_input_manifest_ids = sorted(input_manifest_ids or [])
    sorted_contract_hashes = dict(sorted((contract_hashes or {}).items()))
    body: dict[str, object] = {
        "schema_version": "content-manifest-0.1.0",
        "stage": stage,
        "pipeline_version": PIPELINE_VERSION,
        "input_manifest_ids": sorted_input_manifest_ids,
        "artifacts": [descriptor_record.model_dump(mode="json")],
        "row_accounting": accounting.model_dump(mode="json"),
        "contract_hashes": sorted_contract_hashes,
    }
    manifest = ContentManifest(
        manifest_id=stable_id("manifest", body),
        stage=stage,
        input_manifest_ids=sorted_input_manifest_ids,
        artifacts=[descriptor_record],
        row_accounting=accounting,
        contract_hashes=sorted_contract_hashes,
    )
    manifest_payload = canonical_bytes(manifest) + b"\n"
    manifest_target = root / f"{manifest.manifest_id}.json"

    publish_bytes_atomic(artifact_target, payload)
    try:
        publish_bytes_atomic(manifest_target, manifest_payload)
    except BaseException:
        # Artifact remains a valid content-addressed orphan and can be resumed safely.
        raise
    return manifest
