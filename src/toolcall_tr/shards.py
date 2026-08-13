"""Deterministic fixed-row JSONL shard publication."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from toolcall_tr.artifacts import (
    ArtifactDescriptor,
    ContentManifest,
    RowAccounting,
    publish_bytes_atomic,
)
from toolcall_tr.constants import PIPELINE_VERSION
from toolcall_tr.hashing import canonical_bytes, sha256_bytes, stable_id


def publish_jsonl_shards(
    root: Path,
    *,
    logical_name: str,
    schema_version: str,
    stage: str,
    records: Iterable[object],
    shard_rows: int,
    input_manifest_ids: list[str] | None = None,
    quarantined_rows: int = 0,
    contract_hashes: dict[str, str] | None = None,
    validator: Callable[[Path], None] | None = None,
) -> ContentManifest:
    """Validate all temp shards before publishing any immutable output."""
    if shard_rows < 1:
        raise ValueError("shard_rows must be positive")
    root.mkdir(parents=True, exist_ok=True)
    serialized = [canonical_bytes(record) + b"\n" for record in records]
    chunks: list[list[bytes]] = [
        serialized[index : index + shard_rows] for index in range(0, len(serialized), shard_rows)
    ]
    if not chunks:
        chunks = [[]]
    pending: list[tuple[ArtifactDescriptor, Path, bytes]] = []
    for index, lines in enumerate(chunks):
        payload = b"".join(lines)
        digest = sha256_bytes(payload)
        shard_name = f"{logical_name}-{index:05d}"
        filename = f"{shard_name}-{digest.removeprefix('sha256:')}.jsonl"
        target = root / filename
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
        pending.append(
            (
                ArtifactDescriptor(
                    logical_name=shard_name,
                    relative_path=filename,
                    sha256=digest,
                    size=len(payload),
                    row_count=len(lines),
                    schema_version=schema_version,
                ),
                target,
                payload,
            )
        )
    artifacts = [item[0] for item in pending]
    accounting = RowAccounting(
        input=len(serialized) + quarantined_rows,
        accepted=len(serialized),
        quarantined=quarantined_rows,
    )
    sorted_input_manifest_ids = sorted(input_manifest_ids or [])
    sorted_contract_hashes = dict(sorted((contract_hashes or {}).items()))
    body: dict[str, object] = {
        "schema_version": "content-manifest-0.1.0",
        "stage": stage,
        "pipeline_version": PIPELINE_VERSION,
        "input_manifest_ids": sorted_input_manifest_ids,
        "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
        "row_accounting": accounting.model_dump(mode="json"),
        "contract_hashes": sorted_contract_hashes,
    }
    manifest = ContentManifest(
        manifest_id=stable_id("manifest", body),
        stage=stage,
        input_manifest_ids=sorted_input_manifest_ids,
        artifacts=artifacts,
        row_accounting=accounting,
        contract_hashes=sorted_contract_hashes,
    )
    for _, target, payload in pending:
        publish_bytes_atomic(target, payload)
    publish_bytes_atomic(root / f"{manifest.manifest_id}.json", canonical_bytes(manifest) + b"\n")
    return manifest
