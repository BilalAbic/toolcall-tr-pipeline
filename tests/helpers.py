from __future__ import annotations

from pathlib import Path

from toolcall_tr.adapters import get_adapter
from toolcall_tr.canonicalize import canonicalize
from toolcall_tr.models import CanonicalEpisode
from toolcall_tr.source import ingest_snapshot, register_source


def canonical_fixture(root: Path, adapter_name: str, index: int = 0) -> CanonicalEpisode:
    snapshot = register_source(
        root,
        dataset_namespace=f"fixture-{adapter_name}",
        source_revision="fixture-v1",
        license_id="test-only",
    )
    bronze = list(ingest_snapshot(snapshot, root))[index]
    assert bronze.parsed_record is not None
    adapted = get_adapter(adapter_name).adapt(bronze.parsed_record)
    return canonicalize(bronze, adapted, run_event_id="run_fixture")
