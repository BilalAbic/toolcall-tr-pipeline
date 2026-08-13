"""Disjoint deterministic IDs for snapshots, occurrences, episodes, and variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from toolcall_tr.constants import CANONICAL_SCHEMA_VERSION, ID_VERSION
from toolcall_tr.hashing import stable_id


def snapshot_id(
    dataset_namespace: str,
    source_revision: str,
    files: Sequence[Mapping[str, object]],
) -> str:
    return stable_id(
        "snap",
        {
            "id_version": ID_VERSION,
            "dataset_namespace": dataset_namespace,
            "source_revision": source_revision,
            "files": files,
        },
    )


def occurrence_id(snapshot: str, relative_file_path: str, byte_offset: int) -> str:
    return stable_id(
        "occ",
        {
            "id_version": ID_VERSION,
            "snapshot_id": snapshot,
            "relative_file_path": relative_file_path,
            "byte_offset": byte_offset,
        },
    )


def episode_id(occurrence: str, conversation_id: str, target_message_index: int) -> str:
    return stable_id(
        "ep",
        {
            "id_version": ID_VERSION,
            "source_occurrence_id": occurrence,
            "source_conversation_id": conversation_id,
            "target_message_index": target_message_index,
            "episode_schema_version": CANONICAL_SCHEMA_VERSION,
        },
    )
