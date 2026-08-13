"""Rebuildable streaming membership/no-repeat indexes."""

from __future__ import annotations

from collections.abc import Iterable

from toolcall_tr.hashing import JsonValue
from toolcall_tr.models import CanonicalEpisode

INDEX_FIELDS = (
    "source_occurrence_id",
    "source_native_id",
    "raw_record_sha256",
    "source_episode_fingerprint",
)


def rebuild_membership_index(episodes: Iterable[CanonicalEpisode]) -> list[dict[str, JsonValue]]:
    owners: dict[tuple[str, str], set[str]] = {}
    for episode in episodes:
        for source in episode.provenance.sources:
            values: dict[str, str | None] = {
                "source_occurrence_id": source.source_occurrence_id,
                "source_native_id": source.source_native_id,
                "raw_record_sha256": source.raw_record_sha256,
                "source_episode_fingerprint": episode.source_episode_fingerprint,
            }
            for identity_type, identity_value in values.items():
                if identity_value is not None:
                    owners.setdefault((identity_type, identity_value), set()).add(
                        episode.episode_id
                    )
    records: list[dict[str, JsonValue]] = []
    for (identity_type, identity_value), owner_ids in sorted(owners.items()):
        owner_episode_ids: list[JsonValue] = list(sorted(owner_ids))
        records.append(
            {
                "schema_version": "membership-index-0.1.0",
                "identity_type": identity_type,
                "identity_value": identity_value,
                "owner_episode_ids": owner_episode_ids,
            }
        )
    return records
