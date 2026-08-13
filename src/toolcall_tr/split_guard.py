"""Leakage guards for duplicate/near-duplicate graph components."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import model_validator

from toolcall_tr.models import EpisodeId, NonEmptyStr, StrictModel


class SplitLeakage(StrictModel):
    schema_version: Literal["split-leakage-0.1.0"] = "split-leakage-0.1.0"
    component_episode_ids: list[EpisodeId]
    observed_splits: list[NonEmptyStr]

    @model_validator(mode="after")
    def ordered_leakage(self) -> SplitLeakage:
        if self.component_episode_ids != sorted(set(self.component_episode_ids)):
            raise ValueError("component episode IDs must be unique and sorted")
        if self.observed_splits != sorted(set(self.observed_splits)):
            raise ValueError("observed splits must be unique and sorted")
        if len(self.observed_splits) < 2:
            raise ValueError("split leakage must span at least two splits")
        return self


class SplitGuardError(ValueError):
    pass


def find_split_leakage(
    components: Iterable[Iterable[str]],
    split_by_episode: Mapping[str, str],
) -> list[SplitLeakage]:
    """Report every connected component assigned to more than one split."""
    leaks: list[SplitLeakage] = []
    seen: set[str] = set()
    for raw_component in components:
        component = sorted(set(raw_component))
        if not component:
            raise SplitGuardError("components must not be empty")
        overlap = seen.intersection(component)
        if overlap:
            raise SplitGuardError(f"episode appears in multiple components: {min(overlap)}")
        seen.update(component)
        missing = [episode_id for episode_id in component if episode_id not in split_by_episode]
        if missing:
            raise SplitGuardError(f"missing split assignment: {missing[0]}")
        splits = sorted({split_by_episode[episode_id] for episode_id in component})
        if any(not split for split in splits):
            raise SplitGuardError("split names must be non-empty")
        if len(splits) > 1:
            leaks.append(SplitLeakage(component_episode_ids=component, observed_splits=splits))
    return leaks


def assert_components_in_one_split(
    components: Iterable[Iterable[str]],
    split_by_episode: Mapping[str, str],
) -> None:
    leaks = find_split_leakage(components, split_by_episode)
    if leaks:
        first = leaks[0]
        raise SplitGuardError(
            "connected component crosses splits: "
            f"{','.join(first.component_episode_ids)} -> {','.join(first.observed_splits)}"
        )
