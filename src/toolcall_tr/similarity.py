"""Deterministic near-duplicate candidate retrieval without automatic deletion."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.models import EpisodeId, NonEmptyStr, StrictModel

_WHITESPACE = re.compile(r"\s+")


class SimilarityDocument(StrictModel):
    schema_version: Literal["similarity-document-0.1.0"] = "similarity-document-0.1.0"
    episode_id: EpisodeId
    text: NonEmptyStr


class NearDuplicateCandidate(StrictModel):
    """A review edge; similarity alone is never an authorization to drop data."""

    schema_version: Literal["near-duplicate-candidate-0.1.0"] = "near-duplicate-candidate-0.1.0"
    left_episode_id: EpisodeId
    right_episode_id: EpisodeId
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    ngram_size: Annotated[int, Field(gt=0)]
    disposition: Literal["human_review"] = "human_review"
    automatic_drop: Literal[False] = False

    @model_validator(mode="after")
    def ordered_distinct_pair(self) -> NearDuplicateCandidate:
        if self.left_episode_id >= self.right_episode_id:
            raise ValueError("candidate episode IDs must be distinct and sorted")
        return self


def normalize_similarity_text(text: str) -> str:
    """Apply a version-stable, language-neutral normalization for retrieval only."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", normalized).strip()


def character_ngrams(text: str, *, ngram_size: int = 5) -> frozenset[str]:
    if ngram_size < 1:
        raise ValueError("ngram_size must be positive")
    normalized = normalize_similarity_text(text)
    if not normalized:
        return frozenset()
    if len(normalized) <= ngram_size:
        return frozenset((normalized,))
    return frozenset(
        normalized[index : index + ngram_size] for index in range(len(normalized) - ngram_size + 1)
    )


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union)


def retrieve_near_duplicate_candidates(
    documents: Mapping[str, str],
    *,
    threshold: float = 0.8,
    ngram_size: int = 5,
) -> list[NearDuplicateCandidate]:
    """Return candidate pairs using an inverted n-gram index.

    The result is evidence for human review only. It never selects a representative
    or removes either endpoint.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    if ngram_size < 1:
        raise ValueError("ngram_size must be positive")

    grams_by_id: dict[str, frozenset[str]] = {}
    postings: dict[str, list[str]] = defaultdict(list)
    for episode_id, text in sorted(documents.items()):
        grams = character_ngrams(text, ngram_size=ngram_size)
        if not grams:
            raise ValueError(f"similarity text is empty after normalization: {episode_id}")
        grams_by_id[episode_id] = grams
        for gram in grams:
            postings[gram].append(episode_id)

    possible_pairs: set[tuple[str, str]] = set()
    if threshold == 0.0:
        identifiers = sorted(grams_by_id)
        possible_pairs.update(
            (left, right)
            for left_index, left in enumerate(identifiers)
            for right in identifiers[left_index + 1 :]
        )
    else:
        for identifiers in postings.values():
            for left_index, left in enumerate(identifiers):
                possible_pairs.update((left, right) for right in identifiers[left_index + 1 :])

    candidates: list[NearDuplicateCandidate] = []
    for left, right in sorted(possible_pairs):
        score = jaccard_similarity(grams_by_id[left], grams_by_id[right])
        if score >= threshold:
            candidates.append(
                NearDuplicateCandidate(
                    left_episode_id=left,
                    right_episode_id=right,
                    score=score,
                    ngram_size=ngram_size,
                )
            )
    return candidates


def connected_components(
    episode_ids: Iterable[str],
    edges: Iterable[NearDuplicateCandidate | tuple[str, str]],
) -> list[list[str]]:
    """Return deterministically ordered graph components, including isolated nodes."""
    parent = {episode_id: episode_id for episode_id in episode_ids}

    def find(identifier: str) -> str:
        root = identifier
        while parent[root] != root:
            root = parent[root]
        while parent[identifier] != identifier:
            next_identifier = parent[identifier]
            parent[identifier] = root
            identifier = next_identifier
        return root

    def union(left: str, right: str) -> None:
        if left not in parent or right not in parent:
            raise ValueError("edge endpoint is absent from episode_ids")
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for edge in edges:
        if isinstance(edge, NearDuplicateCandidate):
            union(edge.left_episode_id, edge.right_episode_id)
        else:
            union(*edge)

    members: dict[str, list[str]] = defaultdict(list)
    for episode_id in sorted(parent):
        members[find(episode_id)].append(episode_id)
    return sorted((sorted(component) for component in members.values()), key=lambda item: item[0])
