"""Offline, fail-closed release-manifest verification for Gold JSONL data.

The contract deliberately has no provider, environment, registry, or network
dependency.  A release manifest is exactly one strict JSONL record.  It binds
the ordered list of dataset files, their byte hashes and row counts, and every
released episode to an explicit human review ID.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, cast

from pydantic import Field, ValidationError, model_validator

from toolcall_tr.eval_contract import (
    GoldAcceptance,
    HumanEvaluationReview,
    HumanReviewId,
    ModelEvaluationVerdict,
    ModelVerdictId,
)
from toolcall_tr.hashing import JsonValue, sha256_jcs, stable_id
from toolcall_tr.jsonio import StrictJsonError, iter_jsonl, write_jsonl
from toolcall_tr.models import EpisodeId, NonEmptyStr, Sha256, StrictModel

ReleaseManifestId = Annotated[str, Field(pattern=r"^release_[0-9a-f]{64}$")]


class ReleaseContractError(ValueError):
    """Raised when a release cannot be verified completely offline."""


class ReleaseDatasetFile(StrictModel):
    """The immutable byte and row identity of one release JSONL file."""

    relative_path: NonEmptyStr
    sha256: Sha256
    row_count: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_jsonl_path(self) -> ReleaseDatasetFile:
        if self.relative_path != _normalized_relative_path(self.relative_path):
            raise ValueError("release file path must use canonical POSIX separators")
        if not self.relative_path.endswith(".jsonl"):
            raise ValueError("release dataset files must use the .jsonl suffix")
        return self


class ReleaseGoldMember(StrictModel):
    """One dataset episode and the exact human review that admits it to Gold."""

    episode_id: EpisodeId
    verdict_id: ModelVerdictId
    human_review_id: HumanReviewId


class ReleaseManifest(StrictModel):
    """One content-addressed, self-consistent declaration of a Gold release."""

    schema_version: Literal["release-manifest-0.1.0"] = "release-manifest-0.1.0"
    release_id: ReleaseManifestId
    dataset_sha256: Sha256
    dataset_row_count: Annotated[int, Field(gt=0)]
    files: Annotated[list[ReleaseDatasetFile], Field(min_length=1)]
    gold_members: Annotated[list[ReleaseGoldMember], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_identity_and_membership(self) -> ReleaseManifest:
        paths = [item.relative_path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("release files must be unique and sorted by relative path")

        member_ids = [item.episode_id for item in self.gold_members]
        if member_ids != sorted(member_ids) or len(member_ids) != len(set(member_ids)):
            raise ValueError("gold members must be unique and sorted by episode ID")
        review_ids = [item.human_review_id for item in self.gold_members]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("a human review ID can admit only one released episode")
        if self.dataset_row_count != sum(item.row_count for item in self.files):
            raise ValueError("release dataset row count does not match file row counts")
        if self.dataset_row_count != len(self.gold_members):
            raise ValueError("release dataset row count does not match gold membership")
        if self.dataset_sha256 != _dataset_hash(self.files):
            raise ValueError("release dataset hash does not match ordered file identities")

        body = self.model_dump(mode="json", exclude={"release_id"})
        if self.release_id != stable_id("release", body):
            raise ValueError("release ID does not match deterministic manifest body")
        return self


def build_release_manifest(
    dataset_root: Path,
    *,
    relative_files: list[str],
    gold_members: list[ReleaseGoldMember],
    model_verdicts: Sequence[ModelEvaluationVerdict],
    human_reviews: Sequence[HumanEvaluationReview],
    gold_acceptances: Sequence[GoldAcceptance],
) -> ReleaseManifest:
    """Build and fully verify an in-memory manifest from an immutable local dataset.

    ``relative_files`` must already be in canonical path order.  This avoids a
    builder silently changing the release's byte-level ordering.
    """
    normalized_files = _validated_ordered_relative_files(relative_files)
    root = _resolve_root(dataset_root)
    files = [_describe_dataset_file(root, relative_path)[0] for relative_path in normalized_files]
    ordered_members = list(gold_members)
    dataset_sha256 = _dataset_hash(files)
    dataset_row_count = sum(item.row_count for item in files)
    body: dict[str, object] = {
        "schema_version": "release-manifest-0.1.0",
        "dataset_sha256": dataset_sha256,
        "dataset_row_count": dataset_row_count,
        "files": [item.model_dump(mode="json") for item in files],
        "gold_members": [item.model_dump(mode="json") for item in ordered_members],
    }
    manifest = ReleaseManifest(
        release_id=stable_id("release", body),
        dataset_sha256=dataset_sha256,
        dataset_row_count=dataset_row_count,
        files=files,
        gold_members=ordered_members,
    )
    validate_release_manifest(
        dataset_root,
        manifest,
        model_verdicts=model_verdicts,
        human_reviews=human_reviews,
        gold_acceptances=gold_acceptances,
    )
    return manifest


def write_release_manifest(path: Path, manifest: ReleaseManifest) -> tuple[int, int]:
    """Write exactly one canonical JSONL manifest record without overwriting."""
    return write_jsonl(path, [manifest.model_dump(mode="json")])


def read_release_manifest(path: Path) -> ReleaseManifest:
    """Read exactly one strict JSONL record and validate the manifest identity."""
    try:
        records = iter_jsonl(path)
        first = next(records)
        try:
            next(records)
        except StopIteration:
            pass
        else:
            raise ReleaseContractError("release manifest must contain exactly one JSONL record")
    except StrictJsonError as exc:
        raise ReleaseContractError(f"invalid strict JSONL release manifest: {exc}") from exc
    except StopIteration as exc:
        message = "release manifest must contain exactly one JSONL record"
        raise ReleaseContractError(message) from exc

    if not isinstance(first, dict):
        raise ReleaseContractError("release manifest record must be a JSON object")
    try:
        return ReleaseManifest.model_validate(cast(dict[str, JsonValue], first), strict=True)
    except ValidationError as exc:
        raise ReleaseContractError(f"invalid release manifest record: {exc}") from exc


def validate_release_manifest(
    dataset_root: Path,
    manifest: ReleaseManifest,
    *,
    model_verdicts: Sequence[ModelEvaluationVerdict],
    human_reviews: Sequence[HumanEvaluationReview],
    gold_acceptances: Sequence[GoldAcceptance],
) -> None:
    """Fail closed unless files, row order, and human Gold authority all agree."""
    root = _resolve_root(dataset_root)
    observed_files: list[ReleaseDatasetFile] = []
    observed_episode_ids: list[str] = []
    for expected in manifest.files:
        observed_file, file_episode_ids = _describe_dataset_file(root, expected.relative_path)
        if observed_file.sha256 != expected.sha256:
            raise ReleaseContractError(f"release file hash mismatch: {expected.relative_path}")
        if observed_file.row_count != expected.row_count:
            raise ReleaseContractError(f"release file row count mismatch: {expected.relative_path}")
        observed_files.append(observed_file)
        observed_episode_ids.extend(file_episode_ids)

    if observed_files != manifest.files:
        raise ReleaseContractError("release file identities do not match the manifest")
    if len(observed_episode_ids) != manifest.dataset_row_count:
        raise ReleaseContractError("release dataset row count does not match observed JSONL rows")
    if _dataset_hash(observed_files) != manifest.dataset_sha256:
        raise ReleaseContractError("release dataset hash does not match observed ordered files")

    expected_episode_ids = [member.episode_id for member in manifest.gold_members]
    if observed_episode_ids != expected_episode_ids:
        raise ReleaseContractError(
            "release JSONL episode IDs must exactly match ordered Gold membership"
        )
    _validate_gold_membership(
        manifest.gold_members,
        model_verdicts=model_verdicts,
        human_reviews=human_reviews,
        gold_acceptances=gold_acceptances,
    )


def _validate_gold_membership(
    gold_members: Sequence[ReleaseGoldMember],
    *,
    model_verdicts: Sequence[ModelEvaluationVerdict],
    human_reviews: Sequence[HumanEvaluationReview],
    gold_acceptances: Sequence[GoldAcceptance],
) -> None:
    verdict_by_id = _index_unique(
        model_verdicts,
        key=lambda verdict: verdict.verdict_id,
        label="model verdict ID",
    )
    review_by_id = _index_unique(
        human_reviews,
        key=lambda review: review.review_id,
        label="human review ID",
    )
    acceptance_by_verdict_id = _index_unique(
        gold_acceptances,
        key=lambda acceptance: acceptance.verdict_id,
        label="Gold acceptance verdict ID",
    )

    for member in gold_members:
        verdict = verdict_by_id.get(member.verdict_id)
        if verdict is None:
            raise ReleaseContractError(
                f"Gold member has no supplied model verdict: {member.episode_id}"
            )
        if verdict.evaluation_unit.episode_id != member.episode_id:
            raise ReleaseContractError("Gold member episode ID does not match its model verdict")

        review = review_by_id.get(member.human_review_id)
        if review is None:
            raise ReleaseContractError(
                f"Gold member has no supplied human review: {member.episode_id}"
            )
        if review.verdict_id != member.verdict_id:
            raise ReleaseContractError("human review does not match the Gold member verdict")
        if review.reviewer_authority != "human" or review.decision != "accept_for_gold":
            raise ReleaseContractError("Gold member requires an explicit human acceptance review")

        acceptance = acceptance_by_verdict_id.get(member.verdict_id)
        if acceptance is None:
            raise ReleaseContractError("Gold member has no supplied Gold acceptance decision")
        if (
            acceptance.status != "human_accepted"
            or not acceptance.gold_eligible
            or acceptance.acceptance_authority != "human"
            or acceptance.human_review_id != member.human_review_id
        ):
            raise ReleaseContractError(
                "Gold member is not admitted by its explicit human review ID"
            )


def _index_unique[T, K](
    items: Iterable[T], *, key: Callable[[T], K], label: str
) -> dict[K, T]:
    indexed: dict[K, T] = {}
    for item in items:
        identifier = key(item)
        if identifier in indexed:
            raise ReleaseContractError(f"duplicate supplied {label}: {identifier}")
        indexed[identifier] = item
    return indexed


def _dataset_hash(files: Sequence[ReleaseDatasetFile]) -> str:
    """Hash the ordered file path, byte-hash, and row-count identities."""
    return sha256_jcs(
        [
            {
                "relative_path": item.relative_path,
                "sha256": item.sha256,
                "row_count": item.row_count,
            }
            for item in files
        ]
    )


def _validated_ordered_relative_files(relative_files: Sequence[str]) -> list[str]:
    if not relative_files:
        raise ReleaseContractError("release requires at least one JSONL dataset file")
    normalized = [_normalized_relative_path(item) for item in relative_files]
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        raise ReleaseContractError("release files must be unique and sorted by relative path")
    if any(not item.endswith(".jsonl") for item in normalized):
        raise ReleaseContractError("release dataset files must use the .jsonl suffix")
    return normalized


def _normalized_relative_path(relative_path: str) -> str:
    if not relative_path:
        raise ValueError("release file path must be non-empty")
    native_path = Path(relative_path)
    normalized = PurePosixPath(relative_path.replace("\\", "/"))
    if (
        native_path.is_absolute()
        or normalized.is_absolute()
        or ".." in normalized.parts
        or str(normalized) in {"", "."}
    ):
        raise ValueError(f"unsafe release-relative path: {relative_path}")
    return normalized.as_posix()


def _resolve_root(root: Path) -> Path:
    try:
        resolved = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ReleaseContractError(f"release dataset root does not exist: {root}") from exc
    if not resolved.is_dir():
        raise ReleaseContractError(f"release dataset root is not a directory: {root}")
    return resolved


def _describe_dataset_file(root: Path, relative_path: str) -> tuple[ReleaseDatasetFile, list[str]]:
    normalized = _normalized_relative_path(relative_path)
    try:
        path = (root / Path(normalized)).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ReleaseContractError(f"release dataset file is missing: {normalized}") from exc
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReleaseContractError(f"release dataset file escapes root: {normalized}") from exc
    if not path.is_file():
        raise ReleaseContractError(f"release dataset path is not a file: {normalized}")

    before_hash = _file_hash(path)
    episode_ids = _read_release_episode_ids(path)
    after_hash = _file_hash(path)
    if before_hash != after_hash:
        raise ReleaseContractError(f"release dataset file changed while validating: {normalized}")
    return (
        ReleaseDatasetFile(
            relative_path=normalized,
            sha256=after_hash,
            row_count=len(episode_ids),
        ),
        episode_ids,
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read_release_episode_ids(path: Path) -> list[str]:
    episode_ids: list[str] = []
    try:
        for line_number, value in enumerate(iter_jsonl(path), start=1):
            if not isinstance(value, dict):
                raise ReleaseContractError(
                    f"release dataset record must be an object at {path.name}:{line_number}"
                )
            episode_id = value.get("episode_id")
            if not isinstance(episode_id, str) or not _is_episode_id(episode_id):
                raise ReleaseContractError(
                    f"release dataset record has invalid episode_id at {path.name}:{line_number}"
                )
            episode_ids.append(episode_id)
    except StrictJsonError as exc:
        raise ReleaseContractError(f"invalid strict JSONL dataset file {path.name}: {exc}") from exc
    if not episode_ids:
        raise ReleaseContractError(f"release dataset file is empty: {path.name}")
    return episode_ids


def _is_episode_id(value: str) -> bool:
    suffix = value.removeprefix("ep_")
    return len(suffix) == 64 and value.startswith("ep_") and all(
        character in "0123456789abcdef" for character in suffix
    )
