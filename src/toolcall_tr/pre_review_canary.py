"""Bounded, non-promoting canaries that run before human final acceptance.

A pre-review canary is intentionally not a selection, adjudication, Gold
membership, or release artifact. It deterministically chooses a small set of
technically valid, conflict-free canonical episodes for prompt and provider
contract testing. Human review remains the final acceptance gate for every
later source-valid, S400, Gold, or release transition.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolcall_tr.artifacts import ContentManifest, publish_bytes_atomic, publish_jsonl_artifact
from toolcall_tr.audit import AuditId, ExactConflictAudit
from toolcall_tr.eval_contract import SegmentPathEvidence, build_evaluation_unit
from toolcall_tr.field_policy import FieldPolicy, extract_leaf_segments
from toolcall_tr.hashing import JsonValue, canonical_bytes, sha256_bytes, sha256_jcs, stable_id
from toolcall_tr.jsonio import iter_jsonl, loads_strict_bytes
from toolcall_tr.live_evaluation import LiveEvaluationInput, build_live_evaluation_input
from toolcall_tr.models import CanonicalEpisode, EpisodeId, Sha256, StrictModel
from toolcall_tr.operational_translation import OperationalTranslationResult

PRE_REVIEW_CANARY_POLICY_VERSION = "pre-review-canary-0.1.0"
CanaryId = Annotated[str, Field(pattern=r"^canary_[0-9a-f]{64}$")]


class PreReviewCanaryError(ValueError):
    """Raised before a provisional canary can hide unsafe or ambiguous input."""


class PreReviewCanaryMember(StrictModel):
    """One technical-test member; it has no quality or human-approval verdict."""

    rank: Annotated[int, Field(gt=0)]
    episode_id: EpisodeId
    input_variant_id: Sha256
    rank_key: Sha256
    translatable_segments: Annotated[int, Field(gt=0)]


class PreReviewCanaryManifest(StrictModel):
    """Immutable receipt for a provisional, non-promoting provider canary."""

    schema_version: Literal["pre-review-canary-0.1.0"] = "pre-review-canary-0.1.0"
    canary_id: CanaryId
    canary_policy_version: Literal["pre-review-canary-0.1.0"] = PRE_REVIEW_CANARY_POLICY_VERSION
    input_file_sha256s: Annotated[list[Sha256], Field(min_length=1)]
    audit_ids: list[AuditId]
    field_policy_sha256: Sha256
    requested_episode_count: Annotated[int, Field(ge=1, le=30)]
    max_translatable_segments: Annotated[int, Field(ge=1)]
    members: Annotated[list[PreReviewCanaryMember], Field(min_length=1, max_length=30)]
    total_translatable_segments: Annotated[int, Field(gt=0)]
    canonical_manifest_id: Annotated[str, Field(pattern=r"^manifest_[0-9a-f]{64}$")]
    promotion: Literal["not_eligible"] = "not_eligible"
    human_review: Literal["required_for_final_acceptance"] = "required_for_final_acceptance"

    @model_validator(mode="after")
    def validate_manifest(self) -> PreReviewCanaryManifest:
        if self.input_file_sha256s != sorted(set(self.input_file_sha256s)):
            raise ValueError("canary input file hashes must be unique and sorted")
        if self.audit_ids != sorted(set(self.audit_ids)):
            raise ValueError("canary audit IDs must be unique and sorted")
        if len(self.members) != self.requested_episode_count:
            raise ValueError("canary must contain the requested number of episodes")
        if [member.rank for member in self.members] != list(range(1, len(self.members) + 1)):
            raise ValueError("canary member ranks must be contiguous")
        if len({member.episode_id for member in self.members}) != len(self.members):
            raise ValueError("canary member episode IDs must be unique")
        if self.total_translatable_segments != sum(
            member.translatable_segments for member in self.members
        ):
            raise ValueError("canary segment total must match its member rows")
        if self.total_translatable_segments > self.max_translatable_segments:
            raise ValueError("canary exceeds its declared segment budget")
        body = self.model_dump(mode="json", exclude={"canary_id"})
        if self.canary_id != stable_id("canary", body):
            raise ValueError("canary ID does not match deterministic manifest body")
        return self


def _within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _member_rank_key(episode: CanonicalEpisode) -> str:
    return sha256_jcs(
        {
            "canary_policy_version": PRE_REVIEW_CANARY_POLICY_VERSION,
            "episode_id": episode.episode_id,
            "input_variant_id": episode.variant_id,
        }
    )


def _conflicted_episode_ids(audits: Iterable[ExactConflictAudit]) -> set[str]:
    return {
        episode_id
        for audit in audits
        for candidate in audit.conflict_candidates
        for episode_id in [
            *candidate.left_member_episode_ids,
            *candidate.right_member_episode_ids,
        ]
    }


def _exact_alias_episode_ids(audits: Iterable[ExactConflictAudit]) -> set[str]:
    return {
        episode_id
        for audit in audits
        for group in audit.duplicate_groups
        for episode_id in group.alias_episode_ids
    }


def select_pre_review_canary(
    episodes: Iterable[CanonicalEpisode],
    audits: Iterable[ExactConflictAudit],
    *,
    field_policy: FieldPolicy,
    requested_episode_count: int,
    max_translatable_segments: int,
) -> tuple[list[PreReviewCanaryMember], list[CanonicalEpisode]]:
    """Select a bounded technical-test cohort without assigning quality status.

    Exact-conflict members and exact duplicate aliases are excluded, but no
    record is dropped or repaired. The returned canonical episodes preserve
    original bytes semantically and remain ``promotion=not_eligible`` later.
    """
    if not 1 <= requested_episode_count <= 30:
        raise PreReviewCanaryError("requested canary size must be between 1 and 30")
    if max_translatable_segments < 1:
        raise PreReviewCanaryError("canary segment budget must be positive")

    episode_by_id: dict[str, CanonicalEpisode] = {}
    for episode in episodes:
        if episode.episode_id in episode_by_id:
            raise PreReviewCanaryError(f"duplicate canonical episode ID: {episode.episode_id}")
        episode_by_id[episode.episode_id] = episode
    audit_list = list(audits)
    conflicted = _conflicted_episode_ids(audit_list)
    aliases = _exact_alias_episode_ids(audit_list)

    ordered = sorted(
        (
            episode
            for episode in episode_by_id.values()
            if episode.quality.state == "unreviewed"
            and episode.annotations.decision.evidence_status == "source_explicit"
            and episode.episode_id not in conflicted
            and episode.episode_id not in aliases
        ),
        key=lambda episode: (_member_rank_key(episode), episode.episode_id),
    )
    selected: list[tuple[CanonicalEpisode, int]] = []
    segment_total = 0
    for episode in ordered:
        extraction = extract_leaf_segments(episode, field_policy)
        segment_count = len(extraction.segments)
        if not segment_count or segment_total + segment_count > max_translatable_segments:
            continue
        selected.append((episode, segment_count))
        segment_total += segment_count
        if len(selected) == requested_episode_count:
            break
    if len(selected) != requested_episode_count:
        raise PreReviewCanaryError(
            "insufficient conflict-free, policy-covered episodes within the canary segment budget"
        )
    members = [
        PreReviewCanaryMember(
            rank=rank,
            episode_id=episode.episode_id,
            input_variant_id=episode.variant_id,
            rank_key=_member_rank_key(episode),
            translatable_segments=segment_count,
        )
        for rank, (episode, segment_count) in enumerate(selected, start=1)
    ]
    return members, [episode for episode, _ in selected]


def _read_audit(path: Path) -> ExactConflictAudit:
    value = loads_strict_bytes(path.read_bytes())
    if not isinstance(value, dict):
        raise PreReviewCanaryError("conflict audit must be a JSON object")
    return ExactConflictAudit.model_validate(value, strict=True)


def prepare_pre_review_canary(
    canonical_jsonl_paths: Iterable[Path],
    audit_paths: Iterable[Path],
    output_root: Path,
    *,
    field_policy: FieldPolicy,
    requested_episode_count: int = 30,
    max_translatable_segments: int = 300,
) -> PreReviewCanaryManifest:
    """Publish a pre-review test cohort in a root disjoint from every input."""
    canonical_paths = [path.resolve(strict=True) for path in canonical_jsonl_paths]
    if not canonical_paths:
        raise PreReviewCanaryError("at least one canonical JSONL input is required")
    if any(not path.is_file() or path.suffix.lower() != ".jsonl" for path in canonical_paths):
        raise PreReviewCanaryError("canary inputs must be existing JSONL files")
    resolved_audit_paths = [path.resolve(strict=True) for path in audit_paths]
    if not resolved_audit_paths:
        raise PreReviewCanaryError("at least one conflict audit is required")
    if any(not path.is_file() or path.suffix.lower() != ".json" for path in resolved_audit_paths):
        raise PreReviewCanaryError("canary audits must be existing JSON files")
    root = output_root.resolve(strict=False)
    if output_root.exists() and not output_root.is_dir():
        raise PreReviewCanaryError("canary output root must be a directory")
    for input_path in [*canonical_paths, *resolved_audit_paths]:
        if _within(input_path, root) or _within(root, input_path.parent):
            raise PreReviewCanaryError("canary output root must be disjoint from input evidence")

    input_hashes_by_path = {path: sha256_bytes(path.read_bytes()) for path in canonical_paths}
    audit_hashes_by_path = {path: sha256_bytes(path.read_bytes()) for path in resolved_audit_paths}
    episodes = [
        CanonicalEpisode.model_validate_json(canonical_bytes(record), strict=True)
        for path in canonical_paths
        for record in iter_jsonl(path)
    ]
    audits = [_read_audit(path) for path in resolved_audit_paths]
    members, selected_episodes = select_pre_review_canary(
        episodes,
        audits,
        field_policy=field_policy,
        requested_episode_count=requested_episode_count,
        max_translatable_segments=max_translatable_segments,
    )
    if any(
        sha256_bytes(path.read_bytes()) != digest
        for path, digest in input_hashes_by_path.items()
    ):
        raise PreReviewCanaryError("canonical input changed during canary preparation")
    if any(
        sha256_bytes(path.read_bytes()) != digest
        for path, digest in audit_hashes_by_path.items()
    ):
        raise PreReviewCanaryError("conflict audit changed during canary preparation")
    canonical_manifest = publish_jsonl_artifact(
        root / "canonical",
        logical_name="pre-review-canary-canonical",
        schema_version="0.1.0",
        stage="pre-review-canary",
        records=[
            episode.model_dump(mode="json", exclude_none=False) for episode in selected_episodes
        ],
        contract_hashes={
            "field_policy": sha256_jcs(field_policy),
            "input_canonical_jsonl": sha256_jcs(sorted(input_hashes_by_path.values())),
        },
    )
    input_hashes = sorted(set(input_hashes_by_path.values()))
    body: dict[str, object] = {
        "schema_version": "pre-review-canary-0.1.0",
        "canary_policy_version": PRE_REVIEW_CANARY_POLICY_VERSION,
        "input_file_sha256s": input_hashes,
        "audit_ids": sorted(audit.audit_id for audit in audits),
        "field_policy_sha256": sha256_jcs(field_policy),
        "requested_episode_count": requested_episode_count,
        "max_translatable_segments": max_translatable_segments,
        "members": [member.model_dump(mode="json", exclude_none=False) for member in members],
        "total_translatable_segments": sum(member.translatable_segments for member in members),
        "canonical_manifest_id": canonical_manifest.manifest_id,
        "promotion": "not_eligible",
        "human_review": "required_for_final_acceptance",
    }
    manifest = PreReviewCanaryManifest(
        canary_id=stable_id("canary", body),
        input_file_sha256s=input_hashes,
        audit_ids=sorted(audit.audit_id for audit in audits),
        field_policy_sha256=sha256_jcs(field_policy),
        requested_episode_count=requested_episode_count,
        max_translatable_segments=max_translatable_segments,
        members=members,
        total_translatable_segments=sum(member.translatable_segments for member in members),
        canonical_manifest_id=canonical_manifest.manifest_id,
    )
    publish_bytes_atomic(
        root / "canaries" / f"{manifest.canary_id}.json", canonical_bytes(manifest) + b"\n"
    )
    return manifest


def _read_pointer_string(document: JsonValue, pointer: str) -> str:
    current = document
    tokens = pointer.removeprefix("/").split("/")
    if not pointer.startswith("/"):
        raise PreReviewCanaryError("canary evaluation segment pointer must be absolute")
    for raw_token in tokens:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdecimal() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise PreReviewCanaryError(f"translated segment pointer no longer resolves: {pointer}")
    if not isinstance(current, str):
        raise PreReviewCanaryError(f"translated segment is not a string: {pointer}")
    return current


def build_pre_review_evaluation_inputs(
    episodes: Iterable[CanonicalEpisode],
    translation_results: Iterable[OperationalTranslationResult],
    *,
    field_policy: FieldPolicy,
) -> list[LiveEvaluationInput]:
    """Create complete, exact source/target pairs from host-merged results.

    This derives no text and accepts no partial evidence. A result must be a
    completed, non-promoting translation of exactly one supplied source episode
    under the same field-policy hash; otherwise the prospective model-eval
    input is rejected before an OpenAI request can be made.
    """
    episode_by_id: dict[str, CanonicalEpisode] = {}
    for episode in episodes:
        if episode.episode_id in episode_by_id:
            raise PreReviewCanaryError(f"duplicate canonical episode ID: {episode.episode_id}")
        episode_by_id[episode.episode_id] = episode
    result_by_episode_id: dict[str, OperationalTranslationResult] = {}
    for result in translation_results:
        if result.episode_id in result_by_episode_id:
            raise PreReviewCanaryError(
                f"duplicate translation result episode ID: {result.episode_id}"
            )
        result_by_episode_id[result.episode_id] = result
    if set(result_by_episode_id) != set(episode_by_id):
        raise PreReviewCanaryError(
            "translation results must cover exactly the supplied canary episodes"
        )

    policy_sha256 = sha256_jcs(field_policy)
    inputs: list[LiveEvaluationInput] = []
    for episode_id in sorted(episode_by_id):
        episode = episode_by_id[episode_id]
        result = result_by_episode_id[episode_id]
        if (
            result.input_variant_id != episode.variant_id
            or result.field_policy_sha256 != policy_sha256
            or result.status != "translated"
            or result.translated_episode is None
            or result.promotion != "not_eligible"
        ):
            raise PreReviewCanaryError(
                "translation result is not an exact non-promoting canary output"
            )
        translated_document = result.translated_episode.model_dump(mode="json", exclude_none=False)
        for segment in extract_leaf_segments(episode, field_policy).segments:
            target_text = _read_pointer_string(translated_document, segment.json_pointer)
            unit = build_evaluation_unit(
                episode_id=episode.episode_id,
                segment_id=segment.segment_id,
                path=segment.json_pointer,
                source_text_sha256=sha256_bytes(segment.source_text.encode("utf-8")),
                target_text_sha256=sha256_bytes(target_text.encode("utf-8")),
            )
            inputs.append(
                build_live_evaluation_input(
                    evaluation_unit=unit,
                    evidence=SegmentPathEvidence(
                        segment_id=segment.segment_id,
                        path=segment.json_pointer,
                        source_excerpt=segment.source_text,
                        target_excerpt=target_text,
                    ),
                )
            )
    return sorted(inputs, key=lambda item: item.input_id)


def prepare_pre_review_evaluation_inputs(
    canonical_jsonl: Path,
    translation_results_jsonl: Path,
    output_root: Path,
    *,
    field_policy: FieldPolicy,
) -> ContentManifest:
    """Publish strict pre-review evaluation inputs in a disjoint artifact root."""
    canonical_path = canonical_jsonl.resolve(strict=True)
    results_path = translation_results_jsonl.resolve(strict=True)
    if (
        not canonical_path.is_file()
        or canonical_path.suffix.lower() != ".jsonl"
        or not results_path.is_file()
        or results_path.suffix.lower() != ".jsonl"
    ):
        raise PreReviewCanaryError("canary evaluation inputs must be existing JSONL files")
    root = output_root.resolve(strict=False)
    if output_root.exists() and not output_root.is_dir():
        raise PreReviewCanaryError("canary evaluation output root must be a directory")
    if (
        _within(canonical_path, root)
        or _within(results_path, root)
        or _within(root, canonical_path.parent)
        or _within(root, results_path.parent)
    ):
        raise PreReviewCanaryError(
            "canary evaluation output root must be disjoint from input evidence"
        )
    canonical_sha = sha256_bytes(canonical_path.read_bytes())
    results_sha = sha256_bytes(results_path.read_bytes())
    episodes = [
        CanonicalEpisode.model_validate_json(canonical_bytes(record), strict=True)
        for record in iter_jsonl(canonical_path)
    ]
    translation_results = [
        OperationalTranslationResult.model_validate_json(canonical_bytes(record), strict=True)
        for record in iter_jsonl(results_path)
    ]
    inputs = build_pre_review_evaluation_inputs(
        episodes,
        translation_results,
        field_policy=field_policy,
    )
    if (
        sha256_bytes(canonical_path.read_bytes()) != canonical_sha
        or sha256_bytes(results_path.read_bytes()) != results_sha
    ):
        raise PreReviewCanaryError("canary evaluation input changed during preparation")
    return publish_jsonl_artifact(
        root,
        logical_name="pre-review-live-evaluation-inputs",
        schema_version="live-evaluation-input-0.1.0",
        stage="pre-review-evaluation-inputs",
        records=[item.model_dump(mode="json", exclude_none=False) for item in inputs],
        contract_hashes={
            "canonical_canary": canonical_sha,
            "translation_results": results_sha,
            "field_policy": sha256_jcs(field_policy),
        },
    )
