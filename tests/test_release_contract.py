from __future__ import annotations

from pathlib import Path

import pytest

from toolcall_tr.eval_contract import (
    GoldAcceptance,
    HumanEvaluationReview,
    ModelEvaluationVerdict,
    build_evaluation_unit,
    build_human_evaluation_review,
    build_model_verdict,
    decide_gold_acceptance,
)
from toolcall_tr.jsonio import write_jsonl
from toolcall_tr.release_contract import (
    ReleaseContractError,
    ReleaseGoldMember,
    build_release_manifest,
    read_release_manifest,
    validate_release_manifest,
    write_release_manifest,
)


def _episode_id(character: str) -> str:
    return f"ep_{character * 64}"


def _accepted_gold_evidence(
    episode_id: str, character: str
) -> tuple[ReleaseGoldMember, ModelEvaluationVerdict, HumanEvaluationReview, GoldAcceptance]:
    unit = build_evaluation_unit(
        episode_id=episode_id,
        segment_id=f"seg_{character * 64}",
        path="/conversation/0/content",
        source_text_sha256=f"sha256:{character * 64}",
        target_text_sha256=f"sha256:{'a' * 64}",
    )
    verdict = build_model_verdict(
        evaluation_unit=unit,
        evaluator_label="offline-release-test",
        conclusion="pass",
    )
    review = build_human_evaluation_review(
        verdict_id=verdict.verdict_id,
        reviewer_id="reviewer-17",
        decision="accept_for_gold",
        reviewed_finding_ids=[],
        rationale="Verified manually for offline release fixture.",
    )
    acceptance = decide_gold_acceptance(model_verdict=verdict, human_review=review)
    return (
        ReleaseGoldMember(
            episode_id=episode_id,
            verdict_id=verdict.verdict_id,
            human_review_id=review.review_id,
        ),
        verdict,
        review,
        acceptance,
    )


def _write_dataset(root: Path, episode_ids: list[str]) -> None:
    root.mkdir()
    write_jsonl(root / "gold.jsonl", [{"episode_id": item} for item in episode_ids])


def test_release_manifest_round_trips_as_one_strict_jsonl_record_and_verifies(
    tmp_path: Path,
) -> None:
    first = _accepted_gold_evidence(_episode_id("1"), "3")
    second = _accepted_gold_evidence(_episode_id("2"), "4")
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root, [first[0].episode_id, second[0].episode_id])

    manifest = build_release_manifest(
        dataset_root,
        relative_files=["gold.jsonl"],
        gold_members=[first[0], second[0]],
        model_verdicts=[first[1], second[1]],
        human_reviews=[first[2], second[2]],
        gold_acceptances=[first[3], second[3]],
    )
    manifest_path = tmp_path / "manifests" / "release.jsonl"
    assert write_release_manifest(manifest_path, manifest)[0] == 1
    assert read_release_manifest(manifest_path) == manifest

    validate_release_manifest(
        dataset_root,
        manifest,
        model_verdicts=[first[1], second[1]],
        human_reviews=[first[2], second[2]],
        gold_acceptances=[first[3], second[3]],
    )


def test_release_verification_rejects_a_byte_changed_dataset_even_when_rows_match(
    tmp_path: Path,
) -> None:
    evidence = _accepted_gold_evidence(_episode_id("1"), "3")
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root, [evidence[0].episode_id])
    manifest = build_release_manifest(
        dataset_root,
        relative_files=["gold.jsonl"],
        gold_members=[evidence[0]],
        model_verdicts=[evidence[1]],
        human_reviews=[evidence[2]],
        gold_acceptances=[evidence[3]],
    )
    (dataset_root / "gold.jsonl").write_bytes(
        f'{{"episode_id":"{evidence[0].episode_id}","tampered":true}}\n'.encode()
    )

    with pytest.raises(ReleaseContractError, match="file hash mismatch"):
        validate_release_manifest(
            dataset_root,
            manifest,
            model_verdicts=[evidence[1]],
            human_reviews=[evidence[2]],
            gold_acceptances=[evidence[3]],
        )


def test_release_requires_the_exact_explicit_human_acceptance_id(tmp_path: Path) -> None:
    evidence = _accepted_gold_evidence(_episode_id("1"), "3")
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root, [evidence[0].episode_id])
    manifest = build_release_manifest(
        dataset_root,
        relative_files=["gold.jsonl"],
        gold_members=[evidence[0]],
        model_verdicts=[evidence[1]],
        human_reviews=[evidence[2]],
        gold_acceptances=[evidence[3]],
    )
    pending = decide_gold_acceptance(model_verdict=evidence[1])

    with pytest.raises(ReleaseContractError, match="not admitted by its explicit human review ID"):
        validate_release_manifest(
            dataset_root,
            manifest,
            model_verdicts=[evidence[1]],
            human_reviews=[evidence[2]],
            gold_acceptances=[pending],
        )


def test_release_membership_must_match_the_dataset_episode_order(tmp_path: Path) -> None:
    first = _accepted_gold_evidence(_episode_id("1"), "3")
    second = _accepted_gold_evidence(_episode_id("2"), "4")
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root, [second[0].episode_id, first[0].episode_id])

    with pytest.raises(ReleaseContractError, match="episode IDs must exactly match"):
        build_release_manifest(
            dataset_root,
            relative_files=["gold.jsonl"],
            gold_members=[first[0], second[0]],
            model_verdicts=[first[1], second[1]],
            human_reviews=[first[2], second[2]],
            gold_acceptances=[first[3], second[3]],
        )


def test_release_reader_rejects_multiple_or_invalid_strict_jsonl_records(tmp_path: Path) -> None:
    too_many = tmp_path / "too-many.jsonl"
    too_many.write_bytes(b"{}\n{}\n")
    with pytest.raises(ReleaseContractError, match="exactly one"):
        read_release_manifest(too_many)

    duplicate_key = tmp_path / "duplicate-key.jsonl"
    duplicate_key.write_bytes(b'{"release_id":"x","release_id":"y"}\n')
    with pytest.raises(ReleaseContractError, match="strict JSONL"):
        read_release_manifest(duplicate_key)


def test_release_builder_requires_sorted_relative_file_paths(tmp_path: Path) -> None:
    evidence = _accepted_gold_evidence(_episode_id("1"), "3")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    write_jsonl(dataset_root / "a.jsonl", [{"episode_id": evidence[0].episode_id}])
    write_jsonl(dataset_root / "b.jsonl", [{"episode_id": evidence[0].episode_id}])

    with pytest.raises(ReleaseContractError, match="unique and sorted"):
        build_release_manifest(
            dataset_root,
            relative_files=["b.jsonl", "a.jsonl"],
            gold_members=[evidence[0]],
            model_verdicts=[evidence[1]],
            human_reviews=[evidence[2]],
            gold_acceptances=[evidence[3]],
        )
