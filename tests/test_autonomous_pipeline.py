from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from tests.helpers import canonical_fixture
from tests.test_provider_adapter import config as base_config
from toolcall_tr.audit import audit_exact_conflicts
from toolcall_tr.autonomous_pipeline import (
    AutonomousPipelineError,
    build_automation_consensus,
    build_hierarchical_consensus,
    build_huggingface_review_package,
    prepare_automation_candidates,
    prepare_automation_evaluation_inputs,
    prepare_strong_escalation_inputs,
    read_automation_results,
    run_automation_translation,
)
from toolcall_tr.config import PipelineConfig, ProviderConfig, ProviderRole
from toolcall_tr.field_policy import load_field_policy
from toolcall_tr.hashing import JsonValue, canonical_bytes
from toolcall_tr.jsonio import iter_jsonl
from toolcall_tr.live_evaluation import run_live_evaluation
from toolcall_tr.models import CanonicalEpisode
from toolcall_tr.openai_judge import OpenAIResponsesJudge
from toolcall_tr.prompt_contract import load_prompt_bundle
from toolcall_tr.provider_provenance import ProviderAttemptSink
from toolcall_tr.secure_transport import TransportNetworkError

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TranslationTransport:
    flash_invalid: bool = False
    network_failure: bool = False
    calls: list[str] = field(default_factory=lambda: [])

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        body = cast(dict[str, JsonValue], json.loads(request_body))
        model = body["model"]
        assert isinstance(model, str)
        self.calls.append(model)
        if self.network_failure:
            raise TransportNetworkError()
        if self.flash_invalid and model == "deepseek-v4-flash":
            return canonical_bytes({"choices": []})
        messages = body["messages"]
        assert isinstance(messages, list)
        user_message = messages[1]
        assert isinstance(user_message, dict)
        content = user_message["content"]
        assert isinstance(content, str)
        request = cast(dict[str, JsonValue], json.loads(content))
        segments = request["segments"]
        assert isinstance(segments, list) and len(segments) == 1
        segment = segments[0]
        assert isinstance(segment, dict)
        source_text = segment["source_text"]
        segment_id = segment["segment_id"]
        request_id = request["request_id"]
        assert isinstance(source_text, str)
        assert isinstance(segment_id, str)
        assert isinstance(request_id, str)
        response_text = json.dumps(
            {
                "schema_version": "translation-response-0.1.0",
                "request_id": request_id,
                "status": "translated",
                "segments": [
                    {
                        "segment_id": segment_id,
                        "target_text": f"TR {source_text}",
                        "research_needed": False,
                        "uncertainty_tags": [],
                    }
                ],
                "term_queries": [],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return canonical_bytes(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": response_text},
                    }
                ]
            }
        )


@dataclass
class JudgeTransport:
    calls: int = 0

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        self.calls += 1
        return canonical_bytes(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    '{"conclusion":"pass","findings":[],"unresolved_reasons":[]}'
                                ),
                            }
                        ],
                    }
                ],
            }
        )


@dataclass
class FailingJudgeTransport:
    calls: int = 0

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        self.calls += 1
        body = cast(dict[str, JsonValue], json.loads(request_body))
        user_input = body["input"]
        assert isinstance(user_input, list) and isinstance(user_input[1], dict)
        content = user_input[1]["content"]
        assert isinstance(content, list) and isinstance(content[0], dict)
        evidence = cast(dict[str, JsonValue], json.loads(cast(str, content[0]["text"])))
        target = evidence["evidence"]
        assert isinstance(target, dict)
        return canonical_bytes(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "conclusion": "fail",
                                        "findings": [
                                            {
                                                "category": "accuracy.mistranslation",
                                                "severity": "minor",
                                                "source_excerpt": target["source_excerpt"],
                                                "target_excerpt": target["target_excerpt"],
                                                "rationale": "synthetic actionable failure",
                                            }
                                        ],
                                        "unresolved_reasons": [],
                                    },
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            }
                        ],
                    }
                ],
            }
        )


def _episode(base: CanonicalEpisode, marker: str, namespace: str) -> CanonicalEpisode:
    payload = cast(dict[str, JsonValue], base.model_dump(mode="json", exclude_none=False))
    payload["episode_id"] = f"ep_{marker * 64}"
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    sources = provenance["sources"]
    assert isinstance(sources, list) and isinstance(sources[0], dict)
    sources[0]["dataset_namespace"] = namespace
    conversation = payload["conversation"]
    assert isinstance(conversation, list) and isinstance(conversation[0], dict)
    assert isinstance(conversation[-1], dict)
    conversation[0]["content"] = f"Need summary {marker}."
    conversation[-1]["content"] = f"Answer {marker}."
    return CanonicalEpisode.model_validate_json(canonical_bytes(payload), strict=True)


def _live_config() -> PipelineConfig:
    base = base_config(providers_enabled=True, egress_enabled=True)
    return PipelineConfig(
        schema_version=base.schema_version,
        canonical_schema_version=base.canonical_schema_version,
        diagnostic_catalog_version=base.diagnostic_catalog_version,
        normalizer_version=base.normalizer_version,
        max_record_bytes=base.max_record_bytes,
        jsonl_shard_rows=base.jsonl_shard_rows,
        providers=ProviderConfig(
            enabled=True,
            network_egress_enabled=True,
            translator=ProviderRole(
                provider="deepseek",
                model="deepseek-v4-flash",
                api_key_env="DEEPSEEK_API_KEY",
                endpoint="https://api.deepseek.com/chat/completions",
                temperature=0.0,
                thinking=False,
            ),
            strong_judge=ProviderRole(
                provider="openai",
                model="gpt-5.4",
                api_key_env="OPENAI_API_KEY",
                endpoint="https://api.openai.com/v1/responses",
            ),
            mini_verifier=ProviderRole(
                provider="openai",
                model="gpt-5.4-mini",
                api_key_env="OPENAI_API_KEY",
                endpoint="https://api.openai.com/v1/responses",
            ),
        ),
    )


def _prepare_candidates(tmp_path: Path) -> tuple[Path, list[CanonicalEpisode]]:
    base = canonical_fixture(ROOT / "tests" / "fixtures" / "no_tool", "no_tool", 2)
    episodes = [_episode(base, "1", "source-a"), _episode(base, "2", "source-b")]
    source_root = tmp_path / "source"
    source_root.mkdir()
    canonical_path = source_root / "canonical.jsonl"
    canonical_path.write_bytes(b"".join(canonical_bytes(episode) + b"\n" for episode in episodes))
    audit_path = source_root / "audit.json"
    audit_path.write_bytes(canonical_bytes(audit_exact_conflicts(episodes)) + b"\n")
    candidate_root = tmp_path / "candidate"
    prepare_automation_candidates(
        [canonical_path],
        [audit_path],
        candidate_root,
        field_policy=load_field_policy(ROOT / "configs" / "field_policy.toml"),
        requested_episode_count=2,
        max_translatable_segments=4,
    )
    candidate_path = next((candidate_root / "canonical").glob("*.jsonl"))
    return candidate_path, episodes


def test_automation_refuses_candidate_prompt_before_transport(tmp_path: Path) -> None:
    candidate_path, _ = _prepare_candidates(tmp_path)
    transport = TranslationTransport()
    candidate_prompt = load_prompt_bundle(ROOT / "configs" / "prompt_bundle.toml").model_copy(
        update={"promotion_status": "candidate"}
    )

    with pytest.raises(AutonomousPipelineError, match="promotion_status=validated"):
        run_automation_translation(
            candidate_path,
            tmp_path / "translation",
            config=_live_config(),
            field_policy=load_field_policy(ROOT / "configs" / "field_policy.toml"),
            prompt=candidate_prompt,
            transport=transport,
        )

    assert transport.calls == []


def test_automation_falls_back_without_stopping_and_builds_review_ready_hf_jsonl(
    tmp_path: Path,
) -> None:
    candidate_path, episodes = _prepare_candidates(tmp_path)
    policy = load_field_policy(ROOT / "configs" / "field_policy.toml")
    prompt = load_prompt_bundle(ROOT / "configs" / "prompt_bundle.toml")
    translation_transport = TranslationTransport(flash_invalid=True)
    translation_root = tmp_path / "translation"
    report = run_automation_translation(
        candidate_path,
        translation_root,
        config=_live_config(),
        field_policy=policy,
        prompt=prompt,
        transport=translation_transport,
        max_workers=2,
    )

    assert report.translated_records == 2
    assert report.needs_review_records == 0
    assert report.fallback_routes == 2
    assert translation_transport.calls.count("deepseek-v4-flash") == 2
    assert translation_transport.calls.count("deepseek-v4-pro") == 4
    results_path = next((translation_root / "translation-results").glob("*.jsonl"))
    results = read_automation_results(results_path)
    assert all(result.status == "translated" for result in results)
    assert all(len(result.routes) == 2 for result in results)
    assert {result.routes[0].failure_code for result in results} == {"provider_response_invalid"}

    evaluation_root = tmp_path / "evaluation-inputs"
    inputs_manifest = prepare_automation_evaluation_inputs(
        candidate_path,
        results_path,
        evaluation_root,
        field_policy=policy,
    )
    inputs_path = evaluation_root / inputs_manifest.artifacts[0].relative_path
    mini_transport = JudgeTransport()
    strong_transport = JudgeTransport()

    def judge_factory(role_name: str, transport: JudgeTransport):
        def factory(attempt_sink: ProviderAttemptSink) -> OpenAIResponsesJudge:
            return OpenAIResponsesJudge(
                config=_live_config(),
                role_name=role_name,
                transport=transport,
                attempt_sink=attempt_sink,
            )

        return factory

    mini = run_live_evaluation(
        inputs_path,
        tmp_path / "mini",
        config=_live_config(),
        role_name="mini_verifier",
        run_id="fixture-mini",
        judge_factory=judge_factory("mini_verifier", mini_transport),
    )
    strong = run_live_evaluation(
        inputs_path,
        tmp_path / "strong",
        config=_live_config(),
        role_name="strong_judge",
        run_id="fixture-strong",
        judge_factory=judge_factory("strong_judge", strong_transport),
    )
    mini_results = tmp_path / "mini" / "results" / mini.results_manifest.artifacts[0].relative_path
    strong_results = (
        tmp_path / "strong" / "results" / strong.results_manifest.artifacts[0].relative_path
    )
    consensus_root = tmp_path / "consensus"
    consensus = build_automation_consensus(mini_results, strong_results, consensus_root)
    assert consensus.accepted_units == 4
    consensus_path = next((consensus_root / "consensus").glob("*.jsonl"))

    package = build_huggingface_review_package(
        candidate_path,
        results_path,
        consensus_path,
        tmp_path / "hf-package",
        field_policy=policy,
    )
    train_path = tmp_path / "hf-package" / package.package_id / "data" / "train.jsonl"
    assert package.review_ready_records == len(episodes)
    assert package.status == "pending_human_approval"
    assert next(iter(iter_jsonl(train_path)))["quality_tier"] == "silver_candidate"  # type: ignore[index]
    assert "data/train.jsonl" in (
        tmp_path / "hf-package" / package.package_id / "README.md"
    ).read_text(encoding="utf-8")


def test_automation_does_not_resend_after_unknown_delivery_and_continues(tmp_path: Path) -> None:
    candidate_path, _ = _prepare_candidates(tmp_path)
    report = run_automation_translation(
        candidate_path,
        tmp_path / "translation",
        config=_live_config(),
        field_policy=load_field_policy(ROOT / "configs" / "field_policy.toml"),
        prompt=load_prompt_bundle(ROOT / "configs" / "prompt_bundle.toml"),
        transport=TranslationTransport(network_failure=True),
    )

    assert report.translated_records == 0
    assert report.needs_review_records == 2
    assert report.fallback_routes == 0
    results_path = next((tmp_path / "translation" / "translation-results").glob("*.jsonl"))
    results = read_automation_results(results_path)
    assert all(result.routes[0].failure_code == "network_delivery_unknown" for result in results)


def test_hierarchical_consensus_escalates_mini_non_passes_without_retranslation(
    tmp_path: Path,
) -> None:
    candidate_path, _ = _prepare_candidates(tmp_path)
    policy = load_field_policy(ROOT / "configs" / "field_policy.toml")
    prompt = load_prompt_bundle(ROOT / "configs" / "prompt_bundle.toml")
    translation_root = tmp_path / "translation"
    translation = run_automation_translation(
        candidate_path,
        translation_root,
        config=_live_config(),
        field_policy=policy,
        prompt=prompt,
        transport=TranslationTransport(),
    )
    translation_jsonl = next((translation_root / "translation-results").glob("*.jsonl"))
    inputs_root = tmp_path / "inputs"
    inputs = prepare_automation_evaluation_inputs(
        candidate_path,
        translation_jsonl,
        inputs_root,
        field_policy=policy,
    )
    inputs_jsonl = inputs_root / inputs.artifacts[0].relative_path

    def factory(role_name: str, transport: JudgeTransport | FailingJudgeTransport):
        def build(attempt_sink: ProviderAttemptSink) -> OpenAIResponsesJudge:
            return OpenAIResponsesJudge(
                config=_live_config(),
                role_name=role_name,
                transport=transport,
                attempt_sink=attempt_sink,
            )

        return build

    mini = run_live_evaluation(
        inputs_jsonl,
        tmp_path / "mini",
        config=_live_config(),
        role_name="mini_verifier",
        run_id="mini-fails",
        judge_factory=factory("mini_verifier", FailingJudgeTransport()),
    )
    mini_jsonl = tmp_path / "mini" / "results" / mini.results_manifest.artifacts[0].relative_path
    escalation = prepare_strong_escalation_inputs(
        inputs_jsonl,
        mini_jsonl,
        tmp_path / "strong-inputs",
        pass_sample_basis_points=0,
    )
    assert escalation is not None
    escalation_jsonl = tmp_path / "strong-inputs" / escalation.artifacts[0].relative_path
    assert escalation.artifacts[0].row_count == inputs.artifacts[0].row_count
    strong = run_live_evaluation(
        escalation_jsonl,
        tmp_path / "strong",
        config=_live_config(),
        role_name="strong_judge",
        run_id="strong-final",
        judge_factory=factory("strong_judge", JudgeTransport()),
    )
    strong_jsonl = (
        tmp_path / "strong" / "results" / strong.results_manifest.artifacts[0].relative_path
    )
    consensus = build_hierarchical_consensus(
        mini_jsonl,
        strong_jsonl,
        tmp_path / "consensus",
        pass_sample_basis_points=0,
    )
    assert translation.fallback_routes == 0
    assert consensus.strong_escalated_units == inputs.artifacts[0].row_count
    assert consensus.accepted_units == inputs.artifacts[0].row_count
    consensus_jsonl = next((tmp_path / "consensus" / "consensus").glob("*.jsonl"))
    package = build_huggingface_review_package(
        candidate_path,
        translation_jsonl,
        consensus_jsonl,
        tmp_path / "hf-package",
        field_policy=policy,
    )
    assert package.review_ready_records == 2


def test_hierarchical_consensus_accepts_unsampled_mini_passes(tmp_path: Path) -> None:
    candidate_path, _ = _prepare_candidates(tmp_path)
    policy = load_field_policy(ROOT / "configs" / "field_policy.toml")
    prompt = load_prompt_bundle(ROOT / "configs" / "prompt_bundle.toml")
    translation_root = tmp_path / "translation"
    translation = run_automation_translation(
        candidate_path,
        translation_root,
        config=_live_config(),
        field_policy=policy,
        prompt=prompt,
        transport=TranslationTransport(),
    )
    translation_jsonl = next((translation_root / "translation-results").glob("*.jsonl"))
    inputs = prepare_automation_evaluation_inputs(
        candidate_path,
        translation_jsonl,
        tmp_path / "inputs",
        field_policy=policy,
    )
    inputs_jsonl = tmp_path / "inputs" / inputs.artifacts[0].relative_path

    def factory(attempt_sink: ProviderAttemptSink) -> OpenAIResponsesJudge:
        return OpenAIResponsesJudge(
            config=_live_config(),
            role_name="mini_verifier",
            transport=JudgeTransport(),
            attempt_sink=attempt_sink,
        )

    mini = run_live_evaluation(
        inputs_jsonl,
        tmp_path / "mini",
        config=_live_config(),
        role_name="mini_verifier",
        run_id="mini-passes",
        judge_factory=factory,
    )
    mini_jsonl = tmp_path / "mini" / "results" / mini.results_manifest.artifacts[0].relative_path
    assert (
        prepare_strong_escalation_inputs(
            inputs_jsonl,
            mini_jsonl,
            tmp_path / "strong-inputs",
            pass_sample_basis_points=0,
        )
        is None
    )
    consensus = build_hierarchical_consensus(
        mini_jsonl,
        None,
        tmp_path / "consensus",
        pass_sample_basis_points=0,
    )
    assert translation.translated_records == 2
    assert consensus.strong_escalated_units == 0
    assert consensus.accepted_units == inputs.artifacts[0].row_count


def test_hf_package_requires_consensus_for_every_translated_leaf(tmp_path: Path) -> None:
    candidate_path, _ = _prepare_candidates(tmp_path)
    policy = load_field_policy(ROOT / "configs" / "field_policy.toml")
    prompt = load_prompt_bundle(ROOT / "configs" / "prompt_bundle.toml")
    translation_root = tmp_path / "translation"
    run_automation_translation(
        candidate_path,
        translation_root,
        config=_live_config(),
        field_policy=policy,
        prompt=prompt,
        transport=TranslationTransport(),
    )
    results_path = next((translation_root / "translation-results").glob("*.jsonl"))
    empty_root = tmp_path / "inputs"
    empty_root.mkdir()
    with_path = empty_root / "empty-consensus.jsonl"
    with_path.write_bytes(b"")
    package = build_huggingface_review_package(
        candidate_path,
        results_path,
        with_path,
        tmp_path / "hf-package",
        field_policy=policy,
    )
    assert package.review_ready_records == 0
    assert package.needs_review_records == 2
