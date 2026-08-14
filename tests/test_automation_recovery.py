from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from tests.helpers import canonical_fixture
from tests.test_provider_adapter import config as base_config
from toolcall_tr.audit import audit_exact_conflicts
from toolcall_tr.automation_recovery import inspect_automation_recovery, run_automation_recovery
from toolcall_tr.autonomous_pipeline import (
    prepare_automation_candidates,
    prepare_automation_evaluation_inputs,
    prepare_strong_escalation_inputs,
    run_automation_translation,
)
from toolcall_tr.config import PipelineConfig, ProviderConfig, ProviderRole
from toolcall_tr.field_policy import load_field_policy
from toolcall_tr.hashing import JsonValue, canonical_bytes, sha256_bytes
from toolcall_tr.live_evaluation import JudgeFactory, run_live_evaluation
from toolcall_tr.models import CanonicalEpisode
from toolcall_tr.openai_judge import OpenAIResponsesJudge
from toolcall_tr.prompt_contract import load_prompt_bundle
from toolcall_tr.provider_provenance import ProviderAttemptSink
from toolcall_tr.secure_transport import TransportHttpError

ROOT = Path(__file__).resolve().parents[1]


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


def _episode(base: CanonicalEpisode, marker: str) -> CanonicalEpisode:
    payload = cast(dict[str, JsonValue], base.model_dump(mode="json", exclude_none=False))
    payload["episode_id"] = f"ep_{marker * 64}"
    provenance = cast(dict[str, JsonValue], payload["provenance"])
    sources = cast(list[JsonValue], provenance["sources"])
    source = cast(dict[str, JsonValue], sources[0])
    source["dataset_namespace"] = f"source-{marker}"
    conversation = cast(list[JsonValue], payload["conversation"])
    first = cast(dict[str, JsonValue], conversation[0])
    last = cast(dict[str, JsonValue], conversation[-1])
    first["content"] = f"Request {marker}."
    last["content"] = f"Answer {marker}."
    return CanonicalEpisode.model_validate_json(canonical_bytes(payload), strict=True)


@dataclass
class TranslationTransport:
    fail_marker: str | None = None
    calls: list[str] = field(default_factory=lambda: [])

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        body = cast(dict[str, JsonValue], json.loads(request_body))
        messages = cast(list[JsonValue], body["messages"])
        user = cast(dict[str, JsonValue], messages[1])
        request = cast(dict[str, JsonValue], json.loads(cast(str, user["content"])))
        segment = cast(dict[str, JsonValue], cast(list[JsonValue], request["segments"])[0])
        source = cast(str, segment["source_text"])
        self.calls.append(source)
        if self.fail_marker is not None and self.fail_marker in source:
            raise TransportHttpError(402)
        response = {
            "schema_version": "translation-response-0.1.0",
            "request_id": request["request_id"],
            "status": "translated",
            "segments": [
                {
                    "segment_id": segment["segment_id"],
                    "target_text": f"TR {source}",
                    "research_needed": False,
                    "uncertainty_tags": [],
                }
            ],
            "term_queries": [],
        }
        return canonical_bytes(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                response, ensure_ascii=False, separators=(",", ":")
                            )
                        },
                    }
                ]
            }
        )


@dataclass
class JudgeTransport:
    failures_remaining: int = 0
    calls: int = 0

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        self.calls += 1
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise TransportHttpError(402)
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
                                    '{"conclusion":"pass","findings":[],'
                                    '"unresolved_reasons":[]}'
                                ),
                            }
                        ],
                    }
                ],
            }
        )


def _judge_factory(
    role_name: Literal["mini_verifier", "strong_judge"], transport: JudgeTransport
) -> JudgeFactory:
    def factory(attempt_sink: ProviderAttemptSink) -> OpenAIResponsesJudge:
        return OpenAIResponsesJudge(
            config=_live_config(),
            role_name=role_name,
            transport=transport,
            attempt_sink=attempt_sink,
        )

    return factory


def _build_parent_run(
    tmp_path: Path,
) -> tuple[Path, TranslationTransport, JudgeTransport, JudgeTransport]:
    base = canonical_fixture(ROOT / "tests" / "fixtures" / "no_tool", "no_tool", 2)
    episodes = [_episode(base, "1"), _episode(base, "2")]
    source = tmp_path / "source"
    source.mkdir()
    canonical = source / "canonical.jsonl"
    canonical.write_bytes(b"".join(canonical_bytes(item) + b"\n" for item in episodes))
    audit = source / "audit.json"
    audit.write_bytes(canonical_bytes(audit_exact_conflicts(episodes)) + b"\n")
    parent = tmp_path / "parent"
    policy = load_field_policy(ROOT / "configs" / "field_policy.toml")
    prompt = load_prompt_bundle(ROOT / "configs" / "prompt_bundle.toml")
    candidate = prepare_automation_candidates(
        [canonical],
        [audit],
        parent / "candidate",
        field_policy=policy,
        requested_episode_count=2,
        max_translatable_segments=4,
    )
    candidate_jsonl = next((parent / "candidate" / "canonical").glob("*.jsonl"))
    parent_translation_transport = TranslationTransport(fail_marker="Answer 2.")
    translation = run_automation_translation(
        candidate_jsonl,
        parent / "translation",
        config=_live_config(),
        field_policy=policy,
        prompt=prompt,
        transport=parent_translation_transport,
    )
    translation_jsonl = (
        parent
        / "translation"
        / "translation-results"
        / next((parent / "translation" / "translation-results").glob("*.jsonl")).name
    )
    inputs = prepare_automation_evaluation_inputs(
        candidate_jsonl,
        translation_jsonl,
        parent / "evaluation-inputs",
        field_policy=policy,
    )
    inputs_jsonl = parent / "evaluation-inputs" / inputs.artifacts[0].relative_path
    parent_mini_transport = JudgeTransport(failures_remaining=1)
    mini = run_live_evaluation(
        inputs_jsonl,
        parent / "mini-judge",
        config=_live_config(),
        role_name="mini_verifier",
        run_id=f"{candidate.candidate_id}-mini",
        judge_factory=_judge_factory("mini_verifier", parent_mini_transport),
    )
    mini_jsonl = (
        parent / "mini-judge" / "results" / mini.results_manifest.artifacts[0].relative_path
    )
    escalation = prepare_strong_escalation_inputs(
        inputs_jsonl,
        mini_jsonl,
        parent / "strong-escalation-inputs",
        pass_sample_basis_points=10_000,
    )
    assert escalation is not None
    escalation_jsonl = parent / "strong-escalation-inputs" / escalation.artifacts[0].relative_path
    parent_strong_transport = JudgeTransport(failures_remaining=99)
    run_live_evaluation(
        escalation_jsonl,
        parent / "strong-judge",
        config=_live_config(),
        role_name="strong_judge",
        run_id=f"{candidate.candidate_id}-strong",
        judge_factory=_judge_factory("strong_judge", parent_strong_transport),
    )
    assert translation.translated_records == 1
    return parent, parent_translation_transport, parent_mini_transport, parent_strong_transport


def test_recovery_retries_only_explicit_payment_failures_in_a_new_overlay(tmp_path: Path) -> None:
    parent, parent_translation, parent_mini, parent_strong = _build_parent_run(tmp_path)
    policy = load_field_policy(ROOT / "configs" / "field_policy.toml")
    prompt = load_prompt_bundle(ROOT / "configs" / "prompt_bundle.toml")
    parent_candidate = next((parent / "candidate" / "canonical").glob("*.jsonl"))
    parent_hash = sha256_bytes(parent_candidate.read_bytes())
    recovery_translation = TranslationTransport()
    recovery_mini = JudgeTransport()
    recovery_strong = JudgeTransport()

    report = run_automation_recovery(
        parent,
        tmp_path / "recovery",
        config=_live_config(),
        field_policy=policy,
        prompt=prompt,
        translation_transport=recovery_translation,
        mini_judge_factory=_judge_factory("mini_verifier", recovery_mini),
        strong_judge_factory=_judge_factory("strong_judge", recovery_strong),
        retry_http_statuses=[402],
        strong_pass_sample_basis_points=10_000,
    )

    assert sha256_bytes(parent_candidate.read_bytes()) == parent_hash
    assert report.retried_translation_episodes == 1
    assert report.recovered_translation_episodes == 1
    assert report.retried_mini_units == 3
    assert report.recovered_mini_units == 3
    assert report.retried_strong_units == 4
    assert report.recovered_strong_units == 4
    assert parent_translation.calls
    assert parent_mini.calls == 2
    assert parent_strong.calls == 2
    assert recovery_translation.calls
    assert recovery_mini.calls == 3
    assert recovery_strong.calls == 4
    train = (
        tmp_path
        / "recovery"
        / "hf-review-package"
        / report.hf_package_id
        / "data"
        / "train.jsonl"
    )
    assert train.is_file()
    assert len(train.read_text(encoding="utf-8").splitlines()) == 2


def test_recovery_plan_is_read_only_and_counts_only_selected_statuses(tmp_path: Path) -> None:
    parent, _, _, _ = _build_parent_run(tmp_path)
    parent_candidate = next((parent / "candidate" / "canonical").glob("*.jsonl"))
    parent_hash = sha256_bytes(parent_candidate.read_bytes())

    plan = inspect_automation_recovery(parent, retry_http_statuses=[402])

    assert plan.provider_egress_allowed is False
    assert plan.translation_retry_episodes == 1
    assert plan.mini_retry_units == 1
    assert plan.strong_retry_units == 2
    assert sha256_bytes(parent_candidate.read_bytes()) == parent_hash
