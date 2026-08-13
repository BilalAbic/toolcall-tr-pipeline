"""Injected-transport tests for the explicit live evaluation operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from tests.test_provider_adapter import config as base_config
from toolcall_tr.cli import app
from toolcall_tr.config import PipelineConfig, ProviderConfig, ProviderRole
from toolcall_tr.eval_contract import SegmentPathEvidence, build_evaluation_unit
from toolcall_tr.hashing import canonical_bytes, sha256_bytes
from toolcall_tr.jsonio import iter_jsonl, write_jsonl
from toolcall_tr.live_evaluation import (
    JudgeFactory,
    LiveEvaluationConfigurationError,
    LiveEvaluationInput,
    build_live_evaluation_input,
    run_live_evaluation,
)
from toolcall_tr.openai_judge import OpenAIResponsesJudge
from toolcall_tr.provider_provenance import ProviderAttemptOutcome, ProviderAttemptSink
from toolcall_tr.secure_transport import TransportHttpError


@dataclass
class QueueTransport:
    responses: list[bytes | Exception]
    calls: list[tuple[str, bytes]] = field(default_factory=lambda: [])

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        self.calls.append((endpoint, request_body))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _pass_envelope() -> bytes:
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


def _openai_config() -> PipelineConfig:
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
            translator=base.providers.translator,
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


def _input(character: str, *, source: str, target: str) -> LiveEvaluationInput:
    unit = build_evaluation_unit(
        episode_id=f"ep_{character * 64}",
        segment_id=f"seg_{character * 64}",
        path="/conversation/0/content",
        source_text_sha256=sha256_bytes(source.encode("utf-8")),
        target_text_sha256=sha256_bytes(target.encode("utf-8")),
    )
    return build_live_evaluation_input(
        evaluation_unit=unit,
        evidence=SegmentPathEvidence(
            segment_id=unit.segment_id,
            path=unit.path,
            source_excerpt=source,
            target_excerpt=target,
        ),
    )


def _write_input(root: Path, inputs: list[LiveEvaluationInput]) -> Path:
    source_root = root / "input"
    source_root.mkdir()
    input_jsonl = source_root / "evaluation.jsonl"
    write_jsonl(input_jsonl, [item.model_dump(mode="json") for item in inputs])
    return input_jsonl


def _factory(config: PipelineConfig, transport: QueueTransport) -> JudgeFactory:
    def factory(attempt_sink: ProviderAttemptSink) -> OpenAIResponsesJudge:
        return OpenAIResponsesJudge(
            config=config,
            role_name="mini_verifier",
            transport=transport,
            attempt_sink=attempt_sink,
        )

    return factory


def test_live_evaluation_publishes_immutable_pass_receipts_without_gold_or_raw_attempt_data(
    tmp_path: Path,
) -> None:
    source = "Keep the protected marker unchanged."
    target = "Korunan işaretçiyi değiştirmeyin."
    input_jsonl = _write_input(tmp_path, [_input("1", source=source, target=target)])
    before = input_jsonl.read_bytes()
    transport = QueueTransport([_pass_envelope()])

    artifacts = run_live_evaluation(
        input_jsonl,
        tmp_path / "output",
        config=_openai_config(),
        role_name="mini_verifier",
        run_id="fixture-live-evaluation",
        judge_factory=_factory(_openai_config(), transport),
    )

    assert input_jsonl.read_bytes() == before
    assert len(transport.calls) == 1
    assert artifacts.report.input_rows == 1
    assert artifacts.report.succeeded_rows == 1
    assert artifacts.report.failed_rows == 0
    assert artifacts.report.gold_release_allowed is False
    report_path = tmp_path / "output" / "runs" / f"{artifacts.report.report_id}.json"
    assert report_path.is_file()
    result_path = (
        tmp_path
        / "output"
        / "results"
        / artifacts.results_manifest.artifacts[0].relative_path
    )
    attempt_path = (
        tmp_path
        / "output"
        / "attempts"
        / artifacts.attempts_manifest.artifacts[0].relative_path
    )
    result = next(iter_jsonl(result_path))
    assert result["gold_eligible"] is False  # type: ignore[index]
    assert source not in result_path.read_text(encoding="utf-8")
    assert target not in result_path.read_text(encoding="utf-8")
    assert source not in attempt_path.read_text(encoding="utf-8")
    assert target not in attempt_path.read_text(encoding="utf-8")


def test_live_evaluation_records_terminal_failures_and_continues_other_rows(tmp_path: Path) -> None:
    first = _input("2", source="One source leaf.", target="Bir hedef yaprak.")
    second = _input("3", source="Two source leaf.", target="İki hedef yaprak.")
    input_jsonl = _write_input(tmp_path, [first, second])
    transport = QueueTransport([_pass_envelope(), TransportHttpError(429)])

    artifacts = run_live_evaluation(
        input_jsonl,
        tmp_path / "output",
        config=_openai_config(),
        role_name="mini_verifier",
        run_id="fixture-live-evaluation-failure",
        judge_factory=_factory(_openai_config(), transport),
    )

    assert len(transport.calls) == 2
    assert artifacts.report.succeeded_rows == 1
    assert artifacts.report.failed_rows == 1
    attempt_path = (
        tmp_path
        / "output"
        / "attempts"
        / artifacts.attempts_manifest.artifacts[0].relative_path
    )
    attempts = list(iter_jsonl(attempt_path))
    assert {row["outcome"] for row in attempts} == {"succeeded", "failed"}  # type: ignore[index]
    assert any(row["failure_code"] == "http_transient" for row in attempts)  # type: ignore[index]


def test_live_evaluation_preflight_block_is_published_without_delivery_or_sensitive_text(
    tmp_path: Path,
) -> None:
    sensitive_source = "Contact person@example.com before proceeding."
    input_jsonl = _write_input(
        tmp_path,
        [_input("4", source=sensitive_source, target="Devam etmeden önce irtibat kurun.")],
    )
    transport = QueueTransport([_pass_envelope()])

    artifacts = run_live_evaluation(
        input_jsonl,
        tmp_path / "output",
        config=_openai_config(),
        role_name="mini_verifier",
        run_id="fixture-live-evaluation-preflight",
        judge_factory=_factory(_openai_config(), transport),
    )

    assert transport.calls == []
    assert artifacts.report.succeeded_rows == 0
    assert artifacts.report.failed_rows == 1
    attempt_path = (
        tmp_path
        / "output"
        / "attempts"
        / artifacts.attempts_manifest.artifacts[0].relative_path
    )
    contents = attempt_path.read_text(encoding="utf-8")
    assert sensitive_source not in contents
    attempt = next(iter_jsonl(attempt_path))
    assert attempt["outcome"] == ProviderAttemptOutcome.PREFLIGHT_BLOCKED.value  # type: ignore[index]
    assert attempt["failure_code"] == "preflight_blocked"  # type: ignore[index]


def test_live_evaluation_rejects_invalid_pairs_before_any_transport_or_output(
    tmp_path: Path,
) -> None:
    item = _input("5", source="Original source.", target="Özgün hedef.")
    invalid = item.model_dump(mode="json")
    invalid["evidence"]["source_excerpt"] = "Different source."
    source_root = tmp_path / "input"
    source_root.mkdir()
    input_jsonl = source_root / "invalid.jsonl"
    write_jsonl(input_jsonl, [invalid])
    transport = QueueTransport([_pass_envelope()])

    with pytest.raises(ValidationError, match="source evidence"):
        run_live_evaluation(
            input_jsonl,
            tmp_path / "output",
            config=_openai_config(),
            role_name="mini_verifier",
            run_id="fixture-invalid-input",
            judge_factory=_factory(_openai_config(), transport),
        )

    assert transport.calls == []
    assert not (tmp_path / "output").exists()


def test_live_evaluation_refuses_output_inside_the_input_root(tmp_path: Path) -> None:
    input_jsonl = _write_input(
        tmp_path,
        [_input("6", source="Source leaf.", target="Hedef yaprak.")],
    )

    with pytest.raises(LiveEvaluationConfigurationError, match="disjoint"):
        run_live_evaluation(
            input_jsonl,
            input_jsonl.parent / "derived",
            config=_openai_config(),
            role_name="mini_verifier",
            run_id="fixture-overlap",
            judge_factory=_factory(_openai_config(), QueueTransport([_pass_envelope()])),
        )


def test_live_evaluation_cli_requires_the_explicit_live_switch_before_reading_config(
    tmp_path: Path,
) -> None:
    input_jsonl = _write_input(
        tmp_path,
        [_input("7", source="Source leaf.", target="Hedef yaprak.")],
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluation",
            "run",
            str(input_jsonl),
            "--output",
            str(tmp_path / "output"),
            "--config",
            str(tmp_path / "missing-live.toml"),
            "--role",
            "mini_verifier",
            "--run-id",
            "fixture-cli",
        ],
    )

    assert result.exit_code == 2
    assert "requires --live" in result.stdout
    assert not (tmp_path / "output").exists()
