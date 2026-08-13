from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.helpers import canonical_fixture
from tests.test_deepseek_adapter import live_deepseek_config
from tests.test_provider_adapter import prompt
from toolcall_tr.field_policy import FieldAction, FieldPolicy
from toolcall_tr.hashing import canonical_bytes
from toolcall_tr.operational_translation import (
    OperationalTranslationError,
    run_operational_translation,
)
from toolcall_tr.translation_contract import (
    SegmentTranslation,
    TranslationRequest,
    TranslationResponse,
)


def _calls() -> list[tuple[str, bytes]]:
    return []


@dataclass
class TranslatingTransport:
    calls: list[tuple[str, bytes]] = field(default_factory=_calls)

    def create_response(self, *, endpoint: str, request_body: bytes) -> bytes:
        self.calls.append((endpoint, request_body))
        body = json.loads(request_body)
        request = TranslationRequest.model_validate_json(
            body["messages"][1]["content"], strict=True
        )
        source = request.segments[0]
        response = TranslationResponse(
            request_id=request.request_id,
            status="translated",
            segments=[
                SegmentTranslation(
                    segment_id=source.segment_id,
                    target_text=f"TR: {source.source_text}",
                    research_needed=False,
                    uncertainty_tags=[],
                )
            ],
            term_queries=[],
        )
        return canonical_bytes(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": response.model_dump_json()},
                    }
                ]
            }
        )


def _policy() -> FieldPolicy:
    return FieldPolicy(
        policy_version="test-batch-policy-0.1.0",
        tool_description_action=FieldAction.COPY_EXACT,
        parameter_description_action=FieldAction.COPY_EXACT,
        argument_policies=[],
    )


def _input(tmp_path: Path, fixture_root: Path) -> Path:
    episode = canonical_fixture(fixture_root / "no_tool", "no_tool")
    input_path = tmp_path / "input" / "canonical.jsonl"
    input_path.parent.mkdir()
    input_path.write_bytes(canonical_bytes(episode) + b"\n")
    return input_path


def test_batch_translates_only_leaf_segments_and_resumes_without_second_send(
    tmp_path: Path, fixture_root: Path
) -> None:
    input_path = _input(tmp_path, fixture_root)
    before = input_path.read_bytes()
    output = tmp_path / "derived"
    transport = TranslatingTransport()

    first = run_operational_translation(
        input_path,
        output,
        config=live_deepseek_config(),
        field_policy=_policy(),
        prompt=prompt(),
        transport=transport,
    )
    second_transport = TranslatingTransport()
    second = run_operational_translation(
        input_path,
        output,
        config=live_deepseek_config(),
        field_policy=_policy(),
        prompt=prompt(),
        transport=second_transport,
    )

    assert first == second
    assert first.translated_records == 1
    assert first.provider_attempts == 2
    assert len(transport.calls) == 2
    assert second_transport.calls == []
    assert input_path.read_bytes() == before
    attempt_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (output / "provider-attempts").glob("*.json")
    )
    assert "Book me a flight." not in attempt_text
    assert "Which destination" not in attempt_text
    assert (output / "translation-results" / f"{first.result_manifest_id}.json").is_file()


def test_batch_refuses_output_inside_input_tree_before_transport(
    tmp_path: Path, fixture_root: Path
) -> None:
    input_path = _input(tmp_path, fixture_root)
    transport = TranslatingTransport()

    with pytest.raises(OperationalTranslationError, match="disjoint"):
        run_operational_translation(
            input_path,
            input_path.parent / "derived",
            config=live_deepseek_config(),
            field_policy=_policy(),
            prompt=prompt(),
            transport=transport,
        )

    assert transport.calls == []


def test_batch_refuses_candidate_prompt_before_transport(
    tmp_path: Path, fixture_root: Path
) -> None:
    input_path = _input(tmp_path, fixture_root)
    transport = TranslatingTransport()
    candidate_prompt = prompt().model_copy(update={"promotion_status": "candidate"})

    with pytest.raises(OperationalTranslationError, match="promotion_status=validated"):
        run_operational_translation(
            input_path,
            tmp_path / "derived",
            config=live_deepseek_config(),
            field_policy=_policy(),
            prompt=candidate_prompt,
            transport=transport,
        )

    assert transport.calls == []
