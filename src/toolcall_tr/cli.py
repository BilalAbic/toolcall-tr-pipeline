"""Human-readable CLI over deterministic JSON/JSONL artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from toolcall_tr.adapters import get_adapter
from toolcall_tr.adjudication import ConflictAdjudication, ConflictAdjudicationLog
from toolcall_tr.artifacts import publish_bytes_atomic, publish_jsonl_artifact
from toolcall_tr.audit import ExactConflictAudit, audit_exact_conflicts
from toolcall_tr.canonicalize import canonicalize as canonicalize_record
from toolcall_tr.config import PipelineConfig, inspect_config, load_config, load_live_config
from toolcall_tr.credentials import AllowListedSecretResolver, CredentialResolutionError
from toolcall_tr.deepseek_adapter import DeepSeekTranslationAdapter
from toolcall_tr.diagnostics import CATALOG
from toolcall_tr.eval_contract import (
    GoldAcceptance,
    HumanEvaluationReview,
    ModelEvaluationVerdict,
    SegmentPathEvidence,
    build_evaluation_unit,
    decide_gold_acceptance,
)
from toolcall_tr.events import EventLog
from toolcall_tr.field_policy import load_field_policy
from toolcall_tr.hashing import JsonValue, canonical_bytes, sha256_bytes
from toolcall_tr.human_review_log import HumanEvaluationReviewLog
from toolcall_tr.jsonio import StrictJsonError, iter_jsonl, loads_strict_bytes
from toolcall_tr.live_evaluation import (
    LiveEvaluationConfigurationError,
    LiveEvaluationRuntimeError,
    run_live_evaluation,
)
from toolcall_tr.models import CanonicalEpisode, RawToolDefinition
from toolcall_tr.openai_judge import OpenAIResponsesJudge
from toolcall_tr.operational_translation import (
    OperationalTranslationError,
    run_operational_translation,
)
from toolcall_tr.phase4_config import load_phase4_config
from toolcall_tr.pilot import (
    CanonicalQuarantineRecord,
    PilotConfigurationError,
    run_operational_pilot,
)
from toolcall_tr.prompt_contract import load_prompt_bundle
from toolcall_tr.provider_adapter import ProviderAdapterError
from toolcall_tr.provider_provenance import ProviderAttemptSink
from toolcall_tr.release_contract import (
    ReleaseGoldMember,
    build_release_manifest,
    read_release_manifest,
    validate_release_manifest,
    write_release_manifest,
)
from toolcall_tr.review_queue import ReviewQueueInputError, build_review_tasks
from toolcall_tr.secure_transport import SecureTransportError, StdlibJsonTransport
from toolcall_tr.selection import SelectionCandidate, freeze_s400
from toolcall_tr.similarity import SimilarityDocument, retrieve_near_duplicate_candidates
from toolcall_tr.source import BronzeRecord, SourceSnapshot, ingest_snapshot, register_source
from toolcall_tr.source_array import SourceArrayConversionError, convert_json_array_to_jsonl
from toolcall_tr.source_evidence import (
    SourceEvidenceRequest,
    build_source_evidence,
)
from toolcall_tr.tool_registry import ToolRegistry
from toolcall_tr.translation_contract import (
    ProtectedToken,
    TranslationSegment,
    build_translation_request,
)

app = typer.Typer(help="Deterministic, audit-first Toolcall TR pipeline.", no_args_is_help=True)
source_app = typer.Typer(help="Register and validate immutable source snapshots.")
registry_app = typer.Typer(help="Build and inspect the semantic tool registry.")
events_app = typer.Typer(help="Inspect append-only run event chains.")
audit_app = typer.Typer(help="Deterministic duplicate/conflict audit commands.")
select_app = typer.Typer(help="Deterministic, human-gated selection manifest commands.")
review_app = typer.Typer(help="Append reviewer-authored decisions to strict local event logs.")
release_app = typer.Typer(help="Build and validate human-gated Gold release manifests.")
memory_app = typer.Typer(help="Translation-memory commands (future phase).")
index_app = typer.Typer(help="Derived index commands.")
provider_app = typer.Typer(help="Safe provider configuration inspection commands.")
pilot_app = typer.Typer(help="Fail-closed, provider-free operational pilot commands.")
evaluation_app = typer.Typer(help="Explicit, human-gated live OpenAI evaluation commands.")

app.add_typer(source_app, name="source")
app.add_typer(registry_app, name="registry")
app.add_typer(events_app, name="events")
app.add_typer(audit_app, name="audit")
app.add_typer(select_app, name="select")
app.add_typer(review_app, name="review")
app.add_typer(release_app, name="release")
app.add_typer(memory_app, name="memory")
app.add_typer(index_app, name="index")
app.add_typer(provider_app, name="provider")
app.add_typer(pilot_app, name="pilot")
app.add_typer(evaluation_app, name="evaluation")
console = Console()

_DEFAULT_PIPELINE_CONFIG = Path("configs/pipeline.toml")
_LOCAL_ENV_FILE = Path(".env")


def _read_snapshot(path: Path) -> SourceSnapshot:
    return SourceSnapshot.model_validate_json(path.read_text(encoding="utf-8"), strict=True)


def _canonical_episode_from_json(record: JsonValue) -> CanonicalEpisode:
    """Validate a JSONL row at the JSON boundary while retaining strict semantics."""
    return CanonicalEpisode.model_validate_json(canonical_bytes(record), strict=True)


def _read_exactly_one_jsonl_object(path: Path) -> dict[str, JsonValue]:
    """Read one strict JSON object, rejecting empty or multi-decision inputs."""
    records = iter_jsonl(path)
    try:
        first = next(records)
    except StopIteration as exc:
        raise ValueError("review decision JSONL must contain exactly one record") from exc
    try:
        next(records)
    except StopIteration:
        pass
    else:
        raise ValueError("review decision JSONL must contain exactly one record")
    if not isinstance(first, dict):
        raise ValueError("review decision JSONL record must be an object")
    return first


def _read_strict_json_object(path: Path, *, label: str) -> dict[str, JsonValue]:
    """Read one JSON object with the same fail-closed parser used for JSONL."""
    value = loads_strict_bytes(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_model_verdicts(path: Path) -> list[ModelEvaluationVerdict]:
    """Parse release evidence at the strict JSON boundary; never infer a verdict."""
    verdicts = [
        ModelEvaluationVerdict.model_validate(record, strict=True) for record in iter_jsonl(path)
    ]
    verdict_ids = [verdict.verdict_id for verdict in verdicts]
    if len(verdict_ids) != len(set(verdict_ids)):
        raise ValueError("model verdict JSONL contains duplicate verdict IDs")
    return verdicts


def _read_gold_members(path: Path) -> list[ReleaseGoldMember]:
    """Parse explicit Gold membership declarations without selecting any records."""
    members = [ReleaseGoldMember.model_validate(record, strict=True) for record in iter_jsonl(path)]
    episode_ids = [member.episode_id for member in members]
    if episode_ids != sorted(episode_ids) or len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Gold member JSONL must be unique and sorted by episode ID")
    return members


def _gold_acceptances_from_reviews(
    verdicts: list[ModelEvaluationVerdict], reviews: list[HumanEvaluationReview]
) -> list[GoldAcceptance]:
    """Compute policy receipts from logged human decisions, never from model output alone."""
    review_by_verdict_id = {review.verdict_id: review for review in reviews}
    return [
        decide_gold_acceptance(
            model_verdict=verdict,
            human_review=review_by_verdict_id.get(verdict.verdict_id),
        )
        for verdict in verdicts
    ]


def _is_default_pipeline_config(path: Path) -> bool:
    """Compare paths without treating a spelling change as a distinct config."""
    return path.resolve() == _DEFAULT_PIPELINE_CONFIG.resolve()


def _load_live_smoke_config(path: Path) -> PipelineConfig:
    """Parse a config for inspection without enabling any execution path.

    The standard loader intentionally rejects enabled egress gates.  This local
    parser exists only so ``provider smoke --live`` can report which explicit
    prerequisites are missing.  Its return value is never passed to a provider
    adapter or transport by this CLI command.
    """
    return inspect_config(path)


def _live_smoke_failures(
    config: PipelineConfig, *, role_name: str, env_file_exists: bool
) -> list[str]:
    """Return structural prerequisites only; do not read credentials or contact providers."""
    failures: list[str] = []
    if not config.providers.enabled:
        failures.append("providers.enabled must be true")
    if not config.providers.network_egress_enabled:
        failures.append("providers.network_egress_enabled must be true")
    role = getattr(config.providers, role_name)
    if role.endpoint is None:
        failures.append(f"providers.{role_name}.endpoint must be configured")
    if not env_file_exists:
        failures.append(".env file is missing")
    return failures


def _synthetic_smoke_translation() -> tuple[TranslationSegment, list[ProtectedToken]]:
    """Return non-sensitive fixed text; this must never read a source dataset."""
    tokens = [ProtectedToken(token="⟪S1_P1⟫", occurrence=1)]
    return (
        TranslationSegment(
            segment_id="seg_" + "1" * 64,
            path="/synthetic/content",
            source_text="Please keep ⟪S1_P1⟫ exactly unchanged.",
            protected_tokens=tokens,
        ),
        tokens,
    )


def _send_synthetic_deepseek_smoke(config: PipelineConfig) -> None:
    """Run exactly one fixed synthetic request through the live DeepSeek adapter."""
    role = config.providers.translator
    if role.provider != "deepseek":
        raise ValueError("synthetic smoke currently supports the DeepSeek translator only")
    segment, _ = _synthetic_smoke_translation()
    request = build_translation_request(
        episode_id="ep_" + "2" * 64,
        input_variant_id="sha256:" + "3" * 64,
        field_policy_version="synthetic-smoke-0.1.0",
        segments=[segment],
    )
    resolver = AllowListedSecretResolver(
        allowed_names=frozenset({role.api_key_env}), env_file=_LOCAL_ENV_FILE
    )
    transport = StdlibJsonTransport(
        credential_name=role.api_key_env,
        secret_lookup=resolver.resolve,
    )
    response = DeepSeekTranslationAdapter(
        config=config,
        transport=transport,
        max_output_tokens=256,
    ).translate(
        request=request,
        prompt=load_prompt_bundle(Path("configs/prompt_bundle.toml")),
    )
    if response.status not in {"translated", "research_needed"}:
        raise ValueError("synthetic smoke returned an unsupported translation status")
    console.print(
        "[green]synthetic provider smoke accepted[/green] "
        f"provider={role.provider} model={role.model} segments={len(response.segments)}"
    )


def _send_synthetic_openai_judge_smoke(config: PipelineConfig, *, role_name: str) -> None:
    """Run one fixed model-triage request without a source record or Gold transition."""
    role = getattr(config.providers, role_name)
    source_text = "Keep ⟪S1_P1⟫ unchanged."
    target_text = "⟪S1_P1⟫ öğesini değiştirmeyin."
    unit = build_evaluation_unit(
        episode_id="ep_" + "4" * 64,
        segment_id="seg_" + "5" * 64,
        path="/synthetic/content",
        source_text_sha256=sha256_bytes(source_text.encode("utf-8")),
        target_text_sha256=sha256_bytes(target_text.encode("utf-8")),
    )
    evidence = SegmentPathEvidence(
        segment_id=unit.segment_id,
        path=unit.path,
        source_excerpt=source_text,
        target_excerpt=target_text,
    )
    resolver = AllowListedSecretResolver(
        allowed_names=frozenset({role.api_key_env}), env_file=_LOCAL_ENV_FILE
    )
    transport = StdlibJsonTransport(
        credential_name=role.api_key_env,
        secret_lookup=resolver.resolve,
    )
    verdict = OpenAIResponsesJudge(
        config=config,
        role_name=role_name,
        transport=transport,
        max_output_tokens=256,
    ).judge(evaluation_unit=unit, evidence=evidence)
    console.print(
        "[green]synthetic judge smoke accepted[/green] "
        f"role={role_name} model={role.model} conclusion={verdict.conclusion} "
        f"findings={len(verdict.findings)} gold_eligible=false"
    )


@source_app.command("register")
def source_register(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False, resolve_path=True)],
    dataset_namespace: Annotated[str, typer.Option("--dataset")],
    source_revision: Annotated[str, typer.Option("--revision")],
    license_id: Annotated[str, typer.Option("--license")],
    output: Annotated[Path, typer.Option("--output")] = Path("manifests"),
) -> None:
    """Hash and freeze JSONL files without modifying the source tree."""
    snapshot = register_source(
        root,
        dataset_namespace=dataset_namespace,
        source_revision=source_revision,
        license_id=license_id,
    )
    target = output / f"{snapshot.snapshot_id}.json"
    publish_bytes_atomic(target, canonical_bytes(snapshot) + b"\n")
    console.print(target)


@source_app.command("validate")
def source_validate(
    snapshot_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    snapshot = _read_snapshot(snapshot_manifest)
    console.print(f"[green]valid[/green] {snapshot.snapshot_id} rows={snapshot.total_record_count}")


@source_app.command("json-array-to-jsonl")
def source_json_array_to_jsonl(
    input_json: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    output: Annotated[
        Path,
        typer.Option(..., "--output", help="Required disjoint root for immutable JSONL output."),
    ],
) -> None:
    """Convert a strict object-array JSON source to an immutable JSONL derivative."""
    try:
        report = convert_json_array_to_jsonl(input_json, output)
    except (SourceArrayConversionError, StrictJsonError, ValueError) as exc:
        console.print(f"[red]source conversion refused:[/red] {type(exc).__name__}")
        raise typer.Exit(code=2) from exc
    console.print(
        f"[green]source JSON array converted[/green] id={report.conversion_id} "
        f"rows={report.output_record_count} sha256={report.output_file_sha256}"
    )


@source_app.command("evidence")
def source_evidence(
    canonical_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    evidence_requests_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/source-evidence"),
) -> None:
    """Run deterministic source-evidence Pass 1 from explicit JSON Pointer evidence."""
    canonical = [_canonical_episode_from_json(record) for record in iter_jsonl(canonical_jsonl)]
    requests = [
        SourceEvidenceRequest.model_validate(record, strict=True)
        for record in iter_jsonl(evidence_requests_jsonl)
    ]
    episodes_by_id = {episode.episode_id: episode for episode in canonical}
    requests_by_id = {request.episode_id: request for request in requests}
    if len(episodes_by_id) != len(canonical):
        raise ValueError("canonical input contains duplicate episode IDs")
    if len(requests_by_id) != len(requests):
        raise ValueError("evidence request input contains duplicate episode IDs")
    if set(episodes_by_id) != set(requests_by_id):
        missing = sorted(set(episodes_by_id) - set(requests_by_id))
        extra = sorted(set(requests_by_id) - set(episodes_by_id))
        raise ValueError(
            f"evidence request IDs must match canonical IDs; missing={missing} extra={extra}"
        )
    records = [
        build_source_evidence(
            episodes_by_id[episode_id], requests_by_id[episode_id].argument_evidence
        ).model_dump(mode="json", exclude_none=False)
        for episode_id in sorted(episodes_by_id)
    ]
    manifest = publish_jsonl_artifact(
        output,
        logical_name="source-evidence",
        schema_version="source-evidence-0.1.0",
        stage="source-semantic-pass1",
        records=records,
    )
    passed = sum(record["pass1_result"] == "deterministic_pass" for record in records)
    console.print(
        f"[green]{passed}/{len(records)} deterministic pass[/green] manifest={manifest.manifest_id}"
    )


@app.command("ingest")
def ingest(
    snapshot_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    source_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/bronze"),
    quarantine_output: Annotated[Path, typer.Option("--quarantine-output")] = Path(
        "quarantine/ingest"
    ),
    max_record_bytes: Annotated[int, typer.Option(min=1)] = 8 * 1024 * 1024,
) -> None:
    """Assign physical IDs before strict parsing and account for every row."""
    snapshot = _read_snapshot(snapshot_manifest)
    rows = list(ingest_snapshot(snapshot, source_root, max_record_bytes=max_record_bytes))
    valid = [
        row.model_dump(mode="json", exclude_none=False) for row in rows if row.status == "valid"
    ]
    quarantined = [
        row.model_dump(mode="json", exclude_none=False)
        for row in rows
        if row.status == "quarantined"
    ]
    valid_manifest = publish_jsonl_artifact(
        output,
        logical_name="bronze",
        schema_version="bronze-record-0.1.0",
        stage="ingest",
        records=valid,
        quarantined_rows=len(quarantined),
    )
    if quarantined:
        publish_jsonl_artifact(
            quarantine_output,
            logical_name="ingest-quarantine",
            schema_version="bronze-record-0.1.0",
            stage="ingest-quarantine",
            records=quarantined,
        )
    console.print(
        f"[green]{len(valid)} valid[/green], [yellow]{len(quarantined)} quarantined[/yellow] "
        f"manifest={valid_manifest.manifest_id}"
    )


@registry_app.command("build")
def registry_build(
    definitions_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/registry"),
) -> None:
    definitions = [
        RawToolDefinition.model_validate(row, strict=True) for row in iter_jsonl(definitions_jsonl)
    ]
    registry = ToolRegistry.build(definitions)
    manifest = publish_jsonl_artifact(
        output,
        logical_name="tool-registry",
        schema_version="tool-registry-0.1.0",
        stage="registry",
        records=registry.as_records(),
    )
    console.print(f"[green]{len(registry.tools)} tools[/green] manifest={manifest.manifest_id}")


@app.command("canonicalize")
def canonicalize_command(
    bronze_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    adapter_name: Annotated[str, typer.Option("--adapter")],
    run_event_id: Annotated[str, typer.Option("--run-event-id")],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/canonical"),
) -> None:
    adapter = get_adapter(adapter_name)
    episodes: list[dict[str, JsonValue]] = []
    for raw in iter_jsonl(bronze_jsonl):
        bronze = BronzeRecord.model_validate(raw, strict=True)
        if bronze.parsed_record is None:
            continue
        adapted = adapter.adapt(bronze.parsed_record)
        episodes.append(
            canonicalize_record(bronze, adapted, run_event_id=run_event_id).model_dump(
                mode="json", exclude_none=False
            )
        )
    manifest = publish_jsonl_artifact(
        output,
        logical_name="canonical",
        schema_version="0.1.0",
        stage="canonicalize",
        records=episodes,
    )
    console.print(f"[green]{len(episodes)} episodes[/green] manifest={manifest.manifest_id}")


@app.command("inspect")
def inspect_artifact(
    artifact: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    record_id: Annotated[str | None, typer.Option("--id")] = None,
) -> None:
    for row in iter_jsonl(artifact):
        if not isinstance(row, dict):
            continue
        ids = {
            candidate
            for key in ("episode_id", "source_occurrence_id", "tool_id")
            if isinstance((candidate := row.get(key)), str)
        }
        if record_id is None or record_id in ids:
            console.print_json(json.dumps(row, ensure_ascii=False))
            if record_id is not None:
                return
    if record_id is not None:
        raise typer.Exit(code=1)


@app.command("stats")
def stats(artifact: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    count = 0
    actions: dict[str, int] = {}
    for row in iter_jsonl(artifact):
        count += 1
        if isinstance(row, dict):
            annotations = row.get("annotations")
            if isinstance(annotations, dict):
                decision = annotations.get("decision")
                if isinstance(decision, dict) and isinstance(decision.get("action"), str):
                    action = str(decision["action"])
                    actions[action] = actions.get(action, 0) + 1
    table = Table("metric", "value")
    table.add_row("rows", str(count))
    for action, action_count in sorted(actions.items()):
        table.add_row(f"action.{action}", str(action_count))
    console.print(table)


@events_app.command("show")
def events_show(root: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    for event in EventLog(root).read_verified():
        console.print_json(event.model_dump_json())


@audit_app.command("duplicates")
def audit_duplicates(
    canonical_jsonl: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/audit"),
) -> None:
    """Build a non-destructive exact alias/conflict audit across canonical JSONL files."""
    episodes = [
        _canonical_episode_from_json(record)
        for path in canonical_jsonl
        for record in iter_jsonl(path)
    ]
    audit = audit_exact_conflicts(episodes)
    target = output / f"{audit.audit_id}.json"
    publish_bytes_atomic(target, canonical_bytes(audit) + b"\n")
    console.print(
        f"[green]{len(audit.duplicate_groups)} duplicate groups[/green], "
        f"[yellow]{len(audit.conflict_candidates)} review candidates[/yellow] {target}"
    )


@audit_app.command("near-duplicates")
def audit_near_duplicates(
    documents_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/near-duplicates"),
    config: Annotated[Path, typer.Option("--config")] = Path("configs/phase4.toml"),
) -> None:
    """Retrieve review-only near-duplicate candidates; never drop either record."""
    policy = load_phase4_config(config)
    documents = [
        SimilarityDocument.model_validate(record, strict=True)
        for record in iter_jsonl(documents_jsonl)
    ]
    document_by_id = {document.episode_id: document.text for document in documents}
    if len(document_by_id) != len(documents):
        raise ValueError("similarity documents contain duplicate episode IDs")
    candidates = retrieve_near_duplicate_candidates(
        document_by_id,
        threshold=policy.near_duplicate_candidate_threshold,
        ngram_size=policy.ngram_size,
    )
    manifest = publish_jsonl_artifact(
        output,
        logical_name="near-duplicate-candidates",
        schema_version="near-duplicate-candidate-0.1.0",
        stage="near-duplicate-retrieval",
        records=[candidate.model_dump(mode="json", exclude_none=False) for candidate in candidates],
    )
    console.print(
        f"[yellow]{len(candidates)} review candidates[/yellow] manifest={manifest.manifest_id}"
    )


@app.command("diagnostics")
def diagnostics_catalog() -> None:
    console.print_json(CATALOG.model_dump_json())


@pilot_app.command("run")
def pilot_run(
    input_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    output: Annotated[
        Path,
        typer.Option(
            ...,
            "--output",
            help="Required disjoint root for immutable pilot artifacts.",
        ),
    ],
    dataset_namespace: Annotated[str, typer.Option("--dataset")],
    source_revision: Annotated[str, typer.Option("--revision")],
    license_id: Annotated[str, typer.Option("--license")],
    adapter: Annotated[str, typer.Option("--adapter")],
    run_event_id: Annotated[str, typer.Option("--run-event-id")],
    max_record_bytes: Annotated[int, typer.Option(min=1)] = 8 * 1024 * 1024,
    source_config: Annotated[str | None, typer.Option("--source-config")] = None,
    source_split: Annotated[str | None, typer.Option("--source-split")] = None,
    license_url: Annotated[str | None, typer.Option("--license-url")] = None,
) -> None:
    """Snapshot one JSONL input and run ingest/canonical/audit without models."""
    try:
        report = run_operational_pilot(
            input_jsonl,
            output,
            dataset_namespace=dataset_namespace,
            source_revision=source_revision,
            license_id=license_id,
            adapter_name=adapter,
            run_event_id=run_event_id,
            max_record_bytes=max_record_bytes,
            source_config=source_config,
            source_split=source_split,
            license_url=license_url,
        )
    except PilotConfigurationError as exc:
        console.print(f"[red]pilot refused:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    if report.status == "blocked":
        console.print(
            f"[yellow]pilot blocked[/yellow] id={report.pilot_id} "
            f"canonical={report.canonical_records} "
            f"canonical_quarantine={report.canonical_quarantined_records} "
            f"reasons={','.join(report.block_reasons)}"
        )
        raise typer.Exit(code=2)
    console.print(
        f"[green]pilot passed[/green] id={report.pilot_id} "
        f"canonical={report.canonical_records} "
        f"review_required_conflicts={report.review_required_conflicts}"
    )


@evaluation_app.command("run")
def evaluation_run(
    input_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    output: Annotated[
        Path,
        typer.Option(
            ...,
            "--output",
            help="Required disjoint root for immutable evaluation artifacts.",
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            ...,
            "--config",
            help="Required non-default live config with both egress gates enabled.",
        ),
    ],
    role: Annotated[
        str,
        typer.Option(
            ...,
            "--role",
            help="Required OpenAI evaluator role: mini_verifier or strong_judge.",
        ),
    ],
    run_id: Annotated[str, typer.Option(..., "--run-id")],
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help="Explicitly permit this command to send its declared evaluation rows.",
        ),
    ] = False,
) -> None:
    """Run one immutable JSONL batch through exactly one OpenAI judge role.

    Inputs must be full-leaf source/target pairs whose hashes match their
    evaluation units.  The command creates no Gold acceptance: an authorized
    human must submit review decisions through the existing review command.
    """
    if not live:
        console.print(
            "[yellow]evaluation run requires --live; no provider request was made.[/yellow]"
        )
        raise typer.Exit(code=2)
    if _is_default_pipeline_config(config):
        console.print(
            "[yellow]evaluation run requires an explicit non-default --config path; "
            "no provider request was made.[/yellow]"
        )
        raise typer.Exit(code=2)
    if role not in {"mini_verifier", "strong_judge"}:
        console.print(
            "[yellow]evaluation run requires --role mini_verifier or strong_judge; "
            "no provider request was made.[/yellow]"
        )
        raise typer.Exit(code=2)

    try:
        live_config = inspect_config(config)
        provider_role = getattr(live_config.providers, role)
        resolver = AllowListedSecretResolver(
            allowed_names=frozenset({provider_role.api_key_env}), env_file=_LOCAL_ENV_FILE
        )
        transport = StdlibJsonTransport(
            credential_name=provider_role.api_key_env,
            secret_lookup=resolver.resolve,
        )

        def judge_factory(attempt_sink: ProviderAttemptSink) -> OpenAIResponsesJudge:
            return OpenAIResponsesJudge(
                config=live_config,
                role_name=role,
                transport=transport,
                attempt_sink=attempt_sink,
            )

        artifacts = run_live_evaluation(
            input_jsonl,
            output,
            config=live_config,
            role_name=role,
            run_id=run_id,
            judge_factory=judge_factory,
        )
    except (
        LiveEvaluationConfigurationError,
        LiveEvaluationRuntimeError,
        StrictJsonError,
        ValidationError,
    ):
        console.print("[red]evaluation refused before publication; no source was modified.[/red]")
        raise typer.Exit(code=2) from None
    console.print(
        f"[green]evaluation batch completed[/green] role={artifacts.report.role} "
        f"succeeded={artifacts.report.succeeded_rows} failed={artifacts.report.failed_rows} "
        f"report={artifacts.report.report_id} gold_release_allowed=false"
    )


@provider_app.command("smoke")
def provider_smoke(
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help="Inspect explicit live prerequisites only; never sends a provider request.",
        ),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Explicit non-default config required with --live.",
        ),
    ] = None,
    send: Annotated[
        bool,
        typer.Option(
            "--send",
            help="Send one fixed synthetic request after --live prerequisite checks.",
        ),
    ] = False,
    role: Annotated[
        str,
        typer.Option(
            "--role",
            help="Live role: translator, mini_verifier, or strong_judge.",
        ),
    ] = "translator",
) -> None:
    """Safely inspect provider prerequisites without reading keys or making network calls."""
    if role not in {"translator", "mini_verifier", "strong_judge"}:
        raise typer.BadParameter("role must be translator, mini_verifier, or strong_judge")
    config_path = config if config is not None else _DEFAULT_PIPELINE_CONFIG
    env_file_exists = _LOCAL_ENV_FILE.is_file()

    if send and not live:
        console.print("[yellow]--send requires --live; no provider request was made.[/yellow]")
        raise typer.Exit(code=2)

    if not live:
        # The default path must remain subject to the production offline gate.
        offline_config = load_config(config_path)
        console.print("[green]provider smoke: dry-run[/green]")
        console.print(
            "offline gates: "
            f"providers.enabled={offline_config.providers.enabled}, "
            "network_egress_enabled="
            f"{offline_config.providers.network_egress_enabled}"
        )
        console.print(
            ".env: "
            f"{'present' if env_file_exists else 'missing'} "
            "(contents and credential values not read)"
        )
        console.print("live API: disabled; this command made zero network requests")
        return

    if config is None or _is_default_pipeline_config(config_path):
        console.print(
            "[yellow]--live requires an explicit non-default --config path; "
            "no provider request was made.[/yellow]"
        )
        raise typer.Exit(code=2)

    inspected_config = _load_live_smoke_config(config_path)
    failures = _live_smoke_failures(
        inspected_config, role_name=role, env_file_exists=env_file_exists
    )
    console.print("[yellow]provider smoke: live prerequisite inspection only[/yellow]")
    if send:
        console.print(
            ".env: "
            f"{'present' if env_file_exists else 'missing'} "
            "(value is resolved only by the secret resolver and never displayed)"
        )
    else:
        console.print(
            ".env: "
            f"{'present' if env_file_exists else 'missing'} "
            "(contents and credential values not read)"
        )
    if failures:
        console.print("[red]live API remains disabled:[/red]")
        for failure in failures:
            console.print(f"- {failure}")
        console.print("this command made zero network requests")
        raise typer.Exit(code=2)

    if send:
        try:
            if role == "translator":
                _send_synthetic_deepseek_smoke(inspected_config)
            else:
                _send_synthetic_openai_judge_smoke(inspected_config, role_name=role)
        except (
            CredentialResolutionError,
            ProviderAdapterError,
            SecureTransportError,
            ValueError,
        ) as exc:
            console.print(f"[red]synthetic provider smoke failed: {type(exc).__name__}[/red]")
            raise typer.Exit(code=1) from exc
        return

    console.print("[green]live prerequisites are structurally present[/green]")
    console.print("live API remains disabled; this command made zero network requests")


def _future_phase(name: str) -> None:
    console.print(f"[yellow]{name} is intentionally unavailable in offline Phase 1-5.[/yellow]")
    raise typer.Exit(code=2)


@source_app.command("review")
def source_review() -> None:
    _future_phase("source review")


@select_app.command("freeze")
def select_freeze(
    candidates_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/selection"),
    config: Annotated[Path, typer.Option("--config")] = Path("configs/phase4.toml"),
) -> None:
    """Freeze S400 only from human-adjudicated, fully grounded candidates."""
    load_phase4_config(config)
    candidates = [
        SelectionCandidate.model_validate(record, strict=True)
        for record in iter_jsonl(candidates_jsonl)
    ]
    manifest = freeze_s400(candidates)
    target = output / f"{manifest.selection_manifest_id}.json"
    publish_bytes_atomic(target, canonical_bytes(manifest) + b"\n")
    console.print(f"[green]S400 frozen[/green] {target}")


@app.command("translate")
def translate(
    input_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Option(..., "--output")],
    config: Annotated[Path, typer.Option(..., "--config")],
    field_policy: Annotated[Path, typer.Option(..., "--field-policy")],
    prompt: Annotated[Path, typer.Option(..., "--prompt")],
    live: Annotated[bool, typer.Option("--live")] = False,
) -> None:
    """Translate only policy-approved canonical leaves into a disjoint output root."""
    if not live:
        console.print("[yellow]translate requires --live; no provider request was made.[/yellow]")
        raise typer.Exit(code=2)
    if _is_default_pipeline_config(config):
        console.print("[yellow]translate requires an explicit non-default --config path.[/yellow]")
        raise typer.Exit(code=2)
    try:
        live_config = load_live_config(config)
        role = live_config.providers.translator
        resolver = AllowListedSecretResolver(
            allowed_names=frozenset({role.api_key_env}), env_file=_LOCAL_ENV_FILE
        )
        report = run_operational_translation(
            input_jsonl,
            output,
            config=live_config,
            field_policy=load_field_policy(field_policy),
            prompt=load_prompt_bundle(prompt),
            transport=StdlibJsonTransport(
                credential_name=role.api_key_env,
                secret_lookup=resolver.resolve,
            ),
        )
    except (
        OperationalTranslationError,
        ProviderAdapterError,
        SecureTransportError,
        ValueError,
    ) as exc:
        console.print(f"[red]translation refused or failed: {type(exc).__name__}[/red]")
        raise typer.Exit(code=2) from exc
    console.print(
        f"[green]translation batch completed[/green] id={report.batch_id} "
        f"translated={report.translated_records} "
        f"research_required={report.research_required_records} "
        "promotion=not_eligible"
    )


@app.command("validate")
def validate() -> None:
    _future_phase("translation validation")


@review_app.callback(invoke_without_command=True)
def review(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(
            "Specify review prepare, review submit-evaluation, or review submit-conflict."
        )
        raise typer.Exit(code=2)


@review_app.command("prepare")
def review_prepare(
    canonical_quarantine_jsonl: Annotated[
        list[Path] | None,
        typer.Option(
            "--canonical-quarantine",
            exists=True,
            dir_okay=False,
            help="Repeatable canonical-quarantine JSONL input.",
        ),
    ] = None,
    conflict_audit_json: Annotated[
        list[Path] | None,
        typer.Option(
            "--conflict-audit",
            exists=True,
            dir_okay=False,
            help="Repeatable exact-conflict-audit JSON input.",
        ),
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/review-queue"),
) -> None:
    """Publish an open human worklist from immutable quarantine/audit evidence.

    The command writes no decisions. A conflict task carries everything needed
    to prepare a separate, reviewer-authored ``review submit-conflict`` JSONL
    entry. A quarantine task remains open until approved remediation produces
    a new pilot; neither input record is repaired or dropped here.
    """
    quarantine_paths = canonical_quarantine_jsonl or []
    audit_paths = conflict_audit_json or []
    if not quarantine_paths and not audit_paths:
        raise ValueError("provide at least one --canonical-quarantine or --conflict-audit")
    resolved_output = output.resolve(strict=False)
    for input_path in [*quarantine_paths, *audit_paths]:
        if input_path.resolve(strict=True).is_relative_to(resolved_output):
            raise ValueError("review queue output must not contain an input evidence file")
    try:
        quarantines = [
            CanonicalQuarantineRecord.model_validate(record, strict=True)
            for path in quarantine_paths
            for record in iter_jsonl(path)
        ]
        audits = [
            ExactConflictAudit.model_validate(
                _read_strict_json_object(path, label="conflict audit"), strict=True
            )
            for path in audit_paths
        ]
        tasks = build_review_tasks(quarantines, audits)
    except (ReviewQueueInputError, StrictJsonError, ValidationError, ValueError) as exc:
        console.print(f"[red]review queue refused:[/red] {type(exc).__name__}")
        raise typer.Exit(code=2) from exc
    manifest = publish_jsonl_artifact(
        output,
        logical_name="human-review-tasks",
        schema_version="review-task-0.1.0",
        stage="human-review-queue",
        records=[task.model_dump(mode="json", exclude_none=False) for task in tasks],
    )
    quarantine_count = sum(task.task_kind == "canonical_quarantine" for task in tasks)
    conflict_count = sum(task.task_kind == "conflict_adjudication" for task in tasks)
    console.print(
        "[green]human review queue published[/green] "
        f"manifest={manifest.manifest_id} "
        f"quarantines={quarantine_count} conflicts={conflict_count}"
    )


@review_app.command("submit-evaluation")
def review_submit_evaluation(
    decision_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    run_id: Annotated[str, typer.Option("--run-id")],
    model_verdicts_jsonl: Annotated[
        Path, typer.Option("--model-verdicts-jsonl", exists=True, dir_okay=False)
    ],
    events_root: Annotated[Path, typer.Option("--events-root")] = Path("events/human-evaluations"),
    parent_manifest_id: Annotated[str | None, typer.Option("--parent-manifest-id")] = None,
    timestamp_utc: Annotated[str | None, typer.Option("--timestamp-utc")] = None,
) -> None:
    """Strictly validate and append one reviewer-authored Gold decision.

    The input must already contain the reviewer identity, explicit ``human``
    authority, decision, rationale, and deterministic review ID.  This command
    does not construct, modify, or auto-accept a reviewer decision.  The
    referenced model verdict must be supplied locally so all finding links and
    a potential Gold acceptance can be verified before an event is appended.
    """
    decision = HumanEvaluationReview.model_validate(
        _read_exactly_one_jsonl_object(decision_jsonl), strict=True
    )
    matching_verdicts = [
        verdict
        for verdict in _read_model_verdicts(model_verdicts_jsonl)
        if verdict.verdict_id == decision.verdict_id
    ]
    if len(matching_verdicts) != 1:
        raise ValueError("human review must reference exactly one supplied model verdict")
    decide_gold_acceptance(model_verdict=matching_verdicts[0], human_review=decision)
    entry = HumanEvaluationReviewLog(events_root).append(
        run_id=run_id,
        review=decision,
        parent_manifest_id=parent_manifest_id,
        timestamp_utc=timestamp_utc,
    )
    console.print(
        "[green]human evaluation review appended[/green] "
        f"review_id={entry.review.review_id} event_id={entry.event.event_id}"
    )


@review_app.command("submit-conflict")
def review_submit_conflict(
    decision_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    run_id: Annotated[str, typer.Option("--run-id")],
    events_root: Annotated[Path, typer.Option("--events-root")] = Path(
        "events/conflict-adjudications"
    ),
    parent_manifest_id: Annotated[str | None, typer.Option("--parent-manifest-id")] = None,
    timestamp_utc: Annotated[str | None, typer.Option("--timestamp-utc")] = None,
) -> None:
    """Strictly validate and append an externally authored conflict adjudication."""
    decision = ConflictAdjudication.model_validate(
        _read_exactly_one_jsonl_object(decision_jsonl), strict=True
    )
    entry = ConflictAdjudicationLog(events_root).append(
        run_id=run_id,
        conflict_id=decision.conflict_id,
        left_episode_id=decision.left_episode_id,
        right_episode_id=decision.right_episode_id,
        decision=decision.decision,
        reviewer_id=decision.reviewer_id,
        reviewer_authority=decision.reviewer_authority,
        rubric_version=decision.rubric_version,
        rationale=decision.rationale,
        supersedes_event_id=decision.supersedes_event_id,
        parent_manifest_id=parent_manifest_id,
        timestamp_utc=timestamp_utc,
    )
    console.print(
        "[green]conflict adjudication appended[/green] "
        f"conflict_id={entry.adjudication.conflict_id} event_id={entry.event.event_id}"
    )


@release_app.command("build")
def release_build(
    dataset_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    gold_members_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    model_verdicts_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    dataset_file: Annotated[list[str], typer.Option("--file")],
    review_events_root: Annotated[Path, typer.Option("--review-events-root")] = Path(
        "events/human-evaluations"
    ),
    output: Annotated[Path, typer.Option("--output")] = Path("manifests/release.jsonl"),
) -> None:
    """Create a manifest only after all listed members pass explicit human Gold gates."""
    members = _read_gold_members(gold_members_jsonl)
    verdicts = _read_model_verdicts(model_verdicts_jsonl)
    review_entries = HumanEvaluationReviewLog(review_events_root).read_verified()
    reviews = [entry.review for entry in review_entries]
    manifest = build_release_manifest(
        dataset_root,
        relative_files=dataset_file,
        gold_members=members,
        model_verdicts=verdicts,
        human_reviews=reviews,
        gold_acceptances=_gold_acceptances_from_reviews(verdicts, reviews),
    )
    write_release_manifest(output, manifest)
    console.print(
        f"[green]release manifest created[/green] {output} release_id={manifest.release_id}"
    )


@release_app.command("validate")
def release_validate(
    dataset_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    manifest_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    model_verdicts_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    review_events_root: Annotated[Path, typer.Option("--review-events-root")] = Path(
        "events/human-evaluations"
    ),
) -> None:
    """Re-verify immutable files and every human-review-to-Gold membership link."""
    manifest = read_release_manifest(manifest_jsonl)
    verdicts = _read_model_verdicts(model_verdicts_jsonl)
    review_entries = HumanEvaluationReviewLog(review_events_root).read_verified()
    reviews = [entry.review for entry in review_entries]
    validate_release_manifest(
        dataset_root,
        manifest,
        model_verdicts=verdicts,
        human_reviews=reviews,
        gold_acceptances=_gold_acceptances_from_reviews(verdicts, reviews),
    )
    console.print(f"[green]release manifest valid[/green] release_id={manifest.release_id}")


@app.command("render")
def render() -> None:
    _future_phase("trainer rendering")


@app.command("diff")
def diff() -> None:
    _future_phase("artifact diff")


@index_app.command("rebuild")
def index_rebuild() -> None:
    _future_phase("derived index rebuild")


@memory_app.command("inspect")
@memory_app.command("promote")
@memory_app.command("conflicts")
def memory_command() -> None:
    _future_phase("translation memory")


def main() -> None:
    try:
        app()
    except (ValidationError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


if __name__ == "__main__":
    main()
