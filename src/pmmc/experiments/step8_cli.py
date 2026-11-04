"""Unified Step 8 entry point; development and paper use the same stage DAG."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

import typer

from pmmc.experiments.step8_artifacts import (
    CampaignArtifactStore,
    durable_write,
    sha256_file,
    write_json,
)
from pmmc.experiments.step8_campaign import (
    CampaignJobPlanError,
    build_campaign_jobs,
    load_benchmark_config,
    persist_campaign_job_plan,
    plan_campaign,
)
from pmmc.experiments.step8_jobs import BenchmarkStageJob
from pmmc.experiments.step8_runner import campaign_failure_boundary
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.step8_repositories import PostgresBenchmarkRepository
from pmmc.infrastructure.settings import get_settings
from pmmc.pilots.partition import load_partition
from pmmc.schemas.base import DatasetName
from pmmc.schemas.step8 import (
    BenchmarkCampaignManifest,
    CampaignPhase,
    MethodProtocolFreeze,
    ResourceClass,
    SelectionExpansionManifest,
    Step8BenchmarkConfig,
    build_experiment_matrix,
)

app = typer.Typer(no_args_is_help=True, help="Incremental, frozen Step 8 benchmark campaigns.")


@app.command("replace-conformance-probe")
def replace_conformance_probe(
    campaign_dir: Path = typer.Option(..., exists=True, file_okay=False),
    rate_card: Path = typer.Option(..., exists=True, dir_okay=False),
    approval_id: str = typer.Option(...),
    original_request_id: str = typer.Option(...),
    original_request_hash: str = typer.Option(...),
    original_reserved_usd: str = typer.Option(...),
    budget_usd: str = typer.Option(...),
    approver_id: str = typer.Option(...),
    output: Path = typer.Option(..., dir_okay=False),
    execute: bool = typer.Option(False, help="Dispatch/recover the bounded native probes."),
) -> None:
    """Authorize one replacement, retain unknown spend, and optionally run synthetic probes."""
    from pmmc.experiments.step8_conformance_replacement import ReplacementError
    from pmmc.experiments.step8_failure import failure_diagnostics, safe_failure_code
    from pmmc.experiments.step8_replacement_command import replace_probe, replacement_heartbeat

    try:
        if output.exists():
            raise ReplacementError("replacement_output_exists_use_new_attempt_path")
        manifest, config = _load(campaign_dir)
        with replacement_heartbeat():
            report = asyncio.run(replace_probe(
                settings=get_settings(), campaign_dir=campaign_dir,
                manifest=manifest, config=config,
                selection=_latest_selection(campaign_dir), rate_card=rate_card,
                approval_id=approval_id, original_request_id=original_request_id,
                original_request_hash=original_request_hash,
                original_reserved_usd=Decimal(original_reserved_usd),
                budget_usd=Decimal(budget_usd),
                approver_id=approver_id, execute=execute,
            ))
        write_json(output, report, immutable=True)
    except Exception as exc:
        report = {
            "event": "STEP8_EXPLICIT_REPLACEMENT", "state": "failed",
            "error_code": safe_failure_code(exc),
            "diagnostics": failure_diagnostics(exc), "gpu_operations": 0,
            "api_call_status": "requires_ledger_and_journal_check", "automatic_new_probe": False,
        }
        typer.echo(json.dumps(report, sort_keys=True))
        try:
            write_json(output.with_name(output.stem + "-failure.json"), report)
        except OSError:
            typer.echo("STEP8_EXPLICIT_REPLACEMENT failure_report_write_failed")
        raise typer.Exit(1) from None
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("inspect-provider-recovery")
def inspect_provider_recovery(
    campaign_dir: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
    check_journal_storage: bool = typer.Option(False),
) -> None:
    """Read ledger/journals without API calls or settlement; optionally probe storage."""
    from pmmc.experiments.step8_failure import failure_diagnostics
    from pmmc.experiments.step8_provider_recovery import inspect_provider_recovery as inspect

    try:
        report = asyncio.run(inspect(get_settings(), campaign_dir))
        if check_journal_storage:
            try:
                report["journal_storage"] = _preflight_response_journal(campaign_dir.parent)
            except Exception as exc:
                report["journal_storage"] = {
                    "state": "failed", "diagnostics": failure_diagnostics(exc),
                }
                report["requires_review"] = True
        write_json(output, report, immutable=True)
    except Exception as exc:
        typer.echo(json.dumps({
            "event": "STEP8_PROVIDER_RECOVERY", "state": "failed",
            "diagnostics": failure_diagnostics(exc), "provider_calls": 0,
        }, sort_keys=True))
        raise typer.Exit(1) from None
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("audit-images")
def audit_images(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
    selection: Literal["diagnostic", "regression"] = typer.Option("regression"),
) -> None:
    """Read-only development image audit; no model, GPU, database or gold access."""
    from pmmc.experiments.step8_artifacts import json_bytes
    from pmmc.experiments.step8_image_audit import inspect_step8_images

    typer.echo("STEP8_IMAGE_AUDIT state=start model_calls=0 gpu_operations=0")
    try:
        report = inspect_step8_images(get_settings(), load_benchmark_config(config),
                                      selection=selection)
        durable_write(output, json_bytes(report), immutable=True)
    except Exception:
        # Canonical validation exceptions can include private memory or path fields.
        typer.echo("STEP8_IMAGE_AUDIT state=failed code=input_audit_failed")
        raise typer.Exit(1) from None
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("audit-contexts")
def audit_contexts(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    processor_audit: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
    selection: Literal["diagnostic", "regression"] = typer.Option("regression"),
) -> None:
    """CPU-only exact rendered-context audit; no generation or capacity approval."""
    from pmmc.experiments.step8_artifacts import json_bytes
    from pmmc.experiments.step8_context_audit import Step8ContextAuditError, inspect_step8_contexts

    try:
        report = inspect_step8_contexts(
            get_settings(), load_benchmark_config(config), selection=selection,
            pinned_processor_audit=processor_audit,
        )
        durable_write(output, json_bytes(report), immutable=True)
    except Step8ContextAuditError as exc:
        typer.echo(json.dumps(exc.report, sort_keys=True))
        raise typer.Exit(1) from None
    except Exception:
        typer.echo("STEP8_CONTEXT_AUDIT state=failed code=report_or_config_invalid")
        raise typer.Exit(1) from None
    typer.echo(json.dumps({
        "state": report["state"], "content_hash": report["content_hash"],
        "rendered_request_count": report["aggregate"]["rendered_request_count"],
        "gpu_benchmark_ready": False,
    }, sort_keys=True))


@app.command("prepare-capacity")
def prepare_capacity(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    context_audit: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(..., file_okay=False),
) -> None:
    """Create immutable observed-layout inputs; original scientific inputs stay untouched."""
    from pmmc.experiments.step8_capacity_inputs import prepare_native_capacity_inputs

    try:
        report = prepare_native_capacity_inputs(
            get_settings(), load_benchmark_config(config), audit_path=context_audit,
            output_dir=output_dir,
        )
    except Exception:
        typer.echo("STEP8_CAPACITY_INPUTS state=failed code=capacity_inputs_invalid")
        raise typer.Exit(1) from None
    typer.echo(json.dumps(report, sort_keys=True))


def conformance_aliases(resource: ResourceClass) -> tuple[str, ...]:
    return {
        ResourceClass.GPU_9B: (
            "compiler-qwen3.5-9b",
            "reranker-qwen3.5-9b",
            "answer-qwen3.5-9b",
            "grounding-qwen3.5-9b",
            "judge-qwen3.5-9b",
        ),
        ResourceClass.GPU_27B: ("answer-qwen3.5-27b",),
        ResourceClass.API: ("answer-gpt-6-luna", "answer-claude-haiku-5.5"),
    }.get(resource, ())


@app.command("bootstrap")
def bootstrap() -> None:
    """Initialize persistent CPU services and verify the runtime/evaluator boundary."""
    from pmmc.experiments.step8_bootstrap import Step8BootstrapError
    from pmmc.experiments.step8_bootstrap import bootstrap as initialize

    try:
        report = initialize(get_settings())
    except Step8BootstrapError as exc:
        typer.echo(json.dumps(exc.report, sort_keys=True))
        raise typer.Exit(1) from None
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("prepare-exploration")
def prepare_exploration_command(
    run_id: str = typer.Option(...),
    variant: Literal["wr", "w", "r", "legacy"] = typer.Option("wr"),
) -> None:
    """Materialize the public recipe against user-supplied benchmark assets."""
    from pmmc.experiments.release_inputs import prepare_exploration

    result = prepare_exploration(get_settings(), run_id, variant=variant)
    typer.echo(json.dumps(result, default=str, sort_keys=True))


@app.command("run-exploration")
def run_exploration_command(
    run_id: str = typer.Option(...),
    stage: Literal["main", "analysis", "baselines"] = typer.Option("main"),
    variant: Literal["wr", "w", "r", "legacy"] = typer.Option("wr"),
    worker_pool: Literal["all", "control", "gpu"] = typer.Option("all"),
) -> None:
    """Execute an exact, dependency-closed slice; no implied analysis or paper permission."""
    from pmmc.experiments.release_comparison import record_comparison
    from pmmc.experiments.release_execution import execute_stage, require_comparison_order
    from pmmc.experiments.release_inputs import exploration_campaign_directory

    settings = get_settings()
    directory = exploration_campaign_directory(settings, run_id, variant=variant)
    with campaign_failure_boundary(directory, stage="exploration-" + stage):
        require_comparison_order(settings, run_id, variant, stage)
        if stage == "baselines":
            record_comparison(settings, run_id, require=True)
        result = execute_stage(settings, directory, stage, worker_pool)
        if worker_pool != "gpu" and result["state"] == "stage_complete" and stage != "baselines":
            record_comparison(settings, run_id)
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("quote-exploration")
def quote_exploration_command(
    run_id: str = typer.Option(...), rate_card: Path = typer.Option(..., exists=True),
    variant: Literal["wr", "w", "r", "legacy"] = typer.Option("wr"),
    stage: Literal["main", "analysis", "baselines"] = typer.Option("main"),
) -> None:
    """Estimate remaining API work; this command cannot approve spending."""
    from pmmc.experiments.release_inputs import exploration_campaign_directory
    from pmmc.experiments.release_reporting import estimate_release_budget
    from pmmc.experiments.step8_supervisor import _snapshot

    settings = get_settings()
    directory = exploration_campaign_directory(settings, run_id, variant=variant)
    manifest, _ = _load(directory)
    asyncio.run(_register(directory, budget=None, rate_card=None, approver=None, retry=False))
    quote = estimate_release_budget(directory, jobs=asyncio.run(
        _snapshot(settings, manifest.campaign_id)), scope=stage, rate_card=rate_card,
        project_root=settings.project_root)
    write_json(directory / "reports" / f"budget-quote-{stage}.json", quote)
    typer.echo(json.dumps(quote, sort_keys=True))


@app.command("approve-exploration")
def approve_exploration_command(
    run_id: str = typer.Option(...), rate_card: Path = typer.Option(..., exists=True),
    selection_hash: str = typer.Option(...), approver_id: str = typer.Option(...),
    limit_usd: str = typer.Option(...),
    variant: Literal["wr", "w", "r", "legacy"] = typer.Option("wr"),
    stage: Literal["main", "analysis", "baselines"] = typer.Option("main"),
) -> None:
    """Authorize a new selection only after its quote, never from an older ledger."""
    from pmmc.experiments.release_inputs import exploration_campaign_directory
    from pmmc.experiments.release_reporting import approve_release_budget

    settings = get_settings()
    directory = exploration_campaign_directory(settings, run_id, variant=variant)
    quote = json.loads((directory / "reports" / f"budget-quote-{stage}.json").read_bytes())
    if (quote["selection_hash"] != selection_hash
            or quote["rate_card_sha256"] != sha256_file(rate_card)):
        raise ValueError("budget_quote_does_not_match_selection_and_rates")
    result = asyncio.run(approve_release_budget(directory, selection_hash=selection_hash,
        rate_card=rate_card, project_root=settings.project_root, approver_id=approver_id,
        limit_usd=Decimal(limit_usd)))
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("calibrate")
def calibrate(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Measure the frozen 27B pool with synthetic load; no benchmark query is read."""
    from pmmc.runtime.manager import ManagedRuntimeError
    from pmmc.runtime.step8_capacity_calibration import calibrate_step8_capacity

    settings = get_settings()
    resolved = load_benchmark_config(config)
    try:
        receipt, _path = calibrate_step8_capacity(
            settings.model_copy(
                update={
                    "model_registry_path": settings.resolve_path(resolved.model_registry),
                    "runtime_model_lock_path": settings.resolve_path(resolved.model_lock),
                }
            ),
            profile_path=settings.resolve_path(resolved.runtime_profile),
        )
    except ManagedRuntimeError:
        # The runtime has already emitted safe diagnostics; do not repeat a rich traceback.
        typer.echo(json.dumps({"state": "failed", "code": "step8_capacity_calibration_failed",
                               "benchmark_queries": 0, "new_receipt_published": False},
                              sort_keys=True))
        raise typer.Exit(1) from None
    typer.echo(
        json.dumps(
            {
                "state": receipt.status,
                "replicas": receipt.replica_count,
                "tensor_parallel_size": receipt.tensor_parallel_size,
                "benchmark_queries": 0,
                "model_alias": receipt.model_alias,
            },
            sort_keys=True,
        )
    )


@app.command("prepare-models")
def prepare_models(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    allow_download: bool = typer.Option(False),
    create_lock: bool = typer.Option(False),
) -> None:
    """Explicit cold download/lock preparation; never starts inference or changes a lock."""
    from pmmc.experiments.step8_model_preparation import (
        Step8ModelPreparationError,
        prepare_step8_models,
    )

    try:
        report = prepare_step8_models(
            get_settings(),
            load_benchmark_config(config),
            allow_download=allow_download,
            create_lock=create_lock,
        )
    except Step8ModelPreparationError as exc:
        typer.echo(json.dumps(exc.report, sort_keys=True))
        raise typer.Exit(1) from None
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("reconstruct-development-inputs")
def reconstruct_inputs(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    memlens_snapshot: Path = typer.Option(..., exists=True, file_okay=False),
    memgallery_snapshot: Path = typer.Option(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("artifacts/step8-inputs/reconstructed-development")),
    acknowledge_history_loss: bool = typer.Option(False),
) -> None:
    """Create cold development inputs; use derive-development-inputs for config upgrades."""
    from pmmc.experiments.step8_input_reconstruction import (
        Step8InputReconstructionError,
        reconstruct_development_inputs,
    )

    try:
        report = reconstruct_development_inputs(
            project_root=get_settings().project_root, config_path=config,
            snapshots={DatasetName.MEMLENS: memlens_snapshot,
                       DatasetName.MEMGALLERY: memgallery_snapshot},
            output_dir=output_dir, acknowledge_history_loss=acknowledge_history_loss,
        )
    except Step8InputReconstructionError as exc:
        typer.echo(json.dumps(exc.report, sort_keys=True))
        raise typer.Exit(1) from None
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("derive-development-inputs")
def derive_inputs(
    source_dir: Path = typer.Option(..., exists=True, file_okay=False),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(...),
) -> None:
    """Rebind configuration while preserving verified development inputs byte-for-byte."""
    from pmmc.experiments.step8_input_derivation import (
        Step8InputDerivationError,
        derive_development_inputs,
    )

    try:
        report = derive_development_inputs(
            project_root=get_settings().project_root, source_dir=source_dir,
            config_path=config, output_dir=output_dir,
        )
    except Step8InputDerivationError as exc:
        typer.echo(json.dumps(exc.report, sort_keys=True))
        raise typer.Exit(1) from None
    typer.echo(json.dumps(report, sort_keys=True))


def _load(directory: Path) -> tuple[BenchmarkCampaignManifest, Step8BenchmarkConfig]:
    return (
        BenchmarkCampaignManifest.model_validate_json(
            (directory / "manifests/campaign.json").read_text()
        ),
        Step8BenchmarkConfig.model_validate_json((directory / "manifests/config.json").read_text()),
    )


def _latest_selection(directory: Path) -> SelectionExpansionManifest:
    values = [
        SelectionExpansionManifest.model_validate_json(path.read_text())
        for path in (directory / "manifests/selections").glob("*.json")
    ]
    parents = {item.parent_selection_hash for item in values}
    leaves = [item for item in values if item.content_hash not in parents]
    if len(leaves) != 1:
        raise ValueError("ambiguous selection expansion chain")
    return leaves[0]


@app.command("plan")
def plan(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    phase: CampaignPhase = typer.Option(CampaignPhase.DEVELOPMENT),
    selection: str = typer.Option("regression"),
    campaign_dir: Path | None = typer.Option(None),
    freeze: Path | None = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    settings = get_settings()
    resolved = load_benchmark_config(config)
    directory, manifest, selected = plan_campaign(
        config=resolved,
        project_root=settings.project_root,
        phase=phase,
        selection=selection,
        campaign_dir=campaign_dir,
        freeze_path=freeze,
    )
    _, assignments = load_partition(settings.resolve_path(resolved.partition_dir))
    conversations: dict[DatasetName, dict[str, list[str]]] = {}
    for dataset, ids in selected.query_ids.items():
        conversations[dataset] = {}
        for assignment in assignments:
            if assignment.dataset is dataset and assignment.query_id in ids:
                conversations[dataset].setdefault(assignment.conversation_id, []).append(
                    assignment.query_id
                )
    jobs = build_campaign_jobs(
        manifest=manifest,
        selection=selected,
        conversations=conversations,
        project_root=settings.project_root,
    )
    try:
        inventory = persist_campaign_job_plan(directory=directory, selection=selected, jobs=jobs)
    except CampaignJobPlanError as exc:
        typer.echo(json.dumps({"event": "STEP8_PLAN_JOBS", "state": "failed",
                               "code": exc.code}, sort_keys=True), err=True)
        raise
    typer.echo(json.dumps({"event": "STEP8_PLAN_JOBS", "state": "verified",
                           **inventory}, sort_keys=True), err=True)
    typer.echo(
        json.dumps(
            {
                "state": "planned",
                "campaign_dir": str(directory),
                "phase": phase.value,
                "selection": selection,
                "selected": {key.value: len(ids) for key, ids in selected.query_ids.items()},
                "added": {key.value: len(ids) for key, ids in selected.added_query_ids.items()},
                "jobs": len(jobs),
                "job_inventory": inventory,
                "model_calls": 0,
            },
            sort_keys=True,
        )
    )


def _validate_development_budget(
    manifest: BenchmarkCampaignManifest, config: Step8BenchmarkConfig, budget: Decimal,
) -> None:
    if not budget.is_finite() or budget <= 0:
        raise ValueError("API budget must be finite and positive")
    if (
        manifest.phase is CampaignPhase.DEVELOPMENT
        and budget > config.execution.development_budget_limit_usd
    ):
        raise ValueError("development API budget exceeds configured total limit")


async def _register(
    directory: Path,
    *,
    budget: Decimal | None,
    rate_card: Path | None,
    approver: str | None,
    retry: bool,
) -> tuple[str | None, dict[str, int]]:
    manifest, config = _load(directory)
    if budget is not None:
        if rate_card is None or approver is None:
            raise ValueError("API budget requires --rate-card and --approver-id")
        _preflight_rate_card(rate_card, manifest, config)
        _validate_development_budget(manifest, config, budget)
    selection = _latest_selection(directory)
    saved_budget_path = directory / "manifests" / f"budget-{selection.content_hash}.json"
    if saved_budget_path.is_file():
        saved_budget = json.loads(saved_budget_path.read_text())
        _validate_development_budget(manifest, config, Decimal(saved_budget["limit_usd"]))
    freeze_file = directory / "manifests/method-freeze.json"
    freeze = (
        MethodProtocolFreeze.model_validate_json(freeze_file.read_text())
        if freeze_file.is_file()
        else None
    )
    job_ids = json.loads(
        (directory / "manifests" / f"jobs-{selection.content_hash}.json").read_text()
    )
    jobs = [
        BenchmarkStageJob.model_validate_json(
            (directory / "private/jobs" / f"{identity}.json").read_text()
        )
        for identity in job_ids
    ]
    engine, sessions = create_session_factory(get_settings(), DatabaseRole.RUNTIME)
    try:
        repository = PostgresBenchmarkRepository(sessions)
        await repository.register(
            manifest, selection, build_experiment_matrix(manifest.protocol), freeze
        )
        await repository.add_jobs(jobs)
        approval_id = None
        if budget is not None:
            if rate_card is None or approver is None:
                raise ValueError("API budget requires --rate-card and --approver-id")
            approval_id = await repository.approve_budget(
                campaign_id=manifest.campaign_id,
                selection_hash=selection.content_hash,
                approver_id=approver,
                rate_card_hash=sha256_file(rate_card),
                limit_usd=budget,
            )
            write_json(
                directory / "manifests" / f"budget-{selection.content_hash}.json",
                {
                    "approval_id": approval_id,
                    "selection_hash": selection.content_hash,
                    "limit_usd": str(budget),
                    "approver_id": approver,
                    "rate_card_hash": sha256_file(rate_card),
                },
                immutable=True,
            )
            durable_write(
                directory / "manifests" / f"rate-card-{selection.content_hash}.json",
                rate_card.read_bytes(),
                immutable=True,
            )
        elif (directory / "manifests" / f"budget-{selection.content_hash}.json").is_file():
            approval_id = json.loads(
                (directory / "manifests" / f"budget-{selection.content_hash}.json").read_text()
            )["approval_id"]
        if retry:
            await repository.retry_failed(manifest.campaign_id)
        return approval_id, await repository.counts(manifest.campaign_id)
    finally:
        await engine.dispose()


def _preflight_rate_card(path, manifest, config) -> None:
    from pmmc.models.registry import load_model_registry
    from pmmc.models.step8_accounting import preflight_provider_rate_card

    settings = get_settings()
    report = preflight_provider_rate_card(
        path,
        load_model_registry(settings.resolve_path(config.model_registry)),
        manifest.protocol.answer_backbones,
    )
    typer.echo(json.dumps({"event": "STEP8_RATE_CARD_PREFLIGHT", **report}, sort_keys=True))


@app.command("run")
def run(
    campaign_dir: Path = typer.Option(..., exists=True, file_okay=False),
    resource: ResourceClass | None = typer.Option(None),
    worker_pool: Literal["all", "control", "gpu"] = typer.Option("all"),
    stage: list[str] | None = typer.Option(None),
    budget_usd: str | None = typer.Option(None),
    rate_card: Path | None = typer.Option(None, exists=True, dir_okay=False),
    approver_id: str | None = typer.Option(None),
    retry_failed: bool = typer.Option(False, hidden=True),
    export_on_complete: bool = typer.Option(True),
    exploration_scope: str | None = typer.Option(None, hidden=True),
) -> None:
    """Run the production DAG, or one explicitly selected resource worker."""
    with campaign_failure_boundary(campaign_dir, stage="startup"):
        _run_campaign(
            campaign_dir=campaign_dir,
            resource=resource,
            worker_pool=worker_pool,
            stage=stage,
            budget_usd=budget_usd,
            rate_card=rate_card,
            approver_id=approver_id,
            retry_failed=retry_failed,
            export_on_complete=export_on_complete,
            exploration_scope=exploration_scope,
        )


def _run_campaign(
    *,
    campaign_dir: Path,
    resource: ResourceClass | None,
    worker_pool: Literal["all", "control", "gpu"],
    stage: list[str] | None,
    budget_usd: str | None,
    rate_card: Path | None,
    approver_id: str | None,
    retry_failed: bool,
    export_on_complete: bool,
    exploration_scope: str | None = None,
) -> None:
    manifest, config = _load(campaign_dir)
    if resource is not None and worker_pool != "all":
        raise typer.BadParameter("--worker-pool and --resource cannot be combined")
    job_ids = None
    registered_counts: dict[str, int]
    if manifest.phase is CampaignPhase.EXPLORATION or exploration_scope:
        from pmmc.experiments.release_scope import read_scope

        if (manifest.phase is not CampaignPhase.EXPLORATION or not exploration_scope
                or resource is None or retry_failed or budget_usd is not None):
            raise typer.BadParameter("use run-exploration for a scoped, budget-bound stage")
        selection_hash = _latest_selection(campaign_dir).content_hash
        job_ids = read_scope(campaign_dir, exploration_scope, manifest.campaign_id, selection_hash)
        budget_path = campaign_dir / "manifests" / f"budget-{selection_hash}.json"
        approval = os.getenv("PMMC_STEP8_BUDGET_APPROVAL_ID")
        if approval and (not budget_path.exists()
                         or json.loads(budget_path.read_bytes())["approval_id"] != approval):
            raise ValueError("explicit_approval_does_not_match_exploration_selection")
        registered_counts = {}
    else:
        approval, registered_counts = asyncio.run(
            _register(campaign_dir, budget=Decimal(budget_usd) if budget_usd else None,
                      rate_card=rate_card, approver=approver_id, retry=retry_failed)
        )
    if registered_counts and registered_counts.get("completed", 0) == sum(
        registered_counts.values()
    ):
        verification = asyncio.run(verify_campaign(campaign_dir))
        if not verification["pipeline_complete"]:
            raise ValueError("completed_campaign_failed_readonly_verification")
        result: dict[str, object] = {
            "state": "verified_only",
            "campaign_complete": True,
            "model_calls": 0,
            "physical_startups": {},
        }
        if export_on_complete and worker_pool != "gpu" and resource in {None, ResourceClass.CPU}:
            result["audit"] = asyncio.run(finalize_campaign(campaign_dir))
        typer.echo(json.dumps(result, sort_keys=True))
        audit = result.get("audit")
        if isinstance(audit, dict) and not audit["inventory_complete"]:
            raise typer.Exit(1)
        return
    if resource is ResourceClass.API and approval is None:
        raise typer.BadParameter("API worker requires explicit selection budget approval")
    if rate_card is None:
        saved_card = (
            campaign_dir
            / "manifests"
            / f"rate-card-{_latest_selection(campaign_dir).content_hash}.json"
        )
        if saved_card.is_file():
            rate_card = saved_card
    if approval:
        if rate_card is None:
            raise typer.BadParameter("approved API budget requires its immutable rate card")
        _preflight_rate_card(rate_card, manifest, config)
    _preflight_response_journal(get_settings().resolve_path(config.output_root))
    from pmmc.experiments.step8_prefix import require_real_prefix

    asyncio.run(
        require_real_prefix(
            get_settings(),
            manifest=manifest,
            output_root=get_settings().resolve_path(config.output_root),
            target_directory=campaign_dir,
        )
    )
    from pmmc.experiments.step8_exposure import record_campaign_exposure

    exposure_hash = record_campaign_exposure(
        get_settings().resolve_path(config.output_root), campaign_dir
    )
    typer.echo(
        json.dumps(
            {"event": "STEP8_EXPOSURE", "state": "recorded", "receipt_hash": exposure_hash},
            sort_keys=True,
        )
    )
    if approval:
        os.environ["PMMC_STEP8_BUDGET_APPROVAL_ID"] = approval
    if rate_card:
        os.environ["PMMC_STEP8_RATE_CARD_PATH"] = str(rate_card.resolve())
    os.environ["PMMC_STEP8_OUTPUT_ROOT"] = str(get_settings().resolve_path(config.output_root))
    attempt_id = "attempt-" + uuid.uuid4().hex
    os.environ["PMMC_STEP8_ATTEMPT_ID"] = attempt_id
    get_settings.cache_clear()

    if resource is None:
        from pmmc.experiments.step8_backend import ProductionComponentBackend
        from pmmc.experiments.step8_supervisor import supervise_campaign

        if stage:
            raise typer.BadParameter("--stage requires an explicit --resource worker")
        if approval is None:
            raise typer.BadParameter("the four-backbone campaign requires an explicit API budget")
        settings = get_settings()
        backend = ProductionComponentBackend(
            settings=settings,
            config=config,
            campaign=manifest,
            campaign_dir=campaign_dir,
            store=CampaignArtifactStore(settings.resolve_path(config.output_root)),
        )
        result = supervise_campaign(
            settings=backend.runtime_settings,
            config=config,
            manifest=manifest,
            directory=campaign_dir,
            has_api_budget=True,
            rate_card=rate_card,
            resources=(
                frozenset({ResourceClass.CPU, ResourceClass.API})
                if worker_pool == "control"
                else frozenset(
                    {ResourceClass.GPU_9B, ResourceClass.GPU_27B, ResourceClass.GPU_EMBEDDING}
                )
                if worker_pool == "gpu"
                else None
            ),
        )
        if export_on_complete and worker_pool != "gpu" and result.get("campaign_complete"):
            result["audit"] = asyncio.run(finalize_campaign(campaign_dir))
        typer.echo(json.dumps(result, sort_keys=True))
        audit = result.get("audit")
        if isinstance(audit, dict) and not audit["inventory_complete"]:
            raise typer.Exit(1)
        return

    async def execute() -> dict:
        from pmmc.experiments.step8_backend import ProductionComponentBackend
        from pmmc.experiments.step8_checkpoint import CampaignCheckpointCoordinator
        from pmmc.experiments.step8_runner import CampaignWorker

        settings = get_settings()
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        primary: BaseException | None = None
        try:
            store = CampaignArtifactStore(settings.resolve_path(config.output_root))
            backend = ProductionComponentBackend(
                settings=settings,
                config=config,
                campaign=manifest,
                campaign_dir=campaign_dir,
                store=store,
            )
            aliases = conformance_aliases(resource)
            if aliases and manifest.phase is not CampaignPhase.EXPLORATION:
                from pmmc.experiments.step8_conformance import run_step8_conformance
                from pmmc.models.registry import load_model_registry

                receipt = await run_step8_conformance(
                    settings=backend.settings,
                    registry=load_model_registry(backend.settings.resolved_model_registry_path),
                    protocol_hash=manifest.protocol.content_hash,
                    campaign_id=manifest.campaign_id,
                    selection_hash=_latest_selection(campaign_dir).content_hash,
                    output_root=settings.resolve_path(config.output_root),
                    probe_aliases=aliases,
                )
                write_json(
                    campaign_dir / "private/conformance" / f"{receipt.run_id}.json",
                    receipt.model_dump(mode="json"),
                    immutable=True,
                )
                typer.echo(
                    json.dumps(
                        {
                            "event": "STEP8_CONFORMANCE",
                            "state": "ready",
                            "aliases": list(aliases),
                            "probes": len(receipt.probes),
                            "scientific_assignments": 0,
                            "scope": "transport_and_synthetic_contracts_not_scientific_acceptance",
                        },
                        sort_keys=True,
                    )
                )
            worker = CampaignWorker(
                repository=PostgresBenchmarkRepository(
                    sessions, tuple(stage or ()), job_ids=job_ids
                ),
                backend=backend,
                campaign_id=manifest.campaign_id,
                campaign_dir=campaign_dir,
                resource=resource,
                execution=config.execution,
                artifact_store=store,
                checkpoint=(
                    CampaignCheckpointCoordinator(backend.settings, manifest, campaign_dir, store)
                    if resource is ResourceClass.CPU
                    else None
                ),
                execution_image=os.environ.get("PMMC_CONTAINER_IMAGE", "local-source"),
                attempt_id=attempt_id,
                stages=tuple(stage or ()),
            )
            return await worker.run()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                await engine.dispose()
            except BaseException as secondary:
                if primary is None:
                    raise
                primary.add_note("worker engine disposal failed: " + type(secondary).__name__)

    typer.echo(json.dumps(asyncio.run(execute()), sort_keys=True))


def _preflight_response_journal(output_root: Path) -> dict[str, object]:
    from pmmc.experiments.step8_failure import failure_diagnostics

    event = {"event": "STEP8_JOURNAL_PREFLIGHT", "provider_calls": 0}
    typer.echo(json.dumps({**event, "state": "start"}, sort_keys=True))
    try:
        report = CampaignArtifactStore(output_root).preflight_response_journal()
    except Exception as exc:
        typer.echo(json.dumps({
            **event, "state": "failed", "diagnostics": failure_diagnostics(exc),
        }, sort_keys=True))
        raise
    typer.echo(json.dumps({**event, **report}, sort_keys=True))
    return report


@app.command("resume")
def resume(
    campaign_dir: Path = typer.Option(..., exists=True, file_okay=False),
    resource: ResourceClass | None = typer.Option(None),
    worker_pool: Literal["all", "control", "gpu"] = typer.Option("all"),
    restore_checkpoint: Path | None = typer.Option(None, exists=True, file_okay=False),
) -> None:
    if restore_checkpoint is not None:
        from pmmc.experiments.step8_backend import ProductionComponentBackend
        from pmmc.experiments.step8_checkpoint import CampaignCheckpointCoordinator

        manifest, config = _load(campaign_dir)
        settings = get_settings()
        store = CampaignArtifactStore(settings.resolve_path(config.output_root))
        backend = ProductionComponentBackend(
            settings=settings,
            config=config,
            campaign=manifest,
            campaign_dir=campaign_dir,
            store=store,
        )
        asyncio.run(
            CampaignCheckpointCoordinator(
                backend.settings, manifest, campaign_dir, store
            ).restore_empty(restore_checkpoint)
        )
    run(
        campaign_dir=campaign_dir,
        resource=resource,
        worker_pool=worker_pool,
        stage=None,
        budget_usd=None,
        rate_card=None,
        approver_id=None,
        retry_failed=True,
        export_on_complete=True,
    )


async def verify_campaign(directory: Path) -> dict:
    from pmmc.experiments.step8_verification import verify_components

    manifest, config = _load(directory)
    engine, sessions = create_session_factory(get_settings(), DatabaseRole.RUNTIME)
    store = CampaignArtifactStore(get_settings().resolve_path(config.output_root))
    try:
        jobs = await PostgresBenchmarkRepository(sessions).list_jobs(manifest.campaign_id)
        chain = verify_components(directory=directory, campaign=manifest, rows=jobs, store=store)
        summary = {
            "protocol": "step8-verification-v1",
            "campaign_id": manifest.campaign_id,
            "protocol_hash": manifest.protocol.content_hash,
            "selection_hash": _latest_selection(directory).content_hash,
            **chain,
            "approval": "not_automatic",
            "model_calls": 0,
        }
        write_json(directory / "reports/verification.json", summary)
        return summary
    finally:
        await engine.dispose()


async def finalize_campaign(directory: Path) -> dict:
    """CPU-only review bundle; completion never grants scientific approval."""
    from pmmc.experiments.step8_analysis import analyze_campaign
    from pmmc.experiments.step8_audit import export_campaign
    from pmmc.experiments.step8_inventory import build_campaign_audit_inventory

    verification = await verify_campaign(directory)
    if not verification["pipeline_complete"]:
        raise ValueError("campaign_components_incomplete_before_export")
    _, config = _load(directory)
    settings = get_settings()
    store = CampaignArtifactStore(settings.resolve_path(config.output_root))
    inventory = await build_campaign_audit_inventory(settings, directory, store)
    analyze_campaign(directory, artifact_store=store)
    exported = export_campaign(
        directory,
        visibility="private",
        private_authorized=True,
        artifact_store=store,
        split_bytes=config.execution.transfer_part_bytes,
    )
    return {"inventory_complete": inventory.complete, "scientific_approval": False, **exported}


@app.command("verify")
def verify(campaign_dir: Path = typer.Option(..., exists=True, file_okay=False)) -> None:
    result = asyncio.run(verify_campaign(campaign_dir))
    typer.echo(json.dumps(result, sort_keys=True))
    if not result["pipeline_complete"]:
        raise typer.Exit(1)


@app.command("approve")
def approve(
    campaign_dir: Path = typer.Option(..., exists=True, file_okay=False),
    reviewer_id: str = typer.Option(...),
    rationale_file: Path = typer.Option(..., exists=True),
    match_threshold: float | None = typer.Option(None),
    gap_threshold: float | None = typer.Option(None),
    no_reranker_match_threshold: float | None = typer.Option(None),
    no_reranker_gap_threshold: float | None = typer.Option(None),
    text_relevance_threshold: float | None = typer.Option(None),
    multimodal_relevance_threshold: float | None = typer.Option(None),
) -> None:
    manifest, config = _load(campaign_dir)
    if manifest.phase is not CampaignPhase.DEVELOPMENT:
        raise typer.BadParameter("only development can approve a method protocol")
    verification = asyncio.run(verify_campaign(campaign_dir))
    from pmmc.experiments.step8_approval import approval_evidence
    from pmmc.experiments.step8_inventory import build_campaign_audit_inventory

    store = CampaignArtifactStore(get_settings().resolve_path(config.output_root))
    # Re-read authoritative facts; never approve a hand-edited inventory boolean.
    policy_updates = {
        "match_threshold": match_threshold,
        "gap_threshold": gap_threshold,
        "no_reranker_match_threshold": no_reranker_match_threshold,
        "no_reranker_gap_threshold": no_reranker_gap_threshold,
        "text_relevance_threshold": text_relevance_threshold,
        "multimodal_relevance_threshold": multimodal_relevance_threshold,
    }
    selected_policy = type(manifest.protocol).model_validate(
        {
            **manifest.protocol.model_dump(),
            **{k: v for k, v in policy_updates.items() if v is not None},
        }
    )
    changed_relevance = any(
        getattr(selected_policy, k) != getattr(manifest.protocol, k)
        for k in ("text_relevance_threshold", "multimodal_relevance_threshold")
    )
    inventory_options = {"selected_policy": selected_policy} if changed_relevance else {}
    inventory = asyncio.run(
        build_campaign_audit_inventory(get_settings(), campaign_dir, store, **inventory_options)
    )
    evidence = approval_evidence(
        campaign_dir,
        store,
        fresh_inventory=inventory,
        match_threshold=match_threshold,
        gap_threshold=gap_threshold,
        no_reranker_match_threshold=no_reranker_match_threshold,
        no_reranker_gap_threshold=no_reranker_gap_threshold,
        text_relevance_threshold=text_relevance_threshold,
        multimodal_relevance_threshold=multimodal_relevance_threshold,
    )
    if (
        evidence["protocol_hash"] != manifest.protocol.content_hash
        or not verification["pipeline_complete"]
    ):
        raise ValueError("approval evidence does not bind the completed campaign")
    freeze = MethodProtocolFreeze(
        protocol=evidence["selected_protocol"],
        threshold_approval=evidence["threshold_approval"],
        decision="approved",
        reviewer_id=reviewer_id,
        rationale_sha256=sha256_file(rationale_file),
        approved_at=datetime.now(UTC),
        development_receipt_hash=evidence["regression_receipt_hash"],
        confirmation_receipt_hash=evidence["confirmation_receipt_hash"],
        protocol_integrity_passed=evidence["protocol_integrity_passed"],
        scientific_viability_passed=evidence["scientific_viability_passed"],
        exposure_complete=evidence["exposure_complete"],
    )
    selection_evidence = evidence["threshold_approval"]
    write_json(
        campaign_dir / f"manifests/threshold-approvals/{selection_evidence.content_hash}.json",
        selection_evidence.model_dump(mode="json"),
        immutable=True,
    )
    for name, receipt in evidence["cohort_receipts"].items():
        digest = selection_evidence.cohort_receipt_hashes[name]
        write_json(
            campaign_dir / f"manifests/threshold-receipts/{digest}.json",
            receipt,
            immutable=True,
        )
    write_json(
        campaign_dir / "manifests/method-freeze.json",
        freeze.model_dump(mode="json"),
        immutable=True,
    )
    typer.echo(json.dumps({"state": "approved", "freeze_hash": freeze.content_hash}))


@app.command("analyze")
def analyze(
    campaign_dir: Path = typer.Option(..., exists=True, file_okay=False),
    selection_hash: str | None = typer.Option(None),
) -> None:
    """CPU-only paired analysis of saved facts; never invokes a model."""
    from pmmc.experiments.step8_analysis import analyze_campaign

    _, config = _load(campaign_dir)
    report = analyze_campaign(
        campaign_dir,
        selection_hash=selection_hash,
        artifact_store=CampaignArtifactStore(get_settings().resolve_path(config.output_root)),
    )
    # The full report is persisted; keep the interactive log bounded.
    typer.echo(
        json.dumps(
            {
                "state": "analyzed",
                "scientific_approval": False,
                "model_calls": 0,
                "report_keys": sorted(report),
            },
            sort_keys=True,
        )
    )


@app.command("export")
def export(
    campaign_dir: Path = typer.Option(..., exists=True, file_okay=False),
    visibility: Literal["public", "private"] = typer.Option(...),
    private_authorized: bool = typer.Option(False),
    public_review_file: Path | None = typer.Option(None, exists=True, dir_okay=False),
    inventory_path: Path | None = typer.Option(None, exists=True, dir_okay=False),
    selection_hash: str | None = typer.Option(None),
    refresh_inventory: bool = typer.Option(False),
) -> None:
    """Export checked artifacts; private payload and public publication require explicit review."""
    from pmmc.experiments.step8_audit import export_campaign
    from pmmc.experiments.step8_export import PublicReview

    _, config = _load(campaign_dir)
    store = CampaignArtifactStore(get_settings().resolve_path(config.output_root))
    if refresh_inventory:
        if not private_authorized or inventory_path is not None:
            raise typer.BadParameter(
                "--refresh-inventory requires --private-authorized and no --inventory-path"
            )
        from pmmc.experiments.step8_inventory import build_campaign_audit_inventory

        asyncio.run(build_campaign_audit_inventory(get_settings(), campaign_dir, store))
    review = (
        PublicReview(**json.loads(public_review_file.read_text())) if public_review_file else None
    )
    report = export_campaign(
        campaign_dir,
        visibility=visibility,
        private_authorized=private_authorized,
        public_review=review,
        inventory_path=inventory_path,
        selection_hash=selection_hash,
        artifact_store=store,
        split_bytes=config.execution.transfer_part_bytes,
    )
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("gc")
def gc(
    campaign_dir: Path = typer.Option(..., exists=True, file_okay=False),
    shared_cas: bool = typer.Option(False),
    apply: bool = typer.Option(False),
) -> None:
    """Preview shared CAS collection, or clean explicitly owned terminal scratch."""
    _, config = _load(campaign_dir)
    store = CampaignArtifactStore(get_settings().resolve_path(config.output_root))
    reports = []
    if apply and not shared_cas:
        raise typer.BadParameter("--apply requires --shared-cas")
    cas_report = None
    if shared_cas:
        from pmmc.experiments.step8_gc import collect_campaign_cas

        cas_report = collect_campaign_cas(store, apply=apply)
    for receipt in sorted((campaign_dir / "attempts").glob("*/terminal.json")):
        identity = json.loads(receipt.read_text())["attempt_id"]
        reports.append(store.clean_attempt(identity, durable_receipt=receipt))
    typer.echo(
        json.dumps(
            {
                "state": "safe_cleanup",
                "attempts": reports,
                "shared_cas_deleted": bool(shared_cas and apply),
                "cas": cas_report,
                "model_weights_deleted": False,
            }
        )
    )
