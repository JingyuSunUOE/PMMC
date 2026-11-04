"""PMMC command line interface."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NoReturn, NotRequired, TypedDict

import typer

from pmmc import __version__
from pmmc.benchmarks import BenchmarkAdapter, BenchmarkDataError, create_adapter
from pmmc.benchmarks.repositories import GoldStore, MemoryCorpus, RuntimeQueryStore
from pmmc.benchmarks.snapshot import build_snapshot, find_snapshot, load_manifest
from pmmc.benchmarks.taxonomy import EXCLUDED_SOURCE_CATEGORIES
from pmmc.benchmarks.validation import ProgressCallback, validate_adapter
from pmmc.experiments import (
    CalibrationPolicy,
    DownloadPolicy,
    PipelineRunContext,
    PipelineStage,
    make_run_id,
)
from pmmc.experiments.step8_cli import app as benchmark_app
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.settings import AppMode, Settings, get_settings
from pmmc.models.registry import ModelRegistry, bind_registry_to_manifest, load_model_registry
from pmmc.schemas import (
    DEFAULT_EVALUATION_POLICY_VERSION,
    BaselineMethod,
    BenchmarkSelection,
    DataScope,
    DatasetName,
    ModelVerificationPolicy,
    PartitionRole,
    QuestionType,
    RunManifest,
    ValidationLevel,
    ValidationReport,
)
from pmmc.schemas.base import JsonDict


class DoctorCheck(TypedDict):
    """One non-sensitive diagnostic result."""

    ok: bool
    value: str
    expected: NotRequired[str]


app = typer.Typer(
    name="pmmc",
    help="Prospective Multimodal Memory Compilation research platform.",
    no_args_is_help=True,
)
db_app = typer.Typer(help="Postgres canonical and experiment persistence.")
storage_app = typer.Typer(help="Private snapshot, asset, and experiment artifact storage.")
baseline_app = typer.Typer(help="Shared corpus and context foundation for baseline methods.")
models_app = typer.Typer(help="Model registry and provider endpoint diagnostics.")
runtime_app = typer.Typer(help="Managed local vLLM lifecycle and GPU planning.")
evaluate_app = typer.Typer(help="Gold-isolated baseline evaluation and exports.")
smoke_app = typer.Typer(help="Self-contained, non-paper acceptance fixtures.")
experiment_app = typer.Typer(help="One-command reproducible research experiments.")
compiler_app = typer.Typer(help="PMMC compiler contracts and operator diagnostics.")
pilot_app = typer.Typer(help="Questioner quality pilots, human review, and approval.")
app.add_typer(db_app, name="db")
app.add_typer(storage_app, name="storage")
app.add_typer(baseline_app, name="baseline")
app.add_typer(models_app, name="models")
app.add_typer(runtime_app, name="runtime")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(smoke_app, name="smoke")
app.add_typer(experiment_app, name="experiment")
app.add_typer(compiler_app, name="compiler")
app.add_typer(pilot_app, name="pilot")
app.add_typer(benchmark_app, name="benchmark")


def _persist_managed_child_diagnostic(
    *,
    event: str,
    exc: BaseException,
    diagnostics: Mapping[str, object] | None = None,
) -> None:
    """Append one secret-free child diagnostic for the managed parent process."""

    path_value = os.environ.get("PMMC_MANAGED_CHILD_DIAGNOSTIC_PATH")
    if not path_value:
        return
    payload: dict[str, object] = {
        "command_count": os.environ.get("PMMC_MANAGED_CHILD_COMMAND_COUNT"),
        "command_ordinal": os.environ.get("PMMC_MANAGED_CHILD_COMMAND_ORDINAL"),
        "created_at_unix": round(time.time(), 6),
        "error_type": type(exc).__name__,
        "event": event,
    }
    if diagnostics is not None:
        payload["diagnostics"] = dict(diagnostics)
    safe_summary = getattr(exc, "safe_summary", None)
    if isinstance(safe_summary, str) and safe_summary:
        payload["safe_summary"] = safe_summary
    if diagnostics is None:
        from pmmc.infrastructure.database.errors import (
            safe_database_error_diagnostics,
        )

        database = safe_database_error_diagnostics(exc)
        if "database_error_type" in database:
            payload["diagnostics"] = database
    path = Path(path_value)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            line = (
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            os.write(descriptor, line)
        finally:
            os.close(descriptor)
        os.chmod(path, 0o600)
    except OSError as write_error:
        typer.echo(
            "MANAGED_CHILD_DIAGNOSTIC_WRITE_FAILED "
            + json.dumps(
                {"error_type": type(write_error).__name__, "event": event},
                sort_keys=True,
                separators=(",", ":"),
            ),
            err=True,
        )


def _raise_compiler_workflow_failure(
    exc: BaseException,
    *,
    program_target: bool,
) -> NoReturn:
    """Report a runtime compiler failure without echoing provider or Pydantic input."""

    producer_name = "Compiler program" if program_target else "Questioner"
    safe_diagnostics = getattr(exc, "safe_diagnostics", None)
    persisted_diagnostics: Mapping[str, object] | None = None
    persisted_event = "COMPILER_PROGRAM_FAILURE"
    if callable(safe_diagnostics):
        diagnostic_payload = safe_diagnostics()
        if isinstance(diagnostic_payload, Mapping):
            persisted_diagnostics = diagnostic_payload
            diagnostic_event = getattr(
                exc,
                "diagnostic_event",
                "COMPILER_PROGRAM_FAILURE",
            )
            if not isinstance(diagnostic_event, str) or not diagnostic_event:
                diagnostic_event = "COMPILER_PROGRAM_FAILURE"
            persisted_event = diagnostic_event
            typer.echo(
                diagnostic_event
                + " "
                + json.dumps(
                    dict(diagnostic_payload),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                err=True,
            )
    _persist_managed_child_diagnostic(
        event=persisted_event,
        exc=exc,
        diagnostics=persisted_diagnostics,
    )
    safe_summary = getattr(exc, "safe_summary", None)
    if not isinstance(safe_summary, str) or not safe_summary:
        from pmmc.infrastructure.database.errors import (
            safe_database_error_diagnostics,
            safe_database_error_summary,
        )

        database_diagnostics = safe_database_error_diagnostics(exc)
        if "database_error_type" in database_diagnostics:
            safe_summary = safe_database_error_summary(database_diagnostics)
    suffix = f": {safe_summary}" if isinstance(safe_summary, str) and safe_summary else ""
    typer.echo(
        f"ERROR: {producer_name} run failed ({type(exc).__name__}){suffix}",
        err=True,
    )
    raise typer.Exit(code=1) from exc


@pilot_app.command("build-partition")
def pilot_build_partition(
    snapshot: list[str] = typer.Option(..., "--snapshot"),
    config: Path = typer.Option(
        Path("configs/compiler/questioner-quality-pilot-v1.yaml"), "--config"
    ),
    output_root: Path | None = typer.Option(None, "--output-root"),
) -> None:
    """Build a deterministic conversation-level development/paper-test split."""

    from pmmc.pilots.config import load_questioner_pilot_config
    from pmmc.pilots.partition import build_partition
    from pmmc.schemas import BenchmarkPartitionSpec

    settings = get_settings()
    try:
        pilot_config = load_questioner_pilot_config(settings.resolve_path(config))
        spec = BenchmarkPartitionSpec(
            protocol_version=pilot_config.partition_protocol,
            fold_count=pilot_config.fold_count,
            development_fold=pilot_config.development_fold,
            seed=pilot_config.partition_seed,
            eligibility_policy_version=DEFAULT_EVALUATION_POLICY_VERSION,
        )
        destination, manifest = build_partition(
            snapshot_dirs=[
                find_snapshot(settings.benchmark_artifacts_path, item) for item in snapshot
            ],
            output_root=(
                settings.resolve_path(output_root)
                if output_root is not None
                else settings.artifacts_path / "pilots" / "partitions"
            ),
            spec=spec,
        )
    except Exception as exc:
        typer.echo(f"ERROR: pilot partition build failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {"partition_dir": str(destination), **manifest.model_dump(mode="json")},
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


@pilot_app.command("verify-partition")
def pilot_verify_partition(
    partition_dir: Path = typer.Option(..., "--partition-dir"),
) -> None:
    """Verify partition checksums, fold roles, and conversation isolation."""

    from pmmc.pilots.partition import verify_partition

    try:
        manifest = verify_partition(get_settings().resolve_path(partition_dir))
    except Exception as exc:
        typer.echo(f"ERROR: pilot partition verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))


@pilot_app.command("verify")
def pilot_verify(pilot_dir: Path = typer.Option(..., "--pilot-dir")) -> None:
    """Verify a signed public/private Questioner quality-pilot bundle."""

    from pmmc.pilots.bundle import verify_pilot_bundle

    try:
        manifest = verify_pilot_bundle(get_settings().resolve_path(pilot_dir))
    except Exception as exc:
        typer.echo(f"ERROR: pilot verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))


@pilot_app.command("render-review")
def pilot_render_review(
    pilot_dir: Path = typer.Option(..., "--pilot-dir"),
    benchmark_root: Path = typer.Option(..., "--benchmark-root"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    """Render an ephemeral private HTML workspace outside the signed bundle."""

    from pmmc.pilots.bundle import render_review_workspace

    settings = get_settings()
    resolved_pilot = settings.resolve_path(pilot_dir)
    resolved_output = (
        settings.resolve_path(output_dir)
        if output_dir is not None
        else resolved_pilot.parent / ".review-workspaces" / resolved_pilot.name
    )
    try:
        result = render_review_workspace(
            pilot_dir=resolved_pilot,
            benchmark_root=settings.resolve_benchmark_path(benchmark_root),
            output_dir=resolved_output,
        )
    except Exception as exc:
        typer.echo(f"ERROR: review rendering failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(str(result / "index.html"))


@pilot_app.command("validate-review")
def pilot_validate_review(
    pilot_dir: Path = typer.Option(..., "--pilot-dir"),
    annotations: Path = typer.Option(..., "--annotations"),
    role: str = typer.Option(..., "--role"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Validate task hashes and normalize one complete reviewer annotation file."""

    from pmmc.pilots.approval import validate_annotations
    from pmmc.pilots.io import write_jsonl
    from pmmc.schemas import ReviewRole

    settings = get_settings()
    try:
        review_role = ReviewRole(role)
        result = validate_annotations(
            pilot_dir=settings.resolve_path(pilot_dir),
            annotation_path=settings.resolve_path(annotations),
            role=review_role,
        )
        if output is not None:
            write_jsonl(
                settings.resolve_path(output),
                [item.model_dump(mode="json") for item in result],
            )
    except Exception as exc:
        typer.echo(f"ERROR: review validation failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({"valid": True, "count": len(result), "role": role}, indent=2))


@pilot_app.command("finalize")
def pilot_finalize(
    pilot_dir: Path = typer.Option(..., "--pilot-dir"),
    primary: Path = typer.Option(..., "--primary"),
    secondary: Path = typer.Option(..., "--secondary"),
    adjudications: Path | None = typer.Option(None, "--adjudications"),
    output_root: Path | None = typer.Option(None, "--output-root"),
) -> None:
    """Finalize complete primary/secondary review and all core adjudications."""

    from pmmc.pilots.approval import finalize_review

    settings = get_settings()
    try:
        result = finalize_review(
            pilot_dir=settings.resolve_path(pilot_dir),
            primary_path=settings.resolve_path(primary),
            secondary_path=settings.resolve_path(secondary),
            adjudications_path=(
                settings.resolve_path(adjudications) if adjudications is not None else None
            ),
            output_root=(settings.resolve_path(output_root) if output_root is not None else None),
        )
    except Exception as exc:
        typer.echo(f"ERROR: review finalization failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({"review_dir": str(result), "status": "finalized"}, indent=2))


@pilot_app.command("decide")
def pilot_decide(
    pilot_dir: Path = typer.Option(..., "--pilot-dir"),
    review_dir: Path | None = typer.Option(None, "--review-dir"),
    decision: str = typer.Option(..., "--decision"),
    reviewer_id: str = typer.Option(..., "--reviewer-id"),
    rationale_file: Path = typer.Option(..., "--rationale-file"),
) -> None:
    """Record an explicit research decision and emit a freeze on approval."""

    from pmmc.pilots.approval import decide_pilot
    from pmmc.schemas import PilotDecisionValue

    settings = get_settings()
    try:
        decision_dir, freeze = decide_pilot(
            pilot_dir=settings.resolve_path(pilot_dir),
            review_dir=(settings.resolve_path(review_dir) if review_dir is not None else None),
            decision=PilotDecisionValue(decision),
            reviewer_id=reviewer_id,
            rationale=settings.resolve_path(rationale_file).read_text(encoding="utf-8").strip(),
        )
    except Exception as exc:
        typer.echo(f"ERROR: pilot decision failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "decision_dir": str(decision_dir),
                "freeze_manifest": str(freeze) if freeze is not None else None,
            },
            indent=2,
        )
    )


@pilot_app.command("verify-approval")
def pilot_verify_approval(
    approval: Path = typer.Option(..., "--approval"),
) -> None:
    """Verify an immutable Questioner Freeze Manifest before Planner use."""

    from pmmc.pilots.approval import verify_approval

    try:
        result = verify_approval(get_settings().resolve_path(approval))
    except Exception as exc:
        typer.echo(
            f"ERROR: Questioner approval verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@pilot_app.command("compare")
def pilot_compare(
    left: Path = typer.Option(..., "--left"),
    right: Path = typer.Option(..., "--right"),
) -> None:
    """Compare two public pilot reports without exposing private review data."""

    from pmmc.pilots.bundle import verify_pilot_bundle
    from pmmc.pilots.io import read_json

    settings = get_settings()
    left_dir = settings.resolve_path(left)
    right_dir = settings.resolve_path(right)
    try:
        left_manifest = verify_pilot_bundle(left_dir)
        right_manifest = verify_pilot_bundle(right_dir)
        left_summary = read_json(left_dir / "public" / "automatic_summary.json")
        right_summary = read_json(right_dir / "public" / "automatic_summary.json")
    except Exception as exc:
        typer.echo(f"ERROR: pilot comparison failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "left": left_manifest.pilot_id,
                "right": right_manifest.pilot_id,
                "same_partition": left_manifest.partition_id == right_manifest.partition_id,
                "datasets": {
                    key: {
                        "left": left_summary.get("datasets", {}).get(key),
                        "right": right_summary.get("datasets", {}).get(key),
                    }
                    for key in sorted(
                        set(left_summary.get("datasets", {}))
                        | set(right_summary.get("datasets", {}))
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


@compiler_app.command("validate-config")
def compiler_validate_config(
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Validate the frozen compiler foundation configuration."""

    from pmmc.compiler import compiler_config_hash, load_compiler_config

    settings = get_settings()
    try:
        path = settings.resolve_path(config) if config else settings.resolved_compiler_config_path
        value = load_compiler_config(path)
    except Exception as exc:
        typer.echo(f"ERROR: compiler config validation failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "config_path": str(path),
                "config_hash": compiler_config_hash(value),
                "operator_set_version": value.operator_set_version,
                "operator_count": len(value.operators),
                "valid": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


@compiler_app.command("validate-questioner-config")
def compiler_validate_questioner_config(
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Validate the frozen Questioner, QBank index, and coverage configuration."""

    from pmmc.compiler.questioner_config import (
        load_questioner_config,
        questioner_config_hash,
    )
    from pmmc.compiler.questioner_prompt import questioner_prompt_sha256

    settings = get_settings()
    try:
        path = settings.resolve_path(config) if config else settings.resolved_questioner_config_path
        value = load_questioner_config(path)
    except Exception as exc:
        typer.echo(
            f"ERROR: Questioner config validation failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "config_path": str(path),
                "config_hash": questioner_config_hash(value),
                "prompt_hash": questioner_prompt_sha256(value.prompt_version),
                "question_budgets": value.question_budgets,
                "match_ks": value.match_ks,
                "valid": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


@compiler_app.command("list-operators")
def compiler_list_operators(
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Print the immutable typed operator registry without connecting to Postgres."""

    from pmmc.compiler import load_compiler_config
    from pmmc.retrieval.operators import build_operator_registry

    settings = get_settings()
    try:
        path = settings.resolve_path(config) if config else settings.resolved_compiler_config_path
        value = load_compiler_config(path)
        registry = build_operator_registry(object(), value)  # type: ignore[arg-type]
    except Exception as exc:
        typer.echo(f"ERROR: operator registry loading failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "operator_set_version": registry.version,
                "operator_registry_hash": registry.resolved_hash,
                "operators": [item.model_dump(mode="json") for item in registry.specs()],
            },
            indent=2,
            sort_keys=True,
        )
    )


@compiler_app.command("validate-program")
def compiler_validate_program(
    program: Path = typer.Option(..., "--program"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Run schema and static operator validation for a MemoryProgram JSON file."""

    from pmmc.compiler import load_compiler_config
    from pmmc.memory import ProgramValidator
    from pmmc.retrieval.operators import build_operator_registry
    from pmmc.schemas import MemoryProgram

    settings = get_settings()
    try:
        config_path = (
            settings.resolve_path(config) if config else settings.resolved_compiler_config_path
        )
        foundation = load_compiler_config(config_path)
        registry = build_operator_registry(object(), foundation)  # type: ignore[arg-type]
        value = MemoryProgram.model_validate_json(
            settings.resolve_path(program).read_text(encoding="utf-8")
        )
        result = ProgramValidator(registry, foundation).validate(value)
    except Exception as exc:
        typer.echo(f"ERROR: memory program validation failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(result.model_dump_json(indent=2))
    if not result.valid:
        raise typer.Exit(code=1)


def _dependency_error(component: str, exc: ModuleNotFoundError) -> NoReturn:
    typer.echo(
        f"ERROR: {component} dependencies are not installed; run pip install -e .",
        err=True,
    )
    raise typer.Exit(code=1) from exc


def _not_implemented(command_name: str) -> NoReturn:
    typer.echo(f"{command_name} is not implemented in the current PMMC phase.")
    raise typer.Exit(code=2)


def _parse_datasets(value: str) -> list[DatasetName]:
    normalized = value.strip().lower()
    if normalized == "all":
        return [DatasetName.MEMLENS, DatasetName.MEMGALLERY]
    try:
        dataset = DatasetName(normalized)
    except ValueError as exc:
        raise typer.BadParameter(
            "must be one of: all, memlens, memgallery",
            param_hint="--dataset",
        ) from exc
    if dataset is DatasetName.UNKNOWN:
        raise typer.BadParameter(
            "unknown is not a runnable benchmark",
            param_hint="--dataset",
        )
    return [dataset]


def _default_validation_level(settings: Settings) -> ValidationLevel:
    return ValidationLevel.FULL if settings.mode is AppMode.PAPER else ValidationLevel.FAST


def _adapter_for(settings: Settings, dataset: DatasetName) -> BenchmarkAdapter:
    root = settings.memlens_path if dataset is DatasetName.MEMLENS else settings.memgallery_path
    return create_adapter(dataset, root=root, project_root=settings.benchmark_asset_path)


def _run_context(
    *,
    settings: Settings,
    command: str,
    datasets: list[DatasetName],
    level: ValidationLevel,
) -> PipelineRunContext:
    run_id = make_run_id(command)
    manifest = RunManifest(
        run_id=run_id,
        experiment_name=command,
        dataset=datasets[0] if len(datasets) == 1 else DatasetName.UNKNOWN,
        method="benchmark_data_pipeline",
        paper_mode=settings.mode is AppMode.PAPER,
        evaluation_policy_version=DEFAULT_EVALUATION_POLICY_VERSION,
        evaluation_exclusion_rules={
            dataset.value: sorted(EXCLUDED_SOURCE_CATEGORIES[dataset]) for dataset in datasets
        },
        metadata={
            "datasets": [dataset.value for dataset in datasets],
            "validation_level": level.value,
        },
    )
    return PipelineRunContext(
        runs_root=settings.runs_path,
        manifest=manifest,
        datasets=datasets,
    )


def _progress_callback(stage: PipelineStage) -> ProgressCallback:
    phase_counts: dict[str, int] = {}

    def report(phase: str, current: int, _total: int | None) -> None:
        previous = phase_counts.get(phase, 0)
        if current > previous:
            stage.advance(current - previous, phase=phase)
        phase_counts[phase] = current

    return report


def _print_validation_report(report: ValidationReport) -> None:
    counts = report.counts
    error_count = sum(issue.severity.value == "error" for issue in report.issues)
    warning_count = sum(issue.severity.value == "warning" for issue in report.issues)
    typer.echo(
        f"{report.dataset.value.upper()} validation: "
        f"{'PASS' if report.success else 'FAIL'} ({report.level.value})"
    )
    typer.echo(
        "  "
        f"queries={counts.get('queries', 0)} "
        f"eligible={counts.get('eligible_queries', 0)} "
        f"excluded={counts.get('excluded_queries', 0)} "
        f"sessions={counts.get('sessions', 0)} "
        f"memory_units={counts.get('memory_units', 0)} "
        f"assets={counts.get('assets', 0)}"
    )
    typer.echo(
        f"  warnings={warning_count} errors={error_count} "
        f"categories={json.dumps(report.metadata.get('category_counts', {}), sort_keys=True)}"
    )
    for issue in report.issues:
        typer.echo(f"  {issue.severity.value.upper()} {issue.code}: {issue.message}")


@app.command()
def version() -> None:
    """Print the PMMC package version."""

    typer.echo(__version__)


@smoke_app.command("bootstrap-step5d")
def smoke_bootstrap_step5d(
    confirm_ephemeral: bool = typer.Option(False, "--confirm-ephemeral"),
    result_json: Path = typer.Option(..., "--result-json"),
) -> None:
    """Build a one-query, five-method fake suite in loopback-only services."""

    from pmmc.smoke import bootstrap_step5d_fixture, write_smoke_result

    settings = get_settings()
    try:
        result = asyncio.run(
            bootstrap_step5d_fixture(
                settings,
                confirm_ephemeral=confirm_ephemeral,
            )
        )
        write_smoke_result(result_json, result)
    except Exception as exc:
        typer.echo(
            f"ERROR: Step 5D smoke bootstrap failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


@smoke_app.command("verify-step5d")
def smoke_verify_step5d(
    evaluation_run_id: str = typer.Option(..., "--evaluation-run-id"),
    require_managed_runtime: bool = typer.Option(
        False,
        "--require-managed-runtime",
    ),
    result_json: Path = typer.Option(..., "--result-json"),
) -> None:
    """Verify real-judge lineage and scope isolation for an ephemeral fixture."""

    from pmmc.smoke import verify_step5d_fixture, write_smoke_result

    settings = get_settings()
    try:
        result = asyncio.run(
            verify_step5d_fixture(
                settings,
                evaluation_run_id,
                require_managed_runtime=require_managed_runtime,
            )
        )
        write_smoke_result(result_json, result)
    except Exception as exc:
        typer.echo(
            f"ERROR: Step 5D smoke verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    if not result["valid"]:
        raise typer.Exit(code=1)


@smoke_app.command("bootstrap-step6b-schema")
def smoke_bootstrap_step6b_schema(
    confirm_ephemeral: bool = typer.Option(False, "--confirm-ephemeral"),
    result_json: Path = typer.Option(..., "--result-json"),
) -> None:
    """Import a non-paper multimodal fixture for real Questioner v2 probing."""

    from pmmc.smoke import bootstrap_step6b_schema_fixture, write_smoke_result

    settings = get_settings()
    try:
        result = asyncio.run(
            bootstrap_step6b_schema_fixture(
                settings,
                confirm_ephemeral=confirm_ephemeral,
            )
        )
        write_smoke_result(result_json, result)
    except Exception as exc:
        typer.echo(
            f"ERROR: Step 6B schema fixture bootstrap failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


@smoke_app.command("verify-step6c-planner-conformance")
def smoke_verify_step6c_planner_conformance(
    planner_config: Path = typer.Option(..., "--planner-config"),
    foundation_config: Path = typer.Option(..., "--foundation-config"),
    expected_replica_count: int = typer.Option(..., "--expected-replica-count", min=1),
    result_json: Path = typer.Option(..., "--result-json"),
) -> None:
    """Exercise real Planner parsing, policy, lineage, and operator dry-run."""

    from pmmc.smoke import (
        Step6CPlannerConformanceError,
        run_step6c_planner_conformance,
    )

    if os.environ.get("PMMC_PAPER_RESULT", "false").lower() in {"1", "true", "yes"}:
        typer.echo("ERROR: Step 6C conformance cannot be a paper result", err=True)
        raise typer.Exit(code=2)
    settings = get_settings()
    try:
        result = asyncio.run(
            run_step6c_planner_conformance(
                settings=settings,
                planner_config_path=settings.resolve_path(planner_config),
                foundation_config_path=settings.resolve_path(foundation_config),
                expected_replica_count=expected_replica_count,
                result_path=settings.resolve_path(result_json),
            )
        )
    except Exception as exc:
        detail = (
            f" stage={exc.stage} code={exc.code}"
            if isinstance(exc, Step6CPlannerConformanceError)
            else ""
        )
        typer.echo(
            f"ERROR: Step 6C Planner conformance failed ({type(exc).__name__}){detail}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@smoke_app.command("verify-step6c2-semantic-conformance")
def smoke_verify_step6c2_semantic_conformance(
    semantic_config: Path = typer.Option(..., "--semantic-config"),
    expected_replica_count: int = typer.Option(..., "--expected-replica-count", min=1),
    result_json: Path = typer.Option(..., "--result-json"),
    summary_only: bool = typer.Option(False, "--summary-only"),
) -> None:
    """Exercise strict semantic Judge output and all routed replicas."""

    from pmmc.smoke import (
        Step6C2SemanticConformanceError,
        run_step6c2_semantic_conformance,
    )

    if os.environ.get("PMMC_PAPER_RESULT", "false").lower() in {
        "1",
        "true",
        "yes",
    }:
        typer.echo(
            "ERROR: Step 6C.2 semantic conformance cannot be a paper result",
            err=True,
        )
        raise typer.Exit(code=2)
    settings = get_settings()
    try:
        result = asyncio.run(
            run_step6c2_semantic_conformance(
                settings=settings,
                semantic_config_path=settings.resolve_path(semantic_config),
                expected_replica_count=expected_replica_count,
                result_path=settings.resolve_path(result_json),
            )
        )
    except Exception as exc:
        detail = (
            f" stage={exc.stage} code={exc.code}"
            if isinstance(exc, Step6C2SemanticConformanceError)
            else ""
        )
        typer.echo(
            f"ERROR: Step 6C.2 semantic conformance failed ({type(exc).__name__}){detail}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    if summary_only:
        typer.echo(
            "SEMANTIC_CONFORMANCE_SUMMARY "
            + json.dumps(
                {
                    "content_hash": result.content_hash,
                    "model_alias": result.model_alias,
                    "replica_count": result.replica_count,
                    "request_count": result.request_count,
                    "state": "verified",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@smoke_app.command("verify-step6d-compiler-conformance")
def smoke_verify_step6d_compiler_conformance(
    questioner_config: Path = typer.Option(..., "--questioner-config"),
    planner_config: Path = typer.Option(..., "--planner-config"),
    doubter_config: Path = typer.Option(..., "--doubter-config"),
    foundation_config: Path = typer.Option(..., "--foundation-config"),
    expected_replica_count: int = typer.Option(
        ...,
        "--expected-replica-count",
        min=1,
    ),
    result_json: Path = typer.Option(..., "--result-json"),
    compact_output: bool = typer.Option(False, "--compact-output"),
) -> None:
    """Exercise native Step 6D Q/P/D schemas across every compiler replica."""

    from pmmc.smoke import (
        Step6DCompilerConformanceError,
        run_step6d_compiler_conformance,
    )

    if os.environ.get("PMMC_PAPER_RESULT", "false").lower() in {
        "1",
        "true",
        "yes",
    }:
        typer.echo(
            "ERROR: Step 6D conformance cannot be a paper result",
            err=True,
        )
        raise typer.Exit(code=2)
    settings = get_settings()
    try:
        result = asyncio.run(
            run_step6d_compiler_conformance(
                settings=settings,
                questioner_config_path=settings.resolve_path(questioner_config),
                planner_config_path=settings.resolve_path(planner_config),
                doubter_config_path=settings.resolve_path(doubter_config),
                foundation_config_path=settings.resolve_path(foundation_config),
                expected_replica_count=expected_replica_count,
                result_path=settings.resolve_path(result_json),
            )
        )
    except Exception as exc:
        detail = (
            f" stage={exc.stage} code={exc.code}"
            if isinstance(exc, Step6DCompilerConformanceError)
            else ""
        )
        typer.echo(
            "ERROR: Step 6D compiler conformance failed "
            f"({type(exc).__name__}){detail}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    if compact_output:
        branch_values = result.branch_matrix.model_dump(mode="json")
        typer.echo(
            "STEP6D_CONFORMANCE_RESULT "
            + json.dumps(
                {
                    "branch_count": sum(bool(value) for value in branch_values.values()),
                    "content_hash": result.content_hash,
                    "replica_count": result.replica_count,
                    "request_count": result.request_count,
                    "valid": result.valid,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def doctor() -> None:
    """Run lightweight project diagnostics without reading secret values."""

    settings = get_settings()
    python_ok = (3, 11) <= sys.version_info[:2] < (3, 13)
    full_context_path = settings.resolved_full_context_config_path
    full_context_ok = False
    full_context_value = f"missing:{full_context_path}"
    if full_context_path.is_file():
        try:
            from pmmc.baselines.config import (
                load_full_context_config,
                load_full_context_prompt,
                prompt_sha256,
            )

            full_context_config = load_full_context_config(full_context_path)
            full_context_prompt = load_full_context_prompt(
                full_context_config,
                settings.project_root,
            )
            full_context_ok = True
            full_context_value = f"valid:{prompt_sha256(full_context_prompt)[:12]}"
        except (OSError, ValueError):
            full_context_value = f"invalid:{full_context_path}"
    registry_path = settings.resolved_model_registry_path
    registry_ok = False
    registry_value = f"missing:{registry_path}"
    endpoint_ok = settings.mode is not AppMode.PAPER
    endpoint_value = "not evaluated"
    if registry_path.is_file():
        try:
            registry = load_model_registry(registry_path)
            if settings.mode is AppMode.PAPER:
                registry.assert_paper_ready()
            registry_ok = True
            registry_value = f"valid:{registry.resolved_hash[:12]}"
            configured = [
                entry.alias
                for entry in registry.models
                if {
                    "litellm_openai": settings.litellm_base_url,
                    "vllm_openai": settings.vllm_text_embedding_base_url,
                    "vllm_v2_embed": settings.vllm_multimodal_embedding_base_url,
                    "pmmc_contrastive": settings.contrastive_embedding_base_url,
                    "fake": "fake://local",
                }[entry.endpoint_kind.value]
            ]
            required = [
                entry.alias for entry in registry.models if entry.endpoint_kind.value != "fake"
            ]
            endpoint_ok = settings.mode is not AppMode.PAPER or set(required) <= set(configured)
            endpoint_value = f"configured={len(configured)}/{len(registry.models)}"
        except (OSError, ValueError):
            registry_value = f"invalid:{registry_path}"
    checks: dict[str, DoctorCheck] = {
        "python_version": {
            "ok": python_ok,
            "value": sys.version.split()[0],
            "expected": ">=3.11,<3.13",
        },
        "project_root": {
            "ok": settings.project_root.is_dir(),
            "value": str(settings.project_root),
        },
        "memlens": {
            "ok": settings.memlens_path.is_dir(),
            "value": str(settings.memlens_path),
        },
        "memgallery": {
            "ok": settings.memgallery_path.is_dir(),
            "value": str(settings.memgallery_path),
        },
        ".env": {
            "ok": settings.env_file_path.is_file() and os.access(settings.env_file_path, os.R_OK),
            "value": "present/readable" if settings.env_file_path.is_file() else "missing",
        },
        "model_registry": {
            "ok": registry_ok,
            "value": registry_value,
        },
        "full_context_config": {
            "ok": full_context_ok,
            "value": full_context_value,
        },
        "model_endpoints": {
            "ok": endpoint_ok,
            "value": endpoint_value,
            "expected": "all configured in paper mode",
        },
    }

    all_ok = all(item["ok"] for item in checks.values())
    typer.echo(json.dumps(checks, indent=2, sort_keys=True))
    if not all_ok:
        raise typer.Exit(code=1)


@app.command("show-settings")
def show_settings() -> None:
    """Print non-sensitive settings with secret fields masked."""

    settings = get_settings()
    typer.echo(json.dumps(settings.safe_dump(), indent=2, sort_keys=True))


@models_app.command("validate-config")
def models_validate_config() -> None:
    """Strictly validate the secret-free model registry and print its hash."""

    settings = get_settings()
    try:
        registry = load_model_registry(settings.resolved_model_registry_path)
        if settings.mode is AppMode.PAPER:
            registry.assert_paper_ready()
    except (FileNotFoundError, OSError, ValueError) as exc:
        typer.echo(f"ERROR: model registry validation failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "registry_version": registry.registry_version,
                "registry_hash": registry.resolved_hash,
                "model_count": len(registry.models),
                "aliases": sorted(model.alias for model in registry.models),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


@models_app.command("check")
def models_check(alias: str = typer.Option(..., "--alias")) -> None:
    """Check one configured provider health endpoint without running inference."""

    from pmmc.models.factory import check_model_endpoint

    settings = get_settings()
    try:
        registry = load_model_registry(settings.resolved_model_registry_path)
        result = asyncio.run(check_model_endpoint(settings, registry, alias))
    except Exception as exc:
        typer.echo(f"ERROR: model endpoint check failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


@models_app.command("probe")
def models_probe(alias: str = typer.Option(..., "--alias")) -> None:
    """Run one explicit development-only inference/embedding probe."""

    from pmmc.models.factory import create_embedding_gateway, create_generation_gateway
    from pmmc.schemas import (
        EmbeddingInput,
        EmbeddingRequest,
        GenerationRequest,
        ModelCapability,
        ModelContentKind,
        ModelContentPart,
        ModelMessage,
        SpeakerRole,
    )

    settings = get_settings()
    if settings.mode is AppMode.PAPER:
        typer.echo("ERROR: models probe is disabled in paper mode", err=True)
        raise typer.Exit(code=2)
    try:
        registry = load_model_registry(settings.resolved_model_registry_path)
        entry = registry.get(alias)

        async def probe() -> dict[str, object]:
            if ModelCapability.CHAT_TEXT in entry.capabilities:
                gateway = create_generation_gateway(settings, registry, alias)
                result = await gateway.generate(
                    GenerationRequest(
                        request_id="pmmc-operations-probe",
                        model_alias=alias,
                        messages=[
                            ModelMessage(
                                role=SpeakerRole.USER,
                                parts=[
                                    ModelContentPart(
                                        kind=ModelContentKind.TEXT,
                                        text="Reply with OK.",
                                    )
                                ],
                            )
                        ],
                        max_output_tokens=8,
                        temperature=0,
                    )
                )
                return {
                    "alias": alias,
                    "actual_model_id": result.actual_model_id,
                    "provider_request_id": result.provider_request_id,
                    "usage": result.usage.to_dict(),
                }
            gateway = create_embedding_gateway(settings, registry, alias)
            result = await gateway.embed(
                EmbeddingRequest(
                    request_id="pmmc-operations-probe",
                    model_alias=alias,
                    inputs=[EmbeddingInput(input_id="probe", text="PMMC probe")],
                )
            )
            return {
                "alias": alias,
                "actual_model_id": result.actual_model_id,
                "provider_request_id": result.provider_request_id,
                "dimension": len(result.vectors[0].vector),
            }

        result = asyncio.run(probe())
    except Exception as exc:
        typer.echo(f"ERROR: model probe failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


@models_app.command("lock")
def models_lock(
    output: Path | None = typer.Option(None, "--output"),
    profile: Path | None = typer.Option(None, "--profile"),
    alias: list[str] | None = typer.Option(None, "--alias"),
) -> None:
    """Resolve configured Hugging Face revisions to immutable commits."""

    from pmmc.models.locking import build_model_lock, write_model_lock
    from pmmc.runtime.config import load_runtime_profile

    settings = get_settings()
    try:
        registry = load_model_registry(settings.resolved_model_registry_path)
        destination = output
        if destination is None and profile is not None:
            destination = settings.resolve_path(
                load_runtime_profile(settings.resolve_path(profile)).model_lock_path
            )
        destination = destination or settings.resolved_model_lock_path
        token = settings.hf_token.get_secret_value() if settings.hf_token else None
        lock = build_model_lock(
            registry,
            aliases=alias or None,
            token=token,
        )
        write_model_lock(lock, destination)
    except Exception as exc:
        from pmmc.models.locking import HuggingFaceAuthenticationError

        if isinstance(exc, HuggingFaceAuthenticationError):
            typer.echo(f"ERROR: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"ERROR: model locking failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "lock_path": str(destination),
                "registry_hash": lock.registry_hash,
                "aliases": [item.alias for item in lock.entries],
            },
            indent=2,
            sort_keys=True,
        )
    )


@models_app.command("auth-check")
def models_auth_check() -> None:
    """Verify that the configured HF_TOKEN is accepted without exposing it."""

    from pmmc.models.locking import HuggingFaceAuthenticationError, validate_hf_token

    settings = get_settings()
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    if token is None:
        typer.echo("ERROR: HF_TOKEN is not configured", err=True)
        raise typer.Exit(code=1)
    try:
        validate_hf_token(token)
    except HuggingFaceAuthenticationError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "authenticated": True,
                "source": "HF_TOKEN",
            },
            indent=2,
            sort_keys=True,
        )
    )


@models_app.command("prepare")
def models_prepare(
    lock_path: Path | None = typer.Option(None, "--lock"),
    alias: list[str] | None = typer.Option(None, "--alias"),
    offline: bool = typer.Option(False, "--offline"),
    verification_policy: ModelVerificationPolicy = typer.Option(
        ModelVerificationPolicy.FULL,
        "--verification-policy",
    ),
) -> None:
    """Download immutable model snapshots into the persistent HF cache."""

    from pmmc.models.locking import load_model_lock, prepare_model_cache_audited

    settings = get_settings()
    try:
        lock = load_model_lock(lock_path or settings.resolved_model_lock_path)
        token = settings.hf_token.get_secret_value() if settings.hf_token else None
        preparation = prepare_model_cache_audited(
            lock,
            settings.model_cache_path,
            token=token,
            aliases=alias or None,
            local_files_only=offline or settings.mode is AppMode.PAPER,
            verification_policy=verification_policy,
            allow_full_fallback=not offline,
        )
    except Exception as exc:
        from pmmc.models.locking import HuggingFaceAuthenticationError

        if isinstance(exc, HuggingFaceAuthenticationError):
            typer.echo(f"ERROR: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"ERROR: model cache preparation failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "cache_root": str(settings.model_cache_path),
                "prepared_aliases": sorted(preparation.paths),
                "verification_policy": verification_policy.value,
                "verification_receipts": {
                    alias: result.receipt_id
                    for alias, result in sorted(preparation.verifications.items())
                    if result.receipt_id is not None
                },
                "bytes_hashed": sum(
                    result.bytes_hashed for result in preparation.verifications.values()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@models_app.command("verify-cache")
def models_verify_cache(
    lock_path: Path | None = typer.Option(None, "--lock"),
    alias: list[str] | None = typer.Option(None, "--alias"),
    verification_policy: ModelVerificationPolicy = typer.Option(
        ModelVerificationPolicy.CACHED_RECEIPT,
        "--verification-policy",
    ),
    allow_full_fallback: bool = typer.Option(
        False,
        "--allow-full-fallback/--require-receipt",
    ),
) -> None:
    """Verify locked files from local cache without network access."""

    from pmmc.models.locking import load_model_lock, verify_model_cache

    settings = get_settings()
    try:
        lock = load_model_lock(lock_path or settings.resolved_model_lock_path)
        report = verify_model_cache(
            lock,
            settings.model_cache_path,
            aliases=alias or None,
            verification_policy=verification_policy,
            allow_full_fallback=allow_full_fallback,
        )
    except Exception as exc:
        typer.echo(f"ERROR: model cache verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


@runtime_app.command("inventory")
def runtime_inventory() -> None:
    """Print the anonymized GPUs and topology visible inside this Pod."""

    from pmmc.runtime.hardware import NvmlHardwareProvider

    try:
        inventory = NvmlHardwareProvider().inventory()
    except Exception as exc:
        typer.echo(f"ERROR: GPU inventory failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "hardware_fingerprint": inventory.fingerprint,
                "driver_version": inventory.driver_version,
                "cuda_driver_version": inventory.cuda_driver_version,
                "devices": [item.to_dict() for item in inventory.devices],
                "topology": [item.to_dict() for item in inventory.topology],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


@runtime_app.command("plan")
def runtime_plan(
    profile: Path | None = typer.Option(None, "--profile"),
    stage: str | None = typer.Option(None, "--stage"),
) -> None:
    """Verify locked cache and produce a deterministic GPU allocation plan."""

    from pmmc.runtime.manager import ManagedRuntimeError, plan_runtime

    settings = get_settings()
    try:
        prepared = plan_runtime(
            settings,
            profile_path=profile,
            logical_stage=stage,
        )
    except Exception as exc:
        detail = f": {exc}" if isinstance(exc, ManagedRuntimeError) else ""
        typer.echo(
            f"ERROR: runtime planning failed ({type(exc).__name__}){detail}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(prepared.plan.model_dump_json(indent=2))
    typer.echo(f"Runtime artifacts: {prepared.run_dir}")


@runtime_app.command("calibrate")
def runtime_calibrate(
    profile: Path | None = typer.Option(None, "--profile"),
) -> None:
    """Start each service and record hardware-bound observed GPU memory."""

    from pmmc.runtime.manager import calibrate_runtime

    settings = get_settings()
    try:
        calibrations, run_dir = calibrate_runtime(settings, profile_path=profile)
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(f"ERROR: runtime calibration failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "calibrations": [item.calibration_id for item in calibrations],
                "runtime_artifacts": str(run_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


@runtime_app.command("test")
def runtime_test(
    suite: str = typer.Option(..., "--suite"),
    profile: Path | None = typer.Option(None, "--profile"),
    snapshot: str | None = typer.Option(None, "--snapshot"),
    dataset: DatasetName | None = typer.Option(None, "--dataset"),
    limit: int = typer.Option(1, "--limit", min=1),
) -> None:
    """Start required local services, run one smoke suite, and release GPUs."""

    from pmmc.runtime.manager import run_managed_commands, test_managed_gateway

    settings = get_settings()
    try:
        if suite == "model-gateway":
            _code, run_dir = test_managed_gateway(settings, profile_path=profile)
        elif suite == "full-context":
            if snapshot is None or dataset is None or dataset is DatasetName.UNKNOWN:
                raise typer.BadParameter(
                    "full-context suite requires --snapshot and a concrete --dataset"
                )
            commands = [
                [
                    sys.executable,
                    "-m",
                    "pmmc",
                    "run-baseline",
                    "--method",
                    method.value,
                    "--dataset",
                    dataset.value,
                    "--snapshot",
                    snapshot,
                    "--limit",
                    str(limit),
                ]
                for method in (
                    BaselineMethod.FULL_CONTEXT_CAPTIONS,
                    BaselineMethod.FULL_CONTEXT_RAW_IMAGES,
                )
            ]
            _code, run_dir = run_managed_commands(
                settings,
                profile_path=profile,
                logical_stage="answer",
                commands=commands,
            )
        else:
            raise typer.BadParameter(
                "must be model-gateway or full-context",
                param_hint="--suite",
            )
    except typer.BadParameter:
        raise
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(f"ERROR: managed runtime test failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Runtime artifacts: {run_dir}")


@runtime_app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def runtime_run(
    context: typer.Context,
    profile: Path | None = typer.Option(None, "--profile"),
    stage: str = typer.Option(..., "--stage"),
) -> None:
    """Run an arbitrary command while one managed model stage is active."""

    from pmmc.runtime.manager import run_managed_command

    command = list(context.args)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise typer.BadParameter("provide a command after --")
    try:
        _code, run_dir = run_managed_command(
            get_settings(),
            profile_path=profile,
            logical_stage=stage,
            command=command,
        )
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(f"ERROR: managed runtime command failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Runtime artifacts: {run_dir}")


def _append_repeated_option(command: list[str], name: str, values: Iterable[object] | None) -> None:
    for value in values or []:
        rendered = value.value if hasattr(value, "value") else str(value)
        command.extend([name, str(rendered)])


@runtime_app.command("baseline")
def runtime_baseline(
    method: str = typer.Option(..., "--method"),
    dataset: DatasetName = typer.Option(..., "--dataset"),
    snapshot: str = typer.Option(..., "--snapshot"),
    corpus_id: str = typer.Option(..., "--corpus-id"),
    profile: Path | None = typer.Option(None, "--profile"),
    answer_model: str | None = typer.Option(None, "--answer-model"),
    question_type: list[QuestionType] | None = typer.Option(None, "--question-type"),
    source_category: list[str] | None = typer.Option(None, "--source-category"),
    scenario: list[str] | None = typer.Option(None, "--scenario"),
    query_id: list[str] | None = typer.Option(None, "--query-id"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    sample_size: int | None = typer.Option(None, "--sample-size", min=1),
    seed: int = typer.Option(0, "--seed", min=0),
    shard_count: int = typer.Option(1, "--shard-count", min=1),
    shard_index: int = typer.Option(0, "--shard-index", min=0),
    include_excluded: bool = typer.Option(False, "--include-excluded"),
) -> None:
    """Run one RAG baseline through isolated embedding and answer GPU stages."""

    from pmmc.runtime.manager import run_managed_command, run_managed_commands

    settings = get_settings()
    resolved_method = _parse_rag_method(method)
    if dataset is DatasetName.UNKNOWN:
        raise typer.BadParameter("unknown is not a runnable benchmark", param_hint="--dataset")
    if limit is not None and sample_size is not None:
        raise typer.BadParameter("--limit and --sample-size are mutually exclusive")
    stage_by_method = {
        BaselineMethod.CAPTION_TEXT_RAG: "embedding_text",
        BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG: "embedding_contrastive",
        BaselineMethod.NATIVE_MULTIMODAL_RAG: "embedding_multimodal",
    }
    stable_run_id = make_run_id(resolved_method.value)
    build_command = [
        sys.executable,
        "-m",
        "pmmc",
        "baseline",
        "build-index",
        "--method",
        resolved_method.value,
        "--corpus-id",
        corpus_id,
    ]
    retrieve_command = [
        sys.executable,
        "-m",
        "pmmc",
        "run-baseline",
        "--method",
        resolved_method.value,
        "--dataset",
        dataset.value,
        "--snapshot",
        snapshot,
        "--corpus-id",
        corpus_id,
        "--run-id",
        stable_run_id,
        "--phase",
        "retrieve",
        "--seed",
        str(seed),
        "--shard-count",
        str(shard_count),
        "--shard-index",
        str(shard_index),
    ]
    if answer_model is not None:
        retrieve_command.extend(["--answer-model", answer_model])
    if limit is not None:
        retrieve_command.extend(["--limit", str(limit)])
    if sample_size is not None:
        retrieve_command.extend(["--sample-size", str(sample_size)])
    if include_excluded:
        retrieve_command.append("--include-excluded")
    _append_repeated_option(retrieve_command, "--question-type", question_type)
    _append_repeated_option(retrieve_command, "--source-category", source_category)
    _append_repeated_option(retrieve_command, "--scenario", scenario)
    _append_repeated_option(retrieve_command, "--query-id", query_id)
    answer_command = [
        sys.executable,
        "-m",
        "pmmc",
        "run-baseline",
        "--resume-run-id",
        stable_run_id,
        "--phase",
        "answer",
    ]
    typer.echo(f"Managed RAG run ID: {stable_run_id}")
    try:
        _code, embedding_runtime_dir = run_managed_commands(
            settings,
            profile_path=profile,
            logical_stage=stage_by_method[resolved_method],
            commands=[build_command, retrieve_command],
        )
        _code, answer_runtime_dir = run_managed_command(
            settings,
            profile_path=profile,
            logical_stage="answer",
            command=answer_command,
        )
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(f"ERROR: managed RAG baseline failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "answer_runtime_artifacts": str(answer_runtime_dir),
                "embedding_runtime_artifacts": str(embedding_runtime_dir),
                "run_id": stable_run_id,
                "run_artifacts": str(settings.runs_path / stable_run_id),
            },
            indent=2,
            sort_keys=True,
        )
    )


@runtime_app.command("baseline-suite")
def runtime_baseline_suite(
    suite_id: str = typer.Option(..., "--suite-id"),
    profile: Path | None = typer.Option(None, "--profile"),
) -> None:
    """Run three embedding stages and one shared answer stage for a frozen suite."""

    from pmmc.runtime.manager import run_managed_command

    settings = get_settings()
    stage_by_method = {
        BaselineMethod.CAPTION_TEXT_RAG: "embedding_text",
        BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG: "embedding_contrastive",
        BaselineMethod.NATIVE_MULTIMODAL_RAG: "embedding_multimodal",
    }
    runtime_dirs: dict[str, str] = {}
    try:
        for method, stage in stage_by_method.items():
            command = [
                sys.executable,
                "-m",
                "pmmc",
                "baseline",
                "run-suite",
                "--suite-id",
                suite_id,
                "--phase",
                "retrieve",
                "--method",
                method.value,
            ]
            _code, directory = run_managed_command(
                settings,
                profile_path=profile,
                logical_stage=stage,
                command=command,
            )
            runtime_dirs[stage] = str(directory)
        answer_command = [
            sys.executable,
            "-m",
            "pmmc",
            "baseline",
            "run-suite",
            "--suite-id",
            suite_id,
            "--phase",
            "answer",
        ]
        _code, directory = run_managed_command(
            settings,
            profile_path=profile,
            logical_stage="answer",
            command=answer_command,
        )
        runtime_dirs["answer"] = str(directory)
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(f"ERROR: managed baseline suite failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "suite_id": suite_id,
                "runtime_artifacts": runtime_dirs,
                "suite_artifacts": str(settings.runs_path / "baseline-suites" / suite_id),
            },
            indent=2,
            sort_keys=True,
        )
    )


@runtime_app.command("compile-questions")
def runtime_compile_questions(
    snapshot: str | None = typer.Option(None, "--snapshot"),
    profile: Path | None = typer.Option(None, "--profile"),
    conversation_id: list[str] | None = typer.Option(None, "--conversation-id"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    sample_size: int | None = typer.Option(None, "--sample-size", min=1),
    seed: int = typer.Option(0, "--seed", min=0),
    shard_count: int = typer.Option(1, "--shard-count", min=1),
    shard_index: int = typer.Option(0, "--shard-index", min=0),
    compiler_model: str | None = typer.Option(None, "--compiler-model"),
    config: Path | None = typer.Option(None, "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    resume_run_id: str | None = typer.Option(None, "--resume-run-id"),
    phase: str = typer.Option(
        "all",
        "--phase",
        help="all or questioner",
    ),
    result_json: Path | None = typer.Option(None, "--result-json"),
) -> None:
    """Run Questioner, text QBank index, then multimodal QBank index on managed GPUs."""

    from pmmc.runtime.manager import run_managed_command

    if limit is not None and sample_size is not None:
        raise typer.BadParameter("--limit and --sample-size are mutually exclusive")
    if resume_run_id is None and snapshot is None:
        raise typer.BadParameter("new managed Questioner runs require --snapshot")
    if run_id is not None and resume_run_id is not None:
        raise typer.BadParameter("--run-id and --resume-run-id are mutually exclusive")
    if phase not in {"all", "questioner"}:
        raise typer.BadParameter("must be all or questioner", param_hint="--phase")
    if resume_run_id is not None and any(
        value is not None for value in (snapshot, conversation_id, limit, sample_size)
    ):
        raise typer.BadParameter(
            "--resume-run-id cannot be combined with snapshot or conversation selection"
        )
    settings = get_settings()
    stable_run_id = resume_run_id or run_id or make_run_id("pmmc-questioner")
    producer = [
        sys.executable,
        "-m",
        "pmmc",
        "compile-memory",
        "--target-stage",
        "questions",
        "--phase",
        "questioner",
    ]
    if resume_run_id is not None:
        producer.extend(["--resume-run-id", stable_run_id])
    else:
        assert snapshot is not None
        producer.extend(
            [
                "--snapshot",
                snapshot,
                "--run-id",
                stable_run_id,
                "--seed",
                str(seed),
                "--shard-count",
                str(shard_count),
                "--shard-index",
                str(shard_index),
            ]
        )
        if limit is not None:
            producer.extend(["--limit", str(limit)])
        if sample_size is not None:
            producer.extend(["--sample-size", str(sample_size)])
        _append_repeated_option(producer, "--conversation-id", conversation_id)
    shared_options: list[str] = []
    if config is not None:
        shared_options.extend(["--config", str(config)])
    if compiler_model is not None:
        shared_options.extend(["--compiler-model", compiler_model])
    producer.extend(shared_options)
    text_index = [
        sys.executable,
        "-m",
        "pmmc",
        "compile-memory",
        "--resume-run-id",
        stable_run_id,
        "--phase",
        "index_text",
        *shared_options,
    ]
    multimodal_index = [
        sys.executable,
        "-m",
        "pmmc",
        "compile-memory",
        "--resume-run-id",
        stable_run_id,
        "--phase",
        "index_multimodal",
        *shared_options,
    ]
    runtime_dirs: dict[str, str] = {}
    try:
        stages = [("compiler", producer)]
        if phase == "all":
            stages.extend(
                [
                    ("embedding_text", text_index),
                    ("embedding_multimodal", multimodal_index),
                ]
            )
        for stage, command in stages:
            _code, directory = run_managed_command(
                settings,
                profile_path=profile,
                logical_stage=stage,
                command=command,
            )
            runtime_dirs[stage] = str(directory)
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(f"ERROR: managed Questioner run failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    payload = {
        "run_id": stable_run_id,
        "phase": phase,
        "runtime_artifacts": runtime_dirs,
    }
    if result_json is not None:
        destination = settings.resolve_path(result_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@runtime_app.command("compile-programs")
def runtime_compile_programs(
    snapshot: str | None = typer.Option(None, "--snapshot"),
    profile: Path | None = typer.Option(None, "--profile"),
    conversation_id: list[str] | None = typer.Option(None, "--conversation-id"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    sample_size: int | None = typer.Option(None, "--sample-size", min=1),
    seed: int = typer.Option(0, "--seed", min=0),
    split: str | None = typer.Option(None, "--split"),
    questioner_config: Path = typer.Option(
        Path("configs/compiler/questioner-v3.1.yaml"), "--questioner-config"
    ),
    planner_config: Path = typer.Option(
        Path("configs/compiler/planner-v1.yaml"), "--planner-config"
    ),
    foundation_config: Path = typer.Option(
        Path("configs/compiler/pmmc-v2.yaml"), "--foundation-config"
    ),
    questioner_freeze_manifest: Path | None = typer.Option(None, "--questioner-freeze-manifest"),
    program_pilot_audit_hash: str | None = typer.Option(None, "--program-pilot-audit-hash"),
    run_id: str | None = typer.Option(None, "--run-id"),
    resume_run_id: str | None = typer.Option(None, "--resume-run-id"),
    compiler_replicas: int = typer.Option(1, "--compiler-replicas", min=1),
    embedding_replicas: int = typer.Option(1, "--embedding-replicas", min=1),
    result_json: Path | None = typer.Option(None, "--result-json"),
) -> None:
    """Run Questioner v3.1, Planner v1, indexes, vectors, and real dry-run."""

    from pmmc.runtime.manager import run_managed_commands

    if limit is not None and sample_size is not None:
        raise typer.BadParameter("--limit and --sample-size are mutually exclusive")
    if resume_run_id is None and snapshot is None:
        raise typer.BadParameter("new managed program runs require --snapshot")
    if run_id is not None and resume_run_id is not None:
        raise typer.BadParameter("--run-id and --resume-run-id are mutually exclusive")
    if (questioner_freeze_manifest is None) == (program_pilot_audit_hash is None):
        raise typer.BadParameter(
            "provide exactly one of --questioner-freeze-manifest or --program-pilot-audit-hash"
        )
    if resume_run_id is not None and any(
        value is not None for value in (snapshot, conversation_id, limit, sample_size, split)
    ):
        raise typer.BadParameter(
            "--resume-run-id cannot change snapshot, split, or conversation selection"
        )
    settings = get_settings()
    stable_run_id = resume_run_id or run_id or make_run_id("pmmc-compiler-programs")
    shared = [
        "--target-stage",
        "programs",
        "--config",
        str(questioner_config),
        "--planner-config",
        str(planner_config),
        "--foundation-config",
        str(foundation_config),
    ]
    if questioner_freeze_manifest is not None:
        shared.extend(["--questioner-freeze-manifest", str(questioner_freeze_manifest)])
    else:
        assert program_pilot_audit_hash is not None
        shared.extend(["--program-pilot-audit-hash", program_pilot_audit_hash])

    producer = [
        sys.executable,
        "-m",
        "pmmc",
        "compile-memory",
        *shared,
        "--phase",
        "questioner",
    ]
    if resume_run_id is not None:
        producer.extend(["--resume-run-id", stable_run_id])
    else:
        assert snapshot is not None
        producer.extend(
            [
                "--snapshot",
                snapshot,
                "--run-id",
                stable_run_id,
                "--seed",
                str(seed),
            ]
        )
        if split is not None:
            producer.extend(["--split", split])
        if limit is not None:
            producer.extend(["--limit", str(limit)])
        if sample_size is not None:
            producer.extend(["--sample-size", str(sample_size)])
        _append_repeated_option(producer, "--conversation-id", conversation_id)

    def resumed(phase: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "pmmc",
            "compile-memory",
            *shared,
            "--resume-run-id",
            stable_run_id,
            "--phase",
            phase,
        ]

    runtime_dirs: dict[str, str] = {}
    try:
        _code, directory = run_managed_commands(
            settings,
            profile_path=profile,
            logical_stage="compiler",
            commands=[producer, resumed("planner")],
            replica_count=compiler_replicas,
        )
        runtime_dirs["compiler"] = str(directory)
        _code, directory = run_managed_commands(
            settings,
            profile_path=profile,
            logical_stage="embedding_text",
            commands=[
                resumed("qbank_index_text"),
                resumed("memory_index_text"),
                resumed("program_query_text"),
            ],
            replica_count=embedding_replicas,
        )
        runtime_dirs["embedding_text"] = str(directory)
        _code, directory = run_managed_commands(
            settings,
            profile_path=profile,
            logical_stage="embedding_multimodal",
            commands=[
                resumed("qbank_index_multimodal"),
                resumed("memory_index_multimodal"),
                resumed("program_query_multimodal"),
            ],
            replica_count=embedding_replicas,
        )
        runtime_dirs["embedding_multimodal"] = str(directory)
        for phase in ("dry_run", "finalize"):
            completed = subprocess.run(resumed(phase), check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"program {phase} command exited with code {completed.returncode}"
                )
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(f"ERROR: managed program compiler failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    payload = {
        "run_id": stable_run_id,
        "status": "completed",
        "runtime_artifacts": runtime_dirs,
        "compiler_replicas": compiler_replicas,
        "embedding_replicas": embedding_replicas,
    }
    if result_json is not None:
        destination = settings.resolve_path(result_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@runtime_app.command("compile-verified-qbank")
def runtime_compile_verified_qbank(
    snapshot: str | None = typer.Option(None, "--snapshot"),
    profile: Path | None = typer.Option(None, "--profile"),
    conversation_id: list[str] | None = typer.Option(None, "--conversation-id"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    sample_size: int | None = typer.Option(None, "--sample-size", min=1),
    seed: int = typer.Option(0, "--seed", min=0),
    split: str | None = typer.Option(None, "--split"),
    questioner_config: Path = typer.Option(
        Path("configs/compiler/questioner-v3.4.yaml"), "--questioner-config"
    ),
    planner_config: Path = typer.Option(
        Path("configs/compiler/planner-v1.3.yaml"), "--planner-config"
    ),
    doubter_config: Path = typer.Option(
        Path("configs/compiler/doubter-v1.1.yaml"), "--doubter-config"
    ),
    foundation_config: Path = typer.Option(
        Path("configs/compiler/pmmc-v5.yaml"), "--foundation-config"
    ),
    questioner_freeze_manifest: Path | None = typer.Option(
        None, "--questioner-freeze-manifest"
    ),
    program_pilot_audit_hash: str | None = typer.Option(
        None, "--program-pilot-audit-hash"
    ),
    run_id: str | None = typer.Option(None, "--run-id"),
    resume_run_id: str | None = typer.Option(None, "--resume-run-id"),
    compiler_replicas: int = typer.Option(8, "--compiler-replicas", min=1),
    embedding_replicas: int = typer.Option(8, "--embedding-replicas", min=1),
    result_json: Path | None = typer.Option(None, "--result-json"),
) -> None:
    """Compile Q/P/D rounds through the active-r2 Verified QBank projection."""

    from pmmc.runtime.manager import run_managed_commands

    if limit is not None and sample_size is not None:
        raise typer.BadParameter("--limit and --sample-size are mutually exclusive")
    if resume_run_id is None and snapshot is None:
        raise typer.BadParameter("new verified-QBank runs require --snapshot")
    if run_id is not None and resume_run_id is not None:
        raise typer.BadParameter("--run-id and --resume-run-id are mutually exclusive")
    if (questioner_freeze_manifest is None) == (program_pilot_audit_hash is None):
        raise typer.BadParameter(
            "provide exactly one of --questioner-freeze-manifest or "
            "--program-pilot-audit-hash"
        )
    if resume_run_id is not None and any(
        value is not None
        for value in (snapshot, conversation_id, limit, sample_size, split)
    ):
        raise typer.BadParameter(
            "--resume-run-id cannot change snapshot, split, or conversation selection"
        )
    settings = get_settings()
    stable_run_id = resume_run_id or run_id or make_run_id("pmmc-verified-qbank")
    shared = [
        "--target-stage",
        "verified_qbank",
        "--config",
        str(questioner_config),
        "--planner-config",
        str(planner_config),
        "--doubter-config",
        str(doubter_config),
        "--foundation-config",
        str(foundation_config),
    ]
    if questioner_freeze_manifest is not None:
        shared.extend(
            ["--questioner-freeze-manifest", str(questioner_freeze_manifest)]
        )
    else:
        assert program_pilot_audit_hash is not None
        shared.extend(["--program-pilot-audit-hash", program_pilot_audit_hash])

    producer = [
        sys.executable,
        "-m",
        "pmmc",
        "compile-memory",
        *shared,
        "--phase",
        "questioner",
    ]
    if resume_run_id is not None:
        producer.extend(["--resume-run-id", stable_run_id])
    else:
        assert snapshot is not None
        producer.extend(
            [
                "--snapshot",
                snapshot,
                "--run-id",
                stable_run_id,
                "--seed",
                str(seed),
            ]
        )
        if split is not None:
            producer.extend(["--split", split])
        if limit is not None:
            producer.extend(["--limit", str(limit)])
        if sample_size is not None:
            producer.extend(["--sample-size", str(sample_size)])
        _append_repeated_option(producer, "--conversation-id", conversation_id)

    def resumed(phase: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "pmmc",
            "compile-memory",
            *shared,
            "--resume-run-id",
            stable_run_id,
            "--phase",
            phase,
        ]

    runtime_dirs: dict[str, str] = {}
    try:
        _code, directory = run_managed_commands(
            settings,
            profile_path=profile,
            logical_stage="compiler",
            commands=[producer, resumed("planner")],
            replica_count=compiler_replicas,
        )
        runtime_dirs["compiler_r0"] = str(directory)
        _code, directory = run_managed_commands(
            settings,
            profile_path=profile,
            logical_stage="embedding_text",
            commands=[
                resumed("qbank_index_text"),
                resumed("memory_index_text"),
                resumed("program_query_text"),
            ],
            replica_count=embedding_replicas,
        )
        runtime_dirs["embedding_text"] = str(directory)
        _code, directory = run_managed_commands(
            settings,
            profile_path=profile,
            logical_stage="embedding_multimodal",
            commands=[
                resumed("qbank_index_multimodal"),
                resumed("memory_index_multimodal"),
                resumed("program_query_multimodal"),
            ],
            replica_count=embedding_replicas,
        )
        runtime_dirs["embedding_multimodal"] = str(directory)
        for phase in ("dry_run", "r0_input"):
            completed = subprocess.run(resumed(phase), check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"verified-QBank {phase} exited with code {completed.returncode}"
                )
        _code, directory = run_managed_commands(
            settings,
            profile_path=profile,
            logical_stage="compiler",
            commands=[resumed("doubter")],
            replica_count=compiler_replicas,
        )
        runtime_dirs["compiler_doubter"] = str(directory)
        for phase in ("decision", "finalize"):
            completed = subprocess.run(resumed(phase), check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"verified-QBank {phase} exited with code {completed.returncode}"
                )
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(
            f"ERROR: managed verified-QBank compiler failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    payload = {
        "run_id": stable_run_id,
        "status": "completed",
        "active_round": 2,
        "maximum_round": 3,
        "runtime_artifacts": runtime_dirs,
        "compiler_replicas": compiler_replicas,
        "embedding_replicas": embedding_replicas,
    }
    if result_json is not None:
        destination = settings.resolve_path(result_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@runtime_app.command("evaluate-qbank-coverage")
def runtime_evaluate_qbank_coverage(
    compiler_run_id: str = typer.Option(..., "--compiler-run-id"),
    dataset: DatasetName = typer.Option(..., "--dataset"),
    profile: Path | None = typer.Option(None, "--profile"),
    question_type: list[QuestionType] | None = typer.Option(None, "--question-type"),
    source_category: list[str] | None = typer.Option(None, "--source-category"),
    scenario: list[str] | None = typer.Option(None, "--scenario"),
    query_id: list[str] | None = typer.Option(None, "--query-id"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    sample_size: int | None = typer.Option(None, "--sample-size", min=1),
    seed: int = typer.Option(0, "--seed", min=0),
    shard_count: int = typer.Option(1, "--shard-count", min=1),
    shard_index: int = typer.Option(0, "--shard-index", min=0),
    config: Path | None = typer.Option(None, "--config"),
    result_json: Path | None = typer.Option(None, "--result-json"),
) -> None:
    """Stage QBank query embeddings across GPUs, then score coverage on CPU."""

    from pmmc.compiler.questioner_config import load_questioner_config
    from pmmc.evaluation.qbank_workflow import (
        QBankCoveragePhase,
        run_qbank_coverage,
    )
    from pmmc.models.registry import load_model_registry
    from pmmc.runtime.manager import run_managed_command

    if dataset is DatasetName.UNKNOWN:
        raise typer.BadParameter("unknown is not a runnable benchmark")
    if limit is not None and sample_size is not None:
        raise typer.BadParameter("--limit and --sample-size are mutually exclusive")
    settings = get_settings()
    common = [
        sys.executable,
        "-m",
        "pmmc",
        "evaluate",
        "qbank-coverage",
        "--compiler-run-id",
        compiler_run_id,
        "--dataset",
        dataset.value,
        "--seed",
        str(seed),
        "--shard-count",
        str(shard_count),
        "--shard-index",
        str(shard_index),
    ]
    _append_repeated_option(common, "--question-type", question_type)
    _append_repeated_option(common, "--source-category", source_category)
    _append_repeated_option(common, "--scenario", scenario)
    _append_repeated_option(common, "--query-id", query_id)
    if limit is not None:
        common.extend(["--limit", str(limit)])
    if sample_size is not None:
        common.extend(["--sample-size", str(sample_size)])
    if config is not None:
        common.extend(["--config", str(config)])
    runtime_dirs: dict[str, str] = {}
    try:
        for logical_stage, phase in (
            ("embedding_text", QBankCoveragePhase.EMBED_TEXT),
            ("embedding_multimodal", QBankCoveragePhase.EMBED_MULTIMODAL),
        ):
            _code, directory = run_managed_command(
                settings,
                profile_path=profile,
                logical_stage=logical_stage,
                command=[*common, "--phase", phase.value],
            )
            runtime_dirs[logical_stage] = str(directory)
        config_path = (
            settings.resolve_path(config)
            if config is not None
            else settings.resolved_questioner_config_path
        )
        result = asyncio.run(
            run_qbank_coverage(
                settings=settings,
                registry=load_model_registry(settings.resolved_model_registry_path),
                config=load_questioner_config(config_path),
                compiler_run_id=compiler_run_id,
                selection=BenchmarkSelection(
                    datasets=[dataset],
                    question_types=question_type or [],
                    source_categories=source_category or [],
                    scenario_ids=scenario or [],
                    query_ids=query_id or [],
                    limit=limit,
                    sample_size=sample_size,
                    seed=seed,
                    shard_count=shard_count,
                    shard_index=shard_index,
                ),
                phase=QBankCoveragePhase.SCORE,
            )
        )
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(f"ERROR: managed QBank coverage failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    payload = {
        "coverage_run_id": result.coverage_run_id,
        "compiler_run_id": result.compiler_run_id,
        "runtime_artifacts": runtime_dirs,
        "gate_report": (
            result.report.model_dump(mode="json") if result.report is not None else None
        ),
    }
    if result_json is not None:
        destination = settings.resolve_path(result_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@runtime_app.command("execute-query-run")
def runtime_execute_query_run(
    source_compiler_run_id: str = typer.Option(..., "--source-compiler-run-id"),
    coverage_run_id: str = typer.Option(..., "--coverage-run-id"),
    fallback_run_id: str = typer.Option(..., "--fallback-run-id"),
    query_id: list[str] = typer.Option(..., "--query-id"),
    config: Path = typer.Option(
        Path("configs/runtime/query-executor-v1.yaml"), "--config"
    ),
    foundation_config: Path = typer.Option(
        Path("configs/compiler/pmmc-v5.yaml"), "--foundation-config"
    ),
    answer_model: str = typer.Option("answer-qwen3.5-9b", "--answer-model"),
    run_id: str | None = typer.Option(None, "--run-id"),
    pilot_projection: bool = typer.Option(False, "--pilot-projection"),
    result_json: Path | None = typer.Option(None, "--result-json"),
) -> None:
    """Run or resume one frozen-QBank Online Query Executor denominator."""

    from pmmc.runtime.query_executor_config import load_query_executor_config
    from pmmc.runtime.query_workflow import (
        run_query_executor_workflow,
        safe_query_executor_error_diagnostics,
        verify_query_executor_run,
    )

    if not query_id or len(set(query_id)) != len(query_id):
        raise typer.BadParameter("--query-id values must be non-empty and unique")
    settings = get_settings()
    try:
        executor_config = load_query_executor_config(settings.resolve_path(config))
        registry = load_model_registry(settings.resolved_model_registry_path)

        last_completed = -1
        last_progress_at = 0.0

        def progress(counts: dict[str, int]) -> None:
            nonlocal last_completed, last_progress_at
            completed = counts.get("succeeded", 0) + counts.get("failed", 0)
            if completed == last_completed:
                return
            now = time.monotonic()
            if last_completed >= 0 and completed < len(query_id) and now - last_progress_at < 30:
                return
            last_completed = completed
            last_progress_at = now
            typer.echo(
                "STEP7_QUERY_PROGRESS "
                + json.dumps(
                    {
                        "active_fallback": counts.get("active_fallback", 0),
                        "bound_programs": counts.get("bound_programs", 0),
                        "completed": completed,
                        "completed_semantics": "terminal_jobs_not_answer_success",
                        "execution_counter_scope": "durably_completed_queries_only",
                        "execution_counter_unit": "query_round_facts_not_unique_invocations",
                        "executions": counts.get("executions", 0),
                        "failed": counts.get("failed", 0),
                        "pmmc_answers": counts.get("pmmc_answers", 0),
                        "pmmc_answer_counter_unit": "query_round_variants_not_model_calls",
                        "retried_jobs": counts.get("resumed", 0),
                        "retry_events": counts.get("retrying", 0),
                        "routes": counts.get("routes", 0),
                        "sufficient": counts.get("sufficient", 0),
                        "answer_terminal_failures": counts.get(
                            "terminal_failures", 0
                        ),
                        "total": len(query_id),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )

        result = asyncio.run(
            run_query_executor_workflow(
                settings=settings,
                registry=registry,
                config=executor_config,
                foundation_config_path=foundation_config,
                source_compiler_run_id=source_compiler_run_id,
                coverage_run_id=coverage_run_id,
                fallback_run_id=fallback_run_id,
                query_ids=query_id,
                pilot_projection=pilot_projection,
                run_id=run_id,
                answer_model_alias=answer_model,
                on_progress=progress,
            )
        )
        verification = asyncio.run(
            verify_query_executor_run(settings, result.manifest.query_executor_run_id)
        )
        if not bool(verification["valid"]):
            raise RuntimeError("query executor terminal verification failed")
        payload = {
            "query_executor_run_id": result.manifest.query_executor_run_id,
            "source_compiler_run_id": source_compiler_run_id,
            "coverage_run_id": coverage_run_id,
            "fallback_run_id": fallback_run_id,
            "selected_count": result.batch.selected_count,
            "succeeded_count": result.batch.succeeded_count,
            "failed_count": result.batch.failed_count,
            "technical_failed_count": result.batch.technical_failed_count,
            "no_answer_count": result.batch.failed_count - result.batch.technical_failed_count,
            "resumed_count": result.batch.resumed_count,
            "answer_prompt_hash": result.answer_prompt_hash,
            "verification": verification,
        }
        if result_json is not None:
            destination = settings.resolve_path(result_json)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        safe_diagnostics = safe_query_executor_error_diagnostics(exc)
        if safe_diagnostics:
            typer.echo(
                "STEP7_QUERY_EXECUTOR_FAILURE "
                + json.dumps(
                    safe_diagnostics,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                err=True,
            )
        _persist_managed_child_diagnostic(
            event="step7_query_executor_failed",
            exc=exc,
            diagnostics={
                "query_count": len(query_id),
                "source_compiler_run_hash": hashlib.sha256(
                    source_compiler_run_id.encode("utf-8")
                ).hexdigest()[:16],
                **safe_diagnostics,
            },
        )
        typer.echo(
            f"ERROR: Step 7 query executor failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    if result_json is None:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo("STEP7_QUERY_RESULT " + json.dumps({
            "state": "verified", "result_saved": True,
            "selected_count": result.batch.selected_count,
            "answered_count": result.batch.succeeded_count,
            "no_answer_count": result.batch.failed_count - result.batch.technical_failed_count,
            "technical_failed_count": result.batch.technical_failed_count,
            "answer_success_is_terminal_success": False,
        }, sort_keys=True, separators=(",", ":")))


@runtime_app.command("verify-query-run")
def runtime_verify_query_run(
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Verify one completed Online Query Executor run without model calls."""

    from pmmc.runtime.query_workflow import verify_query_executor_run

    try:
        report = asyncio.run(verify_query_executor_run(get_settings(), run_id))
    except Exception as exc:
        typer.echo(
            f"ERROR: Step 7 query verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not bool(report["valid"]):
        raise typer.Exit(code=1)


@runtime_app.command("export-step7-pilot")
def runtime_export_step7_pilot(
    source_manifest: Path = typer.Option(..., "--source-manifest"),
) -> None:
    """Build the deterministic Step 7 public/private transfer bundle."""

    from pmmc.runtime.step7_pilot import build_step7_pilot_bundle
    from pmmc.schemas import Step7PilotSourceManifest

    settings = get_settings()
    try:
        source = Step7PilotSourceManifest.model_validate_json(
            settings.resolve_path(source_manifest).read_text(encoding="utf-8")
        )
        if source.protocol_version != "pmmc-step7-executor-pilot-v1":
            raise ValueError("use export-step7b-pilot for a Step 7B source manifest")
        roots = {item.dataset: item for item in source.datasets}
        pilot_dir, manifest = build_step7_pilot_bundle(
            launch_id=source.launch_id,
            partition_id=source.partition_id,
            samples=source.samples,
            compiler_runtime_exports={
                dataset: settings.resolve_path(value.compiler_runtime_export_dir)
                for dataset, value in roots.items()
            },
            coverage_exports={
                dataset: settings.resolve_path(value.coverage_export_dir)
                for dataset, value in roots.items()
            },
            query_runtime_exports={
                dataset: settings.resolve_path(value.query_runtime_export_dir)
                for dataset, value in roots.items()
            },
            evaluation_exports={
                dataset: settings.resolve_path(value.evaluation_export_dir)
                for dataset, value in roots.items()
            },
            snapshot_dirs={
                dataset: settings.resolve_path(value.snapshot_dir)
                for dataset, value in roots.items()
            },
            benchmark_root=settings.resolve_benchmark_path(source.benchmark_root),
            output_root=settings.resolve_path(source.output_root),
            safe_execution_log=(
                settings.resolve_path(source.safe_execution_log)
                if source.safe_execution_log is not None
                else None
            ),
        )
    except Exception as exc:
        typer.echo(
            f"ERROR: Step 7 pilot export failed ({type(exc).__name__})", err=True
        )
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "pilot_id": manifest.pilot_id,
                "pilot_dir": str(pilot_dir),
                "status": manifest.status.value,
                "transfer_dir": str(pilot_dir / "transfer"),
            },
            indent=2,
            sort_keys=True,
        )
    )


@runtime_app.command("export-step7b-pilot")
def runtime_export_step7b_pilot(
    source_manifest: Path = typer.Option(..., "--source-manifest"),
) -> None:
    """Build the deterministic Step 7B review and transfer bundle."""

    from pmmc.runtime.step7_pilot import build_step7_pilot_bundle
    from pmmc.schemas import Step7PilotSourceManifest

    settings = get_settings()
    try:
        source = Step7PilotSourceManifest.model_validate_json(
            settings.resolve_path(source_manifest).read_text(encoding="utf-8")
        )
        if source.protocol_version != "pmmc-step7b-executor-pilot-v1":
            raise ValueError("source manifest is not a Step 7B pilot")
        roots = {item.dataset: item for item in source.datasets}
        pilot_dir, manifest = build_step7_pilot_bundle(
            launch_id=source.launch_id,
            partition_id=source.partition_id,
            samples=source.samples,
            compiler_runtime_exports={
                dataset: settings.resolve_path(value.compiler_runtime_export_dir)
                for dataset, value in roots.items()
            },
            coverage_exports={
                dataset: settings.resolve_path(value.coverage_export_dir)
                for dataset, value in roots.items()
            },
            query_runtime_exports={
                dataset: settings.resolve_path(value.query_runtime_export_dir)
                for dataset, value in roots.items()
            },
            evaluation_exports={
                dataset: settings.resolve_path(value.evaluation_export_dir)
                for dataset, value in roots.items()
            },
            snapshot_dirs={
                dataset: settings.resolve_path(value.snapshot_dir)
                for dataset, value in roots.items()
            },
            benchmark_root=settings.resolve_benchmark_path(source.benchmark_root),
            output_root=settings.resolve_path(source.output_root),
            safe_execution_log=(
                settings.resolve_path(source.safe_execution_log)
                if source.safe_execution_log is not None
                else None
            ),
            step7b=True,
        )
    except Exception as exc:
        typer.echo(
            f"ERROR: Step 7B pilot export failed ({type(exc).__name__})", err=True
        )
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "pilot_id": manifest.pilot_id,
                "pilot_dir": str(pilot_dir),
                "status": manifest.status.value,
                "transfer_dir": str(pilot_dir / "transfer"),
            },
            indent=2,
            sort_keys=True,
        )
    )


@runtime_app.command("verify-step7-pilot")
def runtime_verify_step7_pilot(
    pilot_dir: Path | None = typer.Option(None, "--pilot-dir"),
    archive: Path | None = typer.Option(None, "--archive"),
    checksum: Path | None = typer.Option(None, "--checksum"),
) -> None:
    """Verify a Step 7 bundle directory or portable archive."""

    from pmmc.runtime.step7_pilot import (
        verify_step7_pilot_archive,
        verify_step7_pilot_bundle,
    )

    settings = get_settings()
    try:
        if pilot_dir is not None:
            if archive is not None or checksum is not None:
                raise ValueError("--pilot-dir cannot be combined with archive options")
            manifest = verify_step7_pilot_bundle(settings.resolve_path(pilot_dir))
        else:
            if archive is None or checksum is None:
                raise ValueError("provide --pilot-dir or both --archive and --checksum")
            manifest = verify_step7_pilot_archive(
                settings.resolve_path(archive), settings.resolve_path(checksum)
            )
    except Exception as exc:
        typer.echo(
            f"ERROR: Step 7 pilot verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(manifest.model_dump_json(indent=2))


@runtime_app.command("verify-step7b-pilot")
def runtime_verify_step7b_pilot(
    pilot_dir: Path | None = typer.Option(None, "--pilot-dir"),
    archive: Path | None = typer.Option(None, "--archive"),
    checksum: Path | None = typer.Option(None, "--checksum"),
) -> None:
    """Verify a Step 7B directory or portable archive and its protocol tag."""

    from pmmc.runtime.step7_pilot import (
        verify_step7_pilot_archive,
        verify_step7_pilot_bundle,
    )

    settings = get_settings()
    try:
        if pilot_dir is not None:
            if archive is not None or checksum is not None:
                raise ValueError("--pilot-dir cannot be combined with archive options")
            manifest = verify_step7_pilot_bundle(settings.resolve_path(pilot_dir))
        else:
            if archive is None or checksum is None:
                raise ValueError("provide --pilot-dir or both --archive and --checksum")
            manifest = verify_step7_pilot_archive(
                settings.resolve_path(archive), settings.resolve_path(checksum)
            )
        if manifest.protocol_version != "pmmc-step7b-executor-pilot-v1":
            raise ValueError("verified bundle is not a Step 7B pilot")
    except Exception as exc:
        typer.echo(
            f"ERROR: Step 7B pilot verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(manifest.model_dump_json(indent=2))


@runtime_app.command("decide-step7-pilot")
def runtime_decide_step7_pilot(
    pilot_dir: Path = typer.Option(..., "--pilot-dir"),
    decision: str = typer.Option(..., "--decision"),
    reviewer_id: str = typer.Option(..., "--reviewer-id"),
    rationale_file: Path = typer.Option(..., "--rationale-file"),
    match_threshold: float = typer.Option(0.60, "--match-threshold"),
    gap_threshold: float = typer.Option(0.05, "--gap-threshold"),
) -> None:
    """Approve/reject Step 7 and emit all four immutable freezes."""

    from pmmc.runtime.step7_pilot import decide_step7_pilot
    from pmmc.schemas import PilotDecisionValue

    settings = get_settings()
    try:
        decision_dir, questioner, program, qbank, executor = decide_step7_pilot(
            pilot_dir=settings.resolve_path(pilot_dir),
            decision=PilotDecisionValue(decision),
            reviewer_id=reviewer_id,
            rationale=settings.resolve_path(rationale_file)
            .read_text(encoding="utf-8")
            .strip(),
            match_threshold=match_threshold,
            gap_threshold=gap_threshold,
        )
    except Exception as exc:
        typer.echo(
            f"ERROR: Step 7 pilot decision failed ({type(exc).__name__})", err=True
        )
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "decision_dir": str(decision_dir),
                "questioner_freeze_manifest": str(questioner) if questioner else None,
                "compiler_program_freeze_manifest": str(program) if program else None,
                "verified_qbank_freeze_manifest": str(qbank) if qbank else None,
                "query_executor_freeze_manifest": str(executor) if executor else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


@runtime_app.command("decide-step7b-pilot")
def runtime_decide_step7b_pilot(
    pilot_dir: Path = typer.Option(..., "--pilot-dir"),
    decision: str = typer.Option(..., "--decision"),
    reviewer_id: str = typer.Option(..., "--reviewer-id"),
    rationale_file: Path = typer.Option(..., "--rationale-file"),
    match_threshold: float = typer.Option(0.60, "--match-threshold"),
    gap_threshold: float = typer.Option(0.05, "--gap-threshold"),
) -> None:
    """Approve/reject Step 7B after selecting thresholds from its frozen grid."""

    from pmmc.runtime.step7_pilot import decide_step7b_pilot
    from pmmc.schemas import PilotDecisionValue

    settings = get_settings()
    try:
        decision_dir, questioner, program, qbank, executor, aggregate = (
            decide_step7b_pilot(
                pilot_dir=settings.resolve_path(pilot_dir),
                decision=PilotDecisionValue(decision),
                reviewer_id=reviewer_id,
                rationale=settings.resolve_path(rationale_file)
                .read_text(encoding="utf-8")
                .strip(),
                match_threshold=match_threshold,
                gap_threshold=gap_threshold,
            )
        )
    except Exception as exc:
        typer.echo(
            f"ERROR: Step 7B pilot decision failed ({type(exc).__name__})", err=True
        )
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "decision_dir": str(decision_dir),
                "questioner_freeze_manifest": str(questioner) if questioner else None,
                "compiler_program_freeze_manifest": str(program) if program else None,
                "verified_qbank_freeze_manifest": str(qbank) if qbank else None,
                "query_executor_freeze_manifest": str(executor) if executor else None,
                "step7b_freeze_manifest": str(aggregate) if aggregate else None,
                "approved_active_selections": (
                    str(decision_dir / "approved" / "active_selections.jsonl")
                    if aggregate
                    else None
                ),
                "approved_threshold_summary": (
                    str(decision_dir / "approved" / "threshold_summary.json")
                    if aggregate
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _step7b_export_runtime_dir(path: Path) -> Path:
    resolved = get_settings().resolve_path(path)
    runtime = resolved / "runtime"
    if runtime.is_dir():
        return runtime
    if resolved.is_dir():
        return resolved
    raise FileNotFoundError("query runtime export directory is unavailable")


@runtime_app.command("inspect-step7b-route")
def runtime_inspect_step7b_route(
    query_runtime_export: Path = typer.Option(..., "--query-runtime-export"),
    query_id: str = typer.Option(..., "--query-id"),
) -> None:
    """Inspect one saved Step 7B route without running models or retrieval."""

    from pmmc.pilots.io import read_jsonl

    try:
        root = _step7b_export_runtime_dir(query_runtime_export)
        files = {
            "match_batches": "qbank_match_batches.jsonl",
            "fused_candidates": "qbank_match_fused_candidates.jsonl",
            "reranker": "qbank_reranker_assessments.jsonl",
            "routes": "query_route_decisions.jsonl",
            "bindings": "query_program_bindings.jsonl",
            "relevance": "query_evidence_relevance_assessments.jsonl",
            "fallback": "query_fallback_decisions.jsonl",
            "selection": "query_answer_selections.jsonl",
        }
        payload = {
            name: [row for row in read_jsonl(root / filename) if row.get("query_id") == query_id]
            if (root / filename).is_file()
            else []
            for name, filename in files.items()
        }
        if not payload["routes"]:
            raise ValueError("query has no saved Step 7B route")
    except Exception as exc:
        typer.echo(f"ERROR: Step 7B route inspection failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@runtime_app.command("inspect-answer-contract")
def runtime_inspect_answer_contract(
    query_runtime_export: Path = typer.Option(..., "--query-runtime-export"),
    query_id: str = typer.Option(..., "--query-id"),
) -> None:
    """Inspect strict answer parse/retry/reuse facts while omitting answer text."""

    from pmmc.pilots.io import read_jsonl

    try:
        root = _step7b_export_runtime_dir(query_runtime_export)
        facts = [
            row
            for row in read_jsonl(root / "structured_answer_outputs.jsonl")
            if row.get("query_id") == query_id
        ]
        diagnostics = [
            row
            for row in read_jsonl(root / "answer_format_diagnostics.jsonl")
            if row.get("query_id") == query_id
        ]
        reuse = [
            row
            for row in read_jsonl(root / "answer_reuse_links.jsonl")
            if row.get("query_id") == query_id
        ] if (root / "answer_reuse_links.jsonl").is_file() else []
        if not facts and not diagnostics:
            raise ValueError("query has no strict answer contract facts")
        redacted_facts = [
            {key: value for key, value in row.items() if key != "answer"}
            for row in facts
        ]
    except Exception as exc:
        typer.echo(f"ERROR: answer contract inspection failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {"structured_outputs": redacted_facts, "diagnostics": diagnostics, "reuse": reuse},
            indent=2,
            sort_keys=True,
        )
    )


@runtime_app.command("compare-step7b-pilot")
def runtime_compare_step7b_pilot(
    step7b_pilot_dir: Path = typer.Option(..., "--step7b-pilot-dir"),
    step7_pilot_dir: Path = typer.Option(..., "--step7-pilot-dir"),
) -> None:
    """Compare public Step 7 and Step 7B summaries without private payloads."""

    settings = get_settings()
    try:
        current_root = settings.resolve_path(step7b_pilot_dir)
        prior_root = settings.resolve_path(step7_pilot_dir)
        current = json.loads(
            (current_root / "public" / "automatic_summary.json").read_text(encoding="utf-8")
        )
        prior = json.loads(
            (prior_root / "public" / "automatic_summary.json").read_text(encoding="utf-8")
        )
        payload = {
            "step7b_pilot_id": json.loads(
                (current_root / "manifest.json").read_text(encoding="utf-8")
            )["pilot_id"],
            "step7_pilot_id": json.loads(
                (prior_root / "manifest.json").read_text(encoding="utf-8")
            )["pilot_id"],
            "step7b": current,
            "step7": prior,
        }
    except Exception as exc:
        typer.echo(f"ERROR: Step 7B comparison failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@runtime_app.command("verify-query-executor-freeze")
def runtime_verify_query_executor_freeze(
    freeze_manifest: Path = typer.Option(..., "--freeze-manifest"),
) -> None:
    """Verify one approved QueryExecutorFreezeManifest."""

    from pmmc.runtime.step7_pilot import verify_query_executor_freeze

    try:
        freeze = verify_query_executor_freeze(
            get_settings().resolve_path(freeze_manifest)
        )
    except Exception as exc:
        typer.echo(
            f"ERROR: query executor freeze verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(freeze.model_dump_json(indent=2))


@runtime_app.command("evaluate-suite")
def runtime_evaluate_suite(
    suite_id: str = typer.Option(..., "--suite-id"),
    profile: Path | None = typer.Option(None, "--profile"),
) -> None:
    """Run CPU metrics, one managed judge stage, then deterministic analysis."""

    from pmmc.evaluation.workflow import EvaluationRunPhase, run_evaluation_workflow
    from pmmc.runtime.manager import run_managed_command

    settings = get_settings()
    try:
        manifest, _directory = asyncio.run(
            run_evaluation_workflow(
                settings=settings,
                suite_id=suite_id,
                phase=EvaluationRunPhase.DETERMINISTIC,
            )
        )
        judge_command = [
            sys.executable,
            "-m",
            "pmmc",
            "evaluate",
            "run",
            "--resume-evaluation-run-id",
            manifest.evaluation_run_id,
            "--phase",
            "judge",
        ]
        _code, runtime_dir = run_managed_command(
            settings,
            profile_path=profile,
            logical_stage="judge",
            command=judge_command,
        )
        asyncio.run(
            run_evaluation_workflow(
                settings=settings,
                resume_evaluation_run_id=manifest.evaluation_run_id,
                phase=EvaluationRunPhase.ANALYZE,
            )
        )
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(f"ERROR: managed evaluation failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "evaluation_run_id": manifest.evaluation_run_id,
                "evaluation_artifacts": str(
                    settings.runs_path / "evaluations" / manifest.evaluation_run_id
                ),
                "judge_runtime_artifacts": str(runtime_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _resolved_experiment_config(
    *,
    config_path: Path,
    mode: AppMode | None,
    datasets: list[DatasetName] | None,
    methods: list[BaselineMethod] | None,
    answer_model: str | None,
    judge_model: str | None,
    text_embedding_model: str | None,
    contrastive_embedding_model: str | None,
    multimodal_embedding_model: str | None,
    compiler_model: str | None,
    qbank_embedding_model: str | None,
    reranker_model: str | None,
    model_registry_overlays: list[Path] | None,
    question_types: list[QuestionType] | None,
    source_categories: list[str] | None,
    scenarios: list[str] | None,
    query_ids: list[str] | None,
    limit: int | None,
    sample_size: int | None,
    seed: int | None,
    shard_count: int | None,
    shard_index: int | None,
    include_excluded: bool,
    partition_manifest: Path | None,
    partition_role: PartitionRole | None,
    questioner_freeze_manifest: Path | None,
    retrieval_top_k: int | None,
    answer_top_k: int | None,
    total_context_tokens: int | None,
    reserved_output_tokens: int | None,
    answer_batch_size: int | None,
    embedding_batch_size: int | None,
    multimodal_embedding_batch_size: int | None,
    model_concurrency: int | None,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    download_policy: DownloadPolicy | None,
    verification_policy: ModelVerificationPolicy | None,
    calibration_policy: CalibrationPolicy | None,
    no_evaluate: bool,
):
    from pmmc.experiments.launcher import ExperimentLauncherConfig, load_launcher_config

    settings = get_settings()
    resolved_path = settings.resolve_path(config_path)
    value = load_launcher_config(resolved_path)
    model_updates = {
        key: item
        for key, item in {
            "answer_model": answer_model,
            "judge_model": judge_model,
            "text_embedding_model": text_embedding_model,
            "contrastive_embedding_model": contrastive_embedding_model,
            "multimodal_embedding_model": multimodal_embedding_model,
            "compiler_model": compiler_model,
            "qbank_embedding_model": qbank_embedding_model,
            "reranker_model": reranker_model,
        }.items()
        if item is not None
    }
    selection_updates: dict[str, object] = {}
    if datasets:
        selection_updates["datasets"] = datasets
    if question_types:
        selection_updates["question_types"] = question_types
    if source_categories:
        selection_updates["source_categories"] = source_categories
    if scenarios:
        selection_updates["scenario_ids"] = scenarios
    if query_ids:
        selection_updates["query_ids"] = query_ids
    numeric_updates: dict[str, int | None] = {
        "limit": limit,
        "sample_size": sample_size,
        "seed": seed,
        "shard_count": shard_count,
        "shard_index": shard_index,
    }
    for key, item in numeric_updates.items():
        if item is not None:
            selection_updates[key] = item
    if include_excluded:
        selection_updates["include_excluded"] = True
    if partition_manifest is not None:
        selection_updates["partition_manifest_path"] = partition_manifest
    if partition_role is not None:
        selection_updates["partition_role"] = partition_role
    top_updates: dict[str, object] = {}
    if mode is not None:
        top_updates["mode"] = mode
    if methods:
        top_updates["methods"] = methods
    if questioner_freeze_manifest is not None:
        top_updates["questioner_freeze_manifest_path"] = questioner_freeze_manifest
    payload = value.model_dump(mode="python")
    if model_updates:
        payload["models"].update(model_updates)
    if selection_updates:
        payload["selection"].update(selection_updates)
    if model_registry_overlays:
        top_updates["model_registry_overlays"] = model_registry_overlays
    launcher_numeric_updates: dict[str, int | float | None] = {
        "retrieval_top_k": retrieval_top_k,
        "answer_top_k": answer_top_k,
        "total_context_tokens": total_context_tokens,
        "reserved_output_tokens": reserved_output_tokens,
        "answer_batch_size": answer_batch_size,
        "embedding_batch_size": embedding_batch_size,
        "multimodal_embedding_batch_size": multimodal_embedding_batch_size,
        "model_concurrency": model_concurrency,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    for key, launcher_value in launcher_numeric_updates.items():
        if launcher_value is not None:
            top_updates[key] = launcher_value
    if download_policy is not None:
        top_updates["download_policy"] = download_policy
    if verification_policy is not None:
        top_updates["model_verification_policy"] = verification_policy
    if calibration_policy is not None:
        top_updates["calibration_policy"] = calibration_policy
    if no_evaluate:
        top_updates["evaluate"] = False
    payload.update(top_updates)
    return ExperimentLauncherConfig.model_validate(payload)


@experiment_app.command("run")
def experiment_run(
    config: Path = typer.Option(..., "--config"),
    mode: AppMode | None = typer.Option(None, "--mode"),
    dataset: list[DatasetName] | None = typer.Option(None, "--dataset"),
    method: list[BaselineMethod] | None = typer.Option(None, "--method"),
    answer_model: str | None = typer.Option(None, "--answer-model"),
    judge_model: str | None = typer.Option(None, "--judge-model"),
    text_embedding_model: str | None = typer.Option(None, "--text-embedding-model"),
    contrastive_embedding_model: str | None = typer.Option(None, "--contrastive-embedding-model"),
    multimodal_embedding_model: str | None = typer.Option(None, "--multimodal-embedding-model"),
    compiler_model: str | None = typer.Option(None, "--compiler-model"),
    qbank_embedding_model: str | None = typer.Option(None, "--qbank-embedding-model"),
    reranker_model: str | None = typer.Option(None, "--reranker-model"),
    model_registry_overlay: list[Path] | None = typer.Option(None, "--model-registry-overlay"),
    question_type: list[QuestionType] | None = typer.Option(None, "--question-type"),
    source_category: list[str] | None = typer.Option(None, "--source-category"),
    scenario: list[str] | None = typer.Option(None, "--scenario"),
    query_id: list[str] | None = typer.Option(None, "--query-id"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    sample_size: int | None = typer.Option(None, "--sample-size", min=1),
    seed: int | None = typer.Option(None, "--seed", min=0),
    shard_count: int | None = typer.Option(None, "--shard-count", min=1),
    shard_index: int | None = typer.Option(None, "--shard-index", min=0),
    include_excluded: bool = typer.Option(False, "--include-excluded"),
    partition_manifest: Path | None = typer.Option(None, "--partition-manifest"),
    partition_role: PartitionRole | None = typer.Option(None, "--partition-role"),
    questioner_freeze_manifest: Path | None = typer.Option(None, "--questioner-freeze-manifest"),
    retrieval_top_k: int | None = typer.Option(None, "--retrieval-top-k", min=1),
    answer_top_k: int | None = typer.Option(None, "--answer-top-k", min=1),
    total_context_tokens: int | None = typer.Option(None, "--context-tokens", min=2048),
    reserved_output_tokens: int | None = typer.Option(None, "--reserved-output-tokens", min=1),
    answer_batch_size: int | None = typer.Option(None, "--answer-batch-size", min=1),
    embedding_batch_size: int | None = typer.Option(None, "--embedding-batch-size", min=1),
    multimodal_embedding_batch_size: int | None = typer.Option(
        None, "--multimodal-embedding-batch-size", min=1
    ),
    model_concurrency: int | None = typer.Option(None, "--model-concurrency", min=1),
    tensor_parallel_size: int | None = typer.Option(None, "--tensor-parallel-size", min=1),
    gpu_memory_utilization: float | None = typer.Option(
        None, "--gpu-memory-utilization", min=0.01, max=1.0
    ),
    download_policy: DownloadPolicy | None = typer.Option(None, "--download-policy"),
    verification_policy: ModelVerificationPolicy | None = typer.Option(
        None, "--verification-policy"
    ),
    calibration_policy: CalibrationPolicy | None = typer.Option(None, "--calibration-policy"),
    resume_launch_id: str | None = typer.Option(None, "--resume-launch-id"),
    import_upstream_from_launch: str | None = typer.Option(None, "--import-upstream-from-launch"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    prepare_only: bool = typer.Option(False, "--prepare-only"),
    evaluate_only: bool = typer.Option(False, "--evaluate-only"),
    no_evaluate: bool = typer.Option(False, "--no-evaluate"),
    quiet_summary: bool = typer.Option(
        False,
        "--quiet-summary",
        help="Print one safe summary line instead of the full report payload.",
    ),
) -> None:
    """Prepare models and execute a complete baseline experiment with one command."""

    from pmmc.experiments.launcher import run_launcher_command

    try:
        resolved = _resolved_experiment_config(
            config_path=config,
            mode=mode,
            datasets=dataset,
            methods=method,
            answer_model=answer_model,
            judge_model=judge_model,
            text_embedding_model=text_embedding_model,
            contrastive_embedding_model=contrastive_embedding_model,
            multimodal_embedding_model=multimodal_embedding_model,
            compiler_model=compiler_model,
            qbank_embedding_model=qbank_embedding_model,
            reranker_model=reranker_model,
            model_registry_overlays=model_registry_overlay,
            question_types=question_type,
            source_categories=source_category,
            scenarios=scenario,
            query_ids=query_id,
            limit=limit,
            sample_size=sample_size,
            seed=seed,
            shard_count=shard_count,
            shard_index=shard_index,
            include_excluded=include_excluded,
            partition_manifest=partition_manifest,
            partition_role=partition_role,
            questioner_freeze_manifest=questioner_freeze_manifest,
            retrieval_top_k=retrieval_top_k,
            answer_top_k=answer_top_k,
            total_context_tokens=total_context_tokens,
            reserved_output_tokens=reserved_output_tokens,
            answer_batch_size=answer_batch_size,
            embedding_batch_size=embedding_batch_size,
            multimodal_embedding_batch_size=multimodal_embedding_batch_size,
            model_concurrency=model_concurrency,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            download_policy=download_policy,
            verification_policy=verification_policy,
            calibration_policy=calibration_policy,
            no_evaluate=no_evaluate,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        result = run_launcher_command(
            settings=get_settings(),
            config=resolved,
            resume_launch_id=resume_launch_id,
            import_upstream_from_launch=import_upstream_from_launch,
            dry_run=dry_run,
            prepare_only=prepare_only,
            evaluate_only=evaluate_only,
        )
    except FileExistsError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.echo(f"ERROR: research experiment failed (FileNotFoundError): {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(f"ERROR: research experiment failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    if quiet_summary:
        execution = result.get("execution")
        datasets = result.get("datasets")
        typer.echo(
            "EXPERIMENT_SUMMARY "
            + json.dumps(
                {
                    "dataset_count": len(datasets) if isinstance(datasets, dict) else 0,
                    "launch_id": result.get("launch_id"),
                    "restored": (
                        bool(execution.get("restored_checkpoint"))
                        if isinstance(execution, dict)
                        else False
                    ),
                    "status": result.get("status"),
                    "target": result.get("target"),
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return
    summary_path = Path(str(result["launch_directory"])) / "final_summary.md"
    if summary_path.is_file():
        typer.echo(summary_path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


@experiment_app.command("render-job")
def experiment_render_job(
    config: Path = typer.Option(..., "--config"),
    deployment: Path = typer.Option(..., "--deployment"),
    output: Path = typer.Option(Path("job.generated.yaml"), "--output"),
    gpus: int | None = typer.Option(None, "--gpus", min=1),
    cpu: str | None = typer.Option(None, "--cpu"),
    memory: str | None = typer.Option(None, "--memory"),
    shared_memory: str | None = typer.Option(None, "--shared-memory"),
    postgres_cpu: str | None = typer.Option(None, "--postgres-cpu"),
    postgres_memory: str | None = typer.Option(None, "--postgres-memory"),
    minio_cpu: str | None = typer.Option(None, "--minio-cpu"),
    minio_memory: str | None = typer.Option(None, "--minio-memory"),
    scratch_size: str | None = typer.Option(None, "--scratch-size"),
    job_name_prefix: str | None = typer.Option(None, "--job-name-prefix"),
    image: str | None = typer.Option(None, "--image"),
    dataset: list[DatasetName] | None = typer.Option(None, "--dataset"),
    method: list[BaselineMethod] | None = typer.Option(None, "--method"),
    partition_manifest: Path | None = typer.Option(None, "--partition-manifest"),
    partition_role: PartitionRole | None = typer.Option(None, "--partition-role"),
    questioner_freeze_manifest: Path | None = typer.Option(None, "--questioner-freeze-manifest"),
    verification_policy: ModelVerificationPolicy | None = typer.Option(
        None, "--verification-policy"
    ),
    answer_model: str | None = typer.Option(None, "--answer-model"),
    judge_model: str | None = typer.Option(None, "--judge-model"),
    text_embedding_model: str | None = typer.Option(None, "--text-embedding-model"),
    contrastive_embedding_model: str | None = typer.Option(None, "--contrastive-embedding-model"),
    multimodal_embedding_model: str | None = typer.Option(None, "--multimodal-embedding-model"),
    compiler_model: str | None = typer.Option(None, "--compiler-model"),
    qbank_embedding_model: str | None = typer.Option(None, "--qbank-embedding-model"),
    reranker_model: str | None = typer.Option(None, "--reranker-model"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    resume_launch_id: str | None = typer.Option(None, "--resume-launch-id"),
    import_upstream_from_launch: str | None = typer.Option(None, "--import-upstream-from-launch"),
) -> None:
    """Render a credential-free Job that runs the experiment as its main process."""

    from pmmc.experiments.kubernetes import (
        load_deployment_config,
        render_command,
        render_experiment_job,
        write_job_yaml,
    )
    from pmmc.experiments.launcher import (
        ExperimentLauncherConfig,
        ExperimentTarget,
        load_launcher_config,
        resolve_launcher_identity,
    )

    settings = get_settings()
    try:
        config_path = settings.resolve_path(config)
        deployment_path = settings.resolve_path(deployment)
        experiment = load_launcher_config(config_path)
        deployment_config = load_deployment_config(deployment_path)
        if image is not None:
            deployment_payload = deployment_config.model_dump(mode="python")
            deployment_payload["image"] = image
            deployment_config = type(deployment_config).model_validate(deployment_payload)
        experiment_payload = experiment.model_dump(mode="python")
        if dataset:
            experiment_payload["selection"]["datasets"] = dataset
        if method:
            experiment_payload["methods"] = method
        if limit is not None:
            experiment_payload["selection"]["limit"] = limit
            experiment_payload["selection"]["sample_size"] = None
        if partition_manifest is not None:
            experiment_payload["selection"]["partition_manifest_path"] = partition_manifest
        if partition_role is not None:
            experiment_payload["selection"]["partition_role"] = partition_role
        if questioner_freeze_manifest is not None:
            experiment_payload["questioner_freeze_manifest_path"] = questioner_freeze_manifest
        if verification_policy is not None:
            experiment_payload["model_verification_policy"] = verification_policy
        model_values = {
            "answer_model": answer_model,
            "judge_model": judge_model,
            "text_embedding_model": text_embedding_model,
            "contrastive_embedding_model": contrastive_embedding_model,
            "multimodal_embedding_model": multimodal_embedding_model,
            "compiler_model": compiler_model,
            "qbank_embedding_model": qbank_embedding_model,
            "reranker_model": reranker_model,
        }
        experiment_payload["models"].update(
            {key: value for key, value in model_values.items() if value is not None}
        )
        rendered_experiment = ExperimentLauncherConfig.model_validate(experiment_payload)
        resume_check_events = {
            ExperimentTarget.PMMC_COMPILER_VERIFIED_QBANK_PILOT: (
                "STEP6D_RENDER_RESUME_CHECK"
            ),
            ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT: "STEP7_RENDER_RESUME_CHECK",
            ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT: "STEP7B_RENDER_RESUME_CHECK",
        }
        resume_check_event = resume_check_events.get(rendered_experiment.target)
        if import_upstream_from_launch is not None:
            from pmmc.experiments.upstream_import import validate_import_id

            validate_import_id(import_upstream_from_launch)
            if (
                resume_launch_id is not None
                or rendered_experiment.target is not ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
            ):
                raise ValueError("upstream_import_requires_step7b_without_resume")
            typer.echo("STEP7B_RENDER_UPSTREAM_IMPORT validation=runtime_before_models "
                       f"source_launch_id={import_upstream_from_launch}")
        if resume_launch_id is not None and resume_check_event is not None:
            expected_launch_id, _config_hash, _source_inputs = resolve_launcher_identity(
                settings=settings,
                config=rendered_experiment,
            )
            compatible = resume_launch_id == expected_launch_id
            typer.echo(
                f"{resume_check_event} "
                f"requested={resume_launch_id} expected={expected_launch_id} "
                f"compatible={str(compatible).lower()}"
                + ("" if compatible else " action=start_fresh_without_resume"),
                err=not compatible,
            )
            if not compatible:
                raise typer.Exit(code=2)
        overrides: list[str] = []
        _append_repeated_option(overrides, "--dataset", dataset)
        _append_repeated_option(overrides, "--method", method)
        if partition_manifest is not None:
            overrides.extend(["--partition-manifest", str(partition_manifest)])
        if partition_role is not None:
            overrides.extend(["--partition-role", partition_role.value])
        if questioner_freeze_manifest is not None:
            overrides.extend(["--questioner-freeze-manifest", str(questioner_freeze_manifest)])
        if verification_policy is not None:
            overrides.extend(["--verification-policy", verification_policy.value])
        for name, value in (
            ("--answer-model", answer_model),
            ("--judge-model", judge_model),
            ("--text-embedding-model", text_embedding_model),
            ("--contrastive-embedding-model", contrastive_embedding_model),
            ("--multimodal-embedding-model", multimodal_embedding_model),
            ("--compiler-model", compiler_model),
            ("--qbank-embedding-model", qbank_embedding_model),
            ("--reranker-model", reranker_model),
        ):
            if value is not None:
                overrides.extend([name, value])
        if limit is not None:
            overrides.extend(["--limit", str(limit)])
        if resume_launch_id is not None:
            overrides.extend(["--resume-launch-id", resume_launch_id])
        if import_upstream_from_launch is not None:
            overrides.extend(["--import-upstream-from-launch", import_upstream_from_launch])
        resources = render_experiment_job(
            project_root=settings.project_root,
            experiment_config_path=config_path,
            experiment=rendered_experiment,
            deployment=deployment_config,
            gpus=gpus,
            cpu=cpu,
            memory=memory,
            shared_memory=shared_memory,
            postgres_cpu=postgres_cpu,
            postgres_memory=postgres_memory,
            minio_cpu=minio_cpu,
            minio_memory=minio_memory,
            scratch_size=scratch_size,
            job_name_prefix=job_name_prefix,
            cli_overrides=overrides,
        )
        destination = settings.resolve_path(output)
        write_job_yaml(resources, destination)
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"ERROR: Kubernetes Job rendering failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Rendered Kubernetes resources: {destination}")
    typer.echo(f"Container image: {deployment_config.image}")
    if deployment_config.image.endswith(":latest"):
        typer.echo(
            "WARNING: ':latest' is mutable; use --image with a unique tag or digest "
            "to guarantee that Kubernetes runs the image you tested.",
            err=True,
        )
    typer.echo(render_command(resources, destination))


@app.command("validate-data")
def validate_data(
    dataset: str = typer.Option(
        "all",
        "--dataset",
        help="Benchmark to validate: all, memlens, or memgallery.",
    ),
    level: ValidationLevel | None = typer.Option(
        None,
        "--level",
        help="Validation depth. Defaults to full in paper mode and fast otherwise.",
    ),
) -> None:
    """Validate raw benchmark structure, references, paths, and package counts."""

    settings = get_settings()
    datasets = _parse_datasets(dataset)
    resolved_level = level or _default_validation_level(settings)
    run = _run_context(
        settings=settings,
        command="validate-data",
        datasets=datasets,
        level=resolved_level,
    )
    try:
        with run:
            for dataset_name in datasets:
                adapter = _adapter_for(settings, dataset_name)
                expected = adapter.expected_counts()
                total = expected.get("assets", 0) + adapter.source_record_count
                with run.stage(
                    f"validate-{dataset_name.value}",
                    dataset=dataset_name,
                    total=total,
                ) as stage:
                    report = validate_adapter(
                        adapter,
                        resolved_level,
                        progress=_progress_callback(stage),
                    )
                    (run.run_dir / f"validation_{dataset_name.value}.json").write_text(
                        report.to_json() + "\n",
                        encoding="utf-8",
                    )
                    _print_validation_report(report)
                    if not report.success:
                        raise BenchmarkDataError(
                            f"{dataset_name.value} benchmark validation failed"
                        )
    except (BenchmarkDataError, FileNotFoundError, OSError, ValueError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Run artifacts: {run.run_dir}")


@app.command("import-dataset")
def import_dataset(
    dataset: str = typer.Option(
        "all",
        "--dataset",
        help="Benchmark to import: all, memlens, or memgallery.",
    ),
    level: ValidationLevel | None = typer.Option(
        None,
        "--level",
        help="Validation depth. Defaults to full in paper mode and fast otherwise.",
    ),
) -> None:
    """Validate raw assets and create gold-isolated canonical JSONL snapshots."""

    settings = get_settings()
    datasets = _parse_datasets(dataset)
    resolved_level = level or _default_validation_level(settings)
    run = _run_context(
        settings=settings,
        command="import-dataset",
        datasets=datasets,
        level=resolved_level,
    )
    try:
        with run:
            for dataset_name in datasets:
                adapter = _adapter_for(settings, dataset_name)
                expected = adapter.expected_counts()
                one_pass_total = expected.get("assets", 0) + adapter.source_record_count
                with run.stage(
                    f"import-{dataset_name.value}",
                    dataset=dataset_name,
                    total=one_pass_total * 2,
                ) as stage:
                    snapshot_path, manifest = build_snapshot(
                        adapter,
                        output_root=settings.benchmark_artifacts_path,
                        level=resolved_level,
                        progress=_progress_callback(stage),
                    )
                    run.add_snapshot_id(manifest.snapshot_id)
                    typer.echo(f"{dataset_name.value.upper()} snapshot: {manifest.snapshot_id}")
                    typer.echo(
                        "  "
                        f"queries={manifest.counts.get('queries', 0)} "
                        f"eligible={manifest.counts.get('eligible_queries', 0)} "
                        f"excluded={manifest.counts.get('excluded_queries', 0)} "
                        f"memory_units={manifest.counts.get('memory_units', 0)} "
                        f"assets={manifest.counts.get('assets', 0)}"
                    )
                    typer.echo(f"  path={snapshot_path}")
    except (BenchmarkDataError, FileNotFoundError, OSError, ValueError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Run artifacts: {run.run_dir}")


@app.command("inspect-data")
def inspect_data(
    snapshot: str = typer.Option(..., "--snapshot", help="Canonical snapshot ID."),
    query_id: str | None = typer.Option(None, "--query-id"),
    memory_id: str | None = typer.Option(None, "--memory-id"),
    include_gold: bool = typer.Option(
        False,
        "--include-gold",
        help="Explicitly include evaluation-only gold for a query.",
    ),
) -> None:
    """Inspect canonical records and their raw source locators."""

    if (query_id is None) == (memory_id is None):
        raise typer.BadParameter("provide exactly one of --query-id or --memory-id")
    if include_gold and query_id is None:
        raise typer.BadParameter("--include-gold is only valid with --query-id")
    if "/" in snapshot or "\\" in snapshot or snapshot in {".", ".."}:
        raise typer.BadParameter("snapshot must be an ID, not a path", param_hint="--snapshot")

    settings = get_settings()
    try:
        snapshot_path = find_snapshot(settings.benchmark_artifacts_path, snapshot)
        payload: dict[str, object]
        if query_id is not None:
            query_record = next(
                (
                    record
                    for record in RuntimeQueryStore(snapshot_path).iter_records()
                    if record.query.query_id == query_id
                ),
                None,
            )
            if query_record is None:
                raise KeyError(f"query does not exist in snapshot: {query_id}")
            payload = {"query": query_record.model_dump(mode="json")}
            if include_gold:
                payload["gold"] = GoldStore(snapshot_path).get(query_id).model_dump(mode="json")
        else:
            memory_record = next(
                (
                    record
                    for record in MemoryCorpus(snapshot_path).iter_all()
                    if record.memory_id == memory_id
                ),
                None,
            )
            if memory_record is None:
                raise KeyError(f"memory does not exist in snapshot: {memory_id}")
            payload = {"memory": memory_record.model_dump(mode="json")}
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _find_corpus(root: Path, corpus_id: str) -> Path:
    if Path(corpus_id).name != corpus_id or "\\" in corpus_id or corpus_id in {".", ".."}:
        raise ValueError("corpus_id must be a single safe path component")
    matches = sorted(root.glob(f"*/*/{corpus_id}"))
    if not matches:
        raise FileNotFoundError(f"corpus does not exist: {corpus_id}")
    if len(matches) != 1:
        raise ValueError(f"corpus ID is ambiguous: {corpus_id}")
    return matches[0]


@baseline_app.command("build-corpus")
def baseline_build_corpus(
    snapshot: str = typer.Option(..., "--snapshot", help="Canonical snapshot ID."),
    tokenizer_alias: str = typer.Option("qwen3-embedding-8b", "--tokenizer-alias"),
) -> None:
    """Build one immutable snapshot-level corpus without running embeddings."""

    from pmmc.baselines import build_corpus
    from pmmc.models import (
        HeuristicTokenCounter,
        TransformersTokenCounter,
    )
    from pmmc.models.protocols import TokenCounter

    settings = get_settings()
    try:
        snapshot_path = find_snapshot(settings.benchmark_artifacts_path, snapshot)
        registry = load_model_registry(settings.resolved_model_registry_path)
        entry = registry.get(tokenizer_alias)
        counter: TokenCounter
        if settings.mode is AppMode.PAPER:
            registry.assert_paper_ready()
            counter = TransformersTokenCounter(
                entry.tokenizer_name,
                entry.tokenizer_revision,
            )
        else:
            counter = HeuristicTokenCounter(
                entry.tokenizer_name,
                entry.tokenizer_revision,
            )
        run = _persistence_run_context(
            settings,
            command="baseline-build-corpus",
            snapshot_path=snapshot_path,
            method="baseline_corpus_foundation",
            registry=registry,
        )
        with run:
            with run.stage("build-baseline-corpus", dataset=run.datasets[0]) as stage:
                corpus_path, manifest = build_corpus(
                    snapshot_path,
                    output_root=settings.baseline_corpora_path,
                    counter=counter,
                    progress=_progress_callback(stage),
                )
    except Exception as exc:
        typer.echo(f"ERROR: corpus build failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Corpus ready: corpus_id={manifest.corpus_id} windows={manifest.window_count} "
        f"documents={manifest.document_count} path={corpus_path}"
    )
    typer.echo(f"Run artifacts: {run.run_dir}")


@baseline_app.command("validate-corpus")
def baseline_validate_corpus(
    corpus_dir: Path = typer.Option(..., "--corpus-dir"),
) -> None:
    """Validate local corpus hashes, counts, lineage, and gold isolation."""

    from pmmc.baselines import validate_corpus

    try:
        manifest = validate_corpus(corpus_dir)
    except Exception as exc:
        typer.echo(f"ERROR: corpus validation failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Corpus valid: corpus_id={manifest.corpus_id} windows={manifest.window_count} "
        f"documents={manifest.document_count} hash={manifest.content_hash}"
    )


@baseline_app.command("inspect-corpus")
def baseline_inspect_corpus(
    corpus_id: str = typer.Option(..., "--corpus-id"),
) -> None:
    """Print non-sensitive corpus lineage and representation counts."""

    from collections import Counter

    from pmmc.baselines import CorpusReader

    settings = get_settings()
    try:
        corpus_path = _find_corpus(settings.baseline_corpora_path, corpus_id)
        reader = CorpusReader(corpus_path)
        representation_counts = Counter(
            document.representation.value for document in reader.iter_documents()
        )
    except Exception as exc:
        typer.echo(f"ERROR: corpus inspection failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "manifest": reader.manifest.model_dump(mode="json"),
                "representation_counts": dict(sorted(representation_counts.items())),
                "path": str(corpus_path),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


def _parse_rag_method(value: str) -> BaselineMethod:
    try:
        method = BaselineMethod(value)
    except ValueError as exc:
        raise typer.BadParameter(
            "must be caption_text_rag, contrastive_multimodal_rag, or native_multimodal_rag",
            param_hint="--method",
        ) from exc
    from pmmc.baselines.rag_embedding import RAG_METHODS

    if method not in RAG_METHODS:
        raise typer.BadParameter("method is not a Step 5C RAG baseline", param_hint="--method")
    return method


@baseline_app.command("build-index")
def baseline_build_index(
    method: str = typer.Option(..., "--method"),
    corpus_id: str = typer.Option(..., "--corpus-id"),
) -> None:
    """Build or deterministically reuse one snapshot-level RAG embedding index."""

    from pmmc.baselines import RagIndexBuildResult, build_rag_index, load_rag_config

    settings = get_settings()
    resolved_method = _parse_rag_method(method)
    started_at = time.monotonic()
    completed_documents = 0
    total_documents = 0
    last_progress_at = started_at
    last_progress_counts: tuple[int, int] | None = None

    def emit_progress(*, force: bool = False) -> None:
        nonlocal last_progress_at, last_progress_counts
        now = time.monotonic()
        counts = (completed_documents, total_documents)
        if now - last_progress_at < 30 and (not force or counts == last_progress_counts):
            return
        percent = (completed_documents / total_documents) * 100 if total_documents else 0.0
        typer.echo(
            "INDEX_PROGRESS "
            f"method={resolved_method.value} "
            f"documents={completed_documents}/{total_documents or '?'} "
            f"percent={percent:.1f}% elapsed={now - started_at:.1f}s"
        )
        last_progress_at = now
        last_progress_counts = counts

    def progress(delta: int, total: int) -> None:
        nonlocal completed_documents, total_documents
        total_documents = total
        completed_documents = min(total, completed_documents + delta)
        emit_progress(force=delta == 0 or completed_documents == total_documents)

    async def run_index() -> RagIndexBuildResult:
        task = asyncio.create_task(
            build_rag_index(
                settings=settings,
                registry=registry,
                config=config,
                corpus_id=corpus_id,
                method=resolved_method,
                progress=progress,
            )
        )
        while not task.done():
            done, _pending = await asyncio.wait({task}, timeout=10)
            if not done:
                emit_progress()
        return await task

    typer.echo(f"INDEX_START method={resolved_method.value} corpus_id={corpus_id}")
    try:
        registry = load_model_registry(settings.resolved_model_registry_path)
        config = load_rag_config(settings.resolved_rag_config_path)
        result = asyncio.run(run_index())
    except Exception as exc:
        summary = " ".join(str(exc).split())
        safe_summary = (
            summary
            if summary.endswith("embedding index batch(es) failed")
            and summary.split(maxsplit=1)[0].isdigit()
            else ""
        )
        suffix = f": {safe_summary}" if safe_summary else ""
        typer.echo(
            f"ERROR: RAG index build failed ({type(exc).__name__}){suffix}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "collection_id": result.collection.collection_id,
                "document_count": result.document_count,
                "method": resolved_method.value,
                "reused": result.reused,
                "run_id": result.run_id,
                "status": result.collection.status.value,
            },
            indent=2,
            sort_keys=True,
        )
    )


@baseline_app.command("verify-index")
def baseline_verify_index(
    method: str = typer.Option(..., "--method"),
    corpus_id: str = typer.Option(..., "--corpus-id"),
) -> None:
    """Verify collection identity, readiness, dimensions, and document coverage."""

    from pmmc.baselines import load_rag_config
    from pmmc.baselines.rag_embedding import make_document_collection
    from pmmc.infrastructure.database.baseline_repositories import PostgresCorpusRepository
    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork

    settings = get_settings()
    resolved_method = _parse_rag_method(method)

    async def verify() -> dict[str, object]:
        registry = load_model_registry(settings.resolved_model_registry_path)
        config = load_rag_config(settings.resolved_rag_config_path)
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                corpora = PostgresCorpusRepository(session)
                corpus = await corpora.get_manifest(corpus_id)
                document_ids = await corpora.list_document_ids(
                    corpus_id, list(config.representations_for(resolved_method))
                )
            template = make_document_collection(
                corpus=corpus,
                method=resolved_method,
                config=config,
                registry=registry,
                run_id="verification-placeholder",
            )
            async with ExperimentUnitOfWork(sessions) as uow:
                assert uow.embeddings is not None
                collection = await uow.embeddings.get_collection(template.collection_id)
                if collection is None:
                    raise KeyError(f"RAG index does not exist: {template.collection_id}")
                count = await uow.embeddings.count_records(collection.collection_id)
            issues: list[str] = []
            if collection.status.value != "ready":
                issues.append("collection is not ready")
            if collection.corpus_id != corpus_id:
                issues.append("collection corpus lineage differs")
            if collection.dimension != template.dimension:
                issues.append("embedding dimension differs from registry")
            if collection.actual_model_id != template.actual_model_id:
                issues.append("actual model identity differs from registry")
            if count != len(document_ids):
                issues.append("embedding record count differs from eligible document count")
            return {
                "collection_id": collection.collection_id,
                "document_count": len(document_ids),
                "embedding_count": count,
                "issues": issues,
                "method": resolved_method.value,
                "valid": not issues,
            }
        finally:
            await engine.dispose()

    try:
        report = asyncio.run(verify())
    except Exception as exc:
        typer.echo(f"ERROR: RAG index verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(code=1)


@baseline_app.command("inspect-retrieval")
def baseline_inspect_retrieval(
    run_id: str = typer.Option(..., "--run-id"),
    query_id: str = typer.Option(..., "--query-id"),
) -> None:
    """Inspect runtime-safe window ranks and embedding lineage for one query."""

    from pmmc.baselines.rag_retrieval import retrieval_batch_id
    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
    from pmmc.infrastructure.experiment_runs import load_local_run_registration

    settings = get_settings()
    try:
        local = load_local_run_registration(settings.runs_path / run_id)
        method = _parse_rag_method(local.manifest.method)
        batch_id = retrieval_batch_id(run_id, query_id, method)

        async def inspect() -> dict[str, object]:
            engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
            try:
                async with ExperimentUnitOfWork(sessions) as uow:
                    assert uow.retrieval is not None
                    facts = await uow.retrieval.get_batch(batch_id)
                if facts is None:
                    raise KeyError(f"retrieval batch does not exist: {batch_id}")
                batch, traces, components, query_embeddings = facts
                components_by_trace: dict[str, list[dict[str, object]]] = {}
                for component in components:
                    components_by_trace.setdefault(component.retrieval_trace_id, []).append(
                        component.model_dump(mode="json")
                    )
                return {
                    "batch": batch.model_dump(mode="json"),
                    "query_embeddings": [item.model_dump(mode="json") for item in query_embeddings],
                    "windows": [
                        {
                            "rank": trace.rank,
                            "window_id": trace.retrieval_window_id,
                            "winning_document_id": trace.retrieval_document_id,
                            "score": (
                                trace.score.model_dump(mode="json")
                                if trace.score is not None
                                else None
                            ),
                            "components": components_by_trace.get(trace.retrieval_trace_id, []),
                        }
                        for trace in traces
                    ],
                }
            finally:
                await engine.dispose()

        report = asyncio.run(inspect())
    except Exception as exc:
        typer.echo(f"ERROR: retrieval inspection failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


def _persistence_run_context(
    settings: Settings,
    *,
    command: str,
    snapshot_path,
    method: str = "canonical_persistence",
    registry: ModelRegistry | None = None,
) -> PipelineRunContext:
    snapshot = load_manifest(snapshot_path)
    manifest = RunManifest(
        run_id=make_run_id(command),
        experiment_name=command,
        dataset=snapshot.dataset,
        dataset_version=snapshot.dataset_version,
        method=method,
        paper_mode=settings.mode is AppMode.PAPER,
        evaluation_policy_version=snapshot.evaluation_policy_version,
        evaluation_exclusion_rules={
            snapshot.dataset.value: list(snapshot.excluded_source_categories)
        },
        dataset_snapshot_ids=[snapshot.snapshot_id],
        metadata={"snapshot_id": snapshot.snapshot_id},
    )
    if registry is not None:
        manifest = bind_registry_to_manifest(manifest, registry)
    return PipelineRunContext(
        runs_root=settings.runs_path,
        manifest=manifest,
        datasets=[snapshot.dataset],
    )


def _alembic_config(settings: Settings):  # type: ignore[no-untyped-def]
    try:
        from alembic.config import Config

        from pmmc.infrastructure.database.engine import database_url_for_role
    except ModuleNotFoundError as exc:
        _dependency_error("database", exc)
    configuration = Config(str(settings.project_root / "alembic.ini"))
    configuration.attributes["database_url"] = database_url_for_role(settings, DatabaseRole.ADMIN)
    return configuration


@db_app.command("check")
def database_check(
    role: DatabaseRole = typer.Option(DatabaseRole.RUNTIME, "--role"),
) -> None:
    """Check one configured database role without displaying its DSN."""

    try:
        from pmmc.infrastructure.database.engine import create_engine
        from pmmc.infrastructure.database.health import check_database
    except ModuleNotFoundError as exc:
        _dependency_error("database", exc)
    settings = get_settings()

    async def run() -> dict[str, object]:
        engine = create_engine(settings, role)
        try:
            return await check_database(engine, role)
        finally:
            await engine.dispose()

    try:
        result = asyncio.run(run())
    except Exception as exc:
        typer.echo(f"ERROR: database check failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


@db_app.command("current")
def database_current() -> None:
    """Print the current Alembic database revision."""

    try:
        import alembic.command as command
    except ModuleNotFoundError as exc:
        _dependency_error("database", exc)
    try:
        command.current(_alembic_config(get_settings()), verbose=True)
    except Exception as exc:
        typer.echo(f"ERROR: Alembic current failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc


@db_app.command("upgrade")
def database_upgrade(
    revision: str = typer.Option("head", "--revision"),
) -> None:
    """Upgrade the owner database to an explicit Alembic revision."""

    try:
        import alembic.command as command
    except ModuleNotFoundError as exc:
        _dependency_error("database", exc)
    try:
        command.upgrade(_alembic_config(get_settings()), revision)
    except Exception as exc:
        typer.echo(f"ERROR: Alembic upgrade failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Database upgraded to {revision}.")


@db_app.command("import-snapshot")
def database_import_snapshot(
    snapshot: str = typer.Option(..., "--snapshot", help="Canonical snapshot ID."),
) -> None:
    """Atomically import one gold-isolated canonical snapshot into Postgres."""

    try:
        from pmmc.infrastructure.database.engine import create_engine
        from pmmc.infrastructure.database.importer import CanonicalSnapshotImporter
    except ModuleNotFoundError as exc:
        _dependency_error("database", exc)
    settings = get_settings()
    try:
        snapshot_path = find_snapshot(settings.benchmark_artifacts_path, snapshot)
        run = _persistence_run_context(
            settings,
            command="db-import-snapshot",
            snapshot_path=snapshot_path,
        )

        async def execute_import():  # type: ignore[no-untyped-def]
            engine = create_engine(settings, DatabaseRole.ADMIN)
            try:
                importer = CanonicalSnapshotImporter(
                    engine,
                    batch_size=settings.database_import_batch_size,
                )
                with run:
                    with run.stage(
                        "canonical-import",
                        dataset=run.datasets[0],
                        total=None,
                    ) as stage:
                        return await importer.import_snapshot(
                            snapshot_path,
                            run_manifest=run.manifest,
                            progress=_progress_callback(stage),
                        )
            finally:
                await engine.dispose()

        result = asyncio.run(execute_import())
    except Exception as exc:
        typer.echo(f"ERROR: snapshot import failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Snapshot {result.snapshot_id}: state={result.state.value} "
        f"idempotent={str(result.idempotent).lower()}"
    )
    typer.echo(f"Run artifacts: {run.run_dir}")


@db_app.command("verify-snapshot")
def database_verify_snapshot(
    snapshot: str = typer.Option(..., "--snapshot", help="Canonical snapshot ID."),
) -> None:
    """Compare a canonical snapshot manifest with persisted table counts."""

    try:
        from pmmc.benchmarks.canonical import preflight_snapshot
        from pmmc.infrastructure.database.engine import create_engine
        from pmmc.infrastructure.database.importer import CanonicalSnapshotImporter
    except ModuleNotFoundError as exc:
        _dependency_error("database", exc)
    settings = get_settings()
    try:
        snapshot_path = find_snapshot(settings.benchmark_artifacts_path, snapshot)
        report = preflight_snapshot(snapshot_path)

        async def verify() -> dict[str, int]:
            engine = create_engine(settings, DatabaseRole.ADMIN)
            try:
                importer = CanonicalSnapshotImporter(engine)
                async with engine.connect() as connection:
                    return await importer.verify_snapshot(connection, snapshot, report.counts)
            finally:
                await engine.dispose()

        counts = asyncio.run(verify())
    except Exception as exc:
        typer.echo(f"ERROR: snapshot verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(counts, ensure_ascii=True, indent=2, sort_keys=True))


@db_app.command("import-corpus")
def database_import_corpus(
    corpus_dir: Path = typer.Option(..., "--corpus-dir"),
) -> None:
    """Atomically import one validated baseline corpus using the owner role."""

    try:
        from pmmc.infrastructure.database.baseline_repositories import (
            BaselineCorpusImporter,
        )
        from pmmc.infrastructure.database.engine import create_engine
    except ModuleNotFoundError as exc:
        _dependency_error("baseline persistence", exc)
    settings = get_settings()
    try:
        from pmmc.baselines import validate_corpus

        local_manifest = validate_corpus(corpus_dir)
        snapshot_path = find_snapshot(
            settings.benchmark_artifacts_path,
            local_manifest.snapshot_id,
        )
        run = _persistence_run_context(
            settings,
            command="db-import-corpus",
            snapshot_path=snapshot_path,
        )

        async def import_value():  # type: ignore[no-untyped-def]
            engine = create_engine(settings, DatabaseRole.ADMIN)
            try:
                importer = BaselineCorpusImporter(
                    engine,
                    batch_size=settings.database_import_batch_size,
                )
                with run:
                    with run.stage("import-baseline-corpus", dataset=run.datasets[0]) as stage:
                        return await importer.import_corpus(
                            corpus_dir,
                            progress=_progress_callback(stage),
                        )
            finally:
                await engine.dispose()

        manifest, idempotent = asyncio.run(import_value())
    except Exception as exc:
        typer.echo(f"ERROR: corpus import failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Corpus imported: corpus_id={manifest.corpus_id} idempotent={str(idempotent).lower()}"
    )


@db_app.command("verify-corpus")
def database_verify_corpus(
    corpus_id: str = typer.Option(..., "--corpus-id"),
) -> None:
    """Verify persisted corpus state and manifest counts."""

    try:
        from pmmc.infrastructure.database.baseline_repositories import (
            verify_persisted_corpus,
        )
        from pmmc.infrastructure.database.engine import create_engine
    except ModuleNotFoundError as exc:
        _dependency_error("baseline persistence", exc)
    settings = get_settings()

    async def verify() -> dict[str, object]:
        engine = create_engine(settings, DatabaseRole.RUNTIME)
        try:
            return await verify_persisted_corpus(engine, corpus_id)
        finally:
            await engine.dispose()

    try:
        result = asyncio.run(verify())
    except Exception as exc:
        typer.echo(f"ERROR: corpus database verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    if not result["valid"]:
        raise typer.Exit(code=1)


@db_app.command("register-run")
def database_register_run(
    run_dir: Path = typer.Option(..., "--run-dir", help="Completed local run directory."),
) -> None:
    """Register an immutable manifest and its actual selected-query denominator."""

    try:
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.experiment_runs import register_local_run
    except ModuleNotFoundError as exc:
        _dependency_error("experiment persistence", exc)
    settings = get_settings()

    async def register() -> tuple[str, int]:
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session, session.begin():
                status, selected_count = await register_local_run(session, run_dir)
                return status.value, selected_count
        finally:
            await engine.dispose()

    try:
        status, selected_count = asyncio.run(register())
    except Exception as exc:
        typer.echo(f"ERROR: run registration failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Run registered: directory={run_dir} status={status} selected={selected_count}")


@db_app.command("inspect-run")
def database_inspect_run(
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Print non-sensitive state and fact counts for one experiment run."""

    try:
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.experiment_runs import inspect_experiment_run
    except ModuleNotFoundError as exc:
        _dependency_error("experiment persistence", exc)
    settings = get_settings()

    async def inspect() -> dict[str, object]:
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                return await inspect_experiment_run(session, run_id)
        finally:
            await engine.dispose()

    try:
        result = asyncio.run(inspect())
    except Exception as exc:
        typer.echo(f"ERROR: run inspection failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


@db_app.command("verify-run")
def database_verify_run(
    run_id: str = typer.Option(..., "--run-id"),
    include_evaluation: bool = typer.Option(False, "--include-evaluation"),
) -> None:
    """Verify persisted lifecycle and experiment-fact invariants."""

    try:
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.experiment_runs import verify_experiment_run
    except ModuleNotFoundError as exc:
        _dependency_error("experiment persistence", exc)
    settings = get_settings()
    role = DatabaseRole.EVALUATOR if include_evaluation else DatabaseRole.RUNTIME

    async def verify() -> dict[str, object]:
        engine, sessions = create_session_factory(settings, role)
        try:
            async with sessions() as session:
                return await verify_experiment_run(
                    session,
                    run_id,
                    include_evaluation=include_evaluation,
                )
        finally:
            await engine.dispose()

    try:
        result = asyncio.run(verify())
    except Exception as exc:
        typer.echo(f"ERROR: run verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    if not result["valid"]:
        raise typer.Exit(code=1)


@db_app.command("export-run")
def database_export_run(
    run_id: str = typer.Option(..., "--run-id"),
    output_root: Path | None = typer.Option(None, "--output-root"),
    include_evaluation: bool = typer.Option(False, "--include-evaluation"),
    include_referenced_model_artifacts: bool = typer.Option(
        False,
        "--include-referenced-model-artifacts",
    ),
    result_json: Path | None = typer.Option(None, "--result-json"),
) -> None:
    """Create a deterministic frozen JSONL/Parquet export of a completed run."""

    try:
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.experiment_export import ExperimentExporter
        from pmmc.infrastructure.object_store import create_runtime_object_store
    except ModuleNotFoundError as exc:
        _dependency_error("experiment export", exc)
    settings = get_settings()
    role = DatabaseRole.EVALUATOR if include_evaluation else DatabaseRole.RUNTIME
    destination = output_root or settings.experiment_outputs_path

    async def export():  # type: ignore[no-untyped-def]
        engine, sessions = create_session_factory(settings, role)
        try:
            exporter = ExperimentExporter(
                sessions,
                evaluator_sessions=sessions if include_evaluation else None,
                runtime_store=(
                    create_runtime_object_store(settings)
                    if include_referenced_model_artifacts
                    else None
                ),
                batch_size=settings.experiment_export_batch_size,
            )
            return await exporter.export(
                run_id,
                destination,
                include_evaluation=include_evaluation,
                include_referenced_model_artifacts=(include_referenced_model_artifacts),
            )
        finally:
            await engine.dispose()

    try:
        result = asyncio.run(export())
    except Exception as exc:
        typer.echo(f"ERROR: run export failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    payload = {
        "run_id": result.run_id,
        "export_id": result.export_id,
        "export_dir": str(result.export_dir),
        "manifest_path": str(result.manifest_path),
    }
    if result_json is not None:
        destination_path = settings.resolve_path(result_json)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    typer.echo(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


@db_app.command("inspect-compiler-run")
def database_inspect_compiler_run(
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Print runtime-safe compiler denominator and fact counts."""

    try:
        from pmmc.infrastructure.database.compiler_repositories import (
            inspect_compiler_run,
        )
        from pmmc.infrastructure.database.engine import create_session_factory
    except ModuleNotFoundError as exc:
        _dependency_error("compiler persistence", exc)
    settings = get_settings()

    async def inspect() -> dict[str, object]:
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                return await inspect_compiler_run(session, run_id)
        finally:
            await engine.dispose()

    try:
        result = asyncio.run(inspect())
    except Exception as exc:
        typer.echo(f"ERROR: compiler run inspection failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


@db_app.command("verify-compiler-run")
def database_verify_compiler_run(
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Verify compiler unit, job, stage, and execution completeness."""

    try:
        from pmmc.infrastructure.database.compiler_repositories import (
            verify_compiler_run,
        )
        from pmmc.infrastructure.database.engine import create_session_factory
    except ModuleNotFoundError as exc:
        _dependency_error("compiler persistence", exc)
    settings = get_settings()

    async def verify() -> dict[str, object]:
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                return await verify_compiler_run(session, run_id)
        finally:
            await engine.dispose()

    try:
        result = asyncio.run(verify())
    except Exception as exc:
        typer.echo(f"ERROR: compiler run verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    if not result["valid"]:
        raise typer.Exit(code=1)


@storage_app.command("check")
def storage_check() -> None:
    """Check owner access to the private runtime and evaluation buckets."""

    try:
        from pmmc.infrastructure.object_store import create_admin_object_stores
    except ModuleNotFoundError as exc:
        _dependency_error("object storage", exc)
    try:
        runtime_store, evaluation_store = create_admin_object_stores(get_settings())

        async def check() -> None:
            await runtime_store.check()
            await evaluation_store.check()

        asyncio.run(check())
    except Exception as exc:
        typer.echo(f"ERROR: object storage check failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Object storage runtime/evaluation buckets: OK")


@storage_app.command("archive-snapshot")
def storage_archive_snapshot(
    snapshot: str = typer.Option(..., "--snapshot", help="Canonical snapshot ID."),
) -> None:
    """Archive runtime and gold canonical files into separate private buckets."""

    try:
        from pmmc.infrastructure.database.engine import create_engine
        from pmmc.infrastructure.database.storage_updates import persist_snapshot_archive
        from pmmc.infrastructure.object_store import create_admin_object_stores
        from pmmc.infrastructure.storage_sync import (
            EVALUATION_SNAPSHOT_FILES,
            RUNTIME_SNAPSHOT_FILES,
            archive_snapshot,
        )
    except ModuleNotFoundError as exc:
        _dependency_error("persistence", exc)
    settings = get_settings()
    try:
        snapshot_path = find_snapshot(settings.benchmark_artifacts_path, snapshot)
        runtime_store, evaluation_store = create_admin_object_stores(settings)
        run = _persistence_run_context(
            settings,
            command="storage-archive-snapshot",
            snapshot_path=snapshot_path,
        )

        async def archive():  # type: ignore[no-untyped-def]
            engine = create_engine(settings, DatabaseRole.ADMIN)
            try:
                with run:
                    with run.stage(
                        "archive-snapshot",
                        dataset=run.datasets[0],
                        total=len(RUNTIME_SNAPSHOT_FILES) + len(EVALUATION_SNAPSHOT_FILES),
                    ) as stage:
                        result = await archive_snapshot(
                            snapshot_path,
                            runtime_store=runtime_store,
                            evaluation_store=evaluation_store,
                            progress=_progress_callback(stage),
                        )
                    await persist_snapshot_archive(engine, result)
                return result
            finally:
                await engine.dispose()

        result = asyncio.run(archive())
    except Exception as exc:
        typer.echo(f"ERROR: snapshot archive failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Archived {len(result.runtime_objects)} runtime files and "
        f"{len(result.evaluation_objects)} evaluation files."
    )


@storage_app.command("sync-assets")
def storage_sync_assets(
    snapshot: str = typer.Option(..., "--snapshot", help="Canonical snapshot ID."),
) -> None:
    """Upload local benchmark assets by SHA256 and persist their private URIs."""

    try:
        from pmmc.infrastructure.database.engine import create_engine
        from pmmc.infrastructure.database.storage_updates import persist_asset_sync
        from pmmc.infrastructure.object_store import create_admin_object_stores
        from pmmc.infrastructure.storage_sync import sync_assets
    except ModuleNotFoundError as exc:
        _dependency_error("persistence", exc)
    settings = get_settings()
    try:
        snapshot_path = find_snapshot(settings.benchmark_artifacts_path, snapshot)
        snapshot_manifest = load_manifest(snapshot_path)
        runtime_store, _ = create_admin_object_stores(settings)
        run = _persistence_run_context(
            settings,
            command="storage-sync-assets",
            snapshot_path=snapshot_path,
        )

        async def synchronize():  # type: ignore[no-untyped-def]
            engine = create_engine(settings, DatabaseRole.ADMIN)
            try:
                with run:
                    with run.stage(
                        "sync-assets",
                        dataset=run.datasets[0],
                        total=snapshot_manifest.counts.get("assets", 0),
                    ) as stage:
                        result = await sync_assets(
                            snapshot_path,
                            project_root=settings.benchmark_asset_path,
                            runtime_store=runtime_store,
                            progress=_progress_callback(stage),
                        )
                    await persist_asset_sync(engine, result)
                return result
            finally:
                await engine.dispose()

        result = asyncio.run(synchronize())
    except Exception as exc:
        typer.echo(f"ERROR: asset synchronization failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Assets synced={len(result.synced)} remote_only_skipped={result.skipped_remote_only}"
    )


@storage_app.command("verify")
def storage_verify(
    snapshot: str = typer.Option(..., "--snapshot", help="Canonical snapshot ID."),
) -> None:
    """Verify archived canonical files against local SHA256 values."""

    try:
        from pmmc.infrastructure.object_store import create_admin_object_stores
        from pmmc.infrastructure.storage_sync import verify_archive
    except ModuleNotFoundError as exc:
        _dependency_error("object storage", exc)
    settings = get_settings()
    try:
        snapshot_path = find_snapshot(settings.benchmark_artifacts_path, snapshot)
        runtime_store, evaluation_store = create_admin_object_stores(settings)
        result = asyncio.run(
            verify_archive(
                snapshot_path,
                runtime_store=runtime_store,
                evaluation_store=evaluation_store,
            )
        )
    except Exception as exc:
        typer.echo(f"ERROR: archive verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Archive verified: runtime={len(result.runtime_objects)} "
        f"evaluation={len(result.evaluation_objects)}"
    )


@storage_app.command("archive-corpus")
def storage_archive_corpus(
    corpus_dir: Path = typer.Option(..., "--corpus-dir"),
) -> None:
    """Explicitly archive a derived corpus into the private runtime bucket."""

    try:
        from pmmc.infrastructure.baseline_storage import archive_corpus
        from pmmc.infrastructure.object_store import create_runtime_object_store
    except ModuleNotFoundError as exc:
        _dependency_error("baseline object storage", exc)
    try:
        result = asyncio.run(
            archive_corpus(corpus_dir, create_runtime_object_store(get_settings()))
        )
    except Exception as exc:
        typer.echo(f"ERROR: corpus archive failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Corpus archived: corpus_id={result.corpus_id} objects={len(result.objects)}")


@storage_app.command("verify-corpus")
def storage_verify_corpus(
    corpus_dir: Path = typer.Option(..., "--corpus-dir"),
) -> None:
    """Verify local corpus files against runtime-bucket checksums."""

    try:
        from pmmc.infrastructure.baseline_storage import verify_corpus_archive
        from pmmc.infrastructure.object_store import create_runtime_object_store
    except ModuleNotFoundError as exc:
        _dependency_error("baseline object storage", exc)
    try:
        result = asyncio.run(
            verify_corpus_archive(corpus_dir, create_runtime_object_store(get_settings()))
        )
    except Exception as exc:
        typer.echo(f"ERROR: corpus archive verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Corpus archive verified: corpus_id={result.corpus_id} objects={len(result.objects)}"
    )


@storage_app.command("archive-run")
def storage_archive_run(
    export_dir: Path = typer.Option(..., "--export-dir"),
) -> None:
    """Archive a frozen run export with runtime/evaluation bucket isolation."""

    try:
        from pmmc.infrastructure.experiment_export import (
            archive_experiment_export_scoped,
        )
        from pmmc.infrastructure.object_store import create_admin_object_stores
    except ModuleNotFoundError as exc:
        _dependency_error("experiment object storage", exc)
    try:
        runtime_store, evaluation_store = create_admin_object_stores(get_settings())
        result = asyncio.run(
            archive_experiment_export_scoped(
                export_dir,
                runtime_store,
                evaluation_store if (export_dir / "evaluation").is_dir() else None,
            )
        )
    except Exception as exc:
        typer.echo(f"ERROR: run archive failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Run export archived: runtime={len(result['runtime'])} "
        f"evaluation={len(result['evaluation'])}"
    )


@storage_app.command("archive-runtime")
def storage_archive_runtime(
    runtime_id: str = typer.Option(..., "--runtime-id"),
) -> None:
    """Archive redacted runtime plans, telemetry, states, and service logs."""

    from pmmc.infrastructure.object_store import create_runtime_object_store
    from pmmc.infrastructure.runtime_storage import archive_runtime

    settings = get_settings()
    runtime_dir = settings.runs_path / "managed-runtime" / runtime_id
    try:
        result = asyncio.run(archive_runtime(runtime_dir, create_runtime_object_store(settings)))
    except Exception as exc:
        typer.echo(f"ERROR: runtime archive failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


@storage_app.command("verify-runtime")
def storage_verify_runtime(
    runtime_id: str = typer.Option(..., "--runtime-id"),
) -> None:
    """Verify an archived managed-runtime manifest against the runtime bucket."""

    from pmmc.infrastructure.object_store import create_runtime_object_store
    from pmmc.infrastructure.runtime_storage import verify_runtime_archive

    settings = get_settings()
    runtime_dir = settings.runs_path / "managed-runtime" / runtime_id
    try:
        result = asyncio.run(
            verify_runtime_archive(runtime_dir, create_runtime_object_store(settings))
        )
    except Exception as exc:
        typer.echo(f"ERROR: runtime verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    if not result["valid"]:
        raise typer.Exit(code=1)


@storage_app.command("verify-run")
def storage_verify_run(
    export_dir: Path | None = typer.Option(None, "--export-dir"),
    run_id: str | None = typer.Option(None, "--run-id"),
    include_evaluation: bool = typer.Option(False, "--include-evaluation"),
) -> None:
    """Verify an archived export or registered artifacts for one run."""

    try:
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.experiment_artifacts import verify_run_artifacts
        from pmmc.infrastructure.experiment_export import (
            verify_archived_experiment_export,
        )
        from pmmc.infrastructure.object_store import create_admin_object_stores
    except ModuleNotFoundError as exc:
        _dependency_error("experiment object storage", exc)
    if (export_dir is None) == (run_id is None):
        raise typer.BadParameter("provide exactly one of --export-dir or --run-id")
    settings = get_settings()
    try:
        runtime_store, evaluation_store = create_admin_object_stores(settings)
        if export_dir is not None:
            counts = asyncio.run(
                verify_archived_experiment_export(
                    export_dir,
                    runtime_store,
                    evaluation_store if (export_dir / "evaluation").is_dir() else None,
                )
            )
            typer.echo(
                f"Run archive verified: runtime={counts['runtime']} "
                f"evaluation={counts['evaluation']}"
            )
            return

        assert run_id is not None
        role = DatabaseRole.EVALUATOR if include_evaluation else DatabaseRole.RUNTIME

        async def verify_artifacts() -> list[dict[str, object]]:
            engine, sessions = create_session_factory(settings, role)
            try:
                async with sessions() as session:
                    reports = [
                        await verify_run_artifacts(
                            session,
                            runtime_store,
                            run_id,
                            DataScope.RUNTIME,
                        )
                    ]
                    if include_evaluation:
                        reports.append(
                            await verify_run_artifacts(
                                session,
                                evaluation_store,
                                run_id,
                                DataScope.EVALUATION,
                            )
                        )
                    return reports
            finally:
                await engine.dispose()

        reports = asyncio.run(verify_artifacts())
    except Exception as exc:
        typer.echo(f"ERROR: run storage verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(reports, ensure_ascii=True, indent=2, sort_keys=True))
    if not all(report["valid"] for report in reports):
        raise typer.Exit(code=1)


@baseline_app.command("create-suite")
def baseline_create_suite(
    dataset: DatasetName = typer.Option(..., "--dataset"),
    snapshot: str = typer.Option(..., "--snapshot"),
    corpus_id: str | None = typer.Option(None, "--corpus-id"),
    method: list[str] | None = typer.Option(None, "--method"),
    answer_model: str | None = typer.Option(None, "--answer-model"),
    question_type: list[QuestionType] | None = typer.Option(None, "--question-type"),
    source_category: list[str] | None = typer.Option(None, "--source-category"),
    scenario: list[str] | None = typer.Option(None, "--scenario"),
    query_id: list[str] | None = typer.Option(None, "--query-id"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    sample_size: int | None = typer.Option(None, "--sample-size", min=1),
    seed: int = typer.Option(0, "--seed", min=0),
    shard_count: int = typer.Option(1, "--shard-count", min=1),
    shard_index: int = typer.Option(0, "--shard-index", min=0),
    include_excluded: bool = typer.Option(False, "--include-excluded"),
) -> None:
    """Freeze one paired query denominator for foundational baselines."""

    from pmmc.baselines import create_baseline_suite

    if dataset is DatasetName.UNKNOWN:
        raise typer.BadParameter("unknown is not a runnable benchmark", param_hint="--dataset")
    try:
        methods = [BaselineMethod(item) for item in method] if method else None
        selection = BenchmarkSelection(
            datasets=[dataset],
            question_types=question_type or [],
            source_categories=source_category or [],
            scenario_ids=scenario or [],
            query_ids=query_id or [],
            limit=limit,
            sample_size=sample_size,
            seed=seed,
            shard_count=shard_count,
            shard_index=shard_index,
            include_excluded=include_excluded,
        )
        manifest, directory = asyncio.run(
            create_baseline_suite(
                settings=get_settings(),
                snapshot_id=snapshot,
                selection=selection,
                corpus_id=corpus_id,
                methods=methods,
                answer_model_alias=answer_model,
            )
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except Exception as exc:
        typer.echo(f"ERROR: baseline suite creation failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {"suite_id": manifest.suite_id, "suite_artifacts": str(directory)},
            indent=2,
            sort_keys=True,
        )
    )


@baseline_app.command("run-suite")
def baseline_run_suite(
    suite_id: str = typer.Option(..., "--suite-id"),
    phase: str = typer.Option("all", "--phase", help="all, retrieve, or answer"),
    method: list[str] | None = typer.Option(None, "--method"),
) -> None:
    """Run or resume suite children against the frozen denominator."""

    from pmmc.baselines import RagRunPhase, run_baseline_suite
    from pmmc.baselines.rag_runner import RagAnswerBatchFailure
    from pmmc.baselines.suite_contract import BaselineSuiteContractError

    try:
        resolved_phase = RagRunPhase(phase)
        methods = [BaselineMethod(item) for item in method] if method else None
        directory = asyncio.run(
            run_baseline_suite(
                settings=get_settings(),
                suite_id=suite_id,
                phase=resolved_phase,
                methods=methods,
            )
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        diagnostics: dict[str, object] = {
            "method_count": len(method or []),
            "operation": "run_suite",
            "phase": phase,
        }
        if isinstance(exc, RagAnswerBatchFailure):
            diagnostics["answer_job_accounting"] = exc.diagnostics
        if isinstance(exc, BaselineSuiteContractError):
            diagnostics.update(exc.diagnostics)
        _persist_managed_child_diagnostic(
            event="BASELINE_SUITE_FAILURE",
            exc=exc,
            diagnostics=diagnostics,
        )
        typer.echo(
            "BASELINE_SUITE_FAILURE "
            + json.dumps(
                {
                    **diagnostics,
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            err=True,
        )
        typer.echo(f"ERROR: baseline suite failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Suite artifacts: {directory}")


@baseline_app.command("verify-suite")
def baseline_verify_suite(
    suite_id: str = typer.Option(..., "--suite-id"),
) -> None:
    """Verify denominator alignment and all linked baseline runs."""

    from pmmc.baselines import verify_baseline_suite

    try:
        report = asyncio.run(verify_baseline_suite(get_settings(), suite_id))
    except Exception as exc:
        typer.echo(f"ERROR: baseline suite verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(code=1)


@baseline_app.command("register-family")
def baseline_register_family(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    """Register the immutable 2x5 paper-component matrix."""

    from pmmc.baselines import register_baseline_family
    from pmmc.pilots.io import read_json
    from pmmc.schemas import BaselineSuiteFamilyManifest

    settings = get_settings()
    try:
        value = BaselineSuiteFamilyManifest.model_validate(
            read_json(settings.resolve_path(manifest))
        )
        components = asyncio.run(register_baseline_family(settings, value))
    except Exception as exc:
        typer.echo(
            f"ERROR: baseline family registration failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "family_id": value.family_id,
                "component_ids": [item.component_id for item in components],
            },
            indent=2,
            sort_keys=True,
        )
    )


@baseline_app.command("run-component")
def baseline_run_component(
    component_id: str = typer.Option(..., "--component-id"),
    phase: str = typer.Option("all", "--phase", help="all, retrieve, or answer"),
    embedding_replicas: int = typer.Option(1, "--embedding-replicas", min=1),
    answer_replicas: int = typer.Option(1, "--answer-replicas", min=1),
    export_bundle: bool = typer.Option(True, "--export-bundle/--no-export-bundle"),
    output_root: Path | None = typer.Option(None, "--output-root"),
    managed_child: bool = typer.Option(False, "--managed-child", hidden=True),
) -> None:
    """Run one dataset-method component with managed model lifecycle."""

    from pmmc.baselines import (
        RagRunPhase,
        export_baseline_component,
        import_baseline_component,
        run_baseline_component,
        run_managed_baseline_component,
    )

    try:
        resolved_phase = RagRunPhase(phase)
        if managed_child:
            state = asyncio.run(
                run_baseline_component(
                    get_settings(),
                    component_id=component_id,
                    phase=resolved_phase,
                )
            )
        else:
            state = run_managed_baseline_component(
                get_settings(),
                component_id=component_id,
                phase=resolved_phase,
                embedding_replicas=embedding_replicas,
                answer_replicas=answer_replicas,
            )
            if export_bundle and resolved_phase is not RagRunPhase.RETRIEVE:
                settings = get_settings()
                root = (
                    settings.resolve_path(output_root)
                    if output_root is not None
                    else settings.runs_path / "baseline-components"
                )
                bundle_dir, _bundle = asyncio.run(
                    export_baseline_component(
                        settings,
                        component_id=component_id,
                        output_root=root,
                    )
                )
                state = asyncio.run(
                    import_baseline_component(
                        settings,
                        bundle_dir=bundle_dir,
                        imported_by="local-component-runner",
                    )
                )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except Exception as exc:
        typer.echo(f"ERROR: baseline component failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True))


@baseline_app.command("verify-component")
def baseline_verify_component(
    component_id: str = typer.Option(..., "--component-id"),
    bundle: Path | None = typer.Option(None, "--bundle"),
) -> None:
    """Verify one component's database and optional portable bundle facts."""

    from pmmc.baselines import verify_baseline_component

    settings = get_settings()
    try:
        report = asyncio.run(
            verify_baseline_component(
                settings,
                component_id,
                bundle_dir=settings.resolve_path(bundle) if bundle else None,
            )
        )
    except Exception as exc:
        typer.echo(
            f"ERROR: baseline component verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(code=1)


@baseline_app.command("export-component")
def baseline_export_component(
    component_id: str = typer.Option(..., "--component-id"),
    output_root: Path | None = typer.Option(None, "--output-root"),
) -> None:
    """Write a signed runtime-only component bundle immediately after completion."""

    from pmmc.baselines import export_baseline_component

    settings = get_settings()
    root = (
        settings.resolve_path(output_root)
        if output_root is not None
        else settings.runs_path / "baseline-components"
    )
    try:
        directory, manifest = asyncio.run(
            export_baseline_component(
                settings,
                component_id=component_id,
                output_root=root,
            )
        )
    except Exception as exc:
        typer.echo(
            f"ERROR: baseline component export failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "component_id": component_id,
                "bundle_export_id": manifest["bundle_export_id"],
                "bundle_dir": str(directory),
            },
            indent=2,
            sort_keys=True,
        )
    )


@baseline_app.command("import-component")
def baseline_import_component(
    bundle: Path = typer.Option(..., "--bundle"),
    imported_by: str = typer.Option(..., "--imported-by"),
) -> None:
    """Attach a verified bundle to runtime facts restored from a checkpoint."""

    from pmmc.baselines import import_baseline_component

    settings = get_settings()
    try:
        state = asyncio.run(
            import_baseline_component(
                settings,
                bundle_dir=settings.resolve_path(bundle),
                imported_by=imported_by,
            )
        )
    except Exception as exc:
        typer.echo(
            f"ERROR: baseline component import failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True))


@baseline_app.command("aggregate-family")
def baseline_aggregate_family(
    family_id: str = typer.Option(..., "--family-id"),
    evaluate: bool = typer.Option(
        False,
        "--evaluate/--no-evaluate",
        help="Run deterministic metrics, one managed judge pool, analysis, and exports.",
    ),
    judge_replicas: int = typer.Option(4, "--judge-replicas", min=1),
) -> None:
    """Assemble ten verified components and optionally evaluate both suites."""

    from pmmc.baselines import aggregate_baseline_family, evaluate_baseline_family

    try:
        report = (
            evaluate_baseline_family(
                get_settings(),
                family_id,
                judge_replicas=judge_replicas,
            )
            if evaluate
            else asyncio.run(aggregate_baseline_family(get_settings(), family_id))
        )
    except Exception as exc:
        typer.echo(
            f"ERROR: baseline family aggregation failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@baseline_app.command("verify-family")
def baseline_verify_family(
    family_id: str = typer.Option(..., "--family-id"),
) -> None:
    """Verify a baseline component matrix and its terminal family state."""

    from pmmc.baselines import verify_baseline_family

    try:
        report = asyncio.run(verify_baseline_family(get_settings(), family_id))
    except Exception as exc:
        typer.echo(
            f"ERROR: baseline family verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(code=1)


@evaluate_app.command("run")
def evaluate_run(
    suite_id: str | None = typer.Option(None, "--suite-id"),
    query_executor_run_id: str | None = typer.Option(
        None, "--query-executor-run-id"
    ),
    resume_evaluation_run_id: str | None = typer.Option(None, "--resume-evaluation-run-id"),
    phase: str = typer.Option("all", "--phase", help="deterministic, judge, analyze, or all"),
) -> None:
    """Create or resume one independent gold-isolated evaluation run."""

    from pmmc.evaluation.workflow import EvaluationRunPhase, run_evaluation_workflow

    try:
        resolved_phase = EvaluationRunPhase(phase)
        manifest, directory = asyncio.run(
            run_evaluation_workflow(
                settings=get_settings(),
                suite_id=suite_id,
                query_executor_run_id=query_executor_run_id,
                resume_evaluation_run_id=resume_evaluation_run_id,
                phase=resolved_phase,
            )
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(f"ERROR: evaluation run failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "evaluation_run_id": manifest.evaluation_run_id,
                "evaluation_artifacts": str(directory),
            },
            indent=2,
            sort_keys=True,
        )
    )


@evaluate_app.command("inspect")
def evaluate_inspect(
    evaluation_run_id: str = typer.Option(..., "--evaluation-run-id"),
    query_id: str = typer.Option(..., "--query-id"),
) -> None:
    """Inspect private candidate/reference/metric facts for one query."""

    from pmmc.evaluation.workflow import inspect_evaluation_query

    try:
        report = asyncio.run(inspect_evaluation_query(get_settings(), evaluation_run_id, query_id))
    except Exception as exc:
        typer.echo(f"ERROR: evaluation inspection failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


@evaluate_app.command("verify")
def evaluate_verify(
    evaluation_run_id: str = typer.Option(..., "--evaluation-run-id"),
    require_managed_runtime: bool = typer.Option(
        False,
        "--require-managed-runtime",
        help="Require every successful judge call to carry runtime and service lineage.",
    ),
) -> None:
    """Verify evaluation denominator, jobs, and terminal coverage."""

    from pmmc.evaluation.workflow import verify_evaluation_run

    try:
        report = asyncio.run(
            verify_evaluation_run(
                get_settings(),
                evaluation_run_id,
                require_managed_runtime=require_managed_runtime,
            )
        )
    except Exception as exc:
        typer.echo(f"ERROR: evaluation verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(code=1)


@evaluate_app.command("export")
def evaluate_export(
    evaluation_run_id: str = typer.Option(..., "--evaluation-run-id"),
) -> None:
    """Export public summaries and evaluator-private query-level facts."""

    from pmmc.evaluation.export import EvaluationExporter

    try:
        result = asyncio.run(EvaluationExporter(get_settings()).export(evaluation_run_id))
    except Exception as exc:
        typer.echo(f"ERROR: evaluation export failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "evaluation_run_id": result.evaluation_run_id,
                "export_id": result.export_id,
                "export_dir": str(result.export_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


@evaluate_app.command("qbank-coverage")
def evaluate_qbank_coverage(
    compiler_run_id: str = typer.Option(..., "--compiler-run-id"),
    dataset: DatasetName = typer.Option(..., "--dataset"),
    question_type: list[QuestionType] | None = typer.Option(None, "--question-type"),
    source_category: list[str] | None = typer.Option(None, "--source-category"),
    scenario: list[str] | None = typer.Option(None, "--scenario"),
    query_id: list[str] | None = typer.Option(None, "--query-id"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    sample_size: int | None = typer.Option(None, "--sample-size", min=1),
    seed: int = typer.Option(0, "--seed", min=0),
    shard_count: int = typer.Option(1, "--shard-count", min=1),
    shard_index: int = typer.Option(0, "--shard-index", min=0),
    include_excluded: bool = typer.Option(False, "--include-excluded"),
    selection_metadata_json: str | None = typer.Option(
        None,
        "--selection-metadata-json",
        hidden=True,
    ),
    config: Path | None = typer.Option(None, "--config"),
    resume_coverage_run_id: str | None = typer.Option(None, "--resume-coverage-run-id"),
    phase: str = typer.Option("all", "--phase"),
) -> None:
    """Evaluate Questioner QBank evidence coverage with evaluator-only gold access."""

    from pmmc.compiler.questioner_config import load_questioner_config
    from pmmc.evaluation.qbank_workflow import (
        QBankCoveragePhase,
        run_qbank_coverage,
    )
    from pmmc.models.registry import load_model_registry

    if dataset is DatasetName.UNKNOWN:
        raise typer.BadParameter("unknown is not a runnable benchmark", param_hint="--dataset")
    settings = get_settings()
    try:
        selection_metadata: JsonDict = {}
        if selection_metadata_json is not None:
            parsed_metadata = json.loads(selection_metadata_json)
            if not isinstance(parsed_metadata, dict) or any(
                not isinstance(key, str) for key in parsed_metadata
            ):
                raise ValueError("selection metadata must be a JSON object with string keys")
            selection_metadata = parsed_metadata
        selection = BenchmarkSelection(
            datasets=[dataset],
            question_types=question_type or [],
            source_categories=source_category or [],
            scenario_ids=scenario or [],
            query_ids=query_id or [],
            limit=limit,
            sample_size=sample_size,
            seed=seed,
            shard_count=shard_count,
            shard_index=shard_index,
            include_excluded=include_excluded,
            metadata=selection_metadata,
        )
        config_path = (
            settings.resolve_path(config)
            if config is not None
            else settings.resolved_questioner_config_path
        )
        questioner = load_questioner_config(config_path)
        registry = load_model_registry(settings.resolved_model_registry_path)
        resolved_phase = QBankCoveragePhase(phase)

        last_coverage_progress = 0.0

        def progress(_query_locator: str, current: int, total: int) -> None:
            nonlocal last_coverage_progress
            now = time.monotonic()
            if current == total or now - last_coverage_progress >= 30:
                typer.echo(f"PROGRESS stage=qbank_coverage queries={current}/{total}")
                last_coverage_progress = now

        result = asyncio.run(
            run_qbank_coverage(
                settings=settings,
                registry=registry,
                config=questioner,
                compiler_run_id=compiler_run_id,
                selection=selection,
                coverage_run_id=resume_coverage_run_id,
                phase=resolved_phase,
                on_query=progress,
            )
        )
    except KeyboardInterrupt as exc:
        typer.echo("Interrupted; QBank coverage can be resumed by coverage run ID.", err=True)
        raise typer.Exit(code=130) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except Exception as exc:
        typer.echo(f"ERROR: QBank coverage failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "coverage_run_id": result.coverage_run_id,
                "compiler_run_id": result.compiler_run_id,
                "query_count": result.query_count,
                "applicable_query_count": result.applicable_query_count,
                "phase": resolved_phase.value,
                "gate_report": (
                    result.report.model_dump(mode="json") if result.report is not None else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@evaluate_app.command("inspect-qbank-coverage")
def evaluate_inspect_qbank_coverage(
    coverage_run_id: str = typer.Option(..., "--coverage-run-id"),
    query_id: str = typer.Option(..., "--query-id"),
) -> None:
    """Inspect evaluator-private QBank candidates, gold groups, and metrics."""

    from pmmc.evaluation.qbank_workflow import inspect_qbank_coverage_query

    try:
        report = asyncio.run(
            inspect_qbank_coverage_query(get_settings(), coverage_run_id, query_id)
        )
    except Exception as exc:
        typer.echo(f"ERROR: QBank coverage inspection failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


def _qbank_semantic_progress_due(*, current: int, total: int) -> bool:
    """Bound pair-level CLI progress while preserving early and terminal signals."""

    if current <= 0 or total <= 0 or current > total:
        return False
    interval = max(1, (total + 19) // 20)
    return current <= 3 or current == total or current % interval == 0


@evaluate_app.command("qbank-semantic-match")
def evaluate_qbank_semantic_match(
    coverage_run_id: str = typer.Option(..., "--coverage-run-id"),
    config: Path = typer.Option(
        Path("configs/compiler/qbank-semantic-match-v1.yaml"),
        "--config",
    ),
    resume_semantic_run_id: str | None = typer.Option(None, "--resume-semantic-run-id"),
    result_json: Path | None = typer.Option(None, "--result-json"),
) -> None:
    """Run evaluator-private semantic matching over the per-budget top-50 union."""

    from pmmc.evaluation.qbank_semantic import load_qbank_semantic_config
    from pmmc.evaluation.qbank_semantic_workflow import (
        qbank_semantic_failure_diagnostics,
        run_qbank_semantic_match,
    )
    from pmmc.infrastructure.database.errors import safe_database_error_summary
    from pmmc.models.registry import load_model_registry

    settings = get_settings()
    try:
        semantic_config = load_qbank_semantic_config(settings.resolve_path(config))
        registry = load_model_registry(settings.resolved_model_registry_path)

        def progress(
            _query_id: str,
            _qbank_id: str,
            current: int,
            total: int,
        ) -> None:
            if _qbank_semantic_progress_due(current=current, total=total):
                typer.echo(
                    "PROGRESS stage=qbank_semantic_match "
                    f"pairs={current}/{total} percent={100 * current / total:.1f}"
                )

        def semantic_event(
            phase: str,
            details: dict[str, str | int | float | bool],
        ) -> None:
            typer.echo(
                "QBANK_SEMANTIC_EVENT "
                + json.dumps(
                    {
                        "stage": "qbank_semantic_match",
                        "phase": phase,
                        **details,
                    },
                    sort_keys=True,
                )
            )

        result = asyncio.run(
            run_qbank_semantic_match(
                settings=settings,
                registry=registry,
                coverage_run_id=coverage_run_id,
                config=semantic_config,
                semantic_run_id=resume_semantic_run_id,
                on_pair=progress,
                on_event=semantic_event,
            )
        )
    except KeyboardInterrupt as exc:
        typer.echo(
            "Interrupted; semantic matching can be resumed by semantic run ID.",
            err=True,
        )
        raise typer.Exit(code=130) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except Exception as exc:
        diagnostics = qbank_semantic_failure_diagnostics(exc)
        typer.echo(
            "QBANK_SEMANTIC_FAILURE " + json.dumps(diagnostics, sort_keys=True),
            err=True,
        )
        typer.echo(
            "ERROR: QBank semantic matching failed: " + safe_database_error_summary(diagnostics),
            err=True,
        )
        raise typer.Exit(code=1) from exc
    payload = {
        "semantic_run_id": result.semantic_run_id,
        "coverage_run_id": result.coverage_run_id,
        "pair_count": result.pair_count,
        "semantic_match_count": result.semantic_match_count,
        "conservative_fallback_count": result.conservative_fallback_count,
        "reused": result.reused,
    }
    if result_json is not None:
        destination = settings.resolve_path(result_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@evaluate_app.command("verify-qbank-semantic-match")
def evaluate_verify_qbank_semantic_match(
    semantic_run_id: str = typer.Option(..., "--semantic-run-id"),
    config: Path = typer.Option(
        Path("configs/compiler/qbank-semantic-match-v1.yaml"),
        "--config",
    ),
) -> None:
    """Verify semantic pair accounting, model lineage, and the metric grid."""

    from pmmc.evaluation.qbank_semantic import load_qbank_semantic_config
    from pmmc.evaluation.qbank_semantic_workflow import (
        verify_qbank_semantic_match_run,
    )

    settings = get_settings()
    try:
        report = asyncio.run(
            verify_qbank_semantic_match_run(
                settings,
                semantic_run_id,
                load_qbank_semantic_config(settings.resolve_path(config)),
            )
        )
    except Exception as exc:
        typer.echo(
            f"ERROR: QBank semantic verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(code=1)


@evaluate_app.command("verify-qbank-coverage")
def evaluate_verify_qbank_coverage(
    coverage_run_id: str = typer.Option(..., "--coverage-run-id"),
) -> None:
    """Verify QBank coverage denominator, rank grids, embeddings, and report counts."""

    from pmmc.evaluation.qbank_workflow import verify_qbank_coverage_run

    try:
        report = asyncio.run(verify_qbank_coverage_run(get_settings(), coverage_run_id))
    except Exception as exc:
        typer.echo(f"ERROR: QBank coverage verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(code=1)


@evaluate_app.command("export-qbank-coverage")
def evaluate_export_qbank_coverage(
    coverage_run_id: str = typer.Option(..., "--coverage-run-id"),
) -> None:
    """Export deterministic public aggregates and evaluator-private QBank facts."""

    from pmmc.evaluation.qbank_export import QBankCoverageExporter

    try:
        result = asyncio.run(QBankCoverageExporter(get_settings()).export(coverage_run_id))
    except Exception as exc:
        typer.echo(f"ERROR: QBank coverage export failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "coverage_run_id": result.coverage_run_id,
                "export_id": result.export_id,
                "export_dir": str(result.export_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("run-baseline")
def run_baseline(
    method: str | None = typer.Option(
        None,
        "--method",
        help="One of the five foundational baseline method IDs.",
    ),
    dataset: DatasetName | None = typer.Option(None, "--dataset"),
    snapshot: str | None = typer.Option(None, "--snapshot"),
    corpus_id: str | None = typer.Option(
        None, "--corpus-id", help="Required by the three Step 5C RAG methods."
    ),
    answer_model: str | None = typer.Option(None, "--answer-model"),
    question_type: list[QuestionType] | None = typer.Option(None, "--question-type"),
    source_category: list[str] | None = typer.Option(None, "--source-category"),
    scenario: list[str] | None = typer.Option(None, "--scenario"),
    query_id: list[str] | None = typer.Option(None, "--query-id"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    sample_size: int | None = typer.Option(None, "--sample-size", min=1),
    seed: int = typer.Option(0, "--seed", min=0),
    shard_count: int = typer.Option(1, "--shard-count", min=1),
    shard_index: int = typer.Option(0, "--shard-index", min=0),
    include_excluded: bool = typer.Option(False, "--include-excluded"),
    assemble_only: bool = typer.Option(
        False,
        "--assemble-only",
        help="Assemble and audit contexts without jobs, model calls, or paper results.",
    ),
    phase: str = typer.Option(
        "all",
        "--phase",
        help="RAG execution phase: all, retrieve, or answer.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Stable new RAG run ID used by managed multi-stage execution.",
    ),
    resume_run_id: str | None = typer.Option(None, "--resume-run-id"),
) -> None:
    """Run one foundational full-context or RAG baseline without reading gold."""

    from pmmc.baselines.rag_embedding import RAG_METHODS
    from pmmc.baselines.rag_workflow import RagRunPhase, run_rag_workflow
    from pmmc.baselines.workflow import run_full_context_workflow
    from pmmc.infrastructure.experiment_runs import load_local_run_registration

    settings = get_settings()
    try:
        resolved_phase = RagRunPhase(phase)
    except ValueError as exc:
        raise typer.BadParameter("must be all, retrieve, or answer", param_hint="--phase") from exc
    if resume_run_id is None:
        if method is None or dataset is None or snapshot is None:
            raise typer.BadParameter("new runs require --method, --dataset, and --snapshot")
        if dataset is DatasetName.UNKNOWN:
            raise typer.BadParameter("unknown is not a runnable benchmark", param_hint="--dataset")
        try:
            resolved_method = BaselineMethod(method)
        except ValueError as exc:
            raise typer.BadParameter(
                "unknown foundational baseline method",
                param_hint="--method",
            ) from exc
        full_context_methods = {
            BaselineMethod.FULL_CONTEXT_CAPTIONS,
            BaselineMethod.FULL_CONTEXT_RAW_IMAGES,
        }
        if resolved_method not in full_context_methods | RAG_METHODS:
            raise typer.BadParameter(
                "method is not one of the five foundational baselines",
                param_hint="--method",
            )
        if resolved_method in RAG_METHODS:
            if corpus_id is None:
                raise typer.BadParameter(
                    "RAG methods require --corpus-id", param_hint="--corpus-id"
                )
            if assemble_only:
                raise typer.BadParameter("--assemble-only is limited to full-context methods")
        else:
            if corpus_id is not None or run_id is not None:
                raise typer.BadParameter("--corpus-id and --run-id are limited to RAG methods")
            if resolved_phase is not RagRunPhase.ALL:
                raise typer.BadParameter("--phase is limited to RAG methods")
        try:
            selection = BenchmarkSelection(
                datasets=[dataset],
                question_types=question_type or [],
                source_categories=source_category or [],
                scenario_ids=scenario or [],
                query_ids=query_id or [],
                limit=limit,
                sample_size=sample_size,
                seed=seed,
                shard_count=shard_count,
                shard_index=shard_index,
                include_excluded=include_excluded,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    else:
        if any(
            (
                method is not None,
                dataset is not None,
                snapshot is not None,
                corpus_id is not None,
                assemble_only,
                run_id is not None,
            )
        ):
            raise typer.BadParameter(
                "--resume-run-id cannot be combined with method, dataset, snapshot, corpus, "
                "run-id, or assemble-only"
            )
        if any(
            (
                question_type,
                source_category,
                scenario,
                query_id,
                limit,
                sample_size,
                include_excluded,
                seed != 0,
                shard_count != 1,
                shard_index != 0,
            )
        ):
            raise typer.BadParameter("resume cannot change the frozen query selection")
        selection = None
        try:
            local = load_local_run_registration(settings.runs_path / resume_run_id)
            resolved_method = BaselineMethod(local.manifest.method)
        except Exception as exc:
            raise typer.BadParameter("resume run has no readable baseline manifest") from exc
        if resolved_method not in RAG_METHODS and resolved_phase is not RagRunPhase.ALL:
            raise typer.BadParameter("--phase is limited to RAG runs")
    try:
        if resolved_method in RAG_METHODS:
            run_dir = asyncio.run(
                run_rag_workflow(
                    settings=settings,
                    method=resolved_method if resume_run_id is None else None,
                    snapshot_id=snapshot,
                    corpus_id=corpus_id,
                    selection=selection,
                    answer_model_alias=answer_model,
                    phase=resolved_phase,
                    resume_run_id=resume_run_id,
                    run_id=run_id,
                )
            )
        else:
            run_dir = asyncio.run(
                run_full_context_workflow(
                    settings=settings,
                    method=resolved_method if resume_run_id is None else None,
                    snapshot_id=snapshot,
                    selection=selection,
                    answer_model_alias=answer_model,
                    assemble_only=assemble_only,
                    resume_run_id=resume_run_id,
                )
            )
    except KeyboardInterrupt as exc:
        typer.echo("Interrupted; the running database job leases can be resumed later.", err=True)
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        typer.echo(f"ERROR: baseline run failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Run artifacts: {run_dir}")


@app.command("compile-memory")
def compile_memory(
    target_stage: str = typer.Option("questions", "--target-stage"),
    snapshot: str | None = typer.Option(None, "--snapshot"),
    conversation_id: list[str] | None = typer.Option(None, "--conversation-id"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    sample_size: int | None = typer.Option(None, "--sample-size", min=1),
    seed: int = typer.Option(0, "--seed", min=0),
    shard_count: int = typer.Option(1, "--shard-count", min=1),
    shard_index: int = typer.Option(0, "--shard-index", min=0),
    split: str | None = typer.Option(None, "--split"),
    compiler_model: str | None = typer.Option(None, "--compiler-model"),
    config: Path | None = typer.Option(None, "--config"),
    planner_config: Path | None = typer.Option(None, "--planner-config"),
    doubter_config: Path | None = typer.Option(None, "--doubter-config"),
    foundation_config: Path | None = typer.Option(None, "--foundation-config"),
    questioner_freeze_manifest: Path | None = typer.Option(None, "--questioner-freeze-manifest"),
    program_pilot_audit_hash: str | None = typer.Option(
        None,
        "--program-pilot-audit-hash",
        help="Development-only audit identity for pmmc_compiler_program_pilot.",
    ),
    diagnostic_force_revision: bool = typer.Option(
        False,
        "--diagnostic-force-revision",
        help="Step 7B development-prefix conformance only.",
    ),
    run_id: str | None = typer.Option(None, "--run-id"),
    resume_run_id: str | None = typer.Option(None, "--resume-run-id"),
    unit_preparation: Path | None = typer.Option(None, "--unit-preparation"),
    phase: str = typer.Option(
        "all",
        "--phase",
        help=(
            "Questioner: all/questioner/index_text/index_multimodal. "
            "Programs: all/questioner/planner/qbank_index_text/"
            "qbank_index_multimodal/memory_index_text/"
            "memory_index_multimodal/program_query_text/"
            "program_query_multimodal/dry_run/r0_input/doubter/decision/finalize."
        ),
    ),
) -> None:
    """Run Questioner, round-zero programs, or the verified-QBank compiler."""

    from pmmc.compiler.questioner_config import load_questioner_config
    from pmmc.compiler.workflow import QuestionerWorkflowPhase, run_questioner_workflow
    from pmmc.infrastructure.database.compiler_repositories import (
        PostgresCompilerRunRepository,
    )
    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.errors import InvalidStateTransitionError
    from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
    from pmmc.models.registry import load_model_registry
    from pmmc.schemas import CompilationTargetStage, PartitionRole

    try:
        resolved_target = CompilationTargetStage(target_stage)
    except ValueError as exc:
        raise typer.BadParameter(
            "must be questions, programs, or verified_qbank",
            param_hint="--target-stage",
        ) from exc
    program_target = resolved_target in {
        CompilationTargetStage.PROGRAMS,
        CompilationTargetStage.VERIFIED_QBANK,
    }
    if program_target:
        from pmmc.compiler.program_workflow import ProgramWorkflowPhase

        try:
            resolved_program_phase = ProgramWorkflowPhase(phase)
        except ValueError as exc:
            raise typer.BadParameter(
                "invalid Step 6C program phase",
                param_hint="--phase",
            ) from exc
    else:
        try:
            resolved_questioner_phase = QuestionerWorkflowPhase(phase)
        except ValueError as exc:
            raise typer.BadParameter(
                "must be all, questioner, index_text, or index_multimodal",
                param_hint="--phase",
            ) from exc
    if limit is not None and sample_size is not None:
        raise typer.BadParameter("--limit and --sample-size are mutually exclusive")
    if resume_run_id is None and snapshot is None:
        raise typer.BadParameter("new Questioner runs require --snapshot")
    if run_id is not None and resume_run_id is not None:
        raise typer.BadParameter("--run-id and --resume-run-id are mutually exclusive")
    if resume_run_id is not None and any(
        value is not None for value in (snapshot, conversation_id, limit, sample_size, split)
    ):
        raise typer.BadParameter(
            "--resume-run-id cannot be combined with snapshot, split, or conversation selection"
        )
    if unit_preparation is not None and phase not in {
        "all",
        "questioner",
    }:
        raise typer.BadParameter("--unit-preparation is only valid for all or questioner phase")
    if resolved_target is CompilationTargetStage.QUESTIONS and (
        diagnostic_force_revision
        or any(
            value is not None
            for value in (
                planner_config,
                doubter_config,
                foundation_config,
                questioner_freeze_manifest,
                program_pilot_audit_hash,
            )
        )
    ):
        raise typer.BadParameter("compiler program options require a program target")
    if program_target:
        if (questioner_freeze_manifest is None) == (program_pilot_audit_hash is None):
            raise typer.BadParameter(
                "program targets require exactly one of --questioner-freeze-manifest or "
                "--program-pilot-audit-hash"
            )
        if program_pilot_audit_hash is not None:
            if re.fullmatch(r"[0-9a-f]{64}", program_pilot_audit_hash) is None:
                raise typer.BadParameter("--program-pilot-audit-hash must be SHA256")
        if diagnostic_force_revision and (
            resolved_target is not CompilationTargetStage.VERIFIED_QBANK
            or program_pilot_audit_hash is None
        ):
            raise typer.BadParameter(
                "--diagnostic-force-revision requires a development verified-QBank audit run"
            )
    try:
        resolved_split = PartitionRole(split).value if split is not None else None
    except ValueError as exc:
        raise typer.BadParameter(
            "must be development or paper_test",
            param_hint="--split",
        ) from exc

    settings = get_settings()

    async def resume_identity() -> tuple[str, set[str], int, int, int, str | None]:
        assert resume_run_id is not None
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with ExperimentUnitOfWork(sessions) as uow:
                assert uow.runs is not None
                manifest = await uow.runs.get_manifest(resume_run_id)
            async with sessions() as session:
                compiler = await PostgresCompilerRunRepository(session).get(resume_run_id)
            if compiler is None:
                raise KeyError(f"compiler run does not exist: {resume_run_id}")
            conversations = compiler.metadata.get("conversation_ids")
            if not isinstance(conversations, list) or not all(
                isinstance(item, str) for item in conversations
            ):
                raise ValueError("Questioner run has no resumable conversation denominator")
            return (
                compiler.snapshot_id,
                set(conversations),
                int(manifest.seed or 0),
                int(manifest.metadata.get("shard_count", 1)),
                int(manifest.metadata.get("shard_index", 0)),
                manifest.split,
            )
        finally:
            await engine.dispose()

    workflow_started = False
    try:
        config_path = (
            settings.resolve_path(config)
            if config is not None
            else (
                settings.resolve_path(Path("configs/compiler/questioner-v3.3.yaml"))
                if resolved_target is CompilationTargetStage.VERIFIED_QBANK
                else settings.resolved_questioner_config_path
            )
        )
        questioner = load_questioner_config(config_path)
        if compiler_model is not None:
            questioner = questioner.model_copy(update={"compiler_model_alias": compiler_model})
        registry = load_model_registry(settings.resolved_model_registry_path)
        selected_conversations = set(conversation_id or []) or None
        resolved_snapshot = snapshot
        resolved_seed = seed
        resolved_shard_count = shard_count
        resolved_shard_index = shard_index
        if resume_run_id is not None:
            (
                resolved_snapshot,
                selected_conversations,
                resolved_seed,
                resolved_shard_count,
                resolved_shard_index,
                resolved_split,
            ) = asyncio.run(resume_identity())
        if (
            program_target
            and program_pilot_audit_hash is not None
            and resolved_split != PartitionRole.DEVELOPMENT.value
        ):
            raise typer.BadParameter(
                "an unapproved program pilot is restricted to --split development"
            )
        if diagnostic_force_revision:
            if resolved_split != PartitionRole.DEVELOPMENT.value:
                raise typer.BadParameter(
                    "--diagnostic-force-revision is restricted to --split development"
                )
            resolved_run_identity = resume_run_id or run_id or ""
            if not resolved_run_identity.startswith("pmmc-step7b-prefix-"):
                raise typer.BadParameter(
                    "--diagnostic-force-revision requires a pmmc-step7b-prefix run ID"
                )
        assert resolved_snapshot is not None

        completed_units = 0
        planner_completed_units = 0
        planner_programs = 0
        planner_rejected = 0
        dry_run_completed_units = 0
        dry_run_executions = 0
        dry_run_insufficient = 0
        planner_progress_state: dict[str, float | int] = {
            "started_at": 0.0,
            "last_emit_at": 0.0,
            "last_emit_units": 0,
        }
        dry_run_progress_state: dict[str, float | int] = {
            "started_at": 0.0,
            "last_emit_at": 0.0,
            "last_emit_units": 0,
        }
        doubter_progress_state: dict[int, dict[str, float | int]] = {}

        def questioner_progress(
            unit_id: str,
            success: bool,
            error_summary: str | None,
        ) -> None:
            nonlocal completed_units
            completed_units += 1
            state = "succeeded" if success else "failed"
            suffix = (
                f" error={json.dumps(error_summary, ensure_ascii=True)}" if error_summary else ""
            )
            if not questioner.is_v3:
                typer.echo(
                    f"PROGRESS stage=questioner completed={completed_units} "
                    f"unit_id={unit_id} state={state}{suffix}"
                )

        def questioner_aggregate_progress(values: Mapping[str, object]) -> None:
            fields = (
                "completed_units",
                "total_units",
                "reused",
                "failed",
                "opportunities",
                "raw",
                "schema_valid",
                "static_pass",
                "verifier_pass",
                "generation_native_envelopes",
                "generation_terminal_rejections",
                "verifier_native_batches",
                "verifier_fallback_batches",
                "verifier_conservative_rejections",
                "revised",
                "rescued",
                "final_eligible",
                "visual",
                "multi_hop",
                "zero_unit",
                "workers_busy",
                "workers_total",
                "workers_uncapped",
                "worker_limit",
                "workers_avoided",
                "workers_processing",
                "gate_waiters",
                "queued_units",
                "remaining_units",
                "tokens_per_second",
                "recent_tokens_per_second",
                "recent_units_per_minute",
                "last_completion_seconds_ago",
                "failure_attempts",
            )
            completed_value = values.get("completed_units", 0)
            completed_count = (
                completed_value
                if isinstance(completed_value, int) and not isinstance(completed_value, bool)
                else 0
            )
            rate_fields = {
                "tokens_per_second",
                "recent_tokens_per_second",
                "recent_units_per_minute",
            }
            rendered = "scope=execution_attempt_delta " + " ".join(
                f"{key}="
                + (
                    "unknown"
                    if completed_count == 0 and key in rate_fields
                    else str(values.get(key, 0))
                )
                for key in fields
            )
            for key in (
                "failure_stages",
                "failure_codes",
                "failure_operations",
                "failure_database",
            ):
                detail = values.get(key)
                if isinstance(detail, Mapping) and detail:
                    rendered += f" {key}=" + json.dumps(
                        dict(detail), ensure_ascii=True, sort_keys=True
                    )
            typer.echo(f"PROGRESS stage=questioner_v3 {rendered}")

        index_progress_state: dict[str, dict[str, float | int]] = {}

        def index_progress(channel: str, current: int, total: int) -> None:
            now = time.monotonic()
            state = index_progress_state.setdefault(
                channel,
                {
                    "started_at": now,
                    "last_emit_at": now,
                    "last_emit_current": 0,
                },
            )
            last_emit_at = float(state["last_emit_at"])
            last_emit_current = int(state["last_emit_current"])
            if last_emit_current > 0 and current <= last_emit_current:
                return
            should_emit = (
                last_emit_current == 0
                or current >= total
                or now - last_emit_at >= 30.0
            )
            if not should_emit:
                return
            if last_emit_current == 0:
                rate_fields = (
                    "documents_per_second=unknown "
                    "recent_documents_per_second=unknown"
                )
            else:
                recent_elapsed = max(0.001, now - last_emit_at)
                elapsed = max(0.001, now - float(state["started_at"]))
                rate_fields = (
                    f"documents_per_second={current / elapsed:.3f} "
                    "recent_documents_per_second="
                    f"{max(0, current - last_emit_current) / recent_elapsed:.3f}"
                )
            typer.echo(
                f"PROGRESS stage=qbank_index channel={channel} "
                f"documents={current}/{total} "
                f"{rate_fields}"
            )
            state["last_emit_at"] = now
            state["last_emit_current"] = current

        def planner_progress(_unit_id: str, programs: int, rejected: int) -> None:
            nonlocal planner_completed_units, planner_programs, planner_rejected
            planner_completed_units += 1
            planner_programs += programs
            planner_rejected += rejected
            now = time.monotonic()
            last_emit_at = float(planner_progress_state["last_emit_at"])
            last_emit_units = int(planner_progress_state["last_emit_units"])
            first_emit = last_emit_units == 0
            if first_emit:
                planner_progress_state["started_at"] = now
                last_emit_at = now
            if not (
                first_emit
                or planner_completed_units - last_emit_units >= 25
                or now - last_emit_at >= 30.0
            ):
                return
            elapsed = max(0.001, now - float(planner_progress_state["started_at"]))
            recent_elapsed = max(0.001, now - last_emit_at)
            overall_rate = (
                "unknown"
                if first_emit
                else f"{planner_completed_units / elapsed:.3f}"
            )
            recent_rate = (
                "unknown"
                if first_emit
                else f"{(planner_completed_units - last_emit_units) / recent_elapsed:.3f}"
            )
            typer.echo(
                "PROGRESS stage=planner "
                f"units={planner_completed_units} programs={planner_programs} "
                f"rejected={planner_rejected} "
                f"units_per_second={overall_rate} "
                f"recent_units_per_second={recent_rate}"
            )
            planner_progress_state["last_emit_at"] = now
            planner_progress_state["last_emit_units"] = planner_completed_units

        def dry_run_progress(
            _unit_id: str,
            executions: int,
            insufficient: int,
        ) -> None:
            nonlocal dry_run_completed_units, dry_run_executions, dry_run_insufficient
            dry_run_completed_units += 1
            dry_run_executions += executions
            dry_run_insufficient += insufficient
            now = time.monotonic()
            last_emit_at = float(dry_run_progress_state["last_emit_at"])
            last_emit_units = int(dry_run_progress_state["last_emit_units"])
            first_emit = last_emit_units == 0
            if first_emit:
                dry_run_progress_state["started_at"] = now
                last_emit_at = now
            if not (
                first_emit
                or dry_run_completed_units - last_emit_units >= 25
                or now - last_emit_at >= 30.0
            ):
                return
            elapsed = max(0.001, now - float(dry_run_progress_state["started_at"]))
            overall_rate = (
                "unknown"
                if first_emit
                else f"{dry_run_completed_units / elapsed:.3f}"
            )
            typer.echo(
                "PROGRESS stage=dry_run "
                f"units={dry_run_completed_units} executions={dry_run_executions} "
                f"insufficient={dry_run_insufficient} "
                f"units_per_second={overall_rate}"
            )
            dry_run_progress_state["last_emit_at"] = now
            dry_run_progress_state["last_emit_units"] = dry_run_completed_units

        def doubter_progress(
            round_id: int,
            completed: int,
            total: int,
            counts: Mapping[str, int],
        ) -> None:
            now = time.monotonic()
            state = doubter_progress_state.setdefault(
                round_id,
                {"last_emit_at": now, "last_emit_completed": 0},
            )
            last_emit_at = float(state["last_emit_at"])
            last_emit_completed = int(state["last_emit_completed"])
            if completed == last_emit_completed and 0 < total <= completed:
                return
            if not (
                last_emit_completed == 0
                or completed >= total
                or completed - last_emit_completed >= 50
                or now - last_emit_at >= 30.0
            ):
                return
            progress_state = (
                "no_round_candidates" if total == 0
                else "assessment_complete" if completed >= total else "active"
            )
            typer.echo(
                "PROGRESS stage=doubter "
                f"round={round_id} assessed={completed} round_input={total} "
                f"revision_requested={counts.get('revise_program', 0)} "
                f"state={progress_state} "
                f"actions={json.dumps(dict(counts), ensure_ascii=True, sort_keys=True)}"
            )
            state["last_emit_at"] = now
            state["last_emit_completed"] = completed

        if program_target:
            from pmmc.compiler.doubter_config import load_doubter_config
            from pmmc.compiler.planner_config import load_planner_config
            from pmmc.compiler.program_workflow import (
                run_program_workflow,
                validate_program_protocol_compatibility,
            )
            from pmmc.pilots.approval import verify_approval

            planner_path = settings.resolve_path(
                planner_config
                or Path(
                    "configs/compiler/planner-v1.2.yaml"
                    if resolved_target is CompilationTargetStage.VERIFIED_QBANK
                    else "configs/compiler/planner-v1.yaml"
                )
            )
            planner_value = load_planner_config(planner_path)
            validate_program_protocol_compatibility(questioner, planner_value)
            foundation_path = settings.resolve_path(
                foundation_config
                or Path(
                    "configs/compiler/pmmc-v4.yaml"
                    if resolved_target is CompilationTargetStage.VERIFIED_QBANK
                    else "configs/compiler/pmmc-v2.yaml"
                )
            )
            doubter_value = (
                load_doubter_config(
                    settings.resolve_path(
                        doubter_config or Path("configs/compiler/doubter-v1.yaml")
                    )
                )
                if resolved_target is CompilationTargetStage.VERIFIED_QBANK
                else None
            )
            if diagnostic_force_revision and not (
                questioner.is_v35
                and planner_value.is_v14
                and doubter_value is not None
                and doubter_value.is_v12
            ):
                raise typer.BadParameter(
                    "diagnostic revision requires Questioner v3.5, Planner v1.4, "
                    "and Doubter v1.2"
                )
            if (
                resolved_target is CompilationTargetStage.PROGRAMS
                and doubter_config is not None
            ):
                raise typer.BadParameter(
                    "--doubter-config requires --target-stage verified_qbank"
                )
            program_settings = settings.model_copy(update={"compiler_config_path": foundation_path})
            if questioner_freeze_manifest is not None:
                freeze = verify_approval(settings.resolve_path(questioner_freeze_manifest))
                from pmmc.compiler.questioner_freeze import (
                    verify_questioner_freeze_compatibility,
                )

                verify_questioner_freeze_compatibility(freeze, questioner, registry)
                freeze_or_audit_hash = freeze.content_hash
                identity_kind = "approved_freeze"
            else:
                assert program_pilot_audit_hash is not None
                freeze_or_audit_hash = program_pilot_audit_hash
                identity_kind = "development_audit"
            workflow_started = True
            program_result = asyncio.run(
                run_program_workflow(
                    settings=program_settings,
                    registry=registry,
                    questioner=questioner,
                    planner=planner_value,
                    doubter=doubter_value,
                    snapshot_id=resolved_snapshot,
                    questioner_freeze_or_audit_hash=freeze_or_audit_hash,
                    questioner_identity_kind=identity_kind,
                    diagnostic_force_revision=diagnostic_force_revision,
                    run_id=resume_run_id or run_id,
                    conversation_ids=selected_conversations,
                    limit=limit if resume_run_id is None else None,
                    sample_size=sample_size if resume_run_id is None else None,
                    seed=resolved_seed,
                    shard_count=resolved_shard_count,
                    shard_index=resolved_shard_index,
                    split=resolved_split,
                    phase=resolved_program_phase,
                    on_questioner_item=questioner_progress,
                    on_questioner_progress=questioner_aggregate_progress,
                    on_planner_item=planner_progress,
                    on_index_progress=index_progress,
                    on_dry_run_item=dry_run_progress,
                    on_doubter_progress=doubter_progress,
                    unit_preparation=(
                        settings.resolve_path(unit_preparation)
                        if unit_preparation is not None
                        else None
                    ),
                )
            )
            result_payload = program_result.__dict__
        else:
            workflow_started = True
            questioner_result = asyncio.run(
                run_questioner_workflow(
                    settings=settings,
                    registry=registry,
                    config=questioner,
                    snapshot_id=resolved_snapshot,
                    run_id=resume_run_id or run_id,
                    conversation_ids=selected_conversations,
                    limit=limit if resume_run_id is None else None,
                    sample_size=sample_size if resume_run_id is None else None,
                    seed=resolved_seed,
                    shard_count=resolved_shard_count,
                    shard_index=resolved_shard_index,
                    split=resolved_split,
                    phase=resolved_questioner_phase,
                    on_questioner_item=questioner_progress,
                    on_questioner_progress=(
                        questioner_aggregate_progress if questioner.is_v3 else None
                    ),
                    on_index_progress=index_progress,
                    unit_preparation=(
                        settings.resolve_path(unit_preparation)
                        if unit_preparation is not None
                        else None
                    ),
                )
            )
            result_payload = questioner_result.__dict__
    except KeyboardInterrupt as exc:
        typer.echo("Interrupted; Questioner jobs can be resumed by run ID.", err=True)
        raise typer.Exit(code=130) from exc
    except typer.BadParameter:
        raise
    except (KeyError, ValueError) as exc:
        if workflow_started:
            _raise_compiler_workflow_failure(
                exc,
                program_target=program_target,
            )
        raise typer.BadParameter(str(exc)) from exc
    except InvalidStateTransitionError as exc:
        producer_name = (
            "Compiler program"
            if program_target
            else "Questioner"
        )
        typer.echo(
            f"ERROR: {producer_name} run failed (InvalidStateTransitionError): {exc}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        _raise_compiler_workflow_failure(
            exc,
            program_target=program_target,
        )
    if os.environ.get("PMMC_COMPACT_CHILD_SUMMARY") == "1":
        safe_keys = (
            "phase",
            "unit_count",
            "eligible_candidate_count",
            "program_count",
            "r0_program_count",
            "revision_program_count",
            "terminal_outcome_count",
            "doubter_assessment_count",
            "routing_member_count",
            "r0_gate_passed",
            "run_status",
            "memory_index_status",
            "planner_outcome_counts",
        )
        compact = {key: result_payload.get(key) for key in safe_keys}
        compact["planner_outcome_scope"] = "round_0_candidates"
        typer.echo(
            "COMPILER_PHASE_SUMMARY "
            + json.dumps(compact, sort_keys=True, separators=(",", ":"))
        )
    else:
        typer.echo(json.dumps(result_payload, indent=2, sort_keys=True))


@compiler_app.command("prepare-units")
def compiler_prepare_units(
    snapshot: str = typer.Option(..., "--snapshot"),
    run_id: str = typer.Option(..., "--run-id"),
    conversation_id: list[str] | None = typer.Option(None, "--conversation-id"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    sample_size: int | None = typer.Option(None, "--sample-size", min=1),
    seed: int = typer.Option(0, "--seed", min=0),
    shard_count: int = typer.Option(1, "--shard-count", min=1),
    shard_index: int = typer.Option(0, "--shard-index", min=0),
    config: Path | None = typer.Option(None, "--config"),
    execution_config: Path = typer.Option(..., "--execution-config"),
) -> None:
    """Prepare deterministic Questioner units without starting a model service."""

    from pmmc.compiler.preparation import (
        CompilationUnitPreparationManifest,
        CompilationUnitPreparationService,
    )
    from pmmc.compiler.questioner_config import load_questioner_config
    from pmmc.compiler.workflow import _select_conversations
    from pmmc.experiments.execution import load_execution_config
    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.repositories import (
        PostgresMemoryRepository,
        PostgresSessionRepository,
        PostgresSnapshotRepository,
    )
    from pmmc.models.registry import load_model_registry

    if limit is not None and sample_size is not None:
        raise typer.BadParameter("--limit and --sample-size are mutually exclusive")
    settings = get_settings()

    async def prepare() -> tuple[CompilationUnitPreparationManifest, Path]:
        questioner = load_questioner_config(
            settings.resolve_path(config)
            if config is not None
            else settings.resolved_questioner_config_path
        )
        execution = load_execution_config(settings.resolve_path(execution_config))
        prepared_settings = settings.model_copy(
            update={
                "token_counter_backend": execution.unit_preparation.counter_backend,
                "local_asset_cache_enabled": (execution.unit_preparation.local_asset_cache),
            }
        )
        registry = load_model_registry(prepared_settings.resolved_model_registry_path)
        engine, sessions = create_session_factory(prepared_settings, DatabaseRole.RUNTIME)
        try:
            await PostgresSnapshotRepository(sessions).get_manifest(snapshot)
            available_sessions = await PostgresSessionRepository(sessions).list_snapshot(snapshot)
            conversations = _select_conversations(
                [item.conversation_id for item in available_sessions],
                requested=set(conversation_id or []) or None,
                limit=limit,
                sample_size=sample_size,
                seed=seed,
                shard_count=shard_count,
                shard_index=shard_index,
            )
            selected = set(conversations)
            selected_sessions = [
                item for item in available_sessions if item.conversation_id in selected
            ]
            memories = await PostgresMemoryRepository(sessions).list_snapshot(
                snapshot, conversation_ids=selected
            )

            def progress(
                complete: int,
                total: int,
                session_count: int,
                unit_count: int,
                metadata: dict[str, object],
            ) -> None:
                typer.echo(
                    "UNIT_PREP "
                    f"conversations={complete}/{total} sessions={session_count} "
                    f"units={unit_count} reused_conversations="
                    f"{metadata.get('reused_conversations', 0)} "
                    f"counter_calls={metadata.get('processor_calls', 0)} "
                    f"cached_counter_calls="
                    f"{metadata.get('cached_processor_calls', 0)} "
                    f"request_cache_hits={metadata.get('request_cache_hits', 0)} "
                    f"image_cache_hits={metadata.get('image_cache_hits', 0)}"
                )

            manifest, _descriptors, root = CompilationUnitPreparationService(
                settings=prepared_settings,
                registry=registry,
                config=questioner,
                workers=execution.unit_preparation.workers,
                max_tasks_per_child=execution.unit_preparation.max_tasks_per_child,
                cache_policy=execution.unit_preparation.cache_policy,
            ).prepare(
                run_id=run_id,
                snapshot_id=snapshot,
                sessions=selected_sessions,
                memories=memories,
                on_progress=progress,
            )
            return manifest, root
        finally:
            await engine.dispose()

    try:
        manifest, root = asyncio.run(prepare())
    except Exception as exc:
        typer.echo(f"ERROR: unit preparation failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "preparation_id": manifest.preparation_id,
                "preparation_path": str(root),
                "conversation_count": manifest.conversation_count,
                "unit_count": manifest.unit_count,
                "status": "ready",
            },
            indent=2,
            sort_keys=True,
        )
    )


@compiler_app.command("verify-unit-preparation")
def compiler_verify_unit_preparation(
    preparation: Path = typer.Option(..., "--preparation"),
) -> None:
    """Verify every checksum and identity in a prepared unit bundle."""

    from pmmc.compiler.preparation import CompilationUnitPreparationService

    settings = get_settings()
    try:
        manifest, descriptors = CompilationUnitPreparationService.verify(
            settings.resolve_path(preparation)
        )
    except Exception as exc:
        typer.echo(
            f"ERROR: unit preparation verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "preparation_id": manifest.preparation_id,
                "conversation_count": manifest.conversation_count,
                "unit_count": len(descriptors),
                "status": "ready",
            },
            indent=2,
            sort_keys=True,
        )
    )


@compiler_app.command("build-qbank-index")
def compiler_build_qbank_index(
    run_id: str = typer.Option(..., "--run-id"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Resume a Questioner run and build or verify its four frozen QBank collections."""

    from pmmc.compiler.questioner_config import load_questioner_config
    from pmmc.compiler.workflow import run_questioner_workflow
    from pmmc.infrastructure.database.compiler_repositories import (
        PostgresCompilerRunRepository,
    )
    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
    from pmmc.models.registry import load_model_registry
    from pmmc.schemas import CompilerRunSpec

    settings = get_settings()

    async def load_identity() -> tuple[CompilerRunSpec, RunManifest]:
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                compiler = await PostgresCompilerRunRepository(session).get(run_id)
            async with ExperimentUnitOfWork(sessions) as uow:
                assert uow.runs is not None
                manifest = await uow.runs.get_manifest(run_id)
            if compiler is None:
                raise KeyError(f"compiler run does not exist: {run_id}")
            return compiler, manifest
        finally:
            await engine.dispose()

    try:
        compiler, manifest = asyncio.run(load_identity())
        config_path = (
            settings.resolve_path(config)
            if config is not None
            else settings.resolved_questioner_config_path
        )
        questioner = load_questioner_config(config_path)
        registry = load_model_registry(settings.resolved_model_registry_path)
        conversations = compiler.metadata.get("conversation_ids")
        if not isinstance(conversations, list):
            raise ValueError("Questioner run has no conversation denominator")
        result = asyncio.run(
            run_questioner_workflow(
                settings=settings,
                registry=registry,
                config=questioner,
                snapshot_id=compiler.snapshot_id,
                run_id=run_id,
                conversation_ids=set(conversations),
                seed=int(manifest.seed or 0),
                shard_count=int(manifest.metadata.get("shard_count", 1)),
                shard_index=int(manifest.metadata.get("shard_index", 0)),
            )
        )
    except Exception as exc:
        typer.echo(f"ERROR: QBank index build failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result.__dict__, indent=2, sort_keys=True))


@compiler_app.command("verify-questioner-run")
def compiler_verify_questioner_run(
    run_id: str = typer.Option(..., "--run-id"),
    require_managed_runtime: bool = typer.Option(False, "--require-managed-runtime"),
    require_schema_conformance: bool = typer.Option(
        False,
        "--require-schema-conformance",
    ),
) -> None:
    """Verify Questioner denominator, generation/gate accounting, and index readiness."""

    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.questioner_repositories import (
        verify_questioner_run,
    )

    settings = get_settings()

    async def verify() -> dict[str, object]:
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                return await verify_questioner_run(
                    session,
                    run_id,
                    require_managed_runtime=require_managed_runtime,
                    require_schema_conformance=require_schema_conformance,
                )
        finally:
            await engine.dispose()

    try:
        report = asyncio.run(verify())
    except Exception as exc:
        typer.echo(f"ERROR: Questioner verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(code=1)


@compiler_app.command("verify-questioner-schema-conformance")
def compiler_verify_questioner_schema_conformance(
    run_id: str = typer.Option(..., "--run-id"),
    result_json: Path | None = typer.Option(None, "--result-json"),
) -> None:
    """Verify a real managed Questioner v2 call before QBank indexing."""

    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.questioner_repositories import (
        verify_questioner_schema_conformance,
    )

    settings = get_settings()

    async def verify() -> dict[str, object]:
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                return await verify_questioner_schema_conformance(session, run_id)
        finally:
            await engine.dispose()

    try:
        report = asyncio.run(verify())
        if result_json is not None:
            destination = settings.resolve_path(result_json)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        typer.echo(
            f"ERROR: Questioner schema conformance failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(code=1)


@compiler_app.command("inspect-questioner-unit")
def compiler_inspect_questioner_unit(
    run_id: str = typer.Option(..., "--run-id"),
    compiler_unit_id: str = typer.Option(..., "--compiler-unit-id"),
) -> None:
    """Inspect runtime-safe generation and gate counts for one compilation unit."""

    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.questioner_repositories import (
        inspect_questioner_unit,
    )

    settings = get_settings()

    async def inspect() -> dict[str, object]:
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                return await inspect_questioner_unit(
                    session,
                    run_id=run_id,
                    compiler_unit_id=compiler_unit_id,
                )
        finally:
            await engine.dispose()

    try:
        report = asyncio.run(inspect())
    except Exception as exc:
        typer.echo(f"ERROR: Questioner unit inspection failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


@compiler_app.command("verify-questioner-export")
def compiler_verify_questioner_export(
    export_dir: Path = typer.Option(..., "--export-dir"),
) -> None:
    """Verify an offline Questioner export, including private model payloads."""

    from pmmc.compiler.questioner_export import verify_questioner_export

    settings = get_settings()
    try:
        report = verify_questioner_export(settings.resolve_path(export_dir))
    except Exception as exc:
        typer.echo(
            f"ERROR: Questioner export verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


@compiler_app.command("build-memory-index")
def compiler_build_memory_index(
    run_id: str = typer.Option(..., "--run-id"),
    channel: str = typer.Option("all", "--channel"),
    questioner_config: Path = typer.Option(
        Path("configs/compiler/questioner-v3.1.yaml"), "--questioner-config"
    ),
    planner_config: Path = typer.Option(
        Path("configs/compiler/planner-v1.yaml"), "--planner-config"
    ),
    foundation_config: Path = typer.Option(
        Path("configs/compiler/pmmc-v2.yaml"), "--foundation-config"
    ),
) -> None:
    """Build the frozen text/visual memory collections for one Planner run."""

    from pmmc.compiler.planner_config import load_planner_config
    from pmmc.compiler.program_workflow import ProgramWorkflowPhase, run_program_workflow
    from pmmc.compiler.questioner_config import load_questioner_config
    from pmmc.infrastructure.database.compiler_repositories import (
        PostgresCompilerRunRepository,
    )
    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.models.registry import load_model_registry

    phases = {
        "text": [ProgramWorkflowPhase.MEMORY_INDEX_TEXT],
        "multimodal": [ProgramWorkflowPhase.MEMORY_INDEX_MULTIMODAL],
        "all": [
            ProgramWorkflowPhase.MEMORY_INDEX_TEXT,
            ProgramWorkflowPhase.MEMORY_INDEX_MULTIMODAL,
        ],
    }
    if channel not in phases:
        raise typer.BadParameter("must be text, multimodal, or all", param_hint="--channel")
    settings = get_settings()

    async def run() -> list[dict[str, object]]:
        configured = settings.model_copy(
            update={"compiler_config_path": settings.resolve_path(foundation_config)}
        )
        engine, sessions = create_session_factory(configured, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                compiler = await PostgresCompilerRunRepository(session).get(run_id)
            if compiler is None:
                raise KeyError(f"compiler run does not exist: {run_id}")
        finally:
            await engine.dispose()
        if compiler.questioner_freeze_manifest_hash is None:
            raise ValueError("program run has no frozen Questioner/audit identity")
        questioner = load_questioner_config(settings.resolve_path(questioner_config))
        planner = load_planner_config(settings.resolve_path(planner_config))
        registry = load_model_registry(configured.resolved_model_registry_path)
        results: list[dict[str, object]] = []
        for phase in phases[channel]:
            value = await run_program_workflow(
                settings=configured,
                registry=registry,
                questioner=questioner,
                planner=planner,
                snapshot_id=compiler.snapshot_id,
                questioner_freeze_or_audit_hash=(compiler.questioner_freeze_manifest_hash),
                questioner_identity_kind=str(
                    compiler.metadata.get("questioner_identity_kind", "approved_freeze")
                ),
                phase=phase,
                run_id=run_id,
            )
            results.append(value.__dict__)
        return results

    try:
        result = asyncio.run(run())
    except Exception as exc:
        safe_diagnostics = getattr(exc, "safe_diagnostics", None)
        persisted_diagnostics: Mapping[str, object] | None = None
        if callable(safe_diagnostics):
            diagnostic_payload = safe_diagnostics()
            if isinstance(diagnostic_payload, Mapping):
                persisted_diagnostics = diagnostic_payload
                typer.echo(
                    "EMBEDDING_BATCH_FAILURE "
                    + json.dumps(
                        dict(diagnostic_payload),
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    err=True,
                )
        _persist_managed_child_diagnostic(
            event=(
                "EMBEDDING_BATCH_FAILURE"
                if persisted_diagnostics is not None
                else "MEMORY_INDEX_FAILURE"
            ),
            exc=exc,
            diagnostics=persisted_diagnostics,
        )
        typer.echo(f"ERROR: memory index build failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({"run_id": run_id, "phases": result}, indent=2, sort_keys=True))


@compiler_app.command("verify-planner-run")
def compiler_verify_planner_run(
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Verify the complete Questioner/Planner/index/dry-run denominator."""

    from pmmc.infrastructure.database.compiler_repositories import verify_compiler_run
    from pmmc.infrastructure.database.engine import create_session_factory

    settings = get_settings()

    async def verify() -> dict[str, object]:
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                return await verify_compiler_run(session, run_id)
        finally:
            await engine.dispose()

    try:
        report = asyncio.run(verify())
    except Exception as exc:
        typer.echo(f"ERROR: Planner run verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if not bool(report.get("valid")):
        raise typer.Exit(code=1)


@compiler_app.command("inspect-program")
def compiler_inspect_program(
    plan_id: str = typer.Option(..., "--plan-id"),
) -> None:
    """Inspect one program without printing questions, answers, or evidence content."""

    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
    from pmmc.schemas import ProgramValidationKind

    settings = get_settings()

    async def inspect() -> dict[str, object]:
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                assert uow.memory_programs is not None
                assert uow.planner is not None
                assert uow.program_dry_runs is not None
                assert uow.program_executions is not None
                program = await uow.memory_programs.get(plan_id)
                if program is None:
                    raise KeyError(f"memory program does not exist: {plan_id}")
                outcomes = await uow.planner.list_outcomes(str(program.run_id))
                outcome = next((item for item in outcomes if item.plan_id == plan_id), None)
                validations = [
                    await uow.planner.get_validation(plan_id, kind)
                    for kind in ProgramValidationKind
                ]
                dry_run = await uow.program_dry_runs.get(plan_id)
                execution = (
                    await uow.program_executions.get(dry_run[0].execution_id)
                    if dry_run is not None
                    else None
                )
            safe_steps = []
            for step in program.operator_sequence:
                safe_args = {
                    key: value
                    for key, value in step.args.items()
                    if key not in {"entity_mentions", "terms"}
                }
                if "entity_mentions" in step.args or "terms" in step.args:
                    raw_terms = step.args.get("entity_mentions", step.args.get("terms", []))
                    safe_args["term_count"] = len(raw_terms) if isinstance(raw_terms, list) else 0
                safe_steps.append(
                    {
                        "step_id": step.step_id,
                        "op_name": step.op_name,
                        "args": safe_args,
                        "expected_output_type": step.expected_output_type,
                    }
                )
            return {
                "plan_id": plan_id,
                "run_id": program.run_id,
                "operator_set_version": program.operator_set_version,
                "round_id": program.round_id,
                "steps": safe_steps,
                "outcome": outcome.model_dump(mode="json") if outcome else None,
                "validations": [
                    item.model_dump(mode="json") for item in validations if item is not None
                ],
                "dry_run": dry_run[0].model_dump(mode="json") if dry_run else None,
                "execution": (
                    {
                        "execution_id": execution.trace.execution_id,
                        "status": execution.trace.status.value,
                        "control_action": execution.trace.control_action.value,
                        "step_count": len(execution.steps),
                        "evidence_count": len(execution.evidence),
                    }
                    if execution is not None
                    else None
                ),
            }
        finally:
            await engine.dispose()

    try:
        report = asyncio.run(inspect())
    except Exception as exc:
        typer.echo(f"ERROR: program inspection failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


@compiler_app.command("verify-program-export")
def compiler_verify_program_export(
    export_dir: Path = typer.Option(..., "--export-dir"),
) -> None:
    """Verify a complete offline Step 6C runtime export."""

    from pmmc.compiler.program_export import verify_program_export

    try:
        report = verify_program_export(get_settings().resolve_path(export_dir))
    except Exception as exc:
        typer.echo(f"ERROR: program export verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


@compiler_app.command("verify-r0-input")
def compiler_verify_r0_input(
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Verify the immutable deterministic gate before Doubter starts."""

    from pmmc.compiler.r0_gate import build_r0_input_manifest
    from pmmc.infrastructure.database.constants import DatabaseRole
    from pmmc.infrastructure.database.engine import create_session_factory

    async def verify() -> dict[str, object]:
        engine, sessions = create_session_factory(
            get_settings(), DatabaseRole.RUNTIME
        )
        try:
            manifest = await build_r0_input_manifest(sessions, run_id=run_id)
            return manifest.model_dump(mode="json")
        finally:
            await engine.dispose()

    try:
        report = asyncio.run(verify())
    except Exception as exc:
        typer.echo(f"ERROR: R0 input verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if report.get("hard_gate_passed") is not True:
        raise typer.Exit(code=1)


@compiler_app.command("verify-doubter-run")
def compiler_verify_doubter_run(
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Verify continuous r0-r3 feedback and Planner revision lineage."""

    from pmmc.compiler.doubter_runner import verify_doubter_run
    from pmmc.infrastructure.database.constants import DatabaseRole
    from pmmc.infrastructure.database.engine import create_session_factory

    async def verify() -> dict[str, object]:
        engine, sessions = create_session_factory(
            get_settings(), DatabaseRole.RUNTIME
        )
        try:
            return await verify_doubter_run(sessions, run_id=run_id)
        finally:
            await engine.dispose()

    try:
        report = asyncio.run(verify())
    except Exception as exc:
        typer.echo(f"ERROR: Doubter verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if report.get("valid") is not True:
        raise typer.Exit(code=1)


@compiler_app.command("inspect-doubter-feedback")
def compiler_inspect_doubter_feedback(
    run_id: str = typer.Option(..., "--run-id"),
    qbank_id: str | None = typer.Option(None, "--qbank-id"),
    limit: int = typer.Option(20, "--limit", min=1, max=200),
) -> None:
    """Print typed feedback metadata without question, answer, or evidence text."""

    from pmmc.infrastructure.database.constants import DatabaseRole
    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.experiment_repositories import (
        ExperimentUnitOfWork,
    )
    from pmmc.schemas import DataScope

    async def inspect() -> list[dict[str, object]]:
        engine, sessions = create_session_factory(
            get_settings(), DatabaseRole.RUNTIME
        )
        try:
            async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                assert uow.doubter is not None
                assessments = await uow.doubter.list_assessments(run_id)
            rows: list[dict[str, object]] = [
                {
                    "feedback_id": item.feedback_id,
                    "qbank_id": item.qbank_id,
                    "round_id": item.round_id,
                    "action": item.action.value,
                    "issue_codes": [value.value for value in item.issue_codes],
                    "directive_kinds": [
                        value.directive.value for value in item.directives
                    ],
                    "provider_schema_conformant": item.provider_schema_conformant,
                    "content_hash": item.content_hash,
                }
                for item in assessments
                if qbank_id is None or item.qbank_id == qbank_id
            ]
            return rows[:limit]
        finally:
            await engine.dispose()

    try:
        report = asyncio.run(inspect())
    except Exception as exc:
        typer.echo(f"ERROR: Doubter inspection failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


@compiler_app.command("verify-routing-projection")
def compiler_verify_routing_projection(
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Verify r0-r3 projections, active-r2 decisions, and runtime routing."""

    from pmmc.compiler.verified_qbank import verify_routing_projection
    from pmmc.infrastructure.database.constants import DatabaseRole
    from pmmc.infrastructure.database.engine import create_session_factory

    async def verify() -> dict[str, object]:
        engine, sessions = create_session_factory(
            get_settings(), DatabaseRole.RUNTIME
        )
        try:
            return await verify_routing_projection(sessions, run_id=run_id)
        finally:
            await engine.dispose()

    try:
        report = asyncio.run(verify())
    except Exception as exc:
        typer.echo(
            f"ERROR: routing projection verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if report.get("valid") is not True:
        raise typer.Exit(code=1)


@compiler_app.command("export-step6d-pilot")
@compiler_app.command("export-program-pilot")
def compiler_export_program_pilot(
    source_manifest: Path = typer.Option(..., "--source-manifest"),
) -> None:
    """Build a deterministic Step 6C/6D public-private transfer bundle."""

    from pmmc.compiler.program_pilot import build_program_pilot_bundle
    from pmmc.schemas import CompilerProgramPilotSourceManifest

    settings = get_settings()
    try:
        source_path = settings.resolve_path(source_manifest)
        source = CompilerProgramPilotSourceManifest.model_validate_json(
            source_path.read_text(encoding="utf-8")
        )
        roots = {item.dataset: item for item in source.datasets}
        destination, manifest = build_program_pilot_bundle(
            partition_id=source.partition_id,
            samples=source.samples,
            runtime_exports={
                dataset: settings.resolve_path(value.runtime_export_dir)
                for dataset, value in roots.items()
            },
            coverage_exports={
                dataset: settings.resolve_path(value.coverage_export_dir)
                for dataset, value in roots.items()
            },
            snapshot_dirs={
                dataset: settings.resolve_path(value.snapshot_dir)
                for dataset, value in roots.items()
            },
            benchmark_root=settings.resolve_benchmark_path(source.benchmark_root),
            output_root=settings.resolve_path(source.output_root),
        )
    except Exception as exc:
        typer.echo(f"ERROR: program pilot export failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "pilot_id": manifest.pilot_id,
                "pilot_dir": str(destination),
                "status": manifest.status.value,
                "transfer_dir": str(destination / "transfer"),
            },
            indent=2,
            sort_keys=True,
        )
    )


@compiler_app.command("verify-step6d-pilot")
@compiler_app.command("verify-program-pilot")
def compiler_verify_program_pilot(
    pilot_dir: Path | None = typer.Option(None, "--pilot-dir"),
    archive: Path | None = typer.Option(None, "--archive"),
    checksum: Path | None = typer.Option(None, "--checksum"),
) -> None:
    """Verify a Step 6C/6D pilot bundle or its transfer archive."""

    from pmmc.compiler.program_pilot import (
        verify_program_pilot_archive,
        verify_program_pilot_bundle,
    )

    try:
        settings = get_settings()
        if pilot_dir is not None:
            if archive is not None or checksum is not None:
                raise ValueError("--pilot-dir cannot be combined with --archive/--checksum")
            manifest = verify_program_pilot_bundle(settings.resolve_path(pilot_dir))
        else:
            if archive is None or checksum is None:
                raise ValueError("provide --pilot-dir or both --archive and --checksum")
            manifest = verify_program_pilot_archive(
                settings.resolve_path(archive),
                settings.resolve_path(checksum),
            )
    except Exception as exc:
        typer.echo(f"ERROR: program pilot verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(manifest.model_dump_json(indent=2))


@compiler_app.command("compare-step6d-pilot")
@compiler_app.command("compare-program-pilot")
def compiler_compare_program_pilot(
    left: Path = typer.Option(..., "--left"),
    right: Path = typer.Option(..., "--right"),
) -> None:
    """Compare two Step 6C/6D public summaries on their frozen partition."""

    from pmmc.compiler.program_pilot import compare_program_pilots

    settings = get_settings()
    try:
        result = compare_program_pilots(settings.resolve_path(left), settings.resolve_path(right))
    except Exception as exc:
        typer.echo(f"ERROR: program pilot comparison failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@compiler_app.command("decide-r0-pilot")
@compiler_app.command("decide-program-pilot")
def compiler_decide_program_pilot(
    pilot_dir: Path = typer.Option(..., "--pilot-dir"),
    decision: str = typer.Option(..., "--decision"),
    reviewer_id: str = typer.Option(..., "--reviewer-id"),
    rationale_file: Path = typer.Option(..., "--rationale-file"),
) -> None:
    """Approve or reject a verified program pilot and emit immutable freezes."""

    from pmmc.compiler.program_pilot import decide_program_pilot
    from pmmc.schemas import PilotDecisionValue

    settings = get_settings()
    try:
        decision_dir, questioner_freeze, program_freeze = decide_program_pilot(
            pilot_dir=settings.resolve_path(pilot_dir),
            decision=PilotDecisionValue(decision),
            reviewer_id=reviewer_id,
            rationale=settings.resolve_path(rationale_file).read_text(encoding="utf-8").strip(),
        )
    except Exception as exc:
        typer.echo(f"ERROR: program pilot decision failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "decision_dir": str(decision_dir),
                "questioner_freeze_manifest": (
                    str(questioner_freeze) if questioner_freeze else None
                ),
                "compiler_program_freeze_manifest": (
                    str(program_freeze) if program_freeze else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@compiler_app.command("decide-step6d-pilot")
def compiler_decide_step6d_pilot(
    pilot_dir: Path = typer.Option(..., "--pilot-dir"),
    decision: str = typer.Option(..., "--decision"),
    reviewer_id: str = typer.Option(..., "--reviewer-id"),
    rationale_file: Path = typer.Option(..., "--rationale-file"),
) -> None:
    """Approve/reject Step 6D and emit Q, Program, and Verified-QBank freezes."""

    from pmmc.compiler.program_pilot import decide_step6d_pilot
    from pmmc.schemas import PilotDecisionValue

    settings = get_settings()
    try:
        (
            decision_dir,
            questioner_freeze,
            program_freeze,
            verified_qbank_freeze,
        ) = decide_step6d_pilot(
            pilot_dir=settings.resolve_path(pilot_dir),
            decision=PilotDecisionValue(decision),
            reviewer_id=reviewer_id,
            rationale=settings.resolve_path(rationale_file)
            .read_text(encoding="utf-8")
            .strip(),
        )
    except Exception as exc:
        typer.echo(f"ERROR: Step 6D decision failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "decision_dir": str(decision_dir),
                "questioner_freeze_manifest": (
                    str(questioner_freeze) if questioner_freeze else None
                ),
                "compiler_program_freeze_manifest": (
                    str(program_freeze) if program_freeze else None
                ),
                "verified_qbank_freeze_manifest": (
                    str(verified_qbank_freeze)
                    if verified_qbank_freeze
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@compiler_app.command("verify-verified-qbank-freeze")
def compiler_verify_verified_qbank_freeze(
    freeze_manifest: Path = typer.Option(..., "--freeze-manifest"),
) -> None:
    """Verify an approved active-r2 VerifiedQBankFreezeManifest."""

    from pmmc.compiler.program_pilot import verify_verified_qbank_freeze

    try:
        freeze = verify_verified_qbank_freeze(
            get_settings().resolve_path(freeze_manifest)
        )
    except Exception as exc:
        typer.echo(
            f"ERROR: Verified QBank freeze verification failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(freeze.model_dump_json(indent=2))


@compiler_app.command("verify-program-freeze")
def compiler_verify_program_freeze(
    freeze_manifest: Path = typer.Option(..., "--freeze-manifest"),
) -> None:
    """Verify an approved CompilerProgramFreezeManifest."""

    from pmmc.compiler.program_pilot import verify_program_freeze

    try:
        freeze = verify_program_freeze(get_settings().resolve_path(freeze_manifest))
    except Exception as exc:
        typer.echo(f"ERROR: program freeze verification failed ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(freeze.model_dump_json(indent=2))


@app.command()
def answer() -> None:
    """Reserved for the online PMMC query executor."""

    _not_implemented("answer")


@app.command()
def analyze() -> None:
    """Reserved for offline analysis jobs."""

    _not_implemented("analyze")


def main() -> None:
    """Console-script compatible entrypoint."""

    app()


if __name__ == "__main__":
    main()
