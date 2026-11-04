#!/usr/bin/env python3
"""Verify the two-dataset, eight-replica Step 6C GPU smoke."""

from __future__ import annotations

import asyncio
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import bindparam, select, text

from pmmc.compiler.program_export import verify_program_export
from pmmc.experiments.checkpoint import (
    ExperimentCheckpointManifest,
    ExperimentCheckpointService,
)
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_engine
from pmmc.infrastructure.database.runtime_models import ManagedRuntimeServiceModel
from pmmc.infrastructure.settings import get_settings
from pmmc.runtime.hardware import NvmlHardwareProvider
from pmmc.smoke import verify_step6c_planner_conformance_payload

STEP6C_GPU_VERIFIER_CONTRACT_VERSION = "step6c-gpu-verifier-v2"
_MODEL_STAGES = (
    "compiler_program:model-stage",
    "program_text:model-stage",
    "program_multimodal:model-stage",
)
_REQUIRED_ALIASES = {
    "compiler-qwen3.5-9b",
    "qwen3-embedding-8b",
    "qwen3-vl-embedding-8b",
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_bytes().splitlines())


def _capture_model_call_counts(summary_path: Path, output_path: Path) -> None:
    summary = _load(summary_path)
    datasets = summary.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Step 6C summary has no dataset facts")
    counts: dict[str, int] = {}
    for dataset, value in sorted(datasets.items()):
        if not isinstance(value, dict):
            raise ValueError("Step 6C dataset summary is malformed")
        export_dir = Path(str(value.get("runtime_export_dir", "")))
        counts[str(dataset)] = _jsonl_count(export_dir / "runtime/model_calls.jsonl")
    output_path.write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_fixture_junit(path: Path) -> int:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        key: sum(int(item.attrib.get(key, "0")) for item in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if totals["tests"] < 8 or any(totals[key] for key in ("failures", "errors", "skipped")):
        raise ValueError(f"Step 6C deterministic branch fixture failed: {totals}")
    return totals["tests"]


@dataclass(frozen=True)
class _RuntimeServiceProcess:
    runtime_service_id: str
    status: str
    process_id: int | None


async def _database_facts(
    compiler_run_ids: list[str],
) -> tuple[list[_RuntimeServiceProcess], dict[str, dict[str, int]]]:
    engine = create_engine(get_settings(), DatabaseRole.ADMIN)
    try:
        async with engine.connect() as connection:
            service_rows = (
                await connection.execute(
                    select(
                        ManagedRuntimeServiceModel.runtime_service_id,
                        ManagedRuntimeServiceModel.status,
                        ManagedRuntimeServiceModel.process_id,
                    ).order_by(ManagedRuntimeServiceModel.runtime_service_id)
                )
            ).all()
            statement = text(
                "SELECT model_alias, count(*) AS call_count, "
                "count(*) FILTER (WHERE runtime_id IS NOT NULL AND "
                "runtime_service_id IS NOT NULL) AS managed_count, "
                "count(DISTINCT runtime_service_id) FILTER "
                "(WHERE runtime_service_id IS NOT NULL) AS service_count, "
                "count(*) FILTER (WHERE actual_model_id IS NULL) AS missing_identity "
                "FROM pmmc_runtime.model_calls WHERE run_id IN :run_ids "
                "AND status='success' GROUP BY model_alias ORDER BY model_alias"
            ).bindparams(bindparam("run_ids", expanding=True))
            call_rows = (
                (await connection.execute(statement, {"run_ids": compiler_run_ids}))
                .mappings()
                .all()
            )
            revision_statement = text(
                "WITH revisions(model_alias, model_revision) AS ("
                "SELECT questioner_model_alias, questioner_model_revision "
                "FROM pmmc_runtime.compiler_runs WHERE run_id IN :run_ids UNION ALL "
                "SELECT planner_model_alias, planner_model_revision "
                "FROM pmmc_runtime.compiler_runs WHERE run_id IN :run_ids UNION ALL "
                "SELECT memory_text_model_alias, memory_text_model_revision "
                "FROM pmmc_runtime.compiler_runs WHERE run_id IN :run_ids UNION ALL "
                "SELECT memory_multimodal_model_alias, memory_multimodal_model_revision "
                "FROM pmmc_runtime.compiler_runs WHERE run_id IN :run_ids) "
                "SELECT model_alias, count(DISTINCT model_revision) AS revision_count, "
                "count(*) FILTER (WHERE model_revision IS NULL OR "
                "model_revision !~ '^[0-9a-f]{40}$') AS invalid_revision "
                "FROM revisions WHERE model_alias IS NOT NULL "
                "GROUP BY model_alias ORDER BY model_alias"
            ).bindparams(bindparam("run_ids", expanding=True))
            revision_rows = {
                str(row["model_alias"]): {
                    "revision_count": int(row["revision_count"]),
                    "invalid_revision": int(row["invalid_revision"]),
                }
                for row in (
                    await connection.execute(
                        revision_statement,
                        {"run_ids": compiler_run_ids},
                    )
                ).mappings()
            }
            return (
                [
                    _RuntimeServiceProcess(
                        runtime_service_id=str(runtime_service_id),
                        status=str(status),
                        process_id=int(pid) if pid is not None else None,
                    )
                    for runtime_service_id, status, pid in service_rows
                ],
                {
                    str(row["model_alias"]): {
                        "call_count": int(row["call_count"]),
                        "managed_count": int(row["managed_count"]),
                        "service_count": int(row["service_count"]),
                        "missing_identity": int(row["missing_identity"]),
                        **revision_rows.get(
                            str(row["model_alias"]),
                            {"revision_count": 0, "invalid_revision": 1},
                        ),
                    }
                    for row in call_rows
                },
            )
    finally:
        await engine.dispose()


def _verify_checkpoint(summary: dict[str, object], checkpoint_root: Path) -> int:
    manifests = list(checkpoint_root.glob("*/manifest.json"))
    if not manifests:
        raise ValueError("Step 6C GPU smoke produced no checkpoint")
    latest = max(manifests, key=lambda path: int(path.parent.name)).parent
    checkpoint = ExperimentCheckpointManifest.from_json(
        (latest / "manifest.json").read_text(encoding="utf-8")
    )
    ExperimentCheckpointService(
        get_settings(),
        launch_id=str(summary["launch_id"]),
        root=checkpoint_root.parent,
        identity=dict(checkpoint.identity),
        retain_count=2,
    ).verify(latest)
    return checkpoint.sequence


def verify(
    first_summary_path: Path,
    resumed_summary_path: Path,
    performance_path: Path,
    checkpoint_root: Path,
    fixture_junit: Path,
    first_call_counts_path: Path,
) -> dict[str, object]:
    first = _load(first_summary_path)
    resumed = _load(resumed_summary_path)
    if first.get("launch_id") != resumed.get("launch_id"):
        raise ValueError("resume changed the Step 6C launch identity")
    if first.get("status") != "completed" or resumed.get("status") != "completed":
        raise ValueError("Step 6C smoke has the wrong launch terminal state")
    if resumed.get("managed_runtime_cleanup_verified") is not True:
        raise ValueError("Step 6C smoke did not verify managed-runtime cleanup")
    execution = resumed.get("execution")
    first_execution = first.get("execution")
    if not isinstance(execution, dict) or not isinstance(first_execution, dict):
        raise ValueError("Step 6C smoke has no execution facts")
    if execution.get("schedule") != "model-stage-major-v1":
        raise ValueError("Step 6C smoke did not use stage-major execution")
    if not execution.get("restored_checkpoint"):
        raise ValueError("Step 6C resume did not restore a checkpoint")
    planner_conformance = verify_step6c_planner_conformance_payload(
        execution.get("planner_conformance"),
        expected_replica_count=8,
    )
    first_planner_conformance = verify_step6c_planner_conformance_payload(
        first_execution.get("planner_conformance"),
        expected_replica_count=8,
    )
    if planner_conformance.content_hash != first_planner_conformance.content_hash:
        raise ValueError("resume changed the Planner conformance identity")
    pilot = resumed.get("compiler_program_pilot")
    if not isinstance(pilot, dict) or pilot.get("status") != "completed_smoke":
        raise ValueError("Step 6C smoke has the wrong pilot terminal state")

    datasets = resumed.get("datasets")
    first_datasets = first.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(first_datasets, dict):
        raise ValueError("Step 6C summaries have no dataset facts")
    if set(datasets) != {"memlens", "memgallery"}:
        raise ValueError("Step 6C smoke must cover both benchmarks")
    compiler_run_ids: list[str] = []
    first_call_counts = _load(first_call_counts_path)
    export_reports: dict[str, dict[str, object]] = {}
    total_batches = total_programs = total_dry_runs = 0
    for dataset in sorted(datasets):
        value = datasets[dataset]
        first_value = first_datasets[dataset]
        if not isinstance(value, dict) or not isinstance(first_value, dict):
            raise ValueError("Step 6C dataset summary is malformed")
        if value.get("selected_count") != 2:
            raise ValueError("Step 6C smoke requires two queries per benchmark")
        verification = value.get("program_verification")
        if not isinstance(verification, dict) or not verification.get("valid"):
            raise ValueError(f"Planner verification failed for {dataset}")
        if int(verification.get("unit_count", 17)) > 16:
            raise ValueError("Step 6C smoke exceeded its sixteen-unit cap")
        runtime_export = Path(str(value.get("runtime_export_dir", "")))
        report = verify_program_export(runtime_export)
        if report.get("dataset") != dataset or report.get("snapshot_id") != value.get(
            "snapshot_id"
        ):
            raise ValueError("Step 6C runtime export crossed dataset/snapshot scope")
        if report.get("protocol_version") != "pmmc-questioner-v3.1":
            raise ValueError("Step 6C smoke did not use Questioner v3.1")
        if report.get("qbank_collection_count") != 4:
            raise ValueError("Step 6C smoke does not have four QBank collections")
        if len(set(report.get("memory_collection_ids", []))) != 2:
            raise ValueError("Step 6C smoke does not have two memory collections")
        total_batches += int(report.get("planner_batch_count", 0))
        total_programs += int(report.get("program_count", 0))
        total_dry_runs += int(report.get("dry_run_count", 0))
        if _jsonl_count(runtime_export / "runtime/model_calls.jsonl") != int(
            first_call_counts.get(dataset, -1)
        ):
            raise ValueError("resume repeated a successful audited model call")
        export_reports[dataset] = report
        compiler_run_ids.append(str(value["compiler_run_id"]))
    if export_reports["memlens"]["run_id"] == export_reports["memgallery"]["run_id"]:
        raise ValueError("Step 6C benchmarks unexpectedly share a compiler run")

    performance = [
        json.loads(line)
        for line in performance_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    startup_counts = {
        stage: sum(
            int(item.get("model_startup_count", 0))
            for item in performance
            if item.get("stage_id") == stage and item.get("reused") is False
        )
        for stage in _MODEL_STAGES
    }
    if set(startup_counts.values()) != {8}:
        raise ValueError(f"Step 6C model stages did not start 8 replicas: {startup_counts}")
    reused = {str(item.get("stage_id")) for item in performance if item.get("reused") is True}
    if not set(_MODEL_STAGES) <= reused:
        raise ValueError("resume did not reuse every completed Step 6C model stage")

    services, calls = asyncio.run(_database_facts(compiler_run_ids))
    if not _REQUIRED_ALIASES <= set(calls):
        raise ValueError("Step 6C smoke did not call all three real model roles")
    for alias in _REQUIRED_ALIASES:
        facts = calls[alias]
        if facts["call_count"] <= 0 or facts["managed_count"] != facts["call_count"]:
            raise ValueError(f"managed runtime lineage is incomplete for {alias}")
        if facts["missing_identity"]:
            raise ValueError(f"actual model identity is missing for {alias}")
        if facts["revision_count"] != 1 or facts["invalid_revision"]:
            raise ValueError(f"locked model revision identity is invalid for {alias}")
    if calls["compiler-qwen3.5-9b"]["service_count"] != 8:
        raise ValueError("compiler calls did not cover all eight services")
    runtime_service_ids = {item.runtime_service_id for item in services}
    if planner_conformance.runtime_service_id not in runtime_service_ids:
        raise ValueError("Planner conformance is not linked to a persisted runtime service")
    if not services or any(item.status != "stopped" for item in services):
        raise ValueError("managed runtime services were not cleanly stopped")
    process_ids = {item.process_id for item in services if item.process_id is not None}
    if any(NvmlHardwareProvider().process_memory_bytes(process_ids).values()):
        raise ValueError("managed runtime PIDs retained GPU memory")

    return {
        "success": True,
        "launch_id": resumed["launch_id"],
        "datasets": sorted(datasets),
        "query_count_per_dataset": 2,
        "compiler_replica_count": 8,
        "model_startup_counts": startup_counts,
        "model_call_facts": calls,
        "benchmark_planner_batch_count": total_batches,
        "benchmark_program_count": total_programs,
        "benchmark_dry_run_count": total_dry_runs,
        "benchmark_zero_program_sample": total_programs == 0,
        "planner_conformance": planner_conformance.model_dump(mode="json"),
        "real_planner_and_dry_run_exercised": True,
        "checkpoint_sequence": _verify_checkpoint(resumed, checkpoint_root),
        "deterministic_branch_fixture_tests": _verify_fixture_junit(fixture_junit),
        "cross_benchmark_isolation_verified": True,
        "managed_gpu_memory_released": True,
        "resume_repeated_model_calls": False,
        "program_export_reports": export_reports,
    }


def main(arguments: list[str]) -> int:
    if arguments[:1] == ["--capture-model-calls"]:
        if len(arguments) != 3:
            print(
                "usage: verify_step6c_gpu_smoke.py --capture-model-calls SUMMARY OUTPUT",
                file=sys.stderr,
            )
            return 2
        _capture_model_call_counts(Path(arguments[1]), Path(arguments[2]))
        return 0
    if len(arguments) != 7:
        print(
            "usage: verify_step6c_gpu_smoke.py FIRST RESUMED PERFORMANCE "
            "CHECKPOINT_ROOT FIXTURE_JUNIT FIRST_CALL_COUNTS REPORT",
            file=sys.stderr,
        )
        return 2
    report = verify(
        Path(arguments[0]),
        Path(arguments[1]),
        Path(arguments[2]),
        Path(arguments[3]),
        Path(arguments[4]),
        Path(arguments[5]),
    )
    destination = Path(arguments[6])
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
