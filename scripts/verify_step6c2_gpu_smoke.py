#!/usr/bin/env python3
"""Verify the two-dataset, eight-replica Step 6C.2 GPU smoke."""

from __future__ import annotations

import asyncio
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from sqlalchemy import bindparam, text

from pmmc.compiler.program_export import verify_program_export
from pmmc.evaluation.qbank_export import verify_qbank_coverage_export
from pmmc.experiments.checkpoint import (
    ExperimentCheckpointManifest,
    ExperimentCheckpointService,
)
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_engine
from pmmc.infrastructure.settings import get_settings
from pmmc.runtime.hardware import NvmlHardwareProvider
from pmmc.smoke import (
    verify_step6c2_semantic_conformance_payload,
    verify_step6c_planner_conformance_payload,
)

STEP6C2_GPU_VERIFIER_CONTRACT_VERSION = "step6c2-gpu-verifier-v1"
_MODEL_STAGES = (
    "compiler_program:model-stage",
    "program_text:model-stage",
    "program_multimodal:model-stage",
    "qbank_semantic:model-stage",
)
_RUNTIME_ALIASES = {
    "compiler-qwen3.5-9b",
    "qwen3-embedding-8b",
    "qwen3-vl-embedding-8b",
}
_JUDGE_ALIAS = "judge-qwen3.5-9b"
_HARD_ZERO_FIELDS = (
    "verifier_scope_contamination_count",
    "planner_oracle_input_violation_count",
    "program_requirement_dataflow_violation_count",
)


def _emit_verification_event(event: str, **facts: object) -> None:
    payload = {
        "contract_version": STEP6C2_GPU_VERIFIER_CONTRACT_VERSION,
        "event": event,
        **facts,
    }
    print(
        "STEP6C2_GPU_VERIFY_EVENT "
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _emit_verification_failure(error_type: str) -> None:
    print(
        "STEP6C2_GPU_VERIFY_FAILURE "
        + json.dumps(
            {
                "contract_version": STEP6C2_GPU_VERIFIER_CONTRACT_VERSION,
                "error_type": error_type,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.read_bytes().splitlines())


def _capture_model_call_counts(summary_path: Path, output_path: Path) -> None:
    summary = _load(summary_path)
    datasets = summary.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Step 6C.2 summary has no dataset facts")
    counts: dict[str, dict[str, int]] = {}
    for dataset, value in sorted(datasets.items()):
        if not isinstance(value, dict):
            raise ValueError("Step 6C.2 dataset summary is malformed")
        runtime_export = Path(str(value.get("runtime_export_dir", "")))
        coverage_export = Path(str(value.get("coverage_export_dir", "")))
        counts[str(dataset)] = {
            "runtime": _jsonl_count(runtime_export / "runtime/model_calls.jsonl"),
            "semantic": _jsonl_count(
                coverage_export / "private/semantic_model_calls.jsonl"
            ),
        }
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
    if totals["tests"] < 16 or any(
        totals[key] for key in ("failures", "errors", "skipped")
    ):
        raise ValueError(f"Step 6C.2 deterministic fixture failed: {totals}")
    return totals["tests"]


async def _database_facts(
    compiler_run_ids: list[str],
    semantic_run_ids: list[str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    engine = create_engine(get_settings(), DatabaseRole.ADMIN)
    try:
        async with engine.connect() as connection:
            services = [
                dict(row)
                for row in (
                    await connection.execute(
                        text(
                            "SELECT runtime_service_id, status, process_id "
                            "FROM pmmc_runtime.managed_runtime_services "
                            "ORDER BY runtime_service_id"
                        )
                    )
                ).mappings()
            ]
            runtime_statement = text(
                "SELECT model_alias, count(*) AS call_count, "
                "count(*) FILTER (WHERE runtime_id IS NOT NULL AND "
                "runtime_service_id IS NOT NULL) AS managed_count, "
                "count(DISTINCT runtime_service_id) FILTER "
                "(WHERE runtime_service_id IS NOT NULL) AS service_count, "
                "count(*) FILTER (WHERE actual_model_id IS NULL) AS missing_identity "
                "FROM pmmc_runtime.model_calls WHERE run_id IN :run_ids "
                "AND status='success' GROUP BY model_alias"
            ).bindparams(bindparam("run_ids", expanding=True))
            runtime_rows = (
                await connection.execute(
                    runtime_statement,
                    {"run_ids": compiler_run_ids},
                )
            ).mappings()
            facts = {
                str(row["model_alias"]): {
                    "call_count": int(row["call_count"]),
                    "managed_count": int(row["managed_count"]),
                    "service_count": int(row["service_count"]),
                    "missing_identity": int(row["missing_identity"]),
                }
                for row in runtime_rows
            }
            semantic_statement = text(
                "SELECT model_alias, count(*) AS call_count, "
                "count(*) FILTER (WHERE runtime_id IS NOT NULL AND "
                "runtime_service_id IS NOT NULL) AS managed_count, "
                "count(DISTINCT runtime_service_id) FILTER "
                "(WHERE runtime_service_id IS NOT NULL) AS service_count, "
                "count(*) FILTER (WHERE actual_model_id IS NULL) AS missing_identity "
                "FROM pmmc_evaluation.model_calls "
                "WHERE semantic_run_id IN :semantic_run_ids AND status='success' "
                "GROUP BY model_alias"
            ).bindparams(bindparam("semantic_run_ids", expanding=True))
            semantic_rows = (
                await connection.execute(
                    semantic_statement,
                    {"semantic_run_ids": semantic_run_ids},
                )
            ).mappings()
            for row in semantic_rows:
                facts[str(row["model_alias"])] = {
                    "call_count": int(row["call_count"]),
                    "managed_count": int(row["managed_count"]),
                    "service_count": int(row["service_count"]),
                    "missing_identity": int(row["missing_identity"]),
                }
            return services, facts
    finally:
        await engine.dispose()


def _verify_checkpoint(summary: dict[str, object], checkpoint_root: Path) -> int:
    manifests = list(checkpoint_root.glob("*/manifest.json"))
    if not manifests:
        raise ValueError("Step 6C.2 GPU smoke produced no checkpoint")
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


def _model_stage_counts(path: Path) -> tuple[dict[str, int], set[str]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    counts = {
        stage: sum(
            int(item.get("model_startup_count", 0))
            for item in records
            if item.get("stage_id") == stage and item.get("reused") is False
        )
        for stage in _MODEL_STAGES
    }
    reused = {
        str(item.get("stage_id"))
        for item in records
        if item.get("reused") is True
    }
    return counts, reused


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
    _emit_verification_event(
        "summaries_loaded",
        first_status=first.get("status"),
        resumed_status=resumed.get("status"),
        same_launch=first.get("launch_id") == resumed.get("launch_id"),
    )
    if first.get("launch_id") != resumed.get("launch_id"):
        raise ValueError("resume changed the Step 6C.2 launch identity")
    if first.get("status") != "completed" or resumed.get("status") != "completed":
        raise ValueError("Step 6C.2 smoke has the wrong terminal state")
    if resumed.get("managed_runtime_cleanup_verified") is not True:
        raise ValueError("Step 6C.2 did not verify managed-runtime cleanup")

    execution = resumed.get("execution")
    first_execution = first.get("execution")
    if not isinstance(execution, dict) or not isinstance(first_execution, dict):
        raise ValueError("Step 6C.2 execution summary is missing")
    if execution.get("schedule") != "model-stage-major-v1":
        raise ValueError("Step 6C.2 did not use stage-major execution")
    if not execution.get("restored_checkpoint"):
        raise ValueError("Step 6C.2 resume did not restore a checkpoint")
    planner = verify_step6c_planner_conformance_payload(
        execution.get("planner_conformance"),
        expected_replica_count=8,
    )
    semantic_conformance = verify_step6c2_semantic_conformance_payload(
        execution.get("semantic_conformance"),
        expected_replica_count=8,
    )
    first_semantic = verify_step6c2_semantic_conformance_payload(
        first_execution.get("semantic_conformance"),
        expected_replica_count=8,
    )
    if semantic_conformance.content_hash != first_semantic.content_hash:
        raise ValueError("resume changed semantic conformance identity")
    _emit_verification_event(
        "conformance_validated",
        checkpoint_restored=True,
        planner_replica_count=planner.replica_count,
        semantic_replica_count=semantic_conformance.replica_count,
        semantic_request_count=semantic_conformance.request_count,
    )

    pilot = resumed.get("compiler_program_pilot")
    if not isinstance(pilot, dict) or pilot.get("status") != "completed_smoke":
        raise ValueError("Step 6C.2 smoke pilot state is invalid")
    pilot_manifest = pilot.get("automatic_summary")
    if not isinstance(pilot_manifest, dict) or pilot_manifest.get("valid") is not True:
        raise ValueError("Step 6C.2 smoke automatic integrity is invalid")

    datasets = resumed.get("datasets")
    first_datasets = first.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(first_datasets, dict):
        raise ValueError("Step 6C.2 dataset summaries are missing")
    if set(datasets) != {"memlens", "memgallery"}:
        raise ValueError("Step 6C.2 smoke must cover both benchmarks")
    first_call_counts = _load(first_call_counts_path)
    compiler_run_ids: list[str] = []
    semantic_run_ids: list[str] = []
    export_reports: dict[str, dict[str, object]] = {}
    coverage_reports: dict[str, dict[str, object]] = {}
    for dataset in sorted(datasets):
        value = datasets[dataset]
        if not isinstance(value, dict):
            raise ValueError("Step 6C.2 dataset summary is malformed")
        if value.get("selected_count") != 2:
            raise ValueError("Step 6C.2 smoke requires two queries per dataset")
        verification = value.get("program_verification")
        if not isinstance(verification, dict) or verification.get("valid") is not True:
            raise ValueError(f"program verification failed for {dataset}")
        if int(verification.get("unit_count", 17)) > 16:
            raise ValueError("Step 6C.2 smoke exceeded its sixteen-unit cap")

        runtime_export = Path(str(value.get("runtime_export_dir", "")))
        coverage_export = Path(str(value.get("coverage_export_dir", "")))
        program_report = verify_program_export(runtime_export)
        coverage_report = verify_qbank_coverage_export(coverage_export)
        if program_report.get("protocol_version") != "pmmc-questioner-v3.2":
            raise ValueError("Step 6C.2 did not use Questioner v3.2")
        if program_report.get("qbank_collection_count") != 4:
            raise ValueError("Step 6C.2 did not build four QBank collections")
        if len(set(program_report.get("memory_collection_ids", []))) != 2:
            raise ValueError("Step 6C.2 did not build two memory collections")
        for field in _HARD_ZERO_FIELDS:
            if int(program_report.get(field, -1)) != 0:
                raise ValueError(f"{dataset} failed hard contract: {field}")
        if coverage_report.get("semantic_match_present") is not True:
            raise ValueError(f"{dataset} has no semantic match facts")
        if coverage_report.get("semantic_native_output_zero") is not True:
            raise ValueError(f"{dataset} semantic Judge used a structured fallback")
        if int(coverage_report.get("broad_session_round_match_count", -1)) != 0:
            raise ValueError(f"{dataset} contains session-to-round overmatch")
        if coverage_report.get("semantic_run_id") != value.get("semantic_run_id"):
            raise ValueError(f"{dataset} semantic run identity changed")

        captured = first_call_counts.get(dataset)
        if not isinstance(captured, dict):
            raise ValueError("first model-call denominator is malformed")
        if _jsonl_count(runtime_export / "runtime/model_calls.jsonl") != int(
            captured.get("runtime", -1)
        ):
            raise ValueError("resume repeated a successful runtime model call")
        if _jsonl_count(
            coverage_export / "private/semantic_model_calls.jsonl"
        ) != int(captured.get("semantic", -1)):
            raise ValueError("resume repeated a successful semantic Judge call")

        compiler_run_ids.append(str(value["compiler_run_id"]))
        semantic_run_ids.append(str(value["semantic_run_id"]))
        export_reports[dataset] = program_report
        coverage_reports[dataset] = coverage_report

    _emit_verification_event(
        "dataset_exports_validated",
        compiler_run_count=len(compiler_run_ids),
        dataset_count=len(datasets),
        semantic_run_count=len(semantic_run_ids),
    )
    startup_counts, reused = _model_stage_counts(performance_path)
    _emit_verification_event(
        "model_stage_accounting",
        expected_startup_count=8,
        reused_model_stages=sorted(set(_MODEL_STAGES) & reused),
        startup_counts=startup_counts,
    )
    if set(startup_counts.values()) != {8}:
        raise ValueError(
            f"Step 6C.2 stages did not start eight replicas: {startup_counts}"
        )
    if not set(_MODEL_STAGES) <= reused:
        raise ValueError("resume did not reuse every Step 6C.2 model stage")

    services, calls = asyncio.run(
        _database_facts(compiler_run_ids, semantic_run_ids)
    )
    if not _RUNTIME_ALIASES <= set(calls) or _JUDGE_ALIAS not in calls:
        raise ValueError("Step 6C.2 did not persist all four model roles")
    for alias in [*_RUNTIME_ALIASES, _JUDGE_ALIAS]:
        facts = calls[alias]
        if (
            facts["call_count"] <= 0
            or facts["managed_count"] != facts["call_count"]
            or facts["missing_identity"]
        ):
            raise ValueError(f"managed model lineage is incomplete for {alias}")
    if calls["compiler-qwen3.5-9b"]["service_count"] != 8:
        raise ValueError("compiler calls did not cover all eight services")
    if semantic_conformance.request_count != 8:
        raise ValueError("semantic conformance did not exercise eight Judge services")
    _emit_verification_event(
        "model_lineage_validated",
        call_counts={
            alias: facts["call_count"] for alias, facts in sorted(calls.items())
        },
        service_counts={
            alias: facts["service_count"] for alias, facts in sorted(calls.items())
        },
    )
    service_ids = {str(item["runtime_service_id"]) for item in services}
    if not {
        item.runtime_service_id for item in semantic_conformance.replicas
    } <= service_ids:
        raise ValueError("semantic conformance services were not persisted")
    if not services or any(item.get("status") != "stopped" for item in services):
        raise ValueError("managed runtime services were not stopped")
    process_ids: set[int] = set()
    for item in services:
        process_id = item.get("process_id")
        if isinstance(process_id, int) and not isinstance(process_id, bool):
            process_ids.add(process_id)
    if any(NvmlHardwareProvider().process_memory_bytes(process_ids).values()):
        raise ValueError("managed runtime PIDs retained GPU memory")
    _emit_verification_event(
        "managed_runtime_cleanup_validated",
        gpu_memory_retained=False,
        process_count=len(process_ids),
        service_count=len(services),
    )

    checkpoint_sequence = _verify_checkpoint(resumed, checkpoint_root)
    fixture_test_count = _verify_fixture_junit(fixture_junit)
    _emit_verification_event(
        "checkpoint_and_fixture_validated",
        checkpoint_sequence=checkpoint_sequence,
        deterministic_fixture_test_count=fixture_test_count,
    )

    report = {
        "success": True,
        "contract_version": STEP6C2_GPU_VERIFIER_CONTRACT_VERSION,
        "launch_id": resumed["launch_id"],
        "datasets": sorted(datasets),
        "query_count_per_dataset": 2,
        "model_startup_counts": startup_counts,
        "model_call_facts": calls,
        "planner_conformance": planner.model_dump(mode="json"),
        "semantic_conformance": semantic_conformance.model_dump(mode="json"),
        "checkpoint_sequence": checkpoint_sequence,
        "deterministic_branch_fixture_tests": fixture_test_count,
        "program_export_reports": export_reports,
        "coverage_export_reports": coverage_reports,
        "resume_repeated_model_calls": False,
        "cross_benchmark_isolation_verified": True,
        "managed_gpu_memory_released": True,
    }
    _emit_verification_event(
        "completed",
        dataset_count=len(datasets),
        success=True,
    )
    return report


def main(arguments: list[str]) -> int:
    if arguments[:1] == ["--capture-model-calls"]:
        if len(arguments) != 3:
            print(
                "usage: verify_step6c2_gpu_smoke.py "
                "--capture-model-calls SUMMARY OUTPUT",
                file=sys.stderr,
            )
            return 2
        _capture_model_call_counts(Path(arguments[1]), Path(arguments[2]))
        return 0
    if len(arguments) != 7:
        print(
            "usage: verify_step6c2_gpu_smoke.py FIRST RESUMED PERFORMANCE "
            "CHECKPOINT_ROOT FIXTURE_JUNIT FIRST_CALL_COUNTS REPORT",
            file=sys.stderr,
        )
        return 2
    try:
        report = verify(
            Path(arguments[0]),
            Path(arguments[1]),
            Path(arguments[2]),
            Path(arguments[3]),
            Path(arguments[4]),
            Path(arguments[5]),
        )
        destination = Path(arguments[6])
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        _emit_verification_failure(type(exc).__name__)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
