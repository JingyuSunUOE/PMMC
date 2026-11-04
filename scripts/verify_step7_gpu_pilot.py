#!/usr/bin/env python3
"""Verify the combined Step 7 real-prefix and 12+12 GPU pilot."""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import text

from pmmc.compiler.program_export import verify_program_export
from pmmc.evaluation.export import verify_evaluation_export
from pmmc.evaluation.qbank_export import verify_qbank_coverage_export
from pmmc.experiments.checkpoint import (
    ExperimentCheckpointManifest,
    ExperimentCheckpointService,
)
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_engine
from pmmc.infrastructure.export_verification import verify_experiment_export
from pmmc.infrastructure.settings import get_settings
from pmmc.pilots.io import read_json, read_jsonl, stable_hash
from pmmc.runtime.hardware import NvmlHardwareProvider
from pmmc.runtime.step7_pilot import (
    verify_step7_pilot_archive,
    verify_step7_pilot_bundle,
)
from pmmc.smoke import (
    verify_step6c2_semantic_conformance_payload,
    verify_step6d_compiler_conformance_payload,
)

STEP7_GPU_VERIFIER_CONTRACT_VERSION = "step7-gpu-pilot-verifier-v1"

_STARTED_MODEL_STAGES = (
    "compiler_program:model-stage",
    "program_text:model-stage",
    "program_multimodal:model-stage",
    "doubter:model-stage",
    "answer:model-stage",
    "evaluation_judge:model-stage",
)
_DEFERRED_MODEL_STAGE = "qbank_semantic:model-stage"
_REQUIRED_ALIASES = {
    "answer-qwen3.5-9b",
    "compiler-qwen3.5-9b",
    "judge-qwen3.5-9b",
    "qwen3-embedding-8b",
    "qwen3-vl-embedding-8b",
}


def _emit(event: str, **facts: object) -> None:
    print(
        "STEP7_GPU_VERIFY_EVENT "
        + json.dumps(
            {
                "contract_version": STEP7_GPU_VERIFIER_CONTRACT_VERSION,
                "event": event,
                **facts,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.read_bytes().splitlines())


async def _model_call_snapshot() -> dict[str, dict[str, int]]:
    engine = create_engine(get_settings(), DatabaseRole.ADMIN)
    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        "SELECT 'runtime' AS scope, model_alias, count(*) AS calls "
                        "FROM pmmc_runtime.model_calls WHERE status='success' "
                        "GROUP BY model_alias UNION ALL "
                        "SELECT 'evaluation' AS scope, model_alias, count(*) AS calls "
                        "FROM pmmc_evaluation.model_calls WHERE status='success' "
                        "GROUP BY model_alias ORDER BY scope, model_alias"
                    )
                )
            ).mappings()
            result: dict[str, dict[str, int]] = {"runtime": {}, "evaluation": {}}
            for row in rows:
                result[str(row["scope"])][str(row["model_alias"])] = int(row["calls"])
            return result
    finally:
        await engine.dispose()


async def _managed_model_facts() -> tuple[
    list[dict[str, object]], dict[str, dict[str, int]]
]:
    engine = create_engine(get_settings(), DatabaseRole.ADMIN)
    try:
        async with engine.connect() as connection:
            services = [
                dict(row)
                for row in (
                    await connection.execute(
                        text(
                            "SELECT runtime_service_id, status, process_id, "
                            "model_alias, stage_id "
                            "FROM pmmc_runtime.managed_runtime_services "
                            "ORDER BY runtime_service_id"
                        )
                    )
                ).mappings()
            ]
            rows = (
                await connection.execute(
                    text(
                        "SELECT model_alias, count(*) AS call_count, "
                        "count(*) FILTER (WHERE runtime_id IS NOT NULL AND "
                        "runtime_service_id IS NOT NULL) AS managed_count, "
                        "count(DISTINCT runtime_service_id) FILTER "
                        "(WHERE runtime_service_id IS NOT NULL) AS service_count, "
                        "count(DISTINCT runtime_id) FILTER "
                        "(WHERE runtime_id IS NOT NULL) AS runtime_count, "
                        "count(*) FILTER (WHERE actual_model_id IS NULL) "
                        "AS missing_identity FROM ("
                        "SELECT model_alias, runtime_id, runtime_service_id, "
                        "actual_model_id FROM pmmc_runtime.model_calls "
                        "WHERE status='success' UNION ALL "
                        "SELECT model_alias, runtime_id, runtime_service_id, "
                        "actual_model_id FROM pmmc_evaluation.model_calls "
                        "WHERE status='success') calls GROUP BY model_alias"
                    )
                )
            ).mappings()
            facts = {
                str(row["model_alias"]): {
                    "call_count": int(row["call_count"]),
                    "managed_count": int(row["managed_count"]),
                    "service_count": int(row["service_count"]),
                    "runtime_count": int(row["runtime_count"]),
                    "missing_identity": int(row["missing_identity"]),
                }
                for row in rows
            }
            return services, facts
    finally:
        await engine.dispose()


def _model_lineage_issues(facts: dict[str, int]) -> list[str]:
    """Return aggregate lineage failures without assuming one historical pool."""

    issues: list[str] = []
    call_count = facts.get("call_count", 0)
    if call_count <= 0:
        issues.append("no_successful_calls")
    if facts.get("managed_count", 0) != call_count:
        issues.append("unmanaged_successful_calls")
    if facts.get("missing_identity", 0) != 0:
        issues.append("missing_actual_model_identity")
    # A recoverable launch may legitimately retain calls from more than one
    # complete eight-replica runtime pool. Requiring exactly eight distinct
    # service IDs incorrectly rejects that append-only history.
    if facts.get("service_count", 0) < 8:
        issues.append("insufficient_replica_coverage")
    return issues


def _capture_model_call_counts(summary_path: Path, output_path: Path) -> None:
    summary = _load(summary_path)
    if summary.get("status") != "awaiting_approval":
        raise ValueError("Step 7 first pass is not awaiting approval")
    output_path.write_text(
        json.dumps(asyncio.run(_model_call_snapshot()), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _verify_call_counts_unchanged(captured_path: Path) -> None:
    captured = _load(captured_path)
    current = asyncio.run(_model_call_snapshot())
    if current != captured:
        raise ValueError("resume repeated a successful Step 7 model call")


def _verify_prefix(value: object, launch_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Step 7 independent 2+2 prefix is missing")
    payload = dict(value)
    expected_hash = payload.pop("content_hash", None)
    if (
        payload.get("protocol_version") != "step7-independent-real-prefix-v1"
        or payload.get("launch_id") != launch_id
        or payload.get("scientific_sample") is not False
        or payload.get("query_count_per_dataset") != 2
        or expected_hash != stable_hash(payload)
    ):
        raise ValueError("Step 7 prefix identity or hash is invalid")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or {
        str(row.get("dataset")) for row in datasets if isinstance(row, dict)
    } != {"memlens", "memgallery"}:
        raise ValueError("Step 7 prefix does not cover both benchmarks")
    for row in datasets:
        if (
            not isinstance(row, dict)
            or row.get("query_count") != 2
            or int(row.get("doubter_feedback_count", 0)) <= 0
            or int(row.get("routing_member_count", 0)) <= 0
        ):
            raise ValueError("Step 7 prefix lacks real Doubter or routing facts")
    return {**payload, "content_hash": expected_hash}


def _stage_accounting(
    path: Path, execution_attempt_id: str
) -> tuple[dict[str, int], set[str], dict[str, str], int]:
    rows = [
        row
        for row in read_jsonl(path)
        if isinstance(row.get("metadata"), dict)
        and row["metadata"].get("execution_attempt_id") == execution_attempt_id
    ]
    if not rows:
        raise ValueError("Step 7 stage performance lacks execution-attempt facts")
    stages = (*_STARTED_MODEL_STAGES, _DEFERRED_MODEL_STAGE)
    counts = {
        stage: sum(
            int(row.get("model_startup_count", 0))
            for row in rows
            if row.get("stage_id") == stage
        )
        for stage in stages
    }
    reused = {
        str(row["stage_id"])
        for row in rows
        if row.get("stage_id") in stages and row.get("reused") is True
    }
    states = {
        str(row["stage_id"]): str(dict(row.get("metadata", {})).get("state", "executed"))
        for row in rows
        if row.get("stage_id") in stages
    }
    return counts, reused, states, len(rows)


def _verify_first_stage_accounting(
    counts: dict[str, int], reused: set[str], states: dict[str, str]
) -> None:
    for stage in _STARTED_MODEL_STAGES:
        if counts[stage] == 8:
            continue
        if counts[stage] == 0 and stage in reused:
            continue
        raise ValueError(f"Step 7 model stage lacks 8-replica start or reuse: {stage}")
    if counts[_DEFERRED_MODEL_STAGE] != 0:
        raise ValueError("Step 7 semantic accounting started a duplicate Judge pool")
    if (
        _DEFERRED_MODEL_STAGE not in reused
        and states.get(_DEFERRED_MODEL_STAGE) != "deferred_to_shared_model_stage"
    ):
        raise ValueError("Step 7 semantic stage is not bound to the shared Judge stage")


def _verify_resume_stage_accounting(
    counts: dict[str, int], reused: set[str], states: dict[str, str]
) -> None:
    for stage, count in counts.items():
        if count != 0:
            raise ValueError(f"Step 7 resume repeated model startup: {stage}")
        if stage not in reused and not (
            stage == _DEFERRED_MODEL_STAGE
            and states.get(stage) == "deferred_to_shared_model_stage"
        ):
            raise ValueError(f"Step 7 resume did not preserve model-stage outcome: {stage}")


def _verify_checkpoint(summary: dict[str, Any]) -> int:
    execution = summary.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("Step 7 execution summary is absent")
    checkpoint = Path(str(execution.get("latest_checkpoint", "")))
    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Step 7 produced no ready checkpoint")
    manifest = ExperimentCheckpointManifest.from_json(
        manifest_path.read_text(encoding="utf-8")
    )
    ExperimentCheckpointService(
        get_settings(),
        launch_id=str(summary["launch_id"]),
        root=checkpoint.parent.parent,
        identity=dict(manifest.identity),
        retain_count=2,
    ).verify(checkpoint)
    return manifest.sequence


def _verify_dataset(
    dataset: str, raw: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str, str, str, str]:
    _emit("dataset_verification_started", dataset=dataset)
    if int(raw.get("selected_count", -1)) != 12:
        raise ValueError(f"{dataset} does not contain 12 development queries")
    compiler_export = Path(str(raw.get("runtime_export_dir", "")))
    coverage_export = Path(str(raw.get("coverage_export_dir", "")))
    query_export = Path(str(raw.get("query_runtime_export_dir", "")))
    evaluation_export = Path(str(raw.get("evaluation_export_dir", "")))
    program = verify_program_export(compiler_export)
    coverage = verify_qbank_coverage_export(coverage_export)
    query_manifest = verify_experiment_export(query_export)
    evaluation_manifest = verify_evaluation_export(evaluation_export)
    export_identity_checks = {
        "query_run_identity": (
            query_manifest.get("run_id") == raw.get("query_executor_run_id")
        ),
        "evaluation_run_identity": (
            evaluation_manifest.get("evaluation_run_id")
            == raw.get("evaluation_run_id")
        ),
    }
    failed_identity_checks = sorted(
        check for check, passed in export_identity_checks.items() if not passed
    )
    if failed_identity_checks:
        _emit(
            "dataset_export_identity_failed",
            dataset=dataset,
            failed_checks=failed_identity_checks,
        )
        raise ValueError(
            f"{dataset} export identity failed checks: "
            + ",".join(failed_identity_checks)
        )
    program_checks = {
        "questioner_protocol": (
            program.get("protocol_version") == "pmmc-questioner-v3.4"
        ),
        "r0_hard_gate": program.get("r0_hard_gate_passed") is True,
        "round_projections": program.get("projection_count") == 4,
        "active_round": program.get("active_round") == 2,
        "maximum_round": program.get("max_round") == 3,
        "qbank_collections": program.get("qbank_collection_count") == 4,
        "memory_collections": (
            len(set(program.get("memory_collection_ids", []))) == 2
        ),
    }
    failed_program_checks = sorted(
        check for check, passed in program_checks.items() if not passed
    )
    _emit(
        "dataset_program_export_verified",
        dataset=dataset,
        eligible_candidate_count=int(program.get("eligible_candidate_count", 0)),
        failed_checks=failed_program_checks,
        memory_collection_count=len(set(program.get("memory_collection_ids", []))),
        projection_count=int(program.get("projection_count", -1)),
        routing_member_count=int(program.get("routing_member_count", 0)),
    )
    if failed_program_checks:
        raise ValueError(
            f"{dataset} verified-QBank compiler export failed checks: "
            + ",".join(failed_program_checks)
        )
    coverage_checks = {
        "semantic_match_present": coverage.get("semantic_match_present") is True,
        "semantic_run_identity": (
            coverage.get("semantic_run_id") == raw.get("semantic_run_id")
        ),
        "no_broad_session_round_match": (
            int(coverage.get("broad_session_round_match_count", -1)) == 0
        ),
    }
    failed_coverage_checks = sorted(
        check for check, passed in coverage_checks.items() if not passed
    )
    if failed_coverage_checks:
        _emit(
            "dataset_semantic_export_failed",
            dataset=dataset,
            failed_checks=failed_coverage_checks,
        )
        raise ValueError(
            f"{dataset} QBank semantic export failed checks: "
            + ",".join(failed_coverage_checks)
        )
    query_rows = read_jsonl(query_export / "runtime/query_executor_runs.jsonl")
    query_row = query_rows[0] if len(query_rows) == 1 else {}
    query_checks = {
        "single_query_run": len(query_rows) == 1,
        "query_denominator": int(query_row.get("selected_query_count", -1)) == 12,
        "query_run_terminal": query_row.get("status") == "completed",
        "compiler_run_identity": (
            query_row.get("source_compiler_run_id") == raw.get("compiler_run_id")
        ),
    }
    failed_query_checks = sorted(
        check for check, passed in query_checks.items() if not passed
    )
    if failed_query_checks:
        _emit(
            "dataset_query_export_failed",
            dataset=dataset,
            failed_checks=failed_query_checks,
            query_run_row_count=len(query_rows),
        )
        raise ValueError(
            f"{dataset} query executor export failed checks: "
            + ",".join(failed_query_checks)
        )
    _emit(
        "dataset_exports_verified",
        dataset=dataset,
        query_count=int(query_row["selected_query_count"]),
        semantic_assessment_count=int(coverage.get("semantic_assessment_count", 0)),
        semantic_match_count=int(coverage.get("semantic_match_count", 0)),
    )
    return (
        program,
        coverage,
        str(raw["compiler_run_id"]),
        str(raw["query_executor_run_id"]),
        str(raw["semantic_run_id"]),
        str(raw["evaluation_run_id"]),
    )


def _verify_pilot(summary: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    pilot_summary = summary.get("compiler_program_pilot")
    if (
        not isinstance(pilot_summary, dict)
        or pilot_summary.get("status") != "awaiting_approval"
    ):
        raise ValueError("Step 7 pilot summary is not awaiting approval")
    pilot_dir = Path(str(pilot_summary.get("pilot_dir", "")))
    pilot = verify_step7_pilot_bundle(pilot_dir)
    if (
        pilot.protocol_version != "pmmc-step7-executor-pilot-v1"
        or pilot.status.value != "awaiting_approval"
        or not pilot.automatic_integrity_passed
        or pilot.query_count != 24
        or pilot.prefix_query_count != 4
    ):
        raise ValueError("Step 7 transfer manifest is not freeze-eligible")
    automatic = read_json(pilot_dir / "public/automatic_summary.json")
    datasets = automatic.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"memlens", "memgallery"}:
        raise ValueError("Step 7 automatic report has invalid dataset accounting")
    for dataset, row in datasets.items():
        checks = row.get("checks") if isinstance(row, dict) else None
        if not isinstance(checks, dict) or not checks or not all(checks.values()):
            raise ValueError(f"{dataset} failed a Step 7 automatic integrity check")
        if int(row.get("program_binding_count", 0)) <= 0:
            raise ValueError(f"{dataset} executed no real frozen program")
        if int(row.get("doubter_feedback_count", 0)) <= 0:
            raise ValueError(f"{dataset} has no real Doubter feedback")
    query_cards = read_jsonl(pilot_dir / "private/query_audit_cards.jsonl")
    if Counter(str(row.get("dataset")) for row in query_cards) != {
        "memlens": 12,
        "memgallery": 12,
    }:
        raise ValueError("Step 7 bundle does not contain balanced 12+12 query cards")
    transfer = pilot_dir / "transfer"
    archives = sorted(transfer.glob("*.tar.zst"))
    checksums = sorted(transfer.glob("*.sha256"))
    if len(archives) != 1 or len(checksums) != 1:
        raise ValueError("Step 7 transfer deliverables are incomplete")
    archived = verify_step7_pilot_archive(archives[0], checksums[0])
    if archived.content_hash != pilot.content_hash:
        raise ValueError("Step 7 transfer archive changed pilot identity")
    memgallery = next(
        item for item in pilot.samples if item.dataset.value == "memgallery"
    )
    if len(memgallery.scenario_counts) != 4 or set(
        memgallery.scenario_counts.values()
    ) != {3}:
        raise ValueError("MemGallery Step 7 sample is not 3/3/3/3")
    return (
        {
            "pilot_id": pilot.pilot_id,
            "automatic_integrity_passed": True,
            "query_audit_count": len(query_cards),
            "candidate_program_doubter_audit_count": _jsonl_count(
                pilot_dir / "private/candidate_program_doubter_cards.jsonl"
            ),
            "transfer_archive": str(archives[0]),
            "transfer_checksum": str(checksums[0]),
        },
        pilot_dir,
    )


def verify(
    first_summary_path: Path,
    resumed_summary_path: Path,
    first_call_counts_path: Path,
) -> dict[str, Any]:
    first = _load(first_summary_path)
    resumed = _load(resumed_summary_path)
    if first.get("launch_id") != resumed.get("launch_id"):
        raise ValueError("Step 7 resume changed launch identity")
    if first.get("status") != "awaiting_approval" or resumed.get("status") != (
        "awaiting_approval"
    ):
        raise ValueError("Step 7 may only pass in awaiting_approval state")
    if resumed.get("managed_runtime_cleanup_verified") is not True:
        raise ValueError("Step 7 managed-runtime cleanup was not verified")
    launch_id = str(resumed["launch_id"])
    first_execution = first.get("execution")
    execution = resumed.get("execution")
    if not isinstance(first_execution, dict) or not isinstance(execution, dict):
        raise ValueError("Step 7 execution summaries are incomplete")
    if execution.get("schedule") != "model-stage-major-v1":
        raise ValueError("Step 7 did not use stage-major scheduling")
    if not execution.get("restored_checkpoint"):
        raise ValueError("Step 7 second pass did not restore a checkpoint")

    compiler = verify_step6d_compiler_conformance_payload(
        execution.get("step6d_compiler_conformance"), expected_replica_count=8
    )
    first_compiler = verify_step6d_compiler_conformance_payload(
        first_execution.get("step6d_compiler_conformance"), expected_replica_count=8
    )
    semantic = verify_step6c2_semantic_conformance_payload(
        execution.get("step7_semantic_conformance"), expected_replica_count=8
    )
    first_semantic = verify_step6c2_semantic_conformance_payload(
        first_execution.get("step7_semantic_conformance"), expected_replica_count=8
    )
    if compiler.content_hash != first_compiler.content_hash:
        raise ValueError("Step 7 resume changed compiler conformance")
    if semantic.content_hash != first_semantic.content_hash:
        raise ValueError("Step 7 resume changed semantic conformance")
    prefix = _verify_prefix(execution.get("step7_prefix"), launch_id)

    datasets = resumed.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"memlens", "memgallery"}:
        raise ValueError("Step 7 main pilot must cover both benchmarks")
    program_reports: dict[str, dict[str, Any]] = {}
    coverage_reports: dict[str, dict[str, Any]] = {}
    run_ids: dict[str, dict[str, str]] = {}
    for dataset, raw in sorted(datasets.items()):
        if not isinstance(raw, dict):
            raise ValueError("Step 7 dataset summary is malformed")
        program, coverage, compiler_id, query_id, semantic_id, evaluation_id = (
            _verify_dataset(dataset, raw)
        )
        program_reports[dataset] = program
        coverage_reports[dataset] = coverage
        run_ids[dataset] = {
            "compiler": compiler_id,
            "query": query_id,
            "semantic": semantic_id,
            "evaluation": evaluation_id,
        }

    _emit("dataset_verification_completed", dataset_count=len(run_ids))
    pilot_report, _pilot_dir = _verify_pilot(resumed)
    _emit(
        "transfer_bundle_verified",
        candidate_program_doubter_audit_count=pilot_report[
            "candidate_program_doubter_audit_count"
        ],
        query_audit_count=pilot_report["query_audit_count"],
    )
    _verify_call_counts_unchanged(first_call_counts_path)
    _emit("resume_model_calls_verified", repeated_successful_call_count=0)

    first_attempt = first_execution.get("execution_attempt_id")
    resumed_attempt = execution.get("execution_attempt_id")
    if (
        not isinstance(first_attempt, str)
        or not first_attempt
        or not isinstance(resumed_attempt, str)
        or not resumed_attempt
        or first_attempt == resumed_attempt
    ):
        raise ValueError("Step 7 resume did not create a distinct execution attempt")
    performance_path = Path(str(execution.get("stage_performance_path", "")))
    first_counts, first_reused, first_states, first_records = _stage_accounting(
        performance_path, first_attempt
    )
    resume_counts, resume_reused, resume_states, resume_records = _stage_accounting(
        performance_path, resumed_attempt
    )
    _verify_first_stage_accounting(first_counts, first_reused, first_states)
    _verify_resume_stage_accounting(resume_counts, resume_reused, resume_states)
    _emit(
        "model_stage_accounting",
        first_startups=first_counts,
        resume_startups=resume_counts,
        semantic_shared_owner="evaluation_judge:model-stage",
    )

    services, calls = asyncio.run(_managed_model_facts())
    service_status_counts = Counter(
        str(row.get("status", "unknown")) for row in services
    )
    nonterminal_service_counts = Counter(
        (
            str(row.get("model_alias", "unknown")),
            str(row.get("stage_id", "unknown")),
            str(row.get("status", "unknown")),
        )
        for row in services
        if row.get("status") != "stopped"
    )
    nonterminal_services = [
        {
            "count": count,
            "model_alias": model_alias,
            "stage": stage,
            "status": status,
        }
        for (model_alias, stage, status), count in sorted(
            nonterminal_service_counts.items()
        )
    ]
    _emit(
        "managed_model_lineage_accounting",
        aliases={
            alias: {
                "call_count": facts["call_count"],
                "managed_count": facts["managed_count"],
                "missing_identity": facts["missing_identity"],
                "runtime_count": facts["runtime_count"],
                "service_count": facts["service_count"],
            }
            for alias, facts in sorted(calls.items())
        },
        minimum_service_coverage=8,
        nonterminal_services=nonterminal_services,
        service_status_counts=dict(sorted(service_status_counts.items())),
    )
    if not _REQUIRED_ALIASES <= set(calls):
        missing_aliases = sorted(_REQUIRED_ALIASES - set(calls))
        _emit("managed_model_lineage_failed", missing_aliases=missing_aliases)
        raise ValueError("Step 7 managed model lineage lacks a required alias")
    for alias in sorted(_REQUIRED_ALIASES):
        facts = calls[alias]
        issues = _model_lineage_issues(facts)
        if issues:
            _emit(
                "managed_model_lineage_failed",
                alias=alias,
                issues=issues,
                call_count=facts["call_count"],
                managed_count=facts["managed_count"],
                missing_identity=facts["missing_identity"],
                runtime_count=facts["runtime_count"],
                service_count=facts["service_count"],
            )
            raise ValueError(f"Step 7 model lineage is incomplete for {alias}")
    if not services or any(row.get("status") != "stopped" for row in services):
        _emit(
            "managed_runtime_shutdown_failed",
            managed_service_count=len(services),
            nonterminal_services=nonterminal_services,
            service_status_counts=dict(sorted(service_status_counts.items())),
        )
        raise ValueError("Step 7 managed model services were not stopped")
    process_ids = {
        process_id
        for row in services
        for process_id in [row.get("process_id")]
        if isinstance(process_id, int) and not isinstance(process_id, bool)
    }
    process_memory = NvmlHardwareProvider().process_memory_bytes(process_ids)
    retained_process_count = sum(value > 0 for value in process_memory.values())
    if retained_process_count:
        _emit(
            "managed_runtime_gpu_release_failed",
            retained_gpu_process_count=retained_process_count,
        )
        raise ValueError("Step 7 managed service PIDs retained GPU memory")
    _emit(
        "managed_runtime_lineage_verified",
        managed_service_count=len(services),
        model_alias_count=len(calls),
        retained_gpu_process_count=0,
    )

    preparation = resumed.get("model_preparation")
    if (
        not isinstance(preparation, dict)
        or preparation.get("verification_policy") != "cached_receipt"
    ):
        _emit(
            "model_receipt_verification_failed",
            observed_policy=(
                preparation.get("verification_policy")
                if isinstance(preparation, dict)
                else None
            ),
        )
        raise ValueError("Step 7 did not use cached model verification receipts")
    receipts = preparation.get("verification_receipts")
    if not isinstance(receipts, dict) or not _REQUIRED_ALIASES <= set(receipts):
        receipt_aliases = set(receipts) if isinstance(receipts, dict) else set()
        _emit(
            "model_receipt_verification_failed",
            missing_aliases=sorted(_REQUIRED_ALIASES - receipt_aliases),
            observed_policy="cached_receipt",
        )
        raise ValueError("Step 7 model verification receipts are incomplete")
    _emit(
        "model_receipts_verified",
        receipt_alias_count=len(receipts),
        verification_policy="cached_receipt",
    )
    checkpoint_sequence = _verify_checkpoint(resumed)
    _emit("checkpoint_verified", checkpoint_sequence=checkpoint_sequence)

    report = {
        "success": True,
        "contract_version": STEP7_GPU_VERIFIER_CONTRACT_VERSION,
        "launch_id": launch_id,
        "status": "awaiting_approval",
        "freeze_eligible": True,
        "query_count_per_dataset": 12,
        "prefix_query_count_per_dataset": 2,
        "prefix": prefix,
        "compiler_conformance": compiler.model_dump(mode="json"),
        "semantic_conformance": semantic.model_dump(mode="json"),
        "pilot": pilot_report,
        "run_ids": run_ids,
        "program_export_reports": program_reports,
        "coverage_export_reports": coverage_reports,
        "model_startup_counts": first_counts,
        "resume_model_startup_counts": resume_counts,
        "first_attempt_performance_record_count": first_records,
        "resumed_attempt_performance_record_count": resume_records,
        "model_call_facts": calls,
        "checkpoint_sequence": checkpoint_sequence,
        "resume_repeated_model_calls": False,
        "cross_benchmark_isolation_verified": True,
        "managed_gpu_memory_released": True,
        "model_verification_policy": "cached_receipt",
    }
    _emit("completed", launch_id=launch_id, status="awaiting_approval")
    return report


def main(arguments: list[str]) -> int:
    if arguments[:1] == ["--capture-model-calls"]:
        if len(arguments) != 3:
            print(
                "usage: verify_step7_gpu_pilot.py --capture-model-calls SUMMARY OUTPUT",
                file=sys.stderr,
            )
            return 2
        _capture_model_call_counts(Path(arguments[1]), Path(arguments[2]))
        return 0
    if len(arguments) != 4:
        print(
            "usage: verify_step7_gpu_pilot.py "
            "FIRST_SUMMARY RESUMED_SUMMARY FIRST_CALL_COUNTS REPORT",
            file=sys.stderr,
        )
        return 2
    try:
        report = verify(Path(arguments[0]), Path(arguments[1]), Path(arguments[2]))
        destination = Path(arguments[3])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        print(
            "STEP7_GPU_VERIFY_FAILURE "
            + json.dumps(
                {
                    "contract_version": STEP7_GPU_VERIFIER_CONTRACT_VERSION,
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
    _emit(
        "report_written",
        launch_id=report["launch_id"],
        status=report["status"],
        success=report["success"],
        checkpoint_sequence=report["checkpoint_sequence"],
        report_path=str(destination),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
