#!/usr/bin/env python3
"""Verify the combined Step 6D GPU conformance and 50+50 quality pilot."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import bindparam, text

from pmmc.compiler.program_export import verify_program_export
from pmmc.compiler.program_pilot import (
    verify_program_pilot_archive,
    verify_program_pilot_bundle,
)
from pmmc.compiler.step6d_failure import verify_step6d_failure_capsule
from pmmc.evaluation.qbank_export import verify_qbank_coverage_export
from pmmc.experiments.checkpoint import (
    ExperimentCheckpointManifest,
    ExperimentCheckpointService,
)
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_engine
from pmmc.infrastructure.settings import get_settings
from pmmc.pilots.io import read_json, read_jsonl, stable_hash
from pmmc.runtime.hardware import NvmlHardwareProvider
from pmmc.schemas import CompilerProgramPilotAuditManifest
from pmmc.smoke import (
    verify_step6c2_semantic_conformance_payload,
    verify_step6d_compiler_conformance_payload,
)

STEP6D_GPU_VERIFIER_CONTRACT_VERSION = "step6d-gpu-pilot-verifier-v1"
_MODEL_STAGES = (
    "compiler_program:model-stage",
    "program_text:model-stage",
    "program_multimodal:model-stage",
    "doubter:model-stage",
    "qbank_semantic:model-stage",
)
_ALWAYS_STARTED_STAGES = frozenset(_MODEL_STAGES[:3])
_RUNTIME_ALIASES = {
    "compiler-qwen3.5-9b",
    "qwen3-embedding-8b",
    "qwen3-vl-embedding-8b",
}
_JUDGE_ALIAS = "judge-qwen3.5-9b"
_VALID_STATES = {"awaiting_approval", "needs_remediation"}


def _emit(event: str, **facts: object) -> None:
    print(
        "STEP6D_GPU_VERIFY_EVENT "
        + json.dumps(
            {
                "contract_version": STEP6D_GPU_VERIFIER_CONTRACT_VERSION,
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


async def _database_facts(
    compiler_run_ids: list[str],
    semantic_run_ids: list[str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    """Read managed lineage without issuing an empty expanding-IN query."""

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
                "count(*) FILTER (WHERE actual_model_id IS NULL) "
                "AS missing_identity "
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
            if semantic_run_ids:
                semantic_statement = text(
                    "SELECT model_alias, count(*) AS call_count, "
                    "count(*) FILTER (WHERE runtime_id IS NOT NULL AND "
                    "runtime_service_id IS NOT NULL) AS managed_count, "
                    "count(DISTINCT runtime_service_id) FILTER "
                    "(WHERE runtime_service_id IS NOT NULL) AS service_count, "
                    "count(*) FILTER (WHERE actual_model_id IS NULL) "
                    "AS missing_identity "
                    "FROM pmmc_evaluation.model_calls "
                    "WHERE semantic_run_id IN :semantic_run_ids "
                    "AND status='success' GROUP BY model_alias"
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


def _model_stage_counts(
    path: Path,
) -> tuple[dict[str, int], set[str], set[str], int, int]:
    return _model_stage_counts_for_attempt(path, execution_attempt_id=None)


def _model_stage_counts_for_attempt(
    path: Path,
    *,
    execution_attempt_id: str | None,
) -> tuple[dict[str, int], set[str], set[str], int, int]:
    records = read_jsonl(path)
    scoped_records = records
    if execution_attempt_id is not None:
        scoped_records = [
            item
            for item in records
            if isinstance(item.get("metadata"), dict)
            and item["metadata"].get("execution_attempt_id")
            == execution_attempt_id
        ]
        if not scoped_records:
            raise ValueError(
                "Step 6D stage performance has no records for execution attempt"
            )
    model_records = [
        item for item in scoped_records if item.get("stage_id") in _MODEL_STAGES
    ]
    counts = {
        stage: sum(
            (
                0
                if isinstance(item.get("metadata"), dict)
                and item["metadata"].get("state")
                == "not_started_by_hard_gate"
                else int(item.get("model_startup_count", 0))
            )
            for item in model_records
            if item.get("stage_id") == stage and item.get("reused") is False
        )
        for stage in _MODEL_STAGES
    }
    reused = {
        str(item.get("stage_id"))
        for item in model_records
        if item.get("reused") is True
    }
    hard_gate_skipped = {
        str(item.get("stage_id"))
        for item in model_records
        if isinstance(item.get("metadata"), dict)
        and item["metadata"].get("state") == "not_started_by_hard_gate"
    }
    return counts, reused, hard_gate_skipped, len(records), len(scoped_records)


def _validate_model_stage_accounting(
    *,
    startup_counts: dict[str, int],
    reused_stages: set[str],
    hard_gate_skipped_stages: set[str],
    first_pass_restored: bool,
) -> dict[str, str]:
    """Validate fresh starts or equivalent evidence from a restored run."""

    evidence: dict[str, str] = {}
    for stage in _MODEL_STAGES:
        count = startup_counts[stage]
        if count not in {0, 8}:
            raise ValueError(f"Step 6D model stage startup is partial: {stage}")
        if count == 8:
            evidence[stage] = "recorded_eight_replica_start"
            continue
        if stage in hard_gate_skipped_stages:
            if stage in _ALWAYS_STARTED_STAGES:
                raise ValueError(f"Step 6D core model stage was hard-gate skipped: {stage}")
            evidence[stage] = "hard_gate_not_started"
            continue
        if first_pass_restored and stage in reused_stages:
            evidence[stage] = "restored_completed_stage"
            continue
        raise ValueError(f"Step 6D model stage has no startup or restore evidence: {stage}")

    if not first_pass_restored and any(
        startup_counts[stage] != 8 for stage in _ALWAYS_STARTED_STAGES
    ):
        raise ValueError("Step 6D fresh core model stages did not start eight replicas")
    return evidence


def _validate_resume_model_stage_accounting(
    *,
    first_pass_evidence: dict[str, str],
    startup_counts: dict[str, int],
    reused_stages: set[str],
    hard_gate_skipped_stages: set[str],
) -> dict[str, str]:
    """Require the harness's second pass to reuse every first-pass outcome."""

    evidence: dict[str, str] = {}
    for stage in _MODEL_STAGES:
        if startup_counts[stage] != 0:
            raise ValueError(f"Step 6D resume repeated model startup: {stage}")
        if first_pass_evidence[stage] == "hard_gate_not_started":
            if stage in reused_stages:
                evidence[stage] = "reused_hard_gate_terminal"
            elif stage in hard_gate_skipped_stages:
                evidence[stage] = "reconfirmed_hard_gate_not_started"
            else:
                raise ValueError(
                    f"Step 6D resume lost hard-gate model-stage outcome: {stage}"
                )
            continue
        if stage not in reused_stages:
            raise ValueError(f"Step 6D resume did not reuse model stage: {stage}")
        evidence[stage] = "reused_completed_stage"
    return evidence


def _capture_model_call_counts(summary_path: Path, output_path: Path) -> None:
    summary = _load(summary_path)
    datasets = summary.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Step 6D summary has no dataset facts")
    counts: dict[str, dict[str, int]] = {}
    for dataset, raw in sorted(datasets.items()):
        if not isinstance(raw, dict):
            raise ValueError("Step 6D dataset summary is malformed")
        runtime_export = Path(str(raw.get("runtime_export_dir", "")))
        coverage_value = raw.get("coverage_export_dir")
        semantic_count = 0
        if isinstance(coverage_value, str) and coverage_value:
            semantic_count = _jsonl_count(
                Path(coverage_value) / "private/semantic_model_calls.jsonl"
            )
        counts[str(dataset)] = {
            "runtime": _jsonl_count(runtime_export / "runtime/model_calls.jsonl"),
            "semantic": semantic_count,
        }
    output_path.write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_checkpoint(summary: dict[str, Any]) -> int:
    execution = summary.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("Step 6D execution summary is missing")
    checkpoint_path = Path(str(execution.get("latest_checkpoint", "")))
    if not (checkpoint_path / "manifest.json").is_file():
        raise ValueError("Step 6D produced no ready checkpoint")
    manifest = ExperimentCheckpointManifest.from_json(
        (checkpoint_path / "manifest.json").read_text(encoding="utf-8")
    )
    ExperimentCheckpointService(
        get_settings(),
        launch_id=str(summary["launch_id"]),
        root=checkpoint_path.parent.parent,
        identity=dict(manifest.identity),
        retain_count=2,
    ).verify(checkpoint_path)
    return manifest.sequence


def _verify_prefix(value: object, launch_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Step 6D two-query prefix manifest is missing")
    payload = dict(value)
    expected_hash = payload.pop("content_hash", None)
    if (
        payload.get("protocol_version") != "step6d-two-query-prefix-v1"
        or payload.get("launch_id") != launch_id
        or payload.get("paper_result") is not False
        or payload.get("scientific_sample") is not False
        or expected_hash != stable_hash(payload)
    ):
        raise ValueError("Step 6D two-query prefix identity is invalid")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {
        "memlens",
        "memgallery",
    }:
        raise ValueError("Step 6D prefix must cover both benchmarks")
    for dataset, raw in datasets.items():
        if (
            not isinstance(raw, dict)
            or raw.get("query_count") != 2
            or not isinstance(raw.get("query_id_hashes"), list)
            or len(raw["query_id_hashes"]) != 2
        ):
            raise ValueError(f"{dataset} has an invalid two-query prefix")
    return {**payload, "content_hash": expected_hash}


def _verify_call_counts_unchanged(
    summary: dict[str, Any],
    captured_path: Path,
) -> None:
    captured = _load(captured_path)
    datasets = summary.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Step 6D resumed summary has no datasets")
    for dataset, raw in datasets.items():
        if not isinstance(raw, dict):
            raise ValueError("Step 6D resumed dataset summary is malformed")
        expected = captured.get(dataset)
        if not isinstance(expected, dict):
            raise ValueError("Step 6D first-call accounting is incomplete")
        runtime_export = Path(str(raw.get("runtime_export_dir", "")))
        current_runtime = _jsonl_count(
            runtime_export / "runtime/model_calls.jsonl"
        )
        coverage_value = raw.get("coverage_export_dir")
        current_semantic = 0
        if isinstance(coverage_value, str) and coverage_value:
            current_semantic = _jsonl_count(
                Path(coverage_value) / "private/semantic_model_calls.jsonl"
            )
        if (
            current_runtime != int(expected.get("runtime", -1))
            or current_semantic != int(expected.get("semantic", -1))
        ):
            raise ValueError("resume repeated a successful Step 6D model call")


def _verify_full_pilot(
    *,
    summary: dict[str, Any],
    pilot_dir: Path,
    status: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    pilot = verify_program_pilot_bundle(pilot_dir)
    if (
        pilot.schema_version != "0.3.0"
        or pilot.protocol_version != "compiler-program-quality-pilot-v3"
        or pilot.status.value != status
    ):
        raise ValueError("Step 6D pilot protocol or status is invalid")
    audit = CompilerProgramPilotAuditManifest.model_validate(
        read_json(pilot_dir / "private/audit_manifest.json")
    )
    if (
        audit.protocol_version != "compiler-program-audit-v3"
        or audit.candidate_program_audit_count != 160
        or audit.query_coverage_audit_count != 40
        or audit.automatic_integrity_passed
        != (status == "awaiting_approval")
    ):
        raise ValueError("Step 6D audit denominator or hard-gate status is invalid")
    if len(
        read_jsonl(pilot_dir / "private/candidate_program_audit_cards.jsonl")
    ) != 160 or len(
        read_jsonl(pilot_dir / "private/query_coverage_audit_cards.jsonl")
    ) != 40:
        raise ValueError("Step 6D audit card denominator is incomplete")
    transfer = pilot_dir / "transfer"
    archives = sorted(transfer.glob("*.tar.zst"))
    checksums = sorted(transfer.glob("*.sha256"))
    if len(archives) != 1 or len(checksums) != 1:
        raise ValueError("Step 6D transfer deliverables are incomplete")
    archived = verify_program_pilot_archive(archives[0], checksums[0])
    if archived.content_hash != pilot.content_hash:
        raise ValueError("Step 6D transfer archive changed pilot identity")
    return (
        {
            "pilot_id": pilot.pilot_id,
            "automatic_integrity_passed": audit.automatic_integrity_passed,
            "integrity_issue_types": audit.integrity_issue_types,
            "candidate_program_audit_count": 160,
            "query_coverage_audit_count": 40,
            "transfer_archive": str(archives[0]),
            "transfer_checksum": str(checksums[0]),
        },
        list(pilot.source_runtime_export_ids.values()),
        list(pilot.source_coverage_export_ids.values()),
    )


def _verify_remediation_capsule(pilot_dir: Path) -> dict[str, Any]:
    manifest = verify_step6d_failure_capsule(pilot_dir)
    return {
        "pilot_id": manifest["capsule_id"],
        "automatic_integrity_passed": False,
        "integrity_issue_types": read_json(
            pilot_dir / "public/automatic_summary.json"
        ).get("integrity_issue_types", []),
        "candidate_program_audit_count": 0,
        "query_coverage_audit_count": 0,
        "transfer_archive": str(
            next((pilot_dir / "transfer").glob("*.tar.zst"))
        ),
        "transfer_checksum": str(
            next((pilot_dir / "transfer").glob("*.sha256"))
        ),
    }


def verify(
    first_summary_path: Path,
    resumed_summary_path: Path,
    first_call_counts_path: Path,
) -> dict[str, Any]:
    first = _load(first_summary_path)
    resumed = _load(resumed_summary_path)
    if first.get("launch_id") != resumed.get("launch_id"):
        raise ValueError("resume changed the Step 6D launch identity")
    launch_id = str(resumed["launch_id"])
    status = str(resumed.get("status"))
    if status not in _VALID_STATES or first.get("status") != status:
        raise ValueError("Step 6D has an invalid or unstable terminal status")
    if resumed.get("managed_runtime_cleanup_verified") is not True:
        raise ValueError("Step 6D did not verify managed-runtime cleanup")

    execution = resumed.get("execution")
    first_execution = first.get("execution")
    if not isinstance(execution, dict) or not isinstance(first_execution, dict):
        raise ValueError("Step 6D execution facts are missing")
    if execution.get("schedule") != "model-stage-major-v1":
        raise ValueError("Step 6D did not use stage-major execution")
    if not execution.get("restored_checkpoint"):
        raise ValueError("Step 6D resume did not restore a checkpoint")
    compiler = verify_step6d_compiler_conformance_payload(
        execution.get("step6d_compiler_conformance"),
        expected_replica_count=8,
    )
    first_compiler = verify_step6d_compiler_conformance_payload(
        first_execution.get("step6d_compiler_conformance"),
        expected_replica_count=8,
    )
    if compiler.content_hash != first_compiler.content_hash:
        raise ValueError("resume changed compiler conformance identity")
    prefix = _verify_prefix(execution.get("step6d_prefix_manifest"), launch_id)

    pilot_summary = resumed.get("compiler_program_pilot")
    if (
        not isinstance(pilot_summary, dict)
        or pilot_summary.get("status") != status
    ):
        raise ValueError("Step 6D pilot summary is missing or inconsistent")
    pilot_dir = Path(str(pilot_summary.get("pilot_dir", "")))
    capsule_manifest_path = pilot_dir / "manifest.json"
    capsule_protocol = (
        _load(capsule_manifest_path).get("protocol_version")
        if capsule_manifest_path.is_file()
        else None
    )
    remediation_capsule = capsule_protocol == "step6d-failure-capsule-v1"
    if remediation_capsule:
        if status != "needs_remediation":
            raise ValueError("Step 6D remediation capsule has a freezeable status")
        pilot_report = _verify_remediation_capsule(pilot_dir)
    else:
        pilot_report, _runtime_exports, _coverage_exports = _verify_full_pilot(
            summary=resumed,
            pilot_dir=pilot_dir,
            status=status,
        )

    datasets = resumed.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {
        "memlens",
        "memgallery",
    }:
        raise ValueError("Step 6D must cover both benchmarks")
    compiler_run_ids: list[str] = []
    semantic_run_ids: list[str] = []
    export_reports: dict[str, dict[str, Any]] = {}
    coverage_reports: dict[str, dict[str, Any]] = {}
    blocked_datasets: list[str] = []
    for dataset, raw in sorted(datasets.items()):
        if not isinstance(raw, dict) or raw.get("selected_count") != 50:
            raise ValueError(f"{dataset} must contain 50 development queries")
        compiler_run_ids.append(str(raw["compiler_run_id"]))
        if raw.get("r0_gate_passed") is False:
            blocked_datasets.append(dataset)
            verification = raw.get("program_export_verification")
            if (
                not isinstance(verification, dict)
                or verification.get("diagnostic_export") is not True
                or verification.get("r0_hard_gate_passed") is not False
            ):
                raise ValueError(f"{dataset} has no verified R0 diagnostic export")
            continue
        program_report = verify_program_export(
            Path(str(raw.get("runtime_export_dir", "")))
        )
        coverage_report = verify_qbank_coverage_export(
            Path(str(raw.get("coverage_export_dir", "")))
        )
        if (
            program_report.get("protocol_version") != "pmmc-questioner-v3.3"
            or program_report.get("r0_hard_gate_passed") is not True
            or program_report.get("projection_count") != 4
            or program_report.get("active_round") != 2
            or program_report.get("max_round") != 3
            or program_report.get("qbank_collection_count") != 4
            or len(set(program_report.get("memory_collection_ids", []))) != 2
        ):
            raise ValueError(f"{dataset} has an invalid verified-QBank export")
        if (
            coverage_report.get("semantic_match_present") is not True
            or coverage_report.get("semantic_run_id")
            != raw.get("semantic_run_id")
            or int(
                coverage_report.get("broad_session_round_match_count", -1)
            )
            != 0
        ):
            raise ValueError(f"{dataset} semantic evaluation is invalid")
        semantic_run_ids.append(str(raw["semantic_run_id"]))
        export_reports[dataset] = program_report
        coverage_reports[dataset] = coverage_report

    manifest = (
        verify_program_pilot_bundle(pilot_dir)
        if not remediation_capsule
        else None
    )
    if manifest is not None:
        memgallery = next(
            item
            for item in manifest.samples
            if item.dataset.value == "memgallery"
        )
        if (
            len(memgallery.scenario_counts) != 4
            or sorted(memgallery.scenario_counts.values())
            != [12, 12, 13, 13]
        ):
            raise ValueError("MemGallery does not use 13/13/12/12 sampling")
        memgallery_scenarios = dict(sorted(memgallery.scenario_counts.items()))
    else:
        capsule = _load(pilot_dir / "manifest.json")
        sample_rows = capsule.get("samples")
        if not isinstance(sample_rows, list):
            raise ValueError("Step 6D remediation capsule has no sample identity")
        memgallery_row = next(
            (
                item
                for item in sample_rows
                if isinstance(item, dict) and item.get("dataset") == "memgallery"
            ),
            None,
        )
        if not isinstance(memgallery_row, dict):
            raise ValueError("Step 6D remediation capsule lacks MemGallery")
        scenario_counts = memgallery_row.get("scenario_counts")
        if (
            not isinstance(scenario_counts, dict)
            or len(scenario_counts) != 4
            or sorted(int(value) for value in scenario_counts.values())
            != [12, 12, 13, 13]
        ):
            raise ValueError("MemGallery remediation sample is unbalanced")
        memgallery_scenarios = {
            str(key): int(value)
            for key, value in sorted(scenario_counts.items())
        }
    _emit(
        "pilot_bundle_verified",
        blocked_dataset_count=len(blocked_datasets),
        remediation_capsule=remediation_capsule,
        status=status,
    )

    _verify_call_counts_unchanged(resumed, first_call_counts_path)
    performance_path = Path(str(execution.get("stage_performance_path", "")))
    first_attempt_id = first_execution.get("execution_attempt_id")
    resumed_attempt_id = execution.get("execution_attempt_id")
    if (
        not isinstance(first_attempt_id, str)
        or not first_attempt_id
        or not isinstance(resumed_attempt_id, str)
        or not resumed_attempt_id
        or first_attempt_id == resumed_attempt_id
    ):
        raise ValueError("Step 6D execution-attempt identity is missing or reused")
    (
        startup_counts,
        reused_stages,
        hard_gate_skipped_stages,
        performance_record_count,
        first_attempt_record_count,
    ) = _model_stage_counts_for_attempt(
        performance_path,
        execution_attempt_id=first_attempt_id,
    )
    (
        resume_startup_counts,
        resume_reused_stages,
        resume_hard_gate_skipped_stages,
        resume_performance_record_count,
        resumed_attempt_record_count,
    ) = _model_stage_counts_for_attempt(
        performance_path,
        execution_attempt_id=resumed_attempt_id,
    )
    if resume_performance_record_count != performance_record_count:
        raise ValueError("Step 6D performance log changed during verification")
    first_pass_restored = bool(first_execution.get("restored_checkpoint"))
    _emit(
        "model_stage_accounting",
        execution_attempt_id=first_attempt_id,
        first_pass_restored=first_pass_restored,
        hard_gate_skipped_stages=sorted(hard_gate_skipped_stages),
        historical_record_count=(
            performance_record_count
            - first_attempt_record_count
            - resumed_attempt_record_count
        ),
        performance_record_count=performance_record_count,
        reused_stages=sorted(reused_stages),
        scoped_record_count=first_attempt_record_count,
        startup_counts=startup_counts,
    )
    model_stage_evidence = _validate_model_stage_accounting(
        startup_counts=startup_counts,
        reused_stages=reused_stages,
        hard_gate_skipped_stages=hard_gate_skipped_stages,
        first_pass_restored=first_pass_restored,
    )
    _emit(
        "resume_model_stage_accounting",
        execution_attempt_id=resumed_attempt_id,
        hard_gate_skipped_stages=sorted(resume_hard_gate_skipped_stages),
        reused_stages=sorted(resume_reused_stages),
        scoped_record_count=resumed_attempt_record_count,
        startup_counts=resume_startup_counts,
    )
    resume_model_stage_evidence = _validate_resume_model_stage_accounting(
        first_pass_evidence=model_stage_evidence,
        startup_counts=resume_startup_counts,
        reused_stages=resume_reused_stages,
        hard_gate_skipped_stages=resume_hard_gate_skipped_stages,
    )

    semantic = None
    if semantic_run_ids:
        semantic = verify_step6c2_semantic_conformance_payload(
            execution.get("step6d_semantic_conformance"),
            expected_replica_count=8,
        )
        first_semantic = verify_step6c2_semantic_conformance_payload(
            first_execution.get("step6d_semantic_conformance"),
            expected_replica_count=8,
        )
        if semantic.content_hash != first_semantic.content_hash:
            raise ValueError("resume changed semantic conformance identity")
        if (
            model_stage_evidence["qbank_semantic:model-stage"]
            == "hard_gate_not_started"
        ):
            raise ValueError("Step 6D semantic stage was not executed")

    services, calls = asyncio.run(
        _database_facts(compiler_run_ids, semantic_run_ids)
    )
    service_status_counts: dict[str, int] = {}
    for service in services:
        service_status = str(service.get("status", "unknown"))
        service_status_counts[service_status] = (
            service_status_counts.get(service_status, 0) + 1
        )
    _emit(
        "managed_model_lineage",
        aliases={
            alias: {
                "call_count": facts["call_count"],
                "managed_count": facts["managed_count"],
                "missing_identity": facts["missing_identity"],
                "service_count": facts["service_count"],
            }
            for alias, facts in sorted(calls.items())
        },
        service_status_counts=dict(sorted(service_status_counts.items())),
    )
    required_aliases = set(_RUNTIME_ALIASES)
    if semantic_run_ids:
        required_aliases.add(_JUDGE_ALIAS)
    if not required_aliases <= set(calls):
        raise ValueError("Step 6D managed model lineage is incomplete")
    for alias in required_aliases:
        facts = calls[alias]
        if (
            facts["call_count"] <= 0
            or facts["managed_count"] != facts["call_count"]
            or facts["missing_identity"]
        ):
            raise ValueError(f"Step 6D model identity is incomplete for {alias}")
    if calls["compiler-qwen3.5-9b"]["service_count"] != 8:
        raise ValueError("Step 6D compiler calls did not cover eight services")
    if semantic is not None and semantic.request_count != 8:
        raise ValueError("Step 6D semantic conformance missed a Judge replica")
    if not services or any(item.get("status") != "stopped" for item in services):
        raise ValueError("Step 6D managed services were not stopped")
    process_ids = {
        process_id
        for item in services
        for process_id in [item.get("process_id")]
        if isinstance(process_id, int) and not isinstance(process_id, bool)
    }
    if any(NvmlHardwareProvider().process_memory_bytes(process_ids).values()):
        raise ValueError("Step 6D managed PIDs retained GPU memory")

    preparation = resumed.get("model_preparation")
    if (
        not isinstance(preparation, dict)
        or preparation.get("verification_policy") != "cached_receipt"
    ):
        raise ValueError("Step 6D did not use cached model verification receipts")
    receipts = preparation.get("verification_receipts")
    if not isinstance(receipts, dict) or not required_aliases <= set(receipts):
        raise ValueError("Step 6D model verification receipts are incomplete")
    _emit(
        "model_receipts_verified",
        alias_count=len(required_aliases),
        verification_policy="cached_receipt",
    )

    checkpoint_sequence = _verify_checkpoint(resumed)
    _emit("checkpoint_verified", sequence=checkpoint_sequence)
    report = {
        "success": True,
        "contract_version": STEP6D_GPU_VERIFIER_CONTRACT_VERSION,
        "launch_id": launch_id,
        "status": status,
        "freeze_eligible": (
            status == "awaiting_approval" and not remediation_capsule
        ),
        "scientific_pipeline_complete": not remediation_capsule,
        "blocked_datasets": blocked_datasets,
        "query_count_per_dataset": 50,
        "memgallery_scenarios": memgallery_scenarios,
        "compiler_conformance": compiler.model_dump(mode="json"),
        "semantic_conformance": (
            semantic.model_dump(mode="json") if semantic is not None else None
        ),
        "prefix_manifest": prefix,
        "pilot": pilot_report,
        "model_startup_counts": startup_counts,
        "model_stage_execution_evidence": model_stage_evidence,
        "resume_model_startup_counts": resume_startup_counts,
        "resume_model_stage_execution_evidence": resume_model_stage_evidence,
        "performance_record_count": performance_record_count,
        "first_attempt_performance_record_count": first_attempt_record_count,
        "resumed_attempt_performance_record_count": resumed_attempt_record_count,
        "hard_gate_skipped_model_stages": sorted(hard_gate_skipped_stages),
        "model_call_facts": calls,
        "checkpoint_sequence": checkpoint_sequence,
        "program_export_reports": export_reports,
        "coverage_export_reports": coverage_reports,
        "resume_repeated_model_calls": False,
        "cross_benchmark_isolation_verified": True,
        "managed_gpu_memory_released": True,
        "model_verification_policy": "cached_receipt",
    }
    _emit(
        "completed",
        blocked_dataset_count=len(blocked_datasets),
        freeze_eligible=report["freeze_eligible"],
        status=status,
    )
    return report


def main(arguments: list[str]) -> int:
    if arguments[:1] == ["--capture-model-calls"]:
        if len(arguments) != 3:
            print(
                "usage: verify_step6d_gpu_pilot.py "
                "--capture-model-calls SUMMARY OUTPUT",
                file=sys.stderr,
            )
            return 2
        _capture_model_call_counts(Path(arguments[1]), Path(arguments[2]))
        return 0
    if len(arguments) != 4:
        print(
            "usage: verify_step6d_gpu_pilot.py "
            "FIRST_SUMMARY RESUMED_SUMMARY FIRST_CALL_COUNTS REPORT",
            file=sys.stderr,
        )
        return 2
    try:
        report = verify(
            Path(arguments[0]),
            Path(arguments[1]),
            Path(arguments[2]),
        )
        destination = Path(arguments[3])
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(
            "STEP6D_GPU_VERIFY_FAILURE "
            + json.dumps(
                {
                    "contract_version": STEP6D_GPU_VERIFIER_CONTRACT_VERSION,
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
