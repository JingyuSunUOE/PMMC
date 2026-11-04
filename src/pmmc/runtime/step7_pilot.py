"""Portable Step 7 quality-pilot bundles and explicit freeze decisions."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

import zstandard

from pmmc.compiler.program_pilot import (
    _annotate_semantic_matches,
    _annotate_thumbnails,
    _archive_model_artifacts,
    _coverage_payloads,
    _portable_scan_issue,
    _program_candidate_facts,
    _project_model_artifact_value,
    _redact_nonportable_string,
    _runtime_source_identity,
    _semantic_source_identity,
    _step6d_model_artifact_selection,
    verify_program_freeze,
    verify_verified_qbank_freeze,
)
from pmmc.evaluation.export import verify_evaluation_export
from pmmc.evaluation.qbank_export import verify_qbank_coverage_export
from pmmc.infrastructure.export_verification import verify_experiment_export
from pmmc.pilots.io import (
    canonical_json_bytes,
    file_sha256,
    read_json,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)
from pmmc.runtime.query_executor_config import routing_policy_hash_from_protocol
from pmmc.schemas import (
    DatasetName,
    PilotDecisionValue,
    QuestionerFreezeManifest,
    QuestionerPilotSample,
    Step7BFreezeManifest,
    Step7PilotArtifactRef,
    Step7PilotDecision,
    Step7PilotManifest,
    Step7PilotStatus,
)
from pmmc.schemas.base import (
    ANSWER_OUTPUT_SCHEMA_VERSION,
    QBANK_SEMANTIC_SCHEMA_VERSION,
    QUERY_EXECUTOR_SCHEMA_VERSION,
    STEP7_PILOT_SCHEMA_VERSION,
    STEP7B_PILOT_SCHEMA_VERSION,
    VERIFIED_QBANK_SCHEMA_VERSION,
    utc_now,
)
from pmmc.schemas.doubter import VerifiedQBankFreezeManifest
from pmmc.schemas.program_pilot import CompilerProgramFreezeManifest
from pmmc.schemas.query_executor import QueryExecutorFreezeManifest

_PROTOCOL = "pmmc-step7-executor-pilot-v1"
_STEP7B_PROTOCOL = "pmmc-step7b-executor-pilot-v1"
_PORTABLE_FACT_PROJECTION_VERSION = "portable-step7-fact-projection-v1"
_QUERY_AUDIT_CONTRACT_VERSION = "step7-query-audit-contract-v2"
_MATCH_THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80)
_GAP_THRESHOLD_GRID = (0.00, 0.05, 0.10)
_REUSABLE_PILOT_METADATA_FIELDS = (
    "active_round",
    "fact_projection_version",
    "gap_threshold",
    "match_threshold",
    "maximum_round",
    "paper_result",
    "question_budget",
    "underpowered",
)
_PRIVATE_COMPILER_FILES = (
    "model_calls.jsonl",
    "compiler_runs.jsonl",
    "compiler_units.jsonl",
    "compiler_unit_memories.jsonl",
    "compiler_unit_stage_results.jsonl",
    "compiler_unit_links.jsonl",
    "questioner_generations.jsonl",
    "questioner_opportunity_sets.jsonl",
    "questioner_opportunities.jsonl",
    "questioner_opportunity_evidence_refs.jsonl",
    "questioner_verification_batches.jsonl",
    "qbank_questions.jsonl",
    "qbank_paraphrases.jsonl",
    "qbank_evidence_refs.jsonl",
    "questioner_candidate_gate_results.jsonl",
    "questioner_candidate_verifications.jsonl",
    "questioner_candidate_supersessions.jsonl",
    "questioner_answer_claims.jsonl",
    "qbank_candidate_families.jsonl",
    "qbank_candidate_family_members.jsonl",
    "qbank_duplicate_links.jsonl",
    "qbank_index_manifests.jsonl",
    "compiler_stage_jobs.jsonl",
    "planner_generation_batches.jsonl",
    "planner_requirement_contracts.jsonl",
    "planner_candidate_drafts.jsonl",
    "planner_candidate_outcomes.jsonl",
    "planner_batch_programs.jsonl",
    "program_validation_results.jsonl",
    "program_validation_issues.jsonl",
    "memory_operator_index_manifests.jsonl",
    "structured_output_item_diagnostics.jsonl",
    "memory_programs.jsonl",
    "program_steps.jsonl",
    "program_expected_evidence_refs.jsonl",
    "program_executions.jsonl",
    "program_execution_steps.jsonl",
    "program_execution_evidence_refs.jsonl",
    "program_dry_run_assessments.jsonl",
    "program_dry_run_evidence_matches.jsonl",
    "doubter_feedback.jsonl",
    "r0_input_manifests.jsonl",
    "candidate_stage_jobs.jsonl",
    "doubter_assessments.jsonl",
    "doubter_assessment_dimensions.jsonl",
    "doubter_directives.jsonl",
    "planner_revision_lineage.jsonl",
    "planner_revision_patches.jsonl",
    "qbank_round_projection_manifests.jsonl",
    "qbank_round_projection_members.jsonl",
    "routing_projection_manifests.jsonl",
    "routing_projection_members.jsonl",
    "stage_component_manifests.jsonl",
    "qbank_decisions.jsonl",
)


def _step7_pilot_reuse_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project the immutable scientific contract of a portable pilot bundle."""

    samples = [
        item.model_dump(mode="json")
        if isinstance(item, QuestionerPilotSample)
        else dict(item)
        for item in payload.get("samples", [])
    ]
    samples.sort(key=lambda item: (str(item.get("dataset", "")), item["sample_id"]))
    metadata = dict(payload.get("metadata", {}))
    return {
        "automatic_integrity_passed": payload.get("automatic_integrity_passed"),
        "compiler_run_ids": sorted(payload.get("compiler_run_ids", [])),
        "integrity_issue_types": sorted(payload.get("integrity_issue_types", [])),
        "launch_id": payload.get("launch_id"),
        "metadata": {
            key: metadata.get(key) for key in _REUSABLE_PILOT_METADATA_FIELDS
        },
        "partition_id": payload.get("partition_id"),
        "pilot_id": payload.get("pilot_id"),
        "prefix_query_count": payload.get("prefix_query_count"),
        "protocol_version": payload.get("protocol_version"),
        "query_count": payload.get("query_count"),
        "query_executor_run_ids": sorted(payload.get("query_executor_run_ids", [])),
        "samples": samples,
        "schema_version": payload.get("schema_version"),
        "source_compiler_export_ids": dict(
            sorted(dict(payload.get("source_compiler_export_ids", {})).items())
        ),
        "source_coverage_export_ids": dict(
            sorted(dict(payload.get("source_coverage_export_ids", {})).items())
        ),
        "source_evaluation_export_ids": dict(
            sorted(dict(payload.get("source_evaluation_export_ids", {})).items())
        ),
        "source_query_export_ids": dict(
            sorted(dict(payload.get("source_query_export_ids", {})).items())
        ),
        "status": payload.get("status"),
    }
_PRIVATE_QUERY_FILES = (
    "model_calls.jsonl",
    "pmmc_fallback_model_calls.jsonl",
    "reused_query_embedding_model_calls.jsonl",
    "reused_query_embedding_source_runs.jsonl",
    "qbank_match_batches.jsonl",
    "qbank_match_channel_candidates.jsonl",
    "qbank_match_fused_candidates.jsonl",
    "qbank_lexical_term_facts.jsonl",
    "qbank_reranker_assessments.jsonl",
    "query_route_decisions.jsonl",
    "query_program_bindings.jsonl",
    "program_executions.jsonl",
    "program_execution_steps.jsonl",
    "program_execution_evidence_refs.jsonl",
    "query_evidence_sufficiency_assessments.jsonl",
    "query_evidence_relevance_assessments.jsonl",
    "query_evidence_relevance_items.jsonl",
    "query_execution_reuse_links.jsonl",
    "query_fallback_decisions.jsonl",
    "query_answer_selections.jsonl",
    "structured_answer_outputs.jsonl",
    "answer_format_diagnostics.jsonl",
    "answer_reuse_links.jsonl",
    "final_answers.jsonl",
    "pmmc_fallback_run_links.jsonl",
    "pmmc_fallback_query_links.jsonl",
    "pmmc_fallback_answers.jsonl",
    "pmmc_fallback_answer_evidence_ids.jsonl",
    "pmmc_fallback_answer_image_ids.jsonl",
)
_EVALUATION_PUBLIC_JSONL_FILES = (
    "method_summary.jsonl",
    "category_summary.jsonl",
    "judge_method_summary.jsonl",
    "judge_category_summary.jsonl",
    "primary_bad_case_method_summary.jsonl",
    "primary_bad_case_category_summary.jsonl",
    "production_cost_summary.jsonl",
    "evaluation_cost_summary.jsonl",
    "combined_research_cost_summary.jsonl",
)
_SECRET_PATTERNS = (
    re.compile(rb"(?:HF_TOKEN|OPENAI_API_KEY|AWS_SECRET_ACCESS_KEY)="),
    re.compile(rb"X-Amz-(?:Signature|Credential)=", re.IGNORECASE),
    re.compile(rb"PMMC_GOLD_SENTINEL|GOLD_ANSWER_SENTINEL"),
)
_BundleResult = TypeVar("_BundleResult")


class Step7BundleBuildError(RuntimeError):
    """Expose a safe bundle substage without leaking private review content."""

    def __init__(
        self,
        *,
        dataset: DatasetName | None,
        substage: str,
        cause: BaseException,
    ) -> None:
        scope = dataset.value if dataset is not None else "global"
        self.failure_stage = f"step7_bundle:{scope}:{substage}"
        safe_check_code = getattr(cause, "check_code", None)
        suffix = (
            str(safe_check_code)
            if isinstance(safe_check_code, str) and safe_check_code
            else type(cause).__name__
        )
        self.error_code = f"bundle_{substage}_{suffix}"
        super().__init__("Step 7 bundle construction failed")


class Step7BundleVerificationError(ValueError):
    """Identify one safe bundle check without exposing private artifact content."""

    def __init__(self, check_code: str) -> None:
        if re.fullmatch(r"[a-z0-9_]+", check_code) is None:
            raise ValueError("Step 7 bundle check code is unsafe")
        self.check_code = check_code
        super().__init__("Step 7 bundle verification failed")


class _BundleVerificationCheck:
    """Emit bounded check-level diagnostics and normalize private exceptions."""

    def __init__(self, check_code: str) -> None:
        self.check_code = check_code
        self.started_at = 0.0

    def __enter__(self) -> None:
        self.started_at = time.monotonic()
        if self.check_code in {"artifact_integrity", "compressed_fact_archives"}:
            _event(
                "verification_check_started",
                check_code=self.check_code,
            )

    def __exit__(
        self,
        _error_type: object,
        error: BaseException | None,
        _traceback: object,
    ) -> None:
        elapsed_ms = max(0, round((time.monotonic() - self.started_at) * 1000))
        if error is None:
            _event(
                "verification_check_completed",
                check_code=self.check_code,
                elapsed_ms=elapsed_ms,
            )
            return
        if isinstance(error, Step7BundleVerificationError):
            _event(
                "verification_check_failed",
                check_code=error.check_code,
                elapsed_ms=elapsed_ms,
                error_type=type(error).__name__,
            )
            return
        _event(
            "verification_check_failed",
            check_code=self.check_code,
            elapsed_ms=elapsed_ms,
            error_type=type(error).__name__,
        )
        raise Step7BundleVerificationError(self.check_code) from error


def _event(phase: str, **fields: object) -> None:
    print(
        "STEP7_BUNDLE "
        + json.dumps(
            {"phase": phase, "stage": "step7_bundle", **fields},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _bundle_substage(
    *,
    dataset: DatasetName,
    substage: str,
    operation: Callable[[], _BundleResult],
) -> _BundleResult:
    try:
        return operation()
    except Exception as exc:
        _event(
            "dataset_failed",
            dataset=dataset.value,
            error_type=type(exc).__name__,
            substage=substage,
        )
        raise Step7BundleBuildError(
            dataset=dataset,
            substage=substage,
            cause=exc,
        ) from exc


def _bundle_global_substage(
    *,
    substage: str,
    operation: Callable[[], _BundleResult],
) -> _BundleResult:
    try:
        return operation()
    except Exception as exc:
        safe_check_code = getattr(exc, "check_code", None)
        _event(
            "bundle_failed",
            **(
                {"check_code": safe_check_code}
                if isinstance(safe_check_code, str) and safe_check_code
                else {}
            ),
            error_type=type(exc).__name__,
            substage=substage,
        )
        raise Step7BundleBuildError(
            dataset=None,
            substage=substage,
            cause=exc,
        ) from exc


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def _write_zstd_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    path.write_bytes(
        zstandard.ZstdCompressor(
            level=9,
            threads=0,
            write_checksum=True,
            write_content_size=True,
        ).compress(payload)
    )


def _project_portable_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Redact machine-local locators while preserving scientific fact fields."""

    projected_rows: list[dict[str, Any]] = []
    redaction_count = 0
    for row in rows:
        projected, item_redactions = _project_model_artifact_value(dict(row))
        if not isinstance(projected, dict):  # pragma: no cover - mapping input invariant.
            raise TypeError("portable Step 7 fact projection is not an object")
        projected_rows.append(projected)
        redaction_count += item_redactions
    return projected_rows, redaction_count


def _project_portable_object(value: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    projected, redaction_count = _project_model_artifact_value(dict(value))
    if not isinstance(projected, dict):  # pragma: no cover - mapping input invariant.
        raise TypeError("portable Step 7 object projection is not an object")
    return projected, redaction_count


def _read_zstd_jsonl(path: Path) -> list[dict[str, Any]]:
    payload = zstandard.ZstdDecompressor().decompress(path.read_bytes())
    rows: list[dict[str, Any]] = []
    for ordinal, line in enumerate(payload.splitlines(), start=1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Step 7 fact line is not an object: {ordinal}")
        rows.append(value)
    return rows


def _rows(root: Path, names: Sequence[str], dataset: DatasetName) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in names:
        for row in _read_optional_jsonl(root / name):
            result.append({"dataset": dataset.value, "fact_type": name[:-6], **row})
    return result


def _evaluation_public_payload(export_dir: Path) -> dict[str, Any]:
    public = export_dir / "public"
    payload: dict[str, Any] = {
        "summary": read_json(public / "evaluation_summary.json")
    }
    for file_name in _EVALUATION_PUBLIC_JSONL_FILES:
        path = public / file_name
        if not path.is_file():
            raise ValueError(f"evaluation public summary is missing: {file_name}")
        payload[file_name.removesuffix(".jsonl")] = read_jsonl(path)
    return payload


def _evaluation_summary_complete(
    summary: Mapping[str, Any], *, expected_query_count: int, step7b: bool,
) -> bool:
    if not step7b:
        return (
            summary.get("selected_count") == expected_query_count
            and summary.get("methods") == ["pmmc"]
        )
    counts = summary.get("answer_variant_counts")
    missing = summary.get("answer_variant_missing_counts")
    methods = summary.get("methods")
    if (
        summary.get("query_count") != expected_query_count
        or not isinstance(counts, dict) or not isinstance(missing, dict)
        or not isinstance(methods, list)
        or not set(methods) <= {"pmmc", "native_multimodal_rag"}
    ):
        return False
    expected_variants = {"pmmc_r0", "pmmc_r1", "pmmc_r2", "pmmc_r3",
                         "native_mm_counterfactual", "active_final_provisional",
                         "no_fallback_counterfactual"}
    return (
        set(missing) == expected_variants and set(counts) <= expected_variants
        and all(isinstance(value, int) and value >= 0 for value in counts.values())
        and all(
            isinstance(value, int) and value >= 0
            and counts.get(variant, 0) + value == expected_query_count
            for variant, value in missing.items()
        )
        and sum(counts.values()) == summary.get("selected_count")
    )


def _model_call_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, dict[str, float | int]] = {}
    for row in rows:
        stage = str(row.get("stage") or "unknown")
        bucket = by_stage.setdefault(
            stage,
            {
                "model_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": 0.0,
            },
        )
        bucket["model_call_count"] = int(bucket["model_call_count"]) + 1
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            bucket[field] = int(bucket[field]) + int(row.get(field) or 0)
        for field in ("cost_usd", "latency_ms"):
            bucket[field] = float(bucket[field]) + float(row.get(field) or 0.0)
    totals = {
        field: sum(float(value[field]) for value in by_stage.values())
        for field in ("cost_usd", "latency_ms")
    }
    totals.update(
        {
            field: sum(int(value[field]) for value in by_stage.values())
            for field in (
                "model_call_count",
                "input_tokens",
                "output_tokens",
                "total_tokens",
            )
        }
    )
    return {"totals": totals, "by_stage": dict(sorted(by_stage.items()))}


def _amortized_curves(
    *, write_summary: Mapping[str, Any], query_summary: Mapping[str, Any], query_count: int
) -> list[dict[str, float | int]]:
    write = dict(write_summary["totals"])
    query = dict(query_summary["totals"])
    per_query_cost = float(query["cost_usd"]) / query_count
    per_query_tokens = float(query["total_tokens"]) / query_count
    return [
        {
            "query_volume": volume,
            "amortized_cost_usd_per_query": (
                float(write["cost_usd"]) / volume + per_query_cost
            ),
            "amortized_tokens_per_query": (
                float(write["total_tokens"]) / volume + per_query_tokens
            ),
        }
        for volume in (1, 10, 100, 1_000, 10_000)
    ]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _compiler_round_summaries(compiler: Path) -> dict[str, list[dict[str, Any]]]:
    manifests = _read_optional_jsonl(
        compiler / "qbank_round_projection_manifests.jsonl"
    )
    members = _read_optional_jsonl(compiler / "qbank_round_projection_members.jsonl")
    assessments = _read_optional_jsonl(compiler / "doubter_assessments.jsonl")
    programs = _read_optional_jsonl(compiler / "memory_programs.jsonl")
    steps = _read_optional_jsonl(compiler / "program_steps.jsonl")
    revisions = _read_optional_jsonl(compiler / "planner_revision_lineage.jsonl")

    members_by_projection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in members:
        members_by_projection[str(row.get("projection_id"))].append(row)
    projection_rows: list[dict[str, Any]] = []
    for manifest in sorted(manifests, key=lambda row: int(row.get("round_id", -1))):
        round_id = int(manifest.get("round_id", -1))
        projection_members = members_by_projection.get(
            str(manifest.get("projection_id")), []
        )
        outcome_counts = Counter(str(row.get("outcome")) for row in projection_members)
        route_class_counts = Counter(
            str(row.get("route_class"))
            for row in projection_members
            if row.get("outcome") in {"accepted", "carried_forward"}
        )
        rescued = sum(
            row.get("outcome") in {"accepted", "carried_forward"}
            and row.get("passed_at_round") == round_id
            for row in projection_members
        )
        projection_rows.append(
            {
                "round_id": round_id,
                "member_count": len(projection_members),
                "accepted_count": int(manifest.get("accepted_count", 0)),
                "rejected_count": int(manifest.get("rejected_count", 0)),
                "pending_count": int(manifest.get("pending_count", 0)),
                "outcome_counts": dict(sorted(outcome_counts.items())),
                "accepted_route_class_counts": dict(
                    sorted(route_class_counts.items())
                ),
                "rescued_at_round_count": rescued,
            }
        )

    doubter_rows: list[dict[str, Any]] = []
    for round_id in range(4):
        round_assessments = [
            row for row in assessments if int(row.get("round_id", -1)) == round_id
        ]
        action_counts = Counter(str(row.get("action")) for row in round_assessments)
        native_count = sum(
            row.get("provider_schema_conformant") is True
            for row in round_assessments
        )
        doubter_rows.append(
            {
                "round_id": round_id,
                "assessment_count": len(round_assessments),
                "provider_schema_conformant_count": native_count,
                "provider_schema_conformant_rate": _ratio(
                    native_count, len(round_assessments)
                ),
                "action_counts": dict(sorted(action_counts.items())),
            }
        )

    steps_by_plan: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        steps_by_plan[str(row.get("plan_id"))].append(row)
    operator_rows: list[dict[str, Any]] = []
    for round_id in range(4):
        round_programs = [
            row for row in programs if int(row.get("round_id", -1)) == round_id
        ]
        sequences: Counter[str] = Counter()
        operators: Counter[str] = Counter()
        step_counts: list[int] = []
        for program in round_programs:
            plan_steps = sorted(
                steps_by_plan.get(str(program.get("plan_id")), []),
                key=lambda row: int(row.get("step_id", -1)),
            )
            sequence = [str(row.get("op_name")) for row in plan_steps]
            operators.update(sequence)
            sequences[" -> ".join(sequence)] += 1
            step_counts.append(len(sequence))
        operator_rows.append(
            {
                "round_id": round_id,
                "program_count": len(round_programs),
                "revision_count": sum(
                    int(row.get("target_round_id", -1)) == round_id
                    for row in revisions
                ),
                "mean_step_count": (
                    sum(step_counts) / len(step_counts) if step_counts else None
                ),
                "operator_counts": dict(sorted(operators.items())),
                "unique_sequence_count": len(sequences),
                "sequence_counts": dict(sorted(sequences.items())),
            }
        )
    return {
        "projection_rounds": projection_rows,
        "doubter_rounds": doubter_rows,
        "operator_rounds": operator_rows,
    }


def _executor_round_summaries(query: Path) -> list[dict[str, Any]]:
    bindings = _read_optional_jsonl(query / "query_program_bindings.jsonl")
    assessments = _read_optional_jsonl(
        query / "query_evidence_sufficiency_assessments.jsonl"
    )
    assessment_by_binding = {
        str(row.get("query_program_binding_id")): row for row in assessments
    }
    result: list[dict[str, Any]] = []
    for round_id in range(4):
        round_bindings = [
            row for row in bindings if int(row.get("round_id", -1)) == round_id
        ]
        round_assessments = [
            assessment_by_binding[str(row.get("query_program_binding_id"))]
            for row in round_bindings
            if str(row.get("query_program_binding_id")) in assessment_by_binding
        ]
        technical = sum(
            row.get("technical_success") is True for row in round_assessments
        )
        nonempty = sum(
            row.get("nonempty_evidence") is True for row in round_assessments
        )
        sufficient = sum(row.get("sufficient") is True for row in round_assessments)
        raw_image = sum(
            row.get("raw_image_reached") is True for row in round_assessments
        )
        result.append(
            {
                "round_id": round_id,
                "binding_count": len(round_bindings),
                "assessment_count": len(round_assessments),
                "technical_success_count": technical,
                "technical_success_rate": _ratio(technical, len(round_assessments)),
                "nonempty_evidence_count": nonempty,
                "program_execution_success_rate": _ratio(
                    sum(
                        row.get("technical_success") is True
                        and row.get("nonempty_evidence") is True
                        for row in round_assessments
                    ),
                    len(round_assessments),
                ),
                "requirement_sufficient_count": sufficient,
                "requirement_success_rate": _ratio(
                    sufficient, len(round_assessments)
                ),
                "raw_image_reached_count": raw_image,
                "failure_reason_counts": dict(
                    sorted(
                        Counter(
                            str(row.get("failure_reason"))
                            for row in round_assessments
                            if row.get("failure_reason") is not None
                        ).items()
                    )
                ),
            }
        )
    return result


def _active_match_policy_summary(query: Path, *, step7b: bool = False) -> dict[str, Any]:
    batches = _read_optional_jsonl(query / "qbank_match_batches.jsonl")
    fused = _read_optional_jsonl(query / "qbank_match_fused_candidates.jsonl")
    routes = _read_optional_jsonl(query / "query_route_decisions.jsonl")
    fallbacks = _read_optional_jsonl(query / "query_fallback_decisions.jsonl")
    active_batches = {
        str(row.get("match_batch_id")): row
        for row in batches
        if int(row.get("question_budget", -1)) == 8
        and int(row.get("active_round", -1)) == 2
    }
    candidates_by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fused:
        batch_id = str(row.get("match_batch_id"))
        if batch_id in active_batches and row.get("route_class") == "answerable":
            candidates_by_batch[batch_id].append(row)

    confidence_values = [
        float(row["confidence"])
        for row in routes
        if row.get("confidence") is not None
    ]
    gap_values = [
        float(row["top1_top2_gap"])
        for row in routes
        if row.get("top1_top2_gap") is not None
    ]
    threshold_grid: list[dict[str, float | int]] = []
    for threshold in _MATCH_THRESHOLD_GRID:
        for gap_threshold in _GAP_THRESHOLD_GRID:
            selected = 0
            for batch_id in sorted(active_batches):
                candidates = sorted(
                    candidates_by_batch.get(batch_id, []),
                    key=lambda row: (int(row.get("rank", 9999)), str(row.get("qbank_id"))),
                )
                if not candidates or candidates[0].get("modality_compatible") is not True:
                    continue
                if step7b and candidates[0].get("reranker_match_level") not in {
                    "exact", "strong",
                }:
                    continue
                score_field = "route_score" if step7b else "confidence"
                top_confidence = float(candidates[0].get(score_field) or 0.0)
                gap = (
                    None
                    if len(candidates) == 1
                    else max(0.0, top_confidence - float(candidates[1].get(score_field) or 0.0))
                )
                if top_confidence >= threshold and (
                    gap is None or gap >= gap_threshold
                ):
                    selected += 1
            threshold_grid.append(
                {
                    "match_threshold": threshold,
                    "gap_threshold": gap_threshold,
                    "answerable_program_route_count": selected,
                    "query_count": len(active_batches),
                }
            )
    return {
        "active_batch_count": len(active_batches),
        "route_outcome_counts": dict(
            sorted(Counter(str(row.get("outcome")) for row in routes).items())
        ),
        "fallback_reason_counts": dict(
            sorted(
                Counter(
                    str(row.get("reason"))
                    for row in fallbacks
                    if row.get("reason") is not None
                ).items()
            )
        ),
        "confidence": {
            "count": len(confidence_values),
            "mean": (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else None
            ),
            "minimum": min(confidence_values) if confidence_values else None,
            "maximum": max(confidence_values) if confidence_values else None,
        },
        "top1_top2_gap": {
            "count": len(gap_values),
            "mean": sum(gap_values) / len(gap_values) if gap_values else None,
            "minimum": min(gap_values) if gap_values else None,
            "maximum": max(gap_values) if gap_values else None,
        },
        "threshold_grid": threshold_grid,
    }


def _verify_scientific_summaries(
    *,
    evaluation: Mapping[str, Any],
    performance: Mapping[str, Any],
    compiler: Mapping[str, Any],
    doubter: Mapping[str, Any],
    match: Mapping[str, Any],
    executor: Mapping[str, Any],
    expected_query_count: int,
    step7b: bool,
) -> None:
    """Check report dimensions before artifact IO and again on the sealed bundle."""

    failures: list[str] = []
    expected_grid = {
        (threshold, gap)
        for threshold in _MATCH_THRESHOLD_GRID for gap in _GAP_THRESHOLD_GRID
    }

    def mapping(value: object) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def rounds_complete(value: object) -> bool:
        return isinstance(value, list) and all(isinstance(row, dict) for row in value) and [
            row.get("round_id") for row in value
        ] == [0, 1, 2, 3]

    for dataset in (DatasetName.MEMLENS.value, DatasetName.MEMGALLERY.value):
        evaluated = mapping(evaluation.get(dataset))
        measured = mapping(performance.get(dataset))
        compiled = mapping(compiler.get(dataset))
        doubted = mapping(doubter.get(dataset))
        matched = mapping(match.get(dataset))
        executed = mapping(executor.get(dataset))
        policy = mapping(matched.get("active_match_policy"))
        grid = policy.get("threshold_grid")
        grid_rows = grid if isinstance(grid, list) else []
        grid_pairs = [
            (row.get("match_threshold"), row.get("gap_threshold"))
            for row in grid_rows if isinstance(row, dict)
        ]
        # Configuration combinations and query denominators are independent axes.
        checks = {
            "evaluation": _evaluation_summary_complete(
                mapping(evaluated.get("summary")),
                expected_query_count=expected_query_count, step7b=step7b,
            ) and all(
                isinstance(evaluated.get(name.removesuffix(".jsonl")), list)
                for name in _EVALUATION_PUBLIC_JSONL_FILES
            ),
            "performance": isinstance(measured.get("write_time_model_calls"), dict)
            and isinstance(measured.get("query_and_fallback_model_calls"), dict)
            and isinstance(measured.get("amortized_curves"), list)
            and len(measured["amortized_curves"]) == 5,
            "projection_rounds": rounds_complete(compiled.get("projection_rounds")),
            "operator_rounds": rounds_complete(compiled.get("operator_rounds")),
            "doubter_rounds": rounds_complete(doubted.get("rounds")),
            "executor_rounds": rounds_complete(executed.get("executor_rounds")),
            "threshold_grid": len(grid_rows) == len(expected_grid)
            and len(grid_pairs) == len(expected_grid)
            and all(grid_pairs.count(pair) == 1 for pair in expected_grid),
            "threshold_query_denominator": bool(grid_rows) and all(
                isinstance(row, dict)
                and type(row.get("query_count")) is int
                and row["query_count"] == expected_query_count
                and type(row.get("answerable_program_route_count")) is int
                and 0 <= row["answerable_program_route_count"] <= expected_query_count
                for row in grid_rows
            ),
            "active_batch_denominator": type(policy.get("active_batch_count")) is int
            and policy["active_batch_count"] == expected_query_count,
        }
        failed_checks = sorted(name for name, passed in checks.items() if not passed)
        _event(
            "scientific_summary_contract", dataset=dataset,
            state="failed" if failed_checks else "ready",
            expected_query_count=expected_query_count,
            expected_threshold_combination_count=len(expected_grid),
            threshold_combination_count=len(grid_rows),
            checks=checks,
        )
        failures.extend(f"scientific_summary_{dataset}_{name}" for name in failed_checks)
    if failures:
        raise Step7BundleVerificationError(failures[0])


def _index(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): dict(row) for row in rows}


def _native_verified_candidate_ids(
    *,
    questions: Sequence[Mapping[str, Any]],
    generations: Sequence[Mapping[str, Any]],
    verifications: Sequence[Mapping[str, Any]],
    verification_batches: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Return candidates backed only by native generation/revision schemas."""

    generation_by_id = _index(generations, "generation_id")
    batch_by_id = _index(verification_batches, "verification_batch_id")
    native_questions: set[str] = set()
    version_by_qbank: dict[str, int] = {}
    for row in questions:
        qbank_id = str(row.get("qbank_id", ""))
        origin = row.get("candidate_origin")
        version = int(row.get("candidate_version", -1))
        generation = generation_by_id.get(str(row.get("generation_id", "")))
        if (
            qbank_id
            and (origin, version) in {("generated", 0), ("revised", 1)}
            and generation is not None
            and generation.get("parse_status") != "failed"
            and int(generation.get("repair_attempt_count", -1)) == 0
        ):
            native_questions.add(qbank_id)
            version_by_qbank[qbank_id] = version

    native_verifier_passes: set[str] = set()
    for row in verifications:
        qbank_id = str(row.get("qbank_id", ""))
        batch = batch_by_id.get(str(row.get("verification_batch_id", "")))
        metadata = batch.get("metadata") if batch is not None else None
        if (
            qbank_id in native_questions
            and row.get("verdict") == "pass"
            and int(row.get("verification_round", -1)) == version_by_qbank[qbank_id]
            and batch is not None
            and int(batch.get("input_count", -1)) == 1
            and batch.get("scope_qbank_id") == qbank_id
            and isinstance(metadata, dict)
            and int(metadata.get("repair_attempt_count", 0)) == 0
            and not bool(metadata.get("provider_output_fallback", False))
        ):
            native_verifier_passes.add(qbank_id)
    return native_questions & native_verifier_passes


def _stable_cards(
    rows: Sequence[dict[str, Any]], *, dataset: DatasetName, limit: int
) -> list[dict[str, Any]]:
    priority = {
        "planner_rejected": 0,
        "dry_run_insufficient": 1,
        "fallback": 2,
        "abstain": 3,
        "program_ready": 4,
        "questioner_rejected": 5,
    }
    ordered = sorted(
        rows,
        key=lambda row: (
            priority.get(str(row.get("bucket")), 9),
            stable_hash(
                [dataset.value, row.get("bucket"), row.get("qbank_id")]
            ),
            str(row.get("qbank_id")),
        ),
    )
    return ordered[:limit]


def _query_cards(
    *,
    dataset: DatasetName,
    coverage_cards: Sequence[dict[str, Any]],
    query_runtime: Path,
    evaluation_export: Path | None = None,
) -> list[dict[str, Any]]:
    runtime = query_runtime / "runtime"
    routes = _index(read_jsonl(runtime / "query_route_decisions.jsonl"), "query_id")
    selections = _index(
        read_jsonl(runtime / "query_answer_selections.jsonl"), "query_id"
    )
    evaluation_rows = (
        read_jsonl(evaluation_export / "private" / "query_results.jsonl")
        if evaluation_export is not None
        else []
    )
    evaluation_results = _index(evaluation_rows, "query_id")
    evaluation_results_by_answer = _index(evaluation_rows, "answer_id")
    evaluation_variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        evaluation_variants[str(row["query_id"])].append(row)
    fallbacks = _index(
        read_jsonl(runtime / "query_fallback_decisions.jsonl"), "query_id"
    )
    bindings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(runtime / "query_program_bindings.jsonl"):
        bindings[str(row["query_id"])].append(row)
    assessments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(
        runtime / "query_evidence_sufficiency_assessments.jsonl"
    ):
        assessments[str(row["query_id"])].append(row)
    active_match_batches = {
        str(row["query_id"]): row
        for row in read_jsonl(runtime / "qbank_match_batches.jsonl")
        if int(row.get("question_budget", -1)) == 8
        and int(row.get("active_round", -1)) == 2
    }
    fused: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(runtime / "qbank_match_fused_candidates.jsonl"):
        query_id = str(row["query_id"])
        active_batch = active_match_batches.get(query_id)
        if active_batch is not None and row.get("match_batch_id") == active_batch.get(
            "match_batch_id"
        ):
            fused[query_id].append(row)
    cards: list[dict[str, Any]] = []
    for source in sorted(coverage_cards, key=lambda row: str(row["query_id"])):
        query_id = str(source["query_id"])
        selection = selections.get(query_id)
        selected_answer_id = (
            str(selection.get("selected_answer_id"))
            if isinstance(selection, dict)
            and isinstance(selection.get("selected_answer_id"), str)
            and selection.get("selected_answer_id")
            else None
        )
        answer_evaluation = (
            evaluation_results_by_answer.get(selected_answer_id)
            if selected_answer_id is not None
            else None
        )
        query_evaluation = evaluation_results.get(query_id)
        active_evaluation = (answer_evaluation or query_evaluation) if selected_answer_id else None
        evaluation_link = (
            "selected_answer_id"
            if answer_evaluation is not None
            else "query_id"
            if query_evaluation is not None
            else "unresolved"
        )
        by_route_class = {
            route_class: sorted(
                (
                    row
                    for row in fused.get(query_id, [])
                    if row.get("route_class") == route_class
                ),
                key=lambda row: (
                    int(row.get("rank", 9999)),
                    str(row.get("qbank_id", "")),
                ),
            )[:50]
            for route_class in ("answerable", "control")
        }
        cards.append(
            {
                **source,
                "dataset": dataset.value,
                "runtime_match_batch": active_match_batches.get(query_id),
                "runtime_route": routes.get(query_id),
                "runtime_top50": by_route_class["answerable"],
                "runtime_control_top50": by_route_class["control"],
                "program_bindings": sorted(
                    bindings.get(query_id, []),
                    key=lambda row: int(row.get("round_id", 0)),
                ),
                "runtime_sufficiency": assessments.get(query_id, []),
                "fallback_decision": fallbacks.get(query_id),
                "answer_selection": selection,
                "active_evaluation": active_evaluation,
                "active_evaluation_state": (
                    "evaluated" if selected_answer_id else "no_answer_not_judged"
                ),
                "active_evaluation_link": evaluation_link,
                "variant_evaluations": sorted(
                    evaluation_variants.get(query_id, []),
                    key=lambda row: (
                        str(row.get("answer_variant", "")),
                        str(row.get("answer_id", "")),
                    ),
                ),
            }
        )
    return cards


def _validated_fallback_abstention(row: Mapping[str, Any]) -> bool:
    from pmmc.runtime.answering import validate_abstention_fact
    from pmmc.schemas import StructuredAnswerFact

    metadata = row.get("metadata") or {}
    if (
        not isinstance(metadata, dict)
        or metadata.get("counterfactual_terminal_outcome") != "abstained"
    ):
        return False
    try:
        fact = StructuredAnswerFact.model_validate(metadata.get("abstention"))
        validate_abstention_fact(fact)
    except ValueError:
        return False
    return (
        row.get("counterfactual_completed") is False
        and row.get("fallback_answer_id") is None
        and row.get("error_type") == "ValidatedAnswerAbstention"
        and fact.run_id == row.get("fallback_run_id")
        and fact.query_id == row.get("query_id") == row.get("fallback_query_id")
    )


def _query_audit_card_issue(card: Mapping[str, Any]) -> str | None:
    """Return one content-free contract issue for a query audit card."""

    batch = card.get("runtime_match_batch")
    if not isinstance(batch, dict):
        return "match_batch_missing"
    if batch.get("question_budget") != 8:
        return "question_budget_not_b8"
    if batch.get("active_round") != 2:
        return "active_round_not_r2"
    route = card.get("runtime_route")
    if not isinstance(route, dict):
        return "route_missing"
    if route.get("match_batch_id") != batch.get("match_batch_id"):
        return "route_batch_mismatch"

    selection = card.get("answer_selection")
    if not isinstance(selection, dict):
        return "answer_selection_missing"
    if selection.get("terminal_failure") is True:
        fallback = card.get("fallback_decision")
        if not isinstance(fallback, dict) or not _validated_fallback_abstention(fallback):
            return "answer_selection_terminal_failure"
        if (
            selection.get("selected_answer_id") is not None
            or card.get("active_evaluation") is not None
        ):
            return "abstention_has_answer_or_judge"
    else:
        selected_answer_id = selection.get("selected_answer_id")
        if not isinstance(selected_answer_id, str) or not selected_answer_id:
            return "selected_answer_id_missing"
        evaluation = card.get("active_evaluation")
        if not isinstance(evaluation, dict):
            return "active_evaluation_missing"
        if evaluation.get("query_id") != card.get("query_id"):
            return "evaluation_query_mismatch"
        if evaluation.get("answer_id") != selected_answer_id:
            return "evaluation_answer_mismatch"

    for field, route_class in (
        ("runtime_top50", "answerable"),
        ("runtime_control_top50", "control"),
    ):
        values = card.get(field)
        if not isinstance(values, list):
            return f"{route_class}_top50_missing"
        if len(values) > 50:
            return f"{route_class}_top50_oversized"
        if any(
            not isinstance(row, dict)
            or row.get("match_batch_id") != batch.get("match_batch_id")
            or row.get("route_class") != route_class
            for row in values
        ):
            return f"{route_class}_top50_scope_mismatch"
        if [row.get("rank") for row in values] != list(
            range(1, len(values) + 1)
        ):
            return f"{route_class}_rank_not_contiguous"
    return None


def _verify_query_audit_card_contract(
    query_cards: Sequence[Mapping[str, Any]],
    *,
    expected_per_dataset: int = 12,
) -> None:
    """Verify all cards and emit only aggregate, content-free diagnostics."""

    issue_counts: Counter[str] = Counter()
    dataset_counts = Counter(str(row.get("dataset")) for row in query_cards)
    expected_total = expected_per_dataset * 2
    if len(query_cards) != expected_total:
        issue_counts["denominator_mismatch"] += 1
    if dataset_counts != {
        DatasetName.MEMLENS.value: expected_per_dataset,
        DatasetName.MEMGALLERY.value: expected_per_dataset,
    }:
        issue_counts["dataset_balance_mismatch"] += 1
    for card in query_cards:
        issue = _query_audit_card_issue(card)
        if issue is not None:
            issue_counts[issue] += 1
    _event(
        "query_audit_contract",
        card_count=len(query_cards),
        contract_version=_QUERY_AUDIT_CONTRACT_VERSION,
        dataset_counts=dict(sorted(dataset_counts.items())),
        issue_counts=dict(sorted(issue_counts.items())),
        state="ready" if not issue_counts else "failed",
    )
    if issue_counts:
        first_issue = sorted(issue_counts)[0]
        raise Step7BundleVerificationError(
            f"query_audit_cards_{first_issue}"
        )


def _step7b_match_grid_complete(
    *,
    query_ids: set[str],
    grid_by_query: Mapping[str, Sequence[tuple[int, int]]],
    expected_grid: set[tuple[int, int]],
) -> bool:
    """Require each denominator query to own every grid cell exactly once."""

    return set(grid_by_query) == query_ids and all(
        len(cells) == len(expected_grid) and set(cells) == expected_grid
        for cells in grid_by_query.values()
    )


def _step7b_threshold_grid_route_eligible(row: Mapping[str, Any]) -> bool:
    """Check viability under the least restrictive predeclared threshold cell."""

    return (
        row.get("reranker_match_level") in {"exact", "strong"}
        and row.get("route_class") == "answerable"
        and row.get("modality_compatible") is True
        and int(row.get("rank", 0)) == 1
        and float(row.get("route_score") or 0.0) >= 0.50
    )


def _dataset_summary(
    *,
    dataset: DatasetName,
    compiler_runtime: Path,
    query_runtime: Path,
    evaluation_export: Path,
    expected_query_count: int = 12,
) -> tuple[dict[str, Any], list[str]]:
    compiler = compiler_runtime / "runtime"
    query = query_runtime / "runtime"
    generations = read_jsonl(compiler / "questioner_generations.jsonl")
    gates = read_jsonl(compiler / "questioner_candidate_gate_results.jsonl")
    questions = read_jsonl(compiler / "qbank_questions.jsonl")
    verification_batches = read_jsonl(
        compiler / "questioner_verification_batches.jsonl"
    )
    verifications = read_jsonl(
        compiler / "questioner_candidate_verifications.jsonl"
    )
    outcomes = read_jsonl(compiler / "planner_candidate_outcomes.jsonl")
    assessments = read_jsonl(compiler / "doubter_assessments.jsonl")
    dry_run_assessments = _read_optional_jsonl(
        compiler / "program_dry_run_assessments.jsonl"
    )
    compiler_evidence = _read_optional_jsonl(
        compiler / "program_execution_evidence_refs.jsonl"
    )
    r0_manifests = _read_optional_jsonl(compiler / "r0_input_manifests.jsonl")
    revision_patches = _read_optional_jsonl(
        compiler / "planner_revision_patches.jsonl"
    )
    routing_manifests = read_jsonl(
        compiler / "routing_projection_manifests.jsonl"
    )
    routing = read_jsonl(compiler / "routing_projection_members.jsonl")
    match_batches = read_jsonl(query / "qbank_match_batches.jsonl")
    channel_candidates = _read_optional_jsonl(
        query / "qbank_match_channel_candidates.jsonl"
    )
    fused_candidates = _read_optional_jsonl(
        query / "qbank_match_fused_candidates.jsonl"
    )
    reranker = _read_optional_jsonl(query / "qbank_reranker_assessments.jsonl")
    routes = read_jsonl(query / "query_route_decisions.jsonl")
    bindings = read_jsonl(query / "query_program_bindings.jsonl")
    sufficiency = read_jsonl(
        query / "query_evidence_sufficiency_assessments.jsonl"
    )
    relevance = _read_optional_jsonl(
        query / "query_evidence_relevance_assessments.jsonl"
    )
    structured_answers = _read_optional_jsonl(
        query / "structured_answer_outputs.jsonl"
    )
    answer_diagnostics = _read_optional_jsonl(
        query / "answer_format_diagnostics.jsonl"
    )
    fallbacks = read_jsonl(query / "query_fallback_decisions.jsonl")
    selections = read_jsonl(query / "query_answer_selections.jsonl")
    query_runs = read_jsonl(query / "query_executor_runs.jsonl")
    run_queries = read_jsonl(query / "run_queries.jsonl")
    final_answers = _read_optional_jsonl(query / "final_answers.jsonl")
    answer_reuse_links = _read_optional_jsonl(query / "answer_reuse_links.jsonl")
    evaluation_state = read_json(evaluation_export / "public" / "evaluation_summary.json")
    eligible_ids = {
        str(row["qbank_id"])
        for row in gates
        if row.get("status") == "eligible_for_planning"
    }
    final_pass_ids = {
        str(row["qbank_id"])
        for row in verifications
        if row.get("verdict") == "pass"
    }
    native_ids = _native_verified_candidate_ids(
        questions=questions,
        generations=generations,
        verifications=verifications,
        verification_batches=verification_batches,
    )
    planner_outcome_ids = {str(row["qbank_id"]) for row in outcomes}
    active_routing_manifest = (
        len(routing_manifests) == 1
        and int(routing_manifests[0].get("active_round", -1)) == 2
        and int(routing_manifests[0].get("member_count", -1)) == len(routing)
    )
    step7b = expected_query_count == 16
    query_run_metadata = (
        dict(query_runs[0].get("metadata", {})) if len(query_runs) == 1 else {}
    )
    executor_protocol_hash = str(
        query_run_metadata.get("query_executor_protocol_hash") or ""
    )
    executor_policy_hash = str(query_run_metadata.get("routing_policy_hash") or "")
    executor_routing_identity_valid = False
    if re.fullmatch(r"[0-9a-f]{64}", executor_protocol_hash):
        try:
            executor_routing_identity_valid = executor_policy_hash == (
                routing_policy_hash_from_protocol(
                    executor_protocol_hash,
                    match_threshold=float(query_run_metadata["match_threshold"]),
                    gap_threshold=float(query_run_metadata["gap_threshold"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            executor_routing_identity_valid = False
    query_ids = {str(row.get("query_id")) for row in run_queries}
    evidence_execution_ids = {
        str(row.get("execution_id")) for row in compiler_evidence
    }
    empty_path_doubter_passes = [
        row
        for row in assessments
        if row.get("action") == "pass"
        and str(row.get("execution_id")) not in evidence_execution_ids
    ]
    assessments_by_candidate: dict[str, list[int]] = defaultdict(list)
    for row in assessments:
        assessments_by_candidate[str(row.get("qbank_id"))].append(
            int(row.get("round_id", -1))
        )
    continuous_rounds = all(
        sorted(set(rounds)) == list(range(max(rounds) + 1))
        for rounds in assessments_by_candidate.values()
        if rounds and min(rounds) >= 0
    )
    invalid_dry_run_sufficiency = [
        row
        for row in dry_run_assessments
        if int(row.get("expected_group_count", 0)) > 0
        and float(row.get("expected_evidence_recall") or 0.0) == 0.0
        and row.get("evidence_sufficient") is True
    ]
    applied_patches = [row for row in revision_patches if row.get("applied") is True]
    invalid_patches = [
        row
        for row in revision_patches
        if (
            row.get("applied") is True
            and (
                not row.get("after_program_hash")
                or row.get("before_program_hash") == row.get("after_program_hash")
                or row.get("rejection_code") is not None
            )
        )
        or (
            row.get("applied") is not True
            and (
                row.get("after_program_hash") is not None
                or not row.get("rejection_code")
            )
        )
    ]
    lexical_query_ids = {
        str(row.get("query_id"))
        for row in channel_candidates
        if row.get("channel") == "lexical"
    }
    reranked_query_ids = {str(row.get("query_id")) for row in reranker}
    exact_reranker_count = sum(
        row.get("match_level") == "exact" for row in reranker
    )
    provider_invalid_reranker_count = sum(
        row.get("provider_schema_conformant") is not True for row in reranker
    )
    binding_query_ids = {str(row.get("query_id")) for row in bindings}
    fallback_structured = [
        row
        for row in structured_answers
        if row.get("answer_variant") == "native_mm_counterfactual"
        and row.get("parse_status") == "accepted"
        and row.get("provider_schema_conformant") is True
        and row.get("finish_reason") != "length"
    ]
    fallback_structured_query_ids = {
        str(row.get("query_id")) for row in fallback_structured
    }
    accepted_answer_ids = {
        str(row.get("answer_id"))
        for row in structured_answers
        if row.get("parse_status") == "accepted"
        and row.get("provider_schema_conformant") is True
        and row.get("answer_id") is not None
        and row.get("finish_reason") != "length"
    }
    final_answer_by_id = {
        str(row.get("answer_id")): row
        for row in final_answers
        if row.get("answer_id") is not None
    }
    reuse_source_by_target = {
        (str(row.get("query_id")), str(row.get("target_variant"))): str(
            row.get("source_answer_id")
        )
        for row in answer_reuse_links
    }
    final_selection_contract = all(
        row.get("terminal_failure") is True
        or (
            str(row.get("selected_answer_id")) in final_answer_by_id
            and (
                not step7b
                or (
                    final_answer_by_id[str(row.get("selected_answer_id"))].get(
                        "answer_variant"
                    )
                    == "active_final_provisional"
                    and reuse_source_by_target.get(
                        (str(row.get("query_id")), "active_final_provisional")
                    )
                    in accepted_answer_ids
                )
            )
        )
        for row in selections
    )
    invalid_accepted_diagnostics = [
        row
        for row in answer_diagnostics
        if row.get("accepted") is True and row.get("failure_codes")
    ]
    accepted_output_keys = {
        (row.get("query_id"), row.get("answer_variant"))
        for row in structured_answers if row.get("parse_status") == "accepted"
    }
    terminal_format_failure_keys = {
        (row.get("query_id"), row.get("answer_variant"))
        for row in answer_diagnostics
        if row.get("accepted") is False
        and (row.get("query_id"), row.get("answer_variant")) not in accepted_output_keys
    }
    exact_or_strong_fused = [
        row
        for row in fused_candidates
        if _step7b_threshold_grid_route_eligible(row)
    ]
    expected_grid = {(budget, round_id) for budget in (1, 2, 4, 8) for round_id in range(4)}
    grid_by_query: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in match_batches:
        grid_by_query[str(row.get("query_id"))].append(
            (int(row.get("question_budget", -1)), int(row.get("active_round", -1)))
        )
    match_grid_complete = _step7b_match_grid_complete(
        query_ids=query_ids,
        grid_by_query=grid_by_query,
        expected_grid=expected_grid,
    )
    route_query_ids = {str(row.get("query_id")) for row in routes}
    fallback_query_ids = {str(row.get("query_id")) for row in fallbacks}
    selection_query_ids = {str(row.get("query_id")) for row in selections}
    evaluation_query_count = int(
        evaluation_state.get("query_count", evaluation_state.get("selected_count", -1))
    )
    evaluation_variant_counts = evaluation_state.get("answer_variant_counts", {})
    if not isinstance(evaluation_variant_counts, dict):
        evaluation_variant_counts = {}
    issues: list[str] = []
    checks = {
        "query_run_count": len(query_runs) == 1,
        "query_run_completed": bool(query_runs)
        and query_runs[0].get("status") == "completed",
        "selected_query_count": bool(query_runs)
        and int(query_runs[0].get("selected_query_count", -1)) == expected_query_count,
        "native_eligible": bool(eligible_ids) and eligible_ids <= native_ids,
        "eligible_final_verifier": eligible_ids <= final_pass_ids,
        "planner_terminal": planner_outcome_ids == eligible_ids,
        "real_doubter": bool(assessments),
        "active_r2_routing": active_routing_manifest and bool(routing),
        "run_query_denominator": len(run_queries) == expected_query_count
        and len(query_ids) == expected_query_count,
        "match_grid": match_grid_complete,
        "route_denominator": route_query_ids == query_ids
        and len(routes) == expected_query_count,
        "reranker_denominator": (
            bool(reranker) if expected_query_count == 16 else True
        ),
        "fallback_denominator": fallback_query_ids == query_ids
        and len(fallbacks) == expected_query_count
        and all(
            row.get("counterfactual_completed") is True
            or (step7b and _validated_fallback_abstention(row)) for row in fallbacks
        ),
        "selection_denominator": selection_query_ids == query_ids
        and len(selections) == expected_query_count,
        "real_program_execution": bool(bindings),
        "evaluation_denominator": evaluation_query_count == expected_query_count,
    }
    if step7b:
        checks.update(
            {
                "r0_protocol_gate": len(r0_manifests) == 1
                and r0_manifests[0].get("hard_gate_passed") is True
                and r0_manifests[0].get("downstream_viability") is True
                and not r0_manifests[0].get("protocol_integrity_issues"),
                "planner_parse_native": all(
                    row.get("status") != "parse_failed" for row in outcomes
                ),
                "executor_routing_identity": executor_routing_identity_valid,
                "doubter_provider_native": all(
                    row.get("provider_schema_conformant") is True
                    for row in assessments
                ),
                "doubter_round_continuity": continuous_rounds,
                "empty_path_never_passes": not empty_path_doubter_passes,
                "write_time_expected_recall_enforced": not invalid_dry_run_sufficiency,
                "typed_revision_patch_valid": not invalid_patches,
                "lexical_hit_nonzero": bool(lexical_query_ids),
                "reranker_exact_nonzero": exact_reranker_count > 0,
                "reranker_provider_native": provider_invalid_reranker_count == 0,
                "reranker_query_denominator": reranked_query_ids == query_ids,
                "counterfactual_program_denominator": binding_query_ids == query_ids,
                "threshold_grid_has_program_route": bool(exact_or_strong_fused),
                "structured_fallback_denominator": fallback_structured_query_ids
                == query_ids,
                "evaluation_required_variants": int(
                    evaluation_variant_counts.get(
                        "native_mm_counterfactual", -1
                    )
                ) + sum(_validated_fallback_abstention(row) for row in fallbacks)
                == expected_query_count
                and int(
                    evaluation_variant_counts.get(
                        "active_final_provisional", -1
                    )
                ) + sum(row.get("terminal_failure") is True for row in selections)
                == expected_query_count,
                "final_answer_contract": final_selection_contract
                and not invalid_accepted_diagnostics,
                "no_terminal_format_failures": not terminal_format_failure_keys,
            }
        )
    for name, passed in checks.items():
        if not passed:
            issues.append(f"{dataset.value}:{name}")
    compiler_rounds = _compiler_round_summaries(compiler)
    executor_rounds = _executor_round_summaries(query)
    match_policy = _active_match_policy_summary(query, step7b=step7b)
    return (
        {
            "dataset": dataset.value,
            "query_count": expected_query_count,
            "raw_candidate_count": len(questions),
            "eligible_candidate_count": len(eligible_ids),
            "planner_outcome_count": len(outcomes),
            "doubter_feedback_count": len(assessments),
            "doubter_action_counts": dict(
                sorted(Counter(str(row.get("action")) for row in assessments).items())
            ),
            **compiler_rounds,
            "active_r2_routing_member_count": len(routing),
            "match_batch_count": len(match_batches),
            "route_outcome_counts": dict(
                sorted(Counter(str(row.get("outcome")) for row in routes).items())
            ),
            "program_binding_count": len(bindings),
            "runtime_sufficient_count": sum(
                row.get("sufficient") is True for row in sufficiency
            ),
            "fallback_counterfactual_count": sum(
                row.get("counterfactual_completed") is True for row in fallbacks
            ),
            "fallback_abstention_count": sum(
                _validated_fallback_abstention(row) for row in fallbacks
            ),
            "fallback_terminal_count": len(fallbacks),
            "active_fallback_count": sum(
                row.get("active_fallback") is True for row in fallbacks
            ),
            "selected_answer_count": sum(
                row.get("terminal_failure") is not True for row in selections
            ),
            "terminal_answer_failure_count": sum(
                row.get("terminal_failure") is True for row in selections
            ),
            "executor_rounds": executor_rounds,
            "active_match_policy": match_policy,
            "reranker_pair_count": len(reranker),
            "exact_reranker_pair_count": exact_reranker_count,
            "strong_reranker_pair_count": sum(
                row.get("match_level") == "strong" for row in reranker
            ),
            "lexical_query_hit_count": len(lexical_query_ids),
            "fused_candidate_count": len(fused_candidates),
            "reranker_match_levels": dict(
                sorted(
                    Counter(str(row.get("match_level")) for row in reranker).items()
                )
            ),
            "query_relevance_assessment_count": len(relevance),
            "query_relevance_pass_count": sum(
                row.get("passed") is True for row in relevance
            ),
            "planner_parse_failure_count": sum(
                row.get("status") == "parse_failed" for row in outcomes
            ),
            "revision_patch_count": len(revision_patches),
            "revision_patch_applied_count": len(applied_patches),
            "empty_path_doubter_pass_count": len(empty_path_doubter_passes),
            "invalid_dry_run_sufficiency_count": len(
                invalid_dry_run_sufficiency
            ),
            "structured_answer_count": len(structured_answers),
            "accepted_structured_answer_count": sum(
                row.get("parse_status") == "accepted" and row.get("supported") is True
                for row in structured_answers
            ),
            "answer_format_diagnostic_count": len(answer_diagnostics),
            "terminal_answer_contract_failure_count": len(terminal_format_failure_keys),
            "evaluation_answer_count": int(
                evaluation_state.get("selected_count", 0)
            ),
            "evaluation_answer_variant_counts": dict(
                sorted(
                    (str(key), int(value))
                    for key, value in evaluation_variant_counts.items()
                )
            ),
            "failure_layers": {
                "questioner_coverage_failure": max(0, len(generations) - len(eligible_ids)),
                "qbank_matching_failure": sum(
                    row.get("outcome") != "program" for row in routes
                ),
                "reranker_rejection": sum(
                    row.get("match_level") in {"weak", "none"} for row in reranker
                ),
                "program_path_failure": sum(
                    row.get("structural_requirement_success") is False
                    for row in sufficiency
                ),
                "runtime_relevance_failure": sum(
                    row.get("passed") is False for row in relevance
                ),
                "answer_contract_failure": sum(
                    row.get("accepted") is False for row in answer_diagnostics
                ),
                "answer_semantic_failure": int(
                    evaluation_state.get("semantic_failure_count", 0)
                ),
                "fallback_failure": sum(
                    row.get("counterfactual_completed") is not True
                    and not _validated_fallback_abstention(row) for row in fallbacks
                ),
            },
            "checks": checks,
        },
        issues,
    )


def _artifact_refs(root: Path) -> list[Step7PilotArtifactRef]:
    return [
        Step7PilotArtifactRef(
            role="private" if path.is_relative_to(root / "private") else "public",
            relative_path=path.relative_to(root).as_posix(),
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
            private=path.is_relative_to(root / "private"),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_relative_to(root / "transfer")
        and path.name != "manifest.json"
    ]


def _write_transfer(root: Path, manifest: Step7PilotManifest) -> tuple[Path, Path]:
    transfer = root / "transfer"
    transfer.mkdir(parents=True, exist_ok=True)
    label = (
        "step7b"
        if manifest.protocol_version == _STEP7B_PROTOCOL
        else "step7"
    )
    archive_path = transfer / f"aaai-pmmc-{label}-{manifest.pilot_id}.tar.zst"
    checksum_path = transfer / f"aaai-pmmc-{label}-{manifest.pilot_id}.sha256"
    included = [root / "manifest.json"] + [
        root / item.relative_path for item in manifest.artifacts
    ]
    with tempfile.NamedTemporaryFile(dir=transfer, suffix=".tar", delete=False) as handle:
        tar_path = Path(handle.name)
    try:
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(
                included, key=lambda value: value.relative_to(root).as_posix()
            ):
                # The filesystem root may be an atomic staging directory. The
                # portable archive must always expose the stable scientific ID.
                relative = (
                    PurePosixPath(manifest.pilot_id)
                    / path.relative_to(root).as_posix()
                )
                payload = path.read_bytes()
                info = tarfile.TarInfo(relative.as_posix())
                info.size = len(payload)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(payload))
        archive_path.write_bytes(
            zstandard.ZstdCompressor(
                level=9,
                threads=0,
                write_checksum=True,
                write_content_size=True,
            ).compress(tar_path.read_bytes())
        )
    finally:
        tar_path.unlink(missing_ok=True)
    checksum_path.write_text(
        f"{file_sha256(archive_path)}  {archive_path.name}\n", encoding="ascii"
    )
    archive_path.chmod(0o600)
    checksum_path.chmod(0o600)
    return archive_path, checksum_path


def build_step7_pilot_bundle(
    *,
    launch_id: str,
    partition_id: str,
    samples: Sequence[QuestionerPilotSample],
    compiler_runtime_exports: Mapping[DatasetName, Path],
    coverage_exports: Mapping[DatasetName, Path],
    query_runtime_exports: Mapping[DatasetName, Path],
    evaluation_exports: Mapping[DatasetName, Path],
    snapshot_dirs: Mapping[DatasetName, Path],
    benchmark_root: Path,
    output_root: Path,
    safe_execution_log: Path | None = None,
    progress_callback: Callable[[int, DatasetName], None] | None = None,
    step7b: bool = False,
    upstream_import_receipt: Mapping[str, Any] | None = None,
) -> tuple[Path, Step7PilotManifest]:
    """Build a deterministic public/private Step 7 or Step 7B review bundle."""

    query_count_per_dataset = 16 if step7b else 12
    query_count = query_count_per_dataset * 2
    protocol = _STEP7B_PROTOCOL if step7b else _PROTOCOL
    if upstream_import_receipt is not None:
        receipt_body = {k: v for k, v in upstream_import_receipt.items() if k != "content_hash"}
        if (
            not step7b
            or upstream_import_receipt.get("target_launch_id") != launch_id
            or upstream_import_receipt.get("content_hash") != stable_hash(receipt_body)
        ):
            raise ValueError("upstream_import_bundle_receipt_invalid")
    schema_version = (
        STEP7B_PILOT_SCHEMA_VERSION if step7b else STEP7_PILOT_SCHEMA_VERSION
    )

    datasets = {DatasetName.MEMLENS, DatasetName.MEMGALLERY}
    mappings = (
        compiler_runtime_exports,
        coverage_exports,
        query_runtime_exports,
        evaluation_exports,
        snapshot_dirs,
    )
    if any(set(values) != datasets for values in mappings):
        raise ValueError("Step 7 bundle sources must cover both datasets exactly")
    if {item.dataset for item in samples} != datasets:
        raise ValueError("Step 7 bundle requires both frozen pilot samples")

    identities: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    performance_summaries: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    candidate_facts: list[dict[str, Any]] = []
    candidate_cards: list[dict[str, Any]] = []
    query_cards: list[dict[str, Any]] = []
    compiler_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    _event("source_verification_started", dataset_count=2)
    for dataset in sorted(datasets, key=lambda value: value.value):
        compiler_export = compiler_runtime_exports[dataset]
        coverage_export = coverage_exports[dataset]
        query_export = query_runtime_exports[dataset]
        evaluation_export = evaluation_exports[dataset]
        compiler_identity, compiler_verification = _bundle_substage(
            dataset=dataset,
            substage="compiler_source_verification",
            operation=partial(_runtime_source_identity, compiler_export),
        )
        semantic_identity = _bundle_substage(
            dataset=dataset,
            substage="semantic_source_verification",
            operation=partial(_semantic_source_identity, coverage_export),
        )
        coverage_manifest = _bundle_substage(
            dataset=dataset,
            substage="coverage_export_verification",
            operation=partial(verify_qbank_coverage_export, coverage_export),
        )
        query_manifest = _bundle_substage(
            dataset=dataset,
            substage="query_export_verification",
            operation=partial(verify_experiment_export, query_export),
        )
        evaluation_manifest = _bundle_substage(
            dataset=dataset,
            substage="evaluation_export_verification",
            operation=partial(verify_evaluation_export, evaluation_export),
        )
        if str(compiler_identity["dataset"]) != dataset.value:
            raise ValueError("compiler export dataset identity differs")
        query_run_rows = read_jsonl(query_export / "runtime" / "query_executor_runs.jsonl")
        if len(query_run_rows) != 1 or str(query_run_rows[0].get("dataset")) != dataset.value:
            raise ValueError("query export dataset identity differs")
        if query_run_rows[0].get("status") != "completed":
            raise ValueError("query export completion authority is not terminal")
        if query_run_rows[0].get("source_compiler_run_id") != compiler_identity["run_id"]:
            raise ValueError("query run references another compiler export")
        identities[dataset.value] = {
            **compiler_identity,
            "compiler_export_id": compiler_identity["runtime_export_id"],
            "compiler_export_verification": compiler_verification,
            "coverage_export_id": coverage_manifest["export_id"],
            "semantic": semantic_identity,
            "semantic_export_id": coverage_manifest["export_id"],
            "query_export_id": query_manifest["export_id"],
            "query_run_id": query_manifest["run_id"],
            "query_executor": query_run_rows[0],
            "evaluation_export_id": evaluation_manifest["export_id"],
            "evaluation_run_id": evaluation_manifest["evaluation_run_id"],
        }
        summary, dataset_issues = _bundle_substage(
            dataset=dataset,
            substage="scientific_summary",
            operation=partial(
                _dataset_summary,
                dataset=dataset,
                compiler_runtime=compiler_export,
                query_runtime=query_export,
                evaluation_export=evaluation_export,
                expected_query_count=query_count_per_dataset,
            ),
        )
        summaries[dataset.value] = summary
        issues.extend(dataset_issues)
        compiler_call_summary = _model_call_summary(
            read_jsonl(compiler_export / "runtime" / "model_calls.jsonl")
        )
        current_query_calls = [
            *read_jsonl(query_export / "runtime" / "model_calls.jsonl"),
            *read_jsonl(query_export / "runtime" / "pmmc_fallback_model_calls.jsonl"),
        ]
        inherited_query_calls = read_jsonl(
            query_export / "runtime" / "reused_query_embedding_model_calls.jsonl"
        )
        # Reuse costs no new inference, but is not free in logical-method cost curves.
        query_call_summary = _model_call_summary([*current_query_calls, *inherited_query_calls])
        performance_summaries[dataset.value] = {
            "write_time_model_calls": compiler_call_summary,
            "query_and_fallback_model_calls": query_call_summary,
            "current_child_model_calls": _model_call_summary(current_query_calls),
            "inherited_retrieval_model_calls": _model_call_summary(inherited_query_calls),
            "cost_basis": "logical_calls_including_reused_query_embeddings",
            "amortized_curves": _amortized_curves(
                write_summary=compiler_call_summary,
                query_summary=query_call_summary,
                query_count=query_count_per_dataset,
            ),
        }
        facts = _bundle_substage(
            dataset=dataset,
            substage="candidate_program_cards",
            operation=partial(
                _program_candidate_facts,
                dataset=dataset,
                runtime_export=compiler_export,
                snapshot_dir=snapshot_dirs[dataset],
            ),
        )
        candidate_facts.extend(facts)
        dataset_candidate_cards = _stable_cards(facts, dataset=dataset, limit=60)
        candidate_cards.extend(dataset_candidate_cards)
        coverage = _bundle_substage(
            dataset=dataset,
            substage="coverage_cards",
            operation=partial(
                _coverage_payloads,
                dataset=dataset,
                coverage_export=coverage_export,
                runtime_export=compiler_export,
                snapshot_dir=snapshot_dirs[dataset],
            ),
        )
        _bundle_substage(
            dataset=dataset,
            substage="semantic_card_annotation",
            operation=partial(
                _annotate_semantic_matches,
                coverage_export=coverage_export,
                coverage_cards=coverage,
            ),
        )
        dataset_query_cards = _bundle_substage(
            dataset=dataset,
            substage="query_cards",
            operation=partial(
                _query_cards,
                dataset=dataset,
                coverage_cards=coverage,
                query_runtime=query_export,
                evaluation_export=evaluation_export,
            ),
        )
        query_cards.extend(dataset_query_cards)
        compiler_rows.extend(
            _rows(compiler_export / "runtime", _PRIVATE_COMPILER_FILES, dataset)
        )
        query_rows.extend(
            _rows(query_export / "runtime", _PRIVATE_QUERY_FILES, dataset)
        )
        _event(
            "dataset_ready",
            candidate_fact_count=len(facts),
            candidate_review_card_count=len(dataset_candidate_cards),
            dataset=dataset.value,
            integrity_issue_count=len(dataset_issues),
            integrity_issue_types=dataset_issues,
            query_card_count=len(dataset_query_cards),
            progress_unit="review_cards",
            bundle_completed=False,
        )
        if progress_callback is not None:
            progress_callback(
                len(dataset_candidate_cards) + len(dataset_query_cards),
                dataset,
            )

    if len(query_cards) != query_count:
        issues.append("query_audit_denominator")
    compiler_ids = {
        str(identity["run_id"]) for identity in identities.values()
    }
    query_ids = {
        str(identity["query_run_id"]) for identity in identities.values()
    }
    if len(compiler_ids) != 2 or len(query_ids) != 2:
        issues.append("cross_dataset_run_identity")

    # Review-card closure depends only on already verified source exports. Run
    # it before thumbnail/model-artifact projection so a bad join fails fast.
    _bundle_global_substage(
        substage="query_audit_contract",
        operation=partial(
            _verify_query_audit_card_contract,
            query_cards,
            expected_per_dataset=query_count_per_dataset,
        ),
    )
    evaluation_public = {
        dataset.value: _evaluation_public_payload(evaluation_exports[dataset])
        for dataset in sorted(datasets, key=lambda value: value.value)
    }
    _bundle_global_substage(
        substage="scientific_summaries_preflight",
        operation=partial(
            _verify_scientific_summaries,
            evaluation=evaluation_public,
            performance=performance_summaries,
            compiler=summaries,
            doubter={key: {"rounds": value["doubter_rounds"]}
                     for key, value in summaries.items()},
            match=summaries,
            executor=summaries,
            expected_query_count=query_count_per_dataset,
            step7b=step7b,
        ),
    )

    identity_payload = {
        "launch_id": launch_id,
        "partition_id": partition_id,
        "sample_ids": sorted(item.sample_id for item in samples),
        "source_export_ids": {
            key: {
                field: value[field]
                for field in (
                    "compiler_export_id",
                    "coverage_export_id",
                    "query_export_id",
                    "evaluation_export_id",
                )
            }
            for key, value in sorted(identities.items())
        },
    }
    pilot_id = (
        "step7b-pilot-" if step7b else "step7-pilot-"
    ) + stable_hash(identity_payload)[:24]
    automatic_passed = not issues
    status = (
        Step7PilotStatus.AWAITING_THRESHOLD_REVIEW
        if step7b and automatic_passed
        else Step7PilotStatus.NEEDS_REMEDIATION
        if step7b
        else Step7PilotStatus.AWAITING_APPROVAL
        if automatic_passed
        else Step7PilotStatus.FAILED
    )
    source_compiler_export_ids = {
        key: str(value["compiler_export_id"])
        for key, value in sorted(identities.items())
    }
    source_coverage_export_ids = {
        key: str(value["coverage_export_id"])
        for key, value in sorted(identities.items())
    }
    source_query_export_ids = {
        key: str(value["query_export_id"])
        for key, value in sorted(identities.items())
    }
    source_evaluation_export_ids = {
        key: str(value["evaluation_export_id"])
        for key, value in sorted(identities.items())
    }
    scientific_metadata = {
        "active_round": 2,
        "fact_projection_version": _PORTABLE_FACT_PROJECTION_VERSION,
        "gap_threshold": 0.05,
        "match_threshold": 0.60,
        "maximum_round": 3,
        "paper_result": False,
        "question_budget": 8,
        "underpowered": True,
    }
    reusable_payload = {
        "automatic_integrity_passed": automatic_passed,
        "compiler_run_ids": sorted(compiler_ids),
        "integrity_issue_types": sorted(set(issues)),
        "launch_id": launch_id,
        "metadata": scientific_metadata,
        "partition_id": partition_id,
        "pilot_id": pilot_id,
        "prefix_query_count": 4,
        "protocol_version": protocol,
        "query_count": query_count,
        "query_executor_run_ids": sorted(query_ids),
        "samples": [
            item.model_dump(mode="json")
            for item in sorted(samples, key=lambda value: value.dataset.value)
        ],
        "schema_version": schema_version,
        "source_compiler_export_ids": source_compiler_export_ids,
        "source_coverage_export_ids": source_coverage_export_ids,
        "source_evaluation_export_ids": source_evaluation_export_ids,
        "source_query_export_ids": source_query_export_ids,
        "status": status.value,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / pilot_id
    if destination.exists():
        existing = verify_step7_pilot_bundle(destination)
        reuse_contract_match = _step7_pilot_reuse_identity(
            existing.model_dump(mode="json")
        ) == _step7_pilot_reuse_identity(reusable_payload)
        _event(
            "existing_bundle_checked",
            integrity_passed=existing.automatic_integrity_passed,
            pilot_id=pilot_id,
            reuse_contract_match=reuse_contract_match,
        )
        if not reuse_contract_match:
            raise ValueError(
                "Step 7 pilot ID conflicts with an incompatible scientific bundle"
            )
        _event(
            "existing_bundle_reused",
            artifact_count=len(existing.artifacts),
            pilot_id=pilot_id,
            reason="verified_scientific_identity_match",
        )
        return destination, existing

    staging = output_root / f".{pilot_id}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "public").mkdir(parents=True)
    (staging / "private").mkdir(parents=True)

    thumbnail_count = _annotate_thumbnails(
        [candidate_cards, query_cards],
        benchmark_root=benchmark_root,
        thumbnail_root=staging / "private" / "referenced_thumbnails",
    )
    portable_compiler_rows, compiler_redactions = _project_portable_rows(
        compiler_rows
    )
    portable_query_rows, query_redactions = _project_portable_rows(query_rows)
    portable_candidate_cards, candidate_redactions = _project_portable_rows(
        candidate_cards
    )
    portable_query_cards, query_card_redactions = _project_portable_rows(
        query_cards
    )
    portable_identities, identity_redactions = _project_portable_object(identities)
    portability_redactions = (
        compiler_redactions
        + query_redactions
        + candidate_redactions
        + query_card_redactions
        + identity_redactions
    )
    _event(
        "portable_fact_projection_completed",
        artifact_group_count=5,
        projection_version=_PORTABLE_FACT_PROJECTION_VERSION,
        redaction_count=portability_redactions,
    )
    _write_zstd_jsonl(
        staging / "private" / "all_compiler_round_facts.jsonl.zst",
        portable_compiler_rows,
    )
    _write_zstd_jsonl(
        staging / "private" / "all_query_match_top50.jsonl.zst",
        [
            row
            for row in portable_query_rows
            if row["fact_type"].startswith("qbank_match")
        ],
    )
    _write_zstd_jsonl(
        staging / "private" / "all_program_execution_facts.jsonl.zst",
        [
            row
            for row in portable_query_rows
            if row["fact_type"].startswith("program_")
            or row["fact_type"].startswith("query_program")
            or row["fact_type"].startswith("query_evidence")
        ],
    )
    _write_zstd_jsonl(
        staging / "private" / "all_answer_selection_facts.jsonl.zst",
        [
            row
            for row in portable_query_rows
            if row["fact_type"].startswith("query_fallback")
            or row["fact_type"].startswith("query_answer")
            or row["fact_type"] == "final_answers"
            or row["fact_type"].startswith("pmmc_fallback")
        ],
    )
    if step7b:
        _write_zstd_jsonl(
            staging / "private" / "all_query_match_rerank_top50.jsonl.zst",
            [
                row
                for row in portable_query_rows
                if row["fact_type"].startswith("qbank_match")
                or row["fact_type"].startswith("qbank_reranker")
                or row["fact_type"].startswith("qbank_lexical")
            ],
        )
        _write_zstd_jsonl(
            staging / "private" / "all_execution_answer_facts.jsonl.zst",
            [
                row
                for row in portable_query_rows
                if row["fact_type"].startswith("program_")
                or row["fact_type"].startswith("query_program")
                or row["fact_type"].startswith("query_evidence")
                or row["fact_type"].startswith("query_execution_reuse")
                or row["fact_type"].startswith("reused_query_embedding")
                or row["fact_type"].startswith("structured_answer")
                or row["fact_type"].startswith("answer_")
                or row["fact_type"] == "final_answers"
                or row["fact_type"].startswith("pmmc_fallback")
            ],
        )
        write_jsonl(
            staging / "private" / "stage_component_manifests.jsonl",
            [
                row
                for row in portable_compiler_rows
                if row["fact_type"] == "stage_component_manifests"
            ],
        )
    write_jsonl(
        staging / "private" / "query_audit_cards.jsonl", portable_query_cards
    )
    write_jsonl(
        staging / "private" / "candidate_program_doubter_cards.jsonl",
        portable_candidate_cards,
    )
    write_json(staging / "private" / "source_exports.json", portable_identities)

    selection = _step6d_model_artifact_selection(
        runtime_exports=compiler_runtime_exports,
        coverage_exports=coverage_exports,
        candidate_cards=candidate_cards,
        coverage_cards=query_cards,
    )
    query_card_ids: dict[str, set[str]] = defaultdict(set)
    for card in query_cards:
        query_card_ids[str(card["dataset"])].add(str(card["query_id"]))
    for dataset in sorted(query_runtime_exports, key=lambda value: value.value):
        for file_name in ("model_calls.jsonl", "pmmc_fallback_model_calls.jsonl"):
            for call in read_jsonl(
                query_runtime_exports[dataset] / "runtime" / file_name
            ):
                model_call_id = call.get("model_call_id")
                if (
                    call.get("stage") == "answer"
                    and str(call.get("query_id") or "")
                    in query_card_ids[dataset.value]
                    and isinstance(model_call_id, str)
                    and model_call_id
                ):
                    selection.setdefault(
                        (dataset.value, "query_runtime", model_call_id), set()
                    ).add("query_answer_audit")
    artifact_rows = _archive_model_artifacts(
        runtime_exports=compiler_runtime_exports,
        coverage_exports=coverage_exports,
        query_runtime_exports=query_runtime_exports,
        destination_root=staging / "private" / "model_artifacts",
        selected_calls=selection,
    )
    write_jsonl(
        staging / "private" / "model_artifact_manifest.jsonl", artifact_rows
    )
    if safe_execution_log is not None and safe_execution_log.is_file():
        redacted, _count = _redact_nonportable_string(
            safe_execution_log.read_text(encoding="utf-8", errors="replace")
        )
        (staging / "private" / "safe_execution.log").write_text(
            redacted, encoding="utf-8"
        )
    else:
        (staging / "private" / "safe_execution.log").write_text(
            "Safe execution log was not supplied to the offline bundle.\n",
            encoding="utf-8",
        )

    if upstream_import_receipt is not None:
        write_json(staging / "private" / "upstream_import.json", dict(upstream_import_receipt))

    automatic = {
        "pilot_id": pilot_id,
        "status": status.value,
        "paper_result": False,
        "underpowered": True,
        "query_count": query_count,
        "prefix_query_count": 4,
        "automatic_integrity_passed": automatic_passed,
        "integrity_issue_types": sorted(set(issues)),
        "datasets": summaries,
    }
    write_json(staging / "public" / "automatic_summary.json", automatic)
    write_json(
        staging / "public" / "compiler_summary.json",
        {key: {k: v for k, v in value.items() if k in {
            "raw_candidate_count", "eligible_candidate_count",
            "planner_outcome_count", "doubter_feedback_count",
            "doubter_action_counts", "active_r2_routing_member_count",
            "projection_rounds", "operator_rounds",
        }} for key, value in sorted(summaries.items())},
    )
    write_json(
        staging / "public" / "doubter_round_summary.json",
        {
            key: {
                "feedback_count": value["doubter_feedback_count"],
                "action_counts": value["doubter_action_counts"],
                "active_r2_member_count": value["active_r2_routing_member_count"],
                "rounds": value["doubter_rounds"],
                "projection_rounds": value["projection_rounds"],
            }
            for key, value in sorted(summaries.items())
        },
    )
    write_json(
        staging / "public" / "qbank_match_summary.json",
        {key: {k: v for k, v in value.items() if k in {
            "query_count", "match_batch_count", "route_outcome_counts",
            "active_match_policy",
        }} for key, value in sorted(summaries.items())},
    )
    write_json(
        staging / "public" / "executor_summary.json",
        {key: {k: v for k, v in value.items() if k in {
            "program_binding_count", "runtime_sufficient_count",
            "selected_answer_count", "terminal_answer_failure_count",
            "executor_rounds",
        }} for key, value in sorted(summaries.items())},
    )
    write_json(
        staging / "public" / "answer_fallback_summary.json",
        {key: {k: v for k, v in value.items() if k in {
            "fallback_counterfactual_count", "active_fallback_count",
            "selected_answer_count", "terminal_answer_failure_count",
            "active_match_policy",
        }} for key, value in sorted(summaries.items())},
    )
    if step7b:
        write_json(
            staging / "public" / "questioner_summary.json",
            {
                key: {
                    field: value[field]
                    for field in (
                        "raw_candidate_count",
                        "eligible_candidate_count",
                    )
                }
                for key, value in sorted(summaries.items())
            },
        )
        write_json(
            staging / "public" / "planner_summary.json",
            {
                key: {
                    "planner_outcome_count": value["planner_outcome_count"],
                    "operator_rounds": value["operator_rounds"],
                }
                for key, value in sorted(summaries.items())
            },
        )
        write_json(
            staging / "public" / "reranker_summary.json",
            {
                key: {
                    "pair_count": value["reranker_pair_count"],
                    "match_levels": value["reranker_match_levels"],
                    "active_match_policy": value["active_match_policy"],
                }
                for key, value in sorted(summaries.items())
            },
        )
        write_json(
            staging / "public" / "answer_summary.json",
            {
                key: {
                    field: value[field]
                    for field in (
                        "structured_answer_count",
                        "accepted_structured_answer_count",
                        "answer_format_diagnostic_count",
                        "selected_answer_count",
                        "terminal_answer_failure_count",
                    )
                }
                for key, value in sorted(summaries.items())
            },
        )
        write_json(
            staging / "public" / "paired_step7_comparison.json",
            {
                "paired_query_count_per_dataset": 12,
                "step7b_query_count_per_dataset": query_count_per_dataset,
                "historical_step7_bundle_required_for_metric_deltas": True,
                "pairing_contract": "questioner-quality-pilot-v4",
            },
        )
    write_json(staging / "public" / "evaluation_summary.json", evaluation_public)
    write_json(
        staging / "public" / "performance_summary.json",
        {
            "launch_id": launch_id,
            "model_artifact_count": len(artifact_rows),
            "thumbnail_count": thumbnail_count,
            "query_count": query_count,
            "datasets": performance_summaries,
            "note": (
                "Stage resource telemetry remains in safe_execution.log; model-call "
                "and amortized write/query summaries are included here."
            ),
        },
    )
    report_lines = [
        "# PMMC Step 7B Executor Pilot" if step7b else "# PMMC Step 7 Executor Pilot",
        "",
        f"- Status: `{status.value}`",
        f"- Scope: development only; {query_count_per_dataset} queries per benchmark",
        "- Prefix: 2 real queries per benchmark, excluded from the scientific denominator",
        f"- Automatic integrity: `{str(automatic_passed).lower()}`",
        f"- Query audit cards: `{len(query_cards)}`",
        f"- Candidate/program/Doubter cards: `{len(candidate_cards)}`",
        "",
        "Scientific quality metrics are report-only. Approval requires explicit review.",
        "",
    ]
    (staging / "public" / "report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    audit_payload = {
        "pilot_id": pilot_id,
        "protocol_version": (
            "pmmc-step7b-audit-v1" if step7b else "pmmc-step7-audit-v1"
        ),
        "partition_id": partition_id,
        "query_audit_count": len(query_cards),
        "candidate_program_doubter_audit_count": len(candidate_cards),
        "automatic_integrity_passed": automatic_passed,
        "integrity_issue_types": sorted(set(issues)),
        "source_export_ids": identity_payload["source_export_ids"],
        "artifact_selection_policy": "failures-revisions-query-answer-audit-v1",
        "fact_projection_redaction_count": portability_redactions,
        "fact_projection_version": _PORTABLE_FACT_PROJECTION_VERSION,
    }
    audit_payload["content_hash"] = stable_hash(audit_payload)
    write_json(staging / "private" / "audit_manifest.json", audit_payload)

    for directory in [staging, *[path for path in staging.rglob("*") if path.is_dir()]]:
        directory.chmod(0o700)
    for path in staging.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    artifact_refs = _artifact_refs(staging)
    manifest_payload = {
        "pilot_id": pilot_id,
        "schema_version": schema_version,
        "protocol_version": protocol,
        "launch_id": launch_id,
        "partition_id": partition_id,
        "samples": [
            item.model_dump(mode="json")
            for item in sorted(samples, key=lambda value: value.dataset.value)
        ],
        "status": status.value,
        "query_executor_run_ids": sorted(query_ids),
        "compiler_run_ids": sorted(compiler_ids),
        "source_compiler_export_ids": source_compiler_export_ids,
        "source_coverage_export_ids": source_coverage_export_ids,
        "source_query_export_ids": source_query_export_ids,
        "source_evaluation_export_ids": source_evaluation_export_ids,
        "query_count": query_count,
        "prefix_query_count": 4,
        "automatic_integrity_passed": automatic_passed,
        "integrity_issue_types": sorted(set(issues)),
        "artifacts": [item.model_dump(mode="json") for item in artifact_refs],
        "metadata": {
            **scientific_metadata,
            "audit_manifest_hash": audit_payload["content_hash"],
            "model_artifact_count": len(artifact_rows),
            "fact_projection_redaction_count": portability_redactions,
            "thumbnail_count": thumbnail_count,
        },
    }
    manifest = Step7PilotManifest(
        **manifest_payload, content_hash=stable_hash(manifest_payload)
    )
    write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
    (staging / "manifest.json").chmod(0o600)
    _bundle_global_substage(
        substage="transfer_write",
        operation=partial(_write_transfer, staging, manifest),
    )
    _bundle_global_substage(
        substage="staging_verification",
        operation=partial(
            verify_step7_pilot_bundle,
            staging,
            _allow_internal_staging=True,
        ),
    )
    _event(
        "staging_verification_completed",
        artifact_count=len(manifest.artifacts),
        pilot_id=pilot_id,
    )

    if destination.exists():
        existing = verify_step7_pilot_bundle(destination)
        if _step7_pilot_reuse_identity(existing.model_dump(mode="json")) != (
            _step7_pilot_reuse_identity(manifest.model_dump(mode="json"))
        ):
            shutil.rmtree(staging)
            raise ValueError(
                "Step 7 pilot ID conflicts with an incompatible scientific bundle"
            )
        shutil.rmtree(staging)
        _event(
            "existing_bundle_reused",
            artifact_count=len(existing.artifacts),
            pilot_id=pilot_id,
            reason="concurrent_verified_scientific_identity_match",
        )
        return destination, existing
    staging.replace(destination)
    _event(
        "bundle_ready",
        artifact_count=len(manifest.artifacts),
        integrity_passed=automatic_passed,
        pilot_id=pilot_id,
    )
    return destination, manifest


def _scan_file(path: Path) -> None:
    portability_issue = _portable_scan_issue(
        path,
        strict_text=path.suffix.lower() in {".log", ".md", ".txt"},
    )
    if portability_issue is not None:
        raise Step7BundleVerificationError(
            "artifact_portability_"
            + re.sub(r"[^a-z0-9_]+", "_", portability_issue.lower()).strip("_")
        )
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".zst"}:
        return
    payload = path.read_bytes()
    if any(pattern.search(payload) for pattern in _SECRET_PATTERNS):
        raise Step7BundleVerificationError("artifact_secret_marker")


def _pilot_directory_matches_manifest(
    pilot_dir: Path,
    pilot_id: str,
    *,
    allow_internal_staging: bool,
) -> bool:
    if pilot_dir.name == pilot_id:
        return True
    if not allow_internal_staging:
        return False
    prefix = f".{pilot_id}.staging-"
    suffix = pilot_dir.name.removeprefix(prefix)
    return pilot_dir.name.startswith(prefix) and suffix.isdecimal()


def verify_step7_pilot_bundle(
    pilot_dir: Path,
    *,
    require_transfer: bool = True,
    _allow_internal_staging: bool = False,
) -> Step7PilotManifest:
    with _BundleVerificationCheck("manifest_identity"):
        manifest = Step7PilotManifest.model_validate(
            read_json(pilot_dir / "manifest.json")
        )
        if not _pilot_directory_matches_manifest(
            pilot_dir,
            manifest.pilot_id,
            allow_internal_staging=_allow_internal_staging,
        ):
            raise ValueError("Step 7 pilot directory differs from pilot_id")
        if stable_hash(
            manifest.model_dump(mode="json", exclude={"content_hash"})
        ) != manifest.content_hash:
            raise ValueError("Step 7 pilot content hash mismatch")
        step7b = manifest.protocol_version == _STEP7B_PROTOCOL
        expected_query_count = 16 if step7b else 12

    with _BundleVerificationCheck("artifact_inventory"):
        expected = sorted(item.relative_path for item in manifest.artifacts)
        actual = sorted(
            path.relative_to(pilot_dir).as_posix()
            for path in pilot_dir.rglob("*")
            if path.is_file()
            and not path.is_relative_to(pilot_dir / "transfer")
            and path.name != "manifest.json"
        )
        if expected != actual:
            raise ValueError("Step 7 pilot artifact inventory differs")

    with _BundleVerificationCheck("artifact_integrity"):
        for item in manifest.artifacts:
            path = pilot_dir / item.relative_path
            if path.is_symlink() or not path.resolve().is_relative_to(
                pilot_dir.resolve()
            ):
                raise ValueError("Step 7 pilot artifact escapes its root")
            if (
                path.stat().st_size != item.size_bytes
                or file_sha256(path) != item.sha256
            ):
                raise ValueError("Step 7 artifact checksum mismatch")
            if stat.S_IMODE(path.stat().st_mode) != 0o600:
                raise ValueError("Step 7 artifact must use mode 0600")
            _scan_file(path)

    with _BundleVerificationCheck("automatic_integrity"):
        automatic = read_json(pilot_dir / "public" / "automatic_summary.json")
        if (
            automatic.get("automatic_integrity_passed")
            is not manifest.automatic_integrity_passed
            or sorted(automatic.get("integrity_issue_types", []))
            != sorted(manifest.integrity_issue_types)
        ):
            raise ValueError("Step 7 automatic integrity state differs")
    with _BundleVerificationCheck("query_audit_cards"):
        query_cards = read_jsonl(
            pilot_dir / "private" / "query_audit_cards.jsonl"
        )
        candidate_cards = read_jsonl(
            pilot_dir / "private" / "candidate_program_doubter_cards.jsonl"
        )
        _verify_query_audit_card_contract(
            query_cards, expected_per_dataset=expected_query_count
        )

    with _BundleVerificationCheck("dataset_isolation"):
        qbank_by_dataset: dict[str, set[str]] = defaultdict(set)
        for row in candidate_cards:
            qbank_by_dataset[str(row.get("dataset"))].add(
                str(row.get("qbank_id"))
            )
        if qbank_by_dataset[DatasetName.MEMLENS.value] & qbank_by_dataset[
            DatasetName.MEMGALLERY.value
        ]:
            raise ValueError("Step 7 bundle contains cross-dataset QBank identities")

    with _BundleVerificationCheck("compressed_fact_archives"):
        for name in (
            "all_compiler_round_facts.jsonl.zst",
            "all_query_match_top50.jsonl.zst",
            "all_program_execution_facts.jsonl.zst",
            *(
                (
                    "all_query_match_rerank_top50.jsonl.zst",
                    "all_execution_answer_facts.jsonl.zst",
                )
                if step7b
                else ()
            ),
        ):
            _read_zstd_jsonl(pilot_dir / "private" / name)
        answer_facts = _read_zstd_jsonl(
            pilot_dir / "private" / "all_answer_selection_facts.jsonl.zst"
        )

    with _BundleVerificationCheck("fallback_answer_denominator"):
        fallback_answer_counts = Counter(
            str(row.get("dataset"))
            for row in answer_facts
            if row.get("fact_type") == "pmmc_fallback_answers"
            or (step7b and row.get("fact_type") == "query_fallback_decisions"
                and _validated_fallback_abstention(row))
        )
        if fallback_answer_counts != {
            DatasetName.MEMLENS.value: expected_query_count,
            DatasetName.MEMGALLERY.value: expected_query_count,
        }:
            raise ValueError("Step 7 bundle lacks all-query Native-MM fallback answers")

    with _BundleVerificationCheck("scientific_summaries"):
        evaluation = read_json(pilot_dir / "public" / "evaluation_summary.json")
        performance = read_json(pilot_dir / "public" / "performance_summary.json")
        compiler_summary = read_json(
            pilot_dir / "public" / "compiler_summary.json"
        )
        doubter_summary = read_json(
            pilot_dir / "public" / "doubter_round_summary.json"
        )
        match_summary = read_json(
            pilot_dir / "public" / "qbank_match_summary.json"
        )
        executor_summary = read_json(
            pilot_dir / "public" / "executor_summary.json"
        )
        _verify_scientific_summaries(
            evaluation=evaluation,
            performance=dict(performance.get("datasets", {})),
            compiler=compiler_summary,
            doubter=doubter_summary,
            match=match_summary,
            executor=executor_summary,
            expected_query_count=expected_query_count,
            step7b=step7b,
        )

    with _BundleVerificationCheck("model_artifact_lineage"):
        model_artifacts = read_jsonl(
            pilot_dir / "private" / "model_artifact_manifest.jsonl"
        )
        if not any(row.get("scope") == "query_runtime" for row in model_artifacts):
            raise ValueError("Step 7 bundle lacks query-time answer audit artifacts")

    if require_transfer:
        with _BundleVerificationCheck("transfer_checksum"):
            archives = sorted((pilot_dir / "transfer").glob("*.tar.zst"))
            checksums = sorted((pilot_dir / "transfer").glob("*.sha256"))
            if len(archives) != 1 or len(checksums) != 1:
                raise ValueError("Step 7 transfer archive is incomplete")
            parts = checksums[0].read_text(encoding="ascii").split()
            if len(parts) != 2 or parts[0] != file_sha256(archives[0]):
                raise ValueError("Step 7 transfer checksum mismatch")
            if parts[1] != archives[0].name:
                raise ValueError("Step 7 checksum names another archive")
    return manifest


def verify_step7_pilot_archive(
    archive: Path, checksum: Path
) -> Step7PilotManifest:
    parts = checksum.read_text(encoding="ascii").split()
    if len(parts) != 2 or parts[0] != file_sha256(archive) or parts[1] != archive.name:
        raise ValueError("Step 7 archive checksum is invalid")
    with tempfile.TemporaryDirectory(prefix="pmmc-step7-verify-") as temporary:
        root = Path(temporary)
        tar_payload = zstandard.ZstdDecompressor().decompress(archive.read_bytes())
        with tarfile.open(fileobj=io.BytesIO(tar_payload), mode="r:") as reader:
            members = reader.getmembers()
            if not members:
                raise ValueError("Step 7 archive is empty")
            roots = {PurePosixPath(member.name).parts[0] for member in members}
            if len(roots) != 1:
                raise ValueError("Step 7 archive has multiple roots")
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not member.isfile():
                    raise ValueError("Step 7 archive contains an unsafe member")
            reader.extractall(root, filter="data")
        pilot_dir = root / next(iter(roots))
        return verify_step7_pilot_bundle(pilot_dir, require_transfer=False)


def _write_freeze(path: Path, value: object, verifier: Any) -> Path:
    payload = value.model_dump(mode="json")  # type: ignore[attr-defined]
    if path.exists():
        if verifier(path) != value:
            raise ValueError("Step 7 approval conflicts with an existing freeze")
        return path
    temporary = path.with_name(f".{path.name}.staging-{os.getpid()}")
    write_json(temporary, payload)
    temporary.chmod(0o600)
    verifier(temporary)
    temporary.replace(path)
    return path


def _verify_questioner_freeze(path: Path) -> QuestionerFreezeManifest:
    value = QuestionerFreezeManifest.model_validate(read_json(path))
    if stable_hash(
        value.model_dump(mode="json", exclude={"freeze_id", "content_hash"})
    ) != value.content_hash:
        raise ValueError("Questioner freeze hash mismatch")
    return value


def verify_query_executor_freeze(path: Path) -> QueryExecutorFreezeManifest:
    value = QueryExecutorFreezeManifest.model_validate(read_json(path))
    if stable_hash(
        value.model_dump(mode="json", exclude={"freeze_manifest_id", "content_hash"})
    ) != value.content_hash:
        raise ValueError("query executor freeze hash mismatch")
    if value.freeze_manifest_id != "query-executor-freeze-" + value.content_hash[:24]:
        raise ValueError("query executor freeze ID mismatch")
    return value


def verify_step7b_freeze(path: Path) -> Step7BFreezeManifest:
    value = Step7BFreezeManifest.model_validate(read_json(path))
    if stable_hash(
        value.model_dump(mode="json", exclude={"freeze_manifest_id", "content_hash"})
    ) != value.content_hash:
        raise ValueError("Step 7B aggregate freeze hash mismatch")
    if value.freeze_manifest_id != "step7b-freeze-" + value.content_hash[:24]:
        raise ValueError("Step 7B aggregate freeze ID mismatch")
    return value


def _approved_step7b_threshold_projection(
    *,
    pilot_dir: Path,
    decision_dir: Path,
    decision_hash: str,
    match_threshold: float,
    gap_threshold: float,
) -> tuple[str, dict[str, Any]]:
    """Rebuild the approved active selection from frozen scores and answers."""

    match_rows = _read_zstd_jsonl(
        pilot_dir / "private" / "all_query_match_rerank_top50.jsonl.zst"
    )
    answer_rows = _read_zstd_jsonl(
        pilot_dir / "private" / "all_execution_answer_facts.jsonl.zst"
    )
    active_batches = {
        str(row["match_batch_id"]): (str(row["dataset"]), str(row["query_id"]))
        for row in match_rows
        if row.get("fact_type") == "qbank_match_batches"
        and int(row.get("question_budget", -1)) == 8
        and int(row.get("active_round", -1)) == 2
    }
    if not active_batches:
        raise ValueError("Step 7B approval lacks b8/r2 match batches")
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in match_rows:
        if row.get("fact_type") != "qbank_match_fused_candidates":
            continue
        key = active_batches.get(str(row.get("match_batch_id")))
        if key is not None:
            candidates[key].append(row)
    selections = {
        (str(row["dataset"]), str(row["query_id"])): row
        for row in answer_rows
        if row.get("fact_type") == "query_answer_selections"
    }
    answer_ids = {
        str(row["answer_id"])
        for row in answer_rows
        if row.get("fact_type") in {"final_answers", "pmmc_fallback_answers"}
        and row.get("answer_id")
    }
    audit_cards = read_jsonl(pilot_dir / "private" / "query_audit_cards.jsonl")
    evaluation_by_variant: dict[tuple[str, str, str], dict[str, Any]] = {}
    for card in audit_cards:
        key_prefix = (str(card.get("dataset")), str(card.get("query_id")))
        variants = card.get("variant_evaluations")
        if not isinstance(variants, list):
            raise ValueError("Step 7B approval lacks variant-aware evaluation facts")
        for evaluation in variants:
            if not isinstance(evaluation, dict):
                raise ValueError("Step 7B variant evaluation is malformed")
            evaluation_key = (*key_prefix, str(evaluation.get("answer_variant")))
            if evaluation_key in evaluation_by_variant:
                raise ValueError("Step 7B variant evaluation is duplicated")
            evaluation_by_variant[evaluation_key] = evaluation
    if set(selections) != set(active_batches.values()):
        raise ValueError("Step 7B approval answer denominator differs from b8/r2 matching")

    projected: list[dict[str, Any]] = []
    for dataset, query_id in sorted(active_batches.values()):
        rows = sorted(
            candidates.get((dataset, query_id), []),
            key=lambda row: (int(row.get("rank", 10**9)), str(row.get("qbank_id", ""))),
        )
        answerable = [row for row in rows if row.get("route_class") == "answerable"]
        controls = [row for row in rows if row.get("route_class") == "control"]

        def gate(values: Sequence[Mapping[str, Any]]) -> tuple[bool, float | None]:
            if not values:
                return False, None
            top = values[0]
            score = float(top.get("route_score") or 0.0)
            gap = (
                max(0.0, score - float(values[1].get("route_score") or 0.0))
                if len(values) > 1
                else None
            )
            passed = (
                top.get("reranker_match_level") in {"exact", "strong"}
                and top.get("modality_compatible") is True
                and score >= match_threshold
                and (gap is None or gap >= gap_threshold)
            )
            return passed, gap

        program_route, answerable_gap = gate(answerable)
        control_route, control_gap = gate(controls)
        selection = selections[(dataset, query_id)]
        pmmc_answer_id = (
            str(selection["pmmc_answer_id"])
            if selection.get("pmmc_answer_id") is not None
            else None
        )
        fallback_answer_id = (
            str(selection["fallback_answer_id"])
            if selection.get("fallback_answer_id") is not None
            else None
        )
        outcome = "program" if program_route else "fallback"
        reason: str | None = None
        selected_answer_id = pmmc_answer_id if program_route else fallback_answer_id
        selected_source_variant = "pmmc_r2" if program_route else "native_mm_counterfactual"
        if not program_route:
            if control_route:
                reason = "control_route"
            elif not answerable:
                reason = "no_qbank_candidate"
            elif answerable[0].get("modality_compatible") is not True:
                reason = "requirement_insufficient"
            elif answerable[0].get("reranker_match_level") not in {"exact", "strong"}:
                reason = "reranker_rejected"
            elif float(answerable[0].get("route_score") or 0.0) < match_threshold:
                reason = "low_match_confidence"
            else:
                reason = "ambiguous_match"
        elif selected_answer_id is None:
            outcome = "fallback"
            reason = "pmmc_answer_unavailable"
            selected_answer_id = fallback_answer_id
            selected_source_variant = "native_mm_counterfactual"
        if selected_answer_id is not None and selected_answer_id not in answer_ids:
            raise ValueError("Step 7B approved selection references an absent answer")
        evaluation = (
            evaluation_by_variant.get((dataset, query_id, selected_source_variant))
            if selected_answer_id is not None
            else None
        )
        if selected_answer_id is not None and evaluation is None:
            raise ValueError("Step 7B approved selection lacks variant evaluation")
        projected.append(
            {
                "dataset": dataset,
                "query_id": query_id,
                "decision_hash": decision_hash,
                "outcome": outcome,
                "reason": reason,
                "selected_qbank_id": (
                    str(answerable[0]["qbank_id"])
                    if outcome == "program" and answerable
                    else None
                ),
                "selected_plan_id": (
                    str(answerable[0]["plan_id"])
                    if outcome == "program" and answerable
                    else None
                ),
                "route_score": (
                    float(answerable[0].get("route_score") or 0.0)
                    if answerable
                    else None
                ),
                "top1_top2_gap": answerable_gap,
                "control_gap": control_gap,
                "selected_answer_id": selected_answer_id,
                "selected_source_variant": (
                    selected_source_variant if selected_answer_id is not None else None
                ),
                "terminal_failure": selected_answer_id is None,
                "evaluation_answer_id": (
                    str(evaluation["answer_id"]) if evaluation is not None else None
                ),
                "judge_verdict": (
                    str(evaluation["judge_verdict"])
                    if evaluation is not None
                    else None
                ),
                "judge_score": (
                    float(evaluation["judge_score"])
                    if evaluation is not None
                    and evaluation.get("judge_score") is not None
                    else None
                ),
            }
        )
    projection_hash = stable_hash(projected)
    evaluated = [row for row in projected if row["judge_verdict"] is not None]
    scores = [
        float(row["judge_score"])
        for row in evaluated
        if row["judge_score"] is not None
    ]
    summary: dict[str, Any] = {
        "schema_version": "step7b-approved-threshold-projection-v1",
        "decision_hash": decision_hash,
        "match_threshold": match_threshold,
        "gap_threshold": gap_threshold,
        "query_count": len(projected),
        "program_route_count": sum(row["outcome"] == "program" for row in projected),
        "fallback_count": sum(row["outcome"] == "fallback" for row in projected),
        "terminal_failure_count": sum(bool(row["terminal_failure"]) for row in projected),
        "evaluated_selection_count": len(evaluated),
        "judge_pass_count": sum(row["judge_verdict"] == "pass" for row in evaluated),
        "judge_pass_rate": (
            sum(row["judge_verdict"] == "pass" for row in evaluated) / len(evaluated)
            if evaluated
            else None
        ),
        "mean_judge_score": sum(scores) / len(scores) if scores else None,
        "content_hash": projection_hash,
    }
    approved_dir = decision_dir / "approved"
    approved_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    selection_path = approved_dir / "active_selections.jsonl"
    summary_path = approved_dir / "threshold_summary.json"
    write_jsonl(selection_path, projected)
    write_json(summary_path, summary)
    selection_path.chmod(0o600)
    summary_path.chmod(0o600)
    return projection_hash, summary


def _decide_step7_pilot(
    *,
    pilot_dir: Path,
    decision: PilotDecisionValue,
    reviewer_id: str,
    rationale: str,
    match_threshold: float,
    gap_threshold: float,
    expected_step7b: bool,
) -> tuple[Path, Path | None, Path | None, Path | None, Path | None]:
    """Persist an immutable decision and emit all four Step 7 freezes."""

    pilot = verify_step7_pilot_bundle(pilot_dir)
    step7b = pilot.protocol_version == _STEP7B_PROTOCOL
    if step7b != expected_step7b:
        expected_name = "Step 7B" if expected_step7b else "Step 7"
        raise ValueError(f"{expected_name} decision received another pilot protocol")
    reviewer_id = reviewer_id.strip()
    rationale = rationale.strip()
    if not reviewer_id or not rationale:
        raise ValueError("Step 7 decision requires reviewer and rationale")
    if decision is PilotDecisionValue.APPROVE and not pilot.automatic_integrity_passed:
        raise ValueError("an integrity-failing Step 7 pilot cannot be approved")
    if decision is PilotDecisionValue.APPROVE and pilot.status is not (
        Step7PilotStatus.AWAITING_THRESHOLD_REVIEW
        if step7b
        else Step7PilotStatus.AWAITING_APPROVAL
    ):
        raise ValueError("pilot is not in the required approval-review state")
    threshold_grid = [0.5, 0.6, 0.7, 0.8]
    gap_grid = [0.0, 0.05, 0.1]
    if match_threshold not in threshold_grid or gap_threshold not in gap_grid:
        raise ValueError("Step 7 thresholds must come from the frozen grids")

    decided_at = utc_now()
    decision_payload = {
        "pilot_id": pilot.pilot_id,
        "decision": decision.value,
        "reviewer_id": reviewer_id,
        "rationale": rationale,
        "pilot_content_hash": pilot.content_hash,
        "match_threshold": match_threshold,
        "gap_threshold": gap_threshold,
        "decided_at": decided_at,
    }
    decision_record = Step7PilotDecision(
        **decision_payload,
        content_hash=stable_hash(
            {
                **decision_payload,
                "decided_at": decided_at.isoformat().replace("+00:00", "Z"),
            }
        ),
    )
    decision_root = pilot_dir / "decisions"
    decision_dir = decision_root / decision_record.content_hash
    decision_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    decision_path = decision_dir / "decision.json"
    if decision_path.exists():
        stored = Step7PilotDecision.model_validate(read_json(decision_path))
        if stored != decision_record:
            raise ValueError("Step 7 pilot already has a conflicting decision")
    else:
        write_json(decision_path, decision_record.model_dump(mode="json"))
        decision_path.chmod(0o600)
    if decision is PilotDecisionValue.REJECT:
        return decision_dir, None, None, None, None

    approved_selection_hash: str | None = None
    approved_selection_summary: dict[str, Any] | None = None
    if step7b:
        approved_selection_hash, approved_selection_summary = (
            _approved_step7b_threshold_projection(
                pilot_dir=pilot_dir,
                decision_dir=decision_dir,
                decision_hash=decision_record.content_hash,
                match_threshold=match_threshold,
                gap_threshold=gap_threshold,
            )
        )

    source = read_json(pilot_dir / "private" / "source_exports.json")
    identities = [value for _key, value in sorted(source.items())]
    first = identities[0]
    questioner = dict(first["questioner"])
    program = dict(first["program"])
    semantic = dict(first["semantic"])
    for identity in identities[1:]:
        if identity["questioner"] != questioner:
            raise ValueError("Step 7 Questioner identities differ across datasets")
        comparable = {
            key: value
            for key, value in identity["program"].items()
            if key != "routing_projection_hash"
        }
        if comparable != {
            key: value
            for key, value in program.items()
            if key != "routing_projection_hash"
        }:
            raise ValueError("Step 7 Planner/Doubter identities differ across datasets")
    model_commits: dict[str, str] = {}
    for identity in identities:
        for alias, revision in dict(identity["model_commits"]).items():
            prior = model_commits.setdefault(str(alias), str(revision))
            if prior != revision:
                raise ValueError("Step 7 model revisions differ across datasets")
    audit = read_json(pilot_dir / "private" / "audit_manifest.json")
    audit_hash = str(audit["content_hash"])
    q_payload = {
        "pilot_id": pilot.pilot_id,
        "decision_hash": decision_record.content_hash,
        "partition_id": pilot.partition_id,
        "pilot_sample_ids": [item.sample_id for item in pilot.samples],
        "questioner_config_hash": questioner["questioner_config_hash"],
        "prompt_hash": questioner["prompt_hash"],
        "response_schema_hash": questioner["response_schema_hash"],
        "renderer_version": questioner["renderer_version"],
        "parser_version": questioner["parser_version"],
        "unit_builder_version": questioner["unit_builder_version"],
        "gate_version": questioner["gate_version"],
        "dedupe_version": questioner["dedupe_version"],
        "family_dedupe_version": questioner["family_dedupe_version"],
        "matcher_version": questioner["matcher_version"],
        "model_commits": model_commits,
        "runtime_export_ids": pilot.source_compiler_export_ids,
        "coverage_export_ids": pilot.source_coverage_export_ids,
        "questioner_protocol_version": questioner["questioner_protocol_version"],
        "prompt_version": questioner["prompt_version"],
        "audit_manifest_hash": audit_hash,
        "opportunity_builder_version": questioner["opportunity_builder_version"],
        "relation_selector_version": questioner["relation_selector_version"],
        "opportunity_builder_hash": questioner["opportunity_builder_hash"],
        "verifier_prompt_version": questioner["verifier_prompt_version"],
        "verifier_prompt_hash": questioner["verifier_prompt_hash"],
        "verifier_schema_hash": questioner["verifier_schema_hash"],
        "verifier_protocol_hash": questioner["verifier_protocol_hash"],
        "approved_by": reviewer_id,
        "approved_at": decided_at,
    }
    q_hash = stable_hash(
        {**q_payload, "approved_at": decided_at.isoformat().replace("+00:00", "Z")}
    )
    q_freeze = QuestionerFreezeManifest(
        freeze_id="questioner-freeze-" + q_hash[:24],
        **q_payload,
        content_hash=q_hash,
    )
    p_payload = {
        "pilot_id": pilot.pilot_id,
        "decision_hash": decision_record.content_hash,
        "program_pilot_audit_hash": audit_hash,
        "questioner_freeze_id": q_freeze.freeze_id,
        "questioner_freeze_hash": q_freeze.content_hash,
        "partition_id": pilot.partition_id,
        "pilot_sample_ids": [item.sample_id for item in pilot.samples],
        "planner_config_hash": program["planner_config_hash"],
        "planner_prompt_hash": program["planner_prompt_hash"],
        "planner_schema_hash": program["planner_schema_hash"],
        "planner_renderer_version": program["planner_renderer_version"],
        "planner_parser_version": program["planner_parser_version"],
        "program_policy_version": program["program_policy_version"],
        "operator_set_version": program["operator_set_version"],
        "operator_registry_hash": program["operator_registry_hash"],
        "dry_run_policy_version": program["dry_run_policy_version"],
        "memory_index_representation_version": program[
            "memory_index_representation_version"
        ],
        "model_commits": model_commits,
        "runtime_export_ids": pilot.source_compiler_export_ids,
        "r_max": 2,
        "evidence_lineage_version": (
            "evidence-lineage-v4" if step7b else "evidence-lineage-v3"
        ),
        "qbank_semantic_schema_version": (
            QBANK_SEMANTIC_SCHEMA_VERSION if step7b else "0.3.0"
        ),
        "semantic_judge_config_hash": semantic["semantic_config_hash"],
        "semantic_judge_prompt_hash": semantic["semantic_prompt_hash"],
        "semantic_judge_schema_hash": semantic["semantic_response_schema_hash"],
        "semantic_export_ids": pilot.source_coverage_export_ids,
        "approved_by": reviewer_id,
        "approved_at": decided_at,
        "metadata": {
            "paper_result": False,
            "active_round": 2,
            "maximum_round": 3,
            "semantic_protocol": semantic["semantic_protocol_version"],
        },
    }
    p_hash = stable_hash(
        {**p_payload, "approved_at": decided_at.isoformat().replace("+00:00", "Z")}
    )
    p_freeze = CompilerProgramFreezeManifest(
        freeze_id="compiler-program-freeze-" + p_hash[:24],
        **p_payload,
        content_hash=p_hash,
    )
    routing_hashes = [str(value["program"]["routing_projection_hash"]) for value in identities]
    v_payload = {
        "schema_version": VERIFIED_QBANK_SCHEMA_VERSION if step7b else "0.2.0",
        "run_id": pilot.pilot_id,
        "snapshot_ids": sorted(str(value["snapshot_id"]) for value in identities),
        "active_round": 2,
        "max_round": 3,
        "questioner_identity_hash": stable_hash(questioner),
        "planner_identity_hash": stable_hash(
            {key: value for key, value in program.items() if not key.startswith("doubter_")}
        ),
        "doubter_identity_hash": stable_hash(
            {key: value for key, value in program.items() if key.startswith("doubter_")}
        ),
        "operator_registry_hash": str(program["operator_registry_hash"]),
        "routing_projection_hash": stable_hash(sorted(routing_hashes)),
        "semantic_judge_identity_hash": stable_hash(
            {key: value for key, value in semantic.items() if key != "semantic_run_id"}
        ),
        "pilot_content_hash": pilot.content_hash,
        "approved_by": reviewer_id,
        "approved_at": decided_at,
        "decision_content_hash": decision_record.content_hash,
        "metadata": {
            "paper_result": False,
            "run_ids": {
                key: str(value["run_id"]) for key, value in sorted(source.items())
            },
            "questioner_freeze_manifest": q_freeze.freeze_id,
            "compiler_program_freeze_manifest": p_freeze.freeze_id,
        },
    }
    v_hash = stable_hash(
        {**v_payload, "approved_at": decided_at.isoformat().replace("+00:00", "Z")}
    )
    v_freeze = VerifiedQBankFreezeManifest(
        freeze_manifest_id="verified-qbank-freeze-" + v_hash[:24],
        **v_payload,
        content_hash=v_hash,
    )
    query_identities = [dict(value["query_executor"]) for value in identities]
    if len({str(value["config_hash"]) for value in query_identities}) != 1:
        raise ValueError("Step 7 query executor config differs across datasets")
    answer_prompt_hashes = {
        str(dict(value.get("metadata", {})).get("answer_prompt_hash"))
        for value in query_identities
    }
    if len(answer_prompt_hashes) != 1:
        raise ValueError("Step 7 answer prompt differs across datasets")
    query_metadata = [dict(value.get("metadata", {})) for value in query_identities]
    protocol_hashes = {
        str(value.get("query_executor_protocol_hash") or "")
        for value in query_metadata
    }
    query_executor_protocol_hash: str | None = None
    approved_routing_policy_hash: str | None = None
    reranker_identity_hash: str | None = None
    answer_contract_hash: str | None = None
    if step7b:
        if len(protocol_hashes) != 1 or not next(iter(protocol_hashes)):
            raise ValueError("Step 7B query executor protocol identity is incomplete")
        query_executor_protocol_hash = next(iter(protocol_hashes))
        approved_routing_policy_hash = routing_policy_hash_from_protocol(
            query_executor_protocol_hash,
            match_threshold=match_threshold,
            gap_threshold=gap_threshold,
        )
        identity_fields = (
            "matcher_version",
            "graph_version",
            "sufficiency_policy_version",
            "reranker_version",
            "reranker_prompt_version",
            "reranker_model_alias",
            "reranker_schema_hash",
            "answer_contract_version",
            "answer_output_schema_version",
            "answer_output_schema_hash",
        )
        normalized = [
            {name: metadata.get(name) for name in identity_fields}
            for metadata in query_metadata
        ]
        if any(value != normalized[0] for value in normalized[1:]):
            raise ValueError("Step 7B reranker/answer identities differ across datasets")
        expected = {
            "matcher_version": "pmmc-qbank-matcher-v2",
            "graph_version": "pmmc-online-query-graph-v2",
            "sufficiency_policy_version": "query-runtime-evidence-v2",
            "reranker_version": "pmmc-bounded-reranker-v1",
            "reranker_prompt_version": "pmmc-qbank-reranker-v1",
            "reranker_model_alias": "reranker-qwen3.5-9b",
            "answer_contract_version": "strict-answer-contract-v2",
            "answer_output_schema_version": ANSWER_OUTPUT_SCHEMA_VERSION,
        }
        if any(normalized[0].get(name) != value for name, value in expected.items()):
            raise ValueError("Step 7B query runtime identity is not frozen protocol v2")
        reranker_identity_hash = stable_hash(
            {
                name: normalized[0][name]
                for name in (
                    "reranker_version",
                    "reranker_prompt_version",
                    "reranker_model_alias",
                    "reranker_schema_hash",
                )
            }
        )
        answer_contract_hash = stable_hash(
            {
                "answer_contract_version": normalized[0]["answer_contract_version"],
                "answer_output_schema_version": normalized[0][
                    "answer_output_schema_version"
                ],
                "answer_output_schema_hash": normalized[0][
                    "answer_output_schema_hash"
                ],
                "answer_prompt_hash": next(iter(answer_prompt_hashes)),
                "query_executor_protocol_hash": query_executor_protocol_hash,
            }
        )
    x_payload = {
        "schema_version": QUERY_EXECUTOR_SCHEMA_VERSION if step7b else "0.1.0",
        "verified_qbank_freeze_manifest_id": v_freeze.freeze_manifest_id,
        "query_executor_config_hash": str(query_identities[0]["config_hash"]),
        "query_executor_protocol_hash": query_executor_protocol_hash,
        "approved_routing_policy_hash": approved_routing_policy_hash,
        "approved_selection_hash": approved_selection_hash,
        "answer_prompt_hash": next(iter(answer_prompt_hashes)),
        "reranker_identity_hash": reranker_identity_hash,
        "runtime_evidence_policy_version": (
            "query-runtime-evidence-v2" if step7b else None
        ),
        "answer_contract_version": "strict-answer-contract-v2" if step7b else None,
        "answer_output_schema_version": ANSWER_OUTPUT_SCHEMA_VERSION if step7b else None,
        "answer_contract_hash": answer_contract_hash,
        "fallback_policy_version": "native_mm_rag_v1",
        "active_round": 2,
        "question_budget": 8,
        "match_threshold": match_threshold,
        "gap_threshold": gap_threshold,
        "threshold_grid": threshold_grid,
        "gap_grid": gap_grid,
        "pilot_content_hash": pilot.content_hash,
        "approved_by": reviewer_id,
        "approved_at": decided_at,
        "decision_content_hash": decision_record.content_hash,
        "metadata": {
            "paper_result": False,
            "query_executor_run_ids": pilot.query_executor_run_ids,
            "active_round": 2,
            "maximum_round": 3,
            "approved_selection_summary": approved_selection_summary,
        },
    }
    x_hash = stable_hash(
        {**x_payload, "approved_at": decided_at.isoformat().replace("+00:00", "Z")}
    )
    x_freeze = QueryExecutorFreezeManifest(
        freeze_manifest_id="query-executor-freeze-" + x_hash[:24],
        **x_payload,
        content_hash=x_hash,
    )
    q_path = _write_freeze(
        decision_dir / "questioner_freeze_manifest.json",
        q_freeze,
        _verify_questioner_freeze,
    )
    p_path = _write_freeze(
        decision_dir / "compiler_program_freeze_manifest.json",
        p_freeze,
        verify_program_freeze,
    )
    v_path = _write_freeze(
        decision_dir / "verified_qbank_freeze_manifest.json",
        v_freeze,
        verify_verified_qbank_freeze,
    )
    x_path = _write_freeze(
        decision_dir / "query_executor_freeze_manifest.json",
        x_freeze,
        verify_query_executor_freeze,
    )
    return decision_dir, q_path, p_path, v_path, x_path


def decide_step7_pilot(
    *,
    pilot_dir: Path,
    decision: PilotDecisionValue,
    reviewer_id: str,
    rationale: str,
    match_threshold: float,
    gap_threshold: float,
) -> tuple[Path, Path | None, Path | None, Path | None, Path | None]:
    """Decide a historical Step 7 pilot without reinterpreting its versions."""

    return _decide_step7_pilot(
        pilot_dir=pilot_dir,
        decision=decision,
        reviewer_id=reviewer_id,
        rationale=rationale,
        match_threshold=match_threshold,
        gap_threshold=gap_threshold,
        expected_step7b=False,
    )


def decide_step7b_pilot(
    *,
    pilot_dir: Path,
    decision: PilotDecisionValue,
    reviewer_id: str,
    rationale: str,
    match_threshold: float,
    gap_threshold: float,
) -> tuple[Path, Path | None, Path | None, Path | None, Path | None, Path | None]:
    """Decide Step 7B and emit its four child freezes plus aggregate freeze."""

    decision_dir, q_path, p_path, v_path, x_path = _decide_step7_pilot(
        pilot_dir=pilot_dir,
        decision=decision,
        reviewer_id=reviewer_id,
        rationale=rationale,
        match_threshold=match_threshold,
        gap_threshold=gap_threshold,
        expected_step7b=True,
    )
    if decision is PilotDecisionValue.REJECT:
        return decision_dir, q_path, p_path, v_path, x_path, None
    assert q_path is not None
    assert p_path is not None
    assert v_path is not None
    assert x_path is not None
    questioner = _verify_questioner_freeze(q_path)
    program = verify_program_freeze(p_path)
    qbank = verify_verified_qbank_freeze(v_path)
    executor = verify_query_executor_freeze(x_path)
    if executor.answer_contract_hash is None:
        raise ValueError("Step 7B executor freeze lacks its answer contract hash")
    decision_record = Step7PilotDecision.model_validate(
        read_json(decision_dir / "decision.json")
    )
    payload = {
        "pilot_id": decision_record.pilot_id,
        "schema_version": STEP7B_PILOT_SCHEMA_VERSION,
        "questioner_freeze_hash": questioner.content_hash,
        "compiler_program_freeze_hash": program.content_hash,
        "verified_qbank_freeze_hash": qbank.content_hash,
        "query_executor_freeze_hash": executor.content_hash,
        "answer_contract_hash": executor.answer_contract_hash,
        "active_round": 2,
        "question_budget": 8,
        "match_threshold": match_threshold,
        "gap_threshold": gap_threshold,
        "approved_by": reviewer_id.strip(),
        "approved_at": questioner.approved_at,
        "decision_content_hash": decision_record.content_hash,
        "metadata": {
            "paper_result": False,
            "questioner_freeze_id": questioner.freeze_id,
            "compiler_program_freeze_id": program.freeze_id,
            "verified_qbank_freeze_id": qbank.freeze_manifest_id,
            "query_executor_freeze_id": executor.freeze_manifest_id,
            "step8_consumable": True,
        },
    }
    content_hash = stable_hash(
        {
            **payload,
            "approved_at": questioner.approved_at.isoformat().replace("+00:00", "Z"),
        }
    )
    freeze = Step7BFreezeManifest(
        freeze_manifest_id="step7b-freeze-" + content_hash[:24],
        **payload,
        content_hash=content_hash,
    )
    freeze_path = _write_freeze(
        decision_dir / "step7b_freeze_manifest.json",
        freeze,
        verify_step7b_freeze,
    )
    return decision_dir, q_path, p_path, v_path, x_path, freeze_path


__all__ = [
    "build_step7_pilot_bundle",
    "decide_step7b_pilot",
    "decide_step7_pilot",
    "verify_step7b_freeze",
    "verify_query_executor_freeze",
    "verify_step7_pilot_archive",
    "verify_step7_pilot_bundle",
]
