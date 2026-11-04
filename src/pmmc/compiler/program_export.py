"""Offline integrity verification for Planner v1 runtime exports."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import orjson

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.memory_index import (
    TEXT_MEMORY_REPRESENTATION,
    VISUAL_MEMORY_REPRESENTATION,
)
from pmmc.compiler.planner_config import PlannerConfig
from pmmc.compiler.planner_prompt import (
    planner_prompt_sha256,
    planner_prompt_text,
    planner_request_response_schema,
    planner_response_schema_sha256,
)
from pmmc.compiler.questioner_export import verify_questioner_export
from pmmc.infrastructure.export_verification import (
    PROGRAM_RUNTIME_EXPORT_VERSION,
    STEP7_PROGRAM_RUNTIME_EXPORT_VERSION,
    STEP7B_PROGRAM_RUNTIME_EXPORT_VERSION,
    SUPPORTED_PROGRAM_RUNTIME_EXPORT_VERSIONS,
)
from pmmc.schemas import PlannerRequirementContract


class ProgramExportVerificationError(ValueError):
    """Fixed diagnostic code; never forward candidate content or provider errors."""

    def __init__(self, phase: str) -> None:
        self.error_code = f"program_export_{phase}_failed"
        self.database_diagnostics = {"error_code": self.error_code}
        super().__init__(self.error_code)


@contextmanager
def _program_export_check(phase: str) -> Iterator[None]:
    started = time.monotonic()
    try:
        yield
    except ProgramExportVerificationError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        error = ProgramExportVerificationError(phase)
        print(
            f"PROGRAM_EXPORT_CHECK phase={phase} state=failed "
            f"error_code={error.error_code} error_type={type(exc).__name__}",
            flush=True,
        )
        raise error from exc
    else:
        print(
            f"PROGRAM_EXPORT_CHECK phase={phase} state=ready "
            f"elapsed_seconds={time.monotonic() - started:.3f}",
            flush=True,
        )


_REQUIRED_PROGRAM_FILES = (
    "compiler_stage_jobs.jsonl",
    "planner_generation_batches.jsonl",
    "planner_candidate_outcomes.jsonl",
    "planner_batch_programs.jsonl",
    "program_validation_results.jsonl",
    "program_validation_issues.jsonl",
    "memory_operator_index_manifests.jsonl",
    "program_query_embedding_batches.jsonl",
    "program_query_embeddings.jsonl",
    "program_dry_run_assessments.jsonl",
    "program_dry_run_evidence_matches.jsonl",
)
_REQUIRED_PROGRAM_V2_FILES = (
    "planner_requirement_contracts.jsonl",
    "planner_candidate_drafts.jsonl",
    "structured_output_item_diagnostics.jsonl",
)
_REQUIRED_STEP6D_FILES = (
    "questioner_answer_claims.jsonl",
    "qbank_candidate_families.jsonl",
    "qbank_candidate_family_members.jsonl",
    "r0_input_manifests.jsonl",
    "candidate_stage_jobs.jsonl",
    "doubter_assessments.jsonl",
    "doubter_assessment_dimensions.jsonl",
    "doubter_directives.jsonl",
    "planner_revision_lineage.jsonl",
    "qbank_round_projection_manifests.jsonl",
    "qbank_round_projection_members.jsonl",
    "routing_projection_manifests.jsonl",
    "routing_projection_members.jsonl",
    "stage_component_manifests.jsonl",
    "qbank_decisions.jsonl",
)
_REQUIRED_STEP7B_FILES = ("planner_revision_patches.jsonl",)

_PROGRAM_OUTCOMES = {
    "program_ready",
    "program_ready_with_fallback",
    "program_ready_with_abstain",
    "lineage_rejected",
    "query_vector_unavailable",
    "execution_failed",
    "dry_run_insufficient",
}
_READY_OUTCOMES = {
    "program_ready",
    "program_ready_with_fallback",
    "program_ready_with_abstain",
}

_PLANNER_REQUEST_KEYS = frozenset(
    {
        "request_id",
        "model_alias",
        "messages",
        "max_output_tokens",
        "temperature",
        "top_p",
        "seed",
        "timeout_seconds",
        "response_format",
        "metadata",
    }
)
_PLANNER_ORACLE_FIELDS = frozenset(
    {
        "expected_answer",
        "expected_evidence",
        "expected_evidence_ids",
        "evidence_ids",
        "source_memory_ids",
        "source_session_ids",
        "source_image_ids",
    }
)
_DOUBTER_DIMENSION_NAMES = frozenset(
    {
        "candidate_supported",
        "answer_entailed",
        "path_sufficient",
        "modality_sufficient",
        "raw_image_sufficient",
        "context_sufficient",
        "multi_hop_sufficient",
        "temporal_consistent",
        "non_redundant",
        "refusal_appropriate",
    }
)


class _PlannerRequestAuditError(ValueError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


def _planner_audit_event(phase: str, **values: object) -> None:
    payload = {"phase": phase, "stage": "planner_non_oracle_audit", **values}
    print(
        "PLANNER_REQUEST_AUDIT "
        + orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode("utf-8"),
        flush=True,
    )


def _contains_oracle_field(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).strip().lower() in _PLANNER_ORACLE_FIELDS
            or _contains_oracle_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_oracle_field(item) for item in value)
    return False


def _verify_hardened_planner_request_artifact(
    payload: bytes,
    *,
    batch: dict[str, Any],
    questions_by_id: dict[str, dict[str, Any]],
    requirements_by_question: dict[str, dict[str, Any]],
    config: PlannerConfig,
) -> None:
    """Prove that a Planner request is exactly the non-oracle renderer projection."""

    try:
        manifest = orjson.loads(payload)
    except orjson.JSONDecodeError as exc:
        raise _PlannerRequestAuditError("invalid_json") from exc
    if not isinstance(manifest, dict) or set(manifest) != _PLANNER_REQUEST_KEYS:
        raise _PlannerRequestAuditError("request_shape_mismatch")
    if _contains_oracle_field(manifest):
        raise _PlannerRequestAuditError("oracle_field_present")

    model_call_id = str(batch.get("model_call_id", ""))
    if (
        manifest.get("request_id") != model_call_id
        or manifest.get("model_alias") != config.planner_model_alias
        or manifest.get("max_output_tokens") != config.reserved_output_tokens
        or manifest.get("temperature") != 0.0
        or manifest.get("top_p") != 1.0
        or manifest.get("seed") != 0
        or (
            manifest.get("timeout_seconds") is not None
            and not isinstance(manifest.get("timeout_seconds"), (int, float))
        )
    ):
        raise _PlannerRequestAuditError("request_identity_mismatch")

    candidate_ids = [str(value) for value in batch.get("candidate_qbank_ids", [])]
    batch_ordinal = int(batch.get("batch_ordinal", -1))
    aliases = (
        ["Q01"]
        if (config.is_v12 or config.is_v13) and len(candidate_ids) == 1
        else [
            f"Q{batch_ordinal * config.candidates_per_batch + index + 1:02d}"
            for index in range(len(candidate_ids))
        ]
    )
    if not aliases:
        raise _PlannerRequestAuditError("candidate_scope_mismatch")
    expected_contract_hashes: dict[str, str] = {}
    expected_parts: list[dict[str, object]] = []
    for alias, qbank_id in zip(aliases, candidate_ids, strict=True):
        question = questions_by_id.get(qbank_id)
        requirement = requirements_by_question.get(qbank_id)
        if question is None:
            raise _PlannerRequestAuditError("candidate_scope_mismatch")
        if requirement is None:
            raise _PlannerRequestAuditError("requirement_scope_mismatch")
        expected_contract_hashes[alias] = str(requirement["content_hash"])
        requirement_payload = {
            "required_modalities": requirement["required_modalities"],
            "minimum_evidence_groups": requirement["minimum_evidence_groups"],
            "required_capabilities": requirement["required_capabilities"],
            "fallback_required": requirement["fallback_required"],
            "refusal_only": requirement["refusal_only"],
        }
        if config.is_v13:
            requirement_payload["allowed_capabilities"] = requirement[
                "allowed_capabilities"
            ]
        expected_parts.append(
            {
                "candidate_alias": alias,
                "question": question["canonical_question"],
                "question_type": question["question_type"],
                "reasoning_type": question["reasoning_type"],
                "requirement_contract": requirement_payload,
            }
        )

    request_requirement = (
        PlannerRequirementContract.model_validate(
            requirements_by_question[candidate_ids[0]]
        )
        if config.is_v13
        else None
    )
    request_schema = planner_request_response_schema(
        config,
        candidate_alias=aliases[0],
        requirement_contract=request_requirement,
    )
    expected_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": (
                "pmmc_planner_strategy_v1_4"
                if config.is_v14
                else "pmmc_planner_strategy_v1_3"
                if config.is_v13
                else "pmmc_planner_strategy_v1_2"
                if config.is_v12
                else "pmmc_planner_program_batch_v1"
            ),
            "strict": True,
            "schema": request_schema,
        },
    }
    if manifest.get("response_format") != expected_response_format:
        raise _PlannerRequestAuditError("response_schema_mismatch")
    request_schema_hash = hashlib.sha256(
        json.dumps(
            request_schema,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected_metadata = {
        "task": "pmmc_planner",
        "prompt_version": config.prompt_version,
        "renderer_version": config.renderer_version,
        "response_schema_hash": planner_response_schema_sha256(config),
        "request_response_schema_hash": request_schema_hash,
        "candidate_count": len(candidate_ids),
        "non_oracle_input": True,
        "requirement_contract_hashes": expected_contract_hashes,
        **(
            {"single_candidate_scope": True}
            if config.is_v12 or config.is_v13
            else {}
        ),
    }
    if manifest.get("metadata") != expected_metadata:
        raise _PlannerRequestAuditError("request_metadata_mismatch")

    messages = manifest.get("messages")
    expected_system = {
        "role": "system",
        "name": None,
        "parts": [{"kind": "text", "text": planner_prompt_text(config)}],
    }
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or messages[0] != expected_system
    ):
        raise _PlannerRequestAuditError("message_scope_mismatch")
    user = messages[1]
    if (
        not isinstance(user, dict)
        or set(user) != {"role", "name", "parts"}
        or user.get("role") != "user"
        or user.get("name") is not None
        or not isinstance(user.get("parts"), list)
        or len(user["parts"]) != len(expected_parts)
    ):
        raise _PlannerRequestAuditError("user_message_shape_mismatch")
    actual_parts: list[dict[str, object]] = []
    for part in user["parts"]:
        if (
            not isinstance(part, dict)
            or set(part) != {"kind", "text"}
            or part.get("kind") != "text"
            or not isinstance(part.get("text"), str)
        ):
            raise _PlannerRequestAuditError("unsafe_user_part")
        try:
            value = orjson.loads(part["text"])
        except orjson.JSONDecodeError as exc:
            raise _PlannerRequestAuditError("invalid_candidate_projection") from exc
        if not isinstance(value, dict):
            raise _PlannerRequestAuditError("invalid_candidate_projection")
        if _contains_oracle_field(value):
            raise _PlannerRequestAuditError("oracle_field_present")
        actual_parts.append(value)
    if actual_parts != expected_parts:
        raise _PlannerRequestAuditError("candidate_projection_mismatch")


def _read_json(path: Path) -> dict[str, Any]:
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"program export file must contain an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = orjson.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"program export JSONL row is not an object: {path.name}:{line_number}"
                )
            rows.append(value)
    return rows


def _strings(value: object) -> set[str]:
    result: set[str] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            result.add(item)
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return result


def _assert_run_scope(rows: list[dict[str, Any]], run_id: str, label: str) -> None:
    if any(str(row.get("run_id", "")) != run_id for row in rows):
        raise ValueError(f"program {label} crosses compiler-run scope")


def _step6d_payload(
    value: Mapping[str, Any],
    fields: Sequence[str],
) -> dict[str, Any]:
    return {field: value.get(field) for field in fields}


_R0_IDENTITY_FIELDS_V1 = (
    "run_id",
    "snapshot_id",
    "candidate_count",
    "answerable_candidate_count",
    "control_candidate_count",
    "program_count",
    "execution_count",
    "terminal_outcome_count",
    "questioner_identity_hash",
    "planner_identity_hash",
    "operator_registry_hash",
    "memory_index_hash",
    "query_vector_hash",
    "facts_hash",
    "artifact_hash",
    "hard_gate_passed",
    "issue_codes",
)
_R0_IDENTITY_FIELDS_V2 = (
    *_R0_IDENTITY_FIELDS_V1,
    "protocol_integrity_issues",
    "candidate_scientific_rejections",
    "coverage_zero_units",
    "downstream_viability",
)


def _step6d_r0_identity_payload(r0: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = str(r0.get("schema_version", ""))
    fields: Sequence[str]
    if schema_version == "0.1.0":
        fields = _R0_IDENTITY_FIELDS_V1
    elif schema_version in {"0.2.0", "0.3.0"}:
        # Step 7B changes R0 policy, not the hashed field set. Do not rewrite
        # historical payloads or include schema/metadata in their producer hash.
        fields = _R0_IDENTITY_FIELDS_V2
    else:
        raise ValueError("Step 6D R0 manifest schema version is unsupported")
    return _step6d_payload(r0, fields)


def _verify_step6d_eligible_accounting(
    *,
    questioner_report: Mapping[str, Any],
    candidate_ids: set[str],
    r0_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Close the final Questioner denominator over the immutable R0 input."""

    questioner_count = int(questioner_report.get("eligible_candidate_count", -1))
    candidate_count = len(candidate_ids)
    r0_candidate_count = (
        int(r0_rows[0].get("candidate_count", -1)) if len(r0_rows) == 1 else -1
    )
    r0_hard_gate_passed = (
        r0_rows[0].get("hard_gate_passed") is True if len(r0_rows) == 1 else False
    )
    accounting = {
        "gate_verifier_eligible_count": candidate_count,
        "questioner_eligible_candidate_count": questioner_count,
        "r0_candidate_count": r0_candidate_count,
        "r0_hard_gate_passed": r0_hard_gate_passed,
        "r0_manifest_count": len(r0_rows),
        "r0_schema_version": str(r0_rows[0].get("schema_version", "missing"))
        if len(r0_rows) == 1 else "missing",
    }
    print(
        "STEP6D_EXPORT_ELIGIBLE_ACCOUNTING "
        + orjson.dumps(accounting, option=orjson.OPT_SORT_KEYS).decode("utf-8"),
        flush=True,
    )
    if questioner_count != candidate_count:
        raise ValueError("Step 6D eligible candidate count differs from Questioner verification")
    if (
        len(r0_rows) != 1
        or not r0_hard_gate_passed
        or r0_candidate_count != candidate_count
    ):
        raise ValueError("Step 6D immutable R0 input gate is incomplete")


def _verify_step6d_projection_identities(
    *,
    run_id: str,
    candidates: Mapping[str, dict[str, Any]],
    r0: dict[str, Any],
    manifests: Mapping[int, dict[str, Any]],
    members: Mapping[int, Mapping[str, dict[str, Any]]],
    decisions: Mapping[str, dict[str, Any]],
    routing: dict[str, Any],
    routing_members: Sequence[dict[str, Any]],
) -> None:
    r0_payload = _step6d_r0_identity_payload(r0)
    identity_match = r0.get("r0_manifest_id") == stable_compiler_id(
        "r0-input-manifest", r0_payload
    )
    content_hash_match = r0.get("content_hash") == canonical_content_hash(
        r0_payload
    )
    r0_identity_accounting = {
        "candidate_scientific_rejection_bucket_count": len(
            dict(r0.get("candidate_scientific_rejections") or {})
        ),
        "content_hash_match": content_hash_match,
        "coverage_zero_units": int(r0.get("coverage_zero_units", 0)),
        "downstream_viability": r0.get("downstream_viability") is True,
        "identity_match": identity_match,
        "payload_field_count": len(r0_payload),
        "protocol_integrity_issue_count": len(
            list(r0.get("protocol_integrity_issues") or [])
        ),
        "schema_version": str(r0.get("schema_version", "")),
    }
    print(
        "STEP6D_EXPORT_R0_IDENTITY_ACCOUNTING "
        + orjson.dumps(
            r0_identity_accounting,
            option=orjson.OPT_SORT_KEYS,
        ).decode("utf-8"),
        flush=True,
    )
    if not identity_match or not content_hash_match:
        raise ValueError("Step 6D R0 manifest identity or content hash is invalid")

    for round_id in range(4):
        manifest = manifests[round_id]
        projection_payload = {
            "run_id": run_id,
            "snapshot_id": r0["snapshot_id"],
            "round_id": round_id,
            "source_r0_manifest_id": r0["r0_manifest_id"],
            "r0_facts_hash": r0["facts_hash"],
        }
        projection_id = stable_compiler_id(
            "qbank-round-projection",
            projection_payload,
        )
        if (
            manifest.get("projection_id") != projection_id
            or manifest.get("snapshot_id") != r0["snapshot_id"]
            or manifest.get("source_r0_manifest_id") != r0["r0_manifest_id"]
            or dict(manifest.get("metadata") or {}).get("r0_facts_hash")
            != r0["facts_hash"]
        ):
            raise ValueError("Step 6D round projection identity is invalid")

        ordered_members = [
            members[round_id][qbank_id] for qbank_id in sorted(candidates)
        ]
        accepted_count = 0
        rejected_count = 0
        for value in ordered_members:
            metadata = dict(value.get("metadata") or {})
            reason = metadata.get("reason_code")
            member_payload = {
                "projection_id": projection_id,
                "run_id": run_id,
                "qbank_id": value.get("qbank_id"),
                "route_class": value.get("route_class"),
                "outcome": value.get("outcome"),
                "effective_plan_id": value.get("effective_plan_id"),
                "effective_feedback_id": value.get("effective_feedback_id"),
                "passed_at_round": value.get("passed_at_round"),
                "reason": reason,
            }
            if (
                value.get("projection_member_id")
                != stable_compiler_id(
                    "qbank-round-projection-member",
                    member_payload,
                )
                or value.get("content_hash")
                != canonical_content_hash(member_payload)
            ):
                raise ValueError("Step 6D round projection member hash is invalid")
            outcome = str(value.get("outcome"))
            accepted_count += outcome in {"accepted", "carried_forward"}
            rejected_count += outcome in {
                "rejected",
                "max_rounds_exhausted",
            }

        members_hash = canonical_content_hash(ordered_members)
        manifest_payload = {
            "run_id": run_id,
            "snapshot_id": r0["snapshot_id"],
            "round_id": round_id,
            "source_r0_manifest_id": r0["r0_manifest_id"],
            "projection_id": projection_id,
            "member_count": len(ordered_members),
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "pending_count": len(ordered_members)
            - accepted_count
            - rejected_count,
            "members_hash": members_hash,
        }
        if (
            _step6d_payload(manifest, tuple(manifest_payload))
            != manifest_payload
            or manifest.get("content_hash")
            != canonical_content_hash(manifest_payload)
        ):
            raise ValueError("Step 6D round projection manifest hash is invalid")

    ordered_decisions = [decisions[qbank_id] for qbank_id in sorted(candidates)]
    for decision in ordered_decisions:
        qbank_id = str(decision["qbank_id"])
        source_member = members[2][qbank_id]
        decision_payload = {
            key: value
            for key, value in decision.items()
            if key not in {"created_at", "metadata", "decision_hash"}
        }
        expected_decision_id = stable_compiler_id(
            "qbank-decision",
            [run_id, qbank_id, 2],
        )
        if (
            decision.get("decision_id") != expected_decision_id
            or decision.get("decision_hash")
            != canonical_content_hash(decision_payload)
            or dict(decision.get("metadata") or {}).get(
                "source_projection_member_id"
            )
            != source_member["projection_member_id"]
        ):
            raise ValueError("Step 6D active-r2 decision identity is invalid")

    decision_hash = canonical_content_hash(
        [value["decision_hash"] for value in ordered_decisions]
    )
    source_projection_id = str(manifests[2]["projection_id"])
    routing_id = stable_compiler_id(
        "routing-projection",
        [run_id, source_projection_id, decision_hash],
    )
    if (
        routing.get("routing_projection_id") != routing_id
        or routing.get("source_projection_id") != source_projection_id
        or routing.get("decision_hash") != decision_hash
        or routing.get("snapshot_id") != r0["snapshot_id"]
    ):
        raise ValueError("Step 6D routing projection identity is invalid")

    ordered_routing = sorted(
        routing_members,
        key=lambda value: int(value["ordinal"]),
    )
    for ordinal, value in enumerate(ordered_routing):
        qbank_id = str(value["qbank_id"])
        source_member = members[2][qbank_id]
        if (
            int(value["ordinal"]) != ordinal
            or value.get("routing_projection_id") != routing_id
            or value.get("source_projection_member_id")
            != source_member["projection_member_id"]
            or value.get("plan_id") != source_member.get("effective_plan_id")
            or value.get("route_class") != source_member.get("route_class")
            or value.get("routing_member_id")
            != stable_compiler_id(
                "routing-projection-member",
                [routing_id, qbank_id, ordinal],
            )
            or value.get("content_hash")
            != canonical_content_hash(
                [
                    routing_id,
                    qbank_id,
                    value.get("plan_id"),
                    value.get("route_class"),
                    ordinal,
                ]
            )
        ):
            raise ValueError("Step 6D routing projection member identity is invalid")

    members_hash = canonical_content_hash(ordered_routing)
    source_index_id = dict(routing.get("metadata") or {}).get(
        "source_qbank_index_id"
    )
    routing_payload = {
        "routing_projection_id": routing_id,
        "run_id": run_id,
        "snapshot_id": r0["snapshot_id"],
        "active_round": 2,
        "max_round": 3,
        "source_projection_id": source_projection_id,
        "decision_hash": decision_hash,
        "lexical_collection_id": stable_compiler_id(
            "qbank-lexical-projection",
            [run_id, source_index_id, source_projection_id],
        ),
        "text_collection_id": routing.get("text_collection_id"),
        "multimodal_collection_id": routing.get(
            "multimodal_collection_id"
        ),
        "member_count": len(ordered_routing),
        "answerable_count": sum(
            value.get("route_class") == "answerable"
            for value in ordered_routing
        ),
        "control_count": sum(
            value.get("route_class") == "control"
            for value in ordered_routing
        ),
        "members_hash": members_hash,
    }
    if (
        not isinstance(source_index_id, str)
        or not source_index_id
        or _step6d_payload(routing, tuple(routing_payload))
        != routing_payload
        or routing.get("content_hash")
        != canonical_content_hash(routing_payload)
        or dict(routing.get("metadata") or {}).get("embedding_recomputed")
        is not False
    ):
        raise ValueError("Step 6D routing projection manifest hash is invalid")


def _verify_step6d_doubter_chain(
    *,
    run_id: str,
    candidates: Mapping[str, dict[str, Any]],
    programs: Mapping[str, dict[str, Any]],
    executions: Mapping[str, dict[str, Any]],
    dry_runs: Mapping[str, dict[str, Any]],
    assessment_rows: Sequence[dict[str, Any]],
    dimension_rows: Sequence[dict[str, Any]],
    directive_rows: Sequence[dict[str, Any]],
    revision_rows: Sequence[dict[str, Any]],
    candidate_job_rows: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """Reconstruct and verify the append-only candidate graph from JSONL facts."""

    assessments: dict[tuple[str, int], dict[str, Any]] = {}
    assessment_by_id: dict[str, dict[str, Any]] = {}
    assessment_by_feedback: dict[str, dict[str, Any]] = {}
    for row in assessment_rows:
        qbank_id = str(row.get("qbank_id") or "")
        round_id = int(row.get("round_id", -1))
        assessment_id = str(row.get("assessment_id") or "")
        feedback_id = str(row.get("feedback_id") or "")
        key = (qbank_id, round_id)
        if (
            row.get("run_id") != run_id
            or qbank_id not in candidates
            or not 0 <= round_id <= 3
            or not assessment_id
            or not feedback_id
            or key in assessments
            or assessment_id in assessment_by_id
            or feedback_id in assessment_by_feedback
        ):
            raise ValueError("Doubter assessment scope, identity, or round is invalid")
        assessments[key] = row
        assessment_by_id[assessment_id] = row
        assessment_by_feedback[feedback_id] = row

    dimensions_by_assessment: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in dimension_rows:
        assessment_id = str(row.get("assessment_id") or "")
        dimension = str(row.get("dimension") or "")
        passed = row.get("passed")
        parent = assessment_by_id.get(assessment_id)
        if (
            parent is None
            or row.get("run_id") != run_id
            or dimension not in _DOUBTER_DIMENSION_NAMES
            or dimension in dimensions_by_assessment[assessment_id]
            or not isinstance(passed, bool)
        ):
            raise ValueError("Doubter dimension scope or identity is invalid")
        dimensions_by_assessment[assessment_id][dimension] = passed
    if any(
        set(dimensions_by_assessment.get(assessment_id, {}))
        != _DOUBTER_DIMENSION_NAMES
        for assessment_id in assessment_by_id
    ):
        raise ValueError("each Doubter assessment must contain all ten dimensions")

    directives_by_feedback: dict[str, list[dict[str, Any]]] = defaultdict(list)
    directive_ids: set[str] = set()
    for row in directive_rows:
        feedback_id = str(row.get("feedback_id") or "")
        directive_id = str(row.get("directive_id") or "")
        parent = assessment_by_feedback.get(feedback_id)
        if (
            parent is None
            or row.get("run_id") != run_id
            or row.get("qbank_id") != parent.get("qbank_id")
            or int(row.get("round_id", -1))
            != int(parent.get("round_id", -2)) + 1
            or not directive_id
            or directive_id in directive_ids
        ):
            raise ValueError("Doubter directive lineage is invalid")
        directive_ids.add(directive_id)
        directives_by_feedback[feedback_id].append(row)

    for feedback_id, rows in directives_by_feedback.items():
        rows.sort(key=lambda value: int(value.get("ordinal", -1)))
        if [int(value.get("ordinal", -1)) for value in rows] != list(
            range(len(rows))
        ):
            raise ValueError("Doubter directive ordinals are not contiguous")
        for ordinal, row in enumerate(rows):
            top_k = row.get("top_k_tier")
            expected_id = stable_compiler_id(
                "doubter-directive",
                [
                    feedback_id,
                    row["round_id"],
                    ordinal,
                    row["directive"],
                    top_k,
                ],
            )
            payload = _step6d_payload(
                row,
                (
                    "feedback_id",
                    "run_id",
                    "qbank_id",
                    "round_id",
                    "ordinal",
                    "directive",
                    "top_k_tier",
                ),
            )
            if (
                row.get("directive_id") != expected_id
                or row.get("content_hash") != canonical_content_hash(payload)
            ):
                raise ValueError("Doubter directive identity or hash is invalid")

    jobs: dict[tuple[str, str, int], dict[str, Any]] = {}
    unavailable_targets: set[tuple[str, int]] = set()
    for row in candidate_job_rows:
        qbank_id = str(row.get("qbank_id") or "")
        stage = str(row.get("stage") or "")
        round_id = int(row.get("round_id", -1))
        job_key = (qbank_id, stage, round_id)
        status = str(row.get("status") or "")
        metadata = dict(row.get("metadata") or {})
        if (
            row.get("run_id") != run_id
            or qbank_id not in candidates
            or job_key in jobs
            or status not in {"succeeded", "cancelled"}
            or row.get("finished_at") is None
            or row.get("claimed_by") is not None
            or row.get("lease_expires_at") is not None
        ):
            raise ValueError("candidate stage job is duplicated, unfinished, or invalid")
        if status == "cancelled":
            if (
                stage != "planner_revision"
                or metadata.get("terminal_reason")
                != "planner_revision_unavailable"
            ):
                raise ValueError("candidate stage job has an invalid cancellation")
            unavailable_targets.add((qbank_id, round_id))
        jobs[job_key] = row

    revisions: dict[tuple[str, int], dict[str, Any]] = {}
    for row in revision_rows:
        qbank_id = str(row.get("qbank_id") or "")
        target_round = int(row.get("target_round_id", -1))
        revision_key = (qbank_id, target_round)
        source = assessment_by_feedback.get(str(row.get("source_feedback_id") or ""))
        revised_program = programs.get(str(row.get("revised_plan_id") or ""))
        ordered_directives = directives_by_feedback.get(
            str(row.get("source_feedback_id") or ""),
            [],
        )
        if (
            row.get("run_id") != run_id
            or revision_key in revisions
            or source is None
            or source.get("action") != "revise_program"
            or qbank_id != source.get("qbank_id")
            or target_round != int(source.get("round_id", -2)) + 1
            or row.get("source_plan_id") != source.get("plan_id")
            or revised_program is None
            or revised_program.get("qbank_id") != qbank_id
            or int(revised_program.get("round_id", -1)) != target_round
            or not ordered_directives
        ):
            raise ValueError("Planner revision lineage is invalid")
        directive_hash = canonical_content_hash(ordered_directives)
        lineage_payload = _step6d_payload(
            row,
            (
                "run_id",
                "qbank_id",
                "source_plan_id",
                "revised_plan_id",
                "source_feedback_id",
                "target_round_id",
                "producer_model_call_id",
                "directive_hash",
            ),
        )
        if (
            row.get("directive_hash") != directive_hash
            or row.get("revision_lineage_id")
            != stable_compiler_id("planner-revision-lineage", lineage_payload)
            or row.get("content_hash") != canonical_content_hash(lineage_payload)
        ):
            raise ValueError("Planner revision identity or directive hash is invalid")
        revisions[revision_key] = row

    consumed_revisions: set[tuple[str, int]] = set()
    consumed_unavailable: set[tuple[str, int]] = set()
    consumed_assessment_plan_ids: set[str] = set()
    consumed_assessment_execution_ids: set[str] = set()
    terminal_candidates = 0
    for qbank_id in sorted(candidates):
        values = [
            assessments[(qbank_id, round_id)]
            for round_id in range(4)
            if (qbank_id, round_id) in assessments
        ]
        rounds = [int(value["round_id"]) for value in values]
        if not values or rounds != list(range(len(values))):
            raise ValueError("Doubter rounds must be contiguous from round zero")
        terminal = False
        for index, assessment in enumerate(values):
            if terminal:
                raise ValueError("Doubter assessment appears after a terminal action")
            round_id = int(assessment["round_id"])
            action = str(assessment.get("action") or "")
            plan_id = str(assessment.get("plan_id") or "")
            execution_id = str(assessment.get("execution_id") or "")
            program = programs.get(plan_id)
            execution = executions.get(execution_id)
            dry_run = dry_runs.get(plan_id)
            dimensions = dimensions_by_assessment[str(assessment["assessment_id"])]
            ordered_directives = directives_by_feedback.get(
                str(assessment["feedback_id"]),
                [],
            )
            if (
                program is None
                or program.get("qbank_id") != qbank_id
                or int(program.get("round_id", -1)) != round_id
                or execution is None
                or execution.get("plan_id") != plan_id
                or execution.get("source_compiler_run_id") != run_id
                or execution.get("purpose") != "compiler_dry_run"
                or dry_run is None
                or dry_run.get("qbank_id") != qbank_id
                or dry_run.get("execution_id") != execution_id
                or dry_run.get("run_id") != run_id
            ):
                raise ValueError("Doubter assessment lacks its exact real dry-run lineage")
            consumed_assessment_plan_ids.add(plan_id)
            consumed_assessment_execution_ids.add(execution_id)
            if jobs.get((qbank_id, "doubter", round_id), {}).get("status") != "succeeded":
                raise ValueError("Doubter assessment lacks a succeeded candidate job")
            issue_codes = list(assessment.get("issue_codes") or [])
            if (
                assessment.get("prompt_version") == "pmmc-doubter-v1.2"
                and assessment.get("provider_schema_conformant") is False
                and (
                    action != "reject_candidate"
                    or issue_codes != ["structured_output_invalid"]
                    or any(dimensions.values())
                    or ordered_directives
                )
            ):
                raise ValueError("nonconformant Doubter output is not a conservative rejection")
            if (
                action == "pass"
                and (issue_codes or ordered_directives)
            ):
                raise ValueError("Doubter pass contains issues or directives")
            if (
                action == "reject_candidate"
                and ordered_directives
            ):
                raise ValueError("Doubter rejection contains revision directives")
            if action == "revise_program" and (
                round_id >= 3 or not ordered_directives
            ):
                raise ValueError("Doubter revision action is invalid")
            if action not in {"pass", "revise_program", "reject_candidate"}:
                raise ValueError("Doubter action is invalid")

            payload = {
                "feedback_id": assessment["feedback_id"],
                "run_id": run_id,
                "qbank_id": qbank_id,
                "plan_id": plan_id,
                "round_id": round_id,
                "dimensions": dimensions,
                "issues": issue_codes,
                "action": action,
                "directives": ordered_directives,
                "provider_schema_conformant": assessment.get(
                    "provider_schema_conformant"
                ),
            }
            if (
                assessment.get("feedback_id")
                != stable_compiler_id(
                    "doubter-feedback",
                    [run_id, qbank_id, plan_id, round_id],
                )
                or assessment.get("assessment_id")
                != stable_compiler_id("doubter-assessment", payload)
                or assessment.get("content_hash")
                != canonical_content_hash(payload)
            ):
                raise ValueError("Doubter assessment identity or hash is invalid")

            if action in {"pass", "reject_candidate"}:
                terminal = True
                continue
            target = round_id + 1
            if (qbank_id, target) in revisions:
                if (
                    jobs.get(
                        (qbank_id, "planner_revision", target), {}
                    ).get("status")
                    != "succeeded"
                    or jobs.get((qbank_id, "dry_run", target), {}).get("status")
                    != "succeeded"
                    or index + 1 >= len(values)
                    or int(values[index + 1]["round_id"]) != target
                ):
                    raise ValueError("Planner revision did not reach its next real dry-run")
                consumed_revisions.add((qbank_id, target))
            elif (qbank_id, target) in unavailable_targets:
                if index + 1 != len(values):
                    raise ValueError("unavailable Planner revision has later feedback")
                consumed_unavailable.add((qbank_id, target))
                terminal = True
            else:
                raise ValueError("Doubter revise action lacks a Planner revision terminal")
        if not terminal:
            raise ValueError("Doubter candidate lacks a terminal action")
        terminal_candidates += 1

    if set(assessments) - {
        (qbank_id, round_id)
        for qbank_id in candidates
        for round_id in range(4)
    }:
        raise ValueError("Doubter assessment crosses candidate scope")
    if consumed_revisions != set(revisions):
        raise ValueError("Planner revision lineage contains an orphaned revision")
    if consumed_unavailable != unavailable_targets:
        raise ValueError("Planner revision cancellation is orphaned")
    if (
        consumed_assessment_plan_ids != set(programs)
        or consumed_assessment_plan_ids != set(dry_runs)
        or consumed_assessment_execution_ids != set(executions)
    ):
        raise ValueError("Step 6D export contains an orphaned program or dry-run")
    for qbank_id, stage, round_id in jobs:
        if stage == "doubter" and (qbank_id, round_id) not in assessments:
            raise ValueError("candidate stage jobs contain an orphaned Doubter job")
        if stage == "planner_revision" and (
            (qbank_id, round_id) not in revisions
            and (qbank_id, round_id) not in unavailable_targets
        ):
            raise ValueError("candidate stage jobs contain an orphaned revision job")
        if stage == "dry_run" and (qbank_id, round_id) not in revisions:
            raise ValueError("candidate stage jobs contain an orphaned dry-run job")
    return {
        "assessment_count": len(assessments),
        "revision_count": len(revisions),
        "revision_unavailable_count": len(unavailable_targets),
        "terminal_candidate_count": terminal_candidates,
    }


def _verify_step6d_memory_index(
    *,
    rows: Mapping[str, Sequence[dict[str, Any]]],
    compiler: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> list[str]:
    """Verify and report the two frozen memory collections used by Step 6D/7."""

    indexes = rows["memory_operator_index_manifests"]
    if len(indexes) != 1 or indexes[0].get("status") != "ready":
        raise ValueError("verified-QBank export requires one ready memory operator index")
    index = indexes[0]
    snapshot_ids = run_manifest.get("dataset_snapshot_ids")
    if not isinstance(snapshot_ids, list) or len(snapshot_ids) != 1:
        raise ValueError("verified-QBank export must freeze exactly one snapshot")
    snapshot_id = str(snapshot_ids[0])
    dataset = str(run_manifest.get("dataset", ""))
    if (
        str(index.get("snapshot_id", "")) != snapshot_id
        or str(index.get("dataset", "")) != dataset
        or index.get("representation_version") != "memory-operator-index-v1"
        or int(index.get("dimension", 0)) != 4096
        or index.get("distance_metric") != "cosine"
    ):
        raise ValueError("verified-QBank memory index identity differs from the run manifest")

    collections = {
        str(row["collection_id"]): row for row in rows["embedding_collections"]
    }
    if len(collections) != len(rows["embedding_collections"]):
        raise ValueError("verified-QBank export has duplicate embedding collection IDs")
    collection_ids = {
        str(index["text_collection_id"]),
        str(index["visual_collection_id"]),
    }
    if len(collection_ids) != 2 or any(
        collection_id not in collections
        or collections[collection_id].get("status") != "ready"
        or int(collections[collection_id].get("dimension", 0)) != 4096
        for collection_id in collection_ids
    ):
        raise ValueError("verified-QBank memory operator collections are incomplete")

    collection_specs = {
        str(index["text_collection_id"]): (
            "memory_text",
            "text_memory",
            TEXT_MEMORY_REPRESENTATION,
            str(compiler["memory_text_model_alias"]),
            str(compiler["memory_text_model_revision"]),
        ),
        str(index["visual_collection_id"]): (
            "memory_image",
            "visual_memory",
            VISUAL_MEMORY_REPRESENTATION,
            str(compiler["memory_multimodal_model_alias"]),
            str(compiler["memory_multimodal_model_revision"]),
        ),
    }
    for collection_id, (
        source_type,
        role,
        representation,
        alias,
        revision,
    ) in collection_specs.items():
        collection = collections[collection_id]
        metadata = collection.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("verified-QBank memory collection metadata is invalid")
        if (
            index.get("text_model_alias") != compiler.get("memory_text_model_alias")
            or index.get("text_model_revision")
            != compiler.get("memory_text_model_revision")
            or index.get("multimodal_model_alias")
            != compiler.get("memory_multimodal_model_alias")
            or index.get("multimodal_model_revision")
            != compiler.get("memory_multimodal_model_revision")
            or str(collection.get("snapshot_id", "")) != snapshot_id
            or collection.get("source_type") != source_type
            or collection.get("model_alias") != alias
            or collection.get("model_revision") != revision
            or collection.get("distance_metric") != "cosine"
            or collection.get("normalized") is not True
            or metadata.get("collection_role") != role
            or metadata.get("representation") != representation
            or metadata.get("selection_hash") != index.get("selection_hash")
        ):
            raise ValueError("verified-QBank memory operator collection identity is invalid")

    print(
        "STEP6D_EXPORT_MEMORY_INDEX_ACCOUNTING "
        + orjson.dumps(
            {
                "collection_count": len(collection_ids),
                "collection_roles": ["text_memory", "visual_memory"],
                "dimension": 4096,
                "state": "verified",
            },
            option=orjson.OPT_SORT_KEYS,
        ).decode("utf-8"),
        flush=True,
    )
    return sorted(collection_ids)


def _verify_step6d_program_export(
    export_dir: Path,
    *,
    questioner_report: dict[str, Any],
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify the append-only r0-r3 graph and active-r2 routing projection."""

    runtime = export_dir / "runtime"
    for file_name in (*_REQUIRED_PROGRAM_FILES, *_REQUIRED_PROGRAM_V2_FILES):
        if not (runtime / file_name).is_file():
            raise ValueError(f"Step 6D runtime export is missing {file_name}")
    for file_name in _REQUIRED_STEP6D_FILES:
        if not (runtime / file_name).is_file():
            raise ValueError(f"Step 6D runtime export is missing {file_name}")

    run_id = str(questioner_report["run_id"])
    names = {
        path.name.removesuffix(".jsonl"): _read_jsonl(path)
        for path in (
            *(runtime / value for value in _REQUIRED_PROGRAM_FILES),
            *(runtime / value for value in _REQUIRED_PROGRAM_V2_FILES),
            *(runtime / value for value in _REQUIRED_STEP6D_FILES),
            runtime / "compiler_units.jsonl",
            runtime / "compiler_unit_stage_results.jsonl",
            runtime / "qbank_questions.jsonl",
            runtime / "questioner_candidate_gate_results.jsonl",
            runtime / "questioner_candidate_verifications.jsonl",
            runtime / "memory_programs.jsonl",
            runtime / "program_executions.jsonl",
            runtime / "embedding_collections.jsonl",
        )
    }
    for name, values in names.items():
        if name in {
            "embedding_collections",
            "qbank_candidate_families",
            "qbank_candidate_family_members",
        }:
            continue
        _assert_run_scope(values, run_id, name.replace("_", " "))

    compiler_rows = _read_jsonl(runtime / "compiler_runs.jsonl")
    if len(compiler_rows) != 1:
        raise ValueError("Step 6D export requires exactly one compiler run")
    compiler = compiler_rows[0]
    metadata = run_manifest.get("metadata")
    compiler_schema_version = str(compiler.get("compiler_schema_version", ""))
    expected_protocols = {
        "0.6.0": (
            "pmmc-questioner-v3.3",
            "pmmc-planner-v1.2",
            "pmmc-operators-v4",
            "pmmc-doubter-v1",
        ),
        "0.7.0": (
            "pmmc-questioner-v3.4",
            "pmmc-planner-v1.3",
            "pmmc-operators-v5",
            "pmmc-doubter-v1.1",
        ),
        "0.8.0": (
            "pmmc-questioner-v3.5",
            "pmmc-planner-v1.4",
            "pmmc-operators-v6",
            "pmmc-doubter-v1.2",
        ),
    }
    if (
        compiler_schema_version not in expected_protocols
        or compiler.get("target_stage") != "verified_qbank"
        or not isinstance(metadata, dict)
        or metadata.get("target_stage") != "verified_qbank"
        or compiler.get("active_round") != 2
        or compiler.get("max_round") != 3
    ):
        raise ValueError("Step 6D compiler identity or active round is invalid")
    questioner_config = metadata.get("questioner_config")
    expected_questioner, expected_planner, expected_operators, expected_doubter = (
        expected_protocols[compiler_schema_version]
    )
    if (
        not isinstance(questioner_config, dict)
        or questioner_config.get("protocol_version") != expected_questioner
        or compiler.get("planner_prompt_version") != expected_planner
        or compiler.get("operator_set_version") != expected_operators
        or compiler.get("doubter_prompt_version") != expected_doubter
    ):
        raise ValueError("Step 6D Q/P/D protocol identity is invalid")
    with _program_export_check("memory_index"):
        memory_collection_ids = _verify_step6d_memory_index(
            rows=names,
            compiler=compiler,
            run_manifest=run_manifest,
        )

    all_candidates = {
        str(row["qbank_id"]): row
        for row in names["qbank_questions"]
        if row.get("run_id") == run_id
    }
    if len(all_candidates) != len(names["qbank_questions"]):
        raise ValueError("Step 6D candidate IDs are duplicated")
    gate_eligible_ids = {
        str(row["qbank_id"])
        for row in names["questioner_candidate_gate_results"]
        if row.get("status") == "eligible_for_planning"
    }
    verifier_pass_ids = {
        str(row["qbank_id"])
        for row in names["questioner_candidate_verifications"]
        if row.get("verdict") == "pass"
        and int(row.get("verification_round", -1))
        == int(all_candidates.get(str(row["qbank_id"]), {}).get("candidate_version", -2))
    }
    candidate_ids = gate_eligible_ids & verifier_pass_ids
    if (
        not gate_eligible_ids <= set(all_candidates)
        or candidate_ids != gate_eligible_ids
    ):
        raise ValueError("Step 6D final eligible gate/verifier accounting is inconsistent")
    candidates = {qbank_id: all_candidates[qbank_id] for qbank_id in candidate_ids}
    r0_rows = names["r0_input_manifests"]
    _verify_step6d_eligible_accounting(
        questioner_report=questioner_report,
        candidate_ids=candidate_ids,
        r0_rows=r0_rows,
    )

    projection_manifests = {
        int(row["round_id"]): row
        for row in names["qbank_round_projection_manifests"]
    }
    if set(projection_manifests) != {0, 1, 2, 3}:
        raise ValueError("Step 6D requires exactly r0/r1/r2/r3 projections")
    projection_members: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    projection_round = {
        str(row["projection_id"]): int(row["round_id"])
        for row in names["qbank_round_projection_manifests"]
    }
    for row in names["qbank_round_projection_members"]:
        round_id = projection_round.get(str(row["projection_id"]))
        if round_id is None:
            raise ValueError("round projection member has no parent manifest")
        qbank_id = str(row["qbank_id"])
        if qbank_id in projection_members[round_id]:
            raise ValueError("round projection contains a duplicate candidate")
        projection_members[round_id][qbank_id] = row
    if any(set(projection_members[round_id]) != set(candidates) for round_id in range(4)):
        raise ValueError("round projections do not close over the candidate denominator")

    decisions = {
        str(row["qbank_id"]): row for row in names["qbank_decisions"]
    }
    if len(decisions) != len(candidates) or set(decisions) != set(candidates):
        raise ValueError("active-r2 decisions do not close over candidates")
    active_members = projection_members[2]
    for qbank_id, decision in decisions.items():
        accepted = active_members[qbank_id].get("outcome") in {
            "accepted",
            "carried_forward",
        }
        if bool(decision.get("accepted")) != accepted:
            raise ValueError("active-r2 decision differs from its projection")

    routing_manifests = names["routing_projection_manifests"]
    if len(routing_manifests) != 1:
        raise ValueError("Step 6D requires exactly one routing projection")
    routing = routing_manifests[0]
    routing_members = names["routing_projection_members"]
    accepted_ids = {
        qbank_id for qbank_id, row in decisions.items() if row.get("accepted") is True
    }
    routed_ids = {str(row["qbank_id"]) for row in routing_members}
    if (
        routed_ids != accepted_ids
        or int(routing.get("member_count", -1)) != len(routing_members)
        or int(routing.get("active_round", -1)) != 2
        or any(
            not bool(decisions[str(row["qbank_id"])]["accepted"])
            for row in routing_members
        )
    ):
        raise ValueError("runtime routing projection contains missing or rejected members")
    with _program_export_check("projection_identity"):
        _verify_step6d_projection_identities(
            run_id=run_id,
            candidates=candidates,
            r0=r0_rows[0],
            manifests=projection_manifests,
            members=projection_members,
            decisions=decisions,
            routing=routing,
            routing_members=routing_members,
        )

    programs = {str(row["plan_id"]): row for row in names["memory_programs"]}
    executions = {
        str(row["plan_id"]): row for row in names["program_executions"]
    }
    executions_by_id = {
        str(row["execution_id"]): row for row in names["program_executions"]
    }
    if (
        len(programs) != len(names["memory_programs"])
        or len(executions) != len(names["program_executions"])
        or len(executions_by_id) != len(names["program_executions"])
    ):
        raise ValueError("Step 6D program or execution identities are duplicated")
    dry_runs: dict[str, dict[str, Any]] = {}
    for value in names["program_dry_run_assessments"]:
        plan_id = str(value["plan_id"])
        if plan_id in dry_runs:
            raise ValueError("Step 6D export contains duplicate dry-run assessments")
        dry_runs[plan_id] = value
    for row in routing_members:
        qbank_id = str(row["qbank_id"])
        plan_id = str(row["plan_id"])
        program = programs.get(plan_id)
        dry_run = dry_runs.get(plan_id)
        source_member = active_members[qbank_id]
        decision = decisions[qbank_id]
        candidate = candidates[qbank_id]
        if (
            program is None
            or plan_id not in executions
            or dry_run is None
            or str(program.get("qbank_id")) != qbank_id
            or int(program.get("round_id", -1))
            != int(source_member.get("passed_at_round", -2))
            or decision.get("final_plan_id") != plan_id
            or dry_run.get("execution_success") is not True
        ):
            raise ValueError("routed candidate lacks a real executed program")
        if candidate.get("route_class") == "control":
            if dry_run.get("abstain_triggered") is not True:
                raise ValueError("routed control candidate lacks a successful abstain")
        elif not (
            dry_run.get("nonempty_evidence") is True
            and dry_run.get("required_modality_sufficient") is True
            and dry_run.get("pre_control_sufficient") is True
        ):
            raise ValueError("routed answerable candidate lacks sufficient real evidence")

    with _program_export_check("doubter_chain"):
        doubter_chain = _verify_step6d_doubter_chain(
            run_id=run_id,
            candidates=candidates,
            programs=programs,
            executions=executions_by_id,
            dry_runs=dry_runs,
            assessment_rows=names["doubter_assessments"],
            dimension_rows=names["doubter_assessment_dimensions"],
            directive_rows=names["doubter_directives"],
            revision_rows=names["planner_revision_lineage"],
            candidate_job_rows=names["candidate_stage_jobs"],
        )

    structured_diagnostics = names["structured_output_item_diagnostics"]
    invalid_diagnostic_statuses = {
        "invalid",
        "repaired",
        "conservative_reject",
    }
    planner_native_output_issue_count = sum(
        row.get("stage") in {"planner", "planner_revision"}
        and row.get("status") in invalid_diagnostic_statuses
        for row in structured_diagnostics
    )
    doubter_native_output_issue_count = sum(
        row.get("stage") == "doubter"
        and row.get("status") in invalid_diagnostic_statuses
        for row in structured_diagnostics
    )
    nonconformant_doubter_count = sum(
        row.get("provider_schema_conformant") is not True
        for row in names["doubter_assessments"]
    )
    if nonconformant_doubter_count != doubter_native_output_issue_count:
        raise ValueError(
            "Doubter native-output diagnostics do not reconcile with assessments"
        )
    if compiler_schema_version == "0.8.0":
        for assessment in names["doubter_assessments"]:
            if assessment.get("provider_schema_conformant") is True:
                continue
            matching_diagnostics = [
                row for row in structured_diagnostics
                if row.get("model_call_id") == assessment.get("producer_model_call_id")
            ]
            if len(matching_diagnostics) != 1 or any(
                row.get("stage") != "doubter"
                or row.get("status") != "conservative_reject"
                or row.get("compiler_unit_id") != assessment.get("compiler_unit_id")
                or row.get("repair_attempt_count", 0) != 0
                or not row.get("errors")
                for row in matching_diagnostics
            ):
                raise ValueError("Doubter rejection diagnostic lineage is incomplete")

    decision_jobs = [
        row
        for row in names["candidate_stage_jobs"]
        if row.get("stage") == "decision" and int(row.get("round_id", -1)) == 2
    ]
    if (
        {str(row["qbank_id"]) for row in decision_jobs} != set(candidates)
        or any(row.get("status") != "succeeded" for row in decision_jobs)
    ):
        raise ValueError("candidate-level decision jobs are incomplete")

    stages_by_unit: dict[str, set[str]] = defaultdict(set)
    for row in names["compiler_unit_stage_results"]:
        if row.get("status") == "success":
            stages_by_unit[str(row["compiler_unit_id"])].add(str(row["stage"]))
    unit_ids = {str(row["compiler_unit_id"]) for row in names["compiler_units"]}
    required_stages = {"questioner", "planner", "dry_run", "doubter", "decision"}
    if any(not required_stages <= stages_by_unit[unit_id] for unit_id in unit_ids):
        raise ValueError("Step 6D unit-stage accounting is incomplete")

    return {
        **questioner_report,
        "unit_count": len(unit_ids),
        "eligible_candidate_count": len(candidates),
        "planner_batch_count": len(names["planner_generation_batches"]),
        "program_count": len(programs),
        "dry_run_count": len(names["program_dry_run_assessments"]),
        "doubter_assessment_count": doubter_chain["assessment_count"],
        "doubter_native_output_issue_count": doubter_native_output_issue_count,
        "planner_native_output_issue_count": planner_native_output_issue_count,
        "planner_revision_count": doubter_chain["revision_count"],
        "planner_revision_unavailable_count": doubter_chain[
            "revision_unavailable_count"
        ],
        "doubter_terminal_candidate_count": doubter_chain[
            "terminal_candidate_count"
        ],
        "projection_count": 4,
        "decision_count": len(decisions),
        "routing_member_count": len(routing_members),
        "active_round": 2,
        "max_round": 3,
        "r0_hard_gate_passed": True,
        "memory_collection_ids": memory_collection_ids,
        "planner_oracle_input_violation_count": 0,
        "program_requirement_dataflow_violation_count": 0,
    }


def verify_program_export(export_dir: Path) -> dict[str, Any]:
    """Verify a completed legacy or hardened round-zero compiler export."""

    with _program_export_check("contract"):
        return _verify_program_export(export_dir)


def _verify_program_export(export_dir: Path) -> dict[str, Any]:
    questioner_report = verify_questioner_export(export_dir)
    export_version = questioner_report.get("program_runtime_export_version")
    if export_version not in SUPPORTED_PROGRAM_RUNTIME_EXPORT_VERSIONS:
        raise ValueError("export does not use a supported Planner runtime protocol")
    runtime = export_dir / "runtime"
    for file_name in _REQUIRED_PROGRAM_FILES:
        if not (runtime / file_name).is_file():
            raise ValueError(f"program runtime export is missing {file_name}")

    run_id = str(questioner_report["run_id"])
    run_manifest = _read_json(runtime / "run_manifest.json")
    compiler_schema_version = str(run_manifest.get("compiler_schema_version", ""))
    if compiler_schema_version in {"0.6.0", "0.7.0", "0.8.0"}:
        expected_version = (
            STEP7B_PROGRAM_RUNTIME_EXPORT_VERSION
            if compiler_schema_version == "0.8.0"
            else STEP7_PROGRAM_RUNTIME_EXPORT_VERSION
            if compiler_schema_version == "0.7.0"
            else PROGRAM_RUNTIME_EXPORT_VERSION
        )
        if export_version != expected_version:
            raise ValueError(
                "verified-QBank export protocol does not match compiler schema"
            )
        result = _verify_step6d_program_export(
            export_dir,
            questioner_report=questioner_report,
            run_manifest=run_manifest,
        )
        if compiler_schema_version == "0.8.0":
            for file_name in _REQUIRED_STEP7B_FILES:
                if not (runtime / file_name).is_file():
                    raise ValueError(f"Step 7B program export is missing {file_name}")
        return result
    if compiler_schema_version not in {"0.4.0", "0.5.0"}:
        raise ValueError(
            "Planner export requires compiler schema 0.4.0 through 0.8.0"
        )
    hardened = compiler_schema_version == "0.5.0"
    expected_export_version = (
        "compiler-program-runtime-export-v2"
        if hardened
        else "compiler-program-runtime-export-v1"
    )
    if export_version != expected_export_version:
        raise ValueError("program runtime protocol does not match compiler schema")
    if hardened:
        for file_name in _REQUIRED_PROGRAM_V2_FILES:
            if not (runtime / file_name).is_file():
                raise ValueError(f"hardened program export is missing {file_name}")
    metadata = run_manifest.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("target_stage") != "programs":
        raise ValueError("Planner export is not a target_stage=programs run")
    questioner_config = metadata.get("questioner_config")
    expected_questioner = (
        "pmmc-questioner-v3.2" if hardened else "pmmc-questioner-v3.1"
    )
    if (
        not isinstance(questioner_config, dict)
        or questioner_config.get("protocol_version") != expected_questioner
    ):
        raise ValueError(f"Planner export requires Questioner {expected_questioner}")
    compiler_rows = _read_jsonl(runtime / "compiler_runs.jsonl")
    if len(compiler_rows) != 1:
        raise ValueError("program export requires exactly one compiler run")
    compiler = compiler_rows[0]
    planner_config = PlannerConfig(
        protocol_version=(
            "pmmc-planner-v1.1" if hardened else "pmmc-planner-v1"
        ),
        prompt_version=(
            "pmmc-planner-v1.1" if hardened else "pmmc-planner-v1"
        ),
        renderer_version=(
            "planner-non-oracle-renderer-v1.1"
            if hardened
            else "planner-safe-alias-renderer-v1"
        ),
        parser_version=(
            "planner-json-parser-v1.1" if hardened else "planner-json-parser-v1"
        ),
        schema_version=(
            "planner-program-schema-v1.1"
            if hardened
            else "planner-program-schema-v1"
        ),
        policy_version=(
            "planner-program-policy-v2"
            if hardened
            else "planner-program-policy-v1"
        ),
        operator_set_version=(
            "pmmc-operators-v3" if hardened else "pmmc-operators-v2"
        ),
        dry_run_policy=(
            "evidence_driven_v3" if hardened else "evidence_driven_v2"
        ),
    )
    expected_compiler_identity = {
        "run_id": run_id,
        "compiler_schema_version": compiler_schema_version,
        "target_stage": "programs",
        "operator_set_version": planner_config.operator_set_version,
        "planner_prompt_version": planner_config.prompt_version,
        "planner_prompt_hash": planner_prompt_sha256(planner_config),
        "planner_renderer_version": planner_config.renderer_version,
        "planner_parser_version": planner_config.parser_version,
        "planner_schema_version": planner_config.schema_version,
        "program_policy_version": planner_config.policy_version,
        "dry_run_policy_version": planner_config.dry_run_policy,
        "memory_index_representation_version": "memory-operator-index-v1",
    }
    for field_name, expected in expected_compiler_identity.items():
        if compiler.get(field_name) != expected:
            raise ValueError(f"program compiler identity mismatch: {field_name}")
    if (
        not isinstance(compiler.get("questioner_freeze_manifest_hash"), str)
        or compiler.get("questioner_freeze_manifest_hash")
        != metadata.get("questioner_freeze_or_audit_hash")
        or compiler.get("planner_config_hash") != metadata.get("planner_config_hash")
        or metadata.get("planner_response_schema_hash")
        != planner_response_schema_sha256(planner_config)
    ):
        raise ValueError("program compiler approval or Planner schema identity is invalid")
    model_configs = run_manifest.get("model_configs")
    if not isinstance(model_configs, dict):
        raise ValueError("program run manifest has no frozen model registry subset")
    for alias_field, revision_field in (
        ("planner_model_alias", "planner_model_revision"),
        ("memory_text_model_alias", "memory_text_model_revision"),
        ("memory_multimodal_model_alias", "memory_multimodal_model_revision"),
    ):
        alias = compiler.get(alias_field)
        revision = compiler.get(revision_field)
        model = model_configs.get(alias) if isinstance(alias, str) else None
        if (
            not isinstance(model, dict)
            or not isinstance(revision, str)
            or model.get("model_revision") != revision
        ):
            raise ValueError(f"program compiler model identity mismatch: {alias_field}")

    names = (
        "compiler_units",
        "compiler_stage_jobs",
        "compiler_unit_stage_results",
        "qbank_questions",
        "questioner_candidate_gate_results",
        "planner_generation_batches",
        *(
            (
                "planner_requirement_contracts",
                "planner_candidate_drafts",
                "structured_output_item_diagnostics",
            )
            if hardened
            else ()
        ),
        "planner_candidate_outcomes",
        "planner_batch_programs",
        "memory_programs",
        "program_steps",
        "program_expected_evidence_refs",
        "program_validation_results",
        "program_validation_issues",
        "memory_operator_index_manifests",
        "program_query_embedding_batches",
        "program_query_embeddings",
        "program_executions",
        "program_execution_steps",
        "program_execution_evidence_refs",
        "program_dry_run_assessments",
        "program_dry_run_evidence_matches",
        "embedding_collections",
        "embeddings_metadata",
    )
    rows = {name: _read_jsonl(runtime / f"{name}.jsonl") for name in names}
    for name, values in rows.items():
        if name in {"embedding_collections", "embeddings_metadata"}:
            continue
        _assert_run_scope(values, run_id, name.replace("_", " "))

    unit_ids = {str(row["compiler_unit_id"]) for row in rows["compiler_units"]}
    stages_by_unit: dict[str, set[str]] = defaultdict(set)
    for row in rows["compiler_unit_stage_results"]:
        if row.get("status") == "success":
            stages_by_unit[str(row["compiler_unit_id"])].add(str(row["stage"]))
    if any(
        not {"questioner", "planner", "dry_run"} <= stages_by_unit[unit_id] for unit_id in unit_ids
    ):
        raise ValueError("program compiler stages are incomplete")

    stage_job_counts = Counter(
        (str(row["compiler_unit_id"]), str(row["stage"]))
        for row in rows["compiler_stage_jobs"]
        if row.get("status") == "succeeded" and int(row.get("round_id", -1)) == 0
    )
    if any(
        stage_job_counts[(unit_id, stage)] != 1
        for unit_id in unit_ids
        for stage in ("planner", "dry_run")
    ):
        raise ValueError("program stage-job denominator is incomplete")

    gates = {
        str(row["qbank_id"]): row
        for row in rows["questioner_candidate_gate_results"]
        if row.get("status") == "eligible_for_planning"
    }
    outcomes = {str(row["qbank_id"]): row for row in rows["planner_candidate_outcomes"]}
    if len(outcomes) != len(rows["planner_candidate_outcomes"]):
        raise ValueError("Planner candidates have duplicate terminal outcomes")
    if set(outcomes) != set(gates):
        raise ValueError("Planner terminal outcomes do not close over eligible candidates")

    programs = {str(row["plan_id"]): row for row in rows["memory_programs"]}
    if len(programs) != len(rows["memory_programs"]):
        raise ValueError("Planner program IDs are duplicated")
    links = {str(row["qbank_id"]): row for row in rows["planner_batch_programs"]}
    if len(links) != len(rows["planner_batch_programs"]):
        raise ValueError("Planner candidate-to-program links are duplicated")
    for qbank_id, outcome in outcomes.items():
        status = str(outcome["status"])
        if status not in _PROGRAM_OUTCOMES | {
            "parse_failed",
            "static_rejected",
            "policy_rejected",
        }:
            raise ValueError("Planner outcome status is unsupported")
        plan_id = outcome.get("plan_id")
        if status in _PROGRAM_OUTCOMES:
            if not isinstance(plan_id, str) or plan_id not in programs:
                raise ValueError("program-bearing Planner outcome has no program")
            if qbank_id not in links or str(links[qbank_id]["plan_id"]) != plan_id:
                raise ValueError("Planner program link is incomplete")
        elif plan_id is not None or qbank_id in links:
            raise ValueError("rejected Planner outcome unexpectedly references a program")
    if {str(row["plan_id"]) for row in links.values()} != set(programs):
        raise ValueError("persisted programs are not all linked to Planner batches")

    batches = {str(row["planner_batch_id"]): row for row in rows["planner_generation_batches"]}
    if len(batches) != len(rows["planner_generation_batches"]):
        raise ValueError("Planner generation batch IDs are duplicated")
    batch_candidates: set[str] = set()
    for batch in batches.values():
        candidate_ids = [str(item) for item in batch.get("candidate_qbank_ids", [])]
        if int(batch["parsed_count"]) + int(batch["invalid_count"]) != len(candidate_ids):
            raise ValueError("Planner batch candidate accounting is incomplete")
        if batch_candidates & set(candidate_ids):
            raise ValueError("Planner candidate appears in more than one batch")
        batch_candidates.update(candidate_ids)
    if batch_candidates != set(gates):
        raise ValueError("Planner batches do not cover the eligible denominator")
    if hardened:
        requirements = {
            str(row["qbank_id"]): row for row in rows["planner_requirement_contracts"]
        }
        drafts = {
            str(row["qbank_id"]): row for row in rows["planner_candidate_drafts"]
        }
        if (
            len(requirements) != len(rows["planner_requirement_contracts"])
            or set(requirements) != set(gates)
        ):
            raise ValueError("Planner requirement contracts do not close over candidates")
        if (
            len(drafts) != len(rows["planner_candidate_drafts"])
            or set(drafts) != set(gates)
        ):
            raise ValueError("Planner provider drafts do not close over candidates")
        for qbank_id, draft in drafts.items():
            requirement = requirements[qbank_id]
            if (
                draft.get("requirement_contract_id")
                != requirement.get("requirement_contract_id")
                or draft.get("compiler_unit_id")
                != requirement.get("compiler_unit_id")
                or bool(draft.get("parsed"))
                != (draft.get("program_hash") is not None)
            ):
                raise ValueError("Planner draft/requirement lineage is inconsistent")
    stage_jobs = {str(row["stage_job_id"]): row for row in rows["compiler_stage_jobs"]}
    model_call_rows = _read_jsonl(runtime / "model_calls.jsonl")
    model_calls = {str(row["model_call_id"]): row for row in model_call_rows}
    if len(model_calls) != len(model_call_rows):
        raise ValueError("program export has duplicate model-call IDs")
    for batch in batches.values():
        unit_id = str(batch["compiler_unit_id"])
        stage_job = stage_jobs.get(str(batch["stage_job_id"]))
        call = model_calls.get(str(batch["model_call_id"]))
        request_metadata = (
            dict(call.get("request_config") or {}).get("request_metadata", {})
            if call is not None
            else None
        )
        if (
            stage_job is None
            or stage_job.get("stage") != "planner"
            or stage_job.get("compiler_unit_id") != unit_id
            or call is None
            or call.get("stage") != "planner"
            or call.get("status") != "success"
            or call.get("compiler_unit_id") != unit_id
            or call.get("model_alias") != compiler.get("planner_model_alias")
            or not isinstance(call.get("actual_model_id"), str)
            or not call.get("actual_model_id")
            or not isinstance(request_metadata, dict)
            or request_metadata.get("task") != "pmmc_planner"
            or batch.get("prompt_hash") != planner_prompt_sha256(planner_config)
            or batch.get("schema_hash")
            != planner_response_schema_sha256(planner_config)
        ):
            raise ValueError("Planner generation model-call lineage is invalid")

    if hardened:
        questions_by_id = {
            str(row["qbank_id"]): row for row in rows["qbank_questions"]
        }
        artifact_manifest_path = (
            runtime / "private" / "model_artifact_manifest.jsonl"
        )
        artifact_rows = (
            _read_jsonl(artifact_manifest_path)
            if artifact_manifest_path.is_file()
            else []
        )
        planner_input_by_call = {
            str(row["model_call_id"]): export_dir / str(row["relative_path"])
            for row in artifact_rows
            if row.get("role") == "input"
        }
        ordered_batches = sorted(
            batches.values(),
            key=lambda item: (
                str(item.get("compiler_unit_id", "")),
                int(item.get("batch_ordinal", -1)),
            ),
        )
        audit_started = time.monotonic()
        next_audit_report = audit_started + 30.0
        _planner_audit_event(
            "started",
            artifact_input_count=len(planner_input_by_call),
            batch_count=len(ordered_batches),
        )
        for ordinal, batch in enumerate(ordered_batches, start=1):
            payload_path = planner_input_by_call.get(str(batch["model_call_id"]))
            if payload_path is None or not payload_path.is_file():
                _planner_audit_event(
                    "failed",
                    batch_ordinal=int(batch.get("batch_ordinal", -1)),
                    checked_batch_count=ordinal - 1,
                    reason_code="missing_input_artifact",
                )
                raise ValueError("Planner input artifact is missing from hardened export")
            payload = payload_path.read_bytes()
            try:
                _verify_hardened_planner_request_artifact(
                    payload,
                    batch=batch,
                    questions_by_id=questions_by_id,
                    requirements_by_question=requirements,
                    config=planner_config,
                )
            except _PlannerRequestAuditError as exc:
                call_locator = hashlib.sha256(
                    str(batch["model_call_id"]).encode("utf-8")
                ).hexdigest()[:12]
                _planner_audit_event(
                    "failed",
                    batch_ordinal=int(batch.get("batch_ordinal", -1)),
                    call_locator=call_locator,
                    checked_batch_count=ordinal - 1,
                    reason_code=exc.reason_code,
                )
                raise ValueError(
                    "Planner request artifact failed non-oracle audit "
                    f"({exc.reason_code})"
                ) from exc
            now = time.monotonic()
            if now >= next_audit_report:
                _planner_audit_event(
                    "progress",
                    checked_batch_count=ordinal,
                    total_batch_count=len(ordered_batches),
                )
                next_audit_report = now + 30.0
        _planner_audit_event(
            "completed",
            checked_batch_count=len(ordered_batches),
        )

    validation_by_plan: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    validation_ids: set[str] = set()
    for row in rows["program_validation_results"]:
        plan_id = str(row["plan_id"])
        kind = str(row["validation_kind"])
        if kind in validation_by_plan[plan_id]:
            raise ValueError("program has duplicate validation kind")
        validation_by_plan[plan_id][kind] = row
        validation_ids.add(str(row["validation_result_id"]))
    if any(
        str(row["validation_result_id"]) not in validation_ids
        for row in rows["program_validation_issues"]
    ):
        raise ValueError("program validation issue has no parent result")
    assessments = {str(row["plan_id"]): row for row in rows["program_dry_run_assessments"]}
    if len(assessments) != len(rows["program_dry_run_assessments"]):
        raise ValueError("program dry-run assessments are duplicated")
    outcome_by_plan = {
        str(row["plan_id"]): row for row in outcomes.values() if row.get("plan_id") is not None
    }
    for plan_id in programs:
        validations = validation_by_plan.get(plan_id, {})
        if any(
            kind not in validations or validations[kind].get("valid") is not True
            for kind in ("static", "policy")
        ):
            raise ValueError("program lacks successful static/policy validation")
        status = str(outcome_by_plan[plan_id]["status"])
        lineage = validations.get("lineage")
        if status == "lineage_rejected":
            if lineage is None or lineage.get("valid") is not False or plan_id in assessments:
                raise ValueError("lineage-rejected program facts are inconsistent")
        elif status == "query_vector_unavailable":
            if plan_id in assessments:
                raise ValueError("vector-unavailable program unexpectedly has a dry-run")
        elif lineage is None or lineage.get("valid") is not True or plan_id not in assessments:
            raise ValueError("executable program lacks lineage validation or dry-run")

    expected_by_plan: dict[str, set[str]] = defaultdict(set)
    for row in rows["program_expected_evidence_refs"]:
        expected_by_plan[str(row["plan_id"])].add(str(row["evidence_id"]))
    for row in rows["program_steps"]:
        plan_id = str(row["plan_id"])
        if _strings(row.get("args", {})) & expected_by_plan.get(plan_id, set()):
            raise ValueError("Planner operator args contain oracle expected evidence")

    indexes = rows["memory_operator_index_manifests"]
    if len(indexes) != 1 or indexes[0].get("status") != "ready":
        raise ValueError("program export requires one ready memory operator index")
    index = indexes[0]
    snapshot_ids = run_manifest.get("dataset_snapshot_ids")
    if not isinstance(snapshot_ids, list) or len(snapshot_ids) != 1:
        raise ValueError("program export must freeze exactly one snapshot")
    snapshot_id = str(snapshot_ids[0])
    dataset = str(run_manifest.get("dataset", ""))
    if (
        str(index.get("snapshot_id", "")) != snapshot_id
        or str(index.get("dataset", "")) != dataset
        or index.get("representation_version") != "memory-operator-index-v1"
        or int(index.get("dimension", 0)) != 4096
        or index.get("distance_metric") != "cosine"
    ):
        raise ValueError("memory operator index identity differs from the run manifest")
    collections = {str(row["collection_id"]): row for row in rows["embedding_collections"]}
    if len(collections) != len(rows["embedding_collections"]):
        raise ValueError("program export has duplicate embedding collection IDs")
    index_collection_ids = {
        str(index["text_collection_id"]),
        str(index["visual_collection_id"]),
    }
    if len(index_collection_ids) != 2 or any(
        collection_id not in collections
        or collections[collection_id].get("status") != "ready"
        or int(collections[collection_id].get("dimension", 0)) != 4096
        for collection_id in index_collection_ids
    ):
        raise ValueError("memory operator collections are incomplete")
    memory_collection_specs = {
        str(index["text_collection_id"]): (
            "memory_text",
            "text_memory",
            TEXT_MEMORY_REPRESENTATION,
            str(compiler["memory_text_model_alias"]),
            str(compiler["memory_text_model_revision"]),
        ),
        str(index["visual_collection_id"]): (
            "memory_image",
            "visual_memory",
            VISUAL_MEMORY_REPRESENTATION,
            str(compiler["memory_multimodal_model_alias"]),
            str(compiler["memory_multimodal_model_revision"]),
        ),
    }
    for collection_id, (
        source_type,
        role,
        representation,
        alias,
        revision,
    ) in memory_collection_specs.items():
        collection = collections[collection_id]
        collection_metadata = collection.get("metadata")
        if not isinstance(collection_metadata, dict):
            raise ValueError("memory operator collection metadata is invalid")
        if (
            index.get("text_model_alias") != compiler.get("memory_text_model_alias")
            or index.get("text_model_revision") != compiler.get("memory_text_model_revision")
            or index.get("multimodal_model_alias") != compiler.get("memory_multimodal_model_alias")
            or index.get("multimodal_model_revision")
            != compiler.get("memory_multimodal_model_revision")
            or str(collection.get("snapshot_id", "")) != snapshot_id
            or collection.get("source_type") != source_type
            or collection.get("model_alias") != alias
            or collection.get("model_revision") != revision
            or collection.get("distance_metric") != "cosine"
            or collection.get("normalized") is not True
            or collection_metadata.get("collection_role") != role
            or collection_metadata.get("representation") != representation
            or collection_metadata.get("selection_hash") != index["selection_hash"]
        ):
            raise ValueError("memory operator collection identity is invalid")

    vectors_by_question: dict[str, set[str]] = defaultdict(set)
    for row in rows["program_query_embeddings"]:
        vectors_by_question[str(row["qbank_id"])].add(str(row["channel"]))
    required_channels = {"question_text_query", "question_joint_query"}
    if any(vectors_by_question[qbank_id] != required_channels for qbank_id in gates):
        raise ValueError("program query-vector denominator is incomplete")
    embedding_rows = {str(row["embedding_id"]): row for row in rows["embeddings_metadata"]}
    if len(embedding_rows) != len(rows["embeddings_metadata"]):
        raise ValueError("program export has duplicate embedding IDs")
    query_collections_by_channel: dict[str, set[str]] = defaultdict(set)
    for collection_id, collection_row in collections.items():
        collection_metadata = collection_row.get("metadata")
        if not isinstance(collection_metadata, dict):
            continue
        query_role = collection_metadata.get("query_role")
        if query_role in required_channels:
            query_collections_by_channel[str(query_role)].add(collection_id)
    for ref in rows["program_query_embeddings"]:
        embedding = embedding_rows.get(str(ref["embedding_id"]))
        if embedding is None:
            raise ValueError("program query-vector reference is dangling")
        channel = str(ref["channel"])
        if (
            embedding.get("qbank_id") != ref["qbank_id"]
            or embedding.get("source_id") != ref["qbank_id"]
        ):
            raise ValueError("program query-vector typed source lineage is invalid")
        query_collections_by_channel[channel].add(str(embedding["collection_id"]))
    expected_query_models = {
        "question_text_query": (
            str(questioner_config["text_embedding_model_alias"]),
            str(index["text_model_revision"]),
        ),
        "question_joint_query": (
            str(questioner_config["multimodal_embedding_model_alias"]),
            str(index["multimodal_model_revision"]),
        ),
    }
    query_batches = {
        str(row["query_embedding_batch_id"]): row for row in rows["program_query_embedding_batches"]
    }
    if len(query_batches) != len(rows["program_query_embedding_batches"]):
        raise ValueError("program query embedding batch IDs are duplicated")
    batch_qbanks_by_channel: dict[str, set[str]] = defaultdict(set)
    for batch in query_batches.values():
        channel = str(batch["channel"])
        if channel not in required_channels:
            raise ValueError("program query embedding batch has an unknown channel")
        qbank_ids = [str(value) for value in batch.get("qbank_ids", [])]
        if not qbank_ids or len(qbank_ids) != len(set(qbank_ids)):
            raise ValueError("program query embedding batch denominator is invalid")
        if batch_qbanks_by_channel[channel] & set(qbank_ids):
            raise ValueError("program query candidate appears in multiple embedding batches")
        batch_qbanks_by_channel[channel].update(qbank_ids)
        call = model_calls.get(str(batch["model_call_id"]))
        alias = expected_query_models[channel][0]
        request_metadata = (
            dict(call.get("request_config") or {}).get("request_metadata", {})
            if call is not None
            else None
        )
        if (
            call is None
            or call.get("status") != "success"
            or call.get("stage") != "embed"
            or call.get("model_alias") != alias
            or not isinstance(call.get("actual_model_id"), str)
            or not call.get("actual_model_id")
            or not isinstance(request_metadata, dict)
            or request_metadata.get("task") != "program_query_embedding"
            or request_metadata.get("channel") != channel
        ):
            raise ValueError("program query embedding model-call lineage is invalid")
    if any(batch_qbanks_by_channel[channel] != set(gates) for channel in required_channels):
        raise ValueError("program query embedding batches do not cover the denominator")

    refs_by_batch: dict[str, set[str]] = defaultdict(set)
    for ref in rows["program_query_embeddings"]:
        batch_id = str(ref["query_embedding_batch_id"])
        ref_batch = query_batches.get(batch_id)
        if ref_batch is None or ref.get("channel") != ref_batch.get("channel"):
            raise ValueError("program query embedding reference has invalid batch lineage")
        refs_by_batch[batch_id].add(str(ref["qbank_id"]))
    if any(
        refs_by_batch[batch_id] != {str(value) for value in batch.get("qbank_ids", [])}
        for batch_id, batch in query_batches.items()
    ):
        raise ValueError("program query embedding batch fan-out is incomplete")

    for channel, (alias, revision) in expected_query_models.items():
        channel_collections = query_collections_by_channel[channel]
        if len(channel_collections) != 1:
            raise ValueError("program query channel must bind exactly one collection")
        query_collection = collections.get(next(iter(channel_collections)))
        query_metadata = query_collection.get("metadata") if query_collection is not None else None
        if (
            query_collection is None
            or str(query_collection.get("run_id", "")) != run_id
            or str(query_collection.get("snapshot_id", "")) != snapshot_id
            or query_collection.get("source_type") != "query"
            or query_collection.get("model_alias") != alias
            or query_collection.get("model_revision") != revision
            or query_collection.get("status") != "ready"
            or int(query_collection.get("dimension", 0)) != 4096
            or query_collection.get("distance_metric") != "cosine"
            or query_collection.get("normalized") is not True
            or not isinstance(query_metadata, dict)
            or query_metadata.get("query_role") != channel
            or query_metadata.get("expected_images_allowed") is not False
        ):
            raise ValueError("program query collection identity is invalid")

    for embedding in embedding_rows.values():
        if str(embedding["collection_id"]) in memory_collection_specs and (
            not isinstance(embedding.get("model_call_id"), str)
            or not embedding.get("model_call_id")
        ):
            raise ValueError("program memory embedding has no producer model call")

    # Memory collections are content-addressed and may have been produced by an
    # earlier compiler run. Their collection identity and producer call IDs are
    # preserved above, while current-run call closure applies to query vectors.
    audited_embedding_collections = {
        next(iter(query_collections_by_channel[channel])): (
            alias,
            "program_query_embedding",
        )
        for channel, (alias, _revision) in expected_query_models.items()
    }
    for embedding in embedding_rows.values():
        collection_id = str(embedding["collection_id"])
        expected_call = audited_embedding_collections.get(collection_id)
        if expected_call is None:
            continue
        alias, task = expected_call
        call = model_calls.get(str(embedding.get("model_call_id", "")))
        request_metadata = (
            dict(call.get("request_config") or {}).get("request_metadata", {})
            if call is not None
            else None
        )
        if (
            call is None
            or call.get("status") != "success"
            or call.get("stage") != "embed"
            or call.get("model_alias") != alias
            or not isinstance(call.get("actual_model_id"), str)
            or not call.get("actual_model_id")
            or not isinstance(request_metadata, dict)
            or request_metadata.get("task") != task
        ):
            raise ValueError("program memory/query embedding call lineage is invalid")

    executions = {str(row["execution_id"]): row for row in rows["program_executions"]}
    steps_by_execution: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows["program_execution_steps"]:
        steps_by_execution[str(row["execution_id"])].append(row)
    for plan_id, assessment in assessments.items():
        execution_id = str(assessment["execution_id"])
        if execution_id not in executions or str(executions[execution_id]["plan_id"]) != plan_id:
            raise ValueError("dry-run assessment references a missing execution")
        if not steps_by_execution.get(execution_id):
            raise ValueError("dry-run execution has no persisted operator step")
    assessment_ids = {str(row["assessment_id"]) for row in rows["program_dry_run_assessments"]}
    if any(
        str(row["assessment_id"]) not in assessment_ids
        for row in rows["program_dry_run_evidence_matches"]
    ):
        raise ValueError("dry-run evidence match has no assessment")

    status_counts = Counter(str(row["status"]) for row in outcomes.values())
    return {
        **questioner_report,
        "unit_count": len(unit_ids),
        "eligible_candidate_count": len(gates),
        "planner_batch_count": len(batches),
        "program_count": len(programs),
        "program_ready_count": sum(status_counts[value] for value in _READY_OUTCOMES),
        "dry_run_count": len(assessments),
        "program_query_embedding_batch_count": len(query_batches),
        "outcome_counts": dict(sorted(status_counts.items())),
        "memory_collection_ids": sorted(index_collection_ids),
        "planner_oracle_input_violation_count": 0,
        "program_requirement_dataflow_violation_count": 0,
    }


__all__ = ["verify_program_export"]
