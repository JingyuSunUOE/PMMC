"""Portable quality-pilot bundles for the frozen round-zero compiler."""

from __future__ import annotations

import hashlib
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
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

import zstandard

from pmmc.compiler.program_export import verify_program_export
from pmmc.evaluation.qbank_export import verify_qbank_coverage_export
from pmmc.pilots.io import (
    canonical_json_bytes,
    file_sha256,
    read_json,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)
from pmmc.pilots.review import _candidate_payloads, _coverage_payloads
from pmmc.schemas import (
    PROGRAM_PILOT_SCHEMA_VERSION,
    VERIFIED_QBANK_SCHEMA_VERSION,
    CompilerProgramFreezeManifest,
    CompilerProgramPilotAuditManifest,
    CompilerProgramPilotDecision,
    CompilerProgramPilotManifest,
    DatasetName,
    PilotDecisionValue,
    ProgramPilotArtifactRef,
    ProgramPilotStatus,
    QuestionerFreezeManifest,
    QuestionerPilotSample,
    VerifiedQBankFreezeManifest,
)
from pmmc.schemas.base import utc_now

_PROGRAM_PILOT_PROTOCOL = "compiler-program-quality-pilot-v3"
_PROGRAM_AUDIT_PROTOCOL = "compiler-program-audit-v3"
_MODEL_ARTIFACT_PROJECTION_VERSION = "portable-model-artifact-projection-v1"
_PORTABLE_REDACTION = "[REDACTED_PORTABLE_LOCATOR]"
_PORTABILITY_PROGRESS_INTERVAL_SECONDS = 30.0
_PUBLIC_FORBIDDEN_KEYS = {
    "canonical_question",
    "expected_answer",
    "expected_evidence",
    "gold_answer",
    "gold_evidence",
    "raw_query",
    "reviewer_id",
    "rationale",
}
_SECRET_MARKERS = (
    b"HF_TOKEN=",
    b"OPENAI_API_KEY=",
    b"AWS_SECRET_ACCESS_KEY=",
    b"AWS_SESSION_TOKEN=",
    b"X-Amz-Signature=",
    b"PMMC_GOLD_SENTINEL",
    b"GOLD_ANSWER_SENTINEL",
)
_FORBIDDEN_PORTABLE_PATTERNS = (
    (
        "presigned_url",
        re.compile(
            rb"(?:x-amz-(?:signature|credential)|"
            rb"x-goog-(?:signature|credential)|awsaccesskeyid)=",
            re.IGNORECASE,
        ),
    ),
    ("object_store_uri", re.compile(rb"(?:s3|minio)://", re.IGNORECASE)),
    ("file_uri", re.compile(rb"file://", re.IGNORECASE)),
    (
        "loopback_endpoint",
        re.compile(
            rb"https?://(?:127\.0\.0\.1|0\.0\.0\.0|localhost|"
            rb"minio(?=[:/])|postgres(?=[:/]))",
            re.IGNORECASE,
        ),
    ),
    ("host_data_path", re.compile(rb"/data/users/", re.IGNORECASE)),
    ("workspace_path", re.compile(rb"/workspace/", re.IGNORECASE)),
    ("kubelet_path", re.compile(rb"/var/lib/kubelet/", re.IGNORECASE)),
)
_MODEL_LOCATOR_KEYS = {
    "base_url",
    "endpoint",
    "image_url",
    "local_path",
    "object_key",
    "path",
    "presigned_url",
    "remote_url",
    "storage_uri",
    "uri",
    "url",
}
_GENERIC_LOCATOR_KEYS = {"path", "uri", "url"}
_READY_OUTCOMES = {
    "program_ready",
    "program_ready_with_fallback",
    "program_ready_with_abstain",
}
_REJECTED_OUTCOMES = {
    "parse_failed",
    "static_rejected",
    "policy_rejected",
    "lineage_rejected",
    "query_vector_unavailable",
    "execution_failed",
}


def _program_pilot_event(phase: str, **fields: object) -> None:
    payload = {
        "phase": phase,
        "stage": "program_pilot_bundle",
        **fields,
    }
    print(
        "PROGRAM_PILOT_BUNDLE "
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _redact_nonportable_string(value: str) -> tuple[str, int]:
    payload = value.encode("utf-8")
    redactions = 0
    replacement = _PORTABLE_REDACTION.encode("ascii")
    for _reason, pattern in _FORBIDDEN_PORTABLE_PATTERNS:
        payload, count = pattern.subn(replacement, payload)
        redactions += count
    return payload.decode("utf-8"), redactions


def _nonportable_string_reason(value: str) -> str | None:
    payload = value.encode("utf-8")
    for reason, pattern in _FORBIDDEN_PORTABLE_PATTERNS:
        if pattern.search(payload):
            return reason
    return None


def _project_model_artifact_value(value: object) -> tuple[object, int]:
    if isinstance(value, dict):
        projected: dict[str, object] = {}
        redactions = 0
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized_key = key.strip().lower()
            if (
                normalized_key in _MODEL_LOCATOR_KEYS
                and item is not None
                and (
                    normalized_key not in _GENERIC_LOCATOR_KEYS
                    or isinstance(item, str)
                )
            ):
                projected[key] = _PORTABLE_REDACTION
                redactions += 1
                continue
            projected_item, count = _project_model_artifact_value(item)
            projected[key] = projected_item
            redactions += count
        return projected, redactions
    if isinstance(value, list):
        projected_items: list[object] = []
        redactions = 0
        for item in value:
            projected_item, count = _project_model_artifact_value(item)
            projected_items.append(projected_item)
            redactions += count
        return projected_items, redactions
    if isinstance(value, str):
        return _redact_nonportable_string(value)
    return value, 0


def _structured_portability_issue(value: object) -> str | None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for raw_key, child in item.items():
                key = str(raw_key).strip().lower()
                if (
                    key in _MODEL_LOCATOR_KEYS
                    and child is not None
                    and child != _PORTABLE_REDACTION
                    and (
                        key not in _GENERIC_LOCATOR_KEYS
                        or isinstance(child, str)
                    )
                ):
                    if isinstance(child, str):
                        reason = _nonportable_string_reason(child)
                        if reason is not None:
                            return reason
                        if key in _GENERIC_LOCATOR_KEYS:
                            if key == "path":
                                relative = PurePosixPath(child)
                                if relative.is_absolute() or ".." in relative.parts:
                                    return "unsafe_relative_path"
                            continue
                        return f"structured_locator:{key}"
                    return f"structured_locator:{key}"
                if (
                    isinstance(child, str)
                    and (key == "relative_path" or key.endswith("_relative_path"))
                ):
                    relative = PurePosixPath(child)
                    if relative.is_absolute() or ".." in relative.parts:
                        return "unsafe_relative_path"
                pending.append(child)
        elif isinstance(item, list):
            pending.extend(item)
    return None


def _unredacted_model_locator_key(value: object) -> str | None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for raw_key, child in item.items():
                key = str(raw_key).strip().lower()
                if (
                    key in _MODEL_LOCATOR_KEYS
                    and child is not None
                    and child != _PORTABLE_REDACTION
                    and (
                        key not in _GENERIC_LOCATOR_KEYS
                        or isinstance(child, str)
                    )
                ):
                    return key
                pending.append(child)
        elif isinstance(item, list):
            pending.extend(item)
    return None


def _portable_model_artifact_payload(
    payload: bytes,
    *,
    role: str,
    mime_type: object,
) -> tuple[bytes, int]:
    """Project request manifests without changing safe provider responses."""

    if role != "input":
        return payload, 0
    is_json = isinstance(mime_type, str) and (
        mime_type == "application/json" or mime_type.endswith("+json")
    )
    if is_json:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("model request artifact is not valid JSON") from exc
        projected, redactions = _project_model_artifact_value(value)
        if redactions == 0:
            return payload, 0
        return canonical_json_bytes(projected).rstrip(b"\n"), redactions
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload, 0
    projected_text, redactions = _redact_nonportable_string(text)
    return projected_text.encode("utf-8"), redactions


def _index(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"program pilot source has duplicate {key}: {value}")
        result[value] = row
    return result


def _group(
    rows: Iterable[dict[str, Any]],
    key: str,
    *,
    sort_key: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row[key])].append(row)
    if sort_key is not None:
        for values in result.values():
            values.sort(key=lambda item: (int(item.get(sort_key, 0)), stable_hash(item)))
    return dict(result)


def _walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            keys.update(str(key) for key in item)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return keys


def _image_refs(value: object) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if item.get("image_id") and item.get("relative_path"):
                found.append(item)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return found


def _dedupe_evidence(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        key = (str(value.get("evidence_type")), str(value.get("evidence_id")))
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _runtime_source_identity(runtime_export: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verification = verify_program_export(runtime_export)
    export_manifest = read_json(runtime_export / "export_manifest.json")
    run_manifest = read_json(runtime_export / "runtime" / "run_manifest.json")
    compiler_rows = read_jsonl(runtime_export / "runtime" / "compiler_runs.jsonl")
    if len(compiler_rows) != 1:
        raise ValueError("program runtime export requires exactly one compiler run")
    compiler = compiler_rows[0]
    metadata = run_manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("program runtime export has no run metadata")
    questioner = metadata.get("questioner_config")
    if not isinstance(questioner, dict):
        raise ValueError("program runtime export has no Questioner config")
    model_commits: dict[str, str] = {}
    for alias, value in dict(run_manifest.get("model_configs", {})).items():
        if isinstance(value, dict):
            revision = value.get("model_revision") or value.get("expected_model_revision")
            if isinstance(revision, str) and revision:
                model_commits[str(alias)] = revision
    for call in read_jsonl(runtime_export / "runtime" / "model_calls.jsonl"):
        alias = call.get("model_alias")
        revision = call.get("actual_model_revision") or call.get("model_revision")
        if isinstance(alias, str) and isinstance(revision, str) and revision:
            model_commits[alias] = revision
    prompt_hashes = dict(run_manifest.get("prompt_hashes", {}))
    from pmmc.compiler.questioner_prompt import (
        questioner_prompt_sha256,
        questioner_request_contract_hash,
        questioner_response_schema_sha256,
    )

    prompt_version = str(questioner.get("prompt_version", ""))
    verifier_prompt_version = str(questioner.get("verifier_prompt_version", ""))
    opportunity_identity = {
        "opportunity_builder_version": questioner.get("opportunity_builder_version"),
        "unit_builder_version": questioner.get("unit_builder_version"),
        "dataset_unit_builders": questioner.get("dataset_unit_builders", {}),
        "relation_selector_version": questioner.get("relation_selector_version"),
        "fts_lexeme_limit": questioner.get("fts_lexeme_limit"),
        "fts_relation_top_k": questioner.get("fts_relation_top_k"),
        "max_opportunities_per_unit": questioner.get("max_opportunities_per_unit"),
        "request_contract_hash": questioner_request_contract_hash(
            prompt_version=prompt_version,
            renderer_version=str(questioner.get("renderer_version", "")),
            opportunity_builder_version=str(
                questioner.get("opportunity_builder_version", "")
            ),
        ),
    }
    prompt_hash = prompt_hashes.get(prompt_version) or questioner_prompt_sha256(prompt_version)
    verifier_prompt_hash = prompt_hashes.get(verifier_prompt_version) or questioner_prompt_sha256(
        verifier_prompt_version
    )
    verifier_schema_hash = questioner_response_schema_sha256(verifier_prompt_version)
    verifier_protocol_hash = stable_hash(
        {
            "prompt_version": verifier_prompt_version,
            "prompt_hash": verifier_prompt_hash,
            "schema_version": questioner.get("verifier_schema_version"),
            "schema_hash": verifier_schema_hash,
            "semantic_revision_max": questioner.get("semantic_revision_max"),
            "model_alias": questioner.get("compiler_model_alias"),
        }
    )
    planner_schema_hash = metadata.get("planner_response_schema_hash")
    if not isinstance(planner_schema_hash, str):
        raise ValueError("program runtime export has no Planner response-schema hash")
    questioner_identity: dict[str, Any] = {
        "questioner_protocol_version": questioner.get("protocol_version"),
        "questioner_config_hash": metadata.get("questioner_config_hash"),
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "response_schema_hash": questioner_response_schema_sha256(prompt_version),
        "renderer_version": questioner.get("renderer_version"),
        "parser_version": questioner.get("parser_version"),
        "unit_builder_version": questioner.get("unit_builder_version"),
        "gate_version": questioner.get("gate_version"),
        "dedupe_version": questioner.get("dedupe_version"),
        "family_dedupe_version": questioner.get("family_dedupe_version"),
        "matcher_version": questioner.get("matcher_version"),
        "opportunity_builder_version": questioner.get("opportunity_builder_version"),
        "relation_selector_version": questioner.get("relation_selector_version"),
        "opportunity_builder_hash": stable_hash(opportunity_identity),
        "verifier_prompt_version": verifier_prompt_version,
        "verifier_prompt_hash": verifier_prompt_hash,
        "verifier_schema_hash": verifier_schema_hash,
        "verifier_protocol_hash": verifier_protocol_hash,
    }
    program_identity: dict[str, Any] = {
        "questioner_freeze_manifest_hash": compiler.get("questioner_freeze_manifest_hash"),
        "planner_config_hash": compiler.get("planner_config_hash"),
        "planner_model_alias": compiler.get("planner_model_alias"),
        "planner_model_revision": compiler.get("planner_model_revision"),
        "planner_prompt_version": compiler.get("planner_prompt_version"),
        "planner_prompt_hash": compiler.get("planner_prompt_hash"),
        "planner_schema_hash": planner_schema_hash,
        "planner_renderer_version": compiler.get("planner_renderer_version"),
        "planner_parser_version": compiler.get("planner_parser_version"),
        "program_policy_version": compiler.get("program_policy_version"),
        "operator_set_version": compiler.get("operator_set_version"),
        "operator_registry_hash": compiler.get("operator_registry_hash"),
        "dry_run_policy_version": compiler.get("dry_run_policy_version"),
        "memory_index_representation_version": compiler.get("memory_index_representation_version"),
    }
    if compiler.get("compiler_schema_version") in {"0.6.0", "0.7.0", "0.8.0"}:
        routing_rows = read_jsonl(
            runtime_export / "runtime" / "routing_projection_manifests.jsonl"
        )
        if len(routing_rows) != 1:
            raise ValueError("Step 6D export requires one routing projection identity")
        program_identity.update(
            {
                "doubter_config_hash": compiler.get("doubter_config_hash"),
                "doubter_model_alias": compiler.get("doubter_model_alias"),
                "doubter_model_revision": compiler.get("doubter_model_revision"),
                "doubter_prompt_version": compiler.get("doubter_prompt_version"),
                "doubter_prompt_hash": compiler.get("doubter_prompt_hash"),
                "doubter_renderer_version": compiler.get("doubter_renderer_version"),
                "doubter_parser_version": compiler.get("doubter_parser_version"),
                "doubter_schema_version": compiler.get("doubter_schema_version"),
                "routing_projection_version": compiler.get(
                    "routing_projection_version"
                ),
                "active_round": compiler.get("active_round"),
                "max_round": compiler.get("max_round"),
                "routing_projection_hash": routing_rows[0].get("content_hash"),
            }
        )
    dataset = str(compiler.get("dataset") or run_manifest.get("dataset"))
    dataset_builders = questioner.get("dataset_unit_builders", {})
    if not isinstance(dataset_builders, dict):
        raise ValueError("program runtime export has invalid dataset unit-builder identities")
    effective_unit_builder = dataset_builders.get(dataset, questioner.get("unit_builder_version"))
    if not isinstance(effective_unit_builder, str) or not effective_unit_builder:
        raise ValueError("program runtime export has no effective dataset unit builder")
    identity: dict[str, Any] = {
        "runtime_export_id": str(export_manifest["export_id"]),
        "runtime_manifest_sha256": file_sha256(runtime_export / "export_manifest.json"),
        "run_id": str(export_manifest["run_id"]),
        "snapshot_id": str(compiler["snapshot_id"]),
        "dataset": dataset,
        "effective_unit_builder_version": effective_unit_builder,
        "questioner": questioner_identity,
        "program": program_identity,
        "model_commits": dict(sorted(model_commits.items())),
    }
    required = [
        *(
            value
            for key, value in questioner_identity.items()
            if key != "family_dedupe_version"
        ),
        *program_identity.values(),
    ]
    if any(value in {None, ""} for value in required) or not model_commits:
        raise ValueError("program runtime export lacks complete frozen identities")
    return identity, verification


def _program_candidate_facts(
    *,
    dataset: DatasetName,
    runtime_export: Path,
    snapshot_dir: Path,
) -> list[dict[str, Any]]:
    runtime = runtime_export / "runtime"
    candidate_cards = {
        str(item["qbank_id"]): item
        for item in _candidate_payloads(
            dataset=dataset,
            runtime_export=runtime_export,
            snapshot_dir=snapshot_dir,
        )
    }
    questions = _index(read_jsonl(runtime / "qbank_questions.jsonl"), "qbank_id")
    outcomes = _index(read_jsonl(runtime / "planner_candidate_outcomes.jsonl"), "qbank_id")
    programs = _index(read_jsonl(runtime / "memory_programs.jsonl"), "plan_id")
    steps = _group(read_jsonl(runtime / "program_steps.jsonl"), "plan_id", sort_key="step_id")
    expected = _group(
        read_jsonl(runtime / "program_expected_evidence_refs.jsonl"),
        "plan_id",
        sort_key="ordinal",
    )
    validations = _group(read_jsonl(runtime / "program_validation_results.jsonl"), "plan_id")
    issues = _group(
        read_jsonl(runtime / "program_validation_issues.jsonl"),
        "validation_result_id",
        sort_key="ordinal",
    )
    assessments = _index(read_jsonl(runtime / "program_dry_run_assessments.jsonl"), "plan_id")
    matches = _group(
        read_jsonl(runtime / "program_dry_run_evidence_matches.jsonl"),
        "assessment_id",
    )
    executions = _index(read_jsonl(runtime / "program_executions.jsonl"), "execution_id")
    execution_steps = _group(
        read_jsonl(runtime / "program_execution_steps.jsonl"),
        "execution_id",
        sort_key="step_id",
    )
    execution_evidence = _group(
        read_jsonl(runtime / "program_execution_evidence_refs.jsonl"),
        "execution_id",
        sort_key="ordinal",
    )
    links = _index(read_jsonl(runtime / "planner_batch_programs.jsonl"), "qbank_id")
    step6d = (runtime / "qbank_round_projection_manifests.jsonl").is_file()
    round_members_by_qbank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    doubter_by_qbank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    directives_by_feedback: dict[str, list[dict[str, Any]]] = defaultdict(list)
    revisions_by_qbank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions: dict[str, dict[str, Any]] = {}
    routing_by_qbank: dict[str, dict[str, Any]] = {}
    if step6d:
        projection_rounds = {
            str(item["projection_id"]): int(item["round_id"])
            for item in read_jsonl(
                runtime / "qbank_round_projection_manifests.jsonl"
            )
        }
        for item in read_jsonl(runtime / "qbank_round_projection_members.jsonl"):
            value = {
                **item,
                "round_id": projection_rounds[str(item["projection_id"])],
            }
            round_members_by_qbank[str(item["qbank_id"])].append(value)
        for values in round_members_by_qbank.values():
            values.sort(key=lambda item: int(item["round_id"]))
        for item in read_jsonl(runtime / "doubter_directives.jsonl"):
            directives_by_feedback[str(item["feedback_id"])].append(item)
        for item in read_jsonl(runtime / "doubter_assessments.jsonl"):
            value = dict(item)
            value["directives"] = sorted(
                directives_by_feedback.get(str(item["feedback_id"]), []),
                key=lambda row: int(row.get("ordinal", 0)),
            )
            doubter_by_qbank[str(item["qbank_id"])].append(value)
        for values in doubter_by_qbank.values():
            values.sort(key=lambda item: int(item["round_id"]))
        for item in read_jsonl(runtime / "planner_revision_lineage.jsonl"):
            revisions_by_qbank[str(item["qbank_id"])].append(item)
        for values in revisions_by_qbank.values():
            values.sort(key=lambda item: int(item["target_round_id"]))
        decisions = _index(read_jsonl(runtime / "qbank_decisions.jsonl"), "qbank_id")
        routing_by_qbank = _index(
            read_jsonl(runtime / "routing_projection_members.jsonl"),
            "qbank_id",
        )

    facts: list[dict[str, Any]] = []
    for qbank_id, question in sorted(
        questions.items(),
        key=lambda item: (
            str(item[1].get("compiler_unit_id")),
            int(item[1].get("generation_rank", 0)),
            int(item[1].get("candidate_version", 0)),
            item[0],
        ),
    ):
        card = candidate_cards[qbank_id]
        card["expected_evidence"] = _dedupe_evidence(card["expected_evidence"])
        outcome = outcomes.get(qbank_id)
        link = links.get(qbank_id)
        plan_id = str(outcome["plan_id"]) if outcome and outcome.get("plan_id") else None
        program = programs.get(plan_id) if plan_id is not None else None
        assessment = assessments.get(plan_id) if plan_id is not None else None
        status = str(outcome.get("status")) if outcome is not None else "not_planned"
        control = str(assessment.get("control_action")) if assessment else ""
        if status == "dry_run_insufficient":
            bucket = "dry_run_insufficient"
        elif status == "program_ready_with_fallback" or control == "fallback":
            bucket = "fallback"
        elif status == "program_ready_with_abstain" or control == "abstain":
            bucket = "abstain"
        elif status in _REJECTED_OUTCOMES:
            bucket = "planner_rejected"
        elif status in _READY_OUTCOMES:
            bucket = "program_ready"
        else:
            bucket = "questioner_rejected"
        validation_rows: list[dict[str, Any]] = []
        for value in validations.get(plan_id or "", []):
            enriched = dict(value)
            enriched["issues"] = issues.get(str(value["validation_result_id"]), [])
            validation_rows.append(enriched)
        execution = executions.get(str(assessment["execution_id"])) if assessment else None
        fact = {
            "dataset": dataset.value,
            "qbank_id": qbank_id,
            "compiler_unit_id": question.get("compiler_unit_id"),
            "generation_rank": question.get("generation_rank"),
            "candidate_version": question.get("candidate_version"),
            "candidate_origin": question.get("candidate_origin"),
            "bucket": bucket,
            "question": card,
            "planner_outcome": outcome,
            "planner_program_link": link,
            "program": program,
            "program_steps": steps.get(plan_id or "", []),
            "program_expected_evidence": _dedupe_evidence(expected.get(plan_id or "", [])),
            "validations": validation_rows,
            "dry_run_assessment": assessment,
            "dry_run_evidence_matches": (
                matches.get(str(assessment["assessment_id"]), []) if assessment else []
            ),
            "execution": execution,
            "execution_steps": (
                execution_steps.get(str(execution["execution_id"]), []) if execution else []
            ),
            "execution_evidence": (
                execution_evidence.get(str(execution["execution_id"]), []) if execution else []
            ),
            "visual": "image" in set(question.get("required_modalities", [])),
            "multi_hop": int(question.get("hop_count", 0)) > 1,
            "round_projections": round_members_by_qbank.get(qbank_id, []),
            "doubter_assessments": doubter_by_qbank.get(qbank_id, []),
            "planner_revisions": revisions_by_qbank.get(qbank_id, []),
            "active_r2_decision": decisions.get(qbank_id),
            "routing_projection_member": routing_by_qbank.get(qbank_id),
        }
        facts.append(fact)
    return facts


def _step6d_round_summary(
    runtime_exports: Mapping[DatasetName, Path],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset, export_dir in sorted(
        runtime_exports.items(), key=lambda item: item[0].value
    ):
        runtime = export_dir / "runtime"
        manifests = {
            str(item["projection_id"]): int(item["round_id"])
            for item in read_jsonl(
                runtime / "qbank_round_projection_manifests.jsonl"
            )
        }
        outcomes: dict[int, Counter[str]] = {
            round_id: Counter() for round_id in range(4)
        }
        accepted_sets: dict[int, set[str]] = {
            round_id: set() for round_id in range(4)
        }
        for item in read_jsonl(runtime / "qbank_round_projection_members.jsonl"):
            round_id = manifests[str(item["projection_id"])]
            outcome = str(item["outcome"])
            outcomes[round_id][outcome] += 1
            if outcome in {"accepted", "carried_forward"}:
                accepted_sets[round_id].add(str(item["qbank_id"]))
        assessments = read_jsonl(runtime / "doubter_assessments.jsonl")
        assessment_counts = {
            round_id: Counter(
                str(item["action"])
                for item in assessments
                if int(item["round_id"]) == round_id
            )
            for round_id in range(4)
        }
        datasets[dataset.value] = {
            "rounds": [
                {
                    "round_id": round_id,
                    "candidate_count": sum(outcomes[round_id].values()),
                    "outcome_counts": dict(sorted(outcomes[round_id].items())),
                    "doubter_action_counts": dict(
                        sorted(assessment_counts[round_id].items())
                    ),
                    "accepted_count": len(accepted_sets[round_id]),
                    "rescued_count": (
                        len(accepted_sets[round_id] - accepted_sets[round_id - 1])
                        if round_id > 0
                        else len(accepted_sets[0])
                    ),
                }
                for round_id in range(4)
            ],
            "active_round": 2,
            "maximum_round": 3,
            "active_r2_routing_member_count": len(
                read_jsonl(runtime / "routing_projection_members.jsonl")
            ),
            "decision_count": len(read_jsonl(runtime / "qbank_decisions.jsonl")),
            "revision_count": len(
                read_jsonl(runtime / "planner_revision_lineage.jsonl")
            ),
        }
    return {
        "protocol_version": "verified-qbank-round-analysis-v1",
        "active_round": 2,
        "maximum_round": 3,
        "datasets": datasets,
    }


def _stable_order(values: Iterable[dict[str, Any]], *, label: str) -> list[dict[str, Any]]:
    return sorted(
        values,
        key=lambda item: (
            stable_hash([label, item.get("dataset"), item.get("qbank_id"), item.get("query_id")]),
            str(item.get("qbank_id") or item.get("query_id")),
        ),
    )


def _bind_coverage_dataset(
    values: Sequence[dict[str, Any]], *, dataset: DatasetName
) -> list[dict[str, Any]]:
    bound: list[dict[str, Any]] = []
    for value in values:
        declared = value.get("dataset")
        if declared is not None and str(declared) != dataset.value:
            raise ValueError("program coverage payload belongs to another dataset")
        bound.append({**value, "dataset": dataset.value})
    return bound


def _semantic_source_identity(coverage_export: Path) -> dict[str, Any]:
    semantic_run = read_json(coverage_export / "private" / "semantic_match_run.json")
    if not semantic_run:
        raise ValueError("round-zero pilot requires a completed QBank semantic run")
    metadata = semantic_run.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("QBank semantic run has no frozen metadata")
    identity = {
        "semantic_run_id": semantic_run.get("semantic_run_id"),
        "semantic_schema_version": semantic_run.get("schema_version"),
        "semantic_protocol_version": semantic_run.get("protocol_version"),
        "semantic_config_hash": metadata.get("semantic_config_hash"),
        "semantic_prompt_version": semantic_run.get("prompt_version"),
        "semantic_prompt_hash": semantic_run.get("prompt_hash"),
        "semantic_renderer_version": semantic_run.get("renderer_version"),
        "semantic_parser_version": semantic_run.get("parser_version"),
        "semantic_response_schema_hash": metadata.get(
            "semantic_response_schema_hash"
        ),
        "semantic_judge_model_alias": semantic_run.get("judge_model_alias"),
        "semantic_judge_model_revision": semantic_run.get(
            "judge_model_revision"
        ),
        "semantic_pair_count": semantic_run.get("pair_count"),
    }
    required = [
        value
        for key, value in identity.items()
        if key != "semantic_pair_count"
    ]
    if any(value in {None, ""} for value in required):
        raise ValueError("QBank semantic run lacks complete frozen identities")
    return identity


def _annotate_semantic_matches(
    *,
    coverage_export: Path,
    coverage_cards: list[dict[str, Any]],
) -> None:
    assessments = {
        (str(item["query_id"]), str(item["qbank_id"])): item
        for item in read_jsonl(
            coverage_export / "private" / "semantic_match_assessments.jsonl"
        )
    }
    for card in coverage_cards:
        query_id = str(card["query_id"])
        for ranked in card.get("ranked_candidates", []):
            assessment = assessments.get((query_id, str(ranked.get("qbank_id"))))
            ranked["semantic_match_status"] = (
                "evaluated" if assessment is not None else "not_evaluated"
            )
            ranked["semantic_match_assessment"] = assessment


def _model_artifact_reference_locator(
    *,
    dataset: DatasetName | str,
    scope: str,
    artifact_id: object,
    model_call_id: object,
    role: object,
) -> str:
    dataset_value = dataset.value if isinstance(dataset, DatasetName) else str(dataset)
    return stable_hash(
        {
            "dataset": dataset_value,
            "scope": scope,
            "artifact_id": artifact_id,
            "model_call_id": model_call_id,
            "role": role,
        }
    )[:16]


def _semantic_artifact_call_lineage(
    coverage_export: Path,
) -> dict[tuple[str, str], tuple[str, str]]:
    lineage: dict[tuple[str, str], tuple[str, str]] = {}
    calls_path = coverage_export / "private" / "semantic_model_calls.jsonl"
    if not calls_path.is_file():
        raise ValueError("semantic model-call export is unavailable")
    for call in read_jsonl(calls_path):
        model_call_id = str(call.get("model_call_id") or "")
        semantic_job_id = str(call.get("semantic_job_id") or "")
        if not model_call_id or not semantic_job_id:
            raise ValueError("semantic model-call artifact lineage is incomplete")
        for role, raw_artifact_id in (
            ("input", call.get("input_asset_id")),
            ("output", call.get("output_asset_id")),
        ):
            if raw_artifact_id is None:
                continue
            artifact_id = str(raw_artifact_id)
            key = (artifact_id, role)
            prior = lineage.get(key)
            value = (model_call_id, semantic_job_id)
            if not artifact_id or (prior is not None and prior != value):
                raise ValueError("semantic artifact has ambiguous model-call lineage")
            lineage[key] = value
    return lineage


def _prepare_model_artifact_source_rows(
    *,
    runtime_exports: Mapping[DatasetName, Path],
    coverage_exports: Mapping[DatasetName, Path],
    query_runtime_exports: Mapping[DatasetName, Path] | None = None,
) -> list[tuple[DatasetName, str, Path, dict[str, Any]]]:
    sources = [
        (
            dataset,
            "runtime",
            runtime_exports[dataset],
            runtime_exports[dataset]
            / "runtime"
            / "private"
            / "model_artifact_manifest.jsonl",
        )
        for dataset in sorted(runtime_exports, key=lambda item: item.value)
    ]
    sources.extend(
        (
            dataset,
            "semantic",
            coverage_exports[dataset],
            coverage_exports[dataset]
            / "private"
            / "semantic_model_artifact_manifest.jsonl",
        )
        for dataset in sorted(coverage_exports, key=lambda item: item.value)
    )
    if query_runtime_exports is not None:
        sources.extend(
            (
                dataset,
                "query_runtime",
                query_runtime_exports[dataset],
                query_runtime_exports[dataset]
                / "runtime"
                / "private"
                / "model_artifact_manifest.jsonl",
            )
            for dataset in sorted(query_runtime_exports, key=lambda item: item.value)
        )
    manifest_row_count = sum(len(read_jsonl(source_manifest)) for *_, source_manifest in sources)
    _program_pilot_event(
        "model_artifact_lineage_preflight_started",
        reference_count=manifest_row_count,
        source_manifest_count=len(sources),
    )
    prepared: list[tuple[DatasetName, str, Path, dict[str, Any]]] = []
    reference_keys: set[tuple[str, str, str, str]] = set()
    derived_reference_count = 0
    try:
        for dataset, scope, source_root, source_manifest in sources:
            semantic_lineage = (
                _semantic_artifact_call_lineage(source_root)
                if scope == "semantic"
                else {}
            )
            for source_row in read_jsonl(source_manifest):
                artifact_id = str(source_row.get("artifact_id") or "")
                role = str(source_row.get("role") or "")
                model_call_id = str(source_row.get("model_call_id") or "")
                semantic_job_id = source_row.get("semantic_job_id")
                lineage_source = "manifest"
                expected_semantic = semantic_lineage.get((artifact_id, role))
                if scope == "semantic":
                    if expected_semantic is None:
                        rule = "semantic_artifact_not_bound_to_model_call"
                        raise ValueError(rule)
                    expected_call_id, expected_job_id = expected_semantic
                    if model_call_id and model_call_id != expected_call_id:
                        rule = "semantic_model_call_mismatch"
                        raise ValueError(rule)
                    if semantic_job_id is not None and str(semantic_job_id) != expected_job_id:
                        rule = "semantic_job_mismatch"
                        raise ValueError(rule)
                    if not model_call_id:
                        model_call_id = expected_call_id
                        lineage_source = "semantic_model_calls"
                        derived_reference_count += 1
                    semantic_job_id = expected_job_id
                if not artifact_id:
                    rule = "artifact_id_missing"
                    raise ValueError(rule)
                if not model_call_id:
                    rule = "model_call_id_missing"
                    raise ValueError(rule)
                if role not in {"input", "output"}:
                    rule = "artifact_role_invalid"
                    raise ValueError(rule)
                reference_key = (dataset.value, scope, model_call_id, role)
                if reference_key in reference_keys:
                    rule = "duplicate_model_call_role"
                    raise ValueError(rule)
                reference_keys.add(reference_key)
                prepared.append(
                    (
                        dataset,
                        scope,
                        source_root,
                        {
                            **source_row,
                            "model_call_id": model_call_id,
                            "semantic_job_id": semantic_job_id,
                            "model_call_lineage_source": lineage_source,
                        },
                    )
                )
    except Exception as exc:
        row = source_row if "source_row" in locals() else {}
        safe_rule = str(exc)
        if re.fullmatch(r"[a-z0-9_]+", safe_rule) is None:
            safe_rule = "source_lineage_invalid"
        _program_pilot_event(
            "model_artifact_lineage_preflight_failed",
            completed_reference_count=len(prepared),
            error_type=type(exc).__name__,
            reference_locator=_model_artifact_reference_locator(
                dataset=dataset,
                scope=scope,
                artifact_id=row.get("artifact_id"),
                model_call_id=row.get("model_call_id"),
                role=row.get("role"),
            )
            if "dataset" in locals() and "scope" in locals()
            else None,
            rule=safe_rule,
            total_reference_count=manifest_row_count,
        )
        raise
    _program_pilot_event(
        "model_artifact_lineage_preflight_completed",
        derived_reference_count=derived_reference_count,
        reference_count=len(prepared),
        runtime_reference_count=sum(scope == "runtime" for _, scope, _, _ in prepared),
        query_runtime_reference_count=sum(
            scope == "query_runtime" for _, scope, _, _ in prepared
        ),
        semantic_reference_count=sum(scope == "semantic" for _, scope, _, _ in prepared),
    )
    return prepared


def _step6d_model_artifact_selection(
    *,
    runtime_exports: Mapping[DatasetName, Path],
    coverage_exports: Mapping[DatasetName, Path],
    candidate_cards: Sequence[Mapping[str, Any]],
    coverage_cards: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], set[str]]:
    """Select portable raw payloads while full artifacts remain in source exports."""

    selected: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    candidate_ids: dict[str, set[str]] = defaultdict(set)
    semantic_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for card in candidate_cards:
        candidate_ids[str(card["dataset"])].add(str(card["qbank_id"]))
    for card in coverage_cards:
        dataset = str(card["dataset"])
        query_id = str(card["query_id"])
        ranked = card.get("ranked_candidates", [])
        if not isinstance(ranked, list):
            continue
        for item in ranked[:5]:
            if isinstance(item, Mapping) and item.get("qbank_id") is not None:
                semantic_pairs[dataset].add((query_id, str(item["qbank_id"])))

    def add(dataset: str, scope: str, model_call_id: object, reason: str) -> None:
        if isinstance(model_call_id, str) and model_call_id:
            selected[(dataset, scope, model_call_id)].add(reason)

    for dataset in sorted(runtime_exports, key=lambda item: item.value):
        dataset_value = dataset.value
        runtime = runtime_exports[dataset] / "runtime"
        audited_ids = candidate_ids[dataset_value]

        for row in read_jsonl(runtime / "structured_output_item_diagnostics.jsonl"):
            add(
                dataset_value,
                "runtime",
                row.get("model_call_id"),
                "structured_output_diagnostic",
            )

        generation_by_id = {
            str(row["generation_id"]): row
            for row in read_jsonl(runtime / "questioner_generations.jsonl")
        }
        selected_generation_ids = {
            str(row["generation_id"])
            for row in read_jsonl(runtime / "qbank_questions.jsonl")
            if str(row["qbank_id"]) in audited_ids
            and row.get("generation_id") is not None
        }
        for generation_id, row in generation_by_id.items():
            if generation_id in selected_generation_ids:
                add(
                    dataset_value,
                    "runtime",
                    row.get("model_call_id"),
                    "candidate_audit_sample",
                )
            if int(row.get("repair_attempt_count", 0)) > 0 or int(
                row.get("invalid_candidate_count", 0)
            ) > 0:
                add(
                    dataset_value,
                    "runtime",
                    row.get("model_call_id"),
                    "questioner_repair_or_invalid",
                )

        for row in read_jsonl(runtime / "questioner_verification_batches.jsonl"):
            if str(row.get("scope_qbank_id") or "") in audited_ids:
                add(
                    dataset_value,
                    "runtime",
                    row.get("model_call_id"),
                    "candidate_audit_sample",
                )
            if int(row.get("reject_count", 0)) > 0 or int(
                row.get("revise_count", 0)
            ) > 0:
                add(
                    dataset_value,
                    "runtime",
                    row.get("model_call_id"),
                    "questioner_verify_nonpass",
                )

        for row in read_jsonl(runtime / "planner_generation_batches.jsonl"):
            raw_ids = row.get("candidate_qbank_ids", [])
            row_ids = (
                {str(value) for value in raw_ids}
                if isinstance(raw_ids, list)
                else set()
            )
            if row_ids & audited_ids:
                add(
                    dataset_value,
                    "runtime",
                    row.get("model_call_id"),
                    "candidate_audit_sample",
                )
            if int(row.get("invalid_count", 0)) > 0:
                add(
                    dataset_value,
                    "runtime",
                    row.get("model_call_id"),
                    "planner_invalid",
                )

        for row in read_jsonl(runtime / "doubter_assessments.jsonl"):
            qbank_id = str(row.get("qbank_id") or "")
            action = str(row.get("action") or "")
            if qbank_id in audited_ids:
                add(
                    dataset_value,
                    "runtime",
                    row.get("producer_model_call_id"),
                    "candidate_audit_sample",
                )
            if action != "pass":
                add(
                    dataset_value,
                    "runtime",
                    row.get("producer_model_call_id"),
                    "doubter_nonpass",
                )

        for row in read_jsonl(runtime / "planner_revision_lineage.jsonl"):
            add(
                dataset_value,
                "runtime",
                row.get("producer_model_call_id"),
                "planner_revision",
            )

        coverage = coverage_exports[dataset] / "private"
        semantic_jobs = {
            str(row["semantic_job_id"]): row
            for row in read_jsonl(coverage / "semantic_match_jobs.jsonl")
        }
        for row in read_jsonl(
            coverage / "semantic_structured_output_diagnostics.jsonl"
        ):
            add(
                dataset_value,
                "semantic",
                row.get("model_call_id"),
                "structured_output_diagnostic",
            )
        for row in read_jsonl(coverage / "semantic_model_calls.jsonl"):
            job = semantic_jobs.get(str(row.get("semantic_job_id") or ""))
            if job is None:
                continue
            pair = (str(job.get("query_id") or ""), str(job.get("qbank_id") or ""))
            if pair in semantic_pairs[dataset_value]:
                add(
                    dataset_value,
                    "semantic",
                    row.get("model_call_id"),
                    "coverage_audit_top5",
                )
    return dict(selected)


def _archive_model_artifacts(
    *,
    runtime_exports: Mapping[DatasetName, Path],
    coverage_exports: Mapping[DatasetName, Path],
    query_runtime_exports: Mapping[DatasetName, Path] | None = None,
    destination_root: Path,
    selected_calls: Mapping[tuple[str, str, str], set[str]] | None = None,
) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    written: dict[str, Path] = {}
    source_rows = _prepare_model_artifact_source_rows(
        runtime_exports=runtime_exports,
        coverage_exports=coverage_exports,
        query_runtime_exports=query_runtime_exports,
    )
    if selected_calls is not None:
        source_rows = [
            value
            for value in source_rows
            if (
                value[0].value,
                value[1],
                str(value[3].get("model_call_id") or ""),
            )
            in selected_calls
        ]
        if not source_rows:
            raise ValueError("Step 6D artifact selection produced an empty archive")
    started = time.monotonic()
    next_report = started + _PORTABILITY_PROGRESS_INTERVAL_SECONDS
    source_bytes = 0
    portable_bytes = 0
    projected_count = 0
    redaction_count = 0
    _program_pilot_event(
        "model_artifact_projection_started",
        reference_count=len(source_rows),
        projection_version=_MODEL_ARTIFACT_PROJECTION_VERSION,
    )
    try:
        for ordinal, (dataset, scope, source_root, row) in enumerate(
            source_rows,
            start=1,
        ):
            relative = Path(str(row.get("relative_path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("model artifact manifest contains an unsafe path")
            source = source_root / relative
            source_digest = str(row.get("sha256") or "")
            if not source.is_file():
                raise ValueError("referenced model artifact failed source verification")
            source_payload = source.read_bytes()
            if (
                hashlib.sha256(source_payload).hexdigest() != source_digest
                or len(source_payload) != int(row.get("size_bytes", -1))
            ):
                raise ValueError("referenced model artifact failed source verification")
            payload, item_redactions = _portable_model_artifact_payload(
                source_payload,
                role=str(row.get("role") or ""),
                mime_type=row.get("mime_type"),
            )
            digest = hashlib.sha256(payload).hexdigest()
            suffix = source.suffix if source.suffix in {".json", ".bin"} else ".bin"
            target = (
                destination_root
                / "sha256"
                / digest[:2]
                / f"{digest}{suffix}"
            )
            prior = written.get(digest)
            if prior is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                written[digest] = target
            elif prior.read_bytes() != payload:
                raise ValueError("content-addressed model artifact hash collision")
            else:
                target = prior
            source_bytes += len(source_payload)
            portable_bytes += len(payload)
            redaction_count += item_redactions
            projected_count += item_redactions > 0
            manifest_rows.append(
                {
                    "dataset": dataset.value,
                    "scope": scope,
                    "artifact_id": row.get("artifact_id"),
                    "model_call_id": row.get("model_call_id"),
                    "semantic_job_id": row.get("semantic_job_id"),
                    "model_call_lineage_source": row.get(
                        "model_call_lineage_source"
                    ),
                    "role": row.get("role"),
                    "artifact_type": row.get("artifact_type"),
                    "relative_path": target.relative_to(
                        destination_root.parent.parent
                    ).as_posix(),
                    "sha256": digest,
                    "size_bytes": len(payload),
                    "source_sha256": source_digest,
                    "source_size_bytes": len(source_payload),
                    "projection_version": _MODEL_ARTIFACT_PROJECTION_VERSION,
                    "projection_applied": item_redactions > 0,
                    "projection_redaction_count": item_redactions,
                    "mime_type": row.get("mime_type"),
                    "selection_reasons": sorted(
                        selected_calls.get(
                            (
                                dataset.value,
                                scope,
                                str(row.get("model_call_id") or ""),
                            ),
                            set(),
                        )
                    )
                    if selected_calls is not None
                    else ["complete_step6c_archive"],
                }
            )
            now = time.monotonic()
            if now >= next_report:
                _program_pilot_event(
                    "model_artifact_projection_progress",
                    completed_reference_count=ordinal,
                    elapsed_seconds=round(now - started, 1),
                    portable_bytes=portable_bytes,
                    projected_reference_count=projected_count,
                    redaction_count=redaction_count,
                    source_bytes=source_bytes,
                    total_reference_count=len(source_rows),
                    unique_payload_count=len(written),
                )
                next_report = now + _PORTABILITY_PROGRESS_INTERVAL_SECONDS
    except Exception as exc:
        _program_pilot_event(
            "model_artifact_projection_failed",
            completed_reference_count=len(manifest_rows),
            elapsed_seconds=round(time.monotonic() - started, 1),
            error_type=type(exc).__name__,
            portable_bytes=portable_bytes,
            projected_reference_count=projected_count,
            redaction_count=redaction_count,
            source_bytes=source_bytes,
            total_reference_count=len(source_rows),
            unique_payload_count=len(written),
        )
        raise
    manifest_rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["scope"]),
            str(row["model_call_id"]),
            str(row["role"]),
            str(row["artifact_id"]),
        )
    )
    _program_pilot_event(
        "model_artifact_projection_completed",
        completed_reference_count=len(manifest_rows),
        elapsed_seconds=round(time.monotonic() - started, 1),
        portable_bytes=portable_bytes,
        projected_reference_count=projected_count,
        redaction_count=redaction_count,
        source_bytes=source_bytes,
        unique_payload_count=len(written),
    )
    return manifest_rows


def _select_candidate_cards(
    facts: Sequence[dict[str, Any]], *, dataset: DatasetName, count: int = 80
) -> list[dict[str, Any]]:
    if len(facts) < count:
        raise ValueError(
            f"program pilot requires {count} candidate facts for {dataset.value}; "
            f"available={len(facts)}"
        )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(values: Iterable[dict[str, Any]], target: int, label: str) -> None:
        for item in _stable_order(values, label=label):
            qbank_id = str(item["qbank_id"])
            if qbank_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(qbank_id)
            if len(selected_ids) >= target:
                return

    visual = [item for item in facts if item["visual"]]
    multi_hop = [item for item in facts if item["multi_hop"]]
    add(visual, min(10, len(visual)), f"{dataset.value}:visual")
    multi_target = len(selected) + min(10, len(multi_hop))
    add(multi_hop, min(count, multi_target), f"{dataset.value}:multi-hop")

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in facts:
        buckets[str(item["bucket"])].append(item)
    ordered_buckets = [
        "program_ready",
        "planner_rejected",
        "dry_run_insufficient",
        "fallback",
        "abstain",
        "questioner_rejected",
    ]
    positions = {key: 0 for key in ordered_buckets}
    ordered = {
        key: _stable_order(buckets.get(key, []), label=f"{dataset.value}:{key}")
        for key in ordered_buckets
    }
    while len(selected) < count:
        progressed = False
        for key in ordered_buckets:
            values = ordered[key]
            while positions[key] < len(values):
                item = values[positions[key]]
                positions[key] += 1
                if str(item["qbank_id"]) in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(str(item["qbank_id"]))
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            add(facts, count, f"{dataset.value}:remainder")
            break
    if len(selected) != count:
        raise ValueError("program pilot candidate-card allocation did not close")
    return sorted(selected, key=lambda item: str(item["qbank_id"]))


def _select_coverage_cards(
    values: Sequence[dict[str, Any]], *, dataset: DatasetName, count: int = 20
) -> list[dict[str, Any]]:
    if len(values) < count:
        raise ValueError(
            f"program pilot requires {count} coverage facts for {dataset.value}; "
            f"available={len(values)}"
        )
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in values:
        buckets[str(item.get("bucket", "success"))].append(item)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    order = [
        "questioner_coverage_failure",
        "matching_failure",
        "gold_unresolvable",
        "success",
    ]
    positions = {key: 0 for key in order}
    ordered = {
        key: _stable_order(buckets.get(key, []), label=f"{dataset.value}:coverage:{key}")
        for key in order
    }
    while len(selected) < count:
        progressed = False
        for key in order:
            values_for_bucket = ordered[key]
            while positions[key] < len(values_for_bucket):
                item = values_for_bucket[positions[key]]
                positions[key] += 1
                query_id = str(item["query_id"])
                if query_id in seen:
                    continue
                selected.append(item)
                seen.add(query_id)
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError("program pilot coverage-card allocation did not close")
    return sorted(selected, key=lambda item: str(item["query_id"]))


def _annotate_thumbnails(
    values: object,
    *,
    benchmark_root: Path,
    thumbnail_root: Path,
) -> int:
    refs = _image_refs(values)
    generated: dict[tuple[str, str], str] = {}
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - project images install Pillow.
        raise RuntimeError("program pilot thumbnail export requires Pillow") from exc
    benchmark_root = benchmark_root.resolve()
    for ref in sorted(refs, key=lambda item: (str(item["image_id"]), str(item["relative_path"]))):
        relative = Path(str(ref["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("program pilot image path must be benchmark-relative")
        source = (benchmark_root / relative).resolve()
        if not source.is_relative_to(benchmark_root) or not source.is_file():
            raise ValueError(f"program pilot image is unavailable: {relative.as_posix()}")
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        expected_sha = ref.get("sha256")
        if isinstance(expected_sha, str) and expected_sha and expected_sha != source_sha:
            raise ValueError("program pilot source image checksum mismatch")
        key = (str(ref["image_id"]), source_sha)
        target_relative = generated.get(key)
        if target_relative is None:
            payload = io.BytesIO()
            with Image.open(source) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                image.thumbnail((640, 640))
                image.save(payload, format="JPEG", quality=88, optimize=True)
            thumbnail_bytes = payload.getvalue()
            thumbnail_sha = hashlib.sha256(thumbnail_bytes).hexdigest()
            target = thumbnail_root / "sha256" / thumbnail_sha[:2] / f"{thumbnail_sha}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != thumbnail_bytes:
                raise ValueError("program pilot thumbnail hash collision")
            target.write_bytes(thumbnail_bytes)
            target_relative = target.relative_to(thumbnail_root.parent.parent).as_posix()
            generated[key] = target_relative
        ref["thumbnail_relative_path"] = target_relative
        ref["source_sha256"] = source_sha
    return len(generated)


def _write_zstd_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstandard.ZstdCompressor(
        level=9,
        threads=0,
        write_checksum=True,
        write_content_size=True,
    )
    path.write_bytes(compressor.compress(b"".join(canonical_json_bytes(row) for row in rows)))


def _read_zstd_jsonl(path: Path) -> list[dict[str, Any]]:
    payload = zstandard.ZstdDecompressor().decompress(path.read_bytes())
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"program pilot fact is not an object at line {number}")
        rows.append(value)
    return rows


def _portable_scan_issue(
    path: Path,
    *,
    inspect_structured: bool = True,
    strict_text: bool = False,
) -> str | None:
    """Return a safe rule code without loading large model artifacts into memory."""

    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        return None

    def inspect(chunks: Iterable[bytes]) -> str | None:
        tail = b""
        for chunk in chunks:
            payload = tail + chunk
            for reason, pattern in _FORBIDDEN_PORTABLE_PATTERNS:
                if reason == "presigned_url" and pattern.search(payload):
                    return reason
            if any(marker in payload for marker in _SECRET_MARKERS):
                return "secret_or_sentinel"
            for reason, pattern in _FORBIDDEN_PORTABLE_PATTERNS:
                if reason != "presigned_url" and strict_text and pattern.search(payload):
                    return reason
            tail = payload[-512:]
        return None

    def structured_issue() -> str | None:
        if not inspect_structured:
            return None
        if path.name.endswith(".jsonl.zst"):
            rows = _read_zstd_jsonl(path)
            return _structured_portability_issue(rows)
        if path.suffix.lower() == ".json":
            try:
                return _structured_portability_issue(json.loads(path.read_bytes()))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
        if path.suffix.lower() == ".jsonl":
            return _structured_portability_issue(read_jsonl(path))
        return None

    if path.name.endswith(".jsonl.zst"):
        with path.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as decompressed:
                issue = inspect(iter(lambda: decompressed.read(1024 * 1024), b""))
        return issue or structured_issue()
    with path.open("rb") as artifact_stream:
        issue = inspect(iter(lambda: artifact_stream.read(1024 * 1024), b""))
    return issue or structured_issue()


def _verify_transfer_archive(
    *,
    pilot_dir: Path,
    archive_path: Path,
    expected_files: Sequence[Path],
) -> None:
    tar_payload = zstandard.ZstdDecompressor().decompress(archive_path.read_bytes())
    expected = {
        (PurePosixPath(pilot_dir.name) / path.relative_to(pilot_dir).as_posix()).as_posix(): path
        for path in expected_files
    }
    seen: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(tar_payload), mode="r:") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile():
                raise ValueError("program pilot transfer contains an unsafe archive member")
            expected_path = expected.get(name.as_posix())
            if expected_path is None or name.as_posix() in seen:
                raise ValueError("program pilot transfer inventory differs from its manifest")
            if (
                member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or stat.S_IMODE(member.mode) != 0o600
            ):
                raise ValueError("program pilot transfer metadata is not deterministic")
            extracted = archive.extractfile(member)
            if extracted is None or extracted.read() != expected_path.read_bytes():
                raise ValueError("program pilot transfer payload differs from its source")
            seen.add(name.as_posix())
    if seen != set(expected):
        raise ValueError("program pilot transfer archive is incomplete")


def _summaries(
    facts_by_dataset: Mapping[DatasetName, Sequence[dict[str, Any]]],
    verifications: Mapping[DatasetName, Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    questioner: dict[str, Any] = {"datasets": {}}
    planner: dict[str, Any] = {"datasets": {}}
    dry_run: dict[str, Any] = {"datasets": {}}
    operator_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for dataset in sorted(facts_by_dataset, key=lambda item: item.value):
        facts = list(facts_by_dataset[dataset])
        statuses = Counter(
            str((item.get("planner_outcome") or {}).get("status", "not_planned")) for item in facts
        )
        eligible = [item for item in facts if item.get("planner_outcome") is not None]
        programs = [item for item in eligible if item.get("program") is not None]
        assessments = [
            item["dry_run_assessment"]
            for item in programs
            if item.get("dry_run_assessment") is not None
        ]
        questioner["datasets"][dataset.value] = {
            "unit_count": verifications[dataset].get("unit_count"),
            "eligible_candidate_count": verifications[dataset].get("eligible_candidate_count"),
            "candidate_count": len(facts),
            "visual_candidate_count": sum(bool(item["visual"]) for item in facts),
            "multi_hop_candidate_count": sum(bool(item["multi_hop"]) for item in facts),
        }
        planner["datasets"][dataset.value] = {
            "eligible_candidate_count": len(eligible),
            "program_count": len(programs),
            "program_rate": len(programs) / len(eligible) if eligible else None,
            "outcome_counts": dict(sorted(statuses.items())),
            "program_ready_count": sum(statuses[value] for value in _READY_OUTCOMES),
        }
        recalls = [
            float(item["expected_evidence_recall"])
            for item in assessments
            if item.get("expected_evidence_recall") is not None
        ]
        dry_run["datasets"][dataset.value] = {
            "assessment_count": len(assessments),
            "execution_success_count": sum(
                bool(item.get("execution_success")) for item in assessments
            ),
            "nonempty_evidence_count": sum(
                bool(item.get("nonempty_evidence")) for item in assessments
            ),
            "evidence_complete_count": sum(
                bool(item.get("expected_evidence_complete")) for item in assessments
            ),
            "mean_expected_evidence_recall": (sum(recalls) / len(recalls) if recalls else None),
            "fallback_count": sum(item.get("control_action") == "fallback" for item in assessments),
            "abstain_count": sum(item.get("control_action") == "abstain" for item in assessments),
            "raw_image_reached_count": sum(
                bool(item.get("raw_image_reached")) for item in assessments
            ),
            "temporal_operator_used_count": sum(
                bool(item.get("temporal_operator_used")) for item in assessments
            ),
            "context_evidence_contributed_count": sum(
                bool(item.get("context_evidence_contributed")) for item in assessments
            ),
            "temporal_evidence_contributed_count": sum(
                bool(item.get("temporal_evidence_contributed")) for item in assessments
            ),
        }
        answerable_programs = [
            item
            for item in programs
            if item["question"].get("expected_answer") is not None
        ]
        answerable_assessments = [
            item["dry_run_assessment"]
            for item in answerable_programs
            if item.get("dry_run_assessment") is not None
        ]
        refusal_assessments = [
            item["dry_run_assessment"]
            for item in programs
            if item["question"].get("expected_answer") is None
            and item.get("dry_run_assessment") is not None
        ]
        dry_run["datasets"][dataset.value].update(
            {
                "program_execution_success_denominator": len(
                    answerable_assessments
                ),
                "program_execution_success_count": sum(
                    bool(item.get("execution_success"))
                    and bool(item.get("nonempty_evidence"))
                    for item in answerable_assessments
                ),
                "program_requirement_success_count": sum(
                    bool(item.get("evidence_sufficient"))
                    for item in answerable_assessments
                ),
                "fallback_success_count": sum(
                    bool(item.get("execution_success"))
                    and bool(item.get("fallback_triggered"))
                    for item in answerable_assessments
                ),
                "refusal_control_denominator": len(refusal_assessments),
                "refusal_control_success_count": sum(
                    bool(item.get("execution_success"))
                    and bool(item.get("abstain_triggered"))
                    for item in refusal_assessments
                ),
            }
        )
        sequence_counts = Counter(
            tuple(str(step["op_name"]) for step in item["program_steps"]) for item in programs
        )
        for sequence, count in sorted(sequence_counts.items()):
            operator_rows.append(
                {
                    "dataset": dataset.value,
                    "operator_sequence": list(sequence),
                    "step_count": len(sequence),
                    "count": count,
                }
            )
        for status, count in sorted(statuses.items()):
            if status not in _READY_OUTCOMES:
                failure_rows.append(
                    {"dataset": dataset.value, "failure_type": status, "count": count}
                )
    return questioner, planner, dry_run, operator_rows, failure_rows


def _automatic_integrity(
    verifications: Mapping[DatasetName, Mapping[str, Any]],
    coverage_verifications: Mapping[DatasetName, Mapping[str, Any]],
) -> tuple[list[str], int]:
    issues: set[str] = set()
    severe_terminal_errors = 0
    for dataset, verification in verifications.items():
        step6d = verification.get("protocol_version") == "pmmc-questioner-v3.3"
        if verification.get("valid") is not True:
            issues.add(f"{dataset.value}:program_export_invalid")
        if verification.get("protocol_version") not in {
            "pmmc-questioner-v3.2",
            "pmmc-questioner-v3.3",
        }:
            issues.add(f"{dataset.value}:questioner_protocol_mismatch")
        if int(verification.get("questioner_native_output_issue_count", 0)):
            issues.add(f"{dataset.value}:questioner_structured_output_fallback")
        if int(verification.get("planner_native_output_issue_count", 0)):
            issues.add(f"{dataset.value}:planner_structured_output_fallback")
        if step6d and int(
            verification.get("doubter_native_output_issue_count", -1)
        ):
            issues.add(f"{dataset.value}:doubter_structured_output_fallback")
        if int(verification.get("verifier_scope_contamination_count", -1)) != 0:
            issues.add(f"{dataset.value}:questioner_verifier_scope_contamination")
        if int(verification.get("planner_oracle_input_violation_count", -1)) != 0:
            issues.add(f"{dataset.value}:planner_oracle_input_violation")
        if (
            int(
                verification.get(
                    "program_requirement_dataflow_violation_count",
                    -1,
                )
            )
            != 0
        ):
            issues.add(f"{dataset.value}:program_requirement_dataflow_violation")
        severe = int(verification.get("severe_terminal_error_count", 0))
        severe_terminal_errors += severe
        if severe:
            issues.add(f"{dataset.value}:severe_terminal_candidate_error")
        collection_ids = verification.get("memory_collection_ids")
        if not isinstance(collection_ids, list) or len(set(collection_ids)) != 2:
            issues.add(f"{dataset.value}:memory_collection_identity_incomplete")
        if int(verification.get("eligible_candidate_count", -1)) < 0:
            issues.add(f"{dataset.value}:candidate_denominator_missing")
        if int(verification.get("program_count", -1)) < 0:
            issues.add(f"{dataset.value}:program_denominator_missing")
        if step6d:
            if verification.get("r0_hard_gate_passed") is not True:
                issues.add(f"{dataset.value}:r0_input_gate_failed")
            if int(verification.get("projection_count", -1)) != 4:
                issues.add(f"{dataset.value}:round_projection_incomplete")
            if int(verification.get("active_round", -1)) != 2:
                issues.add(f"{dataset.value}:active_round_mismatch")
            if int(verification.get("max_round", -1)) != 3:
                issues.add(f"{dataset.value}:maximum_round_mismatch")
            if int(verification.get("decision_count", -1)) != int(
                verification.get("eligible_candidate_count", -2)
            ):
                issues.add(f"{dataset.value}:decision_accounting_incomplete")
        coverage = coverage_verifications.get(dataset, {})
        if coverage.get("semantic_match_present") is not True:
            issues.add(f"{dataset.value}:semantic_match_missing")
        if coverage.get("semantic_native_output_zero") is not True:
            issues.add(f"{dataset.value}:semantic_structured_output_fallback")
        if int(coverage.get("broad_session_round_match_count", -1)) != 0:
            issues.add(f"{dataset.value}:broad_session_round_match")
        if int(coverage.get("semantic_pair_count", -1)) != int(
            coverage.get("semantic_assessment_count", -2)
        ):
            issues.add(f"{dataset.value}:semantic_pair_accounting_incomplete")
        if int(coverage.get("semantic_model_call_count", -1)) != int(
            coverage.get("semantic_pair_count", -2)
        ):
            issues.add(f"{dataset.value}:semantic_model_lineage_incomplete")
    return sorted(issues), severe_terminal_errors


def _report_markdown(
    *,
    pilot_id: str,
    partition_id: str,
    questioner: Mapping[str, Any],
    planner: Mapping[str, Any],
    dry_run: Mapping[str, Any],
    status: ProgramPilotStatus,
    step6d: bool,
) -> str:
    title = (
        "PMMC Step 6D Verified QBank Pilot"
        if step6d
        else "PMMC Step 6C Program Quality Pilot"
    )
    lines = [
        f"# {title}: {pilot_id}",
        "",
        f"- Partition: `{partition_id}`",
        f"- Status: `{status.value}`",
        "- Paper result: `false`",
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Units | Eligible | Programs | Program ready | Dry-runs | Evidence complete |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted(questioner["datasets"]):
        q = questioner["datasets"][dataset]
        p = planner["datasets"][dataset]
        d = dry_run["datasets"][dataset]
        lines.append(
            f"| {dataset} | {q.get('unit_count')} | {p.get('eligible_candidate_count')} | "
            f"{p.get('program_count')} | {p.get('program_ready_count')} | "
            f"{d.get('assessment_count')} | {d.get('evidence_complete_count')} |"
        )
    lines.extend(
        [
            "",
            "Scientific quality metrics are report-only. Approval requires review of the "
            "private transfer bundle and an explicit research decision.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_refs(root: Path) -> list[ProgramPilotArtifactRef]:
    return [
        ProgramPilotArtifactRef(
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


def _deterministic_tar_zstd(root: Path) -> tuple[Path, Path]:
    transfer = root / "transfer"
    transfer.mkdir(parents=True, exist_ok=True)
    manifest = CompilerProgramPilotManifest.model_validate(
        read_json(root / "manifest.json")
    )
    prefix = (
        "aaai-pmmc-step6d"
        if manifest.protocol_version == "compiler-program-quality-pilot-v3"
        else "aaai-pmmc-step6c"
    )
    destination = transfer / f"{prefix}-{root.name}.tar.zst"
    checksum_path = transfer / f"{prefix}-{root.name}.sha256"
    included = [
        root / "manifest.json",
        *(
            root / item.relative_path
            for item in manifest.artifacts
        ),
    ]
    with tempfile.NamedTemporaryFile(dir=transfer, suffix=".tar", delete=False) as handle:
        tar_path = Path(handle.name)
    try:
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(included, key=lambda item: item.relative_to(root).as_posix()):
                relative = PurePosixPath(root.name) / path.relative_to(root).as_posix()
                info = tarfile.TarInfo(relative.as_posix())
                payload = path.read_bytes()
                info.size = len(payload)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(payload))
        compressor = zstandard.ZstdCompressor(
            level=9,
            threads=0,
            write_checksum=True,
            write_content_size=True,
        )
        destination.write_bytes(compressor.compress(tar_path.read_bytes()))
    finally:
        tar_path.unlink(missing_ok=True)
    digest = file_sha256(destination)
    checksum_path.write_text(f"{digest}  {destination.name}\n", encoding="ascii")
    destination.chmod(0o600)
    checksum_path.chmod(0o600)
    return destination, checksum_path


def build_program_pilot_bundle(
    *,
    partition_id: str,
    samples: Sequence[QuestionerPilotSample],
    runtime_exports: Mapping[DatasetName, Path],
    coverage_exports: Mapping[DatasetName, Path],
    snapshot_dirs: Mapping[DatasetName, Path],
    benchmark_root: Path,
    output_root: Path,
) -> tuple[Path, CompilerProgramPilotManifest]:
    """Build a deterministic, private-reviewable round-0 program pilot."""

    build_started = time.monotonic()
    _program_pilot_event(
        "build_started",
        dataset_count=len(samples),
        projection_version=_MODEL_ARTIFACT_PROJECTION_VERSION,
    )
    sample_datasets = {item.dataset for item in samples}
    if sample_datasets != set(runtime_exports) or sample_datasets != set(coverage_exports):
        raise ValueError("program pilot datasets differ across samples and exports")
    if set(snapshot_dirs) != sample_datasets:
        raise ValueError("program pilot snapshots differ from its samples")
    if any(item.partition_id != partition_id for item in samples):
        raise ValueError("program pilot samples use another partition")

    identities: dict[str, dict[str, Any]] = {}
    verifications: dict[DatasetName, dict[str, Any]] = {}
    coverage_verifications: dict[DatasetName, dict[str, Any]] = {}
    facts_by_dataset: dict[DatasetName, list[dict[str, Any]]] = {}
    coverage_by_dataset: dict[DatasetName, list[dict[str, Any]]] = {}
    coverage_manifest_hashes: dict[str, str] = {}
    for dataset in sorted(sample_datasets, key=lambda item: item.value):
        identity, verification = _runtime_source_identity(runtime_exports[dataset])
        if identity["dataset"] != dataset.value:
            raise ValueError("program runtime export belongs to another dataset")
        coverage_verification = verify_qbank_coverage_export(coverage_exports[dataset])
        coverage_manifest = read_json(coverage_exports[dataset] / "export_manifest.json")
        if str(coverage_manifest.get("compiler_run_id")) != identity["run_id"]:
            raise ValueError("program coverage export belongs to another compiler run")
        identity["coverage_export_id"] = coverage_manifest["export_id"]
        identity["semantic_export_id"] = coverage_manifest["export_id"]
        identity["coverage_manifest_sha256"] = file_sha256(
            coverage_exports[dataset] / "export_manifest.json"
        )
        identity["coverage_verification"] = coverage_verification
        identity["semantic"] = _semantic_source_identity(coverage_exports[dataset])
        identities[dataset.value] = identity
        coverage_manifest_hashes[dataset.value] = str(identity["coverage_manifest_sha256"])
        verifications[dataset] = verification
        coverage_verifications[dataset] = coverage_verification
        facts_by_dataset[dataset] = _program_candidate_facts(
            dataset=dataset,
            runtime_export=runtime_exports[dataset],
            snapshot_dir=snapshot_dirs[dataset],
        )
        coverage_by_dataset[dataset] = _bind_coverage_dataset(
            _coverage_payloads(
                dataset=dataset,
                coverage_export=coverage_exports[dataset],
                runtime_export=runtime_exports[dataset],
                snapshot_dir=snapshot_dirs[dataset],
            ),
            dataset=dataset,
        )
        _annotate_semantic_matches(
            coverage_export=coverage_exports[dataset],
            coverage_cards=coverage_by_dataset[dataset],
        )

    step6d = all(
        value["questioner"].get("questioner_protocol_version")
        == "pmmc-questioner-v3.3"
        for value in identities.values()
    )
    pilot_schema_version = PROGRAM_PILOT_SCHEMA_VERSION if step6d else "0.2.0"
    pilot_protocol = (
        _PROGRAM_PILOT_PROTOCOL
        if step6d
        else "compiler-program-quality-pilot-v2"
    )
    audit_protocol = (
        _PROGRAM_AUDIT_PROTOCOL if step6d else "compiler-program-audit-v2"
    )
    identity_payload = {
        "partition_id": partition_id,
        "sample_ids": [
            item.sample_id for item in sorted(samples, key=lambda item: item.dataset.value)
        ],
        "runtime_manifests": {
            key: value["runtime_manifest_sha256"] for key, value in sorted(identities.items())
        },
        "coverage_manifests": dict(sorted(coverage_manifest_hashes.items())),
        "protocol": pilot_protocol,
        "model_artifact_projection_version": _MODEL_ARTIFACT_PROJECTION_VERSION,
    }
    pilot_id = "compiler-program-pilot-" + stable_hash(identity_payload)[:24]
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = output_root / f".{pilot_id}.staging-{os.getpid()}"
    temporary = staging_root / pilot_id
    if staging_root.exists():
        shutil.rmtree(staging_root)
    (temporary / "public").mkdir(parents=True)
    (temporary / "private" / "referenced_thumbnails").mkdir(parents=True)

    questioner, planner, dry_run, operator_rows, failure_rows = _summaries(
        facts_by_dataset, verifications
    )
    _program_pilot_event(
        "source_facts_ready",
        candidate_fact_count=sum(len(values) for values in facts_by_dataset.values()),
        coverage_query_count=sum(len(values) for values in coverage_by_dataset.values()),
        elapsed_seconds=round(time.monotonic() - build_started, 1),
    )
    integrity_issues, severe_terminal_errors = _automatic_integrity(
        verifications,
        coverage_verifications,
    )
    automatic_integrity_passed = not integrity_issues
    pilot_status = (
        ProgramPilotStatus.AWAITING_APPROVAL
        if automatic_integrity_passed
        else ProgramPilotStatus.NEEDS_REMEDIATION
    )
    automatic = {
        "pilot_id": pilot_id,
        "partition_id": partition_id,
        "status": pilot_status.value,
        "paper_result": False,
        "automatic_integrity_passed": automatic_integrity_passed,
        "integrity_issue_types": integrity_issues,
        "datasets": {
            dataset.value: {
                **questioner["datasets"][dataset.value],
                **planner["datasets"][dataset.value],
                **dry_run["datasets"][dataset.value],
            }
            for dataset in sorted(sample_datasets, key=lambda item: item.value)
        },
    }
    write_json(temporary / "public" / "automatic_summary.json", automatic)
    write_json(temporary / "public" / "questioner_summary.json", questioner)
    write_json(temporary / "public" / "planner_summary.json", planner)
    write_json(temporary / "public" / "dry_run_summary.json", dry_run)
    write_json(
        temporary / "public" / "semantic_summary.json",
        {
            "protocol_version": (
                "qbank-semantic-match-summary-v2"
                if step6d
                else "qbank-semantic-match-summary-v1"
            ),
            "datasets": {
                key: {
                    "semantic_run_id": value["semantic"]["semantic_run_id"],
                    "pair_count": value["coverage_verification"][
                        "semantic_pair_count"
                    ],
                    "semantic_match_count": sum(
                        item.get("semantic_match") is True
                        for item in read_jsonl(
                            coverage_exports[DatasetName(key)]
                            / "private"
                            / "semantic_match_assessments.jsonl"
                        )
                    ),
                    "structured_output_issue_count": value[
                        "coverage_verification"
                    ]["semantic_structured_output_issue_count"],
                }
                for key, value in sorted(identities.items())
            },
        },
    )
    if step6d:
        write_json(
            temporary / "public" / "doubter_round_summary.json",
            _step6d_round_summary(runtime_exports),
        )
    write_jsonl(temporary / "public" / "operator_distribution.jsonl", operator_rows)
    write_jsonl(temporary / "public" / "failure_summary.jsonl", failure_rows)
    write_json(
        temporary / "public" / "performance_summary.json",
        {
            "datasets": {
                key: {
                    "runtime_export_id": value["runtime_export_id"],
                    "program_verification": verifications[DatasetName(key)],
                }
                for key, value in sorted(identities.items())
            }
        },
    )
    (temporary / "public" / "report.md").write_text(
        _report_markdown(
            pilot_id=pilot_id,
            partition_id=partition_id,
            questioner=questioner,
            planner=planner,
            dry_run=dry_run,
            status=pilot_status,
            step6d=step6d,
        ),
        encoding="utf-8",
    )

    all_facts = [
        item
        for dataset in sorted(facts_by_dataset, key=lambda value: value.value)
        for item in facts_by_dataset[dataset]
    ]
    candidate_cards = [
        item
        for dataset in sorted(facts_by_dataset, key=lambda value: value.value)
        for item in _select_candidate_cards(facts_by_dataset[dataset], dataset=dataset)
    ]
    coverage_cards = [
        item
        for dataset in sorted(coverage_by_dataset, key=lambda value: value.value)
        for item in _select_coverage_cards(coverage_by_dataset[dataset], dataset=dataset)
    ]
    fact_by_qbank = {str(item["qbank_id"]): item for item in all_facts}
    for card in coverage_cards:
        for ranked in card.get("ranked_candidates", []):
            fact = fact_by_qbank.get(str(ranked.get("qbank_id")))
            ranked["planner_outcome"] = fact.get("planner_outcome") if fact else None
            ranked["program"] = fact.get("program") if fact else None
            ranked["program_steps"] = fact.get("program_steps", []) if fact else []
            ranked["dry_run_assessment"] = fact.get("dry_run_assessment") if fact else None
    thumbnail_count = _annotate_thumbnails(
        [candidate_cards, coverage_cards],
        benchmark_root=benchmark_root,
        thumbnail_root=temporary / "private" / "referenced_thumbnails",
    )
    _program_pilot_event(
        "audit_cards_ready",
        candidate_card_count=len(candidate_cards),
        coverage_card_count=len(coverage_cards),
        elapsed_seconds=round(time.monotonic() - build_started, 1),
        thumbnail_count=thumbnail_count,
    )
    _write_zstd_jsonl(temporary / "private" / "all_candidate_program_facts.jsonl.zst", all_facts)
    write_jsonl(temporary / "private" / "candidate_program_audit_cards.jsonl", candidate_cards)
    write_jsonl(temporary / "private" / "query_coverage_audit_cards.jsonl", coverage_cards)
    write_json(temporary / "private" / "source_exports.json", identities)
    artifact_selection = (
        _step6d_model_artifact_selection(
            runtime_exports=runtime_exports,
            coverage_exports=coverage_exports,
            candidate_cards=candidate_cards,
            coverage_cards=coverage_cards,
        )
        if step6d
        else None
    )
    artifact_rows = _archive_model_artifacts(
        runtime_exports=runtime_exports,
        coverage_exports=coverage_exports,
        destination_root=temporary / "private" / "model_artifacts",
        selected_calls=artifact_selection,
    )
    write_jsonl(
        temporary / "private" / "model_artifact_manifest.jsonl",
        artifact_rows,
    )

    audit_payload = {
        "pilot_id": pilot_id,
        "protocol_version": audit_protocol,
        "partition_id": partition_id,
        "candidate_program_audit_count": len(candidate_cards),
        "query_coverage_audit_count": len(coverage_cards),
        "candidate_program_audit_sha256": file_sha256(
            temporary / "private" / "candidate_program_audit_cards.jsonl"
        ),
        "query_coverage_audit_sha256": file_sha256(
            temporary / "private" / "query_coverage_audit_cards.jsonl"
        ),
        "all_candidate_program_facts_sha256": file_sha256(
            temporary / "private" / "all_candidate_program_facts.jsonl.zst"
        ),
        "automatic_summary_sha256": file_sha256(temporary / "public" / "automatic_summary.json"),
        "source_runtime_manifest_hashes": {
            key: str(value["runtime_manifest_sha256"]) for key, value in sorted(identities.items())
        },
        "source_coverage_manifest_hashes": dict(sorted(coverage_manifest_hashes.items())),
        "automatic_integrity_passed": automatic_integrity_passed,
        "integrity_issue_types": integrity_issues,
        "severe_terminal_error_count": severe_terminal_errors,
        "metadata": {
            "paper_result": False,
            "thumbnail_count": thumbnail_count,
            "candidate_fact_count": len(all_facts),
            "model_artifact_projection_version": _MODEL_ARTIFACT_PROJECTION_VERSION,
            "model_artifact_selection_policy": (
                "step6d-failures-revisions-audit-samples-v1"
                if step6d
                else "complete-step6c-model-artifacts-v1"
            ),
            "selected_model_call_count": (
                len(artifact_selection)
                if artifact_selection is not None
                else len(
                    {
                        (
                            str(item["dataset"]),
                            str(item["scope"]),
                            str(item["model_call_id"]),
                        )
                        for item in artifact_rows
                    }
                )
            ),
            "source_identities": identities,
        },
    }
    audit = CompilerProgramPilotAuditManifest(
        **audit_payload, content_hash=stable_hash(audit_payload)
    )
    write_json(temporary / "private" / "audit_manifest.json", audit.model_dump(mode="json"))

    for directory in [temporary, *[path for path in temporary.rglob("*") if path.is_dir()]]:
        directory.chmod(0o700)
    for path in temporary.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    artifacts = _artifact_refs(temporary)
    manifest_payload = {
        "pilot_id": pilot_id,
        "schema_version": pilot_schema_version,
        "protocol_version": pilot_protocol,
        "partition_id": partition_id,
        "samples": [
            item.model_dump(mode="json")
            for item in sorted(samples, key=lambda item: item.dataset.value)
        ],
        "status": pilot_status.value,
        "source_runtime_export_ids": {
            key: str(value["runtime_export_id"]) for key, value in sorted(identities.items())
        },
        "source_coverage_export_ids": {
            key: str(value["coverage_export_id"]) for key, value in sorted(identities.items())
        },
        "automatic_integrity_passed": automatic_integrity_passed,
        "artifacts": [item.model_dump(mode="json") for item in artifacts],
        "metadata": {
            "paper_result": False,
            "audit_manifest_hash": audit.content_hash,
            "candidate_program_audit_count": len(candidate_cards),
            "query_coverage_audit_count": len(coverage_cards),
            "thumbnail_count": thumbnail_count,
            "integrity_issue_types": integrity_issues,
            "model_artifact_projection_version": _MODEL_ARTIFACT_PROJECTION_VERSION,
            "model_artifact_selection_policy": (
                "step6d-failures-revisions-audit-samples-v1"
                if step6d
                else "complete-step6c-model-artifacts-v1"
            ),
            "selected_model_call_count": (
                len(artifact_selection)
                if artifact_selection is not None
                else len(
                    {
                        (
                            str(item["dataset"]),
                            str(item["scope"]),
                            str(item["model_call_id"]),
                        )
                        for item in artifact_rows
                    }
                )
            ),
            "active_round": 2 if step6d else 0,
            "maximum_round": 3 if step6d else 0,
        },
    }
    manifest = CompilerProgramPilotManifest(
        **manifest_payload, content_hash=stable_hash(manifest_payload)
    )
    write_json(temporary / "manifest.json", manifest.model_dump(mode="json"))
    (temporary / "manifest.json").chmod(0o600)

    destination = output_root / pilot_id
    _deterministic_tar_zstd(temporary)
    verify_program_pilot_bundle(temporary)
    if destination.exists():
        try:
            existing = verify_program_pilot_bundle(destination)
        except (OSError, ValueError) as exc:
            existing_manifest = destination / "manifest.json"
            quarantine_hash = (
                file_sha256(existing_manifest)[:12]
                if existing_manifest.is_file()
                else stable_hash(
                    {
                        "pilot_id": pilot_id,
                        "error_type": type(exc).__name__,
                    }
                )[:12]
            )
            quarantine = output_root / (
                f".{pilot_id}.invalid-{quarantine_hash}-{os.getpid()}"
            )
            destination.replace(quarantine)
            _program_pilot_event(
                "existing_invalid_bundle_quarantined",
                error_type=type(exc).__name__,
                pilot_id=pilot_id,
                quarantine_name=quarantine.name,
            )
        else:
            if existing.content_hash != manifest.content_hash:
                raise ValueError("program pilot ID conflicts with an existing bundle")
            shutil.rmtree(staging_root)
            _program_pilot_event(
                "build_completed",
                artifact_count=len(existing.artifacts),
                elapsed_seconds=round(time.monotonic() - build_started, 1),
                pilot_id=pilot_id,
                reused=True,
                status=existing.status.value,
            )
            return destination, existing
    temporary.replace(destination)
    staging_root.rmdir()
    _program_pilot_event(
        "build_completed",
        artifact_count=len(manifest.artifacts),
        elapsed_seconds=round(time.monotonic() - build_started, 1),
        pilot_id=pilot_id,
        reused=False,
        status=pilot_status.value,
    )
    return destination, manifest


def _verify_model_artifact_lineage_manifest(
    *,
    rows: Sequence[dict[str, Any]],
    artifacts: Sequence[ProgramPilotArtifactRef],
    projection_required: bool,
) -> dict[str, set[str]]:
    started = time.monotonic()
    artifact_by_path = {item.relative_path: item for item in artifacts}
    if len(artifact_by_path) != len(artifacts):
        raise ValueError("program pilot artifact paths are not unique")
    _program_pilot_event(
        "model_artifact_lineage_audit_started",
        reference_count=len(rows),
    )
    reference_keys: set[tuple[str, str, str, str]] = set()
    artifact_reference_keys: set[tuple[str, str, str, str]] = set()
    artifact_roles: dict[str, set[str]] = defaultdict(set)

    def fail(
        *,
        ordinal: int,
        row: Mapping[str, Any],
        rule: str,
    ) -> NoReturn:
        dataset = str(row.get("dataset") or "")
        scope = str(row.get("scope") or "")
        locator = _model_artifact_reference_locator(
            dataset=dataset,
            scope=scope,
            artifact_id=row.get("artifact_id"),
            model_call_id=row.get("model_call_id"),
            role=row.get("role"),
        )
        _program_pilot_event(
            "model_artifact_lineage_audit_failed",
            error_type="ModelArtifactLineageError",
            reference_locator=locator,
            reference_ordinal=ordinal,
            rule=rule,
            total_reference_count=len(rows),
        )
        raise ValueError(
            "program pilot model artifact lineage is invalid "
            f"(rule={rule}, reference={locator})"
        )

    for ordinal, row in enumerate(rows, start=1):
        dataset = str(row.get("dataset") or "")
        scope = str(row.get("scope") or "")
        artifact_id = str(row.get("artifact_id") or "")
        model_call_id = str(row.get("model_call_id") or "")
        role = str(row.get("role") or "")
        relative = str(row.get("relative_path") or "")
        if dataset not in {
            DatasetName.MEMLENS.value,
            DatasetName.MEMGALLERY.value,
        }:
            fail(ordinal=ordinal, row=row, rule="dataset_invalid")
        if scope not in {"runtime", "semantic"}:
            fail(ordinal=ordinal, row=row, rule="scope_invalid")
        if not artifact_id:
            fail(ordinal=ordinal, row=row, rule="artifact_id_missing")
        if not model_call_id:
            fail(ordinal=ordinal, row=row, rule="model_call_id_missing")
        if role not in {"input", "output"}:
            fail(ordinal=ordinal, row=row, rule="artifact_role_invalid")
        if scope == "semantic" and not str(row.get("semantic_job_id") or ""):
            fail(ordinal=ordinal, row=row, rule="semantic_job_id_missing")
        if not relative.startswith("private/model_artifacts/sha256/"):
            fail(ordinal=ordinal, row=row, rule="relative_path_invalid")
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            fail(ordinal=ordinal, row=row, rule="relative_path_unsafe")
        artifact = artifact_by_path.get(relative)
        if artifact is None:
            fail(ordinal=ordinal, row=row, rule="payload_not_in_manifest")
        if row.get("sha256") != artifact.sha256:
            fail(ordinal=ordinal, row=row, rule="payload_checksum_lineage_mismatch")
        try:
            size_bytes = int(row.get("size_bytes", -1))
        except (TypeError, ValueError):
            fail(ordinal=ordinal, row=row, rule="payload_size_invalid")
        if size_bytes != artifact.size_bytes:
            fail(ordinal=ordinal, row=row, rule="payload_size_lineage_mismatch")

        reference_key = (dataset, scope, model_call_id, role)
        if reference_key in reference_keys:
            fail(ordinal=ordinal, row=row, rule="duplicate_model_call_role")
        artifact_reference_key = (dataset, scope, artifact_id, role)
        if artifact_reference_key in artifact_reference_keys:
            fail(ordinal=ordinal, row=row, rule="duplicate_artifact_role")
        reference_keys.add(reference_key)
        artifact_reference_keys.add(artifact_reference_key)

        if projection_required:
            try:
                source_size_bytes = int(row.get("source_size_bytes", -1))
                redaction_count = row.get("projection_redaction_count")
                projection_applied = row.get("projection_applied")
            except (TypeError, ValueError):
                fail(ordinal=ordinal, row=row, rule="projection_lineage_invalid")
            if (
                row.get("projection_version") != _MODEL_ARTIFACT_PROJECTION_VERSION
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(row.get("source_sha256") or "")
                )
                or source_size_bytes < 0
                or not isinstance(projection_applied, bool)
                or not isinstance(redaction_count, int)
                or redaction_count < 0
                or projection_applied is not (redaction_count > 0)
            ):
                fail(ordinal=ordinal, row=row, rule="projection_lineage_invalid")
        artifact_roles[relative].add(role)
        if ordinal % 5000 == 0:
            _program_pilot_event(
                "model_artifact_lineage_audit_progress",
                checked_reference_count=ordinal,
                elapsed_seconds=round(time.monotonic() - started, 1),
                total_reference_count=len(rows),
            )
    _program_pilot_event(
        "model_artifact_lineage_audit_completed",
        checked_reference_count=len(rows),
        elapsed_seconds=round(time.monotonic() - started, 1),
        unique_model_call_role_count=len(reference_keys),
        unique_payload_count=len(artifact_roles),
    )
    return artifact_roles


def verify_program_pilot_bundle(
    pilot_dir: Path,
    *,
    require_transfer: bool = True,
) -> CompilerProgramPilotManifest:
    """Verify a portable Step 6C quality-pilot bundle and transfer archive."""

    verification_started = time.monotonic()
    manifest = CompilerProgramPilotManifest.model_validate(read_json(pilot_dir / "manifest.json"))
    _program_pilot_event(
        "portability_audit_started",
        artifact_count=len(manifest.artifacts),
        pilot_id=manifest.pilot_id,
    )
    if pilot_dir.name != manifest.pilot_id:
        raise ValueError("program pilot directory does not match pilot_id")
    expected_content_hash = stable_hash(manifest.model_dump(mode="json", exclude={"content_hash"}))
    if expected_content_hash != manifest.content_hash:
        raise ValueError("program pilot manifest content hash mismatch")
    expected_paths = sorted(item.relative_path for item in manifest.artifacts)
    actual_paths = sorted(
        path.relative_to(pilot_dir).as_posix()
        for path in pilot_dir.rglob("*")
        if path.is_file()
        and not path.is_relative_to(pilot_dir / "transfer")
        and path.name != "manifest.json"
    )
    if actual_paths != expected_paths:
        raise ValueError("program pilot artifact inventory differs from manifest")
    artifact_manifest_path = (
        pilot_dir / "private" / "model_artifact_manifest.jsonl"
    )
    if not artifact_manifest_path.is_file():
        raise ValueError("program pilot model artifact manifest is missing")
    model_artifact_rows = read_jsonl(artifact_manifest_path)
    if manifest.protocol_version == "compiler-program-quality-pilot-v3":
        if (
            manifest.metadata.get("model_artifact_selection_policy")
            != "step6d-failures-revisions-audit-samples-v1"
        ):
            raise ValueError("Step 6D model artifact selection policy is missing")
        selected_call_count = len(
            {
                (
                    str(row.get("dataset")),
                    str(row.get("scope")),
                    str(row.get("model_call_id")),
                )
                for row in model_artifact_rows
            }
        )
        if selected_call_count != int(
            manifest.metadata.get("selected_model_call_count", -1)
        ):
            raise ValueError("Step 6D selected model-call accounting differs")
        if any(
            not isinstance(row.get("selection_reasons"), list)
            or not row["selection_reasons"]
            for row in model_artifact_rows
        ):
            raise ValueError("Step 6D model artifacts lack selection reasons")
    projection_required = (
        manifest.metadata.get("model_artifact_projection_version")
        == _MODEL_ARTIFACT_PROJECTION_VERSION
    )
    artifact_roles = _verify_model_artifact_lineage_manifest(
        rows=model_artifact_rows,
        artifacts=manifest.artifacts,
        projection_required=projection_required,
    )
    next_report = verification_started + _PORTABILITY_PROGRESS_INTERVAL_SECONDS
    for ordinal, item in enumerate(manifest.artifacts, start=1):
        path = pilot_dir / item.relative_path
        if file_sha256(path) != item.sha256 or path.stat().st_size != item.size_bytes:
            _program_pilot_event(
                "portability_audit_failed",
                artifact_ordinal=ordinal,
                artifact_relative_path=item.relative_path,
                error_type="ArtifactChecksumMismatch",
                rule="artifact_integrity",
            )
            raise ValueError(f"program pilot artifact checksum mismatch: {item.relative_path}")
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            _program_pilot_event(
                "portability_audit_failed",
                artifact_ordinal=ordinal,
                artifact_relative_path=item.relative_path,
                error_type="ArtifactModeMismatch",
                rule="artifact_mode",
            )
            raise ValueError(f"program pilot artifact must use mode 0600: {item.relative_path}")
        issue = _portable_scan_issue(
            path,
            inspect_structured=artifact_roles.get(item.relative_path) != {"output"},
            strict_text=item.relative_path.startswith("public/")
            and path.suffix.lower() in {".md", ".txt"},
        )
        if issue is not None:
            _program_pilot_event(
                "portability_audit_failed",
                artifact_ordinal=ordinal,
                artifact_relative_path=item.relative_path,
                error_type="PortableArtifactViolation",
                rule=issue,
            )
            if issue == "secret_or_sentinel":
                raise ValueError(
                    "program pilot bundle contains a secret or sentinel "
                    f"(artifact={item.relative_path}, rule={issue})"
                )
            raise ValueError(
                "program pilot bundle contains a non-portable runtime locator "
                f"(artifact={item.relative_path}, rule={issue})"
            )
        roles = artifact_roles.get(item.relative_path, set())
        if "input" in roles and path.suffix.lower() == ".json":
            try:
                request_payload = json.loads(path.read_bytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "program pilot projected model request is not valid JSON"
                ) from exc
            locator_key = _unredacted_model_locator_key(request_payload)
            if locator_key is not None:
                _program_pilot_event(
                    "portability_audit_failed",
                    artifact_relative_path=item.relative_path,
                    error_type="UnredactedModelLocator",
                    rule=f"structured_locator:{locator_key}",
                )
                raise ValueError(
                    "program pilot projected model request contains a runtime locator "
                    f"(artifact={item.relative_path}, "
                    f"rule=structured_locator:{locator_key})"
                )
        if item.relative_path.startswith("public/") and path.suffix in {".json", ".jsonl"}:
            value: object
            if path.suffix == ".jsonl":
                value = read_jsonl(path)
            else:
                value = read_json(path)
            if _walk_keys(value) & _PUBLIC_FORBIDDEN_KEYS:
                raise ValueError("public program pilot artifact exposes private facts")
        now = time.monotonic()
        if now >= next_report:
            _program_pilot_event(
                "portability_audit_progress",
                checked_artifact_count=ordinal,
                elapsed_seconds=round(now - verification_started, 1),
                total_artifact_count=len(manifest.artifacts),
            )
            next_report = now + _PORTABILITY_PROGRESS_INTERVAL_SECONDS

    audit = CompilerProgramPilotAuditManifest.model_validate(
        read_json(pilot_dir / "private" / "audit_manifest.json")
    )
    if audit.pilot_id != manifest.pilot_id or audit.partition_id != manifest.partition_id:
        raise ValueError("program audit manifest belongs to another pilot")
    if stable_hash(audit.model_dump(mode="json", exclude={"content_hash"})) != audit.content_hash:
        raise ValueError("program audit manifest content hash mismatch")
    if manifest.metadata.get("audit_manifest_hash") != audit.content_hash:
        raise ValueError("program pilot does not bind its audit manifest")
    paths = {
        "candidate": pilot_dir / "private" / "candidate_program_audit_cards.jsonl",
        "coverage": pilot_dir / "private" / "query_coverage_audit_cards.jsonl",
        "facts": pilot_dir / "private" / "all_candidate_program_facts.jsonl.zst",
        "automatic": pilot_dir / "public" / "automatic_summary.json",
    }
    expected_hashes = {
        "candidate": audit.candidate_program_audit_sha256,
        "coverage": audit.query_coverage_audit_sha256,
        "facts": audit.all_candidate_program_facts_sha256,
        "automatic": audit.automatic_summary_sha256,
    }
    if any(file_sha256(paths[key]) != value for key, value in expected_hashes.items()):
        raise ValueError("program audit payload hash mismatch")
    automatic = read_json(paths["automatic"])
    if (
        automatic.get("automatic_integrity_passed") is not audit.automatic_integrity_passed
        or manifest.automatic_integrity_passed is not audit.automatic_integrity_passed
    ):
        raise ValueError("program pilot integrity status differs across manifests")
    expected_status = (
        ProgramPilotStatus.AWAITING_APPROVAL
        if audit.automatic_integrity_passed
        else ProgramPilotStatus.NEEDS_REMEDIATION
    )
    if (
        manifest.status is not expected_status
        or automatic.get("status") != expected_status.value
    ):
        raise ValueError("program pilot status disagrees with its integrity gate")
    raw_issue_types = automatic.get("integrity_issue_types")
    if not isinstance(raw_issue_types, list) or sorted(
        str(value) for value in raw_issue_types
    ) != sorted(audit.integrity_issue_types):
        raise ValueError("program pilot integrity issues differ across manifests")
    candidate_cards = read_jsonl(paths["candidate"])
    coverage_cards = read_jsonl(paths["coverage"])
    facts = _read_zstd_jsonl(paths["facts"])
    if len(candidate_cards) != 160 or len(coverage_cards) != 40:
        raise ValueError("program pilot audit-card denominator is incomplete")
    if Counter(str(item.get("dataset")) for item in candidate_cards) != {
        DatasetName.MEMLENS.value: 80,
        DatasetName.MEMGALLERY.value: 80,
    }:
        raise ValueError("program candidate audit cards are not balanced by dataset")
    if Counter(str(item.get("dataset")) for item in coverage_cards) != {
        DatasetName.MEMLENS.value: 20,
        DatasetName.MEMGALLERY.value: 20,
    }:
        raise ValueError("program coverage audit cards are not balanced by dataset")
    for card in coverage_cards:
        dataset = str(card["dataset"])
        if not str(card.get("query_id", "")).startswith(f"{dataset}:query:"):
            raise ValueError("program coverage card references another dataset")
    qbank_by_dataset: dict[str, set[str]] = defaultdict(set)
    for fact in facts:
        qbank_by_dataset[str(fact["dataset"])].add(str(fact["qbank_id"]))
    if qbank_by_dataset[DatasetName.MEMLENS.value] & qbank_by_dataset[DatasetName.MEMGALLERY.value]:
        raise ValueError("program pilot contains cross-dataset QBank IDs")
    for card in candidate_cards:
        qbank_id = str(card["qbank_id"])
        if qbank_id not in qbank_by_dataset[str(card["dataset"])]:
            raise ValueError("program candidate card references another dataset")
    thumbnail_paths = {
        str(ref["thumbnail_relative_path"])
        for ref in _image_refs([candidate_cards, coverage_cards])
        if ref.get("thumbnail_relative_path")
    }
    artifact_paths = set(expected_paths)
    if not thumbnail_paths <= artifact_paths:
        raise ValueError("program audit card references a missing thumbnail")

    source_identities = read_json(pilot_dir / "private" / "source_exports.json")
    if not isinstance(source_identities, dict):
        raise ValueError("program pilot source identity export is invalid")
    expected_semantic_protocol = (
        "qbank-semantic-match-v2"
        if manifest.protocol_version == "compiler-program-quality-pilot-v3"
        else "qbank-semantic-match-v1"
    )
    for dataset, identity in source_identities.items():
        if dataset not in {
            DatasetName.MEMLENS.value,
            DatasetName.MEMGALLERY.value,
        } or not isinstance(identity, dict):
            raise ValueError("program pilot source identity has an invalid dataset")
        semantic = identity.get("semantic")
        if (
            not isinstance(semantic, dict)
            or not semantic.get("semantic_run_id")
            or semantic.get("semantic_protocol_version")
            != expected_semantic_protocol
        ):
            raise ValueError("program pilot lacks a frozen semantic judge identity")
    if manifest.protocol_version == "compiler-program-quality-pilot-v3":
        round_summary = read_json(
            pilot_dir / "public" / "doubter_round_summary.json"
        )
        if (
            round_summary.get("protocol_version")
            != "verified-qbank-round-analysis-v1"
            or round_summary.get("active_round") != 2
            or round_summary.get("maximum_round") != 3
            or set(dict(round_summary.get("datasets", {})))
            != {
                DatasetName.MEMLENS.value,
                DatasetName.MEMGALLERY.value,
            }
        ):
            raise ValueError("Step 6D round summary is incomplete")

    if require_transfer:
        transfer = pilot_dir / "transfer"
        archives = sorted(transfer.glob("*.tar.zst"))
        checksums = sorted(transfer.glob("*.sha256"))
        if len(archives) != 1 or len(checksums) != 1:
            raise ValueError("program pilot transfer archive is incomplete")
        checksum_parts = checksums[0].read_text(encoding="ascii").strip().split()
        if len(checksum_parts) != 2 or checksum_parts[0] != file_sha256(archives[0]):
            raise ValueError("program pilot transfer checksum mismatch")
        if checksum_parts[1] != archives[0].name:
            raise ValueError("program pilot transfer checksum names another archive")
        _verify_transfer_archive(
            pilot_dir=pilot_dir,
            archive_path=archives[0],
            expected_files=[
                pilot_dir / "manifest.json",
                *(pilot_dir / item.relative_path for item in manifest.artifacts),
            ],
        )
    _program_pilot_event(
        "portability_audit_completed",
        checked_artifact_count=len(manifest.artifacts),
        elapsed_seconds=round(time.monotonic() - verification_started, 1),
        pilot_id=manifest.pilot_id,
        transfer_verified=require_transfer,
    )
    return manifest


def verify_program_pilot_archive(
    archive_path: Path,
    checksum_path: Path,
) -> CompilerProgramPilotManifest:
    """Verify a transferred archive without requiring recursive self-inclusion."""

    checksum_parts = checksum_path.read_text(encoding="ascii").strip().split()
    if (
        len(checksum_parts) != 2
        or checksum_parts[1] != archive_path.name
        or checksum_parts[0] != file_sha256(archive_path)
    ):
        raise ValueError("program pilot archive checksum is invalid")
    tar_payload = zstandard.ZstdDecompressor().decompress(archive_path.read_bytes())
    with tempfile.TemporaryDirectory(prefix="pmmc-program-pilot-verify-") as raw:
        destination = Path(raw)
        root_names: set[str] = set()
        with tarfile.open(fileobj=io.BytesIO(tar_payload), mode="r:") as archive:
            for member in archive.getmembers():
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or not member.isfile()
                    or len(name.parts) < 2
                ):
                    raise ValueError("program pilot archive contains an unsafe member")
                root_names.add(name.parts[0])
                target = destination.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("program pilot archive member is unreadable")
                target.write_bytes(source.read())
                target.chmod(0o600)
        if len(root_names) != 1:
            raise ValueError("program pilot archive must contain exactly one root")
        pilot_dir = destination / next(iter(root_names))
        return verify_program_pilot_bundle(pilot_dir, require_transfer=False)


def _verify_questioner_freeze(path: Path) -> QuestionerFreezeManifest:
    freeze = QuestionerFreezeManifest.model_validate(read_json(path))
    payload = freeze.model_dump(
        mode="json", exclude={"freeze_id", "content_hash"}
    )
    if stable_hash(payload) != freeze.content_hash:
        raise ValueError("Questioner freeze content hash mismatch")
    if freeze.freeze_id != "questioner-freeze-" + freeze.content_hash[:24]:
        raise ValueError("Questioner freeze ID does not match its content")
    return freeze


def _publish_program_pilot_decision(
    *,
    decision_root: Path,
    decision_record: CompilerProgramPilotDecision,
    questioner_freeze: QuestionerFreezeManifest | None,
    program_freeze: CompilerProgramFreezeManifest | None,
) -> tuple[Path, Path | None, Path | None]:
    """Publish a complete decision directory with one atomic rename."""

    if (questioner_freeze is None) != (program_freeze is None):
        raise ValueError("approved pilot requires both Questioner and program freezes")
    decision_root.mkdir(parents=True, exist_ok=True)
    decision_root.chmod(0o700)
    decision_dir = decision_root / decision_record.content_hash
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{decision_record.content_hash}.staging-",
            dir=decision_root,
        )
    )
    staging_dir.chmod(0o700)
    try:
        decision_path = staging_dir / "decision.json"
        write_json(decision_path, decision_record.model_dump(mode="json"))
        questioner_path: Path | None = None
        program_path: Path | None = None
        if questioner_freeze is not None and program_freeze is not None:
            questioner_path = staging_dir / "questioner_freeze_manifest.json"
            program_path = staging_dir / "compiler_program_freeze_manifest.json"
            write_json(questioner_path, questioner_freeze.model_dump(mode="json"))
            write_json(program_path, program_freeze.model_dump(mode="json"))
            _verify_questioner_freeze(questioner_path)
            verify_program_freeze(program_path)
        for path in staging_dir.rglob("*"):
            if path.is_file():
                path.chmod(0o600)
        try:
            staging_dir.rename(decision_dir)
        except FileExistsError as error:
            raise ValueError(
                "program pilot decision was published concurrently"
            ) from error
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return (
        decision_dir,
        decision_dir / questioner_path.name if questioner_path is not None else None,
        decision_dir / program_path.name if program_path is not None else None,
    )


def decide_program_pilot(
    *,
    pilot_dir: Path,
    decision: PilotDecisionValue,
    reviewer_id: str,
    rationale: str,
) -> tuple[Path, Path | None, Path | None]:
    """Record an immutable program-pilot decision and emit both freezes."""

    pilot = verify_program_pilot_bundle(pilot_dir)
    audit = CompilerProgramPilotAuditManifest.model_validate(
        read_json(pilot_dir / "private" / "audit_manifest.json")
    )
    if decision is PilotDecisionValue.APPROVE and (
        not audit.automatic_integrity_passed or audit.severe_terminal_error_count != 0
    ):
        raise ValueError("program pilot automatic integrity gate blocks approval")
    rationale = rationale.strip()
    reviewer_id = reviewer_id.strip()
    if not reviewer_id or not rationale:
        raise ValueError("program pilot decision requires reviewer and rationale")
    decision_root = pilot_dir.parent / f"{pilot.pilot_id}-decisions"
    existing = sorted(
        path
        for path in decision_root.glob("*")
        if path.is_dir() and re.fullmatch(r"[0-9a-f]{64}", path.name) is not None
    )
    if existing:
        if len(existing) != 1:
            raise ValueError("program pilot has multiple immutable decisions")
        decision_dir = existing[0]
        stored = CompilerProgramPilotDecision.model_validate(
            read_json(decision_dir / "decision.json")
        )
        stored_payload = stored.model_dump(mode="json", exclude={"content_hash"})
        if stable_hash(stored_payload) != stored.content_hash:
            raise ValueError("persisted program pilot decision hash is invalid")
        if (
            stored.pilot_id != pilot.pilot_id
            or stored.decision is not decision
            or stored.reviewer_id != reviewer_id
            or stored.rationale != rationale
            or stored.audit_manifest_hash != audit.content_hash
        ):
            raise ValueError("program pilot already has a conflicting immutable decision")
        q_path = decision_dir / "questioner_freeze_manifest.json"
        p_path = decision_dir / "compiler_program_freeze_manifest.json"
        if stored.decision is PilotDecisionValue.APPROVE:
            if not q_path.is_file() or not p_path.is_file():
                raise ValueError(
                    "approved program pilot has an incomplete immutable decision"
                )
            _verify_questioner_freeze(q_path)
            verify_program_freeze(p_path)
        elif q_path.exists() or p_path.exists():
            raise ValueError("rejected program pilot unexpectedly contains freezes")
        return (
            decision_dir,
            q_path if q_path.exists() else None,
            p_path if p_path.exists() else None,
        )

    decided_at = utc_now()
    decision_payload = {
        "pilot_id": pilot.pilot_id,
        "decision": decision.value,
        "reviewer_id": reviewer_id,
        "rationale": rationale,
        "audit_manifest_hash": audit.content_hash,
        "decided_at": decided_at,
    }
    decision_record = CompilerProgramPilotDecision(
        **decision_payload,
        content_hash=stable_hash(
            {
                **decision_payload,
                "decided_at": decided_at.isoformat().replace("+00:00", "Z"),
            }
        ),
    )
    if decision is PilotDecisionValue.REJECT:
        return _publish_program_pilot_decision(
            decision_root=decision_root,
            decision_record=decision_record,
            questioner_freeze=None,
            program_freeze=None,
        )

    source = read_json(pilot_dir / "private" / "source_exports.json")
    identities = list(source.values())
    if not identities or not all(isinstance(item, dict) for item in identities):
        raise ValueError("program pilot has no frozen source identities")
    first = identities[0]
    questioner = first["questioner"]
    program = first["program"]
    semantic = first["semantic"]
    semantic_protocol_identity = {
        key: value
        for key, value in semantic.items()
        if key not in {"semantic_run_id", "semantic_pair_count"}
    }
    def protocol_program_identity(value: Mapping[str, Any]) -> dict[str, Any]:
        ignored = (
            {"routing_projection_hash"}
            if pilot.protocol_version == "compiler-program-quality-pilot-v3"
            else set()
        )
        return {key: item for key, item in value.items() if key not in ignored}

    for identity in identities[1:]:
        if (
            identity["questioner"] != questioner
            or protocol_program_identity(identity["program"])
            != protocol_program_identity(program)
        ):
            raise ValueError("program pilot datasets use different Q/P protocol identities")
        if {
            key: value
            for key, value in identity["semantic"].items()
            if key not in {"semantic_run_id", "semantic_pair_count"}
        } != semantic_protocol_identity:
            raise ValueError("program pilot datasets use different semantic judge identities")
    model_commits: dict[str, str] = {}
    for identity in identities:
        for alias, commit in dict(identity["model_commits"]).items():
            prior = model_commits.setdefault(str(alias), str(commit))
            if prior != commit:
                raise ValueError("program pilot model commits differ across datasets")
    semantic_alias = str(semantic["semantic_judge_model_alias"])
    semantic_revision = str(semantic["semantic_judge_model_revision"])
    prior_semantic_revision = model_commits.setdefault(
        semantic_alias, semantic_revision
    )
    if prior_semantic_revision != semantic_revision:
        raise ValueError("semantic judge revision conflicts with the model lock")
    questioner_payload = {
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
        "family_dedupe_version": questioner.get("family_dedupe_version"),
        "matcher_version": questioner["matcher_version"],
        "model_commits": model_commits,
        "runtime_export_ids": pilot.source_runtime_export_ids,
        "coverage_export_ids": pilot.source_coverage_export_ids,
        "questioner_protocol_version": questioner["questioner_protocol_version"],
        "prompt_version": questioner["prompt_version"],
        "audit_manifest_hash": audit.content_hash,
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
    questioner_hash = stable_hash(
        {
            **questioner_payload,
            "approved_at": decided_at.isoformat().replace("+00:00", "Z"),
        }
    )
    questioner_freeze = QuestionerFreezeManifest(
        freeze_id="questioner-freeze-" + questioner_hash[:24],
        **questioner_payload,
        content_hash=questioner_hash,
    )

    program_payload = {
        "pilot_id": pilot.pilot_id,
        "decision_hash": decision_record.content_hash,
        "program_pilot_audit_hash": audit.content_hash,
        "questioner_freeze_id": questioner_freeze.freeze_id,
        "questioner_freeze_hash": questioner_freeze.content_hash,
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
        "memory_index_representation_version": program["memory_index_representation_version"],
        "model_commits": model_commits,
        "runtime_export_ids": pilot.source_runtime_export_ids,
        "r_max": (
            2
            if pilot.protocol_version == "compiler-program-quality-pilot-v3"
            else 0
        ),
        "evidence_lineage_version": "evidence-lineage-v2",
        "qbank_semantic_schema_version": semantic[
            "semantic_schema_version"
        ],
        "semantic_judge_config_hash": semantic["semantic_config_hash"],
        "semantic_judge_prompt_hash": semantic["semantic_prompt_hash"],
        "semantic_judge_schema_hash": semantic[
            "semantic_response_schema_hash"
        ],
        "semantic_export_ids": {
            key: str(value["semantic_export_id"])
            for key, value in sorted(source.items())
        },
        "approved_by": reviewer_id,
        "approved_at": decided_at,
        "metadata": {
            "paper_result": False,
            "questioner_protocol": questioner["questioner_protocol_version"],
            "semantic_protocol": semantic["semantic_protocol_version"],
        },
    }
    program_hash = stable_hash(
        {
            **program_payload,
            "approved_at": decided_at.isoformat().replace("+00:00", "Z"),
        }
    )
    program_freeze = CompilerProgramFreezeManifest(
        freeze_id="compiler-program-freeze-" + program_hash[:24],
        **program_payload,
        content_hash=program_hash,
    )
    return _publish_program_pilot_decision(
        decision_root=decision_root,
        decision_record=decision_record,
        questioner_freeze=questioner_freeze,
        program_freeze=program_freeze,
    )


def decide_step6d_pilot(
    *,
    pilot_dir: Path,
    decision: PilotDecisionValue,
    reviewer_id: str,
    rationale: str,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    """Decide a Step 6D pilot and emit the active-r2 Verified QBank freeze."""

    reviewer_id = reviewer_id.strip()
    rationale = rationale.strip()
    pilot = verify_program_pilot_bundle(pilot_dir)
    if pilot.protocol_version != "compiler-program-quality-pilot-v3":
        raise ValueError("Step 6D decision requires a v3 verified-QBank pilot")
    decision_dir, questioner_path, program_path = decide_program_pilot(
        pilot_dir=pilot_dir,
        decision=decision,
        reviewer_id=reviewer_id,
        rationale=rationale,
    )
    if decision is PilotDecisionValue.REJECT:
        return decision_dir, questioner_path, program_path, None
    if questioner_path is None or program_path is None:
        raise RuntimeError("approved Step 6D pilot did not produce Q/P freezes")
    questioner_freeze = _verify_questioner_freeze(questioner_path)
    program_freeze = verify_program_freeze(program_path)

    source = read_json(pilot_dir / "private" / "source_exports.json")
    identities = [
        value for _key, value in sorted(source.items()) if isinstance(value, dict)
    ]
    if len(identities) != len(pilot.samples):
        raise ValueError("Step 6D source identities are incomplete")
    program_identity = dict(identities[0]["program"])
    semantic_identity = {
        key: value
        for key, value in dict(identities[0]["semantic"]).items()
        if key not in {"semantic_run_id", "semantic_pair_count"}
    }
    routing_hashes: list[str] = []
    run_ids: dict[str, str] = {}
    for dataset, identity in sorted(source.items()):
        run_ids[str(dataset)] = str(identity["run_id"])
        routing_hash = identity["program"].get("routing_projection_hash")
        if not isinstance(routing_hash, str):
            raise ValueError("Step 6D source identity lacks routing projection hash")
        routing_hashes.append(routing_hash)
    questioner_identity_hash = stable_hash(identities[0]["questioner"])
    planner_identity_hash = stable_hash(
        {
            key: value
            for key, value in program_identity.items()
            if not key.startswith("doubter_")
            and key
            not in {
                "routing_projection_version",
                "routing_projection_hash",
                "active_round",
                "max_round",
            }
        }
    )
    doubter_identity_hash = stable_hash(
        {
            key: value
            for key, value in program_identity.items()
            if key.startswith("doubter_")
            or key in {"routing_projection_version", "active_round", "max_round"}
        }
    )
    decision_record = CompilerProgramPilotDecision.model_validate(
        read_json(decision_dir / "decision.json")
    )
    approved_at = decision_record.decided_at
    payload = {
        "schema_version": VERIFIED_QBANK_SCHEMA_VERSION,
        "run_id": pilot.pilot_id,
        "snapshot_ids": sorted(
            str(value["snapshot_id"]) for value in identities
        ),
        "active_round": 2,
        "max_round": 3,
        "questioner_identity_hash": questioner_identity_hash,
        "planner_identity_hash": planner_identity_hash,
        "doubter_identity_hash": doubter_identity_hash,
        "operator_registry_hash": str(program_identity["operator_registry_hash"]),
        "routing_projection_hash": stable_hash(sorted(routing_hashes)),
        "semantic_judge_identity_hash": stable_hash(semantic_identity),
        "pilot_content_hash": pilot.content_hash,
        "approved_by": reviewer_id,
        "approved_at": approved_at,
        "decision_content_hash": decision_record.content_hash,
        "metadata": {
            "paper_result": False,
            "run_ids": run_ids,
            "questioner_freeze_manifest": questioner_freeze.freeze_id,
            "compiler_program_freeze_manifest": program_freeze.freeze_id,
        },
    }
    content_hash = stable_hash(
        {
            **payload,
            "approved_at": approved_at.isoformat().replace("+00:00", "Z"),
        }
    )
    freeze = VerifiedQBankFreezeManifest(
        freeze_manifest_id="verified-qbank-freeze-" + content_hash[:24],
        **payload,
        content_hash=content_hash,
    )
    freeze_path = decision_dir / "verified_qbank_freeze_manifest.json"
    if freeze_path.exists():
        stored = verify_verified_qbank_freeze(freeze_path)
        if stored != freeze:
            raise ValueError("Verified QBank freeze conflicts with prior approval")
    else:
        temporary_path = decision_dir / (
            f".{freeze_path.name}.staging-{os.getpid()}-{content_hash[:12]}"
        )
        write_json(temporary_path, freeze.model_dump(mode="json"))
        temporary_path.chmod(0o600)
        try:
            verify_verified_qbank_freeze(temporary_path)
            temporary_path.replace(freeze_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return decision_dir, questioner_path, program_path, freeze_path


def verify_verified_qbank_freeze(path: Path) -> VerifiedQBankFreezeManifest:
    freeze = VerifiedQBankFreezeManifest.model_validate(read_json(path))
    payload = freeze.model_dump(
        mode="json", exclude={"freeze_manifest_id", "content_hash"}
    )
    if stable_hash(payload) != freeze.content_hash:
        raise ValueError("Verified QBank freeze content hash mismatch")
    if (
        freeze.freeze_manifest_id
        != "verified-qbank-freeze-" + freeze.content_hash[:24]
    ):
        raise ValueError("Verified QBank freeze ID mismatch")
    return freeze


def verify_program_freeze(path: Path) -> CompilerProgramFreezeManifest:
    freeze = CompilerProgramFreezeManifest.model_validate(read_json(path))
    if stable_hash(freeze.model_dump(mode="json", exclude={"freeze_id", "content_hash"})) != (
        freeze.content_hash
    ):
        raise ValueError("compiler program freeze content hash mismatch")
    if freeze.freeze_id != "compiler-program-freeze-" + freeze.content_hash[:24]:
        raise ValueError("compiler program freeze ID does not match its content")
    return freeze


def compare_program_pilots(left: Path, right: Path) -> dict[str, Any]:
    left_manifest = verify_program_pilot_bundle(left)
    right_manifest = verify_program_pilot_bundle(right)
    left_summary = read_json(left / "public" / "automatic_summary.json")
    right_summary = read_json(right / "public" / "automatic_summary.json")
    return {
        "left": left_manifest.pilot_id,
        "right": right_manifest.pilot_id,
        "same_partition": left_manifest.partition_id == right_manifest.partition_id,
        "datasets": {
            key: {
                "left": dict(left_summary.get("datasets", {})).get(key),
                "right": dict(right_summary.get("datasets", {})).get(key),
            }
            for key in sorted(
                set(dict(left_summary.get("datasets", {})))
                | set(dict(right_summary.get("datasets", {})))
            )
        },
    }


__all__ = [
    "build_program_pilot_bundle",
    "compare_program_pilots",
    "decide_program_pilot",
    "decide_step6d_pilot",
    "verify_program_freeze",
    "verify_program_pilot_archive",
    "verify_program_pilot_bundle",
    "verify_verified_qbank_freeze",
]
