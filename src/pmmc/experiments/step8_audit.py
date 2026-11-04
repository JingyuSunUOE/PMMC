"""Offline campaign staging and deterministic export with explicit audit coverage.

``private/audit-inventory.json`` is an explicit evaluator-produced snapshot, not
a recursive search for things that happen to be referenced by successful jobs.
Its schema is ``CampaignAuditInventory`` below. ``expected_records`` MUST be the
authoritative IDs across ALL attempts, including failed/retried calls. ``coverage``
attests which record families were inventoried, including empty families. The
snapshot hash is a provenance attestation, not cryptographic proof of DB truth.

Reranker records require artifacts roles ``lineage``, ``input`` and ``output`` or
``failure``; failure/review_sample records require ``raw``. Attempt records are
generated from every campaign attempt directory and must appear in expected IDs.
All referenced artifacts are actually verified in CAS and copied into immutable
staging. Receipt-only exports always remain audit-incomplete.
Conformance call records bind ``conformance_run_id`` instead of a worker attempt.
They retain research cost/payload evidence but create no scientific assignments.

There is deliberately no database scan here. An upstream DB inventory producer
must read gold with DatabaseRole.EVALUATOR, never runtime/admin credentials, and
write separate runtime/evaluation entries. ``materialize_inventory`` fetches only
explicit private object keys using evaluator-scoped stores; it invokes no models.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_analysis import (
    _analyze_facts,
    load_campaign_facts,
    read_campaign_file,
)
from pmmc.experiments.step8_artifacts import (
    CampaignArtifactStore,
    durable_write,
    json_bytes,
    sha256_file,
)
from pmmc.experiments.step8_export import (
    AuditInventory,
    AuditRecord,
    ExportArtifact,
    ExportPlan,
    LogCutoff,
    PublicReview,
    export_bundle,
)
from pmmc.experiments.step8_jobs import SOURCE_OWNERS

_SHA = r"^[a-f0-9]{64}$"
_COVERAGE = frozenset(
    {
        "reranker_calls",
        "failures",
        "review_samples",
        "attempts",
        "runtime_answers",
        "compiler_patches",
        "variant_cell_reuse",
        "cost_facts",
        "performance",
        "gold",
        "process_facts",
    }
)
_KINDS = Literal[
    "raw_memory",
    "raw_query",
    "gold",
    "image",
    "model_payload",
    "reranker",
    "runtime_answer",
    "compiler_patch",
    "variant_cell_reuse",
    "cost_fact",
    "performance",
    "failure",
    "review_payload",
]

# Versioned authoritative catalog. An index cannot claim complete by omitting a table.
PROCESS_REQUIRED_TABLES = frozenset(
    "pmmc_runtime." + name
    for name in (
        "answer_reuse_links",
        "assets",
        "benchmark_queries",
        "benchmark_sessions",
        "candidate_stage_jobs",
        "compiler_runs",
        "compiler_stage_jobs",
        "compiler_unit_links",
        "compiler_unit_memories",
        "compiler_unit_stage_results",
        "compiler_units",
        "context_assemblies",
        "context_assembly_images",
        "context_assembly_memory_units",
        "context_assembly_retrieval_candidates",
        "dataset_snapshots",
        "doubter_assessment_dimensions",
        "doubter_assessments",
        "doubter_directives",
        "doubter_feedback",
        "embedding_collections",
        "embeddings",
        "images",
        "memory_programs",
        "memory_unit_images",
        "memory_units",
        "memory_window_units",
        "memory_windows",
        "model_calls",
        "planner_batch_programs",
        "planner_candidate_drafts",
        "planner_candidate_outcomes",
        "planner_generation_batches",
        "planner_requirement_contracts",
        "planner_revision_lineage",
        "planner_revision_patches",
        "pmmc_fallback_query_links",
        "pmmc_fallback_run_links",
        "program_dry_run_assessments",
        "program_dry_run_evidence_matches",
        "program_execution_evidence_refs",
        "program_execution_steps",
        "program_executions",
        "program_expected_evidence_refs",
        "program_steps",
        "program_validation_issues",
        "program_validation_results",
        "qbank_candidate_families",
        "qbank_candidate_family_members",
        "qbank_decisions",
        "qbank_duplicate_links",
        "qbank_evidence_refs",
        "qbank_index_manifests",
        "qbank_match_batches",
        "qbank_match_channel_candidates",
        "qbank_match_fused_candidates",
        "qbank_paraphrases",
        "qbank_questions",
        "qbank_reranker_assessments",
        "qbank_round_projection_manifests",
        "qbank_round_projection_members",
        "questioner_answer_claims",
        "questioner_candidate_gate_results",
        "questioner_candidate_supersessions",
        "questioner_candidate_verifications",
        "questioner_generations",
        "questioner_opportunities",
        "questioner_opportunity_evidence_refs",
        "questioner_opportunity_sets",
        "questioner_verification_batches",
        "query_answer_selections",
        "query_embedding_batches",
        "query_embeddings",
        "query_evidence_relevance_assessments",
        "query_evidence_relevance_items",
        "query_evidence_sufficiency_assessments",
        "query_execution_reuse_links",
        "query_executor_runs",
        "query_experiment_cell_results",
        "query_fallback_decisions",
        "query_images",
        "query_program_bindings",
        "query_route_decisions",
        "query_runtime_typed_answers",
        "r0_input_manifests",
        "retrieval_batch_query_embeddings",
        "retrieval_batches",
        "retrieval_corpora",
        "retrieval_document_images",
        "retrieval_documents",
        "retrieval_trace_components",
        "retrieval_traces",
        "routing_projection_manifests",
        "routing_projection_members",
        "run_manifests",
        "run_queries",
        "run_snapshot_links",
        "structured_output_item_diagnostics",
    )
)


def audit_table_kind(name: str) -> _KINDS:
    """Keep DB table artifact classification stable across producer and consumers."""
    if name == "pmmc_runtime.structured_output_item_diagnostics":
        return "compiler_patch"
    if any(
        word in name
        for word in (
            "patch",
            "planner",
            "doubter",
            "memory_program",
            "program_steps",
            "qbank_questions",
            "qbank_candidate_",
            "qbank_duplicate_",
            "qbank_index_",
            "questioner_",
            "compiler_unit_links",
        )
    ):
        return "compiler_patch"
    if any(word in name for word in ("typed_answers", "judge_scores")):
        return "runtime_answer"
    if any(
        word in name
        for word in ("cell", "reuse", "route_decision", "answer_selection", "fallback_decision")
    ):
        return "variant_cell_reuse"
    if any(
        word in name
        for word in (
            "model_calls",
            "budget",
            "spend",
            "provider_rate",
            "provider_reconciliation",
        )
    ):
        return "cost_fact"
    return "gold" if ".query_gold" in name else "performance"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PrivateAuditArtifact(_StrictModel):
    artifact_id: str = Field(min_length=1)
    kind: _KINDS
    scope: Literal["runtime", "evaluation"]
    sha256: str = Field(pattern=_SHA)
    size_bytes: int = Field(ge=0)
    object_key: str | None = None

    @model_validator(mode="after")
    def separation(self):
        if self.kind == "gold" and self.scope != "evaluation":
            raise ValueError("gold must remain evaluator-only")
        if self.object_key is not None:
            path = Path(self.object_key)
            if (
                path.is_absolute()
                or path.as_posix() != self.object_key
                or ".." in path.parts
                or "\\" in self.object_key
            ):
                raise ValueError("unsafe inventory object key")
        return self


class PrivateAuditRecord(_StrictModel):
    kind: Literal["reranker_call", "failure", "review_sample"]
    record_id: str = Field(min_length=1)
    attempt_id: str | None = Field(default=None, min_length=1)
    conformance_run_id: str | None = Field(default=None, min_length=1)
    artifacts: dict[str, str]

    @model_validator(mode="after")
    def payload_roles(self):
        if (self.attempt_id is None) == (self.conformance_run_id is None):
            raise ValueError("audit record needs exactly one worker attempt or conformance run")
        roles = set(self.artifacts)
        if self.kind == "reranker_call":
            if not {"lineage", "input"} <= roles or not roles & {"output", "failure"}:
                raise ValueError("reranker audit requires actual lineage/input/output-or-failure")
        elif "raw" not in roles:
            raise ValueError("failure/review audit requires actual raw payload")
        return self


class CampaignAuditInventory(_StrictModel):
    schema_version: Literal["step8-campaign-audit-inventory-v1"] = (
        "step8-campaign-audit-inventory-v1"
    )
    campaign_id: str = Field(min_length=1)
    selection_hash: str = Field(pattern=_SHA)
    source_role: Literal["evaluator"]
    source_snapshot_sha256: str = Field(pattern=_SHA)
    scope: Literal["all_campaign_attempts"]
    complete: bool = False
    conformance_run_ids: tuple[str, ...] = ()
    coverage: tuple[str, ...] = ()
    expected_records: tuple[
        tuple[Literal["reranker_call", "failure", "review_sample", "attempt"], str], ...
    ] = ()
    artifacts: tuple[PrivateAuditArtifact, ...] = ()
    records: tuple[PrivateAuditRecord, ...] = ()

    @model_validator(mode="after")
    def inventory_contract(self):
        if len(set(self.coverage)) != len(self.coverage) or set(self.coverage) - _COVERAGE:
            raise ValueError("invalid audit coverage declaration")
        ids = {a.artifact_id for a in self.artifacts}
        keys = {(r.kind, r.record_id) for r in self.records}
        if len(ids) != len(self.artifacts) or len(keys) != len(self.records):
            raise ValueError("duplicate audit identities")
        if len(set(self.expected_records)) != len(self.expected_records):
            raise ValueError("duplicate expected audit identity")
        if not keys <= set(self.expected_records):
            raise ValueError("audit records outside authoritative inventory")
        if any(ref not in ids for r in self.records for ref in r.artifacts.values()):
            raise ValueError("audit payload absent from artifact inventory")
        expected_attempts = {key for kind, key in self.expected_records if kind == "attempt"}
        if (
            len(set(self.conformance_run_ids)) != len(self.conformance_run_ids)
            or not all(self.conformance_run_ids)
            or set(self.conformance_run_ids) & expected_attempts
        ):
            raise ValueError("invalid conformance run inventory")
        if any(
            r.attempt_id is not None and r.attempt_id not in expected_attempts for r in self.records
        ):
            raise ValueError("audit record has untracked attempt")
        if any(
            r.conformance_run_id is not None
            and r.conformance_run_id not in self.conformance_run_ids
            for r in self.records
        ):
            raise ValueError("audit record has untracked conformance run")
        by_id = {a.artifact_id: a for a in self.artifacts}
        for record in self.records:
            required = {
                "reranker_call": ("lineage", "reranker"),
                "failure": ("raw", "failure"),
                "review_sample": ("raw", "review_payload"),
            }
            role, kind = required[record.kind]
            if by_id[record.artifacts[role]].kind != kind:
                raise ValueError("audit artifact kind/role mismatch")
        return self


def load_process_facts(
    inventory: CampaignAuditInventory,
    artifact_store: CampaignArtifactStore,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Read checksum-verified private process records, never query IDs for a public report.

    Returns the versioned producer index with ``rows`` added to each table entry.
    Partial inventories require an explicit opt-in; unavailable tables have rows=None.
    Asset rows describe lineage, not verified image binaries or an entire retrieval corpus.
    """
    artifacts = {a.artifact_id: a for a in inventory.artifacts}
    item = artifacts.get("process-facts:index")
    if item is None:
        raise ValueError("process facts index absent")
    if (item.kind, item.scope) != ("performance", "runtime"):
        raise ValueError("process index kind or scope mismatch")
    provenance_item = next(
        (a for a in inventory.artifacts if a.sha256 == inventory.source_snapshot_sha256), None
    )
    if provenance_item is None or (provenance_item.kind, provenance_item.scope) != (
        "performance",
        "evaluation",
    ):
        raise ValueError("process source snapshot not bound to evaluator inventory")
    provenance = artifact_store.load_json(inventory.source_snapshot_sha256)
    index = artifact_store.load_json(item.sha256)
    if (
        index.get("schema_version") != "step8-process-facts-v1"
        or index.get("campaign_id") != inventory.campaign_id
        or index.get("source_role") != "evaluator"
        or provenance.get("schema_version") != "step8-database-audit-snapshot-v1"
        or provenance.get("campaign_id") != inventory.campaign_id
        or provenance.get("source_role") != "evaluator"
        or not index.get("database_snapshot")
        or index["database_snapshot"] != provenance.get("database_snapshot")
        or index["database_snapshot"].get("database_role") != "pmmc_evaluator"
        or set(index.get("required_tables", ())) != PROCESS_REQUIRED_TABLES
        or set(index.get("tables", {})) != PROCESS_REQUIRED_TABLES
    ):
        raise ValueError("process facts index identity mismatch")
    if require_complete and (
        index.get("status") != "complete"
        or index.get("missing")
        or not index.get("tables")
        or "process_facts" not in inventory.coverage
    ):
        raise ValueError("process facts coverage incomplete")
    for name, table in index["tables"].items():
        if table["status"] == "unavailable":
            if require_complete:
                raise ValueError("process table unavailable")
            table["rows"] = None
            continue
        if table["status"] != "available":
            raise ValueError("invalid process table status")
        ref = artifacts.get(table["artifact_id"])
        if (
            ref is None
            or ref.sha256 != table["sha256"]
            or ref.artifact_id != "table:" + name
            or ref.kind != audit_table_kind(name)
            or ref.scope != "runtime"
        ):
            raise ValueError("process table not bound to inventory")
        payload = artifact_store.load_json(ref.sha256)
        if (
            payload.get("table") != name
            or not isinstance(payload.get("rows"), list)
            or not all(isinstance(row, dict) for row in payload["rows"])
            or type(table["row_count"]) is not int
            or len(payload["rows"]) != table["row_count"]
            or provenance.get("table_counts", {}).get(name) != table["row_count"]
            or provenance.get("table_hashes", {}).get(name)
            != canonical_content_hash(payload["rows"])
        ):
            raise ValueError("process table identity or denominator mismatch")
        table["rows"] = payload["rows"]
    return index


async def materialize_inventory(
    inventory: CampaignAuditInventory,
    *,
    settings: Any,
    artifact_store: CampaignArtifactStore,
    private_authorized: bool = False,
) -> dict[str, Any]:
    """Copy explicit objects to CAS, with strict evaluator credential separation.

    Optional network I/O to private storage only. Offline export never calls this
    implicitly. Missing/corrupt objects raise without falling back to other roles.
    """
    if private_authorized is not True:
        raise ValueError("private materialization requires explicit authorization")
    from pmmc.infrastructure.object_store import create_evaluator_object_stores

    runtime, evaluation = create_evaluator_object_stores(settings)
    stores = {"runtime": runtime, "evaluation": evaluation}
    for item in inventory.artifacts:
        try:
            path = artifact_store.verify(item.sha256)
        except (ValueError, OSError):
            if item.object_key is None:
                raise ValueError(
                    "inventory object not materialized and has no private object key"
                ) from None
            payload = await stores[item.scope].get_bytes(item.object_key)
            if (
                len(payload) != item.size_bytes
                or hashlib.sha256(payload).hexdigest() != item.sha256
            ):
                raise ValueError("private inventory object checksum/size mismatch") from None
            artifact_store.put(payload)
        else:
            if path.stat().st_size != item.size_bytes:
                raise ValueError("private inventory object size mismatch")
    return {
        "status": "materialized",
        "artifact_count": len(inventory.artifacts),
        "database_inventory_verified": False,
    }


_EVENTS = {
    "STEP8_ATTEMPT",
    "STEP8_ATTEMPT_COMMIT_FAILURE",
    "STEP8_FAILURE_CAPSULE",
    "STEP8_CAPSULE_FAILURE",
    "STEP8_OBSERVE_FAILURE",
    "STEP8_WORKERS",
    "STEP8_COMPONENT",
    "STEP8_FAILURE",
    "STEP8_PROGRESS",
    "STEP8_CHECKPOINT_FAILURE",
    "STEP8_SCHEDULER",
}
_NUMERIC = {
    "elapsed_seconds",
    "artifact_count",
    "completed",
    "pending",
    "inflight",
    "failed",
    "unknown",
    "reused",
    "denominator",
    "eta_seconds",
    "resumed",
    "database_observation_seconds",
    "component_workers",
    "model_request_limit",
    "database_work_limit",
    "db_inflight",
    "db_peak",
    "db_wait_seconds",
}
_RESOURCE_NUMERIC = {
    "sample_unix_seconds",
    "interval_seconds",
    "cpu_cores_used",
    "cpu_quota_cores",
    "cpu_quota_utilization",
    "cpu_throttled_usec",
    "memory_current_bytes",
    "memory_limit_bytes",
    "gpu_count",
    "gpu_utilization_mean",
    "gpu_memory_bytes",
}
_DIAGNOSTIC_IDENTIFIERS = {
    "error_type",
    "failure_stage",
    "persistence_operation",
    "error_code",
    "provider_error_code",
    "os_error_name",
    "database_error_type",
    "database_guard_code",
    "database_function",
    "schema_name",
    "table_name",
    "column_name",
    "constraint_name",
    "timeout_source",
}
_DIAGNOSTIC_NUMERIC = {
    "provider_status_code",
    "os_errno",
    "database_function_line",
    "persistence_candidate_count",
    "persistence_revised_candidate_count",
    "persistence_verification_batch_count",
    "persistence_supersession_count",
    "persistence_verification_input_count",
    "persistence_verification_pass_count",
    "persistence_verification_revise_count",
    "persistence_verification_reject_count",
    "persistence_verification_provider_fallback",
    "persistence_verification_single_candidate_scope",
}


def _safe_diagnostics(value: dict[str, Any]) -> dict[str, str]:
    result = {}
    for key, item in value.items():
        if not isinstance(item, str):
            continue
        if key in _DIAGNOSTIC_IDENTIFIERS and re.fullmatch(r"[A-Za-z0-9_]{1,128}", item):
            result[key] = item
        elif key in _DIAGNOSTIC_NUMERIC and re.fullmatch(r"[0-9]{1,128}", item):
            result[key] = item
        elif key == "sqlstate" and re.fullmatch(r"[A-Z0-9]{5}", item):
            result[key] = item
    return result


def _safe_resources(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in _RESOURCE_NUMERIC and (
            item is None or (type(item) in (int, float) and math.isfinite(item) and item >= 0)
        ):
            result[key] = item
        elif key in {"sampling", "mode"} and item == "cgroup_v2_interval_nvml_instantaneous":
            result[key] = item
        elif key == "gpu_devices" and isinstance(item, list):
            devices = []
            for device in item:
                if not isinstance(device, dict):
                    continue
                identity = device.get("device_hash")
                utilization, memory = device.get("utilization"), device.get("memory_bytes")
                if (
                    isinstance(identity, str)
                    and re.fullmatch(r"[a-f0-9]{16}", identity)
                    and isinstance(utilization, (int, float))
                    and not isinstance(utilization, bool)
                    and math.isfinite(utilization)
                    and 0 <= utilization <= 100
                    and isinstance(memory, (int, float))
                    and not isinstance(memory, bool)
                    and math.isfinite(memory)
                    and memory >= 0
                ):
                    devices.append(
                        {
                            "device_hash": identity,
                            "utilization": utilization,
                            "memory_bytes": memory,
                        }
                    )
            result[key] = devices
    return result


def _safe_log(payload: bytes) -> tuple[bytes, dict[str, int]]:
    """Project known structured events. Never copy raw provider errors or bodies."""
    events: list[bytes] = []
    rejected = 0
    lines = payload.splitlines(keepends=True)
    for line in lines:
        if not line.endswith(b"\n"):
            rejected += 1
            continue
        try:
            text = line.decode("utf-8").strip()
            parts = text.split(" ", 2)
            if parts[0] in _EVENTS:
                name, fields = parts[0], json.loads(text.split(" ", 1)[1])
            else:
                name, fields = parts[1], json.loads(parts[2])
            if name not in _EVENTS or not isinstance(fields, dict):
                raise ValueError("unknown event")
        except (UnicodeError, ValueError, IndexError):
            rejected += 1
            continue
        safe: dict[str, Any] = {"event": name, "event_index": len(events) + 1}
        for key, value in fields.items():
            if (
                key in _NUMERIC
                and type(value) in (int, float)
                and math.isfinite(value)
                and value >= 0
            ):
                safe[key] = value
            elif key == "database_observation_seconds" and value is None:
                safe[key] = None
            elif key == "resources" and isinstance(value, dict):
                safe[key] = _safe_resources(value)
            elif name == "STEP8_FAILURE" and key == "diagnostics" and isinstance(value, dict):
                safe[key] = _safe_diagnostics(value)
            elif (
                name == "STEP8_FAILURE_CAPSULE"
                and key == "content_hash"
                and isinstance(value, str)
                and re.fullmatch(_SHA, value)
            ):
                safe[key] = value
            elif (
                name
                in {
                    "STEP8_CAPSULE_FAILURE",
                    "STEP8_ATTEMPT_COMMIT_FAILURE",
                    "STEP8_OBSERVE_FAILURE",
                }
                and key == "error_type"
                and isinstance(value, str)
                and re.fullmatch(r"[A-Za-z0-9_]{1,128}", value)
            ):
                safe[key] = value
            elif (
                key in {"reused", "unknown_provider_outcome", "scientific_identity_unchanged"}
                and type(value) is bool
            ):
                safe[key] = value
            elif (
                key == "stage"
                and isinstance(value, str)
                and (value in SOURCE_OWNERS or value == "worker")
            ):
                safe[key] = value
            elif (
                key in {"resource", "resource_class"}
                and isinstance(value, str)
                and value
                in {
                    "cpu",
                    "gpu_9b",
                    "gpu_27b",
                    "gpu_embedding",
                    "api",
                }
            ):
                safe[key] = value
            elif (
                key == "state"
                and isinstance(value, str)
                and value
                in {
                    "started",
                    "running",
                    "completed",
                    "failed",
                    "interrupted",
                    "budget_exhausted",
                }
            ):
                safe[key] = value
            elif key == "error_code":
                safe[key] = "recorded_error"
        events.append(json_bytes(safe))
    return b"".join(events), {
        "event_count": len(events),
        "rejected_line_count": rejected,
        "source_byte_count": len(payload),
    }


def _read_failure_capsule(
    directory: Path, attempt_id: str, terminal: dict[str, Any], safe_log: bytes
) -> tuple[bytes | None, tuple[str, ...]]:
    """Verify private recovery data without exposing or executing its resume argv."""
    events = [json.loads(line) for line in safe_log.splitlines()]
    capsule_events = [e for e in events if e["event"] == "STEP8_FAILURE_CAPSULE"]
    expected = (
        terminal.get("first_failure") is not None
        or capsule_events
        or any(e["event"] == "STEP8_CAPSULE_FAILURE" for e in events)
    )
    try:
        data = read_campaign_file(directory, f"attempts/{attempt_id}/failure-capsule.json")
    except FileNotFoundError:
        return None, ("failure_capsule_missing",) if expected else ()
    except (OSError, ValueError):
        return None, ("failure_capsule_unreadable",)
    problems = set()
    hashes = {e.get("content_hash") for e in capsule_events}
    if hashes != {hashlib.sha256(data).hexdigest()}:
        problems.add("failure_capsule_content_hash_mismatch")
    try:
        capsule = json.loads(data)
        manifest_hash = hashlib.sha256(
            read_campaign_file(directory, "manifests/campaign.json")
        ).hexdigest()
        if (
            not isinstance(capsule, dict)
            or capsule.get("schema_version") != "step8-failure-capsule-v1"
            or capsule.get("campaign_manifest_hash") != manifest_hash
            or not isinstance(capsule.get("failure"), dict)
            or capsule["failure"] != terminal.get("first_failure")
        ):
            problems.add("failure_capsule_identity_mismatch")
    except (OSError, ValueError):
        problems.add("failure_capsule_invalid")
    return data, tuple(sorted(problems))


def _snapshot_object(item: ExportArtifact, destination: Path) -> ExportArtifact:
    """Bounded-memory immutable CAS copy; the packer independently rechecks bytes."""
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError("staging cannot follow symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".step8-audit-", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as target, item.source.open("rb") as source:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if temporary.stat().st_size != item.size_bytes or sha256_file(temporary) != item.sha256:
            raise ValueError("inventory object changed during snapshot")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if (
                destination.stat().st_size != item.size_bytes
                or sha256_file(destination) != item.sha256
            ):
                raise ValueError("immutable staged object conflict") from None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return replace(item, source=destination)


def export_campaign(
    directory: Path,
    *,
    visibility: Literal["public", "private"],
    private_authorized: bool = False,
    public_review: PublicReview | None = None,
    inventory_path: Path | None = None,
    artifact_store: CampaignArtifactStore | None = None,
    selection_hash: str | None = None,
    output_dir: Path | None = None,
    split_bytes: int | None = None,
) -> dict[str, Any]:
    """Stage immutable facts then return JSON-safe paths/hashes from export_bundle.

    Public bundles contain only reviewed aggregate analysis and audit counts.
    Private bundles include verified receipts/CAS, explicit inventory payloads,
    all discovered attempts and sanitized bounded logs. Unknown coverage is
    exported honestly as incomplete; no DB/model/service access happens here.
    """
    if visibility not in {"private", "public"}:
        raise ValueError("visibility must be explicit")
    if visibility == "private" and private_authorized is not True:
        raise ValueError("private export requires explicit authorization")
    if visibility == "public" and not isinstance(public_review, PublicReview):
        raise ValueError("public export requires explicit license/privacy review")
    directory = Path(directory)
    facts = load_campaign_facts(
        directory, selection_hash=selection_hash, artifact_store=artifact_store
    )
    if inventory_path is None:
        try:
            raw_inventory = read_campaign_file(directory, "private/audit-inventory.json")
        except FileNotFoundError:
            raw_inventory = None
    else:
        path = Path(inventory_path).absolute()
        if not path.is_relative_to((directory / "private").resolve()):
            raise ValueError("inventory must be a campaign-private snapshot")
        raw_inventory = read_campaign_file(directory, str(path.relative_to(directory.resolve())))
    if raw_inventory is not None:
        # Cost/performance and the packer must consume this exact chosen snapshot.
        facts.files["private/audit-inventory.json"] = raw_inventory
        facts.input_hashes["private/audit-inventory.json"] = hashlib.sha256(
            raw_inventory
        ).hexdigest()
    report = _analyze_facts(facts)
    # Byte snapshots are taken before the packer starts, including logs and indices.
    staged: dict[str, tuple[bytes, str, LogCutoff | None]] = {}
    objects: dict[str, ExportArtifact] = {}

    def stage(name, value, kind, cutoff=None):
        payload = value if isinstance(value, bytes) else json_bytes(value)
        if name in staged and staged[name] != (payload, kind, cutoff):
            raise ValueError("conflicting staged artifact")
        staged[name] = payload, kind, cutoff

    stage("reports/analysis.json", report, "aggregate")
    problems = set()
    inventory = None
    if raw_inventory is not None:
        inventory = CampaignAuditInventory.model_validate_json(raw_inventory)
        if (inventory.campaign_id, inventory.selection_hash) != (
            facts.manifest.campaign_id,
            facts.selection.content_hash,
        ):
            raise ValueError("audit inventory campaign/selection mismatch")
        if not inventory.complete or set(inventory.coverage) != _COVERAGE:
            problems.add("authoritative_inventory_coverage_incomplete")
        try:
            load_process_facts(inventory, facts.store)
        except (ValueError, OSError, KeyError, TypeError):
            problems.add("process_facts_unavailable_or_incomplete")
    else:
        problems.add("authoritative_inventory_absent")
    records: list[AuditRecord] = []
    attempt_ids, terminal_attempts, log_stats = set(), set(), []
    for attempt in (
        sorted((directory / "attempts").iterdir()) if (directory / "attempts").exists() else ()
    ):
        if (
            attempt.is_symlink()
            or not attempt.is_dir()
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", attempt.name)
        ):
            raise ValueError("unsafe attempt directory")
        attempt_ids.add(attempt.name)
        attempt_paths = []
        terminal_item: dict[str, Any] = {}
        safe = b""
        for filename in ("attempt.json", "terminal.json"):
            try:
                data = read_campaign_file(directory, f"attempts/{attempt.name}/{filename}")
            except FileNotFoundError:
                problems.add("attempt_metadata_or_terminal_absent")
                continue
            item = json.loads(data)
            if item.get("attempt_id") != attempt.name:
                raise ValueError("attempt identity mismatch")
            if filename == "attempt.json" and item.get("campaign_id") != facts.manifest.campaign_id:
                raise ValueError("attempt campaign mismatch")
            if filename == "terminal.json":
                terminal_item = item
                if item.get("state") not in {
                    "completed",
                    "failed",
                    "interrupted",
                    "budget_exhausted",
                }:
                    problems.add("attempt_not_terminal")
                else:
                    terminal_attempts.add(attempt.name)
            archive = f"private/attempts/{attempt.name}/{filename}"
            stage(archive, data, "performance")
            attempt_paths.append(archive)
        try:
            raw_log = read_campaign_file(directory, f"attempts/{attempt.name}/safe-execution.log")
        except FileNotFoundError:
            problems.add("attempt_log_absent")
        else:
            safe, stats = _safe_log(raw_log)
            log_stats.append(stats)
            if stats["rejected_line_count"]:
                problems.add("attempt_log_contains_rejected_or_partial_lines")
            archive = f"private/attempts/{attempt.name}/safe-execution.jsonl"
            stage(archive, safe, "log", LogCutoff(len(safe), f"event-{stats['event_count']}"))
            attempt_paths.append(archive)
            stage(
                f"private/attempts/{attempt.name}/log-index.json",
                {
                    **stats,
                    "source_sha256": hashlib.sha256(raw_log).hexdigest(),
                    "exported_sha256": hashlib.sha256(safe).hexdigest(),
                    "snapshot_is_final": attempt.name in terminal_attempts,
                    "telemetry_scope": "worker_process",
                },
                "performance",
            )
        capsule, capsule_problems = _read_failure_capsule(
            directory, attempt.name, terminal_item, safe
        )
        problems.update(capsule_problems)
        if capsule is not None:
            archive = f"private/attempts/{attempt.name}/failure-capsule.json"
            stage(archive, capsule, "failure")
            attempt_paths.append(archive)
        if attempt_paths:
            records.append(AuditRecord("attempt", attempt.name, tuple(attempt_paths)))
    if not attempt_ids and not (inventory is not None and inventory.conformance_run_ids):
        problems.add("campaign_attempts_absent")
    verified_artifacts = {}
    if inventory is not None:
        stage("private/audit-inventory.json", raw_inventory, "variant_cell_reuse")
        for item in inventory.artifacts:
            try:
                path = facts.store.verify(item.sha256)
                if path.stat().st_size != item.size_bytes:
                    raise ValueError("changed inventory payload")
            except (ValueError, OSError):
                problems.add("inventory_artifact_missing_or_corrupt")
                continue
            archive = f"private/{item.scope}/{item.kind}/{item.sha256}"
            objects[archive] = ExportArtifact(
                path, archive, item.kind, item.size_bytes, item.sha256
            )
            verified_artifacts[item.artifact_id] = archive
        for record in inventory.records:
            if not set(record.artifacts.values()) <= set(verified_artifacts):
                continue
            if record.kind == "reranker_call":
                lineage_path = objects[verified_artifacts[record.artifacts["lineage"]]].source
                with lineage_path.open("rb") as source:
                    lineage = json.load(source)
                if (
                    lineage.get("model_call_id") != record.record_id
                    or lineage.get("attempt_id") != record.attempt_id
                    or lineage.get("conformance_run_id") != record.conformance_run_id
                ):
                    raise ValueError("reranker lineage identity mismatch")
            records.append(
                AuditRecord(
                    record.kind,
                    record.record_id,
                    tuple(sorted({verified_artifacts[ref] for ref in record.artifacts.values()})),
                )
            )
        expected = set(inventory.expected_records)
        actual = {(r.kind, r.record_id) for r in records}
        if (
            actual != expected
            or {key for kind, key in expected if kind == "attempt"} != attempt_ids
        ):
            problems.add("all_attempt_audit_inventory_mismatch")
    else:
        expected = set()
    if facts.missing_artifact_count:
        problems.add("component_artifact_missing_or_corrupt")
    audit_complete = not problems
    audit: dict[str, Any] = {
        "schema_version": "step8-campaign-audit-v1",
        "artifact_completeness": "complete_against_declared_inventory"
        if audit_complete
        else "incomplete",
        "database_inventory_verified": False,
        "inventory_provenance": "evaluator_snapshot_attestation" if inventory else "absent",
        "expected_record_count": len(expected),
        "verified_record_count": len(records),
        "verified_inventory_artifact_count": len(verified_artifacts),
        "attempt_count": len(attempt_ids),
        "conformance_run_count": len(inventory.conformance_run_ids) if inventory else 0,
        "terminal_attempt_count": len(terminal_attempts),
        "safe_log_event_count": sum(s["event_count"] for s in log_stats),
        "missing_cas_artifact_count": facts.missing_artifact_count,
        "limitations": sorted(
            problems
            | {
                "inventory_truth_requires_independent_evaluator_database_review",
                "safe_logs_are_allowlisted_projection_with_explicit_source_cutoff",
            }
        ),
        "scientific_approval": False,
    }
    stage("reports/audit.json", audit, "aggregate")
    if visibility == "private":
        for name, data in facts.files.items():
            stage("private/campaign/" + name, data, "variant_cell_reuse")
        for digest in sorted({h for r in facts.receipts.values() for h in r.artifact_hashes}):
            try:
                path = facts.store.verify(digest)
            except (ValueError, OSError):
                problems.add("component_artifact_missing_or_corrupt")
                continue
            archive = f"private/component-outputs/{digest}.json"
            objects[archive] = ExportArtifact(
                path, archive, "variant_cell_reuse", path.stat().st_size, digest
            )
        from pmmc.experiments.step8_analysis import threshold_proof_hashes

        for threshold_path in report["analysis"]["input_hashes"]:
            if not threshold_path.startswith("private/threshold-facts"):
                continue
            payload = facts.files[threshold_path]
            stage(threshold_path, payload, "variant_cell_reuse")
            for row in json.loads(payload)["rows"]:
                digests = set(threshold_proof_hashes(row))
                try:
                    context = facts.store.load_json(row["context_sha256"])
                    digests.update(threshold_proof_hashes(context))
                    if context.get("matching_sha256"):
                        matching = facts.store.load_json(context["matching_sha256"])
                        digests.update(threshold_proof_hashes(matching))
                except (ValueError, OSError):
                    problems.add("threshold_artifact_missing_or_corrupt")
                for digest in sorted(digests):
                    kind = "model_payload"
                    try:
                        path = facts.store.verify(digest)
                    except (ValueError, OSError):
                        problems.add("threshold_artifact_missing_or_corrupt")
                        continue
                    archive = f"private/threshold-payloads/{kind}/{digest}.json"
                    objects[archive] = ExportArtifact(
                        path, archive, kind, path.stat().st_size, digest
                    )
    else:
        staged = {name: value for name, value in staged.items() if value[1] == "aggregate"}
        records = []
        objects = {}
    if problems:
        audit_complete = False
        audit["artifact_completeness"] = "incomplete"
        audit["limitations"] = sorted(set(audit["limitations"]) | problems)
        staged["reports/audit.json"] = json_bytes(audit), "aggregate", None
    identity = canonical_content_hash(
        {
            **{name: hashlib.sha256(data).hexdigest() for name, (data, _, _) in staged.items()},
            **{name: item.sha256 for name, item in objects.items()},
        }
    )
    staging_root = directory / "private/export-staging" / identity / visibility
    artifacts = []
    for name, (data, kind, cutoff) in sorted(staged.items()):
        destination = staging_root / name
        if any(p.is_symlink() for p in (destination, *destination.parents)):
            raise ValueError("staging cannot follow symlinks")
        durable_write(destination, data, immutable=True)
        artifacts.append(
            ExportArtifact(
                destination,
                name,
                kind,
                len(data),
                hashlib.sha256(data).hexdigest(),
                public_review=public_review if visibility == "public" else None,
                log_cutoff=cutoff,
            )
        )
    # On incomplete exports the helper inventory describes VERIFIED records only;
    # the authoritative expected inventory and missing counts remain in the bundle.
    for name, item in sorted(objects.items()):
        artifacts.append(_snapshot_object(item, staging_root / name))
    helper_inventory = AuditInventory(tuple((r.kind, r.record_id) for r in records), tuple(records))
    status: Literal["complete", "incomplete"] = (
        "complete" if audit_complete and report["status"] == "complete" else "incomplete"
    )
    result = export_bundle(
        ExportPlan(
            bundle_name="step8-" + identity[:24],
            visibility=visibility,
            artifacts=tuple(artifacts),
            private_authorized=private_authorized,
            run_terminal_status=status,
            audit=helper_inventory,
            split_bytes=split_bytes,
        ),
        Path(output_dir)
        if output_dir is not None
        else directory / "transfer" / f"{identity}-{visibility}",
    )
    return {
        "status": status,
        "visibility": visibility,
        "scientific_approval": False,
        "analysis_hash": report["report_hash"],
        "audit": audit,
        "staging_directory": str(staging_root.resolve()),
        "output_dir": str(result.output_dir.resolve()),
        "receipt_path": str(result.receipt_path.resolve()),
        "archive_sha256": result.archive_sha256,
        "parts": [asdict(part) for part in result.parts],
    }
