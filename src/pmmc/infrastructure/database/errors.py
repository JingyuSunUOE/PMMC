"""Persistence-specific exceptions and safe database diagnostics."""

from __future__ import annotations

import errno
import re
from typing import Any

from sqlalchemy.exc import TimeoutError as PoolTimeoutError

_DATABASE_GUARD_CODES = {
    "compiler model call does not match its COMPILE job": "compiler_model_call_job_scope_mismatch",
    "experiment run is not registered": "experiment_run_not_registered",
    "experiment run has no immutable config hash": "experiment_run_config_hash_missing",
    "terminal experiment run is immutable": "experiment_run_terminal_immutable",
    "terminal experiment run state is immutable": "experiment_run_state_terminal_immutable",
    "query embedding denominator or channels are incomplete": "query_embedding_channels_incomplete",
    "active or PMMC source answer lineage is incomplete": "query_active_answer_lineage_incomplete",
    "Native-MM query/answer lineage is incomplete": "query_fallback_lineage_incomplete",
    "query executor terminal accounting incomplete": "query_terminal_accounting_incomplete",
    "query program execution or sufficiency facts are incomplete": (
        "query_execution_facts_incomplete"
    ),
    "query answer selection fallback lineage is incomplete": "query_selection_fallback_mismatch",
    "completed QBank coverage run is immutable": "qbank_coverage_terminal_immutable",
    "QBank coverage query denominator is incomplete": "qbank_coverage_queries_incomplete",
    "QBank coverage gate report is incomplete or failed": "qbank_coverage_report_incomplete",
    "QBank coverage metric grid is incomplete": "qbank_coverage_metric_grid_incomplete",
    "QBank oracle metric grid is incomplete": "qbank_coverage_oracle_grid_incomplete",
    "QBank coverage embedding, gold, or applicability facts are incomplete": (
        "qbank_coverage_embedding_gold_applicability_incomplete"
    ),
    "QBank coverage candidate ranks are not contiguous": "qbank_coverage_ranks_incomplete",
    "compiler fact scope does not match its unit": "compiler_fact_unit_scope_mismatch",
    "compiler producer model-call lineage is inconsistent": (
        "compiler_producer_model_call_lineage_mismatch"
    ),
    "compiler fact qbank lineage does not match its unit": "compiler_fact_qbank_scope_mismatch",
    "compiler fact program lineage does not match its unit": "compiler_fact_program_scope_mismatch",
    "doubter execution lineage is inconsistent": "doubter_execution_lineage_mismatch",
    "compiler run must keep query denominator counters at zero": (
        "compiler_query_denominator_nonzero"
    ),
    "compiler unit denominator is incomplete": "compiler_unit_denominator_incomplete",
    "compiler units require exactly one succeeded COMPILE job": (
        "compiler_compile_job_denominator_incomplete"
    ),
    "compiler unit membership is incomplete": "compiler_unit_membership_incomplete",
    "compiler target stage results are incomplete": "compiler_stage_results_incomplete",
    "compiler stage counts differ from persisted facts": ("compiler_stage_fact_count_mismatch"),
    "Step 7 compiler identity is incomplete": "step7_compiler_identity_incomplete",
    "Step 7 requires one viable R0 v2 manifest": "step7_r0_manifest_incomplete",
    "Step 7 compiler has no native eligible candidates": (
        "step7_native_candidate_denominator_empty"
    ),
    "Step 7 requires r0-r3 projections": "step7_round_projection_grid_incomplete",
    "Step 7 round projection denominator mismatch": ("step7_round_projection_denominator_mismatch"),
    "Step 7 Doubter r0 denominator is incomplete": "step7_doubter_r0_incomplete",
    "Step 7 active-r2 routing projection is incomplete": ("step7_active_r2_projection_incomplete"),
    "Step 7 active-r2 decisions do not close the denominator": (
        "step7_decision_denominator_incomplete"
    ),
    "Step 7 routing contains unsafe or unresolved members": ("step7_routing_member_unresolved"),
    "terminal compiler run is immutable": "terminal_compiler_run_immutable",
    "Step 7B compiler identity is incomplete": "step7b_compiler_identity_incomplete",
    "Step 7B requires one viable R0 v3 manifest": "step7b_r0_manifest_incomplete",
    "Step 7B compiler has no native eligible candidates": "step7b_native_candidates_empty",
    "Step 7B expected evidence recall is zero but path is sufficient": (
        "step7b_zero_expected_recall_sufficient"
    ),
    "Step 7B empty retrieved path cannot pass Doubter": "step7b_empty_path_doubter_pass",
    "Step 7B extension operator did not contribute evidence": "step7b_extension_no_evidence",
    "Step 7B revision lineage lacks an effective typed patch": "step7b_revision_patch_missing",
    "Step 7B requires continuous r0-r3 projections": "step7b_projection_grid_incomplete",
    "Step 7B active-r2 routing projection is incomplete": "step7b_routing_projection_incomplete",
    "Step 7B routing contains unsafe or unresolved members": "step7b_routing_member_unresolved",
    "Step 7B query run has an incompatible compiler parent": "step7b_query_compiler_incompatible",
    "Step 7B reranker denominator is incomplete": "step7b_reranker_denominator_incomplete",
    "Step 7B active route escaped bounded reranker acceptance": "step7b_route_reranker_mismatch",
    "Step 7B query evidence relevance denominator is incomplete": (
        "step7b_query_relevance_denominator_incomplete"
    ),
    "Step 7B PMMC answer lacks an accepted strict contract fact": "step7b_answer_contract_missing",
}

_COMPLETION_FUNCTIONS = {
    "enforce_run_writable",
    "enforce_run_state_transition",
    "enforce_query_executor_completion",
    "enforce_qbank_coverage_completion",
    "enforce_step7b_compiler_completion",
    "enforce_step7b_query_completion",
}
_FORWARDED_DATABASE_KEYS = {
    "error_code",
    "timeout_source",
    "database_error_type",
    "sqlstate",
    "database_guard_code",
    "column_name",
    "database_function",
    "database_function_line",
}


def safe_forwarded_database_diagnostics(values: Any) -> dict[str, str]:
    if not isinstance(values, dict):
        return {}
    return {
        key: value
        for key, value in values.items()
        if key in _FORWARDED_DATABASE_KEYS
        and isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9_]{1,128}", value)
    }


def _completion_function_diagnostics(diag: Any) -> dict[str, str]:
    """Do not emit PL/pgSQL context: it contains SQL and possibly private values."""
    result: dict[str, str] = {}
    context = getattr(diag, "context", None)
    if isinstance(context, str):
        match = re.search(
            r"PL/pgSQL function (?:pmmc_(?:runtime|evaluation)\.)?([a-z0-9_]+)\(\) line ([0-9]+)",
            context,
        )
        if match and match[1] in _COMPLETION_FUNCTIONS:
            result = {"database_function": match[1], "database_function_line": match[2]}
    primary = getattr(diag, "message_primary", None)
    # Only this known schema bug can be decoded from free text. Other identifiers
    # must come from structured driver fields, not from arbitrary error messages.
    if primary == "column v.provider_schema_conformant does not exist":
        result["column_name"] = "provider_schema_conformant"
        result["database_guard_code"] = "step7b_verifier_native_column_missing"
    return result


def _database_guard_code(source: Any, diagnostics: Any) -> str | None:
    candidates: list[str] = []
    for value in (source, diagnostics):
        if value is None:
            continue
        for attribute in ("message_primary", "message"):
            try:
                candidate = getattr(value, attribute, None)
            except Exception:
                candidate = None
            if isinstance(candidate, str):
                candidates.append(candidate)
        try:
            args = getattr(value, "args", ())
        except Exception:
            args = ()
        if isinstance(args, tuple):
            candidates.extend(item for item in args if isinstance(item, str))
    for candidate in candidates:
        normalized = candidate.strip()
        direct = _DATABASE_GUARD_CODES.get(normalized)
        if direct is not None:
            return direct
        # asyncpg adapters may prefix the fixed server message with an exception type.
        for message, code in _DATABASE_GUARD_CODES.items():
            if normalized.endswith(message):
                return code
    return None


def safe_database_error_diagnostics(error: BaseException) -> dict[str, str]:
    """Extract only database identifiers, never SQL text or bound values."""

    diagnostics = {"error_type": type(error).__name__}
    chain: list[BaseException] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 8:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__

    for item in chain:
        if isinstance(item, PoolTimeoutError):
            diagnostics.setdefault("error_code", "database_pool_timeout")
            diagnostics.setdefault("timeout_source", "sqlalchemy_queue_pool")
        diagnostics.update(
            safe_forwarded_database_diagnostics(getattr(item, "database_diagnostics", None))
        )
        if isinstance(item, OSError) and isinstance(item.errno, int):
            diagnostics.setdefault("os_errno", str(item.errno))
            diagnostics.setdefault(
                "os_error_name",
                errno.errorcode.get(item.errno, "UNKNOWN")[:32],
            )
        failure_stage = getattr(item, "failure_stage", None)
        if isinstance(failure_stage, str) and failure_stage and "failure_stage" not in diagnostics:
            diagnostics["failure_stage"] = failure_stage[:128]
        persistence_operation = getattr(item, "persistence_operation", None)
        if (
            isinstance(persistence_operation, str)
            and persistence_operation
            and "persistence_operation" not in diagnostics
        ):
            diagnostics["persistence_operation"] = persistence_operation[:128]
        for attribute in (
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
        ):
            value = getattr(item, attribute, None)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                and attribute not in diagnostics
            ):
                diagnostics[attribute] = str(value)
        error_code = getattr(item, "error_code", None)
        if isinstance(error_code, str) and error_code:
            if "error_code" not in diagnostics:
                diagnostics["error_code"] = error_code[:128]
            elif hasattr(item, "retryable") and "provider_error_code" not in diagnostics:
                diagnostics["provider_error_code"] = error_code[:128]
        status_code = getattr(item, "provider_status_code", None)
        if (
            isinstance(status_code, int)
            and 100 <= status_code <= 599
            and "provider_status_code" not in diagnostics
        ):
            diagnostics["provider_status_code"] = str(status_code)
        retryable = getattr(item, "retryable", None)
        if retryable is False:
            diagnostics["provider_retryable"] = "false"
        elif retryable is True:
            diagnostics.setdefault("provider_retryable", "true")
        category = getattr(item, "provider_error_category", None)
        if isinstance(category, str) and category in {
            "unclassified", "context_length_exceeded", "structured_output_rejected",
            "image_rejected",
        }:
            diagnostics.setdefault("provider_error_category", category)
        for field in (
            "requested_output_tokens", "provider_output_tokens", "provider_output_ceiling",
        ):
            value = getattr(item, field, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                diagnostics.setdefault(field, str(value))

    database_chain: list[Any] = []
    database_seen: set[int] = set()

    def add_database_error(value: Any) -> None:
        while value is not None and id(value) not in database_seen:
            database_chain.append(value)
            database_seen.add(id(value))
            value = getattr(value, "__cause__", None) or getattr(value, "__context__", None)

    for item in chain:
        original = getattr(item, "orig", None)
        if original is not None:
            add_database_error(original)
    if not database_chain:
        return diagnostics
    diagnostics["database_error_type"] = type(database_chain[0]).__name__
    for source in database_chain:
        for source_name in ("sqlstate", "pgcode"):
            try:
                value = getattr(source, source_name, None)
            except Exception:
                value = None
            if isinstance(value, str) and value:
                diagnostics.setdefault("sqlstate", value[:32])
                break
        try:
            database_diagnostics = getattr(source, "diag", None)
        except Exception:
            database_diagnostics = None
        diagnostics.update(_completion_function_diagnostics(database_diagnostics))
        diagnostic_sources = (
            (source, database_diagnostics) if database_diagnostics is not None else (source,)
        )
        guard_code = _database_guard_code(source, database_diagnostics)
        if guard_code is not None:
            diagnostics.setdefault("database_guard_code", guard_code)
        for attribute in (
            "schema_name",
            "table_name",
            "column_name",
            "constraint_name",
        ):
            for diagnostic_source in diagnostic_sources:
                try:
                    value = getattr(diagnostic_source, attribute, None)
                except Exception:
                    value = None
                if isinstance(value, str) and value:
                    diagnostics.setdefault(attribute, value[:256])
                    break
    return diagnostics


def safe_database_error_summary(diagnostics: dict[str, str]) -> str:
    """Render the allow-listed diagnostics in deterministic order."""

    ordered_keys = (
        "error_type",
        "failure_stage",
        "persistence_operation",
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
        "error_code",
        "provider_error_code",
        "provider_status_code",
        "provider_retryable",
        "provider_error_category",
        "requested_output_tokens",
        "provider_output_tokens",
        "provider_output_ceiling",
        "os_errno",
        "os_error_name",
        "database_error_type",
        "sqlstate",
        "database_guard_code",
        "database_function",
        "database_function_line",
        "schema_name",
        "table_name",
        "column_name",
        "constraint_name",
    )
    return "; ".join(f"{key}={diagnostics[key]}" for key in ordered_keys if key in diagnostics)


class PersistenceError(RuntimeError):
    """Base error for database and canonical import operations."""


class DatabaseConfigurationError(PersistenceError):
    """Required database configuration is absent or invalid."""


class SnapshotIntegrityError(PersistenceError):
    """A canonical snapshot does not satisfy its immutable manifest."""


class SnapshotImportError(PersistenceError):
    """A validated snapshot could not be imported atomically."""


class ExperimentPersistenceError(PersistenceError):
    """An experiment fact could not be persisted safely."""


class PersistenceConflictError(ExperimentPersistenceError):
    """A stable record ID already exists with different content."""


class InvalidStateTransitionError(ExperimentPersistenceError):
    """A persisted run or job state transition is invalid."""
