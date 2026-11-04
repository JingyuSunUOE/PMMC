"""Offline, null-preserving cost projections from an evaluator-attested snapshot.

No database, provider, rate lookup, or artifact materialization is performed here.
Deployment projections follow consumed relations, never the whole research run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation.paper_report import CostFact, CostUse, break_even_queries, summarize_costs
from pmmc.evaluation.paper_statistics import MethodVariant
from pmmc.experiments.step8_artifacts import json_bytes
from pmmc.experiments.step8_audit import PROCESS_REQUIRED_TABLES, CampaignAuditInventory
from pmmc.experiments.step8_cost_dependencies import (
    CALL_FIELDS,
    RAG_COST_DEPENDENCIES_ARTIFACT,
    RAG_COST_DEPENDENCIES_VERSION,
    DependencyUnproven,
    load_dependency_artifact,
    validate_rag_index_dependencies,
)
from pmmc.experiments.step8_questioner_cost_dependencies import imported_cost_aliases
from pmmc.schemas.answer_output import TypedAnswerResult
from pmmc.schemas.query_executor import QueryExperimentCellResult, runtime_fact_payload

if TYPE_CHECKING:
    from pmmc.experiments.step8_analysis import CampaignFacts

RUNTIME = "pmmc_runtime."
EVALUATION = "pmmc_evaluation."
AMORTIZATION_N = (1, 10, 100, 1000, 10000)
_HASH = re.compile(r"[a-f0-9]{64}\Z")
_CORE_TABLES = (
    RUNTIME + "model_calls",
    EVALUATION + "model_calls",
    RUNTIME + "benchmark_budget_approvals",
    RUNTIME + "benchmark_spend_reservations",
)
_USE_TABLES = (
    RUNTIME + "query_runtime_typed_answers",
    RUNTIME + "query_experiment_cell_results",
    EVALUATION + "evaluation_run_states",
)


class _Unavailable(ValueError):
    """A content-free reason suitable for the public aggregate."""


def _safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    return value


def _money(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise _Unavailable("invalid_cost_measurement") from None
    if not result.is_finite() or result < 0 or isinstance(value, bool):
        raise _Unavailable("invalid_cost_measurement")
    return result


def _tokens(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise _Unavailable("invalid_token_measurement")
    return value


def _unique(rows: list[dict], key: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        identity = row[key]
        if not isinstance(identity, str) or not identity:
            raise _Unavailable("invalid_row_identity")
        if identity in result and result[identity] != row:
            raise _Unavailable("conflicting_duplicate_row")
        result[identity] = row
    return result


def _dispatch_request_manifest(request: dict) -> dict:
    from pmmc.models.execution import _embedding_manifest, _generation_manifest
    from pmmc.schemas.model import EmbeddingRequest, GenerationRequest

    if "inputs" in request:
        return _embedding_manifest(EmbeddingRequest.model_validate(request))
    return _generation_manifest(GenerationRequest.model_validate(request))


class _Snapshot:
    def __init__(self, facts: CampaignFacts):
        from pmmc.experiments.step8_analysis import read_campaign_file

        self.facts = facts
        self.hashes: dict[str, str] = {}
        self.missing: set[str] = set()
        self.dependency_physical_ids: set[str] = set()
        self.call_aliases: dict[str, str] = {}
        name = "private/audit-inventory.json"
        # Use the same byte snapshot as the enclosing analysis/export, if present.
        raw = facts.files.get(name)
        if raw is None:
            raw = read_campaign_file(facts.directory, name)
        self.inventory = CampaignAuditInventory.model_validate_json(raw)
        inventory = self.inventory
        if (inventory.campaign_id, inventory.selection_hash) != (
            facts.manifest.campaign_id,
            facts.selection.content_hash,
        ):
            raise _Unavailable("inventory_campaign_selection_mismatch")
        self.hashes[name] = hashlib.sha256(raw).hexdigest()
        self.artifacts = {a.artifact_id: a for a in inventory.artifacts}
        source = self.artifacts.get("inventory-provenance")
        if (
            source is None
            or source.sha256 != inventory.source_snapshot_sha256
            or source.scope != "evaluation"
            or source.kind != "performance"
        ):
            raise _Unavailable("inventory_source_snapshot_unbound")
        self.source = self.artifact("inventory-provenance")
        if (
            self.source.get("schema_version") != "step8-database-audit-snapshot-v1"
            or self.source.get("campaign_id") != inventory.campaign_id
            or self.source.get("selection_hash") != inventory.selection_hash
            or self.source.get("source_role") != "evaluator"
            or self.source.get("isolation") != "repeatable_read_read_only"
            or not isinstance(self.source.get("database_snapshot"), dict)
            or self.source.get("database_snapshot", {}).get("database_role") != "pmmc_evaluator"
            or not self.source.get("database_snapshot", {}).get("snapshot")
            or set(self.source.get("conformance_run_ids", ())) != set(inventory.conformance_run_ids)
            or not all(
                isinstance(self.source.get(key), dict)
                for key in ("missing", "table_counts", "table_hashes")
            )
        ):
            raise _Unavailable("inventory_source_snapshot_identity_mismatch")
        missing_families = {k: set(v) for k, v in self.source["missing"].items() if v}
        deployment_only_gap = missing_families == {
            "cost_facts": {"cost_dependency_proof_incomplete"}
        }
        if (
            not inventory.complete or "cost_facts" not in inventory.coverage
        ) and not deployment_only_gap:
            self.missing.add("inventory_cost_coverage_incomplete")
        if self.source.get("missing", {}).get("cost_facts") and not deployment_only_gap:
            self.missing.add("snapshot_cost_coverage_incomplete")
        facts.files[name] = raw
        facts.input_hashes[name] = self.hashes[name]

    def artifact(self, identity: str) -> dict:
        ref = self.artifacts.get(identity)
        if ref is None:
            raise _Unavailable("required_inventory_artifact_missing")
        path = self.facts.store.verify(ref.sha256)
        raw = path.read_bytes()
        if len(raw) != ref.size_bytes or hashlib.sha256(raw).hexdigest() != ref.sha256:
            raise _Unavailable("inventory_artifact_integrity_failure")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise _Unavailable("inventory_artifact_not_object")
        self.hashes[identity] = ref.sha256
        return value

    def table(self, name: str) -> list[dict]:
        value = self.artifact("table:" + name)
        if self.artifacts["table:" + name].scope != (
            "evaluation" if name.startswith(EVALUATION) else "runtime"
        ):
            raise _Unavailable("snapshot_table_scope_mismatch")
        rows = value.get("rows")
        if (
            value.get("table") != name
            or not isinstance(rows, list)
            or not all(isinstance(r, dict) for r in rows)
            or self.source.get("table_counts", {}).get(name) != len(rows)
            or self.source.get("table_hashes", {}).get(name) != canonical_content_hash(rows)
        ):
            raise _Unavailable("snapshot_table_hash_or_count_mismatch")
        return rows

    def dependency_artifact(self, identity: str, version: str) -> dict:
        return load_dependency_artifact(self, identity, version)


def _normalize(snapshot: _Snapshot, tables: dict[str, list[dict]]):
    """Use explicit settlements, never a model-name price or reservation maximum."""
    approvals = _unique(tables[RUNTIME + "benchmark_budget_approvals"], "approval_id")
    reservations = _unique(tables[RUNTIME + "benchmark_spend_reservations"], "request_id")
    calls: dict[tuple[str, str], dict] = {}
    for scope in (RUNTIME, EVALUATION):
        for identity, row in _unique(tables[scope + "model_calls"], "model_call_id").items():
            if row["run_id"] not in set(snapshot.source.get("run_ids", ())) | set(
                snapshot.inventory.conformance_run_ids
            ) and row.get("evaluation_run_id") not in snapshot.source.get("evaluation_run_ids", ()):
                raise _Unavailable("model_call_outside_snapshot_scope")
            calls[scope, identity] = row

    dependency_ids = snapshot.source.get("dependency_model_call_ids", [])
    dependency_runs = snapshot.source.get("dependency_run_ids", [])
    if (
        not isinstance(dependency_ids, list)
        or not isinstance(dependency_runs, list)
        or not all(isinstance(v, str) and v for v in [*dependency_ids, *dependency_runs])
    ):
        raise _Unavailable("dependency_cost_scope_invalid")
    dependency_keys = {(RUNTIME, identity) for identity in dependency_ids}
    if any(
        key not in calls or calls[key]["run_id"] not in dependency_runs for key in dependency_keys
    ):
        raise _Unavailable("dependency_cost_scope_unbound")
    snapshot.call_aliases = imported_cost_aliases(
        snapshot, {identity: row for (scope, identity), row in calls.items() if scope == RUNTIME}
    )
    alias_keys = {(RUNTIME, identity) for identity in snapshot.call_aliases}

    request_ids: dict[tuple[str, str], str] = {}
    request_payloads: dict[tuple[str, str], dict] = {}
    for key, row in calls.items():
        if key in alias_keys:
            continue
        ref = ("runtime:" if key[0] == RUNTIME else "evaluation:") + str(row.get("input_asset_id"))
        if ref not in snapshot.artifacts:
            continue
        if (snapshot.artifacts[ref].kind, snapshot.artifacts[ref].scope) != (
            "model_payload",
            "runtime" if key[0] == RUNTIME else "evaluation",
        ):
            raise _Unavailable("call_request_artifact_scope_mismatch")
        payload = snapshot.artifact(ref)
        if snapshot.artifacts[ref].sha256 != row.get("prompt_hash"):
            raise _Unavailable("call_request_hash_mismatch")
        if payload.get("model_alias") != row.get("model_alias"):
            raise _Unavailable("call_request_model_mismatch")
        if not isinstance(payload.get("request_id"), str) or not payload["request_id"]:
            raise _Unavailable("call_request_identity_missing")
        request_ids[key] = payload["request_id"]
        request_payloads[key] = payload

    facts: dict[str, CostFact] = {}
    owners: dict[tuple[str, str], str] = {}
    categories: dict[str, set[str]] = defaultdict(set)
    reasons: Counter[str] = Counter()
    matched_requests = set(request_ids.values())
    settlements: dict[str, tuple[Decimal | None, str, dict, dict, dict]] = {}
    for request, row in reservations.items():
        approval = approvals.get(row["approval_id"])
        if approval is None or approval.get("campaign_id") != snapshot.inventory.campaign_id:
            raise _Unavailable("settlement_approval_identity_mismatch")
        rate = approval.get("rate_card_hash")
        digest = row["request_hash"]
        if not isinstance(rate, str) or not _HASH.fullmatch(rate) or not _HASH.fullmatch(digest):
            raise _Unavailable("settlement_rate_or_request_hash_invalid")
        intent = snapshot.artifact("provider-dispatch:" + digest)
        without_attempt = {k: v for k, v in intent.items() if k != "attempt_id"}
        if (
            canonical_content_hash(without_attempt) != digest
            or intent.get("request", {}).get("request_id") != request
            or intent.get("approval", {}).get("campaign_id") != snapshot.inventory.campaign_id
            or intent["approval"].get("approval_id") != row["approval_id"]
            or intent["approval"].get("rate_card_hash") != rate
            or intent["approval"].get("selection_hash") != approval.get("selection_hash")
            or intent.get("attempt_id")
            not in {
                *snapshot.inventory.conformance_run_ids,
                *(key for kind, key in snapshot.inventory.expected_records if kind == "attempt"),
            }
        ):
            raise _Unavailable("settlement_dispatch_identity_mismatch")
        envelope = snapshot.artifact("provider-envelope:" + digest)
        if (
            envelope.get("request_id") != request
            or envelope.get("request_hash") != digest
            or envelope.get("approval") != intent["approval"]
            or row.get("response_hash") != snapshot.artifacts["provider-envelope:" + digest].sha256
        ):
            raise _Unavailable("settlement_response_identity_mismatch")
        amount = _money(row.get("actual_usd")) if row["state"] == "settled" else None
        settlements[request] = amount, rate, row, intent, envelope
        # A dispatch with no trace is a real research expense. Only enumerate it
        # separately when all trace request identities rule out a duplicate.
        if request not in matched_requests:
            if len(request_ids) != len(calls) - len(alias_keys):
                snapshot.missing.add("unlinked_dispatch_may_duplicate_trace")
                reasons["unlinked_dispatch"] += 1
                continue
            identity = "dispatch:" + digest
            usage = (envelope.get("result") or {}).get("usage") or {}
            facts[identity] = CostFact(
                identity,
                amount,
                rate,
                input_tokens=_tokens(usage.get("input_tokens")),
                output_tokens=_tokens(usage.get("output_tokens")),
            )
            categories["unpersisted_dispatch"].add(identity)
            if envelope.get("error") or row["state"] != "settled":
                categories["failed_or_unknown"].add(identity)
            if intent["attempt_id"] in snapshot.inventory.conformance_run_ids:
                categories["conformance"].add(identity)

    for key, row in calls.items():
        if key in alias_keys:
            continue
        request_id = request_ids.get(key)
        settlement = settlements.get(request_id) if request_id is not None else None
        amount, rate = None, None
        identity = key[0] + row["model_call_id"]
        if settlement is not None:
            amount, rate, reserved, intent, envelope = settlement
            if intent["request"].get("model_alias") != row["model_alias"]:
                raise _Unavailable("settlement_trace_model_mismatch")
            if canonical_content_hash(request_payloads[key]) != canonical_content_hash(
                _dispatch_request_manifest(intent["request"])
            ):
                raise _Unavailable("settlement_trace_request_content_mismatch")
            result = envelope.get("result")
            if result is not None and (
                result.get("model_alias") != row["model_alias"]
                or result.get("request_id") != request_id
                or result.get("actual_model_id") != row.get("actual_model_id")
                or any(
                    row.get(field) != (result.get("usage") or {}).get(field)
                    for field in ("input_tokens", "output_tokens", "total_tokens")
                )
            ):
                raise _Unavailable("settlement_trace_response_mismatch")
            identity = "dispatch:" + reserved["request_hash"]
        elif key not in dependency_keys:
            reasons["no_verified_rate_card_settlement"] += 1
        measured = CostFact(
            identity,
            amount,
            rate,
            input_tokens=_tokens(row.get("input_tokens")),
            output_tokens=_tokens(row.get("output_tokens")),
        )
        if identity in facts and facts[identity] != measured:
            raise _Unavailable("conflicting_physical_call_measurements")
        facts[identity] = measured
        owners[key] = identity
        retries = row.get("retry_count", 0)
        if type(retries) is not int or retries < 0:
            raise _Unavailable("invalid_retry_count")
        if retries:
            if key not in dependency_keys or settlement is not None:
                categories["bundled_provider_retries"].add(identity)
                snapshot.missing.add("provider_retries_not_individually_measured")
        if key not in dependency_keys or settlement is not None:
            categories["evaluator" if key[0] == EVALUATION else "runtime"].add(identity)
            if row["status"] != "success":
                categories["failed_or_unknown"].add(identity)
            if row["run_id"] in snapshot.inventory.conformance_run_ids:
                categories["conformance"].add(identity)
            role = (row.get("request_config") or {}).get("request_metadata", {}).get("model_role")
            if role in {"answer_format_retry", "grounding_format_retry"}:
                categories["explicit_retry"].add(identity)
    for alias, original in snapshot.call_aliases.items():
        owners[RUNTIME, alias] = owners[RUNTIME, original]
    # External cache producers can be consumed now without being paid for again
    # in this campaign. An actual campaign dispatch still wins over reuse labels.
    snapshot.dependency_physical_ids = (
        {owners[key] for key in dependency_keys}
        - {
            physical
            for key, physical in owners.items()
            if key not in dependency_keys and key not in alias_keys
        }
        - {"dispatch:" + row["request_hash"] for row in reservations.values()}
    )
    for category in (
        "runtime",
        "evaluator",
        "failed_or_unknown",
        "conformance",
        "explicit_retry",
        "bundled_provider_retries",
        "unpersisted_dispatch",
    ):
        categories.setdefault(category, set())
    return facts, owners, calls, categories, reasons


def _answer(row: dict) -> TypedAnswerResult:
    value = row["payload"]["metadata"]["typed_result"]
    if canonical_content_hash(value) != row["content_hash"]:
        raise _Unavailable("typed_answer_hash_mismatch")
    answer = TypedAnswerResult.model_validate(value)
    for field, column in (
        ("typed_answer_id", "typed_answer_id"),
        ("run_id", "query_executor_run_id"),
        ("query_id", "query_id"),
        ("terminal_state", "terminal_state"),
        ("answer_model_call_id", "answer_model_call_id"),
    ):
        if getattr(answer, field) != row[column]:
            raise _Unavailable("typed_answer_lineage_mismatch")
    return answer


def _runtime_cell(row: dict) -> QueryExperimentCellResult:
    cell = QueryExperimentCellResult.model_validate(row["payload"])
    payload = cell.model_dump(mode="json", exclude={"cell_result_id", "content_hash", "created_at"})
    if canonical_content_hash(runtime_fact_payload(payload)) != row["content_hash"]:
        raise _Unavailable("runtime_cell_payload_hash_mismatch")
    for field in (
        "cell_result_id",
        "query_executor_run_id",
        "query_id",
        "cell_id",
        "selected_answer_id",
        "terminal_state",
        "content_hash",
    ):
        if getattr(cell, field) != row[field]:
            raise _Unavailable("runtime_cell_payload_lineage_mismatch")
    for field in ("treatment", "question_budget", "revision_depth"):
        if getattr(cell.cell, field) != row[field]:
            raise _Unavailable("runtime_cell_definition_mismatch")
    return cell


def _project_uses(snapshot, tables, owners, calls):
    """Verified evaluation bindings establish cell reuse, not whole-path accounting."""
    facts = snapshot.facts
    answers = _unique(tables[RUNTIME + "query_runtime_typed_answers"], "typed_answer_id")
    cells = _unique(tables[RUNTIME + "query_experiment_cell_results"], "cell_result_id")
    states = _unique(tables[EVALUATION + "evaluation_run_states"], "evaluation_run_id")
    definitions = {cell.cell_id: cell for cell in facts.cells}
    expected = {(c.cell_id, q) for c in facts.cells for q in facts.selection.query_ids[c.dataset]}
    bound: dict[tuple[str, str], tuple] = {}
    uses: dict[str, CostUse] = {}
    problems: Counter[str] = Counter()
    reused: Counter[str] = Counter()
    bindings: dict[tuple[str, str], dict] = {}
    for job_id in sorted(facts.verified_jobs):
        if facts.jobs[job_id].stage != "evaluation":
            continue
        receipt = facts.receipts[job_id]
        report = receipt.output
        digest = hashlib.sha256(json_bytes(report)).hexdigest()
        if digest not in receipt.artifact_hashes or facts.store.load_json(digest) != report:
            raise _Unavailable("evaluation_receipt_output_unverified")
        state = states.get(report.get("evaluation_run_id"))
        if state is None or state.get("metadata", {}).get("completion_report") != {
            k: v for k, v in report.items() if k != "artifact"
        }:
            problems["evaluation_completion_binding_missing"] += 1
            continue
        if report.get("campaign_id") != facts.manifest.campaign_id:
            raise _Unavailable("evaluation_campaign_mismatch")
        query = report["query_id"]
        for score in report["scores"]:
            key = score["cell_id"], query
            if key not in expected:
                raise _Unavailable("evaluation_cell_outside_selection")
            definition = definitions[key[0]]
            if report["selected_query"]["dataset"] != definition.dataset.value:
                raise _Unavailable("evaluation_cell_dataset_mismatch")
            binding = tuple(
                score.get(k)
                for k in ("source_fact_id", "source_fact_hash", "typed_answer_id", "terminal_state")
            )
            if key in bound:
                if bound[key] != binding:
                    raise _Unavailable("conflicting_cell_reuse_binding")
                continue
            try:
                runtime = None
                ids: list[tuple[str, str]] = []
                source_answers = []
                selected_id = score["typed_answer_id"]
                if definition.method == "pmmc":
                    row = cells.get(score["source_fact_id"])
                    if row is None or row["content_hash"] != score["source_fact_hash"]:
                        raise _Unavailable("runtime_cell_binding_missing")
                    runtime = _runtime_cell(row)
                    treatment = {
                        "active": "pmmc"
                        if definition.treatment.value == "main"
                        else definition.treatment.value,
                        "no_fallback": "no_fallback",
                        "forced_top1": "forced_top1_counterfactual",
                    }.get(definition.source_variant)
                    if (
                        row["query_id"] != query
                        or row["selected_answer_id"] != selected_id
                        or row["question_budget"] != definition.question_budget
                        or row["revision_depth"] != definition.round_id
                        or row["terminal_state"] != score["terminal_state"]
                        or row["treatment"] != treatment
                    ):
                        raise _Unavailable("runtime_cell_binding_mismatch")
                    if runtime.pmmc_answer_id:
                        ids.append((runtime.pmmc_answer_id, "pmmc_attempt"))
                    if selected_id:
                        ids.append((selected_id, "selected"))
                else:
                    if selected_id != score["source_fact_id"]:
                        raise _Unavailable("baseline_answer_binding_mismatch")
                    ids.append((selected_id, "baseline"))
                if not ids and score["terminal_state"] != "no_answer":
                    raise _Unavailable("cell_answer_unavailable")
                cell_uses: dict[str, CostUse] = {}
                for answer_id, purpose in dict(ids).items():
                    row = answers.get(answer_id)
                    if row is None:
                        raise _Unavailable("typed_answer_missing")
                    answer = _answer(row)
                    source_answers.append(answer)
                    if (
                        answer.query_id != query
                        or answer.snapshot_id != facts.manifest.snapshot_ids[definition.dataset]
                    ):
                        raise _Unavailable("answer_query_snapshot_mismatch")
                    if definition.method != "pmmc" and (
                        answer.method != definition.method
                        or answer.answer_variant != "primary"
                        or row["content_hash"] != score["source_fact_hash"]
                    ):
                        raise _Unavailable("baseline_answer_scope_mismatch")
                    if definition.method == "pmmc" and answer.method not in {
                        "pmmc",
                        "native_multimodal_rag",
                    }:
                        raise _Unavailable("pmmc_answer_scope_mismatch")
                    if (
                        answer_id == selected_id
                        and answer.terminal_state != score["terminal_state"]
                    ):
                        raise _Unavailable("selected_answer_terminal_mismatch")
                    if definition.method == "pmmc":
                        assert runtime is not None
                        if (
                            answer.method == "pmmc"
                            and answer.run_id != runtime.query_executor_run_id
                        ):
                            raise _Unavailable("pmmc_answer_run_mismatch")
                        if answer.method == "native_multimodal_rag" and (
                            not runtime.active_fallback
                            or runtime.fallback_answer_id != answer_id
                            or answer.answer_variant != "primary"
                            or runtime.cell.treatment
                            in {"no_fallback", "forced_top1_counterfactual"}
                            or answer.run_id == runtime.query_executor_run_id
                        ):
                            raise _Unavailable("native_fallback_use_unproven")
                        if purpose == "pmmc_attempt" and answer.method != "pmmc":
                            raise _Unavailable("pmmc_attempt_is_not_pmmc")
                    prefix = (
                        "native_mm_fallback"
                        if definition.method == "pmmc" and answer.method == "native_multimodal_rag"
                        else "native_mm_baseline"
                        if answer.method == "native_multimodal_rag"
                        else purpose
                    )
                    for call in answer.calls:
                        trace = calls.get((RUNTIME, call.model_call_id))
                        if (
                            trace is None
                            or (trace["run_id"], trace.get("query_id")) != (answer.run_id, query)
                            or trace.get("request_config", {})
                            .get("request_metadata", {})
                            .get("model_role")
                            != call.role
                        ):
                            raise _Unavailable("answer_role_call_unproven")
                        expected_model = (
                            answer.actual_model_id
                            if call.model_call_id == answer.answer_model_call_id
                            else answer.grounding.actual_model_id
                            if answer.grounding
                            and call.model_call_id == answer.grounding.calls[-1].model_call_id
                            else None
                        )
                        if expected_model is not None and (
                            trace.get("actual_model_id") != expected_model
                            or trace["status"] != "success"
                        ):
                            raise _Unavailable("terminal_answer_call_model_or_status_mismatch")
                        physical = owners[RUNTIME, call.model_call_id]
                        for scope in ("active_path", "logical_ablation"):
                            use_id = canonical_content_hash(
                                [key, physical, prefix, call.role, scope]
                            )
                            cell_uses[use_id] = CostUse(
                                use_id,
                                physical,
                                MethodVariant(definition.method, key[0]),
                                prefix + ":" + call.role,
                                scope,
                            )
                uses.update(cell_uses)
                bound[key] = binding
                bindings[key] = {
                    "runtime": runtime,
                    "answers": source_answers,
                    "uses": tuple(cell_uses.values()),
                }
                if selected_id:
                    reused[selected_id] += 1
            except (ValueError, KeyError, TypeError) as exc:
                problems[
                    str(exc) if isinstance(exc, _Unavailable) else "cell_use_proof_invalid"
                ] += 1
    return (
        tuple(uses.values()),
        {
            "status": "complete" if set(bound) == expected and not problems else "incomplete",
            "expected_query_cell_count": len(expected),
            "proven_query_cell_count": len(bound),
            "missing_query_cell_count": len(expected - set(bound)),
            "shared_selected_answer_count": sum(count > 1 for count in reused.values()),
            "reused_answer_cell_count": sum(count - 1 for count in reused.values()),
            "reasons": dict(sorted(problems.items())),
        },
        bindings,
    )


def _empty(reason: str) -> dict:
    return {
        "schema_version": "step8-campaign-costs-v1",
        "status": "incomplete",
        "actual_research": {"status": "incomplete", "totals": None, "reasons": [reason]},
        "answer_role_projection": {"status": "incomplete", "method_costs": []},
        "active_path": {"status": "incomplete", "totals": None, "reasons": [reason]},
        "amortized": [],
        "input_hashes": {},
    }


class _Deployment:
    """Consumer-side joins over the process index; no new producer attestations."""

    def __init__(self, snapshot, owners, calls):
        self.snapshot, self.owners, self.calls = snapshot, owners, calls
        self.cache: dict[str, list[dict]] = {}
        self.index = snapshot.artifact("process-facts:index")
        if (
            self.index.get("schema_version") != "step8-process-facts-v1"
            or self.index.get("campaign_id") != snapshot.inventory.campaign_id
            or self.index.get("source_role") != "evaluator"
            or self.index.get("database_snapshot") != snapshot.source["database_snapshot"]
        ):
            raise _Unavailable("process_snapshot_identity_mismatch")
        self.incomplete = (
            self.index.get("status") != "complete"
            or bool(self.index.get("missing"))
            or "process_facts" not in snapshot.inventory.coverage
            or bool(snapshot.missing)
            or not PROCESS_REQUIRED_TABLES <= set(self.index.get("tables", {}))
        )

    def rows(self, table):
        name = RUNTIME + table
        if name not in self.cache:
            entry = self.index.get("tables", {}).get(name)
            if not entry or entry.get("status") != "available":
                raise _Unavailable("missing_source:" + name)
            ref = self.snapshot.artifacts.get("table:" + name)
            if (
                ref is None
                or entry.get("artifact_id") != ref.artifact_id
                or entry.get("sha256") != ref.sha256
            ):
                raise _Unavailable("process_table_reference_mismatch:" + name)
            rows = self.snapshot.table(name)
            if entry.get("row_count") != len(rows):
                raise _Unavailable("process_table_denominator_mismatch:" + name)
            self.cache[name] = rows
        return self.cache[name]

    def one(self, table, **keys):
        rows = [r for r in self.rows(table) if all(r.get(k) == v for k, v in keys.items())]
        if len(rows) != 1:
            raise _Unavailable("missing_or_ambiguous_relation:" + RUNTIME + table)
        return rows[0]

    def call(self, target, call_id, role, *, run=None, query=None):
        trace = self.calls.get((RUNTIME, call_id))
        if (
            trace is None
            or trace["run_id"] in self.snapshot.inventory.conformance_run_ids
            or (run is not None and trace["run_id"] != run)
            or (query is not None and trace.get("query_id") not in {None, query})
        ):
            raise _Unavailable("consumed_model_call_scope_unproven")
        target[call_id, role] = self.owners[RUNTIME, call_id]
        if (
            trace["status"] != "success"
            or trace.get("parent_model_call_id")
            or trace.get("retry_count", 0)
        ):
            raise _Unavailable("consumed_call_success_or_parent_dependencies_unproven")

    def baseline(self, answer, query, write_missing, write=None):
        write_missing.add("baseline_write_dependency_unproven")
        run, qid, snapshot = answer.run_id, answer.query_id, answer.snapshot_id
        manifest = self.one("run_manifests", run_id=run)
        if manifest["method"] != answer.method:
            raise _Unavailable("baseline_run_method_mismatch")
        self.one("run_queries", run_id=run, snapshot_id=snapshot, query_id=qid)
        trace = self.calls[RUNTIME, answer.answer_model_call_id]
        context = self.one(
            "context_assemblies", context_assembly_id=trace.get("context_assembly_id")
        )
        if (context["run_id"], context["query_id"], context["snapshot_id"], context["method"]) != (
            run,
            qid,
            snapshot,
            answer.method,
        ):
            raise _Unavailable("baseline_context_scope_mismatch")
        full_context = answer.method in {"full_context_captions", "full_context_raw_images"}
        if full_context:
            if context.get("retrieval_batch_id") is not None or any(
                r["run_id"] == run for r in self.rows("retrieval_batches")
            ):
                raise _Unavailable("full_context_has_unexpected_retrieval")
            # This is zero *model-call* write cost, not free CPU/storage/GPU time.
            if any(
                scope == RUNTIME and t["run_id"] == run and t.get("query_id") is None
                for (scope, _), t in self.calls.items()
            ):
                write_missing.add("full_context_unassigned_write_calls")
            else:
                write_missing.discard("baseline_write_dependency_unproven")
        else:
            batch = self.retrieval(context.get("retrieval_batch_id"), run, qid, snapshot, query)
            try:
                if write is None:
                    raise _Unavailable("index_cost_destination_missing")
                self.rag_index(batch.get("document_collection_id"), snapshot, answer.method, write)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                write_missing.add(
                    str(exc)
                    if isinstance(exc, (_Unavailable, DependencyUnproven))
                    else "index_cost_dependency_invalid"
                )
            else:
                write_missing.discard("baseline_write_dependency_unproven")
        accounted = {identity for identity, _ in query}
        if any(
            scope == RUNTIME
            and t["run_id"] == run
            and t.get("query_id") == qid
            and identity not in accounted
            for (scope, identity), t in self.calls.items()
        ):
            raise _Unavailable("baseline_query_has_unassigned_calls")

    def retrieval(self, batch_id, run, query_id, snapshot, target):
        batch = self.one("retrieval_batches", retrieval_batch_id=batch_id)
        if (batch["run_id"], batch["query_id"], batch["snapshot_id"], batch["status"]) != (
            run,
            query_id,
            snapshot,
            "success",
        ):
            raise _Unavailable("retrieval_batch_scope_or_status_mismatch")
        refs = [
            r
            for r in self.rows("retrieval_batch_query_embeddings")
            if r["retrieval_batch_id"] == batch_id and r["used_for_search"]
        ]
        if not refs:
            raise _Unavailable("retrieval_query_embedding_proof_missing")
        for ref in refs:
            if (ref["run_id"], ref["query_id"], ref["snapshot_id"]) != (run, query_id, snapshot):
                raise _Unavailable("retrieval_embedding_link_scope_mismatch")
            vector = self.one("embeddings", embedding_id=ref["embedding_id"])
            if vector["collection_id"] != ref["query_embedding_collection_id"]:
                raise _Unavailable("retrieval_embedding_collection_mismatch")
            collection = self.one("embedding_collections", collection_id=vector["collection_id"])
            if collection["snapshot_id"] != snapshot:
                raise _Unavailable("retrieval_embedding_snapshot_mismatch")
            self.call(
                target, vector.get("model_call_id"), "retrieval_query_embedding", query=query_id
            )
        if batch.get("reranker_model_call_id"):
            self.call(
                target,
                batch["reranker_model_call_id"],
                "retrieval_reranker",
                run=run,
                query=query_id,
            )
        return batch

    def rag_index(self, collection_id, snapshot_id, method, write):
        ref = self.snapshot.artifacts.get(RAG_COST_DEPENDENCIES_ARTIFACT)
        if ref is None:
            raise _Unavailable(
                "missing_source:complete_document_index_membership_and_producer_calls"
            )
        if ref.scope != "evaluation" or ref.kind != "performance":
            raise _Unavailable("index_dependency_artifact_scope_mismatch")
        manifest = self.snapshot.dependency_artifact(
            RAG_COST_DEPENDENCIES_ARTIFACT, RAG_COST_DEPENDENCIES_VERSION
        )
        if (
            manifest.get("schema_version") != RAG_COST_DEPENDENCIES_VERSION
            or manifest.get("campaign_id") != self.snapshot.inventory.campaign_id
            or manifest.get("selection_hash") != self.snapshot.inventory.selection_hash
            or manifest.get("source_role") != "evaluator"
            or manifest.get("database_snapshot") != self.snapshot.source["database_snapshot"]
        ):
            raise _Unavailable("index_dependency_snapshot_mismatch")
        entries = [e for e in manifest["collections"] if e["collection_id"] == collection_id]
        if len(entries) != 1:
            raise _Unavailable("consumed_index_dependency_missing_or_ambiguous")
        entry = entries[0]
        facts = entry["facts"]
        if entry["facts_hash"] != canonical_content_hash(facts):
            raise _Unavailable("index_dependency_membership_hash_mismatch")
        # Revalidate the producer's proof, rather than trusting its status flag.
        call_ids = validate_rag_index_dependencies(facts)
        collection, index_run = facts["collection"], facts["run"]
        if (
            entry["status"] != "complete"
            or entry["reasons"]
            or collection["collection_id"] != collection_id
            or collection["snapshot_id"] != snapshot_id
            or index_run["method"] != method + "_index"
        ):
            raise _Unavailable("consumed_index_dependency_scope_mismatch")
        traces = {c["model_call_id"]: c for c in facts["calls"]}
        for identity in call_ids:
            trace = self.calls.get((RUNTIME, identity))
            if trace is None or any(trace.get(k) != traces[identity].get(k) for k in CALL_FIELDS):
                raise _Unavailable("index_dependency_trace_mismatch")
            self.call(write, identity, "rag_document_index", run=index_run["run_id"])

    def match_owner(self, runtime):
        target = self.one(
            "query_executor_runs", query_executor_run_id=runtime.query_executor_run_id
        )
        source_id = runtime.route.query_executor_run_id
        if source_id == runtime.query_executor_run_id:
            return target
        source = self.one("query_executor_runs", query_executor_run_id=source_id)
        if (
            target.get("metadata", {}).get("source_match_run_id") != source_id
            or runtime.metadata.get("source_match_run_id") != source_id
            or source.get("schema_version") != "0.3.0"
            or any(
                source[key] != target[key]
                for key in (
                    "routing_projection_id",
                    "config_hash",
                    "snapshot_id",
                    "source_compiler_run_id",
                )
            )
        ):
            raise _Unavailable("source_match_cost_identity_mismatch")
        return source

    def match(self, runtime, definition, query):
        run = self.match_owner(runtime)
        owner = run["query_executor_run_id"]
        batch = self.one("qbank_match_batches", match_batch_id=runtime.route.match_batch_id)
        if (
            (
                batch["query_executor_run_id"],
                batch["query_id"],
                batch["active_round"],
                batch["question_budget"],
            )
            != (
                owner,
                runtime.query_id,
                definition.round_id,
                definition.question_budget,
            )
            or batch["source_compiler_run_id"] != run["source_compiler_run_id"]
            or batch["routing_projection_id"] != run["routing_projection_id"]
            or batch["snapshot_id"] != run["snapshot_id"]
            or run["snapshot_id"] != self.snapshot.facts.manifest.snapshot_ids[definition.dataset]
        ):
            raise _Unavailable("match_cost_scope_mismatch")
        candidates = [
            r
            for r in self.rows("qbank_match_fused_candidates")
            if r["match_batch_id"] == batch["match_batch_id"]
        ]
        if len(candidates) != batch["fused_count"]:
            raise _Unavailable("match_candidate_denominator_mismatch")
        base_top20 = {
            c["qbank_id"]
            for route_class in {c["route_class"] for c in candidates}
            for c in sorted(
                (c for c in candidates if c["route_class"] == route_class),
                key=lambda c: (-c["raw_rrf"], c["qbank_id"]),
            )[:20]
        }
        for candidate in candidates:
            # V3 persists the exact applied assessment before reranking. Reading
            # all assessments in the run would charge other b/r cells as active.
            assessment_id = candidate.get("metadata", {}).get("reranker_assessment_id")
            if definition.treatment.value == "no_reranker":
                if assessment_id:
                    raise _Unavailable("no_reranker_has_consumed_assessment")
                continue
            if assessment_id:
                if candidate["qbank_id"] not in base_top20:
                    raise _Unavailable("reranker_assessment_outside_cell_base_top20")
                assessment = self.one(
                    "qbank_reranker_assessments", reranker_assessment_id=assessment_id
                )
                if (
                    assessment["query_executor_run_id"],
                    assessment["query_id"],
                    assessment["qbank_id"],
                ) != (owner, runtime.query_id, candidate["qbank_id"]):
                    raise _Unavailable("reranker_cost_scope_mismatch")
                self.call(
                    query,
                    assessment["model_call_id"],
                    "routing_reranker",
                    run=owner,
                    query=runtime.query_id,
                )
            elif candidate["qbank_id"] in base_top20:
                raise _Unavailable("routing_reranker_call_lineage_missing")
        return run, batch

    def match_embeddings(self, runtime, query):
        run = self.match_owner(runtime)
        owner = run["query_executor_run_id"]
        vectors = [
            r
            for r in self.rows("query_embeddings")
            if (r["query_executor_run_id"], r["query_id"]) == (owner, runtime.query_id)
        ]
        if {r["channel"] for r in vectors} != {"text", "multimodal"} or len(vectors) != 2:
            raise _Unavailable("match_embedding_channel_coverage_incomplete")
        for vector in vectors:
            batch = self.one(
                "query_embedding_batches",
                query_embedding_batch_id=vector["query_embedding_batch_id"],
            )
            if (
                batch["query_executor_run_id"] != owner
                or batch["snapshot_id"] != run["snapshot_id"]
                or vector["snapshot_id"] != batch["snapshot_id"]
                or vector["channel"] != batch["channel"]
                or runtime.query_id not in batch["query_ids"]
                or batch["committed_count"] != batch["expected_count"]
                or batch["committed_count"] != len(batch["query_ids"])
                or batch.get("metadata", {}).get("coverage_projection")
            ):
                raise _Unavailable("runtime_match_embedding_use_unproven")
            embedding = self.one("embeddings", embedding_id=vector["embedding_id"])
            collection = self.one("embedding_collections", collection_id=vector["collection_id"])
            if (
                embedding["collection_id"] != vector["collection_id"]
                or embedding.get("model_call_id") != batch["model_call_id"]
                or collection["snapshot_id"] != run["snapshot_id"]
            ):
                raise _Unavailable("runtime_match_embedding_producer_mismatch")
            self.call(
                query,
                batch.get("model_call_id"),
                "routing_query_embedding",
                query=runtime.query_id,
            )

    def compiler(self, runtime, definition, run, batch, write):
        from pmmc.experiments.step8_pmmc_cost_dependencies import bound_write_dependencies

        compiler = run["source_compiler_run_id"]
        self.one("compiler_runs", run_id=compiler, snapshot_id=run["snapshot_id"])
        try:
            complete = bound_write_dependencies(
                self.snapshot,
                {key: value for (scope, key), value in self.calls.items() if scope == RUNTIME},
                compiler,
                definition.round_id,
            )
        except (ValueError, KeyError, TypeError, OSError, AttributeError, IndexError):
            pass
        else:
            for identity in complete:
                write[identity, "whole_b8_bank_round_dependencies"] = self.owners[RUNTIME, identity]
            return True
        # The sealed imported graph includes rejected candidates, verifier calls
        # and revisions. Charge its whole measured calls, never b/8 token slices.
        for alias in self.snapshot.call_aliases:
            trace = self.calls[RUNTIME, alias]
            if trace["run_id"] == compiler:
                role = (
                    "r0_prefix_imported_graph"
                    if "r0_prefix_import" in (trace.get("metadata") or {})
                    else "questioner_imported_graph"
                )
                write[alias, role] = self.owners[RUNTIME, alias]
                if trace.get("retry_count"):
                    raise _Unavailable("imported_provider_retries_not_individually_measured")
        # These are producer lower bounds for the consumed program, not the cost
        # of creating every accepted/rejected candidate in the deployed bank.
        if runtime.binding is None:
            return False
        binding = self.one(
            "query_program_bindings",
            query_program_binding_id=runtime.binding.query_program_binding_id,
        )
        if (
            binding["query_executor_run_id"] != run["query_executor_run_id"]
            or binding["query_id"] != runtime.query_id
            or binding["plan_id"] != runtime.binding.plan_id
            or binding["qbank_id"] != runtime.binding.qbank_id
        ):
            raise _Unavailable("compiler_binding_scope_mismatch")
        program = self.one("memory_programs", plan_id=binding["plan_id"])
        candidate = self.one("qbank_questions", qbank_id=binding["qbank_id"])
        if (
            program["run_id"] != compiler
            or candidate["run_id"] != compiler
            or program["qbank_id"] != candidate["qbank_id"]
            or program["round_id"] > definition.round_id
            or candidate["generation_rank"] > definition.question_budget
            or candidate["conversation_id"] != batch["conversation_id"]
        ):
            raise _Unavailable("compiler_producer_scope_mismatch")
        self.call(
            write, candidate.get("producer_model_call_id"), "questioner_producer", run=compiler
        )
        self.planner_dependencies(program, write)
        if definition.round_id > 0:
            projection = self.one(
                "qbank_round_projection_manifests", run_id=compiler, round_id=definition.round_id
            )
            member = self.one(
                "qbank_round_projection_members",
                projection_id=projection["projection_id"],
                qbank_id=candidate["qbank_id"],
            )
            if member["effective_plan_id"] != program["plan_id"]:
                raise _Unavailable("compiler_effective_plan_mismatch")
            if member.get("effective_feedback_id"):
                feedback = self.one("doubter_feedback", feedback_id=member["effective_feedback_id"])
                if feedback["run_id"] != compiler or feedback["qbank_id"] != candidate["qbank_id"]:
                    raise _Unavailable("doubter_producer_scope_mismatch")
                self.call(
                    write, feedback.get("producer_model_call_id"), "doubter_producer", run=compiler
                )
        return False

    def planner_dependencies(self, program, write):
        # P1.5 server patches have no new Planner call. Their source P/D calls
        # remain dependencies, even when the target projection has newer feedback.
        seen = set()
        while program.get("producer_model_call_id") is None:
            if program["plan_id"] in seen:
                raise _Unavailable("planner_revision_cost_cycle")
            seen.add(program["plan_id"])
            if not 0 < program["round_id"] <= 3:
                raise _Unavailable("planner_root_producer_missing")
            lineage = self.one("planner_revision_lineage", revised_plan_id=program["plan_id"])
            if (lineage["run_id"], lineage["qbank_id"], lineage["target_round_id"]) != (
                program["run_id"],
                program["qbank_id"],
                program["round_id"],
            ) or lineage["source_plan_id"] in seen:
                raise _Unavailable("planner_revision_cost_scope_mismatch")
            source = self.one("memory_programs", plan_id=lineage["source_plan_id"])
            feedback = self.one("doubter_feedback", feedback_id=lineage["source_feedback_id"])
            if (
                (source["run_id"], source["qbank_id"], source["snapshot_id"], source["round_id"])
                != (
                    program["run_id"],
                    program["qbank_id"],
                    program["snapshot_id"],
                    program["round_id"] - 1,
                )
                or (
                    feedback["run_id"],
                    feedback["qbank_id"],
                    feedback["plan_id"],
                    feedback["round_id"],
                )
                != (source["run_id"], source["qbank_id"], source["plan_id"], source["round_id"])
                or feedback.get("producer_model_call_id") != lineage["producer_model_call_id"]
            ):
                raise _Unavailable("planner_revision_source_dependency_mismatch")
            self.call(
                write, lineage["producer_model_call_id"], "doubter_producer", run=source["run_id"]
            )
            program = source
        self.call(
            write, program["producer_model_call_id"], "planner_producer", run=program["run_id"]
        )

    def execution(self, runtime, query):
        if runtime.execution_id is None:
            if runtime.binding is not None:
                raise _Unavailable("bound_execution_missing")
            return
        execution = self.one("program_executions", execution_id=runtime.execution_id)
        run = self.match_owner(runtime)
        if runtime.binding is None or (
            execution["run_id"],
            execution["query_id"],
            execution["plan_id"],
            execution["purpose"],
        ) != (
            run["query_executor_run_id"],
            runtime.query_id,
            runtime.binding.plan_id,
            "query_time",
        ):
            raise _Unavailable("execution_cost_scope_mismatch")
        # Only explicit execution IDs establish per-cell use. plan_id by itself
        # cannot separate counterfactual executions of the same program.
        for (scope, identity), trace in self.calls.items():
            if (
                scope == RUNTIME
                and trace.get("metadata", {}).get("execution_id") == runtime.execution_id
            ):
                self.call(
                    query,
                    identity,
                    "program_execution",
                    run=run["query_executor_run_id"],
                    query=runtime.query_id,
                )
        from pmmc.experiments.step8_pmmc_cost_dependencies import (
            bound_write_dependencies,
            pmmc_entry,
            validate_execution_dependencies,
        )

        proof = pmmc_entry(self.snapshot, "executions", "execution_id", runtime.execution_id)
        if any(
            proof["execution"].get(k) != execution.get(k)
            for k in (
                "execution_id",
                "run_id",
                "source_compiler_run_id",
                "snapshot_id",
                "plan_id",
                "query_id",
                "purpose",
                "status",
                "bindings_hash",
            )
        ):
            raise _Unavailable("execution_dependency_trace_mismatch")
        dependencies = validate_execution_dependencies(proof)
        write = bound_write_dependencies(
            self.snapshot,
            {key: value for (scope, key), value in self.calls.items() if scope == RUNTIME},
            run["source_compiler_run_id"],
            runtime.binding.round_id,
        )
        if not dependencies["write"] <= write:
            raise _Unavailable("execution_cache_producers_not_in_write_graph")
        for identity in dependencies["query"]:
            self.call(
                query,
                identity,
                "program_execution",
                run=run["query_executor_run_id"],
                query=runtime.query_id,
            )


def _deployment_costs(snapshot, measured, owners, calls, bindings):
    facts = snapshot.facts
    cells, amortized, all_uses = [], [], []
    write_uses = set()
    try:
        process = _Deployment(snapshot, owners, calls)
        process_error = None
    except (ValueError, KeyError, TypeError, OSError, AttributeError, OverflowError) as exc:
        process = None
        process_error = (
            str(exc) if isinstance(exc, _Unavailable) else "process_inventory_unavailable"
        )

    def totals(ids, missing):
        value = _safe(
            asdict(summarize_costs(tuple(measured[i] for i in sorted(ids)), ()).actual_research)
        )
        if missing:
            value["cost_usd"] = None
            for resource in value["resources"]:
                resource["total"] = None
        return value

    def attempt(missing, action, *args):
        try:
            return action(*args)
        except (ValueError, KeyError, TypeError, OSError, AttributeError, OverflowError) as exc:
            missing.add(str(exc) if isinstance(exc, _Unavailable) else "path_relation_invalid")
            return None

    for definition in sorted(facts.cells, key=lambda c: c.cell_id):
        grouped: dict[str, set[str]] = {"write": set(), "query": set()}
        missing: dict[str, set[str]] = {"write": set(), "query": set()}
        method = MethodVariant(definition.method, definition.cell_id)
        for query_id in facts.selection.query_ids[definition.dataset]:
            bound = bindings.get((definition.cell_id, query_id))
            if bound is None:
                for missing_values in missing.values():
                    missing_values.add("query_cell_binding_unproven")
                continue
            write: dict[tuple[str, str], str] = {}
            query: dict[tuple[str, str], str] = {}
            for answer in bound["answers"]:
                for call in answer.calls:
                    role = (
                        "native_mm_fallback:"
                        if definition.method == "pmmc" and answer.method != "pmmc"
                        else "answer:"
                    )
                    query[call.model_call_id, role + call.role] = owners[
                        RUNTIME, call.model_call_id
                    ]

            if process is None:
                for missing_values in missing.values():
                    missing_values.add(process_error or "process_inventory_unavailable")
            else:
                if process.incomplete:
                    for missing_values in missing.values():
                        missing_values.add("process_or_cost_inventory_incomplete")
                if definition.method == "pmmc":
                    runtime = bound["runtime"]
                    matched = attempt(missing["query"], process.match, runtime, definition, query)
                    attempt(missing["query"], process.match_embeddings, runtime, query)
                    attempt(missing["query"], process.execution, runtime, query)
                    if matched:
                        write_closed = attempt(
                            missing["write"],
                            process.compiler,
                            runtime,
                            definition,
                            matched[0],
                            matched[1],
                            write,
                        )
                        if write_closed is not True:
                            missing["write"].add(
                                "missing_source:complete_compiler_write_dependency_graph"
                            )
                    else:
                        missing["write"].add("compiler_owner_not_bound")
                for answer in bound["answers"]:
                    if answer.method != "pmmc":
                        answer_write_missing: set[str] = set()
                        attempt(
                            missing["query"],
                            process.baseline,
                            answer,
                            query,
                            answer_write_missing,
                            write,
                        )
                        missing["write"].update(answer_write_missing)
            for part, selected in (("write", write), ("query", query)):
                grouped[part].update(selected.values())
                for (call_id, role), physical in selected.items():
                    for scope in ("active_path", "logical_ablation"):
                        if part == "write":
                            use_key = definition.cell_id, physical, scope
                            if use_key in write_uses:
                                continue
                            write_uses.add(use_key)
                        all_uses.append(
                            CostUse(
                                canonical_content_hash(
                                    [definition.cell_id, query_id, part, call_id, role, scope]
                                ),
                                physical,
                                method,
                                part + ":" + role,
                                scope,
                            )
                        )
        values = {
            part: {
                "status": "incomplete" if missing[part] else "complete",
                "totals": totals(grouped[part], missing[part]),
                "reasons": sorted(missing[part]),
            }
            for part in grouped
        }
        count = len(facts.selection.query_ids[definition.dataset])
        cells.append(
            {
                "cell_id": definition.cell_id,
                **definition.model_dump(mode="json"),
                "selected_query_count": count,
                **values,
            }
        )
        write_usd = values["write"]["totals"]["cost_usd"]
        query_usd = values["query"]["totals"]["cost_usd"]
        c_write = Decimal(write_usd) if write_usd is not None else None
        c_query = Decimal(query_usd) / count if query_usd is not None and count else None
        reason = None
        if missing["write"] or missing["query"]:
            reason = "write_or_query_dependency_proof_incomplete"
        elif c_write is None or c_query is None:
            reason = "unknown_model_call_price"
        for n in AMORTIZATION_N:
            amortized.append(
                {
                    "method": definition.method,
                    "cell_id": definition.cell_id,
                    "n": n,
                    "c_write_usd": _safe(c_write),
                    "c_query_usd": _safe(c_query),
                    "cost_usd": _safe(c_write / n + c_query)
                    if c_write is not None and c_query is not None
                    else None,
                    "reason": reason,
                    "lifecycle_scope": "selected_cohort_memory_bank_union",
                    "future_query_cost_assumption": "same_observed_query_mix",
                }
            )
    summary = summarize_costs(tuple(measured.values()), tuple(all_uses))
    cell_index = {c["cell_id"]: c for c in cells}
    method_costs = []
    for item in summary.method_costs:
        value = _safe(asdict(item))
        cell = cell_index[item.method.variant]
        reasons = sorted(set(cell["write"]["reasons"]) | set(cell["query"]["reasons"]))
        value.update(status="incomplete" if reasons else "complete", reasons=reasons)
        if reasons:
            value["totals"]["cost_usd"] = None
            for resource in value["totals"]["resources"]:
                resource["total"] = None
        method_costs.append(value)
    return {
        "status": "complete"
        if all(c[p]["status"] == "complete" for c in cells for p in ("write", "query"))
        else "incomplete",
        "scope": "audited_model_calls_only_not_infrastructure_tco",
        "totals": None,
        "aggregation": "alternative_cells_must_not_be_summed",
        "write_lifecycle": "union_of_dependencies_for_selected_cohort_memory_banks",
        "memory_instance_grouping": {
            "status": "not_evaluated",
            "reason": "per_memory_instance_producer_membership_unproven",
        },
        "query_cost_basis": "deduplicated_active_query_call_cost_divided_by_selected_query_count",
        "batch_policy": "whole_measured_call_no_fractional_token_allocation",
        "cells": cells,
        "method_costs": [c for c in method_costs if c["scope"] == "active_path"],
        "logical_ablation_costs": [c for c in method_costs if c["scope"] == "logical_ablation"],
        "method_costs_semantics": "proven_dependency_subtotals_check_each_cell_completeness",
        "break_even": _break_even_comparisons(cells),
    }, amortized


def _break_even_comparisons(cells: list[dict]) -> list[dict]:
    """Project paired main methods only when both complete cost closures are priced.

    Shared indices have write cost too: use incremental write cost, not just
    PMMC C_write. This is a cohort memory-bank-union projection, not a claimed
    per-conversation lifecycle estimate or infrastructure TCO.
    """
    main = [
        c
        for c in cells
        if (c["treatment"], c["question_budget"], c["round_id"], c["source_variant"])
        == ("main", 8, 2, "active")
    ]
    result = []
    for ours in main:
        if ours["method"] != "pmmc":
            continue
        for baseline in main:
            if baseline["method"] == "pmmc" or (ours["dataset"], ours["answer_model"]) != (
                baseline["dataset"],
                baseline["answer_model"],
            ):
                continue
            reason = None
            costs: list[Decimal] = []
            count = ours["selected_query_count"]
            if not count or count != baseline["selected_query_count"]:
                reason = "paired_selection_denominator_incomplete"
            elif any(
                c[p]["status"] != "complete" for c in (ours, baseline) for p in ("write", "query")
            ):
                reason = "write_or_query_dependency_proof_incomplete"
            else:
                for c in (ours, baseline):
                    for part in ("write", "query"):
                        cost = _money(c[part]["totals"]["cost_usd"])
                        if cost is None:
                            reason = "unknown_model_call_price"
                        else:
                            costs.append(cost / count if part == "query" else cost)
            incremental_write = savings = boundary = None
            if reason is None:
                ours_write, ours_query, baseline_write, baseline_query = costs
                incremental_write = ours_write - baseline_write
                savings = baseline_query - ours_query
                estimate = break_even_queries(
                    max(Decimal(0), incremental_write), ours_query, baseline_query
                )
                boundary, reason = estimate.queries, estimate.reason
            result.append(
                {
                    "dataset": ours["dataset"],
                    "answer_model": ours["answer_model"],
                    "pmmc_cell_id": ours["cell_id"],
                    "baseline_cell_id": baseline["cell_id"],
                    "baseline_method": baseline["method"],
                    "selected_query_count": count,
                    "status": "complete"
                    if reason in {None, "no_positive_query_saving"}
                    else "incomplete",
                    "incremental_write_usd": _safe(incremental_write),
                    "per_query_saving_usd": _safe(savings),
                    "break_even_queries": _safe(boundary),
                    "minimum_queries_for_nonhigher_cost": max(
                        1, int(boundary.to_integral_value(rounding=ROUND_CEILING))
                    )
                    if boundary is not None
                    else None,
                    "reason": reason,
                    "lifecycle_scope": "selected_cohort_memory_bank_union",
                    "future_query_cost_assumption": "same_observed_query_mix",
                }
            )
    return result


def summarize_campaign_costs(facts: CampaignFacts) -> dict:
    """Return safe aggregates; unavailable proof never becomes zero-cost success.

    ``actual_research`` covers all campaign attempts (not just the chosen cohort).
    Projection denominators use the selected cohort's frozen cell universe. This
    function adds read bytes/hashes to CampaignFacts for the enclosing report's
    provenance, but never writes files or changes campaign facts on disk.
    """
    try:
        snapshot = _Snapshot(facts)
        tables = {name: snapshot.table(name) for name in _CORE_TABLES}
        measured, owners, calls, categories, reasons = _normalize(snapshot, tables)
    except FileNotFoundError:
        return _empty("inventory_or_cas_unavailable")
    except (ValueError, KeyError, TypeError, OSError, AttributeError, OverflowError) as exc:
        return _empty(
            str(exc)
            if isinstance(exc, (_Unavailable, DependencyUnproven))
            else "inventory_cost_proof_invalid"
        )
    uses: tuple[CostUse, ...] = ()
    bindings = {}
    try:
        tables.update({name: snapshot.table(name) for name in _USE_TABLES})
        uses, projection, bindings = _project_uses(snapshot, tables, owners, calls)
    except (ValueError, KeyError, TypeError, OSError):
        projection = {
            "status": "incomplete",
            "reasons": {"cell_use_proof_invalid_or_unavailable": 1},
        }
    report = summarize_costs(tuple(measured.values()), uses)
    research = summarize_costs(
        tuple(
            value
            for identity, value in measured.items()
            if identity not in snapshot.dependency_physical_ids
        ),
        (),
    ).actual_research
    totals = _safe(asdict(research))
    complete = not snapshot.missing
    # Partial observations retain known subtotals but must not masquerade as the
    # complete campaign total, including when the observed subset is empty.
    if not complete:
        totals["cost_usd"] = None
        for resource in totals["resources"]:
            resource["total"] = None
    active_path, amortized = _deployment_costs(snapshot, measured, owners, calls, bindings)
    result = {
        "schema_version": "step8-campaign-costs-v1",
        "status": "complete" if complete and active_path["status"] == "complete" else "incomplete",
        "actual_research": {
            "status": "complete" if complete else "incomplete",
            "pricing_status": "complete"
            if complete and not research.unknown_usd_count
            else "incomplete",
            "scope": "all_campaign_attempts",
            "totals": totals,
            "reasons": sorted(snapshot.missing),
            "unknown_price_reasons": dict(sorted(reasons.items())),
            "price_semantics": "verified_rate_card_settlement_not_invoice",
        },
        "research_breakdown": {
            name: _safe(
                asdict(summarize_costs(tuple(measured[k] for k in sorted(ids)), ()).actual_research)
            )
            for name, ids in sorted(categories.items())
        },
        "breakdown_semantics": "overlapping_categories_do_not_sum",
        "reused_dependencies": {
            "scope": "external_preexisting_calls_not_current_campaign_spend",
            "totals": _safe(
                asdict(
                    summarize_costs(
                        tuple(measured[k] for k in sorted(snapshot.dependency_physical_ids)),
                        (),
                    ).actual_research
                )
            ),
        },
        "answer_role_projection": {
            **projection,
            "scope": "answer_and_grounding_only",
            "method_costs": _safe([asdict(item) for item in report.method_costs]),
            "role_use_counts": dict(sorted(Counter(u.role for u in uses).items())),
            "logical_use_count": sum(u.scope == "logical_ablation" for u in uses),
            "active_use_count": sum(u.scope == "active_path" for u in uses),
        },
        "active_path": active_path,
        "amortized": amortized,
        "amortization_n": list(AMORTIZATION_N),
        "latency": {"value": None, "reason": "no_end_to_end_latency_inference_from_cost_facts"},
        "input_hashes": dict(sorted(snapshot.hashes.items())),
        "source_snapshot_sha256": snapshot.inventory.source_snapshot_sha256,
    }
    return result
