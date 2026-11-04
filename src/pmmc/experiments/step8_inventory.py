"""Evaluator-only, read-only PostgreSQL snapshot producer for Step 8 audit export.

The DB is authoritative for enumeration; component receipts are only run-identity
hints, never evidence that a payload exists. All failed calls and all reranker
calls need their actual private input/output-or-error objects. Success review
samples are the deterministic first call per (scope, dataset, method, stage,
model alias) stratum. Full call, cost, patch, answer, cell and attempt rows are
retained independently of sampling.
Query-free conformance runs are enumerated separately with all private artifacts;
their run IDs are never inserted into the worker-attempt or benchmark denominator.

No runtime/admin credentials, models, migrations, or database writes are used.
Missing tables/permissions/lineage/payloads remove the affected coverage family.
Details live in the private provenance artifact and audit-inventory-status.json.
The latest inventory is a replaceable pointer to immutable hash-named snapshots.
Threshold v2 facts are produced in the same pass: rows contain only proven saved
outcomes, coverage includes every declared PMMC query/budget/round/grid point.
Each selection has private/threshold-facts/<selection_hash>.json and immutable
<selection_hash>/<sha256>.json snapshots; threshold-facts.json is the latest pointer.
Changing relevance never reuses a frozen-context answer. Threshold coverage is
separate from audit-artifact completeness and never grants scientific approval.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter
from sqlalchemy import Table, any_, false, or_, select, text, tuple_
from sqlalchemy.exc import SQLAlchemyError

from pmmc.benchmarks.ids import stable_hash
from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_analysis import (
    _threshold_axes,
    read_campaign_file,
    selected_policy_axis,
    validate_selection_chain,
)
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, json_bytes, write_json
from pmmc.experiments.step8_audit import (
    PROCESS_REQUIRED_TABLES,
    CampaignAuditInventory,
    PrivateAuditArtifact,
    PrivateAuditRecord,
    _read_failure_capsule,
    _safe_log,
    audit_table_kind,
)
from pmmc.experiments.step8_conformance import CONFORMANCE_VERSION
from pmmc.experiments.step8_jobs import BenchmarkStageJob
from pmmc.infrastructure.database import baseline_models as blm
from pmmc.infrastructure.database import compiler_models as cm
from pmmc.infrastructure.database import doubter_models as dm
from pmmc.infrastructure.database import evaluation_models as em
from pmmc.infrastructure.database import experiment_models as xm
from pmmc.infrastructure.database import models as bm
from pmmc.infrastructure.database import query_executor_models as qm
from pmmc.infrastructure.database import questioner_models as questioner
from pmmc.infrastructure.database import step8_models as sm
from pmmc.infrastructure.database import step8_provider_control as pc
from pmmc.infrastructure.database.constants import EVALUATION_ROLE, EVALUATION_SCHEMA, DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.errors import DatabaseConfigurationError
from pmmc.infrastructure.object_store import create_evaluator_object_stores
from pmmc.infrastructure.settings import Settings
from pmmc.schemas import RunManifest
from pmmc.schemas.answer_output import TypedAnswerResult
from pmmc.schemas.query_executor import runtime_fact_payload
from pmmc.schemas.step8 import (
    BenchmarkCampaignManifest,
    ExperimentCell,
    ScientificProtocol,
    SelectionExpansionManifest,
)

_FAMILIES = frozenset(
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
_JSON = TypeAdapter(Any)
_TERMINAL = {"completed", "failed", "interrupted", "budget_exhausted"}


def _json(value: Any) -> Any:
    def encode_unknown(item):
        # pgvector may return ndarray; the inventory need not import a numeric runtime.
        if type(item).__module__.startswith("numpy") and hasattr(item, "tolist"):
            return item.tolist()
        raise TypeError("unsupported database process value")

    return _JSON.dump_python(value, mode="json", fallback=encode_unknown)


def _table(model: Any) -> Table:
    return cast(Table, model.__table__)


def _selection(directory: Path, manifest: BenchmarkCampaignManifest) -> SelectionExpansionManifest:
    selections = {}
    for path in sorted((directory / "manifests/selections").glob("*.json")):
        value = SelectionExpansionManifest.model_validate_json(
            read_campaign_file(directory, str(path.relative_to(directory)))
        )
        if value.content_hash != path.stem or (
            value.campaign_id,
            value.phase,
            value.partition_hash,
        ) != (manifest.campaign_id, manifest.phase, manifest.protocol.partition_hash):
            raise ValueError("inventory selection identity mismatch")
        selections[value.content_hash] = value
    return selections[validate_selection_chain(selections)]


class _Snapshot:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.issues: dict[str, set[str]] = defaultdict(set)
        self.identity: dict[str, Any] = {}
        self.runs: set[str] = set()
        self.conformance: set[str] = set()
        self.evaluations: set[str] = set()
        self.selected: set[tuple[str, str]] = set()
        self.process_tables: dict[str, Any] = {}
        self.process_scope: dict[str, Any] = {}
        self.query_scopes: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self.dependency_runs: set[str] = set()
        self.dependency_calls: set[str] = set()
        self.adopted_calls: set[str] = set()
        self.embedding_calls: set[str] = set()
        self.cost_dependencies: dict[str, dict[str, Any]] = {}

    def missing(self, reason: str, families=_FAMILIES) -> None:
        for family in families:
            self.issues[family].add(reason)

    async def read(self, session, model, condition, families=_FAMILIES):
        table = model.__table__
        key = table.fullname
        statement = select(table).where(condition).order_by(*table.primary_key.columns)
        # A savepoint lets a denied/missing family fail without aborting the snapshot.
        try:
            async with session.begin_nested():
                result = await session.execute(
                    statement.execution_options(step8_inventory_table=key)
                )
                rows = [_json(dict(row)) for row in result.mappings().all()]
        except (SQLAlchemyError, ValueError, TypeError):
            self.tables.pop(key, None)
            self.missing("unavailable_table:" + key, families)
            return []
        self.tables[key] = rows
        return rows

    def rows(self, model):
        return self.tables.get(model.__table__.fullname, [])


def _in(column, values):
    return column.in_(sorted(values)) if values else false()


async def _read_snapshot(session, snapshot: _Snapshot, manifest, selection) -> None:
    read = snapshot.read
    campaign_id = manifest.campaign_id
    campaigns = await read(
        session, sm.BenchmarkCampaignModel, sm.BenchmarkCampaignModel.campaign_id == campaign_id
    )
    if (
        len(campaigns) != 1
        or campaigns[0]["protocol_hash"] != manifest.protocol.content_hash
        or campaigns[0]["payload"] != manifest.model_dump(mode="json")
    ):
        snapshot.missing("database_campaign_identity_mismatch")
        return
    selections = await read(
        session, sm.BenchmarkSelectionModel, sm.BenchmarkSelectionModel.campaign_id == campaign_id
    )
    matching = [s for s in selections if s["selection_hash"] == selection.content_hash]
    if len(matching) != 1 or matching[0]["payload"] != selection.model_dump(mode="json"):
        snapshot.missing("database_selection_identity_mismatch")
        return
    for row in selections:
        chosen = SelectionExpansionManifest.model_validate(row["payload"])
        if chosen.campaign_id != campaign_id or chosen.content_hash != row["selection_hash"]:
            raise ValueError("database selection content mismatch")
        snapshot.selected.update(
            (manifest.snapshot_ids[d], qid) for d, qids in chosen.query_ids.items() for qid in qids
        )
    jobs = await read(
        session, sm.BenchmarkStageJobModel, sm.BenchmarkStageJobModel.campaign_id == campaign_id
    )
    await read(
        session,
        sm.BenchmarkExecutionAttemptModel,
        sm.BenchmarkExecutionAttemptModel.campaign_id == campaign_id,
        {"attempts", "performance", "failures"},
    )
    await read(
        session,
        sm.BenchmarkCellAssignmentModel,
        sm.BenchmarkCellAssignmentModel.campaign_id == campaign_id,
        {"variant_cell_reuse"},
    )
    await read(
        session,
        sm.ExperimentCellModel,
        sm.ExperimentCellModel.protocol_hash == manifest.protocol.content_hash,
        {"variant_cell_reuse"},
    )
    receipts = await read(
        session,
        sm.BenchmarkComponentReceiptModel,
        _in(sm.BenchmarkComponentReceiptModel.job_id, {j["job_id"] for j in jobs}),
    )
    approvals = await read(
        session,
        sm.BenchmarkBudgetApprovalModel,
        sm.BenchmarkBudgetApprovalModel.campaign_id == campaign_id,
        {"cost_facts"},
    )
    reservations = await read(
        session,
        sm.BenchmarkSpendReservationModel,
        _in(sm.BenchmarkSpendReservationModel.approval_id, {a["approval_id"] for a in approvals}),
        {"cost_facts", "failures"},
    )
    admissions = await read(
        session,
        pc.ProviderRateAdmissionModel,
        _in(pc.ProviderRateAdmissionModel.request_hash, {r["request_hash"] for r in reservations}),
        {"cost_facts", "performance"},
    )
    await read(
        session,
        pc.ProviderRateScopeModel,
        _in(pc.ProviderRateScopeModel.scope_hash, {r["scope_hash"] for r in admissions}),
        {"cost_facts", "performance"},
    )
    await read(
        session,
        pc.ProviderReconciliationModel,
        _in(pc.ProviderReconciliationModel.request_id, {r["request_id"] for r in reservations}),
        {"cost_facts", "failures"},
    )

    compiler_ids: set[str] = set()
    query_ids: set[str] = set()
    source_matches: dict[str, str] = {}
    hints: set[str] = set()
    from pmmc.experiments.step8_backend import compiler_instance_id

    for row in jobs:
        job = BenchmarkStageJob.model_validate(row["payload"])
        if job.job_id != row["job_id"] or job.campaign_id != campaign_id:
            raise ValueError("database stage job identity mismatch")
        scope = job.parameters
        if scope.get("snapshot_id") not in set(manifest.snapshot_ids.values()):
            raise ValueError("database job outside campaign snapshots")
        if job.operation in {
            "units",
            "compiler",
            "query_match",
            "query_execute",
            "query_answer",
            "query_grounding",
            "evaluate",
        }:
            try:
                compiler_ids.add(compiler_instance_id(manifest, scope))
            except KeyError:
                snapshot.missing("compiler_identity_inputs_unavailable")
        if job.operation in {
            "query_match",
            "query_execute",
            "query_answer",
            "query_grounding",
            "evaluate",
        }:

            def query_identity(parameters):
                return (
                    "step8-query-"
                    + canonical_content_hash(
                        {"protocol": manifest.protocol.content_hash, "scope": parameters}
                    )[:24]
                )

            query_run = query_identity(scope)
            query_ids.add(query_run)
            snapshot.query_scopes[query_run].add((scope["snapshot_id"], scope["query_id"]))
            source_matches[query_run] = query_identity(
                {**scope, "answer_model": manifest.protocol.answer_backbones[0]}
            )
    for row in receipts:
        output = row["payload"].get("output", {})
        hints.update(
            str(output[k])
            for k in ("run_id", "compiler_run_id", "query_executor_run_id")
            if output.get(k)
        )

    # Failed baseline/index calls may have no component receipt or child-run link.
    # Their persisted campaign lineage is an independent root for enumeration.
    campaign_calls = []
    for call_model in (xm.RuntimeModelCallModel, xm.EvaluationModelCallModel):
        campaign_calls.extend(
            await read(
                session,
                call_model,
                or_(
                    call_model.metadata_json["step8_campaign_id"].astext == campaign_id,
                    call_model.metadata_json["campaign_id"].astext == campaign_id,
                ),
            )
        )
    hints.update(row["run_id"] for row in campaign_calls)
    campaign_evaluations = {
        row["evaluation_run_id"] for row in campaign_calls if row.get("evaluation_run_id")
    }

    suite = em.BaselineSuiteModel
    suites = await read(
        session,
        suite,
        _in(suite.snapshot_id, set(manifest.snapshot_ids.values()))
        & (suite.metadata_json["method_protocol_hash"].astext == manifest.protocol.content_hash)
        & (suite.metadata_json["partition_id"].astext == manifest.partition_id),
    )
    suite_queries = await read(
        session,
        em.BaselineSuiteQueryModel,
        _in(em.BaselineSuiteQueryModel.suite_id, {r["suite_id"] for r in suites}),
    )
    by_suite: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in suite_queries:
        by_suite[row["suite_id"]].add((row["snapshot_id"], row["query_id"]))
    owned_suites = {
        r["suite_id"]
        for r in suites
        if by_suite[r["suite_id"]] and by_suite[r["suite_id"]] <= snapshot.selected
    }
    snapshot.tables[_table(em.BaselineSuiteModel).fullname] = [
        r for r in suites if r["suite_id"] in owned_suites
    ]
    snapshot.tables[_table(em.BaselineSuiteQueryModel).fullname] = [
        r for r in suite_queries if r["suite_id"] in owned_suites
    ]
    links = await read(
        session, em.BaselineSuiteRunModel, _in(em.BaselineSuiteRunModel.suite_id, owned_suites)
    )
    baseline_ids = {r["run_id"] for r in links}
    for row in suites:
        if row["suite_id"] in owned_suites:
            baseline_ids.update(
                "baseline-run-" + stable_hash([row["suite_id"], method, "primary"])[:24]
                for method in row["methods"]
            )
    query_runs = await read(
        session,
        qm.QueryExecutorRunModel,
        _in(
            qm.QueryExecutorRunModel.query_executor_run_id, query_ids | set(source_matches.values())
        ),
    )
    by_query_run = {r["query_executor_run_id"]: r for r in query_runs}
    for run_id in sorted(query_ids):
        target = by_query_run.get(run_id, {})
        source_id = target.get("metadata", {}).get("source_match_run_id")
        if not source_id or source_id == run_id:
            continue
        source = by_query_run.get(source_id, {})
        if (
            source_id != source_matches[run_id]
            or source.get("schema_version") != "0.3.0"
            or any(
                not source.get(key) or source[key] != target.get(key)
                for key in (
                    "routing_projection_id",
                    "config_hash",
                    "snapshot_id",
                    "source_compiler_run_id",
                )
            )
        ):
            snapshot.missing(
                "query_embedding_source_match_unproven", {"process_facts", "cost_facts"}
            )
            continue
        query_ids.add(source_id)
        snapshot.query_scopes[source_id].update(snapshot.query_scopes[run_id])
    query_runs = [r for r in query_runs if r["query_executor_run_id"] in query_ids]
    snapshot.tables[_table(qm.QueryExecutorRunModel).fullname] = query_runs
    compiler_ids.update(r["source_compiler_run_id"] for r in query_runs)
    baseline_ids.update(
        r["metadata"]["fallback_run_id"]
        for r in query_runs
        if r.get("metadata", {}).get("fallback_run_id")
    )
    await read(session, cm.CompilerRunModel, _in(cm.CompilerRunModel.run_id, compiler_ids))
    run_ids = hints | compiler_ids | query_ids | baseline_ids
    run_rows = await read(
        session,
        bm.RunManifestModel,
        or_(
            _in(bm.RunManifestModel.run_id, run_ids),
            (bm.RunManifestModel.method == CONFORMANCE_VERSION)
            & (bm.RunManifestModel.manifest_json["metadata"]["campaign_id"].astext == campaign_id),
        ),
    )
    for row in run_rows:
        if row["method"] != CONFORMANCE_VERSION:
            continue
        try:
            probe = RunManifest.model_validate(row["manifest_json"])
            metadata = probe.metadata
            if not (
                probe.run_id == row["run_id"]
                and probe.method == row["method"] == CONFORMANCE_VERSION
                and probe.dataset.value == row["dataset"] == "unknown"
                and not probe.dataset_snapshot_ids
                and probe.paper_mode is False
                and row["paper_mode"] is False
                and metadata.get("campaign_id") == campaign_id
                and metadata.get("protocol_hash") == manifest.protocol.content_hash
                and metadata.get("budget_selection_hash")
                in {r["selection_hash"] for r in selections}
                and type(metadata.get("scientific_assignments")) is int
                and metadata.get("scientific_assignments") == 0
                and metadata.get("paper_result") is False
            ):
                raise ValueError("conformance scope mismatch")
        except (KeyError, ValueError):
            snapshot.missing("conformance_run_identity_mismatch")
        else:
            snapshot.conformance.add(row["run_id"])
    snapshot.runs = {r["run_id"] for r in run_rows}
    run_links = await read(
        session, bm.RunSnapshotLinkModel, _in(bm.RunSnapshotLinkModel.run_id, snapshot.runs)
    )
    linked: dict[str, set[str]] = defaultdict(set)
    for link in run_links:
        linked[link["run_id"]].add(link["snapshot_id"])
    rejected = {
        run
        for run in snapshot.runs
        if (run in snapshot.conformance and linked[run])
        or (
            run not in snapshot.conformance
            and (not linked[run] or not linked[run] <= set(manifest.snapshot_ids.values()))
        )
    }
    if rejected:
        snapshot.missing("scientific_run_snapshot_scope_unproven")
        snapshot.runs -= rejected
        snapshot.conformance -= rejected
        snapshot.tables[_table(bm.RunManifestModel).fullname] = [
            r for r in run_rows if r["run_id"] in snapshot.runs
        ]
    if jobs and not snapshot.runs:
        snapshot.missing("campaign_scientific_runs_absent")
    evals = await read(
        session,
        em.EvaluationRunModel,
        or_(
            em.EvaluationRunModel.metadata_json["campaign_id"].astext == campaign_id,
            _in(em.EvaluationRunModel.evaluation_run_id, campaign_evaluations),
            em.EvaluationRunModel.metadata_json["campaign_id"].astext.is_(None)
            & or_(
                _in(em.EvaluationRunModel.legacy_source_run_id, snapshot.runs),
                _in(em.EvaluationRunModel.suite_id, owned_suites),
            ),
        ),
    )
    snapshot.evaluations = {r["evaluation_run_id"] for r in evals}

    # Include every persisted attempt/status. No WHERE status='success' filter.
    await read(
        session, xm.RuntimeModelCallModel, _in(xm.RuntimeModelCallModel.run_id, snapshot.runs)
    )
    await read(
        session,
        xm.EvaluationModelCallModel,
        or_(
            xm.EvaluationModelCallModel.evaluation_run_id.is_(None)
            & _in(xm.EvaluationModelCallModel.run_id, snapshot.runs),
            _in(xm.EvaluationModelCallModel.evaluation_run_id, snapshot.evaluations),
        ),
    )
    await read(session, xm.RuntimeArtifactModel, _in(xm.RuntimeArtifactModel.run_id, snapshot.runs))
    await read(
        session,
        xm.EvaluationArtifactModel,
        or_(
            xm.EvaluationArtifactModel.evaluation_run_id.is_(None)
            & _in(xm.EvaluationArtifactModel.run_id, snapshot.runs),
            _in(xm.EvaluationArtifactModel.evaluation_run_id, snapshot.evaluations),
        ),
    )

    model: Any
    for model in (
        xm.RunStateModel,
        xm.ExperimentJobModel,
        cm.CompilerStageJobModel,
        dm.CandidateStageJobModel,
        cm.ProgramExecutionModel,
    ):
        await read(
            session,
            model,
            _in(model.run_id, snapshot.runs),
            {"attempts", "performance", "failures"},
        )
    await read(
        session,
        xm.JobAttemptModel,
        _in(xm.JobAttemptModel.job_id, {r["job_id"] for r in snapshot.rows(xm.ExperimentJobModel)}),
        {"attempts", "performance", "failures"},
    )
    await read(
        session,
        qm.QueryExecutorStageJobModel,
        _in(qm.QueryExecutorStageJobModel.query_executor_run_id, query_ids),
        {"attempts", "performance", "failures"},
    )
    await read(
        session,
        em.EvaluationJobModel,
        _in(em.EvaluationJobModel.evaluation_run_id, snapshot.evaluations),
        {"attempts", "performance", "failures"},
    )
    await read(
        session,
        em.EvaluationJobAttemptModel,
        _in(
            em.EvaluationJobAttemptModel.evaluation_job_id,
            {r["evaluation_job_id"] for r in snapshot.rows(em.EvaluationJobModel)},
        ),
        {"attempts", "performance", "failures"},
    )
    await read(
        session,
        em.EvaluationRunStateModel,
        _in(em.EvaluationRunStateModel.evaluation_run_id, snapshot.evaluations),
        {"performance", "failures"},
    )

    for model in (
        cm.QBankQuestionModel,
        cm.MemoryProgramModel,
        cm.ProgramStepModel,
        cm.PlannerGenerationBatchModel,
        cm.DoubterFeedbackModel,
        dm.PlannerRevisionPatchModel,
        dm.PlannerRevisionLineageModel,
        dm.DoubterDirectiveModel,
        dm.DoubterAssessmentModel,
        questioner.QuestionerGenerationModel,
    ):
        await read(session, model, _in(model.run_id, compiler_ids), {"compiler_patches"})
    await read(
        session,
        qm.QueryRuntimeTypedAnswerModel,
        _in(qm.QueryRuntimeTypedAnswerModel.query_executor_run_id, snapshot.runs),
        {"runtime_answers"},
    )
    for model in (
        qm.QueryExperimentCellResultModel,
        qm.AnswerReuseLinkModel,
        qm.QueryExecutionReuseLinkModel,
        qm.QueryRouteDecisionModel,
        qm.QueryAnswerSelectionModel,
        qm.QueryFallbackDecisionModel,
        qm.QueryRuntimePhaseStateModel,
    ):
        await read(
            session, model, _in(model.query_executor_run_id, query_ids), {"variant_cell_reuse"}
        )
    await read(
        session,
        qm.QBankRerankerAssessmentModel,
        _in(qm.QBankRerankerAssessmentModel.query_executor_run_id, query_ids),
        {"reranker_calls"},
    )
    batches = await read(
        session,
        qm.QBankMatchBatchModel,
        _in(qm.QBankMatchBatchModel.query_executor_run_id, query_ids),
        {"variant_cell_reuse"},
    )
    for model in (qm.QBankMatchChannelCandidateModel, qm.QBankMatchFusedCandidateModel):
        await read(
            session,
            model,
            _in(model.match_batch_id, {r["match_batch_id"] for r in batches}),
            {"variant_cell_reuse"},
        )
    for model in (
        qm.AnswerFormatDiagnosticModel,
        questioner.RuntimeStructuredOutputItemDiagnosticModel,
        questioner.EvaluationStructuredOutputItemDiagnosticModel,
    ):
        await read(session, model, _in(model.run_id, snapshot.runs), {"failures", "review_samples"})
    await read(
        session,
        xm.JudgeScoreModel,
        _in(xm.JudgeScoreModel.evaluation_run_id, snapshot.evaluations),
        {"runtime_answers"},
    )
    for model in (
        bm.BenchmarkQueryModel,
        bm.QueryGoldModel,
        bm.QueryGoldEvidenceIdModel,
        bm.QueryGoldMemoryIdModel,
        bm.QueryGoldSessionIdModel,
        bm.QueryGoldImageIdModel,
    ):
        await read(
            session,
            model,
            tuple_(model.snapshot_id, model.query_id).in_(sorted(snapshot.selected)),
            {"gold"},
        )
    await _read_process_snapshot(session, snapshot, manifest, compiler_ids, query_ids)
    await _read_cost_dependencies(session, snapshot, manifest, selection)


async def _read_process_snapshot(session, snapshot, manifest, compiler_ids, query_ids) -> None:
    """Follow scoped relational keys; never enumerate an entire dataset or corpus."""
    runs = snapshot.runs - snapshot.conformance
    compilers, queries = compiler_ids & runs, query_ids & runs
    snapshot.process_scope = {
        "run_ids": sorted(runs),
        "compiler_run_ids": sorted(compilers),
        "query_executor_run_ids": sorted(queries),
        "snapshot_queries": [list(pair) for pair in sorted(snapshot.selected)],
        "corpus_scope": "referenced_documents_and_windows_not_entire_corpus",
        "asset_scope": "database_lineage_not_binary_image_payloads",
    }

    async def keyed(model, columns, keys, *, required=False):
        columns = tuple(columns)
        keys = {tuple(k) for k in keys if all(v is not None for v in k)}
        attrs = [getattr(model, k) for k in columns]
        condition = tuple_(*attrs).in_(sorted(keys)) if keys else false()
        if hasattr(model, "query_id") and set(columns) & {"run_id", "query_executor_run_id"}:
            condition = condition & (
                model.query_id.is_(None) | _in(model.query_id, {q for _, q in snapshot.selected})
            )
        if model in (qm.QueryEmbeddingModel, qm.QueryEmbeddingBatchModel):
            condition = condition & or_(
                false(),
                *(
                    (model.query_executor_run_id == run_id)
                    & (model.snapshot_id == snapshot_id)
                    & (
                        model.query_id == query_id
                        if model is qm.QueryEmbeddingModel
                        else any_(model.query_ids) == query_id
                    )
                    for run_id, pairs in snapshot.query_scopes.items()
                    for snapshot_id, query_id in sorted(pairs & snapshot.selected)
                ),
            )
        snapshot.process_tables[_table(model).fullname] = model
        rows = await snapshot.read(session, model, condition, {"process_facts"})
        links = defaultdict(set)
        for link in snapshot.rows(bm.RunSnapshotLinkModel):
            links[link["run_id"]].add(link["snapshot_id"])

        def in_scope(row):
            if tuple(row.get(k) for k in columns) not in keys:
                return False
            if row.get("snapshot_id") and row["snapshot_id"] not in manifest.snapshot_ids.values():
                return False
            if model in (qm.QueryEmbeddingModel, qm.QueryEmbeddingBatchModel):
                query_values = (
                    [row.get("query_id")]
                    if model is qm.QueryEmbeddingModel
                    else row.get("query_ids", [])
                )
                if not query_values or not all(
                    (row.get("snapshot_id"), q)
                    in snapshot.query_scopes[row.get("query_executor_run_id")] & snapshot.selected
                    for q in query_values
                ):
                    return False
            if row.get("query_id"):
                snapshots = (
                    {row["snapshot_id"]}
                    if row.get("snapshot_id")
                    else links[row.get("run_id", row.get("query_executor_run_id"))]
                )
                return bool(snapshots) and all(
                    (s, row["query_id"]) in snapshot.selected for s in snapshots
                )
            return True

        scoped = [r for r in rows if in_scope(r)]
        if len(scoped) != len(rows):
            snapshot.missing("process_scope_mismatch:" + _table(model).fullname, {"process_facts"})
        if _table(model).fullname in snapshot.tables:
            snapshot.tables[_table(model).fullname] = scoped
        if required and keys - {tuple(r.get(k) for k in columns) for r in scoped}:
            snapshot.missing(
                "process_referenced_row_missing:" + _table(model).fullname, {"process_facts"}
            )
        return scoped

    async def by_run(model, ids, column="run_id"):
        return await keyed(model, (column,), {(r,) for r in ids})

    def values(model, *columns):
        return {tuple(row.get(c) for c in columns) for row in snapshot.rows(model)}

    # These already-read roots are included in the process index and FK checks.
    model: Any
    for model in (
        bm.RunManifestModel,
        bm.RunSnapshotLinkModel,
        cm.CompilerRunModel,
        qm.QueryExecutorRunModel,
        qm.QueryExperimentCellResultModel,
        qm.QueryRouteDecisionModel,
        qm.QueryRuntimeTypedAnswerModel,
        qm.QueryExecutionReuseLinkModel,
        qm.AnswerReuseLinkModel,
        qm.QBankMatchBatchModel,
        qm.QBankMatchChannelCandidateModel,
        qm.QBankMatchFusedCandidateModel,
        qm.QBankRerankerAssessmentModel,
        xm.RuntimeModelCallModel,
    ):
        snapshot.process_tables[_table(model).fullname] = model
        if _table(model).fullname not in snapshot.tables:
            snapshot.missing(
                "process_root_unavailable:" + _table(model).fullname, {"process_facts"}
            )

    for model in (
        cm.CompilerUnitModel,
        cm.CompilerUnitMemoryModel,
        cm.CompilerUnitStageResultModel,
        cm.CompilerStageJobModel,
        cm.PlannerGenerationBatchModel,
        cm.PlannerCandidateDraftModel,
        cm.PlannerCandidateOutcomeModel,
        cm.PlannerBatchProgramModel,
        dm.CandidateStageJobModel,
        dm.DoubterDirectiveModel,
        dm.PlannerRevisionLineageModel,
        dm.PlannerRevisionPatchModel,
        questioner.CompilerUnitLinkModel,
        questioner.QuestionOpportunitySetModel,
        questioner.QuestionOpportunityModel,
        questioner.QuestionOpportunityEvidenceRefModel,
        questioner.QuestionerGenerationModel,
        questioner.QuestionerAnswerClaimModel,
        questioner.QuestionerCandidateGateResultModel,
        questioner.QuestionerVerificationBatchModel,
        questioner.QuestionerCandidateVerificationModel,
        questioner.QuestionerCandidateSupersessionModel,
        questioner.QBankCandidateFamilyModel,
        questioner.QBankCandidateFamilyMemberModel,
        questioner.QBankDuplicateLinkModel,
        questioner.RuntimeStructuredOutputItemDiagnosticModel,
        cm.QBankQuestionModel,
        cm.QBankEvidenceRefModel,
        cm.QBankParaphraseModel,
        cm.MemoryProgramModel,
        cm.ProgramStepModel,
        cm.ProgramExpectedEvidenceRefModel,
        cm.PlannerRequirementContractModel,
        cm.ProgramValidationResultModel,
        cm.ProgramValidationIssueModel,
        cm.QBankDecisionModel,
        cm.DoubterFeedbackModel,
        dm.R0InputManifestModel,
        dm.QBankRoundProjectionManifestModel,
        dm.QBankRoundProjectionMemberModel,
        dm.RoutingProjectionManifestModel,
        dm.RoutingProjectionMemberModel,
        dm.DoubterAssessmentModel,
        cm.ProgramDryRunAssessmentModel,
        cm.ProgramDryRunEvidenceMatchModel,
    ):
        await by_run(model, compilers)
    await by_run(questioner.QBankIndexManifestModel, compilers, "compiler_run_id")
    await keyed(
        dm.DoubterAssessmentDimensionModel,
        ("assessment_id",),
        values(dm.DoubterAssessmentModel, "assessment_id"),
    )
    await by_run(cm.ProgramExecutionModel, runs)
    for model in (cm.ProgramExecutionStepModel, cm.ProgramExecutionEvidenceRefModel):
        await keyed(
            model,
            ("execution_id", "run_id"),
            values(cm.ProgramExecutionModel, "execution_id", "run_id"),
        )
    for model in (
        qm.QueryProgramBindingModel,
        qm.QueryEvidenceSufficiencyAssessmentModel,
        qm.QueryEvidenceRelevanceAssessmentModel,
        qm.QueryEvidenceRelevanceItemModel,
        qm.QueryFallbackDecisionModel,
        qm.QueryAnswerSelectionModel,
        qm.PmmcFallbackRunLinkModel,
        qm.PmmcFallbackQueryLinkModel,
        qm.QueryEmbeddingBatchModel,
        qm.QueryEmbeddingModel,
    ):
        await by_run(model, queries, "query_executor_run_id")

    for model in (
        xm.RunQueryModel,
        xm.RetrievalBatchModel,
        xm.RetrievalTraceModel,
        xm.RetrievalTraceComponentModel,
        xm.RetrievalBatchQueryEmbeddingModel,
        blm.ContextAssemblyModel,
        blm.ContextAssemblyRetrievalCandidateModel,
    ):
        await by_run(model, runs)
    for model in (blm.ContextAssemblyMemoryUnitModel, blm.ContextAssemblyImageModel):
        await keyed(
            model, ("context_assembly_id",), values(blm.ContextAssemblyModel, "context_assembly_id")
        )
    document_ids = values(xm.RetrievalTraceModel, "retrieval_document_id") | values(
        xm.RetrievalTraceComponentModel, "retrieval_document_id"
    )
    await keyed(blm.RetrievalDocumentModel, ("document_id",), document_ids, required=True)
    await keyed(
        blm.RetrievalDocumentImageModel,
        ("document_id",),
        values(blm.RetrievalDocumentModel, "document_id"),
    )
    window_ids = (
        values(xm.RetrievalTraceModel, "retrieval_window_id")
        | values(blm.ContextAssemblyRetrievalCandidateModel, "window_id")
        | values(blm.RetrievalDocumentModel, "parent_window_id")
    )
    await keyed(blm.MemoryWindowModel, ("window_id",), window_ids, required=True)
    await keyed(
        blm.MemoryWindowUnitModel, ("window_id",), values(blm.MemoryWindowModel, "window_id")
    )
    await keyed(
        blm.RetrievalCorpusModel,
        ("corpus_id",),
        values(xm.RetrievalBatchModel, "corpus_id")
        | values(blm.RetrievalDocumentModel, "corpus_id")
        | values(blm.MemoryWindowModel, "corpus_id"),
        required=True,
    )
    await keyed(
        xm.EmbeddingModel,
        ("embedding_id",),
        values(xm.RetrievalBatchQueryEmbeddingModel, "embedding_id")
        | values(qm.QueryEmbeddingModel, "embedding_id"),
        required=True,
    )
    collection_ids = (
        values(xm.RetrievalBatchModel, "embedding_collection_id")
        | values(xm.RetrievalBatchModel, "query_embedding_collection_id")
        | values(xm.RetrievalTraceComponentModel, "embedding_collection_id")
        | values(xm.EmbeddingModel, "collection_id")
        | values(qm.QueryEmbeddingModel, "collection_id")
    )
    for column in (
        "text_question_collection_id",
        "text_paraphrase_collection_id",
        "multimodal_question_collection_id",
        "multimodal_paraphrase_collection_id",
    ):
        collection_ids.update(values(questioner.QBankIndexManifestModel, column))
    await keyed(xm.EmbeddingCollectionModel, ("collection_id",), collection_ids, required=True)
    await _read_embedding_dependencies(session, snapshot, manifest)

    await keyed(
        bm.BenchmarkQueryModel, ("snapshot_id", "query_id"), snapshot.selected, required=True
    )
    await keyed(bm.QueryImageModel, ("snapshot_id", "query_id"), snapshot.selected)
    evidence_models = (
        cm.QBankEvidenceRefModel,
        cm.ProgramExpectedEvidenceRefModel,
        cm.ProgramExecutionEvidenceRefModel,
    )
    # Opportunity refs inherit snapshot identity from their scoped compiler unit.
    unit_snapshots = {
        (r["run_id"], r["compiler_unit_id"]): r["snapshot_id"]
        for r in snapshot.rows(cm.CompilerUnitModel)
    }

    def opportunity_values(column):
        return {
            (unit_snapshots.get((r["run_id"], r["compiler_unit_id"])), r.get(column))
            for r in snapshot.rows(questioner.QuestionOpportunityEvidenceRefModel)
        }

    memory_keys = set()
    for model in (
        *evidence_models,
        cm.CompilerUnitMemoryModel,
        blm.MemoryWindowUnitModel,
        blm.ContextAssemblyMemoryUnitModel,
    ):
        memory_keys.update(values(model, "snapshot_id", "memory_id"))
    memory_keys.update(opportunity_values("memory_id"))
    for row in snapshot.rows(blm.RetrievalDocumentModel):
        memory_keys.update((row["snapshot_id"], m) for m in row["memory_ids"])
    # Old direct-memory retrievals do not have document/window foreign keys.
    batches = {r["retrieval_batch_id"]: r for r in snapshot.rows(xm.RetrievalBatchModel)}
    image_keys = set()
    for row in snapshot.rows(xm.RetrievalTraceModel):
        batch = batches.get(row["retrieval_batch_id"], {})
        if row.get("retrieved_type") == "memory":
            memory_keys.add((batch.get("snapshot_id"), row["retrieved_id"]))
        elif row.get("retrieved_type") == "image":
            image_keys.add((batch.get("snapshot_id"), row["retrieved_id"]))
    for model in (
        *evidence_models,
        blm.ContextAssemblyImageModel,
        blm.RetrievalDocumentImageModel,
        bm.QueryImageModel,
    ):
        image_keys.update(values(model, "snapshot_id", "image_id"))
    image_keys.update(opportunity_values("image_id"))
    await keyed(bm.MemoryUnitImageModel, ("snapshot_id", "image_id"), image_keys)
    memory_keys.update(values(bm.MemoryUnitImageModel, "snapshot_id", "memory_id"))
    # Caption ownership is the authoritative metadata mapping used by EvidenceLineageResolver.
    caption_keys = {
        pair
        for model in evidence_models
        for pair in values(model, "snapshot_id", "caption_id")
        if all(pair)
    }
    caption_keys.update(pair for pair in opportunity_values("caption_id") if all(pair))
    if caption_keys:
        model = bm.MemoryUnitModel
        snapshot.process_tables[_table(model).fullname] = model
        captions = await snapshot.read(
            session,
            model,
            tuple_(model.snapshot_id, model.metadata_json["caption_id"].astext).in_(
                sorted(caption_keys)
            ),
            {"process_facts"},
        )
        scoped_captions = [
            r
            for r in captions
            if (r["snapshot_id"], r.get("metadata", {}).get("caption_id")) in caption_keys
        ]
        if len(captions) != len(scoped_captions):
            snapshot.missing("process_caption_scope_mismatch", {"process_facts"})
        if caption_keys - {
            (r["snapshot_id"], r["metadata"]["caption_id"]) for r in scoped_captions
        }:
            snapshot.missing("process_caption_lineage_missing", {"process_facts"})
        memory_keys.update((r["snapshot_id"], r["memory_id"]) for r in scoped_captions)
    await keyed(bm.MemoryUnitModel, ("snapshot_id", "memory_id"), memory_keys, required=True)
    await keyed(
        bm.MemoryUnitImageModel,
        ("snapshot_id", "memory_id"),
        values(bm.MemoryUnitModel, "snapshot_id", "memory_id"),
    )
    for model in (
        *evidence_models,
        blm.ContextAssemblyImageModel,
        blm.RetrievalDocumentImageModel,
        bm.QueryImageModel,
        bm.MemoryUnitImageModel,
    ):
        image_keys.update(values(model, "snapshot_id", "image_id"))
    await keyed(bm.ImageModel, ("snapshot_id", "image_id"), image_keys, required=True)
    await keyed(
        bm.AssetModel,
        ("snapshot_id", "asset_id"),
        values(bm.ImageModel, "snapshot_id", "asset_id"),
        required=True,
    )
    session_keys = set()
    session_keys.update(opportunity_values("session_id"))
    for model in (*evidence_models, bm.MemoryUnitModel, blm.MemoryWindowModel):
        session_keys.update(values(model, "snapshot_id", "session_id"))
    for model in (bm.MemoryUnitModel, cm.CompilerUnitModel, bm.BenchmarkQueryModel):
        session_keys.update(values(model, "snapshot_id", "available_after_session"))
    await keyed(
        bm.BenchmarkSessionModel, ("snapshot_id", "session_id"), session_keys, required=True
    )
    await keyed(
        bm.DatasetSnapshotModel,
        ("snapshot_id",),
        {(r,) for r in manifest.snapshot_ids.values()},
        required=True,
    )
    _check_process_lineage(snapshot)


async def _read_cost_dependencies(session, snapshot, manifest, selection) -> None:
    """Read complete imported compiler graphs and index membership in this snapshot."""
    from pmmc.experiments.step8_cost_dependencies import (
        RAG_COST_DEPENDENCIES_ARTIFACT,
        collect_rag_index_cost_dependencies,
    )
    from pmmc.experiments.step8_partial_cost_dependencies import (
        PARTIAL_COST_DEPENDENCIES_ARTIFACT,
        collect_partial_prefix_cost_dependencies,
    )
    from pmmc.experiments.step8_pmmc_cost_dependencies import (
        PMMC_COST_DEPENDENCIES_ARTIFACT,
        collect_pmmc_cost_dependencies,
    )
    from pmmc.experiments.step8_questioner_cost_dependencies import (
        QUESTIONER_COST_DEPENDENCIES_ARTIFACT,
        collect_questioner_cost_dependencies,
    )
    from pmmc.experiments.step8_r0_cost_dependencies import (
        R0_COST_DEPENDENCIES_ARTIFACT,
        collect_r0_prefix_cost_dependencies,
    )

    common = dict(
        campaign_id=manifest.campaign_id,
        selection_hash=selection.content_hash,
        database_snapshot=snapshot.identity,
    )
    collections = {
        r["collection_id"] for r in snapshot.rows(xm.EmbeddingCollectionModel) if r.get("corpus_id")
    }
    imported = {
        r["run_id"]
        for r in snapshot.rows(cm.CompilerRunModel)
        if r.get("metadata", {}).get("questioner_upstream_import")
    }
    r0_imported = {
        r["run_id"]
        for r in snapshot.rows(cm.CompilerRunModel)
        if r.get("metadata", {}).get("r0_prefix_upstream_import")
    }
    partial_imported = {
        r["run_id"]
        for r in snapshot.rows(cm.CompilerRunModel)
        if r.get("metadata", {}).get("partial_prefix_upstream_import")
    }
    if partial_imported:
        partial_dependencies = await collect_partial_prefix_cost_dependencies(
            session, compiler_run_ids=partial_imported, **common
        )
        snapshot.cost_dependencies[PARTIAL_COST_DEPENDENCIES_ARTIFACT] = _json(
            partial_dependencies
        )
        # Foreign shared embedding owners need not be compiler runs. Only
        # compiler identities discovered by the reader can own older Q proofs.
        imported.update(partial_dependencies["dependency_compiler_run_ids"])
    if r0_imported:
        r0_dependencies = await collect_r0_prefix_cost_dependencies(
            session, compiler_run_ids=r0_imported, **common
        )
        snapshot.cost_dependencies[R0_COST_DEPENDENCIES_ARTIFACT] = _json(r0_dependencies)
        imported.update(r0_dependencies["dependency_run_ids"])
    if collections:
        snapshot.cost_dependencies[RAG_COST_DEPENDENCIES_ARTIFACT] = _json(
            await collect_rag_index_cost_dependencies(session, collection_ids=collections, **common)
        )
    if imported:
        snapshot.cost_dependencies[QUESTIONER_COST_DEPENDENCIES_ARTIFACT] = _json(
            await collect_questioner_cost_dependencies(session, compiler_run_ids=imported, **common)
        )
    compiler_ids = {row["run_id"] for row in snapshot.rows(cm.CompilerRunModel)}
    execution_ids = {row["execution_id"] for row in snapshot.rows(cm.ProgramExecutionModel)}
    if compiler_ids or execution_ids:
        snapshot.cost_dependencies[PMMC_COST_DEPENDENCIES_ARTIFACT] = _json(
            await collect_pmmc_cost_dependencies(
                session,
                compiler_run_ids=compiler_ids,
                execution_ids=execution_ids,
                r0_dependencies=snapshot.cost_dependencies.get(R0_COST_DEPENDENCIES_ARTIFACT),
                **common,
            )
        )
    if any(
        entry.get("status") != "complete"
        for value in snapshot.cost_dependencies.values()
        for entry in [
            *value.get("imports", []),
            *value.get("collections", []),
            *value.get("write_graphs", []),
            *value.get("executions", []),
        ]
    ):
        snapshot.missing("cost_dependency_proof_incomplete", {"cost_facts"})
    call_ids = {
        key
        for value in snapshot.cost_dependencies.values()
        for key in value["dependency_model_call_ids"]
    }
    owner_ids = {
        key for value in snapshot.cost_dependencies.values() for key in value["dependency_run_ids"]
    }
    snapshot.adopted_calls.update(
        key
        for value in snapshot.cost_dependencies.values()
        for key in value.get("adopted_model_call_ids", [])
    )

    async def extend(model, column, keys):
        original = list(snapshot.rows(model))
        needed = keys - {r[column] for r in original}
        if not needed:
            return
        extra = await snapshot.read(
            session, model, _in(getattr(model, column), needed), {"cost_facts"}
        )
        if any(r[column] not in needed for r in extra) or needed - {r[column] for r in extra}:
            snapshot.missing("cost_dependency_reference_missing", {"cost_facts"})
        if _table(model).fullname in snapshot.tables:
            snapshot.tables[_table(model).fullname] = sorted(
                original + [r for r in extra if r[column] in needed],
                key=lambda r: tuple(str(r[c.name]) for c in _table(model).primary_key),
            )

    await extend(xm.RuntimeModelCallModel, "model_call_id", call_ids)
    if snapshot.adopted_calls - {
        r["model_call_id"] for r in snapshot.rows(xm.RuntimeModelCallModel)
    }:
        snapshot.missing("adopted_cost_call_reference_missing", {"cost_facts"})
    await extend(bm.RunManifestModel, "run_id", owner_ids)
    await extend(bm.RunSnapshotLinkModel, "run_id", owner_ids)
    selected_calls = [
        r for r in snapshot.rows(xm.RuntimeModelCallModel) if r["model_call_id"] in call_ids
    ]
    artifacts = {
        a
        for r in selected_calls
        for a in (
            r.get("input_asset_id"),
            r.get("output_asset_id"),
            r.get("metadata", {}).get("error_artifact_id"),
        )
        if a
    }
    await extend(xm.RuntimeArtifactModel, "artifact_id", artifacts)
    links: dict[str, set[str]] = defaultdict(set)
    for row in snapshot.rows(bm.RunSnapshotLinkModel):
        links[row["run_id"]].add(row["snapshot_id"])
    by_id = {r["artifact_id"]: r for r in snapshot.rows(xm.RuntimeArtifactModel)}
    if any(
        not links[key] or not links[key] <= set(manifest.snapshot_ids.values()) for key in owner_ids
    ) or any(
        r["run_id"] not in owner_ids
        or any(
            by_id.get(a, {}).get("run_id") != r["run_id"]
            for a in (
                r.get("input_asset_id"),
                r.get("output_asset_id"),
                r.get("metadata", {}).get("error_artifact_id"),
            )
            if a
        )
        for r in selected_calls
    ):
        snapshot.missing("cost_dependency_scope_mismatch", {"cost_facts"})
    snapshot.dependency_runs.update(owner_ids - snapshot.runs)
    snapshot.dependency_calls.update(
        r["model_call_id"] for r in selected_calls if r["run_id"] not in snapshot.runs
    )
    snapshot.process_scope.update(
        dependency_run_ids=sorted(snapshot.dependency_runs),
        dependency_model_call_ids=sorted(snapshot.dependency_calls),
    )


async def _read_embedding_dependencies(session, snapshot, manifest) -> None:
    """Import exact consumed producer keys, not a shared cache run's call history."""
    families = {"process_facts", "cost_facts"}

    async def referenced(model, column, keys):
        keys = {key for key in keys if key is not None}
        old = list(snapshot.rows(model))
        existing = {r[column] for r in old}
        needed = keys - existing
        if not needed:
            return
        rows = await snapshot.read(session, model, _in(getattr(model, column), needed), families)
        scoped = [r for r in rows if r[column] in needed]
        if needed - {r[column] for r in scoped}:
            snapshot.missing("embedding_dependency_missing:" + _table(model).fullname, families)
        if _table(model).fullname in snapshot.tables:
            snapshot.tables[_table(model).fullname] = sorted(
                old + scoped, key=lambda r: tuple(str(r[c.name]) for c in _table(model).primary_key)
            )

    batches = {r["query_embedding_batch_id"]: r for r in snapshot.rows(qm.QueryEmbeddingBatchModel)}
    vectors = snapshot.rows(qm.QueryEmbeddingModel)
    embeddings = {r["embedding_id"]: r for r in snapshot.rows(xm.EmbeddingModel)}
    collections = {r["collection_id"]: r for r in snapshot.rows(xm.EmbeddingCollectionModel)}
    call_ids = {r.get("model_call_id") for r in batches.values()} | {
        r.get("model_call_id") for r in embeddings.values()
    }
    snapshot.embedding_calls.update(key for key in call_ids if key is not None)
    original_calls = {r["model_call_id"] for r in snapshot.rows(xm.RuntimeModelCallModel)}
    await referenced(xm.RuntimeModelCallModel, "model_call_id", call_ids)
    calls = {r["model_call_id"]: r for r in snapshot.rows(xm.RuntimeModelCallModel)}
    dependency_calls = {
        key for key in set(calls) - original_calls if calls[key]["run_id"] not in snapshot.runs
    }
    owners = {calls[k]["run_id"] for k in dependency_calls} | {
        r.get("run_id") for r in collections.values() if r.get("run_id")
    }
    dependency_runs = owners - snapshot.runs
    await referenced(bm.RunManifestModel, "run_id", dependency_runs)
    await referenced(bm.RunSnapshotLinkModel, "run_id", dependency_runs)
    links: dict[str, set[str]] = defaultdict(set)
    for row in snapshot.rows(bm.RunSnapshotLinkModel):
        links[row["run_id"]].add(row["snapshot_id"])
    manifests = {r["run_id"] for r in snapshot.rows(bm.RunManifestModel)}
    if any(
        run not in manifests
        or not links[run]
        or not links[run] <= set(manifest.snapshot_ids.values())
        for run in dependency_runs
    ):
        snapshot.missing("embedding_dependency_snapshot_unproven", families)
    snapshot.dependency_runs.update(dependency_runs)
    snapshot.dependency_calls.update(dependency_calls)
    snapshot.process_scope["dependency_run_ids"] = sorted(snapshot.dependency_runs)
    snapshot.process_scope["dependency_model_call_ids"] = sorted(snapshot.dependency_calls)
    snapshot.process_scope["dependency_scope"] = "referenced_embedding_producers_not_entire_runs"
    artifact_ids = {
        value
        for key in dependency_calls
        for value in (
            calls[key].get("input_asset_id"),
            calls[key].get("output_asset_id"),
            calls[key].get("metadata", {}).get("error_artifact_id"),
        )
        if value
    }
    await referenced(xm.RuntimeArtifactModel, "artifact_id", artifact_ids)
    artifacts = {r["artifact_id"]: r for r in snapshot.rows(xm.RuntimeArtifactModel)}
    for key in dependency_calls:
        call = calls[key]
        for artifact_id in (
            call.get("input_asset_id"),
            call.get("output_asset_id"),
            call.get("metadata", {}).get("error_artifact_id"),
        ):
            if artifact_id and artifacts.get(artifact_id, {}).get("run_id") != call["run_id"]:
                snapshot.missing("embedding_dependency_artifact_scope_mismatch", families)

    from pmmc.schemas.query_executor import QueryEmbeddingBatch, QueryEmbeddingFact

    for row in batches.values():
        try:
            batch = QueryEmbeddingBatch.model_validate(row)
            committed = [
                v
                for v in vectors
                if v["query_embedding_batch_id"] == batch.query_embedding_batch_id
            ]
            if len(committed) != batch.committed_count:
                raise ValueError("committed vector count mismatch")
            if batch.metadata.get("coverage_projection") is True:
                raise ValueError("evaluator coverage embedding producer is not runtime lineage")
            call = calls[batch.model_call_id]
            if (
                call.get("snapshot_id") not in (None, batch.snapshot_id)
                or (call.get("query_id") is not None and call["query_id"] not in batch.query_ids)
                or batch.snapshot_id not in links[call["run_id"]]
            ):
                raise ValueError("embedding batch producer scope mismatch")
        except (ValueError, KeyError, TypeError):
            snapshot.missing("query_embedding_batch_lineage_invalid", families)
    for row in vectors:
        try:
            vector = QueryEmbeddingFact.model_validate(row)
            batch_row = batches[vector.query_embedding_batch_id]
            embedding = embeddings[vector.embedding_id]
            collection = collections[vector.collection_id]
            if (
                (vector.query_executor_run_id, vector.snapshot_id, vector.channel.value)
                != (
                    batch_row["query_executor_run_id"],
                    batch_row["snapshot_id"],
                    batch_row["channel"],
                )
                or vector.query_id not in batch_row["query_ids"]
                or embedding["collection_id"] != vector.collection_id
                or embedding["source_id"] != vector.query_id
                or embedding["input_hash"] != vector.input_hash
                or embedding["dimension"] != vector.vector_dimension
                or collection["snapshot_id"] != vector.snapshot_id
                or collection["dimension"] != vector.vector_dimension
                or (
                    batch_row.get("model_call_id")
                    and embedding.get("model_call_id") != batch_row["model_call_id"]
                )
            ):
                raise ValueError("vector producer identity mismatch")
            call = calls.get(batch_row.get("model_call_id"))
            if call and (
                call.get("snapshot_id") not in (None, vector.snapshot_id)
                or call.get("query_id") not in (None, vector.query_id)
                or vector.snapshot_id not in links[call["run_id"]]
            ):
                raise ValueError("producer call scope mismatch")
        except (ValueError, KeyError, TypeError):
            snapshot.missing("query_embedding_vector_lineage_invalid", families)


def _check_process_lineage(snapshot) -> None:
    """Check relational closure and persisted denominators, not process metrics."""

    def missing(reason):
        snapshot.missing(reason, {"process_facts"})

    for name, model in snapshot.process_tables.items():
        for constraint in _table(model).foreign_key_constraints:
            elements = list(constraint.elements)
            parent_table_name = elements[0].target_fullname.rpartition(".")[0]
            if parent_table_name not in snapshot.process_tables:
                continue
            source = [e.parent.name for e in elements]
            target = [e.column.name for e in elements]
            available = {
                tuple(r.get(k) for k in target)
                for r in snapshot.tables.get(parent_table_name, [])
            }
            for row in snapshot.tables.get(name, []):
                if not all(k in row for k in source):
                    if any(e.parent.nullable and e.parent.name not in row for e in elements):
                        continue
                    missing("process_lineage_fields_missing:" + name + ":" + ",".join(source))
                    continue
                reference_values = tuple(row[k] for k in source)
                if (
                    all(v is not None for v in reference_values)
                    and reference_values not in available
                ):
                    missing("process_lineage_missing:" + name + ":" + ",".join(source))

    for parent, child, key, count in (
        (
            questioner.QBankCandidateFamilyModel,
            questioner.QBankCandidateFamilyMemberModel,
            "family_id",
            "member_count",
        ),
        (
            questioner.QuestionerVerificationBatchModel,
            questioner.QuestionerCandidateVerificationModel,
            "verification_batch_id",
            "input_count",
        ),
        (
            questioner.QuestionOpportunitySetModel,
            questioner.QuestionOpportunityModel,
            "opportunity_set_id",
            "opportunity_count",
        ),
        (
            dm.QBankRoundProjectionManifestModel,
            dm.QBankRoundProjectionMemberModel,
            "projection_id",
            "member_count",
        ),
        (
            dm.RoutingProjectionManifestModel,
            dm.RoutingProjectionMemberModel,
            "routing_projection_id",
            "member_count",
        ),
        (xm.RetrievalBatchModel, xm.RetrievalTraceModel, "retrieval_batch_id", "returned_count"),
    ):
        for row in snapshot.rows(parent):
            actual = sum(c[key] == row[key] for c in snapshot.rows(child))
            if row.get(count) != actual:
                missing("process_denominator_mismatch:" + _table(child).fullname)

    for execution in snapshot.rows(cm.ProgramExecutionModel):
        planned = {
            r["step_id"]
            for r in snapshot.rows(cm.ProgramStepModel)
            if r["plan_id"] == execution["plan_id"]
        }
        observed = {
            r["step_id"]
            for r in snapshot.rows(cm.ProgramExecutionStepModel)
            if r["execution_id"] == execution["execution_id"]
        }
        if not observed <= planned or (
            execution["status"] == "success"
            and execution["control_action"] == "continue"
            and (not planned or observed != planned)
        ):
            missing("process_execution_step_denominator_mismatch")
        for output_key, ref_key in (
            ("retrieved_memory_ids", "memory_id"),
            ("retrieved_image_ids", "image_id"),
            ("retrieved_caption_ids", "caption_id"),
            ("retrieved_session_ids", "session_id"),
        ):
            expected = {
                value
                for step in snapshot.rows(cm.ProgramExecutionStepModel)
                if step["execution_id"] == execution["execution_id"] and step.get("success")
                for value in step.get("output", {}).get(output_key, [])
            }
            observed_refs = {
                r.get(ref_key)
                for r in snapshot.rows(cm.ProgramExecutionEvidenceRefModel)
                if r["execution_id"] == execution["execution_id"]
            }
            if not expected <= observed_refs:
                missing("process_execution_evidence_missing:" + ref_key)

    # Cell payloads contain soft references not represented by SQL foreign keys.
    targets = {
        "binding": (qm.QueryProgramBindingModel, "query_program_binding_id"),
        "sufficiency": (qm.QueryEvidenceSufficiencyAssessmentModel, "sufficiency_assessment_id"),
        "relevance": (qm.QueryEvidenceRelevanceAssessmentModel, "relevance_assessment_id"),
    }
    for cell in snapshot.rows(qm.QueryExperimentCellResultModel):
        payload = cell.get("payload", {})
        if payload.get("execution_id") and not any(
            r["execution_id"] == payload["execution_id"]
            and r["run_id"] == cell["query_executor_run_id"]
            and r["query_id"] == cell["query_id"]
            for r in snapshot.rows(cm.ProgramExecutionModel)
        ):
            missing("process_cell_execution_missing")
        for field, (model, key) in targets.items():
            fact = payload.get(field)
            if fact and not any(
                r.get(key) == fact.get(key)
                and r.get("query_executor_run_id") == cell.get("query_executor_run_id")
                and r.get("query_id") == cell.get("query_id")
                and (not fact.get("content_hash") or r.get("content_hash") == fact["content_hash"])
                for r in snapshot.rows(model)
            ):
                missing("process_cell_lineage_missing:" + field)
        items = {
            r["relevance_item_id"]
            for r in snapshot.rows(qm.QueryEvidenceRelevanceItemModel)
            if r["query_executor_run_id"] == cell["query_executor_run_id"]
            and r["query_id"] == cell.get("query_id")
        }
        if any(r["relevance_item_id"] not in items for r in payload.get("relevance_items", [])):
            missing("process_cell_relevance_items_missing")


class _Builder:
    def __init__(self, snapshot: _Snapshot, store: CampaignArtifactStore) -> None:
        self.snapshot = snapshot
        self.store = store
        self.artifacts: dict[str, PrivateAuditArtifact] = {}
        self.records: list[PrivateAuditRecord] = []
        self.expected: set[tuple[str, str]] = set()
        self.journal_stores = [store]

    def fact(self, identity, value, kind="performance", scope="runtime") -> str:
        data = json_bytes(value)
        digest = self.store.put(data)
        self.store.verify(digest)
        item = PrivateAuditArtifact(
            artifact_id=identity, kind=kind, scope=scope, sha256=digest, size_bytes=len(data)
        )
        if identity in self.artifacts and self.artifacts[identity] != item:
            raise ValueError("inventory artifact identity conflict")
        self.artifacts[identity] = item
        return identity


def _explicit_attempt(row: dict[str, Any]) -> str | None:
    values = [
        row,
        row.get("metadata", {}),
        row.get("request_config", {}).get("request_metadata", {}),
    ]
    found = {
        str(item[key])
        for item in values
        if isinstance(item, dict)
        for key in ("step8_attempt_id", "campaign_attempt_id", "attempt_id")
        if item.get(key)
    }
    return next(iter(found)) if len(found) == 1 else None


def _bound_attempt(row, attempts, snapshot):
    explicit = _explicit_attempt(row)
    if explicit in attempts:
        return explicit
    workers = [row.get(key, "") for key in ("worker_id", "claimed_by", "lease_owner")]
    for model, key in (
        (xm.JobAttemptModel, "job_id"),
        (em.EvaluationJobAttemptModel, "evaluation_job_id"),
    ):
        if row.get(key):
            workers.extend(
                r.get("worker_id", "") for r in snapshot.rows(model) if r.get(key) == row[key]
            )
    bound = {
        attempt
        for attempt in attempts
        for worker in workers
        if isinstance(worker, str)
        and re.search(r"(?:^|:)" + re.escape(attempt) + r"(?:$|:)", worker)
    }
    return next(iter(bound)) if len(bound) == 1 else None


def _provider_journal(builder, manifest, request_id):
    """Bind the dispatch's original attempt and verify any persisted failure envelope."""
    reservations = [
        r
        for r in builder.snapshot.rows(sm.BenchmarkSpendReservationModel)
        if r["request_id"] == request_id
    ]
    if len(reservations) != 1:
        return None, None
    reservation = reservations[0]
    digest = reservation["request_hash"]
    if re.fullmatch(r"[a-f0-9]{64}", digest) is None:
        return None, None
    for journal_store in builder.journal_stores:
        try:
            intent = json.loads(
                read_campaign_file(
                    journal_store.root, f"_store/provider-dispatch/{digest}/intent.json"
                )
            )
            attempt = intent.pop("attempt_id")
            if (
                canonical_content_hash(intent) != digest
                or intent["approval"]["campaign_id"] != manifest.campaign_id
                or intent["request"]["request_id"] != request_id
            ):
                continue
            builder.fact(
                "provider-dispatch:" + digest, {**intent, "attempt_id": attempt}, "cost_fact"
            )
        except (OSError, ValueError, KeyError):
            continue
        break
    else:
        return None, None
    decisions = [
        row
        for row in builder.snapshot.rows(pc.ProviderReconciliationModel)
        if row["request_id"] == request_id and row["request_hash"] == digest
    ]
    if decisions:
        if len(decisions) != 1:
            return attempt, None
        try:
            decision = decisions[0]["payload"]
            unsigned = {
                key: value
                for key, value in decision.items()
                if key not in {"reconciliation_hash", "receipt_sha256"}
            }
            if (
                canonical_content_hash(unsigned) != decisions[0]["reconciliation_hash"]
                or decision["reconciliation_hash"] != decisions[0]["reconciliation_hash"]
                or decision["request_hash"] != digest
                or decision["approval_id"] != reservation["approval_id"]
            ):
                raise ValueError("reconciliation proof identity mismatch")
            for field in ("evidence_sha256", "receipt_sha256", "response_sha256"):
                if decision.get(field):
                    _provider_cas_artifact(builder, decision[field], "provider-proof:" + field)
            receipt = builder.store.load_json(decision["receipt_sha256"])
            if receipt != unsigned:
                raise ValueError("reconciliation receipt mismatch")
            original = journal_store.restored_response(request_id=request_id, request_hash=digest)
            if original is not None:
                builder.fact("provider-original-envelope:" + digest, original, "model_payload")
            if decision["disposition"] == "completed":
                if decision["response_sha256"] != reservation["response_hash"]:
                    raise ValueError("reconciled settlement response mismatch")
                envelope = builder.store.load_json(decision["response_sha256"])
                if (
                    envelope.get("request_hash") != digest
                    or envelope.get("request_id") != request_id
                ):
                    raise ValueError("reconciled response identity mismatch")
            else:
                if reservation["state"] != "cancelled" or reservation.get("response_hash"):
                    raise ValueError("reconciled nonexecution settlement mismatch")
                envelope = {
                    "request_id": request_id,
                    "request_hash": digest,
                    "source": "verified_nonexecution_reconciliation",
                    "reconciliation_hash": decision["reconciliation_hash"],
                    "error": {"error_code": "ReconciledNonexecution", "outcome": "failed"},
                }
            return attempt, builder.fact("provider-envelope:" + digest, envelope, "model_payload")
        except (OSError, ValueError, KeyError):
            builder.snapshot.missing(
                "provider_reconciliation_evidence_unavailable", {"cost_facts", "failures"}
            )
            return attempt, None
    try:
        journal = json.loads(
            read_campaign_file(journal_store.root, f"_store/responses/{digest}.json")
        )
        if journal["request_id"] != request_id or journal["request_hash"] != digest:
            return attempt, None
        response_hash = journal["response_sha256"]
        if reservation.get("response_hash") and reservation["response_hash"] != response_hash:
            return attempt, None
        envelope = journal_store.load_json(response_hash)
        if (
            envelope.get("request_id") != request_id
            or envelope.get("request_hash") != digest
            or envelope.get("approval", {}).get("campaign_id") != manifest.campaign_id
        ):
            return attempt, None
        ref = builder.fact("provider-envelope:" + digest, envelope, "model_payload")
        for rejection in envelope.get("rate_admission", {}).get("rate_rejection_sha256", []):
            _provider_cas_artifact(builder, rejection, "provider-rate-rejection")
        return attempt, ref
    except (OSError, ValueError, KeyError):
        return attempt, None


def _provider_cas_artifact(builder, digest: str, prefix: str) -> str:
    for store in builder.journal_stores:
        try:
            data = store.verify(digest).read_bytes()
        except (OSError, ValueError):
            continue
        if builder.store.put(data) != digest:
            raise ValueError("provider evidence copy checksum mismatch")
        identity = prefix + ":" + digest
        builder.artifacts[identity] = PrivateAuditArtifact(
            artifact_id=identity,
            kind="model_payload",
            scope="runtime",
            sha256=digest,
            size_bytes=len(data),
        )
        return identity
    raise ValueError("provider evidence CAS object unavailable")


def _model_wall_timings(builder: _Builder, manifest, attempts: set[str]) -> None:
    root = builder.store.root / "_store/model-timings"
    if not root.exists():
        return
    if root.is_symlink() or root.parent.is_symlink():
        builder.snapshot.missing("model_wall_timing_directory_unsafe", {"performance"})
        return
    jobs = {row["job_id"] for row in builder.snapshot.rows(sm.BenchmarkStageJobModel)}
    calls = {
        row["model_call_id"]
        for model in (xm.RuntimeModelCallModel, xm.EvaluationModelCallModel)
        for row in builder.snapshot.rows(model)
    }
    numeric = (
        "queue_wait_ms",
        "provider_ms",
        "artifact_persistence_ms",
        "trace_commit_ms",
        "wall_completion_ms",
    )
    for path in sorted(root.glob("*.json")):
        owned = False
        try:
            item = json.loads(
                read_campaign_file(builder.store.root, str(path.relative_to(builder.store.root)))
            )
            if item.get("campaign_id") != manifest.campaign_id:
                continue
            owned = True
            if (
                item.get("protocol") != "step8-model-call-wall-timing-v1"
                or item.get("attempt_id") not in attempts
                or item.get("job_id") not in jobs
                or item.get("model_call_id") not in calls
                or type(item.get("response_reused")) is not bool
                or any(
                    type(item.get(key)) not in {int, float}
                    or not math.isfinite(item[key])
                    or item[key] < 0
                    for key in numeric
                )
            ):
                raise ValueError("model timing ownership or fields invalid")
            # Only allowlisted dimensions are exported; filenames are not scope proof.
            safe = {
                key: item[key]
                for key in (
                    "protocol",
                    "campaign_id",
                    "attempt_id",
                    "job_id",
                    "model_call_id",
                    "response_reused",
                    *numeric,
                )
            }
            if "measurement_id" in item:
                if (
                    not isinstance(item["measurement_id"], str)
                    or re.fullmatch(r"[a-f0-9]{32}", item["measurement_id"]) is None
                ):
                    raise ValueError("invalid model timing observation identity")
                safe["measurement_id"] = item["measurement_id"]
            builder.fact("model-wall-timing:" + canonical_content_hash([path.name, safe]), safe)
        except (OSError, ValueError, KeyError, TypeError):
            if owned:
                builder.snapshot.missing("model_wall_timing_invalid", {"performance"})


def _attempts(builder: _Builder, directory: Path, manifest) -> set[str]:
    snapshot = builder.snapshot
    database = {r["attempt_id"]: r for r in snapshot.rows(sm.BenchmarkExecutionAttemptModel)}
    found = set(database)
    root = directory / "attempts"
    for path in sorted(root.iterdir()) if root.exists() else ():
        if path.is_symlink() or not path.is_dir():
            raise ValueError("unsafe campaign attempt inventory")
        found.add(path.name)
        try:
            attempt = json.loads(
                read_campaign_file(directory, f"attempts/{path.name}/attempt.json")
            )
            terminal_data = read_campaign_file(directory, f"attempts/{path.name}/terminal.json")
            terminal = json.loads(terminal_data)
            log = read_campaign_file(directory, f"attempts/{path.name}/safe-execution.log")
            if (
                attempt.get("campaign_id") != manifest.campaign_id
                or attempt.get("attempt_id") != path.name
                or terminal.get("attempt_id") != path.name
            ):
                raise ValueError("campaign attempt identity mismatch")
            if terminal.get("state") not in _TERMINAL or (log and not log.endswith(b"\n")):
                snapshot.missing("nonterminal_attempt_or_partial_log", {"attempts", "performance"})
            if path.name in database and database[path.name]["status"] != terminal["state"]:
                snapshot.missing(
                    "database_local_attempt_state_mismatch", {"attempts", "performance"}
                )
            if (
                path.name in database
                and database[path.name].get("receipt_hash")
                != hashlib.sha256(terminal_data).hexdigest()
            ):
                snapshot.missing(
                    "database_attempt_receipt_hash_mismatch", {"attempts", "performance"}
                )
            safe_log, log_stats = _safe_log(log)
            if log_stats["rejected_line_count"]:
                snapshot.missing(
                    "attempt_log_contains_unrecognized_lines", {"attempts", "performance"}
                )
            capsule, capsule_problems = _read_failure_capsule(
                directory, path.name, terminal, safe_log
            )
            for problem in capsule_problems:
                snapshot.missing(problem, {"attempts", "failures"})
            if capsule is not None or capsule_problems:
                identity = "attempt-capsule:" + path.name
                builder.expected.add(("failure", identity))
                if capsule is not None:
                    digest = builder.store.put(capsule)
                    builder.store.verify(digest)
                    builder.artifacts[identity] = PrivateAuditArtifact(
                        artifact_id=identity,
                        kind="failure",
                        scope="runtime",
                        sha256=digest,
                        size_bytes=len(capsule),
                    )
                    builder.records.append(
                        PrivateAuditRecord(
                            kind="failure",
                            record_id=identity,
                            attempt_id=path.name,
                            artifacts={"raw": identity},
                        )
                    )
            builder.fact(
                "local-attempt:" + path.name,
                {
                    "attempt": attempt,
                    "terminal": terminal,
                    "log_sha256": hashlib.sha256(log).hexdigest(),
                    "log_byte_count": len(log),
                    "telemetry_scope": "worker_process",
                },
            )
        except (OSError, ValueError, KeyError):
            snapshot.missing("local_attempt_evidence_unavailable", {"attempts", "performance"})
    local_ids = {p.name for p in root.iterdir()} if root.exists() else set()
    if set(database) - local_ids:
        snapshot.missing("database_attempt_local_logs_missing", {"attempts", "performance"})
    if local_ids - set(database):
        snapshot.missing("local_attempt_database_record_missing", {"attempts", "performance"})
    if not found and not (snapshot.conformance and snapshot.runs == snapshot.conformance):
        snapshot.missing("campaign_attempts_absent", {"attempts", "performance"})
    builder.expected.update(("attempt", identity) for identity in found)
    return found


async def _populate(builder: _Builder, directory, manifest, settings, selected_policy=None) -> None:
    snapshot = builder.snapshot
    # Replacement consent is a cost/audit fact, never proof of the lost response.
    for path in sorted((directory / "manifests").glob("conformance-replacement-*.json")):
        record = json.loads(read_campaign_file(directory, str(path.relative_to(directory))))
        if (record.get("identity", {}).get("campaign_id") != manifest.campaign_id
                or record.get("content_hash") != canonical_content_hash({
                    k: v for k, v in record.items() if k != "content_hash"
                })):
            raise ValueError("conformance_replacement_export_identity_mismatch")
        builder.fact("conformance-replacement:" + record["content_hash"], record, "cost_fact")
    attempts = _attempts(builder, directory, manifest)
    _model_wall_timings(builder, manifest, attempts)
    for row in snapshot.rows(bm.RunManifestModel):
        if row["run_id"] not in snapshot.conformance:
            continue
        run_id = row["run_id"]
        try:
            local = json.loads(
                read_campaign_file(builder.store.root, f"conformance/{run_id}/manifest.json")
            )
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            snapshot.missing("conformance_private_manifest_unavailable", {"cost_facts", "failures"})
            continue
        if local != row["manifest_json"]:
            snapshot.missing("conformance_private_manifest_mismatch", {"cost_facts", "failures"})
            continue
        builder.journal_stores.append(
            CampaignArtifactStore(builder.store.root / "conformance" / run_id)
        )
        builder.fact("conformance-manifest:" + run_id, local, "cost_fact")
    for name, rows in sorted(snapshot.tables.items()):
        scope = "evaluation" if name.startswith(EVALUATION_SCHEMA + ".") else "runtime"
        builder.fact("table:" + name, {"table": name, "rows": rows}, audit_table_kind(name), scope)
    for identity, value in sorted(snapshot.cost_dependencies.items()):
        builder.fact(identity, value, "performance", "evaluation")

    if set(snapshot.process_tables) != PROCESS_REQUIRED_TABLES:
        snapshot.missing("process_snapshot_not_enumerated", {"process_facts"})
    process_index = {
        "schema_version": "step8-process-facts-v1",
        "campaign_id": manifest.campaign_id,
        "source_role": "evaluator",
        "database_snapshot": snapshot.identity,
        "scope": snapshot.process_scope,
        "status": "incomplete" if "process_facts" in snapshot.issues else "complete",
        "missing": sorted(snapshot.issues.get("process_facts", ())),
        "required_tables": sorted(PROCESS_REQUIRED_TABLES),
        "tables": {
            name: {
                "status": "available" if name in snapshot.tables else "unavailable",
                "row_count": len(snapshot.tables[name]) if name in snapshot.tables else None,
                "primary_key": [
                    c.name for c in _table(snapshot.process_tables[name]).primary_key.columns
                ]
                if name in snapshot.process_tables
                else [],
                "artifact_id": "table:" + name if name in snapshot.tables else None,
                "sha256": builder.artifacts["table:" + name].sha256
                if name in snapshot.tables
                else None,
            }
            for name in sorted(PROCESS_REQUIRED_TABLES)
        },
    }
    builder.fact("process-facts:index", process_index)

    for row in snapshot.rows(qm.QueryRuntimeTypedAnswerModel):
        typed_payload = row.get("payload", {}).get("metadata", {}).get("typed_result")
        try:
            answer = TypedAnswerResult.model_validate(typed_payload)
            if canonical_content_hash(typed_payload) != row["content_hash"] or (
                answer.typed_answer_id,
                answer.query_id,
                answer.run_id,
                answer.answer_model_call_id,
                answer.terminal_state,
            ) != (
                row["typed_answer_id"],
                row["query_id"],
                row["query_executor_run_id"],
                row["answer_model_call_id"],
                row["terminal_state"],
            ):
                raise ValueError("typed answer lineage mismatch")
        except (ValueError, KeyError):
            snapshot.missing("typed_answer_payload_invalid", {"runtime_answers"})
        else:
            builder.fact("typed-answer:" + answer.typed_answer_id, typed_payload, "runtime_answer")
    gold = {(r["snapshot_id"], r["query_id"]) for r in snapshot.rows(bm.QueryGoldModel)}
    if snapshot.selected - gold:
        snapshot.missing("selected_query_gold_absent", {"gold"})

    calls = [("runtime", r) for r in snapshot.rows(xm.RuntimeModelCallModel)] + [
        ("evaluation", r) for r in snapshot.rows(xm.EvaluationModelCallModel)
    ]
    semantic_ledgers = {
        r["evaluation_run_id"]
        for r in snapshot.rows(em.EvaluationRunModel)
        if r.get("metadata", {}).get("semantic_pair_ledger") is True
    }
    for scope in ("runtime", "evaluation"):
        builder.fact(
            "model-call-performance:" + scope,
            {
                "database_observation_is_pool_wait": False,
                "calls": [
                    {
                        "model_call_id": row["model_call_id"],
                        "run_id": row["run_id"],
                        "latency_ms": row.get("latency_ms"),
                        "latency_ms_semantics": "excludes_queue_and_artifact_persistence"
                        if {"queue_wait_ms", "artifact_persistence_ms", "end_to_end_ms"}
                        <= (row.get("request_config") or {}).keys()
                        else "legacy_unspecified",
                        **{
                            key: (row.get("request_config") or {}).get(key)
                            for key in ("queue_wait_ms", "artifact_persistence_ms", "end_to_end_ms")
                        },
                    }
                    for call_scope, row in calls
                    if call_scope == scope
                ],
            },
            "performance",
            scope,
        )
    references = {("runtime", r["artifact_id"]): r for r in snapshot.rows(xm.RuntimeArtifactModel)}
    references.update(
        {("evaluation", r["artifact_id"]): r for r in snapshot.rows(xm.EvaluationArtifactModel)}
    )
    run_rows = {r["run_id"]: r for r in snapshot.rows(bm.RunManifestModel)}
    assessed = {r["model_call_id"] for r in snapshot.rows(qm.QBankRerankerAssessmentModel)}
    call_ids = {r["model_call_id"] for _, r in calls}
    if assessed - call_ids:
        snapshot.missing("reranker_assessment_call_absent", {"reranker_calls"})
    strata = defaultdict(list)
    for scope, row in calls:
        if row["status"] == "success" and not (
            scope == "runtime"
            and row["model_call_id"] in (snapshot.dependency_calls | snapshot.adopted_calls)
        ):
            run = run_rows.get(row["run_id"], {})
            strata[
                (scope, run.get("dataset"), run.get("method"), row["stage"], row["model_alias"])
            ].append(row["model_call_id"])
    samples = {(key[0], min(ids, key=canonical_content_hash)) for key, ids in strata.items()}
    builder.fact(
        "review-sampling-policy",
        {
            "policy": "one-hash-ordered-success-per-scope-dataset-method-stage-model",
            "conformance_success_policy": "all_calls_research_only",
            "semantic_pair_ledger_success_policy": "all_calls",
            "strata": [
                {
                    "key": list(key),
                    "count": len(ids),
                    "sample": min(ids, key=canonical_content_hash),
                }
                for key, ids in sorted(strata.items(), key=lambda pair: str(pair[0]))
            ],
        },
        "review_payload",
        "evaluation",
    )
    stores = None
    materialized: dict[tuple[str, str], str] = {}

    async def payload(scope, artifact_id, families):
        nonlocal stores
        key = scope, artifact_id
        if key in materialized:
            return materialized[key]
        row = references.get(key)
        if row is None or row.get("state") != "ready" or row.get("size_bytes") is None:
            snapshot.missing("required_artifact_not_ready", families)
            return None
        try:
            try:
                path = builder.store.verify(row["sha256"])
            except (ValueError, OSError):
                if stores is None:
                    runtime, evaluation = create_evaluator_object_stores(settings)
                    stores = {"runtime": runtime, "evaluation": evaluation}
                data = await stores[scope].get_bytes(row["object_key"])
                if (
                    len(data) != row["size_bytes"]
                    or hashlib.sha256(data).hexdigest() != row["sha256"]
                ):
                    raise ValueError("object checksum mismatch") from None
                builder.store.put(data)
                path = builder.store.verify(row["sha256"])
            if path.stat().st_size != row["size_bytes"]:
                raise ValueError("object size mismatch")
            item = PrivateAuditArtifact(
                artifact_id=scope + ":" + artifact_id,
                kind="model_payload",
                scope=scope,
                sha256=row["sha256"],
                size_bytes=row["size_bytes"],
                object_key=row["object_key"],
            )
        except Exception as exc:
            # Failure detail must not echo DSNs, provider bodies or object-store credentials.
            snapshot.missing("private_payload_unavailable:" + type(exc).__name__, families)
            return None
        builder.artifacts[item.artifact_id] = item
        materialized[key] = item.artifact_id
        return item.artifact_id

    for (scope, artifact_id), artifact in references.items():
        if artifact["run_id"] in snapshot.conformance or (
            scope == "evaluation" and artifact.get("evaluation_run_id") in semantic_ledgers
        ):
            await payload(scope, artifact_id, {"review_samples", "cost_facts"})

    from pmmc.evaluation.step8_oracle_analysis import materialize_oracle_inventory

    await materialize_oracle_inventory(builder, manifest, payload)

    # Reused cache producers can predate this campaign and have no campaign worker attempt.
    # Their exact consumed calls remain private, independently checksum-verified dependencies.
    for scope, row in calls:
        if scope != "runtime" or row["model_call_id"] not in (
            snapshot.embedding_calls | snapshot.dependency_calls | snapshot.adopted_calls
        ):
            continue
        families = {"process_facts", "cost_facts", "review_samples"}
        input_ref = await payload(scope, row.get("input_asset_id"), families)
        output_ref = (
            await payload(scope, row["output_asset_id"], families)
            if row.get("output_asset_id")
            else None
        )
        error_id = row.get("metadata", {}).get("error_artifact_id")
        error_ref = await payload(scope, error_id, families) if error_id else None
        if not input_ref or not (output_ref or error_ref):
            snapshot.missing("embedding_dependency_payload_absent", families)
        builder.fact(
            "embedding-producer:" + row["model_call_id"],
            {
                "call": row,
                "input": input_ref,
                "output": output_ref,
                "error": error_ref,
                "external_dependency": row["model_call_id"] in snapshot.dependency_calls,
                "adopted_namespace_projection": row["model_call_id"] in snapshot.adopted_calls,
            },
            "cost_fact",
        )

    for reservation in snapshot.rows(sm.BenchmarkSpendReservationModel):
        request_id = reservation["request_id"]
        attempt, envelope_ref = _provider_journal(builder, manifest, request_id)
        envelope = (
            builder.store.load_json(builder.artifacts[envelope_ref].sha256) if envelope_ref else {}
        )
        failed = reservation.get("state") != "settled" or bool(envelope.get("error"))
        record_id = "provider-reservation:" + request_id
        if failed:
            builder.expected.add(("failure", record_id))
        conformance_run = attempt if attempt in snapshot.conformance else None
        if not envelope_ref or (attempt not in attempts and conformance_run is None):
            snapshot.missing("provider_reservation_journal_unavailable", {"cost_facts", "failures"})
            continue
        if failed:
            raw = builder.fact(
                record_id, {"reservation": reservation, "envelope": envelope}, "failure"
            )
            builder.records.append(
                PrivateAuditRecord(
                    kind="failure",
                    record_id=record_id,
                    attempt_id=attempt if conformance_run is None else None,
                    conformance_run_id=conformance_run,
                    artifacts={"raw": raw, "payload": envelope_ref},
                )
            )

    for scope, row in calls:
        call_id = row["model_call_id"]
        if scope == "runtime" and call_id in (snapshot.dependency_calls | snapshot.adopted_calls):
            continue
        conformance_run = row["run_id"] if row["run_id"] in snapshot.conformance else None
        # Step 8 uses one provider attempt; format retries are separate DB calls.
        if (row.get("retry_count") or 0) > 0:
            snapshot.missing("provider_retry_payloads_not_individually_persisted", {"failures"})
        reranker = call_id in assessed or row["model_alias"] == manifest.protocol.reranker_alias
        failed = row["status"] != "success"
        sample = (
            (conformance_run is not None and not failed)
            or (
                scope == "evaluation"
                and not failed
                and row.get("evaluation_run_id") in semantic_ledgers
            )
            or (scope, call_id) in samples
        )
        families = {
            family
            for family, needed in (
                ("reranker_calls", reranker),
                ("failures", failed),
                ("review_samples", sample),
            )
            if needed
        }
        if not families:
            continue
        record_id = call_id if scope == "runtime" else "evaluation:" + call_id
        kinds = [
            kind
            for kind, needed in (
                ("reranker_call", reranker),
                ("failure", failed),
                ("review_sample", sample),
            )
            if needed
        ]
        builder.expected.update((kind, record_id) for kind in kinds)
        metadata = row.get("metadata", {})
        if any(
            metadata.get(key) not in (None, manifest.campaign_id)
            for key in ("step8_campaign_id", "campaign_id")
        ):
            snapshot.missing("model_call_campaign_lineage_conflict", families | {"attempts"})
            continue
        attempt_id = _bound_attempt(row, attempts, snapshot)
        if conformance_run:
            attempt_id = None
            if (
                _explicit_attempt(row) != conformance_run
                or row.get("query_id") is not None
                or row.get("snapshot_id") is not None
            ):
                snapshot.missing("conformance_call_scope_mismatch", families)
                continue
        input_ref = await payload(scope, row.get("input_asset_id"), families)
        output_ref = (
            await payload(scope, row.get("output_asset_id"), families)
            if row.get("output_asset_id")
            else None
        )
        error_id = row.get("metadata", {}).get("error_artifact_id")
        error_ref = await payload(scope, error_id, families) if error_id else None
        if input_ref and scope == "runtime":
            try:
                request = builder.store.load_json(builder.artifacts[input_ref].sha256)
            except (ValueError, OSError):
                request = {}
            journal_attempt, journal_ref = _provider_journal(
                builder, manifest, request.get("request_id")
            )
            if (
                journal_attempt
                and (attempt_id or conformance_run)
                and journal_attempt != (attempt_id or conformance_run)
            ):
                snapshot.missing("provider_and_call_attempt_conflict", families | {"attempts"})
                continue
            if conformance_run is None:
                attempt_id = attempt_id or journal_attempt
            if failed:
                error_ref = error_ref or journal_ref
        if conformance_run is None and attempt_id not in attempts:
            snapshot.missing("model_call_campaign_attempt_unproven", families | {"attempts"})
            continue
        if not input_ref or not (output_ref or error_ref):
            snapshot.missing("raw_call_input_or_output_failure_absent", families)
            continue
        lineage = builder.fact(
            "lineage:" + record_id,
            {
                **row,
                "model_call_id": record_id,
                "original_model_call_id": call_id,
                "attempt_id": attempt_id,
                "conformance_run_id": conformance_run,
                "scope": scope,
                "input_sha256": builder.artifacts[input_ref].sha256,
                "output_sha256": builder.artifacts[output_ref].sha256 if output_ref else None,
                "failure_sha256": builder.artifacts[error_ref].sha256 if error_ref else None,
            },
            "reranker" if reranker else "model_payload",
            scope,
        )
        if reranker:
            refs = {"lineage": lineage, "input": input_ref}
            refs["output" if output_ref else "failure"] = output_ref or error_ref
            builder.records.append(
                PrivateAuditRecord(
                    kind="reranker_call",
                    record_id=record_id,
                    attempt_id=attempt_id,
                    conformance_run_id=conformance_run,
                    artifacts=refs,
                )
            )

        for kind in ("failure", "review_sample"):
            if kind not in kinds:
                continue
            raw = builder.fact(
                kind + ":" + record_id,
                {
                    "call": row,
                    "attempt_id": attempt_id,
                    "conformance_run_id": conformance_run,
                    "input_sha256": builder.artifacts[input_ref].sha256,
                    "output_sha256": builder.artifacts[output_ref].sha256 if output_ref else None,
                    "failure_sha256": builder.artifacts[error_ref].sha256 if error_ref else None,
                },
                "failure" if kind == "failure" else "review_payload",
                scope,
            )
            refs = {"raw": raw, "input": input_ref, "lineage": lineage}
            if output_ref:
                refs["output"] = output_ref
            if error_ref:
                refs["failure"] = error_ref
            builder.records.append(
                PrivateAuditRecord(
                    kind=kind,
                    record_id=record_id,
                    attempt_id=attempt_id,
                    conformance_run_id=conformance_run,
                    artifacts=refs,
                )
            )

    by_call = {(scope, row["model_call_id"]): row for scope, row in calls}
    failure_models = (
        sm.BenchmarkStageJobModel,
        sm.BenchmarkExecutionAttemptModel,
        xm.RunStateModel,
        xm.ExperimentJobModel,
        xm.JobAttemptModel,
        cm.CompilerStageJobModel,
        dm.CandidateStageJobModel,
        cm.ProgramExecutionModel,
        qm.QueryExecutorStageJobModel,
        em.EvaluationRunStateModel,
        em.EvaluationJobModel,
        em.EvaluationJobAttemptModel,
        qm.AnswerFormatDiagnosticModel,
        questioner.RuntimeStructuredOutputItemDiagnosticModel,
        questioner.EvaluationStructuredOutputItemDiagnosticModel,
    )
    for model in failure_models:
        scope = "evaluation" if _table(model).schema == EVALUATION_SCHEMA else "runtime"
        for row in snapshot.rows(model):
            if not (
                row.get("status") in {"failed", "unknown", "invalid", "conservative_reject"}
                or row.get("error_type")
                or row.get("accepted") is False
            ):
                continue
            identity = "db-failure:" + canonical_content_hash([_table(model).fullname, row])
            builder.expected.add(("failure", identity))
            attempt = _bound_attempt(row, attempts, snapshot)
            conformance_run = (
                row.get("run_id") if row.get("run_id") in snapshot.conformance else None
            )
            if not attempt and row.get("model_call_id"):
                linked_call = by_call.get((scope, row["model_call_id"]), {})
                attempt = _bound_attempt(linked_call, attempts, snapshot)
                if linked_call.get("run_id") in snapshot.conformance:
                    conformance_run = linked_call["run_id"]
            if attempt not in attempts and conformance_run is None:
                snapshot.missing("failure_campaign_attempt_unproven", {"failures", "attempts"})
                continue
            raw = builder.fact(
                identity, {"table": _table(model).fullname, "row": row}, "failure", scope
            )
            refs = {"raw": raw}
            error_id = row.get("error_artifact_id")
            if error_id:
                error_ref = await payload(scope, error_id, {"failures"})
                if not error_ref:
                    continue
                refs["payload"] = error_ref
            builder.records.append(
                PrivateAuditRecord(
                    kind="failure",
                    record_id=identity,
                    attempt_id=attempt if conformance_run is None else None,
                    conformance_run_id=conformance_run,
                    artifacts=refs,
                )
            )

    await _produce_threshold_facts(
        builder, directory, manifest, settings, payload, selected_policy=selected_policy
    )


class _ThresholdUnavailable(ValueError):
    """A safe, query-free coverage reason, not a guessed scientific outcome."""


class _ThresholdProducer:
    def __init__(self, builder, manifest, payload, config) -> None:
        self.builder = builder
        self.snapshot = builder.snapshot
        self.manifest = manifest
        self.payload = payload
        self.config = config
        self.scores: dict[tuple[str, str], list[tuple[dict, str, str]]] = defaultdict(list)
        self.evaluation_identities: dict[str, dict] = {}
        self.evaluation_reports: dict[str, dict] = {}
        self.cells = {
            r["cell_result_id"]: r
            for r in self.snapshot.rows(qm.QueryExperimentCellResultModel)
            if "cell_result_id" in r
        }
        self.answers = {
            r["typed_answer_id"]: r for r in self.snapshot.rows(qm.QueryRuntimeTypedAnswerModel)
        }
        self.calls = {r["model_call_id"]: r for r in self.snapshot.rows(xm.RuntimeModelCallModel)}
        self.answer_cache: dict[str, tuple[Any, dict]] = {}
        self.match_cache: dict[str, tuple] = {}
        self.source_cache: dict[tuple[str, str], tuple] = {}

    def save(self, value, kind="variant_cell_reuse", scope="runtime") -> str:
        identity = "threshold:" + canonical_content_hash(value)
        if identity in self.builder.artifacts:
            return self.builder.artifacts[identity].sha256
        ref = self.builder.fact(identity, value, kind, scope)
        return self.builder.artifacts[ref].sha256

    async def artifact(self, scope, identity) -> str:
        # Threshold coverage is separate from completeness of the audit inventory.
        ref = await self.payload(scope, identity, set())
        if not ref:
            raise _ThresholdUnavailable("required_payload_unavailable")
        return self.builder.artifacts[ref].sha256

    async def load_scores(self) -> None:
        runs = {r["evaluation_run_id"]: r for r in self.snapshot.rows(em.EvaluationRunModel)}
        for artifact in self.snapshot.rows(xm.EvaluationArtifactModel):
            if not artifact["artifact_id"].startswith(("step8-scores-", "step8-incomplete-")):
                continue
            try:
                digest = await self.artifact("evaluation", artifact["artifact_id"])
                report = self.builder.store.load_json(digest)
                run = runs[report["evaluation_run_id"]]
                identity = run["metadata"]
                if (
                    report.get("schema_version") != "step8-query-evaluation-v1"
                    or report["campaign_id"] != self.manifest.campaign_id
                    or identity["campaign_id"] != self.manifest.campaign_id
                    or identity["protocol_hash"] != self.manifest.protocol.content_hash
                    or canonical_content_hash(identity) != run["config_hash"]
                    or report["identity_hash"] != run["config_hash"]
                    or report["query_id"] != identity["query_id"]
                    or artifact["query_id"] != report["query_id"]
                    or artifact["evaluation_run_id"] != run["evaluation_run_id"]
                    or report["query_executor_run_id"] != identity["source_run_id"]
                    or artifact["run_id"] != identity["source_run_id"]
                ):
                    continue
                sources = {s["cell_id"]: s for s in identity["sources"]}
                self.evaluation_identities[digest] = identity
                self.evaluation_reports[digest] = report
                for score in report["scores"]:
                    source = sources.get(score["cell_id"], {})
                    if (
                        score.get("status") not in {"answered", "abstention"}
                        or score.get("engineering_error")
                        or score.get("unavailable_reason")
                        or score.get("metric") != "typed_exact_match"
                        or source.get("source_id") != score.get("source_fact_id")
                        or source.get("source_hash") != score.get("source_fact_hash")
                        or source.get("terminal_state") != score.get("terminal_state")
                    ):
                        continue
                    item = score, digest, report["query_executor_run_id"]
                    self.scores[score["cell_id"], report["query_id"]].append(item)
            except (KeyError, ValueError, OSError, TypeError):
                continue

    def source(self, cell, query):
        from pmmc.evaluation.step8_workflow import _validate_cell_row
        from pmmc.schemas.query_executor import QueryExperimentCellResult

        if (cell.cell_id, query) in self.source_cache:
            return self.source_cache[cell.cell_id, query]
        records = self.scores.get((cell.cell_id, query), [])
        unique = {canonical_content_hash(r[0]): r for r in records}
        if len(unique) != 1:
            raise _ThresholdUnavailable("saved_score_missing_or_conflicting")
        score, digest, run_id = next(iter(unique.values()))
        row = self.cells.get(score.get("source_fact_id"))
        if row is None:
            raise _ThresholdUnavailable("runtime_cell_missing")
        _validate_cell_row(qm.QueryExperimentCellResultModel(**row))
        fact = QueryExperimentCellResult.model_validate(row["payload"])
        treatment = {
            "active": "pmmc" if cell.treatment.value == "main" else cell.treatment.value,
            "no_fallback": "no_fallback",
            "forced_top1": "forced_top1_counterfactual",
        }[cell.source_variant]
        if (
            fact.query_executor_run_id != run_id
            or fact.query_id != query
            or fact.content_hash != score["source_fact_hash"]
            or fact.selected_answer_id != score["typed_answer_id"]
            or fact.terminal_state != score["terminal_state"]
            or (fact.cell.treatment, fact.cell.question_budget, fact.cell.revision_depth)
            != (treatment, cell.question_budget, cell.round_id)
            or score.get("method") != cell.method
            or score.get("source_variant") != cell.source_variant
        ):
            raise _ThresholdUnavailable("source_cell_lineage_mismatch")
        self.source_cache[cell.cell_id, query] = fact, score, digest
        return fact, score, digest

    async def call_proof(self, call_id, run_id, query, *, allow_failure=False) -> dict:
        row = self.calls.get(call_id)
        if (
            row is None
            or (row.get("run_id"), row.get("query_id"))
            != (
                run_id,
                query,
            )
            or (row.get("status") != "success" and not allow_failure)
        ):
            raise _ThresholdUnavailable("answer_or_reranker_call_missing")
        input_sha = await self.artifact("runtime", row.get("input_asset_id"))
        output_sha = (
            await self.artifact("runtime", row["output_asset_id"])
            if row.get("output_asset_id")
            else None
        )
        failure_sha = None
        if row.get("status") != "success":
            failure_sha = await self.artifact(
                "runtime", row.get("metadata", {}).get("error_artifact_id")
            )
        elif output_sha is None:
            raise _ThresholdUnavailable("call_output_payload_missing")
        if row.get("prompt_hash") != input_sha:
            raise _ThresholdUnavailable("request_trace_hash_mismatch")
        return {
            "trace_sha256": self.save(row, "cost_fact"),
            "input_sha256": input_sha,
            "output_sha256": output_sha,
            "failure_sha256": failure_sha,
        }

    async def answer(self, identity, query, expected_run, method, dataset, answer_model):
        from pmmc.evaluation.step8_workflow import _decode_answer
        from pmmc.runtime.grounding import grounding_evidence_hash
        from pmmc.runtime.relation_grounding import grounding_request_scope_aliases
        from pmmc.schemas import GenerationRequest
        from pmmc.schemas.answer_output import TypedAnswerContext

        if not identity or identity not in self.answers:
            raise _ThresholdUnavailable("typed_answer_not_generated")
        if identity not in self.answer_cache:
            row = self.answers[identity]
            answer = _decode_answer(qm.QueryRuntimeTypedAnswerModel(**row))
            proofs = {}
            for call in answer.calls:
                proofs[call.model_call_id] = await self.call_proof(
                    call.model_call_id, answer.run_id, query
                )
            request = self.builder.store.load_json(
                proofs[answer.answer_model_call_id]["input_sha256"]
            )
            context_payload = request.get("metadata", {}).get("typed_answer_context")
            context = TypedAnswerContext.model_validate(context_payload)
            grounding_aliases = list(answer.ungated_output.used_evidence_aliases)
            if answer.grounding:
                scopes = []
                request_contracts = []
                try:
                    query_hash = request.get("metadata", {}).get("query_content_hash")
                    if not isinstance(query_hash, str) or not query_hash:
                        raise ValueError("grounding_query_hash_missing")
                    for call in answer.grounding.calls:
                        grounding_request = GenerationRequest.model_validate(
                            self.builder.store.load_json(proofs[call.model_call_id]["input_sha256"])
                        )
                        scopes.append(
                            grounding_request_scope_aliases(
                                grounding_request,
                                answer.ungated_output,
                                context,
                                expected_query_hash=query_hash,
                            )
                        )
                        request_contracts.append(
                            canonical_content_hash(
                                {
                                    **grounding_request.model_dump(
                                        mode="json",
                                        exclude={"request_id", "metadata", "messages"},
                                    ),
                                    "messages": [
                                        message.model_dump(mode="json")
                                        for message in grounding_request.messages[:2]
                                    ],
                                    "runtime_relation_contract_hash": (
                                        grounding_request.metadata.get(
                                            "runtime_relation_contract_hash"
                                        )
                                    ),
                                }
                            )
                        )
                    if scopes and any(scope != scopes[0] for scope in scopes):
                        raise ValueError("grounding_retry_scope_changed")
                    if len(set(request_contracts)) > 1:
                        raise ValueError("grounding_retry_contract_changed")
                except ValueError as exc:
                    raise _ThresholdUnavailable("grounding_relation_scope_mismatch") from exc
                if scopes:
                    grounding_aliases = scopes[0]
            trace = self.calls[answer.answer_model_call_id]
            if (
                trace.get("request_config", {})
                .get("request_metadata", {})
                .get("typed_answer_context")
                != context_payload
            ):
                raise _ThresholdUnavailable("typed_context_request_trace_mismatch")
            if answer.grounding and (
                answer.grounding.ungated_output_hash
                != canonical_content_hash(answer.ungated_output.model_dump(mode="json"))
                or answer.grounding.used_evidence_aliases
                != answer.ungated_output.used_evidence_aliases
                or answer.grounding.evidence_content_hash
                != grounding_evidence_hash(context, grounding_aliases)
                or not answer.grounding.calls
                or answer.grounding.model_alias != self.manifest.protocol.grounding_alias
                or any(
                    (
                        self.calls[c.model_call_id].get("model_alias"),
                        self.calls[c.model_call_id].get("actual_model_id"),
                    )
                    != (answer.grounding.model_alias, answer.grounding.actual_model_id)
                    for c in answer.grounding.calls
                )
            ):
                raise _ThresholdUnavailable("grounding_context_hash_mismatch")
            proof = {
                "typed_answer_sha256": self.save(
                    row["payload"]["metadata"]["typed_result"], "runtime_answer"
                ),
                "typed_fact_sha256": self.save(row, "runtime_answer"),
                "answer_identity_hash": row["answer_identity_hash"],
                "typed_context_sha256": self.save(context.model_dump(mode="json"), "model_payload"),
                "calls": proofs,
                "grounding": runtime_fact_payload(answer.grounding) if answer.grounding else None,
            }
            self.answer_cache[identity] = answer, proof
        answer, proof = self.answer_cache[identity]
        if (answer.query_id, answer.run_id, answer.method, answer.snapshot_id) != (
            query,
            expected_run,
            method,
            self.manifest.snapshot_ids[dataset],
        ) or answer.terminal_state not in {"answered", "abstained"}:
            raise _ThresholdUnavailable("typed_answer_scope_or_terminal_mismatch")
        trace = self.calls[answer.answer_model_call_id]
        if (trace.get("model_alias"), trace.get("actual_model_id")) != (
            answer_model,
            answer.actual_model_id,
        ):
            raise _ThresholdUnavailable("typed_answer_backbone_mismatch")
        return answer, proof

    def native_score(self, definition, query, chosen, report_sha):
        """Reuse only the exact native answer in the original method-blind evaluation."""
        from pmmc.evaluation.step8_workflow import _decode_answer

        records = [
            r for r in self.scores.get((definition.cell_id, query), []) if r[1] == report_sha
        ]
        if len(records) != 1:
            raise _ThresholdUnavailable("native_score_missing_or_conflicting")
        score, _, _ = records[0]
        row = self.answers.get(chosen)
        if row is None:
            raise _ThresholdUnavailable("typed_answer_not_generated")
        answer = _decode_answer(qm.QueryRuntimeTypedAnswerModel(**row))
        if (
            (score.get("source_fact_id"), score.get("typed_answer_id")) != (chosen, chosen)
            or score.get("source_fact_hash") != row["content_hash"]
            or (score.get("method"), score.get("source_variant"))
            != ("native_multimodal_rag", "primary")
            or (answer.method, answer.answer_variant, answer.query_id)
            != ("native_multimodal_rag", "primary", query)
            or score.get("terminal_state") != answer.terminal_state
        ):
            raise _ThresholdUnavailable("native_score_answer_lineage_mismatch")
        identity = self.evaluation_identities[report_sha]
        if (
            identity.get("snapshot_id") != self.manifest.snapshot_ids[definition.dataset]
            or identity.get("judge_config", {}).get("judge_model_alias")
            != self.manifest.protocol.judge_alias
            or any(
                not isinstance(identity.get(key), str) or len(identity[key]) != 64
                for key in ("query_hash", "gold_hash", "judge_prompt_hash", "judge_model_hash")
            )
        ):
            raise _ThresholdUnavailable("native_score_evaluation_protocol_unproven")
        return row, score, report_sha

    async def matching(self, fact, cell):
        from pmmc.runtime.qbank_matcher import decide_runtime_route
        from pmmc.schemas import QBankRouteClass
        from pmmc.schemas.query_executor import QBankMatchBatch, QBankMatchFusedCandidate

        if fact.cell_result_id in self.match_cache:
            return self.match_cache[fact.cell_result_id]
        batches = [
            r
            for r in self.snapshot.rows(qm.QBankMatchBatchModel)
            if r["match_batch_id"] == fact.route.match_batch_id
        ]
        if len(batches) != 1:
            raise _ThresholdUnavailable("match_batch_missing")
        batch = QBankMatchBatch.model_validate(batches[0])
        if (
            batch.query_executor_run_id,
            batch.query_id,
            batch.question_budget,
            batch.active_round,
        ) != (fact.query_executor_run_id, fact.query_id, cell.question_budget, cell.round_id):
            raise _ThresholdUnavailable("match_batch_scope_mismatch")
        classes = {value.value: i for i, value in enumerate(QBankRouteClass)}
        candidates = tuple(
            sorted(
                (
                    QBankMatchFusedCandidate.model_validate(r)
                    for r in self.snapshot.rows(qm.QBankMatchFusedCandidateModel)
                    if r["match_batch_id"] == batch.match_batch_id
                ),
                key=lambda r: (classes[r.route_class.value], r.rank),
            )
        )
        channels = sorted(
            (
                r
                for r in self.snapshot.rows(qm.QBankMatchChannelCandidateModel)
                if r["match_batch_id"] == batch.match_batch_id
            ),
            key=lambda r: (r["channel"], classes[r["route_class"]], r["rank"]),
        )
        batch_payload = {
            "run_id": batch.query_executor_run_id,
            "query_id": batch.query_id,
            "projection_id": batch.routing_projection_id,
            "question_budget": batch.question_budget,
            "round_id": batch.active_round,
            "channel_hashes": [r["content_hash"] for r in channels],
            "fused_hashes": [r.content_hash for r in candidates],
        }
        if (
            len(candidates) != batch.fused_count
            or canonical_content_hash(batch_payload) != batch.content_hash
        ):
            raise _ThresholdUnavailable("match_candidate_denominator_or_hash_mismatch")
        config = type(self.config).model_validate(
            {
                **self.config.model_dump(),
                "treatment": cell.treatment.value,
                "active_match_threshold": self.manifest.protocol.match_threshold,
                "active_gap_threshold": self.manifest.protocol.gap_threshold,
                "no_reranker_match_threshold": self.manifest.protocol.no_reranker_match_threshold,
                "no_reranker_gap_threshold": self.manifest.protocol.no_reranker_gap_threshold,
            }
        )
        rerank = []
        for candidate in candidates:
            if candidate.query_id != fact.query_id:
                raise _ThresholdUnavailable("candidate_query_mismatch")
            if config.reranker_enabled:
                if (
                    canonical_content_hash(
                        candidate.model_dump(mode="json", exclude={"content_hash", "metadata"})
                    )
                    != candidate.content_hash
                ):
                    raise _ThresholdUnavailable("candidate_content_hash_mismatch")
                assessment_id = candidate.metadata.get("reranker_assessment_id")
                if assessment_id:
                    assessments = [
                        r
                        for r in self.snapshot.rows(qm.QBankRerankerAssessmentModel)
                        if r["reranker_assessment_id"] == assessment_id
                    ]
                    if len(assessments) != 1:
                        raise _ThresholdUnavailable("reranker_assessment_missing")
                    assessed = assessments[0]
                    if (
                        assessed["query_executor_run_id"],
                        assessed["query_id"],
                        assessed["qbank_id"],
                        assessed["match_level"],
                        assessed["reranker_score"],
                    ) != (
                        fact.query_executor_run_id,
                        fact.query_id,
                        candidate.qbank_id,
                        candidate.reranker_match_level.value,
                        candidate.reranker_score,
                    ):
                        raise _ThresholdUnavailable("reranker_candidate_mismatch")
                    rerank.append(
                        {
                            "assessment": assessed,
                            "call": await self.call_proof(
                                assessed["model_call_id"],
                                fact.query_executor_run_id,
                                fact.query_id,
                                allow_failure=True,
                            ),
                        }
                    )
            elif candidate.confidence != min(1.0, max(0.0, 61.0 * candidate.raw_rrf)):
                raise _ThresholdUnavailable("no_reranker_score_mismatch")
        route = decide_runtime_route(
            query_executor_run_id=fact.query_executor_run_id,
            query_id=fact.query_id,
            match_batch_id=batch.match_batch_id,
            candidates=candidates,
            config=config,
        )
        if route.model_dump(exclude={"created_at"}) != fact.route.model_dump(
            exclude={"created_at"}
        ):
            raise _ThresholdUnavailable("frozen_route_replay_mismatch")
        result = (
            candidates,
            config,
            {
                "batch": batches[0],
                "channels": channels,
                "candidates": [r.model_dump(mode="json") for r in candidates],
                "reranker": rerank,
            },
        )
        self.match_cache[fact.cell_result_id] = result
        return result

    async def point(self, cell, query, axis, thresholds, frozen, universe):
        if (axis.endswith("relevance") and thresholds != frozen) or axis in {
            "relevance_grid",
            "approved_policy",
        }:
            return await self.relevance_point(cell, query, axis, thresholds, universe)
        from pmmc.runtime.qbank_matcher import decide_runtime_route
        from pmmc.runtime.query_executor_config import JOINT_THRESHOLD

        fact, score, report_sha = self.source(cell, query)
        if fact.relevance is not None and (
            fact.relevance.relevance_threshold != self.manifest.protocol.text_relevance_threshold
            or any(
                item.threshold
                != (
                    {"joint": JOINT_THRESHOLD,
                     "multimodal": self.manifest.protocol.multimodal_relevance_threshold}.get(
                        item.metadata.get("embedding_channel"),
                        self.manifest.protocol.text_relevance_threshold,
                    )
                )
                for item in fact.relevance_items
            )
        ):
            raise _ThresholdUnavailable("frozen_relevance_policy_mismatch")
        candidates, config, matching = await self.matching(fact, cell)
        route = fact.route
        if axis == "match_gap":
            config = type(config).model_validate(
                {
                    **config.model_dump(),
                    "active_match_threshold": thresholds[0],
                    "active_gap_threshold": thresholds[1],
                    "no_reranker_match_threshold": thresholds[0],
                    "no_reranker_gap_threshold": thresholds[1],
                }
            )
            route = decide_runtime_route(
                query_executor_run_id=fact.query_executor_run_id,
                query_id=query,
                match_batch_id=fact.route.match_batch_id,
                candidates=candidates,
                config=config,
            )
        enabled = (
            bool(thresholds[0])
            if axis == "fallback_on_off"
            else cell.source_variant != "no_fallback"
        )
        allowed = (
            route.outcome.value == "program"
            and fact.sufficiency is not None
            and fact.sufficiency.sufficient
        )
        if allowed and (
            fact.binding is None
            or fact.relevance is None
            or not fact.relevance.passed
            or fact.binding.plan_id != route.selected_plan_id
            or fact.binding.qbank_id != route.selected_qbank_id
        ):
            raise _ThresholdUnavailable("candidate_program_evidence_gate_unproven")
        pmmc = self.answers.get(fact.pmmc_answer_id, {})
        terminal_pmmc = pmmc.get("terminal_state") in {"answered", "abstained"}
        if allowed and fact.pmmc_answer_id and not pmmc:
            raise _ThresholdUnavailable("referenced_pmmc_answer_missing")
        if allowed and not terminal_pmmc and pmmc.get("terminal_state") != "fallback_required":
            raise _ThresholdUnavailable("candidate_pmmc_answer_not_generated")
        chosen = (
            fact.pmmc_answer_id
            if allowed and terminal_pmmc
            else fact.fallback_answer_id
            if enabled
            else None
        )
        sources = []
        for definition in universe:
            if (
                chosen is not None
                and chosen == fact.fallback_answer_id
                and enabled
                and definition.method == "native_multimodal_rag"
                and definition.source_variant == "active"
                and (definition.dataset, definition.answer_model, definition.treatment)
                == (cell.dataset, cell.answer_model, cell.treatment)
            ):
                try:
                    sources.append(self.native_score(definition, query, chosen, report_sha))
                except (KeyError, ValueError, TypeError):
                    pass
                continue
            if (
                definition.dataset,
                definition.answer_model,
                definition.treatment,
                definition.question_budget,
                definition.round_id,
                definition.method,
            ) != (
                cell.dataset,
                cell.answer_model,
                cell.treatment,
                cell.question_budget,
                cell.round_id,
                "pmmc",
            ):
                continue
            try:
                other, existing, existing_sha = self.source(definition, query)
            except (KeyError, ValueError, TypeError):
                continue
            if (
                other.query_executor_run_id == fact.query_executor_run_id
                and other.selected_answer_id == chosen
                and existing_sha == report_sha
            ):
                if chosen is not None or (
                    other.cell.treatment == "no_fallback" and other.terminal_state == "no_answer"
                ):
                    sources.append((other, existing, existing_sha))
        if not sources:
            raise _ThresholdUnavailable("selected_answer_score_not_saved")
        selected, saved_score, saved_sha = sources[0]
        if (
            len(
                {
                    canonical_content_hash(
                        {
                            k: s[1].get(k)
                            for k in (
                                "typed_answer_id",
                                "status",
                                "terminal_state",
                                "value",
                                "applicable",
                                "answer_source_method",
                            )
                        }
                    )
                    for s in sources
                }
            )
            != 1
        ):
            raise _ThresholdUnavailable("selected_answer_scores_conflict")
        answer_sha = None
        answer_proof = None
        if chosen is not None:
            expected_method = "pmmc" if chosen == fact.pmmc_answer_id else "native_multimodal_rag"
            expected_run = (
                fact.query_executor_run_id
                if expected_method == "pmmc"
                else self.answers.get(chosen, {}).get("query_executor_run_id")
            )
            answer, answer_proof = await self.answer(
                chosen, query, expected_run, expected_method, cell.dataset, cell.answer_model
            )
            if (answer.method, answer.answer_variant, answer.terminal_state) != (
                saved_score["answer_source_method"],
                saved_score["answer_source_variant"],
                saved_score["terminal_state"],
            ):
                raise _ThresholdUnavailable("answer_score_lineage_mismatch")
            answer_sha = answer_proof["typed_answer_sha256"]
        elif enabled or saved_score["status"] != "abstention" or saved_score["value"] != 0.0:
            raise _ThresholdUnavailable("terminal_refusal_not_observed")
        source_sha = self.save(runtime_fact_payload(fact))
        terminal_sha = self.save(runtime_fact_payload(selected))
        context = {
            "schema_version": "step8-threshold-context-v2",
            "campaign_id": self.manifest.campaign_id,
            "query_id": query,
            "cell_id": cell.cell_id,
            "axis": axis,
            "thresholds": list(thresholds),
            "typed_answer_id": chosen,
            "question_budget": cell.question_budget,
            "round_id": cell.round_id,
            "answer_model": cell.answer_model,
            "treatment": cell.treatment.value,
            "fallback_enabled": enabled,
            "selected_route": runtime_fact_payload(route),
            "selected_answer_method": saved_score["answer_source_method"],
            "source_cell_sha256": source_sha,
            "terminal_fact_sha256": terminal_sha,
            "score_source_kind": "typed_answer" if isinstance(selected, dict) else "runtime_cell",
            "evaluation_identity_sha256": self.save(
                self.evaluation_identities[saved_sha], scope="evaluation"
            ),
            "matching_sha256": self.save(matching),
            "answer_proof": answer_proof,
            "source_score": saved_score,
            "score_report_sha256": saved_sha,
            "original_score_report_sha256": report_sha,
            "relevance_policy": "unchanged_saved_evidence_only",
            "scientific_approval": False,
        }
        # Only experiment identity is projected; the scientific outcome is unchanged.
        projected = {
            **saved_score,
            "cell_id": cell.cell_id,
            "variant": cell.cell_id,
            "method": cell.method,
            "source_variant": cell.source_variant,
        }
        return {
            "cell_id": cell.cell_id,
            "query_id": query,
            "axis": axis,
            "thresholds": list(thresholds),
            "context_sha256": self.save(context, scope="evaluation"),
            "answer_sha256": answer_sha,
            "terminal_fact_sha256": terminal_sha,
            "score_report_sha256": saved_sha,
            "score": projected,
        }

    async def relevance_point(self, cell, query, axis, thresholds, universe):
        from pmmc.experiments.step8_relevance_proofs import (
            no_answer_score,
            project_relevance_score,
            relevance_point,
        )
        from pmmc.runtime.qbank_matcher import decide_runtime_route

        definitions = [
            d
            for d in universe
            if (
                d.dataset,
                d.answer_model,
                d.treatment,
                d.question_budget,
                d.round_id,
                d.method,
                d.source_variant,
            )
            == (
                cell.dataset,
                cell.answer_model,
                cell.treatment,
                cell.question_budget,
                cell.round_id,
                "pmmc",
                "active",
            )
        ]
        if len(definitions) != 1:
            raise _ThresholdUnavailable("relevance_active_source_cell_missing")
        fact, base_score, report_sha = self.source(definitions[0], query)
        candidates, config, matching = await self.matching(fact, definitions[0])
        protocol = self.manifest.protocol
        no_reranker = cell.treatment.value == "no_reranker"
        match = protocol.no_reranker_match_threshold if no_reranker else protocol.match_threshold
        gap = protocol.no_reranker_gap_threshold if no_reranker else protocol.gap_threshold
        text_threshold, mm_threshold = (
            protocol.text_relevance_threshold,
            protocol.multimodal_relevance_threshold,
        )
        if axis == "approved_policy":
            match, gap, text_threshold, mm_threshold = thresholds
        elif axis == "relevance_grid":
            text_threshold, mm_threshold = thresholds
        elif axis == "text_relevance":
            text_threshold = thresholds[0]
        else:
            mm_threshold = thresholds[0]
        config = type(config).model_validate(
            {
                **config.model_dump(),
                "active_match_threshold": match,
                "active_gap_threshold": gap,
                "no_reranker_match_threshold": match,
                "no_reranker_gap_threshold": gap,
            }
        )
        route = decide_runtime_route(
            query_executor_run_id=fact.query_executor_run_id,
            query_id=query,
            match_batch_id=fact.route.match_batch_id,
            candidates=candidates,
            config=config,
        )
        report = self.evaluation_reports[report_sha]
        if not report.get("relevance_grid"):
            raise _ThresholdUnavailable("changed_relevance_context_answer_not_generated")
        point, scored = relevance_point(fact, report, text_threshold, mm_threshold)
        enabled = cell.source_variant != "no_fallback"
        pmmc_id = point.get("typed_answer_id") if point else None
        allowed = bool(
            route.outcome.value == "program"
            and point
            and point.get("sufficient")
            and fact.binding is not None
            and fact.binding.plan_id == route.selected_plan_id
            and fact.binding.qbank_id == route.selected_qbank_id
            and point.get("terminal_state") in {"answered", "abstained"}
        )
        if allowed and pmmc_id not in self.answers:
            raise _ThresholdUnavailable("relevance_typed_answer_not_generated")
        chosen = pmmc_id if allowed else fact.fallback_answer_id if enabled else None
        answer_proof = None
        answer_sha = None
        if chosen is not None:
            method = "pmmc" if allowed else "native_multimodal_rag"
            row = self.answers.get(chosen, {})
            answer, answer_proof = await self.answer(
                chosen,
                query,
                row.get("query_executor_run_id"),
                method,
                cell.dataset,
                cell.answer_model,
            )
            answer_sha = answer_proof["typed_answer_sha256"]
            if allowed:
                assert point is not None and scored is not None
                saved_score = project_relevance_score(
                    base_score, point, scored, answer, row["content_hash"]
                )
            else:
                natives = [
                    d
                    for d in universe
                    if (d.dataset, d.answer_model, d.treatment, d.method, d.source_variant)
                    == (
                        cell.dataset,
                        cell.answer_model,
                        cell.treatment,
                        "native_multimodal_rag",
                        "active",
                    )
                ]
                if len(natives) != 1:
                    raise _ThresholdUnavailable("relevance_native_cell_missing")
                _, saved_score, _ = self.native_score(natives[0], query, chosen, report_sha)
        else:
            if enabled:
                raise _ThresholdUnavailable("relevance_fallback_not_generated")
            saved_score = no_answer_score(base_score, fact)
        projected = {
            **saved_score,
            "cell_id": cell.cell_id,
            "variant": cell.cell_id,
            "method": cell.method,
            "source_variant": cell.source_variant,
        }
        source_sha = self.save(runtime_fact_payload(fact))
        context = {
            "schema_version": "step8-threshold-context-v3",
            "campaign_id": self.manifest.campaign_id,
            "cell_id": cell.cell_id,
            "query_id": query,
            "axis": axis,
            "thresholds": list(thresholds),
            "typed_answer_id": chosen,
            "question_budget": cell.question_budget,
            "round_id": cell.round_id,
            "answer_model": cell.answer_model,
            "treatment": cell.treatment.value,
            "fallback_enabled": enabled,
            "selected_route": runtime_fact_payload(route),
            "selected_answer_method": saved_score["answer_source_method"],
            "source_cell_sha256": source_sha,
            "score_report_sha256": report_sha,
            "evaluation_identity_sha256": self.save(
                self.evaluation_identities[report_sha], scope="evaluation"
            ),
            "matching_sha256": self.save(matching),
            "route_config": config.model_dump(mode="json"),
            "answer_proof": answer_proof,
            "source_score": saved_score,
            "relevance_thresholds": [text_threshold, mm_threshold],
            "relevance_policy": "generated_evidence_subset",
            "selected_grid_point": point,
            "policy_outcome": "pmmc" if allowed else "fallback" if enabled else "no_answer",
            "scientific_approval": False,
        }
        return {
            "cell_id": cell.cell_id,
            "query_id": query,
            "axis": axis,
            "thresholds": list(thresholds),
            "context_sha256": self.save(context, scope="evaluation"),
            "answer_sha256": answer_sha,
            "terminal_fact_sha256": source_sha,
            "score_report_sha256": report_sha,
            "score": projected,
        }


async def _produce_threshold_facts(
    builder, directory, manifest, settings, payload, *, selected_policy=None
) -> None:
    from pmmc.runtime.query_executor_config import load_query_executor_config

    rows, coverage = [], []
    problem = None
    cells: tuple[ExperimentCell, ...] = ()
    try:
        raw = json.loads(read_campaign_file(directory, "manifests/matrix.json"))
        if canonical_content_hash(raw) != manifest.matrix_hash:
            raise _ThresholdUnavailable("threshold_matrix_hash_mismatch")
        cells = tuple(ExperimentCell.model_validate(r) for r in raw)
        frozen_config = json.loads(read_campaign_file(directory, "manifests/config.json"))
        config_path = settings.resolve_path(Path(frozen_config["query_executor_config"]))
        if (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            != manifest.protocol.config_hashes["query_executor_config"]
        ):
            raise _ThresholdUnavailable("query_executor_config_hash_mismatch")
        config = load_query_executor_config(config_path)
        if not config.is_v3:
            raise _ThresholdUnavailable("threshold_producer_requires_runtime_v3")
    except (OSError, ValueError, KeyError):
        config = None
        problem = "threshold_matrix_or_runtime_config_unavailable"
    producer = _ThresholdProducer(builder, manifest, payload, config)
    await producer.load_scores()
    selection = _selection(directory, manifest)
    cohorts = [
        SelectionExpansionManifest.model_validate_json(
            read_campaign_file(directory, str(path.relative_to(directory)))
        )
        for path in sorted((directory / "manifests/selections").glob("*.json"))
    ]
    assignments = {
        (r["cell_id"], r["query_id"])
        for r in builder.snapshot.rows(sm.BenchmarkCellAssignmentModel)
    }
    for cell in cells:
        if cell.method != "pmmc":
            continue
        for query in sorted(
            {q for cohort in cohorts for q in cohort.query_ids.get(cell.dataset, ())}
        ):
            axes = _threshold_axes(cell, manifest.protocol)
            if selected_policy is not None and cell.source_variant in {"active", "no_fallback"}:
                axes.append(selected_policy_axis(cell, selected_policy))
            for axis, grid, frozen in axes:
                for thresholds in grid:
                    key = {
                        "cell_id": cell.cell_id,
                        "query_id": query,
                        "axis": axis,
                        "thresholds": list(thresholds),
                    }
                    try:
                        if problem:
                            raise _ThresholdUnavailable(problem)
                        if (cell.cell_id, query) not in assignments:
                            raise _ThresholdUnavailable("campaign_assignment_missing")
                        row = await producer.point(cell, query, axis, thresholds, frozen, cells)
                    except _ThresholdUnavailable as exc:
                        coverage.append({**key, "status": "incomplete", "reason": str(exc)})
                    except (ValueError, KeyError, TypeError, OSError, AttributeError):
                        coverage.append(
                            {**key, "status": "incomplete", "reason": "saved_fact_invalid"}
                        )
                    else:
                        rows.append(row)
                        coverage.append({**key, "status": "complete", "reason": None})
    # Retain all cohorts, including regression selections before later expansion.
    for cohort in cohorts:
        selected = {(c.cell_id, q) for c in cells for q in cohort.query_ids.get(c.dataset, ())}
        scoped_rows = [r for r in rows if (r["cell_id"], r["query_id"]) in selected]
        scoped_coverage = [r for r in coverage if (r["cell_id"], r["query_id"]) in selected]
        value = {
            "schema_version": "step8-threshold-facts-v2",
            "campaign_id": manifest.campaign_id,
            "selection_hash": cohort.content_hash,
            "phase": manifest.phase.value,
            "source_role": "evaluator",
            "database_snapshot": builder.snapshot.identity,
            "rows": scoped_rows,
            "coverage": scoped_coverage,
            "scientific_approval": False,
            "selected_policy": (
                {
                    k: getattr(selected_policy, k)
                    for k in (
                        "match_threshold",
                        "gap_threshold",
                        "no_reranker_match_threshold",
                        "no_reranker_gap_threshold",
                        "text_relevance_threshold",
                        "multimodal_relevance_threshold",
                    )
                }
                if selected_policy is not None
                else None
            ),
            "scope": "declared_pmmc_cells_all_budget_rounds",
            "status": "complete"
            if not problem
            and scoped_coverage
            and all(r["status"] == "complete" for r in scoped_coverage)
            else "incomplete",
            "limitations": [problem]
            if problem
            else ["changed_relevance_contexts_require_generated_answers"],
        }
        digest = producer.save(value, scope="evaluation")
        write_json(
            directory / f"private/threshold-facts/{cohort.content_hash}/{digest}.json",
            value,
            immutable=True,
        )
        write_json(directory / f"private/threshold-facts/{cohort.content_hash}.json", value)
        if cohort.content_hash == selection.content_hash:
            write_json(directory / "private/threshold-facts.json", value)


async def build_campaign_audit_inventory(
    settings: Settings,
    directory: Path,
    artifact_store: CampaignArtifactStore,
    *,
    selected_policy: ScientificProtocol | None = None,
) -> CampaignAuditInventory:
    """Read one evaluator snapshot, verify required private objects, persist inventory.

    The caller's explicit private-export authorization is required at the CLI;
    this producer never publishes, calls models, or changes scientific DB rows.
    """
    directory = Path(directory)
    manifest = BenchmarkCampaignManifest.model_validate_json(
        read_campaign_file(directory, "manifests/campaign.json")
    )
    if selected_policy is not None:
        from pmmc.experiments.step8_approval import _POLICY_FIELDS

        if manifest.phase.value != "development" or selected_policy.model_dump(
            exclude=set(_POLICY_FIELDS)
        ) != manifest.protocol.model_dump(exclude=set(_POLICY_FIELDS)):
            raise ValueError("approval_policy_changes_scientific_contract")
    selection = _selection(directory, manifest)
    snapshot = _Snapshot()
    engine = None
    try:
        engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
        async with sessions() as session, session.begin():
            await session.execute(
                text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            )
            identity = await session.execute(
                text(
                    "SELECT txid_current_snapshot()::text AS snapshot, "
                    "current_user AS database_role"
                )
            )
            snapshot.identity = dict(identity.mappings().one())
            if snapshot.identity.get("database_role") != EVALUATION_ROLE:
                snapshot.missing("database_role_is_not_evaluator")
            else:
                await _read_snapshot(session, snapshot, manifest, selection)
    except (SQLAlchemyError, DatabaseConfigurationError):
        snapshot.missing("evaluator_database_snapshot_unavailable")
    finally:
        if engine is not None:
            await engine.dispose()
    builder = _Builder(snapshot, artifact_store)
    await _populate(builder, directory, manifest, settings, selected_policy)
    coverage = tuple(sorted(_FAMILIES - snapshot.issues.keys()))
    provenance = {
        "schema_version": "step8-database-audit-snapshot-v1",
        "campaign_id": manifest.campaign_id,
        "selection_hash": selection.content_hash,
        "database_snapshot": snapshot.identity,
        "isolation": "repeatable_read_read_only",
        "source_role": "evaluator",
        "table_hashes": {
            name: canonical_content_hash(rows) for name, rows in sorted(snapshot.tables.items())
        },
        "table_counts": {name: len(rows) for name, rows in sorted(snapshot.tables.items())},
        "run_ids": sorted((snapshot.runs - snapshot.conformance) | snapshot.dependency_runs),
        "dependency_run_ids": sorted(snapshot.dependency_runs),
        "dependency_model_call_ids": sorted(snapshot.dependency_calls),
        "adopted_model_call_ids": sorted(snapshot.adopted_calls),
        "dependency_artifact_hashes": {
            key: builder.artifacts[key].sha256 for key in sorted(snapshot.cost_dependencies)
        },
        "conformance_run_ids": sorted(snapshot.conformance),
        "evaluation_run_ids": sorted(snapshot.evaluations),
        "coverage": coverage,
        "missing": {k: sorted(v) for k, v in sorted(snapshot.issues.items())},
        "scientific_approval": False,
    }
    source = builder.fact("inventory-provenance", provenance, "performance", "evaluation")
    result = CampaignAuditInventory(
        campaign_id=manifest.campaign_id,
        selection_hash=selection.content_hash,
        source_role="evaluator",
        source_snapshot_sha256=builder.artifacts[source].sha256,
        scope="all_campaign_attempts",
        complete=not snapshot.issues,
        conformance_run_ids=tuple(sorted(snapshot.conformance)),
        coverage=coverage,
        expected_records=sorted(builder.expected),
        artifacts=tuple(builder.artifacts[k] for k in sorted(builder.artifacts)),
        records=tuple(sorted(builder.records, key=lambda r: (r.kind, r.record_id))),
    )
    value = result.model_dump(mode="json")
    digest = canonical_content_hash(value)
    write_json(directory / f"private/audit-inventories/{digest}.json", value, immutable=True)
    write_json(directory / "private/audit-inventory-status.json", provenance)
    write_json(directory / "private/audit-inventory.json", value)
    return result
