"""Offline campaign analysis over immutable, completed component outputs.

No evaluator, model gateway, database, or dataset adapter is imported here. A
missing selected query is an engineering-missing row, never a smaller denominator.
Public reports contain aggregates, not raw query/cluster IDs or free-form errors.

Optional ``private/threshold-facts.json`` (or the selection-specific
``private/threshold-facts/<selection_hash>.json``) binds campaign and selection.
Legacy ``step8-threshold-facts-v1`` remains readable. Each row has
cell_id, query_id, axis (match_gap/text_relevance/multimodal_relevance), thresholds
(a list), context_sha256, answer_sha256 and score (the evaluation score row).
Both payloads must exist in the CAS, and the answer's typed_answer_id must match
the score. The context object binds query_id, cell_id, axis, thresholds and
typed_answer_id; the answer is a full TypedAnswerResult, not an ID-only reference.
This is a saved-fact inventory, NOT permission to generate answers or
infer that an answer for another context applies to a changed threshold.
The evaluator inventory producer writes v2 with per-point coverage, fallback_on_off,
verified request/context/grounding and original evaluation-score CAS provenance.
Only v2 permits a null answer for an observed no_fallback/no_answer cell and score.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation.paper_statistics import (
    ComparisonResult,
    MethodVariant,
    PlannedComparison,
    ScoreObservation,
    SelectedQuery,
    StatisticsConfig,
    TerminalStatus,
    TypedScore,
    compare_family,
    holm_adjust,
    summarize_method,
)
from pmmc.experiments.step8_artifacts import (
    CampaignArtifactStore,
    SnapshotArtifactReader,
    json_bytes,
    write_json,
)
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt, validate_job_dag
from pmmc.schemas.answer_output import TypedAnswerResult
from pmmc.schemas.base import DatasetName
from pmmc.schemas.step8 import (
    BASELINE_METHODS,
    GAP_THRESHOLDS,
    MATCH_THRESHOLDS,
    RELEVANCE_THRESHOLDS,
    STEP8_READABLE_REVISIONS,
    AnalysisManifest,
    BenchmarkCampaignManifest,
    CampaignPhase,
    ExperimentCell,
    SelectionExpansionManifest,
)


def read_campaign_file(directory: Path, relative: str) -> bytes:
    """Read only regular campaign-owned files, rejecting all symlink ancestors."""
    root = directory.resolve()
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("unsafe campaign path")
    if any(p.is_symlink() for p in (path, *path.parents) if p != root and p.is_relative_to(root)):
        raise ValueError("campaign inputs cannot be symlinks")
    if not path.is_file():
        raise FileNotFoundError("campaign input is absent")
    return path.read_bytes()


def _digest(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("invalid SHA-256 identity")
    return value


def analysis_source_hashes() -> dict[str, str]:
    """Bind post-processing independently of model-stage scientific fingerprints."""
    package = Path(__file__).resolve().parents[1]
    paths = (
        "experiments/step8_analysis.py",
        "experiments/step8_judge_analysis.py",
        "experiments/step8_process_analysis.py",
        "experiments/step8_stratified_analysis.py",
        "experiments/step8_compiler_analysis.py",
        "experiments/step8_semantic_analysis.py",
        "experiments/step8_cost_analysis.py",
        "experiments/step8_cost_dependencies.py",
        "experiments/step8_questioner_cost_dependencies.py",
        "experiments/step8_r0_cost_dependencies.py",
        "experiments/step8_pmmc_cost_dependencies.py",
        "experiments/release_compiler_reuse.py",
        "experiments/step8_upstream_import.py",
        "infrastructure/database/step8_upstream_import.py",
        "experiments/step8_import_scheduler.py",
        "experiments/step8_supervisor.py",
        "experiments/step8_runner.py",
        "infrastructure/database/step8_repositories.py",
        "experiments/step8_performance_analysis.py",
        "experiments/step8_audit.py",
        "experiments/step8_relevance_proofs.py",
        "experiments/step8_inventory.py",
        "experiments/step8_approval.py",
        "experiments/step8_artifacts.py",
        "runtime/relation_grounding.py",
        "evaluation/paper_statistics.py",
        "evaluation/step8_oracle_analysis.py",
        "evaluation/step8_oracle_hook.py",
        "evaluation/step8_oracle.py",
        "evaluation/paper_report.py",
        "schemas/step8.py",
        "schemas/step8_process.py",
    )
    return {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in paths}


@dataclass
class CampaignFacts:
    directory: Path
    manifest: BenchmarkCampaignManifest
    selection: SelectionExpansionManifest
    cells: tuple[ExperimentCell, ...]
    jobs: dict[str, BenchmarkStageJob]
    receipts: dict[str, ComponentReceipt]
    verified_jobs: set[str]
    input_hashes: dict[str, str]
    files: dict[str, bytes]
    store: CampaignArtifactStore
    missing_artifact_count: int


def validate_selection_chain(selections: dict[str, SelectionExpansionManifest]) -> str:
    """Validate cohort ancestry, not an unconditional cumulative query denominator."""
    parents = {s.parent_selection_hash for s in selections.values()}
    leaves = set(selections) - parents
    if len(leaves) != 1 or sum(s.parent_selection_hash is None for s in selections.values()) != 1:
        raise ValueError("ambiguous selection expansion chain")
    seen = set()
    current: str | None = next(iter(leaves))
    while current is not None:
        if current in seen or current not in selections:
            raise ValueError("broken selection expansion chain")
        seen.add(current)
        child = selections[current]
        current = child.parent_selection_hash
        if current in selections:
            parent = selections[current]
            if (parent.campaign_id, parent.phase, parent.partition_hash) != (
                child.campaign_id,
                child.phase,
                child.partition_hash,
            ):
                raise ValueError("selection cohort identity mismatch")
            if child.phase is CampaignPhase.PAPER:
                if int(child.selection) < int(parent.selection):
                    raise ValueError("paper selection cannot move backwards")
                if any(
                    not set(ids) <= set(child.query_ids[d]) for d, ids in parent.query_ids.items()
                ):
                    raise ValueError("paper selection expansion removed queries")
            elif (parent.selection, child.selection) != ("regression", "confirmation"):
                raise ValueError("development cohorts must advance regression to confirmation")
            elif any(set(ids) & set(child.query_ids[d]) for d, ids in parent.query_ids.items()):
                raise ValueError("development regression and confirmation must be disjoint")
    if seen != set(selections):
        raise ValueError("disconnected selection expansion chain")
    return next(iter(leaves))


def load_campaign_facts(
    directory: Path,
    *,
    selection_hash: str | None = None,
    artifact_store: CampaignArtifactStore | None = None,
) -> CampaignFacts:
    """Validate frozen identities; never use filesystem mtimes to select results."""
    directory = Path(directory)
    files: dict[str, bytes] = {}

    def read(relative: str) -> Any:
        files[relative] = read_campaign_file(directory, relative)
        return json.loads(files[relative])

    manifest = BenchmarkCampaignManifest.model_validate(read("manifests/campaign.json"))
    raw_cells = read("manifests/matrix.json")
    if canonical_content_hash(raw_cells) != manifest.matrix_hash:
        raise ValueError("campaign matrix hash mismatch")
    cells = tuple(ExperimentCell.model_validate(item) for item in raw_cells)
    if not cells or len({c.cell_id for c in cells}) != len(cells):
        raise ValueError("campaign matrix is empty or duplicated")
    if any(c.protocol_hash != manifest.protocol.content_hash for c in cells):
        raise ValueError("cell protocol mismatch")
    selections = {}
    for path in sorted((directory / "manifests/selections").glob("*.json")):
        item = SelectionExpansionManifest.model_validate(read(str(path.relative_to(directory))))
        if path.stem != item.content_hash:
            raise ValueError("selection filename/hash mismatch")
        if (item.campaign_id, item.phase, item.partition_hash) != (
            manifest.campaign_id,
            manifest.phase,
            manifest.protocol.partition_hash,
        ):
            raise ValueError("selection campaign identity mismatch")
        selections[item.content_hash] = item
    latest = validate_selection_chain(selections)
    chosen = selection_hash or latest
    if chosen not in selections:
        raise ValueError("unknown campaign selection")
    selection = selections[chosen]
    job_ids = read(f"manifests/jobs-{chosen}.json")
    if not isinstance(job_ids, list) or len(set(job_ids)) != len(job_ids):
        raise ValueError("invalid campaign job inventory")
    jobs, receipts = {}, {}
    store = artifact_store or CampaignArtifactStore(directory.resolve().parent)
    available, missing_artifacts = set(), set()
    for job_id in job_ids:
        _digest(job_id)
        job = BenchmarkStageJob.model_validate(read(f"private/jobs/{job_id}.json"))
        if job.job_id != job_id or job.campaign_id != manifest.campaign_id:
            raise ValueError("job campaign identity mismatch")
        if manifest.protocol.source_fingerprints.get(job.stage) != job.source_fingerprint:
            raise ValueError("job frozen source mismatch")
        jobs[job_id] = job
        try:
            raw = read(f"checkpoints/components/{job_id}.json")
        except FileNotFoundError:
            continue
        receipt = ComponentReceipt.model_validate(raw)
        if (receipt.job_id, receipt.scientific_input_hash, receipt.source_fingerprint) != (
            job_id,
            job.scientific_input_hash,
            job.source_fingerprint,
        ) or (
            receipt.schema_revision not in STEP8_READABLE_REVISIONS
            or set(receipt.upstream_receipt_hashes) != set(job.dependency_ids)
        ):
            raise ValueError("component receipt identity mismatch")
        receipts[job_id] = receipt
        valid = True
        for digest in receipt.artifact_hashes:
            _digest(digest)
            try:
                store.verify(digest)
            except (ValueError, OSError):
                missing_artifacts.add(digest)
                valid = False
        # The producer saves the exact output in CAS before writing its receipt.
        output_hash = hashlib.sha256(json_bytes(receipt.output)).hexdigest()
        if output_hash not in receipt.artifact_hashes:
            valid = False
            missing_artifacts.add(output_hash)
        if valid:
            available.add(job_id)
    validate_job_dag(tuple(jobs.values()))
    verified: set[str] = set()

    def verify(job_id: str) -> bool:
        if job_id in verified:
            return True
        if job_id not in available:
            return False
        receipt = receipts[job_id]
        for parent_id, expected in receipt.upstream_receipt_hashes.items():
            if parent_id in receipts and receipts[parent_id].content_hash != expected:
                raise ValueError("upstream receipt hash mismatch")
            if not verify(parent_id):
                return False
        verified.add(job_id)
        return True

    for job_id in jobs:
        verify(job_id)
    return CampaignFacts(
        directory,
        manifest,
        selection,
        cells,
        jobs,
        receipts,
        verified,
        {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
        files,
        store,
        len(missing_artifacts),
    )


def _method(cell: ExperimentCell) -> MethodVariant:
    # cell_id separates budgets, rounds, treatments, datasets and answer models.
    return MethodVariant(cell.method, cell.cell_id)


def _observation(query_id: str, cell: ExperimentCell, row: dict) -> ScoreObservation:
    if row.get("cell_id") != cell.cell_id or row.get("method") != cell.method:
        raise ValueError("score/cell identity mismatch")
    source_variant = cell.source_variant if cell.method == "pmmc" else "primary"
    if row.get("source_variant") != source_variant or row.get("metric") != "typed_exact_match":
        raise ValueError("score source variant or metric mismatch")
    status = TerminalStatus(row["status"])
    if row.get("engineering_error") or (
        status is TerminalStatus.ANSWERED and not row.get("typed_answer_id")
    ):
        status = TerminalStatus.ENGINEERING_MISSING
    score = None
    if status is TerminalStatus.ANSWERED:
        score = TypedScore(
            row["value"],
            row["applicable"],
            unavailable_reason=row.get("unavailable_reason"),
        )
    return ScoreObservation(query_id, _method(cell), status, score)


def _collect_scores(facts: CampaignFacts):
    cells = {c.cell_id: c for c in facts.cells}
    selected: dict[tuple[str, str], SelectedQuery] = {}
    rows: dict[tuple[str, str], dict] = {}
    for job_id in sorted(facts.verified_jobs):
        job = facts.jobs[job_id]
        if job.stage != "evaluation":
            continue
        report = facts.receipts[job_id].output
        query = SelectedQuery(
            **{
                **report["selected_query"],
                "strata": tuple(tuple(x) for x in report["selected_query"].get("strata", ())),
            }
        )
        if (report.get("campaign_id"), report.get("query_id")) != (
            facts.manifest.campaign_id,
            query.query_id,
        ) or query.query_id not in facts.selection.query_ids.get(DatasetName(query.dataset), ()):
            raise ValueError("evaluation outside frozen campaign selection")
        if (job.scientific_inputs.get("query_id"), job.scientific_inputs.get("dataset")) != (
            query.query_id,
            query.dataset,
        ):
            raise ValueError("evaluation/job query identity mismatch")
        if (
            job.scientific_inputs.get("snapshot_id")
            != facts.manifest.snapshot_ids[DatasetName(query.dataset)]
        ):
            raise ValueError("evaluation snapshot identity mismatch")
        key = query.dataset, query.query_id
        if key in selected and selected[key] != query:
            raise ValueError("conflicting query cluster or applicability facts")
        selected[key] = query
        for row in report["scores"]:
            cell = cells.get(row["cell_id"])
            if cell is None or cell.dataset.value != query.dataset:
                raise ValueError("evaluation cell outside frozen matrix")
            if cell.answer_model != job.scientific_inputs.get("answer_model"):
                raise ValueError("evaluation answer model mismatch")
            expected_treatment = job.scientific_inputs.get("treatment", "main")
            if cell.treatment.value != expected_treatment:
                raise ValueError("evaluation treatment mismatch")
            if row.get("applicable") is not query.applicable:
                raise ValueError("evaluation applicability mismatch")
            _observation(query.query_id, cell, row)
            row_key = cell.cell_id, query.query_id
            if row_key in rows:
                raise ValueError("duplicate evaluation query/cell fact")
            rows[row_key] = row
    return selected, rows


def _selection(facts: CampaignFacts, metadata: dict, dataset: str) -> tuple[SelectedQuery, ...]:
    return tuple(
        metadata.get((dataset, qid))
        or SelectedQuery(
            qid,
            dataset,
            "unknown:" + qid,
            "scenario" if dataset == "memgallery" else "conversation",
        )
        for qid in facts.selection.query_ids[DatasetName(dataset)]
    )


def _summary(selected, observations, cell, config, metadata):
    result = asdict(summarize_method(selected, observations, _method(cell), config))
    result.pop("missing_query_ids")
    result.pop("method")
    unknown = sum((q.dataset, q.query_id) not in metadata for q in selected)
    result["unknown_query_metadata_count"] = unknown
    if unknown:
        result["cluster_count"] = result["applicable_count"] = None
    return {"cell_id": cell.cell_id, **cell.model_dump(mode="json"), **result}


def _public_comparison(result, *, dataset: str, answer_model: str, metadata_known: bool) -> dict:
    value = asdict(result)
    if not metadata_known:
        value["applicable_count"] = value["cluster_count"] = None
    for name in ("micro", "cluster_macro"):
        effect = value.get(name)
        if effect is not None:
            effect["leave_one_cluster_out"] = [v for _, v in effect["leave_one_cluster_out"]]
    return {"dataset": dataset, "answer_model": answer_model, **value}


def _threshold_axes(cell, protocol):
    axes = []
    if cell.method == "pmmc" and cell.source_variant in {"active", "no_fallback"}:
        no_reranker = cell.treatment.value == "no_reranker"
        axes.append(
            (
                "match_gap",
                [(m, g) for m in MATCH_THRESHOLDS for g in GAP_THRESHOLDS],
                (
                    protocol.no_reranker_match_threshold
                    if no_reranker
                    else protocol.match_threshold,
                    protocol.no_reranker_gap_threshold if no_reranker else protocol.gap_threshold,
                ),
            )
        )
        if cell.source_variant == "active":
            axes.append(("fallback_on_off", [(False,), (True,)], (True,)))
        axes.append(
            (
                "relevance_grid",
                [(t, m) for t in RELEVANCE_THRESHOLDS for m in RELEVANCE_THRESHOLDS],
                (protocol.text_relevance_threshold, protocol.multimodal_relevance_threshold),
            )
        )
    # Native baselines have no Executor relevance gate. Do not label a
    # non-existent baseline treatment as an incomplete experiment.
    if cell.method == "pmmc" and cell.source_variant == "active":
        axes.extend(
            (
                (
                    "text_relevance",
                    [(v,) for v in RELEVANCE_THRESHOLDS],
                    (protocol.text_relevance_threshold,),
                ),
                (
                    "multimodal_relevance",
                    [(v,) for v in RELEVANCE_THRESHOLDS],
                    (protocol.multimodal_relevance_threshold,),
                ),
            )
        )
    return axes


def selected_policy_axis(cell, protocol):
    """One requested joint policy, not a new Cartesian product of model calls."""
    no_reranker = cell.treatment.value == "no_reranker"
    point = (
        protocol.no_reranker_match_threshold if no_reranker else protocol.match_threshold,
        protocol.no_reranker_gap_threshold if no_reranker else protocol.gap_threshold,
        protocol.text_relevance_threshold,
        protocol.multimodal_relevance_threshold,
    )
    return "approved_policy", [point], point


def threshold_proof_hashes(value):
    """Only explicit CAS references, never arbitrary private IDs or DB content hashes."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("_sha256") and item is not None:
                yield _digest(item)
            elif isinstance(item, (dict, list)):
                yield from threshold_proof_hashes(item)
    elif isinstance(value, list):
        for item in value:
            yield from threshold_proof_hashes(item)


def _verify_threshold_v2(store, saved, context):
    if context.get("schema_version") == "step8-threshold-context-v3":
        from pmmc.experiments.step8_relevance_proofs import verify_relevance_projection

        verify_relevance_projection(store, saved, context)
        return
    from pmmc.schemas.query_executor import QueryExperimentCellResult, runtime_fact_payload

    for digest in threshold_proof_hashes(context):
        store.verify(digest)
    matching = store.load_json(_digest(context["matching_sha256"]))
    for digest in threshold_proof_hashes(matching):
        store.verify(digest)
    report = store.load_json(_digest(saved["score_report_sha256"]))
    source = context["source_score"]
    if (
        report.get("campaign_id") != context["campaign_id"]
        or report.get("query_id") != context["query_id"]
        or source not in report.get("scores", [])
        or context["score_report_sha256"] != saved["score_report_sha256"]
    ):
        raise ValueError("threshold score report provenance mismatch")
    projected = {
        **source,
        **{k: saved["score"][k] for k in ("cell_id", "variant", "method", "source_variant")},
    }
    if projected != saved["score"]:
        raise ValueError("threshold score was not reused unchanged")
    if context.get("score_source_kind") == "typed_answer":
        identity = store.load_json(_digest(context["evaluation_identity_sha256"]))
        row = store.load_json(_digest(saved["terminal_fact_sha256"]))
        answer_payload = row["payload"]["metadata"]["typed_result"]
        answer = TypedAnswerResult.model_validate(answer_payload)
        origin = QueryExperimentCellResult.model_validate(
            store.load_json(_digest(context["source_cell_sha256"]))
        )
        trace = store.load_json(
            _digest(context["answer_proof"]["calls"][answer.answer_model_call_id]["trace_sha256"])
        )
        if (
            context["original_score_report_sha256"] != saved["score_report_sha256"]
            or canonical_content_hash(identity) != report["identity_hash"]
            or identity.get("query_id") != answer.query_id
            or identity.get("snapshot_id") != answer.snapshot_id
            or identity.get("source_run_id") != origin.query_executor_run_id
            or source["method"] != "native_multimodal_rag"
            or source["source_variant"] != "primary"
            or answer.method != "native_multimodal_rag"
            or answer.answer_variant != "primary"
            or canonical_content_hash(answer_payload) != row["content_hash"]
            or row["content_hash"] != source["source_fact_hash"]
            or answer.typed_answer_id != row["typed_answer_id"]
            or answer.typed_answer_id != source["source_fact_id"]
            or answer.typed_answer_id != source["typed_answer_id"]
            or answer.typed_answer_id != context["typed_answer_id"]
            or answer.typed_answer_id != origin.fallback_answer_id
            or answer.query_id != context["query_id"]
            or answer.query_id != origin.query_id
            or answer.run_id != row["query_executor_run_id"]
            or answer.terminal_state != source["terminal_state"]
            or answer.terminal_state != row["terminal_state"]
            or trace.get("model_alias") != context["answer_model"]
            or trace.get("actual_model_id") != answer.actual_model_id
            or store.load_json(_digest(saved["answer_sha256"])) != answer_payload
            or not context["fallback_enabled"]
        ):
            raise ValueError("threshold native answer provenance mismatch")
        return
    terminal = QueryExperimentCellResult.model_validate(
        store.load_json(_digest(saved["terminal_fact_sha256"]))
    )
    fact_payload = terminal.model_dump(
        mode="json", exclude={"cell_result_id", "content_hash", "created_at"}
    )
    if (
        canonical_content_hash(runtime_fact_payload(fact_payload)) != terminal.content_hash
        or terminal.cell_result_id != source["source_fact_id"]
        or terminal.content_hash != source["source_fact_hash"]
        or terminal.query_id != context["query_id"]
        or terminal.selected_answer_id != context["typed_answer_id"]
        or terminal.terminal_state != source["terminal_state"]
    ):
        raise ValueError("threshold terminal fact provenance mismatch")
    if saved.get("answer_sha256") is None and (
        terminal.terminal_state != "no_answer"
        or terminal.cell.treatment != "no_fallback"
        or source.get("typed_answer_id") is not None
        or source["status"] != "abstention"
        or source["value"] != 0.0
        or context["fallback_enabled"]
    ):
        raise ValueError("threshold refusal is not an observed no-answer terminal")


def _threshold_scans(facts, metadata, rows, config):
    reader = SnapshotArtifactReader(facts.store)
    recorded = {}
    unavailable = set()
    complete_coverage = set()
    selected_policy = None
    relative = "private/threshold-facts.json"
    try:
        cohort_relative = f"private/threshold-facts/{facts.selection.content_hash}.json"
        payload = read_campaign_file(facts.directory, cohort_relative)
        relative = cohort_relative
    except FileNotFoundError:
        try:
            payload = read_campaign_file(facts.directory, relative)
        except FileNotFoundError:
            payload = None
    if payload is not None:
        inventory = json.loads(payload)
        if inventory.get("schema_version") not in {
            "step8-threshold-facts-v1",
            "step8-threshold-facts-v2",
        } or (
            inventory.get("campaign_id"),
            inventory.get("selection_hash"),
        ) != (facts.manifest.campaign_id, facts.selection.content_hash):
            raise ValueError("threshold inventory identity mismatch")
        facts.input_hashes[relative] = hashlib.sha256(payload).hexdigest()
        facts.files[relative] = payload
        if inventory.get("selected_policy") is not None:
            from pmmc.experiments.step8_approval import _POLICY_FIELDS
            from pmmc.schemas.step8 import ScientificProtocol

            if set(inventory["selected_policy"]) != set(_POLICY_FIELDS):
                raise ValueError("threshold_selected_policy_fields_invalid")
            selected_policy = ScientificProtocol.model_validate(
                {
                    **facts.manifest.protocol.model_dump(),
                    **inventory["selected_policy"],
                }
            )
        for entry in inventory.get("coverage", []):
            key = entry["cell_id"], entry["query_id"], entry["axis"], tuple(entry["thresholds"])
            if entry["status"] == "complete":
                complete_coverage.add(key)
            else:
                unavailable.add(key)
        for row in inventory["rows"]:
            key = row["cell_id"], row["query_id"], row["axis"], tuple(row["thresholds"])
            if key in recorded:
                raise ValueError("duplicate saved threshold fact")
            recorded[key] = row
    protocol = facts.manifest.protocol
    scans = []
    used = set()
    for cell in facts.cells:
        axes = _threshold_axes(cell, protocol)
        if (
            selected_policy is not None
            and cell.method == "pmmc"
            and cell.source_variant in {"active", "no_fallback"}
        ):
            axes.append(selected_policy_axis(cell, selected_policy))
        selected = _selection(facts, metadata, cell.dataset.value)
        for axis, grid, frozen in axes:
            for thresholds in grid:
                observations = []
                for query in selected:
                    # The requested joint policy is not the original campaign policy.
                    row = (
                        rows.get((cell.cell_id, query.query_id))
                        if axis != "approved_policy" and thresholds == frozen
                        else None
                    )
                    key = cell.cell_id, query.query_id, axis, thresholds
                    if key in recorded:
                        used.add(key)
                        saved = recorded[key]
                        try:
                            context = reader.load_json(_digest(saved["context_sha256"]))
                            if axis == "approved_policy" and context.get("schema_version") != (
                                "step8-threshold-context-v3"
                            ):
                                raise ValueError("joint policy requires generated context proof")
                            answer = (
                                TypedAnswerResult.model_validate(
                                    reader.load_json(_digest(saved["answer_sha256"]))
                                )
                                if saved.get("answer_sha256") is not None
                                else None
                            )
                            if context.get("schema_version") in {
                                "step8-threshold-context-v2",
                                "step8-threshold-context-v3",
                            }:
                                _verify_threshold_v2(reader, saved, context)
                            elif answer is None:
                                raise ValueError("legacy threshold fact requires typed answer")
                        except (ValueError, OSError):
                            row = None
                        else:
                            row = saved["score"]
                            if answer is not None and (
                                not row.get("typed_answer_id")
                                or answer.typed_answer_id != row["typed_answer_id"]
                            ):
                                raise ValueError("threshold answer/score identity mismatch")
                            expected_context = {
                                "query_id": query.query_id,
                                "cell_id": cell.cell_id,
                                "axis": axis,
                                "thresholds": list(thresholds),
                                "typed_answer_id": row["typed_answer_id"],
                            }
                            if any(context.get(k) != v for k, v in expected_context.items()):
                                raise ValueError("threshold context/answer binding mismatch")
                            if answer is not None and answer.query_id != query.query_id:
                                raise ValueError("threshold answer query mismatch")
                            if answer is not None and (
                                answer.method != row.get("answer_source_method")
                                or answer.answer_variant != row.get("answer_source_variant")
                                or answer.terminal_state != row.get("terminal_state")
                            ):
                                raise ValueError("threshold answer provenance mismatch")
                            if (query.dataset, query.query_id) not in metadata or row.get(
                                "applicable"
                            ) is not query.applicable:
                                raise ValueError(
                                    "threshold query metadata is unavailable or inconsistent"
                                )
                    # A saved response cannot override an incomplete coverage attestation.
                    if key in unavailable or (
                        axis == "approved_policy" and key not in complete_coverage
                    ):
                        row = None
                    if row is not None:
                        observations.append(_observation(query.query_id, cell, row))
                summary = _summary(selected, tuple(observations), cell, config, metadata)
                scans.append(
                    {
                        "cell_id": cell.cell_id,
                        "axis": axis,
                        "thresholds": list(thresholds),
                        "frozen_point": thresholds == frozen,
                        "exploratory": True,
                        **{
                            k: summary[k]
                            for k in (
                                "status",
                                "selected_count",
                                "engineering_missing_count",
                                "micro",
                                "cluster_macro",
                            )
                        },
                    }
                )
    if set(recorded) != used:
        raise ValueError("threshold fact outside declared selection/grid")
    return scans


def analyze_campaign(
    directory: Path,
    *,
    selection_hash: str | None = None,
    artifact_store: CampaignArtifactStore | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Return a JSON-safe public aggregate report; optionally write a new hash-named file.

    The fixed main family is PMMC active b8/r2/main versus each declared baseline,
    separately per dataset/backbone. Holm correction includes the ENTIRE family,
    not just available comparisons. Threshold scans never establish approval.
    ``artifact_completeness`` stays unverified until an independent audit inventory
    is checked by :func:`pmmc.experiments.step8_audit.export_campaign`.
    """
    facts = load_campaign_facts(
        directory, selection_hash=selection_hash, artifact_store=artifact_store
    )
    return _analyze_facts(facts, write=write)


def _analyze_facts(facts: CampaignFacts, *, write: bool = False) -> dict[str, Any]:
    """Use one byte snapshot for both a report and its enclosing export."""
    metadata, rows = _collect_scores(facts)
    from pmmc.evaluation.step8_oracle_analysis import summarize_campaign_oracles
    from pmmc.experiments.step8_compiler_analysis import summarize_campaign_compiler
    from pmmc.experiments.step8_cost_analysis import summarize_campaign_costs
    from pmmc.experiments.step8_judge_analysis import summarize_judge
    from pmmc.experiments.step8_performance_analysis import summarize_campaign_performance
    from pmmc.experiments.step8_process_analysis import (
        paired_transitions,
        summarize_process_metrics,
    )
    from pmmc.experiments.step8_semantic_analysis import summarize_campaign_semantics
    from pmmc.experiments.step8_stratified_analysis import summarize_strata

    judge_report = summarize_judge(
        cells=facts.cells,
        selections={
            dataset.value: _selection(facts, metadata, dataset.value)
            for dataset in facts.selection.query_ids
        },
        rows=rows,
        answer_models=tuple(facts.manifest.protocol.answer_backbones),
        seed=facts.manifest.protocol.seed,
    )
    cost_report = summarize_campaign_costs(facts)
    performance_report = summarize_campaign_performance(facts)
    compiler_report = summarize_campaign_compiler(facts)
    semantic_report = summarize_campaign_semantics(facts)
    oracle_report = summarize_campaign_oracles(facts)
    config = StatisticsConfig(seed=facts.manifest.protocol.seed)
    observations = {
        cell.cell_id: tuple(
            _observation(qid, cell, row)
            for (cell_id, qid), row in sorted(rows.items())
            if cell_id == cell.cell_id
        )
        for cell in facts.cells
    }
    summaries = [
        _summary(
            _selection(facts, metadata, cell.dataset.value),
            observations[cell.cell_id],
            cell,
            config,
            metadata,
        )
        for cell in facts.cells
    ]
    for summary in summaries:
        answered = [
            row
            for (cell_id, _), row in rows.items()
            if cell_id == summary["cell_id"]
            and row["status"] == "answered"
            and not row.get("engineering_error")
            and row.get("typed_answer_id")
        ]
        summary["true_pmmc_answer_count"] = sum(
            row.get("answer_source_method") == "pmmc" for row in answered
        )
        summary["fallback_answer_count"] = (
            sum(row.get("answer_source_method") in BASELINE_METHODS for row in answered)
            if summary["method"] == "pmmc"
            else 0
        )
        summary["unknown_answer_source_count"] = sum(
            row.get("answer_source_method") not in ("pmmc", *BASELINE_METHODS)
            or not row.get("answer_source_variant")
            for row in answered
        )
    comparisons: list[ComparisonResult] = []
    contexts: list[tuple[str, str]] = []
    # Each dataset is sampled independently; correction is applied globally below.
    for dataset in sorted(facts.selection.query_ids):
        selected = _selection(facts, metadata, dataset.value)
        for model in facts.manifest.protocol.answer_backbones:
            main = {
                c.method: c
                for c in facts.cells
                if c.dataset == dataset
                and c.answer_model == model
                and c.treatment.value == "main"
                and c.question_budget == 8
                and c.round_id == 2
                and c.source_variant == "active"
            }
            family = tuple(
                PlannedComparison(
                    f"{dataset.value}:{model}:pmmc-vs-{baseline}",
                    _method(main["pmmc"])
                    if "pmmc" in main
                    else MethodVariant("pmmc", "unassigned"),
                    _method(main[baseline])
                    if baseline in main
                    else MethodVariant(baseline, "unassigned"),
                )
                for baseline in BASELINE_METHODS
            )
            obs = tuple(o for c in main.values() for o in observations[c.cell_id])
            comparisons.extend(compare_family(selected, obs, family, config))
            contexts.extend((dataset.value, model) for _ in family)
    if all(c.p_value is not None for c in comparisons):
        adjusted = holm_adjust(tuple(c.p_value for c in comparisons if c.p_value is not None))
        comparisons = [
            replace(c, holm_p_value=p) for c, p in zip(comparisons, adjusted, strict=True)
        ]
    else:
        comparisons = [
            replace(
                c,
                holm_p_value=None,
                warnings=tuple(sorted(set((*c.warnings, "primary_family_incomplete")))),
            )
            for c in comparisons
        ]
    groups = defaultdict(list)
    for summary in summaries:
        key = tuple(
            summary[k]
            for k in (
                "answer_model",
                "method",
                "treatment",
                "question_budget",
                "round_id",
                "source_variant",
            )
        )
        groups[key].append(summary)
    macro = []
    for key, values in sorted(groups.items()):
        complete = len(values) == len(facts.selection.query_ids) and all(
            v["status"] == "complete" for v in values
        )
        macro.append(
            {
                **dict(
                    zip(
                        (
                            "answer_model",
                            "method",
                            "treatment",
                            "question_budget",
                            "round_id",
                            "source_variant",
                        ),
                        key,
                        strict=True,
                    )
                ),
                "status": "complete" if complete else "incomplete",
                "dataset_count": len(facts.selection.query_ids),
                "dataset_macro": sum(v["micro"] for v in values) / len(values)
                if complete
                else None,
                "dataset_cluster_macro": sum(v["cluster_macro"] for v in values) / len(values)
                if complete
                else None,
            }
        )
    scans = _threshold_scans(facts, metadata, rows, config)
    declaration = AnalysisManifest(
        campaign_id=facts.manifest.campaign_id,
        selection_hash=facts.selection.content_hash,
        matrix_hash=facts.manifest.matrix_hash,
        seed=config.seed,
        input_hashes=facts.input_hashes,
        analysis_source_hashes=analysis_source_hashes(),
    )
    complete = all(s["status"] in {"complete", "not_applicable"} for s in summaries) and len(
        facts.verified_jobs
    ) == len(facts.jobs)
    complete = complete and oracle_report["status"] != "incomplete"
    main_pmmc = [
        s
        for s in summaries
        if s["method"] == "pmmc"
        and s["treatment"] == "main"
        and s["question_budget"] == 8
        and s["round_id"] == 2
        and s["source_variant"] == "active"
    ]
    report = {
        "schema_version": "step8-campaign-analysis-v1",
        "status": "complete" if complete else "incomplete",
        "completion_scope": "typed_answer_denominators_and_verified_component_receipts",
        "analysis": declaration.model_dump(mode="json"),
        "scientific_approval": False,
        "artifact_completeness": "unverified",
        "real_pmmc_viability_supported": bool(main_pmmc)
        and complete
        and all(
            s["status"] == "complete"
            and s["true_pmmc_answer_count"] > 0
            and s["unknown_answer_source_count"] == 0
            for s in main_pmmc
        ),
        "real_pmmc_viability_definition": "nonzero_answered_pmmc_path_not_an_accuracy_threshold",
        "selected_counts": {d.value: len(ids) for d, ids in facts.selection.query_ids.items()},
        "expected_query_cell_count": sum(
            len(facts.selection.query_ids[c.dataset]) for c in facts.cells
        ),
        "missing_component_count": len(facts.jobs) - len(facts.verified_jobs),
        "missing_cas_artifact_count": facts.missing_artifact_count,
        "cells": summaries,
        "macro": macro,
        "comparison_family": "all_declared_main_dataset_backbone_baseline_comparisons",
        "comparisons": [
            _public_comparison(
                c,
                dataset=d,
                answer_model=m,
                metadata_known=all(
                    (d, qid) in metadata for qid in facts.selection.query_ids[DatasetName(d)]
                ),
            )
            for c, (d, m) in zip(comparisons, contexts, strict=True)
        ],
        "threshold_scans": scans,
        "judge_quality": judge_report,
        "stratified_quality": summarize_strata(
            cells=facts.cells,
            selections={
                dataset.value: _selection(facts, metadata, dataset.value)
                for dataset in facts.selection.query_ids
            },
            observations=observations,
            rows=rows,
            seed=config.seed,
        ),
        "costs": cost_report,
        "performance": performance_report,
        "compiler_quality": compiler_report,
        "qbank_semantic_quality": semantic_report,
        "oracle_at_50": oracle_report,
        "process_quality": summarize_process_metrics(facts),
        "paired_process_transitions": paired_transitions(
            facts.cells,
            {d.value: tuple(ids) for d, ids in facts.selection.query_ids.items()},
            rows,
        ),
        "warnings": [
            "receipt_references_do_not_establish_artifact_completeness",
            "threshold_scans_require_saved_context_specific_answers",
            "cluster_resampling_is_not_model_seed_replication",
            "human_semantic_validation_required",
        ],
    }
    report["report_hash"] = canonical_content_hash(report)
    if write:
        write_json(
            facts.directory / "reports" / f"analysis-{report['report_hash']}.json",
            report,
            immutable=True,
        )
    return report
