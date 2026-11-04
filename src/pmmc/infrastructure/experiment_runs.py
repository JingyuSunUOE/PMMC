"""Registration, inspection, and integrity checks for persisted experiment runs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from pmmc.benchmarks.selection import selection_hash
from pmmc.experiments.manifest import run_config_hash
from pmmc.infrastructure.database.baseline_models import (
    ContextAssemblyModel,
    ContextAssemblyRetrievalCandidateModel,
    RetrievalCorpusModel,
    RetrievalDocumentModel,
)
from pmmc.infrastructure.database.compiler_models import CompilerRunModel
from pmmc.infrastructure.database.compiler_repositories import (
    inspect_compiler_run,
    verify_compiler_run,
)
from pmmc.infrastructure.database.experiment_models import (
    BadCaseLabelModel,
    EmbeddingCollectionModel,
    EmbeddingModel,
    EvaluationArtifactModel,
    EvaluationModelCallModel,
    ExperimentJobModel,
    FinalAnswerModel,
    JobAttemptModel,
    JudgeScoreModel,
    MetricScoreModel,
    RetrievalBatchModel,
    RetrievalBatchQueryEmbeddingModel,
    RetrievalEvaluationModel,
    RetrievalTraceComponentModel,
    RetrievalTraceModel,
    RunQueryModel,
    RunStageSummaryModel,
    RunStateModel,
    RuntimeArtifactModel,
    RuntimeModelCallModel,
)
from pmmc.infrastructure.database.experiment_repositories import PostgresRunRepository
from pmmc.infrastructure.database.models import (
    DatasetSnapshotModel,
    RunManifestModel,
    RunSnapshotLinkModel,
)
from pmmc.schemas import (
    SCHEMA_VERSION,
    SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS,
    ArtifactState,
    BenchmarkQuery,
    BenchmarkSelection,
    EmbeddingCollectionStatus,
    JobStatus,
    PipelineRunSummary,
    RunManifest,
    RunQueryAssignment,
    RunStatus,
    TraceStatus,
)


@dataclass(frozen=True)
class LocalRunRegistration:
    """Validated runtime-safe files required to register one local run."""

    manifest: RunManifest
    selection: BenchmarkSelection
    queries: list[BenchmarkQuery]
    summary: PipelineRunSummary | None


def _invalid_embedding_document_lineage_query(
    collection_ids: set[str],
) -> Select:
    """Validate document lineage without loading vectors or rendered documents."""

    return (
        select(func.count())
        .select_from(EmbeddingModel)
        .join(
            EmbeddingCollectionModel,
            EmbeddingCollectionModel.collection_id == EmbeddingModel.collection_id,
        )
        .outerjoin(
            RetrievalDocumentModel,
            RetrievalDocumentModel.document_id == EmbeddingModel.retrieval_document_id,
        )
        .where(
            EmbeddingModel.collection_id.in_(collection_ids),
            or_(
                EmbeddingModel.retrieval_document_id.is_(None),
                EmbeddingModel.source_id != EmbeddingModel.retrieval_document_id,
                EmbeddingCollectionModel.corpus_id.is_(None),
                RetrievalDocumentModel.document_id.is_(None),
                RetrievalDocumentModel.corpus_id != EmbeddingCollectionModel.corpus_id,
                RetrievalDocumentModel.snapshot_id != EmbeddingCollectionModel.snapshot_id,
            ),
        )
    )


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required run artifact is missing: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_queries(path: Path) -> list[BenchmarkQuery]:
    if not path.is_file():
        raise FileNotFoundError(f"required run artifact is missing: {path.name}")
    records: list[BenchmarkQuery] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    records.append(BenchmarkQuery.from_json(line.decode("utf-8")))
                except Exception as exc:
                    raise ValueError(f"invalid selected query at line {line_number}") from exc
    return records


def load_local_run_registration(run_dir: Path) -> LocalRunRegistration:
    """Read and cross-check a gold-isolated local run directory."""

    manifest = RunManifest.model_validate(_load_json(run_dir / "run_manifest.json"))
    selection_payload = _load_json(run_dir / "selection.json")
    if not isinstance(selection_payload, dict):
        raise ValueError("selection.json must contain an object")
    selection = BenchmarkSelection.model_validate(selection_payload.get("selection"))
    queries = _load_queries(run_dir / "selected_queries.jsonl")
    if not queries:
        raise ValueError("experiment registration requires at least one selected query")

    query_ids = [record.query.query_id for record in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("selected_queries.jsonl contains duplicate query IDs")
    if selection_payload.get("query_ids") != query_ids:
        raise ValueError("selection query IDs do not match selected_queries.jsonl order")
    source_locators = [record.source.model_dump(mode="json") for record in queries]
    if selection_payload.get("source_locators") != source_locators:
        raise ValueError("selection source locators do not match selected queries")

    computed_selection_hash = selection_hash(selection, queries)
    if selection_payload.get("selection_hash") != computed_selection_hash:
        raise ValueError("selection.json hash does not match its content")
    if manifest.selection_hash != computed_selection_hash:
        raise ValueError("run manifest selection_hash does not match selection.json")
    if manifest.seed != selection.seed:
        raise ValueError("run manifest seed does not match selection.json")
    if not manifest.dataset_snapshot_ids:
        raise ValueError("run manifest must link at least one canonical snapshot")
    if manifest.config_hash != run_config_hash(manifest):
        raise ValueError("run manifest config_hash does not match its immutable content")
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError("run uses an unsupported canonical schema version")
    if manifest.experiment_schema_version not in SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS:
        raise ValueError("run uses an unsupported experiment schema version")

    summary_path = run_dir / "stage_summary.json"
    summary = (
        PipelineRunSummary.model_validate(_load_json(summary_path))
        if summary_path.is_file()
        else None
    )
    if summary is not None and summary.run_id != manifest.run_id:
        raise ValueError("stage summary run_id does not match manifest")
    return LocalRunRegistration(manifest, selection, queries, summary)


async def register_local_run(
    session: AsyncSession,
    run_dir: Path,
    *,
    include_stage_summaries: bool = True,
) -> tuple[RunStatus, int]:
    """Register a validated local run and freeze its actual query denominator."""

    local = load_local_run_registration(run_dir)
    snapshots = (
        (
            await session.execute(
                select(DatasetSnapshotModel).where(
                    DatasetSnapshotModel.snapshot_id.in_(local.manifest.dataset_snapshot_ids)
                )
            )
        )
        .scalars()
        .all()
    )
    if len(snapshots) != len(local.manifest.dataset_snapshot_ids):
        raise ValueError("one or more run snapshots are not imported")
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for snapshot in snapshots:
        if snapshot.import_state != "ready":
            raise ValueError(f"snapshot is not ready: {snapshot.snapshot_id}")
        by_dataset[snapshot.dataset].append(snapshot.snapshot_id)

    assignments: list[RunQueryAssignment] = []
    for ordinal, record in enumerate(local.queries):
        candidates = by_dataset[record.query.dataset.value]
        if len(candidates) != 1:
            raise ValueError("each selected dataset must map to exactly one linked snapshot")
        assignments.append(
            RunQueryAssignment(
                run_id=local.manifest.run_id,
                snapshot_id=candidates[0],
                query_id=record.query.query_id,
                ordinal=ordinal,
                selection_hash=local.manifest.selection_hash,
                shard_count=local.selection.shard_count,
                shard_index=local.selection.shard_index,
                created_at=local.manifest.created_at,
                metadata={
                    "source": record.source.model_dump(mode="json"),
                    "eligibility": record.eligibility.model_dump(mode="json"),
                },
            )
        )

    repository = PostgresRunRepository(session)
    state = await repository.register(local.manifest, assignments)
    if include_stage_summaries and local.summary is not None:
        for stage in local.summary.stages:
            await repository.add_stage_summary(local.manifest.run_id, stage)
    return state.status, len(assignments)


async def inspect_experiment_run(session: AsyncSession, run_id: str) -> dict[str, Any]:
    """Return only non-sensitive run state and aggregate counts."""

    manifest = await session.get(RunManifestModel, run_id)
    state = await session.get(RunStateModel, run_id)
    if manifest is None or state is None:
        raise KeyError(f"run does not exist: {run_id}")

    async def count(model: Any) -> int:
        value = await session.scalar(
            select(func.count()).select_from(model).where(model.run_id == run_id)
        )
        return int(value or 0)

    job_rows = await session.execute(
        select(ExperimentJobModel.status, func.count())
        .where(ExperimentJobModel.run_id == run_id)
        .group_by(ExperimentJobModel.status)
    )
    job_ids = select(ExperimentJobModel.job_id).where(ExperimentJobModel.run_id == run_id)
    collection_ids = select(EmbeddingCollectionModel.collection_id).where(
        EmbeddingCollectionModel.run_id == run_id
    )
    referenced_corpus_ids = {
        str(value)
        for value in (
            await session.scalars(
                select(RetrievalBatchModel.corpus_id).where(
                    RetrievalBatchModel.run_id == run_id,
                    RetrievalBatchModel.corpus_id.is_not(None),
                )
            )
        ).all()
    }
    referenced_corpus_ids.update(
        str(value)
        for value in (
            await session.scalars(
                select(EmbeddingCollectionModel.corpus_id).where(
                    (
                        (EmbeddingCollectionModel.run_id == run_id)
                        | (EmbeddingCollectionModel.collection_id.in_(collection_ids))
                    ),
                    EmbeddingCollectionModel.corpus_id.is_not(None),
                )
            )
        ).all()
    )
    compiler_inspection: dict[str, object] | None = None
    if await session.get(CompilerRunModel, run_id) is not None:
        compiler_inspection = await inspect_compiler_run(session, run_id)
    return {
        "run_id": run_id,
        "experiment_name": manifest.experiment_name,
        "method": manifest.method,
        "dataset": manifest.dataset,
        "status": state.status,
        "selected_count": state.selected_count,
        "succeeded_count": state.succeeded_count,
        "failed_count": state.failed_count,
        "job_status_counts": {str(status): int(total) for status, total in job_rows},
        "compiler": compiler_inspection,
        "counts": {
            "queries": await count(RunQueryModel),
            "stage_summaries": await count(RunStageSummaryModel),
            "jobs": await count(ExperimentJobModel),
            "job_attempts": int(
                await session.scalar(
                    select(func.count())
                    .select_from(JobAttemptModel)
                    .where(JobAttemptModel.job_id.in_(job_ids))
                )
                or 0
            ),
            "artifacts": await count(RuntimeArtifactModel),
            "embedding_collections": await count(EmbeddingCollectionModel),
            "embeddings": int(
                await session.scalar(
                    select(func.count())
                    .select_from(EmbeddingModel)
                    .where(EmbeddingModel.collection_id.in_(collection_ids))
                )
                or 0
            ),
            "model_calls": await count(RuntimeModelCallModel),
            "retrieval_batches": await count(RetrievalBatchModel),
            "retrieval_candidates": await count(RetrievalTraceModel),
            "answers": await count(FinalAnswerModel),
            "context_assemblies": await count(ContextAssemblyModel),
            "referenced_corpora": len(referenced_corpus_ids),
        },
    }


async def verify_experiment_run(
    session: AsyncSession,
    run_id: str,
    *,
    include_evaluation: bool = False,
) -> dict[str, Any]:
    """Verify denominator, lifecycle, top-k, token, and artifact invariants."""

    manifest_row = await session.get(RunManifestModel, run_id)
    state = await session.get(RunStateModel, run_id)
    if manifest_row is None or state is None:
        raise KeyError(f"run does not exist: {run_id}")
    manifest = RunManifest.model_validate(manifest_row.manifest_json)
    issues: list[str] = []
    warnings: list[str] = []

    if manifest.config_hash != run_config_hash(manifest):
        issues.append("manifest config_hash mismatch")
    if manifest.schema_version != SCHEMA_VERSION:
        issues.append("canonical schema version mismatch")
    if manifest.experiment_schema_version not in SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS:
        issues.append("experiment schema version mismatch")

    assignments = (
        (
            await session.execute(
                select(RunQueryModel)
                .where(RunQueryModel.run_id == run_id)
                .order_by(RunQueryModel.ordinal)
            )
        )
        .scalars()
        .all()
    )
    if [item.ordinal for item in assignments] != list(range(len(assignments))):
        issues.append("run query ordinals are not contiguous from zero")
    if len(assignments) != state.selected_count:
        issues.append("run state selected_count does not match persisted query denominator")
    if any(item.selection_hash != manifest.selection_hash for item in assignments):
        issues.append("run query selection hash differs from manifest")

    links = set(
        (
            await session.scalars(
                select(RunSnapshotLinkModel.snapshot_id).where(
                    RunSnapshotLinkModel.run_id == run_id
                )
            )
        ).all()
    )
    if links != set(manifest.dataset_snapshot_ids):
        issues.append("persisted snapshot links differ from manifest")

    job_statuses = Counter(
        (
            await session.scalars(
                select(ExperimentJobModel.status).where(ExperimentJobModel.run_id == run_id)
            )
        ).all()
    )
    covered_queries = int(
        await session.scalar(
            select(func.count(func.distinct(ExperimentJobModel.query_id))).where(
                ExperimentJobModel.run_id == run_id,
                ExperimentJobModel.query_id.is_not(None),
                ExperimentJobModel.status == JobStatus.SUCCEEDED.value,
            )
        )
        or 0
    )
    if state.status == RunStatus.COMPLETED.value:
        non_succeeded = sum(
            count for status, count in job_statuses.items() if status != JobStatus.SUCCEEDED.value
        )
        if non_succeeded:
            issues.append("completed run contains non-succeeded jobs")
        if covered_queries != len(assignments):
            issues.append("completed run lacks a successful job for every selected query")

    batches = (
        (
            await session.execute(
                select(RetrievalBatchModel)
                .where(RetrievalBatchModel.run_id == run_id)
                .order_by(RetrievalBatchModel.retrieval_batch_id)
            )
        )
        .scalars()
        .all()
    )
    trace_rows = (
        await session.execute(
            select(RetrievalTraceModel.retrieval_batch_id, RetrievalTraceModel.rank)
            .where(RetrievalTraceModel.run_id == run_id)
            .order_by(RetrievalTraceModel.retrieval_batch_id, RetrievalTraceModel.rank)
        )
    ).all()
    ranks_by_batch: dict[str, list[int]] = defaultdict(list)
    for batch_id, rank in trace_rows:
        ranks_by_batch[str(batch_id)].append(int(rank))
    for batch in batches:
        ranks = ranks_by_batch[batch.retrieval_batch_id]
        if ranks != list(range(1, batch.returned_count + 1)):
            issues.append(f"retrieval batch has incomplete ranks: {batch.retrieval_batch_id}")
        if manifest.retrieval_topk_max is not None:
            if batch.requested_top_k > manifest.retrieval_topk_max:
                issues.append(f"retrieval batch exceeds manifest max K: {batch.retrieval_batch_id}")
            elif (
                manifest.paper_mode
                and batch.status == TraceStatus.SUCCESS.value
                and batch.requested_top_k != manifest.retrieval_topk_max
            ):
                issues.append(
                    f"paper-mode retrieval did not persist max K: {batch.retrieval_batch_id}"
                )

    invalid_tokens = int(
        await session.scalar(
            select(func.count())
            .select_from(RuntimeModelCallModel)
            .where(
                RuntimeModelCallModel.run_id == run_id,
                RuntimeModelCallModel.input_tokens.is_not(None),
                RuntimeModelCallModel.output_tokens.is_not(None),
                RuntimeModelCallModel.total_tokens.is_not(None),
                RuntimeModelCallModel.total_tokens
                != RuntimeModelCallModel.input_tokens + RuntimeModelCallModel.output_tokens,
            )
        )
        or 0
    )
    if invalid_tokens:
        issues.append("model call token totals are inconsistent")

    incomplete_collections = int(
        await session.scalar(
            select(func.count())
            .select_from(EmbeddingCollectionModel)
            .where(
                EmbeddingCollectionModel.run_id == run_id,
                EmbeddingCollectionModel.status != EmbeddingCollectionStatus.READY.value,
            )
        )
        or 0
    )
    if state.status == RunStatus.COMPLETED.value and incomplete_collections:
        issues.append("completed run contains non-ready embedding collections")

    collection_ids_for_run = {
        str(value)
        for value in (
            await session.scalars(
                select(RetrievalBatchModel.embedding_collection_id).where(
                    RetrievalBatchModel.run_id == run_id,
                    RetrievalBatchModel.embedding_collection_id.is_not(None),
                )
            )
        ).all()
    }
    collections = (
        (
            await session.execute(
                select(EmbeddingCollectionModel).where(
                    (EmbeddingCollectionModel.run_id == run_id)
                    | (EmbeddingCollectionModel.collection_id.in_(collection_ids_for_run))
                )
            )
        )
        .scalars()
        .all()
    )
    corpus_ids = {
        str(value)
        for value in (
            await session.scalars(
                select(RetrievalBatchModel.corpus_id).where(
                    RetrievalBatchModel.run_id == run_id,
                    RetrievalBatchModel.corpus_id.is_not(None),
                )
            )
        ).all()
    }
    corpus_ids.update(
        str(collection.corpus_id) for collection in collections if collection.corpus_id is not None
    )
    corpora = {
        corpus.corpus_id: corpus
        for corpus in (
            (
                await session.execute(
                    select(RetrievalCorpusModel).where(
                        RetrievalCorpusModel.corpus_id.in_(corpus_ids)
                    )
                )
            )
            .scalars()
            .all()
        )
    }
    missing_corpora = corpus_ids - set(corpora)
    if missing_corpora:
        issues.append("run references missing retrieval corpora")
    for corpus in corpora.values():
        if corpus.status != "ready":
            issues.append(f"run references non-ready corpus: {corpus.corpus_id}")
        if corpus.snapshot_id not in links:
            issues.append(f"corpus snapshot is not linked to run: {corpus.corpus_id}")

    collection_by_id = {collection.collection_id: collection for collection in collections}
    for batch in batches:
        if batch.embedding_collection_id is None:
            continue
        collection = collection_by_id.get(batch.embedding_collection_id)
        if collection is None:
            issues.append(
                f"retrieval batch references a missing embedding collection: "
                f"{batch.retrieval_batch_id}"
            )
            continue
        if collection.snapshot_id != batch.snapshot_id or (
            collection.corpus_id is not None and collection.corpus_id != batch.corpus_id
        ):
            issues.append(
                f"retrieval batch has invalid collection/corpus lineage: {batch.retrieval_batch_id}"
            )

    document_source_types = {
        "retrieval_document_text_caption",
        "retrieval_document_text",
        "retrieval_document_image",
        "retrieval_document_multimodal",
        "retrieval_document_contrastive",
    }
    for collection in collections:
        if collection.source_type in document_source_types and collection.corpus_id is None:
            issues.append(
                f"retrieval-document collection lacks corpus lineage: {collection.collection_id}"
            )

    document_collection_ids = {
        collection.collection_id
        for collection in collections
        if collection.source_type in document_source_types
    }
    if document_collection_ids:
        invalid_embedding_lineage = int(
            await session.scalar(_invalid_embedding_document_lineage_query(document_collection_ids))
            or 0
        )
        if invalid_embedding_lineage:
            issues.append(
                f"retrieval-document embeddings have invalid lineage: {invalid_embedding_lineage}"
            )

    batches_by_id = {batch.retrieval_batch_id: batch for batch in batches}
    linked_trace_rows = (
        (
            await session.execute(
                select(RetrievalTraceModel).where(
                    RetrievalTraceModel.run_id == run_id,
                    RetrievalTraceModel.retrieval_document_id.is_not(None),
                )
            )
        )
        .scalars()
        .all()
    )
    linked_document_ids = {
        item.retrieval_document_id for item in linked_trace_rows if item.retrieval_document_id
    }
    linked_documents = {
        item.document_id: item
        for item in (
            (
                await session.execute(
                    select(RetrievalDocumentModel).where(
                        RetrievalDocumentModel.document_id.in_(linked_document_ids)
                    )
                )
            )
            .scalars()
            .all()
        )
    }
    for trace in linked_trace_rows:
        linked_batch = batches_by_id.get(trace.retrieval_batch_id)
        trace_document_id = trace.retrieval_document_id
        document = (
            linked_documents.get(trace_document_id) if trace_document_id is not None else None
        )
        window_link_valid = (
            trace.retrieval_window_id is not None
            and trace.retrieved_id == trace.retrieval_window_id
            and document is not None
            and document.parent_window_id == trace.retrieval_window_id
        )
        document_link_valid = (
            trace.retrieval_window_id is None and trace.retrieved_id == trace.retrieval_document_id
        )
        if (
            linked_batch is None
            or document is None
            or linked_batch.corpus_id is None
            or document.corpus_id != linked_batch.corpus_id
            or not (window_link_valid or document_link_valid)
        ):
            issues.append(
                f"retrieval trace has invalid corpus/document lineage: {trace.retrieval_trace_id}"
            )

    context_rows = (
        (
            await session.execute(
                select(ContextAssemblyModel).where(ContextAssemblyModel.run_id == run_id)
            )
        )
        .scalars()
        .all()
    )
    if manifest.paper_mode and any(context.estimated for context in context_rows):
        issues.append("paper-mode run contains estimated context token counts")
    for context in context_rows:
        expected = (
            context.instruction_tokens
            + context.query_tokens
            + context.query_image_tokens
            + context.memory_text_tokens
            + context.memory_image_tokens
            + context.request_token_adjustment
        )
        if context.input_tokens != expected or context.input_tokens > context.max_input_tokens:
            issues.append(
                f"context assembly has an invalid token budget: {context.context_assembly_id}"
            )

    model_call_context_rows = (
        await session.execute(
            select(
                RuntimeModelCallModel.model_call_id,
                RuntimeModelCallModel.query_id,
                ContextAssemblyModel.query_id,
            )
            .join(
                ContextAssemblyModel,
                ContextAssemblyModel.context_assembly_id
                == RuntimeModelCallModel.context_assembly_id,
            )
            .where(
                RuntimeModelCallModel.run_id == run_id,
                RuntimeModelCallModel.context_assembly_id.is_not(None),
            )
        )
    ).all()
    if any(call_query != context_query for _, call_query, context_query in model_call_context_rows):
        issues.append("model call query disagrees with linked context assembly")

    artifact_states = Counter(
        (
            await session.scalars(
                select(RuntimeArtifactModel.state).where(RuntimeArtifactModel.run_id == run_id)
            )
        ).all()
    )
    if state.status == RunStatus.COMPLETED.value:
        if artifact_states[ArtifactState.PENDING.value]:
            issues.append("completed run contains pending artifacts")
        if artifact_states[ArtifactState.FAILED.value]:
            issues.append("completed run contains failed artifacts")

    answer_queries = int(
        await session.scalar(
            select(func.count(func.distinct(FinalAnswerModel.query_id))).where(
                FinalAnswerModel.run_id == run_id
            )
        )
        or 0
    )
    typed_contract = manifest.metadata.get("answer_contract_version") == "typed-answer-contract-v3"
    typed_terminals = None
    if typed_contract:
        from pmmc.infrastructure.database.errors import InvalidStateTransitionError

        try:
            typed_terminals = await PostgresRunRepository(session).assert_typed_answer_terminals(
                run_id, str(manifest.method)
            )
            answer_queries = sum(typed_terminals.values())
        except InvalidStateTransitionError:
            issues.append("typed_answer_terminal_integrity_failed")
    if state.status == RunStatus.COMPLETED.value and answer_queries < len(assignments):
        warnings.append("not every selected query has a persisted final answer")

    if manifest.method in {"full_context_captions", "full_context_raw_images"}:
        if len(context_rows) != len(assignments):
            issues.append("full-context run lacks exactly one context per selected query")
        if any(context.method != manifest.method for context in context_rows):
            issues.append("full-context assembly method differs from run manifest")
        answer_count = int(
            await session.scalar(
                select(func.count())
                .select_from(FinalAnswerModel)
                .where(
                    FinalAnswerModel.run_id == run_id,
                    FinalAnswerModel.answer_variant == "primary",
                    FinalAnswerModel.method == manifest.method,
                )
            )
            or 0
        )
        if typed_contract and typed_terminals is not None:
            answer_count = sum(typed_terminals.values())
        if answer_count != len(assignments):
            issues.append("full-context run lacks exactly one primary answer per query")
        successful_call_queries = int(
            await session.scalar(
                select(func.count(func.distinct(RuntimeModelCallModel.query_id))).where(
                    RuntimeModelCallModel.run_id == run_id,
                    RuntimeModelCallModel.stage == "answer",
                    RuntimeModelCallModel.status == TraceStatus.SUCCESS.value,
                    RuntimeModelCallModel.context_assembly_id.is_not(None),
                )
            )
            or 0
        )
        if successful_call_queries != len(assignments):
            issues.append("full-context run lacks a successful audited answer call per query")
        if batches:
            issues.append("full-context run must not contain retrieval batches")
        if any(collection.run_id == run_id for collection in collections):
            issues.append("full-context run must not contain embedding collections")

    rag_methods = {
        "caption_text_rag",
        "contrastive_multimodal_rag",
        "native_multimodal_rag",
    }
    if manifest.method in rag_methods:
        if len(batches) != len(assignments):
            issues.append("RAG run lacks exactly one retrieval batch per selected query")
        if any(
            batch.method != manifest.method
            or batch.requested_top_k != 50
            or batch.embedding_collection_id is None
            or batch.query_embedding_collection_id is None
            or batch.corpus_id is None
            for batch in batches
        ):
            issues.append("RAG retrieval batches violate the frozen top-50 lineage contract")
        batch_ids = [batch.retrieval_batch_id for batch in batches]
        query_refs = list(
            (
                await session.execute(
                    select(RetrievalBatchQueryEmbeddingModel).where(
                        RetrievalBatchQueryEmbeddingModel.retrieval_batch_id.in_(batch_ids)
                    )
                )
            ).scalars()
        )
        refs_by_batch: dict[str, list[Any]] = defaultdict(list)
        for query_ref in query_refs:
            refs_by_batch[query_ref.retrieval_batch_id].append(query_ref)
        if any(
            not refs_by_batch[batch_id]
            or sum(item.used_for_search for item in refs_by_batch[batch_id]) != 1
            for batch_id in batch_ids
        ):
            issues.append("RAG retrieval lacks exactly one effective query vector per batch")
        trace_ids = [item.retrieval_trace_id for item in linked_trace_rows]
        component_rows = list(
            (
                await session.execute(
                    select(RetrievalTraceComponentModel).where(
                        RetrievalTraceComponentModel.retrieval_trace_id.in_(trace_ids)
                    )
                )
            ).scalars()
        )
        components_by_trace: dict[str, list[Any]] = defaultdict(list)
        for component_row in component_rows:
            components_by_trace[component_row.retrieval_trace_id].append(component_row)
        if any(
            not components_by_trace[trace_id]
            or sum(item.is_winner for item in components_by_trace[trace_id]) != 1
            for trace_id in trace_ids
        ):
            issues.append("RAG window traces lack document score components or a winner")
        if len(context_rows) != len(assignments):
            issues.append("RAG run lacks exactly one answer context per selected query")
        if any(
            context.method != manifest.method or context.retrieval_batch_id is None
            for context in context_rows
        ):
            issues.append("RAG context assembly lacks retrieval lineage")
        context_ids = [context.context_assembly_id for context in context_rows]
        candidate_rows = list(
            (
                await session.execute(
                    select(ContextAssemblyRetrievalCandidateModel).where(
                        ContextAssemblyRetrievalCandidateModel.context_assembly_id.in_(context_ids)
                    )
                )
            ).scalars()
        )
        candidate_ranks: dict[str, list[int]] = defaultdict(list)
        for candidate_row in candidate_rows:
            candidate_ranks[candidate_row.context_assembly_id].append(candidate_row.rank)
        if any(
            sorted(candidate_ranks[context_id])
            != list(range(1, len(candidate_ranks[context_id]) + 1))
            for context_id in context_ids
        ):
            issues.append("RAG context candidate ranks are not contiguous")
        answer_count = int(
            await session.scalar(
                select(func.count())
                .select_from(FinalAnswerModel)
                .where(
                    FinalAnswerModel.run_id == run_id,
                    FinalAnswerModel.answer_variant == "primary",
                    FinalAnswerModel.method == manifest.method,
                )
            )
            or 0
        )
        abstained_count = 0
        if typed_contract and typed_terminals is not None:
            answer_count = typed_terminals["answered_count"]
            abstained_count = typed_terminals["abstained_count"]
        if manifest.metadata.get("answer_contract_version") == "strict-answer-contract-v2":
            from pmmc.infrastructure.database.errors import InvalidStateTransitionError

            try:
                terminals = await PostgresRunRepository(session).assert_rag_answer_terminals(
                    run_id, str(manifest.method)
                )
                abstained_count = terminals["abstained_count"]
            except InvalidStateTransitionError:
                issues.append("rag_answer_terminal_integrity_failed")
        if answer_count + abstained_count != len(assignments):
            issues.append(
                "RAG run lacks exactly one primary answer or validated abstention per query"
            )
        successful_answer_queries = int(
            await session.scalar(
                select(func.count(func.distinct(RuntimeModelCallModel.query_id))).where(
                    RuntimeModelCallModel.run_id == run_id,
                    RuntimeModelCallModel.stage == "answer",
                    RuntimeModelCallModel.status == TraceStatus.SUCCESS.value,
                    RuntimeModelCallModel.context_assembly_id.is_not(None),
                )
            )
            or 0
        )
        if successful_answer_queries != len(assignments):
            issues.append("RAG run lacks a successful audited answer call per query")

    if await session.get(CompilerRunModel, run_id) is not None:
        compiler_verification = await verify_compiler_run(session, run_id)
        issues.extend(
            f"compiler: {issue}" for issue in cast(list[str], compiler_verification["issues"])
        )

    inspection = await inspect_experiment_run(session, run_id)
    evaluation_counts: dict[str, int] | None = None
    if include_evaluation:
        evaluation_models: dict[str, Any] = {
            "artifacts": EvaluationArtifactModel,
            "model_calls": EvaluationModelCallModel,
            "retrieval_evaluations": RetrievalEvaluationModel,
            "metric_scores": MetricScoreModel,
            "judge_scores": JudgeScoreModel,
            "bad_case_labels": BadCaseLabelModel,
        }
        evaluation_counts = {}
        for name, model in evaluation_models.items():
            model = cast(Any, model)
            evaluation_counts[name] = int(
                await session.scalar(
                    select(func.count()).select_from(model).where(model.run_id == run_id)
                )
                or 0
            )
        mismatched_metrics = int(
            await session.scalar(
                select(func.count())
                .select_from(MetricScoreModel)
                .join(FinalAnswerModel, FinalAnswerModel.answer_id == MetricScoreModel.answer_id)
                .where(
                    MetricScoreModel.run_id == run_id,
                    (MetricScoreModel.run_id != FinalAnswerModel.run_id)
                    | (MetricScoreModel.query_id != FinalAnswerModel.query_id),
                )
            )
            or 0
        )
        mismatched_judges = int(
            await session.scalar(
                select(func.count())
                .select_from(JudgeScoreModel)
                .join(FinalAnswerModel, FinalAnswerModel.answer_id == JudgeScoreModel.answer_id)
                .where(
                    JudgeScoreModel.run_id == run_id,
                    (JudgeScoreModel.run_id != FinalAnswerModel.run_id)
                    | (JudgeScoreModel.query_id != FinalAnswerModel.query_id),
                )
            )
            or 0
        )
        mismatched_bad_cases = int(
            await session.scalar(
                select(func.count())
                .select_from(BadCaseLabelModel)
                .join(
                    FinalAnswerModel,
                    FinalAnswerModel.answer_id == BadCaseLabelModel.answer_id,
                )
                .where(
                    BadCaseLabelModel.run_id == run_id,
                    (BadCaseLabelModel.run_id != FinalAnswerModel.run_id)
                    | (BadCaseLabelModel.query_id != FinalAnswerModel.query_id),
                )
            )
            or 0
        )
        mismatched_retrieval = int(
            await session.scalar(
                select(func.count())
                .select_from(RetrievalEvaluationModel)
                .join(
                    RetrievalTraceModel,
                    RetrievalTraceModel.retrieval_trace_id
                    == RetrievalEvaluationModel.retrieval_trace_id,
                )
                .where(
                    RetrievalEvaluationModel.run_id == run_id,
                    (RetrievalEvaluationModel.run_id != RetrievalTraceModel.run_id)
                    | (RetrievalEvaluationModel.query_id != RetrievalTraceModel.query_id),
                )
            )
            or 0
        )
        if any(
            (
                mismatched_metrics,
                mismatched_judges,
                mismatched_bad_cases,
                mismatched_retrieval,
            )
        ):
            issues.append("evaluation records disagree with linked runtime identities")
        pending_evaluation_artifacts = int(
            await session.scalar(
                select(func.count())
                .select_from(EvaluationArtifactModel)
                .where(
                    EvaluationArtifactModel.run_id == run_id,
                    EvaluationArtifactModel.state != ArtifactState.READY.value,
                )
            )
            or 0
        )
        if state.status == RunStatus.COMPLETED.value and pending_evaluation_artifacts:
            issues.append("completed run contains non-ready evaluation artifacts")
    return {
        "run_id": run_id,
        "valid": not issues,
        "issues": issues,
        "warnings": warnings,
        "job_status_counts": dict(sorted(job_statuses.items())),
        "artifact_state_counts": dict(sorted(artifact_states.items())),
        "counts": inspection["counts"],
        "evaluation_counts": evaluation_counts,
    }
