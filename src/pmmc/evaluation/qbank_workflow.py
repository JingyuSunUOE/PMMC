"""Evaluator-only QBank matching and evidence coverage workflow."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.baselines.rag_embedding import normalize_vector
from pmmc.benchmarks.selection import select_queries, selection_hash
from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.questioner_config import QuestionerConfig, questioner_config_hash
from pmmc.evaluation.evidence import resolve_gold_evidence
from pmmc.evaluation.qbank_coverage import (
    QBankCandidateEvidence,
    QBankCoverageBundle,
    score_qbank_coverage,
)
from pmmc.evaluation.qbank_protocols import (
    HARDENED_QBANK_COVERAGE_PROTOCOLS,
    QBANK_EVIDENCE_RESOLUTION_V1,
    QBANK_EVIDENCE_RESOLUTION_V2,
)
from pmmc.evaluation.repositories import PostgresGoldRepository
from pmmc.evidence_lineage import EvidenceLineageResolver
from pmmc.experiments import run_config_hash
from pmmc.infrastructure.database.compiler_repositories import (
    PostgresCompilerRunRepository,
)
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.doubter_repositories import (
    PostgresDoubterRepository,
)
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.errors import safe_database_error_diagnostics
from pmmc.infrastructure.database.experiment_repositories import (
    ExperimentUnitOfWork,
    PostgresEmbeddingRepository,
)
from pmmc.infrastructure.database.questioner_models import (
    QBankCoverageCandidateModel,
    QBankCoverageGoldGroupModel,
    QBankCoverageLabelModel,
    QBankCoverageMetricModel,
    QBankCoverageQueryEmbeddingModel,
    QBankCoverageQueryModel,
)
from pmmc.infrastructure.database.questioner_repositories import (
    PostgresQBankIndexRepository,
    PostgresQuestionerRepository,
)
from pmmc.infrastructure.database.repositories import (
    PostgresMemoryRepository,
    PostgresRuntimeQueryRepository,
    PostgresSessionRepository,
)
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.infrastructure.object_store import (
    create_evaluator_object_stores,
    create_runtime_object_store,
)
from pmmc.infrastructure.settings import AppMode, Settings
from pmmc.models.execution import (
    ModelCallContext,
    ModelExecutionService,
    embedding_result_from_artifact,
)
from pmmc.models.factory import (
    create_embedding_gateway,
    create_image_url_resolver,
    create_model_execution_service,
)
from pmmc.models.protocols import EmbeddingGateway
from pmmc.models.registry import ModelRegistry, bind_registry_to_manifest
from pmmc.qbank.matching import QBankMatchInput, rank_qbank_candidates
from pmmc.schemas import (
    AppStage,
    BenchmarkQuery,
    BenchmarkSelection,
    CoverageGateMode,
    DataScope,
    DatasetName,
    EmbeddingInput,
    EmbeddingRequest,
    QBankCoverageCandidate,
    QBankCoverageGateReport,
    QBankCoverageQuery,
    QBankCoverageQueryEmbedding,
    QBankCoverageRunManifest,
    QBankCoverageStatus,
    QBankIndexManifest,
    QBankMatchChannel,
    QuestionCandidate,
    ResolvedGoldEvidenceGroup,
    RunManifest,
    RunStatus,
    TraceStatus,
)

QBANK_QUERY_INSTRUCTION_VERSION = "qbank-memory-query-v1"
QBANK_QUERY_INSTRUCTION = (
    "Instruct: Given a user query about a conversation history, retrieve the future "
    "question candidates whose evidence would answer it.\nQuery: "
)


def evidence_resolution_policy_for_coverage(coverage_protocol_version: str) -> str:
    """Select hardened lineage for Step 6C.2 and later coverage protocols."""

    return (
        QBANK_EVIDENCE_RESOLUTION_V2
        if coverage_protocol_version in HARDENED_QBANK_COVERAGE_PROTOCOLS
        else QBANK_EVIDENCE_RESOLUTION_V1
    )


@dataclass(frozen=True)
class QBankCoverageWorkflowResult:
    coverage_run_id: str
    compiler_run_id: str
    query_count: int
    applicable_query_count: int
    report: QBankCoverageGateReport | None


@dataclass(frozen=True)
class _ScoredCoverageQuery:
    ordinal: int
    coverage_query: QBankCoverageQuery
    gold_groups: tuple[ResolvedGoldEvidenceGroup, ...]
    bundle: QBankCoverageBundle | None


class QBankCoverageScoringError(RuntimeError):
    """Identify one failed score item without exposing query or gold content."""

    def __init__(
        self,
        *,
        query_id: str,
        operation: str,
        cause_type: str,
        cause_code: str,
    ) -> None:
        self.query_id = query_id
        self.operation = operation
        self.cause_type = cause_type
        self.cause_code = cause_code
        super().__init__(
            "QBank coverage scoring failed "
            f"(query_id={query_id}, operation={operation}, cause_type={cause_type}, "
            f"cause_code={cause_code})"
        )


_SAFE_SCORING_CAUSE_CODES = {
    "QBank coverage query facts were not frozen before scoring": "missing_query_facts",
    "QBank coverage metrics are only partially persisted": "partial_metrics",
    "QBank scoring requires a text query embedding": "missing_text_embedding",
    "QBank scoring requires a multimodal query embedding": ("missing_multimodal_embedding"),
    "QBank matcher returned an unknown question": "unknown_qbank_question",
}


def _safe_scoring_cause_code(error: Exception) -> str:
    """Return a stable diagnostic without exposing database or benchmark content."""

    known = _SAFE_SCORING_CAUSE_CODES.get(str(error))
    if known is not None:
        return known
    error_type = type(error).__name__
    if error_type != "RuntimeError":
        stem = error_type.removesuffix("Error").casefold()
        return f"{stem}_error" if stem else "exception"
    fingerprint = hashlib.sha256(str(error).encode("utf-8")).hexdigest()[:12]
    return f"runtime_unclassified_{fingerprint}"


class QBankCoveragePhase(StrEnum):
    ALL = "all"
    REGISTER = "register"
    EMBED_TEXT = "embed_text"
    EMBED_MULTIMODAL = "embed_multimodal"
    SCORE = "score"


class _RuntimeModelCallWriter:
    """Commit gold-free coverage embedding calls through the runtime boundary."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def add(self, trace):  # type: ignore[no-untyped-def]
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            return await uow.model_calls.add(trace)


def _freeze_query_embedding_map(
    selected: Sequence[BenchmarkQuery],
    embeddings: Sequence[QBankCoverageQueryEmbedding],
    *,
    protocol_version: str = "qbank-coverage-v1",
) -> dict[tuple[str, QBankMatchChannel], QBankCoverageQueryEmbedding]:
    """Validate the complete phase boundary before CPU scoring starts."""

    indexed = {(item.query_id, item.channel): item for item in embeddings}
    expected_text = {
        (record.query.query_id, QBankMatchChannel.TEXT_SEMANTIC) for record in selected
    }
    expected_multimodal = {
        (record.query.query_id, QBankMatchChannel.MULTIMODAL_SEMANTIC)
        for record in selected
        if record.query.query_image_refs or protocol_version == "qbank-coverage-v6"
    }
    missing_text = sorted(expected_text - indexed.keys())
    if missing_text:
        raise RuntimeError("QBank scoring requires a text query embedding")
    missing_multimodal = sorted(expected_multimodal - indexed.keys())
    if missing_multimodal:
        raise RuntimeError("QBank scoring requires a multimodal query embedding")
    if indexed.keys() != expected_text | expected_multimodal or len(indexed) != len(embeddings):
        raise RuntimeError("QBank scoring has unexpected or duplicate query embeddings")
    return indexed


async def _load_frozen_query_embedding_map(
    sessions: async_sessionmaker[AsyncSession],
    coverage_run_id: str,
    selected: Sequence[BenchmarkQuery],
    *,
    protocol_version: str = "qbank-coverage-v1",
) -> dict[tuple[str, QBankMatchChannel], QBankCoverageQueryEmbedding]:
    async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
        assert uow.qbank_coverage is not None
        embeddings = await uow.qbank_coverage.list_query_embeddings(coverage_run_id)
    text_count = sum(item.channel is QBankMatchChannel.TEXT_SEMANTIC for item in embeddings)
    multimodal_count = sum(
        item.channel is QBankMatchChannel.MULTIMODAL_SEMANTIC for item in embeddings
    )
    expected_multimodal = (
        len(selected) if protocol_version == "qbank-coverage-v6"
        else sum(bool(item.query.query_image_refs) for item in selected)
    )
    print(
        "QBANK_EMBEDDING_GATE "
        f"protocol={protocol_version} "
        f"text={text_count}/{len(selected)} "
        f"multimodal={multimodal_count}/{expected_multimodal}",
        flush=True,
    )
    return _freeze_query_embedding_map(selected, embeddings, protocol_version=protocol_version)


def _coverage_embedding_audit_manifest(
    *,
    settings: Settings,
    registry: ModelRegistry,
    coverage: QBankCoverageRunManifest,
    channel: QBankMatchChannel,
    model_alias: str,
    query_count: int,
) -> RunManifest:
    """Build a stable runtime scope for evaluator query-embedding provenance."""

    audit_run_id = stable_compiler_id(
        "qbank-coverage-embedding-audit",
        [coverage.coverage_run_id, channel.value],
    )
    manifest = RunManifest(
        run_id=audit_run_id,
        experiment_name="qbank_coverage_embedding_audit",
        dataset=DatasetName(coverage.dataset),
        method="qbank_coverage_embedding_audit",
        paper_mode=settings.mode is AppMode.PAPER,
        embedding_model_aliases=[model_alias],
        selection_hash=coverage.selection_hash,
        evaluation_policy_version=coverage.protocol_version,
        dataset_snapshot_ids=[coverage.snapshot_id],
        created_at=coverage.created_at,
        metadata={
            "coverage_run_id": coverage.coverage_run_id,
            "source_compiler_run_id": coverage.compiler_run_id,
            "channel": channel.value,
            "query_count": query_count,
            "scope": "evaluation_query_embedding",
        },
    )
    manifest = bind_registry_to_manifest(manifest, registry, aliases=[model_alias])
    return manifest.model_copy(update={"config_hash": run_config_hash(manifest)})


async def _ensure_coverage_embedding_audit_run(
    *,
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
    registry: ModelRegistry,
    coverage: QBankCoverageRunManifest,
    channel: QBankMatchChannel,
    model_alias: str,
    query_count: int,
) -> str:
    manifest = _coverage_embedding_audit_manifest(
        settings=settings,
        registry=registry,
        coverage=coverage,
        channel=channel,
        model_alias=model_alias,
        query_count=query_count,
    )
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.runs is not None
        try:
            state = await uow.runs.get_state(manifest.run_id)
        except KeyError:
            state = await uow.runs.register(manifest, [])
        else:
            stored = await uow.runs.get_manifest(manifest.run_id)
            if stored != manifest:
                raise ValueError(
                    "QBank coverage embedding audit run conflicts with its frozen identity"
                )
        if state.status is RunStatus.CREATED:
            await uow.runs.transition(manifest.run_id, RunStatus.RUNNING)
        elif state.status not in {RunStatus.RUNNING, RunStatus.COMPLETED}:
            raise RuntimeError(
                f"QBank coverage embedding audit run is not resumable: {state.status.value}"
            )
    return manifest.run_id


async def _complete_coverage_embedding_audit_run(
    sessions: async_sessionmaker[AsyncSession],
    audit_run_id: str,
) -> None:
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.runs is not None
        state = await uow.runs.get_state(audit_run_id)
        if state.status is RunStatus.RUNNING:
            await uow.runs.transition(audit_run_id, RunStatus.COMPLETED)
        elif state.status is not RunStatus.COMPLETED:
            raise RuntimeError(
                f"QBank coverage embedding audit run cannot be completed from {state.status.value}"
            )


def _select_coverage_denominator(
    queries: list[BenchmarkQuery],
    selection: BenchmarkSelection,
    compiled_conversations: set[str],
) -> list[BenchmarkQuery]:
    """Freeze the benchmark denominator before checking compiler coverage."""

    selected = select_queries(queries, selection)
    if not selected:
        raise ValueError("QBank coverage query denominator is empty")
    if compiled_conversations:
        uncovered = sorted(
            item.query.query_id
            for item in selected
            if item.query.conversation_id not in compiled_conversations
        )
        if uncovered:
            raise ValueError(
                "QBank coverage query denominator contains conversations that were not "
                "compiled; select Questioner conversations from the frozen query "
                f"denominator (uncovered_query_count={len(uncovered)})"
            )
    return selected


def _combine_parent_scores(
    *channels: list[tuple[str, float]],
    top_k: int,
) -> list[tuple[str, float]]:
    values: dict[str, float] = {}
    for channel in channels:
        for qbank_id, score in channel:
            previous = values.get(qbank_id)
            if previous is None or score > previous:
                values[qbank_id] = score
    return sorted(values.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def _ranks(values: list[tuple[str, float]]) -> dict[str, tuple[int, float]]:
    return {qbank_id: (rank, score) for rank, (qbank_id, score) in enumerate(values, start=1)}


def _filter_channel_for_budget(
    values: Sequence[tuple[str, float]],
    *,
    question_by_id: Mapping[str, QuestionCandidate],
    budget: int,
    top_k: int,
) -> list[tuple[str, float]]:
    filtered: list[tuple[str, float]] = []
    for qbank_id, score in values:
        question = question_by_id.get(qbank_id)
        if question is None:
            raise RuntimeError("QBank matcher returned an unknown question")
        if question.generation_rank <= budget:
            filtered.append((qbank_id, score))
    return sorted(filtered, key=lambda item: (-item[1], item[0]))[:top_k]


def _embedding_batch_id(
    coverage_run_id: str,
    channel: QBankMatchChannel,
    model_revision: str,
    query_ids: Sequence[str],
) -> str:
    return stable_compiler_id(
        "qbank-query-embed-batch",
        [coverage_run_id, channel.value, model_revision, *query_ids],
    )


def _embedding_input_hash(
    *,
    query: BenchmarkQuery,
    entry_hash: str,
    include_images: bool,
) -> str:
    return canonical_content_hash(
        {
            "instruction": QBANK_QUERY_INSTRUCTION,
            "query": query.query.raw_query,
            "images": [
                {"image_id": item.image_id, "sha256": item.sha256}
                for item in (query.query.query_image_refs if include_images else [])
            ],
            "model_hash": entry_hash,
        }
    )


async def _recover_embedding_batch_result(
    *,
    call_sessions: async_sessionmaker[AsyncSession],
    artifact_store,
    model_call_ids: Sequence[str],
):  # type: ignore[no-untyped-def]
    for model_call_id in model_call_ids:
        async with ExperimentUnitOfWork(call_sessions, DataScope.RUNTIME) as uow:
            trace = await uow.model_calls.get(model_call_id)
            if trace is None:
                return None, model_call_id
            if trace.status is TraceStatus.FAILED:
                continue
            if trace.status is not TraceStatus.SUCCESS or trace.output_asset_id is None:
                raise RuntimeError("QBank embedding batch has an incomplete model call")
            artifact = await uow.artifacts.get(trace.output_asset_id)
        if artifact is None or artifact.size_bytes is None:
            raise RuntimeError("QBank embedding batch response artifact is not ready")
        remote = await artifact_store.head(artifact.object_key)
        if (
            remote is None
            or remote.sha256 != artifact.sha256
            or remote.size_bytes != artifact.size_bytes
        ):
            raise RuntimeError("QBank embedding batch response artifact failed verification")
        payload = await artifact_store.get_bytes(artifact.object_key)
        return (embedding_result_from_artifact(payload), trace), None
    raise RuntimeError("QBank embedding batch exhausted its model-call attempts")


async def _embed_query_batch(
    *,
    fact_sessions: async_sessionmaker[AsyncSession],
    call_sessions: async_sessionmaker[AsyncSession],
    execution: ModelExecutionService,
    gateway: EmbeddingGateway,
    artifact_store,
    registry: ModelRegistry,
    settings: Settings,
    compiler_run_id: str,
    model_call_run_id: str,
    coverage_run_id: str,
    snapshot_id: str,
    queries: Sequence[BenchmarkQuery],
    alias: str,
    channel: QBankMatchChannel,
    include_images: bool,
    max_call_attempts: int,
) -> tuple[int, bool]:
    entry = registry.get(alias)
    query_ids = [item.query.query_id for item in queries]
    batch_id = _embedding_batch_id(coverage_run_id, channel, entry.model_revision, query_ids)
    existing: list[QBankCoverageQueryEmbedding] = []
    async with ExperimentUnitOfWork(fact_sessions, DataScope.EVALUATION) as uow:
        assert uow.qbank_coverage is not None
        for query_id in query_ids:
            value = await uow.qbank_coverage.get_query_embedding(
                coverage_run_id, query_id, channel.value
            )
            if value is not None:
                existing.append(value)
    if len(existing) == len(queries):
        return len(existing), True
    if existing:
        raise RuntimeError("QBank embedding batch is only partially persisted")

    request = EmbeddingRequest(
        request_id=batch_id,
        model_alias=alias,
        inputs=[
            EmbeddingInput(
                input_id=query.query.query_id,
                text=QBANK_QUERY_INSTRUCTION + query.query.raw_query,
                image_refs=(list(query.query.query_image_refs) if include_images else []),
                metadata={
                    "task": "qbank_coverage_query",
                    "instruction_version": QBANK_QUERY_INSTRUCTION_VERSION,
                },
            )
            for query in queries
        ],
        dimensions=entry.embedding_dimension,
        metadata={
            "scope": "evaluation",
            "channel": channel.value,
            "batch_id": batch_id,
            "persist_normalized_embedding_result": True,
        },
    )
    model_call_ids = [
        stable_compiler_id("qbank-query-embed-call", [batch_id, f"attempt={attempt}"])
        for attempt in range(1, max_call_attempts + 1)
    ]
    recovered, available_call_id = await _recover_embedding_batch_result(
        call_sessions=call_sessions,
        artifact_store=artifact_store,
        model_call_ids=model_call_ids,
    )
    if recovered is None:
        if available_call_id is None:
            raise RuntimeError("QBank embedding batch has no available model-call attempt")
        audited = await execution.embed_audited(
            gateway,
            request,
            ModelCallContext(
                model_call_id=available_call_id,
                run_id=model_call_run_id,
                stage=AppStage.EMBED,
                snapshot_id=snapshot_id,
            ),
        )
        result, trace = audited.result, audited.trace
        reused = False
    else:
        result, trace = recovered
        reused = True
    if result.request_id != batch_id:
        raise ValueError("QBank embedding response request ID does not match its batch")
    registry.verify_actual_model(
        alias,
        result.actual_model_id,
        paper_mode=settings.mode is AppMode.PAPER,
    )
    vectors = {item.input_id: normalize_vector(item.vector) for item in result.vectors}
    if set(vectors) != set(query_ids):
        raise ValueError("QBank embedding batch response does not match its inputs")
    values = [
        QBankCoverageQueryEmbedding(
            query_embedding_id=stable_compiler_id(
                "qbank-query-embedding",
                [coverage_run_id, query.query.query_id, channel.value],
            ),
            coverage_run_id=coverage_run_id,
            query_id=query.query.query_id,
            channel=channel,
            model_alias=alias,
            model_revision=entry.model_revision,
            input_hash=_embedding_input_hash(
                query=query,
                entry_hash=entry.resolved_hash,
                include_images=include_images,
            ),
            dimension=len(vectors[query.query.query_id]),
            vector=vectors[query.query.query_id],
            model_call_id=trace.model_call_id,
            metadata={
                "instruction_version": QBANK_QUERY_INSTRUCTION_VERSION,
                "batch_id": batch_id,
                "batch_size": len(queries),
                "model_call_run_id": trace.run_id,
                "source_compiler_run_id": compiler_run_id,
            },
        )
        for query in queries
    ]
    if any(value.dimension != entry.embedding_dimension for value in values):
        raise ValueError("QBank query embedding has the wrong dimension")
    async with ExperimentUnitOfWork(fact_sessions, DataScope.EVALUATION) as uow:
        assert uow.qbank_coverage is not None
        for value in values:
            await uow.qbank_coverage.add_query_embedding(value)
    return len(values), reused


async def _embed_query_batches(
    *,
    fact_sessions: async_sessionmaker[AsyncSession],
    call_sessions: async_sessionmaker[AsyncSession],
    execution: ModelExecutionService,
    gateway: EmbeddingGateway,
    artifact_store,
    registry: ModelRegistry,
    settings: Settings,
    compiler_run_id: str,
    model_call_run_id: str,
    coverage_run_id: str,
    snapshot_id: str,
    queries: Sequence[BenchmarkQuery],
    alias: str,
    channel: QBankMatchChannel,
    include_images: bool,
    batch_size: int,
    concurrency: int,
    max_call_attempts: int,
    include_text_only_multimodal: bool = False,
) -> None:
    eligible = [
        item
        for item in queries
        if not include_images
        or include_text_only_multimodal
        or item.query.query_image_refs
    ]
    batches = [
        eligible[index : index + batch_size] for index in range(0, len(eligible), batch_size)
    ]
    semaphore = asyncio.Semaphore(concurrency)

    async def run_batch(batch: Sequence[BenchmarkQuery]) -> None:
        async with semaphore:
            await _embed_query_batch(
                fact_sessions=fact_sessions,
                call_sessions=call_sessions,
                execution=execution,
                gateway=gateway,
                artifact_store=artifact_store,
                registry=registry,
                settings=settings,
                compiler_run_id=compiler_run_id,
                model_call_run_id=model_call_run_id,
                coverage_run_id=coverage_run_id,
                snapshot_id=snapshot_id,
                queries=batch,
                alias=alias,
                channel=channel,
                include_images=include_images,
                max_call_attempts=max_call_attempts,
            )

    tasks = [asyncio.create_task(run_batch(batch)) for batch in batches]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _semantic_channel(
    session: AsyncSession,
    *,
    collection_ids: tuple[str, str],
    vector: list[float],
    compiler_run_id: str,
    snapshot_id: str,
    query: BenchmarkQuery,
    budget: int,
    top_k: int,
) -> list[tuple[str, float]]:
    if query.query.available_after_session is None:
        raise ValueError("QBank coverage query is missing its prospective cutoff")
    repository = PostgresEmbeddingRepository(session)
    canonical = await repository.search_qbank_exact(
        collection_ids[0],
        vector,
        compiler_run_id=compiler_run_id,
        snapshot_id=snapshot_id,
        conversation_id=query.query.conversation_id,
        cutoff_session_id=query.query.available_after_session,
        question_budget=budget,
        top_k=top_k,
    )
    paraphrases = await repository.search_qbank_exact(
        collection_ids[1],
        vector,
        compiler_run_id=compiler_run_id,
        snapshot_id=snapshot_id,
        conversation_id=query.query.conversation_id,
        cutoff_session_id=query.query.available_after_session,
        question_budget=budget,
        top_k=top_k,
    )
    return _combine_parent_scores(canonical, paraphrases, top_k=top_k)


def _rank_all_question_budgets(
    *,
    coverage_run_id: str,
    record: BenchmarkQuery,
    config: QuestionerConfig,
    question_by_id: Mapping[str, QuestionCandidate],
    lexical_all: Sequence[tuple[str, float]],
    text_all: Sequence[tuple[str, float]],
    multimodal_all: Sequence[tuple[str, float]],
) -> dict[int, list[QBankCoverageCandidate]]:
    ranked_by_budget = {}
    for budget in config.question_budgets:
        lexical = _filter_channel_for_budget(
            lexical_all,
            question_by_id=question_by_id,
            budget=budget,
            top_k=config.per_channel_top_k,
        )
        text_values = _filter_channel_for_budget(
            text_all,
            question_by_id=question_by_id,
            budget=budget,
            top_k=config.per_channel_top_k,
        )
        multimodal_values = _filter_channel_for_budget(
            multimodal_all,
            question_by_id=question_by_id,
            budget=budget,
            top_k=config.per_channel_top_k,
        )
        lexical_ranks = _ranks(lexical)
        text_ranks = _ranks(text_values)
        multimodal_ranks = _ranks(multimodal_values)
        candidate_ids = sorted(set(lexical_ranks) | set(text_ranks) | set(multimodal_ranks))
        inputs = []
        for qbank_id in candidate_ids:
            question = question_by_id.get(qbank_id)
            if question is None:
                raise RuntimeError("QBank matcher returned an unknown question")
            lexical_value = lexical_ranks.get(qbank_id)
            text_value = text_ranks.get(qbank_id)
            multimodal_value = multimodal_ranks.get(qbank_id)
            inputs.append(
                QBankMatchInput(
                    qbank_id=qbank_id,
                    generation_rank=question.generation_rank,
                    lexical_rank=lexical_value[0] if lexical_value else None,
                    text_semantic_rank=text_value[0] if text_value else None,
                    multimodal_semantic_rank=(multimodal_value[0] if multimodal_value else None),
                    lexical_raw_score=lexical_value[1] if lexical_value else None,
                    text_semantic_raw_score=text_value[1] if text_value else None,
                    multimodal_semantic_raw_score=(
                        multimodal_value[1] if multimodal_value else None
                    ),
                )
            )
        ranked_by_budget[budget] = rank_qbank_candidates(
            coverage_run_id=coverage_run_id,
            query_id=record.query.query_id,
            candidates=inputs,
            question_budget=budget,
            has_query_image=bool(record.query.query_image_refs),
            rrf_constant=config.rrf_constant,
            top_k=config.per_channel_top_k,
        )
    return ranked_by_budget


async def _score_coverage_query(
    *,
    sessions: async_sessionmaker[AsyncSession],
    ordinal: int,
    record: BenchmarkQuery,
    compiler_run_id: str,
    snapshot_id: str,
    coverage_run_id: str,
    index: QBankIndexManifest,
    config: QuestionerConfig,
    questions: Sequence[QuestionCandidate],
    question_by_id: Mapping[str, QuestionCandidate],
    evidence_by_qbank: Mapping[str, QBankCandidateEvidence],
    query_embeddings: Mapping[tuple[str, QBankMatchChannel], QBankCoverageQueryEmbedding],
    search_scan_k: int,
) -> _ScoredCoverageQuery:
    cutoff = record.query.available_after_session
    if cutoff is None:
        raise ValueError("QBank coverage requires query cutoff sessions")
    async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
        assert uow.qbank_coverage is not None
        coverage_query = await uow.qbank_coverage.get_query(coverage_run_id, record.query.query_id)
        groups = tuple(
            await uow.qbank_coverage.list_gold_groups(coverage_run_id, record.query.query_id)
        )
        existing_metrics = await uow.qbank_coverage.list_metrics(
            coverage_run_id, record.query.query_id
        )
    text_embedding = query_embeddings.get((record.query.query_id, QBankMatchChannel.TEXT_SEMANTIC))
    multimodal_embedding = query_embeddings.get(
        (record.query.query_id, QBankMatchChannel.MULTIMODAL_SEMANTIC)
    )
    if coverage_query is None:
        raise RuntimeError("QBank coverage query facts were not frozen before scoring")
    expected_metric_count = len(config.question_budgets) * (len(config.match_ks) * 4 + 2)
    if len(existing_metrics) == expected_metric_count:
        return _ScoredCoverageQuery(ordinal, coverage_query, groups, None)
    if existing_metrics:
        raise RuntimeError("QBank coverage metrics are only partially persisted")
    if text_embedding is None:
        raise RuntimeError("QBank scoring requires a text query embedding")
    if record.query.query_image_refs and multimodal_embedding is None:
        raise RuntimeError("QBank scoring requires a multimodal query embedding")

    visible_sessions = {
        item.session_id
        async for item in PostgresSessionRepository(sessions).iter_visible(
            snapshot_id,
            record.query.conversation_id,
            cutoff,
        )
    }
    visible_memories = [
        item
        async for item in PostgresMemoryRepository(sessions).iter_visible(
            snapshot_id,
            record.query.conversation_id,
            cutoff,
        )
    ]
    async with sessions() as session:
        lexical_repository = PostgresQBankIndexRepository(session)
        max_budget = max(config.question_budgets)
        scan_k = max(1, search_scan_k)
        lexical_all = await lexical_repository.lexical_search(
            compiler_run_id=compiler_run_id,
            snapshot_id=snapshot_id,
            conversation_id=record.query.conversation_id,
            cutoff_session_id=cutoff,
            query=record.query.raw_query,
            question_budget=max_budget,
            top_k=scan_k,
        )
        text_all = await _semantic_channel(
            session,
            collection_ids=(
                index.text_question_collection_id,
                index.text_paraphrase_collection_id,
            ),
            vector=text_embedding.vector,
            compiler_run_id=compiler_run_id,
            snapshot_id=snapshot_id,
            query=record,
            budget=max_budget,
            top_k=scan_k,
        )
        multimodal_all: list[tuple[str, float]] = []
        if multimodal_embedding is not None:
            multimodal_all = await _semantic_channel(
                session,
                collection_ids=(
                    index.multimodal_question_collection_id,
                    index.multimodal_paraphrase_collection_id,
                ),
                vector=multimodal_embedding.vector,
                compiler_run_id=compiler_run_id,
                snapshot_id=snapshot_id,
                query=record,
                budget=max_budget,
                top_k=scan_k,
            )
    allowed_qbank_ids = set(question_by_id)
    lexical_all = [item for item in lexical_all if item[0] in allowed_qbank_ids]
    text_all = [item for item in text_all if item[0] in allowed_qbank_ids]
    multimodal_all = [
        item for item in multimodal_all if item[0] in allowed_qbank_ids
    ]
    ranked_by_budget = _rank_all_question_budgets(
        coverage_run_id=coverage_run_id,
        record=record,
        config=config,
        question_by_id=question_by_id,
        lexical_all=lexical_all,
        text_all=text_all,
        multimodal_all=multimodal_all,
    )
    visible_questions = [
        item
        for item in questions
        if item.conversation_id == record.query.conversation_id
        and item.created_after_session in visible_sessions
    ]
    oracle = {
        budget: [item.qbank_id for item in visible_questions if item.generation_rank <= budget]
        for budget in config.question_budgets
    }
    bundle = score_qbank_coverage(
        coverage_run_id=coverage_run_id,
        query_id=record.query.query_id,
        ranked_by_budget=ranked_by_budget,
        evidence_by_qbank=dict(evidence_by_qbank),
        oracle_qbank_ids_by_budget=oracle,
        gold_groups=groups,
        match_ks=config.match_ks,
        lineage_resolver=EvidenceLineageResolver(visible_memories),
        protocol_version=config.coverage_protocol_version,
    )
    return _ScoredCoverageQuery(ordinal, coverage_query, groups, bundle)


async def _persist_scored_coverage_query(
    sessions: async_sessionmaker[AsyncSession],
    value: _ScoredCoverageQuery,
) -> None:
    if value.bundle is None:
        return
    async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
        assert uow.qbank_coverage is not None
        await uow.qbank_coverage.add_query_bundle(
            query=value.coverage_query,
            embeddings=[],
            gold_groups=list(value.gold_groups),
            candidates=list(value.bundle.candidates),
            metrics=list(value.bundle.metrics),
            labels=list(value.bundle.labels),
        )


async def _score_queries_bounded(
    *,
    selected: Sequence[BenchmarkQuery],
    concurrency: int,
    score_one: Callable[[int, BenchmarkQuery], Awaitable[_ScoredCoverageQuery]],
    persist_one: Callable[[_ScoredCoverageQuery], Awaitable[None]],
    on_query: Callable[[str, int, int], None] | None,
) -> int:
    """Score fixed-size batches and retry a failed read once in isolation.

    Scoring is read-only, so an isolated retry cannot duplicate model calls or
    evaluation facts. Each successful query bundle is then committed atomically
    in denominator order, which also makes long pilot scoring resumable.
    """

    if concurrency < 1:
        raise ValueError("QBank score concurrency must be positive")
    retry_count = 0
    total = len(selected)
    for start in range(0, total, concurrency):
        batch = list(enumerate(selected[start : start + concurrency], start=start))
        tasks: list[asyncio.Future[_ScoredCoverageQuery]] = [
            asyncio.ensure_future(score_one(ordinal, record)) for ordinal, record in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (ordinal, record), result in zip(batch, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result
                retry_count += 1
                try:
                    result = await score_one(ordinal, record)
                except Exception as retry_error:
                    raise QBankCoverageScoringError(
                        query_id=record.query.query_id,
                        operation="compute",
                        cause_type=type(retry_error).__name__,
                        cause_code=_safe_scoring_cause_code(retry_error),
                    ) from retry_error
            try:
                await persist_one(result)
            except Exception as persist_error:
                raise QBankCoverageScoringError(
                    query_id=record.query.query_id,
                    operation="persist",
                    cause_type=type(persist_error).__name__,
                    cause_code=_safe_scoring_cause_code(persist_error),
                ) from persist_error
            if on_query is not None:
                on_query(record.query.query_id, ordinal + 1, total)
    return retry_count


async def run_qbank_coverage(
    *,
    settings: Settings,
    registry: ModelRegistry,
    config: QuestionerConfig,
    compiler_run_id: str,
    selection: BenchmarkSelection,
    coverage_run_id: str | None = None,
    phase: QBankCoveragePhase = QBankCoveragePhase.ALL,
    on_query: Callable[[str, int, int], None] | None = None,
) -> QBankCoverageWorkflowResult:
    """Evaluate a frozen Questioner run without exposing gold to runtime producers."""

    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    active_score_query_id: str | None = None
    active_score_operation = "compute"
    try:
        async with sessions() as session:
            compiler = await PostgresCompilerRunRepository(session).get(compiler_run_id)
            index = await PostgresQBankIndexRepository(session).get_for_run(compiler_run_id)
            all_questions = await PostgresQuestionerRepository(
                session
            ).list_eligible_questions(
                compiler_run_id
            )
            expected = await PostgresQuestionerRepository(session).list_expected_evidence(
                compiler_run_id
            )
        if compiler is None or index is None:
            raise ValueError("QBank coverage requires a frozen Questioner index identity")
        questions = list(all_questions)
        routing_projection_id: str | None = None
        if compiler.compiler_schema_version == "0.6.0":
            async with sessions() as session:
                routing = await PostgresDoubterRepository(
                    session
                ).get_routing_projection(compiler_run_id)
            if routing is None and phase in {
                QBankCoveragePhase.ALL,
                QBankCoveragePhase.SCORE,
            }:
                raise ValueError(
                    "verified-QBank coverage requires the active routing projection"
                )
            if routing is not None:
                routing_manifest, routing_members = routing
                routing_projection_id = routing_manifest.routing_projection_id
                accepted_answerable = {
                    item.qbank_id
                    for item in routing_members
                    if item.route_class.value == "answerable"
                }
                questions = [
                    item
                    for item in all_questions
                    if item.qbank_id in accepted_answerable
                ]
        building_allowed = phase in {
            QBankCoveragePhase.REGISTER,
            QBankCoveragePhase.EMBED_TEXT,
        }
        if index.status.value != "ready" and not (
            building_allowed and index.status.value == "building"
        ):
            raise ValueError(
                "QBank scoring and multimodal embedding require a completed Questioner index"
            )
        all_queries = await PostgresRuntimeQueryRepository(sessions).list_snapshot(
            compiler.snapshot_id
        )
        compiled_conversations = set(compiler.metadata.get("conversation_ids", []))
        selected = _select_coverage_denominator(
            all_queries,
            selection,
            compiled_conversations,
        )
        fingerprint = selection_hash(selection, selected)
        evidence_resolution_policy = evidence_resolution_policy_for_coverage(
            config.coverage_protocol_version
        )
        config_hash = canonical_content_hash(
            {
                "questioner_config_hash": questioner_config_hash(config),
                "compiler_run_id": compiler_run_id,
                "qbank_index_id": index.qbank_index_id,
                "selection_hash": fingerprint,
                "instruction_version": QBANK_QUERY_INSTRUCTION_VERSION,
                "evidence_resolution_policy_version": evidence_resolution_policy,
            }
        )
        resolved_id = coverage_run_id or stable_compiler_id(
            "qbank-coverage", [compiler_run_id, index.qbank_index_id, config_hash]
        )
        manifest = QBankCoverageRunManifest(
            coverage_run_id=resolved_id,
            compiler_run_id=compiler_run_id,
            qbank_index_id=index.qbank_index_id,
            snapshot_id=compiler.snapshot_id,
            dataset=selected[0].query.dataset.value,
            protocol_version=config.coverage_protocol_version,
            selection_hash=fingerprint,
            config_hash=config_hash,
            question_budgets=config.question_budgets,
            match_ks=config.match_ks,
            gate_mode=config.gate_mode,
            metadata={
                "query_ids": [item.query.query_id for item in selected],
                "instruction_version": QBANK_QUERY_INSTRUCTION_VERSION,
                "evidence_resolution_policy_version": evidence_resolution_policy,
            },
        )
        async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
            assert uow.qbank_coverage is not None
            existing = await uow.qbank_coverage.get_run(resolved_id)
            if existing is None:
                await uow.qbank_coverage.add_run(manifest)
                await uow.qbank_coverage.transition(resolved_id, QBankCoverageStatus.RUNNING)
            else:
                expected_existing = manifest.model_copy(
                    update={
                        "completed_at": existing.completed_at,
                        "created_at": existing.created_at,
                        "status": existing.status,
                    }
                )
                if existing != expected_existing:
                    raise ValueError(
                        "coverage resume conflicts with its frozen manifest"
                    )
                manifest = expected_existing
                if existing.status is QBankCoverageStatus.COMPLETED:
                    report = await uow.qbank_coverage.get_report(resolved_id)
                    if report is None:
                        raise RuntimeError("completed QBank coverage run has no report")
                    return QBankCoverageWorkflowResult(
                        resolved_id,
                        compiler_run_id,
                        report.query_count,
                        report.applicable_query_count,
                        report,
                    )
                if existing.status is QBankCoverageStatus.CREATED:
                    await uow.qbank_coverage.transition(resolved_id, QBankCoverageStatus.RUNNING)

        if phase is QBankCoveragePhase.REGISTER:
            return QBankCoverageWorkflowResult(
                resolved_id,
                compiler_run_id,
                len(selected),
                0,
                None,
            )

        runtime_store, _evaluation_store = create_evaluator_object_stores(settings)
        resolver = create_image_url_resolver(settings, runtime_store)
        text_gateway = (
            create_embedding_gateway(
                settings,
                registry,
                config.text_embedding_model_alias,
                image_url_resolver=resolver,
            )
            if phase in {QBankCoveragePhase.ALL, QBankCoveragePhase.EMBED_TEXT}
            else None
        )
        multimodal_gateway = (
            create_embedding_gateway(
                settings,
                registry,
                config.multimodal_embedding_model_alias,
                image_url_resolver=resolver,
            )
            if phase in {QBankCoveragePhase.ALL, QBankCoveragePhase.EMBED_MULTIMODAL}
            else None
        )
        question_by_id = {item.qbank_id: item for item in questions}
        evidence_by_qbank = {
            qbank_id: QBankCandidateEvidence(
                qbank_id=qbank_id,
                memory_ids=frozenset(
                    item.memory_id for item in values if item.memory_id is not None
                ),
                session_ids=frozenset(
                    item.session_id for item in values if item.session_id is not None
                ),
                image_ids=frozenset(item.image_id for item in values if item.image_id is not None),
                evidence_ids=frozenset(item.evidence_id for item in values),
            )
            for qbank_id, values in expected.items()
        }
        gold_repository = PostgresGoldRepository(sessions)

        # Freeze query and gold-resolution facts before batched model calls. REGISTER
        # intentionally returns above this boundary and therefore never reads gold.
        for ordinal, record in enumerate(selected):
            async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
                assert uow.qbank_coverage is not None
                coverage_query = await uow.qbank_coverage.get_query(
                    resolved_id, record.query.query_id
                )
            if coverage_query is not None:
                continue
            cutoff = record.query.available_after_session
            if cutoff is None:
                raise ValueError("QBank coverage requires query cutoff sessions")
            visible_memories = [
                item
                async for item in PostgresMemoryRepository(sessions).iter_visible(
                    compiler.snapshot_id,
                    record.query.conversation_id,
                    cutoff,
                )
            ]
            gold = await gold_repository.get(compiler.snapshot_id, record.query.query_id)
            groups = resolve_gold_evidence(
                evaluation_run_id=resolved_id,
                snapshot_id=compiler.snapshot_id,
                query_id=record.query.query_id,
                dataset=record.query.dataset,
                gold=gold,
                memories=visible_memories,
                policy_version=evidence_resolution_policy_for_coverage(
                    config.coverage_protocol_version
                ),
            )
            coverage_query = QBankCoverageQuery(
                coverage_run_id=resolved_id,
                query_id=record.query.query_id,
                snapshot_id=compiler.snapshot_id,
                conversation_id=record.query.conversation_id,
                cutoff_session_id=cutoff,
                ordinal=ordinal,
                has_query_image=bool(record.query.query_image_refs),
                applicable=bool(groups),
                gold_group_count=len(groups),
                unavailable_reason=None if groups else "gold_evidence_unresolvable",
            )
            async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
                assert uow.qbank_coverage is not None
                await uow.qbank_coverage.add_query_facts(
                    query=coverage_query,
                    gold_groups=list(groups),
                )

        if phase in {
            QBankCoveragePhase.ALL,
            QBankCoveragePhase.EMBED_TEXT,
            QBankCoveragePhase.EMBED_MULTIMODAL,
        }:
            call_engine, call_sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
            artifact_store = create_runtime_object_store(settings)
            artifacts = ExperimentArtifactService(
                call_sessions,
                artifact_store,
                DataScope.RUNTIME,
                artifact_prefix=settings.experiment_artifact_prefix,
            )
            execution = create_model_execution_service(
                settings,
                registry,
                artifact_service=artifacts,
                model_calls=_RuntimeModelCallWriter(call_sessions),
                data_scope=DataScope.RUNTIME,
            )
            try:
                if phase in {QBankCoveragePhase.ALL, QBankCoveragePhase.EMBED_TEXT}:
                    if text_gateway is None:
                        raise RuntimeError("text embedding gateway is unavailable")
                    text_audit_run_id = await _ensure_coverage_embedding_audit_run(
                        sessions=call_sessions,
                        settings=settings,
                        registry=registry,
                        coverage=manifest,
                        channel=QBankMatchChannel.TEXT_SEMANTIC,
                        model_alias=config.text_embedding_model_alias,
                        query_count=len(selected),
                    )
                    await _embed_query_batches(
                        fact_sessions=sessions,
                        call_sessions=call_sessions,
                        execution=execution,
                        gateway=text_gateway,
                        artifact_store=artifact_store,
                        registry=registry,
                        settings=settings,
                        compiler_run_id=compiler_run_id,
                        model_call_run_id=text_audit_run_id,
                        coverage_run_id=resolved_id,
                        snapshot_id=compiler.snapshot_id,
                        queries=selected,
                        alias=config.text_embedding_model_alias,
                        channel=QBankMatchChannel.TEXT_SEMANTIC,
                        include_images=False,
                        batch_size=settings.qbank_text_query_batch_size,
                        concurrency=settings.qbank_embedding_concurrency,
                        max_call_attempts=config.job_max_attempts,
                    )
                    await _complete_coverage_embedding_audit_run(call_sessions, text_audit_run_id)
                if phase in {
                    QBankCoveragePhase.ALL,
                    QBankCoveragePhase.EMBED_MULTIMODAL,
                }:
                    if multimodal_gateway is None:
                        raise RuntimeError("multimodal embedding gateway is unavailable")
                    include_text_only_multimodal = (
                        config.coverage_protocol_version == "qbank-coverage-v6"
                    )
                    multimodal_query_count = (
                        len(selected)
                        if include_text_only_multimodal
                        else sum(bool(item.query.query_image_refs) for item in selected)
                    )
                    multimodal_audit_run_id = await _ensure_coverage_embedding_audit_run(
                        sessions=call_sessions,
                        settings=settings,
                        registry=registry,
                        coverage=manifest,
                        channel=QBankMatchChannel.MULTIMODAL_SEMANTIC,
                        model_alias=config.multimodal_embedding_model_alias,
                        query_count=multimodal_query_count,
                    )
                    await _embed_query_batches(
                        fact_sessions=sessions,
                        call_sessions=call_sessions,
                        execution=execution,
                        gateway=multimodal_gateway,
                        artifact_store=artifact_store,
                        registry=registry,
                        settings=settings,
                        compiler_run_id=compiler_run_id,
                        model_call_run_id=multimodal_audit_run_id,
                        coverage_run_id=resolved_id,
                        snapshot_id=compiler.snapshot_id,
                        queries=selected,
                        alias=config.multimodal_embedding_model_alias,
                        channel=QBankMatchChannel.MULTIMODAL_SEMANTIC,
                        include_images=True,
                        batch_size=settings.qbank_multimodal_query_batch_size,
                        concurrency=settings.qbank_embedding_concurrency,
                        max_call_attempts=config.job_max_attempts,
                        include_text_only_multimodal=include_text_only_multimodal,
                    )
                    await _complete_coverage_embedding_audit_run(
                        call_sessions, multimodal_audit_run_id
                    )
            finally:
                await call_engine.dispose()
        if phase in {
            QBankCoveragePhase.EMBED_TEXT,
            QBankCoveragePhase.EMBED_MULTIMODAL,
        }:
            channel = (
                QBankMatchChannel.TEXT_SEMANTIC
                if phase is QBankCoveragePhase.EMBED_TEXT
                else QBankMatchChannel.MULTIMODAL_SEMANTIC
            )
            for ordinal, record in enumerate(selected):
                if (
                    channel is QBankMatchChannel.MULTIMODAL_SEMANTIC
                    and not record.query.query_image_refs
                    and config.coverage_protocol_version != "qbank-coverage-v6"
                ):
                    if on_query is not None:
                        on_query(record.query.query_id, ordinal + 1, len(selected))
                    continue
                async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
                    assert uow.qbank_coverage is not None
                    embedding = await uow.qbank_coverage.get_query_embedding(
                        resolved_id,
                        record.query.query_id,
                        channel.value,
                    )
                if embedding is None:
                    raise RuntimeError(f"batched {channel.value} query embedding was not persisted")
                if on_query is not None:
                    on_query(record.query.query_id, ordinal + 1, len(selected))
            async with sessions() as session:
                applicable_count = int(
                    await session.scalar(
                        select(func.count())
                        .select_from(QBankCoverageQueryModel)
                        .where(
                            QBankCoverageQueryModel.coverage_run_id == resolved_id,
                            QBankCoverageQueryModel.applicable.is_(True),
                        )
                    )
                    or 0
                )
            return QBankCoverageWorkflowResult(
                resolved_id,
                compiler_run_id,
                len(selected),
                applicable_count,
                None,
            )

        frozen_query_embeddings = await _load_frozen_query_embedding_map(
            sessions,
            resolved_id,
            selected,
            protocol_version=manifest.protocol_version,
        )
        started = time.perf_counter()
        scoring_finished = settings.qbank_score_concurrency > 1
        if scoring_finished:

            async def score_one(ordinal: int, record: BenchmarkQuery) -> _ScoredCoverageQuery:
                return await _score_coverage_query(
                    sessions=sessions,
                    ordinal=ordinal,
                    record=record,
                    compiler_run_id=compiler_run_id,
                    snapshot_id=compiler.snapshot_id,
                    coverage_run_id=resolved_id,
                    index=index,
                    config=config,
                    questions=questions,
                    question_by_id=question_by_id,
                    evidence_by_qbank=evidence_by_qbank,
                    query_embeddings=frozen_query_embeddings,
                    search_scan_k=len(all_questions),
                )

            await _score_queries_bounded(
                selected=selected,
                concurrency=settings.qbank_score_concurrency,
                score_one=score_one,
                persist_one=lambda value: _persist_scored_coverage_query(sessions, value),
                on_query=on_query,
            )
        for ordinal, record in enumerate([] if scoring_finished else selected):
            active_score_query_id = record.query.query_id
            active_score_operation = "compute"
            async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
                assert uow.qbank_coverage is not None
                coverage_query = await uow.qbank_coverage.get_query(
                    resolved_id, record.query.query_id
                )
            cutoff = record.query.available_after_session
            if cutoff is None:
                raise ValueError("QBank coverage requires query cutoff sessions")
            visible_sessions = {
                item.session_id
                async for item in PostgresSessionRepository(sessions).iter_visible(
                    compiler.snapshot_id,
                    record.query.conversation_id,
                    cutoff,
                )
            }
            if coverage_query is None:
                visible_memories = [
                    item
                    async for item in PostgresMemoryRepository(sessions).iter_visible(
                        compiler.snapshot_id,
                        record.query.conversation_id,
                        cutoff,
                    )
                ]
                gold = await gold_repository.get(compiler.snapshot_id, record.query.query_id)
                groups = resolve_gold_evidence(
                    evaluation_run_id=resolved_id,
                    snapshot_id=compiler.snapshot_id,
                    query_id=record.query.query_id,
                    dataset=record.query.dataset,
                    gold=gold,
                    memories=visible_memories,
                    policy_version=evidence_resolution_policy_for_coverage(
                        config.coverage_protocol_version
                    ),
                )
                coverage_query = QBankCoverageQuery(
                    coverage_run_id=resolved_id,
                    query_id=record.query.query_id,
                    snapshot_id=compiler.snapshot_id,
                    conversation_id=record.query.conversation_id,
                    cutoff_session_id=cutoff,
                    ordinal=ordinal,
                    has_query_image=bool(record.query.query_image_refs),
                    applicable=bool(groups),
                    gold_group_count=len(groups),
                    unavailable_reason=(None if groups else "gold_evidence_unresolvable"),
                )
                async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
                    assert uow.qbank_coverage is not None
                    await uow.qbank_coverage.add_query_facts(
                        query=coverage_query,
                        gold_groups=list(groups),
                    )
            else:
                visible_memories = [
                    item
                    async for item in PostgresMemoryRepository(sessions).iter_visible(
                        compiler.snapshot_id,
                        record.query.conversation_id,
                        cutoff,
                    )
                ]
                async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
                    assert uow.qbank_coverage is not None
                    groups = await uow.qbank_coverage.list_gold_groups(
                        resolved_id, record.query.query_id
                    )

            text_embedding = frozen_query_embeddings.get(
                (record.query.query_id, QBankMatchChannel.TEXT_SEMANTIC)
            )
            multimodal_embedding = frozen_query_embeddings.get(
                (record.query.query_id, QBankMatchChannel.MULTIMODAL_SEMANTIC)
            )

            if phase in {QBankCoveragePhase.ALL, QBankCoveragePhase.EMBED_TEXT}:
                if text_embedding is None:
                    raise RuntimeError("batched text query embedding was not persisted")
            if (
                record.query.query_image_refs
                and phase in {QBankCoveragePhase.ALL, QBankCoveragePhase.EMBED_MULTIMODAL}
                and multimodal_embedding is None
            ):
                raise RuntimeError("batched multimodal query embedding was not persisted")

            if phase in {
                QBankCoveragePhase.EMBED_TEXT,
                QBankCoveragePhase.EMBED_MULTIMODAL,
            }:
                if on_query is not None:
                    on_query(record.query.query_id, ordinal + 1, len(selected))
                continue
            if text_embedding is None:
                raise RuntimeError("QBank scoring requires a text query embedding")
            if record.query.query_image_refs and multimodal_embedding is None:
                raise RuntimeError("QBank scoring requires a multimodal query embedding")

            async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
                assert uow.qbank_coverage is not None
                existing_metrics = await uow.qbank_coverage.list_metrics(
                    resolved_id, record.query.query_id
                )
            expected_metric_count = len(config.question_budgets) * (len(config.match_ks) * 4 + 2)
            if len(existing_metrics) == expected_metric_count:
                if on_query is not None:
                    on_query(record.query.query_id, ordinal + 1, len(selected))
                continue

            ranked_by_budget = {}
            async with sessions() as session:
                lexical_repository = PostgresQBankIndexRepository(session)
                max_budget = max(config.question_budgets)
                scan_k = max(1, len(all_questions))
                lexical_all = await lexical_repository.lexical_search(
                    compiler_run_id=compiler_run_id,
                    snapshot_id=compiler.snapshot_id,
                    conversation_id=record.query.conversation_id,
                    cutoff_session_id=cutoff,
                    query=record.query.raw_query,
                    question_budget=max_budget,
                    top_k=scan_k,
                )
                text_all = await _semantic_channel(
                    session,
                    collection_ids=(
                        index.text_question_collection_id,
                        index.text_paraphrase_collection_id,
                    ),
                    vector=text_embedding.vector,
                    compiler_run_id=compiler_run_id,
                    snapshot_id=compiler.snapshot_id,
                    query=record,
                    budget=max_budget,
                    top_k=scan_k,
                )
                multimodal_all: list[tuple[str, float]] = []
                if multimodal_embedding is not None:
                    multimodal_all = await _semantic_channel(
                        session,
                        collection_ids=(
                            index.multimodal_question_collection_id,
                            index.multimodal_paraphrase_collection_id,
                        ),
                        vector=multimodal_embedding.vector,
                        compiler_run_id=compiler_run_id,
                        snapshot_id=compiler.snapshot_id,
                        query=record,
                        budget=max_budget,
                        top_k=scan_k,
                    )
                allowed_qbank_ids = set(question_by_id)
                lexical_all = [
                    item for item in lexical_all if item[0] in allowed_qbank_ids
                ]
                text_all = [
                    item for item in text_all if item[0] in allowed_qbank_ids
                ]
                multimodal_all = [
                    item
                    for item in multimodal_all
                    if item[0] in allowed_qbank_ids
                ]
                for budget in config.question_budgets:
                    lexical = _filter_channel_for_budget(
                        lexical_all,
                        question_by_id=question_by_id,
                        budget=budget,
                        top_k=config.per_channel_top_k,
                    )
                    text_values = _filter_channel_for_budget(
                        text_all,
                        question_by_id=question_by_id,
                        budget=budget,
                        top_k=config.per_channel_top_k,
                    )
                    multimodal_values = _filter_channel_for_budget(
                        multimodal_all,
                        question_by_id=question_by_id,
                        budget=budget,
                        top_k=config.per_channel_top_k,
                    )
                    lexical_ranks = _ranks(lexical)
                    text_ranks = _ranks(text_values)
                    multimodal_ranks = _ranks(multimodal_values)
                    candidate_ids = sorted(
                        set(lexical_ranks) | set(text_ranks) | set(multimodal_ranks)
                    )
                    inputs = []
                    for qbank_id in candidate_ids:
                        question = question_by_id.get(qbank_id)
                        if question is None:
                            raise RuntimeError("QBank matcher returned an unknown question")
                        lexical_value = lexical_ranks.get(qbank_id)
                        text_value = text_ranks.get(qbank_id)
                        multimodal_value = multimodal_ranks.get(qbank_id)
                        inputs.append(
                            QBankMatchInput(
                                qbank_id=qbank_id,
                                generation_rank=question.generation_rank,
                                lexical_rank=lexical_value[0] if lexical_value else None,
                                text_semantic_rank=text_value[0] if text_value else None,
                                multimodal_semantic_rank=(
                                    multimodal_value[0] if multimodal_value else None
                                ),
                                lexical_raw_score=(lexical_value[1] if lexical_value else None),
                                text_semantic_raw_score=(text_value[1] if text_value else None),
                                multimodal_semantic_raw_score=(
                                    multimodal_value[1] if multimodal_value else None
                                ),
                            )
                        )
                    ranked_by_budget[budget] = rank_qbank_candidates(
                        coverage_run_id=resolved_id,
                        query_id=record.query.query_id,
                        candidates=inputs,
                        question_budget=budget,
                        has_query_image=bool(record.query.query_image_refs),
                        rrf_constant=config.rrf_constant,
                        top_k=config.per_channel_top_k,
                    )
            visible_questions = [
                item
                for item in questions
                if item.conversation_id == record.query.conversation_id
                and item.created_after_session in visible_sessions
            ]
            oracle = {
                budget: [
                    item.qbank_id for item in visible_questions if item.generation_rank <= budget
                ]
                for budget in config.question_budgets
            }
            bundle = score_qbank_coverage(
                coverage_run_id=resolved_id,
                query_id=record.query.query_id,
                ranked_by_budget=ranked_by_budget,
                evidence_by_qbank=evidence_by_qbank,
                oracle_qbank_ids_by_budget=oracle,
                gold_groups=groups,
                match_ks=config.match_ks,
                lineage_resolver=EvidenceLineageResolver(visible_memories),
                protocol_version=config.coverage_protocol_version,
            )
            active_score_operation = "persist"
            async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
                assert uow.qbank_coverage is not None
                await uow.qbank_coverage.add_query_bundle(
                    query=coverage_query,
                    embeddings=[],
                    gold_groups=list(groups),
                    candidates=list(bundle.candidates),
                    metrics=list(bundle.metrics),
                    labels=list(bundle.labels),
                )
            if on_query is not None:
                on_query(record.query.query_id, ordinal + 1, len(selected))
        active_score_query_id = None

        if phase in {
            QBankCoveragePhase.EMBED_TEXT,
            QBankCoveragePhase.EMBED_MULTIMODAL,
        }:
            async with sessions() as session:
                applicable_count = int(
                    await session.scalar(
                        select(func.count())
                        .select_from(QBankCoverageQueryModel)
                        .where(
                            QBankCoverageQueryModel.coverage_run_id == resolved_id,
                            QBankCoverageQueryModel.applicable.is_(True),
                        )
                    )
                    or 0
                )
            return QBankCoverageWorkflowResult(
                resolved_id,
                compiler_run_id,
                len(selected),
                applicable_count,
                None,
            )

        async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
            assert uow.qbank_coverage is not None
            metrics = await uow.qbank_coverage.list_metrics(resolved_id)
        summaries: dict[str, list[float]] = {}
        for metric in metrics:
            if metric.applicable and metric.value is not None:
                key = ":".join(
                    [
                        metric.metric_name,
                        f"b={metric.question_budget}",
                        f"k={metric.k}" if metric.k is not None else "oracle",
                    ]
                )
                summaries.setdefault(key, []).append(metric.value)
        metric_summary = {
            key: sum(values) / len(values) for key, values in sorted(summaries.items())
        }
        label_counts: Counter[str] = Counter()
        async with sessions() as session:
            labels = list(
                await session.scalars(
                    select(QBankCoverageLabelModel).where(
                        QBankCoverageLabelModel.coverage_run_id == resolved_id
                    )
                )
            )
        label_counts.update(item.category for item in labels)
        thresholds_passed = None
        threshold_summary = dict(config.scientific_thresholds)
        if config.gate_mode is CoverageGateMode.ENFORCE:
            thresholds_passed = all(
                metric_summary.get(name, -1.0) >= threshold
                for name, threshold in config.scientific_thresholds.items()
            )
        report_payload = {
            "coverage_run_id": resolved_id,
            "routing_projection_id": routing_projection_id,
            "metric_summary": metric_summary,
            "threshold_summary": threshold_summary,
            "failure_counts": dict(sorted(label_counts.items())),
            "query_count": len(selected),
            "applicable_query_count": 0,
        }
        completion_check = await verify_qbank_coverage_run(
            settings, resolved_id, require_terminal=False
        )
        completion_issues = completion_check["issues"]
        assert isinstance(completion_issues, list)
        print(
            "QBANK_COMPLETION_PREFLIGHT "
            + json.dumps(
                {
                    "state": "ready" if completion_check["valid"] else "failed",
                    "protocol": manifest.protocol_version,
                    "query_count": completion_check["query_count"],
                    "issue_count": len(completion_issues),
                    "next_stages": ["database_completion", "terminal_verification"],
                }, sort_keys=True, separators=(",", ":"),
            ),
            flush=True,
        )
        if not completion_check["valid"]:
            raise RuntimeError("QBank coverage completion preflight failed")
        async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
            assert uow.qbank_coverage is not None
            applicable_count = 0
            for record in selected:
                query = await uow.qbank_coverage.get_query(resolved_id, record.query.query_id)
                if query is not None and query.applicable:
                    applicable_count += 1
            report_payload["applicable_query_count"] = applicable_count
            report = QBankCoverageGateReport(
                gate_report_id=stable_compiler_id(
                    "qbank-coverage-report", [resolved_id, config.coverage_protocol_version]
                ),
                coverage_run_id=resolved_id,
                gate_version=config.coverage_protocol_version,
                mode=config.gate_mode,
                hard_integrity_passed=True,
                scientific_thresholds_passed=thresholds_passed,
                query_count=len(selected),
                applicable_query_count=applicable_count,
                metric_summary=metric_summary,
                threshold_summary=threshold_summary,
                failure_counts=dict(sorted(label_counts.items())),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                content_hash=canonical_content_hash(report_payload),
                metadata={
                    "routing_projection_id": routing_projection_id,
                    "route_class": (
                        "answerable" if routing_projection_id is not None else None
                    ),
                },
            )
            await uow.qbank_coverage.add_report(report)
            await uow.qbank_coverage.transition(resolved_id, QBankCoverageStatus.COMPLETED)
        return QBankCoverageWorkflowResult(
            resolved_id,
            compiler_run_id,
            len(selected),
            report.applicable_query_count,
            report,
        )
    except QBankCoverageScoringError:
        raise
    except Exception as error:
        if phase is QBankCoveragePhase.SCORE and active_score_query_id is None:
            print(
                "QBANK_COMPLETION_FAILURE "
                + json.dumps(safe_database_error_diagnostics(error), sort_keys=True),
                flush=True,
            )
        if phase is QBankCoveragePhase.SCORE and active_score_query_id is not None:
            raise QBankCoverageScoringError(
                query_id=active_score_query_id,
                operation=active_score_operation,
                cause_type=type(error).__name__,
                cause_code=_safe_scoring_cause_code(error),
            ) from error
        raise
    finally:
        await engine.dispose()


async def inspect_qbank_coverage_query(
    settings: Settings,
    coverage_run_id: str,
    query_id: str,
) -> dict[str, object]:
    """Inspect evaluator-private QBank matching and evidence-coverage facts."""

    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    try:
        async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
            assert uow.qbank_coverage is not None
            query = await uow.qbank_coverage.get_query(coverage_run_id, query_id)
            if query is None:
                raise KeyError(f"QBank coverage query does not exist: {query_id}")
            metrics = await uow.qbank_coverage.list_metrics(coverage_run_id, query_id)
            gold_groups = await uow.qbank_coverage.list_gold_groups(coverage_run_id, query_id)
            candidates = {
                str(budget): [
                    item.model_dump(mode="json")
                    for item in await uow.qbank_coverage.list_candidates(
                        coverage_run_id,
                        query_id,
                        budget,
                    )
                ]
                for budget in (1, 2, 4, 8)
            }
        return {
            "query": query.model_dump(mode="json"),
            "gold_groups": [item.model_dump(mode="json") for item in gold_groups],
            "candidates": candidates,
            "metrics": [item.model_dump(mode="json") for item in metrics],
        }
    finally:
        await engine.dispose()


def _expected_coverage_metric_keys(
    protocol_version: str, budgets: Sequence[int], match_ks: Sequence[int]
) -> set[tuple[str, int, int | None]]:
    hardened = protocol_version in HARDENED_QBANK_COVERAGE_PROTOCOLS
    names = (
        ("qbank_evidence_recall_at_k", "qbank_evidence_proxy_cover_at_k",
         "qbank_evidence_proxy_match_at_k", "proxy_projected_fallback_at_k")
        if hardened else
        ("qbank_evidence_recall_at_k", "qbank_cover_at_k",
         "qbank_match_at_k", "projected_fallback_at_k")
    )
    oracles = (
        ("oracle_qbank_evidence_proxy_cover", "oracle_qbank_evidence_proxy_match")
        if hardened else ("oracle_qbank_cover", "oracle_qbank_match")
    )
    return {
        (name, budget, k) for budget in budgets for k in match_ks for name in names
    } | {(name, budget, None) for budget in budgets for name in oracles}


async def verify_qbank_coverage_run(
    settings: Settings,
    coverage_run_id: str,
    *,
    require_terminal: bool = True,
) -> dict[str, object]:
    """Verify the frozen denominator, metric grid, embeddings, ranks, and gate report."""

    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    issues: list[str] = []
    try:
        async with ExperimentUnitOfWork(sessions, scope=DataScope.EVALUATION) as uow:
            assert uow.qbank_coverage is not None
            run = await uow.qbank_coverage.get_run(coverage_run_id)
            report = await uow.qbank_coverage.get_report(coverage_run_id)
        if run is None:
            raise KeyError(f"QBank coverage run does not exist: {coverage_run_id}")
        if require_terminal and run.status is not QBankCoverageStatus.COMPLETED:
            issues.append("coverage run is not completed")
        if require_terminal and report is None:
            issues.append("coverage run has no gate report")
        expected_evidence_policy = evidence_resolution_policy_for_coverage(
            run.protocol_version
        )
        if (
            run.metadata.get("evidence_resolution_policy_version")
            != expected_evidence_policy
        ):
            issues.append("coverage evidence-resolution identity differs")
        async with sessions() as session:
            queries = list(
                await session.scalars(
                    select(QBankCoverageQueryModel)
                    .where(QBankCoverageQueryModel.coverage_run_id == coverage_run_id)
                    .order_by(QBankCoverageQueryModel.ordinal)
                )
            )
            if not queries or [item.ordinal for item in queries] != list(range(len(queries))):
                issues.append("coverage query ordinals are not contiguous")
            mismatched_gold_policy_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(QBankCoverageGoldGroupModel)
                    .where(
                        QBankCoverageGoldGroupModel.coverage_run_id
                        == coverage_run_id,
                        QBankCoverageGoldGroupModel.resolution_policy_version
                        != expected_evidence_policy,
                    )
                )
                or 0
            )
            if mismatched_gold_policy_count:
                issues.append("coverage gold groups use another evidence policy")
            # Batch the fixed denominator checks; do not issue per-query/per-budget reads.
            all_metrics = (await session.execute(
                select(
                    QBankCoverageMetricModel.query_id, QBankCoverageMetricModel.metric_name,
                    QBankCoverageMetricModel.question_budget, QBankCoverageMetricModel.k,
                    QBankCoverageMetricModel.applicable,
                ).where(QBankCoverageMetricModel.coverage_run_id == coverage_run_id)
            )).all()
            all_embeddings = (await session.execute(
                select(
                    QBankCoverageQueryEmbeddingModel.query_id,
                    QBankCoverageQueryEmbeddingModel.channel,
                    QBankCoverageQueryEmbeddingModel.dimension,
                ).where(QBankCoverageQueryEmbeddingModel.coverage_run_id == coverage_run_id)
            )).all()
            all_ranks = (await session.execute(
                select(
                    QBankCoverageCandidateModel.query_id,
                    QBankCoverageCandidateModel.question_budget,
                    QBankCoverageCandidateModel.rank,
                ).where(QBankCoverageCandidateModel.coverage_run_id == coverage_run_id)
            )).all()
            gold_counts = {query_id: count for query_id, count in (await session.execute(
                select(QBankCoverageGoldGroupModel.query_id, func.count())
                .where(QBankCoverageGoldGroupModel.coverage_run_id == coverage_run_id)
                .group_by(QBankCoverageGoldGroupModel.query_id)
            )).all()}
            expected_keys = _expected_coverage_metric_keys(
                run.protocol_version, run.question_budgets, run.match_ks
            )
            for query in queries:
                query_metrics = [item for item in all_metrics if item.query_id == query.query_id]
                actual_keys = {
                    (item.metric_name, item.question_budget, item.k) for item in query_metrics
                }
                if actual_keys != expected_keys or len(query_metrics) != len(expected_keys):
                    issues.append(f"coverage metric grid is incomplete: {query.query_id}")
                if any(item.applicable != query.applicable for item in query_metrics):
                    issues.append("coverage metric applicability differs")
                if gold_counts.get(query.query_id, 0) != query.gold_group_count:
                    issues.append("coverage gold group denominator differs")
                query_embeddings = [
                    item for item in all_embeddings if item.query_id == query.query_id
                ]
                embedding_channels = {item.channel for item in query_embeddings}
                expected_channels = {QBankMatchChannel.TEXT_SEMANTIC.value}
                if query.has_query_image or run.protocol_version == "qbank-coverage-v6":
                    expected_channels.add(QBankMatchChannel.MULTIMODAL_SEMANTIC.value)
                if (embedding_channels != expected_channels
                    or len(query_embeddings) != len(expected_channels)
                    or any(item.dimension != 4096 for item in query_embeddings)):
                    issues.append(
                        f"coverage query embedding channels are incomplete: {query.query_id}"
                    )
                for budget in run.question_budgets:
                    ranks = sorted(
                        item.rank for item in all_ranks
                        if item.query_id == query.query_id and item.question_budget == budget
                    )
                    if ranks != list(range(1, len(ranks) + 1)):
                        issues.append(
                            "coverage candidate ranks are not contiguous: "
                            f"{query.query_id}/b={budget}"
                        )
                    if len(ranks) > 50:
                        issues.append(
                            f"coverage candidate top-k exceeds 50: {query.query_id}/b={budget}"
                        )
        if report is not None:
            applicable = sum(item.applicable for item in queries)
            if report.query_count != len(queries):
                issues.append("gate report query count differs from persisted denominator")
            if report.applicable_query_count != applicable:
                issues.append("gate report applicable count differs from query facts")
            if not report.hard_integrity_passed:
                issues.append("gate report did not pass hard integrity checks")
        return {
            "coverage_run_id": coverage_run_id,
            "compiler_run_id": run.compiler_run_id,
            "status": run.status.value,
            "query_count": len(queries),
            "valid": not issues,
            "issues": issues,
        }
    finally:
        await engine.dispose()
