"""Recoverable text and multimodal query embeddings for Step 7 matching."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.baselines.rag_embedding import normalize_vector
from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.experiments.manifest import run_config_hash
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.repositories import PostgresRuntimeQueryRepository
from pmmc.models.execution import (
    ModelCallContext,
    ModelExecutionService,
    embedding_result_from_artifact,
)
from pmmc.models.protocols import EmbeddingGateway
from pmmc.models.registry import ModelRegistry, ModelRegistryEntry
from pmmc.schemas import (
    AppStage,
    DataScope,
    DatasetName,
    DistanceMetric,
    EmbeddingCollection,
    EmbeddingCollectionStatus,
    EmbeddingInput,
    EmbeddingRecord,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSourceType,
    ExperimentJob,
    JobStatus,
    QBankCoverageQueryEmbedding,
    QBankMatchChannel,
    QBankRuntimeChannel,
    QueryEmbeddingBatch,
    QueryEmbeddingFact,
    QueryExecutorRunManifest,
    QueryInput,
    RunManifest,
    RunStatus,
    TraceStatus,
)


class QueryEmbeddingOwner(Protocol):
    @property
    def query_executor_run_id(self) -> str: ...

    @property
    def snapshot_id(self) -> str: ...

    @property
    def schema_version(self) -> str: ...


QUERY_MATCH_INSTRUCTION_VERSION = "qbank-memory-query-v1"
QUERY_MATCH_INSTRUCTION = (
    "Instruct: Given a user query about a conversation history, retrieve the future "
    "question candidates whose evidence would answer it.\nQuery: "
)
QUERY_EMBEDDING_PREPROCESSING_VERSION = "pmmc-query-embedding-v1"

EmbeddingProgress = Callable[[str, int, int, bool], None]


@dataclass(frozen=True)
class RuntimeQueryCacheOwner:
    query_executor_run_id: str
    snapshot_id: str
    schema_version: str = "0.3.0"


@dataclass(frozen=True)
class RuntimeQueryCacheLease:
    owner: RuntimeQueryCacheOwner
    job_id: str
    worker_id: str
    lease_seconds: int


@asynccontextmanager
async def runtime_query_cache_lease(
    sessions: async_sessionmaker[AsyncSession],
    *,
    cache_id: str,
    snapshot_id: str,
    dataset: DatasetName,
    model_alias: str,
    lease_seconds: int,
) -> AsyncIterator[RuntimeQueryCacheLease | None]:
    """Own one immutable cache shard, independently of compiler/query runs."""
    if lease_seconds < 3:
        raise ValueError("runtime query cache lease must be at least three seconds")
    run_id = stable_compiler_id("runtime-query-cache-owner", cache_id)
    owner = RuntimeQueryCacheOwner(run_id, snapshot_id)
    expected = RunManifest(
        run_id=run_id, experiment_name="runtime-query-embedding-cache",
        dataset=dataset, method="runtime_query_embedding_cache",
        embedding_model_aliases=[model_alias], dataset_snapshot_ids=[snapshot_id],
        selection_hash=canonical_content_hash([cache_id]),
        metadata={"runtime_cache_id": cache_id, "paper_result": False},
    )
    expected = expected.model_copy(update={"config_hash": run_config_hash(expected)})
    job_id = stable_compiler_id("runtime-query-cache-job", cache_id)
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.session is not None and uow.runs is not None and uow.jobs is not None
        # Serialize registration only; no database transaction spans provider I/O.
        await uow.session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:identity, 0))"),
            {"identity": run_id},
        )
        try:
            stored = await uow.runs.get_manifest(run_id)
        except KeyError:
            await uow.runs.register(expected, [])
        else:
            excluded = {"config_hash", "created_at"}
            if stored.config_hash != run_config_hash(stored) or (
                stored.model_dump(exclude=excluded) != expected.model_dump(exclude=excluded)
            ):
                raise ValueError("runtime cache owner identity conflict")
        state = await uow.runs.get_state(run_id)
        if state.status is RunStatus.CREATED:
            await uow.jobs.add(ExperimentJob(
                job_id=job_id, run_id=run_id, snapshot_id=snapshot_id,
                stage=AppStage.EMBED, variant="runtime_cache", max_attempts=2,
            ))
            await uow.runs.transition(run_id, RunStatus.RUNNING)
        elif state.status not in {RunStatus.RUNNING, RunStatus.COMPLETED}:
            raise RuntimeError("runtime cache owner is not resumable")

    worker_id = "query-cache-" + uuid4().hex
    while True:
        async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
            assert uow.jobs is not None and uow.runs is not None
            state = await uow.runs.get_state(run_id)
            if state.status is RunStatus.COMPLETED:
                job = None
                complete = True
            else:
                complete = False
                job = await uow.jobs.claim(
                    worker_id, lease_seconds, run_id=run_id, stage=AppStage.EMBED
                )
                current = await uow.jobs.get(job_id)
                if current is None or current.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
                    raise RuntimeError("runtime query cache exhausted its attempts")
        if complete:
            yield None
            return
        if job is not None:
            break
        await asyncio.sleep(0.1)

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(lease_seconds / 3)
            async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                assert uow.jobs is not None
                await uow.jobs.heartbeat(job_id, worker_id, lease_seconds)

    task = asyncio.create_task(heartbeat())
    try:
        yield RuntimeQueryCacheLease(owner, job_id, worker_id, lease_seconds)
    except Exception as exc:
        async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
            assert uow.jobs is not None
            current = await uow.jobs.get(job_id)
            if current is not None and current.claimed_by == worker_id:
                await uow.jobs.finish(
                    job_id, worker_id, success=False, error_type=type(exc).__name__,
                    error_summary="runtime query cache preparation failed",
                )
        raise
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def required_query_embedding_channels(
    query: QueryInput,
    *,
    include_text_only_multimodal: bool = False,
) -> frozenset[QBankRuntimeChannel]:
    """Return the runtime vector channels that can contribute to this query."""

    channels = {QBankRuntimeChannel.TEXT}
    if query.query_image_refs or include_text_only_multimodal:
        channels.add(QBankRuntimeChannel.MULTIMODAL)
    return frozenset(channels)


@dataclass(frozen=True)
class QueryEmbeddingBuildResult:
    query_executor_run_id: str
    text_collection_id: str
    multimodal_collection_id: str
    query_count: int
    model_call_count: int
    reused_batch_count: int


@dataclass(frozen=True)
class QueryEmbeddingChannelResult:
    query_executor_run_id: str
    channel: QBankRuntimeChannel
    collection_id: str
    query_count: int
    model_call_count: int
    reused_batch_count: int


class QueryEmbeddingWorkflow:
    """Build both runtime query spaces without exposing evaluator-only facts."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        execution: ModelExecutionService,
        registry: ModelRegistry,
        text_gateway: EmbeddingGateway | None = None,
        multimodal_gateway: EmbeddingGateway | None = None,
        text_model_alias: str = "qwen3-embedding-8b",
        multimodal_model_alias: str = "qwen3-vl-embedding-8b",
        text_batch_size: int = 64,
        multimodal_batch_size: int = 16,
        concurrency: int = 32,
        max_call_attempts: int = 2,
        include_text_only_multimodal: bool = False,
        reader_sessions: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        if min(text_batch_size, multimodal_batch_size, concurrency, max_call_attempts) <= 0:
            raise ValueError("query embedding execution controls must be positive")
        if execution.data_scope is not DataScope.RUNTIME:
            raise ValueError("query embeddings require a runtime model execution service")
        self.sessions = sessions
        self.execution = execution
        self.registry = registry
        self.text_gateway = text_gateway
        self.multimodal_gateway = multimodal_gateway
        self.text_model_alias = text_model_alias
        self.multimodal_model_alias = multimodal_model_alias
        self.text_batch_size = text_batch_size
        self.multimodal_batch_size = multimodal_batch_size
        self.concurrency = concurrency
        self.max_call_attempts = max_call_attempts
        self.include_text_only_multimodal = include_text_only_multimodal
        self.queries = PostgresRuntimeQueryRepository(
            reader_sessions if reader_sessions is not None else sessions
        )

    async def build(
        self,
        manifest: QueryExecutorRunManifest,
        query_ids: Sequence[str],
        *,
        on_progress: EmbeddingProgress | None = None,
    ) -> QueryEmbeddingBuildResult:
        if self.text_gateway is None or self.multimodal_gateway is None:
            raise RuntimeError("building both query channels requires both gateways")
        ordered_ids = list(query_ids)
        text_result, multimodal_result = await asyncio.gather(
            self.build_channel(
                manifest,
                ordered_ids,
                channel=QBankRuntimeChannel.TEXT,
                on_progress=on_progress,
            ),
            self.build_channel(
                manifest,
                ordered_ids,
                channel=QBankRuntimeChannel.MULTIMODAL,
                on_progress=on_progress,
            ),
        )
        return QueryEmbeddingBuildResult(
            query_executor_run_id=manifest.query_executor_run_id,
            text_collection_id=text_result.collection_id,
            multimodal_collection_id=multimodal_result.collection_id,
            query_count=len(ordered_ids),
            model_call_count=(text_result.model_call_count + multimodal_result.model_call_count),
            reused_batch_count=(
                text_result.reused_batch_count + multimodal_result.reused_batch_count
            ),
        )

    async def build_channel(
        self,
        manifest: QueryExecutorRunManifest,
        query_ids: Sequence[str],
        *,
        channel: QBankRuntimeChannel,
        on_progress: EmbeddingProgress | None = None,
    ) -> QueryEmbeddingChannelResult:
        """Build one channel so managed text and MM model stages stay disjoint."""

        if channel is QBankRuntimeChannel.LEXICAL:
            raise ValueError("lexical matching does not have a query embedding stage")
        ordered_ids = list(query_ids)
        if len(ordered_ids) != manifest.selected_query_count:
            raise ValueError("query embedding denominator differs from the run manifest")
        loaded = await self.queries.list_ids(manifest.snapshot_id, ordered_ids)
        by_id = {query.query_id: query for query in loaded}
        if set(by_id) != set(ordered_ids):
            raise ValueError("query embedding denominator contains missing runtime queries")
        queries = [by_id[query_id] for query_id in ordered_ids]
        if any(query.dataset is not manifest.dataset for query in queries):
            raise ValueError("query embedding denominator crossed a dataset boundary")
        channel_queries = [
            query
            for query in queries
            if channel
            in required_query_embedding_channels(
                query,
                include_text_only_multimodal=self.include_text_only_multimodal,
            )
        ]

        multimodal = channel is QBankRuntimeChannel.MULTIMODAL
        alias = self.multimodal_model_alias if multimodal else self.text_model_alias
        gateway = self.multimodal_gateway if multimodal else self.text_gateway
        if gateway is None:
            raise RuntimeError(f"{channel.value} query embedding gateway is unavailable")
        entry = self._entry(alias, require_multimodal=multimodal)
        collection = self._collection(manifest, channel, entry)
        await self._ensure_collection(collection)
        model_calls, reused = await self._build_channel(
            manifest=manifest,
            queries=channel_queries,
            channel=channel,
            collection=collection,
            entry=entry,
            gateway=gateway,
            batch_size=(self.multimodal_batch_size if multimodal else self.text_batch_size),
            include_images=multimodal,
            on_progress=on_progress,
        )
        channel_query_ids = [query.query_id for query in channel_queries]
        await self._mark_ready(collection, channel_query_ids)
        return QueryEmbeddingChannelResult(
            query_executor_run_id=manifest.query_executor_run_id,
            channel=channel,
            collection_id=collection.collection_id,
            query_count=len(channel_queries),
            model_call_count=model_calls,
            reused_batch_count=reused,
        )

    def _entry(
        self,
        alias: str,
        *,
        require_multimodal: bool,
    ) -> ModelRegistryEntry:
        entry = self.registry.get(alias)
        if entry.embedding_dimension != 4096:
            raise ValueError("Step 7 query embedding models must produce 4096 dimensions")
        if entry.distance_metric is not DistanceMetric.COSINE:
            raise ValueError("Step 7 query embeddings require cosine distance")
        if entry.normalize_embeddings is not True:
            raise ValueError("Step 7 query embeddings must be normalized")
        capability_values = {item.value for item in entry.capabilities}
        required = "embed_multimodal" if require_multimodal else "embed_text"
        if required not in capability_values:
            raise ValueError(f"query embedding alias lacks {required}")
        return entry

    @staticmethod
    def _collection(
        manifest: QueryEmbeddingOwner,
        channel: QBankRuntimeChannel,
        entry: ModelRegistryEntry,
    ) -> EmbeddingCollection:
        preprocessing_hash = canonical_content_hash(
            {
                "version": QUERY_EMBEDDING_PREPROCESSING_VERSION,
                "instruction_version": QUERY_MATCH_INSTRUCTION_VERSION,
                "channel": channel.value,
                "model_entry_hash": entry.resolved_hash,
            }
        )
        collection_id = stable_compiler_id(
            "query-embedding-collection",
            [
                manifest.query_executor_run_id,
                manifest.snapshot_id,
                channel.value,
                entry.alias,
                entry.model_revision,
                preprocessing_hash,
            ],
        )
        return EmbeddingCollection(
            collection_id=collection_id,
            snapshot_id=manifest.snapshot_id,
            run_id=manifest.query_executor_run_id,
            source_type=EmbeddingSourceType.QUERY,
            provider=entry.endpoint_kind.value,
            model_alias=entry.alias,
            actual_model_id=entry.expected_actual_model_id,
            model_revision=entry.model_revision,
            dimension=4096,
            distance_metric=DistanceMetric.COSINE,
            normalized=True,
            normalization_method="l2-v1",
            preprocessing_hash=preprocessing_hash,
            metadata={
                "channel": channel.value,
                "instruction_version": QUERY_MATCH_INSTRUCTION_VERSION,
                "query_executor_schema_version": manifest.schema_version,
            },
        )

    async def _ensure_collection(self, expected: EmbeddingCollection) -> None:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.embeddings is not None
            current = await uow.embeddings.get_collection(expected.collection_id)
            if current is None:
                await uow.embeddings.add_collection(expected)
                return
            ignored = {"status", "index_type", "index_params", "created_at"}
            if current.model_dump(exclude=ignored) != expected.model_dump(exclude=ignored):
                raise ValueError("query embedding collection identity conflict")
            if current.status is EmbeddingCollectionStatus.FAILED:
                raise RuntimeError("failed query embedding collection is immutable")

    async def _build_channel(
        self,
        *,
        manifest: QueryExecutorRunManifest,
        queries: Sequence[QueryInput],
        channel: QBankRuntimeChannel,
        collection: EmbeddingCollection,
        entry: ModelRegistryEntry,
        gateway: EmbeddingGateway,
        batch_size: int,
        include_images: bool,
        on_progress: EmbeddingProgress | None,
    ) -> tuple[int, int]:
        batches = [
            list(queries[start : start + batch_size])
            for start in range(0, len(queries), batch_size)
        ]
        semaphore = asyncio.Semaphore(self.concurrency)
        progress_lock = asyncio.Lock()
        completed = 0
        model_calls = 0
        reused = 0

        async def execute_batch(ordinal: int, values: Sequence[QueryInput]) -> None:
            nonlocal completed, model_calls, reused
            async with semaphore:
                called, was_reused = await self._build_batch(
                    manifest=manifest,
                    queries=values,
                    channel=channel,
                    collection=collection,
                    entry=entry,
                    gateway=gateway,
                    batch_ordinal=ordinal,
                    include_images=include_images,
                )
            async with progress_lock:
                completed += len(values)
                model_calls += int(called)
                reused += int(was_reused)
                if on_progress is not None:
                    on_progress(channel.value, completed, len(queries), was_reused)

        tasks = [
            asyncio.create_task(execute_batch(ordinal, values))
            for ordinal, values in enumerate(batches)
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return model_calls, reused

    async def _build_batch(
        self,
        *,
        manifest: QueryExecutorRunManifest,
        queries: Sequence[QueryInput],
        channel: QBankRuntimeChannel,
        collection: EmbeddingCollection,
        entry: ModelRegistryEntry,
        gateway: EmbeddingGateway,
        batch_ordinal: int,
        include_images: bool,
    ) -> tuple[bool, bool]:
        query_ids = [query.query_id for query in queries]
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.query_executor is not None
            existing = await uow.query_executor.list_query_embedding_facts(
                run_id=manifest.query_executor_run_id,
                channel=channel.value,
                query_ids=query_ids,
            )
        if len(existing) == len(query_ids):
            return False, True
        if existing:
            raise RuntimeError("query embedding batch is partially persisted")

        batch_id = stable_compiler_id(
            "query-embedding-batch",
            [
                manifest.query_executor_run_id,
                channel.value,
                batch_ordinal,
                query_ids,
                entry.resolved_hash,
                QUERY_MATCH_INSTRUCTION_VERSION,
            ],
        )
        request = EmbeddingRequest(
            request_id=batch_id,
            model_alias=entry.alias,
            inputs=[
                EmbeddingInput(
                    input_id=query.query_id,
                    text=QUERY_MATCH_INSTRUCTION + query.raw_query,
                    image_refs=(list(query.query_image_refs) if include_images else []),
                    metadata={
                        "task": "runtime_qbank_match",
                        "channel": channel.value,
                        "instruction_version": QUERY_MATCH_INSTRUCTION_VERSION,
                    },
                )
                for query in queries
            ],
            dimensions=4096,
            metadata={
                "scope": "runtime",
                "channel": channel.value,
                "batch_id": batch_id,
                "persist_normalized_embedding_result": True,
            },
        )
        result, model_call_id, recovered = await self._load_or_call(
            manifest=manifest,
            request=request,
            gateway=gateway,
        )
        if result.request_id != batch_id:
            raise ValueError("query embedding response has a different request ID")
        vectors = {item.input_id: normalize_vector(item.vector) for item in result.vectors}
        if set(vectors) != set(query_ids):
            raise ValueError("query embedding response does not match its batch")
        if any(len(vector) != 4096 for vector in vectors.values()):
            raise ValueError("query embedding response has a non-4096 vector")

        records: list[EmbeddingRecord] = []
        facts: list[QueryEmbeddingFact] = []
        for query in queries:
            input_hash = canonical_content_hash(
                {
                    "query_id": query.query_id,
                    "query_text": query.raw_query,
                    "query_image_hashes": [
                        item.sha256 or item.image_id
                        for item in (query.query_image_refs if include_images else [])
                    ],
                    "channel": channel.value,
                    "instruction_version": QUERY_MATCH_INSTRUCTION_VERSION,
                    "model_entry_hash": entry.resolved_hash,
                }
            )
            embedding_id = stable_compiler_id(
                "runtime-query-embedding",
                [manifest.query_executor_run_id, query.query_id, channel.value, input_hash],
            )
            records.append(
                EmbeddingRecord(
                    embedding_id=embedding_id,
                    collection_id=collection.collection_id,
                    source_id=query.query_id,
                    input_hash=input_hash,
                    dimension=4096,
                    vector=vectors[query.query_id],
                    model_call_id=model_call_id,
                    metadata={
                        "query_id": query.query_id,
                        "channel": channel.value,
                        "batch_id": batch_id,
                    },
                )
            )
            fact_payload = {
                "run_id": manifest.query_executor_run_id,
                "query_id": query.query_id,
                "channel": channel.value,
                "embedding_id": embedding_id,
                "collection_id": collection.collection_id,
                "input_hash": input_hash,
            }
            facts.append(
                QueryEmbeddingFact(
                    query_embedding_id=stable_compiler_id(
                        "runtime-query-embedding-fact", fact_payload
                    ),
                    query_embedding_batch_id=batch_id,
                    query_executor_run_id=manifest.query_executor_run_id,
                    snapshot_id=manifest.snapshot_id,
                    query_id=query.query_id,
                    channel=channel,
                    embedding_id=embedding_id,
                    collection_id=collection.collection_id,
                    vector_dimension=4096,
                    input_hash=input_hash,
                    content_hash=canonical_content_hash(fact_payload),
                    metadata={"batch_ordinal": batch_ordinal},
                )
            )
        batch_payload = {
            "batch_id": batch_id,
            "run_id": manifest.query_executor_run_id,
            "snapshot_id": manifest.snapshot_id,
            "model_call_id": model_call_id,
            "channel": channel.value,
            "batch_ordinal": batch_ordinal,
            "query_ids": query_ids,
            "fact_hashes": [fact.content_hash for fact in facts],
        }
        batch = QueryEmbeddingBatch(
            query_embedding_batch_id=batch_id,
            query_executor_run_id=manifest.query_executor_run_id,
            snapshot_id=manifest.snapshot_id,
            model_call_id=model_call_id,
            channel=channel,
            batch_ordinal=batch_ordinal,
            query_ids=query_ids,
            expected_count=len(query_ids),
            committed_count=len(query_ids),
            content_hash=canonical_content_hash(batch_payload),
            metadata={
                "instruction_version": QUERY_MATCH_INSTRUCTION_VERSION,
                "recovered_model_call": recovered,
            },
        )
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.query_executor is not None
            assert uow.embeddings is not None
            repeated = await uow.query_executor.list_query_embedding_facts(
                run_id=manifest.query_executor_run_id,
                channel=channel.value,
                query_ids=query_ids,
            )
            if repeated:
                if len(repeated) != len(query_ids):
                    raise RuntimeError("query embedding batch became partially persisted")
                return False, True
            await uow.query_executor.add_embedding_batch(batch)
            await uow.embeddings.add_records(records)
            for fact in facts:
                await uow.query_executor.add_embedding(fact)
        return not recovered, recovered

    async def _load_or_call(
        self,
        *,
        manifest: QueryEmbeddingOwner,
        request: EmbeddingRequest,
        gateway: EmbeddingGateway,
    ) -> tuple[EmbeddingResult, str, bool]:
        available_call_id: str | None = None
        for attempt in range(1, self.max_call_attempts + 1):
            model_call_id = stable_compiler_id(
                "runtime-query-embedding-call",
                [request.request_id, f"attempt={attempt}"],
            )
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.model_calls is not None
                assert uow.artifacts is not None
                trace = await uow.model_calls.get(model_call_id)
                artifact = (
                    await uow.artifacts.get(trace.output_asset_id)
                    if trace is not None and trace.output_asset_id is not None
                    else None
                )
            if trace is None:
                available_call_id = model_call_id
                break
            if trace.status is TraceStatus.FAILED:
                continue
            if trace.status is not TraceStatus.SUCCESS or artifact is None:
                raise RuntimeError("query embedding model call has incomplete lineage")
            remote = await self.execution.artifact_service.store.head(artifact.object_key)
            if (
                remote is None
                or artifact.size_bytes is None
                or remote.sha256 != artifact.sha256
                or remote.size_bytes != artifact.size_bytes
            ):
                raise RuntimeError("query embedding response artifact failed verification")
            payload = await self.execution.artifact_service.store.get_bytes(artifact.object_key)
            result = embedding_result_from_artifact(payload)
            self.registry.verify_actual_model(
                request.model_alias,
                result.actual_model_id,
                paper_mode=self.execution.paper_mode,
            )
            return result, trace.model_call_id, True
        if available_call_id is None:
            raise RuntimeError("query embedding batch exhausted its call attempts")
        audited = await self.execution.embed_audited(
            gateway,
            request,
            ModelCallContext(
                model_call_id=available_call_id,
                run_id=manifest.query_executor_run_id,
                stage=AppStage.EMBED,
                snapshot_id=manifest.snapshot_id,
            ),
        )
        return audited.result, audited.trace.model_call_id, False

    async def _mark_ready(
        self,
        collection: EmbeddingCollection,
        query_ids: Sequence[str],
    ) -> None:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.query_executor is not None
            assert uow.embeddings is not None
            facts = await uow.query_executor.list_query_embedding_facts(
                run_id=collection.run_id or "",
                channel=str(collection.metadata["channel"]),
                query_ids=query_ids,
            )
            if len(facts) != len(query_ids):
                raise RuntimeError("query embedding collection denominator is incomplete")
            await uow.embeddings.mark_collection_ready(
                collection.collection_id,
                index_type="exact_scan",
                index_params={
                    "dimension": 4096,
                    "distance_metric": "cosine",
                    "query_count": len(query_ids),
                },
            )


class CoverageQueryEmbeddingImporter:
    """Project already-audited, gold-free coverage vectors into runtime scope."""

    def __init__(
        self,
        *,
        runtime_sessions: async_sessionmaker[AsyncSession],
        evaluation_sessions: async_sessionmaker[AsyncSession],
        registry: ModelRegistry,
        include_text_only_multimodal: bool = False,
    ) -> None:
        self.runtime_sessions = runtime_sessions
        self.evaluation_sessions = evaluation_sessions
        self.registry = registry
        self.include_text_only_multimodal = include_text_only_multimodal

    async def import_channel(
        self,
        manifest: QueryExecutorRunManifest,
        *,
        coverage_run_id: str,
        query_ids: Sequence[str],
        channel: QBankRuntimeChannel,
    ) -> QueryEmbeddingChannelResult:
        if channel is QBankRuntimeChannel.LEXICAL:
            raise ValueError("lexical matching has no imported query vector")
        coverage_channel = (
            QBankMatchChannel.MULTIMODAL_SEMANTIC
            if channel is QBankRuntimeChannel.MULTIMODAL
            else QBankMatchChannel.TEXT_SEMANTIC
        )
        alias = (
            "qwen3-vl-embedding-8b"
            if channel is QBankRuntimeChannel.MULTIMODAL
            else "qwen3-embedding-8b"
        )
        entry = self.registry.get(alias)
        if (
            entry.embedding_dimension != 4096
            or entry.distance_metric is not DistanceMetric.COSINE
            or entry.normalize_embeddings is not True
        ):
            raise ValueError("coverage query vectors have an incompatible model contract")

        async with ExperimentUnitOfWork(self.evaluation_sessions, DataScope.EVALUATION) as uow:
            assert uow.qbank_coverage is not None
            coverage = await uow.qbank_coverage.get_run(coverage_run_id)
            values = await uow.qbank_coverage.list_query_embeddings(coverage_run_id)
        if coverage is None:
            raise KeyError("coverage run for query-vector projection does not exist")
        if (
            coverage.compiler_run_id != manifest.source_compiler_run_id
            or coverage.snapshot_id != manifest.snapshot_id
            or coverage.dataset != manifest.dataset.value
        ):
            raise ValueError("coverage vectors crossed a compiler or dataset boundary")
        ordered_ids = list(query_ids)
        runtime_queries = await PostgresRuntimeQueryRepository(self.runtime_sessions).list_ids(
            manifest.snapshot_id, ordered_ids
        )
        runtime_by_id = {query.query_id: query for query in runtime_queries}
        if set(runtime_by_id) != set(ordered_ids):
            raise ValueError("runtime query denominator differs from the query run")
        required_ids = [
            query_id
            for query_id in ordered_ids
            if channel
            in required_query_embedding_channels(
                runtime_by_id[query_id],
                include_text_only_multimodal=self.include_text_only_multimodal,
            )
        ]
        selected = [item for item in values if item.channel is coverage_channel]
        by_query = {item.query_id: item for item in selected}
        if (
            len(ordered_ids) != manifest.selected_query_count
            or len(set(ordered_ids)) != len(ordered_ids)
            or set(by_query) != set(required_ids)
        ):
            raise ValueError("coverage vector denominator differs from the query run")
        for value in selected:
            if (
                value.model_alias != entry.alias
                or value.model_revision != entry.model_revision
                or value.dimension != 4096
                or value.model_call_id is None
                or value.metadata.get("instruction_version") != QUERY_MATCH_INSTRUCTION_VERSION
            ):
                raise ValueError("coverage vector identity is not runtime-compatible")

        collection = QueryEmbeddingWorkflow._collection(manifest, channel, entry)
        await self._ensure_collection(collection)
        groups: dict[str, list[str]] = {}
        for query_id in required_ids:
            source = by_query[query_id]
            batch_id = source.metadata.get("batch_id")
            if not isinstance(batch_id, str) or not batch_id:
                raise ValueError("coverage vector has no audited batch identity")
            groups.setdefault(batch_id, []).append(query_id)

        reused = 0
        for batch_ordinal, (source_batch_id, batch_query_ids) in enumerate(groups.items()):
            was_reused = await self._import_batch(
                manifest=manifest,
                coverage_run_id=coverage_run_id,
                source_batch_id=source_batch_id,
                batch_ordinal=batch_ordinal,
                query_ids=batch_query_ids,
                source_by_query=by_query,
                channel=channel,
                collection=collection,
            )
            reused += int(was_reused)
        await self._mark_ready(collection, required_ids, channel)
        print(
            "STEP7_QUERY_EMBEDDING_IMPORT "
            f"channel={channel.value} state=ready "
            f"required_queries={len(required_ids)} total_queries={len(ordered_ids)} "
            f"source_batches={len(groups)} reused_batches={reused}",
            flush=True,
        )
        return QueryEmbeddingChannelResult(
            query_executor_run_id=manifest.query_executor_run_id,
            channel=channel,
            collection_id=collection.collection_id,
            query_count=len(required_ids),
            model_call_count=0,
            reused_batch_count=reused,
        )

    async def _ensure_collection(self, expected: EmbeddingCollection) -> None:
        async with ExperimentUnitOfWork(self.runtime_sessions, DataScope.RUNTIME) as uow:
            assert uow.embeddings is not None
            current = await uow.embeddings.get_collection(expected.collection_id)
            if current is None:
                await uow.embeddings.add_collection(expected)
                return
            ignored = {"status", "index_type", "index_params", "created_at"}
            if current.model_dump(exclude=ignored) != expected.model_dump(exclude=ignored):
                raise ValueError("imported query collection identity conflict")

    async def _import_batch(
        self,
        *,
        manifest: QueryExecutorRunManifest,
        coverage_run_id: str,
        source_batch_id: str,
        batch_ordinal: int,
        query_ids: Sequence[str],
        source_by_query: dict[str, QBankCoverageQueryEmbedding],
        channel: QBankRuntimeChannel,
        collection: EmbeddingCollection,
    ) -> bool:
        sources = [source_by_query[query_id] for query_id in query_ids]
        async with ExperimentUnitOfWork(self.runtime_sessions, DataScope.RUNTIME) as uow:
            assert uow.query_executor is not None
            existing = await uow.query_executor.list_query_embedding_facts(
                run_id=manifest.query_executor_run_id,
                channel=channel.value,
                query_ids=query_ids,
            )
        if len(existing) == len(query_ids):
            return True
        if existing:
            raise RuntimeError("imported query vector batch is partially persisted")

        batch_id = stable_compiler_id(
            "query-embedding-import-batch",
            [
                manifest.query_executor_run_id,
                coverage_run_id,
                source_batch_id,
                channel.value,
                list(query_ids),
            ],
        )
        records: list[EmbeddingRecord] = []
        facts: list[QueryEmbeddingFact] = []
        for source in sources:
            embedding_id = stable_compiler_id(
                "runtime-query-embedding",
                [manifest.query_executor_run_id, source.query_id, channel.value, source.input_hash],
            )
            records.append(
                EmbeddingRecord(
                    embedding_id=embedding_id,
                    collection_id=collection.collection_id,
                    source_id=source.query_id,
                    input_hash=source.input_hash,
                    dimension=source.dimension,
                    vector=source.vector,
                    model_call_id=None,
                    metadata={
                        "coverage_run_id": coverage_run_id,
                        "source_evaluation_model_call_id": source.model_call_id,
                        "source_query_embedding_id": source.query_embedding_id,
                        "source_batch_id": source_batch_id,
                    },
                )
            )
            payload = {
                "run_id": manifest.query_executor_run_id,
                "query_id": source.query_id,
                "channel": channel.value,
                "embedding_id": embedding_id,
                "collection_id": collection.collection_id,
                "input_hash": source.input_hash,
            }
            facts.append(
                QueryEmbeddingFact(
                    query_embedding_id=stable_compiler_id("runtime-query-embedding-fact", payload),
                    query_embedding_batch_id=batch_id,
                    query_executor_run_id=manifest.query_executor_run_id,
                    snapshot_id=manifest.snapshot_id,
                    query_id=source.query_id,
                    channel=channel,
                    embedding_id=embedding_id,
                    collection_id=collection.collection_id,
                    vector_dimension=source.dimension,
                    input_hash=source.input_hash,
                    content_hash=canonical_content_hash(payload),
                    metadata={"coverage_projection": True},
                )
            )
        model_call_ids = {source.model_call_id for source in sources}
        if len(model_call_ids) != 1 or None in model_call_ids:
            raise ValueError("coverage batch must map to exactly one model call")
        model_call_id = next(iter(model_call_ids))
        assert model_call_id is not None
        batch_payload = {
            "batch_id": batch_id,
            "run_id": manifest.query_executor_run_id,
            "coverage_run_id": coverage_run_id,
            "source_batch_id": source_batch_id,
            "model_call_id": model_call_id,
            "channel": channel.value,
            "batch_ordinal": batch_ordinal,
            "query_ids": list(query_ids),
            "fact_hashes": [fact.content_hash for fact in facts],
        }
        batch = QueryEmbeddingBatch(
            query_embedding_batch_id=batch_id,
            query_executor_run_id=manifest.query_executor_run_id,
            snapshot_id=manifest.snapshot_id,
            model_call_id=None,
            channel=channel,
            batch_ordinal=batch_ordinal,
            query_ids=list(query_ids),
            expected_count=len(query_ids),
            committed_count=len(query_ids),
            content_hash=canonical_content_hash(batch_payload),
            metadata={
                "coverage_projection": True,
                "coverage_run_id": coverage_run_id,
                "source_batch_id": source_batch_id,
                "source_evaluation_model_call_id": model_call_id,
            },
        )
        async with ExperimentUnitOfWork(self.runtime_sessions, DataScope.RUNTIME) as uow:
            assert uow.query_executor is not None
            assert uow.embeddings is not None
            repeated = await uow.query_executor.list_query_embedding_facts(
                run_id=manifest.query_executor_run_id,
                channel=channel.value,
                query_ids=query_ids,
            )
            if repeated:
                if len(repeated) != len(query_ids):
                    raise RuntimeError("imported query vector batch became partial")
                return True
            await uow.query_executor.add_embedding_batch(batch)
            await uow.embeddings.add_records(records)
            for fact in facts:
                await uow.query_executor.add_embedding(fact)
        return False

    async def _mark_ready(
        self,
        collection: EmbeddingCollection,
        query_ids: Sequence[str],
        channel: QBankRuntimeChannel,
    ) -> None:
        async with ExperimentUnitOfWork(self.runtime_sessions, DataScope.RUNTIME) as uow:
            assert uow.query_executor is not None
            assert uow.embeddings is not None
            facts = await uow.query_executor.list_query_embedding_facts(
                run_id=collection.run_id or "",
                channel=channel.value,
                query_ids=query_ids,
            )
            if len(facts) != len(query_ids):
                raise RuntimeError("imported query collection denominator is incomplete")
            await uow.embeddings.mark_collection_ready(
                collection.collection_id,
                index_type="exact_scan",
                index_params={
                    "dimension": 4096,
                    "distance_metric": "cosine",
                    "query_count": len(query_ids),
                    "coverage_projection": True,
                },
            )


__all__ = [
    "CoverageQueryEmbeddingImporter",
    "QUERY_EMBEDDING_PREPROCESSING_VERSION",
    "QUERY_MATCH_INSTRUCTION",
    "QUERY_MATCH_INSTRUCTION_VERSION",
    "QueryEmbeddingBuildResult",
    "QueryEmbeddingChannelResult",
    "QueryEmbeddingWorkflow",
    "required_query_embedding_channels",
]
