"""Snapshot-level, recoverable embedding index construction for RAG baselines."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.baselines.leases import baseline_worker_id
from pmmc.baselines.rag_config import RagBaselineConfig, query_instruction_sha256
from pmmc.baselines.rag_embedding import (
    RAG_METHODS,
    embedding_record_id,
    make_document_collection,
    produce_document_embeddings,
)
from pmmc.benchmarks.ids import stable_hash
from pmmc.experiments import make_run_id, run_config_hash
from pmmc.infrastructure.database.baseline_repositories import PostgresCorpusRepository
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.repositories import PostgresImageRepository
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.infrastructure.object_store import create_runtime_object_store
from pmmc.infrastructure.settings import AppMode, Settings
from pmmc.models.execution import ModelExecutionService
from pmmc.models.factory import (
    create_embedding_gateway,
    create_image_url_resolver,
    create_model_execution_service,
)
from pmmc.models.protocols import EmbeddingGateway
from pmmc.models.registry import ModelRegistry, bind_registry_to_manifest
from pmmc.schemas import (
    AppStage,
    AssetType,
    BaselineCorpusManifest,
    BaselineMethod,
    DataScope,
    EmbeddingCollection,
    EmbeddingCollectionStatus,
    ExperimentJob,
    ImageRef,
    JobStatus,
    RunManifest,
    RunStatus,
)

IndexProgress = Callable[[int, int], None]


@dataclass(frozen=True)
class RagIndexBuildResult:
    collection: EmbeddingCollection
    run_id: str
    document_count: int
    reused: bool


class _DetachedImageReader:
    """Materialize image metadata in a short read session before provider I/O."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def get_many(self, snapshot_id: str, image_ids: list[str]) -> list[ImageRef]:
        return await PostgresImageRepository(self.sessions).get_many(snapshot_id, image_ids)


def _index_job_id(collection_id: str, ordinal: int) -> str:
    return "job-index-" + stable_hash([collection_id, ordinal])[:24]


def _index_manifest(
    *,
    run_id: str,
    corpus: BaselineCorpusManifest,
    method: BaselineMethod,
    config: RagBaselineConfig,
    registry: ModelRegistry,
) -> RunManifest:
    alias = config.embedding_alias_for(method)
    selection_fingerprint = stable_hash(
        {
            "kind": "snapshot_embedding_index",
            "corpus_id": corpus.corpus_id,
            "method": method.value,
        }
    )
    manifest = RunManifest(
        run_id=run_id,
        experiment_name="baseline_index_build",
        dataset=corpus.dataset,
        method=f"{method.value}_index",
        paper_mode=False,
        embedding_model_aliases=[alias],
        selection_hash=selection_fingerprint,
        seed=0,
        retrieval_topk_max=config.retrieval_top_k,
        prompt_hashes={
            config.query_instruction_version: query_instruction_sha256(config)
        },
        retrieval_config={
            "distance_metric": config.distance_metric.value,
            "exact_scan": config.exact_scan,
            "reranker": config.use_reranker,
            "corpus_id": corpus.corpus_id,
        },
        dataset_snapshot_ids=[corpus.snapshot_id],
        metadata={
            "rag_config": config.model_dump(mode="json"),
            "protocol_version": config.protocol_version,
            "corpus_content_hash": corpus.content_hash,
            "index_build": True,
        },
    )
    bound = bind_registry_to_manifest(manifest, registry, aliases=[alias])
    return bound.model_copy(update={"config_hash": run_config_hash(bound)})


class RagIndexBuilder:
    """Claim stable document batches and freeze a complete exact-scan collection."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        execution: ModelExecutionService,
        gateway: EmbeddingGateway,
        error_artifacts: ExperimentArtifactService,
        collection: EmbeddingCollection,
        corpus_id: str,
        method: BaselineMethod,
        config: RagBaselineConfig,
        document_ids: list[str],
        lease_seconds: int,
    ) -> None:
        if method not in RAG_METHODS:
            raise ValueError("index builder requires a RAG method")
        if lease_seconds < 3:
            raise ValueError("index job lease must be at least three seconds")
        self.sessions = sessions
        self.execution = execution
        self.gateway = gateway
        self.error_artifacts = error_artifacts
        self.collection = collection
        self.corpus_id = corpus_id
        self.method = method
        self.config = config
        self.document_ids = document_ids
        self.lease_seconds = lease_seconds
        self.batch_size = config.embedding_batch_size_for(method)
        self.batches = [
            document_ids[start : start + self.batch_size]
            for start in range(0, len(document_ids), self.batch_size)
        ]
        self.batch_by_job = {
            _index_job_id(collection.collection_id, ordinal): (ordinal, document_ids)
            for ordinal, document_ids in enumerate(self.batches)
        }
        self.images = _DetachedImageReader(sessions)

    @staticmethod
    def _immutable_job(value: ExperimentJob) -> dict[str, object]:
        return value.model_dump(
            mode="json",
            exclude={
                "status",
                "attempt_count",
                "claimed_by",
                "lease_expires_at",
                "available_at",
                "started_at",
                "finished_at",
                "error_type",
                "error_summary",
                "error_artifact_id",
            },
        )

    async def execute(self, progress: IndexProgress | None = None) -> None:
        run_id = self.collection.run_id
        if run_id is None:
            raise ValueError("document collection requires an index-build run")
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None and uow.runs is not None
            for job_id, (ordinal, document_ids) in self.batch_by_job.items():
                expected = ExperimentJob(
                    job_id=job_id,
                    run_id=run_id,
                    snapshot_id=self.collection.snapshot_id,
                    stage=AppStage.EMBED,
                    variant=f"batch-{ordinal:08d}",
                    max_attempts=self.config.job_max_attempts,
                    metadata={
                        "collection_id": self.collection.collection_id,
                        "batch_ordinal": ordinal,
                        "document_ids": document_ids,
                    },
                )
                stored = await uow.jobs.get(job_id)
                if stored is None:
                    await uow.jobs.add(expected)
                elif self._immutable_job(stored) != self._immutable_job(expected):
                    raise ValueError(f"persisted index job conflicts with collection: {job_id}")
            state = await uow.runs.get_state(run_id)
            if state.status is RunStatus.CREATED:
                await uow.runs.transition(run_id, RunStatus.RUNNING)
            elif state.status not in {RunStatus.RUNNING, RunStatus.COMPLETED}:
                raise RuntimeError(f"index run cannot resume from {state.status.value}")

        async def worker(index: int) -> None:
            worker_id = baseline_worker_id(run_id, "rag-index", index)
            while True:
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    assert uow.jobs is not None
                    job = await uow.jobs.claim(
                        worker_id,
                        self.lease_seconds,
                        run_id=run_id,
                        stage=AppStage.EMBED,
                    )
                    counts = (
                        await uow.jobs.status_counts(run_id, stage=AppStage.EMBED)
                        if job is None
                        else None
                    )
                if job is None:
                    assert counts is not None
                    if counts.get(JobStatus.PENDING, 0) + counts.get(JobStatus.RUNNING, 0):
                        await asyncio.sleep(0.1)
                        continue
                    return
                try:
                    await self._process(job, worker_id)
                except Exception as exc:
                    terminal = await self._finish_failed(job, worker_id, exc)
                    if terminal.status is JobStatus.FAILED and progress:
                        progress(0, len(self.document_ids))
                else:
                    if progress:
                        progress(len(self.batch_by_job[job.job_id][1]), len(self.document_ids))

        await asyncio.gather(
            *(
                worker(index)
                for index in range(
                    min(self.config.embedding_concurrency, max(1, len(self.batches)))
                )
            )
        )
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None and uow.embeddings is not None and uow.runs is not None
            counts = await uow.jobs.status_counts(run_id, stage=AppStage.EMBED)
            failed = counts.get(JobStatus.FAILED, 0)
            if failed:
                await uow.runs.transition(
                    run_id,
                    RunStatus.FAILED,
                    error_type="EmbeddingIndexBuildFailure",
                    error_summary="one or more index batches exhausted their attempts",
                )
                raise RuntimeError(f"{failed} embedding index batch(es) failed")
            stored_count = await uow.embeddings.count_records(self.collection.collection_id)
            if stored_count != len(self.document_ids):
                raise RuntimeError("embedding collection record count is incomplete")
            await uow.embeddings.mark_collection_ready(
                self.collection.collection_id,
                index_type="exact",
                index_params={"ann": False, "distance_metric": "cosine"},
            )
            state = await uow.runs.get_state(run_id)
            if state.status is RunStatus.RUNNING:
                await uow.runs.transition(run_id, RunStatus.COMPLETED)

    async def _process(self, job: ExperimentJob, worker_id: str) -> None:
        ordinal, document_ids = self.batch_by_job[job.job_id]
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(job, worker_id, stop))
        try:
            async with self.sessions() as session:
                documents = await PostgresCorpusRepository(session).get_documents(
                    self.corpus_id, document_ids
                )
            expected_ids = [
                embedding_record_id(self.collection.collection_id, document.document_id)
                for document in documents
            ]
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.embeddings is not None
                stored = [
                    await uow.embeddings.get_record(embedding_id)
                    for embedding_id in expected_ids
                ]
            existing_count = sum(item is not None for item in stored)
            if existing_count == len(expected_ids):
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    assert uow.jobs is not None
                    await uow.jobs.finish(
                        job.job_id, worker_id, success=True,
                        expected_attempt_count=job.attempt_count,
                    )
                return
            if existing_count:
                raise RuntimeError("embedding batch is partially persisted")

            produced = await produce_document_embeddings(
                documents=documents,
                method=self.method,
                collection=self.collection,
                images=self.images,
                gateway=self.gateway,
                execution=self.execution,
                run_id=job.run_id,
                batch_ordinal=ordinal,
                attempt_number=job.attempt_count,
                job_id=job.job_id,
            )
            if (
                self.collection.actual_model_id is not None
                and produced.actual_model_id != self.collection.actual_model_id
            ):
                raise ValueError("embedding provider identity differs from collection identity")
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.embeddings is not None and uow.jobs is not None
                await uow.jobs.finish(
                    job.job_id, worker_id, success=True,
                    expected_attempt_count=job.attempt_count,
                )
                await uow.embeddings.add_records(produced.records)
        finally:
            stop.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

    async def _heartbeat(self, job: ExperimentJob, worker_id: str, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.lease_seconds / 3)
            except TimeoutError:
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    assert uow.jobs is not None
                    await uow.jobs.heartbeat(
                        job.job_id, worker_id, self.lease_seconds,
                        expected_attempt_count=job.attempt_count,
                    )

    async def _finish_failed(
        self, job: ExperimentJob, worker_id: str, error: BaseException
    ) -> ExperimentJob:
        payload = json.dumps(
            {"error_type": type(error).__name__, "message": "RAG index batch failed"},
            sort_keys=True,
        ).encode("utf-8")
        artifact_id: str | None = None
        try:
            artifact = await self.error_artifacts.put_bytes(
                artifact_id=f"{job.job_id}:attempt:{job.attempt_count}:error",
                run_id=job.run_id,
                snapshot_id=job.snapshot_id,
                artifact_type=AssetType.OTHER,
                data=payload,
                mime_type="application/json",
                extension="json",
            )
            artifact_id = artifact.artifact_id
        except Exception:
            artifact_id = None
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None
            return await uow.jobs.finish(
                job.job_id,
                worker_id,
                success=False,
                error_type=type(error).__name__,
                error_summary="RAG index batch failed; inspect runtime artifact",
                error_artifact_id=artifact_id,
                retry_delay_seconds=self.config.job_retry_delay_seconds,
                expected_attempt_count=job.attempt_count,
            )


async def build_rag_index(
    *,
    settings: Settings,
    registry: ModelRegistry,
    config: RagBaselineConfig,
    corpus_id: str,
    method: BaselineMethod,
    progress: IndexProgress | None = None,
) -> RagIndexBuildResult:
    """Build or validate one snapshot-level collection and return its frozen identity."""

    if method not in RAG_METHODS:
        raise ValueError("build-index supports only the three RAG baselines")
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        candidate_run_id = make_run_id(f"index-{method.value}")
        async with sessions() as session:
            corpora = PostgresCorpusRepository(session)
            corpus = await corpora.get_manifest(corpus_id)
            document_ids = await corpora.list_document_ids(
                corpus_id, list(config.representations_for(method))
            )
        if not document_ids:
            raise ValueError("RAG corpus contains no documents for the selected method")
        if progress is not None:
            progress(0, len(document_ids))
        candidate = make_document_collection(
            corpus=corpus,
            method=method,
            config=config,
            registry=registry,
            run_id=candidate_run_id,
        )
        async with ExperimentUnitOfWork(sessions) as uow:
            assert uow.embeddings is not None
            existing = await uow.embeddings.get_collection(candidate.collection_id)
            if existing is not None and existing.status is EmbeddingCollectionStatus.READY:
                count = await uow.embeddings.count_records(existing.collection_id)
                if count != len(document_ids):
                    raise RuntimeError("ready RAG index has an invalid document count")
                if progress is not None:
                    progress(count, len(document_ids))
                return RagIndexBuildResult(
                    collection=existing,
                    run_id=existing.run_id or "snapshot-shared",
                    document_count=count,
                    reused=True,
                )
            if existing is not None and existing.status is EmbeddingCollectionStatus.FAILED:
                raise RuntimeError("failed RAG index is immutable; change its versioned recipe")

        if existing is None:
            manifest = _index_manifest(
                run_id=candidate_run_id,
                corpus=corpus,
                method=method,
                config=config,
                registry=registry,
            ).model_copy(update={"paper_mode": settings.mode is AppMode.PAPER})
            manifest = manifest.model_copy(update={"config_hash": run_config_hash(manifest)})
            async with ExperimentUnitOfWork(sessions) as uow:
                assert uow.runs is not None and uow.embeddings is not None
                await uow.runs.register(manifest, [])
                collection = await uow.embeddings.add_collection(candidate)
        else:
            collection = existing
            if collection.run_id is None:
                raise RuntimeError("building RAG index has no owning build run")

        store = create_runtime_object_store(settings)
        artifacts = ExperimentArtifactService(
            sessions,
            store,
            DataScope.RUNTIME,
            artifact_prefix=settings.experiment_artifact_prefix,
        )
        from pmmc.baselines.runner import TransactionalModelCallWriter

        execution = create_model_execution_service(
            settings,
            registry,
            artifact_service=artifacts,
            model_calls=TransactionalModelCallWriter(sessions),
        )
        gateway = create_embedding_gateway(
            settings,
            registry,
            config.embedding_alias_for(method),
            image_url_resolver=create_image_url_resolver(settings, store),
        )
        await RagIndexBuilder(
            sessions=sessions,
            execution=execution,
            gateway=gateway,
            error_artifacts=artifacts,
            collection=collection,
            corpus_id=corpus_id,
            method=method,
            config=config,
            document_ids=document_ids,
            lease_seconds=settings.experiment_job_lease_seconds,
        ).execute(progress)
        async with ExperimentUnitOfWork(sessions) as uow:
            assert uow.embeddings is not None
            ready = await uow.embeddings.get_collection(collection.collection_id)
        if ready is None or ready.status is not EmbeddingCollectionStatus.READY:
            raise RuntimeError("RAG index did not reach ready state")
        return RagIndexBuildResult(
            collection=ready,
            run_id=ready.run_id or candidate_run_id,
            document_count=len(document_ids),
            reused=False,
        )
    finally:
        await engine.dispose()
