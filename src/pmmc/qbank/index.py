"""Frozen text and native-multimodal QBank embedding index construction."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.baselines.rag_embedding import normalize_vector
from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.questioner_config import QuestionerConfig
from pmmc.infrastructure.database.errors import safe_database_error_diagnostics
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.models import ImageModel
from pmmc.infrastructure.database.repositories import PostgresImageRepository
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.models.execution import ModelCallContext, ModelExecutionService
from pmmc.models.protocols import EmbeddingGateway
from pmmc.models.registry import ModelRegistry, ModelRegistryEntry
from pmmc.schemas import (
    AppStage,
    AssetType,
    DistanceMetric,
    EmbeddingCollection,
    EmbeddingCollectionStatus,
    EmbeddingInput,
    EmbeddingRecord,
    EmbeddingRequest,
    EmbeddingSourceType,
    ExperimentJob,
    ImageRef,
    JobStatus,
    QBankIndexManifest,
    QuestionCandidate,
)

IndexProgress = Callable[[str, int, int], None]


@dataclass(frozen=True)
class QBankEmbeddingDocument:
    source_id: str
    qbank_id: str
    text: str
    image_refs: tuple[ImageRef, ...]
    source_type: EmbeddingSourceType


@dataclass(frozen=True)
class QBankIndexBuildResult:
    manifest: QBankIndexManifest
    reused: bool


class QBankIndexExecutionError(RuntimeError):
    """Aggregate failed QBank batches without exposing provider payloads."""

    def __init__(
        self,
        *,
        failed_job_count: int,
        failures: list[dict[str, object]],
    ) -> None:
        self.failed_job_count = failed_job_count

        def failure_sort_key(item: dict[str, object]) -> tuple[str, int]:
            ordinal = item.get("batch_ordinal")
            return (
                str(item.get("channel", "")),
                ordinal if isinstance(ordinal, int) else -1,
            )

        self.failures = sorted(
            failures,
            key=failure_sort_key,
        )
        channels = sorted({str(item["channel"]) for item in self.failures})
        self.safe_summary = (
            f"failed_jobs={failed_job_count} channels={','.join(channels) or 'unknown'} "
            f"diagnostic_count={len(self.failures)}"
        )
        super().__init__(self.safe_summary)

    def safe_diagnostics(self) -> dict[str, object]:
        error_type_counts: dict[str, int] = {}
        provider_status_counts: dict[str, int] = {}
        for item in self.failures:
            error_type = str(item.get("cause_type", item.get("error_type", "unknown")))
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            status = item.get("provider_status_code")
            if status is not None:
                status_key = str(status)
                provider_status_counts[status_key] = provider_status_counts.get(status_key, 0) + 1
        return {
            "batch_kind": "qbank_index",
            "diagnostic_count": len(self.failures),
            "error_type_counts": error_type_counts,
            "failed_job_count": self.failed_job_count,
            "failure_examples": self.failures[:8],
            "provider_status_counts": provider_status_counts,
        }


def _qbank_batch_failure_diagnostics(
    *,
    channel: str,
    batch_ordinal: int,
    attempt: int,
    document_count: int,
    collection_document_count: int,
    total_input_chars: int,
    image_input_count: int,
    image_reference_count: int,
    max_image_references: int,
    error: BaseException,
) -> dict[str, object]:
    """Return allow-listed diagnostics for one failed embedding batch."""

    database = safe_database_error_diagnostics(error)
    chain: list[BaseException] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 8:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    provider_error = next(
        (
            item
            for item in chain
            if hasattr(item, "provider_status_code") or hasattr(item, "retryable")
        ),
        None,
    )
    diagnostics: dict[str, object] = {
        "attempt": attempt,
        "batch_kind": "qbank_index",
        "batch_ordinal": batch_ordinal,
        "channel": channel,
        "collection_document_count": collection_document_count,
        "document_count": document_count,
        "error_type": type(error).__name__,
        "image_input_count": image_input_count,
        "image_reference_count": image_reference_count,
        "max_image_references": max_image_references,
        "operation": "qbank_embedding_batch",
        "total_input_chars": total_input_chars,
    }
    if provider_error is not None:
        diagnostics["cause_type"] = type(provider_error).__name__
        retryable = getattr(provider_error, "retryable", None)
        if isinstance(retryable, bool):
            diagnostics["retryable"] = retryable
    for key in (
        "database_error_type",
        "error_code",
        "provider_error_code",
        "provider_status_code",
        "sqlstate",
        "schema_name",
        "table_name",
        "column_name",
        "constraint_name",
    ):
        if key in database:
            diagnostics[key] = database[key]
    return diagnostics


def _collection(
    *,
    compiler_run_id: str,
    snapshot_id: str,
    source_type: EmbeddingSourceType,
    channel: str,
    model: ModelRegistryEntry,
    config: QuestionerConfig,
) -> EmbeddingCollection:
    if model.embedding_dimension != 4096:
        raise ValueError(f"QBank-v1 requires 4096-dimensional embeddings: {model.alias}")
    if model.distance_metric is not DistanceMetric.COSINE or not model.normalize_embeddings:
        raise ValueError(f"QBank-v1 requires normalized cosine embeddings: {model.alias}")
    recipe = {
        "compiler_run_id": compiler_run_id,
        "snapshot_id": snapshot_id,
        "source_type": source_type.value,
        "channel": channel,
        "model_alias": model.alias,
        "model_revision": model.model_revision,
        "model_hash": model.resolved_hash,
        "gate_version": config.gate_version,
        "dedupe_version": config.dedupe_version,
        **(
            {"memory_raw_image_policy": "forbid", "representation": "qbank-question-text-only-v1"}
            if config.is_v36 and config.memory_raw_image_policy == "forbid"
            else {}
        ),
    }
    return EmbeddingCollection(
        collection_id=stable_compiler_id("qbank-collection", recipe),
        snapshot_id=snapshot_id,
        run_id=compiler_run_id,
        source_type=source_type,
        provider=model.endpoint_kind.value,
        model_alias=model.alias,
        actual_model_id=model.expected_actual_model_id,
        model_revision=model.model_revision,
        dimension=4096,
        distance_metric=DistanceMetric.COSINE,
        normalized=True,
        normalization_method="l2-v1",
        preprocessing_hash=canonical_content_hash(recipe),
        metadata={
            "qbank_channel": channel,
            "questioner_protocol": config.protocol_version,
            **(
                {
                    "memory_raw_image_policy": "forbid",
                    "representation": "qbank-question-text-only-v1",
                }
                if config.is_v36 and config.memory_raw_image_policy == "forbid"
                else {}
            ),
        },
    )


async def _embedding_documents(
    sessions: async_sessionmaker[AsyncSession],
    *,
    snapshot_id: str,
    questions: list[QuestionCandidate],
    memory_raw_image_policy: str = "allow",
) -> tuple[list[QBankEmbeddingDocument], list[QBankEmbeddingDocument]]:
    if memory_raw_image_policy not in {"allow", "forbid"}:
        raise ValueError("unknown candidate raw-image policy")
    possible_image_ids = sorted(
        {evidence_id for question in questions for evidence_id in question.evidence_ids}
    )
    image_ids = set()
    if memory_raw_image_policy == "allow":
        async with sessions() as session:
            image_ids = set(
                await session.scalars(
                    select(ImageModel.image_id).where(
                        ImageModel.snapshot_id == snapshot_id,
                        ImageModel.image_id.in_(possible_image_ids or ["__none__"]),
                    )
                )
            )
    image_values = (
        await PostgresImageRepository(sessions).get_many(snapshot_id, sorted(image_ids))
        if image_ids
        else []
    )
    images = {item.image_id: item for item in image_values}
    canonical: list[QBankEmbeddingDocument] = []
    paraphrases: list[QBankEmbeddingDocument] = []
    for question in questions:
        refs = tuple(
            images[evidence_id] for evidence_id in question.evidence_ids if evidence_id in images
        )
        canonical.append(
            QBankEmbeddingDocument(
                source_id=question.qbank_id,
                qbank_id=question.qbank_id,
                text=question.canonical_question,
                image_refs=refs,
                source_type=EmbeddingSourceType.QBANK_QUESTION,
            )
        )
        for ordinal, text in enumerate(question.paraphrases):
            paraphrases.append(
                QBankEmbeddingDocument(
                    source_id=stable_compiler_id(
                        "qbank-paraphrase", [question.qbank_id, ordinal, text]
                    ),
                    qbank_id=question.qbank_id,
                    text=text,
                    image_refs=refs,
                    source_type=EmbeddingSourceType.QBANK_PARAPHRASE,
                )
            )
    return canonical, paraphrases


class QBankIndexBuilder:
    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        execution: ModelExecutionService,
        gateways: dict[str, EmbeddingGateway],
        error_artifacts: ExperimentArtifactService,
        config: QuestionerConfig,
        manifest: QBankIndexManifest,
        collections: dict[str, EmbeddingCollection],
        documents: dict[str, list[QBankEmbeddingDocument]],
        lease_seconds: int,
    ) -> None:
        if lease_seconds < 3:
            raise ValueError("QBank index lease must be at least three seconds")
        self.sessions = sessions
        self.execution = execution
        self.gateways = gateways
        self.error_artifacts = error_artifacts
        self.config = config
        self.manifest = manifest
        self.collections = collections
        self.documents = documents
        self.lease_seconds = lease_seconds
        self.failure_diagnostics: dict[str, dict[str, object]] = {}
        self.batches: dict[str, tuple[str, int, list[QBankEmbeddingDocument]]] = {}
        for channel in sorted(collections):
            batch_size = (
                config.multimodal_embedding_batch_size
                if channel.startswith("multimodal")
                else config.text_embedding_batch_size
            )
            for ordinal, start in enumerate(range(0, len(documents[channel]), batch_size)):
                job_id = stable_compiler_id(
                    "qbank-index-job",
                    [collections[channel].collection_id, ordinal],
                )
                self.batches[job_id] = (
                    channel,
                    ordinal,
                    documents[channel][start : start + batch_size],
                )

    async def execute(self, progress: IndexProgress | None = None) -> QBankIndexManifest:
        run_id = self.manifest.compiler_run_id
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None and uow.embeddings is not None
            assert uow.qbank_indexes is not None
            for collection in self.collections.values():
                if collection.status is EmbeddingCollectionStatus.BUILDING:
                    await uow.embeddings.add_collection(collection)
                elif collection.status is not EmbeddingCollectionStatus.READY:
                    raise RuntimeError(
                        f"QBank collection cannot resume from {collection.status.value}"
                    )
            await uow.qbank_indexes.add(self.manifest)
            for job_id, (channel, ordinal, documents) in self.batches.items():
                expected = ExperimentJob(
                    job_id=job_id,
                    run_id=run_id,
                    snapshot_id=self.manifest.snapshot_id,
                    stage=AppStage.EMBED,
                    variant=f"qbank-{channel}-{ordinal:08d}",
                    max_attempts=self.config.job_max_attempts,
                    metadata={
                        "qbank_index_id": self.manifest.qbank_index_id,
                        "collection_id": self.collections[channel].collection_id,
                        "source_ids": [item.source_id for item in documents],
                    },
                )
                stored = await uow.jobs.get(job_id)
                if stored is None:
                    await uow.jobs.add(expected)
                elif self._immutable_job(stored) != self._immutable_job(expected):
                    raise ValueError(f"persisted QBank index job conflicts: {job_id}")

        async def worker(index: int) -> None:
            worker_id = f"qbank-index-{index}"
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
                stop = asyncio.Event()
                heartbeat = asyncio.create_task(self._heartbeat(job.job_id, worker_id, stop))
                try:
                    await self._process(job, worker_id)
                except Exception as exc:
                    await self._finish_failed(job, worker_id, exc)
                else:
                    if progress is not None:
                        channel, _ordinal, _documents = self.batches[job.job_id]
                        async with ExperimentUnitOfWork(self.sessions) as uow:
                            assert uow.embeddings is not None
                            completed = await uow.embeddings.count_records(
                                self.collections[channel].collection_id
                            )
                        progress(channel, completed, len(self.documents[channel]))
                finally:
                    stop.set()
                    heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await heartbeat

        if self.batches:
            await asyncio.gather(
                *(
                    worker(index)
                    for index in range(min(self.config.embedding_concurrency, len(self.batches)))
                )
            )
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None and uow.embeddings is not None
            assert uow.qbank_indexes is not None
            counts = await uow.jobs.status_counts(run_id, stage=AppStage.EMBED)
            failed_job_count = counts.get(JobStatus.FAILED, 0)
            if failed_job_count:
                raise QBankIndexExecutionError(
                    failed_job_count=failed_job_count,
                    failures=list(self.failure_diagnostics.values()),
                )
            for channel, collection in self.collections.items():
                actual = await uow.embeddings.count_records(collection.collection_id)
                if actual != len(self.documents[channel]):
                    raise RuntimeError(f"QBank collection count is incomplete: {channel}")
                await uow.embeddings.mark_collection_ready(
                    collection.collection_id,
                    index_type="exact",
                    index_params={"ann": False, "distance_metric": "cosine"},
                )
            current = await uow.qbank_indexes.get(self.manifest.qbank_index_id)
            if current is None:
                raise RuntimeError("QBank index manifest disappeared during indexing")
            return current

    async def _process(self, job: ExperimentJob, worker_id: str) -> None:
        channel, ordinal, documents = self.batches[job.job_id]
        collection = self.collections[channel]
        expected_ids = [
            stable_compiler_id("qbank-embedding", [collection.collection_id, item.source_id])
            for item in documents
        ]
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.embeddings is not None
            existing = [
                await uow.embeddings.get_record(embedding_id) for embedding_id in expected_ids
            ]
        count = sum(item is not None for item in existing)
        if count == len(documents):
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.jobs is not None
                await uow.jobs.finish(job.job_id, worker_id, success=True)
            return
        if count:
            raise RuntimeError("QBank embedding batch is partially persisted")

        multimodal = channel.startswith("multimodal")
        inputs = [
            EmbeddingInput(
                input_id=item.source_id,
                text=item.text,
                image_refs=list(item.image_refs) if multimodal else [],
                metadata={"qbank_id": item.qbank_id, "channel": channel},
            )
            for item in documents
        ]
        if collection.metadata.get("memory_raw_image_policy") == "forbid" and any(
            item.image_refs for item in inputs
        ):
            raise ValueError("raw images are forbidden in this candidate collection")
        model_call_id = stable_compiler_id(
            "qbank-index-call",
            [collection.collection_id, ordinal, job.attempt_count],
        )
        audited = await self.execution.embed_audited(
            self.gateways[collection.model_alias],
            EmbeddingRequest(
                request_id=model_call_id,
                model_alias=collection.model_alias,
                inputs=inputs,
                dimensions=collection.dimension,
                metadata={
                    "task": "qbank_index",
                    "channel": channel,
                    "job_attempt_number": job.attempt_count,
                },
            ),
            ModelCallContext(
                model_call_id=model_call_id,
                run_id=job.run_id,
                snapshot_id=job.snapshot_id,
                job_id=job.job_id,
                stage=AppStage.EMBED,
            ),
        )
        if audited.result.actual_model_id != collection.actual_model_id:
            raise ValueError("QBank embedding provider identity changed")
        vectors = {item.input_id: item.vector for item in audited.result.vectors}
        if set(vectors) != {item.source_id for item in documents}:
            raise ValueError("QBank embedding response IDs do not match the request")
        records: list[EmbeddingRecord] = []
        for embedding_id, document in zip(expected_ids, documents, strict=True):
            vector = normalize_vector(vectors[document.source_id])
            records.append(
                EmbeddingRecord(
                    embedding_id=embedding_id,
                    collection_id=collection.collection_id,
                    source_id=document.source_id,
                    qbank_id=(
                        document.qbank_id
                        if document.source_type is EmbeddingSourceType.QBANK_QUESTION
                        else None
                    ),
                    paraphrase_id=(
                        document.source_id
                        if document.source_type is EmbeddingSourceType.QBANK_PARAPHRASE
                        else None
                    ),
                    input_hash=canonical_content_hash(
                        {
                            "text": document.text,
                            "images": [
                                {"image_id": item.image_id, "sha256": item.sha256}
                                for item in document.image_refs
                            ],
                            "collection": collection.preprocessing_hash,
                        }
                    ),
                    dimension=collection.dimension,
                    vector=vector,
                    model_call_id=model_call_id,
                    metadata={"parent_qbank_id": document.qbank_id, "channel": channel},
                )
            )
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.embeddings is not None and uow.jobs is not None
            await uow.embeddings.add_records(records)
            await uow.jobs.finish(job.job_id, worker_id, success=True)

    async def _finish_failed(
        self,
        job: ExperimentJob,
        worker_id: str,
        error: BaseException,
    ) -> None:
        channel, ordinal, documents = self.batches[job.job_id]
        diagnostics = _qbank_batch_failure_diagnostics(
            channel=channel,
            batch_ordinal=ordinal,
            attempt=job.attempt_count,
            document_count=len(documents),
            collection_document_count=len(self.documents[channel]),
            total_input_chars=sum(len(item.text) for item in documents),
            image_input_count=sum(bool(item.image_refs) for item in documents),
            image_reference_count=sum(len(item.image_refs) for item in documents),
            max_image_references=max(
                (len(item.image_refs) for item in documents),
                default=0,
            ),
            error=error,
        )
        self.failure_diagnostics[job.job_id] = diagnostics
        print(
            "QBANK_INDEX_BATCH_FAILURE "
            + json.dumps(
                diagnostics,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        payload = json.dumps(
            diagnostics,
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
            await uow.jobs.finish(
                job.job_id,
                worker_id,
                success=False,
                error_type=type(error).__name__,
                error_summary=(
                    "QBank index batch failed; "
                    f"channel={channel}; batch={ordinal}; "
                    f"cause={diagnostics.get('cause_type', type(error).__name__)}"
                ),
                error_artifact_id=artifact_id,
            )

    async def _heartbeat(self, job_id: str, worker_id: str, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.lease_seconds / 3)
            except TimeoutError:
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    assert uow.jobs is not None
                    await uow.jobs.heartbeat(job_id, worker_id, self.lease_seconds)

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
                "created_at",
                "started_at",
                "finished_at",
                "error_type",
                "error_summary",
                "error_artifact_id",
            },
        )


async def prepare_qbank_index(
    *,
    sessions: async_sessionmaker[AsyncSession],
    execution: ModelExecutionService,
    gateways: dict[str, EmbeddingGateway],
    error_artifacts: ExperimentArtifactService,
    registry: ModelRegistry,
    config: QuestionerConfig,
    compiler_run_id: str,
    snapshot_id: str,
    lease_seconds: int,
    progress: IndexProgress | None = None,
    channels: set[str] | None = None,
    finalize: bool = True,
) -> QBankIndexBuildResult:
    async with ExperimentUnitOfWork(sessions) as uow:
        assert uow.questioner is not None and uow.qbank_indexes is not None
        existing = await uow.qbank_indexes.get_for_run(compiler_run_id)
        questions = await uow.questioner.list_eligible_questions(compiler_run_id)
    if existing is not None and existing.status is EmbeddingCollectionStatus.READY:
        if (
            config.is_v36
            and config.memory_raw_image_policy == "forbid"
            and (existing.metadata.get("memory_raw_image_policy") != "forbid")
        ):
            raise ValueError("raw-image candidate index cannot serve a caption-only treatment")
        return QBankIndexBuildResult(existing, reused=True)

    canonical, paraphrases = await _embedding_documents(
        sessions,
        snapshot_id=snapshot_id,
        questions=questions,
        memory_raw_image_policy=config.memory_raw_image_policy if config.is_v36 else "allow",
    )
    text_model = registry.get(config.text_embedding_model_alias)
    multimodal_model = registry.get(config.multimodal_embedding_model_alias)
    collections = {
        "text_question": _collection(
            compiler_run_id=compiler_run_id,
            snapshot_id=snapshot_id,
            source_type=EmbeddingSourceType.QBANK_QUESTION,
            channel="text_question",
            model=text_model,
            config=config,
        ),
        "text_paraphrase": _collection(
            compiler_run_id=compiler_run_id,
            snapshot_id=snapshot_id,
            source_type=EmbeddingSourceType.QBANK_PARAPHRASE,
            channel="text_paraphrase",
            model=text_model,
            config=config,
        ),
        "multimodal_question": _collection(
            compiler_run_id=compiler_run_id,
            snapshot_id=snapshot_id,
            source_type=EmbeddingSourceType.QBANK_QUESTION,
            channel="multimodal_question",
            model=multimodal_model,
            config=config,
        ),
        "multimodal_paraphrase": _collection(
            compiler_run_id=compiler_run_id,
            snapshot_id=snapshot_id,
            source_type=EmbeddingSourceType.QBANK_PARAPHRASE,
            channel="multimodal_paraphrase",
            model=multimodal_model,
            config=config,
        ),
    }
    async with ExperimentUnitOfWork(sessions) as uow:
        assert uow.embeddings is not None
        for channel, candidate in list(collections.items()):
            stored = await uow.embeddings.get_collection(candidate.collection_id)
            if stored is None:
                continue
            comparable = candidate.model_copy(
                update={
                    "created_at": stored.created_at,
                    "status": stored.status,
                    "index_type": stored.index_type,
                    "index_params": stored.index_params,
                }
            )
            if comparable != stored:
                raise ValueError(f"persisted QBank collection conflicts: {channel}")
            collections[channel] = stored
    documents = {
        "text_question": canonical,
        "text_paraphrase": paraphrases,
        "multimodal_question": canonical,
        "multimodal_paraphrase": paraphrases,
    }
    selected_channels = set(collections) if channels is None else set(channels)
    unknown_channels = selected_channels - set(collections)
    if unknown_channels:
        raise ValueError(f"unknown QBank index channel: {sorted(unknown_channels)[0]}")
    if not selected_channels:
        raise ValueError("QBank index requires at least one selected channel")
    recipe = {
        "compiler_run_id": compiler_run_id,
        "snapshot_id": snapshot_id,
        "collections": {key: value.collection_id for key, value in sorted(collections.items())},
        "eligible_question_count": len(canonical),
        "paraphrase_count": len(paraphrases),
        "gate_version": config.gate_version,
        "matcher_version": config.matcher_version,
        "rrf_constant": config.rrf_constant,
        "per_channel_top_k": config.per_channel_top_k,
    }
    manifest = QBankIndexManifest(
        qbank_index_id=stable_compiler_id("qbank-index", recipe),
        compiler_run_id=compiler_run_id,
        snapshot_id=snapshot_id,
        gate_version=config.gate_version,
        matcher_version=config.matcher_version,
        text_question_collection_id=collections["text_question"].collection_id,
        text_paraphrase_collection_id=collections["text_paraphrase"].collection_id,
        multimodal_question_collection_id=collections["multimodal_question"].collection_id,
        multimodal_paraphrase_collection_id=collections["multimodal_paraphrase"].collection_id,
        eligible_question_count=len(canonical),
        paraphrase_count=len(paraphrases),
        rrf_constant=config.rrf_constant,
        per_channel_top_k=config.per_channel_top_k,
        content_hash=canonical_content_hash(recipe),
        metadata={
            "questioner_protocol": config.protocol_version,
            "text_model_alias": text_model.alias,
            "multimodal_model_alias": multimodal_model.alias,
            **(
                {
                    "memory_raw_image_policy": "forbid",
                    "representation": "qbank-question-text-only-v1",
                }
                if config.is_v36 and config.memory_raw_image_policy == "forbid"
                else {}
            ),
        },
    )
    if existing is not None:
        resumed_manifest = manifest.model_copy(
            update={"created_at": existing.created_at, "status": existing.status}
        )
        if resumed_manifest != existing:
            raise ValueError("persisted QBank index conflicts with the frozen recipe")
        manifest = existing
    async with ExperimentUnitOfWork(sessions) as uow:
        assert uow.embeddings is not None and uow.qbank_indexes is not None
        for collection in collections.values():
            if collection.status is EmbeddingCollectionStatus.BUILDING:
                await uow.embeddings.add_collection(collection)
            elif collection.status is not EmbeddingCollectionStatus.READY:
                raise RuntimeError(f"QBank collection cannot resume from {collection.status.value}")
        await uow.qbank_indexes.add(manifest)
    current = await QBankIndexBuilder(
        sessions=sessions,
        execution=execution,
        gateways=gateways,
        error_artifacts=error_artifacts,
        config=config,
        manifest=manifest,
        collections={key: collections[key] for key in sorted(selected_channels)},
        documents={key: documents[key] for key in sorted(selected_channels)},
        lease_seconds=lease_seconds,
    ).execute(progress)
    if "text_question" in selected_channels:
        async with ExperimentUnitOfWork(sessions) as uow:
            assert uow.questioner is not None
            await uow.questioner.record_semantic_nearest_neighbors(
                run_id=compiler_run_id,
                collection_id=collections["text_question"].collection_id,
                method=f"{config.dedupe_version}:semantic-nearest",
            )
    if finalize:
        async with ExperimentUnitOfWork(sessions) as uow:
            assert uow.embeddings is not None and uow.qbank_indexes is not None
            for channel, collection in collections.items():
                actual = await uow.embeddings.count_records(collection.collection_id)
                if actual != len(documents[channel]):
                    raise RuntimeError(f"QBank collection count is incomplete: {channel}")
                stored = await uow.embeddings.get_collection(collection.collection_id)
                if stored is None or stored.status is not EmbeddingCollectionStatus.READY:
                    raise RuntimeError(f"QBank collection is not ready: {channel}")
            current = await uow.qbank_indexes.mark_ready(manifest.qbank_index_id)
    return QBankIndexBuildResult(current, reused=False)
