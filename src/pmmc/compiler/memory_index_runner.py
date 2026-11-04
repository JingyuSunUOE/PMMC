"""Recoverable builders for memory-operator indexes and program query vectors."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.baselines.rag_embedding import normalize_vector
from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.memory_index import (
    PROGRAM_JOINT_QUERY_INSTRUCTION,
    PROGRAM_TEXT_QUERY_INSTRUCTION,
    MemoryIndexInput,
    make_memory_operator_index,
    memory_index_inputs,
)
from pmmc.infrastructure.database.errors import safe_database_error_diagnostics
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.models import EmbeddingGateway, ModelExecutionService, ModelRegistry
from pmmc.models.execution import ModelCallContext, embedding_result_from_artifact
from pmmc.schemas import (
    AppStage,
    DataScope,
    DistanceMetric,
    EmbeddingCollection,
    EmbeddingCollectionStatus,
    EmbeddingInput,
    EmbeddingRecord,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSourceType,
    MemoryOperatorIndexManifest,
    MemoryUnit,
    ModelCallTrace,
    ProgramQueryEmbedding,
    ProgramQueryEmbeddingBatch,
    QuestionCandidate,
    TraceStatus,
)

MemoryIndexProgress = Callable[[str, int, int], None]

_MAX_PERSISTED_EMBEDDING_FAILURES = 32
_MAX_INLINE_EMBEDDING_CHANNEL_ATTEMPTS = 3
_INLINE_EMBEDDING_RETRY_DELAY_SECONDS = 2.0


class PersistedEmbeddingCallFailedError(RuntimeError):
    """A terminal failed call that may be superseded by a new execution attempt."""

    def __init__(self, error_type: str | None) -> None:
        super().__init__("persisted embedding model call failed")
        self.provider_error_type = error_type or "unknown"


class EmbeddingBatchExecutionError(RuntimeError):
    """Secret-free locator for one failed embedding batch."""

    def __init__(
        self,
        *,
        batch_kind: str,
        channel: str,
        batch_ordinal: int,
        batch_start: int,
        batch_size: int,
        total: int,
        operation: str,
        existing_count: int,
        recovery_attempt: int,
        failed_call_count: int,
        total_input_chars: int,
        max_input_chars: int,
        image_input_count: int,
        cause: BaseException,
    ) -> None:
        self.batch_kind = batch_kind
        self.channel = channel
        self.batch_ordinal = batch_ordinal
        self.batch_start = batch_start
        self.batch_size = batch_size
        self.total = total
        self.operation = operation
        self.existing_count = existing_count
        self.recovery_attempt = recovery_attempt
        self.failed_call_count = failed_call_count
        self.total_input_chars = total_input_chars
        self.max_input_chars = max_input_chars
        self.image_input_count = image_input_count
        self.cause_type = type(cause).__name__
        status_code = getattr(cause, "provider_status_code", None)
        if not isinstance(status_code, int):
            response = getattr(cause, "response", None)
            status_code = getattr(response, "status_code", None)
        self.http_status = status_code if isinstance(status_code, int) else None
        provider_error_type = getattr(cause, "error_code", None)
        if not isinstance(provider_error_type, str) or not provider_error_type:
            provider_error_type = str(getattr(cause, "provider_error_type", self.cause_type))
        retryable = getattr(cause, "retryable", None)
        self.retryable = retryable if isinstance(retryable, bool) else None
        self.cause_code = (
            f"{provider_error_type}:http_{self.http_status}"
            if self.http_status is not None
            else provider_error_type
        )
        database = safe_database_error_diagnostics(cause)
        self.database_diagnostics: dict[str, str] = {
            key: database[key]
            for key in (
                "database_error_type",
                "sqlstate",
                "schema_name",
                "table_name",
                "column_name",
                "constraint_name",
            )
            if key in database
        }
        super().__init__(self.safe_summary)

    @property
    def safe_summary(self) -> str:
        database = "".join(
            f" {key}={self.database_diagnostics[key]}"
            for key in ("sqlstate", "schema_name", "table_name", "constraint_name")
            if key in self.database_diagnostics
        )
        return (
            f"kind={self.batch_kind} channel={self.channel} "
            f"batch={self.batch_ordinal} range={self.batch_start}:"
            f"{self.batch_start + self.batch_size}/{self.total} "
            f"operation={self.operation} existing={self.existing_count} "
            f"recovery_attempt={self.recovery_attempt} "
            f"failed_calls={self.failed_call_count} "
            f"input_chars={self.total_input_chars} "
            f"max_input_chars={self.max_input_chars} "
            f"image_inputs={self.image_input_count} cause={self.cause_code}"
            f"{database}"
        )

    def safe_diagnostics(self) -> dict[str, object]:
        return {
            "batch_kind": self.batch_kind,
            "batch_ordinal": self.batch_ordinal,
            "batch_size": self.batch_size,
            "batch_start": self.batch_start,
            "cause_code": self.cause_code,
            "cause_type": self.cause_type,
            "channel": self.channel,
            "existing_count": self.existing_count,
            "failed_call_count": self.failed_call_count,
            "http_status": self.http_status,
            "image_input_count": self.image_input_count,
            "max_input_chars": self.max_input_chars,
            "operation": self.operation,
            "recovery_attempt": self.recovery_attempt,
            "retryable": self.retryable,
            "total": self.total,
            "total_input_chars": self.total_input_chars,
            **self.database_diagnostics,
        }

    @property
    def can_retry_inline(self) -> bool:
        """Whether rerunning the channel can safely recover this batch."""

        if self.retryable is False:
            return False
        if self.operation == "validate_vectors":
            return False
        sqlstate = self.database_diagnostics.get("sqlstate", "")
        if sqlstate.startswith("23"):
            return False
        if 0 < self.existing_count < self.batch_size:
            return False
        return True


async def _run_embedding_channel_with_retries(
    *,
    batch_kind: str,
    channel: str,
    total: int,
    batch_size: int,
    concurrency: int = 1,
    run_once: Callable[[], Awaitable[None]],
) -> None:
    """Retry a recoverable channel while persisted batches remain authoritative."""

    batch_count = (total + batch_size - 1) // batch_size if total else 0
    for attempt in range(1, _MAX_INLINE_EMBEDDING_CHANNEL_ATTEMPTS + 1):
        attempt_started = time.monotonic()
        print(
            "EMBEDDING_CHANNEL_ATTEMPT "
            + json.dumps(
                {
                    "attempt": attempt,
                    "batch_count": batch_count,
                    "batch_kind": batch_kind,
                    "batch_size": batch_size,
                    "channel": channel,
                    "concurrency": concurrency,
                    "effective_concurrency": min(concurrency, batch_count),
                    "max_attempts": _MAX_INLINE_EMBEDDING_CHANNEL_ATTEMPTS,
                    "state": "starting",
                    "total": total,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        try:
            await run_once()
        except EmbeddingBatchExecutionError as exc:
            will_retry = attempt < _MAX_INLINE_EMBEDDING_CHANNEL_ATTEMPTS and exc.can_retry_inline
            print(
                "EMBEDDING_BATCH_RETRY "
                + json.dumps(
                    {
                        **exc.safe_diagnostics(),
                        "channel_attempt": attempt,
                        "max_channel_attempts": (_MAX_INLINE_EMBEDDING_CHANNEL_ATTEMPTS),
                        "sleep_seconds": (
                            _INLINE_EMBEDDING_RETRY_DELAY_SECONDS if will_retry else 0.0
                        ),
                        "state": "retrying" if will_retry else "terminal",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if not will_retry:
                raise
            await asyncio.sleep(_INLINE_EMBEDDING_RETRY_DELAY_SECONDS)
            continue
        print(
            "EMBEDDING_CHANNEL_ATTEMPT "
            + json.dumps(
                {
                    "attempt": attempt,
                    "batch_count": batch_count,
                    "batch_kind": batch_kind,
                    "channel": channel,
                    "concurrency": concurrency,
                    "effective_concurrency": min(concurrency, batch_count),
                    "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                    "state": "completed",
                    "total": total,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return


@dataclass(frozen=True)
class EmbeddingCallSelection:
    model_call_id: str
    recovered: tuple[EmbeddingResult, ModelCallTrace] | None
    recovery_attempt: int
    failed_call_count: int


@dataclass(frozen=True)
class ProgramQueryVectorSet:
    qbank_id: str
    text_vector: list[float]
    joint_vector: list[float]


async def _recover_embedding_result(
    *,
    sessions: async_sessionmaker[AsyncSession],
    execution: ModelExecutionService,
    model_call_id: str,
):  # type: ignore[no-untyped-def]
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        trace = await uow.model_calls.get(model_call_id)
        if trace is None:
            return None
        if trace.status is not TraceStatus.SUCCESS:
            raise PersistedEmbeddingCallFailedError(trace.error_type)
        if trace.output_asset_id is None:
            raise RuntimeError("successful embedding model call has no output artifact")
        artifact = await uow.artifacts.get(trace.output_asset_id)
    if artifact is None or artifact.size_bytes is None:
        raise RuntimeError("embedding response artifact is not ready")
    store = execution.artifact_service.store
    remote = await store.head(artifact.object_key)
    if (
        remote is None
        or remote.sha256 != artifact.sha256
        or remote.size_bytes != artifact.size_bytes
    ):
        raise RuntimeError("embedding response artifact failed integrity verification")
    result = embedding_result_from_artifact(await store.get_bytes(artifact.object_key))
    return result, trace


async def _select_embedding_call(
    *,
    sessions: async_sessionmaker[AsyncSession],
    execution: ModelExecutionService,
    run_id: str,
    logical_model_call_id: str,
) -> EmbeddingCallSelection:
    """Reuse a success, or allocate the first append-only retry after failures."""

    failed_call_count = 0
    for recovery_attempt in range(_MAX_PERSISTED_EMBEDDING_FAILURES + 1):
        model_call_id = (
            logical_model_call_id
            if recovery_attempt == 0
            else stable_compiler_id(
                "embedding-model-call-attempt",
                [run_id, logical_model_call_id, recovery_attempt],
            )
        )
        try:
            recovered = await _recover_embedding_result(
                sessions=sessions,
                execution=execution,
                model_call_id=model_call_id,
            )
        except PersistedEmbeddingCallFailedError:
            failed_call_count += 1
            continue
        return EmbeddingCallSelection(
            model_call_id=model_call_id,
            recovered=recovered,
            recovery_attempt=recovery_attempt,
            failed_call_count=failed_call_count,
        )
    raise RuntimeError("embedding batch exhausted persisted failure recovery budget")


async def load_program_query_vectors(
    sessions: async_sessionmaker[AsyncSession], *, run_id: str
) -> dict[str, ProgramQueryVectorSet]:
    """Load complete text+joint vector pairs without invoking a model."""

    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.embeddings is not None and uow.program_query_embeddings is not None
        refs = await uow.program_query_embeddings.list_for_run(run_id)
        records = {
            item.embedding_id: await uow.embeddings.get_record(item.embedding_id) for item in refs
        }
    by_qbank: dict[str, dict[str, list[float]]] = {}
    for ref in refs:
        record = records[ref.embedding_id]
        if record is None:
            raise RuntimeError("program query embedding reference is dangling")
        by_qbank.setdefault(ref.qbank_id, {})[ref.channel] = record.vector
    return {
        qbank_id: ProgramQueryVectorSet(
            qbank_id=qbank_id,
            text_vector=values["question_text_query"],
            joint_vector=values["question_joint_query"],
        )
        for qbank_id, values in by_qbank.items()
        if set(values) == {"question_text_query", "question_joint_query"}
    }


def _query_collection(
    *,
    run_id: str,
    snapshot_id: str,
    channel: str,
    model_alias: str,
    model_revision: str,
) -> EmbeddingCollection:
    identity = [run_id, snapshot_id, channel, model_alias, model_revision]
    return EmbeddingCollection(
        collection_id=stable_compiler_id("program-query-collection", identity),
        snapshot_id=snapshot_id,
        run_id=run_id,
        source_type=EmbeddingSourceType.QUERY,
        model_alias=model_alias,
        model_revision=model_revision,
        dimension=4096,
        distance_metric=DistanceMetric.COSINE,
        normalized=True,
        normalization_method="l2-v1",
        preprocessing_hash=canonical_content_hash(identity),
        index_type="exact",
        metadata={"query_role": channel, "expected_images_allowed": False},
    )


class MemoryOperatorIndexBuilder:
    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        execution: ModelExecutionService,
        gateways: Mapping[str, EmbeddingGateway],
        registry: ModelRegistry,
        text_batch_size: int = 32,
        multimodal_batch_size: int = 8,
        embedding_concurrency: int | None = None,
    ) -> None:
        resolved_concurrency = (
            int(os.environ.get("PMMC_QBANK_EMBEDDING_CONCURRENCY", "1"))
            if embedding_concurrency is None
            else embedding_concurrency
        )
        if text_batch_size <= 0 or multimodal_batch_size <= 0 or resolved_concurrency <= 0:
            raise ValueError("memory index batch sizes and concurrency must be positive")
        self.sessions = sessions
        self.execution = execution
        self.gateways = gateways
        self.registry = registry
        self.text_batch_size = text_batch_size
        self.multimodal_batch_size = multimodal_batch_size
        self.embedding_concurrency = resolved_concurrency

    async def build(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        dataset: str,
        selection_hash: str,
        memories: Sequence[MemoryUnit],
        text_model_alias: str,
        multimodal_model_alias: str,
        progress: MemoryIndexProgress | None = None,
        channels: set[str] | None = None,
        memory_raw_image_policy: str = "allow",
    ) -> MemoryOperatorIndexManifest:
        selected_channels = channels or {"text_memory", "visual_memory"}
        if not selected_channels <= {"text_memory", "visual_memory"}:
            raise ValueError("unknown memory operator index channel")
        text_entry = self.registry.get(text_model_alias)
        multimodal_entry = self.registry.get(multimodal_model_alias)
        for entry in (text_entry, multimodal_entry):
            if (
                entry.embedding_dimension != 4096
                or entry.distance_metric is not DistanceMetric.COSINE
                or not entry.normalize_embeddings
            ):
                raise ValueError("memory operator indexes require normalized 4096d cosine models")
        manifest, text_collection, visual_collection = make_memory_operator_index(
            run_id=run_id,
            snapshot_id=snapshot_id,
            dataset=dataset,
            selection_hash=selection_hash,
            text_model_alias=text_model_alias,
            text_model_revision=text_entry.model_revision,
            multimodal_model_alias=multimodal_model_alias,
            multimodal_model_revision=multimodal_entry.model_revision,
            memory_raw_image_policy=memory_raw_image_policy,
        )
        text_inputs, visual_inputs = memory_index_inputs(
            memories,
            memory_raw_image_policy=memory_raw_image_policy,
        )
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.embeddings is not None and uow.memory_operator_indexes is not None
            for collection in (text_collection, visual_collection):
                existing = await uow.embeddings.get_collection(collection.collection_id)
                if existing is None:
                    await uow.embeddings.add_collection(collection)
                elif self._collection_identity(existing) != self._collection_identity(collection):
                    raise ValueError("memory embedding collection identity conflicts")
            stored_manifest = await uow.memory_operator_indexes.get_for_run(run_id)
            if stored_manifest is None:
                await uow.memory_operator_indexes.add(manifest)
            elif stored_manifest.content_hash != manifest.content_hash:
                raise ValueError("memory operator index identity conflicts")
            elif stored_manifest.status == "ready":
                return stored_manifest

        if "text_memory" in selected_channels:
            await self._build_channel(
                run_id=run_id,
                collection=text_collection,
                values=text_inputs,
                batch_size=self.text_batch_size,
                progress=progress,
            )
        if "visual_memory" in selected_channels:
            await self._build_channel(
                run_id=run_id,
                collection=visual_collection,
                values=visual_inputs,
                batch_size=self.multimodal_batch_size,
                progress=progress,
            )
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.embeddings is not None and uow.memory_operator_indexes is not None
            channel_values = {
                "text_memory": (text_collection, len(text_inputs)),
                "visual_memory": (visual_collection, len(visual_inputs)),
            }
            for channel in selected_channels:
                collection, total = channel_values[channel]
                actual = await uow.embeddings.count_records(collection.collection_id)
                if actual != total:
                    raise RuntimeError(
                        "memory index cannot become ready before every vector exists"
                    )
                stored_collection = await uow.embeddings.get_collection(collection.collection_id)
                if (
                    stored_collection is None
                    or stored_collection.status is not EmbeddingCollectionStatus.READY
                ):
                    await uow.embeddings.mark_collection_ready(
                        collection.collection_id,
                        index_type="exact",
                        index_params={"ann": False, "distance_metric": "cosine"},
                    )
            stored_collections = [
                await uow.embeddings.get_collection(value.collection_id)
                for value in (text_collection, visual_collection)
            ]
            if all(
                value is not None and value.status.value == "ready" for value in stored_collections
            ):
                return await uow.memory_operator_indexes.mark_ready(manifest.memory_index_id)
            stored = await uow.memory_operator_indexes.get_for_run(run_id)
            assert stored is not None
            return stored

    async def _build_channel(
        self,
        *,
        run_id: str,
        collection: EmbeddingCollection,
        values: Sequence[MemoryIndexInput],
        batch_size: int,
        progress: MemoryIndexProgress | None,
    ) -> None:
        channel = str(collection.metadata["collection_role"])
        reported = 0

        def monotonic_progress(progress_channel: str, current: int, total: int) -> None:
            nonlocal reported
            if progress is None or current <= reported:
                return
            reported = current
            progress(progress_channel, current, total)

        await _run_embedding_channel_with_retries(
            batch_kind="memory_index",
            channel=channel,
            total=len(values),
            batch_size=batch_size,
            concurrency=self.embedding_concurrency,
            run_once=lambda: self._build_channel_once(
                run_id=run_id,
                collection=collection,
                values=values,
                batch_size=batch_size,
                progress=monotonic_progress,
            ),
        )

    async def _build_channel_once(
        self,
        *,
        run_id: str,
        collection: EmbeddingCollection,
        values: Sequence[MemoryIndexInput],
        batch_size: int,
        progress: MemoryIndexProgress | None,
    ) -> None:
        semaphore = asyncio.Semaphore(self.embedding_concurrency)
        progress_lock = asyncio.Lock()
        completed = 0
        channel = str(collection.metadata["collection_role"])

        async def build_batch(batch_ordinal: int, start: int) -> None:
            nonlocal completed
            batch = list(values[start : start + batch_size])
            async with semaphore:
                await self._build_channel_serial(
                    run_id=run_id,
                    collection=collection,
                    values=batch,
                    batch_size=batch_size,
                    progress=None,
                    batch_ordinal_offset=batch_ordinal,
                    batch_start_offset=start,
                    total=len(values),
                )
            async with progress_lock:
                completed += len(batch)
                if progress is not None:
                    progress(channel, completed, len(values))

        results = await asyncio.gather(
            *(
                build_batch(batch_ordinal, start)
                for batch_ordinal, start in enumerate(range(0, len(values), batch_size))
            ),
            return_exceptions=True,
        )
        failures = [value for value in results if isinstance(value, BaseException)]
        if not failures:
            return
        located = [value for value in failures if isinstance(value, EmbeddingBatchExecutionError)]
        if located:
            raise min(located, key=lambda value: value.batch_ordinal)
        raise failures[0]

    async def _build_channel_serial(
        self,
        *,
        run_id: str,
        collection: EmbeddingCollection,
        values: Sequence[MemoryIndexInput],
        batch_size: int,
        progress: MemoryIndexProgress | None,
        batch_ordinal_offset: int = 0,
        batch_start_offset: int = 0,
        total: int | None = None,
    ) -> None:
        total_count = len(values) if total is None else total
        for local_ordinal, local_start in enumerate(range(0, len(values), batch_size)):
            batch_ordinal = batch_ordinal_offset + local_ordinal
            start = batch_start_offset + local_start
            batch = list(values[local_start : local_start + batch_size])
            channel = str(collection.metadata["collection_role"])
            operation = "load_existing"
            existing_count = 0
            recovery_attempt = 0
            failed_call_count = 0
            text_lengths = [len(item.text or "") for item in batch]
            total_input_chars = sum(text_lengths)
            max_input_chars = max(text_lengths, default=0)
            image_input_count = sum(item.image is not None for item in batch)
            try:
                expected_ids = [
                    stable_compiler_id(
                        "memory-index-embedding",
                        [collection.collection_id, item.source_id],
                    )
                    for item in batch
                ]
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.embeddings is not None
                    existing = await uow.embeddings.get_records(expected_ids)
                existing_count = len(existing)
                if existing_count == len(batch):
                    if progress is not None:
                        progress(channel, start + len(batch), total_count)
                    continue
                if existing_count:
                    raise RuntimeError("memory index batch is partially persisted")

                operation = "select_model_call"
                logical_model_call_id = stable_compiler_id(
                    "memory-index-model-call",
                    [collection.collection_id, batch_ordinal],
                )
                selection = await _select_embedding_call(
                    sessions=self.sessions,
                    execution=self.execution,
                    run_id=run_id,
                    logical_model_call_id=logical_model_call_id,
                )
                model_call_id = selection.model_call_id
                recovery_attempt = selection.recovery_attempt
                failed_call_count = selection.failed_call_count
                if failed_call_count:
                    print(
                        "EMBEDDING_BATCH_RECOVERY "
                        + json.dumps(
                            {
                                "batch_kind": "memory_index",
                                "batch_ordinal": batch_ordinal,
                                "channel": channel,
                                "failed_call_count": failed_call_count,
                                "recovered_success": selection.recovered is not None,
                                "recovery_attempt": recovery_attempt,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                request_inputs = [
                    EmbeddingInput(
                        input_id=item.source_id,
                        text=item.text,
                        image_refs=[item.image] if item.image is not None else [],
                        metadata={"memory_id": item.memory_id},
                    )
                    for item in batch
                ]
                if collection.metadata.get("memory_raw_image_policy") == "forbid" and any(
                    item.image_refs for item in request_inputs
                ):
                    raise ValueError("caption-only collection cannot embed raw images")
                request = EmbeddingRequest(
                    request_id=model_call_id,
                    model_alias=collection.model_alias,
                    inputs=request_inputs,
                    dimensions=collection.dimension,
                    metadata={
                        "task": "memory_operator_index",
                        "collection_id": collection.collection_id,
                        "batch_ordinal": batch_ordinal,
                        "recovery_attempt": recovery_attempt,
                        "persist_normalized_embedding_result": True,
                    },
                )
                if selection.recovered is None:
                    operation = "provider_embed"
                    audited = await self.execution.embed_audited(
                        self.gateways[collection.model_alias],
                        request,
                        ModelCallContext(
                            model_call_id=model_call_id,
                            run_id=run_id,
                            snapshot_id=collection.snapshot_id,
                            stage=AppStage.EMBED,
                        ),
                    )
                    embedding_result = audited.result
                else:
                    embedding_result, _ = selection.recovered
                operation = "validate_vectors"
                vectors = {
                    item.input_id: normalize_vector(item.vector)
                    for item in embedding_result.vectors
                }
                if set(vectors) != {item.source_id for item in batch}:
                    raise ValueError("memory embedding provider returned a different input set")
                records = [
                    EmbeddingRecord(
                        embedding_id=embedding_id,
                        collection_id=collection.collection_id,
                        source_id=item.source_id,
                        input_hash=canonical_content_hash(
                            request_inputs[index].model_dump(mode="json")
                        ),
                        dimension=collection.dimension,
                        vector=vectors[item.source_id],
                        model_call_id=model_call_id,
                        metadata={
                            "memory_id": item.memory_id,
                            "collection_role": channel,
                        },
                    )
                    for index, (embedding_id, item) in enumerate(
                        zip(expected_ids, batch, strict=True)
                    )
                ]
                operation = "persist_records"
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.embeddings is not None
                    await uow.embeddings.add_records(records)
                if progress is not None:
                    progress(channel, start + len(batch), total_count)
            except EmbeddingBatchExecutionError:
                raise
            except Exception as exc:
                raise EmbeddingBatchExecutionError(
                    batch_kind="memory_index",
                    channel=channel,
                    batch_ordinal=batch_ordinal,
                    batch_start=start,
                    batch_size=len(batch),
                    total=total_count,
                    operation=operation,
                    existing_count=existing_count,
                    recovery_attempt=recovery_attempt,
                    failed_call_count=failed_call_count,
                    total_input_chars=total_input_chars,
                    max_input_chars=max_input_chars,
                    image_input_count=image_input_count,
                    cause=exc,
                ) from exc

    @staticmethod
    def _collection_identity(value: EmbeddingCollection) -> tuple[object, ...]:
        return (
            value.collection_id,
            value.snapshot_id,
            value.source_type,
            value.model_alias,
            value.model_revision,
            value.dimension,
            value.distance_metric,
            value.preprocessing_hash,
        )


class ProgramQueryEmbeddingBuilder:
    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        execution: ModelExecutionService,
        gateways: Mapping[str, EmbeddingGateway],
        registry: ModelRegistry,
        text_batch_size: int = 32,
        multimodal_batch_size: int = 8,
        embedding_concurrency: int | None = None,
    ) -> None:
        resolved_concurrency = (
            int(os.environ.get("PMMC_QBANK_EMBEDDING_CONCURRENCY", "1"))
            if embedding_concurrency is None
            else embedding_concurrency
        )
        self.sessions = sessions
        self.execution = execution
        self.gateways = gateways
        self.registry = registry
        self.batch_sizes = {
            "question_text_query": text_batch_size,
            "question_joint_query": multimodal_batch_size,
        }
        if min(self.batch_sizes.values()) <= 0 or resolved_concurrency <= 0:
            raise ValueError("program query embedding batch sizes and concurrency must be positive")
        self.embedding_concurrency = resolved_concurrency

    async def build(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        candidates: Sequence[QuestionCandidate],
        text_model_alias: str,
        multimodal_model_alias: str,
        channels: set[str] | None = None,
    ) -> dict[str, ProgramQueryVectorSet]:
        selected_channels = channels or {
            "question_text_query",
            "question_joint_query",
        }
        if not selected_channels <= {
            "question_text_query",
            "question_joint_query",
        }:
            raise ValueError("unknown program query embedding channel")
        aliases = {
            "question_text_query": text_model_alias,
            "question_joint_query": multimodal_model_alias,
        }
        collections = {
            channel: _query_collection(
                run_id=run_id,
                snapshot_id=snapshot_id,
                channel=channel,
                model_alias=alias,
                model_revision=self.registry.get(alias).model_revision,
            )
            for channel, alias in aliases.items()
        }
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.embeddings is not None
            for collection in collections.values():
                existing = await uow.embeddings.get_collection(collection.collection_id)
                if existing is None:
                    await uow.embeddings.add_collection(collection)

        ordered = sorted(candidates, key=lambda item: (item.compiler_unit_id or "", item.qbank_id))
        for channel in ("question_text_query", "question_joint_query"):
            if channel not in selected_channels:
                continue
            await self._build_channel(
                run_id=run_id,
                candidates=ordered,
                channel=channel,
                collection=collections[channel],
            )
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.embeddings is not None and uow.program_query_embeddings is not None
            for channel in selected_channels:
                collection = collections[channel]
                if await uow.embeddings.count_records(collection.collection_id) != len(ordered):
                    raise RuntimeError("program query collection is incomplete")
                await uow.embeddings.mark_collection_ready(
                    collection.collection_id,
                    index_type="exact",
                    index_params={"ann": False, "query_vectors": True},
                )
            refs = await uow.program_query_embeddings.list_for_run(run_id)
            records = {
                item.embedding_id: await uow.embeddings.get_record(item.embedding_id)
                for item in refs
            }
        by_qbank: dict[str, dict[str, list[float]]] = {}
        for ref in refs:
            record = records[ref.embedding_id]
            if record is None:
                raise RuntimeError("program query embedding reference is dangling")
            by_qbank.setdefault(ref.qbank_id, {})[ref.channel] = record.vector
        return {
            qbank_id: ProgramQueryVectorSet(
                qbank_id=qbank_id,
                text_vector=channels["question_text_query"],
                joint_vector=channels["question_joint_query"],
            )
            for qbank_id, channels in by_qbank.items()
            if set(channels) == {"question_text_query", "question_joint_query"}
        }

    async def load(self, *, run_id: str) -> dict[str, ProgramQueryVectorSet]:
        """Load only complete text+joint vector pairs without invoking a model."""
        return await load_program_query_vectors(self.sessions, run_id=run_id)

    async def _build_channel(
        self,
        *,
        run_id: str,
        candidates: Sequence[QuestionCandidate],
        channel: str,
        collection: EmbeddingCollection,
    ) -> None:
        batch_size = self.batch_sizes[channel]
        await _run_embedding_channel_with_retries(
            batch_kind="program_query",
            channel=channel,
            total=len(candidates),
            batch_size=batch_size,
            concurrency=self.embedding_concurrency,
            run_once=lambda: self._build_channel_once(
                run_id=run_id,
                candidates=candidates,
                channel=channel,
                collection=collection,
            ),
        )

    async def _build_channel_once(
        self,
        *,
        run_id: str,
        candidates: Sequence[QuestionCandidate],
        channel: str,
        collection: EmbeddingCollection,
    ) -> None:
        batch_size = self.batch_sizes[channel]
        semaphore = asyncio.Semaphore(self.embedding_concurrency)

        async def build_batch(batch_ordinal: int, start: int) -> None:
            values = list(candidates[start : start + batch_size])
            async with semaphore:
                await self._build_channel_serial(
                    run_id=run_id,
                    candidates=values,
                    channel=channel,
                    collection=collection,
                    batch_ordinal_offset=batch_ordinal,
                    batch_start_offset=start,
                    total=len(candidates),
                )

        results = await asyncio.gather(
            *(
                build_batch(batch_ordinal, start)
                for batch_ordinal, start in enumerate(range(0, len(candidates), batch_size))
            ),
            return_exceptions=True,
        )
        failures = [value for value in results if isinstance(value, BaseException)]
        if not failures:
            return
        located = [value for value in failures if isinstance(value, EmbeddingBatchExecutionError)]
        if located:
            raise min(located, key=lambda value: value.batch_ordinal)
        raise failures[0]

    async def _build_channel_serial(
        self,
        *,
        run_id: str,
        candidates: Sequence[QuestionCandidate],
        channel: str,
        collection: EmbeddingCollection,
        batch_ordinal_offset: int = 0,
        batch_start_offset: int = 0,
        total: int | None = None,
    ) -> None:
        batch_size = self.batch_sizes[channel]
        total_count = len(candidates) if total is None else total
        instruction = (
            PROGRAM_TEXT_QUERY_INSTRUCTION
            if channel == "question_text_query"
            else PROGRAM_JOINT_QUERY_INSTRUCTION
        )
        for local_ordinal, local_start in enumerate(range(0, len(candidates), batch_size)):
            batch_ordinal = batch_ordinal_offset + local_ordinal
            start = batch_start_offset + local_start
            values = list(candidates[local_start : local_start + batch_size])
            rendered_texts = [
                f"Instruct: {instruction}\nQuery: {item.canonical_question}" for item in values
            ]
            operation = "load_existing"
            existing_count = 0
            recovery_attempt = 0
            failed_call_count = 0
            total_input_chars = sum(len(item) for item in rendered_texts)
            max_input_chars = max((len(item) for item in rendered_texts), default=0)
            image_input_count = 0
            try:
                batch_id = stable_compiler_id(
                    "program-query-embedding-batch",
                    [run_id, channel, [item.qbank_id for item in values]],
                )
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.program_query_embeddings is not None
                    existing = await uow.program_query_embeddings.list_for_run(run_id)
                existing_ids = {item.qbank_id for item in existing if item.channel == channel}
                present = [item.qbank_id in existing_ids for item in values]
                existing_count = sum(present)
                if all(present):
                    continue
                if any(present):
                    raise RuntimeError("program query embedding batch is partially persisted")
                inputs = [
                    EmbeddingInput(
                        input_id=item.qbank_id,
                        text=rendered_texts[index],
                        image_refs=[],
                        metadata={"query_role": channel, "text_only": True},
                    )
                    for index, item in enumerate(values)
                ]
                operation = "select_model_call"
                logical_model_call_id = stable_compiler_id(
                    "program-query-model-call",
                    [batch_id, collection.model_revision],
                )
                selection = await _select_embedding_call(
                    sessions=self.sessions,
                    execution=self.execution,
                    run_id=run_id,
                    logical_model_call_id=logical_model_call_id,
                )
                model_call_id = selection.model_call_id
                recovery_attempt = selection.recovery_attempt
                failed_call_count = selection.failed_call_count
                if failed_call_count:
                    print(
                        "EMBEDDING_BATCH_RECOVERY "
                        + json.dumps(
                            {
                                "batch_kind": "program_query",
                                "batch_ordinal": batch_ordinal,
                                "channel": channel,
                                "failed_call_count": failed_call_count,
                                "recovered_success": selection.recovered is not None,
                                "recovery_attempt": recovery_attempt,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                request = EmbeddingRequest(
                    request_id=model_call_id,
                    model_alias=collection.model_alias,
                    inputs=inputs,
                    dimensions=4096,
                    metadata={
                        "task": "program_query_embedding",
                        "channel": channel,
                        "recovery_attempt": recovery_attempt,
                        "persist_normalized_embedding_result": True,
                    },
                )
                if selection.recovered is None:
                    operation = "provider_embed"
                    audited = await self.execution.embed_audited(
                        self.gateways[collection.model_alias],
                        request,
                        ModelCallContext(
                            model_call_id=model_call_id,
                            run_id=run_id,
                            snapshot_id=collection.snapshot_id,
                            stage=AppStage.EMBED,
                        ),
                    )
                    embedding_result = audited.result
                else:
                    embedding_result, _ = selection.recovered
                operation = "validate_vectors"
                vectors = {
                    item.input_id: normalize_vector(item.vector)
                    for item in embedding_result.vectors
                }
                if set(vectors) != {item.qbank_id for item in values}:
                    raise ValueError("program query provider returned a different input set")
                records = [
                    EmbeddingRecord(
                        embedding_id=stable_compiler_id(
                            "program-query-embedding",
                            [collection.collection_id, item.qbank_id],
                        ),
                        collection_id=collection.collection_id,
                        source_id=item.qbank_id,
                        qbank_id=item.qbank_id,
                        input_hash=canonical_content_hash(inputs[index].model_dump(mode="json")),
                        dimension=4096,
                        vector=vectors[item.qbank_id],
                        model_call_id=model_call_id,
                        metadata={
                            "query_role": channel,
                            "instruction_version": instruction,
                        },
                    )
                    for index, item in enumerate(values)
                ]
                batch = ProgramQueryEmbeddingBatch(
                    query_embedding_batch_id=batch_id,
                    run_id=run_id,
                    channel=channel,
                    model_call_id=model_call_id,
                    batch_ordinal=batch_ordinal,
                    qbank_ids=[item.qbank_id for item in values],
                    dimension=4096,
                    content_hash=canonical_content_hash(
                        [run_id, channel, [item.qbank_id for item in values]]
                    ),
                )
                refs = [
                    ProgramQueryEmbedding(
                        program_query_embedding_id=stable_compiler_id(
                            "program-query-ref", [batch_id, item.qbank_id]
                        ),
                        query_embedding_batch_id=batch_id,
                        run_id=run_id,
                        qbank_id=item.qbank_id,
                        channel=channel,
                        embedding_id=record.embedding_id,
                        input_hash=record.input_hash,
                    )
                    for item, record in zip(values, records, strict=True)
                ]
                operation = "persist_records"
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.embeddings is not None
                    assert uow.program_query_embeddings is not None
                    await uow.embeddings.add_records(records)
                    await uow.program_query_embeddings.add_batch(batch, refs)
            except EmbeddingBatchExecutionError:
                raise
            except Exception as exc:
                raise EmbeddingBatchExecutionError(
                    batch_kind="program_query",
                    channel=channel,
                    batch_ordinal=batch_ordinal,
                    batch_start=start,
                    batch_size=len(values),
                    total=total_count,
                    operation=operation,
                    existing_count=existing_count,
                    recovery_attempt=recovery_attempt,
                    failed_call_count=failed_call_count,
                    total_input_chars=total_input_chars,
                    max_input_chars=max_input_chars,
                    image_input_count=image_input_count,
                    cause=exc,
                ) from exc
