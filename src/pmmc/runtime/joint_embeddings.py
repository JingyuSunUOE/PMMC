"""Release-R read-side embeddings. Never bind this collection to a program operator."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from pmmc.baselines.rag_embedding import normalize_vector
from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.models.protocols import EmbeddingGateway
from pmmc.models.registry import ModelRegistryEntry
from pmmc.runtime.answer_context import RuntimeMemoryReader
from pmmc.runtime.query_embeddings import QueryEmbeddingWorkflow, runtime_query_cache_lease
from pmmc.runtime.query_executor_config import JOINT_POLICY
from pmmc.schemas import (
    DataScope,
    DistanceMetric,
    EmbeddingCollection,
    EmbeddingInput,
    EmbeddingRecord,
    EmbeddingRequest,
    EmbeddingSourceType,
    ImageRef,
    MemoryUnit,
    QueryInput,
    RunStatus,
)

JOINT_MODEL = "qwen3-vl-embedding-8b"


@dataclass(frozen=True)
class JointPair:
    owner_id: str
    image_id: str
    input_hash: str
    native_input: EmbeddingInput


def _image_identities(image: ImageRef) -> set[tuple[str, str]]:
    return {
        (key, value)
        for key, value in (
            ("image", image.image_id),
            ("asset", image.asset_id),
            ("sha256", image.sha256.lower() if image.sha256 else None),
            ("storage", image.storage_uri),
            ("source_url", image.remote_url),
            ("source_path", image.local_path),
        )
        if value
    }


def _native_image(image: ImageRef) -> ImageRef | None:
    # Use the synchronized object, not a mutable external URL or a machine-local path.
    try:
        uri = urlsplit(image.storage_uri or "")
    except ValueError:
        return None
    digest = (image.sha256 or "").lower()
    if (
        uri.scheme != "s3"
        or not uri.hostname
        or uri.username is not None
        or not uri.path.lstrip("/")
        or uri.query
        or uri.fragment
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        return None
    return ImageRef(
        image_id=image.image_id,
        asset_id=image.asset_id,
        storage_uri=image.storage_uri,
        sha256=digest,
        mime_type=image.mime_type,
        width=image.width,
        height=image.height,
    )


def visible_joint_pairs(
    snapshot_id: str,
    query: QueryInput,
    memories: Sequence[MemoryUnit],
) -> tuple[list[JointPair], dict[str, str]]:
    """Require an explicit canonical link, not a neighboring turn or a caption."""
    owners = Counter(
        image_id
        for m in memories
        for image_id in (m.image_ids or [i.image_id for i in m.image_refs])
    )
    supplied = {key for img in query.query_image_refs for key in _image_identities(img)}
    pairs: list[JointPair] = []
    reasons: dict[str, str] = {}
    for memory in memories:
        if memory.conversation_id != query.conversation_id or memory.dataset != query.dataset:
            raise ValueError("joint embedding owner escaped the visible conversation")
        refs = {i.image_id: i for i in memory.image_refs}
        for image_id in memory.image_ids or refs:
            image = refs.get(image_id)
            native_image = _native_image(image) if image else None
            if ("image", image_id) in supplied or (image and supplied & _image_identities(image)):
                reason = "query_image"
            elif owners[image_id] != 1:
                reason = "ambiguous_owner"
            elif native_image is None:
                reason = "no_canonical_asset"
            elif not memory.text or not memory.text.strip():
                reason = "no_owner_text"
            else:
                identity = canonical_content_hash(
                    {
                        "policy": JOINT_POLICY,
                        "snapshot": snapshot_id,
                        "conversation": memory.conversation_id,
                        "owner": memory.memory_id,
                        "image": image_id,
                        "image_sha256": native_image.sha256,
                        "text": memory.text,
                    }
                )
                pairs.append(
                    JointPair(
                        memory.memory_id,
                        image_id,
                        identity,
                        EmbeddingInput(
                            input_id=identity,
                            text=memory.text,
                            image_refs=[native_image],
                        ),
                    )
                )
                continue
            reasons[image_id] = reason
    return sorted(pairs, key=lambda p: (p.owner_id, p.image_id)), reasons


async def scoped_joint_pairs(reader: RuntimeMemoryReader, snapshot_id: str, query: QueryInput):
    if not query.available_after_session:
        raise ValueError("joint embedding requires a prospective cutoff")
    memories = [
        m
        async for m in reader.iter_visible(
            snapshot_id, query.conversation_id, query.available_after_session
        )
    ]
    return visible_joint_pairs(snapshot_id, query, memories)


def joint_shards(snapshot_id: str, entry: ModelRegistryEntry, pairs: list[JointPair]):
    # Keep batches deterministic across backbones, W variants and repeated queries.
    if entry.alias != JOINT_MODEL:
        raise ValueError("joint evidence and runtime query must use the native VL embedding space")
    for start in range(0, len(pairs), 16):
        batch = pairs[start : start + 16]
        cache_id = stable_compiler_id(
            "runtime-joint-cache",
            [
                JOINT_POLICY,
                snapshot_id,
                entry.resolved_hash,
                [p.input_hash for p in batch],
            ],
        )
        yield (
            batch,
            cache_id,
            EmbeddingCollection(
                collection_id=stable_compiler_id("runtime-joint-collection", cache_id),
                run_id=stable_compiler_id("runtime-query-cache-owner", cache_id),
                snapshot_id=snapshot_id,
                source_type=EmbeddingSourceType.MEMORY_IMAGE_TEXT_PAIR,
                model_alias=entry.alias,
                model_revision=entry.model_revision,
                actual_model_id=entry.expected_actual_model_id,
                dimension=4096,
                distance_metric=DistanceMetric.COSINE,
                normalized=True,
                normalization_method="l2-v1",
                preprocessing_hash=canonical_content_hash(JOINT_POLICY),
                metadata={"joint_policy": JOINT_POLICY, "model_entry_hash": entry.resolved_hash},
            ),
        )


def _record_id(collection: EmbeddingCollection, pair: JointPair) -> str:
    return stable_compiler_id("runtime-joint-record", [collection.collection_id, pair.input_hash])


def _producer_hash(collection: EmbeddingCollection, call_id: str) -> str:
    return canonical_content_hash([collection.run_id, collection.collection_id, call_id])


async def load_joint_records(sessions, collection, pairs) -> dict[str, EmbeddingRecord]:
    expected = {_record_id(collection, pair): pair for pair in pairs}
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.embeddings is not None and uow.runs is not None
        current = await uow.embeddings.get_collection(collection.collection_id)
        ignored = {"created_at", "status", "index_type", "index_params"}
        if (
            current is None
            or current.status != "ready"
            or current.model_dump(exclude=ignored) != collection.model_dump(exclude=ignored)
            or (await uow.runs.get_state(collection.run_id)).status is not RunStatus.COMPLETED
        ):
            raise RuntimeError("release R requires its completed precomputed joint collection")
        rows = await uow.embeddings.get_records(list(expected))
        if rows.keys() != expected.keys():
            raise ValueError("incomplete release R joint embedding denominator")
        for key, record in rows.items():
            pair = expected[key]
            if (
                record.collection_id != collection.collection_id
                or record.source_id != pair.image_id
                or record.input_hash != pair.input_hash
                or not record.model_call_id
                or record.dimension != collection.dimension
                or record.metadata.get("owner_id") != pair.owner_id
                or record.metadata.get("joint_policy") != JOINT_POLICY
                or record.metadata.get("producer_hash")
                != _producer_hash(collection, record.model_call_id)
            ):
                raise ValueError("joint embedding record has incompatible scientific lineage")
    return {record.source_id: record for record in rows.values()}


async def prepare_joint_shards(
    *,
    builder: QueryEmbeddingWorkflow,
    snapshot_id: str,
    query: QueryInput,
    reader: RuntimeMemoryReader,
    entry: ModelRegistryEntry,
    gateway: Callable[[], EmbeddingGateway],
    lease_seconds: int,
) -> tuple[dict[str, int], dict[str, dict[str, str]]]:
    pairs, _ = await scoped_joint_pairs(reader, snapshot_id, query)
    calls = reused = 0
    producers = {}
    for batch, cache_id, collection in joint_shards(snapshot_id, entry, pairs):
        async with runtime_query_cache_lease(
            builder.sessions,
            cache_id=cache_id,
            snapshot_id=snapshot_id,
            dataset=query.dataset,
            model_alias=entry.alias,
            lease_seconds=lease_seconds,
        ) as lease:
            if lease is None:
                stored = await load_joint_records(builder.sessions, collection, batch)
                call_ids = {r.model_call_id for r in stored.values()}
                if len(call_ids) != 1:
                    raise ValueError("a joint batch must have one original producer call")
                producers[collection.collection_id] = {
                    "run_id": str(collection.run_id),
                    "model_call_id": str(call_ids.pop()),
                }
                reused += 1
                continue
            await builder._ensure_collection(collection)
            request = EmbeddingRequest(
                request_id=stable_compiler_id("runtime-joint-batch", cache_id),
                model_alias=entry.alias,
                dimensions=collection.dimension,
                inputs=[p.native_input for p in batch],
                metadata={
                    "scope": "runtime",
                    "joint_policy": JOINT_POLICY,
                    "persist_normalized_embedding_result": True,
                },
            )
            result, call_id, recovered = await builder._load_or_call(
                manifest=lease.owner,
                request=request,
                gateway=gateway(),
            )
            if (
                result.request_id != request.request_id
                or result.model_alias != entry.alias
                or [v.input_id for v in result.vectors] != [p.input_hash for p in batch]
            ):
                raise ValueError("joint embedding response differs from its exact pair batch")
            records = [
                EmbeddingRecord(
                    embedding_id=_record_id(collection, pair),
                    collection_id=collection.collection_id,
                    source_id=pair.image_id,
                    input_hash=pair.input_hash,
                    model_call_id=call_id,
                    dimension=collection.dimension,
                    vector=normalize_vector(vector.vector),
                    metadata={
                        "owner_id": pair.owner_id,
                        "joint_policy": JOINT_POLICY,
                        "producer_hash": _producer_hash(collection, call_id),
                    },
                )
                for pair, vector in zip(batch, result.vectors, strict=True)
            ]
            async with ExperimentUnitOfWork(builder.sessions, DataScope.RUNTIME) as uow:
                assert uow.embeddings and uow.jobs and uow.runs
                await uow.jobs.heartbeat(lease.job_id, lease.worker_id, lease_seconds)
                await uow.embeddings.add_records(records)
                await uow.embeddings.mark_collection_ready(
                    collection.collection_id,
                    index_type="exact",
                    index_params={"ann": False},
                )
                await uow.jobs.finish(lease.job_id, lease.worker_id, success=True)
                await uow.runs.transition(lease.owner.query_executor_run_id, RunStatus.COMPLETED)
            calls += not recovered
            reused += recovered
            producers[collection.collection_id] = {
                "run_id": lease.owner.query_executor_run_id,
                "model_call_id": call_id,
            }
    return {
        "joint_pair_count": len(pairs),
        "joint_model_call_count": calls,
        "joint_reused_batch_count": reused,
    }, producers
