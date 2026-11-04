"""Deterministic embedding recipes shared by the three RAG baselines."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol

from pmmc.baselines.rag_config import (
    RagBaselineConfig,
    query_instruction_sha256,
    render_qwen_query_instruction,
)
from pmmc.benchmarks.ids import stable_hash
from pmmc.models.execution import ModelCallContext, ModelExecutionService
from pmmc.models.protocols import EmbeddingGateway
from pmmc.models.registry import ModelRegistry, ModelRegistryEntry
from pmmc.schemas import (
    AppStage,
    BaselineCorpusManifest,
    BaselineMethod,
    BenchmarkQuery,
    DistanceMetric,
    EmbeddingCollection,
    EmbeddingCollectionStatus,
    EmbeddingInput,
    EmbeddingRecord,
    EmbeddingRequest,
    EmbeddingSourceType,
    ImageRef,
    QueryEmbeddingRole,
    RepresentationKind,
    RetrievalDocument,
    RetrievalQueryEmbeddingRef,
)

RAG_METHODS = frozenset(
    {
        BaselineMethod.CAPTION_TEXT_RAG,
        BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG,
        BaselineMethod.NATIVE_MULTIMODAL_RAG,
    }
)


class ImageReader(Protocol):
    async def get_many(self, snapshot_id: str, image_ids: list[str]) -> list[ImageRef]: ...


@dataclass(frozen=True)
class ProducedEmbeddingBatch:
    records: list[EmbeddingRecord]
    model_call_id: str
    actual_model_id: str


@dataclass(frozen=True)
class QueryEmbeddingBundle:
    records: list[EmbeddingRecord]
    references: list[RetrievalQueryEmbeddingRef]
    effective_vector: list[float]
    model_call_id: str
    actual_model_id: str


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_rag_method(method: BaselineMethod) -> None:
    if method not in RAG_METHODS:
        raise ValueError(f"not a RAG baseline method: {method.value}")


def document_collection_source(method: BaselineMethod) -> EmbeddingSourceType:
    _require_rag_method(method)
    if method is BaselineMethod.CAPTION_TEXT_RAG:
        return EmbeddingSourceType.RETRIEVAL_DOCUMENT_TEXT_CAPTION
    if method is BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG:
        return EmbeddingSourceType.RETRIEVAL_DOCUMENT_CONTRASTIVE
    return EmbeddingSourceType.RETRIEVAL_DOCUMENT_MULTIMODAL


def collection_preprocessing_hash(
    method: BaselineMethod,
    config: RagBaselineConfig,
    entry: ModelRegistryEntry,
) -> str:
    _require_rag_method(method)
    return _canonical_hash(
        {
            "protocol_version": config.protocol_version,
            "renderer_version": config.renderer_version,
            "method": method.value,
            "representations": [item.value for item in config.representations_for(method)],
            "model": entry.model_dump(mode="json"),
            "query_instruction_version": config.query_instruction_version,
            "query_instruction_sha256": query_instruction_sha256(config),
            "contrastive_text_weight": config.contrastive_text_weight,
            "contrastive_image_weight": config.contrastive_image_weight,
            "exact_scan": config.exact_scan,
            "distance_metric": config.distance_metric.value,
            **({"memory_raw_image_policy": "forbid"}
               if config.memory_raw_image_policy == "forbid" else {}),
        }
    )


def document_collection_id(
    corpus: BaselineCorpusManifest,
    method: BaselineMethod,
    config: RagBaselineConfig,
    entry: ModelRegistryEntry,
) -> str:
    fingerprint = stable_hash(
        {
            "snapshot_id": corpus.snapshot_id,
            "corpus_id": corpus.corpus_id,
            "corpus_hash": corpus.content_hash,
            "method": method.value,
            "model_alias": entry.alias,
            "model_revision": entry.model_revision,
            "dimension": entry.embedding_dimension,
            "normalized": entry.normalize_embeddings,
            "distance_metric": entry.distance_metric.value if entry.distance_metric else None,
            "preprocessing_hash": collection_preprocessing_hash(method, config, entry),
        }
    )
    return f"rag-index-{fingerprint[:24]}"


def query_collection_id(
    *,
    run_id: str,
    snapshot_id: str,
    method: BaselineMethod,
    config: RagBaselineConfig,
    entry: ModelRegistryEntry,
) -> str:
    fingerprint = stable_hash(
        {
            "run_id": run_id,
            "snapshot_id": snapshot_id,
            "method": method.value,
            "model_alias": entry.alias,
            "model_revision": entry.model_revision,
            "dimension": entry.embedding_dimension,
            "preprocessing_hash": collection_preprocessing_hash(method, config, entry),
            "query_instruction_version": config.query_instruction_version,
        }
    )
    return f"rag-query-{fingerprint[:24]}"


def make_document_collection(
    *,
    corpus: BaselineCorpusManifest,
    method: BaselineMethod,
    config: RagBaselineConfig,
    registry: ModelRegistry,
    run_id: str,
) -> EmbeddingCollection:
    alias = config.embedding_alias_for(method)
    entry = registry.get(alias)
    if entry.embedding_dimension is None or entry.distance_metric is None:
        raise ValueError("RAG embedding model requires dimension and distance metric")
    if entry.distance_metric is not DistanceMetric.COSINE:
        raise ValueError("Step 5C main baselines require cosine embedding collections")
    return EmbeddingCollection(
        collection_id=document_collection_id(corpus, method, config, entry),
        snapshot_id=corpus.snapshot_id,
        corpus_id=corpus.corpus_id,
        run_id=run_id,
        source_type=document_collection_source(method),
        provider=entry.endpoint_kind.value,
        model_alias=entry.alias,
        actual_model_id=entry.expected_actual_model_id,
        model_revision=entry.model_revision,
        dimension=entry.embedding_dimension,
        distance_metric=entry.distance_metric,
        normalized=True,
        normalization_method="l2-v1",
        preprocessing_hash=collection_preprocessing_hash(method, config, entry),
        index_type="exact",
        index_params={"ann": False, "reranker": False},
        status=EmbeddingCollectionStatus.BUILDING,
        metadata={
            "protocol_version": config.protocol_version,
            "representations": [item.value for item in config.representations_for(method)],
            **({"memory_raw_image_policy": "forbid"}
               if config.memory_raw_image_policy == "forbid" else {}),
        },
    )


def make_query_collection(
    *,
    run_id: str,
    snapshot_id: str,
    method: BaselineMethod,
    config: RagBaselineConfig,
    registry: ModelRegistry,
) -> EmbeddingCollection:
    entry = registry.get(config.embedding_alias_for(method))
    if entry.embedding_dimension is None or entry.distance_metric is None:
        raise ValueError("RAG embedding model requires dimension and distance metric")
    return EmbeddingCollection(
        collection_id=query_collection_id(
            run_id=run_id,
            snapshot_id=snapshot_id,
            method=method,
            config=config,
            entry=entry,
        ),
        snapshot_id=snapshot_id,
        run_id=run_id,
        source_type=EmbeddingSourceType.QUERY,
        provider=entry.endpoint_kind.value,
        model_alias=entry.alias,
        actual_model_id=entry.expected_actual_model_id,
        model_revision=entry.model_revision,
        dimension=entry.embedding_dimension,
        distance_metric=entry.distance_metric,
        normalized=True,
        normalization_method="l2-v1",
        preprocessing_hash=collection_preprocessing_hash(method, config, entry),
        status=EmbeddingCollectionStatus.BUILDING,
        metadata={
            "protocol_version": config.protocol_version,
            "query_instruction_version": config.query_instruction_version,
        },
    )


def normalize_vector(vector: list[float]) -> list[float]:
    if not vector or any(not math.isfinite(float(value)) for value in vector):
        raise ValueError("embedding vector must contain finite values")
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 0:
        raise ValueError("embedding vector norm must be positive")
    return [float(value) / norm for value in vector]


def fuse_contrastive_query(
    text_vector: list[float],
    image_vectors: list[list[float]],
    *,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
) -> list[float]:
    text = normalize_vector(text_vector)
    if not image_vectors:
        return text
    images = [normalize_vector(vector) for vector in image_vectors]
    if any(len(vector) != len(text) for vector in images):
        raise ValueError("contrastive query vectors must share one dimension")
    centroid = normalize_vector(
        [sum(vector[index] for vector in images) / len(images) for index in range(len(text))]
    )
    return normalize_vector(
        [
            text_weight * text_value + image_weight * image_value
            for text_value, image_value in zip(text, centroid, strict=True)
        ]
    )


def embedding_record_id(collection_id: str, source_id: str) -> str:
    """Return the stable identity used for both document and query vectors."""

    return "emb-" + stable_hash(
        {"collection_id": collection_id, "source_id": source_id}
    )[:28]


async def document_embedding_input(
    document: RetrievalDocument,
    method: BaselineMethod,
    images: ImageReader,
    *, memory_raw_image_policy: str = "allow",
) -> EmbeddingInput:
    expected = {
        BaselineMethod.CAPTION_TEXT_RAG: {RepresentationKind.TEXT_CAPTION},
        BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG: {
            RepresentationKind.TEXT,
            RepresentationKind.IMAGE,
        },
        BaselineMethod.NATIVE_MULTIMODAL_RAG: {RepresentationKind.MULTIMODAL},
    }
    _require_rag_method(method)
    if memory_raw_image_policy == "forbid":
        expected[method] = {RepresentationKind.TEXT_CAPTION}
    if document.representation not in expected[method]:
        raise ValueError("retrieval document representation does not match RAG method")
    image_refs = (
        await images.get_many(document.snapshot_id, document.image_ids)
        if document.image_ids and method is not BaselineMethod.CAPTION_TEXT_RAG
        and memory_raw_image_policy != "forbid"
        else []
    )
    if method is BaselineMethod.CAPTION_TEXT_RAG:
        image_refs = []
    return EmbeddingInput(
        input_id=document.document_id,
        text=document.rendered_text,
        image_refs=image_refs,
        metadata={
            "representation": document.representation.value,
            "content_hash": document.content_hash,
        },
    )


async def produce_document_embeddings(
    *,
    documents: list[RetrievalDocument],
    method: BaselineMethod,
    collection: EmbeddingCollection,
    images: ImageReader,
    gateway: EmbeddingGateway,
    execution: ModelExecutionService,
    run_id: str,
    batch_ordinal: int,
    attempt_number: int = 1,
    job_id: str | None = None,
) -> ProducedEmbeddingBatch:
    if not documents:
        raise ValueError("embedding batch cannot be empty")
    inputs = [
        await document_embedding_input(
            document, method, images,
            memory_raw_image_policy=collection.metadata.get("memory_raw_image_policy", "allow"),
        ) for document in documents
    ]
    model_call_id = "call-" + stable_hash(
        {
            "collection_id": collection.collection_id,
            "batch_ordinal": batch_ordinal,
            "input_ids": [item.input_id for item in inputs],
            "attempt_number": attempt_number,
        }
    )[:28]
    request = EmbeddingRequest(
        request_id=model_call_id,
        model_alias=collection.model_alias,
        inputs=inputs,
        metadata={
            "collection_id": collection.collection_id,
            "batch_ordinal": batch_ordinal,
            "method": method.value,
        },
    )
    audited = await execution.embed_audited(
        gateway,
        request,
        ModelCallContext(
            model_call_id=model_call_id,
            run_id=run_id,
            snapshot_id=collection.snapshot_id,
            job_id=job_id,
            stage=AppStage.EMBED,
        ),
    )
    vectors = {item.input_id: normalize_vector(item.vector) for item in audited.result.vectors}
    expected_input_ids = {item.input_id for item in inputs}
    if set(vectors) != expected_input_ids or len(vectors) != len(audited.result.vectors):
        raise ValueError("embedding provider returned a different document input set")
    records = [
        EmbeddingRecord(
            embedding_id=embedding_record_id(collection.collection_id, document.document_id),
            collection_id=collection.collection_id,
            source_id=document.document_id,
            retrieval_document_id=document.document_id,
            input_hash=document.content_hash,
            dimension=collection.dimension,
            vector=vectors[document.document_id],
            model_call_id=model_call_id,
            metadata={"representation": document.representation.value},
        )
        for document in documents
    ]
    return ProducedEmbeddingBatch(
        records=records,
        model_call_id=model_call_id,
        actual_model_id=audited.result.actual_model_id,
    )


def query_embedding_source_id(
    query_id: str, role: QueryEmbeddingRole, ordinal: int
) -> str:
    return f"{query_id}:{role.value}:{ordinal}"


async def produce_query_embeddings(
    *,
    query: BenchmarkQuery,
    method: BaselineMethod,
    collection: EmbeddingCollection,
    config: RagBaselineConfig,
    gateway: EmbeddingGateway,
    execution: ModelExecutionService,
    retrieval_batch_id: str,
    job_id: str,
    attempt_number: int,
) -> QueryEmbeddingBundle:
    if collection.run_id is None:
        raise ValueError("query embedding collections require run_id")
    instruction = render_qwen_query_instruction(config, query.query.raw_query)
    inputs: list[EmbeddingInput]
    roles: list[QueryEmbeddingRole]
    if method is BaselineMethod.CAPTION_TEXT_RAG:
        inputs = [
            EmbeddingInput(
                input_id=query_embedding_source_id(
                    query.query.query_id, QueryEmbeddingRole.TEXT, 0
                ),
                text=instruction,
            )
        ]
        roles = [QueryEmbeddingRole.TEXT]
    elif method is BaselineMethod.NATIVE_MULTIMODAL_RAG:
        inputs = [
            EmbeddingInput(
                input_id=query_embedding_source_id(
                    query.query.query_id, QueryEmbeddingRole.JOINT, 0
                ),
                text=instruction,
                image_refs=query.query.query_image_refs,
            )
        ]
        roles = [QueryEmbeddingRole.JOINT]
    elif method is BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG:
        inputs = [
            EmbeddingInput(
                input_id=query_embedding_source_id(
                    query.query.query_id, QueryEmbeddingRole.TEXT, 0
                ),
                text=query.query.raw_query,
            ),
            *[
                EmbeddingInput(
                    input_id=query_embedding_source_id(
                        query.query.query_id, QueryEmbeddingRole.IMAGE, ordinal
                    ),
                    image_refs=[image],
                )
                for ordinal, image in enumerate(query.query.query_image_refs)
            ],
        ]
        roles = [QueryEmbeddingRole.TEXT] + [
            QueryEmbeddingRole.IMAGE for _image in query.query.query_image_refs
        ]
    else:
        raise ValueError(f"not a RAG baseline method: {method.value}")

    model_call_id = "call-" + stable_hash(
        {
            "run_id": collection.run_id,
            "query_id": query.query.query_id,
            "method": method.value,
            "attempt_number": attempt_number,
        }
    )[:28]
    audited = await execution.embed_audited(
        gateway,
        EmbeddingRequest(
            request_id=model_call_id,
            model_alias=collection.model_alias,
            inputs=inputs,
            metadata={
                "collection_id": collection.collection_id,
                "method": method.value,
                "query_instruction_version": config.query_instruction_version,
            },
        ),
        ModelCallContext(
            model_call_id=model_call_id,
            run_id=collection.run_id,
            snapshot_id=collection.snapshot_id,
            query_id=query.query.query_id,
            conversation_id=query.query.conversation_id,
            job_id=job_id,
            stage=AppStage.EMBED,
        ),
    )
    vector_by_id = {
        item.input_id: normalize_vector(item.vector) for item in audited.result.vectors
    }
    expected_input_ids = {item.input_id for item in inputs}
    if (
        set(vector_by_id) != expected_input_ids
        or len(vector_by_id) != len(audited.result.vectors)
    ):
        raise ValueError("embedding provider returned a different query input set")
    records: list[EmbeddingRecord] = []
    refs: list[RetrievalQueryEmbeddingRef] = []
    for ordinal, (item, role) in enumerate(zip(inputs, roles, strict=True)):
        vector = vector_by_id[item.input_id]
        record = EmbeddingRecord(
            embedding_id=embedding_record_id(collection.collection_id, item.input_id),
            collection_id=collection.collection_id,
            source_id=item.input_id,
            input_hash=_canonical_hash(item.model_dump(mode="json")),
            dimension=collection.dimension,
            vector=vector,
            model_call_id=model_call_id,
            metadata={"query_id": query.query.query_id, "role": role.value},
        )
        records.append(record)
        refs.append(
            RetrievalQueryEmbeddingRef(
                retrieval_batch_id=retrieval_batch_id,
                embedding_id=record.embedding_id,
                role=role,
                ordinal=ordinal,
                used_for_search=method is not BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG,
            )
        )

    if method is BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG:
        text_vector = records[0].vector
        image_vectors = [record.vector for record in records[1:]]
        fused = fuse_contrastive_query(
            text_vector,
            image_vectors,
            text_weight=config.contrastive_text_weight,
            image_weight=config.contrastive_image_weight,
        )
        source_id = query_embedding_source_id(
            query.query.query_id, QueryEmbeddingRole.FUSED, 0
        )
        fused_record = EmbeddingRecord(
            embedding_id=embedding_record_id(collection.collection_id, source_id),
            collection_id=collection.collection_id,
            source_id=source_id,
            input_hash=_canonical_hash(
                {
                    "component_embedding_ids": [record.embedding_id for record in records],
                    "text_weight": config.contrastive_text_weight,
                    "image_weight": config.contrastive_image_weight,
                }
            ),
            dimension=collection.dimension,
            vector=fused,
            model_call_id=model_call_id,
            metadata={
                "query_id": query.query.query_id,
                "role": QueryEmbeddingRole.FUSED.value,
                "derived_from": [record.embedding_id for record in records],
            },
        )
        records.append(fused_record)
        refs.append(
            RetrievalQueryEmbeddingRef(
                retrieval_batch_id=retrieval_batch_id,
                embedding_id=fused_record.embedding_id,
                role=QueryEmbeddingRole.FUSED,
                ordinal=len(refs),
                used_for_search=True,
            )
        )
        effective = fused
    else:
        effective = records[0].vector

    return QueryEmbeddingBundle(
        records=records,
        references=refs,
        effective_vector=effective,
        model_call_id=model_call_id,
        actual_model_id=audited.result.actual_model_id,
    )


def restore_query_embedding_bundle(
    *,
    records: list[EmbeddingRecord],
    retrieval_batch_id: str,
    method: BaselineMethod,
    expected_image_count: int,
) -> QueryEmbeddingBundle:
    """Reconstruct query-vector lineage after vectors committed before retrieval."""

    _require_rag_method(method)
    expected_roles = (
        [QueryEmbeddingRole.TEXT]
        if method is BaselineMethod.CAPTION_TEXT_RAG
        else [QueryEmbeddingRole.JOINT]
        if method is BaselineMethod.NATIVE_MULTIMODAL_RAG
        else [
            QueryEmbeddingRole.TEXT,
            *[QueryEmbeddingRole.IMAGE for _ in range(expected_image_count)],
            QueryEmbeddingRole.FUSED,
        ]
    )
    by_source = {record.source_id: record for record in records}
    query_ids = {
        str(record.metadata.get("query_id"))
        for record in records
        if record.metadata.get("query_id") is not None
    }
    if len(query_ids) != 1:
        raise ValueError("persisted query embeddings must belong to one query")
    query_id = next(iter(query_ids))
    ordered: list[EmbeddingRecord] = []
    refs: list[RetrievalQueryEmbeddingRef] = []
    role_ordinals: dict[QueryEmbeddingRole, int] = {}
    for ordinal, role in enumerate(expected_roles):
        role_ordinal = role_ordinals.get(role, 0)
        source_id = query_embedding_source_id(query_id, role, role_ordinal)
        role_ordinals[role] = role_ordinal + 1
        record = by_source.get(source_id)
        if record is None:
            raise ValueError("persisted query embedding set is incomplete")
        ordered.append(record)
        refs.append(
            RetrievalQueryEmbeddingRef(
                retrieval_batch_id=retrieval_batch_id,
                embedding_id=record.embedding_id,
                role=role,
                ordinal=ordinal,
                used_for_search=(
                    role is QueryEmbeddingRole.FUSED
                    if method is BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG
                    else ordinal == 0
                ),
            )
        )
    if len(records) != len(ordered):
        raise ValueError("persisted query embedding set contains unexpected vectors")
    effective = next(
        record.vector for record, ref in zip(ordered, refs, strict=True) if ref.used_for_search
    )
    model_call_ids = {record.model_call_id for record in ordered}
    if len(model_call_ids) != 1 or None in model_call_ids:
        raise ValueError("persisted query embeddings require one audited model call")
    return QueryEmbeddingBundle(
        records=ordered,
        references=refs,
        effective_vector=effective,
        model_call_id=str(next(iter(model_call_ids))),
        actual_model_id="persisted",
    )
