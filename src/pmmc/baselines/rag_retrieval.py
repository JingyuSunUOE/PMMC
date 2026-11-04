"""Window-level exact retrieval facts for the three RAG baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from pmmc.benchmarks.ids import stable_hash
from pmmc.schemas import (
    AppStage,
    BaselineMethod,
    BenchmarkQuery,
    EvidenceType,
    RetrievalBatchTrace,
    RetrievalQueryEmbeddingRef,
    RetrievalTrace,
    RetrievalTraceComponent,
    TraceStatus,
    WindowSearchHit,
)


class WindowEmbeddingSearch(Protocol):
    async def search_windows_exact(
        self,
        collection_id: str,
        query_vector: list[float],
        *,
        corpus_id: str,
        top_k: int,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> list[WindowSearchHit]: ...


@dataclass(frozen=True)
class RagRetrievalFacts:
    batch: RetrievalBatchTrace
    traces: list[RetrievalTrace]
    components: list[RetrievalTraceComponent]
    query_embeddings: list[RetrievalQueryEmbeddingRef]


def retrieval_batch_id(run_id: str, query_id: str, method: BaselineMethod) -> str:
    return "retrieval-" + stable_hash(
        {"run_id": run_id, "query_id": query_id, "method": method.value}
    )[:28]


def retrieval_trace_id(batch_id: str, rank: int, window_id: str) -> str:
    return "retrieval-item-" + stable_hash(
        {"retrieval_batch_id": batch_id, "rank": rank, "window_id": window_id}
    )[:28]


def build_retrieval_facts(
    *,
    run_id: str,
    snapshot_id: str,
    corpus_id: str,
    query: BenchmarkQuery,
    method: BaselineMethod,
    document_collection_id: str,
    query_collection_id: str,
    hits: list[WindowSearchHit],
    query_embeddings: list[RetrievalQueryEmbeddingRef],
    requested_top_k: int,
    embed_latency_ms: float,
    search_latency_ms: float,
) -> RagRetrievalFacts:
    batch_id = retrieval_batch_id(run_id, query.query.query_id, method)
    if any(item.retrieval_batch_id != batch_id for item in query_embeddings):
        raise ValueError("query embedding references must target the deterministic batch ID")
    if [item.rank for item in hits] != list(range(1, len(hits) + 1)):
        raise ValueError("window search hits must be contiguous from rank one")
    traces: list[RetrievalTrace] = []
    components: list[RetrievalTraceComponent] = []
    for hit in hits:
        trace_id = retrieval_trace_id(batch_id, hit.rank, hit.window_id)
        traces.append(
            RetrievalTrace(
                retrieval_trace_id=trace_id,
                retrieval_batch_id=batch_id,
                query_id=query.query.query_id,
                run_id=run_id,
                method=method.value,
                retriever_name="exact-cosine-window-v1",
                index_name=document_collection_id,
                chunking_strategy="session_token_window_v1",
                rank=hit.rank,
                retrieved_id=hit.window_id,
                retrieval_document_id=hit.winning_document_id,
                retrieval_window_id=hit.window_id,
                retrieved_type=EvidenceType.MEMORY,
                score=hit.score,
                metadata={"aggregation": "parent_window_max_document"},
            )
        )
        for ordinal, component in enumerate(hit.components):
            components.append(
                RetrievalTraceComponent(
                    retrieval_trace_id=trace_id,
                    retrieval_document_id=component.retrieval_document_id,
                    embedding_collection_id=component.embedding_collection_id,
                    representation=component.representation,
                    ordinal=ordinal,
                    score=component.score,
                    is_winner=(
                        component.retrieval_document_id == hit.winning_document_id
                    ),
                    metadata=component.metadata,
                )
            )
    total_latency_ms = embed_latency_ms + search_latency_ms
    batch = RetrievalBatchTrace(
        retrieval_batch_id=batch_id,
        run_id=run_id,
        snapshot_id=snapshot_id,
        query_id=query.query.query_id,
        stage=AppStage.RETRIEVE,
        method=method.value,
        retriever_name="exact-cosine-window-v1",
        index_name=document_collection_id,
        chunking_strategy="session_token_window_v1",
        requested_top_k=requested_top_k,
        returned_count=len(hits),
        embedding_collection_id=document_collection_id,
        query_embedding_collection_id=query_collection_id,
        corpus_id=corpus_id,
        embed_latency_ms=embed_latency_ms,
        search_latency_ms=search_latency_ms,
        rerank_latency_ms=None,
        total_latency_ms=total_latency_ms,
        status=TraceStatus.SUCCESS,
        metadata={
            "exact_scan": True,
            "reranker": False,
            "window_aggregation": "max_document_score",
        },
    )
    return RagRetrievalFacts(
        batch=batch,
        traces=traces,
        components=components,
        query_embeddings=query_embeddings,
    )


async def search_query_windows(
    *,
    search: WindowEmbeddingSearch,
    run_id: str,
    snapshot_id: str,
    corpus_id: str,
    query: BenchmarkQuery,
    method: BaselineMethod,
    document_collection_id: str,
    query_collection_id: str,
    query_vector: list[float],
    query_embeddings: list[RetrievalQueryEmbeddingRef],
    requested_top_k: int = 50,
    embed_latency_ms: float = 0.0,
) -> RagRetrievalFacts:
    if query.query.available_after_session is None:
        raise ValueError("benchmark RAG queries require a prospective session cutoff")
    started = time.perf_counter()
    hits = await search.search_windows_exact(
        document_collection_id,
        query_vector,
        corpus_id=corpus_id,
        top_k=requested_top_k,
        conversation_id=query.query.conversation_id,
        cutoff_session_id=query.query.available_after_session,
    )
    search_latency_ms = (time.perf_counter() - started) * 1000
    return build_retrieval_facts(
        run_id=run_id,
        snapshot_id=snapshot_id,
        corpus_id=corpus_id,
        query=query,
        method=method,
        document_collection_id=document_collection_id,
        query_collection_id=query_collection_id,
        hits=hits,
        query_embeddings=query_embeddings,
        requested_top_k=requested_top_k,
        embed_latency_ms=embed_latency_ms,
        search_latency_ms=search_latency_ms,
    )
