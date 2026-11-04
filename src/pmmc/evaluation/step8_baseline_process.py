"""Read-only baseline retrieval observations, bound to a native typed terminal.

RAG measures the returned retrieval units before answer-budget truncation.
Full-context measures the evidence actually included in the answer request.
Image hit means explicit canonical image identity (pixels or a source caption),
not raw-image inspection. Query images are never retrieval evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation.evidence import (
    RankedEvidenceCandidate,
    matched_group_ids,
    matched_image_group_ids,
    resolve_gold_evidence,
)
from pmmc.evidence_lineage import EvidenceLineageResolver, EvidenceResolutionPolicy
from pmmc.infrastructure.database.baseline_repositories import (
    PostgresContextAssemblyRepository,
    PostgresCorpusRepository,
)
from pmmc.infrastructure.database.experiment_models import RunQueryModel
from pmmc.infrastructure.database.experiment_repositories import (
    PostgresModelCallRepository,
    PostgresRetrievalRepository,
)
from pmmc.infrastructure.database.models import BenchmarkSessionModel
from pmmc.infrastructure.database.query_executor_models import QueryRuntimeTypedAnswerModel
from pmmc.schemas import (
    AppStage,
    BaselineMethod,
    ContextAssemblyTrace,
    DataScope,
    EvidenceMatchRole,
    EvidenceMemberType,
    MemoryUnit,
    MemoryWindow,
    ModelCallTrace,
    QueryGold,
    QueryInput,
    RepresentationKind,
    RetrievalBatchTrace,
    RetrievalDocument,
    RetrievalTrace,
    TraceStatus,
)
from pmmc.schemas.answer_output import TypedAnswerContext, TypedAnswerResult
from pmmc.schemas.step8_process import RETRIEVED, ProcessCellBinding, ProcessMetric

_FULL = {BaselineMethod.FULL_CONTEXT_CAPTIONS, BaselineMethod.FULL_CONTEXT_RAW_IMAGES}
_RAG = {
    BaselineMethod.CAPTION_TEXT_RAG,
    BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG,
    BaselineMethod.NATIVE_MULTIMODAL_RAG,
}


class _Unavailable(ValueError):
    """Messages are fixed codes only, never source IDs or exception strings."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _Unavailable(reason)


def _missing(
    reason: str,
    status: Literal["incomplete", "not_evaluated", "not_applicable"] = "incomplete",
    *,
    gold_group_count: int | None = None,
) -> dict[str, ProcessMetric]:
    return {
        name: ProcessMetric(
            status=status,
            reason=reason,
            denominator=gold_group_count if name == "retrieved_evidence_recall" else 1,
            denominator_kind="gold_groups" if name == "retrieved_evidence_recall" else "query",
        )
        for name in RETRIEVED
    }


@dataclass(frozen=True)
class _Facts:
    answer: TypedAnswerResult
    answer_hash: str
    call: ModelCallTrace
    context: ContextAssemblyTrace
    visible_sessions: frozenset[str]
    batch: RetrievalBatchTrace | None = None
    hits: tuple[RetrievalTrace, ...] = ()
    windows: dict[str, MemoryWindow] = field(default_factory=dict)
    documents: dict[str, RetrievalDocument] = field(default_factory=dict)


def _decode_answer(
    row: QueryRuntimeTypedAnswerModel, binding: ProcessCellBinding
) -> TypedAnswerResult:
    payload = row.payload.get("metadata", {}).get("typed_result")
    _require(
        isinstance(payload, dict)
        and canonical_content_hash(payload) == row.content_hash == binding.source_fact_hash
        and row.typed_answer_id == binding.source_fact_id,
        "baseline_typed_answer_hash_mismatch",
    )
    answer = TypedAnswerResult.model_validate(payload)
    _require(
        answer.typed_answer_id == row.typed_answer_id
        and answer.run_id == row.query_executor_run_id == binding.source_run_id
        and answer.query_id == row.query_id
        and answer.answer_model_call_id == row.answer_model_call_id
        and answer.terminal_state == row.terminal_state,
        "baseline_typed_answer_row_scope_mismatch",
    )
    return answer


async def _load(
    session: AsyncSession,
    *,
    query: QueryInput,
    snapshot_id: str,
    binding: ProcessCellBinding,
) -> _Facts:
    _require(bool(binding.source_fact_id and binding.source_fact_hash), "baseline_terminal_missing")
    selected = await session.get(
        RunQueryModel, (binding.source_run_id, snapshot_id, query.query_id)
    )
    _require(selected is not None, "baseline_query_not_in_run")
    row = await session.get(QueryRuntimeTypedAnswerModel, binding.source_fact_id)
    _require(row is not None, "baseline_typed_answer_missing")
    assert row is not None
    answer = _decode_answer(row, binding)
    call = await PostgresModelCallRepository(session, DataScope.RUNTIME).get(
        answer.answer_model_call_id
    )
    _require(call is not None and bool(call.context_assembly_id), "baseline_answer_context_missing")
    assert call is not None and call.context_assembly_id is not None
    context = await PostgresContextAssemblyRepository(session).get(call.context_assembly_id)
    cutoff = await session.get(BenchmarkSessionModel, (snapshot_id, query.available_after_session))
    _require(
        cutoff is not None
        and cutoff.conversation_id == query.conversation_id
        and cutoff.dataset == query.dataset.value,
        "baseline_query_cutoff_missing",
    )
    assert cutoff is not None
    visible_sessions = frozenset(
        await session.scalars(
            select(BenchmarkSessionModel.session_id).where(
                BenchmarkSessionModel.snapshot_id == snapshot_id,
                BenchmarkSessionModel.conversation_id == query.conversation_id,
                BenchmarkSessionModel.session_order <= cutoff.session_order,
            )
        )
    )
    batch = None
    hits: list[RetrievalTrace] = []
    windows: dict[str, MemoryWindow] = {}
    documents: dict[str, RetrievalDocument] = {}
    if context.retrieval_batch_id is not None:
        loaded = await PostgresRetrievalRepository(session).get_batch(context.retrieval_batch_id)
        _require(loaded is not None, "baseline_retrieval_batch_missing")
        assert loaded is not None
        batch, hits, _components, _query_refs = loaded
        _require(bool(batch.corpus_id), "baseline_retrieval_corpus_missing")
        assert batch.corpus_id is not None
        corpus = PostgresCorpusRepository(session)
        for hit in hits:
            if hit.retrieval_window_id is not None and hit.retrieval_window_id not in windows:
                windows[hit.retrieval_window_id] = await corpus.get_window(
                    batch.corpus_id, hit.retrieval_window_id
                )
            if hit.retrieval_document_id is not None and hit.retrieval_document_id not in documents:
                documents[hit.retrieval_document_id] = await corpus.get_document(
                    batch.corpus_id, hit.retrieval_document_id
                )
    return _Facts(
        answer,
        row.content_hash,
        call,
        context,
        visible_sessions,
        batch,
        tuple(hits),
        windows,
        documents,
    )


def _images(memory: MemoryUnit) -> set[str]:
    return {*memory.image_ids, *(image.image_id for image in memory.image_refs)}


def _candidate(
    identity: str,
    rank: int,
    memory_ids: list[str],
    image_ids: list[str],
    memories: dict[str, MemoryUnit],
    resolver: EvidenceLineageResolver,
) -> RankedEvidenceCandidate:
    _require(set(memory_ids) <= memories.keys(), "baseline_evidence_outside_visible_lineage")
    visible_images = {image for memory in memories.values() for image in _images(memory)}
    _require(set(image_ids) <= visible_images, "baseline_image_outside_visible_lineage")
    closure = resolver.resolve(
        policy=EvidenceResolutionPolicy.BENCHMARK_GOLD_RESOLUTION,
        memory_ids=memory_ids,
        image_ids=image_ids,
    )
    return RankedEvidenceCandidate(
        candidate_id=identity,
        rank=rank,
        memory_ids=closure.memory_ids,
        session_ids=closure.session_ids,
        image_ids=closure.image_ids,
        evidence_ids=closure.evidence_group_ids,
    )


def _bound_context(
    facts: _Facts,
    query: QueryInput,
    snapshot_id: str,
    binding: ProcessCellBinding,
    memories: dict[str, MemoryUnit],
) -> TypedAnswerContext:
    answer, call, context = facts.answer, facts.call, facts.context
    _require(
        answer.typed_answer_id == binding.source_fact_id
        and facts.answer_hash == binding.source_fact_hash
        and answer.run_id == call.run_id == context.run_id == binding.source_run_id
        and answer.query_id == call.query_id == context.query_id == query.query_id
        and answer.snapshot_id == call.snapshot_id == context.snapshot_id == snapshot_id
        and answer.method == context.method.value == binding.method
        and answer.answer_variant == "primary"
        and answer.terminal_state in {"answered", "abstained"}
        and call.model_call_id == answer.answer_model_call_id
        and call.actual_model_id == answer.actual_model_id
        and call.context_assembly_id == context.context_assembly_id
        and call.conversation_id == context.conversation_id == query.conversation_id
        and call.stage is AppStage.ANSWER
        and call.status is TraceStatus.SUCCESS,
        "baseline_answer_context_scope_mismatch",
    )
    _require(
        query.available_after_session in facts.visible_sessions
        and all(
            memory.dataset == query.dataset
            and memory.conversation_id == query.conversation_id
            and memory.session_id in facts.visible_sessions
            and memory.available_after_session in facts.visible_sessions
            for memory in memories.values()
        ),
        "baseline_visible_memory_scope_mismatch",
    )
    metadata = call.request_config.get("request_metadata", {})
    _require(
        metadata.get("answer_contract_version") == "typed-answer-contract-v3"
        and metadata.get("model_role") in {"answer", "answer_format_retry"}
        and metadata.get("query_content_hash")
        == canonical_content_hash(
            [
                query.raw_query,
                [
                    i.model_dump(mode="json", exclude={"metadata", "caption"})
                    for i in query.query_image_refs
                ],
            ]
        ),
        "baseline_answer_request_scope_mismatch",
    )
    typed = TypedAnswerContext.model_validate(metadata.get("typed_answer_context"))
    _require(
        [source.evidence_id for source in typed.sources] == context.included_memory_ids
        and set(context.included_memory_ids + context.truncated_memory_ids) <= memories.keys(),
        "baseline_context_membership_mismatch",
    )
    for source in typed.sources:
        _require(
            {image.image_id for image in source.images} <= _images(memories[source.evidence_id]),
            "baseline_context_image_owner_mismatch",
        )
        if context.representation is RepresentationKind.TEXT_CAPTION:
            _require(
                all(image.image is None for image in source.images),
                "baseline_caption_context_contains_pixels",
            )
    return typed


def _retrieval_candidates(
    facts: _Facts,
    query: QueryInput,
    snapshot_id: str,
    binding: ProcessCellBinding,
    memories: dict[str, MemoryUnit],
    resolver: EvidenceLineageResolver,
) -> list[RankedEvidenceCandidate]:
    batch, context = facts.batch, facts.context
    _require(batch is not None, "baseline_retrieval_batch_missing")
    assert batch is not None
    _require(
        batch.retrieval_batch_id == context.retrieval_batch_id
        and batch.run_id == binding.source_run_id
        and batch.query_id == query.query_id
        and batch.snapshot_id == snapshot_id
        and batch.method == binding.method
        and batch.status is TraceStatus.SUCCESS
        and bool(batch.corpus_id)
        and batch.returned_count == len(facts.hits)
        and [hit.rank for hit in facts.hits] == list(range(1, batch.returned_count + 1))
        and len({hit.retrieval_trace_id for hit in facts.hits}) == len(facts.hits),
        "baseline_retrieval_batch_scope_or_count_mismatch",
    )
    result = []
    by_trace = {}
    for hit in facts.hits:
        _require(
            hit.run_id == batch.run_id
            and hit.query_id == batch.query_id
            and hit.method == batch.method
            and hit.retriever_name == batch.retriever_name
            and hit.retrieval_batch_id == batch.retrieval_batch_id,
            "baseline_retrieval_hit_scope_mismatch",
        )
        document = facts.documents.get(hit.retrieval_document_id or "")
        _require(document is not None, "baseline_retrieval_document_missing")
        assert document is not None
        _require(
            document.document_id == hit.retrieval_document_id
            and document.corpus_id == batch.corpus_id
            and document.snapshot_id == snapshot_id
            and document.dataset == query.dataset
            and document.conversation_id == query.conversation_id
            and document.session_id in facts.visible_sessions
            and document.available_after_session in facts.visible_sessions
            and set(document.memory_ids) <= memories.keys()
            and all(memories[mid].session_id == document.session_id for mid in document.memory_ids),
            "baseline_retrieval_document_scope_mismatch",
        )
        if hit.retrieval_window_id is not None:
            window = facts.windows.get(hit.retrieval_window_id)
            _require(window is not None, "baseline_retrieval_window_missing")
            assert window is not None
            _require(
                window.window_id == document.parent_window_id == hit.retrieved_id
                and window.corpus_id == batch.corpus_id
                and window.snapshot_id == snapshot_id
                and window.dataset == query.dataset
                and window.conversation_id == query.conversation_id
                and window.session_id == document.session_id
                and window.session_id in facts.visible_sessions
                and window.available_after_session in facts.visible_sessions
                and set(window.memory_ids) <= memories.keys()
                and set(document.memory_ids) <= set(window.memory_ids)
                and all(memories[mid].session_id == window.session_id for mid in window.memory_ids)
                and set(window.image_ids)
                == {image for mid in window.memory_ids for image in _images(memories[mid])},
                "baseline_retrieval_window_scope_mismatch",
            )
            mids, images = window.memory_ids, window.image_ids
        else:
            _require(hit.retrieved_id == document.document_id, "baseline_document_hit_mismatch")
            # An image document's parent window may contain unrelated source rounds.
            mids = (
                [] if document.representation is RepresentationKind.IMAGE else document.memory_ids
            )
            images = document.image_ids
        _require(
            set(document.image_ids)
            <= {image for mid in document.memory_ids for image in _images(memories[mid])},
            "baseline_document_image_owner_mismatch",
        )
        result.append(
            _candidate(hit.retrieval_trace_id, hit.rank, mids, images, memories, resolver)
        )
        by_trace[hit.retrieval_trace_id] = (hit, document)
    top_k = context.metadata.get("answer_top_k")
    _require(type(top_k) is int and top_k > 0, "baseline_context_retrieval_scope_unavailable")
    assert isinstance(top_k, int)
    _require(
        [c.retrieval_trace_id for c in context.retrieval_candidates]
        == [h.retrieval_trace_id for h in facts.hits[:top_k]],
        "baseline_context_retrieval_prefix_mismatch",
    )
    for item in context.retrieval_candidates:
        hit, document = by_trace[item.retrieval_trace_id]
        members = (
            facts.windows[hit.retrieval_window_id].memory_ids
            if hit.retrieval_window_id
            else document.memory_ids
        )
        _require(
            item.rank == hit.rank
            and item.window_id == document.parent_window_id
            and set(item.contributed_memory_ids) <= set(members),
            "baseline_context_retrieval_membership_mismatch",
        )
    return result


def _score(
    facts: _Facts,
    *,
    query: QueryInput,
    snapshot_id: str,
    binding: ProcessCellBinding,
    gold: QueryGold | None,
    memories: list[MemoryUnit],
) -> dict[str, ProcessMetric]:
    method = BaselineMethod(binding.method)
    _require(method in _FULL | _RAG, "baseline_method_unsupported")
    memory_map = {m.memory_id: m for m in memories}
    _require(len(memory_map) == len(memories), "baseline_visible_memory_duplicates")
    resolver = EvidenceLineageResolver(memories)
    typed = _bound_context(facts, query, snapshot_id, binding, memory_map)
    if method in _FULL:
        _require(
            facts.batch is None and facts.context.retrieval_batch_id is None and not facts.hits,
            "baseline_full_context_has_retrieval_batch",
        )
        candidates = [
            _candidate(
                source.evidence_id,
                rank,
                [source.evidence_id],
                [image.image_id for image in source.images],
                memory_map,
                resolver,
            )
            for rank, source in enumerate(typed.sources, 1)
        ]
    else:
        candidates = _retrieval_candidates(facts, query, snapshot_id, binding, memory_map, resolver)
    if gold is None:
        return _missing("gold_evidence_unavailable", "not_evaluated")
    groups = resolve_gold_evidence(
        evaluation_run_id=binding.source_run_id,
        snapshot_id=snapshot_id,
        query_id=query.query_id,
        dataset=query.dataset,
        gold=gold,
        memories=memories,
        policy_version="evidence-resolution-v2",
    )
    if not groups:
        return _missing("gold_evidence_unavailable", "not_evaluated")
    matched = set().union(*(matched_group_ids(c, groups, resolver) for c in candidates))
    images = set().union(*(matched_image_group_ids(c, groups) for c in candidates))

    def ratio(n: int, d: int = 1, kind: Literal["query", "gold_groups"] = "query") -> ProcessMetric:
        return ProcessMetric(
            status="evaluated", value=n / d, numerator=n, denominator=d, denominator_kind=kind
        )

    has_images = any(
        m.member_type is EvidenceMemberType.IMAGE
        and m.match_role is EvidenceMatchRole.DISCRIMINATIVE
        for group in groups
        for m in group.members
    )
    return {
        "retrieved_evidence_recall": ratio(len(matched), len(groups), "gold_groups"),
        "retrieved_group_complete": ratio(int(len(matched) == len(groups))),
        "retrieved_image_hit": ratio(int(bool(images)))
        if has_images
        else ProcessMetric(
            status="not_applicable",
            reason="gold_image_evidence_unavailable",
            denominator=0,
        ),
    }


async def evaluate_baseline_retrieval_metrics(
    sessions: async_sessionmaker[AsyncSession],
    *,
    query: QueryInput,
    snapshot_id: str,
    binding: ProcessCellBinding,
    gold: QueryGold | None,
    memories: list[MemoryUnit],
) -> dict[str, ProcessMetric]:
    """Use evaluator credentials; no runtime writes, models, or expected-candidate evidence."""
    group_count = None
    try:
        if gold is not None:
            group_count = (
                len(
                    resolve_gold_evidence(
                        evaluation_run_id=binding.source_run_id,
                        snapshot_id=snapshot_id,
                        query_id=query.query_id,
                        dataset=query.dataset,
                        gold=gold,
                        memories=memories,
                        policy_version="evidence-resolution-v2",
                    )
                )
                or None
            )
        async with sessions() as session:
            await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
            await session.execute(text("SET TRANSACTION READ ONLY"))
            await session.execute(text("SET LOCAL statement_timeout = '15000ms'"))
            facts = await _load(session, query=query, snapshot_id=snapshot_id, binding=binding)
        return _score(
            facts,
            query=query,
            snapshot_id=snapshot_id,
            binding=binding,
            gold=gold,
            memories=memories,
        )
    except _Unavailable as exc:
        return _missing(str(exc), gold_group_count=group_count)
    except (ValueError, KeyError, TypeError, AttributeError):
        return _missing("baseline_process_facts_invalid", gold_group_count=group_count)
    except SQLAlchemyError:
        return _missing("baseline_process_database_unavailable", gold_group_count=group_count)
