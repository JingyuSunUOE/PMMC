"""Rank-aware, whole-window answer context assembly for RAG baselines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from pmmc.baselines.rag_config import RagBaselineConfig, rag_prompt_sha256
from pmmc.baselines.rag_retrieval import retrieval_batch_id
from pmmc.baselines.rendering import (
    RAG_RENDERER_VERSION,
    render_memory_text,
    render_memory_text_caption,
    render_rag_messages,
)
from pmmc.baselines.typed_answer import adapt_baseline_typed_request
from pmmc.benchmarks.ids import stable_hash
from pmmc.models.images import validate_paper_image_reference
from pmmc.models.protocols import TokenCounter
from pmmc.models.provider_token_budget import (
    ProviderTokenBudgetCounter,
    prepare_context_request,
    validate_context_budget,
)
from pmmc.runtime.answering import answer_format_retry_request
from pmmc.runtime.grounding import load_step8_prompt
from pmmc.schemas import (
    BaselineMethod,
    ContextAssemblyTrace,
    ContextCandidateDisposition,
    ContextRetrievalCandidate,
    GenerationRequest,
    MemoryTokenCount,
    MemoryUnit,
    MemoryWindow,
    PreparedAnswerInput,
    QueryInput,
    RepresentationKind,
    RetrievalBatchTrace,
    RetrievalQueryEmbeddingRef,
    RetrievalTrace,
    RetrievalTraceComponent,
    SessionRecord,
    ThinkingMode,
    TruncationPolicy,
)
from pmmc.schemas.answer_output import (
    ANSWER_GUIDED_SCHEMA_VERSION,
    TYPED_ANSWER_CONTRACT_VERSION,
    answer_response_schema,
)
from pmmc.schemas.base import utc_now


class RagRetrievalReader(Protocol):
    async def get_batch(
        self, retrieval_batch_id: str
    ) -> tuple[
        RetrievalBatchTrace,
        list[RetrievalTrace],
        list[RetrievalTraceComponent],
        list[RetrievalQueryEmbeddingRef],
    ] | None: ...


class RagCorpusReader(Protocol):
    async def get_window(self, corpus_id: str, window_id: str) -> MemoryWindow: ...


class RagMemoryReader(Protocol):
    def iter_visible(
        self, snapshot_id: str, conversation_id: str, cutoff_session_id: str
    ) -> AsyncIterator[MemoryUnit]: ...


class RagSessionReader(Protocol):
    def iter_visible(
        self, snapshot_id: str, conversation_id: str, cutoff_session_id: str
    ) -> AsyncIterator[SessionRecord]: ...


@dataclass(frozen=True)
class RankedWindow:
    trace: RetrievalTrace
    window: MemoryWindow
    memories: tuple[MemoryUnit, ...]


def _memory_image_ids(memory: MemoryUnit) -> list[str]:
    return list(
        dict.fromkeys(
            [*memory.image_ids, *(image.image_id for image in memory.image_refs)]
        )
    )


class RagContextBudgeter:
    def __init__(
        self,
        counter: TokenCounter,
        *,
        total_context_tokens: int = 32768,
        reserved_output_tokens: int = 1024,
        paper_mode: bool = False,
    ) -> None:
        if reserved_output_tokens <= 0 or reserved_output_tokens >= total_context_tokens:
            raise ValueError("invalid RAG context token budget")
        validate_context_budget(
            counter,
            total_context_tokens=total_context_tokens,
            reserved_output_tokens=reserved_output_tokens,
            paper_mode=paper_mode,
        )
        self.counter = counter
        self.total_context_tokens = total_context_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.max_input_tokens = total_context_tokens - reserved_output_tokens

    def assemble(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        query: QueryInput,
        method: BaselineMethod,
        ranked_windows: list[RankedWindow],
        sessions: list[SessionRecord],
        instruction: str,
        config: RagBaselineConfig,
        job_id: str | None = None,
        timeout_seconds: float | None = None,
        created_at: datetime | None = None,
        answer_contract_version: str | None = None,
    ) -> PreparedAnswerInput:
        answer_contract_version = answer_contract_version or config.answer_contract_version
        if method not in {
            BaselineMethod.CAPTION_TEXT_RAG,
            BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG,
            BaselineMethod.NATIVE_MULTIMODAL_RAG,
        }:
            raise ValueError("RAG context budgeter received a non-RAG method")
        if [item.trace.rank for item in ranked_windows] != list(
            range(1, len(ranked_windows) + 1)
        ):
            raise ValueError("ranked windows must be contiguous from one")
        if len(ranked_windows) > config.answer_top_k:
            raise ValueError("RAG context cannot inspect more than answer_top_k windows")

        representation = (
            RepresentationKind.TEXT_CAPTION
            if (
                method is BaselineMethod.CAPTION_TEXT_RAG
                or config.memory_raw_image_policy == "forbid"
            )
            else RepresentationKind.MULTIMODAL
        )
        strict_answer = answer_contract_version == "strict-answer-contract-v2"
        typed = answer_contract_version == TYPED_ANSWER_CONTRACT_VERSION
        if typed:
            instruction = load_step8_prompt("pmmc_answer_contract_v3.txt")
        output_budget = 512 if typed else self.reserved_output_tokens
        max_input_tokens = self.total_context_tokens - output_budget

        def request_for(memories: list[MemoryUnit], request_id: str) -> GenerationRequest:
            evidence_aliases = (
                [f"E{ordinal + 1:02d}" for ordinal in range(len(memories))]
                if strict_answer
                else None
            )
            request = GenerationRequest(
                request_id=request_id,
                model_alias=config.answer_model_alias,
                messages=render_rag_messages(
                    instruction=instruction,
                    query=query,
                    memories=memories,
                    method=(BaselineMethod.CAPTION_TEXT_RAG
                            if config.memory_raw_image_policy == "forbid" else method),
                    sessions=sessions,
                    evidence_aliases=evidence_aliases,
                ),
                max_output_tokens=(512 if strict_answer else config.max_output_tokens),
                temperature=config.temperature,
                top_p=config.top_p,
                seed=config.seed,
                timeout_seconds=timeout_seconds,
                response_format=(
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "strict_answer_output",
                            "strict": True,
                            "schema": answer_response_schema(evidence_aliases),
                        },
                    }
                    if strict_answer
                    else None
                ),
                thinking_mode=(
                    ThinkingMode.DISABLED
                    if strict_answer
                    else ThinkingMode.PROVIDER_DEFAULT
                ),
                metadata={
                    "renderer_version": config.renderer_version,
                    "prompt_hash": rag_prompt_sha256(instruction),
                    "baseline_method": method.value,
                    "answer_contract_version": answer_contract_version,
                    "evidence_aliases": evidence_aliases or [],
                    **({"answer_guided_schema": ANSWER_GUIDED_SCHEMA_VERSION}
                       if strict_answer else {}),
                },
            )
            if typed:
                request = adapt_baseline_typed_request(
                    request, query=query, memories=memories, sessions=sessions, method=method,
                    memory_raw_image_policy=config.memory_raw_image_policy,
                )
            return prepare_context_request(self.counter, request)

        def budget_tokens(request: GenerationRequest) -> int:
            initial = self.counter.count_request(request)
            if not (strict_answer or typed):
                return initial
            return max(initial, self.counter.count_request(answer_format_retry_request(request)))

        fixed = budget_tokens(request_for([], "rag-budget"))
        if fixed > max_input_tokens:
            raise ValueError("instruction and query exceed the fixed RAG input budget")

        selected: list[MemoryUnit] = []
        selected_ids: set[str] = set()
        candidates: list[ContextRetrievalCandidate] = []
        stopped = False
        all_unique: list[MemoryUnit] = []
        all_unique_ids: set[str] = set()
        for ranked in ranked_windows:
            ordered = sorted(ranked.memories, key=lambda item: item.timestamp_order)
            if [item.memory_id for item in ordered] != ranked.window.memory_ids:
                raise ValueError("window MemoryUnit order does not match corpus lineage")
            for memory in ordered:
                if memory.memory_id not in all_unique_ids:
                    all_unique.append(memory)
                    all_unique_ids.add(memory.memory_id)
            new_units = [item for item in ordered if item.memory_id not in selected_ids]
            if stopped:
                disposition = ContextCandidateDisposition.BUDGET_EXCLUDED
                contributed: list[str] = []
            elif not new_units:
                disposition = ContextCandidateDisposition.DEDUPLICATED
                contributed = []
            elif budget_tokens(
                request_for([*selected, *new_units], "rag-budget")
            ) > max_input_tokens:
                disposition = ContextCandidateDisposition.BUDGET_EXCLUDED
                contributed = []
                stopped = True
            else:
                disposition = ContextCandidateDisposition.INCLUDED
                contributed = [item.memory_id for item in new_units]
                selected.extend(new_units)
                selected_ids.update(contributed)
            candidates.append(
                ContextRetrievalCandidate(
                    retrieval_trace_id=ranked.trace.retrieval_trace_id,
                    window_id=ranked.window.window_id,
                    rank=ranked.trace.rank,
                    disposition=disposition,
                    contributed_memory_ids=contributed,
                )
            )

        final_request_tokens = self.counter.count_request(
            request_for(selected, "rag-budget")
        )
        instruction_tokens = self.counter.count_text(instruction)
        query_tokens = self.counter.count_text(query.raw_query)

        def count_image(image) -> int:  # type: ignore[no-untyped-def]
            exact = getattr(self.counter, "count_image_ref", None)
            if exact is not None:
                return int(exact(image))
            return self.counter.count_image(width=image.width, height=image.height)

        query_image_tokens = sum(count_image(image) for image in query.query_image_refs)
        counts: list[MemoryTokenCount] = []
        for memory in all_unique:
            text = (
                render_memory_text_caption(memory)
                if representation is RepresentationKind.TEXT_CAPTION
                else render_memory_text(memory)
            )
            image_tokens = (
                0
                if representation is RepresentationKind.TEXT_CAPTION
                else sum(count_image(image) for image in memory.image_refs)
            )
            counts.append(
                MemoryTokenCount(
                    memory_id=memory.memory_id,
                    text_tokens=self.counter.count_text(text),
                    image_tokens=image_tokens,
                )
            )
        selected_counts = [item for item in counts if item.memory_id in selected_ids]
        memory_text_tokens = sum(item.text_tokens for item in selected_counts)
        memory_image_tokens = sum(item.image_tokens for item in selected_counts)
        components = (
            instruction_tokens
            + query_tokens
            + query_image_tokens
            + memory_text_tokens
            + memory_image_tokens
        )
        adjustment = final_request_tokens - components
        included_images = list(
            dict.fromkeys(
                image_id for memory in selected for image_id in _memory_image_ids(memory)
            )
        )
        truncated = [item for item in all_unique if item.memory_id not in selected_ids]
        truncated_images = list(
            dict.fromkeys(
                image_id for memory in truncated for image_id in _memory_image_ids(memory)
            )
        )
        batch_id = retrieval_batch_id(run_id, query.query_id, method)
        context_id = "context-" + stable_hash(
            {
                "run_id": run_id,
                "snapshot_id": snapshot_id,
                "query_id": query.query_id,
                "retrieval_batch_id": batch_id,
                "method": method.value,
                "renderer_version": RAG_RENDERER_VERSION,
                "prompt_hash": rag_prompt_sha256(instruction),
                "included_memory_ids": [item.memory_id for item in selected],
                "candidate_dispositions": [
                    item.model_dump(mode="json") for item in candidates
                ],
                "counter": [self.counter.alias, self.counter.revision],
                **({"answer_guided_schema": ANSWER_GUIDED_SCHEMA_VERSION}
                   if strict_answer else {}),
                **({"answer_contract_version": answer_contract_version} if typed else {}),
            }
        )[:24]
        trace = ContextAssemblyTrace(
            context_assembly_id=context_id,
            run_id=run_id,
            snapshot_id=snapshot_id,
            query_id=query.query_id,
            conversation_id=query.conversation_id,
            job_id=job_id,
            retrieval_batch_id=batch_id,
            method=method,
            representation=representation,
            truncation_policy=TruncationPolicy.RETRIEVAL_RANK_COMPLETE_UNITS_V1,
            tokenizer_alias=self.counter.alias,
            tokenizer_revision=self.counter.revision,
            total_context_tokens=self.total_context_tokens,
            reserved_output_tokens=output_budget,
            max_input_tokens=max_input_tokens,
            instruction_tokens=instruction_tokens,
            query_tokens=query_tokens,
            query_image_tokens=query_image_tokens,
            memory_text_tokens=memory_text_tokens,
            memory_image_tokens=memory_image_tokens,
            request_token_adjustment=adjustment,
            input_tokens=final_request_tokens,
            estimated=self.counter.estimated,
            memory_token_counts=counts,
            included_memory_ids=[item.memory_id for item in selected],
            truncated_memory_ids=[item.memory_id for item in truncated],
            included_image_ids=included_images,
            truncated_image_ids=[
                image_id for image_id in truncated_images if image_id not in set(included_images)
            ],
            query_image_ids=[image.image_id for image in query.query_image_refs],
            retrieval_candidates=candidates,
            created_at=created_at or utc_now(),
            metadata={
                "renderer_version": RAG_RENDERER_VERSION,
                "prompt_hash": rag_prompt_sha256(instruction),
                "answer_top_k": config.answer_top_k,
                "actual_prompt_image_ids": [
                    *(
                        []
                        if representation is RepresentationKind.TEXT_CAPTION
                        else included_images
                    ),
                    *[image.image_id for image in query.query_image_refs],
                ],
            },
        )
        final_request = request_for(
            selected, f"request-{context_id.removeprefix('context-')}"
        )
        if self.counter.count_request(final_request) != final_request_tokens:
            raise RuntimeError("RAG final request drifted from the budgeted request")
        if isinstance(self.counter, ProviderTokenBudgetCounter):
            self.counter.validate_request(final_request)
            if strict_answer or typed:
                self.counter.validate_request(answer_format_retry_request(final_request))
        request_image_ids = [
            part.image.image_id
            for message in final_request.messages
            for part in message.parts
            if part.image is not None
        ]
        return PreparedAnswerInput(
            method=method,
            query=query,
            request=final_request,
            context_assembly=trace,
            evidence_ids_in_prompt=trace.included_memory_ids,
            image_ids_in_prompt=request_image_ids,
            missing_caption_memory_count=sum(
                1
                for memory in selected
                if _memory_image_ids(memory)
                and not memory.caption
                and not any(image.caption for image in memory.image_refs)
            ),
            metadata={
                "answer_contract_version": answer_contract_version,
                "evidence_aliases": [
                    f"E{ordinal + 1:02d}" for ordinal in range(len(selected))
                ]
                if strict_answer
                else [],
                **(final_request.metadata if typed else {}),
            },
        )


class RagAnswerInputProvider:
    def __init__(
        self,
        *,
        retrieval: RagRetrievalReader,
        corpus: RagCorpusReader,
        memories: RagMemoryReader,
        sessions: RagSessionReader,
        budgeter: RagContextBudgeter,
        config: RagBaselineConfig,
        instruction: str,
        timeout_seconds: float | None = None,
        paper_mode: bool = False,
        runtime_bucket: str | None = None,
        answer_contract_version: str | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.corpus = corpus
        self.memories = memories
        self.sessions = sessions
        self.budgeter = budgeter
        self.config = config
        self.instruction = instruction
        self.timeout_seconds = timeout_seconds
        self.paper_mode = paper_mode
        self.runtime_bucket = runtime_bucket
        self.answer_contract_version = answer_contract_version or config.answer_contract_version
        if paper_mode and not runtime_bucket:
            raise ValueError("paper-mode RAG image preflight requires runtime bucket")

    async def prepare(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        query: QueryInput,
        method: BaselineMethod,
        job_id: str | None = None,
        created_at: datetime | None = None,
    ) -> PreparedAnswerInput:
        if query.available_after_session is None:
            raise ValueError("benchmark query requires an available_after_session cutoff")
        batch_id = retrieval_batch_id(run_id, query.query_id, method)
        persisted = await self.retrieval.get_batch(batch_id)
        if persisted is None:
            raise KeyError(f"retrieval batch does not exist: {batch_id}")
        batch, traces, _components, _query_refs = persisted
        if batch.corpus_id is None or batch.method != method.value:
            raise ValueError("retrieval batch does not match RAG answer method")
        traces = traces[: self.config.answer_top_k]
        visible = {
            item.memory_id: item
            async for item in self.memories.iter_visible(
                snapshot_id, query.conversation_id, query.available_after_session
            )
        }
        session_values = [
            item
            async for item in self.sessions.iter_visible(
                snapshot_id, query.conversation_id, query.available_after_session
            )
        ]
        ranked: list[RankedWindow] = []
        for trace in traces:
            if trace.retrieval_window_id is None:
                raise ValueError("RAG answer requires window-level retrieval traces")
            window = await self.corpus.get_window(batch.corpus_id, trace.retrieval_window_id)
            try:
                window_memories = tuple(visible[memory_id] for memory_id in window.memory_ids)
            except KeyError as exc:
                raise ValueError("retrieval window crosses the prospective query cutoff") from exc
            ranked.append(RankedWindow(trace=trace, window=window, memories=window_memories))
        prepared = self.budgeter.assemble(
            run_id=run_id,
            snapshot_id=snapshot_id,
            query=query,
            method=method,
            ranked_windows=ranked,
            sessions=session_values,
            instruction=self.instruction,
            config=self.config,
            job_id=job_id,
            timeout_seconds=self.timeout_seconds,
            created_at=created_at,
            answer_contract_version=self.answer_contract_version,
        )
        if self.paper_mode:
            assert self.runtime_bucket is not None
            for message in prepared.request.messages:
                for part in message.parts:
                    if part.image is not None:
                        validate_paper_image_reference(
                            part.image, runtime_bucket=self.runtime_bucket
                        )
        return prepared
