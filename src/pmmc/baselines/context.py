"""Shared 32k full-context budget and truncation implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pmmc.baselines.rendering import (
    FULL_CONTEXT_RENDERER_VERSION,
    render_full_context_messages,
    render_memory_text,
    render_memory_text_caption,
)
from pmmc.baselines.typed_answer import adapt_baseline_typed_request
from pmmc.benchmarks.ids import stable_hash
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
    GenerationRequest,
    MemoryTokenCount,
    MemoryUnit,
    QueryInput,
    RepresentationKind,
    SessionRecord,
)
from pmmc.schemas.answer_output import TYPED_ANSWER_CONTRACT_VERSION
from pmmc.schemas.base import utc_now


@dataclass(frozen=True)
class AssembledContext:
    """Selected whole units plus an immutable audit trace."""

    instruction: str
    query: QueryInput
    memories: tuple[MemoryUnit, ...]
    request: GenerationRequest
    trace: ContextAssemblyTrace


class ContextBudgeter:
    """Keep the most recent contiguous history that fits a fixed context budget."""

    def __init__(
        self,
        counter: TokenCounter,
        *,
        total_context_tokens: int = 32768,
        reserved_output_tokens: int = 1024,
        paper_mode: bool = False,
    ) -> None:
        if total_context_tokens <= 0 or reserved_output_tokens <= 0:
            raise ValueError("context and output token budgets must be positive")
        if reserved_output_tokens >= total_context_tokens:
            raise ValueError("reserved output tokens must be smaller than total context")
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
        instruction: str,
        query: QueryInput,
        memories: list[MemoryUnit],
        method: BaselineMethod,
        job_id: str | None = None,
        sessions: list[SessionRecord] | None = None,
        answer_model_alias: str | None = None,
        prompt_hash: str | None = None,
        renderer_version: str = FULL_CONTEXT_RENDERER_VERSION,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = 0,
        timeout_seconds: float | None = None,
        created_at: datetime | None = None,
        answer_contract_version: str | None = None,
    ) -> AssembledContext:
        if method is BaselineMethod.FULL_CONTEXT_CAPTIONS:
            representation = RepresentationKind.TEXT_CAPTION
        elif method is BaselineMethod.FULL_CONTEXT_RAW_IMAGES:
            representation = RepresentationKind.MULTIMODAL
        else:
            raise ValueError("ContextBudgeter only assembles the two full-context baselines")

        typed = answer_contract_version == TYPED_ANSWER_CONTRACT_VERSION
        if typed:
            instruction = load_step8_prompt("pmmc_answer_contract_v3.txt")
        output_budget = 512 if typed else self.reserved_output_tokens
        max_input_tokens = self.total_context_tokens - output_budget
        instruction_tokens = self.counter.count_text(instruction)
        query_tokens = self.counter.count_text(query.raw_query)

        def count_image(image) -> int:  # type: ignore[no-untyped-def]
            exact = getattr(self.counter, "count_image_ref", None)
            if exact is not None:
                return int(exact(image))
            return self.counter.count_image(width=image.width, height=image.height)

        query_image_tokens = sum(
            count_image(image)
            for image in query.query_image_refs
        )
        model_alias = answer_model_alias or self.counter.alias

        def request_for(
            selected: list[MemoryUnit], request_id: str = "context-budget"
        ) -> GenerationRequest:
            request = GenerationRequest(
                request_id=request_id,
                model_alias=model_alias,
                messages=render_full_context_messages(
                    instruction=instruction,
                    query=query,
                    memories=selected,
                    method=method,
                    sessions=sessions,
                ),
                max_output_tokens=output_budget,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                timeout_seconds=timeout_seconds,
                metadata={
                    "renderer_version": renderer_version,
                    "prompt_hash": prompt_hash,
                    "baseline_method": method.value,
                    **({"answer_contract_version": answer_contract_version} if typed else {}),
                },
            )
            if typed:
                request = adapt_baseline_typed_request(
                    request, query=query, memories=selected, sessions=sessions or [], method=method,
                )
            return prepare_context_request(self.counter, request)

        def budget_tokens(request: GenerationRequest) -> int:
            initial = self.counter.count_request(request)
            if typed:
                return max(
                    initial, self.counter.count_request(answer_format_retry_request(request)),
                )
            return initial

        fixed_request_tokens = budget_tokens(request_for([]))
        if fixed_request_tokens > max_input_tokens:
            raise ValueError("instruction and query exceed the fixed input budget")

        ordered = sorted(memories, key=lambda item: item.timestamp_order)
        if len({memory.memory_id for memory in ordered}) != len(ordered):
            raise ValueError("context memories cannot contain duplicate memory IDs")
        if len({memory.timestamp_order for memory in ordered}) != len(ordered):
            raise ValueError("context memories cannot contain duplicate timestamp orders")
        if any(memory.conversation_id != query.conversation_id for memory in ordered):
            raise ValueError("context memories must belong to the query conversation")

        def image_ids(memory: MemoryUnit) -> list[str]:
            return list(
                dict.fromkeys(
                    [
                        *memory.image_ids,
                        *(image.image_id for image in memory.image_refs),
                    ]
                )
            )

        counts: dict[str, MemoryTokenCount] = {}
        for memory in ordered:
            rendered = (
                render_memory_text_caption(memory)
                if representation is RepresentationKind.TEXT_CAPTION
                else render_memory_text(memory)
            )
            text_tokens = self.counter.count_text(rendered)
            image_tokens = 0
            if representation is RepresentationKind.MULTIMODAL:
                refs = {image.image_id: image for image in memory.image_refs}
                image_tokens = sum(
                    count_image(refs[image_id])
                    for image_id in image_ids(memory)
                    if image_id in refs
                )
            counts[memory.memory_id] = MemoryTokenCount(
                memory_id=memory.memory_id,
                text_tokens=text_tokens,
                image_tokens=image_tokens,
            )

        # Exact multimodal processors are expensive. The keep-recent policy selects
        # one contiguous suffix, so locate its boundary with logarithmic full-request
        # counts instead of reprocessing every increasingly large image sequence.
        token_counts_by_start = {len(ordered): fixed_request_tokens}

        def suffix_tokens(start: int) -> int:
            if start not in token_counts_by_start:
                token_counts_by_start[start] = budget_tokens(
                    request_for(ordered[start:])
                )
            return token_counts_by_start[start]

        low = 0
        high = len(ordered)
        while low < high:
            middle = (low + high) // 2
            if suffix_tokens(middle) <= max_input_tokens:
                high = middle
            else:
                low = middle + 1
        selected_start = low

        # Tokenizers should be monotonic for a prefixed memory sequence. These
        # corrections preserve the exact boundary if a template has a small local
        # non-monotonicity around the binary-search result.
        while suffix_tokens(selected_start) > max_input_tokens:
            selected_start += 1
        while (
            selected_start > 0
            and suffix_tokens(selected_start - 1) <= max_input_tokens
        ):
            selected_start -= 1

        selected = ordered[selected_start:]
        used = (self.counter.count_request(request_for(selected)) if typed
                else suffix_tokens(selected_start))
        included_ids = [memory.memory_id for memory in selected]
        included_set = set(included_ids)
        truncated = [memory for memory in ordered if memory.memory_id not in included_set]
        selected_counts = [counts[memory.memory_id] for memory in selected]
        memory_text_tokens = sum(value.text_tokens for value in selected_counts)
        memory_image_tokens = sum(value.image_tokens for value in selected_counts)
        component_tokens = (
            instruction_tokens
            + query_tokens
            + query_image_tokens
            + memory_text_tokens
            + memory_image_tokens
        )
        request_token_adjustment = used - component_tokens
        included_images = list(
            dict.fromkeys(image for memory in selected for image in image_ids(memory))
        )
        truncated_images = list(
            dict.fromkeys(image for memory in truncated for image in image_ids(memory))
        )
        trace_id = "context-" + stable_hash(
            {
                "run_id": run_id,
                "snapshot_id": snapshot_id,
                "query_id": query.query_id,
                "query_text": query.raw_query,
                "query_images": [
                    {
                        "image_id": image.image_id,
                        "sha256": image.sha256,
                        "width": image.width,
                        "height": image.height,
                    }
                    for image in query.query_image_refs
                ],
                "instruction": instruction,
                "prompt_hash": prompt_hash,
                "renderer_version": renderer_version,
                "answer_model_alias": model_alias,
                "decoding": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                },
                "sessions": [
                    {
                        "session_id": session.session_id,
                        "session_order": session.session_order,
                        "source_date_iso": session.source_date_iso,
                        "is_synthetic": session.is_synthetic,
                    }
                    for session in sessions or []
                ],
                "method": method.value,
                "representation": representation.value,
                "included_memory_ids": included_ids,
                "total_context_tokens": self.total_context_tokens,
                "reserved_output_tokens": output_budget,
                **({"answer_contract_version": answer_contract_version} if typed else {}),
                "counter": [self.counter.alias, self.counter.revision],
                "estimated": self.counter.estimated,
            }
        )[:24]
        trace = ContextAssemblyTrace(
            context_assembly_id=trace_id,
            run_id=run_id,
            snapshot_id=snapshot_id,
            query_id=query.query_id,
            conversation_id=query.conversation_id,
            job_id=job_id,
            method=method,
            representation=representation,
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
            request_token_adjustment=request_token_adjustment,
            input_tokens=used,
            estimated=self.counter.estimated,
            memory_token_counts=[counts[memory.memory_id] for memory in ordered],
            included_memory_ids=included_ids,
            truncated_memory_ids=[memory.memory_id for memory in truncated],
            included_image_ids=included_images,
            truncated_image_ids=[
                image for image in truncated_images if image not in set(included_images)
            ],
            query_image_ids=[image.image_id for image in query.query_image_refs],
            created_at=created_at or utc_now(),
            metadata={
                "renderer_version": renderer_version,
                "prompt_hash": prompt_hash,
                "answer_model_alias": model_alias,
                "actual_prompt_image_ids": [
                    *(
                        included_images
                        if method is BaselineMethod.FULL_CONTEXT_RAW_IMAGES
                        else []
                    ),
                    *[image.image_id for image in query.query_image_refs],
                ],
                "missing_caption_memory_count": sum(
                    1
                    for memory in selected
                    if image_ids(memory)
                    and not memory.caption
                    and not any(image.caption for image in memory.image_refs)
                ),
            },
        )
        final_request = request_for(
            selected,
            request_id=f"request-{trace_id.removeprefix('context-')}",
        )
        if self.counter.count_request(final_request) != used:
            raise RuntimeError("final generation request drifted from the budgeted request")
        if isinstance(self.counter, ProviderTokenBudgetCounter):
            self.counter.validate_request(final_request)
            if typed:
                self.counter.validate_request(answer_format_retry_request(final_request))
        return AssembledContext(
            instruction=instruction,
            query=query,
            memories=tuple(selected),
            request=final_request,
            trace=trace,
        )
