"""Evidence-only, exact-budget answer input assembly for the online executor."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.memory.operators import ProgramExecutionResult
from pmmc.models.images import validate_paper_image_reference
from pmmc.models.protocols import TokenCounter
from pmmc.models.provider_token_budget import (
    ProviderTokenBudgetCounter,
    prepare_context_request,
    validate_context_budget,
)
from pmmc.runtime.answering import answer_format_retry_request, prepare_typed_answer_request
from pmmc.runtime.grounding import load_step8_prompt
from pmmc.schemas import (
    AnswerGenerationInput,
    GenerationRequest,
    ImageRef,
    MemoryUnit,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    ProgramExecutionEvidenceRef,
    QueryAnswerVariant,
    QueryInput,
    SpeakerRole,
    ThinkingMode,
)
from pmmc.schemas.answer_output import (
    ANSWER_GUIDED_SCHEMA_VERSION,
    TYPED_ANSWER_CONTRACT_VERSION,
    AnswerEvidenceSource,
    AnswerSourceImage,
    answer_response_schema,
)


class RuntimeMemoryReader(Protocol):
    def iter_visible(
        self,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> AsyncIterator[MemoryUnit]: ...


class RuntimeImageReader(Protocol):
    async def get_many(self, snapshot_id: str, image_ids: list[str]) -> list[ImageRef]: ...


@dataclass(frozen=True)
class _EvidenceItem:
    reference: ProgramExecutionEvidenceRef
    text_lines: tuple[str, ...]
    images: tuple[ImageRef, ...]


def _reference_sort_key(reference: ProgramExecutionEvidenceRef) -> tuple[object, ...]:
    score = reference.score
    normalized = score.normalized_value if score is not None else None
    raw = score.raw_value if score is not None else 0.0
    if score is not None and score.direction.value == "lower_is_better":
        raw = -raw
    return (
        reference.step_id,
        reference.rank,
        -(normalized if normalized is not None else raw),
        reference.evidence_type.value,
        reference.evidence_id,
    )


class PmmcEvidenceAnswerInputProvider:
    """Render only evidence actually emitted by the frozen program execution."""

    def __init__(
        self,
        *,
        memories: RuntimeMemoryReader,
        images: RuntimeImageReader,
        counter: TokenCounter,
        instruction: str,
        answer_model_alias: str = "answer-qwen3.5-9b",
        total_context_tokens: int = 32768,
        reserved_output_tokens: int = 4096,
        timeout_seconds: float | None = None,
        paper_mode: bool = False,
        runtime_bucket: str | None = None,
        answer_prompt_version: str = "pmmc-answer-with-evidence-v1",
        answer_contract_version: str | None = None,
        allow_memory_raw_images: bool = True,
    ) -> None:
        if answer_contract_version == TYPED_ANSWER_CONTRACT_VERSION:
            reserved_output_tokens = 512
            instruction = load_step8_prompt("pmmc_answer_contract_v3.txt")
            answer_prompt_version = "pmmc-answer-contract-v3"
        if reserved_output_tokens <= 0 or reserved_output_tokens >= total_context_tokens:
            raise ValueError("invalid PMMC answer context budget")
        validate_context_budget(
            counter,
            total_context_tokens=total_context_tokens,
            reserved_output_tokens=reserved_output_tokens,
            paper_mode=paper_mode,
        )
        if paper_mode and not runtime_bucket:
            raise ValueError("paper-mode PMMC answering requires a runtime bucket")
        self.memories = memories
        self.images = images
        self.counter = counter
        self.instruction = instruction
        self.instruction_hash = canonical_content_hash(instruction)
        self.answer_model_alias = answer_model_alias
        self.total_context_tokens = total_context_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.max_input_tokens = total_context_tokens - reserved_output_tokens
        self.timeout_seconds = timeout_seconds
        self.paper_mode = paper_mode
        self.runtime_bucket = runtime_bucket
        self.answer_prompt_version = answer_prompt_version
        self.answer_contract_version = answer_contract_version
        self.allow_memory_raw_images = allow_memory_raw_images

    async def prepare(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        query: QueryInput,
        result: ProgramExecutionResult,
        variant: QueryAnswerVariant,
        source_execution_run_id: str | None = None,
    ) -> AnswerGenerationInput:
        if query.available_after_session is None:
            raise ValueError("query-time answer assembly requires a cutoff session")
        if (
            source_execution_run_id is not None
            and self.answer_contract_version != TYPED_ANSWER_CONTRACT_VERSION
        ):
            raise ValueError("shared answer evidence requires the typed answer contract")
        if (
            result.trace.query_id != query.query_id
            or result.trace.run_id != (source_execution_run_id or run_id)
        ):
            raise ValueError("answer evidence execution does not belong to the query run")

        visible = [
            item
            async for item in self.memories.iter_visible(
                snapshot_id,
                query.conversation_id,
                query.available_after_session,
            )
        ]
        memory_by_id = {item.memory_id: item for item in visible}
        ordered_refs = sorted(result.evidence, key=_reference_sort_key)
        direct_image_ids = list(
            dict.fromkeys(
                item.image_id for item in ordered_refs if item.image_id is not None
            )
        )
        direct_images = {
            item.image_id: item
            for item in await self.images.get_many(snapshot_id, direct_image_ids)
        }
        items = self._materialize_items(
            ordered_refs,
            memory_by_id=memory_by_id,
            image_by_id=direct_images,
            typed_images=self.answer_contract_version == TYPED_ANSWER_CONTRACT_VERSION,
            allow_memory_raw_images=self.allow_memory_raw_images,
        )

        selected: list[_EvidenceItem] = []
        request = self._request(
            run_id=run_id,
            query=query,
            variant=variant,
            items=selected,
        )
        def budget_tokens(value: GenerationRequest) -> int:
            initial = self.counter.count_request(value)
            if self.answer_contract_version not in {
                "strict-answer-contract-v2", TYPED_ANSWER_CONTRACT_VERSION,
            }:
                return initial
            return max(initial, self.counter.count_request(answer_format_retry_request(value)))

        if budget_tokens(request) > self.max_input_tokens:
            raise ValueError("PMMC instruction and query exceed the fixed input budget")
        for item in items:
            candidate = self._request(
                run_id=run_id,
                query=query,
                variant=variant,
                items=[*selected, item],
            )
            if budget_tokens(candidate) > self.max_input_tokens:
                break
            selected.append(item)
            request = candidate
        if not selected and self.answer_contract_version != TYPED_ANSWER_CONTRACT_VERSION:
            raise ValueError("no executed evidence fits the PMMC answer context")

        evidence_ids = [item.reference.evidence_id for item in selected]
        evidence_aliases = [f"A{ordinal:02d}" for ordinal in range(1, len(selected) + 1)]
        image_ids = [
            part.image.image_id
            for message in request.messages
            for part in message.parts
            if part.image is not None
        ]
        if self.paper_mode:
            assert self.runtime_bucket is not None
            for message in request.messages:
                for part in message.parts:
                    if part.image is not None:
                        validate_paper_image_reference(
                            part.image,
                            runtime_bucket=self.runtime_bucket,
                        )
        input_tokens = self.counter.count_request(request)
        if budget_tokens(request) > self.max_input_tokens:
            raise RuntimeError("PMMC answer request drifted beyond its exact token budget")
        if isinstance(self.counter, ProviderTokenBudgetCounter):
            self.counter.validate_request(request)
            if self.answer_contract_version in {
                "strict-answer-contract-v2", TYPED_ANSWER_CONTRACT_VERSION,
            }:
                self.counter.validate_request(answer_format_retry_request(request))
        return AnswerGenerationInput(
            method_id="pmmc",
            answer_variant=variant.value,
            run_id=run_id,
            snapshot_id=snapshot_id,
            query=query,
            request=request,
            evidence_ids_in_prompt=evidence_ids,
            image_ids_in_prompt=image_ids,
            source_lineage={
                "execution_id": result.trace.execution_id,
                "plan_id": result.trace.plan_id,
                "evidence_count": len(evidence_ids),
                "evidence_set_hash": canonical_content_hash(evidence_ids),
                **({"source_execution_run_id": source_execution_run_id}
                   if source_execution_run_id is not None else {}),
            },
            metadata={
                "answer_prompt_version": self.answer_prompt_version,
                "answer_prompt_hash": self.instruction_hash,
                "answer_contract_version": self.answer_contract_version,
                "evidence_aliases": evidence_aliases,
                "input_tokens": input_tokens,
                "max_input_tokens": self.max_input_tokens,
                "truncated_evidence_count": len(items) - len(selected),
                **(request.metadata if self.answer_contract_version == TYPED_ANSWER_CONTRACT_VERSION
                   else {}),
            },
        )

    @staticmethod
    def _materialize_items(
        references: Sequence[ProgramExecutionEvidenceRef],
        *,
        memory_by_id: dict[str, MemoryUnit],
        image_by_id: dict[str, ImageRef],
        typed_images: bool = False,
        allow_memory_raw_images: bool = True,
    ) -> list[_EvidenceItem]:
        result: list[_EvidenceItem] = []
        seen: set[str] = set()
        for reference in references:
            if reference.evidence_id in seen:
                continue
            seen.add(reference.evidence_id)
            text_lines: list[str] = []
            images: list[ImageRef] = []
            if reference.memory_id is not None:
                memory = memory_by_id.get(reference.memory_id)
                if memory is None:
                    raise ValueError("program returned memory outside the visible query scope")
                if memory.text:
                    text_lines.append(f"[{memory.speaker.value.upper()}] {memory.text}")
                if memory.caption:
                    text_lines.append(f"[IMAGE CAPTION] {memory.caption}")
                if typed_images:
                    images.extend(image.model_copy(update={
                        "caption": image.caption or (memory.caption
                                                     if len(memory.image_refs) == 1 else None),
                    }) for image in memory.image_refs)
            elif reference.image_id is not None:
                image = image_by_id.get(reference.image_id)
                if image is None:
                    raise ValueError("program returned an unmaterializable image")
                images.append(image)
                if image.caption:
                    text_lines.append(f"[IMAGE CAPTION] {image.caption}")
            elif reference.session_id is not None:
                members = [
                    item
                    for item in memory_by_id.values()
                    if item.session_id == reference.session_id
                ]
                for memory in sorted(
                    members, key=lambda item: (item.timestamp_order, item.memory_id)
                ):
                    if memory.text:
                        text_lines.append(
                            f"[{memory.speaker.value.upper()}] {memory.text}"
                        )
                    if memory.caption:
                        text_lines.append(f"[IMAGE CAPTION] {memory.caption}")
                    if typed_images:
                        images.extend(image.model_copy(update={
                            "caption": image.caption or (memory.caption
                                                         if len(memory.image_refs) == 1 else None),
                        }) for image in memory.image_refs)
            elif reference.caption_id is not None:
                owner = reference.metadata.get("memory_id")
                memory = memory_by_id.get(owner) if isinstance(owner, str) else None
                if memory is None or not memory.caption:
                    raise ValueError("caption evidence lacks a visible owning memory")
                text_lines.append(f"[IMAGE CAPTION] {memory.caption}")
            if not allow_memory_raw_images:
                images = [image for image in images if image.caption]
                if not text_lines and not images:
                    continue
            if not text_lines and not images:
                raise ValueError("program evidence has no answer-model materialization")
            result.append(
                _EvidenceItem(
                    reference=reference,
                    text_lines=tuple(text_lines),
                    images=tuple(images),
                )
            )
        return result

    def _request(
        self,
        *,
        run_id: str,
        query: QueryInput,
        variant: QueryAnswerVariant,
        items: Sequence[_EvidenceItem],
    ) -> GenerationRequest:
        parts = [
            ModelContentPart(
                kind=ModelContentKind.TEXT,
                text="[RETRIEVED EVIDENCE]",
            )
        ]
        used_images: set[str] = set()
        for ordinal, item in enumerate(items, start=1):
            label = f"A{ordinal:02d}"
            for line in item.text_lines:
                parts.append(
                    ModelContentPart(
                        kind=ModelContentKind.TEXT,
                        text=f'<EVIDENCE alias="{label}">{line}</EVIDENCE>',
                    )
                )
            for image in item.images:
                if not self.allow_memory_raw_images:
                    continue
                if image.image_id in used_images:
                    continue
                used_images.add(image.image_id)
                parts.append(
                    ModelContentPart(
                        kind=ModelContentKind.TEXT,
                        text=f'<EVIDENCE alias="{label}">Visual evidence</EVIDENCE>',
                    )
                )
                parts.append(ModelContentPart(kind=ModelContentKind.IMAGE, image=image))
        parts.extend(
            (
                ModelContentPart(
                    kind=ModelContentKind.TEXT,
                    text="[/RETRIEVED EVIDENCE]",
                ),
                ModelContentPart(
                    kind=ModelContentKind.TEXT,
                    text=f"[USER QUERY]\n{query.raw_query}",
                ),
            )
        )
        for image in query.query_image_refs:
            if image.image_id in used_images:
                continue
            used_images.add(image.image_id)
            parts.append(ModelContentPart(kind=ModelContentKind.IMAGE, image=image))
        identity = {
            "run_id": run_id,
            "query_id": query.query_id,
            "variant": variant.value,
            "prompt_hash": self.instruction_hash,
            "evidence": [item.reference.evidence_id for item in items],
            **({"answer_guided_schema": ANSWER_GUIDED_SCHEMA_VERSION}
               if self.answer_contract_version == "strict-answer-contract-v2" else {}),
        }
        request = GenerationRequest(
            request_id=stable_compiler_id("pmmc-answer-request", identity),
            model_alias=self.answer_model_alias,
            messages=[
                ModelMessage(
                    role=SpeakerRole.SYSTEM,
                    parts=[
                        ModelContentPart(
                            kind=ModelContentKind.TEXT,
                            text=self.instruction,
                        )
                    ],
                ),
                ModelMessage(role=SpeakerRole.USER, parts=parts),
            ],
            max_output_tokens=self.reserved_output_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=0,
            timeout_seconds=self.timeout_seconds,
            response_format=(
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pmmc_answer_contract_v2",
                        "strict": True,
                        "schema": answer_response_schema([
                            f"A{ordinal:02d}" for ordinal in range(1, len(items) + 1)
                        ]),
                    },
                }
                if self.answer_contract_version == "strict-answer-contract-v2"
                else None
            ),
            thinking_mode=(
                ThinkingMode.DISABLED
                if self.answer_contract_version == "strict-answer-contract-v2"
                else ThinkingMode.PROVIDER_DEFAULT
            ),
            metadata={
                "answer_prompt_version": self.answer_prompt_version,
                "answer_prompt_hash": self.instruction_hash,
                "answer_contract_version": self.answer_contract_version,
                "evidence_count": len(items),
                **({"answer_guided_schema": ANSWER_GUIDED_SCHEMA_VERSION}
                   if self.answer_contract_version == "strict-answer-contract-v2" else {}),
            },
        )
        if self.answer_contract_version == TYPED_ANSWER_CONTRACT_VERSION:
            request = prepare_typed_answer_request(request, query=query, evidence_sources=[
                AnswerEvidenceSource(
                    evidence_alias=f"A{ordinal:02d}", evidence_id=item.reference.evidence_id,
                    text="\n".join(item.text_lines) or None,
                    images=[AnswerSourceImage(
                        image_id=image.image_id,
                        image=image if self.allow_memory_raw_images else None,
                        caption=image.caption if not self.allow_memory_raw_images else None,
                    )
                            for image in {i.image_id: i for i in item.images}.values()],
                ) for ordinal, item in enumerate(items, 1)
            ])
        return prepare_context_request(self.counter, request)


__all__ = ["PmmcEvidenceAnswerInputProvider"]
