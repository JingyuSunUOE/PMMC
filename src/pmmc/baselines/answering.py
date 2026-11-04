"""Shared answer input and execution boundary for baseline methods."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol

from pmmc.baselines.config import FullContextBaselineConfig, prompt_sha256
from pmmc.baselines.context import ContextBudgeter
from pmmc.benchmarks.ids import stable_hash
from pmmc.models.execution import ModelCallContext, ModelExecutionService
from pmmc.models.images import validate_paper_image_reference
from pmmc.models.protocols import GenerationGateway
from pmmc.runtime.answering import (
    InvalidRuntimeAnswer,
    SharedAnswerGenerationService,
    recover_runtime_answer,
)
from pmmc.runtime.grounding import GroundingCallback
from pmmc.schemas import (
    AnswerGenerationInput,
    AppStage,
    BaselineMethod,
    ExperimentJob,
    FinalAnswerRecord,
    MemoryUnit,
    ModelCallTrace,
    PreparedAnswerInput,
    QueryInput,
    SessionRecord,
)
from pmmc.schemas.answer_output import (
    TYPED_ANSWER_CONTRACT_VERSION,
    TypedAnswerCandidate,
    TypedAnswerContext,
    TypedAnswerResult,
)


class EmptyModelAnswer(RuntimeError):
    """The provider returned no usable answer text."""


class AnswerInputProvider(Protocol):
    async def prepare(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        query: QueryInput,
        method: BaselineMethod,
        job_id: str | None = None,
        created_at: datetime | None = None,
    ) -> PreparedAnswerInput: ...


class _MemoryReader(Protocol):
    def iter_visible(
        self,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> AsyncIterator[MemoryUnit]: ...


class _SessionReader(Protocol):
    def iter_visible(
        self,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> AsyncIterator[SessionRecord]: ...


class FullContextAnswerInputProvider:
    """Build one exact final request from prospective canonical memory."""

    def __init__(
        self,
        *,
        memories: _MemoryReader,
        sessions: _SessionReader,
        budgeter: ContextBudgeter,
        config: FullContextBaselineConfig,
        instruction: str,
        timeout_seconds: float | None = None,
        paper_mode: bool = False,
        runtime_bucket: str | None = None,
        answer_contract_version: str | None = None,
    ) -> None:
        self.memories = memories
        self.sessions = sessions
        self.budgeter = budgeter
        self.config = config
        self.instruction = instruction
        self.prompt_hash = prompt_sha256(instruction)
        self.timeout_seconds = timeout_seconds
        self.paper_mode = paper_mode
        self.runtime_bucket = runtime_bucket
        self.answer_contract_version = answer_contract_version or config.answer_contract_version
        if paper_mode and not runtime_bucket:
            raise ValueError("paper-mode image preflight requires the runtime bucket")

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
        if method not in {
            BaselineMethod.FULL_CONTEXT_CAPTIONS,
            BaselineMethod.FULL_CONTEXT_RAW_IMAGES,
        }:
            raise ValueError("full-context provider received a non-full-context method")
        if query.available_after_session is None:
            raise ValueError("benchmark query requires an available_after_session cutoff")
        visible_memories = [
            item
            async for item in self.memories.iter_visible(
                snapshot_id,
                query.conversation_id,
                query.available_after_session,
            )
        ]
        visible_sessions = [
            item
            async for item in self.sessions.iter_visible(
                snapshot_id,
                query.conversation_id,
                query.available_after_session,
            )
        ]
        assembled = self.budgeter.assemble(
            run_id=run_id,
            snapshot_id=snapshot_id,
            instruction=self.instruction,
            query=query,
            memories=visible_memories,
            sessions=visible_sessions,
            method=method,
            job_id=job_id,
            answer_model_alias=self.config.answer_model_alias,
            prompt_hash=self.prompt_hash,
            renderer_version=self.config.renderer_version,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            seed=self.config.seed,
            timeout_seconds=self.timeout_seconds,
            created_at=created_at,
            answer_contract_version=self.answer_contract_version,
        )
        if self.paper_mode:
            assert self.runtime_bucket is not None
            for message in assembled.request.messages:
                for part in message.parts:
                    if part.image is not None:
                        validate_paper_image_reference(
                            part.image,
                            runtime_bucket=self.runtime_bucket,
                        )
        request_image_ids = [
            part.image.image_id
            for message in assembled.request.messages
            for part in message.parts
            if part.image is not None
        ]
        return PreparedAnswerInput(
            method=method,
            query=query,
            request=assembled.request,
            context_assembly=assembled.trace,
            evidence_ids_in_prompt=assembled.trace.included_memory_ids,
            image_ids_in_prompt=request_image_ids,
            missing_caption_memory_count=int(
                assembled.trace.metadata.get("missing_caption_memory_count", 0)
            ),
            metadata=(dict(assembled.request.metadata)
                      if self.answer_contract_version == TYPED_ANSWER_CONTRACT_VERSION else {}),
        )


def baseline_job_id(run_id: str, query_id: str, method: BaselineMethod) -> str:
    return "job-" + stable_hash([run_id, query_id, method.value, "primary"])[:28]


def baseline_answer_id(run_id: str, query_id: str, method: BaselineMethod) -> str:
    return "answer-" + stable_hash([run_id, query_id, method.value, "primary"])[:28]


def baseline_model_call_id(job: ExperimentJob) -> str:
    return "call-" + stable_hash([job.job_id, job.attempt_count])[:28]


class BaselineAnswerService:
    """Execute one audited answer call and construct a gold-free final answer fact."""

    def __init__(
        self,
        execution: ModelExecutionService,
        gateway: GenerationGateway | None,
        grounding: GroundingCallback | None = None,
    ) -> None:
        self.execution = execution
        self.gateway = gateway
        self.grounding = grounding

    async def answer(
        self,
        prepared: PreparedAnswerInput,
        job: ExperimentJob,
        recovered_trace: ModelCallTrace | None = None,
        phase: str = "answer",
    ) -> FinalAnswerRecord | TypedAnswerResult | TypedAnswerCandidate:
        if phase not in {"answer", "generate", "ground"}:
            raise ValueError("unknown baseline answer phase")
        contract = prepared.metadata.get("answer_contract_version")
        if phase != "answer" and contract != TYPED_ANSWER_CONTRACT_VERSION:
            raise ValueError("split generation/grounding requires typed v3")
        if phase == "ground" and recovered_trace is None:
            raise ValueError("ground phase requires a successful generated answer journal")
        if contract == TYPED_ANSWER_CONTRACT_VERSION and (
            prepared.method in {
                BaselineMethod.FULL_CONTEXT_CAPTIONS, BaselineMethod.CAPTION_TEXT_RAG,
            }
            or prepared.metadata.get("memory_raw_image_policy") == "forbid"
        ):
            context = TypedAnswerContext.model_validate(prepared.metadata["typed_answer_context"])
            query_images = {i.image_id for i in prepared.query.query_image_refs}
            if (
                any(
                    image.image is not None
                    for source in context.sources for image in source.images
                )
                or not set(prepared.image_ids_in_prompt) <= query_images
            ):
                raise ValueError("caption-only answers cannot inspect raw memory images")
        if contract in {"strict-answer-contract-v2", TYPED_ANSWER_CONTRACT_VERSION}:
            runtime_input = AnswerGenerationInput(
                    method_id=prepared.method.value,
                    answer_variant="primary",
                    run_id=job.run_id,
                    snapshot_id=job.snapshot_id,
                    query=prepared.query,
                    request=prepared.request,
                    evidence_ids_in_prompt=prepared.evidence_ids_in_prompt,
                    image_ids_in_prompt=prepared.image_ids_in_prompt,
                    source_lineage={
                        "context_assembly_id": (
                            prepared.context_assembly.context_assembly_id
                        ),
                        "baseline_method": prepared.method.value,
                    },
                    metadata={
                        "answer_contract_version": contract,
                        "evidence_aliases": list(
                            prepared.metadata.get("evidence_aliases") or []
                        ),
                        **(prepared.metadata if contract == TYPED_ANSWER_CONTRACT_VERSION else {}),
                    },
            )
            if contract == TYPED_ANSWER_CONTRACT_VERSION:
                return await SharedAnswerGenerationService(
                    execution=self.execution, gateway=self.gateway, grounding=self.grounding,
                ).answer_typed(
                    runtime_input, model_call_id=(
                        recovered_trace.model_call_id
                        if recovered_trace else baseline_model_call_id(job)
                    ),
                    job_id=job.job_id,
                    context_assembly_id=prepared.context_assembly.context_assembly_id,
                    recovered_trace=recovered_trace,
                    defer_grounding=phase == "generate",
                    allow_generation=phase != "ground",
                )
            generated: FinalAnswerRecord | TypedAnswerResult
            if recovered_trace is not None:
                try:
                    generated = recover_runtime_answer(runtime_input, recovered_trace)
                except InvalidRuntimeAnswer as exc:
                    if recovered_trace.model_call_id.startswith("answer-format-retry-model-call"):
                        raise
                    generated = await SharedAnswerGenerationService(
                        execution=self.execution, gateway=self.gateway,
                    ).answer(
                        runtime_input, model_call_id=recovered_trace.model_call_id,
                        job_id=job.job_id,
                        context_assembly_id=prepared.context_assembly.context_assembly_id,
                        start_attempt=1, prior_attempt_failures=[exc.failures],
                        prior_schema_diagnostics=[exc.schema_diagnostics],
                    )
            else:
                generated = await SharedAnswerGenerationService(
                    execution=self.execution, gateway=self.gateway,
                ).answer(
                    runtime_input, model_call_id=baseline_model_call_id(job),
                    job_id=job.job_id,
                    context_assembly_id=prepared.context_assembly.context_assembly_id,
                )
            if not isinstance(generated, FinalAnswerRecord):
                raise ValueError("legacy baseline answer returned a typed terminal")
            return generated.model_copy(
                update={
                    "answer_id": baseline_answer_id(
                        job.run_id,
                        prepared.query.query_id,
                        prepared.method,
                    ),
                    "metadata": {
                        **generated.metadata,
                        "context_assembly_id": (
                            prepared.context_assembly.context_assembly_id
                        ),
                        "missing_caption_memory_count": (
                            prepared.missing_caption_memory_count
                        ),
                    },
                }
            )
        if self.gateway is None:
            raise ValueError("legacy baseline generation requires a gateway")
        audited = await self.execution.generate_audited(
            self.gateway,
            prepared.request,
            ModelCallContext(
                model_call_id=baseline_model_call_id(job),
                run_id=job.run_id,
                stage=AppStage.ANSWER,
                snapshot_id=job.snapshot_id,
                query_id=prepared.query.query_id,
                conversation_id=prepared.query.conversation_id,
                job_id=job.job_id,
                context_assembly_id=prepared.context_assembly.context_assembly_id,
            ),
        )
        text = audited.result.text.strip()
        if not text:
            raise EmptyModelAnswer("answer model returned empty text")
        return FinalAnswerRecord(
            answer_id=baseline_answer_id(
                job.run_id,
                prepared.query.query_id,
                prepared.method,
            ),
            snapshot_id=job.snapshot_id,
            query_id=prepared.query.query_id,
            run_id=job.run_id,
            answer_variant="primary",
            method=prepared.method.value,
            answer_model_alias=prepared.request.model_alias,
            actual_model_id=audited.result.actual_model_id,
            evidence_ids_in_prompt=prepared.evidence_ids_in_prompt,
            image_ids_in_prompt=prepared.image_ids_in_prompt,
            final_answer=text,
            input_tokens=audited.result.usage.input_tokens,
            output_tokens=audited.result.usage.output_tokens,
            latency_ms=audited.trace.latency_ms,
            model_call_id=audited.trace.model_call_id,
            langfuse_trace_id=audited.trace.langfuse_trace_id,
            metadata={
                "context_assembly_id": prepared.context_assembly.context_assembly_id,
                "finish_reason": audited.result.finish_reason,
                "missing_caption_memory_count": prepared.missing_caption_memory_count,
            },
        )


def recover_answer_from_trace(
    prepared: PreparedAnswerInput,
    job: ExperimentJob,
    parsed_output: dict[str, object] | None,
    *,
    actual_model_id: str | None,
    model_call_id: str,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_ms: float | None,
    langfuse_trace_id: str | None,
) -> FinalAnswerRecord:
    """Recover a final answer after provider success and before answer commit."""

    text = str((parsed_output or {}).get("text") or "").strip()
    if not text:
        raise EmptyModelAnswer("successful model call has no recoverable parsed answer")
    return FinalAnswerRecord(
        answer_id=baseline_answer_id(job.run_id, prepared.query.query_id, prepared.method),
        snapshot_id=job.snapshot_id,
        query_id=prepared.query.query_id,
        run_id=job.run_id,
        answer_variant="primary",
        method=prepared.method.value,
        answer_model_alias=prepared.request.model_alias,
        actual_model_id=actual_model_id,
        evidence_ids_in_prompt=prepared.evidence_ids_in_prompt,
        image_ids_in_prompt=prepared.image_ids_in_prompt,
        final_answer=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model_call_id=model_call_id,
        langfuse_trace_id=langfuse_trace_id,
        metadata={
            "context_assembly_id": prepared.context_assembly.context_assembly_id,
            "recovered_from_model_call": True,
            "finish_reason": (parsed_output or {}).get("finish_reason"),
            "missing_caption_memory_count": prepared.missing_caption_memory_count,
        },
    )
