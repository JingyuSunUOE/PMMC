"""Audited model execution with immutable artifacts and persisted call facts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, TypeVar, cast
from uuid import uuid4

from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.models.campaign_audit import current_campaign_audit
from pmmc.models.protocols import (
    EmbeddingGateway,
    GenerationGateway,
    ModelCallWriter,
    TraceSink,
)
from pmmc.models.providers import ProviderCallError
from pmmc.models.registry import ModelRegistry
from pmmc.models.tracing import NoopTraceSink
from pmmc.schemas import (
    AppStage,
    AssetType,
    AuditedEmbeddingResult,
    DataScope,
    EmbeddingRequest,
    EmbeddingResult,
    GenerationRequest,
    GenerationResult,
    ModelCallTrace,
    ModelContentKind,
    ModelEndpointKind,
    TraceStatus,
)

ResultT = TypeVar("ResultT", GenerationResult, EmbeddingResult)
_EVALUATION_ONLY_KEYS = {
    "answer_session_ids",
    "clue",
    "gold",
    "gold_answer",
    "gold_evidence",
    "gold_image_ids",
    "gold_memory_ids",
    "gold_session_ids",
    "is_gold_evidence",
    "matched_gold_evidence_ids",
}


@dataclass(frozen=True)
class ModelCallContext:
    """Stable experiment associations supplied by a runner."""

    model_call_id: str
    run_id: str
    stage: AppStage
    snapshot_id: str | None = None
    query_id: str | None = None
    conversation_id: str | None = None
    job_id: str | None = None
    evaluation_run_id: str | None = None
    evaluation_job_id: str | None = None
    semantic_run_id: str | None = None
    semantic_job_id: str | None = None
    compiler_unit_id: str | None = None
    qbank_id: str | None = None
    plan_id: str | None = None
    step_id: int | None = None
    parent_model_call_id: str | None = None
    context_assembly_id: str | None = None

    def __post_init__(self) -> None:
        if self.query_id is not None and self.snapshot_id is None:
            raise ModelCallContextError(
                "query_snapshot_missing",
                stage=self.stage,
                compiler_scoped=False,
            )
        if self.compiler_unit_id is None:
            return
        if self.snapshot_id is None:
            raise ModelCallContextError(
                "compiler_snapshot_missing",
                stage=self.stage,
                compiler_scoped=True,
            )
        if self.job_id is None:
            raise ModelCallContextError(
                "compiler_job_missing",
                stage=self.stage,
                compiler_scoped=True,
            )
        if self.stage not in {
            AppStage.QUESTIONER,
            AppStage.PLANNER,
            AppStage.DOUBTER,
        }:
            raise ModelCallContextError(
                "compiler_stage_invalid",
                stage=self.stage,
                compiler_scoped=True,
            )


class ModelCallContextError(ValueError):
    """Secret-free lineage failure raised before a provider request starts."""

    diagnostic_event = "MODEL_CALL_CONTEXT_FAILURE"

    def __init__(
        self,
        code: str,
        *,
        stage: AppStage,
        compiler_scoped: bool,
    ) -> None:
        self.code = code
        self.error_code = code
        self.failure_stage = stage.value
        self.safe_summary = f"model-call context validation failed ({code})"
        self._diagnostics = {
            "compiler_scoped": compiler_scoped,
            "error_code": code,
            "stage": stage.value,
        }
        super().__init__(self.safe_summary)

    def safe_diagnostics(self) -> dict[str, object]:
        return dict(self._diagnostics)


@dataclass(frozen=True)
class AuditedGenerationResult:
    """Provider result paired with the authoritative persisted trace."""

    result: GenerationResult
    trace: ModelCallTrace


@dataclass(frozen=True)
class ParsedModelOutput:
    """Structured generation output plus local repair audit metadata."""

    value: dict[str, Any]
    repair_attempt_count: int = 0
    initial_error: str | None = None


class ModelOutputContractError(ValueError):
    """A classified provider-output violation, not a parser implementation failure."""

    def __init__(self, code: str) -> None:
        self.code = self.error_code = code
        super().__init__(code)


GenerationOutputParser = Callable[[GenerationResult], ParsedModelOutput]
GenerationRecovery = Callable[[GenerationRequest, GenerationResult], Awaitable[GenerationResult]]
_JOURNALED_PROVIDER_ERROR = "model_execution_provider_error"


def _journaled_provider_result(error: ProviderCallError) -> GenerationResult:
    assert error.result is not None
    return error.result.model_copy(
        update={
            "metadata": {
                **error.result.metadata,
                _JOURNALED_PROVIDER_ERROR: {
                    "error_code": error.error_code,
                    "outcome": error.outcome,
                    "provider_status_code": error.provider_status_code,
                    "provider_request_id": error.provider_request_id,
                },
            },
        }
    )


def _typed_schema_handoff(
    error: ProviderCallError, request: GenerationRequest | EmbeddingRequest,
    context: ModelCallContext, enabled: bool,
) -> bool:
    """Deliver a completed raw answer to its existing validator, never accept it here."""
    return bool(
        enabled
        and isinstance(request, GenerationRequest)
        and context.stage is AppStage.ANSWER
        and request.metadata.get("answer_contract_version") == "typed-answer-contract-v3"
        and request.response_format
        and error.error_code == "CanonicalSchemaViolation"
        and error.outcome == "completed"
        and error.result is not None
        and error.result.request_id == request.request_id
        and error.result.model_alias == request.model_alias
        and error.result.finish_reason == "stop"
    )


def _image_manifest(image: Any) -> dict[str, object | None]:
    return {
        "image_id": image.image_id,
        "asset_id": image.asset_id,
        "sha256": image.sha256,
        "storage_uri": image.storage_uri,
        "remote_url": image.remote_url,
        "mime_type": image.mime_type,
        "width": image.width,
        "height": image.height,
    }


def _generation_manifest(request: GenerationRequest) -> dict[str, object]:
    messages: list[dict[str, object]] = []
    for message in request.messages:
        parts: list[dict[str, object | None]] = []
        for part in message.parts:
            if part.kind is ModelContentKind.TEXT:
                parts.append({"kind": "text", "text": part.text})
            else:
                assert part.image is not None
                parts.append(
                    {
                        "kind": "image",
                        "image": _image_manifest(part.image),
                        "detail": part.detail,
                    }
                )
        messages.append({"role": message.role.value, "name": message.name, "parts": parts})
    return {
        "request_id": request.request_id,
        "model_alias": request.model_alias,
        "messages": messages,
        "max_output_tokens": request.max_output_tokens,
        **(
            {"provider_max_output_tokens": request.provider_max_output_tokens}
            if request.provider_max_output_tokens is not None
            else {}
        ),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "seed": request.seed,
        "timeout_seconds": request.timeout_seconds,
        "response_format": request.response_format,
        **(
            {"thinking_mode": request.thinking_mode.value}
            if request.thinking_mode.value != "provider_default"
            else {}
        ),
        "metadata": request.metadata,
    }


def _embedding_manifest(request: EmbeddingRequest) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "model_alias": request.model_alias,
        "inputs": [
            {
                "input_id": item.input_id,
                "text": item.text,
                "images": [_image_manifest(image) for image in item.image_refs],
                "metadata": item.metadata,
            }
            for item in request.inputs
        ],
        "timeout_seconds": request.timeout_seconds,
        "dimensions": request.dimensions,
        "metadata": request.metadata,
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


_EMBEDDING_ARTIFACT_VERSION = "pmmc-embedding-result-v1"


def _embedding_result_manifest(result: EmbeddingResult) -> dict[str, object]:
    """Return a provider-neutral payload that can recover a completed batch."""

    return {
        "artifact_version": _EMBEDDING_ARTIFACT_VERSION,
        "request_id": result.request_id,
        "model_alias": result.model_alias,
        "actual_model_id": result.actual_model_id,
        "vectors": [item.model_dump(mode="json") for item in result.vectors],
        "usage": result.usage.model_dump(mode="json") if result.usage is not None else None,
        "provider_request_id": result.provider_request_id,
        "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
        "raw_response": result.raw_response,
        "metadata": result.metadata,
    }


def embedding_result_from_artifact(data: bytes) -> EmbeddingResult:
    """Rebuild an embedding result from an explicitly recoverable output artifact."""

    value = json.loads(data)
    if not isinstance(value, dict) or value.get("artifact_version") != _EMBEDDING_ARTIFACT_VERSION:
        raise ValueError("model response artifact is not a recoverable embedding result")
    return EmbeddingResult.model_validate(
        {key: item for key, item in value.items() if key != "artifact_version"}
    )


def _langfuse_trace_id(context: ModelCallContext) -> str:
    identity = f"{context.run_id}:{context.query_id or context.model_call_id}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _assert_runtime_safe(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).strip().lower() in _EVALUATION_ONLY_KEYS:
                raise ValueError(f"runtime model request contains evaluation-only field: {key}")
            _assert_runtime_safe(item)
    elif isinstance(value, list):
        for item in value:
            _assert_runtime_safe(item)


class ModelExecutionService:
    """Execute one fixed model alias and persist its full audit trail before return."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        artifact_service: ExperimentArtifactService,
        model_calls: ModelCallWriter,
        data_scope: DataScope = DataScope.RUNTIME,
        trace_sink: TraceSink | None = None,
        paper_mode: bool = False,
        max_attempts: int = 1,
        max_concurrency: int = 8,
        runtime_id: str | None = None,
        runtime_service_ids: dict[str, str] | None = None,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if artifact_service.scope is not data_scope:
            raise ValueError("artifact service and model execution scope must match")
        if paper_mode:
            registry.assert_paper_ready()
        self.registry = registry
        self.artifact_service = artifact_service
        self.model_calls = model_calls
        self.data_scope = data_scope
        self.trace_sink = trace_sink or NoopTraceSink()
        self.paper_mode = paper_mode
        self.max_attempts = max_attempts
        self.max_concurrency = max_concurrency
        self.runtime_id = runtime_id
        self.runtime_service_ids = dict(runtime_service_ids or {})
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def has_saved_generation(
        self, request: GenerationRequest, model_call_id: str,
    ) -> bool:
        """Check the exact old request before applying a newly approved input builder."""
        audit = current_campaign_audit()
        if audit is None:
            return False
        key = audit.identity(model_call_id, _generation_manifest(request),
            self.registry.resolved_hash_for_aliases([request.model_alias]))
        restored = await asyncio.to_thread(audit.restore_result, model_call_id, key)
        if restored is None:
            return False
        GenerationResult.model_validate(restored["result"])
        return True

    async def generate(
        self,
        gateway: GenerationGateway,
        request: GenerationRequest,
        context: ModelCallContext,
    ) -> GenerationResult:
        manifest = _generation_manifest(request)
        result, _trace = await self._execute(
            gateway_call=lambda: gateway.generate(request),
            gateway_recovery=getattr(gateway, "recover_result", None),
            request=request,
            request_manifest=manifest,
            context=context,
        )
        return cast(GenerationResult, result)

    async def generate_audited(
        self,
        gateway: GenerationGateway,
        request: GenerationRequest,
        context: ModelCallContext,
        output_parser: GenerationOutputParser | None = None,
        *,
        typed_schema_handoff: bool = False,
    ) -> AuditedGenerationResult:
        """Return the trace needed to persist a final answer without a second lookup."""

        result, trace = await self._execute(
            gateway_call=lambda: gateway.generate(request),
            gateway_recovery=getattr(gateway, "recover_result", None),
            request=request,
            request_manifest=_generation_manifest(request),
            context=context,
            output_parser=output_parser,
            typed_schema_handoff=typed_schema_handoff,
        )
        if not isinstance(result, GenerationResult):
            raise TypeError("generation gateway returned a non-generation result")
        return AuditedGenerationResult(result=result, trace=trace)

    async def embed(
        self,
        gateway: EmbeddingGateway,
        request: EmbeddingRequest,
        context: ModelCallContext,
    ) -> EmbeddingResult:
        manifest = _embedding_manifest(request)
        result, _trace = await self._execute(
            gateway_call=lambda: gateway.embed(request),
            request=request,
            request_manifest=manifest,
            context=context,
        )
        return cast(EmbeddingResult, result)

    async def embed_audited(
        self,
        gateway: EmbeddingGateway,
        request: EmbeddingRequest,
        context: ModelCallContext,
    ) -> AuditedEmbeddingResult:
        result, trace = await self._execute(
            gateway_call=lambda: gateway.embed(request),
            request=request,
            request_manifest=_embedding_manifest(request),
            context=context,
        )
        if not isinstance(result, EmbeddingResult):
            raise TypeError("embedding gateway returned a non-embedding result")
        return AuditedEmbeddingResult(result=result, trace=trace)

    async def _execute(
        self,
        *,
        gateway_call: Any,
        request: GenerationRequest | EmbeddingRequest,
        request_manifest: dict[str, object],
        context: ModelCallContext,
        output_parser: GenerationOutputParser | None = None,
        gateway_recovery: GenerationRecovery | None = None,
        typed_schema_handoff: bool = False,
    ) -> tuple[ResultT, ModelCallTrace]:
        end_to_end_started = time.perf_counter()
        if self.data_scope is DataScope.RUNTIME:
            _assert_runtime_safe(request_manifest)
        entry = self.registry.get(request.model_alias)
        audit = current_campaign_audit()
        journal_key = (
            audit.identity(
                context.model_call_id,
                request_manifest,
                self.registry.resolved_hash_for_aliases([request.model_alias]),
            )
            if audit
            else None
        )
        restored = (
            await asyncio.to_thread(audit.restore_result, context.model_call_id, journal_key)
            if audit and journal_key
            else None
        )
        restored_result = (
            (
                GenerationResult if isinstance(request, GenerationRequest) else EmbeddingResult
            ).model_validate(restored["result"])
            if restored
            else None
        )
        saved_trace = None
        if audit and journal_key and restored_result is not None:
            saved_trace = await asyncio.to_thread(audit.restore_trace, journal_key)
            if saved_trace is not None:
                if saved_trace.model_call_id != context.model_call_id:
                    raise ValueError("model_trace_journal_call_identity_mismatch")
        input_bytes = _canonical_bytes(request_manifest)
        prompt_hash = hashlib.sha256(input_bytes).hexdigest()
        input_artifact_id: str | None = None
        output_artifact_id: str | None = None
        result: ResultT | None = None
        retry_count = 0
        error: Exception | None = None
        output_contract_error = False
        provider_latency_ms: float | None = None
        parsed_output: dict[str, Any] | None = None
        parse_error: str | None = None
        repair_attempt_count = 0
        queue_wait_ms = 0.0
        artifact_persistence_ms = 0.0
        measured_provider_ms = 0.0
        trace_metadata: dict[str, Any] = dict(
            restored["lineage"] if restored else audit.metadata if audit else {}
        )

        async def persist_artifact(**kwargs: Any) -> Any:
            nonlocal artifact_persistence_ms
            started = time.perf_counter()
            try:
                return await self.artifact_service.put_bytes(**kwargs)
            finally:
                artifact_persistence_ms += (time.perf_counter() - started) * 1000

        try:
            input_artifact = await persist_artifact(
                artifact_id=f"{context.model_call_id}:input",
                run_id=context.run_id,
                evaluation_run_id=context.evaluation_run_id,
                semantic_run_id=context.semantic_run_id,
                snapshot_id=context.snapshot_id,
                query_id=context.query_id,
                artifact_type=AssetType.PROMPT,
                data=input_bytes,
                mime_type="application/json",
                extension="json",
            )
            input_artifact_id = input_artifact.artifact_id
            try:
                if restored_result is not None:
                    result = cast(ResultT, restored_result)
                    pending = result.metadata.get(_JOURNALED_PROVIDER_ERROR)
                    if pending is not None:
                        if not isinstance(pending, dict) or not isinstance(
                            result,
                            GenerationResult,
                        ):
                            raise ValueError("invalid journaled provider error")
                        if (
                            pending["error_code"]
                            in {
                                "ProviderSettlementNeedsReconciliation",
                                "ResponseJournalUnavailable",
                            }
                            and isinstance(request, GenerationRequest)
                            and callable(gateway_recovery)
                        ):
                            # This hook may reconcile accounting, but must never dispatch.
                            result = cast(ResultT, await gateway_recovery(request, result))
                        else:
                            completed_error = ProviderCallError(**pending, result=result)
                            if not _typed_schema_handoff(
                                completed_error, request, context, typed_schema_handoff,
                            ):
                                raise completed_error
                            trace_metadata["typed_schema_handoff"] = dict(pending)
                else:
                    for attempt in range(1, self.max_attempts + 1):
                        retry_count = attempt - 1
                        try:
                            queued_at = time.perf_counter()
                            async with AsyncExitStack() as stack:
                                if audit and audit.request_semaphore is not None:
                                    await stack.enter_async_context(audit.request_semaphore)
                                await stack.enter_async_context(self._semaphore)
                                queue_wait_ms += (time.perf_counter() - queued_at) * 1000
                                provider_started = time.perf_counter()
                                try:
                                    result = await gateway_call()
                                finally:
                                    measured_provider_ms += (
                                        time.perf_counter() - provider_started
                                    ) * 1000
                        except Exception as exc:
                            if isinstance(exc, ProviderCallError) and exc.result is not None:
                                result = cast(ResultT, _journaled_provider_result(exc))
                                error = exc
                                break
                            if result is not None:
                                error = exc
                                break
                            if (
                                attempt == self.max_attempts
                                or getattr(exc, "retryable", None) is False
                                or getattr(exc, "outcome", None) in {"unknown", "completed"}
                            ):
                                raise
                        else:
                            break
                    # Persistence is outside the transport retry loop, including a
                    # completed response carried by a failed accounting operation.
                    if audit and journal_key and result is not None:
                        try:
                            await asyncio.to_thread(
                                audit.save_result,
                                context.model_call_id,
                                journal_key,
                                result,
                                {
                                    "runtime_id": self.runtime_id,
                                    "runtime_service_id": result.metadata.get("runtime_service_id")
                                    or self.runtime_service_ids.get(request.model_alias),
                                },
                            )
                        except Exception as persistence_error:
                            if error is not None:
                                error.add_note(
                                    "response journal persistence failed: "
                                    + type(persistence_error).__name__
                                )
                                raise error from error.__cause__
                            raise
                    if error is not None:
                        if not isinstance(error, ProviderCallError) or not _typed_schema_handoff(
                            error, request, context, typed_schema_handoff,
                        ):
                            raise error
                        trace_metadata["typed_schema_handoff"] = dict(
                            result.metadata[_JOURNALED_PROVIDER_ERROR]
                        )
                        error = None
            finally:
                provider_latency_ms = (
                    restored_result.latency_ms
                    if restored_result is not None
                    else measured_provider_ms
                )
            assert result is not None
            output_payload: object = result.raw_response
            if isinstance(result, EmbeddingResult) and request.metadata.get(
                "persist_normalized_embedding_result"
            ):
                output_payload = _embedding_result_manifest(result)
            output_bytes = _canonical_bytes(output_payload)
            output_artifact = await persist_artifact(
                artifact_id=f"{context.model_call_id}:output",
                run_id=context.run_id,
                evaluation_run_id=context.evaluation_run_id,
                semantic_run_id=context.semantic_run_id,
                snapshot_id=context.snapshot_id,
                query_id=context.query_id,
                artifact_type=AssetType.MODEL_RESPONSE,
                data=output_bytes,
                mime_type="application/json",
                extension="json",
            )
            output_artifact_id = output_artifact.artifact_id
            self.registry.verify_actual_model(
                request.model_alias,
                result.actual_model_id,
                paper_mode=self.paper_mode,
            )
            if isinstance(result, GenerationResult):
                if output_parser is None:
                    parsed_output = {
                        "text": result.text,
                        "finish_reason": result.finish_reason,
                    }
                else:
                    try:
                        parsed = output_parser(result)
                    except ModelOutputContractError as exc:
                        output_contract_error = True
                        trace_metadata["output_contract_error_code"] = exc.code
                        raise
                    parsed_output = parsed.value
                    parse_error = parsed.initial_error
                    repair_attempt_count = parsed.repair_attempt_count
        except Exception as exc:
            error = exc
            if isinstance(result, GenerationResult) and output_parser is not None:
                parse_error = " ".join(str(exc).split())[:1000] or type(exc).__name__
            if audit and input_artifact_id is not None and (result is None or output_artifact_id):
                # Keep safe failure facts even when transport produced no raw response.
                try:
                    failure = await persist_artifact(
                        artifact_id=f"{context.model_call_id}:failure",
                        run_id=context.run_id,
                        evaluation_run_id=context.evaluation_run_id,
                        semantic_run_id=context.semantic_run_id,
                        snapshot_id=context.snapshot_id,
                        query_id=context.query_id,
                        artifact_type=AssetType.MODEL_RESPONSE,
                        data=_canonical_bytes(
                            {
                                "error_type": type(exc).__name__,
                                "provider_response_received": result is not None,
                                **(
                                    {"provider_diagnostics": exc.safe_diagnostics()}
                                    if isinstance(exc, ProviderCallError)
                                    else {}
                                ),
                                "outcome": (
                                    "unknown"
                                    if getattr(exc, "outcome", None) == "unknown"
                                    else "failed"
                                ),
                            }
                        ),
                        mime_type="application/json",
                        extension="json",
                    )
                    trace_metadata["error_artifact_id"] = failure.artifact_id
                except Exception as persistence_error:
                    trace_metadata["failure_artifact_error_type"] = type(persistence_error).__name__

        if error is not None and (
            (
                audit
                and (
                    input_artifact_id is None or (result is not None and not output_contract_error)
                )
            )
            or (isinstance(error, ProviderCallError) and result is not None)
        ):
            # A post-response engineering failure must surface without occupying
            # the immutable trace ID needed for a later successful recovery.
            raise error

        usage = result.usage if result is not None else None
        provider_request_id = result.provider_request_id if result is not None else None
        runtime_id = self.runtime_id
        runtime_service_id = (
            str(result.metadata["runtime_service_id"])
            if result is not None and result.metadata.get("runtime_service_id")
            else self.runtime_service_ids.get(request.model_alias)
        )
        if restored is not None:
            from pmmc.models.campaign_audit import response_runtime

            runtime_id, runtime_service_id = response_runtime(restored)
            if runtime_id is None and entry.endpoint_kind in {
                ModelEndpointKind.VLLM_OPENAI,
                ModelEndpointKind.VLLM_V2_EMBED,
            }:
                raise ValueError("restored_managed_response_producer_missing")
        trace = ModelCallTrace(
            model_call_id=context.model_call_id,
            run_id=context.run_id,
            evaluation_run_id=context.evaluation_run_id,
            semantic_run_id=context.semantic_run_id,
            data_scope=self.data_scope,
            snapshot_id=context.snapshot_id,
            query_id=context.query_id,
            conversation_id=context.conversation_id,
            job_id=context.job_id,
            evaluation_job_id=context.evaluation_job_id,
            semantic_job_id=context.semantic_job_id,
            compiler_unit_id=context.compiler_unit_id,
            qbank_id=context.qbank_id,
            plan_id=context.plan_id,
            step_id=context.step_id,
            parent_model_call_id=context.parent_model_call_id,
            context_assembly_id=context.context_assembly_id,
            runtime_id=runtime_id,
            runtime_service_id=runtime_service_id,
            stage=context.stage,
            provider=entry.endpoint_kind.value,
            model_alias=request.model_alias,
            actual_model_id=result.actual_model_id if result is not None else None,
            prompt_hash=prompt_hash,
            input_asset_id=input_artifact_id,
            output_asset_id=output_artifact_id,
            input_tokens=usage.input_tokens if usage is not None else None,
            output_tokens=usage.output_tokens if usage is not None else None,
            total_tokens=usage.total_tokens if usage is not None else None,
            cost_usd=result.cost_usd if result is not None else None,
            latency_ms=provider_latency_ms,
            temperature=request.temperature if isinstance(request, GenerationRequest) else None,
            top_p=request.top_p if isinstance(request, GenerationRequest) else None,
            seed=request.seed if isinstance(request, GenerationRequest) else None,
            cache_hit=False,
            retry_count=retry_count,
            fallback_used=False,
            litellm_call_id=(
                provider_request_id
                if entry.endpoint_kind is ModelEndpointKind.LITELLM_OPENAI
                else None
            ),
            vllm_request_id=(
                provider_request_id
                if entry.endpoint_kind
                in {ModelEndpointKind.VLLM_OPENAI, ModelEndpointKind.VLLM_V2_EMBED}
                else None
            ),
            langfuse_trace_id=_langfuse_trace_id(context),
            provider_request_id=provider_request_id,
            request_config={
                "max_output_tokens": (
                    request.max_output_tokens if isinstance(request, GenerationRequest) else None
                ),
                **(
                    {"provider_max_output_tokens": request.provider_max_output_tokens}
                    if isinstance(request, GenerationRequest)
                    and request.provider_max_output_tokens is not None
                    else {}
                ),
                "dimensions": request.dimensions if isinstance(request, EmbeddingRequest) else None,
                "paper_mode": self.paper_mode,
                "max_attempts": self.max_attempts,
                "max_concurrency": self.max_concurrency,
                "request_metadata": dict(request.metadata),
                "native_image_transport": (
                    result.metadata.get("native_image_transport") if result is not None else None
                ),
                "runtime_replica_index": (
                    result.metadata.get("replica_index") if result is not None else None
                ),
                "runtime_replica_count": (
                    result.metadata.get("replica_count") if result is not None else None
                ),
                "runtime_replica_failover_count": (
                    result.metadata.get("replica_failover_count") if result is not None else None
                ),
                "successful_provider_latency_ms": (
                    result.latency_ms if result is not None else None
                ),
                "queue_wait_ms": queue_wait_ms,
                "artifact_persistence_ms": artifact_persistence_ms,
                "end_to_end_ms": (time.perf_counter() - end_to_end_started) * 1000,
                "end_to_end_scope": "through_output_artifact_before_trace_commit",
            },
            parsed_output=parsed_output,
            parse_error=parse_error,
            repair_attempt_count=repair_attempt_count,
            error_type=type(error).__name__ if error is not None else None,
            status=TraceStatus.FAILED if error is not None else TraceStatus.SUCCESS,
            metadata=trace_metadata,
        )
        trace_commit_started = time.perf_counter()
        try:
            if saved_trace is not None:
                # Recreate checked artifact rows first after a database checkpoint restore;
                # the immutable trace retains its original latency, cost and attempt.
                trace = saved_trace
            elif audit and journal_key and result is not None:
                await asyncio.to_thread(audit.save_trace, journal_key, trace)
            trace = await self.model_calls.add(trace)
            await self.trace_sink.emit(trace)
            if output_contract_error:
                print(
                    json.dumps(
                        {
                            "event": "MODEL_OUTPUT_CONTRACT_REJECTED",
                            "error_code": trace_metadata["output_contract_error_code"],
                            "response_saved": True,
                            "trace_saved": True,
                            "response_reused": restored_result is not None,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        except Exception as persistence_error:
            if error is not None:
                error.add_note("trace persistence failed: " + type(persistence_error).__name__)
                raise error from error.__cause__
            raise
        if audit is not None:
            from pmmc.compiler.ids import canonical_content_hash
            from pmmc.experiments.step8_artifacts import write_json

            measured = {
                "protocol": "step8-model-call-wall-timing-v1",
                "campaign_id": audit.campaign_id,
                "attempt_id": audit.attempt_id,
                "job_id": audit.job_id,
                "model_call_id": context.model_call_id,
                "response_reused": restored_result is not None,
                "queue_wait_ms": queue_wait_ms,
                "provider_ms": measured_provider_ms,
                "artifact_persistence_ms": artifact_persistence_ms,
                "trace_commit_ms": (time.perf_counter() - trace_commit_started) * 1000,
                "wall_completion_ms": (time.perf_counter() - end_to_end_started) * 1000,
                "measurement_id": uuid4().hex,
            }
            key = canonical_content_hash(measured)
            await asyncio.to_thread(
                write_json,
                audit.store.root / "_store/model-timings" / f"{key}.json",
                measured,
                immutable=True,
            )
        if error is not None:
            raise error
        assert result is not None
        return result, trace
