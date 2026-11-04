"""Recoverable Questioner producer for session-anchored compilation units."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.ids import (
    canonical_content_hash,
    questioner_model_call_id,
    questioner_revision_model_call_id,
    questioner_verifier_model_call_id,
    stable_compiler_id,
)
from pmmc.compiler.opportunities import local_opportunity_id
from pmmc.compiler.questioner import (
    GatedQuestionCandidate,
    QuestionerCandidateGate,
    ResolvedQuestionerV33Draft,
    questioner_output_parser,
    questioner_revision_output_parser,
    questioner_verifier_output_parser,
    questioner_verifier_v3_output_parser,
    questioner_verifier_v5_output_parser,
    questioner_verifier_v6_output_parser,
    resolve_v3_draft,
    resolve_v33_draft,
    resolve_v34_draft,
    resolve_v35_draft,
)
from pmmc.compiler.questioner_config import (
    EVIDENCE_FIRST_VERSION,
    QuestionerConfig,
    questioner_config_hash,
)
from pmmc.compiler.questioner_native import (
    NativeSupportError,
    ResolvedQuestionerV36Draft,
    allocate_opportunities,
    build_questioner_opportunities,
    questioner_verifier_v7_output_parser,
    resolve_v36_draft,
)
from pmmc.compiler.questioner_prompt import (
    questioner_prompt_sha256,
    questioner_request_contract_hash,
    questioner_response_schema_sha256,
    render_questioner_request,
    render_questioner_revision_request,
    render_questioner_verifier_request,
)
from pmmc.compiler.units import BuiltCompilationUnit
from pmmc.infrastructure.database.errors import (
    safe_database_error_diagnostics,
    safe_database_error_summary,
)
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.repositories import (
    PostgresMemoryRepository,
    PostgresSessionRepository,
)
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.models.execution import ModelCallContext, ModelExecutionService
from pmmc.models.protocols import GenerationGateway
from pmmc.schemas import (
    AppStage,
    AssetType,
    CandidateGateStatus,
    CandidateRejectionCode,
    CompilationStage,
    CompilationStageResult,
    CompilationUnit,
    CompilationUnitLink,
    ExperimentJob,
    JobStatus,
    Modality,
    ModelCallTrace,
    QuestionerAnswerableDraftV33,
    QuestionerAnswerableDraftV34,
    QuestionerAnswerableDraftV35,
    QuestionerAnswerClaim,
    QuestionerCandidateDraft,
    QuestionerCandidateDraftV3,
    QuestionerCandidateDraftV32,
    QuestionerCandidateOrigin,
    QuestionerCandidateSupersession,
    QuestionerCandidateVerification,
    QuestionerControlDraftV33,
    QuestionerControlDraftV34,
    QuestionerControlDraftV35,
    QuestionerGeneration,
    QuestionerParseStatus,
    QuestionerRevisionEnvelopeV1,
    QuestionerRevisionEnvelopeV2,
    QuestionerVerificationBatch,
    QuestionerVerificationBatchV1,
    QuestionerVerificationBatchV2,
    QuestionerVerificationDimensionsV2,
    QuestionerVerificationEnvelopeV3,
    QuestionerVerificationReason,
    QuestionerVerificationVerdict,
    RunStatus,
    StructuredOutputItemDiagnostic,
    StructuredOutputItemStatus,
    StructuredOutputStage,
    StructuredOutputValidationError,
    TraceStatus,
)
from pmmc.schemas.base import utc_now

ItemCallback = Callable[[str, bool, str | None], None]
ProgressCallback = Callable[[Mapping[str, object]], None]

_V3_REASON_BY_DIMENSION = {
    "user_facing": QuestionerVerificationReason.NOT_USER_FACING,
    "future_plausible": QuestionerVerificationReason.NOT_FUTURE_PLAUSIBLE,
    "not_historical_replay": QuestionerVerificationReason.HISTORICAL_REPLAY,
    "answer_explicitly_entailed": QuestionerVerificationReason.ANSWER_UNSUPPORTED,
    "evidence_sufficient": QuestionerVerificationReason.EVIDENCE_INSUFFICIENT,
    "taxonomy_correct": QuestionerVerificationReason.TAXONOMY_MISMATCH,
    "modality_correct": QuestionerVerificationReason.MODALITY_MISMATCH,
    "visual_necessary": QuestionerVerificationReason.VISUAL_NOT_NEEDED,
    "multi_hop_genuine": QuestionerVerificationReason.MULTI_HOP_NOT_GENUINE,
    "metadata_free": QuestionerVerificationReason.INTERNAL_METADATA,
}

_TERMINAL_GENERATION_REASON = "generation_structured_output_invalid"
_V3_WORKER_MULTIPLIER = 2


def _questioner_worker_limit_from_environment() -> int | None:
    raw = os.environ.get("PMMC_QUESTIONER_WORKER_LIMIT")
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("PMMC_QUESTIONER_WORKER_LIMIT must be an integer") from exc
    if value <= 0:
        raise ValueError("PMMC_QUESTIONER_WORKER_LIMIT must be positive")
    return value


def _questioner_v3_reason_codes(
    failed_dimensions: set[str],
) -> list[QuestionerVerificationReason]:
    """Derive authoritative audit codes without adding provider-only invariants."""

    return sorted(
        (reason for name, reason in _V3_REASON_BY_DIMENSION.items() if name in failed_dimensions),
        key=lambda item: item.value,
    )


def _is_structured_output_failure(trace: ModelCallTrace | None) -> bool:
    """Distinguish an archived invalid response from a provider/runtime failure."""

    return bool(
        trace is not None
        and trace.status is TraceStatus.FAILED
        and trace.output_asset_id is not None
        and trace.parse_error
    )


def questioner_seed_diagnostics(
    generation: QuestionerGeneration,
) -> tuple[StructuredOutputItemDiagnostic, ...]:
    """Materialize replay-stable diagnostics from mixed parser ordinal schemes."""

    raw_values: list[tuple[str, Mapping[str, object]]] = []
    for kind, metadata_key in (
        ("schema", "invalid_diagnostics"),
        ("resolution", "resolution_diagnostics"),
    ):
        values = generation.metadata.get(metadata_key, [])
        if not isinstance(values, list):
            continue
        raw_values.extend((kind, value) for value in values if isinstance(value, Mapping))

    used_ordinals: set[int] = set()
    repair_ordinal = generation.raw_candidate_count
    reserved_ordinals = {repair_ordinal} if generation.repair_attempt_count else set()
    overflow_ordinal = repair_ordinal + 1

    def allocate(preferred: int) -> int:
        nonlocal overflow_ordinal
        if preferred not in used_ordinals and preferred not in reserved_ordinals:
            used_ordinals.add(preferred)
            return preferred
        while overflow_ordinal in used_ordinals or overflow_ordinal in reserved_ordinals:
            overflow_ordinal += 1
        allocated = overflow_ordinal
        used_ordinals.add(allocated)
        overflow_ordinal += 1
        return allocated

    diagnostics: list[StructuredOutputItemDiagnostic] = []
    for fallback_ordinal, (kind, raw) in enumerate(raw_values):
        raw_ordinal = raw.get("ordinal")
        raw_rank = raw.get("generation_rank")
        if isinstance(raw_ordinal, int) and not isinstance(raw_ordinal, bool):
            source_ordinal = max(0, raw_ordinal)
        elif isinstance(raw_rank, int) and not isinstance(raw_rank, bool):
            # Provider generation ranks are one-based; parser ordinals are zero-based.
            source_ordinal = max(0, raw_rank - 1)
        else:
            source_ordinal = fallback_ordinal
        ordinal = allocate(source_ordinal)
        item_hash = raw.get("item_hash")
        if not isinstance(item_hash, str) or len(item_hash) != 64:
            item_hash = canonical_content_hash(
                [
                    generation.model_call_id,
                    kind,
                    source_ordinal,
                    "invalid",
                ]
            )
        errors: list[StructuredOutputValidationError] = []
        raw_errors = raw.get("errors", [])
        if isinstance(raw_errors, list):
            for item in raw_errors:
                if not isinstance(item, Mapping):
                    continue
                location = item.get("location")
                error_type = item.get("type")
                if isinstance(location, str) and isinstance(error_type, str):
                    errors.append(
                        StructuredOutputValidationError(
                            location=location[:256] or "$",
                            error_type=error_type[:128] or "validation_error",
                        )
                    )
        if not errors:
            errors = [
                StructuredOutputValidationError(
                    location="$",
                    error_type=str(
                        raw.get("reason_code", raw.get("error_type", "resolution_error"))
                    )[:128],
                )
            ]
        diagnostics.append(
            StructuredOutputItemDiagnostic(
                diagnostic_id=stable_compiler_id(
                    "structured-output-diagnostic",
                    [
                        generation.model_call_id,
                        "questioner",
                        ordinal,
                        item_hash,
                        kind,
                    ],
                ),
                run_id=generation.run_id,
                model_call_id=generation.model_call_id,
                compiler_unit_id=generation.compiler_unit_id,
                stage=StructuredOutputStage.QUESTIONER,
                item_ordinal=ordinal,
                item_hash=item_hash,
                status=StructuredOutputItemStatus.INVALID,
                parser_version=generation.parser_version,
                errors=errors,
                created_at=generation.created_at,
                metadata={
                    "diagnostic_kind": kind,
                    "source_item_ordinal": source_ordinal,
                },
            )
        )
    if generation.repair_attempt_count:
        diagnostics.append(
            StructuredOutputItemDiagnostic(
                diagnostic_id=stable_compiler_id(
                    "structured-output-diagnostic",
                    [
                        generation.model_call_id,
                        "questioner",
                        repair_ordinal,
                        "json_repair",
                    ],
                ),
                run_id=generation.run_id,
                model_call_id=generation.model_call_id,
                compiler_unit_id=generation.compiler_unit_id,
                stage=StructuredOutputStage.QUESTIONER,
                item_ordinal=repair_ordinal,
                item_hash=generation.output_hash
                or canonical_content_hash(generation.model_call_id),
                status=StructuredOutputItemStatus.REPAIRED,
                parser_version=generation.parser_version,
                repair_attempt_count=generation.repair_attempt_count,
                errors=[StructuredOutputValidationError(location="$", error_type="json_repaired")],
                created_at=generation.created_at,
                metadata={
                    "diagnostic_kind": "json_repair",
                    "source_item_ordinal": repair_ordinal,
                },
            )
        )
    if (
        generation.parse_status is QuestionerParseStatus.FAILED
        and generation.metadata.get("terminal_reason") == _TERMINAL_GENERATION_REASON
    ):
        ordinal = allocate(0)
        diagnostics.append(
            StructuredOutputItemDiagnostic(
                diagnostic_id=stable_compiler_id(
                    "structured-output-diagnostic",
                    [
                        generation.model_call_id,
                        "questioner",
                        ordinal,
                        _TERMINAL_GENERATION_REASON,
                    ],
                ),
                run_id=generation.run_id,
                model_call_id=generation.model_call_id,
                compiler_unit_id=generation.compiler_unit_id,
                stage=StructuredOutputStage.QUESTIONER,
                item_ordinal=ordinal,
                item_hash=generation.output_hash
                or canonical_content_hash(generation.model_call_id),
                status=StructuredOutputItemStatus.CONSERVATIVE_REJECT,
                parser_version=generation.parser_version,
                repair_attempt_count=int(
                    generation.metadata.get("provider_repair_attempt_count", 0)
                ),
                errors=[
                    StructuredOutputValidationError(
                        location="$",
                        error_type="structured_output_invalid",
                    )
                ],
                created_at=generation.created_at,
                metadata={
                    "diagnostic_kind": "terminal_generation_rejection",
                    "source_item_ordinal": 0,
                },
            )
        )
    return tuple(diagnostics)


class QuestionerInputProvider(Protocol):
    """Rebuild one frozen unit request without retaining all requests in memory."""

    async def load(self, unit: CompilationUnit) -> BuiltCompilationUnit: ...


class PreparedQuestionerInputError(ValueError):
    """Safe error for stale prepared request identities."""

    def __init__(self, diagnostic_code: str) -> None:
        self.questioner_error_code = diagnostic_code
        super().__init__(diagnostic_code)


class QuestionerGenerationContractError(ValueError):
    """Safe generation-contract failure without provider payload disclosure."""

    _ALLOWED_CODES = frozenset(
        {
            "contract_invalid",
            "structured_output_invalid",
        }
    )

    def __init__(self, diagnostic_code: str) -> None:
        if diagnostic_code not in self._ALLOWED_CODES:
            raise ValueError("unsupported Questioner generation diagnostic")
        self.questioner_error_code = diagnostic_code
        super().__init__(diagnostic_code)


class QuestionerStageError(RuntimeError):
    """Redacted stage marker for recoverable Questioner job diagnostics."""

    def __init__(self, failure_stage: str, cause: BaseException) -> None:
        self.failure_stage = failure_stage
        cause_code = getattr(cause, "questioner_error_code", type(cause).__name__)
        if (
            not isinstance(cause_code, str)
            or not cause_code
            or not cause_code.replace("_", "").isalnum()
        ):
            cause_code = type(cause).__name__
        cause_code = cause_code[:80]
        self.error_code = f"questioner_{failure_stage}_{cause_code}"
        super().__init__(self.error_code)
        self.__cause__ = cause


class QuestionerPersistenceOperationError(RuntimeError):
    """Redacted locator for one atomic Questioner persistence operation."""

    _ALLOWED_OPERATIONS = frozenset(
        {
            "generation",
            "gated_candidates",
            "verification_batch",
            "structured_output_diagnostic",
            "supersession",
            "stage_result",
        }
    )

    def __init__(
        self,
        persistence_operation: str,
        cause: BaseException,
        *,
        candidate_count: int,
        revised_candidate_count: int,
        verification_batch_count: int,
        supersession_count: int,
        verification_input_count: int | None = None,
        verification_pass_count: int | None = None,
        verification_revise_count: int | None = None,
        verification_reject_count: int | None = None,
        verification_provider_fallback: bool | None = None,
        verification_single_candidate_scope: bool | None = None,
    ) -> None:
        if persistence_operation not in self._ALLOWED_OPERATIONS:
            raise ValueError("unsupported Questioner persistence operation")
        counts = (
            candidate_count,
            revised_candidate_count,
            verification_batch_count,
            supersession_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("Questioner persistence counts must be non-negative")
        self.persistence_operation = persistence_operation
        self.persistence_candidate_count = candidate_count
        self.persistence_revised_candidate_count = revised_candidate_count
        self.persistence_verification_batch_count = verification_batch_count
        self.persistence_supersession_count = supersession_count
        optional_counts = {
            "persistence_verification_input_count": verification_input_count,
            "persistence_verification_pass_count": verification_pass_count,
            "persistence_verification_revise_count": verification_revise_count,
            "persistence_verification_reject_count": verification_reject_count,
        }
        for attribute, value in optional_counts.items():
            if value is not None:
                if value < 0:
                    raise ValueError("Questioner verification counts must be non-negative")
                setattr(self, attribute, value)
        if verification_provider_fallback is not None:
            self.persistence_verification_provider_fallback = int(verification_provider_fallback)
        if verification_single_candidate_scope is not None:
            self.persistence_verification_single_candidate_scope = int(
                verification_single_candidate_scope
            )
        cause_code = getattr(cause, "questioner_error_code", type(cause).__name__)
        if not isinstance(cause_code, str) or not cause_code:
            cause_code = type(cause).__name__
        self.questioner_error_code = cause_code[:80]
        super().__init__(
            f"questioner_persistence_{persistence_operation}_{self.questioner_error_code}"
        )
        self.__cause__ = cause


class PostgresQuestionerInputProvider:
    """Load canonical unit members and verify the prepared request identity."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        config: QuestionerConfig,
        token_counter: Any | None = None,
        conversation_cache_size: int = 8,
    ) -> None:
        if conversation_cache_size <= 0:
            raise ValueError("Questioner input cache size must be positive")
        self.sessions = sessions
        self.config = config
        self.token_counter = token_counter
        self.conversation_cache_size = conversation_cache_size
        self._cache: OrderedDict[tuple[str, str], tuple[list[Any], list[Any]]] = OrderedDict()
        self._inflight: dict[tuple[str, str], asyncio.Task[tuple[list[Any], list[Any]]]] = {}
        self._cache_lock = asyncio.Lock()

    async def _load_conversation(
        self,
        key: tuple[str, str],
    ) -> tuple[list[Any], list[Any]]:
        snapshot_id, conversation_id = key
        conversation_ids = {conversation_id}
        try:
            # Keep this sequential. A cold-cache burst otherwise asks the bounded
            # SQLAlchemy pool for two connections per worker and can deadlock until
            # the pool timeout before either query can complete.
            sessions = await PostgresSessionRepository(self.sessions).list_snapshot(
                snapshot_id, conversation_ids=conversation_ids
            )
            memories = await PostgresMemoryRepository(self.sessions).list_snapshot(
                snapshot_id, conversation_ids=conversation_ids
            )
            value = (sessions, memories)
            async with self._cache_lock:
                self._cache[key] = value
                self._cache.move_to_end(key)
                while len(self._cache) > self.conversation_cache_size:
                    self._cache.popitem(last=False)
            return value
        finally:
            async with self._cache_lock:
                current = self._inflight.get(key)
                if current is asyncio.current_task():
                    self._inflight.pop(key, None)

    async def _conversation(self, unit: CompilationUnit) -> tuple[list[Any], list[Any]]:
        key = (unit.snapshot_id, unit.conversation_id)
        async with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(self._load_conversation(key))
                self._inflight[key] = task
        # One cancelled worker must not cancel the shared conversation load used
        # by the remaining claimed jobs.
        return await asyncio.shield(task)

    async def load(self, unit: CompilationUnit) -> BuiltCompilationUnit:
        sessions, available_memories = await self._conversation(unit)
        memory_map = {item.memory_id: item for item in available_memories}
        try:
            memories = [memory_map[item.memory_id] for item in unit.memories]
        except KeyError as exc:
            raise KeyError("prepared compilation unit references missing memory") from exc
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.questioner is not None
            links = await uow.questioner.list_links(unit.compiler_unit_id)
        opportunities = (
            build_questioner_opportunities(
                config=self.config,
                unit=unit,
                memories=memories,
                sessions=sessions,
                links=links,
            )
            if self.config.is_v3
            else None
        )
        request = render_questioner_request(
            request_id=f"questioner-{unit.compiler_unit_id}",
            model_alias=self.config.compiler_model_alias,
            unit_id=unit.compiler_unit_id,
            cutoff_session_id=unit.available_after_session,
            memories=memories,
            sessions=sessions,
            max_output_tokens=self.config.reserved_output_tokens,
            prompt_version=self.config.prompt_version,
            renderer_version=self.config.renderer_version,
            opportunity_set=opportunities,
            memory_raw_image_policy=self.config.memory_raw_image_policy,
        )
        actual_contract_hash = questioner_request_contract_hash(
            prompt_version=self.config.prompt_version,
            renderer_version=self.config.renderer_version,
            opportunity_builder_version=(
                self.config.opportunity_builder_version if self.config.is_v3 else None
            ),
        )
        expected_contract_hash = unit.metadata.get("prepared_request_contract_hash")
        if (
            isinstance(expected_contract_hash, str)
            and expected_contract_hash != actual_contract_hash
        ):
            raise PreparedQuestionerInputError("prepared_request_contract_mismatch")
        request_sha256 = hashlib.sha256(request.to_json().encode("utf-8")).hexdigest()
        expected_hash = unit.metadata.get("prepared_request_sha256")
        if expected_hash is not None and expected_hash != request_sha256:
            raise PreparedQuestionerInputError("prepared_request_hash_mismatch")
        if (
            expected_hash is None
            and unit.metadata.get("prepared_request_namespace_rebound") is True
        ):
            source_hash = unit.metadata.get("prepared_request_source_sha256")
            if (
                not isinstance(source_hash, str)
                or len(source_hash) != 64
                or any(character not in "0123456789abcdef" for character in source_hash)
            ):
                raise PreparedQuestionerInputError("prepared_request_namespace_audit_missing")
        prepared_tokens = unit.metadata.get("prepared_input_tokens")
        if isinstance(prepared_tokens, int) and prepared_tokens > 0:
            input_tokens = prepared_tokens
        elif self.token_counter is not None:
            input_tokens = int(self.token_counter.count_request(request))
        else:
            raise ValueError("legacy Questioner units require a token counter")
        return BuiltCompilationUnit(
            unit=unit,
            links=tuple(links),
            memories=tuple(memories),
            sessions=tuple(sessions),
            request=request,
            input_tokens=input_tokens,
            opportunities=opportunities,
        )


@dataclass(frozen=True)
class QuestionerUnitResult:
    generation: QuestionerGeneration
    candidates: tuple[GatedQuestionCandidate, ...]
    stage_result: CompilationStageResult
    verification_batches: tuple[QuestionerVerificationBatch, ...] = ()
    verifications: tuple[QuestionerCandidateVerification, ...] = ()
    supersessions: tuple[QuestionerCandidateSupersession, ...] = ()
    answer_claims: tuple[QuestionerAnswerClaim, ...] = ()


@dataclass(frozen=True)
class QuestionerRunResult:
    run_id: str
    unit_count: int
    succeeded: int
    failed: int


class QuestionerGraphState(TypedDict, total=False):
    built: BuiltCompilationUnit
    job: ExperimentJob
    trace: ModelCallTrace
    generation: QuestionerGeneration
    drafts: list[QuestionerCandidateDraft]
    known_primaries: dict[str, str]
    gated: list[GatedQuestionCandidate]
    result: QuestionerUnitResult


class _LinearCompiledGraph:
    def __init__(
        self,
        nodes: Sequence[Callable[[QuestionerGraphState], Awaitable[dict[str, Any]]]],
    ) -> None:
        self.nodes = list(nodes)

    async def ainvoke(self, state: QuestionerGraphState) -> QuestionerGraphState:
        current = dict(state)
        for node in self.nodes:
            current.update(await node(current))  # type: ignore[arg-type]
        return current  # type: ignore[return-value]


class _CompiledQuestionerGraph(Protocol):
    async def ainvoke(self, state: QuestionerGraphState) -> QuestionerGraphState: ...


class QuestionerCompilationGraph:
    """Fixed per-unit producer graph; Postgres remains the checkpoint authority."""

    node_names = (
        "load_unit",
        "render",
        "generate",
        "parse",
        "lineage_gate",
        "deduplicate",
        "persist",
    )

    def __init__(
        self,
        *,
        execution: ModelExecutionService,
        gateway: GenerationGateway,
        config: QuestionerConfig,
        load_successful_call: Callable[[str], Awaitable[ModelCallTrace | None]],
        load_model_call: Callable[[str], Awaitable[ModelCallTrace | None]],
        before_gate: Callable[[BuiltCompilationUnit], Awaitable[dict[str, str]]],
        persist_v3_seed: Callable[
            [
                BuiltCompilationUnit,
                QuestionerGeneration,
                list[GatedQuestionCandidate],
                list[QuestionerAnswerClaim],
            ],
            Awaitable[None],
        ],
        persist_result: Callable[[QuestionerUnitResult], Awaitable[None]],
    ) -> None:
        self.execution = execution
        self.gateway = gateway
        self.config = config
        self.load_successful_call = load_successful_call
        self.load_model_call = load_model_call
        self.before_gate = before_gate
        self.persist_v3_seed = persist_v3_seed
        self.persist_result = persist_result
        self.gate = QuestionerCandidateGate(config)
        callbacks = (
            self._load_unit,
            self._render,
            self._generate,
            self._parse,
            self._lineage_gate,
            self._deduplicate,
            self._persist,
        )
        self.compiled: _CompiledQuestionerGraph
        try:
            from langgraph.graph import END, START, StateGraph
        except ModuleNotFoundError:
            self.compiled = _LinearCompiledGraph(callbacks)
        else:
            graph = StateGraph(QuestionerGraphState)
            for name, callback in zip(self.node_names, callbacks, strict=True):
                graph.add_node(name, callback)
            graph.add_edge(START, self.node_names[0])
            for source, target in zip(self.node_names, self.node_names[1:], strict=False):
                graph.add_edge(source, target)
            graph.add_edge(self.node_names[-1], END)
            self.compiled = cast(_CompiledQuestionerGraph, graph.compile())

    async def process(
        self,
        built: BuiltCompilationUnit,
        job: ExperimentJob,
    ) -> QuestionerUnitResult:
        if self.config.is_v3:
            return await self._process_v3(built, job)
        state = await self.compiled.ainvoke({"built": built, "job": job})
        return state["result"]

    async def _process_v3(
        self,
        built: BuiltCompilationUnit,
        job: ExperimentJob,
    ) -> QuestionerUnitResult:
        await self._load_unit({"built": built, "job": job})
        if built.opportunities is None:
            raise ValueError("Questioner v3 requires a frozen opportunity set")

        try:
            generation_trace = await self._v3_generation_trace(built, job)
        except QuestionerGenerationContractError as exc:
            if (
                (self.config.is_v33 or self.config.is_v34)
                and exc.questioner_error_code == "structured_output_invalid"
                and job.attempt_count >= job.max_attempts
            ):
                return await self._terminal_v33_generation_rejection(built, job)
            raise QuestionerStageError("generation", exc) from exc
        except QuestionerStageError:
            raise
        except Exception as exc:
            raise QuestionerStageError("generation", exc) from exc
        payload = dict(generation_trace.parsed_output or {})
        from pmmc.schemas.questioner import QuestionerAnswerableDraftV36, QuestionerControlDraftV36

        v33_resolved_by_rank: dict[
            int, ResolvedQuestionerV33Draft | ResolvedQuestionerV36Draft
        ] = {}
        provider_drafts_v33: list[QuestionerAnswerableDraftV33 | QuestionerControlDraftV33] = []
        provider_drafts_v34: list[QuestionerAnswerableDraftV34 | QuestionerControlDraftV34] = []
        provider_drafts_v35: list[QuestionerAnswerableDraftV35 | QuestionerControlDraftV35] = []
        provider_drafts: list[QuestionerCandidateDraftV3] = []
        provider_drafts_v36 = []
        if self.config.is_v36:
            provider_drafts_v36 = [
                QuestionerControlDraftV36.model_validate(item)
                if item.get("candidate_kind") == "refusal"
                else QuestionerAnswerableDraftV36.model_validate(item)
                for item in payload.get("candidates", [])
            ]
        elif self.config.is_v35:
            provider_drafts_v35 = [
                (
                    QuestionerControlDraftV35.model_validate(item)
                    if item.get("candidate_kind") == "refusal"
                    else QuestionerAnswerableDraftV35.model_validate(item)
                )
                for item in payload.get("candidates", [])
            ]
        elif self.config.is_v34:
            provider_drafts_v34 = [
                (
                    QuestionerControlDraftV34.model_validate(item)
                    if item.get("candidate_kind") == "refusal"
                    else QuestionerAnswerableDraftV34.model_validate(item)
                )
                for item in payload.get("candidates", [])
            ]
        elif self.config.is_v33:
            provider_drafts_v33 = [
                (
                    QuestionerControlDraftV33.model_validate(item)
                    if item.get("candidate_kind") == "refusal"
                    else QuestionerAnswerableDraftV33.model_validate(item)
                )
                for item in payload.get("candidates", [])
            ]
        elif self.config.is_v32:
            opportunity_by_alias = {
                local_opportunity_id(item): item for item in built.opportunities.opportunities
            }
            for item in payload.get("candidates", []):
                provider = QuestionerCandidateDraftV32.model_validate(item)
                opportunity = opportunity_by_alias.get(provider.opportunity_id)
                if opportunity is None:
                    # Keep unknown opportunities parse-valid so the prospective
                    # resolution gate records the safe rejection.
                    modalities = [Modality.TEXT]
                else:
                    modalities = [
                        Modality(value.value) for value in opportunity.required_modalities
                    ]
                provider_drafts.append(
                    QuestionerCandidateDraftV3.model_validate(
                        {
                            **provider.model_dump(mode="json"),
                            "required_modalities": [value.value for value in modalities],
                        }
                    )
                )
        else:
            provider_drafts = [
                QuestionerCandidateDraftV3.model_validate(item)
                for item in payload.get("candidates", [])
            ]
        source_draft_count = (
            len(provider_drafts_v36)
            if self.config.is_v36
            else len(provider_drafts_v35)
            if self.config.is_v35
            else len(provider_drafts_v34)
            if self.config.is_v34
            else len(provider_drafts_v33)
            if self.config.is_v33
            else len(provider_drafts)
        )
        candidate_ordinals = [int(item) for item in payload.get("candidate_ordinals", [])]
        if not candidate_ordinals and source_draft_count:
            candidate_ordinals = list(range(1, source_draft_count + 1))
        if (
            len(candidate_ordinals) != source_draft_count
            or len(set(candidate_ordinals)) != len(candidate_ordinals)
            or any(item <= 0 for item in candidate_ordinals)
        ):
            raise ValueError("Questioner v3 candidate ordinals are invalid")
        raw_count = int(payload.get("raw_candidate_count", source_draft_count))
        schema_invalid_count = int(payload.get("invalid_candidate_count", 0))
        resolution_diagnostics: list[dict[str, object]] = []
        resolved: list[tuple[int, QuestionerCandidateDraftV3, QuestionerCandidateDraft]] = []
        if self.config.is_v33 or self.config.is_v34:
            claim_drafts = (
                provider_drafts_v36
                if self.config.is_v36
                else provider_drafts_v35
                if self.config.is_v35
                else provider_drafts_v34
                if self.config.is_v34
                else provider_drafts_v33
            )
            for rank, claim_draft in zip(candidate_ordinals, claim_drafts, strict=True):
                v33: ResolvedQuestionerV33Draft | ResolvedQuestionerV36Draft
                internal: QuestionerCandidateDraft
                try:
                    if isinstance(
                        claim_draft, (QuestionerAnswerableDraftV36, QuestionerControlDraftV36)
                    ):
                        v33 = resolve_v36_draft(
                            claim_draft,
                            opportunity_set=built.opportunities,
                            memories=built.memories,
                            memory_raw_image_policy=self.config.memory_raw_image_policy,
                        )
                    elif isinstance(
                        claim_draft,
                        (QuestionerAnswerableDraftV35, QuestionerControlDraftV35),
                    ):
                        v33 = resolve_v35_draft(
                            claim_draft,
                            opportunity_set=built.opportunities,
                            memories=built.memories,
                        )
                    elif isinstance(
                        claim_draft,
                        (QuestionerAnswerableDraftV34, QuestionerControlDraftV34),
                    ):
                        v33 = resolve_v34_draft(
                            claim_draft,
                            opportunity_set=built.opportunities,
                            memories=built.memories,
                        )
                    else:
                        v33 = resolve_v33_draft(
                            claim_draft,
                            opportunity_set=built.opportunities,
                            memories=built.memories,
                        )
                    v33_resolved_by_rank[rank] = v33
                    provider_for_verifier = v33.provider_for_verifier
                    internal = v33.internal
                except ValueError as exc:
                    resolution_diagnostics.append(
                        {
                            "generation_rank": rank,
                            "item_hash": canonical_content_hash(
                                claim_draft.model_dump(mode="json")
                            ),
                            "error_type": type(exc).__name__,
                            **(
                                {"reason_code": exc.code}
                                if isinstance(exc, NativeSupportError)
                                else {}
                            ),
                        }
                    )
                else:
                    resolved.append((rank, provider_for_verifier, internal))
        else:
            for rank, legacy_draft in zip(candidate_ordinals, provider_drafts, strict=True):
                try:
                    internal = resolve_v3_draft(
                        legacy_draft,
                        opportunity_set=built.opportunities,
                    )
                except ValueError as exc:
                    resolution_diagnostics.append(
                        {
                            "generation_rank": rank,
                            "item_hash": canonical_content_hash(
                                legacy_draft.model_dump(mode="json")
                            ),
                            "error_type": type(exc).__name__,
                        }
                    )
                else:
                    resolved.append((rank, legacy_draft, internal))
        invalid_count = schema_invalid_count + len(resolution_diagnostics)
        if len(resolved) + invalid_count != raw_count:
            raise ValueError("Questioner v3 candidate counts do not reconcile")
        generation = self._v3_generation_fact(
            built=built,
            job=job,
            trace=generation_trace,
            payload=payload,
            valid_count=len(resolved),
            invalid_count=invalid_count,
            resolution_diagnostics=resolution_diagnostics,
        )

        static = self.gate.evaluate(
            run_id=built.unit.run_id,
            snapshot_id=built.unit.snapshot_id,
            unit=built.unit,
            generation_id=generation.generation_id,
            model_call_id=generation.model_call_id,
            drafts=[item[2] for item in resolved],
            memories=list(built.memories),
            known_primaries={},
            created_at=generation.created_at,
            opportunity_set=built.opportunities,
            generation_ranks=[item[0] for item in resolved],
            apply_dedupe=False,
        )
        if (self.config.is_v33 or self.config.is_v34) and generation.repair_attempt_count:
            static = [
                GatedQuestionCandidate(
                    candidate=item.candidate,
                    expected_evidence=item.expected_evidence,
                    gate_result=item.gate_result.model_copy(
                        update={
                            "status": CandidateGateStatus.REJECTED,
                            "rejection_codes": list(
                                dict.fromkeys(
                                    [
                                        *item.gate_result.rejection_codes,
                                        CandidateRejectionCode.INVALID_SCHEMA,
                                    ]
                                )
                            ),
                            "metadata": {
                                **item.gate_result.metadata,
                                "native_provider_output": False,
                                "terminal_reason": ("generation_local_json_repair_not_freezable"),
                            },
                        }
                    ),
                    duplicate_link=item.duplicate_link,
                )
                for item in static
            ]
        answer_claims = self._v33_answer_claim_facts(
            generation=generation,
            static=static,
            resolved_by_rank=v33_resolved_by_rank,
            opportunity_set=built.opportunities,
        )
        try:
            await self.persist_v3_seed(built, generation, static, answer_claims)
        except QuestionerStageError:
            raise
        except Exception as exc:
            raise QuestionerStageError("seed_persistence", exc) from exc

        provider_by_rank = {rank: provider for rank, provider, _ in resolved}
        internal_by_qbank = {
            gated.candidate.qbank_id: internal
            for gated, (_, _, internal) in zip(static, resolved, strict=True)
        }
        terminal: dict[str, GatedQuestionCandidate] = {
            item.candidate.qbank_id: item
            for item in static
            if item.gate_result.status is CandidateGateStatus.REJECTED
        }
        initial_inputs = [
            item
            for item in static
            if item.gate_result.status is CandidateGateStatus.ELIGIBLE_FOR_PLANNING
        ]
        verification_batches: list[QuestionerVerificationBatch] = []
        verifications: list[QuestionerCandidateVerification] = []
        supersessions: list[QuestionerCandidateSupersession] = []
        final_candidates: list[GatedQuestionCandidate] = []
        revised_static_pass: list[tuple[GatedQuestionCandidate, QuestionerCandidateDraftV3]] = []

        if initial_inputs and not (self.config.is_v32 or self.config.is_v33 or self.config.is_v34):
            try:
                (
                    initial_trace,
                    initial_output,
                    initial_parser_metadata,
                ) = await self._run_v3_verifier(
                    built=built,
                    job=job,
                    generation=generation,
                    inputs=[
                        (
                            self._candidate_alias(item.candidate.generation_rank),
                            provider_by_rank[item.candidate.generation_rank],
                        )
                        for item in initial_inputs
                    ],
                    verification_round=0,
                )
            except QuestionerStageError:
                raise
            except Exception as exc:
                raise QuestionerStageError("initial_verifier", exc) from exc
            initial_batch, initial_verifications = self._apply_initial_verifier(
                built=built,
                generation=generation,
                trace=initial_trace,
                output=initial_output,
                static_inputs=initial_inputs,
                provider_by_rank=provider_by_rank,
                internal_by_qbank=internal_by_qbank,
                terminal=terminal,
                final_candidates=final_candidates,
                revised_static_pass=revised_static_pass,
                supersessions=supersessions,
                parser_metadata=initial_parser_metadata,
            )
            verification_batches.append(initial_batch)
            verifications.extend(initial_verifications)
        elif initial_inputs:
            for initial_item in initial_inputs:
                try:
                    alias = self._candidate_alias(initial_item.candidate.generation_rank)
                    (
                        initial_trace,
                        initial_output,
                        initial_parser_metadata,
                    ) = await self._run_v3_verifier(
                        built=built,
                        job=job,
                        generation=generation,
                        inputs=[
                            (
                                alias,
                                provider_by_rank[initial_item.candidate.generation_rank],
                            )
                        ],
                        verification_round=0,
                    )
                except QuestionerStageError:
                    raise
                except Exception as exc:
                    raise QuestionerStageError("initial_verifier", exc) from exc
                initial_batch, initial_verifications = self._apply_initial_verifier(
                    built=built,
                    generation=generation,
                    trace=initial_trace,
                    output=initial_output,
                    static_inputs=[initial_item],
                    provider_by_rank=provider_by_rank,
                    internal_by_qbank=internal_by_qbank,
                    terminal=terminal,
                    final_candidates=final_candidates,
                    revised_static_pass=revised_static_pass,
                    supersessions=supersessions,
                    parser_metadata=initial_parser_metadata,
                )
                verification_batches.append(initial_batch)
                verifications.extend(initial_verifications)

        if revised_static_pass and not (
            self.config.is_v32 or self.config.is_v33 or self.config.is_v34
        ):
            try:
                (
                    final_trace,
                    final_output,
                    final_parser_metadata,
                ) = await self._run_v3_verifier(
                    built=built,
                    job=job,
                    generation=generation,
                    inputs=[
                        (
                            self._candidate_alias(item.candidate.generation_rank),
                            provider,
                        )
                        for item, provider in revised_static_pass
                    ],
                    verification_round=1,
                )
            except QuestionerStageError:
                raise
            except Exception as exc:
                raise QuestionerStageError("final_verifier", exc) from exc
            final_batch, final_verifications = self._apply_final_verifier(
                generation=generation,
                trace=final_trace,
                output=final_output,
                revised_inputs=revised_static_pass,
                terminal=terminal,
                final_candidates=final_candidates,
                parser_metadata=final_parser_metadata,
            )
            verification_batches.append(final_batch)
            verifications.extend(final_verifications)
        elif revised_static_pass:
            for revised_item, revised_provider in revised_static_pass:
                try:
                    final_trace, final_output, final_parser_metadata = await self._run_v3_verifier(
                        built=built,
                        job=job,
                        generation=generation,
                        inputs=[
                            (
                                self._candidate_alias(revised_item.candidate.generation_rank),
                                revised_provider,
                            )
                        ],
                        verification_round=1,
                    )
                except QuestionerStageError:
                    raise
                except Exception as exc:
                    raise QuestionerStageError("final_verifier", exc) from exc
                final_batch, final_verifications = self._apply_final_verifier(
                    generation=generation,
                    trace=final_trace,
                    output=final_output,
                    revised_inputs=[(revised_item, revised_provider)],
                    terminal=terminal,
                    final_candidates=final_candidates,
                    parser_metadata=final_parser_metadata,
                )
                verification_batches.append(final_batch)
                verifications.extend(final_verifications)

        known_primaries = await self.before_gate(built)
        # Opportunity allocation already happened before generation. Dedupe must
        # preserve that generation prefix, not let a later wording replace an
        # earlier family solely because it sorts first lexicographically.
        ordered_final = sorted(
            final_candidates,
            key=lambda item: (
                item.candidate.generation_rank,
                item.candidate.candidate_version,
                item.candidate.qbank_id,
            ),
        )
        if self.config.is_v36:
            opportunity_families: dict[str | None, str] = {}
            allocated_final = []
            for item in ordered_final:
                opportunity_id = item.candidate.opportunity_id
                family_key = item.gate_result.metadata["duplicate_key"]
                previous_family = opportunity_families.setdefault(opportunity_id, family_key)
                if previous_family != family_key:
                    terminal[item.candidate.qbank_id] = self._rejected(
                        item, [CandidateRejectionCode.BUDGET_EXCEEDED]
                    )
                else:
                    allocated_final.append(item)
            ordered_final = allocated_final
        if ordered_final:
            deduplicated = self.gate.evaluate(
                run_id=built.unit.run_id,
                snapshot_id=built.unit.snapshot_id,
                unit=built.unit,
                generation_id=generation.generation_id,
                model_call_id=generation.model_call_id,
                drafts=[internal_by_qbank[item.candidate.qbank_id] for item in ordered_final],
                memories=list(built.memories),
                known_primaries=known_primaries,
                created_at=generation.created_at,
                opportunity_set=built.opportunities,
                generation_ranks=[item.candidate.generation_rank for item in ordered_final],
                producer_model_call_ids=[
                    item.candidate.producer_model_call_id or generation.model_call_id
                    for item in ordered_final
                ],
                candidate_created_ats=[item.candidate.created_at for item in ordered_final],
            )
            terminal.update({item.candidate.qbank_id: item for item in deduplicated})

        terminal_values = tuple(
            sorted(
                terminal.values(),
                key=lambda item: (
                    item.candidate.generation_rank,
                    item.candidate.candidate_version,
                    item.candidate.qbank_id,
                ),
            )
        )
        accepted_ranks = {
            item.candidate.generation_rank
            for item in terminal_values
            if item.gate_result.status is CandidateGateStatus.ELIGIBLE_FOR_PLANNING
        }
        completed_at = utc_now()
        stage = CompilationStageResult(
            stage_result_id=stable_compiler_id(
                "compiler-stage",
                [generation.run_id, generation.compiler_unit_id, "questioner"],
            ),
            run_id=generation.run_id,
            compiler_unit_id=generation.compiler_unit_id,
            stage=CompilationStage.QUESTIONER,
            attempt_number=generation.attempt_number,
            status=TraceStatus.SUCCESS,
            produced_count=generation.raw_candidate_count,
            accepted_count=len(accepted_ranks),
            rejected_count=generation.raw_candidate_count - len(accepted_ranks),
            result_hash=canonical_content_hash(
                {
                    "generation": generation.model_dump(mode="json"),
                    "gates": [item.gate_result.model_dump(mode="json") for item in terminal_values],
                    "verifications": [item.model_dump(mode="json") for item in verifications],
                    "supersessions": [item.model_dump(mode="json") for item in supersessions],
                }
            ),
            started_at=generation_trace.created_at,
            completed_at=completed_at,
            elapsed_ms=max(
                0.0,
                (completed_at - generation_trace.created_at).total_seconds() * 1000,
            ),
            metadata={
                "generation_id": generation.generation_id,
                **(
                    {
                        "questioner_funnel_v1": self._native_funnel(
                            built,
                            payload,
                            resolved,
                            resolution_diagnostics,
                            static,
                            final_candidates,
                            terminal_values,
                        )
                    }
                    if self.config.is_v36
                    else {}
                ),
                "verification_batch_count": len(verification_batches),
                "revised_count": len(supersessions),
                "rescued_count": sum(
                    item.gate_result.status is CandidateGateStatus.ELIGIBLE_FOR_PLANNING
                    and item.candidate.candidate_version == 1
                    for item in terminal_values
                ),
                "verifier_conservative_rejection_count": sum(
                    int(batch.metadata.get("verifier_conservative_rejection_count", 0))
                    for batch in verification_batches
                ),
            },
        )
        result = QuestionerUnitResult(
            generation=generation,
            candidates=terminal_values,
            stage_result=stage,
            verification_batches=tuple(verification_batches),
            verifications=tuple(verifications),
            supersessions=tuple(supersessions),
            answer_claims=tuple(answer_claims),
        )
        try:
            await self.persist_result(result)
        except QuestionerStageError:
            raise
        except Exception as exc:
            raise QuestionerStageError("terminal_persistence", exc) from exc
        return result

    async def _terminal_v33_generation_rejection(
        self,
        built: BuiltCompilationUnit,
        job: ExperimentJob,
    ) -> QuestionerUnitResult:
        """Close an archived invalid final envelope without accepting any candidate."""

        model_call_id = questioner_model_call_id(
            run_id=job.run_id,
            compiler_unit_id=built.unit.compiler_unit_id,
            attempt_number=job.attempt_count,
        )
        trace = await self.load_model_call(model_call_id)
        if not _is_structured_output_failure(trace):
            raise QuestionerStageError(
                "generation",
                QuestionerGenerationContractError("contract_invalid"),
            )
        assert trace is not None
        if built.opportunities is None:
            raise QuestionerStageError(
                "generation",
                QuestionerGenerationContractError("contract_invalid"),
            )
        contract_code = trace.metadata.get("output_contract_error_code")
        if not isinstance(contract_code, str) or contract_code not in {
            "questioner_output_truncated", "native_json_required", "native_envelope_invalid",
        }:
            contract_code = None
        terminal_diagnostics = (
            {
                "output_contract_error_code": contract_code,
                "model_output_truncated": contract_code == "questioner_output_truncated",
                "declared_empty": False,
                "scientific_acceptance": False,
                "terminal_outcome": "generation_contract_failure",
            }
            if self.config.is_v36
            else {}
        )
        output_hash = canonical_content_hash(
            [
                trace.model_call_id,
                trace.output_asset_id,
                _TERMINAL_GENERATION_REASON,
            ]
        )
        generation = QuestionerGeneration(
            generation_id=stable_compiler_id(
                "questioner-generation",
                [job.run_id, built.unit.compiler_unit_id, trace.model_call_id],
            ),
            run_id=job.run_id,
            compiler_unit_id=built.unit.compiler_unit_id,
            job_id=job.job_id,
            model_call_id=trace.model_call_id,
            attempt_number=job.attempt_count,
            prompt_version=self.config.prompt_version,
            renderer_version=self.config.renderer_version,
            parser_version=self.config.parser_version,
            prompt_hash=questioner_prompt_sha256(self.config.prompt_version),
            input_hash=built.unit.input_hash,
            output_hash=output_hash,
            parse_status=QuestionerParseStatus.FAILED,
            raw_candidate_count=0,
            valid_candidate_count=0,
            invalid_candidate_count=0,
            repair_attempt_count=0,
            created_at=trace.created_at,
            metadata={
                "request_prompt_hash": trace.prompt_hash,
                "response_schema_hash": questioner_response_schema_sha256(
                    self.config.prompt_version
                ),
                "invalid_diagnostics": [],
                "resolution_diagnostics": [],
                "input_tokens": built.input_tokens,
                "model_input_tokens": trace.input_tokens or 0,
                "model_output_tokens": trace.output_tokens or 0,
                "model_total_tokens": trace.total_tokens or 0,
                "opportunity_set_id": (built.opportunities.opportunity_set.opportunity_set_id),
                "native_structured_output": False,
                "provider_repair_attempt_count": trace.repair_attempt_count,
                **(
                    {"provider_item_count_status": "unavailable_invalid_envelope"}
                    if self.config.is_v36
                    else {}
                ),
                "response_artifact_ready": True,
                "terminal_reason": _TERMINAL_GENERATION_REASON,
                **terminal_diagnostics,
            },
        )
        try:
            await self.persist_v3_seed(built, generation, [], [])
        except QuestionerStageError:
            raise
        except Exception as exc:
            raise QuestionerStageError("seed_persistence", exc) from exc

        completed_at = utc_now()
        stage = CompilationStageResult(
            stage_result_id=stable_compiler_id(
                "compiler-stage",
                [generation.run_id, generation.compiler_unit_id, "questioner"],
            ),
            run_id=generation.run_id,
            compiler_unit_id=generation.compiler_unit_id,
            stage=CompilationStage.QUESTIONER,
            attempt_number=generation.attempt_number,
            status=TraceStatus.SUCCESS,
            produced_count=0,
            accepted_count=0,
            rejected_count=0,
            result_hash=canonical_content_hash(
                {
                    "generation": generation.model_dump(mode="json"),
                    "terminal_reason": _TERMINAL_GENERATION_REASON,
                }
            ),
            started_at=trace.created_at,
            completed_at=completed_at,
            elapsed_ms=max(
                0.0,
                (completed_at - trace.created_at).total_seconds() * 1000,
            ),
            metadata={
                "generation_id": generation.generation_id,
                "native_structured_output": False,
                "terminal_reason": _TERMINAL_GENERATION_REASON,
                "verification_batch_count": 0,
                **terminal_diagnostics,
                **(
                    {
                        "questioner_funnel_v1": {
                            "counts": {
                                "provider_items": None,
                                "schema_native": 0,
                                "alias_resolved": 0,
                                "static_pass": 0,
                                "verifier_pass": 0,
                                "dedup_eligible": 0,
                            },
                            "items": [],
                            "provider_item_count_status": "unavailable_invalid_envelope",
                        }
                    }
                    if self.config.is_v36
                    else {}
                ),
            },
        )
        result = QuestionerUnitResult(
            generation=generation,
            candidates=(),
            stage_result=stage,
        )
        try:
            await self.persist_result(result)
        except QuestionerStageError:
            raise
        except Exception as exc:
            raise QuestionerStageError("terminal_persistence", exc) from exc
        print(
            "QUESTIONER_TERMINAL_GENERATION "
            + json.dumps(
                {
                    "action": "continue_to_r0_gate",
                    "attempt_number": job.attempt_count,
                    "error_code": "questioner_generation_structured_output_invalid",
                    "max_attempts": job.max_attempts,
                    "output_artifact_present": True,
                    "parse_error_present": True,
                    "unit_ordinal": built.unit.ordinal,
                    **terminal_diagnostics,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return result

    async def _v3_generation_trace(
        self, built: BuiltCompilationUnit, job: ExperimentJob
    ) -> ModelCallTrace:
        for attempt in range(job.attempt_count, 0, -1):
            call_id = questioner_model_call_id(
                run_id=job.run_id,
                compiler_unit_id=built.unit.compiler_unit_id,
                attempt_number=attempt,
            )
            existing = await self.load_model_call(call_id)
            if existing is not None and existing.status is TraceStatus.SUCCESS:
                if existing.parsed_output is None:
                    raise RuntimeError("successful Questioner call has no parsed output")
                return existing
        model_call_id = questioner_model_call_id(
            run_id=job.run_id,
            compiler_unit_id=built.unit.compiler_unit_id,
            attempt_number=job.attempt_count,
        )
        current = await self.load_model_call(model_call_id)
        if current is not None:
            if _is_structured_output_failure(current):
                raise QuestionerGenerationContractError("structured_output_invalid")
            raise RuntimeError("current Questioner attempt already has a failed model call")
        request = built.request.model_copy(
            update={
                "request_id": model_call_id,
                "metadata": {
                    **built.request.metadata,
                    "job_attempt_number": job.attempt_count,
                },
            }
        )
        try:
            audited = await self.execution.generate_audited(
                self.gateway,
                request,
                ModelCallContext(
                    model_call_id=model_call_id,
                    run_id=job.run_id,
                    snapshot_id=job.snapshot_id,
                    conversation_id=built.unit.conversation_id,
                    job_id=job.job_id,
                    compiler_unit_id=built.unit.compiler_unit_id,
                    stage=AppStage.QUESTIONER,
                ),
                output_parser=questioner_output_parser(self.config),
            )
        except ValueError as exc:
            failed_trace = await self.load_model_call(model_call_id)
            diagnostic_code = (
                "structured_output_invalid"
                if _is_structured_output_failure(failed_trace)
                else "contract_invalid"
            )
            raise QuestionerGenerationContractError(diagnostic_code) from exc
        return audited.trace

    def _v3_generation_fact(
        self,
        *,
        built: BuiltCompilationUnit,
        job: ExperimentJob,
        trace: ModelCallTrace,
        payload: dict[str, Any],
        valid_count: int,
        invalid_count: int,
        resolution_diagnostics: list[dict[str, object]],
    ) -> QuestionerGeneration:
        raw_count = int(payload.get("raw_candidate_count", valid_count + invalid_count))
        if invalid_count:
            parse_status = QuestionerParseStatus.PARTIAL
        elif trace.repair_attempt_count:
            parse_status = QuestionerParseStatus.REPAIRED
        else:
            parse_status = QuestionerParseStatus.PARSED
        return QuestionerGeneration(
            generation_id=stable_compiler_id(
                "questioner-generation",
                [job.run_id, built.unit.compiler_unit_id, trace.model_call_id],
            ),
            run_id=job.run_id,
            compiler_unit_id=built.unit.compiler_unit_id,
            job_id=job.job_id,
            model_call_id=trace.model_call_id,
            attempt_number=max(
                1,
                int(
                    dict(trace.request_config.get("request_metadata", {})).get(
                        "job_attempt_number", job.attempt_count
                    )
                ),
            ),
            prompt_version=self.config.prompt_version,
            renderer_version=self.config.renderer_version,
            parser_version=self.config.parser_version,
            prompt_hash=questioner_prompt_sha256(self.config.prompt_version),
            input_hash=built.unit.input_hash,
            output_hash=canonical_content_hash(payload),
            parse_status=parse_status,
            raw_candidate_count=raw_count,
            valid_candidate_count=valid_count,
            invalid_candidate_count=invalid_count,
            repair_attempt_count=trace.repair_attempt_count,
            created_at=trace.created_at,
            metadata={
                "request_prompt_hash": trace.prompt_hash,
                "response_schema_hash": questioner_response_schema_sha256(
                    self.config.prompt_version
                ),
                "invalid_diagnostics": payload.get("invalid_diagnostics", []),
                "resolution_diagnostics": resolution_diagnostics,
                "input_tokens": built.input_tokens,
                "model_input_tokens": trace.input_tokens or 0,
                "model_output_tokens": trace.output_tokens or 0,
                "model_total_tokens": trace.total_tokens or 0,
                "opportunity_set_id": built.opportunities.opportunity_set.opportunity_set_id
                if built.opportunities is not None
                else None,
            },
        )

    async def _run_v3_verifier(
        self,
        *,
        built: BuiltCompilationUnit,
        job: ExperimentJob,
        generation: QuestionerGeneration,
        inputs: list[tuple[str, QuestionerCandidateDraftV3]],
        verification_round: int,
    ) -> tuple[
        ModelCallTrace,
        QuestionerVerificationBatchV1,
        dict[str, object],
    ]:
        if not inputs or built.opportunities is None:
            raise ValueError("Questioner verifier requires candidates and opportunities")
        hardened_scope = self.config.is_v32 or self.config.is_v33 or self.config.is_v34
        if hardened_scope and len(inputs) != 1:
            raise ValueError("hardened Questioner verifier requires one candidate scope")
        candidate_scope = inputs[0][0] if hardened_scope else None

        def conservative_result(
            failed_trace: ModelCallTrace,
        ) -> tuple[
            ModelCallTrace,
            QuestionerVerificationBatchV1,
            dict[str, object],
        ]:
            source_alias, source = inputs[0]
            output = QuestionerVerificationBatchV1.model_validate(
                {
                    "results": [
                        {
                            "candidate_alias": source_alias,
                            "verdict": "reject",
                            "reason_codes": ["evidence_insufficient"],
                            "replacement": None,
                        }
                    ]
                }
            )
            return (
                failed_trace,
                output,
                {
                    "candidate_scope": source_alias,
                    "scope_hash": canonical_content_hash(
                        [
                            source.opportunity_id,
                            source.evidence_aliases,
                            source_alias,
                        ]
                    ),
                    "diagnostic_codes": ["provider_output_invalid"],
                    "fallback_aliases": [source_alias],
                    "provider_output_fallback": True,
                    "output_contract_error_code": failed_trace.metadata.get(
                        "output_contract_error_code"
                    ),
                    "native_structured_output": False,
                },
            )

        trace: ModelCallTrace | None = None
        for attempt in range(job.attempt_count, 0, -1):
            call_id = questioner_verifier_model_call_id(
                run_id=job.run_id,
                compiler_unit_id=built.unit.compiler_unit_id,
                generation_id=generation.generation_id,
                verification_round=verification_round,
                attempt_number=attempt,
                candidate_scope=candidate_scope,
            )
            existing = await self.load_model_call(call_id)
            if existing is not None and existing.status is TraceStatus.SUCCESS:
                trace = existing
                break
        if trace is None:
            call_id = questioner_verifier_model_call_id(
                run_id=job.run_id,
                compiler_unit_id=built.unit.compiler_unit_id,
                generation_id=generation.generation_id,
                verification_round=verification_round,
                attempt_number=job.attempt_count,
                candidate_scope=candidate_scope,
            )
            current = await self.load_model_call(call_id)
            if current is not None:
                if hardened_scope and _is_structured_output_failure(current):
                    return conservative_result(current)
                raise RuntimeError("current verifier attempt already has a failed call")
            request = render_questioner_verifier_request(
                request_id=call_id,
                model_alias=self.config.compiler_model_alias,
                unit_id=built.unit.compiler_unit_id,
                cutoff_session_id=built.unit.available_after_session,
                memories=list(built.memories),
                sessions=list(built.sessions),
                opportunity_set=built.opportunities,
                candidates=inputs,
                final_round=verification_round == 1,
                max_output_tokens=self.config.reserved_output_tokens,
                timeout_seconds=built.request.timeout_seconds,
                questioner_prompt_version=self.config.prompt_version,
                questioner_renderer_version=self.config.renderer_version,
                verifier_prompt_version=self.config.verifier_prompt_version,
                memory_raw_image_policy=self.config.memory_raw_image_policy,
            )
            expected_aliases = [alias for alias, _ in inputs]
            try:
                audited = await self.execution.generate_audited(
                    self.gateway,
                    request,
                    ModelCallContext(
                        model_call_id=call_id,
                        run_id=job.run_id,
                        snapshot_id=job.snapshot_id,
                        conversation_id=built.unit.conversation_id,
                        job_id=job.job_id,
                        compiler_unit_id=built.unit.compiler_unit_id,
                        parent_model_call_id=generation.model_call_id,
                        stage=AppStage.QUESTIONER,
                    ),
                    output_parser=(
                        (
                            questioner_verifier_v7_output_parser(
                                expected_alias=expected_aliases[0],
                                candidate=inputs[0][1],
                            )
                            if self.config.is_v36
                            else questioner_verifier_v6_output_parser(
                                expected_alias=expected_aliases[0],
                                candidate=inputs[0][1],
                            )
                            if self.config.is_v35
                            else questioner_verifier_v5_output_parser(
                                expected_alias=expected_aliases[0],
                                candidate=inputs[0][1],
                            )
                            if self.config.is_v34
                            else questioner_verifier_v3_output_parser(
                                expected_alias=expected_aliases[0]
                            )
                        )
                        if hardened_scope
                        else questioner_verifier_output_parser(
                            expected_aliases=expected_aliases,
                            schema_version=self.config.verifier_prompt_version,
                        )
                    ),
                )
                trace = audited.trace
            except Exception:
                trace = await self.load_model_call(call_id)
                if not hardened_scope or not _is_structured_output_failure(trace):
                    raise
                assert trace is not None
                return conservative_result(trace)
        if trace.parsed_output is None:
            raise RuntimeError("successful Questioner verifier has no parsed output")
        parsed_payload = dict(trace.parsed_output)
        raw_parser_metadata = parsed_payload.pop("_parser_metadata", {})
        parser_metadata = (
            dict(raw_parser_metadata) if isinstance(raw_parser_metadata, Mapping) else {}
        )
        if hardened_scope:
            envelope = QuestionerVerificationEnvelopeV3.model_validate(parsed_payload)
            source_alias, source = inputs[0]
            item = envelope.result
            applicable = self._v31_applicable_dimensions(source, item.dimensions)
            failed = {name for name, passed in applicable.items() if not passed}
            if not failed:
                output = QuestionerVerificationBatchV1.model_validate(
                    {
                        "results": [
                            {
                                "candidate_alias": source_alias,
                                "verdict": "pass",
                                "reason_codes": [],
                                "replacement": None,
                            }
                        ]
                    }
                )
            else:
                # v3.4 taxonomy is server-derived, so its reviser may change wording only.
                reasons = _questioner_v3_reason_codes(failed)
                revisable_dimensions = (
                    {"user_facing"} if self.config.is_v34 else {"user_facing", "taxonomy_correct"}
                )
                revisable = (
                    verification_round == 0
                    and failed <= revisable_dimensions
                    and self.config.semantic_revision_max == 1
                )
                replacement: QuestionerCandidateDraftV3 | None = None
                if revisable:
                    revision_trace, replacement = await self._run_v32_revision(
                        built=built,
                        job=job,
                        generation=generation,
                        verifier_trace=trace,
                        candidate_alias=source_alias,
                        candidate=source,
                        failed_dimensions=sorted(failed),
                        reason_codes=[value.value for value in reasons],
                    )
                    parser_metadata["replacement_model_call_ids"] = {
                        source_alias: revision_trace.model_call_id
                    }
                    parser_metadata["replacement_created_at"] = {
                        source_alias: revision_trace.created_at.isoformat()
                    }
                    parser_metadata["replacement_repair_attempt_counts"] = {
                        source_alias: revision_trace.repair_attempt_count
                    }
                    if replacement is None:
                        parser_metadata["replacement_provider_fallback_aliases"] = [source_alias]
                output = QuestionerVerificationBatchV1.model_validate(
                    {
                        "results": [
                            {
                                "candidate_alias": source_alias,
                                "verdict": "revise" if replacement is not None else "reject",
                                "reason_codes": [value.value for value in reasons],
                                "replacement": (
                                    replacement.model_dump(mode="json")
                                    if replacement is not None
                                    else None
                                ),
                            }
                        ]
                    }
                )
            parser_metadata.update(
                {
                    "candidate_scope": source_alias,
                    "scope_hash": canonical_content_hash(
                        [
                            source.opportunity_id,
                            source.evidence_aliases,
                            source_alias,
                        ]
                    ),
                    "failed_dimensions": sorted(failed),
                    "provider_output_fallback": False,
                }
            )
        elif self.config.is_v31:
            raw_output = QuestionerVerificationBatchV2.model_validate(parsed_payload)
            output = self._derive_v31_verdicts(
                raw_output,
                inputs=inputs,
                final_round=verification_round == 1,
            )
        else:
            output = QuestionerVerificationBatchV1.model_validate(parsed_payload)
        expected_aliases = [alias for alias, _ in inputs]
        actual_aliases = [item.candidate_alias for item in output.results]
        if actual_aliases != expected_aliases:
            raise ValueError("Questioner verifier aliases do not match the request order")
        return trace, output, parser_metadata

    async def _run_v32_revision(
        self,
        *,
        built: BuiltCompilationUnit,
        job: ExperimentJob,
        generation: QuestionerGeneration,
        verifier_trace: ModelCallTrace,
        candidate_alias: str,
        candidate: QuestionerCandidateDraftV3,
        failed_dimensions: list[str],
        reason_codes: list[str],
    ) -> tuple[ModelCallTrace, QuestionerCandidateDraftV3 | None]:
        trace: ModelCallTrace | None = None
        for attempt in range(job.attempt_count, 0, -1):
            call_id = questioner_revision_model_call_id(
                run_id=job.run_id,
                compiler_unit_id=built.unit.compiler_unit_id,
                generation_id=generation.generation_id,
                candidate_scope=candidate_alias,
                attempt_number=attempt,
            )
            existing = await self.load_model_call(call_id)
            if existing is None:
                continue
            if existing.status is TraceStatus.SUCCESS or _is_structured_output_failure(existing):
                trace = existing
                break
            if attempt == job.attempt_count:
                raise RuntimeError("current Questioner revision attempt already has a failed call")
        if trace is None:
            call_id = questioner_revision_model_call_id(
                run_id=job.run_id,
                compiler_unit_id=built.unit.compiler_unit_id,
                generation_id=generation.generation_id,
                candidate_scope=candidate_alias,
                attempt_number=job.attempt_count,
            )
            request = render_questioner_revision_request(
                request_id=call_id,
                model_alias=self.config.compiler_model_alias,
                candidate_alias=candidate_alias,
                candidate=candidate,
                failed_dimensions=failed_dimensions,
                reason_codes=reason_codes,
                max_output_tokens=self.config.reserved_output_tokens,
                timeout_seconds=built.request.timeout_seconds,
                wording_only=self.config.is_v34,
            )
            try:
                audited = await self.execution.generate_audited(
                    self.gateway,
                    request,
                    ModelCallContext(
                        model_call_id=call_id,
                        run_id=job.run_id,
                        snapshot_id=job.snapshot_id,
                        conversation_id=built.unit.conversation_id,
                        job_id=job.job_id,
                        compiler_unit_id=built.unit.compiler_unit_id,
                        parent_model_call_id=verifier_trace.model_call_id,
                        stage=AppStage.QUESTIONER,
                    ),
                    output_parser=questioner_revision_output_parser(
                        expected_alias=candidate_alias,
                        wording_only=self.config.is_v34,
                    ),
                )
                trace = audited.trace
            except Exception:
                trace = await self.load_model_call(call_id)
                if not _is_structured_output_failure(trace):
                    raise
                assert trace is not None
                return trace, None
        if _is_structured_output_failure(trace):
            assert trace is not None
            return trace, None
        if trace.status is not TraceStatus.SUCCESS or trace.parsed_output is None:
            raise RuntimeError("Questioner revision did not produce recoverable output")
        if (self.config.is_v33 or self.config.is_v34) and trace.repair_attempt_count:
            return trace, None
        if self.config.is_v34:
            replacement_v2 = QuestionerRevisionEnvelopeV2.model_validate(
                trace.parsed_output
            ).replacement
            generated_question = replacement_v2.generated_question
            paraphrases = replacement_v2.paraphrases
            question_type = candidate.question_type
            reasoning_type = candidate.reasoning_type
        else:
            replacement_v1 = QuestionerRevisionEnvelopeV1.model_validate(
                trace.parsed_output
            ).replacement
            generated_question = replacement_v1.generated_question
            paraphrases = replacement_v1.paraphrases
            question_type = replacement_v1.question_type
            reasoning_type = replacement_v1.reasoning_type
        return (
            trace,
            QuestionerCandidateDraftV3(
                candidate_kind=candidate.candidate_kind,
                opportunity_id=candidate.opportunity_id,
                generated_question=generated_question,
                question_type=question_type,
                reasoning_type=reasoning_type,
                expected_answer=candidate.expected_answer,
                evidence_aliases=candidate.evidence_aliases,
                required_modalities=candidate.required_modalities,
                paraphrases=paraphrases,
                confidence=candidate.confidence,
            ),
        )

    @staticmethod
    def _v31_applicable_dimensions(
        candidate: QuestionerCandidateDraftV3,
        dimensions: QuestionerVerificationDimensionsV2,
    ) -> dict[str, bool]:
        values = dimensions.model_dump()
        visual = any(item.value == "image" for item in candidate.required_modalities)
        multi_hop = candidate.reasoning_type.value == "multi_hop"
        if not visual:
            values.pop("visual_necessary")
        if not multi_hop:
            values.pop("multi_hop_genuine")
        return values

    def _derive_v31_verdicts(
        self,
        output: QuestionerVerificationBatchV2,
        *,
        inputs: list[tuple[str, QuestionerCandidateDraftV3]],
        final_round: bool,
    ) -> QuestionerVerificationBatchV1:
        by_alias = dict(inputs)
        normalized: list[dict[str, object]] = []
        hard_dimension_names = {
            "answer_explicitly_entailed",
            "evidence_sufficient",
            "metadata_free",
            "visual_necessary",
            "multi_hop_genuine",
        }
        for item in output.results:
            source = by_alias[item.candidate_alias]
            applicable = self._v31_applicable_dimensions(source, item.dimensions)
            failed = {name for name, passed in applicable.items() if not passed}
            if not failed:
                normalized.append(
                    {
                        "candidate_alias": item.candidate_alias,
                        "verdict": QuestionerVerificationVerdict.PASS,
                        "reason_codes": [],
                        "replacement": None,
                    }
                )
                continue

            replacement = item.replacement
            replacement_preserves_private_facts = bool(
                replacement is not None
                and replacement.opportunity_id == source.opportunity_id
                and replacement.expected_answer == source.expected_answer
                and replacement.evidence_aliases == source.evidence_aliases
            )
            revisable = (
                not final_round
                and item.requested_action is QuestionerVerificationVerdict.REVISE
                and replacement_preserves_private_facts
                and not failed.intersection(hard_dimension_names)
            )
            reasons = list(item.reason_codes)
            if not reasons:
                reasons = [QuestionerVerificationReason.NOT_ANSWERABLE]
            replacement_payload = (
                replacement.model_dump(mode="json")
                if revisable and replacement is not None
                else None
            )
            normalized.append(
                {
                    "candidate_alias": item.candidate_alias,
                    "verdict": (
                        QuestionerVerificationVerdict.REVISE
                        if revisable
                        else QuestionerVerificationVerdict.REJECT
                    ),
                    "reason_codes": reasons,
                    "replacement": replacement_payload,
                }
            )
        return QuestionerVerificationBatchV1.model_validate({"results": normalized})

    def _apply_initial_verifier(
        self,
        *,
        built: BuiltCompilationUnit,
        generation: QuestionerGeneration,
        trace: ModelCallTrace,
        output: QuestionerVerificationBatchV1,
        static_inputs: list[GatedQuestionCandidate],
        provider_by_rank: dict[int, QuestionerCandidateDraftV3],
        internal_by_qbank: dict[str, QuestionerCandidateDraft],
        terminal: dict[str, GatedQuestionCandidate],
        final_candidates: list[GatedQuestionCandidate],
        revised_static_pass: list[tuple[GatedQuestionCandidate, QuestionerCandidateDraftV3]],
        supersessions: list[QuestionerCandidateSupersession],
        parser_metadata: Mapping[str, object] | None = None,
    ) -> tuple[QuestionerVerificationBatch, list[QuestionerCandidateVerification]]:
        if built.opportunities is None:
            raise ValueError("Questioner v3 revision requires opportunities")
        replacement_call_ids_raw = (parser_metadata or {}).get("replacement_model_call_ids", {})
        replacement_call_ids = (
            dict(replacement_call_ids_raw) if isinstance(replacement_call_ids_raw, Mapping) else {}
        )
        item_by_alias = {
            self._candidate_alias(item.candidate.generation_rank): item for item in static_inputs
        }
        normalized: list[
            tuple[
                GatedQuestionCandidate,
                QuestionerVerificationVerdict,
                list[QuestionerVerificationReason],
                str | None,
                dict[str, object],
            ]
        ] = []
        for verdict in output.results:
            source = item_by_alias[verdict.candidate_alias]
            if verdict.verdict is QuestionerVerificationVerdict.PASS:
                final_candidates.append(source)
                normalized.append((source, verdict.verdict, [], None, {}))
                continue
            if verdict.verdict is QuestionerVerificationVerdict.REJECT:
                terminal[source.candidate.qbank_id] = self._rejected(
                    source,
                    [
                        CandidateRejectionCode.SEMANTIC_VERIFICATION_FAILED,
                        *self._verification_rejection_codes(verdict.reason_codes),
                    ],
                    verifier_reason_codes=verdict.reason_codes,
                )
                normalized.append((source, verdict.verdict, verdict.reason_codes, None, {}))
                continue

            assert verdict.replacement is not None
            rank = source.candidate.generation_rank
            try:
                if verdict.replacement.opportunity_id != provider_by_rank[rank].opportunity_id:
                    raise ValueError("replacement changed its opportunity")
                replacement_draft = resolve_v3_draft(
                    verdict.replacement,
                    opportunity_set=built.opportunities,
                    candidate_origin=QuestionerCandidateOrigin.REVISED,
                    candidate_version=1,
                    supersedes_qbank_id=source.candidate.qbank_id,
                )
                replacement = self.gate.evaluate(
                    run_id=built.unit.run_id,
                    snapshot_id=built.unit.snapshot_id,
                    unit=built.unit,
                    generation_id=generation.generation_id,
                    model_call_id=trace.model_call_id,
                    drafts=[replacement_draft],
                    memories=list(built.memories),
                    known_primaries={},
                    created_at=trace.created_at,
                    opportunity_set=built.opportunities,
                    generation_ranks=[rank],
                    producer_model_call_ids=[
                        str(replacement_call_ids.get(verdict.candidate_alias, trace.model_call_id))
                    ],
                    apply_dedupe=False,
                )[0]
            except ValueError:
                terminal[source.candidate.qbank_id] = self._rejected(
                    source,
                    [
                        CandidateRejectionCode.SEMANTIC_VERIFICATION_FAILED,
                        CandidateRejectionCode.REVISION_EXHAUSTED,
                    ],
                    verifier_reason_codes=verdict.reason_codes,
                )
                normalized.append(
                    (
                        source,
                        QuestionerVerificationVerdict.REJECT,
                        verdict.reason_codes,
                        None,
                        {"provider_verdict": "revise", "replacement_invalid": True},
                    )
                )
                continue
            internal_by_qbank[replacement.candidate.qbank_id] = replacement_draft
            terminal[source.candidate.qbank_id] = self._rejected(
                source,
                [CandidateRejectionCode.SEMANTIC_REVISION_REQUIRED],
                verifier_reason_codes=verdict.reason_codes,
            )
            supersession = QuestionerCandidateSupersession(
                supersession_id=stable_compiler_id(
                    "questioner-supersession",
                    [source.candidate.qbank_id, replacement.candidate.qbank_id],
                ),
                run_id=generation.run_id,
                compiler_unit_id=generation.compiler_unit_id,
                generation_id=generation.generation_id,
                source_qbank_id=source.candidate.qbank_id,
                replacement_qbank_id=replacement.candidate.qbank_id,
                revision_round=1,
                content_hash=canonical_content_hash(
                    [source.candidate.qbank_id, replacement.candidate.qbank_id, 1]
                ),
                created_at=trace.created_at,
            )
            supersessions.append(supersession)
            if replacement.gate_result.status is CandidateGateStatus.REJECTED:
                terminal[replacement.candidate.qbank_id] = self._rejected(
                    replacement,
                    [CandidateRejectionCode.REVISION_EXHAUSTED],
                )
            else:
                revised_static_pass.append((replacement, verdict.replacement))
            normalized.append(
                (
                    source,
                    verdict.verdict,
                    verdict.reason_codes,
                    replacement.candidate.qbank_id,
                    {},
                )
            )
        return self._verification_facts(
            generation=generation,
            trace=trace,
            verification_round=0,
            values=normalized,
            parser_metadata=parser_metadata or {},
        )

    def _apply_final_verifier(
        self,
        *,
        generation: QuestionerGeneration,
        trace: ModelCallTrace,
        output: QuestionerVerificationBatchV1,
        revised_inputs: list[tuple[GatedQuestionCandidate, QuestionerCandidateDraftV3]],
        terminal: dict[str, GatedQuestionCandidate],
        final_candidates: list[GatedQuestionCandidate],
        parser_metadata: Mapping[str, object] | None = None,
    ) -> tuple[QuestionerVerificationBatch, list[QuestionerCandidateVerification]]:
        by_alias = {
            self._candidate_alias(item.candidate.generation_rank): item
            for item, _ in revised_inputs
        }
        normalized: list[
            tuple[
                GatedQuestionCandidate,
                QuestionerVerificationVerdict,
                list[QuestionerVerificationReason],
                str | None,
                dict[str, object],
            ]
        ] = []
        for verdict in output.results:
            candidate = by_alias[verdict.candidate_alias]
            if verdict.verdict is QuestionerVerificationVerdict.PASS:
                final_candidates.append(candidate)
                normalized.append((candidate, verdict.verdict, [], None, {}))
                continue
            rejection_codes = [
                CandidateRejectionCode.REVISION_EXHAUSTED,
                *self._verification_rejection_codes(verdict.reason_codes),
            ]
            terminal[candidate.candidate.qbank_id] = self._rejected(
                candidate,
                rejection_codes,
                verifier_reason_codes=verdict.reason_codes,
            )
            normalized.append(
                (
                    candidate,
                    QuestionerVerificationVerdict.REJECT,
                    verdict.reason_codes,
                    None,
                    {"provider_verdict": verdict.verdict.value},
                )
            )
        return self._verification_facts(
            generation=generation,
            trace=trace,
            verification_round=1,
            values=normalized,
            parser_metadata=parser_metadata or {},
        )

    def _verification_facts(
        self,
        *,
        generation: QuestionerGeneration,
        trace: ModelCallTrace,
        verification_round: int,
        values: list[
            tuple[
                GatedQuestionCandidate,
                QuestionerVerificationVerdict,
                list[QuestionerVerificationReason],
                str | None,
                dict[str, object],
            ]
        ],
        parser_metadata: Mapping[str, object],
    ) -> tuple[QuestionerVerificationBatch, list[QuestionerCandidateVerification]]:
        scope_qbank_id = (
            values[0][0].candidate.qbank_id
            if (self.config.is_v32 or self.config.is_v33 or self.config.is_v34) and len(values) == 1
            else None
        )
        batch_identity: list[object] = [
            generation.generation_id,
            verification_round,
        ]
        if scope_qbank_id is not None:
            batch_identity.append(scope_qbank_id)
        batch_id = stable_compiler_id(
            "questioner-verification-batch",
            batch_identity,
        )
        raw_fallback_aliases = parser_metadata.get("fallback_aliases", [])
        fallback_aliases = (
            {alias for alias in raw_fallback_aliases if isinstance(alias, str)}
            if isinstance(raw_fallback_aliases, list)
            else set()
        )
        raw_diagnostic_codes = parser_metadata.get("diagnostic_codes", [])
        diagnostic_codes = (
            [code for code in raw_diagnostic_codes if isinstance(code, str)]
            if isinstance(raw_diagnostic_codes, list)
            else []
        )
        verifications = [
            QuestionerCandidateVerification(
                candidate_verification_id=stable_compiler_id(
                    "questioner-candidate-verification",
                    [batch_id, item.candidate.qbank_id],
                ),
                verification_batch_id=batch_id,
                run_id=generation.run_id,
                compiler_unit_id=generation.compiler_unit_id,
                generation_id=generation.generation_id,
                qbank_id=item.candidate.qbank_id,
                candidate_alias=self._candidate_alias(item.candidate.generation_rank),
                verification_round=verification_round,
                verdict=verdict,
                reason_codes=reasons,
                replacement_qbank_id=replacement,
                content_hash=canonical_content_hash(
                    [
                        batch_id,
                        item.candidate.qbank_id,
                        verdict.value,
                        [reason.value for reason in reasons],
                        replacement,
                        *(
                            [parser_metadata.get("support_assessment")]
                            if self.config.is_v36
                            else []
                        ),
                    ]
                ),
                created_at=trace.created_at,
                metadata={
                    **metadata,
                    **(
                        {
                            "support_contract_version": "questioner-exact-support-v1",
                            "support_assessment": parser_metadata.get("support_assessment"),
                        }
                        if self.config.is_v36
                        else {}
                    ),
                    **(
                        {
                            "provider_output_fallback": True,
                            "provider_output_diagnostic_codes": diagnostic_codes,
                        }
                        if self._candidate_alias(item.candidate.generation_rank) in fallback_aliases
                        else {}
                    ),
                },
            )
            for item, verdict, reasons, replacement, metadata in values
        ]
        pass_count = sum(
            item.verdict is QuestionerVerificationVerdict.PASS for item in verifications
        )
        revise_count = sum(
            item.verdict is QuestionerVerificationVerdict.REVISE for item in verifications
        )
        reject_count = len(verifications) - pass_count - revise_count
        raw_replacement_calls = parser_metadata.get("replacement_model_call_ids")
        replacement_calls = (
            {str(key): value for key, value in raw_replacement_calls.items()}
            if isinstance(raw_replacement_calls, Mapping)
            else {}
        )
        raw_repair_counts = parser_metadata.get("replacement_repair_attempt_counts")
        replacement_repair_counts = (
            {str(key): value for key, value in raw_repair_counts.items()}
            if isinstance(raw_repair_counts, Mapping)
            else {}
        )
        raw_replacement_fallbacks = parser_metadata.get("replacement_provider_fallback_aliases")
        replacement_provider_fallback_aliases = (
            sorted(value for value in raw_replacement_fallbacks if isinstance(value, str))
            if isinstance(raw_replacement_fallbacks, list)
            else []
        )
        batch = QuestionerVerificationBatch(
            verification_batch_id=batch_id,
            run_id=generation.run_id,
            compiler_unit_id=generation.compiler_unit_id,
            generation_id=generation.generation_id,
            model_call_id=trace.model_call_id,
            scope_qbank_id=scope_qbank_id,
            verification_round=verification_round,
            prompt_version=self.config.verifier_prompt_version,
            schema_hash=questioner_response_schema_sha256(self.config.verifier_prompt_version),
            input_count=len(verifications),
            pass_count=pass_count,
            revise_count=revise_count,
            reject_count=reject_count,
            content_hash=canonical_content_hash(
                [item.model_dump(mode="json") for item in verifications]
            ),
            created_at=trace.created_at,
            metadata={
                "repair_attempt_count": trace.repair_attempt_count,
                "model_input_tokens": trace.input_tokens or 0,
                "model_output_tokens": trace.output_tokens or 0,
                "model_total_tokens": trace.total_tokens or 0,
                "provider_output_normalized": bool(diagnostic_codes),
                "provider_output_diagnostic_codes": diagnostic_codes,
                "verifier_conservative_rejection_count": len(fallback_aliases),
                "candidate_scope_hash": parser_metadata.get("scope_hash"),
                "single_candidate_scope": scope_qbank_id is not None,
                "provider_output_fallback": bool(
                    parser_metadata.get("provider_output_fallback", False)
                ),
                "replacement_model_call_ids": replacement_calls,
                "replacement_repair_attempt_counts": replacement_repair_counts,
                "replacement_provider_fallback_aliases": (replacement_provider_fallback_aliases),
                **(
                    {"support_assessment": parser_metadata.get("support_assessment")}
                    if self.config.is_v36
                    else {}
                ),
            },
        )
        return batch, verifications

    @staticmethod
    def _candidate_alias(generation_rank: int) -> str:
        return f"Q{generation_rank:02d}"

    @staticmethod
    def _verification_rejection_codes(
        reasons: list[QuestionerVerificationReason],
    ) -> list[CandidateRejectionCode]:
        mapping = {
            QuestionerVerificationReason.NOT_USER_FACING: (
                CandidateRejectionCode.ASSISTANT_FOLLOW_UP
            ),
            QuestionerVerificationReason.NOT_FUTURE_PLAUSIBLE: (
                CandidateRejectionCode.HISTORICAL_QUESTION_REPLAY
            ),
            QuestionerVerificationReason.NOT_ANSWERABLE: (
                CandidateRejectionCode.EVIDENCE_NOT_ENTAILED
            ),
            QuestionerVerificationReason.ANSWER_UNSUPPORTED: (
                CandidateRejectionCode.EVIDENCE_NOT_ENTAILED
            ),
            QuestionerVerificationReason.EVIDENCE_INSUFFICIENT: (
                CandidateRejectionCode.EVIDENCE_NOT_ENTAILED
            ),
            QuestionerVerificationReason.TAXONOMY_MISMATCH: (
                CandidateRejectionCode.TAXONOMY_INVARIANT_FAILED
            ),
            QuestionerVerificationReason.MODALITY_MISMATCH: (
                CandidateRejectionCode.MODALITY_MISMATCH
            ),
            QuestionerVerificationReason.VISUAL_NOT_NEEDED: (
                CandidateRejectionCode.MODALITY_MISMATCH
            ),
            QuestionerVerificationReason.MULTI_HOP_NOT_GENUINE: (
                CandidateRejectionCode.MULTI_HOP_EVIDENCE_MISSING
            ),
            QuestionerVerificationReason.TOO_GENERIC: (
                CandidateRejectionCode.GENERIC_EXTERNAL_QUESTION
            ),
            QuestionerVerificationReason.HISTORICAL_REPLAY: (
                CandidateRejectionCode.HISTORICAL_QUESTION_REPLAY
            ),
            QuestionerVerificationReason.INTERNAL_METADATA: (
                CandidateRejectionCode.SYSTEM_METADATA_REFERENCE
            ),
            QuestionerVerificationReason.UNSAFE_LEAKAGE: (
                CandidateRejectionCode.INTERNAL_IDENTIFIER_LEAK
            ),
        }
        return list(dict.fromkeys(mapping[item] for item in reasons))

    @staticmethod
    def _native_funnel(
        built, payload, resolved, resolution_diagnostics, static, verified, terminal
    ):
        """One cumulative funnel row per provider item, including early losses."""
        from pmmc.compiler.questioner_native import native_support_catalog

        catalog = native_support_catalog(built.opportunities, built.memories)
        scopes = {
            scope["provider_ordinal"]: scope for scope in payload.get("provider_item_scopes", [])
        }
        stages = (
            "provider_items",
            "schema_native",
            "alias_resolved",
            "static_pass",
            "verifier_pass",
            "dedup_eligible",
        )
        schema_ranks = set(payload.get("candidate_ordinals", []))
        resolved_by_rank = {rank: provider for rank, provider, _ in resolved}
        static_by_rank = {item.candidate.generation_rank: item for item in static}
        verified_ranks = {item.candidate.generation_rank for item in verified}
        terminal_by_rank = {item.candidate.generation_rank: item for item in terminal}
        resolution_failures = {
            item["generation_rank"]: item.get("reason_code", "alias_resolution_failed")
            for item in resolution_diagnostics
        }
        invalid = {
            int(item["ordinal"]) + 1: item for item in payload.get("invalid_diagnostics", [])
        }
        opportunities = {
            local_opportunity_id(item): item for item in built.opportunities.opportunities
        }
        rows = []
        for rank in range(1, int(payload.get("raw_candidate_count", 0)) + 1):
            provider = resolved_by_rank.get(rank)
            initial = static_by_rank.get(rank)
            final = terminal_by_rank.get(rank)
            passed = {
                "provider_items": True,
                "schema_native": rank in schema_ranks,
                "alias_resolved": provider is not None,
                "static_pass": initial is not None
                and initial.gate_result.status is CandidateGateStatus.ELIGIBLE_FOR_PLANNING,
                "verifier_pass": rank in verified_ranks,
                "dedup_eligible": final is not None
                and final.gate_result.status is CandidateGateStatus.ELIGIBLE_FOR_PLANNING,
            }
            reason = None
            if rank in invalid:
                errors = invalid[rank].get("errors", [])
                reason = errors[0].get("type", "schema_invalid") if errors else "schema_invalid"
            elif rank in resolution_failures:
                reason = resolution_failures[rank]
            elif final is not None and final.gate_result.rejection_codes:
                reason = final.gate_result.rejection_codes[0].value
            scope = scopes.get(rank, {})
            opportunity = opportunities.get(
                provider.opportunity_id if provider else scope.get("opportunity_alias")
            )
            source_roles = (
                sorted({claim.support.speaker for claim in provider.exact_claims})
                if provider
                else sorted(
                    {
                        catalog[alias].speaker
                        for alias in scope.get("support_aliases", [])
                        if alias in catalog
                    }
                )
            )
            rows.append(
                {
                    "provider_ordinal": rank,
                    "dataset": built.memories[0].dataset.value,
                    "compiler_unit_id": built.unit.compiler_unit_id,
                    "opportunity_type": opportunity.opportunity_type.value
                    if opportunity
                    else "unknown",
                    "source_roles": source_roles or ["unknown"],
                    "first_failure_reason": reason,
                    **passed,
                }
            )
        return {
            "counts": {stage: sum(row[stage] for row in rows) for stage in stages},
            "items": rows,
            "allocation": [
                {
                    "opportunity_id": local_opportunity_id(item),
                    "opportunity_type": item.opportunity_type.value,
                    "budget": 1,
                }
                for item in allocate_opportunities(built.opportunities)
            ],
        }

    def _v33_answer_claim_facts(
        self,
        *,
        generation: QuestionerGeneration,
        static: list[GatedQuestionCandidate],
        resolved_by_rank: Mapping[int, object],
        opportunity_set,
    ) -> list[QuestionerAnswerClaim]:
        if not (self.config.is_v33 or self.config.is_v34):
            return []
        facts: list[QuestionerAnswerClaim] = []
        for gated in static:
            resolved = resolved_by_rank.get(gated.candidate.generation_rank)
            if resolved is None:
                continue
            claims = getattr(resolved, "claims", ())
            aliases = getattr(resolved, "evidence_aliases", ())
            support_aliases = getattr(resolved, "claim_support_aliases", aliases)
            for ordinal, (claim, alias, support_alias) in enumerate(
                zip(claims, aliases, support_aliases, strict=True)
            ):
                evidence = opportunity_set.alias_to_evidence[alias]
                identity = [
                    gated.candidate.qbank_id,
                    ordinal,
                    claim.claim_kind.value,
                    canonical_content_hash(support_alias),
                    evidence.evidence_id,
                    claim.claim_text,
                    claim.support_quote,
                ]
                if self.config.is_v36:
                    identity.append(claim.model_dump(mode="json"))
                facts.append(
                    QuestionerAnswerClaim(
                        answer_claim_id=stable_compiler_id(
                            "questioner-answer-claim",
                            identity,
                        ),
                        run_id=generation.run_id,
                        compiler_unit_id=generation.compiler_unit_id,
                        generation_id=generation.generation_id,
                        qbank_id=gated.candidate.qbank_id,
                        ordinal=ordinal,
                        claim_kind=claim.claim_kind,
                        evidence_alias_hash=canonical_content_hash(support_alias),
                        evidence_id=evidence.evidence_id,
                        claim_text=claim.claim_text,
                        support_quote=claim.support_quote,
                        quote_verified=(
                            claim.claim_kind.value == "text" if self.config.is_v36 else True
                        ),
                        content_hash=canonical_content_hash(identity),
                        created_at=generation.created_at,
                        metadata={
                            **(
                                {
                                    "native_support": claim.support.model_dump(mode="json"),
                                    "subject": claim.subject,
                                    "assertion": claim.assertion,
                                    "span_exists": True,
                                    "claim_entailed": "not_evaluated",
                                    "speaker_subject_aligned": "not_evaluated",
                                    "answer_question_closed": "not_evaluated",
                                    "quote_verified_semantics": "exact_source_range_only",
                                    "support_contract_version": "questioner-exact-support-v1",
                                }
                                if self.config.is_v36
                                else {}
                            ),
                            "support_alias_kind": (
                                "sentence_span"
                                if support_alias in opportunity_set.sentence_span_aliases
                                else "image"
                            ),
                        },
                    )
                )
        return facts

    @staticmethod
    def _rejected(
        value: GatedQuestionCandidate,
        codes: list[CandidateRejectionCode],
        *,
        verifier_reason_codes: list[QuestionerVerificationReason] | None = None,
    ) -> GatedQuestionCandidate:
        merged = list(dict.fromkeys([*value.gate_result.rejection_codes, *codes]))
        gate = value.gate_result.model_copy(
            update={
                "status": CandidateGateStatus.REJECTED,
                "rejection_codes": merged,
                "metadata": {
                    **value.gate_result.metadata,
                    **(
                        {"verifier_reason_codes": [item.value for item in verifier_reason_codes]}
                        if verifier_reason_codes is not None
                        else {}
                    ),
                },
            }
        )
        return GatedQuestionCandidate(
            candidate=value.candidate,
            expected_evidence=value.expected_evidence,
            gate_result=gate,
            duplicate_link=value.duplicate_link,
        )

    async def _load_unit(self, state: QuestionerGraphState) -> dict[str, Any]:
        built = state["built"]
        job = state["job"]
        if (
            job.compiler_unit_id != built.unit.compiler_unit_id
            or job.run_id != built.unit.run_id
            or job.snapshot_id != built.unit.snapshot_id
        ):
            raise ValueError("Questioner job does not match its compilation unit")
        w_catalog = (
            built.opportunities is not None
            and built.opportunities.opportunity_set.builder_version == EVIDENCE_FIRST_VERSION
        )
        w_request = built.request.metadata.get("allocation_version") == EVIDENCE_FIRST_VERSION
        if self.config.evidence_first != w_catalog or self.config.evidence_first != w_request:
            raise PreparedQuestionerInputError("questioner_allocation_policy_mismatch")
        if self.config.evidence_first:
            # Fail before looking up a saved generation: a native v3.6 DTO
            # is not proof of the treatment that prepared this run/unit.
            identity = built.unit.metadata.get("builder_identity", {})
            if identity.get("questioner_config_hash") != questioner_config_hash(self.config):
                raise PreparedQuestionerInputError("evidence_first_unit_config_mismatch")
            contract = questioner_request_contract_hash(
                prompt_version=self.config.prompt_version,
                renderer_version=self.config.renderer_version,
                opportunity_builder_version=self.config.opportunity_builder_version,
            )
            if identity.get("request_contract_hash") != contract:
                raise PreparedQuestionerInputError("evidence_first_unit_contract_mismatch")
            assert built.opportunities is not None  # checked by w_catalog above
            header = built.opportunities.opportunity_set
            if (header.run_id, header.snapshot_id, header.conversation_id) != (
                built.unit.run_id, built.unit.snapshot_id, built.unit.conversation_id
            ):
                raise PreparedQuestionerInputError("evidence_first_catalog_unit_mismatch")
            expected = render_questioner_request(
                request_id=f"questioner-{built.unit.compiler_unit_id}",
                model_alias=self.config.compiler_model_alias, unit_id=built.unit.compiler_unit_id,
                cutoff_session_id=built.unit.available_after_session, memories=list(built.memories),
                sessions=list(built.sessions), max_output_tokens=self.config.reserved_output_tokens,
                timeout_seconds=built.request.timeout_seconds,
                prompt_version=self.config.prompt_version,
                renderer_version=self.config.renderer_version, opportunity_set=built.opportunities,
                memory_raw_image_policy=self.config.memory_raw_image_policy,
            )
            if expected != built.request or not (
                0 < built.input_tokens <= self.config.max_input_tokens
            ):
                raise PreparedQuestionerInputError("evidence_first_prepared_request_mismatch")
        return {}

    async def _render(self, state: QuestionerGraphState) -> dict[str, Any]:
        return {}

    async def _generate(self, state: QuestionerGraphState) -> dict[str, Any]:
        built = state["built"]
        job = state["job"]
        successful = await self.load_successful_call(job.job_id)
        if successful is None:
            model_call_id = questioner_model_call_id(
                run_id=job.run_id,
                compiler_unit_id=built.unit.compiler_unit_id,
                attempt_number=job.attempt_count,
            )
            request = built.request.model_copy(
                update={
                    "request_id": model_call_id,
                    "metadata": {
                        **built.request.metadata,
                        "job_attempt_number": job.attempt_count,
                    },
                }
            )
            audited = await self.execution.generate_audited(
                self.gateway,
                request,
                ModelCallContext(
                    model_call_id=model_call_id,
                    run_id=job.run_id,
                    snapshot_id=job.snapshot_id,
                    conversation_id=built.unit.conversation_id,
                    job_id=job.job_id,
                    compiler_unit_id=built.unit.compiler_unit_id,
                    stage=AppStage.QUESTIONER,
                ),
                output_parser=questioner_output_parser(self.config),
            )
            successful = audited.trace
        if successful.parsed_output is None:
            raise RuntimeError("successful Questioner call has no recoverable parsed output")
        return {"trace": successful}

    async def _parse(self, state: QuestionerGraphState) -> dict[str, Any]:
        trace = state["trace"]
        payload = dict(trace.parsed_output or {})
        drafts = [
            QuestionerCandidateDraft.model_validate(item) for item in payload.get("candidates", [])
        ]
        raw_count = int(payload.get("raw_candidate_count", len(drafts)))
        invalid_count = int(payload.get("invalid_candidate_count", 0))
        if len(drafts) + invalid_count != raw_count:
            raise ValueError("recovered Questioner candidate counts do not reconcile")
        if invalid_count:
            parse_status = QuestionerParseStatus.PARTIAL
        elif trace.repair_attempt_count:
            parse_status = QuestionerParseStatus.REPAIRED
        else:
            parse_status = QuestionerParseStatus.PARSED
        built = state["built"]
        job = state["job"]
        generation = QuestionerGeneration(
            generation_id=stable_compiler_id(
                "questioner-generation",
                [job.run_id, built.unit.compiler_unit_id, trace.model_call_id],
            ),
            run_id=job.run_id,
            compiler_unit_id=built.unit.compiler_unit_id,
            job_id=job.job_id,
            model_call_id=trace.model_call_id,
            attempt_number=max(
                1,
                int(
                    dict(trace.request_config.get("request_metadata", {})).get(
                        "job_attempt_number", job.attempt_count
                    )
                ),
            ),
            prompt_version=self.config.prompt_version,
            renderer_version=self.config.renderer_version,
            parser_version=self.config.parser_version,
            prompt_hash=questioner_prompt_sha256(self.config.prompt_version),
            input_hash=built.unit.input_hash,
            output_hash=canonical_content_hash(payload),
            parse_status=parse_status,
            raw_candidate_count=raw_count,
            valid_candidate_count=len(drafts),
            invalid_candidate_count=invalid_count,
            repair_attempt_count=trace.repair_attempt_count,
            created_at=trace.created_at,
            metadata={
                "request_prompt_hash": trace.prompt_hash,
                "invalid_diagnostics": payload.get("invalid_diagnostics", []),
                "input_tokens": built.input_tokens,
                **(
                    {
                        "response_schema_hash": questioner_response_schema_sha256(
                            self.config.prompt_version
                        )
                    }
                    if self.config.is_v2
                    else {}
                ),
            },
        )
        return {"generation": generation, "drafts": drafts}

    async def _lineage_gate(self, state: QuestionerGraphState) -> dict[str, Any]:
        built = state["built"]
        primaries = await self.before_gate(built)
        return {"known_primaries": primaries}

    async def _deduplicate(self, state: QuestionerGraphState) -> dict[str, Any]:
        built = state["built"]
        generation = state["generation"]
        gated = self.gate.evaluate(
            run_id=built.unit.run_id,
            snapshot_id=built.unit.snapshot_id,
            unit=built.unit,
            generation_id=generation.generation_id,
            model_call_id=generation.model_call_id,
            drafts=state["drafts"],
            memories=list(built.memories),
            known_primaries=state["known_primaries"],
            created_at=generation.created_at,
        )
        return {"gated": gated}

    async def _persist(self, state: QuestionerGraphState) -> dict[str, Any]:
        generation = state["generation"]
        gated = state["gated"]
        accepted = sum(
            item.gate_result.status is CandidateGateStatus.ELIGIBLE_FOR_PLANNING for item in gated
        )
        rejected = generation.invalid_candidate_count + len(gated) - accepted
        completed_at = utc_now()
        trace = state["trace"]
        started_at = trace.created_at
        stage = CompilationStageResult(
            stage_result_id=stable_compiler_id(
                "compiler-stage",
                [generation.run_id, generation.compiler_unit_id, "questioner"],
            ),
            run_id=generation.run_id,
            compiler_unit_id=generation.compiler_unit_id,
            stage=CompilationStage.QUESTIONER,
            attempt_number=generation.attempt_number,
            status=TraceStatus.SUCCESS,
            produced_count=generation.raw_candidate_count,
            accepted_count=accepted,
            rejected_count=rejected,
            result_hash=canonical_content_hash(
                {
                    "generation": generation.model_dump(mode="json"),
                    "gates": [item.gate_result.model_dump(mode="json") for item in gated],
                }
            ),
            started_at=started_at,
            completed_at=completed_at,
            elapsed_ms=max(0.0, (completed_at - started_at).total_seconds() * 1000),
            metadata={"generation_id": generation.generation_id},
        )
        result = QuestionerUnitResult(generation, tuple(gated), stage)
        await self.persist_result(result)
        return {"result": result}


class QuestionerRunner:
    """Generate Questioner facts concurrently while committing gates in unit order."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        execution: ModelExecutionService,
        gateway: GenerationGateway,
        error_artifacts: ExperimentArtifactService,
        config: QuestionerConfig,
        lease_seconds: int,
        concurrency: int = 4,
    ) -> None:
        if lease_seconds < 3:
            raise ValueError("Questioner job lease must be at least three seconds")
        if concurrency <= 0:
            raise ValueError("Questioner concurrency must be positive")
        self.sessions = sessions
        self.execution = execution
        self.gateway = gateway
        self.error_artifacts = error_artifacts
        self.config = config
        self.lease_seconds = lease_seconds
        self.concurrency = concurrency
        self._gate_condition = asyncio.Condition()
        self._pending_gate_ordinals: set[int] = set()
        self._progress_lock = asyncio.Lock()
        self._progress_started = time.monotonic()
        self._progress_last_sample_at = self._progress_started
        self._progress_last_sample_completed = 0
        self._progress_last_sample_new_units = 0
        self._progress_last_sample_tokens = 0
        self._progress_last_completion_at: float | None = None
        self._worker_limit = _questioner_worker_limit_from_environment()
        self._uncapped_worker_count = concurrency
        self._worker_count = concurrency
        self._progress: dict[str, int] = {
            "units": 0,
            "reused": 0,
            "failed": 0,
            "opportunities": 0,
            "raw": 0,
            "schema_valid": 0,
            "static_pass": 0,
            "verifier_pass": 0,
            "generation_native_envelopes": 0,
            "generation_terminal_rejections": 0,
            "verifier_native_batches": 0,
            "verifier_fallback_batches": 0,
            "verifier_conservative_rejections": 0,
            "revised": 0,
            "rescued": 0,
            "final_eligible": 0,
            "visual": 0,
            "multi_hop": 0,
            "zero_unit": 0,
            "model_tokens": 0,
            "workers_busy": 0,
            "gate_waiters": 0,
        }
        self._failure_stages: Counter[str] = Counter()
        self._failure_codes: Counter[str] = Counter()
        self._failure_operations: Counter[str] = Counter()
        self._failure_database: Counter[str] = Counter()
        self._failure_attempts = 0

    async def execute(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        units: list[BuiltCompilationUnit] | None = None,
        prepared_units: Sequence[CompilationUnit] | None = None,
        prepared_links: Mapping[str, Sequence[CompilationUnitLink]] | None = None,
        input_provider: QuestionerInputProvider | None = None,
        on_item: ItemCallback | None = None,
        on_progress: ProgressCallback | None = None,
        progress_interval_seconds: float = 30.0,
    ) -> QuestionerRunResult:
        if units is not None and prepared_units is not None:
            raise ValueError("Questioner runner accepts built or prepared units, not both")
        built_units = list(units or [])
        unit_values = (
            [item.unit for item in built_units] if units is not None else list(prepared_units or [])
        )
        if not unit_values:
            raise ValueError("Questioner run requires at least one compilation unit")
        by_id = {item.unit.compiler_unit_id: item for item in built_units}
        unit_by_id = {item.compiler_unit_id: item for item in unit_values}
        if len(unit_by_id) != len(unit_values):
            raise ValueError("Questioner units contain duplicate IDs")
        if [item.ordinal for item in unit_values] != list(range(len(unit_values))):
            raise ValueError("Questioner unit ordinals must be contiguous")
        if units is None and input_provider is None:
            raise ValueError("prepared Questioner units require an input provider")
        if progress_interval_seconds <= 0:
            raise ValueError("Questioner progress interval must be positive")

        self._progress_started = time.monotonic()
        self._progress_last_sample_at = self._progress_started
        self._progress_last_sample_completed = 0
        self._progress_last_sample_new_units = 0
        self._progress_last_sample_tokens = 0
        self._progress_last_completion_at = None
        self._uncapped_worker_count = min(
            len(unit_values),
            self.concurrency * (_V3_WORKER_MULTIPLIER if self.config.is_v3 else 1),
        )
        self._worker_count = min(
            self._uncapped_worker_count,
            self._worker_limit or self._uncapped_worker_count,
        )
        async with self._progress_lock:
            for key in self._progress:
                self._progress[key] = 0
            self._failure_stages.clear()
            self._failure_codes.clear()
            self._failure_operations.clear()
            self._failure_database.clear()
            self._failure_attempts = 0

        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None and uow.runs is not None
            assert uow.compiler_units is not None and uow.questioner is not None
            pending: set[int] = set()
            for unit in unit_values:
                await uow.compiler_units.add(unit)
                links = (
                    list(by_id[unit.compiler_unit_id].links)
                    if unit.compiler_unit_id in by_id
                    else list((prepared_links or {}).get(unit.compiler_unit_id, ()))
                )
                await uow.questioner.add_links(links)
                stage = await uow.compiler_units.get_stage_result(
                    unit.compiler_unit_id, CompilationStage.QUESTIONER
                )
                if stage is None:
                    pending.add(unit.ordinal)
                else:
                    self._progress["reused"] += 1
                job_id = stable_compiler_id(
                    "compile-job", [run_id, unit.compiler_unit_id, "questions"]
                )
                expected = ExperimentJob(
                    job_id=job_id,
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    stage=AppStage.COMPILE,
                    variant="questions",
                    priority=len(unit_values) - unit.ordinal,
                    max_attempts=self.config.job_max_attempts,
                    metadata={"unit_ordinal": unit.ordinal},
                )
                existing = await uow.jobs.get(job_id)
                if existing is None:
                    await uow.jobs.add(expected)
                elif self._immutable_job(existing) != self._immutable_job(expected):
                    raise ValueError(f"persisted Questioner job conflicts: {job_id}")
            state = await uow.runs.get_state(run_id)
            if state.status is RunStatus.CREATED:
                await uow.runs.transition(run_id, RunStatus.RUNNING)
            elif state.status not in {RunStatus.RUNNING, RunStatus.COMPLETED}:
                raise RuntimeError(f"Questioner run cannot resume from {state.status.value}")
        self._pending_gate_ordinals = pending

        graph = QuestionerCompilationGraph(
            execution=self.execution,
            gateway=self.gateway,
            config=self.config,
            load_successful_call=self._load_successful_call,
            load_model_call=self._load_model_call,
            before_gate=self._before_gate,
            persist_v3_seed=self._persist_v3_seed,
            persist_result=self._persist_result,
        )

        stop_progress = asyncio.Event()

        async def report_progress() -> None:
            if on_progress is None:
                return
            while not stop_progress.is_set():
                try:
                    await asyncio.wait_for(stop_progress.wait(), timeout=progress_interval_seconds)
                except TimeoutError:
                    on_progress(await self._progress_snapshot(len(unit_values)))

        async def worker(index: int) -> None:
            worker_id = f"questioner-{index}"
            while True:
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    assert uow.jobs is not None
                    job = await uow.jobs.claim(
                        worker_id,
                        self.lease_seconds,
                        run_id=run_id,
                        stage=AppStage.COMPILE,
                    )
                    counts = (
                        await uow.jobs.status_counts(run_id, stage=AppStage.COMPILE)
                        if job is None
                        else None
                    )
                if job is None:
                    assert counts is not None
                    if counts.get(JobStatus.PENDING, 0) + counts.get(JobStatus.RUNNING, 0):
                        await asyncio.sleep(0.1)
                        continue
                    return
                unit = unit_by_id.get(job.compiler_unit_id or "")
                if unit is None:
                    error = QuestionerStageError(
                        "unit_lookup", KeyError("compilation unit is absent")
                    )
                    failed = await self._finish_failed(job, worker_id, error)
                    await self._record_failure_attempt(
                        error,
                        unit_ordinal=int(job.metadata.get("unit_ordinal", 0)),
                        attempt_number=job.attempt_count,
                        exhausted=failed.status is JobStatus.FAILED,
                    )
                    if failed.status is JobStatus.FAILED:
                        await self._complete_gate_turn(int(job.metadata.get("unit_ordinal", 0)))
                    continue
                built = by_id.get(unit.compiler_unit_id)
                if built is None:
                    assert input_provider is not None
                    try:
                        built = await input_provider.load(unit)
                    except Exception as exc:
                        error = QuestionerStageError("input_load", exc)
                        failed = await self._finish_failed(job, worker_id, error)
                        await self._record_failure_attempt(
                            error,
                            unit_ordinal=unit.ordinal,
                            attempt_number=job.attempt_count,
                            exhausted=failed.status is JobStatus.FAILED,
                        )
                        if failed.status is JobStatus.FAILED:
                            await self._complete_gate_turn(unit.ordinal)
                        continue
                stop = asyncio.Event()
                heartbeat = asyncio.create_task(self._heartbeat(job.job_id, worker_id, stop))
                await self._change_busy(1)
                try:
                    async with ExperimentUnitOfWork(self.sessions) as uow:
                        assert uow.compiler_units is not None
                        existing_stage = await uow.compiler_units.get_stage_result(
                            built.unit.compiler_unit_id, CompilationStage.QUESTIONER
                        )
                    if existing_stage is None:
                        result = await graph.process(built, job)
                        await self._record_progress(built, result)
                    async with ExperimentUnitOfWork(self.sessions) as uow:
                        assert uow.jobs is not None
                        await uow.jobs.finish(job.job_id, worker_id, success=True)
                    await self._complete_gate_turn(built.unit.ordinal)
                except Exception as exc:
                    error = (
                        exc
                        if isinstance(exc, QuestionerStageError)
                        else QuestionerStageError("pipeline", exc)
                    )
                    failed = await self._finish_failed(job, worker_id, error)
                    await self._record_failure_attempt(
                        error,
                        unit_ordinal=built.unit.ordinal,
                        attempt_number=job.attempt_count,
                        exhausted=failed.status is JobStatus.FAILED,
                    )
                    if failed.status is JobStatus.FAILED:
                        await self._complete_gate_turn(built.unit.ordinal)
                    if on_item and failed.status is JobStatus.FAILED:
                        on_item(
                            built.unit.compiler_unit_id,
                            False,
                            failed.error_summary or type(error).__name__,
                        )
                else:
                    if on_item:
                        on_item(built.unit.compiler_unit_id, True, None)
                finally:
                    await self._change_busy(-1)
                    stop.set()
                    heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await heartbeat

        progress_task = asyncio.create_task(report_progress())
        try:
            await asyncio.gather(*(worker(index) for index in range(self._worker_count)))
        finally:
            stop_progress.set()
            await progress_task
        if on_progress is not None:
            on_progress(await self._progress_snapshot(len(unit_values)))
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None
            counts = await uow.jobs.status_counts(run_id, stage=AppStage.COMPILE)
        succeeded = counts.get(JobStatus.SUCCEEDED, 0)
        failed = counts.get(JobStatus.FAILED, 0)
        if failed:
            async with self._progress_lock:
                failure_stages = dict(sorted(self._failure_stages.items()))
                failure_codes = dict(sorted(self._failure_codes.items()))
                failure_operations = dict(sorted(self._failure_operations.items()))
                failure_database = dict(sorted(self._failure_database.items()))
                failure_attempts = self._failure_attempts
            safe_details = (
                f"stages={json.dumps(failure_stages, sort_keys=True, separators=(',', ':'))}; "
                f"codes={json.dumps(failure_codes, sort_keys=True, separators=(',', ':'))}; "
                f"operations="
                f"{json.dumps(failure_operations, sort_keys=True, separators=(',', ':'))}; "
                f"database={json.dumps(failure_database, sort_keys=True, separators=(',', ':'))}; "
                f"attempts={failure_attempts}"
            )
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.runs is not None
                await uow.runs.transition(
                    run_id,
                    RunStatus.FAILED,
                    error_type="QuestionerJobFailure",
                    error_summary=(
                        f"one or more Questioner jobs exhausted their attempts; {safe_details}"
                    ),
                )
            raise RuntimeError(f"{failed} Questioner compilation job(s) failed; {safe_details}")
        if succeeded != len(unit_values):
            raise RuntimeError("Questioner job denominator is incomplete")
        if self.config.is_v33 or self.config.is_v34:
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.questioner is not None
                await uow.questioner.rebuild_candidate_families(run_id)
        return QuestionerRunResult(run_id, len(unit_values), succeeded, failed)

    async def _change_busy(self, amount: int) -> None:
        async with self._progress_lock:
            self._progress["workers_busy"] = max(0, self._progress["workers_busy"] + amount)

    async def _record_failure_attempt(
        self,
        error: BaseException,
        *,
        unit_ordinal: int,
        attempt_number: int,
        exhausted: bool,
    ) -> None:
        diagnostics = safe_database_error_diagnostics(error)
        database_locator: dict[str, object] = {
            key: diagnostics[key]
            for key in (
                "sqlstate",
                "schema_name",
                "table_name",
                "column_name",
                "constraint_name",
            )
            if key in diagnostics
        }
        persistence_locator: dict[str, object] = {
            key: diagnostics[key]
            for key in (
                "persistence_operation",
                "persistence_candidate_count",
                "persistence_revised_candidate_count",
                "persistence_verification_batch_count",
                "persistence_supersession_count",
                "persistence_verification_input_count",
                "persistence_verification_pass_count",
                "persistence_verification_revise_count",
                "persistence_verification_reject_count",
                "persistence_verification_provider_fallback",
                "persistence_verification_single_candidate_scope",
            )
            if key in diagnostics
        }
        event: dict[str, object] = {
            "attempt_number": attempt_number,
            "error_code": diagnostics.get("error_code", type(error).__name__),
            "exhausted": exhausted,
            "failure_stage": diagnostics.get("failure_stage", "unknown"),
            "unit_ordinal": unit_ordinal,
        }
        event.update(persistence_locator)
        event.update(database_locator)
        for key in (
            "provider_error_code", "provider_status_code", "provider_error_category",
            "provider_retryable", "requested_output_tokens", "provider_output_tokens",
            "provider_output_ceiling",
        ):
            if key in diagnostics:
                event[key] = diagnostics[key]
        print(
            "QUESTIONER_FAILURE " + json.dumps(event, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        async with self._progress_lock:
            self._failure_attempts += 1
            if database_locator:
                locator_key = json.dumps(
                    database_locator,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                self._failure_database[locator_key] += 1
            if exhausted:
                self._progress["failed"] += 1
                self._failure_stages[diagnostics.get("failure_stage", "unknown")] += 1
                persistence_operation = diagnostics.get("persistence_operation")
                if persistence_operation is not None:
                    self._failure_operations[persistence_operation] += 1
                self._failure_codes[
                    diagnostics.get(
                        "provider_error_code",
                        diagnostics.get(
                            "error_code",
                            diagnostics.get("sqlstate", type(error).__name__),
                        ),
                    )
                ] += 1

    async def _record_progress(
        self,
        built: BuiltCompilationUnit,
        result: QuestionerUnitResult,
    ) -> None:
        eligible = result.stage_result.accepted_count
        visual = sum(
            item.gate_result.status is CandidateGateStatus.ELIGIBLE_FOR_PLANNING
            and Modality.IMAGE in item.candidate.required_modalities
            for item in result.candidates
        )
        multi_hop = sum(
            item.gate_result.status is CandidateGateStatus.ELIGIBLE_FOR_PLANNING
            and item.candidate.hop_count >= 2
            for item in result.candidates
        )
        async with self._progress_lock:
            self._progress["units"] += 1
            self._progress["generation_native_envelopes"] += int(
                result.generation.parse_status is not QuestionerParseStatus.FAILED
                and result.generation.repair_attempt_count == 0
            )
            self._progress["generation_terminal_rejections"] += int(
                result.generation.parse_status is QuestionerParseStatus.FAILED
                and result.generation.metadata.get("terminal_reason") == _TERMINAL_GENERATION_REASON
            )
            self._progress["opportunities"] += len(
                built.opportunities.opportunities if built.opportunities is not None else ()
            )
            self._progress["raw"] += result.generation.raw_candidate_count
            self._progress["schema_valid"] += result.generation.valid_candidate_count
            self._progress["static_pass"] += sum(
                item.verification_round == 0 for item in result.verifications
            )
            self._progress["verifier_pass"] += sum(
                item.verdict is QuestionerVerificationVerdict.PASS for item in result.verifications
            )
            self._progress["verifier_native_batches"] += sum(
                not bool(item.metadata.get("provider_output_fallback", False))
                for item in result.verification_batches
            )
            self._progress["verifier_fallback_batches"] += sum(
                bool(item.metadata.get("provider_output_fallback", False))
                for item in result.verification_batches
            )
            self._progress["verifier_conservative_rejections"] += sum(
                int(item.metadata.get("verifier_conservative_rejection_count", 0))
                for item in result.verification_batches
            )
            self._progress["revised"] += len(result.supersessions)
            self._progress["rescued"] += int(result.stage_result.metadata.get("rescued_count", 0))
            self._progress["final_eligible"] += eligible
            self._progress["visual"] += visual
            self._progress["multi_hop"] += multi_hop
            self._progress["zero_unit"] += eligible == 0
            self._progress["model_tokens"] += int(
                result.generation.metadata.get("model_total_tokens", 0)
            ) + sum(
                int(item.metadata.get("model_total_tokens", 0))
                for item in result.verification_batches
            )
            self._progress_last_completion_at = time.monotonic()

    async def _progress_snapshot(self, total_units: int) -> dict[str, object]:
        async with self._progress_lock:
            now = time.monotonic()
            counters = dict(self._progress)
            failure_stages = dict(sorted(self._failure_stages.items()))
            failure_codes = dict(sorted(self._failure_codes.items()))
            failure_operations = dict(sorted(self._failure_operations.items()))
            failure_database = dict(sorted(self._failure_database.items()))
            failure_attempts = self._failure_attempts
            completed = counters["units"] + counters["reused"] + counters["failed"]
            sample_elapsed = max(0.001, now - self._progress_last_sample_at)
            recent_completed = max(0, completed - self._progress_last_sample_completed)
            recent_new_units = max(
                0,
                counters["units"] - self._progress_last_sample_new_units,
            )
            recent_tokens = max(
                0,
                counters["model_tokens"] - self._progress_last_sample_tokens,
            )
            last_completion_seconds = (
                None
                if self._progress_last_completion_at is None
                else round(max(0.0, now - self._progress_last_completion_at), 3)
            )
            self._progress_last_sample_at = now
            self._progress_last_sample_completed = completed
            self._progress_last_sample_new_units = counters["units"]
            self._progress_last_sample_tokens = counters["model_tokens"]
        values: dict[str, object] = dict(counters)
        elapsed = max(0.001, now - self._progress_started)
        queued = max(0, total_units - completed - counters["workers_busy"])
        values.update(
            {
                "total_units": total_units,
                "completed_units": completed,
                "remaining_units": max(0, total_units - completed),
                "queued_units": queued,
                "workers_total": self._worker_count,
                "workers_uncapped": self._uncapped_worker_count,
                "worker_limit": self._worker_limit,
                "workers_avoided": max(
                    0,
                    self._uncapped_worker_count - self._worker_count,
                ),
                "workers_processing": max(
                    0,
                    counters["workers_busy"] - counters["gate_waiters"],
                ),
                "elapsed_seconds": round(elapsed, 3),
                "tokens_per_second": round(counters["model_tokens"] / elapsed, 3),
                "recent_tokens_per_second": round(recent_tokens / sample_elapsed, 3),
                "recent_units_per_minute": round(
                    recent_new_units * 60.0 / sample_elapsed,
                    3,
                ),
                "recent_completed_including_reused": recent_completed,
                "last_completion_seconds_ago": last_completion_seconds,
                "failure_stages": failure_stages,
                "failure_codes": failure_codes,
                "failure_operations": failure_operations,
                "failure_database": failure_database,
                "failure_attempts": failure_attempts,
            }
        )
        return values

    @staticmethod
    def _immutable_job(value: ExperimentJob) -> dict[str, object]:
        return value.model_dump(
            mode="json",
            exclude={
                "status",
                "attempt_count",
                "claimed_by",
                "lease_expires_at",
                "available_at",
                "created_at",
                "started_at",
                "finished_at",
                "error_type",
                "error_summary",
                "error_artifact_id",
            },
        )

    async def _load_successful_call(self, job_id: str) -> ModelCallTrace | None:
        async with ExperimentUnitOfWork(self.sessions) as uow:
            return await uow.model_calls.get_successful_for_job(job_id)

    async def _load_model_call(self, model_call_id: str) -> ModelCallTrace | None:
        async with ExperimentUnitOfWork(self.sessions) as uow:
            return await uow.model_calls.get(model_call_id)

    async def _before_gate(self, built: BuiltCompilationUnit) -> dict[str, str]:
        async with self._progress_lock:
            self._progress["gate_waiters"] += 1
        try:
            async with self._gate_condition:
                await self._gate_condition.wait_for(
                    lambda: (
                        not self._pending_gate_ordinals
                        or built.unit.ordinal == min(self._pending_gate_ordinals)
                    )
                )
        finally:
            async with self._progress_lock:
                self._progress["gate_waiters"] = max(
                    0,
                    self._progress["gate_waiters"] - 1,
                )
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.questioner is not None
            return await uow.questioner.list_duplicate_primaries(built.unit.run_id)

    async def _persist_result(self, result: QuestionerUnitResult) -> None:
        candidate_count = len(result.candidates)
        revised_candidate_count = sum(
            item.candidate.candidate_origin is QuestionerCandidateOrigin.REVISED
            for item in result.candidates
        )
        verification_batch_count = len(result.verification_batches)
        supersession_count = len(result.supersessions)

        async def persist(
            operation: str,
            action: Awaitable[Any],
            *,
            verification_batch: QuestionerVerificationBatch | None = None,
        ) -> None:
            try:
                await action
            except Exception as error:
                raise QuestionerPersistenceOperationError(
                    operation,
                    error,
                    candidate_count=candidate_count,
                    revised_candidate_count=revised_candidate_count,
                    verification_batch_count=verification_batch_count,
                    supersession_count=supersession_count,
                    verification_input_count=(
                        verification_batch.input_count if verification_batch is not None else None
                    ),
                    verification_pass_count=(
                        verification_batch.pass_count if verification_batch is not None else None
                    ),
                    verification_revise_count=(
                        verification_batch.revise_count if verification_batch is not None else None
                    ),
                    verification_reject_count=(
                        verification_batch.reject_count if verification_batch is not None else None
                    ),
                    verification_provider_fallback=(
                        bool(verification_batch.metadata.get("provider_output_fallback", False))
                        if verification_batch is not None
                        else None
                    ),
                    verification_single_candidate_scope=(
                        verification_batch.scope_qbank_id is not None
                        and verification_batch.input_count == 1
                        if verification_batch is not None
                        else None
                    ),
                ) from error

        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.questioner is not None and uow.compiler_units is not None
            await persist("generation", uow.questioner.add_generation(result.generation))
            await persist(
                "gated_candidates",
                uow.questioner.persist_gated_candidates(list(result.candidates)),
            )
            for batch in result.verification_batches:
                values = [
                    item
                    for item in result.verifications
                    if item.verification_batch_id == batch.verification_batch_id
                ]
                await persist(
                    "verification_batch",
                    uow.questioner.add_verification_batch(batch, values),
                    verification_batch=batch,
                )
                repair_count = int(batch.metadata.get("repair_attempt_count", 0))
                provider_fallback = bool(batch.metadata.get("provider_output_fallback", False))
                if repair_count or provider_fallback:
                    await persist(
                        "structured_output_diagnostic",
                        uow.structured_output_diagnostics.add(
                            StructuredOutputItemDiagnostic(
                                diagnostic_id=stable_compiler_id(
                                    "structured-output-diagnostic",
                                    [
                                        batch.model_call_id,
                                        "questioner_verifier",
                                        0,
                                    ],
                                ),
                                run_id=batch.run_id,
                                model_call_id=batch.model_call_id,
                                compiler_unit_id=batch.compiler_unit_id,
                                stage=StructuredOutputStage.QUESTIONER_VERIFIER,
                                item_ordinal=0,
                                item_hash=batch.content_hash,
                                status=(
                                    StructuredOutputItemStatus.CONSERVATIVE_REJECT
                                    if provider_fallback
                                    else StructuredOutputItemStatus.REPAIRED
                                ),
                                parser_version=self.config.verifier_prompt_version,
                                repair_attempt_count=repair_count,
                                errors=[
                                    StructuredOutputValidationError(
                                        location="$",
                                        error_type=(
                                            "provider_output_fallback"
                                            if provider_fallback
                                            else "json_repaired"
                                        ),
                                    )
                                ],
                                created_at=batch.created_at,
                            )
                        ),
                    )
                replacement_call_ids = batch.metadata.get("replacement_model_call_ids", {})
                replacement_repairs = batch.metadata.get("replacement_repair_attempt_counts", {})
                replacement_fallbacks = batch.metadata.get(
                    "replacement_provider_fallback_aliases", []
                )
                fallback_aliases = (
                    {value for value in replacement_fallbacks if isinstance(value, str)}
                    if isinstance(replacement_fallbacks, list)
                    else set()
                )
                if isinstance(replacement_call_ids, Mapping) and isinstance(
                    replacement_repairs, Mapping
                ):
                    for ordinal, alias in enumerate(
                        sorted(key for key in replacement_call_ids if isinstance(key, str))
                    ):
                        replacement_call_id = replacement_call_ids.get(alias)
                        replacement_repair_count = replacement_repairs.get(alias, 0)
                        if not isinstance(replacement_call_id, str) or not isinstance(
                            replacement_repair_count, int
                        ):
                            continue
                        provider_fallback = alias in fallback_aliases
                        if replacement_repair_count <= 0 and not provider_fallback:
                            continue
                        await persist(
                            "structured_output_diagnostic",
                            uow.structured_output_diagnostics.add(
                                StructuredOutputItemDiagnostic(
                                    diagnostic_id=stable_compiler_id(
                                        "structured-output-diagnostic",
                                        [
                                            replacement_call_id,
                                            "questioner_revision",
                                            ordinal,
                                        ],
                                    ),
                                    run_id=batch.run_id,
                                    model_call_id=replacement_call_id,
                                    compiler_unit_id=batch.compiler_unit_id,
                                    stage=StructuredOutputStage.QUESTIONER_REVISION,
                                    item_ordinal=ordinal,
                                    item_hash=canonical_content_hash(
                                        [
                                            batch.content_hash,
                                            alias,
                                            (
                                                "revision_provider_output_invalid"
                                                if provider_fallback
                                                else "revision_json_repaired"
                                            ),
                                        ]
                                    ),
                                    status=(
                                        StructuredOutputItemStatus.CONSERVATIVE_REJECT
                                        if provider_fallback
                                        else StructuredOutputItemStatus.REPAIRED
                                    ),
                                    parser_version=("questioner-revision-json-parser-v1"),
                                    repair_attempt_count=(replacement_repair_count),
                                    errors=[
                                        StructuredOutputValidationError(
                                            location="$",
                                            error_type=(
                                                "provider_output_fallback"
                                                if provider_fallback
                                                else "json_repaired"
                                            ),
                                        )
                                    ],
                                    created_at=batch.created_at,
                                )
                            ),
                        )
            for value in result.supersessions:
                await persist("supersession", uow.questioner.add_supersession(value))
            await persist(
                "stage_result",
                uow.compiler_units.add_stage_result(result.stage_result),
            )

    async def _persist_v3_seed(
        self,
        built: BuiltCompilationUnit,
        generation: QuestionerGeneration,
        candidates: list[GatedQuestionCandidate],
        answer_claims: list[QuestionerAnswerClaim],
    ) -> None:
        if built.opportunities is None:
            raise ValueError("Questioner v3 seed requires opportunities")
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.questioner is not None
            await uow.questioner.add_opportunity_set(
                built.opportunities.opportunity_set,
                list(built.opportunities.opportunities),
                list(built.opportunities.evidence_refs),
            )
            await uow.questioner.add_generation(generation)
            for diagnostic in questioner_seed_diagnostics(generation):
                await uow.structured_output_diagnostics.add(diagnostic)
            for item in candidates:
                await uow.questioner.add_candidate(
                    item.candidate,
                    expected_evidence=list(item.expected_evidence),
                )
                if item.gate_result.status is CandidateGateStatus.REJECTED:
                    await uow.questioner.add_gate(item.gate_result)
            for claim in answer_claims:
                await uow.questioner.add_answer_claim(claim)

    async def _complete_gate_turn(self, ordinal: int) -> None:
        async with self._gate_condition:
            self._pending_gate_ordinals.discard(ordinal)
            self._gate_condition.notify_all()

    async def _heartbeat(self, job_id: str, worker_id: str, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.lease_seconds / 3)
            except TimeoutError:
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    assert uow.jobs is not None
                    await uow.jobs.heartbeat(job_id, worker_id, self.lease_seconds)

    async def _finish_failed(
        self,
        job: ExperimentJob,
        worker_id: str,
        error: BaseException,
    ) -> ExperimentJob:
        diagnostics = safe_database_error_diagnostics(error)
        payload = json.dumps(diagnostics, sort_keys=True).encode("utf-8")
        artifact_id: str | None = None
        try:
            artifact = await self.error_artifacts.put_bytes(
                artifact_id=f"{job.job_id}:attempt:{job.attempt_count}:error",
                run_id=job.run_id,
                snapshot_id=job.snapshot_id,
                artifact_type=AssetType.OTHER,
                data=payload,
                mime_type="application/json",
                extension="json",
            )
            artifact_id = artifact.artifact_id
        except Exception:
            artifact_id = None
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None
            return await uow.jobs.finish(
                job.job_id,
                worker_id,
                success=False,
                error_type=type(error).__name__,
                error_summary=safe_database_error_summary(diagnostics),
                error_artifact_id=artifact_id,
                retryable=diagnostics.get("provider_retryable") != "false",
            )
