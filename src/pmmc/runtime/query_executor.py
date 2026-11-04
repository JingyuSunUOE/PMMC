"""Short-lived Online Query Executor graph over durable repository boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, cast

from sqlalchemy.exc import SQLAlchemyError

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.infrastructure.database.errors import ExperimentPersistenceError
from pmmc.memory.operators import ProgramExecutionResult
from pmmc.runtime.answering import InvalidRuntimeAnswer, UnsupportedRuntimeAnswer
from pmmc.runtime.evidence_sufficiency import (
    EvidenceGroupResolver,
    assess_query_evidence_sufficiency,
)
from pmmc.runtime.qbank_matcher import (
    MatchGridKey,
    RuntimeMatchResult,
    RuntimeQBankChannelHit,
    apply_reranker_assessments,
    counterfactual_program_candidate,
    decide_runtime_route,
    rank_runtime_match_grid,
    reranker_pair_ids,
)
from pmmc.runtime.query_executor_config import (
    QueryExecutorConfig,
    query_executor_routing_policy_hash,
)
from pmmc.runtime.relevance_grid import (
    bind_joint_context,
    bind_runtime_relation_contract,
    project_runtime_relevance_grid,
    requires_independent_relation,
    resolve_relevance_grid_answers,
)
from pmmc.schemas import (
    AnswerGenerationInput,
    AnswerReuseLink,
    FinalAnswerRecord,
    MemoryProgram,
    PlannerRequirementContract,
    QBankMatchFusedCandidate,
    QBankRerankerAssessment,
    QueryAnswerSelection,
    QueryAnswerVariant,
    QueryEvidenceRelevanceAssessment,
    QueryEvidenceRelevanceItem,
    QueryEvidenceSufficiencyAssessment,
    QueryExecutionReuseLink,
    QueryFallbackDecision,
    QueryFallbackReason,
    QueryInput,
    QueryProgramBinding,
    QueryRouteDecision,
    QueryRouteOutcome,
)
from pmmc.schemas.answer_output import TypedAnswerResult
from pmmc.schemas.query_executor import (
    ExperimentCell,
    QueryExperimentCellResult,
    runtime_fact_payload,
)


@dataclass(frozen=True)
class FrozenProgramBinding:
    """Runtime-safe program and requirement loaded from one frozen projection row."""

    binding: QueryProgramBinding
    program: MemoryProgram
    requirement: PlannerRequirementContract
    predicted_question: str
    collection_bindings: Mapping[str, str]
    query_vectors: Mapping[str, list[float]]
    source_program_hash: str


@dataclass(frozen=True)
class RuntimeEvidenceRelevance:
    group_ids: frozenset[str]
    assessment: QueryEvidenceRelevanceAssessment
    items: tuple[QueryEvidenceRelevanceItem, ...]


@dataclass(frozen=True)
class QueryAttemptLease:
    job_id: str
    owner: str
    attempt_count: int


query_attempt_lease: ContextVar[QueryAttemptLease | None] = ContextVar(
    "query_attempt_lease",
    default=None,
)


class OnlineQueryExecutorBackend(Protocol):
    """Durable I/O and model boundaries used by the short-lived graph."""

    @property
    def fallback_run_id(self) -> str: ...

    @property
    def source_projection_hash(self) -> str: ...

    async def reuse_matching_phase(self, run_id: str, query_id: str) -> None: ...

    async def prepare_typed_pmmc_answer(
        self, *, query: QueryInput, result: ProgramExecutionResult, cell: ExperimentCell
    ) -> AnswerGenerationInput: ...

    def typed_request_identity(self, prepared: AnswerGenerationInput) -> str: ...

    async def generate_deferred_typed_answer(self, prepared: AnswerGenerationInput) -> dict: ...

    async def ground_prepared_typed_answer(
        self, prepared: AnswerGenerationInput, reference: dict
    ) -> TypedAnswerResult: ...

    async def persist_matching_phase(
        self, state: QueryExecutorGraphState, prepared: dict
    ) -> None: ...

    async def load_runtime_phase(self, run_id: str, query_id: str, phase: str) -> dict: ...

    async def persist_runtime_phase(
        self, run_id: str, query_id: str, phase: str, payload: dict
    ) -> None: ...

    async def generate_typed_pmmc_answer(
        self,
        *,
        query: QueryInput,
        frozen: FrozenProgramBinding,
        result: ProgramExecutionResult,
        cell: ExperimentCell,
    ) -> TypedAnswerResult: ...

    async def run_typed_native_mm_counterfactual(
        self,
        run_id: str,
        query: QueryInput,
    ) -> TypedAnswerResult: ...

    async def load_query(self, run_id: str, query_id: str) -> QueryInput: ...

    async def load_verified_projection(self, run_id: str, query: QueryInput) -> str: ...

    async def ensure_precomputed_query_embeddings(self, run_id: str, query: QueryInput) -> None: ...

    async def load_channel_hits(
        self, run_id: str, query: QueryInput
    ) -> list[RuntimeQBankChannelHit]: ...

    async def rerank_qbank_candidates(
        self,
        *,
        run_id: str,
        query: QueryInput,
        candidates: tuple[QBankMatchFusedCandidate, ...],
    ) -> dict[str, QBankRerankerAssessment]: ...

    async def bind_frozen_program(
        self,
        *,
        run_id: str,
        query: QueryInput,
        projection_id: str,
        route_decision: QueryRouteDecision,
        candidate: QBankMatchFusedCandidate,
        round_id: int,
        counterfactual: bool,
    ) -> FrozenProgramBinding: ...

    async def execute_memory_program(
        self, query: QueryInput, frozen: FrozenProgramBinding
    ) -> ProgramExecutionResult: ...

    async def relevant_runtime_evidence_groups(
        self,
        *,
        query: QueryInput,
        frozen: FrozenProgramBinding,
        result: ProgramExecutionResult,
    ) -> RuntimeEvidenceRelevance: ...

    async def generate_pmmc_answer(
        self,
        *,
        query: QueryInput,
        frozen: FrozenProgramBinding,
        result: ProgramExecutionResult,
        variant: QueryAnswerVariant,
    ) -> FinalAnswerRecord: ...

    async def run_native_mm_counterfactual(
        self, run_id: str, query: QueryInput
    ) -> FinalAnswerRecord: ...

    async def persist_terminal(self, state: QueryExecutorGraphState) -> None: ...


class QueryExecutorGraphState(TypedDict, total=False):
    run_id: str
    query_id: str
    config: QueryExecutorConfig
    query: QueryInput
    projection_id: str
    channel_hits: list[RuntimeQBankChannelHit]
    match_grid: dict[MatchGridKey, RuntimeMatchResult]
    reranker_assessments: dict[str, QBankRerankerAssessment]
    route_decision: QueryRouteDecision
    frozen_by_round: dict[int, FrozenProgramBinding]
    execution_by_round: dict[int, ProgramExecutionResult]
    sufficiency_by_round: dict[int, QueryEvidenceSufficiencyAssessment]
    relevance_by_round: dict[int, RuntimeEvidenceRelevance]
    execution_reuse_links: tuple[QueryExecutionReuseLink, ...]
    pmmc_answers: dict[int, FinalAnswerRecord]
    fallback_answer: FinalAnswerRecord
    fallback_run_id: str
    fallback_error_type: str
    fallback_abstention: dict[str, Any]
    fallback_decision: QueryFallbackDecision
    answer_selection: QueryAnswerSelection
    active_answer: FinalAnswerRecord
    evaluation_alias_answers: tuple[FinalAnswerRecord, ...]
    pmmc_answer_errors: dict[int, str]
    pmmc_abstentions: dict[int, dict[str, Any]]
    answer_contract_failures: dict[int, tuple[tuple[str, ...], ...]]
    answer_reuse_links: tuple[AnswerReuseLink, ...]
    program_binding_errors: dict[int, str]
    program_execution_errors: dict[int, str]
    cell_results: tuple[QueryExperimentCellResult, ...]
    typed_answers: dict[str, TypedAnswerResult]
    typed_fallback: TypedAnswerResult
    frozen_by_cell: dict[MatchGridKey, FrozenProgramBinding]
    execution_by_cell: dict[MatchGridKey, ProgramExecutionResult]


class _LinearAsyncGraph:
    def __init__(self, nodes: list[Any]) -> None:
        self.nodes = nodes

    async def ainvoke(self, state: QueryExecutorGraphState) -> QueryExecutorGraphState:
        current = dict(state)
        for node in self.nodes:
            current.update(await node(current))
        return cast(QueryExecutorGraphState, current)


class _CompiledGraph(Protocol):
    async def ainvoke(self, state: QueryExecutorGraphState) -> QueryExecutorGraphState: ...


class _QueryGraphNode(Protocol):
    # LangGraph's node protocol preserves the keyword-capable `state` parameter.
    def __call__(self, state: QueryExecutorGraphState) -> Awaitable[dict[str, Any]]: ...


def _diagnostic_node(
    name: str,
    callback: Callable[[QueryExecutorGraphState], Awaitable[dict[str, Any]]],
) -> _QueryGraphNode:
    async def invoke(state: QueryExecutorGraphState) -> dict[str, Any]:
        try:
            return await callback(state)
        except Exception as exc:
            # Only static node names enter diagnostics, never the graph state.
            exc.failure_stage = name  # type: ignore[attr-defined]
            raise

    return invoke


class OnlineQueryExecutor:
    """Execute one query while keeping Postgres/object storage authoritative."""

    node_names = (
        "load_query",
        "load_verified_projection",
        "load_precomputed_query_embeddings",
        "match_qbank",
        "rerank_qbank",
        "fuse_and_route",
        "bind_frozen_program",
        "execute_memory_program",
        "runtime_evidence_sufficiency",
        "assemble_and_generate_pmmc_answer",
        "load_or_run_native_mm_fallback",
        "select_active_answer",
        "persist_terminal_facts",
    )

    def __init__(
        self,
        *,
        backend: OnlineQueryExecutorBackend,
        config: QueryExecutorConfig,
        group_resolver: EvidenceGroupResolver | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.group_resolver = group_resolver
        callbacks: list[Callable[[QueryExecutorGraphState], Awaitable[dict[str, Any]]]] = [
            self._load_query,
            self._load_verified_projection,
            self._load_precomputed_query_embeddings,
            self._match_qbank,
            self._rerank_qbank,
            self._fuse_and_route,
            self._bind_frozen_program,
            self._execute_memory_program,
            self._runtime_evidence_sufficiency,
            self._assemble_and_generate_pmmc_answer,
            self._load_or_run_native_mm_fallback,
            self._select_active_answer,
            self._persist_terminal_facts,
        ]
        diagnostic_nodes = [
            _diagnostic_node(name, callback)
            for name, callback in zip(self.node_names, callbacks, strict=True)
        ]
        try:
            from langgraph.graph import END, START, StateGraph
        except ModuleNotFoundError:
            self.compiled: _CompiledGraph = _LinearAsyncGraph(diagnostic_nodes)
        else:
            graph = StateGraph(QueryExecutorGraphState)
            for name, callback in zip(self.node_names, diagnostic_nodes, strict=True):
                graph.add_node(name, callback)
            graph.add_edge(START, self.node_names[0])
            for source, target in zip(self.node_names, self.node_names[1:], strict=False):
                graph.add_edge(source, target)
            graph.add_edge(self.node_names[-1], END)
            self.compiled = cast(_CompiledGraph, graph.compile())

    async def execute(
        self,
        *,
        run_id: str,
        query_id: str,
        lease: QueryAttemptLease | None = None,
        phase: str = "all",
    ) -> QueryExecutorGraphState:
        if self.config.is_v3:
            token = query_attempt_lease.set(lease)
            try:
                if phase in {"answer", "select"}:
                    return await self._resume_v3(run_id=run_id, query_id=query_id, phase=phase)
                if phase not in {"all", "match"}:
                    raise ValueError("unsupported v3 runtime phase")
                if phase == "match" and getattr(self.backend, "source_match_run_id", None):
                    await self.backend.reuse_matching_phase(run_id, query_id)
                    return {"run_id": run_id, "query_id": query_id}
                return await self._execute_v3(
                    run_id=run_id,
                    query_id=query_id,
                    prepare_only=phase == "match",
                )
            finally:
                query_attempt_lease.reset(token)
        return await self.compiled.ainvoke(
            {"run_id": run_id, "query_id": query_id, "config": self.config}
        )

    async def _execute_v3(
        self,
        *,
        run_id: str,
        query_id: str,
        prepare_only: bool = False,
    ) -> QueryExecutorGraphState:
        """Execute the declared cells, retaining exact reuse and terminal facts."""
        state: QueryExecutorGraphState = {
            "run_id": run_id,
            "query_id": query_id,
            "config": self.config,
        }
        for callback in (
            self._load_query,
            self._load_verified_projection,
            self._load_precomputed_query_embeddings,
            self._match_qbank,
            self._rerank_qbank,
        ):
            state.update(cast(QueryExecutorGraphState, await callback(state)))
        grid = apply_reranker_assessments(
            grid=state["match_grid"],
            assessments=state["reranker_assessments"],
            config=self.config,
        )
        state["match_grid"] = grid
        query = state["query"]
        policy_hash = query_executor_routing_policy_hash(
            self.config,
            match_threshold=self.config.route_match_threshold,
            gap_threshold=self.config.route_gap_threshold,
        )
        projection_hash = self.backend.source_projection_hash
        executions: dict[str, tuple[str, ProgramExecutionResult | None, str | None]] = {}
        answer_sources: dict[str, str] = {}
        answers: dict[str, TypedAnswerResult] = {}
        frozen_by_cell: dict[MatchGridKey, FrozenProgramBinding] = {}
        execution_by_cell: dict[MatchGridKey, ProgramExecutionResult] = {}
        facts: list[QueryExperimentCellResult] = []
        prepared_by_cell: dict[str, AnswerGenerationInput] = {}
        prepared_contexts: dict[tuple[str, str], AnswerGenerationInput] = {}
        grid_answers: dict[str, TypedAnswerResult] = {}
        grid_errors: dict[str, str] = {}
        grid_generated: dict[str, str] = {}
        fallback: TypedAnswerResult | None = None
        fallback_error: str | None = None
        try:
            if not prepare_only:
                fallback = await self.backend.run_typed_native_mm_counterfactual(run_id, query)
                if (
                    fallback.run_id not in {run_id, self.backend.fallback_run_id}
                    or fallback.query_id != query_id
                ):
                    raise ValueError("typed fallback escaped runtime query scope")
                answers[fallback.typed_answer_id] = fallback
        except (SQLAlchemyError, ExperimentPersistenceError):
            raise
        except Exception as exc:
            fallback_error = type(exc).__name__

        active_key = MatchGridKey(self.config.question_budget, self.config.active_round)
        primary = getattr(self.backend, "primary_readout", False) is True
        if primary and set(grid) != {active_key}:
            raise ValueError("primary readout cannot execute analysis cells")
        for key, matched in sorted(
            grid.items(),
            key=lambda item: (
                item[0].question_budget,
                item[0].round_id,
            ),
        ):
            cell = ExperimentCell(
                treatment="pmmc" if self.config.treatment == "main" else self.config.treatment,
                question_budget=key.question_budget,
                revision_depth=key.round_id,
                routing_policy_hash=policy_hash,
                seed=self.config.experiment_seed,
                source_projection_id=state["projection_id"],
                source_projection_hash=projection_hash,
            )
            cell_id = stable_compiler_id("experiment-cell", cell.model_dump(mode="json"))
            route = decide_runtime_route(
                query_executor_run_id=run_id,
                query_id=query_id,
                match_batch_id=stable_compiler_id(
                    "runtime-qbank-match-batch",
                    [run_id, query_id, key.question_budget, key.round_id],
                ),
                candidates=matched.candidates,
                config=self.config,
            )
            if key == active_key:
                state["route_decision"] = route
            candidate = counterfactual_program_candidate(matched.candidates)
            if primary and route.outcome is not QueryRouteOutcome.PROGRAM:
                # 0032 still requires the two contrast rows, not speculative answers.
                candidate = None
            frozen = None
            result = None
            relevance = None
            sufficiency = None
            answer = None
            error = None
            execution_source = None
            answer_source = None
            relevance_grid: list[dict[str, Any]] = []
            joint_contexts: dict[str, Any] = {}
            if candidate is not None:
                try:
                    frozen = await self.backend.bind_frozen_program(
                        run_id=run_id,
                        query=query,
                        projection_id=state["projection_id"],
                        route_decision=route,
                        candidate=candidate,
                        round_id=key.round_id,
                        counterfactual=(route.outcome is not QueryRouteOutcome.PROGRAM),
                    )
                    frozen_by_cell[key] = frozen
                    identity = canonical_content_hash(
                        {
                            "query": query.model_dump(mode="json"),
                            "program_hash": frozen.source_program_hash,
                            "plan_id": frozen.program.plan_id,
                            "requirement": frozen.requirement.content_hash,
                            "collections": dict(frozen.collection_bindings),
                            "vectors": dict(frozen.query_vectors),
                        }
                    )
                    if identity not in executions:
                        try:
                            value = await self.backend.execute_memory_program(query, frozen)
                            executions[identity] = (cell_id, value, None)
                        except (SQLAlchemyError, ExperimentPersistenceError):
                            raise
                        except Exception as exc:
                            executions[identity] = (cell_id, None, type(exc).__name__)
                    execution_source, result, error = executions[identity]
                    if result is not None:
                        execution_by_cell[key] = result
                        structural = assess_query_evidence_sufficiency(
                            query_executor_run_id=run_id,
                            binding=frozen.binding,
                            requirement=frozen.requirement,
                            result=result,
                            group_resolver=self.group_resolver,
                        )
                        relevance = await self.backend.relevant_runtime_evidence_groups(
                            query=query,
                            frozen=frozen,
                            result=result,
                        )
                        sufficient = structural.sufficient and relevance.assessment.passed
                        sufficiency_payload = structural.model_dump(mode="python")
                        sufficiency_payload.update(
                            {
                                "sufficiency_assessment_id": stable_compiler_id(
                                    "query-evidence-sufficiency-v3",
                                    [
                                        cell_id,
                                        structural.content_hash,
                                        relevance.assessment.content_hash,
                                    ],
                                ),
                                "structural_requirement_success": structural.sufficient,
                                "query_evidence_relevance": relevance.assessment.passed,
                                "answer_support": None,
                                "relevant_evidence_group_count": len(relevance.group_ids),
                                "relevance_threshold": (
                                    self.config.query_evidence_relevance_threshold
                                ),
                                "sufficient": sufficient,
                                "failure_reason": None
                                if sufficient
                                else (
                                    structural.failure_reason
                                    or QueryFallbackReason.RUNTIME_RELEVANCE_INSUFFICIENT
                                ),
                                "content_hash": canonical_content_hash(
                                    [
                                        structural.content_hash,
                                        relevance.assessment.content_hash,
                                        self.config.sufficiency_policy_version,
                                    ]
                                ),
                                "metadata": {
                                    "sufficiency_policy_version": (
                                        self.config.sufficiency_policy_version
                                    ),
                                    "answer_grounding_status": "not_evaluated",
                                },
                            }
                        )
                        sufficiency = QueryEvidenceSufficiencyAssessment.model_validate(
                            sufficiency_payload
                        )
                        if getattr(self.backend, "relevance_grid_enabled", False):
                            projections = project_runtime_relevance_grid(
                                run_id=run_id, cell_id=cell_id, binding=frozen.binding,
                                requirement=frozen.requirement, result=result,
                                items=relevance.items,
                                thresholds=self.config.relevance_cosine_grid,
                                group_resolver=self.group_resolver,
                                joint_enabled=self.config.joint_relevance_enabled,
                            )
                            for projection in projections:
                                point = dict(projection.point)
                                if primary and (
                                    point["text_threshold"] != self.config.query_evidence_relevance_threshold
                                    or point["multimodal_threshold"] != self.config.multimodal_evidence_relevance_threshold
                                ):
                                    # Keep the structural inventory for match-source validation.
                                    # No prompt, prepared key, model call or score for this point.
                                    point["status"] = "deferred_analysis"
                                    relevance_grid.append(point)
                                    continue
                                if projection.sufficiency.sufficient:
                                    context_key = (result.trace.execution_id,
                                                   point["evidence_subset_hash"])
                                    if context_key not in prepared_contexts:
                                        prepared_contexts[context_key] = (
                                            await self.backend.prepare_typed_pmmc_answer(
                                                query=query, result=projection.result, cell=cell,
                                            )
                                        )
                                    prepared = prepared_contexts[context_key].model_copy(
                                        update={"answer_variant": cell.variant}
                                    )
                                    prepared = bind_runtime_relation_contract(
                                        prepared, result=projection.result,
                                        minimum_groups=frozen.requirement.minimum_evidence_groups,
                                        relation_required=requires_independent_relation(
                                            frozen.requirement
                                        ),
                                    )
                                    request_key = stable_compiler_id(
                                        "relevance-context", [cell_id, context_key]
                                    )
                                    context_allowed = True
                                    if self.config.joint_relevance_enabled:
                                        prepared = bind_joint_context(
                                            prepared,
                                            result=projection.result,
                                            binding=frozen.binding,
                                            requirement=frozen.requirement,
                                            group_resolver=self.group_resolver,
                                            joint_image_hashes=point["joint_image_hashes"],
                                        )
                                        joint_contexts[request_key] = prepared.metadata[
                                            "joint_context"
                                        ]
                                        context_allowed = joint_contexts[request_key][
                                            "context_sufficient"
                                        ]
                                    prepared_by_cell[request_key] = prepared
                                    identity = self.backend.typed_request_identity(prepared)
                                    point.update(prepared_key=request_key,
                                                 answer_identity_hash=identity)
                                    if (
                                        context_allowed
                                        and not prepare_only
                                        and identity not in grid_answers
                                        and identity not in grid_errors
                                    ):
                                        try:
                                            reference = (
                                                await self.backend.generate_deferred_typed_answer(
                                                    prepared
                                                )
                                            )
                                            if self.config.joint_relevance_enabled:
                                                grid_generated[identity] = reference[
                                                    "model_call_id"
                                                ]
                                            generated_answer = (
                                                await self.backend.ground_prepared_typed_answer(
                                                    prepared, reference
                                                )
                                            )
                                            if (generated_answer.run_id != run_id
                                                    or generated_answer.query_id != query_id):
                                                raise ValueError("grid answer escaped query scope")
                                            grid_answers[identity] = generated_answer
                                        except (SQLAlchemyError, ExperimentPersistenceError):
                                            raise
                                        except Exception as exc:
                                            grid_errors[identity] = type(exc).__name__
                                    if identity in grid_answers:
                                        answers[grid_answers[identity].typed_answer_id] = (
                                            grid_answers[identity]
                                        )
                                relevance_grid.append(point)
                                if (
                                    point["text_threshold"]
                                    == self.config.query_evidence_relevance_threshold
                                    and point["multimodal_threshold"]
                                    == self.config.multimodal_evidence_relevance_threshold
                                ):
                                    sufficiency = projection.sufficiency
                                    sufficient = sufficiency.sufficient
                                    if point["prepared_key"] is not None:
                                        prepared_by_cell[cell_id] = prepared_by_cell[
                                            point["prepared_key"]
                                        ]
                                        if (
                                            point["prepared_key"] not in joint_contexts
                                            or joint_contexts[point["prepared_key"]][
                                                "context_sufficient"
                                            ]
                                        ):
                                            answer = grid_answers.get(point["answer_identity_hash"])
                                            error = grid_errors.get(point["answer_identity_hash"])
                                        if answer is not None:
                                            answer_source = answer_sources.setdefault(
                                                answer.typed_answer_id, cell_id
                                            )
                            if not prepare_only:
                                relevance_grid = resolve_relevance_grid_answers(
                                    relevance_grid,
                                    {
                                        point["prepared_key"]: grid_answers[
                                            point["answer_identity_hash"]
                                        ]
                                        for point in relevance_grid
                                        if point["answer_identity_hash"] in grid_answers
                                        and joint_contexts.get(point["prepared_key"], {}).get(
                                            "context_sufficient", True
                                        )
                                    },
                                    {
                                        point["prepared_key"]: grid_errors[
                                            point["answer_identity_hash"]
                                        ]
                                        for point in relevance_grid
                                        if point["answer_identity_hash"] in grid_errors
                                    },
                                    joint_contexts,
                                    {
                                        p["prepared_key"]: grid_generated[p["answer_identity_hash"]]
                                        for p in relevance_grid
                                        if p["answer_identity_hash"] in grid_generated
                                    },
                                )
                        elif self.config.joint_relevance_enabled:
                            raise ValueError("release R requires scoped relevance projections")
                        elif sufficient and prepare_only:
                            prepared_by_cell[
                                cell_id
                            ] = await self.backend.prepare_typed_pmmc_answer(
                                query=query,
                                result=result,
                                cell=cell,
                            )
                        elif sufficient:
                            answer = await self.backend.generate_typed_pmmc_answer(
                                query=query,
                                frozen=frozen,
                                result=result,
                                cell=cell,
                            )
                            if answer.query_id != query_id or answer.run_id != run_id:
                                raise ValueError("typed PMMC answer escaped query scope")
                            answers[answer.typed_answer_id] = answer
                            answer_source = answer_sources.setdefault(
                                answer.typed_answer_id, cell_id
                            )
                except (SQLAlchemyError, ExperimentPersistenceError):
                    raise
                except Exception as exc:
                    error = type(exc).__name__

            allowed = (
                route.outcome is QueryRouteOutcome.PROGRAM
                and sufficiency is not None
                and sufficiency.sufficient
            )
            pmmc_terminal = answer is not None and answer.terminal_state != "fallback_required"
            active_fallback = not (allowed and pmmc_terminal)
            selected = fallback if active_fallback else answer
            state_name = (
                selected.terminal_state
                if selected is not None and selected.terminal_state != "fallback_required"
                else "technical_failure"
                if error or fallback_error
                else "no_answer"
            )
            payload = {
                "query_executor_run_id": run_id,
                "query_id": query_id,
                "cell_id": cell_id,
                "cell": cell,
                "route": route,
                "binding": frozen.binding if frozen is not None else None,
                "execution_id": result.trace.execution_id if result is not None else None,
                "execution_source_cell_id": execution_source,
                "sufficiency": sufficiency,
                "relevance": relevance.assessment if relevance is not None else None,
                "relevance_items": list(relevance.items) if relevance is not None else [],
                "pmmc_answer_id": answer.typed_answer_id if answer is not None else None,
                "answer_source_cell_id": answer_source,
                "fallback_answer_id": fallback.typed_answer_id if fallback is not None else None,
                "selected_answer_id": selected.typed_answer_id
                if selected is not None and selected.terminal_state != "fallback_required"
                else None,
                "terminal_state": state_name,
                "error_type": error or (fallback_error if active_fallback else None),
                "active_fallback": active_fallback,
                "metadata": {
                    "fallback_error_type": fallback_error,
                    **({"relevance_grid": relevance_grid} if relevance_grid else {}),
                    **(
                        {"joint_contexts": joint_contexts}
                        if self.config.joint_relevance_enabled
                        else {}
                    ),
                },
            }
            fact = self._cell_result(payload)
            facts.append(fact)
            for treatment in ("no_fallback", "forced_top1_counterfactual"):
                contrast = cell.model_copy(update={"treatment": treatment})
                contrast_id = stable_compiler_id(
                    "experiment-cell", contrast.model_dump(mode="json")
                )
                selected_pmmc = (
                    answer
                    if pmmc_terminal and (allowed or treatment == "forced_top1_counterfactual")
                    else None
                )
                facts.append(
                    self._cell_result(
                        {
                            **payload,
                            "cell": contrast,
                            "cell_id": contrast_id,
                            "selected_answer_id": selected_pmmc.typed_answer_id
                            if selected_pmmc
                            else None,
                            "terminal_state": selected_pmmc.terminal_state
                            if selected_pmmc
                            else "technical_failure"
                            if error
                            else "no_answer",
                            "active_fallback": False,
                            "error_type": error,
                        }
                    )
                )
        state.update(
            {
                "cell_results": tuple(facts),
                "typed_answers": answers,
                "frozen_by_cell": frozen_by_cell,
                "execution_by_cell": execution_by_cell,
            }
        )
        if fallback is not None:
            state["typed_fallback"] = fallback
        if prepare_only:
            await self.backend.persist_matching_phase(state, prepared_by_cell)
        else:
            await self.backend.persist_terminal(state)
        return state

    async def _resume_v3(
        self, *, run_id: str, query_id: str, phase: str
    ) -> QueryExecutorGraphState:
        checkpoint = await self.backend.load_runtime_phase(run_id, query_id, "match")
        prepared = {
            key: AnswerGenerationInput.model_validate(value)
            for key, value in checkpoint["prepared"].items()
        }
        if getattr(self.backend, "primary_readout", False) is True:
            self._validate_primary_requests(checkpoint, prepared)
        if phase == "answer":
            references: dict[str, dict] = {}
            for request in prepared.values():
                if (
                    self.config.joint_relevance_enabled
                    and not request.metadata["joint_context"]["context_sufficient"]
                ):
                    continue
                identity = self.backend.typed_request_identity(request)
                if identity not in references:
                    references[identity] = await self.backend.generate_deferred_typed_answer(
                        request
                    )
            await self.backend.persist_runtime_phase(
                run_id,
                query_id,
                "answer",
                {
                    "requests": references,
                },
            )
            return {"run_id": run_id, "query_id": query_id}

        generated = await self.backend.load_runtime_phase(run_id, query_id, "answer")
        grounded: dict[str, TypedAnswerResult] = {}
        failed_groundings: dict[str, str] = {}
        cell_answers: dict[str, TypedAnswerResult] = {}
        errors: dict[str, str] = {}
        generated_calls: dict[str, str] = {}
        answer_sources: dict[str, str] = {}
        pending = [QueryExperimentCellResult.model_validate(value) for value in checkpoint["cells"]]
        prepared_owners = {
            point["prepared_key"]: fact.cell_id
            for fact in pending
            if fact.cell.treatment not in {"no_fallback", "forced_top1_counterfactual"}
            for point in fact.metadata.get("relevance_grid", [])
            if point.get("prepared_key") is not None
        }
        for cell_id, request in prepared.items():
            if (
                self.config.joint_relevance_enabled
                and not request.metadata["joint_context"]["context_sufficient"]
            ):
                continue
            identity = self.backend.typed_request_identity(request)
            reference = generated["requests"].get(identity)
            if reference is None:
                raise ValueError("answer phase is incomplete for a declared context")
            if self.config.joint_relevance_enabled:
                generated_calls[cell_id] = reference["model_call_id"]
            if identity in failed_groundings:
                errors[cell_id] = failed_groundings[identity]
                continue
            try:
                if identity not in grounded:
                    grounded[identity] = await self.backend.ground_prepared_typed_answer(
                        request, reference
                    )
                cell_answers[cell_id] = grounded[identity]
                answer_sources.setdefault(
                    grounded[identity].typed_answer_id, prepared_owners.get(cell_id, cell_id)
                )
            except (SQLAlchemyError, ExperimentPersistenceError):
                raise
            except Exception as exc:
                errors[cell_id] = type(exc).__name__
                failed_groundings[identity] = errors[cell_id]
        query = await self.backend.load_query(run_id, query_id)
        fallback = None
        fallback_error = None
        try:
            fallback = await self.backend.run_typed_native_mm_counterfactual(run_id, query)
        except (SQLAlchemyError, ExperimentPersistenceError):
            raise
        except Exception as exc:
            fallback_error = type(exc).__name__
        main_by_key = {
            (fact.cell.question_budget, fact.cell.revision_depth): fact
            for fact in pending
            if fact.cell.treatment not in {"no_fallback", "forced_top1_counterfactual"}
        }
        facts = []
        for fact in pending:
            source = main_by_key[(fact.cell.question_budget, fact.cell.revision_depth)]
            answer = cell_answers.get(source.cell_id)
            eligible = (
                fact.route.outcome is QueryRouteOutcome.PROGRAM
                and fact.sufficiency is not None
                and fact.sufficiency.sufficient
            )
            terminal = answer is not None and answer.terminal_state != "fallback_required"
            fallback_active = fact.cell.treatment not in {
                "no_fallback",
                "forced_top1_counterfactual",
            } and not (eligible and terminal)
            selected = (
                fallback
                if fallback_active
                else answer
                if terminal and (eligible or fact.cell.treatment == "forced_top1_counterfactual")
                else None
            )
            error = errors.get(source.cell_id) or fact.error_type
            payload = fact.model_dump(
                mode="python", exclude={"cell_result_id", "content_hash", "created_at"}
            )
            payload.update(
                {
                    "pmmc_answer_id": answer.typed_answer_id if answer else None,
                    "answer_source_cell_id": answer_sources.get(answer.typed_answer_id)
                    if answer
                    else None,
                    "fallback_answer_id": fallback.typed_answer_id if fallback else None,
                    "selected_answer_id": selected.typed_answer_id if selected else None,
                    "terminal_state": selected.terminal_state
                    if selected
                    else "technical_failure"
                    if error or (fallback_active and fallback_error)
                    else "no_answer",
                    "active_fallback": fallback_active,
                    "error_type": error or (fallback_error if fallback_active else None),
                    "metadata": {
                        **fact.metadata,
                        **({"relevance_grid": resolve_relevance_grid_answers(
                            fact.metadata["relevance_grid"], cell_answers, errors,
                            fact.metadata.get("joint_contexts"),
                            generated_calls,
                        )} if "relevance_grid" in fact.metadata else {}),
                    },
                }
            )
            facts.append(self._cell_result(payload))
        state: QueryExecutorGraphState = {
            "run_id": run_id,
            "query_id": query_id,
            "query": query,
            "cell_results": tuple(facts),
            "typed_answers": {answer.typed_answer_id: answer for answer in grounded.values()},
            "route_decision": QueryRouteDecision.model_validate(checkpoint["route"]),
        }
        if fallback is not None:
            state["typed_fallback"] = fallback
        await self.backend.persist_terminal(state)
        return state

    def _validate_primary_requests(self, checkpoint, prepared) -> None:
        """Even a restored/shared match may authorize only the active generated context."""
        cells = [QueryExperimentCellResult.model_validate(value) for value in checkpoint["cells"]]
        if len(cells) != 3 or {c.cell.treatment for c in cells} != {
            "pmmc", "no_fallback", "forced_top1_counterfactual",
        } or any((c.cell.question_budget, c.cell.revision_depth) != (8, 2) for c in cells):
            raise ValueError("primary match checkpoint has incompatible declared cells")
        active = next(c for c in cells if c.cell.treatment == "pmmc")
        expected: set[str] = set()
        for point in active.metadata.get("relevance_grid", []):
            if (point["text_threshold"], point["multimodal_threshold"]) != (
                self.config.query_evidence_relevance_threshold,
                self.config.multimodal_evidence_relevance_threshold,
            ):
                if point.get("prepared_key") or point["status"] != "deferred_analysis":
                    raise ValueError("primary match checkpoint prepared an off-grid context")
            elif point.get("prepared_key") is not None:
                if (active.route.outcome is not QueryRouteOutcome.PROGRAM or
                        active.sufficiency is None or not active.sufficiency.sufficient):
                    raise ValueError("primary match checkpoint prepared a rejected route")
                expected.update((active.cell_id, point["prepared_key"]))
        if set(prepared) != expected or (expected and len({
            self.backend.typed_request_identity(request) for request in prepared.values()
        }) != 1):
            raise ValueError("primary match checkpoint contains an unauthorized answer request")

    @staticmethod
    def _cell_result(payload: dict[str, Any]) -> QueryExperimentCellResult:
        identity = [payload["query_executor_run_id"], payload["query_id"], payload["cell_id"]]
        return QueryExperimentCellResult(
            **payload,
            cell_result_id=stable_compiler_id("query-cell-result", identity),
            content_hash=canonical_content_hash(runtime_fact_payload(payload)),
        )

    async def _load_query(self, state: QueryExecutorGraphState) -> dict[str, Any]:
        return {"query": await self.backend.load_query(state["run_id"], state["query_id"])}

    async def _load_verified_projection(self, state: QueryExecutorGraphState) -> dict[str, Any]:
        return {
            "projection_id": await self.backend.load_verified_projection(
                state["run_id"], state["query"]
            )
        }

    async def _load_precomputed_query_embeddings(
        self, state: QueryExecutorGraphState
    ) -> dict[str, Any]:
        await self.backend.ensure_precomputed_query_embeddings(state["run_id"], state["query"])
        return {}

    async def _match_qbank(self, state: QueryExecutorGraphState) -> dict[str, Any]:
        return {
            "channel_hits": await self.backend.load_channel_hits(state["run_id"], state["query"])
        }

    async def _rerank_qbank(self, state: QueryExecutorGraphState) -> dict[str, Any]:
        query = state["query"]
        grid = rank_runtime_match_grid(
            query_executor_run_id=state["run_id"],
            query_id=query.query_id,
            hits=state["channel_hits"],
            has_query_image=bool(query.query_image_refs),
            config=self.config,
        )
        if self.config.is_v3 and not getattr(self.backend, "execute_full_grid", True):
            grid = {
                key: value
                for key, value in grid.items()
                if key == MatchGridKey(self.config.question_budget, self.config.active_round)
            }
        if not self.config.reranker_enabled:
            return {"match_grid": grid, "reranker_assessments": {}}
        selected_ids = set(
            reranker_pair_ids(
                grid,
                top_k=self.config.reranker_top_k,
                separate_route_classes=self.config.is_v3,
            )
        )
        candidates_by_id: dict[str, QBankMatchFusedCandidate] = {}
        for result in grid.values():
            for candidate in result.candidates:
                if candidate.qbank_id in selected_ids:
                    candidates_by_id.setdefault(candidate.qbank_id, candidate)
        assessments = await self.backend.rerank_qbank_candidates(
            run_id=state["run_id"],
            query=query,
            candidates=tuple(candidates_by_id[key] for key in sorted(candidates_by_id)),
        )
        if set(assessments) != set(candidates_by_id):
            raise RuntimeError("reranker accounting does not cover the bounded pair union")
        return {"match_grid": grid, "reranker_assessments": assessments}

    async def _fuse_and_route(self, state: QueryExecutorGraphState) -> dict[str, Any]:
        query = state["query"]
        grid = state["match_grid"]
        if self.config.is_v2:
            grid = apply_reranker_assessments(
                grid=grid,
                assessments=state.get("reranker_assessments", {}),
                config=self.config,
            )
        active = grid[
            MatchGridKey(
                question_budget=self.config.question_budget,
                round_id=self.config.active_round,
            )
        ]
        match_batch_id = (
            active.candidates[0].match_batch_id
            if active.candidates
            else stable_compiler_id(
                "runtime-qbank-match-batch",
                [
                    state["run_id"],
                    query.query_id,
                    self.config.question_budget,
                    self.config.active_round,
                ],
            )
        )
        return {
            "match_grid": grid,
            "route_decision": decide_runtime_route(
                query_executor_run_id=state["run_id"],
                query_id=query.query_id,
                match_batch_id=match_batch_id,
                candidates=active.candidates,
                config=self.config,
            ),
        }

    async def _bind_frozen_program(self, state: QueryExecutorGraphState) -> dict[str, Any]:
        frozen: dict[int, FrozenProgramBinding] = {}
        errors: dict[int, str] = {}
        for round_id in self.config.rounds:
            candidates = state["match_grid"][
                MatchGridKey(
                    question_budget=self.config.question_budget,
                    round_id=round_id,
                )
            ].candidates
            candidate = counterfactual_program_candidate(candidates)
            if candidate is None:
                continue
            try:
                frozen[round_id] = await self.backend.bind_frozen_program(
                    run_id=state["run_id"],
                    query=state["query"],
                    projection_id=state["projection_id"],
                    route_decision=state["route_decision"],
                    candidate=candidate,
                    round_id=round_id,
                    counterfactual=(
                        round_id != self.config.active_round
                        or state["route_decision"].outcome is not QueryRouteOutcome.PROGRAM
                    ),
                )
            except (SQLAlchemyError, ExperimentPersistenceError):
                raise
            except Exception as exc:
                errors[round_id] = type(exc).__name__
        return {"frozen_by_round": frozen, "program_binding_errors": errors}

    async def _execute_memory_program(self, state: QueryExecutorGraphState) -> dict[str, Any]:
        ordered = sorted(state["frozen_by_round"].items())
        grouped: dict[str, list[tuple[int, FrozenProgramBinding]]] = {}
        for round_id, frozen in ordered:
            identity = canonical_content_hash(
                [
                    state["run_id"],
                    state["query"].query_id,
                    frozen.program.plan_id,
                    frozen.source_program_hash,
                ]
            )
            grouped.setdefault(identity, []).append((round_id, frozen))

        async def execute(
            round_id: int,
            frozen: FrozenProgramBinding,
        ) -> tuple[int, ProgramExecutionResult | None, str | None]:
            try:
                return (
                    round_id,
                    await self.backend.execute_memory_program(state["query"], frozen),
                    None,
                )
            except (SQLAlchemyError, ExperimentPersistenceError):
                raise
            except Exception as exc:
                return round_id, None, type(exc).__name__

        representatives = [values[0] for _identity, values in sorted(grouped.items())]
        outcomes = await asyncio.gather(
            *(execute(round_id, frozen) for round_id, frozen in representatives),
            return_exceptions=True,
        )
        values = []
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
            values.append(outcome)
        representative_results = {
            round_id: value for round_id, value, _error in values if value is not None
        }
        results: dict[int, ProgramExecutionResult] = {}
        reuse_links: list[QueryExecutionReuseLink] = []
        for identity, members in sorted(grouped.items()):
            source_round = members[0][0]
            source_result = representative_results.get(source_round)
            if source_result is None:
                continue
            for round_id, _frozen in members:
                results[round_id] = source_result
                if round_id == source_round:
                    continue
                payload = {
                    "run_id": state["run_id"],
                    "query_id": state["query"].query_id,
                    "source_execution_id": source_result.trace.execution_id,
                    "target_round_id": round_id,
                    "execution_identity_hash": identity,
                }
                reuse_links.append(
                    QueryExecutionReuseLink(
                        execution_reuse_link_id=stable_compiler_id(
                            "query-execution-reuse", payload
                        ),
                        query_executor_run_id=state["run_id"],
                        query_id=state["query"].query_id,
                        source_execution_id=source_result.trace.execution_id,
                        target_round_id=round_id,
                        execution_identity_hash=identity,
                        content_hash=canonical_content_hash(payload),
                    )
                )
        representative_errors = {
            round_id: error for round_id, _value, error in values if error is not None
        }
        errors: dict[int, str] = {}
        for members in grouped.values():
            source_round = members[0][0]
            error = representative_errors.get(source_round)
            if error is not None:
                errors.update({round_id: error for round_id, _frozen in members})
        return {
            "execution_by_round": results,
            "program_execution_errors": errors,
            "execution_reuse_links": tuple(reuse_links),
        }

    async def _runtime_evidence_sufficiency(self, state: QueryExecutorGraphState) -> dict[str, Any]:
        assessments: dict[int, QueryEvidenceSufficiencyAssessment] = {}
        relevance_by_round: dict[int, RuntimeEvidenceRelevance] = {}
        for round_id, result in sorted(state["execution_by_round"].items()):
            relevance = (
                await self.backend.relevant_runtime_evidence_groups(
                    query=state["query"],
                    frozen=state["frozen_by_round"][round_id],
                    result=result,
                )
                if self.config.is_v2
                else None
            )
            if relevance is not None:
                relevance_by_round[round_id] = relevance
            relevant_groups = set(relevance.group_ids) if relevance is not None else None
            assessments[round_id] = assess_query_evidence_sufficiency(
                query_executor_run_id=state["run_id"],
                binding=state["frozen_by_round"][round_id].binding,
                requirement=state["frozen_by_round"][round_id].requirement,
                result=result,
                group_resolver=self.group_resolver,
                relevant_group_ids=relevant_groups,
                relevance_threshold=(
                    self.config.query_evidence_relevance_threshold if self.config.is_v2 else None
                ),
                answer_support=(relevance.assessment.passed if relevance is not None else None),
            )
        return {
            "sufficiency_by_round": assessments,
            "relevance_by_round": relevance_by_round,
        }

    async def _assemble_and_generate_pmmc_answer(
        self, state: QueryExecutorGraphState
    ) -> dict[str, Any]:
        abstentions: dict[int, dict[str, Any]] = {}

        async def generate(
            round_id: int,
        ) -> tuple[
            int,
            FinalAnswerRecord | None,
            str | None,
            tuple[tuple[str, ...], ...],
        ]:
            try:
                return (
                    round_id,
                    await self.backend.generate_pmmc_answer(
                        query=state["query"],
                        frozen=state["frozen_by_round"][round_id],
                        result=state["execution_by_round"][round_id],
                        variant=QueryAnswerVariant(f"pmmc_r{round_id}"),
                    ),
                    None,
                    (),
                )
            except InvalidRuntimeAnswer as exc:
                return (
                    round_id,
                    None,
                    type(exc).__name__,
                    tuple(
                        tuple(item.value for item in failures) for failures in exc.attempt_failures
                    ),
                )
            except UnsupportedRuntimeAnswer as exc:
                if exc.fact is not None:
                    abstentions[round_id] = exc.fact.model_dump(mode="json")
                return round_id, None, type(exc).__name__, ()
            except (SQLAlchemyError, ExperimentPersistenceError):
                raise
            except Exception as exc:
                # The terminal selector records a safe typed failure/fallback. Raw
                # provider diagnostics remain in the audited model artifact.
                return round_id, None, type(exc).__name__, ()

        eligible_rounds = [
            round_id
            for round_id, assessment in sorted(state["sufficiency_by_round"].items())
            if assessment.sufficient
        ]
        context_groups: dict[str, list[int]] = {}
        for round_id in eligible_rounds:
            result = state["execution_by_round"][round_id]
            context_hash = canonical_content_hash(
                {
                    "query_id": state["query"].query_id,
                    "evidence": [
                        {
                            "type": item.evidence_type.value,
                            "id": item.evidence_id,
                            "score": (
                                item.score.model_dump(mode="json")
                                if item.score is not None
                                else None
                            ),
                            "image_id": item.image_id,
                        }
                        for item in sorted(
                            result.evidence,
                            key=lambda value: (
                                value.evidence_type.value,
                                value.evidence_id,
                                value.step_id,
                                value.rank,
                            ),
                        )
                    ],
                }
            )
            context_groups.setdefault(context_hash, []).append(round_id)
        representatives = [rounds[0] for _context_hash, rounds in sorted(context_groups.items())]
        outcomes = await asyncio.gather(
            *(generate(round_id) for round_id in representatives),
            return_exceptions=True,
        )
        generated = []
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
            generated.append(outcome)
        generated_by_round = {value[0]: value for value in generated}
        answers: dict[int, FinalAnswerRecord] = {}
        errors: dict[int, str] = {}
        failures: dict[int, tuple[tuple[str, ...], ...]] = {}
        reuse_links: list[AnswerReuseLink] = []
        for context_hash, rounds in sorted(context_groups.items()):
            source_round = rounds[0]
            _round, source_answer, error, attempt_failures = generated_by_round[source_round]
            if source_answer is None:
                if error is not None:
                    errors.update({round_id: error for round_id in rounds})
                if attempt_failures:
                    failures.update({round_id: attempt_failures for round_id in rounds})
                continue
            answers[source_round] = source_answer
            for target_round in rounds[1:]:
                target_variant = QueryAnswerVariant(f"pmmc_r{target_round}")
                target_answer_id = stable_compiler_id(
                    "method-answer",
                    [
                        state["run_id"],
                        state["query"].query_id,
                        "pmmc",
                        target_variant.value,
                    ],
                )
                target_payload = source_answer.model_dump(mode="python")
                target_payload.update(
                    {
                        "answer_id": target_answer_id,
                        "answer_variant": target_variant.value,
                        "metadata": {
                            **source_answer.metadata,
                            "answer_reused": True,
                            "source_answer_id": source_answer.answer_id,
                            "answer_context_hash": context_hash,
                        },
                    }
                )
                answers[target_round] = FinalAnswerRecord.model_validate(target_payload)
                link_payload = {
                    "run_id": state["run_id"],
                    "query_id": state["query"].query_id,
                    "source_answer_id": source_answer.answer_id,
                    "target_variant": target_variant.value,
                    "context_hash": context_hash,
                }
                reuse_links.append(
                    AnswerReuseLink(
                        answer_reuse_link_id=stable_compiler_id("answer-reuse-link", link_payload),
                        query_executor_run_id=state["run_id"],
                        query_id=state["query"].query_id,
                        source_answer_id=source_answer.answer_id,
                        target_variant=target_variant.value,
                        context_hash=context_hash,
                        content_hash=canonical_content_hash(link_payload),
                    )
                )
        return {
            "pmmc_answers": answers,
            "pmmc_abstentions": abstentions,
            "pmmc_answer_errors": errors,
            "answer_contract_failures": failures,
            "answer_reuse_links": tuple(reuse_links),
        }

    async def _load_or_run_native_mm_fallback(
        self, state: QueryExecutorGraphState
    ) -> dict[str, Any]:
        try:
            answer = await self.backend.run_native_mm_counterfactual(
                state["run_id"], state["query"]
            )
        except Exception as exc:
            if isinstance(exc, UnsupportedRuntimeAnswer) and exc.fact is not None:
                return {
                    "fallback_run_id": self.backend.fallback_run_id,
                    "fallback_error_type": "ValidatedAnswerAbstention",
                    "fallback_abstention": exc.fact.model_dump(mode="json"),
                }
            return {
                "fallback_run_id": self.backend.fallback_run_id,
                "fallback_error_type": type(exc).__name__,
            }
        return {
            "fallback_run_id": self.backend.fallback_run_id,
            "fallback_answer": answer,
        }

    async def _select_active_answer(self, state: QueryExecutorGraphState) -> dict[str, Any]:
        route = state["route_decision"]
        active_assessment = state.get("sufficiency_by_round", {}).get(self.config.active_round)
        pmmc = state.get("pmmc_answers", {}).get(self.config.active_round)
        fallback = state.get("fallback_answer")
        reason = route.fallback_reason
        if route.outcome is QueryRouteOutcome.PROGRAM:
            if active_assessment is None:
                reason = (
                    QueryFallbackReason.PROGRAM_EXECUTION_FAILED
                    if self.config.active_round in state.get("program_execution_errors", {})
                    else QueryFallbackReason.PROGRAM_VALIDATION_FAILED
                )
            elif not active_assessment.sufficient:
                reason = active_assessment.failure_reason
            elif pmmc is None:
                reason = QueryFallbackReason.ANSWER_GENERATION_FAILED
            else:
                reason = None
        active_fallback = reason is not None
        fallback_payload = {
            "run_id": state["run_id"],
            "query_id": state["query_id"],
            "fallback_answer_id": fallback.answer_id if fallback else None,
            "reason": reason.value if reason else None,
            "error_type": state.get("fallback_error_type"),
            "abstention_hash": state.get("fallback_abstention", {}).get("content_hash"),
        }
        fallback_decision = QueryFallbackDecision(
            fallback_decision_id=stable_compiler_id("runtime-fallback-decision", fallback_payload),
            query_executor_run_id=state["run_id"],
            query_id=state["query_id"],
            fallback_run_id=(
                str(fallback.metadata.get("fallback_run_id", fallback.run_id))
                if fallback
                else state["fallback_run_id"]
            ),
            fallback_query_id=state["query_id"],
            counterfactual_completed=fallback is not None,
            active_fallback=active_fallback,
            reason=reason,
            fallback_answer_id=fallback.answer_id if fallback else None,
            error_type=state.get("fallback_error_type"),
            content_hash=canonical_content_hash(fallback_payload),
            metadata={
                "counterfactual_terminal_outcome": (
                    "answered"
                    if fallback
                    else "abstained"
                    if state.get("fallback_abstention")
                    else "failed"
                ),
                "abstention": state.get("fallback_abstention"),
            },
        )
        evaluation_aliases: list[FinalAnswerRecord] = []
        alias_reuse_links: list[AnswerReuseLink] = []

        def project_answer(
            source: FinalAnswerRecord,
            *,
            variant: QueryAnswerVariant,
            method: str,
        ) -> FinalAnswerRecord:
            alias_payload = {
                "run_id": state["run_id"],
                "query_id": state["query_id"],
                "source_answer_id": source.answer_id,
                "source_run_id": source.run_id,
                "source_variant": source.answer_variant,
                "target_variant": variant.value,
            }
            context_hash = str(
                source.metadata.get("answer_context_hash", canonical_content_hash(alias_payload))
            )
            projected = source.model_copy(
                update={
                    "answer_id": stable_compiler_id(
                        "pmmc-answer-variant-projection", alias_payload
                    ),
                    "run_id": state["run_id"],
                    "answer_variant": variant.value,
                    "method": method,
                    "model_call_id": None,
                    "metadata": {
                        "answer_reused": True,
                        "source_answer_id": source.answer_id,
                        "source_run_id": source.run_id,
                        "source_variant": source.answer_variant,
                        "answer_context_hash": context_hash,
                    },
                }
            )
            alias_reuse_links.append(
                AnswerReuseLink(
                    answer_reuse_link_id=stable_compiler_id("answer-reuse-link", alias_payload),
                    query_executor_run_id=state["run_id"],
                    query_id=state["query_id"],
                    source_answer_id=source.answer_id,
                    target_variant=variant.value,
                    context_hash=context_hash,
                    content_hash=canonical_content_hash(alias_payload),
                )
            )
            evaluation_aliases.append(projected)
            return projected

        selected_source = fallback if active_fallback else pmmc
        selected = selected_source
        if self.config.is_v2:
            if fallback is not None:
                fallback_alias = project_answer(
                    fallback,
                    variant=QueryAnswerVariant.NATIVE_MM_COUNTERFACTUAL,
                    method="native_multimodal_rag",
                )
                if active_fallback:
                    selected = fallback_alias
            if pmmc is not None:
                project_answer(
                    pmmc,
                    variant=QueryAnswerVariant.NO_FALLBACK_COUNTERFACTUAL,
                    method="pmmc",
                )
        terminal_failure = selected is None
        active_answer = None
        if selected is not None:
            alias_payload = {
                "run_id": state["run_id"],
                "query_id": state["query_id"],
                "selected_answer_id": selected.answer_id,
                "source_variant": selected.answer_variant,
            }
            if self.config.is_v2:
                assert selected_source is not None
                active_answer = project_answer(
                    selected_source,
                    variant=QueryAnswerVariant.ACTIVE_FINAL_PROVISIONAL,
                    method="pmmc",
                ).model_copy(
                    update={
                        "metadata": {
                            **evaluation_aliases[-1].metadata,
                            "active_fallback": active_fallback,
                        }
                    }
                )
                evaluation_aliases[-1] = active_answer
            else:
                active_answer = selected.model_copy(
                    update={
                        "answer_id": stable_compiler_id("pmmc-active-final", alias_payload),
                        "run_id": state["run_id"],
                        "answer_variant": QueryAnswerVariant.ACTIVE_FINAL.value,
                        "method": "pmmc",
                        "model_call_id": None,
                        "metadata": {
                            "selected_source_answer_id": selected.answer_id,
                            "selected_source_run_id": selected.run_id,
                            "selected_source_variant": selected.answer_variant,
                            "active_fallback": active_fallback,
                        },
                    }
                )
        selection_payload = {
            "run_id": state["run_id"],
            "query_id": state["query_id"],
            "selected": active_answer.answer_id if active_answer else None,
            "active_fallback": active_fallback,
            "terminal_failure": terminal_failure,
        }
        answer_selection = QueryAnswerSelection(
            answer_selection_id=stable_compiler_id("runtime-answer-selection", selection_payload),
            query_executor_run_id=state["run_id"],
            query_id=state["query_id"],
            active_variant=(
                QueryAnswerVariant.NATIVE_MM_COUNTERFACTUAL
                if active_fallback
                else QueryAnswerVariant.PMMC_R2
            ),
            selected_answer_id=active_answer.answer_id if active_answer else None,
            pmmc_answer_id=pmmc.answer_id if pmmc else None,
            fallback_answer_id=fallback.answer_id if fallback else None,
            terminal_failure=terminal_failure,
            failure_type=(
                state.get("fallback_error_type", "NoUsableAnswer") if terminal_failure else None
            ),
            content_hash=canonical_content_hash(selection_payload),
        )
        return {
            "fallback_decision": fallback_decision,
            "answer_selection": answer_selection,
            "answer_reuse_links": (
                *state.get("answer_reuse_links", ()),
                *alias_reuse_links,
            ),
            "evaluation_alias_answers": tuple(evaluation_aliases),
            **({"active_answer": active_answer} if active_answer is not None else {}),
        }

    async def _persist_terminal_facts(self, state: QueryExecutorGraphState) -> dict[str, Any]:
        await self.backend.persist_terminal(state)
        return {}


__all__ = [
    "FrozenProgramBinding",
    "OnlineQueryExecutor",
    "OnlineQueryExecutorBackend",
    "QueryExecutorGraphState",
]
