"""LangGraph control flow for one answer evaluation unit."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, TypedDict, cast

from pmmc.evaluation.bad_cases import BadCaseInputs, classify_bad_cases
from pmmc.evaluation.config import BaselineEvaluationConfig
from pmmc.evaluation.evidence import resolve_gold_evidence, score_evidence
from pmmc.evaluation.facts import PostgresEvaluationCaseLoader, RuntimeEvaluationFacts
from pmmc.evaluation.judge import BaselineJudgeService, judge_score_from_trace
from pmmc.evaluation.metrics import AnswerMetricBundle, score_answer
from pmmc.infrastructure.database.evaluation_repositories import EvaluationUnitOfWork
from pmmc.schemas import (
    BadCaseLabel,
    ContextEvidenceEvaluation,
    EvaluationJob,
    EvaluationStage,
    JudgeScore,
    MetricScore,
    QueryGold,
    ResolvedGoldEvidenceGroup,
    RetrievalEvaluation,
)


class EvaluationGraphState(TypedDict, total=False):
    job: EvaluationJob
    stage: EvaluationStage
    runtime: RuntimeEvaluationFacts
    gold: QueryGold
    groups: tuple[ResolvedGoldEvidenceGroup, ...]
    answer_bundle: AnswerMetricBundle
    metrics: tuple[MetricScore, ...]
    retrieval_evaluations: tuple[RetrievalEvaluation, ...]
    context_evaluations: tuple[ContextEvidenceEvaluation, ...]
    judge: JudgeScore
    bad_cases: tuple[BadCaseLabel, ...]


class _LinearCompiledGraph:
    """Minimal fallback for schema-only environments without LangGraph installed."""

    def __init__(
        self,
        nodes: Sequence[Callable[[EvaluationGraphState], Awaitable[dict[str, Any]]]],
    ) -> None:
        self.nodes = list(nodes)

    async def ainvoke(self, state: EvaluationGraphState) -> EvaluationGraphState:
        current = dict(state)
        for node in self.nodes:
            current.update(await node(current))  # type: ignore[arg-type]
        return current  # type: ignore[return-value]


class _CompiledEvaluationGraph(Protocol):
    async def ainvoke(self, state: EvaluationGraphState) -> EvaluationGraphState: ...


class BaselineEvaluationGraph:
    """Fixed method-independent graph for one evaluation job."""

    node_names = (
        "load_runtime_facts",
        "load_gold",
        "resolve_gold_evidence",
        "normalize_answer",
        "score_answer",
        "score_retrieval_and_context",
        "judge",
        "classify_bad_case",
        "persist",
    )

    def __init__(
        self,
        *,
        loader: PostgresEvaluationCaseLoader,
        sessions: Any,
        config: BaselineEvaluationConfig,
        judge_service: BaselineJudgeService | None,
    ) -> None:
        self.loader = loader
        self.sessions = sessions
        self.config = config
        self.judge_service = judge_service
        callbacks = [
            self._load_runtime_facts,
            self._load_gold,
            self._resolve_gold_evidence,
            self._normalize_answer,
            self._score_answer,
            self._score_retrieval_and_context,
            self._judge,
            self._classify_bad_case,
            self._persist,
        ]
        self.compiled: _CompiledEvaluationGraph
        try:
            from langgraph.graph import END, START, StateGraph
        except ModuleNotFoundError:
            self.compiled = _LinearCompiledGraph(callbacks)
        else:
            graph = StateGraph(EvaluationGraphState)
            for name, callback in zip(self.node_names, callbacks, strict=True):
                graph.add_node(name, callback)
            graph.add_edge(START, self.node_names[0])
            for source, target in zip(
                self.node_names, self.node_names[1:], strict=False
            ):
                graph.add_edge(source, target)
            graph.add_edge(self.node_names[-1], END)
            self.compiled = cast(_CompiledEvaluationGraph, graph.compile())

    async def evaluate(
        self, job: EvaluationJob, stage: EvaluationStage
    ) -> EvaluationGraphState:
        return await self.compiled.ainvoke({"job": job, "stage": stage})

    async def _load_runtime_facts(
        self, state: EvaluationGraphState
    ) -> dict[str, Any]:
        return {"runtime": await self.loader.load_runtime(state["job"])}

    async def _load_gold(self, state: EvaluationGraphState) -> dict[str, Any]:
        return {"gold": await self.loader.load_gold(state["job"])}

    async def _resolve_gold_evidence(
        self, state: EvaluationGraphState
    ) -> dict[str, Any]:
        if state["stage"] is not EvaluationStage.DETERMINISTIC:
            return {}
        runtime = state["runtime"]
        job = state["job"]
        return {
            "groups": resolve_gold_evidence(
                evaluation_run_id=job.evaluation_run_id,
                snapshot_id=job.snapshot_id,
                query_id=job.query_id,
                dataset=runtime.query.query.dataset,
                gold=state["gold"],
                memories=list(runtime.memories),
                policy_version=self.config.evidence_policy_version,
                created_at=job.created_at,
            )
        }

    async def _normalize_answer(self, state: EvaluationGraphState) -> dict[str, Any]:
        if state["stage"] is not EvaluationStage.DETERMINISTIC:
            return {}
        runtime = state["runtime"]
        job = state["job"]
        return {
            "answer_bundle": score_answer(
                evaluation_run_id=job.evaluation_run_id,
                answer_id=job.answer_id,
                query_id=job.query_id,
                run_id=job.run_id,
                candidate=runtime.answer.final_answer,
                reference=state["gold"].answer,
                normalizer_version=self.config.normalizer_version,
                created_at=job.created_at,
            )
        }

    async def _score_answer(self, state: EvaluationGraphState) -> dict[str, Any]:
        if state["stage"] is not EvaluationStage.DETERMINISTIC:
            return {}
        return {"metrics": state["answer_bundle"].metrics}

    async def _score_retrieval_and_context(
        self, state: EvaluationGraphState
    ) -> dict[str, Any]:
        if state["stage"] is not EvaluationStage.DETERMINISTIC:
            return {}
        runtime = state["runtime"]
        job = state["job"]
        bundle = score_evidence(
            evaluation_run_id=job.evaluation_run_id,
            answer_id=job.answer_id,
            query_id=job.query_id,
            run_id=job.run_id,
            context_source_kind=runtime.context_source_kind,
            context_assembly_id=runtime.context_assembly_id,
            program_execution_id=runtime.program_execution_id,
            groups=state["groups"],
            retrieval_candidates=list(runtime.retrieval_candidates),
            context_candidate=runtime.context_candidate,
            retrieval_trace_ids=[item.retrieval_trace_id for item in runtime.retrieval_traces],
            ks=tuple(self.config.retrieval_ks),
            policy_version=self.config.evidence_policy_version,
            created_at=job.created_at,
        )
        return {
            "metrics": (*state["metrics"], *bundle.metrics),
            "retrieval_evaluations": bundle.retrieval_evaluations,
            "context_evaluations": bundle.context_evaluations,
        }

    async def _judge(self, state: EvaluationGraphState) -> dict[str, Any]:
        if state["stage"] is not EvaluationStage.JUDGE:
            return {}
        job = state["job"]
        runtime = state["runtime"]
        reference = state["gold"].answer
        if reference is None:
            raise ValueError("judge requires a canonical reference answer")
        successful = await self.loader.load_successful_judge_call(
            job.evaluation_run_id, job.evaluation_job_id
        )
        if successful is not None:
            return {
                "judge": judge_score_from_trace(
                    trace=successful,
                    evaluation_run_id=job.evaluation_run_id,
                    answer_id=job.answer_id,
                    query_id=job.query_id,
                    run_id=job.run_id,
                    config=self.config,
                )
            }
        if self.judge_service is None:
            raise RuntimeError("judge stage requires a configured judge service")
        return {
            "judge": await self.judge_service.judge(
                evaluation_run_id=job.evaluation_run_id,
                evaluation_job_id=job.evaluation_job_id,
                attempt_number=job.attempt_count,
                run_id=job.run_id,
                snapshot_id=job.snapshot_id,
                answer_id=job.answer_id,
                query=runtime.query.query,
                candidate_answer=runtime.answer.final_answer,
                reference_answer=reference,
            )
        }

    async def _classify_bad_case(
        self, state: EvaluationGraphState
    ) -> dict[str, Any]:
        if state["stage"] is not EvaluationStage.ANALYZE:
            return {}
        job = state["job"]
        existing = await self.loader.load_existing(job.evaluation_run_id, job.answer_id)
        groups_resolvable = any(
            item.metric_name == "context_evidence_recall" and item.applicable
            for item in existing.metrics
        )
        labels = classify_bad_cases(
            evaluation_run_id=job.evaluation_run_id,
            answer_id=job.answer_id,
            query_id=job.query_id,
            run_id=job.run_id,
            metrics=list(existing.metrics),
            judge=existing.judge,
            inputs=BadCaseInputs(
                gold_resolvable=groups_resolvable,
                context_budget_excluded=state["runtime"].context_budget_excluded,
            ),
            rule_version=self.config.bad_case_rule_version,
            created_at=job.created_at,
        )
        return {"bad_cases": tuple(labels)}

    async def _persist(self, state: EvaluationGraphState) -> dict[str, Any]:
        stage = state["stage"]
        async with EvaluationUnitOfWork(self.sessions) as uow:
            if stage is EvaluationStage.DETERMINISTIC:
                await uow.facts.add_normalization(state["answer_bundle"].normalization)
                for group in state["groups"]:
                    await uow.facts.add_gold_group(group)
                for retrieval_evaluation in state.get("retrieval_evaluations", ()):
                    await uow.facts.add_retrieval(retrieval_evaluation)
                for context_evaluation in state.get("context_evaluations", ()):
                    await uow.facts.add_context(context_evaluation)
                for metric in state["metrics"]:
                    await uow.facts.add_metric(metric)
            elif stage is EvaluationStage.JUDGE:
                await uow.facts.add_judge(state["judge"])
            else:
                for label in state["bad_cases"]:
                    await uow.facts.add_bad_case(label)
        return {}
