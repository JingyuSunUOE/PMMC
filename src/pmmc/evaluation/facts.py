"""Evaluation-role readers that materialize runtime facts without mutating them."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.evaluation.evidence import RankedEvidenceCandidate
from pmmc.evaluation.repositories import PostgresGoldRepository
from pmmc.infrastructure.database.baseline_models import ContextAssemblyModel
from pmmc.infrastructure.database.baseline_repositories import (
    PostgresContextAssemblyRepository,
    PostgresCorpusRepository,
)
from pmmc.infrastructure.database.compiler_models import ProgramExecutionEvidenceRefModel
from pmmc.infrastructure.database.experiment_models import (
    EvaluationModelCallModel,
    FinalAnswerModel,
    JudgeScoreModel,
    MetricScoreModel,
    RetrievalBatchModel,
    RuntimeModelCallModel,
)
from pmmc.infrastructure.database.experiment_repositories import (
    PostgresAnswerRepository,
    PostgresRetrievalRepository,
)
from pmmc.infrastructure.database.query_executor_models import (
    PmmcFallbackQueryLinkModel,
    QueryAnswerSelectionModel,
    QueryEvidenceSufficiencyAssessmentModel,
    QueryProgramBindingModel,
)
from pmmc.infrastructure.database.repositories import (
    PostgresMemoryRepository,
    PostgresRuntimeQueryRepository,
)
from pmmc.schemas import (
    BadCaseLabel,
    BaselineMethod,
    BenchmarkQuery,
    BenchmarkSelection,
    ContextAssemblyTrace,
    ContextCandidateDisposition,
    DatasetName,
    EvaluationContextSourceKind,
    EvaluationJob,
    FinalAnswerRecord,
    JudgeScore,
    MemoryUnit,
    MetricScore,
    ModelCallTrace,
    QueryGold,
    RetrievalBatchTrace,
    RetrievalTrace,
    TraceStatus,
)


@dataclass(frozen=True)
class RuntimePmmcContext:
    """Minimal method-neutral context lineage used by the shared evaluator."""

    context_source_kind: EvaluationContextSourceKind
    included_memory_ids: tuple[str, ...]
    included_image_ids: tuple[str, ...]
    context_assembly_id: str | None = None
    program_execution_id: str | None = None
    retrieval_candidates: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if self.context_source_kind is EvaluationContextSourceKind.BASELINE_CONTEXT:
            valid = self.context_assembly_id is not None and self.program_execution_id is None
        else:
            valid = self.program_execution_id is not None and self.context_assembly_id is None
        if not valid:
            raise ValueError("PMMC runtime context requires exactly one source lineage")


@dataclass(frozen=True)
class RuntimeEvaluationFacts:
    job: EvaluationJob
    query: BenchmarkQuery
    answer: FinalAnswerRecord
    context: ContextAssemblyTrace | RuntimePmmcContext
    memories: tuple[MemoryUnit, ...]
    retrieval_batch: RetrievalBatchTrace | None
    retrieval_traces: tuple[RetrievalTrace, ...]
    retrieval_candidates: tuple[RankedEvidenceCandidate, ...]
    context_candidate: RankedEvidenceCandidate
    context_budget_excluded: bool

    @property
    def context_source_kind(self) -> EvaluationContextSourceKind:
        if isinstance(self.context, RuntimePmmcContext):
            return self.context.context_source_kind
        return EvaluationContextSourceKind.BASELINE_CONTEXT

    @property
    def context_assembly_id(self) -> str | None:
        if isinstance(self.context, RuntimePmmcContext):
            return self.context.context_assembly_id
        return self.context.context_assembly_id

    @property
    def program_execution_id(self) -> str | None:
        if isinstance(self.context, RuntimePmmcContext):
            return self.context.program_execution_id
        return None


@dataclass(frozen=True)
class ExistingEvaluationFacts:
    metrics: tuple[MetricScore, ...]
    judge: JudgeScore | None
    bad_cases: tuple[BadCaseLabel, ...]


def _candidate(
    candidate_id: str,
    rank: int,
    memory_ids: list[str],
    memory_by_id: dict[str, MemoryUnit],
    *,
    image_ids: list[str] | None = None,
) -> RankedEvidenceCandidate:
    memories = [memory_by_id[item] for item in memory_ids if item in memory_by_id]
    derived_images = {image_id for memory in memories for image_id in memory.image_ids}
    if image_ids is not None:
        derived_images &= set(image_ids)
    return RankedEvidenceCandidate(
        candidate_id=candidate_id,
        rank=rank,
        memory_ids=frozenset(memory.memory_id for memory in memories),
        session_ids=frozenset(memory.session_id for memory in memories),
        image_ids=frozenset(derived_images),
        text_memory_ids=frozenset(memory.memory_id for memory in memories if memory.text),
    )


class PostgresEvaluationCaseLoader:
    """Read runtime and gold facts through an evaluator-scoped session factory."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions
        self.queries = PostgresRuntimeQueryRepository(sessions)
        self.memories = PostgresMemoryRepository(sessions)
        self.gold = PostgresGoldRepository(sessions)

    async def load_runtime(self, job: EvaluationJob) -> RuntimeEvaluationFacts:
        async with self.sessions() as session:
            answer = await PostgresAnswerRepository(session).get(job.answer_id)
            if answer is None:
                raise KeyError(f"final answer does not exist: {job.answer_id}")
            answer_snapshot = await session.scalar(
                select(FinalAnswerModel.snapshot_id).where(
                    FinalAnswerModel.answer_id == job.answer_id
                )
            )
            if answer_snapshot != job.snapshot_id:
                raise ValueError("evaluation job and final answer snapshots differ")
        dataset = await self._query_dataset(job.snapshot_id, job.query_id)
        selected = await self.queries.select(
            job.snapshot_id,
            BenchmarkSelection(
                datasets=[dataset],
                query_ids=[job.query_id],
                include_excluded=True,
            ),
        )
        if len(selected) != 1:
            raise KeyError(f"runtime query does not exist: {job.query_id}")
        query = selected[0]
        cutoff_session_id = query.query.available_after_session
        if cutoff_session_id is None:
            raise ValueError("benchmark evaluation query requires a prospective cutoff")
        visible: list[MemoryUnit] = []
        async for memory in self.memories.iter_visible(
            job.snapshot_id,
            query.query.conversation_id,
            cutoff_session_id,
        ):
            visible.append(memory)
        memory_by_id = {item.memory_id: item for item in visible}

        if answer.method == "pmmc" or job.metadata.get("method") == "pmmc":
            return await self._load_pmmc_runtime(
                job=job,
                query=query,
                answer=answer,
                visible=visible,
                memory_by_id=memory_by_id,
            )

        async with self.sessions() as session:
            context_id = None
            if answer.model_call_id is not None:
                context_id = await session.scalar(
                    select(RuntimeModelCallModel.context_assembly_id).where(
                        RuntimeModelCallModel.model_call_id == answer.model_call_id
                    )
                )
            if context_id is None:
                context_id = await session.scalar(
                    select(ContextAssemblyModel.context_assembly_id).where(
                        ContextAssemblyModel.run_id == job.run_id,
                        ContextAssemblyModel.query_id == job.query_id,
                    )
                )
            if context_id is None:
                raise KeyError("answer has no context assembly lineage")
            context = await PostgresContextAssemblyRepository(session).get(context_id)
            batch_model = await session.scalar(
                select(RetrievalBatchModel).where(
                    RetrievalBatchModel.run_id == job.run_id,
                    RetrievalBatchModel.query_id == job.query_id,
                )
            )
            batch = None
            traces: list[RetrievalTrace] = []
            candidates: list[RankedEvidenceCandidate] = []
            if batch_model is not None:
                loaded = await PostgresRetrievalRepository(session).get_batch(
                    batch_model.retrieval_batch_id
                )
                if loaded is None:
                    raise KeyError("retrieval batch lineage cannot be materialized")
                batch, traces, _components, _query_embeddings = loaded
                if batch.corpus_id is None:
                    raise ValueError("RAG retrieval batch is missing corpus lineage")
                corpus = PostgresCorpusRepository(session)
                for trace in traces:
                    if trace.retrieval_window_id is None:
                        raise ValueError("baseline retrieval trace is not window-level")
                    window = await corpus.get_window(batch.corpus_id, trace.retrieval_window_id)
                    candidates.append(
                        _candidate(
                            trace.retrieval_trace_id,
                            trace.rank,
                            window.memory_ids,
                            memory_by_id,
                        )
                    )

        context_candidate = _candidate(
            context.context_assembly_id,
            1,
            context.included_memory_ids,
            memory_by_id,
            image_ids=answer.image_ids_in_prompt,
        )
        context_budget_excluded = any(
            item.disposition is ContextCandidateDisposition.BUDGET_EXCLUDED
            for item in context.retrieval_candidates
        )
        method = BaselineMethod(answer.method)
        if (
            method
            in {
                BaselineMethod.FULL_CONTEXT_CAPTIONS,
                BaselineMethod.FULL_CONTEXT_RAW_IMAGES,
            }
            and batch is not None
        ):
            raise ValueError("full-context evaluation found unexpected retrieval facts")
        return RuntimeEvaluationFacts(
            job=job,
            query=query,
            answer=answer,
            context=context,
            memories=tuple(visible),
            retrieval_batch=batch,
            retrieval_traces=tuple(traces),
            retrieval_candidates=tuple(candidates),
            context_candidate=context_candidate,
            context_budget_excluded=context_budget_excluded,
        )

    async def _load_pmmc_runtime(
        self,
        *,
        job: EvaluationJob,
        query: BenchmarkQuery,
        answer: FinalAnswerRecord,
        visible: list[MemoryUnit],
        memory_by_id: dict[str, MemoryUnit],
    ) -> RuntimeEvaluationFacts:
        """Materialize operator evidence without requiring a baseline context row."""

        fallback_context: ContextAssemblyTrace | None = None
        async with self.sessions() as session:
            selection = await session.scalar(
                select(QueryAnswerSelectionModel).where(
                    QueryAnswerSelectionModel.query_executor_run_id == job.run_id,
                    QueryAnswerSelectionModel.query_id == job.query_id,
                )
            )
            if selection is None:
                raise ValueError("PMMC answer selection lineage is incomplete")
            evaluated_variant = job.answer_variant
            source_variant = evaluated_variant
            if evaluated_variant in {"active_final", "active_final_provisional"}:
                if selection.selected_answer_id != answer.answer_id:
                    raise ValueError("PMMC active answer selection lineage is incomplete")
                source_variant = selection.active_variant
            elif evaluated_variant == "no_fallback_counterfactual":
                source_variant = str(answer.metadata.get("source_variant", ""))
            execution_id: str | None = None
            if source_variant.startswith("pmmc_r"):
                try:
                    active_round = int(source_variant.removeprefix("pmmc_r"))
                except ValueError as exc:
                    raise ValueError("PMMC answer has an invalid round variant") from exc
                if active_round not in {0, 1, 2, 3}:
                    raise ValueError("PMMC answer round is outside the frozen grid")
                execution_id = await session.scalar(
                    select(QueryEvidenceSufficiencyAssessmentModel.execution_id)
                    .join(
                        QueryProgramBindingModel,
                        QueryProgramBindingModel.query_program_binding_id
                        == QueryEvidenceSufficiencyAssessmentModel.query_program_binding_id,
                    )
                    .where(
                        QueryEvidenceSufficiencyAssessmentModel.query_executor_run_id == job.run_id,
                        QueryEvidenceSufficiencyAssessmentModel.query_id == job.query_id,
                        QueryProgramBindingModel.round_id == active_round,
                    )
                )
                if execution_id is None:
                    raise ValueError("PMMC answer has no program execution lineage")
            elif source_variant == "native_mm_counterfactual":
                fallback_link = await session.scalar(
                    select(PmmcFallbackQueryLinkModel).where(
                        PmmcFallbackQueryLinkModel.query_executor_run_id == job.run_id,
                        PmmcFallbackQueryLinkModel.query_id == job.query_id,
                    )
                )
                if (
                    fallback_link is None
                    or fallback_link.fallback_answer_id
                    != selection.fallback_answer_id
                ):
                    raise ValueError("PMMC fallback answer lineage is incomplete")
                fallback_model_call_id = await session.scalar(
                    select(FinalAnswerModel.model_call_id).where(
                        FinalAnswerModel.answer_id == fallback_link.fallback_answer_id,
                        FinalAnswerModel.run_id == fallback_link.fallback_run_id,
                        FinalAnswerModel.query_id == fallback_link.fallback_query_id,
                    )
                )
                context_id = None
                if fallback_model_call_id is not None:
                    context_id = await session.scalar(
                        select(RuntimeModelCallModel.context_assembly_id).where(
                            RuntimeModelCallModel.model_call_id == fallback_model_call_id,
                            RuntimeModelCallModel.run_id == fallback_link.fallback_run_id,
                            RuntimeModelCallModel.query_id == fallback_link.fallback_query_id,
                        )
                    )
                if context_id is None:
                    context_id = await session.scalar(
                        select(ContextAssemblyModel.context_assembly_id).where(
                            ContextAssemblyModel.run_id == fallback_link.fallback_run_id,
                            ContextAssemblyModel.query_id == fallback_link.fallback_query_id,
                        )
                    )
                if context_id is None:
                    raise ValueError("PMMC fallback answer has no context assembly lineage")
                fallback_context = await PostgresContextAssemblyRepository(session).get(context_id)
            else:
                raise ValueError("PMMC answer has an unsupported source variant")
            refs = []
            if execution_id is not None:
                refs = list(
                    (
                        await session.execute(
                            select(ProgramExecutionEvidenceRefModel)
                            .where(ProgramExecutionEvidenceRefModel.execution_id == execution_id)
                            .order_by(ProgramExecutionEvidenceRefModel.rank)
                        )
                    )
                    .scalars()
                    .all()
                )

        if fallback_context is not None:
            memory_ids = [
                item for item in fallback_context.included_memory_ids if item in memory_by_id
            ]
            image_ids = list(fallback_context.included_image_ids)
            context: ContextAssemblyTrace | RuntimePmmcContext = fallback_context
            context_candidate_id = fallback_context.context_assembly_id
            context_budget_excluded = any(
                item.disposition is ContextCandidateDisposition.BUDGET_EXCLUDED
                for item in fallback_context.retrieval_candidates
            )
        else:
            assert execution_id is not None
            memory_ids = list(
                dict.fromkeys(
                    [row.memory_id for row in refs if row.memory_id in memory_by_id]
                    + [item for item in answer.evidence_ids_in_prompt if item in memory_by_id]
                )
            )
            image_ids = list(
                dict.fromkeys(
                    [row.image_id for row in refs if row.image_id is not None]
                    + list(answer.image_ids_in_prompt)
                )
            )
            context = RuntimePmmcContext(
                context_source_kind=EvaluationContextSourceKind.PROGRAM_EXECUTION,
                included_memory_ids=tuple(memory_ids),
                included_image_ids=tuple(image_ids),
                program_execution_id=execution_id,
            )
            context_candidate_id = execution_id
            context_budget_excluded = False
        memories = [memory_by_id[item] for item in memory_ids]
        return RuntimeEvaluationFacts(
            job=job,
            query=query,
            answer=answer,
            context=context,
            memories=tuple(visible),
            retrieval_batch=None,
            retrieval_traces=(),
            retrieval_candidates=(),
            context_candidate=RankedEvidenceCandidate(
                candidate_id=context_candidate_id,
                rank=1,
                memory_ids=frozenset(memory_ids),
                session_ids=frozenset(item.session_id for item in memories),
                image_ids=frozenset(image_ids),
                text_memory_ids=frozenset(item.memory_id for item in memories if item.text),
            ),
            context_budget_excluded=context_budget_excluded,
        )

    async def load_gold(self, job: EvaluationJob) -> QueryGold:
        return await self.gold.get(job.snapshot_id, job.query_id)

    async def load_existing(
        self, evaluation_run_id: str, answer_id: str
    ) -> ExistingEvaluationFacts:
        from pmmc.infrastructure.database.experiment_mappers import (
            bad_case_label_from_row,
            judge_score_from_row,
            metric_score_from_row,
        )
        from pmmc.infrastructure.database.experiment_models import BadCaseLabelModel
        from pmmc.infrastructure.database.experiment_repositories import _model_mapping

        async with self.sessions() as session:
            metrics = (
                (
                    await session.execute(
                        select(MetricScoreModel)
                        .where(
                            MetricScoreModel.evaluation_run_id == evaluation_run_id,
                            MetricScoreModel.answer_id == answer_id,
                        )
                        .order_by(MetricScoreModel.metric_name)
                    )
                )
                .scalars()
                .all()
            )
            judge_row = await session.scalar(
                select(JudgeScoreModel).where(
                    JudgeScoreModel.evaluation_run_id == evaluation_run_id,
                    JudgeScoreModel.answer_id == answer_id,
                )
            )
            bad_rows = (
                (
                    await session.execute(
                        select(BadCaseLabelModel)
                        .where(
                            BadCaseLabelModel.evaluation_run_id == evaluation_run_id,
                            BadCaseLabelModel.answer_id == answer_id,
                        )
                        .order_by(BadCaseLabelModel.is_primary.desc(), BadCaseLabelModel.category)
                    )
                )
                .scalars()
                .all()
            )
        return ExistingEvaluationFacts(
            metrics=tuple(metric_score_from_row(_model_mapping(item)) for item in metrics),
            judge=(
                judge_score_from_row(_model_mapping(judge_row)) if judge_row is not None else None
            ),
            bad_cases=tuple(bad_case_label_from_row(_model_mapping(item)) for item in bad_rows),
        )

    async def load_successful_judge_call(
        self, evaluation_run_id: str, evaluation_job_id: str
    ) -> ModelCallTrace | None:
        from pmmc.infrastructure.database.experiment_mappers import model_call_from_row
        from pmmc.infrastructure.database.experiment_repositories import _model_mapping
        from pmmc.schemas import DataScope

        async with self.sessions() as session:
            row = await session.scalar(
                select(EvaluationModelCallModel)
                .where(
                    EvaluationModelCallModel.evaluation_run_id == evaluation_run_id,
                    EvaluationModelCallModel.evaluation_job_id == evaluation_job_id,
                    EvaluationModelCallModel.status == TraceStatus.SUCCESS.value,
                )
                .order_by(
                    EvaluationModelCallModel.created_at.desc(),
                    EvaluationModelCallModel.model_call_id.desc(),
                )
                .limit(1)
            )
        return (
            model_call_from_row(_model_mapping(row), DataScope.EVALUATION)
            if row is not None
            else None
        )

    async def _query_dataset(self, snapshot_id: str, query_id: str) -> DatasetName:
        from pmmc.infrastructure.database.models import BenchmarkQueryModel

        async with self.sessions() as session:
            value = await session.scalar(
                select(BenchmarkQueryModel.dataset).where(
                    BenchmarkQueryModel.snapshot_id == snapshot_id,
                    BenchmarkQueryModel.query_id == query_id,
                )
            )
        if value is None:
            raise KeyError(f"query does not exist: {query_id}")
        return DatasetName(value)
