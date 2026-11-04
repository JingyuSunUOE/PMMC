"""Persistence interfaces consumed by PMMC domain modules."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol

from pmmc.compiler.questioner import GatedQuestionCandidate
from pmmc.memory.operators import ProgramExecutionResult
from pmmc.schemas import (
    AppStage,
    BadCaseLabel,
    BaselineCorpusManifest,
    BenchmarkQuery,
    BenchmarkSelection,
    CandidateStageJob,
    CandidateStageKind,
    CompilationStage,
    CompilationStageResult,
    CompilationUnit,
    CompilationUnitLink,
    CompilerRunSpec,
    CompilerStageJob,
    CompilerStageKind,
    ContextAssemblyTrace,
    DatasetSnapshotManifest,
    DoubterAssessment,
    DoubterFeedback,
    EmbeddingCollection,
    EmbeddingRecord,
    EmbeddingSearchHit,
    ExperimentArtifact,
    ExperimentJob,
    FinalAnswerRecord,
    ImageRef,
    JobStatus,
    JudgeScore,
    MemoryOperatorIndexManifest,
    MemoryProgram,
    MemoryUnit,
    MemoryWindow,
    MetricScore,
    ModelCallTrace,
    OperatorEvidence,
    PipelineStageSummary,
    PlannerBatchProgram,
    PlannerCandidateDraftFact,
    PlannerCandidateOutcome,
    PlannerGenerationBatch,
    PlannerRequirementContract,
    PlannerRevisionLineage,
    PlannerRevisionPatch,
    ProgramDryRunAssessment,
    ProgramEvidenceMatch,
    ProgramQueryEmbedding,
    ProgramQueryEmbeddingBatch,
    ProgramStep,
    ProgramValidationKind,
    ProgramValidationResultFact,
    QBankCoverageCandidate,
    QBankCoverageGateReport,
    QBankCoverageLabel,
    QBankCoverageMetric,
    QBankCoverageQuery,
    QBankCoverageQueryEmbedding,
    QBankCoverageRunManifest,
    QBankCoverageStatus,
    QBankDecision,
    QBankIndexManifest,
    QBankRoundProjectionManifest,
    QBankRoundProjectionMember,
    QBankRoutingItem,
    QBankSemanticMatchAssessment,
    QBankSemanticMatchJob,
    QBankSemanticMatchRun,
    QBankSemanticMatchStatus,
    QBankSemanticMetric,
    QueryGold,
    QuestionCandidate,
    QuestionCandidateGateResult,
    QuestionerCandidateSupersession,
    QuestionerCandidateVerification,
    QuestionerGeneration,
    QuestionerVerificationBatch,
    QuestionOpportunity,
    QuestionOpportunityEvidenceRef,
    QuestionOpportunitySet,
    R0InputManifest,
    RepresentationKind,
    ResolvedGoldEvidenceGroup,
    RetrievalBatchTrace,
    RetrievalDocument,
    RetrievalEvaluation,
    RetrievalQueryEmbeddingRef,
    RetrievalTrace,
    RetrievalTraceComponent,
    RoutingProjectionManifest,
    RoutingProjectionMember,
    RunManifest,
    RunQueryAssignment,
    RunState,
    RunStatus,
    SessionRecord,
    StageComponentManifest,
    StructuredOutputItemDiagnostic,
    VerifiedQBankRoutingItemV2,
    WindowSearchHit,
)


class MemoryRepository(Protocol):
    def iter_visible(
        self,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> AsyncIterator[MemoryUnit]: ...


class SessionRepository(Protocol):
    def iter_visible(
        self,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> AsyncIterator[SessionRecord]: ...


class RuntimeQueryRepository(Protocol):
    async def list_snapshot(self, snapshot_id: str) -> list[BenchmarkQuery]: ...

    async def select(
        self,
        snapshot_id: str,
        selection: BenchmarkSelection,
    ) -> list[BenchmarkQuery]: ...


class GoldRepository(Protocol):
    async def get(self, snapshot_id: str, query_id: str) -> QueryGold: ...


class ImageRepository(Protocol):
    async def get_many(self, snapshot_id: str, image_ids: list[str]) -> list[ImageRef]: ...


class SnapshotRepository(Protocol):
    async def get_manifest(self, snapshot_id: str) -> DatasetSnapshotManifest: ...


class CorpusRepository(Protocol):
    async def get_manifest(self, corpus_id: str) -> BaselineCorpusManifest: ...

    def iter_windows(self, corpus_id: str) -> AsyncIterator[MemoryWindow]: ...

    def iter_documents(self, corpus_id: str) -> AsyncIterator[RetrievalDocument]: ...

    async def get_window(self, corpus_id: str, window_id: str) -> MemoryWindow: ...

    async def get_document(self, corpus_id: str, document_id: str) -> RetrievalDocument: ...

    async def list_document_ids(
        self, corpus_id: str, representations: list[RepresentationKind]
    ) -> list[str]: ...

    async def get_documents(
        self, corpus_id: str, document_ids: list[str]
    ) -> list[RetrievalDocument]: ...


class ContextAssemblyRepository(Protocol):
    async def add(self, value: ContextAssemblyTrace) -> ContextAssemblyTrace: ...

    async def get(self, context_assembly_id: str) -> ContextAssemblyTrace: ...


class RunRepository(Protocol):
    async def register(
        self, manifest: RunManifest, assignments: list[RunQueryAssignment]
    ) -> RunState: ...

    async def get_manifest(self, run_id: str) -> RunManifest: ...

    async def add_stage_summary(
        self, run_id: str, summary: PipelineStageSummary
    ) -> PipelineStageSummary: ...

    async def get_state(self, run_id: str) -> RunState: ...

    async def transition(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error_type: str | None = None,
        error_summary: str | None = None,
        error_artifact_id: str | None = None,
    ) -> RunState: ...


class JobRepository(Protocol):
    async def add(self, job: ExperimentJob) -> ExperimentJob: ...

    async def get(self, job_id: str) -> ExperimentJob | None: ...

    async def status_counts(
        self, run_id: str, stage: AppStage | None = None
    ) -> dict[JobStatus, int]: ...

    async def claim(
        self,
        worker_id: str,
        lease_seconds: int,
        *,
        run_id: str | None = None,
        stage: AppStage | None = None,
    ) -> ExperimentJob | None: ...

    async def heartbeat(
        self, job_id: str, worker_id: str, lease_seconds: int,
        *, expected_attempt_count: int | None = None,
    ) -> ExperimentJob: ...

    async def finish(
        self,
        job_id: str,
        worker_id: str,
        *,
        success: bool,
        error_type: str | None = None,
        error_summary: str | None = None,
        error_artifact_id: str | None = None,
        retry_delay_seconds: float = 0,
        retryable: bool = True,
        expected_attempt_count: int | None = None,
    ) -> ExperimentJob: ...


class ArtifactRepository(Protocol):
    async def add(self, artifact: ExperimentArtifact) -> ExperimentArtifact: ...

    async def get(self, artifact_id: str) -> ExperimentArtifact | None: ...

    async def retry_failed(self, artifact_id: str) -> ExperimentArtifact: ...

    async def mark_ready(
        self,
        artifact_id: str,
        *,
        storage_uri: str,
        size_bytes: int,
    ) -> ExperimentArtifact: ...

    async def mark_failed(self, artifact_id: str, error_type: str) -> ExperimentArtifact: ...


class ModelCallRepository(Protocol):
    async def add(self, trace: ModelCallTrace) -> ModelCallTrace: ...

    async def get(self, model_call_id: str) -> ModelCallTrace | None: ...

    async def get_successful_for_job(self, job_id: str) -> ModelCallTrace | None: ...


class EmbeddingRepository(Protocol):
    async def add_collection(self, value: EmbeddingCollection) -> EmbeddingCollection: ...

    async def add_records(self, records: list[EmbeddingRecord]) -> None: ...

    async def mark_collection_ready(
        self,
        collection_id: str,
        *,
        index_type: str | None = None,
        index_params: dict[str, object] | None = None,
    ) -> EmbeddingCollection: ...

    async def mark_collection_failed(self, collection_id: str) -> EmbeddingCollection: ...

    async def search_exact(
        self,
        collection_id: str,
        query_vector: list[float],
        *,
        corpus_id: str,
        top_k: int,
        conversation_id: str | None = None,
        cutoff_session_id: str | None = None,
    ) -> list[EmbeddingSearchHit]: ...

    async def search_windows_exact(
        self,
        collection_id: str,
        query_vector: list[float],
        *,
        corpus_id: str,
        top_k: int,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> list[WindowSearchHit]: ...

    async def search_qbank_exact(
        self,
        collection_id: str,
        query_vector: list[float],
        *,
        compiler_run_id: str,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
        question_budget: int,
        top_k: int,
    ) -> list[tuple[str, float]]: ...

    async def get_collection(self, collection_id: str) -> EmbeddingCollection | None: ...

    async def get_record(self, embedding_id: str) -> EmbeddingRecord | None: ...

    async def get_records(self, embedding_ids: list[str]) -> dict[str, EmbeddingRecord]: ...

    async def get_query_records(
        self, collection_id: str, query_id: str
    ) -> list[EmbeddingRecord]: ...

    async def count_records(self, collection_id: str) -> int: ...


class RetrievalRepository(Protocol):
    async def add_batch(
        self,
        batch: RetrievalBatchTrace,
        traces: list[RetrievalTrace],
        *,
        components: list[RetrievalTraceComponent] | None = None,
        query_embeddings: list[RetrievalQueryEmbeddingRef] | None = None,
    ) -> RetrievalBatchTrace: ...

    async def get_batch(
        self, retrieval_batch_id: str
    ) -> (
        tuple[
            RetrievalBatchTrace,
            list[RetrievalTrace],
            list[RetrievalTraceComponent],
            list[RetrievalQueryEmbeddingRef],
        ]
        | None
    ): ...


class AnswerRepository(Protocol):
    async def add(self, answer: FinalAnswerRecord) -> FinalAnswerRecord: ...

    async def get(self, answer_id: str) -> FinalAnswerRecord | None: ...

    async def get_for_query(
        self,
        run_id: str,
        query_id: str,
        variant: str = "primary",
    ) -> FinalAnswerRecord | None: ...


class EvaluationRepository(Protocol):
    async def add_retrieval(self, value: RetrievalEvaluation) -> RetrievalEvaluation: ...

    async def add_metric(self, value: MetricScore) -> MetricScore: ...

    async def add_judge(self, value: JudgeScore) -> JudgeScore: ...

    async def add_bad_case(self, value: BadCaseLabel) -> BadCaseLabel: ...


class CompilerRunRepository(Protocol):
    async def add(self, value: CompilerRunSpec) -> CompilerRunSpec: ...

    async def get(self, run_id: str) -> CompilerRunSpec | None: ...


class CompilerUnitRepository(Protocol):
    async def add(self, value: CompilationUnit) -> CompilationUnit: ...

    async def get(self, compiler_unit_id: str) -> CompilationUnit | None: ...

    async def list_for_run(self, run_id: str) -> list[CompilationUnit]: ...

    async def add_stage_result(self, value: CompilationStageResult) -> CompilationStageResult: ...

    async def get_stage_result(
        self,
        compiler_unit_id: str,
        stage: CompilationStage,
    ) -> CompilationStageResult | None: ...


class QBankCompilerRepository(Protocol):
    async def add_question(
        self,
        value: QuestionCandidate,
        *,
        expected_evidence: list[OperatorEvidence] | None = None,
    ) -> QuestionCandidate: ...

    async def get_question(self, qbank_id: str) -> QuestionCandidate | None: ...

    async def add_feedback(self, value: DoubterFeedback) -> DoubterFeedback: ...

    async def add_decision(self, value: QBankDecision) -> QBankDecision: ...

    async def get_decision(self, qbank_id: str) -> QBankDecision | None: ...

    async def list_decisions(self, run_id: str) -> list[QBankDecision]: ...


class QuestionerRepository(Protocol):
    async def add_links(self, values: list[CompilationUnitLink]) -> None: ...

    async def list_links(self, compiler_unit_id: str) -> list[CompilationUnitLink]: ...

    async def add_opportunity_set(
        self,
        value: QuestionOpportunitySet,
        opportunities: list[QuestionOpportunity],
        evidence_refs: list[QuestionOpportunityEvidenceRef],
    ) -> QuestionOpportunitySet: ...

    async def get_opportunity_set(
        self, compiler_unit_id: str
    ) -> (
        tuple[
            QuestionOpportunitySet,
            list[QuestionOpportunity],
            list[QuestionOpportunityEvidenceRef],
        ]
        | None
    ): ...

    async def add_generation(self, value: QuestionerGeneration) -> QuestionerGeneration: ...

    async def get_generation(self, generation_id: str) -> QuestionerGeneration | None: ...

    async def get_generation_for_unit(
        self, compiler_unit_id: str, *, successful_only: bool = True
    ) -> QuestionerGeneration | None: ...

    async def persist_gated_candidates(self, values: list[GatedQuestionCandidate]) -> None: ...

    async def add_candidate(
        self,
        value: QuestionCandidate,
        *,
        expected_evidence: list[OperatorEvidence],
    ) -> QuestionCandidate: ...

    async def add_gate(self, value: QuestionCandidateGateResult) -> None: ...

    async def add_verification_batch(
        self,
        batch: QuestionerVerificationBatch,
        values: list[QuestionerCandidateVerification],
    ) -> QuestionerVerificationBatch: ...

    async def get_verification_batch(
        self,
        compiler_unit_id: str,
        generation_id: str,
        verification_round: int,
        *,
        scope_qbank_id: str | None = None,
    ) -> (
        tuple[
            QuestionerVerificationBatch,
            list[QuestionerCandidateVerification],
        ]
        | None
    ): ...

    async def add_supersession(
        self, value: QuestionerCandidateSupersession
    ) -> QuestionerCandidateSupersession: ...

    async def get_gate(self, qbank_id: str) -> QuestionCandidateGateResult | None: ...

    async def list_eligible_questions(self, run_id: str) -> list[QuestionCandidate]: ...

    async def list_duplicate_primaries(self, run_id: str) -> dict[str, str]: ...

    async def list_expected_evidence(self, run_id: str) -> dict[str, list[OperatorEvidence]]: ...


class QBankIndexRepository(Protocol):
    async def add(self, value: QBankIndexManifest) -> QBankIndexManifest: ...

    async def get(self, qbank_index_id: str) -> QBankIndexManifest | None: ...

    async def get_for_run(self, run_id: str) -> QBankIndexManifest | None: ...

    async def mark_ready(self, qbank_index_id: str) -> QBankIndexManifest: ...

    async def lexical_search(
        self,
        *,
        compiler_run_id: str,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
        query: str,
        question_budget: int,
        top_k: int,
    ) -> list[tuple[str, float]]: ...


class QBankCoverageRepository(Protocol):
    async def add_run(self, value: QBankCoverageRunManifest) -> QBankCoverageRunManifest: ...

    async def get_run(self, coverage_run_id: str) -> QBankCoverageRunManifest | None: ...

    async def transition(
        self, coverage_run_id: str, status: QBankCoverageStatus
    ) -> QBankCoverageRunManifest: ...

    async def list_query_embeddings(
        self, coverage_run_id: str
    ) -> list[QBankCoverageQueryEmbedding]: ...

    async def add_query_bundle(
        self,
        *,
        query: QBankCoverageQuery,
        embeddings: list[QBankCoverageQueryEmbedding],
        gold_groups: list[ResolvedGoldEvidenceGroup],
        candidates: list[QBankCoverageCandidate],
        metrics: list[QBankCoverageMetric],
        labels: list[QBankCoverageLabel],
        matched_members: dict[tuple[str, str], list[str]] | None = None,
    ) -> None: ...

    async def add_report(self, value: QBankCoverageGateReport) -> QBankCoverageGateReport: ...


class MemoryProgramRepository(Protocol):
    async def get_for_candidate_round(
        self, run_id: str, qbank_id: str, round_id: int
    ) -> MemoryProgram | None: ...

    async def add(
        self,
        value: MemoryProgram,
        *,
        expected_evidence: list[OperatorEvidence] | None = None,
    ) -> MemoryProgram: ...

    async def get(self, plan_id: str) -> MemoryProgram | None: ...

    async def list_for_run(self, run_id: str) -> list[MemoryProgram]: ...


class ProgramExecutionRepository(Protocol):
    async def add(self, value: ProgramExecutionResult) -> ProgramExecutionResult: ...

    async def get(self, execution_id: str) -> ProgramExecutionResult | None: ...


class CompilerStageJobRepository(Protocol):
    async def add(self, value: CompilerStageJob) -> CompilerStageJob: ...

    async def get(self, stage_job_id: str) -> CompilerStageJob | None: ...

    async def list_for_run(
        self, run_id: str, stage: CompilerStageKind | None = None
    ) -> list[CompilerStageJob]: ...

    async def claim(
        self,
        *,
        run_id: str,
        stage: CompilerStageKind,
        worker_id: str,
        lease_seconds: int,
    ) -> CompilerStageJob | None: ...

    async def heartbeat(
        self, stage_job_id: str, *, worker_id: str, lease_seconds: int
    ) -> CompilerStageJob: ...

    async def finish(
        self,
        stage_job_id: str,
        *,
        worker_id: str,
        succeeded: bool,
        error_type: str | None = None,
        error_summary: str | None = None,
    ) -> CompilerStageJob: ...


class PlannerRepository(Protocol):
    async def add_batch(self, value: PlannerGenerationBatch) -> PlannerGenerationBatch: ...

    async def add_requirement_contract(
        self, value: PlannerRequirementContract
    ) -> PlannerRequirementContract: ...

    async def get_requirement_contract(
        self, qbank_id: str
    ) -> PlannerRequirementContract | None: ...

    async def list_requirement_contracts(self, run_id: str) -> list[PlannerRequirementContract]: ...

    async def add_candidate_draft(
        self, value: PlannerCandidateDraftFact
    ) -> PlannerCandidateDraftFact: ...

    async def get_batch(
        self, compiler_unit_id: str, batch_ordinal: int
    ) -> PlannerGenerationBatch | None: ...

    async def add_program_link(self, value: PlannerBatchProgram) -> None: ...

    async def add_outcome(self, value: PlannerCandidateOutcome) -> PlannerCandidateOutcome: ...

    async def get_outcome(self, qbank_id: str) -> PlannerCandidateOutcome | None: ...

    async def list_outcomes(self, run_id: str) -> list[PlannerCandidateOutcome]: ...

    async def add_validation(
        self, value: ProgramValidationResultFact
    ) -> ProgramValidationResultFact: ...

    async def get_validation(
        self, plan_id: str, kind: ProgramValidationKind
    ) -> ProgramValidationResultFact | None: ...


class MemoryOperatorIndexRepository(Protocol):
    async def add(self, value: MemoryOperatorIndexManifest) -> MemoryOperatorIndexManifest: ...

    async def get_for_run(self, run_id: str) -> MemoryOperatorIndexManifest | None: ...

    async def mark_ready(self, memory_index_id: str) -> MemoryOperatorIndexManifest: ...


class ProgramQueryEmbeddingRepository(Protocol):
    async def add_batch(
        self,
        batch: ProgramQueryEmbeddingBatch,
        values: list[ProgramQueryEmbedding],
    ) -> ProgramQueryEmbeddingBatch: ...

    async def list_for_run(self, run_id: str) -> list[ProgramQueryEmbedding]: ...


class ProgramDryRunRepository(Protocol):
    async def add(
        self,
        value: ProgramDryRunAssessment,
        matches: list[ProgramEvidenceMatch],
    ) -> ProgramDryRunAssessment: ...

    async def get(
        self, plan_id: str
    ) -> tuple[ProgramDryRunAssessment, tuple[ProgramEvidenceMatch, ...]] | None: ...

    async def list_for_run(self, run_id: str) -> list[ProgramDryRunAssessment]: ...


class DoubterRepository(Protocol):
    async def add_r0_manifest(self, value: R0InputManifest) -> R0InputManifest: ...

    async def get_r0_manifest(self, run_id: str) -> R0InputManifest | None: ...

    async def add_candidate_job(self, value: CandidateStageJob) -> CandidateStageJob: ...

    async def get_candidate_job(self, job_id: str) -> CandidateStageJob | None: ...

    async def list_candidate_jobs(
        self,
        run_id: str,
        *,
        stage: CandidateStageKind | None = None,
        round_id: int | None = None,
    ) -> list[CandidateStageJob]: ...

    async def claim_candidate_job(
        self,
        *,
        run_id: str,
        stage: CandidateStageKind,
        round_id: int,
        worker_id: str,
        lease_seconds: int,
        qbank_ids: set[str] | None = None,
    ) -> CandidateStageJob | None: ...

    async def finish_candidate_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        succeeded: bool,
        error_type: str | None = None,
        error_summary: str | None = None,
    ) -> CandidateStageJob: ...

    async def cancel_candidate_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        terminal_reason: str,
    ) -> CandidateStageJob: ...

    async def add_assessment(self, value: DoubterAssessment) -> DoubterAssessment: ...

    async def get_assessment(
        self, run_id: str, qbank_id: str, round_id: int
    ) -> DoubterAssessment | None: ...

    async def list_assessments(self, run_id: str) -> list[DoubterAssessment]: ...

    async def add_revision(self, value: PlannerRevisionLineage) -> PlannerRevisionLineage: ...

    async def list_revisions(self, run_id: str) -> list[PlannerRevisionLineage]: ...

    async def add_revision_patch(self, value: PlannerRevisionPatch) -> PlannerRevisionPatch: ...

    async def list_revision_patches(self, run_id: str) -> list[PlannerRevisionPatch]: ...

    async def add_round_projection(
        self,
        manifest: QBankRoundProjectionManifest,
        members: list[QBankRoundProjectionMember],
    ) -> QBankRoundProjectionManifest: ...

    async def get_round_projection(
        self, run_id: str, round_id: int
    ) -> tuple[QBankRoundProjectionManifest, list[QBankRoundProjectionMember]] | None: ...

    async def add_routing_projection(
        self,
        manifest: RoutingProjectionManifest,
        members: list[RoutingProjectionMember],
    ) -> RoutingProjectionManifest: ...

    async def get_routing_projection(
        self, run_id: str
    ) -> tuple[RoutingProjectionManifest, list[RoutingProjectionMember]] | None: ...

    async def add_stage_component(
        self, value: StageComponentManifest
    ) -> StageComponentManifest: ...

    async def list_stage_components(
        self, launch_id: str, *, ready_only: bool = False
    ) -> list[StageComponentManifest]: ...


class StructuredOutputDiagnosticRepository(Protocol):
    async def add(
        self, value: StructuredOutputItemDiagnostic
    ) -> StructuredOutputItemDiagnostic: ...

    async def list_for_call(self, model_call_id: str) -> list[StructuredOutputItemDiagnostic]: ...


class QBankSemanticMatchRepository(Protocol):
    async def add_run(self, value: QBankSemanticMatchRun) -> QBankSemanticMatchRun: ...

    async def get_run(self, semantic_run_id: str) -> QBankSemanticMatchRun | None: ...

    async def add_job(self, value: QBankSemanticMatchJob) -> QBankSemanticMatchJob: ...

    async def get_job(
        self, semantic_run_id: str, query_id: str, qbank_id: str
    ) -> QBankSemanticMatchJob | None: ...

    async def get_job_by_id(self, semantic_job_id: str) -> QBankSemanticMatchJob | None: ...

    async def list_jobs(self, semantic_run_id: str) -> list[QBankSemanticMatchJob]: ...

    async def transition_run(
        self,
        semantic_run_id: str,
        status: QBankSemanticMatchStatus,
    ) -> QBankSemanticMatchRun: ...

    async def transition_job(
        self,
        semantic_job_id: str,
        status: JobStatus,
        *,
        model_call_id: str | None = None,
    ) -> QBankSemanticMatchJob: ...

    async def add_assessment(
        self, value: QBankSemanticMatchAssessment
    ) -> QBankSemanticMatchAssessment: ...

    async def get_assessment(self, semantic_job_id: str) -> QBankSemanticMatchAssessment | None: ...

    async def add_metrics(self, values: list[QBankSemanticMetric]) -> None: ...

    async def list_assessments(
        self, semantic_run_id: str
    ) -> list[QBankSemanticMatchAssessment]: ...

    async def list_metrics(self, semantic_run_id: str) -> list[QBankSemanticMetric]: ...


class QBankRoutingRepository(Protocol):
    async def get_runtime_projection_v3(self, run_id: str) -> dict[str, Any] | None: ...

    async def list_runtime_program_vector_refs_v3(self, run_id: str) -> list[dict[str, Any]]: ...

    async def list_verified_v4(self, **scope: Any) -> list[Any]: ...

    async def search_lexical_verified_v4(self, **scope: Any) -> list[tuple[str, float]]: ...

    async def search_embedding_verified_v4(self, **scope: Any) -> list[tuple[str, float]]: ...

    async def get_runtime_program_v3(self, plan_id: str) -> tuple[MemoryProgram, str] | None: ...

    async def list_for_conversation(
        self,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> list[QBankRoutingItem]: ...

    async def get_program_steps(self, plan_id: str) -> list[ProgramStep]: ...

    async def get_runtime_program(self, plan_id: str) -> tuple[MemoryProgram, str] | None: ...

    async def list_verified_v2(
        self,
        *,
        routing_projection_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        dataset: str,
        conversation_id: str,
        cutoff_session_id: str,
        question_budget: int,
        round_id: int,
    ) -> list[VerifiedQBankRoutingItemV2]: ...

    async def list_verified_v3(
        self,
        *,
        routing_projection_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        dataset: str,
        conversation_id: str,
        cutoff_session_id: str,
        question_budget: int,
        round_id: int,
    ) -> list[VerifiedQBankRoutingItemV2]: ...

    async def search_lexical_verified_v2(
        self,
        *,
        query_text: str,
        routing_projection_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        dataset: str,
        conversation_id: str,
        cutoff_session_id: str,
        route_class: str,
        question_budget: int,
        round_id: int,
        top_k: int = 50,
    ) -> list[tuple[str, float]]: ...

    async def search_lexical_verified_v3(
        self,
        *,
        query_text: str,
        routing_projection_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        dataset: str,
        conversation_id: str,
        cutoff_session_id: str,
        route_class: str,
        question_budget: int,
        round_id: int,
        top_k: int = 50,
        term_limit: int = 12,
    ) -> list[tuple[str, float]]: ...

    async def search_embedding_verified_v2(
        self,
        *,
        collection_id: str,
        query_vector: Sequence[float],
        routing_projection_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        dataset: str,
        conversation_id: str,
        cutoff_session_id: str,
        route_class: str,
        question_budget: int,
        round_id: int,
        top_k: int = 50,
    ) -> list[tuple[str, float]]: ...

    async def search_embedding_verified_v3(
        self,
        *,
        collection_id: str,
        query_vector: Sequence[float],
        routing_projection_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        dataset: str,
        conversation_id: str,
        cutoff_session_id: str,
        route_class: str,
        question_budget: int,
        round_id: int,
        top_k: int = 50,
    ) -> list[tuple[str, float]]: ...
