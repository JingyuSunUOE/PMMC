"""Deterministic Postgres exports for reproducible experiment analysis."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import Select, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.infrastructure.database.baseline_models import (
    ContextAssemblyImageModel,
    ContextAssemblyMemoryUnitModel,
    ContextAssemblyModel,
    ContextAssemblyRetrievalCandidateModel,
    RetrievalCorpusModel,
)
from pmmc.infrastructure.database.compiler_models import (
    CompilerRunModel,
    CompilerStageJobModel,
    CompilerUnitMemoryModel,
    CompilerUnitModel,
    CompilerUnitStageResultModel,
    DoubterFeedbackModel,
    MemoryOperatorIndexManifestModel,
    MemoryProgramModel,
    PlannerBatchProgramModel,
    PlannerCandidateDraftModel,
    PlannerCandidateOutcomeModel,
    PlannerGenerationBatchModel,
    PlannerRequirementContractModel,
    ProgramDryRunAssessmentModel,
    ProgramDryRunEvidenceMatchModel,
    ProgramExecutionEvidenceRefModel,
    ProgramExecutionModel,
    ProgramExecutionStepModel,
    ProgramExpectedEvidenceRefModel,
    ProgramQueryEmbeddingBatchModel,
    ProgramQueryEmbeddingModel,
    ProgramStepModel,
    ProgramValidationIssueModel,
    ProgramValidationResultModel,
    QBankDecisionModel,
    QBankEvidenceRefModel,
    QBankParaphraseModel,
    QBankQuestionModel,
)
from pmmc.infrastructure.database.doubter_models import (
    CandidateStageJobModel,
    DoubterAssessmentDimensionModel,
    DoubterAssessmentModel,
    DoubterDirectiveModel,
    PlannerRevisionLineageModel,
    PlannerRevisionPatchModel,
    QBankRoundProjectionManifestModel,
    QBankRoundProjectionMemberModel,
    R0InputManifestModel,
    RoutingProjectionManifestModel,
    RoutingProjectionMemberModel,
    StageComponentManifestModel,
)
from pmmc.infrastructure.database.experiment_models import (
    BadCaseLabelModel,
    EmbeddingCollectionModel,
    EmbeddingModel,
    EvaluationArtifactModel,
    EvaluationModelCallModel,
    ExperimentJobModel,
    FinalAnswerEvidenceIdModel,
    FinalAnswerImageIdModel,
    FinalAnswerModel,
    JobAttemptModel,
    JudgeScoreModel,
    MetricScoreModel,
    RetrievalBatchModel,
    RetrievalBatchQueryEmbeddingModel,
    RetrievalEvaluationMatchModel,
    RetrievalEvaluationModel,
    RetrievalTraceComponentModel,
    RetrievalTraceModel,
    RunQueryModel,
    RunStageSummaryModel,
    RunStateModel,
    RuntimeArtifactModel,
    RuntimeModelCallModel,
)
from pmmc.infrastructure.database.models import RunManifestModel
from pmmc.infrastructure.database.query_executor_models import (
    AnswerFormatDiagnosticModel,
    AnswerReuseLinkModel,
    PmmcFallbackQueryLinkModel,
    PmmcFallbackRunLinkModel,
    QBankLexicalTermFactModel,
    QBankMatchBatchModel,
    QBankMatchChannelCandidateModel,
    QBankMatchFusedCandidateModel,
    QBankRerankerAssessmentModel,
    QueryAnswerSelectionModel,
    QueryEmbeddingBatchModel,
    QueryEmbeddingModel,
    QueryEvidenceRelevanceAssessmentModel,
    QueryEvidenceRelevanceItemModel,
    QueryEvidenceSufficiencyAssessmentModel,
    QueryExecutionReuseLinkModel,
    QueryExecutorRunModel,
    QueryExecutorStageJobModel,
    QueryFallbackDecisionModel,
    QueryProgramBindingModel,
    QueryRouteDecisionModel,
    StructuredAnswerOutputModel,
)
from pmmc.infrastructure.database.questioner_models import (
    CompilerUnitLinkModel,
    EvaluationStructuredOutputItemDiagnosticModel,
    QBankCandidateFamilyMemberModel,
    QBankCandidateFamilyModel,
    QBankDuplicateLinkModel,
    QBankIndexManifestModel,
    QBankSemanticMatchAssessmentModel,
    QBankSemanticMatchJobModel,
    QBankSemanticMatchRunModel,
    QBankSemanticMetricModel,
    QuestionerAnswerClaimModel,
    QuestionerCandidateGateResultModel,
    QuestionerCandidateSupersessionModel,
    QuestionerCandidateVerificationModel,
    QuestionerGenerationModel,
    QuestionerVerificationBatchModel,
    QuestionOpportunityEvidenceRefModel,
    QuestionOpportunityModel,
    QuestionOpportunitySetModel,
    RuntimeStructuredOutputItemDiagnosticModel,
)
from pmmc.infrastructure.export_verification import (
    MODEL_ARTIFACT_BUNDLE_VERSION,
    PROGRAM_RUNTIME_EXPORT_VERSION,
    QUESTIONER_RUNTIME_EXPORT_VERSION,
    QUESTIONER_V3_RUNTIME_EXPORT_VERSION,
    STEP7B_PROGRAM_RUNTIME_EXPORT_VERSION,
    verify_experiment_export,
)
from pmmc.infrastructure.object_store import ObjectStore, sha256_file, validate_object_key
from pmmc.schemas import ArtifactState, AssetType, RunStatus

_ARTIFACT_EXPORT_PROGRESS_INTERVAL_SECONDS = 30.0


def _artifact_export_event(phase: str, **values: object) -> None:
    payload = {"phase": phase, "stage": "runtime_model_artifact_export", **values}
    print(
        "MODEL_ARTIFACT_EXPORT "
        + orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode("utf-8"),
        flush=True,
    )


@dataclass(frozen=True)
class ExperimentExportResult:
    run_id: str
    export_id: str
    export_dir: Path
    manifest_path: Path


def _json_bytes(value: Any) -> bytes:
    return orjson.dumps(
        value,
        option=orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z | orjson.OPT_NON_STR_KEYS,
    )


def _row_dict(row: Any) -> dict[str, Any]:
    return {key: value for key, value in dict(row).items()}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            stream.write(_json_bytes(row))
            stream.write(b"\n")


def _validate_run_path_component(run_id: str) -> str:
    path = PurePosixPath(run_id)
    if len(path.parts) != 1 or path.name != run_id or run_id in {".", ".."} or "\\" in run_id:
        raise ValueError("run_id must be a safe filesystem path component")
    return run_id


# Mapping exports accept arbitrary row arity: Select's generic shape differs in 2.0/2.1.
async def _fetch(session: AsyncSession, statement: Select) -> list[dict[str, Any]]:
    result = await session.execute(statement)
    return [_row_dict(row) for row in result.mappings().all()]


async def _write_select_jsonl(
    session: AsyncSession,
    path: Path,
    statement: Select,
    batch_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = await session.stream(statement.execution_options(yield_per=batch_size))
    with path.open("wb") as stream:
        async for row in result.mappings():
            stream.write(_json_bytes(_row_dict(row)))
            stream.write(b"\n")


def _reused_query_embedding_calls(run_ids: Select | list[str]) -> Select:
    """Export inherited call lineage separately from this attempt's incurred cost."""
    return (
        select(EmbeddingModel.model_call_id)
        .join(RetrievalBatchQueryEmbeddingModel,
              RetrievalBatchQueryEmbeddingModel.embedding_id == EmbeddingModel.embedding_id)
        .join(RetrievalBatchModel,
              RetrievalBatchModel.retrieval_batch_id
              == RetrievalBatchQueryEmbeddingModel.retrieval_batch_id)
        .join(RuntimeModelCallModel,
              RuntimeModelCallModel.model_call_id == EmbeddingModel.model_call_id)
        .where(RetrievalBatchModel.run_id.in_(run_ids),
               RuntimeModelCallModel.run_id != RetrievalBatchModel.run_id,
               RuntimeModelCallModel.stage == "embed")
    )


@dataclass(frozen=True)
class _ModelArtifactReference:
    model_call_id: str
    role: str
    artifact_id: str
    source_run_id: str
    snapshot_id: str | None


class ModelArtifactExportError(ValueError):
    """Safe failure identity; source IDs and object locations stay private."""

    def __init__(
        self, error_code: str, *, substage: str,
        reference: _ModelArtifactReference | None = None,
    ) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.substage = substage
        self.reference = reference


class ExperimentExporter:
    """Export one completed run without exposing evaluation data by default."""

    def __init__(
        self,
        runtime_sessions: async_sessionmaker[AsyncSession],
        evaluator_sessions: async_sessionmaker[AsyncSession] | None = None,
        runtime_store: ObjectStore | None = None,
        batch_size: int = 5000,
        embedding_parquet_compression: str = "NONE",
        artifact_concurrency: int = 8,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("export batch_size must be positive")
        self.runtime_sessions = runtime_sessions
        self.evaluator_sessions = evaluator_sessions
        self.runtime_store = runtime_store
        self.batch_size = batch_size
        if embedding_parquet_compression not in {"NONE", "zstd"}:
            raise ValueError("unsupported embedding Parquet compression")
        self.embedding_parquet_compression = embedding_parquet_compression
        if artifact_concurrency < 1:
            raise ValueError("artifact_concurrency must be positive")
        self.artifact_concurrency = artifact_concurrency

    async def export(
        self,
        run_id: str,
        output_root: Path,
        *,
        include_evaluation: bool = False,
        require_completed: bool = True,
        include_referenced_model_artifacts: bool = False,
    ) -> ExperimentExportResult:
        _validate_run_path_component(run_id)
        if include_evaluation and self.evaluator_sessions is None:
            raise ValueError("evaluation export requires evaluator database access")
        if include_referenced_model_artifacts and self.runtime_store is None:
            raise ValueError("referenced model artifact export requires a runtime object store")
        temporary = output_root / f".tmp-{run_id}-{uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            async with self.runtime_sessions() as session:
                manifest = await session.get(RunManifestModel, run_id)
                state = await session.get(RunStateModel, run_id)
                if manifest is None or state is None:
                    raise KeyError(f"run does not exist: {run_id}")
                if require_completed and state.status != RunStatus.COMPLETED.value:
                    raise ValueError("only completed runs can produce a frozen export")
                _write_json(temporary / "runtime" / "run_manifest.json", manifest.manifest_json)
                _write_json(
                    temporary / "runtime" / "run_state.json",
                    self._model_row(state),
                )
                await self._export_runtime(session, run_id, temporary / "runtime")
                revision = await session.scalar(text("SELECT version_num FROM alembic_version"))
            # Object-store reads must not hold the relational export transaction.
            artifact_bundle = None
            if include_referenced_model_artifacts:
                artifact_bundle = await self._archive_referenced_model_artifacts(
                    run_id, temporary / "runtime" / "private",
                )
            if include_evaluation:
                assert self.evaluator_sessions is not None
                async with self.evaluator_sessions() as session:
                    await self._export_evaluation(session, run_id, temporary / "evaluation")

            files = self._file_inventory(temporary)
            if manifest.config_hash is None:
                raise ValueError("completed experiment run is missing config_hash")
            content_payload = {
                "run_id": run_id,
                "canonical_schema_version": manifest.schema_version,
                "experiment_schema_version": manifest.experiment_schema_version,
                "compiler_schema_version": manifest.manifest_json.get(
                    "compiler_schema_version"
                ),
                "alembic_revision": revision,
                "run_config_hash": manifest.config_hash,
                **(
                    {
                        "questioner_runtime_export_version": (
                            QUESTIONER_V3_RUNTIME_EXPORT_VERSION
                            if self._questioner_protocol(manifest.manifest_json)
                            in {
                                "pmmc-questioner-v3",
                                "pmmc-questioner-v3.1",
                                "pmmc-questioner-v3.2",
                                "pmmc-questioner-v3.3",
                                "pmmc-questioner-v3.4",
                                "pmmc-questioner-v3.5",
                            }
                            else QUESTIONER_RUNTIME_EXPORT_VERSION
                        ),
                        "model_artifact_bundle_version": (
                            MODEL_ARTIFACT_BUNDLE_VERSION
                            if include_referenced_model_artifacts
                            else None
                        ),
                        "model_artifact_reference_count": (
                            artifact_bundle["reference_count"]
                            if artifact_bundle is not None
                            else 0
                        ),
                        "model_artifact_payload_count": (
                            artifact_bundle["payload_count"]
                            if artifact_bundle is not None
                            else 0
                        ),
                    }
                    if self._is_questioner_manifest(manifest.manifest_json)
                    else {}
                ),
                **(
                    {
                        "program_runtime_export_version": (
                            STEP7B_PROGRAM_RUNTIME_EXPORT_VERSION
                            if manifest.manifest_json.get("compiler_schema_version")
                            == "0.8.0"
                            else "compiler-program-runtime-export-v4"
                            if manifest.manifest_json.get("compiler_schema_version")
                            == "0.7.0"
                            else PROGRAM_RUNTIME_EXPORT_VERSION
                            if manifest.manifest_json.get("compiler_schema_version")
                            == "0.6.0"
                            else (
                                "compiler-program-runtime-export-v2"
                                if manifest.manifest_json.get(
                                    "compiler_schema_version"
                                )
                                == "0.5.0"
                                else "compiler-program-runtime-export-v1"
                            )
                        )
                    }
                    if self._is_program_manifest(manifest.manifest_json)
                    else {}
                ),
                "file_count": len(files),
                "files": files,
            }
            export_id = hashlib.sha256(_json_bytes(content_payload)).hexdigest()
            export_manifest = {"export_id": export_id, **content_payload}
            _write_json(temporary / "export_manifest.json", export_manifest)
            destination = output_root / run_id / export_id
            if destination.exists():
                verify_experiment_export(destination)
                if (destination / "export_manifest.json").read_bytes() != (
                    temporary / "export_manifest.json"
                ).read_bytes():
                    raise ValueError("existing export ID has different content")
                shutil.rmtree(temporary)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(destination)
            return ExperimentExportResult(
                run_id=run_id,
                export_id=export_id,
                export_dir=destination,
                manifest_path=destination / "export_manifest.json",
            )
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    async def _export_runtime(self, session: AsyncSession, run_id: str, root: Path) -> None:
        specs: list[tuple[str, Select]] = [
            (
                "run_queries.jsonl",
                select(RunQueryModel.__table__)
                .where(RunQueryModel.run_id == run_id)
                .order_by(RunQueryModel.ordinal),
            ),
            (
                "run_stage_summaries.jsonl",
                select(RunStageSummaryModel.__table__)
                .where(RunStageSummaryModel.run_id == run_id)
                .order_by(RunStageSummaryModel.stage),
            ),
            (
                "experiment_jobs.jsonl",
                select(ExperimentJobModel.__table__)
                .where(ExperimentJobModel.run_id == run_id)
                .order_by(ExperimentJobModel.job_id),
            ),
            (
                "run_artifacts.jsonl",
                select(RuntimeArtifactModel.__table__)
                .where(RuntimeArtifactModel.run_id == run_id)
                .order_by(RuntimeArtifactModel.artifact_id),
            ),
            (
                "model_calls.jsonl",
                select(RuntimeModelCallModel.__table__)
                .where(RuntimeModelCallModel.run_id == run_id)
                .order_by(RuntimeModelCallModel.created_at, RuntimeModelCallModel.model_call_id),
            ),
            (
                "retrieval_batches.jsonl",
                select(RetrievalBatchModel.__table__)
                .where(RetrievalBatchModel.run_id == run_id)
                .order_by(RetrievalBatchModel.query_id, RetrievalBatchModel.retrieval_batch_id),
            ),
            (
                "retrieval_traces.jsonl",
                select(RetrievalTraceModel.__table__)
                .where(RetrievalTraceModel.run_id == run_id)
                .order_by(
                    RetrievalTraceModel.query_id,
                    RetrievalTraceModel.retrieval_batch_id,
                    RetrievalTraceModel.rank,
                ),
            ),
            (
                "final_answers.jsonl",
                select(FinalAnswerModel.__table__)
                .where(FinalAnswerModel.run_id == run_id)
                .order_by(FinalAnswerModel.query_id, FinalAnswerModel.answer_variant),
            ),
            (
                "context_assemblies.jsonl",
                select(ContextAssemblyModel.__table__)
                .where(ContextAssemblyModel.run_id == run_id)
                .order_by(ContextAssemblyModel.query_id, ContextAssemblyModel.context_assembly_id),
            ),
            (
                "compiler_runs.jsonl",
                select(CompilerRunModel.__table__)
                .where(CompilerRunModel.run_id == run_id)
                .order_by(CompilerRunModel.run_id),
            ),
            (
                "compiler_units.jsonl",
                select(CompilerUnitModel.__table__)
                .where(CompilerUnitModel.run_id == run_id)
                .order_by(CompilerUnitModel.ordinal, CompilerUnitModel.compiler_unit_id),
            ),
            (
                "compiler_unit_memories.jsonl",
                select(CompilerUnitMemoryModel.__table__)
                .where(CompilerUnitMemoryModel.run_id == run_id)
                .order_by(
                    CompilerUnitMemoryModel.compiler_unit_id,
                    CompilerUnitMemoryModel.ordinal,
                ),
            ),
            (
                "compiler_unit_stage_results.jsonl",
                select(CompilerUnitStageResultModel.__table__)
                .where(CompilerUnitStageResultModel.run_id == run_id)
                .order_by(
                    CompilerUnitStageResultModel.compiler_unit_id,
                    CompilerUnitStageResultModel.stage,
                    CompilerUnitStageResultModel.attempt_number,
                ),
            ),
            (
                "compiler_unit_links.jsonl",
                select(CompilerUnitLinkModel.__table__)
                .where(CompilerUnitLinkModel.run_id == run_id)
                .order_by(
                    CompilerUnitLinkModel.compiler_unit_id,
                    CompilerUnitLinkModel.ordinal,
                    CompilerUnitLinkModel.link_id,
                ),
            ),
            (
                "questioner_generations.jsonl",
                select(QuestionerGenerationModel.__table__)
                .where(QuestionerGenerationModel.run_id == run_id)
                .order_by(
                    QuestionerGenerationModel.compiler_unit_id,
                    QuestionerGenerationModel.attempt_number,
                    QuestionerGenerationModel.generation_id,
                ),
            ),
            (
                "questioner_opportunity_sets.jsonl",
                select(QuestionOpportunitySetModel.__table__)
                .where(QuestionOpportunitySetModel.run_id == run_id)
                .order_by(QuestionOpportunitySetModel.compiler_unit_id),
            ),
            (
                "questioner_opportunities.jsonl",
                select(QuestionOpportunityModel.__table__)
                .where(QuestionOpportunityModel.run_id == run_id)
                .order_by(
                    QuestionOpportunityModel.compiler_unit_id,
                    QuestionOpportunityModel.ordinal,
                ),
            ),
            (
                "questioner_opportunity_evidence_refs.jsonl",
                select(QuestionOpportunityEvidenceRefModel.__table__)
                .where(QuestionOpportunityEvidenceRefModel.run_id == run_id)
                .order_by(
                    QuestionOpportunityEvidenceRefModel.compiler_unit_id,
                    QuestionOpportunityEvidenceRefModel.opportunity_id,
                    QuestionOpportunityEvidenceRefModel.ordinal,
                ),
            ),
            (
                "questioner_verification_batches.jsonl",
                select(QuestionerVerificationBatchModel.__table__)
                .where(QuestionerVerificationBatchModel.run_id == run_id)
                .order_by(
                    QuestionerVerificationBatchModel.compiler_unit_id,
                    QuestionerVerificationBatchModel.verification_round,
                ),
            ),
            (
                "questioner_candidate_verifications.jsonl",
                select(QuestionerCandidateVerificationModel.__table__)
                .where(QuestionerCandidateVerificationModel.run_id == run_id)
                .order_by(
                    QuestionerCandidateVerificationModel.compiler_unit_id,
                    QuestionerCandidateVerificationModel.verification_round,
                    QuestionerCandidateVerificationModel.candidate_alias,
                ),
            ),
            (
                "questioner_candidate_supersessions.jsonl",
                select(QuestionerCandidateSupersessionModel.__table__)
                .where(QuestionerCandidateSupersessionModel.run_id == run_id)
                .order_by(
                    QuestionerCandidateSupersessionModel.compiler_unit_id,
                    QuestionerCandidateSupersessionModel.source_qbank_id,
                ),
            ),
            (
                "questioner_answer_claims.jsonl",
                select(QuestionerAnswerClaimModel.__table__)
                .where(QuestionerAnswerClaimModel.run_id == run_id)
                .order_by(
                    QuestionerAnswerClaimModel.qbank_id,
                    QuestionerAnswerClaimModel.ordinal,
                ),
            ),
            (
                "qbank_candidate_families.jsonl",
                select(QBankCandidateFamilyModel.__table__)
                .where(QBankCandidateFamilyModel.run_id == run_id)
                .order_by(QBankCandidateFamilyModel.family_id),
            ),
            (
                "qbank_candidate_family_members.jsonl",
                select(QBankCandidateFamilyMemberModel.__table__)
                .where(QBankCandidateFamilyMemberModel.run_id == run_id)
                .order_by(
                    QBankCandidateFamilyMemberModel.family_id,
                    QBankCandidateFamilyMemberModel.ordinal,
                ),
            ),
            (
                "questioner_candidate_gate_results.jsonl",
                select(QuestionerCandidateGateResultModel.__table__)
                .where(QuestionerCandidateGateResultModel.run_id == run_id)
                .order_by(
                    QuestionerCandidateGateResultModel.compiler_unit_id,
                    QuestionerCandidateGateResultModel.qbank_id,
                ),
            ),
            (
                "qbank_duplicate_links.jsonl",
                select(QBankDuplicateLinkModel.__table__)
                .where(QBankDuplicateLinkModel.run_id == run_id)
                .order_by(
                    QBankDuplicateLinkModel.duplicate_qbank_id,
                    QBankDuplicateLinkModel.primary_qbank_id,
                ),
            ),
            (
                "qbank_index_manifests.jsonl",
                select(QBankIndexManifestModel.__table__)
                .where(QBankIndexManifestModel.compiler_run_id == run_id)
                .order_by(QBankIndexManifestModel.qbank_index_id),
            ),
            (
                "compiler_stage_jobs.jsonl",
                select(CompilerStageJobModel.__table__)
                .where(CompilerStageJobModel.run_id == run_id)
                .order_by(
                    CompilerStageJobModel.stage,
                    CompilerStageJobModel.compiler_unit_id,
                    CompilerStageJobModel.round_id,
                ),
            ),
            (
                "planner_generation_batches.jsonl",
                select(PlannerGenerationBatchModel.__table__)
                .where(PlannerGenerationBatchModel.run_id == run_id)
                .order_by(
                    PlannerGenerationBatchModel.compiler_unit_id,
                    PlannerGenerationBatchModel.batch_ordinal,
                ),
            ),
            (
                "planner_requirement_contracts.jsonl",
                select(PlannerRequirementContractModel.__table__)
                .where(PlannerRequirementContractModel.run_id == run_id)
                .order_by(PlannerRequirementContractModel.qbank_id),
            ),
            (
                "planner_candidate_drafts.jsonl",
                select(PlannerCandidateDraftModel.__table__)
                .where(PlannerCandidateDraftModel.run_id == run_id)
                .order_by(
                    PlannerCandidateDraftModel.compiler_unit_id,
                    PlannerCandidateDraftModel.planner_batch_id,
                    PlannerCandidateDraftModel.item_ordinal,
                ),
            ),
            (
                "planner_candidate_outcomes.jsonl",
                select(PlannerCandidateOutcomeModel.__table__)
                .where(PlannerCandidateOutcomeModel.run_id == run_id)
                .order_by(
                    PlannerCandidateOutcomeModel.compiler_unit_id,
                    PlannerCandidateOutcomeModel.qbank_id,
                ),
            ),
            (
                "planner_batch_programs.jsonl",
                select(PlannerBatchProgramModel.__table__)
                .where(PlannerBatchProgramModel.run_id == run_id)
                .order_by(
                    PlannerBatchProgramModel.compiler_unit_id,
                    PlannerBatchProgramModel.ordinal,
                ),
            ),
            (
                "program_validation_results.jsonl",
                select(ProgramValidationResultModel.__table__)
                .where(ProgramValidationResultModel.run_id == run_id)
                .order_by(
                    ProgramValidationResultModel.plan_id,
                    ProgramValidationResultModel.validation_kind,
                ),
            ),
            (
                "program_validation_issues.jsonl",
                select(ProgramValidationIssueModel.__table__)
                .where(ProgramValidationIssueModel.run_id == run_id)
                .order_by(
                    ProgramValidationIssueModel.validation_result_id,
                    ProgramValidationIssueModel.ordinal,
                ),
            ),
            (
                "memory_operator_index_manifests.jsonl",
                select(MemoryOperatorIndexManifestModel.__table__)
                .where(MemoryOperatorIndexManifestModel.run_id == run_id)
                .order_by(MemoryOperatorIndexManifestModel.memory_index_id),
            ),
            (
                "program_query_embedding_batches.jsonl",
                select(ProgramQueryEmbeddingBatchModel.__table__)
                .where(ProgramQueryEmbeddingBatchModel.run_id == run_id)
                .order_by(
                    ProgramQueryEmbeddingBatchModel.channel,
                    ProgramQueryEmbeddingBatchModel.batch_ordinal,
                ),
            ),
            (
                "program_query_embeddings.jsonl",
                select(ProgramQueryEmbeddingModel.__table__)
                .where(ProgramQueryEmbeddingModel.run_id == run_id)
                .order_by(
                    ProgramQueryEmbeddingModel.channel,
                    ProgramQueryEmbeddingModel.qbank_id,
                ),
            ),
            (
                "program_dry_run_assessments.jsonl",
                select(ProgramDryRunAssessmentModel.__table__)
                .where(ProgramDryRunAssessmentModel.run_id == run_id)
                .order_by(ProgramDryRunAssessmentModel.plan_id),
            ),
            (
                "program_dry_run_evidence_matches.jsonl",
                select(ProgramDryRunEvidenceMatchModel.__table__)
                .where(ProgramDryRunEvidenceMatchModel.run_id == run_id)
                .order_by(
                    ProgramDryRunEvidenceMatchModel.assessment_id,
                    ProgramDryRunEvidenceMatchModel.match_id,
                ),
            ),
            (
                "structured_output_item_diagnostics.jsonl",
                select(RuntimeStructuredOutputItemDiagnosticModel.__table__)
                .where(RuntimeStructuredOutputItemDiagnosticModel.run_id == run_id)
                .order_by(
                    RuntimeStructuredOutputItemDiagnosticModel.model_call_id,
                    RuntimeStructuredOutputItemDiagnosticModel.stage,
                    RuntimeStructuredOutputItemDiagnosticModel.item_ordinal,
                ),
            ),
            (
                "qbank_questions.jsonl",
                select(QBankQuestionModel.__table__)
                .where(QBankQuestionModel.run_id == run_id)
                .order_by(QBankQuestionModel.qbank_id),
            ),
            (
                "qbank_paraphrases.jsonl",
                select(QBankParaphraseModel.__table__)
                .where(QBankParaphraseModel.run_id == run_id)
                .order_by(QBankParaphraseModel.qbank_id, QBankParaphraseModel.ordinal),
            ),
            (
                "qbank_evidence_refs.jsonl",
                select(QBankEvidenceRefModel.__table__)
                .where(QBankEvidenceRefModel.run_id == run_id)
                .order_by(
                    QBankEvidenceRefModel.qbank_id,
                    QBankEvidenceRefModel.role,
                    QBankEvidenceRefModel.ordinal,
                ),
            ),
            (
                "memory_programs.jsonl",
                select(MemoryProgramModel.__table__)
                .where(MemoryProgramModel.run_id == run_id)
                .order_by(MemoryProgramModel.qbank_id, MemoryProgramModel.round_id),
            ),
            (
                "program_steps.jsonl",
                select(ProgramStepModel.__table__)
                .where(ProgramStepModel.run_id == run_id)
                .order_by(ProgramStepModel.plan_id, ProgramStepModel.step_id),
            ),
            (
                "program_expected_evidence_refs.jsonl",
                select(ProgramExpectedEvidenceRefModel.__table__)
                .where(ProgramExpectedEvidenceRefModel.run_id == run_id)
                .order_by(
                    ProgramExpectedEvidenceRefModel.plan_id,
                    ProgramExpectedEvidenceRefModel.step_id.nullsfirst(),
                    ProgramExpectedEvidenceRefModel.ordinal,
                ),
            ),
            (
                "program_executions.jsonl",
                select(ProgramExecutionModel.__table__)
                .where(ProgramExecutionModel.run_id == run_id)
                .order_by(ProgramExecutionModel.execution_id),
            ),
            (
                "program_execution_steps.jsonl",
                select(ProgramExecutionStepModel.__table__)
                .where(ProgramExecutionStepModel.run_id == run_id)
                .order_by(
                    ProgramExecutionStepModel.execution_id,
                    ProgramExecutionStepModel.step_id,
                ),
            ),
            (
                "program_execution_evidence_refs.jsonl",
                select(ProgramExecutionEvidenceRefModel.__table__)
                .where(ProgramExecutionEvidenceRefModel.run_id == run_id)
                .order_by(
                    ProgramExecutionEvidenceRefModel.execution_id,
                    ProgramExecutionEvidenceRefModel.rank,
                ),
            ),
            (
                "doubter_feedback.jsonl",
                select(DoubterFeedbackModel.__table__)
                .where(DoubterFeedbackModel.run_id == run_id)
                .order_by(DoubterFeedbackModel.qbank_id, DoubterFeedbackModel.round_id),
            ),
            (
                "r0_input_manifests.jsonl",
                select(R0InputManifestModel.__table__)
                .where(R0InputManifestModel.run_id == run_id)
                .order_by(R0InputManifestModel.r0_manifest_id),
            ),
            (
                "candidate_stage_jobs.jsonl",
                select(CandidateStageJobModel.__table__)
                .where(CandidateStageJobModel.run_id == run_id)
                .order_by(
                    CandidateStageJobModel.stage,
                    CandidateStageJobModel.round_id,
                    CandidateStageJobModel.qbank_id,
                ),
            ),
            (
                "doubter_assessments.jsonl",
                select(DoubterAssessmentModel.__table__)
                .where(DoubterAssessmentModel.run_id == run_id)
                .order_by(
                    DoubterAssessmentModel.qbank_id,
                    DoubterAssessmentModel.round_id,
                ),
            ),
            (
                "doubter_assessment_dimensions.jsonl",
                select(DoubterAssessmentDimensionModel.__table__)
                .where(DoubterAssessmentDimensionModel.run_id == run_id)
                .order_by(
                    DoubterAssessmentDimensionModel.assessment_id,
                    DoubterAssessmentDimensionModel.dimension,
                ),
            ),
            (
                "doubter_directives.jsonl",
                select(DoubterDirectiveModel.__table__)
                .where(DoubterDirectiveModel.run_id == run_id)
                .order_by(
                    DoubterDirectiveModel.feedback_id,
                    DoubterDirectiveModel.ordinal,
                ),
            ),
            (
                "planner_revision_lineage.jsonl",
                select(PlannerRevisionLineageModel.__table__)
                .where(PlannerRevisionLineageModel.run_id == run_id)
                .order_by(
                    PlannerRevisionLineageModel.qbank_id,
                    PlannerRevisionLineageModel.target_round_id,
                ),
            ),
            (
                "planner_revision_patches.jsonl",
                select(PlannerRevisionPatchModel.__table__)
                .where(PlannerRevisionPatchModel.run_id == run_id)
                .order_by(
                    PlannerRevisionPatchModel.qbank_id,
                    PlannerRevisionPatchModel.target_round_id,
                ),
            ),
            (
                "qbank_round_projection_manifests.jsonl",
                select(QBankRoundProjectionManifestModel.__table__)
                .where(QBankRoundProjectionManifestModel.run_id == run_id)
                .order_by(QBankRoundProjectionManifestModel.round_id),
            ),
            (
                "qbank_round_projection_members.jsonl",
                select(QBankRoundProjectionMemberModel.__table__)
                .where(QBankRoundProjectionMemberModel.run_id == run_id)
                .order_by(
                    QBankRoundProjectionMemberModel.projection_id,
                    QBankRoundProjectionMemberModel.qbank_id,
                ),
            ),
            (
                "routing_projection_manifests.jsonl",
                select(RoutingProjectionManifestModel.__table__)
                .where(RoutingProjectionManifestModel.run_id == run_id)
                .order_by(RoutingProjectionManifestModel.routing_projection_id),
            ),
            (
                "routing_projection_members.jsonl",
                select(RoutingProjectionMemberModel.__table__)
                .where(RoutingProjectionMemberModel.run_id == run_id)
                .order_by(RoutingProjectionMemberModel.ordinal),
            ),
            (
                "stage_component_manifests.jsonl",
                select(StageComponentManifestModel.__table__)
                .where(StageComponentManifestModel.run_id == run_id)
                .order_by(
                    StageComponentManifestModel.stage,
                    StageComponentManifestModel.round_id.nullsfirst(),
                    StageComponentManifestModel.component_id,
                ),
            ),
            (
                "qbank_decisions.jsonl",
                select(QBankDecisionModel.__table__)
                .where(QBankDecisionModel.run_id == run_id)
                .order_by(QBankDecisionModel.qbank_id),
            ),
            (
                "query_executor_runs.jsonl",
                select(QueryExecutorRunModel.__table__)
                .where(QueryExecutorRunModel.query_executor_run_id == run_id)
                .order_by(QueryExecutorRunModel.query_executor_run_id),
            ),
            (
                "query_executor_stage_jobs.jsonl",
                select(QueryExecutorStageJobModel.__table__)
                .where(QueryExecutorStageJobModel.query_executor_run_id == run_id)
                .order_by(
                    QueryExecutorStageJobModel.query_id,
                    QueryExecutorStageJobModel.stage,
                    QueryExecutorStageJobModel.round_id.nullsfirst(),
                ),
            ),
            (
                "query_embedding_batches.jsonl",
                select(QueryEmbeddingBatchModel.__table__)
                .where(QueryEmbeddingBatchModel.query_executor_run_id == run_id)
                .order_by(
                    QueryEmbeddingBatchModel.channel,
                    QueryEmbeddingBatchModel.batch_ordinal,
                ),
            ),
            (
                "query_embeddings.jsonl",
                select(QueryEmbeddingModel.__table__)
                .where(QueryEmbeddingModel.query_executor_run_id == run_id)
                .order_by(QueryEmbeddingModel.query_id, QueryEmbeddingModel.channel),
            ),
            (
                "qbank_match_batches.jsonl",
                select(QBankMatchBatchModel.__table__)
                .where(QBankMatchBatchModel.query_executor_run_id == run_id)
                .order_by(
                    QBankMatchBatchModel.query_id,
                    QBankMatchBatchModel.question_budget,
                    QBankMatchBatchModel.active_round,
                ),
            ),
            (
                "query_route_decisions.jsonl",
                select(QueryRouteDecisionModel.__table__)
                .where(QueryRouteDecisionModel.query_executor_run_id == run_id)
                .order_by(QueryRouteDecisionModel.query_id),
            ),
            (
                "query_program_bindings.jsonl",
                select(QueryProgramBindingModel.__table__)
                .where(QueryProgramBindingModel.query_executor_run_id == run_id)
                .order_by(
                    QueryProgramBindingModel.query_id,
                    QueryProgramBindingModel.round_id,
                ),
            ),
            (
                "query_evidence_sufficiency_assessments.jsonl",
                select(QueryEvidenceSufficiencyAssessmentModel.__table__)
                .where(
                    QueryEvidenceSufficiencyAssessmentModel.query_executor_run_id
                    == run_id
                )
                .order_by(
                    QueryEvidenceSufficiencyAssessmentModel.query_id,
                    QueryEvidenceSufficiencyAssessmentModel.query_program_binding_id,
                ),
            ),
            (
                "query_evidence_relevance_assessments.jsonl",
                select(QueryEvidenceRelevanceAssessmentModel.__table__)
                .where(
                    QueryEvidenceRelevanceAssessmentModel.query_executor_run_id
                    == run_id
                )
                .order_by(
                    QueryEvidenceRelevanceAssessmentModel.query_id,
                    QueryEvidenceRelevanceAssessmentModel.query_program_binding_id,
                ),
            ),
            (
                "query_execution_reuse_links.jsonl",
                select(QueryExecutionReuseLinkModel.__table__)
                .where(QueryExecutionReuseLinkModel.query_executor_run_id == run_id)
                .order_by(
                    QueryExecutionReuseLinkModel.query_id,
                    QueryExecutionReuseLinkModel.target_round_id,
                ),
            ),
            (
                "structured_answer_outputs.jsonl",
                select(StructuredAnswerOutputModel.__table__)
                .where(StructuredAnswerOutputModel.run_id == run_id)
                .order_by(
                    StructuredAnswerOutputModel.query_id,
                    StructuredAnswerOutputModel.answer_variant,
                ),
            ),
            (
                "answer_format_diagnostics.jsonl",
                select(AnswerFormatDiagnosticModel.__table__)
                .where(AnswerFormatDiagnosticModel.run_id == run_id)
                .order_by(
                    AnswerFormatDiagnosticModel.query_id,
                    AnswerFormatDiagnosticModel.answer_variant,
                    AnswerFormatDiagnosticModel.attempt_ordinal,
                ),
            ),
            (
                "answer_reuse_links.jsonl",
                select(AnswerReuseLinkModel.__table__)
                .where(AnswerReuseLinkModel.query_executor_run_id == run_id)
                .order_by(
                    AnswerReuseLinkModel.query_id,
                    AnswerReuseLinkModel.target_variant,
                ),
            ),
            (
                "query_fallback_decisions.jsonl",
                select(QueryFallbackDecisionModel.__table__)
                .where(QueryFallbackDecisionModel.query_executor_run_id == run_id)
                .order_by(QueryFallbackDecisionModel.query_id),
            ),
            (
                "query_answer_selections.jsonl",
                select(QueryAnswerSelectionModel.__table__)
                .where(QueryAnswerSelectionModel.query_executor_run_id == run_id)
                .order_by(QueryAnswerSelectionModel.query_id),
            ),
            (
                "pmmc_fallback_run_links.jsonl",
                select(PmmcFallbackRunLinkModel.__table__)
                .where(PmmcFallbackRunLinkModel.query_executor_run_id == run_id)
                .order_by(PmmcFallbackRunLinkModel.query_executor_run_id),
            ),
            (
                "pmmc_fallback_query_links.jsonl",
                select(PmmcFallbackQueryLinkModel.__table__)
                .where(PmmcFallbackQueryLinkModel.query_executor_run_id == run_id)
                .order_by(PmmcFallbackQueryLinkModel.query_id),
            ),
        ]
        match_batch_ids = select(QBankMatchBatchModel.match_batch_id).where(
            QBankMatchBatchModel.query_executor_run_id == run_id
        )
        specs.extend(
            [
                (
                    "qbank_lexical_term_facts.jsonl",
                    select(QBankLexicalTermFactModel.__table__)
                    .where(QBankLexicalTermFactModel.match_batch_id.in_(match_batch_ids))
                    .order_by(
                        QBankLexicalTermFactModel.match_batch_id,
                        QBankLexicalTermFactModel.ordinal,
                    ),
                ),
                (
                    "qbank_reranker_assessments.jsonl",
                    select(QBankRerankerAssessmentModel.__table__)
                    .where(
                        QBankRerankerAssessmentModel.query_executor_run_id == run_id
                    )
                    .order_by(
                        QBankRerankerAssessmentModel.query_id,
                        QBankRerankerAssessmentModel.qbank_id,
                    ),
                ),
                (
                    "qbank_match_channel_candidates.jsonl",
                    select(QBankMatchChannelCandidateModel.__table__)
                    .where(
                        QBankMatchChannelCandidateModel.match_batch_id.in_(
                            match_batch_ids
                        )
                    )
                    .order_by(
                        QBankMatchChannelCandidateModel.match_batch_id,
                        QBankMatchChannelCandidateModel.channel,
                        QBankMatchChannelCandidateModel.rank,
                    ),
                ),
                (
                    "qbank_match_fused_candidates.jsonl",
                    select(QBankMatchFusedCandidateModel.__table__)
                    .where(
                        QBankMatchFusedCandidateModel.match_batch_id.in_(
                            match_batch_ids
                        )
                    )
                    .order_by(
                        QBankMatchFusedCandidateModel.match_batch_id,
                        QBankMatchFusedCandidateModel.rank,
                    ),
                ),
            ]
        )
        relevance_assessment_ids = select(
            QueryEvidenceRelevanceAssessmentModel.relevance_assessment_id
        ).where(QueryEvidenceRelevanceAssessmentModel.query_executor_run_id == run_id)
        specs.append(
            (
                "query_evidence_relevance_items.jsonl",
                select(QueryEvidenceRelevanceItemModel.__table__)
                .where(
                    QueryEvidenceRelevanceItemModel.relevance_assessment_id.in_(
                        relevance_assessment_ids
                    )
                )
                .order_by(
                    QueryEvidenceRelevanceItemModel.relevance_assessment_id,
                    QueryEvidenceRelevanceItemModel.ordinal,
                ),
            )
        )
        job_ids = select(ExperimentJobModel.job_id).where(ExperimentJobModel.run_id == run_id)
        specs.append(
            (
                "job_attempts.jsonl",
                select(JobAttemptModel.__table__)
                .where(JobAttemptModel.job_id.in_(job_ids))
                .order_by(JobAttemptModel.job_id, JobAttemptModel.attempt_number),
            )
        )
        answer_ids = select(FinalAnswerModel.answer_id).where(FinalAnswerModel.run_id == run_id)
        context_ids = select(ContextAssemblyModel.context_assembly_id).where(
            ContextAssemblyModel.run_id == run_id
        )
        specs.extend(
            [
                (
                    "final_answer_evidence_ids.jsonl",
                    select(FinalAnswerEvidenceIdModel.__table__)
                    .where(FinalAnswerEvidenceIdModel.answer_id.in_(answer_ids))
                    .order_by(
                        FinalAnswerEvidenceIdModel.answer_id, FinalAnswerEvidenceIdModel.ordinal
                    ),
                ),
                (
                    "final_answer_image_ids.jsonl",
                    select(FinalAnswerImageIdModel.__table__)
                    .where(FinalAnswerImageIdModel.answer_id.in_(answer_ids))
                    .order_by(FinalAnswerImageIdModel.answer_id, FinalAnswerImageIdModel.ordinal),
                ),
                (
                    "context_assembly_memory_units.jsonl",
                    select(ContextAssemblyMemoryUnitModel.__table__)
                    .where(ContextAssemblyMemoryUnitModel.context_assembly_id.in_(context_ids))
                    .order_by(
                        ContextAssemblyMemoryUnitModel.context_assembly_id,
                        ContextAssemblyMemoryUnitModel.ordinal,
                    ),
                ),
                (
                    "context_assembly_images.jsonl",
                    select(ContextAssemblyImageModel.__table__)
                    .where(ContextAssemblyImageModel.context_assembly_id.in_(context_ids))
                    .order_by(
                        ContextAssemblyImageModel.context_assembly_id,
                        ContextAssemblyImageModel.role,
                        ContextAssemblyImageModel.ordinal,
                    ),
                ),
                (
                    "context_assembly_retrieval_candidates.jsonl",
                    select(ContextAssemblyRetrievalCandidateModel.__table__)
                    .where(
                        ContextAssemblyRetrievalCandidateModel.context_assembly_id.in_(
                            context_ids
                        )
                    )
                    .order_by(
                        ContextAssemblyRetrievalCandidateModel.context_assembly_id,
                        ContextAssemblyRetrievalCandidateModel.rank,
                    ),
                ),
            ]
        )
        batch_ids = select(RetrievalBatchModel.retrieval_batch_id).where(
            RetrievalBatchModel.run_id == run_id
        )
        trace_ids = select(RetrievalTraceModel.retrieval_trace_id).where(
            RetrievalTraceModel.run_id == run_id
        )
        specs.extend(
            [
                (
                    "retrieval_batch_query_embeddings.jsonl",
                    select(RetrievalBatchQueryEmbeddingModel.__table__)
                    .where(
                        RetrievalBatchQueryEmbeddingModel.retrieval_batch_id.in_(
                            batch_ids
                        )
                    )
                    .order_by(
                        RetrievalBatchQueryEmbeddingModel.retrieval_batch_id,
                        RetrievalBatchQueryEmbeddingModel.ordinal,
                    ),
                ),
                (
                    "retrieval_trace_components.jsonl",
                    select(RetrievalTraceComponentModel.__table__)
                    .where(
                        RetrievalTraceComponentModel.retrieval_trace_id.in_(trace_ids)
                    )
                    .order_by(
                        RetrievalTraceComponentModel.retrieval_trace_id,
                        RetrievalTraceComponentModel.ordinal,
                    ),
                ),
            ]
        )
        fallback_run_ids = select(PmmcFallbackRunLinkModel.fallback_run_id).where(
            PmmcFallbackRunLinkModel.query_executor_run_id == run_id
        )
        fallback_answer_ids = select(
            PmmcFallbackQueryLinkModel.fallback_answer_id
        ).where(PmmcFallbackQueryLinkModel.query_executor_run_id == run_id)
        inherited_calls = _reused_query_embedding_calls(
            select(RunManifestModel.run_id).where(or_(
                RunManifestModel.run_id == run_id,
                RunManifestModel.run_id.in_(fallback_run_ids),
            ))
        )
        specs.extend(
            [
                (
                    "reused_query_embedding_model_calls.jsonl",
                    select(RuntimeModelCallModel.__table__)
                    .where(RuntimeModelCallModel.model_call_id.in_(inherited_calls))
                    .order_by(RuntimeModelCallModel.model_call_id),
                ),
                (
                    "reused_query_embedding_source_runs.jsonl",
                    select(RunManifestModel.__table__).where(RunManifestModel.run_id.in_(
                        select(RuntimeModelCallModel.run_id).where(
                            RuntimeModelCallModel.model_call_id.in_(inherited_calls)
                        )
                    )).order_by(RunManifestModel.run_id),
                ),
                (
                    "pmmc_fallback_model_calls.jsonl",
                    select(RuntimeModelCallModel.__table__)
                    .where(RuntimeModelCallModel.run_id.in_(fallback_run_ids))
                    .order_by(
                        RuntimeModelCallModel.query_id,
                        RuntimeModelCallModel.created_at,
                        RuntimeModelCallModel.model_call_id,
                    ),
                ),
                (
                    "pmmc_fallback_answers.jsonl",
                    select(FinalAnswerModel.__table__)
                    .where(FinalAnswerModel.answer_id.in_(fallback_answer_ids))
                    .order_by(FinalAnswerModel.query_id, FinalAnswerModel.answer_id),
                ),
                (
                    "pmmc_fallback_answer_evidence_ids.jsonl",
                    select(FinalAnswerEvidenceIdModel.__table__)
                    .where(FinalAnswerEvidenceIdModel.answer_id.in_(fallback_answer_ids))
                    .order_by(
                        FinalAnswerEvidenceIdModel.answer_id,
                        FinalAnswerEvidenceIdModel.ordinal,
                    ),
                ),
                (
                    "pmmc_fallback_answer_image_ids.jsonl",
                    select(FinalAnswerImageIdModel.__table__)
                    .where(FinalAnswerImageIdModel.answer_id.in_(fallback_answer_ids))
                    .order_by(
                        FinalAnswerImageIdModel.answer_id,
                        FinalAnswerImageIdModel.ordinal,
                    ),
                ),
            ]
        )
        collection_ids = select(RetrievalBatchModel.embedding_collection_id).where(
            RetrievalBatchModel.run_id == run_id,
            RetrievalBatchModel.embedding_collection_id.is_not(None),
        )
        query_collection_ids = select(
            RetrievalBatchModel.query_embedding_collection_id
        ).where(
            RetrievalBatchModel.run_id == run_id,
            RetrievalBatchModel.query_embedding_collection_id.is_not(None),
        )
        qbank_collection_rows = (
            await session.execute(
                select(
                    QBankIndexManifestModel.text_question_collection_id,
                    QBankIndexManifestModel.text_paraphrase_collection_id,
                    QBankIndexManifestModel.multimodal_question_collection_id,
                    QBankIndexManifestModel.multimodal_paraphrase_collection_id,
                ).where(QBankIndexManifestModel.compiler_run_id == run_id)
            )
        ).all()
        qbank_collection_ids = sorted(
            {
                str(collection_id)
                for row in qbank_collection_rows
                for collection_id in row
                if collection_id is not None
            }
        )
        memory_collection_rows = (
            await session.execute(
                select(
                    MemoryOperatorIndexManifestModel.text_collection_id,
                    MemoryOperatorIndexManifestModel.visual_collection_id,
                ).where(MemoryOperatorIndexManifestModel.run_id == run_id)
            )
        ).all()
        memory_collection_ids = sorted(
            {
                str(collection_id)
                for row in memory_collection_rows
                for collection_id in row
                if collection_id is not None
            }
        )
        collections = await _fetch(
            session,
            select(EmbeddingCollectionModel.__table__)
            .where(
                (EmbeddingCollectionModel.run_id == run_id)
                | (EmbeddingCollectionModel.collection_id.in_(collection_ids))
                | (EmbeddingCollectionModel.collection_id.in_(query_collection_ids))
                | (EmbeddingCollectionModel.collection_id.in_(qbank_collection_ids))
                | (EmbeddingCollectionModel.collection_id.in_(memory_collection_ids))
            )
            .order_by(EmbeddingCollectionModel.collection_id),
        )
        _write_jsonl(root / "embedding_collections.jsonl", collections)
        ids = [item["collection_id"] for item in collections]
        await self._export_embeddings(session, ids, root)
        corpus_ids = {
            str(item["corpus_id"])
            for item in collections
            if item.get("corpus_id") is not None
        }
        corpus_ids.update(
            str(value)
            for value in (
                await session.scalars(
                    select(RetrievalBatchModel.corpus_id).where(
                        RetrievalBatchModel.run_id == run_id,
                        RetrievalBatchModel.corpus_id.is_not(None),
                    )
                )
            ).all()
        )
        corpus_references = await _fetch(
            session,
            select(
                RetrievalCorpusModel.corpus_id,
                RetrievalCorpusModel.snapshot_id,
                RetrievalCorpusModel.status,
                RetrievalCorpusModel.content_hash,
                RetrievalCorpusModel.manifest_json,
            )
            .where(RetrievalCorpusModel.corpus_id.in_(corpus_ids))
            .order_by(RetrievalCorpusModel.corpus_id),
        )
        _write_jsonl(root / "corpus_references.jsonl", corpus_references)
        for file_name, statement in specs:
            await _write_select_jsonl(
                session,
                root / file_name,
                statement,
                self.batch_size,
            )
        costs = await session.execute(
            text(
                "SELECT * FROM pmmc_runtime.query_cost_summary "
                "WHERE run_id = :run_id ORDER BY query_id NULLS FIRST"
            ),
            {"run_id": run_id},
        )
        _write_jsonl(
            root / "query_cost_summary.jsonl", [_row_dict(row) for row in costs.mappings()]
        )
        run_cost = await session.execute(
            text("SELECT * FROM pmmc_runtime.run_cost_summary WHERE run_id = :run_id"),
            {"run_id": run_id},
        )
        _write_jsonl(
            root / "run_cost_summary.jsonl",
            [_row_dict(row) for row in run_cost.mappings()],
        )
        routing_items = await session.execute(
            text(
                "SELECT * FROM pmmc_runtime.qbank_routing_items "
                "WHERE run_id = :run_id ORDER BY qbank_id"
            ),
            {"run_id": run_id},
        )
        _write_jsonl(
            root / "qbank_routing_items.jsonl",
            [_row_dict(row) for row in routing_items.mappings()],
        )
        routing_steps = await session.execute(
            text(
                "SELECT * FROM pmmc_runtime.qbank_routing_program_steps "
                "WHERE run_id = :run_id ORDER BY qbank_id, step_id"
            ),
            {"run_id": run_id},
        )
        _write_jsonl(
            root / "qbank_routing_program_steps.jsonl",
            [_row_dict(row) for row in routing_steps.mappings()],
        )
        compiler_costs = await session.execute(
            text(
                "SELECT * FROM pmmc_runtime.compiler_run_costs "
                "WHERE run_id = :run_id"
            ),
            {"run_id": run_id},
        )
        _write_jsonl(
            root / "compiler_run_costs.jsonl",
            [_row_dict(row) for row in compiler_costs.mappings()],
        )
        operator_summary = await session.execute(
            text(
                "SELECT * FROM pmmc_runtime.operator_execution_summary "
                "WHERE run_id = :run_id ORDER BY purpose, op_name"
            ),
            {"run_id": run_id},
        )
        _write_jsonl(
            root / "operator_execution_summary.jsonl",
            [_row_dict(row) for row in operator_summary.mappings()],
        )

    async def _load_referenced_model_artifacts(
        self, run_id: str,
    ) -> tuple[list[_ModelArtifactReference], list[RuntimeArtifactModel], int]:
        async with self.runtime_sessions() as session:
            linked_run_ids = list((await session.scalars(
                select(PmmcFallbackRunLinkModel.fallback_run_id).where(
                    PmmcFallbackRunLinkModel.query_executor_run_id == run_id
                )
            )).all())
            artifact_run_ids = [run_id, *sorted(set(linked_run_ids))]
            calls = (await session.execute(
                select(
                    RuntimeModelCallModel.model_call_id, RuntimeModelCallModel.run_id,
                    RuntimeModelCallModel.snapshot_id, RuntimeModelCallModel.input_asset_id,
                    RuntimeModelCallModel.output_asset_id,
                ).where(or_(
                    RuntimeModelCallModel.run_id.in_(artifact_run_ids),
                    RuntimeModelCallModel.model_call_id.in_(
                        _reused_query_embedding_calls(artifact_run_ids)
                    ),
                )).order_by(RuntimeModelCallModel.created_at, RuntimeModelCallModel.model_call_id)
            )).all()
            references = [
                _ModelArtifactReference(
                    str(call.model_call_id), role, str(artifact_id), str(call.run_id),
                    call.snapshot_id,
                )
                for call in calls
                for role, artifact_id in (("input", call.input_asset_id),
                                          ("output", call.output_asset_id))
                if artifact_id is not None
            ]
            artifact_ids = sorted({item.artifact_id for item in references})
            # Exact call references authorize reads, not every artifact in an ancestor run.
            artifacts = list(await session.scalars(
                select(RuntimeArtifactModel).where(
                    RuntimeArtifactModel.artifact_id.in_(artifact_ids)
                ).order_by(RuntimeArtifactModel.artifact_id)
            )) if artifact_ids else []
        return references, artifacts, sum(
            item.source_run_id not in artifact_run_ids for item in references
        )

    async def _archive_referenced_model_artifacts(
        self, run_id: str, root: Path,
    ) -> dict[str, int]:
        assert self.runtime_store is not None
        try:
            references, artifacts, inherited_count = await self._load_referenced_model_artifacts(
                run_id
            )
        except Exception as exc:
            _artifact_export_event(
                "failed", substage="inventory", error_code="artifact_inventory_failed",
                error_type=type(exc).__name__,
            )
            raise ModelArtifactExportError(
                "artifact_inventory_failed", substage="inventory",
            ) from exc
        artifact_map = {item.artifact_id: item for item in artifacts}
        references_by_artifact: dict[str, list[_ModelArtifactReference]] = {}
        for reference in references:
            references_by_artifact.setdefault(reference.artifact_id, []).append(reference)
        manifest_rows: list[dict[str, object]] = []
        payload_paths: set[str] = set()
        started = time.monotonic()
        next_report = started + _ARTIFACT_EXPORT_PROGRESS_INTERVAL_SECONDS
        archived_bytes = 0
        downloaded_count = 0
        _artifact_export_event(
            "started", reference_count=len(references),
            unique_artifact_count=len(references_by_artifact),
            inherited_reference_count=inherited_count,
            download_concurrency=self.artifact_concurrency, database_session_held=False,
        )
        try:
            # Validate the complete metadata closure before downloading any payload.
            for reference in references:
                artifact = artifact_map.get(reference.artifact_id)
                code = None
                if artifact is None:
                    code = "artifact_reference_missing"
                elif artifact.run_id != reference.source_run_id:
                    code = "artifact_call_owner_mismatch"
                elif (artifact.snapshot_id is not None and reference.snapshot_id is not None
                      and artifact.snapshot_id != reference.snapshot_id):
                    code = "artifact_call_snapshot_mismatch"
                elif artifact.state != ArtifactState.READY.value:
                    code = "artifact_not_ready"
                elif artifact.artifact_type not in {
                    AssetType.PROMPT.value,
                    AssetType.MODEL_RESPONSE.value,
                }:
                    code = "artifact_type_invalid"
                elif artifact.size_bytes is None:
                    code = "artifact_size_missing"
                if code is not None:
                    raise ModelArtifactExportError(
                        code, substage="metadata_preflight", reference=reference,
                    )
            _artifact_export_event(
                "preflight_ready", reference_count=len(references),
                unique_artifact_count=len(references_by_artifact),
                inherited_reference_count=inherited_count,
                ownership_policy="exact_model_call_owner_v1",
            )

            async def archive_one(artifact_id: str) -> None:
                nonlocal archived_bytes, downloaded_count, next_report
                assert self.runtime_store is not None
                artifact = artifact_map[artifact_id]
                reference = references_by_artifact[artifact_id][0]
                substage = "object_head"
                try:
                    remote = await self.runtime_store.head(artifact.object_key)
                    if remote is None:
                        raise ModelArtifactExportError(
                            "artifact_object_missing", substage=substage, reference=reference,
                        )
                    if remote.size_bytes != artifact.size_bytes:
                        raise ModelArtifactExportError(
                            "artifact_object_size_mismatch", substage=substage, reference=reference,
                        )
                    if remote.sha256 is not None and remote.sha256 != artifact.sha256:
                        raise ModelArtifactExportError(
                            "artifact_object_checksum_mismatch", substage=substage,
                            reference=reference,
                        )
                    substage = "payload_verification"
                    data = await self.runtime_store.get_bytes(artifact.object_key)
                    digest = hashlib.sha256(data).hexdigest()
                    if len(data) != artifact.size_bytes or digest != artifact.sha256:
                        raise ModelArtifactExportError(
                            "artifact_payload_mismatch", substage=substage, reference=reference,
                        )
                    extension = self._artifact_extension(artifact.mime_type)
                    relative = validate_object_key(
                        "model_artifacts/"
                        f"{artifact.artifact_type}/sha256/{digest[:2]}/{digest}.{extension}"
                    )
                    substage = "payload_write"
                    target = root / relative
                    if relative not in payload_paths:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
                        payload_paths.add(relative)
                        archived_bytes += len(data)
                    downloaded_count += 1
                    for ref in references_by_artifact[artifact_id]:
                        manifest_rows.append({
                            "artifact_id": artifact_id, "model_call_id": ref.model_call_id,
                            "role": ref.role, "artifact_type": artifact.artifact_type,
                            "relative_path": f"runtime/private/{relative}",
                            "sha256": digest, "size_bytes": len(data),
                            "mime_type": artifact.mime_type,
                        })
                    now = time.monotonic()
                    if now >= next_report:
                        _artifact_export_event(
                            "progress", archived_bytes=archived_bytes,
                            completed_reference_count=len(manifest_rows),
                            downloaded_artifact_count=downloaded_count,
                            elapsed_seconds=round(now - started, 1),
                            total_reference_count=len(references),
                            unique_payload_count=len(payload_paths),
                        )
                        next_report = now + _ARTIFACT_EXPORT_PROGRESS_INTERVAL_SECONDS
                except ModelArtifactExportError:
                    raise
                except Exception as exc:
                    raise ModelArtifactExportError(
                        "artifact_io_failed", substage=substage, reference=reference,
                    ) from exc

            artifact_iterator = iter(references_by_artifact)

            async def worker() -> None:
                for artifact_id in artifact_iterator:
                    await archive_one(artifact_id)

            tasks = [asyncio.create_task(worker()) for _ in range(
                min(self.artifact_concurrency, len(references_by_artifact))
            )]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        except Exception as exc:
            ref = exc.reference if isinstance(exc, ModelArtifactExportError) else None
            _artifact_export_event(
                "failed",
                archived_bytes=archived_bytes,
                completed_reference_count=len(manifest_rows),
                elapsed_seconds=round(time.monotonic() - started, 1),
                error_type=type(exc).__name__,
                error_code=(exc.error_code if isinstance(exc, ModelArtifactExportError)
                            else "artifact_archive_failed"),
                substage=(exc.substage if isinstance(exc, ModelArtifactExportError) else "archive"),
                artifact_locator=(hashlib.sha256(ref.artifact_id.encode()).hexdigest()[:16]
                                  if ref else None),
                model_call_locator=(hashlib.sha256(ref.model_call_id.encode()).hexdigest()[:16]
                                    if ref else None),
                total_reference_count=len(references),
                unique_payload_count=len(payload_paths),
            )
            raise
        _write_jsonl(
            root / "model_artifact_manifest.jsonl",
            sorted(
                manifest_rows,
                key=lambda item: (
                    str(item["model_call_id"]),
                    str(item["role"]),
                    str(item["artifact_id"]),
                ),
            ),
        )
        _artifact_export_event(
            "completed",
            archived_bytes=archived_bytes,
            completed_reference_count=len(manifest_rows),
            downloaded_artifact_count=downloaded_count,
            elapsed_seconds=round(time.monotonic() - started, 1),
            unique_payload_count=len(payload_paths),
        )
        return {
            "reference_count": len(manifest_rows),
            "payload_count": len(payload_paths),
            "size_bytes": sum(
                (root / path).stat().st_size for path in payload_paths
            ),
        }

    @staticmethod
    def _artifact_extension(mime_type: str | None) -> str:
        if mime_type == "application/json" or (
            mime_type is not None and mime_type.endswith("+json")
        ):
            return "json"
        if mime_type is not None and mime_type.startswith("text/"):
            return "txt"
        return "bin"

    @staticmethod
    def _is_questioner_manifest(manifest: object) -> bool:
        if not isinstance(manifest, dict):
            return False
        metadata = manifest.get("metadata")
        return isinstance(metadata, dict) and (
            isinstance(metadata.get("questioner_config"), dict)
            or metadata.get("target_stage")
            in {"questions", "programs", "verified_qbank"}
        )

    @staticmethod
    def _is_program_manifest(manifest: object) -> bool:
        if not isinstance(manifest, dict):
            return False
        metadata = manifest.get("metadata")
        return manifest.get("compiler_schema_version") in {
            "0.4.0",
            "0.5.0",
            "0.6.0",
            "0.7.0",
            "0.8.0",
        } or (
            isinstance(metadata, dict)
            and metadata.get("target_stage") in {"programs", "verified_qbank"}
        )

    @staticmethod
    def _questioner_protocol(manifest: object) -> str | None:
        if not isinstance(manifest, dict):
            return None
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict):
            return None
        config = metadata.get("questioner_config")
        if not isinstance(config, dict):
            return None
        value = config.get("protocol_version")
        return str(value) if value is not None else None

    async def _export_evaluation(self, session: AsyncSession, run_id: str, root: Path) -> None:
        semantic_run_ids = select(QBankSemanticMatchRunModel.semantic_run_id).where(
            QBankSemanticMatchRunModel.compiler_run_id == run_id
        )
        specs: list[tuple[str, Select]] = [
            (
                "run_artifacts.jsonl",
                select(EvaluationArtifactModel.__table__)
                .where(EvaluationArtifactModel.run_id == run_id)
                .order_by(EvaluationArtifactModel.artifact_id),
            ),
            (
                "model_calls.jsonl",
                select(EvaluationModelCallModel.__table__)
                .where(EvaluationModelCallModel.run_id == run_id)
                .order_by(
                    EvaluationModelCallModel.created_at, EvaluationModelCallModel.model_call_id
                ),
            ),
            (
                "retrieval_evaluations.jsonl",
                select(RetrievalEvaluationModel.__table__)
                .where(RetrievalEvaluationModel.run_id == run_id)
                .order_by(RetrievalEvaluationModel.retrieval_trace_id),
            ),
            (
                "metric_scores.jsonl",
                select(MetricScoreModel.__table__)
                .where(MetricScoreModel.run_id == run_id)
                .order_by(
                    MetricScoreModel.query_id,
                    MetricScoreModel.metric_name,
                    MetricScoreModel.metric_score_id,
                ),
            ),
            (
                "judge_scores.jsonl",
                select(JudgeScoreModel.__table__)
                .where(JudgeScoreModel.run_id == run_id)
                .order_by(
                    JudgeScoreModel.query_id,
                    JudgeScoreModel.judge_model_alias,
                    JudgeScoreModel.repeat_index,
                ),
            ),
            (
                "bad_case_labels.jsonl",
                select(BadCaseLabelModel.__table__)
                .where(BadCaseLabelModel.run_id == run_id)
                .order_by(BadCaseLabelModel.query_id, BadCaseLabelModel.bad_case_label_id),
            ),
            (
                "qbank_semantic_match_runs.jsonl",
                select(QBankSemanticMatchRunModel.__table__)
                .where(QBankSemanticMatchRunModel.compiler_run_id == run_id)
                .order_by(QBankSemanticMatchRunModel.semantic_run_id),
            ),
            (
                "qbank_semantic_match_jobs.jsonl",
                select(QBankSemanticMatchJobModel.__table__)
                .where(QBankSemanticMatchJobModel.semantic_run_id.in_(semantic_run_ids))
                .order_by(
                    QBankSemanticMatchJobModel.semantic_run_id,
                    QBankSemanticMatchJobModel.query_id,
                    QBankSemanticMatchJobModel.qbank_id,
                ),
            ),
            (
                "qbank_semantic_match_assessments.jsonl",
                select(QBankSemanticMatchAssessmentModel.__table__)
                .where(
                    QBankSemanticMatchAssessmentModel.semantic_run_id.in_(
                        semantic_run_ids
                    )
                )
                .order_by(
                    QBankSemanticMatchAssessmentModel.semantic_run_id,
                    QBankSemanticMatchAssessmentModel.query_id,
                    QBankSemanticMatchAssessmentModel.qbank_id,
                ),
            ),
            (
                "qbank_semantic_match_metrics.jsonl",
                select(QBankSemanticMetricModel.__table__)
                .where(QBankSemanticMetricModel.semantic_run_id.in_(semantic_run_ids))
                .order_by(
                    QBankSemanticMetricModel.semantic_run_id,
                    QBankSemanticMetricModel.query_id,
                    QBankSemanticMetricModel.question_budget,
                    QBankSemanticMetricModel.k,
                    QBankSemanticMetricModel.metric_name,
                ),
            ),
            (
                "structured_output_item_diagnostics.jsonl",
                select(EvaluationStructuredOutputItemDiagnosticModel.__table__)
                .where(
                    EvaluationStructuredOutputItemDiagnosticModel.semantic_run_id.in_(
                        semantic_run_ids
                    )
                )
                .order_by(
                    EvaluationStructuredOutputItemDiagnosticModel.semantic_run_id,
                    EvaluationStructuredOutputItemDiagnosticModel.model_call_id,
                    EvaluationStructuredOutputItemDiagnosticModel.item_ordinal,
                ),
            ),
        ]
        evaluation_ids = select(RetrievalEvaluationModel.retrieval_evaluation_id).where(
            RetrievalEvaluationModel.run_id == run_id
        )
        specs.append(
            (
                "retrieval_evaluation_matches.jsonl",
                select(RetrievalEvaluationMatchModel.__table__)
                .where(RetrievalEvaluationMatchModel.retrieval_evaluation_id.in_(evaluation_ids))
                .order_by(
                    RetrievalEvaluationMatchModel.retrieval_evaluation_id,
                    RetrievalEvaluationMatchModel.ordinal,
                ),
            )
        )
        for file_name, statement in specs:
            await _write_select_jsonl(
                session,
                root / file_name,
                statement,
                self.batch_size,
            )
        combined_query_costs = await session.execute(
            text(
                "SELECT * FROM pmmc_evaluation.query_cost_summary "
                "WHERE run_id = :run_id ORDER BY query_id NULLS FIRST"
            ),
            {"run_id": run_id},
        )
        _write_jsonl(
            root / "combined_query_cost_summary.jsonl",
            [_row_dict(row) for row in combined_query_costs.mappings()],
        )
        combined_run_cost = await session.execute(
            text("SELECT * FROM pmmc_evaluation.run_cost_summary WHERE run_id = :run_id"),
            {"run_id": run_id},
        )
        _write_jsonl(
            root / "combined_run_cost_summary.jsonl",
            [_row_dict(row) for row in combined_run_cost.mappings()],
        )

    async def _export_embeddings(
        self,
        session: AsyncSession,
        collection_ids: list[str],
        root: Path,
    ) -> None:
        metadata_path = root / "embeddings_metadata.jsonl"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if not collection_ids:
            metadata_path.write_bytes(b"")
            return
        statement = (
            select(EmbeddingModel.__table__)
            .where(EmbeddingModel.collection_id.in_(collection_ids))
            .order_by(EmbeddingModel.collection_id, EmbeddingModel.source_id)
            .execution_options(yield_per=self.batch_size)
        )
        result = await session.stream(statement)
        parquet_schema = pa.schema(
            [
                ("embedding_id", pa.string()),
                ("collection_id", pa.string()),
                ("source_id", pa.string()),
                ("dimension", pa.int32()),
                ("embedding", pa.list_(pa.float64())),
            ]
        )
        parquet_writer: pq.ParquetWriter | None = None
        parquet_rows: list[dict[str, Any]] = []
        try:
            with metadata_path.open("wb") as metadata_stream:
                async for mapping in result.mappings():
                    row = _row_dict(mapping)
                    vector = row.pop("embedding")
                    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
                    metadata_stream.write(_json_bytes(row))
                    metadata_stream.write(b"\n")
                    parquet_rows.append(
                        {
                            "embedding_id": row["embedding_id"],
                            "collection_id": row["collection_id"],
                            "source_id": row["source_id"],
                            "dimension": row["dimension"],
                            "embedding": values,
                        }
                    )
                    if len(parquet_rows) >= self.batch_size:
                        parquet_writer = self._write_embedding_batch(
                            root,
                            parquet_schema,
                            parquet_rows,
                            parquet_writer,
                        )
                        parquet_rows = []
                if parquet_rows:
                    parquet_writer = self._write_embedding_batch(
                        root,
                        parquet_schema,
                        parquet_rows,
                        parquet_writer,
                    )
        finally:
            if parquet_writer is not None:
                parquet_writer.close()

    def _write_embedding_batch(
        self,
        root: Path,
        schema: pa.Schema,
        rows: list[dict[str, Any]],
        writer: pq.ParquetWriter | None,
    ) -> pq.ParquetWriter:
        if writer is None:
            writer = pq.ParquetWriter(
                root / "embeddings.parquet",
                schema,
                compression=self.embedding_parquet_compression,
                write_statistics=False,
            )
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        return writer

    @staticmethod
    def _model_row(value: Any) -> dict[str, Any]:
        return {
            prop.columns[0].name: getattr(value, prop.key) for prop in value.__mapper__.column_attrs
        }

    @staticmethod
    def _file_inventory(root: Path) -> list[dict[str, Any]]:
        return [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(item for item in root.rglob("*") if item.is_file())
        ]


async def archive_experiment_export(
    export_dir: Path,
    store: ObjectStore,
) -> list[str]:
    """Archive a runtime-only export into one scope-safe bucket."""

    if (export_dir / "evaluation").exists():
        raise ValueError("evaluation exports require scope-aware archival")
    result = await archive_experiment_export_scoped(export_dir, store)
    return result["runtime"]


async def archive_experiment_export_scoped(
    export_dir: Path,
    runtime_store: ObjectStore,
    evaluation_store: ObjectStore | None = None,
) -> dict[str, list[str]]:
    """Archive runtime and evaluation files into their respective private buckets."""

    manifest = verify_experiment_export(export_dir)
    run_id = str(manifest["run_id"])
    export_id = str(manifest["export_id"])
    uris: dict[str, list[str]] = {"runtime": [], "evaluation": []}
    for path in sorted(item for item in export_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(export_dir).as_posix()
        evaluation_file = relative.startswith("evaluation/")
        if evaluation_file and evaluation_store is None:
            raise ValueError("evaluation export files require an evaluation object store")
        store = evaluation_store if evaluation_file else runtime_store
        assert store is not None
        key = validate_object_key(f"runs/{run_id}/trace_export/{export_id}/{relative}")
        metadata = await store.put_file(
            path,
            key,
            expected_sha256=sha256_file(path),
            content_type=_export_content_type(path),
        )
        scope = "evaluation" if evaluation_file else "runtime"
        uris[scope].append(metadata.uri)
    return uris


async def verify_archived_experiment_export(
    export_dir: Path,
    runtime_store: ObjectStore,
    evaluation_store: ObjectStore | None = None,
) -> dict[str, int]:
    """Verify every archived export object against its local immutable checksum."""

    manifest = verify_experiment_export(export_dir)
    run_id = str(manifest["run_id"])
    export_id = str(manifest["export_id"])
    counts = {"runtime": 0, "evaluation": 0}
    for path in sorted(item for item in export_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(export_dir).as_posix()
        evaluation_file = relative.startswith("evaluation/")
        store = evaluation_store if evaluation_file else runtime_store
        if store is None:
            raise ValueError("evaluation export files require an evaluation object store")
        key = validate_object_key(f"runs/{run_id}/trace_export/{export_id}/{relative}")
        remote = await store.head(key)
        expected = sha256_file(path)
        if remote is None or remote.size_bytes != path.stat().st_size:
            raise ValueError(f"archived export object is missing or truncated: {relative}")
        if remote.sha256 != expected:
            raise ValueError(f"archived export checksum mismatch: {relative}")
        scope = "evaluation" if evaluation_file else "runtime"
        counts[scope] += 1
    return counts


def _export_content_type(path: Path) -> str | None:
    if path.suffix in {".json", ".jsonl"}:
        return "application/json"
    if path.suffix == ".parquet":
        return "application/vnd.apache.parquet"
    return None
