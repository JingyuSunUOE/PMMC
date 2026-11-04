"""Deterministic public/private exports for a frozen QBank coverage run."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text

from pmmc.evaluation.qbank_protocols import (
    HARDENED_QBANK_COVERAGE_PROTOCOLS,
    QBANK_EVIDENCE_RESOLUTION_V2,
)
from pmmc.infrastructure.database.compiler_models import QBankQuestionModel
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.experiment_models import (
    EvaluationArtifactModel,
    EvaluationModelCallModel,
)
from pmmc.infrastructure.database.questioner_models import (
    EvaluationStructuredOutputItemDiagnosticModel,
    QBankCoverageCandidateModel,
    QBankCoverageGateReportModel,
    QBankCoverageGoldGroupMemberModel,
    QBankCoverageGoldGroupModel,
    QBankCoverageLabelModel,
    QBankCoverageMatchModel,
    QBankCoverageMetricModel,
    QBankCoverageQueryEmbeddingModel,
    QBankCoverageQueryModel,
    QBankCoverageRunModel,
    QBankIndexManifestModel,
    QBankSemanticMatchAssessmentModel,
    QBankSemanticMatchJobModel,
    QBankSemanticMatchRunModel,
    QBankSemanticMetricModel,
    QuestionerCandidateGateResultModel,
    QuestionerGenerationModel,
)
from pmmc.infrastructure.object_store import (
    create_evaluator_object_stores,
    sha256_file,
    validate_object_key,
)
from pmmc.infrastructure.settings import Settings
from pmmc.schemas import (
    COMPILER_SCHEMA_VERSION,
    EXPERIMENT_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
    SCHEMA_VERSION,
    QBankCoverageStatus,
)


@dataclass(frozen=True)
class QBankCoverageExportResult:
    coverage_run_id: str
    export_id: str
    export_dir: Path


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            ensure_ascii=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value, pretty=True))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            stream.write(_json_bytes(row))


def _model_row(value: Any, *, exclude: set[str] | None = None) -> dict[str, Any]:
    blocked = exclude or set()
    return {
        prop.columns[0].name: _jsonable(getattr(value, prop.key))
        for prop in value.__mapper__.column_attrs
        if prop.columns[0].name not in blocked
    }


def _metric_summary(metrics: list[Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int | None, int | None], list[Any]] = {}
    for metric in metrics:
        key = (
            metric.metric_name,
            metric.metric_version,
            metric.question_budget,
            metric.k,
        )
        groups.setdefault(key, []).append(metric)
    result: list[dict[str, Any]] = []
    for key, values in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2] or 0,
            item[0][3] or 0,
        ),
    ):
        applicable = [
            float(item.value)
            for item in values
            if getattr(item, "applicable", True)
        ]
        result.append(
            {
                "metric_name": key[0],
                "metric_version": key[1],
                "question_budget": key[2],
                "k": key[3],
                "query_count": len(values),
                "applicable_count": len(applicable),
                "not_applicable_count": len(values) - len(applicable),
                "mean": sum(applicable) / len(applicable) if applicable else None,
            }
        )
    return result


def _semantic_export_dimensions_passed(dimensions: dict[str, Any]) -> bool:
    """Mirror the persisted semantic-assessment tri-state contract exactly."""

    image_binding = dimensions.get("query_image_binding_compatible")
    return (
        dimensions.get("same_information_need") is True
        and dimensions.get("candidate_answer_sufficient") is True
        and dimensions.get("contradiction_free") is True
        and (image_binding is True or image_binding == "not_applicable")
    )


def _semantic_assessment_export_issues(
    *,
    assessment: dict[str, Any],
    semantic_job: dict[str, Any] | None,
    semantic_run_id: str,
    coverage_run_id: str,
) -> tuple[list[str], bool]:
    dimensions = assessment.get("dimensions")
    if not isinstance(dimensions, dict):
        return ["dimensions_unavailable"], False

    dimensions_passed = _semantic_export_dimensions_passed(dimensions)
    expected_match = dimensions_passed and (
        assessment.get("evidence_proxy_complete") is True
    )
    issues: list[str] = []
    if semantic_job is None:
        issues.append("job_missing")
        return issues, dimensions_passed
    if assessment.get("semantic_run_id") != semantic_run_id:
        issues.append("semantic_run_mismatch")
    if assessment.get("coverage_run_id") != coverage_run_id:
        issues.append("coverage_run_mismatch")
    if assessment.get("query_id") != semantic_job.get("query_id"):
        issues.append("query_lineage_mismatch")
    if assessment.get("qbank_id") != semantic_job.get("qbank_id"):
        issues.append("candidate_lineage_mismatch")
    if assessment.get("model_call_id") != semantic_job.get("model_call_id"):
        issues.append("model_call_lineage_mismatch")
    if assessment.get("semantic_dimensions_passed") is not dimensions_passed:
        issues.append("dimension_decision_mismatch")
    if assessment.get("semantic_match") is not expected_match:
        issues.append("semantic_match_decision_mismatch")
    return issues, dimensions_passed


def _write_vector_parquet(path: Path, embeddings: list[Any]) -> None:
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("QBank vector export requires the declared pyarrow dependency") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "query_embedding_id": item.query_embedding_id,
            "coverage_run_id": item.coverage_run_id,
            "query_id": item.query_id,
            "channel": item.channel,
            "model_alias": item.model_alias,
            "model_revision": item.model_revision,
            "dimension": item.dimension,
            "vector": _jsonable(item.embedding),
        }
        for item in embeddings
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
    )


class QBankCoverageExporter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def export(self, coverage_run_id: str) -> QBankCoverageExportResult:
        engine, sessions = create_session_factory(self.settings, DatabaseRole.EVALUATOR)
        temporary = (
            self.settings.outputs_path
            / "qbank-coverage"
            / ".tmp"
            / f"{coverage_run_id}-{hashlib.sha256(coverage_run_id.encode()).hexdigest()[:8]}"
        )
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            async with sessions() as session:
                run = await session.get(QBankCoverageRunModel, coverage_run_id)
                if run is None:
                    raise KeyError(f"QBank coverage run does not exist: {coverage_run_id}")
                if run.status != QBankCoverageStatus.COMPLETED.value:
                    raise ValueError("only completed QBank coverage runs can be exported")
                report = await session.scalar(
                    select(QBankCoverageGateReportModel).where(
                        QBankCoverageGateReportModel.coverage_run_id == coverage_run_id
                    )
                )
                index = await session.get(QBankIndexManifestModel, run.qbank_index_id)
                if report is None or index is None:
                    raise ValueError("QBank coverage export requires its gate report and index")

                queries = list(
                    await session.scalars(
                        select(QBankCoverageQueryModel)
                        .where(QBankCoverageQueryModel.coverage_run_id == coverage_run_id)
                        .order_by(QBankCoverageQueryModel.ordinal)
                    )
                )
                embeddings = list(
                    await session.scalars(
                        select(QBankCoverageQueryEmbeddingModel)
                        .where(
                            QBankCoverageQueryEmbeddingModel.coverage_run_id
                            == coverage_run_id
                        )
                        .order_by(
                            QBankCoverageQueryEmbeddingModel.query_id,
                            QBankCoverageQueryEmbeddingModel.channel,
                        )
                    )
                )
                candidates = list(
                    await session.scalars(
                        select(QBankCoverageCandidateModel)
                        .where(
                            QBankCoverageCandidateModel.coverage_run_id
                            == coverage_run_id
                        )
                        .order_by(
                            QBankCoverageCandidateModel.query_id,
                            QBankCoverageCandidateModel.question_budget,
                            QBankCoverageCandidateModel.rank,
                        )
                    )
                )
                candidate_ids = [item.coverage_candidate_id for item in candidates]
                matches = (
                    list(
                        await session.scalars(
                            select(QBankCoverageMatchModel)
                            .where(
                                QBankCoverageMatchModel.coverage_candidate_id.in_(
                                    candidate_ids
                                )
                            )
                            .order_by(
                                QBankCoverageMatchModel.coverage_candidate_id,
                                QBankCoverageMatchModel.ordinal,
                            )
                        )
                    )
                    if candidate_ids
                    else []
                )
                metrics = list(
                    await session.scalars(
                        select(QBankCoverageMetricModel)
                        .where(QBankCoverageMetricModel.coverage_run_id == coverage_run_id)
                        .order_by(QBankCoverageMetricModel.metric_id)
                    )
                )
                labels = list(
                    await session.scalars(
                        select(QBankCoverageLabelModel)
                        .where(QBankCoverageLabelModel.coverage_run_id == coverage_run_id)
                        .order_by(QBankCoverageLabelModel.label_id)
                    )
                )
                gold_groups = list(
                    await session.scalars(
                        select(QBankCoverageGoldGroupModel)
                        .where(
                            QBankCoverageGoldGroupModel.coverage_run_id
                            == coverage_run_id
                        )
                        .order_by(
                            QBankCoverageGoldGroupModel.query_id,
                            QBankCoverageGoldGroupModel.gold_group_id,
                        )
                    )
                )
                gold_group_ids = [item.gold_group_id for item in gold_groups]
                gold_members = (
                    list(
                        await session.scalars(
                            select(QBankCoverageGoldGroupMemberModel)
                            .where(
                                QBankCoverageGoldGroupMemberModel.gold_group_id.in_(
                                    gold_group_ids
                                )
                            )
                            .order_by(
                                QBankCoverageGoldGroupMemberModel.gold_group_id,
                                QBankCoverageGoldGroupMemberModel.member_type,
                                QBankCoverageGoldGroupMemberModel.member_id,
                            )
                        )
                    )
                    if gold_group_ids
                    else []
                )
                generation_rows = (
                    await session.execute(
                        select(
                            func.count(QuestionerGenerationModel.generation_id),
                            func.coalesce(
                                func.sum(QuestionerGenerationModel.raw_candidate_count), 0
                            ),
                            func.coalesce(
                                func.sum(QuestionerGenerationModel.valid_candidate_count), 0
                            ),
                            func.coalesce(
                                func.sum(QuestionerGenerationModel.invalid_candidate_count), 0
                            ),
                            func.count(QuestionerGenerationModel.generation_id).filter(
                                QuestionerGenerationModel.valid_candidate_count == 0
                            ),
                        ).where(QuestionerGenerationModel.run_id == run.compiler_run_id)
                    )
                ).one()
                generation_details = list(
                    await session.scalars(
                        select(QuestionerGenerationModel)
                        .where(QuestionerGenerationModel.run_id == run.compiler_run_id)
                        .order_by(QuestionerGenerationModel.generation_id)
                    )
                )
                gate_rows: dict[str, int] = {
                    str(status): int(count)
                    for status, count in (
                        await session.execute(
                            select(
                                QuestionerCandidateGateResultModel.status,
                                func.count(),
                            )
                            .where(
                                QuestionerCandidateGateResultModel.run_id
                                == run.compiler_run_id
                            )
                            .group_by(QuestionerCandidateGateResultModel.status)
                        )
                    ).all()
                }
                duplicate_count = int(
                    await session.scalar(
                        select(func.count())
                        .select_from(QuestionerCandidateGateResultModel)
                        .where(
                            QuestionerCandidateGateResultModel.run_id
                            == run.compiler_run_id,
                            QuestionerCandidateGateResultModel.duplicate_of_qbank_id.is_not(
                                None
                            ),
                        )
                    )
                    or 0
                )
                taxonomy = [
                    {"question_type": question_type, "count": int(count)}
                    for question_type, count in (
                        await session.execute(
                            select(QBankQuestionModel.question_type, func.count())
                            .join(
                                QuestionerCandidateGateResultModel,
                                QuestionerCandidateGateResultModel.qbank_id
                                == QBankQuestionModel.qbank_id,
                            )
                            .where(
                                QBankQuestionModel.run_id == run.compiler_run_id,
                                QuestionerCandidateGateResultModel.status
                                == "eligible_for_planning",
                            )
                            .group_by(QBankQuestionModel.question_type)
                            .order_by(QBankQuestionModel.question_type)
                        )
                    ).all()
                ]
                modalities = [
                    {"modality": modality, "count": int(count)}
                    for modality, count in (
                        await session.execute(
                            text(
                                "SELECT modalities.modality, count(*) "
                                "FROM pmmc_runtime.qbank_questions questions "
                                "JOIN pmmc_runtime.questioner_candidate_gate_results gates "
                                "ON gates.qbank_id=questions.qbank_id "
                                "AND gates.run_id=questions.run_id "
                                "CROSS JOIN LATERAL "
                                "unnest(questions.required_modalities) "
                                "AS modalities(modality) "
                                "WHERE questions.run_id=:run_id "
                                "AND gates.status='eligible_for_planning' "
                                "GROUP BY modalities.modality "
                                "ORDER BY modalities.modality"
                            ),
                            {"run_id": run.compiler_run_id},
                        )
                    ).all()
                ]
                semantic_neighbor_count = int(
                    await session.scalar(
                        text(
                            "SELECT count(*) FROM pmmc_runtime.qbank_duplicate_links "
                            "WHERE run_id=:run_id AND NOT hard_duplicate"
                        ),
                        {"run_id": run.compiler_run_id},
                    )
                    or 0
                )
                semantic_run = await session.scalar(
                    select(QBankSemanticMatchRunModel).where(
                        QBankSemanticMatchRunModel.coverage_run_id
                        == coverage_run_id
                    )
                )
                semantic_jobs: list[Any] = []
                semantic_assessments: list[Any] = []
                semantic_metrics: list[Any] = []
                semantic_diagnostics: list[Any] = []
                semantic_model_calls: list[Any] = []
                semantic_artifacts: list[Any] = []
                if semantic_run is not None:
                    if semantic_run.status != "completed":
                        raise ValueError(
                            "QBank coverage export found a non-terminal semantic run"
                        )
                    semantic_jobs = list(
                        await session.scalars(
                            select(QBankSemanticMatchJobModel)
                            .where(
                                QBankSemanticMatchJobModel.semantic_run_id
                                == semantic_run.semantic_run_id
                            )
                            .order_by(
                                QBankSemanticMatchJobModel.query_id,
                                QBankSemanticMatchJobModel.qbank_id,
                            )
                        )
                    )
                    semantic_assessments = list(
                        await session.scalars(
                            select(QBankSemanticMatchAssessmentModel)
                            .where(
                                QBankSemanticMatchAssessmentModel.semantic_run_id
                                == semantic_run.semantic_run_id
                            )
                            .order_by(
                                QBankSemanticMatchAssessmentModel.query_id,
                                QBankSemanticMatchAssessmentModel.qbank_id,
                            )
                        )
                    )
                    semantic_metrics = list(
                        await session.scalars(
                            select(QBankSemanticMetricModel)
                            .where(
                                QBankSemanticMetricModel.semantic_run_id
                                == semantic_run.semantic_run_id
                            )
                            .order_by(
                                QBankSemanticMetricModel.query_id,
                                QBankSemanticMetricModel.question_budget,
                                QBankSemanticMetricModel.k,
                                QBankSemanticMetricModel.metric_name,
                            )
                        )
                    )
                    semantic_diagnostics = list(
                        await session.scalars(
                            select(EvaluationStructuredOutputItemDiagnosticModel)
                            .where(
                                EvaluationStructuredOutputItemDiagnosticModel.semantic_run_id
                                == semantic_run.semantic_run_id
                            )
                            .order_by(
                                EvaluationStructuredOutputItemDiagnosticModel.model_call_id,
                                EvaluationStructuredOutputItemDiagnosticModel.item_ordinal,
                            )
                        )
                    )
                    semantic_model_calls = list(
                        await session.scalars(
                            select(EvaluationModelCallModel)
                            .where(
                                EvaluationModelCallModel.semantic_run_id
                                == semantic_run.semantic_run_id
                            )
                            .order_by(
                                EvaluationModelCallModel.created_at,
                                EvaluationModelCallModel.model_call_id,
                            )
                        )
                    )
                    semantic_artifact_ids = {
                        str(artifact_id)
                        for call in semantic_model_calls
                        for artifact_id in (
                            call.input_asset_id,
                            call.output_asset_id,
                        )
                        if artifact_id is not None
                    }
                    semantic_artifacts = (
                        list(
                            await session.scalars(
                                select(EvaluationArtifactModel)
                                .where(
                                    EvaluationArtifactModel.artifact_id.in_(
                                        sorted(semantic_artifact_ids)
                                    ),
                                    EvaluationArtifactModel.semantic_run_id
                                    == semantic_run.semantic_run_id,
                                )
                                .order_by(EvaluationArtifactModel.artifact_id)
                            )
                        )
                        if semantic_artifact_ids
                        else []
                    )
                    if {
                        item.artifact_id for item in semantic_artifacts
                    } != semantic_artifact_ids:
                        raise ValueError(
                            "semantic model calls reference missing evaluation artifacts"
                        )
                revision = await session.scalar(text("SELECT version_num FROM alembic_version"))

            generation_count = int(generation_rows[0])
            raw_count = int(generation_rows[1])
            valid_count = int(generation_rows[2])
            invalid_count = int(generation_rows[3])
            zero_count = int(generation_rows[4])
            gate_total = sum(int(value) for value in gate_rows.values())
            eligible_count = int(gate_rows.get("eligible_for_planning", 0))
            gate_rejected_count = int(gate_rows.get("rejected", 0))
            error_locations: dict[str, int] = {}
            error_types: dict[str, int] = {}
            for generation in generation_details:
                diagnostics = dict(generation.metadata_json or {}).get(
                    "invalid_diagnostics", []
                )
                if not isinstance(diagnostics, list):
                    continue
                for diagnostic in diagnostics:
                    if not isinstance(diagnostic, dict):
                        continue
                    errors = diagnostic.get("errors", [])
                    if not isinstance(errors, list):
                        continue
                    for error in errors:
                        if not isinstance(error, dict):
                            continue
                        location = str(error.get("location", "unknown"))
                        error_type = str(error.get("type", "validation_error"))
                        error_locations[location] = error_locations.get(location, 0) + 1
                        error_types[error_type] = error_types.get(error_type, 0) + 1
            questioner_summary = {
                "summary_protocol_version": "questioner-summary-v2",
                "compiler_run_id": run.compiler_run_id,
                "prompt_versions": sorted(
                    {str(item.prompt_version) for item in generation_details}
                ),
                "parser_versions": sorted(
                    {str(item.parser_version) for item in generation_details}
                ),
                "unit_generation_count": generation_count,
                "raw_candidate_count": raw_count,
                "schema_valid_candidate_count": valid_count,
                "schema_invalid_candidate_count": invalid_count,
                "eligible_candidate_count": eligible_count,
                "rejected_candidate_count": gate_rejected_count,
                "gate_rejected_candidate_count": gate_rejected_count,
                "total_rejected_candidate_count": invalid_count + gate_rejected_count,
                "duplicate_candidate_count": duplicate_count,
                "semantic_neighbor_count": semantic_neighbor_count,
                "zero_valid_candidate_unit_count": zero_count,
                "schema_valid_rate": valid_count / raw_count if raw_count else None,
                "gate_pass_rate_among_schema_valid": (
                    eligible_count / valid_count if valid_count else None
                ),
                "overall_eligible_rate": (
                    eligible_count / raw_count if raw_count else None
                ),
                "zero_valid_candidate_unit_rate": (
                    zero_count / generation_count if generation_count else None
                ),
                "gate_pass_rate": (
                    eligible_count / gate_total
                    if gate_total
                    else None
                ),
                "duplicate_rate": duplicate_count / gate_total if gate_total else None,
                "zero_candidate_rate": zero_count / generation_count if generation_count else None,
                "schema_error_location_counts": [
                    {"location": key, "count": value}
                    for key, value in sorted(error_locations.items())
                ],
                "schema_error_type_counts": [
                    {"type": key, "count": value}
                    for key, value in sorted(error_types.items())
                ],
                "question_type_distribution": taxonomy,
                "required_modality_distribution": modalities,
            }
            _write_json(
                temporary / "public" / "coverage_summary.json",
                {
                    "coverage_run_id": coverage_run_id,
                    "compiler_run_id": run.compiler_run_id,
                    "dataset": run.dataset,
                    "protocol_version": run.protocol_version,
                    "query_count": report.query_count,
                    "applicable_query_count": report.applicable_query_count,
                    "gate_mode": report.mode,
                    "hard_integrity_passed": report.hard_integrity_passed,
                    "scientific_thresholds_passed": report.scientific_thresholds_passed,
                    "failure_counts": report.failure_counts,
                },
            )
            _write_json(temporary / "public" / "questioner_summary.json", questioner_summary)
            _write_jsonl(
                temporary / "public" / "metric_summary.jsonl",
                _metric_summary(metrics),
            )
            _write_jsonl(
                temporary / "public" / "semantic_metric_summary.jsonl",
                _metric_summary(semantic_metrics),
            )

            private_tables: dict[str, list[Any]] = {
                "coverage_queries.jsonl": queries,
                "coverage_candidates.jsonl": candidates,
                "coverage_matches.jsonl": matches,
                "coverage_metrics.jsonl": metrics,
                "coverage_labels.jsonl": labels,
                "gold_evidence_groups.jsonl": gold_groups,
                "gold_evidence_group_members.jsonl": gold_members,
                "semantic_match_jobs.jsonl": semantic_jobs,
                "semantic_match_assessments.jsonl": semantic_assessments,
                "semantic_match_metrics.jsonl": semantic_metrics,
                "semantic_structured_output_diagnostics.jsonl": semantic_diagnostics,
                "semantic_model_calls.jsonl": semantic_model_calls,
            }
            _write_json(temporary / "private" / "coverage_run.json", _model_row(run))
            _write_json(temporary / "private" / "gate_report.json", _model_row(report))
            _write_json(
                temporary / "private" / "semantic_match_run.json",
                _model_row(semantic_run) if semantic_run is not None else {},
            )
            for file_name, values in private_tables.items():
                _write_jsonl(
                    temporary / "private" / file_name,
                    [_model_row(item) for item in values],
                )
            _write_jsonl(
                temporary / "private" / "query_embedding_metadata.jsonl",
                [_model_row(item, exclude={"embedding"}) for item in embeddings],
            )
            _write_vector_parquet(
                temporary / "private" / "query_embedding_vectors.parquet",
                embeddings,
            )
            artifact_lineage = {
                str(artifact_id): {
                    "model_call_id": call.model_call_id,
                    "semantic_job_id": call.semantic_job_id,
                    "role": role,
                }
                for call in semantic_model_calls
                for role, artifact_id in (
                    ("input", call.input_asset_id),
                    ("output", call.output_asset_id),
                )
                if artifact_id is not None
            }
            expected_artifact_reference_count = sum(
                artifact_id is not None
                for call in semantic_model_calls
                for artifact_id in (call.input_asset_id, call.output_asset_id)
            )
            if len(artifact_lineage) != expected_artifact_reference_count:
                raise ValueError("semantic model artifacts have ambiguous call lineage")
            _runtime_store, evaluation_store = create_evaluator_object_stores(
                self.settings
            )
            archived_artifacts: list[dict[str, Any]] = []
            for artifact in semantic_artifacts:
                if (
                    artifact.state != "ready"
                    or artifact.size_bytes is None
                    or artifact.ready_at is None
                ):
                    raise ValueError(
                        "semantic model artifact is not ready for export"
                    )
                remote = await evaluation_store.head(artifact.object_key)
                if (
                    remote is None
                    or remote.sha256 != artifact.sha256
                    or remote.size_bytes != artifact.size_bytes
                ):
                    raise ValueError(
                        "semantic model artifact failed object-store verification"
                    )
                payload = await evaluation_store.get_bytes(artifact.object_key)
                if (
                    len(payload) != artifact.size_bytes
                    or hashlib.sha256(payload).hexdigest() != artifact.sha256
                ):
                    raise ValueError(
                        "semantic model artifact payload failed checksum verification"
                    )
                suffix = (
                    ".json"
                    if artifact.mime_type == "application/json"
                    else ".bin"
                )
                relative = (
                    Path("private")
                    / "model_artifacts"
                    / str(artifact.artifact_type)
                    / "sha256"
                    / artifact.sha256[:2]
                    / f"{artifact.sha256}{suffix}"
                )
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if hashlib.sha256(destination.read_bytes()).hexdigest() != (
                        artifact.sha256
                    ):
                        raise ValueError(
                            "content-addressed semantic artifact conflicts"
                        )
                else:
                    destination.write_bytes(payload)
                lineage = artifact_lineage.get(artifact.artifact_id)
                if lineage is None:
                    raise ValueError(
                        "semantic model artifact has no model-call lineage"
                    )
                archived_artifacts.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "semantic_run_id": artifact.semantic_run_id,
                        "semantic_job_id": lineage["semantic_job_id"],
                        "model_call_id": lineage["model_call_id"],
                        "role": lineage["role"],
                        "artifact_type": artifact.artifact_type,
                        "relative_path": relative.as_posix(),
                        "sha256": artifact.sha256,
                        "size_bytes": artifact.size_bytes,
                        "mime_type": artifact.mime_type,
                    }
                )
            _write_jsonl(
                temporary
                / "private"
                / "semantic_model_artifact_manifest.jsonl",
                sorted(
                    archived_artifacts,
                    key=lambda item: (
                        str(item["artifact_id"]),
                        str(item["role"]),
                    ),
                ),
            )

            files = [
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(item for item in temporary.rglob("*") if item.is_file())
            ]
            content = {
                "coverage_run_id": coverage_run_id,
                "compiler_run_id": run.compiler_run_id,
                "qbank_index_id": run.qbank_index_id,
                "coverage_config_hash": run.config_hash,
                "protocol_version": run.protocol_version,
                "canonical_schema_version": SCHEMA_VERSION,
                "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
                "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
                "compiler_schema_version": COMPILER_SCHEMA_VERSION,
                "alembic_revision": revision,
                "file_count": len(files),
                "files": files,
            }
            export_id = hashlib.sha256(_json_bytes(content)).hexdigest()
            _write_json(
                temporary / "export_manifest.json",
                {"export_id": export_id, **content},
            )
            destination = (
                self.settings.outputs_path
                / "qbank-coverage"
                / coverage_run_id
                / export_id
            )
            if destination.exists():
                verify_qbank_coverage_export(destination)
                shutil.rmtree(temporary)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(destination)
            return QBankCoverageExportResult(coverage_run_id, export_id, destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        finally:
            await engine.dispose()


def verify_qbank_coverage_export(export_dir: Path) -> dict[str, Any]:
    manifest_path = export_dir / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("file_count") != len(files):
        raise ValueError("QBank coverage export manifest has an invalid file inventory")
    expected = sorted(str(item["path"]) for item in files)
    actual = sorted(
        path.relative_to(export_dir).as_posix()
        for path in export_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    if expected != actual:
        raise ValueError("QBank coverage export files differ from the manifest")
    root = export_dir.resolve()
    for item in files:
        relative = validate_object_key(str(item["path"]))
        path = export_dir / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"QBank coverage export path escapes its root: {relative}")
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"QBank coverage export checksum mismatch: {relative}")
    content = {key: value for key, value in manifest.items() if key != "export_id"}
    if hashlib.sha256(_json_bytes(content)).hexdigest() != manifest["export_id"]:
        raise ValueError("QBank coverage export ID does not match its content")

    def read_jsonl(relative: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            (export_dir / relative).read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"QBank coverage export row is not an object: "
                    f"{relative}:{line_number}"
                )
            rows.append(value)
        return rows

    coverage_run = json.loads(
        (export_dir / "private" / "coverage_run.json").read_text(encoding="utf-8")
    )
    if not isinstance(coverage_run, dict):
        raise ValueError("QBank coverage run export must be an object")
    if coverage_run.get("protocol_version") in HARDENED_QBANK_COVERAGE_PROTOCOLS:
        metadata = coverage_run.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("evidence_resolution_policy_version")
            != QBANK_EVIDENCE_RESOLUTION_V2
        ):
            raise ValueError(
                "hardened QBank coverage run lacks evidence-resolution-v2 identity"
            )
        if any(
            row.get("resolution_policy_version") != QBANK_EVIDENCE_RESOLUTION_V2
            for row in read_jsonl("private/gold_evidence_groups.jsonl")
        ):
            raise ValueError(
                "hardened QBank coverage gold groups use another evidence policy"
            )

    broad_session_round_match_count = 0
    for candidate in read_jsonl("private/coverage_candidates.jsonl"):
        metadata = candidate.get("metadata")
        if not isinstance(metadata, dict):
            continue
        contextual = {
            str(value)
            for value in metadata.get("contextual_only_gold_group_ids", [])
        }
        discriminative = {
            str(value)
            for value in metadata.get("matched_gold_group_ids", [])
        }
        overlap = contextual.intersection(discriminative)
        broad_session_round_match_count += len(overlap)
    if broad_session_round_match_count:
        raise ValueError(
            "contextual-only evidence was counted as a discriminative QBank match"
        )

    semantic_run_path = export_dir / "private" / "semantic_match_run.json"
    semantic_report: dict[str, Any] = {
        "semantic_match_present": False,
        "semantic_run_id": None,
        "semantic_pair_count": 0,
        "semantic_assessment_count": 0,
        "semantic_metric_count": 0,
        "semantic_structured_output_issue_count": 0,
        "semantic_model_call_count": 0,
        "semantic_model_artifact_count": 0,
        "semantic_native_output_zero": True,
    }
    if semantic_run_path.is_file():
        semantic_run = json.loads(semantic_run_path.read_text(encoding="utf-8"))
        if not isinstance(semantic_run, dict):
            raise ValueError("semantic match run export must be an object")
        if semantic_run:
            semantic_run_id = str(semantic_run.get("semantic_run_id") or "")
            coverage_run_id = str(semantic_run.get("coverage_run_id") or "")
            if not semantic_run_id or semantic_run.get("status") != "completed":
                raise ValueError("semantic match export requires one completed run")
            if coverage_run_id != str(manifest.get("coverage_run_id")):
                raise ValueError("semantic match export belongs to another coverage run")

            jobs = read_jsonl("private/semantic_match_jobs.jsonl")
            assessments = read_jsonl("private/semantic_match_assessments.jsonl")
            metrics = read_jsonl("private/semantic_match_metrics.jsonl")
            diagnostics = read_jsonl(
                "private/semantic_structured_output_diagnostics.jsonl"
            )
            model_calls = read_jsonl("private/semantic_model_calls.jsonl")
            artifact_rows = read_jsonl(
                "private/semantic_model_artifact_manifest.jsonl"
            )
            pair_count = int(semantic_run.get("pair_count", -1))
            if pair_count < 0 or len(jobs) != pair_count:
                raise ValueError("semantic match job denominator is incomplete")
            if len(assessments) != pair_count or len(model_calls) != pair_count:
                raise ValueError(
                    "semantic assessment/model-call denominator is incomplete"
                )

            job_by_id: dict[str, dict[str, Any]] = {}
            pair_keys: set[tuple[str, str]] = set()
            for job in jobs:
                job_id = str(job.get("semantic_job_id") or "")
                pair = (str(job.get("query_id") or ""), str(job.get("qbank_id") or ""))
                if (
                    not job_id
                    or not all(pair)
                    or job_id in job_by_id
                    or pair in pair_keys
                    or job.get("semantic_run_id") != semantic_run_id
                    or job.get("coverage_run_id") != coverage_run_id
                    or job.get("status") != "succeeded"
                    or not job.get("model_call_id")
                ):
                    raise ValueError("semantic match job lineage is invalid")
                job_by_id[job_id] = job
                pair_keys.add(pair)

            assessment_by_job: dict[str, dict[str, Any]] = {}
            for assessment_ordinal, assessment in enumerate(assessments, start=1):
                job_id = str(assessment.get("semantic_job_id") or "")
                semantic_job = job_by_id.get(job_id)
                issues, _ = _semantic_assessment_export_issues(
                    assessment=assessment,
                    semantic_job=semantic_job,
                    semantic_run_id=semantic_run_id,
                    coverage_run_id=coverage_run_id,
                )
                if job_id in assessment_by_job:
                    issues.append("duplicate_job_assessment")
                if issues:
                    dimensions = assessment.get("dimensions")
                    image_binding = (
                        dimensions.get("query_image_binding_compatible")
                        if isinstance(dimensions, dict)
                        else "unavailable"
                    )
                    print(
                        "QBANK_SEMANTIC_EXPORT_ASSESSMENT_FAILURE "
                        + json.dumps(
                            {
                                "assessment_ordinal": assessment_ordinal,
                                "conservative_fallback": (
                                    isinstance(assessment.get("metadata"), dict)
                                    and assessment["metadata"].get(
                                        "conservative_fallback"
                                    )
                                    is True
                                ),
                                "dataset": semantic_run.get("dataset"),
                                "image_binding_state": image_binding,
                                "issue_codes": sorted(set(issues)),
                                "semantic_protocol": semantic_run.get(
                                    "protocol_version"
                                ),
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    raise ValueError("semantic assessment lineage or decision is invalid")
                assessment_by_job[job_id] = assessment
            if set(assessment_by_job) != set(job_by_id):
                raise ValueError("semantic assessment set differs from pair jobs")
            print(
                "QBANK_SEMANTIC_EXPORT_ASSESSMENT_ACCOUNTING "
                + json.dumps(
                    {
                        "assessment_count": len(assessments),
                        "conservative_fallback_count": sum(
                            isinstance(item.get("metadata"), dict)
                            and item["metadata"].get("conservative_fallback") is True
                            for item in assessments
                        ),
                        "dataset": semantic_run.get("dataset"),
                        "not_applicable_image_binding_count": sum(
                            isinstance(item.get("dimensions"), dict)
                            and item["dimensions"].get(
                                "query_image_binding_compatible"
                            )
                            == "not_applicable"
                            for item in assessments
                        ),
                        "state": "verified",
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )

            call_by_id: dict[str, dict[str, Any]] = {}
            for call in model_calls:
                call_id = str(call.get("model_call_id") or "")
                job_id = str(call.get("semantic_job_id") or "")
                semantic_job = job_by_id.get(job_id)
                if (
                    not call_id
                    or call_id in call_by_id
                    or semantic_job is None
                    or call_id != semantic_job.get("model_call_id")
                    or call.get("semantic_run_id") != semantic_run_id
                    or call.get("evaluation_run_id") is not None
                    or call.get("model_alias") != semantic_run.get("judge_model_alias")
                ):
                    raise ValueError("semantic model-call parent or model identity is invalid")
                call_by_id[call_id] = call
            if set(call_by_id) != {
                str(job["model_call_id"]) for job in jobs
            }:
                raise ValueError("semantic model-call set differs from pair jobs")

            coverage_queries = read_jsonl("private/coverage_queries.jsonl")
            applicable_query_ids = {
                str(item["query_id"])
                for item in coverage_queries
                if item.get("applicable") is True
            }
            question_budgets = list(coverage_run.get("question_budgets") or [])
            match_ks = list(coverage_run.get("match_ks") or [])
            expected_metric_count = (
                len(applicable_query_ids)
                * len(question_budgets)
                * len(match_ks)
                * 2
            )
            metric_keys: set[tuple[str, str, int, int]] = set()
            allowed_metrics = {
                "qbank_semantic_match_at_k",
                "semantic_projected_fallback_at_k",
            }
            for metric in metrics:
                key = (
                    str(metric.get("query_id") or ""),
                    str(metric.get("metric_name") or ""),
                    int(metric.get("question_budget", 0)),
                    int(metric.get("k", 0)),
                )
                if (
                    metric.get("semantic_run_id") != semantic_run_id
                    or key in metric_keys
                    or key[0] not in applicable_query_ids
                    or key[1] not in allowed_metrics
                    or key[2] not in question_budgets
                    or key[3] not in match_ks
                ):
                    raise ValueError("semantic metric grid contains invalid rows")
                metric_keys.add(key)
            if len(metrics) != expected_metric_count:
                raise ValueError("semantic metric grid is incomplete")

            diagnostic_call_ids = {
                str(item.get("model_call_id") or "") for item in diagnostics
            }
            if (
                any(item.get("semantic_run_id") != semantic_run_id for item in diagnostics)
                or not diagnostic_call_ids <= set(call_by_id)
            ):
                raise ValueError("semantic structured-output diagnostics have invalid lineage")

            expected_artifact_roles = {
                (str(artifact_id), role): (
                    str(call.get("model_call_id") or ""),
                    str(call.get("semantic_job_id") or ""),
                )
                for call in model_calls
                for role, artifact_id in (
                    ("input", call.get("input_asset_id")),
                    ("output", call.get("output_asset_id")),
                )
                if artifact_id is not None
            }
            actual_artifact_roles: set[tuple[str, str]] = set()
            for artifact in artifact_rows:
                artifact_id = str(artifact.get("artifact_id") or "")
                role = str(artifact.get("role") or "")
                relative = str(artifact.get("relative_path") or "")
                path = export_dir / relative
                expected_lineage = expected_artifact_roles.get((artifact_id, role))
                exported_model_call_id = artifact.get("model_call_id")
                exported_semantic_job_id = artifact.get("semantic_job_id")
                if (
                    artifact.get("semantic_run_id") != semantic_run_id
                    or (artifact_id, role) in actual_artifact_roles
                    or expected_lineage is None
                    or (
                        exported_model_call_id is not None
                        and str(exported_model_call_id) != expected_lineage[0]
                    )
                    or (
                        exported_semantic_job_id is not None
                        and str(exported_semantic_job_id) != expected_lineage[1]
                    )
                    or not relative.startswith("private/model_artifacts/")
                    or not path.is_file()
                    or path.stat().st_size != int(artifact.get("size_bytes", -1))
                    or sha256_file(path) != artifact.get("sha256")
                ):
                    raise ValueError("semantic model artifact archive is invalid")
                actual_artifact_roles.add((artifact_id, role))
            if actual_artifact_roles != set(expected_artifact_roles):
                raise ValueError("semantic model artifact set differs from model calls")

            semantic_report = {
                "semantic_match_present": True,
                "semantic_run_id": semantic_run_id,
                "semantic_pair_count": pair_count,
                "semantic_assessment_count": len(assessments),
                "semantic_metric_count": len(metrics),
                "semantic_structured_output_issue_count": len(diagnostics),
                "semantic_model_call_count": len(model_calls),
                "semantic_model_artifact_count": len(artifact_rows),
                "semantic_native_output_zero": len(diagnostics) == 0,
            }
    return {
        **manifest,
        **semantic_report,
        "broad_session_round_match_count": broad_session_round_match_count,
    }
