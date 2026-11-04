"""Lossless Planner v1 and program dry-run row mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pmmc.schemas import (
    CompilerStageJob,
    MemoryOperatorIndexManifest,
    PlannerBatchProgram,
    PlannerCandidateDraftFact,
    PlannerCandidateOutcome,
    PlannerGenerationBatch,
    PlannerRequirementContract,
    ProgramDryRunAssessment,
    ProgramEvidenceMatch,
    ProgramQueryEmbedding,
    ProgramQueryEmbeddingBatch,
    ProgramValidationIssueFact,
    ProgramValidationResultFact,
)

Row = dict[str, Any]


def _row(value: object) -> Row:
    data = value.model_dump(mode="python")  # type: ignore[attr-defined]
    if "metadata" in data:
        data["metadata"] = dict(data["metadata"])
    return data


def compiler_stage_job_to_row(value: CompilerStageJob) -> Row:
    data = _row(value)
    data["stage"] = value.stage.value
    data["status"] = value.status.value
    return data


def compiler_stage_job_from_row(row: Mapping[str, Any]) -> CompilerStageJob:
    return CompilerStageJob.model_validate(dict(row))


def planner_generation_batch_to_row(value: PlannerGenerationBatch) -> Row:
    return _row(value)


def planner_generation_batch_from_row(
    row: Mapping[str, Any],
) -> PlannerGenerationBatch:
    return PlannerGenerationBatch.model_validate(dict(row))


def planner_candidate_outcome_to_row(value: PlannerCandidateOutcome) -> Row:
    data = _row(value)
    data["status"] = value.status.value
    return data


def planner_candidate_outcome_from_row(
    row: Mapping[str, Any],
) -> PlannerCandidateOutcome:
    return PlannerCandidateOutcome.model_validate(dict(row))


def planner_requirement_contract_to_row(value: PlannerRequirementContract) -> Row:
    data = _row(value)
    data["required_modalities"] = [item.value for item in value.required_modalities]
    data["required_capabilities"] = [
        item.value for item in value.required_capabilities
    ]
    data["allowed_capabilities"] = [
        item.value for item in value.allowed_capabilities
    ]
    return data


def planner_requirement_contract_from_row(
    row: Mapping[str, Any],
) -> PlannerRequirementContract:
    return PlannerRequirementContract.model_validate(dict(row))


def planner_candidate_draft_to_row(value: PlannerCandidateDraftFact) -> Row:
    return _row(value)


def planner_candidate_draft_from_row(
    row: Mapping[str, Any],
) -> PlannerCandidateDraftFact:
    return PlannerCandidateDraftFact.model_validate(dict(row))


def planner_batch_program_to_row(value: PlannerBatchProgram) -> Row:
    return _row(value)


def validation_result_to_rows(
    value: ProgramValidationResultFact,
) -> tuple[Row, list[Row]]:
    result = value.model_dump(mode="python", exclude={"issues"})
    result["validation_kind"] = value.validation_kind.value
    result["issue_count"] = len(value.issues)
    result["metadata"] = dict(value.metadata)
    issues: list[Row] = []
    for issue in value.issues:
        row = _row(issue)
        row["severity"] = issue.severity.value
        issues.append(row)
    return result, issues


def validation_result_from_rows(
    row: Mapping[str, Any],
    issues: Sequence[Mapping[str, Any]],
) -> ProgramValidationResultFact:
    data = dict(row)
    declared_count = int(data.pop("issue_count"))
    values = [
        ProgramValidationIssueFact.model_validate(dict(item))
        for item in sorted(issues, key=lambda item: int(item["ordinal"]))
    ]
    if declared_count != len(values):
        raise ValueError("persisted program validation issue count does not close")
    return ProgramValidationResultFact.model_validate({**data, "issues": values})


def memory_index_to_row(value: MemoryOperatorIndexManifest) -> Row:
    return _row(value)


def memory_index_from_row(row: Mapping[str, Any]) -> MemoryOperatorIndexManifest:
    return MemoryOperatorIndexManifest.model_validate(dict(row))


def query_embedding_batch_to_row(value: ProgramQueryEmbeddingBatch) -> Row:
    return _row(value)


def query_embedding_batch_from_row(
    row: Mapping[str, Any],
) -> ProgramQueryEmbeddingBatch:
    return ProgramQueryEmbeddingBatch.model_validate(dict(row))


def query_embedding_to_row(value: ProgramQueryEmbedding) -> Row:
    return _row(value)


def query_embedding_from_row(row: Mapping[str, Any]) -> ProgramQueryEmbedding:
    return ProgramQueryEmbedding.model_validate(dict(row))


def dry_run_assessment_to_rows(
    value: ProgramDryRunAssessment,
    matches: Sequence[ProgramEvidenceMatch],
) -> tuple[Row, list[Row]]:
    assessment = _row(value)
    assessment["status"] = value.status.value
    assessment["control_action"] = value.control_action.value
    rows: list[Row] = []
    for match in matches:
        if match.assessment_id != value.assessment_id or match.run_id != value.run_id:
            raise ValueError("dry-run evidence match does not belong to assessment")
        row = _row(match)
        row["expected_evidence_type"] = match.expected_evidence_type.value
        row["actual_evidence_type"] = match.actual_evidence_type.value
        row["match_kind"] = match.match_kind.value
        rows.append(row)
    return assessment, rows


def dry_run_assessment_from_rows(
    row: Mapping[str, Any],
    matches: Sequence[Mapping[str, Any]],
) -> tuple[ProgramDryRunAssessment, tuple[ProgramEvidenceMatch, ...]]:
    assessment = ProgramDryRunAssessment.model_validate(dict(row))
    values = tuple(
        ProgramEvidenceMatch.model_validate(dict(item))
        for item in sorted(matches, key=lambda item: str(item["match_id"]))
    )
    return assessment, values
