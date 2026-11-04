"""Lossless row mappings for Step 6C.2 semantic and diagnostic facts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pmmc.schemas import (
    QBankSemanticMatchAssessment,
    QBankSemanticMatchJob,
    QBankSemanticMatchRun,
    QBankSemanticMetric,
    StructuredOutputItemDiagnostic,
)

Row = dict[str, Any]


def _row(value: object) -> Row:
    data = value.model_dump(mode="python")  # type: ignore[attr-defined]
    if "metadata" in data:
        data["metadata"] = dict(data["metadata"])
    return data


def structured_diagnostic_to_row(value: StructuredOutputItemDiagnostic) -> Row:
    data = _row(value)
    data["stage"] = value.stage.value
    data["status"] = value.status.value
    data["errors"] = [item.model_dump(mode="json") for item in value.errors]
    return data


def structured_diagnostic_from_row(
    row: Mapping[str, Any],
) -> StructuredOutputItemDiagnostic:
    return StructuredOutputItemDiagnostic.model_validate(dict(row))


def semantic_run_to_row(value: QBankSemanticMatchRun) -> Row:
    data = _row(value)
    data["dataset"] = value.dataset.value
    data["status"] = value.status.value
    return data


def semantic_run_from_row(row: Mapping[str, Any]) -> QBankSemanticMatchRun:
    return QBankSemanticMatchRun.model_validate(dict(row))


def semantic_job_to_row(value: QBankSemanticMatchJob) -> Row:
    data = _row(value)
    data["status"] = value.status.value
    return data


def semantic_job_from_row(row: Mapping[str, Any]) -> QBankSemanticMatchJob:
    return QBankSemanticMatchJob.model_validate(dict(row))


def semantic_assessment_to_row(value: QBankSemanticMatchAssessment) -> Row:
    data = _row(value)
    data["dimensions"] = value.dimensions.model_dump(mode="json")
    data["reason_codes"] = [item.value for item in value.reason_codes]
    return data


def semantic_assessment_from_row(
    row: Mapping[str, Any],
) -> QBankSemanticMatchAssessment:
    return QBankSemanticMatchAssessment.model_validate(dict(row))


def semantic_metric_to_row(value: QBankSemanticMetric) -> Row:
    return _row(value)


def semantic_metric_from_row(row: Mapping[str, Any]) -> QBankSemanticMetric:
    return QBankSemanticMetric.model_validate(dict(row))
