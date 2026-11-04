"""Lossless Step 6D schema-to-row mappings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pmmc.schemas import (
    CandidateStageJob,
    DoubterAssessment,
    DoubterDirective,
    PlannerRevisionLineage,
    QBankRoundProjectionManifest,
    QBankRoundProjectionMember,
    R0InputManifest,
    RoutingProjectionManifest,
    RoutingProjectionMember,
    StageComponentManifest,
)

Row = dict[str, Any]


def _row(value: object) -> Row:
    data = value.model_dump(mode="python")  # type: ignore[attr-defined]
    if "metadata" in data:
        data["metadata"] = dict(data["metadata"])
    return data


def r0_to_row(value: R0InputManifest) -> Row:
    return _row(value)


def r0_from_row(row: Mapping[str, Any]) -> R0InputManifest:
    return R0InputManifest.model_validate(dict(row))


def candidate_job_to_row(value: CandidateStageJob) -> Row:
    row = _row(value)
    row["stage"] = value.stage.value
    row["status"] = value.status.value
    return row


def candidate_job_from_row(row: Mapping[str, Any]) -> CandidateStageJob:
    return CandidateStageJob.model_validate(dict(row))


def assessment_to_rows(
    value: DoubterAssessment,
) -> tuple[Row, list[Row], list[Row]]:
    row = _row(value)
    row["action"] = value.action.value
    row["issue_codes"] = [item.value for item in value.issue_codes]
    row.pop("dimensions")
    row.pop("directives")
    provider_verdicts = value.metadata.get("provider_dimension_verdicts", {})
    if not isinstance(provider_verdicts, Mapping):
        provider_verdicts = {}
    dimensions = [
        {
            "assessment_id": value.assessment_id,
            "run_id": value.run_id,
            "dimension": name,
            "passed": passed,
            "verdict": str(
                provider_verdicts.get(name, "pass" if passed else "fail")
            ),
        }
        for name, passed in value.dimensions.model_dump(mode="python").items()
    ]
    directives = [directive_to_row(item) for item in value.directives]
    return row, dimensions, directives


def assessment_from_rows(
    row: Mapping[str, Any],
    dimensions: list[Mapping[str, Any]],
    directives: list[Mapping[str, Any]],
) -> DoubterAssessment:
    payload = dict(row)
    payload["dimensions"] = {str(item["dimension"]): bool(item["passed"]) for item in dimensions}
    payload["directives"] = [directive_from_row(item) for item in directives]
    return DoubterAssessment.model_validate(payload)


def directive_to_row(value: DoubterDirective) -> Row:
    row = _row(value)
    row["directive"] = value.directive.value
    return row


def directive_from_row(row: Mapping[str, Any]) -> DoubterDirective:
    return DoubterDirective.model_validate(dict(row))


def revision_to_row(value: PlannerRevisionLineage) -> Row:
    return _row(value)


def revision_from_row(row: Mapping[str, Any]) -> PlannerRevisionLineage:
    return PlannerRevisionLineage.model_validate(dict(row))


def round_projection_to_row(value: QBankRoundProjectionManifest) -> Row:
    return _row(value)


def round_projection_from_row(
    row: Mapping[str, Any],
) -> QBankRoundProjectionManifest:
    return QBankRoundProjectionManifest.model_validate(dict(row))


def round_member_to_row(value: QBankRoundProjectionMember) -> Row:
    row = _row(value)
    row["route_class"] = value.route_class.value
    row["outcome"] = value.outcome.value
    return row


def round_member_from_row(row: Mapping[str, Any]) -> QBankRoundProjectionMember:
    return QBankRoundProjectionMember.model_validate(dict(row))


def routing_projection_to_row(value: RoutingProjectionManifest) -> Row:
    return _row(value)


def routing_projection_from_row(
    row: Mapping[str, Any],
) -> RoutingProjectionManifest:
    return RoutingProjectionManifest.model_validate(dict(row))


def routing_member_to_row(value: RoutingProjectionMember) -> Row:
    row = _row(value)
    row["route_class"] = value.route_class.value
    return row


def routing_member_from_row(row: Mapping[str, Any]) -> RoutingProjectionMember:
    return RoutingProjectionMember.model_validate(dict(row))


def stage_component_to_row(value: StageComponentManifest) -> Row:
    row = _row(value)
    row["status"] = value.status.value
    return row


def stage_component_from_row(row: Mapping[str, Any]) -> StageComponentManifest:
    return StageComponentManifest.model_validate(dict(row))
