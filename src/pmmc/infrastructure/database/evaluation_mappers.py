"""Lossless mappings for paired baseline suites and evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pmmc.schemas import (
    AnswerNormalization,
    BaselineMethod,
    BaselineSuiteManifest,
    BaselineSuiteQuery,
    BaselineSuiteRunLink,
    BaselineSuiteState,
    ContextEvidenceEvaluation,
    EvaluationJob,
    EvaluationJobAttempt,
    EvaluationRunManifest,
    EvaluationRunState,
    EvaluationTarget,
    ResolvedGoldEvidenceGroup,
)

Row = dict[str, Any]


def _metadata(data: Row, value: Any) -> Row:
    data["metadata"] = dict(value.metadata)
    return data


def suite_manifest_to_row(value: BaselineSuiteManifest) -> Row:
    data = value.model_dump(mode="python")
    data["dataset"] = value.dataset.value
    data["methods"] = [item.value for item in value.methods]
    return _metadata(data, value)


def suite_manifest_from_row(row: Mapping[str, Any]) -> BaselineSuiteManifest:
    return BaselineSuiteManifest.model_validate(dict(row))


def suite_state_to_row(value: BaselineSuiteState) -> Row:
    data = value.model_dump(mode="python")
    data["status"] = value.status.value
    return _metadata(data, value)


def suite_state_from_row(row: Mapping[str, Any]) -> BaselineSuiteState:
    return BaselineSuiteState.model_validate(dict(row))


def suite_run_to_row(value: BaselineSuiteRunLink) -> Row:
    data = value.model_dump(mode="python")
    data["method"] = value.method.value
    return _metadata(data, value)


def suite_run_from_row(row: Mapping[str, Any]) -> BaselineSuiteRunLink:
    return BaselineSuiteRunLink.model_validate(dict(row))


def suite_query_to_row(value: BaselineSuiteQuery) -> Row:
    return _metadata(value.model_dump(mode="python"), value)


def suite_query_from_row(row: Mapping[str, Any]) -> BaselineSuiteQuery:
    return BaselineSuiteQuery.model_validate(dict(row))


def evaluation_manifest_to_row(value: EvaluationRunManifest) -> Row:
    data = value.model_dump(mode="python", exclude={"source_run_id"})
    data["source_kind"] = value.source_kind.value
    data["legacy_source_run_id"] = value.source_run_id
    return _metadata(data, value)


def evaluation_manifest_from_row(row: Mapping[str, Any]) -> EvaluationRunManifest:
    data = dict(row)
    data["source_run_id"] = data.pop("legacy_source_run_id", None)
    return EvaluationRunManifest.model_validate(data)


def evaluation_state_to_row(value: EvaluationRunState) -> Row:
    data = value.model_dump(mode="python")
    data["status"] = value.status.value
    return _metadata(data, value)


def evaluation_state_from_row(row: Mapping[str, Any]) -> EvaluationRunState:
    return EvaluationRunState.model_validate(dict(row))


def evaluation_target_to_row(value: EvaluationTarget) -> Row:
    data = value.model_dump(mode="python")
    data["source_kind"] = value.source_kind.value
    data["method"] = (
        value.method.value if isinstance(value.method, BaselineMethod) else value.method
    )
    return _metadata(data, value)


def evaluation_target_from_row(row: Mapping[str, Any]) -> EvaluationTarget:
    return EvaluationTarget.model_validate(dict(row))


def evaluation_job_to_row(value: EvaluationJob) -> Row:
    data = value.model_dump(mode="python")
    data["stage"] = value.stage.value
    data["status"] = value.status.value
    return _metadata(data, value)


def evaluation_job_from_row(row: Mapping[str, Any]) -> EvaluationJob:
    return EvaluationJob.model_validate(dict(row))


def evaluation_attempt_to_row(value: EvaluationJobAttempt) -> Row:
    data = value.model_dump(mode="python")
    data["status"] = value.status.value
    return _metadata(data, value)


def evaluation_attempt_from_row(row: Mapping[str, Any]) -> EvaluationJobAttempt:
    return EvaluationJobAttempt.model_validate(dict(row))


def answer_normalization_to_row(value: AnswerNormalization) -> Row:
    return _metadata(value.model_dump(mode="python"), value)


def gold_group_to_rows(value: ResolvedGoldEvidenceGroup) -> dict[str, list[Row]]:
    parent = value.model_dump(mode="python", exclude={"members"})
    parent["granularity"] = value.granularity.value
    _metadata(parent, value)
    members = []
    for member in value.members:
        members.append(
            {
                "gold_group_id": value.gold_group_id,
                "member_type": member.member_type.value,
                "member_id": member.member_id,
                "derived": member.derived,
                "metadata": dict(member.metadata),
            }
        )
    return {"group": [parent], "members": members}


def context_evaluation_to_rows(value: ContextEvidenceEvaluation) -> dict[str, list[Row]]:
    parent = value.model_dump(mode="python", exclude={"matched_gold_group_ids"})
    parent["granularity"] = value.granularity.value
    parent["context_source_kind"] = value.context_source_kind.value
    _metadata(parent, value)
    matches = [
        {
            "context_evaluation_id": value.context_evaluation_id,
            "gold_group_id": group_id,
            "ordinal": ordinal,
        }
        for ordinal, group_id in enumerate(value.matched_gold_group_ids)
    ]
    return {"evaluation": [parent], "matches": matches}
