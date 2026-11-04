"""Lossless schema-to-row mappings for experiment persistence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pmmc.schemas import (
    BadCaseLabel,
    DataScope,
    EmbeddingCollection,
    EmbeddingRecord,
    ExperimentArtifact,
    ExperimentJob,
    FinalAnswerRecord,
    JobAttempt,
    JudgeScore,
    MetricScore,
    ModelCallTrace,
    PipelineStageSummary,
    RankingScore,
    RetrievalBatchTrace,
    RetrievalEvaluation,
    RetrievalQueryEmbeddingRef,
    RetrievalTrace,
    RetrievalTraceComponent,
    RunQueryAssignment,
    RunState,
)

Row = dict[str, Any]


def run_state_to_row(value: RunState) -> Row:
    data = value.model_dump(mode="python")
    data["status"] = value.status.value
    data["metadata"] = dict(value.metadata)
    return data


def run_state_from_row(row: Mapping[str, Any]) -> RunState:
    return RunState.model_validate(dict(row))


def run_query_to_row(value: RunQueryAssignment) -> Row:
    return value.model_dump(mode="python")


def run_query_from_row(row: Mapping[str, Any]) -> RunQueryAssignment:
    return RunQueryAssignment.model_validate(dict(row))


def run_stage_summary_to_row(run_id: str, value: PipelineStageSummary) -> Row:
    return {
        "run_id": run_id,
        **value.model_dump(mode="python", by_alias=True),
    }


def job_to_row(value: ExperimentJob) -> Row:
    data = value.model_dump(mode="python")
    data["stage"] = value.stage.value
    data["status"] = value.status.value
    data["metadata"] = dict(value.metadata)
    return data


def job_from_row(row: Mapping[str, Any]) -> ExperimentJob:
    return ExperimentJob.model_validate(dict(row))


def job_attempt_to_row(value: JobAttempt) -> Row:
    data = value.model_dump(mode="python")
    data["status"] = value.status.value
    data["metadata"] = dict(value.metadata)
    return data


def job_attempt_from_row(row: Mapping[str, Any]) -> JobAttempt:
    return JobAttempt.model_validate(dict(row))


def artifact_to_row(value: ExperimentArtifact) -> Row:
    data = value.model_dump(mode="python")
    data.pop("data_scope")
    if value.data_scope is DataScope.RUNTIME:
        data.pop("evaluation_run_id", None)
        data.pop("semantic_run_id", None)
    data["artifact_type"] = value.artifact_type.value
    data["state"] = value.state.value
    data["metadata"] = dict(value.metadata)
    return data


def artifact_from_row(row: Mapping[str, Any], scope: DataScope) -> ExperimentArtifact:
    return ExperimentArtifact.model_validate({**dict(row), "data_scope": scope.value})


def embedding_collection_to_row(value: EmbeddingCollection) -> Row:
    data = value.model_dump(mode="python")
    data["source_type"] = value.source_type.value
    data["distance_metric"] = value.distance_metric.value
    data["status"] = value.status.value
    data["metadata"] = dict(value.metadata)
    return data


def embedding_collection_from_row(row: Mapping[str, Any]) -> EmbeddingCollection:
    return EmbeddingCollection.model_validate(dict(row))


def embedding_to_row(value: EmbeddingRecord) -> Row:
    data = value.model_dump(mode="python")
    data["embedding"] = data.pop("vector")
    data["metadata"] = dict(value.metadata)
    return data


def embedding_from_row(row: Mapping[str, Any]) -> EmbeddingRecord:
    data = dict(row)
    vector = data.pop("embedding")
    data["vector"] = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    return EmbeddingRecord.model_validate(data)


def model_call_to_row(value: ModelCallTrace) -> Row:
    data = value.model_dump(mode="python")
    data.pop("data_scope")
    if value.data_scope is DataScope.RUNTIME:
        data.pop("evaluation_run_id", None)
        data.pop("evaluation_job_id", None)
        data.pop("semantic_run_id", None)
        data.pop("semantic_job_id", None)
    else:
        data.pop("job_id", None)
        data.pop("compiler_unit_id", None)
    data["stage"] = value.stage.value
    data["status"] = value.status.value
    data["metadata"] = dict(value.metadata)
    return data


def model_call_from_row(row: Mapping[str, Any], scope: DataScope) -> ModelCallTrace:
    return ModelCallTrace.model_validate({**dict(row), "data_scope": scope.value})


def retrieval_batch_to_row(value: RetrievalBatchTrace) -> Row:
    data = value.model_dump(mode="python")
    data["stage"] = value.stage.value
    data["status"] = value.status.value
    data["metadata"] = dict(value.metadata)
    return data


def retrieval_batch_from_row(row: Mapping[str, Any]) -> RetrievalBatchTrace:
    return RetrievalBatchTrace.model_validate(dict(row))


def retrieval_trace_to_row(value: RetrievalTrace) -> Row:
    data: Row = {
        "retrieval_trace_id": value.retrieval_trace_id,
        "retrieval_batch_id": value.retrieval_batch_id,
        "query_id": value.query_id,
        "run_id": value.run_id,
        "method": value.method,
        "retriever_name": value.retriever_name,
        "index_name": value.index_name,
        "chunking_strategy": value.chunking_strategy,
        "rank": value.rank,
        "retrieved_id": value.retrieved_id,
        "retrieval_document_id": value.retrieval_document_id,
        "retrieval_window_id": value.retrieval_window_id,
        "retrieved_type": value.retrieved_type.value,
        "created_at": value.created_at,
        "metadata": dict(value.metadata),
    }
    score = value.score
    data.update(
        {
            "score_type": score.score_type if score else None,
            "score_raw_value": score.raw_value if score else None,
            "score_normalized_value": score.normalized_value if score else None,
            "score_direction": score.direction.value if score else None,
            "normalization_method": score.normalization_method if score else None,
            "normalization_version": score.normalization_version if score else None,
            "score_metadata": dict(score.metadata) if score else {},
        }
    )
    return data


def retrieval_trace_from_row(row: Mapping[str, Any]) -> RetrievalTrace:
    score = None
    if row.get("score_type") is not None:
        score = RankingScore(
            score_type=row["score_type"],
            raw_value=row["score_raw_value"],
            normalized_value=row["score_normalized_value"],
            direction=row["score_direction"],
            normalization_method=row["normalization_method"],
            normalization_version=row["normalization_version"],
            metadata=dict(row["score_metadata"]),
        )
    return RetrievalTrace(
        retrieval_trace_id=row["retrieval_trace_id"],
        retrieval_batch_id=row["retrieval_batch_id"],
        query_id=row["query_id"],
        run_id=row["run_id"],
        method=row["method"],
        retriever_name=row["retriever_name"],
        index_name=row["index_name"],
        chunking_strategy=row["chunking_strategy"],
        rank=row["rank"],
        retrieved_id=row["retrieved_id"],
        retrieval_document_id=row.get("retrieval_document_id"),
        retrieval_window_id=row.get("retrieval_window_id"),
        retrieved_type=row["retrieved_type"],
        score=score,
        created_at=row["created_at"],
        metadata=dict(row["metadata"]),
    )


def retrieval_query_embedding_to_row(
    value: RetrievalQueryEmbeddingRef,
    *,
    batch: RetrievalBatchTrace,
) -> Row:
    if batch.query_embedding_collection_id is None:
        raise ValueError("query embedding references require a query embedding collection")
    return {
        "retrieval_batch_id": value.retrieval_batch_id,
        "embedding_id": value.embedding_id,
        "query_embedding_collection_id": batch.query_embedding_collection_id,
        "run_id": batch.run_id,
        "snapshot_id": batch.snapshot_id,
        "query_id": batch.query_id,
        "role": value.role.value,
        "ordinal": value.ordinal,
        "used_for_search": value.used_for_search,
        "metadata": dict(value.metadata),
    }


def retrieval_query_embedding_from_row(row: Mapping[str, Any]) -> RetrievalQueryEmbeddingRef:
    return RetrievalQueryEmbeddingRef.model_validate(
        {
            "retrieval_batch_id": row["retrieval_batch_id"],
            "embedding_id": row["embedding_id"],
            "role": row["role"],
            "ordinal": row["ordinal"],
            "used_for_search": row["used_for_search"],
            "metadata": dict(row["metadata"]),
        }
    )


def retrieval_component_to_row(
    value: RetrievalTraceComponent,
    *,
    trace: RetrievalTrace,
    snapshot_id: str,
) -> Row:
    score = value.score
    return {
        "retrieval_trace_id": value.retrieval_trace_id,
        "retrieval_document_id": value.retrieval_document_id,
        "embedding_collection_id": value.embedding_collection_id,
        "snapshot_id": snapshot_id,
        "run_id": trace.run_id,
        "query_id": trace.query_id,
        "representation": value.representation.value,
        "ordinal": value.ordinal,
        "score_type": score.score_type,
        "score_raw_value": score.raw_value,
        "score_normalized_value": score.normalized_value,
        "score_direction": score.direction.value,
        "normalization_method": score.normalization_method,
        "normalization_version": score.normalization_version,
        "score_metadata": dict(score.metadata),
        "is_winner": value.is_winner,
        "metadata": dict(value.metadata),
    }


def retrieval_component_from_row(row: Mapping[str, Any]) -> RetrievalTraceComponent:
    return RetrievalTraceComponent.model_validate(
        {
            "retrieval_trace_id": row["retrieval_trace_id"],
            "retrieval_document_id": row["retrieval_document_id"],
            "embedding_collection_id": row["embedding_collection_id"],
            "representation": row["representation"],
            "ordinal": row["ordinal"],
            "score": {
                "score_type": row["score_type"],
                "raw_value": row["score_raw_value"],
                "normalized_value": row["score_normalized_value"],
                "direction": row["score_direction"],
                "normalization_method": row["normalization_method"],
                "normalization_version": row["normalization_version"],
                "metadata": dict(row["score_metadata"]),
            },
            "is_winner": row["is_winner"],
            "metadata": dict(row["metadata"]),
        }
    )


def final_answer_to_rows(value: FinalAnswerRecord) -> dict[str, list[Row]]:
    parent = value.model_dump(mode="python")
    evidence_ids = parent.pop("evidence_ids_in_prompt")
    image_ids = parent.pop("image_ids_in_prompt")
    parent["metadata"] = dict(value.metadata)
    return {
        "answer": [parent],
        "evidence": [
            {"answer_id": value.answer_id, "evidence_id": item, "ordinal": index}
            for index, item in enumerate(evidence_ids)
        ],
        "images": [
            {
                "answer_id": value.answer_id,
                "snapshot_id": value.snapshot_id,
                "image_id": item,
                "ordinal": index,
            }
            for index, item in enumerate(image_ids)
        ],
    }


def final_answer_from_parts(
    parent: Mapping[str, Any], evidence_ids: list[str], image_ids: list[str]
) -> FinalAnswerRecord:
    return FinalAnswerRecord.model_validate(
        {
            **dict(parent),
            "evidence_ids_in_prompt": evidence_ids,
            "image_ids_in_prompt": image_ids,
        }
    )


def retrieval_evaluation_to_rows(value: RetrievalEvaluation) -> dict[str, list[Row]]:
    parent = value.model_dump(mode="python")
    matched = parent.pop("matched_gold_evidence_ids")
    parent["metadata"] = dict(value.metadata)
    return {
        "evaluation": [parent],
        "matches": [
            {
                "retrieval_evaluation_id": value.retrieval_evaluation_id,
                "gold_evidence_id": item,
                "ordinal": index,
            }
            for index, item in enumerate(matched)
        ],
    }


def metric_score_to_row(value: MetricScore) -> Row:
    data = value.model_dump(mode="python")
    data["metric_family"] = value.metric_family.value
    data["metadata"] = dict(value.metadata)
    return data


def metric_score_from_row(row: Mapping[str, Any]) -> MetricScore:
    return MetricScore.model_validate(dict(row))


def judge_score_to_row(value: JudgeScore) -> Row:
    data = value.model_dump(mode="python")
    data["bad_case_category"] = value.bad_case_category.value
    data["verdict"] = value.verdict.value if value.verdict is not None else None
    data["metadata"] = dict(value.metadata)
    return data


def judge_score_from_row(row: Mapping[str, Any]) -> JudgeScore:
    return JudgeScore.model_validate(dict(row))


def bad_case_label_to_row(value: BadCaseLabel) -> Row:
    data = value.model_dump(mode="python")
    data["category"] = value.category.value
    data["metadata"] = dict(value.metadata)
    return data


def bad_case_label_from_row(row: Mapping[str, Any]) -> BadCaseLabel:
    return BadCaseLabel.model_validate(dict(row))
