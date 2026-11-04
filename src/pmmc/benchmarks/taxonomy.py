"""Explicit source-category mappings and PMMC evaluation policy."""

from __future__ import annotations

from pmmc.schemas import (
    DEFAULT_EVALUATION_POLICY_VERSION,
    DatasetName,
    EvaluationEligibility,
    QuestionType,
)

MEMLENS_TYPES = {
    "information_extraction": QuestionType.TEXTUAL_FACT,
    "multi_session_reasoning": QuestionType.MULTI_HOP,
    "temporal_reasoning": QuestionType.TEMPORAL,
    "knowledge_update": QuestionType.KNOWLEDGE_UPDATE,
    "answer_refusal": QuestionType.ANSWER_REFUSAL,
}

MEMGALLERY_TYPES = {
    "FR": QuestionType.TEXTUAL_FACT,
    "VS": QuestionType.VISUAL_SEARCH,
    "TTL": QuestionType.TEST_TIME_LEARNING,
    "TR": QuestionType.TEMPORAL,
    "VR": QuestionType.VISUAL_REASONING,
    "MR": QuestionType.MULTI_ENTITY_REASONING,
    "KR": QuestionType.KNOWLEDGE_UPDATE,
    "CD": QuestionType.CONFLICT_DETECTION,
    "AR": QuestionType.ANSWER_REFUSAL,
}

EXCLUDED_SOURCE_CATEGORIES = {
    DatasetName.MEMLENS: {"answer_refusal"},
    DatasetName.MEMGALLERY: {"AR", "CD"},
}


def map_question_type(dataset: DatasetName, source_category: str) -> QuestionType:
    """Map a source label to the fixed PMMC taxonomy or fail closed."""

    if dataset is DatasetName.UNKNOWN:
        raise ValueError("unknown has no benchmark taxonomy")
    mapping = MEMLENS_TYPES if dataset is DatasetName.MEMLENS else MEMGALLERY_TYPES
    try:
        return mapping[source_category]
    except KeyError as exc:
        raise ValueError(f"unsupported {dataset.value} source category: {source_category}") from exc


def evaluation_eligibility(
    dataset: DatasetName,
    source_category: str,
) -> EvaluationEligibility:
    """Apply the fixed AAAI 2027 evaluation eligibility policy."""

    if dataset is DatasetName.UNKNOWN:
        raise ValueError("unknown has no evaluation eligibility policy")
    if source_category in EXCLUDED_SOURCE_CATEGORIES[dataset]:
        return EvaluationEligibility(
            eligible=False,
            policy_version=DEFAULT_EVALUATION_POLICY_VERSION,
            exclusion_reason=f"source category {source_category} is outside PMMC scope",
        )
    return EvaluationEligibility(
        eligible=True,
        policy_version=DEFAULT_EVALUATION_POLICY_VERSION,
    )
