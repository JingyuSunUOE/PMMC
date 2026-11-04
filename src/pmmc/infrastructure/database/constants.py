"""Stable Postgres names and persistence states."""

from __future__ import annotations

from enum import StrEnum

RUNTIME_SCHEMA = "pmmc_runtime"
EVALUATION_SCHEMA = "pmmc_evaluation"

OWNER_ROLE = "pmmc_owner"
RUNTIME_ROLE = "pmmc_runtime"
EVALUATION_ROLE = "pmmc_evaluator"

# These tuples are the frozen TEXT + CHECK contract for the first persistence
# migration. Extending a public schema enum requires a matching Alembic revision.
DATASET_VALUES = ("memlens", "memgallery", "unknown")
ASSET_TYPE_VALUES = (
    "raw_image",
    "resized_image",
    "prompt",
    "model_response",
    "trace_export",
    "embedding_dump",
    "dataset_snapshot",
    "other",
)
SPEAKER_VALUES = ("user", "assistant", "system", "unknown")
MODALITY_VALUES = (
    "text",
    "caption",
    "image",
    "raw_image",
    "text_embedding",
    "image_embedding",
    "joint_mm_embedding",
    "embedding",
    "multimodal",
)
QUESTION_TYPE_VALUES = (
    "textual_fact",
    "visual_detail",
    "image_text_binding",
    "visual_search",
    "multi_hop",
    "temporal",
    "knowledge_update",
    "conflict_detection",
    "answer_refusal",
    "personal_preference",
    "test_time_learning",
    "visual_reasoning",
    "multi_entity_reasoning",
    "unknown",
)
QUESTION_REASONING_TYPE_VALUES = (
    "single_hop",
    "multi_hop",
    "temporal",
    "update",
    "conflict",
    "refusal",
)
COMPILATION_UNIT_RELATION_TYPE_VALUES = (
    "adjacent_session",
    "shared_entity",
    "shared_image",
    "profile_preference",
    "temporal_neighbor",
)
QUESTIONER_PARSE_STATUS_VALUES = ("parsed", "repaired", "partial", "failed")
CANDIDATE_GATE_STATUS_VALUES = ("eligible_for_planning", "rejected")
CANDIDATE_REJECTION_CODE_VALUES = (
    "invalid_schema",
    "unsupported_question_type",
    "future_or_foreign_reference",
    "missing_expected_answer",
    "missing_expected_evidence",
    "modality_mismatch",
    "visual_evidence_missing",
    "multi_hop_evidence_missing",
    "duplicate",
    "budget_exceeded",
    "internal_identifier_leak",
    "system_metadata_reference",
    "assistant_follow_up",
    "historical_question_replay",
    "generic_external_question",
    "placeholder_expected_answer",
    "opportunity_mismatch",
    "taxonomy_invariant_failed",
    "reasoning_invariant_failed",
    "evidence_not_entailed",
    "semantic_revision_required",
    "semantic_verification_failed",
    "revision_exhausted",
    "question_answer_closure_failed",
    "answer_type_mismatch",
    "question_as_answer",
    "ai_disclaimer",
    "incomplete_answer",
    "source_unnatural",
)
QUESTION_OPPORTUNITY_TYPE_VALUES = (
    "local_fact",
    "visual_detail",
    "image_text_binding",
    "visual_search",
    "temporal_pair",
    "update_pair",
    "conflict_pair",
    "preference",
    "multi_evidence",
    "refusal_boundary",
)
QUESTIONER_CANDIDATE_ORIGIN_VALUES = ("generated", "revised")
QUESTIONER_VERIFICATION_VERDICT_VALUES = ("pass", "revise", "reject")
QUESTIONER_VERIFICATION_REASON_VALUES = (
    "not_user_facing",
    "not_future_plausible",
    "not_answerable",
    "answer_unsupported",
    "evidence_insufficient",
    "taxonomy_mismatch",
    "modality_mismatch",
    "visual_not_needed",
    "multi_hop_not_genuine",
    "too_generic",
    "historical_replay",
    "internal_metadata",
    "unsafe_leakage",
)
QBANK_MATCH_CHANNEL_VALUES = (
    "lexical",
    "text_semantic",
    "multimodal_semantic",
    "hybrid_rrf",
)
QBANK_COVERAGE_STATUS_VALUES = ("created", "running", "completed", "failed")
COVERAGE_GATE_MODE_VALUES = ("report_only", "enforce")
QBANK_COVERAGE_LABEL_VALUES = (
    "questioner_coverage_failure",
    "matching_failure",
    "gold_unresolvable",
)
DATA_SCOPE_VALUES = ("runtime", "evaluation")
RUN_STATUS_VALUES = ("created", "running", "completed", "failed", "cancelled")
JOB_STATUS_VALUES = ("pending", "running", "succeeded", "failed", "cancelled")
ARTIFACT_STATE_VALUES = ("pending", "ready", "failed")
TRACE_STATUS_VALUES = ("success", "failed", "skipped", "partial")
APP_STAGE_VALUES = (
    "compile",
    "questioner",
    "planner",
    "doubter",
    "match",
    "retrieve",
    "execute",
    "answer",
    "judge",
    "analyze",
    "baseline",
    "embed",
)
EVIDENCE_TYPE_VALUES = (
    "memory",
    "text",
    "caption",
    "image",
    "raw_image",
    "image_text_pair",
    "session",
    "query",
    "unknown",
)
EVIDENCE_SUFFICIENCY_VALUES = ("sufficient", "partial", "insufficient")
SCORE_DIRECTION_VALUES = ("higher_is_better", "lower_is_better")
BAD_CASE_CATEGORY_VALUES = (
    "retrieval_top50_miss",
    "retrieval_top5_miss",
    "context_budget_exclusion",
    "visual_evidence_miss",
    "text_evidence_miss",
    "evidence_found_answer_wrong",
    "answer_format_mismatch",
    "judge_metric_disagreement",
    "model_call_failure",
    "gold_unresolvable",
    "questioner_coverage_failure",
    "matching_failure",
    "planner_failure",
    "doubter_failure",
    "evidence_insufficiency",
    "answer_generation_failure",
    "update_conflict_failure",
    "none",
    "unknown",
)
BASELINE_SUITE_STATUS_VALUES = ("created", "running", "ready", "failed", "cancelled")
EVALUATION_RUN_STATUS_VALUES = ("created", "running", "completed", "failed", "cancelled")
EVALUATION_STAGE_VALUES = ("deterministic", "judge", "analyze")
METRIC_FAMILY_VALUES = ("answer", "retrieval", "context", "efficiency")
EVIDENCE_GRANULARITY_VALUES = ("generic", "session", "memory", "image", "text")
EVIDENCE_MEMBER_TYPE_VALUES = ("evidence", "session", "memory", "image")
JUDGE_VERDICT_VALUES = ("incorrect", "partially_correct", "correct")
DISTANCE_METRIC_VALUES = ("cosine", "inner_product", "l2")
EMBEDDING_SOURCE_TYPE_VALUES = (
    "memory_text",
    "memory_caption",
    "memory_image",
    "memory_image_text_pair",
    "qbank_question",
    "qbank_paraphrase",
    "query",
    "retrieval_document_text_caption",
    "retrieval_document_text",
    "retrieval_document_image",
    "retrieval_document_multimodal",
    "retrieval_document_contrastive",
)
EMBEDDING_COLLECTION_STATUS_VALUES = ("building", "ready", "failed")
BASELINE_REPRESENTATION_VALUES = ("text_caption", "text", "image", "multimodal")
CORPUS_STATUS_VALUES = ("building", "ready", "failed")
MANAGED_RUNTIME_STATUS_VALUES = (
    "planned",
    "starting",
    "running",
    "completed",
    "failed",
    "cancelled",
)
MANAGED_SERVICE_STATUS_VALUES = (
    "planned",
    "starting",
    "ready",
    "stopping",
    "stopped",
    "failed",
)
MANAGED_BACKEND_VALUES = (
    "vllm_chat",
    "vllm_embedding",
    "vllm_multimodal_embedding",
    "litellm_proxy",
    "transformers_contrastive",
)
MODEL_ROLE_VALUES = (
    "answer",
    "compiler",
    "judge",
    "text_embedding",
    "multimodal_embedding",
    "contrastive_embedding",
    "reranker",
)
CALIBRATION_STATUS_VALUES = ("ready", "failed")
COMPILATION_TARGET_STAGE_VALUES = ("questions", "programs", "verified_qbank")
COMPILATION_STAGE_VALUES = ("questioner", "planner", "dry_run", "doubter", "decision")
COMPILATION_UNIT_MEMORY_ROLE_VALUES = ("anchor", "support")
QBANK_EVIDENCE_ROLE_VALUES = ("source", "expected")
OPERATOR_CONTROL_ACTION_VALUES = ("continue", "fallback", "abstain")
OPERATOR_EXECUTION_PURPOSE_VALUES = ("compiler_dry_run", "query_time")


def sql_text_values(values: tuple[str, ...]) -> str:
    """Render trusted persistence constants for SQL CHECK expressions."""

    return "(" + ", ".join(f"'{value}'" for value in values) + ")"


def sql_text_array(values: tuple[str, ...]) -> str:
    """Render a trusted PostgreSQL text array for a CHECK expression."""

    return "ARRAY[" + ", ".join(f"'{value}'" for value in values) + "]::text[]"


class DatabaseRole(StrEnum):
    """Configured database identities with intentionally different privileges."""

    RUNTIME = "runtime"
    ONLINE = "online"
    EVALUATOR = "evaluator"
    ADMIN = "admin"


class SnapshotImportState(StrEnum):
    """Lifecycle of one immutable canonical snapshot in Postgres."""

    IMPORTING = "importing"
    READY = "ready"
    FAILED = "failed"
