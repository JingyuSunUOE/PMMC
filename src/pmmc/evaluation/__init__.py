"""Evaluation runners, metrics, judges, and offline analysis helpers."""

from typing import TYPE_CHECKING, Any

from pmmc.evaluation.config import (
    BaselineEvaluationConfig,
    evaluation_prompt_sha256,
    load_evaluation_config,
    load_judge_prompt,
)
from pmmc.evaluation.evidence import (
    EvidenceMetricBundle,
    RankedEvidenceCandidate,
    resolve_gold_evidence,
    score_evidence,
)
from pmmc.evaluation.judge import (
    BaselineJudgeService,
    JudgeVerdictPayload,
    parse_judge_output,
)
from pmmc.evaluation.metrics import (
    ANSWER_METRIC_VERSIONS,
    AnswerMetricBundle,
    normalize_answer,
    score_answer,
    trim_answer,
)
from pmmc.evaluation.qbank_coverage import (
    QBankCandidateEvidence,
    QBankCoverageBundle,
    score_qbank_coverage,
)
from pmmc.evaluation.qbank_export import (
    QBankCoverageExporter,
    QBankCoverageExportResult,
    verify_qbank_coverage_export,
)
from pmmc.evaluation.qbank_semantic import (
    QBankSemanticJudgeService,
    QBankSemanticMatchConfig,
    build_qbank_semantic_request,
    derive_qbank_semantic_metrics,
    load_qbank_semantic_config,
    parse_qbank_semantic_output,
    select_semantic_candidate_union,
)

if TYPE_CHECKING:
    from pmmc.evaluation.qbank_semantic_workflow import QBankSemanticWorkflowResult


def __getattr__(name: str) -> Any:
    if name in {
        "QBankSemanticWorkflowResult",
        "run_qbank_semantic_match",
        "verify_qbank_semantic_match_run",
    }:
        from pmmc.evaluation import qbank_semantic_workflow

        return getattr(qbank_semantic_workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ANSWER_METRIC_VERSIONS",
    "AnswerMetricBundle",
    "BaselineEvaluationConfig",
    "BaselineJudgeService",
    "EvidenceMetricBundle",
    "JudgeVerdictPayload",
    "RankedEvidenceCandidate",
    "QBankCandidateEvidence",
    "QBankCoverageBundle",
    "QBankCoverageExporter",
    "QBankCoverageExportResult",
    "QBankSemanticJudgeService",
    "QBankSemanticMatchConfig",
    "QBankSemanticWorkflowResult",
    "build_qbank_semantic_request",
    "derive_qbank_semantic_metrics",
    "evaluation_prompt_sha256",
    "load_evaluation_config",
    "load_judge_prompt",
    "load_qbank_semantic_config",
    "normalize_answer",
    "parse_judge_output",
    "parse_qbank_semantic_output",
    "resolve_gold_evidence",
    "run_qbank_semantic_match",
    "score_answer",
    "score_evidence",
    "score_qbank_coverage",
    "select_semantic_candidate_union",
    "trim_answer",
    "verify_qbank_coverage_export",
    "verify_qbank_semantic_match_run",
]
