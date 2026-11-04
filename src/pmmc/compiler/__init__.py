"""Offline PMMC memory compiler foundation.

Runtime runners are exposed lazily so low-level database modules can import
compiler identity helpers without pulling the database-backed orchestration
layer back into a partially initialized module.
"""

from importlib import import_module
from typing import Any

from pmmc.compiler.config import (
    OPERATOR_SET_V1,
    OPERATOR_SET_V2,
    OPERATOR_SET_V4,
    CompilerFoundationConfig,
    compiler_config_hash,
    load_compiler_config,
)
from pmmc.compiler.ids import (
    canonical_content_hash,
    compilation_unit_identity,
    feedback_content_hash,
    memory_program_content_hash,
    ordered_bindings_hash,
    qbank_decision_content_hash,
    question_content_hash,
    stable_compiler_id,
)
from pmmc.compiler.planner_config import (
    PlannerConfig,
    load_planner_config,
    planner_config_hash,
)
from pmmc.compiler.planner_prompt import (
    PLANNER_PARSER_VERSION,
    PLANNER_PROMPT,
    PLANNER_PROMPT_VERSION,
    PLANNER_RENDERER_VERSION,
    PlannerEvidenceView,
    planner_output_parser,
    planner_prompt_sha256,
    planner_response_schema_sha256,
    render_planner_request,
)
from pmmc.compiler.preparation import (
    CompilationUnitPreparationManifest,
    CompilationUnitPreparationService,
    PreparedCompilationUnitDescriptor,
    PreparedConversationUnits,
    TokenCounterAudit,
    unit_preparation_identity,
)
from pmmc.compiler.program_export import verify_program_export
from pmmc.compiler.program_pilot import (
    build_program_pilot_bundle,
    compare_program_pilots,
    decide_program_pilot,
    verify_program_freeze,
    verify_program_pilot_archive,
    verify_program_pilot_bundle,
)
from pmmc.compiler.questioner import (
    GatedQuestionCandidate,
    ParsedQuestionerBatch,
    QuestionerCandidateGate,
    normalize_question_text,
    parse_questioner_output,
    questioner_output_parser,
)
from pmmc.compiler.questioner_config import (
    DEFAULT_RELATION_ORDER,
    QuestionerConfig,
    load_questioner_config,
    questioner_config_hash,
)
from pmmc.compiler.questioner_freeze import (
    questioner_freeze_identity,
    verify_questioner_freeze_compatibility,
)
from pmmc.compiler.questioner_prompt import (
    QUESTIONER_PROMPT,
    QUESTIONER_PROMPT_VERSION,
    QUESTIONER_RENDERER_VERSION,
    questioner_prompt_sha256,
    questioner_request_contract_hash,
    render_questioner_request,
)
from pmmc.compiler.units import (
    BuiltCompilationUnit,
    QuestionerInputTooLarge,
    SessionAnchoredUnitBuilder,
)

_LAZY_EXPORTS = {
    "assess_program_dry_run": (
        "pmmc.compiler.dry_run",
        "assess_program_dry_run",
    ),
    "dry_run_outcome_status": (
        "pmmc.compiler.dry_run",
        "dry_run_outcome_status",
    ),
    "PlannerProgramPolicy": (
        "pmmc.compiler.planner",
        "PlannerProgramPolicy",
    ),
    "materialize_planner_program": (
        "pmmc.compiler.planner",
        "materialize_planner_program",
    ),
    "planner_terminal_outcome": (
        "pmmc.compiler.planner",
        "planner_terminal_outcome",
    ),
    "MemoryOperatorIndexBuilder": (
        "pmmc.compiler.memory_index_runner",
        "MemoryOperatorIndexBuilder",
    ),
    "ProgramQueryEmbeddingBuilder": (
        "pmmc.compiler.memory_index_runner",
        "ProgramQueryEmbeddingBuilder",
    ),
    "ProgramQueryVectorSet": (
        "pmmc.compiler.memory_index_runner",
        "ProgramQueryVectorSet",
    ),
    "load_program_query_vectors": (
        "pmmc.compiler.memory_index_runner",
        "load_program_query_vectors",
    ),
    "PlannerRunner": ("pmmc.compiler.planner_runner", "PlannerRunner"),
    "PlannerRunResult": ("pmmc.compiler.planner_runner", "PlannerRunResult"),
    "PostgresPlannerInputProvider": (
        "pmmc.compiler.planner_runner",
        "PostgresPlannerInputProvider",
    ),
    "ProgramDryRunResult": (
        "pmmc.compiler.program_runner",
        "ProgramDryRunResult",
    ),
    "ProgramDryRunRunner": (
        "pmmc.compiler.program_runner",
        "ProgramDryRunRunner",
    ),
    "ProgramWorkflowPhase": (
        "pmmc.compiler.program_workflow",
        "ProgramWorkflowPhase",
    ),
    "ProgramWorkflowResult": (
        "pmmc.compiler.program_workflow",
        "ProgramWorkflowResult",
    ),
    "run_program_workflow": (
        "pmmc.compiler.program_workflow",
        "run_program_workflow",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "OPERATOR_SET_V1",
    "OPERATOR_SET_V2",
    "OPERATOR_SET_V4",
    "CompilerFoundationConfig",
    "DEFAULT_RELATION_ORDER",
    "QuestionerConfig",
    "QuestionerInputTooLarge",
    "SessionAnchoredUnitBuilder",
    "BuiltCompilationUnit",
    "CompilationUnitPreparationManifest",
    "CompilationUnitPreparationService",
    "GatedQuestionCandidate",
    "MemoryOperatorIndexBuilder",
    "ParsedQuestionerBatch",
    "PlannerConfig",
    "PlannerEvidenceView",
    "PlannerProgramPolicy",
    "PlannerRunResult",
    "PlannerRunner",
    "ProgramDryRunResult",
    "ProgramDryRunRunner",
    "ProgramQueryEmbeddingBuilder",
    "ProgramQueryVectorSet",
    "ProgramWorkflowPhase",
    "ProgramWorkflowResult",
    "PostgresPlannerInputProvider",
    "QuestionerCandidateGate",
    "PreparedCompilationUnitDescriptor",
    "PreparedConversationUnits",
    "QUESTIONER_PROMPT",
    "QUESTIONER_PROMPT_VERSION",
    "QUESTIONER_RENDERER_VERSION",
    "PLANNER_PARSER_VERSION",
    "PLANNER_PROMPT",
    "PLANNER_PROMPT_VERSION",
    "PLANNER_RENDERER_VERSION",
    "assess_program_dry_run",
    "canonical_content_hash",
    "compilation_unit_identity",
    "feedback_content_hash",
    "compiler_config_hash",
    "load_compiler_config",
    "load_questioner_config",
    "load_planner_config",
    "load_program_query_vectors",
    "materialize_planner_program",
    "memory_program_content_hash",
    "ordered_bindings_hash",
    "question_content_hash",
    "qbank_decision_content_hash",
    "questioner_config_hash",
    "questioner_freeze_identity",
    "questioner_prompt_sha256",
    "questioner_request_contract_hash",
    "normalize_question_text",
    "parse_questioner_output",
    "questioner_output_parser",
    "render_questioner_request",
    "render_planner_request",
    "run_program_workflow",
    "build_program_pilot_bundle",
    "compare_program_pilots",
    "decide_program_pilot",
    "planner_config_hash",
    "planner_output_parser",
    "planner_prompt_sha256",
    "planner_response_schema_sha256",
    "planner_terminal_outcome",
    "dry_run_outcome_status",
    "stable_compiler_id",
    "TokenCounterAudit",
    "unit_preparation_identity",
    "verify_program_export",
    "verify_program_freeze",
    "verify_program_pilot_archive",
    "verify_program_pilot_bundle",
    "verify_questioner_freeze_compatibility",
]
