"""Planner v1 materialization and deterministic program-policy validation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pmmc.compiler.execution import budget_program_steps
from pmmc.compiler.ids import (
    canonical_content_hash,
    memory_program_content_hash,
    stable_compiler_id,
)
from pmmc.compiler.planner_config import PlannerConfig
from pmmc.memory.evidence_roles import EVIDENCE_ROLE_POLICY, required_evidence_roles
from pmmc.memory.operators import OperatorRegistry
from pmmc.schemas import (
    DoubterDirective,
    DoubterDirectiveCode,
    MemoryProgram,
    PlannerAbstainStepDraft,
    PlannerCandidateOutcome,
    PlannerCapability,
    PlannerEmbeddingStepDraft,
    PlannerExpansionCapability,
    PlannerFallbackStepDraft,
    PlannerImageStepDraft,
    PlannerKeywordStepDraft,
    PlannerKeywordStepDraftV11,
    PlannerOutcomeStatus,
    PlannerProgramDraft,
    PlannerProgramDraftV11,
    PlannerRequiredModality,
    PlannerRequirementContract,
    PlannerRetrievalChannel,
    PlannerStrategyDraftV12,
    PlannerTemporalStepDraft,
    PlannerVerifyStepDraft,
    ProgramPolicyIssue,
    ProgramPolicyResult,
    ProgramPolicySeverity,
    ProgramStep,
    QuestionCandidate,
    QuestionReasoningType,
    QuestionType,
)
from pmmc.schemas.base import utc_now

_OUTPUT_TYPES = {
    "retrieve_text_by_keyword": "memory_ids",
    "retrieve_text_by_embedding": "memory_ids",
    "retrieve_image_by_visual_embedding": "image_ids",
    "retrieve_image_text_pair": "image_text_pairs",
    "retrieve_context_around_image": "memory_ids",
    "expand_temporal_neighbors": "memory_ids",
    "inspect_raw_image": "image_ids",
    "verify_evidence_sufficiency": "verification",
    "fallback_to_multimodal_rag": "control",
    "abstain_if_no_evidence": "control",
}
_IMAGE_INPUT_OPERATORS = {
    "retrieve_image_text_pair",
    "retrieve_context_around_image",
    "inspect_raw_image",
}
_CONTROL_OPERATORS = {"fallback_to_multimodal_rag", "abstain_if_no_evidence"}
_VISUAL_TYPES = {
    QuestionType.VISUAL_DETAIL,
    QuestionType.IMAGE_TEXT_BINDING,
    QuestionType.VISUAL_SEARCH,
}
_TEMPORAL_TYPES = {
    QuestionType.TEMPORAL,
    QuestionType.KNOWLEDGE_UPDATE,
    QuestionType.CONFLICT_DETECTION,
}


class PlannerStrategyPolicyError(ValueError):
    """Candidate-scoped strategy violation that must not abort the Planner run."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        diagnostics: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostics = dict(diagnostics or {})

    def safe_diagnostics(self) -> dict[str, object]:
        return {"error_code": self.code, **self.diagnostics}


_CAPABILITY_OUTPUTS: dict[str, set[PlannerCapability]] = {
    "retrieve_text_by_keyword": {
        PlannerCapability.TEXT,
        PlannerCapability.EVIDENCE_GROUPS,
    },
    "retrieve_text_by_embedding": {
        PlannerCapability.TEXT,
        PlannerCapability.EVIDENCE_GROUPS,
    },
    "retrieve_image_by_visual_embedding": {
        PlannerCapability.IMAGE,
        PlannerCapability.EVIDENCE_GROUPS,
    },
    "retrieve_image_text_pair": {
        PlannerCapability.IMAGE,
        PlannerCapability.TEXT,
        PlannerCapability.IMAGE_TEXT,
        PlannerCapability.EVIDENCE_GROUPS,
    },
    "retrieve_context_around_image": {
        PlannerCapability.IMAGE,
        PlannerCapability.TEXT,
        PlannerCapability.CONTEXT,
        PlannerCapability.EVIDENCE_GROUPS,
    },
    "expand_temporal_neighbors": {
        PlannerCapability.TEXT,
        PlannerCapability.TEMPORAL,
        PlannerCapability.EVIDENCE_GROUPS,
    },
    "inspect_raw_image": {
        PlannerCapability.IMAGE,
        PlannerCapability.RAW_IMAGE,
    },
    "verify_evidence_sufficiency": {PlannerCapability.VERIFICATION},
    "fallback_to_multimodal_rag": {PlannerCapability.CONTROL},
    "abstain_if_no_evidence": {PlannerCapability.CONTROL},
}
_CAPABILITY_INPUTS: dict[str, set[PlannerCapability]] = {
    "retrieve_image_text_pair": {PlannerCapability.IMAGE},
    "retrieve_context_around_image": {PlannerCapability.IMAGE},
    "inspect_raw_image": {PlannerCapability.IMAGE},
    "expand_temporal_neighbors": {PlannerCapability.TEXT},
    "verify_evidence_sufficiency": {PlannerCapability.EVIDENCE_GROUPS},
    "fallback_to_multimodal_rag": {PlannerCapability.VERIFICATION},
}
_KEYWORD_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]{1,}")
_KEYWORD_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "before",
        "could",
        "does",
        "from",
        "have",
        "into",
        "that",
        "their",
        "there",
        "these",
        "they",
        "this",
        "those",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "your",
    }
)


def _canonical_question_terms(question: str, budget: int) -> list[str]:
    """Extract stable FTS terms without accepting provider-supplied literals."""

    values: list[str] = []
    seen: set[str] = set()
    for match in _KEYWORD_TOKEN.finditer(question.casefold()):
        value = match.group(0)
        if value in _KEYWORD_STOP_WORDS or value in seen:
            continue
        seen.add(value)
        values.append(value)
        if len(values) == budget:
            break
    return values or [question.strip()]


def _contains_forbidden_value(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, str):
        return any(item and item in value for item in forbidden)
    if isinstance(value, Mapping):
        return any(_contains_forbidden_value(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_value(item, forbidden) for item in value)
    return False


def _step_args(step: object, *, candidate: QuestionCandidate) -> dict[str, object]:
    if isinstance(step, (PlannerKeywordStepDraft, PlannerKeywordStepDraftV11)):
        term_source = step.term_source
        query_terms = (
            list(step.terms)
            if isinstance(step, PlannerKeywordStepDraft) and step.terms is not None
            else _canonical_question_terms(
                candidate.canonical_question,
                step.term_budget or 8,
            )
        )
        return {
            "top_k": step.top_k,
            "query": " ".join(query_terms),
            "entity_mentions": [],
            **(
                {
                    "term_source": term_source.value,
                    "term_budget": step.term_budget,
                    "server_derived": True,
                }
                if term_source is not None
                else {}
            ),
        }
    if isinstance(step, PlannerEmbeddingStepDraft):
        return {"top_k": step.top_k, "vector_role": step.vector_role}
    if isinstance(step, PlannerImageStepDraft):
        result: dict[str, object] = {"source_step_ids": step.source_step_ids}
        if step.radius is not None:
            result["radius"] = step.radius
        return result
    if isinstance(step, PlannerTemporalStepDraft):
        return {"source_step_ids": step.source_step_ids, "radius": step.radius}
    if isinstance(step, PlannerVerifyStepDraft):
        return {
            "source_step_ids": step.source_step_ids,
            "required_modalities": [item.value for item in step.required_modalities],
            "minimum_evidence_groups": step.minimum_evidence_groups,
        }
    if isinstance(step, PlannerFallbackStepDraft):
        return {
            "source_step_ids": step.source_step_ids,
            "condition": step.condition.value,
            "reason_code": step.reason_code,
        }
    if isinstance(step, PlannerAbstainStepDraft):
        return {
            "source_step_ids": step.source_step_ids,
            "minimum_evidence_groups": step.minimum_evidence_groups,
            "required_modalities": [item.value for item in step.required_modalities],
            "reason_code": step.reason_code,
        }
    raise TypeError("unsupported Planner step draft")


def materialize_planner_program(
    *,
    run_id: str,
    snapshot_id: str,
    compiler_unit_id: str,
    producer_model_call_id: str,
    candidate: QuestionCandidate,
    draft: PlannerProgramDraft | PlannerProgramDraftV11,
    config: PlannerConfig | None = None,
) -> MemoryProgram:
    if candidate.qbank_id == "" or draft.candidate_alias == "":
        raise ValueError("Planner materialization requires stable candidate identities")
    plan_id = stable_compiler_id(
        "memory-program",
        [run_id, compiler_unit_id, candidate.qbank_id, draft.model_dump(mode="json")],
    )
    steps = [
        ProgramStep(
            step_id=step_id,
            op_name=step.op_name,
            args=_step_args(step, candidate=candidate),
            expected_output_type=_OUTPUT_TYPES[step.op_name],
            metadata=(
                {
                    "term_source": step.term_source.value,
                    "term_budget": step.term_budget,
                    "server_derived": True,
                }
                if isinstance(step, (PlannerKeywordStepDraft, PlannerKeywordStepDraftV11))
                and step.term_source is not None
                else {}
            ),
        )
        for step_id, step in enumerate(draft.steps)
    ]
    payload = {
        "plan_id": plan_id,
        "qbank_id": candidate.qbank_id,
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "compiler_unit_id": compiler_unit_id,
        "producer_model_call_id": producer_model_call_id,
        "operator_set_version": (
            config.operator_set_version if config is not None else "pmmc-operators-v2"
        ),
        "round_id": 0,
        "operator_sequence": [item.model_dump(mode="json") for item in steps],
        "expected_evidence_ids": candidate.evidence_ids,
        "fallback_operator": None,
        "required_modalities": [item.value for item in candidate.required_modalities],
        "planner_rationale_short": draft.rationale_summary,
        "planner_confidence": draft.confidence,
        "program_hash": "pending",
        "metadata": {"strategy_code": draft.strategy_code},
    }
    pending = MemoryProgram.model_validate(payload)
    payload["program_hash"] = memory_program_content_hash(pending)
    return MemoryProgram.model_validate(payload)


def _v12_step(
    step_id: int,
    op_name: str,
    args: Mapping[str, object],
    *,
    server_role: str,
) -> ProgramStep:
    return ProgramStep(
        step_id=step_id,
        op_name=op_name,
        args=dict(args),
        expected_output_type=_OUTPUT_TYPES[op_name],
        metadata={
            "server_materialized": True,
            "server_role": server_role,
        },
    )


def materialize_planner_strategy_program(
    *,
    run_id: str,
    snapshot_id: str,
    compiler_unit_id: str,
    producer_model_call_id: str | None,
    candidate: QuestionCandidate,
    requirement: PlannerRequirementContract,
    strategy: PlannerStrategyDraftV12 | None,
    config: PlannerConfig,
    round_id: int = 0,
    revision_directives: Sequence[DoubterDirective] | None = None,
) -> MemoryProgram:
    """Build an operator-v4/v5 program from a non-oracle strategy or control."""

    if not (config.is_v12 or config.is_v13):
        raise ValueError("strategy materialization requires Planner v1.2 or v1.3")
    if requirement.qbank_id != candidate.qbank_id:
        raise ValueError("Planner strategy requirement belongs to another candidate")
    if round_id < 0 or round_id > 3:
        raise ValueError("Planner strategy round must be in [0, 3]")
    directive_values = tuple(revision_directives or ())
    for directive in directive_values:
        if (
            directive.run_id != run_id
            or directive.qbank_id != candidate.qbank_id
            or directive.round_id != round_id
        ):
            raise ValueError("Planner revision directive lineage does not match the target")
    if round_id == 0 and directive_values:
        raise ValueError("Planner round zero cannot receive Doubter directives")
    directive_codes = {item.directive for item in directive_values}
    if config.is_v15 and directive_values:
        raise ValueError("v1.5 revisions must patch the preceding program cumulatively")
    if config.is_v15 and config.planning_mode == "fixed" and not requirement.refusal_only:
        strategy = config.fixed_strategy
    directive_payloads = [
        item.model_dump(mode="json")
        for item in sorted(
            directive_values,
            key=lambda item: item.ordinal,
        )
    ]

    steps: list[ProgramStep] = []
    if requirement.refusal_only:
        if directive_values:
            raise ValueError("refusal programs cannot be revised by Planner directives")
        steps.append(
            _v12_step(
                0,
                "abstain_if_no_evidence",
                {
                    "source_step_ids": [],
                    "minimum_evidence_groups": requirement.minimum_evidence_groups,
                    "required_modalities": [item.value for item in requirement.required_modalities],
                    "reason_code": "questioner_control_candidate",
                },
                server_role="control",
            )
        )
        strategy_payload: object = {"route_class": "control"}
        confidence = 1.0
        strategy_code = "server_control_abstain"
    else:
        if strategy is None or strategy.candidate_alias == "":
            raise PlannerStrategyPolicyError(
                "planner_strategy_missing",
                "answerable Planner program requires one native strategy",
            )
        channels = list(strategy.retrieval_channels)
        expansions = set(strategy.expansion_capabilities)
        required = (
            set()
            if config.is_v15 and config.planning_mode == "fixed"
            else set(requirement.required_capabilities)
        )
        if config.is_v15 and config.memory_raw_image_policy == "forbid":
            if PlannerExpansionCapability.RAW_IMAGE in expansions:
                raise PlannerStrategyPolicyError(
                    "planner_treatment_raw_image_forbidden", "memory raw-image access is disabled"
                )
        if config.is_v13:
            requested: set[PlannerCapability] = {PlannerCapability.EVIDENCE_GROUPS}
            if any(
                value in {PlannerRetrievalChannel.KEYWORD, PlannerRetrievalChannel.TEXT_DENSE}
                for value in channels
            ):
                requested.add(PlannerCapability.TEXT)
            if PlannerRetrievalChannel.VISUAL_DENSE in channels:
                requested.add(PlannerCapability.IMAGE)
            expansion_map = {
                PlannerExpansionCapability.IMAGE_TEXT: PlannerCapability.IMAGE_TEXT,
                PlannerExpansionCapability.CONTEXT: PlannerCapability.CONTEXT,
                PlannerExpansionCapability.TEMPORAL: PlannerCapability.TEMPORAL,
                PlannerExpansionCapability.RAW_IMAGE: PlannerCapability.RAW_IMAGE,
            }
            requested.update(expansion_map[value] for value in expansions)
            allowed = set(requirement.allowed_capabilities)
            if not requested <= allowed:
                forbidden = sorted(value.value for value in requested - allowed)
                raise PlannerStrategyPolicyError(
                    "planner_requested_capability_not_allowed",
                    "Planner requested capabilities outside the requirement allowlist: "
                    + ",".join(forbidden),
                    diagnostics={
                        "requested_capabilities": sorted(value.value for value in requested),
                        "allowed_capabilities": sorted(value.value for value in allowed),
                        "disallowed_capabilities": forbidden,
                    },
                )
            directive_capability = {
                DoubterDirectiveCode.ADD_TEXT_RETRIEVAL: PlannerCapability.TEXT,
                DoubterDirectiveCode.ADD_VISUAL_RETRIEVAL: PlannerCapability.IMAGE,
                DoubterDirectiveCode.ADD_IMAGE_TEXT_PAIR: PlannerCapability.IMAGE_TEXT,
                DoubterDirectiveCode.ADD_CONTEXT: PlannerCapability.CONTEXT,
                DoubterDirectiveCode.ADD_TEMPORAL: PlannerCapability.TEMPORAL,
                DoubterDirectiveCode.ADD_RAW_IMAGE_INSPECTION: PlannerCapability.RAW_IMAGE,
            }
            requested.update(
                capability
                for directive in directive_codes
                if (capability := directive_capability.get(directive)) is not None
            )
            if not requested <= allowed:
                forbidden = sorted(value.value for value in requested - allowed)
                raise PlannerStrategyPolicyError(
                    "planner_revision_capability_not_allowed",
                    "Planner revision requested capabilities outside the requirement "
                    "allowlist: " + ",".join(forbidden),
                    diagnostics={
                        "requested_capabilities": sorted(value.value for value in requested),
                        "allowed_capabilities": sorted(value.value for value in allowed),
                        "disallowed_capabilities": forbidden,
                    },
                )
        if (
            DoubterDirectiveCode.ADD_VISUAL_RETRIEVAL in directive_codes
            and PlannerRetrievalChannel.VISUAL_DENSE not in channels
        ):
            channels.append(PlannerRetrievalChannel.VISUAL_DENSE)
        if DoubterDirectiveCode.ADD_TEXT_RETRIEVAL in directive_codes and not any(
            value
            in {
                PlannerRetrievalChannel.KEYWORD,
                PlannerRetrievalChannel.TEXT_DENSE,
            }
            for value in channels
        ):
            channels.append(PlannerRetrievalChannel.TEXT_DENSE)
        if PlannerCapability.IMAGE in required and (
            PlannerRetrievalChannel.VISUAL_DENSE not in channels
        ):
            channels.append(PlannerRetrievalChannel.VISUAL_DENSE)
        if PlannerCapability.TEXT in required and not any(
            value
            in {
                PlannerRetrievalChannel.KEYWORD,
                PlannerRetrievalChannel.TEXT_DENSE,
            }
            for value in channels
        ):
            channels.append(PlannerRetrievalChannel.TEXT_DENSE)
        requested_top_k = [
            item.top_k_tier
            for item in directive_values
            if item.directive is DoubterDirectiveCode.INCREASE_TOP_K and item.top_k_tier is not None
        ]
        top_k = max([strategy.top_k_tier.top_k, *requested_top_k])

        # At most two independent retrieval roots keep the complete program at
        # eight steps after required expansion, verification, and fallback.
        selected_channels: list[PlannerRetrievalChannel] = []
        for channel in channels:
            if channel in selected_channels:
                continue
            if (
                channel is PlannerRetrievalChannel.KEYWORD
                and PlannerRetrievalChannel.TEXT_DENSE in channels
                and not config.is_v15
            ):
                continue
            selected_channels.append(channel)
        selected_channels = selected_channels if config.is_v15 else selected_channels[:2]
        memory_source_steps: list[int] = []
        image_source_steps: list[int] = []
        text_source_steps: list[int] = []
        image_text_pair_steps: list[int] = []
        context_steps: list[int] = []
        for channel in selected_channels:
            step_id = len(steps)
            if channel is PlannerRetrievalChannel.KEYWORD:
                steps.append(
                    _v12_step(
                        step_id,
                        "retrieve_text_by_keyword",
                        {
                            "top_k": top_k,
                            "query": " ".join(
                                _canonical_question_terms(
                                    candidate.canonical_question,
                                    8,
                                )
                            ),
                            "entity_mentions": [],
                            "term_source": "canonical_question",
                            "term_budget": 8,
                            "server_derived": True,
                        },
                        server_role="retrieval",
                    )
                )
                memory_source_steps.append(step_id)
                text_source_steps.append(step_id)
            elif channel is PlannerRetrievalChannel.TEXT_DENSE:
                steps.append(
                    _v12_step(
                        step_id,
                        "retrieve_text_by_embedding",
                        {
                            "top_k": top_k,
                            "vector_role": "question_text_query",
                        },
                        server_role="retrieval",
                    )
                )
                memory_source_steps.append(step_id)
                text_source_steps.append(step_id)
            else:
                steps.append(
                    _v12_step(
                        step_id,
                        "retrieve_image_by_visual_embedding",
                        {
                            "top_k": top_k,
                            "vector_role": "question_joint_query",
                        },
                        server_role="retrieval",
                    )
                )
                image_source_steps.append(step_id)

        def need(
            capability: PlannerCapability,
            directive: DoubterDirectiveCode,
        ) -> bool:
            return capability in required or directive in directive_codes

        if image_source_steps and (
            PlannerExpansionCapability.IMAGE_TEXT in expansions
            or need(
                PlannerCapability.IMAGE_TEXT,
                DoubterDirectiveCode.ADD_IMAGE_TEXT_PAIR,
            )
        ):
            step_id = len(steps)
            steps.append(
                _v12_step(
                    step_id,
                    "retrieve_image_text_pair",
                    {"source_step_ids": list(image_source_steps)},
                    server_role="expansion",
                )
            )
            image_text_pair_steps.append(step_id)
            text_source_steps.append(step_id)
        if image_source_steps and (
            PlannerExpansionCapability.CONTEXT in expansions
            or need(PlannerCapability.CONTEXT, DoubterDirectiveCode.ADD_CONTEXT)
        ):
            step_id = len(steps)
            steps.append(
                _v12_step(
                    step_id,
                    "retrieve_context_around_image",
                    {"source_step_ids": list(image_source_steps), "radius": 2},
                    server_role="expansion",
                )
            )
            context_steps.append(step_id)
            memory_source_steps.append(step_id)
            text_source_steps.append(step_id)
        if image_source_steps and (
            PlannerExpansionCapability.RAW_IMAGE in expansions
            or need(
                PlannerCapability.RAW_IMAGE,
                DoubterDirectiveCode.ADD_RAW_IMAGE_INSPECTION,
            )
        ):
            steps.append(
                _v12_step(
                    len(steps),
                    "inspect_raw_image",
                    {"source_step_ids": list(image_source_steps)},
                    server_role="inspection",
                )
            )
        if memory_source_steps and (
            PlannerExpansionCapability.TEMPORAL in expansions
            or need(PlannerCapability.TEMPORAL, DoubterDirectiveCode.ADD_TEMPORAL)
        ):
            step_id = len(steps)
            steps.append(
                _v12_step(
                    step_id,
                    "expand_temporal_neighbors",
                    {"source_step_ids": list(memory_source_steps), "radius": 2},
                    server_role="expansion",
                )
            )
            memory_source_steps.append(step_id)
            text_source_steps.append(step_id)

        if len(steps) > config.program_max_steps - 2:
            raise PlannerStrategyPolicyError(
                "planner_operator_budget_exceeded",
                "Planner strategy expands beyond the operator budget",
                diagnostics={
                    "materialized_step_count_before_tail": len(steps),
                    "program_max_steps": config.program_max_steps,
                },
            )
        evidence_steps = [
            item.step_id
            for item in steps
            if item.expected_output_type in {"memory_ids", "image_ids", "image_text_pairs"}
        ]
        if not evidence_steps:
            raise PlannerStrategyPolicyError(
                "planner_evidence_path_missing",
                "Planner strategy produced no evidence path",
            )
        verify_id = len(steps)
        steps.append(
            _v12_step(
                verify_id,
                "verify_evidence_sufficiency",
                {
                    "source_step_ids": evidence_steps,
                    "required_modalities": [item.value for item in requirement.required_modalities],
                    "minimum_evidence_groups": requirement.minimum_evidence_groups,
                },
                server_role="verification",
            )
        )
        steps.append(
            _v12_step(
                len(steps),
                "fallback_to_multimodal_rag",
                {
                    "source_step_ids": [verify_id],
                    "condition": "if_insufficient",
                    "reason_code": "insufficient_retrieved_evidence",
                },
                server_role="control",
            )
        )
        strategy_payload = {
            "strategy": strategy.model_dump(mode="json"),
            "revision_directives": directive_payloads,
            "source_sets": {
                "memory_source_steps": memory_source_steps,
                "image_source_steps": image_source_steps,
                "text_source_steps": text_source_steps,
                "image_text_pair_steps": image_text_pair_steps,
                "context_steps": context_steps,
            },
        }
        confidence = strategy.confidence
        strategy_code = strategy.strategy_code

    budget = None
    if config.is_v15:
        try:
            steps, budget = budget_program_steps(
                steps,
                total_evidence_max=config.total_evidence_max,
                top_k_max=config.operator_top_k_max,
            )
        except ValueError as exc:
            raise PlannerStrategyPolicyError(
                "planner_evidence_budget_infeasible", str(exc)
            ) from exc
    plan_identity = {
        "run_id": run_id,
        "compiler_unit_id": compiler_unit_id,
        "qbank_id": candidate.qbank_id,
        "round_id": round_id,
        "strategy": strategy_payload,
        "steps": [item.model_dump(mode="json") for item in steps],
    }
    plan_id = stable_compiler_id("memory-program", plan_identity)
    effective_capabilities = set().union(
        *(_CAPABILITY_OUTPUTS.get(step.op_name, set()) for step in steps)
    )
    payload = {
        "plan_id": plan_id,
        "qbank_id": candidate.qbank_id,
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "compiler_unit_id": compiler_unit_id,
        "producer_model_call_id": producer_model_call_id,
        "operator_set_version": config.operator_set_version,
        "round_id": round_id,
        "operator_sequence": [item.model_dump(mode="json") for item in steps],
        "expected_evidence_ids": candidate.evidence_ids,
        "fallback_operator": None,
        "required_modalities": [item.value for item in candidate.required_modalities],
        "planner_rationale_short": strategy_code,
        "planner_confidence": confidence,
        "program_hash": "pending",
        "metadata": {
            **(
                {
                    "evidence_budget": budget.model_dump(mode="json"),
                    "planning_mode": config.planning_mode,
                    "memory_raw_image_policy": config.memory_raw_image_policy,
                }
                if budget is not None
                else {}
            ),
            "strategy_code": strategy_code,
            "strategy": (strategy.model_dump(mode="json") if strategy is not None else None),
            "server_materialized": True,
            "requirement_contract_hash": requirement.content_hash,
            "revision_directives": [
                item.directive.value
                for item in sorted(directive_values, key=lambda item: item.ordinal)
            ],
            "revision_directive_payloads": directive_payloads,
            "requested_capabilities": (
                sorted(value.value for value in requested)
                if config.is_v13 and not requirement.refusal_only
                else []
            ),
            "effective_capabilities": sorted(value.value for value in effective_capabilities),
            "strategy_diff": {
                "requested_not_effective": sorted(
                    value.value
                    for value in (
                        requested - effective_capabilities
                        if config.is_v13 and not requirement.refusal_only
                        else set()
                    )
                ),
                "effective_not_requested": sorted(
                    value.value
                    for value in (
                        effective_capabilities - requested
                        if config.is_v13 and not requirement.refusal_only
                        else effective_capabilities
                    )
                ),
            },
        },
    }
    pending = MemoryProgram.model_validate(payload)
    payload["program_hash"] = memory_program_content_hash(pending)
    return MemoryProgram.model_validate(payload)


def build_planner_requirement_contract(
    *,
    run_id: str,
    compiler_unit_id: str,
    candidate: QuestionCandidate,
    config: PlannerConfig | None = None,
) -> PlannerRequirementContract:
    modality_map = {
        "text": (PlannerRequiredModality.TEXT,),
        "caption": (PlannerRequiredModality.CAPTION,),
        "image": (PlannerRequiredModality.IMAGE,),
        "raw_image": (PlannerRequiredModality.RAW_IMAGE,),
        "multimodal": (
            PlannerRequiredModality.TEXT,
            PlannerRequiredModality.IMAGE,
        ),
    }
    modalities = list(
        dict.fromkeys(
            mapped
            for item in candidate.required_modalities
            for mapped in modality_map.get(item.value, ())
        )
    )
    if not modalities:
        modalities = [PlannerRequiredModality.TEXT]
    capabilities: set[PlannerCapability] = {
        PlannerCapability.EVIDENCE_GROUPS,
    }
    refusal = (
        candidate.question_type is QuestionType.ANSWER_REFUSAL
        or candidate.reasoning_type is QuestionReasoningType.REFUSAL
    )
    if refusal:
        capabilities = {PlannerCapability.CONTROL}
    else:
        capabilities.update(
            PlannerCapability.IMAGE
            if item in {PlannerRequiredModality.IMAGE, PlannerRequiredModality.RAW_IMAGE}
            else PlannerCapability.TEXT
            for item in modalities
        )
        capabilities.update({PlannerCapability.VERIFICATION, PlannerCapability.CONTROL})
        if candidate.question_type in _VISUAL_TYPES:
            capabilities.update({PlannerCapability.IMAGE, PlannerCapability.RAW_IMAGE})
            if PlannerRequiredModality.RAW_IMAGE not in modalities:
                modalities.append(PlannerRequiredModality.RAW_IMAGE)
        if candidate.question_type is QuestionType.IMAGE_TEXT_BINDING:
            capabilities.add(PlannerCapability.IMAGE_TEXT)
        if candidate.question_type in _TEMPORAL_TYPES:
            capabilities.add(PlannerCapability.TEMPORAL)
        if candidate.question_type is QuestionType.MULTI_HOP:
            capabilities.add(PlannerCapability.EVIDENCE_GROUPS)
    allowed_capabilities = set(capabilities)
    if not refusal:
        allowed_capabilities.update(
            {
                PlannerCapability.TEXT,
                PlannerCapability.EVIDENCE_GROUPS,
                PlannerCapability.VERIFICATION,
                PlannerCapability.CONTROL,
            }
        )
        if PlannerCapability.IMAGE in capabilities:
            allowed_capabilities.update(
                {
                    PlannerCapability.IMAGE_TEXT,
                    PlannerCapability.CONTEXT,
                    PlannerCapability.RAW_IMAGE,
                }
            )
    minimum_groups = max(
        1,
        2
        if candidate.question_type is QuestionType.MULTI_HOP
        or candidate.reasoning_type is QuestionReasoningType.MULTI_HOP
        else candidate.hop_count,
    )
    treatment = config is not None and config.is_v15 and config.memory_raw_image_policy == "forbid"
    if treatment:
        # Unsupported requirements are scientific rejections, not weaker text tasks.
        if PlannerCapability.RAW_IMAGE in capabilities or (
            PlannerRequiredModality.RAW_IMAGE in modalities
        ):
            raise PlannerStrategyPolicyError(
                "planner_treatment_required_raw_image_unavailable",
                "candidate requires memory raw images unavailable in this treatment",
                diagnostics={
                    "required_capabilities": sorted(item.value for item in capabilities),
                    "required_modalities": [item.value for item in modalities],
                    "memory_raw_image_policy": "forbid",
                },
            )
        allowed_capabilities.discard(PlannerCapability.RAW_IMAGE)
    if config is not None and config.is_v15 and config.planning_mode == "fixed":
        assert config.fixed_strategy is not None
        if PlannerRetrievalChannel.VISUAL_DENSE in config.fixed_strategy.retrieval_channels:
            allowed_capabilities.add(PlannerCapability.IMAGE)
    role_metadata = (
        {
            "required_semantic_roles": required_evidence_roles(
                modalities=[item.value for item in modalities],
                capabilities=[item.value for item in capabilities],
                refusal_only=refusal,
            ),
            "semantic_role_policy": EVIDENCE_ROLE_POLICY,
            "independent_group_relation_required": minimum_groups > 1 and not refusal,
            "relation_assessment_stage": "answer_grounding" if minimum_groups > 1 and not refusal
            else "not_applicable",
        }
        if config is not None and config.is_v15
        else {}
    )
    identity = {
        "run_id": run_id,
        "compiler_unit_id": compiler_unit_id,
        "qbank_id": candidate.qbank_id,
        "required_modalities": [item.value for item in modalities],
        "minimum_evidence_groups": minimum_groups,
        "required_capabilities": sorted(item.value for item in capabilities),
        "allowed_capabilities": sorted(item.value for item in allowed_capabilities),
        "fallback_required": not refusal,
        "refusal_only": refusal,
        **role_metadata,
        **({"memory_raw_image_policy": "forbid"} if treatment else {}),
    }
    return PlannerRequirementContract(
        requirement_contract_id=stable_compiler_id("planner-requirement", identity),
        run_id=run_id,
        compiler_unit_id=compiler_unit_id,
        qbank_id=candidate.qbank_id,
        required_modalities=modalities,
        minimum_evidence_groups=minimum_groups,
        required_capabilities=sorted(capabilities, key=lambda item: item.value),
        allowed_capabilities=sorted(allowed_capabilities, key=lambda item: item.value),
        fallback_required=not refusal,
        refusal_only=refusal,
        content_hash=canonical_content_hash(identity),
        metadata={
            **role_metadata,
            **({"memory_raw_image_policy": "forbid"} if treatment else {}),
        },
    )


class PlannerProgramPolicy:
    """Enforce non-oracle, typed, acyclic Planner v1 programs."""

    def __init__(
        self,
        config: PlannerConfig,
        *,
        registry: OperatorRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry

    def _capabilities(
        self,
        op_name: str,
        metadata_key: str,
        fallback: Mapping[str, set[PlannerCapability]],
    ) -> set[PlannerCapability]:
        if (
            self.registry is None
            or not (self.config.is_v11 or self.config.is_v12 or self.config.is_v13)
            or self.registry.version
            not in {
                "pmmc-operators-v3",
                "pmmc-operators-v4",
                "pmmc-operators-v5",
                "pmmc-operators-v6",
                "pmmc-operators-v7",
            }
        ):
            return set(fallback.get(op_name, set()))
        raw = self.registry.resolve(op_name).spec.metadata.get(metadata_key, [])
        if not isinstance(raw, list):
            return set()
        try:
            return {PlannerCapability(str(value)) for value in raw}
        except ValueError:
            return set()

    def validate(
        self,
        program: MemoryProgram,
        candidate: QuestionCandidate,
        requirement_contract: PlannerRequirementContract | None = None,
    ) -> ProgramPolicyResult:
        issues: list[ProgramPolicyIssue] = []

        def issue(
            code: str,
            message: str,
            step_id: int | None = None,
            metadata: Mapping[str, object] | None = None,
        ) -> None:
            issues.append(
                ProgramPolicyIssue(
                    code=code,
                    message=message,
                    step_id=step_id,
                    metadata=dict(metadata or {}),
                )
            )

        if program.operator_set_version != self.config.operator_set_version:
            issue("operator_set_mismatch", "Planner program uses the wrong operator set")
        if self.config.is_v15:
            try:
                allocated, budget = budget_program_steps(
                    program.operator_sequence,
                    total_evidence_max=self.config.total_evidence_max,
                    top_k_max=self.config.operator_top_k_max,
                )
                if any(
                    a.args != b.args
                    for a, b in zip(allocated, program.operator_sequence, strict=True)
                ):
                    issue(
                        "evidence_budget_exceeded", "retrieval quotas exceed the preflight budget"
                    )
                if program.metadata.get("evidence_budget") != budget.model_dump(mode="json"):
                    issue(
                        "evidence_budget_manifest_mismatch", "program evidence budget is not native"
                    )
                if any(
                    a.metadata != b.metadata
                    for a, b in zip(allocated, program.operator_sequence, strict=True)
                ):
                    issue("evidence_step_quota_mismatch", "step evidence quotas are not frozen")
            except (TypeError, ValueError, KeyError):
                issue("evidence_budget_invalid", "program has infeasible evidence dataflow")
            if self.config.memory_raw_image_policy == "forbid" and any(
                s.op_name == "inspect_raw_image" for s in program.operator_sequence
            ):
                issue("treatment_raw_image_forbidden", "raw-image operator disabled by treatment")
        if (self.config.is_v11 or self.config.is_v12 or self.config.is_v13) and (
            requirement_contract is None
            or requirement_contract.qbank_id != candidate.qbank_id
            or requirement_contract.run_id != (program.run_id or candidate.run_id)
        ):
            issue(
                "requirement_contract_missing",
                "Planner v1.1 requires the server-derived candidate contract",
            )
        if program.fallback_operator is not None:
            issue("implicit_fallback_forbidden", "fallback must be an explicit final step")
        if len(program.operator_sequence) > self.config.program_max_steps:
            issue("step_limit_exceeded", "program exceeds the Planner step limit")

        refusal = (
            candidate.question_type is QuestionType.ANSWER_REFUSAL
            or candidate.reasoning_type is QuestionReasoningType.REFUSAL
        )
        output_by_step: dict[int, str] = {}
        verify_steps: list[int] = []
        terminal_steps: list[int] = []
        capabilities_by_step: dict[int, set[PlannerCapability]] = {}
        retrieval_fingerprints: set[tuple[str, str, int]] = set()
        for step in program.operator_sequence:
            output_by_step[step.step_id] = step.expected_output_type
            source_ids = step.args.get("source_step_ids", [])
            if not isinstance(source_ids, list) or any(
                not isinstance(value, int) or value < 0 or value >= step.step_id
                for value in source_ids
            ):
                issue(
                    "invalid_dataflow",
                    "source steps must be earlier integer steps",
                    step.step_id,
                )
                source_ids = []
            if len(source_ids) != len(set(source_ids)):
                issue("duplicate_source_step", "source steps must be unique", step.step_id)
            source_capabilities = set().union(
                *(capabilities_by_step.get(value, set()) for value in source_ids)
            )
            required_inputs = self._capabilities(
                step.op_name,
                "input_capabilities",
                _CAPABILITY_INPUTS,
            )
            if not required_inputs <= source_capabilities:
                issue(
                    "dataflow_capability_missing",
                    "source steps do not provide the required operator capability",
                    step.step_id,
                )
            produced = set(source_capabilities)
            produced.update(
                self._capabilities(
                    step.op_name,
                    "output_capabilities",
                    _CAPABILITY_OUTPUTS,
                )
            )
            capabilities_by_step[step.step_id] = produced

            if step.op_name in _IMAGE_INPUT_OPERATORS and (
                not source_ids
                or any(output_by_step.get(value) != "image_ids" for value in source_ids)
            ):
                issue(
                    "image_source_required",
                    "operator requires upstream image output",
                    step.step_id,
                )
            if step.op_name == "expand_temporal_neighbors" and (
                not source_ids
                or any(output_by_step.get(value) != "memory_ids" for value in source_ids)
            ):
                issue(
                    "memory_source_required",
                    "temporal expansion requires memory output",
                    step.step_id,
                )
            if step.op_name in {
                "retrieve_text_by_embedding",
                "retrieve_image_by_visual_embedding",
            } and step.args.get("vector_role") not in {
                "question_text_query",
                "question_joint_query",
            }:
                issue(
                    "query_vector_role_forbidden",
                    "embedding role is not a question query",
                    step.step_id,
                )
            top_k = step.args.get("top_k")
            if isinstance(top_k, int) and top_k > self.config.operator_top_k_max:
                issue("top_k_exceeded", "top_k exceeds 50", step.step_id)
            if step.op_name in {
                "retrieve_text_by_keyword",
                "retrieve_text_by_embedding",
                "retrieve_image_by_visual_embedding",
            }:
                fingerprint = (
                    step.op_name,
                    str(step.args.get("vector_role") or step.args.get("query") or ""),
                    int(top_k) if isinstance(top_k, int) else -1,
                )
                if fingerprint in retrieval_fingerprints:
                    issue(
                        "duplicate_retrieval",
                        "equivalent retrieval steps are forbidden",
                        step.step_id,
                    )
                retrieval_fingerprints.add(fingerprint)
            radius = step.args.get("radius")
            if isinstance(radius, int) and radius > self.config.expansion_radius_max:
                issue("radius_exceeded", "expansion radius exceeds 2", step.step_id)
            forbidden_ids = {
                *candidate.evidence_ids,
                *candidate.source_memory_ids,
                *(
                    value
                    for value in (
                        program.run_id,
                        program.snapshot_id,
                        program.compiler_unit_id,
                    )
                    if value is not None
                ),
                candidate.qbank_id,
                *(
                    [candidate.expected_answer]
                    if candidate.expected_answer is not None
                    and len(candidate.expected_answer.strip()) >= 3
                    else []
                ),
            }
            if _contains_forbidden_value(step.args, forbidden_ids):
                issue(
                    "oracle_evidence_reference",
                    "operator args may not contain expected answer/evidence",
                    step.step_id,
                )
            if (
                self.config.is_v11 or self.config.is_v12 or self.config.is_v13
            ) and step.op_name == "retrieve_text_by_keyword":
                term_budget = step.args.get("term_budget")
                expected_terms = (
                    _canonical_question_terms(
                        candidate.canonical_question,
                        term_budget,
                    )
                    if isinstance(term_budget, int) and 1 <= term_budget <= 8
                    else []
                )
                if (
                    step.args.get("term_source") != "canonical_question"
                    or step.args.get("server_derived") is not True
                    or step.args.get("entity_mentions") != []
                    or step.args.get("query") != " ".join(expected_terms)
                ):
                    issue(
                        "keyword_not_server_derived",
                        "keyword query must be derived from the canonical question",
                        step.step_id,
                    )
            if step.op_name == "verify_evidence_sufficiency":
                verify_steps.append(step.step_id)
                if requirement_contract is not None:
                    actual_modalities = {
                        str(value) for value in step.args.get("required_modalities", [])
                    }
                    expected_modalities = {
                        value.value for value in requirement_contract.required_modalities
                    }
                    if actual_modalities != expected_modalities:
                        issue(
                            "requirement_modality_mismatch",
                            "verify modalities must equal the requirement contract",
                            step.step_id,
                        )
                    if (
                        step.args.get("minimum_evidence_groups")
                        != requirement_contract.minimum_evidence_groups
                    ):
                        issue(
                            "requirement_group_mismatch",
                            "verify group minimum must equal the requirement contract",
                            step.step_id,
                        )
            if step.op_name in _CONTROL_OPERATORS:
                terminal_steps.append(step.step_id)
                if step.step_id != len(program.operator_sequence) - 1:
                    issue("terminal_not_last", "control operator must be final", step.step_id)

        if verify_steps:
            last_non_control = max(
                step.step_id
                for step in program.operator_sequence
                if step.op_name not in _CONTROL_OPERATORS
            )
            if verify_steps != [last_non_control]:
                issue("verify_placement", "verification must be the final non-control step")
        elif not refusal:
            issue(
                "evidence_verification_required",
                "answerable programs require an evidence sufficiency step",
            )
        if candidate.question_type in _VISUAL_TYPES and not any(
            step.op_name == "inspect_raw_image" for step in program.operator_sequence
        ):
            issue("raw_image_required", "visual programs must inspect a raw image")
        if candidate.question_type in _TEMPORAL_TYPES and not any(
            step.op_name == "expand_temporal_neighbors" for step in program.operator_sequence
        ):
            issue("temporal_expansion_required", "temporal/update/conflict programs need expansion")

        names = [step.op_name for step in program.operator_sequence]
        if refusal and names != ["abstain_if_no_evidence"]:
            issue("refusal_program_invalid", "refusal candidates compile only to abstain")
        if not refusal and names == ["abstain_if_no_evidence"]:
            issue("answerable_program_abstains", "answerable candidates need an evidence path")
        if len(terminal_steps) > 1:
            issue("multiple_terminal_steps", "program may contain only one terminal control step")
        if (self.config.is_v11 or self.config.is_v12 or self.config.is_v13) and not refusal:
            if len(names) < 2 or names[-2:] != [
                "verify_evidence_sufficiency",
                "fallback_to_multimodal_rag",
            ]:
                issue(
                    "answerable_terminal_sequence",
                    "answerable programs must end in verify then conditional fallback",
                )
            elif program.operator_sequence[-1].args.get("condition") != "if_insufficient":
                issue(
                    "fallback_condition_invalid",
                    "answerable fallback must be conditional on insufficiency",
                    program.operator_sequence[-1].step_id,
                )
            else:
                verify_step = program.operator_sequence[-2]
                fallback_step = program.operator_sequence[-1]
                if fallback_step.args.get("source_step_ids") != [verify_step.step_id]:
                    issue(
                        "fallback_dataflow_invalid",
                        "conditional fallback must consume the final verification step",
                        fallback_step.step_id,
                    )
                verify_sources = verify_step.args.get("source_step_ids", [])
                if not isinstance(verify_sources, list) or not verify_sources:
                    issue(
                        "verify_dataflow_empty",
                        "evidence verification must consume an evidence-producing path",
                        verify_step.step_id,
                    )
        if requirement_contract is not None and program.operator_sequence:
            terminal_step_id = program.operator_sequence[-1].step_id
            all_capabilities = capabilities_by_step.get(terminal_step_id, set())
            if self.config.is_v13:
                forbidden_capabilities = all_capabilities - set(
                    requirement_contract.allowed_capabilities
                )
                if forbidden_capabilities:
                    issue(
                        "program_capability_not_allowed",
                        "program produces capabilities outside its requirement allowlist",
                        metadata={
                            "forbidden_capabilities": sorted(
                                value.value for value in forbidden_capabilities
                            )
                        },
                    )
            missing_capabilities = (
                set(requirement_contract.required_capabilities) - all_capabilities
            )
            if missing_capabilities:
                missing_issue = ProgramPolicyIssue(
                    severity=(
                        ProgramPolicySeverity.WARNING
                        if self.config.is_v15 and self.config.planning_mode == "fixed"
                        else ProgramPolicySeverity.ERROR
                    ),
                    code="program_capability_incomplete",
                    message="program does not satisfy its requirement capabilities",
                    metadata={
                        "missing_capabilities": sorted(v.value for v in missing_capabilities)
                    },
                )
                issues.append(missing_issue)

        valid = not any(item.severity is ProgramPolicySeverity.ERROR for item in issues)
        result_id = stable_compiler_id(
            "program-policy",
            [
                program.plan_id,
                self.config.policy_version,
                [item.model_dump(mode="json") for item in issues],
            ],
        )
        return ProgramPolicyResult(
            policy_result_id=result_id,
            run_id=program.run_id or candidate.run_id,
            plan_id=program.plan_id,
            policy_version=self.config.policy_version,
            valid=valid,
            issues=issues,
            content_hash=canonical_content_hash(
                [valid, [item.model_dump(mode="json") for item in issues]]
            ),
            created_at=utc_now(),
        )


def planner_terminal_outcome(
    *,
    run_id: str,
    compiler_unit_id: str,
    qbank_id: str,
    status: PlannerOutcomeStatus,
    reason_code: str,
    planner_batch_id: str | None = None,
    plan_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> PlannerCandidateOutcome:
    return PlannerCandidateOutcome(
        planner_outcome_id=stable_compiler_id(
            "planner-outcome", [run_id, compiler_unit_id, qbank_id, status.value]
        ),
        run_id=run_id,
        compiler_unit_id=compiler_unit_id,
        qbank_id=qbank_id,
        planner_batch_id=planner_batch_id,
        plan_id=plan_id,
        status=status,
        reason_code=reason_code,
        content_hash=canonical_content_hash(
            [run_id, compiler_unit_id, qbank_id, status.value, reason_code, plan_id]
        ),
        metadata=dict(metadata or {}),
    )
