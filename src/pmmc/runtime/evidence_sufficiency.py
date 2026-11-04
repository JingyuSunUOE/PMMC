"""Gold-free structural evidence checks for query-time program execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.memory.operators import ProgramExecutionResult
from pmmc.schemas import (
    EvidenceType,
    Modality,
    PlannerCapability,
    PlannerRequiredModality,
    PlannerRequirementContract,
    QueryEvidenceSufficiencyAssessment,
    QueryFallbackReason,
    QueryProgramBinding,
    TraceStatus,
)

EvidenceGroupResolver = Callable[[str, EvidenceType], Iterable[str]]


_OPERATOR_CAPABILITIES: dict[str, frozenset[PlannerCapability]] = {
    "retrieve_text_by_keyword": frozenset(
        {PlannerCapability.TEXT, PlannerCapability.EVIDENCE_GROUPS}
    ),
    "retrieve_text_by_embedding": frozenset(
        {PlannerCapability.TEXT, PlannerCapability.EVIDENCE_GROUPS}
    ),
    "retrieve_image_by_visual_embedding": frozenset(
        {PlannerCapability.IMAGE, PlannerCapability.EVIDENCE_GROUPS}
    ),
    "retrieve_image_text_pair": frozenset(
        {
            PlannerCapability.IMAGE,
            PlannerCapability.TEXT,
            PlannerCapability.IMAGE_TEXT,
            PlannerCapability.EVIDENCE_GROUPS,
        }
    ),
    "retrieve_context_around_image": frozenset(
        {
            PlannerCapability.IMAGE,
            PlannerCapability.TEXT,
            PlannerCapability.CONTEXT,
            PlannerCapability.EVIDENCE_GROUPS,
        }
    ),
    "expand_temporal_neighbors": frozenset(
        {
            PlannerCapability.TEXT,
            PlannerCapability.TEMPORAL,
            PlannerCapability.EVIDENCE_GROUPS,
        }
    ),
    "inspect_raw_image": frozenset(
        {PlannerCapability.IMAGE, PlannerCapability.RAW_IMAGE}
    ),
    "verify_evidence_sufficiency": frozenset({PlannerCapability.VERIFICATION}),
    "fallback_to_multimodal_rag": frozenset({PlannerCapability.CONTROL}),
    "abstain_if_no_evidence": frozenset({PlannerCapability.CONTROL}),
}


def _modalities(result: ProgramExecutionResult) -> set[Modality]:
    evidence = result.evidence
    text = any(
        item.memory_id is not None
        or item.caption_id is not None
        or item.evidence_type in {EvidenceType.TEXT, EvidenceType.CAPTION}
        for item in evidence
    )
    caption = any(
        item.caption_id is not None or item.evidence_type is EvidenceType.CAPTION
        for item in evidence
    )
    image = any(item.image_id is not None for item in evidence)
    raw_image = any(
        item.image_id is not None and item.evidence_type is EvidenceType.RAW_IMAGE
        for item in evidence
    )
    output: set[Modality] = set()
    if text:
        output.add(Modality.TEXT)
    if caption:
        output.add(Modality.CAPTION)
    if image:
        output.add(Modality.IMAGE)
    if raw_image:
        output.add(Modality.RAW_IMAGE)
    if text and image:
        output.add(Modality.MULTIMODAL)
    return output


def _required_modalities(
    values: list[PlannerRequiredModality],
) -> set[Modality]:
    return {Modality(value.value) for value in values}


def _evidence_groups(
    result: ProgramExecutionResult,
    resolver: EvidenceGroupResolver | None,
) -> set[str]:
    groups: set[str] = set()
    for item in result.evidence:
        item_groups: set[str] = set()
        explicit = item.metadata.get("evidence_group_ids")
        if isinstance(explicit, list):
            item_groups.update(
                value for value in explicit if isinstance(value, str) and value
            )
        singular = item.metadata.get("evidence_group_id")
        if isinstance(singular, str) and singular:
            item_groups.add(singular)
        if resolver is not None:
            item_groups.update(resolver(item.evidence_id, item.evidence_type))
        if not item_groups and resolver is None:
            # Typed evidence identity is the conservative fallback. It never expands a
            # session down to sibling memories or a MemGallery round.
            item_groups.add(f"{item.evidence_type.value}:{item.evidence_id}")
        groups.update(item_groups)
    return groups


def assess_query_evidence_sufficiency(
    *,
    query_executor_run_id: str,
    binding: QueryProgramBinding,
    requirement: PlannerRequirementContract,
    result: ProgramExecutionResult,
    group_resolver: EvidenceGroupResolver | None = None,
    relevant_group_ids: set[str] | None = None,
    relevance_threshold: float | None = None,
    answer_support: bool | None = None,
) -> QueryEvidenceSufficiencyAssessment:
    """Assess actual runtime evidence without compiler-private or evaluator facts."""

    if requirement.qbank_id != binding.qbank_id:
        raise ValueError("requirement contract and query binding do not match")
    if result.trace.execution_id != binding.metadata.get(
        "execution_id", result.trace.execution_id
    ):
        raise ValueError("query binding execution lineage does not match")
    if result.trace.query_id != binding.query_id:
        raise ValueError("program execution belongs to a different query")
    if result.trace.plan_id != binding.plan_id:
        raise ValueError("program execution belongs to a different frozen plan")

    actual_capabilities: set[PlannerCapability] = set()
    for step in result.steps:
        if step.success:
            actual_capabilities.update(
                _OPERATOR_CAPABILITIES.get(step.op_name, frozenset())
            )
    actual_modalities = _modalities(result)
    required_modalities = _required_modalities(requirement.required_modalities)
    groups = _evidence_groups(result, group_resolver)
    raw_image_reached = Modality.RAW_IMAGE in actual_modalities
    raw_image_required = (
        PlannerRequiredModality.RAW_IMAGE in requirement.required_modalities
        or PlannerCapability.RAW_IMAGE in requirement.required_capabilities
    )
    technical_success = result.trace.status is TraceStatus.SUCCESS
    nonempty = bool(result.evidence)
    structural_success = (
        technical_success
        and nonempty
        and set(requirement.required_capabilities) <= actual_capabilities
        and required_modalities <= actual_modalities
        and len(groups) >= requirement.minimum_evidence_groups
        and (not raw_image_required or raw_image_reached)
    )
    v2 = relevant_group_ids is not None
    query_relevance = (
        len(relevant_group_ids) >= requirement.minimum_evidence_groups
        if relevant_group_ids is not None
        else None
    )
    effective_answer_support = (
        bool(answer_support) if v2 else None
    )
    sufficient = structural_success and (
        bool(query_relevance) and bool(effective_answer_support) if v2 else True
    )
    failure_reason = None
    if not sufficient:
        failure_reason = (
            QueryFallbackReason.PROGRAM_EXECUTION_FAILED
            if not technical_success
            else QueryFallbackReason.EMPTY_EVIDENCE
            if not nonempty
            else QueryFallbackReason.RUNTIME_RELEVANCE_INSUFFICIENT
            if structural_success and v2 and not query_relevance
            else QueryFallbackReason.REQUIREMENT_INSUFFICIENT
        )
    payload = {
        "query_executor_run_id": query_executor_run_id,
        "binding_id": binding.query_program_binding_id,
        "execution_id": result.trace.execution_id,
        "actual_capabilities": sorted(item.value for item in actual_capabilities),
        "required_capabilities": sorted(
            item.value for item in requirement.required_capabilities
        ),
        "actual_modalities": sorted(item.value for item in actual_modalities),
        "required_modalities": sorted(item.value for item in required_modalities),
        "independent_groups": sorted(groups),
        "sufficient": sufficient,
        "query_relevance": query_relevance,
        "answer_support": effective_answer_support,
        "failure_reason": failure_reason.value if failure_reason else None,
    }
    return QueryEvidenceSufficiencyAssessment(
        sufficiency_assessment_id=stable_compiler_id(
            "query-evidence-sufficiency", payload
        ),
        query_executor_run_id=query_executor_run_id,
        query_id=binding.query_id,
        query_program_binding_id=binding.query_program_binding_id,
        execution_id=result.trace.execution_id,
        technical_success=technical_success,
        nonempty_evidence=nonempty,
        actual_capabilities=payload["actual_capabilities"],
        required_capabilities=payload["required_capabilities"],
        actual_modalities=sorted(actual_modalities, key=lambda item: item.value),
        required_modalities=sorted(required_modalities, key=lambda item: item.value),
        independent_group_count=len(groups),
        minimum_evidence_groups=requirement.minimum_evidence_groups,
        raw_image_reached=raw_image_reached,
        raw_image_required=raw_image_required,
        structural_requirement_success=structural_success if v2 else None,
        query_evidence_relevance=query_relevance,
        answer_support=effective_answer_support,
        relevant_evidence_group_count=(len(relevant_group_ids or ()) if v2 else None),
        relevance_threshold=relevance_threshold if v2 else None,
        sufficient=sufficient,
        failure_reason=failure_reason,
        content_hash=canonical_content_hash(payload),
        metadata={
            "policy": (
                "query-runtime-evidence-v2" if v2 else "query-runtime-structural-v1"
            ),
            "sufficiency_policy_version": (
                "query-runtime-evidence-v2" if v2 else "query-runtime-structural-v1"
            ),
        },
    )


__all__ = ["EvidenceGroupResolver", "assess_query_evidence_sufficiency"]
