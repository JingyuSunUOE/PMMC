"""Evidence-driven assessment for real operator-kernel dry-runs."""

from __future__ import annotations

from collections.abc import Sequence

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.evidence_lineage import EvidenceLineageResolver
from pmmc.memory.operators import ProgramExecutionResult
from pmmc.schemas import (
    EvidenceMatchKind,
    EvidenceMatchRole,
    EvidenceType,
    Modality,
    OperatorEvidence,
    PlannerOutcomeStatus,
    ProgramDryRunAssessment,
    ProgramEvidenceMatch,
    ProgramExecutionEvidenceRef,
    QuestionCandidate,
    QuestionType,
    TraceStatus,
)

_VISUAL_QUESTION_TYPES = {
    QuestionType.VISUAL_DETAIL,
    QuestionType.IMAGE_TEXT_BINDING,
    QuestionType.VISUAL_SEARCH,
}
_MATCH_KIND_PRIORITY = {
    EvidenceMatchKind.DIRECT: 0,
    EvidenceMatchKind.DERIVED: 1,
    EvidenceMatchKind.GROUP: 2,
    EvidenceMatchKind.CONTEXTUAL: 3,
}


def _metadata_group_ids(metadata: dict[str, object]) -> frozenset[str]:
    groups: set[str] = set()
    singular = metadata.get("evidence_group_id")
    if isinstance(singular, str) and singular:
        groups.add(singular)
    plural = metadata.get("evidence_group_ids")
    if isinstance(plural, list):
        groups.update(item for item in plural if isinstance(item, str) and item)
    return frozenset(groups)


def _typed_id(value: OperatorEvidence) -> str:
    return (
        value.memory_id
        or value.image_id
        or value.session_id
        or value.caption_id
        or value.evidence_id
    )


def _execution_evidence_identity(value: ProgramExecutionEvidenceRef) -> tuple[str | None, ...]:
    """Identify one evidence object across pass-through operator outputs."""

    return (
        value.memory_id,
        value.image_id,
        value.session_id,
        value.caption_id,
        value.evidence_id,
    )


def _operator_group_roles(
    lineage: EvidenceLineageResolver,
    value: OperatorEvidence,
) -> dict[str, EvidenceMatchRole]:
    explicit_groups = _metadata_group_ids(value.metadata)
    if explicit_groups:
        return {
            group_id: EvidenceMatchRole.DISCRIMINATIVE
            for group_id in explicit_groups
        }
    owning_memory_id = value.metadata.get("owning_memory_id")
    if isinstance(owning_memory_id, str) and owning_memory_id:
        owning_groups = lineage.group_roles_for(memory_id=owning_memory_id)
        if owning_groups:
            return owning_groups
    groups = lineage.group_roles_for(
        memory_id=value.memory_id,
        image_id=value.image_id,
        session_id=value.session_id,
        caption_id=value.caption_id,
        evidence_id=value.evidence_id
        if not any(
            (
                value.memory_id,
                value.image_id,
                value.session_id,
                value.caption_id,
            )
        )
        else None,
    )
    return groups or {value.evidence_id: EvidenceMatchRole.DISCRIMINATIVE}


def _is_derived_lineage_match(
    *,
    expected: OperatorEvidence,
    observed: ProgramExecutionEvidenceRef,
    lineage: EvidenceLineageResolver,
) -> bool:
    closure = lineage.expand(
        memory_ids=(() if observed.memory_id is None else (observed.memory_id,)),
        session_ids=(() if observed.session_id is None else (observed.session_id,)),
        image_ids=(() if observed.image_id is None else (observed.image_id,)),
        caption_ids=(() if observed.caption_id is None else (observed.caption_id,)),
    )
    return bool(
        (expected.memory_id is not None and expected.memory_id in closure.memory_ids)
        or (expected.session_id is not None and expected.session_id in closure.session_ids)
        or (expected.image_id is not None and expected.image_id in closure.image_ids)
        or (expected.caption_id is not None and expected.caption_id in closure.caption_ids)
    )


def assess_program_dry_run(
    *,
    candidate: QuestionCandidate,
    expected_evidence: Sequence[OperatorEvidence],
    result: ProgramExecutionResult,
    lineage: EvidenceLineageResolver,
    require_expected_complete: bool = False,
) -> tuple[ProgramDryRunAssessment, tuple[ProgramEvidenceMatch, ...]]:
    trace = result.trace
    assessment_id = stable_compiler_id(
        "program-dry-run-assessment", [trace.execution_id, candidate.qbank_id]
    )
    actual = list(result.evidence)
    actual_group_roles: dict[str, dict[str, EvidenceMatchRole]] = {}
    for item in actual:
        explicit_groups = _metadata_group_ids(item.metadata)
        actual_group_roles[item.evidence_ref_id] = (
            {
                group_id: EvidenceMatchRole.DISCRIMINATIVE
                for group_id in explicit_groups
            }
            if explicit_groups
            else lineage.group_roles_for(
                memory_id=item.memory_id,
                image_id=item.image_id,
                session_id=item.session_id,
                caption_id=item.caption_id,
                evidence_id=item.evidence_id
                if not any(
                    (
                        item.memory_id,
                        item.image_id,
                        item.session_id,
                        item.caption_id,
                    )
                )
                else None,
            )
        )
        if not actual_group_roles[item.evidence_ref_id]:
            actual_group_roles[item.evidence_ref_id] = {
                item.evidence_id: EvidenceMatchRole.DISCRIMINATIVE
            }
    expected_group_roles: dict[str, dict[str, EvidenceMatchRole]] = {}
    for expected_item in expected_evidence:
        roles = expected_group_roles.setdefault(expected_item.evidence_id, {})
        for group_id, role in _operator_group_roles(lineage, expected_item).items():
            prior = roles.get(group_id)
            if prior is None or role is EvidenceMatchRole.DISCRIMINATIVE:
                roles[group_id] = role
    discriminative_expected_groups = {
        group_id
        for roles in expected_group_roles.values()
        for group_id, role in roles.items()
        if role is EvidenceMatchRole.DISCRIMINATIVE
    }
    contextual_expected_groups = {
        group_id
        for roles in expected_group_roles.values()
        for group_id, role in roles.items()
        if role is EvidenceMatchRole.CONTEXTUAL
    }
    matched_groups: set[str] = set()
    match_observations: dict[
        tuple[str, str],
        list[
            tuple[
                OperatorEvidence,
                ProgramExecutionEvidenceRef,
                EvidenceMatchKind,
                frozenset[str],
                bool,
                bool,
            ]
        ],
    ] = {}
    for expected in expected_evidence:
        expected_id = _typed_id(expected)
        for observed in actual:
            actual_id = (
                observed.memory_id
                or observed.image_id
                or observed.session_id
                or observed.caption_id
                or observed.evidence_id
            )
            expected_roles = expected_group_roles[expected.evidence_id]
            actual_roles = actual_group_roles[observed.evidence_ref_id]
            shared_group_ids = set(expected_roles) & set(actual_roles)
            groups = frozenset(
                group_id
                for group_id in shared_group_ids
                if expected_roles[group_id] is EvidenceMatchRole.DISCRIMINATIVE
                and actual_roles[group_id] is EvidenceMatchRole.DISCRIMINATIVE
            )
            contextual_groups = frozenset(shared_group_ids - set(groups))
            direct = expected_id == actual_id
            if not direct and not groups and not contextual_groups:
                continue
            derived = not direct and _is_derived_lineage_match(
                expected=expected,
                observed=observed,
                lineage=lineage,
            )
            kind = (
                EvidenceMatchKind.CONTEXTUAL
                if contextual_groups and not groups
                else EvidenceMatchKind.DIRECT
                if direct
                else EvidenceMatchKind.DERIVED
                if derived
                else EvidenceMatchKind.GROUP
            )
            matched_groups.update(groups)
            match_observations.setdefault(
                (expected.evidence_id, observed.evidence_id), []
            ).append((expected, observed, kind, groups, direct, derived))

    matches: list[ProgramEvidenceMatch] = []
    for (expected_evidence_id, actual_evidence_id), observations in sorted(
        match_observations.items()
    ):
        expected, observed, kind, _, _, _ = min(
            observations,
            key=lambda item: (
                _MATCH_KIND_PRIORITY[item[2]],
                item[1].evidence_type is not item[0].evidence_type,
                item[1].rank,
                item[1].step_id,
                item[1].evidence_ref_id,
            ),
        )
        group_ids = sorted(set().union(*(set(item[3]) for item in observations)))
        direct_match = any(item[4] for item in observations)
        derived_match = not direct_match and any(item[5] for item in observations)
        contextual_only = not group_ids
        matches.append(
            ProgramEvidenceMatch(
                match_id=stable_compiler_id(
                    "program-evidence-match",
                    [assessment_id, expected_evidence_id, actual_evidence_id],
                ),
                assessment_id=assessment_id,
                run_id=trace.run_id,
                expected_evidence_id=expected_evidence_id,
                actual_evidence_id=actual_evidence_id,
                expected_evidence_type=expected.evidence_type,
                actual_evidence_type=observed.evidence_type,
                match_kind=kind,
                evidence_group_id=group_ids[0] if group_ids else None,
                direct_match=direct_match,
                derived_match=derived_match,
                contextual_only=contextual_only,
                discriminative_match=bool(group_ids),
                metadata={
                    "collapsed_observation_count": len(observations),
                    "observed_evidence_ref_ids": sorted(
                        {item[1].evidence_ref_id for item in observations}
                    ),
                    "observed_evidence_types": sorted(
                        {item[1].evidence_type.value for item in observations}
                    ),
                    "observed_step_ids": sorted(
                        {item[1].step_id for item in observations}
                    ),
                    "evidence_group_ids": group_ids,
                },
            )
        )

    step_names = {item.step_id: item.op_name for item in result.steps}
    first_evidence_step: dict[tuple[str | None, ...], int] = {}
    for observed in sorted(
        actual,
        key=lambda item: (item.step_id, item.rank, item.evidence_ref_id),
    ):
        first_evidence_step.setdefault(
            _execution_evidence_identity(observed), observed.step_id
        )

    def contributed_group_steps(op_name: str) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    observed.step_id
                    for observations in match_observations.values()
                    for (
                        _expected,
                        observed,
                        _kind,
                        groups,
                        _direct,
                        _derived,
                    ) in observations
                    if groups
                    and step_names.get(observed.step_id) == op_name
                    and first_evidence_step[_execution_evidence_identity(observed)]
                    == observed.step_id
                }
            )
        )

    context_contribution_steps = contributed_group_steps(
        "retrieve_context_around_image"
    )
    temporal_contribution_steps = contributed_group_steps("expand_temporal_neighbors")

    step_outputs = [item.output for item in result.steps]
    structural_sufficient = any(
        isinstance(output.get("metadata"), dict) and output["metadata"].get("sufficient") is True
        for output in step_outputs
    )
    raw_image_reached = any(
        item.evidence_type is EvidenceType.RAW_IMAGE and item.image_id is not None
        for item in actual
    )
    memory_text_available = any(item.memory_id for item in actual)
    caption_available = any(item.caption_id for item in actual)
    text_available = memory_text_available or caption_available
    image_available = any(item.image_id for item in actual)
    modality_capabilities = {
        Modality.TEXT: text_available,
        Modality.CAPTION: caption_available,
        Modality.IMAGE: image_available,
        Modality.RAW_IMAGE: raw_image_reached,
        Modality.MULTIMODAL: text_available and image_available,
        Modality.TEXT_EMBEDDING: False,
        Modality.IMAGE_EMBEDDING: False,
        Modality.JOINT_MM_EMBEDDING: False,
        Modality.EMBEDDING: False,
    }
    modality_coverage = {
        modality.value: modality_capabilities.get(modality, False)
        for modality in candidate.required_modalities
    }
    visual_requires_raw = candidate.question_type in _VISUAL_QUESTION_TYPES
    complete = bool(discriminative_expected_groups) and (
        discriminative_expected_groups <= matched_groups
    )
    missing = sorted(discriminative_expected_groups - matched_groups)
    recall = (
        len(matched_groups & discriminative_expected_groups)
        / len(discriminative_expected_groups)
        if discriminative_expected_groups
        else None
    )
    evidence_sufficient = (
        structural_sufficient
        and all(modality_coverage.values())
        and (not visual_requires_raw or raw_image_reached)
        and (not require_expected_complete or complete)
    )
    assessment = ProgramDryRunAssessment(
        assessment_id=assessment_id,
        run_id=trace.run_id,
        compiler_unit_id=trace.compiler_unit_id or "unscoped",
        qbank_id=candidate.qbank_id,
        plan_id=trace.plan_id,
        execution_id=trace.execution_id,
        status=trace.status,
        control_action=trace.control_action,
        execution_success=trace.status is TraceStatus.SUCCESS,
        nonempty_evidence=bool(actual),
        evidence_sufficient=evidence_sufficient,
        expected_group_count=len(discriminative_expected_groups),
        matched_group_count=len(
            matched_groups & discriminative_expected_groups
        ),
        expected_evidence_recall=recall,
        expected_evidence_complete=complete,
        modality_coverage=modality_coverage,
        raw_image_reached=raw_image_reached,
        context_operator_used=any(
            item.op_name == "retrieve_context_around_image" for item in result.steps
        ),
        temporal_operator_used=any(
            item.op_name == "expand_temporal_neighbors" for item in result.steps
        ),
        context_evidence_contributed=bool(context_contribution_steps),
        temporal_evidence_contributed=bool(temporal_contribution_steps),
        successful_step_count=sum(
            item.status is TraceStatus.SUCCESS for item in result.steps
        ),
        required_modality_sufficient=all(modality_coverage.values()),
        discriminative_group_count=len(discriminative_expected_groups),
        matched_discriminative_group_count=len(
            matched_groups & discriminative_expected_groups
        ),
        contextual_match_count=sum(
            item.match_kind is EvidenceMatchKind.CONTEXTUAL for item in matches
        ),
        pre_control_sufficient=structural_sufficient,
        fallback_triggered=trace.control_action.value == "fallback",
        abstain_triggered=trace.control_action.value == "abstain",
        missing_evidence_group_ids=missing,
        failure_code=(
            trace.error_type
            if trace.status is TraceStatus.FAILED
            else "raw_image_unavailable"
            if visual_requires_raw and not raw_image_reached
            else "required_modality_unavailable"
            if not all(modality_coverage.values())
            else "evidence_structurally_insufficient"
            if not structural_sufficient
            else "expected_evidence_incomplete"
            if not complete
            else None
        ),
        elapsed_ms=trace.elapsed_ms,
        content_hash=canonical_content_hash(
            {
                "execution_id": trace.execution_id,
                "candidate": candidate.qbank_id,
                "expected_groups": sorted(discriminative_expected_groups),
                "contextual_expected_groups": sorted(
                    contextual_expected_groups
                ),
                "matched_groups": sorted(matched_groups),
                "context_contribution_steps": context_contribution_steps,
                "temporal_contribution_steps": temporal_contribution_steps,
                "control": trace.control_action.value,
            }
        ),
        metadata={
            "match_protocol": (
                "evidence-lineage-v4"
                if require_expected_complete
                else "evidence-lineage-v2"
            ),
            "assessment_policy": (
                "program-dry-run-assessment-v3"
                if require_expected_complete
                else "program-dry-run-assessment-v2"
            ),
            "deterministic_proxy": True,
            "contextual_expected_group_count": len(
                contextual_expected_groups
            ),
            "candidate_refusal": candidate.question_type
            is QuestionType.ANSWER_REFUSAL,
            "context_contribution_step_ids": list(context_contribution_steps),
            "temporal_contribution_step_ids": list(temporal_contribution_steps),
        },
    )
    return assessment, tuple(matches)


def dry_run_outcome_status(assessment: ProgramDryRunAssessment) -> PlannerOutcomeStatus:
    if not assessment.execution_success:
        return PlannerOutcomeStatus.EXECUTION_FAILED
    if assessment.control_action.value == "fallback":
        return PlannerOutcomeStatus.PROGRAM_READY_WITH_FALLBACK
    if assessment.control_action.value == "abstain":
        return (
            PlannerOutcomeStatus.PROGRAM_READY_WITH_ABSTAIN
            if assessment.metadata.get("candidate_refusal") is True
            else PlannerOutcomeStatus.DRY_RUN_INSUFFICIENT
        )
    if (
        not assessment.nonempty_evidence
        or not assessment.evidence_sufficient
        or not assessment.required_modality_sufficient
        or not assessment.pre_control_sufficient
    ):
        return PlannerOutcomeStatus.DRY_RUN_INSUFFICIENT
    return PlannerOutcomeStatus.PROGRAM_READY
