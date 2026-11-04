"""Deterministic Step 6D decisions around evidence-grounded model judgments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.schemas import (
    DoubterAction,
    DoubterDimensions,
    DoubterDimensionVerdict,
    DoubterDirective,
    DoubterDirectiveCode,
    DoubterIssueCode,
    DoubterProviderResult,
    DoubterProviderResultV11,
    PlannerCapability,
    PlannerRequiredModality,
    PlannerRequirementContract,
    ProgramDryRunAssessment,
    QBankRouteClass,
    QuestionCandidate,
)

_DIRECT_REJECT_ISSUES = {
    DoubterIssueCode.CANDIDATE_UNSUPPORTED,
    DoubterIssueCode.ANSWER_NOT_ENTAILED,
    DoubterIssueCode.REFUSAL_INAPPROPRIATE,
    DoubterIssueCode.STRUCTURED_OUTPUT_INVALID,
}


def normalize_v11_dimensions(
    *,
    result: DoubterProviderResultV11,
    route_class: QBankRouteClass,
    requirement: PlannerRequirementContract,
) -> DoubterDimensions:
    """Validate applicability and map v1.1 tri-state output to persisted booleans."""

    raw = result.dimensions
    applicable = doubter_dimension_applicability(
        route_class=route_class,
        requirement=requirement,
    )
    values: dict[str, bool] = {}
    for name, is_applicable in applicable.items():
        verdict = getattr(raw, name)
        if is_applicable and verdict is DoubterDimensionVerdict.NOT_APPLICABLE:
            raise ValueError(f"applicable Doubter dimension cannot be not_applicable: {name}")
        if not is_applicable and verdict is not DoubterDimensionVerdict.NOT_APPLICABLE:
            raise ValueError(f"inapplicable Doubter dimension must be not_applicable: {name}")
        values["non_redundant" if name == "retrieval_nonredundant" else name] = (
            True if not is_applicable else verdict is DoubterDimensionVerdict.PASS
        )
    return DoubterDimensions.model_validate(values)


def doubter_dimension_applicability(
    *,
    route_class: QBankRouteClass,
    requirement: PlannerRequirementContract,
) -> dict[str, bool]:
    """Return the server-owned applicability contract for the ten v1.1 dimensions."""

    return {
        "candidate_supported": True,
        "answer_entailed": route_class is QBankRouteClass.ANSWERABLE,
        "path_sufficient": route_class is QBankRouteClass.ANSWERABLE,
        "modality_sufficient": route_class is QBankRouteClass.ANSWERABLE,
        "raw_image_sufficient": (
            route_class is QBankRouteClass.ANSWERABLE
            and PlannerRequiredModality.RAW_IMAGE in requirement.required_modalities
        ),
        "context_sufficient": (
            route_class is QBankRouteClass.ANSWERABLE
            and PlannerCapability.CONTEXT in requirement.required_capabilities
        ),
        "multi_hop_sufficient": (
            route_class is QBankRouteClass.ANSWERABLE and requirement.minimum_evidence_groups > 1
        ),
        "temporal_consistent": (
            route_class is QBankRouteClass.ANSWERABLE
            and PlannerCapability.TEMPORAL in requirement.required_capabilities
        ),
        "retrieval_nonredundant": route_class is QBankRouteClass.ANSWERABLE,
        "refusal_appropriate": route_class is QBankRouteClass.CONTROL,
    }


def derive_doubter_issues(
    *,
    dimensions: DoubterDimensions,
    route_class: QBankRouteClass,
) -> list[DoubterIssueCode]:
    """Derive canonical issues from dimensions rather than trusting model labels."""

    checks: tuple[tuple[bool, DoubterIssueCode], ...]
    if route_class is QBankRouteClass.CONTROL:
        checks = (
            (dimensions.candidate_supported, DoubterIssueCode.CANDIDATE_UNSUPPORTED),
            (
                dimensions.refusal_appropriate,
                DoubterIssueCode.REFUSAL_INAPPROPRIATE,
            ),
        )
    else:
        checks = (
            (
                dimensions.candidate_supported,
                DoubterIssueCode.CANDIDATE_UNSUPPORTED,
            ),
            (dimensions.answer_entailed, DoubterIssueCode.ANSWER_NOT_ENTAILED),
            (dimensions.path_sufficient, DoubterIssueCode.PATH_INSUFFICIENT),
            (dimensions.modality_sufficient, DoubterIssueCode.MODALITY_MISSING),
            (dimensions.raw_image_sufficient, DoubterIssueCode.RAW_IMAGE_MISSING),
            (dimensions.context_sufficient, DoubterIssueCode.CONTEXT_MISSING),
            (
                dimensions.multi_hop_sufficient,
                DoubterIssueCode.MULTI_HOP_INCOMPLETE,
            ),
            (
                dimensions.temporal_consistent,
                DoubterIssueCode.TEMPORAL_INCONSISTENT,
            ),
            (dimensions.non_redundant, DoubterIssueCode.REDUNDANT_RETRIEVAL),
        )
    return [issue for passed, issue in checks if not passed]


def derive_doubter_action(
    *,
    issues: Sequence[DoubterIssueCode],
    round_id: int,
    max_round: int,
) -> DoubterAction:
    if not issues:
        return DoubterAction.PASS
    if any(issue in _DIRECT_REJECT_ISSUES for issue in issues):
        return DoubterAction.REJECT_CANDIDATE
    if round_id >= max_round:
        return DoubterAction.REJECT_CANDIDATE
    return DoubterAction.REVISE_PROGRAM


def derive_dry_run_issues(
    *,
    candidate: QuestionCandidate,
    requirement: PlannerRequirementContract,
    assessment: ProgramDryRunAssessment,
) -> list[DoubterIssueCode]:
    """Derive non-overridable issues from the real operator execution."""

    if candidate.route_class is QBankRouteClass.CONTROL:
        return (
            []
            if assessment.execution_success and assessment.abstain_triggered
            else [DoubterIssueCode.REFUSAL_INAPPROPRIATE]
        )

    issues: list[DoubterIssueCode] = []
    if (
        not assessment.execution_success
        or not assessment.nonempty_evidence
        or not assessment.pre_control_sufficient
        or not assessment.expected_evidence_complete
    ):
        issues.append(DoubterIssueCode.PATH_INSUFFICIENT)
    if not assessment.required_modality_sufficient:
        issues.append(DoubterIssueCode.MODALITY_MISSING)
    if (
        PlannerRequiredModality.RAW_IMAGE in requirement.required_modalities
        and not assessment.raw_image_reached
    ):
        issues.append(DoubterIssueCode.RAW_IMAGE_MISSING)
    if PlannerCapability.CONTEXT in requirement.required_capabilities and not (
        assessment.context_evidence_contributed
        if assessment.metadata.get("assessment_policy") == "program-dry-run-assessment-v3"
        else assessment.context_operator_used
    ):
        issues.append(DoubterIssueCode.CONTEXT_MISSING)
    if PlannerCapability.TEMPORAL in requirement.required_capabilities and not (
        assessment.temporal_evidence_contributed
        if assessment.metadata.get("assessment_policy") == "program-dry-run-assessment-v3"
        else assessment.temporal_operator_used
    ):
        issues.append(DoubterIssueCode.TEMPORAL_INCONSISTENT)
    if (
        requirement.minimum_evidence_groups > 1
        and assessment.matched_discriminative_group_count < requirement.minimum_evidence_groups
    ):
        issues.append(DoubterIssueCode.MULTI_HOP_INCOMPLETE)
    return issues


def derive_doubter_directives(
    *,
    feedback_id: str,
    run_id: str,
    candidate: QuestionCandidate,
    requirement: PlannerRequirementContract,
    source_round_id: int,
    issues: Sequence[DoubterIssueCode],
    current_top_k: int,
    strict_capabilities: bool = False,
) -> list[DoubterDirective]:
    """Map revisable evidence defects to a bounded declarative strategy change."""

    allowed = set(requirement.allowed_capabilities)
    mapping: dict[DoubterIssueCode, DoubterDirectiveCode] = {
        DoubterIssueCode.PATH_INSUFFICIENT: DoubterDirectiveCode.INCREASE_TOP_K,
        DoubterIssueCode.MODALITY_MISSING: (
            DoubterDirectiveCode.ADD_VISUAL_RETRIEVAL
            if any(item.value == "image" for item in candidate.required_modalities)
            else DoubterDirectiveCode.ADD_TEXT_RETRIEVAL
        ),
        DoubterIssueCode.RAW_IMAGE_MISSING: (DoubterDirectiveCode.ADD_RAW_IMAGE_INSPECTION),
        DoubterIssueCode.CONTEXT_MISSING: DoubterDirectiveCode.ADD_CONTEXT,
        DoubterIssueCode.MULTI_HOP_INCOMPLETE: (
            DoubterDirectiveCode.ADD_CONTEXT
            if PlannerCapability.CONTEXT in allowed
            else DoubterDirectiveCode.ADD_TEMPORAL
            if PlannerCapability.TEMPORAL in allowed
            else DoubterDirectiveCode.INCREASE_TOP_K
        ),
        DoubterIssueCode.TEMPORAL_INCONSISTENT: DoubterDirectiveCode.ADD_TEMPORAL,
        DoubterIssueCode.REDUNDANT_RETRIEVAL: (DoubterDirectiveCode.REMOVE_REDUNDANT_RETRIEVAL),
    }
    codes: list[DoubterDirectiveCode] = []
    directive_capability = {
        DoubterDirectiveCode.ADD_TEXT_RETRIEVAL: PlannerCapability.TEXT,
        DoubterDirectiveCode.ADD_VISUAL_RETRIEVAL: PlannerCapability.IMAGE,
        DoubterDirectiveCode.ADD_IMAGE_TEXT_PAIR: PlannerCapability.IMAGE_TEXT,
        DoubterDirectiveCode.ADD_CONTEXT: PlannerCapability.CONTEXT,
        DoubterDirectiveCode.ADD_TEMPORAL: PlannerCapability.TEMPORAL,
        DoubterDirectiveCode.ADD_RAW_IMAGE_INSPECTION: PlannerCapability.RAW_IMAGE,
    }
    for issue in issues:
        code = mapping.get(issue)
        capability = directive_capability.get(code) if code is not None else None
        if capability is not None and capability not in allowed and not strict_capabilities:
            # A Doubter may detect a genuine evidence-group shortfall even when
            # the immutable requirement contract intentionally disallows the
            # corresponding expansion operator. Widen retrieval in that case;
            # never emit a directive that Planner must reject by construction.
            code = DoubterDirectiveCode.INCREASE_TOP_K
        if code is not None and code not in codes:
            codes.append(code)
    target_round = source_round_id + 1
    values: list[DoubterDirective] = []
    for ordinal, code in enumerate(codes):
        top_k = (
            next(
                (value for value in (10, 20, 40, 50) if value > current_top_k),
                50,
            )
            if code is DoubterDirectiveCode.INCREASE_TOP_K
            else None
        )
        payload: Mapping[str, object] = {
            "feedback_id": feedback_id,
            "run_id": run_id,
            "qbank_id": candidate.qbank_id,
            "round_id": target_round,
            "ordinal": ordinal,
            "directive": code.value,
            "top_k_tier": top_k,
        }
        values.append(
            DoubterDirective(
                directive_id=stable_compiler_id(
                    "doubter-directive",
                    [feedback_id, target_round, ordinal, code.value, top_k],
                ),
                feedback_id=feedback_id,
                run_id=run_id,
                qbank_id=candidate.qbank_id,
                round_id=target_round,
                ordinal=ordinal,
                directive=code,
                top_k_tier=top_k,
                content_hash=canonical_content_hash(payload),
            )
        )
    return values


def enforce_v13_dimensions(
    *,
    dimensions: DoubterDimensions,
    candidate: QuestionCandidate,
    requirement: PlannerRequirementContract,
    assessment: ProgramDryRunAssessment,
    source_present: bool,
    source_spans_valid: bool,
    path_present: bool,
    inspected_path_image_present: bool,
) -> DoubterDimensions:
    """Deterministic evidence failures may veto, but never upgrade, semantic judgments."""
    updates = {
        "candidate_supported": dimensions.candidate_supported
        and source_present
        and source_spans_valid,
        "answer_entailed": dimensions.answer_entailed and source_present and source_spans_valid,
    }
    if candidate.route_class is QBankRouteClass.ANSWERABLE:
        updates.update(
            {
                "path_sufficient": dimensions.path_sufficient
                and path_present
                and assessment.execution_success
                and assessment.nonempty_evidence
                and assessment.pre_control_sufficient
                and assessment.expected_evidence_complete,
                "modality_sufficient": dimensions.modality_sufficient
                and assessment.required_modality_sufficient,
            }
        )
        if PlannerRequiredModality.RAW_IMAGE in requirement.required_modalities:
            updates["raw_image_sufficient"] = (
                dimensions.raw_image_sufficient
                and assessment.raw_image_reached
                and inspected_path_image_present
            )
        if PlannerCapability.CONTEXT in requirement.required_capabilities:
            updates["context_sufficient"] = (
                dimensions.context_sufficient and assessment.context_evidence_contributed
            )
        if PlannerCapability.TEMPORAL in requirement.required_capabilities:
            updates["temporal_consistent"] = (
                dimensions.temporal_consistent and assessment.temporal_evidence_contributed
            )
        if requirement.minimum_evidence_groups > 1:
            updates["multi_hop_sufficient"] = dimensions.multi_hop_sufficient and (
                assessment.matched_discriminative_group_count >= requirement.minimum_evidence_groups
            )
    return dimensions.model_copy(update=updates)


def validate_provider_issue_consistency(
    *,
    result: DoubterProviderResult,
    route_class: QBankRouteClass,
) -> list[DoubterIssueCode]:
    """Reject contradictory provider labels while keeping decisions deterministic."""

    derived = derive_doubter_issues(
        dimensions=result.dimensions,
        route_class=route_class,
    )
    if set(derived) != set(result.issue_codes):
        raise ValueError("Doubter reason codes conflict with structured dimensions")
    return derived


__all__ = [
    "derive_doubter_action",
    "derive_doubter_directives",
    "derive_dry_run_issues",
    "derive_doubter_issues",
    "normalize_v11_dimensions",
    "validate_provider_issue_consistency",
]
