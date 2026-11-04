"""Cumulative, bounded Step8 typed program revisions (no model calls)."""

from __future__ import annotations

from collections.abc import Sequence

from pmmc.compiler.execution import budget_program_steps
from pmmc.compiler.ids import (
    memory_program_content_hash,
    stable_compiler_id,
)
from pmmc.compiler.planner import _CAPABILITY_OUTPUTS, _v12_step
from pmmc.compiler.planner_config import PlannerConfig
from pmmc.schemas.doubter import (
    DoubterDirective,
    DoubterDirectiveCode,
    PlannerDirectiveEffect,
    PlannerDirectiveStatus,
)
from pmmc.schemas.planner import PlannerCapability, PlannerRequirementContract
from pmmc.schemas.program import MemoryProgram, ProgramStep

_DIRECTIVE_OP = {
    DoubterDirectiveCode.ADD_TEXT_RETRIEVAL: "retrieve_text_by_embedding",
    DoubterDirectiveCode.ADD_VISUAL_RETRIEVAL: "retrieve_image_by_visual_embedding",
    DoubterDirectiveCode.ADD_IMAGE_TEXT_PAIR: "retrieve_image_text_pair",
    DoubterDirectiveCode.ADD_CONTEXT: "retrieve_context_around_image",
    DoubterDirectiveCode.ADD_TEMPORAL: "expand_temporal_neighbors",
    DoubterDirectiveCode.ADD_RAW_IMAGE_INSPECTION: "inspect_raw_image",
}
_ROOTS = {
    "retrieve_text_by_keyword",
    "retrieve_text_by_embedding",
    "retrieve_image_by_visual_embedding",
}


def _capabilities(steps: Sequence[ProgramStep]) -> list[PlannerCapability]:
    return sorted(set().union(*(_CAPABILITY_OUTPUTS.get(s.op_name, set()) for s in steps)))


def _top_k(steps: Sequence[ProgramStep]) -> dict[int, int]:
    return {s.step_id: int(s.args["top_k"]) for s in steps if "top_k" in s.args}


def _semantic_steps(steps: Sequence[ProgramStep]) -> list[object]:
    return [(s.op_name, s.args, s.expected_output_type) for s in steps]


def _rewire(steps: Sequence[ProgramStep], replacements: dict[int, int]) -> list[ProgramStep]:
    mapping = {s.step_id: index for index, s in enumerate(steps)}
    for old, new in replacements.items():
        mapping[old] = mapping[new]
    result = []
    for index, step in enumerate(steps):
        args = dict(step.args)
        if "source_step_ids" in args:
            args["source_step_ids"] = sorted({mapping[i] for i in args["source_step_ids"]})
        result.append(step.model_copy(update={"step_id": index, "args": args}))
    return result


def _append_before_tail(
    steps: list[ProgramStep], op: str, args: dict[str, object]
) -> list[ProgramStep]:
    new_id = max(s.step_id for s in steps) + 1
    addition = _v12_step(new_id, op, args, server_role="revision")
    tail = next(
        (i for i, s in enumerate(steps) if s.op_name == "verify_evidence_sufficiency"), len(steps)
    )
    result = [*steps[:tail], addition, *steps[tail:]]
    result = [
        s.model_copy(
            update={
                "args": {
                    **s.args,
                    "source_step_ids": [*s.args["source_step_ids"], new_id],
                }
            }
        )
        if s.op_name == "verify_evidence_sufficiency"
        else s
        for s in result
    ]
    return _rewire(result, {})


def apply_planner_directives(
    *,
    source_program: MemoryProgram,
    requirement: PlannerRequirementContract,
    directives: Sequence[DoubterDirective],
    config: PlannerConfig,
) -> tuple[MemoryProgram | None, list[PlannerDirectiveEffect]]:
    """Apply each directive transactionally; failed directives leave preceding effects intact."""
    if not config.is_v15:
        raise ValueError("native cumulative patches require Planner v1.5")
    if source_program.operator_set_version != config.operator_set_version:
        raise ValueError("revision operator identity mismatch")
    if (
        requirement.qbank_id != source_program.qbank_id
        or requirement.run_id != source_program.run_id
    ):
        raise ValueError("revision requirement lineage mismatch")
    if source_program.round_id >= 3:
        raise ValueError("revision depth exhausted")
    current = source_program
    effects: list[PlannerDirectiveEffect] = []
    allowed = set(requirement.allowed_capabilities)
    if not set(requirement.required_capabilities) <= allowed:
        raise ValueError("invalid required/allowed capability contract")
    if len({d.directive_id for d in directives}) != len(directives):
        raise ValueError("duplicate directive identity")
    for directive in sorted(directives, key=lambda d: (d.ordinal, d.directive_id)):
        if (directive.run_id, directive.qbank_id, directive.round_id) != (
            source_program.run_id,
            source_program.qbank_id,
            source_program.round_id + 1,
        ):
            raise ValueError("revision directive lineage mismatch")
        before = current
        steps = list(before.operator_sequence)
        status = PlannerDirectiveStatus.NOOP
        reason = "no_applicable_dataflow_change"
        source_ids: list[int] = []
        op = _DIRECTIVE_OP.get(directive.directive)
        if config.planning_mode == "fixed":
            status, reason = PlannerDirectiveStatus.INAPPLICABLE, "fixed_planner_verification_only"
        elif op is not None:
            needed = set(_CAPABILITY_OUTPUTS[op]) - {
                PlannerCapability.EVIDENCE_GROUPS,
            }
            if not needed <= allowed or (
                op == "inspect_raw_image" and config.memory_raw_image_policy == "forbid"
            ):
                status, reason = PlannerDirectiveStatus.INAPPLICABLE, "capability_not_allowed"
            elif any(s.op_name == op for s in steps):
                status, reason = (
                    PlannerDirectiveStatus.ALREADY_SATISFIED,
                    "operator_already_present",
                )
            else:
                args: dict[str, object]
                if op in _ROOTS:
                    args = {
                        "top_k": max(_top_k(steps).values(), default=5),
                        "vector_role": "question_joint_query"
                        if "visual" in op
                        else "question_text_query",
                    }
                else:
                    image_input = op != "expand_temporal_neighbors"
                    source_ids = [
                        s.step_id
                        for s in steps
                        if s.expected_output_type == ("image_ids" if image_input else "memory_ids")
                        and s.op_name != "inspect_raw_image"
                    ]
                    if not source_ids:
                        prerequisite = (
                            "retrieve_image_by_visual_embedding"
                            if image_input
                            else "retrieve_text_by_embedding"
                        )
                        cap = PlannerCapability.IMAGE if image_input else PlannerCapability.TEXT
                        if cap in allowed:
                            steps = _append_before_tail(
                                steps,
                                prerequisite,
                                {
                                    "top_k": 5,
                                    "vector_role": "question_joint_query"
                                    if image_input
                                    else "question_text_query",
                                },
                            )
                            source_ids = [s.step_id for s in steps if s.op_name == prerequisite]
                    args = {"source_step_ids": source_ids}
                    if op in {"retrieve_context_around_image", "expand_temporal_neighbors"}:
                        args["radius"] = config.expansion_radius_max
                if op not in _ROOTS and not source_ids:
                    status, reason = (
                        PlannerDirectiveStatus.INAPPLICABLE,
                        "input_capability_unavailable",
                    )
                else:
                    steps = _append_before_tail(steps, op, args)
        elif directive.directive is DoubterDirectiveCode.INCREASE_TOP_K:
            steps = [
                s.model_copy(
                    update={
                        "args": {
                            **s.args,
                            "top_k": max(int(s.args["top_k"]), int(directive.top_k_tier or 1)),
                        }
                    }
                )
                if "top_k" in s.args
                else s
                for s in steps
            ]
            source_ids = list(_top_k(steps))
            status, reason = PlannerDirectiveStatus.SATURATED, "top_k_or_evidence_budget_saturated"
        elif directive.directive is DoubterDirectiveCode.REMOVE_REDUNDANT_RETRIEVAL:
            roots = [s for s in steps if s.op_name in _ROOTS]
            replacements: dict[int, int] = {}
            for index, root in enumerate(roots):
                equivalent = next(
                    (
                        other
                        for other in roots[:index]
                        if (other.op_name == root.op_name and other.args == root.args)
                    ),
                    None,
                )
                if equivalent is None and root.op_name == "retrieve_text_by_keyword":
                    equivalent = next(
                        (s for s in roots if s.op_name == "retrieve_text_by_embedding"), None
                    )
                if equivalent is not None:
                    replacements[root.step_id] = equivalent.step_id
            if replacements:
                source_ids = sorted(replacements)
                # Resolve chains before reindexing consumers.
                for key, value in replacements.items():
                    while value in replacements:
                        value = replacements[value]
                    replacements[key] = value
                steps = _rewire([s for s in steps if s.step_id not in replacements], replacements)
            else:
                status, reason = (
                    PlannerDirectiveStatus.ALREADY_SATISFIED,
                    "no_redundant_retrieval_root",
                )
        changed = _semantic_steps(steps) != _semantic_steps(before.operator_sequence)
        if changed and status is not PlannerDirectiveStatus.INAPPLICABLE:
            try:
                if len(steps) > config.program_max_steps:
                    raise ValueError("operator_step_budget_saturated")
                steps, budget = budget_program_steps(
                    steps,
                    total_evidence_max=config.total_evidence_max,
                    top_k_max=config.operator_top_k_max,
                )
                if _semantic_steps(steps) == _semantic_steps(before.operator_sequence):
                    status, reason = PlannerDirectiveStatus.SATURATED, "evidence_budget_saturated"
                else:
                    identity = [
                        source_program.plan_id,
                        source_program.round_id + 1,
                        _semantic_steps(steps),
                    ]
                    current = before.model_copy(
                        update={
                            "plan_id": stable_compiler_id("memory-program", identity),
                            "round_id": source_program.round_id + 1,
                            "producer_model_call_id": None,
                            "operator_sequence": steps,
                            "metadata": {
                                **before.metadata,
                                "evidence_budget": budget.model_dump(mode="json"),
                                "effective_capabilities": [c.value for c in _capabilities(steps)],
                                "program_origin": "server_typed_patch",
                            },
                        }
                    )
                    current = current.model_copy(
                        update={"program_hash": memory_program_content_hash(current)}
                    )
                    status, reason = PlannerDirectiveStatus.APPLIED, "dataflow_changed"
            except ValueError as exc:
                status, reason = PlannerDirectiveStatus.SATURATED, str(exc)
        effects.append(
            PlannerDirectiveEffect(
                directive_id=directive.directive_id,
                directive=directive.directive,
                status=status,
                reason_code=reason,
                source_step_ids=source_ids,
                before_capabilities=_capabilities(before.operator_sequence),
                after_capabilities=_capabilities(current.operator_sequence),
                before_top_k=_top_k(before.operator_sequence),
                after_top_k=_top_k(current.operator_sequence),
                before_program_hash=before.program_hash,
                after_program_hash=current.program_hash,
            )
        )
    if current is source_program:
        return None, effects
    return current.model_copy(
        update={
            "metadata": {
                **current.metadata,
                "directive_effects": [e.model_dump(mode="json") for e in effects],
                "source_plan_id": source_program.plan_id,
            }
        }
    ), effects
