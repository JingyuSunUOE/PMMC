"""Operator-v7 evidence preflight shared by initial plans and typed revisions."""

from __future__ import annotations

from collections.abc import Sequence

from pmmc.schemas.planner import PlannerEvidenceBudget
from pmmc.schemas.program import ProgramStep


def program_evidence_quotas(
    steps: Sequence[ProgramStep], *, root_top_k: dict[int, int] | None = None
) -> tuple[dict[int, int], dict[int, int]]:
    """Reproduce v7 quotas from frozen, runtime-safe args, without private metadata."""
    values = (
        root_top_k
        if root_top_k is not None
        else {step.step_id: int(step.args["top_k"]) for step in steps if "top_k" in step.args}
    )
    rows: dict[int, int] = {}
    quotas: dict[int, int] = {}
    fanout: dict[int, int] = {}
    for step in steps:
        sources = set(step.args.get("source_step_ids", []))
        if step.step_id in rows or any(source not in rows for source in sources):
            raise ValueError("evidence budget requires unique, backward-only source steps")
        count = sum(rows[source] for source in sources)
        if step.step_id in values:
            if values[step.step_id] < 1:
                raise ValueError("retrieval top-k must be positive")
            rows[step.step_id] = quotas[step.step_id] = values[step.step_id]
        elif step.op_name in {"retrieve_context_around_image", "expand_temporal_neighbors"}:
            factor = 2 * int(step.args.get("radius", 1)) + 1
            fanout[step.step_id] = factor
            rows[step.step_id] = quotas[step.step_id] = factor * count
        elif step.op_name == "retrieve_image_text_pair":
            rows[step.step_id] = count
            fanout[step.step_id] = 2
            quotas[step.step_id] = 2 * count
        elif step.op_name == "inspect_raw_image":
            rows[step.step_id] = quotas[step.step_id] = count
        elif step.op_name == "verify_evidence_sufficiency":
            rows[step.step_id] = count
            quotas[step.step_id] = sum(quotas[source] for source in sources)
        else:
            rows[step.step_id] = count
            quotas[step.step_id] = 0
    return quotas, fanout


def budget_program_steps(
    steps: Sequence[ProgramStep],
    *,
    total_evidence_max: int,
    top_k_max: int = 50,
) -> tuple[list[ProgramStep], PlannerEvidenceBudget]:
    """Allocate bounded root quotas before running expansion operators.

    Independent roots are conservatively disjoint. Expansion source lists are sets;
    no guessed cross-channel overlap is used to justify exceeding the cap. v7
    executors must enforce the emitted per-step quotas after typed-reference dedup.
    """
    roots = {
        step.step_id: min(int(step.args["top_k"]), top_k_max)
        for step in steps
        if "top_k" in step.args
    }
    if any(value < 1 for value in roots.values()):
        raise ValueError("retrieval top-k must be positive")

    def estimate(values: dict[int, int]) -> tuple[dict[int, int], dict[int, int]]:
        return program_evidence_quotas(steps, root_top_k=values)

    quotas, fanout = estimate(roots)
    while sum(quotas.values()) > total_evidence_max:
        reducible = [key for key, value in roots.items() if value > 1]
        if not reducible:
            raise ValueError("minimum retrieval and expansion fanout exceed evidence budget")

        # Reduce the root with the largest marginal expansion cost; ties are stable.
        def saving(key: int, total: int = sum(quotas.values())) -> tuple[int, int, int]:
            trial = {**roots, key: roots[key] - 1}
            return total - sum(estimate(trial)[0].values()), roots[key], -key

        root = max(reducible, key=saving)
        roots[root] -= 1
        quotas, fanout = estimate(roots)
    budget = PlannerEvidenceBudget(
        total_evidence_max=total_evidence_max,
        total_upper_bound=sum(quotas.values()),
        step_quotas=quotas,
        root_top_k=roots,
        expansion_fanout=fanout,
    )
    allocated = [
        step.model_copy(
            update={
                "args": {
                    **step.args,
                    **({"top_k": roots[step.step_id]} if step.step_id in roots else {}),
                },
                "metadata": {**step.metadata, "evidence_quota": quotas[step.step_id]},
            }
        )
        for step in steps
    ]
    return allocated, budget
