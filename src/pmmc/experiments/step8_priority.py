"""PMMC-first native phases, without completing deferred all-method DAG jobs.

The fallback is the existing suite child, not a second experiment. Later all-method
execution reuses its link, query phase facts and scope-independent Judge calls.
"""

from pmmc.baselines.rag_workflow import RagRunPhase
from pmmc.baselines.suite import run_baseline_suite
from pmmc.evaluation.step8_workflow import evaluate_step8_query
from pmmc.schemas import BaselineMethod


async def complete_primary_query(backend, answer_job):
    """Finish a generated answer with the unchanged fallback/selection contracts."""
    if answer_job.operation != "query_answer" or answer_job.parameters.get("treatment") != "main":
        raise ValueError("pmmc_priority_requires_main_answer_job")
    await backend.preflight(answer_job)
    scope = answer_job.parameters
    suite_id = str(backend._parent_value(answer_job, "suite_id"))
    await run_baseline_suite(
        settings=backend._answer_settings(str(scope["answer_model"])),
        suite_id=suite_id,
        phase=RagRunPhase.ANSWER,
        methods=[BaselineMethod.NATIVE_MULTIMODAL_RAG],
    )
    return await backend._query_phase(answer_job, scope, "select")


async def evaluate_primary_query(backend, answer_job, selected):
    """Score all PMMC assignments, including rejected/missing cells, in this scope."""
    await backend.preflight(answer_job)
    suite_id = str(backend._parent_value(answer_job, "suite_id"))
    run_id = str(backend._parent_value(answer_job, "query_executor_run_id"))
    if (selected["suite_id"] != suite_id or selected["phase"] != "select"
        or selected["query_executor_run_id"] != run_id):
        raise ValueError("pmmc_priority_selection_identity_mismatch")
    return await evaluate_step8_query(
        backend.settings, backend.campaign.campaign_id,
        selected["query_executor_run_id"], suite_id, str(answer_job.parameters["query_id"]),
        method_scope="pmmc",
    )
