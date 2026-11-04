"""Small evaluator-owned DB bridge for Step8 semantic pair sharing.

Prepare before the consuming evaluation's config hash; execute only after its
registration and successful-report cache lookup. The shared parent is an append-
only pair-call ledger, not a completed answer/matrix evaluation. Each consuming
report must retain and verify its own exact cell/pair denominator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import Table, text
from sqlalchemy.dialects.postgresql import insert

from pmmc.evaluation.qbank_semantic import qbank_semantic_prompt_hash
from pmmc.evaluation.step8_process_metrics import ProcessFacts, load_process_facts
from pmmc.evaluation.step8_qbank_semantic import (
    JUDGE_ALIAS,
    VERSION,
    SemanticEvaluationOwner,
    Step8SemanticConfig,
    Step8SemanticError,
    Step8SemanticPlan,
    Step8SemanticReport,
    build_step8_semantic_owner,
    build_step8_semantic_plan,
    evaluate_step8_semantic_plan,
)
from pmmc.infrastructure.database.compiler_models import CompilerRunModel
from pmmc.infrastructure.database.evaluation_models import (
    EvaluationRunModel,
    EvaluationRunStateModel,
)
from pmmc.infrastructure.database.experiment_models import RunQueryModel
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.models import BenchmarkQueryModel
from pmmc.infrastructure.database.query_executor_models import QueryExecutorRunModel
from pmmc.infrastructure.database.repositories import PostgresImageRepository
from pmmc.models import ModelExecutionService
from pmmc.models.protocols import GenerationGateway
from pmmc.models.registry import ModelRegistry
from pmmc.schemas import DataScope, QueryGold, QueryInput
from pmmc.schemas.base import utc_now
from pmmc.schemas.step8_process import ProcessCellBinding

_EVALUATOR_ROLE = text(
    "SELECT pg_has_role(current_user, 'pmmc_evaluator', 'USAGE') "
    "AND NOT pg_has_role(current_user, 'pmmc_runtime', 'USAGE')"
)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Step8SemanticError(code)


async def _require_evaluator(session) -> None:
    _require(await session.scalar(_EVALUATOR_ROLE) is True, "semantic_evaluator_db_role_required")


@dataclass(frozen=True)
class PreparedQuerySemantics:
    plan: Step8SemanticPlan
    config: Step8SemanticConfig
    consumer_query_run_id: str | None = None

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "protocol_version": VERSION,
            "protocol_hash": self.plan.protocol_hash,
            "plan_hash": self.plan.content_hash,
            "owner_scope_hash": self.plan.owner_scope_hash,
            "judge_model_alias": JUDGE_ALIAS,
            "judge_model_hash": self.plan.judge_model_hash,
            "compiler_run_id": self.plan.compiler_run_id,
            "snapshot_id": self.plan.snapshot_id,
            "query_id": self.plan.query.query_id,
            "pair_count": len(self.plan.pairs),
            "cells": [cell.model_dump(mode="json") for cell in self.plan.cells],
        }


async def prepare_query_semantics(
    *,
    sessions,
    query: QueryInput,
    snapshot_id: str,
    query_run_id: str,
    bindings: Sequence[ProcessCellBinding],
    gold: QueryGold | None,
    registry: ModelRegistry,
    config: Step8SemanticConfig | None = None,
    facts: ProcessFacts | None = None,
) -> PreparedQuerySemantics:
    """Read-only preparation for the consuming evaluation's immutable identity.

    Pass only PMMC bindings (all routing policies may share b/r). Missing gold,
    incomplete lineage or missing image checksums raise a code-only error; callers
    must report unavailable/incomplete, never substitute evidence proxy scores.
    """
    config = config or Step8SemanticConfig()
    async with sessions() as session:
        await _require_evaluator(session)
    if facts is None:
        facts = await load_process_facts(sessions, query=query, query_run_id=query_run_id)
    _require(
        facts.load_error is None and facts.run is not None, "semantic_process_facts_unavailable"
    )
    assert facts.run is not None
    async with sessions() as session:
        compiler = await session.get(CompilerRunModel, facts.run["source_compiler_run_id"])
        _require(compiler is not None, "semantic_compiler_missing")
        compiler_version = compiler.compiler_schema_version
    _require(gold is not None and bool(gold.answer), "semantic_gold_answer_missing")
    assert gold is not None and gold.answer is not None
    image_ids = {
        ref["image_id"] for refs in facts.support.values() for ref in refs if ref.get("image_id")
    } | {image.image_id for image in query.query_image_refs}
    images = {
        image.image_id: image
        for image in await PostgresImageRepository(sessions).get_many(
            snapshot_id, sorted(image_ids)
        )
    }
    query = query.model_copy(
        update={
            "query_image_refs": [images[image.image_id] for image in query.query_image_refs],
        }
    )
    candidate_images = {
        qbank_id: [
            images[image_id]
            for image_id in sorted({ref["image_id"] for ref in refs if ref.get("image_id")})
        ]
        for qbank_id, refs in facts.support.items()
    }
    plan = build_step8_semantic_plan(
        query=query,
        snapshot_id=snapshot_id,
        query_run_id=query_run_id,
        compiler_schema_version=compiler_version,
        facts=facts,
        bindings=bindings,
        gold_answer=gold.answer,
        evidence_images=candidate_images,
        judge_model_hash=registry.resolved_hash_for_aliases([JUDGE_ALIAS]),
        config=config,
    )
    # Validate the proposed source before it can enter an evaluation config hash.
    async with sessions() as session:
        await _validate_owner_query_run(session, plan, query_run_id)
    return PreparedQuerySemantics(plan, config, query_run_id)


async def _validate_owner_query_run(session, plan: Step8SemanticPlan, run_id: str) -> None:
    run = await session.get(QueryExecutorRunModel, run_id)
    _require(
        run is not None
        and run.schema_version == "0.3.0"
        and run.status == "completed"
        and run.source_compiler_run_id == plan.compiler_run_id
        and run.snapshot_id == plan.snapshot_id
        and run.dataset == plan.query.dataset.value
        and run.completed_at is not None,
        "semantic_owner_query_run_not_completed_or_wrong_scope",
    )
    membership = await session.get(RunQueryModel, (run_id, plan.snapshot_id, plan.query.query_id))
    query = await session.get(BenchmarkQueryModel, (plan.snapshot_id, plan.query.query_id))
    _require(
        membership is not None
        and query is not None
        and query.conversation_id == plan.query.conversation_id
        and query.available_after_session == plan.query.available_after_session
        and query.dataset == plan.query.dataset.value
        and query.raw_query == plan.query.raw_query,
        "semantic_owner_query_membership_mismatch",
    )


async def register_semantic_owner(
    prepared: PreparedQuerySemantics, *, sessions
) -> SemanticEvaluationOwner:
    """First writer chooses provenance; scientific pair identity is order-independent.

    The transaction is finished before model dispatch. No compiler run is disguised
    as a query run, and no legacy semantic coverage parent or schema is fabricated.
    """
    plan, config = prepared.plan, prepared.config
    # A shared matcher can finish matching before its own answer/select stages.
    # Only the current consumer is required to be terminal at evaluation time.
    proposed = build_step8_semantic_owner(
        plan,
        canonical_query_run_id=prepared.consumer_query_run_id or plan.canonical_query_run_id,
    )
    identity = {
        "adapter": VERSION,
        "semantic_pair_ledger": True,
        "scope_hash": plan.owner_scope_hash,
        "protocol_hash": plan.protocol_hash,
        "judge_model_hash": plan.judge_model_hash,
        "compiler_run_id": plan.compiler_run_id,
        "snapshot_id": plan.snapshot_id,
        "query_id": plan.query.query_id,
        "conversation_id": plan.query.conversation_id,
        "cutoff_session_id": plan.query.available_after_session,
    }
    async with sessions.begin() as session:
        await _require_evaluator(session)
        await _validate_owner_query_run(session, plan, proposed.query_run_id)
        await session.execute(
            insert(cast(Table, EvaluationRunModel.__table__))
            .values(
                evaluation_run_id=proposed.evaluation_run_id,
                source_kind="pmmc_query_run",
                suite_id=None,
                legacy_source_run_id=proposed.query_run_id,
                protocol_version=VERSION,
                normalizer_version="semantic-dimensions-only-v1",
                evidence_policy_version="not_applicable",
                bad_case_rule_version="not_evaluated",
                judge_model_alias=JUDGE_ALIAS,
                judge_prompt_hash=qbank_semantic_prompt_hash(config.renderer),
                judge_rubric_version=VERSION,
                judge_repeat_count=1,
                # Ledger rubric CHECK; semantic pairs remain dimensions-only.
                judge_score_values=[0.0, 0.5, 1.0],
                secondary_judge_enabled=False,
                metric_versions={"semantic_cover": VERSION, "semantic_match_at_k": VERSION},
                config_hash=plan.owner_scope_hash,
                created_at=utc_now(),
                metadata=identity,
            )
            .on_conflict_do_nothing(index_elements=["evaluation_run_id"])
        )
        stored = await session.get(EvaluationRunModel, proposed.evaluation_run_id)
        _require(
            stored is not None
            and stored.config_hash == plan.owner_scope_hash
            and stored.metadata_json == identity
            and stored.source_kind == "pmmc_query_run"
            and stored.protocol_version == VERSION
            and stored.judge_model_alias == JUDGE_ALIAS
            and stored.legacy_source_run_id is not None,
            "semantic_owner_registration_conflict",
        )
        assert stored is not None and stored.legacy_source_run_id is not None
        await _validate_owner_query_run(session, plan, stored.legacy_source_run_id)
        await _register_ledger_state(session, proposed.evaluation_run_id)
        return build_step8_semantic_owner(plan, canonical_query_run_id=stored.legacy_source_run_id)


async def _register_ledger_state(session, evaluation_run_id: str) -> None:
    # Each ledger owns one validated query, not an answer evaluation. Keep its
    # envelope count separate from exact cell/pair denominators in consumer reports.
    metadata = {
        "adapter": VERSION,
        "semantic_pair_ledger": True,
        "completion_claim": False,
        "counter_semantics": "single_query_ledger_not_answer_completion",
        "scientific_denominators": "consumer_report_exact_cell_pools",
    }
    await session.execute(
        insert(cast(Table, EvaluationRunStateModel.__table__))
        .values(
            evaluation_run_id=evaluation_run_id,
            status="running",
            selected_count=1,
            target_count=1,
            succeeded_count=0,
            failed_count=0,
            started_at=utc_now(),
            metadata=metadata,
        )
        .on_conflict_do_nothing(index_elements=["evaluation_run_id"])
    )
    state = await session.get(EvaluationRunStateModel, evaluation_run_id)
    _require(
        state is not None
        and state.status == "running"
        and state.selected_count == 1
        and state.target_count == 1
        and state.succeeded_count == state.failed_count == 0
        and state.completed_at is None
        and state.metadata_json == metadata,
        "semantic_ledger_state_conflict",
    )


async def evaluate_query_semantics(
    prepared: PreparedQuerySemantics,
    *,
    sessions,
    execution: ModelExecutionService,
    gateway: GenerationGateway,
) -> Step8SemanticReport:
    """Post-registration/cache bridge, sharing the actual audited Judge service.

    The caller includes the returned report in its evaluator-only durable score
    artifact and completion checks. An incomplete report must block publication.
    """
    _require(execution.data_scope is DataScope.EVALUATION, "semantic_requires_evaluator_execution")
    owner = await register_semantic_owner(prepared, sessions=sessions)

    async def load_trace(call_id):
        async with ExperimentUnitOfWork(sessions, DataScope.EVALUATION) as uow:
            return await uow.model_calls.get(call_id)

    return await evaluate_step8_semantic_plan(
        prepared.plan,
        owner=owner,
        execution=execution,
        gateway=gateway,
        load_trace=load_trace,
        config=prepared.config,
    )
