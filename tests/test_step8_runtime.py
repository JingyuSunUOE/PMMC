from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pmmc.infrastructure.database.compiler_repositories import PostgresQBankRoutingRepository
from pmmc.infrastructure.database.errors import InvalidStateTransitionError
from pmmc.infrastructure.database.query_executor_repositories import PostgresQueryExecutorRepository
from pmmc.runtime.qbank_matcher import (
    MatchGridKey,
    RuntimeQBankChannelHit,
    apply_reranker_assessments,
    decide_runtime_route,
    rank_runtime_match_grid,
    reranker_pair_ids,
)
from pmmc.runtime.query_backend import PostgresOnlineQueryExecutorBackend
from pmmc.runtime.query_executor import OnlineQueryExecutor, RuntimeEvidenceRelevance
from pmmc.runtime.query_executor_config import QueryExecutorConfig, load_query_executor_config
from pmmc.runtime.query_workflow import _runtime_embedding_cache_identity
from pmmc.schemas import (
    Modality,
    QBankRouteClass,
    QBankRuntimeChannel,
    QueryEvidenceRelevanceAssessment,
    QueryEvidenceRelevanceItem,
    RerankerMatchLevel,
)
from pmmc.schemas.answer_output import (
    AnswerContractV3,
    AnswerRoleCall,
    GroundingAssessment,
    GroundingFact,
    TypedAnswerCandidate,
    TypedAnswerResult,
)
from pmmc.schemas.base import utc_now
from pmmc.schemas.query_executor import runtime_fact_payload
from test_step7_executor_integration import _FakeBackend
from test_step7b_executor_integration import _assessment
from test_step8_answer_contract import prepared as typed_prepared

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_query_executor_config(ROOT / "configs/runtime/query-executor-v3.yaml")


def hit(name="q1", *, round_id=0, budget=1, score=1.0, channel=QBankRuntimeChannel.TEXT, plan=None):
    return RuntimeQBankChannelHit(
        qbank_id=name,
        plan_id=plan or f"plan-{name}",
        route_class=QBankRouteClass.ANSWERABLE,
        generation_rank=budget,
        accepted_round=3,
        required_modalities=(Modality.TEXT,),
        channel=channel,
        raw_score=score,
        candidate_question=f"Question {name}?",
        projection_round=round_id,
        source_program_hash=plan or f"hash-{name}",
    )


def grid(hits, config=CONFIG):
    return rank_runtime_match_grid(
        query_executor_run_id="query-run-1",
        query_id="query-1",
        hits=hits,
        has_query_image=False,
        config=config,
    )


@pytest.mark.parametrize("rounds", [(0,), (1,), (3,)])
def test_nonmonotonic_exact_membership_is_not_accepted_round_filter(rounds):
    values = grid([hit(round_id=r) for r in rounds])
    for r in range(4):
        assert bool(values[MatchGridKey(8, r)].candidates) == (r in rounds)


def test_same_candidate_uses_the_program_of_its_exact_round():
    values = grid([hit(round_id=0, plan="p0"), hit(round_id=1, plan="p1")])
    assert values[MatchGridKey(8, 0)].candidates[0].plan_id == "p0"
    assert values[MatchGridKey(8, 1)].candidates[0].plan_id == "p1"
    assert not values[MatchGridKey(8, 2)].candidates


def test_v3_rejects_implicit_membership_and_does_not_truncate_other_cells():
    with pytest.raises(ValueError, match="exact round"):
        grid([replace(hit(), projection_round=None)])
    hits = [hit("budget1", score=-0.5)] + [hit(f"high{i}", budget=8, score=i) for i in range(70)]
    values = grid(hits)
    assert values[MatchGridKey(1, 0)].candidates[0].qbank_id == "budget1"
    assert "budget1" not in {item.qbank_id for item in values[MatchGridKey(8, 0)].candidates}


def test_missing_channel_zero_no_renormalization_and_normalized_weighted_rrf():
    one = grid([hit()])[MatchGridKey(1, 0)].candidates[0]
    assert one.base_score == pytest.approx(0.6)
    assert one.available_weight_sum == 0.6
    complete = grid([hit(), hit(channel=QBankRuntimeChannel.LEXICAL)])[
        MatchGridKey(1, 0)
    ].candidates[0]
    assert complete.base_score == pytest.approx(1.0)
    reranked = apply_reranker_assessments(
        grid=grid([hit()]),
        assessments={"q1": _assessment("q1", level=RerankerMatchLevel.STRONG)},
        config=CONFIG,
    )
    item = reranked[MatchGridKey(1, 0)].candidates[0]
    assert item.confidence == pytest.approx(0.4 * 0.6 + 0.6 * RerankerMatchLevel.STRONG.score)


def test_top20_pair_union_includes_each_independent_cell_once():
    hits = [hit(f"r{r}-{i}", round_id=r, score=i) for r in range(4) for i in range(30)]
    ids = reranker_pair_ids(grid(hits), top_k=20, separate_route_classes=True)
    assert len(ids) == 80
    assert len(set(ids)) == len(ids)


def test_no_reranker_is_a_distinct_calibrated_route_without_assessments():
    config = QueryExecutorConfig.model_validate({**CONFIG.model_dump(), "treatment": "no_reranker"})
    values = grid([hit(), hit(channel=QBankRuntimeChannel.LEXICAL)], config)
    assert apply_reranker_assessments(grid=values, assessments={}, config=config) is values
    route = decide_runtime_route(
        query_executor_run_id="query-run-1",
        query_id="query-1",
        match_batch_id="batch",
        candidates=values[MatchGridKey(8, 0)].candidates,
        config=config,
    )
    assert route.outcome.value == "program"
    assert route.match_threshold == 0.8


def typed_answer(identity, *, abstain=False, run_id="query-run-1"):
    output = AnswerContractV3(
        answer_kind="abstention" if abstain else "text",
        answer_text=None if abstain else "Blue",
        selected_image_aliases=[],
        supported=not abstain,
        used_evidence_aliases=[] if abstain else ["A01"],
        abstain_reason="insufficient_evidence" if abstain else None,
    )
    grounding = (
        None
        if abstain
        else GroundingFact(
            grounding_id=f"ground-{identity}",
            run_id=run_id,
            query_id="query-1",
            answer_model_call_id=f"call-{identity}",
            model_alias="grounding-qwen3.5-9b",
            actual_model_id="test-grounder",
            assessment=GroundingAssessment(
                supported=True,
                query_closure="pass",
                entity_consistency="pass",
                time_consistency="not_applicable",
                set_consistency="not_applicable",
                source_support="pass",
            ),
            used_evidence_aliases=["A01"],
            evidence_content_hash="source-hash",
            ungated_output_hash="output-hash",
            calls=[
                AnswerRoleCall(role="answer_grounding", model_call_id=f"ground-call-{identity}")
            ],
        )
    )
    return TypedAnswerResult(
        typed_answer_id=f"answer-{identity}",
        run_id=run_id,
        snapshot_id="snapshot-1",
        query_id="query-1",
        method="pmmc",
        answer_variant="pmmc_b8_r2",
        output=output,
        ungated_output=output,
        image_items=[],
        ungated_image_items=[],
        answer_model_call_id=f"call-{identity}",
        actual_model_id="offline-fixture",
        terminal_state="abstained" if abstain else "answered",
        grounding=grounding,
        calls=[
            AnswerRoleCall(role="answer", model_call_id=f"call-{identity}"),
            *(grounding.calls if grounding else []),
        ],
    )


class Backend(_FakeBackend):
    source_projection_hash = "projection-hash"
    execute_full_grid = True

    def __init__(self, hits, level=RerankerMatchLevel.EXACT):
        super().__init__(hits=hits)
        self.level = level
        self.executed = []
        self.pairs = []
        self.answers = {}

    async def rerank_qbank_candidates(self, *, candidates, **kwargs):
        self.pairs = [item.qbank_id for item in candidates]
        return {item.qbank_id: _assessment(item.qbank_id, level=self.level) for item in candidates}

    async def bind_frozen_program(self, **kwargs):
        frozen = await super().bind_frozen_program(**kwargs)
        key = kwargs["candidate"].plan_id
        return replace(
            frozen,
            source_program_hash=key,
            requirement=frozen.requirement.model_copy(update={"content_hash": key}),
            program=frozen.program.model_copy(update={"program_hash": key}),
            binding=frozen.binding.model_copy(
                update={"query_program_binding_id": f"bind-{key}-{kwargs['round_id']}"}
            ),
        )

    async def execute_memory_program(self, query, frozen):
        self.executed.append(frozen.program.plan_id)
        value = await super().execute_memory_program(query, frozen)
        execution_id = f"exec-{frozen.program.plan_id}"
        trace = value.trace.model_copy(update={"execution_id": execution_id})
        evidence = tuple(
            item.model_copy(
                update={"execution_id": execution_id, "evidence_id": frozen.program.plan_id}
            )
            for item in value.evidence
        )
        return replace(value, trace=trace, evidence=evidence)

    async def relevant_runtime_evidence_groups(self, *, query, frozen, result):
        return RuntimeEvidenceRelevance(
            frozenset({"group-1"}),
            QueryEvidenceRelevanceAssessment(
                relevance_assessment_id=f"rel-{frozen.binding.query_program_binding_id}",
                query_executor_run_id="query-run-1",
                query_id=query.query_id,
                query_program_binding_id=frozen.binding.query_program_binding_id,
                relevance_threshold=0.3,
                required_group_count=1,
                relevant_group_count=1,
                passed=True,
                content_hash="rel-hash",
                metadata={
                    "policy_version": "query-runtime-evidence-v3",
                    "required_roles_satisfied": True,
                },
            ),
            (),
        )

    async def generate_typed_pmmc_answer(self, *, result, **kwargs):
        key = result.trace.plan_id
        if key not in self.answers:
            self.answers[key] = typed_answer(key)
        return self.answers[key]

    async def run_typed_native_mm_counterfactual(self, run_id, query):
        return typed_answer("fallback", abstain=True)


def test_integrated_grid_executes_every_budget_and_reuses_execution_and_answer():
    hits = [
        hit(f"q{b}", round_id=r, budget=b, score=b, channel=c)
        for b in [1, 2, 4, 8]
        for r in range(4)
        for c in (QBankRuntimeChannel.TEXT, QBankRuntimeChannel.LEXICAL)
    ]
    backend = Backend(hits)
    state = asyncio.run(
        OnlineQueryExecutor(backend=backend, config=CONFIG).execute(
            run_id="query-run-1", query_id="query-1"
        )
    )
    main = [fact for fact in state["cell_results"] if fact.cell.treatment == "pmmc"]
    assert len(main) == 16
    assert len(state["cell_results"]) == 48
    for budget in [1, 2, 4, 8]:
        for round_id in range(4):
            assert {
                fact.cell.treatment
                for fact in state["cell_results"]
                if fact.cell.question_budget == budget and fact.cell.revision_depth == round_id
            } == {"pmmc", "no_fallback", "forced_top1_counterfactual"}
    assert len(backend.executed) == len(set(backend.executed)) == 4
    assert len(backend.answers) == 4
    assert len(backend.pairs) == 4
    for fact in main:
        assert fact.binding.plan_id == f"plan-q{fact.cell.question_budget}"
        assert fact.pmmc_answer_id == f"answer-plan-q{fact.cell.question_budget}"
        assert fact.sufficiency.answer_support is None
        assert fact.sufficiency.metadata["answer_grounding_status"] == "not_evaluated"


def test_true_no_fallback_does_not_select_forced_low_confidence_top1():
    backend = Backend([hit(round_id=r) for r in range(4)], level=RerankerMatchLevel.NONE)
    state = asyncio.run(
        OnlineQueryExecutor(backend=backend, config=CONFIG).execute(
            run_id="query-run-1", query_id="query-1"
        )
    )
    contrast = [fact for fact in state["cell_results"] if fact.cell.treatment != "pmmc"]
    assert len(contrast) == 32
    for fact in contrast:
        if fact.cell.treatment == "no_fallback":
            assert fact.selected_answer_id is None
            assert fact.terminal_state == "no_answer"
        else:
            assert fact.selected_answer_id == "answer-plan-q1"


def test_native_empty_pool_is_a_complete_legal_abstention():
    backend = Backend([])
    state = asyncio.run(
        OnlineQueryExecutor(backend=backend, config=CONFIG).execute(
            run_id="query-run-1", query_id="query-1"
        )
    )
    assert not backend.executed and not backend.pairs
    main = [fact for fact in state["cell_results"] if fact.cell.treatment == "pmmc"]
    assert len(main) == 16
    assert all(fact.terminal_state == "abstained" for fact in main)


def test_active_only_backbone_emits_three_cells():
    backend = Backend([hit(round_id=2), hit(round_id=2, channel=QBankRuntimeChannel.LEXICAL)])
    backend.execute_full_grid = False
    state = asyncio.run(
        OnlineQueryExecutor(backend=backend, config=CONFIG).execute(
            run_id="query-run-1", query_id="query-1"
        )
    )
    assert len(state["cell_results"]) == 3
    assert all(
        (fact.cell.question_budget, fact.cell.revision_depth) == (8, 2)
        for fact in state["cell_results"]
    )


class PhasedBackend(Backend):
    def __init__(self, hits):
        super().__init__(hits)
        self.phases = {}
        self.events = []

    async def prepare_typed_pmmc_answer(self, *, query, result, cell):
        return typed_prepared().model_copy(
            update={
                "run_id": "query-run-1",
                "query": query,
                "source_lineage": {"plan_id": result.trace.plan_id},
                "answer_variant": cell.variant,
            }
        )

    def typed_request_identity(self, request):
        return request.source_lineage["plan_id"]

    async def persist_matching_phase(self, state, prepared):
        self.phases["match"] = {
            "cells": [fact.model_dump(mode="json") for fact in state["cell_results"]],
            "prepared": {key: value.model_dump(mode="json") for key, value in prepared.items()},
            "route": state["route_decision"].model_dump(mode="json"),
        }

    async def persist_runtime_phase(self, run_id, query_id, phase, payload):
        self.phases[phase] = payload

    async def load_runtime_phase(self, run_id, query_id, phase):
        return self.phases[phase]

    async def generate_deferred_typed_answer(self, request):
        identity = self.typed_request_identity(request)
        self.events.append(("answer", identity))
        return {"model_call_id": f"call-{identity}"}

    async def ground_prepared_typed_answer(self, request, reference):
        identity = self.typed_request_identity(request)
        assert reference == {"model_call_id": f"call-{identity}"}
        self.events.append(("ground", identity))
        return typed_answer(identity)


def test_durable_phases_generate_and_ground_once_without_rematching_or_execution():
    backend = PhasedBackend(
        [
            hit(round_id=r, channel=c)
            for r in range(4)
            for c in (QBankRuntimeChannel.TEXT, QBankRuntimeChannel.LEXICAL)
        ]
    )
    executor = OnlineQueryExecutor(backend=backend, config=CONFIG)

    async def run():
        matched = await executor.execute(run_id="query-run-1", query_id="query-1", phase="match")
        assert len(matched["cell_results"]) == 48
        assert not backend.events and not backend.answers
        assert backend.executed == ["plan-q1"]
        await executor.execute(run_id="query-run-1", query_id="query-1", phase="answer")
        assert backend.events == [("answer", "plan-q1")]
        selected = await executor.execute(run_id="query-run-1", query_id="query-1", phase="select")
        assert backend.events == [("answer", "plan-q1"), ("ground", "plan-q1")]
        assert backend.executed == ["plan-q1"]
        assert len(selected["cell_results"]) == 48
        assert all(fact.selected_answer_id == "answer-plan-q1" for fact in selected["cell_results"])

    asyncio.run(run())


def test_concrete_backend_defers_grounding_and_disallows_generation_in_select(monkeypatch):
    import pmmc.runtime.query_backend as module

    backend = object.__new__(PostgresOnlineQueryExecutorBackend)
    backend.config = CONFIG
    backend.sessions = object()
    backend.manifest = SimpleNamespace(query_executor_run_id="query-run-1")
    prepared = typed_prepared().model_copy(
        update={
            "run_id": "query-run-1",
            "query": typed_prepared().query.model_copy(update={"query_id": "query-1"}),
        }
    )
    answer = typed_answer("candidate")
    candidate = TypedAnswerCandidate(
        run_id=answer.run_id,
        query_id=answer.query_id,
        output=answer.ungated_output,
        image_items=[],
        answer_model_call_id=answer.answer_model_call_id,
        calls=[answer.calls[0]],
    )
    saved = []
    calls = []
    trace = SimpleNamespace(status=SimpleNamespace(value="success"))

    class Repository:
        async def get_typed_answer(self, *args):
            return None

        async def get(self, call_id):
            return trace if call_id == answer.answer_model_call_id else None

        async def add_typed_answer(self, fact):
            saved.append(fact)

    class Uow:
        def __init__(self, *args):
            self.query_executor = self.model_calls = Repository()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class Service:
        async def answer_typed(self, request, **kwargs):
            calls.append(kwargs)
            return candidate if kwargs["defer_grounding"] else answer

    async def fence(uow):
        pass

    backend._fence = fence
    backend.answer_service = Service()
    monkeypatch.setattr(module, "ExperimentUnitOfWork", Uow)

    async def run():
        reference = await backend.generate_deferred_typed_answer(prepared)
        assert reference["candidate"] == candidate.model_dump(mode="json")
        assert not saved
        final = await backend.ground_prepared_typed_answer(prepared, reference)
        assert final == answer and len(saved) == 1
        assert calls[0]["defer_grounding"] and calls[0]["allow_generation"]
        assert not calls[1]["defer_grounding"] and not calls[1]["allow_generation"]
        assert calls[1]["recovered_trace"] is trace

    asyncio.run(run())


def test_relevance_raw_cosine_cannot_pass_using_shifted_score():
    payload = dict(
        relevance_item_id="r",
        relevance_assessment_id="a",
        query_executor_run_id="run",
        query_id="q",
        execution_id="e",
        ordinal=0,
        evidence_id_hash="h",
        evidence_group_hash="g",
        modality="text",
        cosine_similarity=0.1,
        normalized_score=0.55,
        relevant=False,
        threshold=0.3,
        content_hash="hash",
        metadata={"score_transform": "raw-cosine-v1"},
    )
    assert not QueryEvidenceRelevanceItem(**payload).relevant
    with pytest.raises(ValidationError):
        QueryEvidenceRelevanceItem(**{**payload, "relevant": True})


def test_sql_scores_use_every_exact_round_not_r3_filtered(monkeypatch):
    backend = object.__new__(PostgresOnlineQueryExecutorBackend)
    backend.config = CONFIG
    seen = []

    async def load(run_id, query, round_id):
        seen.append(round_id)
        return [hit(round_id=round_id)] if round_id != 2 else []

    backend._load_round_channel_hits = load
    values = asyncio.run(backend.load_channel_hits("run", object()))
    assert seen == [0, 1, 2, 3]
    assert {value.projection_round for value in values} == {0, 1, 3}


def test_v4_sql_scopes_before_score_and_uses_exact_view():
    statements = []

    class Session:
        async def execute(self, statement, params):
            statements.append((str(statement), params))
            return SimpleNamespace(all=lambda: [])

    repo = PostgresQBankRoutingRepository(Session())
    asyncio.run(
        repo.search_embedding_verified_v4(
            collection_id="c",
            query_vector=[0.0] * 4096,
            routing_projection_id="p",
            compiler_run_id="r",
            snapshot_id="s",
            dataset="memlens",
            conversation_id="conv",
            cutoff_session_id="cut",
            route_class="answerable",
            question_budget=8,
            round_id=0,
            top_k=75,
        )
    )
    sql, params = statements[0]
    assert "verified_qbank_routing_v4" in sql
    assert "routing.projection_round = :round_id" in sql
    assert sql.index("available.session_order <= cutoff.session_order") < sql.index("scored AS")
    assert params["top_k"] == 75 and params["round_id"] == 0


def test_unique_attempt_fencing_rejects_reclaimed_and_expired_owner():
    now = utc_now()
    row = SimpleNamespace(
        status="running",
        claimed_by="owner-new",
        attempt_count=2,
        lease_expires_at=now + timedelta(seconds=10),
    )

    class Session:
        async def scalar(self, statement):
            assert "FOR UPDATE" in str(statement)
            return row

    repo = PostgresQueryExecutorRepository(Session())
    with pytest.raises(InvalidStateTransitionError):
        asyncio.run(
            repo.assert_attempt_lease(job_id="job", worker_id="owner-old", attempt_count=1, now=now)
        )
    asyncio.run(
        repo.assert_attempt_lease(job_id="job", worker_id="owner-new", attempt_count=2, now=now)
    )
    row.lease_expires_at = now
    with pytest.raises(InvalidStateTransitionError):
        asyncio.run(
            repo.assert_attempt_lease(job_id="job", worker_id="owner-new", attempt_count=2, now=now)
        )


def test_runtime_cache_identity_independent_of_answer_run_and_projection():
    query = asyncio.run(Backend([]).load_query("ignored", "query-1"))
    entry = SimpleNamespace(resolved_hash="model-lock")
    first = _runtime_embedding_cache_identity("snapshot", query, QBankRuntimeChannel.TEXT, entry)
    assert first == _runtime_embedding_cache_identity(
        "snapshot", query, QBankRuntimeChannel.TEXT, entry
    )
    assert first != _runtime_embedding_cache_identity(
        "different", query, QBankRuntimeChannel.TEXT, entry
    )
    assert first != _runtime_embedding_cache_identity(
        "snapshot", query, QBankRuntimeChannel.MULTIMODAL, entry
    )


def test_fact_payload_excludes_nested_observation_times_not_science():
    assert runtime_fact_payload({"created_at": "a", "nested": {"created_at": "b", "value": 1}}) == {
        "nested": {"value": 1}
    }
