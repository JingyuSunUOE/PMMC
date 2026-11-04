"""Shared execution, independently budgeted answer contexts, offline SDK dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.memory.operators import ProgramExecutionResult
from pmmc.models.provider_token_budget import (
    ProviderConfiguredTokenCounter,
    ProviderTokenBudgetCounter,
)
from pmmc.models.registry import load_model_registry
from pmmc.runtime import query_backend
from pmmc.runtime.answer_context import PmmcEvidenceAnswerInputProvider
from pmmc.runtime.answering import answer_format_retry_request, validate_typed_answer_input
from pmmc.runtime.query_backend import PostgresOnlineQueryExecutorBackend
from pmmc.runtime.query_executor import OnlineQueryExecutor
from pmmc.runtime.query_executor_config import (
    load_query_executor_config,
    query_executor_config_hash,
)
from pmmc.schemas import (
    AnswerGenerationInput,
    GenerationRequest,
    ImageRef,
    MemoryUnit,
    ProgramExecutionEvidenceRef,
    ProgramExecutionTrace,
    QueryAnswerVariant,
    QueryExecutorRunManifest,
    QueryInput,
    QueryProgramBinding,
    QueryRouteDecision,
)
from pmmc.schemas.answer_output import TYPED_ANSWER_CONTRACT_VERSION, AnswerContractV3
from pmmc.schemas.base import utc_now
from pmmc.schemas.query_executor import ExperimentCell
from test_step8_providers import ALIASES, BytesResolver, image_payload, invoke, raw_response

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_query_executor_config(ROOT / "configs/runtime/query-executor-v3.yaml")
REGISTRY = load_model_registry(ROOT / "configs/models/step8-providers.yaml")
SOURCE = "source-9b-query-run"
TARGET = "target-query-run"


class FixtureCounter:
    """Deterministic local-counter substitute; never loads a model/tokenizer."""

    estimated = False
    revision = "offline-fixture-v1"

    def __init__(self, alias: str) -> None:
        self.alias = alias

    def count_text(self, text: str) -> int:
        return len(text)

    def count_image(self, *, width=None, height=None) -> int:
        return 512

    def count_request(self, request: GenerationRequest) -> int:
        sources = request.metadata["typed_answer_context"]["sources"]
        cost = 20000 if self.alias == ALIASES["qwen_vllm"] else 9000
        return 1000 + len(sources) * cost


def manifest(run_id: str, alias: str) -> QueryExecutorRunManifest:
    return QueryExecutorRunManifest(
        query_executor_run_id=run_id, source_compiler_run_id="compiler",
        routing_projection_id="projection", snapshot_id="snapshot", dataset="memlens",
        schema_version="0.3.0", config_hash=query_executor_config_hash(CONFIG),
        selected_query_count=1, pilot_projection=True, content_hash="manifest-hash",
        metadata={
            "source_match_run_id": SOURCE if run_id == TARGET else None,
            "routing_projection_hash": "projection-hash",
            "answer_model_alias": alias,
            "matcher_version": CONFIG.matcher_version,
            "graph_version": CONFIG.graph_version,
            "sufficiency_policy_version": CONFIG.sufficiency_policy_version,
            "answer_prompt_version": CONFIG.answer_prompt_version,
            "answer_prompt_hash": "a" * 64,
            "answer_contract_version": CONFIG.answer_contract_version,
            "answer_output_schema_version": CONFIG.answer_output_schema_version,
            "embedding_ownership": "runtime", "source_compiler_schema_version": "0.9.0",
            "source_operator_set_version": "pmmc-operators-v7",
            "routing_projection_schema_version": "0.4.0",
            "treatment": "main", "execute_full_grid": False,
            "declared_cells": [[8, 2]], "expected_cell_count": 3,
        },
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline test attempted network I/O")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def harness(monkeypatch, kind):
    image = ImageRef(
        image_id="photo", storage_uri="s3://runtime/photo", width=2, height=2,
        mime_type="image/png", sha256=hashlib.sha256(image_payload().data).hexdigest(),
    )
    query = QueryInput(
        query_id="query", dataset="memlens", conversation_id="conversation",
        available_after_session="session", raw_query="Which item is blue?",
        query_image_refs=[image.model_copy(update={"image_id": "query-photo"})],
    )
    memories = [MemoryUnit(
        memory_id=f"memory-{index}", dataset="memlens", conversation_id="conversation",
        session_id="session", turn_id=f"turn-{index}", timestamp_order=index,
        speaker="user", text=f"Executed source {index}: " + "blue " * 400,
        available_after_session="session", image_refs=[image.model_copy(update={
            "image_id": f"photo-{index}", "metadata": {"private": "PRIVATE_SENTINEL"},
        })],
        metadata={"gold": "PRIVATE_SENTINEL"},
    ) for index in range(4)]

    class Memories:
        async def iter_visible(self, snapshot, conversation, cutoff):
            assert (snapshot, conversation, cutoff) == ("snapshot", "conversation", "session")
            for memory in memories:
                yield memory

    class Images:
        async def get_many(self, snapshot, ids):
            assert snapshot == "snapshot" and ids == []
            return []

    def provider(alias):
        entry = REGISTRY.get(alias)
        counter = (
            ProviderTokenBudgetCounter(entry)
            if alias in {ALIASES["openai"], ALIASES["anthropic"]}
            else ProviderConfiguredTokenCounter(FixtureCounter(alias), entry)
        )
        return PmmcEvidenceAnswerInputProvider(
            memories=Memories(), images=Images(), counter=counter, instruction="unused",
            answer_model_alias=alias, answer_contract_version=TYPED_ANSWER_CONTRACT_VERSION,
        )

    now = utc_now()
    execution = ProgramExecutionResult(
        trace=ProgramExecutionTrace(
            execution_id="execution", run_id=SOURCE, source_compiler_run_id="compiler",
            snapshot_id="snapshot", query_id=query.query_id, plan_id="plan",
            purpose="query_time", status="success", bindings_hash="bindings",
            started_at=now, completed_at=now, elapsed_ms=1,
        ),
        steps=(), evidence=tuple(ProgramExecutionEvidenceRef(
            evidence_ref_id=f"ref-{i}", execution_id="execution", snapshot_id="snapshot",
            step_id=0, rank=i + 1, evidence_id=m.memory_id, evidence_type="memory",
            memory_id=m.memory_id,
        ) for i, m in enumerate(memories)),
    )
    route = QueryRouteDecision(
        route_decision_id="route", query_executor_run_id=SOURCE, match_batch_id="match",
        query_id=query.query_id, outcome="program", selected_qbank_id="qbank",
        selected_plan_id="plan", confidence=1, match_threshold=0.8, gap_threshold=0.1,
        content_hash="route-hash",
    )
    binding = QueryProgramBinding(
        query_program_binding_id="binding", query_executor_run_id=SOURCE,
        query_id=query.query_id, route_decision_id="route", routing_projection_id="projection",
        qbank_id="qbank", plan_id="plan", round_id=2, program_hash="program-hash",
        operator_set_version="pmmc-operators-v7", requirement_contract_hash="requirement-hash",
        content_hash="binding-hash",
    )
    cell = ExperimentCell(
        question_budget=8, revision_depth=2, routing_policy_hash="routing-policy", seed=0,
        source_projection_id="projection", source_projection_hash="projection-hash",
    )
    cells = []
    for treatment in ("pmmc", "no_fallback", "forced_top1_counterfactual"):
        variant = cell.model_copy(update={"treatment": treatment})
        cell_id = stable_compiler_id("experiment-cell", variant.model_dump(mode="json"))
        cells.append(OnlineQueryExecutor._cell_result({
            "query_executor_run_id": SOURCE, "query_id": query.query_id,
            "cell_id": cell_id, "cell": variant, "route": route, "binding": binding,
            "execution_id": execution.trace.execution_id, "terminal_state": "no_answer",
        }).model_dump(mode="json"))
    source_input = asyncio.run(provider(ALIASES["qwen_vllm"]).prepare(
        run_id=SOURCE, snapshot_id="snapshot", query=query, result=execution,
        variant=QueryAnswerVariant(cell.variant),
    ))
    assert source_input.evidence_ids_in_prompt == ["memory-0"]
    checkpoint = {
        "cells": cells, "prepared": {cells[0]["cell_id"]: source_input.model_dump(mode="json")},
        "route": route.model_dump(mode="json"),
    }
    state = SimpleNamespace(execution=execution, checkpoint=checkpoint, reads=[], failure=None)

    class UnitOfWork:
        def __init__(self, sessions, scope):
            self.role = sessions.role
            self.query_executor = self.program_executions = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_run(self, run_id):
            assert self.role is DatabaseRole.ONLINE and run_id == SOURCE
            return manifest(SOURCE, ALIASES["qwen_vllm"]).model_dump(mode="json")

        async def get_phase_state(self, run_id, query_id, phase):
            assert self.role is DatabaseRole.ONLINE
            assert (run_id, query_id, phase) == (SOURCE, query.query_id, "match")
            return state.checkpoint

        async def get(self, execution_id):
            assert self.role is DatabaseRole.RUNTIME and execution_id == "execution"
            state.reads.append(execution_id)
            if state.failure is not None:
                raise state.failure
            return state.execution

    monkeypatch.setattr(query_backend, "ExperimentUnitOfWork", UnitOfWork)
    backend = object.__new__(PostgresOnlineQueryExecutorBackend)
    backend.manifest = manifest(TARGET, ALIASES[kind])
    backend.config = CONFIG
    backend.sessions = cast(
        async_sessionmaker[AsyncSession], SimpleNamespace(role=DatabaseRole.RUNTIME)
    )
    backend.reader_sessions = cast(
        async_sessionmaker[AsyncSession], SimpleNamespace(role=DatabaseRole.ONLINE)
    )
    backend.answer_inputs = provider(ALIASES[kind])
    monkeypatch.setattr(backend, "load_query", AsyncMock(return_value=query))
    persisted = AsyncMock()
    monkeypatch.setattr(backend, "persist_runtime_phase", persisted)
    return SimpleNamespace(
        backend=backend, state=state, query=query, persisted=persisted,
        source_input=source_input, cell=cell,
    )


@pytest.mark.parametrize("kind", ALIASES)
def test_shared_match_rebuilds_each_backbone_context_and_dispatches(monkeypatch, kind):
    case = harness(monkeypatch, kind)
    before = canonical_content_hash(case.state.checkpoint)

    async def scenario():
        await case.backend.reuse_matching_phase(TARGET, case.query.query_id)
        saved = case.persisted.call_args.args[3]
        assert len(saved["cells"]) == 3 and saved["source_match_run_id"] == SOURCE
        for source, target in zip(case.state.checkpoint["cells"], saved["cells"], strict=True):
            assert target["query_executor_run_id"] == TARGET
            for key in ("cell_id", "route", "binding", "execution_id", "sufficiency", "relevance"):
                assert target[key] == source[key]
        rebuilt = AnswerGenerationInput.model_validate(next(iter(saved["prepared"].values())))
        expected = await case.backend.answer_inputs.prepare(
            run_id=TARGET, snapshot_id="snapshot", query=case.query, result=case.state.execution,
            variant=QueryAnswerVariant(case.cell.variant), source_execution_run_id=SOURCE,
        )
        assert rebuilt == expected
        assert rebuilt.request.request_id != case.source_input.request.request_id
        assert rebuilt.source_lineage["execution_id"] == "execution"
        assert rebuilt.source_lineage["source_execution_run_id"] == SOURCE
        assert case.state.execution.trace.run_id == SOURCE
        context = validate_typed_answer_input(rebuilt)
        assert len(context.sources) == len(rebuilt.evidence_ids_in_prompt)
        if kind != "qwen_vllm":
            assert "memory-1" in rebuilt.evidence_ids_in_prompt
        counter = case.backend.answer_inputs.counter
        assert rebuilt.metadata["input_tokens"] == counter.count_request(rebuilt.request)
        assert rebuilt.metadata["truncated_evidence_count"] == 4 - len(context.sources)
        for request in (rebuilt.request, answer_format_retry_request(rebuilt.request)):
            assert counter.count_request(request) <= 32768 - 512
            policy = request.metadata["provider_answer_policy"]
            config = REGISTRY.get(ALIASES[kind]).provider_config
            assert config is not None
            assert policy["provider_config"] == config.model_dump(mode="json")
            assert request.provider_max_output_tokens == config.typed_answer_output_tokens
            assert request.max_output_tokens == 512
            assert "PRIVATE_SENTINEL" not in request.model_dump_json()
            answer = AnswerContractV3(
                answer_kind="text", answer_text="blue", selected_image_aliases=[],
                supported=True, used_evidence_aliases=["A01"], abstain_reason=None,
            )
            result, calls = await invoke(
                kind, request=request, resolver=BytesResolver(),
                raw=raw_response(kind, text=answer.model_dump_json()),
            )
            assert len(calls) == 1
            assert result.actual_model_id == REGISTRY.get(ALIASES[kind]).provider_model
            wire = json.loads(calls[0].content)
            key = "max_output_tokens" if kind == "openai" else "max_tokens"
            assert wire[key] == config.typed_answer_output_tokens
        assert case.state.reads == ["execution"]
        assert canonical_content_hash(case.state.checkpoint) == before
    asyncio.run(scenario())


@pytest.mark.parametrize("field,value", [
    ("run_id", "another-run"), ("query_id", "another-query"),
    ("snapshot_id", "another-snapshot"), ("plan_id", "another-plan"),
    ("source_compiler_run_id", "another-compiler"),
])
def test_shared_execution_rejects_foreign_lineage(monkeypatch, field, value):
    case = harness(monkeypatch, "openai")
    case.state.execution = replace(
        case.state.execution, trace=case.state.execution.trace.model_copy(update={field: value}),
    )
    with pytest.raises(ValueError, match="shared answer execution has incompatible"):
        asyncio.run(case.backend.reuse_matching_phase(TARGET, case.query.query_id))
    case.persisted.assert_not_called()


def test_shared_execution_missing_or_database_error_is_not_hidden(monkeypatch):
    case = harness(monkeypatch, "anthropic")
    case.state.execution = None
    with pytest.raises(ValueError, match="shared answer execution is missing"):
        asyncio.run(case.backend.reuse_matching_phase(TARGET, case.query.query_id))
    case.state.failure = OperationalError("read", {}, RuntimeError("database unavailable"))
    with pytest.raises(OperationalError):
        asyncio.run(case.backend.reuse_matching_phase(TARGET, case.query.query_id))
    case.persisted.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("query_executor_run_id", "another-run"), ("query_id", "another-query"),
    ("routing_projection_id", "another-projection"),
])
def test_shared_context_rejects_foreign_binding(monkeypatch, field, value):
    case = harness(monkeypatch, "openai")
    case.state.checkpoint["cells"][0]["binding"][field] = value
    with pytest.raises(ValueError, match="shared answer context has incompatible"):
        asyncio.run(case.backend.reuse_matching_phase(TARGET, case.query.query_id))
    assert case.state.reads == []
    case.persisted.assert_not_called()
