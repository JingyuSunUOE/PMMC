from __future__ import annotations

import asyncio
import json
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from sqlalchemy import insert
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

import pmmc.compiler.questioner_runner as questioner_runner_module
from pmmc.compiler.ids import (
    compilation_unit_selection_lineage,
    questioner_revision_model_call_id,
)
from pmmc.compiler.opportunities import local_opportunity_id
from pmmc.compiler.questioner import (
    QuestionerCandidateGate,
    questioner_verifier_output_parser,
    resolve_v3_draft,
)
from pmmc.compiler.questioner_config import load_questioner_config
from pmmc.compiler.questioner_prompt import (
    QUESTIONER_V3_PROMPT_VERSION,
    QUESTIONER_VERIFIER_PROMPT_VERSION,
    questioner_prompt_text,
    questioner_response_schema,
    render_questioner_verifier_request,
)
from pmmc.compiler.questioner_runner import (
    PostgresQuestionerInputProvider,
    PreparedQuestionerInputError,
    QuestionerCompilationGraph,
    QuestionerGenerationContractError,
    QuestionerPersistenceOperationError,
    QuestionerRunner,
    QuestionerStageError,
    _questioner_worker_limit_from_environment,
    questioner_seed_diagnostics,
)
from pmmc.compiler.units import (
    _rare_anchor_lexemes,
    create_compilation_unit_builder,
)
from pmmc.infrastructure.database.errors import safe_database_error_diagnostics
from pmmc.infrastructure.database.questioner_mappers import (
    question_opportunity_evidence_from_row,
    question_opportunity_evidence_to_row,
    question_opportunity_set_to_row,
    question_opportunity_to_row,
)
from pmmc.infrastructure.database.questioner_models import (
    QuestionOpportunityEvidenceRefModel,
    QuestionOpportunityModel,
    QuestionOpportunitySetModel,
)
from pmmc.models.gateways import ModelGatewayError
from pmmc.models.providers import ProviderCallError
from pmmc.models.tokenization import HeuristicTokenCounter
from pmmc.schemas import (
    AppStage,
    CandidateGateStatus,
    CandidateRejectionCode,
    DatasetName,
    ExperimentJob,
    ImageRef,
    MemoryUnit,
    ModelCallTrace,
    ModelContentKind,
    QuestionerCandidateDraftV3,
    QuestionerCandidateKind,
    QuestionerGeneration,
    QuestionerParseStatus,
    QuestionerVerificationBatchV1,
    QuestionerVerificationItemV1,
    QuestionerVerificationReason,
    QuestionerVerificationVerdict,
    QuestionOpportunityType,
    RankingScore,
    SessionRecord,
    SpeakerRole,
    StructuredOutputItemStatus,
    TraceStatus,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "compiler" / "questioner-v3.yaml"
V32_CONFIG = ROOT / "configs" / "compiler" / "questioner-v3.2.yaml"
V33_CONFIG = ROOT / "configs" / "compiler" / "questioner-v3.3.yaml"
V2_CONFIG = ROOT / "configs" / "compiler" / "questioner-v2.yaml"


def _diagnostic_generation(
    *,
    invalid_diagnostics: list[dict[str, object]],
    resolution_diagnostics: list[dict[str, object]],
) -> QuestionerGeneration:
    return QuestionerGeneration(
        generation_id="generation-diagnostics",
        run_id="run-diagnostics",
        compiler_unit_id="unit-diagnostics",
        job_id="job-diagnostics",
        model_call_id="call-diagnostics",
        attempt_number=1,
        prompt_version="pmmc-questioner-v3.2",
        renderer_version="questioner-safe-alias-renderer-v3.2",
        parser_version="questioner-json-parser-v3.2",
        prompt_hash="prompt-hash",
        input_hash="input-hash",
        output_hash="f" * 64,
        parse_status=QuestionerParseStatus.REPAIRED,
        raw_candidate_count=2,
        valid_candidate_count=0,
        invalid_candidate_count=2,
        repair_attempt_count=1,
        metadata={
            "invalid_diagnostics": invalid_diagnostics,
            "resolution_diagnostics": resolution_diagnostics,
        },
    )


def test_questioner_seed_diagnostics_normalize_mixed_ordinals_and_replay() -> None:
    generation = _diagnostic_generation(
        invalid_diagnostics=[
            {
                "ordinal": 0,
                "item_hash": "a" * 64,
                "errors": [{"location": "question_type", "type": "enum"}],
            }
        ],
        resolution_diagnostics=[
            {
                "generation_rank": 2,
                "item_hash": "b" * 64,
                "error_type": "ValueError",
            }
        ],
    )

    first = questioner_seed_diagnostics(generation)
    second = questioner_seed_diagnostics(generation)

    assert [item.item_ordinal for item in first] == [0, 1, 2]
    assert [item.metadata["source_item_ordinal"] for item in first] == [0, 1, 2]
    assert [item.model_dump(mode="json") for item in first] == [
        item.model_dump(mode="json") for item in second
    ]
    assert all(item.created_at == generation.created_at for item in first)


def test_questioner_seed_diagnostics_allocate_safe_overflow_on_bad_metadata() -> None:
    generation = _diagnostic_generation(
        invalid_diagnostics=[
            {
                "ordinal": 0,
                "item_hash": "a" * 64,
                "error_type": "ValidationError",
            }
        ],
        resolution_diagnostics=[
            {
                "generation_rank": 1,
                "item_hash": "b" * 64,
                "error_type": "ValueError",
            }
        ],
    )

    diagnostics = questioner_seed_diagnostics(generation)

    assert len({item.item_ordinal for item in diagnostics}) == 3
    assert diagnostics[1].item_ordinal > generation.raw_candidate_count
    assert diagnostics[1].metadata["source_item_ordinal"] == 0


def test_questioner_seed_diagnostics_records_terminal_generation_rejection() -> None:
    generation = QuestionerGeneration(
        generation_id="generation-terminal",
        run_id="run-terminal",
        compiler_unit_id="unit-terminal",
        job_id="job-terminal",
        model_call_id="call-terminal",
        attempt_number=2,
        prompt_version="pmmc-questioner-v3.3",
        renderer_version="questioner-safe-alias-renderer-v3.3",
        parser_version="questioner-json-parser-v3.3",
        prompt_hash="prompt-hash",
        input_hash="input-hash",
        output_hash="c" * 64,
        parse_status=QuestionerParseStatus.FAILED,
        metadata={
            "native_structured_output": False,
            "provider_repair_attempt_count": 1,
            "response_artifact_ready": True,
            "terminal_reason": "generation_structured_output_invalid",
        },
    )

    diagnostics = questioner_seed_diagnostics(generation)

    assert len(diagnostics) == 1
    assert diagnostics[0].status is StructuredOutputItemStatus.CONSERVATIVE_REJECT
    assert diagnostics[0].repair_attempt_count == 1
    assert diagnostics[0].errors[0].error_type == "structured_output_invalid"
    assert diagnostics[0].metadata["diagnostic_kind"] == (
        "terminal_generation_rejection"
    )


def test_questioner_input_provider_single_flights_cold_conversation_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class SessionRepository:
        def __init__(self, _sessions: object) -> None:
            pass

        async def list_snapshot(
            self, _snapshot_id: str, *, conversation_ids: set[str]
        ) -> list[str]:
            assert conversation_ids == {"conversation-v3"}
            calls.append("sessions:start")
            await asyncio.sleep(0.01)
            calls.append("sessions:end")
            return ["session"]

    class MemoryRepository:
        def __init__(self, _sessions: object) -> None:
            pass

        async def list_snapshot(
            self, _snapshot_id: str, *, conversation_ids: set[str]
        ) -> list[str]:
            assert conversation_ids == {"conversation-v3"}
            calls.append("memories:start")
            await asyncio.sleep(0.01)
            calls.append("memories:end")
            return ["memory"]

    monkeypatch.setattr(questioner_runner_module, "PostgresSessionRepository", SessionRepository)
    monkeypatch.setattr(questioner_runner_module, "PostgresMemoryRepository", MemoryRepository)
    provider = PostgresQuestionerInputProvider(
        sessions=object(),  # type: ignore[arg-type]
        config=load_questioner_config(CONFIG),
    )
    unit = SimpleNamespace(
        snapshot_id="snapshot-v3",
        conversation_id="conversation-v3",
    )

    async def exercise() -> list[tuple[list[object], list[object]]]:
        return await asyncio.gather(*(provider._conversation(unit) for _ in range(16)))

    results = asyncio.run(exercise())
    assert calls == [
        "sessions:start",
        "sessions:end",
        "memories:start",
        "memories:end",
    ]
    assert all(value is results[0] for value in results)


def test_questioner_stage_diagnostics_preserve_safe_provider_status() -> None:
    provider_error = ModelGatewayError(
        "BadRequestError",
        provider_status_code=400,
        retryable=False,
    )
    error = QuestionerStageError("generation", provider_error)
    diagnostics = safe_database_error_diagnostics(error)
    assert diagnostics == {
        "error_type": "QuestionerStageError",
        "failure_stage": "generation",
        "error_code": "questioner_generation_ModelGatewayError",
        "provider_error_code": "BadRequestError",
        "provider_status_code": "400",
        "provider_retryable": "false",
    }


def test_questioner_worker_limit_is_explicit_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PMMC_QUESTIONER_WORKER_LIMIT", raising=False)
    assert _questioner_worker_limit_from_environment() is None

    monkeypatch.setenv("PMMC_QUESTIONER_WORKER_LIMIT", "32")
    assert _questioner_worker_limit_from_environment() == 32

    monkeypatch.setenv("PMMC_QUESTIONER_WORKER_LIMIT", "0")
    with pytest.raises(ValueError, match="must be positive"):
        _questioner_worker_limit_from_environment()


@pytest.mark.parametrize("retryable", [False, True, None])
def test_questioner_job_retry_respects_nested_transport_policy(monkeypatch, retryable) -> None:
    finish = AsyncMock(return_value=SimpleNamespace(status="failed"))
    artifacts = SimpleNamespace(
        put_bytes=AsyncMock(return_value=SimpleNamespace(artifact_id="err")),
    )

    @asynccontextmanager
    async def uow(_sessions):
        yield SimpleNamespace(jobs=SimpleNamespace(finish=finish))

    monkeypatch.setattr(questioner_runner_module, "ExperimentUnitOfWork", uow)
    runner = QuestionerRunner.__new__(QuestionerRunner)
    runner.sessions = object()
    runner.error_artifacts = artifacts
    cause = (
        ModelGatewayError("BadRequestError", retryable=retryable, provider_status_code=400)
        if retryable is not None else OSError("transient artifact failure")
    )
    job = SimpleNamespace(job_id="job", run_id="run", snapshot_id="snapshot", attempt_count=1)
    asyncio.run(runner._finish_failed(job, "worker", QuestionerStageError("generation", cause)))
    assert finish.call_args.kwargs["retryable"] is (retryable is not False)


def test_questioner_failure_event_includes_safe_provider_cause(capsys) -> None:
    error = ProviderCallError(
        "BadRequestError", outcome="failed", provider_status_code=400,
        provider_error_category="context_length_exceeded",
    )
    error.requested_output_tokens = 4096
    error.provider_output_tokens = 16384
    error.provider_output_ceiling = 16384

    async def exercise():
        runner = QuestionerRunner.__new__(QuestionerRunner)
        runner._progress_lock = asyncio.Lock()
        runner._progress = {"failed": 0}
        for field in (
            "_failure_stages", "_failure_codes", "_failure_operations", "_failure_database",
        ):
            setattr(runner, field, Counter())
        runner._failure_attempts = 0
        await runner._record_failure_attempt(
            QuestionerStageError("generation", error), unit_ordinal=9,
            attempt_number=1, exhausted=True,
        )
        assert runner._failure_codes == {"BadRequestError": 1}

    asyncio.run(exercise())
    event = json.loads(capsys.readouterr().out.removeprefix("QUESTIONER_FAILURE "))
    assert event["provider_status_code"] == "400"
    assert event["provider_error_category"] == "context_length_exceeded"
    assert event["provider_retryable"] == "false"
    assert event["requested_output_tokens"] == "4096"
    assert event["provider_output_tokens"] == event["provider_output_ceiling"] == "16384"


def test_outer_retry_hint_cannot_override_nonretryable_provider_cause() -> None:
    outer = ModelGatewayError("wrapper", retryable=True)
    outer.__cause__ = ProviderCallError(
        "BadRequestError", outcome="failed", provider_status_code=400,
    )
    assert safe_database_error_diagnostics(outer)["provider_retryable"] == "false"


def test_questioner_stage_diagnostics_identify_stale_prepared_request() -> None:
    error = QuestionerStageError(
        "input_load",
        PreparedQuestionerInputError("prepared_request_hash_mismatch"),
    )

    assert safe_database_error_diagnostics(error) == {
        "error_type": "QuestionerStageError",
        "failure_stage": "input_load",
        "error_code": "questioner_input_load_prepared_request_hash_mismatch",
    }


def test_questioner_stage_diagnostics_identify_generation_contract_failure() -> None:
    error = QuestionerStageError(
        "generation",
        QuestionerGenerationContractError("structured_output_invalid"),
    )

    assert safe_database_error_diagnostics(error) == {
        "error_type": "QuestionerStageError",
        "failure_stage": "generation",
        "error_code": "questioner_generation_structured_output_invalid",
    }


def test_questioner_stage_diagnostics_expose_safe_asyncpg_constraint_locator() -> None:
    class PgConstraintError(Exception):
        sqlstate = "23514"
        schema_name = "pmmc_runtime"
        table_name = "qbank_questions"
        constraint_name = "ck_qbank_questions_v32_revision_call_lineage"

    database_error = IntegrityError(
        "PRIVATE SQL MUST NOT LEAK",
        {"answer": "PRIVATE VALUE MUST NOT LEAK"},
        PgConstraintError("PRIVATE DATABASE DETAIL MUST NOT LEAK"),
    )
    diagnostics = safe_database_error_diagnostics(
        QuestionerStageError("terminal_persistence", database_error)
    )

    assert diagnostics["sqlstate"] == "23514"
    assert diagnostics["schema_name"] == "pmmc_runtime"
    assert diagnostics["table_name"] == "qbank_questions"
    assert diagnostics["constraint_name"] == "ck_qbank_questions_v32_revision_call_lineage"
    serialized = json.dumps(diagnostics, sort_keys=True)
    assert "PRIVATE" not in serialized


def test_database_diagnostics_map_completion_guard_without_leaking_payload() -> None:
    class PgCompletionError(Exception):
        sqlstate = "55000"

    database_error = IntegrityError(
        "PRIVATE SQL MUST NOT LEAK",
        {"gold": "PRIVATE VALUE MUST NOT LEAK"},
        PgCompletionError("compiler stage counts differ from persisted facts"),
    )

    diagnostics = safe_database_error_diagnostics(database_error)

    assert diagnostics["sqlstate"] == "55000"
    assert diagnostics["database_guard_code"] == (
        "compiler_stage_fact_count_mismatch"
    )
    assert "PRIVATE" not in json.dumps(diagnostics, sort_keys=True)


def test_safe_diagnostics_include_os_errno_without_leaking_path() -> None:
    error = PermissionError(13, "PRIVATE MESSAGE", "/private/model/snapshot")
    error.failure_stage = "corpus:memlens"
    error.persistence_operation = "tokenizer_load"

    diagnostics = safe_database_error_diagnostics(error)

    assert diagnostics == {
        "error_type": "PermissionError",
        "failure_stage": "corpus:memlens",
        "persistence_operation": "tokenizer_load",
        "os_errno": "13",
        "os_error_name": "EACCES",
    }
    assert "PRIVATE" not in json.dumps(diagnostics, sort_keys=True)
    assert "/private" not in json.dumps(diagnostics, sort_keys=True)


def test_questioner_failure_event_logs_safe_attempt_locator(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class PgConstraintError(Exception):
        sqlstate = "23514"
        schema_name = "pmmc_runtime"
        table_name = "qbank_questions"
        constraint_name = "ck_qbank_questions_v32_revision_call_lineage"

    error = QuestionerStageError(
        "terminal_persistence",
        QuestionerPersistenceOperationError(
            "gated_candidates",
            IntegrityError(
                "PRIVATE SQL MUST NOT LEAK",
                {"answer": "PRIVATE VALUE MUST NOT LEAK"},
                PgConstraintError("PRIVATE DETAIL MUST NOT LEAK"),
            ),
            candidate_count=6,
            revised_candidate_count=1,
            verification_batch_count=6,
            supersession_count=1,
        ),
    )

    async def exercise() -> None:
        runner = QuestionerRunner.__new__(QuestionerRunner)
        runner._progress_lock = asyncio.Lock()
        runner._progress = {"failed": 0}
        runner._failure_stages = Counter()
        runner._failure_codes = Counter()
        runner._failure_operations = Counter()
        runner._failure_database = Counter()
        runner._failure_attempts = 0
        await runner._record_failure_attempt(
            error,
            unit_ordinal=17,
            attempt_number=1,
            exhausted=False,
        )

    asyncio.run(exercise())
    prefix, raw = capsys.readouterr().out.strip().split(" ", 1)
    payload = json.loads(raw)

    assert prefix == "QUESTIONER_FAILURE"
    assert payload["unit_ordinal"] == 17
    assert payload["attempt_number"] == 1
    assert payload["exhausted"] is False
    assert payload["persistence_operation"] == "gated_candidates"
    assert payload["persistence_candidate_count"] == "6"
    assert payload["persistence_revised_candidate_count"] == "1"
    assert payload["persistence_verification_batch_count"] == "6"
    assert payload["persistence_supersession_count"] == "1"
    assert payload["sqlstate"] == "23514"
    assert payload["constraint_name"] == "ck_qbank_questions_v32_revision_call_lineage"
    assert "PRIVATE" not in raw


def test_questioner_failure_event_logs_safe_verification_batch_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class PgConstraintError(Exception):
        sqlstate = "23514"
        schema_name = "pmmc_runtime"
        table_name = "questioner_verification_batches"
        constraint_name = "ck_qv_batches_failed_call_terminal_rejection"

    error = QuestionerStageError(
        "terminal_persistence",
        QuestionerPersistenceOperationError(
            "verification_batch",
            IntegrityError("PRIVATE SQL", {}, PgConstraintError("PRIVATE DETAIL")),
            candidate_count=3,
            revised_candidate_count=0,
            verification_batch_count=2,
            supersession_count=0,
            verification_input_count=1,
            verification_pass_count=0,
            verification_revise_count=0,
            verification_reject_count=1,
            verification_provider_fallback=True,
            verification_single_candidate_scope=True,
        ),
    )

    async def exercise() -> None:
        runner = QuestionerRunner.__new__(QuestionerRunner)
        runner._progress_lock = asyncio.Lock()
        runner._progress = {"failed": 0}
        runner._failure_stages = Counter()
        runner._failure_codes = Counter()
        runner._failure_operations = Counter()
        runner._failure_database = Counter()
        runner._failure_attempts = 0
        await runner._record_failure_attempt(
            error,
            unit_ordinal=9,
            attempt_number=2,
            exhausted=True,
        )

    asyncio.run(exercise())
    _prefix, raw = capsys.readouterr().out.strip().split(" ", 1)
    payload = json.loads(raw)

    assert payload["constraint_name"] == "ck_qv_batches_failed_call_terminal_rejection"
    assert payload["persistence_verification_input_count"] == "1"
    assert payload["persistence_verification_pass_count"] == "0"
    assert payload["persistence_verification_revise_count"] == "0"
    assert payload["persistence_verification_reject_count"] == "1"
    assert payload["persistence_verification_provider_fallback"] == "1"
    assert payload["persistence_verification_single_candidate_scope"] == "1"
    assert "PRIVATE" not in raw


def _session(
    session_id: str,
    order: int,
    *,
    dataset: DatasetName = DatasetName.MEMLENS,
    conversation_id: str = "conversation-v3",
    synthetic: bool = False,
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        dataset=dataset,
        conversation_id=conversation_id,
        session_order=order,
        source_session_id=f"source-{session_id}",
        source_date_iso=f"2026-07-{order + 1:02d}",
        is_synthetic=synthetic,
    )


def _memory(
    memory_id: str,
    session_id: str,
    order: int,
    text: str,
    *,
    dataset: DatasetName = DatasetName.MEMLENS,
    conversation_id: str = "conversation-v3",
    image_id: str | None = None,
    source_round_id: str | None = None,
    speaker: SpeakerRole = SpeakerRole.USER,
) -> MemoryUnit:
    return MemoryUnit(
        memory_id=memory_id,
        dataset=dataset,
        conversation_id=conversation_id,
        session_id=session_id,
        turn_id=f"turn-{memory_id}",
        timestamp_order=order,
        speaker=speaker,
        text=text,
        caption=("A blue collar." if image_id else None),
        image_ids=([image_id] if image_id else []),
        image_refs=([ImageRef(image_id=image_id)] if image_id else []),
        entity_mentions=["Milo"],
        available_after_session=session_id,
        metadata=({"source_round_id": source_round_id} if source_round_id is not None else {}),
    )


def _build_memlens_unit():  # type: ignore[no-untyped-def]
    config = load_questioner_config(CONFIG)
    sessions = [_session("session-old", 0), _session("session-new", 1)]
    memories = [
        _memory(
            "memory-old",
            "session-old",
            0,
            "What color collar does Milo wear? Milo wore a red collar in 2025.",
        ),
        _memory(
            "memory-new",
            "session-new",
            1,
            "Milo now wears a blue collar in 2026.",
            image_id="image-blue",
        ),
    ]
    built = create_compilation_unit_builder(
        config=config,
        token_counter=HeuristicTokenCounter("questioner-v3-fixture", "fixture-revision"),
        dataset=DatasetName.MEMLENS,
    ).build(
        run_id="compiler-run-v3",
        snapshot_id="snapshot-v3",
        sessions=sessions,
        memories=memories,
    )[-1]
    assert built.opportunities is not None
    return config, built


@pytest.mark.parametrize("dataset", [DatasetName.MEMLENS, DatasetName.MEMGALLERY])
def test_full_request_budget_keeps_all_anchors_and_complete_rounds(dataset):
    import re

    from pmmc.compiler.units import FULL_REQUEST_BUDGET_VERSION

    config = load_questioner_config(ROOT / "configs/compiler/questioner-v3.6.yaml")

    class CatalogCounter(HeuristicTokenCounter):
        def count_request(self, request):
            assert request.metadata["prompt_version"] == config.prompt_version
            text = "\n".join(p.text or "" for m in request.messages for p in m.parts)
            # The final catalog has overhead not present in the old v2 probe.
            return 1000 + 7000 * len(set(re.findall(r"BUDGET_MARKER_\d+", text)))

    sessions = [_session("s0", 0, dataset=dataset), _session("s1", 1, dataset=dataset)]
    memories = [
        _memory(
            f"m{i}", f"s{i // 8}", i, f"BUDGET_MARKER_{i} is the saved blue preference.",
            dataset=dataset, image_id=f"image-{i}",
            source_round_id=f"round-{i // 2}" if dataset is DatasetName.MEMGALLERY else None,
            speaker=SpeakerRole.USER if i % 2 == 0 else SpeakerRole.ASSISTANT,
        ) for i in range(16)
    ]
    counter = CatalogCounter()
    built = create_compilation_unit_builder(
        config=config, token_counter=counter, dataset=dataset,
    ).build(run_id="budget-run", snapshot_id="snapshot-budget",
            sessions=sessions, memories=memories)
    assert {m.memory_id for u in built for m in u.unit.memories if m.role.value == "anchor"} == {
        m.memory_id for m in memories
    }
    assert all(u.input_tokens == counter.count_request(u.request) <= config.max_input_tokens
               for u in built)
    assert all(u.unit.metadata["builder_identity"]["budget_policy_version"]
               == FULL_REQUEST_BUDGET_VERSION for u in built)
    original = {m.memory_id: m for m in memories}
    assert all(m == original[m.memory_id] for u in built for m in u.memories)
    if dataset is DatasetName.MEMGALLERY:
        for unit in built:
            counts = Counter(m.metadata["source_round_id"] for m in unit.memories)
            assert set(counts.values()) == {2}


def test_full_request_budget_does_not_hide_an_oversized_anchor():
    import pickle

    from pmmc.compiler.units import QuestionerInputTooLarge

    config = load_questioner_config(ROOT / "configs/compiler/questioner-v3.6.yaml")

    class OversizedCounter(HeuristicTokenCounter):
        def count_request(self, request):
            assert request.metadata["prompt_version"] == config.prompt_version
            return config.max_input_tokens + 1

    builder = create_compilation_unit_builder(
        config=config, token_counter=OversizedCounter(), dataset=DatasetName.MEMLENS,
    )
    with pytest.raises(QuestionerInputTooLarge, match="single anchor") as raised:
        builder.build(run_id="budget-run", snapshot_id="snapshot-budget",
                      sessions=[_session("s0", 0)],
                      memories=[_memory("m0", "s0", 0, "A saved preference.", image_id="i0")])
    restored = pickle.loads(pickle.dumps(raised.value))
    assert restored.error_code == "questioner_single_anchor_input_too_large"


def _local_fact_draft(built, **updates):  # type: ignore[no-untyped-def]
    opportunity = next(
        item
        for item in built.opportunities.opportunities
        if item.opportunity_type is QuestionOpportunityType.LOCAL_FACT
        and any(
            built.opportunities.alias_to_evidence[alias].memory_id == "memory-new"
            for alias in item.evidence_aliases
        )
    )
    payload = {
        "candidate_kind": QuestionerCandidateKind.ANSWERABLE,
        "opportunity_id": local_opportunity_id(opportunity),
        "generated_question": "What color collar does Milo wear now?",
        "question_type": opportunity.allowed_question_types[0],
        "reasoning_type": opportunity.allowed_reasoning_types[0],
        "expected_answer": "blue collar",
        "evidence_aliases": [opportunity.evidence_aliases[0]],
        "required_modalities": opportunity.required_modalities,
        "paraphrases": [],
        "confidence": 0.9,
    }
    payload.update(updates)
    return QuestionerCandidateDraftV3.model_validate(payload)


def test_questioner_v3_provider_schema_requires_every_field() -> None:
    schema = questioner_response_schema(QUESTIONER_V3_PROMPT_VERSION)
    candidate_schema = schema["$defs"]["QuestionerCandidateDraftV3"]
    assert set(candidate_schema["required"]) == set(candidate_schema["properties"])
    assert candidate_schema["additionalProperties"] is False
    assert schema["properties"]["candidates"]["maxItems"] == 8
    with pytest.raises(ValidationError):
        QuestionerCandidateDraftV3.model_validate(
            {
                "candidate_kind": "answerable",
                "opportunity_id": "O01",
                "generated_question": "What color is the collar?",
                "question_type": "textual_fact",
                "reasoning_type": "single_hop",
                "expected_answer": "blue",
                "evidence_aliases": ["M01"],
                "required_modalities": ["text"],
                "paraphrases": [],
                # confidence is intentionally absent; v3 has no provider defaults.
            }
        )


def test_questioner_protocol_selects_its_persisted_compiler_schema() -> None:
    assert load_questioner_config(CONFIG).compiler_schema_version == "0.3.0"
    assert load_questioner_config(V2_CONFIG).compiler_schema_version == "0.2.0"


def test_questioner_v3_renderer_exposes_only_local_aliases() -> None:
    _config, built = _build_memlens_unit()
    rendered_text = "\n".join(
        part.text
        for message in built.request.messages
        for part in message.parts
        if part.kind is ModelContentKind.TEXT and part.text
    )
    for hidden in (
        built.unit.compiler_unit_id,
        built.unit.available_after_session,
        "memory-old",
        "memory-new",
        "image-blue",
        "turn-memory-new",
    ):
        assert hidden not in rendered_text
    assert "[MEMORY M" in rendered_text
    assert "[IMAGE I" in rendered_text
    assert '"opportunity_id":"O' in rendered_text


def test_questioner_v3_alias_resolution_and_static_quality_gate() -> None:
    config, built = _build_memlens_unit()
    opportunity_set = built.opportunities
    assert opportunity_set is not None
    cases = [
        (
            _local_fact_draft(built),
            None,
        ),
        (
            _local_fact_draft(
                built,
                generated_question="What does memory_123 say about Milo?",
            ),
            CandidateRejectionCode.INTERNAL_IDENTIFIER_LEAK,
        ),
        (
            _local_fact_draft(
                built,
                generated_question="What is the session order for Milo's collar?",
            ),
            CandidateRejectionCode.SYSTEM_METADATA_REFERENCE,
        ),
        (
            _local_fact_draft(
                built,
                generated_question="Would you like me to describe Milo's collar?",
            ),
            CandidateRejectionCode.ASSISTANT_FOLLOW_UP,
        ),
        (
            _local_fact_draft(
                built,
                generated_question="What color collar does Milo wear?",
            ),
            CandidateRejectionCode.HISTORICAL_QUESTION_REPLAY,
        ),
        (
            _local_fact_draft(
                built,
                generated_question="What is quantum chromodynamics?",
                expected_answer="a particle theory",
            ),
            CandidateRejectionCode.GENERIC_EXTERNAL_QUESTION,
        ),
        (
            _local_fact_draft(built, expected_answer="TBD"),
            CandidateRejectionCode.PLACEHOLDER_EXPECTED_ANSWER,
        ),
    ]
    resolved = [resolve_v3_draft(value, opportunity_set=opportunity_set) for value, _ in cases]
    gated = QuestionerCandidateGate(config).evaluate(
        run_id=built.unit.run_id,
        snapshot_id=built.unit.snapshot_id,
        unit=built.unit,
        generation_id="generation-v3",
        model_call_id="model-call-v3",
        drafts=resolved,
        memories=list(built.memories),
        opportunity_set=opportunity_set,
        generation_ranks=list(range(1, len(resolved) + 1)),
        apply_dedupe=False,
    )
    assert gated[0].gate_result.status is CandidateGateStatus.ELIGIBLE_FOR_PLANNING
    for result, (_, expected_code) in zip(gated[1:], cases[1:], strict=True):
        assert result.gate_result.status is CandidateGateStatus.REJECTED, (
            expected_code,
            result.candidate.canonical_question,
            result.gate_result.rejection_codes,
        )
        assert expected_code in result.gate_result.rejection_codes


def test_questioner_v3_opportunity_seed_rows_compile_and_round_trip_score() -> None:
    _config, built = _build_memlens_unit()
    opportunity_set = built.opportunities
    assert opportunity_set is not None
    evidence_ref = opportunity_set.evidence_refs[0]
    scored_ref = evidence_ref.model_copy(
        update={
            "evidence": evidence_ref.evidence.model_copy(
                update={
                    "score": RankingScore(
                        score_type="fts_relation",
                        raw_value=0.75,
                        normalized_value=0.75,
                        normalization_method="identity",
                    )
                }
            )
        }
    )
    evidence_row = question_opportunity_evidence_to_row(scored_ref)
    assert "score" not in evidence_row
    assert evidence_row["metadata"]["evidence_score"]["raw_value"] == 0.75
    assert question_opportunity_evidence_from_row(evidence_row) == scored_ref

    rows = (
        (
            QuestionOpportunitySetModel,
            question_opportunity_set_to_row(opportunity_set.opportunity_set),
        ),
        (
            QuestionOpportunityModel,
            question_opportunity_to_row(opportunity_set.opportunities[0]),
        ),
        (QuestionOpportunityEvidenceRefModel, evidence_row),
    )
    for model, row in rows:
        insert(model.__table__).values(row).compile(dialect=postgresql.dialect())


def test_questioner_v3_memgallery_round_is_one_evidence_group() -> None:
    config = load_questioner_config(CONFIG)
    dataset = DatasetName.MEMGALLERY
    conversation = "memgallery:conversation:scenario-a"
    sessions = [_session("mg-session", 0, dataset=dataset, conversation_id=conversation)]
    memories: list[MemoryUnit] = []
    order = 0
    for round_ordinal in range(1, 5):
        source_round = f"round-{round_ordinal}"
        memories.extend(
            [
                _memory(
                    f"mg-{round_ordinal}-user",
                    "mg-session",
                    order,
                    f"The user showed item {round_ordinal}.",
                    dataset=dataset,
                    conversation_id=conversation,
                    image_id=("mg-image-1" if round_ordinal == 1 else None),
                    source_round_id=source_round,
                ),
                _memory(
                    f"mg-{round_ordinal}-assistant",
                    "mg-session",
                    order + 1,
                    f"The item {round_ordinal} was recorded.",
                    dataset=dataset,
                    conversation_id=conversation,
                    source_round_id=source_round,
                    speaker=SpeakerRole.ASSISTANT,
                ),
            ]
        )
        order += 2
    built = create_compilation_unit_builder(
        config=config,
        token_counter=HeuristicTokenCounter("questioner-v3-fixture", "fixture-revision"),
        dataset=dataset,
    ).build(
        run_id="compiler-run-mg-v3",
        snapshot_id="snapshot-mg-v3",
        sessions=sessions,
        memories=memories,
    )[0]
    assert built.opportunities is not None
    aliases = built.opportunities
    round_one_aliases = [
        alias
        for alias, evidence in aliases.alias_to_evidence.items()
        if evidence.memory_id in {"mg-1-user", "mg-1-assistant"}
        or evidence.image_id == "mg-image-1"
    ]
    assert len({aliases.alias_to_group[alias] for alias in round_one_aliases}) == 1
    for opportunity in aliases.opportunities:
        if opportunity.minimum_evidence_groups >= 2:
            assert (
                len({aliases.alias_to_group[alias] for alias in opportunity.evidence_aliases}) >= 2
            )


def test_questioner_v3_memgallery_profile_requires_explicit_preference_relation() -> None:
    class EmptyProvider:
        selector_version = "postgres-simple-fts-v1"

        def search(self, **_kwargs):  # type: ignore[no-untyped-def]
            return []

    config = load_questioner_config(CONFIG)
    dataset = DatasetName.MEMGALLERY
    conversation = "memgallery:conversation:scenario-preference"
    sessions = [
        _session(
            "profile-session",
            0,
            dataset=dataset,
            conversation_id=conversation,
            synthetic=True,
        ),
        _session(
            "dialogue-session",
            1,
            dataset=dataset,
            conversation_id=conversation,
        ),
    ]
    memories = [
        _memory(
            "profile-memory",
            "profile-session",
            0,
            "Milo's owner prefers blue accessories.",
            dataset=dataset,
            conversation_id=conversation,
        )
    ]
    for round_ordinal in range(1, 5):
        memories.extend(
            [
                _memory(
                    f"preference-{round_ordinal}-user",
                    "dialogue-session",
                    round_ordinal * 2,
                    f"Milo tried accessory {round_ordinal}.",
                    dataset=dataset,
                    conversation_id=conversation,
                    source_round_id=f"round-{round_ordinal}",
                ),
                _memory(
                    f"preference-{round_ordinal}-assistant",
                    "dialogue-session",
                    round_ordinal * 2 + 1,
                    "The accessory was recorded.",
                    dataset=dataset,
                    conversation_id=conversation,
                    source_round_id=f"round-{round_ordinal}",
                    speaker=SpeakerRole.ASSISTANT,
                ),
            ]
        )
    units = create_compilation_unit_builder(
        config=config,
        token_counter=HeuristicTokenCounter("questioner-v3-fixture", "fixture-revision"),
        dataset=dataset,
        fts_relation_provider=EmptyProvider(),
    ).build(
        run_id="compiler-run-mg-preference",
        snapshot_id="snapshot-mg-preference",
        sessions=sessions,
        memories=memories,
    )
    dialogue_unit = next(
        unit
        for unit in units
        if "preference-1-user" in set(unit.unit.metadata.get("anchor_memory_ids", []))
    )
    profile_member = next(
        item for item in dialogue_unit.unit.memories if item.memory_id == "profile-memory"
    )
    assert profile_member.metadata["relation_source"] == ("explicit_preference_relation")


def test_questioner_v3_relation_lexemes_prioritize_rare_terms() -> None:
    common = "conversation"
    rare = "xylophone"
    anchor = _memory(
        "anchor",
        "session-new",
        10,
        " ".join([common, rare, *(f"rareterm{index}" for index in range(11))]),
    )
    corpus = [
        anchor,
        _memory("common-1", "session-old", 1, f"{common} ordinary history"),
        _memory("common-2", "session-old", 2, f"{common} another history"),
    ]
    selected, frequencies = _rare_anchor_lexemes([anchor], corpus, limit=12)
    assert rare in selected
    assert common not in selected
    assert frequencies[common] > frequencies[rare]


def test_questioner_v3_builder_uses_frozen_fts_provider_order() -> None:
    class Hit:
        def __init__(self, memory_id: str, score: float) -> None:
            self.memory_id = memory_id
            self.score = score

    class Provider:
        selector_version = "postgres-simple-fts-v1"

        def __init__(self, score: float = 0.75) -> None:
            self.calls: list[dict[str, object]] = []
            self.score = score

        def search(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            return [Hit("memory-related", self.score)]

    provider = Provider()
    config = load_questioner_config(CONFIG)
    sessions = [
        _session("session-old", 0),
        _session("session-middle", 1),
        _session("session-new", 2),
    ]
    memories = [
        _memory("memory-unrelated", "session-old", 0, "Unrelated generic words."),
        _memory("memory-related", "session-old", 1, "Milo changed collars."),
        _memory("memory-anchor", "session-new", 2, "Milo has a blue collar."),
    ]
    built = create_compilation_unit_builder(
        config=config,
        token_counter=HeuristicTokenCounter("questioner-v3-fixture", "fixture-revision"),
        dataset=DatasetName.MEMLENS,
        fts_relation_provider=provider,
    ).build(
        run_id="compiler-run-v3-fts",
        snapshot_id="snapshot-v3-fts",
        sessions=sessions,
        memories=memories,
    )[-1]
    assert provider.calls
    assert provider.calls[-1]["cutoff_session_id"] == "session-new"
    assert provider.calls[-1]["lexeme_limit"] == 12
    assert provider.calls[-1]["top_k"] == 8
    assert "memory-related" in {item.memory_id for item in built.memories}
    assert built.unit.metadata["relation_selector_version"] == provider.selector_version
    related_member = next(
        item for item in built.unit.memories if item.memory_id == "memory-related"
    )
    assert related_member.metadata["score_micros"] == 750_000
    assert related_member.metadata["selector_rank"] == 1
    assert any(link.metadata.get("score_micros") == 750_000 for link in built.links)
    assert built.unit.metadata["relation_selection"] == (
        compilation_unit_selection_lineage(built.unit.memories)
    )

    rescored = create_compilation_unit_builder(
        config=config,
        token_counter=HeuristicTokenCounter("questioner-v3-fixture", "fixture-revision"),
        dataset=DatasetName.MEMLENS,
        fts_relation_provider=Provider(score=0.5),
    ).build(
        run_id="compiler-run-v3-fts",
        snapshot_id="snapshot-v3-fts",
        sessions=sessions,
        memories=memories,
    )[-1]
    assert rescored.unit.compiler_unit_id != built.unit.compiler_unit_id


def test_questioner_v3_verifier_contract_is_bounded_and_redacted() -> None:
    _config, built = _build_memlens_unit()
    opportunity_set = built.opportunities
    assert opportunity_set is not None
    draft = _local_fact_draft(built)
    request = render_questioner_verifier_request(
        request_id="verifier-request",
        model_alias="compiler-qwen3.5-9b",
        unit_id=built.unit.compiler_unit_id,
        cutoff_session_id=built.unit.available_after_session,
        memories=list(built.memories),
        sessions=list(built.sessions),
        opportunity_set=opportunity_set,
        candidates=[("Q01", draft)],
        max_output_tokens=4096,
        final_round=False,
    )
    text = "\n".join(
        part.text or ""
        for message in request.messages
        for part in message.parts
        if part.kind is ModelContentKind.TEXT
    )
    assert "Q01" in text and "O" in text
    assert built.unit.compiler_unit_id not in text
    assert "memory-new" not in text
    verifier_schema = questioner_response_schema(QUESTIONER_VERIFIER_PROMPT_VERSION)
    assert verifier_schema["additionalProperties"] is False
    assert QuestionerVerificationReason.INTERNAL_METADATA.value in json.dumps(
        verifier_schema, sort_keys=True
    )
    assert "chain-of-thought" in questioner_prompt_text(QUESTIONER_VERIFIER_PROMPT_VERSION)


def test_questioner_v3_verifier_parser_rejects_only_invalid_or_missing_items() -> None:
    pass_item = {
        "verdict": "pass",
        "reason_codes": [],
        "replacement": None,
    }
    parsed = questioner_verifier_output_parser(expected_aliases=["Q01", "Q02", "Q03"])(
        SimpleNamespace(
            text=json.dumps(
                {
                    "results": [
                        {"candidate_alias": "Q02", **pass_item},
                        {
                            "candidate_alias": "Q01",
                            "verdict": "pass",
                            "reason_codes": ["not_answerable"],
                            "replacement": None,
                        },
                        {"candidate_alias": "Q99", **pass_item},
                    ]
                }
            )
        )
    )
    payload = dict(parsed.value)
    metadata = payload.pop("_parser_metadata")
    output = QuestionerVerificationBatchV1.model_validate(payload)
    assert [item.candidate_alias for item in output.results] == ["Q01", "Q02", "Q03"]
    assert [item.verdict.value for item in output.results] == ["reject", "pass", "reject"]
    assert metadata["fallback_aliases"] == ["Q01", "Q03"]
    assert metadata["unexpected_alias_count"] == 1
    assert metadata["diagnostic_codes"] == [
        "verifier_alias_missing",
        "verifier_alias_order_or_count_normalized",
        "verifier_alias_unexpected",
        "verifier_item_validation_failed",
    ]
    assert parsed.initial_error is not None


def test_questioner_v3_verifier_parser_conservatively_handles_invalid_envelope() -> None:
    parsed = questioner_verifier_output_parser(expected_aliases=["Q01"])(
        SimpleNamespace(text=json.dumps("not-a-verifier-envelope"))
    )
    payload = dict(parsed.value)
    metadata = payload.pop("_parser_metadata")
    output = QuestionerVerificationBatchV1.model_validate(payload)
    assert output.results[0].verdict is QuestionerVerificationVerdict.REJECT
    assert metadata["fallback_aliases"] == ["Q01"]
    assert "verifier_envelope_invalid" in metadata["diagnostic_codes"]


def test_questioner_v33_verifier_failure_is_audited_as_conservative_fallback() -> None:
    _, built = _build_memlens_unit()
    config = load_questioner_config(V33_CONFIG)
    candidate = _local_fact_draft(built)
    job = ExperimentJob(
        job_id="compile-job-v33-invalid-verifier",
        run_id=built.unit.run_id,
        snapshot_id=built.unit.snapshot_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        stage=AppStage.COMPILE,
        attempt_count=1,
        max_attempts=1,
    )
    generation = QuestionerGeneration(
        generation_id="generation-v33-invalid-verifier",
        run_id=built.unit.run_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        job_id=job.job_id,
        model_call_id="generation-call-v33",
        attempt_number=1,
        prompt_version=config.prompt_version,
        renderer_version=config.renderer_version,
        parser_version=config.parser_version,
        prompt_hash="prompt-hash",
        input_hash=built.unit.input_hash,
        output_hash="output-hash",
        parse_status=QuestionerParseStatus.PARSED,
        raw_candidate_count=1,
        valid_candidate_count=1,
    )
    failed = ModelCallTrace(
        model_call_id=questioner_runner_module.questioner_verifier_model_call_id(
            run_id=job.run_id,
            compiler_unit_id=built.unit.compiler_unit_id,
            generation_id=generation.generation_id,
            verification_round=0,
            attempt_number=1,
            candidate_scope="Q01",
        ),
        run_id=job.run_id,
        snapshot_id=job.snapshot_id,
        conversation_id=built.unit.conversation_id,
        job_id=job.job_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        parent_model_call_id=generation.model_call_id,
        stage=AppStage.QUESTIONER,
        model_alias=config.compiler_model_alias,
        status=TraceStatus.FAILED,
        output_asset_id="artifact-invalid-verifier-response",
        error_type="StructuredOutputValidationError",
        parse_error="structured output validation failed",
    )

    async def load_model_call(_call_id: str) -> ModelCallTrace:
        return failed

    async def no_call(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return None

    class NoExecution:
        async def generate_audited(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("an archived verifier failure must not be called again")

    graph = QuestionerCompilationGraph(
        execution=NoExecution(),  # type: ignore[arg-type]
        gateway=object(),  # type: ignore[arg-type]
        config=config,
        load_successful_call=no_call,
        load_model_call=load_model_call,
        before_gate=no_call,
        persist_v3_seed=no_call,
        persist_result=no_call,
    )
    _trace, output, metadata = asyncio.run(
        graph._run_v3_verifier(
            built=built,
            job=job,
            generation=generation,
            inputs=[("Q01", candidate)],
            verification_round=0,
        )
    )

    assert output.results[0].verdict is QuestionerVerificationVerdict.REJECT
    assert metadata["provider_output_fallback"] is True
    assert metadata["fallback_aliases"] == ["Q01"]
    assert metadata["native_structured_output"] is False


def test_questioner_v33_final_invalid_generation_closes_as_audited_zero_candidate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, built = _build_memlens_unit()
    config = load_questioner_config(V33_CONFIG)
    job = ExperimentJob(
        job_id="compile-job-v33-invalid-generation",
        run_id=built.unit.run_id,
        snapshot_id=built.unit.snapshot_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        stage=AppStage.COMPILE,
        attempt_count=2,
        max_attempts=2,
    )
    failed_call_id = questioner_runner_module.questioner_model_call_id(
        run_id=job.run_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        attempt_number=2,
    )
    failed = ModelCallTrace(
        model_call_id=failed_call_id,
        run_id=job.run_id,
        snapshot_id=job.snapshot_id,
        conversation_id=built.unit.conversation_id,
        job_id=job.job_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        stage=AppStage.QUESTIONER,
        model_alias=config.compiler_model_alias,
        prompt_hash="request-prompt-hash",
        status=TraceStatus.FAILED,
        output_asset_id="artifact-invalid-generation-response",
        error_type="StructuredOutputValidationError",
        parse_error="structured output validation failed",
        repair_attempt_count=1,
        request_config={"request_metadata": {"task": "pmmc_questioner"}},
    )
    persisted_seeds: list[QuestionerGeneration] = []
    persisted_results: list[object] = []

    async def load_model_call(call_id: str) -> ModelCallTrace | None:
        return failed if call_id == failed_call_id else None

    async def persist_seed(
        _built: object,
        generation: QuestionerGeneration,
        candidates: list[object],
        claims: list[object],
    ) -> None:
        assert candidates == []
        assert claims == []
        persisted_seeds.append(generation)

    async def persist_result(result: object) -> None:
        persisted_results.append(result)

    async def no_call(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return None

    class NoExecution:
        async def generate_audited(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("the archived final failure must not call the model")

    graph = QuestionerCompilationGraph(
        execution=NoExecution(),  # type: ignore[arg-type]
        gateway=object(),  # type: ignore[arg-type]
        config=config,
        load_successful_call=no_call,
        load_model_call=load_model_call,
        before_gate=no_call,
        persist_v3_seed=persist_seed,  # type: ignore[arg-type]
        persist_result=persist_result,  # type: ignore[arg-type]
    )

    result = asyncio.run(graph.process(built, job))

    assert result.generation.parse_status is QuestionerParseStatus.FAILED
    assert result.generation.raw_candidate_count == 0
    assert result.generation.repair_attempt_count == 0
    assert result.generation.metadata["provider_repair_attempt_count"] == 1
    assert result.stage_result.status is TraceStatus.SUCCESS
    assert result.stage_result.produced_count == 0
    assert result.candidates == ()
    assert persisted_seeds == [result.generation]
    assert persisted_results == [result]
    event_name, event_json = capsys.readouterr().out.strip().split(" ", 1)
    event = json.loads(event_json)
    assert event_name == "QUESTIONER_TERMINAL_GENERATION"
    assert event == {
        "action": "continue_to_r0_gate",
        "attempt_number": 2,
        "error_code": "questioner_generation_structured_output_invalid",
        "max_attempts": 2,
        "output_artifact_present": True,
        "parse_error_present": True,
        "unit_ordinal": built.unit.ordinal,
    }


def test_questioner_v33_invalid_generation_still_retries_before_final_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, built = _build_memlens_unit()
    config = load_questioner_config(V33_CONFIG)
    job = ExperimentJob(
        job_id="compile-job-v33-invalid-generation-retry",
        run_id=built.unit.run_id,
        snapshot_id=built.unit.snapshot_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        stage=AppStage.COMPILE,
        attempt_count=1,
        max_attempts=2,
    )

    async def no_call(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return None

    graph = QuestionerCompilationGraph(
        execution=object(),  # type: ignore[arg-type]
        gateway=object(),  # type: ignore[arg-type]
        config=config,
        load_successful_call=no_call,
        load_model_call=no_call,
        before_gate=no_call,
        persist_v3_seed=no_call,
        persist_result=no_call,
    )

    async def invalid_trace(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise QuestionerGenerationContractError("structured_output_invalid")

    monkeypatch.setattr(graph, "_v3_generation_trace", invalid_trace)
    with pytest.raises(
        QuestionerStageError,
        match="questioner_generation_structured_output_invalid",
    ):
        asyncio.run(graph.process(built, job))


def test_questioner_v32_revision_reuses_successful_prior_job_attempt() -> None:
    _, built = _build_memlens_unit()
    config = load_questioner_config(V32_CONFIG)
    candidate = _local_fact_draft(built)
    job = ExperimentJob(
        job_id="compile-job-retry",
        run_id=built.unit.run_id,
        snapshot_id=built.unit.snapshot_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        stage=AppStage.COMPILE,
        attempt_count=2,
        max_attempts=2,
    )
    generation = QuestionerGeneration(
        generation_id="generation-v32-retry",
        run_id=built.unit.run_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        job_id=job.job_id,
        model_call_id="generation-call-retry",
        attempt_number=1,
        prompt_version=config.prompt_version,
        renderer_version=config.renderer_version,
        parser_version=config.parser_version,
        prompt_hash="prompt-hash",
        input_hash=built.unit.input_hash,
        output_hash="output-hash",
        parse_status=QuestionerParseStatus.PARSED,
        raw_candidate_count=1,
        valid_candidate_count=1,
    )
    previous_call_id = questioner_revision_model_call_id(
        run_id=job.run_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        generation_id=generation.generation_id,
        candidate_scope="Q01",
        attempt_number=1,
    )
    previous_trace = ModelCallTrace(
        model_call_id=previous_call_id,
        run_id=job.run_id,
        snapshot_id=job.snapshot_id,
        conversation_id=built.unit.conversation_id,
        job_id=job.job_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        parent_model_call_id="verifier-call",
        stage=AppStage.QUESTIONER,
        model_alias=config.compiler_model_alias,
        status=TraceStatus.SUCCESS,
        parsed_output={
            "replacement": {
                "candidate_alias": "Q01",
                "generated_question": "Which collar color does Milo currently wear?",
                "question_type": candidate.question_type.value,
                "reasoning_type": candidate.reasoning_type.value,
                "paraphrases": [],
            }
        },
    )
    loaded_ids: list[str] = []

    async def load_model_call(call_id: str) -> ModelCallTrace | None:
        loaded_ids.append(call_id)
        return previous_trace if call_id == previous_call_id else None

    class NoExecution:
        async def generate_audited(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("a recovered revision must not call the model")

    async def no_call(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return None

    graph = QuestionerCompilationGraph(
        execution=NoExecution(),  # type: ignore[arg-type]
        gateway=object(),  # type: ignore[arg-type]
        config=config,
        load_successful_call=no_call,
        load_model_call=load_model_call,
        before_gate=no_call,
        persist_v3_seed=no_call,
        persist_result=no_call,
    )
    trace, replacement = asyncio.run(
        graph._run_v32_revision(
            built=built,
            job=job,
            generation=generation,
            verifier_trace=previous_trace.model_copy(
                update={"model_call_id": "verifier-call", "parent_model_call_id": None}
            ),
            candidate_alias="Q01",
            candidate=candidate,
            failed_dimensions=["user_facing"],
            reason_codes=["not_user_facing"],
        )
    )

    assert trace.model_call_id == previous_call_id
    assert replacement is not None
    assert replacement.generated_question == "Which collar color does Milo currently wear?"
    assert loaded_ids[-1] == previous_call_id
    assert len(loaded_ids) == 2


def test_questioner_v3_revision_is_reverified_before_becoming_eligible() -> None:
    config, built = _build_memlens_unit()
    assert built.opportunities is not None
    provider = _local_fact_draft(built)
    internal = resolve_v3_draft(provider, opportunity_set=built.opportunities)
    static = QuestionerCandidateGate(config).evaluate(
        run_id=built.unit.run_id,
        snapshot_id=built.unit.snapshot_id,
        unit=built.unit,
        generation_id="generation-v3-revision",
        model_call_id="generation-call",
        drafts=[internal],
        memories=list(built.memories),
        opportunity_set=built.opportunities,
        generation_ranks=[1],
        apply_dedupe=False,
    )
    assert static[0].gate_result.status is CandidateGateStatus.ELIGIBLE_FOR_PLANNING

    async def no_call(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return None

    graph = QuestionerCompilationGraph(
        execution=object(),  # type: ignore[arg-type]
        gateway=object(),  # type: ignore[arg-type]
        config=config,
        load_successful_call=no_call,
        load_model_call=no_call,
        before_gate=no_call,
        persist_v3_seed=no_call,
        persist_result=no_call,
    )
    generation = QuestionerGeneration(
        generation_id="generation-v3-revision",
        run_id=built.unit.run_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        job_id="compile-job",
        model_call_id="generation-call",
        attempt_number=1,
        prompt_version="pmmc-questioner-v3",
        renderer_version="questioner-multimodal-renderer-v3",
        parser_version="questioner-json-parser-v3",
        prompt_hash="prompt-hash",
        input_hash=built.unit.input_hash,
        output_hash="output-hash",
        parse_status=QuestionerParseStatus.PARSED,
        raw_candidate_count=1,
        valid_candidate_count=1,
    )

    def trace(call_id: str) -> ModelCallTrace:
        return ModelCallTrace(
            model_call_id=call_id,
            run_id=built.unit.run_id,
            snapshot_id=built.unit.snapshot_id,
            conversation_id=built.unit.conversation_id,
            job_id="compile-job",
            compiler_unit_id=built.unit.compiler_unit_id,
            stage=AppStage.QUESTIONER,
            model_alias="compiler-qwen3.5-9b",
            status=TraceStatus.SUCCESS,
        )

    replacement = provider.model_copy(
        update={
            "generated_question": "Which collar color did Milo most recently wear?",
            "expected_answer": "blue",
        }
    )
    initial_output = QuestionerVerificationBatchV1(
        results=[
            QuestionerVerificationItemV1(
                candidate_alias="Q01",
                verdict=QuestionerVerificationVerdict.REVISE,
                reason_codes=[QuestionerVerificationReason.NOT_FUTURE_PLAUSIBLE],
                replacement=replacement,
            )
        ]
    )
    terminal = {}
    final_candidates = []
    revised_static_pass = []
    supersessions = []
    internal_by_qbank = {static[0].candidate.qbank_id: internal}
    initial_batch, initial_facts = graph._apply_initial_verifier(
        built=built,
        generation=generation,
        trace=trace("initial-verifier-call"),
        output=initial_output,
        static_inputs=static,
        provider_by_rank={1: provider},
        internal_by_qbank=internal_by_qbank,
        terminal=terminal,
        final_candidates=final_candidates,
        revised_static_pass=revised_static_pass,
        supersessions=supersessions,
    )
    assert initial_batch.revise_count == 1
    assert initial_facts[0].replacement_qbank_id is not None
    assert len(supersessions) == 1
    assert len(revised_static_pass) == 1
    assert revised_static_pass[0][0].candidate.candidate_version == 1

    final_output = QuestionerVerificationBatchV1(
        results=[
            QuestionerVerificationItemV1(
                candidate_alias="Q01",
                verdict=QuestionerVerificationVerdict.PASS,
                reason_codes=[],
                replacement=None,
            )
        ]
    )
    final_batch, final_facts = graph._apply_final_verifier(
        generation=generation,
        trace=trace("final-verifier-call"),
        output=final_output,
        revised_inputs=revised_static_pass,
        terminal=terminal,
        final_candidates=final_candidates,
    )
    assert final_batch.pass_count == 1
    assert final_facts[0].verdict is QuestionerVerificationVerdict.PASS
    assert final_candidates[0].candidate.candidate_version == 1
