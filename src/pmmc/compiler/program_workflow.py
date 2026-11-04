"""Step 6C Questioner-to-Planner-to-dry-run workflow orchestration."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pmmc.baselines.runner import TransactionalModelCallWriter
from pmmc.compiler.config import load_compiler_config
from pmmc.compiler.doubter_config import DoubterConfig
from pmmc.compiler.doubter_runner import (
    DoubterProtocolGateError,
    DoubterRunner,
    verify_doubter_run,
)
from pmmc.compiler.memory_index_runner import (
    MemoryOperatorIndexBuilder,
    ProgramQueryEmbeddingBuilder,
    load_program_query_vectors,
)
from pmmc.compiler.planner_config import PlannerConfig, planner_config_hash
from pmmc.compiler.planner_runner import PlannerRunner, PostgresPlannerInputProvider
from pmmc.compiler.program_runner import ProgramDryRunRunner
from pmmc.compiler.questioner_config import QuestionerConfig
from pmmc.compiler.r0_gate import build_r0_input_manifest, verify_r0_input
from pmmc.compiler.verified_qbank import (
    freeze_verified_qbank,
    verify_routing_projection,
)
from pmmc.compiler.workflow import (
    ProgramWorkflowIdentity,
    QuestionerWorkflowPhase,
    run_questioner_workflow,
)
from pmmc.infrastructure.database.compiler_repositories import (
    summarize_compiler_verification_issues,
    verify_compiler_run,
)
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.errors import safe_database_error_diagnostics
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.repositories import (
    PostgresMemoryRepository,
    PostgresSnapshotRepository,
)
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.infrastructure.object_store import create_runtime_object_store
from pmmc.infrastructure.settings import Settings
from pmmc.memory.operators import ProgramValidator
from pmmc.models.factory import (
    create_embedding_gateway,
    create_generation_gateway,
    create_image_url_resolver,
    create_model_execution_service,
    create_token_counter,
)
from pmmc.models.registry import ModelRegistry
from pmmc.models.staged_token_counter import create_staged_token_counter
from pmmc.retrieval.operators import build_operator_registry
from pmmc.schemas import CompilationTargetStage, DataScope, RunStatus


def _aggregate_count(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class CompilerCompletionVerificationError(RuntimeError):
    """Carry only aggregate invariant codes across the child-process boundary."""

    diagnostic_event = "COMPILER_COMPLETION_FAILURE"
    error_code = "compiler_completion_verification_failed"
    safe_summary = "Compiler completion verification failed; inspect aggregate issue codes"

    def __init__(self, verification: Mapping[str, object]) -> None:
        raw = verification.get("issues", [])
        issues = raw if isinstance(raw, list) else [raw]
        self._diagnostics = {
            "error_code": self.error_code,
            "phase": "finalize",
            "issue_count": len(issues),
            "issue_type_counts": summarize_compiler_verification_issues(issues),
        }
        super().__init__(self.safe_summary)

    def safe_diagnostics(self) -> dict[str, object]:
        return dict(self._diagnostics)


class ProgramWorkflowPhase(StrEnum):
    ALL = "all"
    QUESTIONER = "questioner"
    PLANNER = "planner"
    QBANK_INDEX_TEXT = "qbank_index_text"
    QBANK_INDEX_MULTIMODAL = "qbank_index_multimodal"
    MEMORY_INDEX_TEXT = "memory_index_text"
    MEMORY_INDEX_MULTIMODAL = "memory_index_multimodal"
    PROGRAM_QUERY_TEXT = "program_query_text"
    PROGRAM_QUERY_MULTIMODAL = "program_query_multimodal"
    DRY_RUN = "dry_run"
    R0_INPUT = "r0_input"
    DOUBTER = "doubter"
    DECISION = "decision"
    FINALIZE = "finalize"


def _verification_issues(value: Mapping[str, object]) -> list[object]:
    raw = value.get("issues", [])
    return raw if isinstance(raw, list) else [raw]


def validate_program_protocol_compatibility(
    questioner: QuestionerConfig,
    planner: PlannerConfig,
) -> None:
    """Validate the frozen Questioner/Planner protocol matrix."""

    if questioner.protocol_version == "pmmc-questioner-v3.6" or planner.is_v15:
        if not (questioner.protocol_version == "pmmc-questioner-v3.6" and planner.is_v15):
            raise ValueError("Questioner v3.6 and Planner v1.5 must be selected together")
        return

    if not (questioner.is_v31 or questioner.is_v32 or questioner.is_v33 or questioner.is_v34):
        raise ValueError("program compilation requires Questioner v3.1 through v3.5")
    if questioner.is_v32 != planner.is_v11:
        raise ValueError("Questioner v3.2 and Planner v1.1 must be selected together")
    if questioner.is_v33 != planner.is_v12:
        raise ValueError("Questioner v3.3 and Planner v1.2 must be selected together")
    if questioner.is_v34 != planner.is_v13:
        raise ValueError("Questioner v3.4/v3.5 requires Planner v1.3/v1.4")
    if questioner.is_v35 != planner.is_v14:
        raise ValueError("Questioner v3.5 and Planner v1.4 must be selected together")


def program_completion_identity_checks(compiler) -> dict[str, bool]:  # type: ignore[no-untyped-def]
    """Select the native completion contract, without aliasing Step8 to Step7."""
    identities = {
        "0.9.0": ("v1.3", "v7", "v1.5", "v3.6", "v4", "v4"),
        "0.8.0": ("v1.2", "v6", "v1.4", "v3.5", "v3", "v3"),
    }
    d, operators, p, q, r0, projection = identities.get(
        compiler.compiler_schema_version, ("v1.1", "v5", "v1.3", "v3.4", "v2", "v2")
    )
    return {
        "active_round": compiler.active_round == 2,
        "max_round": compiler.max_round == 3,
        "doubter_prompt": compiler.doubter_prompt_version == f"pmmc-doubter-{d}",
        "operator_set": compiler.operator_set_version == f"pmmc-operators-{operators}",
        "planner_prompt": compiler.planner_prompt_version == f"pmmc-planner-{p}",
        "questioner_prompt": compiler.questioner_prompt_version == f"pmmc-questioner-{q}",
        "r0_policy": compiler.r0_input_policy_version == f"r0-input-gate-{r0}",
        "routing_projection": compiler.routing_projection_version
        == f"verified-qbank-projection-{projection}",
        "target_stage": compiler.target_stage is CompilationTargetStage.VERIFIED_QBANK,
    }


class _NoPlannerGeneration:
    async def generate(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("this treatment uses no Planner model generation")


@dataclass(frozen=True)
class ProgramWorkflowResult:
    run_id: str
    snapshot_id: str
    phase: ProgramWorkflowPhase
    unit_count: int
    eligible_candidate_count: int
    program_count: int
    terminal_outcome_count: int
    planner_outcome_counts: dict[str, int]
    memory_index_status: str | None
    run_status: str
    r0_gate_passed: bool | None = None
    doubter_assessment_count: int = 0
    routing_member_count: int = 0
    r0_program_count: int = 0
    revision_program_count: int = 0


async def _summary(
    *, settings: Settings, run_id: str, phase: ProgramWorkflowPhase
) -> ProgramWorkflowResult:
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
            assert uow.compiler_runs is not None
            assert uow.compiler_units is not None
            assert uow.questioner is not None
            assert uow.memory_programs is not None
            assert uow.planner is not None
            assert uow.memory_operator_indexes is not None
            assert uow.doubter is not None
            assert uow.runs is not None
            compiler = await uow.compiler_runs.get(run_id)
            if compiler is None:
                raise KeyError(f"compiler run does not exist: {run_id}")
            units = await uow.compiler_units.list_for_run(run_id)
            candidates = await uow.questioner.list_eligible_questions(run_id)
            programs = await uow.memory_programs.list_for_run(run_id)
            outcomes = await uow.planner.list_outcomes(run_id)
            memory_index = await uow.memory_operator_indexes.get_for_run(run_id)
            r0_manifest = await uow.doubter.get_r0_manifest(run_id)
            assessments = await uow.doubter.list_assessments(run_id)
            routing = await uow.doubter.get_routing_projection(run_id)
            state = await uow.runs.get_state(run_id)
        return ProgramWorkflowResult(
            run_id=run_id,
            snapshot_id=compiler.snapshot_id,
            phase=phase,
            unit_count=len(units),
            eligible_candidate_count=len(candidates),
            program_count=len(programs),
            r0_program_count=sum(item.round_id == 0 for item in programs),
            revision_program_count=sum(item.round_id > 0 for item in programs),
            terminal_outcome_count=len(outcomes),
            planner_outcome_counts=dict(
                sorted(Counter(item.status.value for item in outcomes).items())
            ),
            memory_index_status=(memory_index.status if memory_index is not None else None),
            run_status=state.status.value,
            r0_gate_passed=(r0_manifest.hard_gate_passed if r0_manifest is not None else None),
            doubter_assessment_count=len(assessments),
            routing_member_count=(len(routing[1]) if routing is not None else 0),
        )
    finally:
        await engine.dispose()


async def _load_run_context(*, settings: Settings, run_id: str):  # type: ignore[no-untyped-def]
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
            assert uow.compiler_runs is not None and uow.compiler_units is not None
            compiler = await uow.compiler_runs.get(run_id)
            if compiler is None:
                raise KeyError(f"compiler run does not exist: {run_id}")
            if compiler.target_stage not in {
                CompilationTargetStage.PROGRAMS,
                CompilationTargetStage.VERIFIED_QBANK,
            }:
                raise ValueError("run is not a Planner compiler run")
            units = await uow.compiler_units.list_for_run(run_id)
        snapshot = await PostgresSnapshotRepository(sessions).get_manifest(compiler.snapshot_id)
        conversations = compiler.metadata.get("conversation_ids")
        if not isinstance(conversations, list) or not all(
            isinstance(value, str) for value in conversations
        ):
            raise ValueError("program run has no frozen conversation denominator")
        memories = await PostgresMemoryRepository(sessions).list_snapshot(
            compiler.snapshot_id,
            conversation_ids=set(conversations),
        )
        return engine, sessions, compiler, units, snapshot, memories
    except Exception:
        await engine.dispose()
        raise


def _execution_services(*, settings: Settings, registry: ModelRegistry, sessions):  # type: ignore[no-untyped-def]
    store = create_runtime_object_store(settings)
    artifacts = ExperimentArtifactService(
        sessions,
        store,
        DataScope.RUNTIME,
        artifact_prefix=settings.experiment_artifact_prefix,
    )
    execution = create_model_execution_service(
        settings,
        registry,
        artifact_service=artifacts,
        model_calls=TransactionalModelCallWriter(sessions),
    )
    resolver = create_image_url_resolver(settings, store)
    return execution, artifacts, resolver


async def run_program_workflow(
    *,
    settings: Settings,
    registry: ModelRegistry,
    questioner: QuestionerConfig,
    planner: PlannerConfig,
    snapshot_id: str,
    doubter: DoubterConfig | None = None,
    questioner_freeze_or_audit_hash: str,
    questioner_identity_kind: str,
    diagnostic_force_revision: bool = False,
    phase: ProgramWorkflowPhase = ProgramWorkflowPhase.ALL,
    run_id: str | None = None,
    conversation_ids: set[str] | None = None,
    limit: int | None = None,
    sample_size: int | None = None,
    seed: int = 0,
    shard_count: int = 1,
    shard_index: int = 0,
    split: str | None = None,
    unit_preparation: Path | None = None,
    on_questioner_item=None,  # type: ignore[no-untyped-def]
    on_questioner_progress=None,  # type: ignore[no-untyped-def]
    on_planner_item=None,  # type: ignore[no-untyped-def]
    on_index_progress=None,  # type: ignore[no-untyped-def]
    on_dry_run_item=None,  # type: ignore[no-untyped-def]
    on_doubter_progress=None,  # type: ignore[no-untyped-def]
) -> ProgramWorkflowResult:
    """Run or resume one immutable Questioner + Planner compiler run."""

    validate_program_protocol_compatibility(questioner, planner)
    if questioner_identity_kind == "step8_method_protocol" and not planner.is_v15:
        raise ValueError("step8_method_protocol is only valid for compiler 0.9")
    foundation = load_compiler_config(settings.resolved_compiler_config_path)
    if foundation.operator_set_version != planner.operator_set_version:
        raise ValueError("compiler foundation and Planner operator sets must match")
    if planner.is_v15:
        if foundation.total_evidence_max != planner.total_evidence_max:
            raise ValueError("Step8 Planner and executor evidence caps must match")
        if doubter is not None and not doubter.is_v13:
            raise ValueError("Step8 Planner requires native Doubter v1.3")
        if diagnostic_force_revision:
            raise ValueError("Step7 diagnostic forced revisions are not a Step8 treatment")
    identity = ProgramWorkflowIdentity(
        questioner_freeze_or_audit_hash=questioner_freeze_or_audit_hash,
        questioner_identity_kind=questioner_identity_kind,
        planner=planner,
        memory_text_model_alias=questioner.text_embedding_model_alias,
        memory_multimodal_model_alias=questioner.multimodal_embedding_model_alias,
        doubter=doubter,
        diagnostic_force_revision=diagnostic_force_revision,
    )

    if phase is ProgramWorkflowPhase.ALL:
        questioner_result = await run_program_workflow(
            settings=settings,
            registry=registry,
            questioner=questioner,
            planner=planner,
            doubter=doubter,
            snapshot_id=snapshot_id,
            questioner_freeze_or_audit_hash=questioner_freeze_or_audit_hash,
            questioner_identity_kind=questioner_identity_kind,
            diagnostic_force_revision=diagnostic_force_revision,
            phase=ProgramWorkflowPhase.QUESTIONER,
            run_id=run_id,
            conversation_ids=conversation_ids,
            limit=limit,
            sample_size=sample_size,
            seed=seed,
            shard_count=shard_count,
            shard_index=shard_index,
            split=split,
            unit_preparation=unit_preparation,
            on_questioner_item=on_questioner_item,
            on_questioner_progress=on_questioner_progress,
        )
        resolved_run_id = questioner_result.run_id
        phases = [
            ProgramWorkflowPhase.PLANNER,
            ProgramWorkflowPhase.QBANK_INDEX_TEXT,
            ProgramWorkflowPhase.QBANK_INDEX_MULTIMODAL,
            ProgramWorkflowPhase.MEMORY_INDEX_TEXT,
            ProgramWorkflowPhase.MEMORY_INDEX_MULTIMODAL,
            ProgramWorkflowPhase.PROGRAM_QUERY_TEXT,
            ProgramWorkflowPhase.PROGRAM_QUERY_MULTIMODAL,
            ProgramWorkflowPhase.DRY_RUN,
        ]
        if doubter is not None:
            phases.extend(
                [
                    ProgramWorkflowPhase.R0_INPUT,
                    ProgramWorkflowPhase.DOUBTER,
                    ProgramWorkflowPhase.DECISION,
                ]
            )
        phases.append(ProgramWorkflowPhase.FINALIZE)
        for next_phase in phases:
            await run_program_workflow(
                settings=settings,
                registry=registry,
                questioner=questioner,
                planner=planner,
                doubter=doubter,
                snapshot_id=snapshot_id,
                questioner_freeze_or_audit_hash=questioner_freeze_or_audit_hash,
                questioner_identity_kind=questioner_identity_kind,
                diagnostic_force_revision=diagnostic_force_revision,
                phase=next_phase,
                run_id=resolved_run_id,
                split=split,
                on_planner_item=on_planner_item,
                on_index_progress=on_index_progress,
                on_dry_run_item=on_dry_run_item,
                on_doubter_progress=on_doubter_progress,
            )
        return await _summary(
            settings=settings, run_id=resolved_run_id, phase=ProgramWorkflowPhase.ALL
        )

    if phase is ProgramWorkflowPhase.QUESTIONER:
        result = await run_questioner_workflow(
            settings=settings,
            registry=registry,
            config=questioner,
            snapshot_id=snapshot_id,
            run_id=run_id,
            conversation_ids=conversation_ids,
            limit=limit,
            sample_size=sample_size,
            seed=seed,
            shard_count=shard_count,
            shard_index=shard_index,
            split=split,
            phase=QuestionerWorkflowPhase.QUESTIONER,
            on_questioner_item=on_questioner_item,
            on_questioner_progress=on_questioner_progress,
            unit_preparation=unit_preparation,
            program_identity=identity,
        )
        return await _summary(settings=settings, run_id=result.run_id, phase=phase)

    if run_id is None:
        raise ValueError("resumed Step 6C phases require --run-id")

    if phase in {
        ProgramWorkflowPhase.QBANK_INDEX_TEXT,
        ProgramWorkflowPhase.QBANK_INDEX_MULTIMODAL,
    }:
        lookup_engine, lookup_sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with ExperimentUnitOfWork(lookup_sessions, DataScope.RUNTIME) as lookup_uow:
                assert lookup_uow.compiler_runs is not None
                assert lookup_uow.runs is not None
                stored_compiler = await lookup_uow.compiler_runs.get(run_id)
                if stored_compiler is None:
                    raise KeyError(f"compiler run does not exist: {run_id}")
                stored_manifest = await lookup_uow.runs.get_manifest(run_id)
            frozen_conversations = stored_compiler.metadata.get("conversation_ids")
            if not isinstance(frozen_conversations, list) or not all(
                isinstance(value, str) for value in frozen_conversations
            ):
                raise ValueError("program run has no frozen conversation denominator")
        finally:
            await lookup_engine.dispose()
        qbank_phase = (
            QuestionerWorkflowPhase.INDEX_TEXT
            if phase is ProgramWorkflowPhase.QBANK_INDEX_TEXT
            else QuestionerWorkflowPhase.INDEX_MULTIMODAL
        )
        await run_questioner_workflow(
            settings=settings,
            registry=registry,
            config=questioner,
            snapshot_id=snapshot_id,
            run_id=run_id,
            conversation_ids=set(frozen_conversations),
            seed=int(stored_manifest.seed or 0),
            shard_count=int(stored_manifest.metadata.get("shard_count", 1)),
            shard_index=int(stored_manifest.metadata.get("shard_index", 0)),
            split=stored_manifest.split,
            phase=qbank_phase,
            on_index_progress=on_index_progress,
            program_identity=identity,
        )
        return await _summary(settings=settings, run_id=run_id, phase=phase)

    engine, sessions, compiler, units, snapshot, memories = await _load_run_context(
        settings=settings, run_id=run_id
    )
    try:
        if compiler.snapshot_id != snapshot_id:
            raise ValueError("program phase snapshot conflicts with frozen run")
        if planner.is_v15 and (
            compiler.compiler_schema_version != "0.9.0"
            or compiler.planner_config_hash != planner_config_hash(planner)
        ):
            raise ValueError("Step8 compiler schema or treatment differs from the frozen run")
        if bool(compiler.metadata.get("diagnostic_force_revision", False)) != (
            diagnostic_force_revision
        ):
            raise ValueError("program diagnostic revision identity conflicts with frozen run")
        execution, _, resolver = _execution_services(
            settings=settings, registry=registry, sessions=sessions
        )
        if phase is ProgramWorkflowPhase.PLANNER:
            gateway = (
                _NoPlannerGeneration()
                if planner.is_v15 and planner.planning_mode == "fixed"
                else create_generation_gateway(
                    settings,
                    registry,
                    planner.planner_model_alias,
                    image_url_resolver=resolver,
                )
            )
            static_registry = build_operator_registry(object(), foundation)  # type: ignore[arg-type]
            planner_result = await PlannerRunner(
                sessions=sessions,
                execution=execution,
                gateway=gateway,
                config=planner,
                static_validator=ProgramValidator(static_registry, foundation),
                lease_seconds=settings.experiment_job_lease_seconds,
                concurrency=settings.model_max_concurrency,
                input_provider=PostgresPlannerInputProvider(sessions),
                diagnostic_force_weak_r0=diagnostic_force_revision,
            ).execute(
                run_id=run_id,
                snapshot_id=snapshot_id,
                units=units,
                on_item=on_planner_item,
            )
            if planner_result.unit_count != len(units):
                raise RuntimeError("Planner unit denominator changed")
        elif phase in {
            ProgramWorkflowPhase.MEMORY_INDEX_TEXT,
            ProgramWorkflowPhase.MEMORY_INDEX_MULTIMODAL,
        }:
            channel = (
                "text_memory"
                if phase is ProgramWorkflowPhase.MEMORY_INDEX_TEXT
                else "visual_memory"
            )
            alias = (
                questioner.text_embedding_model_alias
                if channel == "text_memory"
                else questioner.multimodal_embedding_model_alias
            )
            builder = MemoryOperatorIndexBuilder(
                sessions=sessions,
                execution=execution,
                gateways={
                    alias: create_embedding_gateway(
                        settings, registry, alias, image_url_resolver=resolver
                    )
                },
                registry=registry,
                text_batch_size=questioner.text_embedding_batch_size,
                multimodal_batch_size=questioner.multimodal_embedding_batch_size,
            )
            await builder.build(
                run_id=run_id,
                snapshot_id=snapshot_id,
                dataset=snapshot.dataset.value,
                selection_hash=str(compiler.metadata["selection_hash"]),
                memories=memories,
                text_model_alias=questioner.text_embedding_model_alias,
                multimodal_model_alias=questioner.multimodal_embedding_model_alias,
                progress=on_index_progress,
                channels={channel},
                memory_raw_image_policy=planner.memory_raw_image_policy,
            )
        elif phase in {
            ProgramWorkflowPhase.PROGRAM_QUERY_TEXT,
            ProgramWorkflowPhase.PROGRAM_QUERY_MULTIMODAL,
        }:
            channel = (
                "question_text_query"
                if phase is ProgramWorkflowPhase.PROGRAM_QUERY_TEXT
                else "question_joint_query"
            )
            alias = (
                questioner.text_embedding_model_alias
                if channel == "question_text_query"
                else questioner.multimodal_embedding_model_alias
            )
            async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                assert uow.questioner is not None
                candidates = await uow.questioner.list_eligible_questions(run_id)
            await ProgramQueryEmbeddingBuilder(
                sessions=sessions,
                execution=execution,
                gateways={
                    alias: create_embedding_gateway(
                        settings, registry, alias, image_url_resolver=resolver
                    )
                },
                registry=registry,
                text_batch_size=settings.qbank_text_query_batch_size,
                multimodal_batch_size=settings.qbank_multimodal_query_batch_size,
            ).build(
                run_id=run_id,
                snapshot_id=snapshot_id,
                candidates=candidates,
                text_model_alias=questioner.text_embedding_model_alias,
                multimodal_model_alias=questioner.multimodal_embedding_model_alias,
                channels={channel},
            )
        elif phase is ProgramWorkflowPhase.DRY_RUN:
            async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                assert uow.memory_operator_indexes is not None
                memory_index = await uow.memory_operator_indexes.get_for_run(run_id)
            if memory_index is None:
                raise RuntimeError("program dry-run has no memory operator index")
            vectors = await load_program_query_vectors(sessions, run_id=run_id)
            await ProgramDryRunRunner(
                sessions=sessions,
                config=foundation,
                lease_seconds=settings.experiment_job_lease_seconds,
                concurrency=(
                    settings.compiler_cpu_workers
                    if planner.is_v15 and settings.compiler_cpu_workers is not None
                    else min(32, settings.database_pool_size + settings.database_max_overflow)
                ),
            ).execute(
                run_id=run_id,
                snapshot_id=snapshot_id,
                units=units,
                memory_index=memory_index,
                query_vectors=vectors,
                on_item=on_dry_run_item,
            )
        elif phase is ProgramWorkflowPhase.R0_INPUT:
            if doubter is None:
                raise ValueError("R0 input phase requires a Doubter configuration")
            await build_r0_input_manifest(
                sessions, run_id=run_id,
                planner_config=planner if planner.is_v15 else None,
            )
        elif phase is ProgramWorkflowPhase.DOUBTER:
            if doubter is None:
                raise ValueError("Doubter phase requires a Doubter configuration")
            await verify_r0_input(
                sessions, run_id=run_id,
                planner_config=planner if planner.is_v15 else None,
            )
            async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                assert uow.memory_operator_indexes is not None
                memory_index = await uow.memory_operator_indexes.get_for_run(run_id)
            if memory_index is None or memory_index.status != "ready":
                raise RuntimeError("Doubter requires a ready memory operator index")
            vectors = await load_program_query_vectors(sessions, run_id=run_id)
            doubter_gateway = create_generation_gateway(
                settings,
                registry,
                doubter.doubter_model_alias,
                image_url_resolver=resolver,
            )
            planner_gateway = (
                _NoPlannerGeneration()
                if planner.is_v15
                else create_generation_gateway(
                    settings,
                    registry,
                    planner.planner_model_alias,
                    image_url_resolver=resolver,
                )
            )
            static_registry = build_operator_registry(object(), foundation)  # type: ignore[arg-type]
            doubter_token_counter = create_staged_token_counter(
                settings,
                registry,
                doubter.doubter_model_alias,
                create_runtime_object_store(settings),
            )
            await DoubterRunner(
                sessions=sessions,
                execution=execution,
                doubter_gateway=doubter_gateway,
                planner_gateway=planner_gateway,
                doubter_config=doubter,
                planner_config=planner,
                foundation=foundation,
                static_validator=ProgramValidator(static_registry, foundation),
                lease_seconds=settings.experiment_job_lease_seconds,
                concurrency=settings.model_max_concurrency,
                dry_run_concurrency=(
                    settings.compiler_cpu_workers
                    if planner.is_v15 and settings.compiler_cpu_workers is not None
                    else min(32, settings.database_pool_size + settings.database_max_overflow)
                ),
                database_work_concurrency=min(
                    24,
                    settings.database_pool_size + settings.database_max_overflow,
                ),
                token_counter=doubter_token_counter,
                # Used only to locate an already journaled response, never for new dispatch.
                recovery_token_counter=create_token_counter(
                    settings, registry, doubter.doubter_model_alias, allow_estimated=True,
                ),
                diagnostic_force_revision=diagnostic_force_revision,
            ).execute(
                run_id=run_id,
                snapshot_id=snapshot_id,
                units=units,
                memory_index=memory_index,
                query_vectors=vectors,
                on_progress=on_doubter_progress,
            )
            doubter_verification = await verify_doubter_run(
                sessions,
                run_id=run_id,
            )
            print(
                "DOUBTER_PROTOCOL_PREFLIGHT "
                + json.dumps(
                    {
                        "assessment_count": _aggregate_count(
                            doubter_verification, "assessment_count"
                        ),
                        "issue_codes": _verification_issues(doubter_verification),
                        "provider_nonconformant_count": _aggregate_count(
                            doubter_verification, "provider_nonconformant_count"
                        ),
                        "provider_schema_conformant_count": _aggregate_count(
                            doubter_verification, "provider_schema_conformant_count"
                        ),
                        "provider_terminal_rejection_count": _aggregate_count(
                            doubter_verification, "provider_terminal_rejection_count"
                        ),
                        "provider_integrity_violation_count": _aggregate_count(
                            doubter_verification, "provider_integrity_violation_count"
                        ),
                        "next_stages": ["decision", "routing_projection_verify"],
                        "state": ("ready" if bool(doubter_verification["valid"]) else "blocked"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if not bool(doubter_verification["valid"]):
                raise DoubterProtocolGateError(doubter_verification, phase="doubter")
        elif phase is ProgramWorkflowPhase.DECISION:
            if doubter is None:
                raise ValueError("decision phase requires a Doubter configuration")
            doubter_verification = await verify_doubter_run(
                sessions,
                run_id=run_id,
            )
            if not bool(doubter_verification["valid"]):
                raise DoubterProtocolGateError(doubter_verification, phase="decision")
            await freeze_verified_qbank(sessions, run_id=run_id)
            verification = await verify_routing_projection(sessions, run_id=run_id)
            if not bool(verification["valid"]):
                raise RuntimeError(
                    "verified QBank projection failed: "
                    + "; ".join(str(value) for value in _verification_issues(verification)[:5])
                )
        elif phase is ProgramWorkflowPhase.FINALIZE:
            routing_verification: Mapping[str, object] = {}
            if doubter is not None:
                await verify_r0_input(
                    sessions, run_id=run_id,
                    planner_config=planner if planner.is_v15 else None,
                )
                doubter_verification = await verify_doubter_run(
                    sessions,
                    run_id=run_id,
                )
                if not bool(doubter_verification["valid"]):
                    raise DoubterProtocolGateError(doubter_verification, phase="finalize")
                routing_verification = await verify_routing_projection(sessions, run_id=run_id)
                if not bool(routing_verification["valid"]):
                    raise RuntimeError(
                        "routing projection verification failed: "
                        + "; ".join(
                            str(value) for value in _verification_issues(routing_verification)[:5]
                        )
                    )
            async with sessions() as session:
                verification = await verify_compiler_run(session, run_id)
            if not bool(verification["valid"]):
                raise CompilerCompletionVerificationError(verification)
            compiler_issues = verification.get("issues", [])
            routing_issues = routing_verification.get("issues", [])
            step7_identity_checks = program_completion_identity_checks(compiler)
            completion_preflight = {
                "candidate_count": _aggregate_count(routing_verification, "candidate_count"),
                "compiler_issue_count": (
                    len(compiler_issues) if isinstance(compiler_issues, list) else 1
                ),
                "decision_count": _aggregate_count(routing_verification, "decision_count"),
                "decision_job_count": _aggregate_count(routing_verification, "decision_job_count"),
                "projection_count": _aggregate_count(routing_verification, "projection_count"),
                "routing_issue_count": (
                    len(routing_issues) if isinstance(routing_issues, list) else 1
                ),
                "routing_member_count": _aggregate_count(
                    routing_verification, "routing_member_count"
                ),
                "step7_identity_checks": step7_identity_checks,
                "state": ("ready" if all(step7_identity_checks.values()) else "blocked"),
            }
            print(
                "STEP7_COMPLETION_PREFLIGHT "
                + json.dumps(
                    completion_preflight,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if not all(step7_identity_checks.values()):
                raise RuntimeError(
                    "Step 7 compiler identity preflight failed before database completion"
                )
            async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                assert uow.runs is not None
                state = await uow.runs.get_state(run_id)
                try:
                    if state.status is RunStatus.RUNNING:
                        await uow.runs.transition(run_id, RunStatus.COMPLETED)
                    elif state.status is not RunStatus.COMPLETED:
                        raise RuntimeError("program run cannot be finalized from current state")
                except Exception as exc:
                    diagnostics: dict[str, object] = {
                        **completion_preflight,
                        **safe_database_error_diagnostics(exc),
                        "state": "failed",
                    }
                    print(
                        "STEP7_COMPLETION_GUARD_FAILURE "
                        + json.dumps(
                            diagnostics,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                    raise
        else:
            raise ValueError(f"unsupported program workflow phase: {phase.value}")
    finally:
        await engine.dispose()
    return await _summary(settings=settings, run_id=run_id, phase=phase)


__all__ = [
    "ProgramWorkflowPhase",
    "ProgramWorkflowResult",
    "run_program_workflow",
]
