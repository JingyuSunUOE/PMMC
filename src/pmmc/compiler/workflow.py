"""End-to-end Questioner run registration, production, and QBank indexing."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from pmmc.baselines.runner import TransactionalModelCallWriter
from pmmc.compiler.config import compiler_config_hash, load_compiler_config
from pmmc.compiler.doubter_config import DoubterConfig, doubter_config_hash
from pmmc.compiler.doubter_prompt import (
    doubter_prompt_sha256,
    doubter_response_schema_sha256,
)
from pmmc.compiler.ids import (
    COMPILER_UNIT_RUN_SCOPED_IDENTITY_V1,
    canonical_content_hash,
)
from pmmc.compiler.planner_config import PlannerConfig, planner_config_hash
from pmmc.compiler.planner_prompt import (
    planner_prompt_sha256,
    planner_response_schema_sha256,
)
from pmmc.compiler.preparation import CompilationUnitPreparationService
from pmmc.compiler.questioner_config import QuestionerConfig, questioner_config_hash
from pmmc.compiler.questioner_prompt import (
    questioner_prompt_sha256,
    questioner_request_contract_hash,
    questioner_response_schema_sha256,
)
from pmmc.compiler.questioner_runner import (
    PostgresQuestionerInputProvider,
    QuestionerRunner,
    QuestionerRunResult,
)
from pmmc.compiler.units import create_compilation_unit_builder
from pmmc.experiments import run_config_hash
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.repositories import (
    PostgresMemoryRepository,
    PostgresSessionRepository,
    PostgresSnapshotRepository,
)
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.infrastructure.object_store import create_runtime_object_store
from pmmc.infrastructure.settings import AppMode, Settings
from pmmc.models.factory import (
    create_embedding_gateway,
    create_generation_gateway,
    create_image_url_resolver,
    create_model_execution_service,
    create_token_counter,
)
from pmmc.models.registry import ModelRegistry, bind_registry_to_manifest
from pmmc.qbank.index import prepare_qbank_index
from pmmc.retrieval.operators import build_operator_registry
from pmmc.schemas import (
    CompilationStage,
    CompilationTargetStage,
    CompilerRunSpec,
    DataScope,
    ModelCapability,
    RunManifest,
    RunStatus,
)


@dataclass(frozen=True)
class QuestionerWorkflowResult:
    run_id: str
    snapshot_id: str
    conversation_count: int
    unit_count: int
    eligible_question_count: int
    paraphrase_count: int
    qbank_index_id: str | None
    reused_index: bool


@dataclass(frozen=True)
class ProgramWorkflowIdentity:
    """Immutable Planner/index/Doubter identity attached before run registration."""

    # step8_method_protocol stores the compiler-owned subprotocol hash. The
    # campaign manifest owns the association to the full approved protocol.
    questioner_freeze_or_audit_hash: str
    questioner_identity_kind: str
    planner: PlannerConfig
    memory_text_model_alias: str
    memory_multimodal_model_alias: str
    doubter: DoubterConfig | None = None
    diagnostic_force_revision: bool = False

    def __post_init__(self) -> None:
        if not self.questioner_freeze_or_audit_hash:
            raise ValueError("program run requires a Questioner freeze or pilot audit hash")
        if self.questioner_identity_kind not in {
            "approved_freeze",
            "development_audit",
            "step8_method_protocol",
        }:
            raise ValueError("invalid Questioner identity kind for program run")


class CompilerWorkflowContractError(ValueError):
    """A structural compiler failure that is safe to report without source text."""

    diagnostic_event = "COMPILER_WORKFLOW_CONTRACT_FAILURE"

    def __init__(self, *, stage: str, code: str, failed_checks: list[str]) -> None:
        super().__init__(code)
        self.stage = stage
        self.code = code
        self.failed_checks = sorted(set(failed_checks))
        self.safe_summary = f"stage={stage} code={code}"

    def safe_diagnostics(self) -> dict[str, object]:
        return {
            "code": self.code,
            "failed_checks": self.failed_checks,
            "stage": self.stage,
        }


class QuestionerWorkflowPhase(StrEnum):
    ALL = "all"
    QUESTIONER = "questioner"
    INDEX_TEXT = "index_text"
    INDEX_MULTIMODAL = "index_multimodal"


def questioner_workflow_manifest_versions(config: QuestionerConfig) -> dict[str, str]:
    """Pin Step8 explicitly without changing the defaults of historical manifests."""
    return {
        "compiler_schema_version": config.compiler_schema_version,
        **({"schema_version": "0.15.0"} if config.is_v36 else {}),
    }


def questioner_uses_run_scoped_units(config: QuestionerConfig) -> bool:
    return config.compiler_schema_version in {"0.7.0", "0.8.0", "0.9.0"}


def _validate_step8_program_identity(
    config: QuestionerConfig,
    identity: ProgramWorkflowIdentity | None,
) -> None:
    if not config.is_v36:
        if identity is not None and identity.questioner_identity_kind == "step8_method_protocol":
            raise ValueError("step8_method_protocol requires Questioner v3.6 / compiler 0.9")
        return
    if identity is None or identity.planner.protocol_version != "pmmc-planner-v1.5":
        raise ValueError("Questioner v3.6 requires the Step8 Planner v1.5 identity")
    if identity.planner.memory_raw_image_policy != config.memory_raw_image_policy:
        raise ValueError("Step8 Questioner and Planner raw-image treatments must match")
    if identity.doubter is not None and identity.doubter.protocol_version != "pmmc-doubter-v1.3":
        raise ValueError("Questioner v3.6 requires Doubter v1.3 when Doubter is enabled")


def _select_conversations(
    values: list[str],
    *,
    requested: set[str] | None,
    limit: int | None,
    sample_size: int | None,
    seed: int,
    shard_count: int,
    shard_index: int,
) -> list[str]:
    if limit is not None and sample_size is not None:
        raise ValueError("conversation limit and sample size are mutually exclusive")
    if limit is not None and limit <= 0:
        raise ValueError("conversation limit must be positive")
    if sample_size is not None and sample_size <= 0:
        raise ValueError("conversation sample size must be positive")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid conversation shard")
    available = sorted(set(values))
    if requested is not None:
        missing = sorted(requested - set(available))
        if missing:
            raise KeyError(f"unknown conversation ID: {missing[0]}")
        available = [item for item in available if item in requested]
    available = [
        item
        for item in available
        if int(hashlib.sha256(item.encode("utf-8")).hexdigest(), 16) % shard_count == shard_index
    ]
    if sample_size is not None and sample_size < len(available):
        available = sorted(random.Random(seed).sample(available, sample_size))
    if limit is not None:
        available = available[:limit]
    if not available:
        raise ValueError("Questioner conversation selection is empty")
    return available


async def _run_index_phase(
    *,
    settings: Settings,
    registry: ModelRegistry,
    config: QuestionerConfig,
    sessions,  # type: ignore[no-untyped-def]
    run_id: str,
    snapshot_id: str,
    conversation_count: int,
    unit_values: list,
    phase: QuestionerWorkflowPhase,
    complete_run: bool = True,
    on_index_progress=None,  # type: ignore[no-untyped-def]
) -> QuestionerWorkflowResult:
    async with ExperimentUnitOfWork(sessions) as uow:
        assert uow.compiler_units is not None
        missing = [
            unit.compiler_unit_id
            for unit in unit_values
            if await uow.compiler_units.get_stage_result(
                unit.compiler_unit_id, CompilationStage.QUESTIONER
            )
            is None
        ]
    if missing:
        raise RuntimeError("QBank indexing requires completed Questioner facts for every unit")

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
    channels = {
        "text_question",
        "text_paraphrase",
        "multimodal_question",
        "multimodal_paraphrase",
    }
    finalize = True
    if phase is QuestionerWorkflowPhase.INDEX_TEXT:
        channels = {"text_question", "text_paraphrase"}
        finalize = False
    elif phase is QuestionerWorkflowPhase.INDEX_MULTIMODAL:
        channels = {"multimodal_question", "multimodal_paraphrase"}
    aliases = {
        config.multimodal_embedding_model_alias
        if channel.startswith("multimodal")
        else config.text_embedding_model_alias
        for channel in channels
    }
    gateways = {
        alias: create_embedding_gateway(settings, registry, alias, image_url_resolver=resolver)
        for alias in aliases
    }
    result = await prepare_qbank_index(
        sessions=sessions,
        execution=execution,
        gateways=gateways,
        error_artifacts=artifacts,
        registry=registry,
        config=config,
        compiler_run_id=run_id,
        snapshot_id=snapshot_id,
        lease_seconds=settings.experiment_job_lease_seconds,
        progress=on_index_progress,
        channels=channels,
        finalize=finalize,
    )
    ready = result.manifest
    if ready.status.value == "ready" and complete_run:
        async with ExperimentUnitOfWork(sessions) as uow:
            assert uow.runs is not None
            state = await uow.runs.get_state(run_id)
            if state.status is RunStatus.RUNNING:
                await uow.runs.transition(run_id, RunStatus.COMPLETED)
    return QuestionerWorkflowResult(
        run_id=run_id,
        snapshot_id=snapshot_id,
        conversation_count=conversation_count,
        unit_count=len(unit_values),
        eligible_question_count=ready.eligible_question_count,
        paraphrase_count=ready.paraphrase_count,
        qbank_index_id=ready.qbank_index_id,
        reused_index=result.reused,
    )


async def run_questioner_workflow(
    *,
    settings: Settings,
    registry: ModelRegistry,
    config: QuestionerConfig,
    snapshot_id: str,
    run_id: str | None = None,
    conversation_ids: set[str] | None = None,
    limit: int | None = None,
    sample_size: int | None = None,
    seed: int = 0,
    shard_count: int = 1,
    shard_index: int = 0,
    split: str | None = None,
    phase: QuestionerWorkflowPhase = QuestionerWorkflowPhase.ALL,
    on_questioner_item=None,  # type: ignore[no-untyped-def]
    on_questioner_progress=None,  # type: ignore[no-untyped-def]
    on_index_progress=None,  # type: ignore[no-untyped-def]
    unit_preparation: Path | None = None,
    program_identity: ProgramWorkflowIdentity | None = None,
) -> QuestionerWorkflowResult:
    """Produce the questions target and freeze all four QBank index channels."""

    target_stage = (
        (
            CompilationTargetStage.VERIFIED_QBANK
            if program_identity.doubter is not None
            else CompilationTargetStage.PROGRAMS
        )
        if program_identity is not None
        else CompilationTargetStage.QUESTIONS
    )
    program_compiler_versions = {
        "0.4.0",
        "0.5.0",
        "0.6.0",
        "0.7.0",
        "0.8.0",
        "0.9.0",
    }
    if config.compiler_schema_version in program_compiler_versions and program_identity is None:
        raise ValueError("Questioner v3.1+ must be registered as a programs compiler run")
    if (
        config.compiler_schema_version not in program_compiler_versions
        and program_identity is not None
    ):
        raise ValueError("Planner requires a Questioner programs compiler contract")
    _validate_step8_program_identity(config, program_identity)
    compiler_entry = registry.get(config.compiler_model_alias)
    compiler_entry.require(ModelCapability.CHAT_MULTIMODAL)
    text_entry = registry.get(config.text_embedding_model_alias)
    text_entry.require(ModelCapability.EMBED_TEXT)
    multimodal_entry = registry.get(config.multimodal_embedding_model_alias)
    multimodal_entry.require(ModelCapability.EMBED_MULTIMODAL)
    planner_entry = None
    memory_text_entry = None
    memory_multimodal_entry = None
    doubter_entry = None
    if program_identity is not None:
        planner_entry = registry.get(program_identity.planner.planner_model_alias)
        planner_entry.require(ModelCapability.CHAT_MULTIMODAL)
        memory_text_entry = registry.get(program_identity.memory_text_model_alias)
        memory_text_entry.require(ModelCapability.EMBED_TEXT)
        memory_multimodal_entry = registry.get(program_identity.memory_multimodal_model_alias)
        memory_multimodal_entry.require(ModelCapability.EMBED_MULTIMODAL)
        if program_identity.doubter is not None:
            doubter_entry = registry.get(program_identity.doubter.doubter_model_alias)
            doubter_entry.require(ModelCapability.CHAT_MULTIMODAL)
    if settings.mode is AppMode.PAPER:
        registry.assert_paper_ready()

    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        snapshot = await PostgresSnapshotRepository(sessions).get_manifest(snapshot_id)
        all_sessions = await PostgresSessionRepository(sessions).list_snapshot(snapshot_id)
        selected_conversations = _select_conversations(
            [item.conversation_id for item in all_sessions],
            requested=conversation_ids,
            limit=limit,
            sample_size=sample_size,
            seed=seed,
            shard_count=shard_count,
            shard_index=shard_index,
        )
        selected_set = set(selected_conversations)
        sessions_for_run = [item for item in all_sessions if item.conversation_id in selected_set]
        q_config_hash = questioner_config_hash(config)
        q_request_contract_hash = questioner_request_contract_hash(
            prompt_version=config.prompt_version,
            renderer_version=config.renderer_version,
            opportunity_builder_version=(
                config.opportunity_builder_version if config.is_v3 else None
            ),
        )
        compiler_schema_version = config.compiler_schema_version
        run_recipe = {
            **(questioner_workflow_manifest_versions(config) if config.is_v36 else {}),
            **(
                {"compiler_subprotocol_hash": program_identity.questioner_freeze_or_audit_hash}
                if program_identity is not None
                and program_identity.questioner_identity_kind == "step8_method_protocol"
                else {}
            ),
            "snapshot_id": snapshot_id,
            "conversation_ids": selected_conversations,
            "questioner_config_hash": q_config_hash,
            "questioner_request_contract_hash": q_request_contract_hash,
            **(
                {
                    "questioner_response_schema_hash": (
                        questioner_response_schema_sha256(config.prompt_version)
                    )
                }
                if config.is_v2 or config.is_v3
                else {}
            ),
            "compiler_model_revision": compiler_entry.model_revision,
            "text_embedding_revision": text_entry.model_revision,
            "multimodal_embedding_revision": multimodal_entry.model_revision,
            "seed": seed,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "split": split,
            "target_stage": target_stage.value,
            **(
                {
                    "questioner_freeze_or_audit_hash": (
                        program_identity.questioner_freeze_or_audit_hash
                    ),
                    "questioner_identity_kind": program_identity.questioner_identity_kind,
                    "planner_config_hash": planner_config_hash(program_identity.planner),
                    "planner_model_revision": planner_entry.model_revision,
                    "memory_text_model_revision": memory_text_entry.model_revision,
                    "memory_multimodal_model_revision": (memory_multimodal_entry.model_revision),
                    **(
                        {
                            "doubter_config_hash": doubter_config_hash(program_identity.doubter),
                            "doubter_model_revision": doubter_entry.model_revision,
                        }
                        if program_identity.doubter is not None and doubter_entry is not None
                        else {}
                    ),
                }
                if program_identity is not None
                and planner_entry is not None
                and memory_text_entry is not None
                and memory_multimodal_entry is not None
                else {}
            ),
        }
        resolved_run_id = run_id or (
            ("pmmc-program-" if program_identity is not None else "pmmc-questioner-")
            + canonical_content_hash(run_recipe)[:24]
        )
        if phase in {
            QuestionerWorkflowPhase.INDEX_TEXT,
            QuestionerWorkflowPhase.INDEX_MULTIMODAL,
        }:
            async with ExperimentUnitOfWork(sessions) as uow:
                assert uow.runs is not None
                assert uow.compiler_runs is not None and uow.compiler_units is not None
                stored_manifest = await uow.runs.get_manifest(resolved_run_id)
                stored_compiler = await uow.compiler_runs.get(resolved_run_id)
                if stored_compiler is None:
                    raise KeyError(f"compiler run does not exist: {resolved_run_id}")
                if (
                    stored_compiler.snapshot_id != snapshot_id
                    or stored_compiler.questioner_config_hash != q_config_hash
                    or stored_compiler.metadata.get("questioner_request_contract_hash")
                    != q_request_contract_hash
                    or stored_compiler.metadata.get("conversation_ids") != selected_conversations
                    or stored_manifest.dataset_snapshot_ids != [snapshot_id]
                    or stored_compiler.target_stage is not target_stage
                ):
                    raise ValueError("index phase conflicts with the frozen Questioner run")
                frozen_units = await uow.compiler_units.list_for_run(resolved_run_id)
            if len(frozen_units) != stored_compiler.unit_count:
                raise RuntimeError("frozen Questioner unit denominator is incomplete")
            return await _run_index_phase(
                settings=settings,
                registry=registry,
                config=config,
                sessions=sessions,
                run_id=resolved_run_id,
                snapshot_id=snapshot_id,
                conversation_count=len(selected_conversations),
                unit_values=frozen_units,
                phase=phase,
                complete_run=target_stage is CompilationTargetStage.QUESTIONS,
                on_index_progress=on_index_progress,
            )

        built_units = None
        prepared_units = None
        prepared_links = None
        preparation_id = None
        if unit_preparation is not None:
            preparation_manifest, descriptors = CompilationUnitPreparationService.verify(
                unit_preparation
            )
            expected_request_contract_hash = questioner_request_contract_hash(
                prompt_version=config.prompt_version,
                renderer_version=config.renderer_version,
                opportunity_builder_version=(
                    config.opportunity_builder_version if config.is_v3 else None
                ),
            )
            preparation_checks = {
                "conversation_denominator": (
                    preparation_manifest.conversation_ids == selected_conversations
                ),
                "questioner_config": (preparation_manifest.questioner_config_hash == q_config_hash),
                "request_contract": (
                    preparation_manifest.request_contract_hash == expected_request_contract_hash
                ),
                "run": preparation_manifest.run_id == resolved_run_id,
                "snapshot": preparation_manifest.snapshot_id == snapshot_id,
                "unit_builder": (
                    preparation_manifest.unit_builder_version
                    == config.unit_builder_for(snapshot.dataset)
                ),
            }
            failed_preparation_checks = [
                key for key, passed in preparation_checks.items() if not passed
            ]
            if failed_preparation_checks:
                raise CompilerWorkflowContractError(
                    stage="unit_preparation",
                    code="questioner_identity_mismatch",
                    failed_checks=failed_preparation_checks,
                )
            run_scoped_identity = questioner_uses_run_scoped_units(config)
            prepared_units = [
                item.to_unit(
                    run_id=resolved_run_id,
                    ordinal=ordinal,
                    request_contract_hash=preparation_manifest.request_contract_hash,
                    run_scoped_identity=run_scoped_identity,
                )
                for ordinal, item in enumerate(descriptors)
            ]
            prepared_links = {
                unit.compiler_unit_id: descriptor.rebound_links(
                    run_id=resolved_run_id,
                    run_scoped_identity=run_scoped_identity,
                )
                for unit, descriptor in zip(prepared_units, descriptors, strict=True)
            }
            print(
                "COMPILER_UNIT_NAMESPACE_PREFLIGHT "
                + json.dumps(
                    {
                        "dataset": snapshot.dataset.value,
                        "descriptor_count": len(descriptors),
                        "identity_protocol": (
                            COMPILER_UNIT_RUN_SCOPED_IDENTITY_V1
                            if run_scoped_identity
                            else "content-scoped-legacy"
                        ),
                        "namespace_rebound_count": sum(
                            unit.compiler_unit_id != descriptor.compiler_unit_id
                            for unit, descriptor in zip(prepared_units, descriptors, strict=True)
                        ),
                        "state": "ready",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            preparation_id = preparation_manifest.preparation_id
            counter_revision = preparation_manifest.processor_revision
            counter_estimated = bool(
                preparation_manifest.metadata.get("token_count_estimated", False)
            )
        else:
            memories_for_run = await PostgresMemoryRepository(sessions).list_snapshot(
                snapshot_id, conversation_ids=selected_set
            )
            counter = create_token_counter(
                settings,
                registry,
                config.compiler_model_alias,
                allow_estimated=settings.mode is not AppMode.PAPER,
            )
            fts_relation_provider = None
            if config.is_v3:
                if settings.database_url is None:
                    raise ValueError("Questioner v3 runs require PostgreSQL simple FTS")
                from pmmc.infrastructure.database.questioner_fts import (
                    PostgresSimpleFtsRelationProvider,
                )

                fts_relation_provider = PostgresSimpleFtsRelationProvider(settings)
            try:
                built_units = create_compilation_unit_builder(
                    config=config,
                    token_counter=counter,
                    dataset=snapshot.dataset,
                    fts_relation_provider=fts_relation_provider,
                ).build(
                    run_id=resolved_run_id,
                    snapshot_id=snapshot_id,
                    sessions=sessions_for_run,
                    memories=memories_for_run,
                )
            finally:
                if fts_relation_provider is not None:
                    fts_relation_provider.close()
            counter_revision = counter.revision
            counter_estimated = bool(counter.estimated)
        unit_values = (
            [item.unit for item in built_units]
            if built_units is not None
            else list(prepared_units or [])
        )
        if not unit_values:
            raise ValueError("Questioner unit builder produced an empty denominator")

        foundation = load_compiler_config(settings.resolved_compiler_config_path)
        if config.is_v36 and foundation.operator_set_version != "pmmc-operators-v7":
            raise ValueError("Questioner v3.6 requires the Step8 compiler foundation v7")
        if program_identity is not None:
            if foundation.operator_set_version != program_identity.planner.operator_set_version:
                raise ValueError("Planner and compiler operator-set identities differ")
            if foundation.dry_run_policy != program_identity.planner.dry_run_policy:
                raise ValueError("Planner and compiler dry-run policies differ")
        operator_registry = build_operator_registry(object(), foundation)  # type: ignore[arg-type]
        selection_hash = canonical_content_hash(
            {
                "snapshot_id": snapshot_id,
                "conversation_ids": selected_conversations,
                "unit_ids": [item.compiler_unit_id for item in unit_values],
                "seed": seed,
                "shard_count": shard_count,
                "shard_index": shard_index,
                "split": split,
                "diagnostic_force_revision": (
                    program_identity.diagnostic_force_revision
                    if program_identity is not None
                    else False
                ),
            }
        )
        manifest = RunManifest(
            **questioner_workflow_manifest_versions(config),
            run_id=resolved_run_id,
            experiment_name=(
                "pmmc_compiler_programs" if program_identity is not None else "pmmc_questioner"
            ),
            dataset=snapshot.dataset,
            dataset_version=snapshot.dataset_version,
            method="pmmc_compiler",
            split=split,
            paper_mode=settings.mode is AppMode.PAPER,
            compiler_model_alias=config.compiler_model_alias,
            embedding_model_aliases=[
                config.text_embedding_model_alias,
                config.multimodal_embedding_model_alias,
            ],
            question_budget_max=config.question_budget_max,
            selection_hash=selection_hash,
            seed=seed,
            retrieval_topk_max=config.per_channel_top_k,
            prompt_hashes={
                config.prompt_version: questioner_prompt_sha256(config.prompt_version),
                **(
                    {
                        config.verifier_prompt_version: questioner_prompt_sha256(
                            config.verifier_prompt_version
                        )
                    }
                    if config.is_v3
                    else {}
                ),
                **(
                    {
                        program_identity.doubter.prompt_version: (
                            doubter_prompt_sha256(program_identity.doubter)
                        )
                    }
                    if program_identity is not None and program_identity.doubter is not None
                    else {}
                ),
            },
            retrieval_config={
                "matcher_version": config.matcher_version,
                "rrf_constant": config.rrf_constant,
                "per_channel_top_k": config.per_channel_top_k,
            },
            dataset_snapshot_ids=[snapshot_id],
            operator_set_version=foundation.operator_set_version,
            operator_registry_hash=operator_registry.resolved_hash,
            compiler_config_hash=compiler_config_hash(foundation),
            metadata={
                "target_stage": target_stage.value,
                **(
                    {"compiler_subprotocol_hash": program_identity.questioner_freeze_or_audit_hash}
                    if program_identity is not None
                    and program_identity.questioner_identity_kind == "step8_method_protocol"
                    else {}
                ),
                "questioner_config_hash": q_config_hash,
                "questioner_request_contract_hash": q_request_contract_hash,
                "questioner_config": config.model_dump(mode="json"),
                **(
                    {
                        "questioner_response_schema_hash": (
                            questioner_response_schema_sha256(config.prompt_version)
                        )
                    }
                    if config.is_v2 or config.is_v3
                    else {}
                ),
                **(
                    {
                        "questioner_opportunity_builder_version": (
                            config.opportunity_builder_version
                        ),
                        "questioner_verifier_prompt_version": (config.verifier_prompt_version),
                        "questioner_verifier_prompt_hash": questioner_prompt_sha256(
                            config.verifier_prompt_version
                        ),
                        "questioner_verifier_schema_version": (config.verifier_schema_version),
                        "questioner_verifier_schema_hash": (
                            questioner_response_schema_sha256(config.verifier_prompt_version)
                        ),
                    }
                    if config.is_v3
                    else {}
                ),
                "conversation_ids": selected_conversations,
                "unit_builder_tokenizer_revision": counter_revision,
                "token_count_estimated": counter_estimated,
                **({"unit_preparation_id": preparation_id} if preparation_id is not None else {}),
                "shard_count": shard_count,
                "shard_index": shard_index,
                **({"partition_role": split} if split is not None else {}),
                **(
                    {
                        "questioner_freeze_or_audit_hash": (
                            program_identity.questioner_freeze_or_audit_hash
                        ),
                        "questioner_identity_kind": (program_identity.questioner_identity_kind),
                        "diagnostic_force_revision": (program_identity.diagnostic_force_revision),
                        "planner_config_hash": planner_config_hash(program_identity.planner),
                        "planner_response_schema_hash": (
                            planner_response_schema_sha256(program_identity.planner)
                        ),
                        **(
                            {
                                "doubter_response_schema_hash": (
                                    doubter_response_schema_sha256(program_identity.doubter)
                                ),
                                "active_round": program_identity.doubter.active_round,
                                "max_round": program_identity.doubter.max_round,
                            }
                            if program_identity.doubter is not None
                            else {}
                        ),
                    }
                    if program_identity is not None
                    else {}
                ),
            },
        )
        aliases = [
            config.compiler_model_alias,
            config.text_embedding_model_alias,
            config.multimodal_embedding_model_alias,
        ]
        if program_identity is not None:
            aliases.extend(
                [
                    program_identity.planner.planner_model_alias,
                    program_identity.memory_text_model_alias,
                    program_identity.memory_multimodal_model_alias,
                ]
            )
            if program_identity.doubter is not None:
                aliases.append(program_identity.doubter.doubter_model_alias)
            aliases = list(dict.fromkeys(aliases))
        manifest = bind_registry_to_manifest(
            manifest, registry, aliases=aliases,
            **(
                {"registry_identity_scope": "selected_aliases"}
                if compiler_schema_version == "0.9.0"
                and program_identity is not None
                and program_identity.questioner_identity_kind == "step8_method_protocol"
                else {}
            ),
        )
        manifest = manifest.model_copy(update={"config_hash": run_config_hash(manifest)})
        compiler_run = CompilerRunSpec(
            run_id=resolved_run_id,
            snapshot_id=snapshot_id,
            compiler_schema_version=compiler_schema_version,
            target_stage=target_stage,
            operator_set_version=foundation.operator_set_version,
            operator_registry_hash=operator_registry.resolved_hash,
            compiler_config_hash=compiler_config_hash(foundation),
            questioner_config_hash=q_config_hash,
            questioner_model_alias=config.compiler_model_alias,
            questioner_model_revision=compiler_entry.model_revision,
            questioner_prompt_version=config.prompt_version,
            questioner_prompt_hash=questioner_prompt_sha256(config.prompt_version),
            questioner_renderer_version=config.renderer_version,
            questioner_parser_version=config.parser_version,
            unit_builder_version=config.unit_builder_version,
            candidate_gate_version=config.gate_version,
            dedupe_version=config.dedupe_version,
            qbank_matcher_version=config.matcher_version,
            questioner_freeze_manifest_hash=(
                program_identity.questioner_freeze_or_audit_hash
                if program_identity is not None
                else None
            ),
            planner_config_hash=(
                planner_config_hash(program_identity.planner)
                if program_identity is not None
                else None
            ),
            planner_model_alias=(
                program_identity.planner.planner_model_alias
                if program_identity is not None
                else None
            ),
            planner_model_revision=(
                planner_entry.model_revision if planner_entry is not None else None
            ),
            planner_prompt_version=(
                program_identity.planner.prompt_version if program_identity is not None else None
            ),
            planner_prompt_hash=(
                planner_prompt_sha256(program_identity.planner)
                if program_identity is not None
                else None
            ),
            planner_renderer_version=(
                program_identity.planner.renderer_version if program_identity is not None else None
            ),
            planner_parser_version=(
                program_identity.planner.parser_version if program_identity is not None else None
            ),
            planner_schema_version=(
                program_identity.planner.schema_version if program_identity is not None else None
            ),
            program_policy_version=(
                program_identity.planner.policy_version if program_identity is not None else None
            ),
            dry_run_policy_version=(
                program_identity.planner.dry_run_policy if program_identity is not None else None
            ),
            memory_index_representation_version=(
                "memory-operator-index-v1" if program_identity is not None else None
            ),
            memory_text_model_alias=(
                program_identity.memory_text_model_alias if program_identity is not None else None
            ),
            memory_text_model_revision=(
                memory_text_entry.model_revision if memory_text_entry is not None else None
            ),
            memory_multimodal_model_alias=(
                program_identity.memory_multimodal_model_alias
                if program_identity is not None
                else None
            ),
            memory_multimodal_model_revision=(
                memory_multimodal_entry.model_revision
                if memory_multimodal_entry is not None
                else None
            ),
            doubter_config_hash=(
                doubter_config_hash(program_identity.doubter)
                if program_identity is not None and program_identity.doubter is not None
                else None
            ),
            doubter_model_alias=(
                program_identity.doubter.doubter_model_alias
                if program_identity is not None and program_identity.doubter is not None
                else None
            ),
            doubter_model_revision=(
                doubter_entry.model_revision if doubter_entry is not None else None
            ),
            doubter_prompt_version=(
                program_identity.doubter.prompt_version
                if program_identity is not None and program_identity.doubter is not None
                else None
            ),
            doubter_prompt_hash=(
                doubter_prompt_sha256(program_identity.doubter)
                if program_identity is not None and program_identity.doubter is not None
                else None
            ),
            doubter_renderer_version=(
                program_identity.doubter.renderer_version
                if program_identity is not None and program_identity.doubter is not None
                else None
            ),
            doubter_parser_version=(
                program_identity.doubter.parser_version
                if program_identity is not None and program_identity.doubter is not None
                else None
            ),
            doubter_schema_version=(
                program_identity.doubter.schema_version
                if program_identity is not None and program_identity.doubter is not None
                else None
            ),
            r0_input_policy_version=(
                program_identity.doubter.r0_input_policy_version
                if program_identity is not None and program_identity.doubter is not None
                else None
            ),
            routing_projection_version=(
                program_identity.doubter.routing_projection_version
                if program_identity is not None and program_identity.doubter is not None
                else None
            ),
            active_round=(
                program_identity.doubter.active_round
                if program_identity is not None and program_identity.doubter is not None
                else None
            ),
            max_round=(
                program_identity.doubter.max_round
                if program_identity is not None and program_identity.doubter is not None
                else None
            ),
            unit_count=len(unit_values),
            metadata={
                **(
                    {"compiler_subprotocol_hash": program_identity.questioner_freeze_or_audit_hash}
                    if program_identity is not None
                    and program_identity.questioner_identity_kind == "step8_method_protocol"
                    else {}
                ),
                "conversation_ids": selected_conversations,
                "selection_hash": selection_hash,
                "questioner_request_contract_hash": q_request_contract_hash,
                **({"unit_preparation_id": preparation_id} if preparation_id is not None else {}),
                **({"partition_role": split} if split is not None else {}),
                **(
                    {
                        "questioner_identity_kind": (program_identity.questioner_identity_kind),
                        "diagnostic_force_revision": (program_identity.diagnostic_force_revision),
                        "program_pilot": (
                            program_identity.questioner_identity_kind == "development_audit"
                        ),
                        "planner_response_schema_hash": (
                            planner_response_schema_sha256(program_identity.planner)
                        ),
                        **(
                            {
                                "doubter_response_schema_hash": (
                                    doubter_response_schema_sha256(program_identity.doubter)
                                ),
                                "active_round": program_identity.doubter.active_round,
                                "max_round": program_identity.doubter.max_round,
                            }
                            if program_identity.doubter is not None
                            else {}
                        ),
                    }
                    if program_identity is not None
                    else {}
                ),
                **(
                    {
                        "questioner_response_schema_hash": (
                            questioner_response_schema_sha256(config.prompt_version)
                        )
                    }
                    if config.is_v2 or config.is_v3
                    else {}
                ),
                **(
                    {
                        "questioner_opportunity_builder_version": (
                            config.opportunity_builder_version
                        ),
                        "questioner_verifier_prompt_version": (config.verifier_prompt_version),
                        "questioner_verifier_prompt_hash": questioner_prompt_sha256(
                            config.verifier_prompt_version
                        ),
                        "questioner_verifier_schema_version": (config.verifier_schema_version),
                        "questioner_verifier_schema_hash": (
                            questioner_response_schema_sha256(config.verifier_prompt_version)
                        ),
                    }
                    if config.is_v3
                    else {}
                ),
            },
        )
        async with ExperimentUnitOfWork(sessions) as uow:
            assert uow.runs is not None and uow.compiler_runs is not None
            existing_state = None
            try:
                existing_state = await uow.runs.get_state(resolved_run_id)
            except KeyError:
                pass
            if existing_state is None:
                await uow.runs.register(manifest, [])
                await uow.compiler_runs.add(compiler_run)
            else:
                stored_manifest = await uow.runs.get_manifest(resolved_run_id)
                stored_compiler = await uow.compiler_runs.get(resolved_run_id)
                if stored_compiler is None:
                    raise RuntimeError("Questioner run is missing its compiler denominator")
                manifest = manifest.model_copy(
                    update={"created_at": stored_manifest.created_at, "config_hash": None}
                )
                manifest = manifest.model_copy(update={"config_hash": run_config_hash(manifest)})
                compiler_run = compiler_run.model_copy(
                    update={"created_at": stored_compiler.created_at}
                )
                if stored_manifest != manifest or stored_compiler != compiler_run:
                    raise ValueError("resume run conflicts with the frozen Questioner identity")
                if existing_state.status is RunStatus.COMPLETED:
                    assert uow.qbank_indexes is not None
                    ready = await uow.qbank_indexes.get_for_run(resolved_run_id)
                    if ready is None or ready.status.value != "ready":
                        raise RuntimeError("completed Questioner run has no ready QBank index")
                    return QuestionerWorkflowResult(
                        resolved_run_id,
                        snapshot_id,
                        len(selected_conversations),
                        len(unit_values),
                        ready.eligible_question_count,
                        ready.paraphrase_count,
                        ready.qbank_index_id,
                        True,
                    )

        normalized_unit_values = []
        async with ExperimentUnitOfWork(sessions) as uow:
            assert uow.compiler_units is not None
            for unit in unit_values:
                stored = await uow.compiler_units.get(unit.compiler_unit_id)
                if stored is None:
                    normalized_unit_values.append(unit)
                    continue
                candidate = unit.model_copy(update={"created_at": stored.created_at})
                if candidate != stored:
                    raise CompilerWorkflowContractError(
                        stage="compiler_unit_registration",
                        code="persisted_unit_instance_collision",
                        failed_checks=["run_scoped_unit_identity"],
                    )
                normalized_unit_values.append(stored)
        unit_values = normalized_unit_values
        if built_units is not None:
            by_id = {item.unit.compiler_unit_id: item for item in built_units}
            built_units = [replace(by_id[unit.compiler_unit_id], unit=unit) for unit in unit_values]

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
        if phase in {QuestionerWorkflowPhase.ALL, QuestionerWorkflowPhase.QUESTIONER}:
            generation_gateway = create_generation_gateway(
                settings,
                registry,
                config.compiler_model_alias,
                image_url_resolver=resolver,
            )
            questioner_result: QuestionerRunResult = await QuestionerRunner(
                sessions=sessions,
                execution=execution,
                gateway=generation_gateway,
                error_artifacts=artifacts,
                config=config,
                lease_seconds=settings.experiment_job_lease_seconds,
                concurrency=settings.model_max_concurrency,
            ).execute(
                run_id=resolved_run_id,
                snapshot_id=snapshot_id,
                units=built_units,
                prepared_units=unit_values if built_units is None else None,
                prepared_links=prepared_links,
                input_provider=(
                    PostgresQuestionerInputProvider(
                        sessions=sessions,
                        config=config,
                        token_counter=None,
                    )
                    if built_units is None
                    else None
                ),
                on_item=on_questioner_item,
                on_progress=on_questioner_progress,
            )
            if questioner_result.failed:
                raise RuntimeError("Questioner producer did not complete")
        if phase is QuestionerWorkflowPhase.QUESTIONER:
            async with ExperimentUnitOfWork(sessions) as uow:
                assert uow.questioner is not None
                eligible = await uow.questioner.list_eligible_questions(resolved_run_id)
            return QuestionerWorkflowResult(
                resolved_run_id,
                snapshot_id,
                len(selected_conversations),
                len(unit_values),
                len(eligible),
                sum(len(item.paraphrases) for item in eligible),
                None,
                False,
            )

        async with ExperimentUnitOfWork(sessions) as uow:
            assert uow.compiler_units is not None
            missing_questioner_units = [
                unit.compiler_unit_id
                for unit in unit_values
                if await uow.compiler_units.get_stage_result(
                    unit.compiler_unit_id,
                    CompilationStage.QUESTIONER,
                )
                is None
            ]
        if missing_questioner_units:
            raise RuntimeError("QBank indexing requires completed Questioner facts for every unit")

        return await _run_index_phase(
            settings=settings,
            registry=registry,
            config=config,
            sessions=sessions,
            run_id=resolved_run_id,
            snapshot_id=snapshot_id,
            conversation_count=len(selected_conversations),
            unit_values=unit_values,
            phase=phase,
            complete_run=target_stage is CompilationTargetStage.QUESTIONS,
            on_index_progress=on_index_progress,
        )
    finally:
        await engine.dispose()
