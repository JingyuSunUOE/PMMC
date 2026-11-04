"""CLI-facing orchestration for full-context baseline runs and dry assembly."""

from __future__ import annotations

from pathlib import Path

from pmmc.baselines.answering import (
    BaselineAnswerService,
    FullContextAnswerInputProvider,
)
from pmmc.baselines.config import (
    FullContextBaselineConfig,
    load_full_context_config,
    load_full_context_prompt,
    prompt_sha256,
)
from pmmc.baselines.context import ContextBudgeter
from pmmc.baselines.repositories import JsonlContextAssemblyRepository
from pmmc.baselines.runner import (
    FullContextBaselineRunner,
    TransactionalModelCallWriter,
)
from pmmc.baselines.typed_answer import (
    baseline_answer_contract,
    baseline_model_aliases,
    typed_baseline_metadata,
)
from pmmc.benchmarks.async_repositories import (
    JsonlMemoryRepository,
    JsonlSessionRepository,
)
from pmmc.benchmarks.repositories import RuntimeQueryStore
from pmmc.benchmarks.snapshot import find_snapshot, load_manifest
from pmmc.experiments import PipelineRunContext, make_run_id
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.repositories import (
    PostgresMemoryRepository,
    PostgresRuntimeQueryRepository,
    PostgresSessionRepository,
    PostgresSnapshotRepository,
)
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.infrastructure.experiment_runs import (
    load_local_run_registration,
    register_local_run,
    verify_experiment_run,
)
from pmmc.infrastructure.object_store import create_runtime_object_store
from pmmc.infrastructure.settings import AppMode, Settings
from pmmc.models.factory import (
    create_generation_gateway,
    create_image_url_resolver,
    create_model_execution_service,
    create_token_counter,
)
from pmmc.models.registry import (
    bind_registry_to_manifest,
    load_model_registry,
    verify_manifest_registry_compatibility,
)
from pmmc.runtime.grounding import configured_grounding_service, load_step8_prompt
from pmmc.schemas import (
    BaselineMethod,
    BenchmarkQuery,
    BenchmarkSelection,
    DataScope,
    DatasetSnapshotManifest,
    ModelCapability,
    RunManifest,
    RunStatus,
)
from pmmc.schemas.answer_output import TYPED_ANSWER_CONTRACT_VERSION


def _manifest(
    *,
    run_id: str,
    method: BaselineMethod,
    snapshot: DatasetSnapshotManifest,
    config: FullContextBaselineConfig,
    prompt_hash: str,
    settings: Settings,
    assemble_only: bool,
    run_metadata: dict[str, object] | None = None,
) -> RunManifest:
    extra_metadata = typed_baseline_metadata(config, run_metadata)
    typed = baseline_answer_contract(config, extra_metadata) == TYPED_ANSWER_CONTRACT_VERSION
    prompt_version = "pmmc-answer-contract-v3" if typed else config.prompt_version
    reserved_metadata = {
        "assemble_only",
        "full_context_config",
        "protocol_version",
        "renderer_version",
    }
    if reserved_metadata.intersection(extra_metadata):
        raise ValueError("run metadata cannot override full-context protocol fields")
    manifest = RunManifest(
        run_id=run_id,
        experiment_name="five-foundational-baselines",
        dataset=snapshot.dataset,
        dataset_version=snapshot.dataset_version,
        method=method.value,
        split=(
            str(extra_metadata["partition_role"])
            if extra_metadata.get("partition_role") is not None
            else None
        ),
        paper_mode=settings.mode is AppMode.PAPER,
        answer_model_alias=config.answer_model_alias,
        prompt_hashes={prompt_version: prompt_hash, **({
            "answer-grounding-v1": extra_metadata["grounding_prompt_hash"],
        } if typed else {})},
        decoding_configs={
            "answer": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "seed": config.seed,
                "max_output_tokens": 512 if typed else config.max_output_tokens,
                **({"thinking_mode": "disabled", "format_retries": 1} if typed else {}),
            }
        },
        evaluation_policy_version=snapshot.evaluation_policy_version,
        evaluation_exclusion_rules={
            snapshot.dataset.value: list(snapshot.excluded_source_categories)
        },
        dataset_snapshot_ids=[snapshot.snapshot_id],
        prompt_registry_version=prompt_version,
        metadata={
            "assemble_only": assemble_only,
            "full_context_config": config.model_dump(mode="json"),
            "protocol_version": config.protocol_version,
            "renderer_version": config.renderer_version,
            **extra_metadata,
        },
    )
    return manifest


def _full_context_prompt(config, project_root, metadata=None) -> str:
    if baseline_answer_contract(config, metadata) == TYPED_ANSWER_CONTRACT_VERSION:
        return load_step8_prompt("pmmc_answer_contract_v3.txt")
    return load_full_context_prompt(config, project_root)


def _validate_method_capability(
    method: BaselineMethod,
    selected: list[BenchmarkQuery],
    registry,  # type: ignore[no-untyped-def]
    alias: str,
) -> None:
    entry = registry.get(alias)
    needs_images = method is BaselineMethod.FULL_CONTEXT_RAW_IMAGES or any(
        record.query.query_image_refs for record in selected
    )
    entry.require(
        ModelCapability.CHAT_MULTIMODAL if needs_images else ModelCapability.CHAT_TEXT
    )
    if entry.context_window is None or entry.context_window < 32768:
        raise ValueError("full-context answer model must provide a 32768-token context")


async def _assemble_only(
    *,
    settings: Settings,
    method: BaselineMethod,
    snapshot_id: str,
    selection: BenchmarkSelection,
    answer_model_alias: str | None,
    run_metadata: dict[str, object] | None = None,
) -> Path:
    snapshot_path = find_snapshot(settings.benchmark_artifacts_path, snapshot_id)
    snapshot = load_manifest(snapshot_path)
    if selection.datasets != [snapshot.dataset]:
        raise ValueError("selection dataset must match the canonical snapshot")
    config = load_full_context_config(
        settings.resolved_full_context_config_path,
        answer_model_alias=answer_model_alias,
    )
    prompt = _full_context_prompt(config, settings.project_root, run_metadata)
    registry = load_model_registry(settings.resolved_model_registry_path)
    selected = RuntimeQueryStore(snapshot_path).select(selection)
    if not selected:
        raise ValueError("benchmark selection produced no queries")
    _validate_method_capability(method, selected, registry, config.answer_model_alias)
    counter = create_token_counter(
        settings,
        registry,
        config.answer_model_alias,
        allow_estimated=True,
    )
    provider = FullContextAnswerInputProvider(
        memories=JsonlMemoryRepository(settings.benchmark_artifacts_path),
        sessions=JsonlSessionRepository(settings.benchmark_artifacts_path),
        budgeter=ContextBudgeter(
            counter,
            total_context_tokens=config.total_context_tokens,
            reserved_output_tokens=config.reserved_output_tokens,
            paper_mode=settings.mode is AppMode.PAPER,
        ),
        config=config,
        instruction=prompt,
        timeout_seconds=settings.model_request_timeout_seconds,
        paper_mode=settings.mode is AppMode.PAPER,
        runtime_bucket=settings.object_store_bucket,
        answer_contract_version=baseline_answer_contract(config, run_metadata),
    )
    run_id = make_run_id(f"assemble-{method.value}")
    manifest = bind_registry_to_manifest(
        _manifest(
            run_id=run_id,
            method=method,
            snapshot=snapshot,
            config=config,
            prompt_hash=prompt_sha256(prompt),
            settings=settings,
            assemble_only=True,
            run_metadata=run_metadata,
        ),
        registry,
        aliases=baseline_model_aliases(config, run_metadata, [config.answer_model_alias]),
    )
    run = PipelineRunContext(
        runs_root=settings.runs_path,
        manifest=manifest,
        datasets=[snapshot.dataset],
    )
    with run:
        with run.stage("preflight", dataset=snapshot.dataset, total=1) as stage:
            stage.advance()
        with run.stage(
            "selection-register", dataset=snapshot.dataset, total=len(selected)
        ) as stage:
            selected = run.record_selection(selection, selected)
            stage.advance(len(selected))
        traces = JsonlContextAssemblyRepository(run.run_dir / "context_assemblies")
        with run.stage(
            "answer-jobs", dataset=snapshot.dataset, total=len(selected)
        ) as stage:
            for record in selected:
                prepared = await provider.prepare(
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    query=record.query,
                    method=method,
                )
                traces.add(prepared.context_assembly)
                stage.advance(query_id=record.query.query_id, source=record.source)
        run.succeeded_count = len(selected)
        with run.stage("finalize-verify", dataset=snapshot.dataset, total=1) as stage:
            stage.advance()
    return run.run_dir


async def run_full_context_workflow(
    *,
    settings: Settings,
    method: BaselineMethod | None,
    snapshot_id: str | None,
    selection: BenchmarkSelection | None,
    answer_model_alias: str | None = None,
    assemble_only: bool = False,
    resume_run_id: str | None = None,
    run_id: str | None = None,
    run_metadata: dict[str, object] | None = None,
    phase: str = "all",
) -> Path:
    """Execute or assemble a full-context run while preserving gold isolation."""
    if phase not in {"all", "answer", "generate", "ground"}:
        raise ValueError("unknown full-context phase")
    if phase == "ground" and resume_run_id is None:
        raise ValueError("ground phase must resume generated candidates")

    if resume_run_id is not None and assemble_only:
        raise ValueError("assemble-only runs are deterministic and do not use job resume")
    if resume_run_id is not None and run_id is not None:
        raise ValueError("resume_run_id and run_id are mutually exclusive")
    if resume_run_id is not None and run_metadata:
        raise ValueError("resume cannot change immutable run metadata")
    if resume_run_id is None:
        if method is None or snapshot_id is None or selection is None:
            raise ValueError("new runs require method, snapshot, and selection")
        if method not in {
            BaselineMethod.FULL_CONTEXT_CAPTIONS,
            BaselineMethod.FULL_CONTEXT_RAW_IMAGES,
        }:
            raise ValueError("Step 5B supports only the two full-context methods")
        if settings.mode is AppMode.PAPER and selection.include_excluded:
            raise ValueError("paper mode cannot include excluded benchmark categories")
        if assemble_only:
            return await _assemble_only(
                settings=settings,
                method=method,
                snapshot_id=snapshot_id,
                selection=selection,
                answer_model_alias=answer_model_alias,
                run_metadata=run_metadata,
            )

    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        query_repository = PostgresRuntimeQueryRepository(sessions)
        snapshot_repository = PostgresSnapshotRepository(sessions)
        resume = resume_run_id is not None
        if resume:
            run_dir = settings.runs_path / str(resume_run_id)
            local = load_local_run_registration(run_dir)
            manifest = local.manifest
            method = BaselineMethod(manifest.method)
            if method not in {
                BaselineMethod.FULL_CONTEXT_CAPTIONS,
                BaselineMethod.FULL_CONTEXT_RAW_IMAGES,
            }:
                raise ValueError("resumed run is not a Step 5B full-context run")
            if len(manifest.dataset_snapshot_ids) != 1:
                raise ValueError("full-context runs require exactly one snapshot")
            snapshot_id = manifest.dataset_snapshot_ids[0]
            selection = local.selection
            selected = local.queries
            config = FullContextBaselineConfig.model_validate(
                manifest.metadata.get("full_context_config")
            )
            if answer_model_alias and answer_model_alias != config.answer_model_alias:
                raise ValueError("resume cannot change the answer model alias")
            database_selected = await query_repository.select(snapshot_id, selection)
            if database_selected != selected:
                raise ValueError("resumed local selection differs from Postgres source of truth")
            snapshot = await snapshot_repository.get_manifest(snapshot_id)
        else:
            assert method is not None and snapshot_id is not None and selection is not None
            snapshot = await snapshot_repository.get_manifest(snapshot_id)
            if selection.datasets != [snapshot.dataset]:
                raise ValueError("selection dataset must match the canonical snapshot")
            config = load_full_context_config(
                settings.resolved_full_context_config_path,
                answer_model_alias=answer_model_alias,
            )
            selected = await query_repository.select(snapshot_id, selection)
            if not selected:
                raise ValueError("benchmark selection produced no queries")
            actual_run_id = run_id or make_run_id(method.value)
            prompt = _full_context_prompt(config, settings.project_root, run_metadata)
            registry = load_model_registry(settings.resolved_model_registry_path)
            manifest = bind_registry_to_manifest(
                _manifest(
                    run_id=actual_run_id,
                    method=method,
                    snapshot=snapshot,
                    config=config,
                    prompt_hash=prompt_sha256(prompt),
                    settings=settings,
                    assemble_only=False,
                    run_metadata=run_metadata,
                ),
                registry,
                aliases=baseline_model_aliases(config, run_metadata, [config.answer_model_alias]),
            )
            run_dir = settings.runs_path / actual_run_id

        typed_baseline_metadata(config, manifest.metadata)
        contract = baseline_answer_contract(config, manifest.metadata)
        if phase in {"generate", "ground"} and contract != TYPED_ANSWER_CONTRACT_VERSION:
            raise ValueError("split baseline phases require typed v3")
        prompt = _full_context_prompt(config, settings.project_root, manifest.metadata)
        current_prompt_hash = prompt_sha256(prompt)
        prompt_version = ("pmmc-answer-contract-v3" if contract == TYPED_ANSWER_CONTRACT_VERSION
                          else config.prompt_version)
        if manifest.prompt_hashes.get(prompt_version) != current_prompt_hash:
            raise ValueError("current full-context prompt differs from the immutable run manifest")
        registry = load_model_registry(settings.resolved_model_registry_path)
        verify_manifest_registry_compatibility(
            manifest, registry,
            baseline_model_aliases(config, manifest.metadata, [config.answer_model_alias]),
        )
        _validate_method_capability(method, selected, registry, config.answer_model_alias)
        counter = create_token_counter(settings, registry, config.answer_model_alias)

        if resume:
            async with ExperimentUnitOfWork(sessions) as uow:
                assert uow.runs is not None
                existing_state = await uow.runs.get_state(manifest.run_id)
            if existing_state.status is RunStatus.COMPLETED:
                async with sessions() as session:
                    report = await verify_experiment_run(session, manifest.run_id)
                if not report["valid"]:
                    raise RuntimeError("completed full-context run failed integrity verification")
                return run_dir
            if existing_state.status is not RunStatus.RUNNING:
                raise RuntimeError(
                    f"run cannot be resumed from state: {existing_state.status.value}"
                )

        run = PipelineRunContext(
            runs_root=settings.runs_path,
            manifest=manifest,
            datasets=[snapshot.dataset],
            resume=resume,
        )
        with run:
            with run.stage("preflight", dataset=snapshot.dataset, total=1) as stage:
                stage.advance()
            with run.stage(
                "selection-register",
                dataset=snapshot.dataset,
                total=len(selected) + 1,
            ) as stage:
                if not resume:
                    selected = run.record_selection(selection, selected)
                    manifest = run.manifest
                stage.advance(len(selected))
                if not resume:
                    async with sessions() as session, session.begin():
                        await register_local_run(session, run.run_dir)
                stage.advance()

            store = create_runtime_object_store(settings)
            artifacts = ExperimentArtifactService(
                sessions,
                store,
                DataScope.RUNTIME,
                artifact_prefix=settings.experiment_artifact_prefix,
            )
            writer = TransactionalModelCallWriter(sessions)
            execution = create_model_execution_service(
                settings,
                registry,
                artifact_service=artifacts,
                model_calls=writer,
            )
            resolver = create_image_url_resolver(settings, store)
            grounding = None
            if contract == TYPED_ANSWER_CONTRACT_VERSION and phase != "generate":
                grounding = configured_grounding_service(
                    settings=settings, execution=execution, image_url_resolver=resolver,
                )
                if grounding is None:
                    raise ValueError("typed baseline requires a configured 9B grounding endpoint")
            gateway = None if phase == "ground" else create_generation_gateway(
                settings,
                registry,
                config.answer_model_alias,
                image_url_resolver=resolver,
            )
            provider = FullContextAnswerInputProvider(
                memories=PostgresMemoryRepository(sessions),
                sessions=PostgresSessionRepository(sessions),
                budgeter=ContextBudgeter(
                    counter,
                    total_context_tokens=config.total_context_tokens,
                    reserved_output_tokens=config.reserved_output_tokens,
                    paper_mode=settings.mode is AppMode.PAPER,
                ),
                config=config,
                instruction=prompt,
                timeout_seconds=settings.model_request_timeout_seconds,
                paper_mode=settings.mode is AppMode.PAPER,
                runtime_bucket=settings.object_store_bucket,
                answer_contract_version=contract,
            )
            runner = FullContextBaselineRunner(
                sessions=sessions,
                input_provider=provider,
                answer_service=BaselineAnswerService(execution, gateway, grounding=grounding),
                error_artifacts=artifacts,
                method=method,
                config=config,
                lease_seconds=settings.experiment_job_lease_seconds,
                phase=phase if phase in {"generate", "ground"} else "answer",
            )
            with run.stage(
                "answer-jobs", dataset=snapshot.dataset, total=len(selected)
            ) as stage:

                def on_item(query_id: str, success: bool, error_type: str | None) -> None:
                    record = next(item for item in selected if item.query.query_id == query_id)
                    if success:
                        stage.advance(query_id=query_id, source=record.source)
                    else:
                        stage.fail_item(
                            RuntimeError(error_type or "BaselineJobFailure"),
                            query_id=query_id,
                            source=record.source,
                        )

                result = await runner.execute(
                    run_id=manifest.run_id,
                    snapshot_id=snapshot_id,
                    selected=selected,
                    on_item=on_item,
                )
                run.succeeded_count = result.succeeded
                run.failed_count = result.failed
            if phase == "generate":
                # Only BASELINE-stage generation jobs succeeded. ANSWER jobs and
                # the scientific run remain incomplete until ground finalizes them.
                return run.run_dir
            with run.stage("finalize-verify", dataset=snapshot.dataset, total=1) as stage:
                async with sessions() as session:
                    report = await verify_experiment_run(session, manifest.run_id)
                if not report["valid"]:
                    raise RuntimeError("persisted full-context run failed integrity verification")
                if not result.already_completed:
                    async with ExperimentUnitOfWork(sessions) as uow:
                        assert uow.runs is not None
                        await uow.runs.transition(manifest.run_id, RunStatus.COMPLETED)
                stage.advance()
        return run.run_dir
    finally:
        await engine.dispose()
