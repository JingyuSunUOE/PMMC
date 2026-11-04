"""CLI-facing orchestration for three auditable RAG baseline runs."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.baselines.answering import BaselineAnswerService
from pmmc.baselines.rag_config import (
    RagBaselineConfig,
    load_rag_config,
    query_instruction_sha256,
    rag_prompt_sha256,
)
from pmmc.baselines.rag_context import RagAnswerInputProvider, RagContextBudgeter
from pmmc.baselines.rag_embedding import (
    RAG_METHODS,
    make_query_collection,
)
from pmmc.baselines.rag_index import build_rag_index
from pmmc.baselines.rag_runner import RagAnswerRunner, RagRetrievalRunner
from pmmc.baselines.typed_answer import (
    baseline_answer_contract,
    baseline_model_aliases,
    typed_baseline_metadata,
)
from pmmc.experiments import PipelineRunContext, PipelineStage, make_run_id
from pmmc.infrastructure.database.baseline_repositories import PostgresCorpusRepository
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.experiment_repositories import (
    ExperimentUnitOfWork,
    PostgresRetrievalRepository,
)
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
    create_embedding_gateway,
    create_generation_gateway,
    create_image_url_resolver,
    create_model_execution_service,
    create_token_counter,
)
from pmmc.models.registry import (
    ModelRegistry,
    bind_registry_to_manifest,
    load_model_registry,
    verify_manifest_registry_compatibility,
)
from pmmc.runtime.grounding import configured_grounding_service, load_step8_prompt
from pmmc.schemas import (
    BaselineCorpusManifest,
    BaselineMethod,
    BenchmarkQuery,
    BenchmarkSelection,
    DataScope,
    DatasetSnapshotManifest,
    EmbeddingCollection,
    EmbeddingCollectionStatus,
    MemoryWindow,
    ModelCapability,
    RetrievalBatchTrace,
    RetrievalQueryEmbeddingRef,
    RetrievalTrace,
    RetrievalTraceComponent,
    RunManifest,
    RunStatus,
)
from pmmc.schemas.answer_output import TYPED_ANSWER_CONTRACT_VERSION


class RagRunPhase(StrEnum):
    ALL = "all"
    RETRIEVE = "retrieve"
    ANSWER = "answer"
    GENERATE = "generate"
    GROUND = "ground"


def _restore_run_progress(run: PipelineRunContext, *, selected_count: int) -> None:
    """Use the DB-verified selection, not an optional previous-attempt report."""
    previous_selected = run.selected_count
    run.selected_count = selected_count
    run.succeeded_count = 0
    run.failed_count = 0
    print(
        'RAG_RESUME_PROGRESS_PREFLIGHT {"state":"ready",'
        '"denominator_source":"postgres_verified_selection",'
        f'"selected_count":{selected_count},"previous_selected_count":{previous_selected},'
        '"attempt_result_counts_reset":true}',
        flush=True,
    )


def _complete_verified_job_progress(stage: PipelineStage, *, succeeded_count: int) -> None:
    """Count reuse only after the runner has verified all terminal jobs."""
    if stage.total is None or stage.failed or succeeded_count != stage.total:
        raise ValueError("verified RAG job progress requires a complete successful job denominator")
    reused = stage.total - stage.processed
    if reused > 0:
        processed = stage.processed
        stage.advance(reused, phase="reused")
        print(
            f'RAG_JOB_PROGRESS_REUSE {{"stage":"{stage.name}","state":"verified",'
            f'"newly_processed_count":{processed},"reused_count":{reused},'
            f'"total":{stage.total},"counter_semantics":"processed_jobs_not_answer_success"}}',
            flush=True,
        )


class _DetachedRetrievalReader:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def get_batch(
        self, retrieval_batch_id: str
    ) -> tuple[
        RetrievalBatchTrace,
        list[RetrievalTrace],
        list[RetrievalTraceComponent],
        list[RetrievalQueryEmbeddingRef],
    ] | None:
        async with self.sessions() as session:
            return await PostgresRetrievalRepository(session).get_batch(
                retrieval_batch_id
            )


class _DetachedCorpusReader:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def get_window(self, corpus_id: str, window_id: str) -> MemoryWindow:
        async with self.sessions() as session:
            return await PostgresCorpusRepository(session).get_window(corpus_id, window_id)


def _load_answer_prompt(config: RagBaselineConfig, project_root: Path) -> str:
    path = config.resolved_prompt_path(project_root)
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("RAG answer prompt cannot be empty")
    return prompt


def _answer_prompt_for_metadata(
    config: RagBaselineConfig,
    project_root: Path,
    metadata: dict[str, object] | None,
) -> str:
    values = metadata or {}
    if baseline_answer_contract(config, metadata) == TYPED_ANSWER_CONTRACT_VERSION:
        return load_step8_prompt("pmmc_answer_contract_v3.txt")
    if values.get("answer_contract_version") != "strict-answer-contract-v2":
        return _load_answer_prompt(config, project_root)
    from pmmc.runtime.answer_policy import LEGACY_PROMPT_VERSION, load_answer_policy_prompt

    return load_answer_policy_prompt(
        project_root, str(values.get("answer_prompt_version") or LEGACY_PROMPT_VERSION),
    )


def _validate_capabilities(
    *,
    method: BaselineMethod,
    selected: list[BenchmarkQuery],
    config: RagBaselineConfig,
    registry: ModelRegistry,
) -> None:
    embed = registry.get(config.embedding_alias_for(method))
    if method is BaselineMethod.CAPTION_TEXT_RAG:
        embed.require(ModelCapability.EMBED_TEXT)
    elif method is BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG:
        embed.require(ModelCapability.EMBED_TEXT)
        embed.require(ModelCapability.EMBED_IMAGE)
    else:
        embed.require(ModelCapability.EMBED_MULTIMODAL)
    answer = registry.get(config.answer_model_alias)
    needs_images = (
        method is not BaselineMethod.CAPTION_TEXT_RAG
        or any(record.query.query_image_refs for record in selected)
    )
    answer.require(
        ModelCapability.CHAT_MULTIMODAL if needs_images else ModelCapability.CHAT_TEXT
    )
    if answer.context_window is None or answer.context_window < config.total_context_tokens:
        raise ValueError("RAG answer model must provide the frozen 32768-token context")


def _rag_manifest(
    *,
    run_id: str,
    method: BaselineMethod,
    corpus: BaselineCorpusManifest,
    snapshot: DatasetSnapshotManifest,
    config: RagBaselineConfig,
    answer_prompt: str,
    document_collection_id: str,
    settings: Settings,
    registry: ModelRegistry,
    run_metadata: dict[str, object] | None = None,
) -> RunManifest:
    aliases = baseline_model_aliases(
        config, run_metadata, [config.answer_model_alias, config.embedding_alias_for(method)],
    )
    extra_metadata = typed_baseline_metadata(config, run_metadata)
    reserved_metadata = {
        "rag_config",
        "corpus_id",
        "corpus_content_hash",
        "document_collection_id",
        "protocol_version",
        "renderer_version",
        "excluded_source_categories",
    }
    if reserved_metadata.intersection(extra_metadata):
        raise ValueError("run metadata cannot override RAG protocol fields")
    answer_prompt_version = str(
        extra_metadata.get("answer_prompt_version") or config.answer_prompt_version
    )
    strict_answer = extra_metadata.get("answer_contract_version") == (
        "strict-answer-contract-v2"
    )
    typed = baseline_answer_contract(config, extra_metadata) == TYPED_ANSWER_CONTRACT_VERSION
    manifest = RunManifest(
        run_id=run_id,
        experiment_name="five-foundational-baselines",
        dataset=corpus.dataset,
        dataset_version=snapshot.dataset_version,
        method=method.value,
        split=(
            str(extra_metadata["partition_role"])
            if extra_metadata.get("partition_role") is not None
            else None
        ),
        paper_mode=settings.mode is AppMode.PAPER,
        answer_model_alias=config.answer_model_alias,
        embedding_model_aliases=[config.embedding_alias_for(method)],
        retrieval_topk_max=config.retrieval_top_k,
        prompt_hashes={
            answer_prompt_version: rag_prompt_sha256(answer_prompt),
            config.query_instruction_version: query_instruction_sha256(config),
            **({"answer-grounding-v1": extra_metadata["grounding_prompt_hash"]} if typed else {}),
        },
        decoding_configs={
            "answer": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "seed": config.seed,
                "max_output_tokens": 512 if strict_answer or typed else config.max_output_tokens,
                **({"thinking_mode": "disabled", "format_retries": 1} if typed else {}),
            }
        },
        retrieval_config={
            "protocol_version": config.protocol_version,
            "corpus_id": corpus.corpus_id,
            "document_collection_id": document_collection_id,
            "retrieval_top_k": config.retrieval_top_k,
            "answer_top_k": config.answer_top_k,
            "distance_metric": config.distance_metric.value,
            "exact_scan": config.exact_scan,
            "reranker": config.use_reranker,
            "window_aggregation": "max_document_score",
        },
        evaluation_policy_version=snapshot.evaluation_policy_version,
        dataset_snapshot_ids=[corpus.snapshot_id],
        prompt_registry_version=answer_prompt_version,
        metadata={
            "rag_config": config.model_dump(mode="json"),
            "corpus_id": corpus.corpus_id,
            "corpus_content_hash": corpus.content_hash,
            "document_collection_id": document_collection_id,
            "protocol_version": config.protocol_version,
            "renderer_version": config.renderer_version,
            "excluded_source_categories": snapshot.excluded_source_categories,
            **extra_metadata,
        },
    )
    return bind_registry_to_manifest(manifest, registry, aliases=aliases)


async def _load_collection(
    sessions: async_sessionmaker[AsyncSession], collection_id: str
) -> EmbeddingCollection:
    async with ExperimentUnitOfWork(sessions) as uow:
        assert uow.embeddings is not None
        value = await uow.embeddings.get_collection(collection_id)
    if value is None:
        raise KeyError(f"embedding collection does not exist: {collection_id}")
    return value


async def _rehydrate_resumed_run_registration(
    *,
    sessions: async_sessionmaker[AsyncSession],
    run_dir: Path,
    method: BaselineMethod,
    selected_count: int,
) -> bool:
    """Restore DB registration when checkpoint rollback leaves a newer local run dir."""

    local = load_local_run_registration(run_dir)
    async with ExperimentUnitOfWork(sessions) as uow:
        assert uow.runs is not None
        try:
            state = await uow.runs.get_state(local.manifest.run_id)
        except KeyError:
            state = None
    if state is not None:
        if state.selected_count != selected_count:
            raise ValueError("resumed RAG database denominator differs from local registration")
        print(
            "RAG_RESUME_REGISTRATION_PREFLIGHT "
            f'{{"database_registration":"ready","method":"{method.value}",'
            f'"selected_count":{selected_count},"state":"ready"}}',
            flush=True,
        )
        return False

    print(
        "RAG_RESUME_REGISTRATION_PREFLIGHT "
        f'{{"database_registration":"missing","method":"{method.value}",'
        f'"selected_count":{selected_count},"state":"rehydrating"}}',
        flush=True,
    )
    async with sessions() as session, session.begin():
        status, registered_count = await register_local_run(
            session,
            run_dir,
            include_stage_summaries=False,
        )
    if registered_count != selected_count or status is not RunStatus.CREATED:
        raise RuntimeError("rehydrated RAG run registration has an invalid initial state")
    print(
        "RAG_RESUME_REGISTRATION_PREFLIGHT "
        f'{{"database_registration":"rehydrated","method":"{method.value}",'
        f'"selected_count":{registered_count},"state":"ready"}}',
        flush=True,
    )
    return True


async def _prepare_resumed_run_state(
    *,
    sessions: async_sessionmaker[AsyncSession],
    run_id: str,
    method: BaselineMethod,
    selected_count: int,
) -> RunStatus:
    """Validate a resumed run and start a freshly rehydrated registration."""

    async with ExperimentUnitOfWork(sessions) as uow:
        assert uow.runs is not None
        state = await uow.runs.get_state(run_id)
        initial_status = state.status
        if state.selected_count != selected_count:
            raise ValueError("resumed RAG run denominator differs from local selection")
        if initial_status is RunStatus.CREATED:
            state = await uow.runs.transition(run_id, RunStatus.RUNNING)

    final_status = state.status
    allowed = {RunStatus.RUNNING, RunStatus.COMPLETED}
    print(
        "RAG_RESUME_STATE_PREFLIGHT "
        f'{{"final_status":"{final_status.value}",'
        f'"initial_status":"{initial_status.value}",'
        f'"method":"{method.value}","selected_count":{selected_count},'
        f'"state":"{"ready" if final_status in allowed else "rejected"}",'
        f'"transitioned":{str(initial_status is RunStatus.CREATED).lower()}}}',
        flush=True,
    )
    if final_status not in allowed:
        raise RuntimeError(f"RAG run cannot resume from {final_status.value}")
    return final_status


async def run_rag_workflow(
    *,
    settings: Settings,
    method: BaselineMethod | None,
    snapshot_id: str | None,
    corpus_id: str | None,
    selection: BenchmarkSelection | None,
    answer_model_alias: str | None = None,
    phase: RagRunPhase = RagRunPhase.ALL,
    resume_run_id: str | None = None,
    run_id: str | None = None,
    run_metadata: dict[str, object] | None = None,
) -> Path:
    """Execute retrieval and answer phases while preserving runtime/gold isolation."""
    phase = RagRunPhase(phase)

    if resume_run_id is not None and run_id is not None:
        raise ValueError("resume_run_id and run_id are mutually exclusive")
    if resume_run_id is not None and run_metadata:
        raise ValueError("resume cannot change immutable run metadata")
    if resume_run_id is None:
        if (
            method not in RAG_METHODS
            or snapshot_id is None
            or corpus_id is None
            or selection is None
        ):
            raise ValueError("new RAG runs require method, snapshot, corpus, and selection")
        if phase in {RagRunPhase.ANSWER, RagRunPhase.GENERATE, RagRunPhase.GROUND}:
            raise ValueError("a new RAG run must execute retrieval before answer")
        if settings.mode is AppMode.PAPER and selection.include_excluded:
            raise ValueError("paper mode cannot include excluded benchmark categories")

    registry = load_model_registry(settings.resolved_model_registry_path)
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
            if method not in RAG_METHODS:
                raise ValueError("resumed run is not a Step 5C RAG run")
            if len(manifest.dataset_snapshot_ids) != 1:
                raise ValueError("RAG runs require exactly one canonical snapshot")
            snapshot_id = manifest.dataset_snapshot_ids[0]
            corpus_id = str(manifest.metadata.get("corpus_id") or "")
            if not corpus_id:
                raise ValueError("resumed RAG run is missing corpus lineage")
            selection = local.selection
            selected = local.queries
            config = RagBaselineConfig.model_validate(manifest.metadata.get("rag_config"))
            if answer_model_alias and answer_model_alias != config.answer_model_alias:
                raise ValueError("resume cannot change the answer model alias")
            await _rehydrate_resumed_run_registration(
                sessions=sessions,
                run_dir=run_dir,
                method=method,
                selected_count=len(selected),
            )
            database_selected = await query_repository.select(snapshot_id, selection)
            if database_selected != selected:
                raise ValueError("resumed selection differs from Postgres source of truth")
            snapshot = await snapshot_repository.get_manifest(snapshot_id)
            async with sessions() as session:
                corpus = await PostgresCorpusRepository(session).get_manifest(corpus_id)
            document_collection_id = str(
                manifest.metadata.get("document_collection_id") or ""
            )
            if not document_collection_id:
                raise ValueError("resumed RAG run is missing document index lineage")
            aliases = baseline_model_aliases(config, manifest.metadata, [
                config.answer_model_alias, config.embedding_alias_for(method),
            ])
            verify_manifest_registry_compatibility(manifest, registry, aliases)
        else:
            assert method is not None
            assert snapshot_id is not None and corpus_id is not None and selection is not None
            snapshot = await snapshot_repository.get_manifest(snapshot_id)
            async with sessions() as session:
                corpus = await PostgresCorpusRepository(session).get_manifest(corpus_id)
            if corpus.snapshot_id != snapshot_id:
                raise ValueError("RAG corpus does not belong to the requested snapshot")
            if selection.datasets != [snapshot.dataset]:
                raise ValueError("selection dataset must match the canonical snapshot")
            config = load_rag_config(
                settings.resolved_rag_config_path,
                answer_model_alias=answer_model_alias,
            )
            selected = await query_repository.select(snapshot_id, selection)
            if not selected:
                raise ValueError("benchmark selection produced no queries")
            _validate_capabilities(
                method=method, selected=selected, config=config, registry=registry
            )
            index_result = await build_rag_index(
                settings=settings,
                registry=registry,
                config=config,
                corpus_id=corpus_id,
                method=method,
            )
            document_collection_id = index_result.collection.collection_id
            actual_run_id = run_id or make_run_id(method.value)
            answer_prompt = _answer_prompt_for_metadata(
                config,
                settings.project_root,
                run_metadata,
            )
            manifest = _rag_manifest(
                run_id=actual_run_id,
                method=method,
                corpus=corpus,
                snapshot=snapshot,
                config=config,
                answer_prompt=answer_prompt,
                document_collection_id=document_collection_id,
                settings=settings,
                registry=registry,
                run_metadata=run_metadata,
            )
            run_dir = settings.runs_path / actual_run_id

        assert method is not None and snapshot_id is not None and corpus_id is not None
        assert selection is not None
        typed_baseline_metadata(config, manifest.metadata)
        if phase in {RagRunPhase.GENERATE, RagRunPhase.GROUND} and (
            baseline_answer_contract(config, manifest.metadata) != TYPED_ANSWER_CONTRACT_VERSION
        ):
            raise ValueError("split baseline phases require typed v3")
        answer_prompt = _answer_prompt_for_metadata(
            config,
            settings.project_root,
            manifest.metadata,
        )
        answer_prompt_version = str(
            manifest.metadata.get("answer_prompt_version")
            or config.answer_prompt_version
        )
        if manifest.prompt_hashes.get(answer_prompt_version) != rag_prompt_sha256(
            answer_prompt
        ):
            raise ValueError("current answer prompt differs from immutable RAG manifest")
        if manifest.prompt_hashes.get(
            config.query_instruction_version
        ) != query_instruction_sha256(config):
            raise ValueError("current query instruction differs from immutable RAG manifest")
        _validate_capabilities(
            method=method, selected=selected, config=config, registry=registry
        )
        document_collection = await _load_collection(sessions, document_collection_id)
        if (
            document_collection.status is not EmbeddingCollectionStatus.READY
            or document_collection.corpus_id != corpus_id
        ):
            raise ValueError("RAG run requires a ready index for its exact corpus")

        if resume:
            resumed_status = await _prepare_resumed_run_state(
                sessions=sessions,
                run_id=manifest.run_id,
                method=method,
                selected_count=len(selected),
            )
            if resumed_status is RunStatus.COMPLETED:
                async with sessions() as session:
                    report = await verify_experiment_run(session, manifest.run_id)
                if not report["valid"]:
                    raise RuntimeError("completed RAG run failed integrity verification")
                return run_dir

        run = PipelineRunContext(
            runs_root=settings.runs_path,
            manifest=manifest,
            datasets=[snapshot.dataset],
            resume=resume,
        )
        if resume:
            _restore_run_progress(run, selected_count=len(selected))
        with run:
            with run.stage("preflight", dataset=snapshot.dataset, total=1) as stage:
                stage.advance()
            with run.stage(
                "selection-register", dataset=snapshot.dataset, total=len(selected) + 1
            ) as stage:
                if not resume:
                    selected = run.record_selection(selection, selected)
                    manifest = run.manifest
                stage.advance(len(selected))
                if not resume:
                    async with sessions() as session, session.begin():
                        await register_local_run(session, run.run_dir)
                stage.advance()

            query_collection_template = make_query_collection(
                run_id=manifest.run_id,
                snapshot_id=snapshot_id,
                method=method,
                config=config,
                registry=registry,
            )
            async with ExperimentUnitOfWork(sessions) as uow:
                assert uow.embeddings is not None
                query_collection = await uow.embeddings.get_collection(
                    query_collection_template.collection_id
                )
                if query_collection is None:
                    query_collection = await uow.embeddings.add_collection(
                        query_collection_template
                    )

            store = create_runtime_object_store(settings)
            artifacts = ExperimentArtifactService(
                sessions,
                store,
                DataScope.RUNTIME,
                artifact_prefix=settings.experiment_artifact_prefix,
            )
            from pmmc.baselines.runner import TransactionalModelCallWriter

            execution = create_model_execution_service(
                settings,
                registry,
                artifact_service=artifacts,
                model_calls=TransactionalModelCallWriter(sessions),
            )
            resolver = create_image_url_resolver(settings, store)

            if phase in {RagRunPhase.ALL, RagRunPhase.RETRIEVE}:
                embedding_gateway = create_embedding_gateway(
                    settings,
                    registry,
                    config.embedding_alias_for(method),
                    image_url_resolver=resolver,
                )
                retrieval_runner = RagRetrievalRunner(
                    sessions=sessions,
                    execution=execution,
                    gateway=embedding_gateway,
                    error_artifacts=artifacts,
                    method=method,
                    config=config,
                    corpus_id=corpus_id,
                    document_collection=document_collection,
                    query_collection=query_collection,
                    lease_seconds=settings.experiment_job_lease_seconds,
                )
                by_id = {item.query.query_id: item for item in selected}
                with run.stage(
                    "retrieve-jobs", dataset=snapshot.dataset, total=len(selected)
                ) as stage:

                    def on_retrieval(
                        query_id: str, success: bool, error_type: str | None
                    ) -> None:
                        record = by_id[query_id]
                        if success:
                            stage.advance(query_id=query_id, source=record.source)
                        else:
                            stage.fail_item(
                                RuntimeError(error_type or "RagRetrievalFailure"),
                                query_id=query_id,
                                source=record.source,
                            )

                    retrieved_count, _ = await retrieval_runner.execute(
                        run_id=manifest.run_id,
                        snapshot_id=snapshot_id,
                        selected=selected,
                        on_item=on_retrieval,
                    )
                    _complete_verified_job_progress(stage, succeeded_count=retrieved_count)

            if phase in {
                RagRunPhase.ALL, RagRunPhase.ANSWER, RagRunPhase.GENERATE, RagRunPhase.GROUND,
            }:
                query_collection = await _load_collection(
                    sessions, query_collection_template.collection_id
                )
                if query_collection.status is not EmbeddingCollectionStatus.READY:
                    raise ValueError("RAG answer phase requires frozen query retrieval facts")
                counter = create_token_counter(
                    settings, registry, config.answer_model_alias
                )
                grounding = None
                if phase is not RagRunPhase.GENERATE and baseline_answer_contract(
                    config, manifest.metadata,
                ) == TYPED_ANSWER_CONTRACT_VERSION:
                    grounding = configured_grounding_service(
                        settings=settings, execution=execution, image_url_resolver=resolver,
                    )
                    if grounding is None:
                        raise ValueError(
                            "typed baseline requires a configured 9B grounding endpoint",
                        )
                provider = RagAnswerInputProvider(
                    retrieval=_DetachedRetrievalReader(sessions),
                    corpus=_DetachedCorpusReader(sessions),
                    memories=PostgresMemoryRepository(sessions),
                    sessions=PostgresSessionRepository(sessions),
                    budgeter=RagContextBudgeter(
                        counter,
                        total_context_tokens=config.total_context_tokens,
                        reserved_output_tokens=config.reserved_output_tokens,
                        paper_mode=settings.mode is AppMode.PAPER,
                    ),
                    config=config,
                    instruction=answer_prompt,
                    timeout_seconds=settings.model_request_timeout_seconds,
                    paper_mode=settings.mode is AppMode.PAPER,
                    runtime_bucket=settings.object_store_bucket,
                    answer_contract_version=(
                        str(manifest.metadata["answer_contract_version"])
                        if manifest.metadata.get("answer_contract_version") is not None
                        else None
                    ),
                )
                answer_runner = RagAnswerRunner(
                    sessions=sessions,
                    input_provider=provider,
                    answer_service=BaselineAnswerService(
                        execution,
                        None if phase is RagRunPhase.GROUND else create_generation_gateway(
                            settings,
                            registry,
                            config.answer_model_alias,
                            image_url_resolver=resolver,
                        ),
                        grounding=grounding,
                    ),
                    error_artifacts=artifacts,
                    method=method,
                    config=config,
                    lease_seconds=settings.experiment_job_lease_seconds,
                    phase=phase.value if phase in {RagRunPhase.GENERATE, RagRunPhase.GROUND}
                    else "answer",
                )
                by_id = {item.query.query_id: item for item in selected}
                with run.stage(
                    "answer-jobs", dataset=snapshot.dataset, total=len(selected)
                ) as stage:

                    def on_answer(
                        query_id: str, success: bool, error_type: str | None
                    ) -> None:
                        record = by_id[query_id]
                        if success:
                            stage.advance(query_id=query_id, source=record.source)
                        else:
                            stage.fail_item(
                                RuntimeError(error_type or "RagAnswerFailure"),
                                query_id=query_id,
                                source=record.source,
                            )

                    answered_job_count, _ = await answer_runner.execute(
                        run_id=manifest.run_id,
                        snapshot_id=snapshot_id,
                        selected=selected,
                        on_item=on_answer,
                    )
                    _complete_verified_job_progress(stage, succeeded_count=answered_job_count)
                if phase is RagRunPhase.GENERATE:
                    return run.run_dir
                with run.stage(
                    "finalize-verify", dataset=snapshot.dataset, total=1
                ) as stage:
                    async with sessions() as session:
                        report = await verify_experiment_run(session, manifest.run_id)
                    if not report["valid"]:
                        raise RuntimeError(
                            "completed RAG run failed verification: "
                            + "; ".join(str(item) for item in report["issues"])
                        )
                    stage.advance()
                run.succeeded_count = len(selected)
            else:
                run.succeeded_count = len(selected)
        return run.run_dir
    finally:
        await engine.dispose()
