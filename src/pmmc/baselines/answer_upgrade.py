"""Rebind verified pre-answer Native-MM retrieval to a new answer-only child."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from pmmc.baselines.rag_config import load_rag_config
from pmmc.baselines.rag_embedding import embedding_record_id, make_query_collection
from pmmc.baselines.rag_retrieval import retrieval_batch_id, retrieval_trace_id
from pmmc.baselines.rag_runner import rag_job_id
from pmmc.baselines.suite import baseline_suite_child_run_id
from pmmc.baselines.suite_contract import (
    BaselineSuiteContractError,
    validate_suite_run_answer_contract,
)
from pmmc.baselines.suite_recovery import _restore_registration, preflight_native_fallback_retrieval
from pmmc.benchmarks.ids import stable_hash
from pmmc.experiments.manifest import run_config_hash
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.evaluation_repositories import PostgresBaselineSuiteRepository
from pmmc.infrastructure.database.experiment_models import (
    FinalAnswerModel,
    JobAttemptModel,
    RunQueryModel,
    RuntimeModelCallModel,
)
from pmmc.infrastructure.database.experiment_repositories import (
    PostgresEmbeddingRepository,
    PostgresJobRepository,
    PostgresRetrievalRepository,
    PostgresRunRepository,
)
from pmmc.infrastructure.database.query_executor_models import StructuredAnswerOutputModel
from pmmc.infrastructure.database.repositories import PostgresRuntimeQueryRepository
from pmmc.infrastructure.settings import Settings
from pmmc.models.registry import load_model_registry, verify_manifest_registry_compatibility
from pmmc.runtime.answer_policy import (
    CURRENT_PROMPT_HASH,
    CURRENT_PROMPT_VERSION,
    CURRENT_RETRY_POLICY,
    CURRENT_RETRY_SHA256,
    LEGACY_PROMPT_VERSION,
)
from pmmc.schemas import (
    AppStage,
    BaselineMethod,
    BaselineSuiteManifest,
    BenchmarkSelection,
    EmbeddingCollectionStatus,
    ExperimentJob,
    JobStatus,
    RunManifest,
    RunQueryAssignment,
    RunStatus,
)
from pmmc.schemas.base import utc_now


class AnswerUpgradeError(BaselineSuiteContractError):
    """Content-free diagnostic for a failed pre-answer migration operation."""

    def __init__(
        self, code: str, *, phase: str, dataset: str = "unknown",
        model_identity_differences: dict[str, int] | None = None,
    ) -> None:
        super().__init__(code)
        self.error_code = code
        self.phase = phase
        self.dataset = dataset if dataset in {"memlens", "memgallery"} else "unknown"
        self.model_identity_differences = dict(model_identity_differences or {})

    def safe_diagnostics(self) -> dict[str, object]:
        return {
            "failure_scope": "fallback_answer_upgrade",
            "contract_error_code": self.error_code,
            "upgrade_phase": self.phase,
            "dataset": self.dataset,
            "registry_scope": "prepared_commit_resolved",
            "model_identity_differences": self.model_identity_differences,
        }


def resolved_answer_upgrade_settings(settings: Settings, registry_path: Path) -> Settings:
    """Match the GPU child's frozen registry without changing runtime-planner inputs."""
    if not registry_path.is_file():
        raise AnswerUpgradeError(
            "answer_upgrade_resolved_registry_missing", phase="model_registry_load"
        )
    return settings.model_copy(update={"model_registry_path": registry_path})


@dataclass
class _UpgradeProgress:
    phase: str = "source_retrieval_preflight"
    dataset: str = "unknown"
    query_count: int = 0
    copied_queries: int = 0
    model_identity_differences: dict[str, int] = field(default_factory=dict)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise BaselineSuiteContractError(code)


async def _expected_retrieval_reuse_job(
    session: AsyncSession, target: RunManifest, query_id: str,
) -> ExperimentJob:
    """A logical completion receipt, never a fabricated worker execution attempt."""
    reuse = target.metadata.get("retrieval_reuse", {})
    _require(
        reuse.get("policy") == "native-mm-answer-only-rebind-v1"
        and target.method == BaselineMethod.NATIVE_MULTIMODAL_RAG.value,
        "answer_upgrade_job_reuse_scope_invalid",
    )
    source_id = str(reuse.get("source_run_id", ""))
    source = await PostgresRunRepository(session).get_manifest(source_id)
    _require(
        source_id != target.run_id and source.config_hash == reuse.get("source_config_hash")
        and source.dataset_snapshot_ids == target.dataset_snapshot_ids
        and source.selection_hash == target.selection_hash and source.method == target.method,
        "answer_upgrade_job_source_scope_invalid",
    )
    method = BaselineMethod.NATIVE_MULTIMODAL_RAG
    source_job = await PostgresJobRepository(session).get(
        rag_job_id(source_id, query_id, method, AppStage.RETRIEVE)
    )
    _require(source_job is not None, "answer_upgrade_source_retrieval_job_missing")
    assert source_job is not None
    _require(
        source_job.status is JobStatus.SUCCEEDED and source_job.attempt_count > 0
        and source_job.run_id == source_id and source_job.query_id == query_id
        and source_job.snapshot_id in target.dataset_snapshot_ids
        and source_job.stage is AppStage.RETRIEVE and source_job.variant == "retrieval"
        and source_job.metadata.get("baseline_method") == method.value,
        "answer_upgrade_source_retrieval_job_invalid",
    )
    attempt = await session.scalar(select(JobAttemptModel).where(
        JobAttemptModel.job_id == source_job.job_id,
        JobAttemptModel.attempt_number == source_job.attempt_count,
        JobAttemptModel.status == JobStatus.SUCCEEDED.value,
    ))
    _require(attempt is not None, "answer_upgrade_source_retrieval_attempt_missing")
    assert attempt is not None
    retrieval = PostgresRetrievalRepository(session)
    original = await retrieval.get_batch(retrieval_batch_id(source_id, query_id, method))
    copied = await retrieval.get_batch(retrieval_batch_id(target.run_id, query_id, method))
    _require(original is not None and copied is not None, "answer_upgrade_job_batch_missing")
    assert original is not None and copied is not None
    batch, traces, components, refs = original
    facts_hash = stable_hash([
        batch.model_dump(mode="json"), [t.model_dump(mode="json") for t in traces],
        [c.model_dump(mode="json") for c in components],
        [r.model_dump(mode="json") for r in refs],
    ])
    _require(
        copied[0].metadata.get("source_facts_hash") == facts_hash
        and copied[0].metadata.get("source_run_id") == source_id
        and copied[0].metadata.get("reused_from_retrieval_batch") == batch.retrieval_batch_id,
        "answer_upgrade_job_retrieval_provenance_invalid",
    )
    now = utc_now()
    return ExperimentJob(
        job_id=rag_job_id(target.run_id, query_id, method, AppStage.RETRIEVE),
        run_id=target.run_id, snapshot_id=source_job.snapshot_id, query_id=query_id,
        stage=AppStage.RETRIEVE, variant="retrieval", status=JobStatus.SUCCEEDED,
        max_attempts=int(target.metadata["rag_config"]["job_max_attempts"]),
        created_at=now, available_at=now, finished_at=now,
        metadata={"baseline_method": method.value, "retrieval_reuse": {
            "policy": "native-mm-answer-only-rebind-v1", "source_run_id": source_id,
            "source_job_id": source_job.job_id, "source_attempt_id": attempt.attempt_id,
            "source_job_hash": stable_hash(source_job.model_dump(mode="json")),
            "source_facts_hash": facts_hash, "source_batch_id": batch.retrieval_batch_id,
            "new_execution_attempts": 0, "new_model_calls": 0,
        }},
    )


async def verify_rebound_retrieval_job(session: AsyncSession, job: ExperimentJob) -> None:
    """Check reuse provenance before an ordinary retrieval runner skips this job."""
    _require(job.query_id is not None, "answer_upgrade_job_query_missing")
    assert job.query_id is not None
    target = await PostgresRunRepository(session).get_manifest(job.run_id)
    expected = await _expected_retrieval_reuse_job(session, target, job.query_id)
    _require(
        job.model_dump(exclude={"created_at", "available_at", "finished_at"})
        == expected.model_dump(exclude={"created_at", "available_at", "finished_at"}),
        "answer_upgrade_target_retrieval_job_conflict",
    )


def upgraded_manifest(source: RunManifest, suite: BaselineSuiteManifest) -> RunManifest:
    """New immutable child identity; neither old answers nor old costs are copied."""
    _require(
        source.metadata.get("answer_prompt_version") == LEGACY_PROMPT_VERSION
        and suite.metadata.get("answer_prompt_version") == CURRENT_PROMPT_VERSION
        and source.metadata.get("answer_contract_version") == "strict-answer-contract-v2"
        and suite.metadata.get("answer_contract_version") == "strict-answer-contract-v2"
        and suite.methods == [BaselineMethod.NATIVE_MULTIMODAL_RAG]
        and source.method == BaselineMethod.NATIVE_MULTIMODAL_RAG.value
        and source.dataset == suite.dataset
        and source.dataset_snapshot_ids == [suite.snapshot_id]
        and source.selection_hash == suite.selection_hash
        and source.metadata.get("corpus_id") == suite.metadata.get("corpus_id")
        and source.answer_model_alias == suite.answer_model_alias,
        "answer_upgrade_source_scope_mismatch",
    )
    run_id = baseline_suite_child_run_id(suite.suite_id, BaselineMethod.NATIVE_MULTIMODAL_RAG)
    _require(run_id != source.run_id, "answer_upgrade_requires_new_child")
    hashes = {k: v for k, v in source.prompt_hashes.items() if k != LEGACY_PROMPT_VERSION}
    hashes[CURRENT_PROMPT_VERSION] = CURRENT_PROMPT_HASH
    value = source.model_copy(
        update={
            "run_id": run_id,
            "created_at": suite.created_at,
            "prompt_registry_version": CURRENT_PROMPT_VERSION,
            "prompt_hashes": hashes,
            "metadata": {
                **source.metadata,
                "answer_prompt_version": CURRENT_PROMPT_VERSION,
                "answer_format_retry_policy": CURRENT_RETRY_POLICY,
                "answer_format_retry_sha256": CURRENT_RETRY_SHA256,
                "retrieval_reuse": {
                    "policy": "native-mm-answer-only-rebind-v1",
                    "source_run_id": source.run_id,
                    "source_config_hash": source.config_hash,
                    "new_embedding_calls": 0,
                    "answers_reused": 0,
                    "source_cost_counted_as_new": False,
                },
            },
        }
    )
    value = value.model_copy(update={"config_hash": run_config_hash(value)})
    validate_suite_run_answer_contract(suite, value)
    return value


async def rebind_native_fallback(
    *,
    settings: Settings,
    source_suite_id: str,
    target_suite_id: str,
) -> None:
    """Copy small, typed query vectors/ranks atomically; keep original call lineage."""
    progress = _UpgradeProgress()
    started = monotonic()
    try:
        await _rebind_native_fallback(
            settings=settings, source_suite_id=source_suite_id,
            target_suite_id=target_suite_id, progress=progress,
        )
    except Exception as exc:
        code = (
            str(exc.diagnostics["contract_error_code"])
            if isinstance(exc, BaselineSuiteContractError)
            else f"answer_upgrade_{progress.phase}_failed"
        )
        error = AnswerUpgradeError(
            code, phase=progress.phase, dataset=progress.dataset,
            model_identity_differences=progress.model_identity_differences,
        )
        print("STEP7B_ANSWER_UPGRADE " + json.dumps({
            **error.safe_diagnostics(), "state": "failed", "error_type": type(exc).__name__,
            "elapsed_seconds": round(monotonic() - started, 3),
            "query_count": progress.query_count,
            "copied_queries_before_failure": progress.copied_queries,
            "model_calls": 0,
        }, sort_keys=True), flush=True)
        raise error from exc


async def _rebind_native_fallback(
    *, settings: Settings, source_suite_id: str, target_suite_id: str,
    progress: _UpgradeProgress,
) -> None:
    method = BaselineMethod.NATIVE_MULTIMODAL_RAG
    source_id = baseline_suite_child_run_id(source_suite_id, method)
    target_id = baseline_suite_child_run_id(target_suite_id, method)
    await preflight_native_fallback_retrieval(
        settings=settings,
        suite_id=source_suite_id,
        expected_run_id=source_id,
    )
    progress.phase = "model_registry_load"
    registry = load_model_registry(settings.resolved_model_registry_path)
    progress.phase = "database_open"
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session, session.begin():
            progress.phase = "source_manifest"
            runs = PostgresRunRepository(session)
            suites = PostgresBaselineSuiteRepository(session)
            source = await runs.get_manifest(source_id)
            progress.dataset = source.dataset.value
            suite = await suites.get_manifest(target_suite_id)
            source_suite = await suites.get_manifest(source_suite_id)
            validate_suite_run_answer_contract(source_suite, source)
            progress.phase = "source_answer_guard"
            # Only a pre-answer snapshot is an admissible import boundary.
            for model, conditions in (
                (FinalAnswerModel, [FinalAnswerModel.run_id == source_id]),
                (StructuredAnswerOutputModel, [StructuredAnswerOutputModel.run_id == source_id]),
                (
                    RuntimeModelCallModel,
                    [
                        RuntimeModelCallModel.run_id == source_id,
                        RuntimeModelCallModel.stage == "answer",
                    ],
                ),
            ):
                _require(
                    not await session.scalar(
                        select(func.count())
                        .select_from(model)
                        .where(
                            *conditions,
                        )
                    ),
                    "answer_upgrade_source_has_online_answers",
                )
            progress.phase = "target_manifest"
            target = upgraded_manifest(source, suite)
            progress.phase = "retrieval_config"
            config = load_rag_config(
                settings.resolved_rag_config_path, answer_model_alias=suite.answer_model_alias
            )
            _require(
                source.metadata.get("rag_config") == config.model_dump(mode="json"),
                "answer_upgrade_retrieval_config_changed",
            )
            progress.phase = "model_identity"
            required_aliases = [config.answer_model_alias, config.embedding_alias_for(method)]
            for alias in required_aliases:
                current = registry.get(alias).model_dump(mode="json")
                frozen = source.model_configs.get(alias, {})
                # Field names only; never log model values, endpoints or prompt contents.
                for name in current:
                    if current[name] != frozen.get(name):
                        progress.model_identity_differences[name] = (
                            progress.model_identity_differences.get(name, 0) + 1
                        )
            verify_manifest_registry_compatibility(
                source, registry, required_aliases,
            )
            progress.phase = "target_registration"
            rows = list(
                (
                    await session.scalars(
                        select(RunQueryModel)
                        .where(
                            RunQueryModel.run_id == source_id,
                        )
                        .order_by(RunQueryModel.ordinal)
                    )
                ).all()
            )
            progress.query_count = len(rows)
            print("STEP7B_ANSWER_UPGRADE " + json.dumps({
                "state": "identity_verified", "dataset": progress.dataset,
                "registry_scope": "prepared_commit_resolved", "required_alias_count": 2,
                "query_count": len(rows), "model_calls": 0,
            }, sort_keys=True), flush=True)
            assignments = [
                RunQueryAssignment(
                    run_id=target_id,
                    snapshot_id=row.snapshot_id,
                    query_id=row.query_id,
                    ordinal=row.ordinal,
                    selection_hash=row.selection_hash,
                    shard_count=row.shard_count,
                    shard_index=row.shard_index,
                    created_at=target.created_at,
                    metadata=row.metadata_json,
                )
                for row in rows
            ]
            state = await runs.register(target, assignments)
            progress.phase = "query_collection"
            embeddings = PostgresEmbeddingRepository(session)
            retrieval = PostgresRetrievalRepository(session)
            old_template = make_query_collection(
                run_id=source_id,
                snapshot_id=suite.snapshot_id,
                method=method,
                config=config,
                registry=registry,
            )
            old_collection = await embeddings.get_collection(old_template.collection_id)
            _require(old_collection is not None, "answer_upgrade_source_collection_missing")
            assert old_collection is not None
            identity_fields = {"created_at", "status", "index_type", "index_params", "metadata"}
            _require(
                old_collection.status is EmbeddingCollectionStatus.READY
                and old_collection.model_dump(exclude=identity_fields)
                == old_template.model_dump(exclude=identity_fields),
                "answer_upgrade_source_collection_identity_changed",
            )
            template = make_query_collection(
                run_id=target_id,
                snapshot_id=suite.snapshot_id,
                method=method,
                config=config,
                registry=registry,
            )
            template = template.model_copy(
                update={
                    "created_at": target.created_at,
                    "metadata": {
                        **template.metadata,
                        "reused_from_collection": old_collection.collection_id,
                    },
                }
            )
            existing = await embeddings.get_collection(template.collection_id)
            if existing is None:
                await embeddings.add_collection(template)
            else:
                _require(
                    existing.model_dump(exclude={"status", "index_type", "index_params"})
                    == template.model_dump(exclude={"status", "index_type", "index_params"}),
                    "answer_upgrade_target_collection_conflict",
                )
            copied_ids: set[str] = set()
            progress.phase = "retrieval_copy"
            for row in rows:
                batch_id = retrieval_batch_id(source_id, row.query_id, method)
                facts = await retrieval.get_batch(batch_id)
                _require(facts is not None, "answer_upgrade_source_batch_missing")
                assert facts is not None
                batch, traces, components, refs = facts
                _require(
                    batch.query_embedding_collection_id == old_collection.collection_id,
                    "answer_upgrade_batch_collection_mismatch",
                )
                records = await embeddings.get_records([ref.embedding_id for ref in refs])
                _require(len(records) == len(refs), "answer_upgrade_source_vectors_missing")
                remap = {}
                for record in records.values():
                    _require(
                        record.collection_id == old_collection.collection_id
                        and record.metadata.get("query_id") == row.query_id
                        and record.model_call_id is not None,
                        "answer_upgrade_vector_scope_mismatch",
                    )
                    call = await session.get(RuntimeModelCallModel, record.model_call_id)
                    _require(
                        call is not None
                        and call.status == "success"
                        and call.stage == "embed"
                        and call.run_id == source_id
                        and call.query_id == row.query_id
                        and call.snapshot_id == suite.snapshot_id,
                        "answer_upgrade_embedding_call_lineage_invalid",
                    )
                    new_id = embedding_record_id(template.collection_id, record.source_id)
                    remap[record.embedding_id] = new_id
                    await embeddings.add_records(
                        [
                            record.model_copy(
                                update={
                                    "embedding_id": new_id,
                                    "collection_id": template.collection_id,
                                    "metadata": {
                                        **record.metadata,
                                        "reused_from_embedding": record.embedding_id,
                                        "source_run_id": source_id,
                                        "new_model_call": False,
                                    },
                                }
                            )
                        ]
                    )
                    copied_ids.add(new_id)
                target_batch = retrieval_batch_id(target_id, row.query_id, method)
                trace_ids = {
                    t.retrieval_trace_id: retrieval_trace_id(
                        target_batch,
                        t.rank,
                        str(t.retrieval_window_id),
                    )
                    for t in traces
                }
                provenance = {
                    "reused_from_retrieval_batch": batch_id,
                    "source_run_id": source_id,
                    "new_model_calls": 0,
                }
                await retrieval.add_batch(
                    batch.model_copy(
                        update={
                            "retrieval_batch_id": target_batch,
                            "run_id": target_id,
                            "query_embedding_collection_id": template.collection_id,
                            "embed_latency_ms": 0.0,
                            "search_latency_ms": 0.0,
                            "total_latency_ms": 0.0,
                            "metadata": {
                                **batch.metadata,
                                **provenance,
                                "source_facts_hash": stable_hash(
                                    [
                                        batch.model_dump(mode="json"),
                                        [t.model_dump(mode="json") for t in traces],
                                        [c.model_dump(mode="json") for c in components],
                                        [r.model_dump(mode="json") for r in refs],
                                    ]
                                ),
                            },
                        }
                    ),
                    [
                        t.model_copy(
                            update={
                                "retrieval_trace_id": trace_ids[t.retrieval_trace_id],
                                "retrieval_batch_id": target_batch,
                                "run_id": target_id,
                            }
                        )
                        for t in traces
                    ],
                    components=[
                        c.model_copy(
                            update={
                                "retrieval_trace_id": trace_ids[c.retrieval_trace_id],
                            }
                        )
                        for c in components
                    ],
                    query_embeddings=[
                        r.model_copy(
                            update={
                                "retrieval_batch_id": target_batch,
                                "embedding_id": remap[r.embedding_id],
                            }
                        )
                        for r in refs
                    ],
                )
                progress.copied_queries += 1
            progress.phase = "retrieval_job_reuse"
            jobs = PostgresJobRepository(session)
            created_jobs = 0
            for row in rows:
                expected_job = await _expected_retrieval_reuse_job(session, target, row.query_id)
                existing_job = await jobs.get(expected_job.job_id)
                if existing_job is None:
                    await jobs.add(expected_job)
                    created_jobs += 1
                else:
                    await verify_rebound_retrieval_job(session, existing_job)
            print("STEP7B_ANSWER_UPGRADE_JOB_PREFLIGHT " + json.dumps({
                "state": "verified_before_commit", "dataset": progress.dataset,
                "expected_queries": len(rows), "verified_reused_jobs": len(rows),
                "created_reuse_receipts": created_jobs,
                "existing_reuse_receipts": len(rows) - created_jobs,
                "new_retrieval_attempts": 0, "new_model_calls": 0,
                "target_answers_modified": False,
            }, sort_keys=True), flush=True)
            progress.phase = "collection_finalize"
            _require(
                len(copied_ids) == await embeddings.count_records(old_collection.collection_id),
                "answer_upgrade_vector_denominator_mismatch",
            )
            await embeddings.mark_collection_ready(
                template.collection_id,
                index_type=old_collection.index_type,
                index_params=old_collection.index_params,
            )
            if state.status is RunStatus.CREATED:
                await runs.transition(target_id, RunStatus.RUNNING)
            progress.phase = "database_commit"
        # Local files are reconstructible; the atomic database inventory is authoritative.
        progress.phase = "local_registration"
        selection = BenchmarkSelection.model_validate(suite.metadata["selection"])
        queries = await PostgresRuntimeQueryRepository(sessions).select(
            suite.snapshot_id, selection
        )
        directory = settings.runs_path / target_id
        directory.mkdir(parents=True, exist_ok=True)
        _restore_registration(directory, target, selection, queries)
        progress.phase = "target_retrieval_preflight"
        await preflight_native_fallback_retrieval(
            settings=settings,
            suite_id=target_suite_id,
            expected_run_id=target_id,
        )
        print(
            "STEP7B_ANSWER_UPGRADE "
            + json.dumps(
                {
                    "state": "ready",
                    "dataset": suite.dataset.value,
                    "query_count": len(rows),
                    "reused_vector_count": len(copied_ids),
                    "reused_retrieval_batches": len(rows),
                    "reused_retrieval_jobs": len(rows),
                    "new_retrieval_attempts": 0,
                    "new_embedding_calls": 0,
                    "old_answers_reused": 0,
                    "answer_prompt_version": CURRENT_PROMPT_VERSION,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        await engine.dispose()
