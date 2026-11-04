"""Registration, execution, and verification for one Step 7 query run."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from sqlalchemy.exc import DBAPIError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.baselines.rag_embedding import normalize_vector
from pmmc.baselines.runner import TransactionalModelCallWriter
from pmmc.compiler.config import load_compiler_config
from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.errors import (
    ExperimentPersistenceError,
    safe_database_error_diagnostics,
)
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.repositories import (
    PostgresImageRepository,
    PostgresMemoryRepository,
    PostgresRuntimeQueryRepository,
)
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.infrastructure.object_store import create_runtime_object_store
from pmmc.infrastructure.settings import Settings
from pmmc.models.factory import (
    create_embedding_gateway,
    create_generation_gateway,
    create_image_url_resolver,
    create_model_execution_service,
    create_token_counter,
)
from pmmc.models.registry import ModelRegistry
from pmmc.runtime.answer_context import PmmcEvidenceAnswerInputProvider
from pmmc.runtime.answering import (
    SharedAnswerGenerationService,
    strict_final_answer_failure_codes,
)
from pmmc.runtime.grounding import configured_grounding_service
from pmmc.runtime.joint_embeddings import JOINT_MODEL, prepare_joint_shards
from pmmc.runtime.qbank_reranker import (
    BoundedQBankReranker,
    preflight_reranker_contract,
    reranker_contract_hash,
)
from pmmc.runtime.query_backend import (
    PostgresNativeMultimodalFallbackProvider,
    PostgresOnlineQueryExecutorBackend,
)
from pmmc.runtime.query_embeddings import (
    QUERY_MATCH_INSTRUCTION,
    QUERY_MATCH_INSTRUCTION_VERSION,
    CoverageQueryEmbeddingImporter,
    QueryEmbeddingWorkflow,
    required_query_embedding_channels,
    runtime_query_cache_lease,
)
from pmmc.runtime.query_executor import OnlineQueryExecutor
from pmmc.runtime.query_executor_config import (
    QueryExecutorConfig,
    query_executor_config_hash,
    query_executor_protocol_hash,
    query_executor_routing_policy_hash,
)
from pmmc.runtime.query_runner import QueryExecutorBatchResult, QueryExecutorBatchRunner
from pmmc.schemas import (
    BaselineMethod,
    CompilationTargetStage,
    DataScope,
    DatasetName,
    EmbeddingInput,
    EmbeddingRecord,
    EmbeddingRequest,
    ModelCapability,
    QBankCoverageQueryEmbedding,
    QBankCoverageRunManifest,
    QBankCoverageStatus,
    QBankRerankerOutput,
    QBankRuntimeChannel,
    QueryEmbeddingBatch,
    QueryEmbeddingFact,
    QueryExecutorRunManifest,
    QueryExecutorRunStatus,
    QueryExecutorStage,
    QueryInput,
    RunStatus,
)
from pmmc.schemas.answer_output import AnswerContractV3, answer_response_schema
from pmmc.schemas.base import utc_now

QueryWorkflowProgress = Callable[[dict[str, int]], None]


@dataclass(frozen=True)
class QueryExecutorWorkflowResult:
    manifest: QueryExecutorRunManifest
    batch: QueryExecutorBatchResult
    answer_prompt_hash: str
    imported_text_batch_count: int
    imported_multimodal_batch_count: int


class QueryExecutorParentContractError(ValueError):
    """A parent-data preflight failure with content-free diagnostics."""

    def __init__(self, reason_code: str, diagnostics: dict[str, object]) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.diagnostics = dict(diagnostics)
        self.safe_summary = reason_code

    def safe_diagnostics(self) -> dict[str, object]:
        return {"reason_code": self.reason_code, **self.diagnostics}


def _parent_preflight_event(state: str, **values: object) -> None:
    print(
        "STEP7_QUERY_PARENT_PREFLIGHT "
        + json.dumps(
            {"state": state, **values},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _workflow_phase_event(phase: str, state: str, **values: object) -> None:
    print(
        "STEP7_QUERY_WORKFLOW_PHASE "
        + json.dumps(
            {"phase": phase, "state": state, **values},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _parent_contract_failure(reason_code: str, **diagnostics: object) -> NoReturn:
    safe = {"reason_code": reason_code, **diagnostics}
    _parent_preflight_event("blocked", **safe)
    raise QueryExecutorParentContractError(reason_code, diagnostics)


def safe_query_executor_error_diagnostics(exc: BaseException) -> dict[str, object]:
    """Return only explicitly safe structural details for managed-child logs."""

    if isinstance(exc, QueryExecutorParentContractError):
        return exc.safe_diagnostics()
    if isinstance(exc, QueryExecutorItemFailures):
        return {
            "reason_code": "query_executor_technical_item_failures",
            "technical_failed_count": exc.batch.technical_failed_count,
            "no_answer_count": exc.batch.failed_count - exc.batch.technical_failed_count,
            "failure_diagnostics": list(exc.batch.failure_diagnostics),
        }
    if isinstance(exc, (ExperimentPersistenceError, DBAPIError, PoolTimeoutError)):
        diagnostics: dict[str, object] = dict(safe_database_error_diagnostics(exc))
        return diagnostics
    if isinstance(exc, AttributeError):
        attribute_name = getattr(exc, "name", None)
        if (
            isinstance(attribute_name, str)
            and attribute_name.isidentifier()
            and len(attribute_name) <= 128
        ):
            return {"attribute_name": attribute_name}
    return {}


class QueryExecutorItemFailures(RuntimeError):
    def __init__(self, batch: QueryExecutorBatchResult) -> None:
        super().__init__("query executor contains terminal item failures")
        self.batch = batch


def required_coverage_query_vector_pairs(
    queries: Sequence[QueryInput],
    *,
    include_text_only_multimodal: bool = False,
) -> set[tuple[str, str]]:
    """Return the modality-aware evaluator vector denominator."""

    expected: set[tuple[str, str]] = set()
    for query in queries:
        query_id = query.query_id
        expected.add((query_id, "text_semantic"))
        if QBankRuntimeChannel.MULTIMODAL in required_query_embedding_channels(
            query,
            include_text_only_multimodal=include_text_only_multimodal,
        ):
            expected.add((query_id, "multimodal_semantic"))
    return expected


def validate_coverage_query_vectors(
    *,
    config: QueryExecutorConfig,
    coverage: QBankCoverageRunManifest | None,
    source_compiler_run_id: str,
    snapshot_id: str,
    dataset: DatasetName,
    queries: Sequence[QueryInput],
    vectors: Sequence[QBankCoverageQueryEmbedding],
    require_completed: bool = True,
) -> dict[str, object]:
    """Share one versioned parent boundary between CPU preflight and execution."""

    allowed_statuses = {QBankCoverageStatus.COMPLETED}
    if not require_completed:
        allowed_statuses.add(QBankCoverageStatus.RUNNING)
    if coverage is None or coverage.status not in allowed_statuses:
        _parent_contract_failure(
            "coverage_parent_not_completed"
            if require_completed
            else "coverage_parent_not_embedding_ready"
        )
    if (
        coverage.compiler_run_id != source_compiler_run_id
        or coverage.snapshot_id != snapshot_id
        or coverage.dataset != dataset.value
    ):
        _parent_contract_failure("coverage_parent_scope_incompatible")
    expected_protocol = "qbank-coverage-v6" if config.is_v2 else "qbank-coverage-v5"
    if coverage.protocol_version != expected_protocol:
        _parent_contract_failure(
            "coverage_parent_protocol_incompatible", expected_protocol=expected_protocol
        )
    query_ids = {query.query_id for query in queries}
    if (
        not queries
        or len(query_ids) != len(queries)
        or any(query.dataset is not dataset for query in queries)
    ):
        _parent_contract_failure(
            "runtime_query_denominator_incomplete",
            query_count=len(queries),
            unique_query_count=len(query_ids),
        )
    pairs = {(item.query_id, item.channel.value) for item in vectors}
    expected = required_coverage_query_vector_pairs(
        queries,
        include_text_only_multimodal=config.is_v2,
    )
    counts: dict[str, object] = {
        "query_count": len(queries),
        "query_image_count": sum(bool(query.query_image_refs) for query in queries),
        "expected_text_vector_count": sum(channel == "text_semantic" for _, channel in expected),
        "actual_text_vector_count": sum(channel == "text_semantic" for _, channel in pairs),
        "expected_multimodal_vector_count": sum(
            channel == "multimodal_semantic" for _, channel in expected
        ),
        "actual_multimodal_vector_count": sum(
            channel == "multimodal_semantic" for _, channel in pairs
        ),
        "coverage_protocol": expected_protocol,
        "vector_policy": "all_queries_text_and_mm" if config.is_v2 else "query_image_aware",
    }
    duplicate_count = len(vectors) - len(pairs)
    foreign_count = sum(item.coverage_run_id != coverage.coverage_run_id for item in vectors)
    if pairs != expected or duplicate_count or foreign_count:
        _parent_contract_failure(
            "coverage_query_vector_denominator_incomplete",
            **counts,
            missing_pair_count=len(expected - pairs),
            unexpected_pair_count=len(pairs - expected),
            duplicate_pair_count=duplicate_count,
            foreign_coverage_vector_count=foreign_count,
        )
    return counts


async def preflight_query_coverage_vectors(
    *,
    settings: Settings,
    config: QueryExecutorConfig,
    coverage_run_id: str,
    source_compiler_run_id: str,
    snapshot_id: str,
    dataset: DatasetName,
    query_ids: Sequence[str],
) -> dict[str, object]:
    """Read-only boundary check before loading the post-index chat weights."""

    if config.is_v2:
        _parent_preflight_event(
            "ready",
            phase="reranker_contract_before_post_index_chat",
            dataset=dataset.value,
            **preflight_reranker_contract(),
        )
    runtime_engine, runtime_sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    evaluation_engine, evaluation_sessions = create_session_factory(
        settings,
        DatabaseRole.EVALUATOR,
    )
    try:
        queries = await PostgresRuntimeQueryRepository(runtime_sessions).list_ids(
            snapshot_id,
            list(query_ids),
        )
        if len(query_ids) != len(set(query_ids)) or {q.query_id for q in queries} != set(query_ids):
            _parent_contract_failure(
                "runtime_query_denominator_incomplete",
                expected_query_count=len(query_ids),
                actual_query_count=len(queries),
            )
        async with ExperimentUnitOfWork(evaluation_sessions, DataScope.EVALUATION) as uow:
            assert uow.qbank_coverage is not None
            coverage = await uow.qbank_coverage.get_run(coverage_run_id)
            vectors = await uow.qbank_coverage.list_query_embeddings(coverage_run_id)
        counts = validate_coverage_query_vectors(
            config=config,
            coverage=coverage,
            source_compiler_run_id=source_compiler_run_id,
            snapshot_id=snapshot_id,
            dataset=dataset,
            queries=queries,
            vectors=vectors,
            require_completed=False,
        )
        _parent_preflight_event(
            "ready", phase="before_post_index_chat", dataset=dataset.value, model_calls=0, **counts
        )
        return counts
    finally:
        await runtime_engine.dispose()
        await evaluation_engine.dispose()


def _prompt(config: QueryExecutorConfig, settings: Settings) -> tuple[str, str]:
    value = (
        config.resolved_answer_prompt_path(settings.project_root)
        .read_text(encoding="utf-8")
        .strip()
    )
    if not value:
        raise ValueError("Step 7 answer prompt cannot be empty")
    return value, hashlib.sha256(value.encode("utf-8")).hexdigest()


async def create_query_executor_manifest(
    *,
    settings: Settings,
    config: QueryExecutorConfig,
    source_compiler_run_id: str,
    coverage_run_id: str | None,
    fallback_run_id: str,
    query_ids: Sequence[str],
    pilot_projection: bool,
    run_id: str | None = None,
    answer_model_alias: str = "answer-qwen3.5-9b",
    allow_pending_fallback: bool = False,
    source_match_run_id: str | None = None,
    primary_only: bool = False,
) -> QueryExecutorRunManifest:
    """Validate immutable parents and construct one content-addressed run manifest."""

    ordered_ids = list(query_ids)
    if allow_pending_fallback and not config.is_v3:
        raise ValueError("pending fallback preparation requires v3")
    if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
        _parent_contract_failure(
            "query_denominator_invalid",
            query_count=len(ordered_ids),
            unique_query_count=len(set(ordered_ids)),
        )
    prompt_text, prompt_hash = _prompt(config, settings)
    del prompt_text
    config_hash = query_executor_config_hash(config)
    runtime_engine, runtime_sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    evaluation_engine, evaluation_sessions = (
        (None, None)
        if config.is_v3
        else create_session_factory(
            settings,
            DatabaseRole.EVALUATOR,
        )
    )
    try:
        async with ExperimentUnitOfWork(runtime_sessions, DataScope.RUNTIME) as uow:
            assert uow.compiler_runs is not None
            assert uow.doubter is not None
            assert uow.runs is not None
            assert uow.answers is not None
            compiler = await uow.compiler_runs.get(source_compiler_run_id)
            routing = await uow.doubter.get_routing_projection(source_compiler_run_id)
            fallback_manifest = await uow.runs.get_manifest(fallback_run_id)
            fallback_state = await uow.runs.get_state(fallback_run_id)
            if config.is_v3:
                assert uow.query_executor is not None
                fallback_answers = [
                    await uow.query_executor.get_typed_answer_for_variant(fallback_run_id, query_id)
                    for query_id in ordered_ids
                ]
            else:
                fallback_answers = [
                    await uow.answers.get_for_query(fallback_run_id, query_id, "primary")
                    for query_id in ordered_ids
                ]
            fallback_abstentions = [None] * len(ordered_ids)
            if config.is_v2:
                assert uow.query_executor is not None
                fallback_abstentions = [
                    await uow.query_executor.get_abstention(fallback_run_id, query_id)
                    for query_id in ordered_ids
                ]
        if compiler is None:
            _parent_contract_failure("source_compiler_missing")
        expected_compiler_schema = "0.9.0" if config.is_v3 else "0.8.0" if config.is_v2 else "0.7.0"
        expected_operator_set = (
            "pmmc-operators-v7"
            if config.is_v3
            else "pmmc-operators-v6"
            if config.is_v2
            else "pmmc-operators-v5"
        )
        expected_projection_schema = (
            "0.4.0" if config.is_v3 else "0.3.0" if config.is_v2 else "0.2.0"
        )
        if (
            compiler.compiler_schema_version != expected_compiler_schema
            or compiler.target_stage is not CompilationTargetStage.VERIFIED_QBANK
            or compiler.operator_set_version != expected_operator_set
            or compiler.active_round != 2
            or compiler.max_round != 3
        ):
            _parent_contract_failure("source_compiler_identity_incompatible")
        async with ExperimentUnitOfWork(runtime_sessions, DataScope.RUNTIME) as uow:
            assert uow.runs is not None
            compiler_state = await uow.runs.get_state(source_compiler_run_id)
        if compiler_state.status is not RunStatus.COMPLETED:
            _parent_contract_failure("source_compiler_not_completed")
        if routing is None:
            _parent_contract_failure("routing_projection_missing")
        projection, members = routing
        if (
            projection.run_id != source_compiler_run_id
            or projection.snapshot_id != compiler.snapshot_id
            or projection.schema_version != expected_projection_schema
            or projection.active_round != 2
            or projection.max_round != 3
            or projection.member_count != len(members)
            or (not config.is_v3 and projection.answerable_count <= 0)
        ):
            _parent_contract_failure(
                "routing_projection_incompatible",
                routing_member_count=len(members),
                projected_member_count=projection.member_count,
                answerable_count=projection.answerable_count,
            )

        queries = await PostgresRuntimeQueryRepository(runtime_sessions).list_ids(
            compiler.snapshot_id, ordered_ids
        )
        if {query.query_id for query in queries} != set(ordered_ids):
            _parent_contract_failure(
                "runtime_query_denominator_incomplete",
                expected_query_count=len(ordered_ids),
                actual_query_count=len(queries),
            )
        datasets = {query.dataset for query in queries}
        if len(datasets) != 1:
            _parent_contract_failure(
                "runtime_query_dataset_boundary_crossed",
                dataset_count=len(datasets),
            )
        dataset = next(iter(datasets))
        if dataset is DatasetName.UNKNOWN:
            _parent_contract_failure("runtime_query_dataset_unknown")

        if config.is_v3:
            vector_counts = {
                "actual_text_vector_count": 0,
                "actual_multimodal_vector_count": 0,
                "vector_policy": "runtime-owned-v3",
            }
        else:
            assert evaluation_sessions is not None and coverage_run_id is not None
            async with ExperimentUnitOfWork(evaluation_sessions, DataScope.EVALUATION) as uow:
                assert uow.qbank_coverage is not None
                coverage = await uow.qbank_coverage.get_run(coverage_run_id)
                vectors = await uow.qbank_coverage.list_query_embeddings(coverage_run_id)
            vector_counts = validate_coverage_query_vectors(
                config=config,
                coverage=coverage,
                source_compiler_run_id=source_compiler_run_id,
                snapshot_id=compiler.snapshot_id,
                dataset=dataset,
                queries=queries,
                vectors=vectors,
            )
        query_image_count = sum(bool(query.query_image_refs) for query in queries)

        if fallback_manifest is None:
            _parent_contract_failure("fallback_parent_missing")
        if (
            config.is_v3
            and config.treatment == "no_raw_image"
            and fallback_manifest.metadata.get("allow_memory_raw_images") is not False
        ):
            _parent_contract_failure("caption_only_fallback_required")
        if config.is_v3 and fallback_manifest.answer_model_alias != answer_model_alias:
            _parent_contract_failure("fallback_answer_backbone_incompatible")
        if (
            fallback_manifest.method != BaselineMethod.NATIVE_MULTIMODAL_RAG.value
            or fallback_manifest.dataset != dataset
            or fallback_manifest.dataset_snapshot_ids != [compiler.snapshot_id]
            or (not allow_pending_fallback and fallback_state.status is not RunStatus.COMPLETED)
            or (
                not allow_pending_fallback
                and any(
                    (answer is None) == (abstention is None)
                    for answer, abstention in zip(
                        fallback_answers, fallback_abstentions, strict=True
                    )
                )
            )
        ):
            _parent_contract_failure(
                "fallback_parent_incomplete_or_incompatible",
                query_count=len(ordered_ids),
                fallback_answer_count=sum(answer is not None for answer in fallback_answers),
                fallback_run_completed=fallback_state.status is RunStatus.COMPLETED,
            )
        if config.is_v3:
            if fallback_manifest.metadata.get(
                "answer_contract_version"
            ) != config.answer_contract_version or (
                not allow_pending_fallback
                and any(
                    answer is None
                    or answer.contract_version != config.answer_contract_version
                    or answer.terminal_state == "fallback_required"
                    for answer in fallback_answers
                )
            ):
                _parent_contract_failure("fallback_typed_answer_contract_incompatible")
        elif config.is_v2:
            if config.answer_prompt_version == "pmmc-answer-contract-v2.1":
                from pmmc.runtime.answer_policy import CURRENT_PROMPT_HASH

                if (
                    fallback_manifest.metadata.get("answer_prompt_version")
                    != config.answer_prompt_version
                    or fallback_manifest.prompt_hashes.get(config.answer_prompt_version)
                    != CURRENT_PROMPT_HASH
                ):
                    _parent_contract_failure("fallback_answer_prompt_identity_incompatible")
            failure_counts: Counter[str] = Counter()
            for answer in fallback_answers:
                if answer is None:
                    continue
                failure_counts.update(
                    item.value for item in strict_final_answer_failure_codes(answer)
                )
            if (
                fallback_manifest.metadata.get("answer_contract_version")
                != config.answer_contract_version
                or failure_counts
            ):
                _parent_contract_failure(
                    "fallback_answer_contract_incompatible",
                    contract_identity_matches=(
                        fallback_manifest.metadata.get("answer_contract_version")
                        == config.answer_contract_version
                    ),
                    invalid_answer_count=sum(
                        bool(strict_final_answer_failure_codes(answer))
                        for answer in fallback_answers
                        if answer is not None
                    ),
                    failure_code_counts=dict(sorted(failure_counts.items())),
                )

        _parent_preflight_event(
            "ready",
            dataset=dataset.value,
            query_count=len(queries),
            query_image_count=query_image_count,
            routing_member_count=len(members),
            coverage_text_vector_count=vector_counts["actual_text_vector_count"],
            coverage_multimodal_vector_count=vector_counts["actual_multimodal_vector_count"],
            vector_policy=vector_counts["vector_policy"],
            fallback_answer_count=sum(answer is not None for answer in fallback_answers),
            fallback_abstention_count=sum(value is not None for value in fallback_abstentions),
        )

        payload = {
            "source_compiler_run_id": source_compiler_run_id,
            "routing_projection_id": projection.routing_projection_id,
            "snapshot_id": compiler.snapshot_id,
            "dataset": dataset.value,
            "coverage_run_id": None if config.is_v3 else coverage_run_id,
            "fallback_run_id": fallback_run_id,
            "query_ids": ordered_ids,
            "config_hash": config_hash,
            "answer_model_alias": answer_model_alias,
            "answer_prompt_hash": prompt_hash,
            "pilot_projection": pilot_projection,
            **({"source_match_run_id": source_match_run_id} if config.is_v3 else {}),
        }
        if primary_only:
            if not pilot_projection or not config.is_v3 or config.treatment != "main":
                raise ValueError("primary exploration cannot replace the approved paper grid")
            payload["readout"] = "exploration-primary-b8-r2"
        full_grid = (not primary_only and config.treatment == "main"
                     and answer_model_alias == "answer-qwen3.5-9b")
        if config.uses_reranker:
            payload["reranker_contract_hash"] = reranker_contract_hash()
        resolved_run_id = run_id or stable_compiler_id("pmmc-query-executor", payload)
        content_payload = {"query_executor_run_id": resolved_run_id, **payload}
        return QueryExecutorRunManifest(
            query_executor_run_id=resolved_run_id,
            source_compiler_run_id=source_compiler_run_id,
            routing_projection_id=projection.routing_projection_id,
            snapshot_id=compiler.snapshot_id,
            dataset=dataset,
            schema_version="0.3.0" if config.is_v3 else "0.2.0" if config.is_v2 else "0.1.0",
            config_hash=config_hash,
            active_round=config.active_round,
            max_round=max(config.rounds),
            question_budget=config.question_budget,
            selected_query_count=len(ordered_ids),
            pilot_projection=pilot_projection,
            paper_result=not pilot_projection,
            content_hash=canonical_content_hash(content_payload),
            metadata={
                "coverage_run_id": None if config.is_v3 else coverage_run_id,
                **({"readout": payload["readout"]} if primary_only else {}),
                **({"embedding_ownership": "runtime"} if config.is_v3 else {}),
                **({"joint_relevance_policy": config.joint_relevance_policy,
                    "joint_relevance_threshold": config.joint_relevance_threshold,
                    "joint_relevance_enabled": config.joint_relevance_enabled}
                   if config.joint_relevance_policy is not None else {}),
                **(
                    {
                        "source_match_run_id": source_match_run_id,
                        "treatment": config.treatment,
                        "execute_full_grid": full_grid,
                        "expected_cell_count": 48 if full_grid else 3,
                        "declared_cells": [
                            [budget, round_id]
                            for budget in config.question_budgets
                            for round_id in config.rounds
                        ]
                        if full_grid
                        else [[8, 2]],
                    }
                    if config.is_v3
                    else {}
                ),
                "fallback_run_id": fallback_run_id,
                "query_ids_hash": canonical_content_hash(ordered_ids),
                "answer_model_alias": answer_model_alias,
                "answer_prompt_version": config.answer_prompt_version,
                "answer_prompt_hash": prompt_hash,
                "match_threshold": config.route_match_threshold,
                "gap_threshold": config.route_gap_threshold,
                "query_executor_protocol_hash": query_executor_protocol_hash(config),
                "routing_policy_hash": query_executor_routing_policy_hash(
                    config,
                    match_threshold=config.route_match_threshold,
                    gap_threshold=config.route_gap_threshold,
                ),
                "source_compiler_schema_version": compiler.compiler_schema_version,
                "source_operator_set_version": compiler.operator_set_version,
                "routing_projection_schema_version": projection.schema_version,
                "routing_projection_hash": projection.content_hash,
                "matcher_version": config.matcher_version,
                "graph_version": config.graph_version,
                "sufficiency_policy_version": config.sufficiency_policy_version,
                "reranker_version": config.reranker_version,
                **(
                    {"reranker_contract_hash": reranker_contract_hash()}
                    if config.uses_reranker
                    else {}
                ),
                "reranker_prompt_version": config.reranker_prompt_version,
                "reranker_model_alias": config.reranker_model_alias,
                "reranker_schema_hash": (
                    canonical_content_hash(QBankRerankerOutput.model_json_schema())
                    if config.uses_reranker
                    else None
                ),
                "answer_contract_version": config.answer_contract_version,
                "answer_output_schema_version": config.answer_output_schema_version,
                "answer_output_schema_hash": (
                    canonical_content_hash(AnswerContractV3.model_json_schema())
                    if config.is_v3
                    else canonical_content_hash(answer_response_schema())
                    if config.is_v2
                    else None
                ),
            },
        )
    finally:
        await runtime_engine.dispose()
        if evaluation_engine is not None:
            await evaluation_engine.dispose()


def _runtime_embedding_cache_identity(snapshot_id, query, channel, entry):
    payload = {
        "cache_schema": "runtime-query-cache-v2",
        "snapshot_id": snapshot_id,
        "query_id": query.query_id,
        "query_text": query.raw_query,
        "query_image_hashes": [item.sha256 or item.image_id for item in query.query_image_refs]
        if channel is QBankRuntimeChannel.MULTIMODAL
        else [],
        "channel": channel.value,
        "model_entry_hash": entry.resolved_hash,
        "instruction_version": QUERY_MATCH_INSTRUCTION_VERSION,
    }
    input_hash = canonical_content_hash(payload)
    return stable_compiler_id("runtime-query-vector-cache", input_hash), input_hash


async def _validated_runtime_cache(
    sessions: async_sessionmaker[AsyncSession], *, cache_id: str,
    snapshot_id: str, query: QueryInput, channel: QBankRuntimeChannel,
    model_entry_hash: str, input_hash: str,
) -> dict[str, Any] | None:
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.query_executor is not None and uow.embeddings is not None
        assert uow.runs is not None
        cached = await uow.query_executor.get_runtime_embedding_cache(cache_id)
        if cached is None:
            return None
        expected = {
            "cache_id": cache_id, "snapshot_id": snapshot_id, "query_id": query.query_id,
            "channel": channel.value, "model_entry_hash": model_entry_hash,
            "input_hash": input_hash,
        }
        if any(cached[key] != value for key, value in expected.items()):
            raise ValueError("runtime query cache identity mismatch")
        payload = {**expected, **{key: cached[key] for key in (
            "embedding_id", "collection_id", "model_call_id",
        )}}
        if cached["content_hash"] != canonical_content_hash(payload):
            raise ValueError("runtime query cache hash mismatch")
        collection = await uow.embeddings.get_collection(cached["collection_id"])
        record = await uow.embeddings.get_record(cached["embedding_id"])
        owner_id = stable_compiler_id("runtime-query-cache-owner", cache_id)
        state = await uow.runs.get_state(owner_id)
        if (
            collection is None or collection.status != "ready"
            or collection.run_id != owner_id or collection.snapshot_id != snapshot_id
            or state.status is not RunStatus.COMPLETED
            or record is None or record.collection_id != collection.collection_id
            or record.source_id != query.query_id or record.input_hash != input_hash
            or record.model_call_id != cached["model_call_id"]
            or record.dimension != 4096 or len(record.vector) != 4096
        ):
            raise ValueError("runtime query cache lineage is incomplete")
        return cached


async def prepare_query_executor_runtime_embeddings(
    *,
    settings: Settings,
    registry: ModelRegistry,
    source_compiler_run_id: str,
    snapshot_id: str,
    query_ids: Sequence[str],
    channel: QBankRuntimeChannel | str,
    joint_evidence: bool = False,
) -> dict[str, object]:
    """Pre-D runtime cache. Requires neither routing nor fallback/answer completion."""
    channel = QBankRuntimeChannel(channel)
    if joint_evidence and channel is not QBankRuntimeChannel.MULTIMODAL:
        raise ValueError("release R joint embeddings require the native multimodal phase")
    if channel is QBankRuntimeChannel.LEXICAL:
        raise ValueError("lexical matching has no vector cache")
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("runtime embedding query denominator must be nonempty and unique")
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    reader_engine, reader_sessions = create_session_factory(settings, DatabaseRole.ONLINE)
    try:
        queries = await PostgresRuntimeQueryRepository(reader_sessions).list_ids(
            snapshot_id, list(query_ids)
        )
        if {query.query_id for query in queries} != set(query_ids):
            raise ValueError("runtime embedding queries escaped the requested snapshot")
        alias = (
            "qwen3-embedding-8b" if channel is QBankRuntimeChannel.TEXT else "qwen3-vl-embedding-8b"
        )
        entry = registry.get(alias)
        store = create_runtime_object_store(settings)
        artifacts = ExperimentArtifactService(
            sessions, store, DataScope.RUNTIME, artifact_prefix=settings.experiment_artifact_prefix
        )
        execution = create_model_execution_service(
            settings,
            registry,
            artifact_service=artifacts,
            model_calls=TransactionalModelCallWriter(sessions),
        )
        builder = QueryEmbeddingWorkflow(
            sessions=sessions, reader_sessions=reader_sessions, execution=execution,
            registry=registry,
        )
        call_count = reused_count = 0
        gateway = None
        for query in sorted(queries, key=lambda item: item.query_id):
            cache_id, input_hash = _runtime_embedding_cache_identity(
                snapshot_id, query, channel, entry
            )
            cache_scope = dict(
                cache_id=cache_id, snapshot_id=snapshot_id, query=query, channel=channel,
                model_entry_hash=entry.resolved_hash, input_hash=input_hash,
            )
            if await _validated_runtime_cache(reader_sessions, **cache_scope) is not None:
                reused_count += 1
                continue
            async with runtime_query_cache_lease(
                sessions, cache_id=cache_id, snapshot_id=snapshot_id, dataset=query.dataset,
                model_alias=alias, lease_seconds=settings.experiment_job_lease_seconds,
            ) as lease:
                if lease is None:
                    if await _validated_runtime_cache(reader_sessions, **cache_scope) is None:
                        raise RuntimeError("completed runtime cache owner has no cache fact")
                    reused_count += 1
                    continue
                owner = lease.owner
                collection = builder._collection(owner, channel, entry)
                await builder._ensure_collection(collection)
                if gateway is None:
                    gateway = create_embedding_gateway(
                        settings, registry, alias,
                        image_url_resolver=create_image_url_resolver(settings, store),
                    )
                request = EmbeddingRequest(
                    request_id=stable_compiler_id("runtime-query-cache-batch", cache_id),
                    model_alias=alias, dimensions=4096,
                    inputs=[EmbeddingInput(
                        input_id=query.query_id, text=QUERY_MATCH_INSTRUCTION + query.raw_query,
                        image_refs=list(query.query_image_refs)
                        if channel is QBankRuntimeChannel.MULTIMODAL else [],
                    )],
                    metadata={
                        "scope": "runtime", "persist_normalized_embedding_result": True,
                        "instruction_version": QUERY_MATCH_INSTRUCTION_VERSION,
                    },
                )
                result, call_id, recovered = await builder._load_or_call(
                    manifest=owner, request=request, gateway=gateway,
                )
                if len(result.vectors) != 1 or result.vectors[0].input_id != query.query_id:
                    raise ValueError("runtime embedding cache response denominator mismatch")
                vector = normalize_vector(result.vectors[0].vector)
                if len(vector) != 4096:
                    raise ValueError("runtime embedding cache response dimension mismatch")
                call_count += not recovered
                async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                    assert uow.embeddings is not None and uow.query_executor is not None
                    assert uow.jobs is not None and uow.runs is not None
                    await uow.jobs.heartbeat(lease.job_id, lease.worker_id, lease.lease_seconds)
                    record = EmbeddingRecord(
                        embedding_id=stable_compiler_id("runtime-cached-query-embedding", cache_id),
                        collection_id=collection.collection_id,
                        source_id=query.query_id,
                        input_hash=input_hash,
                        dimension=4096,
                        vector=vector,
                        model_call_id=call_id,
                        metadata={"query_id": query.query_id, "channel": channel.value},
                    )
                    await uow.embeddings.add_records([record])
                    payload = {
                        "cache_id": cache_id,
                        "snapshot_id": snapshot_id,
                        "query_id": query.query_id,
                        "channel": channel.value,
                        "model_entry_hash": entry.resolved_hash,
                        "input_hash": input_hash,
                        "embedding_id": record.embedding_id,
                        "collection_id": collection.collection_id,
                        "model_call_id": call_id,
                    }
                    await uow.query_executor.add_runtime_embedding_cache(
                        {
                            **payload,
                            "content_hash": canonical_content_hash(payload),
                            "created_at": utc_now(),
                        }
                    )
                    await uow.embeddings.mark_collection_ready(
                        collection.collection_id, index_type="exact",
                        index_params={"ann": False, "distance_metric": "cosine"},
                    )
                    await uow.jobs.finish(lease.job_id, lease.worker_id, success=True)
                    await uow.runs.transition(owner.query_executor_run_id, RunStatus.COMPLETED)
        joint_counts: Counter[str] = Counter()
        joint_producers: dict[str, dict[str, str]] = {}
        if joint_evidence:
            def native_gateway():
                nonlocal gateway
                if gateway is None:
                    gateway = create_embedding_gateway(
                        settings, registry, alias,
                        image_url_resolver=create_image_url_resolver(settings, store),
                    )
                return gateway

            for query in queries:
                counts, producers = await prepare_joint_shards(
                    builder=builder, snapshot_id=snapshot_id, query=query,
                    reader=PostgresMemoryRepository(reader_sessions), entry=entry,
                    gateway=native_gateway, lease_seconds=settings.experiment_job_lease_seconds,
                )
                joint_counts.update(counts)
                joint_producers.update(producers)
        return {
            "channel": channel.value,
            "query_count": len(queries),
            "model_call_count": call_count + joint_counts["joint_model_call_count"],
            "reused_query_count": reused_count,
            **joint_counts,
            **({"joint_producers": joint_producers, "raw_query_model_call_count": call_count}
               if joint_evidence else {}),
        }
    finally:
        await engine.dispose()
        await reader_engine.dispose()


async def bind_query_executor_runtime_embeddings(
    *, sessions, registry, manifest, query_ids, reader_sessions=None,
):
    """Bind immutable runtime cache references after exact routing becomes available."""
    readers = reader_sessions if reader_sessions is not None else sessions
    queries = await PostgresRuntimeQueryRepository(readers).list_ids(
        manifest.snapshot_id, list(query_ids)
    )
    if {query.query_id for query in queries} != set(query_ids):
        raise ValueError("runtime embedding binding has an incomplete query denominator")
    for channel in (QBankRuntimeChannel.TEXT, QBankRuntimeChannel.MULTIMODAL):
        entry = registry.get(
            "qwen3-embedding-8b" if channel is QBankRuntimeChannel.TEXT else "qwen3-vl-embedding-8b"
        )
        for ordinal, query in enumerate(sorted(queries, key=lambda item: item.query_id)):
            cache_id, input_hash = _runtime_embedding_cache_identity(
                manifest.snapshot_id, query, channel, entry
            )
            cached = await _validated_runtime_cache(
                readers, cache_id=cache_id, snapshot_id=manifest.snapshot_id, query=query,
                channel=channel, model_entry_hash=entry.resolved_hash, input_hash=input_hash,
            )
            if cached is None:
                raise RuntimeError("runtime query cache incomplete; schedule embedding phase first")
            async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                assert uow.query_executor is not None
                batch_id = stable_compiler_id(
                    "runtime-query-cache-binding", [manifest.query_executor_run_id, cache_id]
                )
                batch = QueryEmbeddingBatch(
                    query_embedding_batch_id=batch_id,
                    query_executor_run_id=manifest.query_executor_run_id,
                    snapshot_id=manifest.snapshot_id,
                    model_call_id=cached["model_call_id"],
                    channel=channel,
                    batch_ordinal=ordinal,
                    query_ids=[query.query_id],
                    expected_count=1,
                    committed_count=1,
                    content_hash=canonical_content_hash([batch_id, cached["content_hash"]]),
                    metadata={"runtime_cache_id": cache_id, "runtime_owned_reuse": True},
                )
                fact = QueryEmbeddingFact(
                    query_embedding_id=stable_compiler_id("runtime-query-cache-fact", batch_id),
                    query_embedding_batch_id=batch_id,
                    query_executor_run_id=manifest.query_executor_run_id,
                    snapshot_id=manifest.snapshot_id,
                    query_id=query.query_id,
                    channel=channel,
                    embedding_id=cached["embedding_id"],
                    collection_id=cached["collection_id"],
                    vector_dimension=4096,
                    input_hash=input_hash,
                    content_hash=canonical_content_hash([batch_id, cache_id]),
                    metadata={"runtime_cache_id": cache_id},
                )
                await uow.query_executor.add_embedding_batch(batch)
                await uow.query_executor.add_embedding(fact)


async def run_query_executor_workflow(
    *,
    settings: Settings,
    registry: ModelRegistry,
    config: QueryExecutorConfig,
    foundation_config_path: Path,
    source_compiler_run_id: str,
    coverage_run_id: str | None,
    fallback_run_id: str,
    query_ids: Sequence[str],
    pilot_projection: bool,
    run_id: str | None = None,
    answer_model_alias: str = "answer-qwen3.5-9b",
    on_progress: QueryWorkflowProgress | None = None,
    phase: str = "select",
    source_match_run_id: str | None = None,
    primary_only: bool = False,
) -> QueryExecutorWorkflowResult:
    """Run or resume query matching, frozen programs, PMMC answers, and fallback."""

    if config.is_v3 and phase not in {"match", "answer", "select"}:
        raise ValueError("unsupported v3 workflow phase; expected match, answer or select")
    manifest = await create_query_executor_manifest(
        settings=settings,
        config=config,
        source_compiler_run_id=source_compiler_run_id,
        coverage_run_id=coverage_run_id,
        fallback_run_id=fallback_run_id,
        query_ids=query_ids,
        pilot_projection=pilot_projection,
        run_id=run_id,
        answer_model_alias=answer_model_alias,
        allow_pending_fallback=config.is_v3 and phase != "select",
        source_match_run_id=source_match_run_id,
        primary_only=primary_only,
    )
    _workflow_phase_event("manifest", "ready", query_count=len(query_ids))
    _workflow_phase_event("runtime_dependencies", "starting")
    answer_entry = registry.get(answer_model_alias)
    answer_entry.require(ModelCapability.CHAT_MULTIMODAL)
    if config.reranker_enabled:
        assert config.reranker_model_alias is not None
        registry.get(config.reranker_model_alias).require(ModelCapability.CHAT_MULTIMODAL)
    prompt, prompt_hash = _prompt(config, settings)
    foundation = load_compiler_config(settings.resolve_path(foundation_config_path))
    runtime_engine, runtime_sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    reader_engine, reader_sessions = (
        create_session_factory(settings, DatabaseRole.ONLINE)
        if config.is_v3 else (None, runtime_sessions)
    )
    evaluation_engine, evaluation_sessions = (
        (None, None)
        if config.is_v3
        else create_session_factory(
            settings,
            DatabaseRole.EVALUATOR,
        )
    )
    try:
        store = create_runtime_object_store(settings)
        artifacts = ExperimentArtifactService(
            runtime_sessions,
            store,
            DataScope.RUNTIME,
            artifact_prefix=settings.experiment_artifact_prefix,
        )
        execution = create_model_execution_service(
            settings,
            registry,
            artifact_service=artifacts,
            model_calls=TransactionalModelCallWriter(runtime_sessions),
        )
        resolver = create_image_url_resolver(settings, store)
        answer_inputs = PmmcEvidenceAnswerInputProvider(
            memories=PostgresMemoryRepository(reader_sessions),
            images=PostgresImageRepository(reader_sessions),
            counter=create_token_counter(settings, registry, answer_model_alias),
            instruction=prompt,
            answer_model_alias=answer_model_alias,
            total_context_tokens=32768,
            reserved_output_tokens=config.max_answer_tokens,
            timeout_seconds=settings.model_request_timeout_seconds,
            paper_mode=settings.mode.value == "paper",
            runtime_bucket=settings.object_store_bucket,
            answer_prompt_version=config.answer_prompt_version,
            answer_contract_version=config.answer_contract_version,
            allow_memory_raw_images=not (config.is_v3 and config.treatment == "no_raw_image"),
        )
        backend = PostgresOnlineQueryExecutorBackend(
            sessions=runtime_sessions,
            reader_sessions=reader_sessions,
            manifest=manifest,
            config=config,
            foundation=foundation,
            joint_entry=registry.get(JOINT_MODEL) if config.joint_relevance_enabled else None,
            answer_inputs=answer_inputs,
            answer_service=SharedAnswerGenerationService(
                execution=execution,
                grounding=(
                    configured_grounding_service(
                        settings=settings,
                        execution=execution,
                        image_url_resolver=resolver,
                    )
                    if config.is_v3
                    else None
                ),
                gateway=None if config.is_v3 and phase != "answer" else create_generation_gateway(
                    settings,
                    registry,
                    answer_model_alias,
                    image_url_resolver=resolver,
                ),
            ),
            fallback=PostgresNativeMultimodalFallbackProvider(
                reader_sessions,
                fallback_run_id=fallback_run_id,
                required_answer_contract_version=(
                    config.answer_contract_version if config.uses_reranker else None
                ),
            ),
            reranker=(
                BoundedQBankReranker(
                    execution=execution,
                    gateway=create_generation_gateway(
                        settings,
                        registry,
                        config.reranker_model_alias,
                        image_url_resolver=resolver,
                    ),
                    model_alias=config.reranker_model_alias,
                    concurrency=config.model_concurrency,
                )
                if config.reranker_enabled and config.reranker_model_alias is not None
                else None
            ),
        )
        runner = QueryExecutorBatchRunner(
            sessions=runtime_sessions,
            executor=OnlineQueryExecutor(backend=backend, config=config),
            concurrency=config.executor_workers,
            lease_seconds=settings.experiment_job_lease_seconds,
            phase=QueryExecutorStage(phase),
        )
        _workflow_phase_event("runtime_dependencies", "ready")
        _workflow_phase_event("runtime_registration", "starting")
        state = await runner.prepare(manifest, query_ids)
        _workflow_phase_event(
            "runtime_registration",
            "ready",
            run_status=state.status.value,
        )
        if state.status is QueryExecutorRunStatus.COMPLETED:
            _workflow_phase_event("query_execution", "reusing_terminal_run")
            batch = await runner.run(
                manifest,
                query_ids,
                on_progress=on_progress,
                prepared_state=state,
            )
            return QueryExecutorWorkflowResult(manifest, batch, prompt_hash, 0, 0)

        if config.is_v3:
            # Vectors are bound from the runtime-owned pre-D cache. No embedding
            # gateway may be called while an answer/reranker backbone is resident.
            if phase == "match":
                await bind_query_executor_runtime_embeddings(
                    sessions=runtime_sessions,
                    reader_sessions=reader_sessions,
                    registry=registry,
                    manifest=manifest,
                    query_ids=query_ids,
                )
            batch = await runner.run(
                manifest, query_ids, on_progress=on_progress, prepared_state=state
            )
            if batch.technical_failed_count:
                raise QueryExecutorItemFailures(batch)
            return QueryExecutorWorkflowResult(manifest, batch, prompt_hash, 0, 0)

        assert evaluation_sessions is not None and coverage_run_id is not None
        importer = CoverageQueryEmbeddingImporter(
            runtime_sessions=runtime_sessions,
            evaluation_sessions=evaluation_sessions,
            registry=registry,
            include_text_only_multimodal=config.is_v2,
        )

        async def prepare_channel(channel: QBankRuntimeChannel):
            return await importer.import_channel(
                manifest,
                coverage_run_id=coverage_run_id,
                query_ids=query_ids,
                channel=channel,
            )

        _workflow_phase_event("query_embedding_import_text", "starting")
        text = await prepare_channel(QBankRuntimeChannel.TEXT)
        _workflow_phase_event(
            "query_embedding_import_text",
            "ready",
            query_count=text.query_count,
            reused_batch_count=text.reused_batch_count,
        )
        _workflow_phase_event("query_embedding_import_multimodal", "starting")
        multimodal = await prepare_channel(QBankRuntimeChannel.MULTIMODAL)
        _workflow_phase_event(
            "query_embedding_import_multimodal",
            "ready",
            query_count=multimodal.query_count,
            reused_batch_count=multimodal.reused_batch_count,
        )
        _workflow_phase_event(
            "query_execution",
            "starting",
            query_count=len(query_ids),
            prepared_run_status=state.status.value,
        )
        try:
            batch = await runner.run(
                manifest,
                query_ids,
                on_progress=on_progress,
                prepared_state=state,
            )
        except Exception as exc:
            safe = safe_query_executor_error_diagnostics(exc)
            _workflow_phase_event(
                "query_execution",
                "failed",
                error_type=type(exc).__name__,
                **{key: value for key, value in safe.items() if key != "error_type"},
            )
            raise
        phase_state = (
            "failed"
            if batch.technical_failed_count
            else "terminal_without_answers"
            if batch.failed_count
            else "ready"
        )
        _workflow_phase_event(
            "query_execution",
            phase_state,
            succeeded_count=batch.succeeded_count,
            failed_count=batch.failed_count,
            technical_failed_count=batch.technical_failed_count,
            retried_job_count=batch.resumed_count,
            failure_diagnostics=list(batch.failure_diagnostics),
        )
        if batch.technical_failed_count or (not config.uses_reranker and batch.failed_count):
            raise QueryExecutorItemFailures(batch)
        return QueryExecutorWorkflowResult(
            manifest=manifest,
            batch=batch,
            answer_prompt_hash=prompt_hash,
            imported_text_batch_count=text.reused_batch_count,
            imported_multimodal_batch_count=multimodal.reused_batch_count,
        )
    finally:
        await runtime_engine.dispose()
        if reader_engine is not None:
            await reader_engine.dispose()
        if evaluation_engine is not None:
            await evaluation_engine.dispose()


async def verify_query_executor_run(
    settings: Settings,
    run_id: str,
) -> dict[str, object]:
    """Verify terminal accounting without exposing query or answer content."""

    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
            assert uow.query_executor is not None
            row = await uow.query_executor.get_run(run_id)
            if row is None:
                raise KeyError("query executor run does not exist")
            state = await uow.query_executor.get_state(run_id)
            counts = await uow.query_executor.terminal_fact_counts(run_id)
            job_counts = await uow.query_executor.stage_status_counts(run_id, "select")
        selected = state.selected_count
        issues: list[str] = []
        if state.status is not QueryExecutorRunStatus.COMPLETED:
            issues.append("run_not_completed")
        if counts["matched_count"] != selected:
            issues.append("route_denominator_mismatch")
        if counts["fallback_answer_count"] != selected:
            issues.append("fallback_denominator_mismatch")
        if (
            str(row["schema_version"]) == "0.2.0"
            and counts["structured_fallback_answer_count"]
            + counts.get("fallback_abstention_count", 0)
            != selected
        ):
            issues.append("structured_fallback_denominator_mismatch")
        if counts["selected_answer_count"] + counts["failed_count"] != selected:
            issues.append("terminal_selection_denominator_mismatch")
        if str(row["schema_version"]) == "0.3.0":
            expected_cells = row["metadata"].get("expected_cell_count")
            if not isinstance(expected_cells, int) or counts["cell_result_count"] != (
                selected * expected_cells
            ):
                issues.append("experiment_cell_denominator_mismatch")
        if job_counts.get("succeeded", 0) != selected or any(
            job_counts.get(value, 0) for value in ("pending", "running", "failed")
        ):
            issues.append("stage_job_denominator_mismatch")
        return {
            "query_executor_run_id": run_id,
            "manifest_hash": str(row["content_hash"]),
            "status": state.status.value,
            "selected_query_count": selected,
            "fact_counts": counts,
            "job_counts": job_counts,
            "valid": not issues,
            "issues": issues,
            "content_hash": canonical_content_hash(
                {
                    "run_id": run_id,
                    "manifest_hash": str(row["content_hash"]),
                    "counts": counts,
                    "job_counts": job_counts,
                    "issues": issues,
                }
            ),
        }
    finally:
        await engine.dispose()


__all__ = [
    "QueryExecutorParentContractError",
    "QueryExecutorWorkflowResult",
    "create_query_executor_manifest",
    "prepare_query_executor_runtime_embeddings",
    "bind_query_executor_runtime_embeddings",
    "preflight_query_coverage_vectors",
    "required_coverage_query_vector_pairs",
    "run_query_executor_workflow",
    "safe_query_executor_error_diagnostics",
    "verify_query_executor_run",
    "validate_coverage_query_vectors",
]
