"""Postgres-backed Step 7 matcher, program executor, and answer persistence."""

from __future__ import annotations

import asyncio
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.config import CompilerFoundationConfig
from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.memory_index_runner import (
    ProgramQueryVectorSet,
    load_program_query_vectors,
)
from pmmc.infrastructure.database.compiler_repositories import (
    SessionScopedOperatorDataRepository,
    classify_step7b_lexical_terms,
)
from pmmc.infrastructure.database.experiment_models import EmbeddingModel
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.repositories import PostgresRuntimeQueryRepository
from pmmc.memory.operators import (
    LinearProgramExecutor,
    ProgramExecutionResult,
    ProgramValidator,
)
from pmmc.models.registry import ModelRegistryEntry
from pmmc.retrieval.operators import build_operator_registry
from pmmc.runtime.answer_context import PmmcEvidenceAnswerInputProvider
from pmmc.runtime.answering import (
    InvalidRuntimeAnswer,
    SharedAnswerGenerationService,
    recover_runtime_answer,
    strict_final_answer_failure_codes,
    validate_typed_answer_input,
)
from pmmc.runtime.joint_embeddings import joint_shards, load_joint_records, scoped_joint_pairs
from pmmc.runtime.qbank_matcher import (
    RuntimeQBankChannelHit,
)
from pmmc.runtime.qbank_reranker import (
    BoundedQBankReranker,
    BoundedRerankerPair,
    reranker_model_call_id,
)
from pmmc.runtime.query_embeddings import required_query_embedding_channels
from pmmc.runtime.query_executor import (
    FrozenProgramBinding,
    QueryExecutorGraphState,
    RuntimeEvidenceRelevance,
    query_attempt_lease,
)
from pmmc.runtime.query_executor_config import (
    JOINT_POLICY,
    JOINT_THRESHOLD,
    QueryExecutorConfig,
    query_executor_config_hash,
)
from pmmc.runtime.relevance_grid import (
    DEFERRED_GROUNDING_ROLES,
    bind_joint_context,
    bind_runtime_relation_contract,
    relevance_answer_input_provider,
    requires_independent_relation,
    score_passes,
    select_execution_evidence,
    validate_joint_outcome,
    validate_relevance_grid_points,
)
from pmmc.schemas import (
    AnswerContractFailure,
    AnswerFormatDiagnostic,
    AnswerGenerationInput,
    DataScope,
    FinalAnswerRecord,
    Modality,
    OperatorExecutionContext,
    OperatorExecutionPurpose,
    PlannerRequirementContract,
    PmmcFallbackQueryLink,
    PmmcFallbackRunLink,
    QBankLexicalTermFact,
    QBankMatchBatch,
    QBankMatchChannelCandidate,
    QBankMatchFusedCandidate,
    QBankRerankerAssessment,
    QBankRouteClass,
    QBankRuntimeChannel,
    QueryAnswerVariant,
    QueryEmbeddingFact,
    QueryEvidenceRelevanceAssessment,
    QueryEvidenceRelevanceItem,
    QueryExecutorRunManifest,
    QueryInput,
    QueryProgramBinding,
    QueryRouteDecision,
    StructuredAnswerFact,
    TraceStatus,
    VerifiedQBankRoutingItemV2,
    VerifiedQBankRoutingItemV3,
)
from pmmc.schemas.answer_output import TypedAnswerCandidate, TypedAnswerResult
from pmmc.schemas.base import utc_now
from pmmc.schemas.query_executor import (
    ExperimentCell,
    QueryExperimentCellResult,
    QueryRuntimeTypedAnswerFact,
    runtime_fact_payload,
)


class NativeMultimodalFallbackProvider(Protocol):
    """A precomputed Native MM child run used for every query counterfactual."""

    @property
    def fallback_run_id(self) -> str: ...

    async def load_or_run(self, query: QueryInput) -> FinalAnswerRecord: ...

    async def load_or_run_typed(self, query: QueryInput) -> TypedAnswerResult: ...


class PostgresNativeMultimodalFallbackProvider:
    """Load a completed Native MM RAG answer without reimplementing the baseline."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        fallback_run_id: str,
        required_answer_contract_version: str | None = None,
    ) -> None:
        self.sessions = sessions
        self._fallback_run_id = fallback_run_id
        self.required_answer_contract_version = required_answer_contract_version

    @property
    def fallback_run_id(self) -> str:
        return self._fallback_run_id

    async def load_or_run_typed(self, query: QueryInput) -> TypedAnswerResult:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.query_executor is not None
            answer = await uow.query_executor.get_typed_answer_for_variant(
                self.fallback_run_id,
                query.query_id,
                "primary",
            )
        if answer is None:
            raise KeyError("typed Native MM counterfactual is not ready")
        if (
            answer.method != "native_multimodal_rag"
            or answer.query_id != query.query_id
            or answer.contract_version != "typed-answer-contract-v3"
        ):
            raise ValueError("typed fallback has incompatible runtime lineage")
        return answer

    async def load_or_run(self, query: QueryInput) -> FinalAnswerRecord:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.answers is not None
            answer = await uow.answers.get_for_query(
                self.fallback_run_id,
                query.query_id,
                "primary",
            )
            if (
                answer is None
                and self.required_answer_contract_version == "strict-answer-contract-v2"
            ):
                from pmmc.runtime.answering import UnsupportedRuntimeAnswer

                assert uow.query_executor is not None
                fact = await uow.query_executor.get_abstention(self.fallback_run_id, query.query_id)
                if fact is not None:
                    error = UnsupportedRuntimeAnswer("Native-MM has a validated abstention")
                    error.fact = fact
                    raise error
        if answer is None:
            raise KeyError("Native MM counterfactual answer is not ready")
        if answer.method != "native_multimodal_rag" or answer.query_id != query.query_id:
            raise ValueError("fallback child answer has incompatible runtime lineage")
        if self.required_answer_contract_version is not None:
            failures = strict_final_answer_failure_codes(answer)
            if (
                answer.metadata.get("answer_contract_version")
                != self.required_answer_contract_version
                or failures
            ):
                raise ValueError(
                    "fallback child answer violates the required structured contract: "
                    + ",".join(item.value for item in failures)
                )
        return answer


def _strict_answer_projection(
    *,
    run_id: str,
    query_id: str,
    answer_variant: str,
    answer: FinalAnswerRecord,
) -> tuple[list[AnswerFormatDiagnostic], StructuredAnswerFact]:
    """Project one accepted source answer into the Step 7B runtime contract."""

    failures = strict_final_answer_failure_codes(answer)
    if failures:
        raise ValueError(
            "accepted answer violates the strict persisted contract: "
            + ",".join(item.value for item in failures)
        )
    raw_attempts = answer.metadata.get("format_failure_codes", [])
    if not isinstance(raw_attempts, list) or len(raw_attempts) > 1:
        raise ValueError("strict answer format retry lineage is malformed")
    diagnostics: list[AnswerFormatDiagnostic] = []
    for attempt_ordinal, raw_codes in enumerate(raw_attempts):
        if not isinstance(raw_codes, list):
            raise ValueError("strict answer failure-code lineage is malformed")
        try:
            codes = [AnswerContractFailure(str(value)) for value in raw_codes]
        except ValueError as exc:
            raise ValueError("strict answer contains an unknown failure code") from exc
        payload = {
            "run_id": run_id,
            "query_id": query_id,
            "variant": answer_variant,
            "attempt": attempt_ordinal,
            "failure_codes": [item.value for item in codes],
        }
        diagnostics.append(
            AnswerFormatDiagnostic(
                diagnostic_id=stable_compiler_id("answer-format-diagnostic", payload),
                run_id=run_id,
                query_id=query_id,
                answer_variant=answer_variant,
                attempt_ordinal=attempt_ordinal,
                failure_codes=codes,
                provider_schema_conformant=False,
                accepted=False,
                content_hash=canonical_content_hash(payload),
                metadata={"source_answer_id": answer.answer_id},
            )
        )
    accepted_ordinal = len(raw_attempts)
    accepted_payload = {
        "run_id": run_id,
        "query_id": query_id,
        "variant": answer_variant,
        "attempt": accepted_ordinal,
        "answer_id": answer.answer_id,
    }
    diagnostics.append(
        AnswerFormatDiagnostic(
            diagnostic_id=stable_compiler_id("answer-format-diagnostic", accepted_payload),
            run_id=run_id,
            query_id=query_id,
            answer_variant=answer_variant,
            attempt_ordinal=accepted_ordinal,
            failure_codes=[],
            provider_schema_conformant=True,
            accepted=True,
            content_hash=canonical_content_hash(accepted_payload),
            metadata={"source_answer_id": answer.answer_id},
        )
    )
    aliases = [str(value) for value in answer.metadata["used_evidence_aliases"]]
    assert answer.model_call_id is not None
    fact = StructuredAnswerFact(
        structured_answer_output_id=stable_compiler_id(
            "structured-answer-output", accepted_payload
        ),
        run_id=run_id,
        query_id=query_id,
        answer_variant=answer_variant,
        attempt_ordinal=accepted_ordinal,
        model_call_id=answer.model_call_id,
        answer_id=answer.answer_id,
        answer=answer.final_answer,
        supported=True,
        used_evidence_aliases=aliases,
        abstain_reason=None,
        parse_status="accepted",
        provider_schema_conformant=True,
        finish_reason=(
            str(answer.metadata["finish_reason"])
            if answer.metadata.get("finish_reason") is not None
            else None
        ),
        content_hash=canonical_content_hash({**accepted_payload, "aliases": aliases}),
        metadata={
            "source_answer_id": answer.answer_id,
            "source_run_id": answer.run_id,
            "source_variant": answer.answer_variant,
        },
    )
    return diagnostics, fact


class PostgresOnlineQueryExecutorBackend:
    """Concrete backend with short transactions and deterministic recovery IDs."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        manifest: QueryExecutorRunManifest,
        config: QueryExecutorConfig,
        foundation: CompilerFoundationConfig,
        answer_inputs: PmmcEvidenceAnswerInputProvider,
        answer_service: SharedAnswerGenerationService,
        fallback: NativeMultimodalFallbackProvider,
        reranker: BoundedQBankReranker | None = None,
        reader_sessions: async_sessionmaker[AsyncSession] | None = None,
        joint_entry: ModelRegistryEntry | None = None,
    ) -> None:
        if manifest.config_hash != query_executor_config_hash(config):
            raise ValueError("query executor manifest/config identity mismatch")
        expected_operator_set = (
            "pmmc-operators-v7"
            if config.is_v3
            else "pmmc-operators-v6"
            if config.is_v2
            else "pmmc-operators-v5"
        )
        if foundation.operator_set_version != expected_operator_set:
            raise ValueError(f"query executor requires {expected_operator_set}")
        if config.reranker_enabled != (reranker is not None):
            raise ValueError("Step 7B requires exactly one bounded reranker service")
        self.sessions = sessions
        self.reader_sessions = reader_sessions if reader_sessions is not None else sessions
        self.manifest = manifest
        self.config = config
        if config.joint_relevance_enabled != (joint_entry is not None):
            raise ValueError("release R requires exactly one frozen native joint embedding model")
        self.joint_entry = joint_entry
        self.foundation = foundation
        self.answer_inputs = answer_inputs
        self.answer_service = answer_service
        self.fallback = fallback
        self.reranker = reranker
        self.queries = PostgresRuntimeQueryRepository(self.reader_sessions)
        self.database_semaphore = asyncio.Semaphore(config.database_semaphore)
        self._program_vectors: dict[str, ProgramQueryVectorSet] | None = None
        self._program_vectors_lock = asyncio.Lock()

    @property
    def fallback_run_id(self) -> str:
        return self.fallback.fallback_run_id

    @property
    def source_projection_hash(self) -> str:
        value = self.manifest.metadata.get("routing_projection_hash")
        if not isinstance(value, str) or not value:
            raise ValueError("v3 requires the frozen source projection hash")
        return value

    @property
    def source_match_run_id(self) -> str | None:
        return self.manifest.metadata.get("source_match_run_id")

    @property
    def execute_full_grid(self) -> bool:
        return self.manifest.metadata.get("execute_full_grid", True) is True

    @property
    def primary_readout(self) -> bool:
        return self.manifest.metadata.get("readout") == "exploration-primary-b8-r2"

    @property
    def relevance_grid_enabled(self) -> bool:
        return self.config.is_v3

    async def reuse_matching_phase(self, run_id: str, query_id: str) -> None:
        self._assert_run(run_id)
        source_id = self.source_match_run_id
        if not source_id or source_id == run_id:
            raise ValueError("match reuse requires a distinct source run")
        async with ExperimentUnitOfWork(self.reader_sessions, DataScope.RUNTIME) as uow:
            assert uow.query_executor is not None
            source_run = await uow.query_executor.get_run(source_id)
            checkpoint = await uow.query_executor.get_phase_state(source_id, query_id, "match")
        if (
            source_run is None
            or checkpoint is None
            or source_run["routing_projection_id"] != self.manifest.routing_projection_id
            or source_run["config_hash"] != self.manifest.config_hash
            or source_run["snapshot_id"] != self.manifest.snapshot_id
            or source_run["source_compiler_run_id"] != self.manifest.source_compiler_run_id
            or source_run["schema_version"] != "0.3.0"
            or source_run.get("metadata", {}).get("readout") != self.manifest.metadata.get("readout")
        ):
            raise ValueError("source match phase has incompatible frozen inputs")
        query = await self.load_query(run_id, query_id)
        cells = []
        prepared = {}
        executions: dict[str, ProgramExecutionResult] = {}
        rebuilt_contexts: dict[tuple[str, str], AnswerGenerationInput] = {}
        for raw in checkpoint["cells"]:
            if not self.execute_full_grid and (
                raw["cell"]["question_budget"],
                raw["cell"]["revision_depth"],
            ) != (8, 2):
                continue
            payload = {
                key: value
                for key, value in raw.items()
                if key not in {"cell_result_id", "content_hash", "created_at"}
            }
            payload["query_executor_run_id"] = run_id
            payload["metadata"] = {**payload.get("metadata", {}), "source_match_run_id": source_id}
            from pmmc.runtime.query_executor import OnlineQueryExecutor

            cell = QueryExperimentCellResult.model_validate(raw)
            grid_points = cell.metadata.get("relevance_grid", [])
            if grid_points:
                validate_relevance_grid_points(
                    grid_points, run_id=source_id, query_id=query_id,
                    execution_id=cell.execution_id or "",
                    thresholds=self.config.relevance_cosine_grid,
                )
            requests: dict[str, list[str] | None] = {}
            joint_contexts = {}
            if raw["cell_id"] in checkpoint["prepared"]:
                active_point = next((point for point in grid_points if (
                    point["text_threshold"] == self.config.query_evidence_relevance_threshold
                    and point["multimodal_threshold"]
                    == self.config.multimodal_evidence_relevance_threshold
                )), None)
                requests[raw["cell_id"]] = (
                    active_point["selected_reference_hashes"] if active_point else None
                )
            for point in grid_points:
                if point.get("prepared_key") is not None:
                    requests[point["prepared_key"]] = point["selected_reference_hashes"]
            for request_key, reference_hashes in requests.items():
                if request_key not in checkpoint["prepared"]:
                    raise ValueError("shared relevance context has no prepared source request")
                value = AnswerGenerationInput.model_validate(checkpoint["prepared"][request_key])
                validate_typed_answer_input(value)
                if (
                    cell.query_executor_run_id != source_id or cell.query_id != query_id
                    or cell.binding is None or cell.execution_id is None
                    or cell.binding.query_executor_run_id != source_id
                    or cell.binding.query_id != query_id
                    or cell.binding.routing_projection_id != self.manifest.routing_projection_id
                    or cell.cell.source_projection_id != self.manifest.routing_projection_id
                    or cell.cell.source_projection_hash != self.source_projection_hash
                    or value.run_id != source_id or value.snapshot_id != self.manifest.snapshot_id
                    or value.query != query or value.method_id != "pmmc"
                    or value.answer_variant != cell.cell.variant
                    or value.source_lineage.get("execution_id") != cell.execution_id
                    or value.source_lineage.get("plan_id") != cell.binding.plan_id
                ):
                    raise ValueError("shared answer context has incompatible execution lineage")
                execution = executions.get(cell.execution_id)
                if execution is None:
                    # Raw execution facts are read by the trusted runtime coordinator,
                    # never by granting compiler/private tables to the ONLINE reader.
                    async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                        assert uow.program_executions is not None
                        execution = await uow.program_executions.get(cell.execution_id)
                    if execution is None:
                        raise ValueError("shared answer execution is missing")
                    executions[cell.execution_id] = execution
                if (
                    execution.trace.run_id != source_id or execution.trace.query_id != query_id
                    or execution.trace.snapshot_id != self.manifest.snapshot_id
                    or execution.trace.source_compiler_run_id
                    != self.manifest.source_compiler_run_id
                    or execution.trace.plan_id != cell.binding.plan_id
                    or execution.trace.purpose is not OperatorExecutionPurpose.QUERY_TIME
                    or execution.trace.status is not TraceStatus.SUCCESS
                ):
                    raise ValueError("shared answer execution has incompatible frozen inputs")
                # Retain the relevance subset, but reapply the target provider's
                # budget to its full execution evidence, not a trimmed source prompt.
                selected_execution = (
                    select_execution_evidence(execution, reference_hashes)
                    if reference_hashes is not None else execution
                )
                context_key = (cell.execution_id, canonical_content_hash(reference_hashes))
                if context_key not in rebuilt_contexts:
                    input_provider = (
                        relevance_answer_input_provider(
                            self.answer_inputs, selected_execution,
                            direct_images_only=self.config.joint_relevance_enabled,
                        )
                        if reference_hashes is not None else self.answer_inputs
                    )
                    rebuilt_contexts[context_key] = await input_provider.prepare(
                        run_id=run_id, snapshot_id=self.manifest.snapshot_id, query=query,
                        result=selected_execution, variant=QueryAnswerVariant(cell.cell.variant),
                        source_execution_run_id=source_id,
                    )
                rebuilt = rebuilt_contexts[context_key].model_copy(
                    update={"answer_variant": cell.cell.variant}
                )
                source_contract = value.metadata.get("runtime_relation_contract")
                if source_contract is not None:
                    contract_payload = {key: item for key, item in source_contract.items()
                                        if key != "content_hash"}
                    if source_contract.get("content_hash") != canonical_content_hash(
                        contract_payload
                    ):
                        raise ValueError("shared relation grounding contract hash mismatch")
                    rebuilt = bind_runtime_relation_contract(
                        rebuilt, result=selected_execution,
                        minimum_groups=source_contract["minimum_evidence_groups"],
                        relation_required=True,
                    )
                if self.config.joint_relevance_enabled:
                    requirement = PlannerRequirementContract.model_validate(
                        value.metadata["joint_requirement"]
                    )
                    if requirement.content_hash != cell.binding.requirement_contract_hash:
                        raise ValueError("shared joint context requirement hash mismatch")
                    joint_hashes = value.metadata["joint_image_hashes"]
                    points = [
                        p
                        for p in grid_points
                        if p.get("prepared_key") == request_key
                        or (
                            request_key == cell.cell_id
                            and p["text_threshold"]
                            == self.config.query_evidence_relevance_threshold
                            and p["multimodal_threshold"]
                            == self.config.multimodal_evidence_relevance_threshold
                        )
                    ]
                    if not points or any(
                        p.get("joint_image_hashes") != joint_hashes for p in points
                    ):
                        raise ValueError("shared joint context differs from its score projection")
                    for point in points:
                        contexts = cell.metadata.get("joint_contexts", {})
                        validate_joint_outcome(point, contexts)
                        if contexts.get(point["prepared_key"]) != value.metadata.get(
                            "joint_context"
                        ):
                            raise ValueError("shared source context receipt mismatch")
                    if requires_independent_relation(requirement) and source_contract is None:
                        raise ValueError("shared release R context lacks relation grounding")
                    rebuilt = bind_joint_context(
                        rebuilt, result=selected_execution, binding=cell.binding,
                        requirement=requirement, joint_image_hashes=joint_hashes,
                    )
                    if any(p.get("prepared_key") == request_key for p in points):
                        joint_contexts[request_key] = rebuilt.metadata["joint_context"]
                prepared[request_key] = rebuilt.model_dump(mode="json")
            if self.config.joint_relevance_enabled:
                payload["metadata"]["joint_contexts"] = joint_contexts
            if grid_points:
                payload["metadata"]["relevance_grid"] = [
                    {
                        **point,
                        "query_executor_run_id": run_id,
                        "source_query_executor_run_id": source_id,
                        "source_grid_point_id": point["grid_point_id"],
                        "grid_point_id": stable_compiler_id(
                            "query-relevance-grid-consumer", [run_id, point["grid_point_id"]]
                        ),
                        "answer_identity_hash": self.typed_request_identity(
                            AnswerGenerationInput.model_validate(prepared[point["prepared_key"]])
                        ) if point.get("prepared_key") else None,
                    }
                    for point in grid_points
                ]
            cells.append(OnlineQueryExecutor._cell_result(payload).model_dump(mode="json"))
        await self.persist_runtime_phase(
            run_id,
            query_id,
            "match",
            {
                "cells": cells,
                "prepared": prepared,
                "route": checkpoint["route"],
                "source_match_run_id": source_id,
            },
        )

    async def _fence(self, uow: ExperimentUnitOfWork) -> None:
        if self.config.is_v3:
            lease = query_attempt_lease.get()
            if lease is None:
                raise RuntimeError("v3 writes require an active attempt lease")
            assert uow.query_executor is not None
            await uow.query_executor.assert_attempt_lease(
                job_id=lease.job_id,
                worker_id=lease.owner,
                attempt_count=lease.attempt_count,
                now=utc_now(),
            )

    def _assert_run(self, run_id: str) -> None:
        if run_id != self.manifest.query_executor_run_id:
            raise ValueError("query backend received another query executor run")

    async def load_query(self, run_id: str, query_id: str) -> QueryInput:
        self._assert_run(run_id)
        query = await self.queries.get(self.manifest.snapshot_id, query_id)
        if query.dataset is not self.manifest.dataset:
            raise ValueError("query dataset differs from the query executor run")
        if query.available_after_session is None:
            raise ValueError("online query requires a prospective cutoff session")
        return query

    async def load_verified_projection(self, run_id: str, query: QueryInput) -> str:
        self._assert_run(run_id)
        if self.config.is_v3:
            async with ExperimentUnitOfWork(self.reader_sessions, DataScope.RUNTIME) as uow:
                assert uow.qbank_routing is not None
                runtime_projection = await uow.qbank_routing.get_runtime_projection_v3(
                    self.manifest.source_compiler_run_id
                )
            if runtime_projection is None or any(
                runtime_projection[key] != expected
                for key, expected in {
                    "routing_projection_id": self.manifest.routing_projection_id,
                    "snapshot_id": self.manifest.snapshot_id,
                    "active_round": self.config.active_round,
                    "max_round": max(self.config.rounds),
                    "content_hash": self.source_projection_hash,
                }.items()
            ) or query.dataset is not self.manifest.dataset:
                raise ValueError("query run and runtime projection disagree")
            return self.manifest.routing_projection_id
        async with self.database_semaphore:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.doubter is not None
                value = await uow.doubter.get_routing_projection(
                    self.manifest.source_compiler_run_id
                )
        if value is None:
            raise KeyError("verified QBank routing projection does not exist")
        projection, _members = value
        if (
            projection.routing_projection_id != self.manifest.routing_projection_id
            or projection.snapshot_id != self.manifest.snapshot_id
            or projection.active_round != self.config.active_round
            or projection.max_round != max(self.config.rounds)
        ):
            raise ValueError("query run and verified QBank projection disagree")
        if query.dataset is not self.manifest.dataset:
            raise ValueError("query escaped the projection dataset")
        return projection.routing_projection_id

    async def ensure_precomputed_query_embeddings(self, run_id: str, query: QueryInput) -> None:
        self._assert_run(run_id)
        async with self.database_semaphore:
            async with ExperimentUnitOfWork(self.reader_sessions, DataScope.RUNTIME) as uow:
                assert uow.query_executor is not None
                values = await uow.query_executor.get_query_embedding_vectors(
                    run_id=run_id,
                    query_id=query.query_id,
                )
        expected = {
            channel.value
            for channel in required_query_embedding_channels(
                query,
                include_text_only_multimodal=self.config.uses_reranker,
            )
        }
        if set(values) != expected:
            raise RuntimeError("query embeddings do not match the query modality")
        if any(len(vector) != 4096 for _fact, vector in values.values()):
            raise RuntimeError("query embedding dimension is not 4096")

    async def _scope(
        self, query: QueryInput, *, round_id: int
    ) -> list[VerifiedQBankRoutingItemV2 | VerifiedQBankRoutingItemV3]:
        assert query.available_after_session is not None
        async with self.database_semaphore:
            async with ExperimentUnitOfWork(self.reader_sessions, DataScope.RUNTIME) as uow:
                assert uow.qbank_routing is not None
                values: list[VerifiedQBankRoutingItemV2 | VerifiedQBankRoutingItemV3] = []
                if self.config.uses_reranker:
                    list_verified = (
                        uow.qbank_routing.list_verified_v4
                        if self.config.is_v3
                        else uow.qbank_routing.list_verified_v3
                    )
                    values.extend(
                        await list_verified(
                            routing_projection_id=self.manifest.routing_projection_id,
                            compiler_run_id=self.manifest.source_compiler_run_id,
                            snapshot_id=self.manifest.snapshot_id,
                            dataset=self.manifest.dataset.value,
                            conversation_id=query.conversation_id,
                            cutoff_session_id=query.available_after_session,
                            question_budget=8,
                            round_id=round_id,
                        )
                    )
                else:
                    values.extend(
                        await uow.qbank_routing.list_verified_v2(
                            routing_projection_id=self.manifest.routing_projection_id,
                            compiler_run_id=self.manifest.source_compiler_run_id,
                            snapshot_id=self.manifest.snapshot_id,
                            dataset=self.manifest.dataset.value,
                            conversation_id=query.conversation_id,
                            cutoff_session_id=query.available_after_session,
                            question_budget=8,
                            round_id=round_id,
                        )
                    )
                return values

    async def _query_vectors(
        self, query_id: str
    ) -> dict[str, tuple[QueryEmbeddingFact, list[float]]]:
        async with self.database_semaphore:
            async with ExperimentUnitOfWork(self.reader_sessions, DataScope.RUNTIME) as uow:
                assert uow.query_executor is not None
                return await uow.query_executor.get_query_embedding_vectors(
                    run_id=self.manifest.query_executor_run_id,
                    query_id=query_id,
                )

    async def load_channel_hits(
        self, run_id: str, query: QueryInput
    ) -> list[RuntimeQBankChannelHit]:
        if self.config.is_v3:
            # Exact SQL scope for each round, with no shared top-k truncation. R0
            # and rescued members can be absent from any later projection.
            result = []
            for round_id in self.config.rounds:
                result.extend(await self._load_round_channel_hits(run_id, query, round_id))
            return result
        return await self._load_round_channel_hits(run_id, query, 3)

    async def _load_round_channel_hits(
        self,
        run_id: str,
        query: QueryInput,
        round_id: int,
    ) -> list[RuntimeQBankChannelHit]:
        self._assert_run(run_id)
        scope = await self._scope(query, round_id=round_id)
        if not scope:
            return []
        by_id = {item.qbank_id: item for item in scope}
        if len(by_id) != len(scope):
            raise RuntimeError("verified routing scope contains duplicate QBank IDs")
        collection_tuples = {
            (
                item.text_collection_id,
                item.multimodal_collection_id,
                item.lexical_collection_id,
            )
            for item in scope
        }
        if len(collection_tuples) != 1:
            raise RuntimeError("verified routing rows disagree on collection identity")
        text_collection, mm_collection, _lexical_collection = next(iter(collection_tuples))
        vectors = await self._query_vectors(query.query_id)
        text_vector = vectors["text"][1]
        mm_vector = vectors.get("multimodal")
        top_k = max(self.config.match_top_k, len(scope))
        cutoff_session_id = query.available_after_session
        assert cutoff_session_id is not None

        async def search(
            channel: QBankRuntimeChannel,
            route_class: QBankRouteClass,
        ) -> tuple[QBankRuntimeChannel, QBankRouteClass, list[tuple[str, float]]]:
            async with self.database_semaphore:
                async with ExperimentUnitOfWork(self.reader_sessions, DataScope.RUNTIME) as uow:
                    assert uow.qbank_routing is not None
                    if channel is QBankRuntimeChannel.LEXICAL:
                        lexical_search = (
                            uow.qbank_routing.search_lexical_verified_v4
                            if self.config.is_v3
                            else uow.qbank_routing.search_lexical_verified_v3
                            if self.config.uses_reranker
                            else uow.qbank_routing.search_lexical_verified_v2
                        )
                        values = await lexical_search(
                            query_text=query.raw_query,
                            routing_projection_id=self.manifest.routing_projection_id,
                            compiler_run_id=self.manifest.source_compiler_run_id,
                            snapshot_id=self.manifest.snapshot_id,
                            dataset=self.manifest.dataset.value,
                            conversation_id=query.conversation_id,
                            cutoff_session_id=cutoff_session_id,
                            route_class=route_class.value,
                            question_budget=8,
                            round_id=round_id,
                            top_k=top_k,
                            **(
                                {"term_limit": self.config.lexical_term_limit}
                                if self.config.uses_reranker
                                else {}
                            ),
                        )
                    else:
                        collection_id = (
                            text_collection
                            if channel is QBankRuntimeChannel.TEXT
                            else mm_collection
                        )
                        query_vector = (
                            text_vector
                            if channel is QBankRuntimeChannel.TEXT
                            else (mm_vector[1] if mm_vector is not None else None)
                        )
                        if query_vector is None:
                            raise RuntimeError("multimodal matcher lacks a query-image vector")
                        embedding_search = (
                            uow.qbank_routing.search_embedding_verified_v4
                            if self.config.is_v3
                            else uow.qbank_routing.search_embedding_verified_v3
                            if self.config.uses_reranker
                            else uow.qbank_routing.search_embedding_verified_v2
                        )
                        values = await embedding_search(
                            collection_id=collection_id,
                            query_vector=query_vector,
                            routing_projection_id=self.manifest.routing_projection_id,
                            compiler_run_id=self.manifest.source_compiler_run_id,
                            snapshot_id=self.manifest.snapshot_id,
                            dataset=self.manifest.dataset.value,
                            conversation_id=query.conversation_id,
                            cutoff_session_id=cutoff_session_id,
                            route_class=route_class.value,
                            question_budget=8,
                            round_id=round_id,
                            top_k=top_k,
                        )
            return channel, route_class, values

        active_channels = [
            QBankRuntimeChannel.LEXICAL,
            QBankRuntimeChannel.TEXT,
        ]
        if query.query_image_refs:
            active_channels.append(QBankRuntimeChannel.MULTIMODAL)
        ranked = await asyncio.gather(
            *(
                search(channel, route_class)
                for channel in active_channels
                for route_class in QBankRouteClass
            )
        )

        result: list[RuntimeQBankChannelHit] = []
        for channel, route_class, values in ranked:
            for qbank_id, score in values:
                item = by_id.get(qbank_id)
                if item is None:
                    raise RuntimeError("SQL matcher returned an item outside the projection")
                if item.route_class is not route_class:
                    raise RuntimeError("SQL matcher crossed a verified route-class boundary")
                accepted_round = item.metadata.get("accepted_round")
                if self.config.is_v3:
                    # This legacy export field is not membership. R0 need never
                    # have passed Doubter; only its exact projection is authoritative.
                    from pmmc.schemas.qbank import VerifiedQBankRoutingItemV4

                    if not isinstance(item, VerifiedQBankRoutingItemV4):
                        raise RuntimeError("v3 executor received a historical routing item")
                    accepted_round = item.program_round
                if not isinstance(accepted_round, int) or accepted_round not in range(4):
                    raise RuntimeError("verified routing item lacks an accepted round")
                result.append(
                    RuntimeQBankChannelHit(
                        qbank_id=qbank_id,
                        plan_id=item.plan_id,
                        route_class=item.route_class,
                        generation_rank=item.generation_rank,
                        accepted_round=accepted_round,
                        required_modalities=tuple(item.required_modalities),
                        channel=channel,
                        raw_score=score,
                        query_image_required=False,
                        candidate_question=item.canonical_question,
                        question_type=item.question_type.value,
                        projection_round=round_id if self.config.is_v3 else None,
                        source_program_hash=item.program_hash if self.config.is_v3 else None,
                    )
                )
        return result

    async def rerank_qbank_candidates(
        self,
        *,
        run_id: str,
        query: QueryInput,
        candidates: tuple[QBankMatchFusedCandidate, ...],
    ) -> dict[str, QBankRerankerAssessment]:
        self._assert_run(run_id)
        if not self.config.uses_reranker:
            return {}
        reranker = self.reranker
        if reranker is None:
            raise RuntimeError("bounded reranker service is unavailable")
        if not candidates:
            return {}
        call_ids = {
            item.qbank_id: reranker_model_call_id(run_id, query.query_id, item.qbank_id)
            for item in candidates
        }
        async with self.database_semaphore:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.query_executor is not None
                existing_calls = await uow.query_executor.get_reranker_calls(
                    run_id,
                    query.query_id,
                    list(call_ids.values()),
                )
        if existing_calls:
            print(
                "STEP7_RERANKER_RECOVERY "
                + json.dumps(
                    {
                        "state": "checking_cached_calls",
                        "pair_count": len(candidates),
                        "cached_call_count": len(existing_calls),
                        "new_call_count": len(candidates) - len(existing_calls),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )

        async def assess(candidate: QBankMatchFusedCandidate) -> QBankRerankerAssessment:
            question = candidate.metadata.get("candidate_question")
            question_type = candidate.metadata.get("question_type")
            modality_values = candidate.metadata.get("required_modalities")
            if (
                not isinstance(question, str)
                or not question
                or not isinstance(question_type, str)
                or not isinstance(modality_values, list)
            ):
                raise ValueError("base matcher candidate lacks runtime-safe reranker fields")
            return await reranker.assess(
                run_id=run_id,
                snapshot_id=self.manifest.snapshot_id,
                query=query,
                existing_trace=existing_calls.get(call_ids[candidate.qbank_id]),
                pair=BoundedRerankerPair(
                    qbank_id=candidate.qbank_id,
                    candidate_question=question,
                    question_type=question_type,
                    required_modalities=tuple(Modality(value) for value in modality_values),
                ),
            )

        # Drain every bounded pair before the whole-query lease can retry. A
        # fail-fast gather leaves sibling provider calls writing the same IDs.
        values = await asyncio.gather(
            *(assess(candidate) for candidate in candidates),
            return_exceptions=True,
        )
        result = {}
        for value in values:
            if isinstance(value, BaseException):
                raise value
            result[value.qbank_id] = value
        return result

    async def _program_query_vectors_for(self, qbank_id: str) -> ProgramQueryVectorSet:
        if self._program_vectors is None:
            async with self._program_vectors_lock:
                if self._program_vectors is None:
                    async with self.database_semaphore:
                        if self.config.is_v3:
                            self._program_vectors = await self._runtime_program_vectors()
                        else:
                            self._program_vectors = await load_program_query_vectors(
                                self.reader_sessions,
                                run_id=self.manifest.source_compiler_run_id,
                            )
        try:
            return self._program_vectors[qbank_id]
        except KeyError as exc:
            raise KeyError("frozen program query vectors are unavailable") from exc

    async def _runtime_program_vectors(self) -> dict[str, ProgramQueryVectorSet]:
        by_question: dict[str, dict[str, list[float]]] = {}
        async with ExperimentUnitOfWork(self.reader_sessions, DataScope.RUNTIME) as uow:
            assert uow.qbank_routing is not None and uow.embeddings is not None
            refs = await uow.qbank_routing.list_runtime_program_vector_refs_v3(
                self.manifest.source_compiler_run_id
            )
            for ref in refs:
                record = await uow.embeddings.get_record(ref["embedding_id"])
                if record is None:
                    raise RuntimeError("runtime program vector reference is dangling")
                by_question.setdefault(ref["qbank_id"], {})[ref["channel"]] = record.vector
        return {
            key: ProgramQueryVectorSet(
                qbank_id=key, text_vector=values["question_text_query"],
                joint_vector=values["question_joint_query"],
            )
            for key, values in by_question.items()
            if set(values) == {"question_text_query", "question_joint_query"}
        }

    async def bind_frozen_program(
        self,
        *,
        run_id: str,
        query: QueryInput,
        projection_id: str,
        route_decision: QueryRouteDecision,
        candidate: QBankMatchFusedCandidate,
        round_id: int,
        counterfactual: bool,
    ) -> FrozenProgramBinding:
        self._assert_run(run_id)
        if projection_id != self.manifest.routing_projection_id:
            raise ValueError("program binding references another projection")
        scope = await self._scope(query, round_id=round_id)
        routing = next(
            (item for item in scope if item.qbank_id == candidate.qbank_id),
            None,
        )
        if routing is None or routing.plan_id != candidate.plan_id:
            raise ValueError("match candidate is absent from the requested round projection")
        async with self.database_semaphore:
            async with ExperimentUnitOfWork(self.reader_sessions, DataScope.RUNTIME) as uow:
                assert uow.qbank_routing is not None
                assert uow.planner is not None
                assert uow.memory_operator_indexes is not None
                if self.config.is_v3:
                    runtime_program = await uow.qbank_routing.get_runtime_program_v3(
                        routing.plan_id
                    )
                    safe = routing.requirement_contract
                    requirement: PlannerRequirementContract | None = PlannerRequirementContract(
                        **{
                            key: safe[key]
                            for key in (
                                "requirement_contract_id",
                                "required_modalities",
                                "required_capabilities",
                                "allowed_capabilities",
                                "minimum_evidence_groups",
                                "fallback_required",
                                "refusal_only",
                                "content_hash",
                            )
                        },
                        run_id=self.manifest.source_compiler_run_id,
                        qbank_id=routing.qbank_id,
                        compiler_unit_id=(
                            runtime_program[0].compiler_unit_id if runtime_program else None
                        ),
                        metadata={
                            "required_semantic_roles": safe.get("required_semantic_roles", []),
                            "independent_group_relation_required": safe.get(
                                "independent_group_relation_required", False
                            ),
                            "relation_assessment_stage": safe.get(
                                "relation_assessment_stage", "not_applicable"
                            ),
                        },
                    )
                else:
                    runtime_program = await uow.qbank_routing.get_runtime_program(routing.plan_id)
                    requirement = await uow.planner.get_requirement_contract(routing.qbank_id)
                memory_index = await uow.memory_operator_indexes.get_for_run(
                    self.manifest.source_compiler_run_id
                )
        if runtime_program is None or requirement is None or memory_index is None:
            raise KeyError("frozen program, requirement, or memory index is missing")
        program, source_program_hash = runtime_program
        if (
            source_program_hash != routing.program_hash
            or source_program_hash
            != candidate.metadata.get("source_program_hash", source_program_hash)
            or program.qbank_id != routing.qbank_id
            or program.run_id != self.manifest.source_compiler_run_id
            or program.snapshot_id != self.manifest.snapshot_id
            or program.operator_set_version
            != (
                "pmmc-operators-v7"
                if self.config.is_v3
                else "pmmc-operators-v6"
                if self.config.is_v2
                else "pmmc-operators-v5"
            )
        ):
            raise ValueError("frozen program identity or scope drifted")
        if memory_index.status != "ready" or memory_index.snapshot_id != self.manifest.snapshot_id:
            raise RuntimeError("memory operator collections are not ready for this snapshot")
        self._validate_requirement(routing, requirement)
        vectors = await self._program_query_vectors_for(routing.qbank_id)
        execution_id = stable_compiler_id(
            "query-program-execution",
            (
                [run_id, query.query_id, program.plan_id, source_program_hash]
                if self.config.uses_reranker
                else [
                    run_id,
                    query.query_id,
                    round_id,
                    program.plan_id,
                    source_program_hash,
                ]
            ),
        )
        payload = {
            "run_id": run_id,
            "query_id": query.query_id,
            "projection_id": projection_id,
            "qbank_id": routing.qbank_id,
            "plan_id": program.plan_id,
            "round_id": round_id,
            "source_program_hash": source_program_hash,
            "runtime_program_hash": program.program_hash,
            "requirement_hash": requirement.content_hash,
            "counterfactual": counterfactual,
            **(
                {"question_budget": candidate.metadata["question_budget"]}
                if self.config.is_v3
                else {}
            ),
        }
        binding = QueryProgramBinding(
            query_program_binding_id=stable_compiler_id("query-program-binding", payload),
            query_executor_run_id=run_id,
            query_id=query.query_id,
            route_decision_id=route_decision.route_decision_id,
            routing_projection_id=projection_id,
            qbank_id=routing.qbank_id,
            plan_id=program.plan_id,
            round_id=round_id,
            program_hash=source_program_hash,
            operator_set_version=str(program.operator_set_version),
            requirement_contract_hash=requirement.content_hash,
            counterfactual=counterfactual,
            content_hash=canonical_content_hash(payload),
            metadata={
                "execution_id": execution_id,
                "source_plan_round": program.round_id,
                "runtime_program_hash": program.program_hash,
                **(
                    {"question_budget": candidate.metadata["question_budget"]}
                    if self.config.is_v3
                    else {}
                ),
            },
        )
        return FrozenProgramBinding(
            binding=binding,
            program=program,
            requirement=requirement,
            predicted_question=routing.canonical_question,
            collection_bindings={
                "text_memory": memory_index.text_collection_id,
                "visual_memory": memory_index.visual_collection_id,
            },
            query_vectors={
                "question_text_query": vectors.text_vector,
                "question_joint_query": vectors.joint_vector,
            },
            source_program_hash=source_program_hash,
        )

    @staticmethod
    def _validate_requirement(
        routing: VerifiedQBankRoutingItemV2,
        requirement: PlannerRequirementContract,
    ) -> None:
        safe = routing.requirement_contract
        expected = {
            "requirement_contract_id": requirement.requirement_contract_id,
            "required_modalities": [item.value for item in requirement.required_modalities],
            "required_capabilities": [item.value for item in requirement.required_capabilities],
            "allowed_capabilities": [item.value for item in requirement.allowed_capabilities],
            "minimum_evidence_groups": requirement.minimum_evidence_groups,
            "fallback_required": requirement.fallback_required,
            "refusal_only": requirement.refusal_only,
            "content_hash": requirement.content_hash,
        }
        for key, value in expected.items():
            if safe.get(key) != value:
                raise ValueError("runtime requirement projection differs from its source")

    async def execute_memory_program(
        self, query: QueryInput, frozen: FrozenProgramBinding
    ) -> ProgramExecutionResult:
        if self.config.is_v3 and self.config.treatment == "no_raw_image":
            if any(
                step.op_name
                in {
                    "inspect_raw_image",
                    "retrieve_image_by_visual_embedding",
                    "retrieve_image_text_pair",
                    "retrieve_context_around_image",
                }
                for step in frozen.program.operator_sequence
            ):
                raise ValueError("caption-only treatment received a raw-image program")
        execution_id = frozen.binding.metadata.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError("query binding lacks a stable execution identity")
        async with self.database_semaphore:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.program_executions is not None
                existing = await uow.program_executions.get(execution_id)
        if existing is not None:
            if (
                existing.trace.run_id != frozen.binding.query_executor_run_id
                or existing.trace.query_id != query.query_id
                or existing.trace.plan_id != frozen.program.plan_id
            ):
                raise ValueError("persisted query execution conflicts with its binding")
            return existing
        if query.available_after_session is None or frozen.program.compiler_unit_id is None:
            raise ValueError("query execution lacks cutoff or compiler-unit scope")
        context = OperatorExecutionContext(
            run_id=frozen.binding.query_executor_run_id,
            source_compiler_run_id=self.manifest.source_compiler_run_id,
            snapshot_id=self.manifest.snapshot_id,
            conversation_id=query.conversation_id,
            cutoff_session_id=query.available_after_session,
            compiler_unit_id=frozen.program.compiler_unit_id,
            query_id=query.query_id,
            question=frozen.predicted_question,
            query_image_ids=[item.image_id for item in query.query_image_refs],
            required_modalities=[
                Modality(item.value) for item in frozen.requirement.required_modalities
            ],
            collection_bindings=dict(frozen.collection_bindings),
            query_vectors={key: list(value) for key, value in frozen.query_vectors.items()},
            constraints={
                "runtime_policy": self.config.sufficiency_policy_version,
                "source_program_hash": frozen.source_program_hash,
            },
            metadata={"counterfactual": frozen.binding.counterfactual},
        )
        data = SessionScopedOperatorDataRepository(self.reader_sessions)
        registry = build_operator_registry(data, self.foundation)
        executor = LinearProgramExecutor(
            registry,
            ProgramValidator(registry, self.foundation),
            data,
            self.foundation,
        )
        async with self.database_semaphore:
            result = await executor.execute(
                execution_id=execution_id,
                program=frozen.program,
                context=context,
                purpose=OperatorExecutionPurpose.QUERY_TIME,
            )
        async with self.database_semaphore:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.program_executions is not None
                await self._fence(uow)
                return await uow.program_executions.add(result)

    async def relevant_runtime_evidence_groups(
        self,
        *,
        query: QueryInput,
        frozen: FrozenProgramBinding,
        result: ProgramExecutionResult,
    ) -> RuntimeEvidenceRelevance:
        """Score actual evidence against the real query without compiler/gold facts."""

        if not self.config.uses_reranker:
            raise RuntimeError("runtime evidence relevance requires query executor v2")
        vectors = await self._query_vectors(query.query_id)
        source_groups: dict[str, set[str]] = defaultdict(set)
        source_channel: dict[str, str] = {}
        group_roles: dict[str, set[str]] = defaultdict(set)
        for evidence in result.evidence:
            groups: set[str] = set()
            plural = evidence.metadata.get("evidence_group_ids")
            if isinstance(plural, list):
                groups.update(value for value in plural if isinstance(value, str) and value)
            singular = evidence.metadata.get("evidence_group_id")
            if isinstance(singular, str) and singular:
                groups.add(singular)
            if not groups:
                groups.add(f"{evidence.evidence_type.value}:{evidence.evidence_id}")
            roles = evidence.metadata.get("semantic_roles", [])
            if not isinstance(roles, list) or any(not isinstance(role, str) for role in roles):
                raise ValueError("runtime semantic role annotations must be strings")
            for group in groups:
                group_roles[group].update(roles)
            sources: list[tuple[str, str]] = []
            if evidence.image_id is not None:
                sources.append((evidence.image_id, "multimodal"))
            if evidence.memory_id is not None:
                sources.extend(
                    (
                        (f"memory-text:{evidence.memory_id}", "text"),
                        (f"memory-caption:{evidence.memory_id}", "text"),
                    )
                )
            owner = evidence.metadata.get("memory_id")
            if evidence.caption_id is not None and isinstance(owner, str) and owner:
                sources.append((f"memory-caption:{owner}", "text"))
            for source_id, channel in sources:
                source_groups[source_id].update(groups)
                source_channel[source_id] = channel
        threshold = self.config.query_evidence_relevance_threshold
        required_groups = frozen.requirement.minimum_evidence_groups
        required_roles = frozen.requirement.metadata.get("required_semantic_roles", [])
        if not isinstance(required_roles, list) or any(
            not isinstance(role, str) for role in required_roles
        ):
            raise ValueError("required semantic roles must be a runtime-safe string list")

        def role_gate(groups: set[str]) -> bool:
            if not self.config.is_v3:
                return True
            present = {role for group in groups for role in group_roles[group]}
            return (set(required_roles) - DEFERRED_GROUNDING_ROLES) <= present

        def relevance_metadata(groups: set[str]) -> dict:
            if not self.config.is_v3:
                return {"policy_version": "query-runtime-evidence-v2"}
            return {
                "policy_version": "query-runtime-evidence-v3",
                **({"joint_policy": JOINT_POLICY, "joint_threshold": JOINT_THRESHOLD}
                   if self.config.joint_relevance_enabled else {}),
                "score_transform": self.config.relevance_score_transform,
                "required_semantic_roles": required_roles,
                "required_roles_satisfied": role_gate(groups),
                "semantic_roles_declared": "required_semantic_roles" in frozen.requirement.metadata,
                "semantic_relation_status": (
                    "pending_grounding" if requires_independent_relation(frozen.requirement)
                    else "not_applicable"
                ),
                "answer_grounding_status": "not_evaluated",
                "channel_thresholds": {
                    "text": threshold,
                    "multimodal": self.config.multimodal_evidence_relevance_threshold,
                },
            }

        assessment_payload = {
            "run_id": self.manifest.query_executor_run_id,
            "query_id": query.query_id,
            "binding_id": frozen.binding.query_program_binding_id,
            "execution_id": result.trace.execution_id,
            "threshold": threshold,
        }
        if self.config.is_v3:
            assessment_payload["multimodal_threshold"] = (
                self.config.multimodal_evidence_relevance_threshold
            )
        if self.config.joint_relevance_enabled:
            assessment_payload["joint_policy"] = JOINT_POLICY
            assessment_payload["joint_threshold"] = JOINT_THRESHOLD
        assessment_id = stable_compiler_id(
            "query-evidence-relevance-assessment", assessment_payload
        )
        if not source_groups:
            assessment = QueryEvidenceRelevanceAssessment(
                relevance_assessment_id=assessment_id,
                query_executor_run_id=self.manifest.query_executor_run_id,
                query_id=query.query_id,
                query_program_binding_id=frozen.binding.query_program_binding_id,
                relevance_threshold=threshold,
                required_group_count=required_groups,
                relevant_group_count=0,
                passed=False,
                content_hash=canonical_content_hash(
                    {**assessment_payload, "relevant_group_hashes": []}
                ),
                metadata=relevance_metadata(set()),
            )
            return RuntimeEvidenceRelevance(group_ids=frozenset(), assessment=assessment, items=())
        collection_ids = set(frozen.collection_bindings.values())
        async with self.reader_sessions() as session:
            rows = list(
                await session.scalars(
                    select(EmbeddingModel).where(
                        EmbeddingModel.collection_id.in_(collection_ids),
                        EmbeddingModel.source_id.in_(list(source_groups)),
                    )
                )
            )
        relevant: set[str] = set()
        relevance_items: list[QueryEvidenceRelevanceItem] = []
        joint_records = {}
        exclusions = {}
        image_ids = {e.image_id for e in result.evidence if e.image_id is not None}
        if self.config.joint_relevance_enabled and image_ids:
            assert self.joint_entry is not None
            pairs, exclusions = await scoped_joint_pairs(
                self.answer_inputs.memories, self.manifest.snapshot_id, query,
            )
            if not image_ids <= {p.image_id for p in pairs} | exclusions.keys():
                raise ValueError("executed image has no visible owner")
            for batch, _, collection in joint_shards(
                self.manifest.snapshot_id, self.joint_entry, pairs
            ):
                wanted = [p for p in batch if p.image_id in image_ids]
                if wanted:
                    joint_records.update(
                        await load_joint_records(self.reader_sessions, collection, wanted)
                    )
            if joint_records and "multimodal" not in vectors:
                raise ValueError("joint relevance requires the native runtime query vector")
        scores: list[tuple[str, str, list[float], dict[str, Any]]] = [
            (row.source_id, source_channel[row.source_id], list(row.embedding), {
                "embedding_id": row.embedding_id, "collection_id": row.collection_id,
                "model_call_id": row.model_call_id,
            } if self.config.joint_relevance_enabled else {})
            for row in sorted(rows, key=lambda value: (value.source_id, value.embedding_id))
        ]
        for image_id, record in sorted(joint_records.items()):
            scores.append((image_id, "joint", record.vector, {
                "joint_policy": JOINT_POLICY,
                "owner_id_hash": canonical_content_hash(record.metadata["owner_id"]),
                "image_id_hash": canonical_content_hash(image_id), "pair_hash": record.input_hash,
                "collection_id": record.collection_id, "embedding_id": record.embedding_id,
                "model_call_id": record.model_call_id,
                "query_embedding_id": vectors["multimodal"][0].embedding_id,
            }))
        for source_id, embedding_channel, evidence_vector, audit in scores:
            query_value = vectors.get(
                "multimodal" if embedding_channel == "joint" else embedding_channel
            )
            if query_value is None:
                continue
            query_vector = query_value[1]
            if len(query_vector) != len(evidence_vector):
                raise ValueError("query/evidence relevance dimension mismatch")
            numerator = math.fsum(
                left * right for left, right in zip(query_vector, evidence_vector, strict=True)
            )
            left_norm = math.sqrt(math.fsum(value * value for value in query_vector))
            right_norm = math.sqrt(math.fsum(value * value for value in evidence_vector))
            cosine = (
                min(1.0, max(-1.0, numerator / (left_norm * right_norm)))
                if left_norm and right_norm
                else -1.0
            )
            normalized = min(1.0, max(0.0, (cosine + 1.0) / 2.0))
            channel_threshold = (
                JOINT_THRESHOLD if embedding_channel == "joint" else
                self.config.multimodal_evidence_relevance_threshold
                if self.config.is_v3 and embedding_channel == "multimodal"
                else threshold
            )
            is_relevant = (cosine if self.config.is_v3 else normalized) >= channel_threshold
            evidence_groups = sorted(source_groups[source_id])
            if self.config.joint_relevance_enabled:
                if embedding_channel == "multimodal" and source_id in exclusions:
                    audit["joint_ineligible_reason"] = exclusions[source_id]
                audit["filter_reason"] = (
                    "query_image" if audit.get("joint_ineligible_reason") == "query_image" else
                    "threshold_met" if is_relevant else "below_threshold"
                )
            if is_relevant and audit.get("joint_ineligible_reason") != "query_image":
                relevant.update(evidence_groups)
            for group_id in evidence_groups:
                payload = {
                    "assessment_id": assessment_id,
                    "source_hash": canonical_content_hash(source_id),
                    "group_hash": canonical_content_hash(group_id),
                    "channel": embedding_channel,
                    "cosine": cosine,
                    "threshold": channel_threshold,
                    **audit,
                }
                relevance_items.append(
                    QueryEvidenceRelevanceItem(
                        relevance_item_id=stable_compiler_id(
                            "query-evidence-relevance-item", payload
                        ),
                        relevance_assessment_id=assessment_id,
                        query_executor_run_id=self.manifest.query_executor_run_id,
                        query_id=query.query_id,
                        execution_id=result.trace.execution_id,
                        ordinal=len(relevance_items),
                        evidence_id_hash=payload["source_hash"],
                        evidence_group_hash=payload["group_hash"],
                        modality=(
                            Modality.IMAGE
                            if embedding_channel in {"joint", "multimodal"}
                            else Modality.TEXT
                        ),
                        cosine_similarity=cosine,
                        normalized_score=normalized,
                        relevant=is_relevant,
                        threshold=channel_threshold,
                        content_hash=canonical_content_hash(payload),
                        metadata={
                            "embedding_channel": embedding_channel,
                            **audit,
                            **({"score_transform": "raw-cosine-v1"} if self.config.is_v3 else {}),
                        },
                    )
                )
        assessment = QueryEvidenceRelevanceAssessment(
            relevance_assessment_id=assessment_id,
            query_executor_run_id=self.manifest.query_executor_run_id,
            query_id=query.query_id,
            query_program_binding_id=frozen.binding.query_program_binding_id,
            relevance_threshold=threshold,
            required_group_count=required_groups,
            relevant_group_count=len(relevant),
            passed=len(relevant) >= required_groups and role_gate(relevant),
            content_hash=canonical_content_hash(
                {
                    **assessment_payload,
                    "relevant_group_hashes": sorted(
                        canonical_content_hash(value) for value in relevant
                    ),
                    "item_hashes": [item.content_hash for item in relevance_items],
                }
            ),
            metadata={
                **relevance_metadata(relevant),
                **(
                    {
                        "cosine_grid": [
                            {
                                "threshold": value,
                                "relevant_group_hashes": sorted(
                                    {
                                        item.evidence_group_hash
                                        for item in relevance_items
                                        if score_passes(item, value, value)
                                    }
                                ),
                                "passed": (
                                    len(
                                        {
                                            item.evidence_group_hash
                                            for item in relevance_items
                                            if score_passes(item, value, value)
                                        }
                                    )
                                    >= required_groups
                                    and role_gate(
                                        {
                                            group
                                            for group in group_roles
                                            if canonical_content_hash(group)
                                            in {
                                                item.evidence_group_hash
                                                for item in relevance_items
                                                if score_passes(item, value, value)
                                            }
                                        }
                                    )
                                ),
                            }
                            for value in self.config.relevance_cosine_grid
                        ]
                    }
                    if self.config.is_v3
                    else {}
                ),
            },
        )
        return RuntimeEvidenceRelevance(
            group_ids=frozenset(relevant),
            assessment=assessment,
            items=tuple(relevance_items),
        )

    async def generate_typed_pmmc_answer(
        self,
        *,
        query: QueryInput,
        frozen: FrozenProgramBinding,
        result: ProgramExecutionResult,
        cell: ExperimentCell,
    ) -> TypedAnswerResult:
        prepared = await self.prepare_typed_pmmc_answer(query=query, result=result, cell=cell)
        answer = await self._generate_prepared_typed_answer(prepared)
        assert isinstance(answer, TypedAnswerResult)
        return answer

    async def prepare_typed_pmmc_answer(
        self,
        *,
        query: QueryInput,
        result: ProgramExecutionResult,
        cell: ExperimentCell,
    ) -> AnswerGenerationInput:
        provider = relevance_answer_input_provider(
            self.answer_inputs, result, direct_images_only=self.config.joint_relevance_enabled,
        )
        prepared = await provider.prepare(
            run_id=self.manifest.query_executor_run_id,
            snapshot_id=self.manifest.snapshot_id,
            query=query,
            result=result,
            variant=QueryAnswerVariant(cell.variant),
        )
        if prepared.metadata.get("answer_contract_version") != "typed-answer-contract-v3":
            raise ValueError("v3 answer preparation returned a historical contract")
        return prepared

    def typed_request_identity(self, prepared: AnswerGenerationInput) -> str:
        return canonical_content_hash(
            {
                "run_id": self.manifest.query_executor_run_id,
                "query_id": prepared.query.query_id,
                "request": prepared.request.model_dump(
                    mode="json", exclude={"request_id", "metadata"}
                ),
                "typed_context": prepared.metadata.get("typed_answer_context"),
                "runtime_relation_contract": prepared.metadata.get("runtime_relation_contract"),
                "grounding_failure_policy": self.config.grounding_failure_policy,
                "contract": self.config.answer_contract_version,
            }
        )

    async def generate_deferred_typed_answer(self, prepared: AnswerGenerationInput) -> dict:
        answer = await self._generate_prepared_typed_answer(prepared, defer_grounding=True)
        if isinstance(answer, TypedAnswerCandidate):
            return {
                "model_call_id": answer.answer_model_call_id,
                "terminal_state": "pending_grounding",
                "candidate": answer.model_dump(mode="json"),
            }
        return {
            "model_call_id": answer.answer_model_call_id,
            "terminal_state": answer.terminal_state,
            "typed_answer_id": answer.typed_answer_id,
        }

    async def ground_prepared_typed_answer(self, prepared: AnswerGenerationInput, reference: dict):
        answer = await self._generate_prepared_typed_answer(prepared, reference=reference)
        assert isinstance(answer, TypedAnswerResult)
        return answer

    async def _generate_prepared_typed_answer(
        self,
        prepared: AnswerGenerationInput,
        *,
        defer_grounding: bool = False,
        reference: dict | None = None,
    ) -> TypedAnswerResult | TypedAnswerCandidate:
        identity = self.typed_request_identity(prepared)
        query = prepared.query
        model_call_id = stable_compiler_id("pmmc-typed-answer-call", identity)
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            await self._fence(uow)
            assert uow.query_executor is not None and uow.model_calls is not None
            existing = await uow.query_executor.get_typed_answer(
                self.manifest.query_executor_run_id,
                query.query_id,
                identity,
            )
            if existing is not None:
                return TypedAnswerResult.model_validate(existing.metadata["typed_result"])
            trace = await uow.model_calls.get(
                reference["model_call_id"] if reference else model_call_id
            )
            if reference is not None and (trace is None or trace.status.value != "success"):
                raise RuntimeError("grounding requires the audited successful answer-phase trace")
        # model_calls.job_id references experiment_jobs, not query-stage leases.
        # The fenced phase journal and typed answer retain query/model-call lineage.
        answer = await self.answer_service.answer_typed(
            prepared,
            model_call_id=model_call_id,
            job_id=None,
            on_grounding_failure=self.config.grounding_failure_policy,
            recovered_trace=trace,
            defer_grounding=defer_grounding,
            allow_generation=reference is None,
        )
        if isinstance(answer, TypedAnswerCandidate):
            return answer
        if reference is not None and reference.get("candidate") is not None:
            candidate = TypedAnswerCandidate.model_validate(reference["candidate"])
            if (
                candidate.run_id != answer.run_id
                or candidate.query_id != answer.query_id
                or candidate.answer_model_call_id != answer.answer_model_call_id
                or candidate.output != answer.ungated_output
            ):
                raise ValueError("grounded answer disagrees with the durable generated candidate")
            calls = {call.model_call_id: call for call in [*candidate.calls, *answer.calls]}
            answer = TypedAnswerResult.model_validate(
                {**answer.model_dump(), "calls": list(calls.values())}
            )
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            await self._fence(uow)
            assert uow.query_executor is not None
            await uow.query_executor.add_typed_answer(self._typed_fact(answer, identity))
        return answer

    async def persist_runtime_phase(self, run_id: str, query_id: str, phase: str, payload: dict):
        self._assert_run(run_id)
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            await self._fence(uow)
            assert uow.query_executor is not None
            await uow.query_executor.add_phase_state(run_id, query_id, phase, payload)

    async def load_runtime_phase(self, run_id: str, query_id: str, phase: str) -> dict:
        self._assert_run(run_id)
        async with ExperimentUnitOfWork(self.reader_sessions, DataScope.RUNTIME) as uow:
            assert uow.query_executor is not None
            value = await uow.query_executor.get_phase_state(run_id, query_id, phase)
        if value is None:
            raise RuntimeError(f"runtime {phase} phase checkpoint is not ready")
        return value

    async def persist_matching_phase(self, state: QueryExecutorGraphState, prepared: dict):
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            await self._fence(uow)
            assert uow.query_executor is not None
            for batch, channels, candidates in self._match_facts(state):
                await uow.query_executor.add_match(batch, channels, candidates)
            await uow.query_executor.add_reranker_assessments(
                list(state["reranker_assessments"].values())
            )
            await uow.query_executor.add_route(state["route_decision"])
            await uow.query_executor.add_phase_state(
                state["run_id"],
                state["query_id"],
                "match",
                {
                    "cells": [fact.model_dump(mode="json") for fact in state["cell_results"]],
                    "prepared": {
                        key: value.model_dump(mode="json") for key, value in prepared.items()
                    },
                    "route": state["route_decision"].model_dump(mode="json"),
                },
            )

    @staticmethod
    def _typed_fact(answer: TypedAnswerResult, identity: str) -> QueryRuntimeTypedAnswerFact:
        payload = runtime_fact_payload(answer)
        return QueryRuntimeTypedAnswerFact(
            typed_answer_id=answer.typed_answer_id,
            query_executor_run_id=answer.run_id,
            query_id=answer.query_id,
            answer_identity_hash=identity,
            answer_model_call_id=answer.answer_model_call_id,
            terminal_state=answer.terminal_state,
            output=answer.output.model_dump(mode="json"),
            ungated_output=answer.ungated_output.model_dump(mode="json"),
            image_items=[item.model_dump(mode="json") for item in answer.image_items],
            grounding=answer.grounding.model_dump(mode="json") if answer.grounding else None,
            content_hash=canonical_content_hash(payload),
            metadata={"typed_result": payload},
        )

    async def run_typed_native_mm_counterfactual(
        self,
        run_id: str,
        query: QueryInput,
    ) -> TypedAnswerResult:
        self._assert_run(run_id)
        source = await self.fallback.load_or_run_typed(query)
        if source.snapshot_id != self.manifest.snapshot_id or source.query_id != query.query_id:
            raise ValueError("typed fallback escaped the frozen source scope")
        # Cell selections reference the original fact. Its grounding and paid
        # model calls remain owned by the baseline run, not duplicated aliases.
        return source

    async def generate_pmmc_answer(
        self,
        *,
        query: QueryInput,
        frozen: FrozenProgramBinding,
        result: ProgramExecutionResult,
        variant: QueryAnswerVariant,
    ) -> FinalAnswerRecord:
        prepared = await self.answer_inputs.prepare(
            run_id=self.manifest.query_executor_run_id,
            snapshot_id=self.manifest.snapshot_id,
            query=query,
            result=result,
            variant=variant,
        )
        async with self.database_semaphore:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.answers is not None
                existing = await uow.answers.get_for_query(
                    self.manifest.query_executor_run_id,
                    query.query_id,
                    variant.value,
                )
        if existing is not None:
            if (
                existing.method != "pmmc"
                or existing.answer_variant != variant.value
                or existing.snapshot_id != self.manifest.snapshot_id
            ):
                raise ValueError("persisted PMMC answer has incompatible lineage")
            return existing

        model_call_id = stable_compiler_id(
            "pmmc-answer-call",
            [
                self.manifest.query_executor_run_id,
                query.query_id,
                variant.value,
                frozen.binding.plan_id,
                prepared.request.request_id,
            ],
        )
        from pmmc.runtime.answering import answer_retry_call_id

        retry_call_id = answer_retry_call_id(prepared.request, model_call_id)
        async with self.database_semaphore:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.model_calls is not None
                base_trace = await uow.model_calls.get(model_call_id)
                retry_trace = await uow.model_calls.get(retry_call_id)

        base_format_error: InvalidRuntimeAnswer | None = None
        answer: FinalAnswerRecord | TypedAnswerResult
        if retry_trace is not None:
            if retry_trace.status.value != "success":
                raise RuntimeError("persisted PMMC answer format retry is terminally failed")
            answer = recover_runtime_answer(prepared, retry_trace)
            answer = answer.model_copy(
                update={
                    "metadata": {
                        **answer.metadata,
                        "format_retried": True,
                        "recovered_retry_model_call_id": retry_call_id,
                    }
                }
            )
        elif base_trace is not None:
            if base_trace.status.value != "success":
                raise RuntimeError("persisted PMMC answer base call is terminally failed")
            try:
                answer = recover_runtime_answer(prepared, base_trace)
            except InvalidRuntimeAnswer as exc:
                base_format_error = exc
                answer = await self.answer_service.answer(
                    prepared,
                    model_call_id=model_call_id,
                    start_attempt=1,
                    prior_attempt_failures=[exc.failures],
                    prior_schema_diagnostics=[exc.schema_diagnostics],
                )
        else:
            answer = await self.answer_service.answer(
                prepared,
                model_call_id=model_call_id,
            )

        if not isinstance(answer, FinalAnswerRecord):
            raise TypeError("historical answer branch returned a typed Step8 answer")
        if base_format_error is not None:
            answer = answer.model_copy(
                update={
                    "metadata": {
                        **answer.metadata,
                        "recovered_base_format_failure_codes": [
                            item.value for item in base_format_error.failures
                        ],
                    }
                }
            )
        async with self.database_semaphore:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.answers is not None
                return await uow.answers.add(answer)

    async def run_native_mm_counterfactual(
        self, run_id: str, query: QueryInput
    ) -> FinalAnswerRecord:
        self._assert_run(run_id)
        return await self.fallback.load_or_run(query)

    async def persist_terminal(self, state: QueryExecutorGraphState) -> None:
        if self.config.is_v3:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                await self._fence(uow)
                assert uow.query_executor is not None
                if "match_grid" in state:
                    for batch, channels, candidates in self._match_facts(state):
                        await uow.query_executor.add_match(batch, channels, candidates)
                    await uow.query_executor.add_reranker_assessments(
                        list(state["reranker_assessments"].values()),
                    )
                if state["route_decision"].query_executor_run_id == state["run_id"]:
                    await uow.query_executor.add_route(state["route_decision"])
                await uow.query_executor.add_fallback_run_link(
                    PmmcFallbackRunLink(
                        query_executor_run_id=state["run_id"],
                        fallback_run_id=self.fallback.fallback_run_id,
                        snapshot_id=self.manifest.snapshot_id,
                        content_hash=canonical_content_hash(
                            [state["run_id"], self.fallback.fallback_run_id,
                             self.manifest.snapshot_id]
                        ),
                    )
                )
                for fact in state["cell_results"]:
                    await uow.query_executor.add_cell_result(fact)
            return
        run_id = state["run_id"]
        self._assert_run(run_id)
        query = state["query"]
        matches = self._match_facts(state)
        fallback = state.get("fallback_answer")
        fallback_projection = (
            _strict_answer_projection(
                run_id=run_id,
                query_id=query.query_id,
                answer_variant=QueryAnswerVariant.NATIVE_MM_COUNTERFACTUAL.value,
                answer=fallback,
            )
            if self.config.is_v2 and fallback is not None
            else None
        )
        async with self.database_semaphore:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.query_executor is not None
                assert uow.answers is not None
                for batch, channels, fused in matches:
                    await uow.query_executor.add_match(batch, channels, fused)
                    if self.config.is_v2:
                        terms = classify_step7b_lexical_terms(
                            query.raw_query,
                            limit=self.config.lexical_term_limit,
                        )
                        lexical_facts = []
                        for ordinal, (term, term_kind) in enumerate(terms):
                            payload = {
                                "match_batch_id": batch.match_batch_id,
                                "ordinal": ordinal,
                                "term_hash": canonical_content_hash(term),
                                "term_kind": term_kind,
                                "exact_boost": term_kind in {"entity", "number", "date"},
                            }
                            lexical_facts.append(
                                QBankLexicalTermFact(
                                    lexical_term_fact_id=stable_compiler_id(
                                        "qbank-lexical-term", payload
                                    ),
                                    match_batch_id=batch.match_batch_id,
                                    query_id=query.query_id,
                                    ordinal=ordinal,
                                    term_hash=payload["term_hash"],
                                    term_kind=term_kind,
                                    exact_boost=payload["exact_boost"],
                                    content_hash=canonical_content_hash(payload),
                                )
                            )
                        await uow.query_executor.add_lexical_terms(lexical_facts)
                if self.config.is_v2:
                    await uow.query_executor.add_reranker_assessments(
                        list(state.get("reranker_assessments", {}).values())
                    )
                await uow.query_executor.add_route(state["route_decision"])
                for execution_reuse_link in state.get("execution_reuse_links", ()):
                    await uow.query_executor.add_execution_reuse(execution_reuse_link)
                for round_id, frozen in sorted(state["frozen_by_round"].items()):
                    await uow.query_executor.add_binding(frozen.binding)
                    assessment = state["sufficiency_by_round"].get(round_id)
                    if assessment is not None:
                        await uow.query_executor.add_sufficiency(assessment)
                    relevance = state.get("relevance_by_round", {}).get(round_id)
                    if relevance is not None:
                        await uow.query_executor.add_relevance(
                            relevance.assessment, relevance.items
                        )
                await uow.query_executor.add_fallback_run_link(
                    PmmcFallbackRunLink(
                        query_executor_run_id=run_id,
                        fallback_run_id=self.fallback.fallback_run_id,
                        snapshot_id=self.manifest.snapshot_id,
                        content_hash=canonical_content_hash(
                            [run_id, self.fallback.fallback_run_id, self.manifest.snapshot_id]
                        ),
                    )
                )
                if fallback is not None:
                    await uow.query_executor.add_fallback_query_link(
                        PmmcFallbackQueryLink(
                            query_executor_run_id=run_id,
                            query_id=query.query_id,
                            fallback_run_id=self.fallback.fallback_run_id,
                            fallback_query_id=fallback.query_id,
                            fallback_answer_id=fallback.answer_id,
                            content_hash=canonical_content_hash(
                                [run_id, query.query_id, fallback.answer_id]
                            ),
                        )
                    )
                if fallback_projection is not None:
                    diagnostics, structured_fallback = fallback_projection
                    for diagnostic in diagnostics:
                        await uow.query_executor.add_answer_diagnostic(diagnostic)
                    await uow.query_executor.add_structured_answer(structured_fallback)
                if self.config.is_v2 and state.get("fallback_abstention"):
                    from pmmc.runtime.answering import validate_abstention_fact

                    source_fact = StructuredAnswerFact.model_validate(state["fallback_abstention"])
                    validate_abstention_fact(source_fact)
                    payload = source_fact.model_dump(
                        mode="json",
                        exclude={
                            "structured_answer_output_id",
                            "content_hash",
                            "created_at",
                        },
                    )
                    payload.update(
                        {
                            "run_id": run_id,
                            "answer_variant": QueryAnswerVariant.NATIVE_MM_COUNTERFACTUAL.value,
                            "metadata": {
                                **source_fact.metadata,
                                "source_run_id": source_fact.run_id,
                                "source_abstention_id": source_fact.structured_answer_output_id,
                                "source_abstention_hash": source_fact.content_hash,
                            },
                        }
                    )
                    await uow.query_executor.add_structured_answer(
                        StructuredAnswerFact.model_validate(
                            {
                                **payload,
                                "structured_answer_output_id": stable_compiler_id(
                                    "answer-abstention", payload
                                ),
                                "content_hash": canonical_content_hash(payload),
                            }
                        )
                    )
                await uow.query_executor.add_fallback(state["fallback_decision"])
                for raw_fact in state.get("pmmc_abstentions", {}).values():
                    abstention_fact = StructuredAnswerFact.model_validate(raw_fact)
                    if (
                        abstention_fact.run_id != run_id
                        or abstention_fact.query_id != query.query_id
                    ):
                        raise ValueError("PMMC abstention escaped query scope")
                    await uow.query_executor.add_structured_answer(abstention_fact)
                for answer in {
                    value.answer_id: value for value in state.get("pmmc_answers", {}).values()
                }.values():
                    await uow.answers.add(answer)
                    if answer.metadata.get("answer_reused") is True:
                        continue
                    format_failures = answer.metadata.get("format_failure_codes", [])
                    if not isinstance(format_failures, list):
                        format_failures = []
                    for attempt_ordinal, raw_codes in enumerate(format_failures):
                        codes = (
                            [
                                AnswerContractFailure(value)
                                for value in raw_codes
                                if isinstance(value, str)
                                and value in {item.value for item in AnswerContractFailure}
                            ]
                            if isinstance(raw_codes, list)
                            else []
                        )
                        diagnostic_payload = {
                            "run_id": run_id,
                            "query_id": query.query_id,
                            "variant": answer.answer_variant,
                            "attempt": attempt_ordinal,
                            "failure_codes": [item.value for item in codes],
                        }
                        await uow.query_executor.add_answer_diagnostic(
                            AnswerFormatDiagnostic(
                                diagnostic_id=stable_compiler_id(
                                    "answer-format-diagnostic", diagnostic_payload
                                ),
                                run_id=run_id,
                                query_id=query.query_id,
                                answer_variant=answer.answer_variant,
                                attempt_ordinal=attempt_ordinal,
                                failure_codes=codes,
                                provider_schema_conformant=False,
                                accepted=False,
                                content_hash=canonical_content_hash(diagnostic_payload),
                            )
                        )
                    accepted_ordinal = len(format_failures)
                    accepted_payload = {
                        "run_id": run_id,
                        "query_id": query.query_id,
                        "variant": answer.answer_variant,
                        "attempt": accepted_ordinal,
                        "answer_id": answer.answer_id,
                    }
                    await uow.query_executor.add_answer_diagnostic(
                        AnswerFormatDiagnostic(
                            diagnostic_id=stable_compiler_id(
                                "answer-format-diagnostic", accepted_payload
                            ),
                            run_id=run_id,
                            query_id=query.query_id,
                            answer_variant=answer.answer_variant,
                            attempt_ordinal=accepted_ordinal,
                            failure_codes=[],
                            provider_schema_conformant=True,
                            accepted=True,
                            content_hash=canonical_content_hash(accepted_payload),
                        )
                    )
                    if answer.model_call_id is None:
                        raise RuntimeError("strict answer lacks model-call lineage")
                    used_aliases = answer.metadata.get("used_evidence_aliases", [])
                    if not isinstance(used_aliases, list):
                        raise RuntimeError("strict answer alias lineage is malformed")
                    await uow.query_executor.add_structured_answer(
                        StructuredAnswerFact(
                            structured_answer_output_id=stable_compiler_id(
                                "structured-answer-output", accepted_payload
                            ),
                            run_id=run_id,
                            query_id=query.query_id,
                            answer_variant=answer.answer_variant,
                            attempt_ordinal=accepted_ordinal,
                            model_call_id=answer.model_call_id,
                            answer_id=answer.answer_id,
                            answer=answer.final_answer,
                            supported=True,
                            used_evidence_aliases=[str(value) for value in used_aliases],
                            abstain_reason=None,
                            parse_status="accepted",
                            provider_schema_conformant=True,
                            finish_reason=(
                                str(answer.metadata["finish_reason"])
                                if answer.metadata.get("finish_reason") is not None
                                else None
                            ),
                            content_hash=canonical_content_hash(
                                {**accepted_payload, "aliases": used_aliases}
                            ),
                        )
                    )
                for answer_reuse_link in state.get("answer_reuse_links", ()):
                    await uow.query_executor.add_answer_reuse(answer_reuse_link)
                for round_id, attempts in state.get("answer_contract_failures", {}).items():
                    variant = f"pmmc_r{round_id}"
                    for attempt_ordinal, raw_codes in enumerate(attempts):
                        codes = [AnswerContractFailure(value) for value in raw_codes]
                        payload = {
                            "run_id": run_id,
                            "query_id": query.query_id,
                            "variant": variant,
                            "attempt": attempt_ordinal,
                            "failure_codes": [item.value for item in codes],
                        }
                        await uow.query_executor.add_answer_diagnostic(
                            AnswerFormatDiagnostic(
                                diagnostic_id=stable_compiler_id(
                                    "answer-format-diagnostic", payload
                                ),
                                run_id=run_id,
                                query_id=query.query_id,
                                answer_variant=variant,
                                attempt_ordinal=attempt_ordinal,
                                failure_codes=codes,
                                provider_schema_conformant=False,
                                accepted=False,
                                content_hash=canonical_content_hash(payload),
                            )
                        )
                active_answer = state.get("active_answer")
                for alias_answer in state.get("evaluation_alias_answers", ()):
                    await uow.answers.add(alias_answer)
                if active_answer is not None:
                    if all(
                        alias.answer_id != active_answer.answer_id
                        for alias in state.get("evaluation_alias_answers", ())
                    ):
                        await uow.answers.add(active_answer)
                await uow.query_executor.add_selection(state["answer_selection"])

    def _match_facts(
        self, state: QueryExecutorGraphState
    ) -> list[
        tuple[
            QBankMatchBatch,
            list[QBankMatchChannelCandidate],
            Sequence[QBankMatchFusedCandidate],
        ]
    ]:
        query = state["query"]
        active_channels = (
            {
                QBankRuntimeChannel.LEXICAL,
                QBankRuntimeChannel.TEXT,
                QBankRuntimeChannel.MULTIMODAL,
            }
            if query.query_image_refs
            else {QBankRuntimeChannel.LEXICAL, QBankRuntimeChannel.TEXT}
        )
        deduplicated: dict[QBankRuntimeChannel, dict[str, RuntimeQBankChannelHit]] = defaultdict(
            dict
        )
        for hit in state["channel_hits"]:
            previous = deduplicated[hit.channel].get(hit.qbank_id)
            if previous is None or hit.raw_score > previous.raw_score:
                deduplicated[hit.channel][hit.qbank_id] = hit
        output: list[
            tuple[
                QBankMatchBatch,
                list[QBankMatchChannelCandidate],
                Sequence[QBankMatchFusedCandidate],
            ]
        ] = []
        for key, result in sorted(
            state["match_grid"].items(),
            key=lambda item: (item[0].question_budget, item[0].round_id),
        ):
            if self.config.is_v3:
                deduplicated = defaultdict(dict)
                for hit in state["channel_hits"]:
                    if hit.projection_round != key.round_id:
                        continue
                    previous = deduplicated[hit.channel].get(hit.qbank_id)
                    if previous is None or hit.raw_score > previous.raw_score:
                        deduplicated[hit.channel][hit.qbank_id] = hit
            channels: list[QBankMatchChannelCandidate] = []
            for channel in sorted(active_channels, key=lambda item: item.value):
                for route_class in QBankRouteClass:
                    eligible = [
                        item
                        for item in deduplicated[channel].values()
                        if item.route_class is route_class
                        and item.generation_rank <= key.question_budget
                        and (self.config.is_v3 or item.accepted_round <= key.round_id)
                    ]
                    eligible.sort(key=lambda item: (-item.raw_score, item.qbank_id))
                    for rank, item in enumerate(eligible[: self.config.match_top_k], start=1):
                        payload = {
                            "match_batch_id": stable_compiler_id(
                                "runtime-qbank-match-batch",
                                [
                                    state["run_id"],
                                    query.query_id,
                                    key.question_budget,
                                    key.round_id,
                                ],
                            ),
                            "channel": channel.value,
                            "route_class": route_class.value,
                            "qbank_id": item.qbank_id,
                            "rank": rank,
                            "raw_score": item.raw_score,
                            "generation_rank": item.generation_rank,
                            "accepted_round": item.accepted_round,
                        }
                        channels.append(
                            QBankMatchChannelCandidate(
                                channel_candidate_id=stable_compiler_id(
                                    "runtime-qbank-channel-candidate", payload
                                ),
                                match_batch_id=payload["match_batch_id"],
                                query_id=query.query_id,
                                qbank_id=item.qbank_id,
                                route_class=route_class,
                                channel=channel,
                                rank=rank,
                                raw_score=item.raw_score,
                                generation_rank=item.generation_rank,
                                accepted_round=item.accepted_round,
                                content_hash=canonical_content_hash(payload),
                            )
                        )
            batch_payload = {
                "run_id": state["run_id"],
                "query_id": query.query_id,
                "projection_id": state["projection_id"],
                "question_budget": key.question_budget,
                "round_id": key.round_id,
                "channel_hashes": [item.content_hash for item in channels],
                "fused_hashes": [item.content_hash for item in result.candidates],
            }
            batch = QBankMatchBatch(
                match_batch_id=stable_compiler_id(
                    "runtime-qbank-match-batch",
                    [state["run_id"], query.query_id, key.question_budget, key.round_id],
                ),
                query_executor_run_id=state["run_id"],
                source_compiler_run_id=self.manifest.source_compiler_run_id,
                routing_projection_id=state["projection_id"],
                snapshot_id=self.manifest.snapshot_id,
                query_id=query.query_id,
                conversation_id=query.conversation_id,
                cutoff_session_id=str(query.available_after_session),
                active_round=key.round_id,
                question_budget=key.question_budget,
                requested_top_k=self.config.match_top_k,
                channel_count=len(active_channels),
                fused_count=len(result.candidates),
                config_hash=self.manifest.config_hash,
                content_hash=canonical_content_hash(batch_payload),
                metadata={
                    "matcher_version": self.config.matcher_version,
                    "lexical_zero_hit_reason": (
                        "no_safe_lexical_terms"
                        if self.config.uses_reranker
                        and not classify_step7b_lexical_terms(
                            query.raw_query,
                            limit=self.config.lexical_term_limit,
                        )
                        else "no_projection_match"
                        if not any(
                            item.channel is QBankRuntimeChannel.LEXICAL
                            for item in state["channel_hits"]
                        )
                        else None
                    ),
                },
            )
            output.append((batch, channels, result.candidates))
        return output


__all__ = [
    "NativeMultimodalFallbackProvider",
    "PostgresNativeMultimodalFallbackProvider",
    "PostgresOnlineQueryExecutorBackend",
]
