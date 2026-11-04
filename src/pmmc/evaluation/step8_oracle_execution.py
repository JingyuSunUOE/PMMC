"""Evaluator-owned execution and answering of frozen, gold-free Oracle@50 programs.

No runtime repository writes are made. The caller supplies read-only operator and
memory readers, evaluator model services, and a private evaluator artifact root.
The branch deliberately executes all returned evidence, with no match/relevance
gate and no RAG fallback. It is not the deployed PMMC policy.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, cast

from pmmc.compiler.ids import (
    canonical_content_hash,
    memory_program_content_hash,
    stable_compiler_id,
)
from pmmc.evaluation.step8_qbank_semantic import _SingleDispatchGateway
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.memory.operators import LinearProgramExecutor, ProgramExecutionResult
from pmmc.models.execution import AuditedGenerationResult, ModelCallContext, ModelExecutionService
from pmmc.runtime.answer_context import PmmcEvidenceAnswerInputProvider
from pmmc.runtime.answering import SharedAnswerGenerationService
from pmmc.runtime.grounding import AuditedGroundingService, load_step8_prompt
from pmmc.runtime.relevance_grid import (
    bind_runtime_relation_contract,
    requires_independent_relation,
)
from pmmc.schemas import (
    DataScope,
    GenerationResult,
    MemoryProgram,
    ModelCallTrace,
    OperatorExecutionContext,
    OperatorExecutionPurpose,
    PlannerRequirementContract,
    ProgramExecutionEvidenceRef,
    ProgramExecutionStepTrace,
    ProgramExecutionTrace,
    QueryAnswerVariant,
    QueryInput,
    TraceStatus,
)
from pmmc.schemas.answer_output import TypedAnswerResult
from pmmc.schemas.query_executor import runtime_fact_payload


def require_evaluator(scope: DataScope) -> None:
    if scope is not DataScope.EVALUATION:
        raise ValueError("oracle_evaluator_boundary_required")


@dataclass(frozen=True)
class OracleProgramInput:
    """Use get_runtime_program_v3, never the private Planner/support projection."""

    program: MemoryProgram
    requirement: PlannerRequirementContract
    predicted_question: str
    source_program_hash: str
    collection_bindings: dict[str, str]
    query_vectors: dict[str, list[float]]

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "program": runtime_fact_payload(self.program),
            "requirement": runtime_fact_payload(self.requirement),
            "predicted_question": self.predicted_question,
            "source_program_hash": self.source_program_hash,
            "collection_bindings": self.collection_bindings,
            "query_vectors": self.query_vectors,
        }

    def validate(self, *, compiler_run_id: str, snapshot_id: str, qbank_id: str, plan_id: str):
        program = self.program
        if (
            (program.run_id, program.snapshot_id, program.qbank_id, program.plan_id)
            != (compiler_run_id, snapshot_id, qbank_id, plan_id)
            or program.operator_set_version != "pmmc-operators-v7"
            or not program.compiler_unit_id
            or not program.accepted
            or program.program_hash != memory_program_content_hash(program)
            or program.expected_evidence_ids
            or program.metadata.get("runtime_safe_projection") is not True
            or self.requirement.qbank_id != qbank_id
            or self.requirement.run_id != compiler_run_id
            or not self.source_program_hash
        ):
            raise ValueError("oracle_program_projection_mismatch")
        _gold_free(self.payload)


def _gold_free(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                str(key).lower()
                in {
                    "gold",
                    "gold_answer",
                    "expected_answer",
                    "expected_evidence_ids",
                    "source_memory_ids",
                    "source_support",
                    "support_refs",
                }
                and child
            ):
                raise ValueError("oracle_generation_input_contains_private_support")
            _gold_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _gold_free(child)


class OracleJournal:
    """Immutable stage receipts. An uncertain dispatch is never repeated implicitly.

    Parent GC must retain the returned report and all receipt CAS references.
    Interrupted dispatched calls require audited reconciliation, not deleting intent.
    """

    def __init__(self, store: CampaignArtifactStore, *, scope: DataScope):
        require_evaluator(scope)
        self.store = store
        self.scope = scope
        self.directory = store.root / "_store/oracle-stages"
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    @asynccontextmanager
    async def lock(self, key: str):
        fd = os.open(
            self.directory / (key + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.05)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    async def perform(
        self,
        identity: dict[str, Any],
        operation: Callable[[], Awaitable[dict[str, Any]]],
        *,
        retry_incomplete: bool = False,
    ) -> dict[str, Any]:
        key = canonical_content_hash(identity)
        async with self.lock(key):
            receipt = self.directory / (key + ".json")
            if receipt.exists():
                import json

                saved = json.loads(receipt.read_bytes())
                if saved["identity"] != identity:
                    raise ValueError("oracle_journal_identity_conflict")
                return self.store.load_json(saved["sha256"])
            intent = self.directory / (key + ".intent.json")
            if intent.exists() and not retry_incomplete:
                raise ValueError("oracle_stage_requires_reconciliation")
            write_json(intent, identity, immutable=True)
            result = await operation()
            digest = self.store.put_json(result)
            write_json(receipt, {"identity": identity, "sha256": digest}, immutable=True)
            return result


class OracleModelExecution:
    """Bind existing answer/grounding services to evaluator-only audit ownership."""

    def __init__(
        self,
        execution: ModelExecutionService,
        *,
        evaluation_run_id: str,
        source_run_id: str,
        journal: OracleJournal,
    ):
        require_evaluator(execution.data_scope)
        require_evaluator(journal.scope)
        if execution.max_attempts != 1 or not evaluation_run_id or not source_run_id:
            raise ValueError("oracle_requires_bounded_registered_evaluator")
        self.execution = execution
        self.evaluation_run_id = evaluation_run_id
        self.source_run_id = source_run_id
        self.journal = journal
        self.data_scope = DataScope.EVALUATION

    async def generate_audited(self, gateway, request, context: ModelCallContext, **kwargs):
        if context.run_id != self.source_run_id or context.evaluation_run_id not in {
            None,
            self.evaluation_run_id,
        }:
            raise ValueError("oracle_model_owner_mismatch")
        context = replace(context, evaluation_run_id=self.evaluation_run_id)
        identity = {
            "stage": "oracle-model-call-v1",
            "evaluation_run_id": self.evaluation_run_id,
            "call_id": context.model_call_id,
            "request_hash": canonical_content_hash(request.model_dump(mode="json")),
            "model_hash": self.execution.registry.resolved_hash_for_aliases([request.model_alias]),
        }

        async def dispatch():
            audited = await self.execution.generate_audited(
                _SingleDispatchGateway(gateway, lambda: None),
                request,
                context,
                **kwargs,
            )
            return {
                "result": audited.result.model_dump(mode="json"),
                "trace": audited.trace.model_dump(mode="json"),
            }

        saved = await self.journal.perform(identity, dispatch)
        trace = ModelCallTrace.model_validate(saved["trace"])
        result = GenerationResult.model_validate(saved["result"])
        if (
            trace.evaluation_run_id != self.evaluation_run_id
            or trace.run_id != context.run_id
            or trace.query_id != context.query_id
            or trace.model_call_id != context.model_call_id
            or trace.status is not TraceStatus.SUCCESS
            or trace.model_alias != request.model_alias
            or trace.actual_model_id
            != self.execution.registry.get(request.model_alias).expected_actual_model_id
            or result.actual_model_id != trace.actual_model_id
        ):
            raise ValueError("oracle_model_trace_mismatch")
        return AuditedGenerationResult(result, trace)


class OracleProgramRunner:
    """Concrete reusable kernel + evidence renderer + bounded answer/grounder path."""

    def __init__(
        self,
        *,
        scope: DataScope,
        executor: LinearProgramExecutor,
        answer_inputs: PmmcEvidenceAnswerInputProvider,
        answer_service: SharedAnswerGenerationService,
        journal: OracleJournal,
        protocol_identity: dict[str, Any],
    ):
        require_evaluator(scope)
        execution = cast(Any, answer_service.execution)
        grounding = answer_service.grounding
        if (
            not isinstance(execution, OracleModelExecution)
            or not isinstance(grounding, AuditedGroundingService)
            or grounding.execution is not execution
            or execution.journal is not journal
            or answer_inputs.answer_contract_version != "typed-answer-contract-v3"
            or not protocol_identity
        ):
            raise ValueError("oracle_evaluator_services_required")
        self.executor = executor
        self.answer_inputs = answer_inputs
        self.answer_service = answer_service
        self.journal = journal
        self.evaluation_run_id = execution.evaluation_run_id
        self.source_run_id = execution.source_run_id
        self.identity = {
            "protocol": protocol_identity,
            "foundation": executor.config.model_dump(mode="json"),
            "answer_model": execution.execution.registry.resolved_hash_for_aliases(
                [answer_inputs.answer_model_alias, grounding.model_alias]
            ),
            "grounding_prompt_hash": canonical_content_hash(
                load_step8_prompt("answer_grounding_v1.txt")
            ),
            "answer_prompt_hash": answer_inputs.instruction_hash,
            "total_context_tokens": answer_inputs.total_context_tokens,
            "reserved_output_tokens": answer_inputs.reserved_output_tokens,
            "allow_memory_raw_images": answer_inputs.allow_memory_raw_images,
            "policy": "all-executed-evidence-no-relevance-gate-no-fallback-v1",
        }

    async def run(
        self,
        *,
        query: QueryInput,
        snapshot_id: str,
        candidate_identity: dict[str, Any],
        source: OracleProgramInput,
    ) -> dict[str, Any]:
        identity = {
            "evaluation_run_id": self.evaluation_run_id,
            "query_id": query.query_id,
            "candidate": candidate_identity,
            "source": source.payload,
            "runner": self.identity,
        }
        key = canonical_content_hash(identity)
        execution_id = stable_compiler_id("oracle-program-execution", key)
        query = query.model_copy(
            update={
                "metadata": {},
                "query_image_refs": [
                    image.model_copy(update={"metadata": {}, "caption": None})
                    for image in query.query_image_refs
                ],
            }
        )

        async def execute():
            context = OperatorExecutionContext(
                run_id=self.source_run_id,
                source_compiler_run_id=source.program.run_id,
                snapshot_id=snapshot_id,
                conversation_id=query.conversation_id,
                cutoff_session_id=query.available_after_session,
                compiler_unit_id=source.program.compiler_unit_id,
                query_id=query.query_id,
                question=source.predicted_question,
                query_image_ids=[image.image_id for image in query.query_image_refs],
                required_modalities=source.requirement.required_modalities,
                collection_bindings=source.collection_bindings,
                query_vectors=source.query_vectors,
                constraints={"source_program_hash": source.source_program_hash},
                metadata={"evaluator_branch": "oracle_at_50"},
            )
            result = await self.executor.execute(
                execution_id=execution_id,
                program=source.program,
                context=context,
                purpose=OperatorExecutionPurpose.QUERY_TIME,
            )
            return {
                "trace": result.trace.model_dump(mode="json"),
                "steps": [step.model_dump(mode="json") for step in result.steps],
                "evidence": [ref.model_dump(mode="json") for ref in result.evidence],
            }

        fact = await self.journal.perform(
            {"stage": "oracle-execute", "key": key},
            execute,
            retry_incomplete=True,
        )
        result = ProgramExecutionResult(
            ProgramExecutionTrace.model_validate(fact["trace"]),
            tuple(ProgramExecutionStepTrace.model_validate(step) for step in fact["steps"]),
            tuple(ProgramExecutionEvidenceRef.model_validate(ref) for ref in fact["evidence"]),
        )
        if (
            result.trace.execution_id != execution_id
            or result.trace.plan_id != source.program.plan_id
            or result.trace.query_id != query.query_id
            or result.trace.run_id != self.source_run_id
            or result.trace.snapshot_id != snapshot_id
        ):
            raise ValueError("oracle_execution_scope_mismatch")
        if result.trace.status is not TraceStatus.SUCCESS:
            return {
                "status": "incomplete",
                "error_code": "oracle_program_execution_failed",
                "execution": fact,
                "answer": None,
            }
        prepared = await self.answer_inputs.prepare(
            run_id=self.source_run_id,
            snapshot_id=snapshot_id,
            query=query,
            result=result,
            variant=QueryAnswerVariant.PMMC_R0,
        )
        prepared = prepared.model_copy(update={"answer_variant": "oracle-at-50-" + key})
        prepared = bind_runtime_relation_contract(
            prepared,
            result=result,
            minimum_groups=source.requirement.minimum_evidence_groups,
            relation_required=requires_independent_relation(source.requirement),
        )
        request_hash = canonical_content_hash(prepared.model_dump(mode="json"))

        async def answer():
            value = await self.answer_service.answer_typed(
                prepared,
                model_call_id=stable_compiler_id("oracle-answer-call", [key, request_hash]),
                on_grounding_failure="abstain",
            )
            if not isinstance(value, TypedAnswerResult):
                raise ValueError("oracle_grounded_answer_required")
            return value.model_dump(mode="json")

        try:
            generated = await self.journal.perform(
                {"stage": "oracle-answer", "key": key, "request_hash": request_hash},
                answer,
                retry_incomplete=True,
            )
        except Exception as exc:
            return {
                "status": "incomplete",
                "error_code": "oracle_answer_incomplete",
                "error_type": type(exc).__name__,
                "execution": fact,
                "prepared": prepared.model_dump(mode="json"),
                "answer": None,
            }
        return {
            "status": "completed",
            "execution": fact,
            "prepared": prepared.model_dump(mode="json"),
            "answer": generated,
        }
