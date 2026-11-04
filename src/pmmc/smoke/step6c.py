"""Model-backed Planner and operator conformance for the Step 6C GPU smoke."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from pmmc.compiler.config import CompilerFoundationConfig, load_compiler_config
from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.planner import (
    PlannerProgramPolicy,
    build_planner_requirement_contract,
    materialize_planner_program,
)
from pmmc.compiler.planner_config import PlannerConfig, load_planner_config
from pmmc.compiler.planner_prompt import planner_output_parser, render_planner_request
from pmmc.infrastructure.settings import Settings
from pmmc.memory.operators import (
    LinearProgramExecutor,
    OperatorDataRepository,
    ProgramValidator,
)
from pmmc.models.factory import create_generation_gateway
from pmmc.models.protocols import GenerationGateway
from pmmc.models.registry import ModelRegistry, load_model_registry
from pmmc.retrieval.operators import build_operator_registry
from pmmc.schemas import (
    BaseSchema,
    EvidenceType,
    GenerationRequest,
    Modality,
    ModelContentKind,
    ModelContentPart,
    OperatorControlAction,
    OperatorEvidence,
    OperatorExecutionContext,
    PlannerGenerationEnvelope,
    PlannerGenerationEnvelopeV11,
    QuestionCandidate,
    QuestionReasoningType,
    QuestionType,
    RuntimeMode,
    TraceStatus,
)
from pmmc.schemas.base import NonEmptyStr, NonNegativeInt, PositiveInt
from pmmc.smoke.step5d import write_smoke_result

STEP6C_PLANNER_CONFORMANCE_VERSION = "step6c-planner-conformance-v1"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONFORMANCE_CANDIDATE_ALIAS = "Q01"
_CONFORMANCE_REASON_CODE = "no_relevant_memory"
_CONFORMANCE_INSTRUCTION = (
    "[NON_SCIENTIFIC_CONFORMANCE] For candidate Q01 return exactly one program with "
    "exactly one abstain_if_no_evidence step. Use an empty source_step_ids list, "
    "minimum_evidence_groups=1, required_modalities=[\"text\"], "
    "reason_code=\"no_relevant_memory\", strategy_code=\"refusal\", and confidence=1.0."
)


class Step6CPlannerConformanceError(ValueError):
    """A stage-classified, content-free conformance failure."""

    def __init__(self, *, stage: str, code: str) -> None:
        self.stage = stage
        self.code = code
        super().__init__(f"stage={stage} code={code}")


class _ConformanceAbstainStep(BaseSchema):
    op_name: Literal["abstain_if_no_evidence"]
    source_step_ids: list[NonNegativeInt] = Field(min_length=0, max_length=0)
    minimum_evidence_groups: Literal[1]
    required_modalities: list[Literal["text"]] = Field(min_length=1, max_length=1)
    reason_code: Literal["no_relevant_memory"]


class _ConformanceProgram(BaseSchema):
    candidate_alias: Literal["Q01"]
    steps: list[_ConformanceAbstainStep] = Field(min_length=1, max_length=1)
    strategy_code: Literal["refusal"]
    rationale_summary: NonEmptyStr
    confidence: float = Field(ge=1.0, le=1.0, allow_inf_nan=False)


class _ConformanceEnvelope(BaseSchema):
    programs: list[_ConformanceProgram] = Field(min_length=1, max_length=1)


class Step6CPlannerConformanceResult(BaseSchema):
    """Gold-free proof that a real Planner response reached the real kernel."""

    protocol_version: Literal["step6c-planner-conformance-v1"]
    valid: Literal[True]
    paper_result: Literal[False]
    scientific_sample: Literal[False]
    fixture_kind: Literal["synthetic_refusal"]
    model_alias: NonEmptyStr
    model_revision: NonEmptyStr
    expected_actual_model_id: NonEmptyStr
    actual_model_id: NonEmptyStr
    runtime_id: NonEmptyStr
    runtime_service_id: NonEmptyStr
    replica_index: NonNegativeInt
    replica_count: PositiveInt
    request_id: NonEmptyStr
    request_sha256: NonEmptyStr
    response_sha256: NonEmptyStr
    provider_response_present: Literal[True]
    repair_attempt_count: NonNegativeInt
    parsed_program_count: Literal[1]
    operator_sequence: list[NonEmptyStr]
    operator_registry_hash: NonEmptyStr
    static_valid: Literal[True]
    static_issue_count: Literal[0]
    policy_valid: Literal[True]
    policy_issue_count: Literal[0]
    lineage_valid: Literal[True]
    execution_status: Literal["success"]
    dry_run_control_action: Literal["abstain"]
    dry_run_step_count: Literal[1]
    dry_run_evidence_count: Literal[0]
    content_hash: NonEmptyStr

    @model_validator(mode="after")
    def _validate_identity(self) -> Step6CPlannerConformanceResult:
        if not _COMMIT_PATTERN.fullmatch(self.model_revision):
            raise ValueError("Planner conformance requires a locked 40-character commit")
        if self.actual_model_id != self.expected_actual_model_id:
            raise ValueError("Planner conformance actual model identity does not match")
        if not self.runtime_service_id.startswith(f"{self.runtime_id}:"):
            raise ValueError("Planner conformance service does not belong to its runtime")
        if self.replica_index >= self.replica_count:
            raise ValueError("Planner conformance replica index is outside its pool")
        if self.operator_sequence != ["abstain_if_no_evidence"]:
            raise ValueError("Planner conformance must execute the refusal-only program")
        for value in (
            self.request_sha256,
            self.response_sha256,
            self.operator_registry_hash,
            self.content_hash,
        ):
            if not _HASH_PATTERN.fullmatch(value):
                raise ValueError("Planner conformance hashes must be lowercase SHA256")
        return self


class _ConformanceOperatorData(OperatorDataRepository):
    """Reject accidental data access by the refusal-only conformance program."""

    async def validate_context(self, context: OperatorExecutionContext) -> None:
        if context.query_id is not None or context.collection_bindings or context.query_vectors:
            raise ValueError("Planner conformance context must remain data-free")

    async def validate_explicit_references(
        self,
        context: OperatorExecutionContext,
        *,
        memory_ids: list[str],
        image_ids: list[str],
        session_ids: list[str],
    ) -> None:
        del context
        if memory_ids or image_ids or session_ids:
            raise ValueError("Planner conformance program referenced canonical evidence")

    async def search_keyword(
        self,
        context: OperatorExecutionContext,
        *,
        query: str,
        top_k: int,
        entity_mentions: list[str],
    ) -> list[OperatorEvidence]:
        del context, query, top_k, entity_mentions
        raise RuntimeError("Planner conformance unexpectedly invoked keyword retrieval")

    async def search_embedding(
        self,
        context: OperatorExecutionContext,
        *,
        collection_id: str,
        vector: list[float],
        top_k: int,
        evidence_type: EvidenceType,
    ) -> list[OperatorEvidence]:
        del context, collection_id, vector, top_k, evidence_type
        raise RuntimeError("Planner conformance unexpectedly invoked embedding retrieval")

    async def resolve_image_text_pairs(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
    ) -> list[OperatorEvidence]:
        del context, image_ids
        raise RuntimeError("Planner conformance unexpectedly resolved image-text pairs")

    async def context_around_images(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
        radius: int,
    ) -> list[OperatorEvidence]:
        del context, image_ids, radius
        raise RuntimeError("Planner conformance unexpectedly expanded image context")

    async def expand_temporal_neighbors(
        self,
        context: OperatorExecutionContext,
        memory_ids: list[str],
        radius: int,
    ) -> list[OperatorEvidence]:
        del context, memory_ids, radius
        raise RuntimeError("Planner conformance unexpectedly expanded temporal context")

    async def inspect_images(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
    ) -> list[OperatorEvidence]:
        del context, image_ids
        raise RuntimeError("Planner conformance unexpectedly inspected an image")


def _content_hash(value: Step6CPlannerConformanceResult) -> str:
    return canonical_content_hash(value.model_dump(mode="json", exclude={"content_hash"}))


def verify_step6c_planner_conformance_payload(
    value: object,
    *,
    expected_replica_count: int | None = None,
) -> Step6CPlannerConformanceResult:
    if isinstance(value, (str, bytes, bytearray)):
        value = json.loads(value)
    result = Step6CPlannerConformanceResult.model_validate(value)
    if _content_hash(result) != result.content_hash:
        raise ValueError("Planner conformance content hash does not match")
    if expected_replica_count is not None and result.replica_count != expected_replica_count:
        raise ValueError("Planner conformance used the wrong replica count")
    return result


def verify_step6c_planner_conformance(
    path: Path,
    *,
    expected_replica_count: int | None = None,
) -> Step6CPlannerConformanceResult:
    return verify_step6c_planner_conformance_payload(
        path.read_text(encoding="utf-8"),
        expected_replica_count=expected_replica_count,
    )


def _render_conformance_request(
    *,
    request_id: str,
    config: PlannerConfig,
    candidate: QuestionCandidate,
    timeout_seconds: float,
) -> GenerationRequest:
    """Render the normal Planner request with a smoke-only refusal constraint."""

    requirement = build_planner_requirement_contract(
        run_id=candidate.run_id,
        compiler_unit_id=str(candidate.compiler_unit_id),
        candidate=candidate,
    )
    request = render_planner_request(
        request_id=request_id,
        config=config,
        candidates=[(_CONFORMANCE_CANDIDATE_ALIAS, candidate)],
        evidence_views={_CONFORMANCE_CANDIDATE_ALIAS: []},
        requirement_contracts={
            _CONFORMANCE_CANDIDATE_ALIAS: requirement
        }
        if config.is_v11
        else None,
        timeout_seconds=timeout_seconds,
    )
    messages = list(request.messages)
    user_message = messages[-1]
    messages[-1] = user_message.model_copy(
        update={
            "parts": [
                *user_message.parts,
                ModelContentPart(
                    kind=ModelContentKind.TEXT,
                    text=_CONFORMANCE_INSTRUCTION,
                ),
            ]
        }
    )
    schema = _ConformanceEnvelope.model_json_schema()
    return request.model_copy(
        update={
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "pmmc_step6c_planner_conformance_v1",
                    "strict": True,
                    "schema": schema,
                },
            },
            "metadata": {
                **request.metadata,
                "conformance_protocol": STEP6C_PLANNER_CONFORMANCE_VERSION,
                "response_schema_hash": canonical_content_hash(schema),
                "scientific_sample": False,
            },
        }
    )


def _refusal_candidate() -> QuestionCandidate:
    question = "What color was the umbrella I brought on my trip?"
    return QuestionCandidate(
        qbank_id="step6c-conformance-candidate",
        run_id="step6c-conformance-run",
        snapshot_id="step6c-conformance-snapshot",
        compiler_unit_id="step6c-conformance-unit",
        question_hash=hashlib.sha256(question.encode("utf-8")).hexdigest(),
        conversation_id="step6c-conformance-conversation",
        created_after_session="step6c-conformance-session",
        canonical_question=question,
        question_type=QuestionType.ANSWER_REFUSAL,
        reasoning_type=QuestionReasoningType.REFUSAL,
        expected_answer=None,
        source_memory_ids=[],
        evidence_ids=[],
        required_modalities=[Modality.TEXT],
        evidence_requirement=[],
        hop_count=1,
        generation_rank=1,
        questioner_confidence=1.0,
        candidate_content_hash=canonical_content_hash([question, "refusal"]),
    )


async def execute_step6c_planner_conformance(
    *,
    settings: Settings,
    registry: ModelRegistry,
    gateway: GenerationGateway,
    planner_config: PlannerConfig,
    foundation_config: CompilerFoundationConfig,
    expected_replica_count: int,
) -> Step6CPlannerConformanceResult:
    if settings.runtime_mode is not RuntimeMode.MANAGED_LOCAL:
        raise Step6CPlannerConformanceError(
            stage="preflight", code="managed_runtime_required"
        )
    if settings.active_runtime_id is None:
        raise Step6CPlannerConformanceError(
            stage="preflight", code="active_runtime_identity_missing"
        )
    if foundation_config.operator_set_version not in {
        "pmmc-operators-v2",
        "pmmc-operators-v3",
    }:
        raise Step6CPlannerConformanceError(
            stage="preflight", code="operator_set_v2_or_v3_required"
        )
    if planner_config.operator_set_version != foundation_config.operator_set_version:
        raise Step6CPlannerConformanceError(
            stage="preflight", code="operator_identity_mismatch"
        )
    entry = registry.get(planner_config.planner_model_alias)
    pool = settings.active_runtime_endpoint_pools.get(entry.alias, [])
    if len(pool) != expected_replica_count:
        raise Step6CPlannerConformanceError(
            stage="preflight", code="endpoint_pool_size_mismatch"
        )

    candidate = _refusal_candidate()
    request_id = stable_compiler_id(
        "planner-conformance",
        [settings.active_runtime_id, entry.alias, planner_config.protocol_version],
    )
    request = _render_conformance_request(
        request_id=request_id,
        config=planner_config,
        candidate=candidate,
        timeout_seconds=settings.model_request_timeout_seconds,
    )
    try:
        result = await gateway.generate(request)
    except Exception as exc:
        raise Step6CPlannerConformanceError(
            stage="provider", code="generation_request_failed"
        ) from exc
    if result.actual_model_id != entry.expected_actual_model_id:
        raise Step6CPlannerConformanceError(
            stage="provider", code="actual_model_identity_mismatch"
        )
    try:
        parsed = planner_output_parser(
            expected_aliases=[_CONFORMANCE_CANDIDATE_ALIAS],
            config=planner_config,
        )(result)
        invalid = parsed.value.get("invalid_candidate_aliases", [])
        envelope_model = (
            PlannerGenerationEnvelopeV11
            if planner_config.is_v11
            else PlannerGenerationEnvelope
        )
        envelope = envelope_model.model_validate(
            {"programs": parsed.value.get("programs", [])}
        )
    except Exception as exc:
        raise Step6CPlannerConformanceError(
            stage="parse", code="planner_response_parse_failed"
        ) from exc
    if invalid or len(envelope.programs) != 1:
        raise Step6CPlannerConformanceError(
            stage="parse", code="planner_response_contract_invalid"
        )

    runtime_service_id = result.metadata.get("runtime_service_id")
    replica_index = result.metadata.get("replica_index")
    replica_count = result.metadata.get("replica_count")
    if not isinstance(runtime_service_id, str):
        raise Step6CPlannerConformanceError(
            stage="lineage", code="runtime_service_id_missing"
        )
    if not isinstance(replica_index, int) or isinstance(replica_index, bool):
        raise Step6CPlannerConformanceError(
            stage="lineage", code="replica_index_missing"
        )
    if not isinstance(replica_count, int) or isinstance(replica_count, bool):
        raise Step6CPlannerConformanceError(
            stage="lineage", code="replica_count_missing"
        )
    bindings = {
        str(item.get("runtime_service_id")): int(item.get("replica_index", -1)) for item in pool
    }
    if bindings.get(runtime_service_id) != replica_index:
        raise Step6CPlannerConformanceError(
            stage="lineage", code="runtime_service_outside_endpoint_pool"
        )
    if replica_count != expected_replica_count:
        raise Step6CPlannerConformanceError(
            stage="lineage", code="response_replica_count_mismatch"
        )

    program = materialize_planner_program(
        run_id=candidate.run_id,
        snapshot_id=str(candidate.snapshot_id),
        compiler_unit_id=str(candidate.compiler_unit_id),
        producer_model_call_id=request_id,
        candidate=candidate,
        draft=envelope.programs[0],
        config=planner_config,
    )
    requirement = build_planner_requirement_contract(
        run_id=candidate.run_id,
        compiler_unit_id=str(candidate.compiler_unit_id),
        candidate=candidate,
    )
    data = _ConformanceOperatorData()
    operator_registry = build_operator_registry(data, foundation_config)
    static_result = ProgramValidator(operator_registry, foundation_config).validate(program)
    if not static_result.valid:
        codes = ",".join(sorted(item.code for item in static_result.issues))
        raise Step6CPlannerConformanceError(
            stage="static_validation", code=f"static_rejected:{codes}"
        )
    policy_result = PlannerProgramPolicy(
        planner_config,
        registry=operator_registry,
    ).validate(
        program,
        candidate,
        requirement if planner_config.is_v11 else None,
    )
    if not policy_result.valid:
        codes = ",".join(sorted(item.code for item in policy_result.issues))
        raise Step6CPlannerConformanceError(
            stage="policy_validation", code=f"policy_rejected:{codes}"
        )

    try:
        execution = await LinearProgramExecutor(
            operator_registry,
            ProgramValidator(operator_registry, foundation_config),
            data,
            foundation_config,
        ).execute(
            execution_id=stable_compiler_id("planner-conformance-execution", [program.plan_id]),
            program=program,
            context=OperatorExecutionContext(
                run_id=candidate.run_id,
                source_compiler_run_id=candidate.run_id,
                snapshot_id=str(candidate.snapshot_id),
                conversation_id=candidate.conversation_id,
                cutoff_session_id=candidate.created_after_session,
                compiler_unit_id=candidate.compiler_unit_id,
                question=candidate.canonical_question,
                required_modalities=candidate.required_modalities,
            ),
        )
    except Exception as exc:
        raise Step6CPlannerConformanceError(
            stage="dry_run", code="operator_execution_failed"
        ) from exc
    if execution.trace.status is not TraceStatus.SUCCESS:
        raise Step6CPlannerConformanceError(
            stage="dry_run", code="execution_status_not_success"
        )
    if execution.trace.control_action is not OperatorControlAction.ABSTAIN:
        raise Step6CPlannerConformanceError(
            stage="dry_run", code="abstain_control_missing"
        )
    if len(execution.steps) != 1 or execution.evidence:
        raise Step6CPlannerConformanceError(
            stage="dry_run", code="unexpected_execution_trace"
        )

    payload = {
        "protocol_version": STEP6C_PLANNER_CONFORMANCE_VERSION,
        "valid": True,
        "paper_result": False,
        "scientific_sample": False,
        "fixture_kind": "synthetic_refusal",
        "model_alias": entry.alias,
        "model_revision": entry.model_revision,
        "expected_actual_model_id": entry.expected_actual_model_id,
        "actual_model_id": result.actual_model_id,
        "runtime_id": settings.active_runtime_id,
        "runtime_service_id": runtime_service_id,
        "replica_index": replica_index,
        "replica_count": replica_count,
        "request_id": request.request_id,
        "request_sha256": canonical_content_hash(request.model_dump(mode="json")),
        "response_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
        "provider_response_present": bool(result.provider_request_id),
        "repair_attempt_count": parsed.repair_attempt_count,
        "parsed_program_count": 1,
        "operator_sequence": [step.op_name for step in program.operator_sequence],
        "operator_registry_hash": operator_registry.resolved_hash,
        "static_valid": True,
        "static_issue_count": 0,
        "policy_valid": True,
        "policy_issue_count": 0,
        "lineage_valid": True,
        "execution_status": execution.trace.status.value,
        "dry_run_control_action": execution.trace.control_action.value,
        "dry_run_step_count": len(execution.steps),
        "dry_run_evidence_count": len(execution.evidence),
    }
    pending = Step6CPlannerConformanceResult.model_validate({**payload, "content_hash": "0" * 64})
    return verify_step6c_planner_conformance_payload(
        {**payload, "content_hash": _content_hash(pending)},
        expected_replica_count=expected_replica_count,
    )


async def run_step6c_planner_conformance(
    *,
    settings: Settings,
    planner_config_path: Path,
    foundation_config_path: Path,
    expected_replica_count: int,
    result_path: Path,
) -> Step6CPlannerConformanceResult:
    registry = load_model_registry(settings.resolved_model_registry_path)
    planner_config = load_planner_config(planner_config_path)
    foundation_config = load_compiler_config(foundation_config_path)
    gateway = create_generation_gateway(
        settings,
        registry,
        planner_config.planner_model_alias,
    )
    try:
        result = await execute_step6c_planner_conformance(
            settings=settings,
            registry=registry,
            gateway=gateway,
            planner_config=planner_config,
            foundation_config=foundation_config,
            expected_replica_count=expected_replica_count,
        )
    except Step6CPlannerConformanceError as exc:
        write_smoke_result(
            result_path,
            {
                "protocol_version": STEP6C_PLANNER_CONFORMANCE_VERSION,
                "valid": False,
                "paper_result": False,
                "scientific_sample": False,
                "fixture_kind": "synthetic_refusal",
                "failure_stage": exc.stage,
                "failure_code": exc.code,
                "error_type": type(exc).__name__,
            },
        )
        raise
    write_smoke_result(result_path, result.model_dump(mode="json"))
    return result


__all__ = [
    "STEP6C_PLANNER_CONFORMANCE_VERSION",
    "Step6CPlannerConformanceError",
    "Step6CPlannerConformanceResult",
    "execute_step6c_planner_conformance",
    "run_step6c_planner_conformance",
    "verify_step6c_planner_conformance",
    "verify_step6c_planner_conformance_payload",
]
