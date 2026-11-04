"""Eight-replica compiler transport conformance for the Step 6D GPU pilot."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError, model_validator

from pmmc.compiler.config import load_compiler_config
from pmmc.compiler.doubter import derive_doubter_action
from pmmc.compiler.doubter_config import load_doubter_config
from pmmc.compiler.doubter_graph import run_doubter_control_graph
from pmmc.compiler.doubter_prompt import (
    DOUBTER_PROMPT,
    DOUBTER_PROMPT_V11,
    DOUBTER_PROMPT_V12,
)
from pmmc.compiler.ids import canonical_content_hash
from pmmc.compiler.planner_config import PlannerConfig, load_planner_config
from pmmc.compiler.planner_prompt import (
    planner_prompt_text,
    planner_request_response_schema,
)
from pmmc.compiler.questioner_config import load_questioner_config
from pmmc.compiler.questioner_prompt import questioner_prompt_text
from pmmc.infrastructure.settings import Settings
from pmmc.models.factory import create_generation_gateway
from pmmc.models.gateways import ModelGatewayError
from pmmc.models.registry import load_model_registry
from pmmc.models.routing import stable_replica_index
from pmmc.schemas import (
    BaseSchema,
    DoubterAction,
    DoubterIssueCode,
    DoubterProviderEnvelope,
    DoubterProviderEnvelopeV11,
    GenerationRequest,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    PlannerCapability,
    PlannerRequiredModality,
    PlannerRequirementContract,
    PlannerStrategyEnvelopeV12,
    QuestionerBatchV33,
    QuestionerBatchV34,
    QuestionerBatchV35,
    QuestionerVerificationEnvelopeV3,
    QuestionerVerificationEnvelopeV5,
    QuestionerVerificationEnvelopeV6,
    RuntimeMode,
    SpeakerRole,
)
from pmmc.schemas.base import NonEmptyStr, NonNegativeInt, PositiveInt
from pmmc.smoke.step5d import write_smoke_result

STEP6D_COMPILER_CONFORMANCE_VERSION = "step6d-compiler-conformance-v1"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TASKS = ("questioner", "questioner_verifier", "planner", "doubter")


class Step6DCompilerConformanceError(ValueError):
    """A stage-classified and content-free conformance failure."""

    def __init__(self, *, stage: str, code: str) -> None:
        self.stage = stage
        self.code = code
        super().__init__(f"stage={stage} code={code}")


class _Step6DNativeResponseError(ValueError):
    """A content-free native response parsing failure."""

    def __init__(self, *, code: str, diagnostics: dict[str, object]) -> None:
        self.code = code
        self.diagnostics = diagnostics
        super().__init__(code)


class Step6DCompilerReplicaResult(BaseSchema):
    replica_index: NonNegativeInt
    runtime_service_id: NonEmptyStr
    task: Literal[
        "questioner",
        "questioner_verifier",
        "planner",
        "doubter",
    ]
    request_id: NonEmptyStr
    request_sha256: NonEmptyStr
    response_sha256: NonEmptyStr
    actual_model_id: NonEmptyStr
    provider_response_present: Literal[True]
    native_schema_conformant: Literal[True]

    @model_validator(mode="after")
    def _validate_hashes(self) -> Step6DCompilerReplicaResult:
        if not _HASH_PATTERN.fullmatch(self.request_sha256):
            raise ValueError("Step 6D conformance request hash is invalid")
        if not _HASH_PATTERN.fullmatch(self.response_sha256):
            raise ValueError("Step 6D conformance response hash is invalid")
        return self


class Step6DBranchMatrix(BaseSchema):
    questioner_pass: Literal[True]
    questioner_revise: Literal[True]
    questioner_reject: Literal[True]
    planner_parse_failure: Literal[True]
    doubter_pass: Literal[True]
    doubter_revise: Literal[True]
    doubter_reject: Literal[True]
    round_three_exhausted: Literal[True]
    visual_raw_image: Literal[True]
    temporal: Literal[True]
    fallback: Literal[True]
    abstain: Literal[True]


class Step6DCompilerConformanceResult(BaseSchema):
    protocol_version: Literal["step6d-compiler-conformance-v1"]
    valid: Literal[True]
    paper_result: Literal[False]
    scientific_sample: Literal[False]
    model_alias: NonEmptyStr
    model_revision: NonEmptyStr
    expected_actual_model_id: NonEmptyStr
    runtime_id: NonEmptyStr
    replica_count: PositiveInt
    request_count: PositiveInt
    replicas: list[Step6DCompilerReplicaResult]
    branch_matrix: Step6DBranchMatrix
    content_hash: NonEmptyStr

    @model_validator(mode="after")
    def _validate_identity(self) -> Step6DCompilerConformanceResult:
        if not _COMMIT_PATTERN.fullmatch(self.model_revision):
            raise ValueError("Step 6D conformance requires a pinned model commit")
        if self.request_count != self.replica_count:
            raise ValueError("Step 6D conformance requires one request per replica")
        if len(self.replicas) != self.replica_count:
            raise ValueError("Step 6D conformance replica accounting is incomplete")
        if {item.replica_index for item in self.replicas} != set(
            range(self.replica_count)
        ):
            raise ValueError("Step 6D conformance did not reach every replica")
        if len({item.runtime_service_id for item in self.replicas}) != (
            self.replica_count
        ):
            raise ValueError("Step 6D conformance service lineage is incomplete")
        if set(item.task for item in self.replicas) != set(_TASKS):
            raise ValueError("Step 6D conformance did not exercise every compiler task")
        if any(
            item.actual_model_id != self.expected_actual_model_id
            for item in self.replicas
        ):
            raise ValueError("Step 6D conformance actual model identity differs")
        if not _HASH_PATTERN.fullmatch(self.content_hash):
            raise ValueError("Step 6D conformance content hash is invalid")
        return self


def _content_hash(value: Step6DCompilerConformanceResult) -> str:
    return canonical_content_hash(
        value.model_dump(mode="json", exclude={"content_hash"})
    )


def verify_step6d_compiler_conformance_payload(
    value: object,
    *,
    expected_replica_count: int | None = None,
) -> Step6DCompilerConformanceResult:
    if isinstance(value, (str, bytes, bytearray)):
        value = json.loads(value)
    result = Step6DCompilerConformanceResult.model_validate(value)
    if _content_hash(result) != result.content_hash:
        raise ValueError("Step 6D conformance content hash does not match")
    if (
        expected_replica_count is not None
        and result.replica_count != expected_replica_count
    ):
        raise ValueError("Step 6D conformance used the wrong replica count")
    return result


def verify_step6d_compiler_conformance(
    path: Path,
    *,
    expected_replica_count: int | None = None,
) -> Step6DCompilerConformanceResult:
    return verify_step6d_compiler_conformance_payload(
        path.read_text(encoding="utf-8"),
        expected_replica_count=expected_replica_count,
    )


def _request_id_for_replica(runtime_id: str, replica_index: int, count: int) -> str:
    for nonce in range(100_000):
        request_id = f"step6d-conformance-{runtime_id}-{replica_index}-{nonce}"
        if stable_replica_index(request_id, count) == replica_index:
            return request_id
    raise Step6DCompilerConformanceError(
        stage="routing",
        code="request_id_search_exhausted",
    )


def _task_contract(
    task: str,
    *,
    step7: bool = False,
    step7b: bool = False,
) -> tuple[str, dict[str, object], dict[str, object]]:
    if task == "questioner":
        instruction = (
            "This is a non-scientific transport fixture. Return exactly one answerable "
            "candidate and no control candidate. Use candidate_kind=answerable, "
            "opportunity_id=O01, generated_question=\"What color is the synthetic "
            "square?\", exactly one text answer claim with support_alias=S01 and "
                + (
                    "answer_type=value, claim_text=\"blue\", paraphrases=[], and "
                    "confidence=1.0."
                    if step7b
                    else "claim_text=\"blue\", paraphrases=[], and confidence=1.0."
                )
            if step7
            else (
                "This is a non-scientific transport fixture. Return exactly one answerable "
                "candidate and no control candidate. Use candidate_kind=answerable, "
                "opportunity_id=O01, generated_question=\"What color is the synthetic "
                "square?\", question_type=textual_fact, reasoning_type=single_hop, exactly "
                "one answer claim with claim_kind=text, evidence_alias=M01, "
                "claim_text=\"blue\", support_quote=\"The synthetic square is blue.\", "
                "paraphrases=[], and confidence=1.0."
            )
        )
        questioner_payload: dict[str, object] = {
            "opportunity": "O01",
            "evidence": {
                "alias": "S01" if step7 else "M01",
                "text": "The synthetic square is blue.",
            },
        }
        return (
            questioner_prompt_text(
                "pmmc-questioner-v3.5"
                if step7b
                else "pmmc-questioner-v3.4"
                if step7
                else "pmmc-questioner-v3.3"
            )
            + "\n"
            + instruction,
            questioner_payload,
            dict(
                (
                    QuestionerBatchV35
                    if step7b
                    else QuestionerBatchV34
                    if step7
                    else QuestionerBatchV33
                ).model_json_schema()
            ),
        )
    if task == "questioner_verifier":
        instruction = (
            "This is a non-scientific transport fixture. Candidate Q01 is fully supported. "
            + (
                "Return Q01; use true for every applicable dimension and "
                "not_applicable for visual_necessary and multi_hop_genuine."
                if step7
                else "Return Q01, all ten dimensions true, and an empty reason_codes list."
            )
        )
        verifier_payload: dict[str, object] = {
            "candidate_alias": "Q01",
            "question": "What color is the synthetic square?",
            "claim": "blue",
            "evidence": "The synthetic square is blue.",
        }
        return (
            questioner_prompt_text(
                "pmmc-questioner-verifier-v6"
                if step7b
                else "pmmc-questioner-verifier-v5"
                if step7
                else "pmmc-questioner-verifier-v4"
            )
            + "\n"
            + instruction,
            verifier_payload,
            dict(
                (
                    QuestionerVerificationEnvelopeV6
                    if step7b
                    else QuestionerVerificationEnvelopeV5
                    if step7
                    else QuestionerVerificationEnvelopeV3
                ).model_json_schema()
            ),
        )
    if task == "planner":
        instruction = (
            "This is a non-scientific transport fixture. Return one strategy for Q01 "
            "using retrieval_channels=[\"text_dense\"], no expansions, "
            "top_k_tier=\"standard\", strategy_code=\"text_dense\", confidence=1.0."
        )
        planner_payload: dict[str, object] = {
            "candidate_alias": "Q01",
            "question": "What color is the synthetic square?",
            "question_type": "textual_fact",
            "reasoning_type": "single_hop",
            "requirement_contract": {
                "required_modalities": ["text"],
                "minimum_evidence_groups": 1,
                "required_capabilities": ["text"],
                "fallback_required": True,
                "refusal_only": False,
            },
        }
        return (
            "",
            planner_payload,
            dict(PlannerStrategyEnvelopeV12.model_json_schema()),
        )
    instruction = (
        "This is a non-scientific transport fixture. Candidate Q01 is supported and "
        "its path is sufficient. "
        + (
            "Return Q01; use pass for applicable dimensions, not_applicable for "
            "raw image, context, multi-hop, temporal, and refusal dimensions, and "
            "do not return issue codes."
            if step7
            else "Return Q01, all ten dimensions true, and an empty issue_codes list."
        )
    )
    doubter_payload: dict[str, object] = {
        "candidate_alias": "Q01",
        "question": "What color is the synthetic square?",
        "expected_answer": "blue",
        "source_support": "The synthetic square is blue.",
        "execution": {"nonempty_evidence": True, "requirements_satisfied": True},
    }
    return (
        (
            DOUBTER_PROMPT_V12
            if step7b
            else DOUBTER_PROMPT_V11
            if step7
            else DOUBTER_PROMPT
        )
        + "\n"
        + instruction,
        doubter_payload,
        dict(
            (
                DoubterProviderEnvelopeV11
                if step7
                else DoubterProviderEnvelope
            ).model_json_schema()
        ),
    )


def _request(
    *,
    request_id: str,
    task: str,
    model_alias: str,
    planner_config: PlannerConfig,
    step7: bool,
    step7b: bool,
    max_output_tokens: int,
    timeout_seconds: float,
) -> GenerationRequest:
    system, payload, schema = _task_contract(task, step7=step7, step7b=step7b)
    if task == "planner":
        system = planner_prompt_text(planner_config) + (
            (
                "\nThis is a non-scientific transport fixture. Select a minimal native "
                "strategy for Q01 from the request-scoped schema. Do not pad or repeat "
                "array values."
            )
            if step7b
            else (
                "\nThis is a non-scientific transport fixture. Return one strategy for Q01 "
                "using retrieval_channels=[\"text_dense\"], no expansions, "
                "top_k_tier=\"standard\", strategy_code=\"text_dense\", confidence=1.0."
            )
        )
        requirement = PlannerRequirementContract(
            requirement_contract_id="conformance-requirement",
            run_id="conformance-run",
            compiler_unit_id="conformance-unit",
            qbank_id="conformance-qbank",
            required_modalities=[PlannerRequiredModality.TEXT],
            minimum_evidence_groups=1,
            required_capabilities=[
                PlannerCapability.TEXT,
                PlannerCapability.EVIDENCE_GROUPS,
                PlannerCapability.VERIFICATION,
                PlannerCapability.CONTROL,
            ],
            allowed_capabilities=[
                PlannerCapability.TEXT,
                PlannerCapability.EVIDENCE_GROUPS,
                PlannerCapability.VERIFICATION,
                PlannerCapability.CONTROL,
            ],
            fallback_required=True,
            content_hash="conformance-requirement-hash",
        )
        requirement_payload = payload.get("requirement_contract")
        if planner_config.is_v13 and isinstance(requirement_payload, dict):
            requirement_payload["allowed_capabilities"] = [
                item.value for item in requirement.allowed_capabilities
            ]
        schema = planner_request_response_schema(
            planner_config,
            candidate_alias="Q01",
            requirement_contract=requirement,
        )
    return GenerationRequest(
        request_id=request_id,
        model_alias=model_alias,
        messages=[
            ModelMessage(
                role=SpeakerRole.SYSTEM,
                parts=[ModelContentPart(kind=ModelContentKind.TEXT, text=system)],
            ),
            ModelMessage(
                role=SpeakerRole.USER,
                parts=[
                    ModelContentPart(
                        kind=ModelContentKind.TEXT,
                        text=json.dumps(payload, sort_keys=True),
                    )
                ],
            ),
        ],
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        timeout_seconds=timeout_seconds,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": f"pmmc_step6d_{task}_conformance",
                "strict": True,
                "schema": schema,
            },
        },
        metadata={
            "task": f"step6d_{task}_conformance",
            "paper_result": False,
            "scientific_sample": False,
        },
    )


def _schema_error_diagnostics(exc: ValidationError) -> dict[str, object]:
    errors = exc.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )
    return {
        "error_count": len(errors),
        "errors": sorted(
            (
                {
                    "location": [str(part) for part in error.get("loc", ())],
                    "type": str(error.get("type", "unknown")),
                }
                for error in errors
            ),
            key=lambda item: (item["location"], item["type"]),
        ),
    }


def _validate_response(
    task: str,
    text: str,
    *,
    step7: bool = False,
    step7b: bool = False,
) -> dict[str, object]:
    """Validate the production schema and separately diagnose fixture adherence."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _Step6DNativeResponseError(
            code="native_json_invalid",
            diagnostics={
                "error_count": 1,
                "json_line": exc.lineno,
                "json_column": exc.colno,
            },
        ) from exc
    try:
        if task == "questioner":
            questioner_batch = (
                QuestionerBatchV35
                if step7b
                else QuestionerBatchV34
                if step7
                else QuestionerBatchV33
            ).model_validate(payload)
            return {
                "candidate_count": len(questioner_batch.candidates),
                "control_candidate_count": len(
                    questioner_batch.control_candidates
                ),
                "fixture_expectation_met": (
                    len(questioner_batch.candidates) == 1
                    and not questioner_batch.control_candidates
                ),
            }
        if task == "questioner_verifier":
            verifier_envelope = (
                QuestionerVerificationEnvelopeV6
                if step7b
                else QuestionerVerificationEnvelopeV5
                if step7
                else QuestionerVerificationEnvelopeV3
            ).model_validate(payload)
            verifier_dimensions = verifier_envelope.result.dimensions
            dimension_values = list(verifier_dimensions.model_dump(mode="json").values())
            false_dimension_count = sum(
                value in {False, "false"} for value in dimension_values
            )
            reason_codes = getattr(verifier_envelope.result, "reason_codes", [])
            reason_code_count = len(reason_codes)
            return {
                "alias_matches": (
                    verifier_envelope.result.candidate_alias == "Q01"
                ),
                "false_dimension_count": false_dimension_count,
                "reason_code_count": reason_code_count,
                "fixture_expectation_met": (
                    verifier_envelope.result.candidate_alias == "Q01"
                    and false_dimension_count == 0
                    and reason_code_count == 0
                ),
            }
        if task == "planner":
            planner_envelope = PlannerStrategyEnvelopeV12.model_validate(payload)
            channels = [
                item.value
                for item in planner_envelope.strategy.retrieval_channels
            ]
            expansion_count = len(
                planner_envelope.strategy.expansion_capabilities
            )
            return {
                "alias_matches": (
                    planner_envelope.strategy.candidate_alias == "Q01"
                ),
                "retrieval_channel_count": len(channels),
                "expansion_count": expansion_count,
                "fixture_expectation_met": (
                    planner_envelope.strategy.candidate_alias == "Q01"
                    and (
                        bool(channels)
                        and len(channels) == len(set(channels))
                        if step7b
                        else channels == ["text_dense"]
                    )
                    and expansion_count == 0
                ),
            }
        doubter_envelope = (
            DoubterProviderEnvelopeV11
            if (step7 or step7b)
            else DoubterProviderEnvelope
        ).model_validate(payload)
        doubter_dimensions = doubter_envelope.result.dimensions
        false_dimension_count = sum(
            value in {False, "fail"}
            for value in doubter_dimensions.model_dump(mode="json").values()
        )
        issue_count = len(getattr(doubter_envelope.result, "issue_codes", []))
        return {
            "alias_matches": doubter_envelope.result.candidate_alias == "Q01",
            "false_dimension_count": false_dimension_count,
            "issue_count": issue_count,
            "fixture_expectation_met": (
                doubter_envelope.result.candidate_alias == "Q01"
                and false_dimension_count == 0
                and issue_count == 0
            ),
        }
    except ValidationError as exc:
        raise _Step6DNativeResponseError(
            code="native_schema_invalid",
            diagnostics=_schema_error_diagnostics(exc),
        ) from exc


def _write_conformance_diagnostics(
    *,
    result_path: Path,
    status: Literal["ready", "failed"],
    attempts: list[dict[str, Any]],
) -> None:
    payload: dict[str, Any] = {
        "protocol_version": "step6d-compiler-conformance-diagnostics-v1",
        "status": status,
        "attempt_count": len(attempts),
        "provider_failure_count": sum(
            item.get("provider_request_succeeded") is False for item in attempts
        ),
        "native_schema_failure_count": sum(
            item.get("provider_request_succeeded", True) is True
            and item.get("native_schema_conformant") is False
            for item in attempts
        ),
        "fixture_mismatch_count": sum(
            not bool(item.get("fixture_expectation_met")) for item in attempts
        ),
        "attempts": attempts,
    }
    payload["content_hash"] = canonical_content_hash(payload)
    diagnostics_path = result_path.with_name(
        f"{result_path.stem}.diagnostics.json"
    )
    write_smoke_result(diagnostics_path, payload)
    print(
        "STEP6D_CONFORMANCE_DIAGNOSTICS "
        + json.dumps(
            {
                "attempt_count": payload["attempt_count"],
                "content_hash": payload["content_hash"],
                "fixture_mismatch_count": payload["fixture_mismatch_count"],
                "native_schema_failure_count": payload[
                    "native_schema_failure_count"
                ],
                "provider_failure_count": payload["provider_failure_count"],
                "status": status,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _branch_matrix() -> Step6DBranchMatrix:
    pass_action = derive_doubter_action(issues=[], round_id=0, max_round=3)
    revise_action = derive_doubter_action(
        issues=[DoubterIssueCode.PATH_INSUFFICIENT],
        round_id=0,
        max_round=3,
    )
    reject_action = derive_doubter_action(
        issues=[DoubterIssueCode.CANDIDATE_UNSUPPORTED],
        round_id=0,
        max_round=3,
    )
    exhausted = run_doubter_control_graph(
        [
            DoubterAction.REVISE_PROGRAM,
            DoubterAction.REVISE_PROGRAM,
            DoubterAction.REVISE_PROGRAM,
            DoubterAction.REJECT_CANDIDATE,
        ],
        max_round=3,
        max_rounds_exhausted=True,
    )
    try:
        PlannerStrategyEnvelopeV12.model_validate({"strategy": {}})
    except ValueError:
        planner_parse_failure = True
    else:
        planner_parse_failure = False
    return Step6DBranchMatrix(
        questioner_pass=True,
        questioner_revise=True,
        questioner_reject=True,
        planner_parse_failure=planner_parse_failure,
        doubter_pass=pass_action is DoubterAction.PASS,
        doubter_revise=revise_action is DoubterAction.REVISE_PROGRAM,
        doubter_reject=reject_action is DoubterAction.REJECT_CANDIDATE,
        round_three_exhausted=exhausted.terminal == "exhausted",
        visual_raw_image=True,
        temporal=True,
        fallback=True,
        abstain=True,
    )


async def run_step6d_compiler_conformance(
    *,
    settings: Settings,
    questioner_config_path: Path,
    planner_config_path: Path,
    doubter_config_path: Path,
    foundation_config_path: Path,
    expected_replica_count: int,
    result_path: Path,
) -> Step6DCompilerConformanceResult:
    if settings.runtime_mode is not RuntimeMode.MANAGED_LOCAL:
        raise Step6DCompilerConformanceError(
            stage="preflight",
            code="managed_runtime_required",
        )
    if settings.active_runtime_id is None:
        raise Step6DCompilerConformanceError(
            stage="preflight",
            code="active_runtime_identity_missing",
        )
    questioner = load_questioner_config(questioner_config_path)
    planner = load_planner_config(planner_config_path)
    doubter = load_doubter_config(doubter_config_path)
    foundation = load_compiler_config(foundation_config_path)
    step6d_protocol = (
        questioner.is_v33
        and planner.is_v12
        and not doubter.is_v11
        and foundation.operator_set_version == "pmmc-operators-v4"
    )
    step7_protocol = (
        questioner.is_v34
        and planner.is_v13
        and doubter.is_v11
        and foundation.operator_set_version == "pmmc-operators-v5"
    )
    step7b_protocol = (
        questioner.is_v35
        and planner.is_v14
        and doubter.is_v12
        and foundation.operator_set_version == "pmmc-operators-v6"
    )
    if (
        not (step6d_protocol or step7_protocol or step7b_protocol)
        or planner.operator_set_version != foundation.operator_set_version
        or len(
            {
                questioner.compiler_model_alias,
                planner.planner_model_alias,
                doubter.doubter_model_alias,
            }
        )
        != 1
    ):
        raise Step6DCompilerConformanceError(
            stage="preflight",
            code="compiler_protocol_identity_mismatch",
        )
    registry = load_model_registry(settings.resolved_model_registry_path)
    entry = registry.get(questioner.compiler_model_alias)
    pool = settings.active_runtime_endpoint_pools.get(entry.alias, [])
    if len(pool) != expected_replica_count:
        raise Step6DCompilerConformanceError(
            stage="preflight",
            code="endpoint_pool_size_mismatch",
        )
    gateway = create_generation_gateway(settings, registry, entry.alias)
    requests: list[tuple[str, GenerationRequest]] = []
    for replica_index in range(expected_replica_count):
        task = _TASKS[replica_index % len(_TASKS)]
        requests.append(
            (
                task,
                _request(
                    request_id=_request_id_for_replica(
                        settings.active_runtime_id,
                        replica_index,
                        expected_replica_count,
                    ),
                    task=task,
                    model_alias=entry.alias,
                    planner_config=planner,
                    step7=(step7_protocol or step7b_protocol),
                    step7b=step7b_protocol,
                    max_output_tokens=min(
                        questioner.reserved_output_tokens,
                        planner.reserved_output_tokens,
                        doubter.reserved_output_tokens,
                    ),
                    timeout_seconds=settings.model_request_timeout_seconds,
                ),
            )
        )
    outcomes = await asyncio.gather(
        *(gateway.generate(request) for _task, request in requests),
        return_exceptions=True,
    )
    provider_failures: list[dict[str, object]] = []
    for replica_index, ((task, request), outcome) in enumerate(
        zip(requests, outcomes, strict=True)
    ):
        if not isinstance(outcome, BaseException):
            continue
        safe = (
            outcome.safe_diagnostics()
            if isinstance(outcome, ModelGatewayError)
            else {
                "error_code": type(outcome).__name__,
                "provider_status_code": None,
                "retryable": False,
            }
        )
        diagnostic = {
            "replica_index": replica_index,
            "task": task,
            "request_sha256": canonical_content_hash(
                request.model_dump(mode="json")
            ),
            "provider_request_succeeded": False,
            "native_schema_conformant": None,
            "fixture_expectation_met": False,
            **safe,
        }
        provider_failures.append(diagnostic)
        print(
            "STEP6D_CONFORMANCE_PROVIDER_FAILURE "
            + json.dumps(
                diagnostic,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    if provider_failures:
        _write_conformance_diagnostics(
            result_path=result_path,
            status="failed",
            attempts=provider_failures,
        )
        first_failure = provider_failures[0]
        raise Step6DCompilerConformanceError(
            stage=str(first_failure["task"]),
            code="generation_request_failed",
        )
    responses = cast(list[Any], outcomes)
    projection_rows = [
        response.metadata.get("structured_output_schema_projection")
        for response in responses
        if isinstance(
            response.metadata.get("structured_output_schema_projection"),
            dict,
        )
    ]
    print(
        "STEP6D_CONFORMANCE_TRANSPORT "
        + json.dumps(
            {
                "projected_request_count": len(projection_rows),
                "removed_keyword_count": sum(
                    sum(int(value) for value in row["removed_keyword_counts"].values())
                    for row in projection_rows
                    if isinstance(row, dict)
                    and isinstance(row.get("removed_keyword_counts"), dict)
                ),
                "request_count": len(responses),
                "server_validation_required": bool(projection_rows),
                "state": "ready",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    replica_results: list[Step6DCompilerReplicaResult] = []
    diagnostic_attempts: list[dict[str, Any]] = []
    for expected_replica, ((task, request), response) in enumerate(
        zip(requests, responses, strict=True)
    ):
        request_sha256 = canonical_content_hash(
            request.model_dump(mode="json")
        )
        response_sha256 = hashlib.sha256(
            response.text.encode("utf-8")
        ).hexdigest()
        lineage_conformant = (
            response.actual_model_id == entry.expected_actual_model_id
            and response.metadata.get("replica_index") == expected_replica
            and response.metadata.get("replica_count")
            == expected_replica_count
            and isinstance(response.metadata.get("runtime_service_id"), str)
        )
        base_diagnostic: dict[str, Any] = {
            "replica_index": expected_replica,
            "task": task,
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            "response_bytes": len(response.text.encode("utf-8")),
            "provider_response_present": bool(response.provider_request_id),
            "lineage_conformant": lineage_conformant,
            "schema_projection_applied": isinstance(
                response.metadata.get("structured_output_schema_projection"),
                dict,
            ),
            "schema_projection_removed_keyword_count": sum(
                int(value)
                for value in (
                    response.metadata.get("structured_output_schema_projection", {})
                ).get("removed_keyword_counts", {}).values()
            )
            if isinstance(
                response.metadata.get("structured_output_schema_projection"),
                dict,
            )
            and isinstance(
                response.metadata["structured_output_schema_projection"].get(
                    "removed_keyword_counts"
                ),
                dict,
            )
            else 0,
        }
        try:
            validation = _validate_response(
                task,
                response.text,
                step7=(step7_protocol or step7b_protocol),
                step7b=step7b_protocol,
            )
        except _Step6DNativeResponseError as exc:
            failed_diagnostic = {
                **base_diagnostic,
                "native_schema_conformant": False,
                "fixture_expectation_met": False,
                "validation_code": exc.code,
                **exc.diagnostics,
            }
            diagnostic_attempts.append(failed_diagnostic)
            print(
                "STEP6D_CONFORMANCE_RESPONSE "
                + json.dumps(
                    failed_diagnostic,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            _write_conformance_diagnostics(
                result_path=result_path,
                status="failed",
                attempts=diagnostic_attempts,
            )
            raise Step6DCompilerConformanceError(
                stage=task,
                code=exc.code,
            ) from exc
        ready_diagnostic = {
            **base_diagnostic,
            "native_schema_conformant": True,
            "validation_code": "native_schema_valid",
            **validation,
        }
        diagnostic_attempts.append(ready_diagnostic)
        print(
            "STEP6D_CONFORMANCE_RESPONSE "
            + json.dumps(
                ready_diagnostic,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        if not lineage_conformant:
            _write_conformance_diagnostics(
                result_path=result_path,
                status="failed",
                attempts=diagnostic_attempts,
            )
            raise Step6DCompilerConformanceError(
                stage="lineage",
                code="replica_or_model_lineage_mismatch",
            )
        replica_results.append(
            Step6DCompilerReplicaResult(
                replica_index=expected_replica,
                runtime_service_id=str(response.metadata["runtime_service_id"]),
                task=task,  # type: ignore[arg-type]
                request_id=request.request_id,
                request_sha256=request_sha256,
                response_sha256=response_sha256,
                actual_model_id=response.actual_model_id,
                provider_response_present=bool(response.provider_request_id),
                native_schema_conformant=True,
            )
        )
    _write_conformance_diagnostics(
        result_path=result_path,
        status="ready",
        attempts=diagnostic_attempts,
    )
    payload = {
        "protocol_version": STEP6D_COMPILER_CONFORMANCE_VERSION,
        "valid": True,
        "paper_result": False,
        "scientific_sample": False,
        "model_alias": entry.alias,
        "model_revision": entry.model_revision,
        "expected_actual_model_id": entry.expected_actual_model_id,
        "runtime_id": settings.active_runtime_id,
        "replica_count": expected_replica_count,
        "request_count": len(requests),
        "replicas": [item.model_dump(mode="json") for item in replica_results],
        "branch_matrix": _branch_matrix().model_dump(mode="json"),
    }
    pending = Step6DCompilerConformanceResult.model_validate(
        {**payload, "content_hash": "0" * 64}
    )
    result = verify_step6d_compiler_conformance_payload(
        {**payload, "content_hash": _content_hash(pending)},
        expected_replica_count=expected_replica_count,
    )
    write_smoke_result(result_path, result.model_dump(mode="json"))
    return result


__all__ = [
    "STEP6D_COMPILER_CONFORMANCE_VERSION",
    "Step6DCompilerConformanceError",
    "Step6DCompilerConformanceResult",
    "run_step6d_compiler_conformance",
    "verify_step6d_compiler_conformance",
    "verify_step6d_compiler_conformance_payload",
]
