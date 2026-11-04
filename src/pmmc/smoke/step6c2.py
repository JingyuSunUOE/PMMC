"""Eight-replica semantic Judge conformance for the Step 6C.2 GPU smoke."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation.qbank_semantic import (
    QBankSemanticMatchConfig,
    load_qbank_semantic_config,
    parse_qbank_semantic_output,
)
from pmmc.infrastructure.settings import Settings
from pmmc.models.factory import create_generation_gateway
from pmmc.models.registry import load_model_registry
from pmmc.models.routing import stable_replica_index
from pmmc.schemas import (
    BaseSchema,
    GenerationRequest,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    QBankSemanticJudgeEnvelope,
    QBankSemanticJudgeEnvelopeV2,
    RuntimeMode,
    SpeakerRole,
)
from pmmc.schemas.base import NonEmptyStr, NonNegativeInt, PositiveInt
from pmmc.smoke.step5d import write_smoke_result

STEP6C2_SEMANTIC_CONFORMANCE_VERSION = "step6c2-semantic-conformance-v1"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SYSTEM_PROMPT_V1 = """You are running a non-scientific transport conformance test.
The query, candidate question, candidate answer, and gold answer below are intentionally
identical. Return the required strict JSON envelope. Set every dimension to true and return
an empty reason_codes list. Do not add prose."""
_SYSTEM_PROMPT_V2 = """You are running a non-scientific transport conformance test.
The query, candidate question, candidate answer, and gold answer below are intentionally
identical. Return the required strict JSON envelope. Set the first three dimensions to true
and query_image_binding_compatible to "not_applicable". Do not add reason codes or prose."""
_SYSTEM_PROMPT_V3 = """You are running a non-scientific Step 7 transport conformance test.
The query, candidate question, candidate answer, and gold answer are intentionally identical.
Return only the dimensions-only strict JSON envelope. Set the first three dimensions to true
and query_image_binding_compatible to "not_applicable". Do not add reasons or prose."""


class Step6C2SemanticConformanceError(ValueError):
    """A stage-classified, content-free semantic conformance failure."""

    def __init__(self, *, stage: str, code: str) -> None:
        self.stage = stage
        self.code = code
        super().__init__(f"stage={stage} code={code}")


class Step6C2SemanticReplicaResult(BaseSchema):
    replica_index: NonNegativeInt
    runtime_service_id: NonEmptyStr
    request_id: NonEmptyStr
    request_sha256: NonEmptyStr
    response_sha256: NonEmptyStr
    actual_model_id: NonEmptyStr
    provider_response_present: Literal[True]
    dimensions_passed: Literal[True]

    @model_validator(mode="after")
    def _validate_hashes(self) -> Step6C2SemanticReplicaResult:
        if not _HASH_PATTERN.fullmatch(self.request_sha256):
            raise ValueError("semantic conformance request hash is invalid")
        if not _HASH_PATTERN.fullmatch(self.response_sha256):
            raise ValueError("semantic conformance response hash is invalid")
        return self


class Step6C2SemanticConformanceResult(BaseSchema):
    protocol_version: Literal["step6c2-semantic-conformance-v1"]
    valid: Literal[True]
    paper_result: Literal[False]
    scientific_sample: Literal[False]
    fixture_kind: Literal["synthetic_exact_semantic_pair"]
    model_alias: NonEmptyStr
    model_revision: NonEmptyStr
    expected_actual_model_id: NonEmptyStr
    runtime_id: NonEmptyStr
    replica_count: PositiveInt
    request_count: PositiveInt
    replicas: list[Step6C2SemanticReplicaResult]
    content_hash: NonEmptyStr

    @model_validator(mode="after")
    def _validate_identity(self) -> Step6C2SemanticConformanceResult:
        if not _COMMIT_PATTERN.fullmatch(self.model_revision):
            raise ValueError("semantic conformance requires a pinned model commit")
        if self.request_count != self.replica_count:
            raise ValueError("semantic conformance requires one request per replica")
        if len(self.replicas) != self.replica_count:
            raise ValueError("semantic conformance replica denominator is incomplete")
        if {item.replica_index for item in self.replicas} != set(
            range(self.replica_count)
        ):
            raise ValueError("semantic conformance did not cover every replica index")
        if len({item.runtime_service_id for item in self.replicas}) != self.replica_count:
            raise ValueError("semantic conformance did not cover every runtime service")
        if any(
            item.actual_model_id != self.expected_actual_model_id
            for item in self.replicas
        ):
            raise ValueError("semantic conformance actual model identity differs")
        if not _HASH_PATTERN.fullmatch(self.content_hash):
            raise ValueError("semantic conformance content hash is invalid")
        return self


def _content_hash(value: Step6C2SemanticConformanceResult) -> str:
    return canonical_content_hash(
        value.model_dump(mode="json", exclude={"content_hash"})
    )


def verify_step6c2_semantic_conformance_payload(
    value: object,
    *,
    expected_replica_count: int | None = None,
) -> Step6C2SemanticConformanceResult:
    if isinstance(value, (str, bytes, bytearray)):
        value = json.loads(value)
    result = Step6C2SemanticConformanceResult.model_validate(value)
    if _content_hash(result) != result.content_hash:
        raise ValueError("semantic conformance content hash does not match")
    if (
        expected_replica_count is not None
        and result.replica_count != expected_replica_count
    ):
        raise ValueError("semantic conformance used the wrong replica count")
    return result


def verify_step6c2_semantic_conformance(
    path: Path,
    *,
    expected_replica_count: int | None = None,
) -> Step6C2SemanticConformanceResult:
    return verify_step6c2_semantic_conformance_payload(
        path.read_text(encoding="utf-8"),
        expected_replica_count=expected_replica_count,
    )


def _request_id_for_replica(runtime_id: str, replica_index: int, count: int) -> str:
    for nonce in range(100_000):
        request_id = (
            f"step6c2-semantic-conformance-{runtime_id}-{replica_index}-{nonce}"
        )
        if stable_replica_index(request_id, count) == replica_index:
            return request_id
    raise Step6C2SemanticConformanceError(
        stage="routing", code="request_id_search_exhausted"
    )


def _request(
    *,
    request_id: str,
    pair_alias: str,
    config: QBankSemanticMatchConfig,
    timeout_seconds: float,
) -> GenerationRequest:
    payload = {
        "pair_alias": pair_alias,
        "query": "What color is the synthetic square?",
        "candidate_question": "What color is the synthetic square?",
        "candidate_expected_answer": "blue",
        "benchmark_gold_answer": "blue",
        "candidate_required_modalities": ["text"],
        "query_has_image": False,
        "scientific_sample": False,
    }
    schema_model = (
        QBankSemanticJudgeEnvelopeV2
        if config.is_dimensions_only
        else QBankSemanticJudgeEnvelope
    )
    schema = schema_model.model_json_schema()
    return GenerationRequest(
        request_id=request_id,
        model_alias=config.judge_model_alias,
        messages=[
            ModelMessage(
                role=SpeakerRole.SYSTEM,
                parts=[
                    ModelContentPart(
                        kind=ModelContentKind.TEXT,
                        text=(
                            _SYSTEM_PROMPT_V3
                            if config.is_v3
                            else _SYSTEM_PROMPT_V2
                            if config.is_v2
                            else _SYSTEM_PROMPT_V1
                        ),
                    )
                ],
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
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        timeout_seconds=timeout_seconds,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "pmmc_step6c2_semantic_conformance_v1",
                "strict": True,
                "schema": schema,
            },
        },
        metadata={
            "task": "step6c2_semantic_conformance",
            "paper_result": False,
            "scientific_sample": False,
        },
    )


async def run_step6c2_semantic_conformance(
    *,
    settings: Settings,
    semantic_config_path: Path,
    expected_replica_count: int,
    result_path: Path,
) -> Step6C2SemanticConformanceResult:
    if settings.runtime_mode is not RuntimeMode.MANAGED_LOCAL:
        raise Step6C2SemanticConformanceError(
            stage="preflight", code="managed_runtime_required"
        )
    if settings.active_runtime_id is None:
        raise Step6C2SemanticConformanceError(
            stage="preflight", code="active_runtime_identity_missing"
        )
    config = load_qbank_semantic_config(semantic_config_path)
    registry = load_model_registry(settings.resolved_model_registry_path)
    entry = registry.get(config.judge_model_alias)
    pool = settings.active_runtime_endpoint_pools.get(entry.alias, [])
    if len(pool) != expected_replica_count:
        raise Step6C2SemanticConformanceError(
            stage="preflight", code="endpoint_pool_size_mismatch"
        )
    gateway = create_generation_gateway(settings, registry, entry.alias)

    requests = [
        _request(
            request_id=_request_id_for_replica(
                settings.active_runtime_id,
                replica_index,
                expected_replica_count,
            ),
            pair_alias=f"P{replica_index:02d}",
            config=config,
            timeout_seconds=settings.model_request_timeout_seconds,
        )
        for replica_index in range(expected_replica_count)
    ]
    try:
        responses = await asyncio.gather(
            *(gateway.generate(request) for request in requests)
        )
    except Exception as exc:
        raise Step6C2SemanticConformanceError(
            stage="provider", code="generation_request_failed"
        ) from exc

    replica_results: list[Step6C2SemanticReplicaResult] = []
    for expected_replica, (request, response) in enumerate(
        zip(requests, responses, strict=True)
    ):
        try:
            parsed = parse_qbank_semantic_output(response.text, config=config)
            envelope_model = (
                QBankSemanticJudgeEnvelopeV2
                if config.is_dimensions_only
                else QBankSemanticJudgeEnvelope
            )
            envelope = envelope_model.model_validate(parsed.value)
        except Exception as exc:
            raise Step6C2SemanticConformanceError(
                stage="parse", code="semantic_response_parse_failed"
            ) from exc
        dimensions = envelope.result.dimensions
        if (
            envelope.result.pair_alias != f"P{expected_replica:02d}"
            or not all(
                (
                    dimensions.same_information_need,
                    dimensions.candidate_answer_sufficient,
                    dimensions.contradiction_free,
                    dimensions.query_image_binding_compatible is not False,
                )
            )
            or (
                not config.is_dimensions_only
                and bool(envelope.result.reason_codes)  # type: ignore[union-attr]
            )
        ):
            raise Step6C2SemanticConformanceError(
                stage="semantic_contract", code="exact_pair_dimensions_failed"
            )
        actual_replica = response.metadata.get("replica_index")
        runtime_service_id = response.metadata.get("runtime_service_id")
        replica_count = response.metadata.get("replica_count")
        if (
            actual_replica != expected_replica
            or replica_count != expected_replica_count
            or not isinstance(runtime_service_id, str)
        ):
            raise Step6C2SemanticConformanceError(
                stage="lineage", code="replica_lineage_mismatch"
            )
        replica_results.append(
            Step6C2SemanticReplicaResult(
                replica_index=expected_replica,
                runtime_service_id=runtime_service_id,
                request_id=request.request_id,
                request_sha256=canonical_content_hash(
                    request.model_dump(mode="json")
                ),
                response_sha256=hashlib.sha256(
                    response.text.encode("utf-8")
                ).hexdigest(),
                actual_model_id=response.actual_model_id,
                provider_response_present=bool(response.provider_request_id),
                dimensions_passed=True,
            )
        )

    payload = {
        "protocol_version": STEP6C2_SEMANTIC_CONFORMANCE_VERSION,
        "valid": True,
        "paper_result": False,
        "scientific_sample": False,
        "fixture_kind": "synthetic_exact_semantic_pair",
        "model_alias": entry.alias,
        "model_revision": entry.model_revision,
        "expected_actual_model_id": entry.expected_actual_model_id,
        "runtime_id": settings.active_runtime_id,
        "replica_count": expected_replica_count,
        "request_count": len(requests),
        "replicas": [item.model_dump(mode="json") for item in replica_results],
    }
    pending = Step6C2SemanticConformanceResult.model_validate(
        {**payload, "content_hash": "0" * 64}
    )
    result = verify_step6c2_semantic_conformance_payload(
        {**payload, "content_hash": _content_hash(pending)},
        expected_replica_count=expected_replica_count,
    )
    write_smoke_result(result_path, result.model_dump(mode="json"))
    return result


__all__ = [
    "STEP6C2_SEMANTIC_CONFORMANCE_VERSION",
    "Step6C2SemanticConformanceError",
    "Step6C2SemanticConformanceResult",
    "run_step6c2_semantic_conformance",
    "verify_step6c2_semantic_conformance",
    "verify_step6c2_semantic_conformance_payload",
]
