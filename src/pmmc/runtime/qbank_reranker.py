"""Gold-free bounded reranking over the auditable base QBank candidate pool."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.infrastructure.database.query_executor_repositories import (
    QueryExecutorPersistenceConflictError,
)
from pmmc.models.execution import (
    ModelCallContext,
    ModelExecutionService,
    _canonical_bytes,
    _generation_manifest,
)
from pmmc.models.protocols import GenerationGateway
from pmmc.schemas import (
    AppStage,
    DataScope,
    GenerationRequest,
    Modality,
    ModelCallTrace,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    QBankRerankerAssessment,
    QBankRerankerDimensions,
    QBankRerankerOutput,
    QueryInput,
    RerankerDimensionVerdict,
    RerankerMatchLevel,
    SpeakerRole,
    ThinkingMode,
    TraceStatus,
)

RERANKER_PROMPT_VERSION = "pmmc-qbank-reranker-v1"
RERANKER_CONTRACT_VERSION = "reranker-required-dimensions-v2"
RERANKER_PROMPT = """Decide whether the candidate question is a useful routing target
for the user's query. Compare only the information need, answer type, and modality.
Do not infer an answer. Return exactly the guided JSON object and no explanation."""


@lru_cache(maxsize=1)
def reranker_contract_hash() -> str:
    return canonical_content_hash({
        "version": RERANKER_CONTRACT_VERSION,
        "schema": QBankRerankerOutput.model_json_schema(),
    })


def _reranker_request_id(run_id: str, query_id: str, qbank_id: str) -> str:
    return stable_compiler_id(
        "qbank-reranker-request", [run_id, query_id, qbank_id, reranker_contract_hash()],
    )


def reranker_model_call_id(run_id: str, query_id: str, qbank_id: str) -> str:
    return stable_compiler_id(
        "qbank-reranker-model-call", _reranker_request_id(run_id, query_id, qbank_id),
    )


class RerankerProviderContractError(ValueError):
    error_code = "reranker_provider_schema_invalid"


def preflight_reranker_contract() -> dict[str, object]:
    """Check guided/Pydantic applicability before any GPU model startup."""
    for model in (QBankRerankerOutput, QBankRerankerDimensions):
        schema = model.model_json_schema()
        for name in ("same_information_need", "candidate_can_answer_query",
                     "answer_type_compatible", "modality_compatible"):
            prop = schema["properties"][name]
            if "$ref" in prop:
                prop = schema["$defs"][prop["$ref"].rsplit("/", 1)[-1]]
            expected = {"pass", "fail"}
            if name == "modality_compatible":
                expected.add("not_applicable")
            if set(prop.get("enum", ())) != expected:
                raise ValueError("reranker guided dimension contract mismatch")
    return {
        "contract_version": RERANKER_CONTRACT_VERSION,
        "contract_hash": reranker_contract_hash(),
        "required_dimensions": "pass_fail",
        "modality_dimension": "pass_fail_not_applicable",
        "model_calls": 0,
    }


@dataclass(frozen=True)
class BoundedRerankerPair:
    qbank_id: str
    candidate_question: str
    question_type: str
    required_modalities: tuple[Modality, ...]


def reranker_pair_union(
    grid_candidates: list[tuple[str, float, str]],
    *,
    top_k: int = 20,
) -> tuple[str, ...]:
    """Deduplicate the top-k union from independently ranked grid projections."""

    if top_k <= 0:
        raise ValueError("reranker top_k must be positive")
    by_grid: dict[str, list[tuple[float, str]]] = {}
    for grid_key, score, qbank_id in grid_candidates:
        by_grid.setdefault(grid_key, []).append((score, qbank_id))
    selected: set[str] = set()
    for values in by_grid.values():
        values.sort(key=lambda item: (-item[0], item[1]))
        selected.update(qbank_id for _score, qbank_id in values[:top_k])
    return tuple(sorted(selected))


def effective_reranker_output(
    output: QBankRerankerOutput,
) -> tuple[QBankRerankerDimensions, RerankerMatchLevel]:
    dimensions = QBankRerankerDimensions(
        same_information_need=output.same_information_need,
        candidate_can_answer_query=output.candidate_can_answer_query,
        answer_type_compatible=output.answer_type_compatible,
        modality_compatible=output.modality_compatible,
    )
    required = (
        dimensions.same_information_need,
        dimensions.candidate_can_answer_query,
        dimensions.answer_type_compatible,
    )
    failed = any(item is not RerankerDimensionVerdict.PASS for item in required)
    failed = failed or dimensions.modality_compatible is RerankerDimensionVerdict.FAIL
    return dimensions, RerankerMatchLevel.NONE if failed else output.match_level


class BoundedQBankReranker:
    """Execute one independent, non-oracle logical request per query/candidate pair."""

    def __init__(
        self,
        *,
        execution: ModelExecutionService,
        gateway: GenerationGateway,
        model_alias: str = "reranker-qwen3.5-9b",
        concurrency: int = 64,
    ) -> None:
        self.execution = execution
        self.gateway = gateway
        self.model_alias = model_alias
        self.semaphore = asyncio.Semaphore(concurrency)

    async def assess(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        query: QueryInput,
        pair: BoundedRerankerPair,
        existing_trace: ModelCallTrace | None = None,
    ) -> QBankRerankerAssessment:
        request_id = _reranker_request_id(run_id, query.query_id, pair.qbank_id)
        parts = [
            ModelContentPart(
                kind=ModelContentKind.TEXT,
                text=(
                    "<QUERY>" + query.raw_query + "</QUERY>\n"
                    "<CANDIDATE alias=\"C01\">"
                    + pair.candidate_question
                    + "</CANDIDATE>\n"
                    "<QUESTION_TYPE>" + pair.question_type + "</QUESTION_TYPE>\n"
                    "<REQUIRED_MODALITIES>"
                    + ",".join(item.value for item in pair.required_modalities)
                    + "</REQUIRED_MODALITIES>"
                ),
            )
        ]
        parts.extend(
            ModelContentPart(kind=ModelContentKind.IMAGE, image=image)
            for image in query.query_image_refs
        )
        request = GenerationRequest(
            request_id=request_id,
            model_alias=self.model_alias,
            messages=[
                ModelMessage(
                    role=SpeakerRole.SYSTEM,
                    parts=[ModelContentPart(kind=ModelContentKind.TEXT, text=RERANKER_PROMPT)],
                ),
                ModelMessage(role=SpeakerRole.USER, parts=parts),
            ],
            max_output_tokens=128,
            temperature=0.0,
            top_p=1.0,
            seed=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "pmmc_qbank_reranker_v1",
                    "strict": True,
                    "schema": QBankRerankerOutput.model_json_schema(),
                },
            },
            thinking_mode=ThinkingMode.DISABLED,
            metadata={
                "prompt_version": RERANKER_PROMPT_VERSION,
                "reranker_contract_hash": reranker_contract_hash(),
                "single_pair_scope": True,
                "private_compiler_facts_present": False,
            },
        )
        model_call_id = reranker_model_call_id(run_id, query.query_id, pair.qbank_id)
        if existing_trace is not None:
            if (
                existing_trace.model_call_id != model_call_id
                or existing_trace.run_id != run_id
                or existing_trace.snapshot_id != snapshot_id
                or existing_trace.query_id != query.query_id
                or existing_trace.model_alias != self.model_alias
                or existing_trace.stage is not AppStage.MATCH
                or existing_trace.data_scope is not DataScope.RUNTIME
                or existing_trace.prompt_hash != hashlib.sha256(
                    _canonical_bytes(_generation_manifest(request))
                ).hexdigest()
            ):
                raise QueryExecutorPersistenceConflictError(
                    "reranker cached call identity differs", operation="reranker_recovery",
                    error_code="reranker_call_identity_conflict",
                )
            if (
                existing_trace.status is not TraceStatus.SUCCESS
                or not existing_trace.input_asset_id or not existing_trace.output_asset_id
                or not existing_trace.actual_model_id
            ):
                raise QueryExecutorPersistenceConflictError(
                    "reranker cached call is not successful", operation="reranker_recovery",
                    error_code="reranker_call_not_recoverable",
                )
            trace = existing_trace
            self.execution.registry.verify_actual_model(
                self.model_alias, existing_trace.actual_model_id,
                paper_mode=self.execution.paper_mode,
            )
        else:
            try:
                async with self.semaphore:
                    audited = await self.execution.generate_audited(
                        self.gateway,
                        request,
                        ModelCallContext(
                            model_call_id=model_call_id,
                            run_id=run_id,
                            stage=AppStage.MATCH,
                            snapshot_id=snapshot_id,
                            query_id=query.query_id,
                            conversation_id=query.conversation_id,
                        ),
                    )
            except SQLAlchemyError as exc:
                exc.persistence_operation = "reranker_audited_call"  # type: ignore[attr-defined]
                raise
            trace = audited.trace
        try:
            output = QBankRerankerOutput.model_validate(
                json.loads((trace.parsed_output or {})["text"])
            )
        except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise RerankerProviderContractError(
                "reranker provider output is not schema conformant",
            ) from exc
        dimensions, effective_level = effective_reranker_output(output)
        payload = {
            "run_id": run_id,
            "query_id": query.query_id,
            "qbank_id": pair.qbank_id,
            "model_call_id": trace.model_call_id,
            "dimensions": dimensions.model_dump(mode="json"),
            "provider_match_level": output.match_level.value,
            "effective_match_level": effective_level.value,
        }
        return QBankRerankerAssessment(
            reranker_assessment_id=stable_compiler_id("qbank-reranker-assessment", payload),
            query_executor_run_id=run_id,
            query_id=query.query_id,
            qbank_id=pair.qbank_id,
            model_call_id=trace.model_call_id,
            dimensions=dimensions,
            provider_match_level=output.match_level,
            effective_match_level=effective_level,
            reranker_score=effective_level.score,
            provider_schema_conformant=True,
            content_hash=canonical_content_hash(payload),
            created_at=trace.created_at,
            metadata={"request_id": request_id, "reranker_contract_hash": reranker_contract_hash()},
        )


__all__ = [
    "BoundedQBankReranker",
    "BoundedRerankerPair",
    "RERANKER_PROMPT",
    "RERANKER_PROMPT_VERSION",
    "effective_reranker_output",
    "reranker_pair_union",
    "reranker_model_call_id",
]
