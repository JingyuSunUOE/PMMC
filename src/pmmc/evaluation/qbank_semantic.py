"""Evaluator-only semantic matching for QBank query-candidate pairs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pmmc.benchmarks.ids import stable_hash
from pmmc.compiler.ids import canonical_content_hash
from pmmc.models import ModelCallContext, ModelExecutionService, ParsedModelOutput
from pmmc.models.protocols import GenerationGateway
from pmmc.schemas import (
    AppStage,
    GenerationRequest,
    ImageRef,
    ModelCallTrace,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    QBankCoverageCandidate,
    QBankSemanticJudgeDimensions,
    QBankSemanticJudgeEnvelope,
    QBankSemanticJudgeEnvelopeV2,
    QBankSemanticMatchAssessment,
    QBankSemanticMetric,
    QBankSemanticReason,
    QueryInput,
    QuestionCandidate,
    SpeakerRole,
)
from pmmc.schemas.semantic import semantic_run_schema_version

SEMANTIC_JUDGE_PROMPT_V1 = """You are an evaluator of one query and one candidate question.
Judge only whether they ask for the same information and whether the candidate expected
answer is sufficient and contradiction-free for the benchmark gold answer. If the query
contains an image, also judge whether the candidate modality is compatible with that image.
Do not infer from memory, evidence locators, programs, or other QBank candidates. Return only
the strict JSON object required by the response schema. For each false dimension emit exactly
its matching code: different_information_need, candidate_answer_insufficient, contradiction,
or query_image_binding_mismatch. Emit no reason for a true dimension. Do not include reasoning
prose."""

SEMANTIC_JUDGE_PROMPT_V2 = """You evaluate exactly one benchmark query and one QBank
candidate. Decide whether they ask for the same information, whether the candidate expected
answer is sufficient for the gold answer, and whether the two answers are contradiction-free.
When both a query image and candidate evidence image are provided, judge their binding
compatibility; otherwise return "not_applicable" for that dimension. Do not infer from memory
outside the supplied pair, evidence locators, programs, or other candidates. Return only the
strict JSON object required by the response schema. Do not emit reason codes or reasoning
prose; the service derives reason codes from the dimensions."""

SEMANTIC_JUDGE_PROMPT_V3 = """You evaluate exactly one benchmark query and one verified
QBank candidate. Return only the dimensions required by the strict JSON schema. Judge the
information need, answer sufficiency, and contradiction independently. For a visual pair,
compare the query image only with the candidate's explicitly referenced evidence image;
otherwise mark image binding not_applicable. Do not inspect memory outside this pair,
programs, routing scores, other candidates, evidence locators, or internal metadata. Do not
emit reason codes, verdicts, or reasoning prose; the evaluator derives them."""

SEMANTIC_JUDGE_PROMPT_V4 = """You evaluate exactly one benchmark query and one Step 7B
verified-QBank candidate. Return only the dimensions required by the strict JSON schema.
Judge information need, answer sufficiency, contradiction, and visual binding independently.
Use only this pair and its explicitly supplied images. Do not inspect programs, route or
reranker scores, answer variants, other candidates, evidence locators, or internal metadata.
Do not emit reason codes, verdicts, or reasoning prose; the evaluator derives them."""


class QBankSemanticMatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    protocol_version: Literal[
        "qbank-semantic-match-v1",
        "qbank-semantic-match-v2",
        "qbank-semantic-match-v3",
        "qbank-semantic-match-v4",
    ] = "qbank-semantic-match-v3"
    prompt_version: Literal[
        "pmmc-qbank-semantic-judge-v1",
        "pmmc-qbank-semantic-judge-v2",
        "pmmc-qbank-semantic-judge-v3",
        "pmmc-qbank-semantic-judge-v4",
    ] = "pmmc-qbank-semantic-judge-v3"
    renderer_version: Literal[
        "qbank-semantic-judge-private-renderer-v1",
        "qbank-semantic-judge-private-renderer-v2",
        "qbank-semantic-judge-private-renderer-v3",
        "qbank-semantic-judge-private-renderer-v4",
    ] = "qbank-semantic-judge-private-renderer-v3"
    parser_version: Literal[
        "qbank-semantic-judge-json-parser-v1",
        "qbank-semantic-judge-json-parser-v2",
        "qbank-semantic-judge-json-parser-v3",
        "qbank-semantic-judge-json-parser-v4",
    ] = "qbank-semantic-judge-json-parser-v3"
    schema_version: Literal[
        "qbank-semantic-judge-schema-v1",
        "qbank-semantic-judge-schema-v2",
        "qbank-semantic-judge-schema-v3",
        "qbank-semantic-judge-schema-v4",
    ] = "qbank-semantic-judge-schema-v3"
    judge_model_alias: str = Field(default="judge-qwen3.5-9b", min_length=1)
    question_budgets: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    match_ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20, 50])
    candidate_pool_k: Literal[50] = 50
    max_output_tokens: int = Field(default=512, ge=1, le=1024)
    concurrency: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    top_p: float = Field(default=1.0, ge=1.0, le=1.0)
    seed: Literal[0] = 0

    @model_validator(mode="after")
    def _validate_protocol(self) -> QBankSemanticMatchConfig:
        identity = (
            self.protocol_version,
            self.prompt_version,
            self.renderer_version,
            self.parser_version,
            self.schema_version,
        )
        if identity not in {
            (
                "qbank-semantic-match-v1",
                "pmmc-qbank-semantic-judge-v1",
                "qbank-semantic-judge-private-renderer-v1",
                "qbank-semantic-judge-json-parser-v1",
                "qbank-semantic-judge-schema-v1",
            ),
            (
                "qbank-semantic-match-v2",
                "pmmc-qbank-semantic-judge-v2",
                "qbank-semantic-judge-private-renderer-v2",
                "qbank-semantic-judge-json-parser-v2",
                "qbank-semantic-judge-schema-v2",
            ),
            (
                "qbank-semantic-match-v3",
                "pmmc-qbank-semantic-judge-v3",
                "qbank-semantic-judge-private-renderer-v3",
                "qbank-semantic-judge-json-parser-v3",
                "qbank-semantic-judge-schema-v3",
            ),
            (
                "qbank-semantic-match-v4",
                "pmmc-qbank-semantic-judge-v4",
                "qbank-semantic-judge-private-renderer-v4",
                "qbank-semantic-judge-json-parser-v4",
                "qbank-semantic-judge-schema-v4",
            ),
        }:
            raise ValueError("QBank semantic protocol identities are frozen")
        if self.question_budgets != [1, 2, 4, 8]:
            raise ValueError("semantic match requires budgets 1/2/4/8")
        if self.match_ks != [1, 3, 5, 10, 20, 50]:
            raise ValueError("semantic match requires K=1/3/5/10/20/50")
        return self

    @property
    def is_v2(self) -> bool:
        return self.protocol_version == "qbank-semantic-match-v2"

    @property
    def is_v3(self) -> bool:
        return self.protocol_version in {
            "qbank-semantic-match-v3",
            "qbank-semantic-match-v4",
        }

    @property
    def is_v4(self) -> bool:
        return self.protocol_version == "qbank-semantic-match-v4"

    @property
    def run_schema_version(self) -> str:
        return semantic_run_schema_version(self.protocol_version)

    @property
    def is_dimensions_only(self) -> bool:
        return self.protocol_version in {
            "qbank-semantic-match-v2",
            "qbank-semantic-match-v3",
            "qbank-semantic-match-v4",
        }

    @property
    def config_hash(self) -> str:
        return canonical_content_hash(self.model_dump(mode="json"))


def load_qbank_semantic_config(path: Path) -> QBankSemanticMatchConfig:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        import yaml  # type: ignore[import-untyped]

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("QBank semantic config root must be a mapping")
    return QBankSemanticMatchConfig.model_validate(raw)


def qbank_semantic_prompt_hash(
    config: QBankSemanticMatchConfig | None = None,
) -> str:
    prompt = (
        SEMANTIC_JUDGE_PROMPT_V4
        if config is not None and config.is_v4
        else SEMANTIC_JUDGE_PROMPT_V3
        if config is not None and config.is_v3
        else SEMANTIC_JUDGE_PROMPT_V2
        if config is not None and config.is_v2
        else SEMANTIC_JUDGE_PROMPT_V1
    )
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def qbank_semantic_response_schema_hash(
    config: QBankSemanticMatchConfig | None = None,
) -> str:
    """Return the frozen provider-facing semantic judge schema identity."""

    schema = (
        QBankSemanticJudgeEnvelopeV2
        if config is not None and config.is_dimensions_only
        else QBankSemanticJudgeEnvelope
    )
    return canonical_content_hash(schema.model_json_schema())


def parse_qbank_semantic_output(
    text: str,
    *,
    config: QBankSemanticMatchConfig | None = None,
) -> ParsedModelOutput:
    schema = (
        QBankSemanticJudgeEnvelopeV2
        if config is not None and config.is_dimensions_only
        else QBankSemanticJudgeEnvelope
    )
    initial_error: str | None = None
    try:
        value = schema.model_validate(json.loads(text))
    except Exception as exc:
        initial_error = " ".join(str(exc).split())[:1000] or type(exc).__name__
        try:
            from json_repair import repair_json
        except ModuleNotFoundError as missing:
            raise ValueError("json-repair is required for semantic judge repair") from missing
        value = schema.model_validate(json.loads(repair_json(text)))
    return ParsedModelOutput(
        value=value.model_dump(mode="json"),
        repair_attempt_count=0 if initial_error is None else 1,
        initial_error=initial_error,
    )


def semantic_pair_hash(
    *,
    semantic_run_id: str,
    query_id: str,
    qbank_id: str,
) -> str:
    return hashlib.sha256(
        "\x1f".join((semantic_run_id, query_id, qbank_id)).encode("utf-8")
    ).hexdigest()


def build_qbank_semantic_request(
    *,
    semantic_run_id: str,
    semantic_job_id: str,
    pair_alias: str,
    query: QueryInput,
    candidate: QuestionCandidate,
    gold_answer: str,
    evidence_images: Sequence[ImageRef] = (),
    config: QBankSemanticMatchConfig,
    timeout_seconds: float,
) -> GenerationRequest:
    if candidate.expected_answer is None:
        raise ValueError("semantic match requires a candidate expected answer")
    payload = {
        "pair_alias": pair_alias,
        "query": query.raw_query,
        "candidate_question": candidate.canonical_question,
        "candidate_expected_answer": candidate.expected_answer,
        "benchmark_gold_answer": gold_answer,
        "candidate_required_modalities": sorted(
            item.value for item in candidate.required_modalities
        ),
        "query_has_image": bool(query.query_image_refs),
        "candidate_evidence_image_count": len(evidence_images),
    }
    parts = [
        ModelContentPart(
            kind=ModelContentKind.TEXT,
            text=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
    ]
    parts.extend(
        ModelContentPart(kind=ModelContentKind.IMAGE, image=image)
        for image in query.query_image_refs
    )
    parts.extend(
        ModelContentPart(kind=ModelContentKind.IMAGE, image=image)
        for image in evidence_images
    )
    response_schema = (
        QBankSemanticJudgeEnvelopeV2
        if config.is_dimensions_only
        else QBankSemanticJudgeEnvelope
    )
    system_prompt = (
        SEMANTIC_JUDGE_PROMPT_V4
        if config.is_v4
        else SEMANTIC_JUDGE_PROMPT_V3
        if config.is_v3
        else SEMANTIC_JUDGE_PROMPT_V2
        if config.is_v2
        else SEMANTIC_JUDGE_PROMPT_V1
    )
    return GenerationRequest(
        request_id="qbank-semantic-request-"
        + stable_hash([semantic_run_id, semantic_job_id, pair_alias])[:24],
        model_alias=config.judge_model_alias,
        messages=[
            ModelMessage(
                role=SpeakerRole.SYSTEM,
                parts=[
                    ModelContentPart(
                        kind=ModelContentKind.TEXT, text=system_prompt
                    )
                ],
            ),
            ModelMessage(role=SpeakerRole.USER, parts=parts),
        ],
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        timeout_seconds=timeout_seconds,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "qbank_semantic_match",
                "strict": True,
                "schema": response_schema.model_json_schema(),
            },
        },
        metadata={
            "task": "qbank_semantic_match",
            "protocol_version": config.protocol_version,
            "semantic_run_id": semantic_run_id,
            "semantic_job_id": semantic_job_id,
        },
    )


class QBankSemanticJudgeService:
    def __init__(
        self,
        *,
        execution: ModelExecutionService,
        gateway: GenerationGateway,
        config: QBankSemanticMatchConfig,
        timeout_seconds: float,
    ) -> None:
        self.execution = execution
        self.gateway = gateway
        self.config = config
        self.timeout_seconds = timeout_seconds

    async def assess(
        self,
        *,
        semantic_run_id: str,
        semantic_job_id: str,
        coverage_run_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        query: QueryInput,
        candidate: QuestionCandidate,
        gold_answer: str,
        evidence_images: Sequence[ImageRef] = (),
        evidence_proxy_complete: bool,
        attempt_number: int,
    ) -> QBankSemanticMatchAssessment:
        pair_alias = "P01"
        model_call_id = "qbank-semantic-call-" + stable_hash(
            [semantic_run_id, query.query_id, candidate.qbank_id, attempt_number]
        )[:24]
        audited = await self.execution.generate_audited(
            self.gateway,
            build_qbank_semantic_request(
                semantic_run_id=semantic_run_id,
                semantic_job_id=semantic_job_id,
                pair_alias=pair_alias,
                query=query,
                candidate=candidate,
                gold_answer=gold_answer,
                evidence_images=evidence_images,
                config=self.config,
                timeout_seconds=self.timeout_seconds,
            ),
            ModelCallContext(
                model_call_id=model_call_id,
                run_id=compiler_run_id,
                semantic_run_id=semantic_run_id,
                semantic_job_id=semantic_job_id,
                snapshot_id=snapshot_id,
                query_id=query.query_id,
                conversation_id=query.conversation_id,
                qbank_id=candidate.qbank_id,
                stage=AppStage.JUDGE,
            ),
            output_parser=lambda result: parse_qbank_semantic_output(
                result.text, config=self.config
            ),
        )
        return semantic_assessment_from_trace(
            trace=audited.trace,
            semantic_run_id=semantic_run_id,
            semantic_job_id=semantic_job_id,
            coverage_run_id=coverage_run_id,
            query_id=query.query_id,
            qbank_id=candidate.qbank_id,
            evidence_proxy_complete=evidence_proxy_complete,
            config=self.config,
        )


def semantic_assessment_from_trace(
    *,
    trace: ModelCallTrace,
    semantic_run_id: str,
    semantic_job_id: str,
    coverage_run_id: str,
    query_id: str,
    qbank_id: str,
    evidence_proxy_complete: bool,
    config: QBankSemanticMatchConfig | None = None,
) -> QBankSemanticMatchAssessment:
    """Rebuild one assessment from an already audited successful model call."""

    schema = (
        QBankSemanticJudgeEnvelopeV2
        if config is not None and config.is_dimensions_only
        else QBankSemanticJudgeEnvelope
    )
    parsed = schema.model_validate(trace.parsed_output)
    result = parsed.result
    if result.pair_alias != "P01":
        raise ValueError("semantic judge returned an unexpected pair alias")
    dimensions_passed = (
        result.dimensions.same_information_need
        and result.dimensions.candidate_answer_sufficient
        and result.dimensions.contradiction_free
        and result.dimensions.query_image_binding_compatible is not False
    )
    reason_codes = semantic_reason_codes(result.dimensions)
    assessment_id = "qbank-semantic-assessment-" + stable_hash(
        [semantic_run_id, query_id, qbank_id]
    )[:24]
    return QBankSemanticMatchAssessment(
        semantic_assessment_id=assessment_id,
        semantic_run_id=semantic_run_id,
        semantic_job_id=semantic_job_id,
        coverage_run_id=coverage_run_id,
        query_id=query_id,
        qbank_id=qbank_id,
        model_call_id=trace.model_call_id,
        dimensions=result.dimensions,
        reason_codes=reason_codes,
        semantic_dimensions_passed=dimensions_passed,
        evidence_proxy_complete=evidence_proxy_complete,
        semantic_match=dimensions_passed and evidence_proxy_complete,
        elapsed_ms=trace.latency_ms or 0.0,
        content_hash=canonical_content_hash(
            {
                "semantic_run_id": semantic_run_id,
                "query_id": query_id,
                "qbank_id": qbank_id,
                "dimensions": result.dimensions.model_dump(mode="json"),
                "reason_codes": [item.value for item in reason_codes],
                "evidence_proxy_complete": evidence_proxy_complete,
            }
        ),
        metadata={
            "pair_alias": result.pair_alias,
            "conservative_fallback": False,
            "reason_codes_server_derived": True,
        },
    )


def semantic_reason_codes(
    dimensions: QBankSemanticJudgeDimensions,
) -> list[QBankSemanticReason]:
    """Derive stable reason codes from the four authoritative Judge dimensions."""

    reason_by_dimension = {
        "same_information_need": QBankSemanticReason.DIFFERENT_INFORMATION_NEED,
        "candidate_answer_sufficient": (
            QBankSemanticReason.CANDIDATE_ANSWER_INSUFFICIENT
        ),
        "contradiction_free": QBankSemanticReason.CONTRADICTION,
        "query_image_binding_compatible": (
            QBankSemanticReason.QUERY_IMAGE_BINDING_MISMATCH
        ),
    }
    return [
        reason
        for name, reason in reason_by_dimension.items()
        if getattr(dimensions, name) is False
    ]


def select_semantic_candidate_union(
    candidates: Sequence[QBankCoverageCandidate],
    *,
    candidate_pool_k: int = 50,
) -> tuple[QBankCoverageCandidate, ...]:
    selected: dict[tuple[str, str], QBankCoverageCandidate] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.query_id,
            item.question_budget,
            item.rank,
            item.qbank_id,
        ),
    ):
        if candidate.rank > candidate_pool_k:
            continue
        selected.setdefault((candidate.query_id, candidate.qbank_id), candidate)
    return tuple(selected[key] for key in sorted(selected))


def derive_qbank_semantic_metrics(
    *,
    semantic_run_id: str,
    candidates: Sequence[QBankCoverageCandidate],
    assessments: Sequence[QBankSemanticMatchAssessment],
    config: QBankSemanticMatchConfig,
    query_ids: Sequence[str] | None = None,
) -> tuple[QBankSemanticMetric, ...]:
    by_pair = {
        (item.query_id, item.qbank_id): item.semantic_match for item in assessments
    }
    ranked: dict[tuple[str, int], list[QBankCoverageCandidate]] = {}
    for candidate in candidates:
        if candidate.rank <= config.candidate_pool_k:
            ranked.setdefault(
                (candidate.query_id, candidate.question_budget), []
            ).append(candidate)
    if query_ids is not None:
        for query_id in sorted(set(query_ids)):
            for budget in config.question_budgets:
                ranked.setdefault((query_id, budget), [])
    metrics: list[QBankSemanticMetric] = []
    for (query_id, budget), values in sorted(ranked.items()):
        values.sort(key=lambda item: (item.rank, item.qbank_id))
        for k in config.match_ks:
            matched = any(
                by_pair.get((query_id, item.qbank_id), False)
                for item in values[:k]
            )
            for metric_name, metric_value in (
                ("qbank_semantic_match_at_k", float(matched)),
                ("semantic_projected_fallback_at_k", float(not matched)),
            ):
                metrics.append(
                    QBankSemanticMetric(
                        semantic_metric_id="qbank-semantic-metric-"
                        + stable_hash(
                            [
                                semantic_run_id,
                                query_id,
                                metric_name,
                                budget,
                                k,
                            ]
                        )[:24],
                        semantic_run_id=semantic_run_id,
                        query_id=query_id,
                        metric_name=metric_name,
                        metric_version=config.protocol_version,
                        question_budget=budget,
                        k=k,
                        value=metric_value,
                        metadata={"candidate_pool_k": config.candidate_pool_k},
                    )
                )
    return tuple(metrics)


def semantic_pair_accounting(
    assessments: Sequence[QBankSemanticMatchAssessment],
) -> Mapping[str, int]:
    return {
        "pair_count": len(assessments),
        "semantic_dimensions_passed": sum(
            item.semantic_dimensions_passed for item in assessments
        ),
        "evidence_proxy_complete": sum(
            item.evidence_proxy_complete for item in assessments
        ),
        "semantic_match": sum(item.semantic_match for item in assessments),
    }
