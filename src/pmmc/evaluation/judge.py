"""Method-blind structured judge request and local JSON repair."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, model_validator

from pmmc.benchmarks.ids import stable_hash
from pmmc.evaluation.config import BaselineEvaluationConfig
from pmmc.models import ModelCallContext, ModelExecutionService, ParsedModelOutput
from pmmc.models.protocols import GenerationGateway
from pmmc.schemas import (
    AppStage,
    BadCaseCategory,
    GenerationRequest,
    JudgeScore,
    JudgeVerdict,
    ModelCallTrace,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    QueryInput,
    SpeakerRole,
)


class JudgeVerdictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    verdict: JudgeVerdict
    score: float
    reason: str

    @model_validator(mode="after")
    def _validate_score(self) -> JudgeVerdictPayload:
        expected = {
            JudgeVerdict.INCORRECT: 0.0,
            JudgeVerdict.PARTIALLY_CORRECT: 0.5,
            JudgeVerdict.CORRECT: 1.0,
        }[self.verdict]
        if self.score != expected:
            raise ValueError("judge verdict and score disagree")
        if not self.reason.strip():
            raise ValueError("judge reason cannot be empty")
        return self


def _decode_payload(text: str) -> JudgeVerdictPayload:
    value = json.loads(text)
    return JudgeVerdictPayload.model_validate(value)


def parse_judge_output(text: str) -> ParsedModelOutput:
    """Parse strict JSON, then perform at most one local repair attempt."""

    try:
        payload = _decode_payload(text)
        return ParsedModelOutput(value=payload.model_dump(mode="json"))
    except Exception as initial:
        initial_error = " ".join(str(initial).split())[:1000] or type(initial).__name__
    try:
        from json_repair import repair_json
    except ModuleNotFoundError as exc:
        raise ValueError("json-repair is required for judge output repair") from exc
    repaired = repair_json(text)
    payload = _decode_payload(repaired)
    return ParsedModelOutput(
        value=payload.model_dump(mode="json"),
        repair_attempt_count=1,
        initial_error=initial_error,
    )


def build_judge_request(
    *,
    evaluation_run_id: str,
    answer_id: str,
    query: QueryInput,
    candidate_answer: str,
    reference_answer: str,
    prompt: str,
    config: BaselineEvaluationConfig,
    timeout_seconds: float,
) -> GenerationRequest:
    """Build a method-blind request; no baseline or retrieval fields are accepted."""

    request_id = "judge-request-" + stable_hash(
        [evaluation_run_id, answer_id, config.judge_prompt_version, 0]
    )[:24]
    payload = json.dumps(
        {
            "question": query.raw_query,
            "candidate_answer": candidate_answer,
            "reference_answer": reference_answer,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    parts = [ModelContentPart(kind=ModelContentKind.TEXT, text=payload)]
    parts.extend(
        ModelContentPart(kind=ModelContentKind.IMAGE, image=image)
        for image in query.query_image_refs
    )
    return GenerationRequest(
        request_id=request_id,
        model_alias=config.judge_model_alias,
        messages=[
            ModelMessage(
                role=SpeakerRole.SYSTEM,
                parts=[ModelContentPart(kind=ModelContentKind.TEXT, text=prompt)],
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
                "name": "baseline_judge_verdict",
                "strict": True,
                "schema": JudgeVerdictPayload.model_json_schema(),
            },
        },
        metadata={
            "task": "baseline_judge",
            "protocol_version": config.protocol_version,
            "rubric_version": config.judge_rubric_version,
        },
    )


class BaselineJudgeService:
    def __init__(
        self,
        execution: ModelExecutionService,
        gateway: GenerationGateway,
        config: BaselineEvaluationConfig,
        prompt: str,
        timeout_seconds: float,
    ) -> None:
        self.execution = execution
        self.gateway = gateway
        self.config = config
        self.prompt = prompt
        self.timeout_seconds = timeout_seconds

    async def judge(
        self,
        *,
        evaluation_run_id: str,
        evaluation_job_id: str,
        attempt_number: int,
        run_id: str,
        snapshot_id: str,
        answer_id: str,
        query: QueryInput,
        candidate_answer: str,
        reference_answer: str,
    ) -> JudgeScore:
        model_call_id = "judge-call-" + stable_hash(
            [evaluation_run_id, answer_id, attempt_number]
        )[:24]
        audited = await self.execution.generate_audited(
            self.gateway,
            build_judge_request(
                evaluation_run_id=evaluation_run_id,
                answer_id=answer_id,
                query=query,
                candidate_answer=candidate_answer,
                reference_answer=reference_answer,
                prompt=self.prompt,
                config=self.config,
                timeout_seconds=self.timeout_seconds,
            ),
            ModelCallContext(
                model_call_id=model_call_id,
                run_id=run_id,
                evaluation_run_id=evaluation_run_id,
                evaluation_job_id=evaluation_job_id,
                snapshot_id=snapshot_id,
                query_id=query.query_id,
                conversation_id=query.conversation_id,
                stage=AppStage.JUDGE,
            ),
            output_parser=lambda result: parse_judge_output(result.text),
        )
        return judge_score_from_trace(
            trace=audited.trace,
            evaluation_run_id=evaluation_run_id,
            answer_id=answer_id,
            query_id=query.query_id,
            run_id=run_id,
            config=self.config,
        )


def judge_score_from_trace(
    *,
    trace: ModelCallTrace,
    evaluation_run_id: str,
    answer_id: str,
    query_id: str,
    run_id: str,
    config: BaselineEvaluationConfig,
) -> JudgeScore:
    """Rebuild a judge fact from a successful audited call after a crash."""

    parsed = JudgeVerdictPayload.model_validate(trace.parsed_output)
    return JudgeScore(
        judge_score_id="judge-score-"
        + stable_hash(
            [
                evaluation_run_id,
                answer_id,
                config.judge_model_alias,
                config.judge_rubric_version,
                0,
            ]
        )[:24],
        evaluation_run_id=evaluation_run_id,
        answer_id=answer_id,
        query_id=query_id,
        run_id=run_id,
        judge_model_alias=config.judge_model_alias,
        judge_actual_model_id=trace.actual_model_id,
        judge_prompt_hash=trace.prompt_hash,
        model_call_id=trace.model_call_id,
        rubric_version=config.judge_rubric_version,
        repeat_index=0,
        verdict=parsed.verdict,
        llm_judge_score=parsed.score,
        judge_reason=parsed.reason,
        bad_case_category=BadCaseCategory.UNKNOWN,
        created_at=trace.created_at,
    )
