"""Method-neutral audited answer generation for PMMC and baseline adapters."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence

from pydantic import ValidationError

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.models.execution import (
    GenerationOutputParser,
    ModelCallContext,
    ModelExecutionService,
    ParsedModelOutput,
)
from pmmc.models.protocols import GenerationGateway
from pmmc.models.provider_config import validate_typed_answer_decoding
from pmmc.runtime.answer_diagnostics import (
    answer_alias_diagnostics,
    typed_answer_schema_diagnostics,
)
from pmmc.runtime.grounding import (
    GroundingCallback,
    GroundingEngineeringError,
    grounding_evidence_hash,
    load_step8_prompt,
    role_call,
    source_content_parts,
    strict_json_object,
)
from pmmc.schemas import (
    AnswerContractFailure,
    AnswerGenerationInput,
    AppStage,
    FinalAnswerRecord,
    GenerationRequest,
    GenerationResult,
    ModelCallTrace,
    ModelContentPart,
    ModelMessage,
    QueryInput,
    StructuredAnswerFact,
    StructuredAnswerOutput,
    ThinkingMode,
    TraceStatus,
)
from pmmc.schemas.answer_output import (
    TYPED_ANSWER_CONTRACT_VERSION,
    TYPED_ANSWER_GUIDED_SCHEMA_VERSION,
    TYPED_ANSWER_OUTPUT_TOKENS,
    AnswerAbstainReason,
    AnswerContractV3,
    AnswerEvidenceSource,
    AnswerImageBinding,
    AnswerKind,
    AnswerRoleCall,
    TypedAnswerCandidate,
    TypedAnswerContext,
    TypedAnswerImageItem,
    TypedAnswerResult,
    typed_answer_response_schema,
)


class EmptyRuntimeAnswer(RuntimeError):
    """The answer provider returned no usable text."""


class InvalidRuntimeAnswer(RuntimeError):
    """A provider response violated the frozen public answer contract."""

    def __init__(
        self,
        failures: list[AnswerContractFailure],
        *,
        attempt_failures: list[list[AnswerContractFailure]] | None = None,
        schema_diagnostics: dict[str, object] | None = None,
        attempt_schema_diagnostics: Sequence[dict[str, object]] | None = None,
    ) -> None:
        self.failures = tuple(dict.fromkeys(failures))
        self.attempt_failures = tuple(
            tuple(dict.fromkeys(values))
            for values in (attempt_failures or [failures])
        )
        self.schema_diagnostics = schema_diagnostics or {}
        self.attempt_schema_diagnostics = tuple(
            attempt_schema_diagnostics or [self.schema_diagnostics]
        )
        super().__init__("strict answer contract failed: " + ",".join(self.failures))


class UnsupportedRuntimeAnswer(RuntimeError):
    """The model declared the supplied evidence insufficient; not a correctness judgment."""

    def __init__(self, message: str, *, output: StructuredAnswerOutput | None = None) -> None:
        self.output = output
        self.fact: StructuredAnswerFact | None = None
        super().__init__(message)


ABSTENTION_POLICY = "independent-answer-abstention-v1"


def abstention_fact(
    prepared: AnswerGenerationInput, trace: ModelCallTrace,
    output: StructuredAnswerOutput, *, attempt: int = 0,
) -> StructuredAnswerFact:
    if output.supported:
        raise ValueError("abstention cannot contain a supported answer")
    payload = {
        "run_id": prepared.run_id, "query_id": prepared.query.query_id,
        "answer_variant": prepared.answer_variant,
        "attempt_ordinal": attempt, "model_call_id": trace.model_call_id,
        "answer_id": None, **output.model_dump(mode="json"),
        "parse_status": "accepted", "provider_schema_conformant": True,
        "finish_reason": (trace.parsed_output or {}).get("finish_reason"),
        "metadata": {
            "terminal_policy": ABSTENTION_POLICY,
            "answer_contract_version": "strict-answer-contract-v2",
            "method": prepared.method_id,
            "source_lineage": prepared.source_lineage,
        },
    }
    return StructuredAnswerFact.model_validate({
        "structured_answer_output_id": stable_compiler_id("answer-abstention", payload),
        "content_hash": canonical_content_hash(payload), **payload,
    })


def validate_abstention_fact(fact: StructuredAnswerFact) -> None:
    payload = fact.model_dump(mode="json", exclude={
        "structured_answer_output_id", "content_hash", "created_at",
    })
    if (
        fact.supported or fact.answer_id is not None
        or fact.parse_status != "accepted" or not fact.provider_schema_conformant
        or fact.finish_reason == "length"
        or fact.metadata.get("terminal_policy") != ABSTENTION_POLICY
        or fact.metadata.get("answer_contract_version") != "strict-answer-contract-v2"
        or fact.content_hash != canonical_content_hash(payload)
    ):
        raise ValueError("invalid independent answer abstention fact")


_INTERNAL_ID_RE = re.compile(
    r"\b(?:memory|session|image|caption|qbank|plan|execution|compiler)[_:-][a-z0-9][\w.:-]*\b",
    flags=re.IGNORECASE,
)
_URL_RE = re.compile(r"(?:https?://|s3://|file://|presigned)", flags=re.IGNORECASE)
_HARNESS_RE = re.compile(
    r"\b(?:harness|local alias|evidence alias|system prompt|bounded opportunity)\b",
    flags=re.IGNORECASE,
)

_TYPED_FORMAT_RETRY_INSTRUCTION = (
    "Return only the six required typed-answer JSON fields. Choose answer_kind from the query. "
    "Text/number/boolean require answer_text and no selected images; image kinds require null "
    "answer_text and unique supplied I-prefixed selected_image_aliases. Single image means exactly "
    "one. Cite A/E-prefixed sources only in used_evidence_aliases. Never leak aliases, IDs, URLs "
    "or thinking in answer_text. If evidence is insufficient, return a legal abstention with "
    "null answer_text, no selected images, supported false and an allowed reason. "
    "Do not invent support. Keep the entire response within 512 tokens."
)


def make_typed_answer_context(sources: Sequence[AnswerEvidenceSource]) -> TypedAnswerContext:
    sources = [source.model_copy(update={"images": [item.model_copy(update={
        "image": item.image.model_copy(update={"metadata": {}, "caption": None})
        if item.image is not None else None,
    }) for item in source.images]}) for source in sources]
    owners: dict[str, list[str]] = {}
    for source in sources:
        for image in source.images:
            owners.setdefault(image.image_id, []).append(source.evidence_alias)
    return TypedAnswerContext(sources=list(sources), image_bindings=[
        AnswerImageBinding(
            image_alias=f"I{ordinal:02d}", image_id=image_id,
            public_media_ref="media:" + canonical_content_hash(["answer-media-v1", image_id]),
            evidence_aliases=aliases,
        ) for ordinal, (image_id, aliases) in enumerate(owners.items(), 1)
    ])


def prepare_typed_answer_request(
    request: GenerationRequest, *, query: QueryInput,
    evidence_sources: Sequence[AnswerEvidenceSource],
) -> GenerationRequest:
    """Explicit v3 adapter used before exact context budgeting, never on historical v2."""
    if request.metadata.get("answer_contract_version") != TYPED_ANSWER_CONTRACT_VERSION:
        raise ValueError("typed adapter requires explicit v3 contract identity")
    context = make_typed_answer_context(evidence_sources)
    prompt = load_step8_prompt("pmmc_answer_contract_v3.txt")
    parts = [ModelContentPart(kind="text", text="[SOURCE EVIDENCE]")]
    parts.extend(source_content_parts(context))
    parts.append(ModelContentPart(kind="text", text=json.dumps({
        "query": query.raw_query,
    }, ensure_ascii=False)))
    shown_images = {p.image.image_id for p in parts if p.image is not None}
    for image in query.query_image_refs:
        if image.image_id in shown_images:
            continue
        parts.append(ModelContentPart(kind="image", image=image.model_copy(update={
            "metadata": {}, "caption": None,
        })))
    return request.model_copy(update={
        "messages": [
            ModelMessage(role="system", parts=[ModelContentPart(kind="text", text=prompt)]),
            ModelMessage(role="user", parts=parts),
        ],
        "max_output_tokens": TYPED_ANSWER_OUTPUT_TOKENS,
        "provider_max_output_tokens": None,
        "thinking_mode": ThinkingMode.DISABLED,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "typed_answer_contract_v3", "strict": True,
            "schema": typed_answer_response_schema(context),
        }},
        "metadata": {
            "answer_contract_version": TYPED_ANSWER_CONTRACT_VERSION,
            "answer_guided_schema": TYPED_ANSWER_GUIDED_SCHEMA_VERSION,
            "answer_prompt_hash": canonical_content_hash(prompt),
            "answer_prompt_version": "pmmc-answer-contract-v3", "model_role": "answer",
            "query_content_hash": canonical_content_hash([
                query.raw_query, [i.model_dump(mode="json", exclude={"metadata", "caption"})
                                  for i in query.query_image_refs],
            ]),
            "evidence_aliases": [s.evidence_alias for s in context.sources],
            "typed_answer_context": context.model_dump(mode="json"),
        },
    })


def validate_typed_answer_input(prepared: AnswerGenerationInput) -> TypedAnswerContext:
    if (
        prepared.metadata.get("answer_contract_version") != TYPED_ANSWER_CONTRACT_VERSION
        or prepared.request.metadata.get("answer_contract_version") != TYPED_ANSWER_CONTRACT_VERSION
        or prepared.request.max_output_tokens != TYPED_ANSWER_OUTPUT_TOKENS
    ):
        raise ValueError("typed answer contract identity or decoding mismatch")
    validate_typed_answer_decoding(prepared.request)
    context = TypedAnswerContext.model_validate(prepared.metadata.get("typed_answer_context"))
    if [s.evidence_id for s in context.sources] != prepared.evidence_ids_in_prompt:
        raise ValueError("typed answer source lineage differs from actual answer evidence")
    expected = prepare_typed_answer_request(
        prepared.request, query=prepared.query, evidence_sources=context.sources,
    )
    if (
        context.model_dump(mode="json") != prepared.request.metadata.get("typed_answer_context")
        or prepared.request.messages not in (
            expected.messages, answer_format_retry_request(expected).messages,
        )
        or prepared.request.response_format != expected.response_format
    ):
        raise ValueError("typed answer request differs from its source-only adapter")
    return context


def typed_answer_public_output_tokens(result: GenerationResult) -> int:
    """Provider-reported visible usage for v3 validation, never a billing mutation."""
    total = result.usage.output_tokens
    if result.metadata.get("provider") != "openai":
        return total
    reasoning = result.usage.metadata.get("reasoning_tokens")
    if (
        result.metadata.get("effective_reasoning_mode") not in {"minimal", "low"}
        or result.usage.metadata.get("usage_reported") is not True
        or type(reasoning) is not int
        or not 0 <= reasoning <= total <= 4096
    ):
        raise ValueError("typed GPT output requires valid provider-reported reasoning usage")
    return total - reasoning


def _typed_provider_output(result: GenerationResult) -> ParsedModelOutput:
    """Persist visible output usage without subtracting reasoning from paid usage."""
    if result.metadata.get("provider") != "openai":
        raise ValueError("typed reasoning requires the OpenAI provider")
    public = typed_answer_public_output_tokens(result)
    return ParsedModelOutput(value={
        "text": result.text,
        "finish_reason": result.finish_reason,
        "public_output_tokens": public,
        "reasoning_tokens": result.usage.output_tokens - public,
    }, repair_attempt_count=0)


def _typed_output_tokens(request: GenerationRequest, trace: ModelCallTrace) -> int | None:
    if request.thinking_mode not in {ThinkingMode.MINIMAL, ThinkingMode.LOW}:
        return trace.output_tokens
    public = (trace.parsed_output or {}).get("public_output_tokens")
    reasoning = (trace.parsed_output or {}).get("reasoning_tokens")
    if (
        type(public) is not int or type(reasoning) is not int
        or public < 0 or reasoning < 0
        or trace.output_tokens is None or public + reasoning != trace.output_tokens
        or trace.output_tokens > 4096
    ):
        raise ValueError("typed GPT trace is missing verified public/reasoning output usage")
    return public


def parse_typed_answer(
    text: str, *, context: TypedAnswerContext, finish_reason: str | None,
    output_tokens: int | None = None,
    forbidden_ids: Sequence[str] = (),
) -> AnswerContractV3:
    failures = []
    if finish_reason == "length":
        failures.append(AnswerContractFailure.FINISH_REASON_LENGTH)
    elif finish_reason not in {"stop", "eos", "end_turn"}:
        failures.append(AnswerContractFailure.INVALID_FINISH_REASON)
    if output_tokens is not None and output_tokens > TYPED_ANSWER_OUTPUT_TOKENS:
        failures.append(AnswerContractFailure.OUTPUT_LIMIT_EXCEEDED)
    try:
        payload = strict_json_object(text)
    except (ValueError, TypeError):
        raise InvalidRuntimeAnswer([*failures, AnswerContractFailure.INVALID_JSON]) from None
    try:
        output = AnswerContractV3.model_validate(payload)
    except ValidationError as exc:
        raise InvalidRuntimeAnswer(
            [*failures, AnswerContractFailure.SCHEMA_INVALID],
            schema_diagnostics=typed_answer_schema_diagnostics(exc),
        ) from None
    if (
        payload["selected_image_aliases"] != output.selected_image_aliases
        or payload["used_evidence_aliases"] != output.used_evidence_aliases
    ):
        failures.append(AnswerContractFailure.SCHEMA_INVALID)
    evidence_aliases = {s.evidence_alias for s in context.sources}
    image_aliases = {i.image_alias for i in context.image_bindings}
    if not set(output.used_evidence_aliases) <= evidence_aliases:
        failures.append(AnswerContractFailure.UNKNOWN_EVIDENCE_ALIAS)
    if not set(output.selected_image_aliases) <= image_aliases:
        failures.append(AnswerContractFailure.UNKNOWN_IMAGE_ALIAS)
    answer = output.answer_text or ""
    if re.search(r"</?think\b", answer, flags=re.IGNORECASE):
        failures.append(AnswerContractFailure.THINKING_TAG_LEAK)
    if re.search(r"\b[AEI][0-9]{2,}\b", answer, flags=re.IGNORECASE):
        failures.append(AnswerContractFailure.EVIDENCE_ALIAS_LEAK)
    private_ids = [s.evidence_id for s in context.sources] + [
        i.image_id for i in context.image_bindings
    ] + list(forbidden_ids)
    if _INTERNAL_ID_RE.search(answer) or any(
        re.search(rf"(?<!\w){re.escape(value)}(?!\w)", answer, re.IGNORECASE)
        for value in private_ids
    ):
        failures.append(AnswerContractFailure.INTERNAL_ID_LEAK)
    if _URL_RE.search(answer) or re.search(r"(?:^|\s)/(?:Users|home|tmp|private)/", answer):
        failures.append(AnswerContractFailure.URL_LEAK)
    if _HARNESS_RE.search(answer):
        failures.append(AnswerContractFailure.HARNESS_LEAK)
    if failures:
        raise InvalidRuntimeAnswer(failures)
    if output.answer_kind is AnswerKind.IMAGE_SET:
        selected = set(output.selected_image_aliases)
        output = output.model_copy(update={
            "selected_image_aliases": [i.image_alias for i in context.image_bindings
                                       if i.image_alias in selected],
        })
    return output


def map_typed_image_items(
    output: AnswerContractV3, context: TypedAnswerContext,
) -> list[TypedAnswerImageItem]:
    by_alias = {b.image_alias: b for b in context.image_bindings}
    if not set(output.selected_image_aliases) <= set(by_alias):
        raise InvalidRuntimeAnswer([AnswerContractFailure.UNKNOWN_IMAGE_ALIAS])
    aliases = output.selected_image_aliases
    if output.answer_kind is AnswerKind.IMAGE_SET:
        aliases = [b.image_alias for b in context.image_bindings if b.image_alias in set(aliases)]
    return [TypedAnswerImageItem(
        ordinal=index, image_alias=alias, image_id=by_alias[alias].image_id,
        public_media_ref=by_alias[alias].public_media_ref,
    ) for index, alias in enumerate(aliases)]


def _validate_typed_recovery_trace(prepared: AnswerGenerationInput, trace: ModelCallTrace) -> None:
    if (
        trace.status is not TraceStatus.SUCCESS or trace.stage is not AppStage.ANSWER
        or trace.run_id != prepared.run_id or trace.query_id != prepared.query.query_id
        or trace.snapshot_id != prepared.snapshot_id
        or trace.model_alias != prepared.request.model_alias
        or trace.request_config.get("request_metadata", {}).get("answer_contract_version")
        != TYPED_ANSWER_CONTRACT_VERSION
        or trace.request_config.get("request_metadata", {}).get("model_role")
        not in {"answer", "answer_format_retry"}
        or trace.request_config.get("request_metadata", {}).get("typed_answer_context")
        != prepared.request.metadata.get("typed_answer_context")
        or trace.request_config.get("request_metadata", {}).get("answer_prompt_hash")
        != prepared.request.metadata.get("answer_prompt_hash")
        or trace.request_config.get("request_metadata", {}).get("query_content_hash")
        != prepared.request.metadata.get("query_content_hash")
        or trace.request_config.get("request_metadata", {}).get("provider_answer_policy")
        != prepared.request.metadata.get("provider_answer_policy")
    ):
        raise ValueError("typed answer recovery trace has incompatible lineage")


def _answer_schema_diagnostics(payload: object, error: ValidationError) -> dict[str, object]:
    """Keep fixed field names and error types, never values/messages/extra keys."""
    errors = error.errors(include_input=False, include_context=False, include_url=False)
    fields = set(StructuredAnswerOutput.model_fields)
    locations = sorted({
        (str(item["loc"][0]) if item["loc"] and item["loc"][0] in fields
         else "object" if not item["loc"] else "extra_field")
        for item in errors
    })
    issues = []
    if isinstance(payload, dict):
        aliases = payload.get("used_evidence_aliases", [])
        if isinstance(aliases, list) and all(isinstance(item, str) for item in aliases):
            normalized = [item.strip() for item in aliases]
            if len(normalized) != len(set(normalized)):
                issues.append("duplicate_evidence_aliases")
        if payload.get("supported") is True:
            if payload.get("answer") is None:
                issues.append("supported_answer_missing")
            if payload.get("abstain_reason") is not None:
                issues.append("supported_has_abstain_reason")
            if not aliases:
                issues.append("supported_evidence_empty")
        elif payload.get("supported") is False:
            if payload.get("answer") is not None:
                issues.append("unsupported_answer_present")
            if payload.get("abstain_reason") is None:
                issues.append("unsupported_reason_missing")
    return {
        "validation_error_count": len(errors),
        "validation_error_types": dict(sorted(Counter(item["type"] for item in errors).items())),
        "validation_fields": locations,
        "cross_field_issues": issues,
    }


_ANSWER_FORMAT_RETRY_INSTRUCTION = (
    "Correct the response format and return only the JSON object required by the schema. "
    "Include all four fields. If supported is true, answer must be nonempty text, "
    "abstain_reason must be null, and used_evidence_aliases must contain at least one "
    "supplied alias. If supported is false, answer must be null and abstain_reason "
    "must be an allowed enum value. Never repeat an evidence alias. "
    "Do not invent support or change the evidence boundary to satisfy the format."
)


def answer_format_retry_request(request: GenerationRequest) -> GenerationRequest:
    """Share the exact retry suffix with context budgeting; never echo provider output."""
    from pmmc.runtime.answer_policy import current_retry_instruction

    instruction = (
        _TYPED_FORMAT_RETRY_INSTRUCTION
        if answer_retry_policy(request) == "typed-answer-format-feedback-v3"
        else current_retry_instruction()
        if answer_retry_policy(request) == "answer-format-feedback-v2"
        else _ANSWER_FORMAT_RETRY_INSTRUCTION
    )
    return request.model_copy(update={
        "messages": [
            *request.messages,
            ModelMessage(role="user", parts=[ModelContentPart(
                kind="text", text=instruction,
            )]),
        ],
    })


def answer_retry_policy(request: GenerationRequest) -> str:
    if request.metadata.get("answer_contract_version") == TYPED_ANSWER_CONTRACT_VERSION:
        return "typed-answer-format-feedback-v3"
    from pmmc.runtime.answer_policy import (
        CURRENT_PROMPT_HASH,
        CURRENT_RETRY_POLICY,
        CURRENT_RUNTIME_PROMPT_HASH,
    )

    prompt_hash = request.metadata.get("answer_prompt_hash") or request.metadata.get("prompt_hash")
    return (
        CURRENT_RETRY_POLICY
        if prompt_hash in {CURRENT_PROMPT_HASH, CURRENT_RUNTIME_PROMPT_HASH}
        else "answer-format-feedback-v1"
    )


def answer_retry_call_id(request: GenerationRequest, model_call_id: str) -> str:
    policy = answer_retry_policy(request)
    suffix = [1, policy] if policy != "answer-format-feedback-v1" else [1]
    return stable_compiler_id("answer-format-retry-model-call", [model_call_id, *suffix])


def _strict_answer_output(
    text: str,
    *,
    finish_reason: str | None,
    allowed_aliases: list[str],
) -> StructuredAnswerOutput:
    failures: list[AnswerContractFailure] = []
    if finish_reason == "length":
        failures.append(AnswerContractFailure.FINISH_REASON_LENGTH)
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        # Do not log decoder messages/snippets. Shape and offset distinguish
        # open strings, whitespace loops, and non-JSON output without content.
        value = text if isinstance(text, str) else ""
        diagnostics: dict[str, object] = {
            "parse_layer": "json", "json_error_type": type(exc).__name__,
            "json_error_offset": exc.pos if isinstance(exc, json.JSONDecodeError) else None,
            "response_char_count": len(value),
            "response_whitespace_count": sum(char.isspace() for char in value),
            "response_control_count": sum(ord(char) < 32 for char in value),
            "response_starts_object": value.lstrip().startswith("{"),
            "response_ends_object": value.rstrip().endswith("}"),
        }
        raise InvalidRuntimeAnswer(
            [*failures, AnswerContractFailure.INVALID_JSON], schema_diagnostics=diagnostics,
        ) from None
    try:
        output = StructuredAnswerOutput.model_validate(payload)
    except ValidationError as exc:
        raise InvalidRuntimeAnswer(
            [*failures, AnswerContractFailure.SCHEMA_INVALID],
            schema_diagnostics=_answer_schema_diagnostics(payload, exc),
        ) from None
    allowed = set(allowed_aliases)
    if not set(output.used_evidence_aliases) <= allowed:
        failures.append(AnswerContractFailure.UNKNOWN_EVIDENCE_ALIAS)
    answer = output.answer or ""
    lowered = answer.casefold()
    if "<think>" in lowered or "</think>" in lowered:
        failures.append(AnswerContractFailure.THINKING_TAG_LEAK)
    if re.search(r"\[E\d{2,}\]", answer, flags=re.IGNORECASE) or any(
        re.search(rf"\b{re.escape(alias)}\b", answer, flags=re.IGNORECASE)
        for alias in allowed
    ):
        failures.append(AnswerContractFailure.EVIDENCE_ALIAS_LEAK)
    if _INTERNAL_ID_RE.search(answer):
        failures.append(AnswerContractFailure.INTERNAL_ID_LEAK)
    if _HARNESS_RE.search(answer):
        failures.append(AnswerContractFailure.HARNESS_LEAK)
    if _URL_RE.search(answer):
        failures.append(AnswerContractFailure.URL_LEAK)
    if failures:
        raise InvalidRuntimeAnswer(
            failures,
            schema_diagnostics=(
                answer_alias_diagnostics(answer, allowed_aliases)
                if AnswerContractFailure.EVIDENCE_ALIAS_LEAK in failures else {}
            ),
        )
    if not output.supported:
        raise UnsupportedRuntimeAnswer(
            "answer model abstained under the strict contract", output=output,
        )
    return output


def strict_final_answer_failure_codes(
    answer: FinalAnswerRecord,
) -> tuple[AnswerContractFailure, ...]:
    """Validate a persisted public answer without requiring its provider payload."""

    failures: list[AnswerContractFailure] = []
    metadata = answer.metadata
    if metadata.get("answer_contract_version") != "strict-answer-contract-v2":
        failures.append(AnswerContractFailure.CONTRACT_IDENTITY_MISMATCH)
    if answer.model_call_id is None:
        failures.append(AnswerContractFailure.MODEL_CALL_LINEAGE_MISSING)
    if metadata.get("provider_schema_conformant") is not True:
        failures.append(AnswerContractFailure.PROVIDER_SCHEMA_NONCONFORMANT)
    if metadata.get("structured_supported") is not True:
        failures.append(AnswerContractFailure.UNSUPPORTED_FINAL_RECORD)
    if metadata.get("finish_reason") == "length":
        failures.append(AnswerContractFailure.FINISH_REASON_LENGTH)

    aliases = metadata.get("used_evidence_aliases")
    if (
        not isinstance(aliases, list)
        or not aliases
        or any(not isinstance(value, str) or not value for value in aliases)
        or len(set(aliases)) != len(aliases)
    ):
        failures.append(AnswerContractFailure.UNKNOWN_EVIDENCE_ALIAS)

    value = answer.final_answer
    lowered = value.casefold()
    if "<think>" in lowered or "</think>" in lowered:
        failures.append(AnswerContractFailure.THINKING_TAG_LEAK)
    if re.search(r"\[E\d{2,}\]", value, flags=re.IGNORECASE) or (
        isinstance(aliases, list)
        and any(
            isinstance(alias, str)
            and re.search(rf"\b{re.escape(alias)}\b", value, flags=re.IGNORECASE)
            for alias in aliases
        )
    ):
        failures.append(AnswerContractFailure.EVIDENCE_ALIAS_LEAK)
    if _INTERNAL_ID_RE.search(value):
        failures.append(AnswerContractFailure.INTERNAL_ID_LEAK)
    if _HARNESS_RE.search(value):
        failures.append(AnswerContractFailure.HARNESS_LEAK)
    if _URL_RE.search(value):
        failures.append(AnswerContractFailure.URL_LEAK)
    return tuple(dict.fromkeys(failures))


class SharedAnswerGenerationService:
    """Execute a gold-free answer request and persistable method-neutral fact."""

    def __init__(
        self,
        *,
        execution: ModelExecutionService,
        gateway: GenerationGateway | None,
        grounding: GroundingCallback | None = None,
    ) -> None:
        self.execution = execution
        self.gateway = gateway
        self.grounding = grounding

    async def answer(
        self,
        prepared: AnswerGenerationInput,
        *,
        model_call_id: str,
        job_id: str | None = None,
        context_assembly_id: str | None = None,
        start_attempt: int = 0,
        prior_attempt_failures: Sequence[Sequence[AnswerContractFailure]] = (),
        prior_schema_diagnostics: Sequence[dict[str, object]] = (),
        grounding: GroundingCallback | None = None,
        on_grounding_failure: str = "abstain",
    ) -> FinalAnswerRecord | TypedAnswerResult:
        if prepared.metadata.get("answer_contract_version") == TYPED_ANSWER_CONTRACT_VERSION:
            result = await self.answer_typed(
                prepared, model_call_id=model_call_id, job_id=job_id,
                context_assembly_id=context_assembly_id, start_attempt=start_attempt,
                prior_attempt_failures=prior_attempt_failures,
                grounding=grounding, on_grounding_failure=on_grounding_failure,
            )
            if not isinstance(result, TypedAnswerResult):
                raise ValueError("terminal answer generation returned an ungrounded candidate")
            return result
        if self.gateway is None:
            raise ValueError("answer generation requires a gateway")
        strict = prepared.metadata.get("answer_contract_version") == "strict-answer-contract-v2"
        attempt_count = 2 if strict else 1
        if start_attempt < 0 or start_attempt >= attempt_count:
            raise ValueError("answer start_attempt is outside the frozen retry policy")
        audited = None
        structured: StructuredAnswerOutput | None = None
        last_error: InvalidRuntimeAnswer | None = None
        format_failures = [list(values) for values in prior_attempt_failures]
        schema_diagnostics = list(prior_schema_diagnostics)
        for attempt in range(start_attempt, attempt_count):
            request = prepared.request
            call_id = model_call_id
            if attempt:
                retry_policy = answer_retry_policy(request)
                retry_identity = (
                    [attempt, retry_policy] if retry_policy != "answer-format-feedback-v1"
                    else [attempt]
                )
                request = answer_format_retry_request(request).model_copy(
                    update={
                        "request_id": stable_compiler_id(
                            "answer-format-retry-request",
                            [prepared.request.request_id, *retry_identity],
                        ),
                        "metadata": {
                            **prepared.request.metadata,
                            "format_retry_ordinal": attempt,
                            "format_retry_policy": retry_policy,
                            "parent_request_id": prepared.request.request_id,
                        },
                    }
                )
                call_id = answer_retry_call_id(prepared.request, model_call_id)
            audited = await self.execution.generate_audited(
                self.gateway,
                request,
                ModelCallContext(
                    model_call_id=call_id,
                    run_id=prepared.run_id,
                    stage=AppStage.ANSWER,
                    snapshot_id=prepared.snapshot_id,
                    query_id=prepared.query.query_id,
                    conversation_id=prepared.query.conversation_id,
                    job_id=job_id,
                    context_assembly_id=context_assembly_id,
                ),
            )
            if not strict:
                break
            try:
                structured = _strict_answer_output(
                    audited.result.text,
                    finish_reason=audited.result.finish_reason,
                    allowed_aliases=list(prepared.metadata.get("evidence_aliases") or []),
                )
            except InvalidRuntimeAnswer as exc:
                last_error = exc
                format_failures.append(list(exc.failures))
                schema_diagnostics.append(exc.schema_diagnostics)
                print("ANSWER_CONTRACT_ATTEMPT " + json.dumps({
                    "state": "retry_pending" if attempt == 0 else "failed",
                    "attempt_ordinal": attempt,
                    "locator_hash": canonical_content_hash([
                        prepared.run_id, prepared.query.query_id,
                    ])[:16],
                    "failure_codes": [item.value for item in exc.failures],
                    "thinking_mode": request.thinking_mode.value,
                    "output_token_budget": request.max_output_tokens,
                    "output_tokens": getattr(getattr(audited.result, "usage", None),
                                             "output_tokens", None),
                    "guided_schema": request.metadata.get("answer_guided_schema"),
                    "format_retry_policy": answer_retry_policy(request),
                    **exc.schema_diagnostics,
                }, sort_keys=True, separators=(",", ":")), flush=True)
                continue
            except UnsupportedRuntimeAnswer as exc:
                assert exc.output is not None
                exc.fact = abstention_fact(prepared, audited.trace, exc.output, attempt=attempt)
                raise
            break
        assert audited is not None
        if strict and structured is None:
            assert last_error is not None
            raise InvalidRuntimeAnswer(
                list(last_error.failures), attempt_failures=format_failures,
                schema_diagnostics=last_error.schema_diagnostics,
                attempt_schema_diagnostics=schema_diagnostics,
            )
        answer = structured.answer if structured is not None else audited.result.text.strip()
        if not answer:
            raise EmptyRuntimeAnswer("answer model returned empty text")
        answer_id = stable_compiler_id(
            "method-answer",
            [
                prepared.run_id,
                prepared.query.query_id,
                prepared.method_id,
                prepared.answer_variant,
            ],
        )
        return FinalAnswerRecord(
            answer_id=answer_id,
            snapshot_id=prepared.snapshot_id,
            query_id=prepared.query.query_id,
            run_id=prepared.run_id,
            answer_variant=prepared.answer_variant,
            method=prepared.method_id,
            answer_model_alias=prepared.request.model_alias,
            actual_model_id=audited.result.actual_model_id,
            evidence_ids_in_prompt=prepared.evidence_ids_in_prompt,
            image_ids_in_prompt=prepared.image_ids_in_prompt,
            final_answer=answer,
            input_tokens=audited.result.usage.input_tokens,
            output_tokens=audited.result.usage.output_tokens,
            latency_ms=audited.trace.latency_ms,
            model_call_id=audited.trace.model_call_id,
            langfuse_trace_id=audited.trace.langfuse_trace_id,
            metadata={
                "finish_reason": audited.result.finish_reason,
                "source_lineage": prepared.source_lineage,
                "answer_contract_version": prepared.metadata.get("answer_contract_version"),
                "used_evidence_aliases": (
                    structured.used_evidence_aliases if structured is not None else []
                ),
                "provider_schema_conformant": structured is not None if strict else None,
                "format_retried": start_attempt > 0 or audited.trace.model_call_id != model_call_id,
                "format_failure_codes": [
                    [item.value for item in failures]
                    for failures in format_failures
                ],
                "format_schema_diagnostics": schema_diagnostics,
                "structured_supported": (
                    structured.supported if structured is not None else None
                ),
                "structured_abstain_reason": (
                    structured.abstain_reason.value
                    if structured is not None and structured.abstain_reason is not None
                    else None
                ),
            },
        )

    async def answer_typed(
        self, prepared: AnswerGenerationInput, *, model_call_id: str,
        job_id: str | None = None, context_assembly_id: str | None = None,
        start_attempt: int = 0,
        prior_attempt_failures: Sequence[Sequence[AnswerContractFailure]] = (),
        grounding: GroundingCallback | None = None, on_grounding_failure: str = "abstain",
        recovered_trace: ModelCallTrace | None = None,
        defer_grounding: bool = False,
        allow_generation: bool = True,
    ) -> TypedAnswerResult | TypedAnswerCandidate:
        context = validate_typed_answer_input(prepared)
        if not allow_generation and recovered_trace is None:
            raise ValueError("ground-only execution requires a generated answer journal")
        if on_grounding_failure not in {"abstain", "fallback"} or (
            on_grounding_failure == "fallback" and prepared.method_id != "pmmc"
        ):
            raise ValueError("only PMMC can route failed grounding to fallback")
        if start_attempt not in {0, 1}:
            raise ValueError("typed answer allows exactly one format retry")
        failures = [list(f) for f in prior_attempt_failures]
        calls = []
        if recovered_trace is not None:
            journal = recovered_trace.request_config.get("request_metadata", {})
            if journal.get("model_role") == "answer_format_retry":
                start_attempt = 1
                journal_failures = [
                    [AnswerContractFailure(code) for code in attempt]
                    for attempt in journal.get("prior_format_failure_codes", [])
                ]
                if failures and journal_failures and failures != journal_failures:
                    raise ValueError("recovered answer format diagnostics disagree")
                failures = failures or journal_failures
                calls = [AnswerRoleCall.model_validate(call)
                         for call in journal.get("prior_answer_calls", [])]
        output = None
        trace = None
        for attempt in range(start_attempt, 2):
            request = prepared.request
            call_id = model_call_id
            if attempt:
                request = answer_format_retry_request(request).model_copy(update={
                    "request_id": stable_compiler_id(
                        "typed-answer-retry-request", request.request_id,
                    ),
                    "metadata": {**request.metadata, "format_retry_ordinal": 1,
                                 "model_role": "answer_format_retry",
                                 "prior_format_failure_codes": [
                                     [code.value for code in codes] for codes in failures
                                 ],
                                 "prior_answer_calls": [c.model_dump(mode="json") for c in calls]},
                })
                call_id = answer_retry_call_id(prepared.request, model_call_id)
            if recovered_trace is not None:
                trace = recovered_trace
                recovered_trace = None
                _validate_typed_recovery_trace(prepared, trace)
                text = (trace.parsed_output or {}).get("text", "")
                finish_reason = (trace.parsed_output or {}).get("finish_reason")
                output_tokens = _typed_output_tokens(request, trace)
            else:
                if not allow_generation or self.gateway is None:
                    raise ValueError("answer generation requires an enabled gateway")
                parser_options: dict[str, GenerationOutputParser] = {}
                if request.thinking_mode in {ThinkingMode.MINIMAL, ThinkingMode.LOW}:
                    parser_options["output_parser"] = _typed_provider_output
                audited = await self.execution.generate_audited(
                    self.gateway, request, ModelCallContext(
                        model_call_id=call_id, run_id=prepared.run_id, stage=AppStage.ANSWER,
                        snapshot_id=prepared.snapshot_id, query_id=prepared.query.query_id,
                        conversation_id=prepared.query.conversation_id, job_id=job_id,
                        context_assembly_id=context_assembly_id,
                        parent_model_call_id=model_call_id if attempt else None,
                    ),
                    typed_schema_handoff=True,
                    **parser_options,
                )
                trace = audited.trace
                text = audited.result.text
                finish_reason = audited.result.finish_reason
                output_tokens = _typed_output_tokens(request, trace)
            calls.append(role_call(trace, "answer_format_retry" if attempt else "answer"))
            try:
                output = parse_typed_answer(
                    text, context=context, finish_reason=finish_reason, output_tokens=output_tokens,
                    forbidden_ids=[i.image_id for i in prepared.query.query_image_refs],
                )
            except InvalidRuntimeAnswer as exc:
                failures.append(list(exc.failures))
                if not allow_generation or attempt or trace.model_call_id.startswith(
                    "answer-format-retry-model-call",
                ):
                    raise InvalidRuntimeAnswer(
                        list(exc.failures), attempt_failures=failures,
                        schema_diagnostics=exc.schema_diagnostics,
                    ) from None
                continue
            break
        assert output is not None and trace is not None
        ungated = output
        ungated_items = map_typed_image_items(output, context)
        if defer_grounding:
            return TypedAnswerCandidate(
                run_id=prepared.run_id, query_id=prepared.query.query_id, output=output,
                image_items=ungated_items, answer_model_call_id=trace.model_call_id, calls=calls,
                format_failure_codes=failures,
            )
        fact = None
        terminal = "answered" if output.supported else "abstained"
        if output.supported:
            from pmmc.runtime.relation_grounding import grounding_scope_aliases

            gate = grounding or self.grounding
            if gate is None:
                raise GroundingEngineeringError(
                    "supported_typed_answer_requires_grounding", calls=calls,
                )
            fact = await gate.assess(
                prepared, output, answer_model_call_id=trace.model_call_id, job_id=job_id,
            )
            if (
                fact.run_id != prepared.run_id or fact.query_id != prepared.query.query_id
                or fact.answer_model_call_id != trace.model_call_id
                or fact.ungated_output_hash != canonical_content_hash(
                    output.model_dump(mode="json"),
                )
                or fact.used_evidence_aliases != output.used_evidence_aliases
                or fact.evidence_content_hash != grounding_evidence_hash(
                    context, grounding_scope_aliases(prepared, output, context),
                )
                or not fact.calls
                or any(c.role not in {"answer_grounding", "grounding_format_retry"}
                       for c in fact.calls)
            ):
                raise GroundingEngineeringError("grounding_fact_lineage_mismatch", calls=calls)
            calls.extend(fact.calls)
            if not fact.assessment.supported:
                output = AnswerContractV3(
                    answer_kind=AnswerKind.ABSTENTION, answer_text=None, selected_image_aliases=[],
                    supported=False, used_evidence_aliases=[],
                    abstain_reason=AnswerAbstainReason.UNSUPPORTED_BY_EVIDENCE,
                )
                terminal = (
                    "fallback_required" if on_grounding_failure == "fallback" else "abstained"
                )
        return TypedAnswerResult(
            typed_answer_id=stable_compiler_id("typed-answer", [
                prepared.run_id, prepared.query.query_id, prepared.method_id,
                prepared.answer_variant, TYPED_ANSWER_CONTRACT_VERSION,
                canonical_content_hash(prepared.request.model_dump(mode="json")),
                trace.model_call_id, on_grounding_failure,
                fact.grounding_id if fact is not None else None,
            ]),
            run_id=prepared.run_id, snapshot_id=prepared.snapshot_id,
            query_id=prepared.query.query_id,
            method=prepared.method_id, answer_variant=prepared.answer_variant,
            output=output, ungated_output=ungated,
            image_items=map_typed_image_items(output, context), ungated_image_items=ungated_items,
            answer_model_call_id=trace.model_call_id, actual_model_id=trace.actual_model_id,
            terminal_state=terminal, grounding=fact, calls=calls, format_failure_codes=failures,
        )


def recover_runtime_answer(
    prepared: AnswerGenerationInput,
    trace: ModelCallTrace,
) -> FinalAnswerRecord:
    """Rebuild an answer after provider success and before answer persistence."""

    if prepared.metadata.get("answer_contract_version") == TYPED_ANSWER_CONTRACT_VERSION:
        raise ValueError("v3 recovery requires async answer_typed(recovered_trace=...) grounding")
    if (
        trace.status is not TraceStatus.SUCCESS
        or trace.run_id != prepared.run_id
        or trace.snapshot_id != prepared.snapshot_id
        or trace.query_id != prepared.query.query_id
        or trace.stage is not AppStage.ANSWER
        or trace.model_alias != prepared.request.model_alias
    ):
        raise ValueError("successful answer trace has incompatible runtime lineage")
    raw_answer = str((trace.parsed_output or {}).get("text") or "").strip()
    strict = prepared.metadata.get("answer_contract_version") == "strict-answer-contract-v2"
    try:
        structured = _strict_answer_output(
            raw_answer,
            finish_reason=(trace.parsed_output or {}).get("finish_reason"),
            allowed_aliases=list(prepared.metadata.get("evidence_aliases") or []),
        ) if strict else None
    except UnsupportedRuntimeAnswer as exc:
        assert exc.output is not None
        exc.fact = abstention_fact(
            prepared, trace, exc.output,
            attempt=int(trace.model_call_id.startswith("answer-format-retry-model-call")),
        )
        raise
    answer = structured.answer if structured is not None else raw_answer
    if not answer:
        raise EmptyRuntimeAnswer("successful model call has no recoverable answer")
    answer_id = stable_compiler_id(
        "method-answer",
        [
            prepared.run_id,
            prepared.query.query_id,
            prepared.method_id,
            prepared.answer_variant,
        ],
    )
    return FinalAnswerRecord(
        answer_id=answer_id,
        snapshot_id=prepared.snapshot_id,
        query_id=prepared.query.query_id,
        run_id=prepared.run_id,
        answer_variant=prepared.answer_variant,
        method=prepared.method_id,
        answer_model_alias=prepared.request.model_alias,
        actual_model_id=trace.actual_model_id,
        evidence_ids_in_prompt=prepared.evidence_ids_in_prompt,
        image_ids_in_prompt=prepared.image_ids_in_prompt,
        final_answer=answer,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        latency_ms=trace.latency_ms,
        model_call_id=trace.model_call_id,
        langfuse_trace_id=trace.langfuse_trace_id,
        metadata={
            "finish_reason": (trace.parsed_output or {}).get("finish_reason"),
            "source_lineage": prepared.source_lineage,
            "recovered_from_model_call": True,
            "provider_schema_conformant": structured is not None if strict else None,
            "structured_supported": structured.supported if structured else None,
            "structured_abstain_reason": None,
            "answer_contract_version": prepared.metadata.get("answer_contract_version"),
            "used_evidence_aliases": (
                structured.used_evidence_aliases if structured is not None else []
            ),
        },
    )


__all__ = [
    "EmptyRuntimeAnswer",
    "InvalidRuntimeAnswer",
    "SharedAnswerGenerationService",
    "UnsupportedRuntimeAnswer",
    "recover_runtime_answer",
    "strict_final_answer_failure_codes",
]
