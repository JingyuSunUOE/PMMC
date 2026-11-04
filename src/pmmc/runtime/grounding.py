"""Uniform, bounded, gold-free answer grounding for PMMC and all baseline methods."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.models.execution import ModelCallContext, ModelExecutionService
from pmmc.models.protocols import GenerationGateway, TokenCounter
from pmmc.models.provider_config import provider_output_budget
from pmmc.models.provider_token_budget import ProviderConfiguredTokenCounter
from pmmc.schemas import (
    AnswerGenerationInput,
    AppStage,
    GenerationRequest,
    ModelCallTrace,
    ModelContentPart,
    ModelMessage,
    ThinkingMode,
)
from pmmc.schemas.answer_output import (
    GROUNDING_POLICY_VERSION,
    AnswerContractV3,
    AnswerRoleCall,
    GroundingAssessment,
    GroundingFact,
    TypedAnswerContext,
)

GROUNDING_MODEL_ALIAS = "grounding-qwen3.5-9b"
GROUNDING_OUTPUT_TOKENS = 512
GROUNDING_CONTEXT_TOKENS = 32768
GROUNDING_BUDGET_VERSION = "step8-grounding-budget-v1"
_PROMPT_ROOT: ContextVar[Path | None] = ContextVar("step8_prompt_root", default=None)
_STEP8_PROMPTS = frozenset({"pmmc_answer_contract_v3.txt", "answer_grounding_v1.txt"})


@contextmanager
def step8_prompt_scope(project_root: Path) -> Iterator[None]:
    token = _PROMPT_ROOT.set(project_root.resolve())
    try:
        yield
    finally:
        _PROMPT_ROOT.reset(token)


def load_step8_prompt(filename: str) -> str:
    from pmmc.infrastructure.settings import get_settings

    if filename not in _STEP8_PROMPTS:
        raise ValueError("step8_prompt_name_not_allowed")
    root = (_PROMPT_ROOT.get() or get_settings().project_root).resolve()
    path = root / "prompts/runtime" / filename
    if not path.resolve().is_relative_to(root):
        raise ValueError("step8_prompt_outside_project_root")
    # Installed wheels live in site-packages, not next to the frozen config assets.
    return path.read_text(encoding="utf-8").strip()


class GroundingEngineeringError(RuntimeError):
    """Missing infrastructure or invalid verifier output, never a science rejection."""

    def __init__(self, code: str, *, calls: Sequence[AnswerRoleCall] = ()) -> None:
        self.code = code
        self.calls = tuple(calls)
        super().__init__(code)


class GroundingCallback(Protocol):
    async def assess(
        self,
        prepared: AnswerGenerationInput,
        output: AnswerContractV3,
        *,
        answer_model_call_id: str,
        job_id: str | None = None,
    ) -> GroundingFact: ...


def role_call(trace: ModelCallTrace, role: str) -> AnswerRoleCall:
    return AnswerRoleCall(
        role=role,
        model_call_id=trace.model_call_id,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        cost_usd=trace.cost_usd,
        latency_ms=trace.latency_ms,
    )


def strict_json_object(text: str) -> dict:
    def pairs(values: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def source_content_parts(
    context: TypedAnswerContext,
    aliases: Sequence[str] | None = None,
) -> list[ModelContentPart]:
    """Whitelist source fields; never copy source/query metadata into model messages."""
    bindings = {item.image_id: item.image_alias for item in context.image_bindings}
    allowed = None if aliases is None else set(aliases)
    parts = []
    shown_images: set[str] = set()
    for source in context.sources:
        if allowed is not None and source.evidence_alias not in allowed:
            continue
        image_descriptions = [
            {"image_alias": bindings[i.image_id], "caption": i.caption} for i in source.images
        ]
        parts.append(
            ModelContentPart(
                kind="text",
                text=json.dumps(
                    {
                        "evidence_alias": source.evidence_alias,
                        "source_text": source.text,
                        "images": image_descriptions,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        )
        for item in source.images:
            if item.image is not None and item.image_id not in shown_images:
                shown_images.add(item.image_id)
                parts.append(ModelContentPart(kind="text", text=f"Image {bindings[item.image_id]}"))
                parts.append(
                    ModelContentPart(
                        kind="image",
                        image=item.image.model_copy(
                            update={
                                "metadata": {},
                                "caption": None,
                            }
                        ),
                    )
                )
    return parts


def grounding_evidence_hash(context: TypedAnswerContext, aliases: Sequence[str]) -> str:
    return canonical_content_hash(
        [s.model_dump(mode="json") for s in context.sources if s.evidence_alias in set(aliases)]
    )


def validate_grounding_request_budget(
    request: GenerationRequest,
    counter: TokenCounter,
) -> GenerationRequest:
    """Count the final verifier input without dropping any proposed answer or evidence."""
    if (
        request.max_output_tokens != GROUNDING_OUTPUT_TOKENS
        or request.provider_max_output_tokens != GROUNDING_OUTPUT_TOKENS
        or request.thinking_mode is not ThinkingMode.DISABLED
        or request.metadata.get("model_role") not in {"answer_grounding", "grounding_format_retry"}
    ):
        raise GroundingEngineeringError("grounding_budget_contract_invalid")
    try:
        alias = counter.alias
        if isinstance(counter, ProviderConfiguredTokenCounter):
            entry = counter.entry
            alias = entry.alias
            config = entry.provider_config
            if (
                entry.context_window is None
                or entry.context_window < GROUNDING_CONTEXT_TOKENS
                or config is None
                or config.kind != "qwen_vllm"
            ):
                raise GroundingEngineeringError("grounding_budget_contract_invalid")
            provider_output_budget(request, config.max_output_tokens, config=config)
        if alias != request.model_alias or counter.estimated is not False or not counter.revision:
            raise GroundingEngineeringError("grounding_counter_invalid")
        input_tokens = counter.count_request(request)
        if type(input_tokens) is not int or input_tokens < 0:
            raise GroundingEngineeringError("grounding_counter_invalid")
        budget = {
            "version": GROUNDING_BUDGET_VERSION,
            "context_tokens": GROUNDING_CONTEXT_TOKENS,
            "provider_output_tokens": GROUNDING_OUTPUT_TOKENS,
            "input_tokens": input_tokens,
            "counter_alias": counter.alias,
            "counter_revision": counter.revision,
            "estimated": False,
        }
    except GroundingEngineeringError:
        raise
    except Exception:
        raise GroundingEngineeringError("grounding_budget_count_failed") from None
    if input_tokens + GROUNDING_OUTPUT_TOKENS > GROUNDING_CONTEXT_TOKENS:
        raise GroundingEngineeringError("grounding_input_budget_exceeded")
    return request.model_copy(update={
        "request_id": stable_compiler_id(
            "grounding-budgeted-request", [request.request_id, budget]
        ),
        "metadata": {**request.metadata, "grounding_budget": budget},
    })


def build_grounding_request(
    prepared: AnswerGenerationInput,
    output: AnswerContractV3,
    *,
    answer_model_call_id: str,
    model_alias: str = GROUNDING_MODEL_ALIAS,
) -> GenerationRequest:
    from pmmc.runtime.answering import validate_typed_answer_input
    from pmmc.runtime.relation_grounding import (
        grounding_relation_requirement,
        grounding_scope_aliases,
    )

    if not output.supported:
        raise ValueError("legal abstention does not require grounding")
    context = validate_typed_answer_input(prepared)
    relation_requirement = grounding_relation_requirement(prepared, output, context)
    supplied_aliases = grounding_scope_aliases(prepared, output, context)
    if not set(output.used_evidence_aliases) <= {s.evidence_alias for s in context.sources}:
        raise GroundingEngineeringError("grounding_unknown_evidence_alias")
    parts = [
        ModelContentPart(
            kind="text",
            text=json.dumps(
                {
                    "query": prepared.query.raw_query,
                    "proposed_answer": output.model_dump(mode="json"),
                    **({"required_evidence_relation": relation_requirement}
                       if relation_requirement is not None else {}),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    ]
    for image in prepared.query.query_image_refs:
        parts.append(
            ModelContentPart(
                kind="image",
                image=image.model_copy(
                    update={
                        "metadata": {},
                        "caption": None,
                    }
                ),
            )
        )
    parts.extend(source_content_parts(context, supplied_aliases))
    prompt = load_step8_prompt("answer_grounding_v1.txt")
    identity = [
        answer_model_call_id,
        model_alias,
        GROUNDING_POLICY_VERSION,
        [GROUNDING_BUDGET_VERSION, GROUNDING_CONTEXT_TOKENS, GROUNDING_OUTPUT_TOKENS],
        canonical_content_hash(output.model_dump(mode="json")),
        grounding_evidence_hash(context, supplied_aliases),
        canonical_content_hash(prompt),
        *([canonical_content_hash(relation_requirement)]
          if relation_requirement is not None else []),
        canonical_content_hash(
            [
                prepared.query.raw_query,
                [
                    i.model_dump(mode="json", exclude={"metadata", "caption"})
                    for i in prepared.query.query_image_refs
                ],
            ]
        ),
    ]
    return GenerationRequest(
        request_id=stable_compiler_id("answer-grounding-request", identity),
        model_alias=model_alias,
        messages=[
            ModelMessage(role="system", parts=[ModelContentPart(kind="text", text=prompt)]),
            ModelMessage(role="user", parts=parts),
        ],
        max_output_tokens=GROUNDING_OUTPUT_TOKENS,
        provider_max_output_tokens=GROUNDING_OUTPUT_TOKENS,
        thinking_mode=ThinkingMode.DISABLED,
        temperature=0,
        top_p=1,
        seed=0,
        timeout_seconds=prepared.request.timeout_seconds,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "source_only_answer_grounding_v1",
                "strict": True,
                "schema": GroundingAssessment.model_json_schema(),
            },
        },
        metadata={
            "model_role": "answer_grounding",
            "grounding_policy": GROUNDING_POLICY_VERSION,
            "assessment_source": "model_assessed",
            "prompt_hash": canonical_content_hash(prompt),
            **({"runtime_relation_contract_hash": relation_requirement["contract_hash"]}
               if relation_requirement is not None else {}),
        },
    )


class AuditedGroundingService:
    def __init__(
        self,
        *,
        execution: ModelExecutionService,
        gateway: GenerationGateway,
        model_alias: str = GROUNDING_MODEL_ALIAS,
        counter: TokenCounter | None = None,
        counter_factory: Callable[[], TokenCounter] | None = None,
    ) -> None:
        if counter is not None and counter_factory is not None:
            raise ValueError("provide only one grounding counter or counter factory")
        self.execution = execution
        self.gateway = gateway
        self.model_alias = model_alias
        self._counter = counter
        self._counter_factory = counter_factory

    def _budget_request(self, request: GenerationRequest) -> GenerationRequest:
        if self._counter is None:
            if self._counter_factory is None:
                raise GroundingEngineeringError("grounding_counter_unavailable")
            try:
                self._counter = self._counter_factory()
            except Exception:
                raise GroundingEngineeringError("grounding_counter_unavailable") from None
            if self._counter is None:
                raise GroundingEngineeringError("grounding_counter_unavailable")
        return validate_grounding_request_budget(request, self._counter)

    async def assess(
        self,
        prepared: AnswerGenerationInput,
        output: AnswerContractV3,
        *,
        answer_model_call_id: str,
        job_id: str | None = None,
    ) -> GroundingFact:
        request = build_grounding_request(
            prepared,
            output,
            answer_model_call_id=answer_model_call_id,
            model_alias=self.model_alias,
        )
        request = self._budget_request(request)
        calls: list[AnswerRoleCall] = []
        for attempt in range(2):
            role = "answer_grounding" if attempt == 0 else "grounding_format_retry"
            current = request
            if attempt:
                current = request.model_copy(
                    update={
                        "request_id": stable_compiler_id(
                            "grounding-format-retry", request.request_id
                        ),
                        "messages": [
                            *request.messages,
                            ModelMessage(
                                role="user",
                                parts=[
                                    ModelContentPart(
                                        kind="text",
                                        text=(
                                            "Return only the six required schema fields. "
                                            "All checks must use pass/fail/not_applicable. "
                                            "supported must agree "
                                            "with the checks. Do not change evidence or invent "
                                            "support to fix formatting."
                                        ),
                                    )
                                ],
                            ),
                        ],
                        "metadata": {
                            **request.metadata,
                            "model_role": role,
                            "format_retry_ordinal": 1,
                        },
                    }
                )
                try:
                    current = self._budget_request(current)
                except GroundingEngineeringError as exc:
                    raise GroundingEngineeringError(exc.code, calls=calls) from None
            call_id = stable_compiler_id(
                "answer-grounding-model-call", [current.request_id, attempt]
            )
            # Transport errors propagate; they are never a negative scientific assessment.
            audited = await self.execution.generate_audited(
                self.gateway,
                current,
                ModelCallContext(
                    model_call_id=call_id,
                    run_id=prepared.run_id,
                    snapshot_id=prepared.snapshot_id,
                    query_id=prepared.query.query_id,
                    conversation_id=prepared.query.conversation_id,
                    job_id=job_id,
                    stage=AppStage.ANSWER,
                    parent_model_call_id=answer_model_call_id,
                ),
            )
            calls.append(role_call(audited.trace, role))
            try:
                if audited.result.finish_reason not in {"stop", "eos", "end_turn"}:
                    raise ValueError("grounding finish reason is not terminal")
                if audited.result.usage.output_tokens > GROUNDING_OUTPUT_TOKENS:
                    raise ValueError("grounding output budget exceeded")
                assessment = GroundingAssessment.model_validate(
                    strict_json_object(audited.result.text)
                )
            except (ValueError, TypeError, ValidationError):
                if not attempt:
                    continue
                raise GroundingEngineeringError("grounding_format_invalid", calls=calls) from None
            context = TypedAnswerContext.model_validate(prepared.metadata["typed_answer_context"])
            from pmmc.runtime.relation_grounding import (
                grounding_relation_requirement,
                grounding_scope_aliases,
            )

            relation = grounding_relation_requirement(prepared, output, context)
            if (relation is not None and assessment.supported
                    and relation["supplied_independent_group_count"]
                    < relation["minimum_evidence_groups"]):
                # Preserve the provider response, but never publish a pass that
                # contradicts the scoped sources supplied to the grounder.
                raise GroundingEngineeringError(
                    "grounding_relation_group_contradiction", calls=calls
                )
            return GroundingFact(
                grounding_id=stable_compiler_id("answer-grounding", request.request_id),
                run_id=prepared.run_id,
                query_id=prepared.query.query_id,
                answer_model_call_id=answer_model_call_id,
                model_alias=self.model_alias,
                actual_model_id=audited.result.actual_model_id,
                assessment=assessment,
                used_evidence_aliases=output.used_evidence_aliases,
                evidence_content_hash=grounding_evidence_hash(
                    context, grounding_scope_aliases(prepared, output, context)
                ),
                ungated_output_hash=canonical_content_hash(output.model_dump(mode="json")),
                calls=calls,
            )
        raise AssertionError("bounded grounding did not terminate")


def configured_grounding_service(*, settings, execution, image_url_resolver=None):
    """Use the existing gateway factory; do not start or duplicate a model service."""
    from pmmc.models.factory import create_generation_gateway, create_token_counter

    registry = execution.registry
    try:
        registry.get(GROUNDING_MODEL_ALIAS)
    except KeyError:
        return None
    return AuditedGroundingService(
        execution=execution,
        gateway=create_generation_gateway(
            settings,
            registry,
            GROUNDING_MODEL_ALIAS,
            image_url_resolver=image_url_resolver,
        ),
        counter_factory=lambda: create_token_counter(settings, registry, GROUNDING_MODEL_ALIAS),
    )
