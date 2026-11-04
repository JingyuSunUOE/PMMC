"""Evidence-scoped Doubter v1 prompt, renderer, and strict parser."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace

from pydantic import ValidationError

from pmmc.compiler.doubter import doubter_dimension_applicability
from pmmc.compiler.doubter_config import DoubterConfig
from pmmc.models.protocols import TokenCounter
from pmmc.models.tokenization import HeuristicTokenCounter
from pmmc.schemas import (
    DoubterProviderEnvelope,
    DoubterProviderEnvelopeV11,
    GenerationRequest,
    ImageRef,
    MemoryProgram,
    Modality,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    PlannerRequirementContract,
    ProgramDryRunAssessment,
    QBankRouteClass,
    QuestionCandidate,
    SpeakerRole,
)

DOUBTER_PROMPT = """You are Doubter v1 for a prospective multimodal memory compiler.
Audit exactly one candidate and its real dry-run. You receive the future question, expected
answer, source-support evidence, a non-oracle requirement contract, the current declarative
program, retrieved evidence, and an execution assessment.

Return exactly ten boolean dimensions: candidate_supported, answer_entailed,
path_sufficient, modality_sufficient, raw_image_sufficient, context_sufficient,
multi_hop_sufficient, temporal_consistent, non_redundant, and refusal_appropriate. Return
only applicable fixed issue_codes. Do not choose pass/revise/reject and do not propose free
text changes. Never modify or regenerate the question, answer, source evidence, or program.
For a control/refusal candidate, set non-applicable path, modality, raw-image, context,
multi-hop, temporal, redundancy, and answer-entailment dimensions to true; decide it using
candidate_supported and refusal_appropriate.
Never output canonical IDs, aliases from outside this request, paths, URLs, SQL, Python,
shell, benchmark query/gold, chain-of-thought, or a rationale. Return only the strict JSON
object."""

DOUBTER_PROMPT_V11 = """You are Doubter v1.1 for a prospective multimodal memory compiler.
Audit exactly one candidate and its real dry-run. Assess only the evidence shown in this
request. Return exactly ten dimensions using pass, fail, or not_applicable. Do not output
issue codes, an action, directives, a rationale, or replacement content; the service derives
all decisions deterministically.

Return exactly this outer structure: {"result":{"candidate_alias":"Q01","dimensions":{...}}}.
The guided response schema fixes which dimensions are applicable for this request. Applicable
dimensions can only be pass or fail; all other dimensions must be not_applicable.

For an answerable candidate, candidate_supported, answer_entailed, path_sufficient, and
modality_sufficient are applicable. Mark raw-image, context, multi-hop, and temporal
dimensions not_applicable unless the requirement contract requests them. Mark
refusal_appropriate not_applicable. For a control/refusal candidate, only
candidate_supported and refusal_appropriate are applicable. Never use not_applicable to
hide an applicable failure.

Never output canonical IDs, paths, URLs, SQL, Python, shell, benchmark query/gold,
chain-of-thought, or free-form explanation. Return only the strict JSON object."""

DOUBTER_PROMPT_V12 = """You are Doubter v1.2 for a prospective multimodal memory compiler.
Audit exactly one candidate and one real dry-run. SOURCE_SUPPORT contains the immutable
evidence supporting the Questioner's answer claim. RETRIEVED_PATH contains only evidence
actually returned by the current program. Judge candidate_supported and answer_entailed only
from SOURCE_SUPPORT. Judge path_sufficient, modality, raw-image, context, multi-hop, temporal,
and redundancy only from RETRIEVED_PATH and the execution assessment. Never transfer evidence
between these roles.

Return exactly ten dimensions using pass, fail, or not_applicable. Do not output issue codes,
actions, directives, explanations, IDs, paths, URLs, benchmark query/gold, or reasoning. The
guided schema fixes applicability and the server derives every action and typed patch. Return
only the strict JSON object."""

DOUBTER_PROMPT_V13 = (
    DOUBTER_PROMPT_V12.replace("v1.2", "v1.3")
    + """
SOURCE_SUPPORT text contains only the cited exact span, with its actual speaker and claim.
Judge entailment of that span, not an uncited neighboring sentence. An assistant's suggestion
is not a user's action or preference. An explicitly stated future plan supports a plan, not
an accomplished event. A why answer must give a cause; a which answer must identify its entity;
a count answer must identify the counted set. Group hits and execution success do not prove
entailment. Caption text is never proof of raw-image inspection. Raw-image path support needs
an inspected image in RETRIEVED_PATH. Empty or incomplete paths fail even if SOURCE_SUPPORT
fully supports the answer. Never use source-only images to repair the retrieved path.
"""
)


DOUBTER_EVIDENCE_PROJECTION_VERSION = "doubter-evidence-projection-v2"
DOUBTER_MAX_REQUEST_IMAGES = 8
DOUBTER_MAX_TEXT_CHARS_PER_EVIDENCE = 1024
DOUBTER_MAX_TOTAL_EVIDENCE_TEXT_CHARS = 16384


@dataclass(frozen=True)
class DoubterEvidenceView:
    alias: str
    evidence_type: str
    text: str | None = None
    image: ImageRef | None = None
    source_support: bool = False
    image_attachment_alias: str | None = None
    speaker: str | None = None
    claim_text: str | None = None
    span_exists: bool | None = None
    raw_image_inspected: bool = False
    claim_subject: str | None = None
    claim_assertion: str | None = None
    support_complete: bool = True


@dataclass(frozen=True)
class DoubterEvidenceProjection:
    views: tuple[DoubterEvidenceView, ...]
    evidence_count: int
    input_image_count: int
    input_unique_image_count: int
    duplicate_image_reference_count: int
    attached_image_count: int
    omitted_image_count: int
    input_text_chars: int
    rendered_text_chars: int
    truncated_text_item_count: int

    def safe_metadata(self) -> dict[str, object]:
        return {
            "evidence_projection_version": DOUBTER_EVIDENCE_PROJECTION_VERSION,
            "evidence_count": self.evidence_count,
            "input_image_count": self.input_image_count,
            "input_unique_image_count": self.input_unique_image_count,
            "duplicate_image_reference_count": self.duplicate_image_reference_count,
            "attached_image_count": self.attached_image_count,
            "omitted_image_count": self.omitted_image_count,
            "input_text_chars": self.input_text_chars,
            "rendered_text_chars": self.rendered_text_chars,
            "truncated_text_item_count": self.truncated_text_item_count,
        }


def project_doubter_evidence_views(
    evidence_views: Sequence[DoubterEvidenceView],
    *,
    max_request_images: int = DOUBTER_MAX_REQUEST_IMAGES,
) -> DoubterEvidenceProjection:
    """Build a bounded, source-first model projection without changing stored evidence."""

    if max_request_images < 0 or max_request_images > DOUBTER_MAX_REQUEST_IMAGES:
        raise ValueError("Doubter request image bound is outside the frozen range")

    priority = sorted(
        range(len(evidence_views)),
        key=lambda index: (not evidence_views[index].source_support, index),
    )
    unique_image_priority: list[tuple[str, int]] = []
    seen_image_ids: set[str] = set()
    for index in priority:
        image = evidence_views[index].image
        if image is None or image.image_id in seen_image_ids:
            continue
        seen_image_ids.add(image.image_id)
        unique_image_priority.append((image.image_id, index))
    selected_images = unique_image_priority[:max_request_images]
    attached_image_indexes = {index for _image_id, index in selected_images}
    attachment_aliases = {
        image_id: evidence_views[index].alias for image_id, index in selected_images
    }

    remaining_text_chars = DOUBTER_MAX_TOTAL_EVIDENCE_TEXT_CHARS
    projected_text: dict[int, str | None] = {}
    truncated_text_item_count = 0
    for index in priority:
        text = evidence_views[index].text
        if text is None:
            projected_text[index] = None
            continue
        allowed = min(
            len(text),
            DOUBTER_MAX_TEXT_CHARS_PER_EVIDENCE,
            remaining_text_chars,
        )
        projected_text[index] = text[:allowed] or None
        remaining_text_chars -= allowed
        if allowed < len(text):
            truncated_text_item_count += 1

    projected = tuple(
        DoubterEvidenceView(
            alias=item.alias,
            evidence_type=item.evidence_type,
            text=projected_text[index],
            image=(item.image if index in attached_image_indexes else None),
            source_support=item.source_support,
            speaker=item.speaker,
            claim_text=item.claim_text,
            span_exists=item.span_exists,
            raw_image_inspected=item.raw_image_inspected,
            claim_subject=item.claim_subject,
            claim_assertion=item.claim_assertion,
            support_complete=item.support_complete,
            image_attachment_alias=(
                attachment_aliases.get(item.image.image_id) if item.image is not None else None
            ),
        )
        for index, item in enumerate(evidence_views)
    )
    input_image_count = sum(item.image is not None for item in evidence_views)
    attached_image_count = len(attached_image_indexes)
    input_unique_image_count = len(unique_image_priority)
    return DoubterEvidenceProjection(
        views=projected,
        evidence_count=len(evidence_views),
        input_image_count=input_image_count,
        input_unique_image_count=input_unique_image_count,
        duplicate_image_reference_count=input_image_count - input_unique_image_count,
        attached_image_count=attached_image_count,
        omitted_image_count=input_unique_image_count - attached_image_count,
        input_text_chars=sum(len(item.text or "") for item in evidence_views),
        rendered_text_chars=sum(len(item.text or "") for item in projected),
        truncated_text_item_count=truncated_text_item_count,
    )


def doubter_prompt_sha256(config: DoubterConfig | None = None) -> str:
    prompt = (
        DOUBTER_PROMPT_V13
        if config is not None and config.is_v13
        else DOUBTER_PROMPT_V12
        if config is not None and config.is_v12
        else DOUBTER_PROMPT_V11
        if config is not None and config.is_v11
        else DOUBTER_PROMPT
    )
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def doubter_response_schema(
    config: DoubterConfig | None = None,
    *,
    route_class: QBankRouteClass | None = None,
    requirement: PlannerRequirementContract | None = None,
) -> dict[str, object]:
    contract = (
        DoubterProviderEnvelopeV11
        if config is not None and config.is_v11
        else DoubterProviderEnvelope
    )
    schema = contract.model_json_schema()
    if (route_class is None) != (requirement is None):
        raise ValueError("Doubter response schema requires route and requirement together")
    if config is None or not config.is_v11 or route_class is None or requirement is None:
        return schema
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise RuntimeError("Doubter v1.1 response schema has no definitions")
    dimensions = definitions.get("DoubterProviderDimensionsV11")
    if not isinstance(dimensions, dict):
        raise RuntimeError("Doubter v1.1 response schema has no dimensions contract")
    properties = dimensions.get("properties")
    if not isinstance(properties, dict):
        raise RuntimeError("Doubter v1.1 dimensions schema has no properties")
    applicability = doubter_dimension_applicability(
        route_class=route_class,
        requirement=requirement,
    )
    for name, is_applicable in applicability.items():
        properties[name] = (
            {"enum": ["pass", "fail"], "type": "string"}
            if is_applicable
            else {"enum": ["not_applicable"], "type": "string"}
        )
    return schema


def doubter_response_schema_sha256(
    config: DoubterConfig | None = None,
    *,
    route_class: QBankRouteClass | None = None,
    requirement: PlannerRequirementContract | None = None,
) -> str:
    payload = json.dumps(
        doubter_response_schema(
            config,
            route_class=route_class,
            requirement=requirement,
        ),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _program_projection(program: MemoryProgram) -> dict[str, object]:
    return {
        "round_id": program.round_id,
        "steps": [
            {
                "step_id": step.step_id,
                "op_name": step.op_name,
                "args": {
                    key: value
                    for key, value in step.args.items()
                    if key
                    in {
                        "top_k",
                        "vector_role",
                        "source_step_ids",
                        "radius",
                        "required_modalities",
                        "minimum_evidence_groups",
                        "condition",
                        "term_source",
                        "term_budget",
                    }
                },
                "expected_output_type": step.expected_output_type,
            }
            for step in program.operator_sequence
        ],
    }


def render_doubter_request(
    *,
    request_id: str,
    config: DoubterConfig,
    candidate: QuestionCandidate,
    requirement: PlannerRequirementContract,
    program: MemoryProgram,
    assessment: ProgramDryRunAssessment,
    evidence_views: Sequence[DoubterEvidenceView],
    token_counter: TokenCounter | None = None,
    timeout_seconds: float | None = None,
) -> GenerationRequest:
    if requirement.qbank_id != candidate.qbank_id or program.qbank_id != candidate.qbank_id:
        raise ValueError("Doubter request candidate scope is inconsistent")
    if assessment.plan_id != program.plan_id:
        raise ValueError("Doubter request dry-run belongs to another program")
    if config.is_v13 and program.metadata.get("memory_raw_image_policy") == "forbid":
        evidence_views = [
            replace(
                item,
                image=None,
                image_attachment_alias=None,
                raw_image_inspected=False,
                support_complete=item.support_complete and (item.image is None or bool(item.text)),
            )
            for item in evidence_views
        ]
    if len(evidence_views) > config.max_evidence_items:
        raise ValueError("Doubter request exceeds the frozen evidence limit")
    aliases = [item.alias for item in evidence_views]
    if len(aliases) != len(set(aliases)):
        raise ValueError("Doubter evidence aliases must be unique")
    counter = token_counter or HeuristicTokenCounter(
        alias=config.doubter_model_alias,
        revision=DOUBTER_EVIDENCE_PROJECTION_VERSION,
    )
    unique_image_count = len(
        {item.image.image_id for item in evidence_views if item.image is not None}
    )

    def build_request(projection: DoubterEvidenceProjection) -> GenerationRequest:
        rendered_evidence = projection.views
        payload = {
            "candidate_alias": "Q01",
            "question": candidate.canonical_question,
            "expected_answer": candidate.expected_answer,
            "route_class": candidate.route_class.value,
            "requirement_contract": {
                "required_modalities": [item.value for item in requirement.required_modalities],
                "minimum_evidence_groups": requirement.minimum_evidence_groups,
                "required_capabilities": [item.value for item in requirement.required_capabilities],
                "fallback_required": requirement.fallback_required,
                "refusal_only": requirement.refusal_only,
            },
            "program": _program_projection(program),
            "execution_assessment": {
                "technical_execution_success": assessment.execution_success,
                "nonempty_evidence": assessment.nonempty_evidence,
                "required_modality_sufficient": assessment.required_modality_sufficient,
                "evidence_group_complete": assessment.expected_evidence_complete,
                "raw_image_reached": assessment.raw_image_reached,
                "context_operator_used": assessment.context_operator_used,
                "temporal_operator_used": assessment.temporal_operator_used,
                "context_evidence_contributed": (assessment.context_evidence_contributed),
                "temporal_evidence_contributed": (assessment.temporal_evidence_contributed),
                "pre_control_sufficient": assessment.pre_control_sufficient,
                "fallback_triggered": assessment.fallback_triggered,
                "abstained": assessment.abstain_triggered,
                "failure_taxonomy": assessment.failure_code,
            },
            **(
                {
                    "source_support_evidence": [
                        {
                            "alias": item.alias,
                            "evidence_type": item.evidence_type,
                            "text": item.text,
                            "image_attachment_alias": item.image_attachment_alias,
                            "image_is_inline": item.image is not None,
                            **(
                                {
                                    "speaker": item.speaker,
                                    "claim_text": item.claim_text,
                                    "span_exists": item.span_exists,
                                    "subject": item.claim_subject,
                                    "assertion": item.claim_assertion,
                                }
                                if config.is_v13
                                else {}
                            ),
                        }
                        for item in rendered_evidence
                        if item.source_support
                    ],
                    "retrieved_path_evidence": [
                        {
                            "alias": item.alias,
                            "evidence_type": item.evidence_type,
                            "text": item.text,
                            "image_attachment_alias": item.image_attachment_alias,
                            "image_is_inline": item.image is not None,
                            **(
                                {"raw_image_inspected": item.raw_image_inspected}
                                if config.is_v13
                                else {}
                            ),
                        }
                        for item in rendered_evidence
                        if not item.source_support
                    ],
                }
                if config.is_v12 or config.is_v13
                else {
                    "evidence": [
                        {
                            "alias": item.alias,
                            "evidence_type": item.evidence_type,
                            "text": item.text,
                            "image_attachment_alias": item.image_attachment_alias,
                            "image_is_inline": item.image is not None,
                            "source_support": item.source_support,
                        }
                        for item in rendered_evidence
                    ]
                }
            ),
        }
        parts = [
            ModelContentPart(
                kind=ModelContentKind.TEXT,
                text=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
            )
        ]
        for item in rendered_evidence:
            if item.image is not None:
                parts.extend(
                    [
                        ModelContentPart(
                            kind=ModelContentKind.TEXT,
                            text=f"[EVIDENCE_IMAGE alias={item.alias}]",
                        ),
                        ModelContentPart(kind=ModelContentKind.IMAGE, image=item.image),
                    ]
                )
        response_schema = doubter_response_schema(
            config,
            route_class=candidate.route_class,
            requirement=requirement,
        )
        return GenerationRequest(
            request_id=request_id,
            model_alias=config.doubter_model_alias,
            messages=[
                ModelMessage(
                    role=SpeakerRole.SYSTEM,
                    parts=[
                        ModelContentPart(
                            kind=ModelContentKind.TEXT,
                            text=(
                                DOUBTER_PROMPT_V13
                                if config.is_v13
                                else DOUBTER_PROMPT_V12
                                if config.is_v12
                                else DOUBTER_PROMPT_V11
                                if config.is_v11
                                else DOUBTER_PROMPT
                            ),
                        )
                    ],
                ),
                ModelMessage(role=SpeakerRole.USER, parts=parts),
            ],
            max_output_tokens=config.reserved_output_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=0,
            timeout_seconds=timeout_seconds,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": (
                        "pmmc_doubter_assessment_v1_3"
                        if config.is_v13
                        else "pmmc_doubter_assessment_v1_1"
                        if config.is_v11
                        else "pmmc_doubter_assessment_v1"
                    ),
                    "strict": True,
                    "schema": response_schema,
                },
            },
            metadata={
                "task": "pmmc_doubter",
                "prompt_version": config.prompt_version,
                "renderer_version": config.renderer_version,
                "response_schema_hash": hashlib.sha256(
                    json.dumps(
                        response_schema,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "single_candidate_scope": True,
                "program_round": program.round_id,
                **(
                    {
                        "memory_raw_image_policy": program.metadata.get(
                            "memory_raw_image_policy", "allow"
                        )
                    }
                    if config.is_v13
                    else {}
                ),
                "requirement_contract_hash": requirement.content_hash,
                **projection.safe_metadata(),
                **(
                    {
                        "evidence_projection_version": "doubter-evidence-projection-v3",
                        "source_support_complete": all(
                            item.support_complete
                            and (not item.text or item.text == rendered.text)
                            and (item.image is None or rendered.image_attachment_alias is not None)
                            for item, rendered in zip(
                                evidence_views, rendered_evidence, strict=True
                            )
                            if item.source_support
                        ),
                        "rendered_raw_path_image_count": sum(
                            not item.source_support
                            and item.raw_image_inspected
                            and item.image_attachment_alias is not None
                            for item in rendered_evidence
                        ),
                    }
                    if config.is_v13
                    else {}
                ),
                "max_request_images": DOUBTER_MAX_REQUEST_IMAGES,
                "max_text_chars_per_evidence": DOUBTER_MAX_TEXT_CHARS_PER_EVIDENCE,
                "max_total_evidence_text_chars": DOUBTER_MAX_TOTAL_EVIDENCE_TEXT_CHARS,
            },
        )

    visual_required = any(
        value in {Modality.IMAGE, Modality.RAW_IMAGE, Modality.MULTIMODAL}
        for value in requirement.required_modalities
    ) or any(item.source_support and item.image is not None for item in evidence_views)
    minimum_images = 1 if unique_image_count and visual_required else 0
    lower = minimum_images
    upper = min(DOUBTER_MAX_REQUEST_IMAGES, unique_image_count)
    selected: tuple[GenerationRequest, DoubterEvidenceProjection, int] | None = None
    while lower <= upper:
        image_bound = (lower + upper) // 2
        projection = project_doubter_evidence_views(
            evidence_views,
            max_request_images=image_bound,
        )
        request = build_request(projection)
        input_tokens = int(counter.count_request(request))
        if input_tokens <= config.max_input_tokens:
            selected = (request, projection, input_tokens)
            lower = image_bound + 1
        else:
            upper = image_bound - 1
    if selected is None:
        raise ValueError("Doubter request exceeds the frozen input-token estimate")
    request, projection, input_tokens = selected
    return request.model_copy(
        update={
            "metadata": {
                **request.metadata,
                "attached_image_count": projection.attached_image_count,
                "omitted_image_count": projection.omitted_image_count,
                "input_tokens": input_tokens,
                "estimated_input_tokens": input_tokens,
                "max_input_tokens": config.max_input_tokens,
                "token_count_estimated": counter.estimated,
                "token_counter_alias": counter.alias,
                "token_counter_revision": counter.revision,
                "token_budget_image_reduction_count": (
                    min(DOUBTER_MAX_REQUEST_IMAGES, unique_image_count)
                    - projection.attached_image_count
                ),
            }
        }
    )


def doubter_output_parser(  # type: ignore[no-untyped-def]
    *, config: DoubterConfig | None = None, expected_alias: str = "Q01"
):
    def parse(result):  # type: ignore[no-untyped-def]
        from pmmc.models.execution import ParsedModelOutput

        diagnostic_codes: list[str] = []
        validation_errors: list[dict[str, str]] = []
        parsed: DoubterProviderEnvelope | DoubterProviderEnvelopeV11 | None = None
        try:
            payload = json.loads(result.text)
            contract = (
                DoubterProviderEnvelopeV11
                if config is not None and config.is_v11
                else DoubterProviderEnvelope
            )
            parsed = contract.model_validate(payload)
            if parsed.result.candidate_alias != expected_alias:
                raise ValueError("Doubter candidate scope mismatch")
        except ValidationError as exc:
            parsed = None
            diagnostic_codes.append(type(exc).__name__)
            validation_errors.extend(
                {
                    "location": ".".join(str(part) for part in item.get("loc", ())) or "$",
                    "error_type": str(item.get("type", "validation_error")),
                }
                for item in exc.errors(include_input=False, include_url=False)[:10]
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            parsed = None
            diagnostic_codes.append(type(exc).__name__)
        finish_reason = getattr(result, "finish_reason", None)
        return ParsedModelOutput(
            value={
                "result": (parsed.result.model_dump(mode="json") if parsed is not None else None),
                "provider_schema_conformant": parsed is not None,
                "diagnostic_codes": sorted(diagnostic_codes),
                "validation_errors": validation_errors,
                "provider_finish_reason": (
                    finish_reason
                    if finish_reason in {"stop", "length", "content_filter", "tool_calls"}
                    else "unknown"
                ),
                "provider_output_char_count": len(result.text),
            },
            repair_attempt_count=0,
            initial_error=("doubter_structured_output_invalid" if parsed is None else None),
        )

    return parse
