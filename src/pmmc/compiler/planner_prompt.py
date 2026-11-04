"""Strict Planner v1 prompt, safe renderer, and recoverable output parser."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations

from pydantic import ValidationError

from pmmc.compiler.planner_config import PlannerConfig
from pmmc.schemas import (
    DoubterDirectiveCode,
    GenerationRequest,
    ImageRef,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    OperatorEvidence,
    PlannerCapability,
    PlannerGenerationEnvelope,
    PlannerGenerationEnvelopeV11,
    PlannerProgramDraft,
    PlannerRequirementContract,
    PlannerStrategyDraftV12,
    PlannerStrategyEnvelopeV12,
    QuestionCandidate,
    SpeakerRole,
)

_REVISION_OPERATORS = frozenset(
    {
        "retrieve_text_by_keyword",
        "retrieve_text_by_embedding",
        "retrieve_image_by_visual_embedding",
        "retrieve_image_text_pair",
        "retrieve_context_around_image",
        "expand_temporal_neighbors",
        "inspect_raw_image",
        "verify_evidence_sufficiency",
        "fallback_to_multimodal_rag",
        "abstain_if_no_evidence",
    }
)
_REVISION_OUTPUT_TYPES = frozenset(
    {"memory_ids", "image_ids", "image_text_pairs", "verification", "control"}
)

PLANNER_PROMPT_VERSION = "pmmc-planner-v1"
PLANNER_RENDERER_VERSION = "planner-safe-alias-renderer-v1"
PLANNER_PARSER_VERSION = "planner-json-parser-v1"
PLANNER_V11_PROMPT_VERSION = "pmmc-planner-v1.1"
PLANNER_V11_RENDERER_VERSION = "planner-non-oracle-renderer-v1.1"
PLANNER_V11_PARSER_VERSION = "planner-json-parser-v1.1"
PLANNER_V12_PROMPT_VERSION = "pmmc-planner-v1.2"
PLANNER_V12_RENDERER_VERSION = "planner-strategy-renderer-v1.2"
PLANNER_V12_PARSER_VERSION = "planner-strategy-json-parser-v1.2"

_SAFE_DIAGNOSTIC_FIELDS = frozenset(
    {
        "candidate_alias",
        "steps",
        "op_name",
        "top_k",
        "terms",
        "vector_role",
        "source_step_ids",
        "radius",
        "required_modalities",
        "minimum_evidence_groups",
        "condition",
        "reason_code",
        "strategy_code",
        "rationale_summary",
        "confidence",
        "program",
        "strategy",
        "retrieval_channels",
        "expansion_capabilities",
        "top_k_tier",
    }
)

PLANNER_PROMPT = """You are Planner v1 for a prospective multimodal memory compiler.
For every candidate alias in the request, emit exactly one declarative Memory Program. Use
only the ten allowed operators and the exact argument contracts below. Steps are zero-based,
linear, and source_step_ids may contain only earlier step numbers.

Allowed operators: retrieve_text_by_keyword, retrieve_text_by_embedding,
retrieve_image_by_visual_embedding, retrieve_image_text_pair,
retrieve_context_around_image, expand_temporal_neighbors, inspect_raw_image,
verify_evidence_sufficiency, fallback_to_multimodal_rag, abstain_if_no_evidence.

Argument contracts:
- retrieve_text_by_keyword: top_k (1..50), terms (one to eight ordinary query terms).
- retrieve_text_by_embedding: top_k (1..50), vector_role=question_text_query.
- retrieve_image_by_visual_embedding: top_k (1..50), vector_role=question_joint_query.
- retrieve_image_text_pair: source_step_ids from an earlier image-producing step.
- retrieve_context_around_image: source_step_ids from an earlier image-producing step,
  radius (0..2).
- inspect_raw_image: source_step_ids from an earlier image-producing step.
- expand_temporal_neighbors: source_step_ids from an earlier memory-producing step,
  radius (1..2).
- verify_evidence_sufficiency: source_step_ids, required_modalities, and
  minimum_evidence_groups. It is the final non-control step for answerable candidates.
- fallback_to_multimodal_rag: source_step_ids, condition=if_insufficient, and a generic
  reason_code. It is an optional final control step.
- abstain_if_no_evidence: source_step_ids, minimum_evidence_groups, required_modalities, and
  a generic reason_code. A refusal candidate uses this as its only step.

Safe templates:
- Text: retrieve_text_by_embedding -> verify_evidence_sufficiency -> optional fallback.
- Visual: retrieve_image_by_visual_embedding -> inspect_raw_image ->
  verify_evidence_sufficiency -> optional fallback.
- Temporal/update/conflict: retrieve_text_by_embedding -> expand_temporal_neighbors ->
  verify_evidence_sufficiency -> optional fallback.
- Refusal: abstain_if_no_evidence only.

Use question_text_query for text retrieval and question_joint_query for visual retrieval.
Never copy any evidence alias such as E01 into terms, strategy_code, rationale_summary,
reason_code, or any other output field. Never output canonical evidence IDs, collection IDs,
run IDs, paths, URLs, SQL, Python, shell, arbitrary code, or functions. Do not put expected
evidence into retrieval args. Visual-detail programs must reach inspect_raw_image.
Temporal/update/conflict programs must use temporal expansion. A control operator is final.
Return only the strict JSON object and at most one short rationale sentence; never return
chain-of-thought."""

PLANNER_V11_PROMPT = """You are Planner v1.1 for a prospective multimodal memory
compiler. For each candidate alias, emit exactly one declarative Memory Program using only
the ten registered operators. You receive a question, taxonomy, reasoning type, and a
server-derived requirement contract. You never receive or infer an expected answer,
expected evidence, canonical identifier, or benchmark gold.

All steps are zero-based and linear. source_step_ids may reference only earlier steps.
Keyword retrieval declares term_source=canonical_question and term_budget; the server
extracts terms. Embedding retrieval uses only question_text_query or question_joint_query.
Visual paths retrieve an image before inspect_raw_image. Image-text paths must use
retrieve_image_text_pair or retrieve_context_around_image. Temporal/update/conflict paths
must use expand_temporal_neighbors. Multi-hop paths must be capable of returning at least
two independent evidence groups.

Every answerable program ends with verify_evidence_sufficiency followed by
fallback_to_multimodal_rag(condition=if_insufficient). The verify modalities and minimum
group count must exactly match the requirement contract. A refusal compiles only to
abstain_if_no_evidence. Never emit answer literals, evidence aliases, IDs, collection names,
paths, URLs, Python, SQL, shell, or arbitrary code. Return strict JSON only and at most one
short strategy summary, never chain-of-thought."""

PLANNER_V12_PROMPT = """You are Planner v1.2 for a prospective multimodal memory
compiler. Select one retrieval strategy for exactly one answerable candidate. You receive
only its user-facing question, taxonomy, reasoning type, and a server-derived requirement
contract. You never receive expected answers, expected evidence, canonical IDs, benchmark
gold, collections, paths, or code.

Choose one or more retrieval_channels from keyword, text_dense, and visual_dense. Choose
only needed expansion_capabilities from image_text, context, temporal, and raw_image.
Choose top_k_tier from narrow, standard, and broad. Return a short strategy_code and
confidence. The server deterministically creates all operator arguments, source_step_ids,
query-vector roles, verification, and conditional fallback. Do not emit operators,
keywords, answer literals, locators, rationale, SQL, Python, shell, URLs, or chain-of-thought.
Return only the strict single-strategy JSON object."""

PLANNER_V12_REVISION_PROMPT = """You are Planner v1.2 revising one retrieval strategy.
You receive only the prior strategy, a server-derived requirement contract, a typed
capability trace, and bounded Doubter directives. Select a replacement retrieval strategy
that addresses those directives. Do not infer or emit an answer, evidence locator,
canonical identifier, collection, path, URL, code, or chain-of-thought. Return only the
same strict single-strategy JSON object used for the initial plan."""

PLANNER_V13_PROMPT = """You are Planner v1.3 for a prospective multimodal memory
compiler. Select one retrieval strategy for exactly one answerable candidate. You receive
only its natural user-facing question and a server-derived requirement contract containing
required and allowed capabilities. Request only provider-selectable capabilities in that
allowlist. control, evidence_groups, and verification are server-managed and must never
appear in retrieval_channels or expansion_capabilities. The server
creates keywords, vectors, source-step IDs, output types, verification, and conditional
fallback.

Cover every provider-selectable required capability: text requires keyword or text_dense;
image requires visual_dense; image_text, context, temporal, and raw_image require the
same-named expansion capability. Optional capabilities may be omitted.

Return exactly {"strategy": {"candidate_alias": "Q01", "retrieval_channels": [...],
"expansion_capabilities": [...], "top_k_tier": "standard", "strategy_code": "...",
"confidence": 0.0}}. Echo the candidate_alias from the request exactly. Allowed retrieval
channels are keyword, text_dense, and visual_dense. Allowed expansion capabilities are
image_text, context, temporal, and raw_image. top_k_tier is narrow, standard, or broad.
Lists must not contain duplicates. strategy_code must be a short identifier, not a sentence.
Never emit expected answers, evidence, canonical locators, collection names, paths, URLs,
code, or chain-of-thought. Return only the strict single-strategy JSON object."""

PLANNER_V13_REVISION_PROMPT = """You are Planner v1.3 revising one non-oracle retrieval
strategy. Use only the prior strategy, requirement allowlist, capability trace, and typed
Doubter directives. Return a replacement strategy within the allowlist. Never emit answers,
evidence locators, identifiers, collections, paths, URLs, code, or chain-of-thought."""

PLANNER_V14_PROMPT = """You are Planner v1.4 for a prospective multimodal memory
compiler. Select one minimal retrieval strategy for exactly one answerable candidate. You
receive only its natural user-facing question and an immutable server-derived requirement
contract. Choose retrieval_channels, allowed expansion_capabilities, one top_k_tier, a short
strategy_code, and confidence. Lists must not contain duplicates. Never emit answers,
evidence, locators, collection names, paths, URLs, code, or reasoning. The server creates all
keywords, vectors, dataflow references, verification, fallback, and any later typed revision.
Cover every provider-selectable required capability: text requires keyword or text_dense;
image requires visual_dense; image_text, context, temporal, and raw_image require the
same-named expansion capability. Never request a capability outside allowed_capabilities.
Arrays are sets, not fixed-width slots: never pad an array and never repeat a value. Use an
empty expansion_capabilities array when no expansion is needed. strategy_code must contain
only lowercase ASCII letters, digits, and underscores and must begin with a letter.
Return exactly {"strategy":{"candidate_alias":"Q01","retrieval_channels":["text_dense"],
"expansion_capabilities":[],"top_k_tier":"standard","strategy_code":"text_dense",
"confidence":0.8}}, replacing values only as allowed by the request schema and echoing the
request's candidate_alias exactly. Return no other keys or text."""


def planner_prompt_text(config: PlannerConfig) -> str:
    if config.is_v15:
        return PLANNER_V14_PROMPT.replace("v1.4", "v1.5") + (
            "\nEvidence expansion fanout and root quotas are preflight-budgeted by the server."
            " Use only the candidate question vectors; never request new query embeddings."
            " Raw-image inspection is forbidden when absent from allowed_capabilities."
        )
    if config.is_v14:
        return PLANNER_V14_PROMPT
    if config.is_v13:
        return PLANNER_V13_PROMPT
    if config.is_v12:
        return PLANNER_V12_PROMPT
    return PLANNER_V11_PROMPT if config.is_v11 else PLANNER_PROMPT


@dataclass(frozen=True)
class PlannerEvidenceView:
    alias: str
    evidence: OperatorEvidence
    text: str | None = None
    image: ImageRef | None = None


def planner_prompt_sha256(config: PlannerConfig | None = None) -> str:
    text = PLANNER_PROMPT if config is None else planner_prompt_text(config)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def planner_revision_prompt_sha256() -> str:
    return hashlib.sha256(PLANNER_V12_REVISION_PROMPT.encode("utf-8")).hexdigest()


def planner_response_schema(config: PlannerConfig | None = None) -> dict[str, object]:
    model = (
        PlannerStrategyEnvelopeV12
        if config is not None and (config.is_v12 or config.is_v13)
        else PlannerGenerationEnvelopeV11
        if config is not None and config.is_v11
        else PlannerGenerationEnvelope
    )
    return model.model_json_schema()


def planner_request_response_schema(
    config: PlannerConfig,
    *,
    candidate_alias: str,
    requirement_contract: PlannerRequirementContract | None = None,
) -> dict[str, object]:
    """Bind the single-candidate scope into the provider's guided JSON schema."""

    schema = deepcopy(planner_response_schema(config))
    if not (config.is_v12 or config.is_v13):
        return schema
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Planner response schema has no properties")
    strategy = properties.get("strategy")
    if not isinstance(strategy, dict):
        raise ValueError("Planner response schema has no strategy property")
    reference = strategy.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
        raise ValueError("Planner strategy schema reference is malformed")
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise ValueError("Planner response schema has no definitions")
    definition = definitions.get(reference.removeprefix("#/$defs/"))
    if not isinstance(definition, dict):
        raise ValueError("Planner strategy schema definition is missing")
    strategy_properties = definition.get("properties")
    if not isinstance(strategy_properties, dict):
        raise ValueError("Planner strategy definition has no properties")
    alias_schema = strategy_properties.get("candidate_alias")
    if not isinstance(alias_schema, dict):
        raise ValueError("Planner strategy candidate alias schema is missing")
    alias_schema["const"] = candidate_alias
    if config.is_v13 and requirement_contract is not None:
        allowed = set(requirement_contract.allowed_capabilities)
        retrieval_channels: list[str] = []
        if PlannerCapability.TEXT in allowed:
            retrieval_channels.extend(["keyword", "text_dense"])
        if PlannerCapability.IMAGE in allowed:
            retrieval_channels.append("visual_dense")
        if not retrieval_channels:
            raise ValueError("answerable Planner contract permits no retrieval channel")
        retrieval_definition = definitions.get("PlannerRetrievalChannel")
        if not isinstance(retrieval_definition, dict):
            raise ValueError("Planner retrieval-channel schema definition is missing")
        retrieval_definition["enum"] = retrieval_channels
        retrieval_property = strategy_properties.get("retrieval_channels")
        if not isinstance(retrieval_property, dict):
            raise ValueError("Planner retrieval-channel property is missing")
        required = set(requirement_contract.required_capabilities)
        valid_retrieval_sets = [
            list(values)
            for size in range(1, min(2, len(retrieval_channels)) + 1)
            for values in combinations(retrieval_channels, size)
            if not ({"keyword", "text_dense"} <= set(values))
            and (
                PlannerCapability.TEXT not in required
                or bool({"keyword", "text_dense"} & set(values))
            )
            and (PlannerCapability.IMAGE not in required or "visual_dense" in values)
        ]
        if not valid_retrieval_sets:
            raise ValueError("Planner required retrieval capabilities exceed the allowlist")
        # The transport projection removes uniqueItems for vLLM/XGrammar. An
        # array-valued enum preserves the exact requirement-scoped set choices,
        # so duplicate or redundant roots cannot be produced by guided decoding.
        retrieval_property["enum"] = valid_retrieval_sets
        retrieval_property["minItems"] = min(len(values) for values in valid_retrieval_sets)
        retrieval_property["maxItems"] = max(len(values) for values in valid_retrieval_sets)

        expansion_values = [
            value
            for value, capability in (
                ("image_text", PlannerCapability.IMAGE_TEXT),
                ("context", PlannerCapability.CONTEXT),
                ("temporal", PlannerCapability.TEMPORAL),
                ("raw_image", PlannerCapability.RAW_IMAGE),
            )
            if capability in allowed
        ]
        required_expansion_values = {
            value
            for value, capability in (
                ("image_text", PlannerCapability.IMAGE_TEXT),
                ("context", PlannerCapability.CONTEXT),
                ("temporal", PlannerCapability.TEMPORAL),
                ("raw_image", PlannerCapability.RAW_IMAGE),
            )
            if capability in required
        }
        if not required_expansion_values <= set(expansion_values):
            raise ValueError("Planner required expansion capabilities exceed the allowlist")
        expansion_property = strategy_properties.get("expansion_capabilities")
        if not isinstance(expansion_property, dict):
            raise ValueError("Planner expansion-capability property is missing")
        valid_expansion_sets = [
            list(values)
            for size in range(0, len(expansion_values) + 1)
            for values in combinations(expansion_values, size)
            if required_expansion_values <= set(values)
        ]
        if not valid_expansion_sets:
            raise ValueError("Planner expansion allowlist has no valid set")
        expansion_property["enum"] = valid_expansion_sets
        expansion_property["minItems"] = min(len(values) for values in valid_expansion_sets)
        expansion_property["maxItems"] = max(len(values) for values in valid_expansion_sets)
        if expansion_values:
            expansion_definition = definitions.get("PlannerExpansionCapability")
            if not isinstance(expansion_definition, dict):
                raise ValueError("Planner expansion schema definition is missing")
            expansion_definition["enum"] = expansion_values
    return schema


def planner_response_schema_sha256(config: PlannerConfig | None = None) -> str:
    payload = json.dumps(
        planner_response_schema(config),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_validation_diagnostics(exc: ValidationError | ValueError) -> list[dict[str, str]]:
    if isinstance(exc, ValidationError):

        def safe_location(parts: Sequence[object]) -> str:
            safe = [
                str(part)
                if isinstance(part, int) or str(part) in _SAFE_DIAGNOSTIC_FIELDS
                else "unknown_field"
                for part in parts
            ]
            return ".".join(safe) or "program"

        values: list[dict[str, str]] = []
        for item in exc.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        ):
            location = safe_location(item.get("loc", ()))
            error_type = str(item.get("type", "validation_error"))
            message = str(item.get("msg", ""))
            if "retrieval_channels must not contain duplicate values" in message:
                location = "strategy.retrieval_channels"
                error_type = "duplicate_items"
            elif "expansion_capabilities must not contain duplicate values" in message:
                location = "strategy.expansion_capabilities"
                error_type = "duplicate_items"
            elif "Planner strategy contains an unsafe provider literal" in message:
                error_type = "unsafe_provider_literal"
            values.append({"location": location, "type": error_type})
        return sorted(values, key=lambda item: (item["location"], item["type"]))
    return [{"location": "program", "type": type(exc).__name__}]


def _validated_revision_context(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the non-oracle Planner revision view at the renderer boundary."""

    if set(value) != {"prior_strategy", "capability_trace", "directives"}:
        raise ValueError(
            "Planner revision context must contain only strategy, capability, and directive facts"
        )
    prior = PlannerStrategyDraftV12.model_validate(value["prior_strategy"])
    trace = value["capability_trace"]
    if not isinstance(trace, Mapping) or set(trace) != {
        "round_id",
        "operators",
        "output_types",
        "top_k",
    }:
        raise ValueError("Planner revision capability trace is malformed")
    round_id = trace["round_id"]
    top_k = trace["top_k"]
    operators = trace["operators"]
    output_types = trace["output_types"]
    if (
        not isinstance(round_id, int)
        or isinstance(round_id, bool)
        or round_id < 0
        or round_id > 2
        or not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
        or top_k > 50
        or not isinstance(operators, list)
        or not isinstance(output_types, list)
        or not operators
        or len(operators) != len(output_types)
        or len(operators) > 8
        or any(item not in _REVISION_OPERATORS for item in operators)
        or any(item not in _REVISION_OUTPUT_TYPES for item in output_types)
    ):
        raise ValueError("Planner revision capability trace is invalid")
    directives = value["directives"]
    if not isinstance(directives, list) or not directives or len(directives) > 8:
        raise ValueError("Planner revision requires one to eight typed directives")
    validated_directives: list[dict[str, object]] = []
    for item in directives:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"directive", "top_k_tier"}
            or not isinstance(item["directive"], str)
        ):
            raise ValueError("Planner revision directive is malformed")
        try:
            directive = DoubterDirectiveCode(item["directive"])
        except ValueError as exc:
            raise ValueError("Planner revision directive is unknown") from exc
        top_k_tier = item["top_k_tier"]
        if top_k_tier is not None and (
            not isinstance(top_k_tier, int)
            or isinstance(top_k_tier, bool)
            or top_k_tier < 1
            or top_k_tier > 50
        ):
            raise ValueError("Planner revision top-k directive is invalid")
        if (directive is DoubterDirectiveCode.INCREASE_TOP_K) != (top_k_tier is not None):
            raise ValueError("Planner revision top-k directive is inconsistent")
        validated_directives.append(
            {
                "directive": directive.value,
                "top_k_tier": top_k_tier,
            }
        )
    payload: dict[str, object] = {
        "prior_strategy": prior.model_dump(mode="json"),
        "capability_trace": {
            "round_id": round_id,
            "operators": list(operators),
            "output_types": list(output_types),
            "top_k": top_k,
        },
        "directives": validated_directives,
    }
    if _contains_unsafe_revision_value(payload):
        raise ValueError("Planner revision context contains an unsafe locator or code")
    return payload


def _contains_unsafe_revision_value(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_unsafe_revision_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_unsafe_revision_value(item) for item in value)
    if not isinstance(value, str):
        return False
    lowered = value.casefold()
    forbidden = (
        "expected_answer",
        "expected_evidence",
        "gold_answer",
        "gold_evidence",
        "canonical_id",
        "collection_id",
        "postgresql://",
        "postgresql+",
        "s3://",
        "minio://",
        "http://",
        "https://",
        "/data/users/",
        "/workspace/",
    )
    return any(item in lowered for item in forbidden)


def render_planner_request(
    *,
    request_id: str,
    config: PlannerConfig,
    candidates: Sequence[tuple[str, QuestionCandidate]],
    evidence_views: Mapping[str, Sequence[PlannerEvidenceView]] | None = None,
    requirement_contracts: Mapping[str, PlannerRequirementContract] | None = None,
    revision_context: Mapping[str, object] | None = None,
    timeout_seconds: float | None = None,
) -> GenerationRequest:
    if (config.is_v12 or config.is_v13) and len(candidates) != 1:
        raise ValueError("Planner strategy protocols require exactly one candidate")
    if not candidates or len(candidates) > config.candidates_per_batch:
        raise ValueError("Planner request requires one to four candidates")
    aliases = [alias for alias, _ in candidates]
    if len(aliases) != len(set(aliases)):
        raise ValueError("Planner candidate aliases must be unique")
    parts: list[ModelContentPart] = []
    evidence_views = evidence_views or {}
    requirement_contracts = requirement_contracts or {}
    for alias, candidate in candidates:
        if config.is_v11 or config.is_v12 or config.is_v13:
            contract = requirement_contracts.get(alias)
            if contract is None or contract.qbank_id != candidate.qbank_id:
                raise ValueError("non-oracle Planner requires a requirement contract")
            requirement_payload: dict[str, object] = {
                "required_modalities": [item.value for item in contract.required_modalities],
                "minimum_evidence_groups": contract.minimum_evidence_groups,
                "required_capabilities": [item.value for item in contract.required_capabilities],
                "fallback_required": contract.fallback_required,
                "refusal_only": contract.refusal_only,
            }
            if config.is_v13:
                requirement_payload["allowed_capabilities"] = [
                    item.value for item in contract.allowed_capabilities
                ]
            payload: dict[str, object] = {
                "candidate_alias": alias,
                "question_type": candidate.question_type.value,
                "reasoning_type": candidate.reasoning_type.value,
                "requirement_contract": requirement_payload,
            }
            if revision_context is None:
                payload["question"] = candidate.canonical_question
            else:
                if not (config.is_v12 or config.is_v13) or len(candidates) != 1:
                    raise ValueError("Planner revision requires one strategy candidate")
                payload["revision"] = _validated_revision_context(revision_context)
            parts.append(
                ModelContentPart(
                    kind=ModelContentKind.TEXT,
                    text=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                )
            )
            continue
        views = list(evidence_views.get(alias, ()))
        typed_evidence = [
            {
                "alias": item.alias,
                "type": item.evidence.evidence_type.value,
                "text": item.text,
                "has_image": item.image is not None,
            }
            for item in views
        ]
        payload = {
            "candidate_alias": alias,
            "question": candidate.canonical_question,
            "question_type": candidate.question_type.value,
            "reasoning_type": candidate.reasoning_type.value,
            "required_modalities": [item.value for item in candidate.required_modalities],
            "expected_answer": candidate.expected_answer,
            "expected_evidence": typed_evidence,
        }
        parts.append(
            ModelContentPart(
                kind=ModelContentKind.TEXT,
                text=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
            )
        )
        for view in views:
            if view.image is not None:
                parts.append(
                    ModelContentPart(
                        kind=ModelContentKind.TEXT,
                        text=f"[EVIDENCE_IMAGE alias={view.alias}]",
                    )
                )
                parts.append(ModelContentPart(kind=ModelContentKind.IMAGE, image=view.image))
    request_schema = planner_request_response_schema(
        config,
        candidate_alias=aliases[0],
        requirement_contract=(requirement_contracts[aliases[0]] if config.is_v13 else None),
    )
    request_schema_hash = hashlib.sha256(
        json.dumps(
            request_schema,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return GenerationRequest(
        request_id=request_id,
        model_alias=config.planner_model_alias,
        messages=[
            ModelMessage(
                role=SpeakerRole.SYSTEM,
                parts=[
                    ModelContentPart(
                        kind=ModelContentKind.TEXT,
                        text=(
                            (
                                PLANNER_V13_REVISION_PROMPT
                                if config.is_v13
                                else PLANNER_V12_REVISION_PROMPT
                            )
                            if revision_context is not None
                            else planner_prompt_text(config)
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
                    "pmmc_planner_strategy_v1_5"
                    if config.is_v15
                    else "pmmc_planner_strategy_v1_4"
                    if config.is_v14
                    else "pmmc_planner_strategy_v1_3"
                    if config.is_v13
                    else "pmmc_planner_strategy_v1_2"
                    if config.is_v12
                    else "pmmc_planner_program_batch_v1"
                ),
                "strict": True,
                "schema": request_schema,
            },
        },
        metadata={
            "task": ("pmmc_planner_revision" if revision_context is not None else "pmmc_planner"),
            "prompt_version": config.prompt_version,
            "renderer_version": config.renderer_version,
            "response_schema_hash": planner_response_schema_sha256(config),
            "request_response_schema_hash": request_schema_hash,
            "candidate_count": len(candidates),
            "non_oracle_input": config.is_v11 or config.is_v12 or config.is_v13,
            "requirement_contract_hashes": (
                {alias: requirement_contracts[alias].content_hash for alias in aliases}
                if config.is_v11 or config.is_v12 or config.is_v13
                else {}
            ),
            **({"single_candidate_scope": True} if config.is_v12 or config.is_v13 else {}),
            **({"revision_scope": True} if revision_context is not None else {}),
        },
    )


def planner_output_parser(
    *,
    expected_aliases: Sequence[str],
    config: PlannerConfig | None = None,
):  # type: ignore[no-untyped-def]
    expected = tuple(expected_aliases)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected Planner aliases must be non-empty and unique")

    def parse(result):  # type: ignore[no-untyped-def]
        from pmmc.models.execution import ParsedModelOutput

        repairs = 0
        initial_error: str | None = None
        try:
            payload = json.loads(result.text)
        except (json.JSONDecodeError, TypeError, ValueError):
            from json_repair import repair_json

            repairs = 1
            initial_error = "planner_json_repaired"
            try:
                payload = json.loads(repair_json(result.text))
            except (json.JSONDecodeError, TypeError, ValueError):
                payload = {}
                initial_error = "planner_json_invalid"

        if config is not None and (config.is_v12 or config.is_v13):
            strategy_payload = payload.get("strategy") if isinstance(payload, Mapping) else None
            diagnostics: list[dict[str, object]] = []
            strategy: PlannerStrategyDraftV12 | None = None
            if strategy_payload is None:
                diagnostics.append(
                    {
                        "candidate_ordinal": 0,
                        "errors": [{"location": "strategy", "type": "missing_strategy"}],
                    }
                )
            else:
                try:
                    strategy = PlannerStrategyEnvelopeV12.model_validate(
                        payload if config.is_v15 else {"strategy": strategy_payload}
                    ).strategy
                    if strategy.candidate_alias != expected[0]:
                        raise ValueError("Planner strategy candidate scope mismatch")
                except (ValidationError, ValueError) as exc:
                    diagnostics.append(
                        {
                            "candidate_ordinal": 0,
                            "errors": _safe_validation_diagnostics(exc),
                        }
                    )
                    strategy = None
            return ParsedModelOutput(
                value={
                    "strategies": (
                        [strategy.model_dump(mode="json")] if strategy is not None else []
                    ),
                    "programs": [],
                    "invalid_candidate_aliases": ([] if strategy is not None else [expected[0]]),
                    "invalid_diagnostics": diagnostics,
                    "unexpected_alias_count": 0,
                },
                repair_attempt_count=repairs,
                initial_error=initial_error,
            )

        raw_programs = payload.get("programs", []) if isinstance(payload, Mapping) else []
        if not isinstance(raw_programs, list):
            raw_programs = []
            initial_error = "planner_envelope_invalid"
        by_alias: dict[str, list[object]] = defaultdict(list)
        for item in raw_programs:
            if isinstance(item, Mapping) and isinstance(item.get("candidate_alias"), str):
                by_alias[item["candidate_alias"]].append(item)

        valid: list[PlannerProgramDraft] = []
        invalid_aliases: list[str] = []
        diagnostics: list[dict[str, object]] = []
        for ordinal, alias in enumerate(expected):
            values = by_alias.get(alias, [])
            if len(values) != 1:
                invalid_aliases.append(alias)
                diagnostics.append(
                    {
                        "candidate_ordinal": ordinal,
                        "errors": [
                            {
                                "location": "candidate_alias",
                                "type": "missing_or_duplicate_candidate",
                            }
                        ],
                    }
                )
                continue
            try:
                if config is not None and config.is_v11:
                    value = PlannerGenerationEnvelopeV11.model_validate(
                        {"programs": [values[0]]}
                    ).programs[0]
                    valid.append(PlannerProgramDraft.model_validate(value.model_dump(mode="json")))
                else:
                    valid.append(PlannerProgramDraft.model_validate(values[0]))
            except (ValidationError, ValueError) as exc:
                invalid_aliases.append(alias)
                diagnostics.append(
                    {
                        "candidate_ordinal": ordinal,
                        "errors": _safe_validation_diagnostics(exc),
                    }
                )
        unexpected_count = sum(alias not in expected for alias in by_alias)
        return ParsedModelOutput(
            value={
                "programs": [item.model_dump(mode="json") for item in valid],
                "invalid_candidate_aliases": invalid_aliases,
                "invalid_diagnostics": diagnostics,
                "unexpected_alias_count": unexpected_count,
            },
            repair_attempt_count=repairs,
            initial_error=initial_error,
        )

    return parse
