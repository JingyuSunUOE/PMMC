"""Questioner v3.6 native support, allocation and verifier-v7 contracts.

All selection inputs are memory-side facts. No benchmark query or gold is accepted.
"""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

from pydantic import ValidationError

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.opportunities import (
    BuiltQuestionOpportunitySet,
    QuestionOpportunityBuilder,
    local_opportunity_id,
    validate_evidence_first_bindings,
)
from pmmc.compiler.questioner_config import EVIDENCE_FIRST_VERSION
from pmmc.models.execution import ModelOutputContractError
from pmmc.schemas import (
    CandidateRejectionCode,
    MemoryUnit,
    Modality,
    QuestionerAnswerClaimKind,
    QuestionerAnswerType,
    QuestionerCandidateKind,
    QuestionOpportunityType,
    QuestionReasoningType,
    QuestionType,
)
from pmmc.schemas.questioner import (
    QuestionerAnswerableDraftV36,
    QuestionerBatchV36,
    QuestionerControlDraftV36,
    QuestionerNativeCandidateDraft,
    QuestionerNativeSupport,
    QuestionerResolvedClaimV36,
    QuestionerVerificationEnvelopeV7,
    QuestionerVerifierCandidateV36,
    QuestionOpportunity,
    QuestionOpportunityEvidenceRef,
)

TYPE_ORDER = (
    "update_pair",
    "temporal_pair",
    "multi_evidence",
    "visual_search",
    "image_text_binding",
    "conflict_pair",
    "visual_detail",
    "local_fact",
    "preference",
)


class NativeSupportError(ValueError):
    """Safe, stable diagnostic, containing no source text."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class NativeVerifierOutputError(NativeSupportError, ModelOutputContractError):
    """Malformed verifier output, retained as a failed audited response."""


def verifier_required_dimensions(candidate) -> tuple[tuple[str, ...], tuple[str, ...]]:
    dimensions = [
        "user_facing", "future_plausible", "not_historical_replay",
        "answer_addresses_question", "answer_explicitly_entailed", "speaker_subject_aligned",
        "evidence_sufficient", "taxonomy_compatible", "metadata_free",
        "question_answer_compatible", "answer_type_correct", "source_naturalness",
    ]
    if Modality.IMAGE in candidate.required_modalities:
        dimensions.append("visual_necessary")
    if candidate.reasoning_type is QuestionReasoningType.MULTI_HOP:
        dimensions.append("multi_hop_genuine")
    closure = []
    if candidate.reasoning_type in {
        QuestionReasoningType.UPDATE,
        QuestionReasoningType.TEMPORAL,
        QuestionReasoningType.CONFLICT,
    }:
        closure.append("same_entity_attribute")
    if candidate.answer_type is QuestionerAnswerType.COUNT or candidate.image_set_scope:
        closure.append("explicit_count_set")
    return tuple(dimensions), tuple(closure)


def questioner_memory_view(config, memories: Sequence[MemoryUnit]) -> list[MemoryUnit]:
    """A treatment-specific view, leaving original memory records and captions intact."""
    if config.is_v36 and config.memory_raw_image_policy == "forbid":
        return [
            memory.model_copy(update={"image_ids": [], "image_refs": []}) for memory in memories
        ]
    return list(memories)


def build_questioner_opportunities(*, config, unit, memories, sessions, links):
    built = QuestionOpportunityBuilder(config).build(
        unit=unit, memories=questioner_memory_view(config, memories), sessions=sessions, links=links
    )
    if config.evidence_first:
        # W binds and deduplicates physical images before catalog selection.
        # Never run the legacy catalog expansion on a W-produced set.
        validate_evidence_first_bindings(built)
        return built
    images = sorted(built.image_aliases.values())
    visual = next(
        (
            item
            for item in built.opportunities
            if item.opportunity_type is QuestionOpportunityType.VISUAL_SEARCH
        ),
        None,
    )
    if not config.is_v36 or len(images) < 2 or visual is None:
        return built
    # A bounded scan is not a multi-hop assertion. All images must be assessed
    # to support a complete set answer, even when some are negative members.
    owners = {mid for mid, _image in built.image_aliases}
    group_ids = list(dict.fromkeys(built.alias_to_group[alias] for alias in images))
    expanded = visual.model_copy(
        update={
            "evidence_aliases": images,
            "evidence_group_ids": group_ids,
            "speaker_scope": sorted(
                {memory.speaker.value for memory in memories if memory.memory_id in owners}
            ),
            "relation_content_hash": canonical_content_hash(
                ["image-set-scan-v1", images, group_ids]
            ),
            "metadata": {**visual.metadata, "image_set_scope": True, "image_set_size": len(images)},
        }
    )
    updated = [
        expanded if item.opportunity_id == visual.opportunity_id else item
        for item in built.opportunities
    ]
    specs = [
        item.model_dump(
            mode="json", exclude={"opportunity_id", "opportunity_set_id", "content_hash"}
        )
        for item in updated
    ]
    identity = ["questioner-native-opportunity-set-v1", built.opportunity_set.content_hash, specs]
    set_id = stable_compiler_id("question-opportunity-set", identity)
    opportunities, refs = [], []
    for item, spec in zip(updated, specs, strict=True):
        oid = stable_compiler_id("question-opportunity", [set_id, spec])
        opportunity = item.model_copy(
            update={
                "opportunity_id": oid,
                "opportunity_set_id": set_id,
                "content_hash": canonical_content_hash(spec),
            }
        )
        opportunities.append(opportunity)
        for ordinal, alias in enumerate(item.evidence_aliases):
            evidence = built.alias_to_evidence[alias]
            refs.append(
                QuestionOpportunityEvidenceRef(
                    opportunity_evidence_ref_id=stable_compiler_id(
                        "question-opportunity-evidence", [oid, ordinal, alias]
                    ),
                    run_id=unit.run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    opportunity_id=oid,
                    alias=alias,
                    ordinal=ordinal,
                    evidence=evidence,
                    evidence_group_id=built.alias_to_group[alias],
                    content_hash=canonical_content_hash(
                        [oid, alias, evidence.model_dump(mode="json")]
                    ),
                )
            )
    return replace(
        built,
        opportunity_set=built.opportunity_set.model_copy(
            update={
                "opportunity_set_id": set_id,
                "content_hash": canonical_content_hash(identity),
                "metadata": {**built.opportunity_set.metadata, "image_set_scan_version": "v1"},
            }
        ),
        opportunities=tuple(opportunities),
        evidence_refs=tuple(refs),
    )


def native_support_catalog(
    opportunity_set: BuiltQuestionOpportunitySet,
    memories: Sequence[MemoryUnit],
) -> dict[str, QuestionerNativeSupport]:
    memory_map = {memory.memory_id: memory for memory in memories}
    parent_owners = {
        **{alias: (mid, "text") for mid, alias in opportunity_set.memory_aliases.items()},
        **{alias: (mid, "caption") for mid, alias in opportunity_set.caption_aliases.items()},
    }
    catalog: dict[str, QuestionerNativeSupport] = {}
    cursors: dict[str, int] = defaultdict(int)
    for alias, (parent, quote) in opportunity_set.sentence_span_aliases.items():
        if parent not in parent_owners:
            raise NativeSupportError("span_owner_missing")
        mid, field = parent_owners[parent]
        if mid not in memory_map:
            raise NativeSupportError("span_owner_missing")
        memory = memory_map[mid]
        evidence = opportunity_set.alias_to_evidence.get(parent)
        if evidence is None or evidence.memory_id != mid:
            raise NativeSupportError("span_owner_mismatch")
        source = getattr(memory, field) or ""
        start = source.find(quote, cursors[parent])
        if not quote or start < 0:
            raise NativeSupportError("span_range_mismatch")
        end = start + len(quote)
        cursors[parent] = end
        catalog[alias] = QuestionerNativeSupport(
            support_id=stable_compiler_id(
                "questioner-span-v1", [mid, field, start, end, canonical_content_hash(source)]
            ),
            support_alias=alias,
            evidence_alias=parent,
            owning_memory_id=mid,
            speaker=memory.speaker.value,
            source_field=field,
            start_char=start,
            end_char=end,
            exact_text=source[start:end],
        )
    for (mid, image_id), alias in opportunity_set.image_aliases.items():
        image_memory = memory_map.get(mid)
        evidence = opportunity_set.alias_to_evidence.get(alias)
        if image_memory is None or evidence is None or evidence.image_id != image_id:
            raise NativeSupportError("image_owner_mismatch")
        if evidence.metadata.get("owning_memory_id") != mid:
            raise NativeSupportError("image_owner_mismatch")
        if image_id not in {
            *image_memory.image_ids,
            *(item.image_id for item in image_memory.image_refs),
        }:
            raise NativeSupportError("image_owner_mismatch")
        catalog[alias] = QuestionerNativeSupport(
            support_id=stable_compiler_id("questioner-image-support-v1", [mid, image_id]),
            support_alias=alias,
            evidence_alias=alias,
            owning_memory_id=mid,
            speaker=image_memory.speaker.value,
            source_field="image",
            image_id=image_id,
        )
    return catalog


def allocate_opportunities(opportunity_set: BuiltQuestionOpportunitySet, budget: int = 8):
    """One slot per opportunity, round-robin by type; no quota is backfilled by guesses."""
    if opportunity_set.opportunity_set.builder_version == EVIDENCE_FIRST_VERSION:
        validate_evidence_first_bindings(opportunity_set)
        return tuple(
            item for item in opportunity_set.opportunities
            if item.opportunity_type is not QuestionOpportunityType.REFUSAL_BOUNDARY
        )[:max(0, min(8, budget))]
    buckets = defaultdict(list)
    for opportunity in opportunity_set.opportunities:
        if opportunity.opportunity_type is not QuestionOpportunityType.REFUSAL_BOUNDARY:
            buckets[opportunity.opportunity_type.value].append(opportunity)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (item.ordinal, item.content_hash))
    selected: list[QuestionOpportunity] = []
    while len(selected) < budget:
        previous = len(selected)
        for kind in TYPE_ORDER:
            if buckets[kind] and len(selected) < budget:
                selected.append(buckets[kind].pop(0))
        if previous == len(selected):
            break
    return tuple(selected)


def opportunity_support_aliases(
    built: BuiltQuestionOpportunitySet, opportunity: QuestionOpportunity,
) -> tuple[str, ...]:
    if built.opportunity_set.builder_version == EVIDENCE_FIRST_VERSION:
        if opportunity.opportunity_type is QuestionOpportunityType.REFUSAL_BOUNDARY:
            return ()
        return tuple(opportunity.metadata["support_aliases"])
    return (
        *(alias for alias, (parent, _) in built.sentence_span_aliases.items()
          if parent in opportunity.evidence_aliases),
        *(alias for alias in built.image_aliases.values() if alias in opportunity.evidence_aliases),
    )


def native_response_schema(opportunity_set: BuiltQuestionOpportunitySet) -> dict:
    schema = QuestionerBatchV36.model_json_schema()
    definitions = schema["$defs"]

    def branch(opportunity, control=False):
        name = "QuestionerControlDraftV36" if control else "QuestionerAnswerableDraftV36"
        result = copy.deepcopy(definitions[name])
        result["properties"]["opportunity_id"] = {
            "type": "string",
            "const": local_opportunity_id(opportunity),
        }
        if not control:
            aliases = opportunity_support_aliases(opportunity_set, opportunity)
            spans = opportunity_set.sentence_span_aliases
            text_aliases = sorted(alias for alias in aliases if alias in spans)
            image_aliases = sorted(alias for alias in aliases if alias not in spans)
            if opportunity_set.opportunity_set.builder_version == EVIDENCE_FIRST_VERSION:
                result["properties"]["answer_type"] = {
                    "type": "string", "enum": opportunity.metadata["expected_answer_types"],
                }
            claim_branches = []
            for kind, aliases in (("text", text_aliases), ("visual", image_aliases)):
                if aliases:
                    claim = copy.deepcopy(definitions["QuestionerClaimDraftV36"])
                    claim["properties"]["claim_kind"] = {"const": kind, "type": "string"}
                    claim["properties"]["support_alias"] = {"type": "string", "enum": aliases}
                    claim_branches.append(claim)
            if not claim_branches:
                return None
            result["properties"]["answer_claims"]["items"] = {"anyOf": claim_branches}
            if opportunity.metadata.get("image_set_scope"):
                result["properties"]["answer_type"] = {"const": "list", "type": "string"}
                result["properties"]["answer_claims"]["minItems"] = len(image_aliases)
        return result

    answerable = [
        value
        for item in allocate_opportunities(opportunity_set)
        if (value := branch(item)) is not None
    ]
    controls = [
        branch(item, True)
        for item in opportunity_set.opportunities
        if item.opportunity_type is QuestionOpportunityType.REFUSAL_BOUNDARY
    ]
    for key, branches, limit in (
        ("candidates", answerable, len(answerable)),
        ("control_candidates", controls, min(1, len(controls))),
    ):
        schema["properties"][key]["maxItems"] = limit
        schema["properties"][key]["items"] = {"anyOf": branches} if branches else {}
    return schema


@dataclass(frozen=True)
class ResolvedQuestionerV36Draft:
    provider_for_verifier: QuestionerVerifierCandidateV36
    internal: QuestionerNativeCandidateDraft
    claims: tuple[QuestionerResolvedClaimV36, ...]
    evidence_aliases: tuple[str, ...]
    claim_support_aliases: tuple[str, ...]
    answer_type: QuestionerAnswerType


def resolve_v36_draft(
    draft: QuestionerAnswerableDraftV36 | QuestionerControlDraftV36,
    *,
    opportunity_set: BuiltQuestionOpportunitySet,
    memories: Sequence[MemoryUnit],
    memory_raw_image_policy: str = "allow",
) -> ResolvedQuestionerV36Draft:
    opportunity = next(
        (
            item
            for item in opportunity_set.opportunities
            if local_opportunity_id(item) == draft.opportunity_id
        ),
        None,
    )
    if opportunity is None:
        raise NativeSupportError("opportunity_unknown")
    control = isinstance(draft, QuestionerControlDraftV36)
    if control != (opportunity.opportunity_type is QuestionOpportunityType.REFUSAL_BOUNDARY):
        raise NativeSupportError("control_without_refusal_opportunity")
    if control and opportunity_set.opportunity_set.builder_version == EVIDENCE_FIRST_VERSION:
        validate_evidence_first_bindings(opportunity_set)
    if not control and opportunity not in allocate_opportunities(opportunity_set):
        raise NativeSupportError("opportunity_not_allocated")
    if (
        isinstance(draft, QuestionerAnswerableDraftV36)
        and opportunity_set.opportunity_set.builder_version == EVIDENCE_FIRST_VERSION
    ):
        allowed = opportunity_support_aliases(opportunity_set, opportunity)
        if any(claim.support_alias not in allowed for claim in draft.answer_claims):
            raise NativeSupportError("support_outside_opportunity")
        if draft.answer_type not in opportunity.metadata["expected_answer_types"]:
            raise NativeSupportError("answer_type_outside_opportunity")
        if opportunity.opportunity_type is QuestionOpportunityType.IMAGE_TEXT_BINDING and {
            claim.claim_kind for claim in draft.answer_claims
        } != {QuestionerAnswerClaimKind.TEXT, QuestionerAnswerClaimKind.VISUAL}:
            raise NativeSupportError("image_text_support_incomplete")
    catalog = native_support_catalog(opportunity_set, memories)
    claims = []
    for claim in draft.answer_claims:
        if (
            memory_raw_image_policy == "forbid"
            and claim.claim_kind is QuestionerAnswerClaimKind.VISUAL
        ):
            raise NativeSupportError("raw_image_forbidden")
        support = catalog.get(claim.support_alias)
        if support is None:
            raise NativeSupportError("support_alias_unknown")
        if support.evidence_alias not in opportunity.evidence_aliases:
            raise NativeSupportError("support_outside_opportunity")
        if (claim.claim_kind is QuestionerAnswerClaimKind.VISUAL) != (
            support.source_field == "image"
        ):
            raise NativeSupportError("support_kind_mismatch")
        claims.append(QuestionerResolvedClaimV36(**claim.model_dump(), support=support))
    aliases = tuple(claim.support.evidence_alias for claim in claims)
    answer_type = (
        QuestionerAnswerType(draft.answer_type)
        if isinstance(draft, QuestionerAnswerableDraftV36)
        else QuestionerAnswerType.REFUSAL
    )
    if opportunity.metadata.get("image_set_scope") and (
        answer_type is not QuestionerAnswerType.LIST
        or set(aliases) != set(opportunity.evidence_aliases)
    ):
        raise NativeSupportError("image_set_scan_incomplete")
    owners = {claim.support.owning_memory_id for claim in claims}
    groups = {opportunity_set.alias_to_group[alias] for alias in aliases}
    independent_sources = {
        ("image", claim.support.image_id)
        if claim.support.image_id is not None
        else ("memory", claim.support.owning_memory_id)
        for claim in claims
    }
    group_count = min(len(owners), len(groups), len(independent_sources))
    if not control and group_count < opportunity.minimum_evidence_groups:
        raise NativeSupportError("independent_support_groups_missing")
    provider = QuestionerVerifierCandidateV36(
        candidate_kind=draft.candidate_kind,
        opportunity_id=draft.opportunity_id,
        generated_question=draft.generated_question,
        question_type=opportunity.allowed_question_types[0],
        reasoning_type=opportunity.allowed_reasoning_types[0],
        expected_answer=None if control else " ".join(claim.claim_text for claim in claims),
        evidence_aliases=list(dict.fromkeys(aliases)),
        required_modalities=opportunity.required_modalities,
        paraphrases=[],
        confidence=draft.confidence,
        answer_type=answer_type,
        exact_claims=claims,
        image_set_scope=bool(opportunity.metadata.get("image_set_scope")),
    )
    evidence_by_id = {
        opportunity_set.alias_to_evidence[alias].evidence_id: opportunity_set.alias_to_evidence[
            alias
        ]
        for alias in reversed(aliases)
    }
    internal = QuestionerNativeCandidateDraft(
        generated_question=draft.generated_question,
        question_type=QuestionType(opportunity.allowed_question_types[0].value),
        reasoning_type=opportunity.allowed_reasoning_types[0],
        expected_answer=provider.expected_answer,
        source_memory_ids=sorted(owners),
        expected_evidence=list(evidence_by_id.values()),
        required_modalities=[Modality(item.value) for item in opportunity.required_modalities],
        evidence_requirement=[opportunity.opportunity_type.value],
        hop_count=max(1, group_count),
        paraphrases=[],
        confidence=draft.confidence,
        opportunity_id=opportunity.opportunity_id,
        candidate_kind=draft.candidate_kind,
        answer_type=answer_type,
        exact_claims=claims,
    )
    return ResolvedQuestionerV36Draft(
        provider,
        internal,
        tuple(claims),
        aliases,
        tuple(claim.support_alias for claim in claims),
        answer_type,
    )


def validate_w_verifier_candidate(
    candidate: QuestionerVerifierCandidateV36, built: BuiltQuestionOpportunitySet,
    memories: Sequence[MemoryUnit], memory_raw_image_policy: str,
) -> None:
    # Reuse resolution, including image-set completeness and independent
    # groups. A known alias in the unit is not sufficient for this opportunity.
    wire = candidate.model_dump(include={
        "candidate_kind", "opportunity_id", "generated_question", "confidence",
    })
    wire["answer_claims"] = [c.model_dump(exclude={"support"}) for c in candidate.exact_claims]
    if candidate.candidate_kind is QuestionerCandidateKind.REFUSAL:
        draft: QuestionerControlDraftV36 | QuestionerAnswerableDraftV36 = (
            QuestionerControlDraftV36.model_validate(wire)
        )
    else:
        draft = QuestionerAnswerableDraftV36.model_validate({
            **wire, "answer_type": candidate.answer_type,
        })
    resolved = resolve_v36_draft(
        draft, opportunity_set=built, memories=memories,
        memory_raw_image_policy=memory_raw_image_policy,
    )
    if resolved.provider_for_verifier != candidate:
        raise NativeSupportError("evidence_first_verifier_candidate_mismatch")


def information_need_key(question: str, answer_type: str | None) -> str:
    from pmmc.compiler.questioner import normalize_question_text

    # Modal strength does not create a new information need. Keep negation and entities.
    words = normalize_question_text(question).split()
    words = [word for word in words if word not in {"will", "would", "might", "could", "can"}]
    return canonical_content_hash(["information-need-v1", answer_type, words])


def native_static_codes(draft: QuestionerNativeCandidateDraft) -> list[CandidateRejectionCode]:
    from pmmc.compiler.questioner import QuestionerCandidateGate, normalize_question_text

    codes = QuestionerCandidateGate._v35_answer_contract_codes(draft)
    if draft.candidate_kind is QuestionerCandidateKind.REFUSAL:
        return codes
    question = draft.generated_question
    answer = draft.expected_answer or ""
    if re.search(r"\b(?:T\d{3}|[MISCOQ]\d{2,3})\b", f"{question} {answer}"):
        codes.append(CandidateRejectionCode.INTERNAL_IDENTIFIER_LEAK)
    assistant_question = bool(
        re.search(
            r"\b(?:assistant|you)\b.*\b(?:suggest|recommend|advise|say|said|ask|think|thought|feel|felt|sentiment)",
            question,
            re.I,
        )
    )
    for claim in draft.exact_claims:
        if (
            claim.support.speaker == "assistant"
            and not assistant_question
            and re.search(r"\b(?:I|my|user|your)\b", question, re.I)
        ):
            codes.append(CandidateRejectionCode.EVIDENCE_NOT_ENTAILED)
        if claim.subject == "user" and claim.support.speaker == "assistant":
            codes.append(CandidateRejectionCode.EVIDENCE_NOT_ENTAILED)
        if claim.assertion == "advice" and not assistant_question:
            codes.append(CandidateRejectionCode.EVIDENCE_NOT_ENTAILED)
        if claim.subject == "assistant" and not assistant_question:
            codes.append(CandidateRejectionCode.EVIDENCE_NOT_ENTAILED)
        if claim.support.exact_text:
            words = set(normalize_question_text(claim.claim_text).split()) - {
                "the",
                "a",
                "an",
                "is",
                "was",
                "user",
                "assistant",
                "and",
                "it",
                "to",
                "of",
            }
            if words and not words.intersection(
                normalize_question_text(claim.support.exact_text).split()
            ):
                codes.append(CandidateRejectionCode.EVIDENCE_NOT_ENTAILED)
    if re.match(r"^why\b", question, re.I) and not re.search(
        r"\b(?:because|due to|in order to|so that|reason|to (?:learn|understand|avoid|improve))\b",
        answer,
        re.I,
    ):
        codes.append(CandidateRejectionCode.QUESTION_ANSWER_CLOSURE_FAILED)
    if re.match(r"^(?:which|what)\b", question, re.I) and re.fullmatch(
        r"(?:the |a )?(?:full[- ]sized|bigger|smaller|same|other|first|second)?"
        r"\s*(?:one|ones|it|that)[.!]?",
        answer.strip(),
        re.I,
    ):
        codes.append(CandidateRejectionCode.QUESTION_ANSWER_CLOSURE_FAILED)
    if re.search(r"\b(?:that's the spirit|looking forward to it)\b", answer, re.I):
        codes.append(CandidateRejectionCode.QUESTION_ANSWER_CLOSURE_FAILED)
    return list(dict.fromkeys(codes))


def questioner_verifier_v7_output_parser(*, expected_alias: str, candidate):
    from types import SimpleNamespace

    from pmmc.compiler.questioner import questioner_verifier_v6_output_parser
    from pmmc.models.execution import ParsedModelOutput

    def parse(result):
        try:
            value = QuestionerVerificationEnvelopeV7.model_validate(json.loads(result.text)).result
        except (json.JSONDecodeError, ValidationError) as exc:
            raise NativeVerifierOutputError("verifier_output_schema_invalid") from exc
        if value.candidate_alias != expected_alias:
            raise NativeVerifierOutputError("verifier_scope_mismatch")
        expected = [
            (index, claim.support_alias) for index, claim in enumerate(candidate.exact_claims)
        ]
        if [(claim.claim_index, claim.support_alias) for claim in value.claims] != expected:
            raise NativeVerifierOutputError("verifier_claim_scope_mismatch")
        required, closure_required = verifier_required_dimensions(candidate)
        if any(getattr(value.dimensions, name).value == "not_applicable" for name in required):
            raise NativeVerifierOutputError("verifier_required_dimension_missing")
        if any(getattr(value, name).value == "not_applicable" for name in closure_required):
            raise NativeVerifierOutputError("verifier_required_dimension_missing")
        temporal = candidate.reasoning_type in {
            QuestionReasoningType.UPDATE,
            QuestionReasoningType.TEMPORAL,
            QuestionReasoningType.CONFLICT,
        }
        count = candidate.answer_type is QuestionerAnswerType.COUNT or candidate.image_set_scope
        support_pass = all(
            claim.span_exists and claim.claim_entailed and claim.speaker_subject_aligned
            for claim in value.claims
        )
        closed = value.answer_question_closed and (
            not temporal or value.same_entity_attribute.value == "true"
        )
        closed = closed and (not count or value.explicit_count_set.value == "true")
        dimensions = value.dimensions.model_dump(mode="json")
        if not support_pass:
            dimensions["answer_explicitly_entailed"] = "false"
        if not closed:
            dimensions["question_answer_compatible"] = "false"
        normalized = questioner_verifier_v6_output_parser(
            expected_alias=expected_alias, candidate=candidate
        )(
            SimpleNamespace(
                text=json.dumps(
                    {"result": {"candidate_alias": expected_alias, "dimensions": dimensions}}
                )
            )
        )
        return ParsedModelOutput(
            value={
                **normalized.value,
                "_parser_metadata": {
                    "support_contract_version": "questioner-exact-support-v1",
                    "support_assessment": value.model_dump(mode="json"),
                },
            },
            repair_attempt_count=0,
            initial_error=None,
        )

    return parse
