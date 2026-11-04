"""Versioned multimodal Questioner prompt and renderer."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from pmmc.compiler.opportunities import (
    QUESTIONER_OPPORTUNITY_BINDING_VERSION,
    BuiltQuestionOpportunitySet,
    local_opportunity_id,
)
from pmmc.compiler.questioner_config import EVIDENCE_FIRST_VERSION
from pmmc.schemas import (
    GenerationRequest,
    ImageRef,
    MemoryUnit,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    QuestionerBatch,
    QuestionerBatchV2,
    QuestionerBatchV3,
    QuestionerBatchV32,
    QuestionerBatchV33,
    QuestionerBatchV34,
    QuestionerBatchV35,
    QuestionerCandidateDraftV3,
    QuestionerRevisionEnvelopeV1,
    QuestionerRevisionEnvelopeV2,
    QuestionerVerificationBatchV1,
    QuestionerVerificationBatchV2,
    QuestionerVerificationEnvelopeV3,
    QuestionerVerificationEnvelopeV5,
    QuestionerVerificationEnvelopeV6,
    SessionRecord,
    SpeakerRole,
)
from pmmc.schemas.questioner import QuestionerBatchV36, QuestionerVerificationEnvelopeV7

QUESTIONER_V36_PROMPT = """You are Questioner v3.6. Use only the supplied memory-side
opportunities, never benchmark queries or answers. Return fewer than eight questions when
support is insufficient, and at most one candidate for each allocated opportunity. The
allocation order is a query-blind type-coverage budget. Do not replace temporal, update,
multi-hop or image-set needs with additional local preferences. Every question must be
self-contained. Return typed claims using the exact allowed span/image aliases; do not
return paraphrases, quotes, invented aliases, taxonomy or reasoning. Multiple spans in
one memory are valid but do not constitute independent hops. Distinguish speaker from
subject: assistant advice is not a user action or preference. Explicit user plans can
support questions about plans, never already-completed events. Why requires an explicit
reason, which requires an identifiable entity, count requires a complete explicit set,
and updates require the same entity and attribute. An image_set_scope opportunity requires
a list answer and a claim for every supplied image, including explicitly excluded members;
identify images by inspectable descriptions, not aliases in the question or answer text.
Empty control_candidates is mandatory
unless an explicit refusal opportunity is supplied. Output only the schema JSON."""

EVIDENCE_FIRST_INSTRUCTION = (
    "Start with the exact cited support, then ask a distinct question that these sources "
    "and the indicated operators could answer. Respect the answer types, evidence groups "
    "and allocation order. Image-text binding needs both a text span and an image. "
    "Skip unsupported or redundant needs; do not fill slots with generic questions "
    "about what is in an image."
)

QUESTIONER_VERIFIER_V7_PROMPT = """Verify exactly one Questioner v3.6 candidate.
For EVERY claim assess span_exists, claim_entailed and speaker_subject_aligned using ONLY
its exact cited text range or supplied image. An existing span is not proof of entailment.
Never borrow a neighboring sentence or another claim's support to repair a wrong citation.
Speaker is the source owner, subject is who the assertion describes. Assistant advice or
assistant sentiments are not user facts. Explicit user plans are valid as plans, not as
completed events. A caption is not raw-image evidence. Verify answer_question_closed:
why must give a cause, which a self-contained identifiable entity (not 'the full-sized
one'), count an explicit complete set, temporal/update/conflict the same entity/attribute.
'Recently fascinated' does not explain why; 'That is the spirit' does not answer a
preference question. Use not_applicable for same_entity_attribute only outside temporal,
update/conflict; explicit_count_set only outside count and image_set_scope. Image-set closure
requires that every supplied image was inspected and included/excluded consistently with
the question, without claiming coverage outside the scan. Return every claim in input order
with the exact claim_index/support_alias. All fields in required_boolean_dimensions and
required_boolean_closure_fields need an explicit true or false, NEVER not_applicable.
False means unsupported or invalid; uncertainty does not make a required check inapplicable.
user_facing: a natural user information need; future_plausible: useful beyond this chat;
not_historical_replay: not merely reciting the chat; answer_addresses_question and
question_answer_compatible: the answer resolves the question; answer_explicitly_entailed
and evidence_sufficient: the cited support establishes the entire answer;
speaker_subject_aligned: no speaker/subject attribution error; taxonomy_compatible and
answer_type_correct: the reasoning and answer type fit; metadata_free: no internal IDs;
source_naturalness: not an artificial question about the data format. For image needs,
visual_necessary assesses whether the image is necessary; multi_hop_genuine assesses
whether multiple independent sources are truly required. Do not emit actions,
replacements or reasoning."""

QUESTIONER_PROMPT_VERSION = "pmmc-questioner-v1"
QUESTIONER_RENDERER_VERSION = "questioner-multimodal-renderer-v1"
QUESTIONER_V2_PROMPT_VERSION = "pmmc-questioner-v2"
QUESTIONER_V2_RENDERER_VERSION = "questioner-multimodal-renderer-v2"
QUESTIONER_V3_PROMPT_VERSION = "pmmc-questioner-v3"
QUESTIONER_V3_RENDERER_VERSION = "questioner-multimodal-renderer-v3"
QUESTIONER_VERIFIER_PROMPT_VERSION = "pmmc-questioner-verifier-v1"
QUESTIONER_V31_PROMPT_VERSION = "pmmc-questioner-v3.1"
QUESTIONER_V31_RENDERER_VERSION = "questioner-safe-alias-renderer-v3.1"
QUESTIONER_VERIFIER_V2_PROMPT_VERSION = "pmmc-questioner-verifier-v2"
QUESTIONER_V32_PROMPT_VERSION = "pmmc-questioner-v3.2"
QUESTIONER_V32_RENDERER_VERSION = "questioner-safe-alias-renderer-v3.2"
QUESTIONER_VERIFIER_V3_PROMPT_VERSION = "pmmc-questioner-verifier-v3"
QUESTIONER_V33_PROMPT_VERSION = "pmmc-questioner-v3.3"
QUESTIONER_V33_RENDERER_VERSION = "questioner-safe-alias-renderer-v3.3"
QUESTIONER_VERIFIER_V4_PROMPT_VERSION = "pmmc-questioner-verifier-v4"
QUESTIONER_V34_PROMPT_VERSION = "pmmc-questioner-v3.4"
QUESTIONER_V34_RENDERER_VERSION = "questioner-safe-alias-renderer-v3.4"
QUESTIONER_VERIFIER_V5_PROMPT_VERSION = "pmmc-questioner-verifier-v5"
QUESTIONER_V35_PROMPT_VERSION = "pmmc-questioner-v3.5"
QUESTIONER_V35_RENDERER_VERSION = "questioner-safe-alias-renderer-v3.5"
QUESTIONER_VERIFIER_V6_PROMPT_VERSION = "pmmc-questioner-verifier-v6"
QUESTIONER_REVISION_V1_PROMPT_VERSION = "pmmc-questioner-reviser-v1"
QUESTIONER_REVISION_V2_PROMPT_VERSION = "pmmc-questioner-reviser-v2"
QUESTIONER_REQUEST_CONSTRUCTION_VERSION = "questioner-request-construction-v2"
QUESTIONER_PROMPT = """You are the write-time Questioner for a long-term multimodal memory system.
Given only the visible compilation unit below, predict up to eight distinct questions that a
user may ask in the future. Generate fewer questions, including an empty list, when the
evidence does not support useful candidates.

Use only these question types: textual_fact, visual_detail, image_text_binding,
visual_search, multi_hop, temporal, knowledge_update, conflict_detection,
answer_refusal, personal_preference.
Every non-refusal candidate must include a concise expected answer and typed expected
evidence using only memory, image, session, or caption IDs explicitly shown in the input.
Visual candidates must reference image evidence. Multi-hop candidates must use at least
two distinct evidence items. Refusal candidates must not invent an expected answer.
Do not reveal chain-of-thought. Return only the requested JSON object and never invent
identifiers."""

QUESTIONER_V2_PROMPT = """You are the write-time Questioner for a long-term
multimodal memory system.
Using only the visible compilation unit, return a JSON object with a candidates array of zero
to eight distinct future questions. Generate fewer candidates, including an empty array, when
the evidence is insufficient. Never invent an identifier.

Allowed question_type values are: textual_fact, visual_detail, image_text_binding,
visual_search, multi_hop, temporal, knowledge_update, conflict_detection, answer_refusal,
personal_preference. Allowed reasoning_type values are: single_hop, multi_hop, temporal,
update, conflict, refusal. required_modalities may contain only text and image.

Each expected_evidence item has exactly two fields: evidence_id and evidence_type. The
evidence_type must be memory, image, session, or caption, and evidence_id must be an ID
shown in the compilation unit. Every non-refusal candidate needs a concise expected_answer
and evidence. A visual candidate needs image evidence. A multi-hop candidate needs at least
two distinct evidence items. An answer_refusal candidate must not invent an answer.

Do not include chain-of-thought, rationale, run metadata, unknown fields, or prose outside the
JSON object."""

QUESTIONER_V3_PROMPT = """You are the write-time Questioner for a long-term
multimodal memory system. Use only the visible memories, images, and bounded opportunity
catalog. Return zero to eight future-facing, user-facing questions. It is correct to return
fewer candidates or an empty array when the evidence does not support useful questions.

Every candidate must select exactly one listed opportunity and may cite only that
opportunity's local evidence aliases. Never emit canonical memory, session, turn, round,
compiler, run, or database identifiers. Do not mention local aliases in the question,
expected answer, or paraphrases.

Questions must be naturally useful after the recorded events. Do not repeat a question that
already appears in the history. Do not generate assistant follow-ups such as "Would you like
me to...", generic external-knowledge questions, system/order/current-time questions, or
questions whose answer is a plan, placeholder, or unsupported guess. Visual questions must
genuinely require the raw image. Multi-hop questions must need at least two independent
evidence groups. Do not generate candidates merely to fill the budget.

Use only the ten declared question types and six declared reasoning types. Every JSON field
is required. An answerable candidate needs a concise factual expected_answer and evidence.
A refusal candidate must use answer_refusal/refusal, expected_answer=null, and no evidence.
Return only the strict JSON object. Do not reveal chain-of-thought or add unknown fields."""

QUESTIONER_VERIFIER_PROMPT = """You are the independent semantic verifier for generated
long-term-memory questions. Judge each candidate only against its selected opportunity and
the visible evidence. Return one result for every candidate alias in the same order.

Use pass only when the question is natural, future-plausible, answerable, correctly typed,
and its expected answer is fully supported by the cited evidence. A visual question must
need the image; a multi-hop question must need multiple independent evidence groups. Reject
internal metadata, local aliases in user-facing text, historical question replay, assistant
follow-ups, generic external knowledge, placeholders, unsupported answers, and leakage.

Use revise only when one bounded replacement can fix the candidate without changing its
opportunity or using new evidence. The replacement must obey the full Questioner v3 schema.
On the final verification round, revise is not allowed and must be returned as reject. Return
only reason codes and the strict JSON object, never chain-of-thought or free-form rationale."""

QUESTIONER_V31_PROMPT = """You are the write-time Questioner for a prospective multimodal
memory compiler. Use only the visible local aliases and bounded opportunities. Return zero
to eight useful future-facing questions; returning fewer or none is preferred to guessing.

Personal state, preference, ownership, choice, and completed action require explicit user
evidence. An assistant question does not establish the user's answer. Ask what the assistant
said only when the cited assistant statement explicitly contains the answer. Dates, counts,
and order must appear in visible content and must never be inferred from hidden metadata.
Update and conflict require the same entity and attribute with explicit changed or incompatible
values. Multi-hop requires two semantically related, independent evidence groups. Visual
claims must be directly visible in the cited image.

Never replay a question from the history, expose aliases or canonical identifiers, ask generic
external-knowledge questions, emit placeholders, or invent an answer. Select exactly one
listed opportunity and only its evidence aliases. Every field is required. A refusal has a null
answer and no evidence. Return strict JSON only, without chain-of-thought or extra fields."""

QUESTIONER_VERIFIER_V2_PROMPT = """Verify each future-memory candidate only against its
selected opportunity and cited evidence. For visual candidates inspect the supplied images.
Return every required boolean dimension: user_facing, future_plausible,
not_historical_replay, answer_explicitly_entailed, evidence_sufficient, taxonomy_correct,
modality_correct, visual_necessary, multi_hop_genuine, and metadata_free.

The server derives pass only when every applicable dimension passes. Use revise only for
wording, taxonomy, reasoning type, modality, or paraphrase corrections; a replacement must
keep the exact opportunity, expected answer, and evidence aliases. Reject unsupported
answers, false visual claims, missing evidence, assistant follow-ups, and metadata-derived
facts. Return strict JSON and reason codes only, never chain-of-thought."""

QUESTIONER_V32_PROMPT = """You are the write-time Questioner for a prospective multimodal
memory compiler. Use only the visible local aliases and one bounded opportunity per
candidate. Return zero to eight useful, natural questions that a user could plausibly ask
later. Returning fewer or none is correct when explicit evidence is insufficient.

The server derives required modalities, hop count, typed evidence, source memories, and
evidence groups. You must output only: candidate_kind, opportunity_id, generated_question,
question_type, reasoning_type, expected_answer, evidence_aliases, paraphrases, and
confidence. Cite only aliases listed by the selected opportunity.

User state, preference, ownership, choice, and completed actions require explicit user
statements. An assistant question never establishes a user answer. An assistant statement
supports only what the assistant explicitly said or recommended. Dates, counts, and order
must be stated in visible memory content. Update or conflict requires the same entity and
attribute with explicit changed or incompatible values. Multi-hop questions require
semantically related evidence from independent groups, not mere adjacency or lexical
similarity. Visual claims must be directly visible in the cited image.

Never replay a historical question, ask generic external-knowledge questions, expose local
aliases or system metadata in user-facing text, emit placeholders, or invent an answer.
Refusal questions must still sound like natural user questions; they use answer_refusal and
refusal, have expected_answer=null, and cite no evidence. Never mention harness concepts
such as local aliases, bounded opportunities, inability to generate, compiler state, or
internal validation. Return strict JSON only without chain-of-thought or extra fields."""

QUESTIONER_VERIFIER_V3_PROMPT = """Verify exactly one future-memory candidate against only
the supplied opportunity and its cited evidence. No other candidate or evidence is in scope.
For visual candidates inspect the supplied cited images.

Return exactly the ten required boolean dimensions and fixed reason codes:
user_facing, future_plausible, not_historical_replay, answer_explicitly_entailed,
evidence_sufficient, taxonomy_correct, modality_correct, visual_necessary,
multi_hop_genuine, and metadata_free. A non-applicable visual or multi-hop dimension is
true. Emit exactly one matching reason for every false dimension and no other reasons:
not_user_facing, not_future_plausible, historical_replay, answer_unsupported,
evidence_insufficient, taxonomy_mismatch, modality_mismatch, visual_not_needed,
multi_hop_not_genuine, or internal_metadata. Do not choose pass, revise, or reject and do
not return a replacement. The server derives the action. Return strict JSON only, without
rationale or chain-of-thought."""

QUESTIONER_REVISION_V1_PROMPT = """Revise exactly one future-memory candidate only when its
wording, taxonomy, reasoning type, or paraphrases are incorrect. Preserve the selected
opportunity, expected answer, and evidence exactly; those immutable values are not part of
your output. Return candidate_alias, generated_question, question_type, reasoning_type, and
paraphrases only. Do not add evidence, answer text, local aliases in user-facing text,
metadata, rationale, or chain-of-thought. Return strict JSON only."""

QUESTIONER_REVISION_V2_PROMPT = """Revise exactly one future-memory candidate's wording.
Preserve its information need, opportunity, answer claims, support, taxonomy, reasoning type,
and all evidence exactly. Return only candidate_alias, generated_question, and paraphrases.
Do not output answers, evidence, aliases in user-facing text, taxonomy, reasoning type,
metadata, rationale, or chain-of-thought. Return strict JSON only."""

QUESTIONER_V33_PROMPT = """You are Questioner v3.3 for a prospective multimodal memory
compiler. Generate zero to eight answerable future questions in candidates and at most one
natural refusal/control question in control_candidates. A control candidate never consumes
the answerable budget. Returning empty arrays is correct when explicit support is absent.

Select one bounded opportunity per candidate. Never expose local aliases, canonical IDs,
round/session ordering, metadata, or harness language in user-facing text. Assistant
questions do not establish a user answer. Future plans, generic knowledge, inferred user
state, and metadata dates are not answer claims.

For each answerable candidate, output one or more answer_claims. A text claim cites one
allowed evidence alias and includes an exact support_quote of at most 32 whitespace tokens
from that evidence. A visual claim cites an actual image alias and has support_quote=null.
The server verifies every quote/alias and deterministically assembles expected_answer and
expected evidence. Multi-hop claims must cover at least two related independent evidence
groups. Never output expected_answer, canonical evidence, modality, hop count, rationale,
or chain-of-thought. Return only the strict JSON envelope."""

QUESTIONER_VERIFIER_V4_PROMPT = """Verify exactly one Questioner v3.3 candidate against
only its selected opportunity, validated answer claims, cited evidence, and cited images.
Check natural future plausibility, exact answer entailment, speaker scope, visual necessity,
genuine independent multi-hop support, and absence of metadata or harness leakage.

Return exactly the ten required boolean dimensions and fixed reason codes used by the
schema. The service derives pass, revise, or reject. Do not inspect or refer to any other
candidate, benchmark query, gold answer, locator, rationale, or chain-of-thought."""

QUESTIONER_V34_PROMPT = """You are Questioner v3.4 for a prospective multimodal memory
compiler. Generate zero to eight answerable future questions and at most one natural
control question. Select exactly one bounded opportunity. A text answer claim cites one
rendered sentence-span alias; a visual claim cites one real image alias. The server derives
taxonomy, reasoning type, modality, evidence, hop count, and expected answer.

Questions must be natural, future-plausible, and supported by explicit assertions. An
assistant question is not a user answer. Never infer ownership, choice, completion, dates,
or visual facts. Never expose aliases, metadata, ordering, harness terms, or canonical IDs.
Do not repeat the canonical question in paraphrases. Return fewer candidates when support
is weak. Return only the strict JSON envelope without rationale or chain-of-thought."""

QUESTIONER_VERIFIER_V5_PROMPT = """Verify exactly one Questioner v3.4 candidate using
only its selected opportunity, validated claims, cited sentence spans, and cited images.
Return only the eleven dimensions in the strict schema. Each value is true, false, or
not_applicable. Use not_applicable only when a dimension genuinely does not apply. Do not
emit reason codes, verdicts, actions, replacements, rationale, or chain-of-thought. The
server derives applicability, reason codes, and pass/revise/reject."""

QUESTIONER_V35_PROMPT = """You are Questioner v3.5 for a prospective multimodal memory
compiler. Generate zero to eight answerable future questions and at most one natural
control question. Select one bounded opportunity. Every text claim cites one rendered
sentence-span alias and every visual claim cites one real image alias. Every claim also
declares exactly one answer_type: entity, value, count, list, boolean, event, or refusal.

The question and answer claim must form a natural closed pair. Never use a question,
disclaimer, placeholder, assistant follow-up, metadata date, incomplete phrase, or harness
language as an answer. Do not infer user ownership, choice, completion, time, or visual
facts. The server derives taxonomy, modality, evidence groups, hop count, and expected
answer. Return only the strict JSON envelope without rationale or hidden reasoning."""

QUESTIONER_VERIFIER_V6_PROMPT = """Verify exactly one Questioner v3.5 candidate using
only its selected opportunity, validated answer claims, cited sentence spans, and cited
images. Return only the fourteen tri-state dimensions in the strict schema. Check that the
question and answer close, the answer type is correct, speaker/subject scope is aligned,
and source wording is natural. Do not emit reasons, actions, replacements, or reasoning."""

_PROMPTS = {
    "pmmc-questioner-v3.6": QUESTIONER_V36_PROMPT,
    "pmmc-questioner-verifier-v7": QUESTIONER_VERIFIER_V7_PROMPT,
    QUESTIONER_PROMPT_VERSION: QUESTIONER_PROMPT,
    QUESTIONER_V2_PROMPT_VERSION: QUESTIONER_V2_PROMPT,
    QUESTIONER_V3_PROMPT_VERSION: QUESTIONER_V3_PROMPT,
    QUESTIONER_VERIFIER_PROMPT_VERSION: QUESTIONER_VERIFIER_PROMPT,
    QUESTIONER_V31_PROMPT_VERSION: QUESTIONER_V31_PROMPT,
    QUESTIONER_VERIFIER_V2_PROMPT_VERSION: QUESTIONER_VERIFIER_V2_PROMPT,
    QUESTIONER_V32_PROMPT_VERSION: QUESTIONER_V32_PROMPT,
    QUESTIONER_VERIFIER_V3_PROMPT_VERSION: QUESTIONER_VERIFIER_V3_PROMPT,
    QUESTIONER_V33_PROMPT_VERSION: QUESTIONER_V33_PROMPT,
    QUESTIONER_VERIFIER_V4_PROMPT_VERSION: QUESTIONER_VERIFIER_V4_PROMPT,
    QUESTIONER_V34_PROMPT_VERSION: QUESTIONER_V34_PROMPT,
    QUESTIONER_VERIFIER_V5_PROMPT_VERSION: QUESTIONER_VERIFIER_V5_PROMPT,
    QUESTIONER_V35_PROMPT_VERSION: QUESTIONER_V35_PROMPT,
    QUESTIONER_VERIFIER_V6_PROMPT_VERSION: QUESTIONER_VERIFIER_V6_PROMPT,
    QUESTIONER_REVISION_V1_PROMPT_VERSION: QUESTIONER_REVISION_V1_PROMPT,
    QUESTIONER_REVISION_V2_PROMPT_VERSION: QUESTIONER_REVISION_V2_PROMPT,
}


def questioner_prompt_text(prompt_version: str = QUESTIONER_PROMPT_VERSION) -> str:
    try:
        return _PROMPTS[prompt_version]
    except KeyError as exc:
        raise ValueError(f"unsupported Questioner prompt version: {prompt_version}") from exc


def questioner_prompt_sha256(
    prompt_version: str = QUESTIONER_PROMPT_VERSION,
) -> str:
    return hashlib.sha256(questioner_prompt_text(prompt_version).encode("utf-8")).hexdigest()


def questioner_response_schema(prompt_version: str) -> dict[str, object]:
    if prompt_version == "pmmc-questioner-v3.6":
        return QuestionerBatchV36.model_json_schema()
    if prompt_version == "pmmc-questioner-verifier-v7":
        return QuestionerVerificationEnvelopeV7.model_json_schema()
    if prompt_version == QUESTIONER_PROMPT_VERSION:
        return QuestionerBatch.model_json_schema()
    if prompt_version == QUESTIONER_V2_PROMPT_VERSION:
        return QuestionerBatchV2.model_json_schema()
    if prompt_version == QUESTIONER_V3_PROMPT_VERSION:
        return QuestionerBatchV3.model_json_schema()
    if prompt_version == QUESTIONER_V31_PROMPT_VERSION:
        return QuestionerBatchV3.model_json_schema()
    if prompt_version == QUESTIONER_V32_PROMPT_VERSION:
        return QuestionerBatchV32.model_json_schema()
    if prompt_version == QUESTIONER_V33_PROMPT_VERSION:
        return QuestionerBatchV33.model_json_schema()
    if prompt_version == QUESTIONER_V34_PROMPT_VERSION:
        return QuestionerBatchV34.model_json_schema()
    if prompt_version == QUESTIONER_V35_PROMPT_VERSION:
        return QuestionerBatchV35.model_json_schema()
    if prompt_version == QUESTIONER_VERIFIER_PROMPT_VERSION:
        return QuestionerVerificationBatchV1.model_json_schema()
    if prompt_version == QUESTIONER_VERIFIER_V2_PROMPT_VERSION:
        return QuestionerVerificationBatchV2.model_json_schema()
    if prompt_version == QUESTIONER_VERIFIER_V3_PROMPT_VERSION:
        return QuestionerVerificationEnvelopeV3.model_json_schema()
    if prompt_version == QUESTIONER_VERIFIER_V4_PROMPT_VERSION:
        return QuestionerVerificationEnvelopeV3.model_json_schema()
    if prompt_version == QUESTIONER_VERIFIER_V5_PROMPT_VERSION:
        return QuestionerVerificationEnvelopeV5.model_json_schema()
    if prompt_version == QUESTIONER_VERIFIER_V6_PROMPT_VERSION:
        return QuestionerVerificationEnvelopeV6.model_json_schema()
    if prompt_version == QUESTIONER_REVISION_V1_PROMPT_VERSION:
        return QuestionerRevisionEnvelopeV1.model_json_schema()
    if prompt_version == QUESTIONER_REVISION_V2_PROMPT_VERSION:
        return QuestionerRevisionEnvelopeV2.model_json_schema()
    raise ValueError(f"unsupported Questioner prompt version: {prompt_version}")


def questioner_response_schema_sha256(prompt_version: str) -> str:
    payload = json.dumps(
        questioner_response_schema(prompt_version),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def questioner_request_contract_hash(
    *,
    prompt_version: str,
    renderer_version: str,
    opportunity_builder_version: str | None = None,
) -> str:
    """Hash every code-level contract that can change prepared request bytes."""

    payload = {
        "request_construction_version": QUESTIONER_REQUEST_CONSTRUCTION_VERSION,
        "prompt_version": prompt_version,
        "prompt_sha256": questioner_prompt_sha256(prompt_version),
        "renderer_version": renderer_version,
        "response_schema_sha256": questioner_response_schema_sha256(prompt_version),
        "opportunity_builder_version": opportunity_builder_version,
        "opportunity_binding_version": (
            QUESTIONER_OPPORTUNITY_BINDING_VERSION
            if opportunity_builder_version is not None
            else None
        ),
    }
    if opportunity_builder_version == EVIDENCE_FIRST_VERSION:
        payload["allocation_instruction_sha256"] = hashlib.sha256(
            EVIDENCE_FIRST_INSTRUCTION.encode("utf-8")
        ).hexdigest()
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _memory_images(memory: MemoryUnit) -> list[ImageRef]:
    refs = {item.image_id: item for item in memory.image_refs}
    ids = list(dict.fromkeys([*memory.image_ids, *(item.image_id for item in memory.image_refs)]))
    return [refs.get(image_id, ImageRef(image_id=image_id)) for image_id in ids]


def render_questioner_request(
    *,
    request_id: str,
    model_alias: str,
    unit_id: str,
    cutoff_session_id: str,
    memories: list[MemoryUnit],
    sessions: list[SessionRecord],
    max_output_tokens: int,
    timeout_seconds: float | None = None,
    prompt_version: str = QUESTIONER_PROMPT_VERSION,
    renderer_version: str = QUESTIONER_RENDERER_VERSION,
    opportunity_set: BuiltQuestionOpportunitySet | None = None,
    memory_raw_image_policy: str = "allow",
) -> GenerationRequest:
    if prompt_version == "pmmc-questioner-v3.6":
        if renderer_version != "questioner-safe-alias-renderer-v3.6" or opportunity_set is None:
            raise ValueError("Questioner v3.6 requires native renderer and opportunities")
        return _render_native_request(
            request_id=request_id,
            model_alias=model_alias,
            unit_id=unit_id,
            memories=memories,
            opportunity_set=opportunity_set,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            memory_raw_image_policy=memory_raw_image_policy,
        )
    if prompt_version == QUESTIONER_PROMPT_VERSION:
        if renderer_version != QUESTIONER_RENDERER_VERSION:
            raise ValueError("Questioner v1 prompt requires the v1 renderer")
        response_schema = questioner_response_schema(prompt_version)
        schema_name = "pmmc_questioner_batch_v1"
    elif prompt_version == QUESTIONER_V2_PROMPT_VERSION:
        if renderer_version != QUESTIONER_V2_RENDERER_VERSION:
            raise ValueError("Questioner v2 prompt requires the v2 renderer")
        response_schema = questioner_response_schema(prompt_version)
        schema_name = "pmmc_questioner_batch_v2"
    elif prompt_version in {
        QUESTIONER_V3_PROMPT_VERSION,
        QUESTIONER_V31_PROMPT_VERSION,
        QUESTIONER_V32_PROMPT_VERSION,
        QUESTIONER_V33_PROMPT_VERSION,
        QUESTIONER_V34_PROMPT_VERSION,
        QUESTIONER_V35_PROMPT_VERSION,
    }:
        expected_renderer = (
            QUESTIONER_V35_RENDERER_VERSION
            if prompt_version == QUESTIONER_V35_PROMPT_VERSION
            else QUESTIONER_V34_RENDERER_VERSION
            if prompt_version == QUESTIONER_V34_PROMPT_VERSION
            else QUESTIONER_V33_RENDERER_VERSION
            if prompt_version == QUESTIONER_V33_PROMPT_VERSION
            else QUESTIONER_V32_RENDERER_VERSION
            if prompt_version == QUESTIONER_V32_PROMPT_VERSION
            else QUESTIONER_V31_RENDERER_VERSION
            if prompt_version == QUESTIONER_V31_PROMPT_VERSION
            else QUESTIONER_V3_RENDERER_VERSION
        )
        if renderer_version != expected_renderer:
            raise ValueError("Questioner v3 prompt requires the v3 renderer")
        if opportunity_set is None:
            raise ValueError("Questioner v3 renderer requires an opportunity set")
        response_schema = questioner_response_schema(prompt_version)
        schema_name = (
            "pmmc_questioner_batch_v35"
            if prompt_version == QUESTIONER_V35_PROMPT_VERSION
            else "pmmc_questioner_batch_v34"
            if prompt_version == QUESTIONER_V34_PROMPT_VERSION
            else "pmmc_questioner_batch_v33"
            if prompt_version == QUESTIONER_V33_PROMPT_VERSION
            else "pmmc_questioner_batch_v32"
            if prompt_version == QUESTIONER_V32_PROMPT_VERSION
            else "pmmc_questioner_batch_v3"
        )
    else:
        raise ValueError(f"unsupported Questioner prompt version: {prompt_version}")
    session_map = {item.session_id: item for item in sessions}
    if len(session_map) != len(sessions):
        raise ValueError("questioner sessions cannot contain duplicate IDs")
    missing = sorted({item.session_id for item in memories} - set(session_map))
    if missing:
        raise ValueError(f"questioner memories have missing sessions: {', '.join(missing)}")

    v3 = prompt_version in {
        QUESTIONER_V3_PROMPT_VERSION,
        QUESTIONER_V31_PROMPT_VERSION,
        QUESTIONER_V32_PROMPT_VERSION,
        QUESTIONER_V33_PROMPT_VERSION,
        QUESTIONER_V34_PROMPT_VERSION,
        QUESTIONER_V35_PROMPT_VERSION,
    }
    parts: list[ModelContentPart] = []
    if not v3:
        parts.append(
            ModelContentPart(
                kind=ModelContentKind.TEXT,
                text=f"[COMPILATION_UNIT id={unit_id} cutoff_session={cutoff_session_id}]",
            )
        )
    previous_session: str | None = None
    for memory in memories:
        session = session_map[memory.session_id]
        if memory.session_id != previous_session:
            date = f" date={session.source_date_iso}" if session.source_date_iso else ""
            kind = "profile" if session.is_synthetic else "dialogue"
            if v3:
                assert opportunity_set is not None
                session_alias = opportunity_set.session_aliases[session.session_id]
                context_suffix = (
                    ""
                    if prompt_version
                    in {
                        QUESTIONER_V32_PROMPT_VERSION,
                        QUESTIONER_V33_PROMPT_VERSION,
                        QUESTIONER_V34_PROMPT_VERSION,
                        QUESTIONER_V35_PROMPT_VERSION,
                    }
                    else f"; event date: {session.source_date_iso or 'date not recorded'}"
                )
                parts.append(
                    ModelContentPart(
                        kind=ModelContentKind.TEXT,
                        text=f"[CONTEXT {session_alias}{context_suffix}]",
                    )
                )
            else:
                parts.append(
                    ModelContentPart(
                        kind=ModelContentKind.TEXT,
                        text=(
                            f"[SESSION id={session.session_id} order={session.session_order}"
                            f"{date} type={kind}]"
                        ),
                    )
                )
            previous_session = memory.session_id
        body = memory.text or ""
        memory_label = (
            opportunity_set.memory_aliases[memory.memory_id]
            if v3 and opportunity_set is not None
            else memory.memory_id
        )
        parts.append(
            ModelContentPart(
                kind=ModelContentKind.TEXT,
                text=(
                    f"[MEMORY {memory_label}; speaker={memory.speaker.value}]\n{body}"
                    if v3
                    else f"[MEMORY id={memory.memory_id} turn={memory.turn_id} "
                    f"speaker={memory.speaker.value}]\n{body}"
                ),
            )
        )
        if prompt_version in {QUESTIONER_V34_PROMPT_VERSION, QUESTIONER_V35_PROMPT_VERSION}:
            assert opportunity_set is not None
            for span_alias, (
                parent_alias,
                span_text,
            ) in opportunity_set.sentence_span_aliases.items():
                if parent_alias == memory_label:
                    parts.append(
                        ModelContentPart(
                            kind=ModelContentKind.TEXT,
                            text=(f"[SPAN {span_alias} belongs_to={parent_alias}] {span_text}"),
                        )
                    )
        if memory.caption:
            parts.append(
                ModelContentPart(
                    kind=ModelContentKind.TEXT,
                    text=(
                        f"[CAPTION {opportunity_set.caption_aliases[memory.memory_id]}] "
                        f"{memory.caption}"
                        if v3 and opportunity_set is not None
                        else f"[CAPTION memory_id={memory.memory_id}] {memory.caption}"
                    ),
                )
            )
            if prompt_version in {QUESTIONER_V34_PROMPT_VERSION, QUESTIONER_V35_PROMPT_VERSION}:
                assert opportunity_set is not None
                caption_parent = opportunity_set.caption_aliases[memory.memory_id]
                for span_alias, (
                    parent_alias,
                    span_text,
                ) in opportunity_set.sentence_span_aliases.items():
                    if parent_alias == caption_parent:
                        parts.append(
                            ModelContentPart(
                                kind=ModelContentKind.TEXT,
                                text=(f"[SPAN {span_alias} belongs_to={parent_alias}] {span_text}"),
                            )
                        )
        for image in _memory_images(memory):
            parts.append(
                ModelContentPart(
                    kind=ModelContentKind.TEXT,
                    text=(
                        "[IMAGE "
                        f"{opportunity_set.image_aliases[(memory.memory_id, image.image_id)]} "
                        f"belongs_to={memory_label}]"
                        if v3 and opportunity_set is not None
                        else f"[IMAGE id={image.image_id} memory_id={memory.memory_id}]"
                    ),
                )
            )
            parts.append(ModelContentPart(kind=ModelContentKind.IMAGE, image=image))
    if v3:
        assert opportunity_set is not None
        opportunity_rows = [
            {
                "opportunity_id": local_opportunity_id(item),
                "allowed_question_types": [value.value for value in item.allowed_question_types],
                "allowed_reasoning_types": [value.value for value in item.allowed_reasoning_types],
                "required_modalities": [value.value for value in item.required_modalities],
                "minimum_evidence_groups": item.minimum_evidence_groups,
                "evidence_aliases": item.evidence_aliases,
                **(
                    {
                        "answer_support_aliases": sorted(
                            {
                                *(
                                    alias
                                    for alias, (parent, _text) in (
                                        opportunity_set.sentence_span_aliases.items()
                                    )
                                    if parent in item.evidence_aliases
                                ),
                                *(
                                    alias
                                    for alias in item.evidence_aliases
                                    if alias.startswith("I")
                                ),
                            }
                        )
                    }
                    if prompt_version
                    in {
                        QUESTIONER_V34_PROMPT_VERSION,
                        QUESTIONER_V35_PROMPT_VERSION,
                    }
                    else {}
                ),
            }
            for item in opportunity_set.opportunities
        ]
        parts.append(
            ModelContentPart(
                kind=ModelContentKind.TEXT,
                text="[OPPORTUNITIES]\n"
                + json.dumps(opportunity_rows, ensure_ascii=True, separators=(",", ":"))
                + "\n[/OPPORTUNITIES]",
            )
        )
    else:
        parts.append(ModelContentPart(kind=ModelContentKind.TEXT, text="[/COMPILATION_UNIT]"))
    return GenerationRequest(
        request_id=request_id,
        model_alias=model_alias,
        messages=[
            ModelMessage(
                role=SpeakerRole.SYSTEM,
                parts=[
                    ModelContentPart(
                        kind=ModelContentKind.TEXT,
                        text=questioner_prompt_text(prompt_version),
                    )
                ],
            ),
            ModelMessage(role=SpeakerRole.USER, parts=parts),
        ],
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        timeout_seconds=timeout_seconds,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_schema,
            },
        },
        metadata={
            "task": "pmmc_questioner",
            "prompt_version": prompt_version,
            "renderer_version": renderer_version,
            **(
                {"response_schema_hash": questioner_response_schema_sha256(prompt_version)}
                if prompt_version
                in {
                    QUESTIONER_V2_PROMPT_VERSION,
                    QUESTIONER_V3_PROMPT_VERSION,
                    QUESTIONER_V31_PROMPT_VERSION,
                    QUESTIONER_V32_PROMPT_VERSION,
                    QUESTIONER_V33_PROMPT_VERSION,
                    QUESTIONER_V34_PROMPT_VERSION,
                    QUESTIONER_V35_PROMPT_VERSION,
                }
                else {}
            ),
            "compiler_unit_id": unit_id,
        },
    )


def _render_native_request(
    *,
    request_id,
    model_alias,
    unit_id,
    memories,
    opportunity_set,
    max_output_tokens,
    timeout_seconds,
    candidate=None,
    memory_raw_image_policy="allow",
) -> GenerationRequest:
    from pmmc.compiler.ids import canonical_content_hash
    from pmmc.compiler.questioner_native import (
        allocate_opportunities,
        native_response_schema,
        native_support_catalog,
        opportunity_support_aliases,
        validate_w_verifier_candidate,
        verifier_required_dimensions,
    )
    from pmmc.schemas.questioner import QuestionerVerifierCandidateV36

    evidence_first = opportunity_set.opportunity_set.builder_version == EVIDENCE_FIRST_VERSION
    if evidence_first and opportunity_set.opportunity_set.compiler_unit_id != unit_id:
        raise ValueError("evidence_first_catalog_unit_mismatch")
    catalog = native_support_catalog(opportunity_set, memories)
    if memory_raw_image_policy == "forbid" and any(
        support.source_field == "image" for support in catalog.values()
    ):
        raise ValueError("raw_image_forbidden_in_opportunity_catalog")
    allocated = allocate_opportunities(opportunity_set)
    payload: dict[str, Any]
    schema: dict[str, Any]
    rows = []
    selected = set()
    if candidate is None:
        for opportunity in (
            *allocated,
            *(
                item
                for item in opportunity_set.opportunities
                if item.opportunity_type.value == "refusal_boundary"
            ),
        ):
            aliases = list(opportunity_support_aliases(opportunity_set, opportunity))
            selected.update(aliases)
            rows.append(
                {
                    "opportunity_id": local_opportunity_id(opportunity),
                    "opportunity_type": opportunity.opportunity_type.value,
                    "support_aliases": aliases,
                    "minimum_evidence_groups": opportunity.minimum_evidence_groups,
                    "candidate_budget": 1,
                    "image_set_scope": bool(opportunity.metadata.get("image_set_scope")),
                }
            )
            if evidence_first and aliases:
                rows[-1].update(
                    expected_answer_types=opportunity.metadata["expected_answer_types"],
                    operator_path=opportunity.metadata["operator_path"],
                )
        payload = {"opportunities": rows}
        if evidence_first:
            payload["allocation_policy"] = {
                "version": EVIDENCE_FIRST_VERSION,
                "instruction": EVIDENCE_FIRST_INSTRUCTION,
            }
        version = "pmmc-questioner-v3.6"
        schema = native_response_schema(opportunity_set)
    else:
        alias, draft = candidate
        if not isinstance(draft, QuestionerVerifierCandidateV36):
            raise ValueError("verifier v7 requires native exact claims")
        if evidence_first:
            validate_w_verifier_candidate(draft, opportunity_set, memories, memory_raw_image_policy)
        selected = {claim.support_alias for claim in draft.exact_claims}
        for claim in draft.exact_claims:
            if catalog.get(claim.support_alias) != claim.support:
                raise ValueError("verifier support changed since resolution")
        payload = {
            "candidate_alias": alias,
            "question": draft.generated_question,
            "expected_answer": draft.expected_answer,
            "answer_type": draft.answer_type.value,
            "candidate_kind": draft.candidate_kind.value,
            "reasoning_type": draft.reasoning_type.value,
            "image_set_scope": draft.image_set_scope,
            "claims": [
                {
                    "claim_index": index,
                    "claim_text": claim.claim_text,
                    "claim_kind": claim.claim_kind.value,
                    "subject": claim.subject,
                    "assertion": claim.assertion,
                    "support_alias": claim.support_alias,
                }
                for index, claim in enumerate(draft.exact_claims)
            ],
        }
        version = "pmmc-questioner-verifier-v7"
        schema = cast(dict[str, Any], questioner_response_schema(version))
        schema["$defs"]["QuestionerVerificationItemV7"]["properties"]["candidate_alias"] = {
            "type": "string",
            "const": alias,
        }
        claims_schema = schema["$defs"]["QuestionerVerificationItemV7"]["properties"]["claims"]
        claims_schema.update(minItems=len(draft.exact_claims), maxItems=len(draft.exact_claims))
        required, closure_required = verifier_required_dimensions(draft)
        payload["required_boolean_dimensions"] = list(required)
        payload["required_boolean_closure_fields"] = list(closure_required)
        for name in required:
            schema["$defs"]["QuestionerVerificationDimensionsV6"]["properties"][name] = {
                "type": "string", "enum": ["true", "false"],
            }
        for name in closure_required:
            schema["$defs"]["QuestionerVerificationItemV7"]["properties"][name] = {
                "type": "string", "enum": ["true", "false"],
            }
    payload["exact_support"] = [
        {
            "support_alias": alias,
            "owning_memory": opportunity_set.memory_aliases[support.owning_memory_id],
            "speaker": support.speaker,
            "source_field": support.source_field,
            "start_char": support.start_char,
            "end_char": support.end_char,
            "exact_text": support.exact_text,
        }
        for alias, support in catalog.items()
        if alias in selected
    ]
    if evidence_first and candidate is None:
        group_aliases: dict[str, str] = {}
        for row in payload["exact_support"]:
            parent = catalog[row["support_alias"]].evidence_alias
            group = opportunity_set.alias_to_group[parent]
            row["evidence_group"] = group_aliases.setdefault(
                group, f"G{len(group_aliases) + 1:02d}"
            )
    parts = [
        ModelContentPart(kind=ModelContentKind.TEXT, text=json.dumps(payload, ensure_ascii=True))
    ]
    for alias in sorted(selected):
        support = catalog[alias]
        if support.image_id is None:
            continue
        owner = next(memory for memory in memories if memory.memory_id == support.owning_memory_id)
        image = next(
            (image for image in _memory_images(owner) if image.image_id == support.image_id), None
        )
        if image is None:
            raise ValueError("native image support requires a renderable image")
        parts.extend(
            [
                ModelContentPart(kind=ModelContentKind.TEXT, text=f"[IMAGE {alias}]"),
                ModelContentPart(kind=ModelContentKind.IMAGE, image=image),
            ]
        )
    return GenerationRequest(
        request_id=request_id,
        model_alias=model_alias,
        messages=[
            ModelMessage(
                role=SpeakerRole.SYSTEM,
                parts=[
                    ModelContentPart(
                        kind=ModelContentKind.TEXT, text=questioner_prompt_text(version)
                    )
                ],
            ),
            ModelMessage(role=SpeakerRole.USER, parts=parts),
        ],
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        timeout_seconds=timeout_seconds,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "pmmc_questioner_native_v36"
                if candidate is None
                else "pmmc_questioner_verifier_v7",
                "strict": True,
                "schema": schema,
            },
        },
        metadata={
            "task": "pmmc_questioner" if candidate is None else "pmmc_questioner_verifier",
            "prompt_version": version,
            "renderer_version": "questioner-safe-alias-renderer-v3.6",
            "compiler_unit_id": unit_id,
            "response_schema_hash": canonical_content_hash(schema),
            "allocation_version": (
                EVIDENCE_FIRST_VERSION if evidence_first
                else "questioner-query-blind-type-budget-v1"
            ),
            "allocated_opportunities": [local_opportunity_id(item) for item in allocated],
            "memory_raw_image_policy": memory_raw_image_policy,
        },
    )


def render_questioner_verifier_request(
    *,
    request_id: str,
    model_alias: str,
    unit_id: str,
    cutoff_session_id: str,
    memories: list[MemoryUnit],
    sessions: list[SessionRecord],
    opportunity_set: BuiltQuestionOpportunitySet,
    candidates: list[tuple[str, QuestionerCandidateDraftV3]],
    final_round: bool,
    max_output_tokens: int,
    timeout_seconds: float | None = None,
    questioner_prompt_version: str = QUESTIONER_V3_PROMPT_VERSION,
    questioner_renderer_version: str = QUESTIONER_V3_RENDERER_VERSION,
    verifier_prompt_version: str = QUESTIONER_VERIFIER_PROMPT_VERSION,
    memory_raw_image_policy: str = "allow",
) -> GenerationRequest:
    if verifier_prompt_version == "pmmc-questioner-verifier-v7":
        if len(candidates) != 1:
            raise ValueError("Questioner verifier v7 requires one native candidate")
        return _render_native_request(
            request_id=request_id,
            model_alias=model_alias,
            unit_id=unit_id,
            memories=memories,
            opportunity_set=opportunity_set,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            candidate=candidates[0],
            memory_raw_image_policy=memory_raw_image_policy,
        )
    single_candidate = verifier_prompt_version in {
        QUESTIONER_VERIFIER_V3_PROMPT_VERSION,
        QUESTIONER_VERIFIER_V4_PROMPT_VERSION,
        QUESTIONER_VERIFIER_V5_PROMPT_VERSION,
        QUESTIONER_VERIFIER_V6_PROMPT_VERSION,
    }
    if single_candidate and len(candidates) != 1:
        raise ValueError("Questioner verifier v3 requires exactly one candidate")
    scoped_opportunity_set = opportunity_set
    scoped_memories = memories
    scoped_sessions = sessions
    if verifier_prompt_version in {
        QUESTIONER_VERIFIER_V2_PROMPT_VERSION,
        QUESTIONER_VERIFIER_V3_PROMPT_VERSION,
        QUESTIONER_VERIFIER_V4_PROMPT_VERSION,
        QUESTIONER_VERIFIER_V5_PROMPT_VERSION,
        QUESTIONER_VERIFIER_V6_PROMPT_VERSION,
    }:
        opportunity_ids = {candidate.opportunity_id for _, candidate in candidates}
        aliases = {alias for _, candidate in candidates for alias in candidate.evidence_aliases}
        opportunities = tuple(
            item
            for item in opportunity_set.opportunities
            if local_opportunity_id(item) in opportunity_ids
        )
        if not opportunities:
            raise ValueError("evidence-scoped verifier request has no matching opportunity")
        evidence_refs = tuple(
            item for item in opportunity_set.evidence_refs if item.alias in aliases
        )
        if not evidence_refs and any(
            candidate.candidate_kind.value != "refusal" for _, candidate in candidates
        ):
            raise ValueError("answerable verifier request has no scoped evidence")
        memory_ids = {
            item.evidence.memory_id for item in evidence_refs if item.evidence.memory_id is not None
        }
        owning_memory_ids = {
            owning_memory_id
            for item in evidence_refs
            if isinstance(
                owning_memory_id := item.evidence.metadata.get("owning_memory_id"),
                str,
            )
            and owning_memory_id
        }
        memory_ids.update(owning_memory_ids)
        image_ids_without_owner = {
            item.evidence.image_id
            for item in evidence_refs
            if item.evidence.image_id is not None
            and not isinstance(item.evidence.metadata.get("owning_memory_id"), str)
        }
        scoped_memories = [
            item
            for item in memories
            if item.memory_id in memory_ids
            or bool(image_ids_without_owner.intersection(item.image_ids))
            or any(ref.image_id in image_ids_without_owner for ref in item.image_refs)
        ]
        scoped_session_ids = {item.session_id for item in scoped_memories}
        scoped_sessions = [item for item in sessions if item.session_id in scoped_session_ids]
        scoped_opportunity_set = BuiltQuestionOpportunitySet(
            opportunity_set=opportunity_set.opportunity_set.model_copy(
                update={"opportunity_count": len(opportunities)}
            ),
            opportunities=opportunities,
            evidence_refs=evidence_refs,
            alias_to_evidence={
                key: value
                for key, value in opportunity_set.alias_to_evidence.items()
                if key in aliases
            },
            alias_to_group={
                key: value
                for key, value in opportunity_set.alias_to_group.items()
                if key in aliases
            },
            memory_aliases=opportunity_set.memory_aliases,
            session_aliases=opportunity_set.session_aliases,
            image_aliases=opportunity_set.image_aliases,
            caption_aliases=opportunity_set.caption_aliases,
            sentence_span_aliases=opportunity_set.sentence_span_aliases,
        )
    base = render_questioner_request(
        request_id=request_id,
        model_alias=model_alias,
        unit_id=unit_id,
        cutoff_session_id=cutoff_session_id,
        memories=scoped_memories,
        sessions=scoped_sessions,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        prompt_version=questioner_prompt_version,
        renderer_version=questioner_renderer_version,
        opportunity_set=scoped_opportunity_set,
    )
    candidate_payload = [
        {"candidate_alias": alias, **candidate.model_dump(mode="json")}
        for alias, candidate in candidates
    ]
    user_parts = list(base.messages[1].parts)
    user_parts.append(
        ModelContentPart(
            kind=ModelContentKind.TEXT,
            text="[CANDIDATE_SCOPE final_round="
            + ("true" if final_round else "false")
            + "]\n"
            + json.dumps(
                candidate_payload[0] if single_candidate else candidate_payload,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            + "\n[/CANDIDATE_SCOPE]",
        )
    )
    schema = questioner_response_schema(verifier_prompt_version)
    return base.model_copy(
        update={
            "messages": [
                ModelMessage(
                    role=SpeakerRole.SYSTEM,
                    parts=[
                        ModelContentPart(
                            kind=ModelContentKind.TEXT,
                            text=questioner_prompt_text(verifier_prompt_version),
                        )
                    ],
                ),
                ModelMessage(role=SpeakerRole.USER, parts=user_parts),
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": (
                        "pmmc_questioner_verifier_v6"
                        if verifier_prompt_version == QUESTIONER_VERIFIER_V6_PROMPT_VERSION
                        else "pmmc_questioner_verifier_v5"
                        if verifier_prompt_version == QUESTIONER_VERIFIER_V5_PROMPT_VERSION
                        else "pmmc_questioner_verifier_v4"
                        if verifier_prompt_version == QUESTIONER_VERIFIER_V4_PROMPT_VERSION
                        else "pmmc_questioner_verifier_v3"
                        if verifier_prompt_version == QUESTIONER_VERIFIER_V3_PROMPT_VERSION
                        else "pmmc_questioner_verifier_batch_v2"
                        if verifier_prompt_version == QUESTIONER_VERIFIER_V2_PROMPT_VERSION
                        else "pmmc_questioner_verifier_batch_v1"
                    ),
                    "strict": True,
                    "schema": schema,
                },
            },
            "metadata": {
                **base.metadata,
                "task": "pmmc_questioner_verifier",
                "prompt_version": verifier_prompt_version,
                "response_schema_hash": questioner_response_schema_sha256(verifier_prompt_version),
                "final_round": final_round,
                "candidate_scope": candidate_payload[0]["candidate_alias"]
                if single_candidate
                else "batch",
            },
        }
    )


def render_questioner_revision_request(
    *,
    request_id: str,
    model_alias: str,
    candidate_alias: str,
    candidate: QuestionerCandidateDraftV3,
    failed_dimensions: list[str],
    reason_codes: list[str],
    max_output_tokens: int,
    timeout_seconds: float | None = None,
    wording_only: bool = False,
) -> GenerationRequest:
    """Render a text-only, immutable-facts-free revision request."""

    prompt_version = (
        QUESTIONER_REVISION_V2_PROMPT_VERSION
        if wording_only
        else QUESTIONER_REVISION_V1_PROMPT_VERSION
    )
    payload = {
        "candidate_alias": candidate_alias,
        "generated_question": candidate.generated_question,
        "paraphrases": candidate.paraphrases,
        "failed_dimensions": sorted(failed_dimensions),
        "reason_codes": sorted(reason_codes),
    }
    if not wording_only:
        payload.update(
            {
                "question_type": candidate.question_type.value,
                "reasoning_type": candidate.reasoning_type.value,
            }
        )
    return GenerationRequest(
        request_id=request_id,
        model_alias=model_alias,
        messages=[
            ModelMessage(
                role=SpeakerRole.SYSTEM,
                parts=[
                    ModelContentPart(
                        kind=ModelContentKind.TEXT,
                        text=questioner_prompt_text(prompt_version),
                    )
                ],
            ),
            ModelMessage(
                role=SpeakerRole.USER,
                parts=[
                    ModelContentPart(
                        kind=ModelContentKind.TEXT,
                        text=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
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
                "name": (
                    "pmmc_questioner_revision_v2" if wording_only else "pmmc_questioner_revision_v1"
                ),
                "strict": True,
                "schema": questioner_response_schema(prompt_version),
            },
        },
        metadata={
            "task": "pmmc_questioner_revision",
            "prompt_version": prompt_version,
            "response_schema_hash": questioner_response_schema_sha256(prompt_version),
            "candidate_scope": candidate_alias,
            "wording_only": wording_only,
        },
    )
