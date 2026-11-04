"""Questioner output parsing, normalization, and hard candidate gate."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from pmmc.compiler.ids import canonical_content_hash, question_content_hash, stable_compiler_id
from pmmc.compiler.opportunities import BuiltQuestionOpportunitySet, local_opportunity_id
from pmmc.compiler.questioner_config import QuestionerConfig
from pmmc.models.execution import ModelOutputContractError
from pmmc.schemas import (
    QUESTIONER_V1_TYPES,
    CandidateGateStatus,
    CandidateRejectionCode,
    CompilationUnit,
    EvidenceType,
    MemoryUnit,
    Modality,
    OperatorEvidence,
    QBankDuplicateLink,
    QBankRouteClass,
    QuestionCandidate,
    QuestionCandidateGateResult,
    QuestionerAnswerableDraftV33,
    QuestionerAnswerableDraftV34,
    QuestionerAnswerableDraftV35,
    QuestionerAnswerClaimDraftV33,
    QuestionerAnswerClaimKind,
    QuestionerAnswerType,
    QuestionerBatch,
    QuestionerCandidateDraft,
    QuestionerCandidateDraftV2,
    QuestionerCandidateDraftV3,
    QuestionerCandidateDraftV32,
    QuestionerCandidateKind,
    QuestionerCandidateOrigin,
    QuestionerControlDraftV33,
    QuestionerControlDraftV34,
    QuestionerControlDraftV35,
    QuestionerDimensionVerdict,
    QuestionerRevisionEnvelopeV1,
    QuestionerRevisionEnvelopeV2,
    QuestionerTextAnswerClaimDraftV33,
    QuestionerVerificationBatchV1,
    QuestionerVerificationBatchV2,
    QuestionerVerificationEnvelopeV3,
    QuestionerVerificationEnvelopeV5,
    QuestionerVerificationEnvelopeV6,
    QuestionerVerificationItemV1,
    QuestionerVerificationItemV2,
    QuestionerVerificationReason,
    QuestionerVerificationVerdict,
    QuestionerVisualAnswerClaimDraftV33,
    QuestionReasoningType,
    QuestionType,
)
from pmmc.schemas.base import utc_now
from pmmc.schemas.questioner import (
    QuestionerAnswerableDraftV36,
    QuestionerControlDraftV36,
    QuestionerNativeCandidateDraft,
)


@dataclass(frozen=True)
class ParsedQuestionerBatch:
    batch: QuestionerBatch
    raw_candidate_count: int
    invalid_candidate_count: int
    invalid_diagnostics: tuple[dict[str, object], ...]
    repair_attempt_count: int = 0
    initial_error: str | None = None
    v3_candidates: tuple[QuestionerCandidateDraftV3, ...] = ()
    v32_candidates: tuple[QuestionerCandidateDraftV32, ...] = ()
    v33_candidates: tuple[QuestionerAnswerableDraftV33 | QuestionerControlDraftV33, ...] = ()
    v34_candidates: tuple[QuestionerAnswerableDraftV34 | QuestionerControlDraftV34, ...] = ()
    v35_candidates: tuple[QuestionerAnswerableDraftV35 | QuestionerControlDraftV35, ...] = ()
    v36_candidates: tuple[QuestionerAnswerableDraftV36 | QuestionerControlDraftV36, ...] = ()
    v3_candidate_ordinals: tuple[int, ...] = ()
    provider_item_scopes: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class GatedQuestionCandidate:
    candidate: QuestionCandidate
    expected_evidence: tuple[OperatorEvidence, ...]
    gate_result: QuestionCandidateGateResult
    duplicate_link: QBankDuplicateLink | None = None


@dataclass(frozen=True)
class ResolvedQuestionerV33Draft:
    """A provider v3.3 draft after deterministic evidence-claim validation."""

    provider_for_verifier: QuestionerCandidateDraftV3
    internal: QuestionerCandidateDraft
    claims: tuple[QuestionerAnswerClaimDraftV33, ...]
    evidence_aliases: tuple[str, ...]
    claim_support_aliases: tuple[str, ...] = ()
    answer_type: QuestionerAnswerType | None = None


def normalize_question_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _decode_items(text: str) -> tuple[list[object], int, str | None]:
    try:
        payload = json.loads(text)
        repairs = 0
        initial_error = None
    except Exception as initial:
        if isinstance(initial, json.JSONDecodeError):
            initial_error = f"JSONDecodeError(line={initial.lineno},column={initial.colno})"
        else:
            initial_error = type(initial).__name__
        from json_repair import repair_json

        payload = json.loads(repair_json(text))
        repairs = 1
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError("Questioner output must be an object containing a candidates list")
    return list(payload["candidates"]), repairs, initial_error


def _sanitized_validation_errors(error: ValidationError) -> list[dict[str, str]]:
    result = [
        {
            "location": ".".join(str(part) for part in item.get("loc", ())) or "$",
            "type": str(item.get("type", "validation_error")),
        }
        for item in error.errors(include_url=False, include_context=False, include_input=False)
    ]
    return sorted(result, key=lambda item: (item["location"], item["type"]))


def _v2_to_internal(draft: QuestionerCandidateDraftV2) -> QuestionerCandidateDraft:
    evidence: list[OperatorEvidence] = []
    for item in draft.expected_evidence:
        typed_field = {
            EvidenceType.MEMORY: "memory_id",
            EvidenceType.IMAGE: "image_id",
            EvidenceType.SESSION: "session_id",
            # Canonical benchmark captions are representations of a MemoryUnit,
            # not independently identified records. The renderer exposes their
            # owning memory_id, so preserve the caption modality while binding
            # prospective lineage to that canonical memory.
            EvidenceType.CAPTION: "memory_id",
        }[EvidenceType(item.evidence_type.value)]
        evidence.append(
            OperatorEvidence(
                evidence_id=item.evidence_id,
                evidence_type=EvidenceType(item.evidence_type.value),
                **{typed_field: item.evidence_id},
            )
        )
    return QuestionerCandidateDraft(
        generated_question=draft.generated_question,
        question_type=QuestionType(draft.question_type.value),
        reasoning_type=draft.reasoning_type,
        expected_answer=draft.expected_answer,
        source_memory_ids=draft.source_memory_ids,
        expected_evidence=evidence,
        required_modalities=[Modality(item.value) for item in draft.required_modalities],
        evidence_requirement=draft.evidence_requirement,
        hop_count=draft.hop_count,
        paraphrases=draft.paraphrases,
        confidence=draft.confidence,
    )


def parse_questioner_output(
    text: str,
    *,
    question_budget_max: int = 8,
    parser_version: str = "questioner-json-parser-v1",
) -> ParsedQuestionerBatch:
    if parser_version not in {
        "questioner-json-parser-v1",
        "questioner-json-parser-v2",
        "questioner-json-parser-v3",
        "questioner-json-parser-v3.1",
        "questioner-json-parser-v3.2",
        "questioner-json-parser-v3.3",
        "questioner-json-parser-v3.4",
        "questioner-json-parser-v3.5",
        "questioner-json-parser-v3.6",
    }:
        raise ValueError(f"unsupported Questioner parser version: {parser_version}")
    if parser_version in {
        "questioner-json-parser-v3.3",
        "questioner-json-parser-v3.4",
        "questioner-json-parser-v3.5",
        "questioner-json-parser-v3.6",
    }:
        try:
            payload = json.loads(text)
            repairs = 0
            initial_error = None
        except json.JSONDecodeError as initial:
            if parser_version == "questioner-json-parser-v3.6":
                raise ModelOutputContractError("native_json_required") from initial
            initial_error = type(initial).__name__
            from json_repair import repair_json

            payload = json.loads(repair_json(text))
            repairs = 1
        if not isinstance(payload, dict):
            if parser_version == "questioner-json-parser-v3.6":
                raise ModelOutputContractError("native_envelope_invalid")
            raise ValueError("Questioner v3 strategy output must be an object")
        if parser_version == "questioner-json-parser-v3.6" and set(payload) != {
            "candidates",
            "control_candidates",
        }:
            raise ModelOutputContractError("native_envelope_invalid")
        answerable_items = payload.get("candidates")
        control_items = payload.get("control_candidates")
        if not isinstance(answerable_items, list) or not isinstance(control_items, list):
            if parser_version == "questioner-json-parser-v3.6":
                raise ModelOutputContractError("native_envelope_invalid")
            raise ValueError(
                "Questioner v3 strategy requires candidates and control_candidates lists"
            )
        tagged_items = [
            *((item, False) for item in answerable_items),
            *((item, True) for item in control_items),
        ]
        items = [item for item, _control in tagged_items]
    else:
        items, repairs, initial_error = _decode_items(text)
        tagged_items = [(item, False) for item in items]
    valid: list[QuestionerCandidateDraft] = []
    valid_v3: list[QuestionerCandidateDraftV3] = []
    valid_v32: list[QuestionerCandidateDraftV32] = []
    valid_v33: list[QuestionerAnswerableDraftV33 | QuestionerControlDraftV33] = []
    valid_v34: list[QuestionerAnswerableDraftV34 | QuestionerControlDraftV34] = []
    valid_v35: list[QuestionerAnswerableDraftV35 | QuestionerControlDraftV35] = []
    valid_v36: list[QuestionerAnswerableDraftV36 | QuestionerControlDraftV36] = []
    valid_v3_ordinals: list[int] = []
    diagnostics: list[dict[str, object]] = []
    scopes: list[dict[str, object]] = []
    answerable_ordinal = 0
    control_ordinal = 0
    for ordinal, (item, is_control) in enumerate(tagged_items):
        if parser_version == "questioner-json-parser-v3.6":
            source = item if isinstance(item, dict) else {}
            opportunity = source.get("opportunity_id")
            claims = source.get("answer_claims", [])
            scopes.append(
                {
                    "provider_ordinal": ordinal + 1,
                    "opportunity_alias": opportunity
                    if isinstance(opportunity, str) and re.fullmatch(r"O\d{2,3}", opportunity)
                    else None,
                    "support_aliases": [
                        claim["support_alias"]
                        for claim in claims
                        if isinstance(claim, dict)
                        and isinstance(claim.get("support_alias"), str)
                        and re.fullmatch(r"[TI]\d{2,3}", claim["support_alias"])
                    ]
                    if isinstance(claims, list)
                    else [],
                }
            )
        over_budget = (
            control_ordinal >= 1 if is_control else answerable_ordinal >= question_budget_max
        )
        if is_control:
            control_ordinal += 1
        else:
            answerable_ordinal += 1
        if over_budget:
            diagnostics.append(
                {
                    "ordinal": ordinal,
                    "error_type": "BudgetExceeded",
                    "summary": "candidate exceeds frozen budget",
                    "item_hash": canonical_content_hash(item),
                    "error_count": 1,
                    "errors": [{"location": "$", "type": "budget_exceeded"}],
                }
            )
            continue
        try:
            if parser_version == "questioner-json-parser-v3.6":
                valid_v36.append(
                    QuestionerControlDraftV36.model_validate(item)
                    if is_control
                    else QuestionerAnswerableDraftV36.model_validate(item)
                )
                valid_v3_ordinals.append(ordinal + 1)
            elif parser_version == "questioner-json-parser-v3.5":
                provider_v35 = (
                    QuestionerControlDraftV35.model_validate(item)
                    if is_control
                    else QuestionerAnswerableDraftV35.model_validate(item)
                )
                valid_v35.append(provider_v35)
                valid_v3_ordinals.append(ordinal + 1)
            elif parser_version == "questioner-json-parser-v3.4":
                provider_v34 = (
                    QuestionerControlDraftV34.model_validate(item)
                    if is_control
                    else QuestionerAnswerableDraftV34.model_validate(item)
                )
                valid_v34.append(provider_v34)
                valid_v3_ordinals.append(ordinal + 1)
            elif parser_version == "questioner-json-parser-v3.3":
                provider = (
                    QuestionerControlDraftV33.model_validate(item)
                    if is_control
                    else QuestionerAnswerableDraftV33.model_validate(item)
                )
                valid_v33.append(provider)
                valid_v3_ordinals.append(ordinal + 1)
            elif parser_version == "questioner-json-parser-v3.2":
                valid_v32.append(QuestionerCandidateDraftV32.model_validate(item))
                valid_v3_ordinals.append(ordinal + 1)
            elif parser_version in {
                "questioner-json-parser-v3",
                "questioner-json-parser-v3.1",
            }:
                valid_v3.append(QuestionerCandidateDraftV3.model_validate(item))
                valid_v3_ordinals.append(ordinal + 1)
            elif parser_version == "questioner-json-parser-v2":
                provider_draft = QuestionerCandidateDraftV2.model_validate(item)
                valid.append(_v2_to_internal(provider_draft))
            else:
                valid.append(QuestionerCandidateDraft.model_validate(item))
        except ValidationError as exc:
            errors = _sanitized_validation_errors(exc)
            diagnostics.append(
                {
                    "ordinal": ordinal,
                    "error_type": "ValidationError",
                    "summary": "candidate failed the strict Questioner schema",
                    "item_hash": canonical_content_hash(item),
                    "error_count": len(errors),
                    "errors": errors,
                }
            )
    return ParsedQuestionerBatch(
        batch=QuestionerBatch(candidates=valid),
        raw_candidate_count=len(items),
        invalid_candidate_count=len(diagnostics),
        invalid_diagnostics=tuple(diagnostics),
        repair_attempt_count=repairs,
        initial_error=initial_error,
        v3_candidates=tuple(valid_v3),
        v32_candidates=tuple(valid_v32),
        v33_candidates=tuple(valid_v33),
        v34_candidates=tuple(valid_v34),
        v35_candidates=tuple(valid_v35),
        v36_candidates=tuple(valid_v36),
        v3_candidate_ordinals=tuple(valid_v3_ordinals),
        provider_item_scopes=tuple(scopes),
    )


def questioner_output_parser(config: QuestionerConfig):  # type: ignore[no-untyped-def]
    def parse(result):  # type: ignore[no-untyped-def]
        from pmmc.models.execution import ParsedModelOutput

        # Even syntactically complete JSON is not a complete generation after a length stop.
        if config.is_v36 and getattr(result, "finish_reason", None) == "length":
            raise ModelOutputContractError("questioner_output_truncated")
        parsed = parse_questioner_output(
            result.text,
            question_budget_max=config.question_budget_max,
            parser_version=config.parser_version,
        )
        value = {
            "candidates": (
                [
                    item.model_dump(mode="json")
                    for item in (
                        parsed.v36_candidates
                        if config.is_v36
                        else parsed.v35_candidates
                        if config.is_v35
                        else parsed.v34_candidates
                        if config.is_v34
                        else parsed.v33_candidates
                        if config.is_v33
                        else parsed.v32_candidates
                        if config.is_v32
                        else parsed.v3_candidates
                    )
                ]
                if config.is_v3
                else parsed.batch.model_dump(mode="json")["candidates"]
            ),
            "raw_candidate_count": parsed.raw_candidate_count,
            "invalid_candidate_count": parsed.invalid_candidate_count,
            "invalid_diagnostics": list(parsed.invalid_diagnostics),
            **(
                {"provider_item_scopes": list(parsed.provider_item_scopes)} if config.is_v36 else {}
            ),
            **({"candidate_ordinals": list(parsed.v3_candidate_ordinals)} if config.is_v3 else {}),
        }
        return ParsedModelOutput(
            value=value,
            repair_attempt_count=parsed.repair_attempt_count,
            initial_error=parsed.initial_error,
        )

    return parse


def resolve_v3_draft(
    draft: QuestionerCandidateDraftV3 | QuestionerCandidateDraftV32,
    *,
    opportunity_set: BuiltQuestionOpportunitySet,
    candidate_origin: QuestionerCandidateOrigin = QuestionerCandidateOrigin.GENERATED,
    candidate_version: int = 0,
    supersedes_qbank_id: str | None = None,
) -> QuestionerCandidateDraft:
    opportunities = {local_opportunity_id(item): item for item in opportunity_set.opportunities}
    opportunity = opportunities.get(draft.opportunity_id)
    if opportunity is None:
        raise ValueError("Questioner v3 candidate references an unknown opportunity")
    if not set(draft.evidence_aliases) <= set(opportunity.evidence_aliases):
        raise ValueError("Questioner v3 candidate references evidence outside its opportunity")
    evidence = [opportunity_set.alias_to_evidence[alias] for alias in draft.evidence_aliases]
    groups = {opportunity_set.alias_to_group[alias] for alias in draft.evidence_aliases}
    source_memory_ids = list(
        dict.fromkeys(
            memory_id
            for item in evidence
            for memory_id in ([item.memory_id] if item.memory_id is not None else [])
        )
    )
    return QuestionerCandidateDraft(
        generated_question=draft.generated_question,
        question_type=QuestionType(draft.question_type.value),
        reasoning_type=draft.reasoning_type,
        expected_answer=draft.expected_answer,
        source_memory_ids=source_memory_ids,
        expected_evidence=evidence,
        required_modalities=[
            Modality(item.value)
            for item in (
                opportunity.required_modalities
                if isinstance(draft, QuestionerCandidateDraftV32)
                else draft.required_modalities
            )
        ],
        evidence_requirement=[opportunity.opportunity_type.value],
        hop_count=max(1, len(groups)),
        paraphrases=draft.paraphrases,
        confidence=draft.confidence,
        opportunity_id=opportunity.opportunity_id,
        candidate_kind=draft.candidate_kind,
        candidate_origin=candidate_origin,
        candidate_version=candidate_version,
        supersedes_qbank_id=supersedes_qbank_id,
    )


def _normalized_evidence_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _claim_quote_is_present(
    *,
    claim: QuestionerAnswerClaimDraftV33,
    evidence: OperatorEvidence,
    memories: Mapping[str, MemoryUnit],
) -> bool:
    if claim.claim_kind is QuestionerAnswerClaimKind.VISUAL:
        return evidence.image_id is not None and evidence.evidence_type in {
            EvidenceType.IMAGE,
            EvidenceType.RAW_IMAGE,
            EvidenceType.IMAGE_TEXT_PAIR,
        }
    owner_id = evidence.memory_id
    if owner_id is None:
        owner = evidence.metadata.get("owning_memory_id")
        owner_id = owner if isinstance(owner, str) and owner else None
    memory = memories.get(owner_id or "")
    if memory is None or claim.support_quote is None:
        return False
    needle = _normalized_evidence_text(claim.support_quote)
    return bool(needle) and any(
        needle in _normalized_evidence_text(value)
        for value in (memory.text, memory.caption)
        if value
    )


def resolve_v33_draft(
    draft: QuestionerAnswerableDraftV33 | QuestionerControlDraftV33,
    *,
    opportunity_set: BuiltQuestionOpportunitySet,
    memories: Sequence[MemoryUnit],
) -> ResolvedQuestionerV33Draft:
    """Resolve v3.3 claims without trusting provider-derived canonical fields."""

    opportunities = {local_opportunity_id(item): item for item in opportunity_set.opportunities}
    opportunity = opportunities.get(draft.opportunity_id)
    if opportunity is None:
        raise ValueError("Questioner v3.3 candidate references an unknown opportunity")
    memory_by_id = {item.memory_id: item for item in memories}

    if isinstance(draft, QuestionerControlDraftV33):
        if opportunity.opportunity_type.value != "refusal_boundary":
            raise ValueError("Questioner control candidate requires a refusal opportunity")
        if len(draft.paraphrases) != len(set(draft.paraphrases)):
            raise ValueError("Questioner control paraphrases must be unique")
        provider = QuestionerCandidateDraftV3(
            candidate_kind=QuestionerCandidateKind.REFUSAL,
            opportunity_id=draft.opportunity_id,
            generated_question=draft.generated_question,
            question_type=draft.question_type,
            reasoning_type=draft.reasoning_type,
            expected_answer=None,
            evidence_aliases=[],
            required_modalities=[Modality.TEXT],
            paraphrases=draft.paraphrases,
            confidence=draft.confidence,
        )
        return ResolvedQuestionerV33Draft(
            provider_for_verifier=provider,
            internal=resolve_v3_draft(provider, opportunity_set=opportunity_set),
            claims=(),
            evidence_aliases=(),
        )

    aliases = tuple(item.evidence_alias for item in draft.answer_claims)
    if len(aliases) != len(set(aliases)):
        raise ValueError("Questioner answer claims must use distinct evidence aliases")
    if len(draft.paraphrases) != len(set(draft.paraphrases)):
        raise ValueError("Questioner answerable paraphrases must be unique")
    if not set(aliases) <= set(opportunity.evidence_aliases):
        raise ValueError("Questioner v3.3 claim references evidence outside its opportunity")
    for claim in draft.answer_claims:
        evidence = opportunity_set.alias_to_evidence.get(claim.evidence_alias)
        if evidence is None:
            raise ValueError("Questioner v3.3 claim references an unknown evidence alias")
        if (
            claim.claim_kind is QuestionerAnswerClaimKind.TEXT
            and claim.support_quote is not None
            and len(claim.support_quote.split()) > 32
        ):
            raise ValueError("Questioner v3.3 text support quotes are limited to 32 tokens")
        if not _claim_quote_is_present(
            claim=claim,
            evidence=evidence,
            memories=memory_by_id,
        ):
            raise ValueError("Questioner v3.3 answer claim is not grounded in its evidence")

    groups = {opportunity_set.alias_to_group[alias] for alias in aliases}
    if draft.reasoning_type is QuestionReasoningType.MULTI_HOP and len(groups) < 2:
        raise ValueError("Questioner v3.3 multi-hop claim requires independent evidence groups")
    if len(groups) < opportunity.minimum_evidence_groups:
        raise ValueError("Questioner v3.3 claims do not satisfy the opportunity group contract")

    expected_answer = "; ".join(item.claim_text.strip() for item in draft.answer_claims)
    provider = QuestionerCandidateDraftV3(
        candidate_kind=QuestionerCandidateKind.ANSWERABLE,
        opportunity_id=draft.opportunity_id,
        generated_question=draft.generated_question,
        question_type=draft.question_type,
        reasoning_type=draft.reasoning_type,
        expected_answer=expected_answer,
        evidence_aliases=list(aliases),
        required_modalities=[Modality(item.value) for item in opportunity.required_modalities],
        paraphrases=draft.paraphrases,
        confidence=draft.confidence,
    )
    return ResolvedQuestionerV33Draft(
        provider_for_verifier=provider,
        internal=resolve_v3_draft(provider, opportunity_set=opportunity_set),
        claims=tuple(draft.answer_claims),
        evidence_aliases=aliases,
        claim_support_aliases=aliases,
    )


def resolve_v34_draft(
    draft: QuestionerAnswerableDraftV34 | QuestionerControlDraftV34,
    *,
    opportunity_set: BuiltQuestionOpportunitySet,
    memories: Sequence[MemoryUnit],
) -> ResolvedQuestionerV33Draft:
    """Resolve minimal span/image claims into the existing authoritative facts."""

    opportunities = {local_opportunity_id(item): item for item in opportunity_set.opportunities}
    opportunity = opportunities.get(draft.opportunity_id)
    if opportunity is None:
        raise ValueError("Questioner v3.4 candidate references an unknown opportunity")
    if isinstance(draft, QuestionerControlDraftV34):
        legacy_control = QuestionerControlDraftV33(
            candidate_kind=QuestionerCandidateKind.REFUSAL,
            opportunity_id=draft.opportunity_id,
            generated_question=draft.generated_question,
            question_type="answer_refusal",
            reasoning_type=QuestionReasoningType.REFUSAL,
            answer_claims=[],
            paraphrases=draft.paraphrases,
            confidence=draft.confidence,
        )
        resolved = resolve_v33_draft(
            legacy_control,
            opportunity_set=opportunity_set,
            memories=memories,
        )
        return ResolvedQuestionerV33Draft(
            provider_for_verifier=resolved.provider_for_verifier,
            internal=resolved.internal,
            claims=resolved.claims,
            evidence_aliases=resolved.evidence_aliases,
            claim_support_aliases=(),
        )

    converted_claims: list[QuestionerAnswerClaimDraftV33] = []
    evidence_aliases: list[str] = []
    support_aliases: list[str] = []
    for claim in draft.answer_claims:
        support_aliases.append(claim.support_alias)
        if claim.claim_kind is QuestionerAnswerClaimKind.TEXT:
            span = opportunity_set.sentence_span_aliases.get(claim.support_alias)
            if span is None:
                raise ValueError("Questioner v3.4 text claim uses an unknown span alias")
            evidence_alias, quote = span
            if evidence_alias not in opportunity.evidence_aliases:
                raise ValueError("Questioner v3.4 span is outside the opportunity")
            converted_claims.append(
                QuestionerTextAnswerClaimDraftV33(
                    claim_kind=QuestionerAnswerClaimKind.TEXT,
                    evidence_alias=evidence_alias,
                    claim_text=claim.claim_text,
                    support_quote=quote,
                )
            )
        else:
            evidence = opportunity_set.alias_to_evidence.get(claim.support_alias)
            if (
                evidence is None
                or evidence.image_id is None
                or claim.support_alias not in opportunity.evidence_aliases
            ):
                raise ValueError("Questioner v3.4 visual claim requires an in-scope image")
            evidence_alias = claim.support_alias
            converted_claims.append(
                QuestionerVisualAnswerClaimDraftV33(
                    claim_kind=QuestionerAnswerClaimKind.VISUAL,
                    evidence_alias=evidence_alias,
                    claim_text=claim.claim_text,
                    support_quote=None,
                )
            )
        evidence_aliases.append(evidence_alias)
    if len(evidence_aliases) != len(set(evidence_aliases)):
        raise ValueError("Questioner v3.4 answer claims must use distinct evidence")
    question_type = opportunity.allowed_question_types[0]
    reasoning_type = opportunity.allowed_reasoning_types[0]
    legacy_answerable = QuestionerAnswerableDraftV33(
        candidate_kind=QuestionerCandidateKind.ANSWERABLE,
        opportunity_id=draft.opportunity_id,
        generated_question=draft.generated_question,
        question_type=question_type,
        reasoning_type=reasoning_type,
        answer_claims=converted_claims,
        paraphrases=draft.paraphrases,
        confidence=draft.confidence,
    )
    resolved = resolve_v33_draft(
        legacy_answerable,
        opportunity_set=opportunity_set,
        memories=memories,
    )
    return ResolvedQuestionerV33Draft(
        provider_for_verifier=resolved.provider_for_verifier,
        internal=resolved.internal,
        claims=resolved.claims,
        evidence_aliases=resolved.evidence_aliases,
        claim_support_aliases=tuple(support_aliases),
    )


def resolve_v35_draft(
    draft: QuestionerAnswerableDraftV35 | QuestionerControlDraftV35,
    *,
    opportunity_set: BuiltQuestionOpportunitySet,
    memories: Sequence[MemoryUnit],
) -> ResolvedQuestionerV33Draft:
    """Resolve v3.5 while retaining the provider's explicit answer-type contract."""

    payload = draft.model_dump(mode="json")
    legacy: QuestionerAnswerableDraftV34 | QuestionerControlDraftV34
    if isinstance(draft, QuestionerControlDraftV35):
        answer_type = QuestionerAnswerType.REFUSAL
        legacy = QuestionerControlDraftV34.model_validate(payload)
    else:
        answer_types = {item.answer_type for item in draft.answer_claims}
        if len(answer_types) != 1:
            raise ValueError("Questioner v3.5 claims must agree on one answer type")
        answer_type = next(iter(answer_types))
        for claim in payload["answer_claims"]:
            claim.pop("answer_type", None)
        legacy = QuestionerAnswerableDraftV34.model_validate(payload)
    resolved = resolve_v34_draft(
        legacy,
        opportunity_set=opportunity_set,
        memories=memories,
    )
    return ResolvedQuestionerV33Draft(
        provider_for_verifier=resolved.provider_for_verifier,
        internal=resolved.internal.model_copy(update={"answer_type": answer_type}),
        claims=resolved.claims,
        evidence_aliases=resolved.evidence_aliases,
        claim_support_aliases=resolved.claim_support_aliases,
        answer_type=answer_type,
    )


def parse_questioner_verifier_output(text: str) -> QuestionerVerificationBatchV1:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        from json_repair import repair_json

        payload = json.loads(repair_json(text))
    return QuestionerVerificationBatchV1.model_validate(payload)


def questioner_verifier_output_parser(
    *,
    expected_aliases: Sequence[str] | None = None,
    schema_version: str = "pmmc-questioner-verifier-v1",
):  # type: ignore[no-untyped-def]
    expected = tuple(expected_aliases) if expected_aliases is not None else None
    if expected is not None and (not expected or len(expected) != len(set(expected))):
        raise ValueError("expected verifier aliases must be non-empty and unique")

    if schema_version not in {
        "pmmc-questioner-verifier-v1",
        "pmmc-questioner-verifier-v2",
    }:
        raise ValueError("unsupported Questioner verifier schema version")
    item_model: type[QuestionerVerificationItemV1] | type[QuestionerVerificationItemV2]
    batch_model: type[QuestionerVerificationBatchV1] | type[QuestionerVerificationBatchV2]
    if schema_version == "pmmc-questioner-verifier-v2":
        item_model = QuestionerVerificationItemV2
        batch_model = QuestionerVerificationBatchV2
    else:
        item_model = QuestionerVerificationItemV1
        batch_model = QuestionerVerificationBatchV1

    def rejected(
        alias: str,
    ) -> QuestionerVerificationItemV1 | QuestionerVerificationItemV2:
        if schema_version == "pmmc-questioner-verifier-v2":
            from pmmc.schemas import QuestionerVerificationDimensionsV2

            return QuestionerVerificationItemV2(
                candidate_alias=alias,
                requested_action=QuestionerVerificationVerdict.REJECT,
                dimensions=QuestionerVerificationDimensionsV2(
                    user_facing=False,
                    future_plausible=False,
                    not_historical_replay=False,
                    answer_explicitly_entailed=False,
                    evidence_sufficient=False,
                    taxonomy_correct=False,
                    modality_correct=False,
                    visual_necessary=False,
                    multi_hop_genuine=False,
                    metadata_free=False,
                ),
                reason_codes=[QuestionerVerificationReason.NOT_ANSWERABLE],
                replacement=None,
            )
        return QuestionerVerificationItemV1(
            candidate_alias=alias,
            verdict=QuestionerVerificationVerdict.REJECT,
            reason_codes=[QuestionerVerificationReason.NOT_ANSWERABLE],
            replacement=None,
        )

    def parse(result):  # type: ignore[no-untyped-def]
        from pmmc.models.execution import ParsedModelOutput

        repairs = 0
        diagnostic_codes: set[str] = set()
        try:
            payload = json.loads(result.text)
        except (json.JSONDecodeError, TypeError, ValueError):
            from json_repair import repair_json

            repairs = 1
            diagnostic_codes.add("verifier_json_repaired")
            try:
                payload = json.loads(repair_json(result.text))
            except (json.JSONDecodeError, TypeError, ValueError):
                payload = None
                diagnostic_codes.add("verifier_json_invalid")

        if expected is None:
            parsed = batch_model.model_validate(payload)
            return ParsedModelOutput(
                value=parsed.model_dump(mode="json"),
                repair_attempt_count=repairs,
                initial_error=(";".join(sorted(diagnostic_codes)) if diagnostic_codes else None),
            )

        raw_results: list[object] = []
        if isinstance(payload, Mapping) and isinstance(payload.get("results"), list):
            raw_results = list(payload["results"])
        else:
            diagnostic_codes.add("verifier_envelope_invalid")

        by_alias: dict[str, list[object]] = defaultdict(list)
        raw_aliases: list[str] = []
        for raw in raw_results:
            if not isinstance(raw, Mapping):
                diagnostic_codes.add("verifier_item_invalid")
                continue
            alias = raw.get("candidate_alias")
            if not isinstance(alias, str) or not alias:
                diagnostic_codes.add("verifier_alias_invalid")
                continue
            by_alias[alias].append(raw)
            raw_aliases.append(alias)

        expected_set = set(expected)
        if any(alias not in expected_set for alias in raw_aliases):
            diagnostic_codes.add("verifier_alias_unexpected")
        if [alias for alias in raw_aliases if alias in expected_set] != list(expected):
            diagnostic_codes.add("verifier_alias_order_or_count_normalized")

        normalized: list[QuestionerVerificationItemV1 | QuestionerVerificationItemV2] = []
        fallback_aliases: list[str] = []
        for alias in expected:
            raw_items = by_alias.get(alias, [])
            if len(raw_items) != 1:
                diagnostic_codes.add(
                    "verifier_alias_missing" if not raw_items else "verifier_alias_duplicate"
                )
                fallback_aliases.append(alias)
                normalized.append(rejected(alias))
                continue
            try:
                item = item_model.model_validate(raw_items[0])
            except (ValidationError, ValueError):
                diagnostic_codes.add("verifier_item_validation_failed")
                fallback_aliases.append(alias)
                normalized.append(rejected(alias))
                continue
            normalized.append(item)

        parsed = batch_model.model_validate(
            {"results": [item.model_dump(mode="json") for item in normalized]}
        )
        parser_metadata = {
            "diagnostic_codes": sorted(diagnostic_codes),
            "fallback_aliases": fallback_aliases,
            "unexpected_alias_count": sum(alias not in expected_set for alias in raw_aliases),
        }
        return ParsedModelOutput(
            value={
                **parsed.model_dump(mode="json"),
                "_parser_metadata": parser_metadata,
            },
            repair_attempt_count=repairs,
            initial_error=(";".join(sorted(diagnostic_codes)) if diagnostic_codes else None),
        )

    return parse


def questioner_verifier_v3_output_parser(*, expected_alias: str):  # type: ignore[no-untyped-def]
    """Parse one candidate-scoped verifier result without conservative coercion."""

    def parse(result):  # type: ignore[no-untyped-def]
        from pmmc.models.execution import ParsedModelOutput

        repairs = 0
        initial_error: str | None = None
        try:
            payload = json.loads(result.text)
        except json.JSONDecodeError:
            from json_repair import repair_json

            repairs = 1
            initial_error = "verifier_json_repaired"
            payload = json.loads(repair_json(result.text))
        parsed = QuestionerVerificationEnvelopeV3.model_validate(payload)
        if parsed.result.candidate_alias != expected_alias:
            raise ValueError("Questioner verifier candidate scope does not match the request")
        return ParsedModelOutput(
            value=parsed.model_dump(mode="json"),
            repair_attempt_count=repairs,
            initial_error=initial_error,
        )

    return parse


def questioner_verifier_v5_output_parser(
    *,
    expected_alias: str,
    candidate: QuestionerCandidateDraftV3,
):  # type: ignore[no-untyped-def]
    """Parse dimensions-only v5 output and derive legacy-compatible safe facts."""

    mandatory = {
        "user_facing",
        "future_plausible",
        "not_historical_replay",
        "answer_addresses_question",
        "answer_explicitly_entailed",
        "speaker_subject_aligned",
        "evidence_sufficient",
        "taxonomy_compatible",
        "metadata_free",
    }
    visual = any(item is Modality.IMAGE for item in candidate.required_modalities)
    multi_hop = candidate.reasoning_type is QuestionReasoningType.MULTI_HOP
    reason_by_dimension = {
        "user_facing": QuestionerVerificationReason.NOT_USER_FACING,
        "future_plausible": QuestionerVerificationReason.NOT_FUTURE_PLAUSIBLE,
        "not_historical_replay": QuestionerVerificationReason.HISTORICAL_REPLAY,
        "answer_addresses_question": QuestionerVerificationReason.ANSWER_UNSUPPORTED,
        "answer_explicitly_entailed": QuestionerVerificationReason.ANSWER_UNSUPPORTED,
        "speaker_subject_aligned": QuestionerVerificationReason.ANSWER_UNSUPPORTED,
        "evidence_sufficient": QuestionerVerificationReason.EVIDENCE_INSUFFICIENT,
        "taxonomy_compatible": QuestionerVerificationReason.TAXONOMY_MISMATCH,
        "visual_necessary": QuestionerVerificationReason.VISUAL_NOT_NEEDED,
        "multi_hop_genuine": QuestionerVerificationReason.MULTI_HOP_NOT_GENUINE,
        "metadata_free": QuestionerVerificationReason.INTERNAL_METADATA,
    }

    def parse(result):  # type: ignore[no-untyped-def]
        from pmmc.models.execution import ParsedModelOutput

        payload = json.loads(result.text)
        parsed = QuestionerVerificationEnvelopeV5.model_validate(payload)
        if parsed.result.candidate_alias != expected_alias:
            raise ValueError("Questioner verifier candidate scope does not match")
        raw = parsed.result.dimensions.model_dump()
        if any(raw[name] is QuestionerDimensionVerdict.NOT_APPLICABLE for name in mandatory):
            raise ValueError("Questioner verifier used not_applicable for a required dimension")
        if visual and raw["visual_necessary"] is QuestionerDimensionVerdict.NOT_APPLICABLE:
            raise ValueError("visual candidate requires a visual necessity decision")
        if multi_hop and raw["multi_hop_genuine"] is QuestionerDimensionVerdict.NOT_APPLICABLE:
            raise ValueError("multi-hop candidate requires a genuine-hop decision")
        failed = {
            name
            for name, value in raw.items()
            if value is QuestionerDimensionVerdict.FALSE
            and (
                name not in {"visual_necessary", "multi_hop_genuine"}
                or (name == "visual_necessary" and visual)
                or (name == "multi_hop_genuine" and multi_hop)
            )
        }
        dimensions = {
            "user_facing": "user_facing" not in failed,
            "future_plausible": "future_plausible" not in failed,
            "not_historical_replay": "not_historical_replay" not in failed,
            "answer_explicitly_entailed": not bool(
                failed
                & {
                    "answer_addresses_question",
                    "answer_explicitly_entailed",
                    "speaker_subject_aligned",
                }
            ),
            "evidence_sufficient": "evidence_sufficient" not in failed,
            "taxonomy_correct": "taxonomy_compatible" not in failed,
            "modality_correct": True,
            "visual_necessary": "visual_necessary" not in failed,
            "multi_hop_genuine": "multi_hop_genuine" not in failed,
            "metadata_free": "metadata_free" not in failed,
        }
        reasons = sorted(
            {reason_by_dimension[name] for name in failed},
            key=lambda item: item.value,
        )
        normalized = QuestionerVerificationEnvelopeV3.model_validate(
            {
                "result": {
                    "candidate_alias": expected_alias,
                    "dimensions": dimensions,
                    "reason_codes": [item.value for item in reasons],
                }
            }
        )
        return ParsedModelOutput(
            value=normalized.model_dump(mode="json"),
            repair_attempt_count=0,
            initial_error=None,
        )

    return parse


def questioner_verifier_v6_output_parser(
    *,
    expected_alias: str,
    candidate: QuestionerCandidateDraftV3,
):  # type: ignore[no-untyped-def]
    """Parse v6 tri-state dimensions and derive the legacy persisted verdict shape."""

    mandatory = {
        "user_facing",
        "future_plausible",
        "not_historical_replay",
        "answer_addresses_question",
        "answer_explicitly_entailed",
        "speaker_subject_aligned",
        "evidence_sufficient",
        "taxonomy_compatible",
        "metadata_free",
        "question_answer_compatible",
        "answer_type_correct",
        "source_naturalness",
    }
    visual = any(item is Modality.IMAGE for item in candidate.required_modalities)
    multi_hop = candidate.reasoning_type is QuestionReasoningType.MULTI_HOP

    def parse(result):  # type: ignore[no-untyped-def]
        from pmmc.models.execution import ParsedModelOutput

        parsed = QuestionerVerificationEnvelopeV6.model_validate(json.loads(result.text))
        if parsed.result.candidate_alias != expected_alias:
            raise ValueError("Questioner verifier candidate scope does not match")
        raw = parsed.result.dimensions.model_dump()
        if any(raw[name] is QuestionerDimensionVerdict.NOT_APPLICABLE for name in mandatory):
            raise ValueError("Questioner verifier used not_applicable for a required dimension")
        if visual and raw["visual_necessary"] is QuestionerDimensionVerdict.NOT_APPLICABLE:
            raise ValueError("visual candidate requires a visual necessity decision")
        if multi_hop and raw["multi_hop_genuine"] is QuestionerDimensionVerdict.NOT_APPLICABLE:
            raise ValueError("multi-hop candidate requires a genuine-hop decision")
        failed = {
            name
            for name, value in raw.items()
            if value is QuestionerDimensionVerdict.FALSE
            and (
                name not in {"visual_necessary", "multi_hop_genuine"}
                or (name == "visual_necessary" and visual)
                or (name == "multi_hop_genuine" and multi_hop)
            )
        }
        answer_failures = {
            "answer_addresses_question",
            "answer_explicitly_entailed",
            "speaker_subject_aligned",
            "question_answer_compatible",
            "answer_type_correct",
        }
        normalized = QuestionerVerificationEnvelopeV3.model_validate(
            {
                "result": {
                    "candidate_alias": expected_alias,
                    "dimensions": {
                        "user_facing": not bool(failed & {"user_facing", "source_naturalness"}),
                        "future_plausible": "future_plausible" not in failed,
                        "not_historical_replay": "not_historical_replay" not in failed,
                        "answer_explicitly_entailed": not bool(failed & answer_failures),
                        "evidence_sufficient": "evidence_sufficient" not in failed,
                        "taxonomy_correct": not bool(
                            failed & {"taxonomy_compatible", "answer_type_correct"}
                        ),
                        "modality_correct": True,
                        "visual_necessary": "visual_necessary" not in failed,
                        "multi_hop_genuine": "multi_hop_genuine" not in failed,
                        "metadata_free": "metadata_free" not in failed,
                    },
                    "reason_codes": [
                        item.value for item in _verification_reasons_for_v6_failures(failed)
                    ],
                }
            }
        )
        return ParsedModelOutput(
            value=normalized.model_dump(mode="json"),
            repair_attempt_count=0,
            initial_error=None,
        )

    return parse


def _verification_reasons_for_v6_failures(
    failed: set[str],
) -> list[QuestionerVerificationReason]:
    mapping = {
        "user_facing": QuestionerVerificationReason.NOT_USER_FACING,
        "source_naturalness": QuestionerVerificationReason.NOT_USER_FACING,
        "future_plausible": QuestionerVerificationReason.NOT_FUTURE_PLAUSIBLE,
        "not_historical_replay": QuestionerVerificationReason.HISTORICAL_REPLAY,
        "answer_addresses_question": QuestionerVerificationReason.ANSWER_UNSUPPORTED,
        "answer_explicitly_entailed": QuestionerVerificationReason.ANSWER_UNSUPPORTED,
        "speaker_subject_aligned": QuestionerVerificationReason.ANSWER_UNSUPPORTED,
        "question_answer_compatible": QuestionerVerificationReason.ANSWER_UNSUPPORTED,
        "answer_type_correct": QuestionerVerificationReason.TAXONOMY_MISMATCH,
        "evidence_sufficient": QuestionerVerificationReason.EVIDENCE_INSUFFICIENT,
        "taxonomy_compatible": QuestionerVerificationReason.TAXONOMY_MISMATCH,
        "visual_necessary": QuestionerVerificationReason.VISUAL_NOT_NEEDED,
        "multi_hop_genuine": QuestionerVerificationReason.MULTI_HOP_NOT_GENUINE,
        "metadata_free": QuestionerVerificationReason.INTERNAL_METADATA,
    }
    return sorted(
        {mapping[name] for name in failed if name in mapping},
        key=lambda item: item.value,
    )


def questioner_revision_output_parser(*, expected_alias: str, wording_only: bool = False):  # type: ignore[no-untyped-def]
    """Parse the isolated v3.2 wording/taxonomy revision envelope."""

    def parse(result):  # type: ignore[no-untyped-def]
        from pmmc.models.execution import ParsedModelOutput

        repairs = 0
        initial_error: str | None = None
        try:
            payload = json.loads(result.text)
        except json.JSONDecodeError:
            from json_repair import repair_json

            repairs = 1
            initial_error = "revision_json_repaired"
            payload = json.loads(repair_json(result.text))
        parsed = (
            QuestionerRevisionEnvelopeV2.model_validate(payload)
            if wording_only
            else QuestionerRevisionEnvelopeV1.model_validate(payload)
        )
        if parsed.replacement.candidate_alias != expected_alias:
            raise ValueError("Questioner revision candidate scope does not match the request")
        return ParsedModelOutput(
            value=parsed.model_dump(mode="json"),
            repair_attempt_count=repairs,
            initial_error=initial_error,
        )

    return parse


class QuestionerCandidateGate:
    def __init__(self, config: QuestionerConfig) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        unit: CompilationUnit,
        generation_id: str,
        model_call_id: str,
        drafts: list[QuestionerCandidateDraft],
        memories: list[MemoryUnit],
        known_primaries: dict[str, str] | None = None,
        created_at: datetime | None = None,
        opportunity_set: BuiltQuestionOpportunitySet | None = None,
        generation_ranks: list[int] | None = None,
        producer_model_call_ids: list[str] | None = None,
        candidate_created_ats: list[datetime] | None = None,
        apply_dedupe: bool = True,
    ) -> list[GatedQuestionCandidate]:
        memory_map = {item.memory_id: item for item in memories}
        visible_memory_ids = {item.memory_id for item in unit.memories}
        if not visible_memory_ids <= set(memory_map):
            raise ValueError("gate requires every compilation-unit memory payload")
        image_ids = {
            image_id
            for memory_id in visible_memory_ids
            for image_id in (
                set(memory_map[memory_id].image_ids)
                | {item.image_id for item in memory_map[memory_id].image_refs}
            )
        }
        session_ids = {memory_map[memory_id].session_id for memory_id in visible_memory_ids}
        primaries = known_primaries if known_primaries is not None else {}
        opportunities = (
            {item.opportunity_id: item for item in opportunity_set.opportunities}
            if opportunity_set is not None
            else {}
        )
        ranks = generation_ranks or list(range(1, len(drafts) + 1))
        if len(ranks) != len(drafts) or len(set(ranks)) != len(ranks):
            raise ValueError("generation ranks must be unique and align with drafts")
        if any(rank <= 0 for rank in ranks):
            raise ValueError("generation ranks must be positive")
        producer_calls = producer_model_call_ids or [model_call_id] * len(drafts)
        if len(producer_calls) != len(drafts) or any(not item for item in producer_calls):
            raise ValueError("producer model-call IDs must align with drafts")
        default_created_at = created_at or utc_now()
        created_ats = candidate_created_ats or [default_created_at] * len(drafts)
        if len(created_ats) != len(drafts):
            raise ValueError("candidate creation times must align with drafts")
        result: list[GatedQuestionCandidate] = []
        for generation_rank, draft, producer_call, candidate_created_at in zip(
            ranks,
            drafts,
            producer_calls,
            created_ats,
            strict=True,
        ):
            visible_source_memory_ids = [
                memory_id
                for memory_id in draft.source_memory_ids
                if memory_id in visible_memory_ids
            ]
            visible_expected_evidence = [
                item
                for item in draft.expected_evidence
                if self._evidence_is_visible(
                    item,
                    visible_memory_ids=visible_memory_ids,
                    visible_image_ids=image_ids,
                    visible_session_ids=session_ids,
                )
            ]
            dropped_source_hashes = sorted(
                canonical_content_hash(memory_id)
                for memory_id in draft.source_memory_ids
                if memory_id not in visible_memory_ids
            )
            dropped_evidence_hashes = sorted(
                canonical_content_hash([item.evidence_type.value, item.evidence_id])
                for item in draft.expected_evidence
                if item not in visible_expected_evidence
            )
            evidence_key = sorted(
                f"{item.evidence_type.value}:{item.evidence_id}" for item in draft.expected_evidence
            )
            normalized = normalize_question_text(draft.generated_question)
            question_hash = canonical_content_hash(normalized)
            identity = [
                run_id,
                unit.compiler_unit_id,
                generation_rank,
                draft.candidate_version,
                question_hash,
                evidence_key,
            ]
            qbank_id = stable_compiler_id("qbank", identity)
            candidate = QuestionCandidate(
                qbank_id=qbank_id,
                run_id=run_id,
                snapshot_id=snapshot_id,
                compiler_unit_id=unit.compiler_unit_id,
                generation_id=generation_id,
                producer_model_call_id=producer_call,
                question_hash="pending",
                conversation_id=unit.conversation_id,
                created_after_session=unit.available_after_session,
                canonical_question=draft.generated_question,
                paraphrases=draft.paraphrases[: self.config.max_paraphrases],
                question_type=draft.question_type,
                reasoning_type=draft.reasoning_type,
                expected_answer=draft.expected_answer,
                source_memory_ids=visible_source_memory_ids,
                evidence_ids=[item.evidence_id for item in visible_expected_evidence],
                required_modalities=draft.required_modalities,
                evidence_requirement=draft.evidence_requirement,
                hop_count=draft.hop_count,
                generation_rank=generation_rank,
                questioner_confidence=draft.confidence,
                opportunity_id=draft.opportunity_id,
                candidate_origin=draft.candidate_origin,
                candidate_version=draft.candidate_version,
                supersedes_qbank_id=draft.supersedes_qbank_id,
                answer_type=draft.answer_type,
                route_class=(
                    QBankRouteClass.CONTROL
                    if draft.candidate_kind is QuestionerCandidateKind.REFUSAL
                    else QBankRouteClass.ANSWERABLE
                ),
                **(
                    {"created_at": candidate_created_at} if candidate_created_at is not None else {}
                ),
                metadata={
                    "normalized_question_hash": question_hash,
                    "dropped_source_reference_count": len(dropped_source_hashes),
                    "dropped_source_reference_hashes": dropped_source_hashes,
                    "dropped_evidence_reference_count": len(dropped_evidence_hashes),
                    "dropped_evidence_reference_hashes": dropped_evidence_hashes,
                },
            )
            content_hash = question_content_hash(candidate)
            candidate = candidate.model_copy(
                update={
                    "question_hash": content_hash,
                    "candidate_content_hash": content_hash,
                }
            )
            codes = self._rejection_codes(
                draft=draft,
                visible_memory_ids=visible_memory_ids,
                visible_image_ids=image_ids,
                visible_session_ids=session_ids,
                memories=[memory_map[memory_id] for memory_id in visible_memory_ids],
                opportunity=opportunities.get(draft.opportunity_id or ""),
                opportunity_set=opportunity_set,
            )
            if self.config.is_v36:
                from pmmc.compiler.questioner_native import native_static_codes

                if not isinstance(draft, QuestionerNativeCandidateDraft):
                    codes.append(CandidateRejectionCode.INVALID_SCHEMA)
                else:
                    codes.extend(native_static_codes(draft))
            if self.config.is_v33 or self.config.is_v34:
                answer_claim_hash = canonical_content_hash([draft.expected_answer, evidence_key])
                requirement_contract_hash = canonical_content_hash(
                    {
                        "required_modalities": sorted(
                            item.value for item in draft.required_modalities
                        ),
                        "hop_count": draft.hop_count,
                        "evidence_requirement": sorted(draft.evidence_requirement),
                        "route_class": candidate.route_class.value,
                    }
                )
                duplicate_key = (
                    canonical_content_hash([unit.conversation_id, normalized])
                    if self.config.is_v34
                    else canonical_content_hash(
                        [
                            unit.conversation_id,
                            normalized,
                            answer_claim_hash,
                            requirement_contract_hash,
                        ]
                    )
                )
                if self.config.is_v36:
                    from pmmc.compiler.questioner_native import information_need_key

                    duplicate_key = canonical_content_hash(
                        [
                            unit.conversation_id,
                            information_need_key(draft.generated_question, draft.answer_type),
                        ]
                    )
            else:
                answer_claim_hash = None
                requirement_contract_hash = None
                duplicate_key = canonical_content_hash(
                    [unit.conversation_id, normalized, evidence_key]
                )
            duplicate_of = primaries.get(duplicate_key) if apply_dedupe else None
            duplicate_link: QBankDuplicateLink | None = None
            if duplicate_of is not None:
                codes.append(CandidateRejectionCode.DUPLICATE)
                duplicate_link = QBankDuplicateLink(
                    duplicate_link_id=stable_compiler_id(
                        "qbank-duplicate",
                        [run_id, qbank_id, duplicate_of, self.config.dedupe_version],
                    ),
                    run_id=run_id,
                    duplicate_qbank_id=qbank_id,
                    primary_qbank_id=duplicate_of,
                    method=self.config.dedupe_version,
                    hard_duplicate=True,
                    **(
                        {"created_at": candidate_created_at}
                        if candidate_created_at is not None
                        else {}
                    ),
                )
            elif not codes and apply_dedupe:
                primaries[duplicate_key] = qbank_id
            codes = list(dict.fromkeys(codes))
            gate = QuestionCandidateGateResult(
                gate_result_id=stable_compiler_id(
                    "question-gate", [qbank_id, self.config.gate_version]
                ),
                run_id=run_id,
                compiler_unit_id=unit.compiler_unit_id,
                generation_id=generation_id,
                qbank_id=qbank_id,
                gate_version=self.config.gate_version,
                status=(
                    CandidateGateStatus.REJECTED
                    if codes
                    else CandidateGateStatus.ELIGIBLE_FOR_PLANNING
                ),
                rejection_codes=codes,
                duplicate_of_qbank_id=duplicate_of,
                content_hash=content_hash,
                **(
                    {"created_at": candidate_created_at} if candidate_created_at is not None else {}
                ),
                metadata={
                    "duplicate_key": duplicate_key,
                    **(
                        {
                            "family_budget_rank": generation_rank,
                            "ranking_version": "questioner-fixed-generation-prefix-v2",
                        }
                        if self.config.is_v36
                        else {}
                    ),
                    **(
                        {
                            "answer_claim_hash": answer_claim_hash,
                            "requirement_contract_hash": requirement_contract_hash,
                            "family_dedupe_version": self.config.family_dedupe_version,
                        }
                        if self.config.is_v33 or self.config.is_v34
                        else {}
                    ),
                },
            )
            result.append(
                GatedQuestionCandidate(
                    candidate=candidate,
                    expected_evidence=tuple(visible_expected_evidence),
                    gate_result=gate,
                    duplicate_link=duplicate_link,
                )
            )
        return result

    @staticmethod
    def _evidence_is_visible(
        evidence: OperatorEvidence,
        *,
        visible_memory_ids: set[str],
        visible_image_ids: set[str],
        visible_session_ids: set[str],
    ) -> bool:
        return (
            (evidence.memory_id is not None and evidence.memory_id in visible_memory_ids)
            or (evidence.image_id is not None and evidence.image_id in visible_image_ids)
            or (evidence.session_id is not None and evidence.session_id in visible_session_ids)
            or (
                evidence.caption_id is not None
                and evidence.caption_id in visible_memory_ids | visible_image_ids
            )
        )

    def _rejection_codes(
        self,
        *,
        draft: QuestionerCandidateDraft,
        visible_memory_ids: set[str],
        visible_image_ids: set[str],
        visible_session_ids: set[str],
        memories: list[MemoryUnit],
        opportunity: object | None,
        opportunity_set: BuiltQuestionOpportunitySet | None,
    ) -> list[CandidateRejectionCode]:
        codes: list[CandidateRejectionCode] = []
        if draft.question_type not in QUESTIONER_V1_TYPES:
            codes.append(CandidateRejectionCode.UNSUPPORTED_QUESTION_TYPE)
        if not set(draft.source_memory_ids) <= visible_memory_ids:
            codes.append(CandidateRejectionCode.FUTURE_OR_FOREIGN_REFERENCE)
        for evidence in draft.expected_evidence:
            if not QuestionerCandidateGate._evidence_is_visible(
                evidence,
                visible_memory_ids=visible_memory_ids,
                visible_image_ids=visible_image_ids,
                visible_session_ids=visible_session_ids,
            ):
                codes.append(CandidateRejectionCode.FUTURE_OR_FOREIGN_REFERENCE)
        refusal = draft.question_type is QuestionType.ANSWER_REFUSAL
        if refusal != (draft.candidate_kind is QuestionerCandidateKind.REFUSAL):
            codes.append(CandidateRejectionCode.TAXONOMY_INVARIANT_FAILED)
        if not refusal and draft.expected_answer is None:
            codes.append(CandidateRejectionCode.MISSING_EXPECTED_ANSWER)
        if refusal and draft.expected_answer is not None:
            codes.append(CandidateRejectionCode.MODALITY_MISMATCH)
        if not refusal and not draft.expected_evidence:
            codes.append(CandidateRejectionCode.MISSING_EXPECTED_EVIDENCE)
        visual = draft.question_type in {
            QuestionType.VISUAL_DETAIL,
            QuestionType.IMAGE_TEXT_BINDING,
            QuestionType.VISUAL_SEARCH,
        }
        visual_evidence = any(
            item.evidence_type
            in {
                EvidenceType.IMAGE,
                EvidenceType.RAW_IMAGE,
                EvidenceType.IMAGE_TEXT_PAIR,
            }
            for item in draft.expected_evidence
        )
        visual_modality = bool(
            set(draft.required_modalities)
            & {Modality.IMAGE, Modality.RAW_IMAGE, Modality.MULTIMODAL}
        )
        if (
            self.config.is_v36
            and self.config.memory_raw_image_policy == "forbid"
            and (visual_evidence or visual_modality)
        ):
            codes.append(CandidateRejectionCode.MODALITY_MISMATCH)
        if visual and (not visual_evidence or not visual_modality):
            codes.append(CandidateRejectionCode.VISUAL_EVIDENCE_MISSING)
        if draft.question_type is QuestionType.MULTI_HOP and (
            draft.hop_count < 2 or len(draft.expected_evidence) < 2
        ):
            codes.append(CandidateRejectionCode.MULTI_HOP_EVIDENCE_MISSING)
        if draft.reasoning_type is QuestionReasoningType.MULTI_HOP and draft.hop_count < 2:
            codes.append(CandidateRejectionCode.MULTI_HOP_EVIDENCE_MISSING)
        if self.config.is_v3:
            from pmmc.schemas import QuestionOpportunity

            parsed_opportunity = (
                opportunity if isinstance(opportunity, QuestionOpportunity) else None
            )
            if parsed_opportunity is None or opportunity_set is None:
                codes.append(CandidateRejectionCode.OPPORTUNITY_MISMATCH)
            else:
                if draft.question_type.value not in {
                    item.value for item in parsed_opportunity.allowed_question_types
                }:
                    codes.append(CandidateRejectionCode.TAXONOMY_INVARIANT_FAILED)
                if draft.reasoning_type not in parsed_opportunity.allowed_reasoning_types:
                    codes.append(CandidateRejectionCode.REASONING_INVARIANT_FAILED)
                expected_modalities = {
                    Modality(item.value) for item in parsed_opportunity.required_modalities
                }
                if set(draft.required_modalities) != expected_modalities:
                    codes.append(CandidateRejectionCode.MODALITY_MISMATCH)
                aliases = {
                    alias
                    for alias in parsed_opportunity.evidence_aliases
                    if opportunity_set.alias_to_evidence[alias] in draft.expected_evidence
                }
                groups = {opportunity_set.alias_to_group[alias] for alias in aliases}
                if (
                    draft.candidate_kind is not QuestionerCandidateKind.REFUSAL
                    and len(groups) < parsed_opportunity.minimum_evidence_groups
                ):
                    codes.append(CandidateRejectionCode.OPPORTUNITY_MISMATCH)
            codes.extend(
                self._semantic_static_codes(
                    draft=draft,
                    memories=memories,
                    opportunity_set=opportunity_set,
                    strict_v31=self.config.is_v31 or self.config.is_v32,
                    strict_v35=self.config.is_v35,
                    allow_explicit_plans=self.config.is_v36,
                )
            )
        return list(dict.fromkeys(codes))

    @staticmethod
    def _semantic_static_codes(
        *,
        draft: QuestionerCandidateDraft,
        memories: list[MemoryUnit],
        opportunity_set: BuiltQuestionOpportunitySet | None,
        strict_v31: bool = False,
        strict_v35: bool = False,
        allow_explicit_plans: bool = False,
    ) -> list[CandidateRejectionCode]:
        codes: list[CandidateRejectionCode] = []
        visible_text = " ".join(
            value
            for memory in memories
            for value in (memory.text or "", memory.caption or "")
            if value
        )
        public_text = " ".join(
            [draft.generated_question, draft.expected_answer or "", *draft.paraphrases]
        )
        if re.search(
            r"\b(?:compiler[-_ ]?unit|qbank|"
            r"(?:memlens|memgallery)[:_-][a-z0-9][\w.:-]*|"
            r"(?:memory|session|turn|round)[_:-][a-z0-9][\w.:-]*|"
            r"[MISCOQ]\d{2})\b",
            public_text,
            flags=re.IGNORECASE,
        ):
            codes.append(CandidateRejectionCode.INTERNAL_IDENTIFIER_LEAK)
        if re.search(
            r"\b(?:session order|current (?:date|time)|cutoff session|"
            r"compilation unit|memory id|system metadata|local aliases?|"
            r"bounded opportunities?|candidate aliases?|evidence aliases?|"
            r"(?:cannot|unable to) generate)\b",
            public_text,
            flags=re.IGNORECASE,
        ):
            codes.append(CandidateRejectionCode.SYSTEM_METADATA_REFERENCE)
        if re.search(
            r"\b(?:would you like me|do you want me|shall i|can i help|"
            r"let me know if)\b",
            draft.generated_question,
            flags=re.IGNORECASE,
        ):
            codes.append(CandidateRejectionCode.ASSISTANT_FOLLOW_UP)
        if draft.expected_answer and re.search(
            r"\b(?:to be determined|tbd|not (?:yet|specified|provided|known)|"
            r"will (?:provide|share|decide)|plans? to|placeholder|unknown(?: for now)?|"
            r"unsure|maybe|probably|it depends)\b",
            draft.expected_answer,
            flags=re.IGNORECASE,
        ):
            explicit_plan = (
                allow_explicit_plans
                and isinstance(draft, QuestionerNativeCandidateDraft)
                and draft.exact_claims
                and all(
                    claim.assertion == "plan"
                    and claim.subject == "user"
                    and claim.support.speaker == "user"
                    for claim in draft.exact_claims
                )
                and bool(
                    re.search(
                        r"\b(?:plan|plans|planning|intend|intends)\b",
                        draft.generated_question,
                        re.I,
                    )
                )
            )
            if not explicit_plan:
                codes.append(CandidateRejectionCode.PLACEHOLDER_EXPECTED_ANSWER)

        if strict_v35:
            codes.extend(QuestionerCandidateGate._v35_answer_contract_codes(draft))

        if strict_v31 and draft.candidate_kind is not QuestionerCandidateKind.REFUSAL:
            memory_map = {item.memory_id: item for item in memories}
            evidence_memories = []
            for evidence in draft.expected_evidence:
                memory_id = evidence.memory_id
                if memory_id is None and evidence.image_id is not None:
                    owner = evidence.metadata.get("owning_memory_id")
                    memory_id = owner if isinstance(owner, str) else None
                if memory_id is not None and memory_id in memory_map:
                    evidence_memories.append(memory_map[memory_id])
            user_state = re.search(
                r"\b(?:prefer|favorite|favourite|choose|chose|own|have|has|"
                r"bought|completed|finished|visited|went|started|stopped|user)\b",
                public_text,
                flags=re.IGNORECASE,
            )
            if user_state and not any(item.speaker.value == "user" for item in evidence_memories):
                codes.append(CandidateRejectionCode.EVIDENCE_NOT_ENTAILED)
            assistant_questions = [
                item
                for item in evidence_memories
                if item.speaker.value == "assistant" and item.text is not None and "?" in item.text
            ]
            if (
                evidence_memories
                and len(assistant_questions) == len(evidence_memories)
                and not re.search(
                    r"\b(?:assistant|you)\s+(?:ask|asked|say|said|suggest|suggested)\b",
                    draft.generated_question,
                    flags=re.IGNORECASE,
                )
            ):
                codes.append(CandidateRejectionCode.ASSISTANT_FOLLOW_UP)

        normalized = normalize_question_text(draft.generated_question)
        generated_tokens = set(normalized.split())
        for memory in memories:
            if memory.speaker.value != "user" or not memory.text:
                continue
            prior_questions = [
                match.group(0) for match in re.finditer(r"[^?]+\?", memory.text)
            ] or [memory.text]
            for prior_question in prior_questions:
                prior = normalize_question_text(prior_question)
                prior_tokens = set(prior.split())
                union = generated_tokens | prior_tokens
                similarity = len(generated_tokens & prior_tokens) / len(union) if union else 0.0
                if normalized == prior or similarity >= 0.9:
                    if draft.reasoning_type not in {
                        QuestionReasoningType.UPDATE,
                        QuestionReasoningType.CONFLICT,
                        QuestionReasoningType.TEMPORAL,
                    }:
                        codes.append(CandidateRejectionCode.HISTORICAL_QUESTION_REPLAY)
                    break
            if CandidateRejectionCode.HISTORICAL_QUESTION_REPLAY in codes:
                break
        generic_stop_tokens = {
            "about",
            "could",
            "does",
            "from",
            "have",
            "that",
            "this",
            "user",
            "were",
            "what",
            "when",
            "where",
            "which",
            "with",
            "would",
        }
        content_tokens = {
            token
            for token in normalize_question_text(visible_text).split()
            if len(token) >= 4 and token not in generic_stop_tokens
        }
        candidate_tokens = {
            token
            for token in normalize_question_text(public_text).split()
            if len(token) >= 4 and token not in generic_stop_tokens
        }
        if (
            draft.candidate_kind is not QuestionerCandidateKind.REFUSAL
            and candidate_tokens
            and not content_tokens & candidate_tokens
        ):
            codes.append(CandidateRejectionCode.GENERIC_EXTERNAL_QUESTION)
        if (
            draft.candidate_kind is not QuestionerCandidateKind.REFUSAL
            and draft.expected_answer
            and opportunity_set is not None
            and not any(item.image_id is not None for item in draft.expected_evidence)
        ):
            memory_map = {item.memory_id: item for item in memories}
            evidence_text = " ".join(
                _value
                for evidence in draft.expected_evidence
                if evidence.memory_id in memory_map
                for _value in (
                    memory_map[evidence.memory_id].text or "",
                    memory_map[evidence.memory_id].caption or "",
                )
                if _value
            )
            answer_tokens = {
                token
                for token in normalize_question_text(draft.expected_answer).split()
                if len(token) >= 3 and token not in {"the", "and", "was", "were", "with"}
            }
            evidence_tokens = set(normalize_question_text(evidence_text).split())
            if len(answer_tokens) >= 2 and not answer_tokens & evidence_tokens:
                codes.append(CandidateRejectionCode.EVIDENCE_NOT_ENTAILED)
        return codes

    @staticmethod
    def _v35_answer_contract_codes(
        draft: QuestionerCandidateDraft,
    ) -> list[CandidateRejectionCode]:
        if draft.candidate_kind is QuestionerCandidateKind.REFUSAL:
            return (
                []
                if draft.answer_type is QuestionerAnswerType.REFUSAL
                else [CandidateRejectionCode.ANSWER_TYPE_MISMATCH]
            )
        answer = (draft.expected_answer or "").strip()
        question = draft.generated_question.strip()
        codes: list[CandidateRejectionCode] = []
        normalized_answer = normalize_question_text(answer)
        normalized_question = normalize_question_text(question)
        answer_tokens = normalized_answer.split()
        question_tokens = normalized_question.split()
        if (
            normalized_answer
            and normalized_question
            and (
                normalized_answer == normalized_question
                or (
                    min(len(answer_tokens), len(question_tokens)) >= 3
                    and (
                        normalized_answer in normalized_question
                        or normalized_question in normalized_answer
                    )
                )
            )
        ):
            codes.append(CandidateRejectionCode.QUESTION_ANSWER_CLOSURE_FAILED)
        if answer.endswith("?") or re.match(
            r"^(?:what|which|who|where|when|why|how|is|are|was|were|do|does|did|can|could)\b",
            answer,
            flags=re.IGNORECASE,
        ):
            codes.append(CandidateRejectionCode.QUESTION_AS_ANSWER)
        if re.search(
            r"\b(?:as an ai|i (?:cannot|can't|am unable to)|language model|"
            r"local aliases?|bounded opportunit(?:y|ies)|harness|candidate alias)\b",
            answer,
            flags=re.IGNORECASE,
        ):
            codes.append(CandidateRejectionCode.AI_DISCLAIMER)
        if (
            not normalized_answer
            or answer.endswith(("...", ":", ";"))
            or re.search(
                r"\b(?:and|or|because|with|without|from|to|of|for|the|a|an)\s*$",
                answer,
                flags=re.IGNORECASE,
            )
        ):
            codes.append(CandidateRejectionCode.INCOMPLETE_ANSWER)

        expected_types: set[QuestionerAnswerType] | None = None
        if re.search(r"\b(?:how many|what number|what count|number of)\b", question, re.I):
            expected_types = {QuestionerAnswerType.COUNT}
        elif re.match(
            r"^(?:is|are|was|were|do|does|did|has|have|can|could|will|would)\b",
            question,
            re.I,
        ):
            expected_types = {QuestionerAnswerType.BOOLEAN}
        elif re.search(
            r"\b(?:list|which (?:items|things|options|products|images))\b",
            question,
            re.I,
        ):
            expected_types = {QuestionerAnswerType.LIST}
        elif re.match(r"^(?:who|whose)\b", question, re.I):
            expected_types = {QuestionerAnswerType.ENTITY}
        elif re.match(r"^when\b", question, re.I):
            expected_types = {QuestionerAnswerType.EVENT, QuestionerAnswerType.VALUE}
        if draft.answer_type is None or (
            expected_types is not None and draft.answer_type not in expected_types
        ):
            codes.append(CandidateRejectionCode.ANSWER_TYPE_MISMATCH)
        return list(dict.fromkeys(codes))
