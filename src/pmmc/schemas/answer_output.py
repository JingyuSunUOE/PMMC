"""Strict structured answer contracts shared by PMMC and Native-MM fallback."""

from __future__ import annotations

import re
from collections.abc import Sequence
from copy import deepcopy
from enum import StrEnum
from typing import Final, Literal

from pydantic import Field, StrictBool, model_validator

from pmmc.schemas.base import (
    ANSWER_OUTPUT_SCHEMA_VERSION,
    BaseSchema,
    JsonDict,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)
from pmmc.schemas.memory import ImageRef


class AnswerAbstainReason(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED_BY_EVIDENCE = "unsupported_by_evidence"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"
    UNSAFE_REQUEST = "unsafe_request"


class AnswerContractFailure(StrEnum):
    INVALID_JSON = "invalid_json"
    SCHEMA_INVALID = "schema_invalid"
    FINISH_REASON_LENGTH = "finish_reason_length"
    THINKING_TAG_LEAK = "thinking_tag_leak"
    EVIDENCE_ALIAS_LEAK = "evidence_alias_leak"
    INTERNAL_ID_LEAK = "internal_id_leak"
    HARNESS_LEAK = "harness_leak"
    URL_LEAK = "url_leak"
    UNKNOWN_EVIDENCE_ALIAS = "unknown_evidence_alias"
    EMPTY_SUPPORTED_ANSWER = "empty_supported_answer"
    CONTRACT_IDENTITY_MISMATCH = "contract_identity_mismatch"
    MODEL_CALL_LINEAGE_MISSING = "model_call_lineage_missing"
    PROVIDER_SCHEMA_NONCONFORMANT = "provider_schema_nonconformant"
    UNSUPPORTED_FINAL_RECORD = "unsupported_final_record"
    UNKNOWN_IMAGE_ALIAS = "unknown_image_alias"
    OUTPUT_LIMIT_EXCEEDED = "output_limit_exceeded"
    INVALID_FINISH_REASON = "invalid_finish_reason"


class StructuredAnswerOutput(BaseSchema):
    """The only provider payload eligible for a Step 7B final answer."""

    answer: NonEmptyStr | None
    supported: bool
    used_evidence_aliases: list[NonEmptyStr] = Field(default_factory=list)
    abstain_reason: AnswerAbstainReason | None = None

    @model_validator(mode="after")
    def _validate_support_contract(self) -> StructuredAnswerOutput:
        ensure_unique_strings(self.used_evidence_aliases, "used_evidence_aliases")
        if self.supported:
            if self.answer is None or self.abstain_reason is not None:
                raise ValueError("supported answers require text and cannot abstain")
            if not self.used_evidence_aliases:
                raise ValueError("supported answers require at least one evidence alias")
        elif self.answer is not None or self.abstain_reason is None:
            raise ValueError("unsupported answers require null text and an abstain reason")
        return self


TYPED_ANSWER_CONTRACT_VERSION: Final = "typed-answer-contract-v3"
TYPED_ANSWER_GUIDED_SCHEMA_VERSION = "typed-answer-guided-json-v3"
TYPED_ANSWER_OUTPUT_TOKENS = 512
GROUNDING_POLICY_VERSION: Final = "source-only-answer-grounding-v1"


class AnswerKind(StrEnum):
    TEXT = "text"
    NUMBER = "number"
    BOOLEAN = "boolean"
    IMAGE = "image"
    IMAGE_SET = "image_set"
    ORDERED_IMAGE_LIST = "ordered_image_list"
    ABSTENTION = "abstention"


IMAGE_ANSWER_KINDS = frozenset({
    AnswerKind.IMAGE, AnswerKind.IMAGE_SET, AnswerKind.ORDERED_IMAGE_LIST,
})


class AnswerContractV3(BaseSchema):
    """Provider-facing payload. IDs, URLs and private lineage are never answer values."""

    answer_kind: AnswerKind
    answer_text: NonEmptyStr | None
    selected_image_aliases: list[NonEmptyStr]
    supported: StrictBool
    used_evidence_aliases: list[NonEmptyStr]
    abstain_reason: AnswerAbstainReason | None

    @model_validator(mode="after")
    def _typed_contract(self) -> AnswerContractV3:
        ensure_unique_strings(self.selected_image_aliases, "selected_image_aliases")
        ensure_unique_strings(self.used_evidence_aliases, "used_evidence_aliases")
        if any(not re.fullmatch(r"I[0-9]{2,}", a) for a in self.selected_image_aliases):
            raise ValueError("selected images require native local image aliases")
        if any(not re.fullmatch(r"[AE][0-9]{2,}", a) for a in self.used_evidence_aliases):
            raise ValueError("evidence requires native local evidence aliases")
        if self.answer_kind is AnswerKind.ABSTENTION:
            if self.supported or self.answer_text is not None or self.selected_image_aliases:
                raise ValueError("abstention cannot contain a supported answer")
            if self.abstain_reason is None:
                raise ValueError("abstention requires a reason")
            return self
        if not self.supported or self.abstain_reason is not None or not self.used_evidence_aliases:
            raise ValueError("supported answers require evidence and no abstain reason")
        if self.answer_kind in IMAGE_ANSWER_KINDS:
            if self.answer_text is not None or not self.selected_image_aliases:
                raise ValueError("image answers require selected images and null text")
            if self.answer_kind is AnswerKind.IMAGE and len(self.selected_image_aliases) != 1:
                raise ValueError("single image answers require exactly one image")
        elif self.answer_text is None or self.selected_image_aliases:
            raise ValueError("textual answers require text and no selected images")
        if self.answer_kind is AnswerKind.NUMBER and not re.fullmatch(
            r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", self.answer_text or "",
        ):
            raise ValueError("number answers require a JSON numeric literal")
        if self.answer_kind is AnswerKind.BOOLEAN and self.answer_text not in {"true", "false"}:
            raise ValueError("boolean answers require true or false")
        return self


class AnswerSourceImage(BaseSchema):
    image_id: NonEmptyStr
    caption: NonEmptyStr | None = None
    image: ImageRef | None = None

    @model_validator(mode="after")
    def _image_identity(self) -> AnswerSourceImage:
        if self.image is not None and self.image.image_id != self.image_id:
            raise ValueError("source image identity mismatch")
        if self.image is None and self.caption is None:
            raise ValueError("selectable images require visible pixels or a source caption")
        return self


class AnswerEvidenceSource(BaseSchema):
    evidence_alias: NonEmptyStr
    evidence_id: NonEmptyStr
    text: NonEmptyStr | None = None
    images: list[AnswerSourceImage] = Field(default_factory=list)

    @model_validator(mode="after")
    def _source(self) -> AnswerEvidenceSource:
        if not re.fullmatch(r"[AE][0-9]{2,}", self.evidence_alias):
            raise ValueError("invalid native evidence alias")
        if not self.text and not self.images:
            raise ValueError("answer evidence must contain actual source content")
        ensure_unique_strings([i.image_id for i in self.images], "source images")
        return self


class AnswerImageBinding(BaseSchema):
    image_alias: NonEmptyStr
    image_id: NonEmptyStr
    public_media_ref: NonEmptyStr = Field(pattern=r"^media:[a-f0-9]{64}$")
    evidence_aliases: list[NonEmptyStr]


class TypedAnswerContext(BaseSchema):
    sources: list[AnswerEvidenceSource]
    image_bindings: list[AnswerImageBinding]

    @model_validator(mode="after")
    def _mapping(self) -> TypedAnswerContext:
        ensure_unique_strings([s.evidence_alias for s in self.sources], "source aliases")
        ensure_unique_strings([s.evidence_id for s in self.sources], "source IDs")
        ensure_unique_strings([i.image_alias for i in self.image_bindings], "image aliases")
        ensure_unique_strings([i.image_id for i in self.image_bindings], "image IDs")
        sources_by_image: dict[str, list[str]] = {}
        for source in self.sources:
            for image in source.images:
                sources_by_image.setdefault(image.image_id, []).append(source.evidence_alias)
        if [i.image_id for i in self.image_bindings] != list(sources_by_image):
            raise ValueError("image mappings must exactly cover source images in encounter order")
        for ordinal, binding in enumerate(self.image_bindings, 1):
            if binding.image_alias != f"I{ordinal:02d}":
                raise ValueError("image aliases must be contiguous native aliases")
            if binding.evidence_aliases != sources_by_image[binding.image_id]:
                raise ValueError("image mapping evidence ownership mismatch")
        return self


class TypedAnswerImageItem(BaseSchema):
    ordinal: NonNegativeInt
    image_alias: NonEmptyStr
    image_id: NonEmptyStr
    public_media_ref: NonEmptyStr = Field(pattern=r"^media:[a-f0-9]{64}$")


class AnswerRoleCall(BaseSchema):
    role: Literal["answer", "answer_format_retry", "answer_grounding", "grounding_format_retry"]
    model_call_id: NonEmptyStr
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    cost_usd: NonNegativeFloat | None = None
    latency_ms: NonNegativeFloat | None = None


class GroundingCheck(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class GroundingAssessment(BaseSchema):
    supported: StrictBool
    query_closure: GroundingCheck
    entity_consistency: GroundingCheck
    time_consistency: GroundingCheck
    set_consistency: GroundingCheck
    source_support: GroundingCheck

    @model_validator(mode="after")
    def _judgement(self) -> GroundingAssessment:
        if GroundingCheck.NOT_APPLICABLE in (self.query_closure, self.source_support):
            raise ValueError("query closure and source support must be assessed")
        checks = [getattr(self, f) for f in type(self).model_fields if f != "supported"]
        if self.supported != (GroundingCheck.FAIL not in checks):
            raise ValueError("grounding support must agree with dimensional checks")
        return self


class GroundingFact(BaseSchema):
    grounding_id: NonEmptyStr
    run_id: NonEmptyStr
    query_id: NonEmptyStr
    answer_model_call_id: NonEmptyStr
    policy_version: Literal["source-only-answer-grounding-v1"] = GROUNDING_POLICY_VERSION
    assessment_source: Literal["model_assessed"] = "model_assessed"
    model_alias: NonEmptyStr
    actual_model_id: NonEmptyStr
    assessment: GroundingAssessment
    used_evidence_aliases: list[NonEmptyStr]
    evidence_content_hash: NonEmptyStr
    ungated_output_hash: NonEmptyStr
    calls: list[AnswerRoleCall]
    created_at: UtcDateTime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _call_lineage(self) -> GroundingFact:
        ensure_unique_strings(self.used_evidence_aliases, "grounding evidence aliases")
        ensure_unique_strings([c.model_call_id for c in self.calls], "grounding model calls")
        if [c.role for c in self.calls] not in [
            ["answer_grounding"], ["answer_grounding", "grounding_format_retry"],
        ]:
            raise ValueError("grounding requires one audited call and at most one format retry")
        if self.answer_model_call_id in {c.model_call_id for c in self.calls}:
            raise ValueError("grounding is a separate model call")
        return self


class TypedAnswerResult(BaseSchema):
    typed_answer_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr
    method: NonEmptyStr
    answer_variant: NonEmptyStr
    contract_version: Literal["typed-answer-contract-v3"] = TYPED_ANSWER_CONTRACT_VERSION
    output: AnswerContractV3
    ungated_output: AnswerContractV3
    image_items: list[TypedAnswerImageItem]
    ungated_image_items: list[TypedAnswerImageItem]
    answer_model_call_id: NonEmptyStr
    actual_model_id: NonEmptyStr
    terminal_state: Literal["answered", "abstained", "fallback_required"]
    grounding: GroundingFact | None = None
    calls: list[AnswerRoleCall]
    format_failure_codes: list[list[AnswerContractFailure]] = Field(default_factory=list)
    created_at: UtcDateTime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _terminal(self) -> TypedAnswerResult:
        ensure_unique_strings([c.model_call_id for c in self.calls], "answer role calls")
        if self.output.supported != (self.terminal_state == "answered"):
            raise ValueError("typed terminal state must agree with support")
        if self.terminal_state == "fallback_required" and self.method != "pmmc":
            raise ValueError("baseline grounding cannot trigger PMMC fallback")
        for output, items in [(self.output, self.image_items),
                              (self.ungated_output, self.ungated_image_items)]:
            if [i.image_alias for i in items] != output.selected_image_aliases:
                raise ValueError("typed image items must match selected aliases and order")
            if [i.ordinal for i in items] != list(range(len(items))):
                raise ValueError("typed image ordinals must be contiguous from zero")
        answer_calls = [c for c in self.calls if c.role in {"answer", "answer_format_retry"}]
        if not answer_calls or answer_calls[-1].model_call_id != self.answer_model_call_id:
            raise ValueError("typed answer requires audited answer call lineage")
        if self.ungated_output.supported:
            if self.grounding is None:
                raise ValueError("supported answer cannot be terminal without grounding")
            if self.output.supported != self.grounding.assessment.supported:
                raise ValueError("typed support must agree with the grounding gate")
            if (
                self.grounding.run_id != self.run_id or self.grounding.query_id != self.query_id
                or self.grounding.answer_model_call_id != self.answer_model_call_id
                or any(c not in self.calls for c in self.grounding.calls)
            ):
                raise ValueError("typed result and grounding lineage disagree")
        elif self.grounding is not None or self.output != self.ungated_output:
            raise ValueError("legal model abstention is terminal without grounding")
        return self

    def public_payload(self) -> JsonDict:
        """Explicit projection: never serialize private IDs or evidence aliases publicly."""
        return {
            "answer_kind": self.output.answer_kind.value,
            "answer_text": self.output.answer_text,
            "images": [i.public_media_ref for i in self.image_items],
            "supported": self.output.supported,
            "abstain_reason": self.output.abstain_reason,
        }


class TypedAnswerCandidate(BaseSchema):
    """Validated generation journal, explicitly not a completed scientific answer."""

    run_id: NonEmptyStr
    query_id: NonEmptyStr
    output: AnswerContractV3
    image_items: list[TypedAnswerImageItem]
    answer_model_call_id: NonEmptyStr
    calls: list[AnswerRoleCall]
    format_failure_codes: list[list[AnswerContractFailure]] = Field(default_factory=list)
    state: Literal["generated_awaiting_finalization"] = "generated_awaiting_finalization"


def typed_answer_response_schema(context: TypedAnswerContext) -> JsonDict:
    """Native image and evidence aliases are different grammar alphabets."""
    schema = AnswerContractV3.model_json_schema()
    aliases = [s.evidence_alias for s in context.sources]
    images = [i.image_alias for i in context.image_bindings]
    branches = []
    for kind in AnswerKind:
        supported = kind is not AnswerKind.ABSTENTION
        if supported and (not aliases or (kind in IMAGE_ANSWER_KINDS and not images)):
            continue
        branch = deepcopy(schema)
        branch.pop("$defs", None)
        props = branch["properties"]
        props["answer_kind"] = {"type": "string", "const": kind.value}
        props["supported"] = {"type": "boolean", "const": supported}
        props["answer_text"] = (
            {"type": "null"} if kind in IMAGE_ANSWER_KINDS or not supported
            else {"type": "string"}
        )
        if kind is AnswerKind.BOOLEAN:
            props["answer_text"]["enum"] = ["true", "false"]
        props["abstain_reason"] = (
            {"type": "null"} if supported else {"$ref": "#/$defs/AnswerAbstainReason"}
        )
        props["used_evidence_aliases"] = {
            "type": "array", "items": {"type": "string", **({"enum": aliases} if aliases else {})},
            "uniqueItems": True, "minItems": int(supported), "maxItems": len(aliases),
        }
        props["selected_image_aliases"] = {
            "type": "array", "items": {"type": "string", **({"enum": images} if images else {})},
            "uniqueItems": True, "minItems": int(kind in IMAGE_ANSWER_KINDS),
            "maxItems": (1 if kind is AnswerKind.IMAGE else len(images))
            if kind in IMAGE_ANSWER_KINDS else 0,
        }
        branches.append(branch)
    return {"type": "object", "$defs": schema.get("$defs", {}), "anyOf": branches}


ANSWER_GUIDED_SCHEMA_VERSION = "answer-guided-json-string-v2"


def answer_response_schema(allowed_aliases: Sequence[str] | None = None) -> JsonDict:
    """Guide the same support alternatives checked by the runtime validator.

    Uniqueness and stripped nonempty text remain server-side hard gates.
    The two complete object branches keep cross-field rules in the grammar.
    """
    aliases = None if allowed_aliases is None else list(allowed_aliases)
    if aliases is not None:
        ensure_unique_strings(aliases, "allowed_aliases")
        if any(not alias or alias != alias.strip() for alias in aliases):
            raise ValueError("answer evidence aliases must be nonempty normalized strings")
    schema = StructuredAnswerOutput.model_json_schema()
    definitions = schema.pop("$defs", {})
    branches = []
    for supported in (True, False):
        if supported and aliases == []:
            continue
        branch = deepcopy(schema)
        branch["required"] = list(StructuredAnswerOutput.model_fields)
        properties = branch["properties"]
        properties["supported"] = {"type": "boolean", "const": supported}
        properties["answer"] = (
            # Use the engine's JSON escape rules. A broad pattern can consume
            # quotes/control bytes; length-constrained strings also differ
            # from the builtin string grammar in supported XGrammar versions.
            {"type": "string"}
            if supported else {"type": "null"}
        )
        properties["abstain_reason"] = (
            {"type": "null"} if supported else {"$ref": "#/$defs/AnswerAbstainReason"}
        )
        items: JsonDict = {"type": "string", "minLength": 1}
        if aliases:
            items["enum"] = aliases
        alias_schema: JsonDict = {
            "type": "array", "items": items, "uniqueItems": True,
            "minItems": 1 if supported else 0,
        }
        if aliases is not None:
            alias_schema["maxItems"] = len(aliases)
        properties["used_evidence_aliases"] = alias_schema
        branches.append(branch)
    return {"type": "object", "$defs": definitions, "anyOf": branches}


class StructuredAnswerFact(BaseSchema):
    structured_answer_output_id: NonEmptyStr
    run_id: NonEmptyStr
    query_id: NonEmptyStr
    answer_variant: NonEmptyStr
    attempt_ordinal: NonNegativeInt
    model_call_id: NonEmptyStr
    answer_id: NonEmptyStr | None = None
    answer: NonEmptyStr | None
    supported: bool
    used_evidence_aliases: list[NonEmptyStr] = Field(default_factory=list)
    abstain_reason: AnswerAbstainReason | None = None
    parse_status: NonEmptyStr
    provider_schema_conformant: bool
    finish_reason: NonEmptyStr | None = None
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_persisted_output(self) -> StructuredAnswerFact:
        if self.attempt_ordinal > 1 or self.parse_status not in {"accepted", "rejected"}:
            raise ValueError("answer output attempt or parse status is invalid")
        StructuredAnswerOutput(
            answer=self.answer,
            supported=self.supported,
            used_evidence_aliases=self.used_evidence_aliases,
            abstain_reason=self.abstain_reason,
        )
        return self


class AnswerFormatDiagnostic(BaseSchema):
    diagnostic_id: NonEmptyStr
    run_id: NonEmptyStr
    query_id: NonEmptyStr
    answer_variant: NonEmptyStr
    attempt_ordinal: NonNegativeInt
    schema_version: NonEmptyStr = ANSWER_OUTPUT_SCHEMA_VERSION
    failure_codes: list[AnswerContractFailure]
    provider_schema_conformant: bool
    accepted: bool
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_diagnostic(self) -> AnswerFormatDiagnostic:
        ensure_unique_strings([item.value for item in self.failure_codes], "failure_codes")
        if self.accepted == bool(self.failure_codes):
            raise ValueError("accepted answer attempts cannot have failures")
        return self


class AnswerReuseLink(BaseSchema):
    answer_reuse_link_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    query_id: NonEmptyStr
    source_answer_id: NonEmptyStr
    target_variant: NonEmptyStr
    context_hash: NonEmptyStr
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)


__all__ = [
    "AnswerContractV3",
    "AnswerKind",
    "AnswerEvidenceSource",
    "AnswerSourceImage",
    "TypedAnswerContext",
    "TypedAnswerImageItem",
    "TypedAnswerResult",
    "GroundingAssessment",
    "GroundingFact",
    "AnswerRoleCall",
    "AnswerAbstainReason",
    "AnswerContractFailure",
    "AnswerFormatDiagnostic",
    "AnswerReuseLink",
    "StructuredAnswerFact",
    "StructuredAnswerOutput",
]
