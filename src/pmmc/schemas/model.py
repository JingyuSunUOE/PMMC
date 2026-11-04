"""Provider-neutral model I/O contracts used by PMMC gateways."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    BaseSchema,
    FiniteFloat,
    JsonDict,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Probability,
    SpeakerRole,
    ensure_unique_strings,
)
from pmmc.schemas.memory import ImageRef


class ModelCapability(StrEnum):
    """Capabilities declared by one immutable model-registry entry."""

    CHAT_TEXT = "chat_text"
    CHAT_MULTIMODAL = "chat_multimodal"
    EMBED_TEXT = "embed_text"
    EMBED_IMAGE = "embed_image"
    EMBED_MULTIMODAL = "embed_multimodal"


class ModelEndpointKind(StrEnum):
    """Supported provider endpoint protocols."""

    LITELLM_OPENAI = "litellm_openai"
    VLLM_OPENAI = "vllm_openai"
    VLLM_V2_EMBED = "vllm_v2_embed"
    PMMC_CONTRASTIVE = "pmmc_contrastive"
    FAKE = "fake"


class ModelContentKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class ThinkingMode(StrEnum):
    """Provider-neutral control for visible/internal reasoning generation."""

    PROVIDER_DEFAULT = "provider_default"
    DISABLED = "disabled"
    MINIMAL = "minimal"
    LOW = "low"


class ModelContentPart(BaseSchema):
    """One text or canonical-image message part without embedded image bytes."""

    kind: ModelContentKind
    text: NonEmptyStr | None = None
    image: ImageRef | None = None
    detail: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_content(self) -> ModelContentPart:
        if self.kind is ModelContentKind.TEXT:
            if self.text is None or self.image is not None or self.detail is not None:
                raise ValueError("text content requires text and cannot include image fields")
        elif self.image is None or self.text is not None:
            raise ValueError("image content requires image and cannot include text")
        return self


class ModelMessage(BaseSchema):
    """Provider-neutral chat message."""

    role: SpeakerRole
    parts: list[ModelContentPart]
    name: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_parts(self) -> ModelMessage:
        if not self.parts:
            raise ValueError("model messages require at least one content part")
        if self.role is SpeakerRole.UNKNOWN:
            raise ValueError("unknown is not a valid model-message role")
        return self


class TokenUsage(BaseSchema):
    """Normalized token usage returned by a model endpoint."""

    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    total_tokens: NonNegativeInt = 0
    image_tokens: NonNegativeInt | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_total(self) -> TokenUsage:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        return self


class GenerationRequest(BaseSchema):
    """One deterministic chat/LVLM generation request."""

    request_id: NonEmptyStr
    model_alias: NonEmptyStr
    messages: list[ModelMessage]
    max_output_tokens: PositiveInt
    provider_max_output_tokens: PositiveInt | None = Field(
        default=None, exclude_if=lambda v: v is None,
    )
    temperature: NonNegativeFloat = 0.0
    top_p: Probability = 1.0
    seed: NonNegativeInt | None = None
    timeout_seconds: NonNegativeFloat | None = None
    response_format: JsonDict | None = None
    thinking_mode: ThinkingMode = ThinkingMode.PROVIDER_DEFAULT
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_messages(self) -> GenerationRequest:
        if not self.messages:
            raise ValueError("generation requests require at least one message")
        if (
            self.provider_max_output_tokens is not None
            and self.provider_max_output_tokens < self.max_output_tokens
        ):
            raise ValueError("provider output budget cannot be below the public output budget")
        return self


class GenerationResult(BaseSchema):
    """Normalized generation result plus provider audit metadata."""

    request_id: NonEmptyStr
    model_alias: NonEmptyStr
    actual_model_id: NonEmptyStr
    text: str
    usage: TokenUsage
    provider_request_id: NonEmptyStr | None = None
    finish_reason: NonEmptyStr | None = None
    latency_ms: NonNegativeFloat | None = None
    cost_usd: NonNegativeFloat | None = None
    raw_response: JsonDict = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)


class EmbeddingInput(BaseSchema):
    """Text, image, or mixed input identified independently of list position."""

    input_id: NonEmptyStr
    text: NonEmptyStr | None = None
    image_refs: list[ImageRef] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_input(self) -> EmbeddingInput:
        if self.text is None and not self.image_refs:
            raise ValueError("embedding input requires text or at least one image")
        ensure_unique_strings(
            [image.image_id for image in self.image_refs],
            "image_refs.image_id",
        )
        return self


class EmbeddingRequest(BaseSchema):
    """Provider-neutral batched embedding request."""

    request_id: NonEmptyStr
    model_alias: NonEmptyStr
    inputs: list[EmbeddingInput]
    timeout_seconds: NonNegativeFloat | None = None
    dimensions: PositiveInt | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_inputs(self) -> EmbeddingRequest:
        if not self.inputs:
            raise ValueError("embedding requests require at least one input")
        ensure_unique_strings([item.input_id for item in self.inputs], "inputs.input_id")
        return self


class EmbeddingVector(BaseSchema):
    """One finite embedding vector keyed by request input ID."""

    input_id: NonEmptyStr
    vector: list[FiniteFloat]

    @model_validator(mode="after")
    def _validate_vector(self) -> EmbeddingVector:
        if not self.vector:
            raise ValueError("embedding vectors cannot be empty")
        return self


class EmbeddingResult(BaseSchema):
    """Normalized embedding response."""

    request_id: NonEmptyStr
    model_alias: NonEmptyStr
    actual_model_id: NonEmptyStr
    vectors: list[EmbeddingVector]
    usage: TokenUsage | None = None
    provider_request_id: NonEmptyStr | None = None
    latency_ms: NonNegativeFloat | None = None
    cost_usd: NonNegativeFloat | None = None
    raw_response: JsonDict = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_vectors(self) -> EmbeddingResult:
        if not self.vectors:
            raise ValueError("embedding results require at least one vector")
        ensure_unique_strings([item.input_id for item in self.vectors], "vectors.input_id")
        dimensions = {len(item.vector) for item in self.vectors}
        if len(dimensions) != 1:
            raise ValueError("all vectors in one embedding result must share a dimension")
        return self
