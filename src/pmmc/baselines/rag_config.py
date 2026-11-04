"""Strict, versioned configuration for the three retrieval baselines."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from pmmc.schemas import BaselineMethod, DistanceMetric, RepresentationKind

QWEN_MEMORY_QUERY_INSTRUCTION = (
    "Instruct: Given a user query about a conversation history, retrieve the memory "
    "passages needed to answer it.\nQuery: {query}"
)


class RagBaselineConfig(BaseModel):
    """Immutable main-experiment contract shared by all three RAG methods."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    protocol_version: Literal["rag-protocol-v1"] = "rag-protocol-v1"
    renderer_version: Literal["rag-renderer-v1"] = "rag-renderer-v1"
    answer_prompt_version: Literal["full-context-answer-v1", "pmmc-answer-contract-v3"] = (
        "full-context-answer-v1"
    )
    answer_contract_version: Literal["typed-answer-contract-v3"] | None = None
    memory_raw_image_policy: Literal["allow", "forbid"] = "allow"
    answer_prompt_path: Path = Path("prompts/baselines/full_context_answer_v1.txt")
    query_instruction_version: Literal["qwen-memory-retrieval-query-v1"] = (
        "qwen-memory-retrieval-query-v1"
    )
    query_instruction_template: str = QWEN_MEMORY_QUERY_INSTRUCTION

    answer_model_alias: str = Field(default="answer-qwen3.5-9b", min_length=1)
    caption_text_embedding_alias: str = Field(default="qwen3-embedding-8b", min_length=1)
    contrastive_embedding_alias: str = Field(
        default="siglip2-so400m-patch14-384", min_length=1
    )
    native_multimodal_embedding_alias: str = Field(
        default="qwen3-vl-embedding-8b", min_length=1
    )

    retrieval_top_k: int = Field(default=50, ge=1)
    answer_top_k: int = Field(default=5, ge=1)
    distance_metric: Literal[DistanceMetric.COSINE] = DistanceMetric.COSINE
    exact_scan: Literal[True] = True
    use_reranker: Literal[False] = False
    contrastive_text_weight: float = Field(default=0.5, ge=0, le=1)
    contrastive_image_weight: float = Field(default=0.5, ge=0, le=1)

    total_context_tokens: int = Field(default=32768, ge=2048)
    reserved_output_tokens: int = Field(default=1024, ge=1)
    max_input_tokens: int = Field(default=31744, ge=1)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, ge=0, le=1)
    seed: int = Field(default=0, ge=0)
    max_output_tokens: int = Field(default=1024, ge=1)

    caption_text_batch_size: int = Field(default=32, ge=1)
    contrastive_batch_size: int = Field(default=32, ge=1)
    native_multimodal_batch_size: int = Field(default=8, ge=1)
    embedding_concurrency: int = Field(default=8, ge=1)
    answer_concurrency: int = Field(default=8, ge=1)
    job_max_attempts: int = Field(default=2, ge=1)
    job_retry_delay_seconds: float = Field(default=5.0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _typed_defaults(cls, value):
        if (
            isinstance(value, dict)
            and value.get("answer_contract_version") == "typed-answer-contract-v3"
        ):
            return {
                "answer_prompt_version": "pmmc-answer-contract-v3",
                "answer_prompt_path": "prompts/runtime/pmmc_answer_contract_v3.txt",
                "reserved_output_tokens": 512, "max_output_tokens": 512,
                "max_input_tokens": int(value.get("total_context_tokens", 32768)) - 512,
                **value,
            }
        return value

    @model_serializer(mode="wrap")
    def _historical_serialization(self, handler):
        result = handler(self)
        if self.answer_contract_version is None:
            result.pop("answer_contract_version", None)
        if self.memory_raw_image_policy == "allow":
            result.pop("memory_raw_image_policy", None)
        return result

    @model_validator(mode="after")
    def _validate_protocol(self) -> RagBaselineConfig:
        if self.memory_raw_image_policy == "forbid" and self.answer_contract_version is None:
            raise ValueError("no_raw memory projection requires explicit typed v3")
        if self.answer_contract_version is not None and (
            self.max_output_tokens != 512 or self.answer_prompt_version != "pmmc-answer-contract-v3"
            or self.answer_prompt_path != Path("prompts/runtime/pmmc_answer_contract_v3.txt")
        ):
            raise ValueError("v3 requires its frozen prompt and 512 output tokens")
        if self.query_instruction_template != QWEN_MEMORY_QUERY_INSTRUCTION:
            raise ValueError("main RAG protocol requires the frozen Qwen query instruction")
        if self.max_input_tokens != self.total_context_tokens - self.reserved_output_tokens:
            raise ValueError("max_input_tokens must equal total minus reserved output")
        if self.max_output_tokens != self.reserved_output_tokens:
            raise ValueError("max_output_tokens must equal reserved_output_tokens")
        if self.answer_top_k > self.retrieval_top_k:
            raise ValueError("answer_top_k cannot exceed retrieval_top_k")
        if self.contrastive_text_weight != 0.5 or self.contrastive_image_weight != 0.5:
            raise ValueError("main RAG protocol requires equal contrastive modality weights")
        if self.temperature != 0.0 or self.top_p != 1.0:
            raise ValueError("main RAG protocol requires deterministic decoding")
        aliases = {
            self.answer_model_alias,
            self.caption_text_embedding_alias,
            self.contrastive_embedding_alias,
            self.native_multimodal_embedding_alias,
        }
        if len(aliases) != 4:
            raise ValueError("RAG answer and embedding model aliases must be distinct")
        return self

    def resolved_prompt_path(self, project_root: Path) -> Path:
        return (
            self.answer_prompt_path
            if self.answer_prompt_path.is_absolute()
            else project_root / self.answer_prompt_path
        )

    def embedding_alias_for(self, method: BaselineMethod) -> str:
        if method is BaselineMethod.CAPTION_TEXT_RAG:
            return self.caption_text_embedding_alias
        if method is BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG:
            return self.contrastive_embedding_alias
        if method is BaselineMethod.NATIVE_MULTIMODAL_RAG:
            return self.native_multimodal_embedding_alias
        raise ValueError(f"not a RAG baseline method: {method.value}")

    def representations_for(self, method: BaselineMethod) -> tuple[RepresentationKind, ...]:
        if self.memory_raw_image_policy == "forbid":
            if method not in {
                BaselineMethod.CAPTION_TEXT_RAG, BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG,
                BaselineMethod.NATIVE_MULTIMODAL_RAG,
            }:
                raise ValueError("no_raw projection requires a RAG method")
            return (RepresentationKind.TEXT_CAPTION,)
        if method is BaselineMethod.CAPTION_TEXT_RAG:
            return (RepresentationKind.TEXT_CAPTION,)
        if method is BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG:
            return (RepresentationKind.TEXT, RepresentationKind.IMAGE)
        if method is BaselineMethod.NATIVE_MULTIMODAL_RAG:
            return (RepresentationKind.MULTIMODAL,)
        raise ValueError(f"not a RAG baseline method: {method.value}")

    def embedding_batch_size_for(self, method: BaselineMethod) -> int:
        if method is BaselineMethod.CAPTION_TEXT_RAG:
            return self.caption_text_batch_size
        if method is BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG:
            return self.contrastive_batch_size
        if method is BaselineMethod.NATIVE_MULTIMODAL_RAG:
            return self.native_multimodal_batch_size
        raise ValueError(f"not a RAG baseline method: {method.value}")


def load_rag_config(
    path: Path,
    *,
    answer_model_alias: str | None = None,
) -> RagBaselineConfig:
    """Load a strict YAML config and apply the supported answer-model override."""

    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        import yaml  # type: ignore[import-untyped]

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("RAG config root must be a mapping")
    if answer_model_alias is not None:
        raw = {**raw, "answer_model_alias": answer_model_alias}
    return RagBaselineConfig.model_validate(raw)


def render_qwen_query_instruction(config: RagBaselineConfig, raw_query: str) -> str:
    query = raw_query.strip()
    if not query:
        raise ValueError("retrieval query cannot be empty")
    return config.query_instruction_template.format(query=query)


def rag_prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def query_instruction_sha256(config: RagBaselineConfig) -> str:
    return rag_prompt_sha256(config.query_instruction_template)
