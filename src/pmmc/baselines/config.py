"""Strict, versioned configuration for the two full-context baselines."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class QueryImagePolicy(StrEnum):
    """How query-provided images are handled by a full-context baseline."""

    PRESERVE = "preserve"


class CaptionPolicy(StrEnum):
    """Allowed source of visual text in the caption baseline."""

    BENCHMARK_ONLY = "benchmark_only"


class FullContextBaselineConfig(BaseModel):
    """Immutable experiment contract shared by both direct-context methods."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    protocol_version: Literal["full-context-protocol-v1"] = "full-context-protocol-v1"
    renderer_version: Literal["full-context-renderer-v1"] = "full-context-renderer-v1"
    prompt_version: Literal["full-context-answer-v1", "pmmc-answer-contract-v3"] = (
        "full-context-answer-v1"
    )
    answer_contract_version: Literal["typed-answer-contract-v3"] | None = None
    prompt_path: Path = Path("prompts/baselines/full_context_answer_v1.txt")
    answer_model_alias: str = Field(default="answer-qwen3.5-9b", min_length=1)
    total_context_tokens: int = Field(default=32768, ge=2048)
    reserved_output_tokens: int = Field(default=1024, ge=1)
    max_input_tokens: int = Field(default=31744, ge=1)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, ge=0, le=1)
    seed: int = Field(default=0, ge=0)
    max_output_tokens: int = Field(default=1024, ge=1)
    job_max_attempts: int = Field(default=2, ge=1)
    job_retry_delay_seconds: float = Field(default=5.0, ge=0)
    concurrency: int = Field(default=8, ge=1)
    query_image_policy: QueryImagePolicy = QueryImagePolicy.PRESERVE
    caption_policy: CaptionPolicy = CaptionPolicy.BENCHMARK_ONLY

    @model_validator(mode="before")
    @classmethod
    def _typed_defaults(cls, value):
        if (
            isinstance(value, dict)
            and value.get("answer_contract_version") == "typed-answer-contract-v3"
        ):
            return {
                "prompt_version": "pmmc-answer-contract-v3",
                "prompt_path": "prompts/runtime/pmmc_answer_contract_v3.txt",
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
        return result

    @model_validator(mode="after")
    def _validate_budget(self) -> FullContextBaselineConfig:
        if self.answer_contract_version is not None and (
            self.max_output_tokens != 512 or self.prompt_version != "pmmc-answer-contract-v3"
            or self.prompt_path != Path("prompts/runtime/pmmc_answer_contract_v3.txt")
        ):
            raise ValueError("v3 requires its frozen prompt and 512 output tokens")
        if self.max_input_tokens != self.total_context_tokens - self.reserved_output_tokens:
            raise ValueError("max_input_tokens must equal total minus reserved output")
        if self.max_output_tokens != self.reserved_output_tokens:
            raise ValueError("max_output_tokens must equal reserved_output_tokens")
        if self.temperature != 0.0 or self.top_p != 1.0:
            raise ValueError("Step 5B fixes temperature=0 and top_p=1")
        return self

    def resolved_prompt_path(self, project_root: Path) -> Path:
        return (
            self.prompt_path
            if self.prompt_path.is_absolute()
            else project_root / self.prompt_path
        )


def load_full_context_config(
    path: Path,
    *,
    answer_model_alias: str | None = None,
) -> FullContextBaselineConfig:
    """Load strict YAML through OmegaConf and apply the one supported CLI override."""

    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        import yaml  # type: ignore[import-untyped]

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("full-context config root must be a mapping")
    if answer_model_alias is not None:
        raw = {**raw, "answer_model_alias": answer_model_alias}
    return FullContextBaselineConfig.model_validate(raw)


def load_full_context_prompt(config: FullContextBaselineConfig, project_root: Path) -> str:
    """Read the immutable English answer instruction and reject empty prompts."""

    prompt = config.resolved_prompt_path(project_root).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("full-context answer prompt cannot be empty")
    return prompt


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
