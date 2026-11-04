"""Strict configuration for paired baseline evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaselineEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    protocol_version: Literal["baseline-evaluation-v1"] = "baseline-evaluation-v1"
    normalizer_version: Literal["answer-normalizer-v1"] = "answer-normalizer-v1"
    evidence_policy_version: Literal["evidence-resolution-v1"] = "evidence-resolution-v1"
    bad_case_rule_version: Literal["bad-case-rules-v1"] = "bad-case-rules-v1"
    judge_prompt_version: Literal["baseline-judge-v1"] = "baseline-judge-v1"
    judge_rubric_version: Literal["baseline-judge-rubric-v1"] = (
        "baseline-judge-rubric-v1"
    )
    judge_prompt_path: Path = Path("prompts/evaluation/baseline_judge_v1.txt")
    judge_model_alias: str = Field(default="judge-qwen3.5-9b", min_length=1)
    judge_repeat_count: Literal[1] = 1
    judge_score_values: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0])
    secondary_judge_enabled: Literal[False] = False
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, ge=0, le=1)
    seed: Literal[0] = 0
    max_output_tokens: int = Field(default=512, ge=1, le=1024)
    retrieval_ks: list[int] = Field(default_factory=lambda: [1, 5, 10, 20, 50])
    concurrency: int = Field(default=8, ge=1)
    job_max_attempts: int = Field(default=2, ge=1)
    job_retry_delay_seconds: float = Field(default=5.0, ge=0)

    @model_validator(mode="after")
    def _validate_protocol(self) -> BaselineEvaluationConfig:
        if self.judge_score_values != [0.0, 0.5, 1.0]:
            raise ValueError("baseline judge scores must be [0, 0.5, 1]")
        if self.retrieval_ks != [1, 5, 10, 20, 50]:
            raise ValueError("main evaluation requires Recall@1/5/10/20/50")
        if self.temperature != 0.0 or self.top_p != 1.0:
            raise ValueError("baseline judge decoding must be deterministic")
        return self

    def resolved_judge_prompt_path(self, project_root: Path) -> Path:
        if self.judge_prompt_path.is_absolute():
            return self.judge_prompt_path
        return project_root / self.judge_prompt_path


def load_evaluation_config(path: Path) -> BaselineEvaluationConfig:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        import yaml  # type: ignore[import-untyped]

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("evaluation config root must be a mapping")
    return BaselineEvaluationConfig.model_validate(raw)


def load_judge_prompt(config: BaselineEvaluationConfig, project_root: Path) -> str:
    prompt = config.resolved_judge_prompt_path(project_root).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("baseline judge prompt cannot be empty")
    return prompt


def evaluation_prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
