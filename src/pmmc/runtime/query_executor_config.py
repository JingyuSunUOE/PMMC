"""Frozen configuration and identity for the Step 7 online query executor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from pmmc.schemas.base import BaseSchema, NonEmptyStr, PositiveInt, Probability

JOINT_POLICY = "visible-image-owner-text-v2"
JOINT_THRESHOLD = 0.30


class QueryExecutorConfig(BaseSchema):
    protocol_version: NonEmptyStr = "pmmc-query-executor-v1"
    matcher_version: NonEmptyStr = "pmmc-qbank-matcher-v1"
    graph_version: NonEmptyStr = "pmmc-online-query-graph-v1"
    sufficiency_policy_version: NonEmptyStr = "query-runtime-structural-v1"
    answer_prompt_version: NonEmptyStr = "pmmc-answer-with-evidence-v1"
    answer_prompt_path: Path = Path("prompts/runtime/pmmc_answer_with_evidence_v1.txt")
    fallback_policy: NonEmptyStr = "native_mm_rag_v1"
    match_top_k: PositiveInt = 50
    rrf_constant: PositiveInt = 60
    text_query_weights: dict[NonEmptyStr, float] = Field(
        default_factory=lambda: {"lexical": 0.5, "text": 0.5}
    )
    multimodal_query_weights: dict[NonEmptyStr, float] = Field(
        default_factory=lambda: {
            "lexical": 0.3333333333,
            "text": 0.3333333333,
            "multimodal": 0.3333333334,
        }
    )
    active_match_threshold: Probability = 0.60
    active_gap_threshold: Probability = 0.05
    threshold_grid: list[Probability] = Field(default_factory=lambda: [0.50, 0.60, 0.70, 0.80])
    gap_grid: list[Probability] = Field(default_factory=lambda: [0.00, 0.05, 0.10])
    question_budgets: list[PositiveInt] = Field(default_factory=lambda: [1, 2, 4, 8])
    rounds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    active_round: int = Field(default=2, ge=0, le=3)
    question_budget: PositiveInt = 8
    matcher_workers: PositiveInt = 32
    executor_workers: PositiveInt = 32
    database_semaphore: PositiveInt = 24
    model_concurrency: PositiveInt = 64
    max_answer_tokens: PositiveInt = 4096
    reranker_version: NonEmptyStr | None = None
    reranker_prompt_version: NonEmptyStr | None = None
    reranker_model_alias: NonEmptyStr | None = None
    reranker_top_k: PositiveInt = 20
    base_score_weight: Probability = 0.40
    reranker_score_weight: Probability = 0.60
    lexical_term_limit: PositiveInt = 12
    query_evidence_relevance_threshold: Probability = 0.55
    multimodal_evidence_relevance_threshold: Probability = 0.30
    answer_contract_version: NonEmptyStr | None = None
    answer_output_schema_version: NonEmptyStr | None = None
    answer_format_retry_policy: NonEmptyStr = "answer-format-feedback-v1"
    answer_format_retry_sha256: NonEmptyStr | None = None
    relevance_cosine_grid: list[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    relevance_score_transform: NonEmptyStr = "raw-cosine-v1"
    joint_relevance_policy: Literal["visible-image-owner-text-v2"] | None = None
    joint_relevance_threshold: float | None = None
    grounding_failure_policy: NonEmptyStr = "fallback"
    experiment_seed: int = 0
    embedding_ownership: NonEmptyStr = "runtime"
    treatment: Literal["main", "fixed_planner", "no_raw_image", "no_reranker"] = "main"
    no_reranker_match_threshold: Probability = 0.80
    no_reranker_gap_threshold: Probability = 0.05

    @model_serializer(mode="wrap")
    def _serialize_identity(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        payload = handler(self)
        if self.joint_relevance_policy is None:
            payload.pop("joint_relevance_policy", None)
            payload.pop("joint_relevance_threshold", None)
        if not self.is_v3:
            for key in ("treatment", "no_reranker_match_threshold", "no_reranker_gap_threshold"):
                payload.pop(key, None)
            for key in (
                "relevance_cosine_grid",
                "relevance_score_transform",
                "grounding_failure_policy",
                "experiment_seed",
                "embedding_ownership",
                "multimodal_evidence_relevance_threshold",
            ):
                payload.pop(key, None)
        if self.answer_prompt_version != "pmmc-answer-contract-v2.1":
            payload.pop("answer_format_retry_policy", None)
            payload.pop("answer_format_retry_sha256", None)
        return payload

    @property
    def is_v2(self) -> bool:
        return self.protocol_version == "pmmc-query-executor-v2"

    @property
    def is_v3(self) -> bool:
        return self.protocol_version in {"pmmc-query-executor-v3", "pmmc-query-executor-release-r"}

    @property
    def joint_relevance_enabled(self) -> bool:
        return self.joint_relevance_policy == JOINT_POLICY and self.treatment != "no_raw_image"

    @property
    def uses_reranker(self) -> bool:
        return self.is_v2 or self.is_v3

    @property
    def reranker_enabled(self) -> bool:
        return self.uses_reranker and not (self.is_v3 and self.treatment == "no_reranker")

    @property
    def route_match_threshold(self) -> float:
        return (
            self.no_reranker_match_threshold
            if self.is_v3 and self.treatment == "no_reranker"
            else self.active_match_threshold
        )

    @property
    def route_gap_threshold(self) -> float:
        return (
            self.no_reranker_gap_threshold
            if self.is_v3 and self.treatment == "no_reranker"
            else self.active_gap_threshold
        )

    @model_validator(mode="after")
    def _validate_frozen_protocol(self) -> QueryExecutorConfig:
        if not self.is_v3 and self.treatment != "main":
            raise ValueError("treatments require runtime v3")
        if (
            self.no_reranker_match_threshold not in self.threshold_grid
            or self.no_reranker_gap_threshold not in self.gap_grid
        ):
            raise ValueError("no-reranker calibration must use the declared development grid")
        identity = (
            self.protocol_version,
            self.matcher_version,
            self.graph_version,
            self.sufficiency_policy_version,
            self.answer_prompt_version,
            self.fallback_policy,
        )
        expected_v1 = (
            "pmmc-query-executor-v1",
            "pmmc-qbank-matcher-v1",
            "pmmc-online-query-graph-v1",
            "query-runtime-structural-v1",
            "pmmc-answer-with-evidence-v1",
            "native_mm_rag_v1",
        )
        expected_v2 = (
            "pmmc-query-executor-v2",
            "pmmc-qbank-matcher-v2",
            "pmmc-online-query-graph-v2",
            "query-runtime-evidence-v2",
            "pmmc-answer-contract-v2",
            "native_mm_rag_v1",
        )
        expected_v21 = (*expected_v2[:4], "pmmc-answer-contract-v2.1", expected_v2[5])
        expected_v3 = (
            "pmmc-query-executor-v3",
            "pmmc-qbank-matcher-v3",
            "pmmc-online-query-graph-v3",
            "query-runtime-evidence-v3",
            "pmmc-answer-contract-v3",
            "native_mm_rag_v1",
        )
        expected_r = ("pmmc-query-executor-release-r", *expected_v3[1:])
        if identity not in {expected_v1, expected_v2, expected_v21, expected_v3, expected_r}:
            raise ValueError("query executor identities are frozen by protocol version")
        if (self.joint_relevance_policy, self.joint_relevance_threshold) != (
            (JOINT_POLICY, JOINT_THRESHOLD) if identity == expected_r else (None, None)
        ):
            raise ValueError("release R requires its independent frozen joint relevance policy")
        if identity == expected_v21:
            from pmmc.runtime.answer_policy import CURRENT_RETRY_POLICY, CURRENT_RETRY_SHA256

            if (
                self.answer_format_retry_policy != CURRENT_RETRY_POLICY
                or self.answer_format_retry_sha256 != CURRENT_RETRY_SHA256
            ):
                raise ValueError("answer prompt and retry policy must share a frozen identity")
        elif (
            self.answer_format_retry_policy != "answer-format-feedback-v1"
            or self.answer_format_retry_sha256 is not None
        ):
            raise ValueError("historical answer retry identity cannot change")
        if self.match_top_k != 50 or self.rrf_constant != 60:
            raise ValueError("query executor freezes top-50 and RRF constant 60")
        expected_text = (
            {"lexical": 0.40, "text": 0.60} if self.uses_reranker else {"lexical": 0.5, "text": 0.5}
        )
        if self.text_query_weights != expected_text:
            raise ValueError("text-query channel weights are frozen by protocol")
        expected_mm = (
            {"lexical": 0.20, "text": 0.35, "multimodal": 0.45}
            if self.uses_reranker
            else {
                "lexical": 0.3333333333,
                "text": 0.3333333333,
                "multimodal": 0.3333333334,
            }
        )
        if self.multimodal_query_weights != expected_mm:
            raise ValueError("multimodal-query channel weights are frozen")
        if abs(sum(self.text_query_weights.values()) - 1.0) > 1e-9:
            raise ValueError("text-query channel weights must sum to one")
        if abs(sum(self.multimodal_query_weights.values()) - 1.0) > 1e-9:
            raise ValueError("multimodal-query channel weights must sum to one")
        if self.threshold_grid != [0.50, 0.60, 0.70, 0.80]:
            raise ValueError("Step 7 threshold grid is frozen")
        if self.gap_grid != [0.00, 0.05, 0.10]:
            raise ValueError("Step 7 gap grid is frozen")
        if self.active_match_threshold not in self.threshold_grid:
            raise ValueError("active match threshold must be in the frozen grid")
        if self.active_gap_threshold not in self.gap_grid:
            raise ValueError("active gap threshold must be in the frozen grid")
        if self.question_budgets != [1, 2, 4, 8] or self.rounds != [0, 1, 2, 3]:
            raise ValueError("Step 7 budget/round projection grid is frozen")
        if self.active_round != 2 or self.question_budget != 8:
            raise ValueError("Step 7 active method is r2/b8")
        if (
            self.matcher_workers != 32
            or self.executor_workers != 32
            or self.database_semaphore != 24
            or self.model_concurrency != 64
        ):
            raise ValueError("Step 7 execution concurrency is frozen")
        if self.uses_reranker:
            if (
                self.reranker_version != "pmmc-bounded-reranker-v1"
                or self.reranker_prompt_version != "pmmc-qbank-reranker-v1"
                or self.reranker_model_alias != "reranker-qwen3.5-9b"
                or self.reranker_top_k != 20
                or self.base_score_weight != 0.40
                or self.reranker_score_weight != 0.60
                or self.lexical_term_limit != 12
                or self.answer_contract_version
                != ("typed-answer-contract-v3" if self.is_v3 else "strict-answer-contract-v2")
                or self.answer_output_schema_version != ("0.2.0" if self.is_v3 else "0.1.0")
                or self.max_answer_tokens != 512
            ):
                raise ValueError("Step 7B reranker and answer contracts are frozen")
            if abs(self.base_score_weight + self.reranker_score_weight - 1.0) > 1e-9:
                raise ValueError("base and reranker score weights must sum to one")
        elif self.max_answer_tokens != 4096:
            raise ValueError("Step 7 answer output budget is frozen at 4096")
        if self.is_v3:
            if (
                self.relevance_cosine_grid != [0.1, 0.2, 0.3, 0.4, 0.5]
                or self.query_evidence_relevance_threshold not in self.relevance_cosine_grid
                or self.multimodal_evidence_relevance_threshold not in self.relevance_cosine_grid
                or self.relevance_score_transform != "raw-cosine-v1"
                or self.grounding_failure_policy not in {"fallback", "abstain"}
                or self.embedding_ownership != "runtime"
            ):
                raise ValueError("v3 requires the declared raw-cosine grid and runtime embeddings")
        return self

    def resolved_answer_prompt_path(self, project_root: Path) -> Path:
        path = (
            self.answer_prompt_path
            if self.answer_prompt_path.is_absolute()
            else project_root / self.answer_prompt_path
        )
        if not path.is_file():
            raise FileNotFoundError(f"Step 7 answer prompt is missing: {path}")
        if self.is_v2:
            from pmmc.runtime.answer_policy import load_answer_policy_prompt

            expected = load_answer_policy_prompt(project_root, self.answer_prompt_version)
            if path.read_text(encoding="utf-8").strip() != expected:
                raise ValueError("query answer prompt path does not match frozen version")
        return path


def load_query_executor_config(path: Path) -> QueryExecutorConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("query executor config must contain a YAML mapping")
    return QueryExecutorConfig.model_validate(raw)


def query_executor_config_hash(config: QueryExecutorConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def query_executor_protocol_hash(config: QueryExecutorConfig) -> str:
    """Hash the executor protocol without the review-selected route thresholds."""

    payload = config.model_dump(
        mode="json",
        exclude={"active_match_threshold", "active_gap_threshold"},
    )
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def query_executor_routing_policy_hash(
    config: QueryExecutorConfig,
    *,
    match_threshold: float,
    gap_threshold: float,
) -> str:
    """Bind a reviewed threshold pair to an otherwise frozen executor protocol."""

    if match_threshold not in config.threshold_grid:
        raise ValueError("match threshold is outside the frozen executor grid")
    if gap_threshold not in config.gap_grid:
        raise ValueError("gap threshold is outside the frozen executor grid")
    return routing_policy_hash_from_protocol(
        query_executor_protocol_hash(config),
        match_threshold=match_threshold,
        gap_threshold=gap_threshold,
    )


def routing_policy_hash_from_protocol(
    protocol_hash: str,
    *,
    match_threshold: float,
    gap_threshold: float,
) -> str:
    """Build the reviewed policy identity from an archived protocol hash."""

    if len(protocol_hash) != 64 or any(
        character not in "0123456789abcdef" for character in protocol_hash
    ):
        raise ValueError("query executor protocol hash is not a SHA256 value")
    payload = {
        "query_executor_protocol_hash": protocol_hash,
        "match_threshold": match_threshold,
        "gap_threshold": gap_threshold,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def apply_approved_routing_thresholds(
    config: QueryExecutorConfig,
    *,
    match_threshold: float,
    gap_threshold: float,
) -> QueryExecutorConfig:
    """Materialize the approved policy while re-running all config validation."""

    payload = config.model_dump(mode="python")
    payload.update(
        {
            "active_match_threshold": match_threshold,
            "active_gap_threshold": gap_threshold,
        }
    )
    return QueryExecutorConfig.model_validate(payload)


__all__ = [
    "QueryExecutorConfig",
    "apply_approved_routing_thresholds",
    "load_query_executor_config",
    "query_executor_config_hash",
    "query_executor_protocol_hash",
    "query_executor_routing_policy_hash",
    "routing_policy_hash_from_protocol",
]
