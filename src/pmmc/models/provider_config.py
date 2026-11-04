"""Opt-in Step 8 transport policy; legacy registry entries remain unchanged."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import model_validator

from pmmc.schemas import GenerationRequest, ThinkingMode
from pmmc.schemas.answer_output import TYPED_ANSWER_CONTRACT_VERSION, TYPED_ANSWER_OUTPUT_TOKENS
from pmmc.schemas.base import BaseSchema, PositiveInt

ProviderKind = Literal["qwen_vllm", "openai", "anthropic"]
_ANSWER_POLICY_VERSION = "step8-typed-answer-provider-output-v1"


@dataclass(frozen=True)
class ProviderCapabilities:
    api: str
    strict_json: bool
    native_images: bool
    sampling: bool
    seed: bool
    reasoning: Literal["off", "low"]


PROVIDER_CAPABILITIES: dict[ProviderKind, ProviderCapabilities] = {
    "qwen_vllm": ProviderCapabilities("chat_completions", True, True, True, True, "off"),
    "openai": ProviderCapabilities("responses", True, True, False, False, "low"),
    "anthropic": ProviderCapabilities("messages", True, True, False, False, "off"),
}

# Read old registries without relabelling historical calls. Dispatch rejects these IDs.
RETIRED_PROVIDER_MODELS = frozenset({
    "gpt-5-mini-2025-08-07", "claude-haiku-4-5-20251001",
})

PROVIDER_MODELS: dict[ProviderKind, frozenset[str]] = {
    "qwen_vllm": frozenset({"Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"}),
    "openai": frozenset({"gpt-6-luna"}),
    "anthropic": frozenset({"claude-haiku-5-5"}),
}


class ProviderConfig(BaseSchema):
    """No endpoints or credentials; hashed with the concrete model identity.

    max_output_tokens is the transport ceiling, including reasoning/JSON overhead.
    It is deliberately independent of the public answer's 512-token payload cap.
    """

    kind: ProviderKind
    reasoning_mode: Literal["off", "minimal", "low"]
    max_output_tokens: PositiveInt

    @property
    def capabilities(self) -> ProviderCapabilities:
        return PROVIDER_CAPABILITIES[self.kind]

    @property
    def typed_answer_output_tokens(self) -> int:
        """Frozen v3 answer total; helper-role ceilings remain independent."""
        return 4096 if self.reasoning_mode in {"minimal", "low"} else TYPED_ANSWER_OUTPUT_TOKENS

    @property
    def thinking_mode(self) -> ThinkingMode:
        return {"minimal": ThinkingMode.MINIMAL, "low": ThinkingMode.LOW}.get(
            self.reasoning_mode, ThinkingMode.DISABLED
        )

    @model_validator(mode="after")
    def _validate_policy(self) -> ProviderConfig:
        legacy = self.kind == "openai" and self.reasoning_mode == "minimal"
        if not legacy and self.reasoning_mode != self.capabilities.reasoning:
            raise ValueError("Step 8 requires GPT low reasoning and Qwen/Claude off")
        return self


def _is_typed_answer(request: GenerationRequest) -> bool:
    return request.metadata.get(
        "answer_contract_version"
    ) == TYPED_ANSWER_CONTRACT_VERSION and request.metadata.get("model_role") in {
        "answer",
        "answer_format_retry",
    }


def _answer_policy(config: ProviderConfig) -> dict[str, object]:
    return {
        "version": (
            _ANSWER_POLICY_VERSION
            if config.reasoning_mode != "low"
            else "step8-typed-answer-provider-output-v2"
        ),
        "provider_config": config.model_dump(mode="json"),
        "public_thinking_mode": ThinkingMode.DISABLED.value,
    }


def provider_output_budget(
    request: GenerationRequest,
    ceiling: int,
    *,
    config: ProviderConfig | None = None,
) -> int:
    """Validate the same output reservation at assembly, accounting and dispatch."""
    # Qwen has thinking disabled: its registry ceiling is not a per-call reserve.
    # In particular, a 4K Questioner request must not become a 16K completion.
    budget = request.provider_max_output_tokens
    if budget is None:
        budget = request.max_output_tokens if config and config.kind == "qwen_vllm" else ceiling
    if config is not None and _is_typed_answer(request):
        expected_mode = config.thinking_mode
        if (
            request.max_output_tokens != TYPED_ANSWER_OUTPUT_TOKENS
            or request.provider_max_output_tokens != config.typed_answer_output_tokens
            or request.thinking_mode is not expected_mode
            or request.metadata.get("provider_answer_policy") != _answer_policy(config)
        ):
            raise ValueError("typed answer requires explicit frozen public/provider output budgets")
    if budget > ceiling or budget < request.max_output_tokens:
        raise ValueError("provider output budget is outside the configured ceiling/public budget")
    return budget


def prepare_provider_answer_request(
    request: GenerationRequest,
    config: ProviderConfig,
) -> GenerationRequest:
    """Opt in only the explicit v3 answer role, before any context token counting."""
    if not _is_typed_answer(request):
        return request
    output = config.typed_answer_output_tokens
    if request.provider_max_output_tokens not in {None, output}:
        raise ValueError("typed answer provider output budget conflicts with the frozen policy")
    policy = _answer_policy(config)
    declared = request.metadata.get("provider_answer_policy")
    if declared is not None and declared != policy:
        raise ValueError("typed answer provider policy conflicts with the registry")
    prepared = request.model_copy(
        update={
            "provider_max_output_tokens": output,
            "thinking_mode": config.thinking_mode,
            "metadata": {**request.metadata, "provider_answer_policy": policy},
        }
    )
    provider_output_budget(prepared, config.max_output_tokens, config=config)
    return prepared


def validate_typed_answer_decoding(request: GenerationRequest) -> None:
    """Accept neutral no-CoT input or a checked provider-only projection.

    This checks the projection's internal contract. Dispatch/accounting additionally
    compare it with the actual registry config; request metadata cannot override it.
    """
    policy = request.metadata.get("provider_answer_policy")
    if policy is None:
        if (
            request.thinking_mode is not ThinkingMode.DISABLED
            or request.provider_max_output_tokens is not None
        ):
            raise ValueError("typed answer decoding requires a neutral or explicit provider policy")
        return
    if not _is_typed_answer(request) or not isinstance(policy, dict):
        raise ValueError("invalid typed answer provider projection")
    config = ProviderConfig.model_validate(policy.get("provider_config"))
    provider_output_budget(request, config.max_output_tokens, config=config)
