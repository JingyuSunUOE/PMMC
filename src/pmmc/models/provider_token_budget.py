"""Offline, conservative Step 8 context reservations, not provider token usage.

No tokenizer, credentials, Hub access, or provider API is needed. Complete image
metadata needs no bytes; an optional resolver recovers missing geometry from
SHA256-verified original bytes without modifying the request.
Text reserves one unit per UTF-8 byte. Images reserve unresized patch areas with
deliberately inflated multipliers and floors, never a resizing/detail discount.
These are versioned experiment policy estimates, NOT exact counts or a proof of
the provider's private serialization. Provider-reported usage remains authoritative.

Vision policy references (Anthropic re-reviewed for Haiku 5.5 on 2026-10-07):
https://developers.openai.com/api/docs/guides/images-vision
https://platform.claude.com/docs/en/build-with-claude/vision
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from pmmc.models.protocols import TokenCounter
from pmmc.models.provider_config import prepare_provider_answer_request, provider_output_budget
from pmmc.models.provider_schema import project_provider_response_format
from pmmc.models.registry import ModelRegistryEntry
from pmmc.schemas import (
    GenerationRequest,
    ImageRef,
    ModelCapability,
    ModelContentKind,
    SpeakerRole,
)

PROVIDER_BUDGET_VERSION = "step8-provider-context-budget-v1"
SHARED_CONTEXT_TOKENS = 32768
_REQUEST_OVERHEAD = 1024
_MESSAGE_OVERHEAD = 32
_PART_OVERHEAD = 16
_MAX_IMAGE_DIMENSION = 8000
_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
# Independent of the registry allowlist. Luna retains an explicitly conservative
# experiment reservation, not a claimed provider tokenizer or exact vision formula.
_IMAGE_POLICIES = {
    ("openai", "gpt-6-luna"): (32, 3, 4096),
    # Haiku 5.5 uses 28px visual patches, including native high-resolution images.
    # Reserve unresized patches with 2x headroom; never apply the old 1568-token cap.
    ("anthropic", "claude-haiku-5-5"): (28, 2, 2048),
}


@dataclass(frozen=True)
class ProviderConfiguredTokenCounter:
    """Carry Step 8 provider policy without replacing or changing an exact counter."""

    counter: TokenCounter
    entry: ModelRegistryEntry

    def __getattr__(self, name: str) -> Any:
        # Preserve optional image/audit hooks as well as the TokenCounter protocol.
        return getattr(self.counter, name)


def prepare_context_request(counter: TokenCounter, request: GenerationRequest) -> GenerationRequest:
    if isinstance(counter, (ProviderConfiguredTokenCounter, ProviderTokenBudgetCounter)):
        if request.model_alias != counter.entry.alias:
            raise ValueError("provider counter does not match the context request model alias")
        config = counter.entry.provider_config
        assert config is not None
        return prepare_provider_answer_request(request, config)
    return request


@dataclass(frozen=True)
class ProviderRequestBudget:
    """Auditable reservation units, deliberately separate from TokenUsage."""

    estimated_input_tokens: int
    provider_output_tokens: int
    public_output_tokens: int

    @property
    def budgeted_input_tokens(self) -> int:
        # Existing assemblers subtract only the public output allowance. Charge
        # the remaining transport/reasoning allowance on the input side once.
        return self.estimated_input_tokens + self.provider_output_tokens - self.public_output_tokens

    @property
    def total_tokens(self) -> int:
        return self.estimated_input_tokens + self.provider_output_tokens

    @property
    def fits(self) -> bool:
        return self.total_tokens <= SHARED_CONTEXT_TOKENS


@dataclass(frozen=True)
class ProviderTokenBudgetCounter:
    """TokenCounter-compatible closed-provider budget with ``estimated=True``.

    ``count_request`` returns reservation units, including the output supplement,
    and intentionally permits over-budget values so assemblers can prune history.
    Call ``validate_request`` on the final request before dispatch. Image metadata
    must be verified against the bytes by the existing native transport resolver,
    which also rejects animated or malformed GIFs without changing their bytes.
    """

    entry: ModelRegistryEntry
    image_dimensions: Callable[[ImageRef], tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        config = self.entry.provider_config
        if config is None or (config.kind, self.entry.provider_model) not in _IMAGE_POLICIES:
            raise ValueError("no reviewed conservative budget policy for this provider/model")
        if (
            self.entry.model_revision != self.entry.provider_model
            or self.entry.expected_actual_model_id != self.entry.provider_model
        ):
            raise ValueError("provider budget requires a pinned actual model identity")
        if self.entry.context_window is None or self.entry.context_window < SHARED_CONTEXT_TOKENS:
            raise ValueError("provider budget requires at least the shared 32768 context")
        if not 0 < config.max_output_tokens < SHARED_CONTEXT_TOKENS - _REQUEST_OVERHEAD:
            raise ValueError("provider output ceiling leaves no shared input budget")
        self.entry.require(ModelCapability.CHAT_TEXT)

    @property
    def alias(self) -> str:
        return self.entry.alias

    @property
    def revision(self) -> str:
        payload = json.dumps(self.budget_policy, sort_keys=True, separators=(",", ":"))
        return f"{PROVIDER_BUDGET_VERSION}:{sha256(payload.encode('utf-8')).hexdigest()}"

    @property
    def estimated(self) -> bool:
        return True

    @property
    def budget_policy(self) -> dict[str, object]:
        config = self.entry.provider_config
        assert config is not None
        patch, multiplier, floor = _IMAGE_POLICIES[config.kind, self.entry.provider_model]
        return {
            "version": PROVIDER_BUDGET_VERSION,
            "estimated": True,
            "provider": config.kind,
            "model_id": self.entry.provider_model,
            "registry_entry_hash": self.entry.resolved_hash,
            "total_context_tokens": SHARED_CONTEXT_TOKENS,
            "text_policy": "one-unit-per-utf8-byte",
            "request_overhead": _REQUEST_OVERHEAD,
            "message_overhead": _MESSAGE_OVERHEAD,
            "part_overhead": _PART_OVERHEAD,
            "schema_policy": "max-canonical-projected-json-utf8-bytes",
            "image_policy": "max-floor-unresized-patches-no-detail-discount",
            "image_patch_size": patch,
            "image_patch_multiplier": multiplier,
            "image_floor": floor,
            "image_max_dimension": _MAX_IMAGE_DIMENSION,
            "image_media_types": sorted(_IMAGE_MEDIA_TYPES),
            "image_metadata_policy": "positive-dimensions-mime-sha256-required",
            "provider_output_ceiling": config.max_output_tokens,
            "typed_answer_provider_output_tokens": config.typed_answer_output_tokens,
            "output_policy": "reserve-transport-minus-public-on-input-side",
        }

    def count_text(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def count_image(self, *, width: int | None = None, height: int | None = None) -> int:
        config = self.entry.provider_config
        assert config is not None
        self.entry.require(ModelCapability.CHAT_MULTIMODAL)
        if not config.capabilities.native_images:
            raise ValueError("provider does not support native images")
        if any(
            type(value) is not int or not 1 <= value <= _MAX_IMAGE_DIMENSION
            for value in (width, height)
        ):
            raise ValueError("provider image dimensions must be integers in [1, 8000]")
        assert width is not None and height is not None
        patch, multiplier, floor = _IMAGE_POLICIES[config.kind, self.entry.provider_model]
        return max(
            floor, ((width + patch - 1) // patch) * ((height + patch - 1) // patch) * multiplier
        )

    def count_image_ref(self, image: ImageRef) -> int:
        if image.mime_type not in _IMAGE_MEDIA_TYPES:
            raise ValueError("provider image budget requires JPEG, PNG, GIF, or WebP mime_type")
        if image.sha256 is None or re.fullmatch(r"[0-9a-fA-F]{64}", image.sha256) is None:
            raise ValueError("provider image budget requires a canonical SHA256")
        width, height = image.width, image.height
        if (width is None or height is None) and self.image_dimensions is not None:
            width, height = self.image_dimensions(image)
            if ((image.width is not None and image.width != width)
                    or (image.height is not None and image.height != height)):
                raise ValueError("native image dimensions mismatch")
        return self.count_image(width=width, height=height)

    def request_budget(self, request: GenerationRequest) -> ProviderRequestBudget:
        if request.model_alias != self.alias:
            raise ValueError("provider budget counter does not match request model alias")
        config = self.entry.provider_config
        assert config is not None
        output = provider_output_budget(request, config.max_output_tokens, config=config)
        total = _REQUEST_OVERHEAD
        for message in request.messages:
            if message.role not in {SpeakerRole.SYSTEM, SpeakerRole.USER, SpeakerRole.ASSISTANT}:
                raise ValueError("provider budget supports only system/user/assistant messages")
            if message.name is not None:
                raise ValueError("provider budget does not support named messages")
            total += _MESSAGE_OVERHEAD + self.count_text(message.role.value)
            for part in message.parts:
                total += _PART_OVERHEAD
                if part.kind is ModelContentKind.TEXT:
                    total += self.count_text(part.text or "")
                elif part.kind is ModelContentKind.IMAGE:
                    if message.role is not SpeakerRole.USER:
                        raise ValueError("provider images must occur in user messages")
                    if part.detail not in {None, "auto", "low", "high"}:
                        raise ValueError("unsupported provider image detail")
                    assert part.image is not None
                    total += self.count_image_ref(part.image)
                else:
                    raise ValueError("unsupported provider content kind")
        if config.kind == "anthropic":
            conversation = [m for m in request.messages if m.role is not SpeakerRole.SYSTEM]
            if not conversation or any(
                m.role is not SpeakerRole.USER for m in (conversation[0], conversation[-1])
            ):
                raise ValueError("Anthropic conversation must start and end with a user message")
        if request.response_format is not None:
            projected, _ = project_provider_response_format(request.response_format, config.kind)
            total += max(
                self.count_text(json.dumps(value, ensure_ascii=True, sort_keys=True))
                for value in (request.response_format, projected)
            )
        return ProviderRequestBudget(total, output, request.max_output_tokens)

    def count_request(self, request: GenerationRequest) -> int:
        return self.request_budget(request).budgeted_input_tokens

    def validate_request(self, request: GenerationRequest) -> ProviderRequestBudget:
        budget = self.request_budget(request)
        if not budget.fits:
            raise ValueError("request exceeds the conservative shared 32768 context budget")
        return budget


def validate_context_budget(
    counter: TokenCounter,
    *,
    total_context_tokens: int,
    reserved_output_tokens: int,
    paper_mode: bool,
) -> None:
    """Explicit integration hook; historical estimated counters remain forbidden.

    Context assemblers must opt into this check in place of their exact-only guard;
    the factory never disguises estimates as exact to bypass that historical guard.
    """
    if not 0 < reserved_output_tokens < total_context_tokens:
        raise ValueError("invalid context/output budget")
    if isinstance(counter, ProviderTokenBudgetCounter):
        if total_context_tokens != SHARED_CONTEXT_TOKENS:
            raise ValueError("provider context policy is frozen at 32768 tokens")
        config = counter.entry.provider_config
        assert config is not None
        if reserved_output_tokens > config.max_output_tokens:
            raise ValueError("public output reserve exceeds the provider output ceiling")
    elif paper_mode and counter.estimated:
        raise ValueError("paper mode requires an exact counter or explicit provider budget policy")
