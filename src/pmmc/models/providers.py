"""Integrated Step 8 Qwen, OpenAI Responses, and Anthropic Messages adapters.

SDK retries are disabled. In particular, ambiguous failures never permit an
automatic retry: campaign reservation/reconciliation belongs to the caller.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

import httpx
from openai import AsyncOpenAI

from pmmc.models.gateways import ModelGatewayError, _openai_api_root
from pmmc.models.provider_config import RETIRED_PROVIDER_MODELS, provider_output_budget
from pmmc.models.provider_images import (
    ImageBytes,
    ImageBytesResolver,
    validate_image_bytes,
    verified_image_dimensions,
)
from pmmc.models.provider_schema import (
    project_provider_response_format,
    validate_canonical_output,
)
from pmmc.models.provider_token_budget import ProviderTokenBudgetCounter
from pmmc.models.registry import ModelRegistry
from pmmc.runtime.errors import ManagedRuntimeError
from pmmc.schemas import (
    GenerationRequest,
    GenerationResult,
    ModelCapability,
    ModelContentKind,
    SpeakerRole,
    ThinkingMode,
    TokenUsage,
)

if TYPE_CHECKING:
    from pmmc.runtime.step8_capacity_requests import CapacityRequestGuard, ExactTokenizer


class CapacityGatewayError(ModelGatewayError, ManagedRuntimeError):
    """Capacity rejection is terminal, including through the shared replica router."""


class ProviderCallError(ModelGatewayError):
    """Safe diagnostics plus any completed result needed for spend reconciliation."""

    def __init__(
        self,
        error_code: str,
        *,
        outcome: Literal["failed", "unknown", "completed"],
        provider_status_code: int | None = None,
        provider_request_id: str | None = None,
        result: GenerationResult | None = None,
        provider_error_category: str = "unclassified",
    ) -> None:
        super().__init__(error_code, provider_status_code=provider_status_code, retryable=False)
        self.outcome = outcome
        self.provider_request_id = provider_request_id
        self.result = result
        self.provider_error_category = provider_error_category
        self.requested_output_tokens: int | None = None
        self.provider_output_tokens: int | None = None
        self.provider_output_ceiling: int | None = None

    def safe_diagnostics(self) -> dict[str, object]:
        return {
            **super().safe_diagnostics(),
            "outcome": self.outcome,
            "provider_request_id": self.provider_request_id,
            "usage": self.result.usage.model_dump(mode="json") if self.result else None,
            "provider_error_category": self.provider_error_category,
            "requested_output_tokens": self.requested_output_tokens,
            "provider_output_tokens": self.provider_output_tokens,
            "provider_output_ceiling": self.provider_output_ceiling,
        }

    @classmethod
    def from_exception(cls, error: BaseException) -> ProviderCallError:
        status = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        if not isinstance(status, int):
            status = getattr(response, "status_code", None)
        request_id = getattr(error, "request_id", None)
        if request_id is None and response is not None:
            request_id = response.headers.get("request-id") or response.headers.get("x-request-id")
        # A 4xx rejection (except request timeout) confirms failure. A read/write
        # timeout, dropped connection, 5xx, or unfamiliar exception does not.
        failed = (isinstance(status, int) and 400 <= status < 500 and status != 408) or (
            isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
        )
        return cls(
            type(error).__name__,
            outcome="failed" if failed else "unknown",
            provider_status_code=status if isinstance(status, int) else None,
            provider_request_id=request_id,
            provider_error_category=_provider_error_category(error),
        )


def _provider_error_category(error: BaseException) -> str:
    """Classify SDK bodies without retaining echoed prompts, URLs, or credentials."""
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return "unclassified"
    body = body.get("error", body)
    if not isinstance(body, dict):
        return "unclassified"
    message = body.get("message")
    message = message[:8192].lower() if isinstance(message, str) else ""
    if body.get("code") == "context_length_exceeded" or any(
        phrase in message for phrase in ("maximum context length", "max model length")
    ):
        return "context_length_exceeded"
    if "max_tokens" in message and "context length" in message:
        return "context_length_exceeded"
    if any(phrase in message for phrase in ("json schema", "json_schema", "grammar")):
        return "structured_output_rejected"
    if any(phrase in message for phrase in ("image format", "decode image", "invalid image")):
        return "image_rejected"
    return "unclassified"


def _usage(raw: dict[str, Any], kind: str) -> TokenUsage:
    value = raw.get("usage") or {}
    if kind == "qwen_vllm":
        input_tokens = value.get("prompt_tokens", 0)
        output_tokens = value.get("completion_tokens", 0)
        cached = (value.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        reasoning = (value.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        creation = 0
    else:
        input_tokens = value.get("input_tokens", 0)
        output_tokens = value.get("output_tokens", 0)
        cached = (value.get("input_tokens_details") or {}).get("cached_tokens", 0)
        reasoning = (value.get("output_tokens_details") or {}).get("reasoning_tokens")
        creation = int((value.get("input_tokens_details") or {}).get("cache_write_tokens", 0) or 0)
        if kind == "anthropic":
            cached = value.get("cache_read_input_tokens", 0) or 0
            creation = value.get("cache_creation_input_tokens", 0) or 0
            input_tokens = (input_tokens or 0) + cached + creation
            reasoning = (
                None if any(
                    part.get("type") in {"thinking", "redacted_thinking"}
                    for part in raw.get("content", [])
                ) else 0
            )
    input_tokens, output_tokens = int(input_tokens or 0), int(output_tokens or 0)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        metadata={
            "reasoning_tokens": reasoning,
            "cached_input_tokens": cached,
            "cache_creation_input_tokens": creation,
            "uncached_input_tokens": input_tokens - int(cached or 0) - creation,
            "usage_reported": bool(value),
        },
    )


class ProviderGenerationGateway:
    """Common preflight/identity/validation boundary; transport is provider-specific."""

    kind: str

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        client: Any,
        image_bytes_resolver: ImageBytesResolver | None = None,
    ) -> None:
        self.registry = registry
        # Injected SDK clients must not reintroduce retries behind the spend ledger.
        self.client = (
            client.with_options(max_retries=0) if hasattr(client, "with_options") else client
        )
        self.image_bytes_resolver = image_bytes_resolver

    async def _create(self, create: Callable[..., Awaitable[Any]], kwargs: dict[str, Any]) -> Any:
        return await create(**kwargs)

    async def _messages(
        self, request: GenerationRequest, *, image_audit: list[dict[str, Any]] | None = None,
        resolved_images: dict[str, ImageBytes] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role not in {SpeakerRole.USER, SpeakerRole.ASSISTANT, SpeakerRole.SYSTEM}:
                raise ValueError("Step 8 providers support only system/user/assistant messages")
            if message.name is not None:
                raise ValueError("Step 8 providers do not support named messages")
            content: list[dict[str, Any]] = []
            for part in message.parts:
                if part.kind is ModelContentKind.TEXT:
                    part_type = "input_text" if self.kind == "openai" else "text"
                    content.append({"type": part_type, "text": part.text})
                    continue
                if message.role is not SpeakerRole.USER:
                    raise ValueError("native image content is supported only in user messages")
                if self.image_bytes_resolver is None:
                    raise ValueError("Step 8 images require an image_bytes_resolver")
                assert part.image is not None
                resolved = (resolved_images or {}).get(part.image.sha256 or "")
                payload = validate_image_bytes(
                    part.image,
                    resolved if resolved is not None else (
                        await self.image_bytes_resolver.resolve(part.image)),
                    allow_verified_mime_correction=getattr(
                        self.image_bytes_resolver, "allow_verified_mime_correction", False,
                    ),
                )
                corrected = bool(
                    part.image.mime_type and part.image.mime_type != payload.media_type
                )
                receipt = {
                    "policy": "sha256-verified-native-mime-v1",
                    "sha256": part.image.sha256,
                    "declared_mime": part.image.mime_type,
                    "transport_mime": payload.media_type,
                    "mime_corrected": corrected,
                    "bytes_modified": False,
                }
                if image_audit is not None:
                    image_audit.append(receipt)
                if corrected:
                    print(json.dumps({"event": "NATIVE_IMAGE_MIME_CORRECTED", **receipt},
                                     sort_keys=True), flush=True)
                detail = part.detail or "auto"
                if detail not in {"auto", "low", "high"}:
                    raise ValueError("unsupported image detail")
                if self.kind == "openai":
                    content.append(
                        {"type": "input_image", "image_url": payload.data_url, "detail": detail}
                    )
                elif self.kind == "anthropic":
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": payload.media_type,
                                "data": payload.base64,
                            },
                        }
                    )
                else:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": payload.data_url,
                                "detail": detail,
                            },
                        }
                    )
            messages.append({"role": message.role.value, "content": content})
        return messages

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        entry = self.registry.get(request.model_alias)
        if entry.provider_model in RETIRED_PROVIDER_MODELS:
            raise ValueError("retired provider model is read-only; plan a new campaign")
        config = entry.provider_config
        if config is None or config.kind != self.kind:
            raise ValueError("request alias does not match this provider adapter")
        has_images = any(
            p.kind is ModelContentKind.IMAGE for m in request.messages for p in m.parts
        )
        entry.require(ModelCapability.CHAT_MULTIMODAL if has_images else ModelCapability.CHAT_TEXT)
        if request.thinking_mode in {ThinkingMode.MINIMAL, ThinkingMode.LOW} and (
            request.thinking_mode is not config.thinking_mode
        ):
            raise ValueError("request reasoning mode differs from frozen provider policy")
        budget = provider_output_budget(request, config.max_output_tokens, config=config)
        resolved_images: dict[str, ImageBytes] = {}
        if config.kind in {"openai", "anthropic"}:
            dimensions: dict[str, tuple[int, int]] = {}
            for message in request.messages:
                for part in message.parts:
                    image = part.image
                    if image is None or (image.width is not None and image.height is not None):
                        continue
                    if not image.mime_type or not image.sha256:
                        raise ValueError("image dimension recovery requires mime_type and SHA256")
                    if self.image_bytes_resolver is None:
                        raise ValueError("Step 8 images require an image_bytes_resolver")
                    key = image.sha256 or ""
                    payload = resolved_images.get(key)
                    if payload is None:
                        payload = await self.image_bytes_resolver.resolve(image)
                    dimensions[key] = verified_image_dimensions(image, payload)
                    resolved_images[key] = payload
            ProviderTokenBudgetCounter(
                entry, lambda image: dimensions[image.sha256 or ""],
            ).validate_request(request)
        projection = None
        transport_format = None
        if request.response_format is not None:
            transport_format, projection = project_provider_response_format(
                request.response_format,
                config.kind,
            )
        image_audit: list[dict[str, Any]] = []
        messages = await self._messages(
            request, image_audit=image_audit, resolved_images=resolved_images)
        kwargs: dict[str, Any] = {
            "model": entry.provider_model,
            "extra_headers": {"X-Client-Request-Id": request.request_id},
        }
        if request.timeout_seconds is not None:
            if request.timeout_seconds <= 0:
                raise ValueError("provider timeout must be positive")
            kwargs["timeout"] = request.timeout_seconds
        if self.kind == "openai":
            kwargs.update(
                input=messages,
                max_output_tokens=budget,
                reasoning={"effort": config.reasoning_mode},
                store=False,
                truncation="disabled",
                service_tier="default",
            )
            if transport_format is not None:
                kwargs["text"] = {
                    "format": {"type": "json_schema", **transport_format["json_schema"]}
                }
            create = self.client.responses.create
        elif self.kind == "anthropic":
            system = [p for m in messages if m["role"] == "system" for p in m["content"]]
            conversation = [m for m in messages if m["role"] != "system"]
            if not conversation or conversation[0]["role"] != "user":
                raise ValueError("Anthropic messages must start with a user message")
            if conversation[-1]["role"] != "user":
                raise ValueError("Anthropic strict output does not support assistant prefill")
            kwargs.update(
                messages=conversation,
                max_tokens=budget,
                thinking={"type": "disabled"},
                output_config={"effort": "low"},
                service_tier="standard_only",
            )
            if system:
                kwargs["system"] = system
            if transport_format is not None:
                kwargs["output_config"]["format"] = {
                    "type": "json_schema",
                    "schema": transport_format["json_schema"]["schema"],
                }
            create = self.client.messages.create
        else:
            kwargs.update(
                messages=messages,
                max_tokens=budget,
                temperature=request.temperature,
                top_p=request.top_p,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            if request.seed is not None:
                kwargs["seed"] = request.seed
            if transport_format is not None:
                kwargs["response_format"] = transport_format
            create = self.client.chat.completions.create
        started = time.perf_counter()
        try:
            response = await self._create(create, kwargs)
        except ManagedRuntimeError as exc:
            code = str(exc)
            if not code.startswith("capacity_") or not code.replace("_", "").isalnum():
                code = "capacity_generation_failed"
            raise CapacityGatewayError(code, retryable=False) from None
        except Exception as exc:
            failure = ProviderCallError.from_exception(exc)
            failure.requested_output_tokens = request.max_output_tokens
            failure.provider_output_tokens = budget
            failure.provider_output_ceiling = config.max_output_tokens
            raise failure from None
        try:
            raw = response.model_dump(mode="json")
            text, finish_reason, provider_finish_reason = self._output(raw)
            request_id = getattr(response, "_request_id", None) or raw.get("id")
            result = GenerationResult(
                request_id=request.request_id,
                model_alias=request.model_alias,
                actual_model_id=raw.get("model") or "missing-provider-model-id",
                provider_request_id=request_id,
                text=text,
                finish_reason=finish_reason,
                usage=_usage(raw, self.kind),
                latency_ms=(time.perf_counter() - started) * 1000,
                raw_response=raw,
                metadata={
                    "provider": self.kind,
                    "native_image_transport": image_audit,
                    "provider_response_id": raw.get("id"),
                    "transport_model_id": raw.get("model"),
                    "model_identity_source": "provider_response",
                    "provider_finish_reason": provider_finish_reason,
                    "requested_thinking_mode": request.thinking_mode.value,
                    "effective_reasoning_mode": config.reasoning_mode,
                    "provider_effort": "low" if self.kind == "anthropic" else None,
                    "provider_service_tier": (
                        (raw.get("usage") or {}).get("service_tier")
                        if self.kind == "anthropic" else raw.get("service_tier")
                    ),
                    "provider_max_output_tokens": budget,
                    "provider_output_ceiling": config.max_output_tokens,
                    "public_max_output_tokens": request.max_output_tokens,
                    "ignored_sampling_fields": (
                        ["temperature", "top_p", "seed"]
                        if self.kind in {"openai", "anthropic"} else []
                    ),
                    "structured_output_schema_projection": projection,
                    "canonical_schema_validated": False,
                    "outcome": "completed",
                },
            )
        except Exception as exc:
            raise ProviderCallError(
                "MalformedProviderResponse",
                outcome="unknown",
                provider_request_id=getattr(response, "_request_id", None),
            ) from exc
        if result.actual_model_id != entry.expected_actual_model_id:
            raise ProviderCallError(
                "ProviderModelIdentityMismatch",
                outcome="completed",
                provider_request_id=request_id,
                result=result,
            )
        if result.finish_reason == "error":
            raise ProviderCallError(
                "ProviderGenerationFailure",
                outcome="completed",
                provider_request_id=request_id,
                result=result,
            )
        if result.finish_reason == "stop" and request.response_format is not None:
            try:
                validate_canonical_output(text, request.response_format)
            except Exception as exc:
                raise ProviderCallError(
                    "CanonicalSchemaViolation",
                    outcome="completed",
                    provider_request_id=request_id,
                    result=result,
                ) from exc
            result.metadata["canonical_schema_validated"] = True
        return result

    def _output(self, raw: dict[str, Any]) -> tuple[str, str, str | None]:
        if self.kind == "openai":
            blocks = [
                part
                for item in raw.get("output", [])
                if item.get("type") == "message"
                for part in item.get("content", [])
            ]
            text = "".join(part.get("text", "") for part in blocks if part["type"] == "output_text")
            reason = (raw.get("incomplete_details") or {}).get("reason") or raw.get("status")
            if any(part["type"] == "refusal" for part in blocks) or reason == "content_filter":
                return text, "refusal", reason
            openai_finish_reasons: dict[str | None, str] = {
                "completed": "stop", "max_output_tokens": "length",
            }
            return (
                text,
                openai_finish_reasons.get(reason, "error"),
                reason,
            )
        if self.kind == "anthropic":
            text = "".join(
                part.get("text", "")
                for part in raw.get("content", [])
                if part.get("type") == "text"
            )
            reason = raw.get("stop_reason")
            anthropic_finish_reasons: dict[str | None, str] = {
                "end_turn": "stop",
                "stop_sequence": "stop",
                "max_tokens": "length",
                "model_context_window_exceeded": "length",
                "refusal": "refusal",
            }
            return (
                text,
                anthropic_finish_reasons.get(reason, "error"),
                reason,
            )
        choice = raw["choices"][0]
        reason = choice.get("finish_reason")
        message = choice["message"]
        finish = (
            "refusal"
            if message.get("refusal") or reason == "content_filter"
            else {
                "stop": "stop",
                "length": "length",
            }.get(reason, "error")
        )
        return message.get("content") or "", finish, reason


class OpenAIResponsesGenerationGateway(ProviderGenerationGateway):
    kind = "openai"

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout_seconds: float = 300.0,
        client: Any = None,
        image_bytes_resolver: ImageBytesResolver | None = None,
    ) -> None:
        super().__init__(
            registry,
            client=client
            if client is not None
            else AsyncOpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.openai.com/v1",
                timeout=timeout_seconds,
                max_retries=0,
            ),
            image_bytes_resolver=image_bytes_resolver,
        )


class AnthropicGenerationGateway(ProviderGenerationGateway):
    kind = "anthropic"

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout_seconds: float = 300.0,
        client: Any = None,
        image_bytes_resolver: ImageBytesResolver | None = None,
    ) -> None:
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(
                api_key=api_key,
                base_url=base_url or "https://api.anthropic.com",
                timeout=timeout_seconds,
                max_retries=0,
            )
        super().__init__(registry, client=client, image_bytes_resolver=image_bytes_resolver)


class QwenGenerationGateway(ProviderGenerationGateway):
    kind = "qwen_vllm"

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 300.0,
        client: Any = None,
        image_bytes_resolver: ImageBytesResolver | None = None,
        capacity_guard: CapacityRequestGuard | None = None,
        local_tokenize: ExactTokenizer | None = None,
        service_tokenize: ExactTokenizer | None = None,
    ) -> None:
        if any(item is not None for item in (capacity_guard, local_tokenize, service_tokenize)):
            if any(item is None for item in (capacity_guard, local_tokenize, service_tokenize)):
                raise ManagedRuntimeError("capacity_exact_tokenizers_required")
        self.capacity_guard = capacity_guard
        self.local_tokenize = local_tokenize
        self.service_tokenize = service_tokenize
        super().__init__(
            registry,
            client=client
            if client is not None
            else AsyncOpenAI(
                api_key=api_key,
                base_url=_openai_api_root(base_url),
                timeout=timeout_seconds,
                max_retries=0,
            ),
            image_bytes_resolver=image_bytes_resolver,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.capacity_guard is not None:
            try:
                self.capacity_guard.validate_entry(self.registry.get(request.model_alias))
            except (ManagedRuntimeError, KeyError):
                raise CapacityGatewayError("capacity_generation_registry_mismatch") from None
        return await super().generate(request)

    async def _create(self, create: Callable[..., Awaitable[Any]], kwargs: dict[str, Any]) -> Any:
        if self.capacity_guard is None:
            return await super()._create(create, kwargs)
        assert self.local_tokenize is not None and self.service_tokenize is not None
        return await self.capacity_guard.dispatch(
            kwargs, local_tokenize=self.local_tokenize,
            service_tokenize=self.service_tokenize, generate=create,
        )
