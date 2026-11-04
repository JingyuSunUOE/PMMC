"""OpenAI-compatible and vLLM model gateway adapters."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from typing import Any

import httpx
from openai import AsyncOpenAI

from pmmc.models.images import ImageUrlResolver, StaticImageUrlResolver
from pmmc.models.protocols import GenerationGateway
from pmmc.models.registry import ModelRegistry
from pmmc.models.routing import stable_replica_index
from pmmc.schemas import (
    EmbeddingInput,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingVector,
    GenerationRequest,
    GenerationResult,
    ModelCapability,
    ModelContentKind,
    TokenUsage,
)


class ModelGatewayError(RuntimeError):
    """Provider failure normalized without response-body leakage."""

    diagnostic_event = "MODEL_GATEWAY_FAILURE"

    def __init__(
        self,
        error_code: str,
        *,
        provider_status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        self.error_code = error_code[:128]
        self.provider_status_code = provider_status_code
        self.retryable = retryable
        self.safe_summary = f"model gateway request failed ({self.error_code})"
        super().__init__(self.error_code)

    def safe_diagnostics(self) -> dict[str, object]:
        return {
            "error_code": self.error_code,
            "provider_status_code": self.provider_status_code,
            "retryable": self.retryable,
        }

    @classmethod
    def from_exception(cls, error: BaseException) -> ModelGatewayError:
        raw_status = getattr(error, "status_code", None)
        if not isinstance(raw_status, int):
            raw_status = getattr(getattr(error, "response", None), "status_code", None)
        status_code = raw_status if isinstance(raw_status, int) else None
        retryable_statuses = {408, 409, 425, 429}
        retryable = bool(
            (status_code is not None and (status_code in retryable_statuses or status_code >= 500))
            or (
                status_code is None
                and (
                    isinstance(error, (httpx.TransportError, TimeoutError))
                    or type(error).__name__
                    in {
                        "APIConnectionError",
                        "APITimeoutError",
                        "ConnectError",
                        "ConnectTimeout",
                        "ReadError",
                        "ReadTimeout",
                    }
                )
            )
        )
        return cls(
            type(error).__name__,
            provider_status_code=status_code,
            retryable=retryable,
        )


_VLLM_STRUCTURED_OUTPUT_PROJECTION_VERSION = "vllm-xgrammar-schema-projection-v1"
_XGRAMMAR_UNSUPPORTED_ARRAY_KEYWORDS = frozenset({"uniqueItems"})


def _canonical_json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _project_xgrammar_schema(
    value: object,
    *,
    removed_keyword_counts: dict[str, int],
) -> object:
    if isinstance(value, list):
        return [
            _project_xgrammar_schema(
                item,
                removed_keyword_counts=removed_keyword_counts,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    is_array_schema = value.get("type") == "array"
    projected: dict[str, object] = {}
    for key, item in value.items():
        if is_array_schema and key in _XGRAMMAR_UNSUPPORTED_ARRAY_KEYWORDS:
            removed_keyword_counts[key] = removed_keyword_counts.get(key, 0) + 1
            continue
        projected[key] = _project_xgrammar_schema(
            item,
            removed_keyword_counts=removed_keyword_counts,
        )
    return projected


def project_vllm_response_format(
    response_format: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, object] | None]:
    """Project canonical JSON Schema onto vLLM 0.24 xgrammar's supported subset.

    The canonical request and its Pydantic parser retain every constraint. The
    transport projection removes only constraints that xgrammar rejects before
    inference, and records hashes/counts so the broader grammar is auditable.
    """

    if response_format.get("type") != "json_schema":
        return response_format, None
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return response_format, None
    schema = json_schema.get("schema")
    if not isinstance(schema, dict):
        return response_format, None

    removed_keyword_counts: dict[str, int] = {}
    projected_schema = _project_xgrammar_schema(
        schema,
        removed_keyword_counts=removed_keyword_counts,
    )
    if not removed_keyword_counts:
        return response_format, None
    assert isinstance(projected_schema, dict)
    projected_format = {
        **response_format,
        "json_schema": {
            **json_schema,
            "schema": projected_schema,
        },
    }
    diagnostics: dict[str, object] = {
        "projection_version": _VLLM_STRUCTURED_OUTPUT_PROJECTION_VERSION,
        "removed_keyword_counts": dict(sorted(removed_keyword_counts.items())),
        "source_schema_sha256": _canonical_json_hash(schema),
        "transport_schema_sha256": _canonical_json_hash(projected_schema),
        "server_validation_required": True,
    }
    return projected_format, diagnostics


def _openai_api_root(base_url: str) -> str:
    """Accept either a service root or an already versioned OpenAI API root."""

    value = base_url.rstrip("/")
    return value if value.endswith("/v1") else value + "/v1"


def _usage(value: Any) -> TokenUsage:
    if value is None:
        return TokenUsage()
    input_tokens = int(getattr(value, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(value, "completion_tokens", 0) or 0)
    total = int(getattr(value, "total_tokens", input_tokens + output_tokens) or 0)
    if total != input_tokens + output_tokens:
        total = input_tokens + output_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
    )


def _response_cost(response: Any, raw: dict[str, Any]) -> float | None:
    hidden = getattr(response, "_hidden_params", None)
    candidates = [
        hidden.get("response_cost") if isinstance(hidden, dict) else None,
        raw.get("response_cost"),
    ]
    for value in candidates:
        if value is not None:
            return float(value)
    return None


def _provider_embedding_dimensions(
    requested_dimension: int | None,
    native_dimension: int | None,
) -> int | None:
    """Only request an actual output-size override from the provider.

    vLLM treats the presence of ``dimensions`` as a Matryoshka projection
    request, even when the value equals the model's native output dimension.
    PMMC still validates the response against the requested/native dimension
    after the call.
    """

    if requested_dimension is None or requested_dimension == native_dimension:
        return None
    return requested_dimension


class OpenAICompatibleGenerationGateway:
    """Chat/LVLM adapter for a LiteLLM or vLLM OpenAI-compatible endpoint."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 300.0,
        client: AsyncOpenAI | None = None,
        image_url_resolver: ImageUrlResolver | None = None,
        verified_actual_model_ids: Mapping[str, str] | None = None,
        provider_model_overrides: Mapping[str, str] | None = None,
    ) -> None:
        self.registry = registry
        self.client = client or AsyncOpenAI(
            base_url=_openai_api_root(base_url),
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.image_url_resolver = image_url_resolver or StaticImageUrlResolver()
        self.verified_actual_model_ids = dict(verified_actual_model_ids or {})
        self.provider_model_overrides = dict(provider_model_overrides or {})
        for alias, actual_model_id in self.verified_actual_model_ids.items():
            expected = self.registry.get(alias).expected_actual_model_id
            if actual_model_id != expected:
                raise ValueError(
                    f"verified actual model ID for {alias!r} does not match registry: "
                    f"expected {expected!r}, received {actual_model_id!r}"
                )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if (
            request.provider_max_output_tokens is not None
            or request.thinking_mode.value in {"minimal", "low"}
        ):
            raise ValueError("provider budget/reasoning requires a Step 8 provider adapter")
        entry = self.registry.get(request.model_alias)
        has_images = any(
            part.kind is ModelContentKind.IMAGE
            for message in request.messages
            for part in message.parts
        )
        entry.require(ModelCapability.CHAT_MULTIMODAL if has_images else ModelCapability.CHAT_TEXT)
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            content: list[dict[str, Any]] = []
            for part in message.parts:
                if part.kind is ModelContentKind.TEXT:
                    content.append({"type": "text", "text": part.text})
                else:
                    assert part.image is not None
                    image_url: dict[str, Any] = {
                        "url": await self.image_url_resolver.resolve(part.image)
                    }
                    if part.detail is not None:
                        image_url["detail"] = part.detail
                    content.append({"type": "image_url", "image_url": image_url})
            messages.append({"role": message.role.value, "content": content})
        started = time.perf_counter()
        schema_projection: dict[str, object] | None = None
        try:
            create: Any = self.client.chat.completions.create
            request_kwargs: dict[str, Any] = {
                "model": self.provider_model_overrides.get(
                    request.model_alias,
                    entry.provider_model,
                ),
                "messages": messages,
                "max_tokens": request.max_output_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "extra_headers": {"X-Request-Id": request.request_id},
            }
            if request.seed is not None:
                request_kwargs["seed"] = request.seed
            if request.response_format is not None:
                transport_format, schema_projection = project_vllm_response_format(
                    request.response_format
                )
                request_kwargs["response_format"] = transport_format
            if request.thinking_mode.value == "disabled":
                request_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            if request.timeout_seconds is not None:
                request_kwargs["timeout"] = request.timeout_seconds
            response = await create(
                **request_kwargs,
            )
        except Exception as exc:
            raise ModelGatewayError.from_exception(exc) from exc
        choice = response.choices[0]
        text = choice.message.content or ""
        raw = response.model_dump(mode="json")
        transport_model_id = str(response.model)
        verified_actual_model_id = self.verified_actual_model_ids.get(request.model_alias)
        return GenerationResult(
            request_id=request.request_id,
            model_alias=request.model_alias,
            actual_model_id=verified_actual_model_id or transport_model_id,
            text=text,
            usage=_usage(response.usage),
            provider_request_id=str(response.id),
            finish_reason=str(choice.finish_reason) if choice.finish_reason else None,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=_response_cost(response, raw),
            raw_response=raw,
            metadata={
                "transport_model_id": transport_model_id,
                "model_identity_source": (
                    "managed_vllm_models_endpoint"
                    if verified_actual_model_id is not None
                    else "provider_response"
                ),
                **(
                    {"structured_output_schema_projection": schema_projection}
                    if schema_projection is not None
                    else {}
                ),
            },
        )


class RoutedGenerationGateway:
    """Stable request routing across identical loopback generation replicas."""

    def __init__(
        self,
        gateways: list[tuple[int, str, GenerationGateway]],
    ) -> None:
        if not gateways:
            raise ValueError("generation replica pool cannot be empty")
        self.gateways = sorted(gateways, key=lambda item: item[0])

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        start = stable_replica_index(request.request_id, len(self.gateways))
        last_error: Exception | None = None
        for offset in range(len(self.gateways)):
            replica_index, runtime_service_id, gateway = self.gateways[
                (start + offset) % len(self.gateways)
            ]
            try:
                result = await gateway.generate(request)
                return result.model_copy(
                    update={
                        "metadata": {
                            **result.metadata,
                            "runtime_service_id": runtime_service_id,
                            "replica_index": replica_index,
                            "replica_count": len(self.gateways),
                            "replica_failover_count": offset,
                        }
                    }
                )
            except Exception as exc:
                last_error = exc
                if isinstance(exc, ModelGatewayError) and not exc.retryable:
                    raise
        assert last_error is not None
        raise last_error


class OpenAICompatibleEmbeddingGateway:
    """Text embedding adapter for vLLM's OpenAI-compatible `/v1/embeddings`."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 300.0,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.registry = registry
        self.client = client or AsyncOpenAI(
            base_url=_openai_api_root(base_url),
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        entry = self.registry.get(request.model_alias)
        entry.require(ModelCapability.EMBED_TEXT)
        if any(item.image_refs or item.text is None for item in request.inputs):
            raise ValueError("OpenAI-compatible text embedding accepts text-only inputs")
        started = time.perf_counter()
        try:
            create: Any = self.client.embeddings.create
            request_kwargs: dict[str, Any] = {
                "model": entry.provider_model,
                "input": [item.text or "" for item in request.inputs],
                "extra_headers": {"X-Request-Id": request.request_id},
            }
            provider_dimensions = _provider_embedding_dimensions(
                request.dimensions,
                entry.embedding_dimension,
            )
            if provider_dimensions is not None:
                request_kwargs["dimensions"] = provider_dimensions
            if request.timeout_seconds is not None:
                request_kwargs["timeout"] = request.timeout_seconds
            response = await create(**request_kwargs)
        except Exception as exc:
            raise ModelGatewayError.from_exception(exc) from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [
            EmbeddingVector(input_id=item.input_id, vector=list(result.embedding))
            for item, result in zip(request.inputs, ordered, strict=True)
        ]
        expected_dimension = request.dimensions or entry.embedding_dimension
        if expected_dimension is not None and any(
            len(vector.vector) != expected_dimension for vector in vectors
        ):
            raise ModelGatewayError("EmbeddingDimensionMismatch")
        raw = response.model_dump(mode="json")
        return EmbeddingResult(
            request_id=request.request_id,
            model_alias=request.model_alias,
            actual_model_id=str(response.model),
            vectors=vectors,
            usage=_usage(response.usage),
            provider_request_id=str(getattr(response, "id", request.request_id)),
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=_response_cost(response, raw),
            raw_response=raw,
        )


async def _vllm_multimodal_messages(
    item: EmbeddingInput,
    resolver: ImageUrlResolver,
) -> list[dict[str, Any]]:
    """Render one input using vLLM's chat-embedding extension."""

    content: list[dict[str, Any]] = []
    for image in item.image_refs:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": await resolver.resolve(image)},
            }
        )
    if item.text is not None:
        content.append({"type": "text", "text": item.text})
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Represent the user's input.",
                }
            ],
        },
        {"role": "user", "content": content},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
        },
    ]


class VLLMMultimodalEmbeddingGateway:
    """Adapter for vLLM's `/v1/embeddings` multimodal chat extension.

    The registry endpoint-kind name remains ``vllm_v2_embed`` for lock-file
    compatibility, but vLLM 0.24 serves this protocol at `/v1/embeddings`.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float = 300.0,
        client: httpx.AsyncClient | None = None,
        image_url_resolver: ImageUrlResolver | None = None,
    ) -> None:
        self.registry = registry
        self.base_url = _openai_api_root(base_url)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds, headers=headers)
        self.image_url_resolver = image_url_resolver or StaticImageUrlResolver()

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        entry = self.registry.get(request.model_alias)
        has_text = any(item.text is not None for item in request.inputs)
        has_images = any(item.image_refs for item in request.inputs)
        if has_text and has_images:
            entry.require(ModelCapability.EMBED_MULTIMODAL)
        elif has_images:
            entry.require(ModelCapability.EMBED_IMAGE)
        else:
            entry.require(ModelCapability.EMBED_TEXT)
        payload: dict[str, Any] = {
            "model": entry.provider_model,
            "messages": [
                await _vllm_multimodal_messages(item, self.image_url_resolver)
                for item in request.inputs
            ],
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
        }
        provider_dimensions = _provider_embedding_dimensions(
            request.dimensions,
            entry.embedding_dimension,
        )
        if provider_dimensions is not None:
            payload["dimensions"] = provider_dimensions
        started = time.perf_counter()
        try:
            request_kwargs: dict[str, Any] = {
                "json": payload,
                "headers": {"X-Request-Id": request.request_id},
            }
            if request.timeout_seconds is not None:
                request_kwargs["timeout"] = request.timeout_seconds
            response = await self.client.post(f"{self.base_url}/embeddings", **request_kwargs)
            response.raise_for_status()
            raw = response.json()
        except Exception as exc:
            raise ModelGatewayError.from_exception(exc) from exc
        data = raw.get("data") or raw.get("embeddings")
        if not isinstance(data, list) or len(data) != len(request.inputs):
            raise ModelGatewayError("InvalidEmbeddingResponse")
        if all(isinstance(value, dict) and "index" in value for value in data):
            data = sorted(data, key=lambda value: int(value["index"]))
        vectors: list[EmbeddingVector] = []
        for item, value in zip(request.inputs, data, strict=True):
            vector = value.get("embedding") if isinstance(value, dict) else value
            vectors.append(EmbeddingVector(input_id=item.input_id, vector=vector))
        expected_dimension = request.dimensions or entry.embedding_dimension
        if expected_dimension is not None and any(
            len(vector.vector) != expected_dimension for vector in vectors
        ):
            raise ModelGatewayError("EmbeddingDimensionMismatch")
        usage_raw = raw.get("usage", {})
        input_tokens = int(usage_raw.get("prompt_tokens", usage_raw.get("total_tokens", 0)))
        return EmbeddingResult(
            request_id=request.request_id,
            model_alias=request.model_alias,
            actual_model_id=str(raw.get("model", entry.provider_model)),
            vectors=vectors,
            usage=TokenUsage(input_tokens=input_tokens, total_tokens=input_tokens),
            provider_request_id=str(raw.get("id", request.request_id)),
            latency_ms=(time.perf_counter() - started) * 1000,
            raw_response=raw,
        )


class ContrastiveEmbeddingGateway:
    """Adapter for the loopback PMMC SigLIP2 text/image embedding worker."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float = 300.0,
        client: httpx.AsyncClient | None = None,
        image_url_resolver: ImageUrlResolver | None = None,
    ) -> None:
        self.registry = registry
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds, headers=headers)
        self.image_url_resolver = image_url_resolver or StaticImageUrlResolver()

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        entry = self.registry.get(request.model_alias)
        inputs: list[dict[str, str]] = []
        for item in request.inputs:
            if item.text is not None and not item.image_refs:
                entry.require(ModelCapability.EMBED_TEXT)
                inputs.append({"type": "text", "text": item.text.lower()})
            elif item.text is None and len(item.image_refs) == 1:
                entry.require(ModelCapability.EMBED_IMAGE)
                inputs.append(
                    {
                        "type": "image_url",
                        "image_url": await self.image_url_resolver.resolve(item.image_refs[0]),
                    }
                )
            else:
                raise ValueError("contrastive embedding inputs must be text-only or one-image-only")
        payload = {"model": entry.provider_model, "input": inputs}
        started = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "json": payload,
                "headers": {"X-Request-Id": request.request_id},
            }
            if request.timeout_seconds is not None:
                kwargs["timeout"] = request.timeout_seconds
            response = await self.client.post(self.base_url + "/v1/embeddings", **kwargs)
            response.raise_for_status()
            raw = response.json()
        except Exception as exc:
            raise ModelGatewayError.from_exception(exc) from exc
        data = raw.get("data")
        if not isinstance(data, list) or len(data) != len(request.inputs):
            raise ModelGatewayError("InvalidEmbeddingResponse")
        ordered = sorted(data, key=lambda item: int(item["index"]))
        vectors = [
            EmbeddingVector(input_id=item.input_id, vector=value["embedding"])
            for item, value in zip(request.inputs, ordered, strict=True)
        ]
        expected_dimension = request.dimensions or entry.embedding_dimension
        if expected_dimension is not None and any(
            len(item.vector) != expected_dimension for item in vectors
        ):
            raise ModelGatewayError("EmbeddingDimensionMismatch")
        usage_raw = raw.get("usage", {})
        input_tokens = int(usage_raw.get("prompt_tokens", 0) or 0)
        return EmbeddingResult(
            request_id=request.request_id,
            model_alias=request.model_alias,
            actual_model_id=str(raw.get("model", entry.provider_model)),
            vectors=vectors,
            usage=TokenUsage(input_tokens=input_tokens, total_tokens=input_tokens),
            provider_request_id=str(raw.get("id", request.request_id)),
            latency_ms=(time.perf_counter() - started) * 1000,
            raw_response=raw,
            metadata={"text_lowercase": True, "text_max_length": 64},
        )


class RoutedEmbeddingGateway:
    """Stable request routing across identical loopback embedding replicas."""

    def __init__(self, gateways: list[tuple[int, str, Any]]) -> None:
        if not gateways:
            raise ValueError("embedding replica pool cannot be empty")
        self.gateways = sorted(gateways, key=lambda item: item[0])

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        start = stable_replica_index(request.request_id, len(self.gateways))
        last_error: Exception | None = None
        for offset in range(len(self.gateways)):
            replica_index, runtime_service_id, gateway = self.gateways[
                (start + offset) % len(self.gateways)
            ]
            try:
                result = await gateway.embed(request)
                return result.model_copy(
                    update={
                        "metadata": {
                            **result.metadata,
                            "runtime_service_id": runtime_service_id,
                            "replica_index": replica_index,
                            "replica_count": len(self.gateways),
                            "replica_failover_count": offset,
                        }
                    }
                )
            except Exception as exc:
                last_error = exc
                if isinstance(exc, ModelGatewayError) and not exc.retryable:
                    raise
        assert last_error is not None
        raise last_error
