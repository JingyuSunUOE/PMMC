"""Settings-driven gateway creation and non-inference endpoint checks."""

from __future__ import annotations

from typing import Any, TypedDict

import httpx

from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.infrastructure.object_store import ObjectStore
from pmmc.infrastructure.settings import AppMode, Settings
from pmmc.models.execution import ModelExecutionService
from pmmc.models.fake import DeterministicFakeGateway
from pmmc.models.gateways import (
    ContrastiveEmbeddingGateway,
    OpenAICompatibleEmbeddingGateway,
    OpenAICompatibleGenerationGateway,
    RoutedEmbeddingGateway,
    RoutedGenerationGateway,
    VLLMMultimodalEmbeddingGateway,
)
from pmmc.models.images import ImageUrlResolver, ObjectStoreImageUrlResolver
from pmmc.models.protocols import GenerationGateway, ModelCallWriter, TokenCounter
from pmmc.models.provider_images import ImageBytesResolver, ObjectStoreImageBytesResolver
from pmmc.models.provider_token_budget import ProviderConfiguredTokenCounter
from pmmc.models.providers import (
    AnthropicGenerationGateway,
    OpenAIResponsesGenerationGateway,
    QwenGenerationGateway,
)
from pmmc.models.registry import ModelRegistry, ModelRegistryEntry
from pmmc.models.tokenization import (
    HeuristicTokenCounter,
    ProcessorTokenCounter,
    QwenVLCachedExactTokenCounter,
    TransformersTokenCounter,
)
from pmmc.models.tracing import create_trace_sink
from pmmc.schemas import DataScope, ModelEndpointKind, RuntimeMode


class _ProviderGatewayOptions(TypedDict):
    timeout_seconds: float
    image_bytes_resolver: ImageBytesResolver | None
    client: Any


def _managed_replica_bindings(
    settings: Settings,
    alias: str,
) -> list[tuple[int, str, str]]:
    if settings.runtime_mode is not RuntimeMode.MANAGED_LOCAL:
        return []
    raw = settings.active_runtime_endpoint_pools.get(alias, [])
    bindings: list[tuple[int, str, str]] = []
    for value in raw:
        replica_index = int(value.get("replica_index", -1))
        runtime_service_id = str(value.get("runtime_service_id", ""))
        endpoint = str(value.get("endpoint", ""))
        if replica_index < 0 or not runtime_service_id or not endpoint:
            raise ValueError(f"invalid managed endpoint-pool binding: {alias}")
        if not runtime_service_id.startswith(f"{settings.active_runtime_id}:"):
            raise ValueError(f"managed endpoint-pool runtime mismatch: {alias}")
        if not endpoint.startswith("http://127.0.0.1:"):
            raise ValueError(f"managed endpoint pool is not loopback-only: {alias}")
        bindings.append((replica_index, runtime_service_id, endpoint))
    bindings.sort(key=lambda item: item[0])
    if bindings and [item[0] for item in bindings] != list(range(len(bindings))):
        raise ValueError(f"managed endpoint-pool ordinals are not contiguous: {alias}")
    return bindings


def _secret(value: object | None, default: str = "EMPTY") -> str:
    if value is None:
        return default
    getter = getattr(value, "get_secret_value", None)
    return str(getter() if getter else value)


def _capacity_generation_gateway(
    settings: Settings, registry: ModelRegistry, alias: str, *,
    common: _ProviderGatewayOptions, provider_base_url: str | None,
) -> GenerationGateway | None:
    from pmmc.runtime.manager import ManagedRuntimeError
    from pmmc.runtime.step8_capacity_runtime import capacity_endpoint, load_capacity_runtime
    from pmmc.runtime.step8_capacity_tokenization import (
        LocalCapacityTokenizer,
        ServiceCapacityTokenizer,
    )

    runtime = load_capacity_runtime(settings, registry, alias)
    if runtime is None:
        return None
    local = LocalCapacityTokenizer(runtime, lambda: create_token_counter(settings, registry, alias))
    key = _secret(settings.vllm_api_key)

    def gateway(endpoint: str) -> QwenGenerationGateway:
        return QwenGenerationGateway(
            registry, base_url=endpoint, api_key=key, **common,
            capacity_guard=runtime.guard, local_tokenize=local,
            service_tokenize=ServiceCapacityTokenizer(
                runtime, endpoint, api_key=key, timeout=settings.model_request_timeout_seconds,
            ),
        )

    if common["client"] is None and provider_base_url is None:
        return RoutedGenerationGateway([
            (index, service_id, gateway(endpoint))
            for index, service_id, endpoint in runtime.bindings
        ])
    client_endpoint = getattr(common["client"], "base_url", None)
    if common["client"] is not None and client_endpoint is None:
        raise ManagedRuntimeError("capacity_injected_client_endpoint_required")
    endpoint = capacity_endpoint(provider_base_url or str(client_endpoint))
    if client_endpoint is not None and capacity_endpoint(str(client_endpoint)) != endpoint:
        raise ManagedRuntimeError("capacity_managed_endpoint_mismatch")
    if endpoint not in {row[2] for row in runtime.bindings}:
        raise ManagedRuntimeError("capacity_managed_endpoint_mismatch")
    return gateway(endpoint)


def _managed_actual_model_ids(
    settings: Settings,
    registry: ModelRegistry,
) -> dict[str, str]:
    """Return identities already verified by the active managed vLLM stage."""

    if (
        settings.runtime_mode is not RuntimeMode.MANAGED_LOCAL
        or settings.active_runtime_id is None
        or not settings.active_runtime_service_map
    ):
        return {}
    prefix = f"{settings.active_runtime_id}:"
    values: dict[str, str] = {}
    for alias, runtime_service_id in settings.active_runtime_service_map.items():
        if not runtime_service_id.startswith(prefix):
            raise ValueError(
                f"managed runtime service binding for {alias!r} does not belong to "
                f"active runtime {settings.active_runtime_id!r}"
            )
        values[alias] = registry.get(alias).expected_actual_model_id
    return values


def create_generation_gateway(
    settings: Settings,
    registry: ModelRegistry,
    alias: str,
    *,
    image_url_resolver: ImageUrlResolver | None = None,
    image_bytes_resolver: ImageBytesResolver | None = None,
    provider_client: Any = None,
    provider_base_url: str | None = None,
    accounting_repository: Any = None,
):  # type: ignore[no-untyped-def]
    entry = registry.get(alias)
    if entry.provider_config is not None:
        if image_bytes_resolver is None and isinstance(
            image_url_resolver, ObjectStoreImageUrlResolver,
        ):
            image_bytes_resolver = ObjectStoreImageBytesResolver(
                image_url_resolver.store, bucket=image_url_resolver.bucket,
                allow_verified_mime_correction=True,
            )
        common: _ProviderGatewayOptions = {
            "timeout_seconds": settings.model_request_timeout_seconds,
            "image_bytes_resolver": image_bytes_resolver,
            "client": provider_client,
        }
        kind = entry.provider_config.kind
        if kind in {"openai", "anthropic"}:
            from pmmc.models.step8_accounting import (
                SettingsAccountingGateway,
                load_settings_rate_card,
            )

            rate_card = load_settings_rate_card(settings, entry)
            credential = settings.openai_api_key if kind == "openai" else settings.anthropic_api_key
            if credential is None and provider_client is None:
                raise ValueError(f"{kind.upper()}_API_KEY is not configured")
            adapter = OpenAIResponsesGenerationGateway if kind == "openai" else (
                AnthropicGenerationGateway
            )
            return SettingsAccountingGateway(
                adapter(
                    registry, api_key=_secret(credential), base_url=provider_base_url, **common,
                ),
                settings=settings, registry=registry, rate_card=rate_card,
                repository=accounting_repository,
            )
        capacity_gateway = _capacity_generation_gateway(
            settings, registry, alias, common=common, provider_base_url=provider_base_url,
        )
        if capacity_gateway is not None:
            return capacity_gateway
        bindings = _managed_replica_bindings(settings, alias)
        if bindings and provider_base_url is None and provider_client is None:
            gateways: list[tuple[int, str, GenerationGateway]] = [
                (index, service_id, QwenGenerationGateway(
                    registry, base_url=endpoint, api_key=_secret(settings.vllm_api_key), **common,
                ))
                for index, service_id, endpoint in bindings
            ]
            return RoutedGenerationGateway(gateways)
        endpoint = provider_base_url or settings.vllm_chat_base_url
        if not endpoint and provider_client is None:
            raise ValueError("Qwen requires provider_base_url or VLLM_CHAT_BASE_URL")
        return QwenGenerationGateway(
            registry, base_url=endpoint or "http://injected-client.invalid",
            api_key=_secret(settings.vllm_api_key), **common,
        )
    if any(value is not None for value in (
        provider_client, provider_base_url, image_bytes_resolver,
    )):
        raise ValueError("provider factory options require an entry with provider_config")
    if entry.endpoint_kind is ModelEndpointKind.FAKE:
        return DeterministicFakeGateway(entry.embedding_dimension or 16)
    if entry.endpoint_kind is not ModelEndpointKind.LITELLM_OPENAI:
        raise ValueError(f"model alias does not use a generation endpoint: {alias}")
    replica_bindings = _managed_replica_bindings(settings, alias)
    if len(replica_bindings) > 1:
        return RoutedGenerationGateway(
            [
                (
                    replica_index,
                    runtime_service_id,
                    OpenAICompatibleGenerationGateway(
                        registry,
                        base_url=endpoint,
                        api_key=_secret(settings.vllm_api_key),
                        timeout_seconds=settings.model_request_timeout_seconds,
                        image_url_resolver=image_url_resolver,
                        verified_actual_model_ids={alias: entry.expected_actual_model_id},
                        provider_model_overrides={alias: entry.expected_actual_model_id},
                    ),
                )
                for replica_index, runtime_service_id, endpoint in replica_bindings
            ]
        )
    if not settings.litellm_base_url:
        raise ValueError("LITELLM_BASE_URL is not configured")
    return OpenAICompatibleGenerationGateway(
        registry,
        base_url=settings.litellm_base_url,
        api_key=_secret(settings.litellm_api_key),
        timeout_seconds=settings.model_request_timeout_seconds,
        image_url_resolver=image_url_resolver,
        verified_actual_model_ids=_managed_actual_model_ids(settings, registry),
    )


def create_image_url_resolver(
    settings: Settings,
    store: ObjectStore,
) -> ObjectStoreImageUrlResolver:
    return ObjectStoreImageUrlResolver(
        store,
        bucket=settings.object_store_bucket,
        expires_seconds=settings.object_store_presign_ttl_seconds,
        paper_mode=settings.mode is AppMode.PAPER,
    )


def create_image_bytes_resolver(
    settings: Settings, store: ObjectStore,
) -> ObjectStoreImageBytesResolver:
    return ObjectStoreImageBytesResolver(
        store, bucket=settings.object_store_bucket, allow_verified_mime_correction=True,
    )


def create_token_counter(
    settings: Settings,
    registry: ModelRegistry,
    alias: str,
    *,
    allow_estimated: bool = False,
):  # type: ignore[no-untyped-def]
    """Create an exact local counter or an explicit native-provider budget policy."""

    entry = registry.get(alias)
    if entry.context_window is None or entry.context_window < 32768:
        raise ValueError("full-context answer model must expose at least 32768 tokens")
    if entry.provider_config is not None and entry.provider_config.kind != "qwen_vllm":
        from pmmc.models.provider_images import LocalImageDimensions
        from pmmc.models.provider_token_budget import ProviderTokenBudgetCounter

        if settings.mode is AppMode.PAPER:
            registry.assert_paper_ready()
        return ProviderTokenBudgetCounter(
            entry, LocalImageDimensions(settings.benchmark_asset_path))

    def configured(counter: TokenCounter) -> TokenCounter:
        if entry.provider_config is not None:
            return ProviderConfiguredTokenCounter(counter, entry)
        return counter

    from pmmc.runtime.step8_capacity_runtime import load_capacity_tokenization_contract

    contract = load_capacity_tokenization_contract(settings, registry, alias)
    exact_options: dict[str, Any] = {}
    if contract is not None:
        exact_options = {
            "processor_kwargs": contract["processor_kwargs"],
            "chat_template_kwargs": contract["chat_template_kwargs"],
        }
    elif (
        entry.provider_config is not None
        and entry.provider_config.kind == "qwen_vllm"
        and entry.provider_config.reasoning_mode == "off"
    ):
        # Match QwenGenerationGateway even without a capacity-gated runtime.
        exact_options["chat_template_kwargs"] = {"enable_thinking": False}
    if contract is None and (entry.endpoint_kind is ModelEndpointKind.FAKE or (
        settings.mode is not AppMode.PAPER and allow_estimated
    )):
        return configured(
            HeuristicTokenCounter(alias=entry.alias, revision=entry.tokenizer_revision),
        )
    if settings.mode is AppMode.PAPER:
        registry.assert_paper_ready()
    local_files_only = (
        settings.mode is AppMode.PAPER or settings.runtime_mode is RuntimeMode.MANAGED_LOCAL
    )
    if entry.processor_name and entry.processor_revision:
        if settings.token_counter_backend == "qwen-vl-cached-exact-v1":
            return configured(QwenVLCachedExactTokenCounter(
                alias=entry.alias,
                tokenizer_name=entry.tokenizer_name,
                tokenizer_revision=entry.tokenizer_revision,
                processor_name=entry.processor_name,
                processor_revision=entry.processor_revision,
                project_root=settings.benchmark_asset_path,
                local_files_only=local_files_only,
                require_image_metadata=settings.mode is AppMode.PAPER,
                cache_dir=settings.model_cache_path,
                local_asset_cache_dir=settings.local_asset_cache_path,
                persistent_cache_dir=settings.token_counter_cache_path,
                strict_conformance=settings.mode is AppMode.PAPER,
                conformance_samples=settings.token_counter_conformance_samples,
                **exact_options,
            ))
        return configured(ProcessorTokenCounter(
            alias=entry.alias,
            tokenizer_name=entry.tokenizer_name,
            tokenizer_revision=entry.tokenizer_revision,
            processor_name=entry.processor_name,
            processor_revision=entry.processor_revision,
            project_root=settings.benchmark_asset_path,
            local_files_only=local_files_only,
            require_image_metadata=settings.mode is AppMode.PAPER,
            cache_dir=settings.model_cache_path,
            local_asset_cache_dir=settings.local_asset_cache_path,
            **exact_options,
        ))
    return configured(TransformersTokenCounter(
        entry.tokenizer_name,
        entry.tokenizer_revision,
        local_files_only=local_files_only,
        cache_dir=settings.model_cache_path,
    ))


def create_embedding_gateway(
    settings: Settings,
    registry: ModelRegistry,
    alias: str,
    *,
    image_url_resolver: ImageUrlResolver | None = None,
):  # type: ignore[no-untyped-def]
    entry = registry.get(alias)
    if entry.endpoint_kind is ModelEndpointKind.FAKE:
        return DeterministicFakeGateway(entry.embedding_dimension or 16)
    replica_bindings = _managed_replica_bindings(settings, alias)
    if entry.endpoint_kind is ModelEndpointKind.VLLM_OPENAI:
        if len(replica_bindings) > 1:
            return RoutedEmbeddingGateway(
                [
                    (
                        replica_index,
                        runtime_service_id,
                        OpenAICompatibleEmbeddingGateway(
                            registry,
                            base_url=endpoint,
                            api_key=_secret(settings.vllm_api_key),
                            timeout_seconds=settings.model_request_timeout_seconds,
                        ),
                    )
                    for replica_index, runtime_service_id, endpoint in replica_bindings
                ]
            )
        endpoint = settings.vllm_text_embedding_base_url
        if not endpoint:
            raise ValueError("VLLM_TEXT_EMBEDDING_BASE_URL is not configured")
        return OpenAICompatibleEmbeddingGateway(
            registry,
            base_url=endpoint,
            api_key=_secret(settings.vllm_api_key),
            timeout_seconds=settings.model_request_timeout_seconds,
        )
    if entry.endpoint_kind is ModelEndpointKind.VLLM_V2_EMBED:
        if len(replica_bindings) > 1:
            return RoutedEmbeddingGateway(
                [
                    (
                        replica_index,
                        runtime_service_id,
                        VLLMMultimodalEmbeddingGateway(
                            registry,
                            base_url=endpoint,
                            api_key=_secret(settings.vllm_api_key),
                            timeout_seconds=settings.model_request_timeout_seconds,
                            image_url_resolver=image_url_resolver,
                        ),
                    )
                    for replica_index, runtime_service_id, endpoint in replica_bindings
                ]
            )
        endpoint = settings.vllm_multimodal_embedding_base_url
        if not endpoint:
            raise ValueError("VLLM_MULTIMODAL_EMBEDDING_BASE_URL is not configured")
        return VLLMMultimodalEmbeddingGateway(
            registry,
            base_url=endpoint,
            api_key=_secret(settings.vllm_api_key),
            timeout_seconds=settings.model_request_timeout_seconds,
            image_url_resolver=image_url_resolver,
        )
    if entry.endpoint_kind is ModelEndpointKind.PMMC_CONTRASTIVE:
        if len(replica_bindings) > 1:
            return RoutedEmbeddingGateway(
                [
                    (
                        replica_index,
                        runtime_service_id,
                        ContrastiveEmbeddingGateway(
                            registry,
                            base_url=endpoint,
                            api_key=_secret(settings.vllm_api_key),
                            timeout_seconds=settings.model_request_timeout_seconds,
                            image_url_resolver=image_url_resolver,
                        ),
                    )
                    for replica_index, runtime_service_id, endpoint in replica_bindings
                ]
            )
        endpoint = settings.contrastive_embedding_base_url
        if not endpoint:
            raise ValueError("PMMC_CONTRASTIVE_EMBEDDING_BASE_URL is not configured")
        return ContrastiveEmbeddingGateway(
            registry,
            base_url=endpoint,
            api_key=_secret(settings.vllm_api_key),
            timeout_seconds=settings.model_request_timeout_seconds,
            image_url_resolver=image_url_resolver,
        )
    raise ValueError(f"model alias does not use an embedding endpoint: {alias}")


def create_model_execution_service(
    settings: Settings,
    registry: ModelRegistry,
    *,
    artifact_service: ExperimentArtifactService,
    model_calls: ModelCallWriter,
    data_scope: DataScope = DataScope.RUNTIME,
) -> ModelExecutionService:
    """Apply mode-specific retry, concurrency, identity, and observability rules."""

    return ModelExecutionService(
        registry=registry,
        artifact_service=artifact_service,
        model_calls=model_calls,
        data_scope=data_scope,
        trace_sink=create_trace_sink(settings),
        paper_mode=settings.mode is AppMode.PAPER,
        max_attempts=settings.model_max_attempts,
        max_concurrency=settings.model_max_concurrency,
        runtime_id=settings.active_runtime_id,
        runtime_service_ids=settings.active_runtime_service_map,
    )


def endpoint_for(settings: Settings, entry: ModelRegistryEntry) -> str | None:
    if entry.provider_config is not None:
        if entry.provider_config.kind == "openai":
            return "https://api.openai.com/v1"
        if entry.provider_config.kind == "anthropic":
            return "https://api.anthropic.com"
        return settings.vllm_chat_base_url
    return {
        ModelEndpointKind.LITELLM_OPENAI: settings.litellm_base_url,
        ModelEndpointKind.VLLM_OPENAI: settings.vllm_text_embedding_base_url,
        ModelEndpointKind.VLLM_V2_EMBED: settings.vllm_multimodal_embedding_base_url,
        ModelEndpointKind.PMMC_CONTRASTIVE: settings.contrastive_embedding_base_url,
        ModelEndpointKind.FAKE: "fake://local",
    }[entry.endpoint_kind]


def _health_root(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


async def check_model_endpoint(
    settings: Settings,
    registry: ModelRegistry,
    alias: str,
) -> dict[str, object]:
    """Check endpoint health without sending model input or incurring inference."""

    entry = registry.get(alias)
    if entry.provider_config is not None and entry.provider_config.kind != "qwen_vllm":
        return {
            "alias": alias, "endpoint_kind": entry.provider_config.kind,
            "healthy": None, "status": "not_checked", "inference_performed": False,
        }
    endpoint = endpoint_for(settings, entry)
    if entry.endpoint_kind is ModelEndpointKind.FAKE:
        return {"alias": alias, "endpoint_kind": entry.endpoint_kind.value, "healthy": True}
    if not endpoint:
        raise ValueError(f"endpoint is not configured for model alias: {alias}")
    credential = (
        settings.litellm_api_key
        if entry.endpoint_kind is ModelEndpointKind.LITELLM_OPENAI and entry.provider_config is None
        else settings.vllm_api_key
    )
    headers = {"Authorization": f"Bearer {_secret(credential)}"}
    health_path = (
        "/health/liveliness"
        if entry.endpoint_kind is ModelEndpointKind.LITELLM_OPENAI and entry.provider_config is None
        else "/health"
    )
    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        response = await client.get(_health_root(endpoint) + health_path)
        response.raise_for_status()
    return {
        "alias": alias,
        "endpoint_kind": entry.endpoint_kind.value,
        "healthy": True,
        "status_code": response.status_code,
    }
