"""Provider-neutral model gateway boundary for PMMC."""

from pmmc.models.execution import (
    AuditedEmbeddingResult,
    AuditedGenerationResult,
    ModelCallContext,
    ModelExecutionService,
    ParsedModelOutput,
)
from pmmc.models.factory import (
    check_model_endpoint,
    create_embedding_gateway,
    create_generation_gateway,
    create_image_bytes_resolver,
    create_image_url_resolver,
    create_model_execution_service,
    create_token_counter,
)
from pmmc.models.fake import DeterministicFakeGateway
from pmmc.models.images import (
    ImageUrlResolver,
    ObjectStoreImageUrlResolver,
    StaticImageUrlResolver,
    validate_paper_image_reference,
)
from pmmc.models.langchain_adapters import (
    PMMCLangChainEmbeddings,
    as_langchain_chat_runnable,
)
from pmmc.models.preflight import (
    ModelSmokeConfigurationError,
    validate_model_smoke_configuration,
)
from pmmc.models.protocols import EmbeddingGateway, GenerationGateway, TokenCounter, TraceSink
from pmmc.models.provider_config import ProviderCapabilities, ProviderConfig
from pmmc.models.provider_images import (
    ImageBytes,
    ImageBytesResolver,
    ObjectStoreImageBytesResolver,
)
from pmmc.models.providers import (
    AnthropicGenerationGateway,
    OpenAIResponsesGenerationGateway,
    ProviderCallError,
    QwenGenerationGateway,
    provider_output_budget,
)
from pmmc.models.registry import (
    ModelRegistry,
    ModelRegistryEntry,
    bind_registry_to_manifest,
    load_model_registry,
    verify_manifest_registry_compatibility,
)
from pmmc.models.tokenization import (
    HeuristicTokenCounter,
    ProcessorTokenCounter,
    TransformersTokenCounter,
)
from pmmc.models.tracing import create_trace_sink

__all__ = [
    "AnthropicGenerationGateway",
    "OpenAIResponsesGenerationGateway",
    "QwenGenerationGateway",
    "ProviderCallError",
    "ProviderCapabilities",
    "ProviderConfig",
    "ImageBytes",
    "ImageBytesResolver",
    "ObjectStoreImageBytesResolver",
    "create_image_bytes_resolver",
    "provider_output_budget",
    "AuditedGenerationResult",
    "AuditedEmbeddingResult",
    "DeterministicFakeGateway",
    "EmbeddingGateway",
    "GenerationGateway",
    "HeuristicTokenCounter",
    "ImageUrlResolver",
    "ModelCallContext",
    "ModelExecutionService",
    "ParsedModelOutput",
    "ModelRegistry",
    "ModelRegistryEntry",
    "ModelSmokeConfigurationError",
    "ObjectStoreImageUrlResolver",
    "ProcessorTokenCounter",
    "StaticImageUrlResolver",
    "validate_paper_image_reference",
    "TokenCounter",
    "TraceSink",
    "TransformersTokenCounter",
    "PMMCLangChainEmbeddings",
    "as_langchain_chat_runnable",
    "bind_registry_to_manifest",
    "check_model_endpoint",
    "create_embedding_gateway",
    "create_generation_gateway",
    "create_image_url_resolver",
    "create_model_execution_service",
    "create_token_counter",
    "create_trace_sink",
    "load_model_registry",
    "validate_model_smoke_configuration",
    "verify_manifest_registry_compatibility",
]
