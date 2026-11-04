"""Secret-free configuration validation for live model gateway smoke tests."""

from __future__ import annotations

from urllib.parse import urlsplit

from pmmc.infrastructure.settings import AppMode, Settings
from pmmc.models.registry import ModelRegistry
from pmmc.schemas import ModelEndpointKind


class ModelSmokeConfigurationError(ValueError):
    """Live gateway smoke configuration is incomplete or structurally invalid."""


def _secret_is_configured(value: object | None) -> bool:
    if value is None:
        return False
    getter = getattr(value, "get_secret_value", None)
    resolved = getter() if getter is not None else value
    return bool(str(resolved).strip())


def _is_safe_http_endpoint(value: str | None) -> bool:
    if value is None or not value.strip():
        return False
    parsed = urlsplit(value.strip())
    return bool(
        parsed.scheme in {"http", "https"}
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def validate_model_smoke_configuration(
    settings: Settings,
    registry: ModelRegistry,
    *,
    chat_alias: str,
    text_embedding_alias: str,
    multimodal_embedding_alias: str,
) -> dict[str, object]:
    """Validate live-service settings without returning endpoint or credential values."""

    if settings.mode is AppMode.PAPER:
        registry.assert_paper_ready()

    alias_contracts = (
        (
            chat_alias,
            ModelEndpointKind.LITELLM_OPENAI,
            "LITELLM_BASE_URL",
            settings.litellm_base_url,
        ),
        (
            text_embedding_alias,
            ModelEndpointKind.VLLM_OPENAI,
            "VLLM_TEXT_EMBEDDING_BASE_URL",
            settings.vllm_text_embedding_base_url,
        ),
        (
            multimodal_embedding_alias,
            ModelEndpointKind.VLLM_V2_EMBED,
            "VLLM_MULTIMODAL_EMBEDDING_BASE_URL",
            settings.vllm_multimodal_embedding_base_url,
        ),
    )
    missing: list[str] = []
    invalid: list[str] = []
    aliases: dict[str, str] = {}
    for alias, expected_kind, variable_name, endpoint in alias_contracts:
        try:
            entry = registry.get(alias)
        except KeyError:
            invalid.append(f"alias:{alias}")
            continue
        if entry.endpoint_kind is not expected_kind:
            invalid.append(f"alias:{alias}:endpoint_kind")
        aliases[alias] = entry.endpoint_kind.value
        if endpoint is None or not endpoint.strip():
            missing.append(variable_name)
        elif not _is_safe_http_endpoint(endpoint):
            invalid.append(variable_name)

    if not _secret_is_configured(settings.litellm_api_key):
        missing.append("LITELLM_API_KEY")
    if not settings.langfuse_enabled:
        missing.append("LANGFUSE_ENABLED=true")
    if not _secret_is_configured(settings.langfuse_public_key):
        missing.append("LANGFUSE_PUBLIC_KEY")
    if not _secret_is_configured(settings.langfuse_secret_key):
        missing.append("LANGFUSE_SECRET_KEY")

    problems: list[str] = []
    if missing:
        problems.append("missing or empty: " + ", ".join(sorted(set(missing))))
    if invalid:
        problems.append("invalid: " + ", ".join(sorted(set(invalid))))
    if problems:
        raise ModelSmokeConfigurationError("; ".join(problems))

    return {
        "mode": settings.mode.value,
        "aliases": dict(sorted(aliases.items())),
        "litellm_credentials": "configured",
        "vllm_credentials": (
            "configured" if _secret_is_configured(settings.vllm_api_key) else "not_required"
        ),
        "langfuse": "configured",
    }
