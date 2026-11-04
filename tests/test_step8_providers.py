"""Offline SDK/HTTP regression tests: no provider or paid model calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from PIL import Image

from pmmc.infrastructure.object_store import ObjectStore
from pmmc.infrastructure.settings import Settings
from pmmc.models.factory import (
    check_model_endpoint,
    create_generation_gateway,
    create_token_counter,
)
from pmmc.models.gateways import OpenAICompatibleGenerationGateway, RoutedGenerationGateway
from pmmc.models.images import ObjectStoreImageUrlResolver
from pmmc.models.provider_config import (
    ProviderConfig,
    ProviderKind,
    prepare_provider_answer_request,
)
from pmmc.models.provider_images import (
    ImageBytes,
    ObjectStoreImageBytesResolver,
    validate_image_bytes,
)
from pmmc.models.provider_schema import project_provider_response_format
from pmmc.models.provider_token_budget import ProviderTokenBudgetCounter
from pmmc.models.providers import (
    AnthropicGenerationGateway,
    OpenAIResponsesGenerationGateway,
    ProviderCallError,
    ProviderGenerationGateway,
    QwenGenerationGateway,
    provider_output_budget,
)
from pmmc.models.registry import ModelRegistryEntry, load_model_registry
from pmmc.runtime.answering import answer_format_retry_request
from pmmc.schemas import GenerationRequest, ImageRef, ModelContentPart, ModelMessage
from pmmc.schemas.answer_output import TYPED_ANSWER_CONTRACT_VERSION, answer_response_schema

ROOT = Path(__file__).parents[1]


def test_staged_counter_preserves_native_request_and_rejects_wrong_bytes(tmp_path):
    from unittest.mock import AsyncMock

    from pmmc.models.staged_token_counter import StagedImageTokenCounter

    buffer = BytesIO()
    Image.new("RGB", (32, 64)).save(buffer, "PNG")
    data = buffer.getvalue()
    image = ImageRef(image_id="native", storage_uri="s3://runtime/image",
                     sha256=hashlib.sha256(data).hexdigest(), mime_type="image/png")
    request = GenerationRequest(request_id="count", model_alias="model", max_output_tokens=4096,
        messages=[ModelMessage(role="user", parts=[ModelContentPart(kind="image", image=image)])])
    before = request.model_dump_json()

    class Exact:
        alias, revision, estimated = "processor", "locked", False

        def count_request(self, local):
            ref = local.messages[0].parts[0].image
            assert Path(ref.local_path).read_bytes() == data
            assert (ref.width, ref.height) == (32, 64)
            return 37604

    resolver = SimpleNamespace(resolve=AsyncMock(return_value=ImageBytes(data, "image/png")))
    counter = StagedImageTokenCounter(Exact(), resolver, tmp_path)
    asyncio.run(counter.prepare_images([image, image]))
    assert counter.count_request(request) == 37604
    assert request.model_dump_json() == before
    assert resolver.resolve.await_count == 1
    bad = image.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="checksum mismatch"):
        asyncio.run(counter.prepare_images([bad]))
    assert not (tmp_path / ("0" * 64)).exists()
ALIASES = {
    "qwen_vllm": "answer-qwen3.5-9b",
    "qwen27": "answer-qwen3.5-27b",
    "openai": "answer-gpt-6-luna",
    "anthropic": "answer-claude-haiku-5.5",
}
ANSWER = '{"answer":"blue","supported":true,"used_evidence_aliases":["E1"],"abstain_reason":null}'


def registry():  # type: ignore[no-untyped-def]
    return load_model_registry(ROOT / "configs/models/step8-providers.yaml")


def request_for(kind: str, **updates) -> GenerationRequest:  # type: ignore[no-untyped-def]
    values = dict(
        request_id="step8-local-request",
        model_alias=ALIASES[kind],
        messages=[
            ModelMessage(role="system", parts=[ModelContentPart(kind="text", text="Answer JSON")]),
            ModelMessage(role="user", parts=[ModelContentPart(kind="text", text="What color?")]),
        ],
        max_output_tokens=512,
        provider_max_output_tokens=1536,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        thinking_mode="disabled",
        timeout_seconds=12,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": answer_response_schema(["E1", "E2"]),
            },
        },
    )
    values.update(updates)
    return GenerationRequest(**values)


def raw_response(kind: str, *, text: str = ANSWER, finish: str = "stop") -> dict:
    model = registry().get(ALIASES[kind]).provider_model
    if kind == "openai":
        content = [{"type": "output_text", "text": text, "annotations": []}]
        if finish == "refusal":
            content = [{"type": "refusal", "refusal": "refused"}]
        return {
            "id": "response-id",
            "object": "response",
            "created_at": 1,
            "model": model,
            "status": "incomplete" if finish == "length" else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if finish == "length" else None,
            "output": [
                {
                    "type": "message",
                    "id": "msg1",
                    "role": "assistant",
                    "status": "completed",
                    "content": content,
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {"cached_tokens": 30},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        }
    if kind == "anthropic":
        return {
            "id": "response-id",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_sequence": None,
            "stop_reason": {"stop": "end_turn", "length": "max_tokens", "refusal": "refusal"}[
                finish
            ],
            "usage": {
                "input_tokens": 70,
                "output_tokens": 20,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 5,
            },
        }
    return {
        "id": "response-id",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop" if finish == "refusal" else finish,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "refusal": "refused" if finish == "refusal" else None,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 30},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


async def invoke(
    kind: str, *, request=None, raw=None, resolver=None, status=200, error=None, captured=None
):  # type: ignore[no-untyped-def]
    captured = [] if captured is None else captured

    def transport(http_request: httpx.Request) -> httpx.Response:
        captured.append(http_request)
        if error is not None:
            raise error
        return httpx.Response(
            status,
            json=raw if raw is not None else raw_response(kind),
            headers={"x-request-id": "http-request-id", "request-id": "http-request-id"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
        cls = AsyncAnthropic if kind == "anthropic" else AsyncOpenAI
        client = cls(
            api_key="offline-test-key",
            base_url="https://mock-provider.invalid" + ("" if kind == "anthropic" else "/v1"),
            http_client=http,
            max_retries=0,
        )
        if kind in {"openai", "anthropic"}:
            adapter = (
                OpenAIResponsesGenerationGateway if kind == "openai" else AnthropicGenerationGateway
            )
            gateway = adapter(
                registry(), api_key="offline-test-key", client=client, image_bytes_resolver=resolver
            )
        else:
            gateway = create_generation_gateway(
                Settings(_env_file=None),
                registry(),
                ALIASES[kind],
                provider_client=client,
                image_bytes_resolver=resolver,
            )
        result = await gateway.generate(request or request_for(kind))
    return result, captured


def image_payload() -> ImageBytes:
    stream = BytesIO()
    Image.new("RGB", (2, 2), "blue").save(stream, format="PNG")
    return ImageBytes(stream.getvalue(), "image/png")


class BytesResolver:
    async def resolve(self, image):  # type: ignore[no-untyped-def]
        return image_payload()


@pytest.mark.parametrize("kind", ALIASES)
def test_factory_sdk_transport_schema_images_identity_and_usage(kind: str) -> None:
    request = request_for(kind)
    request.messages[-1].parts.append(
        ModelContentPart(
            kind="image",
            image=ImageRef(
                image_id="image1",
                storage_uri="s3://private/image1",
                sha256=hashlib.sha256(image_payload().data).hexdigest(),
                width=2,
                height=2,
                mime_type="image/png",
            ),
        )
    )
    canonical = deepcopy(request.response_format)
    result, calls = asyncio.run(invoke(kind, request=request, resolver=BytesResolver()))
    assert len(calls) == 1
    sent = json.loads(calls[0].content)
    assert sent["model"] == registry().get(ALIASES[kind]).provider_model
    assert "s3://" not in calls[0].content.decode()
    assert calls[0].headers["x-client-request-id"] == request.request_id
    assert result.provider_request_id == "http-request-id"
    assert result.metadata["provider_response_id"] == "response-id"
    assert result.metadata["canonical_schema_validated"] is True
    assert result.metadata["provider_max_output_tokens"] == 1536
    assert result.metadata["public_max_output_tokens"] == 512
    assert request.max_output_tokens == 512
    assert request.response_format == canonical
    assert result.usage.metadata["cached_input_tokens"] == 30
    assert result.usage.total_tokens == (125 if kind == "anthropic" else 120)
    assert result.text == ANSWER
    if kind == "openai":
        assert calls[0].url.path == "/v1/responses"
        assert sent["reasoning"] == {"effort": "low"}
        assert sent["max_output_tokens"] == 1536 and sent["store"] is False
        assert not {"temperature", "top_p", "seed", "chat_template_kwargs"} & sent.keys()
        assert sent["input"][1]["content"][1]["image_url"].startswith("data:image/png;base64,")
        assert "anyOf" not in sent["text"]["format"]["schema"]
        assert sent["text"]["format"]["strict"] is True
        assert result.usage.metadata["reasoning_tokens"] == 5
    elif kind == "anthropic":
        assert calls[0].url.path == "/v1/messages"
        assert sent["thinking"] == {"type": "disabled"}
        assert not {
            "temperature", "top_p", "top_k", "seed", "chat_template_kwargs", "response_format",
        } & sent.keys()
        assert sent["output_config"]["effort"] == "low"
        assert sent["service_tier"] == "standard_only"
        assert result.metadata["ignored_sampling_fields"] == ["temperature", "top_p", "seed"]
        assert result.metadata["provider_effort"] == "low"
        assert sent["max_tokens"] == 1536
        assert sent["system"] == [{"type": "text", "text": "Answer JSON"}]
        assert sent["messages"][0]["content"][1]["source"]["data"] == image_payload().base64
        assert sent["output_config"]["format"]["type"] == "json_schema"
        assert result.usage.metadata["cache_creation_input_tokens"] == 5
        assert result.usage.metadata["reasoning_tokens"] == 0
    else:
        assert calls[0].url.path == "/v1/chat/completions"
        assert sent["chat_template_kwargs"] == {"enable_thinking": False}
        assert sent["max_tokens"] == 1536 and sent["seed"] == 42
        assert sent["messages"][1]["content"][1]["image_url"]["url"].startswith("data:")


@pytest.mark.parametrize("kind", ALIASES)
@pytest.mark.parametrize("finish", ["refusal", "length"])
def test_refusal_and_length_are_not_validated_or_repaired(kind: str, finish: str) -> None:
    result, calls = asyncio.run(
        invoke(kind, raw=raw_response(kind, text="not json", finish=finish))
    )
    assert result.finish_reason == finish
    assert result.metadata["canonical_schema_validated"] is False
    assert result.usage.output_tokens == 20
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ALIASES)
@pytest.mark.parametrize(
    "payload",
    [
        {
            "answer": "blue",
            "supported": True,
            "used_evidence_aliases": ["E1", "E1"],
            "abstain_reason": None,
        },
        {
            "answer": None,
            "supported": True,
            "used_evidence_aliases": ["E1"],
            "abstain_reason": None,
        },
        {
            "answer": "blue",
            "supported": True,
            "used_evidence_aliases": ["unknown"],
            "abstain_reason": None,
        },
    ],
)
def test_full_canonical_postvalidation_preserves_usage_on_failure(kind: str, payload: dict) -> None:
    calls: list[httpx.Request] = []
    with pytest.raises(ProviderCallError, match="CanonicalSchemaViolation") as error:
        asyncio.run(invoke(kind, raw=raw_response(kind, text=json.dumps(payload)), captured=calls))
    assert error.value.outcome == "completed"
    assert error.value.result is not None
    assert error.value.result.usage.output_tokens == 20
    assert error.value.provider_request_id == "http-request-id"
    assert error.value.retryable is False
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
def test_schema_projection_preserves_property_names_and_postvalidates_constraints(
    kind: ProviderKind,
) -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "uniqueItems": {
                "type": "array",
                "items": {"type": "integer", "minimum": 10},
                "uniqueItems": True,
                "minItems": 2,
                "maxItems": 3,
            },
            "answer": {"type": "string", "minLength": 2},
        },
        "required": ["uniqueItems", "answer"],
    }
    response_format = {"type": "json_schema", "json_schema": {"schema": schema}}
    original = deepcopy(response_format)
    projected, diagnostics = project_provider_response_format(response_format, kind)
    assert "uniqueItems" in projected["json_schema"]["schema"]["properties"]
    assert "uniqueItems" not in projected["json_schema"]["schema"]["properties"]["uniqueItems"]
    assert response_format == original
    assert diagnostics["canonical_postvalidation_required"] is True
    with pytest.raises(ProviderCallError, match="CanonicalSchemaViolation"):
        asyncio.run(
            invoke(
                kind,
                request=request_for(kind, response_format=response_format),
                raw=raw_response(kind, text='{"uniqueItems":[1,2],"answer":"x"}'),
            )
        )


@pytest.mark.parametrize("kind", ALIASES)
def test_no_model_identity_substitution(kind: str) -> None:
    raw = raw_response(kind)
    raw["model"] = "different-model"
    with pytest.raises(ProviderCallError, match="ProviderModelIdentityMismatch") as error:
        asyncio.run(invoke(kind, raw=raw))
    assert error.value.result is not None
    assert error.value.result.actual_model_id == "different-model"
    assert error.value.outcome == "completed"


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (400, "failed"),
        (401, "failed"),
        (404, "failed"),
        (429, "failed"),
        (408, "unknown"),
        (500, "unknown"),
    ],
)
def test_http_failure_classification_is_nonretrying_and_redacted(
    kind: str, status: int, outcome: str
) -> None:
    calls: list[httpx.Request] = []
    with pytest.raises(ProviderCallError) as error:
        asyncio.run(
            invoke(
                kind,
                status=status,
                raw={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "private-prompt-and-secret",
                    }
                },
                captured=calls,
            )
        )
    assert error.value.outcome == outcome
    assert error.value.provider_status_code == status
    assert error.value.provider_request_id == "http-request-id"
    assert error.value.retryable is False
    assert "private-prompt-and-secret" not in json.dumps(error.value.safe_diagnostics())
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout("secret"), httpx.WriteTimeout("secret"), RuntimeError("secret")]
)
def test_timeout_and_unknown_failure_never_assume_no_spend(kind: str, failure: Exception) -> None:
    calls: list[httpx.Request] = []
    with pytest.raises(ProviderCallError) as error:
        asyncio.run(invoke(kind, error=failure, captured=calls))
    assert error.value.outcome == "unknown"
    assert error.value.retryable is False
    assert error.value.result is None
    assert len(calls) == 1


def test_registry_policy_fixed_helpers_and_no_floating_ids() -> None:
    models = registry()
    assert len(models.models) == 11
    for role in ("compiler", "reranker", "grounding", "judge"):
        entry = models.get(f"{role}-qwen3.5-9b")
        assert entry.provider_model == "Qwen/Qwen3.5-9B"
        values = entry.model_dump()
        values.update(
            provider_model="Qwen/Qwen3.5-27B", expected_actual_model_id="Qwen/Qwen3.5-27B"
        )
        with pytest.raises(ValueError, match="fixed"):
            ModelRegistryEntry.model_validate(values)
    entry = models.get(ALIASES["openai"]).model_dump()
    entry.update(provider_model="gpt-6-luna-latest", expected_actual_model_id="gpt-6-luna-latest")
    with pytest.raises(ValueError, match="concrete"):
        ModelRegistryEntry.model_validate(entry)
    with pytest.raises(ValueError, match="low"):
        ProviderConfig(kind="openai", reasoning_mode="off", max_output_tokens=4096)
    with pytest.raises(ValueError, match="immutable"):
        models.assert_paper_ready()


def test_provider_budget_separate_from_public_payload_and_bounded() -> None:
    assert provider_output_budget(request_for("openai"), 4096) == 1536
    assert (
        provider_output_budget(request_for("openai", provider_max_output_tokens=None), 4096) == 4096
    )
    with pytest.raises(ValueError, match="public"):
        request_for("openai", provider_max_output_tokens=511)
    calls: list[httpx.Request] = []
    with pytest.raises(ValueError, match="ceiling"):
        asyncio.run(
            invoke(
                "openai",
                request=request_for("openai", provider_max_output_tokens=8192),
                captured=calls,
            )
        )
    assert not calls


def test_legacy_registry_serialization_and_factory_unchanged() -> None:
    models = load_model_registry(ROOT / "configs/models/dev.yaml")
    assert all("provider_config" not in entry.model_dump() for entry in models.models)
    request = request_for("openai", provider_max_output_tokens=None)
    assert "provider_max_output_tokens" not in request.model_dump()
    gateway = create_generation_gateway(
        Settings(_env_file=None, LITELLM_BASE_URL="http://localhost"), models, "answer-qwen3.5-9b"
    )
    assert isinstance(gateway, OpenAICompatibleGenerationGateway)
    asyncio.run(gateway.client.close())


def test_factory_classes_clients_retries_and_credentials() -> None:
    classes = {
        "openai": OpenAIResponsesGenerationGateway,
        "anthropic": AnthropicGenerationGateway,
        "qwen_vllm": QwenGenerationGateway,
    }
    for kind, cls in classes.items():
        settings = Settings(_env_file=None, OPENAI_API_KEY="test", ANTHROPIC_API_KEY="test")
        if kind == "qwen_vllm":
            gateway = create_generation_gateway(
                settings, registry(), ALIASES[kind], provider_base_url="https://test.invalid/v1"
            )
        else:
            gateway = cls(registry(), api_key="test", base_url="https://test.invalid")
        assert isinstance(gateway, cls)
        assert isinstance(gateway, ProviderGenerationGateway)
        assert gateway.client.max_retries == 0
        asyncio.run(gateway.client.close())
    with pytest.raises(ValueError, match="accounting"):
        create_generation_gateway(
            Settings(_env_file=None, OPENAI_API_KEY=None), registry(), ALIASES["openai"]
        )


def test_native_health_does_not_send_inference_and_counter_does_not_download() -> None:
    settings = Settings(_env_file=None)
    result = asyncio.run(check_model_endpoint(settings, registry(), ALIASES["anthropic"]))
    assert result["inference_performed"] is False and result["healthy"] is None
    for kind in ("openai", "anthropic"):
        counter = create_token_counter(settings, registry(), ALIASES[kind])
        assert isinstance(counter, ProviderTokenBudgetCounter)
        assert counter.estimated is True
        assert counter.validate_request(request_for(kind)).fits


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
@pytest.mark.parametrize("output_reserve", [None, 1536, 512])
def test_native_dispatch_enforces_full_32768_boundary(kind, output_reserve) -> None:
    counter = ProviderTokenBudgetCounter(registry().get(ALIASES[kind]))
    request = request_for(
        kind,
        provider_max_output_tokens=output_reserve,
        messages=[
            ModelMessage(role="user", parts=[ModelContentPart(kind="text", text="x")]),
        ],
    )
    room = 32768 - counter.request_budget(request).total_tokens
    request.messages[0].parts[0].text = "x" * (room + 1)
    assert counter.request_budget(request).total_tokens == 32768
    _, calls = asyncio.run(invoke(kind, request=request))
    assert len(calls) == 1
    sent = json.loads(calls[0].content)
    assert sent["max_output_tokens" if kind == "openai" else "max_tokens"] == (
        output_reserve or 4096
    )
    request.messages[0].parts[0].text += "x"
    calls = []
    with pytest.raises(ValueError, match="shared 32768"):
        asyncio.run(invoke(kind, request=request, captured=calls))
    assert not calls


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
@pytest.mark.parametrize("oversize", ["text", "schema", "image", "retry"])
def test_native_dispatch_budget_fails_before_image_io_or_transport(kind, oversize) -> None:
    class NoImageIo:
        async def resolve(self, image):
            raise AssertionError("budget rejection must precede image byte resolution")

    request = request_for(kind)
    request.messages[-1].parts.append(
        ModelContentPart(
            kind="image",
            image=ImageRef(
                image_id="query-image",
                width=2,
                height=2,
                mime_type="image/png",
                sha256="a" * 64,
            ),
        )
    )
    if oversize == "text":
        request.messages[-1].parts[0].text = "x" * 32768
    elif oversize == "schema":
        assert request.response_format is not None
        request.response_format["json_schema"]["schema"]["description"] = "x" * 32768
    elif oversize == "image":
        image = request.messages[-1].parts[-1].image
        assert image is not None
        image.width = 8000
        image.height = 8000
    else:
        counter = ProviderTokenBudgetCounter(registry().get(ALIASES[kind]))
        room = 32768 - counter.request_budget(request).total_tokens
        part = request.messages[-1].parts[0]
        assert part.text is not None
        part.text += "x" * room
        assert counter.validate_request(request).total_tokens == 32768
        request = answer_format_retry_request(request)
    calls: list[httpx.Request] = []
    with pytest.raises(ValueError, match="shared 32768"):
        asyncio.run(invoke(kind, request=request, resolver=NoImageIo(), captured=calls))
    assert not calls


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
@pytest.mark.parametrize("missing", ["mime_type", "sha256"])
def test_native_dispatch_rejects_missing_image_metadata_before_io(kind, missing) -> None:
    class NoImageIo:
        async def resolve(self, image):
            raise AssertionError("invalid image metadata must be rejected before reading bytes")

    request = request_for(kind)
    image = ImageRef(
        image_id="query-image", width=2, height=2, mime_type="image/png", sha256="a" * 64
    )
    request.messages[-1].parts.append(
        ModelContentPart(
            kind="image",
            image=image.model_copy(update={missing: None}),
        )
    )
    calls: list[httpx.Request] = []
    with pytest.raises(ValueError, match="dimensions|mime_type|SHA256"):
        asyncio.run(invoke(kind, request=request, resolver=NoImageIo(), captured=calls))
    assert not calls


@pytest.mark.parametrize("kind", ["qwen_vllm", "qwen27"])
def test_qwen_dispatch_does_not_adopt_native_budget_policy(kind, monkeypatch) -> None:
    def no_native_counter(*args):
        raise AssertionError("Qwen must retain its historical tokenizer budget path")

    monkeypatch.setattr("pmmc.models.providers.ProviderTokenBudgetCounter", no_native_counter)
    _, calls = asyncio.run(invoke(kind))
    assert len(calls) == 1


def test_object_store_bytes_and_existing_factory_resolver_integration() -> None:
    class Store:
        async def get_bytes(self, key):  # type: ignore[no-untyped-def]
            assert key == "image.png"
            return image_payload().data

        async def presign_get(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("native providers must not receive private presigned URLs")

    store = cast(ObjectStore, Store())
    resolver = ObjectStoreImageBytesResolver(store, bucket="runtime")
    image = ImageRef(image_id="image", storage_uri="s3://runtime/image.png", width=2, height=2)
    assert asyncio.run(resolver.resolve(image)) == image_payload()
    for updates in (
        {"storage_uri": "s3://other/image.png"},
        {"sha256": "0" * 64},
        {"width": 3},
        {"mime_type": "image/jpeg"},
    ):
        with pytest.raises(ValueError):
            asyncio.run(resolver.resolve(image.model_copy(update=updates)))
    gateway = create_generation_gateway(
        Settings(_env_file=None),
        registry(),
        ALIASES["qwen_vllm"],
        provider_client=SimpleNamespace(),
        image_url_resolver=ObjectStoreImageUrlResolver(store, bucket="runtime"),
    )
    assert isinstance(gateway.image_bytes_resolver, ObjectStoreImageBytesResolver)


def test_private_urls_are_never_forwarded_without_bytes() -> None:
    request = request_for("openai")
    request.messages[-1].parts.append(
        ModelContentPart(
            kind="image",
            image=ImageRef(
                image_id="image",
                storage_uri="http://127.0.0.1/private",
                sha256=hashlib.sha256(image_payload().data).hexdigest(),
                width=2,
                height=2,
                mime_type="image/png",
            ),
        )
    )
    calls: list[httpx.Request] = []
    with pytest.raises(ValueError, match="image_bytes_resolver"):
        asyncio.run(invoke("openai", request=request, captured=calls))
    assert not calls


@pytest.mark.parametrize("kind", ["openai", "anthropic", "qwen_vllm", "qwen27"])
def test_checksum_pinned_legacy_mime_is_audited_without_changing_bytes(kind, capsys):
    payload = image_payload()
    class Store:
        async def get_bytes(self, key):
            return payload.data
    resolver = ObjectStoreImageBytesResolver(
        cast(ObjectStore, Store()), bucket="runtime", allow_verified_mime_correction=True,
    )
    image = ImageRef(
        image_id="legacy", storage_uri="s3://runtime/image.jpg", mime_type="image/jpeg",
        sha256=hashlib.sha256(payload.data).hexdigest(), width=2, height=2,
    )
    request = request_for(kind)
    request.messages[-1].parts.append(ModelContentPart(kind="image", image=image))
    result, calls = asyncio.run(invoke(kind, request=request, resolver=resolver))
    sent = json.loads(calls[0].content)
    assert payload.base64 in json.dumps(sent)
    assert "image/png" in json.dumps(sent) and "image/jpeg" not in json.dumps(sent)
    assert request.messages[-1].parts[-1].image.mime_type == "image/jpeg"
    assert result.metadata["native_image_transport"] == [{
        "policy": "sha256-verified-native-mime-v1", "sha256": image.sha256,
        "declared_mime": "image/jpeg", "transport_mime": "image/png",
        "mime_corrected": True, "bytes_modified": False,
    }]
    assert "NATIVE_IMAGE_MIME_CORRECTED" in capsys.readouterr().out
    for changes in ({"sha256": None}, {"sha256": "0" * 64}, {"width": 3}):
        with pytest.raises(ValueError):
            asyncio.run(resolver.resolve(image.model_copy(update=changes)))
    with pytest.raises(ValueError, match="media type mismatch"):
        validate_image_bytes(image, payload)
    with pytest.raises(ValueError, match="media type mismatch"):
        validate_image_bytes(
            image, ImageBytes(payload.data, "image/jpeg"), allow_verified_mime_correction=True,
        )


def test_step8_managed_replica_routing_does_not_change_old_replica_policy() -> None:
    alias = ALIASES["qwen_vllm"]
    settings = Settings(
        _env_file=None,
        runtime_mode="managed_local",
        PMMC_ACTIVE_RUNTIME_ID="runtime1",
        PMMC_ACTIVE_RUNTIME_ENDPOINT_POOLS={
            alias: [
                {
                    "replica_index": i,
                    "runtime_service_id": f"runtime1:replica{i}",
                    "endpoint": f"http://127.0.0.1:{8100 + i}/v1",
                }
                for i in range(8)
            ]
        },
    )
    gateway = create_generation_gateway(settings, registry(), alias)
    assert isinstance(gateway, RoutedGenerationGateway)
    assert len(gateway.gateways) == 8
    for _, _, replica in gateway.gateways:
        assert isinstance(replica, QwenGenerationGateway)
        asyncio.run(replica.client.close())
    legacy = load_model_registry(ROOT / "configs/models/dev.yaml")
    gateway = create_generation_gateway(settings, legacy, alias)
    assert len(gateway.gateways) == 8
    for _, _, replica in gateway.gateways:
        assert isinstance(replica, OpenAICompatibleGenerationGateway)
        assert replica.provider_model_overrides == {alias: "Qwen/Qwen3.5-9B"}
        asyncio.run(replica.client.close())


@pytest.mark.parametrize("kind", ALIASES)
@pytest.mark.parametrize(
    "text",
    [
        '{"answer":"blue","answer":"red"}',
        '{"answer":NaN}',
        "```json\n{}\n```",
    ],
)
def test_invalid_json_is_rejected_without_repair(kind: str, text: str) -> None:
    with pytest.raises(ProviderCallError, match="CanonicalSchemaViolation"):
        asyncio.run(invoke(kind, raw=raw_response(kind, text=text)))


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
def test_assistant_history_and_leading_system_have_valid_content_types(kind: str) -> None:
    request = request_for(kind)
    request.messages.insert(
        2,
        ModelMessage(
            role="assistant",
            parts=[ModelContentPart(kind="text", text="Earlier answer")],
        ),
    )
    request.messages.append(
        ModelMessage(
            role="user",
            parts=[ModelContentPart(kind="text", text="Now answer again")],
        )
    )
    _, calls = asyncio.run(invoke(kind, request=request))
    sent = json.loads(calls[0].content)
    if kind == "openai":
        assert sent["input"][2]["content"] == [{"type": "input_text", "text": "Earlier answer"}]
    else:
        assert sent["messages"][1]["content"] == [{"type": "text", "text": "Earlier answer"}]


@pytest.mark.parametrize("kind", ALIASES)
def test_provider_errors_do_not_masquerade_as_success(kind: str) -> None:
    raw = raw_response(kind)
    if kind == "openai":
        raw["status"] = "failed"
    elif kind == "anthropic":
        raw["stop_reason"] = "pause_turn"
    else:
        raw["choices"][0]["finish_reason"] = "tool_calls"
    with pytest.raises(ProviderCallError, match="ProviderGenerationFailure") as error:
        asyncio.run(invoke(kind, raw=raw))
    assert error.value.outcome == "completed"
    assert error.value.result is not None
    assert error.value.result.usage.output_tokens == 20


@pytest.mark.parametrize("kind", ALIASES)
def test_missing_usage_is_unknown_not_evidence_of_zero_spend(kind: str) -> None:
    raw = raw_response(kind)
    raw.pop("usage")
    result, _ = asyncio.run(invoke(kind, raw=raw))
    assert result.usage.metadata["usage_reported"] is False
    assert result.usage.total_tokens == 0


def test_recursive_and_external_schemas_fail_before_transport() -> None:
    schemas = [
        {"type": "object", "properties": {"a": {"$ref": "https://private/schema"}}},
        {
            "type": "object",
            "$defs": {
                "Loop": {
                    "type": "object",
                    "properties": {
                        "next": {"$ref": "#/$defs/Loop"},
                    },
                }
            },
            "properties": {"first": {"$ref": "#/$defs/Loop"}},
        },
    ]
    for schema in schemas:
        calls: list[httpx.Request] = []
        request = request_for(
            "anthropic",
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": schema},
            },
        )
        with pytest.raises(ValueError):
            asyncio.run(invoke("anthropic", request=request, captured=calls))
        assert not calls


def test_image_corruption_stops_before_model_transport() -> None:
    class CorruptResolver:
        async def resolve(self, image):  # type: ignore[no-untyped-def]
            return ImageBytes(b"not-image-bytes", "image/png")

    request = request_for("anthropic")
    request.messages[-1].parts.append(
        ModelContentPart(
            kind="image",
            image=ImageRef(
                image_id="broken",
                width=2,
                height=2,
                mime_type="image/png",
                sha256=hashlib.sha256(b"not-image-bytes").hexdigest(),
            ),
        )
    )
    calls: list[httpx.Request] = []
    with pytest.raises(OSError):
        asyncio.run(
            invoke("anthropic", request=request, resolver=CorruptResolver(), captured=calls)
        )
    assert not calls


def test_injected_sdk_client_cannot_enable_automatic_retries() -> None:
    for kind, cls in (("openai", AsyncOpenAI), ("anthropic", AsyncAnthropic)):
        client = cls(api_key="test", max_retries=5)
        adapter = (
            OpenAIResponsesGenerationGateway if kind == "openai" else AnthropicGenerationGateway
        )
        gateway = adapter(registry(), api_key="test", client=client)
        assert gateway.client.max_retries == 0
        asyncio.run(client.close())


def test_gpt_request_low_and_provider_default_use_fixed_low_policy() -> None:
    for mode in ("low", "provider_default", "disabled"):
        result, calls = asyncio.run(
            invoke("openai", request=request_for("openai", thinking_mode=mode))
        )
        assert json.loads(calls[0].content)["reasoning"] == {"effort": "low"}
        assert result.metadata["requested_thinking_mode"] == mode
        assert result.metadata["effective_reasoning_mode"] == "low"
    rejected_calls: list[httpx.Request] = []
    with pytest.raises(ValueError, match="reasoning mode"):
        asyncio.run(
            invoke(
                "anthropic",
                request=request_for("anthropic", thinking_mode="low"),
                captured=rejected_calls,
            )
        )
    assert not rejected_calls


def test_ambiguous_paid_failure_does_not_fan_out_across_replicas() -> None:
    calls: list[str] = []

    class Replica:
        async def generate(self, request):  # type: ignore[no-untyped-def]
            calls.append(request.request_id)
            raise ProviderCallError("APITimeoutError", outcome="unknown")

    gateway = RoutedGenerationGateway([(i, f"r{i}", Replica()) for i in range(8)])
    with pytest.raises(ProviderCallError):
        asyncio.run(gateway.generate(request_for("qwen_vllm")))
    assert len(calls) == 1


@pytest.mark.parametrize("role", ["compiler", "doubter", "reranker", "grounding", "judge"])
@pytest.mark.parametrize(
    "contract", [None, "strict-answer-contract-v2", TYPED_ANSWER_CONTRACT_VERSION]
)
@pytest.mark.parametrize("explicit", [False, True])
def test_helper_role_output_budgets_are_never_limited_to_answer_512(
    role: str,
    contract: str | None,
    explicit: bool,
) -> None:
    alias = f"{'compiler' if role == 'doubter' else role}-qwen3.5-9b"
    config = registry().get(alias).provider_config
    assert config is not None
    request = request_for(
        "qwen_vllm",
        model_alias=alias,
        max_output_tokens=2048,
        provider_max_output_tokens=2048 if explicit else None,
        metadata={"model_role": role, "answer_contract_version": contract},
    )
    assert prepare_provider_answer_request(request, config) is request
    total = 2048
    assert provider_output_budget(request, config.max_output_tokens, config=config) == total
    _, calls = asyncio.run(invoke("qwen_vllm", request=request))
    assert json.loads(calls[0].content)["max_tokens"] == total > 512


@pytest.mark.parametrize("input_tokens", [19933, 19225, 19775])
def test_questioner_4k_output_does_not_reserve_registry_16k(input_tokens: int) -> None:
    """Reproduce all three 2026-10-05 rejected unit sizes at the HTTP boundary."""
    request = request_for(
        "qwen_vllm", model_alias="compiler-qwen3.5-9b",
        max_output_tokens=4096, provider_max_output_tokens=None,
        metadata={"task": "pmmc_questioner"},
    )
    config = registry().get(request.model_alias).provider_config
    assert config is not None and config.max_output_tokens == 16384
    original = request.model_dump(mode="json")
    result, calls = asyncio.run(invoke("qwen_vllm", request=request))
    actual = json.loads(calls[0].content)["max_tokens"]
    assert actual == 4096
    assert input_tokens + actual <= 32768 < input_tokens + config.max_output_tokens
    assert result.metadata["provider_max_output_tokens"] == actual
    assert result.metadata["provider_output_ceiling"] == 16384
    assert request.model_dump(mode="json") == original


@pytest.mark.parametrize("public,explicit", [(16385, None), (4096, 16385), (4096, 2048), (4096, 0)])
def test_qwen_output_budget_violations_never_reach_transport(
    public: int, explicit: int | None,
) -> None:
    calls: list[httpx.Request] = []
    request = request_for(
        "qwen_vllm", model_alias="compiler-qwen3.5-9b",
        max_output_tokens=public, provider_max_output_tokens=None,
    ).model_copy(update={"provider_max_output_tokens": explicit})
    with pytest.raises(ValueError, match="budget"):
        asyncio.run(invoke("qwen_vllm", request=request, captured=calls))
    assert calls == []


@pytest.mark.parametrize("message,category", [
    ("maximum context length is 32768 tokens; PRIVATE QUERY", "context_length_exceeded"),
    ("max_tokens exceeds context length; PRIVATE QUERY", "context_length_exceeded"),
    ("Invalid JSON schema: PRIVATE QUERY", "structured_output_rejected"),
    ("Cannot decode image: https://secret.invalid/key", "image_rejected"),
    ("PRIVATE QUERY https://secret.invalid/key", "unclassified"),
])
def test_rejection_diagnostics_keep_category_and_budget_without_body(
    message: str, category: str,
) -> None:
    request = request_for(
        "qwen_vllm", model_alias="compiler-qwen3.5-9b",
        max_output_tokens=4096, provider_max_output_tokens=None,
    )
    with pytest.raises(ProviderCallError) as failure:
        asyncio.run(invoke("qwen_vllm", request=request, status=400, raw={
            "error": {"message": message, "type": "BadRequestError", "code": 400},
        }))
    diagnostic = failure.value.safe_diagnostics()
    assert diagnostic["provider_error_category"] == category
    assert diagnostic["requested_output_tokens"] == diagnostic["provider_output_tokens"] == 4096
    assert diagnostic["provider_output_ceiling"] == 16384
    assert diagnostic["provider_status_code"] == 400
    assert failure.value.outcome == "failed" and failure.value.retryable is False
    assert "PRIVATE" not in json.dumps(diagnostic)
    assert "secret.invalid" not in json.dumps(diagnostic)


@pytest.mark.parametrize("kind", ALIASES)
@pytest.mark.parametrize("role", ["answer", "answer_format_retry"])
@pytest.mark.parametrize("invalid", ["missing", "total", "public", "thinking", "policy"])
def test_typed_answer_budget_drift_is_rejected_before_transport(
    kind: str,
    role: str,
    invalid: str,
) -> None:
    config = registry().get(ALIASES[kind]).provider_config
    assert config is not None
    request = prepare_provider_answer_request(
        request_for(
            kind,
            provider_max_output_tokens=None,
            metadata={"model_role": role, "answer_contract_version": TYPED_ANSWER_CONTRACT_VERSION},
        ),
        config,
    )
    if invalid == "missing":
        request.provider_max_output_tokens = None
    elif invalid == "total":
        request.provider_max_output_tokens = 512 if kind == "openai" else 4096
    elif invalid == "public":
        request = request.model_copy(update={"max_output_tokens": 1024})
    elif invalid == "thinking":
        request = request.model_copy(update={"thinking_mode": "provider_default"})
    else:
        request.metadata.pop("provider_answer_policy")
    calls: list[httpx.Request] = []
    with pytest.raises(ValueError, match="typed answer"):
        asyncio.run(invoke(kind, request=request, captured=calls))
    assert not calls


@pytest.mark.parametrize("kind", ALIASES)
@pytest.mark.parametrize("contract", [None, "strict-answer-contract-v2"])
def test_non_v3_answer_preserves_existing_output_policy(kind: str, contract: str | None) -> None:
    config = registry().get(ALIASES[kind]).provider_config
    assert config is not None
    request = request_for(
        kind, metadata={"model_role": "answer", "answer_contract_version": contract}
    )
    assert prepare_provider_answer_request(request, config) is request
    assert provider_output_budget(request, config.max_output_tokens, config=config) == 1536
