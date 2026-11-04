"""Fail-closed production dispatch, bound by the factory to a measured replica pool."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
from copy import deepcopy
from io import BytesIO
from typing import Any, Literal

from PIL import Image
from pydantic import Field

from pmmc.compiler.ids import canonical_content_hash
from pmmc.models.registry import ModelRegistryEntry
from pmmc.runtime.manager import ManagedRuntimeError
from pmmc.runtime.step8_capacity import (
    CapacityImageWorkload,
    CapacityPool,
    PositiveCount,
    Step8PoolCapacityReceipt,
    capacity_image_workload,
    capacity_processor_kwargs,
    parse_image_workload,
    validate_capacity_receipt,
)
from pmmc.runtime.step8_capacity_native import NativeCapacityImageWorkload, layout_covers
from pmmc.schemas import BaseSchema, ManagedBackendKind, ManagedServiceSpec

_TEMPLATE = {"enable_thinking": False}
_TRANSPORT_FIELDS = {
    "model", "messages", "max_tokens", "temperature", "top_p", "seed", "response_format",
    "extra_body", "extra_headers", "timeout",
}


class CapacityTokenization(BaseSchema):
    """Trusted adapter result, not fields accepted from GenerationRequest metadata.

    Adapters must report the settings actually used. A scalar cached count or an
    operator assertion cannot establish processor/template conformance.
    """

    kind: Literal["local_exact", "service_exact"]
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_registry_entry_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    processor_kwargs: dict[str, Any]
    chat_template_kwargs: dict[str, Any]
    prompt_tokens: PositiveCount
    image_tokens: list[PositiveCount] = Field(default_factory=list)


ExactTokenizer = Callable[[dict[str, Any]], Awaitable[CapacityTokenization]]


def preflight_capacity_request_contract(
    source: ManagedServiceSpec, entry: ModelRegistryEntry,
) -> dict[str, Any]:
    """Offline config check, suitable before preparation, credentials or GPU startup."""
    if (
        source.metadata.get("requires_capacity_receipt") is not True
        or source.model_alias != entry.alias
        or source.backend is not ManagedBackendKind.VLLM_CHAT
        or entry.provider_config is None
        or entry.provider_config.kind != "qwen_vllm"
        or not source.max_model_len
        or not entry.context_window
        or source.max_model_len > min(entry.context_window, 32768)
    ):
        raise ManagedRuntimeError("capacity_generation_contract_invalid")
    try:
        bounds = capacity_image_workload(source, multimodal=True)
        processor = capacity_processor_kwargs(source)
    except ValueError:
        raise ManagedRuntimeError("capacity_image_configuration_required") from None
    assert bounds is not None
    if not entry.processor_name or any(
        not revision or revision in {"pinned-on-server", "main", "master", "latest", "unknown"}
        for revision in (entry.model_revision, entry.tokenizer_revision, entry.processor_revision)
    ):
        raise ManagedRuntimeError("capacity_exact_processor_identity_required")
    return {
        "model": entry.provider_model,
        "actual_model_id": entry.expected_actual_model_id,
        "model_registry_entry_hash": entry.resolved_hash,
        "counter_identity": {
            "alias": entry.alias, "tokenizer_name": entry.tokenizer_name,
            "tokenizer_revision": entry.tokenizer_revision,
            "processor_name": entry.processor_name, "processor_revision": entry.processor_revision,
        },
        "context_tokens": source.max_model_len,
        "max_output_tokens": entry.provider_config.max_output_tokens,
        "image_workload": bounds.model_dump(),
        "processor_kwargs": processor,
        "chat_template_kwargs": dict(_TEMPLATE),
    }


def _validate_messages(
    messages: object, bounds: CapacityImageWorkload | NativeCapacityImageWorkload,
) -> None:
    if not isinstance(messages, list) or not messages:
        raise ManagedRuntimeError("capacity_generation_messages_invalid")
    count = 0
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in {"user", "assistant", "system"}
            or not isinstance(message["content"], list)
        ):
            raise ManagedRuntimeError("capacity_generation_messages_invalid")
        for part in message["content"]:
            if not isinstance(part, dict):
                raise ManagedRuntimeError("capacity_generation_content_invalid")
            if part.get("type") == "text":
                if set(part) != {"type", "text"} or not isinstance(part["text"], str):
                    raise ManagedRuntimeError("capacity_generation_content_invalid")
                continue
            image = part.get("image_url")
            if (
                set(part) != {"type", "image_url"}
                or part["type"] != "image_url"
                or message["role"] != "user"
                or not isinstance(image, dict)
                or set(image) - {"url", "detail"}
                or image.get("detail", "auto") != "auto"
            ):
                raise ManagedRuntimeError("capacity_generation_image_overrides_forbidden")
            count += 1
            if count > bounds.images_per_sequence:
                raise ManagedRuntimeError("capacity_generation_image_count_exceeded")
            url = image.get("url")
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise ManagedRuntimeError("capacity_generation_inline_images_required")
            try:
                header, data = url.split(",", 1)
                raw = base64.b64decode(data, validate=True)
                with Image.open(BytesIO(raw)) as decoded:
                    orientation = decoded.getexif().get(274, 1)
                    if (
                        decoded.format not in {"PNG", "JPEG", "WEBP", "GIF"}
                        or header != f"data:{Image.MIME[decoded.format]};base64"
                        or getattr(decoded, "n_frames", 1) != 1
                        # EXIF 0 means unspecified orientation; both counters and
                        # vLLM receive the original pixels, with no auto-rotation.
                        or type(orientation) is not int or orientation not in {0, 1}
                        or decoded.width > bounds.image_width
                        or decoded.height > bounds.image_height
                    ):
                        raise ValueError("unsupported image or dimensions")
                    decoded.load()
            except (ValueError, OSError, Image.DecompressionBombError):
                raise ManagedRuntimeError("capacity_generation_image_bounds_invalid") from None


class CapacityRequestGuard:
    """Dispatch the checked wire payload, never a separately reconstructed request."""

    def __init__(
        self, pool: CapacityPool, receipt: Step8PoolCapacityReceipt, entry: ModelRegistryEntry,
    ) -> None:
        receipt = Step8PoolCapacityReceipt.model_validate(receipt.model_dump())
        validate_capacity_receipt(pool, receipt)
        if (
            receipt.workload_kind != "multimodal"
            or receipt.model_alias != entry.alias
            or receipt.model_registry_entry_hash != entry.resolved_hash
        ):
            raise ManagedRuntimeError("capacity_generation_receipt_mismatch")
        self._contract = preflight_capacity_request_contract(pool.services[0], entry)

    def validate_entry(self, entry: ModelRegistryEntry) -> None:
        if entry.resolved_hash != self._contract["model_registry_entry_hash"]:
            raise ManagedRuntimeError("capacity_generation_registry_mismatch")

    async def dispatch(
        self, transport: dict[str, Any], *, local_tokenize: ExactTokenizer,
        service_tokenize: ExactTokenizer, generate: Callable[..., Awaitable[Any]],
    ) -> Any:
        contract = self._contract
        wire = deepcopy(transport)
        if set(wire) - _TRANSPORT_FIELDS or wire.get("model") != contract["model"]:
            raise ManagedRuntimeError("capacity_generation_transport_invalid")
        output = wire.get("max_tokens")
        if type(output) is not int or not 0 < output <= contract["max_output_tokens"]:
            raise ManagedRuntimeError("capacity_generation_output_budget_invalid")
        extra = wire.get("extra_body")
        if (
            not isinstance(extra, dict)
            or set(extra) - {"chat_template_kwargs", "mm_processor_kwargs"}
            or canonical_content_hash(extra.get("chat_template_kwargs"))
            != canonical_content_hash(contract["chat_template_kwargs"])
            or canonical_content_hash(
                extra.get("mm_processor_kwargs", contract["processor_kwargs"]),
            )
            != canonical_content_hash(contract["processor_kwargs"])
        ):
            raise ManagedRuntimeError("capacity_generation_processor_override_forbidden")
        await asyncio.to_thread(
            _validate_messages, wire.get("messages"),
            parse_image_workload(contract["image_workload"]),
        )
        # Make inherited, already-frozen server settings explicit without changing them.
        extra["mm_processor_kwargs"] = deepcopy(contract["processor_kwargs"])
        token_input = {
            "model": wire["model"], "messages": wire["messages"],
            "add_generation_prompt": True,
            "chat_template_kwargs": extra["chat_template_kwargs"],
            "mm_processor_kwargs": extra["mm_processor_kwargs"],
        }
        expected = {
            "input_sha256": canonical_content_hash(token_input),
            "model_registry_entry_hash": contract["model_registry_entry_hash"],
            "processor_kwargs": contract["processor_kwargs"],
            "chat_template_kwargs": contract["chat_template_kwargs"],
        }
        counts = []
        tokenizers = (("local_exact", local_tokenize), ("service_exact", service_tokenize))
        for kind, tokenize in tokenizers:
            try:
                result = await tokenize(deepcopy(token_input))
                if not isinstance(result, CapacityTokenization):
                    raise ValueError("exact tokenizer adapter result required")
                result = CapacityTokenization.model_validate(result.model_dump())
            except Exception:
                raise ManagedRuntimeError("capacity_exact_tokenization_unavailable") from None
            if result.kind != kind or canonical_content_hash(
                result.model_dump(include=set(expected)),
            ) != canonical_content_hash(expected):
                raise ManagedRuntimeError("capacity_exact_processor_contract_mismatch")
            counts.append(result.prompt_tokens)
            bounds = parse_image_workload(contract["image_workload"])
            if kind == "local_exact" and isinstance(bounds, NativeCapacityImageWorkload):
                image_count = sum(part.get("type") == "image_url"
                                  for message in wire["messages"] for part in message["content"])
                if len(result.image_tokens) != image_count or not any(
                    layout_covers(layout, result.image_tokens) for layout in bounds.layouts
                ):
                    raise ManagedRuntimeError("capacity_generation_layout_not_measured")
        if counts[0] != counts[1]:
            raise ManagedRuntimeError("capacity_exact_tokenization_disagrees")
        if counts[0] + output > contract["context_tokens"]:
            raise ManagedRuntimeError("capacity_generation_context_exceeded")
        response = await generate(**wire)
        try:
            raw = response if isinstance(response, dict) else response.model_dump(mode="json")
        except Exception:
            raise ManagedRuntimeError("capacity_generation_usage_unverifiable") from None
        usage = raw.get("usage") if isinstance(raw, dict) else None
        if (
            not isinstance(raw, dict)
            or raw.get("model") != contract["actual_model_id"]
            or not isinstance(usage, dict)
            or any(type(usage.get(key)) is not int for key in (
                "prompt_tokens", "completion_tokens", "total_tokens",
            ))
            or usage["prompt_tokens"] != counts[0]
            or not 0 <= usage["completion_tokens"] <= output
            or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
        ):
            # This is a post-dispatch conformance failure, never permission to retry.
            raise ManagedRuntimeError("capacity_generation_tokenization_drift")
        return response
