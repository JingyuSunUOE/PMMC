"""Synthetic, bounded image requests for isolated capacity measurement only."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
from copy import deepcopy
from typing import Any

import httpx
from PIL import Image

from pmmc.runtime.errors import ManagedRuntimeError
from pmmc.runtime.step8_capacity import CapacityImageWorkload
from pmmc.runtime.step8_capacity_native import NativeImageLayout


def synthetic_native_messages(layout: NativeImageLayout, count: int) -> list:
    """Keep audited shapes; use distinct, compressible pixels, never private images."""
    result = []
    for ordinal in range(count):
        nonce = hashlib.sha256(f"{ordinal}:{layout.identity}".encode()).hexdigest()
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"{nonce} Capacity sequence {ordinal}."},
        ]
        for index, shape in enumerate(layout.images):
            digest = hashlib.sha256(f"{layout.identity}:{ordinal}:{index}".encode()).digest()
            with Image.new("RGB", (shape.width, shape.height), tuple(digest[:3])) as image:
                for offset in range(min(10, shape.width * shape.height)):
                    image.putpixel((offset % shape.width, offset // shape.width),
                                   tuple(digest[offset * 3:offset * 3 + 3]))
                # Distinct pixels without megabytes of incompressible PNG data.
                stream = io.BytesIO()
                image.save(stream, format="PNG")
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(),
            }})
        result.append([{"role": "user", "content": content}])
    return result


def validate_native_messages(layout: NativeImageLayout, prompts: list, *, count: int) -> None:
    if len(prompts) != count:
        raise ManagedRuntimeError("capacity_native_sequence_count_invalid")
    hashes = set()
    for messages in prompts:
        if (len(messages) != 1 or messages[0].get("role") != "user"
                or len(messages[0].get("content", [])) != len(layout.images) + 1):
            raise ManagedRuntimeError("capacity_native_layout_invalid")
        for part, shape in zip(messages[0]["content"][1:], layout.images, strict=True):
            try:
                prefix, value = part["image_url"]["url"].split(",", 1)
                if part["type"] != "image_url" or prefix != "data:image/png;base64":
                    raise ValueError
                payload = base64.b64decode(value, validate=True)
                with Image.open(io.BytesIO(payload)) as image:
                    if image.format != "PNG" or image.size != (shape.width, shape.height):
                        raise ValueError
                    image.load()
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest in hashes:
                    raise ValueError
                hashes.add(digest)
            except (ValueError, OSError, KeyError, TypeError):
                raise ManagedRuntimeError("capacity_native_pixels_invalid") from None


def synthetic_image_messages(
    bounds: CapacityImageWorkload, count: int
) -> list[list[dict[str, Any]]]:
    """Distinct pixels and prefixes prevent cross-request image/prefix-cache sharing."""
    messages = []
    for ordinal in range(count):
        content: list[dict[str, Any]] = [{
            "type": "text", "text": f"Capacity measurement request {ordinal}."
        }]
        for image_index in range(bounds.images_per_sequence):
            rng = random.Random((ordinal + 1) * 1000 + image_index)
            pixels = rng.randbytes(bounds.image_width * bounds.image_height * 3)
            with Image.frombytes("RGB", (bounds.image_width, bounds.image_height), pixels) as image:
                stream = io.BytesIO()
                image.save(stream, format="PNG")
            encoded = base64.b64encode(stream.getvalue()).decode("ascii")
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + encoded
            }})
        messages.append([{"role": "user", "content": content}])
    return messages


def image_request_hash(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def validate_image_messages(
    bounds: CapacityImageWorkload, prompts: list, *, count: int
) -> None:
    """Check actual image bytes, not caller labels, before touching a service."""
    if len(prompts) != count:
        raise ManagedRuntimeError("capacity image workload must cover every sequence")
    image_hashes: set[str] = set()
    for messages in prompts:
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or not isinstance(messages[0], dict)
            or messages[0].get("role") != "user"
        ):
            raise ManagedRuntimeError("capacity image workload requires one user message")
        content = messages[0].get("content")
        if (
            not isinstance(content, list)
            or len(content) != bounds.images_per_sequence + 1
            or not isinstance(content[0], dict)
            or content[0].get("type") != "text"
            or not isinstance(content[0].get("text"), str)
            or not content[0]["text"]
        ):
            raise ManagedRuntimeError("capacity image workload has incorrect image count or text")
        for part in content[1:]:
            url = part.get("image_url") if isinstance(part, dict) else None
            url = url.get("url") if isinstance(url, dict) else None
            prefix = "data:image/png;base64,"
            if (
                not isinstance(part, dict)
                or part.get("type") != "image_url"
                or not isinstance(url, str)
                or not url.startswith(prefix)
            ):
                raise ManagedRuntimeError("capacity images require inline PNG data, not URLs")
            try:
                raw = base64.b64decode(url[len(prefix):], validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    if (
                        image.format != "PNG"
                        or image.mode != "RGB"
                        or image.size != (bounds.image_width, bounds.image_height)
                    ):
                        raise ValueError("image dimensions or format differ from frozen bounds")
                    image.load()
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
            except (ValueError, OSError) as exc:
                raise ManagedRuntimeError(
                    "capacity image pixels do not match frozen bounds"
                ) from exc
            if digest in image_hashes:
                raise ManagedRuntimeError("capacity image pixels must be distinct across the pool")
            image_hashes.add(digest)


async def fit_image_prompt(
    client: httpx.AsyncClient, *, endpoint: str, api_key: str, model: str,
    messages: list[dict[str, Any]], target_tokens: int,
    processor_kwargs: dict[str, object] | None = None,
) -> list[dict[str, Any]]:
    """Fit full context using the actual service tokenizer including image expansion.

    Usage from the subsequent generation is checked independently, so a service
    that omits image/template tokens here can never produce a passing receipt.
    """
    messages = deepcopy(messages)
    prefix = messages[0]["content"][0]["text"]
    padding = 0
    seen = set()
    for _ in range(12):
        if padding in seen or padding < 0:
            break
        seen.add(padding)
        messages[0]["content"][0]["text"] = prefix + " a" * padding
        try:
            response = await client.post(
                endpoint + "/tokenize",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages, "add_generation_prompt": True,
                      "chat_template_kwargs": {"enable_thinking": False},
                      **({"mm_processor_kwargs": processor_kwargs}
                         if processor_kwargs is not None else {})},
            )
            response.raise_for_status()
        except httpx.HTTPError:
            raise ManagedRuntimeError("capacity_tokenizer_http_request_failed") from None
        payload = response.json()
        count = payload.get("count") if isinstance(payload, dict) else None
        if type(count) is not int or count <= 0:
            raise ManagedRuntimeError("capacity multimodal tokenizer did not return a token count")
        if count == target_tokens:
            return messages
        if count > target_tokens and padding == 0:
            raise ManagedRuntimeError("capacity image workload exceeds the configured context")
        padding += target_tokens - count
    raise ManagedRuntimeError("capacity multimodal prompt could not reach the exact token bound")
