"""Native processor capacity layouts, separate from source image safety bounds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pmmc.schemas import BaseSchema

Hash = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Positive = Annotated[int, Field(strict=True, gt=0)]


class NativeImageShape(BaseSchema):
    width: Annotated[int, Field(strict=True, ge=1, le=16384)]
    height: Annotated[int, Field(strict=True, ge=1, le=16384)]
    image_tokens: Positive


class NativeImageLayout(BaseSchema):
    images: list[NativeImageShape] = Field(min_length=1, max_length=512)

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class NativeCapacityImageWorkload(BaseSchema):
    protocol: Literal["native-image-layouts-v1"]
    images_per_sequence: Annotated[int, Field(strict=True, ge=1, le=512)]
    # These constrain original transport bytes, not the model's processed image size.
    image_width: Annotated[int, Field(strict=True, ge=1, le=16384)]
    image_height: Annotated[int, Field(strict=True, ge=1, le=16384)]
    processor_config_sha256: Hash
    context_audit_hash: Hash
    layouts: list[NativeImageLayout] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def valid_layouts(self) -> NativeCapacityImageWorkload:
        if len({x.identity for x in self.layouts}) != len(self.layouts):
            raise ValueError("capacity_native_duplicate_layout")
        if max(len(x.images) for x in self.layouts) != self.images_per_sequence:
            raise ValueError("capacity_native_count_not_measured")
        if any(i.width > self.image_width or i.height > self.image_height
               for layout in self.layouts for i in layout.images):
            raise ValueError("capacity_native_source_bounds_exceeded")
        return self


def verify_native_processor(snapshot: Path, workload: NativeCapacityImageWorkload) -> None:
    """Check the pinned small config, never download or hash model weights."""
    path = snapshot / "preprocessor_config.json"
    if path.stat().st_size > 1_048_576:
        raise ValueError("capacity_native_processor_config_invalid")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != workload.processor_config_sha256:
        raise ValueError("capacity_native_processor_identity_mismatch")
    config = json.loads(payload)
    if (config.get("processor_class") != "Qwen3VLProcessor"
            or config.get("image_processor_type") != "Qwen2VLImageProcessorFast"
            or config.get("patch_size") != 16 or config.get("merge_size") != 2
            or config.get("temporal_patch_size") != 2
            or config.get("size") != {"shortest_edge": 65536, "longest_edge": 16777216}):
        raise ValueError("capacity_native_processor_contract_unsupported")


def layout_covers(layout: NativeImageLayout, tokens: list[int]) -> bool:
    """Empirical encoder-load envelope, not a proof of arbitrary GPU memory use."""
    observed = sorted((image.image_tokens for image in layout.images), reverse=True)
    requested = sorted(tokens, reverse=True)
    return len(requested) <= len(observed) and all(
        actual <= measured for actual, measured in zip(requested, observed, strict=False)
    )
