"""Validate loaded Qwen image implementations without changing native settings."""

from __future__ import annotations

from typing import Any

_IMPLEMENTATIONS = {
    ("transformers.models.qwen2_vl.image_processing_qwen2_vl_fast",
     "Qwen2VLImageProcessorFast"): "legacy_fast",
    ("transformers.models.qwen2_vl.image_processing_qwen2_vl",
     "Qwen2VLImageProcessor"): "torchvision_backend",
}
_GEOMETRY = ("patch_size", "temporal_patch_size", "merge_size", "size")
_OPTIONAL_SETTINGS = (
    "do_resize", "resample", "do_rescale", "rescale_factor", "do_normalize",
    "image_mean", "image_std", "do_convert_rgb",
)


def verify_loaded_qwen_image_processor(
    processor: Any, native_config: dict[str, Any],
) -> dict[str, Any]:
    """Accept the v5 Fast-name migration, never a PIL/unknown backend or size drift.

    The caller verifies the immutable snapshot and config hash before loading.
    Native configuration keeps its original serialized class name; only the
    installed library's resolved implementation identity is normalized here.
    """
    cls = type(processor)
    implementation = _IMPLEMENTATIONS.get((cls.__module__, cls.__name__))
    if (native_config.get("image_processor_type") != "Qwen2VLImageProcessorFast"
            or implementation is None):
        raise ValueError("unexpected_native_image_processor")
    backend = getattr(processor, "backend", None)
    if backend is None and implementation == "legacy_fast":
        backend = "torchvision" if getattr(processor, "is_fast", False) is True else None
    if backend != "torchvision":
        raise ValueError("unexpected_native_image_backend")
    for key in (*_GEOMETRY, *_OPTIONAL_SETTINGS):
        if key not in native_config:
            if key in _GEOMETRY:
                raise ValueError("native_processor_geometry_missing")
            continue
        value = getattr(processor, key, None)
        # v5 stores size as SizeDict and normalization vectors as tuples.
        # Compare representations without mutating the installed processor.
        if key == "size" and value is not None:
            value = dict(value)
        elif key in {"image_mean", "image_std"} and isinstance(value, tuple):
            value = list(value)
        if value != native_config[key]:
            raise ValueError("loaded_native_processor_settings_mismatch")
    return {
        "configured_class": native_config["image_processor_type"],
        "loaded_class": cls.__name__, "loaded_module": cls.__module__,
        "backend": backend, "implementation": implementation,
        "native_settings_verified": True,
        "geometry": {key: native_config[key] for key in _GEOMETRY},
    }
