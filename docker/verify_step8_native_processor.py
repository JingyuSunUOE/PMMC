"""Offline CPU smoke of the installed processor, without tokenizer or weights."""

from __future__ import annotations

import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any


def verify() -> dict[str, Any]:
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor

    from pmmc.models.native_qwen_processor import verify_loaded_qwen_image_processor

    # This fixture exercises serialized Fast-name resolution in the real library.
    # Actual benchmark snapshots must still pass their own immutable hash checks.
    native = {
        "image_processor_type": "Qwen2VLImageProcessorFast",
        "processor_class": "Qwen3VLProcessor", "patch_size": 16,
        "temporal_patch_size": 2, "merge_size": 2,
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
        "do_resize": True, "resample": 3, "do_rescale": True,
        "rescale_factor": 1 / 255, "do_normalize": True, "do_convert_rgb": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    }
    with tempfile.TemporaryDirectory(prefix="step8-processor-", dir="/tmp") as directory:
        root = Path(directory)
        (root / "preprocessor_config.json").write_text(json.dumps(native))
        print("STEP8_NATIVE_PROCESSOR phase=load state=start", flush=True)
        processor = AutoImageProcessor.from_pretrained(
            root, local_files_only=True, trust_remote_code=False, cache_dir=root / "cache",
        )
        print("STEP8_NATIVE_PROCESSOR phase=identity state=start", flush=True)
        identity = verify_loaded_qwen_image_processor(processor, native)
        images = [Image.new("RGB", (32, 64)), Image.new("RGB", (96, 96))]
        try:
            print("STEP8_NATIVE_PROCESSOR phase=cpu_grids state=start", flush=True)
            with torch.device("cpu"):
                result = processor(images=images, return_tensors="pt", device="cpu")
        finally:
            for image in images:
                image.close()
        grids = result["image_grid_thw"]
        pixels = result["pixel_values"]
        if (grids.device.type != "cpu" or pixels.device.type != "cpu"
                or list(grids.shape) != [2, 3] or not bool((grids > 0).all())
                or pixels.ndim != 2 or pixels.shape[1] != 3 * 2 * 16 * 16
                or int(grids.prod(dim=1).sum()) != pixels.shape[0]):
            raise ValueError("native_processor_cpu_grid_contract_failed")
    return {"success": True, "protocol": "step8-native-processor-smoke-v1",
            "scope": "image_processor_only_not_tokenizer_or_capacity",
            "transformers": version("transformers"), "identity": identity,
            "grid_count": 2, "gpu_operations": 0, "provider_calls": 0,
            "model_weights_loaded": False}


if __name__ == "__main__":
    try:
        report = verify()
    except Exception as error:
        category = next((t.__name__ for t in (
            OSError, ImportError, AttributeError, TypeError, ValueError, RuntimeError,
        ) if isinstance(error, t)), "Exception")
        print(json.dumps({"success": False, "code": "native_processor_smoke_failed",
                          "exception_type": category}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps(report, sort_keys=True), flush=True)
