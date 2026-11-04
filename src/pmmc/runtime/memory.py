"""Static model-weight inspection and conservative GPU memory estimation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from pmmc.runtime.config import RuntimeProfile
from pmmc.schemas import ManagedServiceSpec, ModelLockEntry, ModelMemoryEstimate

GIB = 1024**3


@dataclass(frozen=True)
class SnapshotWeightInfo:
    weight_bytes: int
    tensor_count: int
    shard_count: int
    architectures: tuple[str, ...]
    torch_dtype: str | None
    quantization: str | None


def _safetensors_payload_bytes(path: Path) -> tuple[int, int]:
    """Read safetensors headers directly, avoiding Torch and CUDA imports."""

    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header: {path.name}")
        header_length = int.from_bytes(raw_length, "little")
        if header_length <= 0 or header_length > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length: {path.name}")
        try:
            header = json.loads(stream.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid safetensors metadata: {path.name}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors metadata root: {path.name}")
    total = 0
    count = 0
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise ValueError(f"invalid tensor metadata in {path.name}")
        offsets = tensor.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"missing tensor offsets in {path.name}")
        start, end = (int(offsets[0]), int(offsets[1]))
        if start < 0 or end < start:
            raise ValueError(f"invalid tensor offsets in {path.name}")
        payload_size = path.stat().st_size - 8 - header_length
        if end > payload_size:
            raise ValueError(f"tensor offsets exceed payload in {path.name}")
        total += end - start
        count += 1
    return total, count


def _weight_files(snapshot: Path) -> list[Path]:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        value = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = value.get("weight_map") if isinstance(value, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json has no weight_map")
        names = sorted(set(str(item) for item in weight_map.values()))
        relative_paths = [Path(name) for name in names]
        if any(path.is_absolute() or ".." in path.parts for path in relative_paths):
            raise ValueError("safetensors index contains an unsafe shard path")
        files = [snapshot / path for path in relative_paths]
    else:
        files = sorted(snapshot.glob("*.safetensors"))
    for path in files:
        if not path.is_file():
            raise ValueError("safetensors index contains a missing shard path")
    if not files:
        raise ValueError("managed runtime requires safetensors model weights")
    return files


def inspect_snapshot_weights(snapshot: Path) -> SnapshotWeightInfo:
    files = _weight_files(snapshot)
    weight_bytes = 0
    tensor_count = 0
    for path in files:
        size, count = _safetensors_payload_bytes(path)
        weight_bytes += size
        tensor_count += count
    config_path = snapshot / "config.json"
    config: dict[str, object] = {}
    if config_path.is_file():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            config = loaded
    architecture_value = config.get("architectures")
    architectures = (
        tuple(str(item) for item in architecture_value)
        if isinstance(architecture_value, list)
        else ()
    )
    quantization_config = config.get("quantization_config")
    quantization = None
    if isinstance(quantization_config, dict):
        quantization = str(
            quantization_config.get("quant_method")
            or quantization_config.get("quantization_method")
            or "configured"
        )
    return SnapshotWeightInfo(
        weight_bytes=weight_bytes,
        tensor_count=tensor_count,
        shard_count=len(files),
        architectures=architectures,
        torch_dtype=str(config["torch_dtype"]) if config.get("torch_dtype") else None,
        quantization=quantization,
    )


def estimate_service_memory(
    service: ManagedServiceSpec,
    lock_entry: ModelLockEntry,
    snapshot: Path,
    profile: RuntimeProfile,
    *,
    tensor_parallel_size: int,
    gpu_total_memory_bytes: int,
) -> ModelMemoryEstimate:
    if tensor_parallel_size not in service.tensor_parallel_candidates:
        raise ValueError("tensor parallel size is not allowed by the service profile")
    info = inspect_snapshot_weights(snapshot)
    if (
        service.quantization is not None
        and info.quantization is not None
        and service.quantization.lower() != info.quantization.lower()
    ):
        raise ValueError("runtime profile quantization does not match model config")
    sharded = math.ceil(info.weight_bytes / tensor_parallel_size)
    overhead = math.ceil(profile.service_overhead_gib * GIB)
    reserve = max(
        math.ceil(profile.gpu_reserve_min_gib * GIB),
        math.ceil(gpu_total_memory_bytes * profile.gpu_reserve_fraction),
    )
    required = math.ceil(sharded * profile.weight_safety_factor) + overhead
    return ModelMemoryEstimate(
        model_alias=service.model_alias,
        model_commit=lock_entry.commit_sha,
        tensor_parallel_size=tensor_parallel_size,
        weight_bytes=info.weight_bytes,
        sharded_weight_bytes_per_gpu=sharded,
        weight_safety_factor=profile.weight_safety_factor,
        service_overhead_bytes=overhead,
        system_reserve_bytes=reserve,
        required_bytes_per_gpu=required,
        gpu_memory_utilization=min(
            service.gpu_memory_utilization,
            profile.gpu_memory_utilization_max,
        ),
        max_model_len=service.max_model_len,
        max_num_seqs=service.max_num_seqs,
        metadata={
            "tensor_count": info.tensor_count,
            "shard_count": info.shard_count,
            "architectures": list(info.architectures),
            "torch_dtype": info.torch_dtype,
            "quantization": info.quantization or service.quantization,
        },
    )
