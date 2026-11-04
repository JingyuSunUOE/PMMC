"""Versioned vLLM allocation policy; the device-total guard is never relaxed."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from pmmc.schemas import BaseSchema, GpuAllocation, GpuDeviceInfo

# This is additional headroom outside vLLM's profiled weights/activations/cache.
# The v1 1-GiB reserve passed startup but failed real multimodal prefill.
VLLM_RUNTIME_HEADROOM_BYTES = 4 * 1024**3
VLLM_MEMORY_BUDGET_PROTOCOL = "vllm-device-budget-v2"
_POLICY_HEADROOM = {"vllm-device-budget-v1": 1024**3,
                    VLLM_MEMORY_BUDGET_PROTOCOL: VLLM_RUNTIME_HEADROOM_BYTES}
_SCALE = 1_000_000


class VllmMemoryBudget(BaseSchema):
    protocol: Literal["vllm-device-budget-v1", "vllm-device-budget-v2"] = (
        "vllm-device-budget-v2"
    )
    visible_indices: list[int]
    total_bytes: list[int]
    baseline_bytes: list[int]
    device_budget_bytes: int = Field(gt=0)
    runtime_headroom_bytes: Literal[1073741824, 4294967296] = 4294967296
    requested_utilization: float = Field(gt=0, le=1, allow_inf_nan=False)
    effective_utilization: float = Field(gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_policy(self) -> VllmMemoryBudget:
        if self.runtime_headroom_bytes != _POLICY_HEADROOM[self.protocol]:
            raise ValueError("vllm_memory_budget_headroom_version_mismatch")
        if (
            not self.visible_indices
            or len(set(self.visible_indices)) != len(self.visible_indices)
            or any(i < 0 for i in self.visible_indices)
            or len(self.total_bytes) != len(self.visible_indices)
            or len(self.baseline_bytes) != len(self.visible_indices)
            or any(t <= 0 or not 0 <= b <= t for t, b in
                   zip(self.total_bytes, self.baseline_bytes, strict=True))
        ):
            raise ValueError("vllm_memory_budget_inventory_invalid")
        expected = _effective_utilization(
            self.total_bytes, self.baseline_bytes, self.device_budget_bytes,
            self.requested_utilization, self.runtime_headroom_bytes,
        )
        if self.effective_utilization != expected:
            raise ValueError("vllm_memory_budget_policy_mismatch")
        return self


def _effective_utilization(
    totals: list[int], baselines: list[int], budget: int, requested: float,
    headroom: int,
) -> float:
    # One common TP ratio must fit even the largest / most occupied device.
    ppm = min([math.floor(requested * _SCALE), *[
        (budget - baseline - headroom) * _SCALE // total
        for total, baseline in zip(totals, baselines, strict=True)
    ]]) - 1  # Round down, including floating-point CLI serialization error.
    if ppm <= 0:
        raise ValueError("vllm_memory_budget_exhausted")
    return ppm / _SCALE


def derive_vllm_memory_budget(
    devices: tuple[GpuDeviceInfo, ...], budget: int, requested: float,
) -> VllmMemoryBudget:
    totals = [d.total_memory_bytes for d in devices]
    baselines = [max(d.used_memory_bytes, d.total_memory_bytes - d.free_memory_bytes)
                 for d in devices]
    return VllmMemoryBudget(
        visible_indices=[d.visible_index for d in devices], total_bytes=totals,
        baseline_bytes=baselines, device_budget_bytes=budget,
        requested_utilization=requested,
        effective_utilization=_effective_utilization(
            totals, baselines, budget, requested, VLLM_RUNTIME_HEADROOM_BYTES,
        ),
    )


def allocation_vllm_memory_budget(allocation: GpuAllocation) -> VllmMemoryBudget:
    contract = VllmMemoryBudget.model_validate(allocation.metadata.get("vllm_memory_budget"))
    # Historical receipts remain readable, but cannot start a current service.
    if contract.protocol != VLLM_MEMORY_BUDGET_PROTOCOL:
        raise ValueError("vllm_memory_budget_recalibration_required")
    if (
        contract.visible_indices != allocation.visible_indices
        or contract.device_budget_bytes != allocation.memory_budget_bytes_per_gpu
        or contract.requested_utilization != allocation.estimate.gpu_memory_utilization
        or any(int(t * contract.effective_utilization) < allocation.estimate.required_bytes_per_gpu
               for t in contract.total_bytes)
    ):
        raise ValueError("vllm_memory_budget_allocation_mismatch")
    return contract
