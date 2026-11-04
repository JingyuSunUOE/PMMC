"""Conservative GPU-allocation accounting, independent of PID namespaces.

This measures total device occupancy, not ownership or per-process memory.
Baseline bytes are never subtracted from the budget or the measured peak.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from pmmc.runtime.errors import ManagedRuntimeError
from pmmc.schemas import BaseSchema, GpuAllocation, GpuDeviceInfo

if TYPE_CHECKING:
    from pmmc.runtime.hardware import NvmlHardwareProvider

Count = Annotated[int, Field(strict=True, ge=0)]


class DeviceMemorySample(BaseSchema):
    device_id: str = Field(pattern=r"^gpu-[a-zA-Z0-9-]{1,64}$")
    visible_index: Count
    total_bytes: Annotated[int, Field(strict=True, gt=0)]
    used_bytes: Count
    compute_process_count: Count
    graphics_process_count: Count
    sampled_at: datetime


class DeviceLifecycleEvidence(BaseSchema):
    protocol: Literal["allocated-device-lifecycle-v1"] = "allocated-device-lifecycle-v1"
    baseline_subtracted: Literal[False] = False
    release_consecutive_samples: Annotated[int, Field(strict=True, ge=2)]
    before: list[DeviceMemorySample]
    after: list[DeviceMemorySample]
    release_observations: list[list[DeviceMemorySample]]
    peak_bytes_by_device: dict[str, Count]


class DeviceMemoryBudgetExceeded(ManagedRuntimeError):
    def __init__(self, samples: list[DeviceMemorySample], diagnostic: dict[str, object]):
        super().__init__("gpu_device_total_memory_over_budget")
        self.samples = samples
        self.diagnostic = diagnostic


class AllocatedDeviceAccounting:
    def __init__(
        self,
        provider: NvmlHardwareProvider,
        allocations: list[GpuAllocation],
        inventory: list[GpuDeviceInfo],
        *,
        on_budget_failure: Callable[[dict[str, object]], None] | None = None,
    ):
        self.provider = provider
        self.allocations = allocations
        self.expected = {d.visible_index: d for d in inventory}
        self.budgets = {
            i: a.memory_budget_bytes_per_gpu for a in allocations for i in a.visible_indices
        }
        assigned = [
            (i, d)
            for a in allocations
            for i, d in zip(a.visible_indices, a.device_ids, strict=True)
        ]
        if (
            not assigned
            or len(assigned) != len(self.budgets)
            or any(i not in self.expected or self.expected[i].device_id != d for i, d in assigned)
            or len({d for _, d in assigned}) != len(assigned)
        ):
            raise ManagedRuntimeError("gpu_device_allocation_identity_invalid")
        self.before: list[DeviceMemorySample] = []
        self.peaks: dict[str, int] = {}
        self.release_evidence: DeviceLifecycleEvidence | None = None
        self.first_budget_failure: dict[str, object] | None = None
        self.on_budget_failure = on_budget_failure
        self._lock = threading.Lock()

    def _read(self) -> list[DeviceMemorySample]:
        values = self.provider.device_memory_samples(set(self.budgets))
        if len(values) != len(self.budgets) or {v.visible_index for v in values} != set(
            self.budgets
        ):
            raise ManagedRuntimeError("gpu_device_measurement_incomplete")
        for v in values:
            expected = self.expected[v.visible_index]
            if v.device_id != expected.device_id or v.total_bytes != expected.total_memory_bytes:
                raise ManagedRuntimeError("gpu_device_identity_changed")
            if v.used_bytes > v.total_bytes or v.sampled_at.tzinfo is None:
                raise ManagedRuntimeError("gpu_device_measurement_invalid")
        return sorted(values, key=lambda v: v.visible_index)

    @staticmethod
    def _idle(values: list[DeviceMemorySample]) -> bool:
        return all(v.compute_process_count == v.graphics_process_count == 0 for v in values)

    def begin(self) -> list[DeviceMemorySample]:
        with self._lock:
            if self.before:
                raise ManagedRuntimeError("gpu_device_accounting_already_started")
            values = self._read()
            if not self._idle(values):
                raise ManagedRuntimeError("gpu_device_not_idle_before_start")
            self._record(values)
            self.before = values
            return values

    def _record(self, values: list[DeviceMemorySample]) -> None:
        # Retain the complete measured batch before enforcing the unchanged cap.
        for v in values:
            self.peaks[v.device_id] = max(self.peaks.get(v.device_id, 0), v.used_bytes)
        violations = [
            {"visible_index": v.visible_index, "used_bytes": v.used_bytes,
             "budget_bytes": self.budgets[v.visible_index],
             "excess_bytes": v.used_bytes - self.budgets[v.visible_index],
             "baseline_bytes": next((b.used_bytes for b in self.before
                                     if b.visible_index == v.visible_index), v.used_bytes),
             "sampled_at": v.sampled_at.isoformat()}
            for v in values if v.used_bytes > self.budgets[v.visible_index]
        ]
        if violations:
            diagnostic: dict[str, object] = {
                "protocol": "gpu-device-budget-failure-v1",
                "code": "gpu_device_total_memory_over_budget",
                "phase": "sampling" if self.before else "baseline",
                "baseline_subtracted": False, "violations": violations,
            }
            if self.first_budget_failure is None:
                self.first_budget_failure = diagnostic
                try:
                    if self.on_budget_failure is not None:
                        self.on_budget_failure(diagnostic)
                except OSError:
                    diagnostic["artifact_write_failed"] = True
                print("GPU_DEVICE_BUDGET_FAILURE " + json.dumps(diagnostic, sort_keys=True),
                      flush=True)
            raise DeviceMemoryBudgetExceeded(values, diagnostic)

    def sample(self, *, require_active: bool = False) -> list[DeviceMemorySample]:
        with self._lock:
            if not self.before:
                raise ManagedRuntimeError("gpu_device_accounting_not_started")
            values = self._read()
            if require_active and any(
                v.compute_process_count == 0 or v.used_bytes == 0 for v in values
            ):
                raise ManagedRuntimeError("gpu_device_active_measurement_missing")
            self._record(values)
            return values

    def finish(self, timeout_seconds: float) -> DeviceLifecycleEvidence:
        if not self.before:
            raise ManagedRuntimeError("gpu_device_release_baseline_missing")
        baseline = {v.device_id: v.used_bytes for v in self.before}
        deadline = time.monotonic() + timeout_seconds
        consecutive = 0
        observations: list[list[DeviceMemorySample]] = []
        values = self.before
        while time.monotonic() < deadline:
            # Release can take time after an over-budget failure. Do not abort
            # its verification on the first still-occupied shutdown sample.
            try:
                values = self.sample()
            except DeviceMemoryBudgetExceeded as exc:
                values = exc.samples
            released = self._idle(values) and all(
                v.used_bytes <= baseline[v.device_id] for v in values
            )
            consecutive = consecutive + 1 if released else 0
            observations = [*observations, values] if released else []
            if consecutive >= 2:
                self.release_evidence = DeviceLifecycleEvidence(
                    before=self.before,
                    after=values,
                    peak_bytes_by_device=dict(self.peaks),
                    release_consecutive_samples=consecutive,
                    release_observations=observations,
                )
                return self.release_evidence
            time.sleep(0.25)
        occupied = sum(bool(v.compute_process_count or v.graphics_process_count) for v in values)
        print(
            "GPU_DEVICE_RELEASE state=failed "
            f"occupied_devices={occupied} "
            f"above_baseline_devices={sum(v.used_bytes > baseline[v.device_id] for v in values)}",
            flush=True,
        )
        raise ManagedRuntimeError("gpu_device_release_unverified")


def validate_device_lifecycle(
    evidence: DeviceLifecycleEvidence,
    allocations: list[GpuAllocation],
    measured_peaks: dict[str, int],
) -> None:
    evidence = DeviceLifecycleEvidence.model_validate(evidence.model_dump())
    if (
        len(evidence.release_observations) != evidence.release_consecutive_samples
        or evidence.release_observations[-1] != evidence.after
    ):
        raise ValueError("device release consecutive observations are incomplete")
    assigned = {
        d: (i, a.memory_budget_bytes_per_gpu)
        for a in allocations
        for i, d in zip(a.visible_indices, a.device_ids, strict=True)
    }
    before = {v.device_id: v for v in evidence.before}
    after = {v.device_id: v for v in evidence.after}
    if (
        len(before) != len(evidence.before)
        or len(after) != len(evidence.after)
        or set(before) != set(assigned)
        or set(after) != set(assigned)
        or set(evidence.peak_bytes_by_device) != set(assigned)
        or set(measured_peaks) != set(assigned)
    ):
        raise ValueError("device lifecycle coverage is incomplete")
    for device, (index, budget) in assigned.items():
        start, end = before[device], after[device]
        peak = evidence.peak_bytes_by_device[device]
        if (
            start.visible_index != end.visible_index
            or start.visible_index != index
            or start.total_bytes != end.total_bytes
            or not 0 <= end.used_bytes <= start.used_bytes <= peak <= budget
            or peak > start.total_bytes
            or measured_peaks[device] > peak
            or any(v.compute_process_count or v.graphics_process_count for v in (start, end))
            or start.sampled_at.tzinfo is None
            or end.sampled_at.tzinfo is None
            or not start.sampled_at <= end.sampled_at <= datetime.now(UTC)
        ):
            raise ValueError("device lifecycle idle/release/budget evidence is invalid")
    previous = before
    for observation in evidence.release_observations:
        current = {v.device_id: v for v in observation}
        if len(current) != len(observation) or set(current) != set(before):
            raise ValueError("device release observation coverage is incomplete")
        for device, sample in current.items():
            start = before[device]
            if (
                sample.visible_index != start.visible_index
                or sample.total_bytes != start.total_bytes
                or sample.compute_process_count
                or sample.graphics_process_count
                or sample.used_bytes > start.used_bytes
                or sample.sampled_at.tzinfo is None
                or not previous[device].sampled_at <= sample.sampled_at <= datetime.now(UTC)
            ):
                raise ValueError("device release observation is invalid")
        previous = current
