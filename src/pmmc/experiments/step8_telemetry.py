"""Content-free, best-effort resource samples; unavailable values stay null."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any


def _integer(path: Path) -> int | None:
    try:
        value = int(path.read_text().strip())
        return value if value >= 0 else None
    except (OSError, ValueError):
        return None


def _gpu_sample() -> dict[str, Any]:
    missing = {"gpu_count": None, "gpu_utilization_mean": None, "gpu_memory_bytes": None}
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        return missing
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        utilization, memory, devices = [], [], []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            utilization.append(int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu))
            memory.append(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
            identity = pynvml.nvmlDeviceGetUUID(handle)
            encoded = identity if isinstance(identity, bytes) else str(identity).encode()
            devices.append(
                {
                    "device_hash": hashlib.sha256(encoded).hexdigest()[:16],
                    "utilization": utilization[-1],
                    "memory_bytes": memory[-1],
                }
            )
        return {
            "gpu_count": count,
            "gpu_utilization_mean": sum(utilization) / count if count else None,
            "gpu_memory_bytes": sum(memory),
            "gpu_devices": devices,
        }
    except Exception:
        return missing
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


class ResourceSampler:
    def __init__(self, cgroup_root: Path = Path("/sys/fs/cgroup")) -> None:
        self.root = cgroup_root
        self.previous_cpu: int | None = None
        self.previous_time: float | None = None

    def sample(self, *, gpu: bool) -> dict[str, Any]:
        now = time.monotonic()
        interval = now - self.previous_time if self.previous_time is not None else None
        used = None
        quota = None
        throttled = None
        try:
            stats = dict(line.split() for line in (self.root / "cpu.stat").read_text().splitlines())
            used = int(stats["usage_usec"])
            throttled = int(stats["throttled_usec"]) if "throttled_usec" in stats else None
            limit, period = (self.root / "cpu.max").read_text().split()
            quota = int(limit) / int(period) if limit != "max" else None
        except (OSError, KeyError, ValueError, ZeroDivisionError):
            pass
        used = used if used is not None and used >= 0 else None
        throttled = throttled if throttled is not None and throttled >= 0 else None
        quota = quota if quota is not None and quota > 0 else None
        cores = None
        if used is not None and self.previous_cpu is not None and self.previous_time is not None:
            elapsed = now - self.previous_time
            if elapsed > 0 and used >= self.previous_cpu:
                cores = (used - self.previous_cpu) / 1_000_000 / elapsed
        self.previous_cpu, self.previous_time = used, now
        io: dict[str, int] | None = None
        try:
            io = {key: 0 for key in ("rbytes", "wbytes", "rios", "wios")}
            for line in (self.root / "io.stat").read_text().splitlines():
                for field in line.split()[1:]:
                    key, value = field.split("=", 1)
                    if key in io:
                        io[key] += int(value)
        except (OSError, ValueError):
            io = None
        return {
            "sampling": "cgroup_v2_interval_nvml_instantaneous",
            "cpu_cores_used": cores,
            "cpu_quota_cores": quota,
            "cpu_quota_utilization": cores / quota if cores is not None and quota else None,
            "cpu_throttled_usec": throttled,
            "sample_unix_seconds": time.time(),
            "interval_seconds": interval,
            "cgroup_io_totals": io,
            "memory_current_bytes": _integer(self.root / "memory.current"),
            "memory_limit_bytes": _integer(self.root / "memory.max"),
            **(
                _gpu_sample()
                if gpu
                else {
                    "gpu_count": None,
                    "gpu_utilization_mean": None,
                    "gpu_memory_bytes": None,
                }
            ),
        }
