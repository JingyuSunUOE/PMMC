"""GPU inventory and topology discovery without importing Torch or CUDA Python APIs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime

from pmmc.runtime.device_accounting import DeviceMemorySample
from pmmc.schemas import GpuDeviceInfo, GpuTopologyLink


class HardwareInventoryError(RuntimeError):
    """Raised when the Pod does not expose a usable NVIDIA runtime."""


@dataclass(frozen=True)
class HardwareInventory:
    devices: list[GpuDeviceInfo]
    topology: list[GpuTopologyLink]
    driver_version: str
    cuda_driver_version: str | None = None

    @property
    def fingerprint(self) -> str:
        payload = {
            "driver_version": self.driver_version,
            "cuda_driver_version": self.cuda_driver_version,
            "devices": [
                {
                    "name": item.name,
                    "total_memory_bytes": item.total_memory_bytes,
                    "compute_capability": item.compute_capability,
                    "pci_bus_id": item.pci_bus_id,
                }
                for item in self.devices
            ],
            "topology": [
                {
                    "source": item.source_device_id,
                    "target": item.target_device_id,
                    "link_type": item.link_type,
                }
                for item in self.topology
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def anonymize_gpu_uuid(uuid: str) -> str:
    """Create a stable local identifier without persisting the complete UUID."""

    return "gpu-" + hashlib.sha256(uuid.encode("utf-8")).hexdigest()[:16]


def topology_affinity(link_type: str) -> int:
    """Map nvidia-smi topology labels to a deterministic 0-100 affinity."""

    normalized = link_type.strip().upper()
    if normalized.startswith("NV"):
        return 100
    return {
        "PIX": 80,
        "PXB": 65,
        "PHB": 45,
        "NODE": 30,
        "SYS": 10,
    }.get(normalized, 0)


def parse_nvidia_smi_topology(
    output: str,
    devices: list[GpuDeviceInfo],
) -> list[GpuTopologyLink]:
    """Parse the GPU-to-GPU matrix from ``nvidia-smi topo -m``."""

    by_label = {f"GPU{item.visible_index}": item for item in devices}
    lines = [line.split() for line in output.splitlines() if line.strip()]
    header = next((line for line in lines if line and line[0].startswith("GPU")), None)
    if header is None:
        return []
    gpu_columns = [value for value in header if value in by_label]
    links: list[GpuTopologyLink] = []
    for row in lines:
        if row is header:
            continue
        if not row or row[0] not in by_label:
            continue
        source = by_label[row[0]]
        values = row[1 : 1 + len(gpu_columns)]
        for label, value in zip(gpu_columns, values, strict=False):
            target = by_label[label]
            if target.device_id == source.device_id or value.upper() == "X":
                continue
            links.append(
                GpuTopologyLink(
                    source_device_id=source.device_id,
                    target_device_id=target.device_id,
                    link_type=value,
                    affinity_score=topology_affinity(value),
                )
            )
    return sorted(
        links,
        key=lambda item: (item.source_device_id, item.target_device_id),
    )


class NvmlHardwareProvider:
    """Read GPU state through NVML without initializing a CUDA context."""

    def inventory(self) -> HardwareInventory:
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise HardwareInventoryError("nvidia-ml-py is not installed") from exc

        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count <= 0:
                raise HardwareInventoryError("the container exposes no NVIDIA GPUs")
            devices: list[GpuDeviceInfo] = []
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                raw_uuid = pynvml.nvmlDeviceGetUUID(handle)
                name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
                uuid = raw_uuid.decode() if isinstance(raw_uuid, bytes) else str(raw_uuid)
                pci = pynvml.nvmlDeviceGetPciInfo(handle)
                raw_bus_id = getattr(pci, "busId", None)
                bus_id = raw_bus_id.decode() if isinstance(raw_bus_id, bytes) else raw_bus_id
                capability = None
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    capability = f"{major}.{minor}"
                except pynvml.NVMLError:
                    pass
                devices.append(
                    GpuDeviceInfo(
                        device_id=anonymize_gpu_uuid(uuid),
                        visible_index=index,
                        name=name,
                        total_memory_bytes=int(memory.total),
                        free_memory_bytes=int(memory.free),
                        used_memory_bytes=int(memory.used),
                        compute_capability=capability,
                        pci_bus_id=str(bus_id) if bus_id else None,
                    )
                )
            raw_driver = pynvml.nvmlSystemGetDriverVersion()
            driver = raw_driver.decode() if isinstance(raw_driver, bytes) else str(raw_driver)
            cuda_driver = None
            try:
                raw_cuda = int(pynvml.nvmlSystemGetCudaDriverVersion_v2())
                cuda_driver = f"{raw_cuda // 1000}.{(raw_cuda % 1000) // 10}"
            except (AttributeError, pynvml.NVMLError):
                pass
        except HardwareInventoryError:
            raise
        except pynvml.NVMLError as exc:
            raise HardwareInventoryError(f"NVML inventory failed: {type(exc).__name__}") from exc
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        topology: list[GpuTopologyLink] = []
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            topology = parse_nvidia_smi_topology(result.stdout, devices)
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
        return HardwareInventory(
            devices=devices,
            topology=topology,
            driver_version=driver,
            cuda_driver_version=cuda_driver,
        )

    def device_memory_samples(self, indices: set[int]) -> list[DeviceMemorySample]:
        """Total allocated-device usage; no PID translation or baseline subtraction."""
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            values = []
            for index in sorted(indices):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                compute = getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None)
                graphics = getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses_v3", None)
                compute = compute or pynvml.nvmlDeviceGetComputeRunningProcesses
                graphics = graphics or pynvml.nvmlDeviceGetGraphicsRunningProcesses
                values.append(DeviceMemorySample(
                    device_id=anonymize_gpu_uuid(
                        uuid.decode() if isinstance(uuid, bytes) else uuid
                    ),
                    visible_index=index, total_bytes=memory.total, used_bytes=memory.used,
                    compute_process_count=len(compute(handle)),
                    graphics_process_count=len(graphics(handle)), sampled_at=datetime.now(UTC),
                ))
            return values
        finally:
            pynvml.nvmlShutdown()

    def process_memory_bytes(self, process_ids: set[int]) -> dict[int, int]:
        """Return aggregate NVML compute-process memory for selected PIDs."""

        if not process_ids:
            return {}
        import pynvml  # type: ignore[import-not-found]

        usage = {pid: 0 for pid in process_ids}
        observed = set()
        try:
            pynvml.nvmlInit()
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
                except AttributeError:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for process in processes:
                    observed.add(process.pid)
                    if process.pid in usage:
                        value = process.usedGpuMemory
                        if type(value) is not int or not 0 <= value < 2**63:
                            raise HardwareInventoryError("nvml_process_memory_unavailable")
                        usage[process.pid] += value
            if observed - process_ids:
                raise HardwareInventoryError("nvml_process_identity_unverified")
        finally:
            pynvml.nvmlShutdown()
        return usage
