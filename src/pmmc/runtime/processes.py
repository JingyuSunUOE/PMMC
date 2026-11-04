"""Managed loopback model-service subprocesses.

The parent process deliberately avoids importing Torch. GPU discovery and
telemetry remain NVML-only until each isolated vLLM child starts.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx
import yaml  # type: ignore[import-untyped]

from pmmc.models.registry import ModelRegistry, ModelRegistryEntry
from pmmc.runtime.memory_budget import allocation_vllm_memory_budget
from pmmc.schemas import (
    GpuAllocation,
    ManagedBackendKind,
    ManagedServiceSpec,
    ModelRole,
)


class ManagedProcessError(RuntimeError):
    """A managed child failed startup, health validation, or shutdown."""


@dataclass(slots=True)
class ManagedProcessHandle:
    """In-memory child state; credentials are never serialized."""

    service_id: str
    process: subprocess.Popen[bytes]
    endpoint: str
    api_key: str
    log_path: Path
    model_aliases: list[str]
    environment_overlay: dict[str, str] = field(default_factory=dict)


class ManagedServiceRunner(Protocol):
    def start(
        self,
        *,
        spec: ManagedServiceSpec,
        allocation: GpuAllocation,
        entry: ModelRegistryEntry,
        snapshot_path: Path,
        api_key: str,
        log_dir: Path,
        environment: dict[str, str],
    ) -> ManagedProcessHandle: ...

    def wait_ready(
        self,
        handle: ManagedProcessHandle,
        *,
        expected_model_ids: list[str],
        timeout_seconds: int,
        check_health: Callable[[], None] | None = None,
    ) -> None: ...

    def stop(self, handle: ManagedProcessHandle, *, timeout_seconds: int) -> int: ...


def random_runtime_key() -> str:
    return "pmmc-local-" + secrets.token_urlsafe(32)


def _assert_loopback_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            stream.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ManagedProcessError(f"loopback port is unavailable: {port}") from exc


def _selected_cuda_devices(
    environment: dict[str, str],
    visible_indices: list[int],
) -> str:
    """Resolve parent-visible GPU indices into a child-only CUDA device set."""

    parent_value = environment.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not parent_value:
        return ",".join(str(index) for index in visible_indices)
    parent_devices = [item.strip() for item in parent_value.split(",") if item.strip()]
    if any(index >= len(parent_devices) for index in visible_indices):
        raise ManagedProcessError(
            "runtime GPU allocation falls outside the parent CUDA_VISIBLE_DEVICES"
        )
    return ",".join(parent_devices[index] for index in visible_indices)


def _spawn(
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("ab", buffering=0)
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            close_fds=True,
        )
    except Exception:
        log_stream.close()
        raise


def _stop_process(process: subprocess.Popen[bytes], timeout_seconds: int) -> int:
    if process.poll() is not None:
        return int(process.returncode or 0)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return int(process.poll() or 0)
    try:
        return int(process.wait(timeout=timeout_seconds))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return int(process.wait(timeout=max(5, timeout_seconds)))


def _poll_models(
    handle: ManagedProcessHandle,
    *,
    expected_model_ids: list[str],
    timeout_seconds: int,
    health_path: str = "/health",
    check_health: Callable[[], None] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    started_at = time.monotonic()
    next_progress_at = started_at + 30.0
    headers = {"Authorization": f"Bearer {handle.api_key}"}
    last_error = "not ready"
    with httpx.Client(timeout=5.0, headers=headers) as client:
        while time.monotonic() < deadline:
            if check_health is not None:
                check_health()
            return_code = handle.process.poll()
            if return_code is not None:
                diagnostic_codes = service_log_diagnostic_codes(handle.log_path)
                raise ManagedProcessError(
                    f"service {handle.service_id!r} exited during startup with code "
                    f"{return_code}; diagnostic_codes={diagnostic_codes}"
                )
            try:
                health = client.get(handle.endpoint + health_path)
                health.raise_for_status()
                models = client.get(handle.endpoint + "/v1/models")
                models.raise_for_status()
                payload = models.json()
                actual = {
                    str(item.get("id"))
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and item.get("id")
                }
                missing = set(expected_model_ids) - actual
                if not missing:
                    if check_health is not None:
                        check_health()
                    return
                last_error = f"missing served model IDs: {sorted(missing)}"
            except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
                last_error = type(exc).__name__
            now = time.monotonic()
            if now >= next_progress_at:
                print(
                    "RUNTIME_STARTUP "
                    f"service={handle.service_id} state=waiting "
                    f"elapsed={now - started_at:.1f}s last_error={last_error}",
                    flush=True,
                )
                next_progress_at = now + 30.0
            time.sleep(1.0)
    diagnostic_codes = service_log_diagnostic_codes(handle.log_path)
    raise ManagedProcessError(
        f"service {handle.service_id!r} did not become ready: {last_error}; "
        f"diagnostic_codes={diagnostic_codes}"
    )


class VllmServiceRunner:
    """Start one vLLM 0.24 OpenAI-compatible service on loopback."""

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("vllm") or "vllm"

    def build_command(
        self,
        *,
        spec: ManagedServiceSpec,
        allocation: GpuAllocation,
        entry: ModelRegistryEntry,
        snapshot_path: Path,
    ) -> list[str]:
        if spec.port is None:
            raise ValueError(f"managed vLLM service requires a port: {spec.service_id}")
        if any(arg.split("=", 1)[0] in {"--kv-cache-memory-bytes", "--num-gpu-blocks-override"}
               for arg in spec.extra_args):
            raise ValueError("vllm_memory_budget_override_forbidden")
        budget = allocation_vllm_memory_budget(allocation)
        if budget.requested_utilization > spec.gpu_memory_utilization:
            raise ValueError("vllm_memory_budget_exceeds_service_limit")
        command = [
            self.executable,
            "serve",
            str(snapshot_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(spec.port),
            "--served-model-name",
            entry.expected_actual_model_id,
            "--device-ids",
            ",".join(str(index) for index in range(allocation.tensor_parallel_size)),
            "--tensor-parallel-size",
            str(allocation.tensor_parallel_size),
            "--dtype",
            spec.dtype,
            "--gpu-memory-utilization",
            str(budget.effective_utilization),
            "--disable-uvicorn-access-log",
        ]
        if spec.backend is ManagedBackendKind.VLLM_CHAT:
            command.extend(["--generation-config", "vllm"])
        else:
            command.extend(["--runner", "pooling"])
        if spec.quantization is not None:
            command.extend(["--quantization", spec.quantization])
        if spec.max_model_len is not None:
            command.extend(["--max-model-len", str(spec.max_model_len)])
        if spec.max_num_seqs is not None:
            command.extend(["--max-num-seqs", str(spec.max_num_seqs)])
        if spec.max_num_batched_tokens is not None:
            command.extend(["--max-num-batched-tokens", str(spec.max_num_batched_tokens)])
        command.extend(spec.extra_args)
        return command

    def start(
        self,
        *,
        spec: ManagedServiceSpec,
        allocation: GpuAllocation,
        entry: ModelRegistryEntry,
        snapshot_path: Path,
        api_key: str,
        log_dir: Path,
        environment: dict[str, str],
    ) -> ManagedProcessHandle:
        if spec.port is None:
            raise ValueError(f"managed vLLM service requires a port: {spec.service_id}")
        _assert_loopback_port_available(spec.port)
        child_environment = dict(environment)
        child_environment.update(
            {
                "CUDA_VISIBLE_DEVICES": _selected_cuda_devices(
                    environment, allocation.visible_indices
                ),
                "VLLM_API_KEY": api_key,
                "VLLM_NO_USAGE_STATS": "1",
                "DO_NOT_TRACK": "1",
            }
        )
        log_path = log_dir / f"{spec.service_id}.log"
        process = _spawn(
            self.build_command(
                spec=spec,
                allocation=allocation,
                entry=entry,
                snapshot_path=snapshot_path,
            ),
            environment=child_environment,
            log_path=log_path,
        )
        endpoint = f"http://127.0.0.1:{spec.port}"
        return ManagedProcessHandle(
            service_id=spec.service_id,
            process=process,
            endpoint=endpoint,
            api_key=api_key,
            log_path=log_path,
            model_aliases=[entry.alias],
        )

    def wait_ready(
        self,
        handle: ManagedProcessHandle,
        *,
        expected_model_ids: list[str],
        timeout_seconds: int,
        check_health: Callable[[], None] | None = None,
    ) -> None:
        _poll_models(
            handle,
            expected_model_ids=expected_model_ids,
            timeout_seconds=timeout_seconds,
            check_health=check_health,
        )

    def stop(self, handle: ManagedProcessHandle, *, timeout_seconds: int) -> int:
        return _stop_process(handle.process, timeout_seconds)


class ContrastiveServiceRunner:
    """Start the isolated PMMC Transformers contrastive worker on loopback."""

    def build_command(
        self,
        *,
        spec: ManagedServiceSpec,
        entry: ModelRegistryEntry,
        snapshot_path: Path,
    ) -> list[str]:
        if spec.port is None:
            raise ValueError(f"managed contrastive service requires a port: {spec.service_id}")
        return [
            sys.executable,
            "-m",
            "pmmc.models.contrastive_worker",
            "--model-path",
            str(snapshot_path),
            "--served-model-name",
            entry.expected_actual_model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(spec.port),
            "--dtype",
            spec.dtype,
            "--text-max-length",
            "64",
        ]

    def start(
        self,
        *,
        spec: ManagedServiceSpec,
        allocation: GpuAllocation,
        entry: ModelRegistryEntry,
        snapshot_path: Path,
        api_key: str,
        log_dir: Path,
        environment: dict[str, str],
    ) -> ManagedProcessHandle:
        if spec.backend is not ManagedBackendKind.TRANSFORMERS_CONTRASTIVE:
            raise ValueError("contrastive runner received a non-contrastive service")
        if allocation.tensor_parallel_size != 1 or len(allocation.visible_indices) != 1:
            raise ValueError("contrastive worker requires one exclusive GPU")
        if spec.port is None:
            raise ValueError(f"managed contrastive service requires a port: {spec.service_id}")
        _assert_loopback_port_available(spec.port)
        child_environment = dict(environment)
        child_environment.update(
            {
                "CUDA_VISIBLE_DEVICES": _selected_cuda_devices(
                    environment,
                    allocation.visible_indices,
                ),
                "PMMC_CONTRASTIVE_API_KEY": api_key,
                "DO_NOT_TRACK": "1",
            }
        )
        log_path = log_dir / f"{spec.service_id}.log"
        process = _spawn(
            self.build_command(spec=spec, entry=entry, snapshot_path=snapshot_path),
            environment=child_environment,
            log_path=log_path,
        )
        return ManagedProcessHandle(
            service_id=spec.service_id,
            process=process,
            endpoint=f"http://127.0.0.1:{spec.port}",
            api_key=api_key,
            log_path=log_path,
            model_aliases=[entry.alias],
        )

    def wait_ready(
        self,
        handle: ManagedProcessHandle,
        *,
        expected_model_ids: list[str],
        timeout_seconds: int,
        check_health: Callable[[], None] | None = None,
    ) -> None:
        _poll_models(
            handle,
            expected_model_ids=expected_model_ids,
            timeout_seconds=timeout_seconds,
            check_health=check_health,
        )

    def stop(self, handle: ManagedProcessHandle, *, timeout_seconds: int) -> int:
        return _stop_process(handle.process, timeout_seconds)


class LiteLLMServiceRunner:
    """Generate a secret-free local routing config and start LiteLLM proxy."""

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("litellm") or "litellm"

    @staticmethod
    def write_config(
        *,
        registry: ModelRegistry,
        chat_services: list[tuple[ManagedServiceSpec, ManagedProcessHandle]],
        path: Path,
    ) -> None:
        model_list: list[dict[str, object]] = []
        for spec, handle in chat_services:
            for alias in spec.all_served_aliases:
                entry = registry.get(alias)
                model_list.append(
                    {
                        "model_name": entry.provider_model,
                        "litellm_params": {
                            "model": f"openai/{entry.expected_actual_model_id}",
                            "api_base": handle.endpoint + "/v1",
                            "api_key": "os.environ/VLLM_API_KEY",
                        },
                    }
                )
        payload = {
            "model_list": model_list,
            "router_settings": {
                "num_retries": 0,
                "fallbacks": [],
                "enable_pre_call_checks": True,
            },
            "general_settings": {
                "master_key": "os.environ/LITELLM_API_KEY",
                "disable_spend_logs": False,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=False, sort_keys=False),
            encoding="utf-8",
        )

    def start(
        self,
        *,
        registry: ModelRegistry,
        chat_services: list[tuple[ManagedServiceSpec, ManagedProcessHandle]],
        port: int,
        api_key: str,
        log_dir: Path,
        environment: dict[str, str],
    ) -> ManagedProcessHandle:
        if not chat_services:
            raise ValueError("LiteLLM requires at least one active chat service")
        _assert_loopback_port_available(port)
        config_path = log_dir / "litellm.generated.yaml"
        self.write_config(
            registry=registry,
            chat_services=chat_services,
            path=config_path,
        )
        child_environment = dict(environment)
        child_environment.update({"LITELLM_API_KEY": api_key, "VLLM_API_KEY": api_key})
        process = _spawn(
            [
                self.executable,
                "--config",
                str(config_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            environment=child_environment,
            log_path=log_dir / "litellm.log",
        )
        return ManagedProcessHandle(
            service_id="litellm-proxy",
            process=process,
            endpoint=f"http://127.0.0.1:{port}",
            api_key=api_key,
            log_path=log_dir / "litellm.log",
            model_aliases=list(
                dict.fromkeys(
                    alias for spec, _ in chat_services for alias in spec.all_served_aliases
                )
            ),
        )

    def wait_ready(
        self,
        handle: ManagedProcessHandle,
        *,
        expected_model_ids: list[str],
        timeout_seconds: int,
        check_health: Callable[[], None] | None = None,
    ) -> None:
        _poll_models(
            handle,
            expected_model_ids=expected_model_ids,
            timeout_seconds=timeout_seconds,
            health_path="/health/liveliness",
            check_health=check_health,
        )

    def stop(self, handle: ManagedProcessHandle, *, timeout_seconds: int) -> int:
        return _stop_process(handle.process, timeout_seconds)


def endpoint_overlay(
    services: list[tuple[ManagedServiceSpec, ManagedProcessHandle]],
    *,
    litellm: ManagedProcessHandle | None,
    api_key: str,
    runtime_id: str,
    stage_id: str,
) -> dict[str, str]:
    """Build an in-memory Settings overlay for a child PMMC command."""

    service_map: dict[str, str] = {}
    endpoint_pools: dict[str, list[dict[str, object]]] = {}
    for spec, handle in services:
        runtime_service_id = f"{runtime_id}:{stage_id}:{spec.service_id}"
        for alias in spec.all_served_aliases:
            service_map.setdefault(alias, runtime_service_id)
            endpoint_pools.setdefault(alias, []).append(
                {
                    "runtime_service_id": runtime_service_id,
                    "service_id": spec.service_id,
                    "endpoint": handle.endpoint,
                    "replica_index": int(spec.metadata.get("replica_index", 0)),
                }
            )
    for bindings in endpoint_pools.values():
        bindings.sort(
            key=lambda item: (
                int(str(item["replica_index"])),
                str(item["service_id"]),
            )
        )
    overlay = {
        "PMMC_RUNTIME_MODE": "managed_local",
        "PMMC_ACTIVE_RUNTIME_ID": runtime_id,
        "PMMC_ACTIVE_RUNTIME_SERVICE_MAP": json.dumps(service_map, sort_keys=True),
        "PMMC_ACTIVE_RUNTIME_ENDPOINT_POOLS": json.dumps(
            endpoint_pools,
            sort_keys=True,
        ),
        "VLLM_API_KEY": api_key,
    }
    if litellm is not None:
        overlay.update(
            {
                "LITELLM_BASE_URL": litellm.endpoint,
                "LITELLM_API_KEY": api_key,
            }
        )
    for spec, handle in services:
        if spec.role is ModelRole.TEXT_EMBEDDING:
            overlay["VLLM_TEXT_EMBEDDING_BASE_URL"] = handle.endpoint
        elif spec.role is ModelRole.MULTIMODAL_EMBEDDING:
            overlay["VLLM_MULTIMODAL_EMBEDDING_BASE_URL"] = handle.endpoint
        elif spec.role is ModelRole.CONTRASTIVE_EMBEDDING:
            overlay["PMMC_CONTRASTIVE_EMBEDDING_BASE_URL"] = handle.endpoint
        elif spec.role in {
            ModelRole.ANSWER,
            ModelRole.COMPILER,
            ModelRole.JUDGE,
            ModelRole.RERANKER,
        }:
            overlay.setdefault("VLLM_CHAT_BASE_URL", handle.endpoint)
    return overlay


def redact_service_log(path: Path, *, secrets_to_remove: list[str], roots: list[Path]) -> Path:
    """Create a deterministic redacted log suitable for archival."""

    target = path.with_suffix(path.suffix + ".redacted")
    if not path.exists():
        target.write_text("", encoding="utf-8")
        return target
    content = path.read_text(encoding="utf-8", errors="replace")
    for secret in secrets_to_remove:
        if secret:
            content = content.replace(secret, "[REDACTED]")
    for root in roots:
        content = content.replace(str(root), "<LOCAL_PATH>")
    target.write_text(content, encoding="utf-8")
    return target


_SERVICE_LOG_DIAGNOSTIC_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cuda_out_of_memory", ("cuda out of memory", "torch.outofmemoryerror")),
    ("gpu_process_lost", ("cuda error: unknown error", "device-side assert")),
    (
        "engine_start_timeout",
        (
            "timed out waiting for engine core",
            "engine ready timeout",
            "vllm_engine_ready_timeout_s",
        ),
    ),
    ("worker_rpc_timeout", ("execute_model timed out", "rpc timeout")),
    ("address_in_use", ("address already in use", "errno 98")),
    ("cache_lock_timeout", ("lock timeout", "timed out waiting for lock")),
    ("cache_corrupt", ("failed to load cache", "invalid cache", "cache corruption")),
    ("disk_full", ("no space left on device", "errno 28")),
    ("permission_denied", ("permission denied", "errno 13")),
    (
        "structured_output_schema_unsupported",
        (
            "json schema contains features not supported by xgrammar",
            "provided json schema contains features not supported by xgrammar",
        ),
    ),
    ("model_file_missing", ("no such file or directory", "filenotfounderror")),
    ("process_killed", ("killed", "sigkill", "signal 9")),
    (
        "engine_process_failed",
        (
            "engine core failed",
            "engine process failed",
            "engine core initialization failed",
        ),
    ),
)


def service_log_diagnostic_codes(path: Path, *, tail_bytes: int = 1_048_576) -> list[str]:
    """Classify a service failure without exposing model paths or request payloads."""

    if not path.is_file():
        return ["log_missing"]
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - tail_bytes))
            content = stream.read().decode("utf-8", errors="replace").lower()
    except OSError:
        return ["log_unreadable"]
    codes = [
        code
        for code, needles in _SERVICE_LOG_DIAGNOSTIC_PATTERNS
        if any(needle in content for needle in needles)
    ]
    return sorted(set(codes)) or ["unclassified"]


def describe_process(handle: ManagedProcessHandle) -> str:
    """Return a secret-free one-line diagnostic."""

    return json.dumps(
        {
            "service_id": handle.service_id,
            "pid": handle.process.pid,
            "endpoint_host": "127.0.0.1",
            "endpoint_port": int(handle.endpoint.rsplit(":", 1)[1]),
            "model_aliases": handle.model_aliases,
        },
        sort_keys=True,
    )


def fetch_service_info(handle: ManagedProcessHandle) -> dict[str, object]:
    """Read secret-free serving identity/version data for calibration provenance."""

    headers = {"Authorization": f"Bearer {handle.api_key}"}
    with httpx.Client(timeout=10.0, headers=headers) as client:
        models = client.get(handle.endpoint + "/v1/models")
        models.raise_for_status()
        model_ids = sorted(
            str(item["id"])
            for item in models.json().get("data", [])
            if isinstance(item, dict) and item.get("id")
        )
        version_value: str | None = None
        version = client.get(handle.endpoint + "/version")
        if version.is_success:
            payload = version.json()
            if isinstance(payload, dict) and payload.get("version") is not None:
                version_value = str(payload["version"])
    return {"served_model_ids": model_ids, "server_version": version_value}
