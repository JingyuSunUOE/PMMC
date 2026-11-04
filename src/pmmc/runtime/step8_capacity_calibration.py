"""Isolated, measured full-pool Step 8 capacity calibration.

Call calibrate_step8_capacity before starting the gated benchmark service. This
owns the normal Pod runtime lock, uses verified model preparation and the existing
ActiveRuntimeStage lifecycle, and shuts down the complete pool before publishing
evidence. It never alters the source profile or relaxes normal paper startup.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from pmmc.infrastructure.settings import Settings
from pmmc.models.registry import load_model_registry
from pmmc.runtime.config import load_runtime_profile
from pmmc.runtime.device_accounting import DeviceMemoryBudgetExceeded, DeviceMemorySample
from pmmc.runtime.hardware import anonymize_gpu_uuid
from pmmc.runtime.locking import RuntimeProcessLock
from pmmc.runtime.manager import (
    ActiveRuntimeStage,
    ManagedRuntimeError,
    PreparedRuntime,
    _persist_session,
    _runtime_session,
    _service_backend_version,
    _service_process_ids,
    _write_json,
    prepare_runtime,
)
from pmmc.runtime.signals import ManagedSignalGuard
from pmmc.runtime.step8_capacity import (
    CapacityImageWorkload,
    CapacityPool,
    CapacityReplicaWorkload,
    CapacityScenarioMeasurement,
    Step8PoolCapacityReceipt,
    capacity_image_workload,
    capacity_pools,
    capacity_processor_kwargs,
    parse_image_workload,
    validate_capacity_receipt,
)
from pmmc.runtime.step8_capacity_counter_process import NativeCounterProcess
from pmmc.runtime.step8_capacity_images import (
    fit_image_prompt,
    image_request_hash,
    synthetic_image_messages,
    synthetic_native_messages,
    validate_image_messages,
)
from pmmc.runtime.step8_capacity_native import NativeCapacityImageWorkload, NativeImageLayout
from pmmc.schemas import (
    ManagedBackendKind,
    ModelCapability,
    ModelVerificationPolicy,
    RuntimeStage,
    RuntimeStatus,
)


def calibrate_step8_capacity(
    settings: Settings,
    *,
    profile_path: Path | None = None,
    logical_stage: str = "answer_27b",
    verification_policy: ModelVerificationPolicy | None = None,
    verified_cache_manifest_path: Path | None = None,
    output_tokens: int = 256,
    timeout_seconds: float = 900.0,
    sample_interval_seconds: float = 0.1,
) -> tuple[Step8PoolCapacityReceipt, Path]:
    """Measure and atomically write a receipt; return ``(receipt, receipt_path)``.

    Always performs fresh measurement (not an idle-calibration/cache fallback).
    The frozen source profile determines TP, replica count, concurrency and total
    context length. output_tokens splits that total into prompt and decode tokens;
    it cannot reduce total context coverage. timeout_seconds bounds the workload,
    while normal profile timeouts bound service startup/shutdown. Invoke before,
    not inside, another managed runtime lock/session. No external API is used:
    synthetic requests go directly to all managed loopback replicas.
    """
    _validate_bounds(output_tokens, timeout_seconds, sample_interval_seconds)
    path = (
        settings.resolve_path(profile_path)
        if profile_path is not None
        else settings.resolved_runtime_profile_path
    )
    profile = load_runtime_profile(path)
    services = profile.services_for_stage(logical_stage)
    if len(services) != 1 or services[0].metadata.get("requires_capacity_receipt") is not True:
        raise ManagedRuntimeError("capacity calibration requires one isolated gated source pool")
    frozen = services[0].metadata.get("frozen_replica_topology")
    count = frozen.get("replica_count") if isinstance(frozen, dict) else None
    if type(count) is not int or count <= 0:
        raise ManagedRuntimeError("capacity calibration requires a frozen replica count")
    # Reject known-invalid input policy before NFS flock, model verification or NVML.
    # This is particularly important on controllers with a broken NFS lock service.
    registry = load_model_registry(settings.resolved_model_registry_path)
    try:
        bounds = capacity_image_workload(
            services[0], multimodal=ModelCapability.CHAT_MULTIMODAL in
            registry.get(services[0].model_alias).capabilities,
        )
    except ValueError:
        print("STEP8_CAPACITY_PREFLIGHT state=failed "
              "code=capacity_image_configuration_required gpu_started=0", flush=True)
        raise ManagedRuntimeError("capacity_image_configuration_required") from None
    print("STEP8_CAPACITY_PREFLIGHT state=passed "
          f"replicas={count} images_per_sequence={bounds.images_per_sequence if bounds else 0} "
          "gpu_started=0", flush=True)
    runtime_lock = RuntimeProcessLock(
        settings.runs_path / "managed-runtime" / ".runtime.lock",
        f"step8-capacity-{os.getpid()}",
    )
    with runtime_lock, ManagedSignalGuard():
        prepared = prepare_runtime(
            settings,
            profile_path=path,
            logical_stage=logical_stage,
            replica_count=count,
            verification_policy=verification_policy,
            verified_cache_manifest_path=verified_cache_manifest_path,
            # Receipts can authorize paper startup even when measured in dev mode.
            paper_result=True,
        )
        runtime_lock.set_runtime_id(prepared.plan.runtime_id)
        if prepared.profile.resolved_hash != profile.resolved_hash:
            raise ManagedRuntimeError("capacity source profile changed during preparation")
        if len(prepared.plan.stages) != 1:
            raise ManagedRuntimeError(
                "capacity calibration requires one concurrent full-pool stage"
            )
        return _calibrate_prepared_capacity(
            prepared,
            prepared.plan.stages[0],
            output_tokens=output_tokens,
            timeout_seconds=timeout_seconds,
            sample_interval_seconds=sample_interval_seconds,
        )


def _validate_bounds(output_tokens: int, timeout: float, interval: float) -> None:
    if type(output_tokens) is not int or not 1 <= output_tokens <= 1024:
        raise ManagedRuntimeError("capacity decode must be bounded to 1..1024 tokens")
    if (
        not math.isfinite(timeout)
        or not 0 < timeout <= 3600
        or not math.isfinite(interval)
        or not 0.01 <= interval <= 1
    ):
        raise ManagedRuntimeError("capacity timeout/sampling bounds are invalid")


def _synthetic_prompts(snapshot: Path, count: int, length: int) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )
    special = set(tokenizer.all_special_ids)
    vocabulary = sorted(
        {
            i
            for i in tokenizer.get_vocab().values()
            if type(i) is int and i >= 0 and i not in special
        }
    )
    if len(vocabulary) < count:
        raise ManagedRuntimeError("capacity tokenizer has insufficient non-special token IDs")
    # Distinct first tokens prevent prefix-cache sharing, including across replicas.
    prompts = []
    for ordinal in range(count):
        rng = random.Random(ordinal)
        prompts.append([vocabulary[ordinal], *rng.choices(vocabulary, k=length - 1)])
    return prompts


class _NvmlPoolSampler:
    """Per-device *owned process* memory, with no zero-on-error fallback."""

    def __init__(self, prepared: PreparedRuntime, active: ActiveRuntimeStage, pool: CapacityPool):
        self.prepared = prepared
        self.active = active
        self.pool = pool
        self.peaks: dict[str, int] = {}

    def _fail(self, code: str, **counts: int) -> None:
        diagnostic = {"protocol": "step8-capacity-nvml-v1", "state": "failed",
                      "code": code, **counts}
        try:
            _write_json(self.prepared.run_dir / "capacity-nvml-failure.json", diagnostic)
        except OSError:
            diagnostic["artifact_write_failed"] = True
        print("STEP8_CAPACITY_NVML " + json.dumps(diagnostic, sort_keys=True), flush=True)
        raise ManagedRuntimeError(code)

    def sample(self) -> None:
        import pynvml  # type: ignore[import-not-found]

        handles = {spec.service_id: handle for spec, handle in self.active.services}
        inventory = {device.device_id: device for device in self.prepared.inventory.devices}
        pynvml.nvmlInit()
        try:
            for ordinal, allocation in enumerate(self.pool.allocations):
                process = handles[allocation.service_id]
                if process.process.poll() is not None:
                    self._fail("capacity_replica_exited", replica_index=ordinal)
                pids = _service_process_ids(process)
                for device_id, index in zip(
                    allocation.device_ids,
                    allocation.visible_indices,
                    strict=True,
                ):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    raw_uuid = pynvml.nvmlDeviceGetUUID(handle)
                    uuid = raw_uuid.decode() if isinstance(raw_uuid, bytes) else str(raw_uuid)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    if (
                        anonymize_gpu_uuid(uuid) != device_id
                        or int(memory.total) != inventory[device_id].total_memory_bytes
                    ):
                        self._fail("capacity_nvml_device_identity_changed",
                                   replica_index=ordinal, device_index=index)
                    try:
                        processes = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
                    except AttributeError:
                        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    peak = 0
                    diagnostic = {
                        "replica_index": ordinal, "device_index": index,
                        "managed_process_count": len(pids),
                        "nvml_process_count": len(processes),
                        "matched_process_count": sum(item.pid in pids for item in processes),
                        "memory_budget_bytes": allocation.memory_budget_bytes_per_gpu,
                    }
                    if any(item.pid not in pids for item in processes):
                        # Do not infer ownership from device exclusivity or a PID number alone.
                        self._fail("capacity_nvml_process_ownership_unverified", **diagnostic)
                    for item in processes:
                        value = item.usedGpuMemory
                        if (
                            type(value) is not int
                            or value < 0
                            or value > int(memory.total)
                        ):
                            self._fail("capacity_nvml_process_memory_invalid", **diagnostic)
                        peak += value
                    if peak == 0:
                        self._fail("capacity_nvml_process_memory_missing", **diagnostic)
                    self.peaks[device_id] = max(self.peaks.get(device_id, 0), peak)
                    if peak > allocation.memory_budget_bytes_per_gpu:
                        self._fail("capacity_nvml_memory_over_budget", measured_bytes=peak,
                                   **diagnostic)
        finally:
            pynvml.nvmlShutdown()


class _SchedulerMetricsError(ManagedRuntimeError):
    def __init__(self, reason: str, metric: str, line_number: int | None = None):
        # Metric names come only from the fixed allowlist, never response labels/values.
        self.diagnostic = {"reason": reason, "metric": metric, "line_number": line_number}
        super().__init__("capacity_scheduler_metrics_invalid")


class _AllocatedDeviceSampler:
    def __init__(self, active: ActiveRuntimeStage):
        if active.device_accounting is None:
            raise ManagedRuntimeError("capacity_device_accounting_not_started")
        self.accounting = active.device_accounting
        self.peaks: dict[str, int] = {}

    def sample(self) -> None:
        try:
            samples = self.accounting.sample(require_active=True)
        except DeviceMemoryBudgetExceeded as exc:
            self._record(exc.samples)
            raise
        self._record(samples)

    def _record(self, samples: list[DeviceMemorySample]) -> None:
        for value in samples:
            self.peaks[value.device_id] = max(self.peaks.get(value.device_id, 0), value.used_bytes)


def _capacity_memory_diagnostic(
    pool: CapacityPool, *, peaks: dict[str, int], before_generation: dict[str, int],
    scope: str, state: str, phase: str, layout_hash: str | None,
) -> dict[str, Any]:
    devices = []
    for allocation in pool.allocations:
        for device_id, index in zip(allocation.device_ids, allocation.visible_indices, strict=True):
            peak = peaks.get(device_id)
            before = before_generation.get(device_id)
            devices.append({
                "visible_index": index, "budget_bytes": allocation.memory_budget_bytes_per_gpu,
                "pre_generation_peak_bytes": before, "peak_bytes": peak,
                "remaining_margin_bytes": (allocation.memory_budget_bytes_per_gpu - peak
                                           if peak is not None else None),
                "growth_after_pre_generation_bytes": (peak - before
                                                       if peak is not None and before is not None
                                                       else None),
            })
    return {
        "protocol": "step8-capacity-memory-observation-v1", "state": state,
        "phase": phase, "layout_hash": layout_hash, "memory_scope": scope,
        "memory_budget_contracts": pool.identity.get("memory_budget_contracts", []),
        "devices": devices,
    }


def _print_capacity_memory(diagnostic: dict[str, Any]) -> None:
    margins = [d["remaining_margin_bytes"] for d in diagnostic["devices"]
               if d["remaining_margin_bytes"] is not None]
    print("STEP8_CAPACITY_MEMORY " + json.dumps({
        "state": diagnostic["state"], "phase": diagnostic["phase"],
        "layout_hash": diagnostic["layout_hash"], "memory_scope": diagnostic["memory_scope"],
        "sampled_devices": len(margins), "expected_devices": len(diagnostic["devices"]),
        "min_remaining_margin_bytes": min(margins) if margins else None,
    }, sort_keys=True), flush=True)


def _scheduler_metrics(content: str) -> tuple[int, int, int]:
    # Read only these scalar samples, not a general Prometheus document. Optional
    # labels and timestamps do not contribute to the count; absent data fails closed.
    names = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:num_preemptions_total")
    values: dict[str, list[int]] = {name: [] for name in names}
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        name = re.match(r"([^\s{]+)", line)
        # vLLM 0.24 exports num_requests_waiting_by_reason alongside the total.
        # A prefix match would reject valid scrapes or double-count the breakdown.
        if name is None or name[1] not in values:
            continue
        match = re.fullmatch(r"([^\s{]+)(?:\{.*\})?\s+(\S+)(?:\s+\S+)?", line)
        if match is None:
            raise _SchedulerMetricsError("invalid_sample", name[1], line_number)
        try:
            value = float(match[2])
        except ValueError:
            raise _SchedulerMetricsError("invalid_value", name[1], line_number) from None
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise _SchedulerMetricsError("invalid_value", name[1], line_number)
        values[name[1]].append(int(value))
    for metric_name, samples in values.items():
        if not samples:
            raise _SchedulerMetricsError("missing_metric", metric_name)
    running, waiting, preemptions = [sum(values[name]) for name in names]
    return running, waiting, preemptions


def _check_completion(
    payload: object,
    *,
    service_id: str,
    model: str,
    sequences: int,
    prompt_tokens: int,
    output_tokens: int,
) -> CapacityReplicaWorkload:
    if not isinstance(payload, dict) or payload.get("model") != model:
        raise ManagedRuntimeError("capacity response model identity does not match")
    expected_usage = {
        "prompt_tokens": sequences * prompt_tokens,
        "completion_tokens": sequences * output_tokens,
        "total_tokens": sequences * (prompt_tokens + output_tokens),
    }
    usage = payload.get("usage")
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] != value
        for key, value in expected_usage.items()
    ):
        raise ManagedRuntimeError("capacity workload usage is missing or truncated")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != sequences:
        raise ManagedRuntimeError("capacity completion sequence coverage is incomplete")
    indices = []
    for choice in choices:
        if not isinstance(choice, dict):
            raise ManagedRuntimeError("capacity completion is invalid")
        logprobs = choice.get("logprobs")
        tokens = logprobs.get("tokens") if isinstance(logprobs, dict) else None
        if (
            choice.get("finish_reason") != "length"
            or type(choice.get("index")) is not int
            or not isinstance(tokens, list)
            or len(tokens) != output_tokens
        ):
            raise ManagedRuntimeError("capacity completion did not reach every token bound")
        indices.append(choice["index"])
    if sorted(indices) != list(range(sequences)):
        raise ManagedRuntimeError("capacity completion contains duplicate sequence indices")
    return CapacityReplicaWorkload(
        service_id=service_id,
        completed_sequences=sequences,
        prompt_tokens_per_sequence=prompt_tokens,
        completion_tokens_per_sequence=output_tokens,
        observed_running_sequences=sequences,
    )


def _check_image_completion(
    payload: object, *, service_id: str, model: str, prompt_tokens: int,
    output_tokens: int, request_hash: str,
) -> CapacityReplicaWorkload:
    if not isinstance(payload, dict):
        raise ManagedRuntimeError("capacity image response is invalid")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ManagedRuntimeError("capacity image completion sequence coverage is incomplete")
    choice = choices[0]
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list) or len(content) != output_tokens:
        raise ManagedRuntimeError("capacity image completion did not reach the token bound")
    normalized = {**payload, "choices": [{**choice, "logprobs": {"tokens": content}}]}
    result = _check_completion(
        normalized, service_id=service_id, model=model, sequences=1,
        prompt_tokens=prompt_tokens, output_tokens=output_tokens,
    )
    return result.model_copy(update={
        "completed_image_sequences": 1, "image_request_hashes": [request_hash],
    })


async def _measure_workload(
    prepared: PreparedRuntime,
    active: ActiveRuntimeStage,
    pool: CapacityPool,
    prompts: list,
    *,
    output_tokens: int,
    timeout_seconds: float,
    sample_interval_seconds: float,
    native_layout: NativeImageLayout | None = None,
    exact_counter: NativeCounterProcess | None = None,
) -> tuple[list[CapacityReplicaWorkload], dict[str, int], int]:
    spec = pool.services[0]
    assert spec.max_num_seqs is not None and spec.max_model_len is not None
    sequences = spec.max_num_seqs
    model = prepared.resolved_registry.get(spec.model_alias).expected_actual_model_id
    handles = {item.service_id: handle for item, handle in active.services}
    if len(active.services) != len(pool.services) or set(handles) != {
        item.service_id for item in pool.services
    }:
        raise ManagedRuntimeError("capacity active pool does not cover every replica")
    for service_id, handle in handles.items():
        endpoint = urlsplit(handle.endpoint)
        expected_port = next(s.port for s in pool.services if s.service_id == service_id)
        if (
            endpoint.scheme != "http"
            or endpoint.hostname != "127.0.0.1"
            or endpoint.port != expected_port
            or endpoint.path
            or endpoint.query
            or endpoint.fragment
            or endpoint.username
            or endpoint.password
        ):
            raise ManagedRuntimeError("capacity requests require exact managed loopback endpoints")
    multimodal = pool.identity["workload_kind"] == "multimodal"
    processor_kwargs = capacity_processor_kwargs(spec) if multimodal else None
    if native_layout is None and multimodal:
        validate_image_messages(
            CapacityImageWorkload.model_validate(pool.identity["image_workload"]),
            prompts, count=sequences * len(handles),
        )
    sampler = (_AllocatedDeviceSampler(active)
               if prepared.settings.gpu_memory_accounting == "allocated_device"
               else _NvmlPoolSampler(prepared, active, pool))
    tasks: list[asyncio.Task] = []
    progress_tasks: list[asyncio.Task] = []
    progress: dict[str, Any] = {"phase": "validation", "completed": 0, "total": len(prompts)}
    before_generation: dict[str, int] = {}
    started = time.monotonic()

    def memory_diagnostic(state: str) -> dict[str, Any]:
        return _capacity_memory_diagnostic(
            pool, peaks=sampler.peaks, before_generation=before_generation,
            scope=prepared.settings.gpu_memory_accounting, state=state,
            phase=progress["phase"], layout_hash=native_layout.identity if native_layout else None,
        )

    def advance(phase: str, total: int) -> None:
        progress.update(phase=phase, completed=0, total=total)
        print(f"STEP8_CAPACITY_PROGRESS phase={phase} state=start completed=0 "
              f"total={total} replicas={len(handles)} "
              f"elapsed_seconds={int(time.monotonic() - started)}", flush=True)

    async def heartbeat():
        while True:
            await asyncio.sleep(30)
            completed = (sum(task.done() and not task.cancelled() and task.exception() is None
                             for task in progress_tasks)
                         if progress_tasks else progress["completed"])
            print(f"STEP8_CAPACITY_PROGRESS phase={progress['phase']} completed={completed} "
                  f"total={progress['total']} replicas={len(handles)} "
                  f"elapsed_seconds={int(time.monotonic() - started)}", flush=True)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds),
        trust_env=False,
        follow_redirects=False,
        limits=httpx.Limits(max_connections=len(handles) * (sequences + 2)),
    ) as client:

        async def get(handle, route):
            response = await client.get(
                handle.endpoint + route,
                headers={"Authorization": f"Bearer {handle.api_key}"},
            )
            response.raise_for_status()
            return response

        async def metrics():
            responses = await asyncio.gather(
                *(get(handle, "/metrics") for handle in handles.values())
            )
            parsed = []
            for index, response in enumerate(responses):
                try:
                    parsed.append(_scheduler_metrics(response.text))
                except _SchedulerMetricsError as exc:
                    diagnostic = {
                        "protocol": "step8-scheduler-metrics-v1", "state": "failed",
                        "phase": progress["phase"], "replica_index": index,
                        "layout_hash": native_layout.identity if native_layout else None,
                        **exc.diagnostic,
                    }
                    _write_json(prepared.run_dir / "capacity-scheduler-failure.json", diagnostic)
                    print("STEP8_CAPACITY_METRICS " + json.dumps(diagnostic, sort_keys=True),
                          flush=True)
                    raise
            return parsed

        async def complete(ordinal, service_id, handle):
            response = await client.post(
                handle.endpoint + "/v1/completions",
                headers={"Authorization": f"Bearer {handle.api_key}"},
                json={
                    "model": model,
                    "prompt": prompts[ordinal * sequences : (ordinal + 1) * sequences],
                    "max_tokens": output_tokens,
                    "min_tokens": output_tokens,
                    "ignore_eos": True,
                    "add_special_tokens": False,
                    "temperature": 0,
                    "n": 1,
                    "stream": False,
                    "logprobs": 0,
                },
            )
            response.raise_for_status()
            return _check_completion(
                response.json(),
                service_id=service_id,
                model=model,
                sequences=sequences,
                prompt_tokens=spec.max_model_len - output_tokens,
                output_tokens=output_tokens,
            )

        async def complete_image(service_id, handle, messages):
            response = await client.post(
                handle.endpoint + "/v1/chat/completions",
                headers={"Authorization": f"Bearer {handle.api_key}"},
                json={
                    "model": model, "messages": messages, "max_tokens": output_tokens,
                    "min_tokens": output_tokens, "ignore_eos": True, "temperature": 0,
                    "n": 1, "stream": False, "logprobs": True, "top_logprobs": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "mm_processor_kwargs": processor_kwargs,
                },
            )
            response.raise_for_status()
            return _check_image_completion(
                response.json(), service_id=service_id, model=model,
                prompt_tokens=spec.max_model_len - output_tokens, output_tokens=output_tokens,
                request_hash=image_request_hash(messages),
            )

        reporter = asyncio.create_task(heartbeat())
        try:
            async with asyncio.timeout(timeout_seconds):
                if native_layout is not None:
                    advance("validate_pixels", len(prompts))
                    if exact_counter is None:
                        raise ManagedRuntimeError("capacity_native_exact_counter_required")
                    await exact_counter.validate_async(
                        native_layout, prompts, count=sequences * len(handles),
                    )
                advance("service_identity", len(handles))
                for handle in handles.values():
                    version = (await get(handle, "/version")).json()
                    models = (await get(handle, "/v1/models")).json()
                    if version.get("version") != pool.identity["backend_version"] or model not in {
                        item["id"] for item in models.get("data", [])
                    }:
                        raise ManagedRuntimeError("capacity live server identity does not match")
                    progress["completed"] += 1
                advance("scheduler_preflight", len(handles))
                before = await metrics()
                idle = all(not any(sample) for sample in before)
                layout_id = native_layout.identity if native_layout else "text"
                _write_json(prepared.run_dir / f"capacity-scheduler-preflight-{layout_id}.json", {
                    "protocol": "step8-scheduler-metrics-v1",
                    "state": "ready" if idle else "not_idle", "selector": "exact_name",
                    "replicas": [{"replica_index": index, "running": sample[0],
                                  "waiting": sample[1], "preemptions": sample[2]}
                                 for index, sample in enumerate(before)],
                })
                print(f"STEP8_CAPACITY_METRICS state={'ready' if idle else 'not_idle'} "
                      f"replicas={len(handles)} selector=exact_name required_metrics=3", flush=True)
                if not idle:
                    raise ManagedRuntimeError("capacity_scheduler_not_idle")
                # Verify process accounting before expensive multimodal token fitting.
                advance("nvml_preflight", len(pool.allocations))
                sampler.sample()
                if multimodal:
                    advance("fit_context", len(prompts))
                    async with asyncio.TaskGroup() as group:
                        fitted = [group.create_task(fit_image_prompt(
                            client, endpoint=handle.endpoint, api_key=handle.api_key, model=model,
                            messages=prompts[ordinal * sequences + sequence],
                            target_tokens=spec.max_model_len - output_tokens,
                            processor_kwargs=processor_kwargs,
                        )) for ordinal, handle in enumerate(handles.values())
                            for sequence in range(sequences)]
                        progress_tasks = fitted
                    prompts = [task.result() for task in fitted]
                    progress_tasks = []
                    if native_layout is not None:
                        advance("exact_cpu_count", len(prompts))
                        if exact_counter is None:
                            raise ManagedRuntimeError("capacity_native_exact_counter_required")
                        for messages in prompts:
                            total, image_tokens = await exact_counter.count_async(
                                messages, native_layout=native_layout,
                            )
                            if (total != spec.max_model_len - output_tokens or image_tokens != [
                                image.image_tokens for image in native_layout.images
                            ]):
                                raise ManagedRuntimeError(
                                    "capacity_native_exact_tokenization_mismatch"
                                )
                            progress["completed"] += 1
                advance("nvml_before_generation", len(pool.allocations))
                sampler.sample()
                before_generation = dict(sampler.peaks)
                _print_capacity_memory(memory_diagnostic("before_generation"))
                if multimodal:
                    # Track every image request, not just a pending replica wrapper:
                    # lagging scheduler metrics must not certify completed requests.
                    tasks = [asyncio.create_task(complete_image(
                        service_id, handle, prompts[ordinal * sequences + sequence],
                    )) for ordinal, (service_id, handle) in enumerate(handles.items())
                        for sequence in range(sequences)]
                else:
                    tasks = [
                        asyncio.create_task(complete(i, service_id, handle))
                        for i, (service_id, handle) in enumerate(handles.items())
                    ]
                advance("generation", len(tasks))
                progress_tasks = tasks
                full_pool_samples = 0
                while True:
                    all_inflight = all(not task.done() for task in tasks)
                    current = await metrics()
                    sampler.sample()
                    if any(preemptions for _, _, preemptions in current):
                        raise ManagedRuntimeError("capacity workload required scheduler preemption")
                    if (
                        all_inflight
                        and all(not task.done() for task in tasks)
                        and all(
                            running == sequences and waiting == 0 for running, waiting, _ in current
                        )
                    ):
                        full_pool_samples += 1
                    # Propagate a failed request immediately; cancel its siblings in finally.
                    for task in tasks:
                        if task.done():
                            task.result()
                    if all(task.done() for task in tasks):
                        break
                    await asyncio.sleep(sample_interval_seconds)
                if not full_pool_samples:
                    raise ManagedRuntimeError(
                        "capacity workload never observed full-pool concurrency"
                    )
                results = [task.result() for task in tasks]
                if multimodal:
                    results = [CapacityReplicaWorkload(
                        service_id=service_id, completed_sequences=sequences,
                        prompt_tokens_per_sequence=spec.max_model_len - output_tokens,
                        completion_tokens_per_sequence=output_tokens,
                        observed_running_sequences=sequences,
                        completed_image_sequences=sequences,
                        image_request_hashes=[
                            value for result in results[
                                ordinal * sequences : (ordinal + 1) * sequences
                            ] for value in result.image_request_hashes
                        ],
                    ) for ordinal, service_id in enumerate(handles)]
                return results, sampler.peaks, full_pool_samples
        except httpx.HTTPError:
            # httpx diagnostics contain endpoint URLs. Keep operator failures safe,
            # including transport failures obtained from a completed request task.
            raise ManagedRuntimeError("capacity_http_request_failed") from None
        except ExceptionGroup:
            # Tokenizer TaskGroup has already cancelled and joined every sibling.
            raise ManagedRuntimeError("capacity_multimodal_tokenization_failed") from None
        finally:
            failure = sys.exception()
            reporter.cancel()
            await asyncio.gather(reporter, return_exceptions=True)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            memory = memory_diagnostic("failed" if failure is not None else "completed")
            try:
                layout_id = native_layout.identity if native_layout else "text"
                _write_json(prepared.run_dir / f"capacity-memory-{layout_id}.json", memory)
            except OSError:
                if failure is None:
                    raise
                memory["artifact_write_failed"] = True
            _print_capacity_memory(memory)
            if failure is not None:
                # Persist only code locations, never exception text, locals or HTTP payloads.
                code = str(failure)
                if re.fullmatch(r"(?:capacity_|gpu_device_)[a-z0-9_]{1,119}", code) is None:
                    code = type(failure).__name__
                frames = []
                tb = failure.__traceback__
                while tb is not None:
                    module = tb.tb_frame.f_globals.get("__name__", "")
                    if module.startswith("pmmc."):
                        frames.append({"module": module, "function": tb.tb_frame.f_code.co_name,
                                       "line": tb.tb_lineno})
                    tb = tb.tb_next
                diagnostic = {"protocol": "step8-capacity-workload-failure-v1",
                              "substage": progress["phase"], "code": code,
                              "frames": frames, "generation_submitted": len(tasks),
                              "layout_hash": native_layout.identity if native_layout else None}
                try:
                    _write_json(prepared.run_dir / "capacity-workload-failure.json", diagnostic)
                except OSError:
                    diagnostic["artifact_write_failed"] = True
                print("STEP8_CAPACITY_WORKLOAD_FAILURE " + json.dumps(diagnostic, sort_keys=True),
                      flush=True)


def _calibrate_prepared_capacity(
    prepared: PreparedRuntime,
    stage: RuntimeStage,
    *,
    output_tokens: int = 256,
    timeout_seconds: float = 900.0,
    sample_interval_seconds: float = 0.1,
) -> tuple[Step8PoolCapacityReceipt, Path]:
    """Internal measurement under the public entry point's exclusive runtime lock."""
    _validate_bounds(output_tokens, timeout_seconds, sample_interval_seconds)
    pools = capacity_pools(prepared, stage, backend_version=_service_backend_version)
    if (
        len(pools) != 1
        or set(stage.service_ids) != {item.service_id for item in pools[0].services}
        or [a.service_id for a in stage.allocations] != stage.service_ids
    ):
        raise ManagedRuntimeError("capacity calibration requires exactly one isolated full pool")
    pool = pools[0]
    spec = pool.services[0]
    if spec.backend is not ManagedBackendKind.VLLM_CHAT:
        raise ManagedRuntimeError("capacity calibration currently requires vLLM chat")
    assert spec.max_num_seqs is not None and spec.max_model_len is not None
    if output_tokens >= spec.max_model_len:
        raise ManagedRuntimeError("capacity workload requires positive prompt and decode lengths")
    path = prepared.settings.runtime_calibrations_path / pool.filename
    # A failed fresh remeasurement must not leave an older passing receipt usable.
    if path.exists():
        _write_json(
            prepared.run_dir / "capacity-receipt-previous.json",
            {
                "receipt_text": path.read_text(encoding="utf-8"),
            },
        )
        path.unlink()
    started_at = datetime.now(UTC)

    def record_session(status: RuntimeStatus, error_type: str | None = None) -> None:
        session = _runtime_session(
            prepared,
            status,
            started_at=started_at,
            error_type=error_type,
        )
        session.metadata["scope"] = "step8_capacity_calibration_not_scientific_work"
        _write_json(prepared.run_dir / "runtime-session.json", session)
        _persist_session(prepared.settings, session)

    counter: NativeCounterProcess | None = None
    phase = "local_preflight"
    completed_layouts = 0
    try:
        # ActiveRuntimeStage persists service rows with a runtime-session FK.
        record_session(RuntimeStatus.STARTING)
        prompts: list = []
        native_prompts: dict[str, list] = {}
        bounds = (parse_image_workload(pool.identity["image_workload"])
                  if pool.identity["workload_kind"] == "multimodal" else None)
        if isinstance(bounds, NativeCapacityImageWorkload):
            entry = prepared.resolved_registry.get(spec.model_alias)
            counter = NativeCounterProcess(dict(
                alias=spec.model_alias,
                tokenizer_name=str(prepared.snapshot_paths[spec.model_alias]),
                tokenizer_revision=entry.tokenizer_revision or "",
                processor_name=str(prepared.snapshot_paths[spec.model_alias]),
                processor_revision=entry.processor_revision or "",
                project_root=prepared.settings.project_root, local_files_only=True,
                processor_kwargs={}, chat_template_kwargs={"enable_thinking": False},
            ), timeout_seconds=timeout_seconds)
            for ordinal, checked_layout in enumerate(bounds.layouts):
                print(f"STEP8_CAPACITY_LOCAL_PREFLIGHT state=prepare completed={ordinal} "
                      f"total={len(bounds.layouts)} gpu_started=0", flush=True)
                # Prepare compressible synthetic pixels before reserving the GPU pool.
                native_prompts[checked_layout.identity] = synthetic_native_messages(
                    checked_layout, spec.max_num_seqs * len(pool.services),
                )
                prompt = native_prompts[checked_layout.identity][0]
                total, tokens = counter.count_transport_profile(
                    prompt, native_layout=checked_layout,
                )
                if tokens != [image.image_tokens for image in checked_layout.images]:
                    raise ManagedRuntimeError("capacity_native_audited_grid_mismatch")
                if total > spec.max_model_len - output_tokens:
                    raise ManagedRuntimeError("capacity_native_layout_exceeds_context")
                print(f"STEP8_CAPACITY_LOCAL_PREFLIGHT completed={ordinal + 1} "
                      f"total={len(bounds.layouts)} prompt_tokens={total} gpu_started=0",
                      flush=True)
        elif pool.identity["workload_kind"] == "multimodal":
            prompts = synthetic_image_messages(
                CapacityImageWorkload.model_validate(pool.identity["image_workload"]),
                spec.max_num_seqs * len(pool.services),
            )
        else:
            prompts = _synthetic_prompts(
                prepared.snapshot_paths[spec.model_alias],
                spec.max_num_seqs * len(pool.services),
                spec.max_model_len - output_tokens,
            )
        print(
            "STEP8_CAPACITY_WORKLOAD "
            f"kind={pool.identity['workload_kind']} replicas={len(pool.services)} "
            f"sequences_per_replica={spec.max_num_seqs} total_tokens={spec.max_model_len} "
            f"images_per_sequence={bounds.images_per_sequence if bounds else 0}", flush=True,
        )
        phase = "pool_startup"
        with ActiveRuntimeStage(prepared, stage, require_calibration=False) as active:
            record_session(RuntimeStatus.RUNNING)
            scenarios = []
            layouts = bounds.layouts if isinstance(bounds, NativeCapacityImageWorkload) else [None]
            for ordinal, layout in enumerate(layouts):
                phase = "measure_layout"
                if layout is not None:
                    prompts = native_prompts.pop(layout.identity)
                print(f"STEP8_CAPACITY_SCENARIO state=start completed={ordinal} "
                      f"total={len(layouts)} images={len(layout.images) if layout else 0}",
                      flush=True)
                kwargs: dict[str, Any] = ({"native_layout": layout, "exact_counter": counter}
                                         if layout is not None else {})
                workloads, peaks, samples = asyncio.run(_measure_workload(
                    prepared, active, pool, prompts,
                    output_tokens=output_tokens,
                    timeout_seconds=timeout_seconds,
                    sample_interval_seconds=sample_interval_seconds,
                    **kwargs,
                ))
                if layout is not None:
                    scenarios.append(CapacityScenarioMeasurement(
                        layout_hash=layout.identity, peak_memory_bytes_by_device=peaks,
                        full_pool_concurrent_samples=samples, replica_workloads=workloads,
                    ))
                print(f"STEP8_CAPACITY_SCENARIO state=done completed={ordinal + 1} "
                      f"total={len(layouts)} concurrency_samples={samples}", flush=True)
                completed_layouts = ordinal + 1
            if scenarios:
                workloads = scenarios[0].replica_workloads
                peaks = {device: max(item.peak_memory_bytes_by_device[device] for item in scenarios)
                         for device in scenarios[0].peak_memory_bytes_by_device}
                samples = sum(item.full_pool_concurrent_samples for item in scenarios)
            phase = "pool_shutdown"
        # Publication is deliberately after successful shutdown / NVML GPU release.
        phase = "counter_shutdown"
        if counter is not None:
            counter.close()
        phase = "receipt_publication"
        device_mode = prepared.settings.gpu_memory_accounting == "allocated_device"
        lifecycle = (active.device_accounting.release_evidence
                     if device_mode and active.device_accounting is not None else None)
        if device_mode and lifecycle is None:
            raise ManagedRuntimeError("capacity_device_release_evidence_missing")
        identity = {**pool.identity, "measurement_source": (
            "nvml_allocated_device_total" if device_mode else "nvml_compute_process_memory")}
        receipt = Step8PoolCapacityReceipt(
            **identity,
            protocol=("step8-pool-capacity-v4" if device_mode else
                      "step8-pool-capacity-v3" if scenarios else "step8-pool-capacity-v2"),
            device_lifecycle=lifecycle,
            status="passed",
            measured_at=datetime.now(UTC),
            tested_concurrent_sequences_per_replica=spec.max_num_seqs,
            tested_total_tokens_per_sequence=spec.max_model_len,
            peak_memory_bytes_by_device=peaks,
            full_pool_concurrent_samples=samples,
            replica_workloads=workloads,
            scenario_measurements=scenarios,
        )
        current = capacity_pools(prepared, stage, backend_version=_service_backend_version)
        validate_capacity_receipt(current[0], receipt)
        _write_json(prepared.run_dir / "capacity-receipt.json", receipt)
        record_session(RuntimeStatus.COMPLETED)
        _write_json(path, receipt)
        return receipt, path
    except BaseException as exc:
        code = str(exc)
        if re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code) is None:
            code = type(exc).__name__
        print(f"STEP8_CAPACITY_FAILURE phase={phase} completed_layouts={completed_layouts} "
              f"code={code} receipt_published=false", flush=True)
        _write_json(
            prepared.run_dir / "capacity-failure.json",
            {
                "protocol": "step8-pool-capacity-v1",
                "status": "failed",
                "error_type": type(exc).__name__,
                "code": code, "phase": phase, "completed_layouts": completed_layouts,
                "measured_at": datetime.now(UTC).isoformat(),
            },
        )
        record_session(
            RuntimeStatus.CANCELLED if isinstance(exc, KeyboardInterrupt) else RuntimeStatus.FAILED,
            error_type=type(exc).__name__,
        )
        raise
    finally:
        if counter is not None:
            counter.close()
