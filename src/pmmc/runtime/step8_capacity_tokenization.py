"""Exact adapters used by the production capacity dispatch boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import deepcopy
from threading import Lock
from typing import Any

import httpx

from pmmc.compiler.ids import canonical_content_hash
from pmmc.models.provider_token_budget import ProviderConfiguredTokenCounter
from pmmc.models.tokenization import ProcessorTokenCounter
from pmmc.runtime.manager import ManagedRuntimeError
from pmmc.runtime.step8_capacity import parse_image_workload
from pmmc.runtime.step8_capacity_native import NativeCapacityImageWorkload
from pmmc.runtime.step8_capacity_requests import CapacityTokenization
from pmmc.runtime.step8_capacity_runtime import BoundCapacityRuntime, capacity_endpoint


class LocalCapacityTokenizer:
    def __init__(self, runtime: BoundCapacityRuntime, create_counter: Callable[[], Any]) -> None:
        self.contract = deepcopy(runtime.contract)
        self.create_counter = create_counter
        self.counter: ProcessorTokenCounter | None = None
        self._counter_lock = Lock()

    async def __call__(self, payload: dict[str, Any]) -> CapacityTokenization:
        return await asyncio.to_thread(self._tokenize, payload)

    def _tokenize(self, payload: dict[str, Any]) -> CapacityTokenization:
        # A cancelled await does not stop its thread; keep the lock in the worker
        # until construction and all mutable processor access have finished.
        with self._counter_lock:
            if self.counter is None:
                counter = self.create_counter()
                if isinstance(counter, ProviderConfiguredTokenCounter):
                    if counter.entry.resolved_hash != self.contract["model_registry_entry_hash"]:
                        raise ManagedRuntimeError("capacity_exact_local_registry_mismatch")
                    counter = counter.counter
                if not isinstance(counter, ProcessorTokenCounter) or counter.estimated:
                    raise ManagedRuntimeError("capacity_exact_local_processor_required")
                if counter.source_identity != self.contract.get("counter_identity"):
                    raise ManagedRuntimeError("capacity_exact_local_identity_mismatch")
                self.counter = counter
            counter = self.counter
            if (
                payload["model"] != self.contract["model"]
                or payload["add_generation_prompt"] is not True
                or canonical_content_hash(counter.processor_kwargs)
                != canonical_content_hash(payload["mm_processor_kwargs"])
                or canonical_content_hash(counter.chat_template_kwargs)
                != canonical_content_hash(payload["chat_template_kwargs"])
            ):
                raise ManagedRuntimeError("capacity_exact_local_processor_mismatch")
            native = "image_workload" in self.contract and isinstance(
                parse_image_workload(self.contract["image_workload"]), NativeCapacityImageWorkload,
            )
            if native:
                count, image_tokens = counter.count_transport_profile(payload["messages"])
            else:
                count, image_tokens = counter.count_transport_messages(payload["messages"]), []
            return CapacityTokenization(
                kind="local_exact", input_sha256=canonical_content_hash(payload),
                model_registry_entry_hash=self.contract["model_registry_entry_hash"],
                processor_kwargs=counter.processor_kwargs,
                chat_template_kwargs=counter.chat_template_kwargs,
                prompt_tokens=count, image_tokens=image_tokens,
            )


class ServiceCapacityTokenizer:
    def __init__(
        self, runtime: BoundCapacityRuntime, endpoint: str, *, api_key: str, timeout: float,
    ) -> None:
        self.endpoint = capacity_endpoint(endpoint)
        if self.endpoint not in {row[2] for row in runtime.bindings}:
            raise ManagedRuntimeError("capacity_managed_endpoint_mismatch")
        self.contract = deepcopy(runtime.contract)
        self.backend_version = runtime.backend_version
        self.api_key = api_key
        self.timeout = timeout

    async def __call__(self, payload: dict[str, Any]) -> CapacityTokenization:
        if (
            payload["model"] != self.contract["model"]
            or payload["add_generation_prompt"] is not True
            or canonical_content_hash(payload["mm_processor_kwargs"])
            != canonical_content_hash(self.contract["processor_kwargs"])
            or canonical_content_hash(payload["chat_template_kwargs"])
            != canonical_content_hash(self.contract["chat_template_kwargs"])
        ):
            raise ManagedRuntimeError("capacity_exact_service_processor_mismatch")
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, trust_env=False, follow_redirects=False,
                headers={"Authorization": f"Bearer {self.api_key}"},
            ) as client:
                version = await client.get(self.endpoint + "/version")
                version.raise_for_status()
                if version.json().get("version") != self.backend_version:
                    raise ValueError("backend identity mismatch")
                models = await client.get(self.endpoint + "/v1/models")
                models.raise_for_status()
                if self.contract["actual_model_id"] not in {
                    row["id"] for row in models.json()["data"]
                }:
                    raise ValueError("served model mismatch")
                response = await client.post(self.endpoint + "/tokenize", json=payload)
                response.raise_for_status()
                count = response.json()["count"]
                if type(count) is not int or count <= 0:
                    raise ValueError("invalid exact count")
        except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError):
            raise ManagedRuntimeError("capacity_service_tokenization_failed") from None
        return CapacityTokenization(
            kind="service_exact", input_sha256=canonical_content_hash(payload),
            model_registry_entry_hash=self.contract["model_registry_entry_hash"],
            processor_kwargs=self.contract["processor_kwargs"],
            chat_template_kwargs=self.contract["chat_template_kwargs"],
            prompt_tokens=count,
        )
