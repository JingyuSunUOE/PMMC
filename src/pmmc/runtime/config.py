"""Strict managed-runtime profile loading and hashing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pydantic import Field, model_validator

from pmmc.schemas import BaseSchema, ManagedServiceSpec, ModelRole
from pmmc.schemas.base import NonEmptyStr, PositiveInt, ensure_unique_strings


class RuntimeProfile(BaseSchema):
    profile_version: NonEmptyStr = "pmmc-runtime-profile-v1"
    profile_name: NonEmptyStr
    hardware_class: NonEmptyStr
    model_lock_path: Path
    weight_safety_factor: float = Field(default=1.10, ge=1.0, le=2.0)
    gpu_reserve_fraction: float = Field(default=0.10, ge=0.0, le=0.5)
    gpu_reserve_min_gib: float = Field(default=4.0, ge=0.0)
    service_overhead_gib: float = Field(default=4.0, ge=0.0)
    gpu_memory_utilization_max: float = Field(default=0.90, gt=0.0, le=1.0)
    telemetry_interval_seconds: float = Field(default=2.0, gt=0.0)
    shutdown_timeout_seconds: PositiveInt = 60
    require_calibration_in_paper: bool = True
    enable_litellm: bool = True
    litellm_port: PositiveInt = Field(default=4000, le=65535)
    stage_order: list[NonEmptyStr]
    services: list[ManagedServiceSpec]
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_profile(self) -> RuntimeProfile:
        if not self.services:
            raise ValueError("runtime profile requires at least one managed service")
        ensure_unique_strings([item.service_id for item in self.services], "services.service_id")
        ensure_unique_strings(self.stage_order, "stage_order")
        if not self.stage_order:
            raise ValueError("runtime profile requires at least one stage")
        known_stages = {stage for service in self.services for stage in service.stages}
        if set(self.stage_order) != known_stages:
            raise ValueError("stage_order must cover exactly the stages declared by services")
        ports = [item.port for item in self.services if item.port is not None]
        if len(ports) != len(set(ports)) or self.litellm_port in ports:
            raise ValueError("managed service and LiteLLM ports must be unique")
        if any(
            item.gpu_memory_utilization > self.gpu_memory_utilization_max for item in self.services
        ):
            raise ValueError("service gpu_memory_utilization exceeds the profile maximum")
        forbidden_args = {
            "--api-key",
            "--data-parallel-size",
            "--device-ids",
            "--disable-uvicorn-access-log",
            "--distributed-executor-backend",
            "--dtype",
            "--generation-config",
            "--gpu-memory-utilization",
            "--host",
            "--max-model-len",
            "--max-num-batched-tokens",
            "--max-num-seqs",
            "--model",
            "--pipeline-parallel-size",
            "--port",
            "--quantization",
            "--runner",
            "--served-model-name",
            "--tensor-parallel-size",
        }
        for service in self.services:
            if service.port is None:
                raise ValueError(f"managed service requires an explicit port: {service.service_id}")
            for argument in service.extra_args:
                option = argument.split("=", 1)[0]
                if option in forbidden_args:
                    raise ValueError(f"service extra_args cannot override harness option: {option}")
        return self

    @property
    def resolved_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def services_for_stage(self, stage_id: str) -> list[ManagedServiceSpec]:
        if stage_id not in self.stage_order:
            raise KeyError(f"runtime profile does not define stage: {stage_id}")
        return [service for service in self.services if stage_id in service.stages]


def load_runtime_profile(path: Path) -> RuntimeProfile:
    import yaml  # type: ignore[import-untyped]

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid runtime profile YAML: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("runtime profile root must be a mapping")
    return RuntimeProfile.model_validate(value)


def bind_runtime_profile_models(
    profile: RuntimeProfile,
    aliases_by_role: dict[ModelRole, str],
) -> RuntimeProfile:
    """Bind role templates to experiment-selected aliases deterministically."""

    role_owners: dict[ModelRole, str] = {}
    for service in profile.services:
        claimed_roles = [
            service.role,
            *[ModelRole(str(value)) for value in service.metadata.get("served_roles", [])],
        ]
        for role in claimed_roles:
            prior = role_owners.setdefault(role, service.service_id)
            if prior != service.service_id:
                raise ValueError(
                    f"runtime profile maps model role {role.value!r} to multiple services"
                )
    unknown_roles = set(aliases_by_role) - set(role_owners)
    if unknown_roles:
        unknown_role = sorted(item.value for item in unknown_roles)[0]
        raise ValueError(f"runtime profile does not define model role: {unknown_role}")
    services: list[ManagedServiceSpec] = []
    for service in profile.services:
        alias = aliases_by_role.get(service.role, service.model_alias)
        served_roles = [ModelRole(str(value)) for value in service.metadata.get("served_roles", [])]
        served_aliases = [
            aliases_by_role[role]
            for role in served_roles
            if role in aliases_by_role and aliases_by_role[role] != alias
        ]
        metadata = dict(service.metadata)
        if served_roles or "served_aliases" in metadata:
            metadata["served_aliases"] = list(dict.fromkeys(served_aliases))
        if alias == service.model_alias and metadata == service.metadata:
            services.append(service)
            continue
        identity = "|".join([alias, *metadata.get("served_aliases", [])])
        suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
        services.append(
            service.model_copy(
                update={
                    "service_id": f"{service.role.value.replace('_', '-')}-{suffix}",
                    "model_alias": alias,
                    "metadata": metadata,
                }
            )
        )
    return profile.model_copy(update={"services": services})


def project_runtime_profile_roles(
    profile: RuntimeProfile,
    *,
    aliases_by_role: dict[ModelRole, str],
    stages_by_role: dict[ModelRole, str],
    model_lock_path: Path,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> RuntimeProfile:
    """Select logical roles while preserving shared physical services.

    A service may own its primary role plus roles declared in ``served_roles``.
    Projecting only by ``service.role`` drops stages and aliases from shared chat
    services, producing a profile whose stage order cannot be validated.
    """

    if set(aliases_by_role) != set(stages_by_role):
        raise ValueError("runtime profile aliases and stages must cover identical roles")
    bound = bind_runtime_profile_models(profile, aliases_by_role)
    services: list[ManagedServiceSpec] = []
    for service in bound.services:
        declared_roles = [
            service.role,
            *[ModelRole(str(value)) for value in service.metadata.get("served_roles", [])],
        ]
        selected_roles = [role for role in declared_roles if role in aliases_by_role]
        if not selected_roles:
            continue
        update: dict[str, object] = {
            "stages": list(dict.fromkeys(stages_by_role[role] for role in selected_roles))
        }
        if tensor_parallel_size is not None:
            update["tensor_parallel_candidates"] = [tensor_parallel_size]
        if gpu_memory_utilization is not None:
            update["gpu_memory_utilization"] = gpu_memory_utilization
        services.append(service.model_copy(update=update))

    if not services:
        raise ValueError("runtime role projection selected no managed services")
    selected_stages = {stage for service in services for stage in service.stages}
    stage_order = [stage for stage in bound.stage_order if stage in selected_stages]
    missing_stages = selected_stages - set(stage_order)
    if missing_stages:
        raise ValueError(
            "runtime role projection introduced stages absent from the source profile: "
            + ",".join(sorted(missing_stages))
        )
    projected = bound.model_copy(
        update={
            "model_lock_path": model_lock_path,
            "stage_order": stage_order,
            "services": services,
        }
    )
    return RuntimeProfile.model_validate(projected.model_dump(mode="python"))


def runtime_profile_safe_summary(profile: RuntimeProfile) -> dict[str, object]:
    """Return a path- and payload-free profile summary for launch diagnostics."""

    return {
        "profile_hash": profile.resolved_hash,
        "service_count": len(profile.services),
        "shared_service_count": sum(
            bool(service.metadata.get("served_roles")) for service in profile.services
        ),
        "served_alias_count": sum(len(service.all_served_aliases) for service in profile.services),
        "stage_count": len(profile.stage_order),
        "stage_order": list(profile.stage_order),
    }


def write_runtime_profile(profile: RuntimeProfile, path: Path) -> None:
    """Atomically persist a resolved, secret-free runtime profile."""

    import yaml  # type: ignore[import-untyped]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(
            profile.model_dump(mode="json", exclude_none=True),
            allow_unicode=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
