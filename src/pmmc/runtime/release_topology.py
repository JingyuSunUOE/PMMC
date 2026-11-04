"""Freeze the release worker's physical layout without changing the science contract."""

import hashlib
import json
import os
from pathlib import Path

from pmmc.experiments.execution import ExperimentExecutionConfig, load_execution_config
from pmmc.experiments.step8_artifacts import durable_write, json_bytes
from pmmc.runtime.config import RuntimeProfile, load_runtime_profile

RUNTIME = Path("configs/release/runtime.yaml")
EXECUTION = Path("configs/release/execution.yaml")


def calibration_stages(profile, aliases):
    """Resolve only services for the verified role/alias inventory; never guess a fallback."""
    result, covered = [], set()
    for service in profile.services:
        selected = set(aliases).intersection(service.all_served_aliases)
        if not selected:
            continue
        if covered.intersection(selected):
            raise ValueError("ambiguous release calibration service")
        isolated = next(
            (s for s in service.stages if profile.services_for_stage(s) == [service]), None
        )
        if isolated is None:
            raise ValueError("release calibration service has no isolated stage")
        result.append((isolated, service))
        covered.update(selected)
    if covered != set(aliases):
        raise ValueError("release model alias has no runtime service")
    return result


def _scientific_profile(profile):
    raw = profile.model_dump(mode="json", exclude={"profile_name", "hardware_class"})
    for spec in raw["services"]:
        spec.pop("tensor_parallel_candidates")
        for key in ("frozen_replica_topology", "calibrated_placement_options"):
            spec["metadata"].pop(key, None)
    return raw


def derive(root, spec):
    """Return a proposed layout. Only native hardware calibration can authorize it."""
    count, tp, workers = (spec[k] for k in ("gpus", "answer_tp", "workers"))
    if (
        type(count) is not int
        or not 0 <= count <= 64
        or type(tp) is not int
        or tp not in (1, 2, 4, 8)
        or count % tp
        or type(workers) is not int
        or not 1 <= workers <= 4096
    ):
        raise ValueError("invalid release single-node topology")
    source = load_runtime_profile(root / spec["source_profile"])
    if _scientific_profile(source) != _scientific_profile(load_runtime_profile(root / RUNTIME)):
        raise ValueError("release runtime override changes the scientific workload")
    runtime = source.model_dump(mode="json")
    runtime["metadata"]["release_visible_gpus"] = count
    if count == 0:
        runtime["metadata"].update(
            release_planning_only=True, calibration_status="unmeasured_no_gpu_allocation"
        )
    replicas = {}
    for service in runtime["services"]:
        if count == 0:
            # No fabricated replica layout or capacity receipt. The template's
            # models, processor limits, precision and gated services stay intact.
            for key in ("frozen_replica_topology", "calibrated_placement_options"):
                service["metadata"].pop(key, None)
            continue
        parallel = tp if service["metadata"].get("requires_capacity_receipt") is True else 1
        placement = {"tensor_parallel_size": parallel, "replica_count": count // parallel}
        service["tensor_parallel_candidates"] = [parallel]
        service["metadata"].update(
            frozen_replica_topology=placement, calibrated_placement_options=[placement]
        )
        replicas[service["role"]] = placement["replica_count"]
    if count:
        replicas["judge"] = replicas["compiler"]
    execution = load_execution_config(root / EXECUTION).model_dump(mode="json")
    execution["replicas"].update(replicas)
    execution["unit_preparation"]["workers"] = min(workers, 64)
    return RuntimeProfile.model_validate(runtime), ExperimentExecutionConfig.model_validate(
        execution
    )


def materialize(settings):
    """Publish identical, content-addressed bytes from prepare and calibration pods."""
    raw = os.getenv("PMMC_RELEASE_TOPOLOGY")
    if raw is None:
        return {}
    root = settings.project_root.resolve()
    runtime, execution = derive(root, json.loads(raw))
    result = {}
    for key, model in (("runtime_profile", runtime), ("execution_profile", execution)):
        data = json_bytes(model.model_dump(mode="json"))
        path = (
            settings.outputs_path
            / "step8/topologies"
            / f"{key}-{hashlib.sha256(data).hexdigest()}.json"
        )
        if path.resolve() != path or not path.is_relative_to(root):
            raise ValueError("invalid release profile destination")
        durable_write(path, data, immutable=True)
        result[key] = path.relative_to(root)
    return result
