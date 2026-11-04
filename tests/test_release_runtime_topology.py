"""Offline binding of release sizing to native planning and capacity contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pmmc.experiments.execution import load_execution_config
from pmmc.runtime.config import load_runtime_profile
from pmmc.runtime.planner import RuntimePlanningError, build_runtime_plan
from pmmc.runtime.release_topology import RUNTIME, calibration_stages, derive, materialize
from pmmc.runtime.step8_capacity import capacity_image_workload

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("count,tp", [(1, 1), (2, 2), (2, 1), (4, 2)])
def test_single_node_native_profiles_preserve_workload(tmp_path, monkeypatch, count, tp):
    source = load_runtime_profile(ROOT / RUNTIME)
    settings = SimpleNamespace(project_root=ROOT, outputs_path=tmp_path.resolve())
    spec = {"gpus": count, "answer_tp": tp, "workers": 2, "source_profile": str(ROOT / RUNTIME)}
    monkeypatch.setenv("PMMC_RELEASE_TOPOLOGY", json.dumps(spec))
    # The synthetic settings root contains the outputs and the portable template.
    from shutil import copytree

    root = tmp_path.resolve()
    copytree(ROOT / "configs/release", root / "configs/release")
    settings.project_root, settings.outputs_path = root, root / "outputs"
    files = materialize(settings)
    assert materialize(settings) == files
    profile = load_runtime_profile(root / files["runtime_profile"])
    execution = load_execution_config(root / files["execution_profile"])
    assert execution.unit_preparation.workers == 2
    assert execution.replicas.compiler == count
    assert execution.replicas.answer == count // tp
    assert profile.metadata["release_visible_gpus"] == count
    assert {s.service_id for s in profile.services} == {s.service_id for s in source.services}
    (gated,) = profile.services_for_stage("answer_27b")
    (reference,) = source.services_for_stage("answer_27b")
    assert gated.tensor_parallel_candidates == [tp]
    assert gated.metadata["frozen_replica_topology"] == {
        "tensor_parallel_size": tp,
        "replica_count": count // tp,
    }
    assert gated.metadata["requires_capacity_receipt"] is True
    assert (gated.dtype, gated.extra_args, gated.max_model_len, gated.max_num_seqs) == (
        reference.dtype,
        reference.extra_args,
        reference.max_model_len,
        reference.max_num_seqs,
    )
    assert capacity_image_workload(gated, multimodal=True) == capacity_image_workload(
        reference, multimodal=True
    )
    with pytest.raises(RuntimePlanningError, match="actual GPU inventory"):
        build_runtime_plan(profile, None, None, {}, SimpleNamespace(devices=[None] * (count + 1)))


def test_rejects_weaker_runtime_override(tmp_path):
    raw = yaml.safe_load((ROOT / RUNTIME).read_text())
    raw["services"][1]["max_model_len"] //= 2
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="scientific workload"):
        derive(ROOT, {"gpus": 2, "answer_tp": 1, "workers": 2, "source_profile": str(path)})


def test_scope_controls_real_calibration_stages():
    profile = load_runtime_profile(ROOT / RUNTIME)
    stages = calibration_stages(profile, {"compiler-qwen3.5-9b", "answer-qwen3.5-9b"})
    assert [stage for stage, _ in stages] == ["compiler"]
    assert calibration_stages(profile, set()) == []
    stages = calibration_stages(profile, {"answer-qwen3.5-27b"})
    assert [stage for stage, _ in stages] == ["answer_27b"]
    assert stages[0][1].metadata["requires_capacity_receipt"]
    with pytest.raises(ValueError, match="no runtime service"):
        calibration_stages(profile, {"unknown-model"})
