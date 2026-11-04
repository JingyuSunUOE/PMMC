"""Fail-closed, development-only transfer of pre-online Step 7B checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.experiments.checkpoint import (
    CheckpointIdentityMismatch,
    ExperimentCheckpointManifest,
    ExperimentCheckpointService,
    _CheckpointCandidateMetadata,
    checkpoint_revision_is_restore_compatible,
)
from pmmc.experiments.stage_components import (
    source_fingerprint,
    source_fingerprints_compatible,
    stage_family,
)
from pmmc.schemas import (
    EXECUTION_CACHE_SCHEMA_VERSION,
    StageComponentManifest,
    StageComponentStatus,
)
from pmmc.schemas.runtime import ModelLockManifest

if TYPE_CHECKING:
    from pmmc.experiments.launcher import ResearchExperimentLauncher

IMPORT_PROTOCOL = "step7b-pre-online-import-v1"
COMPONENT_PROVENANCE_POLICY = "sealed-component-origins-v1"
MODEL_LOCK_EQUIVALENCE_POLICY = "lock-envelope-provenance-only-v2"
_LEGACY_MODEL_LOCK_POLICY = "lock-derivation-label-only-v1"
_MODEL_LOCK_PROVENANCE_FIELDS = ["created_at", "metadata.derived_from_lock"]
ALLOWED_INPUT_CHANGES = frozenset({"qbank_reranker_response_schema", "qbank_reranker_prompt"})
# Only this audited answer-only transition may import pre-online facts. Config,
# prompt, retry identity and byte counts must all match; arbitrary edits fail closed.
ANSWER_INPUT_TRANSITION = {
    "query_executor_config": (
        ("6c8567a4417b2852887b06814ba8f801a85e2488f0f756729b60ceafdd0bfc32", 1154),
        ("f64fb4e2da11429a45c10747d7caa8a54e5ab7007966336a5f79a27159f9ab0c", 1305),
    ),
    "query_executor_answer_prompt": (
        ("6ff7b09e90e634b270d2cedc16d987571f6c2c5bbe4954e991d210dc95bffdff", 563),
        ("18f728ba9398e5a122a5aa6b05d9875fdc4947d28d6aa1dd1330c899875ce756", 1221),
    ),
}
UPSTREAM_FAMILIES = frozenset(
    {
        "unit_prepare",
        "compiler_program",
        "program_text",
        "program_multimodal",
        "dry_run",
        "doubter",
        "verified_qbank",
        "qbank_coverage",
        "step7_prefix",
    }
)
REPLAY_FAMILIES = frozenset(
    {
        "preflight",
        "model_prepare",
        "infrastructure",
        "model_calibration",
        "benchmark",
        "snapshot_import",
        "partition",
        "pilot_sample",
        "corpus",
        "suite",
        "export",
    }
)
INVALIDATED_STAGES = [
    "query_match",
    "query_execute",
    "answer",
    "qbank_semantic",
    "evaluation_deterministic",
    "evaluation_judge",
    "evaluation_analyze",
    "export",
    "automatic_quality",
    "review_bundle",
    "export_verify",
]


class UpstreamImportError(ValueError):
    """The message is a fixed safe error code, never private input content."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.error_code = code


class ImportedProgramAuditError(UpstreamImportError):
    """A compiler audit could not be bound to a sealed original component."""

    def __init__(self, code: str, dataset: str) -> None:
        super().__init__(code)
        self.dataset = dataset if dataset in {"memlens", "memgallery"} else "unknown"

    def safe_diagnostics(self) -> dict[str, object]:
        return {
            "error_code": self.error_code,
            "dataset": self.dataset,
            "preflight_phase": "imported_program_audit",
            "model_calls": 0,
        }


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise UpstreamImportError(code)


def _model_lock_payload(
    path: Path, *, policy: str = MODEL_LOCK_EQUIVALENCE_POLICY,
) -> tuple[str, dict[str, Any]]:
    _require(
        policy in {MODEL_LOCK_EQUIVALENCE_POLICY, _LEGACY_MODEL_LOCK_POLICY},
        "upstream_import_lock_policy_invalid",
    )
    try:
        raw = path.read_bytes()
        payload = ModelLockManifest.model_validate_json(raw).model_dump(mode="json")
    except (OSError, ValueError) as exc:
        raise UpstreamImportError("upstream_model_lock_unreadable_or_invalid") from exc
    # The envelope's creation time and derivation label describe the lock file,
    # not the model. Do not strip similarly named fields inside model/file metadata.
    payload["metadata"].pop("derived_from_lock", None)
    if policy == MODEL_LOCK_EQUIVALENCE_POLICY:
        payload.pop("created_at")
    return hashlib.sha256(raw).hexdigest(), payload


def verify_model_lock_equivalence(source_path: Path, target_path: Path) -> dict[str, Any]:
    source_hash, source_payload = _model_lock_payload(source_path)
    target_hash, target_payload = _model_lock_payload(target_path)
    if source_payload != target_payload:
        fields = sorted(key for key in source_payload if source_payload[key] != target_payload[key])
        print(
            "STEP7B_UPSTREAM_IMPORT phase=model_lock state=failed "
            f"mismatch_fields={','.join(fields)} "
            "error_code=upstream_model_lock_contract_changed model_calls=0",
            flush=True,
        )
    _require(source_payload == target_payload, "upstream_model_lock_contract_changed")
    return {
        "protocol": MODEL_LOCK_EQUIVALENCE_POLICY,
        "excluded_fields": list(_MODEL_LOCK_PROVENANCE_FIELDS),
        "source_sha256": source_hash,
        "target_sha256": target_hash,
        "contract_hash": canonical_content_hash(source_payload),
    }


def _selection_failure(exc: Exception, phase: str) -> tuple[str, str, str]:
    if isinstance(exc, CheckpointIdentityMismatch):
        return "upstream_checkpoint_identity_mismatch", phase, ",".join(exc.mismatch_keys)
    if isinstance(exc, UpstreamImportError):
        return exc.error_code, phase, "none"
    # Never emit exception text: parser/OS errors can contain prompts and paths.
    return "upstream_checkpoint_invalid", phase, "none"


def validate_import_id(value: str) -> None:
    _require(bool(re.fullmatch(r"launch-[a-f0-9]{24}", value)), "upstream_launch_id_invalid")


def _identity(inputs: dict[str, Any]) -> dict[str, Any]:
    _require(
        isinstance(inputs, dict) and all(isinstance(v, dict) for v in inputs.values()),
        "upstream_input_shape_invalid",
    )
    return {key: value for key, value in inputs.items() if value.get("identity", True) is not False}


def validate_input_delta(
    *,
    source_inputs: dict[str, Any],
    current_inputs: dict[str, Any],
    config_hash: str,
    source_config_hash: str,
    source_launch_id: str,
    target_launch_id: str,
) -> list[str]:
    old, new = _identity(source_inputs), _identity(current_inputs)
    source_hash = canonical_content_hash(
        {"launcher_config_hash": config_hash, "source_inputs": old}
    )
    target_hash = canonical_content_hash(
        {"launcher_config_hash": config_hash, "source_inputs": new}
    )
    _require(source_hash == source_config_hash, "upstream_config_hash_invalid")
    _require(source_launch_id == "launch-" + source_hash[:24], "upstream_launch_hash_invalid")
    _require(target_launch_id == "launch-" + target_hash[:24], "target_launch_hash_invalid")
    changed = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
    answer_keys = set(changed) & ANSWER_INPUT_TRANSITION.keys()
    if answer_keys:
        _require(answer_keys == ANSWER_INPUT_TRANSITION.keys(), "answer_import_partial_identity")
        for key, (previous, current) in ANSWER_INPUT_TRANSITION.items():
            _require(
                old.get(key) == {"sha256": previous[0], "size_bytes": previous[1], "identity": True}
                and new.get(key) == {
                    "sha256": current[0], "size_bytes": current[1], "identity": True,
                },
                "answer_import_unaudited_identity_change",
            )
    _require(
        bool(changed) and set(changed) <= ALLOWED_INPUT_CHANGES | answer_keys,
        "upstream_scientific_inputs_incompatible",
    )
    _require(old.keys() == new.keys(), "upstream_input_inventory_changed")
    return changed


def validate_component(component: StageComponentManifest, config_hash: str, selection: Any) -> None:
    identity = {
        key: getattr(component, key)
        for key in (
            "launch_id",
            "run_id",
            "stage",
            "dataset",
            "round_id",
            "scientific_input_hash",
            "source_fingerprint",
            "upstream_output_hash",
            "output_hash",
            "status",
            "image_digest",
            "alembic_revision",
        )
    }
    identity["execution_attempt_id"] = component.metadata.get("execution_attempt_id")
    _require(
        component.component_id == stable_compiler_id("stage-component", identity),
        "upstream_component_hash_invalid",
    )
    expected = canonical_content_hash(
        {
            "effective_config_hash": config_hash,
            "run_id": component.run_id,
            "selection_hash": selection.get("selection_hash"),
            "stage": component.stage,
            "round_id": component.round_id,
        }
    )
    _require(component.scientific_input_hash == expected, "upstream_component_input_invalid")
    _require(
        component.status is StageComponentStatus.READY
        and component.row_counts.get("completed") == 1
        and component.row_counts.get("failed") == 0,
        "upstream_component_not_ready",
    )


def _receipt_inventory(
    receipt: dict[str, Any], components: list[StageComponentManifest],
) -> list[StageComponentManifest]:
    payload = {key: value for key, value in receipt.items() if key != "content_hash"}
    _require(receipt.get("content_hash") == canonical_content_hash(payload),
             "upstream_import_receipt_hash_invalid")
    _require(receipt.get("protocol") == IMPORT_PROTOCOL, "upstream_import_receipt_identity_invalid")
    for key in ("source_launch_id", "target_launch_id"):
        _require(isinstance(receipt.get(key), str), "upstream_import_receipt_identity_invalid")
        validate_import_id(receipt[key])
    _require(receipt["source_launch_id"] != receipt["target_launch_id"],
             "upstream_component_origin_cycle")
    ids = receipt.get("component_ids")
    _require(isinstance(ids, list) and bool(ids) and all(isinstance(i, str) for i in ids),
             "upstream_import_inventory_invalid")
    assert isinstance(ids, list)
    selected = [item for item in components if item.component_id in ids]
    _require(
        len(selected) == len(ids) == len(set(ids))
        and canonical_content_hash([item.model_dump(mode="json") for item in selected])
        == receipt.get("component_inventory_hash"),
        "upstream_import_inventory_invalid",
    )
    return selected


def _receipt_component_configs(
    receipt: dict[str, Any], components: list[StageComponentManifest],
) -> dict[str, str]:
    """Use only origins attested by the sealed receipt, never arbitrary foreign rows."""
    configs = receipt.get("component_config_hashes")
    if configs is None:
        # Legacy v1 receipts contain one origin, before chained imports were supported.
        _require("component_provenance_policy" not in receipt,
                 "upstream_component_provenance_invalid")
        _require(all(c.launch_id == receipt.get("source_launch_id") for c in components),
                 "upstream_component_origin_unproven")
        configs = {c.component_id: receipt.get("source_config_hash") for c in components}
    else:
        _require(receipt.get("component_provenance_policy") == COMPONENT_PROVENANCE_POLICY,
                 "upstream_component_provenance_invalid")
    _require(isinstance(configs, dict) and set(configs) == {c.component_id for c in components},
             "upstream_component_provenance_invalid")
    for component in components:
        validate_import_id(component.launch_id)
        config_hash = configs[component.component_id]
        _require(isinstance(config_hash, str) and bool(re.fullmatch(r"[a-f0-9]{64}", config_hash)),
                 "upstream_component_origin_config_invalid")
        _require(component.launch_id != receipt.get("target_launch_id"),
                 "upstream_component_origin_cycle")
        if component.launch_id == receipt.get("source_launch_id"):
            _require(config_hash == receipt.get("source_config_hash"),
                     "upstream_component_origin_config_invalid")
    return dict(configs)


def _verify_imported_prefix(state: Any, receipt: dict[str, Any]) -> None:
    prefix = dict(state.metadata.get("step7b_prefix", {}))
    digest = prefix.pop("content_hash", None)
    _require(
        digest == canonical_content_hash(prefix)
        and prefix.get("upstream_import_hash") == receipt["content_hash"]
        and prefix.get("launch_id") == state.launch_id,
        "upstream_import_prefix_invalid",
    )


def _component_selection(state: Any, component: StageComponentManifest) -> dict[str, Any]:
    if stage_family(component.stage) == "unit_prepare" and component.run_id is None:
        return {}  # Preparation is sealed before assigning a compiler run to dataset state.
    return dict(state.dataset_artifacts.get(component.dataset, {}))


def _inherited_component_configs(
    state: Any, manifest: ExperimentCheckpointManifest | _CheckpointCandidateMetadata,
    components: list[StageComponentManifest],
) -> dict[str, str]:
    receipt = state.metadata.get("upstream_import")
    if receipt is None:
        return {}
    _require(isinstance(receipt, dict), "upstream_import_receipt_missing")
    selected = _receipt_inventory(receipt, components)
    _require(
        receipt.get("target_launch_id") == state.launch_id == manifest.launch_id
        and receipt.get("target_config_hash") == manifest.identity.get("launch_config_hash")
        and receipt.get("target_source_inputs_hash") == manifest.identity.get("source_inputs_hash")
        and receipt.get("model_lock_sha256") == manifest.identity.get("model_lock_sha256")
        and receipt.get("snapshots") == manifest.identity.get("snapshots")
        and state.metadata.get("upstream_import_source") == receipt.get("source_launch_id"),
        "upstream_import_receipt_identity_invalid",
    )
    configs = _receipt_component_configs(receipt, selected)
    _verify_imported_prefix(state, receipt)
    for component in selected:
        _require(component.created_at <= datetime.fromisoformat(manifest.created_at),
                 "upstream_component_after_checkpoint")
        validate_component(component, configs[component.component_id],
                           _component_selection(state, component))
    return configs


def select_components(
    *,
    state: Any,
    manifest: ExperimentCheckpointManifest | _CheckpointCandidateMetadata,
    components: list[StageComponentManifest],
    project_root: Path,
) -> list[StageComponentManifest]:
    completed = set(state.completed_stage_ids)
    datasets = {"memlens", "memgallery"}
    _require(set(state.dataset_artifacts) == datasets, "upstream_dataset_inventory_invalid")
    required = {
        "step7_prefix",
        "compiler_program:model-stage",
        "program_text:model-stage",
        "program_multimodal:model-stage",
        "doubter:model-stage",
    }
    required |= {
        f"{stage}:{dataset}"
        for dataset in datasets
        for stage in ("unit_prepare", "dry_run", "verified_qbank")
    }
    _require(required <= completed, "upstream_checkpoint_before_verified_qbank")
    _require(not manifest.milestone.startswith("periodic:"), "upstream_periodic_not_sealed")
    _require(
        all(stage_family(stage) in UPSTREAM_FAMILIES | REPLAY_FAMILIES for stage in completed),
        "upstream_checkpoint_contains_online_stages",
    )
    _require(
        stage_family(manifest.milestone) in UPSTREAM_FAMILIES | {"export"}
        or (manifest.milestone == "upstream_import"
            and isinstance(state.metadata.get("upstream_import"), dict)),
        "upstream_checkpoint_boundary_invalid",
    )
    _require(state.launch_id == manifest.launch_id, "upstream_state_launch_invalid")
    _require(manifest.launch_state_sha256 is not None, "upstream_checkpoint_state_unsealed")
    snapshots = {name: row.get("snapshot_id") for name, row in state.dataset_artifacts.items()}
    _require(
        all(snapshots.values()) and snapshots == manifest.identity.get("snapshots"),
        "upstream_snapshot_identity_invalid",
    )
    for row in state.dataset_artifacts.values():
        _require(
            all(
                row.get(key)
                for key in (
                    "compiler_run_id",
                    "coverage_run_id",
                    "unit_preparation_dir",
                    "selection_hash",
                    "program_pilot_audit_hash",
                    "fallback_run_id",
                    "fallback_corpus_path",
                )
            ),
            "upstream_dataset_facts_incomplete",
        )
        _require(row.get("r0_gate_passed") is True, "upstream_r0_not_passed")
        _require(
            not any(
                row.get(key)
                for key in (
                    "query_executor_run_id",
                    "query_run_id",
                    "semantic_run_id",
                    "evaluation_run_id",
                )
            ),
            "upstream_dataset_contains_online_facts",
        )
    inherited = _inherited_component_configs(state, manifest, components)
    selected: list[StageComponentManifest] = []
    for stage in sorted(completed):
        if stage_family(stage) not in UPSTREAM_FAMILIES:
            continue
        scoped: dict[tuple[str | None, int | None], StageComponentManifest] = {}
        for component in components:
            if (
                component.stage == stage
                and component.status is StageComponentStatus.READY
                and component.created_at <= datetime.fromisoformat(manifest.created_at)
            ):
                scope = (component.dataset, component.round_id)
                if scope not in scoped or scoped[scope].created_at < component.created_at:
                    scoped[scope] = component
        _require(bool(scoped), "upstream_stage_component_missing")
        expected_datasets: set[str | None] = (
            {stage.split(":", 1)[1]} if stage.split(":")[-1] in datasets else set(datasets)
        )
        if stage == "step7_prefix":
            expected_datasets = {None}  # Prefix precedes main unit preparation.
        expected_rounds = {0, 1, 2, 3} if stage == "doubter:model-stage" else {None}
        _require(
            set(scoped) == {(d, r) for d in expected_datasets for r in expected_rounds},
            "upstream_component_scope_incomplete",
        )
        current = source_fingerprint(project_root, stage)
        for component in scoped.values():
            _require(
                component.launch_id == manifest.launch_id or component.component_id in inherited,
                "upstream_component_origin_unproven",
            )
            row = _component_selection(state, component)
            _require(
                component.run_id == row.get("compiler_run_id"), "upstream_component_run_invalid"
            )
            config_hash = inherited.get(component.component_id,
                                        str(manifest.identity["launch_config_hash"]))
            validate_component(component, config_hash, row)
            _require(
                source_fingerprints_compatible(stage, component.source_fingerprint, current),
                "upstream_stage_source_incompatible",
            )
            _require(
                checkpoint_revision_is_restore_compatible(
                    component.alembic_revision, manifest.alembic_revision
                ),
                "upstream_component_revision_incompatible",
            )
            selected.append(component)
    return selected


def verify_receipt(launcher: ResearchExperimentLauncher, state: Any) -> None:
    """Check import provenance without depending on the continued existence of its source."""
    receipt = state.metadata.get("upstream_import")
    _require(isinstance(receipt, dict), "upstream_import_receipt_missing")
    payload = {key: value for key, value in receipt.items() if key != "content_hash"}
    _require(
        receipt.get("content_hash") == canonical_content_hash(payload),
        "upstream_import_receipt_hash_invalid",
    )
    _require(
        receipt.get("protocol") == IMPORT_PROTOCOL
        and receipt.get("target_launch_id") == launcher.launch_id
        and receipt.get("target_config_hash") == launcher.effective_config_hash
        and receipt.get("target_source_inputs_hash")
        == launcher._checkpoint_identity()["source_inputs_hash"]
        and receipt.get("model_lock_sha256")
        == launcher._checkpoint_identity()["model_lock_sha256"],
        "upstream_import_receipt_identity_invalid",
    )
    _require(
        receipt == json.loads((launcher.launch_dir / "upstream_import.json").read_text()),
        "upstream_import_receipt_file_mismatch",
    )
    proof = receipt.get("model_lock_equivalence")
    if proof is not None:
        _require(isinstance(proof, dict), "upstream_import_lock_proof_invalid")
        policy = proof.get("protocol")
        _require(
            policy in (MODEL_LOCK_EQUIVALENCE_POLICY, _LEGACY_MODEL_LOCK_POLICY),
            "upstream_import_lock_policy_invalid",
        )
        if launcher.preparation is None:
            raise UpstreamImportError("upstream_current_model_lock_missing")
        digest, payload = _model_lock_payload(launcher.preparation.lock_path, policy=policy)
        _require(
            proof.get("excluded_fields") == (
                _MODEL_LOCK_PROVENANCE_FIELDS if policy == MODEL_LOCK_EQUIVALENCE_POLICY else None
            )
            and proof.get("target_sha256") == digest == receipt.get("model_lock_sha256")
            and proof.get("contract_hash") == canonical_content_hash(payload)
            and isinstance(proof.get("source_sha256"), str)
            and re.fullmatch(r"[a-f0-9]{64}", proof["source_sha256"]) is not None,
            "upstream_import_lock_proof_invalid",
        )
    inventory = [
        StageComponentManifest.model_validate_json(line)
        for line in (launcher.launch_dir / "stage_components.jsonl").read_text().splitlines()
        if line
    ]
    selected = _receipt_inventory(receipt, inventory)
    configs = _receipt_component_configs(receipt, selected)
    for component in selected:
        validate_component(component, configs[component.component_id],
                           _component_selection(state, component))
    _require(
        state.metadata.get("upstream_import_source") == receipt.get("source_launch_id"),
        "upstream_import_origin_mismatch",
    )
    _require(state.launch_id == launcher.launch_id, "upstream_import_prefix_invalid")
    _verify_imported_prefix(state, receipt)


def imported_program_audit_hash(
    launcher: ResearchExperimentLauncher, *, dataset: str, inputs: dict[str, Any],
) -> str:
    """Recompute the audit using its original compiler origin, not the last importer."""
    verify_receipt(launcher, launcher.state)
    receipt = launcher.state.metadata["upstream_import"]
    inventory = [StageComponentManifest.model_validate_json(line) for line in
                 (launcher.launch_dir / "stage_components.jsonl").read_text().splitlines() if line]
    selected = _receipt_inventory(receipt, inventory)
    configs = _receipt_component_configs(receipt, selected)
    row = launcher.state.dataset_artifacts[dataset]
    origins = {(c.launch_id, configs[c.component_id]) for c in selected
               if c.stage == "compiler_program:model-stage" and c.dataset == dataset
               and c.run_id == row.get("compiler_run_id")}
    if len(origins) != 1:
        raise ImportedProgramAuditError("upstream_program_audit_origin_ambiguous", dataset)
    launch_id, config_hash = next(iter(origins))
    digest = canonical_content_hash({
        **inputs, "launch_id": launch_id, "launcher_config_hash": config_hash,
    })
    if digest != row.get("program_pilot_audit_hash"):
        raise ImportedProgramAuditError("upstream_program_audit_identity_mismatch", dataset)
    return digest


def preflight_imported_program_audits(launcher: ResearchExperimentLauncher) -> None:
    """Fail before GPU calibration, without changing any imported scientific facts."""
    if not launcher.state.metadata.get("upstream_import"):
        return
    from pmmc.compiler.preparation import CompilationUnitPreparationService

    for dataset, row in sorted(launcher.state.dataset_artifacts.items()):
        try:
            preparation = Path(row["unit_preparation_dir"])
            manifest, _ = CompilationUnitPreparationService.verify(preparation)
            if (manifest.snapshot_id != row["snapshot_id"]
                    or manifest.run_id != row["compiler_run_id"]):
                raise ImportedProgramAuditError("upstream_program_preparation_scope", dataset)
            imported_program_audit_hash(launcher, dataset=dataset, inputs={
                "protocol": "compiler-verified-qbank-development-audit-v1",
                "partition_id": row["partition_id"],
                "sample_id": row["pilot_sample_id"],
                "dataset": dataset,
                "snapshot_id": manifest.snapshot_id,
                "selection_hash": row["selection_hash"],
                "conversation_ids": manifest.conversation_ids,
                "unit_preparation_id": preparation.name,
            })
        except (OSError, KeyError, ValueError) as exc:
            if isinstance(exc, ImportedProgramAuditError):
                error = exc
            else:
                error = ImportedProgramAuditError("upstream_program_audit_inputs_invalid", dataset)
            print("STEP7B_PROGRAM_AUDIT " + json.dumps({
                "state": "failed", **error.safe_diagnostics(),
            }, sort_keys=True), flush=True)
            if error is exc:
                raise
            raise error from exc
        print("STEP7B_PROGRAM_AUDIT " + json.dumps({
            "state": "ready", "dataset": dataset, "phase": "before_model_calibration",
            "identity_scope": "sealed_compiler_origin", "audit_hash_preserved": True,
            "unit_count": manifest.unit_count, "model_calls": 0,
        }, sort_keys=True), flush=True)


def load_source(launcher: ResearchExperimentLauncher, source_id: str) -> dict[str, Any]:
    from pmmc.experiments.launcher import ExperimentLaunchManifest, load_launcher_config

    validate_import_id(source_id)
    _require(
        launcher.config.target.value == "pmmc_step7b_executor_pilot"
        and launcher.config.metadata.get("paper_result") is False
        and launcher.config.infrastructure.mode.value == "self_contained",
        "upstream_import_requires_development_pilot",
    )
    _require(source_id != launcher.launch_id, "upstream_import_requires_new_launch")
    root = launcher.settings.runs_path / "launches" / source_id
    _require(root.is_dir() and not root.is_symlink(), "upstream_launch_missing")
    config = load_launcher_config(root / "resolved_config.yaml")
    _require(config.config_hash == launcher.config.config_hash, "upstream_launcher_config_changed")
    source = json.loads((root / "source_inputs.json").read_text())
    manifest = ExperimentLaunchManifest.from_json((root / "launch_manifest.json").read_text())
    _require(manifest.launch_id == source_id, "upstream_launch_manifest_invalid")
    changed = validate_input_delta(
        source_inputs=source,
        current_inputs=launcher.source_inputs,
        config_hash=config.config_hash,
        source_config_hash=manifest.config_hash,
        source_launch_id=source_id,
        target_launch_id=launcher.launch_id,
    )
    _require(
        manifest.metadata.get("source_inputs_hash") == canonical_content_hash(_identity(source)),
        "upstream_source_inputs_hash_invalid",
    )
    state = json.loads((root / "launch_state.json").read_text())
    _require(state.get("status") in {"failed", "completed"}, "upstream_launch_not_quiescent")
    return {
        "source_launch_id": source_id,
        "source_config_hash": manifest.config_hash,
        "source_inputs_hash": canonical_content_hash(_identity(source)),
        "changed_inputs": changed,
        "source_root": root,
    }


async def verify_imported_database(
    settings: Any,
    state: Any,
    components: list[StageComponentManifest],
) -> dict[str, int]:
    from sqlalchemy import text

    from pmmc.compiler.doubter_runner import verify_doubter_run
    from pmmc.compiler.r0_gate import verify_r0_input
    from pmmc.compiler.verified_qbank import verify_routing_projection
    from pmmc.infrastructure.database.constants import RUNTIME_SCHEMA, DatabaseRole
    from pmmc.infrastructure.database.doubter_repositories import PostgresDoubterRepository
    from pmmc.infrastructure.database.engine import create_session_factory

    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session:
            # A whole-database checkpoint is importable only before online work starts.
            count = int(
                await session.scalar(
                    text(f"SELECT count(*) FROM {RUNTIME_SCHEMA}.query_executor_runs")
                )
                or 0
            )
            _require(count == 0, "upstream_database_contains_online_runs")
            for row in state.dataset_artifacts.values():
                for table in ("final_answers", "structured_answer_outputs"):
                    answered = await session.scalar(
                        text(f"SELECT count(*) FROM {RUNTIME_SCHEMA}.{table} WHERE run_id=:run_id"),
                        {"run_id": row["fallback_run_id"]},
                    )
                    _require(not answered, "upstream_database_contains_fallback_answers")
                calls = await session.scalar(
                    text(f"SELECT count(*) FROM {RUNTIME_SCHEMA}.model_calls "
                         "WHERE run_id=:run_id AND stage='answer'"),
                    {"run_id": row["fallback_run_id"]},
                )
                _require(not calls, "upstream_database_contains_fallback_answer_calls")
            stored: dict[str, StageComponentManifest] = {}
            origins = sorted({c.launch_id for c in components if c.run_id is not None})
            for origin in origins:
                records = await PostgresDoubterRepository(session).list_stage_components(origin)
                stored.update({item.component_id: item for item in records})
            for component in components:
                if component.run_id is not None:
                    _require(
                        stored.get(component.component_id) == component,
                        "upstream_database_component_mismatch",
                    )
            for row in state.dataset_artifacts.values():
                existing = await PostgresDoubterRepository(session).get_r0_manifest(
                    row["compiler_run_id"]
                )
                _require(existing is not None, "upstream_database_r0_missing")
        print("STEP7B_UPSTREAM_IMPORT phase=database_components state=ready "
              f"origin_launch_count={len(origins)} verified_components={len(components)} "
              "lookup_scope=verified_origins model_calls=0", flush=True)
        for row in state.dataset_artifacts.values():
            r0 = await verify_r0_input(sessions, run_id=row["compiler_run_id"])
            _require(
                r0.snapshot_id == row["snapshot_id"] and r0.downstream_viability,
                "upstream_r0_scope_or_viability_invalid",
            )
            report = await verify_doubter_run(sessions, run_id=row["compiler_run_id"])
            assessments = report.get("assessment_count")
            _require(
                report.get("valid") is True and isinstance(assessments, int) and assessments > 0,
                "upstream_doubter_facts_invalid",
            )
            routing = await verify_routing_projection(sessions, run_id=row["compiler_run_id"])
            members = routing.get("routing_member_count")
            _require(
                routing.get("valid") is True and isinstance(members, int) and members > 0,
                "upstream_routing_facts_invalid",
            )
        return {
            "verified_components": len(components),
            "online_runs": count,
            "verified_r0_runs": 2,
            "verified_doubter_runs": 2,
            "verified_routing_runs": 2,
        }
    finally:
        await engine.dispose()


def restore_upstream(launcher: ResearchExperimentLauncher, source_id: str) -> None:
    from pmmc.experiments.launcher import (
        ExperimentLaunchState,
        ExperimentLaunchStatus,
        _atomic_json,
        _atomic_jsonl,
    )

    source = load_source(launcher, source_id)
    identity = launcher._checkpoint_identity()
    _require(bool(identity.get("model_lock_sha256")), "upstream_current_model_lock_missing")
    if launcher.preparation is None:
        raise UpstreamImportError("upstream_current_model_lock_missing")
    lock_proof = verify_model_lock_equivalence(
        source["source_root"] / "model_lock.json", launcher.preparation.lock_path
    )
    _require(
        lock_proof["target_sha256"] == identity["model_lock_sha256"],
        "upstream_current_model_lock_changed",
    )
    print(
        "STEP7B_UPSTREAM_IMPORT phase=model_lock state=ready "
        f"byte_equal={str(lock_proof['source_sha256'] == lock_proof['target_sha256']).lower()} "
        f"contract_equal=true policy={MODEL_LOCK_EQUIVALENCE_POLICY} "
        f"excluded_fields={','.join(_MODEL_LOCK_PROVENANCE_FIELDS)} model_calls=0",
        flush=True,
    )
    old_identity = {
        **identity,
        "launch_config_hash": source["source_config_hash"],
        "source_inputs_hash": source["source_inputs_hash"],
        "model_lock_sha256": lock_proof["source_sha256"],
    }
    service = ExperimentCheckpointService(
        launcher.launch_settings,
        launch_id=source_id,
        root=launcher.settings.runs_path / "private" / "checkpoints",
        identity=old_identity,
        retain_milestones=True,
    )
    components = [
        StageComponentManifest.model_validate_json(line)
        for line in (source["source_root"] / "stage_components.jsonl").read_text().splitlines()
        if line
    ]
    revision = asyncio.run(service.current_database_revision())
    selected = None
    candidates = list(reversed(service.directories()))
    failures: Counter[tuple[str, str, str]] = Counter()
    checked = 0
    print(
        f"STEP7B_UPSTREAM_IMPORT phase=selection state=start candidates={len(candidates)}",
        flush=True,
    )
    for directory in candidates:
        checked += 1
        phase = "header"
        try:
            header = service.inspect_candidate(directory, require_identity=False)
            phase = "identity"
            service._assert_restore_identity(header.identity)
            phase = "revision"
            _require(
                checkpoint_revision_is_restore_compatible(header.alembic_revision, revision),
                "upstream_checkpoint_revision_incompatible",
            )
            phase = "state"
            state = ExperimentLaunchState.from_json((directory / "launch_state.json").read_text())
            phase = "components"
            source_receipt = state.metadata.get("upstream_import")
            if source_receipt is not None:
                _require(source_receipt == json.loads(
                    (source["source_root"] / "upstream_import.json").read_text()
                ), "upstream_import_receipt_file_mismatch")
            picked = select_components(
                state=state,
                manifest=header,
                components=components,
                project_root=launcher.settings.project_root,
            )
        except (OSError, ValueError) as exc:
            signature = _selection_failure(exc, phase)
            failures[signature] += 1
            code, substage, fields = signature
            if failures[signature] == 1:
                print(
                    f"STEP7B_UPSTREAM_IMPORT state=skipped sequence={directory.name} "
                    f"substage={substage} error_code={code} mismatch_fields={fields} "
                    f"error_type={type(exc).__name__} repeated_reason_logs=suppressed",
                    flush=True,
                )
            continue
        print(
            f"STEP7B_UPSTREAM_IMPORT state=verifying sequence={directory.name} model_calls=0",
            flush=True,
        )
        # Hash once here; object-store upload independently verifies each payload on restore.
        manifest = service.verify(directory, require_identity=False, verify_artifact_content=False)
        _require(
            manifest.execution_cache_schema_version == EXECUTION_CACHE_SCHEMA_VERSION,
            "upstream_checkpoint_schema_incompatible",
        )
        selected = directory, manifest, state, picked
        break
    print(
        "STEP7B_UPSTREAM_IMPORT "
        + json.dumps(
            {
                "phase": "selection", "state": "done" if selected is not None else "failed",
                "candidates": len(candidates), "checked": checked,
                "selected_sequence": selected[1].sequence if selected is not None else None,
                "rejection_counts": [
                    {"error_code": code, "substage": phase, "mismatch_fields": fields, "count": n}
                    for (code, phase, fields), n in sorted(failures.items())
                ],
                "model_calls": 0,
            }, sort_keys=True,
        ),
        flush=True,
    )
    _require(selected is not None, "upstream_compatible_checkpoint_missing")
    assert selected is not None
    directory, manifest, state, picked = selected
    inherited = _inherited_component_configs(state, manifest, components)
    component_configs = {
        c.component_id: inherited.get(c.component_id, source["source_config_hash"])
        for c in picked
    }
    print("STEP7B_UPSTREAM_IMPORT phase=component_provenance state=ready "
          f"policy={COMPONENT_PROVENANCE_POLICY} "
          f"inherited_components={sum(c.component_id in inherited for c in picked)} "
          f"native_components={sum(c.component_id not in inherited for c in picked)} "
          f"origin_launch_count={len({c.launch_id for c in picked})} "
          "component_ids_rewritten=false model_calls=0", flush=True)
    asyncio.run(service.restore(directory, verified_manifest=manifest, require_empty=True))
    accounting = asyncio.run(verify_imported_database(launcher.launch_settings, state, picked))
    receipt = {
        "protocol": IMPORT_PROTOCOL,
        "source_launch_id": source_id,
        "target_launch_id": launcher.launch_id,
        "source_config_hash": source["source_config_hash"],
        "target_config_hash": launcher.effective_config_hash,
        "source_checkpoint_sequence": manifest.sequence,
        "source_checkpoint_hash": manifest.content_hash,
        "source_inputs_hash": source["source_inputs_hash"],
        "target_source_inputs_hash": identity["source_inputs_hash"],
        "changed_inputs": source["changed_inputs"],
        "model_lock_sha256": identity["model_lock_sha256"],
        "model_lock_equivalence": lock_proof,
        "snapshots": manifest.identity["snapshots"],
        "component_ids": sorted(item.component_id for item in picked),
        "component_inventory_hash": canonical_content_hash(
            [item.model_dump(mode="json") for item in picked]
        ),
        "component_provenance_policy": COMPONENT_PROVENANCE_POLICY,
        "component_config_hashes": component_configs,
        "invalidated_stages": INVALIDATED_STAGES,
        "accounting": accounting,
    }
    receipt["content_hash"] = canonical_content_hash(receipt)
    completed = [
        stage
        for stage in state.completed_stage_ids
        if stage_family(stage) in UPSTREAM_FAMILIES | (REPLAY_FAMILIES - {"export"})
    ]
    # Preserve original scientific IDs. Only the new launch's routing/report identities change.
    metadata = {
        key: value
        for key, value in state.metadata.items()
        if key
        in {
            "step7b_prefix",
            "step6d_compiler_conformance",
            "step6c_planner_conformance",
            "step7b_compiler_conformance",
            "partition_id",
            "pilot_mode",
        }
    }
    prefix = dict(metadata.get("step7b_prefix", {}))
    original_hash = prefix.pop("content_hash", None)
    _require(
        original_hash == canonical_content_hash(prefix) and prefix.get("launch_id") == source_id,
        "upstream_prefix_hash_invalid",
    )
    prefix.update(
        {
            "launch_id": launcher.launch_id,
            "origin_launch_id": source_id,
            "origin_content_hash": original_hash,
            "upstream_import_hash": receipt["content_hash"],
        }
    )
    prefix["content_hash"] = canonical_content_hash(prefix)
    metadata.update(
        {
            "upstream_import": receipt,
            "upstream_import_source": source_id,
            "step7b_prefix": prefix,
            "step7b_prefix_path": str(launcher.launch_dir / "step7b_prefix_verification.json"),
        }
    )
    for key in (
        "step6d_compiler_conformance",
        "step6c_planner_conformance",
        "step7b_compiler_conformance",
    ):
        if isinstance(metadata.get(key), dict):
            path = launcher.launch_dir / f"{key}.json"
            _atomic_json(path, metadata[key])
            metadata[f"{key}_path"] = str(path)
    launcher.state = state.model_copy(
        update={
            "launch_id": launcher.launch_id,
            "status": ExperimentLaunchStatus.RUNNING,
            "started_at": launcher.state.started_at,
            "completed_at": None,
            "error_type": None,
            "completed_stage_ids": completed,
            "stage_results": [
                item.model_copy(update={"reused": True})
                for item in state.stage_results
                if item.stage_id in completed and item.status.value == "completed"
            ],
            "metadata": metadata,
        }
    )
    _atomic_json(launcher.launch_dir / "upstream_import.json", receipt)
    _atomic_json(launcher.launch_dir / "step7b_prefix_verification.json", prefix)
    _atomic_jsonl(launcher.launch_dir / "stage_components.jsonl", picked)
    _atomic_jsonl(launcher.launch_dir / "stage_results.jsonl", launcher.state.stage_results)
    launcher._save_state()
    verify_receipt(launcher, launcher.state)
    # Materialize an independent checkpoint without dumping/downloading the same facts again.
    target = launcher._checkpoint_service()
    seed = target.import_seed(
        service,
        directory,
        manifest,
        launcher.state_path,
        launcher.launch_dir / "stage_results.jsonl",
    )
    launcher.state.metadata.update(
        {"restored_checkpoint": str(seed), "latest_checkpoint": str(seed)}
    )
    launcher._checkpoint_restored = True
    launcher.resume = True
    launcher._save_state()
    print(
        "STEP7B_UPSTREAM_IMPORT "
        + json.dumps(
            {
                "state": "ready",
                "source_launch_id": source_id,
                "target_launch_id": launcher.launch_id,
                "sequence": manifest.sequence,
                "reused_stage_count": len(completed),
                "invalidated_stages": INVALIDATED_STAGES,
                "model_calls": 0,
                **accounting,
            },
            sort_keys=True,
        ),
        flush=True,
    )
