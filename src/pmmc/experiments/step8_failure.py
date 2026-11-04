"""Private recovery instructions and public identifier-only failure diagnostics."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from pmmc.experiments.step8_artifacts import sha256_file, write_json
from pmmc.experiments.step8_config_contract import CONFIG_FIELDS, FIXED_CONFIGS, POLICY_FIELDS
from pmmc.experiments.step8_jobs import SOURCE_OWNERS
from pmmc.infrastructure.database.errors import safe_database_error_diagnostics


def failure_diagnostics(error: BaseException) -> dict[str, str]:
    return {
        key: value
        for key, value in safe_database_error_diagnostics(error).items()
        if re.fullmatch(r"[A-Za-z0-9_]{1,128}", value)
    }


def failure_frames(error: BaseException) -> list[dict[str, str | int]]:
    """Identifiers only: no exception text, source lines, locals or file paths."""
    frames: list[dict[str, str | int]] = []
    tb = error.__traceback__
    while tb is not None:
        module = tb.tb_frame.f_globals.get("__name__", "")
        function = tb.tb_frame.f_code.co_name
        if (isinstance(module, str)
                and re.fullmatch(r"pmmc(?:\.[A-Za-z_][A-Za-z0-9_]*)+", module)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function)):
            frames.append({"module": module, "function": function, "line": tb.tb_lineno})
        tb = tb.tb_next
    return frames[-8:]


_GATE_CODES = {
    "cold_snapshot_differs_from_preserved_partition",
    "snapshot_asset_checksum_coverage_mismatch",
    "snapshot_validated_asset_not_local",
    "snapshot_asset_checksum_identity_mismatch",
    "real_diagnostic_prefix_required_run_diagnostic_selection_first",
    "real_diagnostic_prefix_scientific_viability_failed",
    "diagnostic_prefix_denominator_invalid",
    "diagnostic_prefix_selection_ambiguous",
    "paper_queries_exposed_under_another_protocol",
    "confirmation_queries_exposed_after_selection",
    "exposure_event_integrity_failed",
    "exposure_query_identity_invalid",
    "exposure_selection_campaign_mismatch",
    "exposure_selection_partition_mismatch",
    "doubter_round_barrier_identity_mismatch",
    "doubter_round_barrier_assessment_changed",
    "scratch_cleanup_ownership_mismatch",
    "scratch_cannot_own_persistent_model_cache",
    "gc_unrecognized_campaign_directory",
    "gc_reference_symlink_forbidden",
    "component_campaign_identity_mismatch",
    "component_scientific_source_changed",
    "step8_upcoming_stage_source_changed",
    "step8_upcoming_dependency_missing",
    "step8_upcoming_dependency_receipt_changed",
    "step8_frozen_model_lock_changed",
    "step8_frozen_model_entry_hashes_changed",
    "step8_model_identity_required_alias_missing",
    "step8_model_identity_fake_entry",
    "step8_model_identity_original_registry_mismatch",
    "step8_model_identity_lock_coverage_mismatch",
    "step8_model_identity_hub_entry_mismatch",
    "step8_model_identity_config_alias_mismatch",
    "step8_model_identity_protocol_alias_mismatch",
    "step8_model_identity_entry_missing_or_invalid",
    *(
        f"{prefix}{name}"
        for prefix in (
            "step8_frozen_config_changed_",
            "step8_frozen_config_hash_missing_",
        )
        for name in (*CONFIG_FIELDS, *FIXED_CONFIGS)
    ),
    *("step8_unfrozen_config_override_" + name for name in FIXED_CONFIGS),
    *("step8_frozen_policy_changed_" + name for name in POLICY_FIELDS),
}


def safe_failure_code(error: BaseException) -> str:
    code = getattr(error, "error_code", getattr(error, "code", type(error).__name__))
    # Legacy gates raise ValueError with an identifier, not a structured error.
    # Never promote arbitrary exception text into the safe diagnostic stream.
    if type(error) is ValueError and str(error) in _GATE_CODES:
        code = str(error)
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_]{1,128}", code):
        return code
    return type(error).__name__


def scientific_failure_state(failure: Mapping[str, object]) -> dict[str, object]:
    code = failure.get("error_code")
    if not isinstance(code, str):
        code = ""
    drift = code in {
        "component_scientific_source_changed",
        "step8_upcoming_stage_source_changed",
        "step8_frozen_model_entry_hashes_changed",
    } or (isinstance(code, str) and code.startswith("step8_frozen_policy_changed_"))
    reported = failure.get("affected_stages")
    affected = (
        sorted({stage for stage in reported if isinstance(stage, str) and stage in SOURCE_OWNERS})
        if isinstance(reported, (list, tuple, set))
        else []
    )
    stage = failure.get("stage")
    if code == "component_scientific_source_changed" and isinstance(stage, str):
        if stage in SOURCE_OWNERS:
            affected = [stage]
    return {
        "scientific_identity_unchanged": False if drift else None,
        "scientific_revalidation_required": True,
        "affected_stages": affected or None,
        "invalidated_stages": affected if drift and affected else None,
        "stage_impact_scope": "reported_stages_only_not_transitive",
    }


def write_failure_capsule(directory: Path, attempt: Path, failure: Mapping[str, object]) -> str:
    marker = directory / "checkpoints/last-database-checkpoint.json"
    checkpoint = None
    if marker.is_file():
        try:
            value = json.loads(marker.read_text())
            if (
                isinstance(value, dict)
                and re.fullmatch(r"[0-9]{6}", str(value.get("sequence", "")))
                and re.fullmatch(r"[a-f0-9]{64}", str(value.get("fact_hash", "")))
            ):
                checkpoint = {key: value[key] for key in ("sequence", "fact_hash")}
        except (OSError, ValueError):
            pass
    manifests = directory / "manifests"
    science = scientific_failure_state(failure)
    capsule = {
        "schema_version": "step8-failure-capsule-v1",
        "failure": {**failure, **science},
        "latest_checkpoint": checkpoint,
        "checkpoint_reverification_required": True,
        **science,
        "resume_requires_revalidation": True,
        "campaign_manifest_hash": (
            sha256_file(manifests / "campaign.json")
            if (manifests / "campaign.json").is_file()
            else None
        ),
        # Paths belong only in the private capsule, never in the safe event stream.
        "resume_argv": [
            "pmmc",
            "benchmark",
            "resume",
            "--campaign-dir",
            str(directory.resolve()),
        ],
        "safe_log": "safe-execution.log",
        "response_journal_retained": True,
    }
    target = attempt / "failure-capsule.json"
    write_json(target, capsule, immutable=True)
    return sha256_file(target)
