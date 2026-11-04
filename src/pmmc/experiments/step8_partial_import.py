"""Explicit, development-only migration of completed Q/P0/index facts.

This is not an R0 proof or a general source-fingerprint exception. A migration
policy names the donor, exact component receipts and reviewed execution sources.
The new campaign still executes dry-run, R0 eligibility and Doubter normally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import sha256_file
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.experiments.step8_model_identity import compiler_model_entry_hashes
from pmmc.schemas.step8 import BenchmarkCampaignManifest, Step8BenchmarkConfig

PARTIAL_STAGES = ("questioner", "planner0", "text_index", "multimodal_index")
POLICY_VERSION = "step8-partial-prefix-policy-v1"


def scope_key(scope: dict[str, Any]) -> str:
    return canonical_content_hash(
        {
            key: scope[key]
            for key in (
                "dataset",
                "snapshot_id",
                "conversation_id",
                "treatment",
            )
        }
    )


def load_policy(config: Step8BenchmarkConfig, root: Path) -> dict[str, Any] | None:
    name = config.metadata.get("partial_import_policy")
    if name is None:
        return None
    if not isinstance(name, str) or not name:
        raise ValueError("partial_import_policy_path_invalid")
    path = root / name
    value = json.loads(path.read_bytes())
    if (
        value.get("version") != POLICY_VERSION
        or value.get("approved") is not True
        or value.get("paper_test_allowed") is not False
        or value.get("covered_stages") != list(PARTIAL_STAGES)
        or not value.get("scopes")
    ):
        raise ValueError("partial_import_policy_invalid")
    return value


def validate_policy(
    config: Step8BenchmarkConfig,
    target: BenchmarkCampaignManifest,
    root: Path,
) -> tuple[dict[str, Any], BenchmarkCampaignManifest] | None:
    policy = load_policy(config, root)
    if policy is None:
        return None
    path = root / str(config.metadata["partial_import_policy"])
    if target.protocol.config_hashes.get("partial_import_policy") != sha256_file(path):
        raise ValueError("partial_import_policy_not_frozen")
    donor = root / policy["source_campaign_dir"]
    source = BenchmarkCampaignManifest.model_validate_json(
        (donor / "manifests/campaign.json").read_bytes()
    )
    if (
        not source.diagnostic_only
        or not target.diagnostic_only
        or source.paper_result
        or target.paper_result
        or source.campaign_id == target.campaign_id
        or source.campaign_id != policy["source_campaign_id"]
        or source.protocol.content_hash != policy["source_protocol_hash"]
    ):
        raise ValueError("partial_import_campaign_boundary_invalid")
    before, after = source.protocol, target.protocol
    for name in (
        "seed",
        "partition_hash",
        "exposure_manifest_hash",
        "questioner",
        "verifier",
        "planner",
        "compiler_alias",
    ):
        if getattr(before, name) != getattr(after, name):
            raise ValueError("partial_import_scientific_input_changed_" + name)
    for name in ("questioner_config", "planner_config", "foundation_config"):
        if before.config_hashes[name] != after.config_hashes[name]:
            raise ValueError("partial_import_config_changed_" + name)
    if compiler_model_entry_hashes(before) != compiler_model_entry_hashes(after):
        raise ValueError("partial_import_producer_model_changed")
    for stage in ("unit_prepare", *PARTIAL_STAGES):
        if (
            before.source_fingerprints[stage] != policy["source_fingerprints"][stage]
            or after.source_fingerprints[stage] != policy["target_fingerprints"][stage]
        ):
            raise ValueError("partial_import_source_transition_changed")
    return policy, source


def donor_components(
    policy: dict[str, Any],
    source: BenchmarkCampaignManifest,
    root: Path,
    scope: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, ComponentReceipt]]:
    entry = policy["scopes"].get(scope_key(scope))
    if not entry or entry["scope"] != {key: scope[key] for key in entry["scope"]}:
        raise ValueError("partial_import_scope_not_approved")
    directory = root / policy["source_campaign_dir"]
    receipts = {}
    for stage in PARTIAL_STAGES:
        binding = entry["components"][stage]
        job = BenchmarkStageJob.model_validate_json(
            (directory / "private/jobs" / (binding["job_id"] + ".json")).read_bytes()
        )
        receipt = ComponentReceipt.model_validate_json(
            (directory / "checkpoints/components" / (job.job_id + ".json")).read_bytes()
        )
        if (
            job.campaign_id != source.campaign_id
            or job.stage != stage
            or scope_key(dict(job.parameters)) != scope_key(scope)
            or receipt.content_hash != binding["receipt_hash"]
            or receipt.output.get("compiler_run_id") != entry["source_run_id"]
        ):
            raise ValueError("partial_import_source_component_changed")
        receipts[stage] = receipt
    if scope["treatment"] == "fixed_planner":
        parent_scope = {**scope, "treatment": "main"}
        main = policy["scopes"].get(scope_key(parent_scope))
        if not main or entry.get("paired_main_run_id") != main["source_run_id"]:
            raise ValueError("partial_import_fixed_planner_pair_missing")
    return entry, receipts


def target_registration(backend: Any, scope: dict[str, Any]) -> dict[str, Any]:
    from pmmc.compiler.doubter_config import doubter_config_hash
    from pmmc.compiler.doubter_prompt import doubter_prompt_sha256, doubter_response_schema_sha256
    from pmmc.compiler.planner_config import planner_config_hash
    from pmmc.compiler.planner_prompt import planner_prompt_sha256, planner_response_schema_sha256
    from pmmc.compiler.questioner_prompt import questioner_prompt_sha256
    from pmmc.experiments.step8_backend import compiler_subprotocol_hash
    from pmmc.models.registry import load_model_registry

    q, p, d = backend._compiler_configs(scope)
    registry = load_model_registry(backend.settings.resolved_model_registry_path)
    identity = compiler_subprotocol_hash(backend.campaign, scope)
    updates = {
        "questioner_freeze_manifest_hash": identity,
        "planner_config_hash": planner_config_hash(p),
        "planner_model_alias": p.planner_model_alias,
        "planner_model_revision": registry.get(p.planner_model_alias).model_revision,
        "planner_prompt_version": p.prompt_version,
        "planner_prompt_hash": planner_prompt_sha256(p),
        "planner_renderer_version": p.renderer_version,
        "planner_parser_version": p.parser_version,
        "planner_schema_version": p.schema_version,
        "program_policy_version": p.policy_version,
        "dry_run_policy_version": p.dry_run_policy,
        "doubter_config_hash": doubter_config_hash(d),
        "doubter_model_alias": d.doubter_model_alias,
        "doubter_model_revision": registry.get(d.doubter_model_alias).model_revision,
        "doubter_prompt_version": d.prompt_version,
        "doubter_prompt_hash": doubter_prompt_sha256(d),
        "doubter_renderer_version": d.renderer_version,
        "doubter_parser_version": d.parser_version,
        "doubter_schema_version": d.schema_version,
        "r0_input_policy_version": d.r0_input_policy_version,
        "routing_projection_version": d.routing_projection_version,
        "active_round": d.active_round,
        "max_round": d.max_round,
    }
    return {
        "compiler_updates": updates,
        "manifest_metadata": {
            "compiler_subprotocol_hash": identity,
            "questioner_freeze_or_audit_hash": identity,
            "questioner_identity_kind": "step8_method_protocol",
            "planner_config_hash": planner_config_hash(p),
            "planner_response_schema_hash": planner_response_schema_sha256(p),
            "doubter_response_schema_hash": doubter_response_schema_sha256(d),
            "active_round": d.active_round,
            "max_round": d.max_round,
        },
        "prompt_hashes": {
            q.prompt_version: questioner_prompt_sha256(q.prompt_version),
            q.verifier_prompt_version: questioner_prompt_sha256(q.verifier_prompt_version),
            d.prompt_version: doubter_prompt_sha256(d),
        },
    }


async def partial_stage(backend: Any, job: BenchmarkStageJob, scope: dict[str, Any]):
    if not backend.config.metadata.get("partial_import_policy"):
        return None
    if job.stage not in {"unit_prepare", *PARTIAL_STAGES}:
        return None
    if job.operation not in {"units", "compiler", "adopt_questioner"}:
        return None
    from pmmc.experiments.step8_backend import compiler_instance_id
    from pmmc.infrastructure.database.constants import DatabaseRole
    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.step8_partial_import import (
        PostgresPartialPrefixImportRepository,
    )
    from pmmc.infrastructure.object_store import create_runtime_object_store

    context = validate_policy(backend.config, backend.campaign, backend.settings.project_root)
    assert context is not None
    policy, source = context
    entry, receipts = donor_components(policy, source, backend.settings.project_root, scope)
    target = compiler_instance_id(backend.campaign, scope)
    policy_hash = backend.campaign.protocol.config_hashes["partial_import_policy"]
    engine, sessions = create_session_factory(backend.settings, DatabaseRole.RUNTIME)
    try:

        async def check_proof(value):
            from sqlalchemy import text

            from pmmc.experiments.step8_partial_cost_dependencies import (
                verify_partial_source_receipts,
            )

            if (
                value["source_run_id"] != entry["source_run_id"]
                or value["target_campaign_id"] != backend.campaign.campaign_id
                or value["target_protocol_hash"] != backend.campaign.protocol.content_hash
                or value["source_scope"] != entry["scope"]
                or value["source_receipts"]
                != {stage: receipt.content_hash for stage, receipt in receipts.items()}
            ):
                raise ValueError("partial_import_proof_not_authorized")
            if scope["treatment"] == "fixed_planner":
                original = value.get("source_questioner_upstream_import") or {}
                if original.get("source_run_id") != entry["paired_main_run_id"]:
                    raise ValueError("partial_import_fixed_planner_questioner_not_paired")
                if job.operation == "adopt_questioner":
                    parents = backend._upstream(job)
                    if (
                        len(parents) != 1
                        or parents[0].output.get("partial_prefix_adoption", {}).get("source_run_id")
                        != entry["paired_main_run_id"]
                    ):
                        raise ValueError("partial_import_target_questioner_not_paired")
            async with sessions() as session:
                await session.execute(
                    text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                )
                await verify_partial_source_receipts(session, value)

        repository = PostgresPartialPrefixImportRepository(
            sessions, create_runtime_object_store(backend.settings), validate_proof=check_proof
        )
        proof = await repository.seal(
            run_id=entry["source_run_id"],
            campaign_id=source.campaign_id,
            protocol_hash=source.protocol.content_hash,
            compatibility_hash=policy_hash,
            target_campaign_id=backend.campaign.campaign_id,
            target_protocol_hash=backend.campaign.protocol.content_hash,
            source_scope=entry["scope"],
            source_receipts={stage: value.content_hash for stage, value in receipts.items()},
        )
        if job.operation == "units" or job.operation == "adopt_questioner":
            await repository.adopt(
                proof=proof,
                compatibility_hash=policy_hash,
                target_run_id=target,
                **target_registration(backend, scope),
            )
        else:
            verified = await repository.verify_adopted(target, policy_hash)
            if not verified:
                raise ValueError("partial_import_selected_stage_not_adopted_no_native_fallback")
        proof_json = proof.model_dump(mode="json") if hasattr(proof, "model_dump") else proof
        digest = backend.store.put_json(proof_json)
    finally:
        await engine.dispose()
    print(
        "STEP8_PARTIAL_IMPORT "
        + json.dumps(
            {
                "state": "verified",
                "stage": job.stage,
                "treatment": scope["treatment"],
                "source_run_id": entry["source_run_id"],
                "target_run_id": target,
                "provider_calls": 0,
                "gpu_operations": 0,
                "r0_claimed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "compiler_run_id": target,
        "phase": job.stage,
        "run_status": "running",
        "partial_prefix_adoption": {
            "source_proof_sha256": digest,
            "source_run_id": entry["source_run_id"],
            "policy_hash": policy_hash,
        },
        "provider_calls": 0,
        "r0_claimed": False,
    }
