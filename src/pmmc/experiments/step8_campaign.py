"""One immutable campaign and incrementally extensible benchmark assignments."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import durable_write, json_bytes, sha256_file
from pmmc.experiments.step8_jobs import (
    SOURCE_OWNERS,
    BenchmarkStageJob,
    scientific_source_fingerprint,
    validate_job_dag,
)
from pmmc.experiments.step8_model_identity import (
    load_step8_model_entry_hashes,
    stage_model_entry_hashes,
    validate_model_alias_contract,
)
from pmmc.experiments.step8_selection import (
    ExposureManifest,
    denominator_report,
    select_benchmark_queries,
)
from pmmc.pilots.partition import load_partition, verify_partition
from pmmc.schemas.base import DatasetName
from pmmc.schemas.step8 import (
    BenchmarkCampaignManifest,
    CampaignPhase,
    MethodProtocolFreeze,
    ResourceClass,
    ScientificProtocol,
    SelectionExpansionManifest,
    Step8BenchmarkConfig,
    Treatment,
    build_experiment_matrix,
)


def load_benchmark_config(path: Path) -> Step8BenchmarkConfig:
    return Step8BenchmarkConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def make_protocol(config: Step8BenchmarkConfig, root: Path) -> ScientificProtocol:
    validate_model_alias_contract(config, root)
    model_entry_hashes = load_step8_model_entry_hashes(config, root)
    partition = verify_partition(_resolve(root, config.partition_dir))
    exposure = ExposureManifest.model_validate_json(
        _resolve(root, config.exposure_manifest).read_text(encoding="utf-8")
    )
    paths = {
        name: getattr(config, name)
        for name in (
            "model_registry",
            "questioner_config",
            "planner_config",
            "doubter_config",
            "foundation_config",
            "query_executor_config",
        )
    }
    paths.update(
        baseline_full_context_config=Path("configs/baselines/full_context.yaml"),
        baseline_rag_config=Path("configs/baselines/rag.yaml"),
        evaluation_config=Path("configs/evaluation/baseline.yaml"),
    )
    if config.metadata.get("partial_import_policy") is not None:
        from pmmc.experiments.step8_partial_import import load_policy

        load_policy(config, root)
        paths["partial_import_policy"] = Path(str(config.metadata["partial_import_policy"]))
    return ScientificProtocol(
        seed=config.seed,
        oracle_at_50=config.oracle_at_50,
        model_lock_hash=sha256_file(_resolve(root, config.model_lock)),
        model_entry_hashes=model_entry_hashes,
        partition_hash=partition.content_hash,
        exposure_manifest_hash=exposure.content_hash,
        config_hashes={name: sha256_file(_resolve(root, path)) for name, path in paths.items()},
        source_fingerprints={
            stage: scientific_source_fingerprint(root, stage)
            for stage in SOURCE_OWNERS
            if stage != "export"
        },
        **{
            name: getattr(config, name)
            for name in (
                "match_threshold",
                "gap_threshold",
                "text_relevance_threshold",
                "multimodal_relevance_threshold",
                "no_reranker_match_threshold",
                "no_reranker_gap_threshold",
            )
        },
    )


def plan_campaign(
    *,
    config: Step8BenchmarkConfig,
    project_root: Path,
    phase: CampaignPhase,
    selection: str,
    campaign_dir: Path | None = None,
    freeze_path: Path | None = None,
) -> tuple[Path, BenchmarkCampaignManifest, SelectionExpansionManifest]:
    from pmmc.experiments.step8_exposure import (
        assert_compatible_exposure_history,
        recorded_exposure_query_ids,
    )
    from pmmc.experiments.step8_input_reconstruction import verify_reconstructed_inputs

    reconstruction = verify_reconstructed_inputs(config=config, project_root=project_root)
    if reconstruction is not None and (
        phase is not CampaignPhase.DEVELOPMENT or selection not in {"diagnostic", "regression"}
    ):
        raise ValueError("reconstructed inputs allow development diagnostic/regression only")
    protocol = make_protocol(config, project_root)
    partition, assignments = load_partition(_resolve(project_root, config.partition_dir))
    exposure = ExposureManifest.model_validate_json(
        _resolve(project_root, config.exposure_manifest).read_text(encoding="utf-8")
    )
    regression_ids = {}
    if phase is not CampaignPhase.EXPLORATION:
        regression = json.loads(_resolve(project_root, config.regression_manifest).read_text())
        regression_ids = {DatasetName(k): v for k, v in regression["query_ids"].items()}
    elif freeze_path is not None:
        raise ValueError("exploration cannot consume a paper freeze")
    freeze = None
    if phase is CampaignPhase.PAPER:
        if freeze_path is None:
            raise ValueError("paper requires an explicit MethodProtocolFreeze artifact")
        freeze = MethodProtocolFreeze.model_validate_json(freeze_path.read_text())
        if freeze.decision != "approved" or freeze.protocol != protocol:
            raise ValueError("paper code, configuration, or model identity differs from approval")
    cells = build_experiment_matrix(protocol)
    matrix_hash = canonical_content_hash([item.model_dump(mode="json") for item in cells])
    if campaign_dir is None:
        now = datetime.now(UTC)
        stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
        campaign_dir = _resolve(project_root, config.output_root) / (
            f"{stamp}_{phase.value}_{protocol.content_hash[:8]}"
        )
        manifest = BenchmarkCampaignManifest(
            campaign_id="campaign-"
            + canonical_content_hash(
                [
                    protocol.content_hash,
                    phase.value,
                    stamp,
                ]
            )[:24],
            phase=phase,
            protocol=protocol,
            partition_id=partition.partition_id,
            snapshot_ids={item.dataset: item.snapshot_id for item in partition.datasets},
            matrix_hash=matrix_hash,
            created_at=now,
            paper_result=phase is CampaignPhase.PAPER,
            diagnostic_only=selection == "diagnostic",
            protocol_freeze_hash=freeze.content_hash if freeze else None,
        )
    else:
        manifest = BenchmarkCampaignManifest.model_validate_json(
            (campaign_dir / "manifests/campaign.json").read_text()
        )
        if manifest.protocol != protocol or manifest.phase is not phase:
            raise ValueError("scientific identity changed; create a new versioned campaign")
        if manifest.matrix_hash != matrix_hash:
            raise ValueError("experiment matrix changed")
        if manifest.diagnostic_only != (selection == "diagnostic"):
            raise ValueError(
                "diagnostic campaign cannot be extended into the scientific denominator"
            )
    previous = None
    selection_paths = sorted((campaign_dir / "manifests/selections").glob("*.json"))
    if selection_paths:
        # The predecessor chain, not filesystem modification time, defines expansion order.
        selections = [
            SelectionExpansionManifest.model_validate_json(path.read_text())
            for path in selection_paths
        ]
        parents = {item.parent_selection_hash for item in selections}
        leaves = [item for item in selections if item.content_hash not in parents]
        if len(leaves) != 1:
            raise ValueError("selection history must have one unambiguous latest leaf")
        previous = leaves[0]
        if previous.selection == selection:
            assert_compatible_exposure_history(
                output_root=_resolve(project_root, config.output_root),
                manifest=manifest,
                selection=previous,
            )
            return campaign_dir, manifest, previous
    chosen = select_benchmark_queries(
        campaign_id=manifest.campaign_id,
        phase=phase,
        selection=selection,
        partition_hash=partition.content_hash,
        assignments=assignments,
        exposure=exposure,
        regression_ids=regression_ids,
        seed=protocol.seed,
        previous=previous,
        additional_exposed_ids=(
            recorded_exposure_query_ids(_resolve(project_root, config.output_root))
            if selection == "confirmation"
            else None
        ),
    )
    assert_compatible_exposure_history(
        output_root=_resolve(project_root, config.output_root),
        manifest=manifest,
        selection=chosen,
    )
    for directory in (
        "manifests/selections",
        "datasets",
        "cells",
        "attempts",
        "reports",
        "checkpoints",
        "private",
        "transfer",
    ):
        (campaign_dir / directory).mkdir(parents=True, exist_ok=True)
    immutable = {
        "manifests/campaign.json": manifest.model_dump(mode="json"),
        "manifests/config.json": config.model_dump(mode="json"),
        "manifests/matrix.json": [item.model_dump(mode="json") for item in cells],
        f"manifests/selections/{chosen.content_hash}.json": chosen.model_dump(mode="json"),
        "manifests/exposure.json": exposure.model_dump(mode="json"),
        "reports/denominators.json": denominator_report(assignments, exposure),
    }
    if freeze is not None:
        immutable["manifests/method-freeze.json"] = freeze.model_dump(mode="json")
    if reconstruction is not None:
        immutable["manifests/input-reconstruction.json"] = reconstruction
        immutable["reports/input-limitations.json"] = reconstruction["restrictions"]
    for relative, value in immutable.items():
        durable_write(campaign_dir / relative, json_bytes(value), immutable=True)
    for cell in cells:
        durable_write(
            campaign_dir / "cells" / cell.readable_name / "manifest.json",
            json_bytes(cell.model_dump(mode="json")),
            immutable=True,
        )
    return campaign_dir, manifest, chosen


class CampaignJobPlanError(ValueError):
    """A safe code for a conflicting or damaged immutable plan, without payloads."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _read_job_plan_file(directory: Path, path: Path) -> bytes | None:
    for item in (path, *path.parents):
        if item.is_symlink():
            raise CampaignJobPlanError("unsafe_campaign_job_plan_file")
        if item == directory:
            break
    if not path.exists():
        return None
    if not path.is_file():
        raise CampaignJobPlanError("unsafe_campaign_job_plan_file")
    return path.read_bytes()


def persist_campaign_job_plan(
    *, directory: Path, selection: SelectionExpansionManifest, jobs: list[BenchmarkStageJob]
) -> dict[str, Any]:
    """Reuse legacy inventory order only after checking every immutable job payload."""
    validate_job_dag(jobs)
    if not jobs or any(job.campaign_id != selection.campaign_id for job in jobs):
        raise CampaignJobPlanError("invalid_campaign_job_plan")
    ids = [job.job_id for job in jobs]
    inventory = directory / "manifests" / f"jobs-{selection.content_hash}.json"
    existing = _read_job_plan_file(directory, inventory)
    saved_ids = None
    if existing is not None:
        try:
            saved_ids = json.loads(existing)
        except (ValueError, UnicodeError):
            raise CampaignJobPlanError("invalid_campaign_job_inventory") from None
        if (
            not isinstance(saved_ids, list)
            or any(not isinstance(item, str) for item in saved_ids)
            or len(saved_ids) != len(set(saved_ids))
        ):
            raise CampaignJobPlanError("invalid_campaign_job_inventory")
        if set(saved_ids) != set(ids):
            raise CampaignJobPlanError("campaign_job_inventory_membership_conflict")

    # Validate all existing files before publishing anything. A missing inventory
    # can be an interrupted first plan; a published inventory must be complete.
    pending = []
    for job in jobs:
        path = directory / "private/jobs" / f"{job.job_id}.json"
        payload = json_bytes(job.model_dump(mode="json"))
        saved = _read_job_plan_file(directory, path)
        if saved is None:
            if existing is not None:
                raise CampaignJobPlanError("campaign_job_payload_missing")
            pending.append((path, payload))
        elif saved != payload:
            raise CampaignJobPlanError("campaign_job_payload_conflict")
    for path, payload in pending:
        durable_write(path, payload, immutable=True)
    if existing is None:
        durable_write(inventory, json_bytes(ids), immutable=True)
    return {
        "inventory_state": "reused" if existing is not None else "created",
        "jobs": len(ids),
        "created_job_files": len(pending),
        "reused_job_files": len(ids) - len(pending),
        "legacy_order_preserved": saved_ids is not None and saved_ids != ids,
    }


def build_campaign_jobs(
    *,
    manifest: BenchmarkCampaignManifest,
    selection: SelectionExpansionManifest,
    conversations: dict[DatasetName, dict[str, list[str]]],
    project_root: Path,
) -> list[BenchmarkStageJob]:
    """Conversation compilation is query-blind; online cells bind one selected query."""
    jobs: dict[str, BenchmarkStageJob] = {}
    proto = manifest.protocol

    def add(
        stage: str,
        scope: dict[str, Any],
        deps: list[BenchmarkStageJob],
        resource: ResourceClass,
        operation: str,
        parameters: dict[str, Any] | None = None,
    ) -> BenchmarkStageJob:
        owned_configs = {
            "questioner": ("questioner_config",),
            "planner0": ("planner_config", "foundation_config"),
            "doubter": ("doubter_config", "planner_config", "foundation_config"),
        }.get(stage, ())
        # One resource stage hosts several independent consumers. Bind config
        # values to the operation, not unrelated corpus/suite/compiler siblings.
        owned_configs += {
            "units": ("questioner_config",),
            "suite": ("baseline_full_context_config", "baseline_rag_config"),
            "baseline_index": ("baseline_rag_config",),
            "baseline_retrieve": ("baseline_rag_config",),
            "baseline_answer": ("baseline_full_context_config", "baseline_rag_config"),
            "baseline_grounding": ("baseline_full_context_config", "baseline_rag_config"),
            "native_fallback_generate": ("baseline_rag_config",),
            "native_fallback_ground": ("baseline_rag_config",),
            "primary_evaluate": ("evaluation_config",),
            "analysis_evaluate": ("evaluation_config",),
            "query_match": ("foundation_config", "query_executor_config"),
            "query_answer": ("foundation_config", "query_executor_config"),
            "query_grounding": ("foundation_config", "query_executor_config"),
            "evaluate": ("evaluation_config",),
        }.get(operation, ())
        if manifest.phase is CampaignPhase.EXPLORATION and operation == "query_embedding":
            owned_configs += ("query_executor_config",)
        if "partial_import_policy" in proto.config_hashes and operation in {
            "units", "compiler", "adopt_questioner",
        }:
            owned_configs += ("partial_import_policy",)
        inputs = {
            **scope,
            "model_entry_hashes": stage_model_entry_hashes(
                proto,
                stage=stage,
                operation=operation,
                scope=scope,
            ),
            "partition_hash": proto.partition_hash,
            "config_hashes": {name: proto.config_hashes[name] for name in owned_configs},
            "upstream_inputs": [item.scientific_input_hash for item in deps],
        }
        if (
            operation == "evaluate"
            and proto.oracle_at_50 is not None
            and scope.get("answer_model") == proto.oracle_at_50.answer_model
            and scope.get("treatment") == proto.oracle_at_50.treatment
        ):
            inputs["oracle_at_50"] = proto.oracle_at_50.model_dump(mode="json")
            inputs["config_hashes"]["foundation_config"] = proto.config_hashes["foundation_config"]
        if operation == "units" or stage == "questioner":
            # Preparation persists a run-scoped manifest even though unit
            # selection itself is deterministic; the compiler run consumes seed.
            inputs["seed"] = proto.seed
        if operation == "query_match":
            inputs["matching_policy"] = {
                name: getattr(proto, name)
                for name in (
                    "match_threshold",
                    "gap_threshold",
                    "text_relevance_threshold",
                    "multimodal_relevance_threshold",
                    "no_reranker_match_threshold",
                    "no_reranker_gap_threshold",
                )
            }
        key = stage + "-" + canonical_content_hash(inputs)[:24]
        job = BenchmarkStageJob(
            campaign_id=manifest.campaign_id,
            stage=stage,
            component_key=key,
            resource_class=resource,
            scientific_inputs=inputs,
            source_fingerprint=proto.source_fingerprints[stage],
            dependency_ids=[item.job_id for item in deps],
            operation=operation,
            parameters=parameters or scope,
        )
        jobs[job.job_id] = job
        return job

    # JSON round-tripping sorts dictionary keys; insertion order is not identity.
    for dataset in sorted(conversations):
        by_conversation = conversations[dataset]
        if sorted(query for ids in by_conversation.values() for query in ids) != sorted(
            selection.query_ids[dataset]
        ):
            raise ValueError("conversation assignments do not cover the selected denominator")
        snapshot_id = manifest.snapshot_ids[dataset]
        base: dict[str, Any] = {"dataset": dataset.value, "snapshot_id": snapshot_id}
        snapshot = add("snapshot", base, [], ResourceClass.CPU, "snapshot")
        corpus = add(
            "unit_prepare",
            {**base, "kind": "baseline_corpus"},
            [snapshot],
            ResourceClass.CPU,
            "corpus",
        )
        indexes = {}
        for stage, method in (
            ("text_index", "caption_text_rag"),
            ("contrastive_index", "contrastive_multimodal_rag"),
            ("multimodal_index", "native_multimodal_rag"),
        ):
            indexes[method] = add(
                stage,
                {**base, "method": method},
                [corpus],
                ResourceClass.GPU_EMBEDDING,
                "baseline_index",
            )
        caption_fallback_index = add(
            "multimodal_index",
            {**base, "method": "native_multimodal_rag", "treatment": "no_raw_image"},
            [corpus],
            ResourceClass.GPU_EMBEDDING,
            "baseline_index",
        )
        for conversation, query_ids in sorted(by_conversation.items()):
            projections: dict[Treatment, BenchmarkStageJob] = {}
            compiler_parents: dict[
                Treatment, tuple[BenchmarkStageJob, BenchmarkStageJob, BenchmarkStageJob]
            ] = {}
            shared_questioner = None
            for treatment in (Treatment.MAIN, Treatment.FIXED_PLANNER, Treatment.NO_RAW_IMAGE):
                scope = {**base, "conversation_id": conversation, "treatment": treatment.value}
                if treatment is Treatment.FIXED_PLANNER:
                    assert shared_questioner is not None
                    q = add(
                        "questioner",
                        scope,
                        [shared_questioner],
                        ResourceClass.CPU,
                        "adopt_questioner",
                    )
                else:
                    unit = add("unit_prepare", scope, [snapshot], ResourceClass.CPU, "units")
                    q = add("questioner", scope, [unit], ResourceClass.GPU_9B, "compiler")
                    if treatment is Treatment.MAIN:
                        shared_questioner = q
                p = add("planner0", scope, [q], ResourceClass.GPU_9B, "compiler")
                text_index = add("text_index", scope, [p], ResourceClass.GPU_EMBEDDING, "compiler")
                mm = add(
                    "multimodal_index", scope, [text_index], ResourceClass.GPU_EMBEDDING, "compiler"
                )
                dry = add("dry_run0", scope, [mm], ResourceClass.CPU, "compiler")
                r0 = add("r0_gate", scope, [dry], ResourceClass.CPU, "compiler")
                doubt = add("doubter", scope, [r0], ResourceClass.GPU_9B, "compiler")
                projections[treatment] = add(
                    "projection", scope, [doubt], ResourceClass.CPU, "compiler"
                )
                compiler_parents[treatment] = (p, text_index, mm)
            for query_id in sorted(query_ids):
                shared_main_match: BenchmarkStageJob | None = None
                primary_producer: BenchmarkStageJob | None = None
                for answer_model in proto.answer_backbones:
                    query_scope = {
                        **base,
                        "conversation_id": conversation,
                        "query_id": query_id,
                        "answer_model": answer_model,
                    }
                    resource = (
                        ResourceClass.GPU_9B
                        if answer_model == proto.answer_backbones[0]
                        else ResourceClass.GPU_27B
                        if answer_model == proto.answer_backbones[1]
                        else ResourceClass.API
                    )
                    suite = add(
                        "unit_prepare",
                        {**query_scope, "kind": "suite"},
                        [corpus],
                        ResourceClass.CPU,
                        "suite",
                    )
                    retrievals = []
                    for stage, method in (
                        ("text_index", "caption_text_rag"),
                        ("contrastive_index", "contrastive_multimodal_rag"),
                        ("multimodal_index", "native_multimodal_rag"),
                    ):
                        retrievals.append(
                            add(
                                stage,
                                {**query_scope, "kind": "retrieval", "method": method},
                                [suite, indexes[method]],
                                ResourceClass.GPU_EMBEDDING,
                                "baseline_retrieve",
                            )
                        )
                    answers = add(
                        "answers",
                        {**query_scope, "kind": "baseline"},
                        retrievals,
                        resource,
                        "baseline_answer",
                    )
                    grounded = add(
                        "grounding",
                        {**query_scope, "kind": "baseline"},
                        [answers],
                        ResourceClass.GPU_9B,
                        "baseline_grounding",
                    )
                    caption_retrieval = caption_grounded = None
                    if answer_model == proto.answer_backbones[0]:
                        caption_scope = {**query_scope, "treatment": "no_raw_image"}
                        caption_suite = add(
                            "unit_prepare",
                            {**caption_scope, "kind": "suite"},
                            [corpus],
                            ResourceClass.CPU,
                            "suite",
                        )
                        caption_retrieval = add(
                            "multimodal_index",
                            {
                                **caption_scope,
                                "kind": "retrieval",
                                "method": "native_multimodal_rag",
                            },
                            [caption_suite, caption_fallback_index],
                            ResourceClass.GPU_EMBEDDING,
                            "baseline_retrieve",
                        )
                        caption_answers = add(
                            "answers",
                            {**caption_scope, "kind": "baseline"},
                            [caption_retrieval],
                            resource,
                            "baseline_answer",
                        )
                        caption_grounded = add(
                            "grounding",
                            {**caption_scope, "kind": "baseline"},
                            [caption_answers],
                            ResourceClass.GPU_9B,
                            "baseline_grounding",
                        )
                    treatments = [Treatment.MAIN]
                    if answer_model == proto.answer_backbones[0]:
                        treatments += [
                            Treatment.FIXED_PLANNER,
                            Treatment.NO_RAW_IMAGE,
                            Treatment.NO_RERANKER,
                        ]
                    for treatment in treatments:
                        compiler_treatment = (
                            Treatment.MAIN if treatment is Treatment.NO_RERANKER else treatment
                        )
                        scope = {**query_scope, "treatment": treatment.value}
                        tx = add(
                            "text_index",
                            {**scope, "kind": "query_embedding"},
                            [compiler_parents[compiler_treatment][1]],
                            ResourceClass.GPU_EMBEDDING,
                            "query_embedding",
                        )
                        mm = add(
                            "multimodal_index",
                            {**scope, "kind": "query_embedding"},
                            [tx, compiler_parents[compiler_treatment][2]],
                            ResourceClass.GPU_EMBEDDING,
                            "query_embedding",
                        )
                        fallback_retrieval = retrievals[-1]
                        fallback_grounded = grounded
                        if treatment is Treatment.NO_RAW_IMAGE:
                            assert caption_retrieval is not None and caption_grounded is not None
                            fallback_retrieval, fallback_grounded = (
                                caption_retrieval,
                                caption_grounded,
                            )
                        match_deps = [mm, projections[compiler_treatment], fallback_retrieval]
                        if answer_model != proto.answer_backbones[0]:
                            assert shared_main_match is not None
                            match_deps.append(shared_main_match)
                        matched = add(
                            "matching",
                            scope,
                            match_deps,
                            ResourceClass.GPU_9B,
                            "query_match",
                        )
                        if (
                            answer_model == proto.answer_backbones[0]
                            and treatment is Treatment.MAIN
                        ):
                            shared_main_match = matched
                        executed = add(
                            "execution", scope, [matched], ResourceClass.CPU, "query_execute"
                        )
                        answer = add("answers", scope, [executed], resource, "query_answer")
                        paired_grounded = fallback_grounded
                        if manifest.phase is CampaignPhase.EXPLORATION:
                            native_answer = add("answers", scope, [answer, fallback_retrieval],
                                                resource, "native_fallback_generate")
                            fallback_grounded = add("grounding", scope, [native_answer],
                                                    ResourceClass.GPU_9B, "native_fallback_ground")
                        final = add(
                            "grounding",
                            scope,
                            [answer, fallback_grounded],
                            ResourceClass.GPU_9B,
                            "query_grounding",
                        )
                        if manifest.phase is CampaignPhase.EXPLORATION:
                            add("evaluation", scope, [final], ResourceClass.GPU_9B,
                                "analysis_evaluate")
                            add("evaluation", scope, [final, paired_grounded], ResourceClass.GPU_9B,
                                "evaluate")
                        else:
                            add("evaluation", scope, [final], ResourceClass.GPU_9B, "evaluate")
                        if (manifest.phase is CampaignPhase.EXPLORATION
                                and treatment is Treatment.MAIN):
                            # A narrower, independently identified run does not depend on
                            # baseline answer jobs or claim to score absent ablation cells.
                            primary = {**scope, "readout": "primary"}
                            deps = [mm, projections[treatment], fallback_retrieval]
                            if answer_model != proto.answer_backbones[0]:
                                assert primary_producer is not None
                                deps.append(primary_producer)
                            match = add("matching", primary, deps, ResourceClass.GPU_9B,
                                        "query_match")
                            if answer_model == proto.answer_backbones[0]:
                                primary_producer = match
                            execute = add("execution", primary, [match], ResourceClass.CPU,
                                          "query_execute")
                            generate = add("answers", primary, [execute], resource, "query_answer")
                            fallback = add("answers", primary, [generate, fallback_retrieval],
                                           resource, "native_fallback_generate")
                            ground = add("grounding", primary, [fallback], ResourceClass.GPU_9B,
                                         "native_fallback_ground")
                            select = add("grounding", primary, [generate, ground],
                                         ResourceClass.GPU_9B, "query_grounding")
                            add("evaluation", primary, [select], ResourceClass.GPU_9B,
                                "primary_evaluate")
    return list(jobs.values())
