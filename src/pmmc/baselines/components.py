"""Execution and signed export of independently recoverable baseline components."""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from pmmc.baselines.rag_embedding import RAG_METHODS
from pmmc.baselines.rag_workflow import RagRunPhase
from pmmc.baselines.suite import (
    baseline_suite_child_run_id,
    create_baseline_suite,
    run_baseline_suite,
    verify_baseline_suite,
)
from pmmc.benchmarks.ids import stable_hash
from pmmc.evaluation.config import (
    evaluation_prompt_sha256,
    load_evaluation_config,
    load_judge_prompt,
)
from pmmc.infrastructure.database.baseline_component_repositories import (
    PostgresBaselineComponentRepository,
)
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.evaluation_repositories import (
    PostgresBaselineSuiteRepository,
)
from pmmc.infrastructure.export_verification import verify_experiment_export
from pmmc.infrastructure.object_store import create_runtime_object_store
from pmmc.infrastructure.settings import Settings
from pmmc.models.locking import load_model_lock
from pmmc.pilots.io import (
    canonical_json_bytes,
    file_sha256,
    read_json,
    write_json,
    write_jsonl,
)
from pmmc.pilots.io import (
    stable_hash as content_hash,
)
from pmmc.schemas import (
    BASELINE_COMPONENT_SCHEMA_VERSION,
    FOUNDATIONAL_BASELINE_METHODS,
    BaselineComponentImportRecord,
    BaselineComponentManifest,
    BaselineComponentState,
    BaselineComponentStatus,
    BaselineFamilyStatus,
    BaselineMethod,
    BaselineSuiteFamilyManifest,
    BaselineSuiteRunLink,
    BaselineSuiteStatus,
    BenchmarkSelection,
    DatasetName,
)

BASELINE_COMPONENT_EXPORT_VERSION = "baseline-component-export-v1"


def baseline_component_id(family_id: str, dataset: DatasetName, method: BaselineMethod) -> str:
    return (
        "baseline-component-"
        + stable_hash(
            {
                "schema_version": BASELINE_COMPONENT_SCHEMA_VERSION,
                "family_id": family_id,
                "dataset": dataset.value,
                "method": method.value,
            }
        )[:24]
    )


def make_family_components(
    manifest: BaselineSuiteFamilyManifest,
) -> list[BaselineComponentManifest]:
    selections = manifest.metadata.get("selections")
    corpus_ids = manifest.metadata.get("corpus_ids", {})
    if not isinstance(selections, dict):
        raise ValueError("baseline family metadata must freeze per-dataset selections")
    if not isinstance(corpus_ids, dict):
        raise ValueError("baseline family corpus_ids metadata must be an object")
    components: list[BaselineComponentManifest] = []
    for dataset in sorted(manifest.datasets, key=lambda item: item.value):
        selection = selections.get(dataset.value)
        if not isinstance(selection, dict):
            raise ValueError(f"baseline family is missing {dataset.value} selection")
        validated = BenchmarkSelection.model_validate(selection)
        if validated.datasets != [dataset]:
            raise ValueError("component selection must contain exactly its dataset")
        for method in sorted(manifest.methods, key=lambda item: item.value):
            component_id = baseline_component_id(manifest.family_id, dataset, method)
            component_payload = {
                "family_config_hash": manifest.config_hash,
                "dataset": dataset.value,
                "method": method.value,
                "snapshot_id": manifest.snapshot_ids[dataset],
                "selection_hash": manifest.selection_hashes[dataset],
            }
            components.append(
                BaselineComponentManifest(
                    component_id=component_id,
                    family_id=manifest.family_id,
                    dataset=dataset,
                    method=method,
                    snapshot_id=manifest.snapshot_ids[dataset],
                    selection_hash=manifest.selection_hashes[dataset],
                    selected_count=manifest.selected_counts[dataset],
                    corpus_id=(
                        str(corpus_ids[dataset.value])
                        if corpus_ids.get(dataset.value) is not None
                        else None
                    ),
                    scientific_config_hash=manifest.config_hash,
                    component_config_hash=stable_hash(component_payload),
                    created_at=manifest.created_at,
                    metadata={
                        "selection": selection,
                        "partition_id": manifest.partition_id,
                        "partition_role": "paper_test",
                        "paper_component": True,
                        "paper_result": False,
                    },
                )
            )
    return components


async def register_baseline_family(
    settings: Settings,
    manifest: BaselineSuiteFamilyManifest,
) -> list[BaselineComponentManifest]:
    components = make_family_components(manifest)
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session, session.begin():
            await PostgresBaselineComponentRepository(session).register_family(manifest, components)
        return components
    finally:
        await engine.dispose()


async def register_baseline_family_from_suites(
    settings: Settings,
    *,
    suite_ids: dict[DatasetName, str],
    model_lock_path: Path,
    judge_model_alias: str,
    metadata: dict[str, object] | None = None,
) -> tuple[BaselineSuiteFamilyManifest, list[BaselineComponentManifest]]:
    """Freeze two paired denominators into one portable 2x5 paper family."""

    expected_datasets = {DatasetName.MEMLENS, DatasetName.MEMGALLERY}
    if set(suite_ids) != expected_datasets:
        raise ValueError("paper baseline family requires both benchmark suites")
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session:
            suites = PostgresBaselineSuiteRepository(session)
            manifests = {
                dataset: await suites.get_manifest(suite_id)
                for dataset, suite_id in suite_ids.items()
            }
            states = {
                dataset: await suites.get_state(suite_id)
                for dataset, suite_id in suite_ids.items()
            }
            queries = {
                dataset: await suites.get_queries(suite_id)
                for dataset, suite_id in suite_ids.items()
            }
        ordered = [manifests[item] for item in sorted(manifests, key=lambda value: value.value)]
        first = ordered[0]
        common = (
            first.answer_model_alias,
            first.answer_prompt_hash,
            first.decoding_config,
            first.total_context_tokens,
            first.reserved_output_tokens,
            first.query_image_policy,
            first.evaluation_policy_version,
        )
        for dataset, manifest in manifests.items():
            if manifest.dataset is not dataset:
                raise ValueError("family suite dataset key differs from its manifest")
            if set(manifest.methods) != set(FOUNDATIONAL_BASELINE_METHODS):
                raise ValueError("family denominator suites must declare all five methods")
            if manifest.metadata.get("paper_result") is not False:
                raise ValueError("family denominator suites must not be paper results")
            if manifest.metadata.get("partition_role") != "paper_test":
                raise ValueError("paper components require a paper_test partition")
            current = (
                manifest.answer_model_alias,
                manifest.answer_prompt_hash,
                manifest.decoding_config,
                manifest.total_context_tokens,
                manifest.reserved_output_tokens,
                manifest.query_image_policy,
                manifest.evaluation_policy_version,
            )
            if current != common:
                raise ValueError("family suites disagree on a controlled variable")
            if states[dataset].selected_count != len(queries[dataset]):
                raise ValueError("family suite denominator count is inconsistent")
        partition_ids = {
            str(item.metadata.get("partition_id", "")) for item in manifests.values()
        }
        if len(partition_ids) != 1 or not next(iter(partition_ids)):
            raise ValueError("family suites must share one non-empty partition ID")
        partition_id = next(iter(partition_ids))
        assignment_hashes = {
            dataset.value: str(
                manifests[dataset].metadata.get("partition_dataset_assignment_hash", "")
            )
            for dataset in sorted(manifests, key=lambda value: value.value)
        }
        if any(not value for value in assignment_hashes.values()):
            raise ValueError("family suites are missing partition assignment hashes")

        lock = load_model_lock(model_lock_path)
        answer_revision = lock.get(first.answer_model_alias).commit_sha
        judge_revision = lock.get(judge_model_alias).commit_sha
        evaluation_config = load_evaluation_config(settings.resolved_evaluation_config_path)
        if evaluation_config.judge_model_alias != judge_model_alias:
            raise ValueError("family judge alias differs from evaluation configuration")
        judge_prompt_hash = evaluation_prompt_sha256(
            load_judge_prompt(evaluation_config, settings.project_root)
        )
        selection_hashes = {
            dataset: manifests[dataset].selection_hash for dataset in manifests
        }
        selected_counts = {
            dataset: len(queries[dataset]) for dataset in manifests
        }
        snapshot_ids = {dataset: manifests[dataset].snapshot_id for dataset in manifests}
        protocol_versions = {
            "suite": str(first.metadata.get("protocol_version")),
            "evaluation": evaluation_config.protocol_version,
            "component_export": BASELINE_COMPONENT_EXPORT_VERSION,
        }
        family_payload = {
            "schema_version": BASELINE_COMPONENT_SCHEMA_VERSION,
            "partition_id": partition_id,
            "partition_assignment_hash": stable_hash(assignment_hashes),
            "datasets": [item.value for item in sorted(manifests, key=lambda value: value.value)],
            "methods": [item.value for item in FOUNDATIONAL_BASELINE_METHODS],
            "snapshot_ids": {item.value: snapshot_ids[item] for item in snapshot_ids},
            "selection_hashes": {
                item.value: selection_hashes[item] for item in selection_hashes
            },
            "selected_counts": {
                item.value: selected_counts[item] for item in selected_counts
            },
            "answer_model_alias": first.answer_model_alias,
            "answer_model_revision": answer_revision,
            "judge_model_alias": judge_model_alias,
            "judge_model_revision": judge_revision,
            "answer_prompt_hash": first.answer_prompt_hash,
            "judge_prompt_hash": judge_prompt_hash,
            "decoding_config": first.decoding_config,
            "total_context_tokens": first.total_context_tokens,
            "reserved_output_tokens": first.reserved_output_tokens,
            "retrieval_top_k": 50,
            "answer_top_k": 5,
            "protocol_versions": protocol_versions,
        }
        config_hash = stable_hash(family_payload)
        family_id = "baseline-family-" + config_hash[:24]
        family_metadata = {
            "paper_result": False,
            "source_suite_ids": {
                item.value: suite_ids[item]
                for item in sorted(suite_ids, key=lambda value: value.value)
            },
            "selections": {
                item.value: manifests[item].metadata["selection"]
                for item in sorted(manifests, key=lambda value: value.value)
            },
            "corpus_ids": {
                item.value: manifests[item].metadata.get("corpus_id")
                for item in sorted(manifests, key=lambda value: value.value)
            },
            "partition_dataset_assignment_hashes": assignment_hashes,
            **dict(metadata or {}),
        }
        family_metadata["paper_result"] = False
        family = BaselineSuiteFamilyManifest(
            family_id=family_id,
            partition_id=partition_id,
            partition_assignment_hash=stable_hash(assignment_hashes),
            datasets=sorted(manifests, key=lambda value: value.value),
            methods=list(FOUNDATIONAL_BASELINE_METHODS),
            snapshot_ids=snapshot_ids,
            selection_hashes=selection_hashes,
            selected_counts=selected_counts,
            answer_model_alias=first.answer_model_alias,
            answer_model_revision=answer_revision,
            judge_model_alias=judge_model_alias,
            judge_model_revision=judge_revision,
            answer_prompt_hash=first.answer_prompt_hash,
            judge_prompt_hash=judge_prompt_hash,
            decoding_config=first.decoding_config,
            total_context_tokens=first.total_context_tokens,
            reserved_output_tokens=first.reserved_output_tokens,
            retrieval_top_k=50,
            answer_top_k=5,
            protocol_versions=protocol_versions,
            config_hash=config_hash,
            paper_mode=True,
            created_at=min(item.created_at for item in manifests.values()),
            metadata=family_metadata,
        )
        components = make_family_components(family)
        async with sessions() as session, session.begin():
            await PostgresBaselineComponentRepository(session).register_family(
                family, components
            )
        return family, components
    finally:
        await engine.dispose()


async def run_baseline_component(
    settings: Settings,
    *,
    component_id: str,
    phase: RagRunPhase = RagRunPhase.ALL,
) -> BaselineComponentState:
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session:
            repository = PostgresBaselineComponentRepository(session)
            component, state = await repository.get_component(component_id)
            family = await repository.get_family_manifest(component.family_id)
        if state.status in {
            BaselineComponentStatus.ANSWER_READY,
            BaselineComponentStatus.EXPORTED,
            BaselineComponentStatus.IMPORTED,
            BaselineComponentStatus.VERIFIED,
        }:
            return state
        selection = BenchmarkSelection.model_validate(component.metadata["selection"])
        suite, _directory = await create_baseline_suite(
            settings=settings,
            snapshot_id=component.snapshot_id,
            selection=selection,
            corpus_id=component.corpus_id,
            methods=[component.method],
            answer_model_alias=family.answer_model_alias,
            metadata={
                "baseline_family_id": component.family_id,
                "baseline_component_id": component.component_id,
                "partition_id": family.partition_id,
                "partition_role": "paper_test",
                "paper_component": True,
                "paper_result": False,
            },
        )
        run_id = baseline_suite_child_run_id(suite.suite_id, component.method)
        await run_baseline_suite(
            settings=settings,
            suite_id=suite.suite_id,
            phase=phase,
            methods=[component.method],
        )
        async with sessions() as session, session.begin():
            repository = PostgresBaselineComponentRepository(session)
            await repository.bind_run(
                component_id,
                suite_id=suite.suite_id,
                run_id=run_id,
                corpus_id=component.corpus_id,
            )
            _manifest, current = await repository.get_component(component_id)
            if phase is RagRunPhase.RETRIEVE:
                if component.method not in RAG_METHODS:
                    return current
                if current.status is BaselineComponentStatus.REGISTERED:
                    return await repository.transition_component(
                        component_id, BaselineComponentStatus.RETRIEVAL_READY
                    )
                return current
            if current.status is BaselineComponentStatus.REGISTERED:
                if component.method in RAG_METHODS:
                    current = await repository.transition_component(
                        component_id, BaselineComponentStatus.RETRIEVAL_READY
                    )
                return await repository.transition_component(
                    component_id, BaselineComponentStatus.ANSWER_READY
                )
            if current.status is BaselineComponentStatus.RETRIEVAL_READY:
                return await repository.transition_component(
                    component_id, BaselineComponentStatus.ANSWER_READY
                )
            return current
    finally:
        await engine.dispose()


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "bundle_manifest.json"
    ]


def verify_baseline_component_bundle(bundle_dir: Path) -> dict[str, Any]:
    manifest = read_json(bundle_dir / "bundle_manifest.json")
    if manifest.get("export_version") != BASELINE_COMPONENT_EXPORT_VERSION:
        raise ValueError("unsupported baseline component export version")
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("file_count") != len(files):
        raise ValueError("baseline component bundle has an invalid file inventory")
    listed = [str(item.get("path", "")) for item in files if isinstance(item, dict)]
    actual = [item["path"] for item in _file_inventory(bundle_dir)]
    if sorted(listed) != sorted(actual) or len(listed) != len(set(listed)):
        raise ValueError("baseline component bundle contents differ from its manifest")
    root = bundle_dir.resolve()
    for item in files:
        relative = str(item["path"])
        path = bundle_dir / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("baseline component bundle contains an unsafe path")
        if path.stat().st_size != item["size_bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"baseline component bundle file is corrupt: {relative}")
    content = {key: value for key, value in manifest.items() if key != "bundle_export_id"}
    if content_hash(content) != manifest.get("bundle_export_id"):
        raise ValueError("baseline component bundle ID is invalid")
    family = BaselineSuiteFamilyManifest.model_validate(
        read_json(bundle_dir / "family_manifest.json")
    )
    component = BaselineComponentManifest.model_validate(
        read_json(bundle_dir / "component_manifest.json")
    )
    if component.family_id != family.family_id:
        raise ValueError("component bundle family lineage differs")
    if component.component_id != manifest.get("component_id"):
        raise ValueError("component bundle manifest has the wrong component")
    runtime_export = bundle_dir / "runtime_export"
    runtime_manifest = verify_experiment_export(runtime_export)
    if runtime_manifest.get("run_id") != component.run_id:
        raise ValueError("component bundle runtime export has the wrong run")
    queries = (bundle_dir / "suite_queries.jsonl").read_text(encoding="utf-8").splitlines()
    if len(queries) != component.selected_count:
        raise ValueError("component bundle query denominator has the wrong size")
    forbidden = ("evaluation/", "query_gold", "gold_answer")
    if any(any(token in path.lower() for token in forbidden) for path in actual):
        raise ValueError("component runtime bundle contains evaluation or gold data")
    return manifest


async def export_baseline_component(
    settings: Settings,
    *,
    component_id: str,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    from pmmc.infrastructure.experiment_export import ExperimentExporter

    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    temporary = output_root / f".tmp-{component_id}-{uuid4().hex}"
    try:
        async with sessions() as session:
            repository = PostgresBaselineComponentRepository(session)
            component, state = await repository.get_component(component_id)
            family = await repository.get_family_manifest(component.family_id)
            if state.status not in {
                BaselineComponentStatus.ANSWER_READY,
                BaselineComponentStatus.EXPORTED,
            }:
                raise ValueError("component export requires answer-ready facts")
            if component.suite_id is None or component.run_id is None:
                raise ValueError("component export requires suite and run lineage")
            suites = PostgresBaselineSuiteRepository(session)
            suite = await suites.get_manifest(component.suite_id)
            queries = await suites.get_queries(component.suite_id)
        if state.status is BaselineComponentStatus.EXPORTED:
            candidates = sorted((output_root / component.family_id / component_id).glob("*"))
            for candidate in candidates:
                if candidate.is_dir():
                    report = verify_baseline_component_bundle(candidate)
                    if report["bundle_export_id"] == state.bundle_export_id:
                        return candidate, report
            raise ValueError("exported component bundle is missing from output root")
        temporary.mkdir(parents=True, exist_ok=False)
        write_json(temporary / "family_manifest.json", family.model_dump(mode="json"))
        bound_component = component.model_copy(
            update={"suite_id": suite.suite_id, "run_id": component.run_id}
        )
        write_json(
            temporary / "component_manifest.json",
            bound_component.model_dump(mode="json"),
        )
        write_json(temporary / "suite_manifest.json", suite.model_dump(mode="json"))
        write_jsonl(
            temporary / "suite_queries.jsonl",
            [item.model_dump(mode="json") for item in queries],
        )
        exporter = ExperimentExporter(
            sessions,
            runtime_store=create_runtime_object_store(settings),
            embedding_parquet_compression="zstd",
        )
        runtime_result = await exporter.export(
            component.run_id,
            temporary / ".runtime",
            include_referenced_model_artifacts=True,
        )
        runtime_manifest = read_json(runtime_result.manifest_path)
        shutil.copytree(runtime_result.export_dir, temporary / "runtime_export")
        shutil.rmtree(temporary / ".runtime")
        files = _file_inventory(temporary)
        content = {
            "schema_version": BASELINE_COMPONENT_SCHEMA_VERSION,
            "export_version": BASELINE_COMPONENT_EXPORT_VERSION,
            "family_id": family.family_id,
            "component_id": component.component_id,
            "dataset": component.dataset.value,
            "method": component.method.value,
            "scientific_config_hash": component.scientific_config_hash,
            "component_config_hash": component.component_config_hash,
            "runtime_export_id": runtime_result.export_id,
            "alembic_revision": runtime_manifest["alembic_revision"],
            "paper_component": True,
            "paper_result": False,
            "file_count": len(files),
            "files": files,
        }
        bundle_export_id = content_hash(content)
        bundle_manifest = {"bundle_export_id": bundle_export_id, **content}
        write_json(temporary / "bundle_manifest.json", bundle_manifest)
        bundle_sha256 = file_sha256(temporary / "bundle_manifest.json")
        destination = output_root / family.family_id / component_id / bundle_export_id
        if destination.exists():
            existing = verify_baseline_component_bundle(destination)
            if canonical_json_bytes(existing) != canonical_json_bytes(bundle_manifest):
                raise ValueError("existing component bundle ID has different content")
            shutil.rmtree(temporary)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(destination)
        async with sessions() as session, session.begin():
            await PostgresBaselineComponentRepository(session).transition_component(
                component_id,
                BaselineComponentStatus.EXPORTED,
                bundle_export_id=bundle_export_id,
                bundle_sha256=bundle_sha256,
            )
        return destination, verify_baseline_component_bundle(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        await engine.dispose()


async def import_baseline_component(
    settings: Settings,
    *,
    bundle_dir: Path,
    imported_by: str,
) -> BaselineComponentState:
    bundle = verify_baseline_component_bundle(bundle_dir)
    component = BaselineComponentManifest.model_validate(
        read_json(bundle_dir / "component_manifest.json")
    )
    suite = read_json(bundle_dir / "suite_manifest.json")
    bundle_sha256 = file_sha256(bundle_dir / "bundle_manifest.json")
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session, session.begin():
            repository = PostgresBaselineComponentRepository(session)
            current_manifest, state = await repository.get_component(component.component_id)
            if current_manifest.component_config_hash != component.component_config_hash:
                raise ValueError("imported component identity differs from registered family")
            if state.status is BaselineComponentStatus.REGISTERED:
                await repository.bind_run(
                    component.component_id,
                    suite_id=str(suite["suite_id"]),
                    run_id=str(component.run_id),
                    corpus_id=component.corpus_id,
                )
                if component.method in RAG_METHODS:
                    state = await repository.transition_component(
                        component.component_id,
                        BaselineComponentStatus.RETRIEVAL_READY,
                    )
                state = await repository.transition_component(
                    component.component_id,
                    BaselineComponentStatus.ANSWER_READY,
                )
            if state.status is BaselineComponentStatus.ANSWER_READY:
                state = await repository.transition_component(
                    component.component_id,
                    BaselineComponentStatus.EXPORTED,
                    bundle_export_id=str(bundle["bundle_export_id"]),
                    bundle_sha256=bundle_sha256,
                )
            if state.status is BaselineComponentStatus.EXPORTED:
                import_id = (
                    "baseline-component-import-"
                    + stable_hash(
                        {
                            "component_id": component.component_id,
                            "bundle_export_id": bundle["bundle_export_id"],
                        }
                    )[:24]
                )
                state = await repository.record_import(
                    BaselineComponentImportRecord(
                        import_id=import_id,
                        family_id=component.family_id,
                        component_id=component.component_id,
                        bundle_export_id=str(bundle["bundle_export_id"]),
                        bundle_sha256=bundle_sha256,
                        source_alembic_revision=str(bundle["alembic_revision"]),
                        imported_by=imported_by,
                    )
                )
            if state.status is BaselineComponentStatus.IMPORTED:
                state = await repository.transition_component(
                    component.component_id,
                    BaselineComponentStatus.VERIFIED,
                )
            if state.status is not BaselineComponentStatus.VERIFIED:
                raise ValueError("component import did not reach verified state")
            return state
    finally:
        await engine.dispose()


async def verify_baseline_family(settings: Settings, family_id: str) -> dict[str, Any]:
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session:
            repository = PostgresBaselineComponentRepository(session)
            manifest = await repository.get_family_manifest(family_id)
            state = await repository.get_family_state(family_id)
            components = await repository.list_components(family_id)
        expected = {
            (dataset.value, method.value)
            for dataset in manifest.datasets
            for method in manifest.methods
        }
        actual = {(item.dataset.value, item.method.value) for item, _state in components}
        verified = sum(
            item.status is BaselineComponentStatus.VERIFIED for _manifest, item in components
        )
        issues: list[str] = []
        if actual != expected:
            issues.append("family component matrix differs from manifest")
        if verified != state.verified_component_count:
            issues.append("family verified count differs from component facts")
        if state.status is BaselineFamilyStatus.READY and verified != len(expected):
            issues.append("ready family is missing verified components")
        return {
            "family_id": family_id,
            "status": state.status.value,
            "expected_component_count": len(expected),
            "verified_component_count": verified,
            "components": [
                {
                    "component_id": item.component_id,
                    "dataset": item.dataset.value,
                    "method": item.method.value,
                    "status": component_state.status.value,
                    "run_id": item.run_id,
                }
                for item, component_state in components
            ],
            "valid": not issues,
            "issues": issues,
        }
    finally:
        await engine.dispose()


async def verify_baseline_component(
    settings: Settings,
    component_id: str,
    *,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session:
            component, state = await PostgresBaselineComponentRepository(session).get_component(
                component_id
            )
        issues: list[str] = []
        suite_report: dict[str, Any] | None = None
        if component.suite_id is not None:
            suite_report = await verify_baseline_suite(settings, component.suite_id)
            if (
                state.status
                in {
                    BaselineComponentStatus.ANSWER_READY,
                    BaselineComponentStatus.EXPORTED,
                    BaselineComponentStatus.IMPORTED,
                    BaselineComponentStatus.VERIFIED,
                }
                and not suite_report["valid"]
            ):
                issues.append("component suite failed verification")
            if suite_report["selected_count"] != component.selected_count:
                issues.append("component suite denominator differs")
            if suite_report["methods"] != [component.method.value]:
                issues.append("component suite contains another method")
        elif state.status is not BaselineComponentStatus.REGISTERED:
            issues.append("advanced component state has no suite lineage")
        bundle: dict[str, Any] | None = None
        if bundle_dir is not None:
            bundle = verify_baseline_component_bundle(bundle_dir)
            if bundle["component_id"] != component_id:
                issues.append("component bundle belongs to another component")
            if state.bundle_export_id is not None and (
                bundle["bundle_export_id"] != state.bundle_export_id
            ):
                issues.append("component bundle differs from persisted export")
        return {
            "component_id": component_id,
            "family_id": component.family_id,
            "dataset": component.dataset.value,
            "method": component.method.value,
            "status": state.status.value,
            "suite_id": component.suite_id,
            "run_id": component.run_id,
            "selected_count": component.selected_count,
            "bundle_export_id": state.bundle_export_id,
            "suite": suite_report,
            "bundle": bundle,
            "valid": not issues,
            "issues": issues,
        }
    finally:
        await engine.dispose()


async def aggregate_baseline_family(
    settings: Settings,
    family_id: str,
) -> dict[str, Any]:
    """Bind ten verified component runs into two paired five-method suites."""

    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session:
            repository = PostgresBaselineComponentRepository(session)
            family = await repository.get_family_manifest(family_id)
            state = await repository.get_family_state(family_id)
            components = await repository.list_components(family_id)
        if state.status not in {
            BaselineFamilyStatus.ASSEMBLING,
            BaselineFamilyStatus.READY,
        }:
            raise ValueError("baseline family cannot be aggregated from its current state")
        if any(
            component_state.status is not BaselineComponentStatus.VERIFIED
            for _component, component_state in components
        ):
            raise ValueError("baseline family requires ten verified components")
        suites: dict[str, str] = {}
        for dataset in sorted(family.datasets, key=lambda item: item.value):
            dataset_components = [
                component
                for component, _component_state in components
                if component.dataset is dataset
            ]
            if len(dataset_components) != len(family.methods):
                raise ValueError("baseline family dataset matrix is incomplete")
            exemplar = dataset_components[0]
            selection = BenchmarkSelection.model_validate(exemplar.metadata["selection"])
            suite, _directory = await create_baseline_suite(
                settings=settings,
                snapshot_id=family.snapshot_ids[dataset],
                selection=selection,
                corpus_id=exemplar.corpus_id,
                methods=family.methods,
                answer_model_alias=family.answer_model_alias,
                metadata={
                    "baseline_family_id": family_id,
                    "partition_id": family.partition_id,
                    "partition_role": "paper_test",
                    "aggregate_family": True,
                    "paper_result": True,
                },
            )
            if suite.selection_hash != family.selection_hashes[dataset]:
                raise ValueError("aggregate suite denominator differs from family")
            async with sessions() as session, session.begin():
                suite_repository = PostgresBaselineSuiteRepository(session)
                suite_state = await suite_repository.get_state(suite.suite_id)
                if suite_state.status is BaselineSuiteStatus.CREATED:
                    await suite_repository.transition(suite.suite_id, BaselineSuiteStatus.RUNNING)
                links = {
                    item.method: item
                    for item in await suite_repository.get_run_links(suite.suite_id)
                }
                for component in sorted(dataset_components, key=lambda item: item.method.value):
                    if component.run_id is None:
                        raise ValueError("verified component has no run lineage")
                    if component.method not in links:
                        await suite_repository.link_run(
                            BaselineSuiteRunLink(
                                suite_id=suite.suite_id,
                                method=component.method,
                                run_id=component.run_id,
                                metadata={"baseline_component_id": component.component_id},
                            )
                        )
                refreshed = await suite_repository.get_state(suite.suite_id)
                if refreshed.status is BaselineSuiteStatus.RUNNING:
                    await suite_repository.transition(suite.suite_id, BaselineSuiteStatus.READY)
            report = await verify_baseline_suite(settings, suite.suite_id)
            if not report["valid"] or report["status"] != BaselineSuiteStatus.READY.value:
                raise ValueError("aggregate baseline suite failed verification")
            suites[dataset.value] = suite.suite_id
        async with sessions() as session, session.begin():
            await PostgresBaselineComponentRepository(session).complete_family(family_id)
        return {
            "family_id": family_id,
            "status": BaselineFamilyStatus.READY.value,
            "suite_ids": suites,
            "paper_result": True,
        }
    finally:
        await engine.dispose()


async def complete_baseline_family(settings: Settings, family_id: str) -> dict[str, Any]:
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session, session.begin():
            await PostgresBaselineComponentRepository(session).complete_family(family_id)
        return await verify_baseline_family(settings, family_id)
    finally:
        await engine.dispose()


def run_component_sync(
    settings: Settings,
    *,
    component_id: str,
    phase: RagRunPhase,
) -> BaselineComponentState:
    return asyncio.run(run_baseline_component(settings, component_id=component_id, phase=phase))


def run_managed_baseline_component(
    settings: Settings,
    *,
    component_id: str,
    phase: RagRunPhase,
    embedding_replicas: int = 1,
    answer_replicas: int = 1,
) -> BaselineComponentState:
    """Run one component while managing its model stages inside the current Pod."""

    from pmmc.runtime.manager import run_managed_command

    async def load() -> tuple[BaselineComponentManifest, BaselineComponentState]:
        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                return await PostgresBaselineComponentRepository(session).get_component(
                    component_id
                )
        finally:
            await engine.dispose()

    component, state = asyncio.run(load())
    if state.status in {
        BaselineComponentStatus.ANSWER_READY,
        BaselineComponentStatus.EXPORTED,
        BaselineComponentStatus.IMPORTED,
        BaselineComponentStatus.VERIFIED,
    }:
        return state

    def command(child_phase: RagRunPhase) -> list[str]:
        return [
            sys.executable,
            "-m",
            "pmmc",
            "baseline",
            "run-component",
            "--component-id",
            component_id,
            "--phase",
            child_phase.value,
            "--managed-child",
        ]

    if component.method in RAG_METHODS and phase in {
        RagRunPhase.ALL,
        RagRunPhase.RETRIEVE,
    }:
        logical_stage = {
            BaselineMethod.CAPTION_TEXT_RAG: "embedding_text",
            BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG: "embedding_contrastive",
            BaselineMethod.NATIVE_MULTIMODAL_RAG: "embedding_multimodal",
        }[component.method]
        run_managed_command(
            settings,
            profile_path=settings.resolved_runtime_profile_path,
            logical_stage=logical_stage,
            command=command(RagRunPhase.RETRIEVE),
            replica_count=embedding_replicas,
        )
        if phase is RagRunPhase.RETRIEVE:
            return asyncio.run(load())[1]
    answer_phase = RagRunPhase.ANSWER if component.method in RAG_METHODS else RagRunPhase.ALL
    run_managed_command(
        settings,
        profile_path=settings.resolved_runtime_profile_path,
        logical_stage="answer",
        command=command(answer_phase),
        replica_count=answer_replicas,
    )
    return asyncio.run(load())[1]


def evaluate_baseline_family(
    settings: Settings,
    family_id: str,
    *,
    judge_replicas: int = 4,
) -> dict[str, Any]:
    """Aggregate ten components, then evaluate both paired suites uniformly."""

    from pmmc.evaluation.export import EvaluationExporter
    from pmmc.evaluation.workflow import (
        EvaluationRunPhase,
        run_evaluation_workflow,
        verify_evaluation_run,
    )
    from pmmc.runtime.manager import run_managed_commands

    if judge_replicas <= 0:
        raise ValueError("judge_replicas must be positive")
    aggregate = asyncio.run(aggregate_baseline_family(settings, family_id))
    suite_ids = {
        str(dataset): str(suite_id)
        for dataset, suite_id in dict(aggregate["suite_ids"]).items()
    }
    evaluation_ids: dict[str, str] = {}
    for dataset, suite_id in sorted(suite_ids.items()):
        manifest, _directory = asyncio.run(
            run_evaluation_workflow(
                settings=settings,
                suite_id=suite_id,
                phase=EvaluationRunPhase.DETERMINISTIC,
            )
        )
        evaluation_ids[dataset] = manifest.evaluation_run_id

    run_managed_commands(
        settings,
        profile_path=settings.resolved_runtime_profile_path,
        logical_stage="judge",
        commands=[
            [
                sys.executable,
                "-m",
                "pmmc",
                "evaluate",
                "run",
                "--resume-evaluation-run-id",
                evaluation_id,
                "--phase",
                EvaluationRunPhase.JUDGE.value,
            ]
            for _dataset, evaluation_id in sorted(evaluation_ids.items())
        ],
        replica_count=judge_replicas,
    )

    exports: dict[str, dict[str, object]] = {}
    for dataset, evaluation_id in sorted(evaluation_ids.items()):
        asyncio.run(
            run_evaluation_workflow(
                settings=settings,
                resume_evaluation_run_id=evaluation_id,
                phase=EvaluationRunPhase.ANALYZE,
            )
        )
        verification = asyncio.run(
            verify_evaluation_run(
                settings,
                evaluation_id,
                require_managed_runtime=True,
            )
        )
        if not verification["valid"]:
            raise ValueError(f"aggregate evaluation failed verification: {dataset}")
        exported = asyncio.run(EvaluationExporter(settings).export(evaluation_id))
        exports[dataset] = {
            "suite_id": suite_ids[dataset],
            "evaluation_run_id": evaluation_id,
            "export_id": exported.export_id,
            "export_dir": str(exported.export_dir),
        }
    return {
        **aggregate,
        "evaluation": exports,
        "judge_replicas": judge_replicas,
        "paper_result": True,
    }
