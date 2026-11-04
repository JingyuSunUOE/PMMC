"""Deterministic conversation-level benchmark partitioning and pilot sampling."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pmmc.benchmarks.repositories import RuntimeQueryStore
from pmmc.benchmarks.snapshot import load_manifest
from pmmc.pilots.config import QuestionerPilotConfig
from pmmc.pilots.io import (
    file_sha256,
    read_json,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)
from pmmc.schemas import (
    BenchmarkPartitionAssignment,
    BenchmarkPartitionManifest,
    BenchmarkPartitionSpec,
    BenchmarkQuery,
    BenchmarkSelection,
    DatasetName,
    DatasetPartition,
    PartitionRole,
    QuestionerPilotSample,
)

SCENARIO_PROTOCOLS = {"scenario-balanced-five-fold-v2", "scenario-balanced-five-fold-v3"}
EXACT_PARTITION_PROTOCOL = "scenario-balanced-five-fold-v3"


def _group_hash(spec: BenchmarkPartitionSpec, dataset: DatasetName, conversation_id: str) -> str:
    return stable_hash(
        [spec.protocol_version, spec.seed, dataset.value, conversation_id]
    )


def _fold_cost(
    *,
    candidate_fold: int,
    group_size: int,
    group_categories: Counter[str],
    fold_counts: list[int],
    fold_categories: list[Counter[str]],
    target_count: float,
    category_targets: dict[str, float],
) -> float:
    counts = list(fold_counts)
    categories = [Counter(item) for item in fold_categories]
    counts[candidate_fold] += group_size
    categories[candidate_fold].update(group_categories)
    count_error = sum((value - target_count) ** 2 for value in counts)
    category_error = sum(
        (categories[fold].get(category, 0) - target) ** 2
        for category, target in category_targets.items()
        for fold in range(len(counts))
    )
    return count_error + category_error


def _assign_folds(
    records: list[BenchmarkQuery],
    spec: BenchmarkPartitionSpec,
) -> dict[str, int]:
    if (
        spec.protocol_version in SCENARIO_PROTOCOLS
        and records[0].query.dataset is DatasetName.MEMGALLERY
    ):
        return _assign_memgallery_scenario_folds(records, spec)
    by_conversation: dict[str, list[BenchmarkQuery]] = defaultdict(list)
    for record in records:
        by_conversation[record.query.conversation_id].append(record)
    total_categories = Counter(
        record.source.source_category or "unknown" for record in records
    )
    target_count = len(records) / spec.fold_count
    category_targets = {
        key: value / spec.fold_count for key, value in total_categories.items()
    }
    ordered = sorted(
        by_conversation.items(),
        key=lambda item: (
            -len(item[1]),
            _group_hash(spec, item[1][0].query.dataset, item[0]),
            item[0],
        ),
    )
    fold_counts = [0] * spec.fold_count
    fold_categories = [Counter[str]() for _ in range(spec.fold_count)]
    result: dict[str, int] = {}
    for conversation_id, values in ordered:
        categories = Counter(
            item.source.source_category or "unknown" for item in values
        )
        scores = [
            (
                _partition_cost(
                    spec=spec,
                    total_count=len(records),
                    total_categories=total_categories,
                    candidate_fold=fold,
                    group_size=len(values),
                    group_categories=categories,
                    fold_counts=fold_counts,
                    fold_categories=fold_categories,
                    target_count=target_count,
                    category_targets=category_targets,
                ),
                fold,
            )
            for fold in range(spec.fold_count)
        ]
        fold = min(scores)[1]
        result[conversation_id] = fold
        fold_counts[fold] += len(values)
        fold_categories[fold].update(categories)
    return result


def _partition_cost(
    *, spec: BenchmarkPartitionSpec, total_count: int, total_categories: Counter[str],
    candidate_fold: int, group_size: int, group_categories: Counter[str],
    fold_counts: list[int], fold_categories: list[Counter[str]],
    target_count: float, category_targets: dict[str, float],
) -> int | float:
    if spec.protocol_version != EXACT_PARTITION_PROTOCOL:
        return _fold_cost(
            candidate_fold=candidate_fold, group_size=group_size,
            group_categories=group_categories, fold_counts=fold_counts,
            fold_categories=fold_categories, target_count=target_count,
            category_targets=category_targets,
        )
    # Scale the entire squared-error objective by fold_count**2. Integer ties
    # cannot depend on Python's float-sum implementation or category ordering.
    n = spec.fold_count
    count_error = sum(
        (n * (value + (group_size if fold == candidate_fold else 0)) - total_count) ** 2
        for fold, value in enumerate(fold_counts)
    )
    category_error = sum(
        (n * (fold_categories[fold].get(category, 0)
              + (group_categories.get(category, 0) if fold == candidate_fold else 0))
         - total) ** 2
        for category, total in total_categories.items()
        for fold in range(n)
    )
    return count_error + category_error


def _assign_memgallery_scenario_folds(
    records: list[BenchmarkQuery],
    spec: BenchmarkPartitionSpec,
) -> dict[str, int]:
    """Place exactly four MemGallery scenarios in every frozen fold."""

    by_scenario: dict[str, list[BenchmarkQuery]] = defaultdict(list)
    scenario_conversations: dict[str, set[str]] = defaultdict(set)
    for record in records:
        scenario_id = record.source.scenario_id
        if scenario_id is None:
            raise ValueError("MemGallery scenario-balanced partition requires scenario_id")
        by_scenario[scenario_id].append(record)
        scenario_conversations[scenario_id].add(record.query.conversation_id)
    if len(by_scenario) != 20 or spec.fold_count != 5:
        raise ValueError(
            "scenario-balanced partition requires exactly 20 MemGallery scenarios"
        )

    scenario_target = len(by_scenario) // spec.fold_count
    total_categories = Counter(
        record.source.source_category or "unknown" for record in records
    )
    category_targets = {
        key: value / spec.fold_count for key, value in total_categories.items()
    }
    target_count = len(records) / spec.fold_count
    total_images = sum(bool(item.query.query_image_refs) for item in records)
    target_images = total_images / spec.fold_count
    fold_counts = [0] * spec.fold_count
    fold_images = [0] * spec.fold_count
    fold_categories = [Counter[str]() for _ in range(spec.fold_count)]
    fold_scenarios = [0] * spec.fold_count
    result: dict[str, int] = {}
    ordered = sorted(
        by_scenario.items(),
        key=lambda item: (
            -len(item[1]),
            stable_hash(
                [
                    spec.protocol_version,
                    spec.seed,
                    DatasetName.MEMGALLERY.value,
                    item[0],
                ]
            ),
            item[0],
        ),
    )
    for scenario_id, values in ordered:
        categories = Counter(item.source.source_category or "unknown" for item in values)
        image_count = sum(bool(item.query.query_image_refs) for item in values)
        candidates: list[tuple[int | float, int]] = []
        for fold in range(spec.fold_count):
            if fold_scenarios[fold] >= scenario_target:
                continue
            cost = _partition_cost(
                spec=spec,
                total_count=len(records),
                total_categories=total_categories,
                candidate_fold=fold,
                group_size=len(values),
                group_categories=categories,
                fold_counts=fold_counts,
                fold_categories=fold_categories,
                target_count=target_count,
                category_targets=category_targets,
            )
            projected_images = list(fold_images)
            projected_images[fold] += image_count
            if spec.protocol_version == EXACT_PARTITION_PROTOCOL:
                cost += sum(
                    (spec.fold_count * value - total_images) ** 2 for value in projected_images
                )
            else:
                cost += sum((value - target_images) ** 2 for value in projected_images)
            candidates.append((cost, fold))
        if not candidates:
            raise RuntimeError("MemGallery scenario fold capacity was exhausted")
        fold = min(candidates)[1]
        fold_scenarios[fold] += 1
        fold_counts[fold] += len(values)
        fold_images[fold] += image_count
        fold_categories[fold].update(categories)
        for conversation_id in sorted(scenario_conversations[scenario_id]):
            if conversation_id in result:
                raise ValueError("one MemGallery conversation cannot cross scenarios")
            result[conversation_id] = fold
    if fold_scenarios != [scenario_target] * spec.fold_count:
        raise RuntimeError("MemGallery scenario folds do not have equal cardinality")
    return result


def _eligible_records(snapshot_dir: Path, spec: BenchmarkPartitionSpec) -> list[BenchmarkQuery]:
    records = [
        item
        for item in RuntimeQueryStore(snapshot_dir).iter_records()
        if item.eligibility.eligible
        and item.eligibility.policy_version == spec.eligibility_policy_version
    ]
    if not records:
        raise ValueError(f"snapshot has no eligible partition records: {snapshot_dir}")
    return sorted(records, key=lambda item: item.query.query_id)


def _validate_memgallery_scenarios(records: list[BenchmarkQuery]) -> None:
    by_conversation: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.source.scenario_id is not None:
            by_conversation[record.query.conversation_id].add(record.source.scenario_id)
    conflicts = sorted(
        conversation_id
        for conversation_id, scenarios in by_conversation.items()
        if len(scenarios) > 1
    )
    if conflicts:
        raise ValueError(
            "MemGallery conversations cannot cross scenarios: " + ", ".join(conflicts)
        )


def _assignment_identity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "partition_id"}
        for row in rows
    ]


def _balance_report(
    partition_id: str,
    spec: BenchmarkPartitionSpec,
    rows: list[BenchmarkPartitionAssignment],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in sorted({item.dataset for item in rows}, key=lambda item: item.value):
        values = [item for item in rows if item.dataset is dataset]
        folds: list[dict[str, Any]] = []
        for fold in range(spec.fold_count):
            folded = [item for item in values if item.fold == fold]
            folds.append(
                {
                    "fold": fold,
                    "query_count": len(folded),
                    "conversation_count": len(
                        {item.conversation_id for item in folded}
                    ),
                    "scenario_count": len(
                        {item.scenario_id for item in folded if item.scenario_id is not None}
                    ),
                    "query_image_count": sum(item.has_query_image for item in folded),
                    "source_category_counts": dict(
                        sorted(Counter(item.source_category for item in folded).items())
                    ),
                }
            )
        datasets[dataset.value] = {
            "query_count": len(values),
            "development_fraction": (
                sum(item.role is PartitionRole.DEVELOPMENT for item in values)
                / len(values)
            ),
            "folds": folds,
        }
    return {
        "partition_id": partition_id,
        "protocol_version": spec.protocol_version,
        "datasets": datasets,
    }


def build_partition(
    *,
    snapshot_dirs: list[Path],
    output_root: Path,
    spec: BenchmarkPartitionSpec,
) -> tuple[Path, BenchmarkPartitionManifest]:
    if not snapshot_dirs:
        raise ValueError("partition build requires snapshots")
    manifests = [load_manifest(path) for path in snapshot_dirs]
    if len(manifests) != len({item.dataset for item in manifests}):
        raise ValueError("partition build cannot contain duplicate datasets")

    provisional: list[dict[str, Any]] = []
    dataset_inputs: list[dict[str, Any]] = []
    for snapshot_dir, manifest in sorted(
        zip(snapshot_dirs, manifests, strict=True),
        key=lambda item: item[1].dataset.value,
    ):
        records = _eligible_records(snapshot_dir, spec)
        if manifest.dataset is DatasetName.MEMGALLERY:
            _validate_memgallery_scenarios(records)
        folds = _assign_folds(records, spec)
        dataset_inputs.append(
            {
                "dataset": manifest.dataset.value,
                "snapshot_id": manifest.snapshot_id,
                "snapshot_content_hash": manifest.canonical_content_hash,
            }
        )
        for ordinal, record in enumerate(records):
            fold = folds[record.query.conversation_id]
            provisional.append(
                {
                    "dataset": manifest.dataset.value,
                    "snapshot_id": manifest.snapshot_id,
                    "query_id": record.query.query_id,
                    "conversation_id": record.query.conversation_id,
                    "scenario_id": record.source.scenario_id,
                    "source_category": record.source.source_category or "unknown",
                    "has_query_image": bool(record.query.query_image_refs),
                    "fold": fold,
                    "role": (
                        PartitionRole.DEVELOPMENT.value
                        if fold == spec.development_fold
                        else PartitionRole.PAPER_TEST.value
                    ),
                    "ordinal": ordinal,
                }
            )
    partition_identity = {
        "spec": spec.model_dump(mode="json"),
        "datasets": dataset_inputs,
        "assignments": provisional,
    }
    partition_id = "benchmark-partition-" + stable_hash(partition_identity)[:24]
    assignments = [
        BenchmarkPartitionAssignment(partition_id=partition_id, **row)
        for row in provisional
    ]
    assignment_rows = [item.model_dump(mode="json") for item in assignments]
    report = _balance_report(partition_id, spec, assignments)

    temporary = output_root / ".tmp" / partition_id
    if temporary.exists():
        import shutil

        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    write_jsonl(temporary / "assignments.jsonl", assignment_rows)
    write_json(temporary / "balance_report.json", report)
    dataset_partitions: list[DatasetPartition] = []
    for item in dataset_inputs:
        dataset = DatasetName(item["dataset"])
        values = [row for row in assignments if row.dataset is dataset]
        dataset_partitions.append(
            DatasetPartition(
                dataset=dataset,
                snapshot_id=str(item["snapshot_id"]),
                snapshot_content_hash=str(item["snapshot_content_hash"]),
                query_count=len(values),
                conversation_count=len({row.conversation_id for row in values}),
                development_query_count=sum(
                    row.role is PartitionRole.DEVELOPMENT for row in values
                ),
                paper_test_query_count=sum(
                    row.role is PartitionRole.PAPER_TEST for row in values
                ),
                assignment_hash=stable_hash(
                    _assignment_identity(
                        [row.model_dump(mode="json") for row in values]
                    )
                ),
            )
        )
    manifest_payload = {
        "partition_id": partition_id,
        "schema_version": (
            "0.2.0"
            if spec.protocol_version in SCENARIO_PROTOCOLS
            else "0.1.0"
        ),
        "spec": spec.model_dump(mode="json"),
        "datasets": [item.model_dump(mode="json") for item in dataset_partitions],
        "assignment_count": len(assignments),
        "assignments_sha256": file_sha256(temporary / "assignments.jsonl"),
        "balance_report_sha256": file_sha256(temporary / "balance_report.json"),
        "metadata": {},
    }
    partition_manifest = BenchmarkPartitionManifest(
        **manifest_payload,
        content_hash=stable_hash(manifest_payload),
    )
    write_json(
        temporary / "manifest.json",
        partition_manifest.model_dump(mode="json"),
    )
    destination = output_root / partition_id
    if destination.exists():
        existing = verify_partition(destination)
        if existing != partition_manifest:
            raise ValueError("partition ID conflicts with an existing bundle")
        import shutil

        shutil.rmtree(temporary)
        return destination, existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    return destination, partition_manifest


def load_partition(partition_dir: Path) -> tuple[
    BenchmarkPartitionManifest, list[BenchmarkPartitionAssignment]
]:
    manifest = BenchmarkPartitionManifest.model_validate(
        read_json(partition_dir / "manifest.json")
    )
    assignments = [
        BenchmarkPartitionAssignment.model_validate(row)
        for row in read_jsonl(partition_dir / "assignments.jsonl")
    ]
    return manifest, assignments


def verify_partition(partition_dir: Path) -> BenchmarkPartitionManifest:
    manifest, assignments = load_partition(partition_dir)
    if partition_dir.name != manifest.partition_id:
        raise ValueError("partition directory name does not match partition_id")
    if file_sha256(partition_dir / "assignments.jsonl") != manifest.assignments_sha256:
        raise ValueError("partition assignment checksum mismatch")
    if file_sha256(partition_dir / "balance_report.json") != manifest.balance_report_sha256:
        raise ValueError("partition balance report checksum mismatch")
    if len(assignments) != manifest.assignment_count:
        raise ValueError("partition assignment count mismatch")
    query_ids = [item.query_id for item in assignments]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("partition contains duplicate query IDs")
    roles: dict[tuple[DatasetName, str], set[PartitionRole]] = defaultdict(set)
    for item in assignments:
        if item.partition_id != manifest.partition_id:
            raise ValueError("partition assignment references another partition")
        if item.fold >= manifest.spec.fold_count:
            raise ValueError("partition assignment has an invalid fold")
        expected_role = (
            PartitionRole.DEVELOPMENT
            if item.fold == manifest.spec.development_fold
            else PartitionRole.PAPER_TEST
        )
        if item.role is not expected_role:
            raise ValueError("partition assignment role does not match its fold")
        roles[(item.dataset, item.conversation_id)].add(item.role)
    if any(len(values) != 1 for values in roles.values()):
        raise ValueError("a conversation crosses development and paper_test")
    if manifest.spec.protocol_version in SCENARIO_PROTOCOLS:
        memgallery = [
            item for item in assignments if item.dataset is DatasetName.MEMGALLERY
        ]
        scenario_folds: dict[str, set[int]] = defaultdict(set)
        for item in memgallery:
            if item.scenario_id is None:
                raise ValueError("scenario-balanced MemGallery assignment has no scenario")
            scenario_folds[item.scenario_id].add(item.fold)
        if any(len(folds) != 1 for folds in scenario_folds.values()):
            raise ValueError("a MemGallery scenario crosses partition folds")
        if len(scenario_folds) != 20:
            raise ValueError("scenario-balanced MemGallery partition must contain 20 scenarios")
        for fold in range(manifest.spec.fold_count):
            scenarios = {
                item.scenario_id
                for item in memgallery
                if item.fold == fold and item.scenario_id is not None
            }
            if len(scenarios) != 4:
                raise ValueError("each MemGallery fold must contain exactly four scenarios")
    for dataset in manifest.datasets:
        values = [item for item in assignments if item.dataset is dataset.dataset]
        if len(values) != dataset.query_count:
            raise ValueError("partition dataset query count mismatch")
        if len({item.conversation_id for item in values}) != dataset.conversation_count:
            raise ValueError("partition dataset conversation count mismatch")
        if sum(
            item.role is PartitionRole.DEVELOPMENT for item in values
        ) != dataset.development_query_count:
            raise ValueError("partition development query count mismatch")
        if stable_hash(
            _assignment_identity([item.model_dump(mode="json") for item in values])
        ) != dataset.assignment_hash:
            raise ValueError("partition dataset assignment hash mismatch")
    payload = manifest.model_dump(mode="json", exclude={"content_hash"})
    if stable_hash(payload) != manifest.content_hash:
        raise ValueError("partition manifest content hash mismatch")
    return manifest


def resolve_partition_selection(
    *,
    partition_dir: Path,
    role: PartitionRole,
    dataset: DatasetName,
    selection: BenchmarkSelection,
    snapshot_id: str | None = None,
) -> BenchmarkSelection:
    """Bind a normal benchmark selector to one frozen partition assignment."""

    manifest = verify_partition(partition_dir)
    _manifest, assignments = load_partition(partition_dir)
    dataset_manifest = next(
        (item for item in manifest.datasets if item.dataset is dataset),
        None,
    )
    if dataset_manifest is None:
        raise ValueError(f"partition does not contain {dataset.value}")
    if snapshot_id is not None and dataset_manifest.snapshot_id != snapshot_id:
        raise ValueError(
            "partition snapshot does not match the selected canonical snapshot"
        )
    if selection.datasets != [dataset]:
        raise ValueError("partition resolution requires a single-dataset selection")

    role_assignments = [
        item
        for item in assignments
        if item.dataset is dataset and item.role is role
    ]
    role_query_ids = {item.query_id for item in role_assignments}
    if not role_query_ids:
        raise ValueError(
            f"partition role {role.value} is empty for {dataset.value}"
        )
    requested = set(selection.query_ids)
    outside = requested - role_query_ids
    if outside:
        raise ValueError(
            "explicit query IDs fall outside the frozen partition role: "
            f"{sorted(outside)}"
        )
    resolved_query_ids = (
        list(selection.query_ids)
        if selection.query_ids
        else [item.query_id for item in sorted(role_assignments, key=lambda item: item.ordinal)]
    )
    metadata = {
        key: value
        for key, value in selection.metadata.items()
        if key not in {"partition_manifest_path", "partition_role"}
    }
    metadata.update(
        {
            "partition_id": manifest.partition_id,
            "partition_role": role.value,
            "partition_dataset_assignment_hash": dataset_manifest.assignment_hash,
        }
    )
    return selection.model_copy(
        update={
            "query_ids": resolved_query_ids,
            "metadata": metadata,
        }
    )


def _largest_remainder(
    counts: dict[str, int],
    requested: int,
) -> dict[str, int]:
    population = sum(counts.values())
    if requested > population:
        raise ValueError(
            f"pilot sample size {requested} exceeds development population {population}"
        )
    exact = {key: requested * value / population for key, value in counts.items()}
    allocation = {key: min(counts[key], math.floor(value)) for key, value in exact.items()}
    remaining = requested - sum(allocation.values())
    order = sorted(
        counts,
        key=lambda key: (-(exact[key] - math.floor(exact[key])), key),
    )
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < counts[key]:
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("largest-remainder allocation cannot satisfy sample size")
    return allocation


def _ensure_category_coverage(
    *,
    strata: dict[str, list[BenchmarkPartitionAssignment]],
    allocation: dict[str, int],
    requested: int,
) -> dict[str, int]:
    categories = {
        key.split("|query_image=", 1)[0]
        for key in strata
    }
    if requested < len(categories):
        return allocation
    category_keys = {
        category: sorted(
            key for key in strata if key.split("|query_image=", 1)[0] == category
        )
        for category in categories
    }
    for category in sorted(categories):
        if sum(allocation[key] for key in category_keys[category]) > 0:
            continue
        recipient = min(
            category_keys[category],
            key=lambda key: (-len(strata[key]), key),
        )
        donors = [
            key
            for key in sorted(strata)
            if allocation[key] > 0
            and sum(
                allocation[value]
                for value in category_keys[key.split("|query_image=", 1)[0]]
            )
            > 1
        ]
        if not donors:
            raise RuntimeError("cannot preserve pilot category coverage")
        donor = min(donors, key=lambda key: (-allocation[key], key))
        allocation[donor] -= 1
        allocation[recipient] += 1
    return allocation


def sample_development_queries(
    *,
    manifest: BenchmarkPartitionManifest,
    assignments: list[BenchmarkPartitionAssignment],
    config: QuestionerPilotConfig,
) -> list[QuestionerPilotSample]:
    if manifest.spec.protocol_version != config.partition_protocol:
        raise ValueError(
            "pilot sampling config does not match the frozen partition protocol"
        )
    samples: list[QuestionerPilotSample] = []
    for dataset_partition in manifest.datasets:
        values = [
            item
            for item in assignments
            if item.dataset is dataset_partition.dataset
            and item.role is PartitionRole.DEVELOPMENT
        ]
        if config.protocol_version == "questioner-quality-pilot-v4":
            sampler = (
                _sample_step7b_memgallery_development
                if dataset_partition.dataset is DatasetName.MEMGALLERY
                else _sample_step7b_memlens_development
            )
            samples.append(
                sampler(
                    manifest=manifest,
                    dataset_partition=dataset_partition,
                    values=values,
                    config=config,
                )
            )
            continue
        if (
            config.partition_protocol in SCENARIO_PROTOCOLS
            and dataset_partition.dataset is DatasetName.MEMGALLERY
        ):
            samples.append(
                _sample_memgallery_development(
                    manifest=manifest,
                    dataset_partition=dataset_partition,
                    values=values,
                    config=config,
                )
            )
            continue
        samples.append(
            _sample_generic_development(
                manifest=manifest,
                dataset_partition=dataset_partition,
                values=values,
                config=config,
            )
        )
    if config.partition_protocol in SCENARIO_PROTOCOLS:
        memgallery = next(
            (item for item in samples if item.dataset is DatasetName.MEMGALLERY),
            None,
        )
        if memgallery is None:
            raise ValueError(
                "scenario-balanced pilot sampling requires a MemGallery sample"
            )
        development_scenarios = {
            str(item.scenario_id)
            for item in assignments
            if item.dataset is DatasetName.MEMGALLERY
            and item.role is PartitionRole.DEVELOPMENT
            and item.scenario_id is not None
        }
        if len(development_scenarios) != 4:
            raise ValueError(
                "scenario-balanced pilot sampling requires four development scenarios"
            )
        quota_base, quota_remainder = divmod(
            config.queries_per_dataset,
            len(development_scenarios),
        )
        expected_scenario_quotas = sorted(
            [quota_base + 1] * quota_remainder
            + [quota_base] * (len(development_scenarios) - quota_remainder)
        )
        if (
            len(memgallery.scenario_counts) != 4
            or set(memgallery.scenario_counts) != development_scenarios
            or sum(memgallery.scenario_counts.values()) != config.queries_per_dataset
            or sorted(memgallery.scenario_counts.values())
            != expected_scenario_quotas
            or any(
                count > config.memgallery_max_queries_per_scenario
                for count in memgallery.scenario_counts.values()
            )
        ):
            raise ValueError(
                "MemGallery scenario-first pilot sample failed its frozen postcondition"
            )
    return samples


def _sample_generic_development(
    *,
    manifest: BenchmarkPartitionManifest,
    dataset_partition: DatasetPartition,
    values: list[BenchmarkPartitionAssignment],
    config: QuestionerPilotConfig,
) -> QuestionerPilotSample:
    strata: dict[str, list[BenchmarkPartitionAssignment]] = defaultdict(list)
    for item in values:
        key = f"{item.source_category}|query_image={str(item.has_query_image).lower()}"
        strata[key].append(item)
    allocations = _ensure_category_coverage(
        strata=strata,
        allocation=_largest_remainder(
            {key: len(items) for key, items in strata.items()},
            config.queries_per_dataset,
        ),
        requested=config.queries_per_dataset,
    )
    selected: list[BenchmarkPartitionAssignment] = []
    for key in sorted(strata):
        ordered = sorted(
            strata[key],
            key=lambda item: (
                stable_hash(
                    [
                        config.protocol_version,
                        config.partition_seed,
                        item.dataset.value,
                        item.query_id,
                    ]
                ),
                item.query_id,
            ),
        )
        selected.extend(ordered[: allocations[key]])
    return _sample_from_selected(
        manifest=manifest,
        dataset_partition=dataset_partition,
        selected=selected,
        config=config,
        scenario_counts={},
        paired_query_ids=(),
    )


def _sample_from_selected(
    *,
    manifest: BenchmarkPartitionManifest,
    dataset_partition: DatasetPartition,
    selected: list[BenchmarkPartitionAssignment],
    config: QuestionerPilotConfig,
    scenario_counts: dict[str, int],
    paired_query_ids: tuple[str, ...],
) -> QuestionerPilotSample:
    selected.sort(key=lambda item: item.query_id)
    query_ids = [item.query_id for item in selected]
    identity = {
        "partition_id": manifest.partition_id,
        "dataset": dataset_partition.dataset.value,
        "role": PartitionRole.DEVELOPMENT.value,
        "query_ids": query_ids,
        "scenario_counts": scenario_counts,
        "paired_query_ids_hash": stable_hash(list(paired_query_ids)),
        "config_hash": config.resolved_hash,
    }
    return QuestionerPilotSample(
        sample_id="questioner-pilot-sample-" + stable_hash(identity)[:24],
        partition_id=manifest.partition_id,
        dataset=dataset_partition.dataset,
        query_ids=query_ids,
        conversation_ids=sorted({item.conversation_id for item in selected}),
        stratum_counts=dict(
            sorted(
                Counter(
                    f"{item.source_category}|query_image={str(item.has_query_image).lower()}"
                    for item in selected
                ).items()
            )
        ),
        scenario_counts=scenario_counts,
        selection_hash=stable_hash(identity),
    )


def _step7_legacy_pilot_config(config: QuestionerPilotConfig) -> QuestionerPilotConfig:
    return config.model_copy(
        update={
            "protocol_version": "questioner-quality-pilot-v3",
            "queries_per_dataset": config.paired_step7_queries_per_dataset,
            "memgallery_max_queries_per_scenario": 3,
            "coverage_audit_count": 40,
        }
    )


def _sample_step7b_memlens_development(
    *,
    manifest: BenchmarkPartitionManifest,
    dataset_partition: DatasetPartition,
    values: list[BenchmarkPartitionAssignment],
    config: QuestionerPilotConfig,
) -> QuestionerPilotSample:
    legacy = _sample_generic_development(
        manifest=manifest,
        dataset_partition=dataset_partition,
        values=values,
        config=_step7_legacy_pilot_config(config),
    )
    selected_ids = set(legacy.query_ids)
    by_category: dict[str, list[BenchmarkPartitionAssignment]] = defaultdict(list)
    for item in values:
        by_category[item.source_category].append(item)
    if len(by_category) != 4:
        raise ValueError("Step 7B MEMLENS pilot requires exactly four development categories")
    selected = [item for item in values if item.query_id in selected_ids]
    counts = Counter(item.source_category for item in selected)
    quota = config.memlens_queries_per_category
    for category in sorted(by_category):
        if counts[category] > quota:
            raise ValueError("paired Step 7 MEMLENS sample exceeds the Step 7B category quota")
        candidates = sorted(
            (item for item in by_category[category] if item.query_id not in selected_ids),
            key=lambda item: (
                stable_hash(
                    ["step7b-paired-extension-v1", config.partition_seed, item.query_id]
                ),
                item.query_id,
            ),
        )
        needed = quota - counts[category]
        if len(candidates) < needed:
            raise ValueError("MEMLENS development category cannot satisfy Step 7B quota")
        selected.extend(candidates[:needed])
    final_counts = Counter(item.source_category for item in selected)
    if len(selected) != 16 or set(final_counts.values()) != {quota}:
        raise RuntimeError("Step 7B MEMLENS paired expansion did not close at 4/4/4/4")
    return _sample_from_selected(
        manifest=manifest,
        dataset_partition=dataset_partition,
        selected=selected,
        config=config,
        scenario_counts={},
        paired_query_ids=tuple(sorted(legacy.query_ids)),
    )


def _sample_step7b_memgallery_development(
    *,
    manifest: BenchmarkPartitionManifest,
    dataset_partition: DatasetPartition,
    values: list[BenchmarkPartitionAssignment],
    config: QuestionerPilotConfig,
) -> QuestionerPilotSample:
    legacy = _sample_memgallery_development(
        manifest=manifest,
        dataset_partition=dataset_partition,
        values=values,
        config=_step7_legacy_pilot_config(config),
    )
    selected_ids = set(legacy.query_ids)
    selected = [item for item in values if item.query_id in selected_ids]
    by_scenario: dict[str, list[BenchmarkPartitionAssignment]] = defaultdict(list)
    for item in values:
        if item.scenario_id is None:
            raise ValueError("Step 7B MemGallery assignment is missing its scenario")
        by_scenario[item.scenario_id].append(item)
    counts = Counter(str(item.scenario_id) for item in selected)
    for scenario in sorted(by_scenario):
        candidates = sorted(
            (item for item in by_scenario[scenario] if item.query_id not in selected_ids),
            key=lambda item: (
                stable_hash(
                    [
                        "step7b-paired-extension-v1",
                        config.partition_seed,
                        scenario,
                        item.query_id,
                    ]
                ),
                item.query_id,
            ),
        )
        needed = 4 - counts[scenario]
        if needed < 0 or len(candidates) < needed:
            raise ValueError("MemGallery scenario cannot satisfy Step 7B paired quota")
        selected.extend(candidates[:needed])
    final_counts = Counter(str(item.scenario_id) for item in selected)
    if len(selected) != 16 or len(final_counts) != 4 or set(final_counts.values()) != {4}:
        raise RuntimeError("Step 7B MemGallery paired expansion did not close at 4/4/4/4")
    return _sample_from_selected(
        manifest=manifest,
        dataset_partition=dataset_partition,
        selected=selected,
        config=config,
        scenario_counts=dict(sorted(final_counts.items())),
        paired_query_ids=tuple(sorted(legacy.query_ids)),
    )


def _sample_memgallery_development(
    *,
    manifest: BenchmarkPartitionManifest,
    dataset_partition: DatasetPartition,
    values: list[BenchmarkPartitionAssignment],
    config: QuestionerPilotConfig,
) -> QuestionerPilotSample:
    by_scenario: dict[str, list[BenchmarkPartitionAssignment]] = defaultdict(list)
    for item in values:
        if item.scenario_id is None:
            raise ValueError("MemGallery pilot assignments require scenario IDs")
        by_scenario[item.scenario_id].append(item)
    if len(by_scenario) != 4:
        raise ValueError("MemGallery development pilot requires exactly four scenarios")
    requested = config.queries_per_dataset
    if requested > sum(len(items) for items in by_scenario.values()):
        raise ValueError("MemGallery development population is smaller than the pilot")
    if requested > config.memgallery_max_queries_per_scenario * len(by_scenario):
        raise ValueError("MemGallery scenario cap cannot satisfy the pilot sample")

    scenario_order = sorted(
        by_scenario,
        key=lambda scenario: (
            stable_hash(
                [
                    config.protocol_version,
                    config.partition_seed,
                    DatasetName.MEMGALLERY.value,
                    scenario,
                ]
            ),
            scenario,
        ),
    )
    base, remainder = divmod(requested, len(scenario_order))
    allocation = {
        scenario: min(
            len(by_scenario[scenario]),
            config.memgallery_max_queries_per_scenario,
            base + (1 if ordinal < remainder else 0),
        )
        for ordinal, scenario in enumerate(scenario_order)
    }
    remaining = requested - sum(allocation.values())
    while remaining:
        progressed = False
        for scenario in scenario_order:
            capacity = min(
                len(by_scenario[scenario]),
                config.memgallery_max_queries_per_scenario,
            )
            if allocation[scenario] >= capacity:
                continue
            allocation[scenario] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("MemGallery scenario quotas cannot satisfy sample size")

    selected: list[BenchmarkPartitionAssignment] = []
    stratum_counts: Counter[str] = Counter()
    for scenario in scenario_order:
        scenario_values = by_scenario[scenario]
        strata: dict[str, list[BenchmarkPartitionAssignment]] = defaultdict(list)
        for item in scenario_values:
            key = f"{item.source_category}|query_image={str(item.has_query_image).lower()}"
            strata[key].append(item)
        stratum_allocation = _ensure_category_coverage(
            strata=strata,
            allocation=_largest_remainder(
                {key: len(items) for key, items in strata.items()},
                allocation[scenario],
            ),
            requested=allocation[scenario],
        )
        for key in sorted(strata):
            ordered = sorted(
                strata[key],
                key=lambda item: (
                    stable_hash(
                        [
                            config.protocol_version,
                            config.partition_seed,
                            scenario,
                            item.query_id,
                        ]
                    ),
                    item.query_id,
                ),
            )
            chosen = ordered[: stratum_allocation[key]]
            selected.extend(chosen)
            stratum_counts.update({f"{scenario}|{key}": len(chosen)})
    selected.sort(key=lambda item: item.query_id)
    query_ids = [item.query_id for item in selected]
    conversation_ids = sorted({item.conversation_id for item in selected})
    selected_scenario_counts = Counter(str(item.scenario_id) for item in selected)
    if any(
        selected_scenario_counts.get(scenario, 0) != count
        for scenario, count in allocation.items()
    ):
        raise RuntimeError("MemGallery selected queries do not match scenario quotas")
    # Preserve zero-count scenarios in small engineering samples. The frozen
    # sample identity describes the full four-scenario allocation, not only
    # scenarios that happened to receive a query.
    scenario_counts = dict(sorted(allocation.items()))
    identity = {
        "partition_id": manifest.partition_id,
        "dataset": dataset_partition.dataset.value,
        "role": PartitionRole.DEVELOPMENT.value,
        "query_ids": query_ids,
        "scenario_counts": scenario_counts,
        "config_hash": config.resolved_hash,
    }
    return QuestionerPilotSample(
        sample_id="questioner-pilot-sample-" + stable_hash(identity)[:24],
        partition_id=manifest.partition_id,
        dataset=dataset_partition.dataset,
        query_ids=query_ids,
        conversation_ids=conversation_ids,
        stratum_counts=dict(sorted(stratum_counts.items())),
        scenario_counts=scenario_counts,
        selection_hash=stable_hash(identity),
    )
