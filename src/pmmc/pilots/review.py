"""Deterministic private review tasks and human annotation validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from pmmc.benchmarks.repositories import GoldStore, MemoryCorpus, RuntimeQueryStore
from pmmc.pilots.config import QuestionerPilotConfig
from pmmc.pilots.io import read_jsonl, stable_hash
from pmmc.schemas import (
    BenchmarkPartitionAssignment,
    CandidateQualityAnnotation,
    CoverageQualityAnnotation,
    DatasetName,
    ImageRef,
    QuestionerReviewTask,
    ReviewAssignment,
    ReviewRole,
    ReviewTaskKind,
)


def _review_image_reference(dataset: DatasetName, image: ImageRef) -> dict[str, object]:
    dataset_relative = image.metadata.get("dataset_relative_path")
    if not isinstance(dataset_relative, str) or not dataset_relative:
        raise ValueError("review image is missing its dataset-relative path")
    relative = PurePosixPath(dataset_relative.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("review image dataset-relative path is unsafe")
    metadata_dataset = image.metadata.get("dataset")
    if metadata_dataset is not None and metadata_dataset != dataset.value:
        raise ValueError("review image belongs to another benchmark")
    return {
        "image_id": image.image_id,
        "sha256": image.sha256,
        "relative_path": (PurePosixPath(dataset.value) / relative).as_posix(),
    }


def _stable_pick(
    values: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    label: str,
    stratify_by: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    def order(item: dict[str, Any]) -> tuple[str, str]:
        source_id = str(item.get("qbank_id") or item.get("query_id"))
        return stable_hash([seed, label, source_id]), source_id

    if not stratify_by:
        return sorted(values, key=order)[:count]
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in values:
        groups[
            tuple(
                ",".join(sorted(str(entry) for entry in value))
                if isinstance((value := item.get(field)), list)
                else str(value)
                for field in stratify_by
            )
        ].append(item)
    ordered = {key: sorted(items, key=order) for key, items in groups.items()}
    positions = {key: 0 for key in ordered}
    result: list[dict[str, Any]] = []
    while len(result) < min(count, len(values)):
        progressed = False
        for key in sorted(ordered):
            position = positions[key]
            if position >= len(ordered[key]):
                continue
            result.append(ordered[key][position])
            positions[key] += 1
            progressed = True
            if len(result) == min(count, len(values)):
                break
        if not progressed:
            break
    return result


def _candidate_payloads(
    *,
    dataset: DatasetName,
    runtime_export: Path,
    snapshot_dir: Path,
) -> list[dict[str, Any]]:
    runtime = runtime_export / "runtime"
    questions = read_jsonl(runtime / "qbank_questions.jsonl")
    gates = {
        str(item["qbank_id"]): item
        for item in read_jsonl(runtime / "questioner_candidate_gate_results.jsonl")
    }
    evidence_by_qbank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in read_jsonl(runtime / "qbank_evidence_refs.jsonl"):
        evidence_by_qbank[str(item["qbank_id"])].append(item)
    duplicate_by_qbank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in read_jsonl(runtime / "qbank_duplicate_links.jsonl"):
        duplicate_by_qbank[str(item["duplicate_qbank_id"])].append(item)
        duplicate_by_qbank[str(item["primary_qbank_id"])].append(item)
    verification_path = runtime / "questioner_candidate_verifications.jsonl"
    verification_by_qbank = {
        str(item["qbank_id"]): item
        for item in (read_jsonl(verification_path) if verification_path.exists() else [])
    }
    opportunity_path = runtime / "questioner_opportunities.jsonl"
    opportunity_by_id = {
        str(item["opportunity_id"]): item
        for item in (read_jsonl(opportunity_path) if opportunity_path.exists() else [])
    }
    memories = {item.memory_id: item for item in MemoryCorpus(snapshot_dir).iter_all()}
    image_owners = {
        image.image_id: (memory, image)
        for memory in memories.values()
        for image in memory.image_refs
    }
    result: list[dict[str, Any]] = []
    for question in questions:
        qbank_id = str(question["qbank_id"])
        gate = gates.get(qbank_id, {})
        duplicate_rows = duplicate_by_qbank.get(qbank_id, [])
        is_duplicate = bool(gate.get("duplicate_of_qbank_id")) or any(
            item.get("hard_duplicate") is True for item in duplicate_rows
        )
        bucket = (
            "duplicate_diagnostic"
            if is_duplicate
            else "eligible"
            if gate.get("status") == "eligible_for_planning"
            else "rejected"
        )
        evidence: list[dict[str, Any]] = []
        seen_evidence: set[tuple[str, str]] = set()
        for ref in evidence_by_qbank.get(qbank_id, []):
            evidence_key = (str(ref.get("evidence_type")), str(ref.get("evidence_id")))
            if evidence_key in seen_evidence:
                continue
            seen_evidence.add(evidence_key)
            memory_id = ref.get("memory_id")
            memory = memories.get(str(memory_id)) if memory_id else None
            image_id = ref.get("image_id")
            direct_image = image_owners.get(str(image_id)) if image_id else None
            if memory is None and direct_image is not None:
                memory = direct_image[0]
            review_images = (
                [_review_image_reference(dataset, direct_image[1])]
                if direct_image is not None
                else (
                    [_review_image_reference(dataset, image) for image in memory.image_refs]
                    if memory is not None
                    else []
                )
            )
            evidence.append(
                {
                    "evidence_id": ref.get("evidence_id"),
                    "evidence_type": ref.get("evidence_type"),
                    "memory_id": memory_id,
                    "image_id": image_id,
                    "session_id": ref.get("session_id"),
                    "caption_id": ref.get("caption_id"),
                    "text": memory.text if memory is not None else None,
                    "caption": memory.caption if memory is not None else None,
                    "image_refs": review_images,
                }
            )
        result.append(
            {
                "qbank_id": qbank_id,
                "bucket": bucket,
                "question_type": question.get("question_type"),
                "reasoning_type": question.get("reasoning_type"),
                "required_modalities": question.get("required_modalities", []),
                "hop_count": question.get("hop_count"),
                "canonical_question": question.get("canonical_question"),
                "expected_answer": question.get("expected_answer"),
                "candidate_origin": question.get("candidate_origin", "generated"),
                "candidate_version": question.get("candidate_version", 0),
                "opportunity_type": opportunity_by_id.get(
                    str(question.get("opportunity_id")), {}
                ).get("opportunity_type"),
                "verification": verification_by_qbank.get(qbank_id),
                "gate_status": gate.get("status"),
                "rejection_codes": gate.get("rejection_codes", []),
                "duplicate_links": duplicate_rows,
                "expected_evidence": evidence,
            }
        )
    return result


def _coverage_payloads(
    *,
    dataset: DatasetName,
    coverage_export: Path,
    runtime_export: Path,
    snapshot_dir: Path,
) -> list[dict[str, Any]]:
    private = coverage_export / "private"
    queries = read_jsonl(private / "coverage_queries.jsonl")
    labels = read_jsonl(private / "coverage_labels.jsonl")
    candidates = read_jsonl(private / "coverage_candidates.jsonl")
    questions = {
        str(item["qbank_id"]): item
        for item in read_jsonl(runtime_export / "runtime" / "qbank_questions.jsonl")
    }
    candidate_cards = {
        str(item["qbank_id"]): item
        for item in _candidate_payloads(
            dataset=dataset,
            runtime_export=runtime_export,
            snapshot_dir=snapshot_dir,
        )
    }
    runtime_queries = {
        item.query.query_id: item for item in RuntimeQueryStore(snapshot_dir).iter_records()
    }
    gold_store = GoldStore(snapshot_dir)
    result: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query["query_id"])
        query_labels = sorted(
            {
                str(item["category"])
                for item in labels
                if item.get("query_id") == query_id
            }
        )
        bucket = (
            "questioner_coverage_failure"
            if "questioner_coverage_failure" in query_labels
            else "matching_failure"
            if "matching_failure" in query_labels
            else "gold_unresolvable"
            if "gold_unresolvable" in query_labels
            else "success"
        )
        ranked = [
            item
            for item in candidates
            if item.get("query_id") == query_id
            and item.get("question_budget") == 8
            and int(item.get("rank", 9999)) <= 50
        ]
        ranked.sort(key=lambda item: int(item["rank"]))
        runtime_query = runtime_queries[query_id]
        gold = gold_store.get(query_id)
        result.append(
            {
                "query_id": query_id,
                "bucket": bucket,
                "raw_query": runtime_query.query.raw_query,
                "query_image_refs": [
                    _review_image_reference(dataset, item)
                    for item in runtime_query.query.query_image_refs
                ],
                "gold_answer": gold.answer,
                "gold_evidence": gold.evidence.model_dump(mode="json"),
                "automatic_labels": query_labels,
                "ranked_candidates": [
                    {
                        "rank": item.get("rank"),
                        "qbank_id": item.get("qbank_id"),
                        "question": questions.get(str(item.get("qbank_id")), {}).get(
                            "canonical_question"
                        ),
                        "expected_answer": candidate_cards.get(
                            str(item.get("qbank_id")), {}
                        ).get("expected_answer"),
                        "expected_evidence": candidate_cards.get(
                            str(item.get("qbank_id")), {}
                        ).get("expected_evidence", []),
                        "question_type": candidate_cards.get(
                            str(item.get("qbank_id")), {}
                        ).get("question_type"),
                        "required_modalities": candidate_cards.get(
                            str(item.get("qbank_id")), {}
                        ).get("required_modalities", []),
                        "matched_gold_group_ids": item.get(
                            "matched_gold_group_ids", []
                        ),
                        "individually_sufficient": item.get(
                            "individually_sufficient", False
                        ),
                    }
                    for item in ranked
                ],
            }
        )
    return result


def _select_coverage_review_payloads(
    *,
    dataset: DatasetName,
    coverage: list[dict[str, Any]],
    config: QuestionerPilotConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the frozen review denominator with deterministic quota reallocation."""
    per_dataset = config.coverage_review_count // 2
    query_ids = [str(item.get("query_id") or "") for item in coverage]
    if any(not query_id for query_id in query_ids):
        raise ValueError(f"coverage query is missing its ID in {dataset.value}")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError(f"coverage queries contain duplicate IDs in {dataset.value}")
    if len(coverage) < per_dataset:
        raise ValueError(
            f"insufficient unique coverage queries for review in {dataset.value}: "
            f"required={per_dataset} available={len(coverage)}"
        )

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in coverage:
        by_bucket[str(item["bucket"])].append(item)
    selected: list[dict[str, Any]] = []
    selected_query_ids: set[str] = set()

    def add(values: list[dict[str, Any]], count: int, *, label: str) -> int:
        if count <= 0:
            return 0
        available = [
            item
            for item in values
            if str(item["query_id"]) not in selected_query_ids
        ]
        picked = _stable_pick(
            available,
            count,
            seed=config.partition_seed,
            label=label,
        )
        selected.extend(picked)
        selected_query_ids.update(str(item["query_id"]) for item in picked)
        return len(picked)

    priority_caps = {
        "questioner_coverage_failure": config.coverage_failure_reviews_per_dataset,
        "matching_failure": config.matching_failure_reviews_per_dataset,
    }
    for bucket, count in priority_caps.items():
        add(
            by_bucket[bucket],
            count,
            label=f"{dataset.value}:coverage:{bucket}",
        )

    preferred_fill_count = add(
        [
            item
            for item in coverage
            if item["bucket"] in {"success", "gold_unresolvable"}
        ],
        per_dataset - len(selected),
        label=f"{dataset.value}:coverage:preferred-remainder",
    )
    fallback_fill_count = add(
        coverage,
        per_dataset - len(selected),
        label=f"{dataset.value}:coverage:any-remainder",
    )
    if len(selected) != per_dataset:
        raise ValueError(
            f"insufficient unique coverage queries for review in {dataset.value}: "
            f"required={per_dataset} selected={len(selected)}"
        )

    available_counts = Counter(str(item["bucket"]) for item in coverage)
    selected_counts = Counter(str(item["bucket"]) for item in selected)
    return selected, {
        "requested_total": per_dataset,
        "available_total": len(coverage),
        "available_by_bucket": dict(sorted(available_counts.items())),
        "priority_caps": priority_caps,
        "selected_by_bucket": dict(sorted(selected_counts.items())),
        "preferred_fill_count": preferred_fill_count,
        "fallback_fill_count": fallback_fill_count,
    }


def build_review_assignments(
    *,
    pilot_id: str,
    config: QuestionerPilotConfig,
    tasks: list[QuestionerReviewTask],
) -> list[ReviewAssignment]:
    expected_count = config.candidate_review_count + config.coverage_review_count
    if len(tasks) != expected_count:
        raise ValueError("review task count does not match the frozen pilot config")
    if len({item.task_id for item in tasks}) != len(tasks):
        raise ValueError("review tasks contain duplicate IDs")
    if any(item.pilot_id != pilot_id for item in tasks):
        raise ValueError("review tasks belong to another pilot")
    if config.protocol_version == "questioner-quality-pilot-v3":
        return []
    ordered_tasks = sorted(tasks, key=lambda item: item.task_id)
    primary_ids = [item.task_id for item in ordered_tasks]
    primary_hash = stable_hash(primary_ids)
    primary = ReviewAssignment(
        assignment_id="review-assignment-" + primary_hash[:24],
        pilot_id=pilot_id,
        role=ReviewRole.PRIMARY,
        reviewer_slot="primary",
        task_ids=primary_ids,
        assignment_hash=primary_hash,
    )
    groups: dict[tuple[str, str, str], list[QuestionerReviewTask]] = defaultdict(list)
    for task in ordered_tasks:
        groups[(task.dataset.value, task.kind.value, task.bucket)].append(task)
    secondary_tasks: list[QuestionerReviewTask] = []
    ordered_groups = sorted(groups)
    positions = {key: 0 for key in ordered_groups}
    while len(secondary_tasks) < config.secondary_review_count:
        progressed = False
        for key in ordered_groups:
            ordered = sorted(
                groups[key],
                key=lambda item: (
                    stable_hash([pilot_id, "secondary", item.task_id]),
                    item.task_id,
                ),
            )
            position = positions[key]
            if position < len(ordered):
                secondary_tasks.append(ordered[position])
                positions[key] += 1
                progressed = True
                if len(secondary_tasks) == config.secondary_review_count:
                    break
        if not progressed:
            raise ValueError("cannot allocate the secondary blind-review sample")
    secondary_ids = sorted(item.task_id for item in secondary_tasks)
    secondary_hash = stable_hash(secondary_ids)
    secondary = ReviewAssignment(
        assignment_id="review-assignment-" + secondary_hash[:24],
        pilot_id=pilot_id,
        role=ReviewRole.SECONDARY,
        reviewer_slot="secondary_blind",
        task_ids=secondary_ids,
        assignment_hash=secondary_hash,
    )
    return [primary, secondary]


def build_review_tasks(
    *,
    pilot_id: str,
    config: QuestionerPilotConfig,
    runtime_exports: dict[DatasetName, Path],
    coverage_exports: dict[DatasetName, Path],
    snapshot_dirs: dict[DatasetName, Path],
    assignments: list[BenchmarkPartitionAssignment],
) -> tuple[list[QuestionerReviewTask], list[ReviewAssignment], dict[str, Any]]:
    selected_tasks: list[QuestionerReviewTask] = []
    quota_report: dict[str, Any] = {}
    assignment_by_query = {item.query_id: item for item in assignments}
    for dataset in sorted(runtime_exports, key=lambda item: item.value):
        candidates = _candidate_payloads(
            dataset=dataset,
            runtime_export=runtime_exports[dataset],
            snapshot_dir=snapshot_dirs[dataset],
        )
        by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in candidates:
            by_bucket[str(item["bucket"])].append(item)
        requested = {
            "eligible": config.eligible_candidate_reviews_per_dataset,
            "rejected": config.rejected_candidate_reviews_per_dataset,
            "duplicate_diagnostic": config.duplicate_candidate_reviews_per_dataset,
        }
        picked: list[dict[str, Any]] = []
        for bucket in ("eligible", "rejected", "duplicate_diagnostic"):
            picked.extend(
                _stable_pick(
                    by_bucket[bucket],
                    requested[bucket],
                    seed=config.partition_seed,
                    label=f"{dataset.value}:candidate:{bucket}",
                    stratify_by=(
                        "question_type",
                        "required_modalities",
                        "reasoning_type",
                    ),
                )
            )
        needed = sum(requested.values()) - len(picked)
        if needed:
            selected_ids = {str(item["qbank_id"]) for item in picked}
            for fallback_bucket in (
                "duplicate_diagnostic",
                "rejected",
                "eligible",
            ):
                remainder = [
                    item
                    for item in by_bucket[fallback_bucket]
                    if str(item["qbank_id"]) not in selected_ids
                ]
                additional = _stable_pick(
                    remainder,
                    needed,
                    seed=config.partition_seed,
                    label=(
                        f"{dataset.value}:candidate:reallocated:{fallback_bucket}"
                    ),
                    stratify_by=(
                        "question_type",
                        "required_modalities",
                        "reasoning_type",
                    ),
                )
                picked.extend(additional)
                selected_ids.update(str(item["qbank_id"]) for item in additional)
                needed -= len(additional)
                if needed == 0:
                    break
        if len(picked) != sum(requested.values()):
            raise ValueError(
                f"insufficient Questioner candidates for review in {dataset.value}"
            )
        for item in picked:
            identity = {
                "pilot_id": pilot_id,
                "dataset": dataset.value,
                "kind": ReviewTaskKind.CANDIDATE_INTRINSIC.value,
                "source_id": item["qbank_id"],
                "payload": item,
            }
            task_hash = stable_hash(identity)
            selected_tasks.append(
                QuestionerReviewTask(
                    task_id="questioner-review-" + task_hash[:24],
                    pilot_id=pilot_id,
                    kind=ReviewTaskKind.CANDIDATE_INTRINSIC,
                    dataset=dataset,
                    bucket=str(item["bucket"]),
                    source_id=str(item["qbank_id"]),
                    qbank_id=str(item["qbank_id"]),
                    task_hash=task_hash,
                    payload=item,
                )
            )

        coverage = _coverage_payloads(
            dataset=dataset,
            coverage_export=coverage_exports[dataset],
            runtime_export=runtime_exports[dataset],
            snapshot_dir=snapshot_dirs[dataset],
        )
        for item in coverage:
            assignment = assignment_by_query.get(str(item["query_id"]))
            if assignment is None:
                raise ValueError("coverage export contains a query outside the partition")
            if assignment.dataset is not dataset:
                raise ValueError("coverage export contains a query from another dataset")
        coverage_picked, coverage_quota = _select_coverage_review_payloads(
            dataset=dataset,
            coverage=coverage,
            config=config,
        )
        for item in coverage_picked:
            identity = {
                "pilot_id": pilot_id,
                "dataset": dataset.value,
                "kind": ReviewTaskKind.QUERY_COVERAGE.value,
                "source_id": item["query_id"],
                "payload": item,
            }
            task_hash = stable_hash(identity)
            selected_tasks.append(
                QuestionerReviewTask(
                    task_id="questioner-review-" + task_hash[:24],
                    pilot_id=pilot_id,
                    kind=ReviewTaskKind.QUERY_COVERAGE,
                    dataset=dataset,
                    bucket=str(item["bucket"]),
                    source_id=str(item["query_id"]),
                    query_id=str(item["query_id"]),
                    task_hash=task_hash,
                    payload=item,
                )
            )
        quota_report[dataset.value] = {
            "candidate_requested": requested,
            "candidate_actual": dict(
                sorted(
                    {
                        bucket: sum(
                            task.dataset is dataset
                            and task.kind is ReviewTaskKind.CANDIDATE_INTRINSIC
                            and task.bucket == bucket
                            for task in selected_tasks
                        )
                        for bucket in requested
                    }.items()
                )
            ),
            "coverage_count": len(coverage_picked),
            "coverage_allocation": coverage_quota,
        }
    selected_tasks.sort(key=lambda item: item.task_id)
    review_assignments = build_review_assignments(
        pilot_id=pilot_id,
        config=config,
        tasks=selected_tasks,
    )
    return selected_tasks, review_assignments, quota_report


def annotation_type(task: QuestionerReviewTask) -> type[
    CandidateQualityAnnotation | CoverageQualityAnnotation
]:
    if task.kind is ReviewTaskKind.CANDIDATE_INTRINSIC:
        return CandidateQualityAnnotation
    return CoverageQualityAnnotation
