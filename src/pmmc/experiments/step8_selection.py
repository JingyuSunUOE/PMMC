"""Frozen, nested selections; no benchmark answers participate in sampling."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

from pydantic import Field, model_validator

from pmmc.compiler.ids import canonical_content_hash
from pmmc.schemas.base import BaseSchema, DatasetName, NonEmptyStr
from pmmc.schemas.pilot import BenchmarkPartitionAssignment, PartitionRole
from pmmc.schemas.step8 import CampaignPhase, SelectionExpansionManifest


class ExposureManifest(BaseSchema):
    protocol: str = "benchmark-exposure-v1"
    partition_hash: NonEmptyStr
    complete: bool
    query_ids: dict[DatasetName, list[NonEmptyStr]]
    provenance_hashes: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _provenance(self) -> ExposureManifest:
        if self.protocol != "benchmark-exposure-v1":
            raise ValueError("unsupported exposure manifest")
        if self.complete and not self.provenance_hashes:
            raise ValueError("complete exposure history requires provenance")
        if set(self.query_ids) != {DatasetName.MEMLENS, DatasetName.MEMGALLERY}:
            raise ValueError("exposure history must explicitly describe both datasets")
        if any(len(ids) != len(set(ids)) for ids in self.query_ids.values()):
            raise ValueError("exposure query IDs cannot repeat")
        return self

    @property
    def content_hash(self) -> str:
        return canonical_content_hash(self.model_dump(mode="json"))


def _stratum(item: BenchmarkPartitionAssignment) -> str:
    if item.dataset is DatasetName.MEMGALLERY:
        if not item.scenario_id:
            raise ValueError("MemGallery selection requires scenario identity")
        return item.scenario_id
    return item.source_category


def denominator_report(
    assignments: Sequence[BenchmarkPartitionAssignment],
    exposure: ExposureManifest,
) -> dict[str, dict[str, int | None]]:
    by_key = {(item.dataset, item.query_id): item for item in assignments}
    if len(by_key) != len(assignments):
        raise ValueError("partition contains duplicate dataset/query assignments")
    if any(
        (dataset, query) not in by_key
        for dataset, ids in exposure.query_ids.items()
        for query in ids
    ):
        raise ValueError("exposure history contains queries outside the frozen partition")
    result = {}
    for dataset in (DatasetName.MEMLENS, DatasetName.MEMGALLERY):
        rows = [item for item in assignments if item.dataset is dataset]
        paper = {item.query_id for item in rows if item.role is PartitionRole.PAPER_TEST}
        result[dataset.value] = {
            "eligible_total": len(rows),
            "paper_test": len(paper),
            "development": len(rows) - len(paper),
            "historically_exposed": len(exposure.query_ids[dataset]) if exposure.complete else None,
            "paper_exposure_intersection": (
                len(paper & set(exposure.query_ids[dataset])) if exposure.complete else None
            ),
            "conversation_count": len({item.conversation_id for item in rows}),
            "stratum_count": len({_stratum(item) for item in rows}),
        }
        if not exposure.complete:
            result[dataset.value].update(
                exposure_history_complete=False,
                known_exposed_query_count=len(exposure.query_ids[dataset]),
                known_paper_exposure_intersection=len(paper & set(exposure.query_ids[dataset])),
            )
    return result


def select_benchmark_queries(
    *,
    campaign_id: str,
    phase: CampaignPhase,
    selection: str,
    partition_hash: str,
    assignments: Sequence[BenchmarkPartitionAssignment],
    exposure: ExposureManifest,
    regression_ids: Mapping[DatasetName, Sequence[str]],
    seed: int = 0,
    previous: SelectionExpansionManifest | None = None,
    additional_exposed_ids: Mapping[DatasetName, Sequence[str]] | None = None,
) -> SelectionExpansionManifest:
    if exposure.partition_hash != partition_hash:
        raise ValueError("exposure and partition identities differ")
    report = denominator_report(assignments, exposure)
    if phase is CampaignPhase.PAPER:
        if selection not in {"5", "20", "100"}:
            raise ValueError("paper selections are the nested 5/20/100 percent stages")
        if not exposure.complete or any(
            value["paper_exposure_intersection"] for value in report.values()
        ):
            raise ValueError("paper exposure history is incomplete or contaminated")
        role = PartitionRole.PAPER_TEST
    elif phase is CampaignPhase.EXPLORATION:
        if selection != "dev-all" or previous is not None:
            raise ValueError("exploration requires a standalone dev-all selection")
        role = PartitionRole.DEVELOPMENT
    else:
        if selection not in {"diagnostic", "regression", "confirmation"}:
            raise ValueError("development selection must be diagnostic, regression or confirmation")
        if selection == "confirmation" and not exposure.complete:
            raise ValueError("unseen confirmation requires complete exposure history")
        role = PartitionRole.DEVELOPMENT
    if previous is not None and (
        previous.campaign_id != campaign_id
        or previous.phase is not phase
        or previous.partition_hash != partition_hash
    ):
        raise ValueError("selection expansion cannot switch campaign, phase, or partition")
    if previous is not None and "diagnostic" in {previous.selection, selection}:
        raise ValueError("diagnostic prefix must remain an independent campaign")
    selected: dict[DatasetName, list[str]] = {}
    added: dict[DatasetName, list[str]] = {}
    counts: dict[DatasetName, dict[str, int]] = {}
    eligible_counts: dict[DatasetName, int] = {}
    for dataset in (DatasetName.MEMLENS, DatasetName.MEMGALLERY):
        population = [item for item in assignments if item.dataset is dataset and item.role is role]
        eligible_counts[dataset] = len(population)
        lookup = {item.query_id: item for item in population}
        if not population:
            raise ValueError("selected partition has an empty dataset")
        grouped: dict[str, list[BenchmarkPartitionAssignment]] = defaultdict(list)
        for item in population:
            grouped[_stratum(item)].append(item)
        for rows in grouped.values():
            rows.sort(
                key=lambda item: (
                    canonical_content_hash(
                        [
                            "step8-nested-selection-v1",
                            seed,
                            dataset.value,
                            item.query_id,
                        ]
                    ),
                    item.query_id,
                )
            )
        chosen: list[BenchmarkPartitionAssignment] = []
        if phase is CampaignPhase.PAPER:
            for stratum in sorted(grouped):
                rows = grouped[stratum]
                chosen.extend(rows[: max(1, math.ceil(len(rows) * int(selection) / 100))])
        elif phase is CampaignPhase.EXPLORATION:
            chosen = population
        elif selection in {"diagnostic", "regression"}:
            ids = list(regression_ids.get(dataset, ()))
            if len(ids) != 16 or len(set(ids)) != 16 or not set(ids) <= lookup.keys():
                raise ValueError("regression requires 16 frozen development queries/dataset")
            chosen = [lookup[value] for value in ids]
            distribution: dict[str, int] = defaultdict(int)
            for item in chosen:
                distribution[_stratum(item)] += 1
            if len(distribution) != 4 or set(distribution.values()) != {4}:
                raise ValueError("regression requires four frozen equal strata")
            if selection == "diagnostic":
                chosen.sort(
                    key=lambda item: (
                        canonical_content_hash(["step8-real-prefix-v1", seed, item.query_id]),
                        item.query_id,
                    )
                )
                chosen = chosen[:2]
        else:
            excluded = set(exposure.query_ids[dataset]) | set(regression_ids.get(dataset, ()))
            excluded.update((additional_exposed_ids or {}).get(dataset, ()))
            queues = {
                key: [item for item in values if item.query_id not in excluded]
                for key, values in grouped.items()
            }
            while len(chosen) < 16:
                changed = False
                for stratum in sorted(queues):
                    if queues[stratum] and len(chosen) < 16:
                        chosen.append(queues[stratum].pop(0))
                        changed = True
                if not changed:
                    break
            if not chosen:
                raise ValueError("no unexposed development confirmation queries remain")
        ids = sorted(item.query_id for item in chosen)
        old = set(previous.query_ids[dataset]) if previous is not None else set()
        if phase is CampaignPhase.PAPER and not old <= set(ids):
            raise ValueError("paper selection cannot shrink")
        if previous is not None and phase is CampaignPhase.PAPER:
            if int(selection) < int(previous.selection):
                raise ValueError("paper selection cannot move backwards")
        selected[dataset] = ids
        added[dataset] = sorted(set(ids) - old)
        distribution = defaultdict(int)
        for item in chosen:
            distribution[_stratum(item)] += 1
        counts[dataset] = dict(sorted(distribution.items()))
    return SelectionExpansionManifest(
        campaign_id=campaign_id,
        phase=phase,
        selection=selection,
        partition_hash=partition_hash,
        query_ids=selected,
        added_query_ids=added,
        stratum_counts=counts,
        eligible_counts=eligible_counts,
        parent_selection_hash=previous.content_hash if previous else None,
    )
