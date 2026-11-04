"""Deterministic benchmark subset selection."""

from __future__ import annotations

from collections.abc import Iterable

from pmmc.benchmarks.ids import stable_hash
from pmmc.schemas import BenchmarkQuery, BenchmarkSelection


def select_queries(
    queries: Iterable[BenchmarkQuery],
    selection: BenchmarkSelection,
) -> list[BenchmarkQuery]:
    """Apply eligibility and user filters in one fixed, reproducible order."""

    all_queries = sorted(queries, key=lambda item: item.query.query_id)
    query_by_id = {item.query.query_id: item for item in all_queries}
    if selection.query_ids:
        missing = set(selection.query_ids) - set(query_by_id)
        if missing:
            raise ValueError(f"requested query IDs do not exist: {sorted(missing)}")
        excluded = [
            query_id
            for query_id in selection.query_ids
            if not query_by_id[query_id].eligibility.eligible
        ]
        if excluded and not selection.include_excluded:
            raise ValueError(
                "explicitly requested excluded query IDs require include_excluded=True: "
                f"{sorted(excluded)}"
            )

    selected = [item for item in all_queries if item.query.dataset in selection.datasets]
    if not selection.include_excluded:
        selected = [item for item in selected if item.eligibility.eligible]
    if selection.question_types:
        selected = [item for item in selected if item.query.query_type in selection.question_types]
    if selection.source_categories:
        selected = [
            item for item in selected if item.source.source_category in selection.source_categories
        ]
    if selection.scenario_ids:
        selected = [item for item in selected if item.source.scenario_id in selection.scenario_ids]
    if selection.query_ids:
        requested_ids = set(selection.query_ids)
        selected = [item for item in selected if item.query.query_id in requested_ids]

    if selection.limit is not None:
        selected = selected[: selection.limit]
    elif selection.sample_size is not None:
        if selection.sample_size > len(selected):
            raise ValueError(
                f"sample_size {selection.sample_size} exceeds selected population {len(selected)}"
            )
        selected = sorted(
            selected,
            key=lambda item: stable_hash(
                {
                    "seed": selection.seed,
                    "query_id": item.query.query_id,
                }
            ),
        )[: selection.sample_size]
        selected.sort(key=lambda item: item.query.query_id)

    return [
        item
        for index, item in enumerate(selected)
        if index % selection.shard_count == selection.shard_index
    ]


def selection_hash(
    selection: BenchmarkSelection,
    selected: Iterable[BenchmarkQuery],
) -> str:
    """Fingerprint the full selector and its resolved canonical query IDs."""

    return stable_hash(
        {
            "selection": selection.to_dict(),
            "query_ids": [item.query.query_id for item in selected],
        }
    )
