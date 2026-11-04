"""Cross-record validation for source adapters and canonical objects."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from pmmc.benchmarks.base import BenchmarkAdapter
from pmmc.benchmarks.ids import sha256_file, stable_hash
from pmmc.schemas import (
    AssetRef,
    ValidationIssue,
    ValidationLevel,
    ValidationReport,
    ValidationSeverity,
)
from pmmc.schemas.base import utc_now

ProgressCallback = Callable[[str, int, int | None], None]


@dataclass(slots=True)
class AdapterScan:
    """Internal scan result reused by validation and snapshot writing."""

    report: ValidationReport
    assets: list[AssetRef]
    source_content_hash: str
    category_counts: dict[str, int]
    eligible_category_counts: dict[str, int]
    excluded_category_counts: dict[str, int]


def _error(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(
        severity=ValidationSeverity.ERROR,
        code=code,
        message=message,
    )


def _exception_message(error: Exception) -> str:
    if isinstance(error, ValidationError):
        parts = []
        for detail in error.errors(include_input=False, include_url=False):
            location = ".".join(str(item) for item in detail["loc"]) or "record"
            parts.append(f"{location}: {detail['msg']}")
        return "; ".join(parts)
    return str(error)


def scan_adapter(
    adapter: BenchmarkAdapter,
    level: ValidationLevel,
    *,
    enforce_expected_counts: bool = True,
    progress: ProgressCallback | None = None,
) -> AdapterScan:
    """Validate assets, IDs, references, cutoffs, gold links, and package counts."""

    started_at = utc_now()
    issues: list[ValidationIssue] = []
    assets: list[AssetRef] = []
    counts = Counter(
        source_records=adapter.source_record_count,
        sessions=0,
        memory_units=0,
        queries=0,
        eligible_queries=0,
        excluded_queries=0,
        assets=0,
        referenced_unique_images=0,
    )
    categories: Counter[str] = Counter()
    eligible_categories: Counter[str] = Counter()
    excluded_categories: Counter[str] = Counter()
    asset_ids: set[str] = set()
    session_ids: set[str] = set()
    memory_ids: set[str] = set()
    query_ids: set[str] = set()
    referenced_asset_ids: set[str] = set()

    try:
        expected_asset_count = adapter.expected_counts().get("assets")
        for asset_index, asset in enumerate(adapter.iter_assets(level), start=1):
            if asset.asset_id is None:
                raise ValueError("canonical benchmark assets require asset_id")
            if asset.asset_id in asset_ids:
                raise ValueError(f"duplicate canonical asset ID: {asset.asset_id}")
            asset_ids.add(asset.asset_id)
            assets.append(asset)
            if progress is not None:
                progress("validate_assets", asset_index, expected_asset_count)
        counts["assets"] = len(assets)

        for record_index, conversation in enumerate(adapter.iter_conversations(), start=1):
            local_session_ids = {session.session_id for session in conversation.sessions}
            if len(local_session_ids) != len(conversation.sessions):
                raise ValueError(
                    f"duplicate session IDs in conversation {conversation.conversation_id}"
                )
            expected_orders = list(range(len(conversation.sessions)))
            actual_orders = [session.session_order for session in conversation.sessions]
            if actual_orders != expected_orders:
                raise ValueError(
                    f"session orders are not contiguous in {conversation.conversation_id}"
                )
            if session_ids.intersection(local_session_ids):
                raise ValueError("canonical session IDs must be globally unique")
            session_ids.update(local_session_ids)
            counts["sessions"] += len(conversation.sessions)

            local_memory_ids = {memory.memory_id for memory in conversation.memory_units}
            if len(local_memory_ids) != len(conversation.memory_units):
                raise ValueError(
                    f"duplicate memory IDs in conversation {conversation.conversation_id}"
                )
            if memory_ids.intersection(local_memory_ids):
                raise ValueError("canonical memory IDs must be globally unique")
            memory_ids.update(local_memory_ids)
            counts["memory_units"] += len(conversation.memory_units)
            for memory in conversation.memory_units:
                if memory.conversation_id != conversation.conversation_id:
                    raise ValueError(f"memory conversation mismatch: {memory.memory_id}")
                if memory.session_id not in local_session_ids:
                    raise ValueError(f"memory session does not resolve: {memory.memory_id}")
                if memory.available_after_session not in local_session_ids:
                    raise ValueError(
                        f"memory availability cutoff does not resolve: {memory.memory_id}"
                    )
                for image_ref in memory.image_refs:
                    if image_ref.asset_id is None:
                        raise ValueError(f"image ref lacks asset ID: {image_ref.image_id}")
                    referenced_asset_ids.add(image_ref.asset_id)

            for adapted in conversation.queries:
                query = adapted.record.query
                if query.query_id in query_ids:
                    raise ValueError(f"duplicate canonical query ID: {query.query_id}")
                query_ids.add(query.query_id)
                counts["queries"] += 1
                category = adapted.source.source_category or "unknown"
                categories[category] += 1
                if adapted.eligibility.eligible:
                    counts["eligible_queries"] += 1
                    eligible_categories[category] += 1
                else:
                    counts["excluded_queries"] += 1
                    excluded_categories[category] += 1
                if query.conversation_id != conversation.conversation_id:
                    raise ValueError(f"query conversation mismatch: {query.query_id}")
                if query.available_after_session not in local_session_ids:
                    raise ValueError(
                        f"query availability cutoff does not resolve: {query.query_id}"
                    )
                for image_ref in query.query_image_refs:
                    if image_ref.asset_id is None:
                        raise ValueError(f"query image ref lacks asset ID: {image_ref.image_id}")
                    referenced_asset_ids.add(image_ref.asset_id)
                if adapted.record.gold is None:
                    raise ValueError(f"benchmark query lacks gold record: {query.query_id}")
                gold_evidence = adapted.record.gold.evidence
                if not set(gold_evidence.session_ids).issubset(local_session_ids):
                    raise ValueError(f"gold sessions do not resolve: {query.query_id}")
                if not set(gold_evidence.memory_ids).issubset(local_memory_ids):
                    raise ValueError(f"gold memory IDs do not resolve: {query.query_id}")
            if progress is not None:
                progress("validate_records", record_index, adapter.source_record_count)

        missing_assets = referenced_asset_ids - asset_ids
        if missing_assets:
            raise ValueError(
                f"{len(missing_assets)} referenced image assets do not exist in the asset registry"
            )
        counts["referenced_unique_images"] = len(referenced_asset_ids)
    except Exception as exc:  # validation converts source failures into a report
        issues.append(_error("benchmark_validation_failed", _exception_message(exc)))

    if enforce_expected_counts and not issues:
        for name, expected in adapter.expected_counts().items():
            actual = counts[name]
            if actual != expected:
                issues.append(
                    _error(
                        "unexpected_package_count",
                        f"{name} expected {expected}, found {actual}",
                    )
                )

    source_entries: list[dict[str, str | int | None]] = []
    try:
        for source_file in adapter.source_files():
            if not source_file.is_file():
                raise ValueError(f"source file is missing: {source_file}")
            source_entries.append(
                {
                    "path": source_file.relative_to(adapter.root).as_posix(),
                    "size_bytes": source_file.stat().st_size,
                    "sha256": sha256_file(source_file),
                }
            )
        for asset in assets:
            source_entries.append(
                {
                    "path": str(asset.metadata["dataset_relative_path"]),
                    "size_bytes": asset.size_bytes or 0,
                    "sha256": asset.sha256 if level is ValidationLevel.FULL else None,
                }
            )
    except (OSError, ValueError) as exc:
        issues.append(_error("source_hash_failed", str(exc)))

    source_content_hash = stable_hash(source_entries)
    report = ValidationReport(
        dataset=adapter.dataset,
        level=level,
        success=not any(issue.severity is ValidationSeverity.ERROR for issue in issues),
        issues=issues,
        counts=dict(counts),
        started_at=started_at,
        completed_at=utc_now(),
        metadata={
            "adapter_name": adapter.adapter_name,
            "adapter_version": adapter.adapter_version,
            "dataset_version": adapter.dataset_version,
            "source_content_hash": source_content_hash,
            "category_counts": dict(sorted(categories.items())),
            "eligible_category_counts": dict(sorted(eligible_categories.items())),
            "excluded_category_counts": dict(sorted(excluded_categories.items())),
        },
    )
    return AdapterScan(
        report=report,
        assets=assets,
        source_content_hash=source_content_hash,
        category_counts=dict(sorted(categories.items())),
        eligible_category_counts=dict(sorted(eligible_categories.items())),
        excluded_category_counts=dict(sorted(excluded_categories.items())),
    )


def validate_adapter(
    adapter: BenchmarkAdapter,
    level: ValidationLevel,
    *,
    enforce_expected_counts: bool = True,
    progress: ProgressCallback | None = None,
) -> ValidationReport:
    """Return a public validation report without writing a snapshot."""

    return scan_adapter(
        adapter,
        level,
        enforce_expected_counts=enforce_expected_counts,
        progress=progress,
    ).report
