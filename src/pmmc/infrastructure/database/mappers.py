"""Explicit Pydantic-to-row mappings for canonical persistence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pmmc.schemas import (
    AssetRef,
    BenchmarkQuery,
    DatasetSnapshotManifest,
    EvaluationEligibility,
    GoldEvidence,
    ImageRef,
    MemoryUnit,
    QueryGold,
    QueryGoldRecord,
    QueryInput,
    RunManifest,
    SessionRecord,
    SourceLocator,
)

Row = dict[str, Any]


def manifest_to_row(
    manifest: DatasetSnapshotManifest,
    *,
    import_state: str,
) -> Row:
    """Map a lossless snapshot manifest to one database row."""

    return {
        "snapshot_id": manifest.snapshot_id,
        "dataset": manifest.dataset.value,
        "dataset_version": manifest.dataset_version,
        "adapter_name": manifest.adapter_name,
        "adapter_version": manifest.adapter_version,
        "schema_version": manifest.schema_version,
        "evaluation_policy_version": manifest.evaluation_policy_version,
        "excluded_source_categories": list(manifest.excluded_source_categories),
        "source_content_hash": manifest.source_content_hash,
        "canonical_content_hash": manifest.canonical_content_hash,
        "counts": dict(manifest.counts),
        "category_counts": dict(manifest.category_counts),
        "eligible_category_counts": dict(manifest.eligible_category_counts),
        "excluded_category_counts": dict(manifest.excluded_category_counts),
        "manifest_json": manifest.to_dict(),
        "import_state": import_state,
        "created_at": manifest.created_at,
        "metadata": dict(manifest.metadata),
    }


def manifest_from_row(row: Mapping[str, Any]) -> DatasetSnapshotManifest:
    """Restore the public manifest without exposing persistence-only fields."""

    return DatasetSnapshotManifest.from_dict(dict(row["manifest_json"]))


def session_to_row(snapshot_id: str, value: SessionRecord) -> Row:
    return {
        "snapshot_id": snapshot_id,
        "session_id": value.session_id,
        "dataset": value.dataset.value,
        "conversation_id": value.conversation_id,
        "session_order": value.session_order,
        "source_session_id": value.source_session_id,
        "source_date_raw": value.source_date_raw,
        "source_date_iso": value.source_date_iso,
        "is_synthetic": value.is_synthetic,
        "metadata": dict(value.metadata),
    }


def session_from_row(row: Mapping[str, Any]) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        dataset=row["dataset"],
        conversation_id=row["conversation_id"],
        session_order=row["session_order"],
        source_session_id=row["source_session_id"],
        source_date_raw=row["source_date_raw"],
        source_date_iso=row["source_date_iso"],
        is_synthetic=row["is_synthetic"],
        metadata=dict(row["metadata"]),
    )


def asset_to_row(snapshot_id: str, value: AssetRef) -> Row:
    if value.asset_id is None:
        raise ValueError("canonical benchmark assets require asset_id")
    return {
        "snapshot_id": snapshot_id,
        "asset_id": value.asset_id,
        "asset_type": value.asset_type.value,
        "storage_uri": value.storage_uri,
        "local_path": value.local_path,
        "remote_url": value.remote_url,
        "sha256": value.sha256,
        "mime_type": value.mime_type,
        "size_bytes": value.size_bytes,
        "width": value.width,
        "height": value.height,
        "source_id": value.source_id,
        "run_id": value.run_id,
        "created_at": value.created_at,
        "metadata": dict(value.metadata),
    }


def asset_from_row(row: Mapping[str, Any]) -> AssetRef:
    return AssetRef(
        asset_id=row["asset_id"],
        asset_type=row["asset_type"],
        storage_uri=row["storage_uri"],
        local_path=row["local_path"],
        remote_url=row["remote_url"],
        sha256=row["sha256"],
        mime_type=row["mime_type"],
        size_bytes=row["size_bytes"],
        width=row["width"],
        height=row["height"],
        source_id=row["source_id"],
        run_id=row["run_id"],
        created_at=row["created_at"],
        metadata=dict(row["metadata"]),
    )


def merge_image_refs(current: ImageRef | None, incoming: ImageRef) -> ImageRef:
    """Deterministically fill missing image fields and reject conflicting facts."""

    if current is None:
        return incoming.model_copy(deep=True)
    if current.image_id != incoming.image_id:
        raise ValueError("cannot merge different image IDs")

    payload = current.to_dict()
    for field_name in (
        "asset_id",
        "local_path",
        "remote_url",
        "storage_uri",
        "caption",
        "sha256",
        "mime_type",
        "width",
        "height",
    ):
        existing = payload[field_name]
        candidate = getattr(incoming, field_name)
        if existing is not None and candidate is not None and existing != candidate:
            raise ValueError(f"conflicting {field_name} for image_id={incoming.image_id}")
        if existing is None:
            payload[field_name] = candidate

    metadata = dict(current.metadata)
    for key, candidate in incoming.metadata.items():
        if key in metadata and metadata[key] != candidate:
            raise ValueError(f"conflicting metadata.{key} for image_id={incoming.image_id}")
        metadata[key] = candidate
    payload["metadata"] = metadata
    return ImageRef.from_dict(payload)


def image_to_row(snapshot_id: str, value: ImageRef) -> Row:
    return {
        "snapshot_id": snapshot_id,
        "image_id": value.image_id,
        "asset_id": value.asset_id,
        "local_path": value.local_path,
        "remote_url": value.remote_url,
        "storage_uri": value.storage_uri,
        "caption": value.caption,
        "sha256": value.sha256,
        "mime_type": value.mime_type,
        "width": value.width,
        "height": value.height,
        "metadata": dict(value.metadata),
    }


def image_from_row(row: Mapping[str, Any]) -> ImageRef:
    return ImageRef(
        image_id=row["image_id"],
        asset_id=row["asset_id"],
        local_path=row["local_path"],
        remote_url=row["remote_url"],
        storage_uri=row["storage_uri"],
        caption=row["caption"],
        sha256=row["sha256"],
        mime_type=row["mime_type"],
        width=row["width"],
        height=row["height"],
        metadata=dict(row["metadata"]),
    )


def memory_image_refs(value: MemoryUnit) -> list[ImageRef]:
    """Return ordered image refs, materializing ID-only references when needed."""

    by_id = {item.image_id: item for item in value.image_refs}
    ordered_ids = value.image_ids or [item.image_id for item in value.image_refs]
    return [by_id.get(image_id, ImageRef(image_id=image_id)) for image_id in ordered_ids]


def memory_to_row(snapshot_id: str, value: MemoryUnit) -> Row:
    return {
        "snapshot_id": snapshot_id,
        "memory_id": value.memory_id,
        "dataset": value.dataset.value,
        "conversation_id": value.conversation_id,
        "session_id": value.session_id,
        "turn_id": value.turn_id,
        "timestamp_order": value.timestamp_order,
        "speaker": value.speaker.value,
        "text": value.text,
        "caption": value.caption,
        "entity_mentions": list(value.entity_mentions),
        "modalities": [item.value for item in value.modality],
        "available_after_session": value.available_after_session,
        "token_count": value.token_count,
        "metadata": dict(value.metadata),
    }


def memory_from_row(
    row: Mapping[str, Any],
    image_refs: list[ImageRef],
) -> MemoryUnit:
    return MemoryUnit(
        memory_id=row["memory_id"],
        dataset=row["dataset"],
        conversation_id=row["conversation_id"],
        session_id=row["session_id"],
        turn_id=row["turn_id"],
        timestamp_order=row["timestamp_order"],
        speaker=row["speaker"],
        text=row["text"],
        caption=row["caption"],
        image_ids=[item.image_id for item in image_refs],
        image_refs=image_refs,
        entity_mentions=list(row["entity_mentions"]),
        modality=list(row["modalities"]),
        available_after_session=row["available_after_session"],
        token_count=row["token_count"],
        metadata=dict(row["metadata"]),
    )


def benchmark_query_to_row(snapshot_id: str, value: BenchmarkQuery) -> Row:
    query = value.query
    source = value.source
    eligibility = value.eligibility
    return {
        "snapshot_id": snapshot_id,
        "query_id": query.query_id,
        "dataset": query.dataset.value,
        "conversation_id": query.conversation_id,
        "raw_query": query.raw_query,
        "question_type": query.query_type.value,
        "query_subtype": query.query_subtype,
        "available_after_session": query.available_after_session,
        "eligible": eligibility.eligible,
        "evaluation_policy_version": eligibility.policy_version,
        "exclusion_reason": eligibility.exclusion_reason,
        "source_file": source.source_file,
        "source_record_index": source.source_record_index,
        "source_query_id": source.source_query_id,
        "scenario_id": source.scenario_id,
        "qa_index": source.qa_index,
        "source_session_id": source.source_session_id,
        "source_round_id": source.source_round_id,
        "source_turn_index": source.source_turn_index,
        "source_category": source.source_category,
        "source_subtype": source.source_subtype,
        "query_metadata": dict(query.metadata),
        "source_metadata": dict(source.metadata),
        "eligibility_metadata": dict(eligibility.metadata),
        "metadata": dict(value.metadata),
    }


def benchmark_query_from_row(
    row: Mapping[str, Any],
    image_refs: list[ImageRef],
) -> BenchmarkQuery:
    return BenchmarkQuery(
        query=QueryInput(
            query_id=row["query_id"],
            dataset=row["dataset"],
            conversation_id=row["conversation_id"],
            raw_query=row["raw_query"],
            query_type=row["question_type"],
            query_subtype=row["query_subtype"],
            query_image_refs=image_refs,
            available_after_session=row["available_after_session"],
            metadata=dict(row["query_metadata"]),
        ),
        source=SourceLocator(
            dataset=row["dataset"],
            source_file=row["source_file"],
            source_record_index=row["source_record_index"],
            source_query_id=row["source_query_id"],
            scenario_id=row["scenario_id"],
            qa_index=row["qa_index"],
            source_session_id=row["source_session_id"],
            source_round_id=row["source_round_id"],
            source_turn_index=row["source_turn_index"],
            source_category=row["source_category"],
            source_subtype=row["source_subtype"],
            metadata=dict(row["source_metadata"]),
        ),
        eligibility=EvaluationEligibility(
            eligible=row["eligible"],
            policy_version=row["evaluation_policy_version"],
            exclusion_reason=row["exclusion_reason"],
            metadata=dict(row["eligibility_metadata"]),
        ),
        metadata=dict(row["metadata"]),
    )


def query_gold_to_rows(snapshot_id: str, value: QueryGoldRecord) -> dict[str, list[Row]]:
    evidence = value.gold.evidence
    parent = {
        "snapshot_id": snapshot_id,
        "query_id": value.query_id,
        "answer": value.gold.answer,
        "evidence_metadata": dict(evidence.metadata),
        "gold_metadata": dict(value.gold.metadata),
        "record_metadata": dict(value.metadata),
    }

    def child_rows(field_name: str, id_name: str) -> list[Row]:
        return [
            {
                "snapshot_id": snapshot_id,
                "query_id": value.query_id,
                id_name: identifier,
                "ordinal": ordinal,
            }
            for ordinal, identifier in enumerate(getattr(evidence, field_name))
        ]

    return {
        "query_gold": [parent],
        "evidence_ids": child_rows("evidence_ids", "evidence_id"),
        "memory_ids": child_rows("memory_ids", "memory_id"),
        "session_ids": child_rows("session_ids", "session_id"),
        "image_ids": child_rows("image_ids", "image_id"),
    }


def query_gold_from_parts(
    parent: Mapping[str, Any],
    *,
    evidence_ids: list[str],
    memory_ids: list[str],
    session_ids: list[str],
    image_ids: list[str],
) -> QueryGold:
    return QueryGold(
        answer=parent["answer"],
        evidence=GoldEvidence(
            evidence_ids=evidence_ids,
            memory_ids=memory_ids,
            session_ids=session_ids,
            image_ids=image_ids,
            metadata=dict(parent["evidence_metadata"]),
        ),
        metadata=dict(parent["gold_metadata"]),
    )


def run_manifest_to_row(value: RunManifest) -> Row:
    return {
        "run_id": value.run_id,
        "experiment_name": value.experiment_name,
        "dataset": value.dataset.value,
        "dataset_version": value.dataset_version,
        "method": value.method,
        "paper_mode": value.paper_mode,
        "schema_version": value.schema_version,
        "experiment_schema_version": value.experiment_schema_version,
        "selection_hash": value.selection_hash,
        "seed": value.seed,
        "retrieval_topk_max": value.retrieval_topk_max,
        "config_hash": value.config_hash,
        "git_commit": value.git_commit,
        "created_at": value.created_at,
        "manifest_json": value.to_dict(),
    }
