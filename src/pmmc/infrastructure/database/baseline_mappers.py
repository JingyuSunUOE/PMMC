"""Lossless mappings for baseline corpus and context persistence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pmmc.schemas import (
    BaselineCorpusManifest,
    ContextAssemblyTrace,
    ContextRetrievalCandidate,
    MemoryTokenCount,
    MemoryWindow,
    RetrievalDocument,
)

Row = dict[str, Any]


def corpus_to_row(value: BaselineCorpusManifest, *, building: bool = False) -> Row:
    spec = value.spec
    return {
        "corpus_id": value.corpus_id,
        "snapshot_id": value.snapshot_id,
        "dataset": value.dataset.value,
        "status": "building" if building else value.status.value,
        "content_hash": value.content_hash,
        "windows_hash": value.windows_hash,
        "documents_hash": value.documents_hash,
        "tokenizer_alias": spec.tokenizer_alias,
        "tokenizer_revision": spec.tokenizer_revision,
        "renderer_version": spec.renderer_version,
        "window_strategy": spec.window_strategy,
        "max_text_tokens": spec.max_text_tokens,
        "max_images": spec.max_images,
        "overlap_memory_units": spec.overlap_memory_units,
        "cross_session": spec.cross_session,
        "representations": [item.value for item in spec.representations],
        "window_count": value.window_count,
        "document_count": value.document_count,
        "image_document_count": value.image_document_count,
        "unique_memory_count": value.unique_memory_count,
        "unique_image_count": value.unique_image_count,
        "spec_json": spec.model_dump(mode="json"),
        "manifest_json": value.model_dump(mode="json"),
        "created_at": value.created_at,
        "metadata": dict(value.metadata),
    }


def window_to_rows(value: MemoryWindow) -> dict[str, list[Row]]:
    parent = value.model_dump(mode="python")
    memory_ids = parent.pop("memory_ids")
    parent.pop("image_ids")
    parent["dataset"] = value.dataset.value
    parent["metadata"] = dict(value.metadata)
    return {
        "window": [parent],
        "units": [
            {
                "window_id": value.window_id,
                "snapshot_id": value.snapshot_id,
                "conversation_id": value.conversation_id,
                "memory_id": memory_id,
                "ordinal": ordinal,
            }
            for ordinal, memory_id in enumerate(memory_ids)
        ],
    }


def window_from_parts(
    row: Mapping[str, Any],
    memory_ids: list[str],
    image_ids: list[str],
) -> MemoryWindow:
    return MemoryWindow.model_validate(
        {**dict(row), "memory_ids": memory_ids, "image_ids": image_ids}
    )


def document_to_rows(value: RetrievalDocument) -> dict[str, list[Row]]:
    parent = value.model_dump(mode="python")
    image_ids = parent.pop("image_ids")
    parent["dataset"] = value.dataset.value
    parent["representation"] = value.representation.value
    parent["metadata"] = dict(value.metadata)
    return {
        "document": [parent],
        "images": [
            {
                "document_id": value.document_id,
                "snapshot_id": value.snapshot_id,
                "image_id": image_id,
                "ordinal": ordinal,
            }
            for ordinal, image_id in enumerate(image_ids)
        ],
    }


def document_from_parts(row: Mapping[str, Any], image_ids: list[str]) -> RetrievalDocument:
    return RetrievalDocument.model_validate({**dict(row), "image_ids": image_ids})


def context_to_rows(value: ContextAssemblyTrace) -> dict[str, list[Row]]:
    parent = value.model_dump(mode="python")
    token_counts = {item.memory_id: item for item in value.memory_token_counts}
    included = parent.pop("included_memory_ids")
    truncated = parent.pop("truncated_memory_ids")
    parent.pop("memory_token_counts")
    candidates = parent.pop("retrieval_candidates")
    image_groups = {
        "included": parent.pop("included_image_ids"),
        "truncated": parent.pop("truncated_image_ids"),
        "query": parent.pop("query_image_ids"),
    }
    parent["method"] = value.method.value
    parent["representation"] = value.representation.value
    parent["truncation_policy"] = value.truncation_policy.value
    parent["metadata"] = dict(value.metadata)
    units: list[Row] = []
    for ordinal, memory_id in enumerate([*included, *truncated]):
        count = token_counts.get(memory_id)
        units.append(
            {
                "context_assembly_id": value.context_assembly_id,
                "snapshot_id": value.snapshot_id,
                "conversation_id": value.conversation_id,
                "memory_id": memory_id,
                "ordinal": ordinal,
                "included": memory_id in set(included),
                "text_tokens": count.text_tokens if count else 0,
                "image_tokens": count.image_tokens if count else 0,
            }
        )
    images = [
        {
            "context_assembly_id": value.context_assembly_id,
            "snapshot_id": value.snapshot_id,
            "image_id": image_id,
            "role": role,
            "ordinal": ordinal,
        }
        for role, image_ids in image_groups.items()
        for ordinal, image_id in enumerate(image_ids)
    ]
    candidate_rows = [
        {
            "context_assembly_id": value.context_assembly_id,
            "retrieval_trace_id": item["retrieval_trace_id"],
            "run_id": value.run_id,
            "snapshot_id": value.snapshot_id,
            "query_id": value.query_id,
            "conversation_id": value.conversation_id,
            "window_id": item["window_id"],
            "rank": item["rank"],
            "disposition": item["disposition"],
            "contributed_memory_ids": item["contributed_memory_ids"],
            "metadata": item["metadata"],
        }
        for item in candidates
    ]
    return {
        "context": [parent],
        "units": units,
        "images": images,
        "candidates": candidate_rows,
    }


def context_from_parts(
    row: Mapping[str, Any],
    units: list[Mapping[str, Any]],
    images: list[Mapping[str, Any]],
    candidates: list[Mapping[str, Any]] | None = None,
) -> ContextAssemblyTrace:
    included = [item["memory_id"] for item in units if item["included"]]
    truncated = [item["memory_id"] for item in units if not item["included"]]
    token_counts = [
        MemoryTokenCount(
            memory_id=item["memory_id"],
            text_tokens=item["text_tokens"],
            image_tokens=item["image_tokens"],
        )
        for item in units
    ]
    groups = {
        role: [item["image_id"] for item in images if item["role"] == role]
        for role in ("included", "truncated", "query")
    }
    return ContextAssemblyTrace.model_validate(
        {
            **dict(row),
            "memory_token_counts": token_counts,
            "included_memory_ids": included,
            "truncated_memory_ids": truncated,
            "included_image_ids": groups["included"],
            "truncated_image_ids": groups["truncated"],
            "query_image_ids": groups["query"],
            "retrieval_candidates": [
                ContextRetrievalCandidate.model_validate(
                    {
                        "retrieval_trace_id": item["retrieval_trace_id"],
                        "window_id": item["window_id"],
                        "rank": item["rank"],
                        "disposition": item["disposition"],
                        "contributed_memory_ids": list(item["contributed_memory_ids"]),
                        "metadata": dict(item["metadata"]),
                    }
                )
                for item in (candidates or [])
            ],
        }
    )
