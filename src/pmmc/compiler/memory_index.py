"""Frozen memory-operator collection and program-query embedding contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.schemas import (
    DistanceMetric,
    EmbeddingCollection,
    EmbeddingCollectionStatus,
    EmbeddingInput,
    EmbeddingRequest,
    EmbeddingSourceType,
    ImageRef,
    MemoryOperatorIndexManifest,
    MemoryUnit,
    QuestionCandidate,
)

TEXT_MEMORY_REPRESENTATION = "memory-unit-text-caption-v1"
VISUAL_MEMORY_REPRESENTATION = "canonical-history-image-v1"
CAPTION_MEMORY_REPRESENTATION = "canonical-history-image-caption-only-v1"
PROGRAM_TEXT_QUERY_INSTRUCTION = "program-memory-text-query-v1"
PROGRAM_JOINT_QUERY_INSTRUCTION = "program-memory-joint-query-v1"


@dataclass(frozen=True)
class MemoryIndexInput:
    source_id: str
    memory_id: str
    text: str | None = None
    image: ImageRef | None = None


def memory_index_inputs(
    memories: Sequence[MemoryUnit],
    *,
    memory_raw_image_policy: str = "allow",
) -> tuple[tuple[MemoryIndexInput, ...], tuple[MemoryIndexInput, ...]]:
    text_inputs: list[MemoryIndexInput] = []
    visual_inputs: list[MemoryIndexInput] = []
    seen_images: set[str] = set()
    if memory_raw_image_policy not in {"allow", "forbid"}:
        raise ValueError("unknown memory raw-image policy")
    for memory in sorted(
        memories,
        key=lambda item: (item.timestamp_order, item.session_id, item.memory_id),
    ):
        if memory.text:
            text_inputs.append(
                MemoryIndexInput(
                    source_id=f"memory-text:{memory.memory_id}",
                    memory_id=memory.memory_id,
                    text=memory.text,
                )
            )
        if memory.caption:
            text_inputs.append(
                MemoryIndexInput(
                    source_id=f"memory-caption:{memory.memory_id}",
                    memory_id=memory.memory_id,
                    text=memory.caption,
                )
            )
        refs = {item.image_id: item for item in memory.image_refs}
        for image_id in dict.fromkeys(
            [*memory.image_ids, *(item.image_id for item in memory.image_refs)]
        ):
            if image_id in seen_images:
                continue
            if memory_raw_image_policy == "forbid" and not (memory.caption or memory.text):
                continue
            seen_images.add(image_id)
            visual_inputs.append(
                MemoryIndexInput(
                    source_id=image_id,
                    memory_id=memory.memory_id,
                    text=(memory.caption or memory.text)
                    if memory_raw_image_policy == "forbid"
                    else None,
                    image=(
                        refs.get(image_id, ImageRef(image_id=image_id))
                        if memory_raw_image_policy == "allow"
                        else None
                    ),
                )
            )
    return tuple(text_inputs), tuple(visual_inputs)


def make_memory_operator_index(
    *,
    run_id: str,
    snapshot_id: str,
    dataset: str,
    selection_hash: str,
    text_model_alias: str,
    text_model_revision: str,
    multimodal_model_alias: str,
    multimodal_model_revision: str,
    memory_raw_image_policy: str = "allow",
) -> tuple[MemoryOperatorIndexManifest, EmbeddingCollection, EmbeddingCollection]:
    if memory_raw_image_policy not in {"allow", "forbid"}:
        raise ValueError("unknown memory raw-image policy")
    visual_representation = (
        CAPTION_MEMORY_REPRESENTATION
        if memory_raw_image_policy == "forbid"
        else VISUAL_MEMORY_REPRESENTATION
    )
    treatment_metadata = (
        {"memory_raw_image_policy": "forbid", "image_inputs_allowed": False}
        if memory_raw_image_policy == "forbid"
        else {}
    )
    collection_identity = {
        "snapshot_id": snapshot_id,
        "dataset": dataset,
        "selection_hash": selection_hash,
        "text_representation": TEXT_MEMORY_REPRESENTATION,
        "visual_representation": visual_representation,
        "text_model": [text_model_alias, text_model_revision],
        "multimodal_model": [multimodal_model_alias, multimodal_model_revision],
        "dimension": 4096,
        "distance": "cosine",
    }
    # Collections are scientific assets and may be reused by later compiler runs.
    # The manifest is a run-scoped audit fact, so its identity must remain distinct.
    manifest_identity = {"run_id": run_id, **collection_identity}
    index_id = stable_compiler_id("memory-operator-index", manifest_identity)
    text_collection_id = stable_compiler_id("text-memory-collection", collection_identity)
    visual_collection_id = stable_compiler_id("visual-memory-collection", collection_identity)
    text = EmbeddingCollection(
        collection_id=text_collection_id,
        snapshot_id=snapshot_id,
        run_id=run_id,
        source_type=EmbeddingSourceType.MEMORY_TEXT,
        model_alias=text_model_alias,
        model_revision=text_model_revision,
        dimension=4096,
        distance_metric=DistanceMetric.COSINE,
        normalized=True,
        normalization_method="l2",
        preprocessing_hash=canonical_content_hash(TEXT_MEMORY_REPRESENTATION),
        index_type="exact",
        status=EmbeddingCollectionStatus.BUILDING,
        metadata={
            "collection_role": "text_memory",
            "representation": TEXT_MEMORY_REPRESENTATION,
            "selection_hash": selection_hash,
        },
    )
    visual = EmbeddingCollection(
        collection_id=visual_collection_id,
        snapshot_id=snapshot_id,
        run_id=run_id,
        source_type=EmbeddingSourceType.MEMORY_IMAGE,
        model_alias=multimodal_model_alias,
        model_revision=multimodal_model_revision,
        dimension=4096,
        distance_metric=DistanceMetric.COSINE,
        normalized=True,
        normalization_method="l2",
        preprocessing_hash=canonical_content_hash(visual_representation),
        index_type="exact",
        status=EmbeddingCollectionStatus.BUILDING,
        metadata={
            "collection_role": "visual_memory",
            "representation": visual_representation,
            "selection_hash": selection_hash,
            **treatment_metadata,
        },
    )
    manifest = MemoryOperatorIndexManifest(
        memory_index_id=index_id,
        run_id=run_id,
        snapshot_id=snapshot_id,
        dataset=dataset,
        selection_hash=selection_hash,
        representation_version=(
            "memory-operator-index-caption-only-v1"
            if memory_raw_image_policy == "forbid"
            else "memory-operator-index-v1"
        ),
        text_collection_id=text_collection_id,
        visual_collection_id=visual_collection_id,
        text_model_alias=text_model_alias,
        text_model_revision=text_model_revision,
        multimodal_model_alias=multimodal_model_alias,
        multimodal_model_revision=multimodal_model_revision,
        content_hash=canonical_content_hash(manifest_identity),
        metadata=treatment_metadata,
    )
    return manifest, text, visual


def program_query_embedding_requests(
    *,
    run_id: str,
    candidates: Sequence[QuestionCandidate],
    text_model_alias: str,
    multimodal_model_alias: str,
) -> tuple[EmbeddingRequest, EmbeddingRequest]:
    ordered = sorted(candidates, key=lambda item: (item.generation_rank, item.qbank_id))
    text_inputs = [
        EmbeddingInput(
            input_id=item.qbank_id,
            text=(
                "Instruct: retrieve memory evidence needed to execute this question.\n"
                f"Query: {item.canonical_question}"
            ),
            metadata={"instruction_version": PROGRAM_TEXT_QUERY_INSTRUCTION},
        )
        for item in ordered
    ]
    joint_inputs = [
        EmbeddingInput(
            input_id=item.qbank_id,
            text=(
                "Instruct: jointly retrieve visual memory evidence for this question.\n"
                f"Query: {item.canonical_question}"
            ),
            # Expected evidence images are intentionally never bound here.
            image_refs=[],
            metadata={"instruction_version": PROGRAM_JOINT_QUERY_INSTRUCTION},
        )
        for item in ordered
    ]
    return (
        EmbeddingRequest(
            request_id=stable_compiler_id(
                "program-text-query-batch",
                [run_id, [item.qbank_id for item in ordered]],
            ),
            model_alias=text_model_alias,
            inputs=text_inputs,
            dimensions=4096,
            metadata={"query_role": "question_text_query"},
        ),
        EmbeddingRequest(
            request_id=stable_compiler_id(
                "program-joint-query-batch",
                [run_id, [item.qbank_id for item in ordered]],
            ),
            model_alias=multimodal_model_alias,
            inputs=joint_inputs,
            dimensions=4096,
            metadata={"query_role": "question_joint_query", "text_only": True},
        ),
    )
