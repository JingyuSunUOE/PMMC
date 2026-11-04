"""Concrete prospective-safe operators used by the PMMC compiler kernel."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, cast

from pmmc.compiler.config import (
    OPERATOR_SET_V1,
    OPERATOR_SET_V2,
    OPERATOR_SET_V3,
    OPERATOR_SET_V4,
    OPERATOR_SET_V5,
    OPERATOR_SET_V6,
    OPERATOR_SET_V7,
    CompilerFoundationConfig,
)
from pmmc.compiler.ids import canonical_content_hash
from pmmc.memory.operators import (
    AbstainArgs,
    ContextAroundImageArgs,
    EmbeddingSearchArgs,
    EvidenceSufficiencyArgs,
    FallbackArgs,
    ImageEvidenceArgs,
    KeywordSearchArgs,
    MemoryOperator,
    OperatorDataRepository,
    OperatorRegistry,
    TemporalNeighborArgs,
)
from pmmc.schemas import (
    BaseSchema,
    EvidenceType,
    OperatorControlAction,
    OperatorEvidence,
    OperatorExecutionContext,
    OperatorOutput,
    OperatorSpec,
)


def _args_schema_hash(args_model: type[BaseSchema]) -> str:
    return canonical_content_hash(args_model.model_json_schema())


def _output_from_evidence(
    evidence: Sequence[OperatorEvidence],
    *,
    evidence_type: EvidenceType,
    metadata: dict[str, object] | None = None,
) -> OperatorOutput:
    memory_ids: list[str] = []
    image_ids: list[str] = []
    session_ids: list[str] = []
    caption_ids: list[str] = []
    scores = {}
    for item in evidence:
        for value, target in (
            (item.memory_id, memory_ids),
            (item.image_id, image_ids),
            (item.session_id, session_ids),
            (item.caption_id, caption_ids),
        ):
            if value is not None and value not in target:
                target.append(value)
        if item.score is not None:
            scores[item.evidence_id] = item.score
    evidence_group_ids: list[str] = []
    for item in evidence:
        singular = item.metadata.get("evidence_group_id")
        plural = item.metadata.get("evidence_group_ids")
        candidates = [
            *((singular,) if isinstance(singular, str) else ()),
            *(plural if isinstance(plural, list) else ()),
        ]
        for group_id in candidates:
            if (
                isinstance(group_id, str)
                and group_id
                and group_id not in evidence_group_ids
            ):
                evidence_group_ids.append(group_id)
    return OperatorOutput(
        retrieved_memory_ids=memory_ids,
        retrieved_image_ids=image_ids,
        retrieved_session_ids=session_ids,
        retrieved_caption_ids=caption_ids,
        scores=scores,
        evidence_type=evidence_type,
        metadata={**(metadata or {}), "evidence_group_ids": evidence_group_ids},
    )


def _previous_ids(
    previous_outputs: Sequence[OperatorOutput],
    field_name: str,
    source_step_ids: Sequence[int] = (),
) -> list[str]:
    result: list[str] = []
    selected = (
        [previous_outputs[index] for index in source_step_ids]
        if source_step_ids
        else previous_outputs
    )
    for output in selected:
        for value in getattr(output, field_name):
            if value not in result:
                result.append(value)
    return result


def _selected_outputs(
    previous_outputs: Sequence[OperatorOutput],
    source_step_ids: Sequence[int],
    *,
    operator_set_version: str,
) -> list[OperatorOutput]:
    if source_step_ids:
        return [previous_outputs[index] for index in source_step_ids]
    # v1 implicitly consumed every prior output. V2 makes dataflow exact: an
    # empty source list means no upstream evidence, including for step-0 abstain.
    if operator_set_version in {
        "pmmc-operators-v2",
        "pmmc-operators-v3",
        "pmmc-operators-v4",
        "pmmc-operators-v5",
        "pmmc-operators-v6",
        "pmmc-operators-v7",
    }:
        return []
    return list(previous_outputs)


_OPERATOR_V3_CAPABILITIES: dict[str, dict[str, list[str]]] = {
    "retrieve_text_by_keyword": {
        "outputs": ["text", "evidence_groups"],
        "modalities": ["text"],
        "server_derived": ["query"],
    },
    "retrieve_text_by_embedding": {
        "outputs": ["text", "evidence_groups"],
        "modalities": ["text"],
    },
    "retrieve_image_by_visual_embedding": {
        "outputs": ["image", "evidence_groups"],
        "modalities": ["image"],
    },
    "retrieve_image_text_pair": {
        "inputs": ["image"],
        "outputs": ["image", "text", "image_text", "evidence_groups"],
        "source_types": ["image_ids"],
        "modalities": ["image", "text"],
    },
    "retrieve_context_around_image": {
        "inputs": ["image"],
        "outputs": ["image", "text", "context", "evidence_groups"],
        "source_types": ["image_ids"],
        "modalities": ["image", "text"],
    },
    "expand_temporal_neighbors": {
        "inputs": ["text"],
        "outputs": ["text", "temporal", "evidence_groups"],
        "source_types": ["memory_ids"],
        "modalities": ["text"],
    },
    "inspect_raw_image": {
        "inputs": ["image"],
        "outputs": ["image", "raw_image"],
        "source_types": ["image_ids"],
        "modalities": ["raw_image"],
    },
    "verify_evidence_sufficiency": {
        "inputs": ["evidence_groups"],
        "outputs": ["verification"],
    },
    "fallback_to_multimodal_rag": {
        "inputs": ["verification"],
        "outputs": ["control"],
    },
    "abstain_if_no_evidence": {"outputs": ["control"]},
}


class _OperatorBase:
    name: ClassVar[str]
    description: ClassVar[str]
    evidence_type: ClassVar[EvidenceType]
    output_type: ClassVar[str]
    args_model: ClassVar[type[BaseSchema]]
    terminal: ClassVar[bool] = False
    collection_role: ClassVar[str | None] = None

    def __init__(
        self,
        data: OperatorDataRepository,
        config: CompilerFoundationConfig,
    ) -> None:
        self.data = data
        self.config = config
        capability_contract = _OPERATOR_V3_CAPABILITIES.get(self.name, {})
        self.spec = OperatorSpec(
            name=self.name,
            version=config.operator_set_version,
            description=self.description,
            args_schema_hash=_args_schema_hash(self.args_model),
            output_type=self.output_type,
            evidence_type=self.evidence_type,
            terminal=self.terminal,
            collection_role=self.collection_role,
            max_top_k=config.operator_top_k_max
            if issubclass(self.args_model, (KeywordSearchArgs, EmbeddingSearchArgs))
            else None,
            metadata=(
                {
                    "input_capabilities": capability_contract.get("inputs", []),
                    "output_capabilities": capability_contract.get("outputs", []),
                    "allowed_source_step_types": capability_contract.get(
                        "source_types", []
                    ),
                    "modality_contribution": capability_contract.get(
                        "modalities", []
                    ),
                    "server_derived_parameters": capability_contract.get(
                        "server_derived", []
                    ),
                }
                if config.operator_set_version
                in {
                    "pmmc-operators-v3",
                    "pmmc-operators-v4",
                    "pmmc-operators-v5",
                    "pmmc-operators-v6",
                    "pmmc-operators-v7",
                }
                else {}
            ),
        )


class RetrieveTextByKeyword(_OperatorBase):
    name = "retrieve_text_by_keyword"
    description = "Retrieve visible memory text and captions with PostgreSQL FTS."
    evidence_type = EvidenceType.MEMORY
    output_type = "memory_ids"
    args_model = KeywordSearchArgs

    async def execute(
        self,
        context: OperatorExecutionContext,
        args: BaseSchema,
        previous_outputs: Sequence[OperatorOutput],
    ) -> OperatorOutput:
        del previous_outputs
        parsed = KeywordSearchArgs.model_validate(args)
        evidence = await self.data.search_keyword(
            context,
            query=parsed.query or context.question,
            top_k=parsed.top_k,
            entity_mentions=list(parsed.entity_mentions),
        )
        return _output_from_evidence(evidence, evidence_type=self.evidence_type)


class _EmbeddingOperator(_OperatorBase):
    args_model = EmbeddingSearchArgs

    async def execute(
        self,
        context: OperatorExecutionContext,
        args: BaseSchema,
        previous_outputs: Sequence[OperatorOutput],
    ) -> OperatorOutput:
        del previous_outputs
        parsed = EmbeddingSearchArgs.model_validate(args)
        assert self.collection_role is not None
        collection_id = context.collection_bindings.get(self.collection_role)
        if collection_id is None:
            raise ValueError(f"missing collection binding: {self.collection_role}")
        vector = context.query_vectors.get(parsed.vector_role)
        if vector is None:
            raise ValueError(f"missing query vector role: {parsed.vector_role}")
        evidence = await self.data.search_embedding(
            context,
            collection_id=collection_id,
            vector=list(vector),
            top_k=parsed.top_k,
            evidence_type=self.evidence_type,
        )
        return _output_from_evidence(evidence, evidence_type=self.evidence_type)


class RetrieveTextByEmbedding(_EmbeddingOperator):
    name = "retrieve_text_by_embedding"
    description = "Search the frozen text-memory collection with exact pgvector scan."
    evidence_type = EvidenceType.MEMORY
    output_type = "memory_ids"
    collection_role = "text_memory"


class RetrieveImageByVisualEmbedding(_EmbeddingOperator):
    name = "retrieve_image_by_visual_embedding"
    description = "Search the frozen visual-memory collection with exact pgvector scan."
    evidence_type = EvidenceType.IMAGE
    output_type = "image_ids"
    collection_role = "visual_memory"


class RetrieveImageTextPair(_OperatorBase):
    name = "retrieve_image_text_pair"
    description = "Resolve image evidence to its canonical memory and text lineage."
    evidence_type = EvidenceType.IMAGE_TEXT_PAIR
    output_type = "image_text_pairs"
    args_model = ImageEvidenceArgs

    async def execute(
        self,
        context: OperatorExecutionContext,
        args: BaseSchema,
        previous_outputs: Sequence[OperatorOutput],
    ) -> OperatorOutput:
        parsed = ImageEvidenceArgs.model_validate(args)
        selected = _selected_outputs(
            previous_outputs,
            parsed.source_step_ids,
            operator_set_version=self.config.operator_set_version,
        )
        image_ids = list(parsed.image_ids) or _previous_ids(selected, "retrieved_image_ids")
        evidence = await self.data.resolve_image_text_pairs(context, image_ids)
        return _output_from_evidence(evidence, evidence_type=self.evidence_type)


class RetrieveContextAroundImage(_OperatorBase):
    name = "retrieve_context_around_image"
    description = "Expand image evidence to its turn and neighboring session turns."
    evidence_type = EvidenceType.MEMORY
    output_type = "memory_ids"
    args_model = ContextAroundImageArgs

    async def execute(
        self,
        context: OperatorExecutionContext,
        args: BaseSchema,
        previous_outputs: Sequence[OperatorOutput],
    ) -> OperatorOutput:
        parsed = ContextAroundImageArgs.model_validate(args)
        selected = _selected_outputs(
            previous_outputs,
            parsed.source_step_ids,
            operator_set_version=self.config.operator_set_version,
        )
        image_ids = list(parsed.image_ids) or _previous_ids(selected, "retrieved_image_ids")
        evidence = await self.data.context_around_images(context, image_ids, parsed.radius)
        return _output_from_evidence(evidence, evidence_type=self.evidence_type)


class ExpandTemporalNeighbors(_OperatorBase):
    name = "expand_temporal_neighbors"
    description = "Expand memory evidence by canonical temporal neighbors."
    evidence_type = EvidenceType.MEMORY
    output_type = "memory_ids"
    args_model = TemporalNeighborArgs

    async def execute(
        self,
        context: OperatorExecutionContext,
        args: BaseSchema,
        previous_outputs: Sequence[OperatorOutput],
    ) -> OperatorOutput:
        parsed = TemporalNeighborArgs.model_validate(args)
        selected = _selected_outputs(
            previous_outputs,
            parsed.source_step_ids,
            operator_set_version=self.config.operator_set_version,
        )
        memory_ids = list(parsed.memory_ids) or _previous_ids(selected, "retrieved_memory_ids")
        evidence = await self.data.expand_temporal_neighbors(context, memory_ids, parsed.radius)
        return _output_from_evidence(evidence, evidence_type=self.evidence_type)


class InspectRawImage(_OperatorBase):
    name = "inspect_raw_image"
    description = "Validate and materialize runtime-safe canonical image references."
    evidence_type = EvidenceType.RAW_IMAGE
    output_type = "image_ids"
    args_model = ImageEvidenceArgs

    async def execute(
        self,
        context: OperatorExecutionContext,
        args: BaseSchema,
        previous_outputs: Sequence[OperatorOutput],
    ) -> OperatorOutput:
        parsed = ImageEvidenceArgs.model_validate(args)
        selected = _selected_outputs(
            previous_outputs,
            parsed.source_step_ids,
            operator_set_version=self.config.operator_set_version,
        )
        image_ids = list(parsed.image_ids) or _previous_ids(selected, "retrieved_image_ids")
        evidence = await self.data.inspect_images(context, image_ids)
        return _output_from_evidence(evidence, evidence_type=self.evidence_type)


class VerifyEvidenceSufficiency(_OperatorBase):
    name = "verify_evidence_sufficiency"
    description = "Perform a structural modality and evidence-count sufficiency check."
    evidence_type = EvidenceType.UNKNOWN
    output_type = "verification"
    args_model = EvidenceSufficiencyArgs

    async def execute(
        self,
        context: OperatorExecutionContext,
        args: BaseSchema,
        previous_outputs: Sequence[OperatorOutput],
    ) -> OperatorOutput:
        parsed = EvidenceSufficiencyArgs.model_validate(args)
        selected = _selected_outputs(
            previous_outputs,
            parsed.source_step_ids,
            operator_set_version=self.config.operator_set_version,
        )
        memory_ids = _previous_ids(selected, "retrieved_memory_ids")
        image_ids = _previous_ids(selected, "retrieved_image_ids")
        caption_ids = _previous_ids(selected, "retrieved_caption_ids")
        session_ids = _previous_ids(selected, "retrieved_session_ids")
        available_modalities = set()
        if memory_ids:
            available_modalities.add("text")
        if caption_ids:
            available_modalities.add("caption")
        if image_ids:
            available_modalities.add("image")
        if any(
            output.evidence_type is EvidenceType.RAW_IMAGE
            and bool(output.retrieved_image_ids)
            for output in selected
        ):
            available_modalities.add("raw_image")
        count = len(set(memory_ids + image_ids + caption_ids + session_ids))
        evidence_group_ids = list(
            dict.fromkeys(
                group_id
                for output in selected
                for group_id in output.metadata.get("evidence_group_ids", [])
                if isinstance(group_id, str) and group_id
            )
        )
        group_count = len(evidence_group_ids) if evidence_group_ids else count
        minimum_groups = (
            parsed.minimum_evidence_groups
            if parsed.minimum_evidence_groups is not None
            else parsed.minimum_evidence
        )
        required_modalities = parsed.required_modalities or context.required_modalities
        required = {value.value for value in required_modalities}
        sufficient = group_count >= minimum_groups and required.issubset(available_modalities)
        return OperatorOutput(
            retrieved_memory_ids=memory_ids,
            retrieved_image_ids=image_ids,
            retrieved_caption_ids=caption_ids,
            retrieved_session_ids=session_ids,
            evidence_type=self.evidence_type,
            metadata={
                "sufficient": sufficient,
                "sufficiency_status": "sufficient" if sufficient else "insufficient",
                "evidence_count": count,
                "evidence_group_count": group_count,
                "evidence_group_ids": evidence_group_ids,
                "required_modalities": sorted(required),
            },
        )


class FallbackToMultimodalRag(_OperatorBase):
    name = "fallback_to_multimodal_rag"
    description = "Emit a terminal fallback signal without executing the fallback."
    evidence_type = EvidenceType.UNKNOWN
    output_type = "control"
    args_model = FallbackArgs
    terminal = True

    async def execute(
        self,
        context: OperatorExecutionContext,
        args: BaseSchema,
        previous_outputs: Sequence[OperatorOutput],
    ) -> OperatorOutput:
        del context
        parsed = FallbackArgs.model_validate(args)
        selected = _selected_outputs(
            previous_outputs,
            parsed.source_step_ids,
            operator_set_version=self.config.operator_set_version,
        )
        sufficient = any(output.metadata.get("sufficient") is True for output in selected)
        should_fallback = parsed.condition == "always" or not sufficient
        return OperatorOutput(
            control_action=(
                OperatorControlAction.FALLBACK
                if should_fallback
                else OperatorControlAction.CONTINUE
            ),
            metadata={
                "reason_code": parsed.reason_code,
                "condition": parsed.condition,
                "evidence_sufficient": sufficient,
            },
        )


class AbstainIfNoEvidence(_OperatorBase):
    name = "abstain_if_no_evidence"
    description = "Emit a terminal abstain signal when accumulated evidence is insufficient."
    evidence_type = EvidenceType.UNKNOWN
    output_type = "control"
    args_model = AbstainArgs
    terminal = True

    async def execute(
        self,
        context: OperatorExecutionContext,
        args: BaseSchema,
        previous_outputs: Sequence[OperatorOutput],
    ) -> OperatorOutput:
        del context
        parsed = AbstainArgs.model_validate(args)
        selected = _selected_outputs(
            previous_outputs,
            parsed.source_step_ids,
            operator_set_version=self.config.operator_set_version,
        )
        identifiers = set()
        for field_name in (
            "retrieved_memory_ids",
            "retrieved_image_ids",
            "retrieved_caption_ids",
            "retrieved_session_ids",
        ):
            identifiers.update(_previous_ids(selected, field_name))
        groups = {
            group_id
            for output in selected
            for group_id in output.metadata.get("evidence_group_ids", [])
            if isinstance(group_id, str) and group_id
        }
        group_count = len(groups) if groups else len(identifiers)
        minimum_groups = (
            parsed.minimum_evidence_groups
            if parsed.minimum_evidence_groups is not None
            else parsed.minimum_evidence
        )
        available_modalities = set()
        if any(output.retrieved_memory_ids for output in selected):
            available_modalities.add("text")
        if any(output.retrieved_caption_ids for output in selected):
            available_modalities.add("caption")
        if any(output.retrieved_image_ids for output in selected):
            available_modalities.add("image")
        if any(
            output.evidence_type is EvidenceType.RAW_IMAGE
            and bool(output.retrieved_image_ids)
            for output in selected
        ):
            available_modalities.add("raw_image")
        required = {item.value for item in parsed.required_modalities}
        should_abstain = group_count < minimum_groups or not required.issubset(available_modalities)
        return OperatorOutput(
            control_action=OperatorControlAction.ABSTAIN
            if should_abstain
            else OperatorControlAction.CONTINUE,
            metadata={
                "reason_code": parsed.reason_code,
                "evidence_count": len(identifiers),
                "evidence_group_count": group_count,
                "required_modalities": sorted(required),
            },
        )


def build_operator_registry(
    data: OperatorDataRepository,
    config: CompilerFoundationConfig,
) -> OperatorRegistry:
    """Build the immutable registry in its protocol-defined order."""

    operators = cast(
        Sequence[MemoryOperator],
        (
            RetrieveTextByKeyword(data, config),
            RetrieveTextByEmbedding(data, config),
            RetrieveImageByVisualEmbedding(data, config),
            RetrieveImageTextPair(data, config),
            RetrieveContextAroundImage(data, config),
            ExpandTemporalNeighbors(data, config),
            InspectRawImage(data, config),
            VerifyEvidenceSufficiency(data, config),
            FallbackToMultimodalRag(data, config),
            AbstainIfNoEvidence(data, config),
        ),
    )
    names = tuple(operator.spec.name for operator in operators)
    expected = {
        "pmmc-operators-v1": OPERATOR_SET_V1,
        "pmmc-operators-v2": OPERATOR_SET_V2,
        "pmmc-operators-v3": OPERATOR_SET_V3,
        "pmmc-operators-v4": OPERATOR_SET_V4,
        "pmmc-operators-v5": OPERATOR_SET_V5,
        "pmmc-operators-v6": OPERATOR_SET_V6,
        "pmmc-operators-v7": OPERATOR_SET_V7,
    }[config.operator_set_version]
    if names != expected:
        raise RuntimeError("operator registry order does not match its protocol")
    return OperatorRegistry(config.operator_set_version, operators)


__all__ = [
    "AbstainIfNoEvidence",
    "ExpandTemporalNeighbors",
    "FallbackToMultimodalRag",
    "InspectRawImage",
    "RetrieveContextAroundImage",
    "RetrieveImageByVisualEmbedding",
    "RetrieveImageTextPair",
    "RetrieveTextByEmbedding",
    "RetrieveTextByKeyword",
    "VerifyEvidenceSufficiency",
    "build_operator_registry",
]
