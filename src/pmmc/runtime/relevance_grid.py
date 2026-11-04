"""Gold-free relevance projections of one immutable program execution.

Grid members are not additional experiment cells. They identify the *actual*
evidence subset whose prepared request and answer can be reused offline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from copy import copy
from dataclasses import dataclass, replace
from itertools import product
from typing import Any

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.memory.operators import ProgramExecutionResult
from pmmc.runtime.answer_context import PmmcEvidenceAnswerInputProvider, RuntimeMemoryReader
from pmmc.runtime.evidence_sufficiency import (
    EvidenceGroupResolver,
    assess_query_evidence_sufficiency,
)
from pmmc.runtime.query_executor_config import JOINT_POLICY, JOINT_THRESHOLD
from pmmc.schemas import (
    AnswerGenerationInput,
    MemoryUnit,
    Modality,
    PlannerRequirementContract,
    ProgramExecutionEvidenceRef,
    QueryEvidenceRelevanceItem,
    QueryEvidenceSufficiencyAssessment,
    QueryProgramBinding,
)

RELEVANCE_GRID_PROTOCOL = "runtime-relevance-context-grid-v1"
DEFERRED_GROUNDING_ROLES = frozenset({"independent_group_relation"})


def requires_independent_relation(requirement: PlannerRequirementContract) -> bool:
    return not requirement.refusal_only and (
        requirement.minimum_evidence_groups > 1
        or requirement.metadata.get("independent_group_relation_required") is True
    )


def bind_runtime_relation_contract(
    prepared: AnswerGenerationInput, *, result: ProgramExecutionResult,
    minimum_groups: int, relation_required: bool,
) -> AnswerGenerationInput:
    if not relation_required:
        return prepared
    from pmmc.runtime.answering import validate_typed_answer_input

    if (result.trace.query_id != prepared.query.query_id
            or result.trace.snapshot_id != prepared.snapshot_id
            or result.trace.run_id != prepared.source_lineage.get(
                "source_execution_run_id", prepared.run_id
            )):
        raise ValueError("relation grounding evidence escaped the query scope")
    context = validate_typed_answer_input(prepared)
    groups_by_evidence: dict[str, set[str]] = {}
    for reference in result.evidence:
        groups = reference.metadata.get("evidence_group_ids", [])
        if not isinstance(groups, list) or any(not isinstance(group, str) for group in groups):
            raise ValueError("relation groups must come from typed runtime evidence")
        groups = set(groups)
        singular = reference.metadata.get("evidence_group_id")
        if isinstance(singular, str) and singular:
            groups.add(singular)
        if not groups:
            groups.add(f"{reference.evidence_type.value}:{reference.evidence_id}")
        groups_by_evidence.setdefault(reference.evidence_id, set()).update(groups)
    all_groups = sorted({group for source in context.sources
                         for group in groups_by_evidence.get(source.evidence_id, ())})
    aliases = {group: f"G{index:02d}" for index, group in enumerate(all_groups, 1)}
    contract: dict[str, Any] = {
        "protocol": "runtime-independent-relation-grounding-v1",
        "required_semantic_roles": ["independent_group_relation"],
        "minimum_evidence_groups": minimum_groups,
        "semantic_relation_status": "not_evaluated",
        "sources": [{"evidence_alias": source.evidence_alias,
                     "group_aliases": sorted(aliases[group] for group
                                             in groups_by_evidence.get(source.evidence_id, ()))}
                    for source in context.sources],
    }
    if minimum_groups < 2 or any(not source["group_aliases"] for source in contract["sources"]):
        raise ValueError("independent relation grounding requires scoped independent groups")
    contract["content_hash"] = canonical_content_hash(contract)
    return prepared.model_copy(update={
        "metadata": {**prepared.metadata, "runtime_relation_contract": contract},
    })


def validate_relevance_grid_points(
    points: list[dict[str, Any]], *, run_id: str, query_id: str,
    execution_id: str, thresholds: list[float],
) -> None:
    expected = set(product(thresholds, repeat=2))
    actual = {(point.get("text_threshold"), point.get("multimodal_threshold")) for point in points}
    if actual != expected or len(points) != len(expected):
        raise ValueError("relevance context grid has an incomplete threshold denominator")
    for point in points:
        hashes = point.get("selected_reference_hashes")
        if (
            point.get("protocol") != RELEVANCE_GRID_PROTOCOL
            or point.get("query_executor_run_id") != run_id
            or point.get("query_id") != query_id
            or point.get("execution_id") != execution_id
            or not isinstance(hashes, list)
            or any(not isinstance(value, str) for value in hashes)
            or len(hashes) != len(set(hashes))
            or point.get("evidence_subset_hash") != canonical_content_hash(hashes)
        ):
            raise ValueError("relevance context grid has incompatible execution lineage")


class _RelevanceMemoryReader:
    def __init__(self, source: RuntimeMemoryReader, image_ids: frozenset[str]) -> None:
        self.source = source
        self.image_ids = image_ids

    async def iter_visible(
        self, snapshot_id: str, conversation_id: str, cutoff_session_id: str
    ) -> AsyncIterator[MemoryUnit]:
        async for memory in self.source.iter_visible(
            snapshot_id, conversation_id, cutoff_session_id
        ):
            yield memory.model_copy(update={"image_refs": [
                image for image in memory.image_refs if image.image_id in self.image_ids
            ]})


def relevance_answer_input_provider(
    provider: PmmcEvidenceAnswerInputProvider, result: ProgramExecutionResult, *,
    direct_images_only: bool = False,
) -> PmmcEvidenceAnswerInputProvider:
    """A text-memory hit cannot silently reintroduce unscored owning images."""
    scoped = copy(provider)
    scoped.memories = _RelevanceMemoryReader(
        provider.memories,
        frozenset() if direct_images_only else
        frozenset(item.image_id for item in result.evidence if item.image_id is not None),
    )
    return scoped


def evidence_reference_hash(reference: ProgramExecutionEvidenceRef) -> str:
    return canonical_content_hash(reference.evidence_ref_id)


def evidence_source_hashes(reference: ProgramExecutionEvidenceRef) -> set[str]:
    sources: list[str] = []
    if reference.image_id is not None:
        sources.append(reference.image_id)
    if reference.memory_id is not None:
        sources.extend(
            (f"memory-text:{reference.memory_id}", f"memory-caption:{reference.memory_id}")
        )
    owner = reference.metadata.get("memory_id")
    if reference.caption_id is not None and isinstance(owner, str) and owner:
        sources.append(f"memory-caption:{owner}")
    return {canonical_content_hash(source) for source in sources}


def select_execution_evidence(
    result: ProgramExecutionResult, reference_hashes: list[str] | tuple[str, ...]
) -> ProgramExecutionResult:
    wanted = set(reference_hashes)
    available = {evidence_reference_hash(item) for item in result.evidence}
    if len(wanted) != len(reference_hashes) or not wanted <= available:
        raise ValueError("relevance context references escaped the source execution")
    return replace(
        result,
        evidence=tuple(
            item for item in result.evidence if evidence_reference_hash(item) in wanted
        ),
    )


@dataclass(frozen=True)
class RuntimeRelevanceProjection:
    point: dict[str, Any]
    result: ProgramExecutionResult
    sufficiency: QueryEvidenceSufficiencyAssessment


def score_passes(item: QueryEvidenceRelevanceItem, text: float, image: float) -> bool:
    if item.metadata.get("joint_ineligible_reason") == "query_image":
        return False
    if item.metadata.get("embedding_channel") == "joint":
        return item.cosine_similarity >= JOINT_THRESHOLD
    return item.cosine_similarity >= (image if item.modality is Modality.IMAGE else text)


def project_runtime_relevance_grid(
    *,
    run_id: str,
    cell_id: str,
    binding: QueryProgramBinding,
    requirement: PlannerRequirementContract,
    result: ProgramExecutionResult,
    items: tuple[QueryEvidenceRelevanceItem, ...],
    thresholds: list[float],
    group_resolver: EvidenceGroupResolver | None = None,
    joint_enabled: bool = False,
) -> tuple[RuntimeRelevanceProjection, ...]:
    if not thresholds or len(thresholds) != len(set(thresholds)):
        raise ValueError("relevance grid must declare unique cosine thresholds")
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("invalid relevance cosine threshold")
    roles = requirement.metadata.get("required_semantic_roles", [])
    if not isinstance(roles, list) or any(not isinstance(role, str) for role in roles):
        raise ValueError("relevance roles must be a runtime-safe string list")
    for item in items:
        if joint_enabled and item.metadata.get("embedding_channel") not in {
            "text", "multimodal", "joint",
        }:
            raise ValueError("release R requires separately identified score channels")
        if item.metadata.get("embedding_channel") == "joint" and not (
            joint_enabled and item.threshold == JOINT_THRESHOLD and item.modality is Modality.IMAGE
            and item.metadata.get("joint_policy") == JOINT_POLICY
            and item.metadata.get("image_id_hash") == item.evidence_id_hash
        ):
            raise ValueError("joint score has incompatible release R lineage")
        if (
            item.execution_id != result.trace.execution_id
            or item.query_id != binding.query_id
            or item.query_executor_run_id != run_id
            or item.metadata.get("score_transform") != "raw-cosine-v1"
        ):
            raise ValueError("relevance grid requires scoped raw-cosine execution scores")

    source_items: dict[str, list[QueryEvidenceRelevanceItem]] = {}
    for item in items:
        source_items.setdefault(item.evidence_id_hash, []).append(item)
    score_hash = canonical_content_hash(sorted(item.content_hash for item in items))
    candidates = {
        evidence_reference_hash(ref): [
            item for source in evidence_source_hashes(ref) for item in source_items.get(source, ())
            if not joint_enabled or (item.modality is Modality.IMAGE) == (ref.image_id is not None)
        ] for ref in result.evidence
    }
    # A joint score does not repair a missing historical raw-vector score.
    unscored_references = [
        key
        for key, values in candidates.items()
        if not any(i.metadata.get("embedding_channel") != "joint" for i in values)
    ]
    projections: list[RuntimeRelevanceProjection] = []
    for text_threshold, image_threshold in product(sorted(thresholds), repeat=2):
        selected = [
            reference
            for reference in result.evidence
            if any(
                score_passes(item, text_threshold, image_threshold)
                for item in candidates[evidence_reference_hash(reference)]
            )
        ]
        hashes = [evidence_reference_hash(item) for item in selected]
        filtered = select_execution_evidence(result, hashes)
        structural = assess_query_evidence_sufficiency(
            query_executor_run_id=run_id,
            binding=binding,
            requirement=requirement,
            result=filtered,
            group_resolver=group_resolver,
        )
        present_roles: set[str] = set()
        for reference in selected:
            values = reference.metadata.get("semantic_roles", [])
            if not isinstance(values, list) or any(not isinstance(role, str) for role in values):
                raise ValueError("execution semantic roles must be strings")
            present_roles.update(values)
        operator_roles = set(roles) - DEFERRED_GROUNDING_ROLES
        roles_passed = operator_roles <= present_roles
        sufficient = structural.sufficient and roles_passed and not unscored_references
        point = {
            "protocol": RELEVANCE_GRID_PROTOCOL,
            "query_executor_run_id": run_id,
            "query_id": binding.query_id,
            "source_cell_id": cell_id,
            "execution_id": result.trace.execution_id,
            "binding_id": binding.query_program_binding_id,
            "text_threshold": text_threshold,
            "multimodal_threshold": image_threshold,
            "score_manifest_hash": score_hash,
            "score_coverage_complete": not unscored_references,
            "unscored_reference_hashes": unscored_references,
            "selected_reference_hashes": hashes,
            "evidence_subset_hash": canonical_content_hash(hashes),
            "structural_requirement_success": structural.sufficient,
            "required_roles_satisfied": roles_passed,
            "sufficient": sufficient,
            "answer_support": None,
            "semantic_relation_status": (
                "pending_grounding" if requires_independent_relation(requirement)
                else "not_applicable"
            ),
            "status": ("incomplete" if unscored_references else
                       "pending_answer" if sufficient else "relevance_rejected"),
            "prepared_key": None,
            "answer_identity_hash": None,
            "typed_answer_id": None,
            "answer_model_call_id": None,
            "grounding_call_ids": [],
            "terminal_state": None,
            "error_type": "MissingEvidenceRelevanceScore" if unscored_references else None,
        }
        if joint_enabled:
            point["joint_policy"] = JOINT_POLICY
            point["joint_threshold"] = JOINT_THRESHOLD
            point["selection_channels"] = {
                key: sorted({i.metadata["embedding_channel"] for i in values
                             if score_passes(i, text_threshold, image_threshold)})
                for key, values in candidates.items()
            }
            point["joint_image_hashes"] = sorted({
                item.evidence_id_hash for item in items
                if item.metadata.get("embedding_channel") == "joint" and item.relevant
            })
            point["arrived_joint_image_hashes"] = []
        point["grid_point_id"] = stable_compiler_id("query-relevance-grid-point", point)
        payload = structural.model_dump(mode="python")
        payload.update(
            structural_requirement_success=structural.sufficient,
            query_evidence_relevance=sufficient,
            answer_support=None,
            relevant_evidence_group_count=structural.independent_group_count,
            relevance_threshold=text_threshold,
            sufficient=sufficient,
            failure_reason=(
                None if sufficient
                else structural.failure_reason or "runtime_relevance_insufficient"
            ),
            sufficiency_assessment_id=stable_compiler_id(
                "query-grid-sufficiency", [point["grid_point_id"], structural.content_hash]
            ),
            content_hash=canonical_content_hash(
                [point["grid_point_id"], structural.content_hash, sufficient]
            ),
            metadata={
                "sufficiency_policy_version": "query-runtime-evidence-v3",
                "answer_grounding_status": "not_evaluated",
                "relevance_grid_point_id": point["grid_point_id"],
                "channel_thresholds": {"text": text_threshold, "multimodal": image_threshold},
                "required_roles_satisfied": roles_passed,
                "relevance_assessment_status": "incomplete" if unscored_references else "evaluated",
            },
        )
        projections.append(RuntimeRelevanceProjection(
            point, filtered, QueryEvidenceSufficiencyAssessment.model_validate(payload)
        ))
    return tuple(projections)


def bind_joint_context(
    prepared: AnswerGenerationInput,
    *,
    result: ProgramExecutionResult,
    binding: QueryProgramBinding,
    requirement: PlannerRequirementContract,
    joint_image_hashes: list[str],
    group_resolver: EvidenceGroupResolver | None = None,
) -> AnswerGenerationInput:
    """Recheck the program against actual evidence parts after this backbone's budget."""
    from pmmc.runtime.answering import validate_typed_answer_input

    if (requirement.content_hash != binding.requirement_contract_hash
            or prepared.source_lineage.get("execution_id") != result.trace.execution_id):
        raise ValueError("joint context escaped its program or execution")
    context = validate_typed_answer_input(prepared)
    sources = {source.evidence_id: source for source in context.sources}
    retained = []
    image_hashes = set()
    for ref in result.evidence:
        source = sources.get(ref.evidence_id)
        if source is None:
            continue
        if ref.image_id is not None:
            if not any(
                i.image is not None and i.image.image_id == ref.image_id for i in source.images
            ):
                continue
            image_hashes.add(canonical_content_hash(ref.image_id))
        retained.append(ref)
    supplied = replace(result, evidence=tuple(retained))
    structural = assess_query_evidence_sufficiency(
        query_executor_run_id=binding.query_executor_run_id, binding=binding,
        requirement=requirement, result=supplied, group_resolver=group_resolver,
    )
    roles = {role for ref in retained for role in ref.metadata.get("semantic_roles", [])}
    passed = structural.sufficient and (
        set(requirement.metadata.get("required_semantic_roles", [])) - DEFERRED_GROUNDING_ROLES
        <= roles
    )
    receipt = {
        "policy": JOINT_POLICY,
        "execution_id": result.trace.execution_id,
        "requirement_hash": requirement.content_hash,
        "context_sufficient": passed,
        "context_hash": canonical_content_hash(context.model_dump(mode="json")),
        "supplied_reference_hashes": [evidence_reference_hash(ref) for ref in retained],
        "prepared_joint_image_hashes": sorted(image_hashes.intersection(joint_image_hashes))
        if passed
        else [],
    }
    receipt["content_hash"] = canonical_content_hash(receipt)
    return prepared.model_copy(update={"metadata": {
        **prepared.metadata, "joint_context": receipt,
        "joint_requirement": requirement.model_dump(mode="json", exclude={"created_at"}),
        "joint_image_hashes": joint_image_hashes,
    }})


def validate_joint_outcome(point: dict, contexts: dict) -> None:
    if point.get("joint_policy") not in {None, JOINT_POLICY}:
        raise ValueError("unsupported joint evidence policy")
    if point.get("joint_policy") != JOINT_POLICY:
        if point.get("status") == "context_rejected":
            raise ValueError("context rejection requires release R lineage")
        return
    arrived = point.get("arrived_joint_image_hashes")
    key = point.get("prepared_key")
    if key is None:
        if arrived != []:
            raise ValueError("unprepared point claims joint evidence arrival")
        return
    receipt = contexts.get(key)
    if not isinstance(receipt, dict) or (
        receipt.get("policy") != JOINT_POLICY
        or receipt.get("execution_id") != point.get("execution_id")
        or receipt.get("content_hash")
        != canonical_content_hash({k: v for k, v in receipt.items() if k != "content_hash"})
        or not set(receipt["supplied_reference_hashes"]) <= set(point["selected_reference_hashes"])
        or not set(receipt["prepared_joint_image_hashes"]) <= set(point["joint_image_hashes"])
    ):
        raise ValueError("joint answer context receipt differs from its projection")
    if point["status"] == "context_rejected":
        if (
            receipt["context_sufficient"]
            or arrived != []
            or point.get("typed_answer_id") is not None
            or point.get("answer_model_call_id") is not None
            or point.get("answer_support") is not None
            or point.get("terminal_state") is not None
            or point.get("grounding_call_ids") != []
        ):
            raise ValueError("budget-rejected joint context cannot have answer evidence")
    elif point["status"] == "completed" or (
        point["status"] == "incomplete" and point.get("answer_model_call_id")
    ):
        if not receipt["context_sufficient"] or arrived != receipt["prepared_joint_image_hashes"]:
            raise ValueError("joint evidence arrival differs from the actual answer context")
    elif arrived != []:
        raise ValueError("ungenerated context claims joint evidence arrival")


def resolve_relevance_grid_answers(
    points: list[dict[str, Any]],
    answers: dict[str, Any],
    errors: dict[str, str],
    joint_contexts: dict[str, Any] | None = None,
    generated_calls: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve prepared keys only; absence must never masquerade as a negative answer."""
    resolved = []
    for original in points:
        point = dict(original)
        key = point.get("prepared_key")
        answer = answers.get(key) if isinstance(key, str) else None
        if key is not None and point.get("joint_policy") == JOINT_POLICY:
            receipt = (joint_contexts or {})[key]
            if not receipt["context_sufficient"]:
                if answer is not None:
                    raise ValueError("answer generated from rejected joint context")
                point["status"] = "context_rejected"
                resolved.append(point)
                continue
            call_id = (generated_calls or {}).get(key)
            if call_id:
                point["answer_model_call_id"] = call_id
                point["arrived_joint_image_hashes"] = receipt["prepared_joint_image_hashes"]
        if answer is not None:
            if point.get("joint_policy") == JOINT_POLICY:
                point["arrived_joint_image_hashes"] = receipt["prepared_joint_image_hashes"]
            point.update(
                status="completed", typed_answer_id=answer.typed_answer_id,
                answer_model_call_id=answer.answer_model_call_id,
                grounding_call_ids=[call.model_call_id for call in answer.calls
                                    if call.role in {"answer_grounding", "grounding_format_retry"}],
                terminal_state=answer.terminal_state,
                answer_support=(answer.grounding.assessment.supported
                                if answer.grounding is not None else False),
            )
            if point.get("semantic_relation_status") == "pending_grounding":
                point["semantic_relation_status"] = (
                    "passed" if answer.grounding.assessment.supported else "failed"
                ) if answer.grounding is not None else "not_evaluated"
        elif key is not None:
            point.update(status="incomplete", error_type=errors.get(key, "MissingGridAnswer"))
        resolved.append(point)
    return resolved
