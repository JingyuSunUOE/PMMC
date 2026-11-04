"""Runtime-safe evidence-use roles, not answer entailment or private support IDs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pmmc.schemas.program import OperatorOutput, ProgramStep

EVIDENCE_ROLE_POLICY = "operator-evidence-use-roles-v1"


def required_evidence_roles(
    *,
    modalities: Iterable[str],
    capabilities: Iterable[str],
    refusal_only: bool = False,
) -> list[str]:
    """Derive roles only from the public modality/capability contract."""
    if refusal_only:
        return []
    modalities, capabilities = set(modalities), set(capabilities)
    roles = {
        role
        for modality, role in (
            ("text", "text_support"),
            ("caption", "caption_support"),
            ("image", "image_support"),
            ("raw_image", "raw_image_observation"),
        )
        if modality in modalities
    }
    roles.update(
        role
        for capability, role in (
            ("image_text", "image_text_binding"),
            ("context", "image_context"),
            ("temporal", "temporal_neighbor"),
            ("raw_image", "raw_image_observation"),
        )
        if capability in capabilities
    )
    return sorted(roles)


def produced_evidence_roles(
    *,
    step: ProgramStep,
    output: OperatorOutput,
    previous: Sequence[OperatorOutput],
    reference_kind: str,
    identifier: str,
) -> list[str]:
    """Label observed operator contributions, never assign required roles wholesale.

    These labels express evidence use, not entity-level semantic verification. In
    particular a temporal expansion that merely echoes its sources has no temporal
    role. Verification/control steps cannot mint new support roles.
    """
    if not output.success or step.op_name in {
        "verify_evidence_sufficiency",
        "fallback_to_multimodal_rag",
        "abstain_if_no_evidence",
    }:
        return []
    role = {
        "memory": "text_support",
        "caption": "caption_support",
        "image": "image_support",
    }.get(reference_kind)
    roles = {role} if role else set()
    if step.op_name == "inspect_raw_image" and reference_kind == "image":
        roles.add("raw_image_observation")
    if step.op_name == "retrieve_image_text_pair" and (
        output.retrieved_memory_ids
        and output.retrieved_image_ids
        and reference_kind in {"memory", "image"}
    ):
        roles.add("image_text_binding")
    if reference_kind == "memory" and identifier not in {
        memory_id for prior in previous for memory_id in prior.retrieved_memory_ids
    }:
        if step.op_name == "expand_temporal_neighbors":
            roles.add("temporal_neighbor")
        elif step.op_name == "retrieve_context_around_image":
            roles.add("image_context")
    return sorted(roles)
