"""Local-alias relationship requirements deferred to gold-free answer grounding."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pmmc.compiler.ids import canonical_content_hash
from pmmc.schemas import AnswerGenerationInput, GenerationRequest
from pmmc.schemas.answer_output import AnswerContractV3, TypedAnswerContext


def grounding_relation_requirement(
    prepared: AnswerGenerationInput,
    output: AnswerContractV3,
    context: TypedAnswerContext,
) -> dict[str, Any] | None:
    return grounding_relation_from_metadata(prepared.metadata, output, context)


def grounding_relation_from_metadata(
    metadata: Mapping[str, Any],
    output: AnswerContractV3,
    context: TypedAnswerContext,
) -> dict[str, Any] | None:
    """Validate the same relation scope for live requests and immutable audit replay."""
    contract = metadata.get("runtime_relation_contract")
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise ValueError("invalid_runtime_relation_contract")
    payload = {key: value for key, value in contract.items() if key != "content_hash"}
    if (
        set(payload)
        != {
            "protocol",
            "required_semantic_roles",
            "minimum_evidence_groups",
            "semantic_relation_status",
            "sources",
        }
        or contract.get("content_hash") != canonical_content_hash(payload)
        or contract["protocol"] != "runtime-independent-relation-grounding-v1"
        or contract["required_semantic_roles"] != ["independent_group_relation"]
        or contract["semantic_relation_status"] != "not_evaluated"
        or type(contract["minimum_evidence_groups"]) is not int
        or contract["minimum_evidence_groups"] < 2
    ):
        raise ValueError("invalid_runtime_relation_contract")
    sources = contract["sources"]
    if (
        not isinstance(sources, list)
        or any(
            not isinstance(source, dict) or set(source) != {"evidence_alias", "group_aliases"}
            for source in sources
        )
        or [source["evidence_alias"] for source in sources]
        != [source.evidence_alias for source in context.sources]
    ):
        raise ValueError("runtime_relation_alias_scope_mismatch")
    for source in sources:
        groups = source["group_aliases"]
        if (
            not isinstance(groups, list)
            or not groups
            or any(
                not isinstance(group, str) or not re.fullmatch(r"G[0-9]{2,}", group)
                for group in groups
            )
            or len(groups) != len(set(groups))
        ):
            raise ValueError("runtime_relation_group_alias_invalid")
    count = len({group for source in sources for group in source["group_aliases"]})
    return {
        "requirement": "independent_group_relation",
        "minimum_evidence_groups": contract["minimum_evidence_groups"],
        "supplied_independent_group_count": count,
        "supplied_sources": sources,
        "claimed_evidence_aliases": output.used_evidence_aliases,
        "instruction": (
            "Assess the relationship required by the real query using only the supplied "
            "source contents. source_support and query_closure may pass only when the "
            "answer is jointly established across the required independent groups. "
            "Different group labels, group counts, or retrieval operations alone do not "
            "establish a semantic relationship. Missing required groups must fail "
            "source_support. Do not infer support from unprovided sources."
        ),
        "contract_hash": contract["content_hash"],
    }


def grounding_scope_aliases(
    prepared: AnswerGenerationInput,
    output: AnswerContractV3,
    context: TypedAnswerContext,
) -> list[str]:
    if grounding_relation_requirement(prepared, output, context) is not None:
        return [source.evidence_alias for source in context.sources]
    return list(output.used_evidence_aliases)


def grounding_request_scope_aliases(
    request: GenerationRequest,
    output: AnswerContractV3,
    context: TypedAnswerContext,
    *,
    expected_query_hash: str | None = None,
) -> list[str]:
    """Recover the validated evidence scope from the actual saved grounding request."""
    from pmmc.runtime.grounding import source_content_parts, strict_json_object

    try:
        message = request.messages[1]
        if message.role != "user" or message.parts[0].kind != "text":
            raise ValueError("runtime_relation_request_invalid")
        proposal = strict_json_object(message.parts[0].text or "")
        if proposal.get("proposed_answer") != output.model_dump(mode="json"):
            raise ValueError("grounding_proposed_answer_mismatch")
        relation = proposal.get("required_evidence_relation")
        if relation is None:
            if request.metadata.get("runtime_relation_contract_hash") is not None:
                raise ValueError("runtime_relation_request_contract_missing")
            aliases = list(output.used_evidence_aliases)
        else:
            contract = {
                "protocol": "runtime-independent-relation-grounding-v1",
                "required_semantic_roles": ["independent_group_relation"],
                "minimum_evidence_groups": relation["minimum_evidence_groups"],
                "semantic_relation_status": "not_evaluated",
                "sources": relation["supplied_sources"],
                "content_hash": relation["contract_hash"],
            }
            expected = grounding_relation_from_metadata(
                {"runtime_relation_contract": contract}, output, context
            )
            if (
                relation != expected
                or request.metadata.get("runtime_relation_contract_hash")
                != contract["content_hash"]
            ):
                raise ValueError("runtime_relation_request_scope_mismatch")
            aliases = [source.evidence_alias for source in context.sources]
        sources = source_content_parts(context, aliases)
        source_start = len(message.parts) - len(sources)
        if source_start < 1 or message.parts[source_start:] != sources:
            raise ValueError("grounding_source_payload_mismatch")
        query_parts = message.parts[1:source_start]
        if any(part.kind != "image" or part.image is None for part in query_parts):
            raise ValueError("grounding_query_payload_invalid")
        query = proposal.get("query")
        if not isinstance(query, str):
            raise ValueError("grounding_query_payload_invalid")
        # Grounding retains query images even when the answer adapter deduplicates them.
        query_hash = canonical_content_hash(
            [
                query,
                [
                    part.image.model_dump(mode="json", exclude={"metadata", "caption"})
                    for part in query_parts
                    if part.image is not None
                ],
            ]
        )
        if expected_query_hash is not None and query_hash != expected_query_hash:
            raise ValueError("grounding_query_hash_mismatch")
        return aliases
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("runtime_relation_request_invalid") from exc
