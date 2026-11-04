"""Deterministic identities for compiler facts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from pmmc.schemas.compiler import (
    CompilationUnitMemory,
    CompilationUnitMemoryRole,
    QBankDecision,
)
from pmmc.schemas.feedback import DoubterFeedback
from pmmc.schemas.program import MemoryProgram
from pmmc.schemas.qbank import QuestionCandidate
from pmmc.schemas.questioner import CompilationUnitRelationType

COMPILER_UNIT_RUN_SCOPED_IDENTITY_V1 = "compiler-unit-run-scoped-v1"


def canonical_content_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_compiler_id(prefix: str, value: Any, *, length: int = 24) -> str:
    if not prefix or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in prefix
    ):
        raise ValueError("stable ID prefix must be lowercase ASCII")
    return f"{prefix}-{canonical_content_hash(value)[:length]}"


def questioner_model_call_id(
    *,
    run_id: str,
    compiler_unit_id: str,
    attempt_number: int,
) -> str:
    """Return the request identity used for one Questioner model attempt."""

    if attempt_number <= 0:
        raise ValueError("Questioner attempt number must be positive")
    return stable_compiler_id(
        "questioner-call",
        [run_id, compiler_unit_id, attempt_number],
    )


def questioner_verifier_model_call_id(
    *,
    run_id: str,
    compiler_unit_id: str,
    generation_id: str,
    verification_round: int,
    attempt_number: int,
    candidate_scope: str | None = None,
) -> str:
    if verification_round not in {0, 1}:
        raise ValueError("Questioner verification round must be 0 or 1")
    if attempt_number <= 0:
        raise ValueError("Questioner verifier attempt number must be positive")
    identity: list[object] = [
        run_id,
        compiler_unit_id,
        generation_id,
        verification_round,
        attempt_number,
    ]
    if candidate_scope is not None:
        identity.append(candidate_scope)
    return stable_compiler_id(
        "questioner-verifier-call",
        identity,
    )


def questioner_revision_model_call_id(
    *,
    run_id: str,
    compiler_unit_id: str,
    generation_id: str,
    candidate_scope: str,
    attempt_number: int,
) -> str:
    if not candidate_scope or attempt_number <= 0:
        raise ValueError("Questioner revision requires scope and a positive attempt")
    return stable_compiler_id(
        "questioner-revision-call",
        [
            run_id,
            compiler_unit_id,
            generation_id,
            candidate_scope,
            attempt_number,
        ],
    )


def compilation_unit_selection_lineage(
    memories: Sequence[CompilationUnitMemory],
) -> list[dict[str, object]]:
    """Rebuild the canonical v3 support-selection lineage from frozen members."""

    lineage: list[dict[str, object]] = []
    for item in memories:
        if item.role is not CompilationUnitMemoryRole.SUPPORT:
            continue
        metadata = dict(item.metadata)
        relation_value = metadata.pop("relation_type", None)
        if not isinstance(relation_value, str):
            raise ValueError("v3 support memory requires a relation_type")
        try:
            relation_type = CompilationUnitRelationType(relation_value)
        except ValueError as error:
            raise ValueError(
                "v3 support memory contains an unsupported relation_type"
            ) from error
        lineage.append(
            {
                "memory_id": item.memory_id,
                "relation_type": relation_type.value,
                "metadata": metadata,
            }
        )
    return lineage


def compilation_unit_identity(
    *,
    snapshot_id: str,
    conversation_id: str,
    cutoff_session_id: str,
    builder_version: str,
    memories: Sequence[CompilationUnitMemory],
    selection_lineage: Sequence[dict[str, object]] | None = None,
    run_id: str | None = None,
) -> tuple[str, str]:
    payload = {
        "snapshot_id": snapshot_id,
        "conversation_id": conversation_id,
        "cutoff_session_id": cutoff_session_id,
        "builder_version": builder_version,
        "memories": [
            {
                "memory_id": item.memory_id,
                "ordinal": item.ordinal,
                "role": item.role.value,
            }
            for item in memories
        ],
        **(
            {"selection_lineage": list(selection_lineage)}
            if selection_lineage is not None
            else {}
        ),
    }
    digest = canonical_content_hash(payload)
    unit_id = (
        compilation_unit_instance_id(run_id=run_id, input_hash=digest)
        if run_id is not None
        else compilation_unit_content_id(digest)
    )
    return unit_id, digest


def compilation_unit_content_id(input_hash: str) -> str:
    """Return the stable cross-run identity of one compilation-unit input."""

    if not input_hash:
        raise ValueError("compilation unit input hash cannot be empty")
    return f"compiler-unit-{input_hash[:24]}"


def compilation_unit_instance_id(*, run_id: str, input_hash: str) -> str:
    """Return the persistence identity for one content unit inside one run."""

    if not run_id or not input_hash:
        raise ValueError("run-scoped compilation unit identity requires run and input hash")
    return stable_compiler_id(
        "compiler-unit",
        {
            "identity_protocol": COMPILER_UNIT_RUN_SCOPED_IDENTITY_V1,
            "input_hash": input_hash,
            "run_id": run_id,
        },
    )


def question_content_hash(value: QuestionCandidate) -> str:
    excluded = {
        "accepted",
        "rejection_reason",
        "created_at",
        "metadata",
        "question_hash",
        "candidate_content_hash",
    }
    # Compiler 0.1/0.2 rows predate versioned candidate lineage. Keeping the
    # implicit generated/v0 values out of their hash preserves old manifests,
    # while every v3 candidate is bound through a non-null opportunity ID.
    if value.opportunity_id is None:
        excluded.update(
            {
                "opportunity_id",
                "candidate_origin",
                "candidate_version",
                "supersedes_qbank_id",
            }
        )
    return canonical_content_hash(
        value.model_dump(
            mode="json",
            exclude=excluded,
        )
    )


def memory_program_content_hash(value: MemoryProgram) -> str:
    return canonical_content_hash(
        value.model_dump(
            mode="json",
            exclude={"accepted", "created_at", "metadata", "program_hash"},
        )
    )


def feedback_content_hash(value: DoubterFeedback) -> str:
    return canonical_content_hash(
        value.model_dump(
            mode="json",
            exclude={"created_at", "metadata", "feedback_hash"},
        )
    )


def qbank_decision_content_hash(value: QBankDecision) -> str:
    return canonical_content_hash(
        value.model_dump(
            mode="json",
            exclude={"created_at", "metadata", "decision_hash"},
        )
    )


def ordered_bindings_hash(bindings: Mapping[str, str]) -> str:
    return canonical_content_hash(dict(sorted(bindings.items())))
