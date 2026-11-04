"""Lossless compiler schema-to-row mappings for Postgres persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.schemas import (
    CompilationStageResult,
    CompilationUnit,
    CompilerRunSpec,
    DoubterFeedback,
    EvidenceType,
    MemoryProgram,
    OperatorEvidence,
    ProgramExecutionEvidenceRef,
    ProgramExecutionStepTrace,
    ProgramExecutionTrace,
    QBankDecision,
    QBankEvidenceRole,
    QBankRoutingItem,
    QuestionCandidate,
)

Row = dict[str, Any]


def compiler_run_to_row(value: CompilerRunSpec) -> Row:
    data = value.model_dump(mode="python")
    data["target_stage"] = value.target_stage.value
    data["metadata"] = dict(value.metadata)
    return data


def compiler_run_from_row(row: Mapping[str, Any]) -> CompilerRunSpec:
    return CompilerRunSpec.model_validate(dict(row))


def compilation_unit_to_rows(value: CompilationUnit) -> dict[str, list[Row]]:
    unit = value.model_dump(mode="python", exclude={"memories"})
    unit["metadata"] = dict(value.metadata)
    memories: list[Row] = []
    for member in value.memories:
        row = member.model_dump(mode="python")
        row.update(
            {
                "compiler_unit_id": value.compiler_unit_id,
                "run_id": value.run_id,
                "role": member.role.value,
                "metadata": dict(member.metadata),
            }
        )
        memories.append(row)
    return {"unit": [unit], "memories": memories}


def compilation_unit_from_rows(
    row: Mapping[str, Any],
    memory_rows: Sequence[Mapping[str, Any]],
) -> CompilationUnit:
    memories = []
    for memory_row in sorted(memory_rows, key=lambda item: int(item["ordinal"])):
        data = dict(memory_row)
        data.pop("compiler_unit_id", None)
        data.pop("run_id", None)
        memories.append(data)
    return CompilationUnit.model_validate({**dict(row), "memories": memories})


def compilation_stage_result_to_row(value: CompilationStageResult) -> Row:
    data = value.model_dump(mode="python")
    data["stage"] = value.stage.value
    data["status"] = value.status.value
    data["metadata"] = dict(value.metadata)
    return data


def compilation_stage_result_from_row(
    row: Mapping[str, Any],
) -> CompilationStageResult:
    return CompilationStageResult.model_validate(dict(row))


def question_to_rows(value: QuestionCandidate) -> dict[str, list[Row]]:
    if value.snapshot_id is None or value.compiler_unit_id is None:
        raise ValueError("persisted qbank questions require compiler scope")
    if value.question_hash is None:
        raise ValueError("persisted qbank questions require question_hash")
    question = value.model_dump(
        mode="python",
        exclude={"paraphrases", "source_memory_ids", "evidence_ids"},
    )
    question["question_type"] = value.question_type.value
    question["answer_type"] = (
        value.answer_type.value if value.answer_type is not None else None
    )
    question["reasoning_type"] = value.reasoning_type.value
    question["candidate_origin"] = value.candidate_origin.value
    question["route_class"] = value.route_class.value
    question["required_modalities"] = [item.value for item in value.required_modalities]
    question["accepted"] = question.pop("accepted")
    question["metadata"] = dict(value.metadata)
    paraphrases = [
        {
            "paraphrase_id": stable_compiler_id(
                "qbank-paraphrase", [value.qbank_id, ordinal, text]
            ),
            "qbank_id": value.qbank_id,
            "run_id": value.run_id,
            "snapshot_id": value.snapshot_id,
            "compiler_unit_id": value.compiler_unit_id,
            "ordinal": ordinal,
            "paraphrase": text,
            "paraphrase_hash": canonical_content_hash(text),
            "created_at": value.created_at,
            "metadata": {},
        }
        for ordinal, text in enumerate(value.paraphrases)
    ]
    source = [
        evidence_ref_to_row(
            run_id=value.run_id,
            snapshot_id=value.snapshot_id,
            compiler_unit_id=value.compiler_unit_id,
            owner_id=value.qbank_id,
            role=QBankEvidenceRole.SOURCE,
            ordinal=ordinal,
            evidence=OperatorEvidence(
                evidence_id=memory_id,
                evidence_type=EvidenceType.MEMORY,
                memory_id=memory_id,
            ),
        )
        for ordinal, memory_id in enumerate(value.source_memory_ids)
    ]
    return {"question": [question], "paraphrases": paraphrases, "source": source}


def question_from_rows(
    row: Mapping[str, Any],
    paraphrase_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> QuestionCandidate:
    data = dict(row)
    data.pop("search_document", None)
    data["paraphrases"] = [
        item["paraphrase"]
        for item in sorted(paraphrase_rows, key=lambda item: int(item["ordinal"]))
    ]
    source_rows = [item for item in evidence_rows if item["role"] == "source"]
    expected_rows = [item for item in evidence_rows if item["role"] == "expected"]
    data["source_memory_ids"] = [
        item["memory_id"]
        for item in sorted(source_rows, key=lambda item: int(item["ordinal"]))
        if item.get("memory_id") is not None
    ]
    data["evidence_ids"] = [
        item["evidence_id"]
        for item in sorted(expected_rows, key=lambda item: int(item["ordinal"]))
    ]
    return QuestionCandidate.model_validate(data)


def evidence_ref_to_row(
    *,
    run_id: str,
    snapshot_id: str,
    compiler_unit_id: str,
    owner_id: str,
    role: QBankEvidenceRole,
    ordinal: int,
    evidence: OperatorEvidence,
) -> Row:
    return {
        "evidence_ref_id": stable_compiler_id(
            "qbank-evidence",
            [owner_id, role.value, ordinal, evidence.evidence_id],
        ),
        "qbank_id": owner_id,
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "compiler_unit_id": compiler_unit_id,
        "role": role.value,
        "evidence_id": evidence.evidence_id,
        "evidence_type": evidence.evidence_type.value,
        "memory_id": evidence.memory_id,
        "image_id": evidence.image_id,
        "session_id": evidence.session_id,
        "caption_id": evidence.caption_id,
        "ordinal": ordinal,
        "metadata": dict(evidence.metadata),
    }


def memory_program_to_rows(value: MemoryProgram) -> dict[str, list[Row]]:
    if value.run_id is None or value.snapshot_id is None or value.compiler_unit_id is None:
        raise ValueError("persisted memory programs require compiler scope")
    if value.operator_set_version is None or value.program_hash is None:
        raise ValueError("persisted memory programs require version and content hash")
    program = value.model_dump(
        mode="python",
        exclude={"operator_sequence", "expected_evidence_ids"},
    )
    program["required_modalities"] = [item.value for item in value.required_modalities]
    program["metadata"] = dict(value.metadata)
    steps = []
    for step in value.operator_sequence:
        steps.append(
            {
                "plan_id": value.plan_id,
                "run_id": value.run_id,
                "snapshot_id": value.snapshot_id,
                "compiler_unit_id": value.compiler_unit_id,
                "step_id": step.step_id,
                "op_name": step.op_name,
                "args": dict(step.args),
                "expected_output_type": step.expected_output_type,
                "metadata": dict(step.metadata),
            }
        )
    return {"program": [program], "steps": steps}


def memory_program_from_rows(
    row: Mapping[str, Any],
    step_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> MemoryProgram:
    data = dict(row)
    data["operator_sequence"] = [
        {
            "step_id": item["step_id"],
            "op_name": item["op_name"],
            "args": item["args"],
            "expected_output_type": item["expected_output_type"],
            "expected_evidence_ids": [
                ref["evidence_id"]
                for ref in evidence_rows
                if ref.get("step_id") == item["step_id"]
            ],
            "metadata": item["metadata"],
        }
        for item in sorted(step_rows, key=lambda value: int(value["step_id"]))
    ]
    data["expected_evidence_ids"] = [
        item["evidence_id"]
        for item in sorted(evidence_rows, key=lambda value: int(value["ordinal"]))
        if item.get("step_id") is None
    ]
    return MemoryProgram.model_validate(data)


def program_expected_evidence_to_row(
    *,
    program: MemoryProgram,
    evidence: OperatorEvidence,
    ordinal: int,
    step_id: int | None = None,
) -> Row:
    assert program.run_id is not None
    assert program.snapshot_id is not None
    assert program.compiler_unit_id is not None
    return {
        "evidence_ref_id": stable_compiler_id(
            "program-expected-evidence",
            [program.plan_id, step_id, ordinal, evidence.evidence_id],
        ),
        "plan_id": program.plan_id,
        "run_id": program.run_id,
        "snapshot_id": program.snapshot_id,
        "compiler_unit_id": program.compiler_unit_id,
        "step_id": step_id,
        "evidence_id": evidence.evidence_id,
        "evidence_type": evidence.evidence_type.value,
        "memory_id": evidence.memory_id,
        "image_id": evidence.image_id,
        "session_id": evidence.session_id,
        "caption_id": evidence.caption_id,
        "ordinal": ordinal,
        "metadata": dict(evidence.metadata),
    }


def execution_to_row(value: ProgramExecutionTrace) -> Row:
    data = value.model_dump(mode="python")
    data["purpose"] = value.purpose.value
    data["status"] = value.status.value
    data["control_action"] = value.control_action.value
    data["metadata"] = dict(value.metadata)
    return data


def execution_from_row(row: Mapping[str, Any]) -> ProgramExecutionTrace:
    return ProgramExecutionTrace.model_validate(dict(row))


def execution_step_to_row(value: ProgramExecutionStepTrace) -> Row:
    if value.execution_id is None or value.snapshot_id is None:
        raise ValueError("persisted execution steps require execution and snapshot IDs")
    if value.input_hash is None or value.output_hash is None:
        raise ValueError("persisted execution steps require input and output hashes")
    return {
        "execution_id": value.execution_id,
        "run_id": value.run_id,
        "snapshot_id": value.snapshot_id,
        "compiler_unit_id": value.compiler_unit_id,
        "query_id": value.query_id,
        "plan_id": value.plan_id,
        "step_id": value.step_id,
        "op_name": value.op_name,
        "input_hash": value.input_hash,
        "output_hash": value.output_hash,
        "output": value.model_dump(mode="python", include={
            "retrieved_memory_ids",
            "retrieved_image_ids",
            "retrieved_caption_ids",
            "retrieved_session_ids",
            "scores",
            "output",
        }),
        "success": value.success,
        "error": value.error,
        "latency_ms": value.latency_ms,
        "status": value.status.value,
        "created_at": value.created_at,
        "metadata": dict(value.metadata),
    }


def execution_step_from_row(row: Mapping[str, Any]) -> ProgramExecutionStepTrace:
    payload = dict(row.get("output") or {})
    return ProgramExecutionStepTrace.model_validate(
        {
            "execution_id": row["execution_id"],
            "run_id": row["run_id"],
            "snapshot_id": row["snapshot_id"],
            "compiler_unit_id": row.get("compiler_unit_id"),
            "query_id": row.get("query_id"),
            "plan_id": row["plan_id"],
            "step_id": row["step_id"],
            "op_name": row["op_name"],
            "input_hash": row["input_hash"],
            "output_hash": row["output_hash"],
            **payload,
            "success": row["success"],
            "error": row.get("error"),
            "latency_ms": row.get("latency_ms"),
            "status": row["status"],
            "created_at": row["created_at"],
            "metadata": row["metadata"],
        }
    )


def execution_evidence_to_row(
    value: ProgramExecutionEvidenceRef,
    *,
    run_id: str,
) -> Row:
    data = value.model_dump(mode="python")
    data["run_id"] = run_id
    data["evidence_type"] = value.evidence_type.value
    data["score"] = value.score.model_dump(mode="python") if value.score else None
    data["metadata"] = dict(value.metadata)
    return data


def execution_evidence_from_row(
    row: Mapping[str, Any],
) -> ProgramExecutionEvidenceRef:
    data = dict(row)
    data.pop("run_id", None)
    return ProgramExecutionEvidenceRef.model_validate(data)


def feedback_to_row(value: DoubterFeedback) -> Row:
    if value.run_id is None or value.snapshot_id is None or value.compiler_unit_id is None:
        raise ValueError("persisted doubter feedback requires compiler scope")
    if value.feedback_hash is None:
        raise ValueError("persisted doubter feedback requires feedback_hash")
    data = value.model_dump(mode="python")
    data["evidence_sufficiency"] = value.evidence_sufficiency.value
    data["missing_modalities"] = [item.value for item in value.missing_modalities]
    data["metadata"] = dict(value.metadata)
    return data


def feedback_from_row(row: Mapping[str, Any]) -> DoubterFeedback:
    return DoubterFeedback.model_validate(dict(row))


def decision_to_row(value: QBankDecision) -> Row:
    data = value.model_dump(mode="python")
    data["metadata"] = dict(value.metadata)
    return data


def decision_from_row(row: Mapping[str, Any]) -> QBankDecision:
    return QBankDecision.model_validate(dict(row))


def routing_item_from_row(row: Mapping[str, Any]) -> QBankRoutingItem:
    data = dict(row)
    data.pop("run_id", None)
    return QBankRoutingItem.model_validate(data)
