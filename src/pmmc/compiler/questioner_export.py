"""Offline semantic verification for frozen Questioner runtime exports."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import orjson

from pmmc.infrastructure.export_verification import (
    MODEL_ARTIFACT_BUNDLE_VERSION,
    QUESTIONER_RUNTIME_EXPORT_VERSION,
    QUESTIONER_V3_RUNTIME_EXPORT_VERSION,
    verify_experiment_export,
)
from pmmc.infrastructure.object_store import validate_object_key
from pmmc.schemas.questioner import QUESTIONER_STATIC_REJECTION_CODES

_REQUIRED_RUNTIME_FILES = (
    "compiler_unit_links.jsonl",
    "questioner_generations.jsonl",
    "questioner_candidate_gate_results.jsonl",
    "qbank_duplicate_links.jsonl",
    "qbank_index_manifests.jsonl",
)
_REQUIRED_V3_RUNTIME_FILES = (
    "questioner_opportunity_sets.jsonl",
    "questioner_opportunities.jsonl",
    "questioner_opportunity_evidence_refs.jsonl",
    "questioner_verification_batches.jsonl",
    "questioner_candidate_verifications.jsonl",
    "questioner_candidate_supersessions.jsonl",
)
_GOLD_SENTINEL_MARKERS = (
    b"GOLD_ANSWER_SENTINEL",
    b"PMMC_GOLD_SENTINEL",
)
_V3_STATIC_REJECTION_CODES = QUESTIONER_STATIC_REJECTION_CODES
_V3_POST_VERIFIER_REJECTION_CODES = {
    "duplicate",
    "budget_exceeded",
    "semantic_revision_required",
    "semantic_verification_failed",
    "revision_exhausted",
}
_GENERATION_TERMINAL_REASON = "generation_structured_output_invalid"
_GENERATION_REPAIR_TERMINAL_REASON = "generation_local_json_repair_not_freezable"

_INTERNAL_OR_METADATA_PATTERN = re.compile(
    r"\b(?:compiler[-_ ]?unit|qbank|memlens[:_-]|memgallery[:_-]|"
    r"memory[_:-]|session[_:-]|turn[_:-]|round[_:-]|[MISCOQ]\d{2}|"
    r"session order|current (?:date|time)|cutoff session|memory id|system metadata|"
    r"local aliases?|bounded opportunities?|candidate aliases?|evidence aliases?|"
    r"harness|compiler state|inability to generate)\b",
    flags=re.IGNORECASE,
)
_PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:to be determined|tbd|not (?:yet|specified|provided|known)|"
    r"will (?:provide|share|decide)|plans? to|placeholder|unknown(?: for now)?|"
    r"unsure|maybe|probably|it depends)\b",
    flags=re.IGNORECASE,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Questioner export file must contain an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = orjson.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Questioner export JSONL row is not an object: {path.name}:{line_number}"
                )
            rows.append(value)
    return rows


def _require_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def _require_row_scope(
    rows: list[dict[str, Any]],
    *,
    label: str,
    run_id: str | None = None,
    run_field: str = "run_id",
    snapshot_id: str | None = None,
) -> None:
    """Require every exported row to carry the frozen runtime scope."""

    for row in rows:
        if run_id is not None and str(row.get(run_field, "")) != run_id:
            raise ValueError(f"Questioner {label} crosses compiler-run scope")
        if snapshot_id is not None and str(row.get("snapshot_id", "")) != snapshot_id:
            raise ValueError(f"Questioner {label} crosses snapshot scope")


def _is_pre_verifier_terminal_rejection(gate: dict[str, Any]) -> bool:
    """Identify candidates rejected before a semantic verifier call was required."""

    if gate.get("status") != "rejected":
        return False
    rejection_codes = {str(value) for value in gate.get("rejection_codes", [])}
    if not rejection_codes or rejection_codes & _V3_POST_VERIFIER_REJECTION_CODES:
        return False
    metadata = gate.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if "invalid_schema" in rejection_codes:
        return (
            metadata.get("native_provider_output") is False
            and metadata.get("terminal_reason") == _GENERATION_REPAIR_TERMINAL_REASON
        )
    return bool(rejection_codes & _V3_STATIC_REJECTION_CODES)


def _is_terminal_generation_rejection_call(
    generation: dict[str, Any],
    call: dict[str, Any],
) -> bool:
    """Mirror the database contract for a failed generation call closed as rejection."""

    metadata = generation.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return (
        call.get("status") == "failed"
        and isinstance(call.get("output_asset_id"), str)
        and call.get("parse_error") is not None
        and generation.get("parse_status") == "failed"
        and int(generation.get("raw_candidate_count", -1)) == 0
        and int(generation.get("valid_candidate_count", -1)) == 0
        and int(generation.get("invalid_candidate_count", -1)) == 0
        and int(generation.get("repair_attempt_count", -1)) == 0
        and isinstance(generation.get("output_hash"), str)
        and metadata.get("native_structured_output") is False
        and metadata.get("response_artifact_ready") is True
        and metadata.get("terminal_reason") == _GENERATION_TERMINAL_REASON
    )


def _is_terminal_verifier_rejection_call(
    batch: dict[str, Any],
    values: list[dict[str, Any]],
    call: dict[str, Any],
) -> bool:
    """Mirror the single-candidate failed-verifier conservative rejection contract."""

    metadata = batch.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return (
        call.get("status") == "failed"
        and isinstance(call.get("output_asset_id"), str)
        and call.get("parse_error") is not None
        and isinstance(batch.get("scope_qbank_id"), str)
        and int(batch.get("input_count", -1)) == 1
        and int(batch.get("pass_count", -1)) == 0
        and int(batch.get("revise_count", -1)) == 0
        and int(batch.get("reject_count", -1)) == 1
        and metadata.get("provider_output_fallback") is True
        and metadata.get("single_candidate_scope") is True
        and len(values) == 1
        and values[0].get("qbank_id") == batch.get("scope_qbank_id")
        and values[0].get("verdict") == "reject"
        and isinstance(values[0].get("metadata"), dict)
        and values[0]["metadata"].get("provider_output_fallback") is True
    )


def _verify_v3_lineage(
    *,
    unit_ids: set[str],
    unit_links: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    opportunity_sets: list[dict[str, Any]],
    opportunities: list[dict[str, Any]],
    opportunity_evidence: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    verification_batches: list[dict[str, Any]],
    candidate_verifications: list[dict[str, Any]],
    supersessions: list[dict[str, Any]],
    isolated_candidate_scope: bool = False,
) -> dict[str, int]:
    _require_equal(
        {str(row["compiler_unit_id"]) for row in opportunity_sets},
        unit_ids,
        "Questioner v3 opportunity-set denominator is incomplete",
    )
    if len(opportunity_sets) != len(unit_ids):
        raise ValueError("Questioner v3 requires exactly one opportunity set per unit")
    set_by_id = {str(row["opportunity_set_id"]): row for row in opportunity_sets}
    if len(set_by_id) != len(opportunity_sets):
        raise ValueError("Questioner v3 opportunity-set IDs are duplicated")
    opportunities_by_set: dict[str, list[dict[str, Any]]] = {}
    opportunity_by_id: dict[str, dict[str, Any]] = {}
    for row in opportunities:
        set_id = str(row["opportunity_set_id"])
        if set_id not in set_by_id:
            raise ValueError("Questioner opportunity references a missing set")
        opportunity_id = str(row["opportunity_id"])
        if opportunity_id in opportunity_by_id:
            raise ValueError("Questioner opportunity IDs are duplicated")
        opportunity_by_id[opportunity_id] = row
        opportunities_by_set.setdefault(set_id, []).append(row)
    link_ids_by_unit: dict[str, set[str]] = {}
    for row in unit_links:
        link_ids_by_unit.setdefault(str(row["compiler_unit_id"]), set()).add(str(row["link_id"]))
    for row in opportunities:
        unit_id = str(row["compiler_unit_id"])
        relation_ids = {str(value) for value in row.get("relation_link_ids", [])}
        if not relation_ids <= link_ids_by_unit.get(unit_id, set()):
            raise ValueError("Questioner opportunity relation crosses unit scope")
    for set_id, row in set_by_id.items():
        values = sorted(opportunities_by_set.get(set_id, []), key=lambda item: int(item["ordinal"]))
        if [int(item["ordinal"]) for item in values] != list(range(len(values))):
            raise ValueError("Questioner opportunity ordinals are not contiguous")
        _require_equal(
            len(values),
            int(row["opportunity_count"]),
            "Questioner opportunity count differs from its set manifest",
        )
    ref_keys: set[tuple[str, str]] = set()
    for row in opportunity_evidence:
        opportunity_id = str(row["opportunity_id"])
        if opportunity_id not in opportunity_by_id:
            raise ValueError("Questioner opportunity evidence has no parent")
        key = (opportunity_id, str(row["alias"]))
        if key in ref_keys:
            raise ValueError("Questioner opportunity evidence alias is duplicated")
        evidence_type = str(row.get("evidence_type", ""))
        typed_references = [
            row.get("memory_id"),
            row.get("image_id"),
            row.get("session_id"),
            row.get("caption_id"),
        ]
        if sum(value is not None for value in typed_references) != 1:
            raise ValueError("Questioner opportunity evidence must contain one typed reference")
        expected_id = {
            "memory": row.get("memory_id"),
            "caption": row.get("memory_id"),
            "image": row.get("image_id"),
            "session": row.get("session_id"),
        }.get(evidence_type)
        if expected_id is None or str(row.get("evidence_id", "")) != str(expected_id):
            raise ValueError("Questioner opportunity evidence typed lineage is invalid")
        if evidence_type == "image":
            metadata = row.get("metadata")
            evidence_metadata = (
                metadata.get("evidence_metadata") if isinstance(metadata, dict) else None
            )
            if not isinstance(evidence_metadata, dict) or not isinstance(
                evidence_metadata.get("owning_memory_id"), str
            ):
                raise ValueError("Questioner image opportunity lacks owning-memory audit lineage")
        ref_keys.add(key)
    for opportunity_id, row in opportunity_by_id.items():
        expected_aliases = {str(value) for value in row.get("evidence_aliases", [])}
        actual_aliases = {alias for parent, alias in ref_keys if parent == opportunity_id}
        _require_equal(
            actual_aliases,
            expected_aliases,
            "Questioner opportunity evidence aliases are incomplete",
        )

    generation_by_unit = {
        str(row["compiler_unit_id"]): str(row["generation_id"]) for row in generations
    }
    question_by_id = {str(row["qbank_id"]): row for row in questions}
    if len(question_by_id) != len(questions):
        raise ValueError("Questioner v3 QBank IDs are duplicated")
    for row in questions:
        opportunity = opportunity_by_id.get(str(row.get("opportunity_id", "")))
        if opportunity is None:
            raise ValueError("Questioner v3 candidate has no opportunity")
        if str(opportunity["compiler_unit_id"]) != str(row["compiler_unit_id"]):
            raise ValueError("Questioner candidate crosses opportunity unit scope")
        origin = row.get("candidate_origin")
        version = int(row.get("candidate_version", -1))
        supersedes = row.get("supersedes_qbank_id")
        if (origin, version, supersedes is None) not in {
            ("generated", 0, True),
            ("revised", 1, False),
        }:
            raise ValueError("Questioner candidate version lineage is invalid")
        if str(row.get("generation_id", "")) != generation_by_unit.get(
            str(row["compiler_unit_id"])
        ):
            raise ValueError("Questioner candidate crosses generation scope")

    gate_by_qbank = {str(row["qbank_id"]): row for row in gates}
    _require_equal(
        set(gate_by_qbank),
        set(question_by_id),
        "Questioner v3 candidates do not all have terminal gates",
    )
    batch_by_id = {str(row["verification_batch_id"]): row for row in verification_batches}
    verification_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    items_by_batch: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_verifications:
        batch_id = str(row["verification_batch_id"])
        if batch_id not in batch_by_id:
            raise ValueError("Questioner candidate verification has no batch")
        qbank_id = str(row["qbank_id"])
        if qbank_id not in question_by_id:
            raise ValueError("Questioner verification references a missing candidate")
        verification_key = (qbank_id, int(row["verification_round"]))
        if verification_key in verification_by_key:
            raise ValueError("Questioner candidate has duplicate verifier outcomes")
        verification_by_key[verification_key] = row
        items_by_batch.setdefault(batch_id, []).append(row)
    for batch_id, batch in batch_by_id.items():
        values = items_by_batch.get(batch_id, [])
        counts = {
            verdict: sum(row["verdict"] == verdict for row in values)
            for verdict in ("pass", "revise", "reject")
        }
        if (
            len(values) != int(batch["input_count"])
            or counts["pass"] != int(batch["pass_count"])
            or counts["revise"] != int(batch["revise_count"])
            or counts["reject"] != int(batch["reject_count"])
        ):
            raise ValueError("Questioner verification-batch accounting is incomplete")
        if isolated_candidate_scope and (
            int(batch["input_count"]) != 1
            or not isinstance(batch.get("scope_qbank_id"), str)
            or len(values) != 1
            or values[0].get("qbank_id") != batch.get("scope_qbank_id")
        ):
            raise ValueError(
                "Questioner v3.2 verifier batch escapes single-candidate scope"
            )

    supersession_by_source: dict[str, dict[str, Any]] = {}
    supersession_by_replacement: dict[str, dict[str, Any]] = {}
    for row in supersessions:
        source = str(row["source_qbank_id"])
        replacement = str(row["replacement_qbank_id"])
        if source in supersession_by_source or replacement in supersession_by_replacement:
            raise ValueError("Questioner supersession is not one-to-one")
        source_row = question_by_id.get(source)
        replacement_row = question_by_id.get(replacement)
        if source_row is None or replacement_row is None:
            raise ValueError("Questioner supersession references a missing candidate")
        if (
            replacement_row.get("supersedes_qbank_id") != source
            or int(source_row["generation_rank"]) != int(replacement_row["generation_rank"])
            or source_row["compiler_unit_id"] != replacement_row["compiler_unit_id"]
        ):
            raise ValueError("Questioner supersession lineage is inconsistent")
        initial_verification = verification_by_key.get((source, 0))
        if (
            initial_verification is None
            or initial_verification.get("verdict") != "revise"
            or initial_verification.get("replacement_qbank_id") != replacement
        ):
            raise ValueError("Questioner supersession has no revise verifier outcome")
        supersession_by_source[source] = row
        supersession_by_replacement[replacement] = row
    revised_ids = {
        qbank_id
        for qbank_id, row in question_by_id.items()
        if row.get("candidate_origin") == "revised"
    }
    _require_equal(
        set(supersession_by_replacement),
        revised_ids,
        "Questioner revised candidates are missing supersession lineage",
    )
    for qbank_id, gate in gate_by_qbank.items():
        if gate.get("status") != "eligible_for_planning":
            continue
        version = int(question_by_id[qbank_id].get("candidate_version", 0))
        verification = verification_by_key.get((qbank_id, version))
        if verification is None or verification.get("verdict") != "pass":
            raise ValueError("eligible Questioner v3 candidate lacks verifier pass")

    generated_candidate_count = 0
    initial_verification_count = 0
    pre_verifier_rejection_count = 0
    missing_initial_verification_count = 0
    for qbank_id, row in question_by_id.items():
        if int(row.get("candidate_version", 0)) != 0:
            continue
        generated_candidate_count += 1
        initial = verification_by_key.get((qbank_id, 0))
        gate = gate_by_qbank[qbank_id]
        if initial is not None:
            initial_verification_count += 1
        elif _is_pre_verifier_terminal_rejection(gate):
            pre_verifier_rejection_count += 1
        else:
            missing_initial_verification_count += 1

    lineage_accounting = {
        "generated_candidate_count": generated_candidate_count,
        "initial_verification_count": initial_verification_count,
        "pre_verifier_rejection_count": pre_verifier_rejection_count,
        "missing_initial_verification_count": missing_initial_verification_count,
    }
    print(
        "QUESTIONER_EXPORT_VERIFICATION_ACCOUNTING "
        + orjson.dumps(lineage_accounting, option=orjson.OPT_SORT_KEYS).decode("utf-8"),
        flush=True,
    )
    if missing_initial_verification_count:
        raise ValueError("Questioner v3 generated candidate lacks initial verification")

    questions_by_rank: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in questions:
        questions_by_rank.setdefault(
            (str(row["compiler_unit_id"]), int(row["generation_rank"])), []
        ).append(row)
    for values in questions_by_rank.values():
        generated = [row for row in values if int(row["candidate_version"]) == 0]
        revised = [row for row in values if int(row["candidate_version"]) == 1]
        if len(generated) != 1 or len(revised) > 1:
            raise ValueError("Questioner rank has invalid candidate versions")
        if revised and str(generated[0]["qbank_id"]) not in supersession_by_source:
            raise ValueError("Questioner revised rank has no supersession")
        if (
            sum(
                gate_by_qbank[str(row["qbank_id"])].get("status") == "eligible_for_planning"
                for row in values
            )
            > 1
        ):
            raise ValueError("Questioner rank has multiple terminal eligible candidates")

    generation_by_unit_row = {str(row["compiler_unit_id"]): row for row in generations}
    gates_by_unit: dict[str, list[dict[str, Any]]] = {}
    for row in gates:
        gates_by_unit.setdefault(str(row["compiler_unit_id"]), []).append(row)
    for stage in stages:
        unit_id = str(stage["compiler_unit_id"])
        generation = generation_by_unit_row[unit_id]
        eligible = sum(
            row.get("status") == "eligible_for_planning" for row in gates_by_unit.get(unit_id, [])
        )
        expected = (
            int(generation["raw_candidate_count"]),
            eligible,
            int(generation["raw_candidate_count"]) - eligible,
        )
        actual = (
            int(stage["produced_count"]),
            int(stage["accepted_count"]),
            int(stage["rejected_count"]),
        )
        _require_equal(
            actual,
            expected,
            "Questioner v3 stage accounting differs from terminal rank facts",
        )
    return lineage_accounting


def verify_questioner_export(export_dir: Path) -> dict[str, Any]:
    """Verify checksums, Questioner accounting, indexes, and private model payloads."""

    manifest = verify_experiment_export(export_dir)
    export_protocol = manifest.get("questioner_runtime_export_version")
    program_export = manifest.get("program_runtime_export_version") is not None
    if export_protocol not in {
        QUESTIONER_RUNTIME_EXPORT_VERSION,
        QUESTIONER_V3_RUNTIME_EXPORT_VERSION,
    }:
        raise ValueError("export does not use a supported Questioner runtime protocol")
    if manifest.get("model_artifact_bundle_version") != MODEL_ARTIFACT_BUNDLE_VERSION:
        raise ValueError("Questioner export has no referenced model artifact bundle")
    if any(str(item["path"]).startswith("evaluation/") for item in manifest["files"]):
        raise ValueError("Questioner runtime export contains evaluation-scoped files")
    if any("gold" in Path(str(item["path"])).name.casefold() for item in manifest["files"]):
        raise ValueError("Questioner runtime export contains a gold-named file")
    for item in manifest["files"]:
        data = (export_dir / str(item["path"])).read_bytes()
        if any(marker in data for marker in _GOLD_SENTINEL_MARKERS):
            raise ValueError("Questioner runtime export contains a gold sentinel")

    runtime = export_dir / "runtime"
    for file_name in _REQUIRED_RUNTIME_FILES:
        if not (runtime / file_name).is_file():
            raise ValueError(f"Questioner runtime export is missing {file_name}")

    run_manifest = _read_json(runtime / "run_manifest.json")
    questioner_config = dict(run_manifest.get("metadata", {})).get("questioner_config", {})
    if not isinstance(questioner_config, dict):
        raise ValueError("Questioner run manifest has no frozen Questioner config")
    protocol_version = str(questioner_config.get("protocol_version", ""))
    v2_protocol = protocol_version in {
        "pmmc-questioner-v2",
        "pmmc-questioner-v2.1",
    }
    v3_protocol = protocol_version in {
        "pmmc-questioner-v3",
        "pmmc-questioner-v3.1",
        "pmmc-questioner-v3.2",
        "pmmc-questioner-v3.3",
        "pmmc-questioner-v3.4",
        "pmmc-questioner-v3.5",
        "pmmc-questioner-v3.6",
    }
    if protocol_version != "pmmc-questioner-v1" and not v2_protocol and not v3_protocol:
        raise ValueError("Questioner run manifest has an unsupported protocol")
    expected_export_protocol = (
        QUESTIONER_V3_RUNTIME_EXPORT_VERSION if v3_protocol else QUESTIONER_RUNTIME_EXPORT_VERSION
    )
    if export_protocol != expected_export_protocol:
        raise ValueError("Questioner export protocol does not match its producer version")
    if v3_protocol:
        for file_name in _REQUIRED_V3_RUNTIME_FILES:
            if not (runtime / file_name).is_file():
                raise ValueError(f"Questioner v3 export is missing {file_name}")
    from pmmc.compiler.questioner_prompt import (
        questioner_prompt_sha256,
        questioner_response_schema_sha256,
    )

    prompt_version = str(questioner_config.get("prompt_version", ""))
    expected_schema_hash = questioner_response_schema_sha256(prompt_version)
    declared_schema_hash = dict(run_manifest.get("metadata", {})).get(
        "questioner_response_schema_hash"
    )
    if (declared_schema_hash is not None and declared_schema_hash != expected_schema_hash) or (
        (v2_protocol or v3_protocol) and declared_schema_hash is None
    ):
        raise ValueError("Questioner run manifest has the wrong response schema hash")
    prompt_hashes = dict(run_manifest.get("prompt_hashes", {}))
    declared_prompt_hash = prompt_hashes.get(prompt_version)
    if (
        declared_prompt_hash is not None
        and declared_prompt_hash != questioner_prompt_sha256(prompt_version)
    ) or (v3_protocol and declared_prompt_hash is None):
        raise ValueError("Questioner run manifest has the wrong generation prompt hash")
    if v3_protocol:
        verifier_prompt_version = str(questioner_config.get("verifier_prompt_version", ""))
        expected_verifier_prompt_hash = questioner_prompt_sha256(verifier_prompt_version)
        expected_verifier_schema_hash = questioner_response_schema_sha256(verifier_prompt_version)
        metadata = dict(run_manifest.get("metadata", {}))
        if (
            prompt_hashes.get(verifier_prompt_version) != expected_verifier_prompt_hash
            or metadata.get("questioner_verifier_prompt_hash") != expected_verifier_prompt_hash
            or metadata.get("questioner_verifier_schema_hash") != expected_verifier_schema_hash
        ):
            raise ValueError("Questioner run manifest has the wrong verifier identity")

    compiler_runs = _read_jsonl(runtime / "compiler_runs.jsonl")
    units = _read_jsonl(runtime / "compiler_units.jsonl")
    unit_memories = _read_jsonl(runtime / "compiler_unit_memories.jsonl")
    unit_links = _read_jsonl(runtime / "compiler_unit_links.jsonl")
    generations = _read_jsonl(runtime / "questioner_generations.jsonl")
    jobs = [
        row
        for row in _read_jsonl(runtime / "experiment_jobs.jsonl")
        if row.get("stage") == "compile"
    ]
    stages = [
        row
        for row in _read_jsonl(runtime / "compiler_unit_stage_results.jsonl")
        if row.get("stage") == "questioner"
    ]
    questions = _read_jsonl(runtime / "qbank_questions.jsonl")
    paraphrases = _read_jsonl(runtime / "qbank_paraphrases.jsonl")
    evidence_refs = _read_jsonl(runtime / "qbank_evidence_refs.jsonl")
    gates = _read_jsonl(runtime / "questioner_candidate_gate_results.jsonl")
    duplicate_links = _read_jsonl(runtime / "qbank_duplicate_links.jsonl")
    indexes = _read_jsonl(runtime / "qbank_index_manifests.jsonl")
    collections = _read_jsonl(runtime / "embedding_collections.jsonl")
    embeddings = _read_jsonl(runtime / "embeddings_metadata.jsonl")
    opportunity_sets = (
        _read_jsonl(runtime / "questioner_opportunity_sets.jsonl") if v3_protocol else []
    )
    opportunities = _read_jsonl(runtime / "questioner_opportunities.jsonl") if v3_protocol else []
    opportunity_evidence = (
        _read_jsonl(runtime / "questioner_opportunity_evidence_refs.jsonl") if v3_protocol else []
    )
    verification_batches = (
        _read_jsonl(runtime / "questioner_verification_batches.jsonl") if v3_protocol else []
    )
    candidate_verifications = (
        _read_jsonl(runtime / "questioner_candidate_verifications.jsonl") if v3_protocol else []
    )
    supersessions = (
        _read_jsonl(runtime / "questioner_candidate_supersessions.jsonl") if v3_protocol else []
    )
    structured_diagnostics = (
        _read_jsonl(runtime / "structured_output_item_diagnostics.jsonl")
        if (runtime / "structured_output_item_diagnostics.jsonl").is_file()
        else []
    )

    _require_equal(len(compiler_runs), 1, "Questioner export must contain one compiler run")
    run_id = str(manifest["run_id"])
    if str(run_manifest.get("run_id", "")) != run_id:
        raise ValueError("Questioner run manifest differs from export run identity")
    dataset = str(run_manifest.get("dataset", ""))
    if dataset not in {"memlens", "memgallery"}:
        raise ValueError("Questioner export has an unsupported benchmark identity")
    snapshot_ids = run_manifest.get("dataset_snapshot_ids")
    if not isinstance(snapshot_ids, list) or len(snapshot_ids) != 1:
        raise ValueError("Questioner export must freeze exactly one dataset snapshot")
    snapshot_id = str(snapshot_ids[0])
    if str(compiler_runs[0].get("run_id", "")) != run_id:
        raise ValueError("Questioner compiler run differs from export run identity")
    if str(compiler_runs[0].get("snapshot_id", "")) != snapshot_id:
        raise ValueError("Questioner compiler run differs from the frozen snapshot")
    if not snapshot_id.startswith(f"{dataset}-"):
        raise ValueError("Questioner dataset and snapshot identities are inconsistent")
    for rows, label, include_snapshot in (
        (units, "compilation unit", True),
        (unit_memories, "compilation-unit memory", True),
        (unit_links, "compilation-unit link", True),
        (generations, "generation", False),
        (jobs, "job", True),
        (stages, "stage result", False),
        (questions, "QBank question", True),
        (paraphrases, "QBank paraphrase", True),
        (evidence_refs, "QBank evidence", True),
        (gates, "candidate gate", False),
        (duplicate_links, "duplicate link", False),
        (opportunity_sets, "opportunity set", True),
        (opportunities, "opportunity", True),
        (opportunity_evidence, "opportunity evidence", False),
        (verification_batches, "verification batch", False),
        (candidate_verifications, "candidate verification", False),
        (supersessions, "candidate supersession", False),
        (structured_diagnostics, "structured-output diagnostic", False),
    ):
        _require_row_scope(
            rows,
            label=label,
            run_id=run_id,
            snapshot_id=snapshot_id if include_snapshot else None,
        )
    _require_row_scope(
        indexes,
        label="index",
        run_id=run_id,
        run_field="compiler_run_id",
        snapshot_id=snapshot_id,
    )
    if any(str(row.get("snapshot_id", "")) != snapshot_id for row in collections):
        raise ValueError("Questioner embedding collection crosses snapshot scope")
    if not program_export:
        _require_row_scope(
            collections,
            label="embedding collection",
            run_id=run_id,
            snapshot_id=snapshot_id,
        )
    expected_units = int(compiler_runs[0]["unit_count"])
    for actual, label in (
        (len(units), "compilation units"),
        (len(generations), "Questioner generations"),
        (len(jobs), "COMPILE jobs"),
        (len(stages), "Questioner stage results"),
    ):
        _require_equal(actual, expected_units, f"Questioner {label} denominator is incomplete")
    unit_ids = {str(row["compiler_unit_id"]) for row in units}
    for rows, label in (
        (generations, "generation"),
        (jobs, "job"),
        (stages, "stage result"),
    ):
        _require_equal(
            {str(row["compiler_unit_id"]) for row in rows},
            unit_ids,
            f"Questioner {label} unit lineage is incomplete",
        )
    if any(row.get("status") != "succeeded" for row in jobs):
        raise ValueError("Questioner export contains a non-succeeded COMPILE job")
    if any(row.get("status") != "success" for row in stages):
        raise ValueError("Questioner export contains a non-success stage result")

    raw_count = 0
    valid_count = 0
    invalid_count = 0
    for generation in generations:
        raw = int(generation["raw_candidate_count"])
        valid = int(generation["valid_candidate_count"])
        invalid = int(generation["invalid_candidate_count"])
        _require_equal(
            valid + invalid,
            raw,
            "Questioner generation candidate accounting does not reconcile",
        )
        raw_count += raw
        valid_count += valid
        invalid_count += invalid
        metadata = generation.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Questioner generation metadata must be an object")
        if v2_protocol or v3_protocol:
            diagnostics = metadata.get("invalid_diagnostics", [])
            resolution = metadata.get("resolution_diagnostics", []) if v3_protocol else []
            if (
                not isinstance(diagnostics, list)
                or not isinstance(resolution, list)
                or len(diagnostics) + len(resolution) != invalid
            ):
                raise ValueError("Questioner invalid diagnostic accounting is incomplete")
            for diagnostic in diagnostics:
                if not isinstance(diagnostic, dict) or set(diagnostic) - {
                    "ordinal",
                    "item_hash",
                    "error_type",
                    "summary",
                    "error_count",
                    "errors",
                }:
                    raise ValueError("Questioner diagnostic is not redacted")
                errors = diagnostic.get("errors")
                if (
                    not isinstance(diagnostic.get("ordinal"), int)
                    or not isinstance(diagnostic.get("item_hash"), str)
                    or not isinstance(errors, list)
                    or diagnostic.get("error_count") != len(errors)
                    or diagnostic.get("error_type") not in {"ValidationError", "BudgetExceeded"}
                    or diagnostic.get("summary")
                    not in {
                        "candidate failed the strict Questioner schema",
                        "candidate exceeds frozen budget",
                    }
                ):
                    raise ValueError("Questioner diagnostic is not redacted")
                safe_errors: list[dict[str, str]] = []
                for error in errors:
                    if (
                        not isinstance(error, dict)
                        or set(error) != {"location", "type"}
                        or not isinstance(error.get("location"), str)
                        or not isinstance(error.get("type"), str)
                    ):
                        raise ValueError("Questioner diagnostic is not redacted")
                    safe_errors.append({"location": error["location"], "type": error["type"]})
                if safe_errors != sorted(
                    safe_errors,
                    key=lambda item: (item["location"], item["type"]),
                ):
                    raise ValueError("Questioner diagnostic errors are not sorted")
        generation_schema_hash = metadata.get("response_schema_hash")
        if (
            generation_schema_hash is not None and generation_schema_hash != expected_schema_hash
        ) or ((v2_protocol or v3_protocol) and generation_schema_hash is None):
            raise ValueError("Questioner generation has the wrong response schema hash")
    lineage_accounting: dict[str, int] = {}
    if v3_protocol:
        lineage_accounting = _verify_v3_lineage(
            unit_ids=unit_ids,
            unit_links=unit_links,
            generations=generations,
            opportunity_sets=opportunity_sets,
            opportunities=opportunities,
            opportunity_evidence=opportunity_evidence,
            stages=stages,
            questions=questions,
            gates=gates,
            verification_batches=verification_batches,
            candidate_verifications=candidate_verifications,
            supersessions=supersessions,
            isolated_candidate_scope=protocol_version
            in {
                "pmmc-questioner-v3.2",
                "pmmc-questioner-v3.3",
                "pmmc-questioner-v3.4",
                "pmmc-questioner-v3.5",
                "pmmc-questioner-v3.6",
            },
        )
        verifier_prompt_version = str(questioner_config["verifier_prompt_version"])
        verifier_schema_hash = questioner_response_schema_sha256(verifier_prompt_version)
        if any(
            row.get("prompt_version") != verifier_prompt_version
            or row.get("schema_hash") != verifier_schema_hash
            for row in verification_batches
        ):
            raise ValueError("Questioner verifier batch identity differs from the manifest")
        generated_count = sum(row.get("candidate_origin") == "generated" for row in questions)
        _require_equal(
            generated_count,
            valid_count,
            "Questioner v3 generated question count differs from valid drafts",
        )
    else:
        _require_equal(
            len(questions), valid_count, "QBank question count differs from valid drafts"
        )
    _require_equal(len(gates), len(questions), "QBank questions are missing gate results")
    _require_equal(
        {str(row["qbank_id"]) for row in gates},
        {str(row["qbank_id"]) for row in questions},
        "Questioner gate and QBank question lineage differs",
    )
    question_ids = {str(row["qbank_id"]) for row in questions}
    if any(
        str(row["duplicate_qbank_id"]) not in question_ids
        or str(row["primary_qbank_id"]) not in question_ids
        or row["duplicate_qbank_id"] == row["primary_qbank_id"]
        for row in duplicate_links
    ):
        raise ValueError("Questioner duplicate link has invalid QBank lineage")
    expected_hard_duplicates = {
        (str(row["qbank_id"]), str(row["duplicate_of_qbank_id"]))
        for row in gates
        if row.get("duplicate_of_qbank_id") is not None
    }
    actual_hard_duplicates = {
        (str(row["duplicate_qbank_id"]), str(row["primary_qbank_id"]))
        for row in duplicate_links
        if row.get("hard_duplicate") is True
    }
    _require_equal(
        actual_hard_duplicates,
        expected_hard_duplicates,
        "Questioner hard-duplicate lineage differs from gate facts",
    )

    _require_equal(len(indexes), 1, "Questioner export must contain one QBank index")
    index = indexes[0]
    if index.get("status") != "ready":
        raise ValueError("Questioner QBank index is not ready")
    collection_ids = [
        str(index[name])
        for name in (
            "text_question_collection_id",
            "text_paraphrase_collection_id",
            "multimodal_question_collection_id",
            "multimodal_paraphrase_collection_id",
        )
    ]
    _require_equal(
        len(set(collection_ids)),
        4,
        "Questioner QBank index must reference four distinct collections",
    )
    collection_map = {str(row["collection_id"]): row for row in collections}
    if len(collection_map) != len(collections):
        raise ValueError("Questioner export has duplicate embedding collection IDs")
    if program_export:
        if not set(collection_ids) <= set(collection_map):
            raise ValueError("Questioner QBank collection inventory is incomplete")
    else:
        _require_equal(
            set(collection_ids),
            set(collection_map),
            "Questioner export collection inventory differs from its QBank index",
        )
    qbank_collection_map = {
        collection_id: collection_map[collection_id] for collection_id in collection_ids
    }
    if any(
        row.get("status") != "ready" or int(row.get("dimension", 0)) != 4096
        for row in qbank_collection_map.values()
    ):
        raise ValueError("Questioner QBank collection identity is invalid")
    eligible_qbank_ids = {
        str(row["qbank_id"]) for row in gates if row.get("status") == "eligible_for_planning"
    }
    eligible_count = len(eligible_qbank_ids)
    severe_terminal_error_count = 0
    if v3_protocol:
        paraphrases_by_qbank: dict[str, list[str]] = {}
        for row in paraphrases:
            paraphrases_by_qbank.setdefault(str(row["qbank_id"]), []).append(
                str(row.get("paraphrase", ""))
            )
        for row in questions:
            qbank_id = str(row["qbank_id"])
            if qbank_id not in eligible_qbank_ids:
                continue
            public_text = " ".join(
                [
                    str(row.get("canonical_question", "")),
                    str(row.get("expected_answer", "") or ""),
                    *paraphrases_by_qbank.get(qbank_id, []),
                ]
            )
            if _INTERNAL_OR_METADATA_PATTERN.search(public_text) or (
                row.get("expected_answer")
                and _PLACEHOLDER_PATTERN.search(str(row["expected_answer"]))
            ):
                severe_terminal_error_count += 1
        if severe_terminal_error_count:
            raise ValueError("eligible Questioner v3 candidates contain severe terminal errors")
    eligible_paraphrase_ids = {
        str(row["paraphrase_id"])
        for row in paraphrases
        if str(row["qbank_id"]) in eligible_qbank_ids
    }
    _require_equal(
        int(index["eligible_question_count"]),
        eligible_count,
        "QBank index eligible count differs from gate facts",
    )
    _require_equal(
        int(index["paraphrase_count"]),
        len(eligible_paraphrase_ids),
        "QBank index paraphrase count differs from exported facts",
    )
    qbank_embeddings = [
        row for row in embeddings if str(row["collection_id"]) in set(collection_ids)
    ]
    _require_equal(
        len(qbank_embeddings),
        2 * (eligible_count + len(eligible_paraphrase_ids)),
        "QBank embedding count is incomplete",
    )
    if not program_export and len(qbank_embeddings) != len(embeddings):
        raise ValueError("QBank embeddings reference an unexpected collection")
    expected_embedding_sources = {
        *(
            (collection_id, "question", qbank_id)
            for collection_id in (collection_ids[0], collection_ids[2])
            for qbank_id in eligible_qbank_ids
        ),
        *(
            (collection_id, "paraphrase", paraphrase_id)
            for collection_id in (collection_ids[1], collection_ids[3])
            for paraphrase_id in eligible_paraphrase_ids
        ),
    }
    actual_embedding_sources: set[tuple[str, str, str]] = set()
    for row in qbank_embeddings:
        row_qbank_id = row.get("qbank_id")
        row_paraphrase_id = row.get("paraphrase_id")
        if (row_qbank_id is None) == (row_paraphrase_id is None):
            raise ValueError("QBank embedding does not have one typed source")
        actual_embedding_sources.add(
            (
                str(row["collection_id"]),
                "question" if row_qbank_id is not None else "paraphrase",
                str(row_qbank_id if row_qbank_id is not None else row_paraphrase_id),
            )
        )
    _require_equal(
        actual_embedding_sources,
        expected_embedding_sources,
        "QBank embedding typed source lineage is incomplete",
    )

    model_calls = _read_jsonl(runtime / "model_calls.jsonl")
    _require_row_scope(model_calls, label="model call", run_id=run_id)
    if v3_protocol:
        calls_by_id = {str(row["model_call_id"]): row for row in model_calls}
        if len(calls_by_id) != len(model_calls):
            raise ValueError("Questioner v3 model-call IDs are duplicated")
        jobs_by_unit = {str(row["compiler_unit_id"]): row for row in jobs}
        expected_model_alias = str(questioner_config["compiler_model_alias"])
        generation_rows = {str(row["compiler_unit_id"]): row for row in generations}
        generation_calls = {
            unit_id: calls_by_id.get(str(row["model_call_id"]))
            for unit_id, row in generation_rows.items()
        }
        generation_success_call_count = 0
        generation_terminal_rejection_call_count = 0
        invalid_generation_call_count = 0
        for unit_id, call in generation_calls.items():
            generation = generation_rows[unit_id]
            metadata = (
                dict(call.get("request_config") or {}).get("request_metadata", {})
                if call is not None
                else {}
            )
            common_lineage_valid = bool(
                call is not None
                and isinstance(metadata, dict)
                and metadata.get("task") == "pmmc_questioner"
                and str(call.get("compiler_unit_id")) == unit_id
                and call.get("job_id") == jobs_by_unit[unit_id].get("job_id")
                and call.get("model_alias") == expected_model_alias
                and isinstance(call.get("actual_model_id"), str)
                and bool(call["actual_model_id"])
            )
            if common_lineage_valid and call is not None and call.get("status") == "success":
                generation_success_call_count += 1
            elif (
                common_lineage_valid
                and call is not None
                and _is_terminal_generation_rejection_call(generation, call)
            ):
                generation_terminal_rejection_call_count += 1
            else:
                invalid_generation_call_count += 1
        verification_values_by_batch: dict[str, list[dict[str, Any]]] = {}
        for value in candidate_verifications:
            verification_values_by_batch.setdefault(
                str(value["verification_batch_id"]), []
            ).append(value)
        verifier_success_call_count = 0
        verifier_terminal_rejection_call_count = 0
        invalid_verifier_call_count = 0
        for batch in verification_batches:
            call = calls_by_id.get(str(batch["model_call_id"]))
            generation_call = generation_calls.get(str(batch["compiler_unit_id"]))
            metadata = (
                dict(call.get("request_config") or {}).get("request_metadata", {})
                if call is not None
                else {}
            )
            common_lineage_valid = bool(
                call is not None
                and generation_call is not None
                and isinstance(metadata, dict)
                and metadata.get("task") == "pmmc_questioner_verifier"
                and call.get("compiler_unit_id") == batch.get("compiler_unit_id")
                and call.get("job_id") == generation_call.get("job_id")
                and call.get("parent_model_call_id") == generation_call.get("model_call_id")
                and call.get("model_alias") == generation_call.get("model_alias")
                and call.get("actual_model_id") == generation_call.get("actual_model_id")
            )
            if common_lineage_valid and call is not None and call.get("status") == "success":
                verifier_success_call_count += 1
            elif (
                common_lineage_valid
                and call is not None
                and _is_terminal_verifier_rejection_call(
                    batch,
                    verification_values_by_batch.get(
                        str(batch["verification_batch_id"]), []
                    ),
                    call,
                )
            ):
                verifier_terminal_rejection_call_count += 1
            else:
                invalid_verifier_call_count += 1
        model_call_accounting = {
            "generation_success_call_count": generation_success_call_count,
            "generation_terminal_rejection_call_count": (
                generation_terminal_rejection_call_count
            ),
            "invalid_generation_call_count": invalid_generation_call_count,
            "verifier_success_call_count": verifier_success_call_count,
            "verifier_terminal_rejection_call_count": (
                verifier_terminal_rejection_call_count
            ),
            "invalid_verifier_call_count": invalid_verifier_call_count,
        }
        print(
            "QUESTIONER_EXPORT_MODEL_CALL_ACCOUNTING "
            + orjson.dumps(model_call_accounting, option=orjson.OPT_SORT_KEYS).decode("utf-8"),
            flush=True,
        )
        if invalid_generation_call_count:
            raise ValueError("Questioner v3 generation model-call lineage is invalid")
        if invalid_verifier_call_count:
            raise ValueError("Questioner v3 verifier model-call lineage is invalid")
    else:
        model_call_accounting = {}
    artifact_records = {
        str(row["artifact_id"]): row for row in _read_jsonl(runtime / "run_artifacts.jsonl")
    }
    _require_row_scope(
        list(artifact_records.values()),
        label="runtime artifact",
        run_id=run_id,
    )
    artifact_manifest = _read_jsonl(runtime / "private" / "model_artifact_manifest.jsonl")
    expected_refs = {
        (str(call["model_call_id"]), role, str(artifact_id))
        for call in model_calls
        for role, artifact_id in (
            ("input", call.get("input_asset_id")),
            ("output", call.get("output_asset_id")),
        )
        if artifact_id is not None
    }
    archived_refs = {
        (str(row["model_call_id"]), str(row["role"]), str(row["artifact_id"]))
        for row in artifact_manifest
    }
    _require_equal(
        len(archived_refs),
        len(artifact_manifest),
        "Questioner model artifact manifest contains duplicate references",
    )
    _require_equal(
        archived_refs,
        expected_refs,
        "Questioner model artifact manifest does not cover every model-call reference",
    )
    payload_paths: set[str] = set()
    for row in artifact_manifest:
        if {"object_key", "storage_uri", "presigned_url"} & set(row):
            raise ValueError("Questioner model artifact manifest exposes storage details")
        artifact_id = str(row["artifact_id"])
        record = artifact_records.get(artifact_id)
        if record is None or record.get("state") != "ready":
            raise ValueError("Questioner model artifact is absent or not ready")
        if str(record.get("artifact_type")) != str(row["artifact_type"]):
            raise ValueError("Questioner model artifact type differs from runtime facts")
        if str(record.get("sha256")) != str(row["sha256"]) or int(
            record.get("size_bytes", -1)
        ) != int(row["size_bytes"]):
            raise ValueError("Questioner model artifact identity differs from runtime facts")
        expected_type = "prompt" if row["role"] == "input" else "model_response"
        if row["artifact_type"] != expected_type:
            raise ValueError("Questioner model artifact role and type are inconsistent")
        relative = validate_object_key(str(row["relative_path"]))
        mime_type = row.get("mime_type")
        extension = (
            "json"
            if mime_type == "application/json"
            or (isinstance(mime_type, str) and mime_type.endswith("+json"))
            else "txt"
            if isinstance(mime_type, str) and mime_type.startswith("text/")
            else "bin"
        )
        expected_relative = (
            "runtime/private/model_artifacts/"
            f"{row['artifact_type']}/sha256/{str(row['sha256'])[:2]}/"
            f"{row['sha256']}.{extension}"
        )
        if relative != expected_relative:
            raise ValueError("Questioner model payload path is not content-addressed")
        path = export_dir / relative
        if not path.is_file():
            raise ValueError("Questioner model artifact payload is missing")
        data = path.read_bytes()
        if len(data) != int(row["size_bytes"]):
            raise ValueError("Questioner model artifact payload size is invalid")
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Questioner model artifact payload checksum is invalid")
        lowered = data.lower()
        if any(
            marker in lowered
            for marker in (
                b"x-amz-signature",
                b"x-amz-credential",
                b"awsaccesskeyid",
                b"x-goog-signature",
                b"x-goog-credential",
            )
        ):
            raise ValueError("Questioner model artifact contains a presigned URL")
        payload_paths.add(relative)
    payload_root = runtime / "private" / "model_artifacts"
    actual_payload_paths = (
        {
            path.relative_to(export_dir).as_posix()
            for path in payload_root.rglob("*")
            if path.is_file()
        }
        if payload_root.is_dir()
        else set()
    )
    _require_equal(
        actual_payload_paths,
        payload_paths,
        "Questioner private artifact directory contains unreferenced payloads",
    )
    _require_equal(
        int(manifest.get("model_artifact_reference_count", -1)),
        len(artifact_manifest),
        "Questioner export artifact reference count is inconsistent",
    )
    _require_equal(
        int(manifest.get("model_artifact_payload_count", -1)),
        len(payload_paths),
        "Questioner export artifact payload count is inconsistent",
    )
    return {
        "run_id": run_id,
        "dataset": dataset,
        "snapshot_id": snapshot_id,
        "export_id": str(manifest["export_id"]),
        "program_runtime_export_version": manifest.get(
            "program_runtime_export_version"
        ),
        "protocol_version": protocol_version,
        "unit_count": expected_units,
        "raw_candidate_count": raw_count,
        "schema_valid_candidate_count": valid_count,
        "schema_invalid_candidate_count": invalid_count,
        "eligible_candidate_count": eligible_count,
        "opportunity_count": len(opportunities),
        "verification_batch_count": len(verification_batches),
        "verification_candidate_count": len(candidate_verifications),
        **lineage_accounting,
        **model_call_accounting,
        "revised_candidate_count": sum(
            row.get("candidate_origin") == "revised" for row in questions
        ),
        "revision_rescued_count": sum(
            row.get("candidate_origin") == "revised" and str(row["qbank_id"]) in eligible_qbank_ids
            for row in questions
        ),
        "verifier_scope_contamination_count": 0,
        "severe_terminal_error_count": severe_terminal_error_count,
        "questioner_native_output_issue_count": sum(
            row.get("stage")
            in {
                "questioner",
                "questioner_verifier",
                "questioner_revision",
            }
            and row.get("status")
            in {"invalid", "repaired", "conservative_reject"}
            for row in structured_diagnostics
        ),
        "planner_native_output_issue_count": sum(
            row.get("stage") == "planner"
            and row.get("status")
            in {"invalid", "repaired", "conservative_reject"}
            for row in structured_diagnostics
        ),
        "qbank_collection_count": len(collection_ids),
        "qbank_embedding_count": len(qbank_embeddings),
        "exported_embedding_count": len(embeddings),
        "model_artifact_reference_count": len(artifact_manifest),
        "model_artifact_payload_count": len(payload_paths),
        "valid": True,
        "issues": [],
    }
