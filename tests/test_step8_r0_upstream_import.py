"""Offline contracts for bounded pre-D proofs, projection and verified dispatch."""

from __future__ import annotations

import asyncio
import copy
import hashlib
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from pmmc.compiler.ids import canonical_content_hash
from pmmc.compiler.memory_index import make_memory_operator_index
from pmmc.experiments.step8_artifacts import CampaignArtifactStore
from pmmc.experiments.step8_upstream_import import (
    R0PrefixImportProof,
    assert_compatible,
    assert_r0_prefix_compatible,
    compatible_r0_prefix_proofs,
    publish_r0_prefix_proof,
    r0_prefix_compatibility,
)
from pmmc.infrastructure.database.step8_upstream_import import (
    QUESTIONER_TABLES,
    R0_PREFIX_TABLES,
    PostgresR0PrefixProofRepository,
    _prefix_call_dependencies,
    normalize_r0_prefix_rows,
    project_r0_prefix_rows,
    r0_prefix_rows_hash,
    r0_prefix_table_contract_hash,
    validate_r0_memory_vectors,
    validate_r0_prefix_rows,
)
from test_step8_upstream_import import SCOPE, campaign, proof


def prefix_proof():
    source = campaign()
    compatibility = r0_prefix_compatibility(source, SCOPE)
    return R0PrefixImportProof(
        source_campaign_id=source.campaign_id,
        source_protocol_hash=source.protocol.content_hash,
        source_run_id="source-run",
        compatibility=compatibility,
        compatibility_hash=canonical_content_hash(compatibility),
        schema_revision=proof().schema_revision,
        table_contract_hash=r0_prefix_table_contract_hash(),
        facts_hash="a" * 64,
        row_counts={table.name: 0 for table in (*QUESTIONER_TABLES, *R0_PREFIX_TABLES)},
        artifact_hashes={},
        questioner_proof=proof(),
        r0_content_hash="b" * 64,
    )


def test_doubter_only_changes_are_compatible_without_weakening_pre_d_sources():
    source = campaign()
    changed = source.protocol.model_copy(
        update={
            "doubter": "different-doubter",
            "config_hashes": source.protocol.config_hashes | {"doubter_config": "f" * 64},
            "source_fingerprints": source.protocol.source_fingerprints | {"doubter": "f" * 64},
        }
    )
    target = source.model_copy(update={"campaign_id": "new", "protocol": changed})
    assert target.protocol.content_hash != source.protocol.content_hash
    assert_r0_prefix_compatible(prefix_proof(), r0_prefix_compatibility(target, SCOPE))


@pytest.mark.parametrize(
    "stage",
    [
        "unit_prepare",
        "questioner",
        "planner0",
        "text_index",
        "multimodal_index",
        "dry_run0",
        "r0_gate",
        "projection",
    ],
)
def test_every_non_doubter_scientific_fingerprint_remains_binding(stage):
    source = campaign()
    changed = source.protocol.model_copy(
        update={
            "source_fingerprints": source.protocol.source_fingerprints | {stage: "f" * 64},
        }
    )
    with pytest.raises(ValueError, match="scientific_identity_mismatch"):
        assert_r0_prefix_compatible(
            prefix_proof(),
            r0_prefix_compatibility(
                source.model_copy(update={"protocol": changed}),
                SCOPE,
            ),
        )


@pytest.mark.parametrize("field", ["questioner_config", "planner_config", "foundation_config"])
def test_prefix_config_changes_fail_closed(field):
    source = campaign()
    changed = source.protocol.model_copy(
        update={
            "config_hashes": source.protocol.config_hashes | {field: "f" * 64},
        }
    )
    with pytest.raises(ValueError, match="scientific_identity_mismatch"):
        assert_r0_prefix_compatible(
            prefix_proof(),
            r0_prefix_compatibility(
                source.model_copy(update={"protocol": changed}),
                SCOPE,
            ),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 10),
        ("partition_hash", "new"),
        ("exposure_manifest_hash", "new"),
        ("operator_set", "new"),
        ("planner", "new"),
        ("answer_contract", "new"),
        ("match_threshold", 0.7),
    ],
)
def test_only_doubter_scientific_change_is_in_scope(field, value):
    source = campaign()
    changed = source.protocol.model_copy(update={field: value})
    with pytest.raises(ValueError, match="scientific_identity_mismatch"):
        assert_r0_prefix_compatible(
            prefix_proof(),
            r0_prefix_compatibility(
                source.model_copy(update={"protocol": changed}),
                SCOPE,
            ),
        )


@pytest.mark.parametrize(
    "alias",
    [
        "compiler-qwen3.5-9b",
        "qwen3-embedding-8b",
        "qwen3-vl-embedding-8b",
    ],
)
def test_shared_model_revision_cannot_be_called_doubter_only(alias):
    source = campaign()
    changed = source.protocol.model_copy(
        update={
            "model_entry_hashes": source.protocol.model_entry_hashes | {alias: "f" * 64},
        }
    )
    with pytest.raises(ValueError, match="scientific_identity_mismatch"):
        assert_r0_prefix_compatible(
            prefix_proof(),
            r0_prefix_compatibility(
                source.model_copy(update={"protocol": changed}),
                SCOPE,
            ),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"snapshot_id": "new"},
        {"conversation_id": "new"},
        {"dataset": "memgallery"},
        {"treatment": "no_raw_image"},
        {"treatment": "fixed_planner"},
    ],
)
def test_prefix_scope_cannot_cross_treatment_or_snapshot(change):
    with pytest.raises(ValueError, match="scientific_identity_mismatch"):
        assert_r0_prefix_compatible(
            prefix_proof(),
            r0_prefix_compatibility(
                campaign(),
                SCOPE | change,
            ),
        )


def test_diagnostic_namespace_is_never_a_cross_campaign_source():
    source = campaign().model_copy(update={"diagnostic_only": True})
    assert r0_prefix_compatibility(source, SCOPE) != r0_prefix_compatibility(
        source.model_copy(update={"campaign_id": "another"}), SCOPE
    )


def test_prefix_proof_binds_its_nested_questioner_owner():
    value = prefix_proof()
    corrupt = value.model_copy(
        update={
            "questioner_proof": value.questioner_proof.model_copy(
                update={"source_run_id": "other"}
            ),
        }
    )
    with pytest.raises(ValueError, match="proof_identity_invalid"):
        assert_r0_prefix_compatible(corrupt, value.compatibility)
    corrupt_q = proof().model_copy(update={"schema_revision": "legacy"})
    with pytest.raises(ValueError, match="proof_identity_invalid"):
        assert_compatible(corrupt_q, corrupt_q.compatibility)


def test_prefix_registry_is_immutable_and_checks_cas_bytes(tmp_path):
    store, value = CampaignArtifactStore(tmp_path), prefix_proof()
    digest = publish_r0_prefix_proof(store, value)
    assert compatible_r0_prefix_proofs(store, value.compatibility, exclude_run_id="target") == [
        value
    ]
    assert not compatible_r0_prefix_proofs(store, value.compatibility, exclude_run_id="source-run")
    with pytest.raises(ValueError, match="immutable artifact"):
        publish_r0_prefix_proof(store, value.model_copy(update={"facts_hash": "c" * 64}))
    store.object_path(digest).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        compatible_r0_prefix_proofs(store, value.compatibility, exclude_run_id="target")


def test_prefix_registry_rejects_symlink_ancestors(tmp_path):
    store, value = CampaignArtifactStore(tmp_path), prefix_proof()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "_store").mkdir()
    (tmp_path / "_store/upstream").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="registry_symlink"):
        compatible_r0_prefix_proofs(store, value.compatibility, exclude_run_id="target")
    with pytest.raises(ValueError, match="registry_symlink"):
        publish_r0_prefix_proof(store, value)


def native_empty_prefix():
    """A valid zero-eligible source is still required to account for its units."""
    rows = {table.name: [] for table in (*QUESTIONER_TABLES, *R0_PREFIX_TABLES)}
    rows["compiler_units"] = [{"compiler_unit_id": "unit"}]
    rows["compiler_stage_jobs"] = [
        {
            "stage_job_id": "stage-job-" + stage,
            "compiler_unit_id": "unit",
            "stage": stage,
            "status": "succeeded",
            "round_id": 0,
        }
        for stage in ("planner", "dry_run")
    ]
    rows["qbank_index_manifests"] = [
        {
            "qbank_index_id": "qi",
            "status": "ready",
            "eligible_question_count": 0,
            "paraphrase_count": 0,
            "text_question_collection_id": "tq",
            "text_paraphrase_collection_id": "tp",
            "multimodal_question_collection_id": "mq",
            "multimodal_paraphrase_collection_id": "mp",
        }
    ]
    rows["memory_operator_index_manifests"] = [
        {
            "memory_index_id": "mi",
            "status": "ready",
            "text_collection_id": "tm",
            "visual_collection_id": "vm",
        }
    ]
    rows["embedding_collections"] = [
        {
            "collection_id": name,
            "run_id": "source",
            "status": "ready",
            "dimension": 4096,
            "model_revision": "commit",
        }
        for name in ("tq", "tp", "mq", "mp", "tm", "vm")
    ]
    rows["r0_input_manifests"] = [
        {
            "r0_manifest_id": "r0",
            "hard_gate_passed": True,
            "candidate_count": 0,
            "program_count": 0,
            "execution_count": 0,
            "terminal_outcome_count": 0,
        }
    ]
    return rows


def native_program_prefix():
    from pmmc.compiler.ids import memory_program_content_hash
    from pmmc.infrastructure.database.compiler_mappers import (
        memory_program_to_rows,
        question_to_rows,
    )
    from test_step8_compiler_revision import dry_run, fixture

    question, requirement, program = fixture()
    question = question.model_copy(update={"candidate_content_hash": question.question_hash})
    program = program.model_copy(
        update={
            "expected_evidence_ids": [],
            "operator_sequence": [
                step.model_copy(update={"expected_evidence_ids": []})
                for step in program.operator_sequence
            ],
        }
    )
    program = program.model_copy(update={"program_hash": memory_program_content_hash(program)})
    rows = native_empty_prefix()
    rows["compiler_units"][0].update(run_id="run", input_hash="unit-input", metadata={})
    qrows = question_to_rows(question)
    rows["qbank_questions"] = qrows["question"]
    rows["qbank_evidence_refs"] = qrows["source"]
    rows["questioner_candidate_gate_results"] = [
        {
            "gate_result_id": "gate",
            "run_id": "run",
            "qbank_id": "q1",
            "status": "eligible_for_planning",
            "content_hash": question.question_hash,
            "metadata": {},
        }
    ]
    rows["planner_requirement_contracts"] = [requirement.model_dump(mode="json")]
    prows = memory_program_to_rows(program)
    rows["memory_programs"], rows["program_steps"] = prows["program"], prows["steps"]
    rows["planner_candidate_outcomes"] = [
        {
            "planner_outcome_id": "outcome",
            "qbank_id": "q1",
            "run_id": "run",
            "content_hash": "outcome-hash",
            "status": "program_ready",
            "metadata": {},
        }
    ]
    rows["program_executions"] = [
        {
            "execution_id": "exec",
            "plan_id": program.plan_id,
            "run_id": "run",
            "source_compiler_run_id": "run",
            "purpose": "compiler_dry_run",
            "query_id": None,
            "status": "success",
            "metadata": {},
        }
    ]
    rows["program_dry_run_assessments"] = [dry_run(program).model_dump(mode="json")]
    rows["model_calls"] = [
        {
            "model_call_id": "p-call",
            "run_id": "run",
            "parent_model_call_id": None,
            "job_id": "p-job",
            "stage": "planner",
            "metadata": {},
            "cache_hit": False,
            "request_config": {"request_id": "p-call", "text": "run"},
            "input_tokens": 42,
        }
    ]
    index, text, visual = make_memory_operator_index(
        run_id="run",
        snapshot_id="snap",
        dataset="memlens",
        selection_hash="source-selection",
        text_model_alias="text",
        text_model_revision="t-commit",
        multimodal_model_alias="joint",
        multimodal_model_revision="m-commit",
    )
    rows["memory_operator_index_manifests"] = [{**index.model_dump(mode="json"), "status": "ready"}]
    rows["embedding_collections"] = rows["embedding_collections"][:4] + [
        {**collection.model_dump(mode="json"), "status": "ready"} for collection in (text, visual)
    ]
    for name in ("qt", "qm"):
        rows["embedding_collections"].append(
            {
                "collection_id": name,
                "run_id": "run",
                "status": "ready",
                "dimension": 4096,
                "model_revision": "commit",
                "metadata": {},
            }
        )
    for collection in rows["embedding_collections"]:
        collection.update(run_id="run", metadata=collection.get("metadata", {}))
    for name in ("tq", "mq", "qt", "qm"):
        rows["embeddings"].append(
            {
                "embedding_id": "vector-" + name,
                "collection_id": name,
                "qbank_id": "q1",
                "dimension": 4096,
                "embedding": [1.0] + [0.0] * 4095,
                "input_hash": "input-" + name,
                "metadata": {},
            }
        )
    rows["program_query_embeddings"] = [
        {
            "program_query_embedding_id": "ref-" + name,
            "qbank_id": "q1",
            "embedding_id": "vector-" + name,
            "channel": channel,
            "input_hash": "input-" + name,
            "metadata": {},
        }
        for name, channel in (("qt", "question_text_query"), ("qm", "question_joint_query"))
    ]
    rows["qbank_index_manifests"][0]["eligible_question_count"] = 1
    rows["r0_input_manifests"][0].update(
        candidate_count=1,
        program_count=1,
        execution_count=1,
        terminal_outcome_count=1,
    )
    questioner_rows = {t.name: copy.deepcopy(rows[t.name]) for t in QUESTIONER_TABLES}
    questioner_rows["model_calls"] = []
    value = prefix_proof()
    value = value.model_copy(
        update={
            "source_run_id": "run",
            "questioner_proof": value.questioner_proof.model_copy(update={"source_run_id": "run"}),
        }
    )
    validate_r0_prefix_rows(rows, "run")
    return rows, questioner_rows, value


def test_target_program_passes_native_validator_after_projection():
    from pmmc.compiler.ids import memory_program_content_hash
    from pmmc.infrastructure.database.compiler_mappers import memory_program_from_rows
    from pmmc.memory.operators import ProgramValidator
    from pmmc.retrieval.operators import build_operator_registry
    from test_step8_operator_runtime import config

    rows, qrows, value = native_program_prefix()
    projected, identities = project_r0_prefix_rows(
        rows,
        qrows,
        proof=value,
        target_run_id="target",
        target_selection_hash="target-selection",
    )
    row = projected["memory_programs"][0]
    program = memory_program_from_rows(row, projected["program_steps"], [])
    validator = ProgramValidator(build_operator_registry(object(), config()), config())
    result = validator.validate(program)
    assert result.valid, result.issues
    assert program.program_hash == memory_program_content_hash(program)
    assert program.run_id == "target" and program.qbank_id == identities["q1"]
    assert (
        row["metadata"]["r0_prefix_import"]["source_program_hash"]
        == rows["memory_programs"][0]["program_hash"]
    )
    assert projected["program_steps"][0]["args"] == rows["program_steps"][0]["args"]
    assert projected["r0_input_manifests"] == []
    call = projected["model_calls"][0]
    assert call["request_config"] == rows["model_calls"][0]["request_config"]
    assert call["input_tokens"] == 42 and call["cache_hit"] is True
    assert "r0_prefix_import" in call["metadata"] and "upstream_import" not in call["metadata"]
    assert "upstream_import" in projected["qbank_questions"][0]["metadata"]
    validate_r0_prefix_rows(projected, "target", require_r0=False)
    validate_r0_memory_vectors(projected, [], memory_raw_image_policy="allow")


def test_post_d_acceptance_does_not_change_r0_denominator_proof_or_projection():
    rows, qrows, value = native_program_prefix()
    rejected = copy.deepcopy(rows)
    rejected["memory_programs"][0]["accepted"] = False
    assert normalize_r0_prefix_rows(rows) == normalize_r0_prefix_rows(rejected)
    assert r0_prefix_rows_hash(rows) == r0_prefix_rows_hash(rejected)
    options = dict(proof=value, target_run_id="target", target_selection_hash="selection")
    accepted_projection, _ = project_r0_prefix_rows(rows, qrows, **options)
    rejected_projection, _ = project_r0_prefix_rows(rejected, qrows, **options)
    assert accepted_projection == rejected_projection
    assert len(rejected_projection["memory_programs"]) == 1
    assert rejected_projection["memory_programs"][0]["accepted"] is True
    assert rejected["memory_programs"][0]["accepted"] is False  # Never mutate producer rows.
    changed = copy.deepcopy(rows)
    changed["program_steps"][0]["args"]["top_k"] = 999
    assert r0_prefix_rows_hash(changed) != r0_prefix_rows_hash(rows)
    with pytest.raises(ValueError, match="program_hash_mismatch"):
        project_r0_prefix_rows(changed, qrows, **options)


def test_direct_embedding_attempts_and_failed_call_artifacts_are_separately_accounted():
    calls = [
        {
            "model_call_id": "direct",
            "job_id": None,
            "stage": "embed",
            "input_asset_id": "input",
            "output_asset_id": None,
            "metadata": {"error_artifact_id": "failed-response"},
        },
        {
            "model_call_id": "queued",
            "job_id": "job",
            "stage": "planner",
            "input_asset_id": "p-input",
            "output_asset_id": "p-output",
            "metadata": {},
        },
    ]
    jobs, direct, artifacts = _prefix_call_dependencies(calls)
    assert jobs == {"job"} and direct == {"direct"}
    assert artifacts == {"input", "failed-response", "p-input", "p-output"}
    calls[0]["stage"] = "planner"
    with pytest.raises(ValueError, match="unbound_non_embedding_call"):
        _prefix_call_dependencies(calls)


@pytest.mark.parametrize(
    "change",
    [
        {"stage": "answer"},
        {"stage": "doubter"},
        {"plan_id": "revision-source"},
        {"query_id": "runtime-query"},
    ],
)
def test_prefix_call_closure_rejects_runtime_or_revision_ancestry(change):
    row = {
        "model_call_id": "call",
        "stage": "embed",
        "job_id": None,
        "query_id": None,
        "plan_id": None,
        "input_asset_id": None,
        "output_asset_id": None,
        "metadata": {},
        **change,
    }
    with pytest.raises(ValueError, match="non_prefix_call"):
        _prefix_call_dependencies([row])


def test_reader_retains_unbound_failed_attempt_error_and_normalizes_only_program_acceptance(
    monkeypatch,
):
    from pmmc.infrastructure.database.repositories import PostgresMemoryRepository
    from pmmc.infrastructure.database.step8_upstream_import import (
        PostgresQuestionerImportRepository,
    )

    rows, qrows, _ = native_program_prefix()
    rows["memory_programs"][0]["accepted"] = False
    rows["model_calls"][0].update(input_asset_id=None, output_asset_id=None)
    rows["model_calls"].append(
        {
            "model_call_id": "failed-direct",
            "run_id": "run",
            "stage": "embed",
            "job_id": None,
            "parent_model_call_id": None,
            "input_asset_id": "direct-input",
            "output_asset_id": None,
            "metadata": {"error_artifact_id": "direct-error"},
        }
    )
    rows["experiment_jobs"] = [
        {"job_id": "p-job", "run_id": "run", "status": "succeeded", "error_artifact_id": None}
    ]
    rows["run_artifacts"] = [
        {"artifact_id": aid, "run_id": "run"} for aid in ("direct-input", "direct-error")
    ]
    headers = {
        "compiler": {"snapshot_id": "snap", "metadata": {"conversation_ids": ["conversation"]}},
        "manifest": {"metadata": {"questioner_config": {}}},
    }
    selected = {}

    class Session:
        async def execute(self, statement):
            table = statement.get_final_froms()[0].name
            selected.setdefault(table, []).append(statement.compile().params)
            return SimpleNamespace(mappings=lambda: copy.deepcopy(rows[table]))

        async def scalars(self, statement):
            return [r["model_call_id"] for r in rows["model_calls"]]

    session = Session()

    async def read(self, borrowed, run_id, *, include_index_diagnostics=False):
        assert borrowed is session and run_id == "run"
        assert include_index_diagnostics is True
        return copy.deepcopy(headers), copy.deepcopy(qrows)

    async def memories(self, snapshot_id, **kwargs):
        assert snapshot_id == "snap"
        return []

    monkeypatch.setattr(PostgresQuestionerImportRepository, "_read", read)
    monkeypatch.setattr(PostgresMemoryRepository, "list_snapshot", memories)
    repository = PostgresR0PrefixProofRepository(None, None)
    actual_headers, actual = asyncio.run(repository._read_prefix(session, "run"))
    assert actual_headers["direct_model_call_ids"] == ["failed-direct"]
    assert {r["model_call_id"] for r in actual["model_calls"]} == {"p-call", "failed-direct"}
    assert actual["memory_programs"][0]["accepted"] is True
    assert {r["artifact_id"] for r in actual["run_artifacts"]} == {"direct-input", "direct-error"}
    assert {r["job_id"] for r in actual["experiment_jobs"]} == {"p-job"}
    params = selected["experiment_jobs"][0]
    assert list(params.values()) == [["p-job"]]
    artifact_ids = next(iter(selected["run_artifacts"][0].values()))
    assert set(artifact_ids) == {"direct-input", "direct-error"}


@pytest.mark.parametrize("accepted", [False, True])
def test_legacy_post_d_question_projection_is_rejected_not_used_as_q_gate(accepted):
    rows, qrows, value = native_program_prefix()
    rows["qbank_questions"][0]["accepted"] = accepted
    with pytest.raises(ValueError, match="legacy_question_decision_forbidden"):
        project_r0_prefix_rows(
            rows,
            qrows,
            proof=value,
            target_run_id="target",
            target_selection_hash="selection",
        )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (None, None),
        ("source_run_id", "consumer_proof_conflict"),
        ("target_selection_hash", "consumer_proof_conflict"),
        ("target_rows_hash", "consumer_fact_conflict"),
        ("identity_map_hash", "consumer_fact_conflict"),
        ("source_proof_sha256", "consumer_proof_conflict"),
        ("imported_stages", "consumer_lineage_invalid"),
        ("program_hash", "consumer_fact_conflict"),
        ("provider_usage", "consumer_fact_conflict"),
        ("artifact_bytes", "artifact_checksum"),
        ("r0_semantics", "target_r0_semantics_changed"),
    ],
)
def test_target_verification_reconstructs_projection_and_native_r0(monkeypatch, mutation, error):
    from pmmc.compiler import r0_gate
    from pmmc.experiments.step8_artifacts import json_bytes
    from pmmc.experiments.step8_upstream_import import R0_ADOPTION_VERSION, R0_PREFIX_STAGES
    from pmmc.infrastructure.database.step8_upstream_import import _prefix_registration_hash

    source, qrows, value = native_program_prefix()
    source_q_headers = {"compiler": {"metadata": {}}, "manifest": {"metadata": {}}}
    value = value.model_copy(
        update={
            "questioner_proof": value.questioner_proof.model_copy(
                update={
                    "facts_hash": canonical_content_hash(
                        {"headers": source_q_headers, "rows": qrows}
                    ),
                }
            )
        }
    )
    source["r0_input_manifests"][0].update(
        answerable_candidate_count=1,
        control_candidate_count=0,
        issue_codes=[],
        protocol_integrity_issues=[],
        candidate_scientific_rejections=[],
        coverage_zero_units=[],
        downstream_viability="viable",
    )
    projected, identities = project_r0_prefix_rows(
        source,
        qrows,
        proof=value,
        target_run_id="target",
        target_selection_hash="selection",
    )
    target_headers = {
        "compiler": {"metadata": {}},
        "manifest": {"selection_hash": "selection", "metadata": {}},
    }
    lineage = {
        "version": R0_ADOPTION_VERSION,
        "imported_stages": list(R0_PREFIX_STAGES[1:]),
        "provider_calls": 0,
        "proof_json": value.model_dump(mode="json"),
        "proof_hash": value.content_hash,
        "source_run_id": value.source_run_id,
        "source_campaign_id": value.source_campaign_id,
        "source_protocol_hash": value.source_protocol_hash,
        "source_facts_hash": value.facts_hash,
        "compatibility_hash": value.compatibility_hash,
        "source_proof_sha256": hashlib.sha256(
            json_bytes(value.model_dump(mode="json"))
        ).hexdigest(),
        "target_selection_hash": "selection",
        "target_rows_hash": r0_prefix_rows_hash(projected),
        "identity_map_hash": canonical_content_hash(identities),
        "target_registration_hash": _prefix_registration_hash(target_headers),
    }
    target_headers["compiler"]["metadata"]["r0_prefix_upstream_import"] = lineage
    target_headers["manifest"]["metadata"]["r0_prefix_upstream_import"] = lineage
    if mutation in lineage:
        lineage[mutation] = "tampered"
    elif mutation == "program_hash":
        projected["memory_programs"][0]["program_hash"] = "tampered"
    elif mutation == "provider_usage":
        projected["model_calls"][0]["input_tokens"] += 1
    checked = []

    @asynccontextmanager
    async def sessions():
        yield object()

    async def registration(run_id):
        assert run_id == "target"
        return lineage

    async def verify_source(proof, compatibility):
        assert_r0_prefix_compatible(proof, compatibility)
        checked.append("source")
        return {}, source

    async def read(session, run_id):
        return (source_q_headers, qrows) if run_id == "run" else (target_headers, {})

    async def read_prefix(session, run_id, *, require_r0):
        assert run_id == "target" and require_r0
        return target_headers, projected

    async def artifacts(rows):
        checked.append("artifacts")
        if mutation == "artifact_bytes":
            raise ValueError("artifact_checksum_mismatch")
        return {}

    async def r0(sessions, *, run_id):
        assert run_id == "target"
        checked.append("native_r0")
        values = dict(source["r0_input_manifests"][0])
        if mutation == "r0_semantics":
            values["program_count"] += 1
        return SimpleNamespace(**values)

    repository = PostgresR0PrefixProofRepository(sessions, None)
    monkeypatch.setattr(repository, "prefix_registration", registration)
    monkeypatch.setattr(repository, "verify_source", verify_source)
    monkeypatch.setattr(repository, "_read", read)
    monkeypatch.setattr(repository, "_read_prefix", read_prefix)
    monkeypatch.setattr(repository, "_artifacts", artifacts)
    monkeypatch.setattr(r0_gate, "verify_r0_input", r0)
    if error:
        with pytest.raises(ValueError, match=error):
            asyncio.run(repository.verify_target("target", value.compatibility))
    else:
        assert asyncio.run(repository.verify_target("target", value.compatibility)) == lineage
        assert checked == ["source", "artifacts", "native_r0"]


@pytest.mark.parametrize(
    "stage",
    [
        "planner0",
        "text_index",
        "multimodal_index",
        "dry_run0",
        "r0_gate",
        "doubter",
        "projection",
    ],
)
def test_import_dispatch_verifies_every_consumer_and_never_skips_doubter(monkeypatch, stage):
    from pmmc.experiments import step8_upstream_import as api
    from pmmc.infrastructure import object_store
    from pmmc.infrastructure.database import engine
    from pmmc.infrastructure.database.step8_upstream_import import PostgresR0PrefixImportRepository

    calls = []
    value = prefix_proof()

    async def dispose():
        calls.append("disposed")

    async def verify(self, run_id, compatibility):
        assert_r0_prefix_compatible(value, compatibility)
        calls.append("verified")
        return {"proof_json": value.model_dump(mode="json")}

    async def seal(backend, scope):
        calls.append("sealed")
        return {"r0_prefix_import_proof_sha256": "digest"}

    monkeypatch.setattr(
        engine, "create_session_factory", lambda *args: (SimpleNamespace(dispose=dispose), None)
    )
    monkeypatch.setattr(object_store, "create_runtime_object_store", lambda settings: None)
    monkeypatch.setattr(PostgresR0PrefixImportRepository, "verify_target", verify)
    monkeypatch.setattr(api, "seal_r0_prefix", seal)
    result = asyncio.run(
        api.verified_r0_prefix_stage(
            SimpleNamespace(campaign=campaign(), settings=None), SCOPE, stage
        )
    )
    assert calls[:2] == ["verified", "disposed"]
    if stage in {"doubter", "projection"}:
        assert result is None
    else:
        assert result["verified_only"] and result["phase"] == stage
    assert ("sealed" in calls) == (stage == "r0_gate")


def test_empty_eligible_denominator_does_not_mean_missing_units_are_ok():
    rows = native_empty_prefix()
    validate_r0_prefix_rows(rows, "source")
    rows["compiler_stage_jobs"].pop()
    with pytest.raises(ValueError, match="stage_not_terminal"):
        validate_r0_prefix_rows(rows, "source")


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "foreign-producer"),
        ("status", "building"),
        ("model_revision", ""),
    ],
)
def test_foreign_or_unfinished_collections_are_not_relabeled(field, value):
    rows = native_empty_prefix()
    rows["embedding_collections"][0][field] = value
    with pytest.raises(ValueError, match="collection_owner_or_identity_invalid"):
        validate_r0_prefix_rows(rows, "source")


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda r: r.pop("embeddings"), "table_denominator"),
        (lambda r: r["compiler_units"].append(r["compiler_units"][0]), "duplicate_primary_key"),
        (lambda r: r["r0_input_manifests"].clear(), "r0_denominator"),
        (lambda r: r["r0_input_manifests"][0].update(hard_gate_passed=False), "r0_denominator"),
        (lambda r: r["r0_input_manifests"][0].update(program_count=1), "r0_denominator"),
    ],
)
def test_incomplete_prefix_is_never_a_proof(change, error):
    rows = native_empty_prefix()
    change(rows)
    with pytest.raises(ValueError, match=error):
        validate_r0_prefix_rows(rows, "source")


def test_actual_vector_values_are_validated_and_hash_bound():
    rows = native_empty_prefix()
    vector = {
        "embedding_id": "v",
        "collection_id": "tm",
        "embedding": [0.0] * 4096,
        "dimension": 4096,
    }
    rows["embeddings"] = [vector]
    validate_r0_prefix_rows(rows, "source")
    digest = canonical_content_hash(rows)
    vector["embedding"][2000] = 1.0
    assert canonical_content_hash(rows) != digest
    vector["embedding"][2000] = float("nan")
    with pytest.raises(ValueError, match="vector_bytes_invalid"):
        validate_r0_prefix_rows(rows, "source")


def test_memory_index_manifest_and_physical_denominator_are_reconstructed():
    index, _, _ = make_memory_operator_index(
        run_id="source",
        snapshot_id="snapshot",
        dataset="memlens",
        selection_hash="selection",
        text_model_alias="text",
        text_model_revision="t-commit",
        multimodal_model_alias="joint",
        multimodal_model_revision="m-commit",
    )
    rows = {"memory_operator_index_manifests": [index.model_dump()], "embeddings": []}
    validate_r0_memory_vectors(rows, [], memory_raw_image_policy="allow")
    bad = copy.deepcopy(rows)
    bad["memory_operator_index_manifests"][0]["content_hash"] = "changed"
    with pytest.raises(ValueError, match="memory_index_identity_mismatch"):
        validate_r0_memory_vectors(bad, [], memory_raw_image_policy="allow")
    rows["embeddings"] = [
        {"collection_id": index.text_collection_id, "source_id": "extra", "input_hash": "hash"}
    ]
    with pytest.raises(ValueError, match="memory_vector_denominator_or_input_mismatch"):
        validate_r0_memory_vectors(rows, [], memory_raw_image_policy="allow")


def test_prefix_table_allowlist_excludes_doubter_projection_and_evaluation():
    names = {table.name for table in R0_PREFIX_TABLES}
    assert {"r0_input_manifests", "embeddings", "program_execution_steps"} <= names
    assert not any(
        word in name
        for name in names
        for word in (
            "doubter",
            "revision",
            "projection",
            "evaluation",
            "query_runtime",
        )
    )
    assert len(r0_prefix_table_contract_hash()) == 64


def test_source_artifact_bytes_are_checked_without_db_or_provider():
    class Store:
        async def get_bytes(self, key):
            assert key == "producer/output"
            return b"actual"

    repository = PostgresR0PrefixProofRepository(None, Store())
    row = {
        "artifact_id": "a",
        "object_key": "producer/output",
        "state": "ready",
        "size_bytes": 6,
        "sha256": hashlib.sha256(b"actual").hexdigest(),
    }
    assert asyncio.run(repository._artifacts([row])) == {"a": row["sha256"]}
    row["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="artifact_checksum_mismatch"):
        asyncio.run(repository._artifacts([row]))


def test_bad_prefix_proof_is_rejected_before_any_database_access():
    def sessions():
        pytest.fail("incompatible proof must not touch the database")

    repository = PostgresR0PrefixProofRepository(sessions, None)
    value = prefix_proof()
    with pytest.raises(ValueError, match="scientific_identity_mismatch"):
        asyncio.run(repository.verify_source(value, value.compatibility | {"treatment": "changed"}))
    value = value.model_copy(update={"table_contract_hash": "changed"})
    with pytest.raises(ValueError, match="table_contract_mismatch"):
        asyncio.run(repository.verify_source(value, value.compatibility))


def test_q_cost_contract_remains_separate_and_rejects_disguised_deeper_import():
    from pmmc.experiments.step8_cost_dependencies import DependencyUnproven
    from pmmc.experiments.step8_questioner_cost_dependencies import (
        validate_questioner_cost_dependencies,
    )
    from test_step8_cost_dependencies import adoption_graph

    graph = adoption_graph(
        [
            {
                "model_call_id": "q-call",
                "run_id": "q-source",
                "snapshot_id": "snapshot",
                "stage": "questioner",
                "query_id": None,
                "cache_hit": False,
                "parent_model_call_id": None,
                "metadata": {},
            }
        ]
    )
    aliases = validate_questioner_cost_dependencies(graph)
    assert list(aliases.values()) == ["q-call"]
    lineage = graph["target"]["headers"]["compiler"]["metadata"]["questioner_upstream_import"]
    assert lineage["imported_stages"] == ["questioner"]
    assert "planner0" in prefix_proof().covered_stages
    lineage["imported_stages"].append("planner0")
    with pytest.raises(DependencyUnproven, match="questioner_cost_graph_identity_mismatch"):
        validate_questioner_cost_dependencies(graph)


@pytest.mark.parametrize(
    "failure,error",
    [
        (None, None),
        ("missing_stage", "source_components_not_committed"),
        ("receipt_hash", "source_receipt_identity_invalid"),
        ("dependency", "upstream_receipt_lineage_mismatch"),
        ("proof_receipt", "source_proof_not_committed"),
        ("counts", "source_fact_hash_mismatch"),
        ("artifacts", "artifact_proof_mismatch"),
        ("r0", "source_r0_mismatch"),
        ("q", "questioner_proof_mismatch"),
        ("mutation", "source_changed_during_verification"),
        ("campaign", "source_campaign_missing_or_changed"),
    ],
)
def test_source_verifier_requires_committed_unchanged_lineage(monkeypatch, failure, error):
    from pmmc.experiments.step8_artifacts import json_bytes
    from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
    from pmmc.infrastructure.database.step8_models import (
        BenchmarkCampaignModel,
        BenchmarkComponentReceiptModel,
        BenchmarkStageJobModel,
    )

    source = campaign()
    headers = {
        "compiler": {
            "snapshot_id": SCOPE["snapshot_id"],
            "metadata": {
                "conversation_ids": [SCOPE["conversation_id"]],
            },
        },
        "manifest": {
            "dataset": SCOPE["dataset"],
            "seed": source.protocol.seed,
            "split": "development",
            "metadata": {"questioner_config": {}},
        },
    }
    rows = native_empty_prefix()
    value = prefix_proof().model_copy(
        update={
            "facts_hash": canonical_content_hash({"headers": headers, "rows": rows}),
            "row_counts": {name: len(values) for name, values in rows.items()},
        }
    )
    if failure == "counts":
        value = value.model_copy(update={"row_counts": value.row_counts | {"compiler_units": 2}})
    if failure == "artifacts":
        value = value.model_copy(update={"artifact_hashes": {"missing": "f" * 64}})
    proof_sha = hashlib.sha256(json_bytes(value.model_dump(mode="json"))).hexdigest()
    q_sha = hashlib.sha256(json_bytes(value.questioner_proof.model_dump(mode="json"))).hexdigest()
    records, jobs, receipts = [], {}, {}
    upstream = {}
    for stage in value.covered_stages:
        job = BenchmarkStageJob(
            campaign_id=source.campaign_id,
            stage=stage,
            component_key=stage,
            resource_class="cpu",
            source_fingerprint=source.protocol.source_fingerprints[stage],
            scientific_inputs={"fixture": stage},
            operation="compiler",
            parameters=SCOPE,
            dependency_ids=list(upstream),
        )
        output = {"compiler_run_id": value.source_run_id}
        artifact_hashes = []
        if stage == "questioner":
            output["questioner_import_proof_sha256"] = q_sha
            artifact_hashes.append(q_sha)
        if stage == "r0_gate":
            output["r0_prefix_import_proof_sha256"] = (
                "wrong" if failure == "proof_receipt" else proof_sha
            )
            artifact_hashes.append(proof_sha)
        receipt = ComponentReceipt(
            job_id=job.job_id,
            source_fingerprint=job.source_fingerprint,
            scientific_input_hash=job.scientific_input_hash,
            schema_revision=value.schema_revision,
            upstream_receipt_hashes=upstream,
            output=output,
            artifact_hashes=artifact_hashes,
            row_counts={"outputs": 1},
        )
        stored_job = SimpleNamespace(
            **job.model_dump(),
            payload=job.model_dump(),
            job_id=job.job_id,
            scientific_input_hash=job.scientific_input_hash,
            status="completed",
            receipt_hash=receipt.content_hash,
        )
        stored_receipt = SimpleNamespace(
            job_id=job.job_id,
            payload=receipt.model_dump(),
            receipt_hash=receipt.content_hash,
            artifact_hashes=receipt.artifact_hashes,
            component_identity=receipt.component_identity,
        )
        jobs[job.job_id], receipts[receipt.content_hash] = stored_job, stored_receipt
        records.append((stored_job, stored_receipt))
        upstream = {job.job_id: receipt.content_hash}
    if failure == "receipt_hash":
        records[0][1].receipt_hash = "wrong"
    if failure == "missing_stage":
        records.pop()
    if failure == "dependency":
        records[0][0].status = "failed"
    source_row = SimpleNamespace(
        protocol_hash="wrong" if failure == "campaign" else source.protocol.content_hash,
        payload=source.model_dump(mode="json"),
    )

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, statement):
            return SimpleNamespace(all=lambda: records)

        async def get(self, model, key):
            if model is BenchmarkCampaignModel:
                assert key == source.campaign_id
                return source_row
            if model is BenchmarkStageJobModel:
                return jobs.get(key)
            assert model is BenchmarkComponentReceiptModel
            return receipts.get(key)

    repository = PostgresR0PrefixProofRepository(Session, None)
    reads = []

    async def read(session, run_id):
        assert run_id == value.source_run_id
        reads.append(run_id)
        current = copy.deepcopy(headers)
        if failure == "mutation" and len(reads) > 1:
            current["compiler"]["changed_after_io"] = True
        return current, copy.deepcopy(rows)

    async def artifacts(values):
        assert values == []
        return {}

    async def seal(**kwargs):
        return (
            value.questioner_proof.model_copy(update={"facts_hash": "changed"})
            if (failure == "q")
            else value.questioner_proof
        )

    async def verify_r0(sessions, *, run_id):
        assert reads  # An existing R0 graph must be read before build-or-verify.
        return SimpleNamespace(content_hash="wrong" if failure == "r0" else value.r0_content_hash)

    monkeypatch.setattr(repository, "_read_prefix", read)
    monkeypatch.setattr(repository, "_artifacts", artifacts)
    monkeypatch.setattr(repository, "seal", seal)
    monkeypatch.setattr("pmmc.compiler.r0_gate.verify_r0_input", verify_r0)
    if error:
        with pytest.raises(ValueError, match=error):
            asyncio.run(repository.verify_source(value, value.compatibility))
    else:
        assert asyncio.run(repository.verify_source(value, value.compatibility)) == (headers, rows)
        assert len(reads) == 2
