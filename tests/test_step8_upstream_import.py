"""Controlled Q adoption and strict upstream identity, without a live provider."""

from __future__ import annotations

import asyncio
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pmmc.compiler.ids import (
    canonical_content_hash,
    compilation_unit_instance_id,
    stable_compiler_id,
)
from pmmc.compiler.planner_config import load_planner_treatment_config
from pmmc.experiments.step8_artifacts import CampaignArtifactStore
from pmmc.experiments.step8_backend import ProductionComponentBackend, compiler_subprotocol_hash
from pmmc.experiments.step8_campaign import build_campaign_jobs
from pmmc.experiments.step8_upstream_import import (
    QuestionerImportProof,
    assert_compatible,
    compatible_proofs,
    publish_proof,
    questioner_compatibility,
    remap_questioner_rows,
)
from pmmc.infrastructure.database.constants import RUNTIME_SCHEMA
from pmmc.infrastructure.database.step8_upstream_import import (
    QUESTIONER_TABLES,
    PostgresQuestionerImportRepository,
    QuestionerImportConflict,
    _questioner_duplicate_rows,
    _rehash_questions,
    _sort_rows,
    table_contract_hash,
)
from pmmc.infrastructure.object_store import FileSystemObjectStore
from pmmc.infrastructure.settings import Settings
from pmmc.schemas.base import utc_now
from pmmc.schemas.step8 import (
    STEP8_ALEMBIC_REVISION,
    BenchmarkCampaignManifest,
    CampaignPhase,
    Step8BenchmarkConfig,
)
from test_step8_campaign import DATASETS, choose, protocol

ROOT = Path(__file__).resolve().parents[1]
SCOPE = {
    "snapshot_id": "snapshot-memlens",
    "conversation_id": "conversation",
    "dataset": "memlens",
    "treatment": "main",
}


def campaign():
    return BenchmarkCampaignManifest(
        campaign_id="source-campaign",
        phase="development",
        protocol=protocol(),
        partition_id="partition",
        snapshot_ids={d: "snapshot-" + d.value for d in DATASETS},
        matrix_hash="matrix",
        created_at=utc_now(),
    )


def proof():
    expected = questioner_compatibility(campaign(), SCOPE)
    return QuestionerImportProof(
        source_campaign_id="source-campaign",
        source_protocol_hash=campaign().protocol.content_hash,
        source_run_id="source-run",
        compatibility=expected,
        compatibility_hash=canonical_content_hash(expected),
        schema_revision=STEP8_ALEMBIC_REVISION,
        table_contract_hash=table_contract_hash(),
        facts_hash="a" * 64,
        row_counts={"compiler_units": 1},
        artifact_hashes={},
        unit_count=1,
        eligible_count=0,
    )


def test_fixed_and_main_share_q_identity_but_not_program_identity():
    main = campaign()
    fixed = {**SCOPE, "treatment": "fixed_planner"}
    assert questioner_compatibility(main, SCOPE) == questioner_compatibility(main, fixed)
    assert compiler_subprotocol_hash(main, SCOPE) != compiler_subprotocol_hash(main, fixed)
    updated = main.protocol.model_copy(
        update={
            "config_hashes": {
                **main.protocol.config_hashes,
                "planner_config": "f" * 64,
                "doubter_config": "f" * 64,
            }
        }
    )
    other = main.model_copy(update={"protocol": updated, "campaign_id": "new-campaign"})
    assert questioner_compatibility(other, SCOPE) == questioner_compatibility(main, SCOPE)
    assert compiler_subprotocol_hash(other, SCOPE) != compiler_subprotocol_hash(main, SCOPE)


@pytest.mark.parametrize(
    "change",
    [
        {"snapshot_id": "changed"},
        {"conversation_id": "changed"},
        {"dataset": "memgallery"},
        {"treatment": "no_raw_image"},
    ],
)
def test_changed_scope_cannot_import(change):
    with pytest.raises(ValueError, match="scientific_identity_mismatch"):
        assert_compatible(proof(), questioner_compatibility(campaign(), SCOPE | change))


@pytest.mark.parametrize("key", ["questioner_config", "foundation_config"])
def test_scientific_config_changes_invalidate_q(key):
    value = campaign()
    changed = value.protocol.model_copy(
        update={"config_hashes": value.protocol.config_hashes | {key: "f" * 64}}
    )
    with pytest.raises(ValueError, match="scientific_identity_mismatch"):
        assert_compatible(
            proof(), questioner_compatibility(value.model_copy(update={"protocol": changed}), SCOPE)
        )


def test_diagnostic_namespace_is_distinct_and_cannot_feed_main():
    main = campaign()
    diagnostic = main.model_copy(update={"diagnostic_only": True})
    second = diagnostic.model_copy(update={"campaign_id": "other-diagnostic"})
    assert compiler_subprotocol_hash(diagnostic, SCOPE) != compiler_subprotocol_hash(second, SCOPE)
    assert compiler_subprotocol_hash(main, SCOPE) == compiler_subprotocol_hash(
        main.model_copy(update={"campaign_id": "other-main"}), SCOPE
    )
    with pytest.raises(ValueError, match="scientific_identity_mismatch"):
        assert_compatible(proof(), questioner_compatibility(diagnostic, SCOPE))


def test_proof_registry_verifies_bytes_and_is_immutable(tmp_path):
    store, value = CampaignArtifactStore(tmp_path), proof()
    digest = publish_proof(store, value)
    assert publish_proof(store, value) == digest
    assert compatible_proofs(store, value.compatibility, exclude_run_id="target") == [value]
    assert compatible_proofs(store, value.compatibility, exclude_run_id="source-run") == []
    with pytest.raises(ValueError, match="immutable artifact"):
        publish_proof(store, value.model_copy(update={"facts_hash": "b" * 64}))
    store.object_path(digest).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        compatible_proofs(store, value.compatibility, exclude_run_id="target")


def test_q_table_allowlist_does_not_import_plans_doubter_evaluation_or_r0():
    names = {table.name for table in QUESTIONER_TABLES}
    assert {
        "questioner_answer_claims",
        "questioner_generations",
        "questioner_candidate_verifications",
        "qbank_candidate_families",
        "qbank_candidate_family_members",
    } <= names
    assert not any(
        word in name for name in names for word in ("program", "doubter", "r0", "evaluation")
    )
    assert all(table.schema == RUNTIME_SCHEMA for table in QUESTIONER_TABLES)


def _semantic_link(run_id):
    method = "qbank-dedupe-v3:semantic-nearest"
    return {
        "run_id": run_id,
        "duplicate_qbank_id": "second",
        "primary_qbank_id": "first",
        "method": method,
        "duplicate_link_id": stable_compiler_id(
            "qbank-semantic-neighbor", [run_id, "second", "first", method]
        ),
        "hard_duplicate": False,
        "score": {"score_type": "cosine", "metadata": {"collection_id": "text-index"}},
        "metadata": {"diagnostic_only": True},
    }


@pytest.mark.parametrize(
    "change",
    [
        {"hard_duplicate": True},
        {"metadata": {}},
        {"method": "qbank-dedupe-v3:jaccard"},
        {"score": None},
        {"score": {"score_type": "cosine", "metadata": {}}},
        {"score": {"score_type": "jaccard", "metadata": {"collection_id": "index"}}},
    ],
)
def test_questioner_scope_keeps_real_or_unrecognized_duplicate_links(change, capsys):
    native = _semantic_link("source")
    other = native | change
    rows = [native, other]
    before = copy.deepcopy(rows)
    assert _questioner_duplicate_rows(rows) == [other]
    assert rows == before
    assert "index_diagnostics=1 database_rows_deleted=0" in capsys.readouterr().out


def test_r0_remapped_diagnostic_stays_outside_questioner_ownership():
    source = {"qbank_duplicate_links": [_semantic_link("source")]}
    projected, _ = remap_questioner_rows(
        source,
        primary_keys={"qbank_duplicate_links": ["duplicate_link_id"]},
        source_run_id="source",
        target_run_id="target",
        proof_hash="r0-proof",
    )
    row = projected["qbank_duplicate_links"][0]
    assert row["duplicate_link_id"] != source["qbank_duplicate_links"][0]["duplicate_link_id"]
    assert _questioner_duplicate_rows([row]) == []
    assert len(projected["qbank_duplicate_links"]) == 1


@pytest.mark.parametrize("damage", [None, "source", "target", "header", "dedupe", "gate"])
def test_adoption_consumer_ignores_only_post_seal_index_diagnostics(damage):
    compatibility = {"contract": "fixture"}
    headers = {"source": {"memory": "original"}, "target": {"memory": "original"}}
    rows = {
        run: {
            "run_artifacts": [],
            "qbank_questions": [{"qbank_id": run, "question": "unchanged"}],
            "questioner_candidate_gate_results": [{"status": "eligible_for_planning"}],
            "qbank_duplicate_links": [],
        }
        for run in headers
    }
    lineage = {
        "version": proof().version,
        "compatibility_hash": canonical_content_hash(compatibility),
        "imported_stages": ["questioner"],
        "source_run_id": "source",
        "target_rows_hash": canonical_content_hash(rows["target"]),
        "source_facts_hash": canonical_content_hash(
            {"headers": headers["source"], "rows": rows["source"]}
        ),
    }
    # Reproduce main text index followed by fixed text and multimodal consumption.
    for run in rows:
        rows[run]["qbank_duplicate_links"].append(_semantic_link(run))
    if damage in {"source", "target"}:
        rows[damage]["qbank_questions"][0]["question"] = "changed"
    elif damage == "header":
        headers["source"]["memory"] = "changed"
    elif damage == "dedupe":
        rows["source"]["qbank_duplicate_links"][0]["hard_duplicate"] = True
    elif damage == "gate":
        rows["source"]["questioner_candidate_gate_results"][0]["status"] = "rejected"

    session = AsyncMock()
    session.get.return_value = SimpleNamespace(
        metadata_json={"questioner_upstream_import": lineage}
    )
    session.__aenter__.return_value = session
    repository = PostgresQuestionerImportRepository(lambda: session, object())

    async def read(_, run_id):
        result = copy.deepcopy(rows[run_id])
        result["qbank_duplicate_links"] = _questioner_duplicate_rows(
            result["qbank_duplicate_links"]
        )
        return headers[run_id], result

    repository._read = read
    repository._scope = lambda *_: None
    repository._artifacts = AsyncMock(return_value={})
    if damage is None:
        assert asyncio.run(repository.verify_adopted("target", compatibility)) is True
        repository._artifacts.assert_awaited_once_with([])
    else:
        code = "fact" if damage == "target" else "source"
        with pytest.raises(QuestionerImportConflict) as error:
            asyncio.run(repository.verify_adopted("target", compatibility))
        assert error.value.error_code == f"questioner_import_consumer_{code}_conflict"
        repository._artifacts.assert_not_awaited()


def test_remap_preserves_native_science_and_complete_ownership_lineage():
    rows = {
        "compiler_units": [
            {"compiler_unit_id": "unit", "run_id": "source", "input_hash": "hash", "metadata": {}}
        ],
        "model_calls": [
            {
                "model_call_id": "call",
                "run_id": "source",
                "compiler_unit_id": "unit",
                "cache_hit": False,
                "output_asset_id": "artifact",
                "output_tokens": 30,
                "metadata": {},
            }
        ],
        "run_artifacts": [
            {
                "artifact_id": "artifact",
                "run_id": "source",
                "object_key": "source/provider-response.json",
                "metadata": {},
            }
        ],
        "qbank_questions": [
            {
                "qbank_id": "question",
                "run_id": "source",
                "compiler_unit_id": "unit",
                "question": "Which piano did I buy?",
                "expected_answer": "Yamaha P45",
                "source_memory_ids": ["canonical-memory"],
                "metadata": {"model_call_id": "call"},
            }
        ],
    }
    before = copy.deepcopy(rows)
    keys = {name: [next(iter(values[0]))] for name, values in rows.items()}
    target, ids = remap_questioner_rows(
        rows, primary_keys=keys, source_run_id="source", target_run_id="target", proof_hash="proof"
    )
    assert rows == before
    assert ids["unit"] == compilation_unit_instance_id(run_id="target", input_hash="hash")
    assert ids["question"] != "question"
    assert target["qbank_questions"][0]["expected_answer"] == "Yamaha P45"
    assert target["qbank_questions"][0]["source_memory_ids"] == ["canonical-memory"]
    assert target["qbank_questions"][0]["metadata"]["model_call_id"] == ids["call"]
    assert target["model_calls"][0]["output_asset_id"] == ids["artifact"]
    assert target["model_calls"][0]["cache_hit"] is True
    assert target["model_calls"][0]["output_tokens"] == 30
    assert target["run_artifacts"][0]["object_key"] == "source/provider-response.json"
    for name, values in target.items():
        for index, row in enumerate(values):
            assert row["run_id"] == "target"
            assert row["metadata"]["upstream_import"]["source_row_hash"] == canonical_content_hash(
                rows[name][index]
            )
    assert remap_questioner_rows(
        rows, primary_keys=keys, source_run_id="source", target_run_id="target", proof_hash="proof"
    ) == (target, ids)


def test_model_call_parent_must_be_closed_and_topologically_ordered():
    table = next(t for t in QUESTIONER_TABLES if t.name == "model_calls")
    rows = [
        {"model_call_id": "a-child", "parent_model_call_id": "z-parent"},
        {"model_call_id": "z-parent", "parent_model_call_id": None},
    ]
    assert [r["model_call_id"] for r in _sort_rows(table, rows)] == ["z-parent", "a-child"]
    with pytest.raises(ValueError, match="parent_not_closed"):
        _sort_rows(table, rows[:1])


def test_native_text_equal_to_owned_id_is_not_rewritten():
    rows = {
        "qbank_questions": [
            {
                "qbank_id": "q",
                "run_id": "source",
                "canonical_question": "source",
                "expected_answer": "q",
                "metadata": {},
            }
        ]
    }
    copied, _ = remap_questioner_rows(
        rows,
        primary_keys={"qbank_questions": ["qbank_id"]},
        source_run_id="source",
        target_run_id="target",
        proof_hash="proof",
    )
    assert copied["qbank_questions"][0]["canonical_question"] == "source"
    assert copied["qbank_questions"][0]["expected_answer"] == "q"


def test_adopted_candidate_hash_obeys_normal_repository_contract():
    from pmmc.compiler.ids import question_content_hash
    from pmmc.infrastructure.database.compiler_mappers import question_from_rows, question_to_rows
    from test_step8_compiler_revision import candidate

    value = candidate().model_copy(update={"candidate_content_hash": "qh"})
    values = question_to_rows(value)
    rows = {
        "qbank_questions": values["question"],
        "qbank_paraphrases": values["paraphrases"],
        "qbank_evidence_refs": values["source"],
        "questioner_candidate_gate_results": [
            {
                "gate_result_id": "gate",
                "qbank_id": value.qbank_id,
                "run_id": "run",
                "content_hash": "qh",
                "metadata": {},
            }
        ],
    }
    copied, _ = remap_questioner_rows(
        rows,
        primary_keys={
            "qbank_questions": ["qbank_id"],
            "qbank_paraphrases": ["paraphrase_id"],
            "qbank_evidence_refs": ["evidence_ref_id"],
            "questioner_candidate_gate_results": ["gate_result_id"],
        },
        source_run_id="run",
        target_run_id="adopted",
        proof_hash="proof",
    )
    _rehash_questions(copied)
    row = copied["qbank_questions"][0]
    adopted = question_from_rows(row, copied["qbank_paraphrases"], copied["qbank_evidence_refs"])
    assert adopted.question_hash == question_content_hash(adopted)
    assert row["metadata"]["upstream_import"]["source_question_hash"] == "qh"
    gate = copied["questioner_candidate_gate_results"][0]
    assert gate["content_hash"] == row["candidate_content_hash"] == row["question_hash"]
    assert gate["content_hash"] != "qh"
    assert gate["metadata"]["upstream_import"]["source_row_hash"] == canonical_content_hash(
        rows["questioner_candidate_gate_results"][0]
    )
    assert rows["questioner_candidate_gate_results"][0]["content_hash"] == "qh"
    gate["qbank_id"] = "missing"
    with pytest.raises(ValueError, match="gate_candidate_hash_missing"):
        _rehash_questions(copied)


def test_fixed_dag_has_no_independent_questioner_or_units():
    selection = choose(CampaignPhase.DEVELOPMENT, "regression")
    jobs = build_campaign_jobs(
        manifest=campaign(),
        selection=selection,
        conversations={d: {"conversation": selection.query_ids[d]} for d in DATASETS},
        project_root=ROOT,
    )
    for dataset in DATASETS:
        selected = [j for j in jobs if j.parameters.get("dataset") == dataset.value]
        main = next(
            j for j in selected if j.stage == "questioner" and j.parameters["treatment"] == "main"
        )
        fixed = [j for j in selected if j.parameters.get("treatment") == "fixed_planner"]
        adoption = next(j for j in fixed if j.stage == "questioner")
        assert adoption.operation == "adopt_questioner"
        assert adoption.resource_class.value == "cpu"
        assert adoption.dependency_ids == [main.job_id]
        assert not any(j.operation == "units" for j in fixed)
        assert not any(j.operation == "compiler" and j.stage == "questioner" for j in fixed)
        assert next(j for j in fixed if j.stage == "planner0").dependency_ids == [adoption.job_id]


def test_backend_fixed_uses_frozen_shared_strategy_helper():
    backend = object.__new__(ProductionComponentBackend)
    backend.settings = Settings(project_root=ROOT, _env_file=None)
    backend.config = Step8BenchmarkConfig(
        partition_dir=Path("partition"),
        exposure_manifest=Path("exposure.json"),
        regression_manifest=Path("regression.json"),
        model_lock=Path("model.lock.json"),
    )
    _, fixed, _ = backend._compiler_configs(SCOPE | {"treatment": "fixed_planner"})
    assert fixed == load_planner_treatment_config(
        ROOT / backend.config.planner_config, treatment="fixed_planner"
    )


def test_artifact_proof_reads_actual_ready_bytes(tmp_path):
    store = FileSystemObjectStore(tmp_path, "runtime")
    repository = PostgresQuestionerImportRepository(None, store)
    payload = b'{"native":true}'
    row = {
        "state": "ready",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "artifact_id": "artifact",
        "object_key": "native.json",
    }

    async def run():
        await store.put_bytes(payload, "native.json", content_type="application/json")
        assert await repository._artifacts([row]) == {"artifact": row["sha256"]}
        with pytest.raises(ValueError, match="checksum_mismatch"):
            await repository._artifacts([row | {"sha256": "a" * 64}])
        with pytest.raises(ValueError, match="not_ready"):
            await repository._artifacts([row | {"state": "pending"}])

    asyncio.run(run())
