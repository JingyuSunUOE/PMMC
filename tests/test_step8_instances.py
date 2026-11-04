from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.memory_index import make_memory_operator_index
from pmmc.experiments import step8_instances as instances
from pmmc.experiments.step8_artifacts import CampaignArtifactStore
from pmmc.experiments.step8_backend import compiler_subprotocol_hash
from pmmc.schemas import DataScope
from pmmc.schemas.base import utc_now
from pmmc.schemas.compiler import CompilationUnit, CompilationUnitMemory, CompilerRunSpec
from pmmc.schemas.doubter import (
    QBankRoundProjectionManifest,
    QBankRoundProjectionMember,
    R0InputManifest,
    RoutingProjectionManifest,
    RoutingProjectionMember,
)
from pmmc.schemas.experiment import RunManifest
from pmmc.schemas.persistence import EmbeddingCollection, EmbeddingCollectionStatus, RunState
from pmmc.schemas.questioner import QBankIndexManifest
from pmmc.schemas.step8 import (
    BenchmarkCampaignManifest,
    CompiledQBankInstanceManifest,
    ScientificProtocol,
)
from step8_fixture_approval import saved_protocol_freeze
from step8_model_fixtures import model_entry_hashes

PRIVATE = "PRIVATE_QUERY_GOLD_SECRET_SENTINEL"


def protocol():
    return ScientificProtocol(
        model_lock_hash="a" * 64,
        model_entry_hashes=model_entry_hashes(),
        partition_hash="b" * 64,
        exposure_manifest_hash="c" * 64,
        source_fingerprints={
            key: "d" * 64
            for key in (
                "unit_prepare",
                "questioner",
                "planner0",
                "text_index",
                "multimodal_index",
                "dry_run0",
                "r0_gate",
                "doubter",
                "projection",
            )
        },
        config_hashes={
            key: "e" * 64
            for key in (
                "questioner_config",
                "planner_config",
                "doubter_config",
                "foundation_config",
            )
        },
    )


def _backend_hash(value, scope, diagnostic_namespace=None):
    campaign = BenchmarkCampaignManifest(
        campaign_id=diagnostic_namespace or "campaign",
        diagnostic_only=diagnostic_namespace is not None,
        phase="development",
        protocol=value,
        partition_id="partition",
        snapshot_ids={"memlens": "snapshot", "memgallery": "gallery"},
        matrix_hash="matrix",
        created_at=utc_now(),
    )
    return compiler_subprotocol_hash(campaign, scope)


def _compiler(run_id, subhash, scope):
    return CompilerRunSpec(
        run_id=run_id,
        snapshot_id=scope["snapshot_id"],
        compiler_schema_version="0.9.0",
        target_stage="verified_qbank",
        operator_set_version="pmmc-operators-v7",
        operator_registry_hash="operator-hash",
        compiler_config_hash="compiler-config",
        questioner_config_hash="questioner-config",
        questioner_model_alias="compiler-qwen3.5-9b",
        questioner_model_revision="locked",
        questioner_prompt_version="pmmc-questioner-v3.6",
        questioner_prompt_hash="questioner-prompt",
        questioner_renderer_version="renderer",
        questioner_parser_version="parser",
        unit_builder_version="builder",
        candidate_gate_version="gate",
        dedupe_version="dedupe",
        qbank_matcher_version="matcher",
        questioner_freeze_manifest_hash=subhash,
        planner_config_hash="planner-config",
        planner_model_alias="compiler-qwen3.5-9b",
        planner_model_revision="locked",
        planner_prompt_version="pmmc-planner-v1.5",
        planner_prompt_hash="planner-prompt",
        planner_renderer_version="renderer",
        planner_parser_version="parser",
        planner_schema_version="0.4.0",
        program_policy_version="policy",
        dry_run_policy_version="dry-run",
        memory_index_representation_version="memory-operator-index-v1",
        memory_text_model_alias="text-model",
        memory_text_model_revision="locked-text",
        memory_multimodal_model_alias="mm-model",
        memory_multimodal_model_revision="locked-mm",
        doubter_config_hash="doubter-config",
        doubter_model_alias="compiler-qwen3.5-9b",
        doubter_model_revision="locked",
        doubter_prompt_version="pmmc-doubter-v1.3",
        doubter_prompt_hash="doubter-prompt",
        doubter_renderer_version="renderer",
        doubter_parser_version="parser",
        doubter_schema_version="0.4.0",
        r0_input_policy_version="r0-input-gate-v4",
        routing_projection_version="verified-qbank-projection-v4",
        active_round=2,
        max_round=3,
        unit_count=1,
        metadata={
            "conversation_ids": [scope["conversation_id"]],
            "selection_hash": "selection",
            "questioner_identity_kind": "step8_method_protocol",
            "compiler_subprotocol_hash": subhash,
            "private": PRIVATE,
        },
    )


def _round(r0, round_id):
    identity = dict(
        run_id=r0.run_id,
        snapshot_id=r0.snapshot_id,
        round_id=round_id,
        source_r0_manifest_id=r0.r0_manifest_id,
        r0_facts_hash=r0.facts_hash,
    )
    projection_id = stable_compiler_id("qbank-round-projection", identity)
    members = []
    for qbank_id, route in (("q1", "answerable"), ("q2", "control")):
        accepted = (
            round_id == 2
            or (round_id == 0 and qbank_id == "q1")
            or (round_id == 1 and qbank_id == "q2")
        )
        terminal = {
            "projection_terminal": True,
            "projection_round": round_id,
            "r0_uses_doubter_facts": False if round_id == 0 else None,
        }
        payload = dict(
            projection_id=projection_id,
            run_id=r0.run_id,
            qbank_id=qbank_id,
            route_class=route,
            outcome="accepted" if accepted else "rejected",
            effective_plan_id=f"plan-{qbank_id}-r{round_id}" if accepted else None,
            effective_feedback_id=None,
            passed_at_round=round_id if accepted else None,
            reason="accepted" if accepted else "rejected",
            terminal_state=terminal,
        )
        members.append(
            QBankRoundProjectionMember(
                **{
                    key: value
                    for key, value in payload.items()
                    if key not in {"reason", "terminal_state"}
                },
                projection_member_id=stable_compiler_id("qbank-round-projection-member", payload),
                content_hash=canonical_content_hash(payload),
                metadata={"reason_code": payload["reason"], **terminal},
            )
        )
    count = sum(item.outcome == "accepted" for item in members)
    payload = dict(
        run_id=r0.run_id,
        snapshot_id=r0.snapshot_id,
        round_id=round_id,
        source_r0_manifest_id=r0.r0_manifest_id,
        projection_id=projection_id,
        member_count=2,
        accepted_count=count,
        rejected_count=2 - count,
        pending_count=0,
        members_hash=canonical_content_hash([item.model_dump(mode="json") for item in members]),
    )
    manifest = QBankRoundProjectionManifest(
        **payload,
        schema_version="0.4.0",
        content_hash=canonical_content_hash(payload),
        metadata={"r0_facts_hash": r0.facts_hash},
    )
    return manifest, members


def _routing(r0, qbank, active):
    projection, sources = active
    decision_hash = canonical_content_hash("decisions")
    routing_id = stable_compiler_id(
        "routing-projection", [r0.run_id, projection.projection_id, decision_hash]
    )
    members = [
        RoutingProjectionMember(
            routing_member_id=stable_compiler_id(
                "routing-projection-member", [routing_id, source.qbank_id, ordinal]
            ),
            routing_projection_id=routing_id,
            run_id=r0.run_id,
            qbank_id=source.qbank_id,
            plan_id=source.effective_plan_id,
            route_class=source.route_class,
            source_projection_member_id=source.projection_member_id,
            ordinal=ordinal,
            content_hash=canonical_content_hash(
                [
                    routing_id,
                    source.qbank_id,
                    source.effective_plan_id,
                    source.route_class.value,
                    ordinal,
                ]
            ),
        )
        for ordinal, source in enumerate(sources)
    ]
    payload = dict(
        routing_projection_id=routing_id,
        run_id=r0.run_id,
        snapshot_id=r0.snapshot_id,
        active_round=2,
        max_round=3,
        source_projection_id=projection.projection_id,
        decision_hash=decision_hash,
        lexical_collection_id=stable_compiler_id(
            "qbank-lexical-projection", [r0.run_id, qbank.qbank_index_id, projection.projection_id]
        ),
        text_collection_id=qbank.text_question_collection_id,
        multimodal_collection_id=qbank.multimodal_question_collection_id,
        member_count=2,
        answerable_count=1,
        control_count=1,
        members_hash=canonical_content_hash([item.model_dump(mode="json") for item in members]),
    )
    return RoutingProjectionManifest(
        **payload, schema_version="0.4.0", content_hash=canonical_content_hash(payload)
    ), members


class RuntimeFacts:
    """Typed persisted facts with a deliberately runtime-only repository surface."""

    def __init__(self, treatment="main", seed=0):
        self.protocol = protocol().model_copy(update={"seed": seed})
        self.scope = {
            "dataset": "memlens",
            "snapshot_id": "snapshot",
            "conversation_id": "conversation-" + PRIVATE,
            "treatment": treatment,
        }
        self.subhash = _backend_hash(self.protocol, self.scope)
        self.run_id = "step8-compiler-" + self.subhash[:24]
        self.compiler = _compiler(self.run_id, self.subhash, self.scope)
        self.run = RunManifest(
            run_id=self.run_id,
            experiment_name="compiler",
            dataset="memlens",
            method="pmmc_compiler",
            compiler_schema_version="0.9.0",
            operator_set_version="pmmc-operators-v7",
            operator_registry_hash="operator-hash",
            compiler_config_hash="compiler-config",
            dataset_snapshot_ids=["snapshot"],
            selection_hash="selection",
            seed=seed,
            metadata=dict(self.compiler.metadata),
        )
        self.state = RunState(run_id=self.run_id, status="completed", completed_at=utc_now())
        self.units = [
            CompilationUnit(
                compiler_unit_id="unit",
                run_id=self.run_id,
                snapshot_id="snapshot",
                conversation_id=self.scope["conversation_id"],
                available_after_session="session",
                ordinal=0,
                builder_version="builder",
                input_hash="input-hash",
                memories=[
                    CompilationUnitMemory(
                        snapshot_id="snapshot",
                        conversation_id=self.scope["conversation_id"],
                        memory_id="memory",
                        ordinal=0,
                        role="anchor",
                    )
                ],
            )
        ]
        channels = {
            key: key + "-collection"
            for key in (
                "text_question",
                "text_paraphrase",
                "multimodal_question",
                "multimodal_paraphrase",
            )
        }
        recipe = dict(
            compiler_run_id=self.run_id,
            snapshot_id="snapshot",
            collections=channels,
            eligible_question_count=2,
            paraphrase_count=2,
            gate_version="gate",
            matcher_version="matcher",
            rrf_constant=60,
            per_channel_top_k=50,
        )
        self.qbank = QBankIndexManifest(
            **{key: value for key, value in recipe.items() if key != "collections"},
            **{key + "_collection_id": value for key, value in channels.items()},
            qbank_index_id=stable_compiler_id("qbank-index", recipe),
            content_hash=canonical_content_hash(recipe),
            status="ready",
        )
        self.collections = {
            value: EmbeddingCollection(
                collection_id=value,
                snapshot_id="snapshot",
                run_id=self.run_id,
                source_type="qbank_question" if "question" in key else "qbank_paraphrase",
                model_alias="embedding",
                model_revision="locked",
                dimension=4096,
                distance_metric="cosine",
                normalized=True,
                normalization_method="l2",
                preprocessing_hash="preprocessing",
                status="ready",
                metadata={"private": PRIVATE},
            )
            for key, value in channels.items()
        }
        memory, text, visual = make_memory_operator_index(
            run_id=self.run_id,
            snapshot_id="snapshot",
            dataset="memlens",
            selection_hash="selection",
            text_model_alias="text-model",
            text_model_revision="locked-text",
            multimodal_model_alias="mm-model",
            multimodal_model_revision="locked-mm",
            memory_raw_image_policy="forbid" if treatment == "no_raw_image" else "allow",
        )
        self.memory = memory.model_copy(update={"status": "ready"})
        self.collections.update(
            {
                item.collection_id: item.model_copy(
                    update={"status": EmbeddingCollectionStatus.READY}
                )
                for item in (text, visual)
            }
        )
        questioner = {
            "config": "questioner-config",
            "prompt": "questioner-prompt",
            "renderer": "renderer",
            "parser": "parser",
            "gate": "gate",
            "dedupe": "dedupe",
        }
        planner = {
            "config": "planner-config",
            "prompt": "planner-prompt",
            "renderer": "renderer",
            "parser": "parser",
            "policy": "policy",
        }
        payload = dict(
            run_id=self.run_id,
            snapshot_id="snapshot",
            candidate_count=2,
            answerable_candidate_count=1,
            control_candidate_count=1,
            program_count=2,
            execution_count=2,
            terminal_outcome_count=2,
            questioner_identity_hash=canonical_content_hash(questioner),
            planner_identity_hash=canonical_content_hash(planner),
            operator_registry_hash="operator-hash",
            memory_index_hash=memory.content_hash,
            query_vector_hash=canonical_content_hash("query-vectors"),
            facts_hash=canonical_content_hash("r0-facts"),
            artifact_hash=canonical_content_hash([self.qbank.content_hash, memory.content_hash]),
            hard_gate_passed=True,
            issue_codes=[],
            protocol_integrity_issues=[],
            candidate_scientific_rejections={},
            coverage_zero_units=0,
            downstream_viability=True,
        )
        self.r0 = R0InputManifest(
            **payload,
            schema_version="0.4.0",
            r0_manifest_id=stable_compiler_id("r0-input-manifest", payload),
            content_hash=canonical_content_hash(payload),
            metadata={"private": PRIVATE, "r0_input_policy_version": "r0-input-gate-v4"},
        )
        self.rounds = {round_id: _round(self.r0, round_id) for round_id in range(4)}
        self.routing = _routing(self.r0, self.qbank, self.rounds[2])
        self.session = Mock(connection=AsyncMock(), execute=AsyncMock())
        self.compiler_runs = Mock(get=AsyncMock(side_effect=lambda _: self.compiler))
        self.compiler_units = Mock(list_for_run=AsyncMock(side_effect=lambda _: self.units))
        self.runs = Mock(
            get_manifest=AsyncMock(side_effect=lambda _: self.run),
            get_state=AsyncMock(side_effect=lambda _: self.state),
        )
        self.qbank_indexes = Mock(get_for_run=AsyncMock(side_effect=lambda _: self.qbank))
        self.memory_operator_indexes = Mock(
            get_for_run=AsyncMock(side_effect=lambda _: self.memory)
        )
        self.doubter = Mock(
            get_r0_manifest=AsyncMock(side_effect=lambda _: self.r0),
            get_round_projection=AsyncMock(
                side_effect=lambda _, round_id: self.rounds.get(round_id)
            ),
            get_routing_projection=AsyncMock(side_effect=lambda _: self.routing),
        )
        self.embeddings = Mock(get_collection=AsyncMock(side_effect=self.collections.get))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def install(self, monkeypatch):
        def factory(sessions, scope):
            assert sessions is None
            assert scope is DataScope.RUNTIME
            return self

        monkeypatch.setattr(instances, "ExperimentUnitOfWork", factory)

    def build(self, **kwargs):
        arguments = dict(
            protocol=self.protocol,
            compiler_run_id=self.run_id,
            compiler_subprotocol_hash=self.subhash,
            scope=self.scope,
        )
        return asyncio.run(
            instances.build_compiled_qbank_instance_manifest(None, **(arguments | kwargs))
        )

    def freeze(self):
        return saved_protocol_freeze(self.protocol, self.approval_root)

    def verify(self, manifest, **kwargs):
        arguments = dict(
            manifest=manifest,
            protocol=self.protocol,
            compiler_run_id=self.run_id,
            compiler_subprotocol_hash=self.subhash,
            scope=self.scope,
            expected_manifest_hash=manifest.content_hash,
            freeze=self.freeze(),
        )
        return asyncio.run(
            instances.verify_compiled_qbank_instance_manifest(None, **(arguments | kwargs))
        )


@pytest.fixture
def facts(monkeypatch, tmp_path):
    result = RuntimeFacts()
    result.approval_root = tmp_path / "offline-approval"
    result.install(monkeypatch)
    return result


def test_complete_compiler09_builds_stable_immutable_manifest(facts, tmp_path, capsys):
    assert CompilerRunSpec.model_validate(facts.compiler.model_dump()) == facts.compiler
    manifest = facts.build()
    assert isinstance(manifest, CompiledQBankInstanceManifest)
    assert manifest.content_hash == facts.build().content_hash
    assert set(manifest.round_membership_hashes) == {0, 1, 2, 3}
    assert len(set(manifest.round_membership_hashes.values())) == 4
    assert len(manifest.collection_hashes) == 7
    assert manifest.r0_manifest_hash == facts.r0.content_hash
    assert facts.qbank.content_hash in manifest.artifact_hashes
    assert facts.memory.content_hash in manifest.artifact_hashes
    assert all(value[0].content_hash in manifest.artifact_hashes for value in facts.rounds.values())
    with pytest.raises(ValidationError):
        manifest.compiler_run_id = "other"
    with pytest.raises(TypeError):
        manifest.round_membership_hashes[0] = "other"
    with pytest.raises(TypeError):
        manifest.collection_hashes.clear()
    with pytest.raises(TypeError):
        manifest.artifact_hashes.append("other")
    store = CampaignArtifactStore(tmp_path)
    artifact_id = store.put_json(manifest.model_dump(mode="json"))
    restored = CompiledQBankInstanceManifest.model_validate(store.load_json(artifact_id))
    assert facts.verify(restored).content_hash == manifest.content_hash
    assert PRIVATE not in store.verify(artifact_id).read_text()
    assert PRIVATE not in manifest.model_dump_json()
    assert not capsys.readouterr().out
    assert str(facts.session.execute.call_args.args[0]) == "SET TRANSACTION READ ONLY"
    assert facts.session.connection.call_args.kwargs == {
        "execution_options": {"isolation_level": "REPEATABLE READ"}
    }


@pytest.mark.parametrize(
    "field", ["doubter_config_hash", "doubter_model_revision", "doubter_prompt_hash"]
)
def test_missing_doubter_identity_rejected_by_schema_and_builder(facts, field):
    broken = facts.compiler.model_dump()
    broken[field] = None
    with pytest.raises(ValidationError, match="Doubter identity"):
        CompilerRunSpec.model_validate(broken)
    facts.compiler = facts.compiler.model_copy(update={field: None})
    with pytest.raises(instances.InstanceManifestError, match="runtime_fact_missing_or_invalid"):
        facts.build()


def test_approval_is_protocol_not_development_run(facts):
    manifest = facts.build()
    assert facts.freeze().development_receipt_hash != manifest.compiler_run_id
    assert facts.verify(manifest).content_hash == manifest.content_hash


def test_approval_fixture_binds_saved_cohort_evidence(facts):
    freeze = facts.freeze()
    evidence = freeze.threshold_approval
    assert evidence is not None
    root = facts.approval_root
    store = CampaignArtifactStore(root)
    assert evidence.provisional_protocol.match_threshold != freeze.protocol.match_threshold
    assert evidence.source_campaign_manifest_hash == canonical_content_hash(
        json.loads((root / "campaign.json").read_bytes())
    )
    assert store.load_json(evidence.audit_inventory_sha256)["scientific_assignments"] == 0
    for cohort in ("regression", "confirmation"):
        assert evidence.cohort_selection_hashes[cohort] == canonical_content_hash(
            json.loads((root / cohort / "selection.json").read_bytes())
        )
        assert evidence.cohort_receipt_hashes[cohort] == canonical_content_hash(
            json.loads((root / cohort / "receipt.json").read_bytes())
        )
        assert store.load_json(evidence.cohort_proof_sha256[cohort])["selected_policy"] == (
            evidence.selected_policy
        )
    assert PRIVATE not in (root / "threshold-approval.json").read_text()


@pytest.mark.parametrize("change", ["missing", "partial_policy", "receipt_mismatch"])
def test_approval_fixture_cannot_bypass_threshold_evidence_gate(facts, change):
    freeze = facts.freeze()
    payload = freeze.model_dump(mode="json")
    if change == "missing":
        payload["threshold_approval"] = None
    elif change == "partial_policy":
        payload["threshold_approval"]["selected_policy"].pop("gap_threshold")
    else:
        payload["development_receipt_hash"] = "0" * 64
    with pytest.raises(ValidationError):
        type(freeze).model_validate(payload)


@pytest.mark.parametrize(
    "change,code",
    [
        ({"freeze": None}, "approval_missing"),
        ({"decision": "rejected"}, "not_approved"),
        ({"protocol_integrity_passed": False}, "not_approved"),
        ({"scientific_viability_passed": False}, "not_approved"),
        ({"exposure_complete": False}, "not_approved"),
    ],
)
def test_paper_requires_explicit_complete_approval(facts, change, code):
    manifest = facts.build()
    freeze = None if "freeze" in change else facts.freeze().model_copy(update=change)
    with pytest.raises(instances.InstanceManifestError, match=code):
        facts.verify(manifest, freeze=freeze)
    assert facts.verify(manifest, freeze=None, paper=False).content_hash == manifest.content_hash


def test_wrong_approved_protocol_and_artifact_fail_before_database(facts):
    manifest = facts.build()
    changed = facts.protocol.model_copy(update={"match_threshold": 0.7})
    facts.session.execute.reset_mock()
    with pytest.raises(instances.InstanceManifestError, match="approved_protocol_mismatch"):
        facts.verify(manifest, freeze=facts.freeze().model_copy(update={"protocol": changed}))
    with pytest.raises(instances.InstanceManifestError, match="artifact_hash_mismatch"):
        facts.verify(manifest, expected_manifest_hash="wrong")
    facts.session.execute.assert_not_called()


@pytest.mark.parametrize(
    "target,field,value,code",
    [
        ("compiler", "compiler_schema_version", "0.8.0", "incompatible"),
        ("state", "status", "running", "incomplete"),
        ("state", "selected_count", 1, "incompatible"),
        ("compiler", "snapshot_id", "other", "snapshot_or_scope"),
        ("run", "dataset", "memgallery", "snapshot_or_scope"),
        ("qbank", "status", "building", "index_not_ready"),
        ("qbank", "snapshot_id", "other", "index_not_ready"),
        ("qbank", "content_hash", "bad", "qbank_index_hash"),
        ("memory", "status", "failed", "index_not_ready"),
        ("memory", "content_hash", "bad", "memory_index_hash"),
        ("r0", "content_hash", "bad", "r0_hash_or_gate"),
        ("r0", "hard_gate_passed", False, "r0_hash_or_gate"),
        ("r0", "memory_index_hash", "bad", "r0_hash_or_gate"),
        ("r0", "schema_version", "0.3.0", "r0_hash_or_gate"),
        ("r0", "metadata", {}, "r0_hash_or_gate"),
    ],
)
def test_incompatible_or_tampered_runtime_facts_fail_closed(facts, target, field, value, code):
    setattr(facts, target, getattr(facts, target).model_copy(update={field: value}))
    with pytest.raises(instances.InstanceManifestError, match=code):
        facts.build()


@pytest.mark.parametrize("target", ["compiler", "qbank", "memory", "r0", "routing"])
def test_absent_facts_never_synthesized(facts, target):
    setattr(facts, target, None)
    with pytest.raises(instances.InstanceManifestError, match="missing"):
        facts.build()


@pytest.mark.parametrize("round_id", range(4))
def test_every_exact_round_is_required_and_hashed(facts, round_id):
    original = facts.rounds.pop(round_id)
    with pytest.raises(instances.InstanceManifestError, match="round_missing"):
        facts.build()
    manifest, members = original
    facts.rounds[round_id] = manifest.model_copy(update={"content_hash": "bad"}), members
    with pytest.raises(instances.InstanceManifestError, match="round_hash_mismatch"):
        facts.build()


def test_member_edits_and_duplicate_members_rejected(facts):
    manifest, members = facts.rounds[1]
    facts.rounds[1] = manifest, [members[0], members[0]]
    with pytest.raises(instances.InstanceManifestError, match="denominator_mismatch"):
        facts.build()
    facts.rounds[1] = manifest, [members[0].model_copy(update={"content_hash": "bad"}), members[1]]
    with pytest.raises(instances.InstanceManifestError, match="round_member_invalid"):
        facts.build()


def test_routing_and_collection_changes_are_verified(facts):
    manifest = facts.build()
    routing, members = facts.routing
    facts.routing = routing.model_copy(update={"content_hash": "bad"}), members
    with pytest.raises(instances.InstanceManifestError, match="routing_hash_mismatch"):
        facts.verify(manifest)
    facts.routing = routing, members
    key = facts.qbank.text_question_collection_id
    original = facts.collections[key]
    facts.collections[key] = original.model_copy(update={"status": "building"})
    with pytest.raises(instances.InstanceManifestError, match="collection_not_ready"):
        facts.verify(manifest)
    facts.collections[key] = original.model_copy(update={"preprocessing_hash": "changed"})
    with pytest.raises(instances.InstanceManifestError, match="runtime_facts_changed"):
        facts.verify(manifest)
    del facts.collections[key]
    with pytest.raises(instances.InstanceManifestError, match="collection_missing"):
        facts.build()


def test_scope_ignores_query_details_but_not_compiler_treatment(facts):
    manifest = facts.build()
    scope = facts.scope | {
        "query_id": PRIVATE,
        "gold": PRIVATE,
        "round_id": 3,
        "question_budget": 1,
        "answer_model": "unrelated",
    }
    assert facts.build(scope=scope).content_hash == manifest.content_hash
    assert (
        facts.build(scope=scope | {"treatment": "no_reranker"}).content_hash
        == manifest.content_hash
    )
    for key in ("snapshot_id", "conversation_id", "treatment"):
        changed = scope | {key: "fixed_planner" if key == "treatment" else "other"}
        with pytest.raises(instances.InstanceManifestError, match="protocol_subhash_mismatch"):
            facts.build(scope=changed)
    with pytest.raises(instances.InstanceManifestError, match="scope_missing"):
        facts.build(scope={})


def test_protocol_subhash_cannot_be_taken_from_unrelated_compiler(facts):
    changed = facts.protocol.model_copy(
        update={
            "model_entry_hashes": facts.protocol.model_entry_hashes
            | {"compiler-qwen3.5-9b": "f" * 64},
        }
    )
    with pytest.raises(instances.InstanceManifestError, match="protocol_subhash_mismatch"):
        facts.build(protocol=changed)
    facts.compiler.metadata["compiler_subprotocol_hash"] = "other"
    with pytest.raises(instances.InstanceManifestError, match="compiler_subprotocol_mismatch"):
        facts.build()


@pytest.mark.parametrize("stage", ["unit_prepare", "text_index", "multimodal_index", "r0_gate"])
def test_added_compiler_stage_sources_change_instance_identity(facts, stage):
    changed = facts.protocol.model_copy(
        update={
            "source_fingerprints": facts.protocol.source_fingerprints | {stage: "changed"},
        }
    )
    assert _backend_hash(changed, facts.scope) != facts.subhash
    with pytest.raises(instances.InstanceManifestError, match="protocol_subhash_mismatch"):
        facts.build(protocol=changed)


@pytest.mark.parametrize("treatment", ["main", "fixed_planner", "no_raw_image", "no_reranker"])
@pytest.mark.parametrize("seed", [0, 7])
def test_backend_compiler_recipe_and_treatments_match(monkeypatch, treatment, seed):
    facts = RuntimeFacts(treatment, seed=seed)
    facts.install(monkeypatch)
    manifest = facts.build()
    assert manifest.treatment == ("main" if treatment == "no_reranker" else treatment)


def test_compiler_seed_changes_instance_identity_and_rejects_old_run(facts):
    changed = facts.protocol.model_copy(update={"seed": 7})
    normalized = instances._scope(facts.scope)
    assert _backend_hash(changed, facts.scope) == instances._subprotocol_hash(changed, normalized)
    assert _backend_hash(changed, facts.scope) != facts.subhash
    with pytest.raises(instances.InstanceManifestError, match="protocol_subhash_mismatch"):
        facts.build(protocol=changed)


def test_development_namespace_matches_producer_and_survives_query_verification(facts):
    namespace = "campaign-development"
    facts.subhash = _backend_hash(facts.protocol, facts.scope, namespace)
    facts.compiler = facts.compiler.model_copy(
        update={"questioner_freeze_manifest_hash": facts.subhash}
    )
    facts.compiler.metadata["compiler_subprotocol_hash"] = facts.subhash
    facts.run.metadata["compiler_subprotocol_hash"] = facts.subhash
    for wrong in (None, "another-campaign"):
        with pytest.raises(instances.InstanceManifestError, match="protocol_subhash_mismatch"):
            facts.build(diagnostic_namespace=wrong)
    manifest = facts.build(diagnostic_namespace=namespace)
    assert (
        facts.verify(manifest, diagnostic_namespace=namespace, paper=False, freeze=None) == manifest
    )
    with pytest.raises(instances.InstanceManifestError, match="namespace_not_paper"):
        facts.verify(manifest, diagnostic_namespace=namespace)


def test_shared_memory_collections_allow_other_run_owner(facts):
    for key in (facts.memory.text_collection_id, facts.memory.visual_collection_id):
        facts.collections[key] = facts.collections[key].model_copy(
            update={"run_id": "previous-run"}
        )
    assert facts.verify(facts.build()).compiler_run_id == facts.run_id


def test_errors_do_not_echo_validation_inputs_or_private_ids(facts, capsys):
    facts.compiler_runs.get.side_effect = KeyError(PRIVATE)
    with pytest.raises(instances.InstanceManifestError) as error:
        facts.build()
    assert PRIVATE not in str(error.value)
    assert not capsys.readouterr().out


def test_database_order_and_wallclock_do_not_change_identity(facts):
    original = facts.build()
    for round_id, (manifest, members) in facts.rounds.items():
        facts.rounds[round_id] = manifest, list(reversed(members))
    facts.compiler = facts.compiler.model_copy(update={"created_at": utc_now() + timedelta(days=1)})
    assert facts.build().content_hash == original.content_hash


@pytest.mark.parametrize("ordinal", range(6))
def test_all_six_embedding_collections_must_be_ready(facts, ordinal):
    key = list(facts.collections)[ordinal]
    facts.collections[key] = facts.collections[key].model_copy(
        update={"status": EmbeddingCollectionStatus.BUILDING}
    )
    with pytest.raises(instances.InstanceManifestError, match="collection_not_ready"):
        facts.build()


def test_instance_tampering_and_wrong_runtime_scope_are_rejected(facts):
    original = facts.build()
    edited = CompiledQBankInstanceManifest.model_validate(original.model_dump())
    edited.round_membership_hashes[3] = "edited"
    with pytest.raises(instances.InstanceManifestError, match="artifact_hash_mismatch"):
        facts.verify(edited, expected_manifest_hash=original.content_hash)
    facts.units[0] = facts.units[0].model_copy(update={"conversation_id": "other-conversation"})
    with pytest.raises(instances.InstanceManifestError, match="unit_scope_mismatch"):
        facts.verify(original)


@pytest.mark.parametrize("treatment", ["main", "no_raw_image"])
def test_imported_indexes_preserve_adoption_identity_and_native_rounds(monkeypatch, tmp_path, treatment):
    from copy import deepcopy

    facts = RuntimeFacts(treatment)
    facts.approval_root = tmp_path
    facts.install(monkeypatch)
    metadata = {"partial_prefix_import": {"proof_hash": "a" * 64}}
    facts.qbank = facts.qbank.model_copy(
        update={
            "qbank_index_id": "q-adopt-index",
            "content_hash": "b" * 64,
            "metadata": metadata,
        }
    )
    facts.memory = facts.memory.model_copy(
        update={"metadata": {**facts.memory.metadata, **metadata}}
    )
    facts.collections = {
        key: c.model_copy(update={"metadata": {**c.metadata, **metadata}})
        for key, c in facts.collections.items()
    }
    facts.embeddings.get_collection.side_effect = facts.collections.get
    rows = {
        "qbank_index_manifests": [facts.qbank.model_dump(mode="json")],
        "memory_operator_index_manifests": [facts.memory.model_dump(mode="json")],
        "embedding_collections": [c.model_dump(mode="json") for c in facts.collections.values()],
    }
    reader = AsyncMock(return_value=deepcopy(rows))
    monkeypatch.setattr(instances, "_imported_index_rows", reader)
    payload = instances._payload(
        facts.r0, "r0_manifest_id", "schema_version", "content_hash", "created_at", "metadata"
    )
    payload["artifact_hash"] = canonical_content_hash(
        [facts.qbank.content_hash, facts.memory.content_hash]
    )
    facts.r0 = facts.r0.model_copy(
        update={
            **payload,
            "content_hash": canonical_content_hash(payload),
            "r0_manifest_id": stable_compiler_id("r0-input-manifest", payload),
        }
    )
    facts.rounds = {r: _round(facts.r0, r) for r in range(4)}
    facts.routing = _routing(facts.r0, facts.qbank, facts.rounds[2])
    manifest = facts.build()
    assert facts.qbank.content_hash in manifest.artifact_hashes
    assert instances._fact_hash(facts.memory) in manifest.artifact_hashes
    facts.verify(manifest)
    assert reader.await_count == 2  # Construction and query-time verification share the path.
    key = facts.memory.text_collection_id
    facts.collections[key] = facts.collections[key].model_copy(
        update={"metadata": {"forged": True}}
    )
    with pytest.raises(instances.InstanceManifestError, match="import_collection_fact_mismatch"):
        facts.build()


def test_import_row_marker_without_compiler_proof_cannot_bypass_hash(facts):
    facts.qbank = facts.qbank.model_copy(
        update={
            "content_hash": "b" * 64,
            "metadata": {"partial_prefix_import": {"proof_hash": "a" * 64}},
        }
    )
    with pytest.raises(instances.InstanceManifestError, match="qbank_index_hash_mismatch"):
        facts.build()
