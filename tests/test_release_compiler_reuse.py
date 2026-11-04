"""Native exploration plans, dispatch, instance attestation and producer accounting.

The committed-row lookup executes SQL against disposable SQLite tables. The bank
facts use the typed runtime-only repositories from the native instance tests.
Neither database transport requires a live service, GPU or model gateway.
"""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import JSON, Column, MetaData, Table, create_engine, select, update
from sqlalchemy.orm import Session, registry
from sqlalchemy.sql import sqltypes

import test_step8_instances as bank
from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.experiments import release_compiler_reuse as reuse
from pmmc.experiments import release_inputs, step8_instances
from pmmc.experiments.release_compiler_reuse import REFERENCE, STAGES
from pmmc.experiments.step8_artifacts import CampaignArtifactStore
from pmmc.experiments.step8_backend import (
    ProductionComponentBackend,
    compiler_instance_id,
    compiler_subprotocol_hash,
)
from pmmc.experiments.step8_cost_dependencies import DependencyUnproven
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.experiments.step8_pmmc_cost_dependencies import (
    _kernel_receipt_lineage,
    _validate_kernel_lineage,
    validate_execution_dependencies,
)
from pmmc.experiments.step8_upstream_import import publish_proof, questioner_compatibility
from pmmc.infrastructure.database import engine as db_engine
from pmmc.infrastructure.database import step8_models
from pmmc.schemas.base import utc_now
from pmmc.schemas.step8 import STEP8_ALEMBIC_REVISION, BenchmarkCampaignManifest
from test_release_inputs import project as project  # noqa: F401 - shared synthetic release project
from test_step8_pmmc_cost_dependencies import execution_graph
from test_step8_upstream_import import proof as questioner_proof


def native_plan(project, variant):
    pointer = release_inputs.prepare_exploration(project, "bank-reuse", variant=variant)
    directory = project.project_root / pointer["campaign_dir"]
    campaign = BenchmarkCampaignManifest.model_validate_json(
        (directory / "manifests/campaign.json").read_bytes()
    )
    jobs = [
        BenchmarkStageJob.model_validate_json(p.read_bytes())
        for p in (directory / "private/jobs").glob("*.json")
    ]
    return campaign, jobs


def stage_jobs(jobs, treatment):
    scoped = [
        j
        for j in jobs
        if j.parameters.get("dataset") == "memlens"
        and j.parameters.get("treatment") == treatment
        and j.operation in ("compiler", "adopt_questioner")
    ]
    scope = scoped[0].parameters
    return {j.stage: j for j in scoped if j.parameters == scope}


class Database:
    def __init__(self, monkeypatch):
        self.engine = create_engine("sqlite://")
        self.maps = registry(metadata=MetaData())
        self.models = {}
        for name in (
            "BenchmarkCampaignModel",
            "BenchmarkStageJobModel",
            "BenchmarkComponentReceiptModel",
            "RunStateModel",
        ):
            original = getattr(reuse, name)
            cols = [
                Column(
                    c.key,
                    JSON if isinstance(c.type, (sqltypes.ARRAY, sqltypes.JSON)) else c.type,
                    primary_key=c.primary_key,
                )
                for c in original.__table__.columns
            ]
            table = Table(original.__tablename__, self.maps.metadata, *cols)
            model = type(name, (), {})
            self.maps.map_imperatively(model, table)
            monkeypatch.setattr(reuse, name, model)
            self.models[name] = model
        self.maps.metadata.create_all(self.engine)
        monkeypatch.setattr(
            db_engine,
            "create_session_factory",
            lambda *_: (SimpleNamespace(dispose=AsyncMock()), self.session),
        )

    def add(self, name, **values):
        with Session(self.engine) as session, session.begin():
            session.add(self.models[name](**values))

    def session(self):
        session = Session(self.engine)

        class Reader:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                session.close()

            async def get(self, *args):
                return session.get(*args)

            async def execute(self, statement):
                if str(statement) == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY":
                    return None
                return session.execute(statement)

        return Reader()

    def owner(self, m):
        self.add(
            "BenchmarkCampaignModel",
            campaign_id=m.campaign_id,
            phase=m.phase.value,
            manifest_hash=canonical_content_hash(m.model_dump(mode="json")),
            protocol_hash=m.protocol.content_hash,
            payload=m.model_dump(mode="json"),
        )

    def commit(self, j, r):
        self.add(
            "BenchmarkStageJobModel",
            job_id=j.job_id,
            campaign_id=j.campaign_id,
            component_key=j.component_key,
            stage=j.stage,
            status="completed",
            completed_at=utc_now(),
            scientific_input_hash=j.scientific_input_hash,
            source_fingerprint=j.source_fingerprint,
            dependency_ids=j.dependency_ids,
            receipt_hash=r.content_hash,
            payload=j.model_dump(mode="json"),
        )
        self.add(
            "BenchmarkComponentReceiptModel",
            job_id=j.job_id,
            receipt_hash=r.content_hash,
            component_identity=r.component_identity,
            artifact_hashes=r.artifact_hashes,
            payload=r.model_dump(mode="json"),
        )

    async def read(self, model, predicate):
        with Session(self.engine) as session:
            return [
                dict(row)
                for row in session.execute(select(model.__table__).where(predicate)).mappings()
            ]

    def run_status(self, run, status):
        model = self.models["RunStateModel"]
        with Session(self.engine) as session, session.begin():
            session.execute(update(model).where(model.run_id == run).values(status=status))


def runtime_bank(mp, owner, scope):
    # Construct typed, fully hashed native facts using the actual snapshot and
    # conversation in the release plan (not hard-coded compiler or receipt IDs).
    def identity(proto, fixture_scope):
        fixture_scope.update(scope)
        return compiler_subprotocol_hash(owner, scope)

    mp.setattr(bank, "_backend_hash", identity)
    mp.setattr(bank, "protocol", lambda: owner.protocol)
    f = bank.RuntimeFacts(scope["treatment"], owner.protocol.seed)
    snapshot = scope["snapshot_id"]
    f.run = f.run.model_copy(update={"dataset_snapshot_ids": [snapshot]})
    f.units = [
        u.model_copy(
            update={
                "snapshot_id": snapshot,
                "memories": [v.model_copy(update={"snapshot_id": snapshot}) for v in u.memories],
            }
        )
        for u in f.units
    ]
    channels = {
        k: getattr(f.qbank, k + "_collection_id")
        for k in (
            "text_question",
            "text_paraphrase",
            "multimodal_question",
            "multimodal_paraphrase",
        )
    }
    q_recipe = f.qbank.model_dump(
        mode="json",
        exclude={
            "qbank_index_id",
            "created_at",
            "status",
            "content_hash",
            "metadata",
            *(k + "_collection_id" for k in channels),
        },
    )
    q_recipe.update(snapshot_id=snapshot, collections=channels)
    f.qbank = f.qbank.model_copy(
        update={
            "snapshot_id": snapshot,
            "content_hash": canonical_content_hash(q_recipe),
            "qbank_index_id": stable_compiler_id("qbank-index", q_recipe),
        }
    )
    memory, text, visual = bank.make_memory_operator_index(
        run_id=f.run_id,
        snapshot_id=snapshot,
        dataset="memlens",
        selection_hash=f.run.selection_hash,
        text_model_alias=f.memory.text_model_alias,
        text_model_revision=f.memory.text_model_revision,
        multimodal_model_alias=f.memory.multimodal_model_alias,
        multimodal_model_revision=f.memory.multimodal_model_revision,
        memory_raw_image_policy="forbid" if scope["treatment"] == "no_raw_image" else "allow",
    )
    f.memory = memory.model_copy(update={"status": "ready"})
    for key in channels.values():
        f.collections[key] = f.collections[key].model_copy(update={"snapshot_id": snapshot})
    for c in (text, visual):
        f.collections[c.collection_id] = c.model_copy(
            update={"status": bank.EmbeddingCollectionStatus.READY}
        )
    r = f.r0.model_dump(
        mode="json",
        exclude={"r0_manifest_id", "schema_version", "created_at", "metadata", "content_hash"},
    )
    r.update(
        snapshot_id=snapshot,
        memory_index_hash=f.memory.content_hash,
        artifact_hash=canonical_content_hash([f.qbank.content_hash, f.memory.content_hash]),
    )
    f.r0 = f.r0.model_copy(
        update={
            **r,
            "content_hash": canonical_content_hash(r),
            "r0_manifest_id": stable_compiler_id("r0-input-manifest", r),
        }
    )
    f.rounds = {i: bank._round(f.r0, i) for i in range(4)}
    f.routing = bank._routing(f.r0, f.qbank, f.rounds[2])
    mp.setattr(step8_instances, "ExperimentUnitOfWork", lambda *_: f)
    return f


def receipt(j, output, hashes, upstream):
    return ComponentReceipt(
        job_id=j.job_id,
        scientific_input_hash=j.scientific_input_hash,
        source_fingerprint=j.source_fingerprint,
        schema_revision=STEP8_ALEMBIC_REVISION,
        upstream_receipt_hashes=upstream,
        output=output,
        artifact_hashes=hashes,
        row_counts={"component_outputs": 1},
    )


def produce(db, store, owner, stages, facts):
    scope, run = stages["projection"].parameters, facts.run_id
    instance = asyncio.run(
        step8_instances.build_compiled_qbank_instance_manifest(
            db.session,
            protocol=owner.protocol,
            compiler_run_id=run,
            compiler_subprotocol_hash=facts.subhash,
            scope=scope,
            diagnostic_namespace="exploration-dev-all",
        )
    )
    db.add(
        "RunStateModel", run_id=run, status="completed", completed_at=utc_now(), selected_count=0
    )
    compat = questioner_compatibility(owner, scope)
    proof = questioner_proof().model_copy(
        update={
            "source_campaign_id": owner.campaign_id,
            "source_protocol_hash": owner.protocol.content_hash,
            "source_run_id": run,
            "compatibility": compat,
            "compatibility_hash": canonical_content_hash(compat),
        }
    )
    qdigest = publish_proof(store, proof)
    saved = {}
    for stage in STAGES:
        j = stages[stage]
        output = dict(
            compiler_run_id=run,
            phase=stage,
            program_count=2,
            eligible_candidate_count=2,
            run_status="completed" if stage == "projection" else "running",
        )
        artifacts = []
        if stage == "questioner":
            output["questioner_import_proof_sha256"] = qdigest
            artifacts.append(qdigest)
            if j.operation == "adopt_questioner":
                donor = store.put_json({"fixture": "paired main adoption proof"})
                output["questioner_adoption"] = {"source_proof_sha256": donor}
                artifacts.append(donor)
        if stage == "projection":
            output.update(
                compiled_instance=instance.model_dump(mode="json"),
                compiled_instance_hash=instance.content_hash,
            )
        r = receipt(
            j,
            output,
            [store.put_json(output), *artifacts],
            {
                key: saved[key].content_hash if key in saved else "a" * 64
                for key in j.dependency_ids
            },
        )
        saved[j.job_id] = r
        db.commit(j, r)
    return saved


def backend(project, campaign, store):
    b = object.__new__(ProductionComponentBackend)
    b.campaign, b.settings, b.store = campaign, project, store
    return b


@pytest.mark.parametrize("variants", [("wr", "w"), ("r", "legacy")])
def test_native_plans_reference_only_committed_compiler(project, monkeypatch, variants):
    (source, sjobs), (target, tjobs) = (native_plan(project, v) for v in variants)
    assert source.protocol.content_hash != target.protocol.content_hash
    assert (
        source.protocol.config_hashes["query_executor_config"]
        != target.protocol.config_hashes["query_executor_config"]
    )
    db, store = Database(monkeypatch), CampaignArtifactStore(project.outputs_path / "step8")
    db.owner(target)
    db.owner(source)
    monkeypatch.setattr("pmmc.infrastructure.settings.get_settings", lambda: project)
    for name in db.models.keys() - {"RunStateModel"}:
        monkeypatch.setattr(step8_models, name, db.models[name])
    b = backend(project, target, store)
    monkeypatch.setattr(b, "_seal_questioner", AsyncMock(side_effect=AssertionError("reseal")))
    monkeypatch.setattr(
        "pmmc.compiler.program_workflow.run_program_workflow",
        AsyncMock(side_effect=AssertionError("native phase replay")),
    )
    monkeypatch.setattr(
        b, "_try_questioner_import", AsyncMock(side_effect=AssertionError("readopt"))
    )
    for treatment in ("main", "fixed_planner"):
        src, dest = stage_jobs(sjobs, treatment), stage_jobs(tjobs, treatment)
        scope = src["projection"].parameters
        assert set(src) == set(dest) == set(STAGES)
        assert all(
            src[s].scientific_input_hash == dest[s].scientific_input_hash
            and src[s].job_id != dest[s].job_id
            for s in STAGES
        )
        assert dest["questioner"].operation == (
            "adopt_questioner" if treatment == "fixed_planner" else "compiler"
        )
        facts = runtime_bank(monkeypatch, source, scope)
        run = compiler_instance_id(target, scope)
        assert facts.run_id == run
        produced = produce(db, store, source, src, facts)
        index = store.root / "_store/upstream/questioner"
        registry = {p: p.read_bytes() for p in index.rglob("*.json")}
        consumed = {}
        for s in STAGES:
            j = dest[s]
            upstream = {
                key: consumed[key].content_hash if key in consumed else "b" * 64
                for key in j.dependency_ids
            }
            output, artifacts, counts = asyncio.run(b.execute(j, upstream))
            original = produced[src[s].job_id]
            ref = output[REFERENCE]
            assert (
                ref["producer_receipt_hash"] == original.content_hash
                and ref["new_model_calls"] == 0
            )
            assert (
                ref["producer_job_id"] == src[s].job_id
                and ref["producer_campaign_id"] == source.campaign_id
            )
            assert set(original.artifact_hashes) <= set(artifacts) and counts == {
                "component_outputs": 1
            }
            if s == "projection":
                assert output["compiled_instance_hash"] != original.output["compiled_instance_hash"]
                assert output["compiled_instance"]["protocol_hash"] == target.protocol.content_hash
            else:
                assert {k: v for k, v in output.items() if k != REFERENCE} == original.output
            r = receipt(j, output, artifacts, upstream)
            asyncio.run(b.verify_recovery(j, upstream, r))
            if s == "questioner":
                with pytest.raises(ValueError, match="changed_on_recovery"):
                    asyncio.run(
                        b.verify_recovery(
                            j, upstream, r.model_copy(update={"output": original.output})
                        )
                    )
            db.commit(j, r)
            consumed[j.job_id] = r
        assert registry == {p: p.read_bytes() for p in index.rglob("*.json")}
        _, rows = asyncio.run(reuse.read_facts(db.session, run))
        assert len(rows) == 2 * len(STAGES)
        args = ({"purpose": "dry_run"}, {"round_id": 0, "run_id": run})
        _validate_kernel_lineage(rows, *args)
        _validate_kernel_lineage(rows, args[0], {**args[1], "round_id": 1})
        # One query-time owner (not the compiler owner) and two references to
        # the same vector producer incur one physical write-call dependency.
        query = execution_graph()
        query["execution"]["source_compiler_run_id"] = query["compiler"]["run_id"] = query[
            "program"
        ]["run_id"] = run
        for vector in query["query_vectors"]:
            vector["run_id"] = run
        original_cost = validate_execution_dependencies(query)
        if treatment == "main":
            (qrow,) = query["kernel_lineage"]
            db.owner(BenchmarkCampaignManifest.model_validate(qrow["campaign"]["payload"]))
            db.commit(
                BenchmarkStageJob.model_validate(qrow["job"]["payload"]),
                ComponentReceipt.model_validate(qrow["receipt"]["payload"]),
            )
        query["kernel_lineage"] = asyncio.run(
            _kernel_receipt_lineage(db.read, run, query["execution"]["run_id"])
        )
        assert {e["job"]["stage"] for e in query["kernel_lineage"]} == {*STAGES, "execution"}
        assert len(query["kernel_lineage"]) == 2 * len(STAGES) + 1
        assert (
            validate_execution_dependencies(query)
            == original_cost
            == {"query": set(), "write": {"shared-producer"}}
        )
        broken = deepcopy(rows)
        alias = next(e for e in broken if e["job"]["campaign_id"] == target.campaign_id)
        forged = ComponentReceipt.model_validate(alias["receipt"]["payload"])
        old_hash = reuse.output_hash(forged.output)
        forged.output[REFERENCE]["producer_receipt_hash"] = "f" * 64
        new_hash = reuse.output_hash(forged.output)
        forged = forged.model_copy(
            update={
                "artifact_hashes": [
                    new_hash if h == old_hash else h for h in forged.artifact_hashes
                ]
            }
        )
        alias["receipt"].update(
            payload=forged.model_dump(mode="json"),
            receipt_hash=forged.content_hash,
            artifact_hashes=forged.artifact_hashes,
        )
        alias["job"]["receipt_hash"] = forged.content_hash
        with pytest.raises(DependencyUnproven, match="reference_unproven"):
            _validate_kernel_lineage(broken, *args)
        query["kernel_lineage"] = [
            e for e in query["kernel_lineage"] if e["job"]["stage"] == "execution"
        ] + broken
        with pytest.raises(DependencyUnproven, match="reference_unproven"):
            validate_execution_dependencies(query)
        with pytest.raises(DependencyUnproven, match="reference_unproven"):
            _validate_kernel_lineage([e for e in rows if e["job"]["stage"] != "projection"], *args)
        with pytest.raises(ValueError, match="stage_identity_changed"):
            asyncio.run(
                b._compiler(
                    dest["planner0"].model_copy(update={"scientific_inputs": {"changed": True}}),
                    scope,
                )
            )
        db.run_status(run, "running")
        with pytest.raises(ValueError, match="projection_without_completed_run"):
            asyncio.run(b._compiler(dest["planner0"], scope))
        db.run_status(run, "completed")
        # Actual native attestation of the same completed bank must still succeed on each lookup.
        facts.qbank = facts.qbank.model_copy(update={"status": "building"})
        with pytest.raises(step8_instances.InstanceManifestError):
            asyncio.run(b._compiler(dest["projection"], scope))
