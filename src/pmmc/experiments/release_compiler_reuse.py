"""Read-only references to an already-produced exploration compiler bank.

An identical compiler ID alone is not authority. Follow the committed projection's
dependency hashes back to the exact stage and compare the bank with live runtime
facts. The consumer gets its own receipt, never a new physical producer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, text

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import json_bytes
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.infrastructure.database.experiment_models import RunStateModel
from pmmc.infrastructure.database.step8_models import (
    BenchmarkCampaignModel,
    BenchmarkComponentReceiptModel,
    BenchmarkStageJobModel,
)
from pmmc.schemas.step8 import (
    STEP8_ALEMBIC_REVISION,
    BenchmarkCampaignManifest,
    CampaignPhase,
    CompiledQBankInstanceManifest,
)

REFERENCE = "compiler_producer_reference"
STAGES = (
    "questioner",
    "planner0",
    "text_index",
    "multimodal_index",
    "dry_run0",
    "r0_gate",
    "doubter",
    "projection",
)


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError("exploration_compiler_reference_" + code)


def output_hash(payload: dict) -> str:
    return hashlib.sha256(json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class Fact:
    campaign: BenchmarkCampaignManifest
    job: BenchmarkStageJob
    receipt: ComponentReceipt

    @classmethod
    def read(cls, entry: dict) -> Fact:
        owner, row, saved = entry["campaign"], entry["job"], entry["receipt"]
        m = BenchmarkCampaignManifest.model_validate(owner["payload"])
        j = BenchmarkStageJob.model_validate(row["payload"])
        r = ComponentReceipt.model_validate(saved["payload"])
        require(
            m.phase is CampaignPhase.EXPLORATION
            and not m.diagnostic_only
            and not m.paper_result
            and owner["phase"] == m.phase.value
            and owner["manifest_hash"] == canonical_content_hash(owner["payload"])
            and owner["campaign_id"] == m.campaign_id == row["campaign_id"] == j.campaign_id
            and owner["protocol_hash"] == m.protocol.content_hash
            and row["status"] == "completed"
            and bool(row["completed_at"])
            and row["stage"] == j.stage
            and j.stage in STAGES
            and j.operation in {"compiler", "adopt_questioner"}
            and row["job_id"] == saved["job_id"] == j.job_id == r.job_id
            and row["component_key"] == j.component_key
            and row["dependency_ids"] == j.dependency_ids
            and set(j.dependency_ids) == set(r.upstream_receipt_hashes)
            and row["scientific_input_hash"] == j.scientific_input_hash == r.scientific_input_hash
            and row["source_fingerprint"] == j.source_fingerprint == r.source_fingerprint
            and j.source_fingerprint == m.protocol.source_fingerprints[j.stage]
            and r.schema_revision == STEP8_ALEMBIC_REVISION
            and row["receipt_hash"] == saved["receipt_hash"] == r.content_hash
            and saved["component_identity"] == r.component_identity
            and saved["artifact_hashes"] == r.artifact_hashes
            and output_hash(r.output) in r.artifact_hashes,
            "uncommitted_or_changed_fact",
        )
        for key in ("questioner_import_proof_sha256", "r0_prefix_import_proof_sha256"):
            if key in r.output:
                require(r.output[key] in r.artifact_hashes, "unbound_stage_proof")
        if j.stage == "questioner":
            require("questioner_import_proof_sha256" in r.output, "questioner_proof_missing")
        for key in ("questioner_adoption", "r0_prefix_adoption"):
            if key in r.output:
                require(
                    r.output[key]["source_proof_sha256"] in r.artifact_hashes,
                    "unbound_adoption_proof",
                )
        return cls(m, j, r)


def physical_chain(entries: list[dict], run: str) -> dict[str, Fact]:
    """Resolve only the unique physical projection; never follow consumer aliases."""
    physical = [
        Fact.read(e)
        for e in entries
        if e["job"]["stage"] in STAGES
        and e["receipt"]["payload"]["output"].get("compiler_run_id") == run
        and REFERENCE not in e["receipt"]["payload"]["output"]
    ]
    projections = [f for f in physical if f.job.stage == "projection"]
    require(len(projections) == 1, "physical_projection_not_unique")
    f = projections[0]
    owner, scope = f.campaign, f.job.parameters
    by_id = {p.job.job_id: p for p in physical}
    require(len(by_id) == len(physical), "duplicate_physical_job")
    chain = {}
    for stage in reversed(STAGES):
        require(
            f.job.stage == stage
            and f.campaign == owner
            and f.job.parameters == scope
            and f.receipt.output.get("compiler_run_id") == run,
            "producer_chain_scope",
        )
        chain[stage] = f
        if stage != "questioner":
            require(len(f.job.dependency_ids) == 1, "producer_dependency_shape")
            parent = by_id.get(f.job.dependency_ids[0])
            require(
                parent is not None
                and parent.receipt.content_hash
                == f.receipt.upstream_receipt_hashes[parent.job.job_id],
                "producer_dependency_changed",
            )
            assert parent is not None
            f = parent
    require(len(chain) == len(physical), "ambiguous_physical_producer")
    return chain


def referenced_output(chain, consumer, job) -> dict:
    from pmmc.experiments.step8_backend import compiler_instance_id, compiler_subprotocol_hash

    producer, projection = chain[job.stage], chain["projection"]
    owner = producer.campaign
    run = compiler_instance_id(consumer, job.parameters)
    subhash = compiler_subprotocol_hash(consumer, job.parameters)
    require(
        consumer.phase is CampaignPhase.EXPLORATION
        and not consumer.paper_result
        and not consumer.diagnostic_only
        and consumer.campaign_id == job.campaign_id
        and producer.campaign.campaign_id != job.campaign_id,
        "consumer_scope",
    )
    require(
        compiler_instance_id(owner, producer.job.parameters) == run
        and compiler_subprotocol_hash(owner, producer.job.parameters) == subhash
        and producer.job.parameters == job.parameters,
        "compiler_identity_changed",
    )
    require(
        producer.job.operation == job.operation
        and producer.job.resource_class == job.resource_class
        and producer.job.scientific_input_hash == job.scientific_input_hash
        and producer.job.source_fingerprint == job.source_fingerprint
        and consumer.protocol.source_fingerprints[job.stage] == job.source_fingerprint,
        "stage_identity_changed",
    )
    old = CompiledQBankInstanceManifest.model_validate(
        projection.receipt.output["compiled_instance"]
    )
    require(
        old.compiler_run_id == run
        and old.protocol_hash == producer.campaign.protocol.content_hash
        and old.content_hash == projection.receipt.output.get("compiled_instance_hash")
        and old.scope_hash == canonical_content_hash(job.parameters)
        and old.snapshot_id == job.parameters["snapshot_id"]
        and old.dataset.value == job.parameters["dataset"]
        and old.treatment.value == job.parameters["treatment"]
        and subhash in old.artifact_hashes
        and projection.receipt.output.get("run_status") == "completed",
        "producer_instance_changed",
    )
    bound = old.model_copy(update={"protocol_hash": consumer.protocol.content_hash})
    result = dict(producer.receipt.output)
    if job.stage == "projection":
        result.update(
            compiled_instance=bound.model_dump(mode="json"),
            compiled_instance_hash=bound.content_hash,
        )
    result[REFERENCE] = {
        "protocol": "exploration-compiler-reference-v1",
        "compiler_subprotocol_hash": subhash,
        "producer_campaign_id": producer.campaign.campaign_id,
        "producer_protocol_hash": old.protocol_hash,
        "producer_job_id": producer.job.job_id,
        "producer_receipt_hash": producer.receipt.content_hash,
        "projection_job_id": projection.job.job_id,
        "projection_receipt_hash": projection.receipt.content_hash,
        "producer_instance_hash": old.content_hash,
        "consumer_instance_hash": bound.content_hash,
        "producer_artifacts": list(
            dict.fromkeys([*producer.receipt.artifact_hashes, *projection.receipt.artifact_hashes])
        ),
        "new_model_calls": 0,
    }
    return result


def check_consumer_parent(consumers, job, run, hashes=None):
    # The Q parent is a preparation or paired main Q, not a producer in this bank.
    # Its committed hash is checked by the ordinary fenced repository.
    if job.stage == "questioner":
        return
    require(len(job.dependency_ids) == 1, "consumer_dependency_shape")
    parent = consumers.get(job.dependency_ids[0])
    require(
        parent is not None
        and parent.campaign.campaign_id == job.campaign_id
        and parent.job.stage == STAGES[STAGES.index(job.stage) - 1]
        and parent.job.parameters == job.parameters
        and parent.receipt.output["compiler_run_id"] == run
        and (hashes is None or hashes == {parent.job.job_id: parent.receipt.content_hash}),
        "consumer_dependency_changed",
    )


def verify_reference_lineage(entries: list[dict]) -> dict[str, Fact]:
    """Cost-side validation: references must reproduce the full committed payload."""
    refs = [e for e in entries if REFERENCE in e["receipt"]["payload"]["output"]]
    facts = [Fact.read(e) for e in refs]
    by_job = {f.job.job_id: f for f in facts}
    require(len(by_job) == len(facts), "duplicate_consumer_receipt")
    chains = {}
    for f in facts:
        run = f.receipt.output["compiler_run_id"]
        if run not in chains:
            chains[run] = physical_chain(entries, run)
        expected = referenced_output(chains[run], f.campaign, f.job)
        check_consumer_parent(by_job, f.job, run, f.receipt.upstream_receipt_hashes)
        require(
            f.receipt.row_counts == {"component_outputs": 1}
            and f.receipt.output == expected
            and set(f.receipt.artifact_hashes)
            == {output_hash(expected), *expected[REFERENCE]["producer_artifacts"]},
            "consumer_receipt_changed",
        )
    return by_job


async def read_facts(sessions, run: str) -> tuple[str | None, list[dict]]:
    async with sessions() as session:
        await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        state = await session.get(RunStateModel, run)
        status = state.status if state is not None else None
        if status == "completed":
            require(state.completed_at is not None, "missing_terminal_timestamp")
        c, j, r = BenchmarkCampaignModel, BenchmarkStageJobModel, BenchmarkComponentReceiptModel
        stmt = (
            select(c, j, r)
            .select_from(r)
            .join(j, j.job_id == r.job_id)
            .join(c, c.campaign_id == j.campaign_id)
            .where(r.payload["output"]["compiler_run_id"].as_string() == run, j.stage.in_(STAGES))
        )
        if status != "completed":
            # A restored/mutated database with committed terminal evidence must
            # not be treated as a fresh compiler run.
            conflict = await session.execute(stmt.where(j.stage == "projection").limit(1))
            require(conflict.first() is None, "projection_without_completed_run")
            return status, []
        rows = (await session.execute(stmt)).all()
        return status, [
            {
                k: {col.key: getattr(obj, col.key) for col in obj.__table__.columns}
                for k, obj in zip(("campaign", "job", "receipt"), row, strict=True)
            }
            for row in rows
        ]


async def reuse_completed(backend: Any, job: BenchmarkStageJob, scope: dict) -> dict | None:
    from pmmc.experiments.step8_backend import compiler_instance_id, compiler_subprotocol_hash
    from pmmc.experiments.step8_instances import build_compiled_qbank_instance_manifest
    from pmmc.infrastructure.database.constants import DatabaseRole
    from pmmc.infrastructure.database.engine import create_session_factory

    run = compiler_instance_id(backend.campaign, scope)
    engine, sessions = create_session_factory(backend.settings, DatabaseRole.RUNTIME)
    try:
        status, entries = await read_facts(sessions, run)
        if status != "completed":
            return None
        if job.stage == "projection" and not any(
            e["job"]["stage"] == "projection" for e in entries
        ):
            # Native finalize can commit the run before the worker commits its
            # projection receipt. Only the original owner's fenced retry may finish it.
            parents = [Fact.read(e) for e in entries if e["job"]["job_id"] in job.dependency_ids]
            require(
                len(parents) == 1
                and parents[0].campaign.campaign_id == job.campaign_id
                and parents[0].job.stage == "doubter"
                and parents[0].job.parameters == scope,
                "terminal_projection_uncommitted",
            )
            return None
        chain = physical_chain(entries, run)
        consumers = verify_reference_lineage(entries)
        check_consumer_parent(consumers, job, run)
        output = referenced_output(chain, backend.campaign, job)
        ref = output[REFERENCE]
        for digest in ref["producer_artifacts"]:
            backend.store.verify(digest)
        # Verify the original immutable proof by digest and provenance; do not
        # consult a mutable registry pointer or publish under the consumer ID.
        stage = chain[job.stage]
        from pmmc.experiments.step8_upstream_import import (
            QuestionerImportProof,
            R0PrefixImportProof,
            questioner_compatibility,
            r0_prefix_compatibility,
        )

        for field, schema, compatibility in (
            ("questioner_import_proof_sha256", QuestionerImportProof, questioner_compatibility),
            ("r0_prefix_import_proof_sha256", R0PrefixImportProof, r0_prefix_compatibility),
        ):
            if field in stage.receipt.output:
                proof = schema.model_validate(backend.store.load_json(stage.receipt.output[field]))
                require(
                    proof.source_campaign_id == stage.campaign.campaign_id
                    and proof.source_protocol_hash == stage.campaign.protocol.content_hash
                    and proof.source_run_id == run
                    and proof.compatibility == compatibility(stage.campaign, stage.job.parameters),
                    "producer_proof_identity_changed",
                )
        current = await build_compiled_qbank_instance_manifest(
            sessions,
            protocol=backend.campaign.protocol,
            compiler_run_id=run,
            compiler_subprotocol_hash=compiler_subprotocol_hash(backend.campaign, scope),
            scope=scope,
            diagnostic_namespace="exploration-dev-all",
        )
        require(current.content_hash == ref["consumer_instance_hash"], "live_instance_changed")
        return output
    finally:
        await engine.dispose()
