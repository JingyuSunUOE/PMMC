"""Bounded upstream adoption and immutable pre-Doubter proof contracts.

Questioner adoption is independent of the stricter R0-prefix contract. Neither
contract licenses a receipt-only skip or mutation of the scientific producer.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from pmmc.compiler.ids import canonical_content_hash, compilation_unit_instance_id
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, sha256_file, write_json
from pmmc.experiments.step8_model_identity import compiler_model_entry_hashes
from pmmc.schemas.base import BaseSchema, NonEmptyStr
from pmmc.schemas.step8 import STEP8_ALEMBIC_REVISION, BenchmarkCampaignManifest

IMPORT_VERSION = "step8-questioner-fact-adoption-v1"
R0_PREFIX_VERSION = "step8-r0-prefix-proof-v1"
R0_ADOPTION_VERSION = "step8-r0-prefix-adoption-v1"
R0_PREFIX_STAGES = (
    "questioner",
    "planner0",
    "text_index",
    "multimodal_index",
    "dry_run0",
    "r0_gate",
)


def r0_prefix_compatibility(
    campaign: BenchmarkCampaignManifest, scope: dict[str, Any]
) -> dict[str, Any]:
    """Only a Doubter implementation/config change is outside this boundary.

    Keep broad stage fingerprints intact, even when a shared Doubter schema or
    workflow edit consequently invalidates R0. Model aliases are shared by Q/P/D,
    so a Doubter model change is NOT a Doubter-only change for this contract.
    """
    protocol = campaign.protocol.model_dump(mode="json")
    protocol.pop("doubter")
    protocol["config_hashes"].pop("doubter_config")
    protocol["source_fingerprints"].pop("doubter")
    treatment = scope.get("treatment", "main")
    if treatment == "no_reranker":
        treatment = "main"
    if treatment not in {"main", "fixed_planner", "no_raw_image"}:
        raise ValueError("r0_prefix_import_unknown_treatment")
    return {
        "version": R0_PREFIX_VERSION,
        "questioner": questioner_compatibility(campaign, scope),
        "protocol_without_doubter": protocol,
        "treatment": treatment,
        # Diagnostic campaigns are deliberately not cross-campaign donors.
        "diagnostic_namespace": campaign.campaign_id if campaign.diagnostic_only else None,
        "paper_result": campaign.paper_result,
    }


class R0PrefixImportProof(BaseSchema):
    version: str = R0_PREFIX_VERSION
    covered_stages: tuple[str, ...] = R0_PREFIX_STAGES
    source_campaign_id: NonEmptyStr
    source_protocol_hash: NonEmptyStr
    source_run_id: NonEmptyStr
    compatibility: dict[str, Any]
    compatibility_hash: NonEmptyStr
    schema_revision: NonEmptyStr
    table_contract_hash: NonEmptyStr
    facts_hash: NonEmptyStr
    row_counts: dict[str, int]
    artifact_hashes: dict[str, str]
    questioner_proof: QuestionerImportProof
    r0_content_hash: NonEmptyStr

    @model_validator(mode="after")
    def _identities(self) -> R0PrefixImportProof:
        if (
            self.version != R0_PREFIX_VERSION
            or self.covered_stages != R0_PREFIX_STAGES
            or self.schema_revision != STEP8_ALEMBIC_REVISION
            or self.compatibility_hash != canonical_content_hash(self.compatibility)
            or not self.row_counts
            or any(value < 0 for value in self.row_counts.values())
            or self.questioner_proof.source_run_id != self.source_run_id
            or self.questioner_proof.source_campaign_id != self.source_campaign_id
            or self.questioner_proof.source_protocol_hash != self.source_protocol_hash
            or self.questioner_proof.compatibility != self.compatibility.get("questioner")
        ):
            raise ValueError("r0_prefix_import_proof_identity_invalid")
        return self

    @property
    def content_hash(self) -> str:
        return canonical_content_hash(self.model_dump(mode="json"))


def assert_r0_prefix_compatible(proof: R0PrefixImportProof, expected: dict[str, Any]) -> None:
    # Revalidate model_copy inputs too; callers must not bypass the proof schema.
    R0PrefixImportProof.model_validate(proof.model_dump(mode="json"))
    if proof.compatibility != expected or proof.compatibility_hash != canonical_content_hash(
        expected
    ):
        raise ValueError("r0_prefix_import_scientific_identity_mismatch")


def publish_r0_prefix_proof(store: CampaignArtifactStore, proof: R0PrefixImportProof) -> str:
    assert_r0_prefix_compatible(proof, proof.compatibility)
    directory = store.root / "_store/upstream/r0-prefix" / proof.compatibility_hash
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise ValueError("r0_prefix_import_registry_symlink")
    digest = store.put_json(proof.model_dump(mode="json"))
    name = hashlib.sha256(proof.source_run_id.encode()).hexdigest() + ".json"
    write_json(
        directory / name,
        {"proof_sha256": digest, "source_run_id": proof.source_run_id},
        immutable=True,
    )
    return digest


def compatible_r0_prefix_proofs(
    store: CampaignArtifactStore, expected: dict[str, Any], *, exclude_run_id: str
) -> list[R0PrefixImportProof]:
    directory = store.root / "_store/upstream/r0-prefix" / canonical_content_hash(expected)
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise ValueError("r0_prefix_import_registry_symlink")
    proofs = []
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink():
            raise ValueError("r0_prefix_import_registry_symlink")
        entry = json.loads(path.read_bytes())
        proof = R0PrefixImportProof.model_validate(store.load_json(entry["proof_sha256"]))
        assert_r0_prefix_compatible(proof, expected)
        if proof.source_run_id != entry["source_run_id"]:
            raise ValueError("r0_prefix_import_registry_owner_mismatch")
        if proof.source_run_id != exclude_run_id:
            proofs.append(proof)
    return sorted(proofs, key=lambda value: (value.source_campaign_id, value.source_run_id))


def questioner_compatibility(
    campaign: BenchmarkCampaignManifest, scope: dict[str, Any]
) -> dict[str, Any]:
    """Exclude P/D/answer changes but retain all source and privacy dependencies."""
    protocol = campaign.protocol
    return {
        "version": IMPORT_VERSION,
        "import_policy_hash": sha256_file(Path(__file__)),
        "schema_revision": STEP8_ALEMBIC_REVISION,
        "compiler_schema": "0.9.0",
        "snapshot_id": scope["snapshot_id"],
        "dataset": scope["dataset"],
        "conversation_id": scope["conversation_id"],
        "partition_hash": protocol.partition_hash,
        "partition_role": "paper_test" if campaign.paper_result else "development",
        "diagnostic_only": campaign.diagnostic_only,
        "memory_raw_image_policy": (
            "forbid" if scope.get("treatment") == "no_raw_image" else "allow"
        ),
        "seed": protocol.seed,
        "questioner": protocol.questioner,
        "verifier": protocol.verifier,
        "questioner_config_hash": protocol.config_hashes["questioner_config"],
        # Foundation participates in registration/unit identity. Until it has a
        # separately proven boundary, an operator change cannot import Q facts.
        "foundation_config_hash": protocol.config_hashes["foundation_config"],
        "models": compiler_model_entry_hashes(protocol),
        "sources": {
            name: protocol.source_fingerprints[name] for name in ("unit_prepare", "questioner")
        },
    }


class QuestionerImportProof(BaseSchema):
    version: str = IMPORT_VERSION
    source_campaign_id: NonEmptyStr
    source_protocol_hash: NonEmptyStr
    source_run_id: NonEmptyStr
    compatibility: dict[str, Any]
    compatibility_hash: NonEmptyStr
    schema_revision: NonEmptyStr
    table_contract_hash: NonEmptyStr
    facts_hash: NonEmptyStr
    row_counts: dict[str, int]
    artifact_hashes: dict[str, str]
    eligible_count: int = Field(ge=0)
    unit_count: int = Field(gt=0)

    @model_validator(mode="after")
    def _identities(self) -> QuestionerImportProof:
        if (
            self.version != IMPORT_VERSION
            or self.schema_revision != STEP8_ALEMBIC_REVISION
            or self.compatibility_hash != canonical_content_hash(self.compatibility)
            or any(value < 0 for value in self.row_counts.values())
        ):
            raise ValueError("questioner_import_proof_identity_invalid")
        return self

    @property
    def content_hash(self) -> str:
        return canonical_content_hash(self.model_dump(mode="json"))


def assert_compatible(proof: QuestionerImportProof, expected: dict[str, Any]) -> None:
    QuestionerImportProof.model_validate(proof.model_dump(mode="json"))
    if proof.compatibility != expected or proof.compatibility_hash != canonical_content_hash(
        expected
    ):
        raise ValueError("questioner_import_scientific_identity_mismatch")


def publish_proof(store: CampaignArtifactStore, proof: QuestionerImportProof) -> str:
    digest = store.put_json(proof.model_dump(mode="json"))
    name = hashlib.sha256(proof.source_run_id.encode()).hexdigest() + ".json"
    write_json(
        store.root / "_store/upstream/questioner" / proof.compatibility_hash / name,
        {"proof_sha256": digest, "source_run_id": proof.source_run_id},
        immutable=True,
    )
    return digest


def compatible_proofs(
    store: CampaignArtifactStore, expected: dict[str, Any], *, exclude_run_id: str
) -> list[QuestionerImportProof]:
    directory = store.root / "_store/upstream/questioner" / canonical_content_hash(expected)
    if directory.is_symlink():
        raise ValueError("questioner_import_registry_symlink")
    proofs = []
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink():
            raise ValueError("questioner_import_registry_symlink")
        entry = json.loads(path.read_bytes())
        proof = QuestionerImportProof.model_validate(store.load_json(entry["proof_sha256"]))
        assert_compatible(proof, expected)
        if proof.source_run_id != entry["source_run_id"]:
            raise ValueError("questioner_import_registry_owner_mismatch")
        if proof.source_run_id != exclude_run_id:
            proofs.append(proof)
    return sorted(proofs, key=lambda value: (value.source_campaign_id, value.source_run_id))


def rebind_value(value: Any, identities: dict[str, str]) -> Any:
    """Rewrite exact owned identities only; never alter text or canonical locators."""
    if isinstance(value, str):
        return identities.get(value, value)
    if isinstance(value, list):
        return [rebind_value(item, identities) for item in value]
    if isinstance(value, dict):
        return {key: rebind_value(item, identities) for key, item in value.items()}
    return copy.deepcopy(value)


def remap_questioner_rows(
    rows: dict[str, list[dict[str, Any]]],
    *,
    primary_keys: dict[str, list[str]],
    source_run_id: str,
    target_run_id: str,
    proof_hash: str,
    identity_overrides: dict[str, str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Create a bijective namespace projection without synthesizing model output.

    Model calls retain the measured producer payload/usage and record cache reuse.
    Their input artifacts retain the original request identity, so cost accounting
    can resolve one physical dispatch instead of charging the adoption again.
    """
    identities = {source_run_id: target_run_id}
    for table, values in rows.items():
        keys = primary_keys[table]
        if len(keys) != 1:
            continue
        key = keys[0]
        if key == "run_id":
            continue
        for row in values:
            old = row[key]
            if not isinstance(old, str) or old in identities:
                raise ValueError("questioner_import_owned_identity_collision")
            identities[old] = (
                compilation_unit_instance_id(run_id=target_run_id, input_hash=row["input_hash"])
                if table == "compiler_units"
                else "q-adopt-" + canonical_content_hash([target_run_id, table, old])[:40]
            )
    if identity_overrides:
        if (
            not identity_overrides.keys() <= identities.keys()
            or source_run_id in identity_overrides
        ):
            raise ValueError("questioner_import_identity_override_invalid")
        identities.update(identity_overrides)
    if len(set(identities.values())) != len(identities):
        raise ValueError("questioner_import_target_identity_collision")
    rebound: dict[str, list[dict[str, Any]]] = {}
    for table, values in rows.items():
        rebound[table] = []
        for row in values:
            value = rebind_value(row, identities)
            # Native text/provider output is never an ownership reference, even
            # if a user string happens to equal an internal identifier exactly.
            for field in (
                "canonical_question",
                "paraphrase",
                "expected_answer",
                "claim_text",
                "support_quote",
                "rejection_reason",
                "parsed_output",
                "parse_error",
                "request_config",
                "object_key",
                "storage_uri",
                "error_summary",
            ):
                if field in row:
                    value[field] = copy.deepcopy(row[field])
            if "metadata" in value:
                value["metadata"] = {
                    **value["metadata"],
                    "upstream_import": {
                        "version": IMPORT_VERSION,
                        "source_run_id": source_run_id,
                        "source_row_hash": canonical_content_hash(row),
                        "proof_hash": proof_hash,
                        "source_primary_key": {key: row[key] for key in primary_keys[table]},
                    },
                }
            if table == "model_calls":
                # This is a consumption projection, not a second provider event.
                value["cache_hit"] = True
            if table == "experiment_jobs":
                value["claimed_by"] = None
                value["lease_expires_at"] = None
            rebound[table].append(value)
    return rebound, identities


async def try_r0_prefix_import(backend: Any, scope: dict[str, Any]) -> dict[str, Any] | None:
    """Prefer a complete committed prefix before registering a new Q owner."""
    from pmmc.experiments.step8_backend import compiler_instance_id
    from pmmc.infrastructure.database.constants import DatabaseRole
    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.step8_upstream_import import PostgresR0PrefixImportRepository
    from pmmc.infrastructure.object_store import create_runtime_object_store

    # Fixed-planner Q must still be adopted from its paired main component.
    if scope.get("treatment") == "fixed_planner":
        return None
    target = compiler_instance_id(backend.campaign, scope)
    expected = r0_prefix_compatibility(backend.campaign, scope)
    engine, sessions = create_session_factory(backend.settings, DatabaseRole.RUNTIME)
    try:
        repository = PostgresR0PrefixImportRepository(
            sessions, create_runtime_object_store(backend.settings)
        )
        registered, qlineage = await repository.target_registration(target)
        lineage = await repository.prefix_registration(target)
        if registered and qlineage is None:
            return None
        if lineage:
            proof = R0PrefixImportProof.model_validate(lineage["proof_json"])
        else:
            proofs = compatible_r0_prefix_proofs(backend.store, expected, exclude_run_id=target)
            if qlineage:
                proofs = [
                    p for p in proofs if p.questioner_proof.content_hash == qlineage["proof_hash"]
                ]
            if not proofs:
                return None
            proof = proofs[0]
        await repository.verify_source(proof, expected)
    finally:
        await engine.dispose()
    q_digest = backend.store.put_json(proof.questioner_proof.model_dump(mode="json"))
    return await backend._try_questioner_import(
        scope,
        required_proof_hash=q_digest,
        prefix_proof=proof,
    )


async def seal_r0_prefix(backend: Any, scope: dict[str, Any]) -> dict[str, Any]:
    from pmmc.experiments.step8_backend import compiler_instance_id
    from pmmc.infrastructure.database.constants import DatabaseRole
    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.step8_upstream_import import PostgresR0PrefixImportRepository
    from pmmc.infrastructure.object_store import create_runtime_object_store

    engine, sessions = create_session_factory(backend.settings, DatabaseRole.RUNTIME)
    try:
        repository = PostgresR0PrefixImportRepository(
            sessions, create_runtime_object_store(backend.settings)
        )
        try:
            proof = await repository.seal_prefix(
                run_id=compiler_instance_id(backend.campaign, scope),
                campaign_id=backend.campaign.campaign_id,
                protocol_hash=backend.campaign.protocol.content_hash,
                compatibility=r0_prefix_compatibility(backend.campaign, scope),
            )
        except ValueError as error:
            # Existing native runs may share earlier producers' memory collections.
            # They remain usable natively, but this bounded importer cannot own them.
            if str(error) in {
                "r0_prefix_import_collection_owner_or_identity_invalid",
                "r0_prefix_import_call_lineage_not_closed",
            }:
                return {"r0_prefix_import_available": False, "r0_prefix_import_reason": str(error)}
            raise
        return {"r0_prefix_import_proof_sha256": publish_r0_prefix_proof(backend.store, proof)}
    finally:
        await engine.dispose()


async def verified_r0_prefix_stage(
    backend: Any, scope: dict[str, Any], stage: str
) -> dict[str, Any] | None:
    from pmmc.experiments.step8_backend import compiler_instance_id
    from pmmc.infrastructure.database.constants import DatabaseRole
    from pmmc.infrastructure.database.engine import create_session_factory
    from pmmc.infrastructure.database.step8_upstream_import import PostgresR0PrefixImportRepository
    from pmmc.infrastructure.object_store import create_runtime_object_store

    run_id = compiler_instance_id(backend.campaign, scope)
    engine, sessions = create_session_factory(backend.settings, DatabaseRole.RUNTIME)
    try:
        repository = PostgresR0PrefixImportRepository(
            sessions, create_runtime_object_store(backend.settings)
        )
        lineage = await repository.verify_target(
            run_id, r0_prefix_compatibility(backend.campaign, scope)
        )
    finally:
        await engine.dispose()
    if lineage is None or stage not in R0_PREFIX_STAGES[1:]:
        return None  # D/projection must execute, even with a verified imported R0.
    proof = R0PrefixImportProof.model_validate(lineage["proof_json"])
    output = {
        "compiler_run_id": run_id,
        "phase": stage,
        "run_status": "running",
        "program_count": proof.row_counts["memory_programs"],
        "eligible_candidate_count": proof.questioner_proof.eligible_count,
        "r0_prefix_adoption": lineage,
        "verified_only": True,
    }
    if stage == "r0_gate":
        output.update(await seal_r0_prefix(backend, scope))
    return output
