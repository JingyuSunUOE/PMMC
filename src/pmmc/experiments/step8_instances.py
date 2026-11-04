"""Read-only runtime instance attestations, separate from method approval.

The caller supplies the compiler subprotocol hash derived from its protocol and
scope, persists the returned JSON in its artifact store, and verifies that same
artifact immediately before query execution. No query/evaluator repositories,
model calls, or artifact writes belong at this boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

from pydantic import ConfigDict, ValidationError, field_validator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.memory_index import make_memory_operator_index
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.schemas import CompilationTargetStage, DataScope, RunStatus
from pmmc.schemas.base import BaseSchema
from pmmc.schemas.compiler import CompilerRunSpec
from pmmc.schemas.doubter import QBankRoundProjectionManifest, QBankRoundProjectionMember
from pmmc.schemas.persistence import EmbeddingCollection
from pmmc.schemas.questioner import QBankIndexManifest
from pmmc.schemas.step8 import (
    CompiledQBankInstanceManifest,
    MethodProtocolFreeze,
    ScientificProtocol,
    Treatment,
)


class InstanceManifestError(ValueError):
    """Content-free failure code, safe to include in a component receipt."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise InstanceManifestError(code)


def _immutable(*args: Any, **kwargs: Any) -> NoReturn:
    raise TypeError("compiled instance manifest is immutable")


class _FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo: dict) -> _FrozenDict:
        return self


class _FrozenList(list):
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = _immutable
    reverse = sort = __iadd__ = __imul__ = _immutable

    def __deepcopy__(self, memo: dict) -> _FrozenList:
        return self


class _ImmutableInstance(CompiledQBankInstanceManifest):
    model_config = ConfigDict(frozen=True)

    @field_validator("round_membership_hashes", "collection_hashes")
    @classmethod
    def _freeze_maps(cls, value: dict) -> dict:
        return _FrozenDict(value)

    @field_validator("artifact_hashes")
    @classmethod
    def _freeze_list(cls, value: list) -> list:
        return _FrozenList(value)


def _scope(scope: Mapping[str, Any]) -> dict[str, str]:
    # Query IDs, answer models, budgets, and requested rounds are not compiler scope.
    required = ("dataset", "snapshot_id", "conversation_id", "treatment")
    _require(
        all(isinstance(scope.get(key), str) and scope[key].strip() for key in required),
        "instance_scope_missing",
    )
    try:
        treatment = Treatment(scope["treatment"])
    except ValueError:
        raise InstanceManifestError("instance_treatment_invalid") from None
    return {
        **{key: scope[key] for key in required if key != "treatment"},
        "treatment": Treatment.MAIN.value
        if treatment is Treatment.NO_RERANKER
        else treatment.value,
    }


def _payload(value: BaseSchema, *excluded: str) -> dict[str, Any]:
    return value.model_dump(mode="json", exclude=set(excluded))


def _fact_hash(value: BaseSchema) -> str:
    # Retain private metadata only behind a digest; exclude wall-clock provenance.
    return canonical_content_hash(_payload(value, "created_at"))


async def _imported_index_rows(uow: Any, compiler: Any, run: Any, protocol: Any, scope: dict):
    from pmmc.experiments.step8_cost_dependencies import DependencyUnproven
    from pmmc.experiments.step8_partial_cost_dependencies import (
        _collect_source_receipts,
        validate_partial_cost_dependencies,
    )
    from pmmc.experiments.step8_questioner_cost_dependencies import _NoSourceIO, _sealed_json
    from pmmc.infrastructure.database.step8_partial_import import (
        LINEAGE_KEY,
        PostgresPartialPrefixImportRepository,
    )

    lineage = compiler.metadata.get(LINEAGE_KEY)
    if lineage is None:
        return None
    try:
        proof = lineage["proof_json"]
        _require(
            run.metadata.get(LINEAGE_KEY) == lineage
            and proof["target_protocol_hash"] == protocol.content_hash
            and proof["compatibility_hash"] == protocol.config_hashes.get("partial_import_policy")
            and proof["source_scope"] == scope,
            "instance_import_scope_mismatch",
        )
        # Reuse the existing exact adoption proof in this same read-only snapshot.
        # No object-store reads, re-embedding, writes, or external transactions.
        forbidden: Any = _NoSourceIO()
        reader = PostgresPartialPrefixImportRepository(forbidden, forbidden)
        sh, sr = await reader._read_prefix(uow.session, proof["source_run_id"])
        th, tr = await reader._read_prefix(uow.session, compiler.run_id)
        facts = _sealed_json(
            {
                "source": {"headers": sh, "rows": sr},
                "target": {"headers": th, "rows": tr},
                **await _collect_source_receipts(uow.session, proof),
            }
        )
        _require(
            th["compiler"]["metadata"].get(LINEAGE_KEY) == lineage,
            "instance_import_registration_mismatch",
        )
        validate_partial_cost_dependencies(facts)
        return tr
    except InstanceManifestError:
        raise
    except (DependencyUnproven, ValueError, KeyError, TypeError, IndexError):
        raise InstanceManifestError("instance_import_identity_unproven") from None


def _index_payload(value: BaseSchema, imported: bool, *excluded: str) -> dict[str, Any]:
    payload = _payload(value, *excluded)
    if imported:
        # Only after exact proof validation; retain all scientific metadata.
        payload["metadata"].pop("partial_prefix_import", None)
    return payload


def _subprotocol_hash(
    protocol: ScientificProtocol, scope: dict[str, str], diagnostic_namespace: str | None = None
) -> str:
    from pmmc.experiments.step8_model_identity import compiler_model_entry_hashes

    # Match the backend's compiler-owned recipe, without importing its execution adapters.
    _require(
        diagnostic_namespace is None
        or (isinstance(diagnostic_namespace, str) and bool(diagnostic_namespace.strip())),
        "instance_diagnostic_namespace_invalid",
    )
    try:
        return canonical_content_hash(
            {
                **(
                    {"diagnostic_namespace": diagnostic_namespace}
                    if diagnostic_namespace is not None
                    else {}
                ),
                "snapshot": scope["snapshot_id"],
                "conversation": scope["conversation_id"],
                "treatment": scope["treatment"],
                "seed": protocol.seed,
                "questioner": protocol.config_hashes["questioner_config"],
                "planner": protocol.config_hashes["planner_config"],
                "doubter": protocol.config_hashes["doubter_config"],
                "operators": protocol.config_hashes["foundation_config"],
                "model_entries": compiler_model_entry_hashes(protocol),
                "compiler_sources": {
                    key: protocol.source_fingerprints[key]
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
            }
        )
    except (KeyError, ValueError):
        raise InstanceManifestError("instance_protocol_identity_missing") from None


def _r0_producer_hashes(compiler: Any) -> tuple[str, str]:
    questioner = {
        key: getattr(compiler, field)
        for key, field in {
            "config": "questioner_config_hash",
            "prompt": "questioner_prompt_hash",
            "renderer": "questioner_renderer_version",
            "parser": "questioner_parser_version",
            "gate": "candidate_gate_version",
            "dedupe": "dedupe_version",
        }.items()
    }
    planner = {
        key: getattr(compiler, field)
        for key, field in {
            "config": "planner_config_hash",
            "prompt": "planner_prompt_hash",
            "renderer": "planner_renderer_version",
            "parser": "planner_parser_version",
            "policy": "program_policy_version",
        }.items()
    }
    _require(
        all(questioner.values()) and all(planner.values()), "instance_producer_identity_missing"
    )
    return canonical_content_hash(questioner), canonical_content_hash(planner)


def _round(
    manifest: QBankRoundProjectionManifest,
    members: list[QBankRoundProjectionMember],
    *,
    compiler: Any,
    r0: Any,
    round_id: int,
) -> None:
    identity = {
        "run_id": compiler.run_id,
        "snapshot_id": compiler.snapshot_id,
        "round_id": round_id,
        "source_r0_manifest_id": r0.r0_manifest_id,
        "r0_facts_hash": r0.facts_hash,
    }
    _require(
        manifest.schema_version == "0.4.0"
        and manifest.projection_id == stable_compiler_id("qbank-round-projection", identity)
        and all(
            getattr(manifest, key) == value
            for key, value in identity.items()
            if key != "r0_facts_hash"
        )
        and manifest.metadata.get("r0_facts_hash") == r0.facts_hash,
        "instance_round_identity_mismatch",
    )
    _require(
        len(members) == manifest.member_count == r0.candidate_count
        and len({item.qbank_id for item in members}) == len(members),
        "instance_round_denominator_mismatch",
    )
    for member in members:
        payload = _payload(member, "projection_member_id", "content_hash", "metadata")
        payload["reason"] = member.metadata.get("reason_code", "")
        payload["terminal_state"] = {
            key: value for key, value in member.metadata.items() if key != "reason_code"
        }
        _require(
            member.run_id == compiler.run_id
            and member.projection_id == manifest.projection_id
            and member.content_hash == canonical_content_hash(payload)
            and member.projection_member_id
            == stable_compiler_id("qbank-round-projection-member", payload)
            and member.metadata.get("projection_terminal") is True
            and member.metadata.get("projection_round") == round_id
            and (member.passed_at_round is None or member.passed_at_round <= round_id)
            and (
                round_id != 0
                or (
                    member.effective_feedback_id is None
                    and member.metadata.get("r0_uses_doubter_facts") is False
                )
            ),
            "instance_round_member_invalid",
        )
    accepted = sum(item.outcome in {"accepted", "carried_forward"} for item in members)
    rejected = sum(item.outcome in {"rejected", "max_rounds_exhausted"} for item in members)
    _require(
        (manifest.accepted_count, manifest.rejected_count, manifest.pending_count)
        == (accepted, rejected, 0)
        and accepted + rejected == len(members)
        and sum(item.route_class == "answerable" for item in members)
        == r0.answerable_candidate_count
        and manifest.members_hash
        == canonical_content_hash(
            [
                item.model_dump(mode="json")
                for item in sorted(members, key=lambda item: item.qbank_id)
            ]
        )
        and manifest.content_hash
        == canonical_content_hash(
            _payload(manifest, "schema_version", "content_hash", "created_at", "metadata")
        ),
        "instance_round_hash_mismatch",
    )


async def build_compiled_qbank_instance_manifest(
    sessions: async_sessionmaker[AsyncSession],
    *,
    protocol: ScientificProtocol,
    compiler_run_id: str,
    compiler_subprotocol_hash: str,
    scope: Mapping[str, Any],
    diagnostic_namespace: str | None = None,
) -> CompiledQBankInstanceManifest:
    """Attest one completed compiler 0.9 instance using only runtime Postgres.

    ``compiler_subprotocol_hash`` must come from the caller's protocol recipe,
    never from the stored compiler. Missing facts fail closed. Persistence is the
    caller's responsibility; the return value is deeply immutable and conforms
    to ``CompiledQBankInstanceManifest``'s existing JSON schema.
    """
    normalized = _scope(scope)
    _require(
        len(compiler_subprotocol_hash) == 64
        and all(char in "0123456789abcdef" for char in compiler_subprotocol_hash),
        "instance_subprotocol_hash_invalid",
    )
    _require(
        compiler_subprotocol_hash == _subprotocol_hash(protocol, normalized, diagnostic_namespace),
        "instance_protocol_subhash_mismatch",
    )
    try:
        async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
            session = uow.session
            if session is None:
                raise InstanceManifestError("instance_runtime_session_missing")
            # A single database snapshot prevents mixing different projection generations.
            await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
            await session.execute(text("SET TRANSACTION READ ONLY"))
            return await _build(
                uow, protocol, compiler_run_id, compiler_subprotocol_hash, normalized
            )
    except (KeyError, ValidationError):
        raise InstanceManifestError("instance_runtime_fact_missing_or_invalid") from None


async def _build(
    uow: Any,
    protocol: ScientificProtocol,
    run_id: str,
    subprotocol_hash: str,
    scope: dict[str, str],
) -> CompiledQBankInstanceManifest:
    _require(
        all(
            getattr(uow, name, None) is not None
            for name in (
                "compiler_runs",
                "compiler_units",
                "runs",
                "doubter",
                "qbank_indexes",
                "memory_operator_indexes",
                "embeddings",
            )
        ),
        "instance_runtime_repository_missing",
    )
    compiler = await uow.compiler_runs.get(run_id)
    _require(compiler is not None, "instance_compiler_missing")
    compiler = CompilerRunSpec.model_validate(compiler.model_dump(mode="python"))
    run = await uow.runs.get_manifest(run_id)
    state = await uow.runs.get_state(run_id)
    _require(
        compiler.run_id == run_id
        and compiler.compiler_schema_version == "0.9.0"
        and compiler.target_stage == CompilationTargetStage.VERIFIED_QBANK
        and compiler.operator_set_version == protocol.operator_set == "pmmc-operators-v7"
        and compiler.active_round == protocol.active_round == 2
        and compiler.max_round == protocol.max_round == 3
        and run.seed == protocol.seed
        and state.run_id == run_id
        and state.status == RunStatus.COMPLETED
        and state.selected_count == 0,
        "instance_compiler_incompatible_or_incomplete",
    )
    _require(
        compiler.metadata.get("questioner_identity_kind") == "step8_method_protocol"
        and compiler.questioner_freeze_manifest_hash == subprotocol_hash
        and compiler.metadata.get("compiler_subprotocol_hash") == subprotocol_hash
        and run.metadata.get("compiler_subprotocol_hash") == subprotocol_hash,
        "instance_compiler_subprotocol_mismatch",
    )
    _require(
        run.run_id == run_id
        and run.dataset == scope["dataset"]
        and run.dataset_snapshot_ids == [scope["snapshot_id"]]
        and compiler.snapshot_id == scope["snapshot_id"]
        and compiler.metadata.get("conversation_ids") == [scope["conversation_id"]]
        and run.metadata.get("conversation_ids") == [scope["conversation_id"]]
        and bool(run.selection_hash)
        and compiler.metadata.get("selection_hash") == run.selection_hash,
        "instance_snapshot_or_scope_mismatch",
    )
    units = await uow.compiler_units.list_for_run(run_id)
    _require(
        len(units) == compiler.unit_count
        and sorted(item.ordinal for item in units) == list(range(compiler.unit_count))
        and all(
            item.run_id == run_id
            and item.snapshot_id == scope["snapshot_id"]
            and item.conversation_id == scope["conversation_id"]
            for item in units
        ),
        "instance_unit_scope_mismatch",
    )
    qbank = await uow.qbank_indexes.get_for_run(run_id)
    memory = await uow.memory_operator_indexes.get_for_run(run_id)
    r0 = await uow.doubter.get_r0_manifest(run_id)
    _require(
        qbank is not None and memory is not None and r0 is not None, "instance_index_or_r0_missing"
    )
    _require(
        qbank.status == memory.status == "ready"
        and qbank.compiler_run_id == memory.run_id == run_id
        and qbank.snapshot_id == memory.snapshot_id == compiler.snapshot_id
        and memory.dataset == scope["dataset"]
        and memory.selection_hash == run.selection_hash,
        "instance_index_not_ready_or_out_of_scope",
    )
    _require(
        qbank.gate_version == compiler.candidate_gate_version
        and qbank.matcher_version == compiler.qbank_matcher_version
        and memory.text_model_alias == compiler.memory_text_model_alias
        and memory.text_model_revision == compiler.memory_text_model_revision
        and memory.multimodal_model_alias == compiler.memory_multimodal_model_alias
        and memory.multimodal_model_revision == compiler.memory_multimodal_model_revision,
        "instance_index_producer_mismatch",
    )
    qbank_channels = {
        key: getattr(qbank, key + "_collection_id")
        for key in (
            "text_question",
            "text_paraphrase",
            "multimodal_question",
            "multimodal_paraphrase",
        )
    }
    recipe = {
        **_payload(
            qbank,
            "qbank_index_id",
            "content_hash",
            "status",
            "created_at",
            "metadata",
            *(key + "_collection_id" for key in qbank_channels),
        ),
        "collections": qbank_channels,
    }
    imported = await _imported_index_rows(uow, compiler, run, protocol, scope)
    if imported is None:
        _require(
            qbank.content_hash == canonical_content_hash(recipe)
            and qbank.qbank_index_id == stable_compiler_id("qbank-index", recipe),
            "instance_qbank_index_hash_mismatch",
        )
    else:
        _require(
            qbank == QBankIndexManifest.model_validate(imported["qbank_index_manifests"][0])
            and memory
            == type(memory).model_validate(imported["memory_operator_index_manifests"][0]),
            "instance_import_index_fact_mismatch",
        )
    expected_memory, expected_text, expected_visual = make_memory_operator_index(
        run_id=run_id,
        snapshot_id=compiler.snapshot_id,
        dataset=scope["dataset"],
        selection_hash=run.selection_hash,
        text_model_alias=memory.text_model_alias,
        text_model_revision=memory.text_model_revision,
        multimodal_model_alias=memory.multimodal_model_alias,
        multimodal_model_revision=memory.multimodal_model_revision,
        memory_raw_image_policy="forbid" if scope["treatment"] == "no_raw_image" else "allow",
    )
    _require(
        _index_payload(memory, imported is not None, "status", "created_at")
        == _payload(expected_memory, "status", "created_at"),
        "instance_memory_index_hash_mismatch",
    )
    r0_payload = _payload(
        r0, "r0_manifest_id", "schema_version", "content_hash", "created_at", "metadata"
    )
    questioner_hash, planner_hash = _r0_producer_hashes(compiler)
    _require(
        r0.run_id == run_id
        and r0.snapshot_id == compiler.snapshot_id
        and r0.schema_version == "0.4.0"
        and r0.metadata.get("r0_input_policy_version")
        == compiler.r0_input_policy_version
        == "r0-input-gate-v4"
        and r0.hard_gate_passed
        and not r0.issue_codes
        and not r0.protocol_integrity_issues
        and r0.content_hash == canonical_content_hash(r0_payload)
        and r0.r0_manifest_id == stable_compiler_id("r0-input-manifest", r0_payload)
        and r0.memory_index_hash == memory.content_hash
        and r0.operator_registry_hash == compiler.operator_registry_hash
        and r0.questioner_identity_hash == questioner_hash
        and r0.planner_identity_hash == planner_hash
        and r0.artifact_hash == canonical_content_hash([qbank.content_hash, memory.content_hash])
        and r0.candidate_count == qbank.eligible_question_count,
        "instance_r0_hash_or_gate_mismatch",
    )
    rounds = {}
    for round_id in range(4):
        projection = await uow.doubter.get_round_projection(run_id, round_id)
        _require(projection is not None, "instance_round_missing")
        manifest, members = projection
        _round(manifest, members, compiler=compiler, r0=r0, round_id=round_id)
        rounds[round_id] = projection
    denominator = {(item.qbank_id, item.route_class) for item in rounds[0][1]}
    _require(
        all(
            {(item.qbank_id, item.route_class) for item in members} == denominator
            for _, members in rounds.values()
        ),
        "instance_round_membership_mismatch",
    )
    routing = await uow.doubter.get_routing_projection(run_id)
    _require(routing is not None, "instance_routing_missing")
    routing_manifest, routing_members = routing
    active, active_members = rounds[2]
    accepted = {
        item.qbank_id: item
        for item in active_members
        if item.outcome in {"accepted", "carried_forward"}
    }
    ordered = sorted(routing_members, key=lambda item: item.ordinal)
    _require(
        routing_manifest.run_id == run_id
        and routing_manifest.snapshot_id == compiler.snapshot_id
        and routing_manifest.schema_version == "0.4.0"
        and routing_manifest.active_round == 2
        and routing_manifest.max_round == 3
        and routing_manifest.source_projection_id == active.projection_id
        and routing_manifest.routing_projection_id
        == stable_compiler_id(
            "routing-projection", [run_id, active.projection_id, routing_manifest.decision_hash]
        )
        and routing_manifest.text_collection_id == qbank.text_question_collection_id
        and routing_manifest.multimodal_collection_id == qbank.multimodal_question_collection_id
        and routing_manifest.lexical_collection_id
        == stable_compiler_id(
            "qbank-lexical-projection", [run_id, qbank.qbank_index_id, active.projection_id]
        )
        and routing_manifest.member_count == len(ordered) == len(accepted)
        and {item.qbank_id for item in ordered} == set(accepted)
        and [item.ordinal for item in ordered] == list(range(len(ordered))),
        "instance_routing_identity_mismatch",
    )
    for item in ordered:
        source = accepted[item.qbank_id]
        _require(
            item.run_id == run_id
            and item.routing_projection_id == routing_manifest.routing_projection_id
            and item.plan_id == source.effective_plan_id
            and item.route_class == source.route_class
            and item.source_projection_member_id == source.projection_member_id
            and item.routing_member_id
            == stable_compiler_id(
                "routing-projection-member",
                [routing_manifest.routing_projection_id, item.qbank_id, item.ordinal],
            )
            and item.content_hash
            == canonical_content_hash(
                [
                    item.routing_projection_id,
                    item.qbank_id,
                    item.plan_id,
                    item.route_class.value,
                    item.ordinal,
                ]
            ),
            "instance_routing_member_invalid",
        )
    _require(
        routing_manifest.members_hash
        == canonical_content_hash([item.model_dump(mode="json") for item in ordered])
        and routing_manifest.answerable_count
        == sum(item.route_class == "answerable" for item in ordered)
        and routing_manifest.control_count == sum(item.route_class == "control" for item in ordered)
        and routing_manifest.content_hash
        == canonical_content_hash(
            _payload(routing_manifest, "schema_version", "content_hash", "created_at", "metadata")
        ),
        "instance_routing_hash_mismatch",
    )
    collection_hashes = {}
    memory_collections = {
        expected_text.collection_id: expected_text,
        expected_visual.collection_id: expected_visual,
    }
    collection_ids = [*qbank_channels.values(), *memory_collections]
    _require(len(set(collection_ids)) == 6, "instance_collection_identity_collision")
    for collection_id in sorted(collection_ids):
        collection = await uow.embeddings.get_collection(collection_id)
        _require(collection is not None, "instance_collection_missing")
        _require(
            collection.collection_id == collection_id
            and collection.status == "ready"
            and collection.snapshot_id == compiler.snapshot_id,
            "instance_collection_not_ready_or_out_of_scope",
        )
        if imported is not None:
            expected = [
                r for r in imported["embedding_collections"] if r["collection_id"] == collection_id
            ]
            _require(
                len(expected) == 1
                and collection == EmbeddingCollection.model_validate(expected[0]),
                "instance_import_collection_fact_mismatch",
            )
        if collection_id in memory_collections:
            # Memory vectors are shared by scientific scope, not by compiler run.
            _require(
                _index_payload(
                    collection,
                    imported is not None,
                    "run_id",
                    "created_at",
                    "status",
                    "index_type",
                    "index_params",
                )
                == _payload(
                    memory_collections[collection_id],
                    "run_id",
                    "created_at",
                    "status",
                    "index_type",
                    "index_params",
                ),
                "instance_memory_collection_mismatch",
            )
        else:
            _require(collection.run_id == run_id, "instance_qbank_collection_scope_mismatch")
        collection_hashes[collection_id] = _fact_hash(collection)
    # The lexical projection is virtual, not an embedding collection row.
    collection_hashes[routing_manifest.lexical_collection_id] = routing_manifest.content_hash
    artifacts = [
        subprotocol_hash,
        _fact_hash(compiler),
        _fact_hash(qbank),
        _fact_hash(memory),
        qbank.content_hash,
        memory.content_hash,
        r0.content_hash,
        r0.artifact_hash,
        routing_manifest.content_hash,
        *(manifest.content_hash for manifest, _ in rounds.values()),
    ]
    return _ImmutableInstance(
        protocol_hash=protocol.content_hash,
        dataset=scope["dataset"],
        snapshot_id=compiler.snapshot_id,
        compiler_run_id=run_id,
        scope_hash=canonical_content_hash(scope),
        treatment=scope["treatment"],
        round_membership_hashes={key: value[0].members_hash for key, value in rounds.items()},
        collection_hashes=collection_hashes,
        artifact_hashes=sorted(set(artifacts)),
        r0_manifest_hash=r0.content_hash,
    )


async def verify_compiled_qbank_instance_manifest(
    sessions: async_sessionmaker[AsyncSession],
    *,
    manifest: CompiledQBankInstanceManifest,
    protocol: ScientificProtocol,
    compiler_run_id: str,
    compiler_subprotocol_hash: str,
    scope: Mapping[str, Any],
    expected_manifest_hash: str,
    freeze: MethodProtocolFreeze | None = None,
    paper: bool = True,
    diagnostic_namespace: str | None = None,
) -> CompiledQBankInstanceManifest:
    """Fail closed before querying; approval binds the protocol, not a dev run.

    ``expected_manifest_hash`` is the canonical content hash saved in the parent
    receipt (not necessarily the byte hash used by its artifact store). Paper
    callers must pass their campaign's freeze artifact. Development callers must
    explicitly set ``paper=False``. The verified immutable manifest is returned.
    """
    if paper:
        _require(diagnostic_namespace is None, "instance_diagnostic_namespace_not_paper")
        if freeze is None:
            raise InstanceManifestError("instance_protocol_approval_missing")
        _require(
            freeze.decision == "approved"
            and freeze.protocol_integrity_passed
            and freeze.scientific_viability_passed
            and freeze.exposure_complete,
            "instance_protocol_not_approved",
        )
        _require(
            freeze.protocol.content_hash == protocol.content_hash,
            "instance_approved_protocol_mismatch",
        )
    _require(manifest.protocol_hash == protocol.content_hash, "instance_protocol_mismatch")
    _require(manifest.content_hash == expected_manifest_hash, "instance_artifact_hash_mismatch")
    current = await build_compiled_qbank_instance_manifest(
        sessions,
        protocol=protocol,
        compiler_run_id=compiler_run_id,
        compiler_subprotocol_hash=compiler_subprotocol_hash,
        scope=scope,
        diagnostic_namespace=diagnostic_namespace,
    )
    _require(current.content_hash == expected_manifest_hash, "instance_runtime_facts_changed")
    return current


__all__ = [
    "InstanceManifestError",
    "build_compiled_qbank_instance_manifest",
    "verify_compiled_qbank_instance_manifest",
]
