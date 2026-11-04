"""Offline scientific and production-path checks for the release R channel."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation.step8_relevance_grid import evaluate_relevance_grid_answers
from pmmc.evaluation.step8_typed_gold import resolve_typed_gold
from pmmc.experiments.step8_relevance_proofs import _verify_point_id, relevance_point
from pmmc.runtime import joint_embeddings, query_backend, query_workflow
from pmmc.runtime.answer_context import PmmcEvidenceAnswerInputProvider
from pmmc.runtime.answering import validate_typed_answer_input
from pmmc.runtime.joint_embeddings import JOINT_MODEL, joint_shards, visible_joint_pairs
from pmmc.runtime.query_backend import PostgresOnlineQueryExecutorBackend
from pmmc.runtime.query_executor import OnlineQueryExecutor
from pmmc.runtime.query_executor_config import (
    JOINT_POLICY,
    QueryExecutorConfig,
    load_query_executor_config,
    query_executor_config_hash,
)
from pmmc.runtime.relevance_grid import (
    bind_joint_context,
    bind_runtime_relation_contract,
    evidence_reference_hash,
    project_runtime_relevance_grid,
    validate_joint_outcome,
)
from pmmc.schemas import (
    DataScope,
    EvidenceType,
    ImageRef,
    MemoryUnit,
    PlannerRequiredModality,
    PlannerRequirementContract,
    QueryGold,
    QueryInput,
    QueryProgramBinding,
)
from pmmc.schemas.answer_output import TYPED_ANSWER_CONTRACT_VERSION, GroundingCheck
from test_step8_query_cache import UnitOfWork, cache_harness, registry  # noqa: F401
from test_step8_runtime import PhasedBackend, hit

CONFIGS = Path(__file__).resolve().parents[1] / "configs/runtime"
R = load_query_executor_config(CONFIGS / "query-executor-release-r.yaml")


def image(name):
    return ImageRef(
        image_id=name,
        storage_uri=f"s3://runtime/{name}",
        sha256=canonical_content_hash(name),
        caption="do not embed this caption",
    )


def memory(name, refs, text="owner words"):
    return MemoryUnit(
        memory_id=name,
        conversation_id="conversation-1",
        session_id="session-2",
        dataset="memlens",
        turn_id=name,
        timestamp_order=1,
        available_after_session="session-2",
        text=text,
        caption="caption only",
        image_refs=refs,
    )


def test_protocol_is_opt_in_and_historical_identities_do_not_drift():
    for version, frozen in {
        "v1": "10ecbe347cc9ad6f789c0f9027e37d1306d63fa188edd2939d1e06326491873d",
        "v2": "9905bb67135e6d0eb3e1a6681b9354b3fb287fdd31407f0d7dcc25c26c29dec4",
        "v3": "2bc4b33f26aba93b1ccfa99c6efbe7b5d9f109b59d10a4391adcce72ac0c37d4",
    }.items():
        old = load_query_executor_config(CONFIGS / f"query-executor-{version}.yaml")
        assert not old.joint_relevance_enabled and query_executor_config_hash(old) == frozen
        assert "joint_relevance_policy" not in old.model_dump()
    assert R.is_v3 and R.joint_relevance_enabled and query_executor_config_hash(R) != frozen
    assert not R.model_copy(update={"treatment": "no_raw_image"}).joint_relevance_enabled
    with pytest.raises(ValueError, match="independent frozen"):
        QueryExecutorConfig.model_validate({**R.model_dump(), "joint_relevance_threshold": 0.2})


def test_pair_ownership_uses_real_assets_not_captions_neighbors_or_query_images():
    a, b, q = map(image, ("a", "b", "query"))
    query = QueryInput(
        query_id="q",
        conversation_id="conversation-1",
        dataset="memlens",
        raw_query="never embed this question as history",
        query_image_refs=[q],
    )
    pairs, reasons = visible_joint_pairs(
        "snapshot",
        query,
        [
            memory("owner", [a, b, q, q.model_copy(update={"image_id": "alias"})]),
            memory("ambiguous", [b]),
            memory("caption", [image("c")], None),
            memory("bare", [ImageRef(image_id="bare")]),
            memory("other turn", [], "unrelated"),
        ],
    )
    assert reasons == {
        "b": "ambiguous_owner",
        "query": "query_image",
        "alias": "query_image",
        "c": "no_owner_text",
        "bare": "no_canonical_asset",
    }
    assert [(p.owner_id, p.image_id) for p in pairs] == [("owner", "a")]
    native = pairs[0].native_input
    assert native.text == "owner words" and [i.image_id for i in native.image_refs] == ["a"]
    assert native.image_refs[0].caption is None
    entry = registry().get(JOINT_MODEL)
    key = next(joint_shards("snapshot", entry, pairs))[1]
    changed, _ = visible_joint_pairs("snapshot", query, [memory("owner", [a], "revised")])
    assert next(joint_shards("snapshot", entry, changed))[1] != key
    assert (
        next(joint_shards("snapshot", entry.model_copy(update={"model_revision": "next"}), pairs))[
            1
        ]
        != key
    )


def test_joint_input_rejects_query_asset_aliases_and_does_not_journal_alternate_locators():
    qref = image("query-photo").model_copy(
        update={
            "sha256": None,
            "asset_id": "shared",
            "remote_url": "https://invalid/?QUERY_SENTINEL",
            "local_path": "/tmp/QUERY_SENTINEL",
        }
    )
    query = QueryInput(
        query_id="q",
        dataset="memlens",
        conversation_id="conversation-1",
        raw_query="query",
        query_image_refs=[qref, image("q2")],
    )
    private = image("history").model_copy(
        update={
            "local_path": "/PRIVATE_SENTINEL/path",
            "remote_url": "https://invalid/?PRIVATE_SENTINEL",
            "metadata": {"source": "PRIVATE_SENTINEL"},
            "caption": "PRIVATE_SENTINEL",
        }
    )
    refs = [
        private,
        image("asset-alias").model_copy(update={"asset_id": "shared"}),
        image("uri-alias").model_copy(update={"storage_uri": qref.storage_uri}),
        image("sha-alias").model_copy(update={"sha256": image("q2").sha256.upper()}),
        image("remote-alias").model_copy(update={"remote_url": qref.remote_url}),
        image("local-alias").model_copy(update={"local_path": qref.local_path}),
        image("local-only").model_copy(update={"storage_uri": None, "local_path": "/tmp/photo"}),
        image("signed").model_copy(update={"storage_uri": "s3://bucket/key?PRIVATE_SENTINEL"}),
    ]
    pairs, excluded = visible_joint_pairs("snapshot", query, [memory("owner", refs)])
    assert excluded == {
        "asset-alias": "query_image",
        "uri-alias": "query_image",
        "sha-alias": "query_image",
        "remote-alias": "query_image",
        "local-alias": "query_image",
        "local-only": "no_canonical_asset",
        "signed": "no_canonical_asset",
    }
    assert len(pairs) == 1 and pairs[0].image_id == "history"
    assert pairs[0].native_input.image_refs[0].storage_uri == private.storage_uri
    assert "PRIVATE_SENTINEL" not in pairs[0].native_input.model_dump_json()
    with pytest.raises(ValueError, match="unsupported joint"):
        validate_joint_outcome({"joint_policy": "visible-image-owner-text-v1"}, {})


class Visible:
    def __init__(self, snapshot="snapshot-1", conversation="conversation-1"):
        self.scope = (snapshot, conversation, "session-2")
        self.refs = [image(name) for name in ("photo", "sibling")]

    async def iter_visible(self, snapshot, conversation, cutoff):
        assert (snapshot, conversation, cutoff) == self.scope
        yield memory("owner", self.refs).model_copy(update={"conversation_id": conversation})

    async def get_many(self, snapshot, ids):
        assert snapshot == self.scope[0]
        return [ref for ref in self.refs if ref.image_id in ids]


@pytest.mark.usefixtures("cache_harness")
def test_preparation_builds_separate_recoverable_cache_without_replacing_raw(
    request,
    monkeypatch,
    tmp_path,
):
    h = request.getfixturevalue("cache_harness")
    h.facts.queries = [
        q.model_copy(update={"available_after_session": "session-2"}) for q in h.facts.queries
    ]
    monkeypatch.setattr(
        query_workflow, "PostgresMemoryRepository", lambda _: Visible("snapshot", "conversation")
    )
    monkeypatch.setattr(joint_embeddings, "ExperimentUnitOfWork", UnitOfWork)

    async def get_records(self, ids):
        return {key: self.facts.records[key] for key in ids}

    monkeypatch.setattr(UnitOfWork, "get_records", get_records, raising=False)

    async def run():
        await h.prepare(["q1"], channel="multimodal")
        raw = dict(h.facts.records)
        params = dict(
            settings=query_workflow.Settings(_env_file=None, project_root=tmp_path),
            registry=registry(),
            source_compiler_run_id="different-compiler",
            snapshot_id="snapshot",
            query_ids=["q1"],
            channel="multimodal",
            joint_evidence=True,
        )
        receipts = await asyncio.gather(
            *(query_workflow.prepare_query_executor_runtime_embeddings(**params) for _ in range(2))
        )
        first, second = sorted(receipts, key=lambda r: r["joint_model_call_count"], reverse=True)
        assert first["reused_query_count"] == 1 and first["joint_model_call_count"] == 1
        assert second["joint_model_call_count"] == 0 and second["joint_reused_batch_count"] == 1
        assert len(h.facts.calls) == 2 and all(h.facts.records[k] == v for k, v in raw.items())
        owner, request = h.facts.calls[-1]
        assert second["joint_producers"] == first["joint_producers"]
        assert list(first["joint_producers"].values()) == [
            {"run_id": owner, "model_call_id": "call-" + request.request_id}
        ]
        assert (
            request.model_alias == JOINT_MODEL and request.metadata["joint_policy"] == JOINT_POLICY
        )
        assert [(v.text, len(v.image_refs)) for v in request.inputs] == [("owner words", 1)] * 2
        for key in h.facts.records.keys() - raw.keys():
            row = h.facts.records[key]
            assert h.facts.collections[row.collection_id].run_id == owner
            assert row.model_call_id == "call-" + request.request_id
            h.facts.records[key] = row.model_copy(update={"model_call_id": "call-foreign-producer"})
        with pytest.raises(ValueError, match="incompatible scientific lineage"):
            await query_workflow.prepare_query_executor_runtime_embeddings(**params)
        assert len(h.facts.calls) == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    "budget,grounding_fails,expected",
    [
        (True, False, "answered"),
        (True, True, "fallback_required"),
        (False, False, "context_rejected"),
    ],
)
def test_actual_scores_generated_phases_grounding_and_budget(
    monkeypatch, budget, grounding_fails, expected
):
    from test_step8_runtime import typed_answer

    visible = Visible()

    class Counter:
        revision = "fixture"
        estimated = False

        def count_request(self, request):
            return (
                33000
                if not budget
                and any(
                    source["images"]
                    for source in request.metadata["typed_answer_context"]["sources"]
                )
                else 100
            )

    native = PmmcEvidenceAnswerInputProvider(
        memories=visible,
        images=visible,
        counter=Counter(),
        instruction="unused",
        answer_contract_version=TYPED_ANSWER_CONTRACT_VERSION,
    )
    scorer = object.__new__(PostgresOnlineQueryExecutorBackend)
    scorer.config, scorer.answer_inputs, scorer.joint_entry = R, native, registry().get(JOINT_MODEL)
    scorer.manifest = SimpleNamespace(snapshot_id="snapshot-1", query_executor_run_id="query-run-1")
    scorer._query_vectors = AsyncMock(
        return_value={
            channel: (SimpleNamespace(embedding_id="query-" + channel), [1.0, 0.0])
            for channel in ("text", "multimodal")
        }
    )
    rows = [
        SimpleNamespace(
            source_id=key,
            embedding_id=key,
            embedding=vec,
            collection_id="raw",
            model_call_id="raw-call",
        )
        for key, vec in (
            ("memory-text:owner", [1.0, 0.0]),
            ("photo", [0.0, 1.0]),
            ("sibling", [0.0, 1.0]),
        )
    ]

    @asynccontextmanager
    async def session():
        yield SimpleNamespace(scalars=AsyncMock(return_value=rows))

    scorer.reader_sessions = session

    async def joint_records(_sessions, collection, pairs):
        return {
            p.image_id: SimpleNamespace(
                vector=[1.0, 0.0] if p.image_id == "photo" else [-1.0, 0.0],
                input_hash=p.input_hash,
                collection_id=collection.collection_id,
                embedding_id=p.input_hash,
                model_call_id="native-call",
                metadata={"owner_id": p.owner_id},
            )
            for p in pairs
        }

    monkeypatch.setattr(query_backend, "load_joint_records", joint_records)

    class Backend(PhasedBackend):
        relevance_grid_enabled = True
        execute_full_grid = False

        async def bind_frozen_program(self, **kwargs):
            frozen = await super().bind_frozen_program(**kwargs)
            req = frozen.requirement.model_copy(
                update={"required_modalities": [PlannerRequiredModality.IMAGE]}
            )
            return replace(
                frozen,
                requirement=req,
                binding=frozen.binding.model_copy(
                    update={"requirement_contract_hash": req.content_hash}
                ),
            )

        async def execute_memory_program(self, query, frozen):
            result = await super().execute_memory_program(query, frozen)
            head = result.evidence[0].model_copy(
                update={"memory_id": "owner", "evidence_id": "owner"}
            )
            return replace(
                result,
                evidence=(
                    head,
                    *(
                        head.model_copy(
                            update={
                                "evidence_ref_id": "ref-" + i.image_id,
                                "evidence_type": EvidenceType.IMAGE,
                                "memory_id": None,
                                "image_id": i.image_id,
                                "evidence_id": i.image_id,
                                "rank": n,
                            }
                        )
                        for n, i in enumerate(visible.refs, 2)
                    ),
                ),
            )

        relevant_runtime_evidence_groups = scorer.relevant_runtime_evidence_groups
        prepare_typed_pmmc_answer = scorer.prepare_typed_pmmc_answer

        async def generate_deferred_typed_answer(self, request):
            context = validate_typed_answer_input(request)
            assert [i.image_id for i in context.image_bindings] == ["photo"]
            assert [s.evidence_id for s in context.sources] == ["owner", "photo"]
            return await super().generate_deferred_typed_answer(request)

        async def ground_prepared_typed_answer(self, request, reference):
            answer = await super().ground_prepared_typed_answer(request, reference)
            if not grounding_fails:
                return answer
            denied = answer.grounding.assessment.model_copy(
                update={"supported": False, "source_support": GroundingCheck.FAIL}
            )
            return answer.model_copy(
                update={
                    "grounding": answer.grounding.model_copy(update={"assessment": denied}),
                    "terminal_state": "fallback_required",
                    "output": typed_answer("x", abstain=True).output,
                }
            )

    async def run():
        backend = Backend([hit(round_id=2)])
        executor = OnlineQueryExecutor(backend=backend, config=R)
        ids = {"run_id": "query-run-1", "query_id": "query-1"}
        match = await executor.execute(**ids, phase="match")
        fact = next(f for f in match["cell_results"] if f.cell.treatment == "pmmc" and f.binding)
        assert fact.error_type is None
        items = [
            (i.metadata["embedding_channel"], i.cosine_similarity) for i in fact.relevance_items
        ]
        assert sorted(items) == sorted(
            [("text", 1), ("multimodal", 0), ("multimodal", 0), ("joint", 1), ("joint", -1)]
        )
        for item in fact.relevance_items:
            if item.metadata["embedding_channel"] == "joint":
                assert item.metadata["image_id_hash"] == item.evidence_id_hash
                assert item.metadata["owner_id_hash"] == canonical_content_hash("owner")
                assert {"owner_id", "image_id"}.isdisjoint(item.metadata)
        assert not backend.events
        await executor.execute(**ids, phase="answer")
        terminal = await executor.execute(**ids, phase="select")
        cell = next(f for f in terminal["cell_results"] if f.cell.treatment == "pmmc" and f.binding)
        for point in cell.metadata["relevance_grid"]:
            _verify_point_id(point)
            validate_joint_outcome(point, cell.metadata["joint_contexts"])
            assert point["selection_channels"][canonical_content_hash("ref-sibling")] == []
            if budget:
                assert point["status"] == "completed" and point["terminal_state"] == expected
                assert point["arrived_joint_image_hashes"] == [canonical_content_hash("photo")]
                assert point["answer_support"] is (not grounding_fails)
                assert cell.active_fallback is grounding_fails
            else:
                assert point["status"] == expected and not point["arrived_joint_image_hashes"]
                assert cell.active_fallback and point["answer_support"] is None
        assert [e[0] for e in backend.events] == (["answer", "ground"] if budget else [])
        query = terminal["query"]
        grid = await evaluate_relevance_grid_answers(
            query=query,
            scope=DataScope.EVALUATION,
            snapshot_id="snapshot-1",
            query_run_id=ids["run_id"],
            cells=[cell],
            answers=terminal["typed_answers"],
            thresholds=R.relevance_cosine_grid,
            resolution=resolve_typed_gold(query, QueryGold(answer="Blue")),
            judge=AsyncMock(return_value={"value": 1.0, "model_call_id": "offline-judge"}),
        )
        assert grid["status"] == "completed"
        checked, _ = relevance_point(cell, {"relevance_grid": grid}, 0.3, 0.3)
        assert (
            checked["arrived_joint_image_hashes"] == grid["points"][0]["arrived_joint_image_hashes"]
        )

    asyncio.run(run())


def test_shared_match_rechecks_target_budget_instead_of_inheriting_rejection(monkeypatch):
    from itertools import product

    import test_step8_query_backbone_context as shared
    from test_step8_query_backbone_context import SOURCE, TARGET, harness

    monkeypatch.setattr(shared, "CONFIG", R)
    case = harness(monkeypatch, "openai")
    checkpoint = case.state.checkpoint
    binding = QueryProgramBinding.model_validate(checkpoint["cells"][0]["binding"])
    req = PlannerRequirementContract(
        requirement_contract_id="req",
        run_id="compiler",
        compiler_unit_id="unit",
        qbank_id=binding.qbank_id,
        required_modalities=["text"],
        minimum_evidence_groups=2,
        required_capabilities=[],
        fallback_required=True,
        content_hash=binding.requirement_contract_hash,
    )
    source = bind_joint_context(
        bind_runtime_relation_contract(
            case.source_input,
            result=case.state.execution,
            minimum_groups=2,
            relation_required=True,
        ),
        result=case.state.execution,
        binding=binding,
        requirement=req,
        joint_image_hashes=[],
    )
    assert not source.metadata["joint_context"]["context_sufficient"]
    key = checkpoint["cells"][0]["cell_id"]
    checkpoint["prepared"] = {key: source.model_dump(mode="json")}
    refs = [evidence_reference_hash(ref) for ref in case.state.execution.evidence]
    points = [
        {
            "protocol": "runtime-relevance-context-grid-v1",
            "joint_policy": JOINT_POLICY,
            "joint_image_hashes": [],
            "arrived_joint_image_hashes": [],
            "status": "pending_answer",
            "query_executor_run_id": SOURCE,
            "query_id": case.query.query_id,
            "source_cell_id": key,
            "execution_id": "execution",
            "grid_point_id": f"id-{t}-{v}",
            "text_threshold": t,
            "multimodal_threshold": v,
            "selected_reference_hashes": refs,
            "evidence_subset_hash": canonical_content_hash(refs),
            "prepared_key": key,
        }
        for t, v in product(R.relevance_cosine_grid, repeat=2)
    ]
    for cell in checkpoint["cells"]:
        cell["metadata"]["relevance_grid"] = points
        cell["metadata"]["joint_contexts"] = {key: source.metadata["joint_context"]}
    original = canonical_content_hash(checkpoint)
    asyncio.run(case.backend.reuse_matching_phase(TARGET, case.query.query_id))
    saved = case.persisted.call_args.args[3]
    for cell in saved["cells"]:
        receipt = cell["metadata"]["joint_contexts"][key]
        assert receipt["context_sufficient"] and len(receipt["supplied_reference_hashes"]) >= 2
        assert receipt["prepared_joint_image_hashes"] == []
    assert canonical_content_hash(checkpoint) == original and case.state.reads == ["execution"]


def test_joint_threshold_is_fixed_across_grid_and_does_not_mask_a_missing_raw_score():
    from test_step8_relevance_grid import fixture, score

    _, _, _, frozen, execution, items = asyncio.run(fixture())
    ref = execution.evidence[1].model_copy(
        update={
            "memory_id": None,
            "image_id": "photo",
            "evidence_type": EvidenceType.IMAGE,
        }
    )
    execution = replace(execution, evidence=(execution.evidence[0], ref))
    raw = score(execution, ref, -0.1, image=True)
    raw = raw.model_copy(update={"metadata": {**raw.metadata, "embedding_channel": "multimodal"}})
    text = items[0].model_copy(
        update={"metadata": {**items[0].metadata, "embedding_channel": "text"}}
    )

    def project(cosine, *, missing_raw=False):
        native = score(execution, ref, cosine, image=True)
        native = native.model_copy(
            update={
                "metadata": {
                    **native.metadata,
                    "embedding_channel": "joint",
                    "image_id_hash": canonical_content_hash("photo"),
                    "joint_policy": JOINT_POLICY,
                }
            }
        )
        return project_runtime_relevance_grid(
            run_id="query-run-1",
            cell_id="cell",
            binding=frozen.binding,
            requirement=frozen.requirement,
            result=execution,
            items=(text, native) if missing_raw else (text, raw, native),
            thresholds=[0.1, 0.5],
            joint_enabled=True,
        )

    assert not any(ref in p.result.evidence for p in project(0.29))
    assert all(ref in p.result.evidence for p in project(0.30))
    assert all(p.point["status"] == "incomplete" for p in project(0.30, missing_raw=True))


def test_inventory_does_not_reinterpret_the_joint_score_as_a_text_score():
    from pmmc.experiments.step8_inventory import _ThresholdProducer, _ThresholdUnavailable

    item = SimpleNamespace(metadata={"embedding_channel": "joint"}, threshold=0.3)
    fact = SimpleNamespace(
        relevance=SimpleNamespace(relevance_threshold=0.5), relevance_items=[item]
    )
    producer = SimpleNamespace(
        manifest=SimpleNamespace(
            protocol=SimpleNamespace(
                text_relevance_threshold=0.5,
                multimodal_relevance_threshold=0.4,
            )
        ),
        source=lambda *_: (fact, None, None),
        matching=AsyncMock(side_effect=LookupError("validated frozen policy")),
    )
    args = (producer, None, "query", "fallback_on_off", (1,), (1,), None)
    with pytest.raises(LookupError, match="validated frozen policy"):
        asyncio.run(_ThresholdProducer.point(*args))
    item.threshold = 0.5
    with pytest.raises(_ThresholdUnavailable, match="frozen_relevance_policy_mismatch"):
        asyncio.run(_ThresholdProducer.point(*args))
