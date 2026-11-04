"""Exercise the real cache workflow/leases with strict role and lifecycle doubles."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.settings import Settings
from pmmc.models.registry import ModelRegistry, ModelRegistryEntry
from pmmc.runtime import query_embeddings, query_workflow
from pmmc.schemas import (
    DataScope,
    EmbeddingCollectionStatus,
    EmbeddingResult,
    EmbeddingVector,
    JobStatus,
    QueryInput,
    RunStatus,
)
from pmmc.schemas.base import utc_now


def registry() -> ModelRegistry:
    return ModelRegistry(registry_version="cache-test-v1", models=[ModelRegistryEntry(
        alias=alias, endpoint_kind="fake", provider_model="fixture", model_revision="fixture",
        expected_actual_model_id="fixture", tokenizer_name="fixture", tokenizer_revision="fixture",
        capabilities=[capability], embedding_dimension=4096,
        normalize_embeddings=True, distance_metric="cosine",
    ) for alias, capability in (
        ("qwen3-embedding-8b", "embed_text"),
        ("qwen3-vl-embedding-8b", "embed_multimodal"),
    )])


class Facts:
    def __init__(self):
        self.manifests, self.states, self.jobs = {}, {}, {}
        self.collections, self.records, self.cache = {}, {}, {}
        self.calls, self.roles = [], []
        self.registration_lock = asyncio.Lock()
        self.fail = False
        self.queries = [QueryInput(
            query_id=name, dataset="memlens", conversation_id="conversation", raw_query=name,
        ) for name in ("q1", "q2")]


class UnitOfWork:
    def __init__(self, sessions, scope):
        assert scope is DataScope.RUNTIME
        self.facts, self.role = sessions.facts, sessions.role
        self.runs = self.jobs = self.embeddings = self.query_executor = self.session = self
        self.locked = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        if self.locked:
            self.facts.registration_lock.release()

    def writer(self, run_id=None):
        assert self.role is DatabaseRole.RUNTIME, "ONLINE must never write"
        if run_id is not None:
            assert self.facts.states[run_id].status is not RunStatus.COMPLETED

    async def execute(self, statement, params):
        self.writer()
        assert "pg_advisory_xact_lock" in str(statement)
        await self.facts.registration_lock.acquire()
        self.locked = True

    async def get_manifest(self, run_id):
        self.writer()
        return self.facts.manifests[run_id]

    async def register(self, manifest, assignments):
        self.writer()
        assert assignments == [] and manifest.method == "runtime_query_embedding_cache"
        self.facts.manifests[manifest.run_id] = manifest
        self.facts.states[manifest.run_id] = SimpleNamespace(status=RunStatus.CREATED)

    async def get_state(self, run_id):
        return self.facts.states[run_id]

    async def transition(self, run_id, status):
        self.writer(run_id)
        if status is RunStatus.COMPLETED:
            assert all(c.status == "ready" for c in self.facts.collections.values()
                       if c.run_id == run_id)
            assert all(j.status is JobStatus.SUCCEEDED for j in self.facts.jobs.values()
                       if j.run_id == run_id)
        self.facts.states[run_id] = SimpleNamespace(status=status)

    async def add(self, job):
        self.writer(job.run_id)
        self.facts.jobs[job.job_id] = job

    async def get(self, job_id):
        return self.facts.jobs.get(job_id)

    async def claim(self, worker, lease_seconds, *, run_id, stage):
        self.writer(run_id)
        for job in self.facts.jobs.values():
            if job.run_id == run_id and job.status is JobStatus.PENDING:
                job = job.model_copy(update={
                    "claimed_by": worker, "status": JobStatus.RUNNING,
                    "attempt_count": job.attempt_count + 1,
                })
                self.facts.jobs[job.job_id] = job
                return job
        return None

    async def heartbeat(self, job_id, worker, _lease):
        self.writer()
        assert self.facts.jobs[job_id].claimed_by == worker

    async def finish(self, job_id, worker, *, success, **kwargs):
        self.writer()
        job = self.facts.jobs[job_id]
        assert job.claimed_by == worker
        self.facts.jobs[job_id] = job.model_copy(update={
            "claimed_by": None, "finished_at": utc_now(),
            "status": JobStatus.SUCCEEDED if success else JobStatus.PENDING,
        })

    async def get_collection(self, key):
        return self.facts.collections.get(key)

    async def add_collection(self, collection):
        self.writer(collection.run_id)
        assert collection.collection_id not in self.facts.collections
        self.facts.collections[collection.collection_id] = collection

    async def add_records(self, records):
        self.writer()
        for record in records:
            collection = self.facts.collections[record.collection_id]
            self.writer(collection.run_id)
            assert collection.status is EmbeddingCollectionStatus.BUILDING
            self.facts.records[record.embedding_id] = record

    async def get_record(self, key):
        return self.facts.records.get(key)

    async def mark_collection_ready(self, key, **kwargs):
        self.writer(self.facts.collections[key].run_id)
        self.facts.collections[key] = self.facts.collections[key].model_copy(
            update={"status": EmbeddingCollectionStatus.READY, **kwargs}
        )

    async def get_runtime_embedding_cache(self, key):
        return self.facts.cache.get(key)

    async def add_runtime_embedding_cache(self, row):
        self.writer()
        assert row["cache_id"] not in self.facts.cache
        self.facts.cache[row["cache_id"]] = row


@pytest.fixture
def cache_harness(monkeypatch, tmp_path):
    facts = Facts()
    engines = []

    def factory(_settings, role):
        facts.roles.append(role)
        engine = SimpleNamespace(dispose=AsyncMock())
        engines.append(engine)
        return engine, SimpleNamespace(facts=facts, role=role)

    class Queries:
        def __init__(self, sessions):
            assert sessions.role is DatabaseRole.ONLINE

        async def list_ids(self, _snapshot, query_ids):
            return [query for query in facts.queries if query.query_id in query_ids]

    async def load_or_call(builder, *, manifest, request, gateway):
        assert builder.sessions.role is DatabaseRole.RUNTIME
        assert facts.states[manifest.query_executor_run_id].status is RunStatus.RUNNING
        await asyncio.sleep(0)
        if facts.fail:
            facts.fail = False
            raise RuntimeError("fixture provider failure")
        facts.calls.append((manifest.query_executor_run_id, request))
        return EmbeddingResult(
            request_id=request.request_id, model_alias=request.model_alias,
            actual_model_id="fixture", vectors=[EmbeddingVector(
                input_id=item.input_id, vector=[1.0] + [0.0] * 4095,
            ) for item in request.inputs],
        ), "call-" + request.request_id, False

    monkeypatch.setattr(query_workflow, "create_session_factory", factory)
    monkeypatch.setattr(query_workflow, "PostgresRuntimeQueryRepository", Queries)
    monkeypatch.setattr(query_workflow, "ExperimentUnitOfWork", UnitOfWork)
    monkeypatch.setattr(query_embeddings, "ExperimentUnitOfWork", UnitOfWork)
    monkeypatch.setattr(query_workflow, "create_runtime_object_store", lambda *_: object())
    monkeypatch.setattr(query_workflow, "create_image_url_resolver", lambda *_: None)
    monkeypatch.setattr(query_workflow, "create_embedding_gateway", lambda *_args, **_kw: object())
    monkeypatch.setattr(query_workflow, "create_model_execution_service",
                        lambda *_args, **_kw: SimpleNamespace(data_scope=DataScope.RUNTIME))
    monkeypatch.setattr(query_embeddings.QueryEmbeddingWorkflow, "_load_or_call", load_or_call)
    settings = Settings(_env_file=None, project_root=tmp_path)

    async def prepare(query_ids, compiler="completed-compiler", channel="text"):
        return await query_workflow.prepare_query_executor_runtime_embeddings(
            settings=settings, registry=registry(), source_compiler_run_id=compiler,
            snapshot_id="snapshot", query_ids=query_ids, channel=channel,
        )

    return SimpleNamespace(facts=facts, prepare=prepare, engines=engines)


def test_per_query_collections_survive_compiler_completion_and_reordered_reuse(cache_harness):
    async def scenario():
        h = cache_harness
        assert (await h.prepare(["q1"]))["model_call_count"] == 1
        assert (await h.prepare(["q2"]))["model_call_count"] == 1
        assert (await h.prepare(["q2", "q1"], compiler="another-compiler"))[
            "reused_query_count"
        ] == 2
        assert len(h.facts.collections) == len(h.facts.manifests) == len(h.facts.calls) == 2
        assert all(owner.startswith("runtime-query-cache-owner-") for owner, _ in h.facts.calls)
        assert all(state.status is RunStatus.COMPLETED for state in h.facts.states.values())
        assert set(h.facts.roles) == {DatabaseRole.ONLINE, DatabaseRole.RUNTIME}
        assert all(engine.dispose.await_count == 1 for engine in h.engines)
    asyncio.run(scenario())


def test_concurrent_same_cache_has_one_owner_call(cache_harness):
    async def scenario():
        h = cache_harness
        results = await asyncio.gather(h.prepare(["q1"]), h.prepare(["q1"], compiler="other"))
        assert sum(r["model_call_count"] for r in results) == 1
        assert sum(r["reused_query_count"] for r in results) == 1
        assert len(h.facts.calls) == len(h.facts.cache) == 1
    asyncio.run(scenario())


def test_failed_build_keeps_shard_building_and_retries(cache_harness):
    async def scenario():
        h = cache_harness
        h.facts.fail = True
        with pytest.raises(RuntimeError, match="fixture provider failure"):
            await h.prepare(["q1"])
        assert not h.facts.cache
        assert next(iter(h.facts.collections.values())).status == "building"
        assert (await h.prepare(["q1"]))["model_call_count"] == 1
        assert next(iter(h.facts.jobs.values())).attempt_count == 2
    asyncio.run(scenario())


def test_cached_lineage_is_validated_before_reuse(cache_harness):
    async def scenario():
        h = cache_harness
        await h.prepare(["q1"])
        next(iter(h.facts.cache.values()))["model_call_id"] = "another-call"
        with pytest.raises(ValueError, match="hash mismatch"):
            await h.prepare(["q1"])
        assert len(h.facts.calls) == 1
    asyncio.run(scenario())


def test_safe_views_project_only_explicit_runtime_columns():
    from test_step8_database_contracts import _migration, _statements
    migration = _migration(30)
    assert not migration._ONLINE_INSERT_TABLES
    for name in ("query_runtime_projections_v3", "query_runtime_program_vectors_v3"):
        statement = next(
            sql for sql in _statements(30) if f"CREATE VIEW pmmc_runtime.{name}" in sql
        )
        projected = statement.split("SELECT", 1)[1].split("FROM", 1)[0]
        assert "*" not in projected and "metadata" not in projected
        assert "expected" not in projected and "parsed_output" not in projected
        assert name in migration._ONLINE_VIEWS


def test_backend_safe_reads_and_trusted_phase_write_use_distinct_sessions(monkeypatch):
    from pmmc.runtime import query_backend

    reader, writer = object(), object()
    used = []
    projection = {
        "routing_projection_id": "projection", "snapshot_id": "snapshot",
        "active_round": 2, "max_round": 3, "content_hash": "projection-hash",
    }
    routing = SimpleNamespace(
        get_runtime_projection_v3=AsyncMock(return_value=projection),
        list_runtime_program_vector_refs_v3=AsyncMock(return_value=[]),
    )
    facts = SimpleNamespace(
        get_phase_state=AsyncMock(return_value={"phase": "match"}),
        add_phase_state=AsyncMock(),
    )

    class SafeUow:
        def __init__(self, sessions, _scope):
            used.append(sessions)
            self.qbank_routing = routing
            self.query_executor = facts
            self.embeddings = SimpleNamespace(get_record=AsyncMock())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    monkeypatch.setattr(query_backend, "ExperimentUnitOfWork", SafeUow)
    backend = query_backend.PostgresOnlineQueryExecutorBackend.__new__(
        query_backend.PostgresOnlineQueryExecutorBackend
    )
    backend.sessions, backend.reader_sessions = writer, reader
    backend.config = SimpleNamespace(is_v3=True, active_round=2, rounds=[0, 1, 2, 3])
    query = QueryInput(query_id="query", dataset="memlens", conversation_id="c", raw_query="q")
    backend.manifest = SimpleNamespace(
        query_executor_run_id="run", source_compiler_run_id="compiler", snapshot_id="snapshot",
        routing_projection_id="projection", dataset=query.dataset,
        metadata={"routing_projection_hash": "projection-hash"},
    )
    backend._fence = AsyncMock()

    async def scenario():
        assert await backend.load_verified_projection("run", query) == "projection"
        assert await backend._runtime_program_vectors() == {}
        assert await backend.load_runtime_phase("run", "query", "match") == {"phase": "match"}
        await backend.persist_runtime_phase("run", "query", "answer", {})
        assert used == [reader, reader, reader, writer]
        backend._fence.assert_awaited_once()
    asyncio.run(scenario())
