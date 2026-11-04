"""Restore fallback registration files only from verified, indexed Postgres facts."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from pmmc.baselines.suite import _write_json, baseline_suite_child_run_id
from pmmc.baselines.suite_contract import (
    BaselineSuiteContractError,
    validate_suite_run_answer_contract,
)
from pmmc.benchmarks.selection import selection_hash
from pmmc.experiments.manifest import run_config_hash
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.evaluation_repositories import PostgresBaselineSuiteRepository
from pmmc.infrastructure.database.experiment_models import (
    EmbeddingCollectionModel,
    RetrievalBatchModel,
    RunQueryModel,
)
from pmmc.infrastructure.database.experiment_repositories import (
    PostgresRetrievalRepository,
    PostgresRunRepository,
)
from pmmc.infrastructure.database.repositories import PostgresRuntimeQueryRepository
from pmmc.infrastructure.experiment_runs import load_local_run_registration
from pmmc.infrastructure.settings import Settings
from pmmc.schemas import BaselineMethod, BenchmarkQuery, BenchmarkSelection, RunManifest, RunStatus


def _require(value: bool, code: str) -> None:
    if not value:
        raise BaselineSuiteContractError(code)


def _restore_registration(
    directory: Path,
    manifest: RunManifest,
    selection: BenchmarkSelection,
    queries: list[BenchmarkQuery],
) -> int:
    """Fill missing files; an existing conflicting file is never overwritten."""

    payloads = {
        "run_manifest.json": manifest.model_dump(mode="json"),
        "selection.json": {
            "selection": selection.model_dump(mode="json"),
            "selection_hash": manifest.selection_hash,
            "query_ids": [q.query.query_id for q in queries],
            "source_locators": [q.source.model_dump(mode="json") for q in queries],
        },
    }
    selected = directory / "selected_queries.jsonl"
    for name, payload in payloads.items():
        path = directory / name
        if path.exists():
            _require(
                json.loads(path.read_text()) == payload, "fallback_local_registration_conflict"
            )
    if selected.exists():
        _require(
            [
                BenchmarkQuery.from_json(line)
                for line in selected.read_text().splitlines()
                if line.strip()
            ]
            == queries,
            "fallback_local_queries_conflict",
        )
    restored = 0
    for name, payload in payloads.items():
        path = directory / name
        if not path.exists():
            _write_json(path, payload)
            restored += 1
    if not selected.exists():
        temporary = selected.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(q.to_json() + "\n" for q in queries), encoding="utf-8")
        temporary.replace(selected)
        restored += 1
    load_local_run_registration(directory)
    return restored


async def preflight_native_fallback_retrieval(
    *,
    settings: Settings,
    suite_id: str,
    expected_run_id: str,
) -> dict[str, object]:
    """Check both identity and real retrieval before an expensive chat startup."""

    dataset = "unknown"
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        method = BaselineMethod.NATIVE_MULTIMODAL_RAG
        _require(
            baseline_suite_child_run_id(suite_id, method) == expected_run_id,
            "fallback_child_identity_mismatch",
        )
        async with sessions() as session:
            suites = PostgresBaselineSuiteRepository(session)
            suite = await suites.get_manifest(suite_id)
            dataset = suite.dataset.value
            _require(suite.methods == [method], "fallback_suite_method_mismatch")
            manifest = await PostgresRunRepository(session).get_manifest(expected_run_id)
            state = await PostgresRunRepository(session).get_state(expected_run_id)
            validate_suite_run_answer_contract(suite, manifest)
            _require(
                manifest.metadata.get("corpus_id") == suite.metadata.get("corpus_id"),
                "fallback_child_corpus_mismatch",
            )
            _require(
                manifest.config_hash == run_config_hash(manifest), "fallback_manifest_hash_invalid"
            )
            _require(
                state.status in {RunStatus.RUNNING, RunStatus.COMPLETED},
                "fallback_run_not_resumable",
            )
            selection = BenchmarkSelection.model_validate(suite.metadata.get("selection"))
            suite_queries = await suites.get_queries(suite_id)
            assignments = list(
                (
                    await session.scalars(
                        select(RunQueryModel)
                        .where(
                            RunQueryModel.run_id == expected_run_id,
                        )
                        .order_by(RunQueryModel.ordinal)
                    )
                ).all()
            )
            query_ids = [q.query_id for q in suite_queries]
            _require(
                bool(query_ids)
                and [q.query_id for q in assignments] == query_ids
                and [q.ordinal for q in assignments] == list(range(len(query_ids)))
                and state.selected_count == len(query_ids)
                and all(
                    q.snapshot_id == suite.snapshot_id and q.selection_hash == suite.selection_hash
                    for q in assignments
                ),
                "fallback_query_denominator_mismatch",
            )
            batches = list(
                (
                    await session.scalars(
                        select(RetrievalBatchModel).where(
                            RetrievalBatchModel.run_id == expected_run_id,
                        )
                    )
                ).all()
            )
            _require(
                len(batches) == len(query_ids) and {b.query_id for b in batches} == set(query_ids),
                "fallback_retrieval_incomplete",
            )
            collection_ids: set[str] = set()
            for batch in batches:
                _require(
                    batch.method == method.value
                    and batch.snapshot_id == suite.snapshot_id
                    and batch.corpus_id == suite.metadata.get("corpus_id")
                    and batch.requested_top_k == 50
                    and batch.embedding_collection_id
                    == manifest.metadata.get("document_collection_id")
                    and bool(batch.query_embedding_collection_id)
                    and batch.embedding_collection_id != batch.query_embedding_collection_id,
                    "fallback_retrieval_lineage_mismatch",
                )
                facts = await PostgresRetrievalRepository(session).get_batch(
                    batch.retrieval_batch_id
                )
                _require(facts is not None, "fallback_retrieval_missing")
                assert facts is not None
                _, traces, _, refs = facts
                _require(
                    [t.rank for t in traces] == list(range(1, batch.returned_count + 1))
                    and sum(ref.used_for_search for ref in refs) == 1,
                    "fallback_retrieval_facts_incomplete",
                )
                for identity, query_collection in (
                    (batch.embedding_collection_id, False),
                    (batch.query_embedding_collection_id, True),
                ):
                    _require(isinstance(identity, str), "fallback_collection_missing")
                    assert isinstance(identity, str)
                    if identity in collection_ids:
                        continue
                    collection = await session.get(EmbeddingCollectionModel, identity)
                    _require(
                        collection is not None
                        and collection.status == "ready"
                        and collection.snapshot_id == suite.snapshot_id
                        and (
                            collection.run_id == expected_run_id
                            if query_collection
                            else collection.corpus_id == batch.corpus_id
                        ),
                        "fallback_collection_not_ready_or_wrong_scope",
                    )
                    collection_ids.add(identity)
        # Release the read transaction before resolving queries and materializing local files.
        queries = await PostgresRuntimeQueryRepository(sessions).select(
            suite.snapshot_id, selection
        )
        _require(
            [q.query.query_id for q in queries] == query_ids
            and selection_hash(selection, queries) == manifest.selection_hash,
            "fallback_canonical_selection_mismatch",
        )
        restored = _restore_registration(
            settings.runs_path / expected_run_id, manifest, selection, queries
        )
        report: dict[str, object] = {
            "state": "ready",
            "dataset": dataset,
            "query_count": len(query_ids),
            "retrieval_batch_count": len(batches),
            "ready_collection_count": len(collection_ids),
            "restored_registration_files": restored,
            "model_calls": 0,
            "phase": "before_post_index_chat",
            "source_of_truth": "postgres",
        }
        print(
            "STEP7B_FALLBACK_RETRIEVAL_PREFLIGHT " + json.dumps(report, sort_keys=True), flush=True
        )
        return report
    except Exception as exc:
        print(
            "STEP7B_FALLBACK_RETRIEVAL_PREFLIGHT "
            + json.dumps(
                {
                    "state": "failed",
                    "dataset": dataset,
                    "model_calls": 0,
                    "phase": "before_post_index_chat",
                    "error_type": type(exc).__name__,
                    "error_code": str(exc)
                    if isinstance(exc, BaselineSuiteContractError)
                    else "fallback_preflight_failed",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    finally:
        await engine.dispose()
