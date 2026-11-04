"""One synthetic query, two evidence subsets and their local, content-addressed receipts."""

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments import step8_inventory as inv
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.models.execution import _generation_manifest
from pmmc.runtime.answering import prepare_typed_answer_request
from pmmc.runtime.grounding import build_grounding_request, grounding_evidence_hash
from pmmc.runtime.query_backend import PostgresOnlineQueryExecutorBackend
from pmmc.runtime.query_executor import OnlineQueryExecutor
from pmmc.schemas.answer_output import AnswerEvidenceSource, TypedAnswerContext
from pmmc.schemas.base import DatasetName
from pmmc.schemas.query_executor import QBankRuntimeChannel, runtime_fact_payload
from pmmc.schemas.step8 import (
    ANSWER_BACKBONES,
    BenchmarkCampaignManifest,
    CampaignPhase,
    SelectionExpansionManifest,
    build_experiment_matrix,
)
from test_step8_campaign import protocol
from test_step8_relevance_grid import GridBackend
from test_step8_runtime import CONFIG, hit, typed_answer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path("configs/runtime/query-executor-v3.yaml")
RUN = "query-run-1"
QUERY = "query-1"
SNAP = "snapshot-1"
MODEL = ANSWER_BACKBONES[0]


class _Answers(GridBackend):
    """Generate deterministic answers in place of a provider, not in place of the proof reader."""

    typed_request_identity = PostgresOnlineQueryExecutorBackend.typed_request_identity

    def __init__(self, *, grid):
        super().__init__()
        self.grid = grid
        self.hits = [
            hit(round_id=2, channel=c)
            for c in (QBankRuntimeChannel.TEXT, QBankRuntimeChannel.LEXICAL)
        ]
        self.config = CONFIG.model_copy(update={"treatment": "no_reranker"})
        self.manifest = SimpleNamespace(
            query_executor_run_id=RUN,
            source_compiler_run_id="compiler",
            snapshot_id=SNAP,
            config_hash="fixture-config",
        )
        self.observed = {}

    async def prepare_typed_pmmc_answer(self, *, query, result, cell):
        prepared = await super().prepare_typed_pmmc_answer(query=query, result=result, cell=cell)
        sources = [
            AnswerEvidenceSource(
                evidence_alias=f"A{i:02d}",
                evidence_id=r.memory_id,
                text=f"The {r.memory_id} card was blue.",
            )
            for i, r in enumerate(result.evidence, 1)
        ]
        request = prepare_typed_answer_request(
            prepared.request.model_copy(update={"model_alias": MODEL}),
            query=query,
            evidence_sources=sources,
        )
        return prepared.model_copy(
            update={
                "request": request,
                "metadata": request.metadata,
                "snapshot_id": SNAP,
                "image_ids_in_prompt": [],
                "evidence_ids_in_prompt": [s.evidence_id for s in sources],
            }
        )

    async def ground_prepared_typed_answer(self, prepared, reference):
        identity = self.typed_request_identity(prepared)
        answer = typed_answer(
            identity, abstain=not self.grid or len(prepared.evidence_ids_in_prompt) > 1
        )
        assert reference["model_call_id"] == answer.answer_model_call_id
        requests = [(prepared.request, answer.ungated_output)]
        if answer.grounding:
            context = TypedAnswerContext.model_validate(prepared.metadata["typed_answer_context"])
            grounding = answer.grounding.model_copy(
                update={
                    "evidence_content_hash": grounding_evidence_hash(context, ["A01"]),
                    "ungated_output_hash": canonical_content_hash(
                        answer.ungated_output.model_dump(mode="json")
                    ),
                }
            )
            answer = answer.model_copy(update={"grounding": grounding})
            requests.append(
                (
                    build_grounding_request(
                        prepared,
                        answer.ungated_output,
                        answer_model_call_id=answer.answer_model_call_id,
                    ),
                    grounding.assessment,
                )
            )
        self.observed[identity] = answer, requests
        return answer


async def _execute(backend):
    executor = OnlineQueryExecutor(backend=backend, config=backend.config)
    params = {"run_id": RUN, "query_id": QUERY}
    matched = await executor.execute(**params, phase="match")
    await executor.execute(**params, phase="answer")
    terminal = await executor.execute(**params, phase="select")
    fact = next(
        f
        for f in terminal["cell_results"]
        if f.cell.question_budget == 8
        and f.cell.revision_depth == 2
        and f.cell.treatment == "no_reranker"
    )
    batch = next(
        b
        for b in PostgresOnlineQueryExecutorBackend._match_facts(backend, matched)
        if b[0].match_batch_id == fact.route.match_batch_id
    )
    return fact, batch


def _case(tmp_path, monkeypatch, *, grid):
    store = CampaignArtifactStore(tmp_path)
    snapshot = inv._Snapshot()

    def append(model, row):
        snapshot.tables.setdefault(model.__table__.fullname, []).append(row)

    def artifact(model, artifact_id, data, **scope):
        digest = store.put_json(data)
        append(
            model,
            dict(
                artifact_id=artifact_id,
                object_key=artifact_id,
                sha256=digest,
                size_bytes=store.verify(digest).stat().st_size,
                state="ready",
                **scope,
            ),
        )
        return digest

    # Any attempt to fall back to an external object store is a fixture failure.
    monkeypatch.setattr(
        inv, "create_evaluator_object_stores", lambda *_: pytest.fail("unexpected object store")
    )
    backend = _Answers(grid=grid)
    fact, (batch, channels, candidates) = asyncio.run(_execute(backend))
    append(inv.qm.QBankMatchBatchModel, batch.model_dump(mode="json"))
    for model, rows in (
        (inv.qm.QBankMatchChannelCandidateModel, channels),
        (inv.qm.QBankMatchFusedCandidateModel, candidates),
    ):
        for row in rows:
            append(model, row.model_dump(mode="json"))
    append(
        inv.qm.QueryExperimentCellResultModel,
        dict(
            **{
                k: getattr(fact, k)
                for k in (
                    "cell_result_id",
                    "query_executor_run_id",
                    "query_id",
                    "cell_id",
                    "selected_answer_id",
                    "terminal_state",
                    "content_hash",
                )
            },
            treatment=fact.cell.treatment,
            question_budget=8,
            revision_depth=2,
            payload=runtime_fact_payload(fact),
        ),
    )
    evaluated = []
    for identity, (answer, requests) in backend.observed.items():
        raw = runtime_fact_payload(answer)
        common = dict(
            typed_answer_id=answer.typed_answer_id,
            query_executor_run_id=RUN,
            query_id=QUERY,
            answer_identity_hash=identity,
            answer_model_call_id=answer.answer_model_call_id,
            terminal_state=answer.terminal_state,
            content_hash=canonical_content_hash(raw),
        )
        append(
            inv.qm.QueryRuntimeTypedAnswerModel,
            {
                **common,
                "payload": {
                    **common,
                    "output": raw["output"],
                    "ungated_output": raw["ungated_output"],
                    "grounding": raw["grounding"],
                    "metadata": {"typed_result": raw},
                },
            },
        )
        for call, (request, output) in zip(answer.calls, requests, strict=True):
            input_id, output_id = call.model_call_id + ":input", call.model_call_id + ":output"
            response = {"text": output.model_dump_json(), "finish_reason": "stop"}
            digest = artifact(
                inv.xm.RuntimeArtifactModel, input_id, _generation_manifest(request), run_id=RUN
            )
            artifact(inv.xm.RuntimeArtifactModel, output_id, response, run_id=RUN)
            append(
                inv.xm.RuntimeModelCallModel,
                dict(
                    model_call_id=call.model_call_id,
                    run_id=RUN,
                    query_id=QUERY,
                    status="success",
                    model_alias=request.model_alias,
                    actual_model_id=answer.actual_model_id
                    if call.role == "answer"
                    else answer.grounding.actual_model_id,
                    stage=call.role,
                    input_asset_id=input_id,
                    output_asset_id=output_id,
                    parsed_output=response,
                    prompt_hash=digest,
                    request_config={"request_metadata": request.metadata},
                ),
            )
        evaluated.append(
            dict(
                typed_answer_id=answer.typed_answer_id,
                status="completed",
                terminal_state=answer.terminal_state,
                answer_content_hash=common["content_hash"],
                typed_score=float(answer.terminal_state == "answered"),
                judge_unavailable_reason="fixture",
            )
        )

    p = protocol()
    p = p.model_copy(
        update={
            "no_reranker_match_threshold": backend.config.route_match_threshold,
            "config_hashes": {
                **p.config_hashes,
                "query_executor_config": hashlib.sha256(
                    (ROOT / CONFIG_PATH).read_bytes()
                ).hexdigest(),
            },
        }
    )
    cell = next(
        c
        for c in build_experiment_matrix(p)
        if c.dataset == DatasetName.MEMLENS
        and c.answer_model == MODEL
        and c.treatment.value == "no_reranker"
        and c.method == "pmmc"
        and c.question_budget == 8
        and c.round_id == 2
        and c.source_variant == "active"
    )
    matrix = [cell.model_dump(mode="json")]
    manifest = BenchmarkCampaignManifest(
        campaign_id="fixture-campaign",
        phase=CampaignPhase.DEVELOPMENT,
        protocol=p,
        partition_id="fixture",
        snapshot_ids={DatasetName.MEMLENS: SNAP, DatasetName.MEMGALLERY: "other"},
        matrix_hash=canonical_content_hash(matrix),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    queries = {DatasetName.MEMLENS: [QUERY], DatasetName.MEMGALLERY: ["other-query"]}
    selection = SelectionExpansionManifest(
        campaign_id=manifest.campaign_id,
        phase=manifest.phase,
        selection="regression",
        partition_hash=p.partition_hash,
        query_ids=queries,
        added_query_ids=queries,
        eligible_counts={d: len(ids) for d, ids in queries.items()},
        stratum_counts={d: {"all": len(ids)} for d, ids in queries.items()},
    )
    directory = tmp_path / "campaign"
    write_json(directory / "manifests/matrix.json", matrix)
    write_json(
        directory / "manifests/config.json", {"query_executor_config": CONFIG_PATH.as_posix()}
    )
    write_json(
        directory / f"manifests/selections/{selection.content_hash}.json",
        selection.model_dump(mode="json"),
    )
    append(inv.sm.BenchmarkCellAssignmentModel, {"cell_id": cell.cell_id, "query_id": QUERY})
    source = dict(
        cell_id=cell.cell_id,
        source_id=fact.cell_result_id,
        source_hash=fact.content_hash,
        terminal_state=fact.terminal_state,
    )
    scope = dict(
        campaign_id=manifest.campaign_id,
        protocol_hash=p.content_hash,
        source_run_id=RUN,
        query_id=QUERY,
        snapshot_id=SNAP,
        sources=[source],
    )
    score = dict(
        cell_id=cell.cell_id,
        method=cell.method,
        variant=cell.cell_id,
        source_variant="active",
        source_fact_id=fact.cell_result_id,
        source_fact_hash=fact.content_hash,
        typed_answer_id=fact.selected_answer_id,
        terminal_state=fact.terminal_state,
        status="answered" if fact.terminal_state == "answered" else "abstention",
        metric="typed_exact_match",
        applicable=True,
        value=float(fact.terminal_state == "answered"),
        answer_source_method="pmmc",
        answer_source_variant="pmmc_b8_r2",
    )
    report = dict(
        schema_version="step8-query-evaluation-v1",
        campaign_id=manifest.campaign_id,
        query_id=QUERY,
        query_executor_run_id=RUN,
        evaluation_run_id="evaluation",
        scores=[score],
        identity_hash=canonical_content_hash(scope),
    )
    if grid:
        link = dict(
            protocol="step8-relevance-answer-evaluation-v1",
            query_run_id=RUN,
            query_id=QUERY,
            snapshot_id=SNAP,
            source_cell_hashes=[fact.content_hash],
            answers=evaluated,
            points=[
                {**point, "cell_id": fact.cell_id} for point in fact.metadata["relevance_grid"]
            ],
        )
        report["relevance_grid"] = {**link, "content_hash": canonical_content_hash(link)}
    append(
        inv.em.EvaluationRunModel,
        dict(evaluation_run_id="evaluation", metadata=scope, config_hash=report["identity_hash"]),
    )
    artifact(
        inv.xm.EvaluationArtifactModel,
        "step8-scores-fixture",
        report,
        run_id=RUN,
        query_id=QUERY,
        evaluation_run_id="evaluation",
    )
    return SimpleNamespace(
        directory=directory,
        store=store,
        snapshot=snapshot,
        settings=SimpleNamespace(resolve_path=ROOT.joinpath),
        manifest=manifest,
        selection=selection,
        cells=[cell],
        report=report,
    )


def threshold_fixture(tmp_path, monkeypatch):
    return _case(tmp_path, monkeypatch, grid=False)


def relevance_fixture(tmp_path, monkeypatch):
    return _case(tmp_path, monkeypatch, grid=True)


def produce_thresholds(case):
    asyncio.run(
        inv._populate(
            inv._Builder(case.snapshot, case.store), case.directory, case.manifest, case.settings
        )
    )
    return json.loads(
        (
            case.directory / f"private/threshold-facts/{case.selection.content_hash}.json"
        ).read_bytes()
    )
