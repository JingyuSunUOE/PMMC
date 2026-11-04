"""Independent real-memory diagnostic campaigns use the same production DAG."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.experiments.step8_backend import compiler_instance_id
from pmmc.experiments.step8_development_admission import require_engineering_admission
from pmmc.experiments.step8_jobs import BenchmarkStageJob
from pmmc.experiments.step8_verification import verify_components
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.step8_repositories import PostgresBenchmarkRepository
from pmmc.infrastructure.settings import Settings
from pmmc.schemas.base import DatasetName
from pmmc.schemas.step8 import BenchmarkCampaignManifest, CampaignPhase, SelectionExpansionManifest

_VIABILITY = text("""
SELECT
 (SELECT count(*) FROM pmmc_runtime.doubter_assessments
  WHERE run_id = ANY(CAST(:runs AS text[])) AND provider_schema_conformant) AS feedback,
 (SELECT count(*) FROM pmmc_runtime.routing_projection_members
  WHERE run_id = ANY(CAST(:runs AS text[])) AND route_class='answerable') AS members,
 (SELECT count(*) FROM pmmc_runtime.query_route_decisions d
  JOIN pmmc_runtime.query_executor_runs r USING (query_executor_run_id)
  WHERE r.source_compiler_run_id = ANY(CAST(:runs AS text[]))
    AND d.outcome='program' AND d.query_id=ANY(CAST(:queries AS text[]))) AS routes
""")


async def require_real_prefix(
    settings: Settings,
    *,
    manifest: BenchmarkCampaignManifest,
    output_root: Path,
    target_directory: Path,
    source_bound_job_ids: frozenset[str] | None = None,
) -> dict[str, object]:
    if manifest.phase is CampaignPhase.EXPLORATION:
        return {"state": "exploratory_not_paper_approved"}
    admission = require_engineering_admission(
        manifest=manifest, directory=target_directory, source_bound_job_ids=source_bound_job_ids
    )
    if admission is not None:
        return admission
    if manifest.diagnostic_only or manifest.phase is CampaignPhase.PAPER:
        return {"state": "not_required_for_this_campaign"}
    store = CampaignArtifactStore(output_root)
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        for path in sorted(output_root.glob("*/manifests/campaign.json")):
            source = BenchmarkCampaignManifest.model_validate_json(path.read_bytes())
            if not source.diagnostic_only or source.protocol != manifest.protocol:
                continue
            directory = path.parent.parent
            selection_paths = list((directory / "manifests/selections").glob("*.json"))
            if len(selection_paths) != 1:
                raise ValueError("diagnostic_prefix_selection_ambiguous")
            selection = SelectionExpansionManifest.model_validate_json(
                selection_paths[0].read_bytes()
            )
            if selection.selection != "diagnostic" or any(
                len(ids) != 2 for ids in selection.query_ids.values()
            ):
                raise ValueError("diagnostic_prefix_denominator_invalid")
            rows = await PostgresBenchmarkRepository(sessions).list_jobs(source.campaign_id)
            verified = verify_components(
                directory=directory, campaign=source, rows=rows, store=store
            )
            if not verified["pipeline_complete"]:
                continue
            counts = {}
            for dataset in (DatasetName.MEMLENS, DatasetName.MEMGALLERY):
                jobs = [BenchmarkStageJob.model_validate(row["job"]) for row in rows]
                run_ids = sorted(
                    {
                        compiler_instance_id(source, j.parameters)
                        for j in jobs
                        if j.operation == "compiler"
                        and j.parameters.get("dataset") == dataset.value
                        and j.parameters.get("treatment") == "main"
                    }
                )
                async with sessions() as session:
                    values = (
                        (
                            await session.execute(
                                _VIABILITY,
                                {
                                    "runs": run_ids,
                                    "queries": selection.query_ids[dataset],
                                },
                            )
                        )
                        .mappings()
                        .one()
                    )
                counts[dataset.value] = {key: int(value) for key, value in values.items()}
            if any(value <= 0 for values in counts.values() for value in values.values()):
                raise ValueError("real_diagnostic_prefix_scientific_viability_failed")
            receipt = {
                "protocol": "step8-real-prefix-gate-v1",
                "state": "passed",
                "source_campaign_id": source.campaign_id,
                "method_protocol_hash": manifest.protocol.content_hash,
                "selection_hash": selection.content_hash,
                "counts": counts,
                "main_scientific_assignments": 0,
            }
            write_json(target_directory / "private/real-prefix-gate.json", receipt, immutable=True)
            return receipt
    finally:
        await engine.dispose()
    raise ValueError("real_diagnostic_prefix_required_run_diagnostic_selection_first")
