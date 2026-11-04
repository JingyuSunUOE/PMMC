"""Read-only, exact Questioner adoption cost lineage.

An adoption is a namespace projection, not a provider event. Reconstruct the
importer's graph before aliasing calls; cache_hit or request_id alone is not
proof. The inventory owns the transaction and all object materialization.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_cost_dependencies import DependencyUnproven
from pmmc.experiments.step8_upstream_import import IMPORT_VERSION, remap_questioner_rows

QUESTIONER_COST_DEPENDENCIES_VERSION = "step8-questioner-cost-dependencies-v1"
QUESTIONER_COST_DEPENDENCIES_ARTIFACT = "cost-dependencies:questioner-imports"


class _NoSourceIO:
    def __call__(self, *args, **kwargs):
        raise DependencyUnproven("questioner_cost_collector_external_io_forbidden")

    def __getattr__(self, name):
        raise DependencyUnproven("questioner_cost_collector_external_io_forbidden")


def _sealed_json(value):
    # Import seals use canonical_content_hash(default=str), not Pydantic's
    # datetime JSON encoding. Preserve that representation through the CAS.
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def _call_hash(row):
    if isinstance(row, dict) and isinstance(row.get("created_at"), str):
        row = {**row, "created_at": datetime.fromisoformat(row["created_at"])}
    return canonical_content_hash(row)


def validate_questioner_cost_dependencies(facts: dict[str, Any]) -> dict[str, str]:
    """Return adopted-call -> source-call aliases only for an exact sealed graph."""
    from pmmc.infrastructure.database.step8_upstream_import import (
        QUESTIONER_TABLES,
        _rehash_questions,
        _sort_rows,
    )

    source, target = facts["source"], facts["target"]
    sh, th = source["headers"], target["headers"]
    sr, tr = sh["compiler"]["run_id"], th["compiler"]["run_id"]
    lineage = th["compiler"]["metadata"]["questioner_upstream_import"]
    if (
        sr == tr
        or lineage.get("version") != IMPORT_VERSION
        or lineage.get("source_run_id") != sr
        or lineage.get("imported_stages") != ["questioner"]
        or type(lineage.get("provider_calls")) is not int
        or lineage["provider_calls"] != 0
        or th["manifest"]["metadata"].get("questioner_upstream_import") != lineage
        or sh["manifest"]["run_id"] != sr
        or th["manifest"]["run_id"] != tr
        or sh["compiler"]["snapshot_id"] != th["compiler"]["snapshot_id"]
        or sh["compiler"].get("compiler_schema_version") != "0.9.0"
        or th["compiler"].get("compiler_schema_version") != "0.9.0"
        or canonical_content_hash(source) != lineage.get("source_facts_hash")
        or canonical_content_hash(target["rows"]) != lineage.get("target_rows_hash")
        or set(source["rows"]) != {t.name for t in QUESTIONER_TABLES}
        or set(target["rows"]) != set(source["rows"])
    ):
        raise DependencyUnproven("questioner_cost_graph_identity_mismatch")
    units = {u["compiler_unit_id"] for u in source["rows"]["compiler_units"]}
    if (
        not units
        or len(units) != sh["compiler"].get("unit_count")
        or len(units) != len(source["rows"]["compiler_units"])
        or units
        != {
            s["compiler_unit_id"]
            for s in source["rows"]["compiler_unit_stage_results"]
            if s["stage"] == "questioner" and s["status"] == "success" and s["run_id"] == sr
        }
    ):
        raise DependencyUnproven("questioner_cost_unit_denominator_incomplete")
    rebound, identities = remap_questioner_rows(
        source["rows"],
        primary_keys={t.name: [k.name for k in t.primary_key] for t in QUESTIONER_TABLES},
        source_run_id=sr,
        target_run_id=tr,
        proof_hash=lineage["proof_hash"],
    )
    _rehash_questions(rebound)
    rebound = {t.name: _sort_rows(t, rebound[t.name]) for t in QUESTIONER_TABLES}
    if canonical_content_hash(rebound) != lineage["target_rows_hash"] or canonical_content_hash(
        identities
    ) != lineage.get("identity_map_hash"):
        raise DependencyUnproven("questioner_cost_import_projection_mismatch")
    calls = source["rows"]["model_calls"]
    if not calls or any(
        c["run_id"] != sr
        or c["stage"] != "questioner"
        or c.get("snapshot_id") != sh["compiler"]["snapshot_id"]
        or c.get("query_id") is not None
        for c in calls
    ):
        raise DependencyUnproven("questioner_cost_producer_scope_mismatch")
    return {identities[c["model_call_id"]]: c["model_call_id"] for c in calls}


async def collect_questioner_cost_dependencies(
    session, *, compiler_run_ids, campaign_id: str, selection_hash: str, database_snapshot: dict
) -> dict[str, Any]:
    """Collect complete Q source graphs in the caller's evaluator snapshot.

    No sessions, object stores, or external services are opened here. Export all
    returned dependency calls and their payloads through the normal inventory
    path. Bind this artifact's SHA256 in provenance.dependency_artifact_hashes.
    Unreadable graphs remain incomplete; no identity is inferred from a cache bit.
    """
    from pmmc.infrastructure.database import compiler_models as cm
    from pmmc.infrastructure.database.step8_upstream_import import (
        PostgresQuestionerImportRepository,
    )

    # _read only uses its explicit session. Fail closed if it ever starts using
    # the repository's service handles; never accidentally open a new snapshot.
    forbidden: Any = _NoSourceIO()
    reader = PostgresQuestionerImportRepository(forbidden, forbidden)
    pending, seen = set(compiler_run_ids), set()
    entries: list[dict[str, Any]] = []
    runs: set[str] = set()
    calls: set[str] = set()
    adopted_calls: set[str] = set()
    while pending:
        target_id = min(pending)
        pending.remove(target_id)
        if target_id in seen:
            continue
        seen.add(target_id)
        facts: dict[str, Any] = {}
        reason = None
        source_id = None
        try:
            async with session.begin_nested():
                compiler = await session.get(cm.CompilerRunModel, target_id)
                if compiler is None:
                    raise DependencyUnproven("questioner_cost_target_missing")
                lineage = compiler.metadata_json.get("questioner_upstream_import")
                if lineage is None:
                    continue
                source_id = lineage["source_run_id"]
                if not isinstance(source_id, str) or not source_id or source_id == target_id:
                    raise DependencyUnproven("questioner_cost_source_identity_invalid")
                sh, sr = await reader._read(session, source_id)
                th, tr = await reader._read(session, target_id)
                facts = _sealed_json(
                    {
                        "source": {"headers": sh, "rows": sr},
                        "target": {"headers": th, "rows": tr},
                    }
                )
                adopted_calls.update(validate_questioner_cost_dependencies(facts))
                runs.add(source_id)
                calls.update(c["model_call_id"] for c in sr["model_calls"])
                pending.add(source_id)
        except DependencyUnproven as exc:
            reason = str(exc)
        except (SQLAlchemyError, ValueError, KeyError, TypeError, AttributeError):
            reason = "questioner_cost_source_unavailable_or_invalid"
        entries.append(
            {
                "target_run_id": target_id,
                "source_run_id": source_id,
                "status": "incomplete" if reason else "complete",
                "reasons": [reason] if reason else [],
                "facts": facts,
                "facts_hash": canonical_content_hash(facts),
            }
        )
    return {
        "schema_version": QUESTIONER_COST_DEPENDENCIES_VERSION,
        "campaign_id": campaign_id,
        "selection_hash": selection_hash,
        "source_role": "evaluator",
        "database_snapshot": database_snapshot,
        "imports": entries,
        "dependency_run_ids": sorted(runs),
        "dependency_model_call_ids": sorted(calls),
        "adopted_model_call_ids": sorted(adopted_calls),
    }


def questioner_cost_aliases(snapshot, calls: dict[str, dict]) -> dict[str, str]:
    """Validate Q-only alias edges; resolve together with deeper imports below."""
    imported = {
        key for key, row in calls.items() if "upstream_import" in (row.get("metadata") or {})
    }
    if not imported and QUESTIONER_COST_DEPENDENCIES_ARTIFACT not in snapshot.artifacts:
        if any(
            c.get("stage") == "questioner"
            and c.get("cache_hit") is True
            and "partial_prefix_import" not in (c.get("metadata") or {})
            for c in calls.values()
        ):
            raise DependencyUnproven("questioner_cost_cache_lineage_unproven")
        return {}
    manifest = snapshot.dependency_artifact(
        QUESTIONER_COST_DEPENDENCIES_ARTIFACT, QUESTIONER_COST_DEPENDENCIES_VERSION
    )
    aliases: dict[str, str] = {}
    for entry in manifest["imports"]:
        facts = entry["facts"]
        if (
            entry["status"] != "complete"
            or entry["reasons"]
            or canonical_content_hash(facts) != entry["facts_hash"]
            or facts["source"]["headers"]["compiler"]["run_id"] != entry["source_run_id"]
            or facts["target"]["headers"]["compiler"]["run_id"] != entry["target_run_id"]
        ):
            raise DependencyUnproven("questioner_cost_graph_unproven")
        mapped = validate_questioner_cost_dependencies(facts)
        if aliases.keys() & mapped.keys():
            raise DependencyUnproven("questioner_cost_graph_duplicated")
        for graph in (facts["source"], facts["target"]):
            for row in graph["rows"]["model_calls"]:
                if _call_hash(calls.get(row["model_call_id"])) != _call_hash(row):
                    raise DependencyUnproven("questioner_cost_exported_call_mismatch")
        aliases.update(mapped)
    if aliases.keys() != imported:
        raise DependencyUnproven("questioner_cost_alias_denominator_mismatch")
    return aliases


def imported_cost_aliases(snapshot, calls: dict[str, dict]) -> dict[str, str]:
    """Resolve disjoint Q, R0 and combined partial projections to physical calls."""
    from pmmc.experiments.step8_partial_cost_dependencies import partial_cost_alias_edges
    from pmmc.experiments.step8_r0_cost_dependencies import r0_cost_alias_edges

    aliases = questioner_cost_aliases(snapshot, calls)
    for deeper in (r0_cost_alias_edges(snapshot, calls), partial_cost_alias_edges(snapshot, calls)):
        if aliases.keys() & deeper.keys():
            raise DependencyUnproven("imported_cost_graph_conflicting_domains")
        aliases.update(deeper)
    resolved = {}
    for key in aliases:
        current, seen = key, set()
        while current in aliases:
            if current in seen:
                raise DependencyUnproven("questioner_cost_import_cycle")
            seen.add(current)
            current = aliases[current]
        if current not in calls or calls[current].get("cache_hit") is not False:
            raise DependencyUnproven("questioner_cost_original_dispatch_unproven")
        resolved[key] = current
    return resolved
