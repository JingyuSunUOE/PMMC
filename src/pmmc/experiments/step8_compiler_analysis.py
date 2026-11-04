"""Offline compiler observations from evaluator-attested, checksum-verified CAS.

No provider output or payload artifact is inspected. Counts describe persisted
producer facts, not scientific acceptance, billing, or deployed-query cost.
Each observation retains its own completeness; missing facts never become zero.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from copy import copy
from decimal import Decimal, InvalidOperation
from functools import partial
from typing import TYPE_CHECKING, Any

from pmmc.schemas.doubter import (
    PlannerDirectiveEffect,
    QBankRoundProjectionManifest,
    QBankRoundProjectionMember,
    R0InputManifest,
)
from pmmc.schemas.planner import PlannerOutcomeStatus

if TYPE_CHECKING:
    from pmmc.experiments.step8_analysis import CampaignFacts

VERSION = "step8-compiler-analysis-v1"
_PREFIX = "pmmc_runtime."
_STAGES = {
    "questioner",
    "planner0",
    "text_index",
    "multimodal_index",
    "dry_run0",
    "r0_gate",
    "doubter",
    "projection",
}


class _Unavailable(ValueError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _Unavailable(reason)


def _count(value: Any) -> int:
    _require(type(value) is int and value >= 0, "invalid_count")
    return value


def _bool(value: Any) -> bool:
    _require(type(value) is bool, "invalid_boolean")
    return value


def _unique(rows: list[dict], *keys: str) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        _require(all(v is not None for v in key), "row_identity_missing")
        _require(key not in result, "duplicate_row_identity")
        result[key] = row
    return result


def _hist(rows: list[dict], key: str, allowed: set[str]) -> dict[str, int]:
    _require(all(r[key] in allowed for r in rows), "invalid_categorical_fact")
    counts = Counter(r[key] for r in rows)
    return {name: counts[name] for name in sorted(allowed)}


def _observe(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        data = fn()
        missing = data.pop("missing_facts", []) if isinstance(data, dict) else []
        return {
            "status": "incomplete" if missing else "complete",
            "data": data,
            "missing_facts": missing,
        }
    except (ValueError, KeyError, TypeError, OSError, InvalidOperation) as exc:
        reason = str(exc) if isinstance(exc, _Unavailable) else "invalid_or_unavailable_facts"
        return {"status": "incomplete", "data": None, "missing_facts": [reason]}


class _Snapshot:
    def __init__(self, facts: CampaignFacts):
        from pmmc.experiments.step8_analysis import read_campaign_file
        from pmmc.experiments.step8_audit import (
            CampaignAuditInventory,
            audit_table_kind,
            load_process_facts,
        )

        self.facts = facts
        self.table_kind = audit_table_kind
        name = "private/audit-inventory.json"
        raw = facts.files.get(name)
        if raw is None:
            raw = read_campaign_file(facts.directory, name)
        self.inventory = CampaignAuditInventory.model_validate_json(raw)
        _require(
            (self.inventory.campaign_id, self.inventory.selection_hash)
            == (facts.manifest.campaign_id, facts.selection.content_hash),
            "inventory_campaign_selection_mismatch",
        )
        self.hashes = {name: hashlib.sha256(raw).hexdigest()}
        self.artifacts = {a.artifact_id: a for a in self.inventory.artifacts}
        self.source = self.artifact("inventory-provenance")
        _require(
            self.artifacts["inventory-provenance"].sha256 == self.inventory.source_snapshot_sha256
            and self.artifacts["inventory-provenance"].scope == "evaluation"
            and self.source.get("selection_hash") == facts.selection.content_hash
            and self.source.get("isolation") == "repeatable_read_read_only",
            "inventory_provenance_mismatch",
        )
        self.index = load_process_facts(self.inventory, facts.store, require_complete=False)
        # Verify byte sizes too, including tables read by load_process_facts.
        self.artifact("process-facts:index")
        facts.files[name] = raw
        facts.input_hashes[name] = self.hashes[name]
        self._tables: dict[str, list[dict]] = {}
        self.runs: dict[str, tuple[str, str, str, str]] = {}
        expected = set()
        for job in facts.jobs.values():
            if not (
                (job.stage in _STAGES and job.operation == "compiler") or job.operation == "units"
            ):
                continue
            p = job.scientific_inputs
            scope = tuple(p[k] for k in ("dataset", "snapshot_id", "conversation_id", "treatment"))
            _require(
                scope[3] in {"main", "fixed_planner", "no_raw_image"}, "compiler_treatment_invalid"
            )
            _require(
                scope[1] == facts.manifest.snapshot_ids.get(scope[0]),
                "compiler_scope_snapshot_mismatch",
            )
            expected.add(scope)
            if job.job_id not in facts.verified_jobs:
                continue
            receipt = facts.receipts[job.job_id]
            run = receipt.output.get("compiler_run_id")
            if not run:
                continue
            _require(isinstance(run, str), "compiler_receipt_identity_invalid")
            _require(run not in self.runs or self.runs[run] == scope, "compiler_scope_conflict")
            self.runs[run] = scope
        _require(
            bool(expected) and set(self.runs.values()) == expected, "compiler_receipts_missing"
        )
        _require(len(set(self.runs.values())) == len(self.runs), "multiple_runs_for_compiler_scope")
        _require(
            set(self.runs) <= set(self.index.get("scope", {}).get("compiler_run_ids", ())),
            "compiler_runs_outside_process_scope",
        )
        runs = self.rows("compiler_runs", "run_id")
        _require({r["run_id"] for r in runs} == set(self.runs), "compiler_run_rows_missing")
        for run in runs:
            _require(
                run["snapshot_id"] == self.runs[run["run_id"]][1]
                and run["compiler_schema_version"] == "0.9.0",
                "compiler_run_identity_mismatch",
            )
        self.units = self.rows("compiler_units", "compiler_unit_id")
        for run in runs:
            units = [u for u in self.units if u["run_id"] == run["run_id"]]
            _require(len(units) == _count(run["unit_count"]), "compiler_unit_denominator_mismatch")
            _require(
                all(u["conversation_id"] == self.runs[run["run_id"]][2] for u in units),
                "compiler_unit_conversation_mismatch",
            )
        self.questions = self.rows("qbank_questions", "qbank_id")
        units_by_id = {u["compiler_unit_id"]: u for u in self.units}
        for question in self.questions:
            unit = units_by_id.get(question["compiler_unit_id"])
            _require(
                unit is not None
                and unit["run_id"] == question["run_id"]
                and unit["conversation_id"] == question["conversation_id"],
                "candidate_unit_scope_mismatch",
            )
        self.qids = {q["qbank_id"] for q in self.questions}

    def artifact(self, identity: str) -> dict:
        ref = self.artifacts.get(identity)
        if ref is None:
            raise _Unavailable("artifact_missing:" + identity)
        raw = self.facts.store.verify(ref.sha256).read_bytes()
        _require(len(raw) == ref.size_bytes, "artifact_size_mismatch")
        value = json.loads(raw)
        _require(isinstance(value, dict), "artifact_not_object")
        self.hashes[identity] = ref.sha256
        return value

    def rows(self, name: str, *keys: str) -> list[dict]:
        from pmmc.compiler.ids import canonical_content_hash

        fullname = _PREFIX + name
        entry = self.index["tables"].get(fullname)
        if entry is not None:
            _require(entry.get("status") == "available", "table_unavailable:" + name)
        if fullname not in self._tables:
            value = self.artifact("table:" + fullname)
            ref = self.artifacts["table:" + fullname]
            rows = value.get("rows")
            _require(
                ref.scope == "runtime"
                and ref.kind == self.table_kind(fullname)
                and value.get("table") == fullname
                and isinstance(rows, list)
                and all(isinstance(r, dict) for r in rows)
                and self.source.get("table_counts", {}).get(fullname) == len(rows)
                and self.source.get("table_hashes", {}).get(fullname)
                == canonical_content_hash(rows),
                "table_provenance_mismatch:" + name,
            )
            assert isinstance(rows, list)
            self._tables[fullname] = rows
        rows = self._tables[fullname]
        # Query-time executions and other campaign model calls are intentionally excluded.
        rows = [r for r in rows if r.get("run_id") in self.runs]
        _unique(rows, *keys)
        for row in rows:
            _require(
                not row.get("snapshot_id") or row["snapshot_id"] == self.runs[row["run_id"]][1],
                "row_snapshot_scope_mismatch:" + name,
            )
        return rows

    def for_treatment(self, treatment: str) -> _Snapshot:
        selected = copy(self)
        selected.runs = {r: scope for r, scope in self.runs.items() if scope[3] == treatment}
        selected.units = [u for u in self.units if u["run_id"] in selected.runs]
        selected.questions = [q for q in self.questions if q["run_id"] in selected.runs]
        selected.qids = {q["qbank_id"] for q in selected.questions}
        return selected

    def programs(self) -> list[dict]:
        rows = self.rows("memory_programs", "plan_id")
        _unique(rows, "qbank_id", "round_id")
        qrun = {q["qbank_id"]: q["run_id"] for q in self.questions}
        for row in rows:
            _require(qrun.get(row["qbank_id"]) == row["run_id"], "program_candidate_scope_mismatch")
            _require(
                type(row["round_id"]) is int and 0 <= row["round_id"] <= 3, "program_round_invalid"
            )
        return rows


def _generations(s: _Snapshot) -> dict:
    rows = s.rows("questioner_generations", "generation_id")
    calls = {r["model_call_id"]: r for r in s.rows("model_calls", "model_call_id")}
    unknown = 0
    raw = valid = invalid = 0
    for row in rows:
        call = calls.get(row["model_call_id"])
        _require(call is not None and call["run_id"] == row["run_id"], "questioner_call_missing")
        total, good, bad = (
            _count(row[k])
            for k in ("raw_candidate_count", "valid_candidate_count", "invalid_candidate_count")
        )
        _require(good + bad == total, "questioner_candidate_counts_mismatch")
        # Zero in a failed-envelope terminal is an accounting sentinel, not an observation.
        known = row["parse_status"] in {"parsed", "partial", "repaired"} and not row.get(
            "metadata", {}
        ).get("provider_item_count_status")
        unknown += not known
        if known:
            raw += total
            valid += good
            invalid += bad
    _require(
        {r["compiler_unit_id"] for r in rows} == {u["compiler_unit_id"] for u in s.units},
        "questioner_generation_unit_denominator_missing",
    )
    return {
        "generation_attempts": len(rows),
        "unknown_provider_count_attempts": unknown,
        "provider_raw_items": None if unknown else raw,
        "known_provider_raw_items": raw,
        "resolved_native_items": None if unknown else valid,
        "invalid_or_unresolved_items": None if unknown else invalid,
        "known_resolved_native_items": valid,
        "known_invalid_or_unresolved_items": invalid,
        "count_basis": "persisted_generation_accounting_not_provider_body",
        "missing_facts": ["provider_item_count_unavailable"] if unknown else [],
    }


def _eligible(s: _Snapshot) -> set[str]:
    gates = s.rows("questioner_candidate_gate_results", "qbank_id")
    _require({g["qbank_id"] for g in gates} == s.qids, "questioner_gate_denominator_missing")
    verifiers = _unique(
        s.rows("questioner_candidate_verifications", "candidate_verification_id"),
        "qbank_id",
        "verification_round",
    )
    questions = {q["qbank_id"]: q for q in s.questions}
    eligible = set()
    for gate in gates:
        q = questions[gate["qbank_id"]]
        _require(gate["run_id"] == q["run_id"], "questioner_gate_scope_mismatch")
        if gate["status"] != "eligible_for_planning":
            continue
        verifier = verifiers.get((q["qbank_id"], q["candidate_version"]))
        if verifier is None:
            raise _Unavailable("eligible_verification_missing")
        _require(
            verifier["run_id"] == q["run_id"],
            "eligible_verification_missing",
        )
        _require(verifier["verdict"] == "pass", "eligible_verification_not_passed")
        eligible.add(q["qbank_id"])
    return eligible


def _families(s: _Snapshot) -> dict:
    eligible = _eligible(s)
    rows = s.rows("qbank_candidate_families", "family_id")
    members = s.rows("qbank_candidate_family_members", "family_member_id")
    _unique(members, "run_id", "qbank_id")
    _require(
        {m["family_id"] for m in members} == {r["family_id"] for r in rows},
        "family_membership_missing",
    )
    _require(eligible <= {m["qbank_id"] for m in members}, "eligible_family_missing")
    for family in rows:
        own = [m for m in members if m["family_id"] == family["family_id"]]
        _require(
            len(own) == _count(family["member_count"])
            and all(m["run_id"] == family["run_id"] and m["qbank_id"] in s.qids for m in own)
            and [m["qbank_id"] for m in own if _bool(m["is_primary"])]
            == [family["primary_qbank_id"]]
            and family["primary_qbank_id"] in eligible,
            "family_denominator_or_primary_mismatch",
        )
    return {"families": len(rows), "members": len(members), "eligible_candidates": len(eligible)}


def _stage_attempts(s: _Snapshot) -> dict:
    rows = s.rows("compiler_unit_stage_results", "stage_result_id")
    _unique(rows, "compiler_unit_id", "stage", "attempt_number")
    result = {}
    for stage in ("questioner", "planner", "dry_run", "doubter"):
        own = [r for r in rows if r["stage"] == stage]
        result[stage] = {
            "attempts": len(own),
            "success": sum(r["status"] == "success" for r in own),
            "failed": sum(r["status"] == "failed" for r in own),
            "basis": "unit_stage_status_not_candidate_rejections",
        }
    return result


def _planner(s: _Snapshot) -> dict:
    batches = s.rows("planner_generation_batches", "planner_batch_id")
    _unique(batches, "model_call_id")
    eligible = _eligible(s)
    contracts = {
        r["qbank_id"]: r for r in s.rows("planner_requirement_contracts", "requirement_contract_id")
    }
    outcomes = {
        r["qbank_id"]: r for r in s.rows("planner_candidate_outcomes", "planner_outcome_id")
    }
    expected = set()
    for q in s.questions:
        if q["qbank_id"] not in eligible:
            continue
        outcome = outcomes.get(q["qbank_id"])
        if outcome is None:
            raise _Unavailable("planner_outcome_denominator_missing")
        if outcome["status"] == "lineage_rejected" and outcome.get("planner_batch_id") is None:
            continue
        contract = contracts.get(q["qbank_id"])
        if contract is None:
            raise _Unavailable("planner_requirement_contract_missing")
        _require(
            contract["run_id"] == q["run_id"],
            "planner_requirement_contract_missing",
        )
        if not _bool(contract["refusal_only"]) and s.runs[q["run_id"]][3] != "fixed_planner":
            expected.add(q["qbank_id"])
    submitted = []
    for row in batches:
        ids = row["candidate_qbank_ids"]
        _require(
            isinstance(ids, list) and len(set(ids)) == len(ids) and set(ids) <= eligible,
            "planner_batch_candidate_scope_mismatch",
        )
        submitted.extend(ids)
        _require(
            _count(row["parsed_count"]) + _count(row["invalid_count"]) == len(ids),
            "planner_parse_denominator_mismatch",
        )
    _require(
        set(submitted) == expected and len(submitted) == len(set(submitted)),
        "planner_candidate_denominator_missing_or_duplicate",
    )
    initial = [p for p in s.programs() if p["round_id"] == 0]
    _require({p["qbank_id"] for p in initial} <= eligible, "initial_program_not_eligible")
    return {
        "batches": len(batches),
        "submitted_candidates": len(submitted),
        "deterministic_or_pre_model_rejection_candidates": len(eligible) - len(expected),
        "parsed_candidates": sum(r["parsed_count"] for r in batches),
        "invalid_candidates": sum(r["invalid_count"] for r in batches),
        "initial_programs": len(initial),
    }


def _planner_outcomes(s: _Snapshot) -> dict:
    eligible = _eligible(s)
    rows = s.rows("planner_candidate_outcomes", "planner_outcome_id")
    _unique(rows, "qbank_id")
    _require({r["qbank_id"] for r in rows} == eligible, "planner_outcome_denominator_missing")
    programs = {p["plan_id"]: p for p in s.programs() if p["round_id"] == 0}
    for row in rows:
        if row.get("plan_id") is not None:
            program = programs.get(row["plan_id"])
            _require(
                program is not None
                and program["qbank_id"] == row["qbank_id"]
                and program["run_id"] == row["run_id"],
                "planner_outcome_program_mismatch",
            )
    return {
        "candidate_denominator": len(rows),
        "statuses": _hist(rows, "status", {s.value for s in PlannerOutcomeStatus}),
        "basis": "candidate_outcomes_not_engineering_attempt_failures",
    }


def _policy(s: _Snapshot) -> dict:
    programs = {p["plan_id"] for p in s.programs()}
    rows = s.rows("program_validation_results", "validation_result_id")
    _unique(rows, "plan_id", "validation_kind")
    _require(all(r["plan_id"] in programs for r in rows), "validation_program_missing")
    result = {}
    for kind in ("static", "policy"):
        own = [r for r in rows if r["validation_kind"] == kind]
        _require({r["plan_id"] for r in own} == programs, "program_validation_denominator_missing")
        result[kind] = {
            "programs": len(own),
            "valid": sum(_bool(r["valid"]) for r in own),
            "invalid": sum(not r["valid"] for r in own),
        }
    return result


def _revisions(s: _Snapshot) -> dict:
    programs = {p["plan_id"]: p for p in s.programs()}
    rows = s.rows("planner_revision_lineage", "revision_lineage_id")
    _unique(rows, "qbank_id", "target_round_id")
    depths: Counter[int] = Counter()
    for qid in s.qids:
        own = sorted(
            (p for p in programs.values() if p["qbank_id"] == qid), key=lambda p: p["round_id"]
        )
        if not own:
            continue
        _require([p["round_id"] for p in own] == list(range(len(own))), "program_revision_gap")
        for source, revised in zip(own[:-1], own[1:], strict=True):
            links = [r for r in rows if r["revised_plan_id"] == revised["plan_id"]]
            _require(
                len(links) == 1
                and links[0]["source_plan_id"] == source["plan_id"]
                and links[0]["target_round_id"] == revised["round_id"]
                and links[0]["qbank_id"] == qid
                and links[0]["run_id"] == revised["run_id"],
                "revision_lineage_missing",
            )
        depths[own[-1]["round_id"]] += 1
    _require(
        len(rows) == sum(p["round_id"] > 0 for p in programs.values()), "orphan_revision_lineage"
    )
    return {
        "programs": len(programs),
        "revised_programs": len(rows),
        "candidate_denominator": sum(depths.values()),
        "candidates_without_program": len(s.qids) - sum(depths.values()),
        "actual_max_depth_counts": {str(i): depths[i] for i in range(4)},
    }


def _doubter(s: _Snapshot, round_id: int | None = None) -> dict:
    programs = {p["plan_id"]: p for p in s.programs()}
    feedback = {r["feedback_id"]: r for r in s.rows("doubter_feedback", "feedback_id")}
    rows = s.rows("doubter_assessments", "assessment_id")
    _unique(rows, "qbank_id", "round_id")
    dry_runs = s.rows("program_dry_run_assessments", "assessment_id")
    _require(
        {r["plan_id"] for r in rows} == {r["plan_id"] for r in dry_runs},
        "doubter_assessment_denominator_missing",
    )
    for row in rows:
        p, f = programs.get(row["plan_id"]), feedback.get(row["feedback_id"])
        _require(
            p is not None
            and f is not None
            and all(row[k] == p[k] == f[k] for k in ("run_id", "qbank_id", "round_id", "plan_id")),
            "doubter_feedback_program_scope_mismatch",
        )
    _require(len(feedback) == len(rows), "feedback_assessment_denominator_mismatch")
    if round_id is not None:
        rows = [r for r in rows if r["round_id"] == round_id]
    return {
        "feedback_count": len(rows),
        "assessment_count": len(rows),
        "actions": _hist(rows, "action", {"pass", "revise_program", "reject_candidate"}),
        "provider_schema_conformant": sum(_bool(r["provider_schema_conformant"]) for r in rows),
        "provider_schema_nonconformant": sum(not r["provider_schema_conformant"] for r in rows),
    }


def _directives(s: _Snapshot) -> dict:
    directives = s.rows("doubter_directives", "directive_id")
    patches = s.rows("planner_revision_patches", "patch_id")
    _unique(patches, "qbank_id", "target_round_id")
    by_id = {r["directive_id"]: r for r in directives}
    effects = []
    seen = set()
    for patch in patches:
        _require(
            _bool(patch["applied"]) == bool(patch.get("after_program_hash"))
            and patch["applied"] == (patch.get("rejection_code") is None),
            "patch_application_inconsistent",
        )
        raw = patch.get("metadata", {}).get("directive_effects")
        _require(isinstance(raw, list), "directive_effects_missing")
        for item in raw:
            effect = PlannerDirectiveEffect.model_validate(item)
            directive = by_id.get(effect.directive_id)
            _require(
                directive is not None
                and effect.directive_id not in seen
                and directive["run_id"] == patch["run_id"]
                and directive["qbank_id"] == patch["qbank_id"]
                and directive["round_id"] == patch["target_round_id"]
                and directive["directive"] == effect.directive.value,
                "directive_effect_scope_mismatch",
            )
            seen.add(effect.directive_id)
            effects.append({"status": effect.status.value})
    _require(seen == set(by_id), "directive_application_denominator_missing")
    return {
        "directives": len(directives),
        "patches": len(patches),
        "applied_patches": sum(p["applied"] for p in patches),
        "rejected_patches": sum(not p["applied"] for p in patches),
        "effects": _hist(
            effects, "status", {"applied", "already_satisfied", "inapplicable", "saturated", "noop"}
        ),
    }


def _r0(s: _Snapshot) -> dict:
    rows = [R0InputManifest.model_validate(r) for r in s.rows("r0_input_manifests", "run_id")]
    _require({r.run_id for r in rows} == set(s.runs), "r0_manifest_missing")
    _require(all(r.schema_version == "0.4.0" for r in rows), "r0_manifest_version_mismatch")
    return {
        "manifests": len(rows),
        "hard_gate_passed": sum(r.hard_gate_passed for r in rows),
        "protocol_integrity_issue_count": sum(len(r.protocol_integrity_issues) for r in rows),
        "candidate_scientific_rejections": sum(
            _count(n) for r in rows for n in r.candidate_scientific_rejections.values()
        ),
        "coverage_zero_units": sum(r.coverage_zero_units for r in rows),
        "downstream_viable_runs": sum(r.downstream_viability for r in rows),
        "candidate_denominator": sum(r.candidate_count for r in rows),
    }


def _round_members(s: _Snapshot, round_id: int) -> list[QBankRoundProjectionMember]:
    from pmmc.compiler.ids import canonical_content_hash

    manifests = [
        QBankRoundProjectionManifest.model_validate(r)
        for r in s.rows("qbank_round_projection_manifests", "projection_id")
        if r["round_id"] == round_id
    ]
    _require(
        {m.run_id for m in manifests} == set(s.runs) and len(manifests) == len(s.runs),
        "exact_round_projection_missing",
    )
    rows = s.rows("qbank_round_projection_members", "projection_member_id")
    programs = {p["plan_id"]: p for p in s.programs()}
    feedback = {f["feedback_id"]: f for f in s.rows("doubter_feedback", "feedback_id")}
    dry_runs = {d["plan_id"]: d for d in s.rows("program_dry_run_assessments", "assessment_id")}
    r0_ids = {m["run_id"]: m["r0_manifest_id"] for m in s.rows("r0_input_manifests", "run_id")}
    eligible = _eligible(s)
    all_members = []
    for manifest in manifests:
        members = [
            QBankRoundProjectionMember.model_validate(r)
            for r in rows
            if r["projection_id"] == manifest.projection_id
        ]
        _require(
            manifest.schema_version == "0.4.0"
            and manifest.source_r0_manifest_id == r0_ids.get(manifest.run_id)
            and manifest.member_count == len(members)
            and {m.qbank_id for m in members}
            == {
                q["qbank_id"]
                for q in s.questions
                if q["run_id"] == manifest.run_id and q["qbank_id"] in eligible
            }
            and len({m.qbank_id for m in members}) == len(members)
            and all(m.run_id == manifest.run_id for m in members)
            and manifest.members_hash
            == canonical_content_hash(
                [m.model_dump(mode="json") for m in sorted(members, key=lambda m: m.qbank_id)]
            )
            and manifest.content_hash
            == canonical_content_hash(
                manifest.model_dump(
                    mode="json",
                    exclude={"schema_version", "content_hash", "created_at", "metadata"},
                )
            ),
            "round_projection_hash_or_denominator_mismatch",
        )
        counts = Counter(m.outcome.value for m in members)
        _require(
            manifest.accepted_count == counts["accepted"] + counts["carried_forward"]
            and manifest.rejected_count == counts["rejected"] + counts["max_rounds_exhausted"]
            and manifest.pending_count == counts["pending"],
            "round_outcome_counts_mismatch",
        )
        for member in members:
            if member.effective_plan_id is not None:
                plan = programs.get(member.effective_plan_id)
                passed = feedback.get(member.effective_feedback_id)
                _require(
                    plan is not None
                    and plan["qbank_id"] == member.qbank_id
                    and plan["run_id"] == member.run_id
                    and plan["round_id"] <= round_id
                    and member.passed_at_round is not None
                    and member.passed_at_round == plan["round_id"] <= round_id
                    and (member.outcome == "accepted") == (member.passed_at_round == round_id)
                    and (
                        round_id == 0
                        or (
                            passed is not None
                            and passed["verifier_pass"] is True
                            and all(
                                passed[k] == plan[k]
                                for k in ("plan_id", "run_id", "qbank_id", "round_id")
                            )
                        )
                    ),
                    "round_effective_program_invalid",
                )
                if round_id == 0:
                    dry = dry_runs.get(member.effective_plan_id)
                    _require(
                        dry is not None
                        and dry["execution_success"] is True
                        and (
                            dry["abstain_triggered"] is True
                            if member.route_class == "control"
                            else dry["nonempty_evidence"] is True
                            and dry["required_modality_sufficient"] is True
                            and dry["pre_control_sufficient"] is True
                        ),
                        "r0_projection_dry_run_not_ready",
                    )
        all_members.extend(members)
    return all_members


def _round(s: _Snapshot, round_id: int) -> dict:
    members = _round_members(s, round_id)
    accepted = {"accepted", "carried_forward"}
    base = {m.qbank_id: m for m in _round_members(s, 0)}
    return {
        "candidate_denominator": len(members),
        "outcomes": dict(sorted(Counter(m.outcome.value for m in members).items())),
        "accepted_including_carry": sum(m.outcome in accepted for m in members),
        "early_carry": sum(m.outcome == "carried_forward" for m in members),
        "newly_accepted": sum(
            m.outcome in accepted and m.passed_at_round == round_id for m in members
        ),
        "rescued_from_r0": sum(
            m.outcome in accepted and base[m.qbank_id].outcome not in accepted for m in members
        ),
        "rescue_candidate_denominator": sum(m.outcome not in accepted for m in base.values()),
    }


def _dry_run(s: _Snapshot, round_id: int) -> dict:
    programs = {p["plan_id"]: p for p in s.programs() if p["round_id"] == round_id}
    executions = {r["execution_id"]: r for r in s.rows("program_executions", "execution_id")}
    rows = [
        r
        for r in s.rows("program_dry_run_assessments", "assessment_id")
        if r["plan_id"] in programs
    ]
    _unique(rows, "plan_id")
    # Policy-invalid plans legitimately have no execution; do not hide their count.
    validations = s.rows("program_validation_results", "validation_result_id")
    for kind in ("static", "policy"):
        _require(
            {v["plan_id"] for v in validations if v["validation_kind"] == kind} >= set(programs),
            "dry_run_validation_denominator_missing",
        )
    executable = {
        pid
        for pid in programs
        if any(
            v["plan_id"] == pid and v["validation_kind"] == "policy" and v["valid"]
            for v in validations
        )
    }
    _require({r["plan_id"] for r in rows} == executable, "dry_run_program_denominator_missing")
    expected = matched = 0
    for row in rows:
        execution = executions.get(row["execution_id"])
        _require(
            execution is not None
            and execution["run_id"] == row["run_id"]
            and execution["plan_id"] == row["plan_id"]
            and execution.get("query_id") is None,
            "dry_run_execution_scope_mismatch",
        )
        n, d = _count(row["matched_group_count"]), _count(row["expected_group_count"])
        value = row["expected_evidence_recall"]
        _require(
            n <= d
            and (
                (d == 0 and value is None)
                or (d > 0 and isinstance(value, (float, int)) and math.isclose(value, n / d))
            ),
            "expected_group_recall_mismatch",
        )
        matched += n
        expected += d
    return {
        "program_denominator": len(programs),
        "executed_program_denominator": len(rows),
        "programs_without_execution": len(programs) - len(rows),
        "technical_success": sum(_bool(r["execution_success"]) for r in rows),
        "nonempty_evidence": sum(_bool(r["nonempty_evidence"]) for r in rows),
        "raw_image_reached": sum(_bool(r["raw_image_reached"]) for r in rows),
        "expected_group_recall": {
            "numerator": matched,
            "denominator": expected,
            "value": matched / expected if expected else None,
            "basis": "compiler_expected_groups_not_benchmark_gold",
        },
    }


def _calls(s: _Snapshot) -> tuple[list[dict], dict[str, int]]:
    rows = s.rows("model_calls", "model_call_id")
    calls = {r["model_call_id"]: r for r in rows}
    rounds: dict[str, set[int]] = defaultdict(set)
    # Use the original producer, not revision lineage that may reuse a Doubter call.
    for table, key, round_key in (
        ("questioner_generations", "model_call_id", None),
        ("questioner_verification_batches", "model_call_id", None),
        ("planner_generation_batches", "model_call_id", None),
        ("doubter_assessments", "producer_model_call_id", "round_id"),
    ):
        for row in s.rows(table, key):
            _require(
                row[key] in calls and calls[row[key]]["run_id"] == row["run_id"],
                "producer_model_call_missing",
            )
            rounds[row[key]].add(row[round_key] if round_key else 0)
    programs = {p["plan_id"]: p for p in s.programs()}
    for call in rows:
        if call["stage"] == "questioner":
            rounds[call["model_call_id"]].add(0)
        if call.get("plan_id") is not None:
            plan = programs.get(call["plan_id"])
            if plan is None:
                raise _Unavailable("write_call_plan_scope_mismatch")
            _require(
                plan["run_id"] == call["run_id"],
                "write_call_plan_scope_mismatch",
            )
            rounds[call["model_call_id"]].add(plan["round_id"])
    return rows, {key: next(iter(value)) for key, value in rounds.items() if len(value) == 1}


def _cost(s: _Snapshot, round_id: int | None = None) -> dict:
    rows, attributed = _calls(s)
    unassigned = sum(r["model_call_id"] not in attributed for r in rows)
    if round_id is not None:
        rows = [r for r in rows if attributed.get(r["model_call_id"]) == round_id]
    values = []
    for row in rows:
        value = row.get("cost_usd")
        if value is not None:
            number = Decimal(str(value))
            _require(number.is_finite() and number >= 0, "invalid_model_call_cost")
            values.append(number)
    tokens = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        observed = [_count(r[key]) for r in rows if r.get(key) is not None]
        tokens[key] = {
            "known_sum": sum(observed) if observed else None,
            "unknown_call_count": len(rows) - len(observed),
        }
    return {
        "write_call_count": len(rows),
        "failed_call_count": sum(r["status"] == "failed" for r in rows),
        "round_unattributed_call_count": unassigned,
        "known_cost_call_count": len(values),
        "unknown_cost_call_count": len(rows) - len(values),
        "known_cost_usd": str(sum(values, Decimal(0))) if values else None,
        "tokens": tokens,
        "basis": "retained_write_call_trace_not_billing_or_deployed_cost",
        "missing_facts": sorted(
            ({"write_call_cost_unknown"} if len(values) < len(rows) else set())
            | (
                {"write_call_round_attribution_incomplete"}
                if round_id is not None and unassigned
                else set()
            )
            | (
                {"write_call_tokens_unknown"}
                if any(v["unknown_call_count"] for v in tokens.values())
                else set()
            )
        ),
    }


def summarize_campaign_compiler(facts: CampaignFacts) -> dict[str, Any]:
    """Aggregate each verified compiler run once, independent of answer backbones.

    Every leaf observation is {status, data, missing_facts}. ``data=None`` means
    unknown, never zero. ``status=complete`` attests accounting, not model quality.
    """
    try:
        s = _Snapshot(facts)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {
            "version": VERSION,
            "status": "incomplete",
            "denominators": None,
            "missing_facts": [
                str(exc) if isinstance(exc, _Unavailable) else "compiler_inventory_unavailable"
            ],
            "questioner": None,
            "planner": None,
            "doubter": None,
            "rounds": [],
            "costs": None,
            "input_hashes": {},
        }
    result = _summarize_snapshot(s)
    result["by_treatment"] = {
        treatment: _summarize_snapshot(s.for_treatment(treatment))
        for treatment in sorted({scope[3] for scope in s.runs.values()})
    }
    return result


def _summarize_snapshot(s: _Snapshot) -> dict[str, Any]:
    observations = {
        "questioner": {
            "generations": _observe(lambda: _generations(s)),
            "eligibility": _observe(
                lambda: {
                    "eligible_candidates": len(_eligible(s)),
                    "native_candidate_denominator": len(s.qids),
                }
            ),
            "families": _observe(lambda: _families(s)),
        },
        "planner": {
            "initial": _observe(lambda: _planner(s)),
            "candidate_outcomes": _observe(lambda: _planner_outcomes(s)),
            "validation": _observe(lambda: _policy(s)),
            "revisions": _observe(lambda: _revisions(s)),
        },
        "doubter": {
            "assessments": _observe(lambda: _doubter(s)),
            "directive_application": _observe(lambda: _directives(s)),
        },
        "stage_attempts": _observe(lambda: _stage_attempts(s)),
        "r0": _observe(lambda: _r0(s)),
        "rounds": [
            {
                "round_id": r,
                "projection": _observe(partial(_round, s, r)),
                "dry_run": _observe(partial(_dry_run, s, r)),
                "doubter": _observe(partial(_doubter, s, r)),
                "write_costs": _observe(partial(_cost, s, r)),
            }
            for r in range(4)
        ],
        "costs": _observe(lambda: _cost(s)),
    }
    missing = set()

    def visit(value):
        if isinstance(value, dict):
            missing.update(value.get("missing_facts", []))
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(observations)
    if (
        s.index.get("status") != "complete"
        or s.index.get("missing")
        or not s.inventory.complete
        or "process_facts" not in s.inventory.coverage
    ):
        missing.add("process_inventory_coverage_incomplete")
    return {
        "version": VERSION,
        "status": "incomplete" if missing else "complete",
        "denominators": {
            "compiler_runs": len(s.runs),
            "conversations": len({scope[:3] for scope in s.runs.values()}),
            "units": len(s.units),
            "native_candidates": len(s.qids),
        },
        **observations,
        "missing_facts": sorted(missing),
        "input_hashes": dict(sorted(s.hashes.items())),
    }
