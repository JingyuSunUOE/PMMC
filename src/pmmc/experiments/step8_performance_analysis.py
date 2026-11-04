"""Offline, identifier-free performance summaries from verified audit evidence.

Provider-call seconds are additive request durations, not campaign wall time or
GPU reservation time. NVML interval estimates deduplicate physical devices across
overlapping workers. Legacy sample distributions remain explicitly unweighted.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_cost_dependencies import load_dependency_artifact
from pmmc.experiments.step8_questioner_cost_dependencies import imported_cost_aliases

if TYPE_CHECKING:
    from pmmc.experiments.step8_analysis import CampaignFacts

_SCOPES = ("runtime", "evaluation")
_RESOURCES = ("cpu", "api", "gpu_9b", "gpu_27b", "gpu_embedding")
_TIMING = "excludes_queue_and_artifact_persistence"
_SAMPLING = "cgroup_v2_interval_nvml_instantaneous"
_RESOURCE_FIELDS = {
    "cpu_cores_used": "cgroup_interval_average",
    "cpu_quota_utilization": "cgroup_interval_average_fraction",
    "memory_current_bytes": "cgroup_instantaneous",
    "gpu_count": "visible_device_count_at_sample",
    "gpu_utilization_mean": "visible_device_instantaneous_mean_percent",
    "gpu_memory_bytes": "visible_device_instantaneous_sum_not_worker_attribution",
}


class _Unavailable(ValueError):
    """Only fixed, content-free reason codes leave this module."""


def _number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _distribution(values: list[object], *, known: bool = True) -> dict[str, Any]:
    observed = sorted(v for value in values if (v := _number(value)) is not None)
    middle = len(observed) // 2
    median = (
        (observed[middle] if len(observed) % 2 else observed[middle - 1] / 2 + observed[middle] / 2)
        if observed
        else None
    )
    return {
        "observed_count": len(observed),
        "missing_count": len(values) - len(observed) if known else None,
        "min": observed[0] if observed else None,
        "median": median,
        "p95": observed[math.ceil(0.95 * len(observed)) - 1] if observed else None,
        "max": observed[-1] if observed else None,
        "mean": math.fsum(v / len(observed) for v in observed) if observed else None,
    }


def _rows(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise _Unavailable("model_call_rows_invalid")
    result = {}
    for row in value:
        if not isinstance(row, dict):
            raise _Unavailable("model_call_rows_invalid")
        identity = row.get("model_call_id")
        if (
            not isinstance(identity, str)
            or not identity
            or not isinstance(row.get("run_id"), str)
            or not row["run_id"]
        ):
            raise _Unavailable("model_call_identity_invalid")
        if identity in result:
            raise _Unavailable("model_call_identity_duplicated")
        result[identity] = row
    return result


class _Evidence:
    def __init__(self, facts: CampaignFacts) -> None:
        from pmmc.experiments.step8_analysis import read_campaign_file
        from pmmc.experiments.step8_audit import CampaignAuditInventory

        self.facts = facts
        name = "private/audit-inventory.json"
        raw = facts.files.get(name)
        if raw is None:
            raw = read_campaign_file(facts.directory, name)
        facts.files[name] = raw
        facts.input_hashes[name] = hashlib.sha256(raw).hexdigest()
        self.inventory = CampaignAuditInventory.model_validate_json(raw)
        if (self.inventory.campaign_id, self.inventory.selection_hash) != (
            facts.manifest.campaign_id,
            facts.selection.content_hash,
        ):
            raise _Unavailable("inventory_campaign_selection_mismatch")
        self.artifacts = {a.artifact_id: a for a in self.inventory.artifacts}
        ref = self.artifacts.get("inventory-provenance")
        if ref is None or ref.sha256 != self.inventory.source_snapshot_sha256:
            raise _Unavailable("inventory_provenance_unbound")
        self.source = self.read("inventory-provenance", "performance", "evaluation")
        if (
            self.source.get("schema_version") != "step8-database-audit-snapshot-v1"
            or self.source.get("campaign_id") != self.inventory.campaign_id
            or self.source.get("selection_hash") != self.inventory.selection_hash
            or self.source.get("source_role") != "evaluator"
            or self.source.get("isolation") != "repeatable_read_read_only"
            or not isinstance(self.source.get("database_snapshot"), dict)
            or self.source["database_snapshot"].get("database_role") != "pmmc_evaluator"
            or not self.source["database_snapshot"].get("snapshot")
            or self.source.get("conformance_run_ids") != list(self.inventory.conformance_run_ids)
            or not all(
                isinstance(self.source.get(key), dict)
                for key in ("missing", "table_counts", "table_hashes")
            )
        ):
            raise _Unavailable("inventory_provenance_invalid")

    def read(self, identity: str, kind: str, scope: str) -> dict[str, Any]:
        ref = self.artifacts.get(identity)
        if ref is None:
            raise _Unavailable("indexed_artifact_missing")
        if (ref.kind, ref.scope) != (kind, scope):
            raise _Unavailable("indexed_artifact_scope_mismatch")
        path = self.facts.store.verify(ref.sha256)
        raw = path.read_bytes()
        if len(raw) != ref.size_bytes or hashlib.sha256(raw).hexdigest() != ref.sha256:
            raise _Unavailable("indexed_artifact_integrity_failure")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise _Unavailable("indexed_artifact_payload_invalid")
        return value

    def calls(self, scope: str) -> dict[str, dict[str, Any]]:
        name = "pmmc_" + scope + ".model_calls"
        payload = self.read("table:" + name, "cost_fact", scope)
        rows = _rows(payload.get("rows"))
        if (
            payload.get("table") != name
            or self.source.get("table_counts", {}).get(name) != len(rows)
            or self.source.get("table_hashes", {}).get(name)
            != canonical_content_hash(payload["rows"])
        ):
            raise _Unavailable("model_call_table_provenance_mismatch")
        return rows

    def artifact(self, identity: str) -> dict[str, Any]:
        return self.read(identity, "performance", "evaluation")

    def dependency_artifact(self, identity: str, version: str) -> dict:
        return load_dependency_artifact(self, identity, version)


def _physical_calls(evidence, scope, calls, timings, seen_physical):
    """Exclude adoption aliases, partition old producers, deduplicate dispatch replays."""
    dependencies = evidence.source.get("dependency_model_call_ids", [])
    runs = evidence.source.get("dependency_run_ids", [])
    if (
        not isinstance(dependencies, list)
        or not isinstance(runs, list)
        or not all(isinstance(v, str) and v for v in [*dependencies, *runs])
    ):
        raise _Unavailable("performance_dependency_scope_invalid")
    external = set(dependencies) if scope == "runtime" else set()
    if any(key not in calls or calls[key]["run_id"] not in runs for key in external):
        raise _Unavailable("performance_dependency_scope_unbound")
    aliases = imported_cost_aliases(evidence, calls) if scope == "runtime" else {}
    owners = {key: scope + ":" + key for key in calls}
    # Complete inventories carry settlement evidence. Reuse the cost verifier,
    # so performance cannot turn an unbound request_id into a physical-call ID.
    if "table:pmmc_runtime.benchmark_spend_reservations" in evidence.artifacts:
        from pmmc.experiments.step8_cost_analysis import _CORE_TABLES, _normalize, _Snapshot

        snapshot = _Snapshot(evidence.facts)
        _, physical, _, _, _ = _normalize(
            snapshot, {name: snapshot.table(name) for name in _CORE_TABLES}
        )
        owners = {key: physical["pmmc_" + scope + ".", key] for key in calls}
        external = {key for key in calls if owners[key] in snapshot.dependency_physical_ids}
    grouped = {}
    for key in sorted(timings.keys() - aliases.keys()):
        physical = owners[key]
        row = {**timings[key], "_cohort": "reused_dependencies" if key in external else None}
        if physical in grouped:
            old = grouped[physical]
            fields = ("latency_ms", "queue_wait_ms", "artifact_persistence_ms", "end_to_end_ms")
            if any(old.get(field) != row.get(field) for field in fields):
                raise _Unavailable("physical_call_timing_conflict")
        else:
            grouped[physical] = row
    result = [row for physical, row in grouped.items() if physical not in seen_physical]
    seen_physical.update(grouped)
    return result


def _call_summary(
    timings: list[dict[str, Any]] | None,
    calls: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    known = timings is not None
    timings = timings or []
    modern = [r for r in timings if r.get("latency_ms_semantics") == _TIMING]
    latency = {
        "queue_wait_ms": _distribution([r.get("queue_wait_ms") for r in timings], known=known),
        "provider_ms": _distribution(
            [
                r.get("latency_ms") if r.get("latency_ms_semantics") == _TIMING else None
                for r in timings
            ],
            known=known,
        ),
        "artifact_persistence_ms": _distribution(
            [r.get("artifact_persistence_ms") for r in timings], known=known
        ),
        "end_to_end_ms": _distribution([r.get("end_to_end_ms") for r in timings], known=known),
    }
    tokens = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        values = [
            calls.get(r["model_call_id"], {}).get(field) if calls is not None else None
            for r in timings
        ]
        observed = [v for v in values if type(v) is int and _number(v) is not None]
        tokens[field] = {
            "observed_count": len(observed),
            "missing_count": len(values) - len(observed) if known else None,
            "observed_sum": sum(observed)
            if observed
            else (0 if known and calls is not None and not timings else None),
            "sum": sum(observed)
            if known and calls is not None and len(observed) == len(values)
            else None,
        }
    pairs: list[tuple[int, float]] = []
    for row in modern:
        call = calls.get(row["model_call_id"], {}) if calls is not None else {}
        output, elapsed = call.get("output_tokens"), _number(row.get("latency_ms"))
        config = call.get("request_config") or {}
        if (
            call.get("status") == "success"
            and call.get("cache_hit") is False
            and type(call.get("retry_count")) is int
            and call["retry_count"] == 0
            and isinstance(config, dict)
            and config.get("runtime_replica_failover_count") in (None, 0)
            and type(output) is int
            and _number(output) is not None
            and elapsed is not None
            and elapsed > 0
        ):
            pairs.append((output, elapsed / 1000))
    try:
        seconds = math.fsum(elapsed for _, elapsed in pairs) if pairs else None
    except OverflowError:
        seconds = None
    output_sum = sum(output for output, _ in pairs) if pairs else None
    rate = None
    if output_sum is not None and seconds:
        try:
            rate = _number(output_sum / seconds)
        except OverflowError:
            pass
    return {
        "call_count": len(timings) if known else None,
        "latency": latency,
        "legacy_unspecified_latency_ms": _distribution(
            [r.get("latency_ms") for r in timings if r.get("latency_ms_semantics") != _TIMING],
            known=known,
        ),
        "tokens": tokens,
        "throughput": {
            "metric": "recorded_output_tokens_per_summed_provider_second",
            "value": rate,
            "paired_call_count": len(pairs),
            "excluded_call_count": len(timings) - len(pairs) if known else None,
            "observed_output_tokens": output_sum,
            "observed_provider_seconds": seconds,
            "denominator": "sum_of_positive_provider_call_durations_for_paired_successes",
            "not_campaign_wall_throughput_or_decode_speed": True,
        },
        "batch_wall_throughput": {
            "metric": "recorded_output_tokens_per_batch_wall_second",
            "value": None,
            "wall_seconds": None,
            "reason": "authoritative_dispatch_and_completion_wall_window_unavailable",
            "provider_duration_sum_is_not_wall_time": True,
        },
    }


def _weighted_gpu_samples(samples: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Right-endpoint interval estimate, not a continuously observed NVML integral.

    For overlapping intervals on the same hashed physical device, take the
    nearest following sample, with worker identity as a stable tie-break. A
    sweep over endpoints counts each physical GPU-second at most once.
    """
    devices: dict[str, list[tuple[float, float, str, float, float]]] = {}
    invalid = first = duplicates = 0
    seen: dict[tuple[str, str, float], tuple[float, float, str, float, float]] = {}
    conflicts: set[tuple[str, str, float]] = set()
    for worker, sample in samples:
        duration = _number(sample.get("interval_seconds"))
        end = _number(sample.get("sample_unix_seconds"))
        entries = sample.get("gpu_devices")
        if end is not None and sample.get("interval_seconds") is None and isinstance(entries, list):
            first += 1
            continue
        if (
            end is None
            or duration is None
            or duration <= 0
            or end < duration
            or not isinstance(entries, list)
            or not entries
            or type(sample.get("gpu_count")) is not int
            or sample["gpu_count"] != len(entries)
        ):
            invalid += 1
            continue
        pending = {}
        start = end - duration
        try:
            datetime.fromtimestamp(end, UTC)
        except (OSError, ValueError, OverflowError):
            invalid += 1
            continue
        for item in entries:
            if not isinstance(item, dict):
                break
            identity = item.get("device_hash")
            utilization, memory = (
                _number(item.get("utilization")),
                _number(item.get("memory_bytes")),
            )
            if (
                not isinstance(identity, str)
                or re.fullmatch(r"[a-f0-9]{16}", identity) is None
                or identity in pending
                or utilization is None
                or utilization > 100
                or memory is None
            ):
                break
            pending[identity] = (start, end, worker, utilization, memory)
        if len(pending) != len(entries) or not start < end:
            invalid += 1
            continue
        for identity, interval in pending.items():
            key = worker, identity, end
            if key in seen:
                if seen[key] != interval:
                    conflicts.add(key)
                else:
                    duplicates += 1
            else:
                seen[key] = interval
    for key, interval in seen.items():
        if key not in conflicts:
            devices.setdefault(key[1], []).append(interval)
    weighted_util, weighted_memory, seconds, active_seconds = [], [], [], []
    raw_seconds = 0.0
    for intervals in devices.values():
        intervals.sort()
        points = sorted({point for start, end, *_ in intervals for point in (start, end)})
        active: list[tuple[float, str, float, float, float]] = []
        index = 0
        raw_seconds += math.fsum(end - start for start, end, *_ in intervals)
        for left, right in zip(points, points[1:], strict=False):
            while index < len(intervals) and intervals[index][0] <= left:
                start, end, worker, util, memory = intervals[index]
                heapq.heappush(active, (end, worker, start, util, memory))
                index += 1
            while active and active[0][0] <= left:
                heapq.heappop(active)
            if not active:
                continue
            _, _, _, util, memory = active[0]
            elapsed = right - left
            seconds.append(elapsed)
            active_seconds.append(elapsed if util > 0 else 0)
            weighted_util.append((util, elapsed))
            weighted_memory.append((memory, elapsed))
    elapsed = math.fsum(seconds)
    observed = bool(seconds)
    return {
        "status": "incomplete"
        if invalid or conflicts
        else "observed_intervals_only"
        if observed
        else "not_observed",
        "physical_device_count": len(devices) if observed else None,
        "observed_device_seconds": elapsed if observed else None,
        "overlapping_device_seconds_removed": max(0.0, raw_seconds - elapsed) if observed else None,
        "active_device_seconds_estimate": math.fsum(active_seconds) if observed else None,
        "utilization_percent_time_weighted": math.fsum(
            value * (dt / elapsed) for value, dt in weighted_util
        )
        if observed
        else None,
        "memory_bytes_per_device_time_weighted": math.fsum(
            value * (dt / elapsed) for value, dt in weighted_memory
        )
        if observed
        else None,
        "invalid_sample_count": invalid,
        "conflicting_device_sample_count": len(conflicts),
        "duplicate_device_sample_count": duplicates,
        "first_sample_without_interval_count": first,
        "estimator": "right_endpoint_hold_on_recorded_worker_interval",
        "overlap_policy": "nearest_following_sample_then_stable_worker_order",
        "active_definition": "recorded_utilization_greater_than_zero",
        "reservation_seconds": None,
        "coverage": "observed_intervals_only_not_entire_attempt_or_campaign",
        "time_alignment": "worker_unix_clocks_assumed_synchronized",
        "reason": None if observed else "device_identity_and_interval_evidence_unavailable",
    }


def _resources(evidence: _Evidence | None, issues: set[str]) -> dict[str, Any]:
    from pmmc.experiments.step8_analysis import read_campaign_file
    from pmmc.experiments.step8_audit import _safe_log

    samples: dict[str, list[dict[str, Any]]] = {key: [] for key in _RESOURCES}
    gpu_samples: list[tuple[str, dict[str, Any]]] = []
    if evidence is not None and (
        "attempts" not in evidence.inventory.coverage
        or evidence.source.get("missing", {}).get("attempts")
    ):
        issues.add("inventory_attempt_coverage_incomplete")
    expected = (
        {key for kind, key in evidence.inventory.expected_records if kind == "attempt"}
        if evidence is not None
        else set()
    )
    verified = 0
    for identity in sorted(expected):
        try:
            assert evidence is not None
            if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", identity) is None:
                raise _Unavailable("attempt_identity_invalid")
            item = evidence.read("local-attempt:" + identity, "performance", "runtime")
            if (
                item.get("attempt", {}).get("campaign_id") != evidence.inventory.campaign_id
                or item.get("attempt", {}).get("attempt_id") != identity
                or item.get("terminal", {}).get("attempt_id") != identity
                or item.get("telemetry_scope") != "worker_process"
            ):
                raise _Unavailable("attempt_binding_invalid")
            if item["terminal"].get("state") not in {
                "completed",
                "failed",
                "interrupted",
                "budget_exhausted",
            }:
                issues.add("attempt_not_terminal")
            raw = read_campaign_file(
                evidence.facts.directory, f"attempts/{identity}/safe-execution.log"
            )
            if len(raw) != item.get("log_byte_count") or hashlib.sha256(
                raw
            ).hexdigest() != item.get("log_sha256"):
                raise _Unavailable("attempt_log_integrity_failure")
            safe, stats = _safe_log(raw)
            if stats["rejected_line_count"] or (raw and not raw.endswith(b"\n")):
                issues.add("attempt_log_partial_or_invalid")
            verified += 1
            for line in safe.splitlines():
                event = json.loads(line)
                if event.get("event") != "STEP8_PROGRESS":
                    continue
                resource, values = event.get("resource"), event.get("resources")
                if (
                    resource not in samples
                    or not isinstance(values, dict)
                    or values.get("sampling") != _SAMPLING
                ):
                    issues.add("resource_sample_semantics_missing")
                    continue
                samples[resource].append(values)
                if resource.startswith("gpu_"):
                    gpu_samples.append((identity, values))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
            issues.add("attempt_resource_evidence_unavailable")
    if not any(samples.values()):
        issues.add("resource_samples_missing")
    return {
        "expected_attempt_count": len(expected) if evidence is not None else None,
        "verified_log_count": verified,
        "aggregation": "unweighted_distributions_with_separate_deduplicated_gpu_interval_estimate",
        "by_resource_class": {
            resource: {
                field: {"sampling": mode, **_distribution([s.get(field) for s in rows])}
                for field, mode in _RESOURCE_FIELDS.items()
            }
            for resource, rows in samples.items()
        },
        "gpu_observed_intervals": _weighted_gpu_samples(gpu_samples),
        "gpu_reservation_seconds": None,
        "reservation_time_status": "not_observed_in_worker_resource_samples",
    }


def summarize_campaign_performance(facts: CampaignFacts) -> dict[str, Any]:
    """Summarize all inventoried attempts, not selected-query or wall-time throughput."""
    issues: set[str] = set()
    evidence = None
    try:
        evidence = _Evidence(facts)
    except FileNotFoundError:
        issues.add("audit_inventory_missing")
    except _Unavailable as error:
        issues.add(str(error))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        issues.add("audit_inventory_invalid_or_unavailable")
    groups: dict[str, dict[str, Any]] = {}
    seen_physical: set[str] = set()
    for scope in _SCOPES:
        timing_rows = calls = None
        if evidence is not None:
            try:
                performance = evidence.read("model-call-performance:" + scope, "performance", scope)
                timing_rows = _rows(performance.get("calls"))
            except (OSError, ValueError, TypeError, KeyError):
                issues.add(scope + "_timings_unavailable")
            try:
                calls = evidence.calls(scope)
                if timing_rows is not None and (
                    calls.keys() != timing_rows.keys()
                    or any(
                        calls[key]["run_id"] != row["run_id"] for key, row in timing_rows.items()
                    )
                ):
                    raise _Unavailable("call_binding_mismatch")
                if timing_rows is not None:
                    for key, row in timing_rows.items():
                        config = calls[key].get("request_config") or {}
                        fields = ("queue_wait_ms", "artifact_persistence_ms", "end_to_end_ms")
                        if (
                            not isinstance(config, dict)
                            or row.get("latency_ms") != calls[key].get("latency_ms")
                            or any(row.get(field) != config.get(field) for field in fields)
                            or row.get("latency_ms_semantics")
                            != (_TIMING if set(fields) <= config.keys() else "legacy_unspecified")
                        ):
                            calls = None
                            timing_rows = None
                            issues.add(scope + "_performance_projection_mismatch")
                            break
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                calls = None
                timing_rows = None
                issues.add(scope + "_token_evidence_unavailable_or_mismatched")
        physical_rows = None
        if evidence is not None and timing_rows is not None and calls is not None:
            try:
                physical_rows = _physical_calls(evidence, scope, calls, timing_rows, seen_physical)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
                issues.add(scope + "_physical_call_identity_unproven")
        conformances = set(evidence.inventory.conformance_run_ids) if evidence else set()
        groups[scope] = {}
        for cohort in ("campaign", "conformance", "reused_dependencies"):
            rows = (
                [
                    r
                    for r in physical_rows
                    if (
                        r.get("_cohort")
                        or ("conformance" if r["run_id"] in conformances else "campaign")
                    )
                    == cohort
                ]
                if physical_rows is not None
                else None
            )
            summary = _call_summary(rows, calls)
            if rows and (
                any(metric["missing_count"] for metric in summary["latency"].values())
                or any(metric["missing_count"] for metric in summary["tokens"].values())
            ):
                issues.add(scope + "_measurements_incomplete")
            groups[scope][cohort] = summary
    if evidence is not None and (
        not evidence.inventory.complete
        or "performance" not in evidence.inventory.coverage
        or evidence.source.get("missing", {}).get("performance")
    ):
        issues.add("inventory_performance_coverage_incomplete")
    resources = _resources(evidence, issues)
    return {
        "schema_version": "step8-performance-analysis-v1",
        "status": "incomplete" if issues else "complete",
        "completion_scope": "verified_recorded_fields_not_unobserved_wall_or_reservation_metrics",
        "scope": "all_campaign_attempts_not_selected_query_denominator",
        "call_identity_basis": "verified_dispatch_when_available_else_trace_id",
        "cross_scope_replay_attribution": "first_runtime_then_evaluation",
        "model_calls": groups,
        "resources": resources,
        "missing": sorted(issues),
        "quantile_method": "nearest_rank_p95",
        "token_semantics": "recorded_provider_counts_may_include_reasoning_not_public_text_counts",
        "scientific_approval": False,
    }
