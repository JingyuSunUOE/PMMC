"""Signed public/private pilot bundles and deterministic task generation."""

from __future__ import annotations

import hashlib
import html
import json
import shutil
import stat
from pathlib import Path
from typing import Any

from pmmc.compiler.questioner_export import verify_questioner_export
from pmmc.pilots.analysis import analyze_questioner_quality
from pmmc.pilots.config import QuestionerPilotConfig
from pmmc.pilots.io import (
    file_sha256,
    read_json,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)
from pmmc.pilots.partition import load_partition, verify_partition
from pmmc.pilots.review import build_review_tasks
from pmmc.schemas import (
    QUESTIONER_PILOT_SCHEMA_VERSION,
    DatasetName,
    PartitionRole,
    QuestionerPilotArtifactRef,
    QuestionerPilotAuditManifest,
    QuestionerPilotManifest,
    QuestionerPilotSample,
    QuestionerPilotState,
    QuestionerPilotStatus,
    ReviewAssignment,
    ReviewRole,
    ReviewTaskKind,
)

_PUBLIC_FORBIDDEN_KEYS = {
    "expected_answer",
    "gold_answer",
    "gold_evidence",
    "reviewer_id",
    "approved_by",
    "notes",
    "rationale",
    "raw_query",
}
_SECRET_MARKERS = (
    b"GOLD_ANSWER_SENTINEL",
    b"PMMC_GOLD_SENTINEL",
    b"HF_TOKEN=",
    b"OPENAI_API_KEY=",
    b"X-Amz-Signature=",
)
_SMOKE_MANIFEST_FIELDS = {
    "mode",
    "launch_id",
    "partition_id",
    "sample_ids",
    "runtime_exports",
    "coverage_exports",
    "paper_result",
}


def _walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            keys.update(str(key) for key in item)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return keys


def verify_questioner_smoke_bundle(pilot_dir: Path) -> dict[str, object]:
    """Verify the minimal, non-paper smoke bundle used by checkpoint resume."""

    manifest = read_json(pilot_dir / "manifest.json")
    if set(manifest) != _SMOKE_MANIFEST_FIELDS:
        raise ValueError("Questioner smoke manifest fields do not match its protocol")
    launch_id = manifest.get("launch_id")
    partition_id = manifest.get("partition_id")
    sample_ids = manifest.get("sample_ids")
    runtime_exports = manifest.get("runtime_exports")
    coverage_exports = manifest.get("coverage_exports")
    if manifest.get("mode") != "smoke" or manifest.get("paper_result") is not False:
        raise ValueError("Questioner smoke bundle has an invalid mode or paper-result flag")
    if not isinstance(launch_id, str) or not launch_id:
        raise ValueError("Questioner smoke bundle has no launch identity")
    if pilot_dir.name != f"smoke-{launch_id}":
        raise ValueError("Questioner smoke directory does not match its launch identity")
    if not isinstance(partition_id, str) or not partition_id:
        raise ValueError("Questioner smoke bundle has no partition identity")
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or any(not isinstance(item, str) or not item for item in sample_ids)
        or len(sample_ids) != len(set(sample_ids))
    ):
        raise ValueError("Questioner smoke bundle has invalid sample identities")
    for field_name, value in (
        ("runtime_exports", runtime_exports),
        ("coverage_exports", coverage_exports),
    ):
        if (
            not isinstance(value, dict)
            or not value
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(path, str)
                or not path
                for key, path in value.items()
            )
        ):
            raise ValueError(f"Questioner smoke bundle has invalid {field_name}")
    if set(runtime_exports) != set(coverage_exports):  # type: ignore[arg-type]
        raise ValueError("Questioner smoke runtime and coverage datasets differ")
    automatic_path = pilot_dir / "automatic_quality.json"
    automatic = read_json(automatic_path)
    if not automatic:
        raise ValueError("Questioner smoke automatic quality report is empty")
    actual_files = sorted(
        path.relative_to(pilot_dir).as_posix()
        for path in pilot_dir.rglob("*")
        if path.is_file()
    )
    if actual_files != ["automatic_quality.json", "manifest.json"]:
        raise ValueError("Questioner smoke bundle contains an unexpected file inventory")
    return manifest


def _source_identity(runtime_export: Path, coverage_export: Path) -> dict[str, Any]:
    runtime_manifest = read_json(runtime_export / "export_manifest.json")
    coverage_manifest = read_json(coverage_export / "export_manifest.json")
    run_manifest = read_json(runtime_export / "runtime" / "run_manifest.json")
    metadata = run_manifest.get("metadata", {})
    questioner = metadata.get("questioner_config", {}) if isinstance(metadata, dict) else {}
    model_calls = read_jsonl(runtime_export / "runtime" / "model_calls.jsonl")
    model_commits: dict[str, str] = {}
    for alias, model_config in dict(run_manifest.get("model_configs", {})).items():
        if isinstance(model_config, dict):
            revision = model_config.get("model_revision") or model_config.get(
                "expected_model_revision"
            )
            if revision:
                model_commits[str(alias)] = str(revision)
    for call in model_calls:
        alias = call.get("model_alias")
        revision = call.get("actual_model_revision") or call.get("model_revision")
        if alias and revision:
            model_commits[str(alias)] = str(revision)
    verifier_prompt_version = questioner.get("verifier_prompt_version")
    prompt_hashes = dict(run_manifest.get("prompt_hashes", {}))
    opportunity_identity = {
        "opportunity_builder_version": questioner.get("opportunity_builder_version"),
        "unit_builder_version": questioner.get("unit_builder_version"),
        "dataset_unit_builders": questioner.get("dataset_unit_builders", {}),
        "relation_selector_version": questioner.get("relation_selector_version"),
        "fts_lexeme_limit": questioner.get("fts_lexeme_limit"),
        "fts_relation_top_k": questioner.get("fts_relation_top_k"),
        "max_opportunities_per_unit": questioner.get("max_opportunities_per_unit"),
    }
    verifier_schema_hash: str | None = None
    verifier_prompt_hash: str | None = None
    verifier_protocol_hash: str | None = None
    if verifier_prompt_version:
        from pmmc.compiler.questioner_prompt import (
            questioner_prompt_sha256,
            questioner_response_schema_sha256,
        )

        verifier_prompt_hash = prompt_hashes.get(str(verifier_prompt_version)) or (
            questioner_prompt_sha256(str(verifier_prompt_version))
        )
        verifier_schema_hash = questioner_response_schema_sha256(
            str(verifier_prompt_version)
        )
        verifier_protocol_hash = stable_hash(
            {
                "prompt_version": verifier_prompt_version,
                "prompt_hash": verifier_prompt_hash,
                "schema_version": questioner.get("verifier_schema_version"),
                "schema_hash": verifier_schema_hash,
                "semantic_revision_max": questioner.get("semantic_revision_max"),
                "model_alias": questioner.get("compiler_model_alias"),
            }
        )
    return {
        "runtime_export_id": runtime_manifest["export_id"],
        "runtime_manifest_sha256": file_sha256(runtime_export / "export_manifest.json"),
        "coverage_export_id": coverage_manifest["export_id"],
        "coverage_manifest_sha256": file_sha256(coverage_export / "export_manifest.json"),
        "compiler_run_id": runtime_manifest.get("run_id"),
        "coverage_run_id": coverage_manifest.get("coverage_run_id"),
        "run_created_at": run_manifest.get("created_at"),
        "questioner_identity": {
            "questioner_protocol_version": questioner.get("protocol_version"),
            "prompt_version": questioner.get("prompt_version"),
            "questioner_config_hash": metadata.get("questioner_config_hash"),
            "prompt_hash": dict(run_manifest.get("prompt_hashes", {})).get(
                questioner.get("prompt_version")
            ),
            "response_schema_hash": metadata.get("questioner_response_schema_hash"),
            "renderer_version": questioner.get("renderer_version"),
            "parser_version": questioner.get("parser_version"),
            "unit_builder_version": questioner.get("unit_builder_version"),
            "gate_version": questioner.get("gate_version"),
            "dedupe_version": questioner.get("dedupe_version"),
            "matcher_version": questioner.get("matcher_version"),
            "opportunity_builder_version": questioner.get(
                "opportunity_builder_version"
            ),
            "relation_selector_version": questioner.get(
                "relation_selector_version"
            ),
            "opportunity_builder_hash": (
                stable_hash(opportunity_identity)
                if questioner.get("opportunity_builder_version")
                else None
            ),
            "verifier_prompt_version": verifier_prompt_version,
            "verifier_prompt_hash": verifier_prompt_hash,
            "verifier_schema_hash": verifier_schema_hash,
            "verifier_protocol_hash": verifier_protocol_hash,
            "model_commits": dict(sorted(model_commits.items())),
        },
    }


def _report_markdown(
    pilot_id: str,
    partition_id: str,
    automatic: dict[str, Any],
    *,
    status: QuestionerPilotStatus,
) -> str:
    lines = [
        f"# Questioner Quality Pilot {pilot_id}",
        "",
        f"- Status: `{status.value}`",
        f"- Partition: `{partition_id}`",
        "- Scientific gate: explicit human approval",
        "",
    ]
    for dataset, summary in sorted(automatic["datasets"].items()):
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Units: {summary['unit_count']}",
                f"- Raw candidates: {summary['raw_candidate_count']}",
                f"- Schema-valid rate: {summary['schema_valid_rate']}",
                "- Gate pass among schema-valid: "
                f"{summary['gate_pass_rate_among_schema_valid']}",
                f"- Zero-valid unit rate: {summary['zero_valid_unit_rate']}",
                f"- Failures: {summary['failure_counts']}",
                "",
            ]
        )
    if status is QuestionerPilotStatus.AWAITING_APPROVAL:
        lines.extend(
            [
                "Automated metrics are report-only. Planner execution remains blocked until ",
                "the signed audit is explicitly approved by the research lead.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Automated metrics are report-only. Planner execution remains blocked until ",
                "human review is finalized and an approval manifest is issued.",
                "",
            ]
        )
    return "\n".join(lines)


def build_pilot_bundle(
    *,
    config: QuestionerPilotConfig,
    partition_dir: Path,
    samples: list[QuestionerPilotSample],
    runtime_exports: dict[DatasetName, Path],
    coverage_exports: dict[DatasetName, Path],
    snapshot_dirs: dict[DatasetName, Path],
    output_root: Path,
) -> tuple[Path, QuestionerPilotManifest]:
    audit_only = config.protocol_version == "questioner-quality-pilot-v3"
    target_status = (
        QuestionerPilotStatus.AWAITING_APPROVAL
        if audit_only
        else QuestionerPilotStatus.AWAITING_REVIEW
    )
    partition = verify_partition(partition_dir)
    _loaded, assignments = load_partition(partition_dir)
    if {item.dataset for item in samples} != set(runtime_exports):
        raise ValueError("pilot samples and Questioner exports cover different datasets")
    if any(
        sample.partition_id != partition.partition_id
        or sample.role is not PartitionRole.DEVELOPMENT
        or len(sample.query_ids) != config.queries_per_dataset
        for sample in samples
    ):
        raise ValueError("pilot samples do not match the frozen development protocol")
    runtime_verifications: dict[str, dict[str, object]] = {}
    if audit_only:
        runtime_verifications = {
            dataset.value: verify_questioner_export(runtime_export)
            for dataset, runtime_export in sorted(
                runtime_exports.items(), key=lambda item: item[0].value
            )
        }
        if any(
            report.get("protocol_version") != "pmmc-questioner-v3"
            or report.get("valid") is not True
            for report in runtime_verifications.values()
        ):
            raise ValueError("Questioner v3 pilot requires verified v3 runtime exports")
    sample_query_ids = {query_id for sample in samples for query_id in sample.query_ids}
    selected_assignments = [
        item for item in assignments if item.query_id in sample_query_ids
    ]
    if len(selected_assignments) != len(sample_query_ids) or any(
        item.role is not PartitionRole.DEVELOPMENT for item in selected_assignments
    ):
        raise ValueError("pilot sample contains a missing or non-development query")
    for sample in samples:
        coverage_query_ids = {
            str(item["query_id"])
            for item in read_jsonl(
                coverage_exports[sample.dataset]
                / "private"
                / "coverage_queries.jsonl"
            )
        }
        if coverage_query_ids != set(sample.query_ids):
            raise ValueError("coverage export does not match the frozen pilot sample")
        compiler_conversations = {
            str(item["conversation_id"])
            for item in read_jsonl(
                runtime_exports[sample.dataset] / "runtime" / "compiler_units.jsonl"
            )
        }
        if compiler_conversations != set(sample.conversation_ids):
            raise ValueError("Questioner export does not match pilot conversations")
    automatic = analyze_questioner_quality(
        config=config,
        runtime_exports=runtime_exports,
        coverage_exports=coverage_exports,
        assignments=selected_assignments,
    )
    source_identities = {
        dataset.value: _source_identity(
            runtime_exports[dataset], coverage_exports[dataset]
        )
        for dataset in sorted(runtime_exports, key=lambda item: item.value)
    }
    pilot_identity = {
        "protocol_version": config.protocol_version,
        "config_hash": config.resolved_hash,
        "partition_id": partition.partition_id,
        "samples": [item.model_dump(mode="json") for item in samples],
        "source_exports": source_identities,
    }
    pilot_id = "questioner-pilot-" + stable_hash(pilot_identity)[:24]
    tasks, review_assignments, quota_report = build_review_tasks(
        pilot_id=pilot_id,
        config=config,
        runtime_exports=runtime_exports,
        coverage_exports=coverage_exports,
        snapshot_dirs=snapshot_dirs,
        assignments=selected_assignments,
    )
    temporary = output_root / ".tmp" / pilot_id
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "public").mkdir(parents=True, mode=0o755)
    (temporary / "private").mkdir(parents=True, mode=0o700)
    (temporary / "private").chmod(0o700)
    public_summary = {
        "pilot_id": pilot_id,
        "protocol_version": config.protocol_version,
        "partition_id": partition.partition_id,
        "status": target_status.value,
        "datasets": automatic["datasets"],
        "marginal_analysis": automatic["marginal_analysis"],
        "underpowered_stratum_size": config.underpowered_stratum_size,
        "scientific_gate": config.scientific_gate,
    }
    write_json(temporary / "public" / "automatic_summary.json", public_summary)
    write_jsonl(temporary / "public" / "coverage_curves.jsonl", automatic["coverage_curves"])
    write_jsonl(
        temporary / "public" / "stratified_coverage.jsonl",
        automatic["stratified_coverage"],
    )
    write_jsonl(
        temporary / "public" / "distribution_summary.jsonl",
        automatic["distributions"],
    )
    write_jsonl(
        temporary / "public" / "failure_summary.jsonl",
        automatic["failure_summary"],
    )
    source_created_at = sorted(
        str(value["run_created_at"])
        for value in source_identities.values()
        if value.get("run_created_at")
    )
    if not source_created_at:
        raise ValueError("Questioner exports do not expose a stable run timestamp")
    write_json(
        temporary / "public" / "pilot_state.json",
        QuestionerPilotState(
            pilot_id=pilot_id,
            status=target_status,
            updated_at=source_created_at[-1],
            metadata={"paper_result": False},
        ).model_dump(mode="json"),
    )
    (temporary / "public" / "pilot_report.md").write_text(
        _report_markdown(
            pilot_id,
            partition.partition_id,
            automatic,
            status=target_status,
        ),
        encoding="utf-8",
    )
    write_json(temporary / "private" / "source_exports.json", source_identities)
    candidate_tasks = [
        item.model_dump(mode="json")
        for item in tasks
        if item.kind is ReviewTaskKind.CANDIDATE_INTRINSIC
    ]
    coverage_tasks = [
        item.model_dump(mode="json")
        for item in tasks
        if item.kind is ReviewTaskKind.QUERY_COVERAGE
    ]
    candidate_task_name = (
        "candidate_audit_cards.jsonl" if audit_only else "candidate_review_tasks.jsonl"
    )
    coverage_task_name = (
        "coverage_audit_cards.jsonl" if audit_only else "coverage_review_tasks.jsonl"
    )
    write_jsonl(temporary / "private" / candidate_task_name, candidate_tasks)
    write_jsonl(temporary / "private" / coverage_task_name, coverage_tasks)
    write_json(temporary / "private" / "review_quota_report.json", quota_report)
    secondary_ids: set[str] = set()
    if not audit_only:
        write_jsonl(
            temporary / "private" / "review_assignments.jsonl",
            [item.model_dump(mode="json") for item in review_assignments],
        )
        primary_template = [
            {
                "task_id": item.task_id,
                "task_hash": item.task_hash,
                "kind": item.kind.value,
                "reviewer_id": None,
                "role": "primary",
                "annotation": None,
            }
            for item in tasks
        ]
        secondary_ids = set(review_assignments[1].task_ids)
        secondary_template = [
            {
                "task_id": item.task_id,
                "task_hash": item.task_hash,
                "kind": item.kind.value,
                "reviewer_id": None,
                "role": "secondary",
                "annotation": None,
            }
            for item in tasks
            if item.task_id in secondary_ids
        ]
        write_jsonl(
            temporary / "private" / "primary.template.jsonl", primary_template
        )
        write_jsonl(
            temporary / "private" / "secondary.template.jsonl", secondary_template
        )
    write_jsonl(
        temporary / "private" / "failure_analysis.jsonl", automatic["failures"]
    )

    audit_manifest: QuestionerPilotAuditManifest | None = None
    if audit_only:
        severe_counts = [
            report.get("severe_terminal_error_count")
            for report in runtime_verifications.values()
        ]
        if any(not isinstance(value, int) for value in severe_counts):
            raise ValueError("Questioner v3 export lacks a terminal-error count")
        severe_terminal_error_count = sum(
            value for value in severe_counts if isinstance(value, int)
        )
        audit_payload = {
            "pilot_id": pilot_id,
            "protocol_version": config.protocol_version,
            "partition_id": partition.partition_id,
            "candidate_audit_count": len(candidate_tasks),
            "coverage_audit_count": len(coverage_tasks),
            "candidate_audit_sha256": file_sha256(
                temporary / "private" / candidate_task_name
            ),
            "coverage_audit_sha256": file_sha256(
                temporary / "private" / coverage_task_name
            ),
            "automatic_summary_sha256": file_sha256(
                temporary / "public" / "automatic_summary.json"
            ),
            "source_runtime_manifest_hashes": {
                key: str(value["runtime_manifest_sha256"])
                for key, value in source_identities.items()
            },
            "source_coverage_manifest_hashes": {
                key: str(value["coverage_manifest_sha256"])
                for key, value in source_identities.items()
            },
            "automatic_integrity_passed": True,
            "integrity_issue_types": [],
            "severe_terminal_error_count": severe_terminal_error_count,
            "metadata": {
                "paper_result": False,
                "approval_requires_zero_severe_terminal_errors": True,
                "runtime_export_verification": {
                    dataset: {
                        "protocol_version": report.get("protocol_version"),
                        "unit_count": report.get("unit_count"),
                        "eligible_candidate_count": report.get(
                            "eligible_candidate_count"
                        ),
                        "verification_batch_count": report.get(
                            "verification_batch_count"
                        ),
                        "revised_candidate_count": report.get(
                            "revised_candidate_count"
                        ),
                    }
                    for dataset, report in sorted(runtime_verifications.items())
                },
            },
        }
        audit_manifest = QuestionerPilotAuditManifest(
            **audit_payload,
            content_hash=stable_hash(audit_payload),
        )
        write_json(
            temporary / "private" / "audit_manifest.json",
            audit_manifest.model_dump(mode="json"),
        )

    for path in (temporary / "private").rglob("*"):
        if path.is_file():
            path.chmod(0o600)

    artifact_refs = [
        QuestionerPilotArtifactRef(
            role=("private" if path.is_relative_to(temporary / "private") else "public"),
            relative_path=path.relative_to(temporary).as_posix(),
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
            private=path.is_relative_to(temporary / "private"),
        )
        for path in sorted(temporary.rglob("*"))
        if path.is_file()
    ]
    manifest_payload = {
        "pilot_id": pilot_id,
        "schema_version": QUESTIONER_PILOT_SCHEMA_VERSION,
        "protocol_version": config.protocol_version,
        "config_hash": config.resolved_hash,
        "partition_id": partition.partition_id,
        "samples": [item.model_dump(mode="json") for item in samples],
        "status": target_status.value,
        "source_runtime_export_ids": {
            key: str(value["runtime_export_id"])
            for key, value in source_identities.items()
        },
        "source_coverage_export_ids": {
            key: str(value["coverage_export_id"])
            for key, value in source_identities.items()
        },
        "artifacts": [item.model_dump(mode="json") for item in artifact_refs],
        "metadata": {
            "paper_result": False,
            "review_task_count": len(tasks),
            "candidate_review_count": len(candidate_tasks),
            "coverage_review_count": len(coverage_tasks),
            "secondary_review_count": len(secondary_ids),
            "audit_manifest_hash": (
                audit_manifest.content_hash if audit_manifest is not None else None
            ),
        },
    }
    manifest = QuestionerPilotManifest(
        **manifest_payload,
        content_hash=stable_hash(manifest_payload),
    )
    write_json(temporary / "manifest.json", manifest.model_dump(mode="json"))
    destination = output_root / pilot_id
    if destination.exists():
        existing = verify_pilot_bundle(destination)
        if existing.content_hash != manifest.content_hash:
            raise ValueError("pilot ID conflicts with an existing bundle")
        shutil.rmtree(temporary)
        return destination, existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    return destination, manifest


def verify_pilot_bundle(pilot_dir: Path) -> QuestionerPilotManifest:
    manifest = QuestionerPilotManifest.model_validate(read_json(pilot_dir / "manifest.json"))
    audit_only = manifest.protocol_version == "questioner-quality-pilot-v3"
    if pilot_dir.name != manifest.pilot_id:
        raise ValueError("pilot directory name does not match pilot_id")
    private_dir = pilot_dir / "private"
    if audit_only and stat.S_IMODE(private_dir.stat().st_mode) != 0o700:
        raise ValueError("private pilot directory must use mode 0700")
    expected = sorted(item.relative_path for item in manifest.artifacts)
    actual = sorted(
        path.relative_to(pilot_dir).as_posix()
        for path in pilot_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    if expected != actual:
        raise ValueError("pilot files differ from the signed inventory")
    for item in manifest.artifacts:
        path = pilot_dir / item.relative_path
        if (
            path.stat().st_size != item.size_bytes
            or file_sha256(path) != item.sha256
        ):
            raise ValueError(f"pilot artifact checksum mismatch: {item.relative_path}")
        data = path.read_bytes()
        if any(marker in data for marker in _SECRET_MARKERS):
            raise ValueError(
                "pilot artifact contains a forbidden secret marker: "
                f"{item.relative_path}"
            )
        if item.relative_path.startswith("public/") and path.suffix in {
            ".json",
            ".jsonl",
        }:
            values = read_jsonl(path) if path.suffix == ".jsonl" else [read_json(path)]
            forbidden = (
                set().union(*(_walk_keys(value) for value in values))
                & _PUBLIC_FORBIDDEN_KEYS
            )
            if forbidden:
                raise ValueError(
                    "public pilot artifact contains private fields: "
                    + ", ".join(sorted(forbidden))
                )
        if audit_only and item.private and stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ValueError(f"private pilot artifact must use mode 0600: {item.relative_path}")
    payload = manifest.model_dump(mode="json", exclude={"content_hash"})
    valid_hash = stable_hash(payload) == manifest.content_hash
    if not valid_hash and manifest.protocol_version == "questioner-quality-pilot-v1":
        valid_hash = any(
            stable_hash({**payload, "schema_version": version})
            == manifest.content_hash
            for version in ("0.1.0", "0.2.0")
        )
    if not valid_hash:
        raise ValueError("pilot manifest content hash mismatch")
    candidate_path = pilot_dir / "private" / (
        "candidate_audit_cards.jsonl"
        if audit_only
        else "candidate_review_tasks.jsonl"
    )
    coverage_path = pilot_dir / "private" / (
        "coverage_audit_cards.jsonl"
        if audit_only
        else "coverage_review_tasks.jsonl"
    )
    candidate_count = len(read_jsonl(candidate_path))
    coverage_count = len(read_jsonl(coverage_path))
    expected_tasks = int(manifest.metadata.get("review_task_count", -1))
    if candidate_count + coverage_count != expected_tasks:
        raise ValueError("pilot review task count does not reconcile")
    if candidate_count != int(manifest.metadata.get("candidate_review_count", -1)):
        raise ValueError("pilot candidate review count does not reconcile")
    if coverage_count != int(manifest.metadata.get("coverage_review_count", -1)):
        raise ValueError("pilot coverage review count does not reconcile")
    if audit_only:
        if manifest.status is not QuestionerPilotStatus.AWAITING_APPROVAL:
            raise ValueError("Questioner v3 pilot must await explicit approval")
        audit = QuestionerPilotAuditManifest.model_validate(
            read_json(pilot_dir / "private" / "audit_manifest.json")
        )
        audit_payload = audit.model_dump(mode="json", exclude={"content_hash"})
        if stable_hash(audit_payload) != audit.content_hash:
            raise ValueError("Questioner v3 audit manifest content hash mismatch")
        if (
            audit.pilot_id != manifest.pilot_id
            or audit.partition_id != manifest.partition_id
            or audit.candidate_audit_count != candidate_count
            or audit.coverage_audit_count != coverage_count
            or audit.candidate_audit_sha256 != file_sha256(candidate_path)
            or audit.coverage_audit_sha256 != file_sha256(coverage_path)
            or audit.automatic_summary_sha256
            != file_sha256(pilot_dir / "public" / "automatic_summary.json")
            or manifest.metadata.get("audit_manifest_hash") != audit.content_hash
        ):
            raise ValueError("Questioner v3 audit manifest does not match its bundle")
        forbidden_review_files = {
            "review_assignments.jsonl",
            "primary.template.jsonl",
            "secondary.template.jsonl",
        }
        if any((pilot_dir / "private" / name).exists() for name in forbidden_review_files):
            raise ValueError("Questioner v3 audit bundle contains legacy review assignments")
        return manifest

    review_assignments = [
        ReviewAssignment.model_validate(item)
        for item in read_jsonl(pilot_dir / "private" / "review_assignments.jsonl")
    ]
    assignments_by_role = {item.role: item for item in review_assignments}
    if set(assignments_by_role) != {ReviewRole.PRIMARY, ReviewRole.SECONDARY}:
        raise ValueError("pilot must contain primary and secondary assignments")
    task_ids = {
        str(item["task_id"])
        for item in [
            *read_jsonl(candidate_path),
            *read_jsonl(coverage_path),
        ]
    }
    primary = assignments_by_role[ReviewRole.PRIMARY]
    secondary = assignments_by_role[ReviewRole.SECONDARY]
    if set(primary.task_ids) != task_ids or not set(secondary.task_ids) <= task_ids:
        raise ValueError("pilot review assignments reference the wrong task set")
    for assignment in review_assignments:
        if stable_hash(assignment.task_ids) != assignment.assignment_hash:
            raise ValueError("pilot review assignment hash mismatch")
    if len(secondary.task_ids) != int(
        manifest.metadata.get("secondary_review_count", -1)
    ):
        raise ValueError("pilot secondary review count does not reconcile")
    return manifest


def render_review_workspace(
    *,
    pilot_dir: Path,
    benchmark_root: Path,
    output_dir: Path,
) -> Path:
    pilot_manifest = verify_pilot_bundle(pilot_dir)
    if output_dir.is_relative_to(pilot_dir):
        raise ValueError("review workspace must be outside the signed pilot bundle")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    audit_only = pilot_manifest.protocol_version == "questioner-quality-pilot-v3"
    candidate_name = (
        "candidate_audit_cards.jsonl"
        if audit_only
        else "candidate_review_tasks.jsonl"
    )
    coverage_name = (
        "coverage_audit_cards.jsonl"
        if audit_only
        else "coverage_review_tasks.jsonl"
    )
    tasks = [
        *read_jsonl(pilot_dir / "private" / candidate_name),
        *read_jsonl(pilot_dir / "private" / coverage_name),
    ]
    assignments = (
        {"audit": [str(item["task_id"]) for item in tasks]}
        if audit_only
        else {
            str(item["role"]): list(item["task_ids"])
            for item in read_jsonl(
                pilot_dir / "private" / "review_assignments.jsonl"
            )
        }
    )
    benchmark_root = benchmark_root.resolve()
    if not benchmark_root.is_dir():
        raise ValueError("benchmark root does not exist")
    thumbnail_dir = output_dir / "thumbnails"
    thumbnail_dir.mkdir()

    def image_refs(value: object) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                if item.get("image_id") and item.get("relative_path"):
                    found.append(item)
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        return found

    thumbnails: dict[tuple[str, str], str] = {}
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - the project image includes Pillow.
        raise RuntimeError("render-review requires Pillow") from exc
    for ref in sorted(
        image_refs([item["payload"] for item in tasks]),
        key=lambda item: (str(item["image_id"]), str(item["relative_path"])),
    ):
        relative = Path(str(ref["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("review image path must be dataset-relative")
        source = (benchmark_root / relative).resolve()
        if not source.is_relative_to(benchmark_root) or not source.is_file():
            raise ValueError(f"review image is unavailable: {relative.as_posix()}")
        expected_sha = str(ref.get("sha256") or "")
        actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if expected_sha and expected_sha != actual_sha:
            raise ValueError(f"review image SHA256 mismatch: {relative.as_posix()}")
        key = (str(ref["image_id"]), actual_sha)
        if key in thumbnails:
            continue
        destination = thumbnail_dir / f"{actual_sha[:24]}.jpg"
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            image.thumbnail((640, 640))
            image.save(destination, format="JPEG", quality=88, optimize=True)
        thumbnails[key] = destination.relative_to(output_dir).as_posix()

    sections: list[str] = []
    for task in sorted(tasks, key=lambda item: str(item["task_id"])):
        payload = task["payload"]
        task_images = []
        for ref in image_refs(payload):
            expected_sha = str(ref.get("sha256") or "")
            matching = [
                path
                for (image_id, sha256), path in thumbnails.items()
                if image_id == str(ref["image_id"])
                and (not expected_sha or expected_sha == sha256)
            ]
            if matching:
                task_images.append(
                    "<figure><img loading='lazy' src='"
                    + html.escape(matching[0], quote=True)
                    + "' alt='Evidence image'><figcaption>"
                    + html.escape(str(ref["image_id"]))
                    + "</figcaption></figure>"
                )
        sections.append(
            "<section data-task-id='"
            + html.escape(str(task["task_id"]), quote=True)
            + "'><h2>"
            + html.escape(str(task["task_id"]))
            + "</h2><p>"
            + html.escape(str(task["kind"]))
            + " / "
            + html.escape(str(task["bucket"]))
            + "</p><div class='images'>"
            + "".join(task_images)
            + "</div><pre>"
            + html.escape(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            + "</pre><div class='annotation'></div></section>"
        )
    task_payload = json.dumps(tasks, ensure_ascii=False, sort_keys=True).replace(
        "</", "<\\/"
    )
    assignment_payload = json.dumps(assignments, sort_keys=True).replace("</", "<\\/")
    review_script = """
const tasks = JSON.parse(document.querySelector('#tasks').textContent);
const assignments = JSON.parse(document.querySelector('#assignments').textContent);
const ratings = ['', 'no', 'partial', 'yes'];
const roots = [
  '', 'questioner', 'unit_builder', 'evidence_lineage', 'gold_resolution',
  'matcher', 'no_error', 'unresolvable'
];
function selectField(name, values) {
  const options = values.map(
    value => `<option value='${value}'>${value || '--'}</option>`
  ).join('');
  return `<label class='field'>${name}<select data-field='${name}'>` +
    `${options}</select></label>`;
}
function checkboxField(name) {
  return `<label class='field'><input type='checkbox' data-field='${name}'>` +
    `${name}</label>`;
}
function render() {
  const role = document.querySelector('#role').value;
  const allowed = new Set(assignments[role]);
  for (const task of tasks) {
    const section = document.querySelector(`[data-task-id='${task.task_id}']`);
    section.hidden = !allowed.has(task.task_id);
    if (section.dataset.ready) continue;
    let fields = '';
    if (task.kind === 'candidate_intrinsic') {
      const required = [
        'answerability', 'expected_answer_support', 'evidence_sufficiency',
        'future_usefulness', 'taxonomy_correctness', 'modality_correctness'
      ];
      for (const name of required) fields += selectField(name, ratings);
      fields += selectField('visual_necessity', ratings);
      fields += selectField('multi_hop_validity', ratings);
      for (const name of ['template_like', 'duplicate', 'over_trivial', 'leakage']) {
        fields += checkboxField(name);
      }
    } else {
      fields += selectField('semantic_coverage', ratings);
      fields += selectField('failure_root', roots);
      fields += selectField('automatic_classification_correct', ['', 'true', 'false']);
      fields += checkboxField('leakage');
    }
    fields += `<label class='field'>notes <input data-field='notes'></label>`;
    section.querySelector('.annotation').innerHTML = fields;
    section.dataset.ready = '1';
  }
}
document.querySelector('#role').addEventListener('change', render);
render();
document.querySelector('#export').addEventListener('click', () => {
  const role = document.querySelector('#role').value;
  const reviewer = document.querySelector('#reviewer').value.trim();
  if (!reviewer) {
    alert('Reviewer ID is required');
    return;
  }
  const byId = new Map(tasks.map(task => [task.task_id, task]));
  const rows = [];
  for (const id of assignments[role]) {
    const task = byId.get(id);
    const section = document.querySelector(`[data-task-id='${id}']`);
    const annotation = {
      task_id: id,
      task_hash: task.task_hash,
      role,
      reviewer_id: reviewer,
      annotation_hash: 'TO_BE_COMPUTED'
    };
    let missing = false;
    for (const input of section.querySelectorAll('[data-field]')) {
      const name = input.dataset.field;
      if (name === 'automatic_classification_correct') {
        annotation[name] = input.value ? input.value === 'true' : null;
      } else {
        annotation[name] = input.type === 'checkbox' ? input.checked : input.value || null;
      }
      const optional = ['visual_necessity', 'multi_hop_validity'];
      if (input.tagName === 'SELECT' && !input.value && !optional.includes(name)) {
        missing = true;
      }
    }
    if (missing) {
      alert(`Complete required fields for ${id}`);
      return;
    }
    rows.push(JSON.stringify(annotation));
  }
  const blob = new Blob([rows.join('\\n') + '\\n'], {
    type: 'application/x-ndjson'
  });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `${role}.annotations.jsonl`;
  link.click();
  URL.revokeObjectURL(link.href);
});
"""
    role_options = (
        "<option>audit</option>"
        if audit_only
        else "<option>primary</option><option>secondary</option>"
    )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Questioner review</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1180px;margin:auto;padding:20px}"
        "header{position:sticky;top:0;background:#fff;border-bottom:1px solid #bbb;padding:12px 0}"
        "section{border-bottom:1px solid #ccc;padding:24px 0}pre{white-space:pre-wrap}"
        ".images{display:flex;gap:12px;flex-wrap:wrap}.images img{max-width:320px;max-height:320px}"
        ".field{display:inline-flex;gap:6px;margin:6px 16px 6px 0}select,input{font:inherit}"
        "</style></head><body><header><h1>Private Questioner review</h1>"
        "<label>Role <select id='role'>"
        + role_options
        + "</select></label> <label>Reviewer ID <input id='reviewer' required></label> "
        + "<button id='export'>Export annotation JSONL</button></header>"
        + "".join(sections)
        + "<script id='tasks' type='application/json'>"
        + task_payload
        + "</script><script id='assignments' type='application/json'>"
        + assignment_payload
        + "</script><script>"
        + review_script
        + "</script></body></html>"
    )
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    write_json(
        output_dir / "workspace_manifest.json",
        {
            "pilot_id": verify_pilot_bundle(pilot_dir).pilot_id,
            "task_count": len(tasks),
            "thumbnail_count": len(thumbnails),
            "thumbnail_sha256": {
                path: file_sha256(output_dir / path)
                for path in sorted(thumbnails.values())
            },
        },
    )
    return output_dir
