"""Human review finalization and immutable Questioner approval manifests."""

from __future__ import annotations

import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from pmmc.pilots.bundle import verify_pilot_bundle
from pmmc.pilots.io import (
    file_sha256,
    read_json,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)
from pmmc.schemas import (
    CandidateQualityAnnotation,
    CoverageQualityAnnotation,
    PilotDecisionValue,
    QuestionerFreezeManifest,
    QuestionerPilotDecision,
    QuestionerPilotState,
    QuestionerPilotStatus,
    QuestionerReviewTask,
    ReviewAdjudication,
    ReviewAgreementReport,
    ReviewAssignment,
    ReviewRating,
    ReviewRole,
    ReviewTaskKind,
)
from pmmc.schemas.base import utc_now

Annotation = CandidateQualityAnnotation | CoverageQualityAnnotation

_CANDIDATE_CORE_FIELDS = (
    "answerability",
    "expected_answer_support",
    "evidence_sufficiency",
    "taxonomy_correctness",
    "modality_correctness",
    "leakage",
)
_COVERAGE_CORE_FIELDS = (
    "semantic_coverage",
    "failure_root",
    "leakage",
)
_ORDERED_FIELDS = {
    "answerability",
    "expected_answer_support",
    "evidence_sufficiency",
    "future_usefulness",
    "taxonomy_correctness",
    "modality_correctness",
    "visual_necessity",
    "multi_hop_validity",
    "semantic_coverage",
}


def _tasks(pilot_dir: Path) -> dict[str, QuestionerReviewTask]:
    rows = [
        *read_jsonl(pilot_dir / "private" / "candidate_review_tasks.jsonl"),
        *read_jsonl(pilot_dir / "private" / "coverage_review_tasks.jsonl"),
    ]
    tasks = {str(row["task_id"]): QuestionerReviewTask.model_validate(row) for row in rows}
    if len(tasks) != len(rows):
        raise ValueError("pilot review tasks contain duplicate IDs")
    return tasks


def _assignments(pilot_dir: Path) -> dict[ReviewRole, ReviewAssignment]:
    values = [
        ReviewAssignment.model_validate(row)
        for row in read_jsonl(pilot_dir / "private" / "review_assignments.jsonl")
    ]
    result = {item.role: item for item in values}
    if set(result) != {ReviewRole.PRIMARY, ReviewRole.SECONDARY}:
        raise ValueError("pilot requires primary and secondary assignments")
    return result


def _annotation_hash(payload: dict[str, Any]) -> str:
    return stable_hash(
        {key: value for key, value in payload.items() if key != "annotation_hash"}
    )


def _normalize_annotation(
    raw: dict[str, Any],
    *,
    task: QuestionerReviewTask,
    role: ReviewRole,
) -> Annotation:
    payload = dict(raw.get("annotation", raw))
    if not isinstance(payload, dict):
        raise ValueError(f"annotation payload must be an object: {task.task_id}")
    payload.pop("kind", None)
    payload.setdefault("task_id", task.task_id)
    payload.setdefault("task_hash", task.task_hash)
    payload.setdefault("role", role.value)
    if payload.get("task_id") != task.task_id or payload.get("task_hash") != task.task_hash:
        raise ValueError(f"annotation task identity mismatch: {task.task_id}")
    if payload.get("role") != role.value:
        raise ValueError(f"annotation role mismatch: {task.task_id}")
    reviewer_id = payload.get("reviewer_id") or raw.get("reviewer_id")
    if not reviewer_id:
        raise ValueError(f"annotation requires reviewer_id: {task.task_id}")
    payload["reviewer_id"] = reviewer_id
    payload.setdefault(
        "annotation_id",
        "review-annotation-" + stable_hash([task.task_id, role.value, reviewer_id])[:24],
    )
    expected_hash = _annotation_hash(payload)
    supplied_hash = payload.get("annotation_hash")
    if supplied_hash not in {None, "", "TO_BE_COMPUTED", expected_hash}:
        raise ValueError(f"annotation hash mismatch: {task.task_id}")
    payload["annotation_hash"] = expected_hash
    model = (
        CandidateQualityAnnotation
        if task.kind is ReviewTaskKind.CANDIDATE_INTRINSIC
        else CoverageQualityAnnotation
    )
    return model.model_validate(payload)


def validate_annotations(
    *,
    pilot_dir: Path,
    annotation_path: Path,
    role: ReviewRole,
    require_complete: bool = True,
) -> list[Annotation]:
    verify_pilot_bundle(pilot_dir)
    tasks = _tasks(pilot_dir)
    assignment = _assignments(pilot_dir)[role]
    rows = read_jsonl(annotation_path)
    seen: set[str] = set()
    annotations: list[Annotation] = []
    for row in rows:
        task_id = str(row.get("task_id") or dict(row.get("annotation", {})).get("task_id"))
        if task_id not in tasks:
            raise ValueError(f"annotation references an unknown task: {task_id}")
        if task_id not in assignment.task_ids:
            raise ValueError(f"annotation is outside its blind assignment: {task_id}")
        if task_id in seen:
            raise ValueError(f"review task was annotated more than once: {task_id}")
        seen.add(task_id)
        annotations.append(
            _normalize_annotation(row, task=tasks[task_id], role=role)
        )
    if require_complete and seen != set(assignment.task_ids):
        missing = sorted(set(assignment.task_ids) - seen)
        raise ValueError(f"review assignment is incomplete ({len(missing)} missing tasks)")
    annotations.sort(key=lambda item: item.task_id)
    return annotations


def _cohen_kappa(left: list[object], right: list[object]) -> float | None:
    if not left:
        return None
    labels = sorted({str(item) for item in [*left, *right]})
    if len(labels) < 2:
        return None
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(str(item) for item in left)
    right_counts = Counter(str(item) for item in right)
    expected = sum(
        left_counts[label] * right_counts[label] for label in labels
    ) / (len(left) * len(left))
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def _weighted_kappa(left: list[object], right: list[object]) -> float | None:
    def rating(value: object) -> ReviewRating:
        return value if isinstance(value, ReviewRating) else ReviewRating(str(value))

    usable = [
        (rating(a), rating(b))
        for a, b in zip(left, right, strict=True)
        if a is not None and b is not None
    ]
    if not usable:
        return None
    order = {ReviewRating.NO: 0, ReviewRating.PARTIAL: 1, ReviewRating.YES: 2}
    left_values = [order[a] for a, _ in usable]
    right_values = [order[b] for _, b in usable]
    if len(set(left_values + right_values)) < 2:
        return None
    observed = sum(
        (a - b) ** 2 / 4
        for a, b in zip(left_values, right_values, strict=True)
    ) / len(usable)
    left_counts = Counter(left_values)
    right_counts = Counter(right_values)
    expected = sum(
        left_counts[a] * right_counts[b] * ((a - b) ** 2 / 4)
        for a in range(3)
        for b in range(3)
    ) / (len(usable) * len(usable))
    if math.isclose(expected, 0.0):
        return None
    return 1 - observed / expected


def _validate_resolved_annotation(
    *,
    task_id: str,
    resolved: dict[str, Any],
    primary: Annotation,
    secondary: Annotation,
) -> Annotation:
    immutable = {
        "annotation_id",
        "task_id",
        "task_hash",
        "reviewer_id",
        "role",
        "annotation_hash",
    }
    if immutable & set(resolved):
        raise ValueError(f"adjudication cannot replace annotation identity: {task_id}")
    model = type(primary)
    if not isinstance(secondary, model):
        raise ValueError(f"adjudication annotation kinds differ: {task_id}")
    unknown = set(resolved) - set(model.model_fields)
    if unknown:
        raise ValueError(f"adjudication contains unknown fields: {task_id}")
    fields = (
        _CANDIDATE_CORE_FIELDS
        if isinstance(primary, CandidateQualityAnnotation)
        else _COVERAGE_CORE_FIELDS
    )
    differing = {
        field
        for field in fields
        if getattr(primary, field) != getattr(secondary, field)
    }
    if not differing <= set(resolved):
        raise ValueError(
            f"adjudication does not resolve every core disagreement: {task_id}"
        )
    payload = primary.model_dump(mode="json")
    payload.update(resolved)
    payload["annotation_hash"] = _annotation_hash(payload)
    return model.model_validate(payload)


def _agreement(
    *,
    pilot_id: str,
    primary: dict[str, Annotation],
    secondary: dict[str, Annotation],
    adjudicated: set[str],
) -> ReviewAgreementReport:
    overlap = sorted(set(primary) & set(secondary))
    raw: dict[str, float | None] = {}
    kappa: dict[str, float | None] = {}
    weighted: dict[str, float | None] = {}
    disagreements: set[str] = set()
    all_fields = sorted(_ORDERED_FIELDS | {"failure_root", "leakage"})
    for field in all_fields:
        pairs = [
            (getattr(primary[task_id], field, None), getattr(secondary[task_id], field, None))
            for task_id in overlap
            if hasattr(primary[task_id], field) and hasattr(secondary[task_id], field)
        ]
        if not pairs:
            raw[field] = None
            kappa[field] = None
            if field in _ORDERED_FIELDS:
                weighted[field] = None
            continue
        left = [item[0] for item in pairs]
        right = [item[1] for item in pairs]
        raw[field] = sum(a == b for a, b in pairs) / len(pairs)
        kappa[field] = _cohen_kappa(left, right)
        if field in _ORDERED_FIELDS:
            weighted[field] = _weighted_kappa(left, right)
    for task_id in overlap:
        fields = (
            _CANDIDATE_CORE_FIELDS
            if isinstance(primary[task_id], CandidateQualityAnnotation)
            else _COVERAGE_CORE_FIELDS
        )
        if any(
            getattr(primary[task_id], field) != getattr(secondary[task_id], field)
            for field in fields
        ):
            disagreements.add(task_id)
    payload = {
        "pilot_id": pilot_id,
        "overlap_count": len(overlap),
        "raw_agreement": raw,
        "cohen_kappa": kappa,
        "weighted_kappa": weighted,
        "disagreement_task_ids": sorted(disagreements),
        "adjudicated_task_ids": sorted(adjudicated),
    }
    return ReviewAgreementReport(**payload, content_hash=stable_hash(payload))


def _load_adjudications(
    path: Path | None,
    *,
    primary: dict[str, Annotation],
    secondary: dict[str, Annotation],
) -> dict[str, ReviewAdjudication]:
    if path is None:
        return {}
    result: dict[str, ReviewAdjudication] = {}
    for raw in read_jsonl(path):
        task_id = str(raw.get("task_id"))
        if task_id not in primary or task_id not in secondary:
            raise ValueError(f"adjudication is not for a blind-overlap task: {task_id}")
        if raw.get("primary_annotation_hash") != primary[task_id].annotation_hash:
            raise ValueError(f"adjudication primary hash mismatch: {task_id}")
        if raw.get("secondary_annotation_hash") != secondary[task_id].annotation_hash:
            raise ValueError(f"adjudication secondary hash mismatch: {task_id}")
        resolved = dict(raw.get("resolved_annotation", {}))
        resolved_hash = stable_hash(resolved)
        if raw.get("resolved_annotation_hash") != resolved_hash:
            raise ValueError(f"adjudication resolved hash mismatch: {task_id}")
        payload = {key: value for key, value in raw.items() if key != "content_hash"}
        content_hash = stable_hash(payload)
        supplied = raw.get("content_hash")
        if supplied not in {None, "", "TO_BE_COMPUTED", content_hash}:
            raise ValueError(f"adjudication content hash mismatch: {task_id}")
        payload["content_hash"] = content_hash
        item = ReviewAdjudication.model_validate(payload)
        _validate_resolved_annotation(
            task_id=task_id,
            resolved=resolved,
            primary=primary[task_id],
            secondary=secondary[task_id],
        )
        if task_id in result:
            raise ValueError(f"task was adjudicated more than once: {task_id}")
        result[task_id] = item
    return result


def finalize_review(
    *,
    pilot_dir: Path,
    primary_path: Path,
    secondary_path: Path,
    adjudications_path: Path | None,
    output_root: Path | None = None,
) -> Path:
    pilot = verify_pilot_bundle(pilot_dir)
    primary_values = validate_annotations(
        pilot_dir=pilot_dir,
        annotation_path=primary_path,
        role=ReviewRole.PRIMARY,
    )
    secondary_values = validate_annotations(
        pilot_dir=pilot_dir,
        annotation_path=secondary_path,
        role=ReviewRole.SECONDARY,
    )
    primary = {item.task_id: item for item in primary_values}
    secondary = {item.task_id: item for item in secondary_values}
    primary_reviewers = {item.reviewer_id for item in primary_values}
    secondary_reviewers = {item.reviewer_id for item in secondary_values}
    if len(primary_reviewers) != 1 or len(secondary_reviewers) != 1:
        raise ValueError("each review role must use one stable reviewer identity")
    if primary_reviewers == secondary_reviewers:
        raise ValueError("primary and secondary reviewers must be different")
    adjudications = _load_adjudications(
        adjudications_path, primary=primary, secondary=secondary
    )
    preliminary = _agreement(
        pilot_id=pilot.pilot_id,
        primary=primary,
        secondary=secondary,
        adjudicated=set(adjudications),
    )
    unresolved = set(preliminary.disagreement_task_ids) - set(adjudications)
    if unresolved:
        raise ValueError(
            f"core review disagreements require adjudication ({len(unresolved)} unresolved)"
        )
    for task_id, annotation in primary.items():
        if task_id in adjudications:
            resolved = _validate_resolved_annotation(
                task_id=task_id,
                resolved=dict(adjudications[task_id].resolved_annotation),
                primary=annotation,
                secondary=secondary[task_id],
            )
            if resolved.leakage:
                raise ValueError("adjudicated review still contains unresolved leakage")
        elif annotation.leakage:
            raise ValueError("final review contains unresolved leakage")
    agreement = _agreement(
        pilot_id=pilot.pilot_id,
        primary=primary,
        secondary=secondary,
        adjudicated=set(adjudications),
    )
    identity = {
        "pilot_content_hash": pilot.content_hash,
        "primary_hashes": [item.annotation_hash for item in primary_values],
        "secondary_hashes": [item.annotation_hash for item in secondary_values],
        "adjudication_hashes": [
            item.content_hash
            for item in sorted(
                adjudications.values(), key=lambda item: item.task_id
            )
        ],
        "agreement_hash": agreement.content_hash,
    }
    review_hash = stable_hash(identity)
    root = output_root or pilot_dir.parent
    temporary = root / ".tmp" / f"{pilot.pilot_id}-review-{review_hash[:24]}"
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "private").mkdir(parents=True)
    write_jsonl(
        temporary / "private" / "primary_annotations.jsonl",
        [item.model_dump(mode="json") for item in primary_values],
    )
    write_jsonl(
        temporary / "private" / "secondary_annotations.jsonl",
        [item.model_dump(mode="json") for item in secondary_values],
    )
    write_jsonl(
        temporary / "private" / "adjudications.jsonl",
        [
            item.model_dump(mode="json")
            for item in sorted(adjudications.values(), key=lambda item: item.task_id)
        ],
    )
    write_json(
        temporary / "private" / "agreement_report.json",
        agreement.model_dump(mode="json"),
    )
    files = [
        {
            "path": path.relative_to(temporary).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(temporary.rglob("*"))
        if path.is_file()
    ]
    review_manifest = {
        "review_id": "questioner-review-" + review_hash[:24],
        "pilot_id": pilot.pilot_id,
        "pilot_content_hash": pilot.content_hash,
        "status": "finalized",
        "primary_reviewer_id": next(iter(primary_reviewers)),
        "secondary_reviewer_id": next(iter(secondary_reviewers)),
        "primary_count": len(primary_values),
        "secondary_count": len(secondary_values),
        "adjudication_count": len(adjudications),
        "agreement_report_hash": agreement.content_hash,
        "files": files,
        "content_hash": review_hash,
    }
    write_json(temporary / "review_manifest.json", review_manifest)
    destination = root / str(review_manifest["review_id"])
    if destination.exists():
        verify_review(destination)
        shutil.rmtree(temporary)
    else:
        temporary.replace(destination)
    return destination


def verify_review(review_dir: Path) -> dict[str, Any]:
    manifest = read_json(review_dir / "review_manifest.json")
    expected = sorted(str(item["path"]) for item in manifest.get("files", []))
    actual = sorted(
        path.relative_to(review_dir).as_posix()
        for path in review_dir.rglob("*")
        if path.is_file() and path.name != "review_manifest.json"
    )
    if expected != actual:
        raise ValueError("review files differ from the signed inventory")
    for item in manifest["files"]:
        path = review_dir / str(item["path"])
        if (
            path.stat().st_size != item["size_bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise ValueError(f"review artifact checksum mismatch: {item['path']}")
    identity = {
        "pilot_content_hash": manifest["pilot_content_hash"],
        "primary_hashes": [
            item["annotation_hash"]
            for item in read_jsonl(review_dir / "private" / "primary_annotations.jsonl")
        ],
        "secondary_hashes": [
            item["annotation_hash"]
            for item in read_jsonl(review_dir / "private" / "secondary_annotations.jsonl")
        ],
        "adjudication_hashes": [
            item["content_hash"]
            for item in read_jsonl(review_dir / "private" / "adjudications.jsonl")
        ],
        "agreement_hash": read_json(
            review_dir / "private" / "agreement_report.json"
        )["content_hash"],
    }
    if stable_hash(identity) != manifest["content_hash"]:
        raise ValueError("review manifest content hash mismatch")
    return manifest


def _approval_source_identity(
    pilot_dir: Path,
    *,
    audit_mode: bool,
) -> tuple[dict[str, Any], dict[str, object], dict[str, str]]:
    sources = read_json(pilot_dir / "private" / "source_exports.json")
    identities = [
        dict(value["questioner_identity"])
        for _, value in sorted(sources.items())
    ]
    if not identities or any(value != identities[0] for value in identities[1:]):
        raise ValueError("pilot datasets do not share one frozen Questioner identity")
    identity = identities[0]
    model_commits = {
        str(alias): str(revision)
        for alias, revision in dict(identity.get("model_commits", {})).items()
    }
    required: dict[str, object] = {
        "questioner_config_hash": identity.get("questioner_config_hash"),
        "prompt_hash": identity.get("prompt_hash"),
        "response_schema_hash": identity.get("response_schema_hash"),
        "renderer_version": identity.get("renderer_version"),
        "parser_version": identity.get("parser_version"),
        "unit_builder_version": identity.get("unit_builder_version"),
        "gate_version": identity.get("gate_version"),
        "dedupe_version": identity.get("dedupe_version"),
        "matcher_version": identity.get("matcher_version"),
    }
    if audit_mode:
        required.update(
            {
                "questioner_protocol_version": identity.get(
                    "questioner_protocol_version"
                ),
                "prompt_version": identity.get("prompt_version"),
                "opportunity_builder_version": identity.get(
                    "opportunity_builder_version"
                ),
                "relation_selector_version": identity.get(
                    "relation_selector_version"
                ),
                "opportunity_builder_hash": identity.get(
                    "opportunity_builder_hash"
                ),
                "verifier_prompt_version": identity.get(
                    "verifier_prompt_version"
                ),
                "verifier_prompt_hash": identity.get("verifier_prompt_hash"),
                "verifier_schema_hash": identity.get("verifier_schema_hash"),
                "verifier_protocol_hash": identity.get("verifier_protocol_hash"),
            }
        )
    if any(value in {None, ""} for value in required.values()) or not model_commits:
        raise ValueError("pilot source exports lack the complete Questioner/model identity")
    return identity, required, model_commits


def decide_pilot(
    *,
    pilot_dir: Path,
    review_dir: Path | None,
    decision: PilotDecisionValue,
    reviewer_id: str,
    rationale: str,
) -> tuple[Path, Path | None]:
    pilot = verify_pilot_bundle(pilot_dir)
    audit_mode = pilot.protocol_version == "questioner-quality-pilot-v3"
    review: dict[str, Any] | None = None
    audit_hash: str | None = None
    review_hash: str | None = None
    review_id: str | None = None
    if audit_mode:
        if review_dir is not None:
            raise ValueError("Questioner v3 approval uses its signed audit, not a review dir")
        from pmmc.schemas import QuestionerPilotAuditManifest

        audit = QuestionerPilotAuditManifest.model_validate(
            read_json(pilot_dir / "private" / "audit_manifest.json")
        )
        if audit.pilot_id != pilot.pilot_id or audit.partition_id != pilot.partition_id:
            raise ValueError("Questioner audit does not belong to the frozen pilot bundle")
        if decision is PilotDecisionValue.APPROVE and (
            not audit.automatic_integrity_passed
            or audit.severe_terminal_error_count != 0
        ):
            raise ValueError("Questioner v3 automatic integrity gate blocks approval")
        audit_hash = audit.content_hash
        decision_root = pilot_dir.parent / f"{pilot.pilot_id}-decisions"
    else:
        if review_dir is None:
            raise ValueError("legacy Questioner pilots require a finalized review dir")
        review = verify_review(review_dir)
        if (
            review["pilot_id"] != pilot.pilot_id
            or review["pilot_content_hash"] != pilot.content_hash
        ):
            raise ValueError("review does not belong to the frozen pilot bundle")
        review_hash = str(review["content_hash"])
        review_id = str(review["review_id"])
        decision_root = review_dir.parent / f"{review_id}-decisions"
    existing_directories = (
        sorted(path for path in decision_root.iterdir() if path.is_dir())
        if decision_root.is_dir()
        else []
    )
    if existing_directories:
        if len(existing_directories) != 1:
            raise ValueError("pilot review has multiple conflicting decisions")
        existing_dir = existing_directories[0]
        existing = QuestionerPilotDecision.model_validate(
            read_json(existing_dir / "decision.json")
        )
        if (
            existing.pilot_id != pilot.pilot_id
            or existing.review_manifest_hash != review_hash
            or existing.audit_manifest_hash != audit_hash
            or existing.decision is not decision
            or existing.reviewer_id != reviewer_id
            or existing.rationale != rationale
        ):
            raise ValueError("pilot review already has an immutable decision")
        freeze_path = existing_dir / "questioner_freeze_manifest.json"
        state = QuestionerPilotState.model_validate(
            read_json(existing_dir / "pilot_state.json")
        )
        if state.decision_hash != existing.content_hash:
            raise ValueError("pilot decision state does not match the decision")
        if existing.decision is PilotDecisionValue.APPROVE:
            freeze = verify_approval(freeze_path)
            if state.status is not QuestionerPilotStatus.APPROVED or (
                state.freeze_id != freeze.freeze_id
            ):
                raise ValueError("approved pilot state does not match its freeze")
            return existing_dir, freeze_path
        if state.status is not QuestionerPilotStatus.REJECTED:
            raise ValueError("rejected pilot state does not match its decision")
        if freeze_path.exists():
            raise ValueError("rejected pilot unexpectedly contains a freeze manifest")
        return existing_dir, None

    freeze_identity: dict[str, Any] | None = None
    freeze_required: dict[str, object] | None = None
    freeze_model_commits: dict[str, str] | None = None
    if decision is PilotDecisionValue.APPROVE:
        freeze_identity, freeze_required, freeze_model_commits = (
            _approval_source_identity(pilot_dir, audit_mode=audit_mode)
        )

    decided_at = utc_now()
    decision_payload = {
        "pilot_id": pilot.pilot_id,
        "decision": decision.value,
        "reviewer_id": reviewer_id,
        "rationale": rationale,
        **(
            {"audit_manifest_hash": audit_hash}
            if audit_mode
            else {"review_manifest_hash": review_hash}
        ),
        "decided_at": decided_at,
    }
    decision_record = QuestionerPilotDecision(
        **decision_payload,
        content_hash=stable_hash(
            {
                **decision_payload,
                "decided_at": decided_at.isoformat().replace("+00:00", "Z"),
            }
        ),
    )
    decision_dir = decision_root / decision_record.content_hash
    if decision is PilotDecisionValue.REJECT:
        decision_dir.mkdir(parents=True)
        write_json(
            decision_dir / "decision.json", decision_record.model_dump(mode="json")
        )
        write_json(
            decision_dir / "pilot_state.json",
            QuestionerPilotState(
                pilot_id=pilot.pilot_id,
                status=QuestionerPilotStatus.REJECTED,
                updated_at=decided_at,
                review_id=review_id,
                decision_hash=decision_record.content_hash,
            ).model_dump(mode="json"),
        )
        return decision_dir, None

    assert freeze_identity is not None
    assert freeze_required is not None
    assert freeze_model_commits is not None
    freeze_payload = {
        "pilot_id": pilot.pilot_id,
        "decision_hash": decision_record.content_hash,
        "partition_id": pilot.partition_id,
        "pilot_sample_ids": [item.sample_id for item in pilot.samples],
        **freeze_required,
        "model_commits": freeze_model_commits,
        "runtime_export_ids": pilot.source_runtime_export_ids,
        "coverage_export_ids": pilot.source_coverage_export_ids,
        **(
            {
                "audit_manifest_hash": audit_hash,
                "questioner_protocol_version": freeze_identity.get(
                    "questioner_protocol_version"
                ),
                "prompt_version": freeze_identity.get("prompt_version"),
                "opportunity_builder_version": freeze_identity.get(
                    "opportunity_builder_version"
                ),
                "relation_selector_version": freeze_identity.get(
                    "relation_selector_version"
                ),
                "opportunity_builder_hash": freeze_identity.get(
                    "opportunity_builder_hash"
                ),
                "verifier_prompt_version": freeze_identity.get(
                    "verifier_prompt_version"
                ),
                "verifier_prompt_hash": freeze_identity.get("verifier_prompt_hash"),
                "verifier_schema_hash": freeze_identity.get("verifier_schema_hash"),
                "verifier_protocol_hash": freeze_identity.get(
                    "verifier_protocol_hash"
                ),
            }
            if audit_mode
            else {
                "review_hash": review_hash,
                "agreement_report_hash": (
                    review["agreement_report_hash"] if review is not None else None
                ),
            }
        ),
        "approved_by": reviewer_id,
        "approved_at": decided_at,
    }
    freeze_hash_payload = {
        **freeze_payload,
        "approved_at": decided_at.isoformat().replace("+00:00", "Z"),
    }
    content_hash = stable_hash(freeze_hash_payload)
    freeze = QuestionerFreezeManifest(
        freeze_id="questioner-freeze-" + content_hash[:24],
        **freeze_payload,
        content_hash=content_hash,
    )
    decision_dir.mkdir(parents=True)
    write_json(decision_dir / "decision.json", decision_record.model_dump(mode="json"))
    freeze_path = decision_dir / "questioner_freeze_manifest.json"
    write_json(freeze_path, freeze.model_dump(mode="json"))
    write_json(
        decision_dir / "pilot_state.json",
        QuestionerPilotState(
            pilot_id=pilot.pilot_id,
            status=QuestionerPilotStatus.APPROVED,
            updated_at=decided_at,
            review_id=review_id,
            decision_hash=decision_record.content_hash,
            freeze_id=freeze.freeze_id,
        ).model_dump(mode="json"),
    )
    return decision_dir, freeze_path


def verify_approval(path: Path) -> QuestionerFreezeManifest:
    freeze = QuestionerFreezeManifest.model_validate(read_json(path))
    payload = freeze.model_dump(mode="json", exclude={"freeze_id", "content_hash"})
    compatible_payload = freeze.model_dump(
        mode="json",
        exclude={"freeze_id", "content_hash"},
        exclude_none=True,
    )
    if all(
        stable_hash(value) != freeze.content_hash
        for value in (payload, compatible_payload)
    ):
        raise ValueError("Questioner freeze content hash mismatch")
    expected_id = "questioner-freeze-" + freeze.content_hash[:24]
    if freeze.freeze_id != expected_id:
        raise ValueError("Questioner freeze ID does not match its content")
    return freeze
