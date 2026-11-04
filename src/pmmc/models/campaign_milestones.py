"""Durable compiler round barriers; no provider text enters milestone logs."""

from __future__ import annotations

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import write_json
from pmmc.models.campaign_audit import current_campaign_audit


def record_doubter_round(*, run_id: str, round_id: int, assessment_hashes: dict[str, str]) -> None:
    audit = current_campaign_audit()
    if audit is None:
        return
    if round_id not in range(4):
        raise ValueError("invalid_doubter_round_milestone")
    value = {
        "protocol": "step8-doubter-round-barrier-v1",
        "campaign_id": audit.campaign_id,
        "job_id": audit.job_id,
        "source_fingerprint": audit.source_fingerprint,
        "run_id": run_id,
        "round_id": round_id,
        "assessment_hashes": assessment_hashes,
    }
    digest = audit.store.put_json(value)
    key = canonical_content_hash([audit.campaign_id, audit.job_id, run_id, round_id])
    write_json(
        audit.store.root / "_store/round-barriers" / f"{key}.json",
        {
            "barrier_sha256": digest,
            "campaign_id": audit.campaign_id,
            "job_id": audit.job_id,
            "round_id": round_id,
        },
        immutable=True,
    )
