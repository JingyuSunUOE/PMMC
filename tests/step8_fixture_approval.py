"""Persist synthetic approval inputs for downstream boundary tests, never live approval."""

from pathlib import Path

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.schemas.base import utc_now
from pmmc.schemas.step8 import MethodProtocolFreeze, ScientificProtocol, ThresholdApprovalEvidence


def saved_protocol_freeze(protocol: ScientificProtocol, root: Path) -> MethodProtocolFreeze:
    store = CampaignArtifactStore(root)
    policy = {
        name: getattr(protocol, name)
        for name in (
            "match_threshold",
            "gap_threshold",
            "no_reranker_match_threshold",
            "no_reranker_gap_threshold",
            "text_relevance_threshold",
            "multimodal_relevance_threshold",
        )
    }
    provisional = ScientificProtocol.model_validate(
        {
            **protocol.model_dump(),
            "match_threshold": 0.5,
        }
    )
    campaign = {
        "campaign_id": "offline-approval-fixture",
        "protocol": provisional.model_dump(mode="json"),
        "fixture_only": True,
        "scientific_assignments": 0,
    }
    write_json(root / "campaign.json", campaign)
    selections, proofs, receipts = {}, {}, {}
    for cohort in ("regression", "confirmation"):
        selection = {"campaign_id": campaign["campaign_id"], "selection": cohort}
        write_json(root / cohort / "selection.json", selection)
        selections[cohort] = canonical_content_hash(selection)
        proofs[cohort] = store.put_json(
            {
                "fixture_only": True,
                "selection_hash": selections[cohort],
                "provisional_protocol_hash": provisional.content_hash,
                "selected_policy": policy,
            }
        )
        receipt = {
            "campaign_id": campaign["campaign_id"],
            "provisional_protocol_hash": provisional.content_hash,
            "selected_protocol_hash": protocol.content_hash,
            "selected_policy": policy,
            "selection": selections[cohort],
            "threshold_proof_sha256": proofs[cohort],
        }
        write_json(root / cohort / "receipt.json", receipt)
        receipts[cohort] = canonical_content_hash(receipt)
    evidence = ThresholdApprovalEvidence(
        source_campaign_id=campaign["campaign_id"],
        source_campaign_manifest_hash=canonical_content_hash(campaign),
        provisional_protocol=provisional,
        selected_policy=policy,
        cohort_selection_hashes=selections,
        cohort_proof_sha256=proofs,
        cohort_receipt_hashes=receipts,
        audit_inventory_sha256=store.put_json(
            {
                "campaign_id": campaign["campaign_id"],
                "fixture_only": True,
                "scientific_assignments": 0,
                "cohort_proof_sha256": proofs,
            }
        ),
    )
    write_json(root / "threshold-approval.json", evidence.model_dump(mode="json"))
    return MethodProtocolFreeze(
        protocol=protocol,
        decision="approved",
        reviewer_id="offline-fixture-reviewer",
        rationale_sha256=store.put_json({"fixture_only": True, "evidence": evidence.content_hash}),
        development_receipt_hash=receipts["regression"],
        confirmation_receipt_hash=receipts["confirmation"],
        approved_at=utc_now(),
        protocol_integrity_passed=True,
        scientific_viability_passed=True,
        exposure_complete=True,
        threshold_approval=evidence,
    )
