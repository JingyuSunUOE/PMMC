"""Conservative CAS reachability collection, separate from object storage and weights."""

from __future__ import annotations

import json
import re
from typing import Any

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.experiments.step8_jobs import ComponentReceipt

_HASH = re.compile(r"[a-f0-9]{64}\Z")


def _hashes(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if _HASH.fullmatch(value) else set()
    if isinstance(value, dict):
        return set().union(*(_hashes(item) | _hashes(key) for key, item in value.items()))
    if isinstance(value, list):
        return set().union(*(_hashes(item) for item in value))
    return set()


def collect_campaign_cas(store: CampaignArtifactStore, *, apply: bool = False) -> dict[str, Any]:
    """Keep all campaign/checkpoint/journal roots and their transitive CAS references.

    Missing mandatory component objects or malformed roots abort collection. This
    deliberately over-retains incidental hash references. S3 and model caches are
    outside this collector; it never claims to collect those stores.
    """
    with store.collection_lock(exclusive=True):
        retained: set[str] = set()
        root_files: dict[str, str] = {}
        for directory in sorted(store.root.iterdir()):
            if directory.name == "_cache" or directory.name.startswith(".gc-"):
                continue
            if directory.is_symlink():
                raise ValueError("gc_root_symlink_forbidden")
            if not directory.is_dir():
                continue
            if directory.name not in {"_store", "integration"} and not (
                directory / "manifests/campaign.json"
            ).is_file():
                raise ValueError("gc_unrecognized_campaign_directory")
            for path in sorted(directory.rglob("*")):
                if path.is_symlink():
                    raise ValueError("gc_reference_symlink_forbidden")
                if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
                    continue
                if path.is_relative_to(store.store):
                    continue
                payload = path.read_bytes()
                values = (
                    [json.loads(line) for line in payload.splitlines() if line.strip()]
                    if path.suffix == ".jsonl"
                    else [json.loads(payload)]
                )
                for value in values:
                    if (
                        isinstance(value, dict)
                        and value.get("protocol") == "step8-component-receipt-v1"
                    ):
                        receipt = ComponentReceipt.model_validate(value)
                        for digest in receipt.artifact_hashes:
                            store.verify(digest)
                    retained.update(d for d in _hashes(value) if store.object_path(d).is_file())
                root_files[str(path.relative_to(store.root))] = canonical_content_hash(values)
        pending = list(retained)
        while pending:
            digest = pending.pop()
            path = store.verify(digest)
            try:
                nested = json.loads(path.read_bytes())
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            found = {d for d in _hashes(nested) if store.object_path(d).is_file()} - retained
            retained.update(found)
            pending.extend(found)
        report = store.gc(referenced_hashes=retained, reference_scan_complete=True, apply=apply)
        report.update(
            protocol="step8-cas-mark-sweep-v1",
            root_count=len(root_files),
            root_set_hash=canonical_content_hash(root_files),
            model_weights_deleted=False,
            object_storage_deleted=False,
        )
        write_json(store.root / ".gc-last-receipt.json", report)
        return report
