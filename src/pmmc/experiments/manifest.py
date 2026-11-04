"""Deterministic experiment-manifest fingerprints."""

from __future__ import annotations

from pmmc.benchmarks.ids import stable_hash
from pmmc.schemas import RunManifest


def run_config_hash(manifest: RunManifest) -> str:
    """Hash the immutable run configuration without its self-referential hash field."""

    return stable_hash(manifest.model_dump(mode="json", exclude={"config_hash"}))
