"""Shared deterministic routing primitives for managed model replicas."""

from __future__ import annotations

import hashlib


def stable_replica_index(request_id: str, replica_count: int) -> int:
    """Map a request identity to one replica without process-local state."""

    if not request_id:
        raise ValueError("replica routing requires a request ID")
    if replica_count <= 0:
        raise ValueError("replica count must be positive")
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % replica_count
