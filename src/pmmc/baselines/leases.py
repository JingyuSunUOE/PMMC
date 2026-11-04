"""Execution-only owner identities; never used in scientific request identities."""

from uuid import uuid4

from pmmc.benchmarks.ids import stable_hash


def baseline_worker_id(run_id: str, prefix: str, index: int) -> str:
    # A repeated execute() in the same process must not reuse a former lease owner.
    return f"{prefix[:128]}-{stable_hash(run_id)[:16]}-{uuid4().hex}-{index}"
