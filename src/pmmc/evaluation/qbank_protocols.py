"""Dependency-free protocol identities shared by QBank evaluation stages."""

from __future__ import annotations

from types import MappingProxyType

QBANK_EVIDENCE_RESOLUTION_V1 = "evidence-resolution-v1"
QBANK_EVIDENCE_RESOLUTION_V2 = "evidence-resolution-v2"

HARDENED_QBANK_COVERAGE_PROTOCOLS = frozenset(
    {
        "qbank-coverage-v3",
        "qbank-coverage-v4",
        "qbank-coverage-v5",
        "qbank-coverage-v6",
    }
)

QBANK_SEMANTIC_COVERAGE_PROTOCOLS = MappingProxyType(
    {
        "qbank-semantic-match-v1": "qbank-coverage-v3",
        "qbank-semantic-match-v2": "qbank-coverage-v4",
        "qbank-semantic-match-v3": "qbank-coverage-v5",
        "qbank-semantic-match-v4": "qbank-coverage-v6",
    }
)

__all__ = [
    "HARDENED_QBANK_COVERAGE_PROTOCOLS",
    "QBANK_EVIDENCE_RESOLUTION_V1",
    "QBANK_EVIDENCE_RESOLUTION_V2",
    "QBANK_SEMANTIC_COVERAGE_PROTOCOLS",
]
