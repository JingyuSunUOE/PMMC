"""Unified PMMC benchmark ingestion and canonical data access."""

from pmmc.benchmarks.base import BenchmarkAdapter, BenchmarkDataError
from pmmc.benchmarks.canonical import (
    CanonicalPreflightReport,
    preflight_snapshot,
    recompute_canonical_content_hash,
)
from pmmc.benchmarks.models import AdaptedQuery, CanonicalConversation
from pmmc.benchmarks.registry import create_adapter
from pmmc.benchmarks.repositories import GoldStore, MemoryCorpus, RuntimeQueryStore
from pmmc.benchmarks.selection import select_queries, selection_hash
from pmmc.benchmarks.snapshot import build_snapshot, find_snapshot, load_manifest
from pmmc.benchmarks.validation import validate_adapter

__all__ = [
    "AdaptedQuery",
    "BenchmarkAdapter",
    "BenchmarkDataError",
    "CanonicalPreflightReport",
    "CanonicalConversation",
    "GoldStore",
    "MemoryCorpus",
    "RuntimeQueryStore",
    "build_snapshot",
    "create_adapter",
    "find_snapshot",
    "load_manifest",
    "preflight_snapshot",
    "recompute_canonical_content_hash",
    "select_queries",
    "selection_hash",
    "validate_adapter",
]
