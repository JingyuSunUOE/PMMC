"""Synthetic operator configuration for Step 8 program validation tests."""

from pmmc.compiler.config import CompilerFoundationConfig


def config() -> CompilerFoundationConfig:
    return CompilerFoundationConfig(
        operator_set_version="pmmc-operators-v7", dry_run_policy="evidence_driven_v7"
    )
