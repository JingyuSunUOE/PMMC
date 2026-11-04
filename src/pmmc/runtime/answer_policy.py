"""Immutable prompt/retry pairs; the output acceptance contract remains strict v2."""

from __future__ import annotations

import hashlib
from pathlib import Path

LEGACY_PROMPT_VERSION = "pmmc-answer-contract-v2"
CURRENT_PROMPT_VERSION = "pmmc-answer-contract-v2.1"
LEGACY_PROMPT_HASH = "73a4f274fc4fc00b08f2f41f57ff4ba43d2c7f8f0dadc489842710345065d142"
CURRENT_PROMPT_HASH = "5cc9fdee744b0c39c66f56c27df1bdd44d12a2c99e1ada74a7d298fa8af52225"
# PMMC hashes its JSON-serialized instruction; Native-MM hashes the UTF-8 text.
CURRENT_RUNTIME_PROMPT_HASH = "367e74c8a4e1e1c6113b3ebd7999b14fc4cabc6dde2a385aeaf05016eeab2336"
PROMPTS = {
    LEGACY_PROMPT_VERSION: ("prompts/runtime/pmmc_answer_contract_v2.txt", LEGACY_PROMPT_HASH),
    CURRENT_PROMPT_VERSION: ("prompts/runtime/pmmc_answer_contract_v2_1.txt", CURRENT_PROMPT_HASH),
}
CURRENT_RETRY_POLICY = "answer-format-feedback-v2"
CURRENT_RETRY_SHA256 = "02d64037b37738300311a2df2e2d957e73103058b59e945a1edbfde57d0a2a38"
CURRENT_RETRY_INSTRUCTION = (
    "Return only the corrected four-field JSON object. The answer field must contain "
    "only the public answer, never evidence labels, bracketed citations such as [E01] "
    "or [A01], or bare labels such as E01 or A01. Put supporting aliases ONLY in "
    "used_evidence_aliases, each supplied alias at most once. Do not substitute an "
    "invented name for an evidence label. If supported is true, answer must be "
    "nonempty text, used_evidence_aliases must be nonempty, and abstain_reason must "
    "be null. If supported is false, answer must be null, used_evidence_aliases must "
    "be empty, and abstain_reason must be an allowed enum value. Never invent support "
    "or change the evidence boundary to satisfy the format."
)


def load_answer_policy_prompt(project_root: Path, version: str) -> str:
    if version not in PROMPTS:
        raise ValueError("unsupported strict answer prompt version")
    relative, expected = PROMPTS[version]
    prompt = (project_root / relative).read_text(encoding="utf-8").strip()
    if hashlib.sha256(prompt.encode()).hexdigest() != expected:
        raise ValueError("strict answer prompt source hash mismatch")
    return prompt


def current_retry_instruction() -> str:
    if hashlib.sha256(CURRENT_RETRY_INSTRUCTION.encode()).hexdigest() != CURRENT_RETRY_SHA256:
        raise ValueError("strict answer retry source hash mismatch")
    return CURRENT_RETRY_INSTRUCTION
