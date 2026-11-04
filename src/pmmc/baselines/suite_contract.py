"""Versioned answer identity for ordinary suites and internal Native-MM children."""

from __future__ import annotations

from dataclasses import dataclass

from pmmc.benchmarks.ids import stable_hash
from pmmc.runtime.answer_policy import LEGACY_PROMPT_VERSION, PROMPTS
from pmmc.runtime.grounding import GROUNDING_MODEL_ALIAS, load_step8_prompt
from pmmc.schemas import BaselineSuiteManifest, RunManifest
from pmmc.schemas.answer_output import GROUNDING_POLICY_VERSION, TYPED_ANSWER_CONTRACT_VERSION
from pmmc.schemas.base import JsonDict
from pmmc.schemas.baseline_suite import is_internal_pmmc_fallback_suite

STRICT_SUITE_CONTRACT = "native-mm-strict-suite-v1"
STRICT_ANSWER_CONTRACT = "strict-answer-contract-v2"
STRICT_PROMPT_VERSION = "pmmc-answer-contract-v2"
STRICT_PROMPT_HASH = "73a4f274fc4fc00b08f2f41f57ff4ba43d2c7f8f0dadc489842710345065d142"
LEGACY_PROMPT_HASH = "9052aa190250705454415dd3dc8d07018352b8c611947d4232287611681203e5"
TYPED_SUITE_CONTRACT = "five-baseline-typed-suite-v3"


class BaselineSuiteContractError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.diagnostics = {"failure_scope": "suite_answer_contract", "contract_error_code": code}
        super().__init__(code)


@dataclass(frozen=True)
class SuiteAnswerContract:
    prompt_hash: str
    decoding: JsonDict
    mode: str


def suite_answer_contract(suite: BaselineSuiteManifest) -> SuiteAnswerContract:
    if suite.metadata.get("answer_contract_version") == TYPED_ANSWER_CONTRACT_VERSION:
        from pmmc.baselines.config import prompt_sha256

        expected = prompt_sha256(load_step8_prompt("pmmc_answer_contract_v3.txt"))
        decoding = {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "max_output_tokens": 512,
            "thinking_mode": "disabled",
            "format_retries": 1,
        }
        if (
            suite.metadata.get("answer_suite_contract_version") != TYPED_SUITE_CONTRACT
            or suite.metadata.get("answer_prompt_version") != "pmmc-answer-contract-v3"
            or suite.metadata.get("grounding_policy_version") != GROUNDING_POLICY_VERSION
            or suite.metadata.get("grounding_model_alias") != GROUNDING_MODEL_ALIAS
            or suite.metadata.get("grounding_prompt_hash")
            != prompt_sha256(load_step8_prompt("answer_grounding_v1.txt"))
            or suite.answer_prompt_hash != expected
            or suite.decoding_config != decoding
            or suite.reserved_output_tokens != 512
        ):
            raise BaselineSuiteContractError("typed_suite_contract_mismatch")
        return SuiteAnswerContract(expected, decoding, "native_typed_manifest")
    if suite.metadata.get("answer_contract_version") != STRICT_ANSWER_CONTRACT:
        return SuiteAnswerContract(
            suite.answer_prompt_hash,
            dict(suite.decoding_config),
            "manifest",
        )
    if (
        not is_internal_pmmc_fallback_suite(suite.methods, suite.metadata)
        or suite.metadata.get("answer_prompt_version") not in PROMPTS
        or suite.metadata.get("answer_output_schema_version") != "0.1.0"
    ):
        raise BaselineSuiteContractError("strict_suite_scope_or_version_invalid")
    version = str(suite.metadata["answer_prompt_version"])
    expected_hash = PROMPTS[version][1]
    if suite.metadata.get("answer_suite_contract_version") == STRICT_SUITE_CONTRACT:
        if (
            suite.answer_prompt_hash != expected_hash
            or suite.decoding_config.get("max_output_tokens") != 512
            or suite.reserved_output_tokens != 512
        ):
            raise BaselineSuiteContractError("strict_suite_header_mismatch")
        mode = "native_strict_manifest"
    elif (
        "answer_suite_contract_version" not in suite.metadata
        and version == LEGACY_PROMPT_VERSION
        and suite.answer_prompt_hash == LEGACY_PROMPT_HASH
        and suite.reserved_output_tokens in {1024, 4096}
        and suite.decoding_config.get("max_output_tokens") == suite.reserved_output_tokens
    ):
        # The old header was created from full-context defaults although its
        # immutable metadata already selected strict v2. Preserve that header;
        # persist this explicit binding when linking its strict child, not a rewrite.
        mode = "legacy_step7b_metadata_binding"
    else:
        raise BaselineSuiteContractError("unrecognized_strict_suite_header")
    decoding = {**suite.decoding_config, "max_output_tokens": 512}
    return SuiteAnswerContract(expected_hash, decoding, mode)


def validate_suite_run_answer_contract(
    suite: BaselineSuiteManifest,
    run: RunManifest,
) -> JsonDict:
    contract = suite_answer_contract(suite)
    typed = contract.mode == "native_typed_manifest"
    strict = contract.mode not in {"manifest", "native_typed_manifest"}
    for matches, code in (
        (run.method in {method.value for method in suite.methods}, "method_mismatch"),
        (run.dataset == suite.dataset, "dataset_mismatch"),
        (run.dataset_snapshot_ids == [suite.snapshot_id], "snapshot_mismatch"),
        (run.selection_hash == suite.selection_hash, "selection_mismatch"),
        (run.answer_model_alias == suite.answer_model_alias, "answer_model_mismatch"),
        (run.decoding_configs.get("answer") == contract.decoding, "answer_decoding_mismatch"),
    ):
        if not matches:
            raise BaselineSuiteContractError(code)
    if typed:
        if (
            run.metadata.get("answer_contract_version") != TYPED_ANSWER_CONTRACT_VERSION
            or run.prompt_registry_version != "pmmc-answer-contract-v3"
            or run.prompt_hashes.get("pmmc-answer-contract-v3") != contract.prompt_hash
            or any(
                run.metadata.get(key) != suite.metadata.get(key)
                for key in (
                    "grounding_policy_version",
                    "grounding_model_alias",
                    "grounding_prompt_hash",
                    "memory_raw_image_policy",
                    "allow_memory_raw_images",
                    "method_protocol_hash",
                    "internal_pmmc_fallback",
                )
            )
            or run.prompt_hashes.get("answer-grounding-v1")
            != suite.metadata["grounding_prompt_hash"]
        ):
            raise BaselineSuiteContractError("typed_child_answer_identity_mismatch")
    elif strict:
        version = str(suite.metadata["answer_prompt_version"])
        if (
            run.metadata.get("answer_contract_version") != STRICT_ANSWER_CONTRACT
            or run.metadata.get("answer_prompt_version") != version
            or run.metadata.get("answer_output_schema_version") != "0.1.0"
            or run.prompt_registry_version != version
            or run.prompt_hashes.get(version) != contract.prompt_hash
        ):
            raise BaselineSuiteContractError("strict_child_answer_identity_mismatch")
    elif contract.prompt_hash not in run.prompt_hashes.values():
        raise BaselineSuiteContractError("answer_prompt_mismatch")
    binding: JsonDict = {
        "policy": "baseline-suite-answer-binding-v1",
        "mode": contract.mode,
        "suite_config_hash": suite.config_hash,
        "recorded_suite_prompt_hash": suite.answer_prompt_hash,
        "effective_prompt_hash": contract.prompt_hash,
        "effective_decoding_config": contract.decoding,
    }
    return {**binding, "content_hash": stable_hash(binding)}
