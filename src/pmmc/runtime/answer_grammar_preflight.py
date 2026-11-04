"""CPU-only checks of the installed guided decoder, without model weights."""

from __future__ import annotations

import json
from importlib.metadata import version
from typing import Any

from pmmc.compiler.ids import canonical_content_hash
from pmmc.models.gateways import project_vllm_response_format
from pmmc.schemas.answer_output import ANSWER_GUIDED_SCHEMA_VERSION, answer_response_schema


class AnswerGrammarPreflightError(ValueError):
    def __init__(self, code: str, ordinal: int) -> None:
        self.code = code
        self.ordinal = ordinal
        super().__init__(f"{code}_{ordinal}")


def answer_grammar_cases(alias: str) -> tuple[list[str], list[str]]:
    """Synthetic cases only; never read queries, gold, or provider artifacts."""
    supported: dict[str, object] = {
        "answer": "Blue.", "supported": True,
        "used_evidence_aliases": [alias], "abstain_reason": None,
    }
    refused: dict[str, object] = {
        "answer": None, "supported": False, "used_evidence_aliases": [],
        "abstain_reason": "insufficient_evidence",
    }
    valid = [json.dumps({**supported, "answer": value}, ensure_ascii=False) for value in (
        "Blue.", 'A "quoted" value', "Two\nlines", "A\\B", "caf\u00e9 \u56fe",
    )]
    valid.append(json.dumps(refused))
    supported_changes: tuple[dict[str, object], ...] = (
        {"answer": None}, {"used_evidence_aliases": []},
        {"used_evidence_aliases": ["UNKNOWN"]}, {"abstain_reason": "unsafe_request"},
    )
    invalid = [json.dumps({**supported, **change}) for change in supported_changes]
    refusal_changes: tuple[dict[str, object], ...] = (
        {"answer": "Blue."}, {"abstain_reason": None},
    )
    invalid.extend(json.dumps({**refused, **change}) for change in refusal_changes)
    for value in ('raw"quote', "raw\nnewline", "raw\ttab", "raw\x01control", r"bad\qescape"):
        invalid.append(json.dumps(supported).replace('"Blue."', '"' + value + '"'))
    invalid.extend([json.dumps(supported) + "JUNK", json.dumps(supported)[:-1]])
    return valid, invalid


def verify_compiled_answer_grammar(engine: Any, compiled: Any, alias: str) -> dict[str, int]:
    valid, invalid = answer_grammar_cases(alias)
    for ordinal, text in enumerate(valid):
        matcher = engine.GrammarMatcher(compiled)
        if not matcher.accept_string(text) or not matcher.accept_token(256):
            raise AnswerGrammarPreflightError("valid_answer_not_terminable", ordinal)
        if not matcher.is_terminated():
            raise AnswerGrammarPreflightError("answer_eos_not_terminal", ordinal)
        # Before EOS, a complete answer must not retain an alternative parse
        # that swallows the object terminator as part of its answer string.
        matcher = engine.GrammarMatcher(compiled)
        if not matcher.accept_string(text) or matcher.accept_string("INVALID_TRAILING_TOKEN"):
            raise AnswerGrammarPreflightError("answer_string_consumes_object_boundary", ordinal)
    for ordinal, text in enumerate(invalid):
        matcher = engine.GrammarMatcher(compiled)
        if matcher.accept_string(text) and matcher.accept_token(256):
            raise AnswerGrammarPreflightError("invalid_answer_grammar_accepted", ordinal)
    return {"valid_cases": len(valid), "invalid_cases": len(invalid)}


def verify_answer_grammar() -> dict[str, object]:
    import xgrammar as xgr

    vocabulary = [bytes([value]) for value in range(256)] + [b"<eos>"]
    compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo(vocabulary, stop_token_ids=[256]), max_threads=1,
    )
    schemas = []
    valid_count = invalid_count = 0
    for alias in ("E01", "A01"):
        transport, _ = project_vllm_response_format({
            "type": "json_schema", "json_schema": {"schema": answer_response_schema([alias])},
        })
        schema = transport["json_schema"]["schema"]
        counts = verify_compiled_answer_grammar(xgr, compiler.compile_json_schema(schema), alias)
        valid_count += counts["valid_cases"]
        invalid_count += counts["invalid_cases"]
        schemas.append(schema)
    return {
        "state": "ready", "engine": "xgrammar", "engine_version": version("xgrammar"),
        "protocol": ANSWER_GUIDED_SCHEMA_VERSION, "schema_hash": canonical_content_hash(schemas),
        "valid_cases": valid_count, "invalid_cases": invalid_count,
        "model_calls": 0, "gpu_required": False,
    }


def main() -> int:
    try:
        report = verify_answer_grammar()
    except Exception as exc:
        # Engine exceptions can contain a complete schema: do not echo them.
        report = {
            "state": "failed", "error_code": "answer_grammar_preflight_failed",
            "error_type": type(exc).__name__, "model_calls": 0,
            **({"case_failure": exc.code, "case_ordinal": exc.ordinal}
               if isinstance(exc, AnswerGrammarPreflightError) else {}),
        }
    print("ANSWER_GRAMMAR_PREFLIGHT " + json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["state"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
