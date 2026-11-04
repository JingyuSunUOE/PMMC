"""Content-free answer diagnostics; this module does not decide acceptance."""

from __future__ import annotations

import re
from collections.abc import Sequence


def answer_alias_diagnostics(answer: str, allowed_aliases: Sequence[str]) -> dict[str, object]:
    """Explain the existing alias gate without exposing text or alias values."""
    bracket_matches = list(re.finditer(r"\[E\d{2,}\]", answer, flags=re.IGNORECASE))
    token_matches = [
        match
        for alias in set(allowed_aliases)
        for match in re.finditer(rf"\b{re.escape(alias)}\b", answer, flags=re.IGNORECASE)
    ]
    bracketed_tokens = sum(
        answer[max(0, match.start() - 1):match.start()] == "["
        and answer[match.end():match.end() + 1] == "]"
        for match in token_matches
    )
    spans = sorted({(item.start(), item.end()) for item in [*bracket_matches, *token_matches]})
    return {
        "alias_diagnostic_version": "answer-alias-diagnostics-v1",
        "answer_char_count": len(answer),
        "allowed_alias_count": len(set(allowed_aliases)),
        "bracketed_evidence_marker_count": len(bracket_matches),
        "allowed_alias_token_count": len(token_matches),
        "unbracketed_allowed_alias_count": len(token_matches) - bracketed_tokens,
        "alias_at_answer_start": bool(spans and not answer[:spans[0][0]].strip()),
        "alias_at_answer_end": bool(spans and not answer[spans[-1][1]:].strip()),
    }


def typed_answer_schema_diagnostics(error) -> dict[str, object]:
    """Keep diagnostics content-free even when extra fields contain private values."""
    from pmmc.schemas.answer_output import AnswerContractV3

    errors = error.errors(include_input=False, include_context=False, include_url=False)
    fields = set(AnswerContractV3.model_fields)
    return {
        "validation_error_count": len(errors),
        "validation_error_types": sorted({item["type"] for item in errors}),
        "validation_fields": sorted({
            str(item["loc"][0]) if item["loc"] and item["loc"][0] in fields
            else "object" if not item["loc"] else "extra_field" for item in errors
        }),
    }
