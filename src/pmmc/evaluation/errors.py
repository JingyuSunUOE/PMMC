"""Safe evaluation error diagnostics that never serialize request payloads."""

from __future__ import annotations

from pmmc.infrastructure.database.errors import (
    safe_database_error_diagnostics,
    safe_database_error_summary,
)

_SAFE_VALUE_ERROR_CODES = {
    "paper mode images require SHA256, width, height, and runtime storage URI": (
        "paper_image_metadata_incomplete"
    ),
    "paper mode requires a private runtime-bucket s3:// image URI": (
        "paper_image_not_synchronized"
    ),
    "image storage URI points to the wrong runtime bucket": "image_bucket_mismatch",
    "judge verdict and score disagree": "judge_verdict_score_mismatch",
    "judge reason cannot be empty": "judge_reason_empty",
}


def safe_error_diagnostics(error: Exception) -> dict[str, str]:
    """Extract database identifiers without serializing SQL or bound values."""

    diagnostics = safe_database_error_diagnostics(error)
    if type(error) is ValueError:
        error_code = _SAFE_VALUE_ERROR_CODES.get(str(error))
        if error_code is not None:
            diagnostics["error_code"] = error_code
    return diagnostics


def safe_error_summary(diagnostics: dict[str, str]) -> str:
    """Render only the allow-listed diagnostics stored with a failed job."""

    return safe_database_error_summary(diagnostics)
