"""Validate pip check output for the PMMC vLLM-based container image."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_HEALTHY_MESSAGE = "No broken requirements found."
_ALLOWED_BASE_IMAGE_ISSUES = (
    re.compile(
        r"pygobject \S+ requires pycairo, which is not installed\.",
        flags=re.IGNORECASE,
    ),
)


def validate_pip_check_output(output: str) -> tuple[list[str], list[str]]:
    """Return allowed inherited issues and unexpected dependency failures."""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines == [_HEALTHY_MESSAGE]:
        return [], []

    allowed: list[str] = []
    unexpected: list[str] = []
    for line in lines:
        if any(pattern.fullmatch(line) for pattern in _ALLOWED_BASE_IMAGE_ISSUES):
            allowed.append(line)
        else:
            unexpected.append(line)
    if not lines:
        unexpected.append("pip check produced no output")
    return allowed, unexpected


def main(arguments: list[str]) -> int:
    if len(arguments) != 1:
        print("usage: verify_pip_check.py <pip-check-output>", file=sys.stderr)
        return 2

    report_path = Path(arguments[0])
    allowed, unexpected = validate_pip_check_output(
        report_path.read_text(encoding="utf-8")
    )
    for issue in allowed:
        print(f"allowed vLLM base-image issue: {issue}", file=sys.stderr)
    if unexpected:
        print("unexpected dependency issues:", file=sys.stderr)
        for issue in unexpected:
            print(f"  {issue}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
