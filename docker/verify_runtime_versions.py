"""Validate pinned runtime package versions with PEP 440 semantics."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from importlib import metadata

from packaging.requirements import InvalidRequirement, Requirement


class VersionCheck:
    __slots__ = ("name", "actual", "expected")

    def __init__(self, name: str, actual: str, expected: str) -> None:
        self.name = name
        self.actual = actual
        self.expected = expected


def validate_runtime_versions(
    requirements: Sequence[str],
    *,
    version_lookup: Callable[[str], str] = metadata.version,
) -> tuple[list[VersionCheck], list[str]]:
    """Return successful checks and actionable validation errors."""

    checks: list[VersionCheck] = []
    errors: list[str] = []
    for requirement_text in requirements:
        try:
            requirement = Requirement(requirement_text)
        except InvalidRequirement as exc:
            errors.append(f"invalid requirement {requirement_text!r}: {exc}")
            continue
        if requirement.url or requirement.extras or requirement.marker or not requirement.specifier:
            errors.append(
                f"runtime check requires one plain package with a version constraint: "
                f"{requirement_text!r}"
            )
            continue
        try:
            actual = version_lookup(requirement.name)
        except metadata.PackageNotFoundError:
            errors.append(f"{requirement.name}: package is not installed")
            continue

        expected = str(requirement.specifier)
        if not requirement.specifier.contains(actual, prereleases=True):
            errors.append(f"{requirement.name}: installed {actual}, expected {expected}")
            continue
        checks.append(VersionCheck(requirement.name, actual, expected))
    return checks, errors


def main(arguments: list[str]) -> int:
    if not arguments:
        print(
            "usage: verify_runtime_versions.py <package-version-requirement> [...]",
            file=sys.stderr,
        )
        return 2

    checks, errors = validate_runtime_versions(arguments)
    for check in checks:
        print(f"validated runtime package: {check.name} {check.actual} satisfies {check.expected}")
    if errors:
        print("runtime package version mismatches:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
