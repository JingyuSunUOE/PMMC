"""CPU-only host workload for the selected public benchmark sources."""

import json
import os
import sys
from pathlib import Path

from pmmc.benchmarks.release_data import prepare_data
from pmmc.release_plan import plan_from_environment, plan_hash


def main() -> None:
    plan = plan_from_environment(os.environ)
    if plan is None:
        raise ValueError("release assets require PMMC_RELEASE_REQUEST")
    root = Path(os.environ.get("PMMC_BENCHMARK_STAGING_ROOT", "/pmmc-stage"))
    try:
        if (
            not root.is_absolute()
            or not root.is_dir()
            or root.resolve()
            == Path(os.environ.get("PMMC_BENCHMARK_ASSET_ROOT", "/pmmc-inputs")).resolve()
        ):
            raise ValueError("a distinct writable benchmark staging mount is required")
        sources = prepare_data(root, plan["benchmarks"])
    except (KeyError, OSError, ValueError) as exc:
        # Never render transport exceptions (which may include signed redirect URLs).
        print(
            json.dumps(dict(protocol="release-assets-v2", ready=False, error=type(exc).__name__)),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    print(
        json.dumps(
            dict(
                protocol="release-assets-v2", ready=True, plan_hash=plan_hash(plan), sources=sources
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
