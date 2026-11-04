"""Materialize the actual child selection IDs before quoting a shared plan."""

import json
import os
import sys

from pmmc.experiments.release_inputs import prepare_exploration
from pmmc.infrastructure.settings import get_settings
from pmmc.release_plan import plan_from_environment


def main():
    run, anchor = sys.argv[1:]
    plan = plan_from_environment(os.environ)
    if plan is None or run != plan["run_id"] or anchor not in plan["variants"]:
        raise ValueError("plan binding required")
    children = {}
    for variant in plan["variants"]:
        print(f"PMMC_PREPARE variant={variant}", flush=True)
        children[variant] = prepare_exploration(get_settings(), run, variant=variant)
    print(json.dumps(children[anchor], sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
