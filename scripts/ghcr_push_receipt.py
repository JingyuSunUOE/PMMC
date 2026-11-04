"""Resolve a completed candidate push by its receipt and the exact remote manifest bytes."""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def pushed_ref(repo, tag, log, revision, source_url):
    if not tag.startswith(f"sha-{revision}-") or not repo.startswith("ghcr.io/"):
        raise ValueError("not a source-qualified GHCR candidate")
    receipts = re.findall(
        rf"^{re.escape(tag)}: digest: (sha256:[a-f0-9]{{64}}) size: ([1-9][0-9]*)\s*$", log, re.M
    )
    if len(receipts) != 1:
        raise ValueError("expected one docker push receipt for the candidate tag")
    digest, size = receipts[0]
    ref = f"{repo}@{digest}"
    inspect = ["docker", "buildx", "imagetools", "inspect"]
    for image in (f"{repo}:{tag}", ref):
        data = subprocess.check_output([*inspect, "--raw", image], timeout=120)
        if len(data) != int(size) or "sha256:" + hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("GHCR manifest does not match docker push receipt")
    cfg = json.loads(
        subprocess.check_output([*inspect, "--format", "{{json .Image}}", ref], timeout=120)
    )
    cfg = cfg.get("linux/amd64", cfg)
    labels = cfg.get("config", {}).get("Labels", {})
    if (
        (cfg.get("os"), cfg.get("architecture")) != ("linux", "amd64")
        or labels.get("org.opencontainers.image.revision") != revision
        or labels.get("org.opencontainers.image.source") != source_url
    ):
        raise ValueError("GHCR image platform or source identity mismatch")
    return ref


if __name__ == "__main__":
    repo, tag, receipt = sys.argv[1:]
    print(
        pushed_ref(
            repo,
            tag,
            Path(receipt).read_text(),
            os.environ["GITHUB_SHA"],
            os.environ["GITHUB_SERVER_URL"] + "/" + os.environ["GITHUB_REPOSITORY"],
        )
    )
