# PMMC v0.1.0 source distribution

This is a source distribution, not an experiment result, model bundle, container
image or approved scientific campaign. No publication, tag, execution budget or
historical exception is implied. The package metadata version is not a Git tag.
See LICENSE, NOTICE and THIRD_PARTY.md.

## What changed

- The `exploration-dev-all` suite operates only on the complete development
  partition of the two licensed datasets. Its denominators come from the
  actual assignments. It neither opens paper-test nor grants confirmation.
- W binds prospective questions to evidence available at write time; R adds a
  separately recorded native image-and-text relevance channel. The legacy raw
  cosine channel, thresholds, genuine image delivery and final grounding
  remain observable. Four independent arms support paired descriptive analysis.
- The controller exposes the same run/resume interface on EIDF and local
  Linux/NVIDIA. It records admission and progress, separates open execution
  from selection-specific paid approval, and emits a private diagnostic bundle
  on completion or failure. Start with the primary PMMC cells, then analyses
  and ablations, then the five independent baselines.

No end-to-end metric or speedup is claimed by releasing this software.
The README links the input layout, deployment prerequisites, output files
and recovery procedure. Existing private research data are not backups
of the new deployment and are not GitHub release assets.

## Build from the extracted tree

Check the outer archive SHA-256 against a separately trusted value. Extract into
a new directory, then run the following from the extracted `pmmc-source/` root:

```sh
python3 -B docker/verify_public_source.py .
python3 -B docker/verify_source_contracts.py src
python3 -B docker/verify_step8_source_contracts.py .
```

On a fresh Git checkout of the public tree, use `--git-checkout` with the
first command to ignore only root Git metadata; do this before dependency
installation, test runs or wheel builds create any extra files.

With Python 3.11 or 3.12 and a PEP 517 frontend, `python -m pip wheel --no-deps
-w ../wheels .` builds the PMMC wheel without downloading the GPU runtime. The
wheel is a Python package, **not** a replacement for this tree: keep Alembic,
configs and PMMC-authored prompts at their source-relative paths. Set
`PMMC_PROJECT_ROOT` to the extracted root when using the installed package.
Verify the pristine tree before running a build that creates extra files in it.

For a Linux NVIDIA deployment, build from the extracted directory with
`docker build --platform linux/amd64 .`. The Dockerfile uses the upstream
vLLM 0.24.0 base, Postgres client 16, platform-specific pip constraints from
`uv.lock`, project metadata and pip/runtime version
checks. It exercises legacy and Step 8 source/API checks, including the loaded
runtime prompts. A builder needs network access to the upstream registries,
package indexes and apt repositories. This is not an offline image or a claim
that dependency resolution, GPU behavior or image bytes are deterministic.
Override USER_ID/GROUP_ID for storage ownership when needed.

The default `VLLM_IMAGE` is
`vllm/vllm-openai:v0.24.0@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f`.
The official registry lookup for this candidate identified its linux/amd64
child manifest as `sha256:f9de5cd9fa907fbf6dbba691eb7db095d48ad58ea283e3eba7142f9a91e186e8`.
The default pins the index, not just a movable tag. An operator can override
it with `--build-arg VLLM_IMAGE=<compatible-verified-reference>` and repeat the
runtime checks for that base. A manifest lookup verifies neither a successful
Docker build nor CUDA compatibility on the deployment host.

## Distribution boundary

Project-authored compiler/runtime code, metadata, migrations, selected configs,
source-contract verifiers and the hash-bound PMMC prompt files are included.
Only exact manifest entries are shipped; workspace tests, docs, launch configs
and scripts are not included by default. No site-specific NFS/queue settings
or private registry image is a portable default. Supply your own deployment,
service credentials, data, model revisions, locks and permitted runtime inputs.
No private archive, data, image, weight, upstream benchmark prompt, run output,
approval or policy JSON is included.

The old private campaign exception modules are replaced **in staging only**.
A matching job/protocol/source fingerprint needs no exception. A mismatch,
old receipt or admission request cannot authorize work in this distribution.
No historical approval is anonymized, reissued or synthesized. Use the
unaltered private checkout for authorized historical recovery.
