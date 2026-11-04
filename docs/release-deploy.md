# Public exploration deployment

`./scripts/pmmc` runs on the **host**. It does not install the scientific package
on the host. Containers run from the digest-pinned source image, working directory
`/workspace`, with `PYTHONPATH=/workspace/src`. There is no source-tree bind mount,
host `.env` import, synthetic data, paper approval, or fallback backend.

## Images

Use a verified, unpacked **public SOURCE archive** as the build context. Never use
a private research checkout as a Docker build context or remove its deny-all
`.dockerignore`. Run these commands **from the unpacked archive root**, where the
archive's exact-path Docker context allowlist, Dockerfile, IAM Dockerfile,
verifiers, migrations, configs, prompts and public source are present. You need
Buildx, jq, an authenticated registry you control, a verified Linux/amd64 vLLM
base matching the Dockerfile, and permission to publish there. These are operator
commands; the launcher does not build/push images or query a registry for digests.

```sh
# Set REGISTRY_PROJECT to your registry/repository prefix (no credentials).
# VLLM_IMAGE is the owner-verified vllm/vllm-openai:v0.24.0 image @sha256 digest.
: "${REGISTRY_PROJECT:?}" "${VLLM_IMAGE:?}"
APP="$REGISTRY_PROJECT/pmmc"
IAM="$REGISTRY_PROJECT/pmmc-iam"
META="$(mktemp -d)"
docker buildx build --platform linux/amd64 --push \
  --build-arg VLLM_IMAGE="$VLLM_IMAGE" -t "$APP:exploration" \
  --metadata-file "$META/app.json" .
APP_REF="$APP@$(jq -er '."containerimage.digest"' "$META/app.json")"
docker buildx build --platform linux/amd64 --push -f docker/minio/step8.Dockerfile \
  --build-arg PMMC_IMAGE="$APP_REF" -t "$IAM:exploration" \
  --metadata-file "$META/iam.json" .
IAM_REF="$IAM@$(jq -er '."containerimage.digest"' "$META/iam.json")"
printf 'app = "%s"\niam = "%s"\n' "$APP_REF" "$IAM_REF"
```

The IAM image uses the same app digest and the mc digest pinned in its Dockerfile.
Build with the app Dockerfile's default UID/GID 10001:10001, or supply your
cluster-assigned `--build-arg USER_ID=...` and `--build-arg GROUP_ID=...` and set
the same positive integers in TOML `[identity]` (`uid`, `gid`). Verify image
provenance, native CLI/entrypoints, release configs and registry access before use.

`images.postgres` is **required**: use an approved **PostgreSQL 16 + pgvector**
digest, e.g. your verified digest for the upstream `pgvector/pgvector:pg16`
Linux/amd64 image. Use the plain service, not the historical init-roles image;
native bootstrap provisions the logins. No verified PostgreSQL digest is
shipped. The example MinIO digest is the one pinned in the repository Compose
file (RELEASE.2025-04-22T22-12-26Z); the local healthcheck needs curl in the
image. Verify/replace with an approved compatible digest. Each of app, IAM,
Postgres and MinIO must be a registry reference ending in `@sha256:<64 hex>`.
No mutable tag or invented digest is accepted in place of an operator pin.

## Host and storage

Python 3.11+ on a POSIX host with flock. Keep a private, backed-up host state
directory, e.g. `export PMMC_DEPLOY_STATE="$HOME/.local/state/pmmc-release"`
(the default). Protect the parent directory. One trusted controller and one
state directory per deployment; do not use multiple independent controllers
against the same volumes/PVCs. Full normalized TOML and image identities are
frozen at first launch. A different configuration requires a different instance
and dedicated storage. Losing the state directory loses credentials and the
mapping to detached workloads; restore it rather than initializing over data.

Copy either `deploy/release/local-nvidia.toml` or `deploy/release/eidf.toml` to a
private location, set its images and storage, then export
`PMMC_DEPLOY_CONFIG=/absolute/path/to/deployment.toml`. Alternatively pass
`--config /absolute/path`. Templates intentionally reject execution with unresolved
image/cluster placeholders. Config contains **no secrets**. Unknown keys are errors.

| Mount key | Container path | Contents and access |
| --- | --- | --- |
| inputs | /pmmc-inputs | Read-only licensed raw data at benchmarks/memlens and benchmarks/memgallery; optional reviewed rates/policy and known-exposure.json |
| outputs | /workspace/outputs | Writable, durable campaign pointers, plans, worker locks, journals, receipts and reports |
| artifacts | /workspace/artifacts | Writable validated snapshots (including private evaluator material) and hardware receipts |
| cache | /pmmc-cache | Writable shared HF snapshots and .pmmc verification receipts |

All four directories must be disjoint and pre-created with suitable access for
the app UID/GID. Use dedicated storage, never an old research tree. Outputs and
artifacts must support POSIX locks, hard links, fsync, and atomic renames across
all workers. Inputs, artifacts, cache and outputs are private; do not publish them.

- **local-nvidia:** Linux, local Docker daemon in context `default`, Docker Compose
  v2 with `up --wait` and NVIDIA Container Toolkit. The supplied native topology
  requires eight compatible H100s on one host; this is not a laptop/single-GPU
  fallback. Provision CPU/RAM, 64 GiB GPU shm limit and scratch/disk space for
  staging weights. Mount values are absolute host directories. The launcher
  creates a dedicated Compose project, internal network, persistent named Postgres
  and MinIO volumes, and waits for healthy services. It publishes no host ports.
- **eidf:** working kubectl credentials for the **explicit** context/namespace,
  Kueue LocalQueue, NVIDIA device plugin, H100 node label, registry access on the
  namespace's default service account, dynamic RWO storage class, and a
  pre-provisioned bound RWX PVC shared by CPU and GPU workers. Mount values are
  relative subdirectories of that PVC, not NFS server paths. TOML `[cluster]` keys:
  `context`, `namespace`, `queue`, `shared_pvc`, `storage_class`. RBAC must allow
  get/create/apply of the dedicated services, Secrets, PVCs and StatefulSets,
  Job create/get/watch/delete, get/list Pods and logs, read namespace ResourceQuota,
  LocalQueue and Kueue Workloads. Never
  disable Kueue gates. Each Job is created suspended with queue labels on workload
  and Pod; Kueue admits it. Services use dedicated RWO PVCs (300 Gi Postgres,
  120 Gi MinIO). The default concurrent allocation is **92 CPU, 768 GiB, 8 H100s**
  plus platform overhead. The LocalQueue must admit **both** worker pools together.
  The GPU pool gets 64 CPU/600 GiB, control 20/128, Postgres 6/32 and MinIO 2/8.
  The eight replicas are capped at seven CPU threads each by default; the cap
  tracks GPU CPU overrides (one minimum, eight maximum). Override resources in
  either profile with e.g. `[resources.control]` and integer `cpu = 16`,
  `memory_gib = 128`. All role names in the offline plan plus postgres/minio accept
  overrides. Neither overrides nor CPU thread caps reduce scientific cell counts.

The EIDF profile has `cluster.cpu_budget=100` and
`cluster.admission_timeout_seconds=600`. The launcher rejects a configured peak
above the CPU budget before creating any resources. Before reconciling services
and again before active workers, it compares the actual namespace ResourceQuota
hard/used values with peak requests, accounting for already active managed Pods.
`PMMC_ADMISSION` reports queue, peak and headroom (memory in bytes); insufficient
headroom or unreadable quota stops the launch. Empty quota inventory is **not**
evidence of Kueue capacity. Kueue flavors, other members of a cohort, instantaneous
GPU availability, scoped quotas, PVC placement, admission checks and overhead
can still prevent scheduling. Requests/limits are equal on EIDF. A pending Job
reports its Workload/condition or Pod wait reason. While the controller is
attached it fails and stops the pending Job and live peer after the configured
timeout. Unready services also time out with a reason; it does not delete their
volumes. Ctrl-C explicitly detaches the controller and its timeout monitor;
resume to continue monitoring. An unadmitted detached Job is not guaranteed
to expire by itself. Do not leave a detached deployment unattended.

## Launch and resume

```sh
./scripts/pmmc plan --profile eidf --suite exploration-dev-all --stage main
./scripts/pmmc run --profile eidf --suite exploration-dev-all --stage main --run-id trial-01
./scripts/pmmc status --run trial-01
./scripts/pmmc resume --run trial-01
./scripts/pmmc resume --run trial-01 --stage analysis
./scripts/pmmc resume --run trial-01 --stage baselines
```

Substitute `local-nvidia` for Compose. Omit `--run-id` to generate an ID (printed
in the first JSON event). `plan` is offline and does not create state. `resume`
finds the frozen configuration from the host state, independent of the current
directory. Detached Jobs/containers have deterministic per-attempt names. Resume
reattaches after interrupted submission; a previously observed, now missing
workload is a manual-review error, not permission to start another paid worker.
A known terminal failure is retryable via `resume` after fixing its cause. The
launcher stops a still-active peer on confirmed failure. It does not delete
service volumes, reset the database, retry ambiguous provider requests, or
clean up scientific attempts. Terminal workload logs remain private on the backend.

For each unit the launcher reconciles services; runs strict file-projected MinIO
IAM, native `benchmark bootstrap`, cache reconciliation and native
`prepare-exploration --run-id ID --variant V` (CPU); validates the immutable
pointer `outputs/step8/exploration/ID/V/pointer.json`; then starts
`python -m pmmc benchmark run-exploration --run-id ID --variant V --stage S
--worker-pool control|gpu` in **separate** containers. The pointer is not a result.

- Main = WR main. Analysis = WR analysis plus W, R, legacy **main** comparisons
  after WR main completed. These four units run serially through the same GPU
  allocation. Each may pause independently; no implicit variant budgets.
  `--stage analysis --variant w` targets just W main (`r`, `legacy`, `wr` also
  supported). Baselines = WR baselines only, after **all four** analysis units.
- Native `awaiting_budget` means open-model work finished and paid work is
  unapproved. Host exit **3**, not completion. A GPU's `waiting_for_workers`
  receipt may precede the final control receipt. The host requires the same
  native scope hash and a terminal control receipt with all scoped jobs counted.
  Exit 0 means only the requested units completed, never paper approval.
- Ctrl-C detaches; it does **not** cancel an admitted job or spending authority.
  Run `status` for the last host checkpoint and `resume` to follow live progress.

## Cold start

The shipped release registry, model lock, runtime profile and cache-preparation
config are under `/workspace/configs/release/{models.yaml,models.lock.json,
runtime.yaml,prepare-models.yaml}`. In `[native]` these correspond to
`model_registry`, `model_lock`, `runtime_profile`, `benchmark_config`.
Keep them consistent with the native pointer recipe. Optional `release_profile`
is passed verbatim as `PMMC_RELEASE_PROFILE` (an absolute container config path);
the stock recipe does not require or interpret that variable. It is not an
override for native selection, pricing, calibration, or authorization gates.

Templates enable `[models] allow_download=true, create_lock=false,
use_hf_token=false, calibrate=true`. CPU preparation first checks live,
offline cached receipts against **all exact locked aliases** and the registry.
On missing/unverified files the real native FULL verifier fetches pinned commits
as needed and requires offline re-verification. It doesn't resolve a newer model
or refresh a shipped lock. Set `allow_download=false` to prohibit Hub access;
missing weights then fail. `use_hf_token=true` exposes HF_TOKEN only to the model
preparation job for authorized downloads. Lock creation is rejected for the
public release. On the GPU allocation, before native dispatch, the generic
calibrator reuses verified hardware-specific file/DB records and measures missing
ones. The capacity gate independently validates the full pool, hardware,
model/profile/layout and backend identity; if missing/mismatched it measures the
actual pool there. Fresh measurement must pass. Receipts live under
`artifacts/runtime/calibrations/release/<app-digest>`; no old pilot is assumed.
These checks consume real GPU quota, even when only doing validation. A single
capacity receipt name can be replaced when admitted on different hardware.
`calibrate=false` disables automatic measurement, **not** native readiness checks.

## Pricing and credentials

The host generates independent service/DB/S3 credentials once in private host
state (0600), not from the user's checkout. Kubernetes receives an immutable
dedicated Secret via stdin; only explicitly referenced keys are exposed per
workload. No Secret values in manifests saved to disk, CLI arguments, progress
or diagnostics. Local Compose creates one private six-file IAM projection
readable by the host UID. Cluster IAM has no shared mount or SA token. Bootstrap
has DB owner and S3 root; CPU control needs DB owner for import/checkpoints, but
never S3 root. GPU gets worker/evaluator/online credentials, **no DB owner, HF
token, MinIO root, or provider key**. Quote/approve have the runtime DB URL only.

Stage your reviewed rates (e.g. `/pmmc-inputs/rates.yaml`) and matching
`provider-policy.json` side by side as required by the native pricing verifier.
The release rate-card template is not a reviewed price list. `--rate-card` is
a **container** path, not a host path; optional `native.rate_card` supplies
the default. Use the returned selection hash and an explicit finite USD limit:

```sh
./scripts/pmmc quote --run trial-01 --variant wr --stage main --rate-card /pmmc-inputs/rates.yaml
./scripts/pmmc approve --run trial-01 --variant wr --stage main --rate-card /pmmc-inputs/rates.yaml \
  --selection-hash "$SELECTION_SHA256" --approver-id "$REVIEWER" --limit-usd "$APPROVED_LIMIT"
./scripts/pmmc resume --run trial-01
```

Quote and approval are CPU-only native commands. Neither starts model inference
nor calls a provider. Approval is persisted by the native service and is specific
to that variant, campaign, selection and rates. Quote/approve each alternate
independently (`--variant w|r|legacy --stage main`); never copy a ledger or an
approval ID across variants. For an external pre-existing native approval, pass
`--budget-approval SHA256 --variant V` to run/resume. A read-only preflight
checks the actual variant pointer, selection-specific budget ID and archived
rate card before provider keys are injected into control. Native registration
also checks the requested ID. No inherited/private approval is synthesized.

For a paid local control worker supply `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` to
the launcher environment, not the TOML. For EIDF pre-create an operator-managed
Secret `<instance>-providers` with those exact keys; for gated model downloads
use Secret `<instance>-hub`, key `HF_TOKEN` (local: host env). Open runs need
neither provider Secret nor provider keys. Kube Secret encryption/RBAC, isolation
of privileged host/daemon administrators, network policy, TLS where required,
storage backup/rotation and registry access remain operator responsibilities.
Service traffic is internal HTTP; no public ingress or password rotation provided.

## Diagnostics and limits

Events show native unit, attempt/job names, running/queued state and final
validated scope counts. `SYNC_REVIEW_FILE=<host-path>` identifies the private
tar.gz emitted after each terminal/interrupted task. `./scripts/pmmc diagnostics
--run ID` repacks the saved evidence; `--refresh` collects current evidence.
The archive contains normalized deployment/progress, a bounded event tail and
up to 32 recent attempt evidence summaries: native manifest digests, safe error
codes and PMMC module/function/line frames, projected safe attempt log events,
stage receipts and numeric metric samples. It never copies prompts, raw exception
text, gold, responses, Secret values, snapshots or entire campaign trees.
The EIDF reader is a time-bounded CPU Kueue Job with **only** a read-only outputs
PVC subpath, no credentials or SA token. A missing pointer, oversized report,
expired logs or unavailable cluster is marked unavailable/omitted. Review the
original private backend logs/PVC under approved access when projections are
insufficient. Host topology, reviewer IDs and selection hashes still require
review before sharing this archive.

Offline verification from the source root: `python -m pytest -q
tests/test_release_host.py`. These tests simulate command/admission boundaries
and exercise real argument/rendering, state recovery, budgets, variant barriers,
cold-cache gates and redaction. They do not demonstrate live EIDF admission,
Compose/NVIDIA, registry compatibility, measured capacity, paid-provider
conformance or scientific success. Insufficient queue quota, disk, hardware,
licenses, data, network, rates or a rejected native gate blocks execution.
