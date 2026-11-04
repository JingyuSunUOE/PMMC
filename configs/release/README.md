# Exploration CPU preparation

From the release project root:

    python -m pmmc benchmark prepare-exploration --run-id trial-01
    python -m pmmc benchmark prepare-exploration --run-id trial-01 --variant r

Both commands are CPU-only. Preparation invokes the native Typer `benchmark plan
--phase exploration --selection dev-all` to publish the complete job DAG. It
does not register or execute jobs, configure Postgres/MinIO, install software,
download weights, authorize spend, or start a model. No network, GPU, DB or
provider credentials are required. The planner needs the release source tree,
prompts, configs and native Python application dependencies at `PMMC_PROJECT_ROOT`.

**Paths and mounts.** Stage raw materials privately at
`$PMMC_BENCHMARK_ASSET_ROOT/benchmarks/{memlens,memgallery}`. Mount that root
read-only; keep the artifacts directory and project outputs writable and
persistent. Snapshots under `artifacts/benchmarks` contain gold and private
benchmark data. Never include inputs, outputs, model caches, run pointers,
receipts or credentials in a public image, archive or upload. Only the source
allowlist and shipped configs belong in an image build context. The output
filesystem must support POSIX file locks, hard links, fsync and atomic renames.

**CLI/worker interface.** `prepare_exploration(settings, run_id, *, variant="wr")`
returns a JSON dict with `state="planned"`, `config_path`, `campaign_dir`,
`pointer_path`, `source_identity`, `campaign_id`, `protocol_hash`,
`partition_hash`, `selection_hash`, `exposure_hash`, selected counts and
execution-input checksums. Paths are project-relative. Only publish a success
after this function returns. The immutable pointer is at
`outputs/step8/exploration/<run_id>/<variant>/pointer.json`. Workers call
`exploration_campaign_directory(settings, run_id, *, variant="wr") -> Path`:
it validates that exact pointer against the campaign and returns its absolute
path; it never creates or searches for a campaign. Run IDs are single ASCII
components (alphanumeric first, then letters, digits, dot, dash, underscore;
96 chars max). Do not route by timestamp, protocol hash alone or mtime.

**Cohort and identity.** Both datasets undergo FULL asset and canonical
validation with the native package counts. The input receipt seals both snapshot
IDs, source/canonical/file hashes, and the original scenario-balanced five-fold
**v2** partition (seed 0, development fold 0; original left-to-right float
summation). The packaged `original-split-v2.json` holds only non-identifying
digests, derived from the named historical v2 assignment manifest with content
hash `2118aeb55a25db9e346a2cd4c0fa6ee74d84ae26f17be1a9c80359b0d4bb1096`.
FULL snapshot/partition IDs legitimately differ from their historical FAST IDs.
Before planning, verify the FULL data identities and exact membership of every
fold, not just counts. The assignment digest is SHA-256 of canonical JSON for
sorted `[query_id, fold, role]` triples per dataset; role digests use sorted
query IDs alone. Workers recheck these digests and the actual selected IDs.
No upstream private assignment file is required. On the named datasets the
only accepted dev cohorts are 140 MEMLENS and 287 MemGallery. Old v3 plans are
not resumable; new data needs a separately versioned dataset and reviewed split
contract, not merely a new run ID. Do not update hashes from a failed candidate.
For an independent local comparison, optionally set
`PMMC_EXPLORATION_REFERENCE_PARTITION` to the directory of the named private
upstream bundle and run `pytest tests/test_release_inputs.py -k full_release_matches_private`.
This test rebuilds FULL data in a temporary directory and compares all assignments;
it neither plans nor executes jobs. The environment variable does not override
the release checksum gate.
Dev-all selects every eligible
development query; paper-test queries are never selected. Only the CPU importer
needs to touch gold; it is not used for sampling. The run freezes an explicitly
incomplete exposure history from the common Step 8 production ledger, retained
campaign-private dispatch receipts, frozen release exposure manifests, optional
historical sample records and prior selection manifests. Planned legacy
assignments, and release selections with execution attempts/reports, count
conservatively as possible exposures. An operator may optionally mount
`$PMMC_BENCHMARK_ASSET_ROOT/known-exposure.json` as a native `ExposureManifest`
with the source partition hash and an explicit ID list for each dataset. No
private history file is required or uploaded; the union remains incomplete even
if an individual historical file claimed completeness. Known IDs outside the
new partition fail closed. Missing history
means unknown, not unseen. No historical regression file is required. The
shipped `no-regression.json` is an absence marker for the native reconstruction
check, not an invented historical sample.

| Variant | Questioner | Query executor |
| --- | --- | --- |
| `wr` (default/main) | release W | release R |
| `w` | release W | legacy v3 |
| `r` | legacy v3.6 | release R |
| `legacy` | legacy v3.6 | legacy v3 |

Each variant has a separate immutable config, campaign and job inventory;
all variants of one ID share the same exposure and partition. The source hash
binds the same scientific `SOURCE_OWNERS` stage fingerprints as the native
planner (including its prompt owners, excluding export), both W/R configs,
legacy method configs, scientific policy and locked model identities. On lookup
the worker recomputes that identity and independently makes the native protocol
from the mounted sources. CLI, reporting, docs and image-only changes do not
invalidate the scientific pointer. The execution image belongs in each
attempt's `PMMC_CONTAINER_IMAGE`; the runtime/execution YAML checksums are
validated separately. Use a new run ID for a changed scientific method or
frozen execution profile. A new ID creates a new
campaign even if the scientific protocol is unchanged. New exposure evidence
can legitimately change that protocol. No inherited approval, budget, paper
freeze, reconstructed method, or private import policy is put in a fresh plan.
A retry after an interrupted publication resumes only a unique campaign whose
config matches both ownership keys; disagreement/ambiguity fails closed.

## Models and GPU handoff

1. **CPU cache staging** (independent of benchmark preparation):
   `python -m pmmc benchmark prepare-models --config configs/release/prepare-models.yaml`.
   Offline by default. On an empty cache only, an operator may explicitly add
   `--allow-download` to fetch public locked weights. The registry pins all
   open-model weights, tokenizers and processors to the five exact Hub commits;
   the shipped native lock contains file sizes/digests and separate closed-provider
   alias hashes. Use the shipped lock, not `--create-lock`. The native service
   performs full cache verification followed by an offline cached-receipt pass.
   This template is not a campaign config; planning uses the run's `benchmark.yaml`.
2. **GPU startup is separate.** Use the same source image, cache, registry, lock,
   `runtime.yaml`, and private calibration directory for calibration and workers.
   Set `PMMC_MODEL_REGISTRY_PATH=configs/release/models.yaml`,
   `PMMC_RUNTIME_MODEL_LOCK_PATH=configs/release/models.lock.json` and the
   hardware/accounting settings required by the eight-H100 deployment. On
   separately authorized hardware run `python -m pmmc runtime calibrate
   --profile configs/release/runtime.yaml` for generic hardware evidence and
   `python -m pmmc benchmark calibrate --config configs/release/prepare-models.yaml`
   for the synthetic 27B full-pool receipt. These commands **do** start GPU
   services; neither is invoked by prepare-exploration. Never copy a passing
   receipt from another profile or assume an idle calibration covers 27B.
3. **Capacity boundary.** The release profile freezes the shared 9B TP1x8 and
   27B TP2x4 pools. Its 27B native workload contains only numeric image layouts
   from the retained CPU audit identified by its content hash; no queries,
   images, weights or approval are included. Its 29-image envelope preserves
   native processor kwargs and explicitly disables prefix caching. Calibration
   must publish `step8-exploration-27b-native-capacity.json`. Even a matching
   receipt does not prove every dev-all request fits the envelope. Out-of-bounds
   requests must fail the existing runtime guard. Expand/review the workload
   and prepare a new run/profile and measurement; do not shrink images, drop
   queries or claim completion.
4. **Execution** uses the parent-owned `run-exploration` dispatch and deployment
   service bootstrap. Provider spend requires separate explicit selection-bound
   authorization. A prepared pointer is never a GPU-readiness or paper approval.
