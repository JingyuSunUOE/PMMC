# Prospective Multimodal Memory Compilation

PMMC compiles executable evidence-access programs **before a future query is known**. At query time it matches the question bank, executes a program, filters evidence and grounds the answer. Its native retrieval fallback remains part of PMMC, but a fallback answer is not a successful program answer.

The public suite is **full development exploration**, not independent confirmation or paper-test. At the reference partition, MEMLENS contributes 140 queries and MemGallery 287; with all four Answer backbones this means 1,708 primary score positions, **not** 1,708 calls. Each run records its actual denominator. No accuracy improvement or hardware speedup is implied.

## Requirements

Use a Kubernetes cluster with Linux/amd64 workers, including a single-node cluster. The controller needs Python 3.11/3.12 and kubectl; it does **not** need Torch, vLLM, a GPU, or the research Python environment. Supply a dedicated namespace, permission to manage the project workloads/storage/Secrets, a suitable StorageClass and a writable work volume (pre-provisioned or explicitly managed by PMMC). GPU workers need compatible NVIDIA devices, drivers, device plugin, enough memory for the chosen models, and an accepted capacity profile. A CPU-only controller is not a CPU inference backend.

The public configuration does not name anyone's cluster. Standard Kubernetes is the default; Kueue is optional. Tensor parallelism stays within one node. More independent replicas increase throughput only when they fit; adding CPU or GPUs cannot make a model fit an incompatible device. Do not shorten context, drop images or silently switch weights to fit hardware. Host state, dataset assets, registry credentials and results are private.

## First deployment

1. Clone the **release tag**. Obtain the `images.lock.json` asset and its independently published SHA256 from the [release](https://github.com/JingyuSunUOE/PMMC/releases/tag/v0.2.0). The source cannot contain its own image digest. The lock binds the exact source to the application, IAM, PostgreSQL/pgvector and MinIO images. See [image verification](docs/release-images.md) for mirrors and source verification.
2. Copy the generic deployment template outside the checkout, choose the namespace, cluster context, storage class, shared claim and per-node resource envelope for your own installation. Keep host state on persistent private storage. See [Kubernetes deployment](docs/release-deploy.md).
3. Ensure the cluster can pull the locked images and fetch the selected licensed benchmark/model sources. No benchmark or weights are distributed with PMMC. The [benchmark guide](benchmarks/README.md) documents the upstream layout and licensing boundaries.

```bash
git clone --branch v0.2.0 --depth 1 https://github.com/JingyuSunUOE/PMMC.git
cd PMMC
curl -fsSL https://github.com/JingyuSunUOE/PMMC/releases/download/v0.2.0/images.lock.json -o deploy/release/images.lock.json
export PMMC_IMAGE_LOCK_SHA256="$(curl -fsSL https://api.github.com/repos/JingyuSunUOE/PMMC/releases/tags/v0.2.0 |
  python3 -c 'import json,sys; print(next(a["digest"].removeprefix("sha256:") for a in json.load(sys.stdin)["assets"] if a["name"]=="images.lock.json"))')"
install -d -m 700 "$HOME/.config/pmmc"
cp -n deploy/release/kubernetes.toml "$HOME/.config/pmmc/cluster.toml"
export PMMC_DEPLOY_CONFIG="$HOME/.config/pmmc/cluster.toml"
${EDITOR:-vi} "$PMMC_DEPLOY_CONFIG"
```

Start the main experiment directly. The controller reconciles services, assets, snapshots, model cache and capacity before dispatching the native dependency graph. No smoke campaign or paid probe is inserted:

```bash
./scripts/pmmc run --run-id dev-main --stage main
./scripts/pmmc status --run dev-main
./scripts/pmmc resume --run dev-main
```

Alternatively, prepare in steps. `services up` creates or reuses two separate persistent volumes, one for PostgreSQL and one for MinIO. A foreign volume or conflicting credentials stop the operation; existing data are not deleted. Asset/model caches and outputs are separate from these service volumes.

```bash
./scripts/pmmc services up
./scripts/pmmc services status
./scripts/pmmc prepare --run-id gallery-9b --benchmark memgallery --model answer-qwen3.5-9b
./scripts/pmmc run --run-id gallery-9b --benchmark memgallery --model answer-qwen3.5-9b
```

On second use the launcher checks identities and fills missing dependencies rather than downloading every weight again. Ready services do not need a new volume or password. A different treatment, model, asset revision, or selection is not a resume. Use a new run ID; reuse only proven producer results.

## Select an experiment

The public selectors enumerate supported, scientifically bound contracts.
For a new adapter, model or method see [extension points](docs/extensions.md).

`--benchmark`, `--model`, `--method`, `--variant`, `--analysis` and `--component` accept repeated options. `--gpus`, `--cpus`, `--memory` set total ceilings, not per-worker allowances. Run `./scripts/pmmc plan --help` for the supported catalog. `plan` prints a normalized intent and hash without creating a workload. Actual query counts and job identities are computed from the prepared snapshots.

```bash
# One backbone and dataset
./scripts/pmmc run --run-id gallery-limited --benchmark memgallery \
  --model answer-qwen3.5-9b --gpus 2 --cpus 24 --memory 160Gi

# One independent baseline; no PMMC compiler or other baseline requested
./scripts/pmmc run --run-id rag-only --stage baselines --method caption_text_rag \
  --benchmark memlens --model answer-qwen3.5-9b

# Descriptive cost report using the prior main run; no new model requests
./scripts/pmmc run --run-id cost-analysis --stage analysis --analysis cost --from-run dev-main

# Full workflow: PMMC primary, analyses/ablations, then five baselines
./scripts/pmmc run --run-id development-all --stage all
```

Standalone analysis reads sealed results. If a requested source is absent, it names the missing dependencies and stops. Only add `--with-deps` when you intend to execute missing scientific work. Subset scores are reported against the declared subset, not padded to a full benchmark. Unselected positions, pending work, engineering failures, model refusal, grounding rejection, no-answer, fallback and genuinely supported answers have different meanings. Paired W/R effects need matching observed cohorts; missing values are not zeros.

## Credentials and paid work

PMMC does not read a checkout `.env` automatically. Create a private file **outside** Git with mode 0600, for example `$HOME/.config/pmmc/providers.env`. Populate only the credentials you intend to use:

```dotenv
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
HF_TOKEN=your-optional-hub-token
```

```bash
./scripts/pmmc secrets import --env-file "$HOME/.config/pmmc/providers.env"
./scripts/pmmc quote --run dev-main --rate-card /pmmc-inputs/rate-card.yaml
# Use the exact plan/selection hashes from the quote. Choose ONE budget mode.
./scripts/pmmc approve --run dev-main --rate-card /pmmc-inputs/rate-card.yaml \
  --plan-hash "$PLAN_SHA" --selection-hash "$SELECTION_SHA" --approver-id "$REVIEWER" \
  --limit-usd "$MAX_USD"
./scripts/pmmc resume --run dev-main
```

The rate card and companion reviewed provider policy describe actual pricing and account limits. Start from [the unapproved template](configs/models/step8-provider-rate-card.template.yaml); fill actual provider prices and your own account limits. Keep `provider-policy.json` next to it: the pricing review binds the card name and SHA256, registry SHA256, reviewer/date and HTTPS price sources; it is not an API key or a spend authorization.

API keys alone cannot authorize spending. For explicitly uncapped use **`--uncapped` instead of `--limit-usd`**: the plan may incur charges without a PMMC total cap. Requests, RPM/TPM, cost reservations, settlements and unknown outcomes remain accounted for. Do not retry an unknown charge automatically. Expanding the plan requires a new approval. No historical $20 allowance transfers.

```bash
./scripts/pmmc spend --run dev-main
# Permanently stop NEW reservations; existing reservations are not released.
./scripts/pmmc stop-budget --run dev-main --approval-id "$APPROVAL_ID" \
  --plan-hash "$PLAN_SHA" --selection-hash "$SELECTION_SHA" --operator-id "$REVIEWER" \
  --reason-code operator_stop
```

Stopping is confirmed only by the ledger receipt, not by interrupting the terminal. For a single manual handoff of the public source plus encrypted private settings/credentials, see [private deployment](docs/private-deploy.md). Neither packing nor importing secrets authorizes spending.

## Files, results and recovery

| Location | Use |
| --- | --- |
| `src/pmmc/`, `scripts/pmmc` | Native compiler, runtime, evaluator, job graph and host commands |
| `deploy/release/`, `configs/release/` | Generic cluster settings, pinned sources, registered models and experiment recipe |
| Private assets, HF cache and `artifacts/` | Original files, validated snapshots, model identities and measured hardware capacity |
| `outputs/step8/exploration/<run-id>/` | Variant pointers, selected scope, sealed analysis outputs |
| `outputs/step8/<campaign>/` | Manifests, immutable job assignments, attempts, receipts, checkpoints and reports |
| `reports/exploration-*.json`, `reports/seal-*.json` | Selected scores, coverage, outcome categories, protocol, provenance and readout identity |
| Controller private state | Configuration, durable progress, workload ownership and bounded private diagnostics |

During execution, follow the phase, queue reason, model progress and elapsed time. Exit 3 means paid work is waiting for approval; exit 4 means required analysis evidence is missing. Preserve the single file printed as `SYNC_REVIEW_FILE` for **private** debugging. It is not a database backup or a complete copy of the campaign. Do not publish secrets, raw responses, licensed images or diagnostics. Restart with the same run ID and frozen plan; do not empty the namespace to resolve a mismatch.

Original PMMC code: [Apache-2.0](LICENSE). Third-party materials retain their own terms; see [third-party notices](THIRD_PARTY.md). Cite the software tag, model/source lock, partition, plan hash and selected denominator when reporting exploratory results.
