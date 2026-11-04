# Prospective Multimodal Memory Compilation (PMMC)

PMMC compiles multimodal conversation memories *before* it sees a benchmark
query. A Questioner proposes evidence-grounded information needs, a Planner
compiles memory-access programs, and a Doubter checks those programs. At read
time the system matches a query to the question bank, executes its program,
checks evidence and grounds the answer. If this fails, the primary method
uses a native multimodal retrieval fallback. A fallback answer is **not** a
successful answer from a compiled program.

## What this release runs

The public suite is **exploration-dev-all**. It takes every eligible query in
the development partition of MEMLENS and MemGallery, not the reserved paper
partition or an independent confirmation cohort. It records the actual
denominator from the supplied snapshots. With the historical partition this
would be 140 + 287 queries, or 1,708 primary query/backbone positions across
four answer backbones. Positions are not API calls.

The software version is not a scientific approval. In an earlier development
run, all 112 newly scored primary positions used fallback. Twenty retrieved
images in the only program execution all scored below the active 0.30 cosine
threshold; lowering it did not produce a supported answer in the recorded
grid. Historical exposure is incomplete, and those results are not a paper
benchmark. This release does not claim that the new W (write) or R (read)
policy improves results. Assess their effects separately with the same
assignments, evaluator, budget/depth (8/2), and explicit missing outcomes.

## Requirements

| Environment | Requirements |
| --- | --- |
| Controller | Python 3.11 or 3.12, POSIX filesystem, writable private state and a durable data location. The host launcher does not need PyTorch or vLLM installed on the controller. |
| Local execution | Linux, Docker Engine with Compose v2, NVIDIA container runtime, and NVIDIA GPUs compatible with the configured model/precision/topology. macOS and CPU-only hosts are not full inference targets. |
| EIDF execution | Docker image build/registry access, kubectl for your project, a Kueue LocalQueue, writable persistent volumes, a pull secret when required, and sufficient CPU, memory and GPU quota. Do not use another project's namespace, NFS address or credentials. |
| Inference | The release runtime targets H100-class 80 GiB devices. The reference 9B pool uses eight TP1 replicas; 27B uses four TP2 replicas. A different machine requires a compatible measured profile; changing hardware is not permission to shorten prompts, drop images or substitute a model. |
| Storage | Allow for the pinned model snapshots, container images, original benchmark assets, database, object storage, CAS, checkpoints and scratch. Keep cache and output storage outside the checkout. Exact sizes depend on the materialized inputs; GPU resources are not needed for the host controller. |

## Inputs and credentials

1. Obtain the licensed benchmark payloads from their original sources. No
   benchmark images, source judge prompts, weights, API keys, private receipts
   or experimental outputs are distributed. See [benchmark layout](benchmarks/README.md).
   Mount the documented MEMLENS and MemGallery layouts under the profile's
   read-only input root. Preserve native identifiers, image files and MIME
   types. The release derives snapshots and the partition on first use.
2. Choose a profile in `deploy/release/`, copy it **outside** the checkout,
   fill in your storage and pinned image references, and set
   `PMMC_DEPLOY_CONFIG` to its absolute path. Both example files are templates,
   not someone else's live cluster configuration. See the [deployment guide](docs/release-deploy.md)
   for image construction, volumes, permissions, resource requirements and
   credential placement.
3. Keep provider credentials outside Git. A Hugging Face credential is only
   needed when the upstream repository requires one. Closed GPT-6 Luna and
   Claude Haiku 5.5 answers require separate selection-bound budget approval.
   Having credentials does not grant permission to spend. The earlier $20
   development cap does not transfer to this experiment.

## Run the primary experiment

After provisioning the profile prerequisites, run **one** command from the
repository root. Choose one backend, not both:

```bash
# EIDF
./scripts/pmmc run --profile eidf --suite exploration-dev-all --stage main

# Linux with local NVIDIA GPUs
./scripts/pmmc run --profile local-nvidia --suite exploration-dev-all --stage main
```

The controller reconciles persistent services, initializes the database and
object store, prepares locked models and the local dataset, then dispatches
the native Step 8 dependency graph. First use may need downloads, migration
and hardware-specific capacity preparation. A satisfied dependency is reused;
no smoke suite, pilot, regression subset, paper selection, or paid conformance
probe is part of exploratory startup. Never delete a volume to fix an image
or configuration mismatch.

Open-model work is scheduled first. An `awaiting_budget` receipt means that
the eligible open-model scope finished, **not** that the four-backbone matrix
is complete. Pending and failed positions remain in the reports. Keep the run
ID printed by the launcher. To resume the same deployment:

```bash
./scripts/pmmc resume --run "$RUN_ID"
```

Resume uses the frozen deployment and existing component receipts. It does
not authorize resending a request whose outcome is unknown. Reconcile failures
against the response journal, budget ledger and attempt record. A scientific
change requires a different protocol, not an in-place rewrite of an old
manifest. Reuse is valid only when the producing identity and input contract
match.

To price the pending closed-model portion, mount your reviewed rate card and
its companion `provider-policy.json` as private inputs. Quote first; approval
is a separate operation, tied to the printed selection hash. Do not substitute
the historical development cap for an approved experiment limit:

```bash
./scripts/pmmc quote --run "$RUN_ID" --variant wr --stage main \
  --rate-card /pmmc-inputs/rate-card.yaml
# Only after deciding an explicit limit, with the quote's exact selection hash:
./scripts/pmmc approve --run "$RUN_ID" --variant wr --stage main \
  --selection-hash "$SELECTION_HASH" --approver-id "$APPROVER_ID" \
  --limit-usd "$APPROVED_LIMIT_USD" --rate-card /pmmc-inputs/rate-card.yaml
./scripts/pmmc resume --run "$RUN_ID"
```

## Stage order and interpretation

Run the PMMC primary stage (open 9B/27B, then separately approved closed
answers), then analysis/ablations, then the five independent baselines:
full context with captions, full context with raw images, caption/text RAG,
contrastive multimodal RAG, and native multimodal RAG. The fallback needed
by a primary answer is *not* deferred until the baseline stage. W and R have
separate versioned configurations. Never treat results from an incompatible
older protocol as the control arm. Report genuine program answers, fallback,
model abstention, grounding rejection, no-answer and engineering failure
separately. Report coverage and denominators alongside conditional metrics.

```bash
./scripts/pmmc resume --run "$RUN_ID" --stage analysis
./scripts/pmmc resume --run "$RUN_ID" --stage baselines
```

Each comparison variant has its own selection. A WR approval does not
authorize W, R, or legacy. A stage with pending paid positions is unfinished;
the baseline stage waits for the PMMC and comparison stages to finish.

## Code, outputs and recovery

| Path | Purpose |
| --- | --- |
| `src/pmmc/` | Compiler, query runtime, model adapters, persistence, evaluation and native campaign workers |
| `configs/release/`, `deploy/release/`, `scripts/pmmc` | Public reproducible inputs, deployment templates and host entry point |
| `artifacts/` | Generated private snapshots and model state; not required in a fresh source checkout |
| `outputs/step8/exploration/<run-id>/` | Immutable input/variant pointers to actual campaign directories |
| `outputs/step8/<campaign>/manifests/`, `checkpoints/`, `attempts/` | Assigned denominator, job plan, completed component receipts, retry and failure evidence |
| `outputs/step8/<campaign>/reports/` | Stage receipts, scoped exploratory results, pending-budget estimates when requested |
| `outputs/step8/_store/` | Private shared CAS, exposure and model-response state. Not a disposable log directory |

On failure retain the **single file printed as `SYNC_REVIEW_FILE`** for private
review. Do not publish it. A diagnostics bundle is not a database/backing-store
backup. Before wiping any machine follow [private preservation and recovery](docs/release-recovery.md);
verify an independent off-host restore first.

## License and attribution

Original PMMC code is licensed under [Apache-2.0](LICENSE). Upstream datasets,
model weights, provider services and third-party prompt assets keep their own
terms; publication of the software grants no permission to redistribute them.
For reproducibility cite the repository tag, scientific protocol, model lock,
dataset sources, selection hash and stage/scope of the reported experiment.
