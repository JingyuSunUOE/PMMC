# Public benchmark preparation

PMMC ships the source lock and split hashes, not the benchmark payloads, prompts,
canonical snapshots, query IDs, or images. Obtain the data directly from its
authors. Keep the populated volume, its staging cache, generated artifacts and
outputs out of source trees, images, releases and public reports.

| Benchmark | Data and terms at the pinned revision | Prompt source |
| --- | --- | --- |
| MEMLENS 32K | [MEMLENS](https://huggingface.co/datasets/xiyuRenBill/MEMLENS/tree/afa101a1907cc37db40b50d649547964387b96b7); [LICENSE-DATA](https://huggingface.co/datasets/xiyuRenBill/MEMLENS/blob/afa101a1907cc37db40b50d649547964387b96b7/LICENSE-DATA). The authors' metadata is CC BY 4.0. The images retain their original source-site terms. | [xrenaf/MEMLENS](https://github.com/xrenaf/MEMLENS/tree/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2/judge_prompts); [MIT code license](https://github.com/xrenaf/MEMLENS/blob/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2/LICENSE-CODE). |
| MemGallery, 20 scenarios | [Mem-Gallery dataset card](https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery/blob/af912daba984e896e253016b7c7e334ef92c2a6f/README.md) identifies the dataset as MIT and supplies the authors' citation. | [YuanchenBei/Mem-Gallery](https://github.com/YuanchenBei/Mem-Gallery/tree/a93959e1e978a6a7d77798ae92c2ffe41c538c62/benchmark); [MIT license](https://github.com/YuanchenBei/Mem-Gallery/blob/a93959e1e978a6a7d77798ae92c2ffe41c538c62/LICENSE). |

Review the linked terms and attribution requirements before provisioning. The
PMMC source license does not grant rights to third-party works. The adapter layout
changes directory names, not file contents. No upstream Python prompt is executed.

## Operator contract

The assets job runs `python -m pmmc.experiments.release_assets`. Pass the
normalized JSON request in `PMMC_RELEASE_REQUEST`. Mount the inputs volume at
`/pmmc-stage` **only for this writable job**, or configure its mount location
using `PMMC_BENCHMARK_STAGING_ROOT`. Other jobs mount the volume read-only at
`/pmmc-inputs`, with `PMMC_BENCHMARK_ASSET_ROOT=/pmmc-inputs`. Do not mount the
outputs or source checkout as the staging root. No Hugging Face token, provider
credentials, DB or GPU is needed to prepare benchmark assets.

Each selected benchmark publishes independently at `benchmarks/<name>`. The
dataset-specific lock serializes writers. Downloads are anonymous, streamed,
size- and SHA-256-checked; verified files are retained by hash across interrupted
runs. The MemGallery resolver checks the pinned revision's LFS object hashes
and the individually locked regular blobs against the complete tree digest
before downloading. MEMLENS uses only its original 32K JSON and image archive.
Candidate trees are checked against the lock over sorted
`[relative_path, byte_count, sha256]` triples. Complete local trees are checked
without any network access. Stderr shows aggregate progress; the last stdout
line is a `release-assets-v2` ready receipt with the plan hash, selected
source identities, and exact `file_counts`. The dependency-free
`pmmc.release_asset_receipt.validate_asset_receipt(receipt, plan)` rejects wrong
plans, missing/extra datasets, wrong counts and unlocked trees. A failed check
must not produce a ready receipt.

Use one filesystem for staging and publication. First install is an atomic
directory rename. Repair uses atomic directory exchange (Linux or macOS; the
filesystem must support it), retaining the displaced tree under
`.release-data-stage/<name>/candidate-*` for operator review. A damaged
download or failed exchange leaves the previously published tree untouched.
Interrupted unpublished candidates are not inputs; they can be removed by
the volume owner while no preparer is active. Allow space for the blob cache,
candidate tree, and a previous generation during repairs.

## Membership evidence

Cold downloads from the pinned sources above were independently taken through
the native FULL snapshot, original five-fold partition and single-benchmark
dev-all plan. Both runs matched all four digest gates (canonical content,
assignments, development IDs and paper IDs) in
`configs/release/original-split-v2.json`; the FULL snapshot IDs matched too.

| Benchmark | Locked files | FULL snapshot | Eligible | Selected dev | Published tree SHA-256 |
| --- | ---: | --- | ---: | ---: | --- |
| MEMLENS | 4,703 | `memlens-4bf865094312f9a4` | 699 | 140 | `3c9f8489f1f3238f1e45c66b73156d20adb2240b5305a8a5c204455376abf59a` |
| MemGallery | 1,515 | `memgallery-b6e0fbd79ef2b0ee` | 1,446 | 287 | `9f5d4e5a2256265f87606e1aad60f49d0944f76149fc5c6a941153d9cc98c50e` |

A one-benchmark partition has a different partition ID from the original
combined one: it contains only the requested rows. Certification compares each
selected dataset's exact sorted `(query_id, fold, role)` triples and both role
membership digests against that dataset's original reference. Matching counts
alone are insufficient. A new source revision or membership requires a reviewed
versioned lock and split, never an in-place substitution.

To repeat the anonymous cold check in a checkout with test dependencies:

```sh
PMMC_TEST_PUBLIC_COLD_DOWNLOAD=1 PYTHONPATH=src python -m pytest \
  tests/test_release_data.py -k explicit_cold_upstream_membership
```

This uses fresh temporary volumes for both downloads. For just one benchmark,
append `and memlens` or `and memgallery` to the quoted `-k` expression.

After planning, the models workload validates the complete immutable model
registry and lock. It provisions the HF entries used by the selected native job
inventory (including tokenizer-only dependencies), not every possible answer
backbone. Native provider identities are verified, not downloaded or invoked.
Use `PMMC_RELEASE_VARIANT` and `PMMC_RELEASE_STAGE` to provision a single
execution stage. Explicit prepare uses `PMMC_RELEASE_STAGE=prepare-all`: it
requires a verified pointer for every selected variant, and unions the stages
actually present in those inventories. A CPU-only analysis which needs a main
producer does not invent an analysis worker stage. GPU service calibration
uses the selected jobs' execution classes; a model needed only for token
counting does not authorize starting its GPU pool. The per-variant pointer
binds `selected` (development denominator), `gpu_stages`,
`local_gpu_required_by_stage`, `local_gpu_required`, model aliases, selection,
job inventory and normalized request hash. An empty service set does not
consult or measure the 27B pool.
