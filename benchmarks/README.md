# Local Benchmark Assets

PMMC currently supports two multimodal long-term-memory benchmarks through a
single canonical ingestion layer:

```text
benchmarks/
  memlens/
    data/
    images/
    prompts/judge/
  memgallery/
    data/dialogs/
    images/
    prompts/source/
```

Third-party data, images, and source prompts are intentionally ignored by Git.
Only these documentation files and empty directory placeholders are part of the
PMMC source distribution. Obtain benchmark payloads from their original sources
and confirm their licenses and citation requirements before redistribution.

PMMC preserves all source records for auditability. The default evaluation
policy `pmmc-aaai27-v1` excludes MEMLENS Answer Refusal and MemGallery AR/CD.
Excluded records can only be selected through an explicit debug override.

Validate local payloads with:

```bash
pmmc validate-data --dataset all --level fast
pmmc validate-data --dataset all --level full
```

Create deterministic, gold-isolated canonical snapshots with:

```bash
pmmc import-dataset --dataset all
```

Snapshots are written below `artifacts/benchmarks/`; run metadata and structured
progress events are written below `runs/`.
