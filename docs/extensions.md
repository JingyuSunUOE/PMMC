# Extending a released exploration suite

An arbitrary dataset name or model endpoint is not a validated experiment.
Extensions are reviewed source/configuration changes with a new protocol identity;
they do not reinterpret an existing run or permission to spend.

**Benchmark.** Implement `pmmc.benchmarks.base.BenchmarkAdapter`, whose interface
emits deterministic conversations and assets and declares expected counts. Add
the named dataset to the typed schema and `pmmc.benchmarks.registry.ADAPTER_TYPES`.
Provide a pinned source layout and terms, stable original identifiers, a
source-to-target mapping and verified fold/role membership. Keep query gold
and evaluation-only assets inaccessible to the compiler. Extend the public
plan catalog and tests only after independent cold preparation reproduces the
declared membership; do not reuse either built-in benchmark's denominator.

**Answer backbone.** Register a new `ModelRegistryEntry` in the release registry
with a new alias, exact provider/model identity, locked revision, modalities,
token counting/processor and output contract. Add that alias to the new
protocol's ordered plan catalog without modifying historical alias tuples.
For local inference bind a compatible single-node topology and measurement;
for a paid provider supply a verified pricing policy and request accounting.
Validate generation, image delivery, typed answer, grounding, unified scoring
and cost identity. An endpoint or marketing name alone is insufficient.

**Method or report.** Add the native operation and its dependency roots, explicit
source/fingerprint owners, a typed score/readout scope and denominator checks.
Do not compute a full matrix and then drop rows, manufacture evidence for
unselected methods, or count the same physical producer twice. Add one
standalone and one shared-producer regression before publishing that catalog.
