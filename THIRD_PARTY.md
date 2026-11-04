# Third-party materials

The Apache-2.0 grant in LICENSE is for PMMC's original contributions. This
source package does not redistribute third-party benchmark payloads, upstream
prompts, model weights, dependency wheels, container layers or provider data.
Identifiers, adapter code, file-layout descriptions and dependency references
are not grants of rights to the referenced material.

| Referenced material | In this source distribution | Recipient action |
| --- | --- | --- |
| MEMLENS and MemGallery | PMMC adapters only; no records, human annotations, benchmark prompts, captions, derived corpora or media | Obtain from the upstream authors under the applicable terms; check citation, access and redistribution conditions. |
| PMMC answer, grounding and baseline judge templates | Only the explicitly inventoried first-party files under `prompts/` and original inline PMMC logic | Do not move upstream prompts into these paths; changes require a new content review. |
| Models/tokenizers (including Qwen, SigLIP and API model identifiers) | References only; no cached snapshots, locks with private paths, weights or keys | Review the specific revision's terms, provider terms, access and budget separately. |
| Python dependencies | Declared in pyproject.toml; not vendored | Keep upstream licenses and notices if separately distributing dependency bundles. |
| vLLM/CUDA base, distro packages, PostgreSQL, pgvector, MinIO | Build recipes only; images/binaries are not included | Review all component terms, notices and base image provenance before redistributing an image. |

Unknown provenance is an exclusion, not permission. A source archive, a
dependency bundle, a trained model and a container image are different
deliverables; this inventory covers only the first.
