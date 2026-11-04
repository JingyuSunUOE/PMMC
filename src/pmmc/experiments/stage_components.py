"""Immutable Step 6D/Step 7 stage-component identities and compatibility checks."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.schemas import StageComponentManifest, StageComponentStatus
from pmmc.schemas.base import utc_now

STEP6D_ALEMBIC_REVISION = "20260726_0019"
STEP7_ALEMBIC_REVISION = "20260730_0023"

# The summary-grid fix changes only portable reports, not verified source
# exports. Reuse those source milestones, but never a completed review bundle,
# export verification, or final report under the old report contract.
_SOURCE_EXPORT_ONLY_FINGERPRINT_TRANSITIONS = frozenset({(
    "de75c185ba5e80cb9ead5c5ae752b018c82554dffb6067ed0c9652e939d09c39",
    "615e086cb6792d6c70a0845091c38806d002b2f31eddd5bd5936976be25120bc",
)})

# This transition changes only a mypy annotation on an existing diagnostics
# dictionary. It has no runtime or scientific effect, so completed text/MM
# model facts remain reusable. Any other source-fingerprint transition stays
# incompatible.
_TYPE_ONLY_SOURCE_FINGERPRINT_TRANSITIONS = frozenset(
    {
        (
            "program_text",
            "cb189786eecba3c6f014628d1d2dadffd332aead8f602f43d3d80699e8bf7537",
            "c24886030d31e324c18861e3ca7dcf1f7bc6b1f70b539cc44e5c1b33fd48d0af",
        ),
        (
            "program_multimodal",
            "cb189786eecba3c6f014628d1d2dadffd332aead8f602f43d3d80699e8bf7537",
            "c24886030d31e324c18861e3ca7dcf1f7bc6b1f70b539cc44e5c1b33fd48d0af",
        ),
    }
)

# These launcher-only transitions add corpus snapshot/checkpoint handling, safer
# diagnostics, stricter prefix-record verification, the internal fallback-suite
# contract, a one-time exact-round report upgrade, and run-scoped persistence IDs
# for main-pilot compilation units. The latter preserves the content identity and
# prepared model input while preventing prefix/main primary-key collisions. None
# of these transitions changes already persisted prefix Q/P/D facts.
_RECOVERY_ONLY_SOURCE_FINGERPRINT_TRANSITIONS = frozenset(
    {
        # Attribute final pilot rejection to export_verify and log read-only
        # matching counters. Prefix scientific inputs, facts and gates are intact.
        (
            "step7_prefix",
            "6ebebbb7fa232006c5a3c43136c74b81f7d92339b2a3727b2f0a89d980d66c19",
            "8ea8014c53a0009f75f3c515cd6e090622e8c44445880264d22d1dbf916f75b4",
        ),
        # Label review-card/query reuse progress and count verified Judge facts.
        # No prefix inputs or facts change; report compatibility is separate.
        (
            "step7_prefix",
            "be31a263efea81b652265a5d1141399bd00aa331cf58bf82ea652f4033842d9d",
            "6ebebbb7fa232006c5a3c43136c74b81f7d92339b2a3727b2f0a89d980d66c19",
        ),
        # Checkpoint restore can invalidate only read-only dataset exports while
        # retaining independently verified model facts. Terminal answer/Judge
        # replay is fail-closed and avoids model startup. Prefix science is intact.
        (
            "step7_prefix",
            "10a648bbfccabbbf8ec99da99ac0bab398214f860aa53e8ac57786a6d9aa3587",
            "be31a263efea81b652265a5d1141399bd00aa331cf58bf82ea652f4033842d9d",
        ),
        # Shared Judge failure accounting and dispatch-only progress metadata;
        # no prefix Q/P/D inputs or decisions change. Evaluation must be rerun.
        (
            "step7_prefix",
            "a44a1b16389c86612c0fcd9e5b317d7f1e482c7d5c4dfc86cae4efe9e205893a",
            "10a648bbfccabbbf8ec99da99ac0bab398214f860aa53e8ac57786a6d9aa3587",
        ),
        # RAG resume reporting restores its verified selection denominator and
        # counts terminal-job reuse. No embedding request, vector or rank changes;
        # answer completion and its repaired summary must still be rechecked.
        (
            "program_multimodal",
            "478b7a98f2de82d24620dff68056f156d9c3a3d09135e5d3b8a5168b2b806ee5",
            "8e49938dcf5eef9f8c0c24fb2ae8c57f09dfae0bf8b999ea540cdfc1cb5d6269",
        ),
        # RAG job replay ignores creation timestamps and verifies answer-only
        # retrieval receipts. Embedding requests/ranks are unchanged. Re-run
        # answer completion with its stricter inventory; do not bridge answers.
        (
            "program_multimodal",
            "8efa1be5a38f31dfa981f68d95baa38651001d20080c58785fa31423d5876f08",
            "478b7a98f2de82d24620dff68056f156d9c3a3d09135e5d3b8a5168b2b806ee5",
        ),
        # Pre-answer CPU rebinding uses the GPU child's resolved registry and
        # reports typed failures. No persisted prefix Q/P/D inputs/facts change.
        (
            "step7_prefix",
            "aeb4cfa633335b558dc86adc0925641ad1991c111de9039a5a4f0296ba4d676a",
            "a44a1b16389c86612c0fcd9e5b317d7f1e482c7d5c4dfc86cae4efe9e205893a",
        ),
        # Chained-import compiler audit resolution and pre-GPU validation only;
        # original Q/P/D programs, prompts and scientific IDs are unchanged.
        (
            "step7_prefix",
            "50a742e2d3b66da8d9eb1f689f830598621dd2723f0985b7aa8844545a5e1a13",
            "aeb4cfa633335b558dc86adc0925641ad1991c111de9039a5a4f0296ba4d676a",
        ),
        # Answer v2.1 changes only downstream prompting and child rebinding.
        # Existing text/MM vectors and prefix Q/P/D facts are identical. Never
        # bridge online matching, execution, answer or evaluation components.
        (
            "program_text",
            "aa1c59d820e08f15e7351431dfe7e29c1b83bf94410796c45f59da1bfeb5eabc",
            "d35d310c1c0b21946dd0e54b02bbd98a3685cdb0f126be146690b9b3d57d01d3",
        ),
        (
            "program_multimodal",
            "5efd0475a79cd73416cc2d26d49bf48a60feda495836eb0f5635801fe65b2317",
            "8efa1be5a38f31dfa981f68d95baa38651001d20080c58785fa31423d5876f08",
        ),
        (
            "step7_prefix",
            "bd10f1e6a70a7136e0df764ae8be56d12c7a43c85854cf3b3a6aa3a3be5c906a",
            "50a742e2d3b66da8d9eb1f689f830598621dd2723f0985b7aa8844545a5e1a13",
        ),
        # DDL-only query completion upgrade and a post-answer failure checkpoint.
        # No prefix Q/P/D inputs, programs, embeddings or decisions are changed.
        # Online answer completion is deliberately not bridged.
        (
            "step7_prefix",
            "432c29817c21e366ef7b5aaf90229777a92c6f6fe69f07a4910b4b65a176847e",
            "bd10f1e6a70a7136e0df764ae8be56d12c7a43c85854cf3b3a6aa3a3be5c906a",
        ),
        # Keep the source-launch fallback suite on import and verify retrieval
        # before chat startup. No prefix Q/P/D or embedding/retrieval changes.
        (
            "step7_prefix",
            "5b8a8457f8a8b9a1f4d7788a7e5a7a4da40c0634c1d23a8d93fa83d22600aef6",
            "432c29817c21e366ef7b5aaf90229777a92c6f6fe69f07a4910b4b65a176847e",
        ),
        (
            "program_multimodal",
            "a383c4336b76384e0266a2aeddbeb184371b88f8e7176879b232a876143fc956",
            "5efd0475a79cd73416cc2d26d49bf48a60feda495836eb0f5635801fe65b2317",
        ),
        # Preserve existing model-lock bytes on self-resume. No changes to the
        # prefix model identities, native Q/P/D inputs, or scientific facts.
        (
            "step7_prefix",
            "09645baf0c38260c9771f35448673ca09063d673f515bd384fc8b08fa2aa200b",
            "5b8a8457f8a8b9a1f4d7788a7e5a7a4da40c0634c1d23a8d93fa83d22600aef6",
        ),
        # Separate, fail-closed cross-launch import and receipt export. Existing
        # prefix Q/P/D facts and native inference requests are unchanged.
        (
            "step7_prefix",
            "49da092c9aebe68be43ea29ef6fb9b85cd94178cd594e4597b8008cb7af4492e",
            "09645baf0c38260c9771f35448673ca09063d673f515bd384fc8b08fa2aa200b",
        ),
        # Main-pilot CPU query-vector preflight and safe failure-phase reporting
        # only. Prefix Q/P/D inputs, execution and conformance are unchanged;
        # query/answer completion is intentionally not bridged.
        (
            "step7_prefix",
            "2c1a2deee12213414afb63864e90f3cbda58be359500a8adc6f47086d5f832c6",
            "49da092c9aebe68be43ea29ef6fb9b85cd94178cd594e4597b8008cb7af4492e",
        ),
        # Strict suite headers now reflect the already-selected v2 answer contract.
        # Exact legacy lookup preserves existing child IDs and their MM facts;
        # embedding/retrieval inputs and execution are unchanged. The downstream
        # answer family must revalidate and persist the explicit suite binding.
        (
            "program_multimodal",
            "511226d32d71255f15f023c361681ac539be92816752ae5b6abc4a2eee90ca48",
            "a383c4336b76384e0266a2aeddbeb184371b88f8e7176879b232a876143fc956",
        ),
        # Strict RAG answer completion/verification now admits validated independent
        # abstentions. Shared repository embedding paths are unchanged; never reuse
        # answer-stage completion under the old gate.
        (
            "program_text",
            "f3d0dfa4eab5dcefb5b67b0049921d1b535ba59e17294723ce2649c1dcd31ca5",
            "aa1c59d820e08f15e7351431dfe7e29c1b83bf94410796c45f59da1bfeb5eabc",
        ),
        (
            "program_multimodal",
            "90100f36dc3a35bb91ece3595403df62b0fb7abb5d4ad2c8a721a243ef3b86de",
            "511226d32d71255f15f023c361681ac539be92816752ae5b6abc4a2eee90ca48",
        ),
        # Only answer-error diagnostics changed in the shared RAG runner. No
        # embedding/retrieval logic changed. Guided answer schema and retry
        # changes belong to the answer family and must not be bridged.
        (
            "program_multimodal",
            "5116a257e6d5279cec9454ab99d31433365a1a41efa8250597924420dbd2ea2a",
            "90100f36dc3a35bb91ece3595403df62b0fb7abb5d4ad2c8a721a243ef3b86de",
        ),
        # Independent answer abstention changes only downstream answer/evaluation
        # contracts. Shared baseline helpers retain identical successful embedding
        # behavior, and the prefix stops at Q/P/D. Never bridge the answer/export
        # families: their terminal inventories and approval semantics changed.
        (
            "program_text",
            "5df7c18a17162405de6e2ba0cef97ce4497548a80666c0c37bb9022162c60750",
            "f3d0dfa4eab5dcefb5b67b0049921d1b535ba59e17294723ce2649c1dcd31ca5",
        ),
        (
            "program_multimodal",
            "c063126e123ca376158598dde975ba4c1511ea5fdf5c177e813ce801d47faaf1",
            "5116a257e6d5279cec9454ab99d31433365a1a41efa8250597924420dbd2ea2a",
        ),
        (
            "step7_prefix",
            "91ce505d91b9f146a300dee10eda196fe167e49b517f9ae5e9c60011513dd576",
            "2c1a2deee12213414afb63864e90f3cbda58be359500a8adc6f47086d5f832c6",
        ),
        # Answer failure diagnostics and pre-child shared-service accounting only.
        # Successful MM retrieval and prefix Q/P/D facts are unchanged. The answer
        # stage is deliberately not bridged; its failing tasks must be diagnosed.
        (
            "program_multimodal",
            "9765b7e01dd5cf6a28e32850311413fc07bc3377667df0d6d3e5ca6bb6ae4e77",
            "c063126e123ca376158598dde975ba4c1511ea5fdf5c177e813ce801d47faaf1",
        ),
        (
            "step7_prefix",
            "1cedc1d2578ecf5210daa15f35be55567e29a64a4d7fe812460f21f105588312",
            "91ce505d91b9f146a300dee10eda196fe167e49b517f9ae5e9c60011513dd576",
        ),
        # Defer the shared GPU session until CPU export checks pass on restored
        # Doubter runs; correct reused accounting and storage-restore labels. Prefix facts are
        # unchanged. Export verification itself is invalidated, not bridged.
        (
            "step7_prefix",
            "b0060aa8ef858d77a0d9c3678d62ca95fd6b209f4932e21f98c90ea2ff12852d",
            "1cedc1d2578ecf5210daa15f35be55567e29a64a4d7fe812460f21f105588312",
        ),
        # 2026-09-07: align the launcher preflight/component label with migration
        # 0027 and check catalogs before restore. Prefix Q/P/D facts are unchanged.
        (
            "step7_prefix",
            "1a6f8f1df03123790935c8e7443730d75632ccc5ba3850909ac804c178d10755",
            "b0060aa8ef858d77a0d9c3678d62ca95fd6b209f4932e21f98c90ea2ff12852d",
        ),
        # 2026-09-07: versioned coverage completion and terminal progress.
        # Only score/verification branches changed; frozen text/MM embeddings
        # and prefix Q/P/D science remain identical. Coverage itself is invalidated.
        (
            "program_multimodal",
            "b502d562dfb078ed7ab855a46d8579a141faea7911cafa02cdf62d379a791b0d",
            "9765b7e01dd5cf6a28e32850311413fc07bc3377667df0d6d3e5ca6bb6ae4e77",
        ),
        (
            "program_text",
            "532988c2902d71a197dd84fa3d4d5ae60feb4b1b497ec3adc4cbc43e92d1d79d",
            "5df7c18a17162405de6e2ba0cef97ce4497548a80666c0c37bb9022162c60750",
        ),
        (
            "step7_prefix",
            "04a3e174a6aee1bbd2f3fa8c867a7f54ee6c4eb6e13b9610540fc467236f271a",
            "1a6f8f1df03123790935c8e7443730d75632ccc5ba3850909ac804c178d10755",
        ),
        # 2026-09-07: scoped/bulk program reads, bounded DB work, safe failure
        # diagnostics, checkpoint preservation and accurate progress. No prompt,
        # strategy, evidence, decision or output schema changes; successful facts
        # are unchanged. Accept only these audited before/after hashes.
        (
            "compiler_program",
            "4dfa3384b0a9c3600d796716b6852d9fdeaa3efb2096f20c92e84d99c4952c4e",
            "0f3fc9541ea63eaf1bd82b14915df275790e4608ccdcc74850efe7d5368ee3d7",
        ),
        (
            "doubter",
            "4609f7e86bead27969087b2f8839536550843244b401c0a086394664c6534d81",
            "40686605bb6be0611e2c8b80dae5e899960d6f67e989834918c5f1283b2cb8bd",
        ),
        (
            "dry_run",
            "1432278753650f89e44eb7a4888b49d3c2427c2dafa7eb1ff63d521e824e3ad3",
            "b85e3329eac8a85bb7536e787338c90a91f0cf295ea42973d479bcbea3a4bd43",
        ),
        (
            "program_multimodal",
            "d19e52718e5681b768d85df90528df5ee9c37643d1fd05ee04cd73a881ceff73",
            "b502d562dfb078ed7ab855a46d8579a141faea7911cafa02cdf62d379a791b0d",
        ),
        (
            "program_text",
            "32dbd655b27003e9a80b00f07c0045bf93d363cfbfc2ede8353ec162c6dd86d9",
            "532988c2902d71a197dd84fa3d4d5ae60feb4b1b497ec3adc4cbc43e92d1d79d",
        ),
        (
            "step7_prefix",
            "2116afa15e637af5adb96de4a4f813de7be8f944b4ea9a8533c1726ce14fbbe8",
            "04a3e174a6aee1bbd2f3fa8c867a7f54ee6c4eb6e13b9610540fc467236f271a",
        ),
        (
            "verified_qbank",
            "019971cc1a94728c287c30b910b6f8e30acab9d5ff41fda3c2822938af89e2a6",
            "af5a29f16bd2d3db48def802ea7fdb4e630c7cdb499fa0740a2f2bec2b9aecf2",
        ),
        (
            "step7_prefix",
            "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
            "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
        ),
        (
            "step7_prefix",
            "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
            "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
        ),
        (
            "step7_prefix",
            "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
            "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
        ),
        (
            "step7_prefix",
            "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
            "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
        ),
        (
            "step7_prefix",
            "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
            "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
        ),
        (
            "step7_prefix",
            "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
            "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
        ),
        (
            "step7_prefix",
            "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
            "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
        ),
        (
            "step7_prefix",
            "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
            "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
        ),
        (
            "step7_prefix",
            "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
            "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
        ),
        (
            "step7_prefix",
            "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
            "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
        ),
        (
            "step7_prefix",
            "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
            "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
        ),
        (
            "step7_prefix",
            "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
            "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
        ),
        (
            "step7_prefix",
            "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
            "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
        ),
        (
            "step7_prefix",
            "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
            "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
        ),
        (
            "step7_prefix",
            "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
            "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
        ),
        # Doubter request bounding only affects image-rich main-pilot calls.
        # The prefix is an engineering preflight outside the scientific
        # denominator, so its already verified facts remain reusable.
        (
            "step7_prefix",
            "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
            "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
        ),
        (
            "step7_prefix",
            "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
            "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
        ),
        (
            "step7_prefix",
            "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
            "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
        ),
        (
            "step7_prefix",
            "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
            "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
        ),
        (
            "step7_prefix",
            "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
            "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
        ),
        (
            "step7_prefix",
            "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
            "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
        ),
        (
            "unit_prepare",
            "758b217a050595790146543aa3615eae50e00be1b913f653f023c3c92e79ed88",
            "c23ae119012fa22f1240b515dcc6b60b120fd8f2e1400aa9d0caff5d6392294d",
        ),
        # The exact Doubter token counter is created only inside the DOUBTER
        # branch of program_workflow. The shared orchestrator fingerprint changes,
        # but completed Questioner/Planner/index/dry-run facts are byte-for-byte
        # unaffected and remain valid recovery inputs.
        (
            "compiler_program",
            "9fe1cb394b62d4f25930f09ab901ca4a32e4a63b57f401fb0e0f851ba83abde1",
            "cdfeab76d87228b821dc4fcb4b2d74a65950357f874b716262184f8b7ae556a4",
        ),
        (
            "program_text",
            "1cdfd550adfaea24e479d86edd6b9570e73fd6c317b419f39525a23d34ac0796",
            "31a7d64f6c6ae63863e4400d8aa4c888fd530abf569ed4c7e839fe8b683c428c",
        ),
        (
            "program_multimodal",
            "1cdfd550adfaea24e479d86edd6b9570e73fd6c317b419f39525a23d34ac0796",
            "31a7d64f6c6ae63863e4400d8aa4c888fd530abf569ed4c7e839fe8b683c428c",
        ),
        (
            "dry_run",
            "ccc46fa61e09cecd6544760e82f566db7423c721e18f44341dcab00cf6682ad8",
            "0f2f52151d9a77ff4e6941003bdfd3243970ad7a3c1550558af16747ae975b21",
        ),
        # Finalization-only contract repair: invalid-schema Questioner output is
        # already terminally rejected and never enters the eligible denominator.
        # The second transition adds identifier-free verifier diagnostics. Neither
        # changes persisted compiler, embedding, dry-run, or Doubter facts.
        (
            "compiler_program",
            "cdfeab76d87228b821dc4fcb4b2d74a65950357f874b716262184f8b7ae556a4",
            "379c2c1bb49ce7e281c0b8baefff47be591089ecd1e3bc80215e0b47d522de17",
        ),
        # Legacy Step 7 checkpoints named the shared compiler boundary
        # questioner:model-stage and fingerprinted the launcher harness. The
        # generated Q/P facts use the exact compiler-program sources above.
        (
            "compiler_program",
            "3b5bc8993792d1ff28becf72f5397142ff0f03cbc40d518700b6d796d02f7fa5",
            "379c2c1bb49ce7e281c0b8baefff47be591089ecd1e3bc80215e0b47d522de17",
        ),
        (
            "dry_run",
            "0f2f52151d9a77ff4e6941003bdfd3243970ad7a3c1550558af16747ae975b21",
            "c9a8b68ac7246028b32249bfca5946281eb29f79e4f7874cc0ed3a0d4b2684e2",
        ),
        # planner_verify calls the database compiler verifier and does not read
        # any offline export implementation. This explicit ownership repair
        # preserves checkpoint 000049 while keeping verified-QBank source
        # changes incompatible.
        (
            "verified_qbank",
            "f6f1c397ac13da72e032b6d758772b54df0d967cc2b2c1bb442dfdef42073e7a",
            "1d63e5931c657c6e2e4b0f94c111c82dab29f4444133af5997c9453941ca76ef",
        ),
        # Checkpoint 000036 already contains a verified immutable fallback index.
        # This recovery-only change starts a rehydrated child run from CREATED
        # before retrieval; it does not alter the index or its embedding facts.
        (
            "program_multimodal",
            "1799e92fc5a274e7f6ac6df02afff66b18b84c23c0486a1b6a883dbeaa1caaa3",
            "ece3ac17b460c6429a522e4b15f57621c4306fefc55df3937d15a7e8acd171dc",
        ),
        (
            "program_multimodal",
            "83715278c05b7bff341c3f9de046606186d70f2d68a809d4989f633499a28414",
            "ece3ac17b460c6429a522e4b15f57621c4306fefc55df3937d15a7e8acd171dc",
        ),
        # Preserve the previously audited prefix transition as a historical
        # identity. Compatibility is intentionally explicit rather than
        # transitively accepting arbitrary intermediate fingerprints.
        *(
            (
                "step7_prefix",
                recorded,
                "cf0e050218193e8addf7b72555642b38d8b075d27ade27adfec2e015cdc2a7db",
            )
            for recorded in (
                "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
                "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
                "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
                "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
                "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
                "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
                "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
            )
        ),
        # Prefix facts are outside the scientific denominator and already passed
        # their real-Q/P/D engineering gate. The latest launcher-only change adds
        # a post-index recovery checkpoint and does not alter prefix Q/P/D facts.
        # Main-pilot Doubter facts are not covered by this family transition.
        *(
            (
                "step7_prefix",
                recorded,
                "f314aaa5268cd292b08af8cd8b1cdaab487fe69465d8c7a8beec8d9f5d7c5046",
            )
            for recorded in (
                "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
                "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
                "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
                "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
                "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
                "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
                "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
                "cf0e050218193e8addf7b72555642b38d8b075d27ade27adfec2e015cdc2a7db",
            )
        ),
        # Skipping an already restored fallback-index milestone only avoids a
        # duplicate checkpoint archive. It does not alter the verified prefix
        # compiler, dry-run, Doubter, or routing facts.
        *(
            (
                "step7_prefix",
                recorded,
                "5df24ddf915b5b593bc4d4c089818c18a4ba621caac2396a93a6f7357fe84c48",
            )
            for recorded in (
                "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
                "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
                "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
                "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
                "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
                "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
                "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
                "cf0e050218193e8addf7b72555642b38d8b075d27ade27adfec2e015cdc2a7db",
                "f314aaa5268cd292b08af8cd8b1cdaab487fe69465d8c7a8beec8d9f5d7c5046",
            )
        ),
        # The modality-aware online-query fix changes only the post-coverage
        # runtime projection/import path. It does not reinterpret the already
        # persisted QBank, memory-index, or fallback embedding facts.
        (
            "program_text",
            "31a7d64f6c6ae63863e4400d8aa4c888fd530abf569ed4c7e839fe8b683c428c",
            "e55f6ec02c0652f4edaccaca15d44300a51a3d17969b0d31a9414eb53ceef382",
        ),
        (
            "program_multimodal",
            "ece3ac17b460c6429a522e4b15f57621c4306fefc55df3937d15a7e8acd171dc",
            "ae9de490ae7d9b7e6cd5c51ebaf2e93ef4b6c2a330e3196ccb65f0223e2a92d3",
        ),
        # The accompanying launcher edit only recognizes migration 0022 during
        # preflight. Prefix Q/P/D facts remain outside the scientific denominator.
        *(
            (
                "step7_prefix",
                recorded,
                "493403e1999630fd84f9a43fedae5f9c74ab7309426cf7a80343a9651eb78fa5",
            )
            for recorded in (
                "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
                "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
                "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
                "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
                "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
                "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
                "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
                "cf0e050218193e8addf7b72555642b38d8b075d27ade27adfec2e015cdc2a7db",
                "f314aaa5268cd292b08af8cd8b1cdaab487fe69465d8c7a8beec8d9f5d7c5046",
                "5df24ddf915b5b593bc4d4c089818c18a4ba621caac2396a93a6f7357fe84c48",
            )
        ),
        # Query-stage idempotency diagnostics and pre-answer checkpoint
        # coalescing only affect work after the verified prefix. The prefix Q/P/D
        # facts and their scientific denominator remain byte-for-byte reusable.
        *(
            (
                "step7_prefix",
                recorded,
                "4e9c2aead6366e3c5f7c616d7dace04fd5ddec4bca30ecf20d469d71fdbdb2bb",
            )
            for recorded in (
                "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
                "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
                "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
                "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
                "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
                "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
                "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
                "cf0e050218193e8addf7b72555642b38d8b075d27ade27adfec2e015cdc2a7db",
                "f314aaa5268cd292b08af8cd8b1cdaab487fe69465d8c7a8beec8d9f5d7c5046",
                "5df24ddf915b5b593bc4d4c089818c18a4ba621caac2396a93a6f7357fe84c48",
                "493403e1999630fd84f9a43fedae5f9c74ab7309426cf7a80343a9651eb78fa5",
            )
        ),
        # Migration 0023 and the accompanying launcher preflight change only
        # evaluator context lineage. Prefix Q/P/D and routing facts are unchanged.
        *(
            (
                "step7_prefix",
                recorded,
                "1aedba15fdc5c152f958396908cd41ab5745092a996465585e9ad0803b832588",
            )
            for recorded in (
                "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
                "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
                "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
                "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
                "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
                "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
                "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
                "cf0e050218193e8addf7b72555642b38d8b075d27ade27adfec2e015cdc2a7db",
                "f314aaa5268cd292b08af8cd8b1cdaab487fe69465d8c7a8beec8d9f5d7c5046",
                "5df24ddf915b5b593bc4d4c089818c18a4ba621caac2396a93a6f7357fe84c48",
                "493403e1999630fd84f9a43fedae5f9c74ab7309426cf7a80343a9651eb78fa5",
                "4e9c2aead6366e3c5f7c616d7dace04fd5ddec4bca30ecf20d469d71fdbdb2bb",
            )
        ),
        # Cross-process Query Executor replay now preserves first-insert audit
        # fields, and the Answer stage checks terminal facts before starting its
        # model pool. Both changes are downstream of the verified prefix, so the
        # already persisted prefix Q/P/D and routing facts remain reusable.
        *(
            (
                "step7_prefix",
                recorded,
                "fb7e3ab4c001444e048c0de83773ca10b69def2984ebf2df41fc4ff971a42988",
            )
            for recorded in (
                "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
                "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
                "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
                "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
                "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
                "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
                "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
                "cf0e050218193e8addf7b72555642b38d8b075d27ade27adfec2e015cdc2a7db",
                "f314aaa5268cd292b08af8cd8b1cdaab487fe69465d8c7a8beec8d9f5d7c5046",
                "5df24ddf915b5b593bc4d4c089818c18a4ba621caac2396a93a6f7357fe84c48",
                "493403e1999630fd84f9a43fedae5f9c74ab7309426cf7a80343a9651eb78fa5",
                "4e9c2aead6366e3c5f7c616d7dace04fd5ddec4bca30ecf20d469d71fdbdb2bb",
                "1aedba15fdc5c152f958396908cd41ab5745092a996465585e9ad0803b832588",
                "7cb4dfb8f481469d270215ac40df3bab1b8b793ff77d13c00e33f02cf490d36d",
            )
        ),
        # The final answer-resume audit uses the shared typed integer reader for
        # fallback denominators. This is type-safe validation downstream of the
        # verified prefix and does not alter persisted prefix scientific facts.
        *(
            (
                "step7_prefix",
                recorded,
                "ed8ed6233f29ea4215e0067d68837aba83e5ad07ebfce1ac4a050c2979fcc326",
            )
            for recorded in (
                "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
                "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
                "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
                "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
                "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
                "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
                "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
                "cf0e050218193e8addf7b72555642b38d8b075d27ade27adfec2e015cdc2a7db",
                "f314aaa5268cd292b08af8cd8b1cdaab487fe69465d8c7a8beec8d9f5d7c5046",
                "5df24ddf915b5b593bc4d4c089818c18a4ba621caac2396a93a6f7357fe84c48",
                "493403e1999630fd84f9a43fedae5f9c74ab7309426cf7a80343a9651eb78fa5",
                "4e9c2aead6366e3c5f7c616d7dace04fd5ddec4bca30ecf20d469d71fdbdb2bb",
                "1aedba15fdc5c152f958396908cd41ab5745092a996465585e9ad0803b832588",
                "7cb4dfb8f481469d270215ac40df3bab1b8b793ff77d13c00e33f02cf490d36d",
                "fb7e3ab4c001444e048c0de83773ca10b69def2984ebf2df41fc4ff971a42988",
            )
        ),
        # Semantic-parent preflight and compact conformance output execute only
        # after the verified prefix. They do not alter prefix Questioner,
        # Planner, dry-run, Doubter, decision, or routing facts.
        *(
            (
                "step7_prefix",
                recorded,
                "601e140c02205217c3d29d94be410283293efd0e5067fa33b1491d475a07f6eb",
            )
            for recorded in (
                "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
                "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
                "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
                "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
                "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
                "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
                "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
                "cf0e050218193e8addf7b72555642b38d8b075d27ade27adfec2e015cdc2a7db",
                "f314aaa5268cd292b08af8cd8b1cdaab487fe69465d8c7a8beec8d9f5d7c5046",
                "5df24ddf915b5b593bc4d4c089818c18a4ba621caac2396a93a6f7357fe84c48",
                "493403e1999630fd84f9a43fedae5f9c74ab7309426cf7a80343a9651eb78fa5",
                "4e9c2aead6366e3c5f7c616d7dace04fd5ddec4bca30ecf20d469d71fdbdb2bb",
                "1aedba15fdc5c152f958396908cd41ab5745092a996465585e9ad0803b832588",
                "7cb4dfb8f481469d270215ac40df3bab1b8b793ff77d13c00e33f02cf490d36d",
                "fb7e3ab4c001444e048c0de83773ca10b69def2984ebf2df41fc4ff971a42988",
                "ed8ed6233f29ea4215e0067d68837aba83e5ad07ebfce1ac4a050c2979fcc326",
                "a43a5f19c8877a72b640f0862127a9e22894ce9fa9bb448d96be90a671c31a61",
            )
        ),
        # Restore candidate scanning now verifies only checkpoint metadata until
        # a source/schema-compatible candidate is selected. It also deduplicates
        # compatibility logs and reports attempt-local progress. This changes no
        # persisted prefix Q/P/D, dry-run, decision, or routing fact.
        *(
            (
                "step7_prefix",
                recorded,
                "6269f3ba23c69d7f2fefb187edcd8a33fc6c1355b1ac1b06e3ef8aeb787ba8a7",
            )
            for recorded in (
                "ddecda873edfe7234561dca20a58a4e68c3a2331f3b27965599179bdb6a9349d",
                "70b7af6fcca3a4d7e757b10e198954c67aba80d9c8c04011e604a103cfb4d8ff",
                "c8ab5ad4a832024513a3906edb11de576f7c0182c7947e227032c375212cfb52",
                "dcc02e6281c1622d386d44f194d2a33f3b74d572f4e6e9c6547e9ccde600a3a1",
                "bfc1b2cd3ae370965a33702f50afce3a10ee3ee9ede3a0cb46881b7328204feb",
                "ff78ad912ccf88a8bc9389f437cb8a6f8bff1dd581e483d5a149287060e223c1",
                "20e45719ad4464777a3244190400a84301b8870bdd86c986923c0cfc3d13ccbc",
                "cf0e050218193e8addf7b72555642b38d8b075d27ade27adfec2e015cdc2a7db",
                "f314aaa5268cd292b08af8cd8b1cdaab487fe69465d8c7a8beec8d9f5d7c5046",
                "5df24ddf915b5b593bc4d4c089818c18a4ba621caac2396a93a6f7357fe84c48",
                "493403e1999630fd84f9a43fedae5f9c74ab7309426cf7a80343a9651eb78fa5",
                "4e9c2aead6366e3c5f7c616d7dace04fd5ddec4bca30ecf20d469d71fdbdb2bb",
                "1aedba15fdc5c152f958396908cd41ab5745092a996465585e9ad0803b832588",
                "7cb4dfb8f481469d270215ac40df3bab1b8b793ff77d13c00e33f02cf490d36d",
                "fb7e3ab4c001444e048c0de83773ca10b69def2984ebf2df41fc4ff971a42988",
                "ed8ed6233f29ea4215e0067d68837aba83e5ad07ebfce1ac4a050c2979fcc326",
                "a43a5f19c8877a72b640f0862127a9e22894ce9fa9bb448d96be90a671c31a61",
                "601e140c02205217c3d29d94be410283293efd0e5067fa33b1491d475a07f6eb",
            )
        ),
        # Incremental checkpoint storage, checkpoint coalescing, safe revision
        # parser diagnostics, and post-query aggregate logging do not change the
        # already verified 2+2 prefix denominator or its persisted Q/P/D facts.
        # Main-pilot Doubter/query/evaluation components retain their own source
        # fingerprints and are intentionally invalidated where behavior changed.
        (
            "step7_prefix",
            "6269f3ba23c69d7f2fefb187edcd8a33fc6c1355b1ac1b06e3ef8aeb787ba8a7",
            "c3cb6bec06af21613031ede26331b40e9a4efe9ad44a828c0b451d30b70a6d0d",
        ),
        # Preserve prefix facts across follow-up launcher-only recovery changes.
        (
            "step7_prefix",
            "c3cb6bec06af21613031ede26331b40e9a4efe9ad44a828c0b451d30b70a6d0d",
            "709a1a91ccf2a56136e07d3c8a339a89babc75e844dbc5879695da14d9bf266a",
        ),
        # Query-runtime exports use their dedicated terminal state as the
        # completion authority. This does not alter persisted prefix facts.
        (
            "step7_prefix",
            "709a1a91ccf2a56136e07d3c8a339a89babc75e844dbc5879695da14d9bf266a",
            "e4093bae7054c7fa621f0ece18f360797310ef8762def92f3829ba43863ece92",
        ),
        # Importing the completion authority reader into finalization changes
        # the launcher fingerprint but cannot change persisted prefix facts.
        (
            "step7_prefix",
            "e4093bae7054c7fa621f0ece18f360797310ef8762def92f3829ba43863ece92",
            "1ccc856a47b2485cfb4da2dfd56f067da69a7527ae0e710cdb2e50095d75a82c",
        ),
        # Persist automatic-quality source exports before review-bundle work so
        # a packaging-only failure cannot force those exports to run again.
        # This changes checkpoint placement only, not prefix Q/P/D facts.
        (
            "step7_prefix",
            "1ccc856a47b2485cfb4da2dfd56f067da69a7527ae0e710cdb2e50095d75a82c",
            "aa81c7f456044520b014951352884052027cecb9576955e8acda614a4db6730e",
        ),
        # Reuse the already verified automatic-quality source manifest after a
        # packaging-only failure. This launcher change avoids repeated exports;
        # it cannot alter persisted prefix Questioner/Planner/Doubter facts.
        (
            "step7_prefix",
            "aa81c7f456044520b014951352884052027cecb9576955e8acda614a4db6730e",
            "cc2f5b1825f5ac8582199fe185f64712112ea5a07ffafb139787e53f899f9821",
        ),
        # A completed Judge boundary already contains evaluator model calls and
        # semantic assessments. This launcher-only recovery path verifies those
        # terminal facts before skipping model startup; it does not alter prefix
        # or main-pilot scientific facts.
        (
            "step7_prefix",
            "cc2f5b1825f5ac8582199fe185f64712112ea5a07ffafb139787e53f899f9821",
            "a9cf3b250ef93e4aae71f6bce1fcc409036e556573d4222df0fe93baba19f912",
        ),
        # The launcher now forwards already materialized review-card counts to
        # the progress reporter. Prefix Q/P/D calls and persisted facts are not
        # touched, so the verified prefix checkpoint remains reusable.
        (
            "step7_prefix",
            "a9cf3b250ef93e4aae71f6bce1fcc409036e556573d4222df0fe93baba19f912",
            "cbf0b65df076dde47c88f103e0eba1fa34847a0ea9b72b923d861e750d5fe11c",
        ),
        # Requiring the exported query_executor_runs row to be terminal only
        # strengthens bundle verification; existing dataset exports are intact.
        (
            "export",
            "d3b9eadae98d8782ec2bba7f0ce21ea0ba0b742370ed6d6b209b0f68f6605ccb",
            "5336d370bef128cd421edc333f06ea507bb47146335bea61a09213410083ebf8",
        ),
        # The semantic assessment was already persisted under the tri-state
        # schema. This export-only repair makes the offline verifier honor
        # `not_applicable` exactly like that schema and adds safe diagnostics;
        # no Questioner, Planner, Doubter, executor, or Judge fact changes.
        (
            "export",
            "5336d370bef128cd421edc333f06ea507bb47146335bea61a09213410083ebf8",
            "5eec7f93a8baaea4a0ae84ee9668fe23beca29e004fbea084c60f9a9b6138783",
        ),
        # Rejected and pending round-projection members legitimately persist a
        # null passed_at_round. The review-bundle summary now preserves those
        # facts instead of coercing null to int, and emits only safe dataset /
        # substage diagnostics. Persisted scientific facts are unchanged.
        (
            "export",
            "5eec7f93a8baaea4a0ae84ee9668fe23beca29e004fbea084c60f9a9b6138783",
            "6022ab62b5d2e925b556c276f36d4ace5267721f4620c52d19ca047355926029",
        ),
        # Atomic bundle construction verifies a PID-suffixed staging directory
        # before rename. Permit only that internal name during the private
        # check, while forcing the portable tar root to the stable pilot ID.
        # This is export-only and reuses all compiler/evaluator source facts.
        (
            "export",
            "6022ab62b5d2e925b556c276f36d4ace5267721f4620c52d19ca047355926029",
            "0862f57991346fba465801bf93408ec9893fa41198bc7e6beccd18e19ff7ee6b",
        ),
        # Distinguish the complete candidate-fact denominator from the bounded
        # review-card denominator in safe progress logs. Bundle bytes and all
        # persisted scientific facts remain unchanged.
        (
            "export",
            "0862f57991346fba465801bf93408ec9893fa41198bc7e6beccd18e19ff7ee6b",
            "c1eeeaa9f088895687ea57458331e4e7be703e89ec2134237a469e8153f5948f",
        ),
        # Transfer-only hardening projects machine-local locators out of raw
        # database facts and emits safe verifier check codes. Persisted compiler,
        # evaluator, and query-runtime facts remain byte-for-byte unchanged.
        (
            "export",
            "c1eeeaa9f088895687ea57458331e4e7be703e89ec2134237a469e8153f5948f",
            "4fb593972bfc086e75902d0c8b72e7abef5455afa0f82f812e9c69e4c1bbcef5",
        ),
        (
            "export",
            "4fb593972bfc086e75902d0c8b72e7abef5455afa0f82f812e9c69e4c1bbcef5",
            "6db442a24165e520972d638c860daf99f880b3771e6d4eed01eb7f5c5b31f8ea",
        ),
        (
            "export",
            "6db442a24165e520972d638c860daf99f880b3771e6d4eed01eb7f5c5b31f8ea",
            "392d320b082a13bbb9016f9e266456e93a43ee1e5cdd4fb91828f483b614b999",
        ),
        # Resolve query-card evaluation lineage through the persisted selected
        # answer and run the content-free card contract before artifact
        # projection. This changes only the private audit view and diagnostics;
        # compiler, matcher, executor, answer, and evaluator facts are intact.
        (
            "export",
            "392d320b082a13bbb9016f9e266456e93a43ee1e5cdd4fb91828f483b614b999",
            "c0bb035f4886dc5328e0deaebb707061ceb427624241add0f78d7b3bcddebb13",
        ),
        # The previous export revision already rebuilt evaluator answer scope and
        # produced complete compiler exports. This transition only makes the
        # verified-QBank report expose the two already persisted memory
        # collections and improves final-verifier diagnostics; export bytes and
        # all scientific facts remain unchanged.
        (
            "export",
            "6dddb3e34190ec012f33e18031a8227621aeb79cb9c7ff1ee9d63261b602d3bb",
            "ace0edf80b0a497b3d6feb8504d37faa660f13b63f7e0e4686fee0bbcb20ed00",
        ),
        # Reuse an already verified Step 7 bundle when its immutable scientific
        # source identity is unchanged. The new progress phase sink and bounded
        # diagnostics are execution telemetry only; no compiler, matcher,
        # executor, answer, or evaluator facts change.
        (
            "export",
            "ace0edf80b0a497b3d6feb8504d37faa660f13b63f7e0e4686fee0bbcb20ed00",
            "868502a740788e5036370a75dffea0ca5927d350567bc622a76baa44e2654e0a",
        ),
        # Report the already materialized 60 candidate and 12 query cards per
        # dataset as real review-bundle progress. This is observability only;
        # bundle sources, projections, and scientific outputs are unchanged.
        (
            "export",
            "868502a740788e5036370a75dffea0ca5927d350567bc622a76baa44e2654e0a",
            "a9439be9ff63adb56fc9923d5bef84ff24eecbdbeb5fb5305f054e6fcfa7dcea",
        ),
    }
)

_STAGE_SOURCE_FILES: dict[str, tuple[str, ...]] = {
    "unit_prepare": (
        "src/pmmc/compiler/preparation.py",
        "src/pmmc/compiler/units.py",
        "src/pmmc/compiler/opportunities.py",
        "src/pmmc/compiler/questioner_config.py",
        "src/pmmc/infrastructure/database/questioner_fts.py",
        "configs/compiler/questioner-v3.3.yaml",
        "configs/compiler/questioner-v3.4.yaml",
        "configs/compiler/questioner-v3.5.yaml",
        "configs/compiler/pmmc-v4.yaml",
        "configs/compiler/pmmc-v5.yaml",
        "configs/compiler/pmmc-v6.yaml",
    ),
    "compiler_program": (
        "alembic/versions/20260725_0018_step6d_doubter_verified_qbank.py",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
        "src/pmmc/compiler/opportunities.py",
        "src/pmmc/compiler/questioner.py",
        "src/pmmc/compiler/questioner_runner.py",
        "src/pmmc/compiler/questioner_prompt.py",
        "src/pmmc/compiler/questioner_config.py",
        "src/pmmc/compiler/planner.py",
        "src/pmmc/compiler/planner_runner.py",
        "src/pmmc/compiler/planner_prompt.py",
        "src/pmmc/compiler/planner_config.py",
        "src/pmmc/compiler/program_workflow.py",
        "src/pmmc/compiler/workflow.py",
        "src/pmmc/infrastructure/database/planner_repositories.py",
        "src/pmmc/infrastructure/database/constants.py",
        "src/pmmc/infrastructure/database/questioner_models.py",
        "src/pmmc/infrastructure/database/questioner_repositories.py",
        "src/pmmc/models/gateways.py",
        "src/pmmc/schemas/questioner.py",
        "src/pmmc/schemas/planner.py",
        "src/pmmc/smoke/step6d.py",
        "configs/compiler/questioner-v3.3.yaml",
        "configs/compiler/questioner-v3.4.yaml",
        "configs/compiler/questioner-v3.5.yaml",
        "configs/compiler/planner-v1.2.yaml",
        "configs/compiler/planner-v1.3.yaml",
        "configs/compiler/planner-v1.4.yaml",
        "configs/compiler/pmmc-v4.yaml",
        "configs/compiler/pmmc-v5.yaml",
        "configs/compiler/pmmc-v6.yaml",
    ),
    "program_text": (
        "alembic/versions/20260726_0019_step6d_query_embedding_lineage.py",
        "src/pmmc/compiler/memory_index_runner.py",
        "src/pmmc/compiler/program_workflow.py",
        "src/pmmc/evaluation/qbank_protocols.py",
        "src/pmmc/evaluation/qbank_workflow.py",
        "src/pmmc/infrastructure/database/experiment_repositories.py",
        "src/pmmc/infrastructure/database/protocols.py",
        "src/pmmc/runtime/query_embeddings.py",
        "alembic/versions/20260727_0020_step7_online_query_executor.py",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
        "configs/runtime/query-executor-v2.yaml",
    ),
    "program_multimodal": (
        "src/pmmc/baselines/suite_contract.py",
        "src/pmmc/infrastructure/database/evaluation_repositories.py",
        "alembic/versions/20260726_0019_step6d_query_embedding_lineage.py",
        "configs/baselines/rag.yaml",
        "src/pmmc/baselines/rag_embedding.py",
        "src/pmmc/baselines/rag_retrieval.py",
        "src/pmmc/baselines/rag_runner.py",
        "src/pmmc/baselines/rag_workflow.py",
        "src/pmmc/baselines/suite.py",
        "src/pmmc/compiler/memory_index_runner.py",
        "src/pmmc/compiler/program_workflow.py",
        "src/pmmc/evaluation/qbank_protocols.py",
        "src/pmmc/evaluation/qbank_workflow.py",
        "src/pmmc/infrastructure/experiment_runs.py",
        "src/pmmc/infrastructure/database/experiment_repositories.py",
        "src/pmmc/infrastructure/database/protocols.py",
        "src/pmmc/runtime/query_embeddings.py",
        "alembic/versions/20260727_0020_step7_online_query_executor.py",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
        "configs/runtime/query-executor-v2.yaml",
    ),
    "dry_run": (
        "src/pmmc/compiler/program_runner.py",
        "src/pmmc/compiler/dry_run.py",
        "src/pmmc/compiler/r0_gate.py",
        "src/pmmc/compiler/program_workflow.py",
        "src/pmmc/infrastructure/database/compiler_repositories.py",
        "src/pmmc/infrastructure/database/doubter_repositories.py",
        "src/pmmc/memory/operators.py",
        "src/pmmc/schemas/doubter.py",
        "src/pmmc/evidence_lineage.py",
        "configs/compiler/pmmc-v4.yaml",
        "configs/compiler/pmmc-v5.yaml",
        "configs/compiler/pmmc-v6.yaml",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
    ),
    "doubter": (
        "src/pmmc/compiler/doubter_runner.py",
        "src/pmmc/compiler/doubter.py",
        "src/pmmc/compiler/doubter_config.py",
        "src/pmmc/compiler/doubter_graph.py",
        "src/pmmc/compiler/doubter_prompt.py",
        "src/pmmc/compiler/r0_gate.py",
        "src/pmmc/compiler/planner.py",
        "src/pmmc/compiler/planner_config.py",
        "src/pmmc/compiler/planner_prompt.py",
        "src/pmmc/compiler/program_runner.py",
        "src/pmmc/compiler/program_workflow.py",
        "src/pmmc/compiler/dry_run.py",
        "src/pmmc/infrastructure/database/doubter_repositories.py",
        "src/pmmc/infrastructure/database/planner_repositories.py",
        "src/pmmc/models/gateways.py",
        "src/pmmc/schemas/doubter.py",
        "src/pmmc/schemas/planner.py",
        "configs/compiler/doubter-v1.yaml",
        "configs/compiler/doubter-v1.1.yaml",
        "configs/compiler/doubter-v1.2.yaml",
        "configs/compiler/planner-v1.2.yaml",
        "configs/compiler/planner-v1.3.yaml",
        "configs/compiler/planner-v1.4.yaml",
        "configs/compiler/pmmc-v4.yaml",
        "configs/compiler/pmmc-v5.yaml",
        "configs/compiler/pmmc-v6.yaml",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
    ),
    "verified_qbank": (
        "alembic/versions/20260907_0026_step7b_completion_verifier_native.py",
        "src/pmmc/infrastructure/database/completion_preflight.py",
        "src/pmmc/compiler/verified_qbank.py",
        "src/pmmc/compiler/r0_gate.py",
        "src/pmmc/infrastructure/database/compiler_repositories.py",
        "src/pmmc/infrastructure/database/doubter_repositories.py",
        "src/pmmc/schemas/doubter.py",
        "alembic/versions/20260727_0020_step7_online_query_executor.py",
        "alembic/versions/20260727_0021_step7_round_projection_routing_fix.py",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
    ),
    "qbank_coverage": (
        "alembic/versions/20260907_0027_qbank_coverage_protocol_completion.py",
        "src/pmmc/evaluation/qbank_coverage.py",
        "src/pmmc/evaluation/qbank_protocols.py",
        "src/pmmc/evaluation/qbank_workflow.py",
        "src/pmmc/evidence_lineage.py",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
    ),
    "qbank_semantic": (
        "src/pmmc/evaluation/qbank_protocols.py",
        "src/pmmc/evaluation/qbank_semantic.py",
        "src/pmmc/evaluation/qbank_semantic_workflow.py",
        "src/pmmc/models/gateways.py",
        "src/pmmc/schemas/semantic.py",
        "configs/compiler/qbank-semantic-match-v2.yaml",
        "configs/compiler/qbank-semantic-match-v3.yaml",
        "configs/compiler/qbank-semantic-match-v4.yaml",
    ),
    "step7_prefix": (
        "src/pmmc/experiments/launcher.py",
        "src/pmmc/compiler/questioner_runner.py",
        "src/pmmc/compiler/planner_runner.py",
        "src/pmmc/compiler/doubter_runner.py",
        "src/pmmc/compiler/r0_gate.py",
        "src/pmmc/compiler/verified_qbank.py",
        "src/pmmc/models/gateways.py",
        "configs/compiler/questioner-v3.4.yaml",
        "configs/compiler/questioner-v3.5.yaml",
        "configs/compiler/planner-v1.3.yaml",
        "configs/compiler/planner-v1.4.yaml",
        "configs/compiler/doubter-v1.1.yaml",
        "configs/compiler/doubter-v1.2.yaml",
        "configs/compiler/pmmc-v5.yaml",
        "configs/compiler/pmmc-v6.yaml",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
    ),
    "query_match": (
        "src/pmmc/schemas/query_executor.py",
        "alembic/versions/20260727_0020_step7_online_query_executor.py",
        "alembic/versions/20260727_0021_step7_round_projection_routing_fix.py",
        "alembic/versions/20260729_0022_step7_modality_aware_query_embeddings.py",
        "src/pmmc/runtime/qbank_matcher.py",
        "src/pmmc/runtime/query_backend.py",
        "src/pmmc/runtime/query_embeddings.py",
        "src/pmmc/runtime/query_executor_config.py",
        "src/pmmc/runtime/query_workflow.py",
        "src/pmmc/infrastructure/database/query_executor_repositories.py",
        "src/pmmc/models/gateways.py",
        "configs/runtime/query-executor-v1.yaml",
        "configs/runtime/query-executor-v2.yaml",
        "src/pmmc/runtime/qbank_reranker.py",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
    ),
    "query_execute": (
        "src/pmmc/schemas/query_executor.py",
        "alembic/versions/20260727_0020_step7_online_query_executor.py",
        "alembic/versions/20260727_0021_step7_round_projection_routing_fix.py",
        "alembic/versions/20260729_0022_step7_modality_aware_query_embeddings.py",
        "src/pmmc/runtime/query_executor.py",
        "src/pmmc/runtime/query_executor_config.py",
        "src/pmmc/runtime/query_runner.py",
        "src/pmmc/runtime/query_backend.py",
        "src/pmmc/runtime/evidence_sufficiency.py",
        "src/pmmc/memory/operators.py",
        "src/pmmc/infrastructure/database/query_executor_repositories.py",
        "configs/runtime/query-executor-v1.yaml",
        "configs/compiler/pmmc-v5.yaml",
        "configs/compiler/pmmc-v6.yaml",
        "configs/runtime/query-executor-v2.yaml",
        "src/pmmc/runtime/qbank_reranker.py",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
    ),
    "answer": (
        "src/pmmc/baselines/answer_upgrade.py",
        "src/pmmc/runtime/answer_policy.py",
        "prompts/runtime/pmmc_answer_contract_v2_1.txt",
        "alembic/versions/20260911_0029_step7b_query_terminal_contract.py",
        "src/pmmc/baselines/suite_recovery.py",
        "src/pmmc/baselines/suite_contract.py",
        "src/pmmc/infrastructure/database/evaluation_repositories.py",
        "src/pmmc/infrastructure/database/experiment_repositories.py",
        "src/pmmc/runtime/answer_grammar_preflight.py",
        "src/pmmc/models/execution.py",
        "src/pmmc/baselines/answering.py",
        "src/pmmc/baselines/rag_config.py",
        "src/pmmc/baselines/rag_context.py",
        "src/pmmc/baselines/rag_runner.py",
        "src/pmmc/baselines/rag_workflow.py",
        "src/pmmc/baselines/suite.py",
        "src/pmmc/infrastructure/experiment_runs.py",
        "src/pmmc/runtime/answer_context.py",
        "src/pmmc/runtime/answering.py",
        "src/pmmc/runtime/query_executor.py",
        "src/pmmc/runtime/query_executor_config.py",
        "src/pmmc/runtime/query_backend.py",
        "src/pmmc/runtime/query_runner.py",
        "src/pmmc/runtime/query_workflow.py",
        "src/pmmc/infrastructure/database/query_executor_repositories.py",
        "configs/baselines/rag.yaml",
        "prompts/baselines/full_context_answer_v1.txt",
        "prompts/runtime/pmmc_answer_with_evidence_v1.txt",
        "prompts/runtime/pmmc_answer_contract_v2.txt",
        "configs/runtime/query-executor-v1.yaml",
        "configs/runtime/query-executor-v2.yaml",
        "src/pmmc/schemas/answer_output.py",
        "src/pmmc/models/gateways.py",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
    ),
    "evaluation_deterministic": (
        "alembic/versions/20260907_0028_independent_answer_abstention.py",
        "alembic/versions/20260730_0023_step7_evaluation_context_lineage.py",
        "src/pmmc/schemas/evaluation.py",
        "src/pmmc/infrastructure/database/evaluation_models.py",
        "src/pmmc/infrastructure/database/evaluation_mappers.py",
        "src/pmmc/evaluation/workflow.py",
        "src/pmmc/evaluation/facts.py",
        "src/pmmc/evaluation/evidence.py",
        "src/pmmc/evaluation/graph.py",
        "src/pmmc/evaluation/qbank_protocols.py",
        "src/pmmc/evaluation/qbank_semantic_workflow.py",
        "src/pmmc/schemas/answer_output.py",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
    ),
    "evaluation_judge": (
        "src/pmmc/schemas/semantic.py",
        "src/pmmc/evaluation/workflow.py",
        "src/pmmc/evaluation/qbank_protocols.py",
        "src/pmmc/evaluation/qbank_semantic.py",
        "src/pmmc/evaluation/qbank_semantic_workflow.py",
        "src/pmmc/models/gateways.py",
        "configs/compiler/qbank-semantic-match-v3.yaml",
        "configs/compiler/qbank-semantic-match-v4.yaml",
    ),
    "evaluation_analyze": (
        "src/pmmc/evaluation/workflow.py",
        "src/pmmc/evaluation/facts.py",
        "src/pmmc/evaluation/metrics.py",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
    ),
    "export": (
        "src/pmmc/infrastructure/experiment_export.py",
        "src/pmmc/compiler/questioner_export.py",
        "src/pmmc/compiler/program_export.py",
        "src/pmmc/compiler/program_pilot.py",
        "src/pmmc/compiler/step6d_failure.py",
        "src/pmmc/evaluation/qbank_export.py",
        "src/pmmc/evaluation/qbank_protocols.py",
        "src/pmmc/runtime/step7_pilot.py",
        "src/pmmc/evaluation/export.py",
        "alembic/versions/20260802_0024_step7b_scientific_contract_hardening.py",
        "alembic/versions/20260803_0025_step7b_candidate_gate_rejection_codes.py",
    ),
}


def stage_family(stage: str) -> str:
    """Map a concrete launcher stage to its source-ownership family."""

    if stage == "questioner:model-stage":
        return "compiler_program"
    if stage in {"step7_prefix:doubter", "step7_prefix:doubter_failure"}:
        return "doubter"
    if stage == "step7_prefix:r0":
        return "dry_run"
    if stage.startswith("planner_verify:"):
        return "verified_qbank"
    if stage in {
        "automatic_quality",
        "review_bundle",
        "export_verify",
        "final_report",
    } or stage.startswith("export:"):
        return "export"
    prefix = stage.split(":", 1)[0]
    if prefix in _STAGE_SOURCE_FILES:
        return prefix
    if stage.endswith(":model-stage"):
        return stage.removesuffix(":model-stage")
    return prefix


def tracks_scientific_compatibility(stage: str) -> bool:
    """Whether a stage source change invalidates that checkpoint and its suffix."""

    return stage_family(stage) in _STAGE_SOURCE_FILES


def checkpoint_projection_rebuild_allowed(stage: str) -> bool:
    """These read-only projections can be rebuilt without discarding model facts.

    This is not export compatibility: completion must be invalidated on restore.
    Automatic quality, approval and review bundles still require exact identity.
    """
    return stage in {"export:memlens", "export:memgallery"}


def source_fingerprint(project_root: Path, stage: str) -> str:
    """Hash only the source files owned by a stage, in stable path order."""

    family = stage_family(stage)
    relative_paths = _STAGE_SOURCE_FILES.get(
        family,
        (
            "src/pmmc/experiments/launcher.py",
            "src/pmmc/experiments/stage_components.py",
        ),
    )
    rows: list[tuple[str, str]] = []
    for relative in sorted(relative_paths):
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"stage source fingerprint file is missing: {relative}")
        rows.append((relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    return canonical_content_hash(rows)


def source_fingerprints_compatible(
    stage: str,
    recorded: str,
    current: str,
) -> bool:
    """Accept exact identities or a chain of explicitly audited transitions."""

    if recorded == current:
        return True
    if stage in {"automatic_quality", "export:memlens", "export:memgallery"} and (
        recorded, current
    ) in _SOURCE_EXPORT_ONLY_FINGERPRINT_TRANSITIONS:
        return True
    family = stage_family(stage)
    transitions = (
        _TYPE_ONLY_SOURCE_FINGERPRINT_TRANSITIONS
        | _RECOVERY_ONLY_SOURCE_FINGERPRINT_TRANSITIONS
    )
    frontier = [recorded]
    visited = {recorded}
    while frontier:
        source = frontier.pop(0)
        for transition_family, transition_source, target in transitions:
            if transition_family != family or transition_source != source:
                continue
            if target == current:
                return True
            if target not in visited:
                visited.add(target)
                frontier.append(target)
    return False


def build_stage_component(
    *,
    project_root: Path,
    launch_id: str,
    run_id: str | None,
    stage: str,
    dataset: str | None,
    round_id: int | None,
    scientific_input_hash: str,
    upstream_output_hash: str,
    row_counts: Mapping[str, int],
    artifact_hashes: Sequence[str],
    image_digest: str,
    status: StageComponentStatus,
    output_facts: object,
    metadata: Mapping[str, object] | None = None,
    alembic_revision: str = STEP6D_ALEMBIC_REVISION,
) -> StageComponentManifest:
    """Build an append-only component; image identity is execution-only."""

    fingerprint = source_fingerprint(project_root, stage)
    output_hash = canonical_content_hash(output_facts)
    identity = {
        "launch_id": launch_id,
        "run_id": run_id,
        "stage": stage,
        "dataset": dataset,
        "round_id": round_id,
        "scientific_input_hash": scientific_input_hash,
        "source_fingerprint": fingerprint,
        "upstream_output_hash": upstream_output_hash,
        "output_hash": output_hash,
        "status": status.value,
        "image_digest": image_digest,
        "execution_attempt_id": (metadata or {}).get("execution_attempt_id"),
        "alembic_revision": alembic_revision,
    }
    now = utc_now()
    return StageComponentManifest(
        component_id=stable_compiler_id("stage-component", identity),
        launch_id=launch_id,
        run_id=run_id,
        stage=stage,
        dataset=dataset,
        round_id=round_id,
        scientific_input_hash=scientific_input_hash,
        source_fingerprint=fingerprint,
        upstream_output_hash=upstream_output_hash,
        output_hash=output_hash,
        alembic_revision=alembic_revision,
        row_counts=dict(sorted(row_counts.items())),
        artifact_hashes=sorted(set(artifact_hashes)),
        status=status,
        image_digest=image_digest,
        created_at=now,
        completed_at=now if status is not StageComponentStatus.BUILDING else None,
        metadata=dict(metadata or {}),
    )


def deepest_compatible_prefix(
    *,
    expected: Sequence[StageComponentManifest],
    available: Sequence[StageComponentManifest],
) -> list[StageComponentManifest]:
    """Return the longest ready component prefix with exact scientific identities."""

    by_scope = {
        (item.stage, item.dataset, item.round_id): item
        for item in available
        if item.status is StageComponentStatus.READY
    }
    compatible: list[StageComponentManifest] = []
    previous_output = (
        expected[0].upstream_output_hash
        if expected
        else canonical_content_hash(["stage-component-root"])
    )
    for item in expected:
        actual = by_scope.get((item.stage, item.dataset, item.round_id))
        if (
            actual is None
            or actual.scientific_input_hash != item.scientific_input_hash
            or not source_fingerprints_compatible(
                item.stage,
                actual.source_fingerprint,
                item.source_fingerprint,
            )
            or actual.upstream_output_hash != previous_output
        ):
            break
        compatible.append(actual)
        previous_output = actual.output_hash
    return compatible


__all__ = [
    "STEP6D_ALEMBIC_REVISION",
    "STEP7_ALEMBIC_REVISION",
    "build_stage_component",
    "deepest_compatible_prefix",
    "source_fingerprint",
    "source_fingerprints_compatible",
    "stage_family",
    "tracks_scientific_compatibility",
]
