# Release image contract

The release asset `images.lock.json` is installed at
`deploy/release/images.lock.json` in an extracted source release or a clean
checkout of the exact source commit. It is **not** a source file. Do not
commit it or copy it into either image build context.

Host integration: `from scripts.release_images import load_images`;
`load_images(root, overrides=None)` returns a dict of the four final
repository@sha256 references. A missing or invalid lock raises `ValueError`.
Set `PMMC_IMAGE_LOCK_SHA256` to the independently reviewed checksum from
the release tag/attestation, not a hash computed from an untrusted download.
The reader checks the exact lock bytes, the commit (in a Git checkout),
and the hash of SOURCE.sha256. An extracted archive is bound via its
verified source manifest. Optional `overrides` map individual roles to
full mirror repository@sha256 references with the identical digest.

JSON object (no extra fields):

- `schema_version`: integer 1
- `source_commit`: 40 lower-case hex Git commit
- `source_manifest_sha256`: SHA-256 of SOURCE.sha256 bytes
- `images`: exactly `app`, `iam`, `postgres`, `minio`, each a
  repository@sha256 reference; no mutable tags

The producer uses GHCR references for app and IAM and separately verified
upstream references for PostgreSQL 16 with pgvector and MinIO. No release
tag is created by the build. An operator can mirror any image by digest.

## Public source

Work only in the public repository. The committed `.dockerignore` is also
the exact source inventory: paths, never directory globs granting new files
access. Subdirectory build contexts have their own deny-all allowlists.
Stage only deliberately reviewed public files and removals, not a private
checkout, downloads or generated outputs. In the public root:

```sh
python3 scripts/release_source.py inventory --index > .dockerignore
# Review the path diff before accepting new files. Never auto-stage a private tree.
git add -- .dockerignore
python3 scripts/release_source.py manifest --index > SOURCE.sha256
git add -- SOURCE.sha256
```

After the parent commits, `python3 scripts/release_source.py archive --output
/path/to/new/source.tar.gz` packages HEAD only. It requires exact agreement
among the Git tree, the inventory and the source hashes. Runs at the same
Python/zlib versions emit identical bytes; no local ownership, timestamps,
credentials, local model files or untracked state are copied from the host.
The archive root is the project root. Verify the downloaded archive's checksum
**before** extracting, and run `python3 -B docker/verify_public_source.py .`
in the pristine extraction **before** adding external assets.

## Candidate publication

1. Parent pushes the reviewed public source branch and dispatches
   `release-check.yml` on that branch with `publish_images=true`. The existing
   workflow path must be registered on the default branch. Push and PR events
   only run read-only checks. The `runtime_image=true` dispatch can exercise
   both cold builds without pushing.
2. The workflow verifies the public source archive and requires the real Kind
   test: separate bound RWO claims, a PG16 pgvector row and an S3 object, both
   pods gone and recreated, both original PV bindings and marker values intact.
   It then builds the Linux/amd64 app, runs offline scientific tests with no
   network, publishes it, and builds IAM with that digest in a separate
   `docker/minio` context. Candidate tags are run-specific; only digest
   references enter the lock. No source file requires an image's future digest.
3. Download artifact `release-candidate-<40-hex-commit>` from the passing run.
   It contains `source.tar.gz`, `images.lock.json` and `SHA256SUMS`. Verify the
   attestations on **both** assets against the reviewed repository, workflow
   and commit (not an arbitrary run in the same account):

```sh
: "${REPO:?public OWNER/REPO}" "${COMMIT:?reviewed full commit ID}"
for asset in source.tar.gz images.lock.json; do
  gh attestation verify "$asset" --repo "$REPO" --source-digest "$COMMIT" \
    --signer-workflow "$REPO/.github/workflows/release-check.yml" --deny-self-hosted-runners
done
sha256sum -c SHA256SUMS
```

4. Confirm both GHCR packages are publicly pullable by digest with no account
   credentials. New packages may need a parent-controlled visibility change.
   Only the parent creates the release tag **at the exact attested commit**
   and attaches the unchanged candidate assets. Put the lock and archive
   SHA-256 values in the signed/reviewed tag or another independently trusted
   record. A checksum beside an unsigned, unreviewed download is not trust.
   After the release already exists, the parent can use `gh release upload
   "$TAG" source.tar.gz images.lock.json SHA256SUMS --repo "$REPO"`.
5. Users verify the release assets, extract the source, install the reviewed
   lock at the path above, and export `PMMC_IMAGE_LOCK_SHA256` before launch.
   For air-gapped sites, transfer the verified bytes and mirror images without
   changing their manifest digests. Rebuilding locally is not mirroring.

## Upstream receipts

`deploy/release/upstream.json` stores the **index** digests. Anonymous registry
GETs verified the byte SHA-256, linux/amd64 child and image configuration:

| Image | Verified Linux/amd64 manifest | Runtime contract |
| --- | --- | --- |
| docker.io/pgvector/pgvector:pg16 | `sha256:c0cd736fe8b8b5d408db56b780570bac404fe518e66e39de624aa46f9aabc60a` | PG 16.15; pgvector v0.8.7 |
| MinIO RELEASE.2025-04-22T22-12-26Z | `sha256:3f97c5651cb6662b880c787a232b6b34fec8d8922e08d6617b25d241a21164bb` | v0.1's original index, byte-identical public mirror |

The original Quay location no longer allowed anonymous pulls at verification.
The [upstream MinIO Dockerfile](https://github.com/minio/minio/blob/RELEASE.2025-04-22T22-12-26Z/Dockerfile.release)
bundles `/usr/bin/mc`. IAM copies that binary from the same mirrored index and
verifies its actual SHA-256, `ac90da87a35641be5a0ac75d49de5161ddb47d629b5ba01261b0ae9e00aea15f`,
matching the upstream release checksum bundled with it. Compatibility pins
are not security-support claims. Keep these services private to the workload.
