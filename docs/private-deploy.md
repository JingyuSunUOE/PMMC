# Private deployment pack

One binary age ciphertext contains the original **public release tarball bytes**,
the original reviewed external image lock, a derived asset lock binding the
source tarball/manifest and image/model/benchmark locks, validated deployment
settings, and only the explicitly selected credential fields. No checkout
scan, ambient dotenv lookup, model weights, benchmark data, workload state,
database backup, registry credentials, identity, or private recovery tree.
This is not a backup or a budget approval.

Run the examples from trusted, verified public release tools on a POSIX host
with Python 3.11+ and the official age CLI in a trusted PATH. Use trusted
operator-owned directories, not shared writable paths, symlinked ancestors,
or a public checkout for private inputs/outputs. Protect the machine, swap,
core dumps, shell history, debugging, and CI logs. A private file supplied by
path must already be owned by the operator, regular, singly linked and mode
0400 or 0600. No automatic chmod or printing of file contents.

## Pack

Use a published source tarball and external images.lock.json. Verify both
digests against the trusted release channel, not a neighboring downloaded
checksum. Supply a native public age recipient obtained from its owner over
an authenticated channel; no default recipient, passphrase mode, generated
key, or embedded identity. Never paste secret values into flags.

~~~sh
python3 -B -m scripts.release_host.private_deploy private pack \
  --source-archive "$PUBLIC_SOURCE" --source-sha256 "$SOURCE_SHA256" \
  --images-lock "$IMAGE_LOCK" --images-sha256 "$IMAGE_LOCK_SHA256" \
  --config "$PRIVATE_CONFIG" --recipient "$AGE_PUBLIC_RECIPIENT" \
  --env-file "$SCOPED_DOTENV" --role providers --role hub \
  --output "$PRIVATE_OUT/deployment.age"
~~~

Omit the dotenv and both roles for a credential-free pack. Select just one
role when appropriate. TOML must conform to the launcher's Kubernetes schema;
omit the images table to use the verified lock, or supply mirrors with the
same digests. TOML comments are not archived. A hub role requires an explicit
models.use_hf_token opt-in. No host environment variables are harvested.

The output is a new mode-0600 file, never an overwrite. Send its reported
bundle SHA-256 through an authenticated channel separate from the transfer.
Age authenticates its ciphertext, **not the sender**: anyone with the public
recipient can encrypt a different packet. Substituting ciphertext and an
unauthenticated checksum together is not a verified release.

## Install

Start from installer code you already trust. Obtain the SHA-256 of the exact
published source tarball and the separate `images.lock.json` for the same
release commit from authenticated release provenance. Do not substitute a
fresh tar of a checkout, a reserialized lock, or hashes supplied only inside
the received pack. The installer checks bytes against these pins; it does not
establish publisher identity or fetch or vet images itself.

Set shell positional parameters once for the following examples:

~~~sh
set -- --bundle "$RECEIVED_BUNDLE" --bundle-sha256 "$TRUSTED_BUNDLE_SHA256" \
  --source-sha256 "$SOURCE_SHA256" --images-sha256 "$IMAGE_LOCK_SHA256" \
  --identity "$EXTERNAL_AGE_IDENTITY"

python3 -B -m scripts.release_host.private_deploy private install "$@" \
  --destination "$PRIVATE_OUT/install"
~~~

Use an external noninteractive age identity file; it is passed as a verified
file descriptor to age, not read into Python, copied, searched for, or logged.
The destination must be new. Installation does not run extracted code, build or
pull images, start services, or contact a cluster by default.

Install verifies the entire age decryption, exact envelope schema and internal
checksums, externally pinned tarball/image bytes, source membership and file
hashes, reviewed image digests, and both scientific asset locks before writing.
It rejects traversal, absolute/special/linked/duplicate/case-colliding entries,
extra files and trailing archives. It does not call tar extractall or restore
archive ownership/modes. The source tarball limit is 24 MiB, expanded limit
64 MiB, with at most 4096 members. Decrypted content is kept in process memory.

The new 0700 directory contains the byte-identical SOURCE.tar.gz (or SOURCE.tar
for an uncompressed input), source/ with the original files plus the separate
deploy/release/images.lock.json, assets.lock.json, and the original deployment.age.
Only public data and ciphertext hit disk. The installed image lock is external
to the source manifest; it does not alter the original tarball or source bytes.
Use the original archive as the pristine source build context. Set
PMMC_IMAGE_LOCK_SHA256 to the independently verified digest for the launcher.

## Use Private Settings

The installer does **not** leave a decrypted TOML, dotenv or Secret YAML.
To pass just validated TOML to a trusted launcher via stdin, use a dedicated
pipe descriptor. The helper will not write to a regular file or terminal.
In this POSIX shell example the private TOML travels on fd 3 to the launcher's
stdin, and the helper's non-sensitive receipt goes to stderr:

~~~sh
export PMMC_IMAGE_LOCK_SHA256="$IMAGE_LOCK_SHA256"
python3 -B -m scripts.release_host.private_deploy private config "$@" \
  --to-fd 3 3>&1 1>&2 |
  "$PRIVATE_OUT/install/source/scripts/pmmc" plan --config /dev/stdin
~~~

The standalone helper can also be run from the installed source root. Use
only a trusted reader. Never pipe to tee, logs, public artifact storage, or
diagnostics. The handoff contains no provider/HF fields. The controller is a
separate trust boundary: commands that persist operational state must use an
access-controlled tmpfs or encrypted private state volume, not a public or
plaintext state directory. Do not discard a live controller's state.

## Secret Import

The launcher's integration is:

~~~sh
./scripts/pmmc secrets import --config "$PRIVATE_CONFIG" --env-file "$SCOPED_DOTENV"
~~~

For explicit role selection or deliberate rotation the standalone helper is:

~~~sh
python3 -B -m scripts.release_host.private_deploy secrets import \
  --config "$PRIVATE_CONFIG" --env-file - --role providers < "$SCOPED_DOTENV"
~~~

Only a supplied file or stdin is parsed; never source a dotenv into a shell.
The parser supports single-line literal assignments, optional export, quotes,
and comments. Unselected fields are ignored; shell commands, expansions in
selected values, empty values and duplicate selected fields fail closed.
With no role flags, import only recognized roles present in the input. A
providers role may contain just the key needed by the selected paid model.
An OpenAI-only plan needs no Anthropic key, and vice versa. An open-model
plan does not read the provider Secret. Approved control refuses to proceed
without an explicit model selection rather than requesting both keys.

| Role | Derived Secret name | Permitted fields | Workload access |
| --- | --- | --- | --- |
| providers | `<instance>-providers` | OPENAI_API_KEY and/or ANTHROPIC_API_KEY | Approved control only, for the selected provider |
| hub | `<instance>-hub` | HF_TOKEN | Opted-in model/asset preparation only |

Never create or modify arbitrary Secret names, DB/S3 service credentials,
pull Secrets, ConfigMaps, RBAC, or namespaces through this importer. No
credential is a command-line argument; Kubernetes requests use stdin and
subprocess errors are redacted. Import neither approves a budget nor restarts
existing pods. Set API audit policy to avoid recording Secret request/response
bodies, use API-server Secret encryption at rest and restrict pod/Secret
access. Use a trusted kubeconfig for the exact context and namespace. The
helper passes only PATH, HOME and KUBECONFIG to kubectl, not cloud/provider
credential environment variables. Use an appropriate preconfigured credential
plugin if the cluster needs one.

To import **from the encrypted pack**, use explicit roles at install time,
or retry later without reinstalling:

~~~sh
python3 -B -m scripts.release_host.private_deploy private import "$@" --role providers
~~~

Creation never overwrites: a racing creator fails; rerun for comparison.
An exact existing Opaque Secret is unchanged, including an operator-owned one;
managed status is not adopted. A different value fails by default. A deliberate
--rotate changes only values of an existing mutable importer-labeled Secret
with the exact same keys. It uses a resourceVersion-guarded JSON patch, never
force, delete/recreate, client-side apply, or last-applied annotations. Immutable,
unmanaged, extra-key and missing-key objects require operator migration.
Restart consumers separately and revoke superseded credentials. RBAC: get the
specific named Secrets; create in the target namespace only as needed (native
RBAC cannot restrict create by resourceName; use admission policy for names);
patch only those named Secrets for rotation. No list/get-all, delete, update
or cluster-wide privilege is needed.

Imports preflight selected roles before writes, but Kubernetes has no atomic
multi-Secret transaction. If an API call fails, rerun the same import to
converge; never auto-delete a successfully imported role. A failed cluster
import after install leaves the verified public files/ciphertext in place.

## Offline Test

From the verified extracted public source root, with Python 3.11+, pytest,
Pillow (the shared test fixture dependency), and `age` and `age-keygen` on
PATH, run:

~~~sh
PMMC_REQUIRE_REAL_AGE=1 PYTHONPATH=src python3 -B -m pytest -q \
  tests/test_private_deploy.py::test_real_age_roundtrip
~~~

The test creates its own tiny source tarball, lock, synthetic provider field,
and disposable identities. It verifies rejection of a wrong identity and of
tampered ciphertext even with a matching outer checksum, then runs the
installer CLI and compares the original source and lock bytes. Temporary keys
and packets are removed. No real dotenv, delivery, cluster, or provider is used.
Without `PMMC_REQUIRE_REAL_AGE=1`, this test skips if the age tools are
unavailable. The clean-source CI job installs both tools and requires the test
to run; the rest of that test file also covers single-provider Secret imports.
