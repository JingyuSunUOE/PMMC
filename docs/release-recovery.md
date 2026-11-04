# Private preservation and recovery runbook

**Default disposition: keep in place.** Do not delete, prune, compact, rewrite,
move, sync with deletion, or run an environment/volume/registry cleaner. The
public source archive is not a backup. No script in this workflow authorizes
deletion. Do not publish the inventory, backup or restore receipt.

| Private holding | Preserve before any cleanup discussion |
| --- | --- |
| Whole checkout, hidden files, untracked work, old policy modules/JSON, manuscripts, IDE state, environment files | Quiesced filesystem tree with originals; credentials only on access-controlled encrypted media. Separate secret escrow/rotation where necessary. |
| outputs, runs, artifacts, review bundles, generated jobs, checkpoints, manifests, model locks and evidence | Immutable snapshots, provenance, counts and digests. Keep historical IDs, approvals and original ACLs private. |
| Benchmark payloads, source prompts, images, model caches, local virtual environments and build caches | Inventory even supposedly reproducible caches. Resolve links to separately mounted data or cache roots; preserve lawfully held versions and rights records privately. |
| Postgres (including migrations/roles), object stores, volumes, server scratch, caches outside this checkout, remote artifact stores | Separate authorized logical dumps/snapshots and object version inventories, keys, checksums and actual isolated restore drills. A local filesystem walk does not cover these. |

## Operator workflow

1. Stop concurrent writers or use an immutable filesystem snapshot; record
   host, snapshot ID, owner, scope, rights, free space and retention plan in
   a protected ledger. Locate all mounts/links, environment overrides and
   off-tree roots without copying secret values into public docs. Do not dump
   live database directories as a substitute for consistent backups.
2. With Python 3.11+, use `scripts/private_recovery.py inventory --source ROOT
   --report PRIVATE/new-inventory.json` (prefix the commands here with
   `python3`). The report must be outside the source. `--quick` counts
   logical bytes and records metadata only; it is **not** a restore proof.
3. Use `backup --source ROOT --destination EXTERNAL/new-backup` on an
   encrypted, access-controlled, independently held mounted device with
   enough space. No upload. A different filesystem is required but does
   not prove independent hardware, encryption or safe custody. Confirm those
   out of band. The copy is unencrypted at the file level. Backup filenames,
   link targets and contents are sensitive. Do not use public/shared folders.
4. Remount/read back the backup independently. Use `restore-check --source ROOT
   --backup EXTERNAL/new-backup --destination PRIVATE/new-drill` against the
   same frozen source. The tool compares all regular-file bytes, sizes, modes,
   directory entries and link text, then copies to a fresh private directory
   and compares again. A missing/changing file or extra entry fails. Retain
   both copies and investigate; never force a passing receipt.
5. Independently restore the services and check usable migrated DB state,
   object counts and bytes, artifact lineage and runtime access. The file
   tool does not follow symlinks, snapshot open writes, preserve hard-link
   topology or certify ownership/ACLs/xattrs, disk images or remote storage.
   Back up and test any required metadata separately using native tools.
6. Record external device identity, custody, independent hash record, link
   coverage, drill evidence and a named owner's **actual** signoff. Until
   all scopes pass, the disposition is HOLD. Even after verification, any
   cleanup needs a separately reviewed path-by-path proposal and explicit
   permission. No cleanup, remote job, upload, publication or tag is part of
   the release procedure.
