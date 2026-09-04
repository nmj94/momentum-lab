# Private recovery bundles

v0.16 adds versioned backups for one idle, tracked output directory and its
**entire shared research registry**. This protects output files and observation
history together; it is not an executable environment, a fresh test dataset,
automatic resume, encryption or a tamper-proof custody service.

## Commands and consent

```bash
momentum-lab backup create experiments/gld-dev --output gld-dev.mlbackup.zip --acknowledge-sensitive
momentum-lab backup inspect gld-dev.mlbackup.zip
momentum-lab backup restore gld-dev.mlbackup.zip --output recovery/gld-dev --acknowledge-sensitive
```

All commands return JSON without performance payloads. `create` and `restore`
require `--acknowledge-sensitive` on each invocation (or exactly
`acknowledge_sensitive=True` in Python). There is no config-file consent, force
overwrite, history filtering, resume or reveal flag in these commands.

The returned `archive_sha256` identifies the complete archive. Record it in a
separate trusted location, then supply it as `--expected-sha256` to `inspect` or
`restore`. Without an independent digest, matching internal file receipts prove
only internal consistency: someone able to change the archive can replace the
receipts too. Hashes are not digital signatures or proof of correct economics.

```python
from momentum_lab import create_backup, inspect_backup, restore_backup

receipt = create_backup("experiments/gld-dev", "gld-dev.mlbackup.zip", acknowledge_sensitive=True)
inspect_backup("gld-dev.mlbackup.zip", expected_sha256=receipt["archive_sha256"])
restore_backup(
    "gld-dev.mlbackup.zip", "recovery/gld-dev",
    acknowledge_sensitive=True, expected_sha256=receipt["archive_sha256"],
)
```

The output archive/directory must not already exist, even if empty. A backup must
be outside the selected run and its sibling run-control storage. Commands return
exit code 0 on success and a usage error (2) on invalid, busy, corrupt or unsupported
inputs. They do not display cached test scores. Inspect reads all member bytes to
verify them but never extracts files, opens archived databases or executes code.

## What is preserved

| Component | Bundle location | Contract |
| --- | --- | --- |
| Selected run output | `run/` | Every supported regular file, including nested notes and old `*.bak.*` sets. |
| Operational history | `control/state.sqlite3` | All attempts and their original timestamps, stages, errors and receipts. |
| Shared research registry | `registry/research.sqlite3` | All studies, observations, failed/reserved accesses, cached reveals and extension tables. |
| Manifest | `manifest.json` | Schema/UUID, creation time, original paths/identities, observed state and per-file sizes/hashes. |

Source names and file bytes are not redacted. The shared registry can contain
other studies' results that are not visible in the selected run's reports. Treat
the archive as sensitive even if that particular run is still sealed. Use private,
access-controlled storage and appropriate encryption outside this tool. The
`.mlbackup.zip` suffix and default `recovery/` folder are ignored by git, but this
does not prevent someone from explicitly publishing them or using another name.
POSIX private modes are requested; Windows users must control inherited ACLs.

External datasets, membership manifests, provider caches, original source files,
dependency wheels, installed environments and outputs of other runs are **not**
automatically collected. Their existing provenance/configuration references are
retained unchanged. Files deliberately stored inside the selected run are copied,
but that does not certify a complete licensed input/environment snapshot.

Empty directories, filesystem permissions/timestamps and OS ownership locks are
not reproduced. SQLite `-wal`, `-shm` and rollback sidecars are consumed through
the corresponding database snapshot, not archived as ordinary files; an orphan
sidecar fails rather than being silently dropped. Corrupt SQLite files fail.

## Consistency and failure behavior

The existing run lock is acquired non-blockingly and held until archive
publication. No lock/journal is initialized for an untracked run; no new attempt
is added. Busy owners/verifiers cause a failure, not a forced unlock. The original
configuration must name an existing registry with the recorded UUID. If the
latest attempt completed, its published artifacts must match its existing receipt.
Failed and interrupted attempts may be backed up, but are never relabelled as
successful research. An abandoned persisted `running` row remains unchanged in
the copied journal; the manifest reports its observed interruption separately.

SQLite files are copied with the [online backup API](https://www.sqlite.org/backup.html),
which preserves a consistent database snapshot including committed WAL data.
Snapshots are integrity-checked and stored in standalone DELETE-journal form;
their logical contents are preserved, not necessarily identical physical file
bytes. Source tables are not filtered, schemas are not rebuilt and no source
checkpoint is requested. SQLite may manage its normal WAL/shared-memory support
files; the command does not reserve, reveal or change application history.

The selected run cannot advance while its lock is held. Other runs may still
write to the shared registry: its backup is a consistent snapshot at capture,
**not** a global stop-the-world snapshot or a promise to contain later accesses.
Database lock failures/time limits abort cleanly. Stop unrelated writers and retry
if necessary. Manual/old-version writers that ignore run ownership are outside
the guarantee; detected file/inventory changes abort publication.

A private staging archive is fully checked before exclusive hard-link publication.
The destination is never replaced; a racing creator's file survives. This needs a
local filesystem supporting hard links. There is no overwrite-based fallback.
An export failure cleans only its own temporary staging, leaves source histories
intact and does not publish a partial archive.

## Restore is deliberately inactive

A valid restoration creates a new outer directory containing `payload/` with the
four components above, plus a final `RESTORE.json` receipt. All payloads are first
copied into private staging and hash-checked. A new destination is then reserved;
the completion receipt is published atomically and last. An interrupted publication
can leave a reserved directory without `RESTORE.json`. Preserve it for inspection
and choose a new destination on retry; the tool does not delete it for you.

Restore does **not**:

- Replace, merge or configure the live registry, including one with the same UUID.
- Rewrite `run_config.json`, cached run paths, frozen protocols or source hashes.
- Rebind the copied operational journal to a different run path or recreate an OS lock.
- Evaluate a strategy, reveal a sealed test or declare previously observed data fresh.
- Open archived databases or execute archived scripts during extraction.

`inspect_run` on `payload/run` therefore reports untracked: it is an inactive
recovery copy, not a newly certified live run. The archive/restore receipt refers
to the verified snapshot, not to future edits of restored files. Reinspect the
original archive to verify its recorded contents.

If the current registry still exists, keep it authoritative: it may include
observations made **after** the backup. Matching registry UUIDs do not prove equal
recency. Never replace it with the older copy to bypass reuse checks. In a genuine
loss, preserve the recovery copy and all newer evidence, restore the original
source/environment/data separately, and reconcile history before any explicit
research invocation. Active restoration and monotonic history merging require a
separate migration contract; this release does not offer an unsafe shortcut.

## Format and resource limits

Backup schema 1 is a constrained stored ZIP, not a general-purpose ZIP extractor.
It allows at most 4,096 archive members including the manifest, 1 GiB of total
payload, a 1 MiB manifest and a bounded central directory. Source traversal also
limits file/directory counts. Member paths are at most 512 UTF-8 bytes, with at
most 16 components of 128 characters; unsafe Windows names, traversal, control
and non-printing characters, surrounding whitespace, trailing dots/spaces and
case/Unicode collisions are rejected.

Central-directory count/length/entry metadata are checked before allocation by
the ZIP parser. Exact membership, sizes, SHA-256 and ZIP CRC checks are enforced.
Compressed/encrypted members, ZIP64, comments/extra metadata, symlinks, hard-linked
source files, reparse-point inputs, duplicate members, unlisted members and unsafe
directory/file collisions fail. Nothing is passed to
[`extractall`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.extractall)
or to a shell. Inputs exceeding this first-version scope require a separately
managed backup process; limits are not silently relaxed.

This adds no dependency or accounting change. Engine/checkpoint schema 5,
run-state schema 1, research/portfolio schemas and all frozen accounting fixtures
remain unchanged. Tracked v0.15 outputs can be archived without running their
strategies; their source/version-locked research still needs its original
environment. Older untracked runs are not scanned or certified automatically.
NFS/SMB, cloud-sync race behavior, hostile filesystem replacement and distributed
locking are not supported consistency guarantees.
