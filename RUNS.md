# Run ownership, status and recovery

Since v0.15, each accepted search, exploratory portfolio or registered portfolio
study has an operational attempt record. A cooperating coordinator owns its
output directory through output checks, research and publication. A competing
writer fails immediately with `RunBusyError`; it does not wait indefinitely or
erase the existing attempt. Portfolio input loading/validation can occur before
ownership, but evaluation and output publication cannot.

This is operational state, **not research permission**. The existing study
registry remains authoritative for observation history, frozen protocols, test
claims and cached results. Failure can occur after a test-access claim commits.
Neither recovery nor a new run ID clears that exposure.

## Inspect without revealing results

```bash
momentum-lab runs status experiments/gld-dev
momentum-lab runs history experiments/gld-dev --limit 10
momentum-lab runs status experiments/gld-dev --verify
```

Use an exact output path, not a study ID. Output is JSON with the observed
`status`, `lock`, latest `attempt`, `integrity` and `recovery` guidance. History
adds the most recent attempts in descending sequence order (default 20, allowed
1–100). Metadata includes workflow, invocation mode, stage, UTC timestamps,
package version, owner PID/host, outcome and exception **class**, never exception
messages, parameters or performance scores. The path and host are local metadata;
review them before sharing status output publicly.

Default inspection does not read artifact contents. `--verify` streams recorded
files through SHA-256 without parsing or displaying their contents. Inspection
never initializes a journal, adds an attempt, edits research history, resumes,
reveals or recalculates anything. It briefly acquires an existing OS lock if
available; verification holds it throughout the hash pass, so a writer may
temporarily fail busy while verification is running. Probing requires appropriate
permissions on the ownership file, even though its bytes are not changed.

Python API:

```python
from momentum_lab import RunBusyError, RunStateError, inspect_run

result = inspect_run("experiments/gld-dev", verify=True, limit=10)
# The existing run_search/run_portfolio/run_portfolio_study APIs acquire locks.
# Catch RunBusyError to report contention; do not delete a lock or force a retry.
```

## States and exit codes

| Observed status | Meaning | Safe next action |
| --- | --- | --- |
| `running` / `busy` | An owner or verifier holds the lock. | Wait; never delete its ownership file. |
| `completed` | The latest attempt recorded successful publication. | Use `--verify` to check the current files. |
| `failed` | An accepted attempt raised an ordinary exception or omitted completion. | Inspect the original error and provenance before recovery. |
| `interrupted` | A caught interrupt, or an unfinished record with an available lock. | Preserve history and use the workflow-specific rules below. |
| `unknown` | Ownership cannot be established, e.g. the lock file is missing. | Stop competing owners; do not infer success or force ownership. |
| `untracked` | No usable attempt is recorded, including old-version outputs. | Inspect original artifacts; do not infer completion. |
| `not_found` | Neither output nor a tracked attempt exists. | Check the path. |

`integrity` is separate: `not_checked`, `unavailable`, `verified` or `mismatch`.
Only the latest attempt can be verified, and only when it is completed and the
lock is available. A changed/missing file does not rewrite past completion;
`changed_artifacts` identifies mismatches. `outcome=no_results` means a search
finished normally without a usable candidate, not a successful investment result.

CLI exit codes: **0** means inspection succeeded (including busy, failed,
interrupted or unknown status), **1** means not found or integrity mismatch,
and **2** means invalid arguments or a control/journal error. Automation must
check JSON `status` and `integrity`, not treat exit code 0 as research success.

An OS-released lock with a persisted `running` row is reported as interrupted
without writing to disk. The next accepted attempt records `UncleanExit` and
`detected_at` for its predecessor; `finished_at` remains null because the true
crash time is unknown. PIDs and timestamps are descriptive, not liveness tests.
A preflight rejection does not append an attempt. Once `start` has accepted an
invocation, later provenance, calculation or publication failures are retained.

## What a completion receipt covers

After the coordinator's exports succeed, it records simple filenames, byte sizes
and SHA-256 hashes in one short state transaction. Search snapshots cover its
standard configuration, summary, candidate/stage/sensitivity CSVs and reports
that exist at completion. Portfolio snapshots include configuration, summary,
reports and exported books; a cached portfolio-study replay deliberately covers
only its four summary/configuration/report files, not a newly computed book.

Mutable search SQLite journals/WALs, backups, raw provider/dataset files and
unrelated user files are **not** part of this receipt. Existing data/protocol and
resume fingerprints still apply independently. History retains old receipts
internally, but `--verify` checks only the latest completed attempt. Filenames
are bounded and cannot traverse directories; symbolic/hard links, changed file
identity and concurrent content changes detected during hashing are rejected.

Hash receipts detect accidental missing/changed exports. They are not signatures:
someone who can edit local artifacts and state can forge both. An intact receipt
does not prove correct economics, authorized data, unseen tests or profitability.
Publication remains atomic per file, **not** a whole-directory transaction or
automatic rollback. A crash can leave partial exports; consult attempt state.

## Recovery and backups

1. Inspect the exact path and preserve the original error, files and history.
   Wait if an owner is active. There is no timeout-based or `--force` unlock.
2. For a **single-asset search**, use the original invocation with `--resume`
   only after restoring its original config, data, compatible source/environment
   and registry identity. Existing checks remain mandatory. Reveal still needs
   `--reveal-test`, and observed-test reuse still needs explicit consent/reason.
3. For **exploratory portfolios or portfolio studies**, use a new output run ID;
   in-place overwrite/resume is unsupported. Keep the original recipe/registry.
   An already committed study reveal can be explicitly replayed from its frozen
   cache. An incomplete reveal may require acknowledged reuse; it is not fresh
   out-of-sample evidence.
4. If provenance or state is missing/corrupt, stop rather than reset it. Restore
   the complete compatible backup set at its original paths, or create a new
   run/protocol while retaining research access history. Do not silently rerun a
   revealed test to make a damaged completion receipt match.

Operational files live outside the output directory at
`<output-parent>/.momentum-runs/<output-name>/`: stable `owner.lock` and
`state.sqlite3` (run-state schema 1). This keeps existing output-file contracts
unchanged. Do not choose `.momentum-runs` itself as an output name. OS-native
case/Unicode path aliases resolve through the same filesystem namespace, including
before output creation. Control paths and files must not be links.

Back up only after coordinators/verification have stopped, retaining outputs,
their sibling run-control directory, original data/config/source/environment and
the research registry with its identity and history. The registry may be outside
the result tree; use its documented SQLite-safe backup procedure. Do not copy a
live database without its transactional state. Keep legacy `*.bak.*` sets together.
Do not replace an ownership inode while any process may still hold it. Missing
ownership beside existing state fails closed; it is not automatically recreated.

Moving only outputs loses their operational association; moving control journals
to new absolute paths is not an automatic identity migration. Keep originals or
restore the complete set at its original location. A versioned portable backup/
restore command is future work, not a guarantee of this release.

## Compatibility and platform limits

No research/access schema, accounting formula, dependency or frozen ledger has
changed. Run-state schema 1 is separate from engine/checkpoint schema 5 and the
existing research registries. Old outputs are not scanned or retroactively
certified. Source/version-locked old research needs its original environment;
upgrade using new runs/protocols without bypassing checks.

Ownership uses POSIX [`flock`](https://docs.python.org/3/library/fcntl.html) or
Windows [`msvcrt.locking`](https://docs.python.org/3/library/msvcrt.html), not PID
files. POSIX fork callbacks close inherited ownership descriptors without
unlocking the parent's shared open-file description, preventing worker processes
from retaining a dead coordinator's lock. Normal exits explicitly release locks;
process termination releases OS ownership. Native path-alias and process-death
tests run on Linux/macOS/Windows; POSIX-only fork tests are skipped on Windows.

The guarantee is limited to cooperating current-version coordinators using a
local filesystem with the required locking semantics. Older versions, direct
low-level writers and manual file edits do not participate. NFS/SMB, cloud-synced
folders, hostile filesystem replacement and cross-host distributed execution are
not supported ownership guarantees. No new broker integration or live execution
is introduced.
