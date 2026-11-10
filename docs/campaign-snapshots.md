# Durable campaign snapshots

`scripts/campaign_snapshot.py` packages a quiescent campaign state for recovery in another executor. The GitHub release tag is `campaign-state` in `terryvanbelle/battlecode26-alpha0`. Local creation never contacts GitHub; publishing and downloading are explicit subcommands.

The archive includes:

- `runs/campaign/v2`: all model checkpoints, latest checkpoint, config, plans, replay index/game arrays, and game outcomes needed to resume.
- `progress/campaign-v2`: evaluation cells/reports, result ledgers, metrics and graphs.
- `runs/gate-ledgers/*.json`: consumed holdout-gate records, required to preserve one-time holdout use.
- `runs/campaign/status.json`, when present.

Only approved data extensions are included. Logs, temporary files, locks, hidden files and credential-like filenames are excluded; symlinks are rejected. No environment variables, credential files, source-repository checkout, virtual environment, engine installation or SSH material is packaged. The Git commit and dirty-worktree flag are recorded. Commit and push source changes separately so that a replacement executor can recover the code required by the archive.

## Create at a checkpoint boundary

After the learner and evaluation workers have stopped writing:

```bash
.venv/bin/python scripts/campaign_snapshot.py create --root . --checkpoint-boundary
```

`--checkpoint-boundary` is for the campaign itself while it holds the campaign lock and has made its workers quiescent. A manual snapshot of a stopped campaign should omit this flag; the script then acquires the campaign lock and refuses an active campaign.

The latest checkpoint is read with `torch.load(weights_only=True)` to resolve its specific `replay_path` generation; older checkpoints without that field use `replay/`. The referenced replay index and compressed game hashes are verified. Every selected file is hashed and checked for changes while archiving. The archive contains a checksum manifest, and a separate `.manifest.json` contains the complete archive SHA-256. The archive and sidecar are published to their final local names only after complete creation. Names contain a timestamp and hash prefix and are immutable. A failed or changing-source snapshot must be retried at another safe boundary.

## Publish after successful creation

Use the archive path printed by `create`:

```bash
.venv/bin/python scripts/campaign_snapshot.py publish --root . --archive /absolute/path/to/campaign-state-TIMESTAMP-HASH.tar.gz
```

Publishing creates the prerelease tag if absent, uploads the archive first and checksum manifest last, then verifies both uploaded assets exist. Existing assets are checked for equality and never clobbered. Downloaders consider only complete pairs, so an interrupted upload does not replace the preceding recoverable snapshot. The release has a 2 GiB per-asset limit; this tool refuses archives above 1900 MiB.

Only a successful publish updates `progress/campaign-v2/snapshot.json`. Commit and push that small pointer with the progress graphs. It records the exact immutable archive, full hash, source commit and release URL. Snapshots are not declared durable before upload succeeds. Old remote snapshots are retained; this script does not delete release assets.

## Recover in a replacement executor

Clone the repository, check out the recorded source revision when necessary, bootstrap dependencies, and ensure no campaign is running. Download the snapshot named by the committed pointer:

```bash
.venv/bin/python scripts/campaign_snapshot.py download --root .
.venv/bin/python scripts/campaign_snapshot.py restore --root . --archive /absolute/path/printed/by/download.tar.gz
```

If a successful upload occurred before its pointer was pushed, `download --latest` discovers the newest complete immutable archive/manifest pair. The default uses the committed pointer and verifies its full hash. Neither download nor restore starts training.

Restore verifies the archive SHA-256, every member's checksum and size, an allowlist of paths, and absence of duplicate paths, symlinks, hardlinks, traversal or special files. It stages the data before publishing the complete run directory. Known operational JSON/JSONL path fields (checkpoint, opponent, output, stop file and map directory) beneath the old repository root are rebased to the new root after checksum validation. Hash-bound identity/configuration subtrees are preserved exactly; binary checkpoints and replay arrays remain byte-identical. Restoration records those changes in `runs/campaign/restore.json`.

Restore refuses an active campaign or any existing v2 run data. Fresh-clone Git progress and archive progress are reconciled in staging before the run is published. Immutable evaluation request/attempt files must agree; new files are retained from either side. Training rows are unioned by iteration and conflicting duplicate iterations fail. Evaluation summaries are reconstructed from the union of immutable attempts and identity-bound rows; every error attempt is retained. Reports can advance only to a same-identity/same-schedule superset of completed cells; an already fuller report is preserved. Any conflicting raw game evidence fails before publication.

Graphs, README, latest-summary files and process status are historical presentation and may take the archive version; the next campaign regenerates them from merged evidence. The restore report records if training metrics are ahead of the archived model: metrics cannot reconstruct missing model weights, so restore sets `review_required` and creates/preserves `runs/campaign/STOP` to prevent automatic retraining of duplicate iterations. A continuing research agent must resolve that discrepancy before intentionally clearing the stop marker. Existing holdout gate ledgers still require exact semantic equality; consumed holdouts are never replaced by older ledger state. There is no force-overwrite option. The restored status file describes the previous process and is not proof that a process is running. `runs/campaign/STOP`, if present, is preserved.

The source files are protected against accidental changes during packaging, but this is not a filesystem snapshot of arbitrary concurrent training. Always create at a verified checkpoint boundary. A persistent archive allows recovery; it does not keep a cloud executor alive.


## Git fallback for release-upload authentication

Release storage remains preferred. If GitHub specifically rejects release-upload credentials with an authentication/authorization error, publishing can preserve archives up to **90 MiB** in the separate `campaign-checkpoints` branch. Ordinary network errors do not silently change storage, and larger archives fail with a clear request for chunked storage or corrected upload credentials.

The fallback uses a temporary isolated repository and index. It appends immutable archives/checksum manifests under `snapshots/`, updates that branch's `latest.json`, and pushes with normal fast-forward checks. It never force-pushes, deletes old archives, checks out a different branch in the source workspace, or changes credentials. Concurrent conflicting publication fails safely.

The main progress pointer records `storage: git`, the exact storage commit, branch, archive path and full SHA-256. Download fetches that exact commit and verifies the archive, even if later snapshots have advanced the branch. `download --latest` considers both complete release pairs and the Git branch pointer. Push the small main-branch pointer after a successful publication so future executors have a direct durable reference.
