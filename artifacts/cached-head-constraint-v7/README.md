# V7 training-only recovery

This directory holds the separate recovery export for the preregistered cached-head update-budget/anchor diagnostic. The archive passed staging verification: 575 files, 264 checkpoint payloads, six snapshots and 256 recovery generation pairs. After baseline quiescence, an isolated restore also succeeded: 573 files and all 264 checkpoint payloads verified, with the relocation STOP guard retained. [Restore report](isolated-restore.json).

Archive: [recovery-20261110T095144883025Z-bcd84dbf2d18.tar.gz](recovery-20261110T095144883025Z-bcd84dbf2d18.tar.gz). SHA-256: `bcd84dbf2d18bc749f217c42ad2b33f5c4b9734f40bd25a34c995fd5c7ea6d5f`. [Verification report](verification.json).

V7 uses only v6's fixed training numerical inputs: 32 roots, 512 anchors and the unchanged selected-action labels. Its two policy-head arms use anchor coefficients 1 and 0, each for 2,048 updates. No new rollout, validation game or benchmark game belongs to this experiment. Recovery does not promote any head or authorize further fitting.

## Exact retained state

The complete export retains all six fixed snapshots—steps 128, 512 and 2,048 for both `kl_1` and `kl_0`—plus all **256 immutable recovery-generation PT/JSON pairs**, one per 16 completed updates per arm. No earlier generation is compacted away. It preserves model/policy tensors, both AdamW parameter states, Torch and NumPy RNG, the complete 2,048×32 root and anchor batch arrays, loss histories, exact 128-step reproduction proof, snapshot sidecars and fit manifest. Partial exports retain completed paired generations without claiming complete coverage; missing/corrupt half-publications require review.

Owned source snapshots, runtime/input identities, budget invocation records, status, reports, plots and final audits are included. Keeping every invocation matters: the 900-second budget is shared across resumptions, and an unclosed attempt retains its conservative reserved-time charge. STOP, PID, lock and worker process-log files are excluded. No downloaded engine/JDK/benchmark classes or virtual environment is bundled.

Eight checksum-pinned dependencies are packaged at their original paths: cp2, v6 training NPZ, its metadata, the v6 terminal checkpoint, original 128-step batch arrays, v6 experiment identity, final audit, and training-only diagnosis. V6 metadata retains source/anchor references and exclusion-hash strings; those are provenance, not additional fitting inputs. The original source games and continuation journals remain in the separately verified [v6 archive](../terminal-target-head-v6/README.md):

`artifacts/terminal-target-head-v6/recovery-20261110T092723489573Z-b5f724bb879e.tar.gz`

SHA-256: `b5f724bb879e9a36b816c29c2ac306070d1657b418f2fd25960f2478c2fe3c38`.

## Verify and restore

After the recovery pointer is available, from a checkout containing both recovery helpers and the project Python environment:

```bash
V7_ARCHIVE="$(.venv/bin/python -c 'import json; print(json.load(open("progress/cached-head-constraint-v7/recovery.json"))["archive"])')"
.venv/bin/python scripts/export_constraint_v7.py verify --archive "$V7_ARCHIVE"
```

Restore only to a checkout with no existing v7 run and no conflicting progress or dependency files:

```bash
.venv/bin/python scripts/export_constraint_v7.py restore \
  --archive "$V7_ARCHIVE" \
  --root /workspace/battlecode26-alpha0
```

An isolated inspection copy can instead use `--root /tmp/constraint-v7-recovery`. The helper first verifies every archive member in staging, then reloads and compares exact model, optimizer, RNG, schedule and history payloads after restoration. It refuses path traversal, symlinks, conflicting evidence and an existing v7 run. Existing byte-identical pinned v6 inputs may be reused; they are never overwritten. Current main code/config/protocol files remain untouched, and identity-bound paths are never rewritten.

A relocated restore, missing prior dependency, or mismatched numerical/source runtime creates `runs/cached-head-constraint-v7/STOP`. Resolve those conditions against the archive's recorded identity before any future execution. Exact historical code reproduction belongs in an isolated checkout; do not reset or overwrite newer work. The exporter reuses pure archive utilities from `scripts/export_terminal_v6.py`, so keep that helper alongside `scripts/export_constraint_v7.py`.

## Quiescence and publication

The exporter acquires five locks: campaign, warmstart, v5, v6 and v7. Matching Python actors or Java engines block export; an unreadable working directory does not establish that a matching process is inactive. All five STOP files are preserved, and a quiescent export remains allowed while stopped. No fitter, actor or game is invoked by export/restore.

Finalize existing progress README, plots and audits **before** export. Then verify the immutable archive, restore independently, and publish the archive, sidecar and small final evidence together. Only new recovery-pointer/verification files and this artifact README should be added afterward; changing an already captured progress file calls for a new quiescent export to avoid conflicts in a fresh clone.
