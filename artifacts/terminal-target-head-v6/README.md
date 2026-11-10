# V6 recovery archive

The completed v6 pilot is preserved in a separately verified recovery archive. It generated **32 training roots and 16 validation roots**, completed **1,458 selection continuations, 64 deduplicated validation continuations and 72 benchmark games**, and fitted both policy heads for the fixed 128 updates. The independent audit verified integrity, but the preregistered expansion rule **did not pass**. No candidate was promoted and no all-map victory was claimed.

| Artifact | Value |
| --- | --- |
| Archive | [recovery-20261110T092723489573Z-b5f724bb879e.tar.gz](recovery-20261110T092723489573Z-b5f724bb879e.tar.gz) |
| Checksum manifest | [Archive sidecar](recovery-20261110T092723489573Z-b5f724bb879e.tar.gz.manifest.json) |
| Independent recovery check | [Verification report](recovery-20261110T092723489573Z-b5f724bb879e.tar.gz.verification.json) |
| Archive SHA-256 | `b5f724bb879e9a36b816c29c2ac306070d1657b418f2fd25960f2478c2fe3c38` |
| Archive size | 15,496,496 bytes |
| Source commit recorded at export | `c2208816534a7c62a0526f8e44d4661a3c45d72a` |
| Recorded source root | `/workspace/battlecode26-alpha0` |
| Final audit SHA-256 | `ab8f824a645432916417e8142d61c7cd13ec5e69561b96869115ff95288dd564` |
| Underfit diagnostic SHA-256 | `4ff88c4f8fe68b623d35941415a92ac3741907d840da4e7aea405bf9b0f4da68` |

The archive contains **3,987 checksum-listed files**. An independent isolated restore byte-verified all **3,985 restored payload files**, reloaded all **21 checkpoint payloads**, and confirmed the relocation STOP guard. This includes cp2 and both final heads with exact model, optimizer, RNG and batch state, plus preserved intermediate fitting generations. Two historical protocol/config reference files were verified in staging and deliberately not copied over current main files. The new machine-readable pointer is [progress/terminal-target-head-v6/recovery.json](../../progress/terminal-target-head-v6/recovery.json).

## Raw journals and retained evidence

The raw continuation journals live at `progress/terminal-target-head-v6/cells/`. They are intentionally Git-ignored to keep the main repository's ordinary progress updates small; **they are not deleted or omitted from recovery**. The verified archive retains every JSON/JSONL record under that directory, including selection and validation cells, immutable attempt-start records, successful terminal transcripts, failures, and interruptions. Full action tables retain unfavorable alternatives. Archive members keep their original repository-relative paths, and the manifest lists each file's byte length and SHA-256.

The exporter walks the run and progress directories directly, without consulting Git's tracked-file list or ignore rules. Main retains the smaller plots, manifests, summaries and audits. A fresh Git clone therefore needs the recovery archive to recover the full raw journal history.

This export contains **all 3,044 raw continuation journal files**, totaling **125,646,460 uncompressed bytes**. Their archive paths and hashes were compared against the complete local journal inventory. It also preserves all 72 distinct successful benchmark games. Discovery used 19 completed training source games and nine completed validation source games, including reserves; those source counts are separate from root and continuation counts.

The archive also preserves owned source-game transcripts and lossless NPZ inputs, discovery/root/anchor manifests, immutable plans and identities, source snapshots, fitting datasets, completed candidates and interrupted fitting state when present, optimizer/RNG/batch state, and the frozen cp2 dependency. Benchmark records remain under `progress/terminal-target-head-v6/benchmarks/`, separate from gradient data. Downloaded bot source, compiled classes, engine/JDK binaries, virtual environments, locks, PIDs, STOP markers and process logs are not bundled.

## Publication order

1. Wait for every v6 actor and Java engine to finish. Finalize the progress README, plots, audit and other existing progress files.
2. Acquire the campaign, warmstart, v5 and v6 locks and verify there are no matching actors or orphan engines. A matching actor whose working directory cannot be inspected is treated as potentially active.
3. Create the immutable export, then verify the archive and independently restore it into an isolated directory. Check that every raw journal file appears in the archive manifest with its exact bytes and that model/optimizer/RNG/batch payloads round-trip.
4. Publish the archive, sidecar, final small progress artifacts, and a new recovery pointer together. Adding a new pointer does not replace archived evidence. Changes to an already archived progress file require a new quiescent export if the current checkout is to remain directly restorable.

Do not export while the run is active. The exporter preserves all four STOP files and never launches actors or clears a stop request.

## Verify and restore

Select the exact archive from a checkout with the recovery pointer and project Python environment:

```bash
V6_ARCHIVE="$(.venv/bin/python -c 'import json; print(json.load(open("progress/terminal-target-head-v6/recovery.json"))["archive"])')"
.venv/bin/python scripts/export_terminal_v6.py verify --archive "$V6_ARCHIVE"
```

For recovery in a fresh checkout at the original canonical repository path, with the v6 run absent and any existing v6 progress byte-identical to the archive:

```bash
.venv/bin/python scripts/export_terminal_v6.py restore \
  --archive "$V6_ARCHIVE" \
  --root /workspace/battlecode26-alpha0
```

For an isolated audit copy instead:

```bash
.venv/bin/python scripts/export_terminal_v6.py restore \
  --archive "$V6_ARCHIVE" \
  --root /tmp/terminal-v6-recovery
```

The helper verifies the entire archive in staging before copying. It refuses an existing v6 run or conflicting progress, preserves current main source/config/protocol files, and does not rewrite identity-bound paths. A relocated copy receives `runs/terminal-target-head-v6/STOP`; missing or different pinned dependencies also block resumption. It restores the ignored raw journals to their original paths. If newer progress conflicts, preserve it and restore to a separate checkout rather than overwriting evidence. Exact source reproduction belongs in an isolated checkout of the archive's recorded commit.

Restore is a recovery/audit operation. It does not launch actors, promote a candidate, grant a passing research result, or change the preregistered allocation and stopping rules.

## Baseline and runtime dependencies

The frozen baseline is `runs/warmstart-v3/checkpoint-0002.pt`, SHA-256 `20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c`; it is small enough to include. Its separate full warmstart recovery remains at `artifacts/warmstart-v3/iteration-002/resume.tar.gz`.

Cp2 retains its original prior lineage at `artifacts/bootstrap-v3/candidate-002-winners/warmstart.pt`, SHA-256 `c1473de81dda381c9f14868ea246c11f5bbd644fb59b0b0420e547598b26aa0e`. That ancestor remains a pinned external dependency. Provision the recorded Python/JDK/engine/bridge dependencies from the repository setup tools and verify the preserved runtime and source identities before any future execution. Current main source files are never automatically replaced by historical snapshots.

The historical v5 audit snapshot and v5/v6 coverage-comparison report preserve references to v5 source-game hashes. Those older source games remain in the [separate v5 recovery archive](../terminal-target-head-v5/README.md); they are not v6 gradient inputs and are not copied into this archive. All own v6 source-data references remain required and verified. The completed progress README, plots, final audit and underfit report were finalized before this export; only this artifact README and new recovery metadata were written afterward.
