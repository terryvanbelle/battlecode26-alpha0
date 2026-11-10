# V5 partial recovery export

V5 stopped at `review_required`: the fixed discovery schedule completed **20 training source games** but found only **24 of 32 required training roots**. No terminal target cells, validation source games or validation cells ran, and **no SGD or fitted candidates were produced**. This preserves the unsuccessful coverage experiment; it is not a completed training run or a victory claim.

| Artifact | Value |
| --- | --- |
| Archive | [recovery-20261110T073626027858Z-26bdc0f95343.tar.gz](recovery-20261110T073626027858Z-26bdc0f95343.tar.gz) |
| Checksum manifest | [Archive sidecar](recovery-20261110T073626027858Z-26bdc0f95343.tar.gz.manifest.json) |
| Archive SHA-256 | `26bdc0f953439b05b534d459f2e7e2ba11e5eb02fb451dc042f931b2cf8dc0f3` |
| Archive size | 3,099,022 bytes |
| Frozen source commit | `aadc02939e6ff628eb7cc3dd1acf513d902068cf` |
| Recorded source root | `/workspace/battlecode26-alpha0` |
| Experiment identity file SHA-256 | `689422b96fda65d711671e7c857b2752097d37b39a9f437debbfac47c00202ed` |

The archive contains 166 checksum-listed files: complete source transcripts and lossless cached inputs, discovery plans and root evidence, experiment identities and owned source snapshots, status/progress, protocol/config references, and unchanged warmstart cp2. No downloaded benchmark source, compiled classes, engine/JDK binaries or virtual environment is included. Benchmark records, when present in this export format, remain separate from gradient data.

The exporter acquired the campaign, warmstart and v5 locks and checked for orphan actors. Separate archive verification passed. An isolated restore then verified all **164 restored payload files** byte-for-byte and reloaded cp2's exact checkpoint state. The two historical protocol/config reference files were verified in staging but were not copied over current main files. Relocation correctly created a v5 `STOP` marker.

## Verify and restore

From a checkout with the project Python environment:

```bash
.venv/bin/python scripts/export_terminal_v5.py verify \
  --archive artifacts/terminal-target-head-v5/recovery-20261110T073626027858Z-26bdc0f95343.tar.gz
```

For recovery in a fresh checkout at the original canonical path, with the v5 run absent and any existing v5 progress byte-identical to the archive:

```bash
.venv/bin/python scripts/export_terminal_v5.py restore \
  --archive artifacts/terminal-target-head-v5/recovery-20261110T073626027858Z-26bdc0f95343.tar.gz \
  --root /workspace/battlecode26-alpha0
```

An isolated audit copy can instead use `--root /tmp/terminal-v5-recovery`. A relocated restore preserves all identity-bound bytes and creates `runs/terminal-target-head-v5/STOP`. The helper also stops a restored run when required pinned code/config/runtime or ancestor dependencies are missing or different. It never rewrites identities, overwrites an existing v5 run, replaces conflicting progress, or changes current source/config/protocol files. If current progress has advanced, use a separate checkout without those conflicting files; preserve the newer evidence. Exact historical reproduction belongs in an isolated checkout of the frozen source commit, not a reset of active work.

Restore does not launch actors or authorize more fitting. V5's root-coverage failure still requires research review; restoring it does not change the fixed allocation or permit SGD on the 24-root subset.

## Pinned dependencies

Cp2 is included at `runs/warmstart-v3/checkpoint-0002.pt`, SHA-256 `20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c`. Its separately verified full warmstart recovery remains at `artifacts/warmstart-v3/iteration-002/resume.tar.gz`.

Cp2 retains its original initialization lineage: `artifacts/bootstrap-v3/candidate-002-winners/warmstart.pt`, SHA-256 `c1473de81dda381c9f14868ea246c11f5bbd644fb59b0b0420e547598b26aa0e`, with source dataset SHA-256 `8b608a88e563c5998445ca0443001a3300d0162c77aaf631898bec2a79be3ac4`. That ancestor is an external pinned dependency, not a newly bundled or modified prior. Provision the pinned Python/JDK/engine/bridge dependencies using the repository setup tools, and compare them with the preserved experiment runtime identity before any future execution.

The archive and sidecar are small enough to publish directly in Git. The machine-readable pointer is [progress/terminal-target-head-v5/recovery.json](../../progress/terminal-target-head-v5/recovery.json).
