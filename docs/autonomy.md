> **Current objective amendment (November 10, 2026):** the user replaced the
> all-opponent/all-map success requirement with improvement on a broader-map,
> smaller opponent panel. See [active protocol](development-benchmarks-v3.md). This supersedes conflicting historical
> instructions below. Publish opponent-by-map W/L/D and coverage in the README.

# Resumable campaign operation

The goal is one learned policy that beats all twelve frozen opponents on **every one of the 74 available maps**. The user permits small map subsets for routine evaluations, with occasional full-map checks. Each campaign invocation has a **maximum three-hour budget**. Checkpoints, replay data, evaluation rows, and machine-readable status allow later cycles to resume. A cycle, smoke result or partial map panel cannot establish success.

Schema 2 of [`configs/victory_gate.json`](../configs/victory_gate.json), implemented by [`alpha0.victory`](../alpha0/victory.py), requires **888 opponent–map pairs** to pass independently: at least 60% score and at least 50% in each seat, both in **3,552 standard games** and **7,104 fresh-seed confirmation games**. The same candidate must pass both stages. No pooled score substitutes. See [all-maps-victory.md](all-maps-victory.md) for the frozen inventory and exact completion rules.

Routine smoke or other predeclared small-subset diagnostics guide research. Automatic standard qualification starts only after all twelve opponents have complete smoke results with at least 50% score each; it does not run every four weak iterations. A periodic research review may explicitly schedule an additional all-map check. The winner-only initialization and its first two search checkpoints each scored 20/24 on one map, losing both seats to two opponents, so none qualifies from those smoke results.

Confirmation uses four fresh seeds on all maps, not a disjoint map set. Lock one candidate per version before the first game. Failed confirmation consumes the seed set; a later candidate needs a new version and four unused seeds across the complete inventory. Keep `Stash`, `Hike`, `heart` and `ratsareants` out of gradient training through the entire initialization lineage, reserve confirmation seeds, and exclude every benchmark trajectory from training.

## Current research boundary

The neutral campaign is preserved at checkpoint 4 and warmstart at checkpoint 2.
The [terminal-root-ranking-v4 mechanism test](root-ranking-v4.md) completed
418 valid continuations and passed its precommitted rule, with an independent
final audit and unchanged checkpoint/runtime. Its
[published evidence](../progress/root-ranking-v4/README.md) is separate from
benchmark qualification. No SGD or benchmark games occurred in that test.

The recorded review selects [terminal-target-head-v5](terminal-target-head-v5.md)
for the next bounded cycle: implement the pinned fresh-self-play protocol,
matched PUCT control, policy-head-only fits, independent validation and all-12
smoke checks. At this review boundary the protocol is published but the pilot
has not been implemented or run. Do not repeat unchanged target-3 warmstart or
neutral training, or rerun the already complete diagnostic as new evidence.
Preserve the existing champion and use the pilot's new run/progress paths.
Inspect actual process commands, STOP markers and all applicable locks before
starting; if a run is healthy, inspect progress without launching a duplicate.

Only if recovery reveals missing v4 cells, resume them with unchanged
source/configuration:

```bash
.venv/bin/python -u -m alpha0.root_ranking_v4 --seconds 7200
```

It respects `runs/campaign/STOP`, `runs/warmstart-v3/STOP` and
`runs/root-ranking-v4/STOP`, and enforces a two-hour wall deadline. Restore its
tracked `progress/root-ranking-v4/` records and the separately verified warmstart
checkpoint-2 dependency before resuming. Never replace current source/gate or
newer progress with files from the historical warmstart archive. Diagnostic
completion does not authorize relabeling its outcomes as benchmark victories.
Future pilot cycles are bounded by three hours and must preserve incomplete
work rather than fit a completion-selected subset.

## Run one neutral campaign cycle

From this checkout, after bootstrap:

```bash
bash scripts/run_campaign.sh --hours 3
```

The launcher invokes `.venv/bin/python -u -m alpha0.campaign --root . --hours 3`. It uses `flock` to prevent overlapping campaign invocations in this checkout, appends output to `runs/campaign/campaign.log`, and records a process ID in `runs/campaign/campaign.pid`. The campaign owns `runs/campaign/status.json` and its resumable experiment state. The hour budget is bounded by the campaign's safe stopping points; an in-flight game or evaluation may need to finish before exit.

To detach from a terminal hangup while this executor remains running:

```bash
nohup bash scripts/run_campaign.sh --hours 3 \
  </dev/null >/tmp/bc26-campaign-launch.log 2>&1 &
tail -f runs/campaign/campaign.log
```

The PID file is a diagnostic record, not proof that the process is alive: it can remain after a process exits, and operating systems may reuse process IDs. Inspect the status timestamp and process command when checking progress. The lock is authoritative for overlapping launches made through this script. Direct manual invocations of the Python campaign bypass the launcher's lock and should be avoided.

## Stop or resume

Request a graceful stop:

```bash
touch runs/campaign/STOP
```

The campaign checks this marker at safe boundaries. The launcher also refuses to start while it exists. It does not delete the marker, checkpoints, or prior results. To intentionally resume after the campaign has stopped, remove only the marker and launch another cycle:

```bash
rm runs/campaign/STOP
bash scripts/run_campaign.sh --hours 3
```

Normal expiration of the time budget does not create a STOP marker, so a later scheduled continuation can start the next bounded cycle.

## Persistence and recurring continuation

`nohup` protects against a terminal hangup. It does **not** establish that a process survives a paused, stopped, replaced, or reclaimed managed environment. This executor is not running systemd as PID 1, and no persistent service manager or cron job has been installed. We make no claim of continuous compute after environment shutdown.

An hourly **Battlecode self-play research** continuation was created and enabled on 2026-11-10. A subsequent tool lookup confirmed that it inherits this thread's managed **Default Cloud** configuration and `/workspace` working directory. The first scheduled instruction was received in the active conversation. This confirms scheduling and environment selection; it does not establish uninterrupted compute or successful completion of any future run.

The continuation inspects campaign status, resumes interrupted work, reviews measurements, improves the implementation, and publishes checkpoint and progress artifacts. Its instructions now use the all-map schema-2 goal while allowing small-subset evaluations most of the time. It reports meaningful milestones or blockers and pauses itself only after the recorded all-map standard and confirmation stages pass. A laptop sleeping or closing does not directly terminate this cloud process, but the managed runtime can still be paused or replaced. [Release snapshots](campaign-snapshots.md) provide the separate durable recovery path.

The continuing agent should respect `runs/campaign/STOP`, the campaign lock, and any later user instruction to stop. It should verify current process and checkpoint state before launching work, and should never claim that every benchmark has been beaten from training loss, a self-play win, or a partial benchmark panel.

## Existing GCloud VM

`battlecode-dev` is an optional future execution host. Its recorded access information and current credential limitation are in [compute.md](compute.md). This campaign does not connect to it, alter its services, or provision paid infrastructure. Notify the user before using that VM, as requested.

Durable snapshots and recovery commands are documented in [campaign-snapshots.md](campaign-snapshots.md).


## Current policy-transfer research boundary

V4 completed its small mechanism rule; V5 stopped before SGD with a coverage
shortfall. [V6 is complete](terminal-v6-results.md): all 1,458 selection cells,
64 shared validation games and 72 smoke games were valid. Both fitted heads
and fresh cp2 remained 20/24; zero validation advantage means the expansion rule
failed. Preserve its [verified archive](../artifacts/terminal-target-head-v6/README.md),
audits and all prior runs. Do not repeat completed v6 work as new evidence.

The selected next experiment is the training-only
[cached-head-constraint-v7 diagnostic](cached-head-constraint-v7.md). It compares
fixed 2,048-update fits with anchor coefficients 1 and 0 on the unchanged v6
training data. It uses no new games, validation or benchmark outcomes, and
promotes no policy. The 128-update KL=1 control must reproduce v6 exactly.

After inspecting actual commands, status, all five locks and corresponding STOPs:

```bash
.venv/bin/python -u -m alpha0.constraint_v7 --preflight
.venv/bin/python -u -m alpha0.constraint_v7
```

The supervisor holds campaign, warmstart, v5, v6 and v7 locks, runs one numerical
worker with one Torch thread, and enforces **900 total wall seconds across
resumptions**. An unclosed invocation conservatively consumes its remaining
reservation. Never reset that ledger or rerun a completed diagnostic. Review its
complete fixed snapshots and classification before selecting later research.

Its own state is in `runs/cached-head-constraint-v7` and
`progress/cached-head-constraint-v7`. Preserve all six diagnostic snapshots,
optimizer/RNG/batch state and intermediate recovery generations in its separate
verified export. A fresh validation protocol is required for any later learning
experiment. None of these subset pilots satisfies the all-map victory gate.

On the managed host, `/proc/PID/cwd` can be unreadable across sandbox namespaces
even when the command line is readable. The frozen v6 internal scan can falsely
return no actors. Use actual command lines and flock ownership; v7 and the
separate v6 exporter conservatively block matching actors whose cwd is unreadable.
Do not edit pinned runtime/config/protocol files while an actor is active. Never
restore historical source or progress over current main or newer local work.
