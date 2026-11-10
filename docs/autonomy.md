# Resumable campaign operation

The goal is to keep improving the learned policy and measuring it against all twelve frozen opponents. Each campaign invocation is a bounded cycle. Checkpoints, replay data, evaluation rows, and machine-readable status allow later cycles to resume. Completion of a cycle is not evidence that the research objective has been reached; that requires the campaign's recorded benchmark results and success criterion.

## Run one cycle

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

The continuation inspects campaign status, resumes interrupted work, reviews measurements, improves the implementation, and publishes checkpoint and progress artifacts. It is instructed to report meaningful milestones or blockers, and to pause itself when the frozen victory gate passes. A laptop sleeping or closing does not directly terminate this cloud process, but the managed runtime can still be paused or replaced. [Release snapshots](campaign-snapshots.md) provide the separate durable recovery path.

The continuing agent should respect `runs/campaign/STOP`, the campaign lock, and any later user instruction to stop. It should verify current process and checkpoint state before launching work, and should never claim that every benchmark has been beaten from training loss, a self-play win, or a partial benchmark panel.

## Existing GCloud VM

`battlecode-dev` is an optional future execution host. Its recorded access information and current credential limitation are in [compute.md](compute.md). This campaign does not connect to it, alter its services, or provision paid infrastructure. Notify the user before using that VM, as requested.

Durable snapshots and recovery commands are documented in [campaign-snapshots.md](campaign-snapshots.md).
