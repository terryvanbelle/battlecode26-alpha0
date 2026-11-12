# Economical research continuation

This is the short current operating handoff. Update `progress/research-continuation.json` before yielding; keep it short and replace obsolete current-state fields instead of appending history. Put completed evidence in `docs/research-log.md`. Do not reread the full log or old automation prompt on every wake.

The user authorizes independent work toward a reasonable imitation-initialized model, then RL and measured development-v4 improvement. Exact zdb imitation is not required. Engineering budgets may be raised for the measured workload on this managed cloud. No paid infrastructure or GCloud battlecode-dev, and no tournament submission.

## Each continuation

1. Run `.venv/bin/python scripts/research_status.py` from `/workspace/battlecode26-alpha0`. Inspect actual actor commands/cwd, locks, status and STOPs before mutations or launch. Unknown cwd blocks conservatively; resolve via read-only process inspection, never assume idle. Never duplicate a healthy owner or edit its pinned inputs. Preflight is a snapshot, not a launch lock; experimental supervisors still acquire all shared locks atomically.
2. Read the small JSON handoff and current protocol's top amendment. If a healthy owner is working, inspect its compact status only; perform useful independent nonpinned work if needed. Suppress repetitive status messages. Don't poll in a tight loop.
3. If idle and authorized work remains, implement/execute the next substantive unit. Continue through implementation, appropriate checks and the next dependent step where feasible. A small test passing or a documentation commit is not itself a reason to end the work cycle. Keep bounded supervisors for long jobs and record their actual command/status path. Never fabricate background activity.
4. At a real review, context/runtime boundary or blocker, save a concise handoff: actual owner (or null), phase, completed evidence, exact next action and any failure requiring review. Inspect incomplete attempts before recovery; no blind retries, no successful game reruns, no SGD on incomplete data. Preserve failures. Publish meaningful source/results and recoverable raw artifacts without overwriting history.
5. Notify only a verified major milestone, a blocker requiring user action, or confirmed improvement. When nothing material changes, suppress delivery. The hourly schedule is a recovery mechanism, not guaranteed continuous compute; managed-host suspension or tool failures can delay it.

## Scientific invariants

- V8 through V11 are complete; do not rerun games, fits or finalizers. V11 both heads failed. Original cp2 remains83W173L/256; fresh V11 control81W175L is not a replacement baseline. Evidence/recovery pointers live under their existing progress directories.
- Preserve all historical gates/results/archives and ancestral training-excluded maps `Stash`, `Hike`, `heart`, `ratsareants`. Keep `runs/zdb-allmaps-cp2/STOP` intact (user stopped0W60L).
- No benchmark/confirmation trajectories in training. zdb demonstrations belong only to a clearly disclosed new exposed-teacher lineage. Use fresh preregistered sources and whole-game train/validation splits. Student inputs use legal public observations, never audit state or teacher private globals.
- Freeze numerical practical imitation-only gameplay criteria before observing those evaluation outcomes. Exact teacher-action agreement is diagnostic, not a pass requirement. Use an independent opponent panel and matched untrained baseline. Then RL with the planned controls.
- Development-v4 stays8opponents×8maps×2seeds×2seats=256; retain zdb/bestsofar/basic51 without individual-win requirements. Improvement rule remains≥5pp score gain, paired MAP bootstrap95%lower>0, aggregate zdb nonregression. Run `scripts/compare_development.py --config configs/improvement-v4.json` on complete comparable panels, then fresh preregistered confirmation; no auto-promotion or tuning on confirmation.
- At180wins/256 for a frozen policy, prioritize tournament compatibility: allJava deployment, current official limits/legal APIs. README retains actual per-opponent/map WLD, completion/checkpoint and all-bot graph with bold black overall winrate; refresh with explicit v4 manifest only when new benchmark cells exist.

Read detailed authoritative docs/configs when that phase needs them: `docs/development-benchmarks-v4.md`, `configs/improvement-v4.json`, `configs/benchmarks-development-v4.json`, `DESIGN.md`, `docs/autonomy.md`, `configs/research-roadmap.json`, `docs/tournament-transition.md`. Latest user corrections override obsolete historical requirements.
