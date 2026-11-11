# Target reliability recheck V9r2

V8 is complete: cp2 82/256 wins, terminal head 81/256, PUCT head79/256. Both learned arms fail the development improvement rule. All768 terminal games are valid; two first-cycle deadline interruptions were retained and independently recovered. Complete V8 evidence was exported, verified and isolated-restored before this experiment. No checkpoint was promoted.

V9r2 tests whether exhaustive two-seed target selection overestimated action quality. It does **not** train a model or establish playing-strength improvement. The alternative explanation is poor transfer from useful training-root targets. Testing reliability before expanding the corpus costs184 continuations rather than another1508-cell target collection.

## Frozen schedule and interpretation

Use all32 V8 training roots from `progress/terminal-target-head-v8/roots-training.json`, with the existing terminal-selected and cp2-greedy actions in `diagnostics/target-recheck-planning/action-inventory.json`. Preserve source/input/state/prefix hashes. There are14 changed-action roots and18 identical pairs. Four new continuation seeds219003,219013,219029,219041, shared between actions, yield184 unique terminal games. Identical-action cells are physically shared and contribute zero paired difference; all32 roots remain in the denominator. No root, action or seed is replaced after outcomes.

Seed evidence is `diagnostics/target-recheck-planning/seed-inventory-final-20261111.json`: local recorded JSON/JSONL and available recovery archives, supplemented by a review of explicit reserved lists and formula ranges. These are mechanism-test seeds, not benchmark confirmation seeds.

Reconstruct each root using the unchanged V8 replay checker and frozen cp2. After the forced action, sample both teams from cp2 at temperature0.35 with an independent NumPy generator initialized to the continuation seed. No learned V8 head is used as continuation policy. Verify terminal state/action hashes and root-player score. No benchmark code, trajectories, or outcomes enter these games or any gradients; there are zero gradients.

Report the32 rootwise mean paired differences across four seeds and their equal-root mean. Bootstrap entire source-game clusters with replacement10,000 times, RNG219101, retaining all roots within each selected cluster and dividing by sampled root count. Expansion is supported only if the mean is at least0.10 and the95% percentile lower bound exceeds zero, with all184 valid cells. Report all root differences, original biased selection-seed differences, source groups, roles, teams and maps. No completed classification from missing data. A pass motivates a new separately preregistered learning experiment; a failure does not prove no learning method can improve. No automatic model promotion or confirmation.

## Operations

Separate namespaces: `runs/target-recheck-v9r2`, `progress/target-recheck-v9r2`. One one-thread Python worker and one Java engine at a time, bounded3-hour supervisor. Own lock/STOP plus all seven V8 shared locks/STOPs. Parent polls every0.5seconds, watches180-second cell and1800-second reconstruction budgets, sends SIGTERM then process-group SIGKILL after5seconds. Worker has parent-death SIGKILL and retained per-attempt start/final journals. Parent-death handling does not replace shared ownership checks. Any error or interrupted cell stops for explicit evidence review; no automatic retry or silent deletion. Completed valid cells are never rerun. A cycle can resume only unchanged identity and completed-cell checks; interrupted-cell recovery needs a separate recorded review.

Freeze protocol, config and implementation before preflight/launch. The V9r2 identity pins those files, exact action/root manifests and the full frozen V8 runtime identity. V8 files remain unchanged. Each root reconstruction verifies source, prefix, physical state, actor, legal mask and canonical input. Runtime identity is checked before and after each worker and at completion.

Run `python scripts/run_target_recheck_v9.py --preflight`, then `python -u scripts/run_target_recheck_v9.py --seconds10800` with spaces between option and value. Use the repository `.venv/bin/python`. This experiment contains no SGD or benchmark evaluation. Preserve all V8 models and evidence. After quiescence, export V9r2 journals and identity separately with a verified V8 recovery dependency, verify file hashes and isolated restore before publishing a recovery pointer.

## Revision provenance

Original V9 preflight passed but the first worker stopped during root reconstruction because the wrapper omitted the replay checker’s required `worker_task_path`. No forced action or continuation outcome was generated. Its identity, status and logs remain in the original V9 namespace. V9r2 adds that task-journal path, preserving all scientific settings, roots, actions and still-unused seeds in a separate namespace. This is an infrastructure repair, not a research-setting change.

## Journal compatibility repair

V9r1 completed the first eight terminal games but its reporter rejected missing redundant `decisions` metadata. V9r2 preserves every original record byte-for-byte and imports all eight through a frozen path/SHA256 manifest. The reader derives only `decisions=len(actions)` before running the full terminal/action-hash verifier. Outcomes, actions, source states and seeds are unchanged, and none of these successful cells is rerun. New records explicitly include the field. The error status is written before report generation so a report error cannot leave stale running status. Scientific schedule and decision rule are unchanged.
