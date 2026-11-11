# V9r2 target reliability recheck

**Active:** 184 distinct continuations for all32 fixed V8 training roots, selected action versus cp2 greedy on four fresh seeds. No SGD or benchmark games. [Frozen protocol](../../docs/target-recheck-v9r2.md).

![Target reliability and completion coverage](target-reliability.png)

[Measured summary](summary.json) · [Frozen identity](identity.json) · [Action pairs](actions.json). No final interpretation until all184 cells are valid. All32 roots remain in the denominator; identical-action pairs share physical games.

Original V9 stopped before a game due to a missing replay-task path. V9r1 then completed eight terminal games but stopped when the reader rejected a missing redundant decisions count. V9r2 preserves those records unchanged, pins their hashes, derives the count from the action transcript only in its compatibility reader, and fully verifies terminal/action proof. All eight games are retained and none is rerun. New records explicitly include the count. [Inherited journal inventory](../../configs/target-recheck-v9r2-inherited.json).

Status: `runs/target-recheck-v9r2/status.json`; log `/tmp/target-recheck-v9r2.log`. One one-thread actor with an isolated Java engine, eight shared/own locks, STOP checks and hard process-group watchdogs. Do not edit its pinned code/config/protocol while active. Failures stop for review; no automatic retries.

After quiescence, `scripts/export_target_recheck_v9r2.py` exports and hash-verifies own state and both earlier launch records, performs an isolated restore, and records the complete verified V8 dependency. Restore that dependency first without overwriting newer files, then verify the V9r2 archive and restore in isolation. Publish the archive/manifest/recovery pointer at the review boundary.
