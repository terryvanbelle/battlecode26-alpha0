# zdp: zdb with decision points

A copy of zdb (2026b `bots/zdb`, commit 75c15b7b) in package `zdp`. The plan: route zdb's control choices through a
few dozen decision points that a small net can take over (see docs/decision-points.md). First pass: bug-for-bug
compatible with zdb, checked by `scripts/replay_check.py --bot zdp --ref zdb` (identical games under tournament limits).
