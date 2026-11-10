# Terminal-target-head-v6: collection in progress

The source-temperature amendment filled **32/32 training roots, 512/512 anchors
and 16/16 input-disjoint validation roots**. There are 19 completed training
source games and 9 validation sources. All 32 PUCT comparators are frozen.

At `2026-11-10T08:02:40.033889+00:00`, target collection has **423/1,458 valid terminal
games**; attempts by status: `{'ok': 423}`. Both fitted heads, independent terminal
validation and the 72 fresh benchmark games remain pending. No learned
improvement or all-map victory has been established; historical cp2 remains **20/24**.

![Paired source coverage](source-coverage-comparison.png)

[Exact paired coverage and limitations](source-coverage-comparison.md) ·
[Preregistered protocol](../../docs/terminal-target-head-v6.md) ·
[Unchanged all-map gate](../../docs/all-maps-victory.md)

This compares source coverage, not playing strength. Initial-game seeds are
paired with v5, all trajectories are regenerated, and reserves follow the fixed
matching-parity rule. Source self-play uses T=1.0 on both teams and every map;
terminal continuations retain T=0.35. Full self-play recovery assets will be
published after the bounded run becomes quiescent.

Frozen identity: `8ca3b86522d9c7a4d794bb8ab62a1f1173f4fd780350a244fda1f78b4e8a5a26`.
