# Switch to tournament compatibility at 70% wins

The user set this research priority on November 10, 2026. Continue measured
strength research until one frozen checkpoint and policy reaches at least 70%
overall win rate on the complete current development benchmark suite. For the
current 256-game schedule this means at least 180 wins. Draws are not wins.
Use the full fixed opponent/map/seed/seat schedule, with valid results throughout;
a partial panel or a crash-excluded display rate does not establish this milestone.
Keep the existing improvement/confirmation protocol for strength claims and
promotion. The 70% milestone is a change of research priority, not an additional
requirement to beat each opponent, and does not require delaying the compatibility
work until a separate promotion confirmation finishes.

At that milestone, freeze and publish the checkpoint, runtime, complete W/L/D
evidence and graphs. Shift the main effort to a tournament-compatible bot before
trying to improve win rate further. The deployed bot must run entirely in Java,
use only standard API calls and legally available observations, and meet the
standard bytecode, memory, compilation and wall-clock limits. No privileged
engine state, modified engine budgets, external Python inference process or
model server may be required during play.

Before implementation, verify the exact current Battlecode 2026 rules and
submission infrastructure rather than guessing numeric limits. Export or distill
the model and redesign sensing/controller behavior as needed. Validate compilation
and games under the unmodified official restrictions, retaining resource-limit
failures as failures. Report the compatible bot's strength separately from the
unrestricted research agent; conversion can lose strength. Optimize compliance
first, then resume strength optimization within those restrictions.

The machine-readable priority is `configs/research-roadmap.json`. This does not
change frozen historical benchmark protocols or authorize a tournament submission.
