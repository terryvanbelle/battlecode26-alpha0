# Macro interface v2

Select `action_version="macro_v2"` in `EngineEnv` (or the JSON reset request).
The default remains `macro_v1` with its original 17-action meanings, so old
checkpoints and saved trajectories remain playable. Network checkpoints must
record the action version and 40-action dimension; v1/v2 replay data cannot mix.
`MacroV2.java` was authored from the official rules and controller APIs, without
copying frozen benchmark implementations.

The most consequential changes fix limitations of the first action abstraction:
baby rats now bite a legal **perimeter tile** of a king's 3×3 footprint instead
of trying to bite its unreachable center; attacks on cats use their full 2×2
footprint; cheese transfers consider every king footprint tile; and king digging
uses the footprint's leading edge. Combat actions reserve action cooldown for
combat rather than automatically transferring cheese first. Turning to an
already-held direction is avoided, and an already-legal bite does not waste a
turn action. These are controller choices; the underlying game rules are intact.

| ID | Meaning |
|---|---|
| 0 | Hold and attack only cats during cooperation, or hostile units after betrayal |
| 1–8 | N, NE, E, SE, S, SW, W, NW movement/digging; opportunistic enemy-priority combat before and after movement |
| 9–16 | Build a baby in the corresponding compass direction |
| 17 | Gather cheese; return to a king when carrying at least 10 |
| 18 | Return and transfer carried cheese to a king |
| 19 | Occupy a cheese mine with a preference for less allied crowding |
| 20 | Guard an allied king; approach a nearby threat or remain near the king |
| 21 | Assault the nearest enemy king |
| 22 | Hunt the nearest enemy baby |
| 23 | Hunt cats and attack their legal footprint tiles |
| 24 | Retreat toward a locally safer adjacent tile |
| 25 | Move a king toward available cheese, falling back to mines |
| 26 | Spawn toward cheese or mines |
| 27 | Spawn toward an enemy king |
| 28 | Place a legal rat trap near an enemy's likely approach |
| 29 | Place a legal cat trap near a cat |
| 30 | Remove a nearby **own** trap to free trap capacity; enemy traps cannot be removed via the official API |
| 31 | Form a king when legal, otherwise move into a clear allied baby cluster |
| 32 | Approach and attempt to carry an enemy baby |
| 33 | Approach and carry an allied baby below 50 HP |
| 34 | Throw a carried rat along a direction scored for damaging enemy collisions |
| 35 | Throw toward an enemy king, using a predicted eight-tile flight path |
| 36 | Drop a carried rat on a locally safer legal tile |
| 37 | Powered bite, capped at 64 cheese and reserving four cheese per allied king |
| 38 | Clear nearby dirt or navigate into digging range |
| 39 | Cooperating economy: gather/return cheese and attack cats, without initiating rat attacks |

All actions are complete robot turns. Free legal cheese pickup occurs before
and after the primary action. Economic/support actions may then deposit cheese;
combat/carry/throw actions retain their action cooldown priorities. A macro may
be selectable because a relevant target exists yet make little immediate
progress when blocked or on cooldown. It never bypasses an official `can*` check
to manufacture an effect. Terrain and targeting are omniscient research inputs,
not tournament-compatible robot observations.

Navigation uses Dijkstra over robot-center positions. King footprint walls are
hard constraints; dirt, temporary unit congestion and known enemy traps are
costs. Economic routes additionally penalize nearby hostile units. Only the
first step is executed, through the real controller. Future occupancy is a
heuristic, not a claim that other robots move as predicted. Resource selection
combines reachable route cost with cheese amount; mine selection discounts
allied crowding. The route finder examines legal destination centers around
target footprint tiles rather than requiring access to an occupied center.

Retreat, trap placement, target priority, throw rays and king-cluster formation
are fixed heuristics inside the action abstraction. The policy learns when to
choose them. Promotion still requires the engine's bank, nearby babies, clear
3×3 area and king-count limits; extra kings still consume normal upkeep. Carry
and throw use ordinary facing, cooldown and collision rules. No enemy private
bot state is consulted.

Exact self-play snapshots include the action version and all mutable physical
state. Benchmark execution continues against the actual Java opponents and
currently uses greedy network actions. A possible future planning mode could
copy benchmark *physical* state while replacing live Java sandboxes with a
neural opponent model; that would be an explicit approximation, not an exact
snapshot of a benchmark's private program state, and is not enabled here.

Validation includes synthetic official-world tests for perimeter attacks and
transfers, 300 v2 actions with repeated snapshot/replay agreement, and an
archived 4,610-action v1 episode replay matching every pre-existing terminal
physical field (round 152, winner B).
