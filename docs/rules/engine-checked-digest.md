# Battlecode 2026 "Uneasy Alliances": engine-checked rules digest (engine 1.2.6)

Sources: `docs/rules/specs.txt` (cited as `specs.txt:N`) and engine 1.2.6 sources in
`reference/bc26-engine-src/battlecode/` (checked byte-identical to `engine/battlecode26-java-1.2.6-sources.jar`
for GameWorld, InternalRobot, RobotControllerImpl, GameConstants, UnitType, TrapType, MapLocation, TeamInfo,
GameMapIO, LiveMap; the jar's `MethodCosts.txt` is identical to the source copy). **When spec and engine
disagree, the engine is authoritative**; every disagreement is listed in §14.
Tag format: `[File.java:line]`. `MethodCosts.txt` = `instrumenter/bytecode/resources/MethodCosts.txt`.

---------------------------------------------------------------------------------------------------
## 1. Game flow and turn order

### 1.1 Match / game
- Each map is one game. The engine runs one `GameWorld` per map; with `bc.server.alternate-order` the teams are swapped on alternate games [Server.java:162-175, GameMapIO.java:425-427].
- Max rounds = `GAME_MAX_NUMBER_OF_ROUNDS` = 2000 [GameConstants.java:47, GameMapIO.java:228]. Rounds are numbered 1..2000 (`currentRound++` happens at the start of a round, from an initial 0) [GameWorld.java:149, GameWorld.java:815].
- Each team starts with 2500 global cheese [GameConstants.java:62, GameWorld.java:171-172], 0 dirt [TeamInfo.java:31], and exactly 1 rat king (the map is validated for that) [LiveMap.java:435-464].
- The shared array, traps, cooperation state and cheese all belong to the `GameWorld`, so each game starts fresh [GameWorld.java:26-28, 197].

### 1.2 One round (`GameWorld.runRound`) [GameWorld.java:304-335]
1. `processBeginningOfRound`: round++, then every robot runs `processBeginningOfRound`, which drops its squeaks with `round <= current-5`, clears its indicator string and clears `diedLocation` [GameWorld.java:814-823, InternalRobot.java:1104-1108, 1082-1087].
2. Every robot takes a turn in **exec order** (§1.3). For each robot: `processBeginningOfTurn` → player code → `processEndOfTurn` [GameWorld.java:347-362].
   - **Beginning of turn** [InternalRobot.java:1110-1174]:
     a. reset the squeak-sent counter;
     b. `runCheeseMines()`, which actually runs only on the first robot turn of the round (it uses a per-round flag) [GameWorld.java:122-130];
     c. carried-rat bookkeeping (§5.2), then thrown-rat flight (§5.3);
     d. **cooldown decrement**: action, turning and movement cooldowns each `= max(0, cd-10)`, **unless** the robot is being carried, is being thrown, or is a sleeping cat [InternalRobot.java:1162-1170];
     e. the bytecode limit is reset to the robot's current type's limit [InternalRobot.java:1172].
   - **End of turn** [InternalRobot.java:1176-1435]:
     a. `turnsSinceThrownOrDropped++` if the robot is not carried or flying;
     b. **rat-king consumption happens here, at the end of each king's own turn**: if team global cheese < 2 the king takes 10 damage, otherwise 2 global cheese are deducted [InternalRobot.java:1181-1189];
     c. the cat AI runs here (cats have a `NullControlProvider`, so they run no player code) [InternalRobot.java:1191-1418, Server.java:357-360].
3. `processEndOfRound`: reset the cheese-mine flag, write team stats, clear the "has traveled" set for thrown rats, then `checkEndOfMatch` (the round-2000 check). If a winner is set, the game stops after this round [GameWorld.java:1006-1025].

### 1.3 Exec order
- Robots act in **spawn order**: an ID list appended in `createRobot` [ObjectInfo.java:157]. Each round iterates over a snapshot of that list [ObjectInfo.java:98-113], so a rat spawned mid-round first acts in the **next** round.
- Initial bodies (2 kings plus the cats) spawn in map-file order [GameWorld.java:199-206]. The spec says cats act after all initial rats (specs.txt:278-281). That depends on the map file and is not enforced by the engine.
- Promoting a baby rat to king keeps the same robot and the same position in the order [RobotControllerImpl.java:1268, InternalRobot.java:471-474].
- A newly created robot has all cooldowns = 10 [InternalRobot.java:109-111]. They drop to 0 at its first turn, so it can move, turn and act on that first turn.

### 1.4 Cheese spawning [GameWorld.java:506-557, CheeseMine.java:46-55]
- Each round (on the first robot turn), every mine in index order (`x + width*y`) runs `spawnCheese`: it spawns if `rand.nextFloat() < 1 - 0.99^r`, where `r` = rounds since that mine's `last_spawn_round` (initially 0) [GameWorld.java:507, CheeseMine.java:14, 46-51]. The RNG is `new Random(mapSeed)` [GameWorld.java:165].
- Offset: `dx, dy` uniform in [-4, 4] (9x9 square) [GameWorld.java:510-511]. Paired location: HORIZONTAL symmetry uses `(dx, -dy)`, VERTICAL uses `(-dx, dy)`, ROTATIONAL uses `(-dx, -dy)`, applied to the paired mine [GameWorld.java:518-525]. The pair is the mine at `symmetryLocation(mine)` [GameWorld.java:192-195].
- Validity requires both cells to be on the map and not walls. **Dirt is allowed.** The 5 "retries" reuse the same dx/dy (the offset is never re-rolled), so an invalid offset puts the cheese **on the two mine tiles themselves** [GameWorld.java:513-539].
- On a spawn, +20 cheese is added (it accumulates) to both cells, and **both** mines' `last_spawn_round` is set to the current round [GameWorld.java:546-553, GameConstants.java:121].
- Consequence: each mine of a pair rolls independently while neither has spawned, so per pair the spawn hazard is `1-(0.99^r)^2`. The computed mean interval is **8.86 rounds** (median 8) versus 12.5 for a single roll. So each mine yields about 20/8.86 ≈ **2.26 cheese/round**.
- Initial map cheese comes from the map file [GameMapIO.java:241].

---------------------------------------------------------------------------------------------------
## 2. Unit stats [UnitType.java:5-7]

| | BABY_RAT | RAT_KING | CAT |
|---|---|---|---|
| Max/initial HP | 100 | **600** (initial kings too) [GameWorld.java:1044, InternalRobot.java:100] | 4000 |
| Footprint | 1x1 | 3x3 (location = center) | 2x2 (location = **bottom-left** tile) [UnitType.java:58-75] |
| Vision r² / cone | 20 / 90° | 25 / 360° | 17 / 180° |
| Distance origin | tile center | tile center | `(x+0.5, y+0.5)`, the 2x2 center (`usesBottomLeftLocationForDistance` = size even) [UnitType.java:30-32] |
| `actionCooldown` (attack) | 10 | 10 | **30** |
| Movement cooldown | 10 forward / **18** any other dir [GameConstants.java:100, InternalRobot.java:493-508] | 40, any dir | 20 (pounce 40) |
| Turning cooldown | 10 | 10 | 10 [GameConstants.java:252] |
| Bytecode limit | 17500 | 20000 | 17500 (unused) |

- `addHealth` caps at the type's max HP and kills at ≤0. Damage to non-cats is tallied as `damageSuffered` [InternalRobot.java:551-564].
- Act/move/turn is allowed when the corresponding cooldown is `< 10` [InternalRobot.java:354-370, GameConstants.java:244].
- **Strafe rule**: a baby rat pays 18 if the move direction ≠ its current facing [InternalRobot.java:496-498]. Turning uses a separate cooldown, so **`turn(d)` then `move(d)` in the same turn costs 10 move + 10 turn**: 1 tile/turn in any direction with no strafe penalty.
- **Raw-cheese penalty (baby rats only)**: the move and action cooldown increments are `(int)(base * (1.0 + 0.01*rawCheese))`. Turning is not affected [InternalRobot.java:483-486, 502-505]. Examples: 50 raw gives forward move 15 and strafe 27; 100 raw gives 20 and 36. The action multiplier is computed from the cheese held *before* the action is executed (e.g. `attack` adds the cooldown before spending bite cheese [RobotControllerImpl.java:1153-1155]). `transferCheese` subtracts the cheese first, so its cooldown uses the post-transfer amount [RobotControllerImpl.java:1385-1389]. Float rounding: c=16 with base 25 gives 28, not 29 (the only mismatch for c ≤ 300 across the bases used).
- **Carry multiplier is dead code**: `(int)1.5 == 1`, so carrying a rat does **not** change any cooldown [InternalRobot.java:480-481, 500, 514-515, GameConstants.java:274].
- Robot IDs: initial bodies keep their map-file IDs. Spawned robots (and traps) draw from shuffled blocks of 4096 starting at 10001 (seeded by the map seed) [IDGenerator.java:7-60, RobotControllerImpl.java:1029].

### 2.1 Vision / cone semantics [MapLocation.java:179-206, InternalRobot.java:405-408]
- `canSenseLocation(loc)`: always true for your own location. Otherwise it needs `dist² <= r²` (bottom-left/center version for cats) **and** the angle between the facing vector and `(loc - origin)` ≤ `coneAngle/2 + 0.001°`, computed with `acos` of cosine similarity. 360° (king) skips the angle test.
- Baby rat facing N: tiles with `dy >= |dx|` and `dx²+dy² <= 20`. That is **21 tiles including itself** (22 when facing a diagonal), and only **3 of the 8 adjacent tiles** are visible (N: NW/N/NE; NE: N/NE/E). Everything adjacency-based (bite, pickup, carry, dig, place, trap) needs the target to be sensable, so baby rats act only on those 3 front-adjacent tiles plus their own tile.
- Cat: 180° cone around its 2x2 center, radius √17 from `(x+0.5, y+0.5)`.
- Multi-tile robots are sensed if **any** part tile is sensable [RobotControllerImpl.java:618-626]. `canSenseRobot(id)` checks only the robot's `location` (center for kings, bottom-left for cats) and is false for flying robots [RobotControllerImpl.java:561-564].
- Off-map → `CANT_SENSE_THAT` [RobotControllerImpl.java:214-222]. Action range checks always re-check vision first [RobotControllerImpl.java:224-240].

---------------------------------------------------------------------------------------------------
## 3. RobotController API: legality, costs, cooldowns (BC = bytecode cost from MethodCosts.txt)

Common gates: **ActionReady** = action cd < 10 and not thrown and not carried [RobotControllerImpl.java:717-727]. **MoveReady** / **TurnReady** are the same with their own cooldowns [RobotControllerImpl.java:744-766].
**Range(r)** = target sensable and `dist²(origin, target) <= r` (bottom-left origin for cats) [RobotControllerImpl.java:224-240].
All `canX` methods return false where `X` would throw a `GameActionException`. `GameActionException` types: `CANT_SENSE_THAT`, `OUT_OF_RANGE`, `IS_NOT_READY`, `CANT_DO_THAT`, `CANT_MOVE_THERE`.

### 3.1 Queries (all BC 1 unless noted) [MethodCosts.txt:29-46]
`getRoundNum, getMapWidth, getMapHeight, isCooperation, getID, getTeam, getLocation, getDirection, getHealth, getRawCheese, getGlobalCheese, getAllCheese (raw+global), getDirt (team stash), getType, getCarrying (RobotInfo|null), isBeingThrown, isBeingCarried, isActionReady, isMovementReady, isTurningReady, get*CooldownTurns, getCurrentRatCost`. `getAllPartLocations` costs 5.
**Free (not in MethodCosts, and battlecode classes are not instrumented [ClassReferenceUtil.java:105-106])**: `getBackstabbingTeam()` (null in cooperation) [GameWorld.java:408-410], `getNumberRatTraps()`, `getNumberCatTraps()` [RobotControllerImpl.java:169-177], `MapInfo.flyingRobot()`, `MapLocation.bottomLeftDistanceSquaredTo`.

### 3.2 Sensing
| Method | BC | Semantics |
|---|---|---|
| `onTheMap(loc)` | 5 | bounds only [RobotControllerImpl.java:207-212] |
| `canSenseLocation(loc)` | 5 | §2.1 |
| `isLocationOccupied / canSenseRobotAtLocation` | 5 | ground robot on the tile (not flying, not carried) [RobotControllerImpl.java:539-551] |
| `senseRobotAtLocation(loc)` | 15 | RobotInfo or null; any part tile of a multi-tile robot [RobotControllerImpl.java:554-558] |
| `canSenseRobot(id)` / `senseRobot(id)` | 5 / 25 | by ID. Works on yourself (own tile is always sensable), so you can read your own `chirality` |
| `senseNearbyRobots(...)` (4 overloads) | **100** | radius -1 = vision, otherwise `min(r, vision)`. Includes robots with any sensable part within r of `center`, each once, excluding self. Optional team filter (cats are `Team.NEUTRAL` [GameMapIO.java:421-423]). Order = chirality scan order of the first part tile found [RobotControllerImpl.java:602-635]. Radius < -1 → `CANT_DO_THAT` [RobotControllerImpl.java:574-578] |
| `sensePassability(loc)` | 5 | not wall, not dirt, no flying rat [GameWorld.java:478-482] |
| `senseMapInfo(loc)` | 5 | MapInfo [RobotControllerImpl.java:74-84] |
| `senseNearbyMapInfos(...)` | **100** | all sensable tiles within `min(r, vision)` of `center`, in chirality order [RobotControllerImpl.java:672-692] |
| `getAllLocationsWithinRadiusSquared(c, r)` | **100** | sensable locations only [RobotControllerImpl.java:700-711] |
| `adjacentLocation(d)` | 1 | |

- **MapInfo** fields: `isPassable, isWall, isDirt, getCheeseAmount, hasCheeseMine, getTrap` (**only your own team's trap**, otherwise `NONE`), and `flyingRobot()` (RobotInfo of a thrown rat on the tile) [RobotControllerImpl.java:74-84, MapInfo.java:56-75]. Getters cost 1-2 BC [MethodCosts.txt:113-120].
- **RobotInfo** (public fields): `ID, team, type, health, location, direction, chirality, cheeseAmount` (an enemy's raw cheese is visible; kings always show 0), and `carryingRobot` [RobotInfo.java:13-75, InternalRobot.java:323-345]. A carried rat is not on the grid: you only see it through its carrier's `carryingRobot` or through `senseRobot(id)`.

### 3.3 Movement / turning
| Method | BC | Legal if | Effect |
|---|---|---|---|
| `canMove(d)/move(d)`, `canMoveForward/moveForward` | 10 / 1 | MoveReady. For every part tile + d: on the map, passable, and not occupied by another robot (**exception: a cat may enter baby-rat tiles**) [RobotControllerImpl.java:806-837] | translate. A cat crushes (kills) the baby rats it enters, and the rats they carry [RobotControllerImpl.java:891-904]. Then enemy-trap triggers are processed for **every** new part tile [RobotControllerImpl.java:908-911], then the move cooldown is added [RobotControllerImpl.java:913] |
| `canTurn()` / `canTurn(d)`, `turn(d)` | 1 / 1 | TurnReady. d not null and not CENTER [RobotControllerImpl.java:920-930] | set facing, turning cd +10 [RobotControllerImpl.java:953-958, InternalRobot.java:513-517] |

`move(CENTER)` passes `canMove` when MoveReady (every part's target tile is its own) and costs a strafe-penalized cooldown for baby rats. Kings cannot move onto any robot, including allied baby rats.

### 3.4 Building / king actions
| Method | BC | Legal if | Cost / cooldown / effect |
|---|---|---|---|
| `getCurrentRatCost()` | 1 | | `10 + 10*floor(numBabyRats/4)` [RobotControllerImpl.java:965-969] (see the §6 bug: the count includes promoted rats) |
| `canBuildRat(loc)/buildRat(loc)` | 10/20 | Range(8) from king center (the 16-tile ring around the 3x3, corners included), ActionReady, caller is a king, **global** cheese ≥ cost, tile unoccupied and passable [RobotControllerImpl.java:977-999] | −cost global, action cd +10, new rat faces the **king's facing**, same chirality [RobotControllerImpl.java:1012-1022]. Spawning does not trigger traps |
| `canBecomeRatKing()/becomeRatKing()` | 10/20 | §6 | §6 |

### 3.5 Combat
| Method | BC | Legal if | Effect |
|---|---|---|---|
| `canAttack(loc[,c])/attack(loc[,c])` (rat bite) | 10/10 | ActionReady. Range(vision) (so in cone). `1 <= distSq(center,loc) <= 2` (baby) or `<= 8` (king, the 16-tile ring). `loc` passable. Raw+global ≥ c, c ≥ 0. A robot is on `loc` and it is not your team [RobotControllerImpl.java:1040-1080] | action cd +10 (×cheese mult), then damage `10 + (c>0 ? ceil(sqrt(c)) : 0)`. Cheese is taken raw-first [InternalRobot.java:580-628, 251-265]. A bite on a cat adds `min(dmg, catHP)` to your cat damage [InternalRobot.java:617-619]. A bite on any non-cat **triggers backstab** [InternalRobot.java:623-625] |

You cannot bite a flying rat (not on the grid) or a carried rat (the carrier occupies the tile, so the carrier is hit).

### 3.6 Cheese
| Method | BC | Legal if | Effect |
|---|---|---|---|
| `canPickUpCheese(loc)/pickUpCheese(loc[,amt])` | 5/5 | Range(2) (baby: own tile + visible adjacent; king: its own 3x3), cheese > 0, rat, not flying. **No ActionReady check, so it also works while being carried** [RobotControllerImpl.java:461-477] | takes all (or `clamp(amt,0,avail)`) cheese. **No cooldown**, repeatable within a turn. A baby rat holds it as raw; for a king it goes straight to global **and counts as "cheese transferred" for scoring** [RobotControllerImpl.java:490-526, InternalRobot.java:252-255]. Works on dirt tiles too (no passability check) |
| `canTransferCheese(loc,amt)/transferCheese` | 5/5 | Range(9) from the baby rat to any king tile (in cone), ActionReady, robot on `loc` is an allied king, caller is a baby rat, `0 < amt <= raw` [RobotControllerImpl.java:1343-1371] | raw→global, counts as transferred, action cd +10 (×mult after the transfer) [RobotControllerImpl.java:1383-1391] |

### 3.7 Dirt
| Method | BC | Legal if | Effect |
|---|---|---|---|
| `canRemoveDirt/removeDirt(loc)` | 5/5 | ActionReady. Range(2) baby / 8 king / 4.5 cat. Rats need raw+global ≥ 5. Tile has dirt [RobotControllerImpl.java:276-292] | dirt removed, team dirt +1 (cats: the dirt is destroyed), −5 cheese (raw first), action cd +25 (×mult) [RobotControllerImpl.java:450-459, TeamInfo.java:223-233] |
| `canPlaceDirt/placeDirt(loc)` | 5/5 | ActionReady, same ranges, team dirt ≥ 1, cheese ≥ 0. Not wall, no robot, no dirt, no mine [RobotControllerImpl.java:253-274]. **Cheese, traps and flying rats on the tile are not checked** | dirt placed, team dirt −1, 0 cheese, action cd +25 (×mult) [RobotControllerImpl.java:305-313, GameConstants.java:225] |

Dirt placed on cheese hides that cheese under impassable dirt. It can still be picked up from an adjacent tile.

### 3.8 Traps [TrapType.java:12,17]
| | RAT_TRAP | CAT_TRAP |
|---|---|---|
| Cost (raw first) / action cd | 20 / 15 | 10 / 10 |
| Max active per team | 25 | 10 |
| Trigger | enemy rat (baby or king) whose part enters the 3x3 around the trap (r² 2) | a cat part enters the 3x3 |
| Damage / stun | 50 / **sets** move cd = 30 | 100 / **sets** move cd = 20 |

- `canPlaceRatTrap/placeRatTrap`, `canPlaceCatTrap/placeCatTrap` (BC 5/5): ActionReady, Range(2) baby / 8 king. Cat traps also need `catTrapsAllowed` (§7). The tile must be passable, have no robot, have no own-team trap (an **enemy trap on the tile is allowed**), the team must be below the max count, cheese must cover the cost, and there is no mine [RobotControllerImpl.java:339-364]. Effect: cooldown ×mult, cheese, and trigger-list registration over the 3x3 [RobotControllerImpl.java:1024-1034, GameWorld.java:589-603].
- `canRemoveRatTrap/removeRatTrap`, `canRemoveCatTrap/removeCatTrap` (BC 5/5): Range(2/8) and own trap of that type. **No ActionReady check, no cooldown, no refund** [RobotControllerImpl.java:325-337, 366-377, 406-447].
- Trigger processing [RobotControllerImpl.java:864-883, GameWorld.java:627-651]: it runs on move (all new part tiles), pounce landing, drop, thrown-rat landing/collision, and the 8 ring tiles of a new king. Own-team traps and wrong-type traps are ignored. For each trap: set the victim's move cd to `stunTime`, credit cat-trap damage `min(100, catHP)` to the trap owner, a rat trap **triggers backstab**, remove the trap, then deal the damage. Several traps can fire at once: damage stacks; the stun is a set, so it does not stack.
- Since the stun is set *before* the move cooldown is added, a trapped rat ends with move cd 30+10 (forward) and a trapped cat with 20+20 (walk) or 20+40 (pounce).
- Visibility: only through MapInfo of your own team. Enemy rats and cats cannot sense traps [RobotControllerImpl.java:77-78].

### 3.9 Ratnapping (baby rats only)
| Method | BC | Legal if | Effect |
|---|---|---|---|
| `canCarryRat(loc)/carryRat(loc)` | 5/10 | Range(2) & sensable, ActionReady, not already carrying, not carried. Target is a baby rat, not flying, not self. Not (enemy and you were its last grabber and its `turnsSinceThrownOrDropped < 2`). **And** one of: target cannot sense your location, same team, or `target.HP < your HP` (strict) [RobotControllerImpl.java:1460-1537] | target leaves the grid and drops whatever it carried onto its tile, carry timer = 10 [InternalRobot.java:685-700]. Action cd +20 (×mult). **Grabbing an enemy triggers backstab** [InternalRobot.java:643-651] |
| `canDropRat(d)/dropRat(d)` | 10/10 | ActionReady, carrying, `loc+d` on the map, passable, unoccupied [RobotControllerImpl.java:1411-1426] | carried rat placed (keeps its own pre-carry facing), traps processed, no damage. Action cd +20 [RobotControllerImpl.java:1454-1458, InternalRobot.java:731-752] |
| `canThrowRat()/throwRat()` | 10/25 | ActionReady, carrying, the tile in **your facing** direction is on the map, passable, unoccupied [RobotControllerImpl.java:1393-1409] | action cd +20, flight (§5.3) [RobotControllerImpl.java:1438-1442] |

### 3.10 Communication / misc
| Method | BC | Notes |
|---|---|---|
| `squeak(int)` | 50 | returns false if you already squeaked this turn. No cooldown, works while carried/flying (§8) [RobotControllerImpl.java:1286-1294] |
| `readSqueaks(round)` | 10 | §8 |
| `writeSharedArray(i,v)` | 50 | kings only, `0<=i<64`, `0<=v<=1023`, otherwise `CANT_DO_THAT` [RobotControllerImpl.java:1307-1317] |
| `readSharedArray(i)` | 10 | anyone. **No bounds check**: a bad index throws `ArrayIndexOutOfBoundsException` (not a GameActionException) [RobotControllerImpl.java:1320-1322, GameWorld.java:1094-1096] |
| `disintegrate()` | 0 | kills self [RobotControllerImpl.java:1557-1559] |
| `resign()` | 0 | destroys all your robots, opponent wins [RobotControllerImpl.java:1562-1571] |
| `setIndicatorString/Dot/Line`, `setTimelineMarker` | 1 | string truncated to 256 / label to 64 [GameConstants.java:53-56] |
| `Clock.yield/getBytecodesLeft/getBytecodeNum/getTimeLeft/getTimeElapsed` | 0 | [MethodCosts.txt:1-5] |

---------------------------------------------------------------------------------------------------
## 4. Bite damage
- `damage = 10 + ceil(sqrt(c))` for `c > 0` cheese, otherwise 10 [InternalRobot.java:608-613, GameConstants.java:202].
- Marginal efficiency: c=1 gives +1, 4 → +2, 9 → +3, 16 → +4, 25 → +5. So c = k² buys +k (1 dmg/cheese only at c=1). Kill thresholds: a 100-HP rat takes 10 plain bites. A 4000-HP cat takes 400 plain bites.

---------------------------------------------------------------------------------------------------
## 5. Carry / throw / drop details

### 5.1 Being carried
- The carried rat is removed from the grid and its location follows the carrier [InternalRobot.java:689, 462-464, 1138]. It cannot move, turn or act (ActionReady fails), and its **cooldowns do not decrease** [InternalRobot.java:1166]. It **can** squeak, sense, `pickUpCheese` and remove traps (those have no ready check).
- It is immune to bites and scratches because the carrier occupies the tile. A cat move or pounce onto the carrier kills both [RobotControllerImpl.java:897-901, InternalRobot.java:954-958].
- Carried rats are not on the grid, so they do **not receive squeaks** [GameWorld.java:1079].

### 5.2 Auto-release (both teams) [InternalRobot.java:1117-1141]
- Each of the carried rat's turns starts with:
  - if the carrier is dead → dropped at the carrier's location;
  - else if the timer == 0 → release;
  - else timer−1.
- So release happens on the carried rat's 11th turn after the grab.
- Release target = carrier location + **the carried rat's own facing**. If that tile is on the map, passable and empty, the rat is dropped there (traps processed). Otherwise the roles **swap**: the carried rat becomes the carrier of its former carrier, with a 10-turn timer [InternalRobot.java:659-683].
- When a carrier dies, its carried rat is dropped at its tile [GameWorld.java:1145-1148].

### 5.3 Flight [InternalRobot.java:702-842, 1143-1160]
- A throw sets `remaining=4` and immediately travels 2 tiles in the **thrower's facing** direction. On each later turn of the thrown rat: `remaining−1`; if it is 0, land; otherwise travel 2 tiles (unless it already traveled this round); if it is then 1, land.
- **Range depends on exec order**:
  - thrown rat acts **after** the thrower (spawned later): **6 tiles**, landing 2 rounds after the throw;
  - thrown rat acts **before** the thrower: **8 tiles**, landing 3 rounds later.
- Each 1-tile step:
  - off map → `hitGround`;
  - tile holds a cat → **the rat dies and the cat sleeps 2 turns** (no cat damage, no score);
  - tile holds a robot, wall, dirt or flying rat → `hitTarget`;
  - otherwise move.
- `hitGround`: 10 damage, then traps at the landing tile, then move/action/turn cd +10 each [InternalRobot.java:754-778].
- `hitTarget`: damage `10 + 4*(2*remaining + (firstStepOfTurn ? 1 : 0))`. That is 46/42 on the throw turn, 38/34, 30/26, 22/18. The same damage goes to the ground robot hit (any team, including kings, **without triggering backstab**). A flying rat that is hit is forced to land on its next turn. Then the thrown rat's cooldowns +20 each [InternalRobot.java:780-813].
- A thrown rat is impassable (it shows in MapInfo), cannot be sensed by `senseNearbyRobots` or attacked, cannot pick up cheese, and does not trigger traps mid-flight [GameWorld.java:478-482, RobotControllerImpl.java:474-476].

---------------------------------------------------------------------------------------------------
## 6. Rat king formation [RobotControllerImpl.java:1158-1279]
- Conditions (caller is a baby rat; checked in this order):
  1. ActionReady;
  2. **global** cheese ≥ 50;
  3. team kings < 5;
  4. not (`round > 1200` and kings ≥ 2), so round 1200 itself is still allowed [GameConstants.java:68-82];
  5. all 9 tiles of the 3x3 centered on the caller are on the map, no cat or king on them, all passable (no wall, dirt or flying rat);
  6. **≥7 allied baby rats among the 9 tiles, counting the caller**, so 6 others.
- Effects:
  - HP = sum of the HP of allied robots on the 9 tiles (caller included; carried rats are not counted), capped at 600.
  - Every other robot on the 8 tiles dies, **including enemy baby rats (no backstab)**. Their raw cheese, and that of rats they carry, goes to your global cheese; rats they carry die.
  - A rat carried by the caller dies, and its cheese drops on the center tile.
  - −50 global cheese. The caller's raw cheese goes to global; this does *not* count as transferred.
  - Type becomes RAT_KING with the same ID. Cooldowns are unchanged and **no cooldown is added**.
  - Traps in range of the 8 ring tiles are triggered (rat traps hurt the new king and backstab).
  - Kings +1.
- **Bookkeeping bug**: `numBabyRats` is never decremented for the promoted rat (only `addRatKings(1)`) [RobotControllerImpl.java:1278, GameWorld.java:1053-1056, 1132-1135]. Each promotion permanently inflates the baby-rat count by 1. That raises `getCurrentRatCost` and the "rats alive" tiebreak.
- Spawn cost (§3.4): `10 + 10*floor(babyRats/4)`. Costs 10 for 0-3 rats, 20 for 4-7, ..., 110 for 40-43. Cumulative cost for the first 20 rats is 600; for the first 40 it is 2200.

---------------------------------------------------------------------------------------------------
## 7. Cooperation → backstab [GameWorld.java:404-431]
- State starts as cooperation. `backstab(team)` is a one-way switch: it records `backstabRound` and `backstabber`.
- Triggers (the only three call sites):
  1. a bite that damages a non-cat enemy [InternalRobot.java:623-625];
  2. an enemy **rat trap** being triggered, where the trap owner is the backstabber; this includes traps fired by king formation, drops and landings [GameWorld.java:639-642];
  3. `carryRat` on an enemy [InternalRobot.java:648-650].
- **Not** triggers: thrown-rat collision damage, king-formation kills, cat traps, biting cats.
- What changes:
  - (a) the scoring weights (§9);
  - (b) the game no longer ends when the last cat dies [GameWorld.java:853, 1181];
  - (c) new cat traps can only be placed by the **non-backstabbing** team while `currentRound - backstabRound <= 100` [GameWorld.java:412-423, GameConstants.java:65]. Existing cat traps remain.
- Nothing else changes: squeaks stay team-only, and there is no damage multiplier.

---------------------------------------------------------------------------------------------------
## 8. Communication
- **Shared array**: 64 ints per team, 0..1023, kings write, everyone reads, effective immediately for robots acting later the same round. It is never cleared within a game [GameWorld.java:197, 1090-1096, GameConstants.java:233-236].
- **Squeak**: `Message(content, senderID, round, senderLocation)`. Content is a full 32-bit int; location is the center for kings and the carrier tile for carried rats [RobotControllerImpl.java:1289-1290].
  - Delivered to every ground robot with **any part tile within dist² 16 of the sender's location** that is **on the sender's team or a cat**, once per robot, never to the sender [GameWorld.java:1073-1088]. Enemy rats never hear your squeaks. Flying and carried rats hear nothing.
  - One per turn [GameConstants.java:185].
  - Lifetime: a squeak from round R is readable in rounds R..R+4 (purged at the start of R+5) [InternalRobot.java:1082-1087, GameConstants.java:182].
  - `readSqueaks(-1)` returns the whole queue (arrival order). `readSqueaks(r)` returns only those with `round == r` [RobotControllerImpl.java:1297-1304].
- **Cats hear squeaks from both teams** and react only in attack mode (§10).

---------------------------------------------------------------------------------------------------
## 9. Scoring, win conditions, tiebreaks
- Immediate loss: when a team's last king dies (any cause, including starvation, a cat, or `resign`), the opponent is set as winner (`KILL_ALL_RAT_KINGS`). Play continues to the end of that round, then stops [GameWorld.java:1179-1180, 995-997, 1023-1024].
  - If the other team's last king also dies later in the **same round**, the winner is recomputed by points → global cheese → rats alive → random [GameWorld.java:983-991].
- Cooperation and the last cat dies: the winner is decided immediately by points → cheese → rats → random. The game ends after the round [GameWorld.java:852-867, 1181-1182].
- Round 2000 with no winner yet: points → cheese → rats → random [GameWorld.java:961-978].
- **Points as implemented** [GameWorld.java:911-952]:
  - Weights are (cat, king, cheese) = (0.5, 0.3, 0.2) in cooperation and (0.3, 0.5, 0.2) after a backstab, using the state at the moment of scoring.
  - Each share = team / (sum of both teams), or 0 if the sum is 0. The shares are:
    - living kings **at that moment**;
    - `cheeseTransferred`: baby-rat `transferCheese` **plus cheese picked up by kings** [RobotControllerImpl.java:498-501, 522-525, 1388];
    - `damageToCats`: bites `min(dmg, hp)` plus cat traps `min(100, hp)` [InternalRobot.java:618, GameWorld.java:635-637].
  - `points = (int)(100*(wc*catShare + wk*kingShare + wch*cheeseShare))`. This is **truncation, not rounding** (float math), and the two totals can sum to 99.
- Tiebreak 1: **global** cheese (raw cheese is ignored) [GameWorld.java:872-887]. Tiebreak 2: `numBabyRats + numRatKings`, where the baby count includes the promotion ghosts from §6 [GameWorld.java:892-906]. Tiebreak 3: `Math.random()`, which is unseeded and nondeterministic [GameWorld.java:957-959].
- Team execution-time cap: 1.2e12 ns (1200 s) of total player CPU per game. Exceeding it forces a `resign` [GameConstants.java:90, PlayerControlProvider.java:215-218].

---------------------------------------------------------------------------------------------------
## 10. Cat AI as implemented [InternalRobot.java:1191-1418]

The AI runs at the end of the cat's turn.

**Sleeping**: if `sleepTimeRemaining > 0` (set to 2 by a thrown rat flying into it [InternalRobot.java:825-832]), it decrements that and does nothing. Its cooldowns also do not tick while it sleeps [InternalRobot.java:1164-1166]. Nothing else wakes it early.

**Reference corner (chirality)** [InternalRobot.java:977-1001]: chirality 0 uses the bottom-left tile (`location`). Chirality 1 uses bottom-right for VERTICAL maps, top-left for HORIZONTAL, top-right for ROTATIONAL.

**Waypoints** come from the map file, keyed by cat ID [GameMapIO.java:254-270, InternalRobot.java:131-141].

**BFS** [GameWorld.java:208-297]:
- Precomputed for every (from, to) pair and both chiralities. It is 8-directional and checks **walls and map edges only**, testing the 2x2 footprint from the reference corner (flipped for chirality 1).
- Dirt, robots and traps are ignored. Neighbor order is N, E, S, W, NE, SE, SW, NW (mirrored for chirality 1).

**EXPLORE** (initial state, target `waypoints[0]`):
1. `stuck >= 4`: turn to a random non-center direction (only if TurnReady; the RNG is static `Random(1092)` shared across the JVM [InternalRobot.java:59]).
   `stuck == 0`: if the corner equals the current waypoint, either switch to ATTACK (first arrival; this turn does nothing else) or, if returning from an attack at that waypoint, advance to the next waypoint (cyclic). Then set the facing to the BFS direction toward the waypoint (no turn cooldown used), falling back to `directionTo`, then random.
   `stuck` 1-3: keep the facing.
2. If `canMove(facing)`: move (crushing baby rats) and set `stuck=0`.
   Otherwise, for each part tile: if it can dig the tile ahead, dig (25+5 = 30 action cd); else if it can scratch the robot on the tile ahead, scratch it.
   If neither works, try a pounce to `corner + 2*facing`. If everything fails, `stuck++`, otherwise `stuck=0`.
   - **Because `canMove` also fails while the move cooldown is ≥ 10, on every second turn of normal walking the cat scratches any rat (either team, baby or king) standing on a tile directly ahead of it.** That turn also counts as stuck=1. "Ignores rats in explore mode" is therefore only approximately true.

**ATTACK** (lasts 8 turns):
1. `catTurns++`. If it exceeds 8, set `catTurns=0` and switch to EXPLORE; that turn does nothing else.
2. Take the **oldest** queued squeak (in explore the queue was never cleared, so it can be up to 4 rounds old), then **clear the queue**.
3. `senseNearbyRobots()` **using the facing from before the squeak**.
4. If a squeak was heard and its source is not at its own location, set the facing toward the source (no cooldown).
5. Keep the previous target if it is still in the sensed list. Otherwise the new target is the **last** baby rat or king in the sensed list (chirality scan order, either team).
6. With a target, in this order:
   - scratch it if `canAttack` (action ready and target tile in the cone within √17); scratch = 20 damage, action cd +30;
   - else pounce if `canPounce(targetLoc)`;
   - else face along BFS toward the target and move;
   - else dig the tile ahead.
   A successful scratch, pounce or move ends the turn.
7. Otherwise (no target, or the dig/move attempt failed) it rotates 45° (clockwise for chirality 0, counter-clockwise for 1) if TurnReady. On the first turn back in EXPLORE the target is effectively replaced by the waypoint, and the cat walks back to the same waypoint before advancing.

**Pounce** [InternalRobot.java:882-975]:
- Requires: MoveReady; target tile on the map and passable; bottom-left distance ≤ 13 (≈ 3.6 tiles from the 2x2 center).
- Try each corner (chirality order) landing on the target. All 4 landing tiles must be on the map, passable (no wall, dirt or flying rat), and free of cats and kings. Walls in between are ignored.
- Kills every robot on the landing tiles, including carried rats. Then cat traps at all 4 tiles are processed. Move cd +20 twice (= 40).
- Cats can never pounce onto a king.

**Scratch** works at any range inside the cat's vision cone (√17), not just adjacent [RobotControllerImpl.java:1082-1099].

**Cat lethality summary**:
- baby rat: crushed by a move, killed by a pounce, or 5 scratches;
- king (600 HP): only scratches, 20 per 3 turns.
In one 8-turn attack visit, a cat with cooldowns ready at the start manages at most 3 scratches (turns 1, 4, 7), and pounces in between when its move cooldown allows.

---------------------------------------------------------------------------------------------------
## 11. Maps, symmetry, chirality
- Validation (`assertIsValid`, run when `bc.server.validate-maps` is set) [LiveMap.java:421-511, Server.java:162, 255-258]:
  - 20 ≤ width, height ≤ 60;
  - exactly one initial body per team and it must be a RAT_KING (cats belong to NEUTRAL);
  - walls `< 20%` and dirt `< 50%` (strict);
  - no wall on a mine;
  - mines pairwise dist² ≥ 25;
  - **no wall within dist² 8 of any mine** (the 5x5 around a mine is wall-free).
  - **Terrain symmetry is not verified by the engine.**
- Map file (flatbuffer `GameMap`) [GameMapIO.java:222-282]:
  - `size`, `symmetry` (ordinal: **0 = ROTATIONAL, 1 = HORIZONTAL, 2 = VERTICAL** [MapSymmetry.java:6-10]), `randomSeed`;
  - `walls[]`, `dirt[]`, `cheese[]` (bytes), `cheeseMines` (xs, ys);
  - `catWaypointIds` and `catWaypointVecs`;
  - `initialBodies` (id, x, y, dir, chirality, team, type).
  - The index is `x + width*y` [LiveMap.java:407-419].
- Symmetry meaning [GameWorld.java:79-99]: HORIZONTAL is `(x, H-1-y)` (a mirror across a horizontal line), VERTICAL is `(W-1-x, y)`, ROTATIONAL is `(W-1-x, H-1-y)`. **Not exposed through the RobotController**: infer it yourself (the two initial kings are presumably symmetric).
- Initial facing: the map writer stores "toward the map center", and `spawnRobot` converts CENTER to `directionTo((W-1)/2, (H-1)/2)` [GameMapIO.java:317-322, GameWorld.java:1038-1042].
- **Chirality** comes from the map file per initial body. Spawned rats inherit the king's chirality [RobotControllerImpl.java:1017]. It affects:
  - Sensing order [GameWorld.java:746-793]: x outer and ascending, y inner and ascending, over the bounding box. Chirality 1 reverses x (VERTICAL), y (HORIZONTAL), or both (ROTATIONAL). This order applies to `senseNearbyRobots`, `senseNearbyMapInfos` and `getAllLocationsWithinRadiusSquared`.
  - Rat part order [InternalRobot.java:191-230]: x ascending, then y **descending**, with the same flips.
  - Cat corner, part order (clockwise for chirality 0), BFS mirroring, and the attack-mode rotation direction.

---------------------------------------------------------------------------------------------------
## 12. Helpers and bytecode costs
- `MapLocation`: `add, subtract, translate, distanceSquaredTo, isWithinDistanceSquared` (both overloads), `isAdjacentTo, directionTo, equals, hashCode, compareTo, toString` = **2** each; `valueOf` = 25; `bottomLeftDistanceSquaredTo` is free [MethodCosts.txt:16-28].
  - `directionTo` uses the octant ratio 2.414: if `|dx| >= 2.414|dy|` the result is E/W (CENTER when equal) [MapLocation.java:232-269].
  - `isAdjacentTo` is true for the same tile [MapLocation.java:214-218].
- `Direction`: `rotateLeft/Right, opposite, allDirections, cardinalDirections, fromDelta, getDeltaX/Y, getDirectionOrderNum` = 1 [MethodCosts.txt:6-15]. **`allDirections()` returns all 9 values including CENTER (last)** [Direction.java:140-142]. Enum order: N, NE, E, SE, S, SW, W, NW, CENTER.
- `RobotInfo` getters cost 1-2 and `equals/hashCode/toString` cost 15. Direct field reads are cheaper. `Message` getters cost 3 [MethodCosts.txt:121-138]. `UnitType` getters cost 3 [MethodCosts.txt:142-156].
- `java.lang.Math`/`StrictMath` functions and listed String/StringBuilder methods cost 1. `java.util` and other library code is counted as your own bytecode [MethodCosts.txt]. Allowed packages: `java.io, java.lang(.invoke), java.math, java.util(.function/.regex/.stream), java.text, battlecode.common, scala.*`, minus `DisallowedClasses.txt` [AllowedPackages.txt].
- Exceptions: entering a catch handler in your code costs **+500** bytecodes [InstrumentingMethodVisitor.java:639-643, GameConstants.java:59]. Array creation costs its length (per specs.txt:765-771; array instructions are instrumented at [InstrumentingMethodVisitor.java:124-125, 645-650]).
- Overrunning the limit pauses mid-code. The overrun becomes debt: next turn's budget is `limit + (negative remainder)` [RobotMonitor.java:148-176, 322-329]. A king's budget becomes 20000 starting the turn after promotion [InternalRobot.java:1172].

## 13. Free / leaked information
- Free calls: map width/height, round number, `isCooperation`, **`getBackstabbingTeam`**, own trap counts. `getCurrentRatCost` (1 BC) reveals your baby-rat count (in steps of 4).
- `senseRobot(myID).chirality` gives your own chirality (25 BC). Cats' chirality is visible too.
- Enemy RobotInfo exposes **raw cheese carried**, HP, facing, chirality and the carried robot. Cats' facing shows where their scratch cone points.
- Thrown rats are visible through `MapInfo.flyingRobot()`. Your own traps are visible; enemy traps are not.
- Initial king positions are mirror images (by map design; the engine does not check this), so the enemy king's start is one of ≤3 candidate reflections of your own start.
- Squeaks carry sender ID, round and exact location.

---------------------------------------------------------------------------------------------------
## 14. Spec vs engine disagreements (engine wins)
1. **King HP**: spec says kings start with 500 (specs.txt:437). Engine: 600 max and 600 initial [UnitType.java:6, GameWorld.java:1044].
2. **Cooldown tick**: spec says cooldowns decrease at the start of each round (specs.txt:293-295). Engine: at the start of each robot's own turn, floored at 0, and skipped while carried, flying, or a sleeping cat [InternalRobot.java:1162-1170].
3. **King consumption timing**: spec says end of round (specs.txt:435). Engine: end of each king's own turn, interleaved with other robots [InternalRobot.java:1181-1189].
4. **Cheese cooldown penalty**: spec says multiplier "0.01*amount" (specs.txt:227-228). Engine: `(int)(base*(1+0.01*raw))`, moves and actions only, never turning [InternalRobot.java:483-505].
5. **Rat trap stun**: spec says "adding 30" (specs.txt:395-396). Engine **sets** move cd to 30 (cat trap sets 20) before the move cooldown is added [GameWorld.java:633].
6. **Cat scratch cooldown**: spec says 15 (specs.txt:492-493). Engine: 30 [UnitType.java:7, RobotControllerImpl.java:1142-1143].
7. **Cat vision**: spec says both "17" and "vision cone of 30" (specs.txt:478, 503). Engine: r² 17 measured from the 2x2 center [UnitType.java:7, MapLocation.java:189-194].
8. **Cat spawn**: spec says cats spawn at the map center (specs.txt:515). Engine: positions come from the map file [GameWorld.java:199-206].
9. **Explore mode ignoring rats** (specs.txt:539, 546-547): the engine scratches rats directly ahead whenever a move fails, including every off-cooldown walking turn [InternalRobot.java:1246-1277].
10. **Attack-mode order**: spec says turn toward the squeak, then look for the target (specs.txt:569-577). Engine senses **before** turning toward the squeak, uses the **oldest** queued squeak (up to 4 rounds old on the first attack turn), and picks the **last** sensed rat as the new target [InternalRobot.java:1310-1344].
11. **Points rounding**: spec says `round(...)` (specs.txt:118, 123). Engine: `(int)` truncation [GameWorld.java:938-939].
12. **Cheese-transferred metric**: spec says "via baby rats" (specs.txt:131-132). Engine also counts cheese **picked up by kings** [RobotControllerImpl.java:498-501, 522-525].
13. **King formation count**: spec says "7 or more allied rats within the 3x3" (specs.txt:444-446). Engine counts the center rat itself, so 6 others suffice [RobotControllerImpl.java:1183-1196, Direction.java:140-142]. The promoted rat also stays in `numBabyRats` (raising spawn cost) [RobotControllerImpl.java:1278].
14. **Auto-drop location**: spec says "in front of the carrier" (specs.txt:327-328). Engine: carrier tile + the **carried rat's** facing [InternalRobot.java:1123].
15. **Carried rats acting**: spec says they cannot act beyond squeak/sense (specs.txt:348-350). Engine lets them `pickUpCheese` and remove own traps (no ready check) [RobotControllerImpl.java:325-337, 461-477].
16. **Throw distance**: spec says 2 tiles/turn for 4 turns (specs.txt:359-361). Engine: 6 or 8 tiles depending on exec order [InternalRobot.java:1144-1160, 712].
17. **Throw collision damage**: spec says `4*remaining tiles` (specs.txt:365-369). Engine: `10 + 4*(2*remaining + (firstStep?1:0))` to both rats [InternalRobot.java:781-783].
18. **Cheese spawn location**: spec says only cat-accessible tiles, re-sampled (specs.txt:213-219). Engine checks only on-map and not-wall (dirt is OK), never re-rolls, and falls back to the mine tiles [GameWorld.java:517-539].
19. **Placing dirt on cheese**: spec forbids it (specs.txt:425-426). Engine does not check cheese, traps or flying rats [RobotControllerImpl.java:253-274].
20. **Cat dig cooldown**: spec says 30 (specs.txt:522-526). Engine: 25 + 5 additional = 30, which matches; dig costs cats no cheese [InternalRobot.java:1262-1263].
21. **Carry cooldown multiplier**: `CARRY_COOLDOWN_MULTIPLIER=1.5` is documented [GameConstants.java:274] but has no effect because of the int cast [InternalRobot.java:481].
22. **Map limits**: spec says "no more than 20%/50%" (specs.txt:177-178). Engine requires strictly less, and adds the wall-free 5x5 around mines [LiveMap.java:485-510].
23. **Backstab via throws or king formation**: the spec implies any HP loss caused by a team starts a backstab (specs.txt:74-75). Engine: thrown-rat collision damage and king-formation kills do **not** [InternalRobot.java:785-788, RobotControllerImpl.java:1252-1253].
24. **Rat traps triggered when a king forms**: the spec order matches, but only the 8 ring tiles are processed [RobotControllerImpl.java:1270-1275].

---------------------------------------------------------------------------------------------------
## 15. Facts that matter for strategy (computed)
1. **King upkeep**: 2 cheese/round per king. With zero income and zero spending, 2500 cheese lasts 1250 rounds for 1 king, 625 for 2, 250 for 5. A starving king loses 10 HP/round, so a full 600-HP king dies after 60 starving rounds, and your last king starving = game loss.
2. **Income**: one mine ≈ 2.26 cheese/round on average (20 per ≈8.86 rounds, because a pair gets two rolls). Roughly one mine's output pays for one king. Over 2000 rounds a mine yields ≈ 4500 cheese.
3. **Spawn curve**: rats 1-4 cost 10, 5-8 cost 20, and so on, with the count including promotion ghosts. The first 20 rats cost 600 total; 40 rats cost 2200.
4. **Cheapest cat damage**: a cat trap gives 100 damage for 10 cheese (10 dmg/cheese) and stops the cat's movement (move cd set to 20). 10 active traps = 1000 damage = 25% of one cat. A plain bite gives 10 damage per 10 action cd and 0 cheese.
5. **Bite cheese is poor value**: +k damage costs k² cheese, so only 1 cheese per bite (+10%) is efficient.
6. **Rat trap**: 20 cheese buys 50 damage (half a baby rat) plus move cd 30. It is invisible to the enemy, max 25, and **firing it on an enemy is a backstab by you**.
7. **Cats kill baby rats instantly** by walking or pouncing (≤ √13 from the cat center, over walls). Scratches do 20 every 3 turns at any range in a 180°/√17 cone. Kings cannot be crushed or pounced, so the only cat threat to a king is 20 HP per scratch.
8. **Cat attack windows** last 8 turns at each waypoint. Squeaks only steer a cat in attack mode, and cats hear both teams' squeaks within dist² 16.
9. **Feeding**: any thrown rat that hits a cat dies and the cat sleeps 2 turns (frozen cooldowns, no damage, no score).
10. **Movement**: `turn` then `moveForward` gives a 1 tile/turn any-direction gait. Strafe (18) is only useful when the turn cooldown is unavailable. Raw cheese slows moves: 100 raw = 1 tile per 2 turns. Deliver often: transfer range is dist² 9 to any king tile (up to 3 tiles out) and must be in the rat's cone.
11. **Rat vision is tiny**: 21-22 tiles and 3 of 8 adjacent tiles. Kings see 81 tiles at 360°.
12. **Second king**: needs 50 global cheese and 7 baby rats (≥6 others) on a passable 3x3. The others are consumed and the HP sum is capped at 600 (7 full rats = 700, so 100 is wasted). Upkeep doubles. After round 1200 you are capped at 2 kings for new formations.
13. **Cooperation scoring** (cat damage 50, kings 30, cheese 20): doing all the cat damage and transfers with equal kings gives (int)(50+15+20) = 85 vs 15. **Backstab scoring** (kings 50): 2 kings vs 1 is worth 33 vs 16 on that term alone. Every term is a share, so denying the enemy matters as much as gaining.
14. **Backstab is permanent and asymmetric**: the backstabbed team may still place cat traps for 100 rounds, the backstabber never again. After a backstab, killing all cats no longer ends the game.
15. **King pickup counts as transferred cheese**: a king standing on a mine spawn area collects score-relevant cheese without rats (pickup has no cooldown, but only covers its own 3x3 tiles).
16. **Tiebreaks** favour **global** cheese held at the end, so cheese spent or carried raw at the end does not help.
17. **No-backstab harm**: throwing a rat into enemies deals up to 46 damage to both rats, and forming a king kills enemy baby rats in the 3x3 (stealing their cheese). Neither ends cooperation.
18. **Ratnap**: you can grab an enemy that faces away or has less HP. The victim's cooldowns freeze for up to 10 turns. Grabbing ends cooperation.
19. **Exec order matters**: rats spawned earlier act earlier. A thrown older rat flies 8 tiles, a younger one 6. Same-round information (shared array, squeaks) only reaches robots acting later that round.
20. **The round-2000 result** uses the scoring weights of the current state. A cooperation game that reaches 2000 with cats alive uses the 50/30/20 weights.

(Map census intentionally omitted; handled by a separate tool.)
