package zdp;

/** Behaviour switches and constants. Every experiment adds a switch whose default is the incumbent. */
public final class Cfg {
    public static final int MAX_BABIES = 18, MAX_BABIES_WAR = 18;
    /** With RICH_BANK global cheese or more the baby cap rises to RICH_BABIES (0 = off): a surplus buys bodies. */
    public static final int RICH_BABIES = 40, RICH_BANK = 1000;
    /** A second tier: with RICH2_BANK or more the cap rises to RICH2_BABIES (0 = off). u0m's set-2 superioristhmus
     *  games (3796/3797 vs outercloud) ended with 22-31k cheese banked and 40 rats (the RICH_BABIES cap) from r500 on,
     *  while they ran 53-66; abcdezyxwv runs 50-127 with ~2k banked. Babies eat nothing: only the spawn price. */
    public static final int RICH2_BABIES = 80, RICH2_BANK = 3000;
    /** Cheese kept when spawning under attack. */
    public static final int WAR_FLOOR = 300;
    /** Hard floor under every spawn (king upkeep). */
    public static final int SPAWN_FLOOR = 300;
    /** With fewer than LOW_BABIES babies every floor drops to LOW_FLOOR: a king without rats has no income. */
    public static final int LOW_BABIES = 8, LOW_FLOOR = 100;
    public static final int BABY_TRAP_DIST2 = 13;
    /** Cheese (global, or raw+global for babies) below which no rat trap is placed. */
    public static final int RAT_TRAP_MIN_CHEESE = 200;
    /** Most live rat traps we keep (engine max 25): new ones are bought only as old ones fire. */
    public static final int TRAP_CAP = 10;
    /** The post-move throw scan runs only with this many bytecodes left (it costs up to ~6000). */
    public static final int THROW_BC = 7000;
    /** Micro score bonus for a step onto cheese. */
    public static final int MICRO_CHEESE = 15;
    /** KING_PLUG's ring test at a fraction of the bytecodes, same answers: the 16 ring tiles read unrolled (ring on the
     *  map) with the corner rule and the arc count in bit operations, the king's own tile counted once a move, and the
     *  test skipped (no unplug step, every step allowed) under PLUG_LEFT bytecodes left. u3z's field games: 1296 king
     *  overruns in 31 matches, 1279 of them on mercifullattice (both seats, from r22) and learn_your_lesson (seat B), all
     *  four lost; locally (ko1/ko2, u3v vs araid) the king overran every other turn there, in move(): unplug() counts
     *  the ring of its tile and all 8 neighbours (~1150 bytecodes each) whenever its own ring is split, then closer()
     *  and step() count two more per candidate step. */
    public static final int PLUG_LEFT = 3000;
    public static final int KING_MINE_R2 = 400;
    /** Cats within this dist2 of the king's centre make it dodge. */
    public static final int KING_DODGE_R2 = 25;
    /** The king digs only with this much global cheese or more. */
    public static final int KING_DIG_MIN = 100;
    /** With no enemy or cat in view the king walks to cheese within Chebyshev KING_ROAM of its mine (0 = off). */
    public static final int KING_ROAM = 4;
    /** A second king: the first king names a home-half mine as the site once it has FORM_BABIES babies and
     *  FORM_CHEESE global cheese after round FORM_FROM; babies within dist2 FORM_PULL_R2 gather in its 3x3. */
    public static final int FORM_FROM = 80, FORM_BABIES = 16, FORM_CHEESE = 300, FORM_TTL = 60, FORM_RETRY = 150, FORM_PULL_R2 = 72;
    /** Kings we form up to (engine max 5 before round 1200). */
    public static final int FORM_MAX = 2;
    /** Rich kings (0 = off): up to round 1150, while the bank holds FORM_RICH_BANK or more, the first king names sites
     *  past the cap, up to FORM_RICH_MAX kings, with no income test and a ring site beside it when pickFormSite has
     *  none. Kings alive at round 2000 weigh 0.5 of war points and only a second can form after round 1200. rift vs
     *  spsquared (4342, 4343, 4443): at the 39-40 baby cap we banked 3.5k by r600 and 6-8k by r1050 on one or two kings
     *  (13k at the end) while they held three, and lost each on points 2 kings to 3 with more cat damage and transfer. */
    public static final int FORM_RICH_BANK = 3000, FORM_RICH_MAX = 4;
    /** Form only while the trusted income (x100 per round) beats every king's upkeep, the new one's included, by
     *  FORM_NET100 (Integer.MIN_VALUE = no income test). */
    public static final int FORM_NET100 = -100;
    /** A baby outside the site's 3x3 joins only while it sees fewer than FORM_JOIN allied babies nearer the site;
     *  the rest keep working (0 = every baby within FORM_PULL_R2 joins). */
    public static final int FORM_JOIN = 8;
    /** Formation v3: the site is a home-half mine itself, though the king has not seen it. It is the listed mine
     *  nearest the king, more than 4 from it, that no king holds (none within Chebyshev 4), that has not failed as a
     *  site, and with FORM_MINE_SAFE * dist2 to our start at most its dist2 to the enemy start (4 = pickFormSite's
     *  "clearly ours"); known dirt in its 3x3 ranks it after the clean ones. No mine's 3x3 holds a wall or leaves the
     *  map (415 mines on 53 maps; 59 hold dirt). The field forms its extra kings on mines across its half and ours
     *  (abcdezyxwv on superioristhmus: four, r330-563, 0-1 from a mine; spsquared three), ours form 4-10 from any
     *  mine beside the first king, as the old site needs tiles the king has seen. To r400 in s9s's field games our
     *  mines ranked 4th+ from the king hold 1-2 of our babies and lose 49-59% of their cheese unpicked. Babies within
     *  FORM_MINE_PULL_R2 of a mine site join (beyond FORM_PULL_R2) when (slot >>> 3) % FORM_MINE_SHARE is 0, attackers
     *  never. A mine site skips the income test (FORM_NET100): on where_are_you all four home mines are listed by
     *  r400, 19-30 tiles out, income stays 100-330 a hundred rounds and 2000 cheese lies at each by r1000. A formed
     *  king squeaks its old home king's tile with each spawn, and its babies use that for the half test and the
     *  mirrored enemy start. */
    public static final int FORM_MINE_SAFE = 2;
    /** An endgame king: from round END_KING a lone king names a site for a second (a clear 3x3 off its body, else
     *  pickFormSite's) whenever 7 babies and END_KING_CHEESE global cheese are there, with no income test and no
     *  retry wait past FORM_TTL (0 = off). Kings alive at the end weigh 0.5 of the points at war, 0.3 in cooperation,
     *  so a second king at 1-1 moves 8-16 points to us; s9s's set-4 limit games (9 of 42) all ended at 1-1 or less
     *  for us, and averystrangespace (3287) was lost 24-25 with no cat damage either side. */
    public static final int END_KING = 1850, END_KING_CHEESE = 400;
    /** Raiders (idle fighters sent to enemy-half mines): a raid target is dropped after RAID_STAY rounds; RAID_SPREAD
     *  is the random spread added to each candidate mine's dist2. */
    public static final int RAID_STAY = 60, RAID_SPREAD = 60;
    /** One in RAID_EVERY fighters is a raider: on our half it ignores enemies beyond dist2 RAID_ENGAGE. */
    public static final int RAID_ENGAGE = 8, RAID_HP = 60;
    /** Bytecodes the king's first-turn sense leaves for the rest of that turn (spawn first of all). 5000 can be short:
     *  on DefaultMedium an s7e king ended its first turn at 20001 of 20000 and missed the round-2 spawn (s7g: 7000). */
    public static final int KING_SENSE_RESERVE = 7000;
    /** King press: at war a rat that sees the enemy king squeaks it (allies relay it PRESS_HOPS times); a rat with
     *  PRESS_HP health or more walks to a king sighting at most PRESS_AGE rounds old within PRESS_D2 (gatherers
     *  carrying less than DELIVER_AT: PRESS_GATHER_D2) and bites it, ahead of standoffs and facing. */
    public static final int PRESS_HP = 40, PRESS_AGE = 8, PRESS_D2 = 100, PRESS_GATHER_D2 = 20;
    /** Health at or below which a baby near enemies retreats to its king (0 = never). */
    public static final int RETREAT_HP = 40;
    /** At war a baby with health <= AMMO_HP within Chebyshev AMMO_R of our king's centre, whose nearest enemy rat is
     *  within dist2 AMMO_D2 and has more health (so it can grab us facing it or not), steps away from the king ahead of
     *  retreat(): such rats are the enemy's ammunition. tiny (rsu1): s8u killed s7d's king with 11 of our own rats
     *  thrown point-blank into it (about 42 each); field kings died the same way to rats at Chebyshev 2-3, many <= 40 HP. */
    public static final int AMMO_HP = 50, AMMO_R = 3, AMMO_D2 = 8;
    /** Guard facing: with an enemy rat within this squared distance, steps strafe so we keep facing it, and a
     *  turn ends facing it (rats are grabbed from behind). 0 = off. */
    public static final int GUARD_DIST2 = 13;
    /** Bite cheese (c = k*k gives +k damage), only when it makes the bite a kill: a king pays up to KING_BITE_MAX
     *  above KING_BITE_FLOOR, a baby up to FINISH_K*FINISH_K above FINISH_FLOOR. */
    public static final int KING_BITE_MAX = 100, KING_BITE_FLOOR = 300, FINISH_K = 5, FINISH_FLOOR = 200;
    /** Throws: an enemy baby hit beyond tile 2 counts THROW_FAR_PCT percent; below a THROW_MIN score (42 = a
     *  point-blank wall or rat) the carrier holds for up to THROW_HOLD rounds after the grab. */
    public static final int THROW_FAR_PCT = 40, THROW_MIN = 84, THROW_HOLD = 6;
    /** The carry aim stops its search once AIM_LEFT or fewer bytecodes are left this turn and takes the best throw found
     *  so far. Stash (local u4t vs araid, seed 0, A): a carrier's aimCarry spent 6623 of its last 7000 bytecodes, found no
     *  throw, and the turn ran over in the pick-up stage that came after (r136 and r317, OVS 2021/2122). */
    public static final int AIM_LEFT = 3000;
    /** Squared distance within which a baby answers a king's alert. */
    public static final int ALERT_DIST2 = 225;
    /** King defense: enemy rats within KING_THREAT_D2 of a king's centre raise its alert's threat count; gatherers
     *  within DEFEND_D2 of it then fight (war movement, no standoff against the king's attackers). */
    public static final int KING_THREAT_D2 = 18, DEFEND_D2 = 100;
    /** Raw cheese at which a gatherer heads home. */
    public static final int DELIVER_AT = 30;
    /** Raw cheese at which a gatherer with no pile in sight heads home (at or above DELIVER_AT it goes anyway). The
     *  top field bots carry 33-41 cheese a transfer to our 24-31 (pipes, safelycontained, sittingducks vs AlexT101 and
     *  abcdezyxwv); our gatherers spend 29-39% of their turns walking to a mine ('M') and 11-19% delivering. */
    public static final int DELIVER_IDLE = 10;
    /** Babies spawned before any fighter is assigned. */
    public static final int GATHER_FIRST = 6;
    /** One in FIGHT_EVERY later babies is a cat fighter. */
    public static final int FIGHT_EVERY = 2;
    /** The cat trap of an approaching cat (tryCatTrap) may also go on any tile beside the cat's body, which the cat sets
     *  off on its next step, and during cooperation needs only TRAP_FLOOR, not the projected surplus. spsquared on rose
     *  (3266): when a cat reached their king, 5-6 rats ringed it with cat traps and six fired at r479 alone (63 of 73
     *  fired; cat damage 8080 to our 0). */
    public static final int CAT_LAY_GAP = 20, CAT_LAY_AGE = 3, CAT_LAY_FLOOR = 400;
    /** Rush: babies with spawn index in [RUSH_FROM, RUSH_TO) attack the enemy king from birth. */
    public static final int RUSH_FROM = 4, RUSH_TO = 16;
    /** Stalled attackers hunt cats (0 = off): during cooperation an attacker that has stayed within Chebyshev 3 of one
     *  tile for STALL_CAT rounds walks, for the rest of cooperation, to a cat in view (approachCat), else the cat last
     *  known, else the centre tile on its own side of each centre line; the turn's usual steps bite the cat and lay its
     *  cat trap. Cooperation points weigh cat damage 0.5. u2g's 144 field games: 39% of attack-move turns were stalled
     *  (within 3 tiles of the rat's tile 50 rounds before), 1380-1630 a game. averystrangespace vs AdamTan12_Icarus
     *  (5080/5081, 5335/5336): cooperation to r2000, kings 2 to 1, delivered even (4505 vs 4575), cat damage 0 to
     *  87-289, lost on points on both sides; one Icarus rat bit a cat 23 times (r127-149), while our 12 rushers stood
     *  in the dead end at (7-8, 22-26) from r500 to r1500. */
    public static final int STALL_CAT = 100;
    /** Rush spawns stop at this round (0 = never): the spawn index is the live-baby count, so without it every
     *  replacement born with 4-15 babies alive is an attacker all game (rsu1: s8u lost both DefaultSmall games to s7d
     *  in a long war, 7 rats to 13-15 at r150). */
    public static final int RUSH_UNTIL = 300;
    /** From this round on (0 = never) every baby bites and grabs enemy rats during cooperation too: the field rushers
     *  open the war at r36-40 (abrk, whose attackers must first walk over, at r66). */
    public static final int BREAK_ROUND = 30;
    /** Squared distance (to the cat's 2x2 centre, doubled coords /4) a gatherer keeps from cats. */
    public static final int GATHER_CAT_DIST = 20;
    /** A cat within dist2 CAT_DEFEND_R2 of one of our kings draws every rat that sees it, gatherers included
     *  (0 = off): attack_on_cat, a lone cat scratched our king to death in 113 rounds while 17 gatherers kept away. */
    public static final int CAT_DEFEND_R2 = 36;
    /** Dig-aware flow-field navigation, requested after FLOW_STALL turns without progress. */
    public static final int FLOW_STALL = 4, FLOW_TTL = 200, FLOW_RESERVE = 2500, FLOW_HOLD = 3, FLOW_REJOIN = 40, FLOW_RETRY = 30;
    /** Reverse flow search (target to rat): the rat walks on (greedy/bug) while it runs instead of standing still, and the
     *  path starts wherever the rat is when a tile beside it is reached. Hike: 35% of rat-turns stood still in
     *  searches of ~10 turns (621 bytecodes a node, ~15 nodes a turn). Searches give up after FLOW_REV_TURNS. */
    public static final int FLOW_REV_TURNS = 60;
    /** Path cost of a dirt tile (a dig is 5 cheese and 2.5 turns of action cooldown, plus the step). */
    public static final int DIG_COST = 4;
    /** Random samples per explore target, taking the first never-seen tile. */
    public static final int EXPLORE_TRIES = 12;
    /** Scouts: one in SCOUT_EVERY gatherers explores our half (never-seen tiles) until round SCOUT_UNTIL, then
     *  gathers (0 = off). */
    public static final int SCOUT_EVERY = 4, SCOUT_UNTIL = 150;
    /** Mine seeking (0 = off): a gatherer at its mine that sees SEEK_CROWD or more allied babies nearer that mine
     *  explores never-seen tiles of our half for SEEK_TTL rounds, at most once per SEEK_RETRY rounds, and reports what
     *  it finds (streetsofnewyork: two of our three mines sat behind 5-deep dirt bands, unseen all game, while
     *  17846 rat-rounds crowded the third). */
    public static final int SEEK_CROWD = 6, SEEK_TTL = 150, SEEK_RETRY = 200;
    /** Idle seeking (0 = off): a gatherer that spends SEEK_IDLE turns within Chebyshev SEEK_IDLE_R of its mine without
     *  a pickup seeks as SEEK_CROWD does (that test counts allies in the 90-degree cone only). wallsofparadis (3800):
     *  70% of our rat-turns crowded one mine while our three other home mines, behind dirt rings, went unseen to
     *  r392-503; the field reached their twins by r54-135 and took 12920 from them to our 770. */
    public static final int SEEK_IDLE = 20, SEEK_IDLE_R = 4;
    /** A baby digs only with this much cheese (global plus raw) or more. */
    public static final int DIG_MIN_CHEESE = 60;
    /** Live babies kept unconditionally (while CORE_FLOOR cheese remain) before the income test applies. */
    public static final int CORE_BABIES = 18, CORE_FLOOR = 300;
    /** On maps of CORE_SMALL_AREA tiles or fewer (0 = off) the core count is CORE_SMALL, not CORE_BABIES: past it
     *  every peacetime spawn passes the income test. Replica canyoudig vs Xzibit (8 z3x-line games, all lost at
     *  r288-339; in both z3x games the king starved): 19-20 babies at r50 to their 15-16, bank at r100 305-410 to
     *  800-895; z3x's spawns by r100 cost 1550-1870 at 40 each to their 1280-1360 at 32 (the price rises 10 a 4
     *  babies), deaths even (49-53 to 56-58 by r300); from r100 our bank sits under CORE_FLOOR + cost and the army
     *  falls to 0-2 by r300 while theirs holds 8-12. u5t (CORE_BABIES 12 on every map) went net -1 on the replica:
     *  of its 6 lost cells 5 were on maps of 900+ tiles (t_minus_five_minutes B the other); it gained tiny A. */
    public static final int CORE_SMALL = 12, CORE_SMALL_AREA = 441;
    /** Projected end-of-game cheese a spawn must leave. */
    public static final int SPAWN_MARGIN = 100;
    /** Fraction (percent) of measured income trusted when projecting forward. */
    public static final int INCOME_TRUST = 75;
    /** Rounds of income history in the rate estimate. */
    public static final int INCOME_WINDOW = 100;
    /** Projected surplus (cheese) a baby needs before placing a cat trap / biting with cheese. */
    public static final int TRAP_SURPLUS = 200, CHEESE_BITE_SURPLUS = 600;
    /** Cheese below which no trap is placed; cats within this squared distance of a king are trapped on the floor alone. */
    public static final int TRAP_FLOOR = 600, TRAP_DEFEND_DIST2 = 100;
    /** Optional per-turn work (trap-risk observation) needs this many bytecodes left. */
    public static final int BC_OPTIONAL = 8000;
    /** Below this many bytecodes left after sensing, end the turn rather than overrun. */
    public static final int BC_BAIL = 3500;
    /** The re-sense after moving (Baby.sense2) needs this many bytecodes left. IND_LEFT: the indicator string is
     *  built only with this many left (0 = always). Both are for a newborn's first turn on a big map: s9s overran
     *  by 3-11 there on superioristhmus (rz1, al1; field 3292, 3293), once at 17424 used before the indicator, once
     *  in the re-sense or flow work, and the next turn went to finishing it. Replica replays drop indicators. */
    public static final int SENSE2_LEFT = 6000, IND_LEFT = 1000;
    /** No formation site at war before this round (0 = off): the merge takes 7 babies from a war just begun (z0k's
     *  switch, here in a ledger hook). z1z's replica games at war still running at r250: 2 won, 12 lost with a second
     *  king formed by then; 51 won, 42 lost without (~/diag26b/z1zb*.jsonl). z0k (u9p, drifting bytecodes) pooled
     *  D 27, net +5, t 0.96 over three blocks. */
    public static final int FORM_WAR_FROM = 300;
    /** Rich formation fixes (z2a's FORM_OFF, FORM_FILL and FORM_RICH_RETRY), only for sites opened with two or more
     *  kings (the third king and on), under the bytecode ledger so games without such a site replay u6x's exactly:
     *  the first king does not walk within Chebyshev 5 of the open site; a site with one of our kings' bodies in its
     *  3x3 is dropped at once; a rich site that times out is banned and the next opens RICH_FIX_RETRY rounds later
     *  (u6x waits FORM_RETRY 150); a joiner one tile out steps onto any free tile of the 3x3, and a rat inside steps
     *  on along its entry heading. Field r2000 losses one more king flips (block files, 732 games): superioristhmus vs
     *  spsquared 7 times, 2-3 kings to their 3-4 with 6-14k banked; local z2a vs u9p on superioristhmus formed 12 of
     *  14 bursts to u9p's 6 of 13 (kings at r1190 4/4/4/3 vs 2/2/3/3). */
    public static final int RICH_FIX_RETRY = 20;
    /** Endgame cat damage (0 = off): from round END_CAT at war, a baby of any role that holds less than DELIVER_AT raw,
     *  carries no rat, is not needed at a threatened king and has no enemy rat within dist2 8 goes for the nearest seen
     *  or known cat (Baby.endCat), and the bite step bites it. At war the r2000 points are 0.5 kings, 0.3 cat damage and
     *  0.2 transfer shares. z0c (u9p + END_CAT 1650) pooled D 19, net +5, t 1.15 over two replica blocks, cat share 0.658
     *  to 0.584, and was rejected only by the transfer floor (set 1: 0.86 of the twin per round, floor 0.9). From r1850
     *  the errand covers 7.5% of a full game, so transfer can fall at most about that much. */
    public static final int END_CAT = 1850;
    /** Rescue formation (0 = off): while global cheese is under CASH_BANK (and at least CASH_FLOOR, the 50 a king costs
     *  plus a margin), a baby holding CASH_RAW+ raw cheese, Chebyshev CASH_FAR+ from every king, that sees CASH_N-1 more
     *  of our babies within Chebyshev CASH_R holding CASH_MIN raw with its own, none heavier, leads a formation on a
     *  clear 3x3 at or beside itself. It squeaks the site (SQ_CASH) and every baby that hears it within Chebyshev
     *  CASH_PULL walks into the 3x3 and relays it. The engine banks the raw cheese of every rat in the 3x3 when the
     *  centre promotes (becomeRatKing), so a stuck crowd's load reaches the bank without a walk home, and the new king
     *  stands where they were. A site lives CASH_TTL rounds from its start; then no rescue for CASH_TTL more.
     *  Replica starvation losses (435 dumped games): in 92 our last king starved, in 17 our babies held 300+ raw then.
     *  When the bank first fell under 150, a group of 7+ of ours within Chebyshev 5 held 320-935 raw in 11 games:
     *  fullofsecrets (outside the king's sealed pocket, 3-7 out), rose (the stem, 5-8 out), clockwisdom, uneasy_alliance.
     *  Local fullofsecrets (u9p vs araid A): attackers held 120-777 each around the centre mines from r400, 20 out. */
    public static final int CASH_BANK = 250, CASH_FLOOR = 60, CASH_RAW = 60, CASH_N = 3, CASH_R = 4, CASH_MIN = 250,
        CASH_FAR = 5, CASH_PULL = 8, CASH_TTL = 50;
    /** Endgame rescue (0 = off): from this round a lone king's babies run CASH's formation for a second king, with no
     *  raw cheese needed, while the bank keeps the new king alive to round 2000 (Baby.cashEnd). At r2000 kings are half
     *  of the war points (0.3 in cooperation): 1 king to their 1 splits 25-25, 2 to 1 is 33-17. u6x's END_KING needs
     *  400 banked and the king to name a site; 9 of the 12 lone-king replica points losses never had 400. */
    public static final int CASH_END = 1880;
    /** Last stand (0 = off): with 3 or fewer live babies (the baby cost at its base 10) every spawn floor drops to
     *  LAST_FLOOR, so a king without income spends its upkeep reserve on rats instead of starving on it. Replica dumps
     *  (lowstand.py, 27 games with 10+ rounds of <= 2 babies and >= 20 banked): lost 25; the bank then was 22-104, the
     *  LOW_FLOOR 100 plus a spawn's cost. averyfineline vs Icarus (18352, A): 2 rats and 100 banked at r120, 0 rats
     *  from r140 with 60, no spawn after r111, the king starved at r257. u6u (LOW_FLOOR 30 under 8 babies) went D 3
     *  (+2 -1) at screen zc3 and was dropped on its starved-king count (6 to 3). */
    public static final int LAST_FLOOR = 20;
    /** cashTick stops reading squeaks once fewer than this many bytecodes are left (1 = never: the read alone), z7u's
     *  line. A crowd on a rescue site squeaks it every turn, and z3x read them all: zc4 (z3x + RAW_TRAPS) overran 4 times
     *  on zd312, and on ov4 the overrun is z3x's own cashTick in the r1880 CASH_END recall crowd (rat 13481, r1884-1891,
     *  stage 6 to 7 costing 11k; no trap laid that turn). z7m's rats overran the same way on arrows (z7u's evidence). */
    public static final int CASH_READ_LEFT = 6000;

    // ---- Pass 2B fixes ----
    /** Pass 2B fix: count cheese the king picks up after moving (move()'s pickUp) as Econ income. false = zdb's original behaviour. */
    public static final boolean FIX_KING_PICKUP_INCOME = true;
    /** Pass 2B fix: the king's nearCheese skips piles under its own 3x3 (pickUp empties them this turn). false = zdb's original behaviour. */
    public static final boolean FIX_KING_NEAR_CHEESE = true;
    /** Pass 2B fix: the king bites an enemy king on any body tile in reach, not only its (never reachable) centre. false = zdb's original behaviour. */
    public static final boolean FIX_KING_BITE_KING = true;
    /** Pass 2B fix: dig-first is skipped only for a flow search toward t, not one toward an old target. false = zdb's original behaviour. */
    public static final boolean FIX_DIG_FIRST_TGT = true;
    /** Pass 2B fix: Flow.start resets the robot-blocked counter held. false = zdb's original behaviour. */
    public static final boolean FIX_FLOW_HELD = true;
    /** Pass 2B fix: ALERT keeps the round's most-threatened king instead of the last writer. false = zdb's original behaviour. */
    public static final boolean FIX_ALERT_MAX = true;
    /** Pass 2B fix: TrapRisk dates arrival to the round before (the move followed that round's observe). false = zdb's original behaviour. */
    public static final boolean FIX_TRAP_ARRIVED = true;
    /** Pass 2B fix: a carried or thrown baby neither updates its TrapRisk arrival nor clears marks. false = zdb's original behaviour. */
    public static final boolean FIX_TRAP_CARRIED = true;
    /** Pass 2B fix: a baby bites any enemy-king tile it can attack, not only the one nearest it. false = zdb's original behaviour. */
    public static final boolean FIX_BITE_KING_TILE = true;
    /** Pass 2B fix: a throw into a cat scores 200 (it kills the carried enemy), not -50. false = zdb's original behaviour. */
    public static final boolean FIX_THROW_CAT = true;
    /** Pass 2B fix: a carry from the engine's role swap restarts carryRound, so THROW_HOLD applies to it. false = zdb's original behaviour. */
    public static final boolean FIX_SWAP_HOLD = true;
    /** Pass 2B fix: an attacker forgets lastEKing once it senses that spot with no king in view. false = zdb's original behaviour. */
    public static final boolean FIX_EKING_FORGET = true;
    /** Pass 2B fix: wanderAround draws within Chebyshev 2 (atMine's radius), not 3, ending the 2-tile bounce. false = zdb's original behaviour. */
    public static final boolean FIX_WANDER_BOX = true;
    /** Pass 2B fix: a mine walk stalled past 25 turns bumps mineShift, so the re-pick is a different mine. false = zdb's original behaviour. */
    public static final boolean FIX_STALL_SHIFT = true;
    /** Pass 2B fix: a seek ends only on real news (a mine the king lacked), not on sensing a listed mine. false = zdb's original behaviour. */
    public static final boolean FIX_SEEK_NEWS = true;
    /** Pass 2B fix: a FORM_KING centre rat promotes when action-ready (not only movement-ready) and on its step-on turn. false = zdb's original behaviour. */
    public static final boolean FIX_FORM_PROMOTE = true;
}
