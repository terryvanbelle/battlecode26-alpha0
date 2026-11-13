package zdp;

import battlecode.common.*;

/** Rat king: spawns, collects, publishes shared state, defends, steps away from cats. */
public final class King {
    static int slot = -1;
    static MapLocation start;
    public static int spawned;

    static void claimSlot() {
        for (int k = 0; k < Comms.KING_SLOTS; k++) {
            int b = Comms.KING0 + 3 * k;
            if (Comms.read(b) == 0 || !Comms.fresh(Comms.read(b + 2), 3)) { slot = k; return; }
        }
        slot = 0;
    }

    public static void turn() throws GameActionException {
        RobotController rc = G.rc;
        spawnWhy = '-';
        if (slot < 0) {
            claimSlot();
            start = G.me;
            G.frame(start);
            if (G.round <= 1) initialSymmetry();
        }
        Debug.mark(41);
        Comms.readKings();
        Econ.startTurn();
        Comms.readMines();
        Debug.mark(42);
        sense();
        Debug.mark(43);
        readSqueaks();
        if (Cfg.SYM_MINELIST) Sym.listCheck(Comms.mines, Comms.numMines, true);
        if (Cfg.MINE_MIRROR) mirrorMines();
        if (Cfg.TRIP_HALF != 0) mirrorAll();
        publish();
        Debug.mark(44);

        RobotInfo[] enemies = rc.senseNearbyRobots(-1, G.them);
        RobotInfo[] cats = rc.senseNearbyRobots(-1, Team.NEUTRAL);
        if (Cfg.CAT_LANE) for (RobotInfo k : cats) markLane(k.location);

        int before = rc.getGlobalCheese();
        pickUp();
        Econ.received(rc.getGlobalCheese() - before);
        Debug.mark(45);
        // a fresh rat (100 HP, bites every turn) is worth far more than one king bite: spawn first
        int sp = spawned;
        spawn(cats, enemies);
        // a king formed from a baby (Cfg.FORM_MINE) passes on its old home king's tile: the half test and the
        // mirrored enemy start of its babies stay those of our first king, not of this mine
        boolean formed = Cfg.FORM_MINE && Baby.home != null;
        if ((Cfg.HOME_START || Cfg.MINE_RR || formed) && spawned > sp && cats.length == 0) {
            MapLocation o = Baby.origin != null ? Baby.origin : formed ? Baby.home : start;
            rc.squeak(Comms.squeak(Comms.SQ_START, o.x, o.y, Cfg.MINE_RR ? rrNext() : 0));
        }
        Debug.mark(46);
        if (!G.coop) bite(enemies);
        if (Cfg.KING_CAT_BITE) biteCat(cats);
        if (Cfg.PRESS_SHARED && !G.coop) noteEKing(enemies);
        if (!G.coop && enemies.length > 0) {
            int threat = 0;
            for (RobotInfo e : enemies)
                if (e.type != UnitType.RAT_KING && e.location.distanceSquaredTo(G.me) <= Cfg.KING_THREAT_D2) threat++;
            Comms.write(Comms.ALERT, (G.me.x + 1) | (Math.min(15, threat) << 6));
            Comms.write(Comms.ALERT + 1, G.me.y + 1);
            Comms.write(Comms.ALERT + 2, Comms.stamp());
        }
        if (Cfg.KING_TRAPS && !G.coop) ratTrap(enemies);
        if (Cfg.MOAT && !G.coop) Moat.king(enemies);
        if (Cfg.FORM_KING) formation();
        Debug.mark(47);
        if (!(Cfg.MOAT && Moat.up() && cats.length == 0)) move(cats, enemies);
        Debug.mark(48);
        publish();
        Comms.write(Comms.BUDGET, Math.max(0, Math.min(1023, Econ.surplus() / 4)));
        Econ.endTurn();
        if (Cfg.KING_PATH) KPath.tick();
    }

    static int mirroredN = -1, mirroredMask;
    /** Cfg.TRIP_HALF: the list length + 1 (0: never) and mask when mirrorAll last ran. */
    static int tripMirN, tripMirMask;

    /** Cfg.TRIP_HALF: with one symmetry left, list every listed mine's twin, on either half (the timed walks to both
     *  decide whose it is: uneruesansfin's (27,10), the twin of (32,49), went unlisted until r1600 when (32,49) was
     *  handed to the enemy). Runs when the list or the symmetry changed since the last run. */
    static void mirrorAll() {
        int s = Sym.mask;
        if (Integer.bitCount(s) != 1 || Comms.numMines + 1 == tripMirN && s == tripMirMask) return;
        for (int k = 0, n = Comms.numMines; k < n; k++) {
            MapLocation t = G.mirror(Comms.mines[k], s);
            Comms.addMine(t.x, t.y);
        }
        tripMirN = Comms.numMines + 1;
        tripMirMask = s;
    }

    /** Cfg.MINE_MIRROR: with one symmetry left, list each listed mine's twin when it lies on our half. Runs when the
     *  list or the symmetry changed since the last run. */
    static void mirrorMines() {
        int s = Sym.mask;
        if (Integer.bitCount(s) != 1 || Comms.numMines == mirroredN && s == mirroredMask) return;
        mirrorInto(start, s);
        mirroredN = Comms.numMines;
        mirroredMask = s;
    }

    /** List the twin under symmetry s of every listed mine that is nearer home than home's own twin is (with
     *  Cfg.MINE_ANY_COOP, every twin). */
    static void mirrorInto(MapLocation home, int s) {
        MapLocation away = G.mirror(home, s);
        for (int k = 0, n = Comms.numMines; k < n; k++) {
            MapLocation t = G.mirror(Comms.mines[k], s);
            if (Cfg.MINE_ANY_COOP || t.distanceSquaredTo(home) < t.distanceSquaredTo(away)) Comms.addMine(t.x, t.y);
        }
    }

    /** Round 1: our start and the enemy start are mirror images; a candidate that maps us onto ourselves is impossible. */
    static void initialSymmetry() {
        int bad = gapMask(G.me, Cfg.SYM_KING_GAP ? 2 : 0);
        if ((bad & Sym.HOR) != 0) Sym.eliminate(Sym.HOR);
        if ((bad & Sym.VER) != 0) Sym.eliminate(Sym.VER);
        if ((bad & Sym.ROT) != 0) Sym.eliminate(Sym.ROT);
    }

    /** The symmetries that mirror a king start c within Chebyshev g of itself (g 2: the two kings' bodies overlap). */
    static int gapMask(MapLocation c, int g) {
        int bad = 0;
        for (int s = 1; s <= 4; s <<= 1)
            if (Math.abs(G.mirX(c.x, s) - c.x) <= g && Math.abs(G.mirY(c.y, s) - c.y) <= g) bad |= s;
        return bad;
    }

    static MapLocation nearCheese;   // nearest cheese pile within the roam leash, this turn
    static boolean strict = Cfg.KING_STRICT;

    static void sense() throws GameActionException {
        MapInfo[] infos = G.rc.senseNearbyMapInfos();
        MapLocation me = G.rc.getLocation(), anchor = homeMine != null ? homeMine : start;
        nearCheese = null;
        int bd = Integer.MAX_VALUE;
        // the first turn sees every tile new and pays for Sym.init too (60x60: 13.7k + 3.6k of 20k), so it stops
        // KING_SENSE_RESERVE short of the limit rather than overrun; the rest is recorded next turn
        int floor = G.round <= 1 ? Cfg.KING_SENSE_RESERVE : 0;
        for (MapInfo mi : infos) {
            if (floor > 0 && Clock.getBytecodesLeft() + (G.debt += 8) - 6 < floor) break;
            MapLocation l = mi.getMapLocation();
            boolean mine = mi.hasCheeseMine();
            Sym.record(l.x, l.y, mi.isWall(), mi.isDirt(), mine);
            if (mine) Comms.addMine(l.x, l.y);
            if (Cfg.KING_ROAM > 0 && anchor != null && mi.getCheeseAmount() > 0 && G.cheb(l, anchor) <= Cfg.KING_ROAM) {
                int d = me.distanceSquaredTo(l);
                if (d < bd) { bd = d; nearCheese = l; }
            }
        }
    }

    static void readSqueaks() {
        // Kings act before the babies they spawned, so this round's squeaks mostly arrive after our turn:
        // read last round's too.
        Message[] ms = G.rc.readSqueaks(-1);
        int n = ms.length, used = 0;
        for (int i = 0; i < n && used < 16; i++) {
            if (ms[i].getRound() < G.round - 1) continue;
            used++;
            int m = ms[i].getBytes();
            switch (Comms.sqType(m)) {
                case Comms.SQ_MINE: Comms.addMine(Comms.sqX(m), Comms.sqY(m)); break;
                case Comms.SQ_SYM: Sym.merge(Comms.sqExtra(m) & 7); break;
                case Comms.SQ_EKING:
                    Comms.write(Comms.EKING, Comms.sqX(m) + 1);
                    Comms.write(Comms.EKING + 1, Comms.sqY(m) + 1);
                    Comms.write(Comms.EKING + 2, Comms.stamp());
                    break;
                case Comms.SQ_CAT:
                    noteCat(Comms.sqX(m), Comms.sqY(m));
                    if (Cfg.CAT_LANE) markLane(new MapLocation(Comms.sqX(m), Comms.sqY(m)));
                    break;
                default:
                    if (Cfg.BURN && Comms.sqType(m) == Comms.SQ_BURN) burnHeard = true;
                    if (Cfg.TRIP_HALF != 0 && Comms.sqType(m) == Comms.SQ_TRIP) Comms.noteTrip(Comms.sqX(m), Comms.sqY(m), Comms.sqExtra(m));
                    break;
            }
        }
    }

    static void noteCat(int x, int y) {
        // Slot 0 holds the most recent sighting; the older one shifts to slot 1 unless it is the same cat area.
        MapLocation c0 = Comms.readLoc(Comms.CAT0);
        if (c0 != null && (Math.abs(c0.x - x) > 6 || Math.abs(c0.y - y) > 6)) {
            Comms.write(Comms.CAT0 + 3, c0.x + 1);
            Comms.write(Comms.CAT0 + 4, c0.y + 1);
            Comms.write(Comms.CAT0 + 5, Comms.read(Comms.CAT0 + 2));
        }
        Comms.write(Comms.CAT0, x + 1);
        Comms.write(Comms.CAT0 + 1, y + 1);
        Comms.write(Comms.CAT0 + 2, Comms.stamp());
    }

    static void publish() {
        int b = Comms.KING0 + 3 * slot;
        Comms.write(b, G.rc.getLocation().x + 1);
        Comms.write(b + 1, G.rc.getLocation().y + 1);
        Comms.write(b + 2, Comms.stamp());
        int s = Comms.read(Comms.SYM);
        int mask = s == 0 ? Sym.mask : (s & Sym.mask);
        if (mask == 0) mask = Sym.mask;
        Sym.mask = mask;
        if (Cfg.BURN && (burnHeard || (s & Comms.SYM_BURN) != 0)) mask |= Comms.SYM_BURN;
        Comms.write(Comms.SYM, mask);
    }

    /** Cfg.RISK_ADAPT: an SQ_BURN reached us (an enemy rat trap fired on one of ours); publish sets Comms.SYM_BURN. */
    static boolean burnHeard;

    static void pickUp() throws GameActionException {
        MapLocation c = G.rc.getLocation();
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                MapLocation l = c.translate(dx, dy);
                if (G.rc.canPickUpCheese(l)) G.rc.pickUpCheese(l);
            }
    }

    /** An enemy king we see goes in the shared enemy-king slot (as a squeaked sighting does in readSqueaks). */
    static void noteEKing(RobotInfo[] enemies) {
        for (RobotInfo e : enemies) {
            if (e.type != UnitType.RAT_KING) continue;
            Comms.write(Comms.EKING, e.location.x + 1);
            Comms.write(Comms.EKING + 1, e.location.y + 1);
            Comms.write(Comms.EKING + 2, Comms.stamp());
            return;
        }
    }

    static void bite(RobotInfo[] enemies) throws GameActionException {
        if (!G.rc.isActionReady()) return;
        RobotInfo best = null;
        for (RobotInfo e : enemies) {
            if (!G.rc.canAttack(e.location)) continue;
            if (best == null || e.health < best.health) best = e;
        }
        if (best == null) return;
        int c = biteCheese(best.health, G.rc.getGlobalCheese());
        if (c > 0 && G.rc.canAttack(best.location, c)) G.rc.attack(best.location, c);
        else G.rc.attack(best.location);
    }

    /** Cfg.KING_CAT_BITE: bite the weakest cat in reach (nearest to our centre on equal health), any tile of its 2x2. */
    static void biteCat(RobotInfo[] cats) throws GameActionException {
        RobotController rc = G.rc;
        if (cats.length == 0 || !rc.isActionReady()) return;
        MapLocation c = rc.getLocation(), at = null;
        int bh = 0, bd = 0;
        for (RobotInfo k : cats) {
            MapLocation t = catTile(k.location);
            if (t == null) continue;
            int d = Baby.catDist4(k.location, c);
            if (at == null || betterCat(k.health, d, bh, bd)) { at = t; bh = k.health; bd = d; }
        }
        if (at != null) rc.attack(at);
    }

    /** A tile of the cat's 2x2 (bottom-left bl) the king can attack, or null. */
    static MapLocation catTile(MapLocation bl) {
        RobotController rc = G.rc;
        for (int dx = 0; dx <= 1; dx++)
            for (int dy = 0; dy <= 1; dy++) {
                MapLocation t = bl.translate(dx, dy);
                if (rc.canAttack(t)) return t;
            }
        return null;
    }

    /** Cat pick for KING_CAT_BITE: less health first, then nearer (dist4 from the cat's centre). */
    static boolean betterCat(int hp, int d4, int bhp, int bd4) {
        return hp < bhp || hp == bhp && d4 < bd4;
    }

    /** Cheese for a king bite on a target with hp: the least that kills now (at most KING_BITE_MAX), else 0. */
    static int biteCheese(int hp, int global) {
        int budget = Math.min(Cfg.KING_BITE_MAX, global - Cfg.KING_BITE_FLOOR);
        if (budget <= 0 || hp <= 10) return 0;
        int k = hp - 10;
        return k * k <= budget ? k * k : 0;
    }

    /** Most babies we keep: the war or peace cap, raised to RICH_BABIES while the bank holds RICH_BANK. */
    static int babyCap(boolean war, int bank) {
        int cap = war ? Cfg.MAX_BABIES_WAR : Cfg.MAX_BABIES;
        if (Cfg.RICH_MIN_AREA > 0 && G.W * G.H < Cfg.RICH_MIN_AREA) return cap;
        if (Cfg.RICH2_BABIES > 0 && bank >= Cfg.RICH2_BANK) return Math.max(Math.max(cap, Cfg.RICH_BABIES), Cfg.RICH2_BABIES);
        return Cfg.RICH_BABIES > cap && bank >= Cfg.RICH_BANK ? Cfg.RICH_BABIES : cap;
    }

    /** A spawn floor f: LOW_FLOOR under LOW_BABIES babies, and at most siege when that is positive (Cfg.SIEGE_FLOOR
     *  with enemies in the king's sight at war). */
    static int spawnFloor(int f, boolean low, int siege) {
        int r = low ? Cfg.LOW_FLOOR : f;
        return siege > 0 ? Math.min(r, siege) : r;
    }

    /** Cfg.TRAP_SPAWN_FIRST: a rat trap now would leave the king, under the baby cap, short of its next spawn. */
    static boolean trapStarvesSpawn(RobotController rc) {
        if (!Cfg.TRAP_SPAWN_FIRST) return false;
        int global = rc.getGlobalCheese();
        return starves(rc.getCurrentRatCost(), global, babyCap(!G.coop, global));
    }

    /** At rat cost `cost` (the cost step gives the baby count) and `global` cheese, a 20-cheese trap would leave
     *  less than the next spawn plus SPAWN_FLOOR while the count is under `cap`. */
    static boolean starves(int cost, int global, int cap) {
        return (cost - 10) / 10 * 4 < cap && global < cost + Econ.floor(Cfg.SPAWN_FLOOR) + 20;
    }

    /** The last spawn() outcome (Debug.KWHY): n not action-ready, s no surplus, f under the floor, c at the cap,
     *  b built, d no free tile (dug the ring). sawEnemy: enemies in view after cooperation. */
    static char spawnWhy = '-';
    static boolean sawEnemy;
    static final int[] rrCount = new int[3];

    /** Cfg.MINE_RR: the newborn's place in its role's spawn order, 1-based (the rat reads its role off the rat cost
     *  after this spawn, as we do here). */
    static int rrNext() throws GameActionException {
        int idx = (G.rc.getCurrentRatCost() - 10) / 10 * 4;
        int r = Baby.roleFor(idx, G.round);
        return ++rrCount[r] % 4096 + 1;
    }

    static void spawn(RobotInfo[] cats, RobotInfo[] enemies) throws GameActionException {
        RobotController rc = G.rc;
        sawEnemy = !G.coop && enemies.length > 0;
        spawnWhy = 'n';
        if (!rc.isActionReady()) return;
        int cost = rc.getCurrentRatCost();
        int babies = (cost - 10) / 10 * 4; // lower bound of live babies (cost step)
        // a few babies always (they pay for themselves on any map with cheese); beyond that the projected
        // end-of-game cheese at the measured income must stay positive after paying for this one
        // a core population always (rats pay for themselves and are the army); beyond it, the income test
        boolean low = babies < Cfg.LOW_BABIES;
        boolean siege = !G.coop && enemies.length > 0;
        int sf = siege ? Cfg.SIEGE_FLOOR : 0;
        int cdl = G.debt, cbl = G.bc();   // ledger (K 11: 9, iload last and ifeq)
        boolean last = Cfg.LAST_FLOOR > 0 && cost == 10;
        G.debt = cdl + G.bc() - cbl + 11;
        if (last) sf = Cfg.LAST_FLOOR;
        int cdc = G.debt, cbc = G.bc();   // ledger (K 9: iload core replaces the base's bipush)
        int core = Cfg.CORE_SMALL_AREA > 0 && G.W * G.H <= Cfg.CORE_SMALL_AREA ? Cfg.CORE_SMALL : Cfg.CORE_BABIES;
        G.debt = cdc + G.bc() - cbc + 9;
        boolean boot = babies < core && rc.getGlobalCheese() >= cost + Econ.floor(spawnFloor(Cfg.CORE_FLOOR, low, sf));
        // at war with enemies in sight, bodies now beat cheese later: spawn down to a short upkeep floor
        boolean war = siege && rc.getGlobalCheese() >= cost + Econ.floor(spawnFloor(Cfg.WAR_FLOOR, low, sf));
        spawnWhy = 's';
        if (!boot && !war && Econ.surplus() < cost + Cfg.SPAWN_MARGIN) return;
        // never spend the king's upkeep: SPAWN_FLOOR cheese (or what the rest of the game needs) stays
        spawnWhy = 'f';
        if (rc.getGlobalCheese() < cost + Econ.floor(Math.min(spawnFloor(Cfg.SPAWN_FLOOR, low, sf), 2 * (2000 - G.round)))) return;
        spawnWhy = 'c';
        if (babies >= babyCap(war, rc.getGlobalCheese())) return;
        // Spawn toward the most useful direction: away from visible cats, else toward the map centre.
        MapLocation c = rc.getLocation();
        MapLocation goal = new MapLocation(G.W / 2, G.H / 2);
        MapLocation cat = null;
        for (RobotInfo r : cats) { cat = r.location; break; }
        boolean centreGoal = G.coop || enemies.length == 0;
        if (!G.coop && enemies.length > 0) goal = enemies[0].location; // fresh rats join the fight at once
        MapLocation best = null;
        int bs = Integer.MIN_VALUE;
        for (int dx = -2; dx <= 2; dx++)
            for (int dy = -2; dy <= 2; dy++) {
                if (Math.abs(dx) < 2 && Math.abs(dy) < 2) continue;
                MapLocation l = c.translate(dx, dy);
                if (!rc.canBuildRat(l)) continue;
                int s = -l.distanceSquaredTo(goal);
                if (cat != null) s += 4 * l.distanceSquaredTo(cat);
                // ring corners are poorer delivery lanes; keep them as last choice
                if (Math.abs(dx) == 2 && Math.abs(dy) == 2) s -= 50;
                if (Cfg.SYM_CENTRE && centreGoal) {
                    // the same score x4 about the true centre (W / 2 sits half a tile off it on even sizes)
                    s = -G.centreD2x4(l) - 200 * (Math.abs(dx) == 2 && Math.abs(dy) == 2 ? 1 : 0);
                    if (cat != null) s += 4 * Baby.catDist4(cat, l);
                }
                s = s * 4 + (G.pickBit(c, l) ? 1 : 0);
                if (s > bs || Cfg.SYM_TIE && s == bs && G.tieFirst(c, l, best)) { bs = s; best = l; }
            }
        if (best != null) {
            // a newborn takes its king's facing: face the spawn tile, so it looks outward along a line its twin mirrors
            if (faceSpawn) { Direction f = c.directionTo(best); if (rc.getDirection() != f && rc.canTurn(f)) rc.turn(f); }
            rc.buildRat(best);
            spawned++;
            spawnWhy = 'b';
        } else {
            digRing(goal);
            spawnWhy = 'd';
        }
    }

    public static int traps;
    /** Cfg.SYM_FACE (a field, so tests can switch it). */
    static boolean faceSpawn = Cfg.SYM_FACE;
    /** Cfg.KING_FRONT (a field, so tests can set it). */
    static int kingFront = Cfg.KING_FRONT;

    /** War defense: a rat trap on the reachable tile nearest an approaching enemy rat (50 damage + stun). */
    static void ratTrap(RobotInfo[] enemies) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady() || rc.getGlobalCheese() < Econ.floor(Cfg.RAT_TRAP_MIN_CHEESE) || rc.getNumberRatTraps() >= Baby.trapCap()) return;
        if (trapStarvesSpawn(rc)) return;
        MapLocation c = rc.getLocation();
        MapLocation foe = null;
        int fd = Integer.MAX_VALUE;
        for (RobotInfo e : enemies) {
            if (e.type == UnitType.RAT_KING) continue;
            int d = c.distanceSquaredTo(e.location);
            if (d < fd) { fd = d; foe = e.location; }
        }
        boolean mid = false;
        if (foe == null || fd > Cfg.KING_TRAP_DIST2) {
            // fortress: no one close yet, so pre-lay a few traps on the side facing the map centre
            if (Cfg.FORT_TRAPS <= 0 || rc.getNumberRatTraps() >= Cfg.FORT_TRAPS) return;
            foe = new MapLocation(G.W / 2, G.H / 2);
            mid = Cfg.SYM_CENTRE;
        }
        MapLocation best = null;
        int bs = Integer.MAX_VALUE;
        for (int dx = -2; dx <= 2; dx++)
            for (int dy = -2; dy <= 2; dy++) {
                if (Math.abs(dx) < 2 && Math.abs(dy) < 2) continue;
                MapLocation l = c.translate(dx, dy);
                if (!rc.canPlaceRatTrap(l)) continue;
                // the trap fires when the enemy enters its 3x3: covering the tile next to the foe's line in
                int s2 = (mid ? G.centreD2x4(l) : l.distanceSquaredTo(foe)) * 2 + (G.pickBit(c, l) ? 1 : 0);
                if (s2 < bs || Cfg.SYM_TIE && s2 == bs && G.tieFirst(c, l, best)) { bs = s2; best = l; }
            }
        if (best != null) { rc.placeRatTrap(best); traps++; }
    }

    public static int digs;

    /** No spawn tile is open: dig the ring dirt tile nearest the goal (kings dig within range 8). */
    static void digRing(MapLocation goal) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady() || rc.getGlobalCheese() < 5 + 2 * 20) return;
        MapLocation c = rc.getLocation();
        MapLocation best = null;
        int bs = Integer.MAX_VALUE;
        for (int dx = -2; dx <= 2; dx++)
            for (int dy = -2; dy <= 2; dy++) {
                if (Math.abs(dx) < 2 && Math.abs(dy) < 2) continue;
                MapLocation l = c.translate(dx, dy);
                if (!rc.canRemoveDirt(l)) continue;
                int s = l.distanceSquaredTo(goal) * 2 + (G.pickBit(c, l) ? 1 : 0);
                if (s < bs || Cfg.SYM_TIE && s == bs && G.tieFirst(c, l, best)) { bs = s; best = l; }
            }
        if (best != null) { rc.removeDirt(best); digs++; }
    }

    static void move(RobotInfo[] cats, RobotInfo[] enemies) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isMovementReady()) return;
        MapLocation c = rc.getLocation();
        if (Cfg.KING_DODGE) {
            if (dodge(c, cats)) { mineWhy = 'c'; return; }
        } else for (RobotInfo cat : cats) {
            MapLocation b = cat.location;
            if (Cfg.SYM_CENTRE ? Baby.catDist4(b, c) <= 80 : c.distanceSquaredTo(b) <= 20) {
                Direction away = Cfg.SYM_CENTRE ? G.dirOf(2 * c.x - 2 * b.x - 1, 2 * c.y - 2 * b.y - 1) : b.directionTo(c);
                Direction[] cand = G.fan(c, away);
                for (Direction d : cand) if (rc.canMove(d)) { rc.move(d); return; }
            }
        }
        if (Cfg.CAT_LANE && offLane(c, cats)) { mineWhy = 'l'; return; }
        if (!Cfg.KING_MINE) return;
        if (Cfg.FORM_STILL && formSite != null) { mineWhy = 'f'; return; }
        int cd0 = G.debt, cb0 = G.bc();   // ledger (K 11: 9, iload off and ifeq)
        boolean off = Cfg.RICH_FIX && G.fixOn(Comms.numKings) && offSite(c);
        G.debt = cd0 + G.bc() - cb0 + 11;
        if (off) { mineWhy = 'f'; return; }
        if (enemies.length > 0 || cats.length > 0) { mineWhy = enemies.length > 0 ? 'e' : 'c'; return; }
        if (Cfg.KING_PLUG && unplug(c)) { mineWhy = 'u'; return; }
        if (Cfg.KING_PATH && KPath.follow(c)) return;
        if (homeMine == null) homeMine = pickHomeMine(start);
        // KING_FORAGE: a king bound to starve before round 2000, with no mine of its own, walks to the nearest home
        // mine no other king of ours holds, at any distance
        if (Cfg.KING_FORAGE && (homeMine == null || held(homeMine)) && forageDue(G.round, forageRound, Econ.surplus())) {
            forageRound = G.round;
            MapLocation f = forageMine(c);
            if (f != null) homeMine = f;
        }
        // a pile the king can reach is cheese no rat has to carry: walk until it is under our 3x3
        if (nearCheese != null && G.cheb(c, nearCheese) > 1 && walkStep(c, nearCheese)) { mineWhy = 'p'; pickUp(); return; }
        if (homeMine == null) { mineWhy = 'n'; return; }
        if (G.cheb(c, homeMine) <= 1) { mineWhy = 'a'; return; }
        mineWhy = 'm';
        if (!Cfg.KING_DIG) {
            if (walkStep(c, homeMine)) pickUp();
            return;
        }
        // a step that gets closer, else dig one (a sidestep that only keeps the distance jitters at a dirt face)
        if (closer(c, homeMine)) { pickUp(); return; }
        // blocked: tag why the king does not dig (x cooldown, y cheese, z no dirt-only step, q engine refused)
        if (!G.rc.isActionReady()) { mineWhy = 'x'; return; }
        if (G.rc.getGlobalCheese() < Cfg.KING_DIG_MIN) { mineWhy = 'y'; return; }
        MapLocation d = digTile(c, homeMine);
        if (d == null) { mineWhy = 'z'; if (walkStep(c, homeMine)) pickUp(); return; }
        if (!G.rc.canRemoveDirt(d)) { mineWhy = 'q'; return; }
        G.rc.removeDirt(d); digs++; Sym.mem[G.idx(d)] &= ~Sym.DIRT; mineWhy = 'D';
    }

    /** Lane tiles marked so far, and the lane under the king's 3x3 where move() found it (Cfg.CAT_LANE). */
    static int laneN, lane0;

    /** Mark a cat's 2x2 (bottom-left b) as lane (Cfg.CAT_LANE). */
    static void markLane(MapLocation b) {
        int[] mem = Sym.mem;
        for (int dx = 0; dx <= 1; dx++)
            for (int dy = 0; dy <= 1; dy++) {
                int x = b.x + dx, y = b.y + dy;
                if (x < 0 || y < 0 || x >= G.W || y >= G.H) continue;
                int i = x + G.W * y;
                if ((mem[i] & Sym.CAT) == 0) { mem[i] |= Sym.CAT; laneN++; }
            }
    }

    /** Lane tiles under a king's 3x3 centred at k (unrolled: a stuck king asks every turn). */
    static int laneCount(MapLocation k) {
        if (laneN == 0) return 0;
        int[] mem = Sym.mem;
        int w = G.W, x = k.x, y = k.y;
        if (x < 1 || y < 1 || x + 1 >= w || y + 1 >= G.H) {
            int n = 0;
            for (int xx = Math.max(0, x - 1); xx <= Math.min(w - 1, x + 1); xx++)
                for (int yy = Math.max(0, y - 1); yy <= Math.min(G.H - 1, y + 1); yy++)
                    if ((mem[xx + w * yy] & Sym.CAT) != 0) n++;
            return n;
        }
        int i = x - 1 + w * (y - 1), n = 0;
        if ((mem[i] & Sym.CAT) != 0) n++;
        if ((mem[i + 1] & Sym.CAT) != 0) n++;
        if ((mem[i + 2] & Sym.CAT) != 0) n++;
        i += w;
        if ((mem[i] & Sym.CAT) != 0) n++;
        if ((mem[i + 1] & Sym.CAT) != 0) n++;
        if ((mem[i + 2] & Sym.CAT) != 0) n++;
        i += w;
        if ((mem[i] & Sym.CAT) != 0) n++;
        if ((mem[i + 1] & Sym.CAT) != 0) n++;
        if ((mem[i + 2] & Sym.CAT) != 0) n++;
        return n;
    }

    /** A king step in d from move()'s tile adds no lane to its 3x3, or leaves less than lane0 (Cfg.CAT_LANE; always
     *  true off or with no lane known). */
    static boolean laneOk(MapLocation c, Direction d) {
        if (!Cfg.CAT_LANE || laneN == 0) return true;
        int n = laneCount(c.add(d));
        return n == 0 || n < lane0;
    }

    /** On lane: the step leaving the least lane under the 3x3 (ties: farther from the nearest seen cat, then hand). */
    static boolean offLane(MapLocation c, RobotInfo[] cats) throws GameActionException {
        int n0 = laneCount(c);
        lane0 = n0;
        if (n0 == 0) return false;
        RobotController rc = G.rc;
        int best = n0 * 1000, bi = -1;
        for (int i = 0; i < 8; i++) {
            Direction d = G.DIRS[i];
            if (!rc.canMove(d) || !unplugged(c, d)) continue;
            MapLocation nc = c.add(d);
            int nl = laneCount(nc);
            if (nl >= n0) continue;
            int s = nl * 1000 + (G.hand(c, d) > 0 ? 1 : 0);
            int cd = 1000;
            for (RobotInfo cat : cats) cd = Math.min(cd, Baby.catDist4(cat.location, nc));
            s -= Math.min(cd, 998) / 2 * 2;
            if (s < best) { best = s; bi = i; }
        }
        if (bi < 0) return false;
        rc.move(G.DIRS[bi]);
        G.me = rc.getLocation();
        lane0 = laneCount(G.me);
        return true;
    }

    /**
     * A cat walking to its waypoint scratches whatever blocks its next footprint, and stays put while it does: step the
     * 3x3 out of a near cat's next two footprints (then off its body, then away) when some step is safer than staying.
     */
    static boolean dodge(MapLocation c, RobotInfo[] cats) throws GameActionException {
        RobotController rc = G.rc;
        boolean near = false;
        for (RobotInfo cat : cats) if (c.distanceSquaredTo(cat.location) <= Cfg.KING_DODGE_R2) near = true;
        if (!near) return false;
        int best = 2 * catRisk(c, cats), bi = -1;
        for (int i = 0; i < 8; i++) {
            Direction d = G.DIRS[i];
            if (!rc.canMove(d)) continue;
            int s = 2 * catRisk(c.add(d), cats) + (G.hand(c, d) > 0 ? 1 : 0);
            if (s < best) { best = s; bi = i; }
        }
        if (bi >= 0) {
            rc.move(G.DIRS[bi]);
            G.me = rc.getLocation();
            return true;
        }
        // boxed in by dirt (a king's home is often a dirt pocket): dig the leading edge of the safest dirt-only step
        if (!Cfg.KING_DODGE_DIG || !rc.isActionReady() || rc.getGlobalCheese() < Cfg.KING_DIG_MIN) return false;
        MapLocation dig = null;
        for (int i = 0; i < 8; i++) {
            Direction d = G.DIRS[i];
            MapLocation t = dirtStep(c, d);
            if (t == null) continue;
            int s = 2 * catRisk(c.add(d), cats) + (G.hand(c, d) > 0 ? 1 : 0);
            if (s < best) { best = s; dig = t; }
        }
        if (dig == null || !rc.canRemoveDirt(dig)) return false;
        rc.removeDirt(dig);
        digs++;
        Sym.mem[G.idx(dig)] &= ~Sym.DIRT;
        mineWhy = 'C';
        return true;
    }

    /** The first dirt tile among those a king step in d adds to its 3x3, when the step adds dirt and no wall, unseen
     *  or off-map tile; else null. The added tiles are within dist2 8 of the king, its dig range. */
    static MapLocation dirtStep(MapLocation c, Direction d) {
        MapLocation nc = c.add(d), first = null;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                int x = nc.x + dx, y = nc.y + dy;
                if (Math.abs(x - c.x) <= 1 && Math.abs(y - c.y) <= 1) continue;
                if (x < 0 || y < 0 || x >= G.W || y >= G.H) return null;
                int m = Sym.mem[x + G.W * y];
                if ((m & Sym.SEEN) == 0 || (m & Sym.WALL) != 0) return null;
                if ((m & Sym.DIRT) != 0 && first == null) first = new MapLocation(x, y);
            }
        return first;
    }

    /** Lower is safer for a king centred at k: 250 per king tile in a cat's next footprint, 75 per tile in its second
     *  (the 3x3 is wider than the 2x2 lane, so leaving it can take two steps), 100 touching its body, less up to 100
     *  for distance from the body's centre. */
    static int catRisk(MapLocation k, RobotInfo[] cats) {
        int s = 0;
        for (RobotInfo cat : cats) {
            MapLocation b = cat.location;
            if (k.distanceSquaredTo(b) > Cfg.KING_DODGE_R2 + 16) continue;
            MapLocation n1 = b.add(cat.direction), n2 = n1.add(cat.direction);
            s += 250 * catOverlap(k, n1) + 75 * catOverlap(k, n2);
            if (catGap(k, b) <= 1) s += 100;
            s -= Math.min(100, Baby.catDist4(b, k));
        }
        return s;
    }

    /** Tiles shared by a king's 3x3 centred at k and a cat's 2x2 from bottom-left b. */
    static int catOverlap(MapLocation k, MapLocation b) {
        int ox = Math.min(k.x + 1, b.x + 1) - Math.max(k.x - 1, b.x) + 1;
        int oy = Math.min(k.y + 1, b.y + 1) - Math.max(k.y - 1, b.y) + 1;
        return ox > 0 && oy > 0 ? ox * oy : 0;
    }

    /** Chebyshev gap between a king's 3x3 centred at k and a cat's 2x2 from bottom-left b: 0 overlap, 1 touching. */
    static int catGap(MapLocation k, MapLocation b) {
        int gx = Math.max(0, Math.max(b.x - (k.x + 1), (k.x - 1) - (b.x + 1)));
        int gy = Math.max(0, Math.max(b.y - (k.y + 1), (k.y - 1) - (b.y + 1)));
        return Math.max(gx, gy);
    }

    /** The 16 tiles around a king's 3x3 at c, in ring order (corners at 0, 4, 8, 12). */
    static final int[] RX = {-2, -1, 0, 1, 2, 2, 2, 2, 2, 1, 0, -1, -2, -2, -2, -2},
                       RY = {-2, -2, -2, -2, -2, -1, 0, 1, 2, 2, 2, 2, 2, 1, 0, -1};

    /** Open arcs of the ring around a king's 3x3 at c (Cfg.KING_PLUG): a wall, dirt or off-map tile closes a ring tile,
     *  except a closed corner whose two ring neighbours are open (rats cut it diagonally). 1 with none closed, 0 with
     *  all closed; 2 or more means the body may cut one open area from another. */
    static int ringArcs(MapLocation c) {
        int[] mem = Sym.mem;
        int w = G.W, h = G.H, blk = Sym.WALL | Sym.DIRT, closedMask = 0;
        for (int i = 0; i < 16; i++) {
            int x = c.x + RX[i], y = c.y + RY[i];
            if (x < 0 || y < 0 || x >= w || y >= h || (mem[x + w * y] & blk) != 0) closedMask |= 1 << i;
        }
        for (int i = 0; i < 16; i += 4)
            if ((closedMask >> i & 1) != 0 && (closedMask >> ((i + 1) & 15) & 1) == 0 && (closedMask >> ((i + 15) & 15) & 1) == 0)
                closedMask &= ~(1 << i);
        if (closedMask == 0) return 1;
        int arcs = 0;
        for (int i = 0; i < 16; i++)
            if ((closedMask >> i & 1) == 0 && (closedMask >> ((i + 15) & 15) & 1) != 0) arcs++;
        return arcs;
    }

    /** Cfg.PLUG_FAST: ringArcs(c) with the ring's 16 tiles read unrolled when the ring lies on the map (ringArcs itself
     *  otherwise), the corner rule and the arc count in bit operations. Same answer as ringArcs. */
    static int ringArcsFast(MapLocation c) {
        int x = c.x, y = c.y, w = G.W;
        if (x < 2 || y < 2 || x + 2 >= w || y + 2 >= G.H) return ringArcs(c);
        int[] mem = Sym.mem;
        int blk = Sym.WALL | Sym.DIRT, w2 = w + w, w3 = w2 + w, w4 = w3 + w;
        int b = x - 2 + w * (y - 2), cm = 0;
        if ((mem[b] & blk) != 0) cm |= 1;
        if ((mem[b + 1] & blk) != 0) cm |= 2;
        if ((mem[b + 2] & blk) != 0) cm |= 4;
        if ((mem[b + 3] & blk) != 0) cm |= 8;
        if ((mem[b + 4] & blk) != 0) cm |= 16;
        if ((mem[b + 4 + w] & blk) != 0) cm |= 32;
        if ((mem[b + 4 + w2] & blk) != 0) cm |= 64;
        if ((mem[b + 4 + w3] & blk) != 0) cm |= 128;
        if ((mem[b + 4 + w4] & blk) != 0) cm |= 256;
        if ((mem[b + 3 + w4] & blk) != 0) cm |= 512;
        if ((mem[b + 2 + w4] & blk) != 0) cm |= 1024;
        if ((mem[b + 1 + w4] & blk) != 0) cm |= 2048;
        if ((mem[b + w4] & blk) != 0) cm |= 4096;
        if ((mem[b + w3] & blk) != 0) cm |= 8192;
        if ((mem[b + w2] & blk) != 0) cm |= 16384;
        if ((mem[b + w] & blk) != 0) cm |= 32768;
        return arcsOf(cm);
    }

    /** Cfg.PLUG_FAST: ringArcs's corner rule and arc count for a 16-bit closed mask in ring order. */
    static int arcsOf(int cm) {
        int open = ~cm & 0xFFFF;
        // a closed corner (bit 0, 4, 8, 12) with both ring neighbours open opens: bit i of nx is ring tile i+1 open,
        // of pv tile i-1 open
        int nx = (open >>> 1 | open << 15) & 0xFFFF, pv = (open << 1 | open >>> 15) & 0xFFFF;
        cm &= ~(cm & 0x1111 & nx & pv);
        if (cm == 0) return 1;
        // arcs: open tiles whose previous ring tile is closed
        int v = ~cm & (cm << 1 | cm >>> 15) & 0xFFFF;
        v = v - (v >>> 1 & 0x5555);
        v = (v & 0x3333) + (v >>> 2 & 0x3333);
        v = v + (v >>> 4) & 0x0F0F;
        return v + (v >>> 8) & 0x1F;
    }

    /** Cfg.PLUG_FAST: the ring arcs of the king's own tile, counted once a move (tile and round they were counted for). */
    static MapLocation arcsAt;
    static int arcsHere, arcsRound = -1;

    static int arcsHere(MapLocation c) {
        if (arcsRound != G.round || !c.equals(arcsAt)) { arcsAt = c; arcsRound = G.round; arcsHere = ringArcsFast(c); }
        return arcsHere;
    }

    /** Cfg.KING_PLUG: a move from c by d does not plug a passage (or plugs no worse than where the king stands). */
    static boolean unplugged(MapLocation c, Direction d) {
        if (!Cfg.KING_PLUG) return true;
        if (Cfg.PLUG_FAST) {
            if (KPath.left() < Cfg.PLUG_LEFT) return true;
            int a = ringArcsFast(c.add(d));
            return a <= 1 || a <= arcsHere(c);
        }
        int a = ringArcs(c.add(d));
        return a <= 1 || a <= ringArcs(c);
    }

    /** Cfg.KING_PLUG: standing where the ring splits, step to the neighbour with the fewest arcs (fewer than here). */
    static boolean unplug(MapLocation c) throws GameActionException {
        if (Cfg.PLUG_FAST && KPath.left() < Cfg.PLUG_LEFT) return false;
        RobotController rc = G.rc;
        int here = Cfg.PLUG_FAST ? arcsHere(c) : ringArcs(c);
        if (here <= 1) return false;
        int best = 2 * here, bi = -1;
        for (int i = 0; i < 8; i++) {
            Direction d = G.DIRS[i];
            if (!rc.canMove(d)) continue;
            int s = 2 * (Cfg.PLUG_FAST ? ringArcsFast(c.add(d)) : ringArcs(c.add(d))) + (G.hand(c, d) > 0 ? 1 : 0);
            if (s < best) { best = s; bi = i; }
        }
        if (bi < 0) return false;
        rc.move(G.DIRS[bi]);
        G.me = rc.getLocation();
        return true;
    }

    /** The walk to a pile or the home mine: strictly closer steps under Cfg.KING_STRICT, else step's (sidesteps allowed). */
    static boolean walkStep(MapLocation c, MapLocation t) throws GameActionException {
        return strict ? closer(c, t) : step(c, t);
    }

    /** One king step toward t that strictly shortens the distance: straight or 45 degrees off. */
    static boolean closer(MapLocation c, MapLocation t) throws GameActionException {
        RobotController rc = G.rc;
        Direction to = c.directionTo(t);
        int d0 = c.distanceSquaredTo(t);
        Direction[] cand = G.fan(c, to);
        for (Direction d : cand) {
            if (!rc.canMove(d) || c.add(d).distanceSquaredTo(t) >= d0 || !unplugged(c, d) || !laneOk(c, d)) continue;
            rc.move(d); G.me = rc.getLocation(); return true;
        }
        return false;
    }

    /** Dirt blocks the king's walk to t: a dirt tile of the step toward t (or 45 degrees off it, never farther from
     *  t) with the fewest dirt tiles and no wall, unseen or off-map tile among the tiles the step adds to the 3x3.
     *  Every added tile is within dist2 8 of the king, its dig range. Null when no step is blocked only by dirt. */
    static MapLocation digTile(MapLocation c, MapLocation t) {
        Direction to = c.directionTo(t);
        if (to == Direction.CENTER) return null;
        Direction[] cand = G.fan(c, to);
        MapLocation best = null;
        int bn = Integer.MAX_VALUE;
        for (Direction d : cand) {
            MapLocation nc = c.add(d);
            if (G.cheb(nc, t) > G.cheb(c, t) || !laneOk(c, d)) continue;
            int n = 0;
            MapLocation first = null;
            boolean ok = true;
            for (int dx = -1; dx <= 1 && ok; dx++)
                for (int dy = -1; dy <= 1; dy++) {
                    int x = nc.x + dx, y = nc.y + dy;
                    if (Math.abs(x - c.x) <= 1 && Math.abs(y - c.y) <= 1) continue;   // already under us
                    if (x < 0 || y < 0 || x >= G.W || y >= G.H) { ok = false; break; }
                    int m = Sym.mem[x + G.W * y];
                    if ((m & Sym.SEEN) == 0 || (m & Sym.WALL) != 0) { ok = false; break; }
                    if ((m & Sym.DIRT) != 0) {
                        n++;
                        if (first == null) first = new MapLocation(x, y);
                    }
                }
            if (ok && n > 0 && n < bn) { bn = n; best = first; }
        }
        return best;
    }

    /** One king step toward t; babies crowd a king, so sidestep up to 90 degrees off the line, never farther. */
    static boolean step(MapLocation c, MapLocation t) throws GameActionException {
        RobotController rc = G.rc;
        Direction to = c.directionTo(t);
        Direction[] cand = G.fan5(c, to);
        for (Direction d : cand) {
            if (!rc.canMove(d)) continue;
            if (G.cheb(c.add(d), t) > G.cheb(c, t) || !unplugged(c, d) || !laneOk(c, d)) continue;
            rc.move(d); G.me = rc.getLocation(); return true;
        }
        return false;
    }

    static MapLocation formSite;
    static int formStart = -1000, formKings;

    /** The first king names a formation site and keeps it fresh until the king cap is reached, a king forms there
     *  (Cfg.FORM_GAP) or it times out. */
    static void formation() throws GameActionException {
        if (slot != 0) return;
        int cdw = G.debt, cbw = G.bc();   // ledger (K 11: 9, iload w and ifeq)
        boolean w = Cfg.FORM_WAR_FROM > 0 && !G.coop && G.round < Cfg.FORM_WAR_FROM;
        G.debt = cdw + G.bc() - cbw + 11;
        if (w) { formSite = null; Comms.write(Comms.FORM_T, 0); return; }
        if (Comms.numKings >= Math.min(formCap(G.coop), Comms.KING_SLOTS) && !(Cfg.FORM_RICH_BANK > 0 && rich())) {
            // Cfg.RICH_GAP: a king that reaches the cap formed at our open site, so the gap starts now (the gap in the
            // open-site branch below is never reached for it)
            int cd0 = G.debt, cb0 = G.bc();   // ledger (K 11: 9, iload cg and ifeq)
            boolean cg = Cfg.RICH_GAP > 0 && formSite != null && formKings >= Cfg.RICH_GAP_KINGS && Comms.numKings > formKings
                && G.round >= Cfg.RICH_GAP_FROM;
            G.debt = cd0 + G.bc() - cb0 + 11;
            if (cg) formStart = G.round + Cfg.RICH_GAP - Cfg.FORM_TTL - Cfg.FORM_RETRY;
            formSite = null; Comms.write(Comms.FORM_T, 0); return;
        }
        if (formSite != null) {
            int cd1 = G.debt, cb1 = G.bc();   // ledger (K 13: 9, iload drop and ifeq, iload gap and ifeq)
            boolean drop = Cfg.RICH_FIX && G.fixOn(formKings) && Comms.numKings <= formKings
                && (kingOn(formSite) || G.round - formStart > Cfg.FORM_TTL && richRetry());
            boolean gap = Cfg.RICH_GAP > 0 && formKings >= Cfg.RICH_GAP_KINGS && Comms.numKings > formKings
                && G.round >= Cfg.RICH_GAP_FROM;
            G.debt = cd1 + G.bc() - cb1 + 13;
            if (drop) { richDrop(); return; }
            if (gap) { richGap(); return; }
            boolean formed = Cfg.FORM_GAP > 0 && Comms.numKings > formKings;
            if (formed || G.round - formStart > Cfg.FORM_TTL) {
                if ((Cfg.FORM_MINE || Cfg.END_KING_MSITE) && Comms.numKings <= formKings) ban(formSite);
                formSite = null;
                Comms.write(Comms.FORM_T, 0);
                if (formed) formStart = G.round + Cfg.FORM_GAP - Cfg.FORM_TTL - Cfg.FORM_RETRY;
                return;
            }
            Comms.write(Comms.FORM_T, Comms.stamp());
            return;
        }
        int babies = (G.rc.getCurrentRatCost() - 10) / 10 * 4;
        if (Cfg.END_KING > 0 && endKing(G.round, Comms.numKings, babies, G.rc.getGlobalCheese(), formStart)) {
            // the crowd count and the site checks cost several thousand: wait a turn rather than overrun (u0g, ZeroDay r1974)
            if (Cfg.END_KING_CROWD && Clock.getBytecodesLeft() + (G.debt += 8) - 6 < Cfg.END_KING_LEFT) return;
            int s = (Sym.mask & Sym.ROT) != 0 ? Sym.ROT : (Sym.mask & Sym.HOR) != 0 ? Sym.HOR : Sym.VER;
            // END_KING_MINE: a home mine first (its 5x5 is wall-free and its gatherers already wait there)
            // END_KING_MSITE: FORM_MINE's site from the babies' mine list (pickFormSite wants tiles the king has seen)
            formSite = Cfg.END_KING_MSITE ? mineSite(G.me, G.mirror(start, s)) : Cfg.END_KING_MINE ? pickFormSite(G.me) : null;
            if (formSite == null) formSite = Cfg.END_KING_CROWD ? crowdSite(G.me, G.mirror(start, s)) : ringSite(G.me, G.mirror(start, s));
            if (formSite == null) formSite = pickFormSite(G.me);
            if (formSite == null) return;
            formStart = G.round;
            formKings = Comms.numKings;
            Comms.write(Comms.FORM_X, formSite.x + 1);
            Comms.write(Comms.FORM_Y, formSite.y + 1);
            Comms.write(Comms.FORM_T, Comms.stamp());
            return;
        }
        int last = Cfg.FORM_LATE && Comms.numKings < 2 ? 1940 : 1150;
        if (G.round < Cfg.FORM_FROM || G.round > last || G.round - formStart < Cfg.FORM_TTL + Cfg.FORM_RETRY
                || babies < Cfg.FORM_BABIES || G.rc.getGlobalCheese() < Cfg.FORM_CHEESE) return;
        if (Cfg.FORM_RICH_BANK > 0 && rich()) {
            int cd2 = G.debt, cb2 = G.bc();   // ledger (K 11: 9, iload rb and ifeq)
            boolean rb = Cfg.RICH_FIX && richBans;
            G.debt = cd2 + G.bc() - cb2 + 11;
            if (rb) richPick();
            else {
                formSite = pickFormSite(G.me);
                if (formSite == null) {
                    int s = (Sym.mask & Sym.ROT) != 0 ? Sym.ROT : (Sym.mask & Sym.HOR) != 0 ? Sym.HOR : Sym.VER;
                    formSite = ringSite(G.me, G.mirror(start, s));
                }
            }
        }
        else if (formIncome()) formSite = pickFormSite(G.me);
        // Cfg.FORM_MINE: a king on a far mine is the cure for a low income (where_are_you: income 100-330 a hundred
        // rounds with all four home mines listed, 19-30 tiles out, and 2000 cheese lying at each by r1000), so a mine
        // site needs no income test
        else if (Cfg.FORM_MINE) {
            int s = (Sym.mask & Sym.ROT) != 0 ? Sym.ROT : (Sym.mask & Sym.HOR) != 0 ? Sym.HOR : Sym.VER;
            formSite = mineSite(G.me, G.mirror(start, s));
        }
        if (formSite == null) return;
        formStart = G.round;
        formKings = Comms.numKings;
        Comms.write(Comms.FORM_X, formSite.x + 1);
        Comms.write(Comms.FORM_Y, formSite.y + 1);
        Comms.write(Comms.FORM_T, Comms.stamp());
    }

    /** Kings we form up to: FORM_MAX, or FORM_MAX_WAR once cooperation is over (kings weigh 0.5 of points then, 0.3 before). */
    static int formCap(boolean coop) { return coop ? Cfg.FORM_MAX : Math.max(Cfg.FORM_MAX, Cfg.FORM_MAX_WAR); }

    /** Cfg.RICH_FIX: a site opened with two or more kings failed (timed out in rich() mode: banned, and the next opens
     *  RICH_FIX_RETRY rounds later) or has one of our kings' bodies in its 3x3 (dropped at once). */
    static boolean richBans;

    static void richDrop() {
        if (richRetry()) { ban(formSite); richBans = true; formStart = G.round + Cfg.RICH_FIX_RETRY - Cfg.FORM_TTL - Cfg.FORM_RETRY; }
        formSite = null;
        Comms.write(Comms.FORM_T, 0);
    }

    /** Cfg.RICH_GAP: a king formed at a rich site, so close it; the next site may open RICH_GAP rounds from now. */
    static void richGap() {
        formSite = null;
        Comms.write(Comms.FORM_T, 0);
        formStart = G.round + Cfg.RICH_GAP - Cfg.FORM_TTL - Cfg.FORM_RETRY;
    }

    static boolean richRetry() { return Cfg.FORM_RICH_BANK > 0 && rich(); }

    /** Cfg.RICH_FIX: the rich path's site pick once a rich site has been banned (banned sites skipped). */
    static void richPick() throws GameActionException {
        formSite = pickFormSite2(G.me);
        if (formSite == null) {
            int s = (Sym.mask & Sym.ROT) != 0 ? Sym.ROT : (Sym.mask & Sym.HOR) != 0 ? Sym.HOR : Sym.VER;
            formSite = ringSite2(G.me, G.mirror(start, s));
        }
    }

    /** Cfg.RICH_FIX: the open formation site (the first king's own; other kings read its heartbeat), or null. */
    static MapLocation openSite() {
        if (slot == 0) return formSite;
        int t = Comms.read(Comms.FORM_T), x = Comms.read(Comms.FORM_X);
        if (t == 0 || x == 0 || !Comms.fresh(t, 2)) return null;
        return new MapLocation(x - 1, Comms.read(Comms.FORM_Y) - 1);
    }

    /** Cfg.RICH_FIX: we stand within Chebyshev 5 of the open site (the first king does not walk then). */
    static boolean offSite(MapLocation c) {
        MapLocation fs = openSite();
        return fs != null && G.cheb(c, fs) <= 5;
    }

    /** Cfg.RICH_FIX: one of our kings stands within Chebyshev 2 of s (its body is in the site's 3x3). */
    static boolean kingOn(MapLocation s) {
        for (int k = 0; k < Comms.numKings; k++) if (G.cheb(s, Comms.kings[k]) <= 2) return true;
        return G.cheb(s, G.me) <= 2;
    }

    /** Cfg.FORM_RICH_BANK: the bank pays for one more king before the cutoff (FORM_RICH_MAX kings at most). */
    static boolean rich() {
        boolean r = rich0();
        int cd = G.debt, cb = G.bc();   // ledger (K = 12: 9, the istore of the line above, iload r and ireturn)
        if (!r && Cfg.RICH_LATE > 0) r = lateRich(G.round, Comms.numKings, G.rc.getGlobalCheese());
        if (!r && Cfg.RICH_MORE > 0) r = moreRich(G.round, Comms.numKings, G.rc.getGlobalCheese());
        G.debt = cd + G.bc() - cb + 12;
        return r;
    }

    static boolean rich0() {
        return G.round <= 1150 && Comms.numKings < Math.min(Cfg.FORM_RICH_MAX, Comms.KING_SLOTS)
            && G.rc.getGlobalCheese() >= Cfg.FORM_RICH_BANK;
    }

    /** Cfg.RICH_LATE: from that round to the r1150 cut, two kings and a bank short of FORM_RICH_BANK still pay for a
     *  third while Econ projects RICH_LATE_SUR banked at r2000 with three kings' upkeep. */
    static boolean lateRich(int round, int kings, int bank) {
        return round >= Cfg.RICH_LATE && round <= 1150 && kings == 2 && bank >= Cfg.RICH_LATE_BANK
            && Econ.surplus(3) >= Cfg.RICH_LATE_SUR;
    }

    /** Cfg.RICH_MORE: three or four kings name one more site before the cut while the bank projected to r2000 with
     *  its upkeep stays RICH_MORE_SUR or more. */
    static boolean moreRich(int round, int kings, int bank) {
        return round >= Cfg.RICH_MORE && round <= 1150 && kings >= 3 && kings < Comms.KING_SLOTS
            && bank >= Cfg.RICH_MORE_BANK && Econ.surplus(kings + 1) >= Cfg.RICH_MORE_SUR;
    }

    /** Cfg.END_KING: a lone king's endgame site is due (babies from the rat cost, last site's start formStart). */
    static boolean endKing(int round, int kings, int babies, int cheese, int formStart) {
        return round >= Cfg.END_KING && kings < 2 && babies >= 7 && cheese >= Cfg.END_KING_CHEESE
            && round - formStart > Cfg.FORM_TTL;
    }

    /** The income test for one more king (Cfg.FORM_NET100). */
    static boolean formIncome() {
        if (Cfg.FORM_SURPLUS != Integer.MIN_VALUE) return Econ.surplus(Econ.kings() + 1) >= Cfg.FORM_SURPLUS;
        if (Cfg.FORM_NET100 == Integer.MIN_VALUE) return true;
        return Econ.rate100() * Cfg.INCOME_TRUST / 100 - 200 * (Econ.kings() + 1) >= Cfg.FORM_NET100;
    }

    /** The nearest clearly-ours mine more than 4 tiles from us with a clear 3x3, else a clear spot behind us (with
     *  Cfg.FORM_RING: a clear 3x3 just off our body). */
    static MapLocation pickFormSite(MapLocation from) throws GameActionException {
        int s = (Sym.mask & Sym.ROT) != 0 ? Sym.ROT : (Sym.mask & Sym.HOR) != 0 ? Sym.HOR : Sym.VER;
        MapLocation ek = G.mirror(start, s);
        if (Cfg.FORM_MINE) {
            MapLocation m = mineSite(from, ek);
            if (m != null) return m;
        }
        MapLocation best = null;
        int bd = Integer.MAX_VALUE;
        for (int i = 0; i < Comms.numMines; i++) {
            MapLocation m = Comms.mines[i];
            // clearly ours: at most half as far from our start as from theirs
            if (G.cheb(m, from) <= 4 || 4 * m.distanceSquaredTo(start) > m.distanceSquaredTo(ek) || !clear3(m)) continue;
            if (nearKing(m)) continue;
            int d = from.distanceSquaredTo(m);
            if (d < bd) { bd = d; best = m; }
        }
        if (best != null) return best;
        if (Cfg.FORM_RING) return ringSite(from, ek);
        // else a clear spot four tiles behind us, away from the enemy start
        Direction back = ek.directionTo(from);
        MapLocation b = from.translate(4 * back.dx, 4 * back.dy);
        return clear3(b) ? b : null;
    }

    /** Cfg.RICH_FIX: pickFormSite skipping banned sites. */
    static MapLocation pickFormSite2(MapLocation from) throws GameActionException {
        int s = (Sym.mask & Sym.ROT) != 0 ? Sym.ROT : (Sym.mask & Sym.HOR) != 0 ? Sym.HOR : Sym.VER;
        MapLocation ek = G.mirror(start, s);
        if (Cfg.FORM_MINE) {
            MapLocation m = mineSite(from, ek);
            if (m != null) return m;
        }
        MapLocation best = null;
        int bd = Integer.MAX_VALUE;
        for (int i = 0; i < Comms.numMines; i++) {
            MapLocation m = Comms.mines[i];
            // clearly ours: at most half as far from our start as from theirs
            if (G.cheb(m, from) <= 4 || 4 * m.distanceSquaredTo(start) > m.distanceSquaredTo(ek) || !clear3(m)) continue;
            if (nearKing(m) || banned(m)) continue;
            int d = from.distanceSquaredTo(m);
            if (d < bd) { bd = d; best = m; }
        }
        if (best != null) return best;
        if (Cfg.FORM_RING) return ringSite2(from, ek);
        // else a clear spot four tiles behind us, away from the enemy start
        Direction back = ek.directionTo(from);
        MapLocation b = from.translate(4 * back.dx, 4 * back.dy);
        return clear3(b) && !banned(b) ? b : null;
    }

    /** Cfg.FORM_MINE: the listed mine nearest {@code from}, more than 4 from it, safe enough (FORM_MINE_SAFE), held by no
     *  king and never failed as a site; mines with known dirt in their 3x3 come after the clean ones; ties by
     *  G.pickBit. */
    static MapLocation mineSite(MapLocation from, MapLocation ek) {
        MapLocation best = null;
        long bk = Long.MAX_VALUE;
        for (int i = 0; i < Comms.numMines; i++) {
            MapLocation m = Comms.mines[i];
            if (m.x < 1 || m.y < 1 || m.x > G.W - 2 || m.y > G.H - 2) continue;
            if (G.cheb(m, from) <= 4 || Cfg.FORM_MINE_SAFE * m.distanceSquaredTo(start) > m.distanceSquaredTo(ek)) continue;
            if (nearKing(m) || banned(m)) continue;
            long k = siteKey(dirt3(m), from.distanceSquaredTo(m), G.pickBit(from, m));
            if (k < bk || Cfg.SYM_TIE && k == bk && G.tieFirst(from, m, best)) { bk = k; best = m; }
        }
        return best;
    }

    /** mineSite's rank: clean before dirty, then nearer, then the tie bit. */
    static long siteKey(boolean dirty, int d2, boolean bit) {
        return ((dirty ? 1L : 0L) << 40) + ((long) d2 << 1) + (bit ? 0 : 1);
    }

    /** Some tile of the 3x3 around c is known to be dirt. */
    static boolean dirt3(MapLocation c) {
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                if ((Sym.mem[(c.x + dx) + G.W * (c.y + dy)] & Sym.DIRT) != 0) return true;
        return false;
    }

    /** Sites where a formation timed out without a king (Cfg.FORM_MINE); not named again. */
    static final MapLocation[] siteBans = new MapLocation[8];
    static int nBans;

    static void ban(MapLocation s) { if (s != null && nBans < siteBans.length && !banned(s)) siteBans[nBans++] = s; }

    static boolean banned(MapLocation s) {
        for (int i = 0; i < nBans; i++) if (siteBans[i].equals(s)) return true;
        return false;
    }

    /** Site centres 3 tiles out from the king (its body and the site's 3x3 then leave a one-tile lane between them),
     *  every site tile within the king's view; offsets are in the side frame, so mirrored kings scan alike. */
    static final int[][] RING = {{3, 0}, {0, 3}, {3, 1}, {1, 3}, {3, -1}, {-1, 3}, {0, -3}, {-3, 0}, {1, -3}, {-3, 1}, {-1, -3}, {-3, -1}};

    /** Cfg.FORM_RING: the clear 3x3 off our king's body farthest from the enemy start, or null. */
    static MapLocation ringSite(MapLocation c, MapLocation ek) throws GameActionException {
        int sx = 2 * c.x >= G.W - 1 ? 1 : -1, sy = 2 * c.y >= G.H - 1 ? 1 : -1;
        MapLocation best = null;
        int bd = -1;
        for (int[] o : RING) {
            MapLocation s = c.translate(o[0] * sx, o[1] * sy);
            int d = s.distanceSquaredTo(ek);
            if (d > bd && clearNow(s)) { bd = d; best = s; }
        }
        return best;
    }

    /** Cfg.RICH_FIX: ringSite skipping banned sites. */
    static MapLocation ringSite2(MapLocation c, MapLocation ek) throws GameActionException {
        int sx = 2 * c.x >= G.W - 1 ? 1 : -1, sy = 2 * c.y >= G.H - 1 ? 1 : -1;
        MapLocation best = null;
        int bd = -1;
        for (int[] o : RING) {
            MapLocation s = c.translate(o[0] * sx, o[1] * sy);
            int d = s.distanceSquaredTo(ek);
            if (d > bd && !banned(s) && clearNow(s)) { bd = d; best = s; }
        }
        return best;
    }

    /** Cfg.END_KING_CROWD: the clear 3x3 off our king's body with the most of our babies within Chebyshev 2 of its
     *  centre (ties: farther from the enemy start), or null. */
    static MapLocation crowdSite(MapLocation c, MapLocation ek) throws GameActionException {
        // our babies' offsets from the king, only those that can stand within 2 of a ring site (Chebyshev 5)
        RobotInfo[] mine = G.rc.senseNearbyRobots(-1, G.us);
        int[] rx = new int[mine.length], ry = new int[mine.length];
        int m = 0;
        for (RobotInfo r : mine) {
            int dx = r.location.x - c.x, dy = r.location.y - c.y;
            if (r.type != UnitType.BABY_RAT || dx < -5 || dx > 5 || dy < -5 || dy > 5) continue;
            rx[m] = dx; ry[m++] = dy;
        }
        int sx = 2 * c.x >= G.W - 1 ? 1 : -1, sy = 2 * c.y >= G.H - 1 ? 1 : -1;
        MapLocation best = null;
        int bn = -1, bd = -1;
        for (int[] o : RING) {
            int ox = o[0] * sx, oy = o[1] * sy, n = 0;
            for (int i = m; --i >= 0; ) {
                int ax = rx[i] - ox, ay = ry[i] - oy;
                if (ax >= -2 && ax <= 2 && ay >= -2 && ay <= 2) n++;
            }
            MapLocation s = c.translate(ox, oy);
            int d = s.distanceSquaredTo(ek);
            if (crowdBetter(n, d, bn, bd) && clearNow(s)) { bn = n; bd = d; best = s; }
        }
        return best;
    }

    static boolean crowdBetter(int n, int d, int bn, int bd) { return n > bn || n == bn && d > bd; }

    /** The 3x3 around s is on the map, in view, passable, and holds no cat or king. */
    static boolean clearNow(MapLocation s) throws GameActionException {
        if (s.x < 1 || s.y < 1 || s.x > G.W - 2 || s.y > G.H - 2) return false;
        RobotController rc = G.rc;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                MapLocation t = s.translate(dx, dy);
                if (!rc.canSenseLocation(t) || !rc.senseMapInfo(t).isPassable()) return false;
                RobotInfo r = rc.senseRobotAtLocation(t);
                if (r != null && r.type != UnitType.BABY_RAT) return false;
            }
        return true;
    }

    /** Some live king of ours is within Chebyshev 4 of m (it holds that mine). */
    static boolean nearKing(MapLocation m) {
        for (int k = 0; k < Comms.numKings; k++)
            if (G.cheb(m, Comms.kings[k]) <= 4) return true;
        return false;
    }

    /** The 3x3 around c is on the map and known free of walls and dirt. */
    static boolean clear3(MapLocation c) {
        if (c.x < 1 || c.y < 1 || c.x > G.W - 2 || c.y > G.H - 2) return false;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                int m = Sym.mem[(c.x + dx) + G.W * (c.y + dy)];
                if ((m & Sym.SEEN) == 0 || (m & (Sym.WALL | Sym.DIRT)) != 0) return false;
            }
        return true;
    }

    static MapLocation homeMine;
    static char mineWhy = '-';

    static int forageRound;          // 0 at start: FORAGE_FROM >= FORAGE_EVERY, so the first check is due

    /** Cfg.KING_FORAGE: time to look for a mine (from FORAGE_FROM, every FORAGE_EVERY rounds, projected bank < 0). */
    static boolean forageDue(int round, int last, int surplus) {
        return round >= Cfg.FORAGE_FROM && round - last >= Cfg.FORAGE_EVERY && surplus < 0;
    }

    /** Another live king of ours is within Chebyshev 2 of m. */
    static boolean held(MapLocation m) {
        for (int k = 0; k < Comms.numKings; k++)
            if (G.cheb(Comms.kings[k], m) <= 2 && !Comms.kings[k].equals(G.me)) return true;
        return false;
    }

    /** Cfg.KING_FORAGE: the listed mine nearest {@code from} on our half (by our start) that no other king holds. */
    static MapLocation forageMine(MapLocation from) {
        int s = (Sym.mask & Sym.ROT) != 0 ? Sym.ROT : (Sym.mask & Sym.HOR) != 0 ? Sym.HOR : Sym.VER;
        MapLocation ek = G.mirror(start, s);
        MapLocation best = null;
        int bd = Integer.MAX_VALUE;
        for (int i = 0; i < Comms.numMines; i++) {
            MapLocation m = Comms.mines[i];
            if (m.distanceSquaredTo(start) > m.distanceSquaredTo(ek) || held(m)) continue;
            int d = from.distanceSquaredTo(m);
            if (d < bd || Cfg.SYM_TIE && d == bd && G.tieFirst(from, m, best)) { bd = d; best = m; }
        }
        return best;
    }

    /** The known mine nearest our start, within KING_MINE_R2 and on our half: the king sits on it. */
    static MapLocation pickHomeMine(MapLocation from) {
        int s = (Sym.mask & Sym.ROT) != 0 ? Sym.ROT : (Sym.mask & Sym.HOR) != 0 ? Sym.HOR : Sym.VER;
        MapLocation ek = G.mirror(from, s);
        MapLocation best = null;
        int bd = Cfg.KING_MINE_R2 + 1;
        for (int i = 0; i < Comms.numMines; i++) {
            MapLocation m = Comms.mines[i];
            int d = from.distanceSquaredTo(m);
            int e = m.distanceSquaredTo(ek);
            if (d >= bd || d > e || d == e && !Cfg.MINE_TIES) continue;
            // KING_FRONT: not one near the bisector
            if (kingFront > 0 && 100 * (e - d) < kingFront * from.distanceSquaredTo(ek)) continue;
            // another king already holds it
            boolean held = false;
            for (int k = 0; k < Comms.numKings; k++)
                if (G.cheb(Comms.kings[k], m) <= 2 && !Comms.kings[k].equals(G.me)) held = true;
            if (!held) { bd = d; best = m; }
        }
        return best;
    }
}
