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
        DK.hold = false;   // net mode's KING_ACT A_NONE flag: this turn's only
        if (slot < 0) {
            claimSlot();
            start = G.me;
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
        publish();
        Debug.mark(44);

        RobotInfo[] enemies = rc.senseNearbyRobots(-1, G.them);
        RobotInfo[] cats = rc.senseNearbyRobots(-1, Team.NEUTRAL);

        int before = rc.getGlobalCheese();
        pickUp();
        if (!Cfg.FIX_KING_PICKUP_INCOME) Econ.received(rc.getGlobalCheese() - before);
        Debug.mark(45);
        // a fresh rat (100 HP, bites every turn) is worth far more than one king bite: spawn first
        try { spawn(cats, enemies); } finally { if (D.point == DK.KING_SPAWN) D.emit(); }
        Debug.mark(46);
        boolean dA = rc.isActionReady();   // KING_ACT decision point
        if (D.ON && dA) DK.kingAct(enemies, cats);
        try {
        if (D.NET && dA) DK.netKingAct(enemies, cats);
        if (!G.coop && !(D.NET && DK.hold)) bite(enemies);
        if (!G.coop && enemies.length > 0) {
            int threat = 0;
            for (RobotInfo e : enemies)
                if (e.type != UnitType.RAT_KING && e.location.distanceSquaredTo(G.me) <= Cfg.KING_THREAT_D2) threat++;
            // FIX_ALERT_MAX: a king that wrote earlier this round with at least our threat keeps the alert
            if (!(Cfg.FIX_ALERT_MAX && Comms.read(Comms.ALERT + 2) == Comms.stamp()
                  && (Comms.read(Comms.ALERT) >> 6) >= Math.min(15, threat))) {
                Comms.write(Comms.ALERT, (G.me.x + 1) | (Math.min(15, threat) << 6));
                Comms.write(Comms.ALERT + 1, G.me.y + 1);
                Comms.write(Comms.ALERT + 2, Comms.stamp());
            }
        }
        } finally { if (dA && D.point == DK.KING_ACT) { if (D.lab < 0) D.lab = DK.A_NONE; D.emit(); } }
        // KING_FORM decision point (the slot-0 king)
        boolean dF = slot == 0;
        if (D.ON && dF) DK.kingForm();
        try { if (!(D.NET && dF && DK.netKingForm())) formation(); } finally { if (dF && D.point == DK.KING_FORM) D.emit(); }
        Debug.mark(47);
        // KING_MOVE decision point
        boolean dM = rc.isMovementReady();
        if (D.ON && dM) DK.kingMove(cats, enemies);
        try { if (!(D.NET && dM && DK.netKingMove(cats, enemies))) move(cats, enemies); }
        finally { if (dM && D.point == DK.KING_MOVE) D.emit(); }
        Debug.mark(48);
        publish();
        Comms.write(Comms.BUDGET, Math.max(0, Math.min(1023, Econ.surplus() / 4)));
        Econ.endTurn();
    }

    /** Round 1: our start and the enemy start are mirror images; a candidate that maps us onto ourselves is impossible. */
    static void initialSymmetry() {
        int bad = gapMask(G.me, 0);
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
            if (anchor != null && mi.getCheeseAmount() > 0 && G.cheb(l, anchor) <= Cfg.KING_ROAM
                && !(Cfg.FIX_KING_NEAR_CHEESE && G.cheb(l, me) <= 1)) {
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
                case Comms.SQ_CAT: noteCat(Comms.sqX(m), Comms.sqY(m)); break;
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
        Comms.write(Comms.SYM, mask);
    }

    /** Picks up every pile under our 3x3 (FIX_KING_PICKUP_INCOME: and counts it as Econ income, every caller). */
    static void pickUp() throws GameActionException {
        int b = Cfg.FIX_KING_PICKUP_INCOME ? G.rc.getGlobalCheese() : 0;
        MapLocation c = G.rc.getLocation();
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                MapLocation l = c.translate(dx, dy);
                if (G.rc.canPickUpCheese(l)) G.rc.pickUpCheese(l);
            }
        if (Cfg.FIX_KING_PICKUP_INCOME) Econ.received(G.rc.getGlobalCheese() - b);
    }

    static void bite(RobotInfo[] enemies) throws GameActionException {
        if (!G.rc.isActionReady()) return;
        RobotInfo best = weakest(enemies);
        if (best == null) return;
        MapLocation at = weakAt;
        int c = biteCheese(best.health, G.rc.getGlobalCheese());
        if (c > 0 && G.rc.canAttack(at, c)) { G.rc.attack(at, c); D.lab = DK.A_BITE_KILL; }
        else { G.rc.attack(at); D.lab = DK.A_BITE; }
    }

    static MapLocation weakAt;   // weakest()'s tile to bite

    /** The enemy in attack reach with the least health (the first such in sense order), or null; weakAt is the tile
     *  to bite (its location; FIX_KING_BITE_KING: any reachable tile of an enemy king's 3x3). Also net mode's
     *  A_BITE_KILL / A_BITE target. */
    static RobotInfo weakest(RobotInfo[] enemies) {
        RobotInfo best = null;
        weakAt = null;
        for (RobotInfo e : enemies) {
            MapLocation t = e.location;
            if (Cfg.FIX_KING_BITE_KING && e.type == UnitType.RAT_KING) { if ((t = kingTile(t)) == null) continue; }
            else if (!G.rc.canAttack(t)) continue;
            if (best == null || e.health < best.health) { best = e; weakAt = t; }
        }
        return best;
    }

    /** A tile of the enemy king's 3x3 (centre ctr) we can attack, or null. */
    static MapLocation kingTile(MapLocation ctr) {
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                MapLocation t = ctr.translate(dx, dy);
                if (G.rc.canAttack(t)) return t;
            }
        return null;
    }

    /** Net option A_BITE_CAT: bite the weakest cat in reach (nearest to our centre on equal health), any tile of its 2x2. */
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
        if (at != null) { rc.attack(at); D.lab = DK.A_BITE_CAT; }
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

    /** Cat pick for biteCat: less health first, then nearer (dist4 from the cat's centre). */
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
        if (bank >= Cfg.RICH2_BANK) return Math.max(Math.max(cap, Cfg.RICH_BABIES), Cfg.RICH2_BABIES);
        return Cfg.RICH_BABIES > cap && bank >= Cfg.RICH_BANK ? Cfg.RICH_BABIES : cap;
    }

    /** A spawn floor f: LOW_FLOOR under LOW_BABIES babies, and at most siege when that is positive (Cfg.SIEGE_FLOOR
     *  with enemies in the king's sight at war). */
    static int spawnFloor(int f, boolean low, int siege) {
        int r = low ? Cfg.LOW_FLOOR : f;
        return siege > 0 ? Math.min(r, siege) : r;
    }

    /** The last spawn() outcome (Debug.KWHY): n not action-ready, s no surplus, f under the floor, c at the cap,
     *  b built, d no free tile (dug the ring), h net mode's S_HOLD (never in zdb). sawEnemy: enemies in view after cooperation. */
    static char spawnWhy = '-';
    static boolean sawEnemy;

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
        int sf = 0;
        int cdl = G.debt, cbl = G.bc();   // ledger (K 11: 9, iload last and ifeq)
        boolean last = cost == 10;
        G.debt = cdl + G.bc() - cbl + 11;
        if (last) sf = Cfg.LAST_FLOOR;
        int cdc = G.debt, cbc = G.bc();   // ledger (K 9: iload core replaces the base's bipush)
        int core = G.W * G.H <= Cfg.CORE_SMALL_AREA ? Cfg.CORE_SMALL : Cfg.CORE_BABIES;
        G.debt = cdc + G.bc() - cbc + 9;
        boolean boot = babies < core && rc.getGlobalCheese() >= cost + Econ.floor(spawnFloor(Cfg.CORE_FLOOR, low, sf));
        // at war with enemies in sight, bodies now beat cheese later: spawn down to a short upkeep floor
        boolean war = siege && rc.getGlobalCheese() >= cost + Econ.floor(spawnFloor(Cfg.WAR_FLOOR, low, sf));
        if (D.ON) DK.kingSpawn(cats, enemies, cost, babies, low, sf, core, boot, war);   // KING_SPAWN decision point
        if (D.NET && DK.netKingSpawn(cats, enemies)) return;
        spawnWhy = 's';
        if (!boot && !war && Econ.surplus() < cost + Cfg.SPAWN_MARGIN) { D.lab = DK.S_NO_SURPLUS; return; }
        // never spend the king's upkeep: SPAWN_FLOOR cheese (or what the rest of the game needs) stays
        spawnWhy = 'f';
        if (rc.getGlobalCheese() < cost + Econ.floor(Math.min(spawnFloor(Cfg.SPAWN_FLOOR, low, sf), 2 * (2000 - G.round)))) { D.lab = DK.S_FLOOR; return; }
        spawnWhy = 'c';
        if (babies >= babyCap(war, rc.getGlobalCheese())) { D.lab = DK.S_CAP; return; }
        // Spawn toward the most useful direction: away from visible cats, else toward the map centre.
        MapLocation c = rc.getLocation();
        MapLocation goal = new MapLocation(G.W / 2, G.H / 2);
        MapLocation cat = null;
        for (RobotInfo r : cats) { cat = r.location; break; }
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
                s = s * 4 + (G.pickBit(c, l) ? 1 : 0);
                if (s > bs) { bs = s; best = l; }
            }
        if (best != null) {
            D.lab = DK.S_BUILD_0 + DK.ring(best.x - c.x, best.y - c.y);
            build(best);
        } else {
            int dg0 = digs;
            digRing(goal);
            D.lab = digs > dg0 ? DK.S_DIG : DK.S_BLOCKED;
            spawnWhy = 'd';
        }
    }

    /** Build a rat on l (a tile canBuildRat allows): spawn's build tail, and net mode's S_BUILD_* executor. */
    static void build(MapLocation l) throws GameActionException {
        G.rc.buildRat(l);
        spawned++;
        spawnWhy = 'b';
    }

    /** Rat traps placed (net mode's A_TRAP_* options, DK.trapToward). */
    public static int traps;

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
                if (s < bs) { bs = s; best = l; }
            }
        if (best != null) { rc.removeDirt(best); digs++; }
    }

    static void move(RobotInfo[] cats, RobotInfo[] enemies) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isMovementReady()) return;
        MapLocation c = rc.getLocation();
        Direction flee = fleeDir(c, cats);
        if (flee != null) { D.lab = DK.K_CAT_FLEE; rc.move(flee); return; }
        if (enemies.length > 0 || cats.length > 0) { mineWhy = enemies.length > 0 ? 'e' : 'c'; D.lab = DK.K_THREAT_HOLD; return; }
        if (unplug(c)) { mineWhy = 'u'; D.lab = DK.K_UNPLUG; return; }
        if (homeMine == null) homeMine = pickHomeMine(start);
        // a pile the king can reach is cheese no rat has to carry: walk until it is under our 3x3
        if (nearCheese != null && G.cheb(c, nearCheese) > 1 && step(c, nearCheese)) { mineWhy = 'p'; D.lab = DK.K_PILE; pickUp(); return; }
        if (homeMine == null) { mineWhy = 'n'; D.lab = DK.K_NO_HOME; return; }
        if (G.cheb(c, homeMine) <= 1) { mineWhy = 'a'; D.lab = DK.K_AT_HOME; return; }
        mineWhy = 'm';
        // a step that gets closer, else dig one (a sidestep that only keeps the distance jitters at a dirt face)
        if (closer(c, homeMine)) { D.lab = DK.K_CLOSER; pickUp(); return; }
        // blocked: tag why the king does not dig (x cooldown, y cheese, z no dirt-only step, q engine refused)
        if (!G.rc.isActionReady()) { mineWhy = 'x'; D.lab = DK.K_DIG_WAIT; return; }
        if (G.rc.getGlobalCheese() < Cfg.KING_DIG_MIN) { mineWhy = 'y'; D.lab = DK.K_DIG_POOR; return; }
        MapLocation d = digTile(c, homeMine);
        if (d == null) { mineWhy = 'z'; D.lab = DK.K_STUCK; if (step(c, homeMine)) { D.lab = DK.K_SIDESTEP; pickUp(); } return; }
        if (!G.rc.canRemoveDirt(d)) { mineWhy = 'q'; D.lab = DK.K_DIG_REFUSED; return; }
        D.lab = DK.K_DIG; digAt(d, 'D');
    }

    /** The cat flee: the first movable direction of the fan straight away from the first cat within dist2 20 that
     *  has one, or null (move()'s K_CAT_FLEE exit and its net executor). */
    static Direction fleeDir(MapLocation c, RobotInfo[] cats) {
        RobotController rc = G.rc;
        for (RobotInfo cat : cats) {
            MapLocation b = cat.location;
            if (c.distanceSquaredTo(b) <= 20)
                for (Direction d : G.fan(c, b.directionTo(c))) if (rc.canMove(d)) return d;
        }
        return null;
    }

    /** Dig dirt tile t (canRemoveDirt allowed it), clear it in Sym.mem and tag the move's outcome why. */
    static void digAt(MapLocation t, char why) throws GameActionException {
        G.rc.removeDirt(t);
        digs++;
        Sym.mem[G.idx(t)] &= ~Sym.DIRT;
        mineWhy = why;
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
        if (!rc.isActionReady() || rc.getGlobalCheese() < Cfg.KING_DIG_MIN) return false;
        MapLocation dig = null;
        for (int i = 0; i < 8; i++) {
            Direction d = G.DIRS[i];
            MapLocation t = dirtStep(c, d);
            if (t == null) continue;
            int s = 2 * catRisk(c.add(d), cats) + (G.hand(c, d) > 0 ? 1 : 0);
            if (s < best) { best = s; dig = t; }
        }
        if (dig == null || !rc.canRemoveDirt(dig)) return false;
        digAt(dig, 'C');
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
        if (Flow.left() < Cfg.PLUG_LEFT) return true;
        int a = ringArcsFast(c.add(d));
        return a <= 1 || a <= arcsHere(c);
    }

    /** Cfg.KING_PLUG: standing where the ring splits, step to the neighbour with the fewest arcs (fewer than here). */
    static boolean unplug(MapLocation c) throws GameActionException {
        if (Flow.left() < Cfg.PLUG_LEFT) return false;
        RobotController rc = G.rc;
        int here = arcsHere(c);
        if (here <= 1) return false;
        int best = 2 * here, bi = -1;
        for (int i = 0; i < 8; i++) {
            Direction d = G.DIRS[i];
            if (!rc.canMove(d)) continue;
            int s = 2 * ringArcsFast(c.add(d)) + (G.hand(c, d) > 0 ? 1 : 0);
            if (s < best) { best = s; bi = i; }
        }
        if (bi < 0) return false;
        rc.move(G.DIRS[bi]);
        G.me = rc.getLocation();
        return true;
    }

    /** One king step toward t that strictly shortens the distance: straight or 45 degrees off. */
    static boolean closer(MapLocation c, MapLocation t) throws GameActionException {
        RobotController rc = G.rc;
        Direction to = c.directionTo(t);
        int d0 = c.distanceSquaredTo(t);
        Direction[] cand = G.fan(c, to);
        for (Direction d : cand) {
            if (!rc.canMove(d) || c.add(d).distanceSquaredTo(t) >= d0 || !unplugged(c, d)) continue;
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
            if (G.cheb(nc, t) > G.cheb(c, t)) continue;
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
            if (G.cheb(c.add(d), t) > G.cheb(c, t) || !unplugged(c, d)) continue;
            rc.move(d); G.me = rc.getLocation(); return true;
        }
        return false;
    }

    static MapLocation formSite;
    static int formStart = -1000, formKings;

    /** The first king names a formation site and keeps it fresh until the king cap is reached or it times out. */
    static void formation() throws GameActionException {
        if (slot != 0) return;
        int cdw = G.debt, cbw = G.bc();   // ledger (K 11: 9, iload w and ifeq)
        boolean w = !G.coop && G.round < Cfg.FORM_WAR_FROM;
        G.debt = cdw + G.bc() - cbw + 11;
        if (w) { closeForm(); D.lab = DK.F_WAR_OFF; return; }
        if (Comms.numKings >= Math.min(formCap(), Comms.KING_SLOTS) && !rich()) {
            D.lab = DK.F_CAP_OFF;
            closeForm(); return;
        }
        if (formSite != null) {
            if (G.round - formStart > Cfg.FORM_TTL) {
                D.lab = DK.F_DROP_TIMEOUT;
                closeForm();
                return;
            }
            D.lab = DK.F_HOLD;
            Comms.write(Comms.FORM_T, Comms.stamp());
            return;
        }
        int babies = (G.rc.getCurrentRatCost() - 10) / 10 * 4;
        if (endKing(G.round, Comms.numKings, babies, G.rc.getGlobalCheese(), formStart)) {
            formSite = ringSite(G.me, G.mirror(start, sym()));
            if (formSite == null) formSite = pickFormSite(G.me);
            if (formSite == null) { D.lab = DK.F_END_NONE; return; }
            D.lab = DK.formKind(formSite);
            openForm(formSite);
            return;
        }
        if (G.round < Cfg.FORM_FROM || G.round > 1150 || G.round - formStart < Cfg.FORM_TTL + Cfg.FORM_RETRY
                || babies < Cfg.FORM_BABIES || G.rc.getGlobalCheese() < Cfg.FORM_CHEESE) { D.lab = DK.F_IDLE; return; }
        if (rich()) {
            formSite = pickFormSite(G.me);
            if (formSite == null) formSite = ringSite(G.me, G.mirror(start, sym()));
        }
        else if (formIncome()) formSite = pickFormSite(G.me);
        else D.lab = DK.F_NO_INCOME;
        if (formSite == null) { if (D.lab < 0) D.lab = DK.F_NO_SITE; return; }
        D.lab = DK.formKind(formSite);
        openForm(formSite);
    }

    /** Open formation site s (no checks): formation's success tail, and net mode's DK.formOpen. */
    static void openForm(MapLocation s) {
        formSite = s;
        formStart = G.round;
        formKings = Comms.numKings;
        Comms.write(Comms.FORM_X, s.x + 1);
        Comms.write(Comms.FORM_Y, s.y + 1);
        Comms.write(Comms.FORM_T, Comms.stamp());
    }

    /** Close the formation site (none open is fine). */
    static void closeForm() {
        formSite = null;
        Comms.write(Comms.FORM_T, 0);
    }

    /** The first live symmetry (rotation, then horizontal, then vertical): the one the enemy start is mirrored by. */
    static int sym() { return (Sym.mask & Sym.ROT) != 0 ? Sym.ROT : (Sym.mask & Sym.HOR) != 0 ? Sym.HOR : Sym.VER; }

    /** Kings we form up to (war or peace alike). */
    static int formCap() { return Cfg.FORM_MAX; }

    /** Net option F_RICH_DROP: close the site. zdb's rich-fix ban (ban the site, retry RICH_FIX_RETRY rounds later)
     *  is off, so zdb never bans and neither does the net. */
    static void richDrop() {
        closeForm();
    }

    /** Net option F_RICH_GAP: close the site; the next may open at once (a gap of 0 rounds). */
    static void richGap() {
        closeForm();
        formStart = G.round - Cfg.FORM_TTL - Cfg.FORM_RETRY;
    }

    /** One of our kings stands within Chebyshev 2 of s (its body is in the site's 3x3): a KING_FORM feature. */
    static boolean kingOn(MapLocation s) {
        for (int k = 0; k < Comms.numKings; k++) if (G.cheb(s, Comms.kings[k]) <= 2) return true;
        return G.cheb(s, G.me) <= 2;
    }

    /** Cfg.FORM_RICH_BANK: the bank pays for one more king before the cutoff (FORM_RICH_MAX kings at most). */
    static boolean rich() {
        boolean r = rich0();
        int cd = G.debt, cb = G.bc();   // ledger (K = 12: 9, the istore of the line above, iload r and ireturn)
        G.debt = cd + G.bc() - cb + 12;
        return r;
    }

    static boolean rich0() {
        return G.round <= 1150 && Comms.numKings < Math.min(Cfg.FORM_RICH_MAX, Comms.KING_SLOTS)
            && G.rc.getGlobalCheese() >= Cfg.FORM_RICH_BANK;
    }

    /** Cfg.END_KING: a lone king's endgame site is due (babies from the rat cost, last site's start formStart). */
    static boolean endKing(int round, int kings, int babies, int cheese, int formStart) {
        return round >= Cfg.END_KING && kings < 2 && babies >= 7 && cheese >= Cfg.END_KING_CHEESE
            && round - formStart > Cfg.FORM_TTL;
    }

    /** The income test for one more king (Cfg.FORM_NET100). */
    static boolean formIncome() {
        return Econ.rate100() * Cfg.INCOME_TRUST / 100 - 200 * (Econ.kings() + 1) >= Cfg.FORM_NET100;
    }

    /** The nearest clearly-ours mine more than 4 tiles from us with a clear 3x3, else a clear spot behind us. */
    static MapLocation pickFormSite(MapLocation from) throws GameActionException {
        MapLocation ek = G.mirror(start, sym());
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
        // else a clear spot four tiles behind us, away from the enemy start
        Direction back = ek.directionTo(from);
        MapLocation b = from.translate(4 * back.dx, 4 * back.dy);
        return clear3(b) ? b : null;
    }

    /** Net option F_START_MSITE: the listed mine nearest {@code from}, more than 4 from it, safe enough (FORM_MINE_SAFE), held by no
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
            if (k < bk) { bk = k; best = m; }
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

    /** Banned formation sites (zdb's rich-fix ban, off: nothing bans today); mineSite and net option F_START_BACK
     *  skip them. */
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

    /** The clear 3x3 off our king's body farthest from the enemy start, or null. */
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

    /** Net option F_START_CROWD: the clear 3x3 off our king's body with the most of our babies within Chebyshev 2 of its
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

    /** Another live king of ours is within Chebyshev 2 of m. */
    static boolean held(MapLocation m) {
        for (int k = 0; k < Comms.numKings; k++)
            if (G.cheb(Comms.kings[k], m) <= 2 && !Comms.kings[k].equals(G.me)) return true;
        return false;
    }

    /** The known mine nearest our start, within KING_MINE_R2 and on our half: the king sits on it. */
    static MapLocation pickHomeMine(MapLocation from) {
        MapLocation ek = G.mirror(from, sym());
        MapLocation best = null;
        int bd = Cfg.KING_MINE_R2 + 1;
        for (int i = 0; i < Comms.numMines; i++) {
            MapLocation m = Comms.mines[i];
            int d = from.distanceSquaredTo(m);
            int e = m.distanceSquaredTo(ek);
            if (d >= bd || d >= e) continue;
            // another king already holds it
            boolean held = false;
            for (int k = 0; k < Comms.numKings; k++)
                if (G.cheb(Comms.kings[k], m) <= 2 && !Comms.kings[k].equals(G.me)) held = true;
            if (!held) { bd = d; best = m; }
        }
        return best;
    }
}
