package zdp;

import battlecode.common.*;

/**
 * King decision points (docs/decision-points.md): KING_MOVE (King.move), KING_SPAWN (King.spawn), KING_ACT (the action
 * block after spawn: King.bite) and KING_FORM (King.formation). D holds the record buffer;
 * this class holds the king points' ids, option constants, feature snapshots and (net mode, D.NET) option executors.
 *
 * Feature code has no side effects on bot state (no Comms.write, no King.arcsHere, no King.rich(),
 * no assignment to King fields) and never aborts the king's turn: an exception drops the record (point = 0).
 * Bytecode limits are off for this work, so these hooks are not credited to the G.debt ledger.
 */
public final class DK {
    // decision points (BABY_* use 1..9, KING_* 10..19)
    static final int KING_MOVE = 10, KING_SPAWN = 11, KING_ACT = 12, KING_FORM = 13;
    /** Pass 3: the king's end-of-turn shared-array / squeak / trap-removal edits, its value values (secondary slot),
     *  and the newborn's facing. */
    static final int KING_COMMS = 14, KING_VAL = 15, KING_TURN = 16;

    // feature counts per point (scripts/train_dp.py feature_counts)
    static final int KING_MOVE_FEATS = 48, KING_SPAWN_FEATS = 41, KING_ACT_FEATS = 38, KING_FORM_FEATS = 38,
        KING_COMMS_FEATS = 94, KING_VAL_FEATS = 22, KING_TURN_FEATS = 24;

    // KING_MOVE options: the exits of King.move in zdb's order, then primitives (never labelled by zdb)
    static final int K_DODGE = 0, K_CAT_FLEE = 1, K_LANE = 2, K_NO_MINE = 3, K_FORM_STILL = 4, K_OFF_SITE = 5,
        K_THREAT_HOLD = 6, K_UNPLUG = 7, K_PATH = 8, K_PILE = 9, K_NO_HOME = 10, K_AT_HOME = 11, K_WALK_HOME = 12,
        K_CLOSER = 13, K_DIG_WAIT = 14, K_DIG_POOR = 15, K_SIDESTEP = 16, K_STUCK = 17, K_DIG_REFUSED = 18, K_DIG = 19,
        K_STAY = 20, K_MOVE_N = 21, K_DIGSTEP_N = 29;
    static final int KING_MOVE_OPTIONS = 37;

    // KING_SPAWN options: spawn's exits after the boot/war tests, builds by ring tile (King.RX/RY order), primitives
    static final int S_NO_SURPLUS = 0, S_FLOOR = 1, S_CAP = 2, S_BUILD_0 = 3, S_DIG = 19, S_BLOCKED = 20, S_HOLD = 21;
    static final int KING_SPAWN_OPTIONS = 22;

    // KING_ACT options
    static final int A_NONE = 0, A_BITE_KILL = 1, A_BITE = 2, A_BITE_CAT = 3, A_TRAP_FOE = 4, A_TRAP_FORT = 5,
        A_MOAT_FILL = 6, A_BITE_NEAREST = 7, A_BITE_KING = 8, A_CAT_TRAP = 9, A_DIG_RING = 10;
    /** Pass 3: bite the weakest reachable enemy with cheese D.BITE_C[i] (A_BITE_C0 + i). */
    static final int A_BITE_C0 = 11;
    static final int KING_ACT_OPTIONS = 17;

    // KING_FORM options: formation's exits, the kinds of site it opens, then a site at each RING offset
    static final int F_WAR_OFF = 0, F_CAP_OFF = 1, F_CAP_GAP = 2, F_RICH_DROP = 3, F_RICH_GAP = 4, F_DROP_FORMED = 5,
        F_DROP_TIMEOUT = 6, F_HOLD = 7, F_END_WAIT = 8, F_END_NONE = 9, F_IDLE = 10, F_NO_INCOME = 11, F_NO_SITE = 12,
        F_START_MINE = 13, F_START_BACK = 14, F_START_RING = 15, F_START_CROWD = 16, F_START_MSITE = 17, F_RING_0 = 18;
    static final int KING_FORM_OPTIONS = 30;

    // KING_COMMS options (pass 3). KC_ZDB keeps zdb's writes of this turn (its label); the rest edit them. T terminal,
    // C continue (the loop reopens the point, at most KC_CHAIN times).
    static final int KC_ZDB = 0, KC_DROP_ALL = 1, KC_DROP0 = 2, KC_BUDGET0 = 9, KC_ALERT_CALL = 12, KC_ALERT_CLEAR = 13,
        KC_SQUEAK = 14, KC_RM_RAT = 15, KC_RM_CAT = 16, KC_SET0 = 17;
    static final int KING_COMMS_OPTIONS = 81;
    static final int KC_CHAIN = 12;
    static final int[] KC_BUDGET = {0, 100, 1023};
    /** KC_DROP0 + g reverts group g's pending writes: SLOT (1..16), ALERT (17..19), REG (20..22, the free registers
     *  R0-R2), FORM (0, 23, 63), MINE (24..55), CAT (56..61), BUDGET (62). */
    static final long[] KC_GROUP = {((1L << 17) - 1) & ~1L, 7L << 17, 7L << 20, 1L | 1L << 23 | 1L << 63,
        ((1L << 32) - 1) << 24, 63L << 56, 1L << 62};

    // KING_VAL options: a base-32 value
    static final int KING_VAL_OPTIONS = 32;

    // KING_TURN options (pass 3): keep the facing (zdb), or turn to G.DIRS[i] (KT_DIR0 + i)
    static final int KT_ZDB = 0, KT_DIR0 = 1;
    static final int KING_TURN_OPTIONS = 9;

    /** The ring index (King.RX/RY) of offset (dx, dy) from the king, or -1. spawn's 5x5-minus-3x3 loop covers exactly
     *  these 16 tiles. */
    static int ring(int dx, int dy) {
        for (int i = 0; i < 16; i++) if (King.RX[i] == dx && King.RY[i] == dy) return i;
        return -1;
    }

    /** The kind of a site formation just opened, from its Chebyshev distance to the king (G.me equals formation's
     *  `from`: the king has not moved yet this turn). ringSite sites are at 3 (RING offsets), the back spot at 4 (0 when
     *  the enemy start mirrors onto us and back is CENTER), mine-list sites at 5 or more. */
    static int formKind(MapLocation s) {
        int c = G.cheb(G.me, s);
        if (c == 3) return F_START_RING;
        if (c <= 4) return F_START_BACK;
        return F_START_MINE;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // features (recording)

    /** KING_MOVE features, before King.move's cascade (the king is movement-ready and has not moved this turn). */
    static void kingMove(RobotInfo[] cats, RobotInfo[] enemies) {
        D.open(KING_MOVE);
        try {
            RobotController rc = G.rc;
            MapLocation c = rc.getLocation();
            D.f(G.coop);
            D.f(G.round / 10);
            D.f(D.cap(rc.getGlobalCheese() / 10, 2000));
            D.f(Econ.surplus() / 10 + 32768);
            D.f(D.cap(Econ.rate100() / 10, 65535));
            D.f(rc.getHealth());
            D.f(rc.isActionReady());
            D.f(D.cap(cats.length, 4));
            int cd = 400;
            Direction cdir = Direction.CENTER;
            boolean flee = false;
            for (RobotInfo k : cats) {
                int d = c.distanceSquaredTo(k.location);
                if (d < cd) { cd = d; cdir = c.directionTo(k.location); }
                if (d <= 20) flee = true;
            }
            D.f(cd);
            D.f(cdir.ordinal());
            D.f(flee);
            D.f(King.catRisk(c, cats) + 1000);
            D.f(D.cap(enemies.length, 15));
            int ek = 0, ed = 100;
            Direction edir = Direction.CENTER;
            for (RobotInfo e : enemies) {
                if (e.type == UnitType.RAT_KING) ek++;
                int d = c.distanceSquaredTo(e.location);
                if (d < ed) { ed = d; edir = c.directionTo(e.location); }
            }
            D.f(ek);
            D.f(ed);
            D.f(edir.ordinal());
            D.f(D.cap(King.ringArcsFast(c), 16));
            int mv = 0, dm = 0;
            for (int i = 0; i < 8; i++) {
                if (rc.canMove(G.DIRS[i])) mv |= 1 << i;
                if (King.dirtStep(c, G.DIRS[i]) != null) dm |= 1 << i;
            }
            D.f(mv);
            D.f(dm);
            MapLocation nc = King.nearCheese;
            D.f(nc != null);
            D.f(nc != null ? D.cap(G.cheb(c, nc), 63) : 63);
            D.f(nc != null ? c.directionTo(nc).ordinal() : 8);
            D.f(King.homeMine != null);
            MapLocation hm = King.homeMine != null ? King.homeMine : King.pickHomeMine(King.start);
            D.f(hm != null);
            D.f(hm != null ? D.cap(G.cheb(c, hm), 63) : 63);
            D.f(hm != null ? c.directionTo(hm).ordinal() : 8);
            D.f(hm != null && King.held(hm));
            D.f(hm != null && King.digTile(c, hm) != null);
            D.f(rc.getGlobalCheese() >= Cfg.KING_DIG_MIN);
            D.f(King.formSite != null);
            D.f(King.slot);
            D.f(D.cap(Comms.numKings, 8));
            D.f(D.cap(Comms.numMines, 63));
            D.f(King.start != null ? D.cap(G.cheb(c, King.start), 63) : 63);
            D.f("-nsfcbdh".indexOf(King.spawnWhy) + 1);
            D.f(D.cap((rc.getCurrentRatCost() - 10) / 10 * 4, 255));
            D.f(c.x);
            D.f(c.y);
            D.f(G.W);
            D.f(G.H);
            D.kmini(c);   // pass 3 (40..47)
        } catch (Exception e) { D.point = 0; }
    }

    /** KING_SPAWN features, inside spawn after its pure pre-computation (cost .. war are spawn's locals). */
    static void kingSpawn(RobotInfo[] cats, RobotInfo[] enemies, int cost, int babies, boolean low, int sf, int core,
                          boolean boot, boolean war) {
        D.open(KING_SPAWN);
        try {
            RobotController rc = G.rc;
            MapLocation c = rc.getLocation();
            int g = rc.getGlobalCheese();
            D.f(G.coop);
            D.f(G.round / 10);
            D.f(D.cap(g / 10, 2000));
            int sur = Econ.surplus();
            D.f(sur / 10 + 32768);
            D.f(D.cap(Econ.rate100() / 10, 65535));
            D.f(D.cap(cost, 1000));
            D.f(D.cap(babies, 255));
            D.f(D.cap(King.spawned, 255));
            D.f(low);
            D.f(boot);
            D.f(war);
            D.f(cost == 10);
            D.f(core);
            D.f(D.cap(King.babyCap(war, g), 255));
            D.f(sur - cost - Cfg.SPAWN_MARGIN + 32768);
            D.f(g - cost - Econ.floor(Math.min(King.spawnFloor(Cfg.SPAWN_FLOOR, low, sf), 2 * (2000 - G.round))) + 32768);
            D.f(D.cap(enemies.length, 15));
            D.f(D.cap(cats.length, 4));
            D.f(!G.coop && enemies.length > 0 ? c.directionTo(enemies[0].location).ordinal() : 8);
            int cd = 400;
            Direction cdir = Direction.CENTER;
            for (RobotInfo k : cats) {
                int d = c.distanceSquaredTo(k.location);
                if (d < cd) { cd = d; cdir = c.directionTo(k.location); }
            }
            D.f(cd);
            D.f(cdir.ordinal());
            D.f(c.directionTo(new MapLocation(G.W / 2, G.H / 2)).ordinal());
            int bm = 0, dm = 0;
            for (int i = 0; i < 16; i++) {
                MapLocation l = c.translate(King.RX[i], King.RY[i]);
                if (rc.canBuildRat(l)) bm |= 1 << i;
                if (rc.canRemoveDirt(l)) dm |= 1 << i;
            }
            D.f(bm);
            D.f(dm);
            D.f(D.cap(Comms.numKings, 8));
            D.f(King.slot);
            D.f(D.cap(Comms.numMines, 63));
            D.f(King.homeMine != null ? D.cap(G.cheb(c, King.homeMine), 63) : 63);
            D.f(rc.getHealth());
            D.f(c.x);
            D.f(c.y);
            D.f(G.W);
            D.f(G.H);
            D.kmini(c);   // pass 3 (33..40)
        } catch (Exception e) { D.point = 0; }
    }

    /** KING_ACT features: the king is still action-ready after spawn. */
    static void kingAct(RobotInfo[] enemies, RobotInfo[] cats) {
        D.open(KING_ACT);
        try {
            RobotController rc = G.rc;
            MapLocation me = G.me;
            int g = rc.getGlobalCheese();
            D.f(G.coop);
            D.f(G.round / 10);
            D.f(rc.getHealth());
            D.f(D.cap(g, 65535));
            int babies = 0, kings = 0, threat = 0, nd = 100, reach = 0;
            boolean kingReach = false;
            RobotInfo weak = null;
            for (RobotInfo e : enemies) {
                boolean k = e.type == UnitType.RAT_KING;
                int d = e.location.distanceSquaredTo(me);
                if (k) kings++;
                else {
                    babies++;
                    if (d <= Cfg.KING_THREAT_D2) threat++;
                    if (d < nd) nd = d;
                }
                if (!rc.canAttack(e.location)) continue;
                reach++;
                if (k) kingReach = true;
                if (weak == null || e.health < weak.health) weak = e;
            }
            D.f(D.cap(babies, 15));
            D.f(D.cap(kings, 3));
            D.f(D.cap(threat, 15));
            D.f(nd);
            D.f(D.cap(reach, 8));
            D.f(weak != null ? weak.health : 0);
            D.f(weak != null && weak.type == UnitType.RAT_KING);
            int kc;
            D.f(weak != null && (kc = King.biteCheese(weak.health, g)) > 0 && rc.canAttack(weak.location, kc));
            D.f(kingReach);
            D.f(D.cap(cats.length, 3));
            int cd = 400;
            boolean catReach = false;
            for (RobotInfo k : cats) {
                int d = Baby.catDist4(k.location, me);
                if (d < cd) cd = d;
                if (King.catTile(k.location) != null) catReach = true;
            }
            D.f(cd);
            D.f(catReach);
            D.f(D.cap(rc.getNumberRatTraps(), 31));
            D.f(D.cap(Baby.trapCap(), 31));
            int nt = 0, np = 0, nr = 0;
            for (int i = 0; i < 16; i++) {
                MapLocation l = me.translate(King.RX[i], King.RY[i]);
                if (rc.canPlaceRatTrap(l)) nt++;
                if (rc.canPlaceDirt(l)) np++;
                if (rc.canRemoveDirt(l)) nr++;
            }
            D.f(nt);
            D.f(np);
            D.f(nr);
            D.f(D.cap(rc.getDirt(), 255));
            D.f(false);   // Moat.up(): never true (Cfg.MOAT off)
            D.f("-nsfcbdh".indexOf(King.spawnWhy));
            D.f(D.cap(King.spawned, 255));
            D.f(Comms.numKings);
            D.f(me.x);
            D.f(me.y);
            D.f(G.W);
            D.f(G.H);
            D.kmini(me);   // pass 3 (30..37)
        } catch (Exception e) { D.point = 0; }
    }

    /** KING_FORM features: the slot-0 king, before formation(). */
    static void kingForm() {
        D.open(KING_FORM);
        try {
            RobotController rc = G.rc;
            MapLocation me = G.me;
            int gc = rc.getGlobalCheese();
            int babies = (rc.getCurrentRatCost() - 10) / 10 * 4;
            int nk = Comms.numKings;
            D.f(G.coop);
            D.f(G.round);
            D.f(nk);
            MapLocation fs = King.formSite;
            D.f(fs != null);
            D.f(D.cap(G.round - King.formStart, 2000));
            D.f(King.formKings);
            D.f(D.cap(babies, 255));
            D.f(D.cap(gc, 65535));
            D.f(D.cap(Econ.rate100() / 10, 65535));
            D.f(D.cap(Econ.surplus() / 10 + 10000, 65535));
            D.f(!G.coop && G.round < Cfg.FORM_WAR_FROM);
            D.f(nk >= Math.min(King.formCap(), Comms.KING_SLOTS));
            D.f(King.rich0());
            D.f(King.endKing(G.round, nk, babies, gc, King.formStart));
            D.f(King.formIncome());
            D.f(!(G.round < Cfg.FORM_FROM || G.round > 1150
                || G.round - King.formStart < Cfg.FORM_TTL + Cfg.FORM_RETRY || babies < Cfg.FORM_BABIES || gc < Cfg.FORM_CHEESE));
            D.f(fs != null ? D.cap(G.cheb(me, fs), 63) : 63);
            D.f(fs != null && King.kingOn(fs));
            MapLocation pf = King.pickFormSite(me);
            int pc = pf != null ? G.cheb(me, pf) : 63;
            D.f(pf == null ? 0 : pc >= 5 ? 1 : pc == 4 ? 2 : 3);
            D.f(D.cap(pc, 63));
            int sx = 2 * me.x >= G.W - 1 ? 1 : -1, sy = 2 * me.y >= G.H - 1 ? 1 : -1, rm = 0;
            for (int i = 0; i < 12; i++)
                if (King.clearNow(me.translate(King.RING[i][0] * sx, King.RING[i][1] * sy))) rm |= 1 << i;
            D.f(rm);
            MapLocation ek = G.mirror(King.start, King.sym());
            D.f(D.cap(G.cheb(me, ek), 63));
            D.f(D.cap(Comms.numMines, 63));
            D.f(Integer.bitCount(Sym.mask));
            D.f(King.sawEnemy);
            int nb = 0;
            for (RobotInfo r : rc.senseNearbyRobots(-1, G.us)) if (r.type == UnitType.BABY_RAT) nb++;
            D.f(D.cap(nb, 63));
            D.f(me.x);
            D.f(me.y);
            D.f(G.W);
            D.f(G.H);
            D.kmini(me);   // pass 3 (30..37)
        } catch (Exception e) { D.point = 0; }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // net mode (D.NET): the net picks an option for the open point; its executor runs it. An executor returns false
    // when the option cannot act, and zdb's own code then runs as the fallback. Dead code in the recording build.

    static final float NET_T = D.NET_T;   // set in D (build_dpnet.py); the floor is per point (DNet.floor)

    /** KING_MOVE options masked out of the net head: Cfg-dead in zdb and always failing or meaningless here. */
    static final boolean[] MOVE_ALLOWED = new boolean[KING_MOVE_OPTIONS];
    static {
        for (int k = 0; k < KING_MOVE_OPTIONS; k++) MOVE_ALLOWED[k] = true;
        MOVE_ALLOWED[K_LANE] = MOVE_ALLOWED[K_NO_MINE] = MOVE_ALLOWED[K_OFF_SITE] = MOVE_ALLOWED[K_PATH] = false;
    }

    /** The net's option for the open point (D.F holds its features), or -1 (no net, no open point, nothing allowed). */
    static int netPick(int point, boolean[] allowed) {
        if (D.point != point) return -1;
        float[] z = DNet.logits(point, D.F, D.n);
        if (z == null) return -1;
        int k = DNet.sample(z, D.fit(allowed, z.length, DNet.reach(point)), NET_T, DNet.floor(point), D.alt(point));
        if (D.RL) D.take(false);   // king records go through D.emit (primary slot)
        return k;
    }

    /** KING_MOVE in net mode: true when the net's option acted (move() is then skipped this turn). */
    static boolean netKingMove(RobotInfo[] cats, RobotInfo[] enemies) {
        int k = netPick(KING_MOVE, MOVE_ALLOWED);
        if (k < 0) return false;
        try {
            if (!execKingMove(k, cats, enemies)) return false;
        } catch (GameActionException e) { return false; }
        D.lab = k;
        return true;
    }

    static boolean execKingMove(int k, RobotInfo[] cats, RobotInfo[] enemies) throws GameActionException {
        RobotController rc = G.rc;
        MapLocation c = rc.getLocation();
        switch (k) {
            case K_DODGE:
                if (King.dodge(c, cats)) { King.mineWhy = 'c'; return true; }
                return false;
            case K_CAT_FLEE: {
                // zdb's cat flee, G.me left stale as zdb does
                Direction d = King.fleeDir(c, cats);
                if (d == null) return false;
                rc.move(d);
                return true;
            }
            case K_FORM_STILL: return King.formSite != null;
            case K_UNPLUG:
                if (King.unplug(c)) { King.mineWhy = 'u'; return true; }
                return false;
            case K_PILE:
                if (King.nearCheese != null && G.cheb(c, King.nearCheese) > 1 && King.step(c, King.nearCheese)) {
                    King.mineWhy = 'p'; King.pickUp(); return true;
                }
                return false;
            case K_CLOSER: {
                MapLocation h = home();
                if (h != null && G.cheb(c, h) > 1 && King.closer(c, h)) { King.mineWhy = 'm'; King.pickUp(); return true; }
                return false;
            }
            case K_WALK_HOME:
            case K_SIDESTEP: {
                MapLocation h = home();
                if (h != null && G.cheb(c, h) > 1 && King.step(c, h)) { King.mineWhy = 'z'; King.pickUp(); return true; }
                return false;
            }
            case K_DIG: {
                MapLocation h = home();
                if (h == null || G.cheb(c, h) <= 1) return false;
                MapLocation t = King.digTile(c, h);
                return t != null && dig(t);
            }
            case K_THREAT_HOLD: case K_NO_HOME: case K_AT_HOME: case K_DIG_WAIT: case K_DIG_POOR:
            case K_STUCK: case K_DIG_REFUSED: case K_STAY:
                return true;
            default:
                if (k >= K_MOVE_N && k < K_MOVE_N + 8) {
                    Direction d = G.DIRS[k - K_MOVE_N];
                    if (!rc.canMove(d)) return false;
                    rc.move(d);
                    G.me = rc.getLocation();
                    King.pickUp();
                    return true;
                }
                if (k >= K_DIGSTEP_N && k < K_DIGSTEP_N + 8) {
                    MapLocation t = King.dirtStep(c, G.DIRS[k - K_DIGSTEP_N]);
                    return t != null && dig(t);
                }
                return false;
        }
    }

    /** zdb's lazy home-mine pick (move() assigns it mid-cascade; an executor that acts skips that). */
    static MapLocation home() {
        if (King.homeMine == null) King.homeMine = King.pickHomeMine(King.start);
        return King.homeMine;
    }

    /** move()'s dig of a dirt tile t, without its KING_DIG_MIN policy guard (canRemoveDirt covers cooldown and cost). */
    static boolean dig(MapLocation t) throws GameActionException {
        if (!G.rc.canRemoveDirt(t)) return false;
        King.digAt(t, 'D');
        return true;
    }

    /** KING_SPAWN in net mode (called inside spawn right after the features): true when the option acted, and spawn
     *  then returns. */
    static boolean netKingSpawn(RobotInfo[] cats, RobotInfo[] enemies) {
        int k = netPick(KING_SPAWN, null);
        if (k < 0) return false;
        try {
            if (!execKingSpawn(k, cats, enemies)) return false;
        } catch (GameActionException e) { return false; }
        D.lab = k;
        return true;
    }

    static boolean execKingSpawn(int k, RobotInfo[] cats, RobotInfo[] enemies) throws GameActionException {
        RobotController rc = G.rc;
        switch (k) {
            case S_NO_SURPLUS: King.spawnWhy = 's'; return true;
            case S_FLOOR: King.spawnWhy = 'f'; return true;
            case S_CAP: King.spawnWhy = 'c'; return true;
            case S_DIG: {
                int d0 = King.digs;
                MapLocation goal = !G.coop && enemies.length > 0 ? enemies[0].location : new MapLocation(G.W / 2, G.H / 2);
                King.digRing(goal);
                if (King.digs == d0) return false;
                King.spawnWhy = 'd';
                return true;
            }
            case S_BLOCKED: King.spawnWhy = 'd'; return true;
            case S_HOLD: King.spawnWhy = 'h'; return true;
            default: {
                int i = k - S_BUILD_0;
                if (i < 0 || i >= 16) return false;
                MapLocation c = rc.getLocation(), l = c.translate(King.RX[i], King.RY[i]);
                if (!rc.isActionReady() || !rc.canBuildRat(l)) return false;
                King.build(l);
                return true;
            }
        }
    }

    /** KING_ACT net mode: A_NONE sets hold, which skips zdb's bite for the turn (King.turn clears it each turn). */
    static boolean hold;

    static void netKingAct(RobotInfo[] enemies, RobotInfo[] cats) {
        hold = false;
        int k = netPick(KING_ACT, actOk(enemies));
        if (k < 0) return;
        try {
            if (execKingAct(k, enemies, cats)) D.lab = k;
        } catch (GameActionException e) { /* zdb's lines run as the fallback */ }
    }

    static boolean execKingAct(int k, RobotInfo[] enemies, RobotInfo[] cats) throws GameActionException {
        RobotController rc = G.rc;
        MapLocation me = G.me;
        switch (k) {
            case A_NONE: hold = true; return true;
            case A_BITE_KILL: case A_BITE: {
                RobotInfo b = King.weakest(enemies);
                if (b == null) return false;
                MapLocation at = King.weakAt;
                if (k == A_BITE) { rc.attack(at); return true; }
                int c = King.biteCheese(b.health, rc.getGlobalCheese());
                if (c <= 0 || !rc.canAttack(at, c)) return false;
                rc.attack(at, c);
                return true;
            }
            case A_BITE_CAT: King.biteCat(cats); return !rc.isActionReady();
            case A_TRAP_FOE: {
                MapLocation foe = null;
                int fd = Integer.MAX_VALUE;
                for (RobotInfo e : enemies) {
                    if (e.type == UnitType.RAT_KING) continue;
                    int d = me.distanceSquaredTo(e.location);
                    if (d < fd) { fd = d; foe = e.location; }
                }
                return foe != null && trapToward(foe, false);
            }
            case A_TRAP_FORT: return trapToward(new MapLocation(G.W / 2, G.H / 2), false);
            case A_MOAT_FILL: {
                MapLocation near = null;
                int nd = Integer.MAX_VALUE;
                for (RobotInfo e : enemies) {
                    if (e.type != UnitType.BABY_RAT) continue;
                    int d = me.distanceSquaredTo(e.location);
                    if (d < nd) { nd = d; near = e.location; }
                }
                return Moat.fill(me, 8, near);
            }
            case A_BITE_NEAREST: {
                RobotInfo b = null;
                int bd = Integer.MAX_VALUE;
                for (RobotInfo e : enemies) {
                    if (!rc.canAttack(e.location)) continue;
                    int d = me.distanceSquaredTo(e.location);
                    if (d < bd) { bd = d; b = e; }
                }
                if (b == null) return false;
                rc.attack(b.location);
                return true;
            }
            case A_BITE_KING:
                for (RobotInfo e : enemies)
                    if (e.type == UnitType.RAT_KING && rc.canAttack(e.location)) { rc.attack(e.location); return true; }
                return false;
            case A_CAT_TRAP: {
                MapLocation goal = null;
                int cd = Integer.MAX_VALUE;
                for (RobotInfo e : cats) {
                    int d = me.distanceSquaredTo(e.location);
                    if (d < cd) { cd = d; goal = e.location; }
                }
                return trapToward(goal != null ? goal : new MapLocation(G.W / 2, G.H / 2), true);
            }
            case A_DIG_RING:
                King.digRing(new MapLocation(G.W / 2, G.H / 2));
                return !rc.isActionReady();
            default: {
                if (k < A_BITE_C0 || k >= A_BITE_C0 + 6) return false;
                MapLocation at = weakestLoc(enemies);
                int c = D.BITE_C[k - A_BITE_C0];
                if (at == null || !rc.canAttack(at, c)) return false;
                rc.attack(at, c);
                return true;
            }
        }
    }

    /** King.weakest's target tile without its side effect (King.weakAt untouched), or null. */
    static MapLocation weakestLoc(RobotInfo[] enemies) {
        RobotInfo best = null;
        MapLocation at = null;
        for (RobotInfo e : enemies) {
            MapLocation t = e.location;
            if (Cfg.FIX_KING_BITE_KING && e.type == UnitType.RAT_KING) { if ((t = King.kingTile(t)) == null) continue; }
            else if (!G.rc.canAttack(t)) continue;
            if (best == null || e.health < best.health) { best = e; at = t; }
        }
        return at;
    }

    /** KING_ACT net mask: options 0..10 as before (all), the bites where they can land. */
    static boolean[] actOk(RobotInfo[] enemies) {
        boolean[] a = new boolean[KING_ACT_OPTIONS];
        for (int i = 0; i < A_BITE_C0; i++) a[i] = true;
        MapLocation at = weakestLoc(enemies);
        for (int i = 0; i < 6; i++) a[A_BITE_C0 + i] = at != null && G.rc.canAttack(at, D.BITE_C[i]);
        return a;
    }

    /** ratTrap's ring pick toward goal (no cheese floor or trap cap: the engine's can* checks only); a cat trap with
     *  cat set. */
    static boolean trapToward(MapLocation goal, boolean cat) throws GameActionException {
        RobotController rc = G.rc;
        MapLocation c = rc.getLocation(), best = null;
        int bs = Integer.MAX_VALUE;
        for (int i = 0; i < 16; i++) {
            MapLocation l = c.translate(King.RX[i], King.RY[i]);
            if (cat ? !rc.canPlaceCatTrap(l) : !rc.canPlaceRatTrap(l)) continue;
            int s = l.distanceSquaredTo(goal) * 2 + (G.pickBit(c, l) ? 1 : 0);
            if (s < bs) { bs = s; best = l; }
        }
        if (best == null) return false;
        if (cat) rc.placeCatTrap(best);
        else { rc.placeRatTrap(best); King.traps++; }
        return true;
    }

    /** KING_FORM net mode: true when the option acted (formation() is then skipped this turn). */
    static boolean netKingForm() {
        int k = netPick(KING_FORM, formAllowed());
        if (k < 0) return false;
        try {
            if (!execKingForm(k)) return false;
        } catch (GameActionException e) { return false; }
        D.lab = k;
        return true;
    }

    /** KING_FORM options the net may pick now: with a site open the 'no site' exits (which would fail and fall back
     *  to formation()) are masked; with none open the exits that need one (drop, hold) are. */
    static boolean[] formAllowed() {
        boolean[] a = new boolean[KING_FORM_OPTIONS];
        boolean open = King.formSite != null;
        for (int k = 0; k < KING_FORM_OPTIONS; k++) a[k] = true;
        a[F_RICH_DROP] = a[F_RICH_GAP] = a[F_DROP_FORMED] = a[F_DROP_TIMEOUT] = a[F_HOLD] = open;
        a[F_END_WAIT] = a[F_END_NONE] = a[F_IDLE] = a[F_NO_INCOME] = a[F_NO_SITE] = !open;
        boolean full = Comms.numKings >= Comms.KING_SLOTS;
        for (int k = F_START_MINE; k < KING_FORM_OPTIONS; k++) a[k] = !open && !full;   // formOpen refuses these
        return a;
    }

    static boolean execKingForm(int k) throws GameActionException {
        MapLocation me = G.me, fs = King.formSite;
        MapLocation ek = G.mirror(King.start, King.sym());
        switch (k) {
            case F_WAR_OFF: case F_CAP_OFF: case F_CAP_GAP: return formClose();   // F_CAP_GAP: Cfg.RICH_GAP 0, no gap
            case F_RICH_DROP:
                if (fs == null) return false;
                King.richDrop();
                return true;
            case F_RICH_GAP:
                if (fs == null) return false;
                King.richGap();
                return true;
            case F_DROP_FORMED: case F_DROP_TIMEOUT: return fs != null && formClose();   // F_DROP_FORMED: Cfg.FORM_GAP 0
            case F_HOLD:
                if (fs == null) return false;
                Comms.write(Comms.FORM_T, Comms.stamp());
                return true;
            case F_END_WAIT: case F_END_NONE: case F_IDLE: case F_NO_INCOME: case F_NO_SITE:
                return fs == null;
            case F_START_MINE: {
                MapLocation s = King.pickFormSite(me);
                return s != null && G.cheb(me, s) >= 5 && formOpen(s);
            }
            case F_START_BACK: {
                Direction back = ek.directionTo(me);
                if (back == Direction.CENTER) return false;
                MapLocation b = me.translate(4 * back.dx, 4 * back.dy);
                return King.clear3(b) && !King.banned(b) && formOpen(b);
            }
            case F_START_RING: return formOpen(King.ringSite(me, ek));
            case F_START_CROWD: return formOpen(King.crowdSite(me, ek));
            case F_START_MSITE: return formOpen(King.mineSite(me, ek));
            default: {
                int i = k - F_RING_0;
                if (i < 0 || i >= 12) return false;
                int sx = 2 * me.x >= G.W - 1 ? 1 : -1, sy = 2 * me.y >= G.H - 1 ? 1 : -1;
                MapLocation s = me.translate(King.RING[i][0] * sx, King.RING[i][1] * sy);
                return King.clearNow(s) && formOpen(s);
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // pass 3: KING_COMMS (the end-of-turn edit loop), KING_VAL (its values), KING_TURN (the newborn's facing)

    /** The shared array at the start of this king's turn: what differs at the end is this king's own writes (robots
     *  run one at a time). netSet: slots the net wrote (KC_SET_i; not zdb's pending writes any more). */
    static final int[] pre = new int[64];
    static long netSet;
    static int kcEdits;
    static boolean kcSqueaked;

    /** Snapshot the array (hook state only). */
    static void saPre() {
        for (int i = 0; i < 64; i++) pre[i] = Comms.read(i);
        netSet = 0; kcEdits = 0; kcSqueaked = false;
    }

    /** Slots that differ from the snapshot. */
    static long changed() {
        long m = 0;
        for (int i = 0; i < 64; i++) if (Comms.read(i) != pre[i]) m |= 1L << i;
        return m;
    }

    /** zdb's pending writes this turn (changed slots the net did not set). */
    static long pending() { return changed() & ~netSet; }

    /** Own traps of one kind within dist2 8 (the 5x5 around the king; bit (dy + 2) * 5 + dx + 2). */
    static int trapMask(boolean cat) {
        RobotController rc = G.rc;
        MapLocation c = rc.getLocation();
        int m = 0;
        for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
            MapLocation l = c.translate(dx, dy);
            if (rc.onTheMap(l) && (cat ? rc.canRemoveCatTrap(l) : rc.canRemoveRatTrap(l))) m |= 1 << ((dy + 2) * 5 + dx + 2);
        }
        return m;
    }

    /** KING_COMMS features (94) at loop step step. Side-effect free. */
    static void kingComms(int step, RobotInfo[] enemies, RobotInfo[] cats) {
        D.open(KING_COMMS);
        try {
            RobotController rc = G.rc;
            MapLocation me = rc.getLocation();
            long P = pending(), ch = changed();
            D.f(step);
            D.f(kcEdits);
            D.f(kcSqueaked);
            int gm = 0;
            for (int g = 0; g < 7; g++) if ((P & KC_GROUP[g]) != 0) gm |= 1 << g;
            D.f(gm);
            D.f(Long.bitCount(P));
            D.f(G.coop);
            D.f(G.round / 10);
            D.f(D.cap(rc.getGlobalCheese() / 10, 2000));
            D.f(Math.min(65535, Econ.surplus() / 10 + 32768));
            D.f(D.cap(Econ.rate100() / 10, 65535));
            D.f(Comms.numKings);
            D.f(King.slot);
            D.f(rc.getHealth());
            D.f(D.cap(enemies.length, 15));
            D.f(D.cap(cats.length, 4));
            int nb = 0;
            for (RobotInfo r : rc.senseNearbyRobots(-1, G.us)) if (r.type == UnitType.BABY_RAT) nb++;
            D.f(D.cap(nb, 63));
            D.f(rc.isActionReady());
            D.f(King.sawEnemy);
            D.f(Integer.bitCount(trapMask(false)));
            D.f(Integer.bitCount(trapMask(true)));
            D.heard(me);
            for (int i = 0; i < 64; i++) D.f(Comms.read(i));
            for (int w = 0; w < 4; w++) D.f((int) (ch >>> (16 * w)) & 0xFFFF);
        } catch (Exception e) { D.point = 0; }
    }

    /** KING_VAL features (22, secondary slot): value pos of total for purpose (0 a slot value, 1 a squeak, 2 / 3 a
     *  rat / cat trap tile), the values so far (prefix), the target slot idx (64: none), the own-trap mask. */
    static void kingVal(int purpose, int pos, int total, int prefix, int idx, int traps) {
        D.open2(KING_VAL);
        try {
            RobotController rc = G.rc;
            MapLocation me = rc.getLocation();
            D.f(purpose);
            D.f(pos);
            D.f(total);
            D.f(prefix >>> 16);
            D.f(prefix & 0xFFFF);
            D.f(idx);
            D.f(idx < 64 ? Comms.read(idx) : 1024);
            D.f(idx < 64 && (pending() >>> idx & 1) != 0);
            D.f(traps & 0xFFFF);
            D.f(traps >>> 16);
            D.f(G.coop);
            D.f(G.round / 10);
            D.f(D.cap(rc.getGlobalCheese() / 10, 2000));
            D.f(Comms.numKings);
            D.f(King.slot);
            D.f(me.x);
            D.heard(me);
        } catch (Exception e) { D.point2 = 0; }
        D.sec = false;
    }

    /** KING_COMMS net mask at loop step step (past KC_CHAIN: the terminal options only). */
    static boolean[] commsOk(int step) {
        boolean[] a = new boolean[KING_COMMS_OPTIONS];
        long P = pending();
        boolean cont = step < KC_CHAIN, val = cont && DNet.has(KING_VAL);
        a[KC_ZDB] = true;
        a[KC_DROP_ALL] = P != 0;
        for (int g = 0; g < 7; g++) a[KC_DROP0 + g] = cont && (P & KC_GROUP[g]) != 0;
        int b = Comms.read(Comms.BUDGET), band = b < 50 ? 0 : b < 150 ? 1 : 2;
        for (int i = 0; i < 3; i++) a[KC_BUDGET0 + i] = cont && band != i;
        a[KC_ALERT_CALL] = cont && !alertHere();
        a[KC_ALERT_CLEAR] = cont && Comms.read(Comms.ALERT + 2) != 0;
        a[KC_SQUEAK] = val && !kcSqueaked;
        a[KC_RM_RAT] = val && trapMask(false) != 0;
        a[KC_RM_CAT] = val && trapMask(true) != 0;
        for (int i = 0; i < 64; i++) a[KC_SET0 + i] = val;
        return a;
    }

    /** The ALERT already puts threat >= 1 at this king this round. */
    static boolean alertHere() {
        MapLocation a = Comms.alertLoc();
        return Comms.read(Comms.ALERT + 2) == Comms.stamp() && a != null && a.equals(G.rc.getLocation())
            && Comms.alertThreat() >= 1;
    }

    /** Revert the pending writes in mask m. */
    static void revert(long m) {
        for (int i = 0; i < 64; i++) if ((m >>> i & 1) != 0) Comms.write(i, pre[i]);
    }

    /** KING_COMMS in net mode: the edit loop (each continue option is its own record; the final, terminal pick is
     *  the open record King.turn emits). No net: zdb's writes stand (label KC_ZDB). */
    static void netKingComms(RobotInfo[] enemies, RobotInfo[] cats) {
        for (int step = 0; ; ) {
            int k = netPick(KING_COMMS, commsOk(step));
            if (k < 0) return;
            if (k == KC_ZDB || k == KC_DROP_ALL) {
                if (k == KC_DROP_ALL) revert(pending());
                D.lab = k;
                return;
            }
            boolean ok;
            try { ok = execKingComms(k); } catch (Exception e) { ok = false; }
            if (!ok) { D.point = 0; return; }   // whatever was written stands; the record is dropped
            D.lab = k;
            D.emit();
            kcEdits++;
            step++;
            kingComms(step, enemies, cats);
            if (D.point != KING_COMMS) return;
        }
    }

    /** Run a continue option of KING_COMMS; true when it acted. */
    static boolean execKingComms(int k) throws GameActionException {
        RobotController rc = G.rc;
        MapLocation me = rc.getLocation();
        if (k >= KC_DROP0 && k < KC_DROP0 + 7) { revert(pending() & KC_GROUP[k - KC_DROP0]); return true; }
        if (k >= KC_BUDGET0 && k < KC_BUDGET0 + 3) { Comms.write(Comms.BUDGET, KC_BUDGET[k - KC_BUDGET0]); return true; }
        if (k == KC_ALERT_CALL) {
            MapLocation a = Comms.alertLoc();
            int t = Comms.read(Comms.ALERT + 2) == Comms.stamp() && a != null && a.equals(me) ? Comms.alertThreat() : 0;
            Comms.write(Comms.ALERT, (me.x + 1) | (Math.max(1, t) << 6));
            Comms.write(Comms.ALERT + 1, me.y + 1);
            Comms.write(Comms.ALERT + 2, Comms.stamp());
            return true;
        }
        if (k == KC_ALERT_CLEAR) { Comms.write(Comms.ALERT + 2, 0); return true; }
        if (k == KC_SQUEAK) {
            int v = D.digits(32, 1, true, 64, 0);
            if (D.dfail) return false;
            kcSqueaked = true;
            return rc.squeak(v);
        }
        if (k == KC_RM_RAT || k == KC_RM_CAT) {
            boolean cat = k == KC_RM_CAT;
            int m = trapMask(cat);
            if (m == 0) return false;
            int v = D.digits(5, cat ? 3 : 2, true, 64, m);
            if (D.dfail) return false;
            MapLocation l = me.translate(v % 5 - 2, v / 5 - 2);
            if (cat) { if (!rc.canRemoveCatTrap(l)) return false; rc.removeCatTrap(l); }
            else { if (!rc.canRemoveRatTrap(l)) return false; rc.removeRatTrap(l); }
            return true;
        }
        if (k >= KC_SET0 && k < KC_SET0 + 64) {
            int i = k - KC_SET0;
            int v = D.digits(10, 0, true, i, 0);
            if (D.dfail) return false;
            Comms.write(i, v);
            netSet |= 1L << i;
            return true;
        }
        return false;
    }

    /** The compass direction + 1 from c to t (0: none or here). */
    static int dir1(MapLocation c, MapLocation t) {
        if (t == null) return 0;
        Direction d = c.directionTo(t);
        return d == Direction.CENTER ? 0 : d.ordinal() + 1;
    }

    /** KING_TURN features (24): opens when a spawn is possible (turning- and action-ready, rat cost covered), the only
     *  time the king's facing matters (a newborn faces it). Side-effect free. */
    static void kingTurn(RobotInfo[] cats, RobotInfo[] enemies) {
        D.open(KING_TURN);
        try {
            RobotController rc = G.rc;
            MapLocation c = rc.getLocation();
            D.f(rc.getDirection().ordinal());
            D.f(G.round / 10);
            D.f(G.coop);
            D.f(D.cap(rc.getGlobalCheese() / 10, 2000));
            D.f(D.cap((rc.getCurrentRatCost() - 10) / 10, 255));
            D.f(Math.min(65535, Econ.surplus() / 10 + 32768));
            MapLocation ne = null, nc = null;
            int ed = 400, cd = 400;
            for (RobotInfo e : enemies) { int d = c.distanceSquaredTo(e.location); if (d < ed) { ed = d; ne = e.location; } }
            for (RobotInfo k : cats) { int d = c.distanceSquaredTo(k.location); if (d < cd) { cd = d; nc = k.location; } }
            D.f(dir1(c, ne));
            D.f(ed);
            D.f(dir1(c, nc));
            D.f(cd);
            D.f(dir1(c, King.homeMine));
            D.f(King.start != null ? dir1(c, G.mirror(King.start, King.sym())) : 0);
            D.f(dir1(c, new MapLocation(G.W / 2, G.H / 2)));
            int nb = 0;
            for (RobotInfo r : rc.senseNearbyRobots(-1, G.us)) if (r.type == UnitType.BABY_RAT) nb++;
            D.f(D.cap(nb, 63));
            D.f(Comms.numKings);
            D.f(Comms.numMines);
            D.f(King.slot);
            D.f(King.sawEnemy);
            D.f(rc.getHealth());
            D.f(D.cap(King.spawned, 255));
            D.f(c.x);
            D.f(c.y);
            D.f(G.W);
            D.f(G.H);
        } catch (Exception e) { D.point = 0; }
    }

    /** KING_TURN in net mode: KT_ZDB keeps the facing; KT_DIR0 + i turns to G.DIRS[i]. */
    static void netKingTurn() {
        RobotController rc = G.rc;
        boolean[] a = new boolean[KING_TURN_OPTIONS];
        a[KT_ZDB] = true;
        for (int i = 0; i < 8; i++) a[KT_DIR0 + i] = G.DIRS[i] != rc.getDirection() && rc.canTurn(G.DIRS[i]);
        int k = netPick(KING_TURN, a);
        if (k < 0) return;
        try {
            if (k >= KT_DIR0) rc.turn(G.DIRS[k - KT_DIR0]);
            D.lab = k;
        } catch (GameActionException e) { /* the facing stays; label KT_ZDB */ }
    }

    /** Close the formation site (none open is fine). */
    static boolean formClose() {
        King.closeForm();
        return true;
    }

    /** Open site s as formation's success tail does; false with a site already open or every king slot taken (a new
     *  king would then take over slot 0 through claimSlot's fallback). */
    static boolean formOpen(MapLocation s) {
        if (s == null || King.formSite != null || Comms.numKings >= Comms.KING_SLOTS) return false;
        King.openForm(s);
        return true;
    }

    private DK() {}
}
