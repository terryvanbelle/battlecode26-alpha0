package zdp;

import battlecode.common.*;

/**
 * King decision points (docs/decision-points.md): KING_MOVE (King.move), KING_SPAWN (King.spawn), KING_ACT (the action
 * block after spawn: bite, biteCat, ratTrap, Moat.king) and KING_FORM (King.formation). D holds the record buffer;
 * this class holds the king points' ids, option constants, feature snapshots and (net mode, D.NET) option executors.
 *
 * Feature code has no side effects on bot state (no Comms.write, no King.arcsHere, no KPath.follow, no King.rich(),
 * no assignment to King fields) and never aborts the king's turn: an exception drops the record (point = 0).
 * Bytecode limits are off for this work, so these hooks are not credited to the G.debt ledger.
 */
public final class DK {
    // decision points (BABY_* use 1..9, KING_* 10..19)
    static final int KING_MOVE = 10, KING_SPAWN = 11, KING_ACT = 12, KING_FORM = 13;

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
    static final int KING_ACT_OPTIONS = 11;

    // KING_FORM options: formation's exits, the kinds of site it opens, then a site at each RING offset
    static final int F_WAR_OFF = 0, F_CAP_OFF = 1, F_CAP_GAP = 2, F_RICH_DROP = 3, F_RICH_GAP = 4, F_DROP_FORMED = 5,
        F_DROP_TIMEOUT = 6, F_HOLD = 7, F_END_WAIT = 8, F_END_NONE = 9, F_IDLE = 10, F_NO_INCOME = 11, F_NO_SITE = 12,
        F_START_MINE = 13, F_START_BACK = 14, F_START_RING = 15, F_START_CROWD = 16, F_START_MSITE = 17, F_RING_0 = 18;
    static final int KING_FORM_OPTIONS = 30;

    /** The ring index (King.RX/RY) of offset (dx, dy) from the king, or -1. spawn's 5x5-minus-3x3 loop covers exactly
     *  these 16 tiles. */
    static int ring(int dx, int dy) {
        for (int i = 0; i < 16; i++) if (King.RX[i] == dx && King.RY[i] == dy) return i;
        return -1;
    }

    /** The kind of a site formation just opened, from its Chebyshev distance to the king (G.me equals formation's
     *  `from`: the king has not moved yet this turn). ringSite/crowdSite sites are at 3 (RING offsets), the back spot at
     *  4 (0 when the enemy start mirrors onto us and back is CENTER), mine-list and mineSite sites at 5 or more. */
    static int formKind(MapLocation s, boolean end) {
        int c = G.cheb(G.me, s);
        if (c == 3) return end && Cfg.END_KING_CROWD ? F_START_CROWD : F_START_RING;
        if (c <= 4) return F_START_BACK;
        return Cfg.FORM_MINE || end && Cfg.END_KING_MSITE ? F_START_MSITE : F_START_MINE;
    }

    static int sym() { return (Sym.mask & Sym.ROT) != 0 ? Sym.ROT : (Sym.mask & Sym.HOR) != 0 ? Sym.HOR : Sym.VER; }

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
            D.f("-nsfcbd".indexOf(King.spawnWhy) + 1);
            D.f(D.cap((rc.getCurrentRatCost() - 10) / 10 * 4, 255));
            D.f(c.x);
            D.f(c.y);
            D.f(G.W);
            D.f(G.H);
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
            D.f(Moat.up());
            D.f("-nsfcbd".indexOf(King.spawnWhy));
            D.f(D.cap(King.spawned, 255));
            D.f(Comms.numKings);
            D.f(me.x);
            D.f(me.y);
            D.f(G.W);
            D.f(G.H);
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
            D.f(Cfg.FORM_WAR_FROM > 0 && !G.coop && G.round < Cfg.FORM_WAR_FROM);
            D.f(nk >= Math.min(King.formCap(G.coop), Comms.KING_SLOTS));
            D.f(Cfg.FORM_RICH_BANK > 0 && (King.rich0() || Cfg.RICH_LATE > 0 && King.lateRich(G.round, nk, gc)
                || Cfg.RICH_MORE > 0 && King.moreRich(G.round, nk, gc)));
            D.f(Cfg.END_KING > 0 && King.endKing(G.round, nk, babies, gc, King.formStart));
            D.f(King.formIncome());
            D.f(!(G.round < Cfg.FORM_FROM || G.round > (Cfg.FORM_LATE && nk < 2 ? 1940 : 1150)
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
            MapLocation ek = G.mirror(King.start, sym());
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
        } catch (Exception e) { D.point = 0; }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // net mode (D.NET): the net picks an option for the open point; its executor runs it. An executor returns false
    // when the option cannot act, and zdb's own code then runs as the fallback. Dead code in the recording build.

    static final float NET_T = D.NET_T, NET_FLOOR = D.NET_FLOOR;   // set in D (build_dpnet.py)

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
        return DNet.sample(z, allowed, NET_T, NET_FLOOR);
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
            case K_CAT_FLEE:
                // zdb's cat flee, G.me left stale as zdb does
                for (RobotInfo cat : cats) {
                    MapLocation b = cat.location;
                    if (Cfg.SYM_CENTRE ? Baby.catDist4(b, c) <= 80 : c.distanceSquaredTo(b) <= 20) {
                        Direction away = Cfg.SYM_CENTRE ? G.dirOf(2 * c.x - 2 * b.x - 1, 2 * c.y - 2 * b.y - 1) : b.directionTo(c);
                        for (Direction d : G.fan(c, away)) if (rc.canMove(d)) { rc.move(d); return true; }
                    }
                }
                return false;
            case K_LANE: return Cfg.CAT_LANE && King.offLane(c, cats);
            case K_FORM_STILL: return King.formSite != null;
            case K_OFF_SITE: return King.offSite(c);
            case K_UNPLUG:
                if (King.unplug(c)) { King.mineWhy = 'u'; return true; }
                return false;
            case K_PATH: return Cfg.KING_PATH && KPath.follow(c);
            case K_PILE:
                if (King.nearCheese != null && G.cheb(c, King.nearCheese) > 1 && King.walkStep(c, King.nearCheese)) {
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
                if (h != null && G.cheb(c, h) > 1 && King.walkStep(c, h)) { King.mineWhy = 'z'; King.pickUp(); return true; }
                return false;
            }
            case K_DIG: {
                MapLocation h = home();
                if (h == null || G.cheb(c, h) <= 1) return false;
                MapLocation t = King.digTile(c, h);
                return t != null && dig(t);
            }
            case K_NO_MINE: case K_THREAT_HOLD: case K_NO_HOME: case K_AT_HOME: case K_DIG_WAIT: case K_DIG_POOR:
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
        RobotController rc = G.rc;
        if (!rc.canRemoveDirt(t)) return false;
        rc.removeDirt(t);
        King.digs++;
        Sym.mem[G.idx(t)] &= ~Sym.DIRT;
        King.mineWhy = 'D';
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
            case S_HOLD: return true;
            default: {
                int i = k - S_BUILD_0;
                if (i < 0 || i >= 16) return false;
                MapLocation c = rc.getLocation(), l = c.translate(King.RX[i], King.RY[i]);
                if (!rc.isActionReady() || !rc.canBuildRat(l)) return false;
                if (King.faceSpawn) { Direction f = c.directionTo(l); if (rc.getDirection() != f && rc.canTurn(f)) rc.turn(f); }
                rc.buildRat(l);
                King.spawned++;
                King.spawnWhy = 'b';
                return true;
            }
        }
    }

    /** KING_ACT net mode: A_NONE sets hold, which skips zdb's bite/biteCat/ratTrap lines for the turn. */
    static boolean hold;

    static void netKingAct(RobotInfo[] enemies, RobotInfo[] cats) {
        hold = false;
        int k = netPick(KING_ACT, null);
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
                RobotInfo b = null;
                for (RobotInfo e : enemies) {
                    if (!rc.canAttack(e.location)) continue;
                    if (b == null || e.health < b.health) b = e;
                }
                if (b == null) return false;
                if (k == A_BITE) { rc.attack(b.location); return true; }
                int c = King.biteCheese(b.health, rc.getGlobalCheese());
                if (c <= 0 || !rc.canAttack(b.location, c)) return false;
                rc.attack(b.location, c);
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
            default: return false;
        }
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
        int k = netPick(KING_FORM, null);
        if (k < 0) return false;
        try {
            if (!execKingForm(k)) return false;
        } catch (GameActionException e) { return false; }
        D.lab = k;
        return true;
    }

    static boolean execKingForm(int k) throws GameActionException {
        MapLocation me = G.me, fs = King.formSite;
        MapLocation ek = G.mirror(King.start, sym());
        switch (k) {
            case F_WAR_OFF: case F_CAP_OFF: return formClose(0);
            case F_CAP_GAP: return formClose(Cfg.RICH_GAP);
            case F_RICH_DROP:
                if (fs == null) return false;
                King.richDrop();
                return true;
            case F_RICH_GAP:
                if (fs == null) return false;
                King.richGap();
                return true;
            case F_DROP_FORMED: return fs != null && formClose(Cfg.FORM_GAP);
            case F_DROP_TIMEOUT:
                if (fs == null) return false;
                if ((Cfg.FORM_MINE || Cfg.END_KING_MSITE) && Comms.numKings <= King.formKings) King.ban(fs);
                return formClose(0);
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

    /** Close the formation site (none open is fine); gap > 0 starts that gap timer as formation's gap branches do. */
    static boolean formClose(int gap) {
        King.formSite = null;
        Comms.write(Comms.FORM_T, 0);
        if (gap > 0) King.formStart = G.round + gap - Cfg.FORM_TTL - Cfg.FORM_RETRY;
        return true;
    }

    /** Open site s as formation's success tail does; false with a site already open or every king slot taken (a new
     *  king would then take over slot 0 through claimSlot's fallback). */
    static boolean formOpen(MapLocation s) {
        if (s == null || King.formSite != null || Comms.numKings >= Comms.KING_SLOTS) return false;
        King.formSite = s;
        King.formStart = G.round;
        King.formKings = Comms.numKings;
        Comms.write(Comms.FORM_X, s.x + 1);
        Comms.write(Comms.FORM_Y, s.y + 1);
        Comms.write(Comms.FORM_T, Comms.stamp());
        return true;
    }

    private DK() {}
}
