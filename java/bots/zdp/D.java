package zdp;

import battlecode.common.*;

/**
 * Decision points (docs/decision-points.md). At each point zdb's own code still decides (pass 1 is bug-for-bug zdb);
 * this class snapshots the point's features before the decision, notes which branch zdb took (lab), and sends the
 * record to the research harness through the indicator string (the harness intercepts strings that start with MAGIC).
 * BABY_MOVE's hooks credit their bytecodes to the ledger (G.debt); the later hooks do not (bytecode limits are off
 * for this work, the user's decision), so replay checks compare zdp and zdb with limits off.
 *
 * Two record slots: the primary one (F, n, point, lab) and a secondary one (F2, n2, point2, lab2) for a point whose
 * span lies inside a primary point's span (BABY_COMMS inside BABY_ACT_PRE, MINE_PICK inside BABY_MOVE). f() writes to
 * the slot the last open()/open2() chose; each secondary feature function ends with sec = false.
 *
 * Wire format (chars): [0] MAGIC, [1] point, [2] label (0xFFFF: none), [3] n, [4..4+n) features (unsigned 16-bit).
 */
public final class D {
    /** Send decision records to the harness (false in tournament builds). */
    static final boolean EMIT = true;
    /** Net-mode build: a decision net picks the option at each point (zdb's code is the fallback). */
    static final boolean NET = false;
    /** Compute features at all (recording or net mode). */
    static final boolean ON = EMIT || NET;
    /** Net-mode sampling: softmax temperature and uniform probability floor. */
    static final float NET_T = 1f;   // net-mode sampling temperature (build_dpnet.py sets it)
    static final float NET_FLOOR = 0.02f;   // net-mode probability floor, spread over allowed options (build_dpnet.py)
    static final char MAGIC = '\uDEC0';

    // decision points (BABY_* 1-9; king points 10-19 live in DK)
    static final int BABY_MOVE = 1;
    static final int BABY_ACT_PRE = 2;
    static final int BABY_ACT_POST = 3;
    static final int BABY_FACE = 4;
    static final int BABY_ROLE = 5;
    static final int BABY_COMMS = 6;
    static final int MINE_PICK = 7;

    // BABY_MOVE options: the branches of Baby.moveDecision, in zdb's priority order
    static final int M_GRAB = 0, M_CAT_DEFEND = 1, M_END_CAT = 2, M_FLEE_CAT = 3, M_YIELD_KING = 4, M_AIM_CARRY = 5,
        M_RESCUE = 6, M_AMMO_AWAY = 7, M_RETREAT = 8, M_HUNGRY = 9, M_FORM_KING = 10, M_CASH = 11, M_RAID = 12,
        M_FLANK = 13, M_PRESS = 14, M_MICRO = 15, M_FACE = 16, M_ATK_DELIVER = 17, M_STALL_CAT = 18, M_ATTACK = 19,
        M_WAR = 20, M_FORT = 21, M_FIGHT_CAT = 22, M_DELIVER = 23, M_COURIER = 24, M_CHEESE_PICK = 25, M_CHEESE = 26,
        M_KNOWN_CAT = 27, M_RAID_FIGHT = 28, M_SCOUT_HOME = 29, M_SCOUT = 30, M_SEEK = 31, M_MINE_EXPLORE = 32,
        M_IDLE_SEEK = 33, M_START_SEEK = 34, M_MINE_STILL = 35, M_MINE_WANDER = 36, M_MINE_GO = 37, M_EXPLORE = 38;
    static final int BABY_MOVE_OPTIONS = 39;
    /** Net-mode BABY_MOVE primitives (never labelled by zdb): step in G.DIRS[i], face G.DIRS[i], hold. */
    static final int P_STEP0 = 39, P_TURN0 = 47, P_HOLD = 55;
    static final int BABY_MOVE_NET_OPTIONS = 56;

    // BABY_ACT_PRE / BABY_ACT_POST options (shared): the action behaviour that fired (A_X_*: primitives zdb never takes)
    static final int A_PASS = 0, A_DROP_ALLY = 1, A_THROW = 2, A_NAP = 3, A_BITE_KING = 4, A_BITE_RAT = 5,
        A_BITE_CAT = 6, A_CAT_TRAP = 7, A_LAY_LANE = 8, A_DELIVER = 9, A_CASH_KING = 10, A_RAT_TRAP = 11,
        A_X_BITE_FWD = 12, A_X_CARRY_FWD = 13, A_X_THROW_NOW = 14, A_X_DROP_FWD = 15, A_X_RAT_TRAP_FWD = 16,
        A_X_CAT_TRAP_FWD = 17, A_X_DIRT_PLACE_FWD = 18, A_X_DIRT_REMOVE_FWD = 19, A_X_BECOME_KING = 20;
    static final int BABY_ACT_PRE_OPTIONS = 21;
    static final int BABY_ACT_POST_OPTIONS = 21;
    static final int BABY_ACT_OPTIONS = 21;

    // BABY_FACE options: guardTurn's target (F_DIR0 + i: face G.DIRS[i], primitives)
    static final int F_NONE = 0, F_SEEN = 1, F_MEMORY = 2, F_HEARD = 3, F_GUARD = 4, F_DIR0 = 5;
    static final int BABY_FACE_OPTIONS = 13;

    // BABY_ROLE options
    static final int R_GATHER = 0, R_SCOUT = 1, R_FIGHT = 2, R_RAIDER = 3, R_ATTACK = 4;
    static final int BABY_ROLE_OPTIONS = 5;

    // BABY_COMMS options: the squeak this turn
    static final int C_NONE = 0, C_MINE_GOSSIP = 1, C_MINE_RELAY = 2, C_SYM_KING = 3, C_MINE_KING = 4, C_CAT = 5,
        C_CASH_LEAD = 6, C_CASH_RELAY = 7, C_EKING = 8, C_SYM_GOSSIP = 9, C_BURN = 10, C_TRIP = 11, C_TRAP = 12;
    static final int BABY_COMMS_OPTIONS = 13;

    // MINE_PICK options: rank i in pickMine's candidate order
    static final int MINE_PICK_OPTIONS = 40;

    static final char[] F = new char[96], F2 = new char[96];
    static int n, point;
    /** The option the open point's decision took (set at each branch; -1: none yet). */
    static int lab = -1;
    /** The secondary record slot. */
    static int n2, point2, lab2 = -1;
    /** f() writes to the secondary slot. */
    static boolean sec;
    /** MINE_PICK: which moveDecision call site asked (0 fresh/reset, 1 IDLE_SHIFT, 2 retry). */
    static int pickSite;

    static void open(int p) { F[0] = MAGIC; F[1] = (char) p; n = 4; point = p; lab = -1; sec = false; }

    static void open2(int p) { F2[0] = MAGIC; F2[1] = (char) p; n2 = 4; point2 = p; lab2 = -1; sec = true; }

    static void f(int v) {
        char c = (char) (v < 0 ? 0 : v > 65535 ? 65535 : v);
        if (sec) F2[n2++] = c; else F[n++] = c;
    }

    static void f(boolean b) { f(b ? 1 : 0); }

    static int cap(int v, int c) { return v > c ? c : v; }

    /** Close the open point and send its record. */
    static void emit() {
        if (!EMIT || point == 0) return;
        F[2] = (char) (lab < 0 ? 0xFFFF : lab);
        F[3] = (char) (n - 4);
        G.rc.setIndicatorString(new String(F, 0, n));
        point = 0;
    }

    /** Close the open secondary point and send its record. */
    static void emit2() {
        if (!EMIT || point2 == 0) return;
        F2[2] = (char) (lab2 < 0 ? 0xFFFF : lab2);
        F2[3] = (char) (n2 - 4);
        G.rc.setIndicatorString(new String(F2, 0, n2));
        point2 = 0;
    }

    /** BABY_ACT_*: the behaviour performed its action (overwrites a turn-only label). */
    static void act(int k) { if (point == BABY_ACT_PRE || point == BABY_ACT_POST) lab = k; }

    /** BABY_ACT_*: the behaviour only turned to face its target (kept unless an action follows). */
    static void turnOnly(int k) { if ((point == BABY_ACT_PRE || point == BABY_ACT_POST) && lab == A_PASS) lab = k; }

    /** BABY_FACE: guardTurn chose this target. */
    static void face(int k) { if (point == BABY_FACE) lab = k; }

    /** Net mode: the net's option for the open primary point (-1: no net for it, or nothing allowed). */
    static int pick(int p, boolean[] ok) {
        float[] z = DNet.logits(p, F, n);
        return z == null ? -1 : DNet.sample(z, fit(ok, z.length), NET_T, NET_FLOOR);
    }

    /** Net mode: the net's option for the open secondary point. */
    static int pick2(int p, boolean[] ok) {
        float[] z = DNet.logits(p, F2, n2);
        return z == null ? -1 : DNet.sample(z, fit(ok, z.length), NET_T, NET_FLOOR);
    }

    /** An allowed mask sized to the net's output (options past the mask are not allowed). */
    static boolean[] fit(boolean[] ok, int len) {
        if (ok == null || ok.length == len) return ok;
        boolean[] a = new boolean[len];
        System.arraycopy(ok, 0, a, 0, Math.min(len, ok.length));
        return a;
    }

    /** BABY_MOVE features: the rat's situation as zdb's movement cascade sees it (state from Baby.turn's sensing). */
    static void babyMove() throws GameActionException {
        RobotController rc = G.rc;
        open(BABY_MOVE);
        MapLocation me = G.me;
        f(Baby.role);
        f(cap(rc.getRawCheese(), 1000));
        f(cap(G.globalCheese / 10, 2000));
        f(G.coop);
        f(G.round / 10);
        f(rc.getHealth());
        RobotInfo cat = Baby.cat;
        f(cat != null);
        f(cat != null ? cap(Baby.catDist4(cat.location, me), 400) : 400);
        RobotInfo ne = Baby.nearestEnemy;
        f(ne != null);
        f(ne != null ? cap(me.distanceSquaredTo(ne.location), 100) : 100);
        f(Baby.hunting);
        f(false);   // rescue target (Cfg.RESCUE off: never set)
        f(false);   // carrier guard (Cfg.GUARD_CARRY_R off: never set)
        RobotInfo c = rc.getCarrying();
        f(c == null ? 0 : c.team == G.us ? 1 : 2);
        MapLocation t = Baby.cheeseTarget;
        f(t != null);
        f(t != null ? cap(G.cheb(me, t), 63) : 63);
        t = Baby.mineTarget;
        f(t != null);
        f(t != null ? cap(G.cheb(me, t), 63) : 63);
        t = Comms.nearestKing(me);
        f(t != null);
        f(t != null ? cap(G.cheb(me, t), 63) : 63);
        f(Comms.numKings);
        f(Comms.numMines);
        f(Baby.raider);
        f(Baby.scout);
        f(Baby.eKing != null);
        t = Baby.lastEKing;
        f(t != null ? cap(G.round - Baby.lastEKingRound, 255) : 255);
        f(t != null ? cap(G.cheb(me, t), 63) : 63);
        f(rc.isActionReady());
        f(cap(Nav.stall, 63));
        f(Baby.cashSite != null);
        f(G.round < Baby.seekUntil);
        f(cap(Baby.myMineCount, 24));
        int foes = 0, friends = 0, foeKings = 0;
        for (RobotInfo r : Baby.seen) {
            if (r.team == G.them) { if (r.type == UnitType.RAT_KING) foeKings++; else foes++; }
            else if (r.team == G.us) friends++;
        }
        f(cap(foes, 15));
        f(cap(friends, 15));
        f(foeKings);
        f(Baby.threatenedKing() != null);
        f(Baby.formSite() != null);
        f(Baby.home != null ? cap(G.cheb(me, Baby.home), 63) : 63);
        f(me.x);
        f(me.y);
        f(G.W);
        f(G.H);
        f(G.facing.ordinal());
        // appended (47..52): availability of the net-mode primitives and of some zdb options
        int mv = 0;
        for (int i = 0; i < 8; i++) if (rc.canMove(G.DIRS[i])) mv |= 1 << i;
        f(mv);
        f(Nav.avoid & 255);
        f(rc.isTurningReady());
        f(Baby.knownCat() != null);
        f(Baby.cheeseTarget != null && G.cheb(me, Baby.cheeseTarget) == 1);
        f(G.round >= Baby.seekNext);
    }

    /** BABY_ACT_PRE / BABY_ACT_POST features (53, shared layout; feature 0 is the phase). Side-effect free. */
    static void babyAct(int p, RobotInfo[] robots, boolean moved, char why) throws GameActionException {
        RobotController rc = G.rc;
        open(p);
        lab = A_PASS;
        MapLocation me = rc.getLocation();
        f(p == BABY_ACT_POST);
        f(Baby.role);
        f(cap(rc.getRawCheese(), 1000));
        f(cap(rc.getAllCheese() / 10, 2000));
        f(G.coop);
        f(G.round / 10);
        f(rc.getHealth());
        f(Baby.hostile());
        f(rc.isMovementReady());
        f(rc.isTurningReady());
        f(moved);
        f(why);
        RobotInfo c = rc.getCarrying();
        f(c == null ? 0 : c.team == G.us ? 1 : 2);
        f(c != null ? cap(G.round - Baby.carryRound, 63) : 63);
        f(c != null && rc.canThrowRat());
        int foes = 0, friends = 0, eAdj = 0, eAtk = 0, eCarry = 0, eHold = 0, minHp = 255, kAdj = 0, kAtk = 0, kHp = 0;
        for (RobotInfo r : robots) {
            if (r.team == G.them) {
                if (r.type == UnitType.RAT_KING) {
                    MapLocation t = r.location.translate(Math.max(-1, Math.min(1, me.x - r.location.x)),
                        Math.max(-1, Math.min(1, me.y - r.location.y)));
                    if (me.distanceSquaredTo(t) <= 2) { kAdj++; kHp = r.health; if (rc.canAttack(t)) kAtk++; }
                } else {
                    foes++;
                    if (me.distanceSquaredTo(r.location) <= 2) {
                        eAdj++;
                        if (rc.canAttack(r.location)) eAtk++;
                        if (rc.canCarryRat(r.location)) eCarry++;
                        if (r.health < minHp) minHp = r.health;
                        if (r.carryingRobot != null && r.carryingRobot.team == G.us) eHold++;
                    }
                }
            } else if (r.team == G.us) friends++;
        }
        f(cap(foes, 15));
        f(cap(friends, 15));
        f(eAdj);
        f(eAtk);
        f(eCarry);
        f(eHold);
        f(minHp);
        f(kAdj);
        f(kAtk);
        f(cap(kHp, 600));
        RobotInfo cat = Baby.cat;
        int cAdj = 0, cAtk = 0;
        if (cat != null) for (int i = 0; i < 4; i++) {
            MapLocation t = cat.location.translate(i & 1, i >> 1);
            if (me.distanceSquaredTo(t) <= 2) { cAdj++; if (rc.canAttack(t)) cAtk++; }
        }
        f(cat != null);
        f(cat != null ? cap(Baby.catDist4(cat.location, me), 400) : 400);
        f(cAdj);
        f(cAtk);
        MapLocation k = Comms.nearestKing(me);
        f(k != null ? cap(G.cheb(me, k), 63) : 63);
        f(k != null && cat != null ? cap(k.distanceSquaredTo(cat.location), 200) : 200);
        f(cap(rc.getNumberCatTraps(), 15));
        f(cap(rc.getNumberRatTraps(), 31));
        f(cap(Comms.read(Comms.BUDGET), 4000));
        int lane = 0;
        for (Direction d : G.DIRS) {
            MapLocation l = me.add(d);
            if (CatLane.lane(l.x, l.y) && CatLane.age(l.x, l.y, G.round) >= Cfg.CAT_LAY_AGE && rc.canPlaceCatTrap(l)) lane++;
        }
        f(lane);
        int raw = rc.getRawCheese();
        boolean give = false;
        if (raw > 0) for (RobotInfo r : rc.senseNearbyRobots(9, G.us)) {
            if (r.type != UnitType.RAT_KING) continue;
            for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++)
                if (rc.canTransferCheese(r.location.translate(dx, dy), raw)) give = true;
        }
        f(give);
        f(Baby.cashSite != null);
        f(Baby.cashSite != null && me.equals(Baby.cashSite));
        f(rc.canBecomeRatKing());
        RobotInfo ne = Baby.nearestEnemy;
        f(ne != null);
        f(ne != null ? cap(me.distanceSquaredTo(ne.location), 100) : 100);
        f(Baby.hunting);
        Direction fd = rc.getDirection();
        MapLocation fl = me.add(fd);
        boolean on = rc.onTheMap(fl);
        f(fd.ordinal());
        RobotInfo fr = on && rc.canSenseLocation(fl) ? rc.senseRobotAtLocation(fl) : null;
        f(fr == null ? 0 : fr.team == Team.NEUTRAL ? 5 : (fr.team == G.us ? 1 : 3) + (fr.type == UnitType.RAT_KING ? 1 : 0));
        f(on && rc.canPlaceRatTrap(fl));
        f(on && rc.canPlaceCatTrap(fl));
        f(on && rc.canPlaceDirt(fl));
        f(on && rc.canRemoveDirt(fl));
        f(cap(rc.getDirt(), 255));
        f(me.x);
        f(me.y);
        f(G.W);
        f(G.H);
    }

    /** BABY_FACE features (31): guardTurn's view. Side-effect free. */
    static void babyFace(boolean moved) throws GameActionException {
        RobotController rc = G.rc;
        open(BABY_FACE);
        lab = F_NONE;
        MapLocation me = rc.getLocation();
        f(Baby.role);
        f(G.round / 10);
        f(rc.getHealth());
        f(cap(rc.getRawCheese(), 1000));
        f(rc.isMovementReady());
        f(rc.isActionReady());
        f(moved);
        f(rc.getDirection().ordinal());
        RobotInfo best = null;
        int bd = Integer.MAX_VALUE;
        for (RobotInfo r : rc.senseNearbyRobots(Cfg.GUARD_DIST2, G.them)) {
            if (r.type != UnitType.BABY_RAT) continue;
            int d = me.distanceSquaredTo(r.location);
            if (d < bd) { bd = d; best = r; }
        }
        f(best != null ? me.directionTo(best.location).ordinal() + 1 : 0);
        f(best != null ? bd : 0);
        MapLocation mt = null;
        int md = Integer.MAX_VALUE;
        for (RobotInfo r : Baby.seen) {
            if (r.team != G.them || r.type != UnitType.BABY_RAT) continue;
            int d = me.distanceSquaredTo(r.location);
            if (d <= Cfg.GUARD_DIST2 && d < md) { md = d; mt = r.location; }
        }
        f(mt != null ? me.directionTo(mt).ordinal() + 1 : 0);
        f(mt != null ? md : 0);
        // heard enemies (count, direction + 1, dist2): none, since no ally squeaks them (Cfg.TRAP_SHARE off)
        f(0);
        f(0);
        f(0);
        f(Nav.guard != null ? me.directionTo(Nav.guard).ordinal() + 1 : 0);
        int[] cnt = new int[9];
        for (RobotInfo r : rc.senseNearbyRobots(-1, G.them)) cnt[me.directionTo(r.location).ordinal()]++;
        for (int i = 0; i < 8; i++) f(cap(cnt[i], 7));
        f(Baby.cat != null ? me.directionTo(Baby.cat.location).ordinal() + 1 : 0);
        f(me.x);
        f(me.y);
        f(G.W);
        f(G.H);
    }

    /** BABY_ROLE features (32): ph 0 at birth (after born()), 1 at a review; prev is the role before the review
     *  (-1 at birth). Reads neither role, scout nor raider. Side-effect free. */
    static void babyRole(int ph, int prev) throws GameActionException {
        RobotController rc = G.rc;
        open(BABY_ROLE);
        MapLocation me = G.me, home = Baby.home != null ? Baby.home : me;
        f(ph);
        f(prev < 0 ? 3 : prev);
        f(cap((rc.getCurrentRatCost() - 10) / 10, 63));
        f(cap(G.round / 10, 255));
        f(G.round % Cfg.FIGHT_EVERY);
        f(G.round < Cfg.RUSH_UNTIL);
        f((Baby.slot >>> 4) % Cfg.SCOUT_EVERY);
        f(G.coop);
        f(cap(G.globalCheese / 10, 2000));
        f(Comms.numKings);
        f(Comms.numMines);
        f(Sym.count());
        f(cap(G.cheb(home, G.mirror(home, Baby.firstSym())), 63));
        f(cap(G.cheb(me, home), 63));
        f(cap((G.round - Baby.birthRound) / 10, 255));
        f(rc.getHealth());
        f(cap(rc.getRawCheese(), 1000));
        int ab = 0, ak = 0, er = 0, ek = 0;
        for (RobotInfo r : rc.senseNearbyRobots(-1, G.us)) { if (r.type == UnitType.BABY_RAT) ab++; else if (r.type == UnitType.RAT_KING) ak++; }
        for (RobotInfo r : rc.senseNearbyRobots(-1, G.them)) { if (r.type == UnitType.RAT_KING) ek++; else er++; }
        f(cap(ab, 15));
        f(cap(ak, 15));
        f(cap(er, 15));
        f(ek > 0);
        f(rc.senseNearbyRobots(-1, Team.NEUTRAL).length > 0);
        f(Comms.fresh(Comms.read(Comms.ALERT + 2), 2));
        f(Comms.alertThreat());
        f(cap(Baby.myMineCount, 24));
        f(cap(Baby.delivered / 10, 2000));
        f(cap(Nav.stall, 63));
        f(cap(G.round - Baby.ekSeen, 255));
        f(me.x);
        f(me.y);
        f(G.W);
        f(G.H);
    }

    /** BABY_ROLE's label: zdb's final role combination. */
    static int roleLab() {
        return Baby.role == Baby.ATTACK ? R_ATTACK : Baby.role == Baby.FIGHT ? (Baby.raider ? R_RAIDER : R_FIGHT)
            : Baby.scout ? R_SCOUT : R_GATHER;
    }

    /** BABY_COMMS features (37, secondary slot), before cashTick/report. Side-effect free. */
    static void babyComms() {
        RobotController rc = G.rc;
        open2(BABY_COMMS);
        lab2 = C_NONE;
        MapLocation me = G.me;
        MapLocation k = Comms.numKings > 0 ? Comms.nearestKing(me) : null;
        f(Baby.role);
        f(G.coop);
        f(cap(G.round / 10, 255));
        f(k != null && me.distanceSquaredTo(k) <= 16);
        f(k != null ? cap(G.cheb(me, k), 63) : 63);
        f(Comms.numKings);
        f(Comms.numMines);
        f(cap(Baby.myMineCount, 24));
        int u = 0;
        for (int i = 0; i < Baby.myMineCount; i++) if (!Comms.hasMine(Baby.myMines[i] / 64, Baby.myMines[i] % 64)) u++;
        f(u);
        f(Baby.unreported() >= 0);
        int s = Comms.read(Comms.SYM);
        f(s != 0 && (Sym.mask & s) != s);
        f(Sym.count());
        f(Baby.catSeen != null && G.round - Baby.catSeenRound <= 2);
        f(Baby.cat != null && Baby.catInEarshot(Baby.cat.location, me));
        f(Baby.eKing != null);
        int raw = rc.getRawCheese();
        int foes = 0, allies = 0, cashN = 0, cashHeavier = 0;
        for (RobotInfo r : Baby.seen) {
            if (r.team == G.them && r.type != UnitType.RAT_KING) foes++;
            if (r.team == G.us && r.type == UnitType.BABY_RAT) {
                if (me.distanceSquaredTo(r.location) <= 16) allies++;
                if (G.cheb(r.location, me) <= Cfg.CASH_R) { cashN++; if (r.cheeseAmount > raw) cashHeavier++; }
            }
        }
        f(cap(foes, 15));
        f(cap(allies, 15));
        f(cap(raw, 1000));
        f(cap(G.globalCheese / 10, 2000));
        f(rc.getHealth());
        f(Baby.cashOk(G.globalCheese, Comms.numKings, G.round));
        f(Baby.cashEnd(G.globalCheese, Comms.numKings, G.round));
        f(G.round < Baby.cashBan);
        MapLocation cs = Baby.cashSite;
        f(cs != null);
        f(Baby.cashLead);
        f(cs != null ? cap(G.cheb(me, cs), 63) : 63);
        f(cs != null ? cap(G.round - Baby.cashStart, 255) : 255);
        f(Baby.cashFar(me));
        f(cap(cashN, 15));
        f(cap(cashHeavier, 15));
        int hc = 0, hm = 0, hs = 0;
        for (Message m : rc.readSqueaks(-1)) {
            int c = m.getBytes(), t = Comms.sqType(c);
            if (t == Comms.SQ_CASH && G.round - m.getRound() <= 2) hc++;
            else if (t == Comms.SQ_MINE) hm++;
            else if (t == Comms.SQ_SYM) hs++;
        }
        f(cap(hc, 15));
        f(cap(hm, 15));
        f(cap(hs, 15));
        f(me.x);
        f(me.y);
        f(G.W);
        f(G.H);
        sec = false;
    }

    /** MINE_PICK features (45, secondary slot): pickMine's candidates cand[0..n) (the first m eligible, ours on our
     *  half), before its final choice. Side-effect free. */
    static void minePick(int n, int ours, int m) {
        RobotController rc = G.rc;
        open2(MINE_PICK);
        MapLocation me = G.me;
        f(cap(m, 40));
        f(cap(n, 40));
        f(cap(ours, 40));
        f(m > 0 ? Baby.slot % m : 0);
        f(m > 0 ? cap(Baby.mineShift % m, 40) : 0);
        f(pickSite);
        f(Baby.role);
        f(G.coop);
        f(cap(G.round / 10, 255));
        f(Comms.numMines);
        f(cap(Baby.myMineCount, 24));
        f(Comms.numKings);
        f(cap(rc.getRawCheese(), 1000));
        for (int i = 0; i < 8; i++) {
            if (i >= m) { f(63); f(63); f(0); f(0); continue; }
            MapLocation c = Baby.cand[i];
            f(cap(G.cheb(me, c), 63));
            f(Baby.home != null ? cap(G.cheb(Baby.home, c), 63) : 63);
            f(Baby.kingHeld(c));
            int a = 0;
            for (RobotInfo r : Baby.seen) if (r.team == G.us && r.type == UnitType.BABY_RAT && G.cheb(r.location, c) <= 4) a++;
            f(cap(a, 15));
        }
        sec = false;
    }

    private D() {}
}
