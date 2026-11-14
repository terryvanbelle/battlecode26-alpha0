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
 * RL builds (RL true) append a trailer to the features, counted in n (docs/rl-plan.md section 4): k (0xFFFF: no pick),
 * q = min(65535, round(-ln p_k * 4096)), flags (bit0 fallback, bit1 secondary slot), ceil(nOpt/16) mask words (bit i of
 * word w: option 16w + i allowed; all ones when the mask is null), NET_VERSION, trailerLen = 5 + ceil(nOpt/16)
 * (scripts/decisions.py split_trailer/read_rl).
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
    /** RL build: append the sampling trailer to each record (recording only; build_dpnet.py --rl sets it). */
    static final boolean RL = false;
    /** RL build: the net version written into each trailer (build_dpnet.py --version). */
    static final int NET_VERSION = 0;

    // decision points (BABY_* 1-9; king points 10-19 live in DK)
    static final int BABY_MOVE = 1;
    static final int BABY_ACT_PRE = 2;
    static final int BABY_ACT_POST = 3;
    static final int BABY_FACE = 4;
    static final int BABY_ROLE = 5;
    static final int BABY_COMMS = 6;
    static final int MINE_PICK = 7;
    static final int BABY_FREE = 8;   // pass 3: cheese pickup and own-trap removal (free actions)
    static final int BABY_VAL = 9;    // pass 3: a base-32 value digit (secondary slot; BABY_COMMS C_RAW's squeak)

    // feature counts per point (scripts/train_dp.py feature_counts; a record's n minus its RL trailer)
    static final int BABY_MOVE_FEATS = 76, BABY_ACT_PRE_FEATS = 74, BABY_ACT_POST_FEATS = 74, BABY_FACE_FEATS = 32,
        BABY_ROLE_FEATS = 112, BABY_COMMS_FEATS = 53, MINE_PICK_FEATS = 45, BABY_FREE_FEATS = 36, BABY_VAL_FEATS = 30;

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
    /** Pass 3: step in G.DIRS[i] without turning (strafe; avoid-free tiles only), and a step onto a tile Nav.avoid
     *  forbids (the exact complement of P_STEP). */
    static final int P_STRAFE0 = 56, P_RISK0 = 64;
    static final int BABY_MOVE_NET_OPTIONS = 72;

    // BABY_ACT_PRE / BABY_ACT_POST options (shared): the action behaviour that fired (A_X_*: primitives zdb never takes)
    static final int A_PASS = 0, A_DROP_ALLY = 1, A_THROW = 2, A_NAP = 3, A_BITE_KING = 4, A_BITE_RAT = 5,
        A_BITE_CAT = 6, A_CAT_TRAP = 7, A_LAY_LANE = 8, A_DELIVER = 9, A_CASH_KING = 10, A_RAT_TRAP = 11,
        A_X_BITE_FWD = 12, A_X_CARRY_FWD = 13, A_X_THROW_NOW = 14, A_X_DROP_FWD = 15, A_X_RAT_TRAP_FWD = 16,
        A_X_CAT_TRAP_FWD = 17, A_X_DIRT_PLACE_FWD = 18, A_X_DIRT_REMOVE_FWD = 19, A_X_BECOME_KING = 20;
    /** Pass 3: bite forward with cheese BITE_C[i] (A_X_BITE_C0 + i), drop the carried rat in G.DIRS[i] (not forward),
     *  transfer 10 / 50 / half the raw cheese to the nearest allied king tile, disintegrate (Cfg.DP_ALLOW_DISINTEGRATE). */
    static final int A_X_BITE_C0 = 21, A_X_DROP_DIR0 = 27, A_X_XFER_10 = 35, A_X_XFER_50 = 36, A_X_XFER_HALF = 37,
        A_X_DISINTEGRATE = 38;
    static final int[] BITE_C = {1, 4, 9, 16, 36, 100};
    static final int BABY_ACT_PRE_OPTIONS = 39;
    static final int BABY_ACT_POST_OPTIONS = 39;
    static final int BABY_ACT_OPTIONS = 39;

    // BABY_FACE options: guardTurn's target (F_DIR0 + i: face G.DIRS[i], primitives)
    static final int F_NONE = 0, F_SEEN = 1, F_MEMORY = 2, F_HEARD = 3, F_GUARD = 4, F_DIR0 = 5;
    static final int BABY_FACE_OPTIONS = 13;

    // BABY_ROLE options
    static final int R_GATHER = 0, R_SCOUT = 1, R_FIGHT = 2, R_RAIDER = 3, R_ATTACK = 4;
    /** Pass 3: keep the current role (a review's only encoding of "no change": the current role is masked). */
    static final int R_KEEP = 5;
    static final int BABY_ROLE_OPTIONS = 6;

    // BABY_COMMS options: the squeak this turn
    static final int C_NONE = 0, C_MINE_GOSSIP = 1, C_MINE_RELAY = 2, C_SYM_KING = 3, C_MINE_KING = 4, C_CAT = 5,
        C_CASH_LEAD = 6, C_CASH_RELAY = 7, C_EKING = 8, C_SYM_GOSSIP = 9, C_BURN = 10, C_TRIP = 11, C_TRAP = 12;
    /** Pass 3: a 32-bit squeak chosen as 7 BABY_VAL digits; re-squeak the newest net-type squeak heard. */
    static final int C_RAW = 13, C_RELAY = 14;
    static final int BABY_COMMS_OPTIONS = 15;

    // MINE_PICK options: rank i in pickMine's candidate order
    static final int MINE_PICK_OPTIONS = 40;

    // BABY_FREE options (pass 3): zdb's pickup here, take all, take none, take up to a cap of raw cheese (this turn),
    // then (chained, at most FREE_CHAIN) remove our rat / cat trap on our tile or fan tile 0..2 (G.fan order)
    static final int FR_ZDB = 0, FR_ALL = 1, FR_NONE = 2, FR_CAP0 = 3, FR_RM0 = 8, FR_RM_RAT0 = 8, FR_RM_CAT0 = 12;
    static final int[] FREE_CAP = {10, 25, 50, 100, 200};
    static final int FREE_CHAIN = 2;
    static final int BABY_FREE_OPTIONS = 16;

    // BABY_VAL options: a base-32 value digit
    static final int BABY_VAL_OPTIONS = 32;

    static final char[] F = new char[192], F2 = new char[192];
    static int n, point;
    /** The option the open point's decision took (set at each branch; -1: none yet). */
    static int lab = -1;
    /** The secondary record slot. */
    static int n2, point2, lab2 = -1;
    /** f() writes to the secondary slot. */
    static boolean sec;
    /** MINE_PICK: which moveDecision call site asked (0 fresh/reset, 1 IDLE_SHIFT, 2 retry). */
    static int pickSite;
    /** RL build: the net's sample for the open primary / secondary point (DNet's last*): option (-1: none), its
     *  probability, the option count (0: no sample) and a copy of its mask (null: all allowed). */
    static int pk = -1, pn, pk2 = -1, pn2;
    static float pp, pp2;
    static boolean[] pmask, pmask2;

    static void open(int p) { F[0] = MAGIC; F[1] = (char) p; n = 4; point = p; lab = -1; sec = false; if (RL) { pk = -1; pn = 0; pmask = null; } }

    static void open2(int p) { F2[0] = MAGIC; F2[1] = (char) p; n2 = 4; point2 = p; lab2 = -1; sec = true; if (RL) { pk2 = -1; pn2 = 0; pmask2 = null; } }

    /** RL build: copy DNet's last sample into the primary (second false) or secondary slot. */
    static void take(boolean second) {
        boolean[] m = DNet.lastMask == null ? null : DNet.lastMask.clone();
        if (second) { pk2 = DNet.lastK; pp2 = DNet.lastP; pn2 = DNet.lastN; pmask2 = m; }
        else { pk = DNet.lastK; pp = DNet.lastP; pn = DNet.lastN; pmask = m; }
    }

    /** RL build: write the trailer into B at `at`; returns the new length (unchanged when it would not fit). */
    static int trailer(char[] B, int at, int k, float p, int nOpt, boolean[] mask, int flags) {
        int words = (nOpt + 15) >> 4;
        if (at + 5 + words > B.length) return at;
        B[at++] = (char) (k < 0 ? 0xFFFF : k);
        long q = k < 0 ? 0 : Math.round(-Math.log(p) * 4096);
        B[at++] = (char) (q < 0 ? 0 : q > 65535 ? 65535 : q);
        B[at++] = (char) flags;
        for (int w = 0; w < words; w++) {
            int v = 0;
            if (mask == null) v = 0xFFFF;
            else for (int i = 0; i < 16; i++) { int o = 16 * w + i; if (o < nOpt && o < mask.length && mask[o]) v |= 1 << i; }
            B[at++] = (char) v;
        }
        B[at++] = (char) NET_VERSION;
        B[at++] = (char) (5 + words);
        return at;
    }

    /** RL build: the fallback flag (zdb's code ran instead of the net's option); a diagnostic only. */
    static int fallback(int p, int k, int l) {
        if (p == BABY_MOVE) return l >= 0 && (l & 0x8000) != 0 ? 1 : 0;
        return k >= 0 && l != k ? 1 : 0;
    }

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
        if (RL) n = trailer(F, n, pk, pp, pn, pmask, fallback(point, pk, lab));
        F[3] = (char) (n - 4);
        G.rc.setIndicatorString(new String(F, 0, n));
        point = 0;
    }

    /** Close the open secondary point and send its record. */
    static void emit2() {
        if (!EMIT || point2 == 0) return;
        F2[2] = (char) (lab2 < 0 ? 0xFFFF : lab2);
        if (RL) n2 = trailer(F2, n2, pk2, pp2, pn2, pmask2, fallback(point2, pk2, lab2) | 2);
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
        if (z == null) return -1;
        int k = DNet.sample(z, fit(ok, z.length, DNet.reach(p)), NET_T, DNet.floor(p), alt(p));
        if (RL) take(false);
        return k;
    }

    /** Net mode: the net's option for the open secondary point. */
    static int pick2(int p, boolean[] ok) {
        float[] z = DNet.logits(p, F2, n2);
        if (z == null) return -1;
        int k = DNet.sample(z, fit(ok, z.length, DNet.reach(p)), NET_T, DNet.floor(p), alt(p));
        if (RL) take(true);
        return k;
    }

    /** Pass 3's new points sample from DNet's second stream. */
    static boolean alt(int p) { return p == BABY_FREE || p == BABY_VAL || p >= DK.KING_COMMS; }

    /** Option k of point p is reachable in this build (gates behaviour that is not an option, e.g. role resets). */
    static boolean has(int p, int k) { return k < DNet.reach(p); }

    /** fit(ok, len) cut to the first act options (act >= len: unchanged, null stays null). */
    static boolean[] fit(boolean[] ok, int len, int act) {
        if (act >= len) return fit(ok, len);
        boolean[] a = new boolean[len];
        for (int i = 0; i < act; i++) a[i] = ok == null || i < ok.length && ok[i];
        return a;
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
        // appended (53..63): the state behind labels and counters the features above miss (pass 2C)
        f(Baby.idleDue());                       // Cfg.SEEK_IDLE: the idle tick trips this turn (M_IDLE_SEEK)
        f(cap(Baby.idleCount(), 63));
        f(Baby.catHunt);                         // Cfg.STALL_CAT (M_STALL_CAT)
        f(Baby.stallAt != null && G.cheb(me, Baby.stallAt) <= 3 ? cap(G.round - Baby.stallSince, 255) : 0);
        f(Baby.cheeseTarget != null && Nav.isDirt(Baby.cheeseTarget));   // M_CHEESE_PICK vs M_CHEESE
        f(Baby.unreported() >= 0);               // M_SCOUT_HOME, seek's carry-home
        t = Baby.formSite();
        f(t != null && me.distanceSquaredTo(t) <= Cfg.FORM_PULL_R2);   // M_FORM_KING's pull
        f(Baby.mineTarget != null && Baby.atMine(Baby.mineTarget));   // M_START_SEEK / M_MINE_WANDER vs M_MINE_GO
        f(Baby.mineTarget != null && Baby.nearerAllies(Baby.mineTarget) >= Cfg.SEEK_CROWD);   // M_START_SEEK's crowd
        f(cat != null && Baby.catAtKing(cat.location));   // M_CAT_DEFEND
        f(G.round < Baby.seekUntil);             // a seek is running (M_SEEK; also feature 31)
        cb(me);                                  // pass 3 (60..75): the comms block
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
        // pass 3 (53..57): the new act options' state, then the comms block (58..73)
        f(fr == null ? 0 : cap(fr.health, 255));
        int dm = 0;
        if (c != null) for (int i = 0; i < 8; i++) if (rc.canDropRat(G.DIRS[i])) dm |= 1 << i;
        f(dm);
        f(on && rc.canAttack(fl));
        int kd = 255;
        for (RobotInfo r : robots) {
            if (r.team != G.us || r.type != UnitType.RAT_KING) continue;
            for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) {
                int d = me.distanceSquaredTo(r.location.translate(dx, dy));
                if (d < kd) kd = d;
            }
        }
        f(kd);
        f(trapCode(me, fl, on));
        cb(me);
    }

    /** Own traps on our tile and the tile ahead: bit0 rat trap here, bit1 cat trap here, bit2 / bit3 the same ahead. */
    static int trapCode(MapLocation me, MapLocation fl, boolean on) {
        RobotController rc = G.rc;
        return (rc.canRemoveRatTrap(me) ? 1 : 0) | (rc.canRemoveCatTrap(me) ? 2 : 0)
            | (on && rc.canRemoveRatTrap(fl) ? 4 : 0) | (on && rc.canRemoveCatTrap(fl) ? 8 : 0);
    }

    /** BABY_FACE features (34): guardTurn's view. Opens on every turning-ready turn. Side-effect free. */
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
        // appended (31..33): the point opens in coop and while carrying too (pass 2C), where zdb keeps F_NONE
        f(G.coop);
        RobotInfo c = rc.getCarrying();
        f(c == null ? 0 : c.team == G.us ? 1 : 2);
        f(Nav.guard != null);
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
        cb(me);                                   // pass 3 (32..47): the comms block
        for (int i = 0; i < 64; i++) f(Comms.read(i));   // (48..111): the raw shared array
    }

    /** BABY_ROLE net mask (pass 3): ph 0 the five roles; at a review with R_KEEP reachable, KEEP and the four other
     *  roles (one encoding of "no change"); otherwise (a net without R_KEEP) the five roles, as before. */
    static boolean[] roleOk(int ph) {
        boolean[] a = new boolean[BABY_ROLE_OPTIONS];
        boolean keep = ph == 1 && has(BABY_ROLE, R_KEEP);
        int cur = keep ? roleLab() : -1;
        for (int i = 0; i < R_KEEP; i++) a[i] = i != cur;
        a[R_KEEP] = keep;
        return a;
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
        cb(me);                                   // pass 3 (37..52): the comms block
        sec = false;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // pass 3: read side of comms, BABY_FREE, BABY_VAL

    /** Age in rounds of a round%1024 stamp (Comms.stamp). */
    static int age(int stamp) { return ((G.round & 1023) - stamp + 1024) & 1023; }

    /** A zdb-protocol squeak type (Comms.SQ_*); every other top nibble is "net-type". */
    static boolean zdbType(int t) { return t >= 1 && t <= 4 || t == Comms.SQ_CASH; }

    /** Squeaks heard in the last 2 rounds (side-effect free; filled by hear()): H[0] count (cap 15), H[1] newest
     *  net-type squeak's age (7: none), H[2] / H[3] its content's high / low 16 bits, H[4] its sender's dist2 (cap
     *  255), H[5] zdb-type count (cap 15), H[6] the newest net-type sender is a live king, H[7] the newest king
     *  squeak's type + 1 (0: none). HM: the newest net-type message (null: none). */
    static final int[] H = new int[8];
    static Message HM;

    static boolean kingAt(MapLocation l) {
        for (int k = 0; k < Comms.numKings; k++) if (Comms.kings[k] != null && Comms.kings[k].equals(l)) return true;
        return false;
    }

    static void hear(MapLocation me) {
        Message nn = null, nk = null;
        int cnt = 0, zc = 0;
        for (Message m : G.rc.readSqueaks(-1)) {
            if (G.round - m.getRound() > 2) continue;
            cnt++;
            if (zdbType(Comms.sqType(m.getBytes()))) zc++;
            else if (nn == null || m.getRound() >= nn.getRound()) nn = m;
            if (m.getSource() != null && kingAt(m.getSource()) && (nk == null || m.getRound() >= nk.getRound())) nk = m;
        }
        HM = nn;
        H[0] = cap(cnt, 15);
        H[1] = nn == null ? 7 : G.round - nn.getRound();
        H[2] = nn == null ? 0 : nn.getBytes() >>> 16;
        H[3] = nn == null ? 0 : nn.getBytes() & 0xFFFF;
        H[4] = nn == null || nn.getSource() == null ? 255 : cap(me.distanceSquaredTo(nn.getSource()), 255);
        H[5] = cap(zc, 15);
        H[6] = nn != null && nn.getSource() != null && kingAt(nn.getSource()) ? 1 : 0;
        H[7] = nk == null ? 0 : Comms.sqType(nk.getBytes()) + 1;
    }

    /** The heard block (6). */
    static void heard(MapLocation me) {
        hear(me);
        for (int i = 0; i < 6; i++) f(H[i]);
    }

    /** The baby comms block CB (16): registers R0-R2 (slots 20-22), raw BUDGET, ALERT age / threat / distance,
     *  formation age, the heard block, then H[6] and H[7]. Side-effect free. */
    static void cb(MapLocation me) {
        f(Comms.read(Comms.EKING));
        f(Comms.read(Comms.EKING + 1));
        f(Comms.read(Comms.EKING + 2));
        f(Comms.read(Comms.BUDGET));
        int st = Comms.read(Comms.ALERT + 2);
        f(st == 0 ? 63 : cap(age(st), 63));
        f(Comms.read(Comms.ALERT) >> 6);
        MapLocation al = Comms.alertLoc();
        f(al != null ? cap(G.cheb(me, al), 63) : 63);
        f(Comms.read(Comms.FORM_X) == 0 ? 63 : cap(age(Comms.read(Comms.FORM_T)), 62));
        heard(me);
        f(H[6]);
        f(H[7]);
    }

    /** The king block (8): the heard block, then registers R0 and R1. */
    static void kmini(MapLocation me) {
        heard(me);
        f(Comms.read(Comms.EKING));
        f(Comms.read(Comms.EKING + 1));
    }

    /** Cheese pickable at l (0: none, or out of reach). */
    static int pickable(MapLocation l) throws GameActionException {
        RobotController rc = G.rc;
        return rc.onTheMap(l) && rc.canPickUpCheese(l) ? rc.senseMapInfo(l).getCheeseAmount() : 0;
    }

    /** BABY_FREE's tiles: our own, then G.fan's three (the tiles pickUp scans). */
    static MapLocation[] freeTiles() {
        MapLocation c = G.rc.getLocation();
        Direction[] fr = G.fan(c, G.rc.getDirection());
        return new MapLocation[]{c, c.add(fr[0]), c.add(fr[1]), c.add(fr[2])};
    }

    /** BABY_FREE (36): opens at pickup site ph (0 before the role review, 1 after the move) when cheese is pickable
     *  or an own trap is removable on freeTiles, or always when step > 0 (the chain reopens it). True when open. */
    static boolean babyFree(int ph, int step) {
        try { return babyFree0(ph, step); } catch (Exception e) { point = 0; return false; }
    }

    static boolean babyFree0(int ph, int step) throws GameActionException {
        RobotController rc = G.rc;
        MapLocation[] ts = freeTiles();
        int[] av = new int[4];
        int reach = 0, traps = 0;
        for (int i = 0; i < 4; i++) {
            MapLocation l = ts[i];
            boolean on = rc.onTheMap(l);
            av[i] = pickable(l);
            reach += av[i];
            traps |= ((on && rc.canRemoveRatTrap(l) ? 1 : 0) | (on && rc.canRemoveCatTrap(l) ? 2 : 0)) << (2 * i);
        }
        if (step == 0 && reach == 0 && traps == 0) return false;
        open(BABY_FREE);
        MapLocation me = ts[0];
        f(step);
        f(ph);
        f(roleLab());
        int raw = rc.getRawCheese();
        f(cap(raw, 1000));
        f(cap(reach, 1000));
        for (int i = 0; i < 4; i++) f(cap(av[i], 1023));
        f(traps);
        f(cap(G.globalCheese / 10, 2000));
        f(G.coop);
        f(G.round / 10);
        f(rc.getHealth());
        f(rc.isMovementReady());
        f(rc.isActionReady());
        RobotInfo c = rc.getCarrying();
        f(c == null ? 0 : c.team == G.us ? 1 : 2);
        RobotInfo cat = Baby.cat;
        f(cat != null);
        f(cat != null ? cap(Baby.catDist4(cat.location, me), 400) : 400);
        int nd = 100;
        for (RobotInfo r : Baby.seen) if (r.team == G.them) nd = Math.min(nd, me.distanceSquaredTo(r.location));
        f(nd);
        cb(me);
        return true;
    }

    /** BABY_VAL (30, secondary slot): digit pos of a value chosen for purpose (0: a squeak), prefix the digits so
     *  far. Side-effect free. */
    static void babyVal(int purpose, int pos, int prefix) {
        RobotController rc = G.rc;
        open2(BABY_VAL);
        MapLocation me = G.me;
        f(purpose);
        f(pos);
        f(prefix >>> 16);
        f(prefix & 0xFFFF);
        f(roleLab());
        f(G.round / 10);
        f(cap(rc.getRawCheese(), 1000));
        f(rc.getHealth());
        MapLocation k = Comms.nearestKing(me);
        f(k != null ? cap(G.cheb(me, k), 63) : 63);
        f(Baby.cat != null);
        RobotInfo ne = Baby.nearestEnemy;
        f(ne != null ? cap(me.distanceSquaredTo(ne.location), 100) : 100);
        f(me.x);
        f(me.y);
        RobotInfo c = rc.getCarrying();
        f(c == null ? 0 : c.team == G.us ? 1 : 2);
        cb(me);
        sec = false;
    }

    /** Net mode: a bits-wide value as base-32 values, most significant first (the first masked to its bits), each a
     *  BABY_VAL (baby) or DK.KING_VAL (king) pick in the secondary slot, each its own record. dfail: a pick failed
     *  (no net, or the features threw); the value is then 0. */
    static boolean dfail;

    static int digits(int bits, int purpose, boolean king, int idx, int trapMask) {
        int nd = (bits + 4) / 5, v = 0;
        dfail = false;
        for (int d = 0; d < nd; d++) {
            int b = bits - 5 * (nd - 1 - d);
            if (b > 5) b = 5;
            boolean[] ok = new boolean[32];
            for (int i = 0; i < 32; i++) ok[i] = i < (1 << b) && (trapMask == 0 || (trapMask >> i & 1) != 0);
            int p = king ? DK.KING_VAL : BABY_VAL, k = -1;
            if (king) DK.kingVal(purpose, d, nd, v, idx, trapMask); else babyVal(purpose, d, v);
            if (point2 == p) k = pick2(p, ok);
            if (k < 0) { point2 = 0; dfail = true; return 0; }
            lab2 = k;
            emit2();
            v = v << 5 | k;
        }
        return v;
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
