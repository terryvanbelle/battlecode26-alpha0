package zdp;

import battlecode.common.*;

/**
 * Decision points (docs/decision-points.md). At each point zdb's own code still decides (pass 1 is bug-for-bug zdb);
 * this class snapshots the point's features before the decision, notes which branch zdb took (lab), and sends the
 * record to the research harness through the indicator string (the harness intercepts strings that start with MAGIC).
 * Every hook's bytecodes are credited to the ledger (G.debt), so zdb's budget reads, and its games, are unchanged.
 *
 * Wire format (chars): [0] MAGIC, [1] point, [2] label (0xFFFF: none), [3] n, [4..4+n) features (unsigned 16-bit).
 */
public final class D {
    /** Send decision records to the harness (false in tournament builds). */
    static final boolean EMIT = true;
    /** Net mode (docs/decision-points.md): the decision net picks the option at each point (false: zdb decides). */
    static final boolean NET = false;
    static final char MAGIC = '\uDEC0';

    // decision points
    static final int BABY_MOVE = 1;

    // BABY_MOVE options: the branches of Baby.moveDecision, in zdb's priority order
    static final int M_GRAB = 0, M_CAT_DEFEND = 1, M_END_CAT = 2, M_FLEE_CAT = 3, M_YIELD_KING = 4, M_AIM_CARRY = 5,
        M_RESCUE = 6, M_AMMO_AWAY = 7, M_RETREAT = 8, M_HUNGRY = 9, M_FORM_KING = 10, M_CASH = 11, M_RAID = 12,
        M_FLANK = 13, M_PRESS = 14, M_MICRO = 15, M_FACE = 16, M_ATK_DELIVER = 17, M_STALL_CAT = 18, M_ATTACK = 19,
        M_WAR = 20, M_FORT = 21, M_FIGHT_CAT = 22, M_DELIVER = 23, M_COURIER = 24, M_CHEESE_PICK = 25, M_CHEESE = 26,
        M_KNOWN_CAT = 27, M_RAID_FIGHT = 28, M_SCOUT_HOME = 29, M_SCOUT = 30, M_SEEK = 31, M_MINE_EXPLORE = 32,
        M_IDLE_SEEK = 33, M_START_SEEK = 34, M_MINE_STILL = 35, M_MINE_WANDER = 36, M_MINE_GO = 37, M_EXPLORE = 38;
    static final int BABY_MOVE_OPTIONS = 39;

    static final char[] F = new char[72];
    static int n, point;
    /** The option the open point's decision took (set at each branch; -1: none yet). */
    static int lab = -1;

    static void open(int p) { F[0] = MAGIC; F[1] = (char) p; n = 4; point = p; lab = -1; }

    static void f(int v) { F[n++] = (char) (v < 0 ? 0 : v > 65535 ? 65535 : v); }

    static void f(boolean b) { F[n++] = b ? (char) 1 : (char) 0; }

    static int cap(int v, int c) { return v > c ? c : v; }

    /** Close the open point and send its record. */
    static void emit() {
        if (!EMIT || point == 0) return;
        F[2] = (char) (lab < 0 ? 0xFFFF : lab);
        F[3] = (char) (n - 4);
        G.rc.setIndicatorString(new String(F, 0, n));
        point = 0;
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
        f(Baby.rescueT != null);
        f(Baby.guardCarry);
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
    }

    private D() {}
}
