package zdn;

import battlecode.common.*;

/**
 * zdn memory registers (docs/zdn.md, "Memory registers"): the robot's commitments, written by the net at the start of
 * each turn (in teacher builds: zdb's own intent variables), and the code-kept counters about them ("rounds since
 * this register changed" and similar). cur* hold the values from the start of the turn, nxt* the values chosen for
 * this turn; commit() moves nxt into cur at the end of the turn.
 */
public final class ZR {
    // ---- baby registers
    public static final int R_ROLE = 0, R_MODE = 1, R_MINE = 2, R_EXPL = 3, R_CASH = 4, R_GOAL = 5, NB = 6;
    // ---- king registers
    public static final int R_HOME = 0, R_FORM = 1, NK = 2;

    public static final int ROLES = 4, GATHER = 0, SCOUT = 1, FIGHT = 2, ATTACK = 3;
    /** Modes: zdb's movement branches (zdp D.M_* numbering) plus NONE. */
    public static final int MODES = 40, M_NONE = 39;
    /** Goal kinds (GOAL register candidates). */
    public static final int G_NONE = 0, G_KING = 1, G_MINE = 2, G_CHEESE = 3, G_ENEMY = 4, G_EKING = 5, G_RUSH = 6,
        G_CATRING = 7, G_CAT = 8, G_ALERT = 9, G_FORM = 10, G_CASH = 11, G_EXPL = 12, G_CENTRE = 13, GOALS = 14;

    public static int curRole = GATHER, nxtRole = GATHER;
    public static int curMode = M_NONE, nxtMode = M_NONE;
    public static MapLocation curMine, nxtMine, curExpl, nxtExpl, curCash, nxtCash, curGoal, nxtGoal;
    public static int curGoalKind = G_NONE, nxtGoalKind = G_NONE;
    public static boolean curCashLead, nxtCashLead;
    public static int curCashStart, nxtCashStart;
    public static MapLocation curHome, nxtHome, curForm, nxtForm;

    // ---- counters
    public static int roleSince, modeSince, goalSince, mineSince, explSince, cashSince, cashHeard, homeSince, formOpen = -1000;
    public static final int[] modeLast = new int[MODES], modeCount = new int[MODES];
    public static int mineChanges, minesAtSet, myMinesAtSeek, goalBest = Integer.MAX_VALUE, goalStall, idle, cashBanUntil;
    public static boolean born;

    static {
        java.util.Arrays.fill(modeLast, -1000);
    }

    /** Begin a turn: the chosen values start as the current ones. */
    public static void begin() {
        nxtRole = curRole; nxtMode = curMode; nxtMine = curMine; nxtExpl = curExpl; nxtCash = curCash;
        nxtCashLead = curCashLead; nxtCashStart = curCashStart; nxtGoal = curGoal; nxtGoalKind = curGoalKind;
        nxtHome = curHome; nxtForm = curForm;
    }

    static boolean same(MapLocation a, MapLocation b) { return a == null ? b == null : a.equals(b); }

    /** End a turn: update the counters from what changed, then make the chosen values current. */
    public static void commit() {
        int r = ZG.round;
        MapLocation me = ZG.rc.getLocation();
        if (!born || nxtRole != curRole) roleSince = r;
        born = true;
        if (nxtMode != curMode) {
            modeSince = r;
            if (nxtMode >= 0 && nxtMode < MODES) { modeLast[nxtMode] = r; modeCount[nxtMode]++; }
            if (isSeek(nxtMode) && !isSeek(curMode)) myMinesAtSeek = ZM.myMineCount;
        }
        if (!same(nxtMine, curMine)) { mineSince = r; minesAtSet = ZC.numMines; if (curMine != null) mineChanges++; idle = 0; }
        if (!same(nxtExpl, curExpl)) explSince = r;
        if (!same(nxtCash, curCash)) {
            cashSince = r;
            if (nxtCash == null && curCash != null && r - curCashStart > 50) cashBanUntil = r + 50;
        }
        if (nxtCash != null) for (int i = 0; i < ZC.nCash; i++) if (ZC.cashSite[i].equals(nxtCash)) cashHeard = r;
        if (!same(nxtGoal, curGoal)) { goalSince = r; goalBest = nxtGoal == null ? Integer.MAX_VALUE : me.distanceSquaredTo(nxtGoal); goalStall = 0; }
        else if (nxtGoal != null) {
            int d = me.distanceSquaredTo(nxtGoal);
            if (d < goalBest) { goalBest = d; goalStall = 0; } else goalStall++;
        }
        // idle: turns near the mine since the last pickup (zdb's idleTurns, approximately)
        if (ZM.lastPick == r) idle = 0;
        else if (nxtMine != null && ZG.cheb(me, nxtMine) <= 4) idle++;
        if (!same(nxtHome, curHome)) homeSince = r;
        if (nxtForm != null && !same(nxtForm, curForm)) formOpen = r;
        curRole = nxtRole; curMode = nxtMode; curMine = nxtMine; curExpl = nxtExpl; curCash = nxtCash;
        curCashLead = nxtCashLead; curCashStart = nxtCashStart; curGoal = nxtGoal; curGoalKind = nxtGoalKind;
        curHome = nxtHome; curForm = nxtForm;
    }

    public static boolean isSeek(int m) { return m == 31 || m == 33 || m == 34; }

    public static int since(int round) { return Math.min(999, ZG.round - round); }
}
