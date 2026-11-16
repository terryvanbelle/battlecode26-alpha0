package zdp;

import battlecode.common.*;

/**
 * Plan heads (design C, docs/rl-es-plan.md section 21): the net picks zdb's strategy constants (Plan) at coarse,
 * committed times, and zdb's code executes them every turn. A king calls plan() each turn after publishing; a plan
 * is due on the team's first king turn, in the first king turn of each 50-round epoch, when cooperation has ended
 * since the last plan, and when the live king count has changed. The first king to take its turn while one is due
 * opens the heads in order (TP_POSTURE .. TP_KINGS), one record each, and writes the new plan (Plan.write), so later
 * kings that round see it as current.
 *
 * Heads open only while they can bind: POSTURE on the first plan in cooperation (it holds for the game), RUSH before
 * Cfg.RUSH_UNTIL, KINGS through round 1150. A head that is not opened keeps its bin. Without a net for a head (any
 * build without one, D.pick -1) its label is zdb's bin, so such a plan reproduces zdb exactly. Every head sees the
 * same team features (FEATS), including the plan in force and the bins already drawn in this decision, so the heads
 * of one decision form an autoregressive policy. Exceptions drop the record and keep the head's bin.
 */
public final class DP {
    // decision points (ids 20..27; dp_points.py NAMES)
    static final int TP_POSTURE = 20, TP_ARMY = 21, TP_CORE = 22, TP_THRIFT = 23, TP_ROLES = 24, TP_RUSH = 25,
        TP_LOAD = 26, TP_KINGS = 27;
    // feature counts (literal: dp_points.py reads NAME_FEATS = int)
    static final int TP_POSTURE_FEATS = 40, TP_ARMY_FEATS = 40, TP_CORE_FEATS = 40, TP_THRIFT_FEATS = 40,
        TP_ROLES_FEATS = 40, TP_RUSH_FEATS = 40, TP_LOAD_FEATS = 40, TP_KINGS_FEATS = 40;
    // option counts (Plan.K)
    static final int TP_POSTURE_OPTIONS = 7, TP_ARMY_OPTIONS = 7, TP_CORE_OPTIONS = 7, TP_THRIFT_OPTIONS = 7,
        TP_ROLES_OPTIONS = 7, TP_RUSH_OPTIONS = 6, TP_LOAD_OPTIONS = 7, TP_KINGS_OPTIONS = 4;
    /** A plan's cause (feature 2). */
    static final int PC_FIRST = 0, PC_EPOCH = 1, PC_BREAK = 2, PC_KINGS = 3;
    static final int PLAN_EVERY = 50, KINGS_UNTIL = 1150;

    /** The first round this king saw cooperation over (-1: not yet). */
    static int coopEnd = -1;

    /** Live kings from the array's heartbeats (as Comms.readKings counts them, after this king's publish). */
    static int liveKings() {
        int n = 0;
        for (int k = 0; k < Comms.KING_SLOTS; k++) {
            int b = Comms.KING0 + 3 * k;
            if (Comms.read(b) != 0 && Comms.fresh(Comms.read(b + 2), 2)) n++;
        }
        return n;
    }

    static void plan() {
        Plan.read();
        if (!G.coop && coopEnd < 0) coopEnd = G.round;
        int ep = G.round / PLAN_EVERY, kings = Math.min(7, liveKings());
        int cause = Plan.stamp == 0 ? PC_FIRST : Plan.stamp != ((ep + 1) & 63) ? PC_EPOCH
            : Plan.coopAt && !G.coop ? PC_BREAK : Plan.kingsAt != kings ? PC_KINGS : -1;
        if (cause < 0) return;
        int[] nb = Plan.bin.clone(), drawn = new int[Plan.HEADS];
        RobotInfo[] enemies = null;
        for (int h = 0; h < Plan.HEADS; h++) {
            if (h == Plan.POSTURE && !(cause == PC_FIRST && G.coop)) continue;
            if (h == Plan.RUSH && G.round >= Cfg.RUSH_UNTIL) continue;
            if (h == Plan.KINGS && G.round > KINGS_UNTIL) continue;
            int b = nb[h];
            if (D.ON) {
                try {
                    if (enemies == null) enemies = G.rc.senseNearbyRobots(-1, G.them);
                    D.open(TP_POSTURE + h);
                    feats(cause, kings, drawn, enemies);
                    int k = D.NET ? D.pick(TP_POSTURE + h, null) : -1;
                    b = k >= 0 && k < Plan.K[h] ? k : Plan.ZDB[h];
                    D.lab = b;
                    D.emit();
                } catch (Exception e) { D.point = 0; b = nb[h]; }
            }
            nb[h] = b;
            drawn[h] = b + 1;
        }
        Plan.write(nb, ep, G.coop, kings);
    }

    /** The team features (FEATS = 40), scaled by fixed constants. */
    static void feats(int cause, int kings, int[] drawn, RobotInfo[] enemies) {
        RobotController rc = G.rc;
        D.f(G.round / 10);                                              // 0
        D.f(G.coop);                                                    // 1
        D.f(cause);                                                     // 2
        D.f(coopEnd < 0 ? 0 : D.cap(G.round - coopEnd, 2000));          // 3 rounds since cooperation ended
        D.f(D.cap((rc.getCurrentRatCost() - 10) / 10 * 4, 255));        // 4 live babies (cost-step lower bound)
        D.f(D.cap(King.spawned, 255));                                  // 5
        D.f(kings);                                                     // 6
        D.f(D.cap(rc.getGlobalCheese() / 10, 2000));                    // 7
        D.f(Econ.surplus() / 10 + 32768);                               // 8
        D.f(D.cap(Econ.rate100() / 10, 65535));                         // 9
        D.f(Comms.read(Comms.BUDGET));                                  // 10
        D.f(G.W);                                                       // 11
        D.f(G.H);                                                       // 12
        D.f(G.W * G.H / 10);                                            // 13
        D.f(Sym.mask);                                                  // 14
        MapLocation st = King.start == null ? G.me : King.start;
        D.f(G.cheb(st, new MapLocation(G.W / 2, G.H / 2)));             // 15 our start's distance to the centre
        D.f(Comms.numMines);                                            // 16
        int near = 0;
        for (int k = 0; k < Comms.numMines; k++) if (st.distanceSquaredTo(Comms.mines[k]) <= 400) near++;
        D.f(near);                                                      // 17 known mines within 20 of our start
        int cats = 0;
        for (int k = 0; k < Comms.CAT_SLOTS; k++) if (Comms.fresh(Comms.read(Comms.CAT0 + 3 * k + 2), 50)) cats++;
        D.f(cats);                                                      // 18
        int es = Comms.read(Comms.EKING + 2);
        D.f(es == 0 ? 1023 : D.age(es));                                // 19 rounds since an enemy king was seen
        D.f(Comms.fresh(Comms.read(Comms.ALERT + 2), 3));               // 20
        D.f(Comms.alertThreat());                                       // 21
        D.f(D.cap(enemies.length, 15));                                 // 22 enemies in this king's sight
        D.f(D.cap(rc.getCurrentRatCost(), 1000));                       // 23
        for (int h = 0; h < Plan.HEADS; h++) D.f(Plan.bin[h]);          // 24..31 the plan in force
        for (int h = 0; h < Plan.HEADS; h++) D.f(drawn[h]);             // 32..39 this decision's draws (bin + 1; 0: none)
    }

    private DP() {}
}
