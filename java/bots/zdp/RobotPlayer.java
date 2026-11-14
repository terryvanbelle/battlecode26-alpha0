package zdp;

import battlecode.common.*;

/**
 * Turn loop. Compute monitor: an overrun shows as the round advancing during our own turn;
 * a near miss is a turn using more than 90% of the limit. Both are counted and shown in the
 * indicator string with greppable tags (OVR=, NEAR=, EXC=).
 */
public strictfp class RobotPlayer {
    public static int overruns, nearMiss, exceptions, maxBc;

    public static void run(RobotController rc) {
        G.debt = 0;
        G.init(rc);
        Sym.init();
        while (true) {
            int startRound = rc.getRoundNum();
            Debug.begin(startRound);
            try {
                G.startTurn();
                if (G.king) King.turn();
                else Baby.turn();
            } catch (Exception e) {
                caught(e);
            }
            try {
                if (!G.king && rc.getRoundNum() == startRound) Flow.work(Cfg.FLOW_RESERVE);
            } catch (Exception e) {
                caught(e);
            }
            int bc = Clock.getBytecodeNum();
            int limit = rc.getType() == UnitType.RAT_KING ? 20000 : 17500;
            Debug.mark(99);
            if (rc.getRoundNum() != startRound) overruns++;
            else {
                if (bc > maxBc) maxBc = bc;
                if (bc > limit * 9 / 10) nearMiss++;
            }
            if (Clock.getBytecodesLeft() >= Cfg.IND_LEFT) Debug.indicator();
            G.debt = -2;   // ledger: the next turn starts with no extra bytecodes (Debug.mark(99) above adds to it; the tag pads after)
            if (Debug.TAG && rc.getRoundNum() == startRound && Clock.getBytecodesLeft() >= Debug.TAG_MIN_LEFT) Debug.tag();
            Clock.yield();
        }
    }

    /** A turn stage threw: count it and keep it for the indicator. */
    static void caught(Exception e) {
        exceptions++;
        Debug.exception(e);
        if (Debug.ON) e.printStackTrace();
    }
}
