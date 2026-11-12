package alpha0.student;

import battlecode.common.*;
import java.util.*;

/**
 * Called by the transformed RobotControllerImpl around every public primitive mutator.
 * Records (observation-before, legal mask, primitive index) for the one teacher robot whose
 * turn the harness has marked active. Robots run strictly one at a time, so static state is safe.
 * Runs outside instrumented bot code, so it is not charged to the teacher's bytecode budget.
 */
public final class Hook {
    private static final Object NESTED = new Object(), IGNORED = new Object();
    private static final class Pending { int label; boolean notLegal, king; }

    static volatile int activeId = -1;
    static int depth, step, team;
    static Obs obs;
    static Records sink;
    static final byte[] buf = new byte[Obs.SIZE];
    static final boolean[] mask = new boolean[Actions.N];
    public static final TreeMap<String, Integer> unmapped = new TreeMap<>(), ignored = new TreeMap<>();
    public static long recorded, labelNotLegal, failedCalls;
    public static Throwable error;

    public static void start(int robotId, int teamIndex, Obs o, Records r) {
        activeId = robotId; team = teamIndex; obs = o; sink = r; depth = 0; step = 0;
    }

    public static int stop() { activeId = -1; return step; }

    public static Object begin(RobotController rc, String op, Object[] args) {
        if (activeId < 0 || rc.getID() != activeId) return null;
        if (depth++ > 0) return NESTED;
        try {
            if (op.equals("squeak") || op.equals("writeSharedArray") || op.equals("disintegrate") || op.equals("resign")) {
                ignored.merge(op, 1, Integer::sum);
                return IGNORED;
            }
            int label = Actions.fromCall(rc, op, args);
            if (label < 0) { unmapped.merge(op, 1, Integer::sum); return IGNORED; }
            obs.encode(rc, step, buf);
            Actions.legal(rc, mask);
            Pending p = new Pending();
            p.label = label;
            p.notLegal = !mask[label];
            p.king = rc.getType() == UnitType.RAT_KING;
            return p;
        } catch (Throwable t) {
            if (error == null) error = t;
            return IGNORED;
        }
    }

    public static void finish(RobotController rc, Object token, Throwable failure) {
        if (token == null) return;
        depth--;
        if (token == NESTED || token == IGNORED) return;
        if (failure != null) { failedCalls++; return; }
        Pending p = (Pending) token;
        try {
            if (p.notLegal) { labelNotLegal++; mask[p.label] = true; }
            sink.write(activeId, rc.getRoundNum(), p.label, team, p.king,
                       step, p.notLegal, buf, mask);
            recorded++;
            step++;
        } catch (Throwable t) { if (error == null) error = t; }
    }

    private Hook() {}
}
