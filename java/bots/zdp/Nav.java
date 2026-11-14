package zdp;

import battlecode.common.*;

/**
 * Facing-aware movement for baby rats (turn + forward = one tile per turn in any direction) and
 * greedy-with-bug-escape path choice. Bug escape starts only after greedy is blocked by terrain;
 * its handedness comes from {@link G#hand}, so mirrored rats circle mirrored ways.
 */
public final class Nav {
    static MapLocation target;
    static boolean bugging;
    static int bugStartDist;
    static Direction bugDir;
    static int bugHand;
    static int bugTurns;
    /** Turns since we last reduced the best distance to the current target (survives retargets to the same spot). */
    public static int stall;
    static int bestDist;
    /** Avoid-mask: bit d set means do not step in DIRS[d] this turn (cat lanes etc). */
    public static int avoid;
    public static int digs;

    /** Tile bits that block greedy steps and start bug walks: walls and dirt (our kings' bodies, Sym.KING, block
     *  flow paths only). A field so tests can set it. */
    static int terrain = Sym.WALL | Sym.DIRT;

    /** Static blocker (wall/dirt) from memory or sensing; robots are not static blockers. */
    static boolean terrainBlocked(MapLocation l) {
        if (!G.onMap(l.x, l.y)) return true;
        return (Sym.mem[G.idx(l)] & terrain) != 0;
    }

    static MapLocation[] kingMarks = new MapLocation[Comms.KING_SLOTS];
    static int kingMarkN;

    /** Keep Sym.KING on our kings' 3x3 bodies (from the shared array; Cfg.KING_FLOW): re-marked when a king moves,
     *  appears or dies. */
    public static void markKings() {
        int n = Comms.numKings;
        if (n == kingMarkN) {
            int k = 0;
            while (k < n && kingMarks[k].equals(Comms.kings[k])) k++;
            if (k == n) return;
        }
        for (int k = 0; k < kingMarkN; k++) body(kingMarks[k], 0);
        for (int k = 0; k < n; k++) { kingMarks[k] = Comms.kings[k]; body(kingMarks[k], Sym.KING); }
        kingMarkN = n;
    }

    static void body(MapLocation c, int bit) {
        int[] mem = Sym.mem;
        for (int x = Math.max(0, c.x - 1), x1 = Math.min(G.W - 1, c.x + 1); x <= x1; x++)
            for (int y = Math.max(0, c.y - 1), y1 = Math.min(G.H - 1, c.y + 1); y <= y1; y++) {
                int i = x + y * G.W;
                mem[i] = mem[i] & ~Sym.KING | bit;
            }
    }

    /** True when t is inside one of our kings' bodies (a delivery target): such paths may cross king tiles. */
    static boolean atKing(MapLocation t) {
        for (int k = 0; k < Comms.numKings; k++) if (G.cheb(Comms.kings[k], t) <= 1) return true;
        return false;
    }

    static boolean free(Direction d) {
        if ((avoid & (1 << d.ordinal())) != 0) return false;
        return G.rc.canMove(d);
    }

    /** Move one step toward t. Returns true if moved. */
    public static boolean goTo(MapLocation t) throws GameActionException {
        if (t == null) return false;
        if (target == null || !target.equals(t)) {
            target = t;
            bugging = false;
            bestDist = G.me.distanceSquaredTo(t);
            stall = 0;
        }
        if (G.me.equals(t)) return false;
        if (!G.rc.isMovementReady()) return false;
        // a dirt tile straight ahead with no closer free side step is dug and entered at once; flow (below) is for
        // stalls the local rule cannot see past. Not while a flow path is being followed or searched.
        if (!(Flow.ready(t) || Flow.building) && digToward(t)) {
            Direction d = G.me.directionTo(t);
            return free(d) && step(d);
        }
        int d0 = G.me.distanceSquaredTo(t);
        if (d0 < bestDist) { bestDist = d0; stall = 0; } else stall++;
        if (G.cheb(G.me, t) > 1) {
            if (stall >= Cfg.FLOW_STALL || bugging && bugTurns >= Cfg.FLOW_STALL) Flow.request(t);
            MapLocation was = G.me;
            if (Flow.follow(t)) return !G.me.equals(was);
            // no hold while the search runs: it grows from the target and takes the path from wherever we are, so
            // walk on. The call still gives up a search that has run too long.
            Flow.searching(t);
        }
        Direction d = choose(t);
        if (d == null) {
            tryDig(t);
            return false;
        }
        return step(d);
    }

    /** Enemy to keep facing while stepping (strafe instead of turning our back), or null. */
    public static MapLocation guard;

    /** Executes a step in direction d, turning first so the move is a forward move (or toward guard). */
    public static boolean step(Direction d) throws GameActionException {
        RobotController rc = G.rc;
        if (d == null || d == Direction.CENTER) return false;
        Direction f = d;
        if (guard != null) {
            Direction g = G.me.add(d).directionTo(guard);
            if (g != Direction.CENTER) f = g;
        }
        if (rc.getDirection() != f && rc.canTurn(f)) rc.turn(f);
        if (rc.canMove(d)) {
            rc.move(d);
            G.me = rc.getLocation();
            G.facing = rc.getDirection();
            return true;
        }
        return false;
    }

    static Direction choose(MapLocation t) {
        Direction toT = G.me.directionTo(t);
        if (!bugging) {
            if (free(toT)) return toT;
            Direction l = toT.rotateLeft(), r = toT.rotateRight();
            boolean fl = free(l), fr = free(r);
            if (fl && fr) {
                int dl = G.me.add(l).distanceSquaredTo(t), dr = G.me.add(r).distanceSquaredTo(t);
                if (dl != dr) return dl < dr ? l : r;
                return G.hand(G.me, toT) > 0 ? l : r;
            }
            if (fl) return l;
            if (fr) return r;
            // Blocked ahead. Only terrain starts a bug walk; robots just make us wait or sidestep.
            boolean terrain = terrainBlocked(G.me.add(toT)) || terrainBlocked(G.me.add(l)) && terrainBlocked(G.me.add(r));
            if (!terrain) {
                Direction l2 = l.rotateLeft(), r2 = r.rotateRight();
                int h = G.hand(G.me, toT);
                Direction a = h > 0 ? l2 : r2, b = h > 0 ? r2 : l2;
                if (stall > 2 && free(a)) return a;
                if (stall > 2 && free(b)) return b;
                return null;
            }
            bugging = true;
            bugStartDist = G.me.distanceSquaredTo(t);
            bugHand = G.hand(G.me, toT);
            // pretend we were already walking along the wall, so the first sweep starts at toT itself
            bugDir = bugHand > 0 ? toT.rotateLeft().rotateLeft() : toT.rotateRight().rotateRight();
            bugTurns = 0;
        }
        bugTurns++;
        if (bugTurns > 4 * (G.W + G.H)) { bugging = false; return null; }
        if (G.me.distanceSquaredTo(t) < bugStartDist && free(toT)) {
            bugging = false;
            return toT;
        }
        // Wall follow: start pointing back into the wall, then sweep away from it.
        Direction d = bugHand > 0 ? bugDir.rotateRight().rotateRight() : bugDir.rotateLeft().rotateLeft();
        for (int i = 0; i < 8; i++) {
            if (free(d)) { bugDir = d; return d; }
            d = bugHand > 0 ? d.rotateLeft() : d.rotateRight();
        }
        return null;
    }

    static boolean isDirt(MapLocation l) {
        return G.onMap(l.x, l.y) && (Sym.mem[G.idx(l)] & Sym.DIRT) != 0;
    }

    /**
     * Dirt straight toward the target is dug at once unless a free side step still gets closer:
     * a dig costs 5 cheese and 2.5 turns, a detour around a dirt field usually far more.
     */
    static boolean digToward(MapLocation t) throws GameActionException {
        if (G.globalCheese + G.rawCheese < Cfg.DIG_MIN_CHEESE) return false;
        RobotController rc = G.rc;
        if (!rc.isActionReady()) return false;
        Direction toT = G.me.directionTo(t);
        MapLocation n = G.me.add(toT);
        if (!isDirt(n) || free(toT)) return false;
        int d0 = G.me.distanceSquaredTo(t);
        Direction l = toT.rotateLeft(), r = toT.rotateRight();
        if (free(l) && G.me.add(l).distanceSquaredTo(t) < d0) return false;
        if (free(r) && G.me.add(r).distanceSquaredTo(t) < d0) return false;
        if (rc.getDirection() != toT && rc.canTurn(toT)) rc.turn(toT);
        if (rc.canRemoveDirt(n)) {
            dig(n);
            bugging = false;
            return true;
        }
        return false;
    }

    /** Dig through dirt when stalled and the straight step is dirt. */
    static void tryDig(MapLocation t) throws GameActionException {
        if (stall < 6 || G.globalCheese + G.rawCheese < 60) return;
        Direction toT = G.me.directionTo(t);
        MapLocation n = G.me.add(toT);
        RobotController rc = G.rc;
        if (rc.getDirection() != toT && rc.canTurn(toT)) rc.turn(toT);
        if (rc.canRemoveDirt(n)) dig(n);
    }

    /** Dig out the dirt tile n (the caller checked canRemoveDirt) and forget it in memory. */
    static void dig(MapLocation n) throws GameActionException {
        G.rc.removeDirt(n);
        digs++;
        Sym.mem[G.idx(n)] &= ~Sym.DIRT;
    }

    /** Step away from a threat location, preferring directions that increase distance most. */
    public static boolean flee(MapLocation threat) throws GameActionException {
        return flee2(2 * threat.x, 2 * threat.y);
    }

    /** flee() from a point in doubled coordinates (a 2x2 cat's centre, so mirror twins flee mirrored ways). */
    public static boolean flee2(int tx2, int ty2) throws GameActionException {
        if (!G.rc.isMovementReady()) return false;
        int ax = 2 * G.me.x - tx2, ay = 2 * G.me.y - ty2;
        Direction away = G.dirOf(ax, ay);
        if (away == Direction.CENTER) away = G.facing.opposite();
        Direction best = null;
        int bd = ax * ax + ay * ay;
        int h = G.hand(G.me, away);
        Direction[] cand = {away, h > 0 ? away.rotateLeft() : away.rotateRight(), h > 0 ? away.rotateRight() : away.rotateLeft(),
            h > 0 ? away.rotateLeft().rotateLeft() : away.rotateRight().rotateRight(),
            h > 0 ? away.rotateRight().rotateRight() : away.rotateLeft().rotateLeft()};
        for (Direction d : cand) {
            if (!G.rc.canMove(d)) continue;
            int dx = ax + 2 * d.dx, dy = ay + 2 * d.dy, dd = dx * dx + dy * dy;
            if (dd > bd) { bd = dd; best = d; }
        }
        return best != null && step(best);
    }
}
