package zdp;

import battlecode.common.*;

/**
 * Breadcrumbs home: the loop-free chain of tiles this baby has walked since it was spawned beside a king. Stepping
 * back onto a tile of the chain cuts off the loop after it, so the chain is always a simple, walked path from the
 * spawn tile to here: no walls on it, whatever the map. A carrier whose delivery stalls (Cfg.TRAIL) walks it back,
 * each turn to the free neighbour earliest on the chain, which also cuts corners where the chain passes close to
 * itself. A jump of more than one tile (carried, thrown) starts a new chain where the rat lands.
 */
public final class Trail {
    /** Chain tiles in walk order (tiles[0] = start), and tile -> chain position + 1 (0 = not on the chain). */
    static int[] tiles, at;
    public static int n;

    /** Extend the chain to G.me (once a turn, at its start). */
    public static void record() {
        int W = G.W;
        if (at == null) {
            int c = W * G.H;
            if (Flow.left() < 2 * c + Cfg.BC_OPTIONAL) return;
            at = new int[c];
            tiles = new int[c];
        }
        int i = G.me.x + G.me.y * W;
        if (n > 0) {
            int l = tiles[n - 1];
            if (l == i) return;
            if (Math.abs(l % W - G.me.x) > 1 || Math.abs(l / W - G.me.y) > 1) clear();
        }
        int p = at[i];
        if (p > 0) while (n > p) at[tiles[--n]] = 0;
        else { tiles[n++] = i; at[i] = n; }
    }

    static void clear() { while (n > 0) at[tiles[--n]] = 0; }

    static final int MOVED = 0, HELD = 1, END = 2;

    /** One step back along the chain, to the free neighbour (Nav.avoid respected) earliest on it, if that is earlier
     *  than here: MOVED; HELD when every earlier neighbour is occupied (or movement is not ready); END when no
     *  neighbour is earlier (the start of the chain, or no chain). */
    static int back() throws GameActionException {
        if (at == null || n < 2) return END;
        RobotController rc = G.rc;
        int W = G.W, x = G.me.x, y = G.me.y, p0 = at[x + y * W], best = p0 == 0 ? Integer.MAX_VALUE : p0;
        boolean earlier = false;
        Direction bd = null;
        for (Direction d : G.DIRS) {
            int vx = x + d.dx, vy = y + d.dy;
            if (vx < 0 || vy < 0 || vx >= W || vy >= G.H) continue;
            int p = at[vx + vy * W];
            if (p == 0 || p >= best) continue;
            earlier = true;
            if ((Nav.avoid & (1 << d.ordinal())) != 0 || !rc.canMove(d)) continue;
            best = p;
            bd = d;
        }
        if (bd == null) return earlier ? HELD : END;
        return Nav.step(bd) ? MOVED : HELD;
    }
}
