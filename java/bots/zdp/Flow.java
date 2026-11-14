package zdp;

import battlecode.common.*;

/**
 * Dig-aware path planning for stuck rats: a reverse A* from a target back toward the rat over remembered terrain
 * (unknown = open, dirt = DIG_COST, wall = blocked), built with spare bytecode over one or more turns, then
 * followed step by step (digging dirt on the path). Requested only after greedy/bug navigation stalls, so open
 * maps pay nothing. g[] entries are gen*SPAN + cost, so a new search needs no reset pass; Dial buckets on
 * f = g + chebyshev (f grows by at most DIG_COST + 1 per step).
 */
public final class Flow {
    static final int SPAN = 1 << 16, RING = 1024, RMASK = RING - 1, NB = 8, BMASK = NB - 1;
    /** Neighbour order of the search (par[] holds the index k of the step taken). */
    static final int[] DX = {0, 1, 1, 1, 0, -1, -1, -1}, DY = {1, 1, 0, -1, -1, -1, 0, 1};
    static int[] g, par;
    static int gen;
    static MapLocation tgt;
    static int tIdx, tx, ty, base;
    static boolean building;
    /** Tile bits a path may not cross (the target tile excepted): walls, and our kings' bodies (Cfg.KING_FLOW) unless
     *  the target is inside one. */
    static int block = Sym.WALL;
    /** Round the current path was completed, or -1. */
    static int builtRound = -1;
    static int[] path, buf;
    static int plen, ppos;
    /** The rat's tile when a reverse search began: the goal its heuristic aims at. The search grows from the target
     *  back toward the rat, so every tile it reaches knows its way to the target; the rat keeps walking meanwhile and
     *  the path starts where it ends up. */
    static int sx0, sy0;
    /** Distance table: AB[d + AC] = |d| for |d| <= AC (maps are at most 60 wide), built once with the search arrays. */
    static final int AC = 64;
    static int[] AB;
    static int[][] bucket;
    static int[] head = new int[NB], tail = new int[NB];
    static int curF, pending, held;
    public static int builds, steps, done, lastTurns, fails;
    /** Debug counts of how paths and searches ended: 0 reached the end, 1 held by a robot, 2 wall on the path,
     *  3 drifted off it, 4 replaced while ready (new target), 5 search replaced before it finished. */
    public static final int[] ends = new int[6];
    static int startRound, failRound;
    static MapLocation failTgt;

    static int left() { return Clock.getBytecodesLeft() + G.debt; }

    /** True when a complete, fresh path toward t exists. */
    public static boolean ready(MapLocation t) {
        return builtRound >= 0 && tgt != null && G.cheb(tgt, t) <= 1 && G.round - builtRound <= Cfg.FLOW_TTL;
    }

    /** True while a search toward t is running. */
    public static boolean searching(MapLocation t) {
        if (building && G.round - startRound > Cfg.FLOW_REV_TURNS) {
            // too slow (little spare bytecode, or a huge open search): give up so the rat moves again
            building = false;
            fails++;
            failTgt = tgt;
            failRound = G.round;
        }
        return building && tgt != null && G.cheb(tgt, t) <= 1;
    }

    /** Ask for a path toward t (no-op if one is ready or already being built toward t). */
    public static void request(MapLocation t) {
        if (tgt != null && G.cheb(tgt, t) <= 1 && (building || ready(t))) return;
        if (failTgt != null && G.cheb(failTgt, t) <= 1 && G.round - failRound < Cfg.FLOW_RETRY) return;
        start(t);
    }

    static void start(MapLocation t) {
        if (building) ends[5]++;
        else if (tgt != null && builtRound >= 0 && G.round - builtRound <= Cfg.FLOW_TTL) ends[4]++;
        tgt = t;
        tIdx = G.idx(t);
        block = !Nav.atKing(t) ? Sym.WALL | Sym.KING : Sym.WALL;
        tx = t.x; ty = t.y;
        builtRound = -1;
        building = true;
        gen++;
        if (gen >= (Integer.MAX_VALUE / SPAN) - 1) gen = 1;
        base = gen * SPAN;
        for (int b = 0; b < NB; b++) { head[b] = 0; tail[b] = 0; }
        pending = 0;
        builds++;
        startRound = G.round;
        if (g != null && bucket != null) seed();
    }

    static void seed() {
        // the target is the root; the heuristic aims at where the rat stands now
        sx0 = G.me.x; sy0 = G.me.y;
        g[tIdx] = base;
        curF = Math.max(Math.abs(tx - sx0), Math.abs(ty - sy0));
        int b = curF & BMASK;
        bucket[b][tail[b] & RMASK] = tIdx;
        tail[b]++;
        pending = 1;
    }

    /** Continue the search while more than reserve bytecodes are left (allocating the search arrays first). */
    public static void work(int reserve) {
        if (!building) return;
        if (bucket == null && !alloc(G.W * G.H, reserve)) return;
        workRev(reserve);
    }

    /** Bucket rows being allocated (bucket is set once all are there). */
    static int[][] bk;

    /** Allocate the search arrays a piece at a time, each once its own length (an allocation costs its length in
     *  bytecodes) fits in the bytecodes left past the reserve, and seed once all are there; true then. */
    static boolean alloc(int n, int reserve) {
        if (g == null) {
            if (left() < n + reserve + 50) return false;
            g = new int[n];
            par = null; bk = null; path = null;
        }
        if (par == null) { if (left() < n + reserve + 50) return false; par = new int[n]; }
        if (AB == null) {
            if (left() < 2 * AC + reserve + 1700) return false;
            int[] ab = AB = new int[2 * AC + 1];
            for (int i = 0; i <= 2 * AC; i++) ab[i] = i < AC ? AC - i : i - AC;
        }
        if (bk == null) bk = new int[NB][];
        for (int b = 0; b < NB; b++)
            if (bk[b] == null) { if (left() < RING + reserve + 50) return false; bk[b] = new int[RING]; }
        if (path == null) { if (left() < 1024 + reserve + 50) return false; path = buf = new int[1024]; }
        bucket = bk;
        seed();
        return true;
    }

    /** The reverse search: pops in f order (g from the target + Chebyshev to where the rat stood at the start) until a
     *  tile beside the rat's current tile comes up; the cost of a step is that of the tile entered (the popped one).
     *  The eight neighbours are unrolled off the map edge, the heuristic comes from AB and the queue state is kept
     *  in locals. */
    static void workRev(int reserve) {
        int[] gg = g, pp = par, mem = Sym.mem, ab = AB, hd = head, tl = tail, OX = DX, OY = DY;
        int[][] bk = bucket;
        int W = G.W, W1 = W - 1, H1 = G.H - 1, bs = base, dig = Cfg.DIG_COST, blk = block, cf = curF, pend = pending;
        int FU = OX[1], FV = OY[0], WV = FV * W, cx = AC - sx0, cy = AC - sy0;
        int mx0 = G.me.x - 1, mx1 = G.me.x + 1, my0 = G.me.y - 1, my1 = G.me.y + 1;
        while (pend > 0) {
            if (left() < reserve) { pending = pend; curF = cf; return; }
            int b = cf & BMASK, u;
            if (hd[b] == tl[b]) { cf++; continue; }
            u = bk[b][hd[b]++ & RMASK];
            pend--;
            int ux = u % W, uy = u / W, gu = gg[u] - bs, bx = ux + cx, by = uy + cy, a = ab[bx], c = ab[by];
            if (gu + (a > c ? a : c) != cf) continue; // stale entry
            if (ux >= mx0 && ux <= mx1 && uy >= my0 && uy <= my1) {
                // the path walks parents from here to the target: ~40 bc per tile
                if (left() < reserve + 40 * (gu + 3)) {
                    hd[b]--;
                    pending = pend + 1; curF = cf;
                    return;
                }
                pending = pend; curF = cf;
                finishRev(u);
                return;
            }
            int ng = gu + ((mem[u] & Sym.DIRT) != 0 ? dig : 1), nb = bs + ng, v, gv, f, fb;
            if (ux > 0 && uy > 0 && ux < W1 && uy < H1) {
                // xr/yu: the heuristic one step along +x/+y
                int xr = ab[bx + FU], xl = ab[bx - FU], yu = ab[by + FV], yd = ab[by - FV];
                v = u + WV; // N
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 0; f = ng + (a > yu ? a : yu); fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v; pend++;
                }
                v = u + WV + FU; // NE
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 1; f = ng + (xr > yu ? xr : yu); fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v; pend++;
                }
                v = u + FU; // E
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 2; f = ng + (xr > c ? xr : c); fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v; pend++;
                }
                v = u - WV + FU; // SE
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 3; f = ng + (xr > yd ? xr : yd); fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v; pend++;
                }
                v = u - WV; // S
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 4; f = ng + (a > yd ? a : yd); fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v; pend++;
                }
                v = u - WV - FU; // SW
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 5; f = ng + (xl > yd ? xl : yd); fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v; pend++;
                }
                v = u - FU; // W
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 6; f = ng + (xl > c ? xl : c); fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v; pend++;
                }
                v = u + WV - FU; // NW
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 7; f = ng + (xl > yu ? xl : yu); fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v; pend++;
                }
            } else {
                for (int k = 0; k < 8; k++) {
                    int vx = ux + OX[k], vy = uy + OY[k];
                    if (vx < 0 || vy < 0 || vx > W1 || vy > H1) continue;
                    v = vx + vy * W;
                    if ((mem[v] & blk) != 0 || (gv = gg[v]) >= bs && gv <= nb) continue;
                    gg[v] = nb; pp[v] = k;
                    int h1 = ab[vx + cx], h2 = ab[vy + cy];
                    f = ng + (h1 > h2 ? h1 : h2); fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v; pend++;
                }
            }
        }
        // the rat is unreachable from the target over what we know
        pending = 0; curF = cf;
        building = false;
        fails++;
        failTgt = tgt;
        failRound = G.round;
    }

    /** Path from the rat's tile through u and its parents to the target. */
    static void finishRev(int u) {
        int W = G.W, me = G.idx(G.me), v = u, len = 0, T = tIdx;
        int[] p = buf;
        if (u != me) p[len++] = me;
        while (v != T && len < p.length - 1) {
            p[len++] = v;
            int k = par[v];
            v = v - DX[k] - DY[k] * W;
        }
        p[len++] = T;
        path = p;
        plen = len;
        ppos = 0;
        building = false;
        builtRound = G.round;
        done++;
        lastTurns = G.round - startRound;
    }

    static void drop() { builtRound = -1; tgt = null; }

    /**
     * One step along the path toward t: move to the next tile, or dig it when it is dirt.
     * Returns false (caller falls back to greedy) when no path is ready or we have left it.
     */
    public static boolean follow(MapLocation t) throws GameActionException {
        if (!ready(t)) return false;
        int me = G.idx(G.me);
        int nxi = ppos + 1;
        if (path[ppos] != me) {
            // we drifted (moved while the search ran): rejoin at the farthest path tile next to us
            int hi = Math.min(plen, ppos + Cfg.FLOW_REJOIN), found = -1;
            for (int i = ppos; i < hi; i++) {
                int p = path[i];
                if (Math.abs(p % G.W - G.me.x) <= 1 && Math.abs(p / G.W - G.me.y) <= 1) found = i;
            }
            if (found < 0) { ends[3]++; drop(); request(t); return false; }
            if (path[found] == me) { ppos = found; nxi = found + 1; }
            else { ppos = Math.max(0, found - 1); nxi = found; }   // found 0: beside the start tile (ppos -1 threw every turn)
        }
        if (nxi >= plen) { ends[0]++; drop(); return false; }
        int nx = path[nxi];
        MapLocation n = new MapLocation(nx % G.W, nx / G.W);
        if ((Sym.mem[nx] & block) != 0) { ends[2]++; drop(); return false; } // newly seen wall (or a king moved) on the path
        RobotController rc = G.rc;
        Direction d = G.me.directionTo(n);
        if ((Sym.mem[nx] & Sym.DIRT) != 0) {
            if (G.globalCheese + G.rawCheese < 5 || !rc.isActionReady()) return true; // wait for the dig
            if (rc.getDirection() != d && rc.canTurn(d)) { rc.turn(d); G.facing = d; }
            if (rc.canRemoveDirt(n)) Nav.dig(n);
            return true;
        }
        if ((Nav.avoid & (1 << d.ordinal())) != 0 || !rc.canMove(d)) {
            // blocked by a robot: hold briefly, then let greedy route around it
            if (++held > Cfg.FLOW_HOLD) { held = 0; ends[1]++; drop(); return false; }
            return true;
        }
        held = 0;
        steps++;
        if (Nav.step(d)) ppos = nxi;
        return true;
    }
}
