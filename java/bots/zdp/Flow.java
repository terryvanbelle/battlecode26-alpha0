package zdp;

import battlecode.common.*;

/**
 * Dig-aware path planning for stuck rats: A* from here to a target over remembered terrain
 * (unknown = open, dirt = DIG_COST, wall = blocked), built with spare bytecode over one or more
 * turns, then followed step by step (digging dirt on the path). Requested only after greedy/bug
 * navigation stalls, so open maps pay nothing. g[] entries are gen*SPAN + cost, so a new search
 * needs no reset pass; Dial buckets on f = g + chebyshev (f grows by at most DIG_COST + 1 per step).
 */
public final class Flow {
    static final int SPAN = 1 << 16, RING = 1024, RMASK = RING - 1, NB = 8, BMASK = NB - 1;
    static final int[] DX = {0, 1, 1, 1, 0, -1, -1, -1}, DY = {1, 1, 0, -1, -1, -1, 0, 1};
    /** Neighbour order of the search: DX/DY in our side frame (Cfg.SYM_FLOW), so a mirrored rat's search expands
     *  mirrored neighbours in the same order and breaks ties between equal paths the mirrored way. */
    static int[] ox = DX, oy = DY;

    /** Side frame (signs of the home king's offset from the centre), set once at birth. */
    static void frame(int fu, int fv) { if (Cfg.SYM_FLOW) setFrame(fu, fv); }

    static void setFrame(int fu, int fv) {
        ox = new int[8];
        oy = new int[8];
        for (int k = 0; k < 8; k++) { ox[k] = DX[k] * fu; oy[k] = DY[k] * fv; }
    }
    static int[] g, par;
    static int gen;
    static MapLocation tgt;
    static int tIdx, tx, ty, base, sIdx;
    static boolean building;
    /** Tile bits a path may not cross (the target tile excepted): walls, and our kings' bodies (Cfg.KING_BLOCK, KING_FLOW) unless
     *  the target is inside one. */
    static int block = Sym.WALL;
    /** Round the current path was completed, or -1. */
    static int builtRound = -1;
    static int[] path, buf;
    static int plen, ppos;
    /** Completed paths kept for reuse (Cfg.FLOW_CACHE): a request toward a cached target from beside its path (or toward
     *  its start, walked backwards) adopts the rest of it at once instead of searching. cround is the build round. */
    static final int CK = 4;
    static int[][] cpath = new int[CK][];
    static int[] ctgt = new int[CK], cround = new int[CK];
    static int cnext;
    public static int hits;
    /** Cfg.FLOW_CACHE (a field so tests can set it). */
    static boolean cache = Cfg.FLOW_CACHE;
    /** Cfg.FLOW_LIFO (a field so tests can set it): each bucket is a stack, so ties on f go to the newest (deepest)
     *  tile first. */
    static boolean lifo = Cfg.FLOW_LIFO;
    /** Cfg.FLOW_REV (a field so tests can set it): the search grows from the target back toward the rat, so every tile
     *  it reaches knows its way to the target; the rat keeps walking meanwhile and the path starts where it ends up. */
    static boolean rev = Cfg.FLOW_REV;
    /** The rat's tile when a reverse search began: the goal its heuristic aims at. */
    static int sx0, sy0;
    /** Cfg.FLOW_FAST (a field so tests can set it), and its distance table: AB[d + AC] = |d| for |d| <= AC (maps are
     *  at most 60 wide), built once with the search arrays. */
    static boolean fast = Cfg.FLOW_FAST;
    static final int AC = 64;
    static int[] AB;
    static int[][] bucket;
    static int[] head = new int[NB], tail = new int[NB];
    static int curF, pending, held;
    public static int builds, steps, done, lastTurns, fails;
    /** Debug: nodes popped and bytecodes spent in work() (Debug.PROBE). */
    public static int pops, workBc;
    /** Target and tile count of the last completed path (kept after the path is dropped). */
    static MapLocation lastTgt;
    static int lastLen;
    /** Debug counts of how paths and searches ended: 0 reached the end, 1 held by a robot, 2 wall on the path,
     *  3 drifted off it, 4 replaced while ready (new target), 5 search replaced before it finished. */
    public static final int[] ends = new int[6];
    static int startRound, failRound;
    static MapLocation failTgt;
    /** Cfg.FLOW_SPLICE: a detour search is running toward stail[0], a tile of the cut path (tgt stays the errand's
     *  target); the rest of that path (slen tiles from stail[0]) is joined on when it completes. sFail* hold the failure
     *  record from before the detour, put back if the detour fails. */
    static boolean splicing;
    static int[] stail;
    static int slen, sFailRound;
    static MapLocation sFailTgt;
    /** Tests run outside the engine, where Clock reports 0 bytecodes left. */
    public static boolean unlimited;

    static int left() { return Clock.getBytecodesLeft() + G.debt; }

    /** Cfg.FLOW_QUOTA: search loop steps left in round quotaRound (quotaMax a round; tests set it). */
    static int quota, quotaRound = -1, quotaMax = Cfg.FLOW_QUOTA;

    /** Spends one of this round's Cfg.FLOW_QUOTA search steps; false once they are gone. The bytecodes left differ
     *  by a few between mirrored rats (min-scans over the engine's sense order update their best a different number
     *  of times), so a search cut by bytecodes alone ends a turn earlier on one side. */
    static boolean step() {
        if (quotaRound != G.round) { quotaRound = G.round; quota = quotaMax; }
        return quota-- > 0;
    }

    /** True when a complete, fresh path toward t exists. */
    public static boolean ready(MapLocation t) {
        return builtRound >= 0 && tgt != null && G.cheb(tgt, t) <= 1 && G.round - builtRound <= Cfg.FLOW_TTL;
    }

    /** True while a search toward t is running. */
    public static boolean searching(MapLocation t) {
        if (Cfg.FLOW_SPLICE > 0 && splicing && building && G.round - startRound > Cfg.FLOW_SPLICE_TURNS) { building = false; spliceFail(); }
        if (building && G.round - startRound > (rev ? Cfg.FLOW_REV_TURNS : Cfg.FLOW_MAX_TURNS)) {
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
        if (cache && adopt(t)) return;
        if (failTgt != null && G.cheb(failTgt, t) <= 1 && G.round - failRound < Cfg.FLOW_RETRY) return;
        start(t, false);
    }

    /** The current search is a frontier search (Cfg.BREAKOUT_TTL): forward from here with no heuristic, ending at the
     *  first never-seen tile it pops, which becomes its target. */
    static boolean frontier;
    /** Round the last frontier search found nothing reachable that is unseen. */
    static int frontierFail = -1000;

    /** Ask for a path to the never-seen tile nearest by path cost over what we know (open 1, dirt DIG_COST, walls
     *  blocked); no-op while one is being built or one is ready toward a still-unseen tile. */
    public static void requestFrontier() {
        if (frontier && (building || builtRound >= 0 && !Sym.known(tIdx) && G.round - builtRound <= Cfg.FLOW_TTL)) return;
        if (G.round - frontierFail < Cfg.FLOW_RETRY) return;
        start(G.me, true);
    }

    /** A frontier search is running, or its path is ready toward a still-unseen tile. */
    public static boolean frontierLive() {
        if (frontier && building && G.round - startRound > Cfg.FLOW_REV_TURNS) { building = false; fails++; frontierFail = G.round; }
        return frontier && (building || builtRound >= 0 && !Sym.known(tIdx) && G.round - builtRound <= Cfg.FLOW_TTL);
    }

    static void start(MapLocation t, boolean front) {
        if (Cfg.FLOW_SPLICE > 0) splicing = false;
        if (building) ends[5]++;
        else if (tgt != null && builtRound >= 0 && G.round - builtRound <= Cfg.FLOW_TTL) ends[4]++;
        frontier = front;
        tgt = t;
        tIdx = G.idx(t);
        block = (Cfg.KING_BLOCK || Cfg.KING_FLOW) && !Nav.atKing(t) ? Sym.WALL | Sym.KING : Sym.WALL;
        if (Cfg.FLOW_HELD > 0) block = heldBlock(block);
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
        if (rev && !frontier) {
            // the target is the root; the heuristic aims at where the rat stands now
            sx0 = G.me.x; sy0 = G.me.y;
            g[tIdx] = base;
            curF = Math.max(Math.abs(tx - sx0), Math.abs(ty - sy0));
            int b = curF & BMASK;
            bucket[b][tail[b] & RMASK] = tIdx;
            tail[b]++;
            pending = 1;
            return;
        }
        int s = sIdx = G.idx(G.me);
        g[s] = base;
        curF = frontier ? 0 : Math.max(Math.abs(G.me.x - tx), Math.abs(G.me.y - ty));
        int b = curF & BMASK;
        bucket[b][tail[b] & RMASK] = s;
        tail[b]++;
        pending = 1;
    }

    /** Continue the search while more than reserve bytecodes are left. */
    public static void work(int reserve) {
        if (!building) return;
        if (Cfg.FLOW_SPLICE > 0 && splicing) { workSplice(reserve); return; }
        if (Debug.PROBE) {
            int b0 = Clock.getBytecodeNum();
            work0(reserve);
            workBc += Clock.getBytecodeNum() - b0;
            return;
        }
        work0(reserve);
    }

    /** Cfg.FLOW_ALLOC: bucket rows being allocated (bucket is set once all are there). */
    static int[][] bk;

    /** Cfg.FLOW_ALLOC: allocate the search arrays a piece at a time, each once its own length (an allocation costs its
     *  length in bytecodes) fits in the bytecodes left past the reserve, and seed once all are there; true then. */
    static boolean alloc(int n, int reserve) {
        if (g == null) {
            if (left() < n + reserve + 50) return false;
            g = new int[n];
            par = null; bk = null; path = null;
        }
        if (par == null) { if (left() < n + reserve + 50) return false; par = new int[n]; }
        if (fast && AB == null) {
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

    static void work0(int reserve) {
        if (rev && !frontier && g != null && bucket != null) { workRev(reserve); return; }
        int n = G.W * G.H;
        if (Cfg.FLOW_ALLOC) {
            if (bucket == null && !alloc(n, reserve)) return;
        } else {
            if (g == null) {
                if (left() < 2 * n + reserve + (fast ? 1700 : 0)) return;
                g = new int[n];
                par = new int[n];
                if (fast) {
                    int[] ab = AB = new int[2 * AC + 1];
                    for (int i = 0; i <= 2 * AC; i++) ab[i] = i < AC ? AC - i : i - AC;
                }
                if (bucket != null) seed();
            }
            if (bucket == null) {
                if (left() < NB * RING + 1024 + reserve) return;
                bucket = new int[NB][RING];
                path = buf = new int[1024];
                seed();
            }
        }
        if (frontier) { workFrontier(reserve); return; }
        if (rev) { workRev(reserve); return; }
        int[] gg = g, pp = par, mem = Sym.mem, OX = ox, OY = oy;
        int W = G.W, H = G.H, bs = base, dig = Cfg.DIG_COST, X = tx, Y = ty, blk = block;
        while (pending > 0) {
            if (left() < reserve || Cfg.FLOW_QUOTA > 0 && !step()) return;
            int b = curF & BMASK, u;
            if (head[b] == tail[b]) { curF++; continue; }
            if (lifo) u = bucket[b][--tail[b] & RMASK];
            else { u = bucket[b][head[b] & RMASK]; head[b]++; }
            pending--;
            if (Debug.PROBE) pops++;
            int ux = u % W, uy = u / W, gu = gg[u] - bs;
            if (gu + Math.max(Math.abs(ux - X), Math.abs(uy - Y)) != curF) continue; // stale entry
            if (u == tIdx) {
                // backtrack + reverse costs ~40 bc per tile and the path has at most gu + 1 tiles: wait a turn if short
                if (left() < reserve + (cache ? 50 : 40) * (gu + 2)) { if (lifo) tail[b]++; else head[b]--; pending++; return; }
                finish();
                return;
            }
            for (int k = 0; k < 8; k++) {
                int vx = ux + OX[k], vy = uy + OY[k];
                if (vx < 0 || vy < 0 || vx >= W || vy >= H) continue;
                int v = vx + vy * W, m = mem[v];
                if ((m & blk) != 0 && v != tIdx) continue;
                int ng = gu + ((m & Sym.DIRT) != 0 ? dig : 1);
                int gv = gg[v];
                if (gv >= bs && gv - bs <= ng) continue;
                gg[v] = bs + ng;
                pp[v] = k;
                int f = ng + Math.max(Math.abs(vx - X), Math.abs(vy - Y));
                int fb = f & BMASK;
                bucket[fb][tail[fb] & RMASK] = v;
                tail[fb]++;
                pending++;
            }
        }
        // target unreachable over what we know
        building = false;
        fails++;
        failTgt = tgt;
        failRound = G.round;
    }

    /** The frontier search: Dijkstra from the rat (Dial buckets on g) until it pops a never-seen tile, which becomes the
     *  target; nothing unseen reachable sets frontierFail. Neighbours expand in the side frame order (ox/oy). */
    static void workFrontier(int reserve) {
        int[] gg = g, pp = par, mem = Sym.mem, OX = ox, OY = oy;
        int W = G.W, H = G.H, bs = base, dig = Cfg.DIG_COST, blk = block, s = sIdx;
        while (pending > 0) {
            if (left() < reserve || Cfg.FLOW_QUOTA > 0 && !step()) return;
            int b = curF & BMASK, u;
            if (head[b] == tail[b]) { curF++; continue; }
            u = bucket[b][head[b] & RMASK]; head[b]++;
            pending--;
            int gu = gg[u] - bs;
            if (gu != curF) continue; // stale entry
            if ((mem[u] & Sym.SEEN) == 0 && u != s) {
                if (left() < reserve + (cache ? 50 : 40) * (gu + 2)) { head[b]--; pending++; return; }
                tIdx = u; tx = u % W; ty = u / W;
                tgt = new MapLocation(tx, ty);
                finish();
                return;
            }
            int ux = u % W, uy = u / W;
            for (int k = 0; k < 8; k++) {
                int vx = ux + OX[k], vy = uy + OY[k];
                if (vx < 0 || vy < 0 || vx >= W || vy >= H) continue;
                int v = vx + vy * W, m = mem[v];
                if ((m & blk) != 0) continue;
                int ng = gu + ((m & Sym.DIRT) != 0 ? dig : 1);
                int gv = gg[v];
                if (gv >= bs && gv - bs <= ng) continue;
                gg[v] = bs + ng;
                pp[v] = k;
                int fb = ng & BMASK;
                bucket[fb][tail[fb] & RMASK] = v;
                tail[fb]++;
                pending++;
            }
        }
        building = false;
        fails++;
        frontierFail = G.round;
    }

    /** The reverse search: pops in f order (g from the target + Chebyshev to where the rat stood at the start) until a
     *  tile beside the rat's current tile comes up; the cost of a step is that of the tile entered (the popped one). */
    static void workRev(int reserve) {
        if (Cfg.FLOW_W > 1 && fast) { workRevW(reserve); return; }
        if (fast) { workRevFast(reserve); return; }
        int[] gg = g, pp = par, mem = Sym.mem, OX = ox, OY = oy;
        int W = G.W, H = G.H, bs = base, dig = Cfg.DIG_COST, X = sx0, Y = sy0, blk = block, mx = G.me.x, my = G.me.y;
        while (pending > 0) {
            if (left() < reserve || Cfg.FLOW_QUOTA > 0 && !step()) return;
            int b = curF & BMASK, u;
            if (head[b] == tail[b]) { curF++; continue; }
            if (lifo) u = bucket[b][--tail[b] & RMASK];
            else { u = bucket[b][head[b] & RMASK]; head[b]++; }
            pending--;
            if (Debug.PROBE) pops++;
            int ux = u % W, uy = u / W, gu = gg[u] - bs;
            if (gu + Math.max(Math.abs(ux - X), Math.abs(uy - Y)) != curF) continue; // stale entry
            if (Math.abs(ux - mx) <= 1 && Math.abs(uy - my) <= 1) {
                // the path walks parents from here to the target: ~40 bc per tile
                if (left() < reserve + (cache ? 50 : 40) * (gu + 3)) {
                    if (lifo) tail[b]++; else head[b]--;
                    pending++;
                    return;
                }
                finishRev(u);
                return;
            }
            int ng = gu + ((mem[u] & Sym.DIRT) != 0 ? dig : 1);
            for (int k = 0; k < 8; k++) {
                int vx = ux + OX[k], vy = uy + OY[k];
                if (vx < 0 || vy < 0 || vx >= W || vy >= H) continue;
                int v = vx + vy * W;
                if ((mem[v] & blk) != 0) continue;
                int gv = gg[v];
                if (gv >= bs && gv - bs <= ng) continue;
                gg[v] = bs + ng;
                pp[v] = k;
                int f = ng + Math.max(Math.abs(vx - X), Math.abs(vy - Y));
                int fb = f & BMASK;
                bucket[fb][tail[fb] & RMASK] = v;
                tail[fb]++;
                pending++;
            }
        }
        // the rat is unreachable from the target over what we know
        building = false;
        fails++;
        failTgt = tgt;
        failRound = G.round;
    }

    /** workRev with the eight neighbours unrolled off the map edge, the heuristic from AB and the queue state
     *  in locals (Cfg.FLOW_FAST): the same pops in the same order. The neighbour order is ox/oy (k's step is
     *  (DX[k]·fu, DY[k]·fv)), so a mirrored rat runs the mirrored search at the same cost. */
    static void workRevFast(int reserve) {
        int[] gg = g, pp = par, mem = Sym.mem, ab = AB, hd = head, tl = tail, OX = ox, OY = oy;
        int[][] bk = bucket;
        int W = G.W, W1 = W - 1, H1 = G.H - 1, bs = base, dig = Cfg.DIG_COST, blk = block, cf = curF, pend = pending;
        int FU = OX[1], FV = OY[0], WV = FV * W, cx = AC - sx0, cy = AC - sy0;
        int mx0 = G.me.x - 1, mx1 = G.me.x + 1, my0 = G.me.y - 1, my1 = G.me.y + 1;
        boolean st = lifo;
        while (pend > 0) {
            if (left() < reserve || Cfg.FLOW_QUOTA > 0 && !step()) { pending = pend; curF = cf; return; }
            int b = cf & BMASK, u;
            if (hd[b] == tl[b]) { cf++; continue; }
            if (st) u = bk[b][--tl[b] & RMASK];
            else u = bk[b][hd[b]++ & RMASK];
            pend--;
            if (Debug.PROBE) pops++;
            int ux = u % W, uy = u / W, gu = gg[u] - bs, bx = ux + cx, by = uy + cy, a = ab[bx], c = ab[by];
            if (gu + (a > c ? a : c) != cf) continue; // stale entry
            if (ux >= mx0 && ux <= mx1 && uy >= my0 && uy <= my1) {
                if (left() < reserve + (cache ? 50 : 40) * (gu + 3)) {
                    if (st) tl[b]++; else hd[b]--;
                    pending = pend + 1; curF = cf;
                    return;
                }
                pending = pend; curF = cf;
                finishRev(u);
                return;
            }
            int ng = gu + ((mem[u] & Sym.DIRT) != 0 ? dig : 1), nb = bs + ng, v, gv, f, fb;
            if (ux > 0 && uy > 0 && ux < W1 && uy < H1) {
                // xr/yu: the heuristic one step along +ox/+oy
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

    /** Cfg.FLOW_W (wt > 0 overrides it; tests) and its scaled distance table ABW[d + AC] = w·|d| (built for weight wAt);
     *  wGen is the search workRevW last seeded. */
    static int wt, wAt, wGen;
    static int[] ABW;

    /** workRevFast on f = g + w·h (Cfg.FLOW_W), with f clamped up to the bucket being popped: a step changes f by
     *  1 - w to DIG_COST + w, so the buckets still come up in order. A clamped f can't be recomputed from g, so each
     *  entry carries the g it was pushed with (tile | g << 12; maps hold at most 3600 tiles) and is stale once a
     *  cheaper g has been found for its tile. The first call of a search re-seeds the target at w·h. */
    static void workRevW(int reserve) {
        int w = wt > 0 ? wt : Cfg.FLOW_W;
        if (ABW == null || wAt != w) {
            if (left() < reserve + 1500) return;
            int[] s = ABW = new int[2 * AC + 1], ab0 = AB;
            for (int i = 0; i <= 2 * AC; i++) s[i] = w * ab0[i];
            wAt = w;
        }
        if (wGen != gen) {
            wGen = gen;
            for (int b = 0; b < NB; b++) { head[b] = 0; tail[b] = 0; }
            int h0 = ABW[tx - sx0 + AC], h1 = ABW[ty - sy0 + AC];
            curF = h0 > h1 ? h0 : h1;
            int b = curF & BMASK;
            bucket[b][0] = tIdx;
            tail[b] = 1;
            pending = 1;
        }
        int[] gg = g, pp = par, mem = Sym.mem, ab = ABW, hd = head, tl = tail, OX = ox, OY = oy;
        int[][] bk = bucket;
        int W = G.W, W1 = W - 1, H1 = G.H - 1, bs = base, dig = Cfg.DIG_COST, blk = block, cf = curF, pend = pending;
        int FU = OX[1], FV = OY[0], WV = FV * W, cx = AC - sx0, cy = AC - sy0;
        int mx0 = G.me.x - 1, mx1 = G.me.x + 1, my0 = G.me.y - 1, my1 = G.me.y + 1;
        boolean st = lifo;
        while (pend > 0) {
            if (left() < reserve || Cfg.FLOW_QUOTA > 0 && !step()) { pending = pend; curF = cf; return; }
            int b = cf & BMASK, e;
            if (hd[b] == tl[b]) { cf++; continue; }
            if (st) e = bk[b][--tl[b] & RMASK];
            else e = bk[b][hd[b]++ & RMASK];
            pend--;
            if (Debug.PROBE) pops++;
            int u = e & 0xFFF, gu = gg[u] - bs;
            if ((e >>> 12) != gu) continue; // stale entry
            int ux = u % W, uy = u / W;
            if (ux >= mx0 && ux <= mx1 && uy >= my0 && uy <= my1) {
                if (left() < reserve + (cache ? 50 : 40) * (gu + 3)) {
                    if (st) tl[b]++; else hd[b]--;
                    pending = pend + 1; curF = cf;
                    return;
                }
                pending = pend; curF = cf;
                finishRev(u);
                return;
            }
            int bx = ux + cx, by = uy + cy, a = ab[bx], c = ab[by];
            int ng = gu + ((mem[u] & Sym.DIRT) != 0 ? dig : 1), nb = bs + ng, ge = ng << 12, v, gv, f, fb;
            if (ux > 0 && uy > 0 && ux < W1 && uy < H1) {
                int xr = ab[bx + FU], xl = ab[bx - FU], yu = ab[by + FV], yd = ab[by - FV];
                v = u + WV; // N
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 0; f = ng + (a > yu ? a : yu); if (f < cf) f = cf; fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v | ge; pend++;
                }
                v = u + WV + FU; // NE
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 1; f = ng + (xr > yu ? xr : yu); if (f < cf) f = cf; fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v | ge; pend++;
                }
                v = u + FU; // E
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 2; f = ng + (xr > c ? xr : c); if (f < cf) f = cf; fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v | ge; pend++;
                }
                v = u - WV + FU; // SE
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 3; f = ng + (xr > yd ? xr : yd); if (f < cf) f = cf; fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v | ge; pend++;
                }
                v = u - WV; // S
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 4; f = ng + (a > yd ? a : yd); if (f < cf) f = cf; fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v | ge; pend++;
                }
                v = u - WV - FU; // SW
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 5; f = ng + (xl > yd ? xl : yd); if (f < cf) f = cf; fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v | ge; pend++;
                }
                v = u - FU; // W
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 6; f = ng + (xl > c ? xl : c); if (f < cf) f = cf; fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v | ge; pend++;
                }
                v = u + WV - FU; // NW
                if ((mem[v] & blk) == 0 && ((gv = gg[v]) < bs || gv > nb)) {
                    gg[v] = nb; pp[v] = 7; f = ng + (xl > yu ? xl : yu); if (f < cf) f = cf; fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v | ge; pend++;
                }
            } else {
                for (int k = 0; k < 8; k++) {
                    int vx = ux + OX[k], vy = uy + OY[k];
                    if (vx < 0 || vy < 0 || vx > W1 || vy > H1) continue;
                    v = vx + vy * W;
                    if ((mem[v] & blk) != 0 || (gv = gg[v]) >= bs && gv <= nb) continue;
                    gg[v] = nb; pp[v] = k;
                    int h1 = ab[vx + cx], h2 = ab[vy + cy];
                    f = ng + (h1 > h2 ? h1 : h2); if (f < cf) f = cf; fb = f & BMASK;
                    bk[fb][tl[fb]++ & RMASK] = v | ge; pend++;
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

    /** Path from the rat's tile through u and its parents to the target (reverse search). */
    static void finishRev(int u) {
        int W = G.W, me = G.idx(G.me), v = u, len = 0, T = tIdx;
        int[] p = buf;
        if (u != me) p[len++] = me;
        while (v != T && len < p.length - 1) {
            p[len++] = v;
            int k = par[v];
            v = v - ox[k] - oy[k] * W;
        }
        p[len++] = T;
        if (cache) {
            int[] c = new int[len];
            for (int i = 0; i < len; i++) c[i] = p[i];
            p = c;
            store(c);
        }
        path = p;
        plen = len;
        ppos = 0;
        lastTgt = tgt;
        lastLen = len;
        building = false;
        builtRound = G.round;
        done++;
        lastTurns = G.round - startRound;
        if (Cfg.FLOW_SPLICE > 0 && splicing) join();
    }

    /** Cfg.FLOW_SPLICE: the next path tile (index nxi) is a newly seen wall. Start a detour search toward the path tile
     *  k - 1 past the first clear one after it (k = Cfg.FLOW_SPLICE), keeping the rest of the path; false (search afresh) when that
     *  tile would be the path's end, with FLOW_CACHE, or short of bytecodes. */
    static boolean splice(int nxi, int k) {
        if (cache || tgt == null) return false;
        int[] p = path, mem = Sym.mem;
        int blk = block, j = nxi + 1, n = plen;
        while (j < n && (mem[p[j]] & blk) != 0) j++;
        j += k - 1;
        if (j >= n - 1 || left() < 4 * (n - j) + 400) return false;
        int[] s = new int[n - j];
        for (int i = j; i < n; i++) s[i - j] = p[i];
        MapLocation T = tgt;
        MapLocation ft = failTgt;
        int fr = failRound;
        ends[2]++;
        builtRound = -1;
        start(new MapLocation(s[0] % G.W, s[0] / G.W), false);
        tgt = T;
        stail = s; slen = n - j; splicing = true;
        sFailTgt = ft; sFailRound = fr;
        return true;
    }

    /** Cfg.FLOW_SPLICE: the detour (now path[0..plen)) ends at stail[0]; join the rest of the old path on and aim at
     *  the errand's target again. */
    static void join() {
        splicing = false;
        if (tgt == null) return;
        int[] p = path, s = stail;
        int len = plen, n = slen;
        if (len + n - 1 > p.length) n = p.length - len + 1;
        for (int i = 1; i < n; i++) p[len++] = s[i];
        plen = len;
        lastLen = len;
        tIdx = G.idx(tgt); tx = tgt.x; ty = tgt.y;
        stail = null;
    }

    /** Cfg.FLOW_SPLICE: run the detour search; when it finds no way, search for the whole path instead. */
    static void workSplice(int reserve) {
        work0(reserve);
        if (splicing && !building && builtRound < 0) spliceFail();
    }

    /** Cfg.FLOW_SPLICE: the detour failed (no way over what we know, or too slow): forget it (put the failure record
     *  back) and start the full search toward the errand's target. */
    static void spliceFail() {
        splicing = false;
        stail = null;
        failTgt = sFailTgt; failRound = sFailRound;
        if (tgt != null) start(tgt, false);
    }

    /** Keep a completed path in the cache (the slot holding the same target, else the oldest). */
    static void store(int[] c) {
        int k = 0;
        while (k < CK && (cpath[k] == null || ctgt[k] != tIdx)) k++;
        if (k == CK) { k = cnext; cnext = (cnext + 1) % CK; }
        cpath[k] = c; ctgt[k] = tIdx; cround[k] = G.round;
    }

    static void finish() {
        int W = G.W, s = sIdx, v = tIdx, len = 0;
        int[] p = buf;
        // walk parents back from the target, then reverse
        while (v != s && len < p.length - 1) {
            p[len++] = v;
            int k = par[v];
            v = v - ox[k] - oy[k] * W;
        }
        p[len++] = s;
        if (cache) {
            // an exact-size copy, kept: buf is reused by the next search
            int[] c = new int[len];
            for (int i = 0, j = len - 1; j >= 0; i++, j--) c[i] = p[j];
            p = c;
            store(c);
        } else for (int i = 0, j = len - 1; i < j; i++, j--) { int t = p[i]; p[i] = p[j]; p[j] = t; }
        path = p;
        plen = len;
        ppos = 0;
        lastTgt = tgt;
        lastLen = len;
        building = false;
        builtRound = G.round;
        done++;
        lastTurns = G.round - startRound;
    }

    static void drop() { builtRound = -1; tgt = null; }

    /** Cfg.FLOW_HELD: 1 + the path tile a robot held (its path dropped), 0 for none; kept out of searches until
     *  round heldUntil. */
    static int held1, heldUntil;

    /** Cfg.FLOW_HELD: keep tile i out of searches for FLOW_HELD rounds (one tile at a time; not beside the target). */
    static void markHeld(int i) {
        if (held1 > 0) Sym.mem[held1 - 1] &= ~Sym.HELD;
        held1 = 0;
        if (Math.max(Math.abs(i % G.W - tx), Math.abs(i / G.W - ty)) <= 1) return;
        Sym.mem[i] |= Sym.HELD;
        held1 = i + 1;
        heldUntil = G.round + Cfg.FLOW_HELD;
    }

    /** Cfg.FLOW_HELD: block mask b for a new search, plus the held tile while it lasts (an expired one is cleared). */
    static int heldBlock(int b) {
        if (held1 > 0 && G.round > heldUntil) { Sym.mem[held1 - 1] &= ~Sym.HELD; held1 = 0; }
        return held1 > 0 ? b | Sym.HELD : b;
    }

    /** Take a cached path toward t (or one that started beside t, walked backwards) that passes beside us: the path
     *  becomes ready from its farthest tile beside us. False when none fits. */
    static boolean adopt(MapLocation t) {
        int W = G.W, mx = G.me.x, my = G.me.y, tx0 = t.x, ty0 = t.y;
        for (int k = 0; k < CK; k++) {
            int[] p = cpath[k];
            if (p == null || G.round - cround[k] > Cfg.FLOW_CACHE_TTL) continue;
            int n = p.length, c = ctgt[k], s = p[0], at = -1;
            if (Math.max(Math.abs(c % W - tx0), Math.abs(c / W - ty0)) <= 1) {
                for (int i = n - 1; i >= 0; i--) {
                    int q = p[i];
                    if (Math.abs(q % W - mx) <= 1 && Math.abs(q / W - my) <= 1) { at = i; break; }
                }
            } else if (Math.max(Math.abs(s % W - tx0), Math.abs(s / W - ty0)) <= 1) {
                for (int i = 0; i < n; i++) {
                    int q = p[i];
                    if (Math.abs(q % W - mx) <= 1 && Math.abs(q / W - my) <= 1) { at = i; break; }
                }
                if (at < 0) continue;
                int[] r = new int[n];
                for (int i = 0, j = n - 1; j >= 0; i++, j--) r[i] = p[j];
                p = r;
                at = n - 1 - at;
                c = s;
            }
            if (at < 0) continue;
            frontier = false;
            tgt = new MapLocation(c % W, c / W);
            tIdx = c;
            block = (Cfg.KING_BLOCK || Cfg.KING_FLOW) && !Nav.atKing(tgt) ? Sym.WALL | Sym.KING : Sym.WALL;
            if (building) { building = false; ends[5]++; }
            path = p;
            plen = n;
            ppos = p[at] == G.idx(G.me) ? at : Math.max(0, at - 1);
            held = 0;
            builtRound = G.round;
            lastTgt = tgt;
            lastLen = n - at;
            hits++;
            return true;
        }
        return false;
    }

    /** Forget the cached copy of the current path (a wall was found on it). */
    static void uncache() {
        for (int k = 0; k < CK; k++) if (cpath[k] == path) cpath[k] = null;
    }

    /** The way to t is known to be long: the last path toward t was longer than 2 * Chebyshev distance + 2 tiles, or
     *  the last failed search was toward t. */
    static boolean detour(MapLocation t) {
        if (failTgt != null && G.cheb(failTgt, t) <= 1 && G.round - failRound < Cfg.FLOW_RETRY) return true;
        return lastTgt != null && G.cheb(lastTgt, t) <= 1 && lastLen - 1 > 2 * G.cheb(G.me, t) + 2;
    }

    /** A free step that lands beside one of the next few path tiles after the blocked one (the farthest wins; ties go
     *  by G.hand so mirrored rats pick mirrored sides), or null. follow() rejoins the path from there next turn. */
    static Direction sidestep(int nxi) {
        RobotController rc = G.rc;
        int hi = Math.min(plen - 1, nxi + 3), bs = -1;
        Direction best = null;
        for (Direction d : G.DIRS) {
            if ((Nav.avoid & (1 << d.ordinal())) != 0 || !rc.canMove(d)) continue;
            MapLocation l = G.me.add(d);
            int s = -1;
            for (int j = hi; j > nxi; j--) {
                int p = path[j], dx = p % G.W - l.x, dy = p / G.W - l.y;
                if (Math.abs(dx) <= 1 && Math.abs(dy) <= 1) { s = j * 4 - dx * dx - dy * dy; break; }
            }
            if (s < 0) continue;
            s = s * 2 + (G.hand(G.me, d) > 0 ? 1 : 0);
            if (s > bs) { bs = s; best = d; }
        }
        return best;
    }

    /**
     * One step along the path toward t: move to the next tile, or dig it when it is dirt.
     * Returns false (caller falls back to greedy) when no path is ready or we have left it.
     */
    public static boolean follow(MapLocation t) throws GameActionException {
        if (!ready(t)) return false;
        int me = G.idx(G.me);
        int nxi = ppos + 1;
        if (path[ppos] != me) {
            // we drifted (moved while the search ran, or sidestepped): rejoin at the farthest path tile next to us
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
        if (Cfg.FLOW_SPLICE > 0 && (Sym.mem[nx] & block) != 0 && splice(nxi, Cfg.FLOW_SPLICE)) return false;
        if ((Sym.mem[nx] & block) != 0) { ends[2]++; if (cache) uncache(); drop(); return false; } // newly seen wall (or a king moved) on the path
        RobotController rc = G.rc;
        Direction d = G.me.directionTo(n);
        if (Cfg.MOAT && (Sym.mem[nx] & Sym.DIRT) != 0 && Moat.keep(n)) { drop(); return false; }
        if ((Sym.mem[nx] & Sym.DIRT) != 0) {
            if (G.globalCheese + G.rawCheese < 5 || !rc.isActionReady()) return true; // wait for the dig
            if (rc.getDirection() != d && rc.canTurn(d)) { rc.turn(d); G.facing = d; }
            if (rc.canRemoveDirt(n)) {
                rc.removeDirt(n);
                Nav.digs++;
                Sym.mem[nx] &= ~Sym.DIRT;
            }
            return true;
        }
        if ((Nav.avoid & (1 << d.ordinal())) != 0 || !rc.canMove(d)) {
            // an ally baby on the path is carried past (Cfg.SWAP_HOLD)
            if (Nav.swapHold > 0 && held + 1 >= Nav.swapHold && Nav.swapPast(d)) { held = 0; steps++; ppos = nxi; return true; }
            // blocked by a robot: hold briefly, then step around it on the path (FLOW_JAM) or let greedy route around it
            if (++held > Cfg.FLOW_HOLD) {
                if (Cfg.FLOW_JAM && held <= Cfg.FLOW_JAM_HOLD) {
                    Direction s = sidestep(nxi);
                    if (s != null) Nav.step(s);
                    return true;
                }
                held = 0; ends[1]++; if (Cfg.FLOW_HELD > 0) markHeld(nx); drop(); return false;
            }
            return true;
        }
        held = 0;
        steps++;
        if (Nav.step(d)) ppos = nxi;
        return true;
    }
}
