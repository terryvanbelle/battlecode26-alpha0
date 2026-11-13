package zdp;

import battlecode.common.*;

/**
 * Cfg.BFLOW: navigation down a distance field grown from the target one wave at a time over row bitboards (one long
 * per map row, bit x = column x; maps are at most 60 wide). A wave dilates the last wave's tiles one step in all eight
 * directions and keeps what is neither a known wall nor already reached; a dirt tile settles DIG_COST - 1 waves after
 * it is reached (its dig), so a tile's wave is its path cost to the target. Each wave's tiles are kept, so a rat finds
 * its own distance and its neighbours' wherever it stands: a robot in the way or a sidestep never loses the path.
 * Unknown tiles count as open. A wave costs ~60-110 bytecodes a row in its range and nothing per tile, where Flow's
 * A* pays ~250 a tile: on groveisbeingdesecrated (a 60x60 maze of bands) its searches never finished.
 */
public final class BitFlow {
    /** Known walls and dirt by row (Sym.record and the dig sites keep them). */
    static long[] wallR, dirtR;
    /** Tiles reached this search (dirt still settling included), and settled. */
    static long[] reach, done;
    /** The last wave's tiles and the next's, by row with one padding row each side (index y + 1). */
    static long[] fr, nf;
    /** Settled tiles of wave d, rows layerLo[d] .. layerLo[d] + layer[d].length - 1. */
    static long[][] layer;
    static int[] layerLo;
    /** Dirt reached and settling at wave d, in slot d % RING over rows pLo .. pHi; pendN rows pending in all. */
    static long[][] pend;
    static int[] pLo, pHi;
    static int pendN;
    static final int MAXD = 1000;
    static final long[] EMPTY = new long[0];
    static int RING, DELAY;
    static long FULL;
    /** The last wave grown and its row range (lo > hi: empty). */
    static int wave, lo, hi;
    static MapLocation tgt, failTgt;
    static int startRound = -10000, failRound = -10000;
    /** A search toward tgt is under way (waves left to grow); seeded once its arrays are set for tgt. */
    static boolean building, seeded;
    /** The rat's tile when its wave was last found, that wave, and the wave at which the field first covered it. */
    static int meIdx = -1, meD = -1, meAt = -1;
    public static int held, builds, steps, fails, restarts;

    /** Row masks for the map (Sym.init). */
    static void init() {
        wallR = new long[G.H];
        dirtR = new long[G.H];
        FULL = G.W >= 64 ? -1L : (1L << G.W) - 1;
        DELAY = Math.max(0, Cfg.DIG_COST - 1);
        RING = DELAY + 1;
    }

    /** Sym.record saw tile (x, y) with memory bits v. */
    static void mark(int x, int y, int v) {
        long b = 1L << x;
        if ((v & Sym.WALL) != 0) wallR[y] |= b; else wallR[y] &= ~b;
        if ((v & Sym.DIRT) != 0) dirtR[y] |= b; else dirtR[y] &= ~b;
    }

    /** The dirt at (x, y) was dug. */
    static void undirt(int x, int y) { dirtR[y] &= ~(1L << x); }

    /** Tests run outside the engine (Flow.unlimited). */
    static int left() { return Clock.getBytecodesLeft() + G.debt; }

    /** A field toward t is being grown or is fresh. */
    public static boolean active(MapLocation t) {
        return tgt != null && G.cheb(tgt, t) <= 1 && G.round - startRound <= Cfg.FLOW_TTL;
    }

    public static void request(MapLocation t) {
        if (active(t)) return;
        if (failTgt != null && G.cheb(failTgt, t) <= 1 && G.round - failRound < Cfg.FLOW_RETRY) return;
        start(t);
    }

    static void start(MapLocation t) {
        tgt = t;
        startRound = G.round;
        building = true;
        seeded = false;
        meIdx = -1; meD = -1; meAt = -1;
        held = 0;
        builds++;
    }

    /** The search arrays, once their lengths fit in the bytecodes left past the reserve (an allocation costs its length). */
    static boolean alloc(int reserve) {
        if (left() < reserve + 2 * MAXD + (RING + 4) * (G.H + 2) + 500) return false;
        int h = G.H;
        reach = new long[h];
        done = new long[h];
        fr = new long[h + 2];
        nf = new long[h + 2];
        layer = new long[MAXD][];
        layerLo = new int[MAXD];
        pend = new long[RING][h];
        pLo = new int[RING];
        pHi = new int[RING];
        for (int s = 0; s < RING; s++) { pLo[s] = h; pHi[s] = -1; }
        return true;
    }

    /** Clear the last search and put the target in wave 0. */
    static void seed() {
        int h = G.H;
        long[] R = reach, D = done;
        for (int y = 0; y < h; y++) { R[y] = 0; D[y] = 0; }
        if (lo <= hi) for (int y = lo; y <= hi; y++) fr[y + 1] = 0;
        for (int s = 0; s < RING; s++) {
            long[] P = pend[s];
            for (int y = pLo[s]; y <= pHi[s]; y++) P[y] = 0;
            pLo[s] = h; pHi[s] = -1;
        }
        pendN = 0;
        long b = 1L << tgt.x;
        R[tgt.y] = b;
        D[tgt.y] = b;
        fr[tgt.y + 1] = b;
        layer[0] = new long[] { b };
        layerLo[0] = tgt.y;
        wave = 0;
        lo = hi = tgt.y;
        seeded = true;
    }

    /** Grow the field while more than reserve bytecodes are left: until the rat's tile is EXTRA waves inside it, or
     *  nothing is left to settle (the rat unreachable: a failure). */
    public static void work(int reserve) {
        if (!building) return;
        if (reach == null && !alloc(reserve)) return;
        if (!seeded) {
            if (left() < reserve + 3 * G.H + 300) return;
            seed();
        }
        int x = G.me.x, y = G.me.y;
        while (true) {
            if (meAt < 0 && (done[y] >>> x & 1) != 0) meAt = wave;
            if (meAt >= 0 && wave >= meAt + Cfg.BFLOW_EXTRA && (done[y] >>> x & 1) != 0) return;
            int s = (wave + 1) % RING, span = hi - lo + 3;
            if (pLo[s] <= pHi[s]) span += pHi[s] - pLo[s] + 1;
            if (left() < reserve + 120 * span + 400) return;
            if (!grow()) {
                building = false;
                if ((done[y] >>> x & 1) == 0) { fails++; failTgt = tgt; failRound = G.round; }
                return;
            }
        }
    }

    /** One wave: the tiles at distance wave + 1. False when nothing is left to settle. */
    static boolean grow() {
        int d = wave + 1, h = G.H;
        if (d >= MAXD) return false;
        int s = d % RING, s2 = (d + DELAY) % RING;
        int a = lo <= hi ? lo - 1 : h, b = lo <= hi ? hi + 1 : -1;
        if (pLo[s] <= pHi[s]) { if (pLo[s] < a) a = pLo[s]; if (pHi[s] > b) b = pHi[s]; }
        if (a < 0) a = 0;
        if (b >= h) b = h - 1;
        long[] F = fr, NF = nf, R = reach, Wl = wallR, Dr = dirtR, Dn = done, P = pend[s], P2 = pend[s2];
        long full = FULL;
        boolean delay = DELAY > 0;
        int nlo = h, nhi = -1;
        for (int y = a; y <= b; y++) {
            long f = F[y] | F[y + 1] | F[y + 2], n = 0;
            if (f != 0) {
                n = (f | f << 1 | f >>> 1) & full & ~(R[y] | Wl[y]);
                if (n != 0) {
                    R[y] |= n;
                    long dt = n & Dr[y];
                    if (dt != 0 && delay) {
                        n ^= dt;
                        P2[y] |= dt;
                        pendN++;
                        if (y < pLo[s2]) pLo[s2] = y;
                        if (y > pHi[s2]) pHi[s2] = y;
                    }
                }
            }
            long p = P[y];
            if (p != 0) { n |= p; P[y] = 0; pendN--; }
            if (n != 0) {
                NF[y + 1] = n;
                Dn[y] |= n;
                if (y < nlo) nlo = y;
                nhi = y;
            }
        }
        if (s != s2) { pLo[s] = h; pHi[s] = -1; }
        if (lo <= hi) for (int y = lo; y <= hi; y++) F[y + 1] = 0;
        fr = NF;
        nf = F;
        if (nlo <= nhi) {
            long[] L = new long[nhi - nlo + 1];
            for (int y = nlo; y <= nhi; y++) L[y - nlo] = NF[y + 1];
            layer[d] = L;
            layerLo[d] = nlo;
        } else {
            layer[d] = EMPTY;
        }
        wave = d;
        lo = nlo;
        hi = nhi;
        return nlo <= nhi || pendN > 0;
    }

    /** Tile (x, y) settled in wave d. */
    static boolean inWave(int d, int x, int y) {
        long[] L = layer[d];
        int r = y - layerLo[d];
        return r >= 0 && r < L.length && (L[r] >>> x & 1) != 0;
    }

    /** The wave of settled tile (x, y), searched outward from guess. */
    static int find(int x, int y, int guess) {
        int top = wave;
        if (guess > top) guess = top;
        if (guess < 0) guess = 0;
        for (int k = 0; k <= top; k++) {
            int d = guess - k;
            if (d >= 0 && inWave(d, x, y)) return d;
            d = guess + k;
            if (k > 0 && d <= top && inWave(d, x, y)) return d;
            if (guess - k < 0 && guess + k >= top) break;
        }
        return -1;
    }

    /** The wave of a neighbour of a tile at wave dm when it is dm or less (a step changes the wave by at most
     *  DIG_COST), else -1. */
    static int near(int x, int y, int dm) {
        int lo0 = dm - RING;
        if (lo0 < 0) lo0 = 0;
        for (int d = dm; d >= lo0; d--) if (inWave(d, x, y)) return d;
        return -1;
    }

    /** Turns between directions a and b (0-4). */
    static int turns(Direction a, Direction b) {
        int k = (a.ordinal() - b.ordinal() + 8) & 7;
        return k > 4 ? 8 - k : k;
    }

    /**
     * One step down the field toward t: the free neighbour of least wave below ours (ties: nearest the straight line
     * to t, then G.hand), digging it first when dirt. Robots in every downhill step: wait, after FLOW_HOLD turns step
     * to a free tile of our own wave, after 3 * FLOW_HOLD give up (greedy). Returns false when the rat is not on a
     * field toward t (the caller walks greedy meanwhile); nothing downhill (a wall seen since the waves passed) grows
     * a fresh field.
     */
    public static boolean follow(MapLocation t) throws GameActionException {
        if (!seeded || tgt == null || G.cheb(tgt, t) > 1) return false;
        if (G.round - startRound > Cfg.FLOW_TTL) { tgt = null; building = false; return false; }
        int x = G.me.x, y = G.me.y, W = G.W, H = G.H;
        if ((done[y] >>> x & 1) == 0) return false;
        int m = x + y * W;
        if (m != meIdx) { meD = find(x, y, meIdx >= 0 ? meD : wave); meIdx = m; }
        if (meD <= 0) return false;
        RobotController rc = G.rc;
        Direction toT = G.me.directionTo(t), best = null, eq = null;
        int bk = Integer.MAX_VALUE, ek = Integer.MAX_VALUE, bd = -1;
        boolean robot = false;
        int[] mem = Sym.mem;
        for (int k = 0; k < 8; k++) {
            Direction d = G.DIRS[k];
            int vx = x + d.dx, vy = y + d.dy;
            if (vx < 0 || vy < 0 || vx >= W || vy >= H || (done[vy] >>> vx & 1) == 0) continue;
            int mm = mem[vx + vy * W];
            if ((mm & Sym.WALL) != 0) continue;
            int dv = near(vx, vy, meD);
            if (dv < 0) continue;
            boolean dirt = (mm & Sym.DIRT) != 0;
            int tie = turns(d, toT) * 2 + (G.hand(G.me, d) > 0 ? 0 : 1);
            if (!dirt && ((Nav.avoid & (1 << d.ordinal())) != 0 || !rc.canMove(d))) {
                if (dv < meD) robot = true;
                continue;
            }
            if (dv == meD) {
                if (!dirt && tie < ek) { ek = tie; eq = d; }
                continue;
            }
            int key = dv * 16 + tie;
            if (key < bk) { bk = key; best = d; bd = dv; }
        }
        if (best == null) {
            if (!robot) { restarts++; start(tgt); return false; }
            held++;
            if (held > 3 * Cfg.FLOW_HOLD) { held = 0; return false; }
            if (held > Cfg.FLOW_HOLD && eq != null && Nav.step(eq)) { meIdx = G.idx(G.me); }
            return true;
        }
        held = 0;
        MapLocation n = G.me.add(best);
        int ni = n.x + n.y * W;
        if ((mem[ni] & Sym.DIRT) != 0) {
            if (G.globalCheese + G.rawCheese < 5 || !rc.isActionReady()) return true;
            if (rc.getDirection() != best && rc.canTurn(best)) { rc.turn(best); G.facing = best; }
            if (!(Cfg.MOAT && Moat.keep(n)) && rc.canRemoveDirt(n)) {
                rc.removeDirt(n);
                Nav.digs++;
                mem[ni] &= ~Sym.DIRT;
                undirt(n.x, n.y);
            }
            return true;
        }
        if (Nav.step(best)) {
            steps++;
            meIdx = ni;
            meD = bd;
        }
        return true;
    }
}
