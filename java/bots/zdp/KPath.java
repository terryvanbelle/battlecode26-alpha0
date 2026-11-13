package zdp;

import battlecode.common.*;

/**
 * Cfg.KING_PATH: the king's walk to a home mine over its 3x3 body. A centre is blocked when a wall the king has seen
 * lies within Chebyshev 1 of it or its 3x3 leaves the map (unseen tiles and dirt count as open). A reverse search
 * (Dial buckets on f = g + Chebyshev to the king, LIFO within a bucket) grows from every seat (a centre within
 * Chebyshev 1 of a home-half mine no other king holds) at once and stops when it reaches the king, so g is the
 * remaining steps to the mine nearest by path; the king steps to a neighbour of lower g. It searches again when a
 * new mine is listed or a newly seen wall blocks the way, and holds still while a search runs. King.homeMine's pick
 * was the first listed mine within dist2 400 of the start, walked to greedily: rift (4342) took (6,12), 13 away,
 * not (12,26), 8 away, and stopped at (6,19) under a wall row for 2000 rounds while spsquared's kings sat on mines
 * (its kings picked up 2680 cheese to ours 60); wallsofparadis (4344-4345) has no home mine within 400, so our king
 * stayed 26 from its nearest mine and our deliveries were 1490 to AdamTan12's 7281.
 */
public final class KPath {
    static final int SPAN = 1 << 12, POOL = 4096;
    static final int[] DX = {0, 1, 1, 1, 0, -1, -1, -1}, DY = {1, 1, 0, -1, -1, -1, 0, 1};
    /** Neighbour order in our side frame, so mirrored kings expand mirrored neighbours in the same order. */
    static int[] ox = DX, oy = DY;
    /** Centre blocked by a seen wall within Chebyshev 1. */
    static boolean[] kb;
    static int[] g, poolV, poolN, head;
    static int gen, base, pn, curF, pending, maxF;
    static boolean building, ready, seated;
    /** The king's centre the current search aims at, and its index. */
    static int kx, ky, kIdx;
    static int startRound = -1000, failRound = -1000, builtRound = -1000, minesAt = -1;
    /** Cfg.KP_FRONT: front(x, y) = fa * x + fb * y + fc is 2 * |home - away| times the signed distance from the
     *  bisector (positive on our side); centres below fMin are off the path. */
    static int fa, fb, fc, fMin = Integer.MIN_VALUE;
    static int lastScan = -1;
    public static int searches, fails, steps, digs;
    public static boolean unlimited;
    /** With unlimited (tests): the bytecodes left that left() reports. */
    public static int fakeLeft = Integer.MAX_VALUE;

    static int left() { return Clock.getBytecodesLeft() + G.debt; }

    /** Side frame from our start: the signs of its offset from the map centre. */
    static void frame(MapLocation home) {
        int fu = 2 * home.x < G.W - 1 ? 1 : -1, fv = 2 * home.y < G.H - 1 ? 1 : -1;
        ox = new int[8];
        oy = new int[8];
        for (int k = 0; k < 8; k++) { ox[k] = DX[k] * fu; oy[k] = DY[k] * fv; }
    }

    /** Allocate the arrays a piece at a time while the bytecodes allow; true once all are there. */
    static boolean alloc(int reserve) {
        int n = G.W * G.H;
        if (kb == null) { if (left() < n + reserve) return false; kb = new boolean[n]; }
        if (g == null) { if (left() < n + reserve) return false; g = new int[n]; }
        if (poolV == null) { if (left() < POOL + reserve) return false; poolV = new int[POOL]; }
        if (poolN == null) { if (left() < POOL + reserve) return false; poolN = new int[POOL]; head = new int[Cfg.KP_MAX + 4]; }
        return true;
    }

    /** Block the centres around every wall remembered within Chebyshev 6 of c (the king's view and a margin); false
     *  when fewer than reserve bytecodes were left before a row (the marks so far stand; the next turn scans again). */
    static boolean scan(MapLocation c, int reserve) {
        int W = G.W, H = G.H, at = c.x + W * c.y;
        if (at == lastScan) return true;
        int[] mem = Sym.mem;
        boolean[] b = kb;
        int x0 = Math.max(0, c.x - 6), x1 = Math.min(W - 1, c.x + 6), y0 = Math.max(0, c.y - 6), y1 = Math.min(H - 1, c.y + 6);
        for (int y = y0; y <= y1; y++) {
            if (left() < reserve) return false;
            for (int x = x0, i = x0 + W * y; x <= x1; x++, i++) {
                if ((mem[i] & Sym.WALL) == 0) continue;
                for (int yy = Math.max(0, y - 1); yy <= Math.min(H - 1, y + 1); yy++)
                    for (int xx = Math.max(0, x - 1); xx <= Math.min(W - 1, x + 1); xx++) b[xx + W * yy] = true;
            }
        }
        lastScan = at;
        return true;
    }

    /** A centre whose 3x3 lies on the map and touches no seen wall (nor, with Cfg.KP_FRONT, lies too near the front). */
    static boolean open(int x, int y) {
        return x >= 1 && y >= 1 && x < G.W - 1 && y < G.H - 1 && !kb[x + G.W * y]
            && (Cfg.KP_FRONT == 0 || fa * x + fb * y + fc >= fMin);
    }

    /** Cfg.KP_FRONT: the front line from our start home and the enemy's away: |v - away|^2 - |v - home|^2 is linear
     *  in v, 2 * (home - away) . v + |away|^2 - |home|^2. */
    static void front(MapLocation home, MapLocation away) {
        fa = 2 * (home.x - away.x);
        fb = 2 * (home.y - away.y);
        fc = away.x * away.x + away.y * away.y - home.x * home.x - home.y * home.y;
        fMin = (fa * home.x + fb * home.y + fc) * Cfg.KP_FRONT / 100;
    }

    /** m is on our half: nearer our start home than the enemy start away (King.pickHomeMine's test). */
    static boolean homeHalf(MapLocation m, MapLocation home, MapLocation away) {
        int d = home.distanceSquaredTo(m), e = m.distanceSquaredTo(away);
        return d < e || d == e && Cfg.MINE_TIES;
    }

    /** Another of our kings sits within Chebyshev 2 of m. */
    static boolean held(MapLocation m, MapLocation me) {
        for (int k = 0; k < Comms.numKings; k++)
            if (G.cheb(Comms.kings[k], m) <= 2 && !Comms.kings[k].equals(me)) return true;
        return false;
    }

    static void push(int v, int f) {
        if (pn >= POOL) return;
        int e = pn++;
        poolV[e] = v;
        poolN[e] = head[f];
        head[f] = e + 1;
        pending++;
    }

    /** Start a search toward the king at c from the seats of mines[0..n) on our half (home vs away) that no other
     *  king holds; false when no seat is near enough to seed, or when fewer than Cfg.KP_RESERVE bytecodes were left
     *  before a mine (then nothing is searched and the next turn starts again). */
    static boolean start(MapLocation c, MapLocation[] mines, int n, MapLocation home, MapLocation away) {
        int W = G.W, max = Cfg.KP_MAX;
        gen++;
        if (gen >= Integer.MAX_VALUE / SPAN - 1) gen = 1;
        base = gen * SPAN;
        for (int f = 0; f < head.length; f++) head[f] = 0;
        pn = 0; pending = 0; maxF = max + 1; curF = maxF + 1;
        kx = c.x; ky = c.y; kIdx = c.x + W * c.y;
        startRound = G.round; minesAt = Comms.numMines; searches++;
        ready = false; seated = false;
        if (Cfg.KP_FRONT != 0) front(home, away);
        int[] gg = g;
        for (int i = 0; i < n; i++) {
            if (left() < Cfg.KP_RESERVE) { building = false; pending = 0; minesAt = -1; return false; }
            MapLocation m = mines[i];
            if (G.cheb(m, c) > max + 2 || !homeHalf(m, home, away) || held(m, c)) continue;
            for (int k = 0; k < 9; k++) {
                int sx = m.x + (k == 8 ? 0 : ox[k]), sy = m.y + (k == 8 ? 0 : oy[k]);
                if (!open(sx, sy) && !(sx == kx && sy == ky)) continue;
                int s = sx + W * sy;
                if (gg[s] >= base) continue;
                int f = Math.max(Math.abs(sx - kx), Math.abs(sy - ky));
                if (f > maxF) continue;
                gg[s] = base;
                push(s, f);
                if (f < curF) curF = f;
            }
        }
        building = pending > 0;
        if (!building) failRound = G.round;
        return building;
    }

    /** Continue the search while more than reserve bytecodes are left. */
    static void work(int reserve) {
        if (!building) return;
        int[] gg = g, pv = poolV, pnx = poolN, hd = head, OX = ox, OY = oy;
        boolean[] b = kb;
        int W = G.W, H = G.H, bs = base, X = kx, Y = ky, K = kIdx, mf = maxF, A = fa, B = fb, C = fc, F = fMin;
        while (pending > 0) {
            if (left() < reserve) return;
            if (curF > mf) break;
            int e = hd[curF];
            if (e == 0) { curF++; continue; }
            hd[curF] = pnx[e - 1];
            pending--;
            int u = pv[e - 1], ux = u % W, uy = u / W, gu = gg[u] - bs;
            if (gu + Math.max(Math.abs(ux - X), Math.abs(uy - Y)) != curF) continue; // stale entry
            if (u == K) { building = false; ready = true; builtRound = G.round; return; }
            int ng = gu + 1;
            for (int k = 0; k < 8; k++) {
                int vx = ux + OX[k], vy = uy + OY[k];
                if (vx < 1 || vy < 1 || vx >= W - 1 || vy >= H - 1) continue;
                int v = vx + W * vy;
                if (b[v] && v != K) continue;
                if (Cfg.KP_FRONT != 0 && A * vx + B * vy + C < F && v != K) continue;
                int gv = gg[v];
                if (gv >= bs && gv - bs <= ng) continue;
                int f = ng + Math.max(Math.abs(vx - X), Math.abs(vy - Y));
                if (f > mf) continue;
                gg[v] = bs + ng;
                push(v, f);
            }
        }
        building = false;
        fails++;
        failRound = G.round;
    }

    /** Steps left from centre c to a seat by the current search, or -1 when c is off it. */
    static int cost(MapLocation c) {
        int i = c.x + G.W * c.y, v = g[i];
        return v >= base && v - base < SPAN ? v - base : -1;
    }

    /** The neighbours of c with fewer steps left, best first (ties in side-frame order); null when none. */
    static Direction[] downhill(MapLocation c) {
        int gc = cost(c);
        if (gc <= 0) return null;
        int W = G.W, n = 0;
        int[] key = new int[8];
        Direction[] out = new Direction[8];
        for (int k = 0; k < 8; k++) {
            int vx = c.x + ox[k], vy = c.y + oy[k];
            if (!open(vx, vy)) continue;
            int v = g[vx + W * vy];
            if (v < base || v - base >= gc) continue;
            int s = (v - base) * 8 + k, j = n++;
            while (j > 0 && key[j - 1] > s) { key[j] = key[j - 1]; out[j] = out[j - 1]; j--; }
            key[j] = s;
            out[j] = G.dirOf(ox[k], oy[k]);
        }
        if (n == 0) return null;
        Direction[] r = new Direction[n];
        for (int i = 0; i < n; i++) r[i] = out[i];
        return r;
    }

    /** End of the king's turn: allocate, remember walls around it, (re)start the search when due, and work on it.
     *  A turn that allocates does nothing else (the charge for an array lands at once; EscapeTheNight r4 ran over
     *  when the scan and the seeding followed the last allocation), and a scan cut short waits for the next turn. */
    static void tick() {
        if (poolN == null) { alloc(Cfg.KP_RESERVE); return; }
        MapLocation c = G.rc.getLocation();
        if (ox == DX && King.start != null) frame(King.start);
        if (!scan(c, Cfg.KP_RESERVE)) return;
        if (seated && King.homeMine != null && G.cheb(c, King.homeMine) > Cfg.KING_ROAM + 2) seated = false;
        if (!seated && !building && due(c)) {
            MapLocation home = King.start != null ? King.start : c;
            int s = (Sym.mask & Sym.ROT) != 0 ? Sym.ROT : (Sym.mask & Sym.HOR) != 0 ? Sym.HOR : Sym.VER;
            start(c, Comms.mines, Comms.numMines, home, G.mirror(home, s));
        }
        if (building && G.round - startRound > Cfg.KP_TURNS) { building = false; fails++; failRound = G.round; }
        work(Cfg.KP_RESERVE);
    }

    /** A new search is due: none yet, a retry after a failure, a new mine listed, or the way ahead walled off. */
    static boolean due(MapLocation c) {
        if (Comms.numMines == 0) return false;
        if (!ready) return Comms.numMines != minesAt || G.round - failRound >= Cfg.KP_RETRY;
        if (Comms.numMines != minesAt && G.round - startRound >= 8) return true;
        int gc = cost(c);
        return gc < 0 || gc > 0 && downhill(c) == null;
    }

    /** King.move with no enemy or cat in view and movement ready: true when it walked, dug, or holds for a search;
     *  false when seated at a mine (King.homeMine set to it) or with no path, so King.move goes on as before. */
    static boolean follow(MapLocation c) throws GameActionException {
        if (seated) return false;
        if (building) { King.mineWhy = 'S'; return true; }
        if (!ready) return false;
        int gc = cost(c);
        if (gc == 0) {
            seated = true;
            for (int i = 0; i < Comms.numMines; i++)
                if (G.cheb(Comms.mines[i], c) <= 1) { King.homeMine = Comms.mines[i]; break; }
            return false;
        }
        Direction[] ds = downhill(c);
        if (ds == null) { King.mineWhy = 'B'; return true; }
        RobotController rc = G.rc;
        for (Direction d : ds) {
            if (!rc.canMove(d) || !King.unplugged(c, d) || !King.laneOk(c, d)) continue;
            rc.move(d); G.me = rc.getLocation(); steps++; King.mineWhy = 'P'; King.pickUp();
            return true;
        }
        // blocked: dig the dirt of the best step when that is all that stops it
        if (Cfg.KING_DIG && rc.isActionReady() && rc.getGlobalCheese() >= Cfg.KING_DIG_MIN) {
            MapLocation t = c.add(ds[0]);
            MapLocation d = King.digTile(c, t);
            if (d != null && rc.canRemoveDirt(d)) {
                rc.removeDirt(d); digs++; King.digs++; Sym.mem[G.idx(d)] &= ~Sym.DIRT; King.mineWhy = 'G';
                return true;
            }
        }
        King.mineWhy = 'W';
        return true;
    }
}
