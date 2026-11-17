package zdn;

import battlecode.common.*;

/** zdn globals and geometry (docs/zdn.md). Pure helpers; no decisions. */
public final class ZG {
    public static RobotController rc;
    public static Team us, them;
    public static int W, H, id, cx2, cy2;
    public static int round;
    public static MapLocation me;
    public static Direction facing;
    public static boolean king, coop;
    public static int global, raw;

    public static final Direction[] DIRS = {
        Direction.NORTH, Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST,
        Direction.SOUTH, Direction.SOUTHWEST, Direction.WEST, Direction.NORTHWEST };
    public static final int[] DX = {0, 1, 1, 1, 0, -1, -1, -1}, DY = {1, 1, 0, -1, -1, -1, 0, 1};

    public static void init(RobotController r) {
        rc = r;
        us = r.getTeam();
        them = us.opponent();
        W = r.getMapWidth();
        H = r.getMapHeight();
        id = r.getID();
        cx2 = W - 1;
        cy2 = H - 1;
    }

    public static void refresh() {
        round = rc.getRoundNum();
        me = rc.getLocation();
        facing = rc.getDirection();
        king = rc.getType() == UnitType.RAT_KING;
        coop = rc.isCooperation();
        global = rc.getGlobalCheese();
        raw = rc.getRawCheese();
    }

    /** Index 0..7 of a compass direction in DIRS order, -1 for CENTER/null. */
    public static int di(Direction d) {
        if (d == null) return -1;
        for (int k = 0; k < 8; k++) if (DIRS[k] == d) return k;
        return -1;
    }

    public static int di(int dx, int dy) {
        dx = Integer.signum(dx); dy = Integer.signum(dy);
        for (int k = 0; k < 8; k++) if (DX[k] == dx && DY[k] == dy) return k;
        return -1;
    }

    /** Rotation steps (0..4) between two directions. */
    public static int rot(Direction a, Direction b) {
        int i = di(a), j = di(b);
        if (i < 0 || j < 0) return 0;
        int r = Math.abs(i - j) % 8;
        return r > 4 ? 8 - r : r;
    }

    public static boolean onMap(int x, int y) { return x >= 0 && y >= 0 && x < W && y < H; }

    public static int idx(int x, int y) { return x + y * W; }

    public static int cheb(MapLocation a, MapLocation b) { return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y)); }

    public static int d2(MapLocation a, MapLocation b) { return a.distanceSquaredTo(b); }

    /** Side-symmetric handedness (zdp G.hand). */
    public static int hand(MapLocation from, Direction d) {
        int px = 2 * from.x - cx2, py = 2 * from.y - cy2;
        int c = px * d.dy - py * d.dx;
        if (c > 0) return 1;
        if (c < 0) return -1;
        return px * d.dx + py * d.dy >= 0 ? 1 : -1;
    }

    public static int pickBit(MapLocation from, MapLocation l) {
        if (from.equals(l)) return 0;
        return hand(from, from.directionTo(l)) > 0 ? 1 : 0;
    }

    public static final int ROT = 1, HOR = 2, VER = 4;

    public static int mirX(int x, int s) { return s == HOR ? x : W - 1 - x; }

    public static int mirY(int y, int s) { return s == VER ? y : H - 1 - y; }

    public static MapLocation mirror(MapLocation l, int s) { return new MapLocation(mirX(l.x, s), mirY(l.y, s)); }

    /** First live symmetry in priority ROT, HOR, VER. */
    public static int firstSym(int mask) { return (mask & ROT) != 0 ? ROT : (mask & HOR) != 0 ? HOR : VER; }

    public static int clampX(int x) { return Math.max(0, Math.min(W - 1, x)); }

    public static int clampY(int y) { return Math.max(0, Math.min(H - 1, y)); }

    /** Saturate to int16 for record features. */
    public static int sat(int v) { return v > 32767 ? 32767 : v < -32767 ? -32767 : v; }

    /** Chebyshev distance from p to a 2x2 body with lower-left bl. */
    public static int chebToBody(MapLocation bl, MapLocation p) {
        int dx = p.x < bl.x ? bl.x - p.x : p.x > bl.x + 1 ? p.x - bl.x - 1 : 0;
        int dy = p.y < bl.y ? bl.y - p.y : p.y > bl.y + 1 ? p.y - bl.y - 1 : 0;
        return Math.max(dx, dy);
    }

    /** 4 x dist2 from p to the centre of a 2x2 body with lower-left bl. */
    public static int catDist4(MapLocation bl, MapLocation p) {
        int dx = 2 * p.x - (2 * bl.x + 1), dy = 2 * p.y - (2 * bl.y + 1);
        return dx * dx + dy * dy;
    }

    /** Enemy-cone test (zdp Baby.sees): tile t lies in robot r's vision cone (90 degrees, dist2 <= 20). */
    public static boolean sees(RobotInfo r, MapLocation t) {
        if (r.direction == null || r.direction == Direction.CENTER) return true;
        int dx = t.x - r.location.x, dy = t.y - r.location.y;
        if (dx * dx + dy * dy > 20) return false;
        int fx = r.direction.dx, fy = r.direction.dy;
        int dot = dx * fx + dy * fy, cross = dx * fy - dy * fx;
        return dot >= Math.abs(cross);
    }

    /** Deterministic per-robot stream for code-drawn candidate samples (explore points). */
    public static int rng = 1;

    public static void seed(int s) { rng = s * 0x9E3779B1 + 0x7F4A7C15; if (rng == 0) rng = 1; }

    public static int rand(int n) {
        int x = rng;
        x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
        rng = x;
        return (x >>> 1) % n;
    }
}
