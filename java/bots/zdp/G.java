package zdp;

import battlecode.common.*;

/** Globals shared by every layer. Refreshed once per turn by {@link #startTurn()}. */
public final class G {
    /** Bytecode ledger (z2h): the bytecodes this robot spent this turn on code its base (u6x) does not run, minus 2.
     *  Every budget read adds it back, so the base's budget checks (Flow.work above all, which spends whatever a turn
     *  leaves) see the same counts as in the base, and games replay the base's until the new code acts. The left()
     *  variants read the clock 2 bytecodes earlier than the base's (no test-only unlimited check, same length), which
     *  the minus 2 absorbs; RobotPlayer resets it after Flow.work. */
    public static int debt;

    /** The bytecode count, read in a method of its own: the engine counts a basic block's instructions only at its end
     *  (a jump, label, return or call of our own code), and a Clock read is no block end, so a read inline would see a
     *  count that depends on javac's line labels. The call to bc() ends the caller's block; its own return counts 1. */
    static int bc() { return Clock.getBytecodeNum(); }

    /** Ledger: close a hook opened with {@code int cd = G.debt, cb = G.bc();}. The hook's cost, plus k (its counted
     *  instructions outside the two reads), is credited only while the cost is at most SETTLE_CAP: a hook that costs
     *  more has acted, and its bytecodes must reach the budget reads. Crediting an acting hook made left() overstate the
     *  budget by thousands, and Seal.work/sense2 overran (z2j on wallsofparadis B: 219 overrun turns from r1888, when
     *  CASH_END's formation ran). */
    static void settle(int cd, int cb, int k) {
        int c = bc() - cb;
        debt = cd;
        if (c <= SETTLE_CAP) debt = cd + c + k;
    }

    static final int SETTLE_CAP = 300;

    /** Cfg.RICH_FIX applies at a site opened with k kings: always from two, from Cfg.RICH_FIX_FROM with one. */
    static boolean fixOn(int k) { return k >= 2 || k >= Cfg.RICH_FIX_KINGS && round >= Cfg.RICH_FIX_FROM; }

    public static RobotController rc;
    public static Team us, them;
    public static int W, H, myId;
    public static int round;
    public static MapLocation me;
    public static Direction facing;
    public static boolean king;
    public static boolean coop;
    public static int globalCheese, rawCheese;

    /** Map centre in doubled coordinates (so half-integer centres stay integral). */
    public static int cx2, cy2;

    /** The eight compass directions, enum order. Never used for tie-breaks on its own. */
    public static final Direction[] DIRS = {
        Direction.NORTH, Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST,
        Direction.SOUTH, Direction.SOUTHWEST, Direction.WEST, Direction.NORTHWEST };

    public static void init(RobotController r) {
        rc = r;
        us = r.getTeam();
        them = us.opponent();
        W = r.getMapWidth();
        H = r.getMapHeight();
        myId = r.getID();
        cx2 = W - 1;
        cy2 = H - 1;
    }

    public static void startTurn() {
        round = rc.getRoundNum();
        me = rc.getLocation();
        facing = rc.getDirection();
        king = rc.getType() == UnitType.RAT_KING;
        coop = rc.isCooperation();
        globalCheese = rc.getGlobalCheese();
        rawCheese = rc.getRawCheese();
    }

    // ---------------------------------------------------------------- geometry

    public static int idx(int x, int y) { return x + y * W; }

    public static int idx(MapLocation l) { return l.x + l.y * W; }

    public static boolean onMap(int x, int y) { return x >= 0 && y >= 0 && x < W && y < H; }

    /** d, then its two 45-degree neighbours: left first, or (Cfg.SYM_HAND) the side hand(from, d) names, so mirror
     *  twins (a reflection swaps left and right) try mirrored candidates in the same order. */
    public static Direction[] fan(MapLocation from, Direction d) {
        Direction l = d.rotateLeft(), r = d.rotateRight();
        return Cfg.SYM_HAND && hand(from, d) < 0 ? new Direction[]{d, r, l} : new Direction[]{d, l, r};
    }

    /** fan() plus the two 90-degree neighbours, in the same side order. */
    public static Direction[] fan5(MapLocation from, Direction d) {
        Direction l = d.rotateLeft(), r = d.rotateRight();
        return Cfg.SYM_HAND && hand(from, d) < 0 ? new Direction[]{d, r, l, r.rotateRight(), l.rotateLeft()}
            : new Direction[]{d, l, r, l.rotateLeft(), r.rotateRight()};
    }

    /** 4x the squared distance from l to the true map centre (doubled coordinates, so exact on even-sized maps:
     *  mirror twins get equal values). */
    public static int centreD2x4(MapLocation l) {
        int dx = 2 * l.x - cx2, dy = 2 * l.y - cy2;
        return dx * dx + dy * dy;
    }

    /** 2x the Chebyshev distance from (x, y) to the true map centre. */
    public static int centreCheb2(int x, int y) { return Math.max(Math.abs(2 * x - cx2), Math.abs(2 * y - cy2)); }

    static final MapLocation ORIGIN = new MapLocation(0, 0);

    /** The compass direction of vector (dx, dy) (directionTo is scale-free, so doubled vectors work). */
    public static Direction dirOf(int dx, int dy) { return ORIGIN.directionTo(new MapLocation(dx, dy)); }

    /** Direction from l toward the true map centre (directionTo is scale-free and sign-symmetric, so mirror twins
     *  get mirrored directions). */
    public static Direction toCentre(MapLocation l) {
        return dirOf(cx2 - 2 * l.x, cy2 - 2 * l.y);
    }

    public static int cheb(MapLocation a, MapLocation b) {
        return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y));
    }

    /**
     * Side-symmetric handedness for a robot at {@code from} heading along {@code d}: +1 or -1.
     * cross(from - centre, d) is invariant under 180-degree rotation and flips sign under a reflection,
     * so mirrored robots on the two sides make mirrored choices. Exact zero falls back to the
     * parity of the doubled offset, which is also mirror-consistent for rotation.
     */
    public static int hand(MapLocation from, Direction d) {
        int px = 2 * from.x - cx2, py = 2 * from.y - cy2;
        int c = px * d.dy - py * d.dx;
        if (c > 0) return Cfg.HAND_FLIP ? -1 : 1;
        if (c < 0) return Cfg.HAND_FLIP ? 1 : -1;
        // d points along the line through the centre: the dot product is the same for a reflected twin, so the
        // side frame's chirality (Cfg.SYM_HAND) breaks the tie the mirrored way
        int dot = px * d.dx + py * d.dy;
        return Cfg.HAND_FLIP ? (dot >= 0 ? -chir : chir) : (dot >= 0 ? chir : -chir);
    }

    /** Tie bit for ranking candidate tiles l seen from {@code from} (spawn, trap and target picks), the same for a
     *  rotated or a reflected twin when Cfg.SYM_HAND: the cross product of from's centre offset with the exact vector
     *  to l, times the side chirality (both flip under a reflection), else the sign of their dot product. Without
     *  SYM_HAND, hand() of the compass direction as before: two tiles a knight's move apart share one direction, so
     *  their tie fell to loop order, which does not mirror (where_are_you's second spawn), and a bonus from hand()
     *  alone flips under a reflection. */
    public static boolean pickBit(MapLocation from, MapLocation l) {
        return Cfg.SYM_HAND ? pickBitSym(from, l) : hand(from, from.directionTo(l)) > 0;
    }

    static boolean pickBitSym(MapLocation from, MapLocation l) {
        int px = 2 * from.x - cx2, py = 2 * from.y - cy2, dx = l.x - from.x, dy = l.y - from.y;
        int c = px * dy - py * dx;
        return c != 0 ? c * chir > 0 : px * dx + py * dy >= 0;
    }

    /** Side chirality for hand()'s tie: the product of the signs of the home king's offset from the centre (the same
     *  for a rotated twin, opposite for a reflected one), or 1. */
    public static int chir = 1;

    /** Set the chirality from our home king's position (Cfg.SYM_HAND, Cfg.SYM_TIE). */
    public static void frame(MapLocation home) { if (Cfg.SYM_HAND || Cfg.SYM_TIE) chir = chirOf(home); }

    /** Cfg.SYM_TIE: a ranks before b among tiles of equal score seen from {@code from}. The cross product of from's
     *  centre offset with the offset to the tile (signed by chir), then the dot product, order tiles the same way for a
     *  rotated or a reflected twin (from off the centre); one pick bit leaves two tiles on the same side to the loop. */
    public static boolean tieFirst(MapLocation from, MapLocation a, MapLocation b) {
        int px = 2 * from.x - cx2, py = 2 * from.y - cy2;
        int ax = a.x - from.x, ay = a.y - from.y, bx = b.x - from.x, by = b.y - from.y;
        int ca = (px * ay - py * ax) * chir, cb = (px * by - py * bx) * chir;
        if (ca != cb) return ca > cb;
        return px * ax + py * ay > px * bx + py * by;
    }

    static int chirOf(MapLocation home) {
        int px = 2 * home.x - cx2, py = 2 * home.y - cy2;
        return (px >= 0 ? 1 : -1) * (py >= 0 ? 1 : -1);
    }

    /** Mirror of a location under symmetry s (Sym.ROT/HOR/VER bit). */
    public static int mirX(int x, int s) { return s == Sym.HOR ? x : W - 1 - x; }

    public static int mirY(int y, int s) { return s == Sym.VER ? y : H - 1 - y; }

    public static MapLocation mirror(MapLocation l, int s) { return new MapLocation(mirX(l.x, s), mirY(l.y, s)); }

    /** Deterministic, side-symmetric pseudo-random stream (seeded from mirror-invariant integers). */
    public static int rngState = 1;

    public static void seed(int s) { rngState = s * 0x9E3779B1 + 0x7F4A7C15; if (rngState == 0) rngState = 1; }

    public static int rand(int n) {
        int x = rngState;
        x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
        rngState = x;
        return (x >>> 1) % n;
    }
}
