package zdn;

import battlecode.common.*;

/**
 * zdn perception memory (docs/zdn.md, "The boundary"): what the robot has seen, with ages, plus symmetry candidates.
 * Everything here records or summarises observations; nothing chooses an action. Ported from zdp's Sym, CatLane and
 * TrapRisk, and from the observation parts of Baby (myMines, the remembered cheese tile, sightings).
 */
public final class ZM {
    public static final int SEEN = 1, WALL = 2, DIRT = 4, MINE = 8, RTRAP = 16, CTRAP = 32, KING = 128;

    public static int[] mem;
    /** Cheese on the tile when last seen, and the round + 1 it was last seen (0 = never). */
    public static int[] cheese, seenAt;
    public static int mask = ZG.ROT | ZG.HOR | ZG.VER;

    // ---- sightings this turn
    public static RobotInfo[] robots = new RobotInfo[0];
    public static RobotInfo cat, eKing, nearEnemy;
    public static int nEnemy, nAlly, nCat;

    // ---- remembered observations
    public static final int[] myMines = new int[24];
    public static int myMineCount;
    /** Nearest visible cheese, kept until a nearer one is seen or it is seen empty (zdp Baby.cheeseTarget's
     *  perception part). */
    public static MapLocation cheeseT;
    public static MapLocation lastEKing;
    public static int lastEKingRound = -1000;
    public static MapLocation catSeen;
    public static int catSeenRound = -1000;
    /** Per symmetry bit (1, 2, 4 -> 0, 1, 2): round its enemy-king spot (mirror of home) was last seen empty. */
    public static final int[] spotEmpty = {-1000, -1000, -1000};

    // ---- birth facts (babies)
    public static MapLocation home;
    /** A king's first tile (zdp King.start). */
    public static MapLocation start;
    public static int fu = 1, fv = 1, birthRound = -1, slot, babyIdx;
    public static boolean scoutBit;

    // ---- own motion
    public static MapLocation anchor;
    public static int anchorSince, carryStart = -1000, lastPick = -1000, picks, moves;
    public static MapLocation lastPos;
    public static int arrived = -1;

    public static void init() {
        int n = ZG.W * ZG.H;
        mem = new int[n];
        cheese = new int[n];
        seenAt = new int[n];
        lane = new int[n];
        risk = new int[n];
    }

    public static boolean known(int x, int y) { return (mem[x + y * ZG.W] & SEEN) != 0; }

    public static boolean blocked(int x, int y) { return !ZG.onMap(x, y) || (mem[x + y * ZG.W] & WALL) != 0; }

    public static boolean dirt(int x, int y) { return ZG.onMap(x, y) && (mem[x + y * ZG.W] & DIRT) != 0; }

    // ------------------------------------------------------------------ terrain and symmetry

    /** Sense the map in view (zdp Sym.record + Baby.sense's cheese and mine bookkeeping). */
    public static void senseMap() throws GameActionException {
        MapInfo[] infos = ZG.rc.senseNearbyMapInfos();
        MapLocation me = ZG.rc.getLocation();
        MapLocation best = null;
        int bd = Integer.MAX_VALUE, r1 = ZG.round + 1;
        for (MapInfo mi : infos) {
            MapLocation l = mi.getMapLocation();
            boolean mine = mi.hasCheeseMine();
            int i = l.x + l.y * ZG.W;
            TrapType t = mi.getTrap();
            record(l.x, l.y, mi.isWall(), mi.isDirt(), mine);
            if (t == TrapType.RAT_TRAP) mem[i] |= RTRAP;
            else if (t == TrapType.CAT_TRAP) mem[i] |= CTRAP;
            if (mine) noteMine(l.x, l.y);
            int c = mi.getCheeseAmount();
            cheese[i] = c;
            seenAt[i] = r1;
            if (c > 0) {
                int d = me.distanceSquaredTo(l);
                if (d < bd) { bd = d; best = l; }
            } else if (cheeseT != null && cheeseT.equals(l)) cheeseT = null;
        }
        if (best != null && (cheeseT == null || me.distanceSquaredTo(cheeseT) > bd)) cheeseT = best;
    }

    public static boolean record(int x, int y, boolean wall, boolean dirt, boolean mine) {
        int i = x + y * ZG.W;
        int v = SEEN | (wall ? WALL : 0) | (dirt ? DIRT : 0) | (mine ? MINE : 0);
        int old = mem[i];
        mem[i] = v | (old & KING);
        if ((old & SEEN) != 0 || mask == ZG.ROT || mask == ZG.HOR || mask == ZG.VER) return false;
        int before = mask, stat = v & (WALL | MINE);
        if ((mask & ZG.ROT) != 0) check(stat, ZG.W - 1 - x, ZG.H - 1 - y, ZG.ROT);
        if ((mask & ZG.HOR) != 0) check(stat, x, ZG.H - 1 - y, ZG.HOR);
        if ((mask & ZG.VER) != 0) check(stat, ZG.W - 1 - x, y, ZG.VER);
        return mask != before;
    }

    private static void check(int stat, int mx, int my, int s) {
        int m = mem[mx + my * ZG.W];
        if ((m & SEEN) == 0) return;
        if ((m & (WALL | MINE)) != stat) eliminate(s);
    }

    public static void eliminate(int s) { int n = mask & ~s; if (n != 0) mask = n; }

    public static void merge(int m) { int n = mask & m; if (n != 0) mask = n; }

    public static int symCount() { return (mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1); }

    /** Mark live king bodies (team comms) as KING tiles; clear old marks. */
    public static void markKings(MapLocation[] kings, int n) {
        for (int i = 0; i < mem.length; i++) mem[i] &= ~KING;
        for (int k = 0; k < n; k++) {
            MapLocation c = kings[k];
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++) {
                    int x = c.x + dx, y = c.y + dy;
                    if (ZG.onMap(x, y)) mem[x + y * ZG.W] |= KING;
                }
        }
    }

    public static void noteMine(int x, int y) {
        int key = x * 64 + y;
        for (int i = 0; i < myMineCount; i++) if (myMines[i] == key) return;
        if (myMineCount < myMines.length) myMines[myMineCount++] = key;
    }

    // ------------------------------------------------------------------ robots

    public static void senseRobots() throws GameActionException {
        RobotController rc = ZG.rc;
        MapLocation me = rc.getLocation();
        robots = rc.senseNearbyRobots(-1, null);
        cat = null; eKing = null; nearEnemy = null;
        nEnemy = nAlly = nCat = 0;
        int bc = Integer.MAX_VALUE, be = Integer.MAX_VALUE;
        for (RobotInfo r : robots) {
            if (r.team == Team.NEUTRAL) {
                nCat++;
                if (ZG.coop) seeLane(r.location);
                int d = ZG.catDist4(r.location, me);
                if (d < bc) { bc = d; cat = r; }
            } else if (r.team == ZG.them) {
                nEnemy++;
                if (r.type == UnitType.RAT_KING) eKing = r;
                else {
                    int d = me.distanceSquaredTo(r.location);
                    if (d < be) { be = d; nearEnemy = r; }
                }
            } else nAlly++;
        }
        if (cat != null) { catSeen = cat.location; catSeenRound = ZG.round; }
        if (eKing != null) { lastEKing = eKing.location; lastEKingRound = ZG.round; }
        if (!ZG.coop) observeRisk(me);
        // enemy-king spots (mirror of home under each live symmetry) seen empty
        if (home != null) {
            for (int b = 0; b < 3; b++) {
                int s = 1 << b;
                if ((mask & s) == 0) continue;
                MapLocation c = ZG.mirror(home, s);
                if (me.distanceSquaredTo(c) <= 13 && rc.canSenseLocation(c)) {
                    RobotInfo r = rc.senseRobotAtLocation(c);
                    if (r == null || r.team != ZG.them || r.type != UnitType.RAT_KING) spotEmpty[b] = ZG.round;
                }
            }
        }
    }

    /** Own motion bookkeeping, once per turn at its start. */
    public static void motion() throws GameActionException {
        RobotController rc = ZG.rc;
        MapLocation me = rc.getLocation();
        if (anchor == null || ZG.cheb(me, anchor) > 3) { anchor = me; anchorSince = ZG.round; }
        if (rc.getCarrying() == null) carryStart = -1000;
        else if (carryStart < 0 || ZG.round - carryStart > 11) carryStart = ZG.round;
        if (lastPos == null || !lastPos.equals(me)) { lastPos = me; arrived = ZG.round - 1; }
    }

    // ------------------------------------------------------------------ cat lanes (zdp CatLane)

    static final int LANE = 1 << 16, LROUND = 0xFFFF, LANE_GAP = 20;
    public static int[] lane;

    static void seeLane(MapLocation bl) {
        for (int dy = 0; dy <= 1; dy++)
            for (int dx = 0; dx <= 1; dx++) {
                int x = bl.x + dx, y = bl.y + dy;
                if (!ZG.onMap(x, y)) continue;
                int i = x + y * ZG.W, v = lane[i];
                int l = v & LANE;
                if (v != 0 && ZG.round - ((v & LROUND) - 1) >= LANE_GAP) l = LANE;
                lane[i] = l | (ZG.round + 1);
            }
    }

    public static boolean isLane(int x, int y) { return ZG.onMap(x, y) && (lane[x + y * ZG.W] & LANE) != 0; }

    public static int laneAge(int x, int y) {
        if (!ZG.onMap(x, y)) return 4000;
        int v = lane[x + y * ZG.W];
        return v == 0 ? 4000 : Math.min(4000, ZG.round - ((v & LROUND) - 1));
    }

    // ------------------------------------------------------------------ enemy rat-trap belief (zdp TrapRisk)

    static final int RISK_TTL = 150;
    public static int[] risk;

    static void observeRisk(MapLocation me) {
        boolean grounded = !ZG.rc.isBeingCarried() && !ZG.rc.isBeingThrown();
        for (RobotInfo r : robots) {
            if (r.team != ZG.them || r.type != UnitType.BABY_RAT) continue;
            Direction f = r.direction;
            if (f == null || f == Direction.CENTER) continue;
            putRisk(r.location.add(f));
            putRisk(r.location.add(f.rotateLeft()));
            putRisk(r.location.add(f.rotateRight()));
        }
        if (!grounded) return;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                int x = me.x + dx, y = me.y + dy;
                if (!ZG.onMap(x, y)) continue;
                int i = x + ZG.W * y;
                if (risk[i] != 0 && risk[i] - 1 < arrived) risk[i] = 0;
            }
    }

    static void putRisk(MapLocation l) { if (ZG.onMap(l.x, l.y)) risk[l.x + ZG.W * l.y] = ZG.round + 1; }

    /** A live enemy-trap mark within Chebyshev 1 of (x, y). */
    public static boolean risky(int x, int y) {
        int r = ZG.round - RISK_TTL;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                int a = x + dx, b = y + dy;
                if (!ZG.onMap(a, b)) continue;
                int k = risk[a + ZG.W * b];
                if (k != 0 && k - 1 >= r) return true;
            }
        return false;
    }

    // ------------------------------------------------------------------ birth (babies)

    /** Birth facts: the spawning king (nearest live king), the side frame, and zdb's per-rat hash. */
    public static void born(MapLocation nearestKing) {
        MapLocation me = ZG.rc.getLocation();
        home = nearestKing != null ? nearestKing : me;
        fu = 2 * home.x >= ZG.W - 1 ? 1 : -1;
        fv = 2 * home.y >= ZG.H - 1 ? 1 : -1;
        birthRound = ZG.round;
        int seed = ZG.round * 977 + Math.abs(me.x - home.x) * 31 + Math.abs(me.y - home.y) * 7 + 1;
        slot = (seed * 0x9E3779B1) >>> 8;
        scoutBit = (slot >>> 4) % 4 == 0;
        ZG.seed(seed ^ 0x5bd1e995);
        babyIdx = (ZG.rc.getCurrentRatCost() - 10) / 10 * 4;
    }

    /** l is nearer home than home's twin under the first live symmetry. */
    public static boolean homeHalf(MapLocation l) {
        if (home == null) return true;
        return l.distanceSquaredTo(home) < l.distanceSquaredTo(ZG.mirror(home, ZG.firstSym(mask)));
    }

    // ------------------------------------------------------------------ the turn's perception

    /** Start-of-turn perception, shared by the player and the teacher's recorder: comms, birth (babies) or first king
     *  turn facts, map memory, robots, own motion. */
    public static void perceive() throws GameActionException {
        ZC.readAll();
        ZC.hear();
        markKings(ZC.kings, ZC.numKings);
        if (ZG.king) {
            if (start == null) {
                start = ZG.me;
                if (ZG.round <= 1)
                    for (int s = 1; s <= 4; s <<= 1)
                        if (ZG.mirX(ZG.me.x, s) == ZG.me.x && ZG.mirY(ZG.me.y, s) == ZG.me.y) eliminate(s);
            }
        } else if (birthRound < 0) born(ZC.nearestKing(ZG.me));
        senseMap();
        senseRobots();
        motion();
    }
}
