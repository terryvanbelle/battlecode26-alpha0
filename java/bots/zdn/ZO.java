package zdn;

import battlecode.common.*;

/**
 * zdn inputs (docs/zdn.md, "Inputs"): the state vector S, the local grid L, the register heads' options and the
 * step head's options, each option with its kind and feature row. Shared by the player (ZPlayer) and the teacher's
 * recorder (ZT), so imitation sees exactly what the player will see. Pure perception: every comparison and selection
 * is left to the net.
 */
public final class ZO {
    // ------------------------------------------------------------------ primitive ops (RcLog / ZT numbering)
    public static final int OP_MOVE_FWD = 0, OP_MOVE = 1, OP_TURN = 2, OP_BUILD = 3, OP_BECOME_KING = 4,
        OP_PLACE_DIRT = 5, OP_REMOVE_DIRT = 6, OP_PLACE_RAT_TRAP = 7, OP_REMOVE_RAT_TRAP = 8, OP_PLACE_CAT_TRAP = 9,
        OP_REMOVE_CAT_TRAP = 10, OP_PICKUP = 11, OP_ATTACK = 12, OP_WRITE = 13, OP_TRANSFER = 14, OP_THROW = 15,
        OP_DROP = 16, OP_CARRY = 17, OP_DISINTEGRATE = 18, OP_RESIGN = 19, OP_SQUEAK = 20, OP_END = 21;

    // ------------------------------------------------------------------ step option kinds (one space, both units)
    public static final int K_END = 0, K_MOVE = 1, K_TURN = 2, K_BITE = 3, K_PICK = 4, K_CARRY = 5, K_DROP = 6,
        K_THROW = 7, K_CTRAP = 8, K_RTRAP = 9, K_DIG = 10, K_DIRT = 11, K_XFER = 12, K_KING = 13, K_SQ_MINE = 14,
        K_SQ_SYM = 15, K_SQ_CAT = 16, K_SQ_CASH = 17, K_SQ_EKING = 18, K_BUILD = 19, K_ALERT = 20, NK = 21;

    public static final int DS = 20;              // step row width
    public static final int NCB = 77, NCK = 16;   // step context width (baby, king)
    public static final int R = 4, SIDE = 2 * R + 1, CH = 13, NL = SIDE * SIDE * CH;
    public static final int[] REG_D_BABY = {2, 3, 14, 10, 10, 14}, REG_K_BABY = {4, 40, 2, 5, 4, 14};
    public static final int[] REG_D_KING = {10, 12}, REG_K_KING = {2, 5};

    /** An option list: kinds, feature rows, and what executing each option means. */
    public static final class Opts {
        public final int D;
        public int n;
        public final int[] kind, op, dir, lx, ly, amt, ex, ey;
        public final int[][] x;

        public Opts(int D, int cap) {
            this.D = D;
            kind = new int[cap]; op = new int[cap]; dir = new int[cap]; lx = new int[cap]; ly = new int[cap];
            amt = new int[cap]; ex = new int[cap]; ey = new int[cap];
            x = new int[cap][D];
        }

        public void clear() { n = 0; }

        int add(int k, int o, int d, int px, int py, int a, int qx, int qy) {
            if (n >= kind.length) return -1;
            int i = n++;
            kind[i] = k; op[i] = o; dir[i] = d; lx[i] = px; ly[i] = py; amt[i] = a; ex[i] = qx; ey[i] = qy;
            java.util.Arrays.fill(x[i], 0);
            return i;
        }

        public void copyFrom(Opts s) {
            n = s.n;
            System.arraycopy(s.kind, 0, kind, 0, n); System.arraycopy(s.op, 0, op, 0, n); System.arraycopy(s.dir, 0, dir, 0, n);
            System.arraycopy(s.lx, 0, lx, 0, n); System.arraycopy(s.ly, 0, ly, 0, n); System.arraycopy(s.amt, 0, amt, 0, n);
            System.arraycopy(s.ex, 0, ex, 0, n); System.arraycopy(s.ey, 0, ey, 0, n);
            for (int i = 0; i < n; i++) System.arraycopy(s.x[i], 0, x[i], 0, D);
        }

        /** Index of the first option of this kind, or -1. */
        public int first(int k) {
            for (int i = 0; i < n; i++) if (kind[i] == k) return i;
            return -1;
        }

        /** Index of the first option of kind >= kmin with this target tile, or -1. */
        public int findTile(int kmin, int px, int py) {
            for (int i = 0; i < n; i++) if (kind[i] >= kmin && lx[i] == px && ly[i] == py) return i;
            return -1;
        }

        /** Index of the option with this kind and target tile, or -1. */
        public int find(int k, int px, int py) {
            for (int i = 0; i < n; i++) if (kind[i] == k && lx[i] == px && ly[i] == py) return i;
            return -1;
        }
    }

    public static final Opts step = new Opts(DS, 200);
    public static final Opts[] regB = new Opts[ZR.NB], regK = new Opts[ZR.NK];

    static {
        for (int r = 0; r < ZR.NB; r++) regB[r] = new Opts(REG_D_BABY[r], 160);
        for (int r = 0; r < ZR.NK; r++) regK[r] = new Opts(REG_D_KING[r], 48);
    }

    // ------------------------------------------------------------------ helpers

    static MapLocation nearestKingTo(MapLocation p) { return ZC.nearestKing(p); }

    static int d2or(MapLocation a, MapLocation b, int none) { return a == null || b == null ? none : a.distanceSquaredTo(b); }

    static int cap(int v, int c) { return v > c ? c : v < -c ? -c : v; }

    static boolean inTransferReach(MapLocation p, MapLocation k) {
        int dx = Math.max(0, Math.abs(p.x - k.x) - 1), dy = Math.max(0, Math.abs(p.y - k.y) - 1);
        return dx * dx + dy * dy <= 9 && !(Math.abs(p.x - k.x) <= 1 && Math.abs(p.y - k.y) <= 1);
    }

    static boolean inCatBody(MapLocation bl, MapLocation p) { return p.x >= bl.x && p.x <= bl.x + 1 && p.y >= bl.y && p.y <= bl.y + 1; }

    /** The tile is in the cat's body or its next two footprints along its facing (zdp Baby.laneMask). */
    static boolean catLane(RobotInfo c, MapLocation p) {
        if (c == null) return false;
        if (inCatBody(c.location, p)) return true;
        Direction cd = c.direction;
        if (cd == null || cd == Direction.CENTER) return false;
        MapLocation n1 = c.location.add(cd), n2 = n1.add(cd);
        return inCatBody(n1, p) || inCatBody(n2, p);
    }

    static boolean catEarshot(MapLocation bl, MapLocation p) {
        for (int dx = 0; dx <= 1; dx++) for (int dy = 0; dy <= 1; dy++) if (p.distanceSquaredTo(bl.translate(dx, dy)) <= 16) return true;
        return false;
    }

    /** Ring of 16 tiles at Chebyshev 2 (zdp King.RX/RY order). */
    static final int[] RX = {-2, -1, 0, 1, 2, 2, 2, 2, 2, 1, 0, -1, -2, -2, -2, -2},
                       RY = {-2, -2, -2, -2, -2, -1, 0, 1, 2, 2, 2, 2, 2, 1, 0, -1};

    /** zdp King.ringArcs: open arcs of the 16-tile ring around a king centre over the map memory. */
    static int ringArcs(MapLocation c) {
        int closed = 0;
        for (int i = 0; i < 16; i++) {
            int x = c.x + RX[i], y = c.y + RY[i];
            if (!ZG.onMap(x, y) || (ZM.mem[x + ZG.W * y] & (ZM.WALL | ZM.DIRT)) != 0) closed |= 1 << i;
        }
        for (int i = 0; i < 16; i += 4)
            if ((closed >> i & 1) != 0 && (closed >> ((i + 1) & 15) & 1) == 0 && (closed >> ((i + 15) & 15) & 1) == 0) closed &= ~(1 << i);
        if (closed == 0) return 1;
        int arcs = 0;
        for (int i = 0; i < 16; i++) if ((closed >> i & 1) == 0 && (closed >> ((i + 15) & 15) & 1) != 0) arcs++;
        return arcs;
    }

    /** Throw flight score along d from p (zdp Baby.tryThrow, before its tie bit); -999 when the first tile blocks. */
    static int throwScore(MapLocation p, int d, RobotInfo[] robots) {
        int ddx = ZG.DX[d], ddy = ZG.DY[d];
        int x = p.x + ddx, y = p.y + ddy;
        if (!ZG.onMap(x, y) || (ZM.mem[x + ZG.W * y] & (ZM.WALL | ZM.DIRT)) != 0 || robotAt(robots, x, y) != null) return -999;
        for (int k = 2; k <= 8; k++) {
            x += ddx; y += ddy;
            int dmg = 10 + 4 * (2 * (4 - (k - 1) / 2) + ((k & 1) == 1 ? 1 : 0));
            if (!ZG.onMap(x, y)) return 10;
            if ((ZM.mem[x + ZG.W * y] & (ZM.WALL | ZM.DIRT)) != 0) return dmg;
            RobotInfo r = robotAt(robots, x, y);
            if (r != null) {
                boolean far = k > 2 && r.type == UnitType.BABY_RAT;
                if (r.team == ZG.them) return far ? dmg * 2 * 40 / 100 : dmg * (r.type == UnitType.RAT_KING ? 3 : 2);
                if (r.team == ZG.us) return -dmg;
                return 200;
            }
        }
        return 10;
    }

    static RobotInfo robotAt(RobotInfo[] robots, int x, int y) {
        for (RobotInfo r : robots) {
            int ox = x - r.location.x, oy = y - r.location.y;
            if (r.type == UnitType.RAT_KING) { if (ox >= -1 && ox <= 1 && oy >= -1 && oy <= 1) return r; }
            else if (r.team == Team.NEUTRAL) { if (ox >= 0 && ox <= 1 && oy >= 0 && oy <= 1) return r; }
            else if (ox == 0 && oy == 0) return r;
        }
        return null;
    }

    /** Fan rank of d in fan5(to) = [to, L, R, LL, RR], 5 when outside it. */
    static int fanRank(Direction to, Direction d) {
        if (to == null || to == Direction.CENTER) return 5;
        Direction l = to.rotateLeft(), r = to.rotateRight();
        if (d == to) return 0;
        if (d == l) return 1;
        if (d == r) return 2;
        if (d == l.rotateLeft()) return 3;
        if (d == r.rotateRight()) return 4;
        return 5;
    }

    // ------------------------------------------------------------------ state vector S

    static int[] sb = new int[256];
    static int sn;

    static void put(int v) { sb[sn++] = ZG.sat(v); }

    static void putBit(boolean b) { sb[sn++] = b ? 1 : 0; }

    static void oneHot(int v, int n) { for (int i = 0; i < n; i++) sb[sn++] = i == v ? 1 : 0; }

    /** The state vector for this turn (call after perception at the start of the turn). */
    public static int[] state() throws GameActionException {
        sn = 0;
        RobotController rc = ZG.rc;
        MapLocation me = ZG.me;
        int r = ZG.round;
        // common
        put(r); putBit(r >= 30); put(r & 1); putBit(ZG.coop); putBit(rc.getBackstabbingTeam() != null);
        put(ZG.W); put(ZG.H); put(rc.getHealth()); put(ZG.raw); put(ZG.global); put(rc.getAllCheese());
        int cost = rc.getCurrentRatCost();
        put(cost); put((cost - 10) / 10 * 4);
        putBit(rc.isActionReady()); putBit(rc.isMovementReady()); putBit(rc.isTurningReady());
        put(rc.getActionCooldownTurns()); put(rc.getMovementCooldownTurns()); put(rc.getTurningCooldownTurns());
        Direction f = ZG.facing;
        put(f == null ? 0 : f.dx); put(f == null ? 0 : f.dy);
        RobotInfo carry = rc.getCarrying();
        putBit(carry != null && carry.team == ZG.us); putBit(carry != null && carry.team != ZG.us);
        put(carry == null ? 0 : ZR.since(ZM.carryStart)); putBit(rc.isBeingCarried());
        put(me.x); put(me.y); put(2 * me.x - ZG.cx2); put(2 * me.y - ZG.cy2);
        for (int b = 0; b < 3; b++) put(ZM.mask >> b & 1);
        put(ZC.numKings); put(ZC.numMines); put(ZM.myMineCount); put(ZC.budget);
        put(rc.getNumberCatTraps()); put(rc.getNumberRatTraps());
        MapLocation k = ZC.nearestKing(me);
        put(d2or(me, k, 9999)); put(k == null ? 99 : ZG.cheb(me, k)); put(k == null ? 0 : k.x - me.x); put(k == null ? 0 : k.y - me.y);
        putBit(k != null && inTransferReach(me, k)); putBit(k != null && me.distanceSquaredTo(k) <= 16);
        // alert, formation, enemy king, cats from comms
        put(Math.min(ZC.alertAge, 99)); put(ZC.alertThreat); put(d2or(me, ZC.alert, 9999));
        boolean threatened = !ZG.coop && ZC.alert != null && ZC.alertAge <= 2 && ZC.alertThreat >= 1 && me.distanceSquaredTo(ZC.alert) <= 100;
        putBit(threatened);
        MapLocation fs = ZC.formSite();
        putBit(fs != null); put(d2or(me, fs, 9999)); put(fs == null ? 99 : ZG.cheb(me, fs)); put(fs == null ? 0 : nearerAllies(fs, me));
        putBit(rc.canBecomeRatKing());
        put(Math.min(ZC.ekingAge, 99)); put(d2or(me, ZC.eking, 9999));
        for (int c = 0; c < 2; c++) { put(Math.min(ZC.catAge[c], 99)); put(d2or(me, ZC.cats[c], 9999)); }
        // sightings
        put(ZM.nEnemy); put(ZM.nAlly); put(ZM.nCat);
        RobotInfo e = ZM.nearEnemy;
        put(e == null ? 9999 : me.distanceSquaredTo(e.location)); put(e == null ? 0 : e.health);
        putBit(e != null && ZG.sees(e, me)); put(e == null ? 0 : e.location.x - me.x); put(e == null ? 0 : e.location.y - me.y);
        putBit(e != null && e.health < rc.getHealth());
        RobotInfo ek = ZM.eKing;
        putBit(ek != null); put(ek == null ? 9999 : me.distanceSquaredTo(ek.location)); put(ek == null ? 0 : ek.health);
        put(ZR.since(ZM.lastEKingRound)); put(d2or(me, ZM.lastEKing, 9999));
        RobotInfo c = ZM.cat;
        put(c == null ? 9999 : ZG.catDist4(c.location, me)); put(c == null ? 99 : ZG.chebToBody(c.location, me));
        putBit(c != null && catLane(c, me));
        boolean catAtKing = false;
        if (c != null) for (int i = 0; i < ZC.numKings; i++) if (ZC.kings[i].distanceSquaredTo(c.location) <= 36) catAtKing = true;
        putBit(catAtKing);
        put(c == null ? 0 : c.direction.dx); put(c == null ? 0 : c.direction.dy);
        put(ZR.since(ZM.catSeenRound)); put(d2or(me, ZM.catSeen, 9999));
        // cheese
        putBit(ZM.cheeseT != null); put(d2or(me, ZM.cheeseT, 9999));
        // endgame quantities
        put(2000 - r);
        int bar = Math.min(400, Math.max(60, 4 * (2000 - r - 55) + 10));
        put(ZG.global - bar);
        put(ZG.global - Math.max(60, 4 * (2000 - r - 55) + 10));
        // birth facts and identity (babies)
        put(ZM.home == null ? 0 : ZM.home.x - me.x); put(ZM.home == null ? 0 : ZM.home.y - me.y);
        put(ZM.fu); put(ZM.fv); putBit(ZM.homeHalf(me)); put(ZR.since(ZM.birthRound)); put(ZM.babyIdx); putBit(ZM.scoutBit);
        put(ZM.birthRound & 1);
        // registers (values at the start of the turn) and their counters
        oneHot(ZR.curRole, ZR.ROLES);
        put(ZR.curMode); put(ZR.since(ZR.modeSince));
        put(ZR.since(Math.max(Math.max(ZR.modeLast[31], ZR.modeLast[33]), ZR.modeLast[34])));
        put(ZR.since(ZR.modeLast[18])); put(ZR.modeCount[18] > 0 ? 1 : 0);
        put(ZM.myMineCount - ZR.myMinesAtSeek);
        oneHot(ZR.curGoalKind, ZR.GOALS);
        put(d2or(me, ZR.curGoal, 9999)); put(ZR.since(ZR.goalSince)); put(ZR.goalStall);
        putBit(ZR.curMine != null); put(d2or(me, ZR.curMine, 9999)); put(ZR.since(ZR.mineSince)); put(ZR.mineChanges);
        put(ZC.numMines - ZR.minesAtSet); put(ZR.idle); put(ZR.since(ZM.lastPick));
        putBit(ZR.curExpl != null); put(d2or(me, ZR.curExpl, 9999)); put(ZR.since(ZR.explSince));
        putBit(ZR.curExpl != null && ZM.known(ZR.curExpl.x, ZR.curExpl.y));
        putBit(ZR.curCash != null); putBit(ZR.curCashLead); put(ZR.curCash == null ? 99 : ZG.cheb(me, ZR.curCash));
        put(ZR.curCash == null ? 0 : r - ZR.curCashStart); put(ZR.since(ZR.cashHeard)); put(Math.max(0, ZR.cashBanUntil - r));
        put(ZR.since(ZM.anchorSince));
        int cn = 0, heavy = 0, csum = 0;
        for (RobotInfo q : ZM.robots) {
            if (q.team != ZG.us || q.type != UnitType.BABY_RAT || ZG.cheb(q.location, me) > 4) continue;
            cn++; csum += q.cheeseAmount;
            if (q.cheeseAmount > ZG.raw || q.cheeseAmount == ZG.raw && q.ID < ZG.id) heavy++;
        }
        put(cn); put(heavy); put(csum);
        // king
        put(ZC.slot); putBit(ZC.slot == 0); put(ZC.rate100()); put(ZC.surplus());
        putBit(ZR.curHome != null); put(d2or(me, ZR.curHome, 9999)); put(ZR.curHome == null ? 99 : ZG.cheb(me, ZR.curHome));
        putBit(ZR.curForm != null); put(ZR.since(ZR.formOpen)); put(ZR.since(ZR.homeSince));
        MapLocation st = ZM.start;
        put(st == null ? 0 : st.x - me.x); put(st == null ? 0 : st.y - me.y);
        putBit(ZG.W * ZG.H <= 441); put(ringArcs(me));
        int threat = 0;
        for (RobotInfo q : ZM.robots) if (q.team == ZG.them && q.type != UnitType.RAT_KING && q.location.distanceSquaredTo(me) <= 18) threat++;
        put(threat);
        int[] out = new int[sn];
        System.arraycopy(sb, 0, out, 0, sn);
        return out;
    }

    /** Allied babies seen at the start of the turn strictly nearer the site than p (zdp Baby.nearerAllies). */
    static int nearerAllies(MapLocation site, MapLocation p) {
        int d = p.distanceSquaredTo(site), n = 0;
        for (RobotInfo q : ZM.robots) if (q.team == ZG.us && q.type == UnitType.BABY_RAT && q.location.distanceSquaredTo(site) < d) n++;
        return n;
    }

    // ------------------------------------------------------------------ local grid L

    /** Egocentric 9x9 window, CH channels, channel-major: [ch][dy][dx]. */
    public static int[] grid() {
        int[] g = new int[NL];
        MapLocation me = ZG.me;
        int A = SIDE * SIDE;
        for (int dy = -R; dy <= R; dy++)
            for (int dx = -R; dx <= R; dx++) {
                int x = me.x + dx, y = me.y + dy, cell = (dx + R) + SIDE * (dy + R);
                if (!ZG.onMap(x, y)) { g[cell] = 1; continue; }   // off the map reads as wall
                int i = x + ZG.W * y, m = ZM.mem[i];
                if ((m & ZM.WALL) != 0) g[cell] = 1;
                if ((m & ZM.DIRT) != 0) g[A + cell] = 1;
                if ((m & ZM.SEEN) == 0) g[2 * A + cell] = 1;
                if ((m & ZM.MINE) != 0) g[3 * A + cell] = 1;
                if (ZM.seenAt[i] > 0 && ZG.round + 1 - ZM.seenAt[i] <= 30) g[4 * A + cell] = Math.min(ZM.cheese[i], 100);
                if ((m & ZM.RTRAP) != 0) g[10 * A + cell] = 1;
                if ((m & ZM.CTRAP) != 0) g[10 * A + cell] = 2;
                if (ZM.risky(x, y)) g[11 * A + cell] = 1;
                if (ZM.isLane(x, y)) g[12 * A + cell] = 1;
            }
        for (RobotInfo q : ZM.robots) {
            int ch;
            if (q.team == Team.NEUTRAL) ch = 9;
            else if (q.team == ZG.us) ch = q.type == UnitType.RAT_KING ? 6 : 5;
            else ch = q.type == UnitType.RAT_KING ? 8 : 7;
            int v = q.team == ZG.them ? Math.max(1, q.health) : 1;
            int lo = q.type == UnitType.RAT_KING ? -1 : 0, hi = q.type == UnitType.RAT_KING ? 1 : q.team == Team.NEUTRAL ? 1 : 0;
            for (int ox = lo; ox <= hi; ox++)
                for (int oy = lo; oy <= hi; oy++) {
                    int dx = q.location.x + ox - me.x, dy = q.location.y + oy - me.y;
                    if (dx < -R || dx > R || dy < -R || dy > R) continue;
                    g[ch * A + (dx + R) + SIDE * (dy + R)] = v;
                }
        }
        return g;
    }

    // ------------------------------------------------------------------ step options

    public static int[] ctx = new int[NCB];

    /** Per-turn step bookkeeping (what has been done this turn), kept by the caller. */
    public static int doneMoved, doneActed, doneTurned, doneSqueaked, donePicks, doneSteps, doneAlert;
    public static MapLocation turnStartLoc;

    public static void resetDone() {
        doneMoved = doneActed = doneTurned = doneSqueaked = donePicks = doneSteps = doneAlert = 0;
        turnStartLoc = ZG.rc.getLocation();
    }

    /** Note an executed primitive (player executor and teacher post-hook). */
    public static void noteDone(int op) {
        doneSteps++;
        switch (op) {
            case OP_MOVE: case OP_MOVE_FWD: doneMoved = 1; break;
            case OP_TURN: doneTurned = 1; break;
            case OP_SQUEAK: doneSqueaked = 1; break;
            case OP_PICKUP: donePicks++; ZM.lastPick = ZG.round; ZM.picks++; break;
            case OP_WRITE: doneAlert = 1; break;
            case OP_END: break;
            default: doneActed = 1;
        }
    }

    /** Write the register part of a step context (baby 0..61, king 0..1) from the values chosen this turn. */
    public static void ctxRegisters(int[] c, boolean king) {
        if (!king) {
            for (int i = 0; i < 62; i++) c[i] = 0;
            c[ZR.nxtRole] = 1;
            c[4 + ZR.nxtMode] = 1;
            c[44 + ZR.nxtGoalKind] = 1;
            c[58] = ZR.nxtMine != null ? 1 : 0; c[59] = ZR.nxtExpl != null ? 1 : 0; c[60] = ZR.nxtCash != null ? 1 : 0;
            c[61] = ZR.nxtCashLead ? 1 : 0;
            return;
        }
        c[0] = ZR.nxtHome != null ? 1 : 0; c[1] = ZR.nxtForm != null ? 1 : 0;
    }

    /** The step context: registers chosen this turn plus what has been done (live). */
    public static int[] context() {
        RobotController rc = ZG.rc;
        MapLocation here = rc.getLocation();
        Direction f = rc.getDirection();
        RobotInfo carry = rc.getCarrying();
        if (!ZG.king) {
            int[] c = new int[NCB];
            ctxRegisters(c, false);
            c[62] = doneMoved; c[63] = doneActed; c[64] = doneTurned; c[65] = doneSqueaked; c[66] = donePicks; c[67] = doneSteps;
            c[68] = rc.isActionReady() ? 1 : 0; c[69] = rc.isMovementReady() ? 1 : 0; c[70] = rc.isTurningReady() ? 1 : 0;
            c[71] = f == null ? 0 : f.dx; c[72] = f == null ? 0 : f.dy;
            c[73] = here.x - turnStartLoc.x; c[74] = here.y - turnStartLoc.y;
            c[75] = carry == null ? 0 : carry.team == ZG.us ? 1 : 2;
            c[76] = rc.getRawCheese();
            return c;
        }
        int[] c = new int[NCK];
        ctxRegisters(c, true);
        c[2] = doneMoved; c[3] = doneActed; c[4] = doneTurned; c[5] = doneAlert; c[6] = donePicks; c[7] = doneSteps;
        c[8] = rc.isActionReady() ? 1 : 0; c[9] = rc.isMovementReady() ? 1 : 0; c[10] = rc.isTurningReady() ? 1 : 0;
        c[11] = f == null ? 0 : f.dx; c[12] = f == null ? 0 : f.dy;
        c[13] = here.x - turnStartLoc.x; c[14] = here.y - turnStartLoc.y;
        c[15] = rc.getGlobalCheese();
        return c;
    }

    /** Build the step options for the robot's current (live) state; goal columns 4-6 are left for fillGoal. */
    public static void stepOptions() throws GameActionException {
        if (ZG.king) kingStep(); else babyStep();
    }

    static void babyStep() throws GameActionException {
        Opts o = step;
        o.clear();
        RobotController rc = ZG.rc;
        MapLocation here = rc.getLocation();
        Direction f = rc.getDirection();
        RobotInfo[] robots = rc.senseNearbyRobots(-1, null);
        MapLocation k = ZC.nearestKing(here);
        RobotInfo cat = ZM.cat, en = ZM.nearEnemy;
        int idx;
        o.add(K_END, OP_END, -1, here.x, here.y, 0, here.x, here.y);
        // moves
        idx = 0;
        for (int d = 0; d < 8; d++) {
            Direction dir = ZG.DIRS[d];
            if (!rc.canMove(dir)) continue;
            MapLocation t = here.add(dir);
            int i = o.add(K_MOVE, OP_MOVE, d, t.x, t.y, 0, t.x, t.y);
            int[] x = o.x[i];
            dirCols(x, here, t, f, d);
            x[7] = k == null ? 0 : cap(t.distanceSquaredTo(k) - here.distanceSquaredTo(k), 999);
            x[8] = cat == null ? 0 : cap(ZG.catDist4(cat.location, t) - ZG.catDist4(cat.location, here), 999);
            x[9] = en == null ? 0 : cap(t.distanceSquaredTo(en.location) - here.distanceSquaredTo(en.location), 999);
            x[10] = catLane(cat, t) ? 1 : 0;
            x[11] = ZM.risky(t.x, t.y) ? 1 : 0;
            x[12] = ZM.dirt(t.x, t.y) ? 1 : 0;
            x[14] = dir == f ? 1 : 0;
            x[15] = en == null ? 9 : ZG.cheb(t, en.location);
            x[16] = k != null && inTransferReach(t, k) ? 1 : 0;
            MapLocation fs = ZC.formSite();
            x[17] = fs == null ? 0 : ZG.cheb(t, fs) - ZG.cheb(here, fs);
            x[18] = en != null && ZG.sees(en, t) ? 1 : 0;
            x[19] = idx++;
        }
        // turns
        if (rc.isTurningReady()) {
            idx = 0;
            RobotInfo carry = rc.getCarrying();
            Direction toE = en == null ? null : here.directionTo(en.location);
            Direction toK = k == null ? null : here.directionTo(k);
            Direction toC = cat == null ? null : here.directionTo(cat.location.translate(cat.location.x >= here.x ? 0 : 1, cat.location.y >= here.y ? 0 : 1));
            Direction toCh = ZM.cheeseT == null ? null : here.directionTo(ZM.cheeseT);
            for (int d = 0; d < 8; d++) {
                Direction dir = ZG.DIRS[d];
                if (dir == f || !rc.canTurn(dir)) continue;
                int i = o.add(K_TURN, OP_TURN, d, here.x + ZG.DX[d], here.y + ZG.DY[d], 0, here.x, here.y);
                int[] x = o.x[i];
                x[0] = ZG.DX[d]; x[1] = ZG.DY[d]; x[3] = ZG.rot(f, dir); x[13] = ZG.hand(here, dir);
                x[12] = carry != null && carry.team != ZG.us ? throwScore(here, d, robots) : 0;
                x[15] = dir == toE ? 1 : 0; x[16] = dir == toK ? 1 : 0; x[17] = dir == toC ? 1 : 0; x[18] = dir == toCh ? 1 : 0;
                x[19] = idx++;
            }
        }
        if (rc.isActionReady()) {
            biteOptions(o, here, f, robots, 2);
            // carry
            idx = 0;
            if (rc.getCarrying() == null)
                for (RobotInfo q : robots) {
                    if (q.type != UnitType.BABY_RAT || here.distanceSquaredTo(q.location) > 2 || !rc.canCarryRat(q.location)) continue;
                    int i = o.add(K_CARRY, OP_CARRY, -1, q.location.x, q.location.y, 0, q.location.x, q.location.y);
                    int[] x = o.x[i];
                    tileCols(x, here, q.location, f);
                    x[15] = q.health; x[16] = q.team == ZG.them ? 1 : 0; x[17] = q.cheeseAmount; x[19] = idx++;
                }
            RobotInfo carry = rc.getCarrying();
            if (carry != null) {
                idx = 0;
                for (int d = 0; d < 8; d++) {
                    Direction dir = ZG.DIRS[d];
                    if (!rc.canDropRat(dir)) continue;
                    MapLocation t = here.add(dir);
                    int i = o.add(K_DROP, OP_DROP, d, t.x, t.y, 0, t.x, t.y);
                    dirCols(o.x[i], here, t, f, d);
                    o.x[i][15] = carry.team == ZG.us ? 1 : 0; o.x[i][19] = idx++;
                }
                if (rc.canThrowRat()) {
                    int i = o.add(K_THROW, OP_THROW, ZG.di(f), here.x, here.y, 0, here.x, here.y);
                    int[] x = o.x[i];
                    int fd = ZG.di(f);
                    x[15] = fd < 0 ? 0 : throwScore(here, fd, robots); x[16] = ZR.since(ZM.carryStart); x[17] = carry.team == ZG.them ? 1 : 0;
                    x[13] = fd < 0 ? 0 : ZG.hand(here, f);
                }
            }
            trapAndDirtOptions(o, here, f, cat, en, 2);
            // transfer to a king
            int raw = rc.getRawCheese();
            if (raw > 0) {
                idx = 0;
                for (RobotInfo q : robots) {
                    if (q.team != ZG.us || q.type != UnitType.RAT_KING || here.distanceSquaredTo(q.location) > 18) continue;
                    for (int dx = -1; dx <= 1; dx++)
                        for (int dy = -1; dy <= 1; dy++) {
                            MapLocation t = q.location.translate(dx, dy);
                            if (!rc.canTransferCheese(t, raw)) continue;
                            int i = o.add(K_XFER, OP_TRANSFER, -1, t.x, t.y, raw, t.x, t.y);
                            tileCols(o.x[i], here, t, f);
                            o.x[i][15] = raw; o.x[i][16] = dx == 0 && dy == 0 ? 1 : 0; o.x[i][19] = idx++;
                        }
                }
            }
        }
        pickOptions(o, here, f, false);
        if (rc.canBecomeRatKing()) {
            int i = o.add(K_KING, OP_BECOME_KING, -1, here.x, here.y, 0, here.x, here.y);
            MapLocation fs = ZC.formSite();
            o.x[i][15] = fs != null && fs.equals(here) ? 1 : 0;
            o.x[i][16] = ZR.nxtCash != null && ZR.nxtCash.equals(here) ? 1 : 0;
            o.x[i][17] = ZM.nAlly;
        }
        if (doneSqueaked == 0) squeakOptions(o, here, k);
    }

    /** Columns 0-3 and 13 for a step from here to t along d. */
    static void dirCols(int[] x, MapLocation here, MapLocation t, Direction f, int d) {
        x[0] = ZG.DX[d]; x[1] = ZG.DY[d]; x[2] = here.distanceSquaredTo(t); x[3] = ZG.rot(f, ZG.DIRS[d]);
        x[13] = ZG.hand(here, ZG.DIRS[d]);
    }

    /** Columns 0-3 and 13 for an action on tile t. */
    static void tileCols(int[] x, MapLocation here, MapLocation t, Direction f) {
        x[0] = t.x - here.x; x[1] = t.y - here.y; x[2] = here.distanceSquaredTo(t);
        x[3] = t.equals(here) ? 0 : ZG.rot(f, here.directionTo(t));
        x[13] = ZG.pickBit(here, t) == 1 ? 1 : -1;
    }

    static void biteOptions(Opts o, MapLocation here, Direction f, RobotInfo[] robots, int reach) throws GameActionException {
        RobotController rc = ZG.rc;
        int idx = 0, g = rc.getGlobalCheese();
        int lim = ZG.king ? 2 : 1;
        for (RobotInfo q : robots) {
            if (q.team == ZG.us) continue;
            int lo = q.type == UnitType.RAT_KING ? -1 : 0;
            int hi = q.type == UnitType.RAT_KING ? 1 : q.team == Team.NEUTRAL ? 1 : 0;
            int type = q.team == Team.NEUTRAL ? 3 : q.type == UnitType.RAT_KING ? 2 : 1;
            for (int ox = lo; ox <= hi; ox++)
                for (int oy = lo; oy <= hi; oy++) {
                    MapLocation t = q.location.translate(ox, oy);
                    if (ZG.cheb(here, t) > lim + 1 || !rc.canAttack(t)) continue;
                    int kill = q.health > 10 ? (q.health - 10) * (q.health - 10) : 0;
                    int[] cs = {0, 1, kill};
                    for (int ci = 0; ci < 3; ci++) {
                        int c = cs[ci];
                        if (ci > 0 && (c <= 0 || (ci == 2 && c == 1) || c > g || !rc.canAttack(t, c))) continue;
                        int i = o.add(K_BITE, OP_ATTACK, -1, t.x, t.y, c, t.x, t.y);
                        if (i < 0) return;
                        int[] x = o.x[i];
                        tileCols(x, here, t, f);
                        x[15] = q.health; x[16] = type; x[17] = c; x[14] = ox == 0 && oy == 0 ? 1 : 0;
                        x[18] = q.team == ZG.them && q.type == UnitType.BABY_RAT && ZG.sees(q, here) ? 1 : 0;
                        x[19] = idx++;
                    }
                }
        }
    }

    static void pickOptions(Opts o, MapLocation here, Direction f, boolean king) throws GameActionException {
        RobotController rc = ZG.rc;
        int idx = 0, lim = king ? 1 : 1;
        for (int dx = -lim; dx <= lim; dx++)
            for (int dy = -lim; dy <= lim; dy++) {
                MapLocation t = here.translate(dx, dy);
                if (!rc.canPickUpCheese(t)) continue;
                int i = o.add(K_PICK, OP_PICKUP, -1, t.x, t.y, 0, t.x, t.y);
                if (i < 0) return;
                int[] x = o.x[i];
                tileCols(x, here, t, f);
                int amt = rc.canSenseLocation(t) ? rc.senseMapInfo(t).getCheeseAmount() : 0;
                x[15] = amt; x[16] = dx == 0 && dy == 0 ? 1 : 0; x[17] = rc.getRawCheese(); x[19] = idx++;
            }
    }

    static void trapAndDirtOptions(Opts o, MapLocation here, Direction f, RobotInfo cat, RobotInfo en, int r2) throws GameActionException {
        RobotController rc = ZG.rc;
        int lim = ZG.king ? 2 : 1;
        int ic = 0, ir = 0, id = 0, ip = 0;
        MapLocation n1 = null, n2 = null, n3 = null;
        if (cat != null && cat.direction != null && cat.direction != Direction.CENTER) {
            n1 = cat.location.add(cat.direction); n2 = n1.add(cat.direction); n3 = n2.add(cat.direction);
        }
        for (int dx = -lim; dx <= lim; dx++)
            for (int dy = -lim; dy <= lim; dy++) {
                if (dx == 0 && dy == 0) continue;
                MapLocation t = here.translate(dx, dy);
                if (!ZG.onMap(t.x, t.y)) continue;
                if (rc.canPlaceCatTrap(t)) {
                    int i = o.add(K_CTRAP, OP_PLACE_CAT_TRAP, -1, t.x, t.y, 0, t.x, t.y);
                    if (i < 0) return;
                    int[] x = o.x[i];
                    tileCols(x, here, t, f);
                    int s = 99;
                    if (cat != null) {
                        s = ZG.chebToBody(cat.location, t);
                        if (n1 != null) s = Math.min(s, Math.min(ZG.chebToBody(n1, t), Math.min(ZG.chebToBody(n2, t), ZG.chebToBody(n3, t))));
                    }
                    x[15] = s; x[16] = cat == null ? 9999 : ZG.catDist4(cat.location, t);
                    x[17] = ZM.isLane(t.x, t.y) ? Math.min(ZM.laneAge(t.x, t.y), 999) : -1;
                    x[18] = 2 * Math.abs(2 * t.x - ZG.cx2) + 2 * Math.abs(2 * t.y - ZG.cy2);
                    x[19] = ic++;
                }
                if (rc.canPlaceRatTrap(t)) {
                    int i = o.add(K_RTRAP, OP_PLACE_RAT_TRAP, -1, t.x, t.y, 0, t.x, t.y);
                    if (i < 0) return;
                    tileCols(o.x[i], here, t, f);
                    o.x[i][15] = en == null ? 9999 : t.distanceSquaredTo(en.location); o.x[i][19] = ir++;
                }
                if (rc.canRemoveDirt(t)) {
                    int i = o.add(K_DIG, OP_REMOVE_DIRT, -1, t.x, t.y, 0, t.x, t.y);
                    if (i < 0) return;
                    tileCols(o.x[i], here, t, f);
                    o.x[i][15] = rc.getGlobalCheese() + rc.getRawCheese();
                    o.x[i][16] = Math.max(Math.abs(dx), Math.abs(dy)) == 2 ? 1 : 0;
                    o.x[i][17] = ZG.king && ZR.nxtHome != null ? (t.equals(kingDigTile(here, ZR.nxtHome)) ? 1 : 0) : 0;
                    o.x[i][19] = id++;
                }
                if (rc.canPlaceDirt(t)) {
                    int i = o.add(K_DIRT, OP_PLACE_DIRT, -1, t.x, t.y, 0, t.x, t.y);
                    if (i < 0) return;
                    tileCols(o.x[i], here, t, f);
                    o.x[i][15] = rc.getDirt(); o.x[i][19] = ip++;
                }
            }
    }

    static void squeakOptions(Opts o, MapLocation here, MapLocation k) {
        boolean near = k != null && here.distanceSquaredTo(k) <= 16;
        int nm = ZM.myMineCount;
        for (int j = 0; j < nm; j++) {
            int key = ZM.myMines[j], mx = key / 64, my = key % 64;
            int i = o.add(K_SQ_MINE, OP_SQUEAK, -1, mx, my, ZC.squeak(ZC.SQ_MINE, mx, my, 0), here.x, here.y);
            if (i < 0) return;
            int[] x = o.x[i];
            x[0] = mx - here.x; x[1] = my - here.y; x[2] = here.distanceSquaredTo(new MapLocation(mx, my));
            x[15] = ZC.hasMine(mx, my) ? 1 : 0; x[16] = near ? 1 : 0; x[17] = ((j - ZG.round % nm) % nm + nm) % nm;
            x[18] = ZC.numMines == 0 ? 1 : 0; x[19] = j;
        }
        {
            int i = o.add(K_SQ_SYM, OP_SQUEAK, -1, 0, 0, ZC.squeak(ZC.SQ_SYM, 0, 0, ZM.mask), here.x, here.y);
            if (i < 0) return;
            int sh = ZC.symSlot;
            o.x[i][15] = sh != 0 && (ZM.mask & sh) != sh ? 1 : 0; o.x[i][16] = near ? 1 : 0; o.x[i][17] = ZM.mask;
        }
        if (ZM.catSeen != null) {
            MapLocation c = ZM.catSeen;
            int i = o.add(K_SQ_CAT, OP_SQUEAK, -1, c.x, c.y, ZC.squeak(ZC.SQ_CAT, c.x, c.y, 0), here.x, here.y);
            if (i < 0) return;
            o.x[i][0] = c.x - here.x; o.x[i][1] = c.y - here.y; o.x[i][2] = here.distanceSquaredTo(c);
            o.x[i][15] = ZR.since(ZM.catSeenRound); o.x[i][16] = near ? 1 : 0; o.x[i][17] = ZM.cat != null && catEarshot(ZM.cat.location, here) ? 1 : 0;
        }
        if (ZR.nxtCash != null) {
            MapLocation c = ZR.nxtCash;
            int e = ZC.cashTag(c.x, c.y) << 10 | (ZR.nxtCashStart & 1023);
            int i = o.add(K_SQ_CASH, OP_SQUEAK, -1, c.x, c.y, ZC.squeak(ZC.SQ_CASH, c.x, c.y, e), here.x, here.y);
            if (i < 0) return;
            o.x[i][0] = c.x - here.x; o.x[i][1] = c.y - here.y; o.x[i][2] = ZG.cheb(here, c);
            o.x[i][15] = ZR.nxtCashLead ? 1 : 0; o.x[i][16] = ZG.round - ZR.nxtCashStart;
            o.x[i][17] = ZM.cat != null && catEarshot(ZM.cat.location, here) ? 1 : 0;
        }
        if (ZM.lastEKing != null) {
            MapLocation c = ZM.lastEKing;
            int i = o.add(K_SQ_EKING, OP_SQUEAK, -1, c.x, c.y, ZC.squeak(ZC.SQ_EKING, c.x, c.y, (ZM.lastEKingRound & 1023) << 2), here.x, here.y);
            if (i < 0) return;
            o.x[i][0] = c.x - here.x; o.x[i][1] = c.y - here.y; o.x[i][2] = here.distanceSquaredTo(c);
            o.x[i][15] = ZR.since(ZM.lastEKingRound); o.x[i][16] = near ? 1 : 0;
        }
    }

    // ---- king steps

    /** zdp King.digTile: the first dirt tile of the step toward t that adds the fewest dirt tiles, or null. */
    static MapLocation kingDigTile(MapLocation c, MapLocation t) {
        Direction to = c.directionTo(t);
        if (to == Direction.CENTER) return null;
        Direction[] cand = {to, to.rotateLeft(), to.rotateRight()};
        MapLocation best = null;
        int bn = Integer.MAX_VALUE;
        for (Direction d : cand) {
            MapLocation nc = c.add(d);
            if (ZG.cheb(nc, t) > ZG.cheb(c, t)) continue;
            int n = 0;
            MapLocation first = null;
            boolean ok = true;
            for (int dx = -1; dx <= 1 && ok; dx++)
                for (int dy = -1; dy <= 1; dy++) {
                    int x = nc.x + dx, y = nc.y + dy;
                    if (Math.abs(x - c.x) <= 1 && Math.abs(y - c.y) <= 1) continue;
                    if (!ZG.onMap(x, y)) { ok = false; break; }
                    int m = ZM.mem[x + ZG.W * y];
                    if ((m & ZM.SEEN) == 0 || (m & ZM.WALL) != 0) { ok = false; break; }
                    if ((m & ZM.DIRT) != 0) { n++; if (first == null) first = new MapLocation(x, y); }
                }
            if (ok && n > 0 && n < bn) { bn = n; best = first; }
        }
        return best;
    }

    /** Dirt tiles a king step along d adds to its 3x3 (-1 when one is off-map, unseen or wall). */
    static int stepDirt(MapLocation c, Direction d) {
        MapLocation nc = c.add(d);
        int n = 0;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                int x = nc.x + dx, y = nc.y + dy;
                if (Math.abs(x - c.x) <= 1 && Math.abs(y - c.y) <= 1) continue;
                if (!ZG.onMap(x, y)) return -1;
                int m = ZM.mem[x + ZG.W * y];
                if ((m & ZM.SEEN) == 0 || (m & ZM.WALL) != 0) return -1;
                if ((m & ZM.DIRT) != 0) n++;
            }
        return n;
    }

    /** Nearest cheese pile within Chebyshev 4 of the anchor (home mine, else start) and outside the 3x3 (zdp
     *  King.nearCheese), from the start-of-turn memory. */
    public static MapLocation kingPile(MapLocation me) {
        MapLocation anchor = ZR.nxtHome != null ? ZR.nxtHome : ZM.start;
        if (anchor == null) return null;
        MapLocation best = null;
        int bd = Integer.MAX_VALUE;
        for (int dx = -5; dx <= 5; dx++)
            for (int dy = -5; dy <= 5; dy++) {
                int x = me.x + dx, y = me.y + dy;
                if (!ZG.onMap(x, y)) continue;
                int i = x + ZG.W * y;
                if (ZM.seenAt[i] != ZG.round + 1 || ZM.cheese[i] <= 0) continue;
                MapLocation l = new MapLocation(x, y);
                if (ZG.cheb(l, anchor) > 4 || ZG.cheb(l, me) <= 1) continue;
                int d = me.distanceSquaredTo(l);
                if (d < bd) { bd = d; best = l; }
            }
        return best;
    }

    static void kingStep() throws GameActionException {
        Opts o = step;
        o.clear();
        RobotController rc = ZG.rc;
        MapLocation here = rc.getLocation();
        Direction f = rc.getDirection();
        RobotInfo[] robots = rc.senseNearbyRobots(-1, null);
        MapLocation home = ZR.nxtHome, pile = kingPile(here);
        RobotInfo cat0 = null, en0 = null;
        int nEn = 0, nCat = 0;
        for (RobotInfo q : robots) {
            if (q.team == Team.NEUTRAL) { if (cat0 == null) cat0 = q; nCat++; }
            else if (q.team == ZG.them) { if (en0 == null) en0 = q; nEn++; }
        }
        int arcsHere = ringArcs(here);
        o.add(K_END, OP_END, -1, here.x, here.y, 0, here.x, here.y);
        int idx = 0;
        Direction toHome = home == null ? null : here.directionTo(home), toPile = pile == null ? null : here.directionTo(pile);
        Direction away = cat0 == null ? null : cat0.location.directionTo(here);
        for (int d = 0; d < 8; d++) {
            Direction dir = ZG.DIRS[d];
            if (!rc.canMove(dir)) continue;
            MapLocation t = here.add(dir);
            int i = o.add(K_MOVE, OP_MOVE, d, t.x, t.y, 0, t.x, t.y);
            int[] x = o.x[i];
            dirCols(x, here, t, f, d);
            x[7] = home == null ? 0 : cap(t.distanceSquaredTo(home) - here.distanceSquaredTo(home), 999);
            x[8] = cat0 == null ? 0 : cap(t.distanceSquaredTo(cat0.location) - here.distanceSquaredTo(cat0.location), 999);
            x[9] = pile == null ? 0 : cap(t.distanceSquaredTo(pile) - here.distanceSquaredTo(pile), 999);
            int arcs = ringArcs(t);
            x[10] = arcs; x[11] = arcsHere;
            x[12] = stepDirt(here, dir);
            x[14] = home == null ? 0 : ZG.cheb(t, home) - ZG.cheb(here, home);
            x[15] = fanRank(toHome, dir); x[16] = fanRank(toPile, dir); x[17] = fanRank(away, dir);
            x[18] = arcs <= 1 || arcs <= arcsHere ? 1 : 0;
            x[19] = idx++;
        }
        if (rc.isTurningReady()) {
            idx = 0;
            for (int d = 0; d < 8; d++) {
                Direction dir = ZG.DIRS[d];
                if (dir == f || !rc.canTurn(dir)) continue;
                int i = o.add(K_TURN, OP_TURN, d, here.x + ZG.DX[d], here.y + ZG.DY[d], 0, here.x, here.y);
                int[] x = o.x[i];
                x[0] = ZG.DX[d]; x[1] = ZG.DY[d]; x[3] = ZG.rot(f, dir); x[13] = ZG.hand(here, dir);
                x[15] = toHome == dir ? 1 : 0; x[16] = en0 != null && here.directionTo(en0.location) == dir ? 1 : 0;
                x[19] = idx++;
            }
        }
        if (rc.isActionReady()) {
            MapLocation goal = !ZG.coop && en0 != null ? en0.location : new MapLocation(ZG.W / 2, ZG.H / 2);
            idx = 0;
            for (int dx = -2; dx <= 2; dx++)
                for (int dy = -2; dy <= 2; dy++) {
                    if (Math.abs(dx) < 2 && Math.abs(dy) < 2) continue;
                    MapLocation t = here.translate(dx, dy);
                    if (!rc.canBuildRat(t)) continue;
                    int i = o.add(K_BUILD, OP_BUILD, -1, t.x, t.y, 0, t.x, t.y);
                    int[] x = o.x[i];
                    tileCols(x, here, t, f);
                    x[15] = t.distanceSquaredTo(goal); x[16] = cat0 == null ? 0 : t.distanceSquaredTo(cat0.location);
                    x[17] = Math.abs(dx) == 2 && Math.abs(dy) == 2 ? 1 : 0; x[18] = t.distanceSquaredTo(new MapLocation(ZG.W / 2, ZG.H / 2));
                    x[14] = en0 == null ? 0 : t.distanceSquaredTo(en0.location);
                    x[19] = idx++;
                }
            biteOptions(o, here, f, robots, 8);
            trapAndDirtOptions(o, here, f, cat0, ZM.nearEnemy, 8);
        }
        pickOptions(o, here, f, true);
        if (doneAlert == 0) {
            int threat = 0;
            for (RobotInfo q : robots) if (q.team == ZG.them && q.type != UnitType.RAT_KING && q.location.distanceSquaredTo(ZG.me) <= 18) threat++;
            int i = o.add(K_ALERT, OP_WRITE, -1, here.x, here.y, threat, here.x, here.y);
            int[] x = o.x[i];
            boolean thisRound = ZC.read(ZC.ALERT + 2) == ZC.stamp();
            x[15] = threat; x[16] = thisRound ? ZC.read(ZC.ALERT) >> 6 : -1; x[17] = nEn; x[18] = nCat;
        }
    }

    /** Fill goal columns 4-6 of the options (dest = ex, ey) from where the robot stood (hx, hy). */
    public static void fillGoal(int n, int[] kind, int[][] x, int[] ex, int[] ey, int[] dir, int hx, int hy, MapLocation goal) {
        if (goal == null) return;
        MapLocation here = new MapLocation(hx, hy);
        int[] fld = ZF.field(goal, false);
        int ph = ZF.at(fld, hx, hy), d0 = here.distanceSquaredTo(goal), c0 = ZG.cheb(here, goal);
        Direction toG = here.directionTo(goal);
        for (int i = 0; i < n; i++) {
            int k = kind[i];
            if (k == K_END) continue;
            if (k == K_TURN) { x[i][6] = dir[i] >= 0 && ZG.DIRS[dir[i]] == toG ? 1 : 0; continue; }
            MapLocation t = new MapLocation(ex[i], ey[i]);
            x[i][4] = cap(t.distanceSquaredTo(goal) - d0, 999);
            int pt = ZF.at(fld, ex[i], ey[i]);
            x[i][5] = pt >= ZF.INF || ph >= ZF.INF ? (pt >= ZF.INF ? 64 : -64) : cap(pt - ph, 64);
            x[i][6] = ZG.cheb(t, goal) - c0;
        }
    }

    /** The goal the step head's goal columns use: the GOAL register (babies), the home mine (kings). */
    public static MapLocation stepGoal() { return ZG.king ? ZR.nxtHome : ZR.nxtGoal; }

    // ------------------------------------------------------------------ register options

    /** Exploration samples: kind (2 explore, 3 home, 4 wander) and tile. The player draws them from zdn's RNG
     *  (fillSamples); the teacher first adds zdb's own draws of the turn, then tops up the same way. */
    public static final int[] smpK = new int[40], smpX = new int[40], smpY = new int[40];
    public static int nSmp;

    public static void addSample(int k, int x, int y) {
        if (nSmp < smpK.length) { smpK[nSmp] = k; smpX[nSmp] = x; smpY[nSmp++] = y; }
    }

    static int countSamples(int k) { int c = 0; for (int i = 0; i < nSmp; i++) if (smpK[i] == k) c++; return c; }

    /** One uniform draw in the side frame (zdp Baby.explore / exploreHome). */
    static void frameDraw(int k) {
        int ax = ZG.rand(ZG.W), ay = ZG.rand(ZG.H);
        addSample(k, ZM.fu > 0 ? ax : ZG.W - 1 - ax, ZM.fv > 0 ? ay : ZG.H - 1 - ay);
    }

    /** Top the samples up to 12 explore, 12 home and (with a mine chosen) 1 wander draw. */
    public static void fillSamples() {
        for (int k = countSamples(2); k < 12; k++) frameDraw(2);
        for (int k = countSamples(3); k < 12; k++) frameDraw(3);
        MapLocation m = ZR.nxtMine;
        if (m != null && countSamples(4) == 0) {
            int ax = ZG.rand(5) - 2, ay = ZG.rand(5) - 2;
            addSample(4, ZG.clampX(m.x + ZM.fu * ax), ZG.clampY(m.y + ZM.fv * ay));
        }
    }

    static int pathCap(int[] f, int x, int y) { int v = ZF.at(f, x, y); return v >= ZF.INF ? 999 : Math.min(v, 999); }

    static int nearKingD2(MapLocation p) { MapLocation k = ZC.nearestKing(p); return k == null ? 9999 : p.distanceSquaredTo(k); }

    static boolean is(MapLocation a, int x, int y) { return a != null && a.x == x && a.y == y; }

    /** Options of register r, given the values already chosen for the registers before it (ZR.nxt*). Uses
     *  start-of-turn perception only (ZG.me, ZM, ZC), so the teacher can build them at the end of zdb's turn. */
    public static Opts regOptions(int r) {
        return ZG.king ? kingReg(r) : babyReg(r);
    }

    /** Context of register r: one-hots of the option kinds chosen for registers 0..r-1, zero beyond. */
    public static int regCtxLen(boolean king) {
        int[] k = king ? REG_K_KING : REG_K_BABY;
        int n = 0;
        for (int v : k) n += v;
        return n;
    }

    static Opts babyReg(int r) {
        MapLocation me = ZG.me;
        int round = ZG.round;
        Opts o = regB[r];
        o.clear();
        switch (r) {
            case ZR.R_ROLE:
                for (int k = 0; k < ZR.ROLES; k++) {
                    int i = o.add(k, -1, -1, 0, 0, 0, 0, 0);
                    o.x[i][0] = ZR.curRole == k ? 1 : 0;
                    o.x[i][1] = ZR.curRole == k ? ZR.since(ZR.roleSince) : 0;
                }
                break;
            case ZR.R_MODE:
                for (int k = 0; k < ZR.MODES; k++) {
                    int i = o.add(k, -1, -1, 0, 0, 0, 0, 0);
                    o.x[i][0] = ZR.curMode == k ? 1 : 0;
                    o.x[i][1] = ZR.since(ZR.modeLast[k]);
                    o.x[i][2] = Math.min(ZR.modeCount[k], 999);
                }
                break;
            case ZR.R_MINE: {
                o.add(0, -1, -1, 0, 0, 0, 0, 0);
                // zdb's candidate list and order (pickMine): team list then own mines, home half first, by mineKey
                int n = 0;
                MapLocation[] cand = mineCand;
                for (int i = 0; i < ZC.numMines && n < cand.length; i++) n = addCand(cand, n, ZC.mines[i]);
                for (int i = 0; i < ZM.myMineCount && n < cand.length; i++) n = addCand(cand, n, new MapLocation(ZM.myMines[i] / 64, ZM.myMines[i] % 64));
                if (n == 0) break;
                MapLocation[] ord = mineOrd;
                System.arraycopy(cand, 0, ord, 0, n);
                MapLocation h = ZM.home != null ? ZM.home : me;
                int ours = 0;
                for (int i = 0; i < n; i++) if (ZM.homeHalf(ord[i])) { MapLocation t = ord[ours]; ord[ours++] = ord[i]; ord[i] = t; }
                int m = ours > 0 ? ours : n;
                long[] key = new long[m];
                for (int i = 0; i < m; i++) key[i] = ((long) ord[i].distanceSquaredTo(h) << 16) + ((ord[i].x - h.x) * ZM.fu + 64) * 128 + (ord[i].y - h.y) * ZM.fv + 64;
                for (int i = 1; i < m; i++) {
                    long k = key[i];
                    MapLocation c = ord[i];
                    int j = i;
                    for (; j > 0 && k < key[j - 1]; j--) { key[j] = key[j - 1]; ord[j] = ord[j - 1]; }
                    key[j] = k; ord[j] = c;
                }
                int[] fld = ZF.field(me, true);
                int sm = ZM.slot % m;
                for (int c = 0; c < n; c++) {
                    MapLocation l = cand[c];
                    int rank = -1;
                    for (int j = 0; j < m; j++) if (ord[j].equals(l)) { rank = j; break; }
                    int i = o.add(1, -1, -1, l.x, l.y, 0, l.x, l.y);
                    if (i < 0) break;
                    int[] x = o.x[i];
                    x[0] = l.x - me.x; x[1] = l.y - me.y; x[2] = me.distanceSquaredTo(l); x[3] = pathCap(fld, l.x, l.y);
                    x[4] = l.distanceSquaredTo(h); x[5] = ZM.homeHalf(l) ? 1 : 0;
                    boolean cur = l.equals(ZR.curMine);
                    x[6] = cur ? 1 : 0; x[7] = ZC.hasMine(l.x, l.y) ? 1 : 0; x[8] = c >= ZC.numMines || mineMine(l) ? 1 : 0;
                    x[9] = rank; x[10] = m; x[11] = rank < 0 ? 99 : ((rank - sm) % m + m) % m;
                    x[12] = nearKingD2(l); x[13] = cur ? ZR.since(ZR.mineSince) : 0;
                }
                break;
            }
            case ZR.R_EXPL: {
                o.add(0, -1, -1, 0, 0, 0, 0, 0);
                int[] fld = ZF.field(me, true);
                if (ZR.curExpl != null) {
                    MapLocation e = ZR.curExpl;
                    int i = o.add(1, -1, -1, e.x, e.y, 0, e.x, e.y);
                    explRow(o.x[i], e.x, e.y, fld);
                    o.x[i][9] = ZR.since(ZR.explSince);
                }
                int[] seen = new int[5];
                for (int s = 0; s < nSmp; s++) {
                    int i = o.add(smpK[s], -1, -1, smpX[s], smpY[s], 0, smpX[s], smpY[s]);
                    if (i < 0) break;
                    explRow(o.x[i], smpX[s], smpY[s], fld);
                    o.x[i][9] = seen[smpK[s]]++;
                }
                break;
            }
            case ZR.R_CASH: {
                o.add(0, -1, -1, 0, 0, 0, 0, 0);
                if (ZR.curCash != null) {
                    MapLocation c = ZR.curCash;
                    int i = o.add(1, -1, -1, c.x, c.y, 0, c.x, c.y);
                    cashRow(o.x[i], c.x, c.y);
                    o.x[i][3] = round - ZR.curCashStart; o.x[i][6] = 1; o.x[i][7] = ZR.curCashLead ? 1 : 0;
                }
                for (int j = 0; j < ZC.nCash; j++) {
                    MapLocation c = ZC.cashSite[j];
                    int i = o.add(2, -1, -1, c.x, c.y, ZC.cashStart[j], c.x, c.y);
                    cashRow(o.x[i], c.x, c.y);
                    int rank = 0;
                    for (int q = 0; q < ZC.nCash; q++) {
                        MapLocation d = ZC.cashSite[q];
                        if (ZC.cashStart[q] < ZC.cashStart[j] || ZC.cashStart[q] == ZC.cashStart[j] && d.x * 64 + d.y < c.x * 64 + c.y) rank++;
                    }
                    o.x[i][3] = round - ZC.cashStart[j]; o.x[i][5] = j; o.x[i][6] = c.equals(ZR.curCash) ? 1 : 0; o.x[i][7] = rank;
                    o.x[i][8] = ZC.cashAgeOf[j];
                }
                for (int d = -1; d < 8; d++) {
                    int x = me.x + (d < 0 ? 0 : ZG.DX[d]), y = me.y + (d < 0 ? 0 : ZG.DY[d]);
                    if (!ZG.onMap(x, y)) continue;
                    int i = o.add(3, -1, -1, x, y, ZG.round, x, y);
                    cashRow(o.x[i], x, y);
                    o.x[i][5] = d + 1; o.x[i][6] = is(ZR.curCash, x, y) ? 1 : 0;
                    o.x[i][8] = d < 0 ? -1 : ZG.rot(ZG.facing, ZG.DIRS[d]);
                }
                break;
            }
            case ZR.R_GOAL:
                goalOptions(o);
                break;
            default:
        }
        return o;
    }

    static final MapLocation[] mineCand = new MapLocation[40], mineOrd = new MapLocation[40];

    static int addCand(MapLocation[] cand, int n, MapLocation l) {
        for (int i = 0; i < n; i++) if (cand[i].equals(l)) return n;
        cand[n] = l;
        return n + 1;
    }

    static boolean mineMine(MapLocation l) {
        int key = l.x * 64 + l.y;
        for (int i = 0; i < ZM.myMineCount; i++) if (ZM.myMines[i] == key) return true;
        return false;
    }

    static void explRow(int[] x, int tx, int ty, int[] fld) {
        MapLocation me = ZG.me, t = new MapLocation(tx, ty);
        x[0] = tx - me.x; x[1] = ty - me.y; x[2] = me.distanceSquaredTo(t); x[3] = ZG.cheb(me, t);
        x[4] = Math.max(Math.abs(tx - (ZG.W - 1) / 2), Math.abs(ty - (ZG.H - 1) / 2));
        x[5] = ZM.known(tx, ty) ? 1 : 0; x[6] = ZM.homeHalf(t) ? 1 : 0;
        x[7] = ZR.nxtMine == null ? 99 : ZG.cheb(t, ZR.nxtMine); x[8] = pathCap(fld, tx, ty);
    }

    /** zdp Baby.cashClear from start-of-turn perception: the 3x3 on the map, no wall or dirt, no cat or king on a
     *  tile seen this turn, no remembered king body elsewhere. */
    static boolean cashClear(int sx, int sy) {
        if (sx < 1 || sy < 1 || sx > ZG.W - 2 || sy > ZG.H - 2) return false;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                int x = sx + dx, y = sy + dy, i = x + ZG.W * y, m = ZM.mem[i];
                if ((m & (ZM.WALL | ZM.DIRT)) != 0) return false;
                if (ZM.seenAt[i] == ZG.round + 1) {
                    RobotInfo r = robotAt(ZM.robots, x, y);
                    if (r != null && r.type != UnitType.BABY_RAT) return false;
                } else if ((m & ZM.KING) != 0) return false;
            }
        return true;
    }

    static void cashRow(int[] x, int tx, int ty) {
        MapLocation me = ZG.me, t = new MapLocation(tx, ty);
        x[0] = tx - me.x; x[1] = ty - me.y; x[2] = ZG.cheb(me, t); x[4] = cashClear(tx, ty) ? 1 : 0;
        int kc = 99;
        for (int k = 0; k < ZC.numKings; k++) kc = Math.min(kc, ZG.cheb(ZC.kings[k], t));
        x[9] = kc;
    }

    static MapLocation centreTile() {
        MapLocation h = ZM.home != null ? ZM.home : ZG.me;
        return new MapLocation((ZG.cx2 + (2 * h.x < ZG.cx2 ? 0 : 1)) / 2, (ZG.cy2 + (2 * h.y < ZG.cy2 ? 0 : 1)) / 2);
    }

    static int goalRow(Opts o, int kind, MapLocation t, int age, int idx, int flag, int extra, int[] fld) {
        if (t == null || !ZG.onMap(t.x, t.y)) return -1;
        int i = o.add(kind, -1, -1, t.x, t.y, 0, t.x, t.y);
        if (i < 0) return -1;
        MapLocation me = ZG.me;
        int[] x = o.x[i];
        x[0] = t.x - me.x; x[1] = t.y - me.y; x[2] = me.distanceSquaredTo(t); x[3] = ZG.cheb(me, t);
        x[4] = pathCap(fld, t.x, t.y);
        boolean cur = t.equals(ZR.curGoal);
        x[5] = cur ? 1 : 0; x[6] = cur ? ZR.since(ZR.goalSince) : 0; x[7] = cur ? Math.min(ZR.goalStall, 999) : 0;
        x[8] = Math.min(age, 999); x[9] = nearKingD2(t); x[10] = ZM.known(t.x, t.y) ? 1 : 0;
        x[11] = flag; x[12] = idx; x[13] = extra;
        return i;
    }

    static void goalOptions(Opts o) {
        MapLocation me = ZG.me;
        int[] fld = ZF.field(me, true);
        o.add(ZR.G_NONE, -1, -1, 0, 0, 0, 0, 0);
        MapLocation nk = ZC.nearestKing(me);
        for (int k = 0; k < ZC.numKings; k++) goalRow(o, ZR.G_KING, ZC.kings[k], 0, k, ZC.kings[k].equals(nk) ? 1 : 0, ZC.kingSlot[k], fld);
        goalRow(o, ZR.G_MINE, ZR.nxtMine, 0, 0, 0, 0, fld);
        goalRow(o, ZR.G_CHEESE, ZM.cheeseT, 0, 0, 0, 0, fld);
        RobotInfo e = ZM.nearEnemy;
        if (e != null) goalRow(o, ZR.G_ENEMY, e.location, 0, 0, e.health < ZG.rc.getHealth() ? 1 : 0, e.health, fld);
        if (ZM.eKing != null) goalRow(o, ZR.G_EKING, ZM.eKing.location, 0, 0, 1, ZM.eKing.health, fld);
        if (ZM.lastEKing != null) goalRow(o, ZR.G_EKING, ZM.lastEKing, ZG.round - ZM.lastEKingRound, 1, 0, 0, fld);
        if (ZC.eking != null) goalRow(o, ZR.G_EKING, ZC.eking, ZC.ekingAge, 2, 0, 0, fld);
        if (ZM.home != null)
            for (int b = 0; b < 3; b++) {
                int s = 1 << b;
                if ((ZM.mask & s) == 0) continue;
                MapLocation c = ZG.mirror(ZM.home, s);
                goalRow(o, ZR.G_RUSH, c, ZG.round - ZM.spotEmpty[b], b, me.distanceSquaredTo(c) <= 13 ? 1 : 0, ZG.pickBit(me, c), fld);
            }
        RobotInfo cat = ZM.cat;
        if (cat != null) {
            int idx = 0;
            for (int dx = -1; dx <= 2; dx++)
                for (int dy = -1; dy <= 2; dy++) {
                    if (dx >= 0 && dx <= 1 && dy >= 0 && dy <= 1) continue;
                    MapLocation t = cat.location.translate(dx, dy);
                    if (!ZG.onMap(t.x, t.y) || (ZM.mem[t.x + ZG.W * t.y] & ZM.WALL) != 0) { idx++; continue; }
                    int fl = (catLane(cat, t) ? 1 : 0) | (ZM.dirt(t.x, t.y) ? 2 : 0);
                    goalRow(o, ZR.G_CATRING, t, 0, idx++, fl, ZG.pickBit(me, t), fld);
                }
        }
        if (ZM.catSeen != null) goalRow(o, ZR.G_CAT, ZM.catSeen, ZG.round - ZM.catSeenRound, 0, 1, 0, fld);
        for (int c = 0; c < 2; c++) if (ZC.cats[c] != null) goalRow(o, ZR.G_CAT, ZC.cats[c], ZC.catAge[c], c + 1, 0, 0, fld);
        if (ZC.alert != null) goalRow(o, ZR.G_ALERT, ZC.alert, ZC.alertAge, 0, 0, ZC.alertThreat, fld);
        if (ZC.form != null) goalRow(o, ZR.G_FORM, ZC.form, ZC.formAge, 0, nearerAllies(ZC.form, me), 0, fld);
        goalRow(o, ZR.G_CASH, ZR.nxtCash, ZR.nxtCash == null ? 0 : ZG.round - ZR.nxtCashStart, 0, ZR.nxtCashLead ? 1 : 0, 0, fld);
        goalRow(o, ZR.G_EXPL, ZR.nxtExpl, 0, 0, 0, 0, fld);
        goalRow(o, ZR.G_CENTRE, centreTile(), 0, 0, 0, 0, fld);
    }

    // ---- king registers

    static MapLocation enemyStart() {
        MapLocation st = ZM.start != null ? ZM.start : ZG.me;
        return ZG.mirror(st, ZG.firstSym(ZM.mask));
    }

    /** zdp King.clear3 over memory. */
    static boolean clear3(MapLocation c) {
        if (c.x < 1 || c.y < 1 || c.x > ZG.W - 2 || c.y > ZG.H - 2) return false;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                int m = ZM.mem[(c.x + dx) + ZG.W * (c.y + dy)];
                if ((m & ZM.SEEN) == 0 || (m & (ZM.WALL | ZM.DIRT)) != 0) return false;
            }
        return true;
    }

    /** zdp King.clearNow from start-of-turn perception: 3x3 on the map, seen this turn, passable, no cat or king. */
    static boolean clearNow(MapLocation s) {
        if (s.x < 1 || s.y < 1 || s.x > ZG.W - 2 || s.y > ZG.H - 2) return false;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                int x = s.x + dx, y = s.y + dy, i = x + ZG.W * y;
                if (ZM.seenAt[i] != ZG.round + 1 || (ZM.mem[i] & (ZM.WALL | ZM.DIRT)) != 0) return false;
                RobotInfo r = robotAt(ZM.robots, x, y);
                if (r != null && r.type != UnitType.BABY_RAT) return false;
            }
        return true;
    }

    static boolean heldByOther(MapLocation m, int r) {
        for (int k = 0; k < ZC.numKings; k++) if (ZG.cheb(ZC.kings[k], m) <= r && !ZC.kings[k].equals(ZG.me)) return true;
        return false;
    }

    static final int[][] RING_SITES = {{3, 0}, {0, 3}, {3, 1}, {1, 3}, {3, -1}, {-1, 3}, {0, -3}, {-3, 0}, {1, -3}, {-3, 1}, {-1, -3}, {-3, -1}};

    static Opts kingReg(int r) {
        MapLocation me = ZG.me, st = ZM.start != null ? ZM.start : me, ek = enemyStart();
        Opts o = regK[r];
        o.clear();
        o.add(0, -1, -1, 0, 0, 0, 0, 0);
        if (r == ZR.R_HOME) {
            for (int j = 0; j < ZC.numMines; j++) {
                MapLocation m = ZC.mines[j];
                int i = o.add(1, -1, -1, m.x, m.y, 0, m.x, m.y);
                int[] x = o.x[i];
                x[0] = m.x - me.x; x[1] = m.y - me.y; x[2] = st.distanceSquaredTo(m); x[3] = m.distanceSquaredTo(ek);
                x[4] = heldByOther(m, 2) ? 1 : 0; x[5] = m.equals(ZR.curHome) ? 1 : 0; x[6] = me.distanceSquaredTo(m);
                x[7] = j; x[8] = ZG.cheb(me, m); x[9] = m.equals(ZR.curHome) ? ZR.since(ZR.homeSince) : 0;
            }
            return o;
        }
        // FORM: 1 KEEP, 2 mine site, 3 back spot, 4 ring site
        if (ZR.curForm != null) formRow(o, 1, ZR.curForm, 0, me, st, ek);
        for (int j = 0; j < ZC.numMines; j++) formRow(o, 2, ZC.mines[j], j, me, st, ek);
        Direction back = ek.directionTo(me);
        formRow(o, 3, me.translate(4 * back.dx, 4 * back.dy), 0, me, st, ek);
        int sx = 2 * me.x >= ZG.W - 1 ? 1 : -1, sy = 2 * me.y >= ZG.H - 1 ? 1 : -1;
        for (int j = 0; j < RING_SITES.length; j++) formRow(o, 4, me.translate(RING_SITES[j][0] * sx, RING_SITES[j][1] * sy), j, me, st, ek);
        return o;
    }

    static void formRow(Opts o, int kind, MapLocation s, int idx, MapLocation me, MapLocation st, MapLocation ek) {
        int i = o.add(kind, -1, -1, s.x, s.y, 0, s.x, s.y);
        if (i < 0) return;
        int[] x = o.x[i];
        x[0] = s.x - me.x; x[1] = s.y - me.y; x[2] = me.distanceSquaredTo(s); x[3] = s.distanceSquaredTo(ek);
        x[4] = ZG.sat(4 * s.distanceSquaredTo(st) - s.distanceSquaredTo(ek)); x[5] = clear3(s) ? 1 : 0; x[6] = clearNow(s) ? 1 : 0;
        x[7] = heldByOther(s, 4) ? 1 : 0; x[8] = ZG.cheb(me, s); x[9] = idx;
        int n = 0;
        for (RobotInfo q : ZM.robots) if (q.team == ZG.us && q.type == UnitType.BABY_RAT && ZG.cheb(q.location, s) <= 2) n++;
        x[10] = n; x[11] = kind == 1 ? ZR.since(ZR.formOpen) : 0;
    }
}
