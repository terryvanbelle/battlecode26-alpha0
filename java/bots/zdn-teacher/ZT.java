package zdn;

import battlecode.common.*;

/**
 * zdn teacher recorder (docs/zdn.md, "Teacher records"). It runs zdn's perception beside zdb (on the real
 * RobotController), sees every primitive zdb calls through RcLog, and writes zdn records of those choices:
 *   - every turn: perception, register bookkeeping with zdb's intent variables as the register values, done flags;
 *   - sampled turns: the REG record (S, L, each register's options and zdb's value among them) and one STEP record
 *     per primitive (the legal options just before the call, the call as the label) plus END.
 * Built only into the teacher package (scripts/build_zdn.py), where zdb's classes share this package.
 */
public final class ZT {
    public static final int MOVE_FWD = ZO.OP_MOVE_FWD, MOVE = ZO.OP_MOVE, TURN = ZO.OP_TURN, BUILD = ZO.OP_BUILD,
        BECOME_KING = ZO.OP_BECOME_KING, PLACE_DIRT = ZO.OP_PLACE_DIRT, REMOVE_DIRT = ZO.OP_REMOVE_DIRT,
        PLACE_RAT_TRAP = ZO.OP_PLACE_RAT_TRAP, REMOVE_RAT_TRAP = ZO.OP_REMOVE_RAT_TRAP,
        PLACE_CAT_TRAP = ZO.OP_PLACE_CAT_TRAP, REMOVE_CAT_TRAP = ZO.OP_REMOVE_CAT_TRAP, PICKUP = ZO.OP_PICKUP,
        ATTACK = ZO.OP_ATTACK, WRITE = ZO.OP_WRITE, TRANSFER = ZO.OP_TRANSFER, THROW = ZO.OP_THROW, DROP = ZO.OP_DROP,
        CARRY = ZO.OP_CARRY, DISINTEGRATE = ZO.OP_DISINTEGRATE, RESIGN = ZO.OP_RESIGN, SQUEAK = ZO.OP_SQUEAK;

    /** Sampled turns per thousand (set by the build). */
    static final int BABY_PER_MILLE = 15, KING_PER_MILLE = 100;
    static final int MAXB = 24;

    static RobotController real;
    static boolean on, sampled, king;
    static int[] S, L;
    static MapLocation goal;
    static int mode;

    static final ZO.Opts[] bOpts = new ZO.Opts[MAXB];
    static final int[][] bCtx = new int[MAXB][];
    static final int[] bLab = new int[MAXB], bHx = new int[MAXB], bHy = new int[MAXB];
    static int nb;

    static boolean pend, pendBuf;
    static int pOp, pG0;
    static MapLocation pLoc;

    public static void init(RobotController rc) {
        real = rc;
        ZG.init(rc);
        ZM.init();
    }

    static boolean pick() {
        int h = ZG.id * 0x9E3779B1 + ZG.round * 0x85EBCA6B;
        h ^= h >>> 15; h *= 0x2C1B3C6D; h ^= h >>> 12; h *= 0x297A2D39; h ^= h >>> 15;
        return ((h >>> 1) % 1000) < (king ? KING_PER_MILLE : BABY_PER_MILLE);
    }

    /** Start of zdb's turn (after G.startTurn, before Baby/King.turn). */
    public static void start() {
        on = false;
        try {
            ZG.refresh();
            king = ZG.king;
            if (king) ZC.slot = King.slot;
            ZM.perceive();
            if (king) ZC.econStart();
            ZR.begin();
            ZO.nSmp = 0;
            goal = null;
            mode = -1;
            nb = 0;
            pend = pendBuf = false;
            ZO.resetDone();
            sampled = pick();
            if (sampled) { S = ZO.state(); L = ZO.grid(); }
            on = true;
        } catch (Throwable e) { ZE.error(ZE.W_START, e); }
    }

    /** zdb's Nav.goTo target (build patch at Nav.goTo's start). */
    public static void goal(MapLocation t) { if (t != null) goal = t; }

    /** zdb's movement branch of this turn (build patch after moveDecision; -1 when no branch was taken). */
    public static void mode(int lab) { if (lab >= 0 && lab < ZR.MODES - 1) mode = lab; }

    /** One of zdb's exploration draws (build patches in explore, exploreHome, wanderAround). */
    public static void sample(int kind, int x, int y) { if (on) ZO.addSample(kind, x, y); }

    /** zdb's intent variables as zdn register values. */
    static void readIntent() {
        if (king) {
            ZR.nxtHome = King.homeMine;
            ZR.nxtForm = King.formSite;
            return;
        }
        int r = Baby.role;
        ZR.nxtRole = r == Baby.FIGHT ? ZR.FIGHT : r == Baby.ATTACK ? ZR.ATTACK : Baby.scout ? ZR.SCOUT : ZR.GATHER;
        if (mode >= 0) ZR.nxtMode = mode;
        ZR.nxtMine = Baby.mineTarget;
        ZR.nxtExpl = Baby.exploreTarget;
        ZR.nxtCash = Baby.cashSite;
        ZR.nxtCashLead = Baby.cashSite != null && Baby.cashLead;
        ZR.nxtCashStart = Baby.cashStart;
        ZR.nxtGoal = goal;
    }

    /** Before a state-changing call of zdb's (RcLog). */
    public static void pre(int op, Direction d, MapLocation loc, int i0, int i1) {
        pend = pendBuf = false;
        if (!on) return;
        try {
            // protocol writes are mechanical (the alert's stamp write is the alert step; FORM is a register)
            if (op == WRITE && i0 != ZC.ALERT + 2) return;
            if (op == DISINTEGRATE || op == RESIGN) return;
            pend = true;
            pOp = op;
            pLoc = loc;
            pG0 = real.getGlobalCheese();
            if (!sampled || nb >= MAXB) return;
            readIntent();
            ZO.stepOptions();
            ZO.Opts o = ZO.step;
            if (bOpts[nb] == null) bOpts[nb] = new ZO.Opts(ZO.DS, 200);
            bOpts[nb].copyFrom(o);
            bCtx[nb] = ZO.context();
            bLab[nb] = match(o, op, d, loc, i0);
            MapLocation h = real.getLocation();
            bHx[nb] = h.x;
            bHy[nb] = h.y;
            pendBuf = true;
        } catch (Throwable e) { ZE.error(ZE.W_PRE, e); pendBuf = false; }
    }

    /** After the call returned normally. */
    public static void post() {
        if (!on || !pend) return;
        pend = false;
        try {
            if (pOp == PICKUP && king) ZC.econReceived(real.getGlobalCheese() - pG0);
            if (pOp == REMOVE_DIRT && pLoc != null) { ZM.mem[pLoc.x + ZG.W * pLoc.y] &= ~ZM.DIRT; ZF.invalidate(); }
            if (pOp == PLACE_DIRT && pLoc != null) { ZM.mem[pLoc.x + ZG.W * pLoc.y] |= ZM.DIRT; ZF.invalidate(); }
            ZO.noteDone(pOp == MOVE_FWD ? ZO.OP_MOVE : pOp);
            if (pendBuf) { nb++; pendBuf = false; }
        } catch (Throwable e) { ZE.error(ZE.W_POST, e); }
    }

    static int match(ZO.Opts o, int op, Direction d, MapLocation loc, int i0) {
        MapLocation h = real.getLocation();
        switch (op) {
            case MOVE_FWD: d = real.getDirection();
            // fall through
            case MOVE: return d == null ? -1 : o.find(ZO.K_MOVE, h.x + d.dx, h.y + d.dy);
            case TURN: return d == null ? -1 : o.find(ZO.K_TURN, h.x + d.dx, h.y + d.dy);
            case DROP: return d == null ? -1 : o.find(ZO.K_DROP, h.x + d.dx, h.y + d.dy);
            case BUILD: return o.find(ZO.K_BUILD, loc.x, loc.y);
            case BECOME_KING: return o.first(ZO.K_KING);
            case PLACE_DIRT: return o.find(ZO.K_DIRT, loc.x, loc.y);
            case REMOVE_DIRT: return o.find(ZO.K_DIG, loc.x, loc.y);
            case PLACE_RAT_TRAP: return o.find(ZO.K_RTRAP, loc.x, loc.y);
            case PLACE_CAT_TRAP: return o.find(ZO.K_CTRAP, loc.x, loc.y);
            case PICKUP: return o.find(ZO.K_PICK, loc.x, loc.y);
            case TRANSFER: return o.find(ZO.K_XFER, loc.x, loc.y);
            case CARRY: return o.find(ZO.K_CARRY, loc.x, loc.y);
            case THROW: return o.first(ZO.K_THROW);
            case WRITE: return o.first(ZO.K_ALERT);
            case ATTACK: {
                int a = i0 < 0 ? 0 : i0;
                for (int i = 0; i < o.n; i++) if (o.kind[i] == ZO.K_BITE && o.lx[i] == loc.x && o.ly[i] == loc.y && o.amt[i] == a) return i;
                return -1;
            }
            case SQUEAK:
                switch (ZC.sqType(i0)) {
                    case ZC.SQ_MINE: return o.find(ZO.K_SQ_MINE, ZC.sqX(i0), ZC.sqY(i0));
                    case ZC.SQ_SYM: return o.first(ZO.K_SQ_SYM);
                    case ZC.SQ_CAT: return o.first(ZO.K_SQ_CAT);
                    case ZC.SQ_CASH: return o.first(ZO.K_SQ_CASH);
                    case ZC.SQ_EKING: return o.first(ZO.K_SQ_EKING);
                    default: return -1;
                }
            default: return -1;
        }
    }

    /** zdb's value of register r among its options. */
    static int regLabel(int r, ZO.Opts o) {
        if (king) {
            if (r == ZR.R_HOME) return ZR.nxtHome == null ? o.first(0) : o.find(1, ZR.nxtHome.x, ZR.nxtHome.y);
            MapLocation f = ZR.nxtForm;
            if (f == null) return o.first(0);
            if (f.equals(ZR.curForm) && o.first(1) >= 0) return o.first(1);
            return o.findTile(2, f.x, f.y);
        }
        switch (r) {
            case ZR.R_ROLE: return o.first(ZR.nxtRole);
            case ZR.R_MODE: return o.first(ZR.nxtMode);
            case ZR.R_MINE: return ZR.nxtMine == null ? o.first(0) : o.find(1, ZR.nxtMine.x, ZR.nxtMine.y);
            case ZR.R_EXPL: {
                MapLocation e = ZR.nxtExpl;
                if (e == null) return o.first(0);
                if (e.equals(ZR.curExpl) && o.first(1) >= 0) return o.first(1);
                return o.findTile(2, e.x, e.y);
            }
            case ZR.R_CASH: {
                MapLocation c = ZR.nxtCash;
                if (c == null) return o.first(0);
                if (c.equals(ZR.curCash) && o.first(1) >= 0) return o.first(1);
                return o.find(ZR.nxtCashLead ? 3 : 2, c.x, c.y);
            }
            case ZR.R_GOAL: return ZR.nxtGoal == null ? o.first(0) : o.findTile(1, ZR.nxtGoal.x, ZR.nxtGoal.y);
            default: return -1;
        }
    }

    /** End of zdb's turn: registers, then (sampled turns) the records. */
    public static void end() {
        if (!on) return;
        on = false;
        try {
            if (king != (real.getType() == UnitType.RAT_KING)) { ZR.commit(); return; }   // promoted this turn
            readIntent();
            int nR = king ? ZR.NK : ZR.NB;
            int[] nk = king ? ZO.REG_K_KING : ZO.REG_K_BABY;
            ZO.Opts[] regs = king ? ZO.regK : ZO.regB;
            int[] lab = new int[nR];
            if (sampled) {
                for (int r = 0; r < nR; r++) {
                    if (!king && r == ZR.R_EXPL) ZO.fillSamples();
                    lab[r] = regLabel(r, ZO.regOptions(r));
                }
            } else if (!king) lab[ZR.R_GOAL] = regLabel(ZR.R_GOAL, ZO.regOptions(ZR.R_GOAL));
            if (!king) {
                ZO.Opts g = regs[ZR.R_GOAL];
                int li = lab[ZR.R_GOAL];
                ZR.nxtGoalKind = li >= 0 ? g.kind[li] : ZR.nxtGoal == null ? ZR.G_NONE : ZR.curGoalKind;
            }
            if (sampled) {
                ZE.reg(king, S, L, regs, nk, lab);
                // the END step, from the state zdb left
                if (nb < MAXB) {
                    ZO.stepOptions();
                    if (bOpts[nb] == null) bOpts[nb] = new ZO.Opts(ZO.DS, 200);
                    bOpts[nb].copyFrom(ZO.step);
                    bCtx[nb] = ZO.context();
                    bLab[nb] = ZO.step.first(ZO.K_END);
                    MapLocation h = real.getLocation();
                    bHx[nb] = h.x; bHy[nb] = h.y;
                    nb++;
                }
                MapLocation g = ZO.stepGoal();
                for (int i = 0; i < nb; i++) {
                    ZO.Opts o = bOpts[i];
                    ZO.fillGoal(o.n, o.kind, o.x, o.ex, o.ey, o.dir, bHx[i], bHy[i], g);
                    ZO.ctxRegisters(bCtx[i], king);
                    ZE.step(king, i, bCtx[i], o, ZO.NK, bLab[i]);
                }
            }
            if (king) ZC.econEnd();
            ZR.commit();
        } catch (Throwable e) { ZE.error(ZE.W_END, e); }
    }
}
