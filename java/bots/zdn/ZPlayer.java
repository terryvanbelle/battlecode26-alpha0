package zdn;

import battlecode.common.*;

/**
 * zdn's turn (docs/zdn.md, "Turn"): perception, then the net writes its registers one by one, then picks primitives
 * from the legal options until it picks END. No if/then policy: every choice is a softmax over options; the code only
 * keeps the protocol's mechanical writes (king beacon, symmetry merge, mine list, cat relay, budget).
 */
public final class ZPlayer {
    static ZNet babyNet, kingNet;
    static boolean babyTried, kingTried;
    public static final int MAX_STEPS = 16;
    /** Recorded turns per mille, babies' and kings' (RL collection: REG/STEP records of a random sample of turns); 0
     *  in plain play. */
    public static int RECORD_PER_MILLE = 0;
    public static int RECORD_KING_PER_MILLE = 0;
    static int recRng;

    /** Whether to record this turn: a per-game stream (seeded like ZNet.draw), never the policy's own draws. */
    static boolean recordTurn() {
        int pm = ZG.king ? RECORD_KING_PER_MILLE : RECORD_PER_MILLE;
        if (pm <= 0) return false;
        if (pm >= 1000) return true;
        if (recRng == 0) { recRng = (ZG.id * 0x85EBCA6B + 0x1B873593) ^ (int) (long) (Math.random() * 4294967296.0); if (recRng == 0) recRng = 1; }
        int x = recRng;
        x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
        recRng = x;
        return (x >>> 1) % 1000 < pm;
    }

    public static void run(RobotController rc) {
        ZG.init(rc);
        ZM.init();
        while (true) {
            try {
                turn();
            } catch (Throwable e) {
                ZE.error(ZE.W_PLAYER, e);   // a bad turn must not kill the robot
            }
            Clock.yield();
        }
    }

    static ZNet net(boolean king) {
        try {
            if (king && !kingTried) { kingTried = true; kingNet = ZNet.of(ZW.KING); }
            if (!king && !babyTried) { babyTried = true; babyNet = ZNet.of(ZW.BABY); }
        } catch (Throwable e) { }
        return king ? kingNet : babyNet;
    }

    static float[] concat(int[] a, int[] b) {
        float[] x = new float[a.length + b.length];
        for (int i = 0; i < a.length; i++) x[i] = a[i];
        for (int i = 0; i < b.length; i++) x[a.length + i] = b[i];
        return x;
    }

    static float[] toF(int[] a) {
        float[] x = new float[a.length];
        for (int i = 0; i < a.length; i++) x[i] = a[i];
        return x;
    }

    /** log p of the last choose() (the record's q). */
    static double logp;

    /** Pick one option: the net's softmax at ZW.TEMPERATURE, or uniform when there is no net. */
    static int choose(ZNet net, ZNet.Scorer sc, float[] h, float[] ctx, ZO.Opts o) {
        logp = 0;
        if (o.n <= 1) return 0;
        if (net == null || sc == null) { logp = -Math.log(o.n); return ZNet.draw(o.n); }
        float[] z = ZNet.logits(sc, h, ctx, o.n, o.kind, o.x);
        int pick = ZNet.sample(z, o.n, ZW.TEMPERATURE);
        logp = ZNet.lastLogp;
        return pick;
    }

    static void turn() throws GameActionException {
        RobotController rc = ZG.rc;
        ZG.refresh();
        boolean king = ZG.king;
        ZM.perceive();
        if (king) kingPre();
        int[] S = ZO.state(), L = ZO.grid();
        ZNet net = net(king);
        float[] h = net == null ? null : net.trunk(concat(S, L));
        ZR.begin();
        ZO.nSmp = 0;
        boolean rec = recordTurn();

        // ---- registers
        int nR = king ? ZR.NK : ZR.NB;
        int[] nk = king ? ZO.REG_K_KING : ZO.REG_K_BABY;
        ZO.Opts[] regs = king ? ZO.regK : ZO.regB;
        float[] rctx = new float[ZO.regCtxLen(king)];
        int[] lab = new int[nR], q = new int[nR];
        int off = 0;
        for (int r = 0; r < nR; r++) {
            if (!king && r == ZR.R_EXPL) ZO.fillSamples();
            ZO.Opts o = ZO.regOptions(r);
            int pick = choose(net, net == null ? null : net.regs[r], h, rctx, o);
            lab[r] = pick;
            q[r] = ZE.q(logp);
            setRegister(king, r, o, pick);
            int k = o.kind[pick];
            if (k >= 0 && k < nk[r]) rctx[off + k] = 1;
            off += nk[r];
        }
        if (rec) ZE.reg(king, S, L, regs, nk, lab, q, ZW.VERSION);
        if (king) formWrites();

        // ---- primitive steps
        ZO.resetDone();
        UnitType type = rc.getType();
        MapLocation goal = ZO.stepGoal();
        for (int s = 0; s < MAX_STEPS; s++) {
            ZO.stepOptions();
            ZO.Opts o = ZO.step;
            MapLocation here = rc.getLocation();
            ZO.fillGoal(o.n, o.kind, o.x, o.ex, o.ey, o.dir, here.x, here.y, goal);
            int[] ctx = ZO.context();
            int pick = choose(net, net == null ? null : net.step, h, toF(ctx), o);
            if (rec) ZE.step(king, s, ctx, o, ZO.NK, pick, ZE.q(logp));
            if (o.kind[pick] == ZO.K_END) break;
            if (!exec(o, pick)) break;
            ZO.noteDone(o.op[pick]);
            if (rc.getType() != type) break;   // promoted: the king's first turn is next round
        }

        if (king) kingPost();
        ZR.commit();
    }

    /** Make option `pick` of register r the register's value for this turn. */
    static void setRegister(boolean king, int r, ZO.Opts o, int pick) {
        int k = o.kind[pick];
        MapLocation t = new MapLocation(o.lx[pick], o.ly[pick]);
        if (king) {
            if (r == ZR.R_HOME) ZR.nxtHome = k == 0 ? null : t;
            else if (k == 0) ZR.nxtForm = null;
            else if (k != 1) ZR.nxtForm = t;
            return;
        }
        switch (r) {
            case ZR.R_ROLE: ZR.nxtRole = k; break;
            case ZR.R_MODE: ZR.nxtMode = k; break;
            case ZR.R_MINE: ZR.nxtMine = k == 0 ? null : t; break;
            case ZR.R_EXPL: if (k == 0) ZR.nxtExpl = null; else if (k != 1) ZR.nxtExpl = t; break;
            case ZR.R_CASH:
                if (k == 0) { ZR.nxtCash = null; ZR.nxtCashLead = false; }
                else if (k != 1) { ZR.nxtCash = t; ZR.nxtCashStart = o.amt[pick]; ZR.nxtCashLead = k == 3; }
                break;
            case ZR.R_GOAL: ZR.nxtGoal = k == 0 ? null : t; ZR.nxtGoalKind = k; break;
            default:
        }
    }

    /** Run one primitive; false when the engine refused it. */
    static boolean exec(ZO.Opts o, int i) {
        RobotController rc = ZG.rc;
        MapLocation loc = new MapLocation(o.lx[i], o.ly[i]);
        try {
            switch (o.op[i]) {
                case ZO.OP_MOVE: rc.move(ZG.DIRS[o.dir[i]]); break;
                case ZO.OP_TURN: rc.turn(ZG.DIRS[o.dir[i]]); break;
                case ZO.OP_ATTACK: if (o.amt[i] > 0) rc.attack(loc, o.amt[i]); else rc.attack(loc); break;
                case ZO.OP_PICKUP: {
                    int g0 = rc.getGlobalCheese();
                    rc.pickUpCheese(loc);
                    if (ZG.king) ZC.econReceived(rc.getGlobalCheese() - g0);
                    break;
                }
                case ZO.OP_CARRY: rc.carryRat(loc); break;
                case ZO.OP_DROP: rc.dropRat(ZG.DIRS[o.dir[i]]); break;
                case ZO.OP_THROW: rc.throwRat(); break;
                case ZO.OP_PLACE_CAT_TRAP: rc.placeCatTrap(loc); break;
                case ZO.OP_PLACE_RAT_TRAP: rc.placeRatTrap(loc); break;
                case ZO.OP_REMOVE_DIRT: rc.removeDirt(loc); ZM.mem[loc.x + ZG.W * loc.y] &= ~ZM.DIRT; ZF.invalidate(); break;
                case ZO.OP_PLACE_DIRT: rc.placeDirt(loc); ZM.mem[loc.x + ZG.W * loc.y] |= ZM.DIRT; ZF.invalidate(); break;
                case ZO.OP_TRANSFER: rc.transferCheese(loc, o.amt[i]); break;
                case ZO.OP_BECOME_KING: rc.becomeRatKing(); break;
                case ZO.OP_SQUEAK: rc.squeak(o.amt[i]); break;
                case ZO.OP_BUILD: rc.buildRat(loc); break;
                case ZO.OP_WRITE: ZC.writeAlert(o.amt[i]); break;
                default: return false;
            }
            return true;
        } catch (GameActionException e) {
            return false;
        }
    }

    // ------------------------------------------------------------------ king protocol (mechanical)

    static void kingPre() {
        if (ZC.slot < 0) ZC.claimSlot();
        ZC.econStart();
        for (int i = 0; i < ZM.myMineCount; i++) ZC.addMine(ZM.myMines[i] / 64, ZM.myMines[i] % 64);
        ZC.kingRelay();
        ZC.publish();
    }

    static void kingPost() {
        ZG.me = ZG.rc.getLocation();
        ZC.publish();
        ZC.writeBudget();
        ZC.econEnd();
    }

    /** The FORM register is the formation slot's content: open on a new site, heartbeat while held, close on none. */
    static void formWrites() {
        if (ZR.nxtForm == null) { if (ZR.curForm != null) ZC.formClose(); return; }
        if (ZR.nxtForm.equals(ZR.curForm)) ZC.formHold();
        else ZC.formOpen(ZR.nxtForm);
    }
}
