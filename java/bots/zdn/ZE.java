package zdn;

/**
 * zdn records (docs/zdn.md, "Records"; read by scripts/zdn_data.py), sent through the decision-record channel
 * (setIndicatorString with a leading U+DEC0; the engine hook writes decisions-v1).
 *
 *   REG  (200 baby, 210 king), label unused:  VER, nS, nL, nR, S[nS]*, L[nL], then per register r:
 *        K_r, lab_r, D_r, NK_r, K_r rows of (kind, D_r features*); then (VER 2) the trailer: net version, q_r per register
 *   STEP (201 baby, 211 king), label = chosen option:  VER, step, nC, D, K, NK, ctx[nC]*, K rows of (kind, D features*);
 *        then (VER 2) q
 *
 *   ERROR (299), label = where it happened (ZE.W_*): the exception's text (chars, at most 200)
 *
 * (* signed, stored + 32768.) lab 0xFFFF = no option matched. A record that would overflow the buffer is dropped.
 * q = round(-4096 log p) of the chosen option under the policy that chose it (capped at 65534; 0xFFFF = not sampled,
 * as in the teacher's records of zdb's choices); version = ZW.VERSION of the player's nets (0xFFFF: none).
 */
public final class ZE {
    public static final int VER = 2, P_REG_BABY = 200, P_STEP_BABY = 201, P_REG_KING = 210, P_STEP_KING = 211, NONE = 0xFFFF;
    /** Error records: where the exception was caught. */
    public static final int P_ERROR = 299, W_PLAYER = 1, W_START = 2, W_PRE = 3, W_POST = 4, W_END = 5;
    public static int errors;

    static final char[] b = new char[32000];
    static int n;
    static boolean over;
    public static int sent, dropped;

    static void begin(int point, int label) {
        b[0] = '\uDEC0';
        b[1] = (char) point;
        b[2] = (char) (label < 0 ? NONE : label);
        n = 4;
        over = false;
    }

    static void sv(int v) { if (n < b.length) b[n++] = (char) (ZG.sat(v) + 32768); else over = true; }

    static void uv(int v) { if (n < b.length) b[n++] = (char) Math.max(0, Math.min(65535, v)); else over = true; }

    static void send() {
        if (over || n - 4 > 65535) { dropped++; return; }
        b[3] = (char) (n - 4);
        ZG.rc.setIndicatorString(new String(b, 0, n));
        sent++;
    }

    static void rows(ZO.Opts o) {
        for (int i = 0; i < o.n; i++) {
            uv(o.kind[i]);
            int[] x = o.x[i];
            for (int d = 0; d < o.D; d++) sv(x[d]);
        }
    }

    /** Report a caught exception (the turn goes on; tests and collection logs count these). */
    public static void error(int where, Throwable e) {
        errors++;
        try {
            begin(P_ERROR, where);
            String t = String.valueOf(e);
            StackTraceElement[] st = e.getStackTrace();
            if (st != null && st.length > 0) t += " at " + st[0].getClassName() + "." + st[0].getMethodName() + ":" + st[0].getLineNumber();
            for (int i = 0; i < t.length() && i < 200; i++) uv(t.charAt(i));
            send();
        } catch (Throwable ignored) { }
    }

    /** q of a log-probability (see the class comment). */
    public static int q(double logp) { return (int) Math.min(65534, Math.round(-4096 * logp)); }

    /** The teacher's REG record (labels, not samples). */
    public static void reg(boolean king, int[] S, int[] L, ZO.Opts[] regs, int[] nk, int[] lab) {
        reg(king, S, L, regs, nk, lab, null, -1);
    }

    /** The teacher's STEP record. */
    public static void step(boolean king, int step, int[] ctx, ZO.Opts o, int nk, int label) {
        step(king, step, ctx, o, nk, label, NONE);
    }

    /** The REG record of a turn: inputs, every register's options and label, and the trailer (q per register: null
     *  when the labels were not sampled; version < 0: none). */
    public static void reg(boolean king, int[] S, int[] L, ZO.Opts[] regs, int[] nk, int[] lab, int[] q, int version) {
        begin(king ? P_REG_KING : P_REG_BABY, NONE);
        uv(VER); uv(S.length); uv(L.length); uv(regs.length);
        for (int v : S) sv(v);
        for (int v : L) uv(v);
        for (int r = 0; r < regs.length; r++) {
            ZO.Opts o = regs[r];
            uv(o.n); uv(lab[r] < 0 ? NONE : lab[r]); uv(o.D); uv(nk[r]);
            rows(o);
        }
        uv(version < 0 ? NONE : version);
        for (int r = 0; r < regs.length; r++) uv(q == null ? NONE : q[r]);
        send();
    }

    /** One STEP record: the options at this point of the turn, the one taken, and its q (NONE: not sampled). */
    public static void step(boolean king, int step, int[] ctx, ZO.Opts o, int nk, int label, int q) {
        begin(king ? P_STEP_KING : P_STEP_BABY, label);
        uv(VER); uv(step); uv(ctx.length); uv(o.D); uv(o.n); uv(nk);
        for (int v : ctx) sv(v);
        rows(o);
        uv(q);
        send();
    }
}
