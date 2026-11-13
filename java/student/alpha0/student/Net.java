package alpha0.student;

import java.io.*;
import java.util.Random;

/**
 * Student network inference (student-mlp-v3).
 * Inputs: dense = obs[0..DENSE) / 255; bits = obs[DENSE..) unpacked LSB-first; map bits = bits[0..B_SHARED),
 * other bits = bits[B_SHARED..BITS).
 *   m  = relu(Wm map + bm)                                  (map encoder; sparse over set bits)
 *   h1 = relu(W1 [dense; other; m] + b1)                    (sparse over nonzero inputs)
 *   h2 = relu(W2 h1 + b2)
 *   logits = Wp h2 + bp,  value = tanh(wv . h2 + bv)
 *   param head, given chosen action a:  q1 = relu(Wq h2 + E[a] + Wc c + bq),  q2 = relu(Wq2 q1 + bq2),
 *     bit logits = Wb q2 + bb (independent bits).  c = ctx(obs, a): binary copies of integers a value is often built
 *     from (own x, y, round, all cheese, the target slot's/register's current value, newest heard squeak, last-known
 *     enemy king x, y). The bits are still free, so every value stays reachable; c only makes copying easy.
 * File (big-endian): int32 magic 0x53544E33, dense, mapBits, otherBits, hm, h1, h2, hq, nActions, nParamBits, nCtx;
 * float32 WmT[mapBits][hm], bm[hm], W1T[dense+otherBits+hm][h1], b1[h1], W2[h2][h1], b2[h2], Wp[A][h2], bp[A],
 * wv[h2], bv, Wq[hq][h2], E[A][hq], bq[hq], Wc[hq][nCtx], Wq2[hq][hq], bq2[hq], Wb[nParamBits][hq], bb[nParamBits].
 */
public final class Net {
    public static final int MAGIC = 0x53544E33;
    /** Param context layout: bit widths of x, y, round, all cheese, current target value, newest squeak, enemy king x, y. */
    public static final int[] CTX_W = {6, 6, 11, 16, 10, 32, 6, 6};
    public static final int CTX = 93;
    final int dense, mapBits, otherBits, hm, h1, h2, hq, na, nq, nc, in1;
    final float[] wmt, bm, w1t, b1, w2, b2, wp, bp, wv, wq, emb, bq, wc, wq2, bq2, wb, bb;
    final float bv;
    private final float[] m, a1, a2, logits, q, q2;
    private final boolean[] cbits = new boolean[CTX];
    private byte[] lastX;

    public Net(File file) throws IOException {
        try (DataInputStream d = new DataInputStream(new BufferedInputStream(new FileInputStream(file), 1 << 16))) {
            if (d.readInt() != MAGIC) throw new IOException("bad weights magic (not student-mlp-v3)");
            dense = d.readInt(); mapBits = d.readInt(); otherBits = d.readInt(); hm = d.readInt();
            h1 = d.readInt(); h2 = d.readInt(); hq = d.readInt(); na = d.readInt(); nq = d.readInt(); nc = d.readInt();
            if (dense != Obs.DENSE || mapBits != Obs.B_SHARED || otherBits != Obs.BITS - Obs.B_SHARED || na != Actions.N
                || nq != Actions.PARAM_MAX_BITS || nc != CTX)
                throw new IOException("weights do not match obs/action version: dense=" + dense + " bits=" + mapBits + "+" + otherBits + " na=" + na);
            in1 = dense + otherBits + hm;
            wmt = read(d, mapBits * hm); bm = read(d, hm);
            w1t = read(d, in1 * h1); b1 = read(d, h1); w2 = read(d, h2 * h1); b2 = read(d, h2);
            wp = read(d, na * h2); bp = read(d, na); wv = read(d, h2); bv = d.readFloat();
            wq = read(d, hq * h2); emb = read(d, na * hq); bq = read(d, hq); wc = read(d, hq * nc);
            wq2 = read(d, hq * hq); bq2 = read(d, hq); wb = read(d, nq * hq); bb = read(d, nq);
        }
        m = new float[hm]; a1 = new float[h1]; a2 = new float[h2]; logits = new float[na]; q = new float[hq]; q2 = new float[hq];
    }

    private static float[] read(DataInputStream d, int n) throws IOException {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = d.readFloat();
        return v;
    }

    private static void addRow(float[] acc, float[] w, int row, int n, float scale) {
        int o = row * n;
        for (int u = 0; u < n; u++) acc[u] += scale * w[o + u];
    }

    /** Runs the trunk; returns value estimate. */
    public float forward(byte[] x) {
        lastX = x;
        int bitBase = dense;   // byte offset of the bit section
        // map encoder
        System.arraycopy(bm, 0, m, 0, hm);
        int mapBytes = mapBits >> 3;
        for (int by = 0; by < mapBytes; by++) {
            int v = x[bitBase + by] & 0xff;
            while (v != 0) {
                int b = Integer.numberOfTrailingZeros(v);
                v &= v - 1;
                addRow(m, wmt, (by << 3) + b, hm, 1f);
            }
        }
        for (int u = 0; u < hm; u++) if (m[u] < 0) m[u] = 0;
        // layer 1
        System.arraycopy(b1, 0, a1, 0, h1);
        for (int i = 0; i < dense; i++) if (x[i] != 0) addRow(a1, w1t, i, h1, (x[i] & 0xff) * (1f / 255f));
        for (int k = 0; k < otherBits; k++) {
            int bit = mapBits + k;
            if ((x[bitBase + (bit >> 3)] >> (bit & 7) & 1) != 0) addRow(a1, w1t, dense + k, h1, 1f);
        }
        for (int u = 0; u < hm; u++) if (m[u] != 0) addRow(a1, w1t, dense + otherBits + u, h1, m[u]);
        for (int u = 0; u < h1; u++) if (a1[u] < 0) a1[u] = 0;
        for (int u = 0; u < h2; u++) {
            float s = b2[u];
            int o = u * h1;
            for (int t = 0; t < h1; t++) s += w2[o + t] * a1[t];
            a2[u] = s > 0 ? s : 0;
        }
        float v = bv;
        for (int u = 0; u < h2; u++) v += wv[u] * a2[u];
        return (float) Math.tanh(v);
    }

    /** Action group: END, MOVE, TURN, 11 target types, THROW, DROP, BECOME_KING, SQUEAK, WRITE_SHARED, WRITE_REG, DISINTEGRATE. */
    public static int group(int a) {
        if (a == Actions.END) return 0;
        if (a < Actions.TURN0) return 1;
        if (a < Actions.TARGET0) return 2;
        if (a < Actions.THROW) return 3 + (a - Actions.TARGET0) / Actions.TC;
        if (a == Actions.THROW) return 3 + Actions.NT;
        if (a < Actions.BECOME_KING) return 4 + Actions.NT;
        if (a == Actions.BECOME_KING) return 5 + Actions.NT;
        if (a == Actions.SQUEAK) return 6 + Actions.NT;
        if (a < Actions.WRITE_REG0) return 7 + Actions.NT;
        if (a < Actions.DISINTEGRATE) return 8 + Actions.NT;
        return 9 + Actions.NT;
    }
    public static final int GROUPS = 10 + Actions.NT;
    private final double[] groupMass = new double[GROUPS];

    /**
     * Choose a legal action. temperature <= 0: pick the group with the largest total probability,
     * then the most likely action within it (plain argmax would favour END, whose mass is not split
     * across directions/targets). temperature > 0: sample from softmax(logits / T).
     */
    public int act(boolean[] mask, double temperature, Random rng) {
        float bestL = Float.NEGATIVE_INFINITY;
        for (int a = 0; a < na; a++) {
            if (!mask[a]) continue;
            float s = bp[a];
            int o = a * h2;
            for (int u = 0; u < h2; u++) s += wp[o + u] * a2[u];
            logits[a] = s;
            if (s > bestL) bestL = s;
        }
        if (temperature <= 0) {
            java.util.Arrays.fill(groupMass, 0);
            for (int a = 0; a < na; a++) if (mask[a]) groupMass[group(a)] += Math.exp(logits[a] - bestL);
            int g = 0;
            for (int k = 1; k < GROUPS; k++) if (groupMass[k] > groupMass[g]) g = k;
            int best = -1;
            for (int a = 0; a < na; a++) if (mask[a] && group(a) == g && (best < 0 || logits[a] > logits[best])) best = a;
            return best;
        }
        double total = 0;
        for (int a = 0; a < na; a++) if (mask[a]) total += Math.exp((logits[a] - bestL) / temperature);
        double r = rng.nextDouble() * total;
        int last = 0;
        for (int a = 0; a < na; a++) if (mask[a]) { last = a; r -= Math.exp((logits[a] - bestL) / temperature); if (r <= 0) return a; }
        return last;
    }

    private static int bit(byte[] x, int p) { return (x[Obs.DENSE + (p >> 3)] >> (p & 7)) & 1; }
    private static int u8(byte[] x, int i) { return x[i] & 0xff; }

    /** Param context bits for action a given observation x (mirrored in alpha0/student/train.py param_ctx). */
    public static void ctx(byte[] x, int a, boolean[] out) {
        int s = Obs.LOCAL + Obs.GLOBAL;
        int mx = u8(x, s + 63), my = u8(x, s + 64);
        int cur = 0;
        if (a >= Actions.WRITE_SHARED0 && a < Actions.WRITE_REG0) {
            int base = Obs.B_SHARED + (a - Actions.WRITE_SHARED0) * 10;
            for (int b = 0; b < 10; b++) cur |= bit(x, base + b) << b;
        } else if (a >= Actions.WRITE_REG0 && a < Actions.DISINTEGRATE) {
            int base = Obs.B_REG + (a - Actions.WRITE_REG0) * 10;
            for (int b = 0; b < 10; b++) cur |= bit(x, base + b) << b;
        }
        int sq = 0;
        if (u8(x, Obs.SQMETA) != 0) for (int b = 0; b < 32; b++) sq |= bit(x, Obs.B_SQ + b) << b;
        int ex = 0, ey = 0;
        if (u8(x, s + 42) != 0) {
            ex = Math.max(0, Math.min(63, mx + ((u8(x, s + 43) - 128) >> 2)));
            ey = Math.max(0, Math.min(63, my + ((u8(x, s + 44) - 128) >> 2)));
        }
        int[] v = {mx, my, u8(x, s + 65) << 8 | u8(x, s + 66), u8(x, s + 55) << 8 | u8(x, s + 56), cur, sq, ex, ey};
        int k = 0;
        for (int f = 0; f < v.length; f++) for (int b = 0; b < CTX_W[f]; b++) out[k++] = (v[f] >>> b & 1) != 0;
    }

    /** Bit logits of the parameter for action a (after forward). out[0..nq). */
    public void paramLogits(int a, float[] out) {
        ctx(lastX, a, cbits);
        int eo = a * hq;
        for (int u = 0; u < hq; u++) {
            float s = bq[u] + emb[eo + u];
            int o = u * h2;
            for (int t = 0; t < h2; t++) s += wq[o + t] * a2[t];
            int co = u * nc;
            for (int c = 0; c < nc; c++) if (cbits[c]) s += wc[co + c];
            q[u] = s > 0 ? s : 0;
        }
        for (int u = 0; u < hq; u++) {
            float s = bq2[u];
            int o = u * hq;
            for (int t = 0; t < hq; t++) s += wq2[o + t] * q[t];
            q2[u] = s > 0 ? s : 0;
        }
        for (int b = 0; b < nq; b++) {
            float s = bb[b];
            int o = b * hq;
            for (int u = 0; u < hq; u++) s += wb[o + u] * q2[u];
            out[b] = s;
        }
    }

    private final float[] pl = new float[Actions.PARAM_MAX_BITS];

    /** Parameter for action a (0 if it takes none). temperature <= 0: each bit by sign; else Bernoulli(sigmoid(l/T)). */
    public int param(int a, double temperature, Random rng) {
        int kind = Actions.paramKind(a);
        if (kind == Actions.P_NONE) return 0;
        paramLogits(a, pl);
        int bits = Actions.PARAM_BITS[kind], v = 0;
        for (int b = 0; b < bits; b++) {
            boolean on = temperature <= 0 ? pl[b] > 0 : rng.nextDouble() < 1.0 / (1.0 + Math.exp(-pl[b] / temperature));
            if (on) v |= 1 << b;
        }
        return v;
    }
}
