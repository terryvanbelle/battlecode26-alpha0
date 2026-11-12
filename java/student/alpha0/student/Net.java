package alpha0.student;

import java.io.*;
import java.util.Random;

/**
 * Student MLP inference (student-mlp-v1). Input = observation bytes / 255.
 * h1 = relu(W1 x + b1) (sparse over nonzero inputs), h2 = relu(W2 h1 + b2),
 * logits = Wp h2 + bp, value = tanh(wv . h2 + bv).
 * File (big-endian): int32 magic 0x53544E31, in, h1, h2, nActions; float32 W1T[in][h1], b1[h1],
 * W2[h2][h1], b2[h2], Wp[A][h2], bp[A], wv[h2], bv.
 */
public final class Net {
    public static final int MAGIC = 0x53544E31;
    final int in, h1, h2, na;
    final float[] w1t, b1, w2, b2, wp, bp, wv;
    final float bv;
    private final float[] a1, a2, logits;
    private final int[] nz;

    public Net(File file) throws IOException {
        try (DataInputStream d = new DataInputStream(new BufferedInputStream(new FileInputStream(file), 1 << 16))) {
            if (d.readInt() != MAGIC) throw new IOException("bad weights magic");
            in = d.readInt(); h1 = d.readInt(); h2 = d.readInt(); na = d.readInt();
            if (in != Obs.SIZE || na != Actions.N) throw new IOException("weights do not match obs/action version: in=" + in + " na=" + na);
            w1t = read(d, in * h1); b1 = read(d, h1); w2 = read(d, h2 * h1); b2 = read(d, h2);
            wp = read(d, na * h2); bp = read(d, na); wv = read(d, h2); bv = d.readFloat();
        }
        a1 = new float[h1]; a2 = new float[h2]; logits = new float[na]; nz = new int[in];
    }

    private static float[] read(DataInputStream d, int n) throws IOException {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = d.readFloat();
        return v;
    }

    /** Runs the trunk; returns value estimate. */
    public float forward(byte[] x) {
        System.arraycopy(b1, 0, a1, 0, h1);
        int k = 0;
        for (int i = 0; i < in; i++) if (x[i] != 0) nz[k++] = i;
        for (int j = 0; j < k; j++) {
            int i = nz[j];
            float v = (x[i] & 0xff) * (1f / 255f);
            int o = i * h1;
            for (int u = 0; u < h1; u++) a1[u] += v * w1t[o + u];
        }
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

    /** Action group: END, MOVE, TURN, 11 target types, THROW, DROP, BECOME_KING. */
    public static int group(int a) {
        if (a == Actions.END) return 0;
        if (a < Actions.TURN0) return 1;
        if (a < Actions.TARGET0) return 2;
        if (a < Actions.THROW) return 3 + (a - Actions.TARGET0) / Actions.TC;
        if (a == Actions.THROW) return 3 + Actions.NT;
        if (a < Actions.BECOME_KING) return 4 + Actions.NT;
        return 5 + Actions.NT;
    }
    public static final int GROUPS = 6 + Actions.NT;
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
}
