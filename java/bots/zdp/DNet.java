package zdp;

/**
 * The decision net (net-mode builds): one small MLP per decision point, weights from the generated class W
 * (scripts/build_dpnet.py). Each point's net maps the point's raw features (D.F[4..4+n)) to option logits;
 * input normalisation is folded into the first layer. Hidden layers use ReLU.
 *
 * W.P[point] packs the net as chars: nLayers, then (in, out) per layer, then per layer in*out weights (row-major
 * [out][in]) and out biases, each a float's bits in two chars (high first).
 */
public final class DNet {
    static float[][][] w;   // [point][layer][out * in]
    static float[][][] b;   // [point][layer][out]
    static int[][] dims;    // [point][2 * layer] = in, out

    static boolean has(int point) {
        if (w == null) load();
        return point < w.length && w[point] != null;
    }

    static void load() {
        int np = W.P.length;
        w = new float[np][][]; b = new float[np][][]; dims = new int[np][];
        for (int p = 0; p < np; p++) {
            String s = W.P[p];
            if (s == null || s.isEmpty()) continue;
            int i = 0, nl = s.charAt(i++);
            int[] d = new int[2 * nl];
            for (int l = 0; l < 2 * nl; l++) d[l] = s.charAt(i++);
            float[][] wl = new float[nl][], bl = new float[nl][];
            for (int l = 0; l < nl; l++) {
                int in = d[2 * l], out = d[2 * l + 1];
                float[] a = new float[in * out];
                for (int k = 0; k < a.length; k++, i += 2) a[k] = Float.intBitsToFloat(s.charAt(i) << 16 | s.charAt(i + 1));
                float[] c = new float[out];
                for (int k = 0; k < out; k++, i += 2) c[k] = Float.intBitsToFloat(s.charAt(i) << 16 | s.charAt(i + 1));
                wl[l] = a; bl[l] = c;
            }
            w[p] = wl; b[p] = bl; dims[p] = d;
        }
    }

    /** Option logits for the open point's features (D.F[4..4+n)); null when this build has no net for the point. */
    static float[] logits(int point, char[] F, int n) {
        if (!has(point)) return null;
        int[] d = dims[point];
        int in0 = d[0];
        float[] x = new float[in0];
        for (int k = 0; k < in0 && k < n; k++) x[k] = F[4 + k];
        float[][] wl = w[point], bl = b[point];
        for (int l = 0; l < wl.length; l++) {
            int in = d[2 * l], out = d[2 * l + 1];
            float[] a = wl[l], c = bl[l], y = new float[out];
            for (int o = 0; o < out; o++) {
                float s = c[o];
                int base = o * in;
                for (int k = 0; k < in; k++) s += a[base + k] * x[k];
                y[o] = l < wl.length - 1 && s < 0 ? 0 : s;
            }
            x = y;
        }
        return x;
    }

    /** Net-mode sampling stream (separate from zdb's G.rand, so zdb's own random choices keep their sequence). */
    static int rng;

    static float uniform() {
        int x = rng;
        if (x == 0) x = G.myId * 0x9E3779B1 + G.round * 0x85EBCA6B + 1;
        x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
        rng = x;
        return (x >>> 8) / 16777216f;
    }

    /** RL builds (D.RL): the last sample()'s option (-1: none allowed), the probability it had (the floor-mixed p[]
     *  value used), its mask (a reference; null: all allowed) and its option count (z.length). Recording only. */
    static int lastK = -1, lastN;
    static float lastP;
    static boolean[] lastMask;

    /** Sample an option: softmax(logits / t) mixed with a uniform floor (probability floor / n each); t = 0 takes the
     *  argmax. allowed (may be null) masks options out. Returns -1 when no option is allowed. */
    static int sample(float[] z, boolean[] allowed, float t, float floor) {
        if (D.RL) { lastK = -1; lastP = 1f; lastMask = allowed; lastN = z.length; }
        int n = z.length, best = -1;
        for (int k = 0; k < n; k++) if ((allowed == null || allowed[k]) && (best < 0 || z[k] > z[best])) best = k;
        if (best < 0) return -1;
        int m = 0;
        for (int k = 0; k < n; k++) if (allowed == null || allowed[k]) m++;
        if (t <= 0 && floor <= 0) { if (D.RL) lastK = best; return best; }
        float[] p = new float[n];
        float tot = 0;
        if (t > 0) {
            for (int k = 0; k < n; k++) if (allowed == null || allowed[k]) { p[k] = (float) Math.exp((z[k] - z[best]) / t); tot += p[k]; }
            for (int k = 0; k < n; k++) p[k] = p[k] / tot * (1 - floor);
        } else p[best] = 1 - floor;
        for (int k = 0; k < n; k++) if (allowed == null || allowed[k]) p[k] += floor / m;
        float u = uniform(), acc = 0;
        for (int k = 0; k < n; k++) { acc += p[k]; if (u < acc) { if (D.RL) { lastK = k; lastP = p[k]; } return k; } }
        if (D.RL) { lastK = best; lastP = p[best]; }
        return best;
    }

    private DNet() {}
}
