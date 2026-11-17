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
    static float[][][] w;   // [point][layer][in * out]: transposed at load, input-major (logits walks inputs outside)
    static float[][][] b;   // [point][layer][out]
    static int[][] dims;    // [point][2 * layer] = in, out

    static boolean has(int point) {
        if (w == null) load();
        return point < w.length && w[point] != null;
    }

    /** The net's output count for point p (0: no net). */
    static int nOut(int p) { return has(p) ? dims[p][dims[p].length - 1] : 0; }

    /** The options of point p the net may pick: the first W.A[p] (missing or 0: all of its outputs). */
    static int reach(int p) {
        int a = p < W.A.length ? W.A[p] : 0, n = nOut(p);
        return a > 0 && a < n ? a : n;
    }

    /** Point p's sampling floor (W.FL[p]; missing or < 0: D.NET_FLOOR). */
    static float floor(int p) { return p < W.FL.length && W.FL[p] >= 0 ? W.FL[p] : D.NET_FLOOR; }

    /** How point p spreads its floor over the allowed options (null: uniformly, floor / m each). W.G[p] gives option
     *  k's group (char k; 0 or beyond the string: dormant, never floor-sampled) and W.GW[p][g] group g's weight; each
     *  allowed group gets its weight's share of the floor, split evenly over its allowed options. All zeros when no
     *  allowed option is in a weighted group (sample() then gives the floor's mass to the softmax).
     *  scripts/dpnet_io.py floor_weights mirrors this arithmetic. */
    static float[] floorWeights(int p, boolean[] allowed, int n) {
        if (p >= W.G.length || W.G[p] == null || W.G[p].isEmpty()) return null;
        String g = W.G[p];
        float[] gw = W.GW[p];
        int[] cnt = new int[gw.length];
        for (int k = 0; k < n && k < g.length(); k++) if (allowed == null || allowed[k]) cnt[g.charAt(k)]++;
        float gam = 0;
        for (int j = 1; j < gw.length; j++) if (cnt[j] > 0) gam += gw[j];
        float[] w = new float[n];
        if (gam <= 0) return w;
        for (int k = 0; k < n && k < g.length(); k++) {
            int j = g.charAt(k);
            if (j > 0 && (allowed == null || allowed[k])) w[k] = gw[j] / gam / cnt[j];
        }
        return w;
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
                float[] a = new float[in * out];   // packed row-major [out][in]; stored as [in][out]
                for (int o = 0; o < out; o++)
                    for (int k = 0; k < in; k++, i += 2) a[k * out + o] = Float.intBitsToFloat(s.charAt(i) << 16 | s.charAt(i + 1));
                float[] c = new float[out];
                for (int k = 0; k < out; k++, i += 2) c[k] = Float.intBitsToFloat(s.charAt(i) << 16 | s.charAt(i + 1));
                wl[l] = a; bl[l] = c;
            }
            w[p] = wl; b[p] = bl; dims[p] = d;
        }
    }

    /** Option logits for the open point's features (D.F[4..4+n)); null when this build has no net for the point.
     *  Inputs outside, outputs inside: each output still sums bias + w * x over the inputs in increasing order, as
     *  the row-by-row form does, so the floats are the same; a zero input (about half of the features and of the
     *  ReLU units) is skipped, which leaves every sum unchanged except for the sign of an exact zero, and neither
     *  the argmax nor the softmax in sample() can see that. */
    static float[] logits(int point, char[] F, int n) {
        if (!has(point)) return null;
        int[] d = dims[point];
        int in0 = d[0];
        float[] x = new float[in0];
        for (int k = 0; k < in0 && k < n; k++) x[k] = F[4 + k];
        float[][] wl = w[point], bl = b[point];
        int last = wl.length - 1;
        for (int l = 0; l <= last; l++) {
            int in = d[2 * l], out = d[2 * l + 1];
            float[] a = wl[l], y = bl[l].clone();
            for (int k = 0, base = 0; k < in; k++, base += out) {
                float xk = x[k];
                if (xk == 0) continue;
                for (int o = 0; o < out; o++) y[o] += a[base + o] * xk;
            }
            if (l < last) for (int o = 0; o < out; o++) if (y[o] < 0) y[o] = 0;
            x = y;
        }
        return x;
    }

    /** Net-mode sampling stream (separate from zdb's G.rand, so zdb's own random choices keep their sequence). */
    static int rng;

    /** Per-game entropy for the streams' seeds. Robot IDs repeat on every game of a map, so a stream seeded from the
     *  ID and round alone drew the same numbers in every game (rl-p1/rl-p2's plan heads: the first plan's picks were
     *  fixed per map and seat). Our runner (alpha0.student.Game) seeds the sandbox's Math.random from the game seed,
     *  team and robot; under the official engine it is seeded by the robot ID, which adds nothing but costs nothing. */
    static int entropy() { return (int) (long) (Math.random() * 4294967296.0); }

    static float uniform() {
        int x = rng;
        if (x == 0) { x = (G.myId * 0x9E3779B1 + G.round * 0x85EBCA6B + 1) ^ entropy(); if (x == 0) x = 1; }
        x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
        rng = x;
        return (x >>> 8) / 16777216f;
    }

    /** The second sampling stream (pass 3's new points, D.alt): their samples never shift the draws of the others. */
    static int rng2;

    static float uniform2() {
        int x = rng2;
        if (x == 0) { x = (G.myId * 0x2545F491 + G.round * 0x27D4EB2F + 7) ^ entropy(); if (x == 0) x = 1; }
        x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
        rng2 = x;
        return (x >>> 8) / 16777216f;
    }

    /** RL builds (D.RL): the last sample()'s option (-1: none allowed), the probability it had (the floor-mixed p[]
     *  value used), its mask (a reference; null: all allowed) and its option count (z.length). Recording only. */
    static int lastK = -1, lastN;
    static float lastP;
    static boolean[] lastMask;

    /** Sample an option: softmax(logits / t) mixed with a uniform floor (probability floor / n each); t = 0 takes the
     *  argmax. allowed (may be null) masks options out. Returns -1 when no option is allowed. */
    static int sample(float[] z, boolean[] allowed, float t, float floor) { return sample(z, allowed, t, floor, false); }

    /** sample() drawing from the second stream (uniform2) when alt. */
    static int sample(float[] z, boolean[] allowed, float t, float floor, boolean alt) { return sample(z, allowed, t, floor, alt, null); }

    /** sample() with the floor spread by fw (floorWeights; null: floor / m on each allowed option). An all-zero fw
     *  gives no floor: the softmax keeps all the mass. */
    static int sample(float[] z, boolean[] allowed, float t, float floor, boolean alt, float[] fw) {
        if (D.RL) { lastK = -1; lastP = 1f; lastMask = allowed; lastN = z.length; }
        if (fw != null) {
            float s = 0;
            for (int k = 0; k < z.length; k++) s += fw[k];
            if (s <= 0) floor = 0;
        }
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
        if (fw == null) { for (int k = 0; k < n; k++) if (allowed == null || allowed[k]) p[k] += floor / m; }
        else for (int k = 0; k < n; k++) if (allowed == null || allowed[k]) p[k] += floor * fw[k];
        float u = alt ? uniform2() : uniform(), acc = 0;
        for (int k = 0; k < n; k++) { acc += p[k]; if (u < acc) { if (D.RL) { lastK = k; lastP = p[k]; } return k; } }
        if (D.RL) { lastK = best; lastP = p[best]; }
        return best;
    }

    private DNet() {}
}
