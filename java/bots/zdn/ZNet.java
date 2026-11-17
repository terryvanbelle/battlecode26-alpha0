package zdn;

/**
 * zdn's policy net (docs/zdn.md), the Java mirror of scripts/zdn_train.py's ZNet with every standardisation folded in.
 *
 *   trunk   h = relu(W2 relu(W1 [S; L] + b1) + b2)                  once per turn
 *   scorer  u = A h + C ctx + a;  logit_c = w . relu(u + B[kind_c] x_c + E[kind_c]) + w0
 *
 * One net per unit type. ZW.BABY / ZW.KING pack a net as chars (scripts/zdn_pack.py): a header of dims, then floats as
 * two chars each (high half first) in this order: W1 [H1][nIn], b1, W2 [H][H1], b2, the step scorer (A [Hh][H], a, C
 * [Hh][nC], B [nK][Hh][D], E [nK][Hh], w [Hh], w0), then each register scorer (the same; C [Hh][nC_r] when nC_r > 0).
 */
public final class ZNet {
    static final class Scorer {
        int nC, nK, D, Hh;
        float[] A, a, C, B, E, w;   // A [Hh][H], C [Hh][nC], B [nK][Hh][D], E [nK][Hh]
        float w0;
    }

    final int nS, nL, nIn, H1, H, Hh;
    final float[] W1, b1, W2, b2;
    final Scorer step;
    final Scorer[] regs;

    private int at;
    private final String s;

    ZNet(String packed) {
        s = packed;
        at = 0;
        int magic = ch();
        if (magic != 0x5A4E) throw new IllegalStateException("bad net");
        nS = ch(); nL = ch(); H1 = ch(); H = ch(); Hh = ch();
        nIn = nS + nL;
        int nC = ch(), nKs = ch(), Ds = ch(), nR = ch();
        int[] rk = new int[nR], rd = new int[nR], rc = new int[nR];
        for (int r = 0; r < nR; r++) { rk[r] = ch(); rd[r] = ch(); rc[r] = ch(); }
        W1 = fl(H1 * nIn); b1 = fl(H1); W2 = fl(H * H1); b2 = fl(H);
        step = scorer(nC, nKs, Ds);
        regs = new Scorer[nR];
        for (int r = 0; r < nR; r++) regs[r] = scorer(rc[r], rk[r], rd[r]);
    }

    private int ch() { return s.charAt(at++); }

    private float[] fl(int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++, at += 2) a[i] = Float.intBitsToFloat(s.charAt(at) << 16 | s.charAt(at + 1));
        return a;
    }

    private Scorer scorer(int nC, int nK, int D) {
        Scorer c = new Scorer();
        c.nC = nC; c.nK = nK; c.D = D; c.Hh = Hh;
        c.A = fl(Hh * H); c.a = fl(Hh);
        c.C = nC > 0 ? fl(Hh * nC) : null;
        c.B = fl(nK * Hh * D); c.E = fl(nK * Hh); c.w = fl(Hh);
        c.w0 = fl(1)[0];
        return c;
    }

    /** A net packed in several string constants (class-file constants are limited to 65535 bytes). */
    static ZNet of(String[] parts) {
        if (parts == null || parts.length == 0) return null;
        StringBuilder sb = new StringBuilder();
        for (String p : parts) sb.append(p);
        return new ZNet(sb.toString());
    }

    /** Logits of every option (kinds the net does not know get -inf). */
    static float[] logits(Scorer c, float[] h, float[] ctx, int n, int[] kind, int[][] x) {
        float[] u = base(c, h, ctx);
        float[] z = new float[n];
        float[] xf = new float[c.D];
        for (int i = 0; i < n; i++) {
            if (kind[i] < 0 || kind[i] >= c.nK) { z[i] = Float.NEGATIVE_INFINITY; continue; }
            int[] xi = x[i];
            for (int d = 0; d < c.D; d++) xf[d] = xi[d];
            z[i] = logit(c, u, kind[i], xf);
        }
        return z;
    }

    /** The policy's own random stream. Robot IDs repeat on every game of a map, so a stream seeded from them (or from
     *  birth facts, like ZG.rand) would replay the same draws on every game of a map and seat (the rl-p2 bug,
     *  docs/rl-es-plan.md section 21). Math.random is seeded per game, team and robot by our runner
     *  (alpha0.student.Game reseed) and per robot by the official engine. */
    static int prng;

    static int draw(int n) {
        if (prng == 0) { prng = (ZG.id * 0x9E3779B1 + 0x632BE5AB) ^ (int) (long) (Math.random() * 4294967296.0); if (prng == 0) prng = 1; }
        int x = prng;
        x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
        prng = x;
        return (x >>> 1) % n;
    }

    /** log p of the option the last sample() returned (0 when it was forced: one option, or T <= 0). */
    static double lastLogp;

    /** Sample an index from softmax(z / T); T <= 0 takes the argmax. */
    static int sample(float[] z, int n, float T) {
        int best = 0;
        for (int i = 1; i < n; i++) if (z[i] > z[best]) best = i;
        lastLogp = 0;
        if (T <= 0 || n <= 1) return best;
        double m = z[best], tot = 0;
        double[] p = new double[n];
        for (int i = 0; i < n; i++) { p[i] = z[i] == Float.NEGATIVE_INFINITY ? 0 : Math.exp((z[i] - m) / T); tot += p[i]; }
        double u = (draw(1 << 24) + 0.5) / (1 << 24) * tot;
        int pick = best;
        for (int i = 0; i < n; i++) { u -= p[i]; if (u <= 0 && p[i] > 0) { pick = i; break; } }
        lastLogp = Math.log(p[pick] / tot);
        return pick;
    }

    /** The trunk on the turn's inputs x = [S; L] (raw values; the standardisation is folded into W1). */
    float[] trunk(float[] x) {
        float[] y = new float[H1];
        for (int o = 0, base = 0; o < H1; o++, base += nIn) {
            float t = b1[o];
            for (int k = 0; k < nIn; k++) { float xk = x[k]; if (xk != 0) t += W1[base + k] * xk; }
            y[o] = t > 0 ? t : 0;
        }
        float[] h = new float[H];
        for (int o = 0, base = 0; o < H; o++, base += H1) {
            float t = b2[o];
            for (int k = 0; k < H1; k++) { float yk = y[k]; if (yk != 0) t += W2[base + k] * yk; }
            h[o] = t > 0 ? t : 0;
        }
        return h;
    }

    /** u = A h + C ctx + a for one scorer (ctx may be null when the scorer has none). */
    static float[] base(Scorer c, float[] h, float[] ctx) {
        int Hh = c.Hh, H = h.length;
        float[] u = new float[Hh];
        for (int o = 0, b = 0; o < Hh; o++, b += H) {
            float t = c.a[o];
            for (int k = 0; k < H; k++) { float hk = h[k]; if (hk != 0) t += c.A[b + k] * hk; }
            if (c.C != null) for (int k = 0, cb = o * c.nC; k < c.nC; k++) t += c.C[cb + k] * ctx[k];
            u[o] = t;
        }
        return u;
    }

    /** The logit of one option (kind, raw features x[0..D)) given u. */
    static float logit(Scorer c, float[] u, int kind, float[] x) {
        int Hh = c.Hh, D = c.D;
        float z = c.w0;
        int bb = kind * Hh * D, eb = kind * Hh;
        for (int o = 0; o < Hh; o++) {
            float t = u[o] + c.E[eb + o];
            for (int d = 0, j = bb + o * D; d < D; d++, j++) { float xd = x[d]; if (xd != 0) t += c.B[j] * xd; }
            if (t > 0) z += c.w[o] * t;
        }
        return z;
    }
}
