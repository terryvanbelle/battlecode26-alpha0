package zdn;

import battlecode.common.*;

/**
 * Path-distance fields over the remembered map (docs/zdn.md: "Path distances ... given to the net as inputs").
 * Flow's cost model: entering an open or unknown tile costs 1, a dirt tile DIG (4: dig, then step); walls, off-map
 * tiles and our kings' bodies are blocked (a king body is open when the field's root lies inside one). A field is
 * cached per root for the rest of the turn.
 */
public final class ZF {
    public static final int DIG = 4, INF = 1 << 20;
    static final int SLOTS = 4;
    static final int[][] fields = new int[SLOTS][];
    static final int[] key = new int[SLOTS], made = new int[SLOTS];
    static int next;
    static int[][] bucket;
    static int[] blen;

    /** Distance from every tile to root (forward=false: cost of walking tile -> root) or from root to every tile
     *  (forward=true). */
    public static int[] field(MapLocation root, boolean forward) {
        int k = (root.x + root.y * ZG.W) * 2 + (forward ? 1 : 0);
        for (int i = 0; i < SLOTS; i++) if (made[i] == ZG.round + 1 && key[i] == k && fields[i] != null) return fields[i];
        int s = next;
        next = (next + 1) % SLOTS;
        int n = ZG.W * ZG.H;
        if (fields[s] == null) fields[s] = new int[n];
        build(fields[s], root, forward);
        key[s] = k;
        made[s] = ZG.round + 1;
        return fields[s];
    }

    /** Forget cached fields (the map memory changed mid-turn, e.g. after a dig). */
    public static void invalidate() { for (int i = 0; i < SLOTS; i++) made[i] = 0; }

    static int enter(int i, int block) {
        int m = ZM.mem[i];
        if ((m & block) != 0) return -1;
        return (m & ZM.DIRT) != 0 ? DIG : 1;
    }

    static void build(int[] d, MapLocation root, boolean forward) {
        int W = ZG.W, H = ZG.H, n = W * H;
        if (bucket == null) { bucket = new int[DIG + 1][n * 2]; blen = new int[DIG + 1]; }
        java.util.Arrays.fill(d, INF);
        int r = root.x + root.y * W;
        int block = ZM.WALL | ((ZM.mem[r] & ZM.KING) != 0 ? 0 : ZM.KING);
        for (int b = 0; b <= DIG; b++) blen[b] = 0;
        d[r] = 0;
        bucket[0][blen[0]++] = r;
        int cur = 0, pending = 1;
        while (pending > 0) {
            int b = cur % (DIG + 1);
            if (blen[b] == 0) { cur++; continue; }
            int v = bucket[b][--blen[b]];
            pending--;
            if (d[v] != cur) continue;
            int vx = v % W, vy = v / W;
            int cv = forward ? 0 : enter(v, 0);   // reverse: the walker leaving u enters v
            for (int k = 0; k < 8; k++) {
                int ux = vx + ZG.DX[k], uy = vy + ZG.DY[k];
                if (ux < 0 || uy < 0 || ux >= W || uy >= H) continue;
                int u = ux + uy * W;
                int c;
                if (forward) { c = enter(u, block); if (c < 0) continue; }
                else { if (enter(u, block) < 0) continue; c = v == r ? 1 : cv; }
                int nd = cur + c;
                if (nd < d[u]) {
                    d[u] = nd;
                    int nb = nd % (DIG + 1);
                    bucket[nb][blen[nb]++] = u;
                    pending++;
                }
            }
        }
    }

    /** Field value at (x, y), INF off the map. */
    public static int at(int[] f, int x, int y) { return ZG.onMap(x, y) ? f[x + y * ZG.W] : INF; }
}
