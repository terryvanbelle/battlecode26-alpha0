package zdp;

import battlecode.common.*;

/**
 * TrapRisk's marks stored one map row at a time (Cfg.RISK_ROWS): a row is allocated on its first mark. The flat
 * W*H array costs about 3*W*H bytecodes to allocate in one turn, and TrapRisk waits for a turn with that much left
 * plus Cfg.BC_OPTIONAL: 18800 on a 60x60 map, above a baby's 17500 limit, so no rat there ever had marks. Same
 * semantics as TrapRisk's mark array: round + 1 the tile was last marked, 0 = none.
 */
public final class TrapRows {
    static int[][] rows;

    static void init() {
        if (rows == null) rows = new int[G.H][];
    }

    static void put(MapLocation l) {
        if (!G.onMap(l.x, l.y)) return;
        int[] row = rows[l.y];
        if (row == null) row = rows[l.y] = new int[G.W];
        row[l.x] = G.round + 1;
    }

    /** Clear marks within Chebyshev 1 of me made before we arrived (a trap there would have fired on us). */
    static void clear(MapLocation me, int arrived) {
        for (int dy = -1; dy <= 1; dy++) {
            int y = me.y + dy;
            if (y < 0 || y >= G.H) continue;
            int[] row = rows[y];
            if (row == null) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int x = me.x + dx;
                if (x < 0 || x >= G.W) continue;
                if (row[x] != 0 && row[x] - 1 < arrived) row[x] = 0;
            }
        }
    }

    static boolean risky(MapLocation l) {
        if (rows == null) return false;
        for (int dy = -1; dy <= 1; dy++) {
            int y = l.y + dy;
            if (y < 0 || y >= G.H) continue;
            int[] row = rows[y];
            if (row == null) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int x = l.x + dx;
                if (x < 0 || x >= G.W) continue;
                int m = row[x];
                if (m != 0 && G.round - (m - 1) <= TrapRisk.TTL) return true;
            }
        }
        return false;
    }

    static int mask(MapLocation me, int ttl) {
        if (rows == null) return 0;
        int m = 0, w = G.W, x0 = me.x, y0 = me.y, r = G.round - ttl;
        int[] near = TrapRisk.NEAR;
        for (int dy = -2; dy <= 2; dy++) {
            int y = y0 + dy;
            if (y < 0 || y >= G.H) continue;
            int[] row = rows[y];
            if (row == null) continue;
            for (int dx = -2; dx <= 2; dx++) {
                int x = x0 + dx;
                if (x < 0 || x >= w) continue;
                int k = row[x];
                if (k == 0 || k - 1 < r) continue;
                m |= near[(dx + 2) + 5 * (dy + 2)];
            }
        }
        return m == 0xFF ? 0 : m;
    }
}
