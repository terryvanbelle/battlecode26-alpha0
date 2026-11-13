package zdp;

import battlecode.common.*;

/**
 * Cat lanes (Cfg.CAT_LAY): the tiles cats have walked, one map row at a time (a row is allocated on its first mark).
 * A cat walks a fixed waypoint loop by BFS, so a tile it covers on two passes at least Cfg.CAT_LAY_GAP rounds apart
 * is on its lap and it will come back. Each entry: (round + 1) the tile was last under a cat, plus LANE once two
 * passes were seen.
 */
public final class CatLane {
    static final int LANE = 1 << 16, ROUND = 0xFFFF;
    static int[][] rows;

    /** A cat body (2x2 with its lower-left tile at bl) seen this round. */
    static void see(MapLocation bl, int round) {
        if (rows == null) rows = new int[G.H][];
        for (int dy = 0; dy <= 1; dy++) {
            int y = bl.y + dy;
            if (y < 0 || y >= G.H) continue;
            int[] row = rows[y];
            if (row == null) row = rows[y] = new int[G.W];
            for (int dx = 0; dx <= 1; dx++) {
                int x = bl.x + dx;
                if (x < 0 || x >= G.W) continue;
                row[x] = mark(row[x], round, Cfg.CAT_LAY_GAP);
            }
        }
    }

    /** The entry after a cat is seen on the tile at round: a second pass gap or more rounds after the last makes it lane. */
    static int mark(int v, int round, int gap) {
        int lane = v & LANE;
        if (v != 0 && round - ((v & ROUND) - 1) >= gap) lane = LANE;
        return lane | (round + 1);
    }

    static boolean lane(int x, int y) {
        if (rows == null || y < 0 || y >= G.H || x < 0 || x >= G.W) return false;
        int[] row = rows[y];
        return row != null && (row[x] & LANE) != 0;
    }

    /** Rounds since a cat was last on the tile (a large number if never). */
    static int age(int x, int y, int round) {
        if (rows == null || y < 0 || y >= G.H || x < 0 || x >= G.W) return 1 << 20;
        int[] row = rows[y];
        if (row == null || row[x] == 0) return 1 << 20;
        return round - ((row[x] & ROUND) - 1);
    }
}
