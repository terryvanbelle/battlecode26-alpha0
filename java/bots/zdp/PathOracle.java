package zdp;

import battlecode.common.MapLocation;

/**
 * Cfg.PATH_ORACLE (an experiment, never a candidate): offline half-test verdicts for the mines whose 8-way path
 * distance (walls block, dirt +4) from the two starts disagrees with the Euclidean test. Rows are {W, H, start A x,
 * y, mine x, y, 1 when the mine is nearer A by path}; all three maps rotate. uneruesansfin: (32,49) is 809 vs 2465
 * in d2 from A (4,54) but 137 vs 92 steps; wallsofparadis and cheesebottles split near-equal pairs the other way.
 */
public final class PathOracle {
    static final int[][] T = {
        {60, 60, 4, 54, 6, 6, 0}, {60, 60, 4, 54, 27, 10, 1}, {60, 60, 4, 54, 32, 49, 0}, {60, 60, 4, 54, 53, 53, 1},
        {60, 60, 3, 2, 55, 5, 1}, {60, 60, 3, 2, 38, 22, 1}, {60, 60, 3, 2, 21, 37, 0}, {60, 60, 3, 2, 4, 54, 0},
        {59, 20, 2, 11, 19, 2, 0}, {59, 20, 2, 11, 39, 17, 1},
    };

    /** +1 when mine l is nearer by path to our start r than to the enemy's, -1 when farther, 0 without a verdict. */
    static int verdict(MapLocation l, MapLocation r) {
        for (int[] t : T) {
            if (t[0] != G.W || t[1] != G.H || t[4] != l.x || t[5] != l.y) continue;
            MapLocation a = new MapLocation(t[2], t[3]);
            boolean weA = r.distanceSquaredTo(a) < r.distanceSquaredTo(G.mirror(a, Sym.ROT));
            return weA == (t[6] == 1) ? 1 : -1;
        }
        return 0;
    }
}
