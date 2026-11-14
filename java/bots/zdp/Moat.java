package zdp;

import battlecode.common.*;

/**
 * King moat (net option DK.A_MOAT_FILL; zdb's own moat, Cfg.MOAT, is off): fill the king's Chebyshev-2 ring with
 * dirt, one tile per action. A bite reaches dist2 2, so a rat outside the ring cannot touch the king, and a thrown rat
 * stops at the dirt. The ring tile farthest from the map centre that is not a wall stays open for spawns (the port).
 */
public final class Moat {
    public static int placed;

    /** A ring tile's port rank (higher is more port-like): distance from the map centre, then the side frame's bit.
     *  The same for the king and its babies, and mirrored with the map. */
    static int key(MapLocation c, MapLocation l) {
        int px = 2 * l.x - G.cx2, py = 2 * l.y - G.cy2;
        return 2 * (px * px + py * py) + (G.pickBitSym(c, l) ? 1 : 0);
    }

    /** Ring tiles of king c ranked below this may be filled: the rank of the best ring tile we know is open (seen,
     *  on the map, no wall). The king's port is the best truly open one, ranked at least as high, so a baby that has
     *  not seen every ring tile never fills it. */
    static int portKey(MapLocation c) {
        int best = Integer.MIN_VALUE;
        for (int dx = -2; dx <= 2; dx++)
            for (int dy = -2; dy <= 2; dy++) {
                if (dx > -2 && dx < 2 && dy > -2 && dy < 2) continue;
                int x = c.x + dx, y = c.y + dy;
                if (!G.onMap(x, y)) continue;
                int i = G.idx(x, y);
                if (!Sym.known(i) || (Sym.mem[i] & Sym.WALL) != 0) continue;
                int k = key(c, new MapLocation(x, y));
                if (k > best) best = k;
            }
        return best;
    }

    /** Place dirt on the fillable ring tile of king c (within dist2 r of us) nearest `near` (else the map centre). */
    static boolean fill(MapLocation c, int r, MapLocation near) throws GameActionException {
        RobotController rc = G.rc;
        MapLocation me = rc.getLocation();
        int pk = portKey(c);
        MapLocation best = null;
        int bs = Integer.MAX_VALUE;
        for (int dx = -2; dx <= 2; dx++)
            for (int dy = -2; dy <= 2; dy++) {
                if (dx > -2 && dx < 2 && dy > -2 && dy < 2) continue;
                MapLocation l = c.translate(dx, dy);
                if (me.distanceSquaredTo(l) > r || !G.onMap(l.x, l.y) || key(c, l) >= pk || !rc.canPlaceDirt(l)) continue;
                int px = 2 * l.x - G.cx2, py = 2 * l.y - G.cy2;
                int s = 2 * (near != null ? l.distanceSquaredTo(near) : px * px + py * py) + (G.pickBit(c, l) ? 1 : 0);
                if (s < bs) { bs = s; best = l; }
            }
        if (best == null) return false;
        rc.placeDirt(best);
        Sym.mem[G.idx(best)] |= Sym.DIRT;
        placed++;
        return true;
    }
}
