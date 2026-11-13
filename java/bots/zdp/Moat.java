package zdp;

import battlecode.common.*;

/**
 * Cfg.MOAT: at war a king that sees an enemy rat within MOAT_D2 fills its Chebyshev-2 ring with dirt, one tile per
 * action, helped by babies beside the ring while the king's alert (Comms.ALERT) is under MOAT_IDLE rounds old. A bite
 * reaches dist2 2, so a rat outside the ring cannot touch the king, and a thrown rat stops at the dirt. With
 * MOAT_PORT the ring tile farthest from the map centre that is not a wall stays open for spawns. The king holds
 * still while the moat is up (MOAT_IDLE rounds after its last sighting), and at war no baby digs the alerting
 * king's ring.
 */
public final class Moat {
    /** King: the last round it saw an enemy rat within MOAT_D2 at war. */
    static int seen = -1000;
    public static int placed, dug;

    static boolean up() { return G.round - seen <= Cfg.MOAT_IDLE; }

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
        if (Cfg.MOAT_PORT == 0) return Integer.MAX_VALUE;
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

    /** King at war, after its bites: note an enemy rat within MOAT_D2, then fill one ring tile while the moat is up. */
    static void king(RobotInfo[] enemies) throws GameActionException {
        RobotController rc = G.rc;
        MapLocation c = rc.getLocation(), near = null;
        int nd = Integer.MAX_VALUE;
        for (RobotInfo e : enemies) {
            if (e.type != UnitType.BABY_RAT) continue;
            int d = c.distanceSquaredTo(e.location);
            if (d < nd) { nd = d; near = e.location; }
        }
        if (nd <= Cfg.MOAT_D2) seen = G.round;
        if (up() && rc.isActionReady()) fill(c, 8, near);
    }

    /** The live king of ours whose war alert is under MOAT_IDLE rounds old, or null. */
    static MapLocation alertKing() {
        int a = Comms.read(Comms.ALERT);
        if (a == 0 || !Comms.fresh(Comms.read(Comms.ALERT + 2), Cfg.MOAT_IDLE)) return null;
        int x = (a & 63) - 1, y = Comms.read(Comms.ALERT + 1) - 1;
        for (int k = 0; k < Comms.numKings; k++)
            if (Comms.kings[k].x == x && Comms.kings[k].y == y) return Comms.kings[k];
        return null;
    }

    /** Baby at war with its action left: fill a ring tile of the alerting king next to us. */
    static void baby(MapLocation near) throws GameActionException {
        if (!G.rc.isActionReady()) return;
        MapLocation k = alertKing();
        if (k == null) return;
        int c = G.cheb(G.rc.getLocation(), k);
        if ((c == 2 || c == 3) && fill(k, 2, near)) return;
        if (Cfg.MOAT_DIG > 0 && c <= Cfg.MOAT_DIG_R && G.rc.getDirt() < Cfg.MOAT_DIG) dig(k);
    }

    /** Cfg.MOAT_DIG: with the team's dirt short, dig a dirt tile next to us off king k's ring (5 cheese) for it. */
    static boolean dig(MapLocation k) throws GameActionException {
        RobotController rc = G.rc;
        if (rc.getAllCheese() < Cfg.MOAT_DIG_CHEESE) return false;
        MapLocation me = rc.getLocation(), best = null;
        int bs = Integer.MAX_VALUE;
        for (Direction d : G.DIRS) {
            MapLocation l = me.add(d);
            if (G.cheb(l, k) <= 2 || !rc.canRemoveDirt(l)) continue;
            int s = 2 * l.distanceSquaredTo(k) + (G.pickBit(me, l) ? 1 : 0);
            if (s < bs) { bs = s; best = l; }
        }
        if (best == null) return false;
        rc.removeDirt(best);
        Sym.mem[G.idx(best)] &= ~Sym.DIRT;
        dug++;
        return true;
    }

    /** At war: n is on the alerting king's ring, so a baby leaves its dirt. */
    static boolean keep(MapLocation n) {
        if (G.coop) return false;
        MapLocation k = alertKing();
        return k != null && G.cheb(n, k) == 2;
    }
}
