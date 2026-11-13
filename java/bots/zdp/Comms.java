package zdp;

import battlecode.common.*;

/**
 * Shared-array schema (kings write, everyone reads; values 0..1023) and squeak encoding.
 * Locations are stored as x+1 / y+1 so 0 means empty.
 */
public final class Comms {
    public static final int SYM = 1;
    public static final int FORM_X = 0, FORM_Y = 23, FORM_T = 63;  // king formation site x+1, y+1, heartbeat
    public static final int KING0 = 2, KING_SLOTS = 5;           // 3 ints each: x+1, y+1, heartbeat round%1024
    public static final int ALERT = 17;    // (x+1) | threat<<6, y+1, round%1024: a king sees enemies in war; threat =
                                           // enemy rats within Cfg.KING_THREAT_D2 of it (at most 15)
    public static final int EKING = 20;                          // x+1, y+1, round%1024 last seen
    public static final int MINE0 = 24, MINE_SLOTS = 16;         // 2 ints each
    public static final int CAT0 = 56, CAT_SLOTS = 2;            // 3 ints each: x+1, y+1, round%1024
    public static final int BUDGET = 62;                         // spendable surplus / 4 (kings' projection)
    public static final int SYM_BURN = 512;                      // SYM slot bit (Cfg.RISK_ADAPT): enemy rat traps fired on us

    // squeak types (top 4 bits)
    public static final int SQ_MINE = 1, SQ_SYM = 2, SQ_EKING = 3, SQ_CAT = 4, SQ_TRAP = 5, SQ_START = 6;
    public static final int SQ_BURN = 7;                         // Cfg.RISK_ADAPT: an enemy rat trap fired on us
    public static final int SQ_TRIP = 8;                         // Cfg.TRIP_HALF: a timed walk to mine x, y (extra: steps)
    public static final int SQ_CASH = 9;                         // Cfg.CASH_BANK: rescue site x, y (extra: tag<<10 | start)

    public static int squeak(int type, int x, int y, int extra) {
        return (type << 28) | (extra << 12) | (x << 6) | y;
    }

    public static int sqType(int m) { return (m >>> 28) & 15; }

    public static int sqX(int m) { return (m >> 6) & 63; }

    public static int sqY(int m) { return m & 63; }

    public static int sqExtra(int m) { return (m >> 12) & 0xFFFF; }

    // ------------------------------------------------------------------ readers

    public static int read(int i) {
        try { return G.rc.readSharedArray(i); } catch (GameActionException e) { return 0; }
    }

    public static void write(int i, int v) {
        try { if (G.rc.readSharedArray(i) != v) G.rc.writeSharedArray(i, v); } catch (GameActionException e) { }
    }

    public static boolean fresh(int stamp, int maxAge) {
        if (stamp == 0) return false;
        int age = ((G.round & 1023) - stamp + 1024) & 1023;
        return age <= maxAge;
    }

    public static int stamp() { int s = G.round & 1023; return s == 0 ? 1024 - 1 : s; }

    /** Live king locations from the array (heartbeat within 2 rounds). */
    public static MapLocation[] kings = new MapLocation[KING_SLOTS];
    public static int numKings;

    public static void readKings() {
        numKings = 0;
        for (int k = 0; k < KING_SLOTS; k++) {
            int b = KING0 + 3 * k;
            int x = read(b);
            if (x == 0) continue;
            if (!fresh(read(b + 2), 2)) continue;
            kings[numKings++] = new MapLocation(x - 1, read(b + 1) - 1);
        }
    }

    public static MapLocation nearestKing(MapLocation from) {
        MapLocation best = null;
        int bd = Integer.MAX_VALUE;
        for (int k = 0; k < numKings; k++) {
            int d = from.distanceSquaredTo(kings[k]);
            if (d < bd) { bd = d; best = kings[k]; }
        }
        return best;
    }

    public static int numMines;
    public static MapLocation[] mines = new MapLocation[MINE_SLOTS];

    public static void readMines() {
        if (Cfg.TRIP_HALF != 0) { readMinesTrip(); return; }
        numMines = 0;
        for (int k = 0; k < MINE_SLOTS; k++) {
            int x = read(MINE0 + 2 * k);
            if (x == 0) break;
            mines[numMines++] = new MapLocation(x - 1, read(MINE0 + 2 * k + 1) - 1);
        }
    }

    /** Cfg.TRIP_HALF: each listed mine's fastest timed walk in steps / 4, rounded up (0: none yet), from bits 6-9 of
     *  its x word (low four) and 6-7 of its y word; tripSig changes whenever one does. */
    public static int[] trip;
    public static int tripSig;

    /** readMines with the walk bits masked off the coordinates and kept in trip. */
    static void readMinesTrip() {
        if (trip == null) trip = new int[MINE_SLOTS];
        numMines = 0;
        int sig = 0;
        for (int k = 0; k < MINE_SLOTS; k++) {
            int x = read(MINE0 + 2 * k);
            if (x == 0) break;
            int y = read(MINE0 + 2 * k + 1), q = x >> 6 | (y >> 6) << 4;
            trip[numMines] = q;
            sig = sig * 31 + q;
            mines[numMines++] = new MapLocation((x & 63) - 1, (y & 63) - 1);
        }
        tripSig = sig;
    }

    /** King: a timed walk of t steps to listed mine (x, y) becomes its walk when faster than the one kept. */
    static void noteTrip(int x, int y, int t) {
        int q = Math.max(1, Math.min(63, (t + 3) / 4));
        for (int k = 0; k < numMines; k++) {
            if (mines[k].x != x || mines[k].y != y) continue;
            if (trip[k] != 0 && trip[k] <= q) return;
            trip[k] = q;
            write(MINE0 + 2 * k, x + 1 | (q & 15) << 6);
            write(MINE0 + 2 * k + 1, y + 1 | (q >> 4) << 6);
            return;
        }
    }

    public static boolean hasMine(int x, int y) {
        for (int k = 0; k < numMines; k++) if (mines[k].x == x && mines[k].y == y) return true;
        return false;
    }

    /** King only: append a mine if new. */
    public static void addMine(int x, int y) {
        if (hasMine(x, y) || numMines >= MINE_SLOTS) return;
        write(MINE0 + 2 * numMines, x + 1);
        write(MINE0 + 2 * numMines + 1, y + 1);
        mines[numMines++] = new MapLocation(x, y);
    }

    /** The alerting king's location (ALERT's x carries the threat count in its upper bits), or null. */
    public static MapLocation alertLoc() {
        int v = read(ALERT);
        if (v == 0) return null;
        return new MapLocation((v & 63) - 1, read(ALERT + 1) - 1);
    }

    public static int alertThreat() { return read(ALERT) >> 6; }

    public static MapLocation readLoc(int i) {
        int x = read(i);
        if (x == 0) return null;
        return new MapLocation(x - 1, read(i + 1) - 1);
    }
}
