package zdn;

import battlecode.common.*;

/**
 * zdn comms: zdb's wire protocol (shared array and squeaks), decoded into facts, plus the king's mechanical writes
 * (beacon, symmetry merge, mine list, cat relay, budget) and the income estimate behind the budget. The net decides
 * what to say (alert, formation, squeak facts); this class only encodes and decodes.
 */
public final class ZC {
    public static final int SYM = 1, FORM_X = 0, FORM_Y = 23, FORM_T = 63, KING0 = 2, KING_SLOTS = 5, ALERT = 17,
        EKING = 20, MINE0 = 24, MINE_SLOTS = 16, CAT0 = 56, BUDGET = 62;
    public static final int SQ_MINE = 1, SQ_SYM = 2, SQ_EKING = 3, SQ_CAT = 4, SQ_CASH = 9;

    public static int squeak(int type, int x, int y, int extra) { return (type << 28) | (extra << 12) | (x << 6) | y; }

    public static int sqType(int m) { return (m >>> 28) & 15; }
    public static int sqX(int m) { return (m >> 6) & 63; }
    public static int sqY(int m) { return m & 63; }
    public static int sqExtra(int m) { return (m >> 12) & 0xFFFF; }

    static final int[] arr = new int[64];

    public static int read(int i) { return i >= MINE0 && i < MINE0 + 2 * MINE_SLOTS ? arr[i] & 63 : arr[i]; }

    public static void write(int i, int v) {
        try {
            int old = ZG.rc.readSharedArray(i);
            if (i >= MINE0 && i < MINE0 + 2 * MINE_SLOTS) v = (v & 63) | (old & ~63);
            if (old != v) ZG.rc.writeSharedArray(i, v);
            arr[i] = v;
        } catch (GameActionException e) { }
    }

    public static int stamp() { int s = ZG.round & 1023; return s == 0 ? 1023 : s; }

    /** Age in rounds of a stamp (large when never written). */
    public static int age(int stamp) { return stamp == 0 ? 1023 : ((ZG.round & 1023) - stamp + 1024) & 1023; }

    public static boolean fresh(int stamp, int maxAge) { return stamp != 0 && age(stamp) <= maxAge; }

    // ---- decoded facts
    public static final MapLocation[] kings = new MapLocation[KING_SLOTS];
    public static final int[] kingSlot = new int[KING_SLOTS];
    public static int numKings;
    public static final MapLocation[] mines = new MapLocation[MINE_SLOTS];
    public static int numMines, symSlot, budget;
    public static MapLocation alert, form, eking;
    public static int alertThreat, alertAge = 1023, formAge = 1023, ekingAge = 1023;
    public static final MapLocation[] cats = new MapLocation[2];
    public static final int[] catAge = {1023, 1023};

    /** Read the whole array and decode it (every robot, every turn). */
    public static void readAll() {
        try { for (int i = 0; i < 64; i++) arr[i] = ZG.rc.readSharedArray(i); } catch (GameActionException e) { }
        numKings = 0;
        for (int k = 0; k < KING_SLOTS; k++) {
            int b = KING0 + 3 * k, x = read(b);
            if (x == 0 || !fresh(read(b + 2), 2)) continue;
            kingSlot[numKings] = k;
            kings[numKings++] = new MapLocation(x - 1, read(b + 1) - 1);
        }
        numMines = 0;
        for (int k = 0; k < MINE_SLOTS; k++) {
            int x = read(MINE0 + 2 * k);
            if (x == 0) break;
            mines[numMines++] = new MapLocation(x - 1, read(MINE0 + 2 * k + 1) - 1);
        }
        symSlot = read(SYM) & 7;
        budget = read(BUDGET) * 4;
        int a = read(ALERT);
        alert = a == 0 ? null : new MapLocation((a & 63) - 1, read(ALERT + 1) - 1);
        alertThreat = a >> 6;
        alertAge = age(read(ALERT + 2));
        int fx = read(FORM_X);
        form = fx == 0 ? null : new MapLocation(fx - 1, read(FORM_Y) - 1);
        formAge = age(read(FORM_T));
        int ex = read(EKING);
        eking = ex == 0 ? null : new MapLocation(ex - 1, read(EKING + 1) - 1);
        ekingAge = age(read(EKING + 2));
        for (int c = 0; c < 2; c++) {
            int b = CAT0 + 3 * c, x = read(b);
            cats[c] = x == 0 ? null : new MapLocation(x - 1, read(b + 1) - 1);
            catAge[c] = x == 0 ? 1023 : age(read(b + 2));
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

    public static boolean hasMine(int x, int y) {
        for (int k = 0; k < numMines; k++) if (mines[k].x == x && mines[k].y == y) return true;
        return false;
    }

    /** The formation site while its heartbeat is fresh (zdp Baby.formSite), else null. */
    public static MapLocation formSite() { return form != null && formAge <= 2 ? form : null; }

    // ---- squeaks heard this turn
    public static final MapLocation[] cashSite = new MapLocation[8];
    public static final int[] cashStart = new int[8], cashAgeOf = new int[8];
    public static int nCash;
    public static int[] heard = new int[16];
    public static int[] heardRound = new int[16];
    public static int nHeard;

    /** Read queued squeaks: mines go to own memory, cash sites are kept for the CASH register's candidates. */
    public static void hear() {
        Message[] ms = ZG.rc.readSqueaks(-1);
        nCash = 0;
        nHeard = 0;
        for (Message msg : ms) {
            int m = msg.getBytes(), r = msg.getRound();
            if (nHeard < 16 && r >= ZG.round - 1) { heard[nHeard] = m; heardRound[nHeard++] = r; }
            switch (sqType(m)) {
                case SQ_MINE: ZM.noteMine(sqX(m), sqY(m)); break;
                case SQ_CASH: {
                    if (ZG.round - r > 2 || nCash >= cashSite.length) break;
                    int x = sqX(m), y = sqY(m), e = sqExtra(m);
                    if ((e >> 10 & 63) != cashTag(x, y)) break;
                    int start = ZG.round - (((ZG.round & 1023) - (e & 1023)) & 1023);
                    if (ZG.round - start > 50) break;
                    MapLocation s = new MapLocation(x, y);
                    int j = 0;
                    for (; j < nCash; j++) if (cashSite[j].equals(s)) break;
                    if (j == nCash) { cashSite[nCash] = s; cashStart[nCash] = start; cashAgeOf[nCash++] = ZG.round - r; }
                    break;
                }
                default: break;
            }
        }
    }

    public static int cashTag(int x, int y) { return (x * 3 + y * 5 + 17) & 63; }

    // ------------------------------------------------------------------ king mechanics (player builds only)

    public static int slot = -1;

    public static void claimSlot() {
        for (int k = 0; k < KING_SLOTS; k++) {
            int b = KING0 + 3 * k;
            if (read(b) == 0 || !fresh(read(b + 2), 3)) { slot = k; return; }
        }
        slot = 0;
    }

    public static void publish() {
        MapLocation me = ZG.rc.getLocation();
        int b = KING0 + 3 * slot;
        write(b, me.x + 1);
        write(b + 1, me.y + 1);
        write(b + 2, stamp());
        int s = read(SYM);
        int m = s == 0 ? ZM.mask : (s & ZM.mask);
        if (m == 0) m = ZM.mask;
        ZM.mask = m;
        write(SYM, m);
    }

    public static void addMine(int x, int y) {
        if (hasMine(x, y) || numMines >= MINE_SLOTS) return;
        write(MINE0 + 2 * numMines, x + 1);
        write(MINE0 + 2 * numMines + 1, y + 1);
        mines[numMines++] = new MapLocation(x, y);
    }

    static void noteCat(int x, int y) {
        int c0x = read(CAT0);
        if (c0x != 0 && (Math.abs(c0x - 1 - x) > 6 || Math.abs(read(CAT0 + 1) - 1 - y) > 6)) {
            write(CAT0 + 3, c0x);
            write(CAT0 + 4, read(CAT0 + 1));
            write(CAT0 + 5, read(CAT0 + 2));
        }
        write(CAT0, x + 1);
        write(CAT0 + 1, y + 1);
        write(CAT0 + 2, stamp());
    }

    /** The king's relay of squeaks it heard (zdp King.readSqueaks): mines, symmetry, enemy king, cats. */
    public static void kingRelay() {
        int used = 0;
        for (int i = 0; i < nHeard && used < 16; i++) {
            used++;
            int m = heard[i];
            switch (sqType(m)) {
                case SQ_MINE: addMine(sqX(m), sqY(m)); break;
                case SQ_SYM: ZM.merge(sqExtra(m) & 7); break;
                case SQ_EKING: write(EKING, sqX(m) + 1); write(EKING + 1, sqY(m) + 1); write(EKING + 2, stamp()); break;
                case SQ_CAT: noteCat(sqX(m), sqY(m)); break;
                default: break;
            }
        }
    }

    public static void writeAlert(int threat) {
        MapLocation me = ZG.me;
        write(ALERT, (me.x + 1) | (Math.min(15, threat) << 6));
        write(ALERT + 1, me.y + 1);
        write(ALERT + 2, stamp());
    }

    public static void formOpen(MapLocation s) { write(FORM_X, s.x + 1); write(FORM_Y, s.y + 1); write(FORM_T, stamp()); }

    public static void formHold() { write(FORM_T, stamp()); }

    public static void formClose() { write(FORM_T, 0); }

    // ------------------------------------------------------------------ income estimate (zdp Econ)

    static int prevEnd = -1, pending;
    static final int[] ring = new int[100];
    static int sum, n, head;

    static int liveKings() { return Math.max(1, numKings); }

    public static void econStart() {
        int c = ZG.rc.getGlobalCheese();
        pending = prevEnd < 0 ? 0 : Math.max(0, c - prevEnd + 2 * liveKings());
    }

    public static void econReceived(int x) { if (x > 0) pending += x; }

    public static void econEnd() {
        if (n == ring.length) sum -= ring[head]; else n++;
        ring[head] = pending;
        sum += pending;
        head = (head + 1) % ring.length;
        prevEnd = ZG.rc.getGlobalCheese();
    }

    public static int rate100() { return n == 0 ? 0 : sum * 100 / n; }

    public static int surplus() {
        int left = Math.max(0, 2000 - ZG.round);
        int net100 = rate100() * 75 / 100 - 200 * liveKings();
        return ZG.rc.getGlobalCheese() + net100 * left / 100;
    }

    public static void writeBudget() { write(BUDGET, Math.max(0, Math.min(1023, surplus() / 4))); }
}
