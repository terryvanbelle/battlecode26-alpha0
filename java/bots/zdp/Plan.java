package zdp;

/**
 * The team plan (design C, docs/rl-es-plan.md section 21): live values of zdb's strategy constants, one bin per head.
 * A king opens the plan heads (DP) on its first turn, every 50 rounds and on two events, and writes the bins into
 * the spare high bits of the MINE slots (Comms.PLAN0); every robot reads them at the start of its turn (read()).
 * Each head's zdb bin (ZDB) returns zdb's own Cfg constant, so a plan of zdb bins plays zdb exactly; no plan written
 * yet (stamp 0) means zdb bins.
 *
 * Plan word (PLAN_BITS bits, 4 per slot over PLAN_SLOTS slots): bits 0-23 the 8 heads' bins (3 bits each), 24-29 the
 * plan's epoch (round / 50) + 1 (0: no plan), 30 cooperation when planned, 31-33 the live kings when planned.
 */
final class Plan {
    static final int POSTURE = 0, ARMY = 1, CORE = 2, THRIFT = 3, ROLES = 4, RUSH = 5, LOAD = 6, KINGS = 7, HEADS = 8;
    /** Options per head and zdb's bin. */
    static final int[] K = {7, 7, 7, 7, 7, 6, 7, 4};
    static final int[] ZDB = {0, 3, 4, 3, 3, 3, 3, 1};
    // bin values (the zdb bin's entry documents zdb's constant; the code returns Cfg's value for it)
    static final int[] BREAK = {30, 60, 120, 250, 500, 1000, 5000};      // first round hostile in cooperation
    static final int[] CAP = {8, 12, 15, 18, 24, 30, 40};                // baby cap, peace and war alike
    static final int[] CORE_BIG = {8, 10, 12, 15, 18, 21, 24};           // core babies (maps over CORE_SMALL_AREA)
    static final int[] CORE_SM = {4, 6, 8, 10, 12, 15, 18};              // core babies on small maps
    static final int[] THRIFT_PCT = {33, 50, 75, 100, 150, 200, 300};    // CORE/WAR/SPAWN floors and SPAWN_MARGIN, %
    static final int[] FIGHT_D = {1, 4, 3, 2, 3, 4, 1}, FIGHT_N = {0, 1, 1, 1, 2, 3, 1};   // fighter: round % D < N
    static final int[] RUSH_TO = {4, 8, 12, 16, 24, 32};                 // rushers: spawn index RUSH_FROM..RUSH_TO-1
    static final int[] DELIVER = {10, 15, 20, 30, 45, 60, 90};           // raw cheese a gatherer delivers at
    static final int[] FORM = {1, 2, 3, 4};                              // kings formed up to

    static final int PLAN_SLOTS = 9, PLAN_BITS = 36;

    static final int[] bin = ZDB.clone();
    /** The plan's epoch + 1 (0: no plan yet), and the cooperation flag and live kings when it was made. */
    static int stamp, kingsAt;
    static boolean coopAt;

    /** Read the plan word from the array (high bits of Comms.PLAN0 ..). */
    static void read() {
        long w = 0;
        for (int i = 0; i < PLAN_SLOTS; i++) w |= (long) Comms.planBits(i) << (4 * i);
        decode(w);
    }

    static void decode(long w) {
        stamp = (int) (w >>> 24) & 63;
        if (stamp == 0) { System.arraycopy(ZDB, 0, bin, 0, HEADS); coopAt = true; kingsAt = 0; return; }
        for (int h = 0; h < HEADS; h++) { int b = (int) (w >>> (3 * h)) & 7; bin[h] = b < K[h] ? b : ZDB[h]; }
        coopAt = ((w >>> 30) & 1) != 0;
        kingsAt = (int) (w >>> 31) & 7;
    }

    static long encode(int[] b, int epoch, boolean coop, int kings) {
        long w = 0;
        for (int h = 0; h < HEADS; h++) w |= (long) (b[h] & 7) << (3 * h);
        w |= (long) ((epoch + 1) & 63) << 24;
        if (coop) w |= 1L << 30;
        w |= (long) Math.min(7, kings) << 31;
        return w;
    }

    /** King: publish a new plan (and adopt it for this turn). */
    static void write(int[] b, int epoch, boolean coop, int kings) {
        long w = encode(b, epoch, coop, kings);
        for (int i = 0; i < PLAN_SLOTS; i++) Comms.setPlanBits(i, (int) (w >>> (4 * i)) & 15);
        decode(w);
    }

    static boolean zdb(int h) { return bin[h] == ZDB[h]; }

    // ------------------------------------------------------------------ live values (zdb bin: Cfg's constant)

    static int breakRound() { return zdb(POSTURE) ? Cfg.BREAK_ROUND : BREAK[bin[POSTURE]]; }

    static int cap(boolean war) { return zdb(ARMY) ? (war ? Cfg.MAX_BABIES_WAR : Cfg.MAX_BABIES) : CAP[bin[ARMY]]; }

    static int core(boolean small) {
        return zdb(CORE) ? (small ? Cfg.CORE_SMALL : Cfg.CORE_BABIES) : small ? CORE_SM[bin[CORE]] : CORE_BIG[bin[CORE]];
    }

    /** A spawn floor or margin scaled by the thrift bin. */
    static int thrift(int v) { return zdb(THRIFT) ? v : v * THRIFT_PCT[bin[THRIFT]] / 100; }

    /** A baby born in round r past GATHER_FIRST fights. */
    static boolean fighter(int r) {
        if (zdb(ROLES)) return r % Cfg.FIGHT_EVERY == 0;
        int b = bin[ROLES];
        return r % FIGHT_D[b] < FIGHT_N[b];
    }

    static int rushTo() { return zdb(RUSH) ? Cfg.RUSH_TO : RUSH_TO[bin[RUSH]]; }

    static int deliverAt() { return zdb(LOAD) ? Cfg.DELIVER_AT : DELIVER[bin[LOAD]]; }

    static int formMax() { return zdb(KINGS) ? Cfg.FORM_MAX : FORM[bin[KINGS]]; }

    private Plan() {}
}
