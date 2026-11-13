package zdp;

/**
 * King-side income estimate. Each king turn: income = rise in global cheese since the end of our last turn plus
 * the upkeep all kings paid meanwhile, plus what we picked up ourselves. Others' spending counts against income,
 * so the estimate is conservative. surplus() projects global cheese to round 2000 at a discounted rate.
 */
public final class Econ {
    static int prevEnd = -1, pending;
    static final int[] ring = new int[Cfg.INCOME_WINDOW];
    static int sum, n, head;

    static int kings() { return Math.max(1, Comms.numKings); }

    /** A cheese floor f for our live kings (Cfg.FLOOR_PER_KING), else f. */
    static int floor(int f) { return Cfg.FLOOR_PER_KING ? f * kings() : f; }

    public static void startTurn() {
        int c = G.rc.getGlobalCheese();
        pending = prevEnd < 0 ? 0 : Math.max(0, c - prevEnd + 2 * kings());
    }

    public static void received(int x) { if (x > 0) pending += x; }

    public static void endTurn() {
        if (n == ring.length) sum -= ring[head]; else n++;
        ring[head] = pending;
        sum += pending;
        head = (head + 1) % ring.length;
        prevEnd = G.rc.getGlobalCheese();
    }

    /** Income per round x100 over the window. */
    public static int rate100() { return n == 0 ? 0 : sum * 100 / n; }

    /** Projected global cheese at the end of the game, minus nothing spent: negative means we would starve. */
    public static int surplus() { return surplus(kings()); }

    /** surplus() as if we had k kings. */
    public static int surplus(int k) {
        int left = Math.max(0, 2000 - G.round);
        int net100 = rate100() * Cfg.INCOME_TRUST / 100 - 200 * k;
        return G.rc.getGlobalCheese() + net100 * left / 100;
    }
}
