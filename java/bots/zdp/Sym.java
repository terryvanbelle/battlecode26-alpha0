package zdp;

/**
 * Map memory (one int per tile) and symmetry inference.
 * Symmetry candidates are eliminated only by static features (walls, mines); dirt changes during play.
 */
public final class Sym {
    public static final int ROT = 1, HOR = 2, VER = 4;
    public static final int SEEN = 1, WALL = 2, DIRT = 4, MINE = 8;
    /** A tile under one of our kings' 3x3 bodies (Nav.markKings, Cfg.KING_FLOW): flow paths go around it. */
    public static final int KING = 128;

    public static int[] mem;
    public static int mask = ROT | HOR | VER;
    /** Count of eliminations rejected because they would have emptied the mask (must stay 0). */
    public static int contradictions;

    public static void init() { mem = new int[G.W * G.H]; }

    public static boolean known(int i) { return (mem[i] & SEEN) != 0; }

    /** Record a sensed tile; returns true when the symmetry mask shrank. */
    public static boolean record(int x, int y, boolean wall, boolean dirt, boolean mine) {
        int i = x + y * G.W;
        int v = SEEN | (wall ? WALL : 0) | (dirt ? DIRT : 0) | (mine ? MINE : 0);
        int old = mem[i];
        mem[i] = v | (old & KING);
        if ((old & SEEN) != 0 || mask == ROT || mask == HOR || mask == VER) return false;
        int before = mask;
        int stat = v & (WALL | MINE);
        if ((mask & ROT) != 0) check(stat, G.W - 1 - x, G.H - 1 - y, ROT);
        if ((mask & HOR) != 0) check(stat, x, G.H - 1 - y, HOR);
        if ((mask & VER) != 0) check(stat, G.W - 1 - x, y, VER);
        return mask != before;
    }

    private static void check(int stat, int mx, int my, int s) {
        int m = mem[mx + my * G.W];
        if ((m & SEEN) == 0) return;
        if ((m & (WALL | MINE)) != stat) eliminate(s);
    }

    public static void eliminate(int s) {
        int next = mask & ~s;
        if (next == 0) { contradictions++; return; }
        mask = next;
    }

    /** Apply a mask learned from a teammate. */
    public static void merge(int m) {
        int next = mask & m;
        if (next == 0) { contradictions++; return; }
        mask = next;
    }

    public static int count() { return (mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1); }
}
