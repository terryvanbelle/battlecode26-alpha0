package zdp;

import battlecode.common.Clock;
import battlecode.common.MapLocation;

/** Cfg.SEALED: cheese in a pocket no rat can walk into. A flood fill from the pile over every tile not known to be a
 *  wall (unseen tiles count as open, dirt is dug; with one symmetry left a wall's twin counts as a wall) that runs out
 *  within cap tiles without reaching us has found a sealed pocket. Its tiles, and their twins under a known symmetry,
 *  get Sym.SEAL, and no rat chases cheese there. learn_your_lesson: four mines sit in 5x5 rooms walled all round, and
 *  rats see their cheese through the walls; 5416 of u4e's 6051 cheese-chase turns against s7d (ly1) went to piles in
 *  those rooms, 42% of all its rat turns. sense() only reads verdicts and names a pile to judge (want); the fill runs
 *  at the end of the turn (work) on the bytecodes above RESERVE, pausing and resuming on later turns: a 5x5 room costs
 *  about 9k, and run inside sense() it overran a fighter's turn. */
final class Seal {
    /** Cfg.SEALED (a field, so tests can set it before the first fill). */
    static int cap = Cfg.SEALED;
    /** Cfg.SEAL_SEEN (a field, so tests can set it). */
    static boolean seenOnly = Cfg.SEAL_SEEN;
    static int[] seen, queue;
    static int stamp;
    /** Verdicts (tile index + 1 open, -(tile index + 1) unknown under seenOnly, and the round), trusted OPEN_TTL
     *  rounds: walls seen later can still close the pocket. */
    static int[] openT, openR;
    static int openN;
    /** The pile to judge next (tile index + 1, 0 = none). */
    static int want;
    /** The fill left unfinished (start tile index + 1, 0 = none), its queue head and tail, the round it stopped and
     *  the round it began (dropped after PEND_MAX rounds). */
    static int pend, pendH, pendT, pendRound = -1, pendStart;
    static boolean pendLeak;
    /** sense2 still runs above its own bar after a fill. */
    static final int OPEN_TTL = 20, CACHE = 8, RESERVE = Cfg.SENSE2_LEFT + 500, PEND_MAX = 10;
    static int fills, found, paused, unknown;
    /** Tests: bytecodes left, counting down one per check (-1 = ask the clock). */
    static int fakeLeft = -1;

    static int left() {
        if (fakeLeft >= 0) return fakeLeft--;
        return Clock.getBytecodesLeft() + G.debt;
    }

    /** seenOnly: the pile at tile index s has a fresh unknown verdict (not worth a walk for now). */
    static boolean skipped(int s) {
        if (openT == null) return false;
        for (int k = 0; k < CACHE; k++) if (openT[k] == -(s + 1) && G.round - openR[k] < OPEN_TTL) return true;
        return false;
    }

    /** The verdict on the pile at tile index s: 1 sealed (or skipped under seenOnly), 0 open, -1 not judged. */
    static int verdict(int s) {
        if ((Sym.mem[s] & Sym.SEAL) != 0) return 1;
        if (openT == null) return -1;
        for (int k = 0; k < CACHE; k++) {
            if (G.round - openR[k] >= OPEN_TTL) continue;
            if (openT[k] == s + 1) return 0;
            if (openT[k] == -(s + 1)) return 1;
        }
        return -1;
    }

    /** sense(): the pile at l has no verdict; judge it at the end of the turn unless another is named already. */
    static void want(MapLocation l) { if (want == 0) want = l.x + l.y * G.W + 1; }

    /** End of turn: go on with the paused fill, else judge the named pile. */
    static void work() {
        for (int i = 0; i < 2; i++) {
            int s;
            if (pend != 0 && G.round - pendStart <= PEND_MAX) s = pend - 1;
            else {
                pend = 0;
                if (want == 0) return;
                s = want - 1;
                want = 0;
                if (verdict(s) >= 0) continue;
            }
            if (fill(s) < 0) return;
        }
    }

    /** Judge cheese at l now (tests; the bot names piles with want and judges them in work). */
    static boolean sealed(MapLocation l) {
        int s = l.x + l.y * G.W, v = verdict(s);
        return v >= 0 ? v > 0 : fill(s) > 0;
    }

    /** The fill from tile index s (see the class comment): 1 sealed (or unknown under seenOnly), 0 open, -1 paused or
     *  not started for want of bytecodes. */
    static int fill(int s) {
        int[] mem = Sym.mem;
        int W = G.W, H = G.H;
        // one fill a turn once one has paused (a second would throw away its queue)
        if (pendRound == G.round) return -1;
        int h, t, st;
        boolean leak = false;
        if (pend == s + 1) { h = pendH; t = pendT; st = stamp; leak = pendLeak; }
        else {
            if (left() < RESERVE) return -1;
            // allocated after the check: the engine charges an array's length in one go, past the limit
            if (seen == null) { seen = new int[W * H]; queue = new int[cap + 1]; openT = new int[CACHE]; openR = new int[CACHE]; }
            fills++;
            st = ++stamp;
            seen[s] = st; queue[0] = s;
            h = 0; t = 1;
            pend = s + 1; pendStart = G.round;
        }
        boolean uni = Sym.unique();
        int sym = Sym.mask, me = G.me.x + G.me.y * W, W1 = W - 1, H1 = H - 1;
        boolean fx = sym != Sym.HOR, fy = sym != Sym.VER;
        while (h < t) {
            if (left() < RESERVE) { pendH = h; pendT = t; pendLeak = leak; pendRound = G.round; paused++; return -1; }
            int c = queue[h++], cx = c % W, cy = c / W;
            for (int nx = cx - 1; nx <= cx + 1; nx++) {
                if (nx < 0 || nx > W1) continue;
                int tx = fx ? W1 - nx : nx;
                for (int ny = cy - 1; ny <= cy + 1; ny++) {
                    if (ny < 0 || ny > H1) continue;
                    int n = nx + ny * W;
                    if (seen[n] == st) continue;
                    seen[n] = st;
                    int m = uni ? mem[n] | mem[tx + (fy ? H1 - ny : ny) * W] : mem[n];
                    if ((m & Sym.WALL) != 0) continue;
                    if (seenOnly && (m & Sym.SEEN) == 0) { leak = true; continue; }
                    if (n == me || t >= cap) { pend = 0; openT[openN] = s + 1; openR[openN] = G.round; openN = (openN + 1) % CACHE; return 0; }
                    queue[t++] = n;
                }
            }
        }
        pend = 0;
        if (leak) { unknown++; openT[openN] = -(s + 1); openR[openN] = G.round; openN = (openN + 1) % CACHE; return 1; }
        found++;
        for (int i = 0; i < t; i++) {
            int c = queue[i];
            mem[c] |= Sym.SEAL;
            if (uni) mem[(fx ? W1 - c % W : c % W) + (fy ? H1 - c / W : c / W) * W] |= Sym.SEAL;
        }
        return 1;
    }
}
