package zdp;

import battlecode.common.*;

/**
 * Cfg.GRAB_MICRO: grab-aware movement for the rat war. A rat grabs an adjacent enemy it can see (it faces it) when
 * that enemy cannot see the grabber (the grabber stands in its blind spot) or is weaker. Field replays (3934-3937)
 * showed our rats ending turns where enemies could step into their blind spots: with 0, 1, 2 and 3+ such enemies
 * 0.1%, 6-20%, 8-32% and 14-65% of them were grabbed before their next turn, while one-step grabs went unused.
 *
 * Every reachable end tile (stay or one step) and final facing is scored: a grab there (+GRAB), a bite (+BITE, more
 * when it makes the target grabbable by us or ends its health edge, or kills), and minus THREAT for each enemy that can
 * step next to that tile into our blind spot (or anywhere next to it, when it is healthier). The rat acts when the
 * best option beats standing still by MARGIN; otherwise the normal move runs with the clearly worse steps avoided.
 *
 * Tiles are bits of a 7x7 window around us: bit (ox + 3) + 8 * (oy + 3), column 7 a guard against row wrap; the
 * neighbours of a tile are an 8-bit ring, bit k the neighbour in G.DIRS direction k.
 */
public final class GrabMicro {
    static final int GRAB = 100, THREAT = Cfg.GRAB_THREAT, BITE = Cfg.GRAB_BITE, FLIP = BITE + 15, KILL = BITE + 40,
                     MARGIN = Cfg.GRAB_MARGIN;
    static final int[] DX = {0, 1, 1, 1, 0, -1, -1, -1}, DY = {1, 1, 0, -1, -1, -1, 0, 1};
    static final long CBIT = 1L << 27;
    /** The 3x3 around our tile, and its 8 neighbours. */
    static final long K3 = (7L << 18) | (7L << 26) | (7L << 34), ADJ = K3 & ~CBIT;
    /** Tiles within Chebyshev 2 of us: the only ones next to an end tile. */
    static final long INNER = 0x00003E3E3E3E3E00L;
    /** The nearest MAXE enemy babies within Chebyshev 3 (ties: the mirror-invariant key). */
    static final int MAXE = 6;
    static final int[] ex = new int[MAXE], ey = new int[MAXE], ehp = new int[MAXE], edir = new int[MAXE];
    static final boolean[] ecan = new boolean[MAXE];
    static final RobotInfo[] er = new RobotInfo[MAXE];
    static final int[] ech = new int[MAXE], ekey = new int[MAXE];
    /** The enemy babies within Chebyshev 3 (at most the 7x7 window), their Chebyshev distances and keys. */
    static final RobotInfo[] CR = new RobotInfo[49];
    static final int[] CC = new int[49], CK = new int[49];
    /** Per end tile: threat facings of the enemies that reach it, and of the adjacent ones (also after our bite),
     *  the facings that see each adjacent one, whether we can grab it, its key and index. */
    static final int[] atf = new int[MAXE], atfb = new int[MAXE], av = new int[MAXE], ai = new int[MAXE];
    /** Per enemy: the ring bits its facing sees (all, when it has none). */
    static final int[] eseen = new int[MAXE];
    /** Tie keys of the eight facings (our current one first) and steps; per facing, the adjacent enemy grabbed and
     *  bitten (-1 none). */
    static final int[] FK = new int[8], DK = new int[8], PG = new int[8], PB = new int[8];
    /** SAFE[r]: the facings (bits) whose cone covers every neighbour in ring mask r (bit k = the neighbour in G.DIRS
     *  direction k). NEAR[(dx + 4) * 9 + dy + 4]: the ring tiles within Chebyshev 1 of offset (dx, dy).
     *  RING9[v]: the ring mask of a 3x3 block, bit (dx + 1) + 3 * (dy + 1). DIRK[(dx + 1) * 3 + dy + 1]: the direction
     *  index of a unit offset (8 = none). Strings cost no initializer; BotTest checks them against their definitions. */
    static final String SAFE = "\377\203\007\003\016\002\006\002\034\000\004\000\014\000\004\000\070\000\000\000\010\000\000\000\030\000\000\000\010\000\000\000\160\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\060\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\340\200\000\000\000\000\000\000\000\000\000\000\000\000\000\000\040\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\140\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\040\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\301\201\001\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\100\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\300\200\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\100\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000";
    static final String NEAR = "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\040\140\340\300\200\000\000\000\000\060\160\361\301\201\000\000\000\000\070\174\377\307\203\000\000\000\000\030\034\037\007\003\000\000\000\000\010\014\016\006\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000";
    static final String RING9 = "\000\040\020\060\010\050\030\070\100\140\120\160\110\150\130\170\000\040\020\060\010\050\030\070\100\140\120\160\110\150\130\170\004\044\024\064\014\054\034\074\104\144\124\164\114\154\134\174\004\044\024\064\014\054\034\074\104\144\124\164\114\154\134\174\200\240\220\260\210\250\230\270\300\340\320\360\310\350\330\370\200\240\220\260\210\250\230\270\300\340\320\360\310\350\330\370\204\244\224\264\214\254\234\274\304\344\324\364\314\354\334\374\204\244\224\264\214\254\234\274\304\344\324\364\314\354\334\374\001\041\021\061\011\051\031\071\101\141\121\161\111\151\131\171\001\041\021\061\011\051\031\071\101\141\121\161\111\151\131\171\005\045\025\065\015\055\035\075\105\145\125\165\115\155\135\175\005\045\025\065\015\055\035\075\105\145\125\165\115\155\135\175\201\241\221\261\211\251\231\271\301\341\321\361\311\351\331\371\201\241\221\261\211\251\231\271\301\341\321\361\311\351\331\371\205\245\225\265\215\255\235\275\305\345\325\365\315\355\335\375\205\245\225\265\215\255\235\275\305\345\325\365\315\355\335\375\002\042\022\062\012\052\032\072\102\142\122\162\112\152\132\172\002\042\022\062\012\052\032\072\102\142\122\162\112\152\132\172\006\046\026\066\016\056\036\076\106\146\126\166\116\156\136\176\006\046\026\066\016\056\036\076\106\146\126\166\116\156\136\176\202\242\222\262\212\252\232\272\302\342\322\362\312\352\332\372\202\242\222\262\212\252\232\272\302\342\322\362\312\352\332\372\206\246\226\266\216\256\236\276\306\346\326\366\316\356\336\376\206\246\226\266\216\256\236\276\306\346\326\366\316\356\336\376\003\043\023\063\013\053\033\073\103\143\123\163\113\153\133\173\003\043\023\063\013\053\033\073\103\143\123\163\113\153\133\173\007\047\027\067\017\057\037\077\107\147\127\167\117\157\137\177\007\047\027\067\017\057\037\077\107\147\127\167\117\157\137\177\203\243\223\263\213\253\233\273\303\343\323\363\313\353\333\373\203\243\223\263\213\253\233\273\303\343\323\363\313\353\333\373\207\247\227\267\217\257\237\277\307\347\327\367\317\357\337\377\207\247\227\267\217\257\237\277\307\347\327\367\317\357\337\377";
    static final String DIRK = "\005\006\007\004\010\000\003\002\001";
    /** The best option: end tile (0-7 a step in G.DIRS order, 8 stay), facing, the enemy grabbed or bitten (-1). */
    static int bestD, bestF, grabI, biteI, bestS, curS;
    /** Best score per end tile (Integer.MIN_VALUE: not reachable). */
    static final int[] dirBest = new int[9];
    /** The round the micro took this turn's movement (Baby.moveDecision then stands down). */
    public static int round = -1;
    public static int acts, steps, grabs, avoids;

    static int b(int ox, int oy) { return (ox + 3) + 8 * (oy + 3); }

    static long bit(int ox, int oy) { return 1L << b(ox, oy); }

    static long shift(long m, int ox, int oy) {
        int s = ox + 8 * oy;
        return s >= 0 ? m << s : m >>> -s;
    }

    /** Ring bits seen by facing f (or the facings that see ring tile f): f and its two neighbours. */
    static int arc(int f) { return (1 << ((f + 7) & 7)) | (1 << f) | (1 << ((f + 1) & 7)); }

    /** The 3x3 block of window bitboard m around offset (tx, ty) as 9 bits, bit (dx + 1) + 3 * (dy + 1). */
    static int block(long m, int tx, int ty) {
        int s = b(tx - 1, ty - 1);
        return (int) (m >>> s & 7) | (int) (m >>> (s + 8) & 7) << 3 | (int) (m >>> (s + 16) & 7) << 6;
    }

    /** The three tiles next to our tile that facing f (G.DIRS index) sees: ahead and the two front diagonals. */
    static long vis(int f) {
        int l = (f + 7) & 7, r = (f + 1) & 7;
        return bit(DX[f], DY[f]) | bit(DX[l], DY[l]) | bit(DX[r], DY[r]);
    }

    /** Whether a rat at offset (x, y) facing f (G.DIRS index, -1 for none) sees offset (tx, ty): the engine's
     *  90-degree cone of radius squared 20 (Baby.sees). */
    static boolean sees(int x, int y, int f, int tx, int ty) {
        int dx = tx - x, dy = ty - y;
        if (f < 0) return true;
        int fx = DX[f], fy = DY[f];
        return dx * dx + dy * dy <= 20 && dx * fx + dy * fy >= Math.abs(dx * fy - dy * fx);
    }

    /** Mirror-invariant tie key of offset (dx, dy) seen from a tile whose doubled centre offset is (px, py): the cross
     *  product signed by the side chirality, then the dot product (G.tieFirst's order). Larger ranks first. */
    static int key(int px, int py, int chir, int dx, int dy) {
        return (px * dy - py * dx) * chir * 4096 + px * dx + py * dy;
    }

    /**
     * Score every end tile and facing. Enemies are offsets from us with health, facing (-1 none) and whether they can
     * grab (not carrying). free: window tiles an enemy may step onto (walls, dirt, robots and our tile cleared).
     * moveMask: bit d for each legal step d (stay is always legal). Sets bestD/bestF/grabI/biteI/bestS, curS (stay,
     * current facing) and dirBest; returns bestS. Per end tile, each enemy that can reach a tile next to it gets an
     * 8-bit mask of the facings that leave it a threat (SAFE: the facings whose cone covers every tile it reaches),
     * so a facing costs a few bit tests per enemy (the bitboard loops of the first version overran: 58 turns in one
     * tiny game, gm1).
     */
    static int evaluate(int myHp, int cur, boolean turnOk, int moveMask, boolean actOk, int n, long free,
                        int px, int py, int chir) {
        long fm = free | CBIT;
        bestS = Integer.MIN_VALUE; bestD = 8; bestF = cur; grabI = -1; biteI = -1; curS = 0;
        int bestKey = 0, bestFKey = 0;
        int[] fk = FK, dk = DK, pg = PG, pb = PB;
        for (int f = 0; f < 8; f++) { dk[f] = key(px, py, chir, DX[f], DY[f]); fk[f] = f == cur ? Integer.MAX_VALUE : dk[f]; }
        for (int i = 0; i < n; i++) { eseen[i] = edir[i] < 0 ? 255 : arc(edir[i]); ekey[i] = key(px, py, chir, ex[i], ey[i]); }
        int allowed = turnOk ? 255 : 1 << cur;
        for (int d = 8; d >= 0; d--) {
            dirBest[d] = Integer.MIN_VALUE;
            if (d < 8 && (moveMask & (1 << d)) == 0) continue;
            int tx = d < 8 ? DX[d] : 0, ty = d < 8 ? DY[d] : 0;
            // s0, s1, s2: bit-sliced count, per facing, of the enemies that stay a threat with it
            int na = 0, fr = -1, s0 = 0, s1 = 0, s2 = 0;
            for (int i = 0; i < n; i++) {
                int dx = ex[i] - tx, dy = ey[i] - ty;
                if (dx < -2 || dx > 2 || dy < -2 || dy > 2) continue;
                if (fr < 0) fr = RING9.charAt(block(fm, tx, ty));
                boolean adj = dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1;
                int k = adj ? DIRK.charAt((dx + 1) * 3 + dy + 1) : 0;
                // the tiles next to t it can stand on after one step (its own, when next to t)
                int r = NEAR.charAt((dx + 4) * 9 + dy + 4) & (adj ? fr | 1 << k : fr);
                int h = ehp[i];
                int tf = !ecan[i] || r == 0 ? 0 : h > myHp ? 255 : ~SAFE.charAt(r) & 255;
                if (tf != 0) { int c0 = s0 & tf; s0 ^= tf; int c1 = s1 & c0; s1 ^= c0; s2 |= c1; }
                if (adj) {
                    if (na == 0) for (int f = 0; f < 8; f++) { pg[f] = -1; pb[f] = -1; }
                    atf[na] = tf;
                    atfb[na] = tf == 0 || h <= 10 ? 0 : h - 10 > myHp ? 255 : ~SAFE.charAt(r) & 255;
                    ai[na] = i;
                    // the facings that see it (k and its two neighbours) take it when it is their best grab or bite;
                    // equal choices fall to the mirror-invariant key of its tile
                    if (actOk && (h < myHp || (eseen[i] >> ((k + 4) & 7) & 1) == 0)) {
                        int x = GRAB + h / 10;
                        av[na] = x;
                        for (int q = 7; q <= 9; q++) {
                            int f = (k + q) & 7, o = pg[f];
                            if (o < 0 || x > av[o] || x == av[o] && ekey[i] > ekey[ai[o]]) pg[f] = na;
                        }
                    } else if (actOk) {
                        int x = h <= 10 ? KILL : h >= myHp && h - 10 < myHp ? FLIP : BITE;
                        av[na] = x;
                        for (int q = 7; q <= 9; q++) {
                            int f = (k + q) & 7, o = pb[f];
                            if (o < 0 || x > av[o] || x == av[o] && (h < ehp[ai[o]] || h == ehp[ai[o]] && ekey[i] > ekey[ai[o]]))
                                pb[f] = na;
                        }
                    }
                    na++;
                }
            }
            int tKey = d == 8 ? Integer.MAX_VALUE : dk[d];
            boolean strafe = d < 8 && cur != d;
            if (na == 0) {
                // no grab or bite at t: the facings with the fewest threats, then no strafe, our facing, the key
                int m = allowed & ~(s0 | s1 | s2), c = 0;
                if (m == 0) { m = allowed & s0 & ~s1 & ~s2; c = 1; }
                if (m == 0) { m = allowed & ~s0 & s1 & ~s2; c = 2; }
                if (m == 0) { m = allowed & s0 & s1 & ~s2; c = 3; }
                for (int q = 4; m == 0; q++) { m = allowed & s2 & ((q & 1) != 0 ? s0 : ~s0) & ((q & 2) != 0 ? s1 : ~s1); c = q; }
                int f, s;
                if (strafe && (m >> d & 1) != 0) { f = d; s = -THREAT * c; }
                else {
                    s = -THREAT * c - (strafe ? 1 : 0);
                    f = -1;
                    for (int g = 0; g < 8; g++) if ((m >> g & 1) != 0 && (f < 0 || fk[g] > fk[f])) f = g;
                }
                if (d == 8) curS = -THREAT * ((s0 >> cur & 1) + 2 * (s1 >> cur & 1) + 4 * (s2 >> cur & 1));
                dirBest[d] = s;
                if (s > bestS || s == bestS && (tKey > bestKey || tKey == bestKey && fk[f] > bestFKey)) {
                    bestS = s; bestD = d; bestF = f; bestKey = tKey; bestFKey = fk[f]; grabI = -1; biteI = -1;
                }
                continue;
            }
            for (int f = turnOk ? 0 : cur, f1 = turnOk ? 7 : cur; f <= f1; f++) {
                int cnt = (s0 >> f & 1) + 2 * (s1 >> f & 1) + 4 * (s2 >> f & 1), v = 0;
                int g = pg[f], bi = -1;
                if (g >= 0) { v = av[g]; cnt -= atf[g] >> f & 1; }
                else if ((bi = pb[f]) >= 0) { v = av[bi]; cnt += (atfb[bi] >> f & 1) - (atf[bi] >> f & 1); }
                int s = v - THREAT * cnt - (strafe && f != d ? 1 : 0);
                if (d == 8 && f == cur) curS = s;
                if (s > dirBest[d]) dirBest[d] = s;
                if (s > bestS || s == bestS && (tKey > bestKey || tKey == bestKey && fk[f] > bestFKey)) {
                    bestS = s; bestD = d; bestF = f; bestKey = tKey; bestFKey = fk[f];
                    grabI = g >= 0 ? ai[g] : -1;
                    biteI = bi >= 0 ? ai[bi] : -1;
                }
            }
        }
        return bestS;
    }

    /**
     * Before this turn's bites: take the movement (and the action, for a grab or bite) when grab safety or a grab
     * says so; true when it acted (the caller skips its pre-move bite and Baby.moveDecision stands down). Otherwise
     * the steps clearly worse than staying are added to Nav.avoid.
     */
    static boolean act(RobotInfo[] robots) throws GameActionException {
        RobotController rc = G.rc;
        if (rc.getCarrying() != null || Flow.left() < Cfg.GRAB_BC) return false;
        MapLocation me = rc.getLocation();
        int mx = me.x, my = me.y, px = 2 * mx - G.cx2, py = 2 * my - G.cy2, chir = G.chir;
        int m = 0, minC = 9;
        long occ = 0;
        Team them = G.them;
        RobotInfo[] cr = CR;
        int[] cc = CC, ck = CK;
        for (RobotInfo r : robots) {
            MapLocation l = r.location;
            int ox = l.x - mx, oy = l.y - my;
            if (ox < -3 || ox > 3 || oy < -3 || oy > 3) {
                if (r.type == UnitType.RAT_KING && ox >= -4 && ox <= 4 && oy >= -4 && oy <= 4) occ |= shift(K3, ox, oy);
                continue;
            }
            if (r.type == UnitType.RAT_KING) {
                // an enemy king next to us is bitten, not danced around
                if (r.team == them && ox >= -2 && ox <= 2 && oy >= -2 && oy <= 2) return false;
                occ |= shift(K3, ox, oy);
            } else if (r.team == Team.NEUTRAL) {
                occ |= shift(K3 & ((3L << 27) | (3L << 35)), ox, oy);
            } else {
                occ |= 1L << (ox + 27 + 8 * oy);
                if (r.team == them) {
                    int c = ox < 0 ? -ox : ox, c2 = oy < 0 ? -oy : oy;
                    if (c2 > c) c = c2;
                    if (c < minC) minC = c;
                    cr[m] = r; cc[m] = c; ck[m++] = (px * oy - py * ox) * chir * 4096 + px * ox + py * oy;
                }
            }
        }
        // no enemy within Chebyshev 2: nothing can be grabbed and staying is safe (skip the cost; the steps toward
        // a Chebyshev-3 enemy keep the normal move)
        if (minC > 2) return false;
        // the nearest MAXE by (Chebyshev, key): enemy order does not change the scores, only which ones are kept
        int n = m < MAXE ? m : MAXE;
        for (int j = 0; j < n; j++) {
            if (m > MAXE) {
                int b = j;
                for (int q = j + 1; q < m; q++) if (cc[q] < cc[b] || cc[q] == cc[b] && ck[q] > ck[b]) b = q;
                RobotInfo t = cr[b]; cr[b] = cr[j]; cr[j] = t;
                int v = cc[b]; cc[b] = cc[j]; cc[j] = v;
                v = ck[b]; ck[b] = ck[j]; ck[j] = v;
            }
            RobotInfo r = cr[j];
            ex[j] = r.location.x - mx; ey[j] = r.location.y - my; ehp[j] = r.health; er[j] = r; ech[j] = cc[j];
            Direction f = r.direction;
            edir[j] = f == null || f == Direction.CENTER ? -1 : f.ordinal();
            ecan[j] = r.carryingRobot == null;
        }
        long free = 0;
        int W = G.W, H = G.H, x0 = mx - 2, x1 = mx + 2;
        int[] mem = Sym.mem;
        for (int oy = -2; oy <= 2; oy++) {
            int y = my + oy;
            if (y < 0 || y >= H) continue;
            int base = y * W, sh = 27 + 8 * oy - mx;
            for (int x = x0 < 0 ? 0 : x0, xe = x1 < W ? x1 : W - 1; x <= xe; x++)
                if ((mem[base + x] & (Sym.WALL | Sym.DIRT)) == 0) free |= 1L << (sh + x);
        }
        free &= ~occ & ~CBIT;
        boolean moveOk = rc.isMovementReady();
        int moveMask = 0;
        if (moveOk) for (int d = 0; d < 8; d++) if ((Nav.avoid & (1 << d)) == 0 && rc.canMove(G.DIRS[d])) moveMask |= 1 << d;
        int cur = rc.getDirection().ordinal();
        evaluate(rc.getHealth(), cur, rc.isTurningReady(), moveMask, rc.isActionReady(), n, free, px, py, chir);
        if (grabI < 0 && bestS < curS + MARGIN) {
            for (int d = 0; d < 8; d++)
                if ((moveMask & (1 << d)) != 0 && dirBest[d] <= curS - MARGIN) { Nav.avoid |= 1 << d; avoids++; }
            return false;
        }
        execute();
        return true;
    }

    static void execute() throws GameActionException {
        RobotController rc = G.rc;
        Direction f = G.DIRS[bestF];
        Direction cur = rc.getDirection();
        if (bestD < 8) {
            Direction d = G.DIRS[bestD];
            if (cur != d && f == d && rc.canTurn(d)) { rc.turn(d); cur = d; }
            if (rc.canMove(d)) { rc.move(d); steps++; }
        }
        if (cur != f && rc.canTurn(f)) rc.turn(f);
        G.me = rc.getLocation();
        G.facing = rc.getDirection();
        if (grabI >= 0) {
            MapLocation l = er[grabI].location;
            if (rc.canCarryRat(l)) {
                rc.carryRat(l);
                Baby.carryRound = G.round;
                Baby.naps++;
                grabs++;
            }
        } else if (biteI >= 0) {
            MapLocation l = er[biteI].location;
            if (rc.canAttack(l)) {
                int c = Baby.finishCheese(er[biteI].health, rc.getGlobalCheese());
                if (c > 0 && rc.canAttack(l, c)) rc.attack(l, c); else rc.attack(l);
                Baby.bitesRat++;
            }
        }
        round = G.round;
        acts++;
        Baby.why = 'm';
    }
}
