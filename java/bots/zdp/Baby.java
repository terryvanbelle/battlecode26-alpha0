package zdp;

import battlecode.common.*;

/** Baby rat: gather and deliver cheese, fight cats, lay cat traps, explore. */
public final class Baby {
    public static final int GATHER = 0, FIGHT = 1, ATTACK = 2;
    public static int role = -1;
    static int birthRound;
    /** The round the net's R_ATTACK last made this rat an attacker (-10000: never; a D feature). */
    static int ekSeen = -10000;
    /** Cfg.STALL_CAT: the tile this attacker has stayed near since stallSince, and whether it hunts cats now. */
    static MapLocation stallAt;
    static int stallSince, catHunters;
    static boolean catHunt;
    static MapLocation home;          // spawning king's location at birth
    static int fu = 1, fv = 1;        // side frame (signs of our king's offset from centre)

    static MapLocation cheeseTarget;  // last seen cheese tile
    static MapLocation mineTarget;    // sticky assigned mine
    static MapLocation exploreTarget;
    static MapLocation catSeen;       // bottom-left of nearest cat sensed this turn
    static RobotInfo cat;
    static int catSeenRound = -100;
    /** AIM_LEFT: aimCarry's bytecode floor (a field, so tests can set it). */
    static int aimLeft = Cfg.AIM_LEFT;

    // mines this rat has sensed (to report to kings)
    static int[] myMines = new int[24];
    static int lowBc;   // turns cut short for lack of bytecodes
    static int myMineCount;

    public static int delivered, bitesCat, bitesRat, trapsPlaced, picked;

    public static void turn() throws GameActionException {
        RobotController rc = G.rc;
        D.point = 0;
        D.point2 = 0;
        if (role < 0) {
            born();
            if (D.ON) {   // BABY_ROLE at birth
                D.babyRole(0, -1);
                if (D.NET) { int k = D.pick(D.BABY_ROLE, null); if (k >= 0) roleExec(k); }
                D.lab = D.roleLab();
                D.emit();
            }
        }
        int roleAt0 = role;
        Comms.readKings();
        Nav.markKings();
        Comms.readMines();
        int s = Comms.read(Comms.SYM);
        if (s != 0) Sym.merge(s);

        Debug.mark(1);
        sense();
        RobotInfo[] robots = rc.senseNearbyRobots(-1, null);
        seen = robots;
        RobotInfo enemyKing = null;
        cat = null;
        int bestCat = Integer.MAX_VALUE;
        for (RobotInfo r : robots) {
            if (r.team == Team.NEUTRAL) {
                if (G.coop) CatLane.see(r.location, G.round);
                int d = catDist4(r.location, G.me);
                if (d < bestCat) { bestCat = d; cat = r; }
            } else if (r.team == G.them && r.type == UnitType.RAT_KING) enemyKing = r;
        }
        if (cat != null) { catSeen = cat.location; catSeenRound = G.round; }

        Debug.mark(2);
        if (role != ATTACK) pickUp();
        Nav.avoid = cat != null ? laneMask(cat) : 0;
        // a newborn's first turn pays for array allocation and a full first sense: skip optional work when short
        if (!G.coop && Flow.left() > Cfg.BC_OPTIONAL) {
            TrapRisk.observe(robots);
        }

        Debug.mark(3);
        eKing = enemyKing;
        if (D.ON && (role != roleAt0 || G.round > birthRound && (G.round - birthRound) % 10 == 0)) {   // BABY_ROLE review
            D.babyRole(1, roleAt0);
            if (D.NET) { int k = D.pick(D.BABY_ROLE, null); if (k >= 0) roleExec(k); }
            D.lab = D.roleLab();
            D.emit();
        }
        nearestEnemy = null;
        hunting = false;
        if (!G.coop) {
            int bd = Integer.MAX_VALUE;
            for (RobotInfo r : robots) {
                if (r.team != G.them || r.type == UnitType.RAT_KING) continue;
                int d = G.me.distanceSquaredTo(r.location);
                if (d < bd) { bd = d; nearestEnemy = r; }
            }
            // fighters close on a rat they can grab (lower health than ours: trapped or bitten) before anything else
            if (rc.getCarrying() == null) {
                RobotInfo p = prey(robots, rc.getHealth());
                // gatherers only grab prey that is close; fighters hunt it across their vision
                if (p != null && (role != GATHER || G.me.distanceSquaredTo(p.location) <= 8)) {
                    nearestEnemy = p;
                    hunting = true;
                }
            }
        }
        Debug.mark(30);
        if (Flow.left() < Cfg.BC_BAIL) { lowBc++; return; }
        Nav.guard = null;
        if (!G.coop && nearestEnemy != null && rc.getCarrying() == null
                && G.me.distanceSquaredTo(nearestEnemy.location) <= Cfg.GUARD_DIST2) Nav.guard = nearestEnemy.location;
        boolean engaged = nearestEnemy != null && G.me.distanceSquaredTo(nearestEnemy.location) <= 20;
        // trap risk: off while hunting prey or engaged (an enemy within dist2 20)
        if (!G.coop && Nav.stall < 8 && !hunting && !engaged) Nav.avoid |= TrapRisk.mask(G.me);
        Debug.mark(31);
        if (D.ON && rc.isActionReady()) D.babyAct(D.BABY_ACT_PRE, robots, false, '-');   // BABY_ACT_PRE opens
        boolean netAct = false;
        if (D.NET && D.point == D.BABY_ACT_PRE) {
            int k = D.pick(D.BABY_ACT_PRE, null);
            netAct = k >= 0 && actExec(k, robots);
            if (netAct) D.lab = k;
            G.facing = rc.getDirection();
            if (rc.getType() == UnitType.RAT_KING) { D.emit(); return; }
        }
        // 0. a carried enemy is thrown as soon as we can act
        if (!netAct) {
        if (rc.getCarrying() != null) { if (rc.getCarrying().team == G.us) dropAlly(); else tryThrow(robots); }
        }
        // 1. bites: enemy rats once cooperation is over (they shoot back; cats wait), then cats.
        //    Attackers open the war themselves, but only on the enemy king.
        Debug.mark(4);
        boolean hostile = hostile();
        if (!netAct) {
        if (hostile) tryCarry(robots);
        Debug.mark(5);
        if (hostile) biteEnemies(robots);
        else if (role == ATTACK && enemyKing != null) biteEnemies(new RobotInfo[]{enemyKing});
        if (cat != null) tryBiteCat(cat);
        // 2. cat trap ahead of an approaching cat
        if (cat != null) tryCatTrap(cat);
        }

        Debug.mark(6);
        if (D.ON) D.babyComms();   // BABY_COMMS opens (secondary slot)
        int netC = -1;
        if (D.NET && D.point2 == D.BABY_COMMS) {
            // the net's squeak goes first; zdb's cashTick/report still run (listening, cash state) and their squeak
            // then fails (one squeak a turn)
            int k = D.pick2(D.BABY_COMMS, null);
            if (k >= 0 && commsExec(k)) netC = k;
        }
        // ledger (K = 20 with G.settle): the hook's bytecodes outside the two reads (u6x runs only the report() call)
        int cd0 = G.debt, cb0 = G.bc();
        boolean cashSq = cashTick();
        G.settle(cd0, cb0, 20);
        if (cashSq) D.lab2 = cashLead ? D.C_CASH_LEAD : D.C_CASH_RELAY;
        if (!cashSq) report();
        if (netC >= 0) D.lab2 = netC;
        D.emit2();

        // 3. movement
        Debug.mark(7);
        MapLocation before = rc.getLocation();
        // Cfg.CASH_BANK: the rat on a rescue site promotes once it can, movement-ready or not (heavy rats rarely are)
        int cd1 = G.debt, cb1 = G.bc();   // ledger (K = 18 with G.settle)
        if (!netAct && cashPromote(G.me)) {
            D.act(D.A_CASH_KING);
            D.emit();
            return;
        }
        G.settle(cd1, cb1, 18);
        D.emit();   // BABY_ACT_PRE closes
        if (rc.isMovementReady()) {
            int cdD = G.debt, cbD = G.bc();   // ledger (K 9): decision record, BABY_MOVE features
            D.babyMove();
            G.debt = cdD + G.bc() - cbD + 9;
            if (D.NET) netMove(); else moveDecision();
            int cdE = G.debt, cbE = G.bc();   // ledger (K 9)
            D.emit();
            G.debt = cdE + G.bc() - cbE + 9;
        }
        boolean moved = !rc.getLocation().equals(before);
        if (rc.getType() == UnitType.RAT_KING) return;   // just promoted: the king code runs from next turn
        Debug.mark(20);
        pickUp();
        if (D.ON && rc.isActionReady()) D.babyAct(D.BABY_ACT_POST, rc.senseNearbyRobots(-1, null), moved, moved ? why : '-');
        boolean netAct2 = false;
        if (D.NET && D.point == D.BABY_ACT_POST) {
            boolean[] ok = new boolean[D.BABY_ACT_POST_OPTIONS];
            for (int i = 0; i < ok.length; i++) ok[i] = i != D.A_CASH_KING || cashSite != null && rc.getLocation().equals(cashSite);
            int k = D.pick(D.BABY_ACT_POST, ok);
            netAct2 = k >= 0 && actExec(k, null);
            if (netAct2) D.lab = k;
            G.facing = rc.getDirection();
            if (rc.getType() == UnitType.RAT_KING) { D.emit(); return; }
        }
        if (!netAct2) {
        if (rc.getCarrying() != null && moved) {
            if (rc.getCarrying().team == G.us) dropAlly();
            else if (Clock.getBytecodesLeft() + (G.debt += 8) - 6 > Cfg.THROW_BC) tryThrow(rc.senseNearbyRobots(-1, null));
        }
        Debug.mark(21);
        if (hostile) tryCarry(rc.senseNearbyRobots(2, G.them));
        Debug.mark(22);
        if (hostile) biteEnemies(rc.senseNearbyRobots(2, G.them));
        else if (role == ATTACK) {
            RobotInfo[] near = rc.senseNearbyRobots(8, G.them);
            for (RobotInfo r : near) if (r.type == UnitType.RAT_KING) biteEnemies(new RobotInfo[]{r});
        }
        if (cat != null) tryBiteCat(cat);
        if (G.coop) layLane();
        Debug.mark(8);
        tryDeliver();
        }
        D.emit();   // BABY_ACT_POST closes
        if (D.ON && !G.coop && rc.isTurningReady() && rc.getCarrying() == null) D.babyFace(moved);
        if (D.NET && D.point == D.BABY_FACE) {
            int k = D.pick(D.BABY_FACE, null);
            if (k >= 0 && faceExec(k)) D.lab = k; else guardTurn();
        } else guardTurn();
        D.emit();   // BABY_FACE closes
        Debug.mark(9);
        sense2();
    }

    /** Whether this rat bites and grabs enemy rats now: after cooperation, or a RUSH_BREAK attacker at any time. */
    static boolean hostile() {
        return !G.coop || role == ATTACK || G.round >= Cfg.BREAK_ROUND;
    }

    /** End the turn facing the nearest enemy rat close enough to grab us next round. */
    static void guardTurn() throws GameActionException {
        RobotController rc = G.rc;
        if (G.coop || !rc.isTurningReady() || rc.getCarrying() != null) return;
        MapLocation t = seenThreat();
        if (t != null) D.face(D.F_SEEN);
        else {
            // nothing in view: an enemy we saw at the start of the turn may be behind us
            t = memoryThreat();
            if (t != null) D.face(D.F_MEMORY);
        }
        if (t == null) { t = Nav.guard; if (t != null) D.face(D.F_GUARD); }
        if (t == null) return;
        faceTo(t);
    }

    /** The nearest enemy baby in view within dist2 GUARD_DIST2, or null. */
    static MapLocation seenThreat() throws GameActionException {
        MapLocation t = null;
        int bd = Integer.MAX_VALUE;
        for (RobotInfo r : G.rc.senseNearbyRobots(Cfg.GUARD_DIST2, G.them)) {
            if (r.type != UnitType.BABY_RAT) continue;
            int d = G.me.distanceSquaredTo(r.location);
            if (d < bd) { bd = d; t = r.location; }
        }
        return t;
    }

    /** The nearest enemy baby seen at the start of the turn within dist2 GUARD_DIST2 of us now, or null. */
    static MapLocation memoryThreat() {
        MapLocation t = null;
        int bd = Integer.MAX_VALUE;
        for (RobotInfo r : seen) {
            if (r.team != G.them || r.type != UnitType.BABY_RAT) continue;
            int d = G.me.distanceSquaredTo(r.location);
            if (d <= Cfg.GUARD_DIST2 && d < bd) { bd = d; t = r.location; }
        }
        return t;
    }

    /** Cfg.STALL_CAT: once this attacker has stayed within Chebyshev 3 of one tile for STALL_CAT rounds it hunts cats:
     *  it walks to a cat in view, else the cat last known, else the centre (centreTile). False while it has not. */
    static boolean stallCat() throws GameActionException {
        if (!catHunt) {
            if (stallAt == null || G.cheb(G.me, stallAt) > 3) { stallAt = G.me; stallSince = G.round; return false; }
            if (G.round - stallSince < Cfg.STALL_CAT) return false;
            catHunt = true;
            catHunters++;
        }
        if (cat != null && approachCat(cat)) return true;
        MapLocation ct = knownCat();
        if (ct == null) ct = centreTile(home, G.cx2, G.cy2);
        Nav.goTo(ct);
        return true;
    }

    /** The map's centre tile on the side of each centre line that holds h (doubled centre cx2, cy2): mirror images of h
     *  get mirror-image tiles under every symmetry. */
    static MapLocation centreTile(MapLocation h, int cx2, int cy2) {
        return new MapLocation((cx2 + (2 * h.x < cx2 ? 0 : 1)) / 2, (cy2 + (2 * h.y < cy2 ? 0 : 1)) / 2);
    }

    static void born() {
        RobotController rc = G.rc;
        birthRound = G.round;
        int cost = rc.getCurrentRatCost();
        int idx = (cost - 10) / 10 * 4; // approximate spawn index from the cost step
        role = GATHER;
        if (idx >= Cfg.GATHER_FIRST && (G.round % Cfg.FIGHT_EVERY) == 0) role = FIGHT;
        if (rusher(idx, G.round)) role = ATTACK;
        Comms.readKings();
        home = Comms.nearestKing(G.me);
        if (home == null) home = G.me;
        int px = 2 * home.x - G.cx2, py = 2 * home.y - G.cy2;
        fu = px >= 0 ? 1 : -1;
        fv = py >= 0 ? 1 : -1;
        // mirror-invariant seed: round and the unsigned offset from the king
        int seed = birthSeed(G.round, G.me, home);
        G.seed(seed);
        slot = (seed * 0x9E3779B1) >>> 8;
        scout = (slot >>> 4) % Cfg.SCOUT_EVERY == 0;
    }

    /** Cfg.RUSH: a baby born with idx live babies in round r attacks the enemy king. */
    static boolean rusher(int idx, int r) {
        return idx >= Cfg.RUSH_FROM && idx < Cfg.RUSH_TO && r < Cfg.RUSH_UNTIL;
    }

    /** A rat's birth seed: its round and unsigned offset from the king that spawned it, the same for its mirror twin. */
    static int birthSeed(int round, MapLocation me, MapLocation home) {
        return round * 977 + Math.abs(me.x - home.x) * 31 + Math.abs(me.y - home.y) * 7 + 1;
    }

    /** 4 x squared distance from a tile to a cat's 2x2 centre. */
    static int catDist4(MapLocation catBL, MapLocation p) {
        int dx = 2 * p.x - (2 * catBL.x + 1), dy = 2 * p.y - (2 * catBL.y + 1);
        return dx * dx + dy * dy;
    }

    static boolean inCatBody(MapLocation catBL, MapLocation p) {
        return p.x >= catBL.x && p.x <= catBL.x + 1 && p.y >= catBL.y && p.y <= catBL.y + 1;
    }

    /** Bit d set when stepping in DIRS[d] lands in the cat's next two footprints along its facing. */
    static int laneMask(RobotInfo c) {
        Direction cd = c.direction;
        int mask = 0;
        MapLocation n1 = c.location.add(cd), n2 = n1.add(cd);
        for (int i = 0; i < 8; i++) {
            MapLocation p = G.me.add(G.DIRS[i]);
            if (inCatBody(n1, p) || inCatBody(n2, p) || inCatBody(c.location, p)) mask |= 1 << i;
        }
        return mask;
    }

    static boolean inLane(RobotInfo c, MapLocation p) {
        MapLocation n1 = c.location.add(c.direction), n2 = n1.add(c.direction);
        return inCatBody(n1, p) || inCatBody(n2, p);
    }

    static void sense() throws GameActionException {
        MapInfo[] infos = G.rc.senseNearbyMapInfos();
        MapLocation bestCheese = null;
        int bd = Integer.MAX_VALUE;
        for (MapInfo mi : infos) {
            MapLocation l = mi.getMapLocation();
            boolean mine = mi.hasCheeseMine();
            Sym.record(l.x, l.y, mi.isWall(), mi.isDirt(), mine);
            if (mine) noteMine(l.x, l.y);
            if (mi.getCheeseAmount() > 0) {
                int d = G.me.distanceSquaredTo(l);
                if (d < bd) { bd = d; bestCheese = l; }
            } else if (cheeseTarget != null && cheeseTarget.equals(l)) {
                cheeseTarget = null;
            }
        }
        if (bestCheese != null) {
            if (cheeseTarget == null || G.me.distanceSquaredTo(cheeseTarget) > bd) cheeseTarget = bestCheese;
        }
    }

    /** Re-sense after moving (new cone) only to refresh cheese; terrain is caught next turn. */
    static void sense2() throws GameActionException {
        if (Clock.getBytecodesLeft() + (G.debt += 8) - 6 < Cfg.SENSE2_LEFT) return;
        sense();
    }

    /** A mine we know that the king's list lacks (rotating through them by round), else -1. */
    static int unreported() {
        for (int j = 0; j < myMineCount; j++) {
            int key = myMines[(G.round + j) % myMineCount];
            if (!Comms.hasMine(key / 64, key % 64)) return key;
        }
        return -1;
    }

    static void noteMine(int x, int y) {
        int key = x * 64 + y;
        for (int i = 0; i < myMineCount; i++) if (myMines[i] == key) return;
        if (myMineCount < myMines.length) myMines[myMineCount++] = key;
    }

    static void pickUp() throws GameActionException {
        RobotController rc = G.rc;
        MapLocation c = rc.getLocation();
        if (rc.canPickUpCheese(c)) { rc.pickUpCheese(c); picked++; }
        Direction f = rc.getDirection();
        Direction[] fr = G.fan(c, f);
        for (Direction d : fr) {
            MapLocation l = c.add(d);
            if (rc.canPickUpCheese(l)) { rc.pickUpCheese(l); picked++; }
        }
        G.rawCheese = rc.getRawCheese();
    }

    static RobotInfo eKing, nearestEnemy;
    static RobotInfo[] seen = new RobotInfo[0];
    public static int naps, throwsDone, ratTraps;
    static int carryRound = -100;

    /** Grab an adjacent enemy baby rat (the engine allows it when it cannot see us or is weaker): the healthiest. */
    static boolean tryCarry(RobotInfo[] robots) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady() || rc.getCarrying() != null) return false;
        RobotInfo best = null;
        int bs = Integer.MIN_VALUE;
        for (RobotInfo r : robots) {
            if (r.team != G.them || r.type != UnitType.BABY_RAT) continue;
            if (G.me.distanceSquaredTo(r.location) > 2) continue;
            if (!rc.canCarryRat(r.location)) continue;
            int s = r.health;
            if (s > bs) { bs = s; best = r; }
        }
        if (best == null) return false;
        rc.carryRat(best.location);
        D.act(D.A_NAP);
        carryRound = G.round;
        naps++;
        return true;
    }

    /**
     * Throw the carried rat along the direction that hurts most: the flight (6-8 tiles) hits the first robot,
     * wall or dirt for 46/42 (first two tiles), 38/34, 30/26, 22/18; the victim takes the same damage as the robot hit.
     * Hitting our own rat costs its damage; an empty flight lands for 10.
     */
    static final RobotInfo[] grid = new RobotInfo[289];
    static final int[] gridAt = new int[289];
    /** Damage when the flight ends at tile k (2 = point-blank 42, then 38, 34, ...). */
    static final int[] FLIGHT_DMG = new int[9];
    static { for (int k = 2; k <= 8; k++) FLIGHT_DMG[k] = 10 + 4 * (2 * (4 - (k - 1) / 2) + ((k & 1) == 1 ? 1 : 0)); }
    static int gridStamp;

    static void gridPut(int ox, int oy, RobotInfo r, int stamp) {
        if (ox < -8 || ox > 8 || oy < -8 || oy > 8) return;
        int i = (ox + 8) + 17 * (oy + 8);
        grid[i] = r;
        gridAt[i] = stamp;
    }

    /** The robots we see by offset from me (kings cover their 3x3, cats their 2x2), so flight scans never sense tile
     *  by tile; returns the stamp that marks this fill. */
    static int gridFill(MapLocation me, RobotInfo[] robots) {
        int stamp = ++gridStamp;
        for (RobotInfo r : robots) {
            int ox = r.location.x - me.x, oy = r.location.y - me.y;
            if (r.type == UnitType.RAT_KING) {
                for (int ex = -1; ex <= 1; ex++) for (int ey = -1; ey <= 1; ey++) gridPut(ox + ex, oy + ey, r, stamp);
            } else if (r.team == Team.NEUTRAL) {
                for (int ex = 0; ex <= 1; ex++) for (int ey = 0; ey <= 1; ey++) gridPut(ox + ex, oy + ey, r, stamp);
            } else gridPut(ox, oy, r, stamp);
        }
        return stamp;
    }

    /** D3[(dx + 1) + 3 * (dy + 1)]: the direction of a unit offset (CENTER in the middle). */
    static final Direction[] D3 = {Direction.SOUTHWEST, Direction.SOUTH, Direction.SOUTHEAST, Direction.WEST,
                                   Direction.CENTER, Direction.EAST, Direction.NORTHWEST, Direction.NORTH, Direction.NORTHEAST};

    /** Carrying an enemy (Cfg.CARRY_AIM): from our tile or a neighbour we can step to, a throw whose point-blank tile
     *  (two out, the first one empty) holds an enemy robot: a baby 84, a king's body 126 (tryThrow's scale). Turn to
     *  it, step there (a strafe when it is not ahead) and throw now if the action is ready, else hold the aim for
     *  next turn. Only enemy tiles within Chebyshev 3 can be such targets, so the scan starts from them (a scan of
     *  every step and throw cost ~7000 bytecodes and overran). Ties: stay, then G.hand of the step, of the throw, and
     *  G.pickBit of the target tile. */
    static boolean aimCarry() throws GameActionException {
        RobotController rc = G.rc;
        if (seen == null || Flow.left() < Cfg.THROW_BC) return false;
        MapLocation me = rc.getLocation();
        int[] tx = AIM_TX, ty = AIM_TY, tv = AIM_TV;
        int n = 0;
        for (RobotInfo r : seen) {
            if (r.team != G.them) continue;
            int ox = r.location.x - me.x, oy = r.location.y - me.y;
            if (r.type == UnitType.RAT_KING) {
                if (ox > 4 || ox < -4 || oy > 4 || oy < -4) continue;
                for (int ex = -1; ex <= 1; ex++)
                    for (int ey = -1; ey <= 1; ey++) {
                        int x = ox + ex, y = oy + ey;
                        if (x > 3 || x < -3 || y > 3 || y < -3 || n >= 24) continue;
                        tx[n] = x; ty[n] = y; tv[n++] = 126;
                    }
            } else if (ox <= 3 && ox >= -3 && oy <= 3 && oy >= -3 && n < 24) { tx[n] = ox; ty[n] = oy; tv[n++] = 84; }
        }
        if (n == 0) return false;
        int stamp = gridFill(me, seen);
        int[] mem = Sym.mem;
        int w = G.W, h = G.H, blk = Sym.WALL | Sym.DIRT;
        boolean canMove = rc.isMovementReady();
        int best = Integer.MIN_VALUE;
        Direction bm = null, bt = null;
        for (int j = 0; j < 9; j++) {
            if (aimLeft > 0 && Flow.left() <= aimLeft) break;
            int mx = j % 3 - 1, my = j / 3 - 1;
            boolean stay = mx == 0 && my == 0;
            if (!stay && !canMove) continue;
            Direction m = D3[j];
            boolean checked = stay;
            for (int i = 0; i < n; i++) {
                int vx = tx[i] - mx, vy = ty[i] - my;
                if (vx != 0 && vx != 2 && vx != -2 || vy != 0 && vy != 2 && vy != -2 || vx == 0 && vy == 0) continue;
                int fx = mx + vx / 2, fy = my + vy / 2, x = me.x + fx, y = me.y + fy;
                // the first flight tile: on the map, open terrain, no robot (the tile we leave is free)
                if (x < 0 || y < 0 || x >= w || y >= h || (mem[x + w * y] & blk) != 0
                        || (fx != 0 || fy != 0) && gridAt[(fx + 8) + 17 * (fy + 8)] == stamp) continue;
                if (!checked) { if (!rc.canMove(m)) break; checked = true; }
                Direction d = D3[(vx / 2 + 1) + 3 * (vy / 2 + 1)];
                int key = tv[i] * 16 + (stay ? 8 : G.hand(me, m) > 0 ? 4 : 0) + (G.hand(me.add(m), d) > 0 ? 2 : 0)
                        + (G.pickBit(me, me.translate(tx[i], ty[i])) ? 1 : 0);
                if (key > best) { best = key; bm = m; bt = d; }
            }
        }
        if (bt == null) return false;
        if (rc.getDirection() != bt) { if (!rc.canTurn(bt)) return false; rc.turn(bt); G.facing = bt; }
        if (bm != Direction.CENTER) { rc.move(bm); G.me = rc.getLocation(); }
        if (rc.canThrowRat()) { rc.throwRat(); throwsDone++; aimThrows++; }
        return true;
    }

    static final int[] AIM_TX = new int[24], AIM_TY = new int[24], AIM_TV = new int[24];

    static int aimThrows;

    /** Set down a carried ally once we can act: on the free tile farthest from our errand. */
    static void dropAlly() throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady()) return;
        MapLocation me = rc.getLocation(), t = Nav.target;
        Direction best = null;
        int bs = -1;
        for (Direction d : G.DIRS) {
            if (!rc.canDropRat(d)) continue;
            MapLocation l = me.add(d);
            int s = 2 * (t != null ? l.distanceSquaredTo(t) : 0) + (G.hand(me, d) > 0 ? 1 : 0);
            if (s > bs) { bs = s; best = d; }
        }
        if (best != null) { rc.dropRat(best); D.act(D.A_DROP_ALLY); }
    }

    static boolean tryThrow(RobotInfo[] robots) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady()) return false;
        MapLocation me = rc.getLocation();
        if (robots == null) robots = rc.senseNearbyRobots(-1, null);
        int stamp = gridFill(me, robots);
        Direction bestD = null;
        int bs = Integer.MIN_VALUE;
        int[] mem = Sym.mem;
        int w = G.W, h = G.H, blk = Sym.WALL | Sym.DIRT;
        for (int i = 0; i < 8; i++) {
            Direction d = G.DIRS[i];
            int ddx = d.dx, ddy = d.dy;
            int x = me.x + ddx, y = me.y + ddy;
            if (x < 0 || y < 0 || x >= w || y >= h || (mem[x + w * y] & blk) != 0 || gridAt[(ddx + 8) + 17 * (ddy + 8)] == stamp) continue;
            int s2 = 10;
            for (int k = 2; k <= 8; k++) {
                x += ddx; y += ddy;
                int dmg = FLIGHT_DMG[k];
                if (x < 0 || y < 0 || x >= w || y >= h) { s2 = 10; break; }
                if ((mem[x + w * y] & blk) != 0) { s2 = dmg; break; }
                int gi = (k * ddx + 8) + 17 * (k * ddy + 8);
                if (gridAt[gi] == stamp) {
                    RobotInfo r = grid[gi];
                    // beyond tile 2 the hit lands next round: a baby has usually moved by then (kings are slow)
                    boolean far = k > 2 && r.type == UnitType.BABY_RAT;
                    if (r.team == G.them) s2 = far ? dmg * 2 * Cfg.THROW_FAR_PCT / 100 : dmg * (r.type == UnitType.RAT_KING ? 3 : 2);
                    else if (r.team == G.us) s2 = -dmg;
                    else s2 = -50; // a cat eats the rat: no damage, it sleeps
                    break;
                }
            }
            s2 = s2 * 2 + (G.hand(me, d) > 0 ? 1 : 0);
            if (s2 > bs) { bs = s2; bestD = d; }
        }
        if (bestD == null) return false;
        // no point-blank hit (42) yet: keep holding (the carried rat is out of the fight) while the carry lasts
        if (bs / 2 < Cfg.THROW_MIN && G.round - carryRound < Cfg.THROW_HOLD) return false;
        if (rc.getDirection() != bestD) { if (!rc.canTurn(bestD)) return false; rc.turn(bestD); G.facing = bestD; D.turnOnly(D.A_THROW); }
        if (!rc.canThrowRat()) return false;
        rc.throwRat();
        D.act(D.A_THROW);
        throwsDone++;
        return true;
    }

    /** Rat trap on the front tile nearest an approaching enemy rat (50 damage + stun when it enters the 3x3). */
    static boolean tryRatTrap() throws GameActionException {
        RobotController rc = G.rc;
        if (nearestEnemy == null || !rc.isActionReady() || rc.getNumberRatTraps() >= trapCap()) return false;
        if (rc.getAllCheese() < Econ.floor(Cfg.RAT_TRAP_MIN_CHEESE)) return false;
        int d0 = G.me.distanceSquaredTo(nearestEnemy.location);
        if (d0 <= 2 || d0 > Cfg.BABY_TRAP_DIST2) return false;
        MapLocation best = trapTile(rc.getLocation(), rc.getDirection(), nearestEnemy.location, d0);
        if (best == null) return false;
        rc.placeRatTrap(best);
        D.act(D.A_RAT_TRAP);
        ratTraps++;
        return true;
    }

    /** The front tile for a rat trap against an enemy at e (d0 = our dist2 to it): the one nearest e if nearer than we
     *  are, else null. */
    static MapLocation trapTile(MapLocation me, Direction f, MapLocation e, int d0) throws GameActionException {
        RobotController rc = G.rc;
        MapLocation best = null;
        int bs = Integer.MAX_VALUE;
        for (Direction d : G.fan(me, f)) {
            MapLocation l = me.add(d);
            if (!rc.canPlaceRatTrap(l)) continue;
            int s2 = l.distanceSquaredTo(e);
            if (s2 < bs) { bs = s2; best = l; }
        }
        return bs < d0 ? best : null;
    }

    /** The live rat traps past which no combat trap is laid (Cfg.TRAP_CAP). */
    static int trapCap() { return Cfg.TRAP_CAP; }

    /** Bite the best adjacent enemy: a king first, then the lowest health; turn to face it when needed. */
    static boolean biteEnemies(RobotInfo[] robots) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady()) return false;
        MapLocation me = rc.getLocation();
        RobotInfo best = null, bestTurn = null;
        int bs = Integer.MAX_VALUE, bt = Integer.MAX_VALUE;
        for (RobotInfo r : robots) {
            if (r.team != G.them) continue;
            MapLocation t = r.location;
            if (r.type == UnitType.RAT_KING) {
                // any king tile adjacent to us
                int dx = Math.max(-1, Math.min(1, me.x - t.x)), dy = Math.max(-1, Math.min(1, me.y - t.y));
                t = t.translate(dx, dy);
            }
            if (me.distanceSquaredTo(t) > 2) continue;
            int s = r.type == UnitType.RAT_KING ? r.health : 1000 + r.health;
            if (rc.canAttack(t)) { if (s < bs) { bs = s; best = r; } }
            else if (s < bt) { bt = s; bestTurn = r; }
        }
        // turning is free of action cooldown: a better target behind us wins if we can turn this turn
        RobotInfo r = best;
        if (bestTurn != null && (best == null || bt < bs) && rc.isTurningReady()) r = bestTurn;
        if (r == null) return false;
        MapLocation t = r.location;
        if (r.type == UnitType.RAT_KING) {
            int dx = Math.max(-1, Math.min(1, me.x - t.x)), dy = Math.max(-1, Math.min(1, me.y - t.y));
            t = t.translate(dx, dy);
        }
        if (!rc.canAttack(t)) {
            Direction d = me.directionTo(t);
            if (rc.getDirection() != d && rc.canTurn(d)) { rc.turn(d); D.turnOnly(r.type == UnitType.RAT_KING ? D.A_BITE_KING : D.A_BITE_RAT); }
            G.facing = rc.getDirection();
        }
        if (rc.canAttack(t)) {
            int c = r.type == UnitType.BABY_RAT ? finishCheese(r.health, rc.getGlobalCheese()) : 0;
            if (c > 0 && rc.canAttack(t, c)) rc.attack(t, c); else rc.attack(t);
            D.act(r.type == UnitType.RAT_KING ? D.A_BITE_KING : D.A_BITE_RAT);
            bitesRat++;
            return true;
        }
        return false;
    }

    /** Cheese that turns a bite on a target with hp into a kill (k*k for +k, k <= FINISH_K), else 0. */
    static int finishCheese(int hp, int global) {
        if (hp <= 10 || global < Cfg.FINISH_FLOOR) return 0;
        int k = hp - 10;
        return k <= Cfg.FINISH_K ? k * k : 0;
    }

    /** Bite any adjacent tile of the cat's 2x2 body, turning to face it when needed. */
    static boolean tryBiteCat(RobotInfo c) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady()) return false;
        MapLocation me = rc.getLocation();
        MapLocation turnTo = null;
        for (int i = 0; i < 4; i++) {
            MapLocation t = c.location.translate(i & 1, i >> 1);
            if (me.distanceSquaredTo(t) > 2) continue;
            if (rc.canAttack(t)) {
                int cheese = Comms.read(Comms.BUDGET) * 4 >= Cfg.CHEESE_BITE_SURPLUS ? 1 : 0;
                if (cheese > 0 && rc.canAttack(t, cheese)) rc.attack(t, cheese); else rc.attack(t);
                D.act(D.A_BITE_CAT);
                bitesCat++;
                return true;
            }
            if (turnTo == null) turnTo = t;
        }
        if (turnTo != null && rc.canTurn()) {
            rc.turn(me.directionTo(turnTo));
            D.turnOnly(D.A_BITE_CAT);
            if (rc.canAttack(turnTo)) { rc.attack(turnTo); D.act(D.A_BITE_CAT); bitesCat++; return true; }
        }
        return false;
    }

    /** Place a cat trap on a visible adjacent tile in front of an approaching cat. */
    /** Cfg.CAT_LAY: a cat trap on an adjacent cat-lane tile the cat left CAT_LAY_AGE+ rounds ago, the most recently
     *  walked first (laneKey). */
    static void layLane() throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady() || rc.getNumberCatTraps() >= 10 || rc.getAllCheese() < Cfg.CAT_LAY_FLOOR) return;
        MapLocation me = rc.getLocation(), best = null;
        int bs = Integer.MAX_VALUE;
        for (Direction d : G.DIRS) {
            MapLocation l = me.add(d);
            if (!CatLane.lane(l.x, l.y)) continue;
            int a = CatLane.age(l.x, l.y, G.round);
            if (a < Cfg.CAT_LAY_AGE || !rc.canPlaceCatTrap(l)) continue;
            int s = laneKey(a, l, G.pickBit(me, l));
            if (s < bs) { bs = s; best = l; }
        }
        if (best != null) { rc.placeCatTrap(best); D.act(D.A_LAY_LANE); trapsPlaced++; }
    }

    /** layLane's rank: age, then dist2 from the map centre (doubled coords; both mirror-invariant), then the pick bit. */
    static int laneKey(int age, MapLocation l, boolean bit) {
        int px = 2 * l.x - G.cx2, py = 2 * l.y - G.cy2;
        return ((Math.min(age, 4000) << 15) + Math.min(px * px + py * py, 32767)) * 2 + (bit ? 0 : 1);
    }

    static void tryCatTrap(RobotInfo c) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady()) return;
        if (rc.getAllCheese() < Cfg.TRAP_FLOOR) return;
        // defending a king is survival: traps there need only the floor; elsewhere they need projected surplus
        MapLocation k = Comms.nearestKing(G.me);
        boolean defend = k != null && k.distanceSquaredTo(c.location) <= Cfg.TRAP_DEFEND_DIST2;
        if (!defend && !G.coop && Comms.read(Comms.BUDGET) * 4 < Cfg.TRAP_SURPLUS) return;
        if (rc.getNumberCatTraps() >= 10) return;
        int d4 = catDist4(c.location, G.me);
        if (d4 > 4 * 36) return;
        MapLocation best = null;
        int bs = Integer.MAX_VALUE;
        MapLocation me = rc.getLocation();
        Direction f = rc.getDirection();
        Direction[] fr = G.fan(me, f);
        MapLocation n1 = c.location.add(c.direction), n2 = n1.add(c.direction), n3 = n2.add(c.direction);
        for (Direction d : fr) {
            MapLocation l = me.add(d);
            if (!rc.canPlaceCatTrap(l)) continue;
            // trigger zone is the 3x3 around the trap: the cat must pass within Chebyshev 1 of it
            int s = Math.min(Math.min(chebToBody(n1, l), chebToBody(n2, l)), chebToBody(n3, l));
            s = Math.min(s, chebToBody(c.location, l));
            if (s > 1) continue;
            int score = s * 100 + catDist4(c.location, l);
            if (score < bs) { bs = score; best = l; }
        }
        if (best != null) { rc.placeCatTrap(best); D.act(D.A_CAT_TRAP); trapsPlaced++; }
    }

    static int chebToBody(MapLocation bl, MapLocation p) {
        int dx = p.x < bl.x ? bl.x - p.x : (p.x > bl.x + 1 ? p.x - bl.x - 1 : 0);
        int dy = p.y < bl.y ? bl.y - p.y : (p.y > bl.y + 1 ? p.y - bl.y - 1 : 0);
        return Math.max(dx, dy);
    }

    static void tryDeliver() throws GameActionException {
        RobotController rc = G.rc;
        int raw = rc.getRawCheese();
        if (raw <= 0 || !rc.isActionReady()) return;
        RobotInfo[] ks = rc.senseNearbyRobots(9, G.us);
        for (RobotInfo k : ks) {
            if (k.type != UnitType.RAT_KING) continue;
            // any king tile within range 9 and in our cone
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++) {
                    MapLocation t = k.location.translate(dx, dy);
                    if (rc.canTransferCheese(t, raw)) {
                        rc.transferCheese(t, raw);
                        D.act(D.A_DELIVER);
                        delivered += raw;
                        G.rawCheese = 0;
                        return;
                    }
                }
        }
    }

    /** Squeak one fact: to a king in hearing range a mine it may not know or a sharper symmetry mask; else mine news
     *  rat to rat; a cat near the king. */
    static void report() {
        MapLocation k = Comms.numKings == 0 ? null : Comms.nearestKing(G.me);
        boolean near = k != null && G.me.distanceSquaredTo(k) <= 16;
        if (Comms.numMines == 0) {
            // no mine known to the team yet: pass mine news rat to rat, which outruns a courier walking home
            hearMines(-1);
            if (myMineCount > 0 && !near) {
                int key = myMines[G.round % myMineCount];
                D.lab2 = D.C_MINE_GOSSIP;
                squeakMine(key);
                return;
            }
        } else {
            hearMines(G.round - 1);
            if (!near) {
                int key = unreported();
                if (key >= 0) { D.lab2 = D.C_MINE_RELAY; squeakMine(key); return; }
            }
        }
        // a squeak reaches robots with any tile within dist2 16 of us: the king's centre within 16 is always heard
        if (near) {
            if (symNews()) { D.lab2 = D.C_SYM_KING; squeakSym(); return; }
            int key = unlisted();
            if (key >= 0) { D.lab2 = D.C_MINE_KING; squeakMine(key); return; }
        }
        if (near && catNews()) {
            D.lab2 = D.C_CAT;
            squeakCat();
        }
    }

    /** Note the mines squeaked since round r (-1: all we can hear). */
    static void hearMines(int r) {
        for (Message m : G.rc.readSqueaks(r)) {
            int c = m.getBytes();
            if (Comms.sqType(c) == Comms.SQ_MINE) noteMine(Comms.sqX(c), Comms.sqY(c));
        }
    }

    /** Squeak the mine at key = x * 64 + y; true when it went out. */
    static boolean squeakMine(int key) {
        return G.rc.squeak(Comms.squeak(Comms.SQ_MINE, key / 64, key % 64, 0));
    }

    /** Our symmetry mask is sharper than the kings' shared one. */
    static boolean symNews() {
        int sh = Comms.read(Comms.SYM);
        return sh != 0 && (Sym.mask & sh) != sh;
    }

    static boolean squeakSym() {
        return G.rc.squeak(Comms.squeak(Comms.SQ_SYM, 0, 0, Sym.mask));
    }

    /** The first of our mines the king's list lacks (x * 64 + y), else -1. */
    static int unlisted() {
        for (int i = 0; i < myMineCount; i++) {
            int x = myMines[i] / 64, y = myMines[i] % 64;
            if (!Comms.hasMine(x, y)) return myMines[i];
        }
        return -1;
    }

    /** A cat was seen this round or within the last two. */
    static boolean catNews() { return catSeen != null && G.round - catSeenRound <= 2; }

    static boolean squeakCat() {
        return G.rc.squeak(Comms.squeak(Comms.SQ_CAT, catSeen.x, catSeen.y, 0));
    }

    /** Some tile of a cat (2x2 from catBL) is within squeak range (d2 16) of p. */
    static boolean catInEarshot(MapLocation catBL, MapLocation p) {
        int dx = Math.max(0, Math.max(catBL.x - p.x, p.x - catBL.x - 1));
        int dy = Math.max(0, Math.max(catBL.y - p.y, p.y - catBL.y - 1));
        return dx * dx + dy * dy <= 16;
    }

    // ---------------------------------------------------------------- movement

    /** True this turn when nearestEnemy is a prey (a rat we could ratnap). */
    static boolean hunting;

    /** Which branch took the last movement decision (debug tag mv=). */
    static char why = '-';

    static void moveDecision() throws GameActionException {
        why = '-';
        RobotController rc = G.rc;
        int raw = rc.getRawCheese();
        boolean late = G.round > 1960;

        if (cat != null && catAtKing(cat.location) && approachCat(cat)) { why = 'g'; D.lab = D.M_CAT_DEFEND; G.debt += 6; return; }
        int cde = G.debt, cbe = G.bc();   // ledger (K 11: 9, iload endGo and ifeq)
        boolean endGo = !G.coop && G.round >= Cfg.END_CAT;
        G.debt = cde + G.bc() - cbe + 11;
        if (endGo && endCat(raw)) { why = 'e'; D.lab = D.M_END_CAT; G.debt += 6; return; }
        // keep gatherers out of cat reach
        if (cat != null && role == GATHER && catDist4(cat.location, G.me) <= 4 * Cfg.GATHER_CAT_DIST) {
            // flee the cat's centre (its location is the 2x2 body's low corner, which mirrors to the far corner)
            if (Nav.flee(cat.location)) { why = 'c'; D.lab = D.M_FLEE_CAT; G.debt += 6; return; }
        }
        if (!G.coop && rc.getCarrying() != null && rc.getCarrying().team == G.them && aimCarry()) { why = 'T'; D.lab = D.M_AIM_CARRY; G.debt += 6; return; }
        Debug.mark(10);
        if (!G.coop && retreat()) { why = 'r'; D.lab = D.M_RETREAT; G.debt += 6; return; }
        if (formKing()) { why = 'K'; D.lab = D.M_FORM_KING; G.debt += 6; return; }
        int cd2 = G.debt, cb2 = G.bc();   // ledger (K = 20 with G.settle)
        boolean cashMv = cashSite != null && cashMove();
        G.settle(cd2, cb2, 20);
        if (cashMv) { why = 'Q'; D.lab = D.M_CASH; G.debt += 6; return; }
        Debug.mark(11);
        if (!G.coop && raider && raw == 0 && rc.getHealth() >= Cfg.RAID_HP && homeHalf(G.me)
                && (nearestEnemy == null || G.me.distanceSquaredTo(nearestEnemy.location) > Cfg.RAID_ENGAGE) && raid()) { why = 'z'; D.lab = D.M_RAID; G.debt += 6; return; }
        Debug.mark(12);
        if (!G.coop && flank(seen)) { why = 'k'; D.lab = D.M_FLANK; G.debt += 6; return; }
        Debug.mark(13);
        Debug.mark(14);
        if (!G.coop && (role != GATHER || nearEnemy(2)) && faceEnemy()) { why = 'f'; D.lab = D.M_FACE; G.debt += 6; return; }
        if (role == ATTACK && G.coop && stallCat()) { why = 'g'; D.lab = D.M_STALL_CAT; G.debt += 6; return; }
        if (role == ATTACK && attack()) { why = 'a'; D.lab = D.M_ATTACK; G.debt += 6; return; }
        Debug.mark(15);
        if (!G.coop && war()) { why = 'w'; D.lab = D.M_WAR; G.debt += 6; return; }
        Debug.mark(16);
        why = 'g';
        if (cat != null && role == FIGHT) {
            if (approachCat(cat)) { D.lab = D.M_FIGHT_CAT; G.debt += 6; return; }
        }

        if ((raw >= Cfg.DELIVER_AT || (raw > 0 && (late || cheeseTarget == null && raw >= Cfg.DELIVER_IDLE))) && deliver()) { D.lab = D.M_DELIVER; G.debt += 6; return; }
        if (Comms.numMines == 0 && myMineCount > 0) {
            // we found a mine nobody else knows: squeaks only reach the king from dist2 16, so carry the news home
            MapLocation k = Comms.nearestKing(G.me);
            if (k != null) { Nav.goTo(k); why = 'm'; D.lab = D.M_COURIER; G.debt += 6; return; }
        }
        if (cheeseTarget != null) {
            if (G.cheb(G.me, cheeseTarget) == 1 && Nav.isDirt(cheeseTarget)) {
                // cheese on dirt: pick it up from here, facing it, instead of digging onto the tile
                pickAdjacent();
                D.lab = D.M_CHEESE_PICK; G.debt += 6; return;
            }
            chaseCheese();
            D.lab = D.M_CHEESE; G.debt += 6; return;
        }
        if (role == FIGHT) {
            MapLocation ct = knownCat();
            if (ct != null) { Nav.goTo(ct); D.lab = D.M_KNOWN_CAT; G.debt += 6; return; }
            if (raider && rc.getHealth() >= Cfg.RAID_HP && raid()) { why = 'z'; D.lab = D.M_RAID_FIGHT; G.debt += 6; return; }
        }
        if (scouting()) {
            // a scout with news the king lacks carries it home (squeaks reach the king from dist2 16 only)
            MapLocation k = Comms.nearestKing(G.me);
            if (k != null && unreported() >= 0) {
                Nav.goTo(k); why = 's'; D.lab = D.M_SCOUT_HOME; G.debt += 6; return;
            }
            why = 'S';
            D.lab = D.M_SCOUT; G.debt += 6; 
            if (!exploreHome()) explore();
            return;
        }
        if (role == GATHER && G.round < seekUntil && seek()) { why = 'x'; D.lab = D.M_SEEK; G.debt += 6; return; }
        if (role == GATHER || role == FIGHT) {
            refreshMine();
            if (mineTarget != null) {
                if (idle(mineTarget)) {
                    shiftMine();
                    if (role == GATHER && idleSeek() && seek()) { why = 'x'; D.lab = D.M_IDLE_SEEK; G.debt += 6; return; }
                    if (mineTarget == null) { D.pickSite = 2; mineTarget = pickMine(); }
                }
                if (atMine(mineTarget)) {
                    if (role == GATHER && startSeek(mineTarget) && seek()) { why = 'x'; D.lab = D.M_START_SEEK; G.debt += 6; return; }
                    // wander inside the spawn square, facing outward to see new cheese
                    wanderAround(mineTarget);
                    why = 'W';
                    D.lab = D.M_MINE_WANDER; G.debt += 6; 
                } else {
                    why = 'M';
                    D.lab = D.M_MINE_GO; G.debt += 6; 
                    goMine();
                }
                return;
            }
        }
        why = 'E';
        D.lab = D.M_EXPLORE; G.debt += 6; 
        explore();
    }

    /** Cheese one step away: face it and pick it up from here (cheeseTarget cleared once it is gone). */
    static void pickAdjacent() throws GameActionException {
        RobotController rc = G.rc;
        Direction d = G.me.directionTo(cheeseTarget);
        if (rc.getDirection() != d && rc.canTurn(d)) { rc.turn(d); G.facing = d; }
        pickUp();
        if (!rc.canSenseLocation(cheeseTarget) || rc.senseMapInfo(cheeseTarget).getCheeseAmount() == 0) cheeseTarget = null;
        why = 'P';
    }

    /** Walk to cheeseTarget; a walk stalled past 12 turns gives it up. */
    static void chaseCheese() throws GameActionException {
        why = 'C';
        Nav.goTo(cheeseTarget);
        if (Nav.stall > 12) cheeseTarget = null;
    }

    /** The mine refresh: drop mineTarget once the team's mine list changed since the pick (Cfg.MINE_BALANCE), then
     *  pick one when there is none. */
    static void refreshMine() {
        if (mineTarget != null && Comms.numMines != minesAtPick) mineTarget = null;
        if (mineTarget == null) { D.pickSite = 0; mineTarget = pickMine(); }
    }

    /** Cfg.IDLE_SHIFT: an idle rat moves one place along the pick list. */
    static void shiftMine() {
        mineShift++;
        D.pickSite = 1;
        mineTarget = pickMine();
    }

    /** Walk to mineTarget; a walk stalled past 25 turns gives it up. */
    static void goMine() throws GameActionException {
        Nav.goTo(mineTarget);
        if (Nav.stall > 25) mineTarget = null;
    }

    /** Head for the nearest king to hand over our cheese (face it once in transfer reach); false with no king known. */
    static boolean deliver() throws GameActionException {
        MapLocation k = Comms.nearestKing(G.me);
        if (k == null) return false;
        if (inTransferReach(G.me, k)) {
            // in transfer range of the king's body (dist2 9 to its nearest tile): face it, stay off the ring
            Direction d = G.me.directionTo(k);
            if (d != Direction.CENTER && G.rc.getDirection() != d && G.rc.canTurn(d)) G.rc.turn(d);
            why = 'D';
            return true;
        }
        why = 'd';
        Nav.goTo(k);
        return true;
    }

    static int noKing;               // symmetry candidates whose enemy-king spot we saw empty
    static MapLocation lastEKing;
    static int lastEKingRound = -100;
    public static int presses;

    /** King press movement (Cfg.KING_PRESS): walk to a fresh enemy-king sighting in range and stay against it. */
    static boolean pressKing() throws GameActionException {
        RobotController rc = G.rc;
        MapLocation ek = eKing != null ? eKing.location : G.round - lastEKingRound <= Cfg.PRESS_AGE ? lastEKing : null;
        if (ek == null || rc.getHealth() < Cfg.PRESS_HP) return false;
        if (role == GATHER && rc.getRawCheese() >= Cfg.DELIVER_AT) return false;
        int d = G.me.distanceSquaredTo(ek);
        if (d > (role == GATHER ? Cfg.PRESS_GATHER_D2 : Cfg.PRESS_D2)) return false;
        if (eKing == null && d <= 8 && rc.canSenseLocation(ek)) {
            RobotInfo r = rc.senseRobotAtLocation(ek);
            if (r == null || r.type != UnitType.RAT_KING || r.team != G.them) { lastEKing = null; return false; }
        }
        presses++;
        if (G.cheb(G.me, ek) <= 2) {
            // touching the king's 3x3: face its nearest tile and bite (bites run before and after movement)
            int dx = Math.max(-1, Math.min(1, G.me.x - ek.x)), dy = Math.max(-1, Math.min(1, G.me.y - ek.y));
            Direction f = G.me.directionTo(ek.translate(dx, dy));
            if (f != Direction.CENTER && rc.getDirection() != f && rc.canTurn(f)) { rc.turn(f); G.facing = f; }
            return true;
        }
        Nav.goTo(ek);
        return true;
    }

    /** Attacker movement: the visible enemy king, else the last place we saw it, else the nearest live mirror spot. */
    static boolean attack() throws GameActionException {
        if (eKing != null) { lastEKing = eKing.location; lastEKingRound = G.round; Nav.goTo(eKing.location); return true; }
        if (lastEKing != null && G.round - lastEKingRound < 30 && G.me.distanceSquaredTo(lastEKing) > 8) {
            Nav.goTo(lastEKing);
            return true;
        }
        MapLocation t = rushTarget();
        if (t == null) return false;
        Nav.goTo(t);
        return true;
    }

    static MapLocation rushTarget() throws GameActionException {
        RobotController rc = G.rc;
        MapLocation best = null;
        int bd = Integer.MAX_VALUE;
        for (int s = 1; s <= 4; s <<= 1) {
            if ((Sym.mask & s) == 0 || (noKing & s) != 0) continue;
            MapLocation c = G.mirror(home, s);
            if (G.me.distanceSquaredTo(c) <= 13 && rc.canSenseLocation(c)) {
                RobotInfo r = rc.senseRobotAtLocation(c);
                if (r == null || r.type != UnitType.RAT_KING || r.team != G.them) { noKing |= s; continue; }
            }
            int d = G.me.distanceSquaredTo(c) * 2 + (G.pickBit(G.me, c) ? 1 : 0);
            if (d < bd) { bd = d; best = c; }
        }
        if (best == null && noKing != 0) { noKing = 0; return null; }
        return best;
    }

    /**
     * Anti-ratnap stance: an enemy within two tiles must stay in our cone (a rat can only be grabbed by an enemy it
     * cannot see, or by a healthier one). Turn to face it instead of moving; once facing, advance only forward.
     */
    static boolean faceEnemy() throws GameActionException {
        if (nearestEnemy == null) return false;
        RobotController rc = G.rc;
        MapLocation e = nearestEnemy.location;
        int d = G.me.distanceSquaredTo(e);
        if (d > 8) return false;
        Direction to = G.me.directionTo(e);
        if (!rc.canSenseLocation(e) || (d > 2 && rc.getDirection() != to)) {
            if (rc.getDirection() != to && rc.canTurn(to)) { rc.turn(to); G.facing = to; }
            return true;
        }
        if (d > 2 && rc.canMove(to) && (Nav.avoid & (1 << to.ordinal())) == 0) {
            rc.move(to);
            G.me = rc.getLocation();
        }
        return true;
    }

    /** War movement: close on a visible enemy king or nearby enemy rat, or answer a king's alert. */
    static boolean war() throws GameActionException {
        int raw = G.rc.getRawCheese();
        MapLocation threatened = threatenedKing();
        if (threatened != null) defends++;
        if (role == GATHER && !hunting && threatened == null) {
            // gatherers keep the economy going: only an adjacent enemy stops them
            if (!nearEnemy(2)) return false;
            Direction f = G.me.directionTo(nearestEnemy.location);
            if (G.rc.getDirection() != f && G.rc.canTurn(f)) { G.rc.turn(f); G.facing = f; }
            return true;
        }
        if (eKing != null && (role == FIGHT || raw < Cfg.DELIVER_AT)) {
            Nav.goTo(eKing.location);
            return true;
        }
        if (nearestEnemy != null) {
            int d = G.me.distanceSquaredTo(nearestEnemy.location);
            if (d <= 2) {
                // stay and face it: biting happens before and after movement
                Direction f = G.me.directionTo(nearestEnemy.location);
                if (G.rc.getDirection() != f && G.rc.canTurn(f)) G.rc.turn(f);
                return true;
            }
            if (d <= 20 && (role == FIGHT || raw < Cfg.DELIVER_AT || d <= 8)) {
                // standoff: stepping into the front of an enemy that sees us (and is not weaker) walks onto its
                // traps and into a grab; hold, face it and let it come (flank and prey moves still engage)
                // (not against a rat at our threatened king: it is biting the king while we wait)
                boolean atKing = threatened != null && nearestEnemy.location.distanceSquaredTo(threatened) <= Cfg.KING_THREAT_D2;
                if (!hunting && !atKing && sees(nearestEnemy, G.me) && nearestEnemy.health >= G.rc.getHealth()) {
                    Direction f = G.me.directionTo(nearestEnemy.location);
                    if (G.rc.getDirection() != f && G.rc.canTurn(f)) { G.rc.turn(f); G.facing = f; }
                    standoffs++;
                    return true;
                }
                Nav.goTo(nearestEnemy.location);
                return true;
            }
        }
        if (Comms.fresh(Comms.read(Comms.ALERT + 2), 3)) {
            MapLocation a = Comms.alertLoc();
            if (a != null && G.me.distanceSquaredTo(a) <= Cfg.ALERT_DIST2 && G.me.distanceSquaredTo(a) > 8) {
                Nav.goTo(a);
                return true;
            }
        }
        return false;
    }

    public static int defends;

    /** Our king's location when it reported enemy rats close to it this or last round and we are near enough to
     *  help (gatherers then fight instead of gathering), else null. */
    static MapLocation threatenedKing() {
        if (G.coop || !Comms.fresh(Comms.read(Comms.ALERT + 2), 2) || Comms.alertThreat() < 1) return null;
        MapLocation a = Comms.alertLoc();
        return a != null && G.me.distanceSquaredTo(a) <= Cfg.DEFEND_D2 ? a : null;
    }

    /** True when a baby rat r (facing r.direction) can see tile t: dot >= |cross| with its facing, dist2 <= 20. */
    static boolean sees(RobotInfo r, MapLocation t) {
        int dx = t.x - r.location.x, dy = t.y - r.location.y;
        Direction f = r.direction;
        if (f == null || f == Direction.CENTER) return true;
        return dx * dx + dy * dy <= 20 && dx * f.dx + dy * f.dy >= Math.abs(dx * f.dy - dy * f.dx);
    }

    /** Step onto a tile adjacent to an enemy baby that it cannot see, keeping it in view; the post-move carry grabs it. */
    static boolean flank(RobotInfo[] robots) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady() || !rc.isMovementReady() || rc.getCarrying() != null) return false;
        RobotInfo bestE = null;
        Direction bestD = null;
        int bs = Integer.MAX_VALUE;
        for (RobotInfo e : robots) {
            if (e.team != G.them || e.type != UnitType.BABY_RAT) continue;
            if (G.me.distanceSquaredTo(e.location) > 8) continue;
            for (int i = 0; i < 8; i++) {
                Direction d = G.DIRS[i];
                MapLocation t = G.me.add(d);
                if (G.cheb(t, e.location) != 1 || sees(e, t)) continue;
                if ((Nav.avoid & (1 << i)) != 0 || !rc.canMove(d)) continue;
                int sc = e.health * 4 + (d == rc.getDirection() ? 0 : 1) * 2 + (G.hand(G.me, d) > 0 ? 1 : 0);
                if (sc < bs) { bs = sc; bestE = e; bestD = d; }
            }
        }
        if (bestE == null) return false;
        MapLocation g = Nav.guard;
        Nav.guard = bestE.location;
        boolean moved = Nav.step(bestD);
        Nav.guard = g;
        if (moved) flanks++;
        return moved;
    }

    public static int flanks, standoffs;

    /** Nearest enemy baby within dist2 20 that we could ratnap (strictly less health than ours), or null. */
    static RobotInfo prey(RobotInfo[] robots, int hp) {
        RobotInfo prey = null;
        int pd = Integer.MAX_VALUE;
        for (RobotInfo r : robots) {
            if (r.team != G.them || r.type != UnitType.BABY_RAT || r.health >= hp) continue;
            int d = G.me.distanceSquaredTo(r.location);
            if (d <= 20 && d < pd) { pd = d; prey = r; }
        }
        return prey;
    }

    /** A wounded rat near enemies falls back to its king instead of being finished off. */
    /** An empty rat on the ring around our king's 3x3 (Chebyshev 2 from its centre) while the cat we see is near
     *  that king: step to a tile off the ring and out of the cat's lane, farthest from the king. */
    static boolean yieldKing() throws GameActionException {
        MapLocation k = Comms.nearestKing(G.me);
        if (k == null || G.cheb(G.me, k) != 2 || King.catGap(k, cat.location) > 3) return false;
        RobotController rc = G.rc;
        if (!rc.isMovementReady()) return false;
        int lane = laneMask(cat), best = -1, bd = -1;
        for (int i = 0; i < 8; i++) {
            if ((lane & (1 << i)) != 0 || !rc.canMove(G.DIRS[i])) continue;
            MapLocation t = G.me.add(G.DIRS[i]);
            if (G.cheb(t, k) < 3) continue;
            int d = 2 * t.distanceSquaredTo(k) + (G.hand(G.me, G.DIRS[i]) > 0 ? 1 : 0);
            if (d > bd) { bd = d; best = i; }
        }
        return best >= 0 && Nav.step(G.DIRS[best]);
    }

    /** Cfg.AMMO_AWAY: step away from our king (or hold) while ammo(...) holds; false when it doesn't. */
    static boolean ammoAway() throws GameActionException {
        if (nearestEnemy == null) return false;
        MapLocation k = Comms.nearestKing(G.me);
        if (k == null || !ammo(G.rc.getHealth(), nearestEnemy.health, G.me.distanceSquaredTo(nearestEnemy.location), G.cheb(G.me, k))) return false;
        Nav.flee(k);
        return true;
    }

    /** True when a baby with health hp, cheb from our king's centre, is grabbable ammunition for an enemy rat with
     *  health ehp at dist2 d2 (Cfg.AMMO_AWAY). */
    static boolean ammo(int hp, int ehp, int d2, int cheb) {
        return hp <= Cfg.AMMO_HP && ehp > hp && d2 <= Cfg.AMMO_D2 && cheb <= Cfg.AMMO_R;
    }

    static boolean retreat() throws GameActionException {
        if (G.rc.getHealth() > Cfg.RETREAT_HP || !nearEnemy(20)) return false;
        MapLocation k = Comms.nearestKing(G.me);
        if (k == null || G.cheb(G.me, k) <= 2) return false;
        Nav.goTo(k);
        return true;
    }

    static boolean nearEnemy(int d2) {
        return nearestEnemy != null && G.me.distanceSquaredTo(nearestEnemy.location) <= d2;
    }

    /** True when the nearest tile of a 3x3 king centred at k is within transfer range (dist2 9) of p. */
    static boolean inTransferReach(MapLocation p, MapLocation k) {
        int dx = Math.max(0, Math.abs(p.x - k.x) - 1), dy = Math.max(0, Math.abs(p.y - k.y) - 1);
        return dx * dx + dy * dy <= 9 && !(Math.abs(p.x - k.x) <= 1 && Math.abs(p.y - k.y) <= 1);
    }

    /** Cfg.END_CAT: walk to the nearest seen or known cat (the bite step bites it); false when this baby has other work
     *  (a load, a carried rat, a threatened king, an enemy in reach, a formation site to join). */
    static boolean endCat(int raw) throws GameActionException {
        if (raw >= Cfg.DELIVER_AT || G.rc.getCarrying() != null) return false;
        if (threatenedKing() != null || nearEnemy(8)) return false;
        // a king formation (END_KING, CASH_END) outweighs the cat: its joiners go on to formKing and cashMove
        if (formSite() != null || cashSite != null) return false;
        if (Comms.numKings < 2 && G.globalCheese >= endCatBar(G.round)) return false;
        return goCat();
    }

    /** Walk to the cat in view (approachCat), else to the nearest known cat; false when neither. */
    static boolean goCat() throws GameActionException {
        if (cat != null) return approachCat(cat);
        MapLocation ct = knownCat();
        if (ct == null) return false;
        Nav.goTo(ct);
        return true;
    }

    /** Cfg.END_CAT with a lone king: the bank below which no endgame king can form (END_KING's END_KING_CHEESE, CASH_END's
     *  bar), so the babies have no formation to stay for. */
    static int endCatBar(int round) {
        return Math.min(Cfg.END_KING_CHEESE, Math.max(Cfg.CASH_FLOOR, 4 * (2000 - round - 55) + 10));
    }

    static MapLocation knownCat() {
        if (catSeen != null && G.round - catSeenRound < 10) return catSeen;
        MapLocation best = null;
        int bd = Integer.MAX_VALUE;
        for (int k = 0; k < Comms.CAT_SLOTS; k++) {
            int b = Comms.CAT0 + 3 * k;
            if (!Comms.fresh(Comms.read(b + 2), 30)) continue;
            MapLocation l = Comms.readLoc(b);
            if (l == null) continue;
            int d = G.me.distanceSquaredTo(l);
            if (d < bd) { bd = d; best = l; }
        }
        return best;
    }

    /** Move to a tile adjacent to the cat body, outside its forward lane. */
    /** A cat at c is within dist2 Cfg.CAT_DEFEND_R2 of one of our live kings. */
    static boolean catAtKing(MapLocation c) {
        for (int k = 0; k < Comms.numKings; k++)
            if (c.distanceSquaredTo(Comms.kings[k]) <= Cfg.CAT_DEFEND_R2) return true;
        return false;
    }

    static boolean approachCat(RobotInfo c) throws GameActionException {
        MapLocation me = G.me;
        if (chebToBody(c.location, me) == 1 && !inLane(c, me)) return true; // already in place: stay
        MapLocation best = null;
        int bd = Integer.MAX_VALUE;
        for (int dx = -1; dx <= 2; dx++)
            for (int dy = -1; dy <= 2; dy++) {
                if (dx >= 0 && dx <= 1 && dy >= 0 && dy <= 1) continue;
                MapLocation t = c.location.translate(dx, dy);
                if (!G.onMap(t.x, t.y) || inLane(c, t)) continue;
                if ((Sym.mem[G.idx(t)] & (Sym.WALL | Sym.DIRT)) != 0) continue;
                int d = me.distanceSquaredTo(t) * 2 + (G.pickBit(me, t) ? 1 : 0);
                if (d < bd) { bd = d; best = t; }
            }
        if (best == null) return false;
        Nav.goTo(best);
        return true;
    }

    static MapLocation pickMine() {
        // A mine drops 20 cheese about every 12 rounds into its 9x9 square, however many rats wait there, so
        // gatherers spread over the MINE_SPREAD nearest known mines on our half (cheesefarm: 4 mines, and every rat
        // used to crowd the nearest). Order: distance from home, ties by the mirror-invariant offset.
        int n = 0;
        for (int i = 0; i < Comms.numMines && n < cand.length; i++) n = addCand(Comms.mines[i], n);
        for (int i = 0; i < myMineCount && n < cand.length; i++) n = addCand(new MapLocation(myMines[i] / 64, myMines[i] % 64), n);
        if (n == 0) return null;
        int ours = 0;
        for (int i = 0; i < n; i++) if (homeHalf(cand[i])) { MapLocation t = cand[ours]; cand[ours++] = cand[i]; cand[i] = t; }
        int m = ours > 0 ? ours : n;
        long[] key = candKey;
        for (int i = 0; i < m; i++) key[i] = mineKey(cand[i], home);
        for (int i = 1; i < m; i++) {   // insertion sort of the first m by mineKey
            long k = key[i];
            MapLocation c = cand[i];
            int j = i;
            for (; j > 0 && k < key[j - 1]; j--) { key[j] = key[j - 1]; cand[j] = cand[j - 1]; }
            key[j] = k; cand[j] = c;
        }
        minesAtPick = Comms.numMines;
        if (D.ON) D.minePick(n, ours, m);   // MINE_PICK opens (secondary slot; BABY_MOVE is open)
        if (D.NET && D.point2 == D.MINE_PICK) {
            MapLocation p = minePickExec(D.pick2(D.MINE_PICK, null), n);
            if (p != null) { D.emit2(); return p; }
        }
        // Cfg.MINE_BALANCE: the rat's slot spreads the pick, shifted by Cfg.IDLE_SHIFT
        D.lab2 = (slot % m + mineShift) % m; D.emit2(); return cand[(slot % m + mineShift) % m];
    }

    /** One of our kings stands within Chebyshev 4 of mine m (a D feature). */
    static boolean kingHeld(MapLocation m) {
        for (int k = 0; k < Comms.numKings; k++) if (G.cheb(Comms.kings[k], m) <= 4) return true;
        return false;
    }

    /** A per-rat hash (the scout pick, Cfg.MINE_BALANCE), and the team's known-mine count when mineTarget was picked. */
    static int slot, minesAtPick;

    static final MapLocation[] cand = new MapLocation[40];
    static final long[] candKey = new long[40];

    static int addCand(MapLocation l, int n) {
        for (int i = 0; i < n; i++) if (cand[i].equals(l)) return n;
        cand[n] = l;
        return n + 1;
    }

    /** Sort key: squared distance from h, then the offset along our outward axes (same for mirrored rats). */
    static long mineKey(MapLocation m, MapLocation h) {
        return ((long) m.distanceSquaredTo(h) << 16) + ((m.x - h.x) * fu + 64) * 128 + (m.y - h.y) * fv + 64;
    }

    /** The site the king named for a second king, while its heartbeat is fresh. */
    static MapLocation formSite() {
        int t = Comms.read(Comms.FORM_T), x = Comms.read(Comms.FORM_X);
        if (t == 0 || x == 0 || !Comms.fresh(t, 2)) return null;
        return new MapLocation(x - 1, Comms.read(Comms.FORM_Y) - 1);
    }

    /** Join a king formation: walk into the site's 3x3 and hold; the rat on the centre promotes once 7 stand there. */
    static boolean formKing() throws GameActionException {
        MapLocation site = formSite();
        if (site == null) return false;
        int sd = G.me.distanceSquaredTo(site);
        if (sd > Cfg.FORM_PULL_R2) return false;
        if (G.cheb(G.me, site) > 1 && nearerAllies(site) >= Cfg.FORM_JOIN) return false;
        RobotController rc = G.rc;
        if (G.me.equals(site)) {
            if (rc.canBecomeRatKing()) { rc.becomeRatKing(); formed++; }
            return true;
        }
        if (G.cheb(G.me, site) <= 1) {
            // step onto the centre when it is free, else hold our ring tile facing it
            Direction d = G.me.directionTo(site);
            if (rc.canMove(d)) {
                if (rc.getDirection() != d && rc.canTurn(d)) { rc.turn(d); G.facing = d; }
                if (rc.canMove(d)) { rc.move(d); G.me = rc.getLocation(); }
            }
            return true;
        }
        Nav.goTo(site);
        return true;
    }

    /** Allied babies in view strictly nearer the site than we are (no id tie-break: mirrored rats decide alike). */
    static int nearerAllies(MapLocation site) {
        int d = G.me.distanceSquaredTo(site), n = 0;
        for (RobotInfo r : seen)
            if (r.team == G.us && r.type == UnitType.BABY_RAT && r.location.distanceSquaredTo(site) < d) n++;
        return n;
    }

    static int formed;

    // ---------------------------------------------------------------- Cfg.CASH_BANK: rescue formation

    /** Cfg.CASH_BANK: the rescue site we lead or heard (null: none), its start round, the round we last heard it,
     *  whether we lead it, the round before which no rescue runs (after one timed out), and rescues we promoted. */
    static MapLocation cashSite;
    static int cashStart, cashHeard, cashBan, cashed;
    static boolean cashLead;

    /** A rescue may run: Cfg.CASH_END's endgame second king (the low-bank rescue needs Cfg.CASH_KINGS, which is 0). */
    static boolean cashOk(int bank, int kings, int round) {
        return cashEnd(bank, kings, round);
    }

    /** Cfg.CASH_END: an endgame second king may run, whatever the bank holds in raw: from round CASH_END a lone king,
     *  and a bank that keeps a new king alive to round 2000 (both kings eat 2 a round; a starving king loses 10 health
     *  a round and a new one has up to 600, so it outlives 55 rounds on an empty bank). */
    static boolean cashEnd(int bank, int kings, int round) {
        return round >= Cfg.CASH_END && kings == 1
            && bank >= Math.max(Cfg.CASH_FLOOR, 4 * (2000 - round - 55) + 10);
    }

    /** SQ_CASH's check bits: a squeak whose tag does not match its site is not ours. */
    static int cashTag(int x, int y) { return (x * 3 + y * 5 + 17) & 63; }

    /** Of two rescue sites the one started first wins, then the lower tile key: two leaders in one crowd merge. */
    static boolean cashFirst(int start, int key, int start2, int key2) { return start < start2 || start == start2 && key < key2; }

    /** Cfg.CASH_BANK, in place of report(): hear rescue sites, lead one when we qualify, and squeak ours (true when we
     *  squeaked; the squeak carries the site's start round, so relays die with it). */
    static boolean cashTick() throws GameActionException {
        RobotController rc = G.rc;
        if (!cashOk(G.globalCheese, Comms.numKings, G.round) || G.round < cashBan) { cashSite = null; cashLead = false; return false; }
        if (cashSite != null && G.round - cashStart > Cfg.CASH_TTL) {
            cashSite = null;
            cashLead = false;
            cashBan = G.round + Cfg.CASH_TTL;
            return false;
        }
        MapLocation best = null;
        int bs = 0;
        for (Message m : rc.readSqueaks(-1)) {
            // Cfg.CASH_READ_LEFT: stop reading once the turn runs low (a ledgered budget read, as END_KING_CROWD's)
            if (Clock.getBytecodesLeft() + (G.debt += 8) - 6 < Cfg.CASH_READ_LEFT) break;
            int c = m.getBytes();
            if (Comms.sqType(c) != Comms.SQ_CASH || G.round - m.getRound() > 2) continue;
            int x = Comms.sqX(c), y = Comms.sqY(c), e = Comms.sqExtra(c);
            if ((e >>> 10) != cashTag(x, y)) continue;
            int start = G.round - ((((G.round & 1023) - (e & 1023)) + 1024) & 1023);
            if (G.round - start > Cfg.CASH_TTL) continue;
            if (best == null || cashFirst(start, x * 64 + y, bs, best.x * 64 + best.y)) { best = new MapLocation(x, y); bs = start; }
        }
        if (best != null && (cashSite == null
                || !best.equals(cashSite) && cashFirst(bs, best.x * 64 + best.y, cashStart, cashSite.x * 64 + cashSite.y))) {
            cashSite = best;
            cashStart = bs;
            cashLead = false;
        }
        if (best != null && best.equals(cashSite)) cashHeard = G.round;
        if (cashSite != null && !cashLead && G.round - cashHeard > 2) cashSite = null;
        if (cashSite == null) {
            int raw = rc.getRawCheese();
            boolean end = cashEnd(G.globalCheese, Comms.numKings, G.round);   // no raw needed: the king itself scores
            if (!end && raw < Cfg.CASH_RAW || !cashFar(G.me)) return false;
            int n = 1, sum = raw, id = rc.getID();
            for (RobotInfo r : seen) {
                if (r.team != G.us || r.type != UnitType.BABY_RAT || G.cheb(r.location, G.me) > Cfg.CASH_R) continue;
                if (r.cheeseAmount > raw || r.cheeseAmount == raw && r.ID < id) return false;
                n++;
                sum += r.cheeseAmount;
            }
            if (n < Cfg.CASH_N || !end && sum < Cfg.CASH_MIN) return false;
            MapLocation s = cashSpot(G.me);
            if (s == null) return false;
            cashSite = s;
            cashStart = G.round;
            cashHeard = G.round;
            cashLead = true;
        }
        if (G.cheb(G.me, cashSite) > Cfg.CASH_PULL) { if (!cashLead) cashSite = null; return false; }
        if (cat != null && catInEarshot(cat.location, G.me)) return false;
        return rc.squeak(Comms.squeak(Comms.SQ_CASH, cashSite.x, cashSite.y, cashTag(cashSite.x, cashSite.y) << 10 | (cashStart & 1023)));
    }

    /** p is at least Chebyshev CASH_FAR from every king's centre. */
    static boolean cashFar(MapLocation p) {
        for (int k = 0; k < Comms.numKings; k++) if (G.cheb(Comms.kings[k], p) < Cfg.CASH_FAR) return false;
        return true;
    }

    /** A rescue site at c or beside it: its 3x3 on the map, no wall, dirt, cat or king in it (unseen tiles by memory). */
    static MapLocation cashSpot(MapLocation c) throws GameActionException {
        if (cashClear(c)) return c;
        for (Direction d : G.fan(c, G.facing)) {
            MapLocation s = c.add(d);
            if (cashClear(s)) return s;
        }
        return null;
    }

    static boolean cashClear(MapLocation s) throws GameActionException {
        if (s.x < 1 || s.y < 1 || s.x > G.W - 2 || s.y > G.H - 2) return false;
        RobotController rc = G.rc;
        int[] mem = Sym.mem;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                MapLocation t = s.translate(dx, dy);
                if (rc.canSenseLocation(t)) {
                    if (!rc.senseMapInfo(t).isPassable()) return false;
                    RobotInfo r = rc.senseRobotAtLocation(t);
                    if (r != null && r.type != UnitType.BABY_RAT) return false;
                } else if ((mem[t.x + t.y * G.W] & (Sym.WALL | Sym.DIRT | Sym.KING)) != 0) return false;
            }
        return true;
    }

    /** The rat on the rescue site promotes (Cfg.CASH_BANK; me: its tile); true when it became a king. */
    static boolean cashPromote(MapLocation me) throws GameActionException {
        RobotController rc = G.rc;
        if (cashSite == null || !me.equals(cashSite) || !rc.canBecomeRatKing()) return false;
        rc.becomeRatKing();
        formed++;
        cashed++;
        return true;
    }

    /** Cfg.CASH_BANK: walk into the rescue site's 3x3 and hold (the rat on the centre promotes before moving). */
    static boolean cashMove() throws GameActionException {
        MapLocation site = cashSite;
        if (G.cheb(G.me, site) > Cfg.CASH_PULL) return false;
        RobotController rc = G.rc;
        if (G.me.equals(site)) return true;
        if (G.cheb(G.me, site) <= 1) {
            // step onto the centre when it is free, else hold our ring tile
            Direction d = G.me.directionTo(site);
            if (rc.canMove(d)) {
                if (rc.getDirection() != d && rc.canTurn(d)) { rc.turn(d); G.facing = d; }
                if (rc.canMove(d)) { rc.move(d); G.me = rc.getLocation(); }
            }
            return true;
        }
        Nav.goTo(site);
        return true;
    }

    static MapLocation raidTarget;
    static boolean raider;           // a fighter that leaves our half to contest theirs

    /** l is closer to our king than to its mirror under the first live symmetry. */
    static boolean homeHalf(MapLocation l) {
        return nearer(l, home, firstSym());
    }

    /** Whether l is nearer h than h's twin under symmetry s (ties do not count). */
    static boolean nearer(MapLocation l, MapLocation h, int s) {
        int a = l.distanceSquaredTo(h), b = l.distanceSquaredTo(G.mirror(h, s));
        return a < b;
    }

    static int firstSym() {
        return (Sym.mask & Sym.ROT) != 0 ? Sym.ROT : (Sym.mask & Sym.HOR) != 0 ? Sym.HOR : Sym.VER;
    }
    static int raidSince = -1;

    /** Idle fighter: contest the enemy half at a mine closer to the enemy king than to ours; wander there a while. */
    static boolean raid() throws GameActionException {
        if (raidTarget != null && (Nav.stall > 20 || raidSince >= 0 && G.round - raidSince > Cfg.RAID_STAY)) {
            raidTarget = null;
            raidSince = -1;
        }
        if (raidTarget == null) raidTarget = pickRaid();
        if (raidTarget == null) return false;
        Nav.guard = null;
        if (G.cheb(G.me, raidTarget) <= 2) {
            if (raidSince < 0) raidSince = G.round;
            wanderAround(raidTarget);
        } else {
            Nav.goTo(raidTarget);
        }
        return true;
    }

    /** A mine on the enemy half: known mines and the mirrors of our own, under the first live symmetry. */
    static MapLocation pickRaid() {
        int s = firstSym();
        MapLocation ek = G.mirror(home, s);
        MapLocation best = null;
        int bd = Integer.MAX_VALUE;
        for (int i = 0; i < Comms.numMines; i++) {
            MapLocation m = Comms.mines[i];
            for (int k = 0; k < 2; k++) {
                MapLocation c = k == 0 ? m : G.mirror(m, s);
                if (c.distanceSquaredTo(ek) >= c.distanceSquaredTo(home)) continue;
                int d = G.me.distanceSquaredTo(c) + G.rand(Cfg.RAID_SPREAD);
                if (d < bd) { bd = d; best = c; }
            }
        }
        return best != null ? best : new MapLocation((home.x + ek.x) / 2, (home.y + ek.y) / 2);
    }

    static void wanderAround(MapLocation m) throws GameActionException {
        if (exploreTarget == null || G.me.equals(exploreTarget) || G.cheb(exploreTarget, m) > 3 || Nav.stall > 6) {
            exploreTarget = new MapLocation(clampX(m.x + fu * (G.rand(7) - 3)), clampY(m.y + fv * (G.rand(7) - 3)));
        }
        Nav.goTo(exploreTarget);
    }

    /** At mine m: within Chebyshev 2 (Cfg.MINE_KING_R is 0). */
    static boolean atMine(MapLocation m) {
        return G.cheb(G.me, m) <= 2;
    }

    static void explore() throws GameActionException {
        if (exploreTarget == null || G.cheb(G.me, exploreTarget) <= 2 || Nav.stall > 20) {
            // target in our side frame: mirrored rats pick mirrored targets. Take the nearest never-seen
            // sample: once open ground is mapped, what is left unseen is buried in dirt, and Flow digs to it.
            // With no mine known yet, rank by closeness to the map centre instead: mines that nobody has seen
            // are buried between the kings, and shallow edge pockets do not reveal them.
            boolean centre = Comms.numMines == 0;
            int ox = centre ? (G.W - 1) / 2 : G.me.x, oy = centre ? (G.H - 1) / 2 : G.me.y;
            int x = 0, y = 0, bd = Integer.MAX_VALUE;
            for (int k = 0; k < Cfg.EXPLORE_TRIES; k++) {
                int ax = G.rand(G.W), ay = G.rand(G.H);
                int cx = fu > 0 ? ax : G.W - 1 - ax, cy = fv > 0 ? ay : G.H - 1 - ay;
                int d = Sym.known(G.idx(cx, cy)) ? 1 << 20 : Math.max(Math.abs(cx - ox), Math.abs(cy - oy));
                if (d < bd) { bd = d; x = cx; y = cy; }
            }
            exploreTarget = new MapLocation(x, y);
        }
        Nav.goTo(exploreTarget);
    }

    static boolean scout;

    /** A gatherer scout still scouting: until round SCOUT_UNTIL. */
    static boolean scouting() {
        return scout && role == GATHER && G.round < Cfg.SCOUT_UNTIL;
    }

    /** Scout movement: the first never-seen tile of our half among EXPLORE_TRIES samples (Cfg.HOME_PICK 1); false
     *  (no move) when no sample is one. */
    static boolean exploreHome() throws GameActionException {
        if (exploreTarget == null || G.cheb(G.me, exploreTarget) <= 2 || Nav.stall > 20 || Sym.known(G.idx(exploreTarget.x, exploreTarget.y))) {
            int x = -1, y = -1;
            for (int k = 0; k < Cfg.EXPLORE_TRIES; k++) {
                int ax = G.rand(G.W), ay = G.rand(G.H);
                int cx = fu > 0 ? ax : G.W - 1 - ax, cy = fv > 0 ? ay : G.H - 1 - ay;
                if (Sym.known(G.idx(cx, cy)) || !homeHalf(new MapLocation(cx, cy))) continue;
                x = cx; y = cy;
                break;
            }
            if (x < 0) return false;
            exploreTarget = new MapLocation(x, y);
        }
        Nav.goTo(exploreTarget);
        return true;
    }

    /** Mine seeking (Cfg.SEEK_CROWD) runs until round seekUntil and may start again from seekNext; seekMines is
     *  myMineCount when it started. */
    static int seekUntil, seekNext, seekMines;

    /** Cfg.SEEK_IDLE: turns spent near the mine since the last pickup (idlePicked: Baby.picked then); mineShift is
     *  Cfg.IDLE_SHIFT's offset into the pick list. */
    static int idleTurns, idlePicked, mineShift;

    /** Cfg.SEEK_IDLE: count a turn within SEEK_IDLE_R of the mine with no pickup since the last; true (and the count
     *  restarts) on the SEEK_IDLE-th. */
    static boolean idle(MapLocation mine) {
        if (picked != idlePicked) { idlePicked = picked; idleTurns = 0; }
        if (G.cheb(G.me, mine) > Cfg.SEEK_IDLE_R || ++idleTurns < Cfg.SEEK_IDLE) return false;
        idleTurns = 0;
        return true;
    }

    /** Cfg.SEEK_IDLE: an idle gatherer starts seeking unless it sought within SEEK_RETRY rounds. */
    static boolean idleSeek() {
        if (G.round < seekNext) return false;
        seekUntil = G.round + Cfg.SEEK_TTL;
        seekNext = G.round + Cfg.SEEK_RETRY;
        seekMines = myMineCount;
        exploreTarget = null;
        return true;
    }

    /** A gatherer at a crowded mine starts seeking: SEEK_CROWD allied babies nearer the mine than us. */
    static boolean startSeek(MapLocation mine) {
        if (G.round < seekNext || nearerAllies(mine) < Cfg.SEEK_CROWD) return false;
        seekUntil = G.round + Cfg.SEEK_TTL;
        seekNext = G.round + Cfg.SEEK_RETRY;
        seekMines = myMineCount;
        exploreTarget = null;
        return true;
    }

    /** One seeking step; false once it is over: a mine found and reported to the king, or nothing left unseen. */
    static boolean seek() throws GameActionException {
        if (myMineCount > seekMines) {
            // squeaks reach the king from dist2 16 only: carry the news home, then gather (the new mine included)
            MapLocation k = Comms.nearestKing(G.me);
            if (k != null && unreported() >= 0) { Nav.goTo(k); return true; }
            mineTarget = null;
        } else if (exploreHome()) return true;
        seekUntil = 0;
        return false;
    }

    static int clampX(int x) { return Math.max(0, Math.min(G.W - 1, x)); }

    static int clampY(int y) { return Math.max(0, Math.min(G.H - 1, y)); }

    // ---------------------------------------------------------------- net-mode executors (D.NET builds only)
    // Each runs one option's behaviour standalone, behind the validity guards it needs; true = it acted (the option
    // owns the decision), false = zdb's own code runs as the fallback. Pass 1 wires them only behind D.NET.

    /** BABY_ROLE: set the role option k (no guard needed). */
    static void roleExec(int k) {
        switch (k) {
            case D.R_GATHER: role = GATHER; scout = false; raider = false; break;
            case D.R_SCOUT: role = GATHER; scout = true; raider = false; break;
            case D.R_FIGHT: role = FIGHT; raider = false; break;
            case D.R_RAIDER: role = FIGHT; raider = true; break;
            case D.R_ATTACK: if (role != ATTACK) ekSeen = G.round; role = ATTACK; raider = false; break;
            default: break;
        }
    }

    /** BABY_ACT_PRE / BABY_ACT_POST: run action option k (robots null: sense here). A_PASS keeps the action unused. */
    static boolean actExec(int k, RobotInfo[] robots) throws GameActionException {
        RobotController rc = G.rc;
        if (k == D.A_PASS) return true;
        if (!rc.isActionReady()) return false;
        if (robots == null) robots = rc.senseNearbyRobots(-1, null);
        MapLocation me = rc.getLocation(), l = me.add(rc.getDirection());
        switch (k) {
            case D.A_DROP_ALLY: {
                RobotInfo c = rc.getCarrying();
                if (c == null || c.team != G.us) return false;
                dropAlly();
                return rc.getCarrying() == null;
            }
            case D.A_THROW: {
                RobotInfo c = rc.getCarrying();
                if (c == null || c.team == G.us) return false;
                Direction f0 = rc.getDirection();
                return tryThrow(robots) || rc.getDirection() != f0;
            }
            case D.A_NAP: return tryCarry(robots);
            case D.A_BITE_KING: return biteSome(robots, true);
            case D.A_BITE_RAT: return biteSome(robots, false);
            case D.A_BITE_CAT: {
                if (cat == null) return false;
                Direction f0 = rc.getDirection();
                boolean b = tryBiteCat(cat);
                G.facing = rc.getDirection();
                return b || rc.getDirection() != f0;
            }
            case D.A_CAT_TRAP: {
                if (cat == null) return false;
                int tp = trapsPlaced;
                tryCatTrap(cat);
                return trapsPlaced != tp;
            }
            case D.A_LAY_LANE: { int tp = trapsPlaced; layLane(); return trapsPlaced != tp; }
            case D.A_DELIVER: { int raw = rc.getRawCheese(); tryDeliver(); return rc.getRawCheese() != raw; }
            case D.A_CASH_KING: return cashPromote(me);
            case D.A_RAT_TRAP: return tryRatTrap();
            case D.A_X_BITE_FWD:
                if (!rc.onTheMap(l) || !rc.canAttack(l)) return false;
                rc.attack(l);
                return true;
            case D.A_X_CARRY_FWD:
                if (!rc.onTheMap(l) || !rc.canCarryRat(l)) return false;
                rc.carryRat(l); carryRound = G.round;
                return true;
            case D.A_X_THROW_NOW:
                if (!rc.canThrowRat()) return false;
                rc.throwRat(); throwsDone++;
                return true;
            case D.A_X_DROP_FWD: {
                Direction d = rc.getDirection();
                if (d == Direction.CENTER || !rc.canDropRat(d)) return false;
                rc.dropRat(d);
                return true;
            }
            case D.A_X_RAT_TRAP_FWD:
                if (!rc.onTheMap(l) || !rc.canPlaceRatTrap(l)) return false;
                rc.placeRatTrap(l); ratTraps++;
                return true;
            case D.A_X_CAT_TRAP_FWD:
                if (!rc.onTheMap(l) || !rc.canPlaceCatTrap(l)) return false;
                rc.placeCatTrap(l); trapsPlaced++;
                return true;
            case D.A_X_DIRT_PLACE_FWD:
                if (!rc.onTheMap(l) || !rc.canPlaceDirt(l)) return false;
                rc.placeDirt(l); Sym.mem[G.idx(l)] |= Sym.DIRT;
                return true;
            case D.A_X_DIRT_REMOVE_FWD:
                if (!rc.onTheMap(l) || !rc.canRemoveDirt(l)) return false;
                rc.removeDirt(l); Sym.mem[G.idx(l)] &= ~Sym.DIRT;
                return true;
            case D.A_X_BECOME_KING:
                if (!rc.canBecomeRatKing()) return false;
                rc.becomeRatKing(); formed++;
                return true;
            default: return false;
        }
    }

    /** actExec's bites: biteEnemies over the enemy kings (kings) or the enemy rats (!kings) in robots; true when it
     *  bit or turned to bite. */
    static boolean biteSome(RobotInfo[] robots, boolean kings) throws GameActionException {
        int m = 0;
        for (RobotInfo r : robots) if (r.team == G.them && (r.type == UnitType.RAT_KING) == kings) m++;
        if (m == 0) return false;
        RobotInfo[] a = new RobotInfo[m];
        m = 0;
        for (RobotInfo r : robots) if (r.team == G.them && (r.type == UnitType.RAT_KING) == kings) a[m++] = r;
        Direction f0 = G.rc.getDirection();
        return biteEnemies(a) || G.rc.getDirection() != f0;
    }

    /** BABY_FACE: face option k. */
    static boolean faceExec(int k) throws GameActionException {
        RobotController rc = G.rc;
        switch (k) {
            case D.F_NONE: return true;
            case D.F_SEEN: {
                MapLocation t = seenThreat();
                return t != null && faceTo(t);
            }
            case D.F_MEMORY: {
                MapLocation t = memoryThreat();
                return t != null && faceTo(t);
            }
            case D.F_HEARD: return false;   // nothing is heard: Cfg.TRAP_SHARE is off
            case D.F_GUARD: return Nav.guard != null && faceTo(Nav.guard);
            default:
                if (k < D.F_DIR0 || k >= D.F_DIR0 + 8) return false;
                Direction to = G.DIRS[k - D.F_DIR0];
                if (rc.getDirection() != to && rc.canTurn(to)) { rc.turn(to); G.facing = to; }
                return rc.getDirection() == to;
        }
    }

    /** Turn to face t; true when we face it. */
    static boolean faceTo(MapLocation t) throws GameActionException {
        RobotController rc = G.rc;
        Direction to = G.me.directionTo(t);
        if (to == Direction.CENTER) return false;
        if (rc.getDirection() != to && rc.canTurn(to)) { rc.turn(to); G.facing = to; }
        return rc.getDirection() == to;
    }

    /** BABY_COMMS: squeak option k (true when it squeaked). C_NONE cannot silence zdb's report in pass 1: false. */
    static boolean commsExec(int k) throws GameActionException {
        RobotController rc = G.rc;
        switch (k) {
            case D.C_MINE_GOSSIP:
                return myMineCount > 0 && squeakMine(myMines[G.round % myMineCount]);
            case D.C_MINE_RELAY: {
                int key = unreported();
                return key >= 0 && squeakMine(key);
            }
            case D.C_SYM_KING:
            case D.C_SYM_GOSSIP:
                return symNews() && squeakSym();
            case D.C_MINE_KING: {
                int key = unlisted();
                return key >= 0 && squeakMine(key);
            }
            case D.C_CAT:
                return catNews() && squeakCat();
            case D.C_CASH_LEAD:
            case D.C_CASH_RELAY: {
                if (k == D.C_CASH_LEAD && cashSite == null) {
                    MapLocation sp = cashSpot(G.me);
                    if (sp == null) return false;
                    cashSite = sp; cashStart = cashHeard = G.round; cashLead = true;
                }
                if (cashSite == null || G.cheb(G.me, cashSite) > Cfg.CASH_PULL) return false;
                return rc.squeak(Comms.squeak(Comms.SQ_CASH, cashSite.x, cashSite.y,
                    cashTag(cashSite.x, cashSite.y) << 10 | (cashStart & 1023)));
            }
            case D.C_EKING:
                if (eKing == null) return false;
                return rc.squeak(Comms.squeak(Comms.SQ_EKING, eKing.location.x, eKing.location.y, (G.round & 1023) << 2));
            default: return false;   // C_NONE, C_BURN, C_TRIP, C_TRAP (Cfg.TRAP_SHARE off): never squeak
        }
    }

    /** MINE_PICK: candidate rank k of pickMine's cand[0..n) (sets the label), else null (zdb's pick). */
    static MapLocation minePickExec(int k, int n) {
        if (k < 0 || k >= n) return null;
        D.lab2 = k;
        return cand[k];
    }

    /** BABY_MOVE net option mask (sized D.BABY_MOVE_NET_OPTIONS). */
    static final boolean[] netOk = new boolean[D.BABY_MOVE_NET_OPTIONS];

    /** Net-mode movement: the net's option, or zdb's cascade when it cannot act (label 0x8000 | k << 6 | zdb's). */
    static void netMove() throws GameActionException {
        int k = -1;
        if (D.point == D.BABY_MOVE) { netAllowed(); k = D.pick(D.BABY_MOVE, netOk); }
        boolean acted = false;
        if (k >= 0) {
            why = '-';
            try { acted = netExec(k); } catch (GameActionException e) { acted = false; }
        }
        if (acted) D.lab = k;
        else {
            moveDecision();
            if (k >= 0) D.lab = 0x8000 | (k << 6) | (D.lab & 63);
        }
    }

    /** Fill netOk: side-effect-free applicability of each BABY_MOVE option. */
    static void netAllowed() {
        RobotController rc = G.rc;
        boolean[] a = netOk;
        RobotInfo c = rc.getCarrying();
        MapLocation k = Comms.nearestKing(G.me);
        boolean kc = knownCat() != null, mineOk = mineTarget != null || Comms.numMines + myMineCount > 0;
        int raw = rc.getRawCheese();
        a[D.M_GRAB] = false;
        a[D.M_CAT_DEFEND] = cat != null;
        a[D.M_END_CAT] = cat != null || kc;
        a[D.M_FLEE_CAT] = cat != null;
        a[D.M_YIELD_KING] = cat != null;
        a[D.M_AIM_CARRY] = c != null && c.team == G.them;
        a[D.M_RESCUE] = false;   // no rescue target is ever set (Cfg.RESCUE off)
        a[D.M_AMMO_AWAY] = nearestEnemy != null && k != null;
        a[D.M_RETREAT] = k != null && nearEnemy(20);
        a[D.M_HUNGRY] = false;   // alias of M_DELIVER
        a[D.M_FORM_KING] = formSite() != null;
        a[D.M_CASH] = cashSite != null;
        a[D.M_RAID] = true;
        a[D.M_FLANK] = nearestEnemy != null && rc.isActionReady() && c == null;
        a[D.M_PRESS] = eKing != null || lastEKing != null && G.round - lastEKingRound <= Cfg.PRESS_AGE;
        a[D.M_MICRO] = nearestEnemy != null;
        a[D.M_FACE] = nearEnemy(8);
        a[D.M_ATK_DELIVER] = false;   // alias of M_DELIVER
        a[D.M_STALL_CAT] = role == ATTACK;
        a[D.M_ATTACK] = true;
        a[D.M_WAR] = nearestEnemy != null || eKing != null || threatenedKing() != null || Comms.fresh(Comms.read(Comms.ALERT + 2), 3);
        a[D.M_FORT] = false;   // Cfg.FORT_R is 0
        a[D.M_FIGHT_CAT] = cat != null;
        a[D.M_DELIVER] = raw > 0 && k != null;
        a[D.M_COURIER] = k != null;
        a[D.M_CHEESE_PICK] = cheeseTarget != null && G.cheb(G.me, cheeseTarget) == 1;
        a[D.M_CHEESE] = cheeseTarget != null;
        a[D.M_KNOWN_CAT] = kc;
        a[D.M_RAID_FIGHT] = true;
        a[D.M_SCOUT_HOME] = k != null;
        a[D.M_SCOUT] = true;
        a[D.M_SEEK] = true;
        a[D.M_MINE_EXPLORE] = true;
        a[D.M_IDLE_SEEK] = G.round >= seekNext;
        a[D.M_START_SEEK] = mineTarget != null && G.round >= seekNext;
        a[D.M_MINE_STILL] = true;
        a[D.M_MINE_WANDER] = mineOk;
        a[D.M_MINE_GO] = mineOk;
        a[D.M_EXPLORE] = true;
        for (int i = 0; i < 8; i++) {
            Direction d = G.DIRS[i];
            a[D.P_STEP0 + i] = rc.canMove(d) && (Nav.avoid & (1 << i)) == 0;
            a[D.P_TURN0 + i] = rc.isTurningReady() && rc.getDirection() != d;
        }
        a[D.P_HOLD] = true;
    }

    /** Run BABY_MOVE option k; true when it took the movement decision (as the cascade's return after it). */
    static boolean netExec(int k) throws GameActionException {
        RobotController rc = G.rc;
        switch (k) {
            case D.M_GRAB: return false;
            case D.M_CAT_DEFEND:
                if (cat == null || !approachCat(cat)) return false;
                why = 'g'; return true;
            case D.M_END_CAT:
                if (!goCat()) return false;
                why = 'e'; return true;
            case D.M_FLEE_CAT:
                if (cat == null || !Nav.flee(cat.location)) return false;
                why = 'c'; return true;
            case D.M_YIELD_KING:
                if (cat == null || !yieldKing()) return false;
                why = 'y'; return true;
            case D.M_AIM_CARRY: {
                RobotInfo c = rc.getCarrying();
                if (c == null || c.team != G.them || !aimCarry()) return false;
                why = 'T'; return true;
            }
            case D.M_RESCUE: return false;   // rescueT is never set (Cfg.RESCUE off)
            case D.M_AMMO_AWAY:
                if (nearestEnemy == null || !ammoAway()) return false;
                why = 'A'; return true;
            case D.M_RETREAT:
                if (!retreat()) return false;
                why = 'r'; return true;
            case D.M_HUNGRY:
                if (rc.getRawCheese() <= 0 || !deliver()) return false;
                why = 'H'; return true;
            case D.M_FORM_KING:
                if (!formKing()) return false;
                why = 'K'; return true;
            case D.M_CASH:
                if (cashSite == null || !cashMove()) return false;
                why = 'Q'; return true;
            case D.M_RAID:
            case D.M_RAID_FIGHT:
                if (!raid()) return false;
                why = 'z'; return true;
            case D.M_FLANK:
                if (seen == null || !flank(seen)) return false;
                why = 'k'; return true;
            case D.M_PRESS:
                if (!pressKing()) return false;
                why = 'p'; return true;
            case D.M_MICRO:
                if (seen == null || !Micro.act(seen)) return false;
                why = 'm'; return true;
            case D.M_FACE:
                if (!faceEnemy()) return false;
                why = 'f'; return true;
            case D.M_ATK_DELIVER:
                return rc.getRawCheese() > 0 && deliver();
            case D.M_STALL_CAT:
                if (!stallCat()) return false;
                why = 'g'; return true;
            case D.M_ATTACK:
                if (!attack()) return false;
                why = 'a'; return true;
            case D.M_WAR:
                if (!war()) return false;
                why = 'w'; return true;
            case D.M_FORT: return false;   // Cfg.FORT_R is 0
            case D.M_FIGHT_CAT:
                if (cat == null) return false;
                why = 'g';
                return approachCat(cat);
            case D.M_DELIVER:
                if (rc.getRawCheese() <= 0) return false;
                why = 'g';
                return deliver();
            case D.M_COURIER: {
                MapLocation kl = Comms.nearestKing(G.me);
                if (kl == null) return false;
                Nav.goTo(kl); why = 'm'; return true;
            }
            case D.M_CHEESE_PICK:
                if (cheeseTarget == null || G.cheb(G.me, cheeseTarget) != 1) return false;
                pickAdjacent();
                return true;
            case D.M_CHEESE:
                if (cheeseTarget == null) return false;
                chaseCheese();
                return true;
            case D.M_KNOWN_CAT: {
                MapLocation ct = knownCat();
                if (ct == null) return false;
                why = 'g';
                Nav.goTo(ct); return true;
            }
            case D.M_SCOUT_HOME: {
                MapLocation kl = Comms.nearestKing(G.me);
                if (kl == null) return false;
                Nav.goTo(kl); why = 's'; return true;
            }
            case D.M_SCOUT:
                why = 'S';
                if (!exploreHome()) explore();
                return true;
            case D.M_SEEK:
                if (!seek()) return false;
                why = 'x'; return true;
            case D.M_MINE_EXPLORE:
                if (!exploreHome()) return false;
                mineTarget = null; why = 'x'; return true;
            case D.M_IDLE_SEEK:
                if (!idleSeek() || !seek()) return false;
                why = 'x'; return true;
            case D.M_START_SEEK:
                if (mineTarget == null || !startSeek(mineTarget) || !seek()) return false;
                why = 'x'; return true;
            case D.M_MINE_STILL: {
                // hold still and sweep the cone around the spawn square (zdb's Cfg.GATHER_STILL, off in the cascade)
                Direction t = G.facing.rotateRight().rotateRight();
                if (rc.canTurn(t)) { rc.turn(t); G.facing = t; }
                return true;
            }
            case D.M_MINE_WANDER:
                if (!netMine()) return false;
                wanderAround(mineTarget);
                why = 'W'; return true;
            case D.M_MINE_GO:
                if (!netMine()) return false;
                why = 'M';
                goMine();
                return true;
            case D.M_EXPLORE:
                why = 'E'; explore(); return true;
            case D.P_HOLD:
                why = 'v'; return true;
            default:
                if (k >= D.P_STEP0 && k < D.P_STEP0 + 8) {
                    Direction d = G.DIRS[k - D.P_STEP0];
                    if (!rc.canMove(d)) return false;
                    if (rc.getDirection() != d && rc.canTurn(d)) rc.turn(d);
                    if (!rc.canMove(d)) return false;
                    rc.move(d);
                    G.me = rc.getLocation(); G.facing = rc.getDirection();
                    Nav.stall++;
                    why = 'v'; return true;
                }
                if (k >= D.P_TURN0 && k < D.P_TURN0 + 8) {
                    Direction d = G.DIRS[k - D.P_TURN0];
                    if (rc.getDirection() == d || !rc.canTurn(d)) return false;
                    rc.turn(d); G.facing = d;
                    why = 'v'; return true;
                }
                return false;
        }
    }

    /** The cascade's mine refresh and idle shift (its idle-seek branch is the separate option M_IDLE_SEEK); true with
     *  a mine. */
    static boolean netMine() {
        refreshMine();
        if (mineTarget != null && (role == GATHER || role == FIGHT) && idle(mineTarget)) {
            shiftMine();
            if (mineTarget == null) { D.pickSite = 2; mineTarget = pickMine(); }
        }
        return mineTarget != null;
    }
}
