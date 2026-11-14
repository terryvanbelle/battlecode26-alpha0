package zdp;

import battlecode.common.*;

/** Baby rat: gather and deliver cheese, fight cats, lay cat traps, explore. */
public final class Baby {
    public static final int GATHER = 0, FIGHT = 1, ATTACK = 2;
    public static int role = -1;
    static int birthRound;
    /** Cfg.RUSH_LIFE: the round this attacker last saw the enemy king, and the attackers demoted to gatherers. */
    static int ekSeen = -10000, demoted;
    /** Cfg.STALL_CAT: the tile this attacker has stayed near since stallSince, and whether it hunts cats now. */
    static MapLocation stallAt;
    static int stallSince, catHunters;
    static boolean catHunt;
    static MapLocation home;          // spawning king's location at birth
    /** Cfg.TRIP_HALF: the round this rat's walk out began (birth or last delivery; 0: none under way), a timed walk not
     *  squeaked yet (tripT steps to mine tripKey = x * 64 + y; 0: none), and Comms.tripSig at the last mine pick. */
    static int tripFrom, tripKey, tripT, tripSigAtPick;
    static MapLocation origin;        // our first king's start, squeaked by the spawning king (Cfg.HOME_START)
    static int fu = 1, fv = 1;        // side frame (signs of our king's offset from centre)

    static MapLocation cheeseTarget;  // last seen cheese tile
    /** Cheese given up after a stall (Cfg.CHEESE_BAN): a ring of the latest, with the round each was banned. */
    static final MapLocation[] banned = new MapLocation[4];
    static final int[] bannedAt = new int[4];
    static int banHead;
    /** Cfg.BREAKOUT_TTL: breaking out (no chase of walled-off cheese; explore by frontier search) until this round;
     *  wallStrikes counts chases of walled-off cheese given up on a stall, the last in round lastStrike. */
    static int breakoutUntil = -1, wallStrikes, lastStrike = -1000;
    static MapLocation mineTarget;    // sticky assigned mine
    static boolean coopAtPick;        // Cfg.FIGHT_POST: the mine was picked during cooperation
    /** Cfg.FIGHT_POST (a field so tests can set it). */
    static int postMargin = Cfg.FIGHT_POST;
    /** Cfg.LOAD_AT (a field so tests can set it). */
    static int loadAt = Cfg.LOAD_AT;
    static MapLocation exploreTarget;
    static MapLocation catSeen;       // bottom-left of nearest cat sensed this turn
    static RobotInfo cat;
    static int catSeenRound = -100;
    /** CAT_REACH: no cat branch before this round; reachTurns is Cfg.CAT_REACH (a field, so tests can set it). */
    static int catBanUntil = -1, reachTurns = Cfg.CAT_REACH;
    /** AIM_LEFT: aimCarry's bytecode floor (a field, so tests can set it). */
    static int aimLeft = Cfg.AIM_LEFT;
    static MapLocation lastCatTarget;

    // mines this rat has sensed (to report to kings)
    static int[] myMines = new int[24];
    static int lowBc;   // turns cut short for lack of bytecodes
    static int myMineCount;
    static int reportedSym = 7;

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
        if (Cfg.WAR_STRIKE > 0 && !G.coop && striker(role, birthRound, Cfg.WAR_STRIKE)) role = ATTACK;
        if (Cfg.TRAIL) Trail.record();
        Comms.readKings();
        if (Cfg.KING_BLOCK || Cfg.KING_FLOW) Nav.markKings();
        Comms.readMines();
        int s = Comms.read(Comms.SYM);
        if (s != 0) Sym.merge(s);

        Debug.mark(1);
        sense();
        if (Cfg.SYM_MINELIST && Flow.left() > Cfg.BC_OPTIONAL) Sym.listCheck(Comms.mines, Comms.numMines, false);
        if (Cfg.CAT_BLIND && rc.isTurningReady() && catHit(hpEnd - rc.getHealth(), Cfg.CAT_BLIND_DMG) && blindHurt(rc.getHealth(), hpEnd, rc.senseNearbyRobots(-1, null), G.them)) glance();
        RobotInfo[] robots = rc.senseNearbyRobots(-1, null);
        seen = robots;
        RobotInfo enemyKing = null, enemy = null;
        cat = null;
        int bestCat = Integer.MAX_VALUE;
        for (RobotInfo r : robots) {
            if (r.team == Team.NEUTRAL) {
                if (Cfg.CAT_LAY && G.coop) CatLane.see(r.location, G.round);
                int d = catDist4(r.location, G.me);
                if (d < bestCat) { bestCat = d; cat = r; }
            } else if (r.team == G.them) {
                if (r.type == UnitType.RAT_KING) enemyKing = r;
                else if (enemy == null || r.health < enemy.health) enemy = r;
            }
        }
        if (cat != null) { catSeen = cat.location; catSeenRound = G.round; }
        if (enemy != null) threatRound = G.round;

        Debug.mark(2);
        if (role != ATTACK) pickUp();
        Nav.avoid = cat != null ? laneMask(cat) : 0;
        if (Cfg.SWAP_YIELD && Nav.swapHold > 0 && rc.getCarrying() == null) Nav.avoid |= carrierMask(robots);
        // a newborn's first turn pays for array allocation and a full first sense: skip optional work when short
        if (!G.coop && Cfg.TRAP_RISK && (Cfg.RISK_MIN_AREA == 0 || G.W * G.H >= Cfg.RISK_MIN_AREA) && Flow.left() > Cfg.BC_OPTIONAL) {
            TrapRisk.observe(robots);
        }

        Debug.mark(3);
        eKing = enemyKing;
        if (Cfg.RUSH_LIFE > 0 && role == ATTACK) {
            if (enemyKing != null) ekSeen = G.round;
            // a rush that has not found the enemy king in RUSH_LIFE rounds has failed for this rat: it gathers
            else if (rushFailed(G.round, birthRound, ekSeen, Cfg.RUSH_LIFE)) { role = GATHER; demoted++; }
        }
        if (Cfg.RUSH_BURN > 0 && role == ATTACK && trapsKnown(burnRound, Comms.read(Comms.SYM))
                && rushStands(G.me, ref(), Cfg.RUSH_BURN)) { role = GATHER; demoted++; }
        if (D.ON && (role != roleAt0 || G.round > birthRound && (G.round - birthRound) % 10 == 0)) {   // BABY_ROLE review
            D.babyRole(1, roleAt0);
            if (D.NET) { int k = D.pick(D.BABY_ROLE, null); if (k >= 0) roleExec(k); }
            D.lab = D.roleLab();
            D.emit();
        }
        if (enemyKing != null && Cfg.KING_PRESS) { lastEKing = enemyKing.location; lastEKingRound = G.round; ekHops = 0; }
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
            if (Cfg.POUNCE && rc.getCarrying() == null) {
                RobotInfo p = prey(robots, rc.getHealth());
                // gatherers only grab prey that is close; fighters hunt it across their vision
                if (p != null && (role != GATHER || G.me.distanceSquaredTo(p.location) <= 8)) {
                    nearestEnemy = p;
                    hunting = true;
                }
            }
            guardKing = null;
            if (Cfg.GUARD_KING_R > 0 && rc.getCarrying() == null) {
                RobotInfo b = kingBiter(robots, Cfg.GUARD_KING_R);
                if (b != null) { nearestEnemy = b; hunting = true; guards++; }
            }
            rescueT = Cfg.RESCUE && rc.getCarrying() == null ? rescueTarget(robots) : null;
            if (Cfg.GUARD_CARRY_R > 0) {
                guardCarry = rescueT == null && rc.getCarrying() == null && (rescueT = kingCarrier(robots, Cfg.GUARD_CARRY_R, Cfg.GUARD_CARRY_REACH)) != null;
                if (guardCarry) carryGuards++;
            }
        }
        Debug.mark(30);
        if (Flow.left() < Cfg.BC_BAIL) { lowBc++; return; }
        Nav.guard = null;
        if (!G.coop && Cfg.GUARD_DIST2 > 0 && nearestEnemy != null && rc.getCarrying() == null
                && G.me.distanceSquaredTo(nearestEnemy.location) <= Cfg.GUARD_DIST2) Nav.guard = nearestEnemy.location;
        boolean engaged = nearestEnemy != null && G.me.distanceSquaredTo(nearestEnemy.location) <= 20;
        if (Cfg.RISK_ADAPT > 0) adaptRisk = false;
        if (!G.coop && Cfg.TRAP_RISK && (Cfg.RISK_MIN_AREA == 0 || G.W * G.H >= Cfg.RISK_MIN_AREA) && Nav.stall < 8) {
            if (!(hunting && Cfg.HUNT_IGNORES_RISK) && (Cfg.RISK_IN_FIGHT || !engaged || Cfg.RISK_GATHER && role == GATHER))
                Nav.avoid |= TrapRisk.mask(G.me);
            else if (Cfg.RISK_FRESH > 0) Nav.avoid |= TrapRisk.mask(G.me, Cfg.RISK_FRESH);
            else if (Cfg.RISK_ADAPT > 0 && (adaptRisk = trapsKnown(burnRound, Comms.read(Comms.SYM))))
                Nav.avoid |= TrapRisk.mask(G.me, Cfg.RISK_ADAPT);
        }
        if (Cfg.WOUNDED_HP > 0 && !G.coop && rc.getHealth() <= Cfg.WOUNDED_HP && trapsKnown(burnRound, Comms.read(Comms.SYM))) {
            Nav.avoid |= woundMask(G.me, robots);
            wounded++;
        }
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
        if (hostile && Cfg.NAP) tryCarry(robots);
        // a rescue one step away keeps the action for the grab after the move
        boolean saving = rescueT != null && rc.isMovementReady() && rc.isActionReady();
        if (Cfg.GRAB_MICRO && hostile && !saving && GrabMicro.act(robots)) saving = true;
        if (!G.coop && Cfg.BABY_RAT_TRAPS && trapOk && !saving) tryRatTrap();
        Debug.mark(5);
        if (hostile && !saving) biteEnemies(robots);
        else if (!(Cfg.PEACE && G.coop) && role == ATTACK && enemyKing != null) biteEnemies(new RobotInfo[]{enemyKing});
        if (cat != null) tryBiteCat(cat);
        // 2. cat trap ahead of an approaching cat
        if (cat != null && Cfg.CAT_TRAPS) tryCatTrap(cat);
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
        boolean cashSq = Cfg.CASH_BANK > 0 && cashTick();
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
        if (!netAct && Cfg.CASH_BANK > 0 && cashSite != null && G.me.equals(cashSite) && rc.canBecomeRatKing()) {
            rc.becomeRatKing();
            formed++;
            cashed++;
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
        // TRIP_HALF times only walks spent walking to the mine: any other errand on the way voids the timing
        if (Cfg.TRIP_HALF != 0 && tripFrom > 0 && tripFrom < G.round && why != 'M') tripFrom = 0;
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
        if (hostile && Cfg.NAP) tryCarry(rc.senseNearbyRobots(2, G.them));
        Debug.mark(22);
        if (Cfg.GRAB_MICRO && GrabMicro.round == G.round) { }
        else if (hostile) biteEnemies(rc.senseNearbyRobots(2, G.them));
        else if (!(Cfg.PEACE && G.coop) && role == ATTACK) {
            RobotInfo[] near = rc.senseNearbyRobots(8, G.them);
            for (RobotInfo r : near) if (r.type == UnitType.RAT_KING) biteEnemies(new RobotInfo[]{r});
        }
        if (cat != null) tryBiteCat(cat);
        if (Cfg.CAT_LAY && G.coop) layLane();
        Debug.mark(8);
        tryDeliver();
        }
        D.emit();   // BABY_ACT_POST closes
        if (D.ON && !G.coop && Cfg.GUARD_DIST2 != 0 && rc.isTurningReady() && rc.getCarrying() == null) D.babyFace(moved);
        if (D.NET && D.point == D.BABY_FACE) {
            int k = D.pick(D.BABY_FACE, null);
            if (k >= 0 && faceExec(k)) D.lab = k; else guardTurn();
        } else guardTurn();
        D.emit();   // BABY_FACE closes
        if (Cfg.MOAT && !G.coop) Moat.baby(nearestEnemy != null ? nearestEnemy.location : null);
        Debug.mark(9);
        if (Cfg.SEALED > 0) Seal.work();
        sense2();
        hpEnd = rc.getHealth();
    }

    static int threatRound = -1000, glances;

    /** Cfg.CAT_BLIND: our health at the end of our last turn (0 before the first). */
    static int hpEnd;

    /** Cfg.CAT_BLIND_DMG: a health loss that may be a cat's (step, or twice it; any loss when step is 0). */
    static boolean catHit(int loss, int step) { return step == 0 ? loss > 0 : loss == step || loss == 2 * step; }

    /** Cfg.CAT_BLIND: health fell below last (the end of our last turn) and no cat or enemy robot is in view. */
    static boolean blindHurt(int hp, int last, RobotInfo[] robots, Team them) {
        if (hp >= last) return false;
        for (RobotInfo r : robots) if (r.team == Team.NEUTRAL || r.team == them) return false;
        return true;
    }

    /** Whether a turn with no enemy known near should end facing back: an enemy was seen or heard within mem rounds. */
    static boolean glanceDue(int round, int threat, int mem) { return mem > 0 && round - threat <= mem; }

    /** Face the opposite way with the turn this turn left unused (the next step turns back at no movement cost). */
    static void glance() throws GameActionException {
        Direction back = G.rc.getDirection().opposite();
        if (back != Direction.CENTER && G.rc.canTurn(back)) { G.rc.turn(back); G.facing = back; glances++; }
    }

    /** Cfg.RISK_ADAPT: health at the start of our turn; the round an enemy trap fired on us or we heard of one (0 = never);
     *  our last SQ_BURN; whether this turn's fresh mask is on. */
    static int hpStart, burnRound, burnSqueaked;
    static boolean adaptRisk;

    /** Cfg.RISK_ADAPT: enemy rat traps are known to be in play (our own burn or a heard one, or the kings' shared bit). */
    static boolean trapsKnown(int burn, int sym) { return burn > 0 || (sym & Comms.SYM_BURN) != 0; }

    /** Cfg.RUSH_BURN: a rusher at me farther than Chebyshev r from the enemy start (our start home mirrored by the
     *  first live symmetry) stands down once enemy traps are known. */
    static boolean rushStands(MapLocation me, MapLocation home, int r) {
        return home != null && G.cheb(me, G.mirror(home, firstSym())) > r;
    }

    /** Cfg.WOUNDED_HP: turns held to the wound mask. */
    static int wounded;

    /** Cfg.WOUNDED_HP: Nav.avoid bits of the steps from me that end within Chebyshev 2 of a visible enemy baby. */
    static int woundMask(MapLocation me, RobotInfo[] robots) {
        int m = 0;
        for (RobotInfo r : robots) {
            if (r.team != G.them || r.type != UnitType.BABY_RAT) continue;
            int ex = r.location.x - me.x, ey = r.location.y - me.y;
            if (ex > 3 || ex < -3 || ey > 3 || ey < -3) continue;
            for (int i = 0; i < 8; i++) {
                int dx = ex - G.DIRS[i].dx, dy = ey - G.DIRS[i].dy;
                if (dx <= 2 && dx >= -2 && dy <= 2 && dy >= -2) m |= 1 << i;
            }
        }
        return m;
    }

    /** Whether this rat bites and grabs enemy rats now: after cooperation, or a RUSH_BREAK attacker at any time. */
    static boolean hostile() {
        if (Cfg.PEACE && G.coop) return false;
        return !G.coop || Cfg.RUSH_BREAK && role == ATTACK || Cfg.BREAK_ROUND > 0 && G.round >= Cfg.BREAK_ROUND;
    }

    /** End the turn facing the nearest enemy rat close enough to grab us next round. */
    static void guardTurn() throws GameActionException {
        RobotController rc = G.rc;
        if (G.coop || Cfg.GUARD_DIST2 == 0 || !rc.isTurningReady() || rc.getCarrying() != null) return;
        RobotInfo best = null;
        int bd = Integer.MAX_VALUE;
        for (RobotInfo r : rc.senseNearbyRobots(Cfg.GUARD_DIST2, G.them)) {
            if (r.team != G.them || r.type != UnitType.BABY_RAT) continue;
            int d = G.me.distanceSquaredTo(r.location);
            if (d < bd) { bd = d; best = r; }
        }
        MapLocation t = best != null ? best.location : null;
        if (best != null) D.face(D.F_SEEN);
        if (t == null && Cfg.GUARD_MEMORY) {
            // nothing in view: an enemy we saw at the start of the turn, or one an ally squeaked, may be behind us
            for (RobotInfo r : seen) {
                if (r.team != G.them || r.type != UnitType.BABY_RAT) continue;
                int d = G.me.distanceSquaredTo(r.location);
                if (d <= Cfg.GUARD_DIST2 && d < bd) { bd = d; t = r.location; D.face(D.F_MEMORY); }
            }
            for (int i = 0; i < TrapRisk.nHeard; i++) {
                int d = G.me.distanceSquaredTo(TrapRisk.heardE[i]);
                if (d <= 8 && d < bd) { bd = d; t = TrapRisk.heardE[i]; D.face(D.F_HEARD); }
            }
        }
        if (t == null) { t = Nav.guard; if (t != null) D.face(D.F_GUARD); }
        if (TrapRisk.nHeard > 0) threatRound = G.round;
        if (t == null) { if (glanceDue(G.round, threatRound, Cfg.GLANCE_MEM)) glance(); return; }
        Direction to = G.me.directionTo(t);
        if (to != Direction.CENTER && rc.getDirection() != to && rc.canTurn(to)) { rc.turn(to); G.facing = to; }
    }

    /** Cfg.RUSH_LIFE: an attacker born at birth that last saw the enemy king at seen has rushed for life rounds in vain. */
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

    static boolean rushFailed(int round, int birth, int seen, int life) {
        return round - birth > life && round - seen > life;
    }

    static void born() {
        RobotController rc = G.rc;
        birthRound = G.round;
        if (Cfg.TRIP_HALF != 0) tripFrom = G.round;
        int cost = rc.getCurrentRatCost();
        int idx = (cost - 10) / 10 * 4; // approximate spawn index from the cost step
        role = GATHER;
        if (idx >= Cfg.GATHER_FIRST && (G.round % Cfg.FIGHT_EVERY) == 0) role = FIGHT;
        raider = role == FIGHT && Cfg.RAID_EVERY > 0 && (G.round / Cfg.FIGHT_EVERY) % Cfg.RAID_EVERY == 0;
        int base = role;
        if (rusher(idx, G.round)) role = ATTACK;
        Comms.readKings();
        home = Comms.nearestKing(G.me);
        if (home == null) home = G.me;
        if (Cfg.RUSH_DIST > 0 && role == ATTACK && !rushNear(home, liveSyms(Comms.read(Comms.SYM)), Cfg.RUSH_DIST)) role = base;
        trapOk = trapsAllowed(home, Comms.read(Comms.SYM), Cfg.TRAP_DIST);
        if (Cfg.HOME_START || Cfg.MINE_RR || Cfg.FORM_MINE) origin = heardStart(rc.readSqueaks(-1));
        int px = 2 * home.x - G.cx2, py = 2 * home.y - G.cy2;
        fu = px >= 0 ? 1 : -1;
        fv = py >= 0 ? 1 : -1;
        Flow.frame(fu, fv);
        G.frame(home);
        // mirror-invariant seed: round and the unsigned offset from the king
        int seed = birthSeed(G.round, G.me, home);
        G.seed(seed);
        slot = ((Cfg.SYM_SLOT ? seed : rc.getID()) * 0x9E3779B1) >>> 8;
        scout = Cfg.SCOUT_EVERY > 0 && (slot >>> 4) % Cfg.SCOUT_EVERY == 0;
    }

    /** Cfg.RUSH: a baby born with idx live babies in round r attacks the enemy king. */
    /** The role born() gives a rat with idx live babies in round r. */
    static int roleFor(int idx, int r) {
        if (rusher(idx, r)) return ATTACK;
        return idx >= Cfg.GATHER_FIRST && (r % Cfg.FIGHT_EVERY) == 0 ? FIGHT : GATHER;
    }

    /** Cfg.RUSH_DIST: the symmetries the kings' SYM slot still allows (our own mask when the slot is empty). */
    /** Rat traps allowed for this baby (Cfg.TRAP_DIST), fixed at birth. */
    static boolean trapOk = true;

    /** Cfg.TRAP_DIST's test: true when d is 0 or no live symmetry (the kings' slot) mirrors home within Chebyshev d. */
    static boolean trapsAllowed(MapLocation home, int slot, int d) { return d == 0 || !rushNear(home, liveSyms(slot), d); }

    static int liveSyms(int slot) {
        int m = slot & Sym.mask;
        return m == 0 ? Sym.mask : m;
    }

    /** Cfg.RUSH_DIST: some symmetry in mask puts the enemy start (home mirrored) within Chebyshev d of home. */
    static boolean rushNear(MapLocation home, int mask, int d) {
        for (int s = Sym.ROT; s <= Sym.VER; s <<= 1)
            if ((mask & s) != 0 && G.cheb(home, G.mirror(home, s)) <= d) return true;
        return false;
    }

    static boolean rusher(int idx, int r) {
        return Cfg.RUSH && idx >= Cfg.RUSH_FROM && idx < Cfg.RUSH_TO && (Cfg.RUSH_UNTIL == 0 || r < Cfg.RUSH_UNTIL);
    }

    /** A rat's birth seed: its round and unsigned offset from the king that spawned it, the same for its mirror twin. */
    static int birthSeed(int round, MapLocation me, MapLocation home) {
        return round * 977 + Math.abs(me.x - home.x) * 31 + Math.abs(me.y - home.y) * 7 + 1;
    }

    /** Cfg.WAR_STRIKE k: whether a rat of this role born in this round is one of the 1 in k fighters that strike at
     *  the enemy king once at war (k 0: none). */
    static boolean striker(int role, int birth, int k) {
        return k > 0 && role == FIGHT && (birth / Cfg.FIGHT_EVERY) % k == 0;
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
    /** Cfg.SWAP_YIELD: the directions whose tile is next to an allied baby carrying an ally (its drop tile). */
    static int carrierMask(RobotInfo[] robots) {
        int mask = 0;
        for (RobotInfo r : robots) {
            if (r.team != G.us || r.carryingRobot == null || r.carryingRobot.team != G.us) continue;
            for (int i = 0; i < 8; i++) if (G.cheb(G.me.add(G.DIRS[i]), r.location) <= 1) mask |= 1 << i;
        }
        return mask;
    }

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

    /** Cfg.CHEESE_LOS: the piles seen this sense, judged once every wall in view is recorded. */
    static final MapLocation[] piles = new MapLocation[24];

    static void sense() throws GameActionException {
        MapInfo[] infos = G.rc.senseNearbyMapInfos();
        MapLocation bestCheese = null;
        int bd = Integer.MAX_VALUE, np = 0;
        boolean los = Cfg.CHEESE_LOS || breakingOut() || Cfg.SCOUT_FRONTIER && scouting();
        for (MapInfo mi : infos) {
            MapLocation l = mi.getMapLocation();
            boolean mine = mi.hasCheeseMine();
            if (Sym.record(l.x, l.y, mi.isWall(), mi.isDirt(), mine)) { }
            if (mine) noteMine(l.x, l.y);
            if (mi.getCheeseAmount() > 0) {
                if (los) { if (np < piles.length) piles[np++] = l; continue; }
                int d = G.me.distanceSquaredTo(l);
                if (d < bd && (Cfg.CHEESE_BAN == 0 || !bannedCheese(l, Cfg.CHEESE_BAN))
                        && (Cfg.SEALED == 0 || (Sym.mem[l.x + l.y * G.W] & Sym.SEAL) == 0)
                        && (!Cfg.SEAL_SEEN || !Seal.skipped(l.x + l.y * G.W))) { bd = d; bestCheese = l; }
            } else if (cheeseTarget != null && cheeseTarget.equals(l)) {
                cheeseTarget = null;
            }
        }
        if (los) {
            for (int i = 0; i < np; i++) {
                MapLocation l = piles[i];
                int d = G.me.distanceSquaredTo(l);
                if (d < bd && (Cfg.CHEESE_BAN == 0 || !bannedCheese(l, Cfg.CHEESE_BAN)) && !walledOff(G.me, l)) { bd = d; bestCheese = l; }
            }
            if (cheeseTarget != null && walledOff(G.me, cheeseTarget)) cheeseTarget = null;
        }
        // SEALED: drop a pile judged sealed; name one not yet judged for the end-of-turn fill (Seal.work)
        if (Cfg.SEALED > 0) {
            if (bestCheese != null) {
                int v = Seal.verdict(bestCheese.x + bestCheese.y * G.W);
                if (v > 0) bestCheese = null; else if (v < 0) Seal.want(bestCheese);
            }
            if (cheeseTarget != null) {
                int v = Seal.verdict(cheeseTarget.x + cheeseTarget.y * G.W);
                if (v > 0) cheeseTarget = null; else if (v < 0) Seal.want(cheeseTarget);
            }
        }
        if (bestCheese != null) {
            if (cheeseTarget == null || G.me.distanceSquaredTo(cheeseTarget) > bd) cheeseTarget = bestCheese;
        }
    }

    /** The greedy line from a to b (directionTo steps, as a straight walk would take) crosses a known wall. */
    static boolean walledOff(MapLocation a, MapLocation b) {
        int[] mem = Sym.mem;
        MapLocation c = a;
        while (true) {
            c = c.add(c.directionTo(b));
            if (c.equals(b)) return false;
            if ((mem[G.idx(c)] & Sym.WALL) != 0) return true;
        }
    }

    /** Give up on cheese at l for Cfg.CHEESE_BAN rounds, its Chebyshev-2 neighbourhood with it. */
    static void banCheese(MapLocation l) {
        banned[banHead] = l;
        bannedAt[banHead] = G.round;
        banHead = (banHead + 1) & 3;
    }

    /** Cheese at l lies within Chebyshev 2 of a pile banned less than ttl rounds ago. */
    static boolean bannedCheese(MapLocation l, int ttl) {
        for (int i = 0; i < 4; i++) {
            MapLocation b = banned[i];
            if (b != null && G.round - bannedAt[i] < ttl && G.cheb(b, l) <= 2) return true;
        }
        return false;
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
        if (canTake(c)) { rc.pickUpCheese(c); picked++; }
        Direction f = rc.getDirection();
        Direction[] fr = G.fan(c, f);
        for (Direction d : fr) {
            MapLocation l = c.add(d);
            if (canTake(l)) { rc.pickUpCheese(l); picked++; }
        }
        G.rawCheese = rc.getRawCheese();
    }

    /** A pile we may take: any when empty-handed, else only one that keeps us within CARRY_MAX (each unit carried
     *  slows moves and actions 1%: on fullofsecrets rats hauled 400-700 home at a fifth of their speed). */
    static boolean canTake(MapLocation l) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.canPickUpCheese(l)) return false;
        if (Cfg.ATK_CARRY > 0 && role == ATTACK) return atkTake(rc.getRawCheese(), rc.senseMapInfo(l).getCheeseAmount(), Cfg.ATK_CARRY);
        if (Cfg.CARRY_MAX <= 0) return true;
        int raw = rc.getRawCheese();
        return raw == 0 || takeOk(raw, rc.senseMapInfo(l).getCheeseAmount(), Cfg.CARRY_MAX);
    }

    static boolean takeOk(int raw, int pile, int cap) { return cap <= 0 || raw == 0 || raw + pile <= cap; }

    /** Cfg.ATK_CARRY: an attacker's take, capped even when empty-handed. */
    static boolean atkTake(int raw, int pile, int cap) { return raw + pile <= cap; }

    static RobotInfo eKing, nearestEnemy;
    static RobotInfo[] seen = new RobotInfo[0];
    public static int naps, throwsDone, ratTraps, mineTraps;
    static int carryRound = -100;

    /** Grab an adjacent enemy baby rat (the engine allows it when it cannot see us or is weaker): the healthiest,
     *  or under Cfg.RESCUE first one carrying one of ours (it drops it). */
    static boolean tryCarry(RobotInfo[] robots) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady() || rc.getCarrying() != null) return false;
        RobotInfo best = null;
        int bs = Integer.MIN_VALUE;
        for (RobotInfo r : robots) {
            if (r.team != G.them || r.type != UnitType.BABY_RAT) continue;
            if (G.me.distanceSquaredTo(r.location) > 2) continue;
            if (!rc.canCarryRat(r.location)) continue;
            int s = r.health + ((Cfg.RESCUE || Cfg.GUARD_CARRY_R > 0) && holdsOurs(r) ? 1000 : 0);
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
            if (!stay && Cfg.RISK_FRESH > 0 && (Nav.avoid & (1 << m.ordinal())) != 0) continue;
            if (Cfg.RISK_ADAPT > 0 && adaptRisk && !stay && (Nav.avoid & (1 << m.ordinal())) != 0) continue;
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

    /** Set down the ally carried past (Nav.swapPast) once we can act: on the tile it was taken from when that is free
     *  and beside us, else on the free tile farthest from our errand. */
    static void dropAlly() throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady()) return;
        MapLocation me = rc.getLocation(), t = Nav.target, f = Nav.swapFrom;
        Direction best = null;
        int bs = -1;
        for (Direction d : G.DIRS) {
            if (!rc.canDropRat(d)) continue;
            MapLocation l = me.add(d);
            int s = (l.equals(f) ? 1 << 24 : 0) + 2 * (t != null ? l.distanceSquaredTo(t) : 0) + (G.hand(me, d) > 0 ? 1 : 0);
            if (s > bs) { bs = s; best = d; }
        }
        if (best != null) { rc.dropRat(best); D.act(D.A_DROP_ALLY); Nav.swapFrom = null; }
    }

    static boolean tryThrow(RobotInfo[] robots) throws GameActionException {
        if (Cfg.THROW_CHECK) return throwChecked(robots);
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

    static RobotInfo ownKing;
    public static int throwAborts;

    /** Cfg.THROW_CHECK: tryThrow, with our unseen kings (the king channel) on the flights, a hit on our king scored
     *  three times a rat's, no throw whose best flight hits one of ours until THROW_LATE rounds into the carry, and
     *  the flight re-checked once the turn brings it into view. Field games (3934-3937, 3986-3992) threw into our own
     *  robots 0-17 times a game, up to 370 damage and 4 hits on our own king. */
    static boolean throwChecked(RobotInfo[] robots) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady()) return false;
        MapLocation me = rc.getLocation();
        if (robots == null) robots = rc.senseNearbyRobots(-1, null);
        int stamp = gridFill(me, robots);
        // our kings we cannot see stand where the king channel last put them
        for (int q = 0; q < Comms.numKings; q++) {
            MapLocation k = Comms.kings[q];
            int ox = k.x - me.x, oy = k.y - me.y;
            if (ox < -9 || ox > 9 || oy < -9 || oy > 9) continue;
            if (ownKing == null) ownKing = new RobotInfo(0, G.us, UnitType.RAT_KING, 600, k, Direction.NORTH, 0, 0, null);
            for (int ex = -1; ex <= 1; ex++) for (int ey = -1; ey <= 1; ey++) {
                int px = ox + ex, py = oy + ey;
                if (px < -8 || px > 8 || py < -8 || py > 8 || gridAt[(px + 8) + 17 * (py + 8)] == stamp) continue;
                gridPut(px, py, ownKing, stamp);
            }
        }
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
                    else if (r.team == G.us) s2 = r.type == UnitType.RAT_KING ? -3 * dmg : -dmg;
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
        // every flight hits one of ours: hold while the carry lasts (at its end the victim wriggles free)
        boolean late = G.round - carryRound >= Cfg.THROW_LATE;
        if (bs < 0 && !late) return false;
        if (rc.getDirection() != bestD) {
            if (!rc.canTurn(bestD)) return false;
            rc.turn(bestD);
            G.facing = bestD;
            D.turnOnly(D.A_THROW);
            // the flight is in view now: one of ours on it, unseen before the turn, stops the throw
            if (!late) {
                MapLocation l = me.add(bestD);
                for (int k = 2; k <= 8; k++) {
                    l = l.add(bestD);
                    if (!rc.onTheMap(l) || !rc.canSenseLocation(l) || (mem[l.x + w * l.y] & blk) != 0) break;
                    RobotInfo r = rc.senseRobotAtLocation(l);
                    if (r == null) continue;
                    if (r.team == G.us) { throwAborts++; return false; }
                    break;
                }
            }
        }
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
        if (rc.getAllCheese() < Econ.floor(Cfg.RAT_TRAP_MIN_CHEESE) || King.trapStarvesSpawn(rc)) return false;
        if (Cfg.TRAP_OPEN_TO > 0 && openTrapBlocked(rc.getRoundNum(), rc.getAllCheese(), Cfg.TRAP_OPEN_TO, Econ.floor(Cfg.TRAP_OPEN_FLOOR))) return false;
        int d0 = G.me.distanceSquaredTo(nearestEnemy.location);
        if (Cfg.TRAP_ADJ ? d0 > Cfg.BABY_TRAP_DIST2 : d0 <= 2 || d0 > Cfg.BABY_TRAP_DIST2) return false;
        MapLocation best = trapTile(rc.getLocation(), rc.getDirection(), nearestEnemy.location, d0, Cfg.TRAP_ADJ);
        if (best == null) return false;
        rc.placeRatTrap(best);
        D.act(D.A_RAT_TRAP);
        ratTraps++;
        return true;
    }

    /** Cfg.TRAP_OPEN_TO: before round `to` a rat trap needs `floor` banked (bodies first in the opening). */
    static boolean openTrapBlocked(int round, int bank, int to, int floor) { return round < to && bank < floor; }

    /** The front tile for a rat trap against an enemy at e (d0 = our dist2 to it): the one nearest e if nearer than we
     *  are, or with adj (Cfg.TRAP_ADJ) one within e's 3x3, nearest first (ties by pick bit, SYM_TIE); null if none. */
    static MapLocation trapTile(MapLocation me, Direction f, MapLocation e, int d0, boolean adj) throws GameActionException {
        RobotController rc = G.rc;
        MapLocation best = null;
        int bs = Integer.MAX_VALUE;
        for (Direction d : G.fan(me, f)) {
            MapLocation l = me.add(d);
            if (!rc.canPlaceRatTrap(l)) continue;
            int s2 = l.distanceSquaredTo(e);
            if (adj) {
                if (s2 > 2) continue;
                s2 = s2 * 2 + (G.pickBit(me, l) ? 0 : 1);
                if (s2 < bs || Cfg.SYM_TIE && s2 == bs && G.tieFirst(me, l, best)) { bs = s2; best = l; }
            } else if (s2 < bs) { bs = s2; best = l; }
        }
        return adj || bs < d0 ? best : null;
    }

    /** The live rat traps past which no combat trap is laid (TRAP_CAP, plus MINE_TRAP_CAP with trapped mines). */
    static int trapCap() { return Cfg.MINE_TRAPS > 0 || Cfg.RAID_TRAPS > 0 ? Math.min(25, Cfg.TRAP_CAP + Cfg.MINE_TRAP_CAP) : Cfg.TRAP_CAP; }

    /** Trap sites around a mine in our side frame (offsets times fu, fv); the order breaks distance ties. */
    static final int[][] MINE_SITES = {{-3, -3}, {-3, 0}, {0, -3}, {-3, 3}, {3, -3}, {0, 3}, {3, 0}, {3, 3}};

    /** l's rank among mine m's eight trap sites by distance to ek (ties in MINE_SITES order), or 99 if l is no site. */
    static int mineSiteRank(MapLocation m, MapLocation l, MapLocation ek) {
        int ox = l.x - m.x, oy = l.y - m.y;
        if (ox % 3 != 0 || oy % 3 != 0 || ox < -3 || ox > 3 || oy < -3 || oy > 3 || ox == 0 && oy == 0) return 99;
        int dl = l.distanceSquaredTo(ek), rank = 0;
        boolean seen = false;
        for (int[] s : MINE_SITES) {
            int sx = s[0] * fu, sy = s[1] * fv;
            if (sx == ox && sy == oy) { seen = true; continue; }
            int d = m.translate(sx, sy).distanceSquaredTo(ek);
            if (d < dl || d == dl && !seen) rank++;
        }
        return rank;
    }

    /** A rat trap on a front tile that is one of mine m's first `sites` trap sites (MINE_TRAPS, RAID_TRAPS). */
    static boolean mineTrap(MapLocation m, int sites) throws GameActionException {
        RobotController rc = G.rc;
        if (G.coop || !rc.isActionReady() || rc.getNumberRatTraps() >= Cfg.MINE_TRAP_CAP) return false;
        if (rc.getAllCheese() < Econ.floor(Cfg.MINE_TRAP_CHEESE) || King.trapStarvesSpawn(rc)) return false;
        MapLocation ek = G.mirror(ref(), firstSym()), me = rc.getLocation();
        for (Direction d : G.fan(me, rc.getDirection())) {
            MapLocation l = me.add(d);
            if (mineSiteRank(m, l, ek) < sites && rc.canPlaceRatTrap(l)) {
                rc.placeRatTrap(l);
                ratTraps++; mineTraps++;
                return true;
            }
        }
        return false;
    }

    /** Bite the best adjacent enemy: a king first, then the lowest health (BITE_CARRIER: a rat carrying one of ours
     *  first among rats; BITE_GUARD: a rat our bite leaves no healthier than us, and so unable to grab us, right after
     *  the kills); turn to face it when needed. */
    static boolean biteEnemies(RobotInfo[] robots) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady()) return false;
        MapLocation me = rc.getLocation();
        int myHp = rc.getHealth();
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
            int s = r.type == UnitType.RAT_KING ? r.health : biteScore(r.health, r.cheeseAmount, Cfg.CARRY_BITE);
            if (Cfg.BITE_GUARD && r.type == UnitType.BABY_RAT && guardBite(r.health, myHp)) s = 1011;
            // a rat carrying one of ours throws it next (often into our king): bite it before any other rat
            if (Cfg.BITE_CARRIER && r.carryingRobot != null && r.carryingRobot.team == G.us) s = 1000;
            // a rat in reach of our king bites it next: after an enemy king, before any other rat
            if (guardKing != null && r.type == UnitType.BABY_RAT && G.cheb(r.location, guardKing) <= 2) s = 650 + r.health;
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

    /** A baby rat's bite rank (lower first): 1000 + hp, less its loot up to cap (Cfg.CARRY_BITE), never under 1001. */
    static int biteScore(int hp, int cheese, int cap) {
        int loot = Math.min(cheese, cap);
        return loot > 0 ? Math.max(1001, 1000 + hp - loot) : 1000 + hp;
    }

    /** An enemy rat with hp can grab us (strictly healthier) and one bite (10) takes that away; not a kill. */
    static boolean guardBite(int hp, int myHp) { return hp > myHp && hp <= myHp + 10 && hp > 10; }

    /** Cheese that turns a bite on a target with hp into a kill (k*k for +k, k <= FINISH_K), else 0. */
    static int finishCheese(int hp, int global) {
        if (hp <= 10 || global < Cfg.FINISH_FLOOR) return 0;
        int k = hp - 10;
        return k <= Cfg.FINISH_K ? k * k : 0;
    }

    static boolean tryBite(RobotInfo r) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady()) return false;
        MapLocation l = r.location;
        if (G.me.distanceSquaredTo(l) > 2) return false;
        if (!rc.canAttack(l)) {
            Direction d = G.me.directionTo(l);
            if (rc.getDirection() != d && rc.canTurn(d)) rc.turn(d);
        }
        if (rc.canAttack(l)) { rc.attack(l); bitesRat++; return true; }
        return false;
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
            if (s < bs || Cfg.SYM_TIE && s == bs && G.tieFirst(me, l, best)) { bs = s; best = l; }
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
        if (!defend && !(Cfg.CAT_RING && G.coop) && Comms.read(Comms.BUDGET) * 4 < Cfg.TRAP_SURPLUS) return;
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
            if (Cfg.CAT_RING) s = Math.min(s, chebToBody(c.location, l));
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
                        if (Cfg.TRIP_HALF != 0) tripFrom = G.round;
                        return;
                    }
                }
        }
    }

    /** Squeak one fact: to a king in hearing range a mine it may not know or a sharper symmetry mask; at war, an enemy's front. */
    static void report() {
        MapLocation k = Comms.numKings == 0 ? null : Comms.nearestKing(G.me);
        boolean near = k != null && (Cfg.SYM_HEAR_TILE ? kingInEarshot(G.me) : G.me.distanceSquaredTo(k) <= 16);
        // RISK_ADAPT: news of an enemy trap goes rat to rat and to the kings until the shared bit is set
        if (Cfg.BURN && burnRound > 0 && G.round - burnSqueaked >= Cfg.BURN_EVERY
                && (Comms.read(Comms.SYM) & Comms.SYM_BURN) == 0) {
            G.rc.squeak(Comms.squeak(Comms.SQ_BURN, G.me.x, G.me.y, 0));
            burnSqueaked = G.round;
            D.lab2 = D.C_BURN;
            return;
        }
        if (Cfg.GOSSIP && Comms.numMines == 0) {
            // no mine known to the team yet: pass mine news rat to rat, which outruns a courier walking home
            for (Message m : G.rc.readSqueaks(-1)) {
                int c = m.getBytes();
                if (Comms.sqType(c) == Comms.SQ_MINE) noteMine(Comms.sqX(c), Comms.sqY(c));
                else if (Cfg.SYM_GOSSIP && Comms.sqType(c) == Comms.SQ_SYM) Sym.merge(Comms.sqExtra(c) & 7);
            }
            if (symGossip(Cfg.SYM_GOSSIP, near)) return;
            if (myMineCount > 0 && !near) {
                int key = myMines[G.round % myMineCount];
                D.lab2 = D.C_MINE_GOSSIP;
                G.rc.squeak(Comms.squeak(Comms.SQ_MINE, key / 64, key % 64, 0));
                return;
            }
        } else if (Cfg.GOSSIP_ALWAYS) {
            for (Message m : G.rc.readSqueaks(G.round - 1)) {
                int c = m.getBytes();
                if (Comms.sqType(c) == Comms.SQ_MINE) noteMine(Comms.sqX(c), Comms.sqY(c));
                else if (Cfg.SYM_GOSSIP && Comms.sqType(c) == Comms.SQ_SYM) Sym.merge(Comms.sqExtra(c) & 7);
            }
            if (symGossip(Cfg.SYM_GOSSIP, near)) return;
            if (!near) {
                int key = unreported();
                if (key >= 0) { D.lab2 = D.C_MINE_RELAY; G.rc.squeak(Comms.squeak(Comms.SQ_MINE, key / 64, key % 64, 0)); return; }
            }
        }
        // a squeak reaches robots with any tile within dist2 16 of us: the king's centre within 16 is always heard
        if (near) {
            int sharedSym = Cfg.BURN ? Comms.read(Comms.SYM) & 7 : Comms.read(Comms.SYM);
            if (sharedSym != 0 && (Sym.mask & sharedSym) != sharedSym) { D.lab2 = D.C_SYM_KING;
                G.rc.squeak(Comms.squeak(Comms.SQ_SYM, 0, 0, Sym.mask));
                return;
            }
            for (int i = 0; i < myMineCount; i++) {
                int x = myMines[i] / 64, y = myMines[i] % 64;
                if (!Comms.hasMine(x, y)) { D.lab2 = D.C_MINE_KING; G.rc.squeak(Comms.squeak(Comms.SQ_MINE, x, y, 0)); return; }
            }
            if (Cfg.TRIP_HALF != 0 && tripT > 0) { D.lab2 = D.C_TRIP; G.rc.squeak(Comms.squeak(Comms.SQ_TRIP, tripKey / 64, tripKey % 64, Math.min(tripT, 0xFFFF))); tripT = 0; return; }
        }
        // at war: the enemy king first (seen, or news to relay once per sighting), then an enemy's front
        if (!G.coop && Cfg.KING_PRESS && lastEKing != null && G.round - lastEKingRound <= 2
                && (ekHops == 0 && eKing != null || ekHops < Cfg.PRESS_HOPS && ekRelayed != lastEKingRound)) {
            int hops = eKing != null ? 0 : ekHops + 1;
            G.rc.squeak(Comms.squeak(Comms.SQ_EKING, lastEKing.x, lastEKing.y, hops | (lastEKingRound & 1023) << 2));
            ekRelayed = lastEKingRound;
            D.lab2 = D.C_EKING;
            return;
        }
        if (!G.coop && Cfg.TRAP_SHARE && nearestEnemy != null && nearestEnemy.direction != null) {
            D.lab2 = D.C_TRAP;
            int d = nearestEnemy.direction.ordinal();
            G.rc.squeak(Comms.squeak(Comms.SQ_TRAP, nearestEnemy.location.x, nearestEnemy.location.y, d));
            return;
        }
        if (k != null && G.me.distanceSquaredTo(k) <= 16 && catSeen != null && G.round - catSeenRound <= 2) {
            D.lab2 = D.C_CAT;
            G.rc.squeak(Comms.squeak(Comms.SQ_CAT, catSeen.x, catSeen.y, 0));
        }
    }

    static int symSqueaked = -100;

    /** Some tile of one of our kings (3x3) is within squeak range (d2 16) of p. */
    static boolean kingInEarshot(MapLocation p) {
        for (int i = 0; i < Comms.numKings; i++) {
            MapLocation k = Comms.kings[i];
            int dx = Math.max(0, Math.abs(p.x - k.x) - 1), dy = Math.max(0, Math.abs(p.y - k.y) - 1);
            if (dx * dx + dy * dy <= 16) return true;
        }
        return false;
    }

    /** Some tile of a cat (2x2 from catBL) is within squeak range (d2 16) of p. */
    static boolean catInEarshot(MapLocation catBL, MapLocation p) {
        int dx = Math.max(0, Math.max(catBL.x - p.x, p.x - catBL.x - 1));
        int dy = Math.max(0, Math.max(catBL.y - p.y, p.y - catBL.y - 1));
        return dx * dx + dy * dy <= 16;
    }

    /** SYM_GOSSIP: away from the kings, squeak a mask sharper than the shared one every SYM_GOSSIP_EVERY rounds,
     *  when an ally can hear it, no enemy is in view at war, and no cat can hear it. */
    static boolean symGossip(boolean on, boolean near) {
        if (!on || near || G.round - symSqueaked < Cfg.SYM_GOSSIP_EVERY) return false;
        int shared = Cfg.BURN ? Comms.read(Comms.SYM) & 7 : Comms.read(Comms.SYM);
        if (shared == 0 || (Sym.mask & shared) == shared) return false;
        if (!G.coop && nearestEnemy != null || cat != null && catInEarshot(cat.location, G.me)) return false;
        boolean ally = false;
        for (RobotInfo r : seen) if (r.team == G.us && G.me.distanceSquaredTo(r.location) <= 16) { ally = true; break; }
        if (!ally) return false;
        G.rc.squeak(Comms.squeak(Comms.SQ_SYM, 0, 0, Sym.mask));
        symSqueaked = G.round;
        D.lab2 = D.C_SYM_GOSSIP;
        return true;
    }

    // ---------------------------------------------------------------- movement

    /** True this turn when nearestEnemy is a prey (a rat we could ratnap). */
    static boolean hunting;

    /** Which branch took the last movement decision (debug tag mv=). */
    static char why = '-';

    static void moveDecision() throws GameActionException {
        if (Cfg.GRAB_MICRO && GrabMicro.round == G.round) { D.lab = D.M_GRAB; G.debt += 6; return; }
        why = '-';
        RobotController rc = G.rc;
        int raw = rc.getRawCheese();
        boolean late = G.round > 1960;

        if (Cfg.CAT_DEFEND_R2 > 0 && cat != null && catReach() && catAtKing(cat.location) && approachCat(cat)) { why = 'g'; D.lab = D.M_CAT_DEFEND; G.debt += 6; return; }
        int cde = G.debt, cbe = G.bc();   // ledger (K 11: 9, iload endGo and ifeq)
        boolean endGo = Cfg.END_CAT > 0 && !G.coop && G.round >= Cfg.END_CAT;
        G.debt = cde + G.bc() - cbe + 11;
        if (endGo && endCat(raw)) { why = 'e'; D.lab = D.M_END_CAT; G.debt += 6; return; }
        // keep gatherers out of cat reach
        if (cat != null && role == GATHER && catDist4(cat.location, G.me) <= 4 * Cfg.GATHER_CAT_DIST) {
            // flee the cat's centre (its location is the 2x2 body's low corner, which mirrors to the far corner)
            if (Cfg.SYM_CENTRE ? Nav.flee2(2 * cat.location.x + 1, 2 * cat.location.y + 1) : Nav.flee(cat.location)) { why = 'c'; D.lab = D.M_FLEE_CAT; G.debt += 6; return; }
        }
        if (Cfg.YIELD_KING && cat != null && raw == 0 && yieldKing()) { why = 'y'; D.lab = D.M_YIELD_KING; G.debt += 6; return; }
        if (Cfg.CARRY_AIM && !G.coop && rc.getCarrying() != null && rc.getCarrying().team == G.them && aimCarry()) { why = 'T'; D.lab = D.M_AIM_CARRY; G.debt += 6; return; }
        if (rescueT != null && rescue()) { why = 'R'; D.lab = D.M_RESCUE; G.debt += 6; return; }
        Debug.mark(10);
        if (Cfg.AMMO_AWAY && !G.coop && ammoAway()) { why = 'A'; D.lab = D.M_AMMO_AWAY; G.debt += 6; return; }
        if (!G.coop && retreat()) { why = 'r'; D.lab = D.M_RETREAT; G.debt += 6; return; }
        // the bank is nearly empty and the king starves at 0: whatever we carry goes home before any fight or errand
        if (raw > 0 && G.globalCheese < Cfg.HUNGRY && deliver()) { why = 'H'; D.lab = D.M_HUNGRY; G.debt += 6; return; }
        if (Cfg.FORM_KING && formKing()) { why = 'K'; D.lab = D.M_FORM_KING; G.debt += 6; return; }
        int cd2 = G.debt, cb2 = G.bc();   // ledger (K = 20 with G.settle)
        boolean cashMv = Cfg.CASH_BANK > 0 && cashSite != null && cashMove();
        G.settle(cd2, cb2, 20);
        if (cashMv) { why = 'Q'; D.lab = D.M_CASH; G.debt += 6; return; }
        Debug.mark(11);
        if (!G.coop && raider && raw == 0 && rc.getHealth() >= Cfg.RAID_HP && homeHalf(G.me)
                && (nearestEnemy == null || G.me.distanceSquaredTo(nearestEnemy.location) > Cfg.RAID_ENGAGE) && raid()) { why = 'z'; D.lab = D.M_RAID; G.debt += 6; return; }
        Debug.mark(12);
        if (!G.coop && Cfg.FLANK && Cfg.NAP && flank(seen)) { why = 'k'; D.lab = D.M_FLANK; G.debt += 6; return; }
        if (!G.coop && Cfg.KING_PRESS && pressKing()) { why = 'p'; D.lab = D.M_PRESS; G.debt += 6; return; }
        Debug.mark(13);
        if (!G.coop && Cfg.MICRO && role != ATTACK && Micro.act(seen)) { why = 'm'; D.lab = D.M_MICRO; G.debt += 6; return; }
        Debug.mark(14);
        if (!G.coop && Cfg.FACE_ENEMY && (role != GATHER || Cfg.WAR_GATHER < 2 || nearEnemy(2)) && faceEnemy()) { why = 'f'; D.lab = D.M_FACE; G.debt += 6; return; }
        if (Cfg.ATK_DELIVER > 0 && atkHome(raw) && deliver()) { D.lab = D.M_ATK_DELIVER; G.debt += 6; return; }
        if (Cfg.STALL_CAT > 0 && role == ATTACK && G.coop && stallCat()) { why = 'g'; D.lab = D.M_STALL_CAT; G.debt += 6; return; }
        if (role == ATTACK && attack()) { why = 'a'; D.lab = D.M_ATTACK; G.debt += 6; return; }
        Debug.mark(15);
        if (!G.coop && war()) { why = 'w'; D.lab = D.M_WAR; G.debt += 6; return; }
        if (!G.coop && role == FIGHT && Cfg.FORT_R > 0 && fort()) { why = 'h'; D.lab = D.M_FORT; G.debt += 6; return; }
        Debug.mark(16);
        why = 'g';
        if (cat != null && role == FIGHT && catReach()) {
            if (approachCat(cat)) { D.lab = D.M_FIGHT_CAT; G.debt += 6; return; }
        }

        if (loadAt > 0) {
            // full loads: no trip home with a part load; with no pile in sight the gatherer stays at its mine
            if (fullLoad(raw, late, loadAt) && deliver()) { D.lab = D.M_DELIVER; G.debt += 6; return; }
        } else if ((raw >= Cfg.DELIVER_AT || (raw > 0 && (late || cheeseTarget == null && raw >= Cfg.DELIVER_IDLE))) && deliver()) { D.lab = D.M_DELIVER; G.debt += 6; return; }
        if (Cfg.COURIER && Comms.numMines == 0 && myMineCount > 0) {
            // we found a mine nobody else knows: squeaks only reach the king from dist2 16, so carry the news home
            MapLocation k = Comms.nearestKing(G.me);
            if (k != null) { Nav.goTo(k); why = 'm'; D.lab = D.M_COURIER; G.debt += 6; return; }
        }
        if (cheeseTarget != null) {
            if (G.cheb(G.me, cheeseTarget) == 1 && Nav.isDirt(cheeseTarget)) {
                // cheese on dirt: pick it up from here, facing it, instead of digging onto the tile
                Direction d = G.me.directionTo(cheeseTarget);
                if (rc.getDirection() != d && rc.canTurn(d)) { rc.turn(d); G.facing = d; }
                pickUp();
                if (!rc.canSenseLocation(cheeseTarget) || rc.senseMapInfo(cheeseTarget).getCheeseAmount() == 0) cheeseTarget = null;
                why = 'P';
                D.lab = D.M_CHEESE_PICK; G.debt += 6; return;
            }
            why = 'C';
            Nav.goTo(cheeseTarget);
            if (Nav.stall > 12) {
                if (Cfg.CHEESE_BAN > 0 && (!Cfg.CHEESE_BAN_WALL || Flow.detour(cheeseTarget))) banCheese(cheeseTarget);
                if (Cfg.BREAKOUT_TTL > 0 && walledOff(G.me, cheeseTarget)) wallStrike(G.round, Cfg.BREAKOUT_TTL);
                cheeseTarget = null;
            }
            D.lab = D.M_CHEESE; G.debt += 6; return;
        }
        if (role == FIGHT) {
            MapLocation ct = catReach() ? knownCat() : null;
            if (ct != null) { Nav.goTo(ct); catStalled(ct); D.lab = D.M_KNOWN_CAT; G.debt += 6; return; }
            if (Cfg.RAID && raider && rc.getHealth() >= Cfg.RAID_HP && raid()) { why = 'z'; D.lab = D.M_RAID_FIGHT; G.debt += 6; return; }
        }
        if (scouting()) {
            // a scout with news the king lacks carries it home (squeaks reach the king from dist2 16 only)
            MapLocation k = Comms.nearestKing(G.me);
            if (k != null && unreported() >= 0) {
                if (Cfg.SWEEP > 0) sweepCell = sweepScan = 0; // the cell's time budget restarts after the walk home
                Nav.goTo(k); why = 's'; D.lab = D.M_SCOUT_HOME; G.debt += 6; return;
            }
            why = 'S';
            D.lab = D.M_SCOUT; G.debt += 6; 
            if (Cfg.SCOUT_FRONTIER && exploreFrontier()) return;
            if (!exploreHome()) explore();
            return;
        }
        if (role == GATHER && G.round < seekUntil && seek()) { why = 'x'; D.lab = D.M_SEEK; G.debt += 6; return; }
        if (Cfg.MINE_EXPLORE > 0 && role == GATHER && G.round < Cfg.MINE_EXPLORE_UNTIL
                && explorer(slot, homeMineCount(), Cfg.MINE_EXPLORE) && exploreHome()) { mineTarget = null; why = 'x'; D.lab = D.M_MINE_EXPLORE; G.debt += 6; return; }
        if (role == GATHER || role == FIGHT && Cfg.FIGHT_MINE) {
            if (Cfg.MINE_BALANCE && mineTarget != null && (Cfg.HOME_ALL_SYM ? Comms.numMines != minesAtPick || Sym.mask != maskAtPick : Comms.numMines != minesAtPick)) mineTarget = null;
            if (Cfg.MINE_ANY_COOP && !G.coop && mineTarget != null && !homeHalf(mineTarget)) mineTarget = null;
            if (postMargin > 0 && role == FIGHT && coopAtPick && !G.coop) mineTarget = null;
            if (Cfg.KMINE_SKIP > 0 && mineTarget != null && G.round % 10 == 0 && kingHeld(mineTarget)
                    && homeMineCount() >= Cfg.KMINE_SKIP) mineTarget = null;
            if (Cfg.TRIP_HALF != 0 && mineTarget != null && Comms.tripSig != tripSigAtPick) mineTarget = null;
            if (mineTarget == null) { D.pickSite = 0; mineTarget = pickMine(); }
            if (mineTarget != null) {
                if (Cfg.SEEK_IDLE > 0 && (role == GATHER || Cfg.FIGHT_SHIFT && role == FIGHT) && idle(mineTarget)) {
                    if (Cfg.IDLE_SHIFT) { mineShift++; D.pickSite = 1; mineTarget = pickMine(); }
                    if (role == GATHER && idleSeek() && seek()) { why = 'x'; D.lab = D.M_IDLE_SEEK; G.debt += 6; return; }
                    if (mineTarget == null) { D.pickSite = 2; mineTarget = pickMine(); }
                }
                if (atMine(mineTarget, Cfg.MINE_KING_R)) {
                    if (Cfg.TRIP_HALF != 0) tripArrive(mineTarget, G.round);
                    if (role == GATHER && startSeek(mineTarget) && seek()) { why = 'x'; D.lab = D.M_START_SEEK; G.debt += 6; return; }
                    if (Cfg.GATHER_STILL) {
                        // hold still (moves set off traps laid at our mines) and sweep the cone around the spawn square
                        Direction t = G.facing.rotateRight().rotateRight();
                        if (rc.canTurn(t)) { rc.turn(t); G.facing = t; }
                        if (Cfg.MINE_TRAPS > 0) mineTrap(mineTarget, Cfg.MINE_TRAPS);
                        D.lab = D.M_MINE_STILL; G.debt += 6; return;
                    }
                    // wander inside the spawn square, facing outward to see new cheese
                    wanderAround(mineTarget);
                    if (Cfg.MINE_TRAPS > 0) mineTrap(mineTarget, Cfg.MINE_TRAPS);
                    why = 'W';
                    D.lab = D.M_MINE_WANDER; G.debt += 6; 
                } else {
                    why = 'M';
                    D.lab = D.M_MINE_GO; G.debt += 6; 
                    Nav.goTo(mineTarget);
                    if (Nav.stall > 25) mineTarget = null;
                }
                return;
            }
        }
        why = 'E';
        D.lab = D.M_EXPLORE; G.debt += 6; 
        explore();
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
        if (Cfg.TRAIL && homeLeg(k)) return true;
        Nav.goTo(Cfg.DELIVER_ANCHOR > 0 ? anchorFor(k, Cfg.DELIVER_ANCHOR) : k);
        return true;
    }

    static MapLocation anchor;
    /** Cfg.DELIVER_ANCHOR: the delivery target for king k, re-set to k once k is more than r from it. */
    static MapLocation anchorFor(MapLocation k, int r) {
        if (anchor == null || G.cheb(anchor, k) > r) anchor = k;
        return anchor;
    }

    static int homeBest, homeStall, homeRound = -10, homeHeld;
    /** Cfg.TRAIL_GAP (a field so tests can set it). */
    static int trailGap = Cfg.TRAIL_GAP;
    static boolean homing;

    /** Cfg.TRAIL: after FLOW_STALL turns of delivery without getting closer to the nearest king (a roaming king does
     *  not reset the count, unlike Nav's stall), walk the trail back until it ends; while it is held by robots, wait
     *  FLOW_HOLD turns, then let Nav route around them. True when this took the turn. */
    static boolean homeLeg(MapLocation k) throws GameActionException {
        if (trailGap > 0 ? G.round - homeRound > trailGap : homeRound != G.round - 1) { homing = false; homeBest = Integer.MAX_VALUE; homeStall = 0; homeHeld = 0; }
        homeRound = G.round;
        int c = G.cheb(G.me, k);
        if (c < homeBest) { homeBest = c; homeStall = 0; } else homeStall++;
        if (!homing && homeStall < Cfg.FLOW_STALL) return false;
        homing = true;
        if (!G.rc.isMovementReady()) return true;
        int r = Trail.back();
        if (r == Trail.MOVED) { homeHeld = 0; return true; }
        if (r == Trail.END) { homing = false; homeStall = 0; homeBest = c; return false; }
        return ++homeHeld <= Cfg.FLOW_HOLD;
    }

    static int noKing;               // symmetry candidates whose enemy-king spot we saw empty
    static MapLocation lastEKing;
    static int lastEKingRound = -100;
    /** Relay hops of lastEKing's news (0 = seen ourselves), and the sighting round we last relayed. */
    static int ekHops, ekRelayed = -100;
    public static int presses;

    /** An SQ_EKING squeak: extra = hops | (sighting round % 1024) << 2. A newer sighting (or the same one in fewer
     *  hops) replaces ours. */
    static void heardEKing(int c) {
        int e = Comms.sqExtra(c), s = (e >> 2) & 1023;
        int seenRound = G.round - (((G.round & 1023) - s + 1024) & 1023);
        int hops = e & 3;
        if (seenRound > lastEKingRound || seenRound == lastEKingRound && hops < ekHops) {
            lastEKing = new MapLocation(Comms.sqX(c), Comms.sqY(c));
            lastEKingRound = seenRound;
            ekHops = hops;
        }
    }

    /** King press movement (Cfg.KING_PRESS): walk to a fresh enemy-king sighting in range and stay against it. */
    static boolean pressKing() throws GameActionException {
        RobotController rc = G.rc;
        MapLocation ek = eKing != null ? eKing.location : G.round - lastEKingRound <= Cfg.PRESS_AGE ? lastEKing : null;
        if (ek == null && Cfg.PRESS_SHARED) ek = sharedEKing();
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

    /** The kings' shared enemy-king sighting (Comms.EKING) when at most PRESS_AGE rounds old, else null. */
    static MapLocation sharedEKing() {
        int x = Comms.read(Comms.EKING);
        if (x == 0 || !Comms.fresh(Comms.read(Comms.EKING + 2), Cfg.PRESS_AGE)) return null;
        return new MapLocation(x - 1, Comms.read(Comms.EKING + 1) - 1);
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
        if (Cfg.RUSH_STICK) return rushStick();
        if (Cfg.RUSH_BAN > 0) return rushBan(Cfg.RUSH_BAN);
        RobotController rc = G.rc;
        MapLocation best = null;
        int bd = Integer.MAX_VALUE;
        for (int s = 1; s <= 4; s <<= 1) {
            if ((Sym.mask & s) == 0 || (noKing & s) != 0) continue;
            MapLocation c = G.mirror(ref(), s);
            if (G.me.distanceSquaredTo(c) <= 13 && rc.canSenseLocation(c)) {
                RobotInfo r = rc.senseRobotAtLocation(c);
                if (r == null || r.type != UnitType.RAT_KING || r.team != G.them) { noKing |= s; continue; }
            }
            int d = G.me.distanceSquaredTo(c) * 2 + (G.pickBit(G.me, c) ? 1 : 0);
            if (d < bd || Cfg.SYM_TIE && d == bd && G.tieFirst(G.me, c, best)) { bd = d; best = c; }
        }
        if (best == null && noKing != 0) { noKing = 0; return null; }
        return best;
    }

    /** Cfg.RUSH_BAN: the round a flow search toward each symmetry's enemy-king spot last failed (index: symmetry bit
     *  1, 2, 4 as 0, 1, 2), allocated on first use. */
    static int[] rushFail;

    /** rushTarget with Cfg.RUSH_BAN: the nearest live mirror spot, passing over spots whose search failed within
     *  ttl (Cfg.RUSH_BAN) rounds while another is live. */
    static MapLocation rushBan(int ttl) throws GameActionException {
        RobotController rc = G.rc;
        int[] rf = rushFail;
        if (rf == null) { rf = rushFail = new int[3]; rf[0] = rf[1] = rf[2] = -100000; }
        MapLocation ft = Flow.failTgt, best = null;
        boolean fresh = ft != null && G.round - Flow.failRound <= 1;
        int bd = Integer.MAX_VALUE;
        for (int s = 1, i = 0; s <= 4; s <<= 1, i++) {
            if ((Sym.mask & s) == 0 || (noKing & s) != 0) continue;
            MapLocation c = G.mirror(ref(), s);
            if (G.me.distanceSquaredTo(c) <= 13 && rc.canSenseLocation(c)) {
                RobotInfo r = rc.senseRobotAtLocation(c);
                if (r == null || r.type != UnitType.RAT_KING || r.team != G.them) { noKing |= s; continue; }
            }
            if (fresh && G.cheb(ft, c) <= 1) rf[i] = Flow.failRound;
            int d = banKey(G.round - rf[i] < ttl, G.me.distanceSquaredTo(c), G.pickBit(G.me, c));
            if (d < bd || Cfg.SYM_TIE && d == bd && G.tieFirst(G.me, c, best)) { bd = d; best = c; }
        }
        if (best == null && noKing != 0) { noKing = 0; return null; }
        return best;
    }

    /** rushBan's order: spots not banned first, then by squared distance, then the mirror-invariant bit. */
    static int banKey(boolean banned, int d2, boolean bit) {
        return (banned ? 1 << 20 : 0) + d2 * 2 + (bit ? 1 : 0);
    }

    /** Cfg.RUSH_STICK: the symmetry whose enemy-king spot this attacker walks to (0 = none yet). */
    static int rushSym;

    /** rushTarget with Cfg.RUSH_STICK: the nearest live mirror spot is picked once and kept while it stays live. */
    static MapLocation rushStick() throws GameActionException {
        RobotController rc = G.rc;
        MapLocation best = null;
        int bd = Integer.MAX_VALUE, bs = 0, live = 0;
        for (int s = 1; s <= 4; s <<= 1) {
            if ((Sym.mask & s) == 0 || (noKing & s) != 0) continue;
            MapLocation c = G.mirror(ref(), s);
            if (G.me.distanceSquaredTo(c) <= 13 && rc.canSenseLocation(c)) {
                RobotInfo r = rc.senseRobotAtLocation(c);
                if (r == null || r.type != UnitType.RAT_KING || r.team != G.them) { noKing |= s; continue; }
            }
            live |= s;
            int d = G.me.distanceSquaredTo(c) * 2 + (G.pickBit(G.me, c) ? 1 : 0);
            if (d < bd || Cfg.SYM_TIE && d == bd && G.tieFirst(G.me, c, best)) { bd = d; best = c; bs = s; }
        }
        if (best == null && noKing != 0) { noKing = 0; rushSym = 0; return null; }
        rushSym = stickSym(live, rushSym, bs);
        return rushSym == bs ? best : G.mirror(ref(), rushSym);
    }

    /** The symmetry an attacker walks to: its current one while still live, else this turn's nearest. */
    static int stickSym(int live, int cur, int best) {
        return (live & cur) != 0 ? cur : best;
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
        if (Cfg.FACE_REACH && d > 2 && !reach1(G.me, e)) { faceSkips++; return false; }
        Direction to = G.me.directionTo(e);
        if (!rc.canSenseLocation(e) || (d > 2 && rc.getDirection() != to)) {
            if (rc.getDirection() != to && rc.canTurn(to)) { rc.turn(to); G.facing = to; }
            return true;
        }
        if (d > 2 && rc.canMove(to) && (Nav.avoid & (1 << to.ordinal())) == 0) {
            if (Cfg.FACE_HOLD && faceHold()) { faceHolds++; return true; }
            rc.move(to);
            G.me = rc.getLocation();
        }
        return true;
    }

    public static int faceHolds, faceSkips;

    /** Cfg.FACE_REACH: some tile next to both a and e (on the map, not a known wall or dirt) is open, so e can get
     *  next to a in one move. */
    static boolean reach1(MapLocation a, MapLocation e) {
        int[] mem = Sym.mem;
        int W = G.W, H = G.H, x1 = Math.min(a.x, e.x) + 1, y0 = Math.max(a.y, e.y) - 1, y1 = Math.min(a.y, e.y) + 1;
        for (int x = Math.max(a.x, e.x) - 1; x <= x1; x++) {
            if (x < 0 || x >= W) continue;
            for (int y = y0; y <= y1; y++) {
                if (y < 0 || y >= H || x == a.x && y == a.y || x == e.x && y == e.y) continue;
                if ((mem[x + y * W] & (Sym.WALL | Sym.DIRT)) == 0) return true;
            }
        }
        return false;
    }

    /** Cfg.FACE_HOLD: hold at Chebyshev 2 instead of stepping into contact with nearestEnemy (see Cfg). */
    static boolean faceHold() {
        if (hunting || nearestEnemy.health < G.rc.getHealth() || !sees(nearestEnemy, G.me)) return false;
        if (!trapsKnown(burnRound, Comms.read(Comms.SYM))) return false;
        MapLocation th = threatenedKing();
        return th == null || nearestEnemy.location.distanceSquaredTo(th) > Cfg.KING_THREAT_D2;
    }

    /** War movement: close on a visible enemy king or nearby enemy rat, or answer a king's alert. */
    static boolean war() throws GameActionException {
        int raw = G.rc.getRawCheese();
        MapLocation threatened = threatenedKing();
        if (threatened != null) defends++;
        if (role == GATHER && Cfg.WAR_GATHER > 0 && !hunting && threatened == null) {
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
        if (nearestEnemy != null && !(role == FIGHT && Cfg.FORT_R > 0 && !nearHome(nearestEnemy.location, Cfg.FORT_R + 3))) {
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
                if (Cfg.STANDOFF && !hunting && !atKing && sees(nearestEnemy, G.me) && nearestEnemy.health >= G.rc.getHealth()) {
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
        if (!Cfg.DEFEND || G.coop || !Comms.fresh(Comms.read(Comms.ALERT + 2), 2) || Comms.alertThreat() < 1) return null;
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

    /** Cfg.RESCUE: this turn's enemy carrier of one of ours within RESCUE_D2, or null. */
    static RobotInfo rescueT;
    public static int rescues;

    static boolean holdsOurs(RobotInfo r) { return r.carryingRobot != null && r.carryingRobot.team == G.us; }

    /** The nearest enemy baby within dist2 RESCUE_D2 carrying one of ours (ties: the healthier one held, then
     *  G.pickBit), or null; none in cooperation (a grab is a backstab). */
    static RobotInfo rescueTarget(RobotInfo[] robots) {
        if (G.coop) return null;
        RobotInfo best = null;
        int bd = Integer.MAX_VALUE;
        for (RobotInfo r : robots) {
            if (r.team != G.them || r.type != UnitType.BABY_RAT || !holdsOurs(r)) continue;
            int d = G.me.distanceSquaredTo(r.location);
            if (d > Cfg.RESCUE_D2) continue;
            int s = (d * 128 + 100 - Math.min(100, r.carryingRobot.health)) * 2 + (G.pickBit(G.me, r.location) ? 0 : 1);
            if (s < bd || Cfg.SYM_TIE && s == bd && G.tieFirst(G.me, r.location, best.location)) { bd = s; best = r; }
        }
        return best;
    }

    /** Cfg.GUARD_CARRY_R: rescueT came from kingCarrier this turn (rescue then walks toward a blind tile). */
    static boolean guardCarry;
    public static int carryGuards;

    /** Cfg.GUARD_CARRY_R: within Chebyshev reach of our nearest king, the enemy baby nearest us holding one of ours
     *  within Chebyshev rad of the king's centre (ties as rescueTarget), or null; none in cooperation. */
    static RobotInfo kingCarrier(RobotInfo[] robots, int rad, int reach) {
        if (G.coop) return null;
        MapLocation k = Comms.nearestKing(G.me);
        if (k == null || G.cheb(G.me, k) > reach) return null;
        RobotInfo best = null;
        int bd = Integer.MAX_VALUE;
        for (RobotInfo r : robots) {
            if (r.team != G.them || r.type != UnitType.BABY_RAT || !holdsOurs(r) || G.cheb(r.location, k) > rad) continue;
            int s = (G.me.distanceSquaredTo(r.location) * 128 + 100 - Math.min(100, r.carryingRobot.health)) * 2 + (G.pickBit(G.me, r.location) ? 0 : 1);
            if (s < bd || Cfg.SYM_TIE && s == bd && G.tieFirst(G.me, r.location, best.location)) { bd = s; best = r; }
        }
        return best;
    }

    /** Get behind rescueT: on a tile next to it that it cannot see, turn to it (the post-move carry grabs it); else
     *  step to such a tile (turning to it first, as flank does). False when neither is possible. */
    static boolean rescue() throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isActionReady()) return false;
        if (Cfg.GUARD_CARRY_HP && guardCarry && rescueT.health < rc.getHealth()) return rescueWeak();
        MapLocation c = rescueT.location;
        if (G.cheb(G.me, c) == 1 && !sees(rescueT, G.me)) {
            Direction f = G.me.directionTo(c);
            if (rc.getDirection() != f && rc.canTurn(f)) { rc.turn(f); G.facing = f; }
            rescues++;
            return true;
        }
        if (!rc.isMovementReady()) return false;
        Direction bestD = null;
        int bs = Integer.MAX_VALUE;
        for (int i = 0; i < 8; i++) {
            Direction d = G.DIRS[i];
            MapLocation t = G.me.add(d);
            if (G.cheb(t, c) != 1 || sees(rescueT, t)) continue;
            if ((Nav.avoid & (1 << i)) != 0 || !rc.canMove(d)) continue;
            int sc = (d == rc.getDirection() ? 0 : 1) * 2 + (G.hand(G.me, d) > 0 ? 1 : 0);
            if (sc < bs) { bs = sc; bestD = d; }
        }
        if (bestD == null) return Cfg.RESCUE_BLOCK && blockMuzzle(c) || Cfg.RESCUE_WALK && walkBehind(c)
                || Cfg.GUARD_CARRY_R > 0 && guardCarry && walkBehind(c);
        MapLocation g = Nav.guard;
        Nav.guard = c;
        boolean moved = Nav.step(bestD);
        Nav.guard = g;
        if (moved) rescues++;
        return moved;
    }

    /** Cfg.GUARD_CARRY_HP: rescueT has less health than ours, so any tile next to it will do for the grab: beside it,
     *  turn to it; else step next to it (tiles it cannot see first); else walk toward it. */
    static boolean rescueWeak() throws GameActionException {
        RobotController rc = G.rc;
        MapLocation c = rescueT.location;
        if (G.cheb(G.me, c) == 1) {
            Direction f = G.me.directionTo(c);
            if (rc.getDirection() != f && rc.canTurn(f)) { rc.turn(f); G.facing = f; }
            rescues++;
            return true;
        }
        if (!rc.isMovementReady()) return false;
        Direction bestD = null;
        int bs = Integer.MAX_VALUE;
        for (int i = 0; i < 8; i++) {
            Direction d = G.DIRS[i];
            MapLocation t = G.me.add(d);
            if (G.cheb(t, c) != 1) continue;
            if ((Nav.avoid & (1 << i)) != 0 || !rc.canMove(d)) continue;
            int sc = ((sees(rescueT, t) ? 1 : 0) * 2 + (d == rc.getDirection() ? 0 : 1)) * 2 + (G.hand(G.me, d) > 0 ? 1 : 0);
            if (sc < bs) { bs = sc; bestD = d; }
        }
        if (bestD == null) { Nav.goTo(c); rescues++; return true; }
        MapLocation g = Nav.guard;
        Nav.guard = c;
        boolean moved = Nav.step(bestD);
        Nav.guard = g;
        if (moved) rescues++;
        return moved;
    }

    /** Cfg.RESCUE_BLOCK: a throw needs the tile in front of the carrier empty; step into it, facing the carrier. */
    static boolean blockMuzzle(MapLocation c) throws GameActionException {
        MapLocation m = c.add(rescueT.direction);
        if (G.cheb(G.me, m) != 1) return false;
        Direction d = G.me.directionTo(m);
        if ((Nav.avoid & (1 << d.ordinal())) != 0 || !G.rc.canMove(d)) return false;
        MapLocation g = Nav.guard;
        Nav.guard = c;
        boolean moved = Nav.step(d);
        Nav.guard = g;
        if (moved) blocks++;
        return moved;
    }

    public static int blocks;

    /** Cfg.RESCUE_WALK: no blind tile next to rescueT is one step away; walk toward the nearest open one. */
    static boolean walkBehind(MapLocation c) throws GameActionException {
        MapLocation best = null;
        int bd = Integer.MAX_VALUE;
        for (int i = 0; i < 8; i++) {
            MapLocation t = c.add(G.DIRS[i]);
            if (sees(rescueT, t) || !G.rc.onTheMap(t) || G.rc.canSenseLocation(t) && !G.rc.sensePassability(t)) continue;
            int d = (G.me.distanceSquaredTo(t) * 2) + (G.pickBit(G.me, t) ? 0 : 1);
            if (d < bd || Cfg.SYM_TIE && d == bd && G.tieFirst(G.me, t, best)) { bd = d; best = t; }
        }
        if (best == null) return false;
        Nav.goTo(best);
        rescues++;
        return true;
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

    /** Our king's centre while this turn's nearestEnemy is a rat in reach of it (Cfg.GUARD_KING_R), else null. */
    static MapLocation guardKing;
    public static int guards;

    /** Cfg.GUARD_KING_R: when we are within Chebyshev reach of our nearest king, the visible enemy baby nearest us among
     *  those within Chebyshev 2 of its centre (they can bite it), else null. Ties: lower health, then G.pickBit. */
    static RobotInfo kingBiter(RobotInfo[] robots, int reach) {
        MapLocation k = Comms.nearestKing(G.me);
        if (k == null || G.cheb(G.me, k) > reach) return null;
        RobotInfo best = null;
        int bk = Integer.MAX_VALUE;
        for (RobotInfo r : robots) {
            if (r.team != G.them || r.type != UnitType.BABY_RAT || G.cheb(r.location, k) > 2) continue;
            int key = (G.me.distanceSquaredTo(r.location) * 128 + r.health) * 2 + (G.pickBit(G.me, r.location) ? 0 : 1);
            if (key < bk || Cfg.SYM_TIE && key == bk && G.tieFirst(G.me, r.location, best.location)) { bk = key; best = r; }
        }
        if (best != null) guardKing = k;
        return best;
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

    /** True when l is within Chebyshev r of our nearest king. */
    static boolean nearHome(MapLocation l, int r) {
        MapLocation k = Comms.nearestKing(l);
        return k != null && G.cheb(k, l) <= r;
    }

    /**
     * Fortress: at war, fighters hold a ring around the king (where its traps are) instead of meeting the enemy
     * in the open; war() engages anything that comes within FORT_R + 3 of the king.
     */
    static boolean fort() throws GameActionException {
        MapLocation k = Comms.nearestKing(G.me);
        if (k == null) return false;
        if (cat != null && nearHome(cat.location, Cfg.FORT_R + 5)) return false; // defend against a cat at home
        if (G.cheb(G.me, k) > Cfg.FORT_R) { Nav.goTo(k); return true; }
        // on station: face outward, toward the map centre
        MapLocation mid = new MapLocation(G.W / 2, G.H / 2);
        Direction f = Cfg.SYM_CENTRE ? G.toCentre(k) : k.directionTo(mid);
        if (f != Direction.CENTER && G.rc.getDirection() != f && G.rc.canTurn(f)) { G.rc.turn(f); G.facing = f; }
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
        if (raw >= Cfg.DELIVER_AT || G.rc.getCarrying() != null || !catReach()) return false;
        if (threatenedKing() != null || nearEnemy(8)) return false;
        // a king formation (END_KING, CASH_END) outweighs the cat: its joiners go on to formKing and cashMove
        if (formSite() != null || cashSite != null) return false;
        if (Comms.numKings < 2 && G.globalCheese >= endCatBar(G.round)) return false;
        if (cat != null) return approachCat(cat);
        MapLocation ct = knownCat();
        if (ct == null) return false;
        Nav.goTo(ct);
        catStalled(ct);
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
                if (d < bd || Cfg.SYM_TIE && d == bd && G.tieFirst(me, t, best)) { bd = d; best = t; }
            }
        if (best == null) return false;
        Nav.goTo(best);
        catStalled(best);
        return true;
    }

    /** CAT_REACH: the cat branches are open (no ban running). */
    static boolean catReach() { return reachTurns == 0 || G.round >= catBanUntil; }

    /** CAT_REACH: after a step toward the cat at t, a walk stalled reachTurns turns or known to be a detour closes the
     *  cat branches for CAT_REACH_TTL rounds. */
    static void catStalled(MapLocation t) {
        if (reachTurns > 0 && (Nav.stall >= reachTurns || Flow.detour(t))) catBanUntil = G.round + Cfg.CAT_REACH_TTL;
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
        int m = ours > 0 && !(Cfg.MINE_ANY || Cfg.MINE_ANY_COOP && G.coop) ? ours : n;
        long[] key = candKey;
        boolean rr = Cfg.MINE_RR && rrSlot >= 0 && origin != null;
        MapLocation kh = rr ? origin : home;
        for (int i = 0; i < m; i++) key[i] = mineKey(cand[i], kh);
        for (int i = 1; i < m; i++) {   // insertion sort of the first m by mineKey
            long k = key[i];
            MapLocation c = cand[i];
            int j = i;
            for (; j > 0 && k < key[j - 1]; j--) { key[j] = key[j - 1]; cand[j] = cand[j - 1]; }
            key[j] = k; cand[j] = c;
        }
        if (postMargin > 0 && role == FIGHT && !G.coop && m == ours) {
            // a fighter at war holds a contested home mine, where enemy thieves pick; none known: keep all
            MapLocation r = ref(), e = G.mirror(r, firstSym());
            int k = 0;
            for (int i = 0; i < m; i++) if (G.cheb(cand[i], e) - G.cheb(cand[i], r) <= postMargin) cand[k++] = cand[i];
            if (k > 0) m = k;
        }
        coopAtPick = G.coop;
        if (Cfg.KMINE_SKIP > 0 && ours >= Cfg.KMINE_SKIP) {
            // a king works its own mine: drop king-held mines from the pick, keeping the order (all held: keep all)
            int k = 0;
            for (int i = 0; i < m; i++) if (!kingHeld(cand[i])) cand[k++] = cand[i];
            if (k > 0) m = k;
        }
        minesAtPick = Comms.numMines;
        if (Cfg.HOME_ALL_SYM) maskAtPick = Sym.mask;
        if (Cfg.TRIP_HALF != 0) tripSigAtPick = Comms.tripSig;
        if (D.ON) D.minePick(n, ours, m);   // MINE_PICK opens (secondary slot; BABY_MOVE is open)
        if (D.NET && D.point2 == D.MINE_PICK) {
            MapLocation p = minePickExec(D.pick2(D.MINE_PICK, null), n);
            if (p != null) { D.emit2(); return p; }
        }
        if (rr) { D.lab2 = rrSlot % m; D.emit2(); return cand[rrSlot % m]; }
        if (Cfg.MINE_BALANCE) { D.lab2 = Cfg.IDLE_SHIFT ? (slot % m + mineShift) % m : slot % m; D.emit2(); return Cfg.IDLE_SHIFT ? cand[(slot % m + mineShift) % m] : cand[slot % m]; }
        int kr = G.rand(Math.min(m, Cfg.MINE_SPREAD)); D.lab2 = kr; D.emit2(); return cand[kr];
    }

    /** Cfg.LOAD_AT: a gatherer carrying raw cheese heads home once it holds at least the load, or late in the game. */
    static boolean fullLoad(int raw, boolean late, int load) {
        return raw > 0 && (raw >= load || late);
    }

    /** Cfg.ATK_DELIVER: an attacker carrying raw cheese at the threshold or more heads home with it. */
    static boolean atkHome(int raw) {
        return Cfg.ATK_DELIVER > 0 && role == ATTACK && raw >= Cfg.ATK_DELIVER;
    }

    /** Cfg.KMINE_SKIP: one of our kings stands within Chebyshev 4 of mine m. */
    static boolean kingHeld(MapLocation m) {
        for (int k = 0; k < Comms.numKings; k++) if (G.cheb(Comms.kings[k], m) <= 4) return true;
        return false;
    }

    /** Cfg.MINE_EXPLORE: the rat with hash `slot` explores while it knows m mines on our half (extra exploring slots). */
    static boolean explorer(int slot, int m, int extra) {
        return slot % (m + extra) >= m;
    }

    /** Distinct mines on our half that we know (the king's list and our own). */
    static int homeMineCount() {
        int n = 0;
        for (int i = 0; i < Comms.numMines && n < cand.length; i++) n = addCand(Comms.mines[i], n);
        for (int i = 0; i < myMineCount && n < cand.length; i++) n = addCand(new MapLocation(myMines[i] / 64, myMines[i] % 64), n);
        int ours = 0;
        for (int i = 0; i < n; i++) if (homeHalf(cand[i])) ours++;
        return ours;
    }

    /** A per-rat hash (the scout pick, Cfg.MINE_BALANCE), and the team's known-mine count when mineTarget was picked. */
    static int slot, minesAtPick, maskAtPick;

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
        if (sd > Cfg.FORM_PULL_R2 && !(Cfg.FORM_MINE && nearestEnemy == null && minePull(sd, role, slot))) return false;
        if (Cfg.FORM_JOIN > 0 && G.cheb(G.me, site) > 1 && nearerAllies(site) >= Cfg.FORM_JOIN) return false;
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
            // Cfg.RICH_FIX (a rich site: two or more kings): off the centre, go on the way we came in, onto a free
            // tile of the 3x3 (holes drift to the joiners)
            int cd0 = G.debt, cb0 = G.bc();   // ledger (K 9)
            if (Cfg.RICH_FIX && G.fixOn(Comms.numKings) && !G.me.equals(site) && site.equals(fillSite) && fillHead != null)
                shift(site, G.fan(G.me, fillHead));
            G.debt = cd0 + G.bc() - cb0 + 9;
            return true;
        }
        int cd1 = G.debt, cb1 = G.bc();   // ledger (K 11: 9, iload sh and ifne)
        // Cfg.RICH_FIX: one tile out, step onto any free tile of the 3x3 rather than queue for the centre
        MapLocation was = G.me;
        boolean sh = Cfg.RICH_FIX && G.fixOn(Comms.numKings) && G.cheb(G.me, site) == 2 && shift(site, G.fan5(G.me, G.me.directionTo(site)));
        G.debt = cd1 + G.bc() - cb1 + 11;
        if (!sh) Nav.goTo(site);
        int cd2 = G.debt, cb2 = G.bc();   // ledger (K 9)
        if (Cfg.RICH_FIX && G.fixOn(Comms.numKings) && G.cheb(G.me, site) <= 1) { fillSite = site; fillHead = was.directionTo(G.me); }
        G.debt = cd2 + G.bc() - cb2 + 9;
        return true;
    }

    /** Cfg.RICH_FIX: the site we last walked into and the heading we entered its 3x3 with. */
    static MapLocation fillSite;
    static Direction fillHead;

    /** Cfg.RICH_FIX: step to the first free tile of site's 3x3 (never its centre: that step is formKing's) among
     *  dirs. Inside, dirs are the entry heading and its 45-degree neighbours, so every shift gains along the heading
     *  and a rat shifts at most twice per site: no two rats trade tiles back and forth. */
    static boolean shift(MapLocation site, Direction[] dirs) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isMovementReady()) return false;
        for (Direction d : dirs) {
            MapLocation t = G.me.add(d);
            if (G.cheb(t, site) > 1 || t.equals(site) || !rc.canMove(d)) continue;
            if (rc.getDirection() != d && rc.canTurn(d)) { rc.turn(d); G.facing = d; }
            if (!rc.canMove(d)) return false;
            rc.move(d); G.me = rc.getLocation();
            return true;
        }
        return false;
    }

    /** Cfg.FORM_MINE: a baby beyond FORM_PULL_R2 (d2 from the site) still joins within FORM_MINE_PULL_R2 when its
     *  slot is drawn; attackers never. */
    static boolean minePull(int d2, int role, int slot) {
        return d2 <= Cfg.FORM_MINE_PULL_R2 && role != ATTACK && (slot >>> 3) % Cfg.FORM_MINE_SHARE == 0;
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

    /** A rescue may run: the bank is low but pays for a king, a king lives, and the engine allows one more (only a
     *  second after round 1200). */
    static boolean cashOk(int bank, int kings, int round) {
        return bank < Cfg.CASH_BANK && bank >= Cfg.CASH_FLOOR && kings > 0 && kings <= Cfg.CASH_KINGS
            && kings < (round > 1190 ? 2 : Comms.KING_SLOTS)
            || cashEnd(bank, kings, round);
    }

    /** Cfg.CASH_END: an endgame second king may run, whatever the bank holds in raw: from round CASH_END a lone king,
     *  and a bank that keeps a new king alive to round 2000 (both kings eat 2 a round; a starving king loses 10 health
     *  a round and a new one has up to 600, so it outlives 55 rounds on an empty bank). */
    static boolean cashEnd(int bank, int kings, int round) {
        return Cfg.CASH_END > 0 && round >= Cfg.CASH_END && kings == 1
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
        if (Cfg.PATH_ORACLE) { int v = PathOracle.verdict(l, ref()); if (v != 0) return v > 0; }
        if (Cfg.TRIP_HALF != 0) { int v = tripVerdict(l, Cfg.TRIP_HALF); if (v != 0) return v > 0; }
        return Cfg.HOME_ALL_SYM ? nearerAll(l, ref(), Sym.mask) : nearer(l, ref(), firstSym());
    }

    /** Cfg.HOME_ALL_SYM: l is nearer h than h's twin under every symmetry in mask (the first live one if none is). */
    static boolean nearerAll(MapLocation l, MapLocation h, int mask) {
        if (mask == 0) return nearer(l, h, firstSym());
        for (int s = Sym.ROT; s <= Sym.VER; s <<= 1) if ((mask & s) != 0 && !nearer(l, h, s)) return false;
        return true;
    }

    /** Whether l is nearer h than h's twin under symmetry s (ties count with Cfg.MINE_TIES). */
    static boolean nearer(MapLocation l, MapLocation h, int s) {
        int a = l.distanceSquaredTo(h), b = l.distanceSquaredTo(G.mirror(h, s));
        return a < b || Cfg.MINE_TIES && a == b;
    }

    /** The king start squeaked to us (Comms.SQ_START), or null; sets rrSlot from its spawn-order field. */
    static MapLocation heardStart(Message[] ms) {
        for (Message m : ms) {
            int c = m.getBytes();
            if (Comms.sqType(c) == Comms.SQ_START) {
                rrSlot = Comms.sqExtra(c) - 1;
                return new MapLocation(Comms.sqX(c), Comms.sqY(c));
            }
        }
        return null;
    }

    /** Cfg.MINE_RR: our place in our role's spawn order, squeaked by the king that spawned us (-1: not heard). */
    static int rrSlot = -1;

    /** Home for the half test and mirrors: our first king's start with Cfg.HOME_START once heard, else home. */
    static MapLocation ref() { return (Cfg.HOME_START || Cfg.FORM_MINE) && origin != null ? origin : home; }

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
            if (Cfg.RAID_TRAPS > 0) mineTrap(raidTarget, Cfg.RAID_TRAPS);
        } else {
            Nav.goTo(raidTarget);
        }
        return true;
    }

    /** A mine on the enemy half: known mines and the mirrors of our own, under the first live symmetry. */
    static MapLocation pickRaid() {
        int s = firstSym();
        MapLocation ek = G.mirror(ref(), s);
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
            // a tile under one of our kings can never be reached: draw again (twice at most)
            for (int k = 0; k < 2 && Cfg.MINE_KING_R > 0 && Nav.atKing(exploreTarget); k++)
                exploreTarget = new MapLocation(clampX(m.x + fu * (G.rand(7) - 3)), clampY(m.y + fv * (G.rand(7) - 3)));
        }
        Nav.goTo(exploreTarget);
    }

    /** At mine m: within Chebyshev 2, or within kingR (when > 0) if one of our kings sits within 2 of m. */
    static boolean atMine(MapLocation m, int kingR) {
        int d = G.cheb(G.me, m);
        if (d <= 2) return true;
        if (kingR <= 2 || d > kingR) return false;
        for (int k = 0; k < Comms.numKings; k++) if (G.cheb(Comms.kings[k], m) <= 2) return true;
        return false;
    }

    /** Cfg.TRIP_HALF: this rat reached mine m in round r; a walk under way is timed, and kept for a king when the mine
     *  is listed and the walk beats the one kept for it. */
    static void tripArrive(MapLocation m, int r) {
        if (tripFrom <= 0) return;
        int t = r - tripFrom;
        tripFrom = 0;
        for (int k = 0; k < Comms.numMines; k++) {
            if (Comms.mines[k].x != m.x || Comms.mines[k].y != m.y) continue;
            if (Comms.trip[k] == 0 || 4 * Comms.trip[k] > t) { tripKey = m.x * 64 + m.y; tripT = t; }
            return;
        }
    }

    /** Cfg.TRIP_HALF: +1 when listed mine l is ours by timed walks (margin steps), -1 theirs, 0 without a verdict. */
    static int tripVerdict(MapLocation l, int margin) {
        MapLocation t = G.mirror(l, firstSym());
        if (t.x == l.x && t.y == l.y) return 0;
        int a = -1, b = -1;
        for (int k = 0; k < Comms.numMines; k++) {
            MapLocation m = Comms.mines[k];
            if (m.x == l.x && m.y == l.y) a = Comms.trip[k];
            else if (m.x == t.x && m.y == t.y) b = Comms.trip[k];
        }
        MapLocation r = ref();
        return tripSide(a, b, G.cheb(l, r), G.cheb(t, r), margin);
    }

    /** Walks a to a mine and b to its twin (steps / 4; 0 not timed, -1 not listed) and their Chebyshev distances ca,
     *  cb from our start (the least walk): +1 ours when its walk beats the twin's by margin, -1 when it trails by
     *  margin and the twin is listed (so rats can go and time it), else 0. Untimed walks count at their least. */
    static int tripSide(int a, int b, int ca, int cb, int margin) {
        if (a < 0 || a == 0 && b <= 0) return 0;
        int wa = a > 0 ? 4 * a : ca, wb = b > 0 ? 4 * b : cb;
        if (wa + margin <= wb) return 1;
        if (b >= 0 && wb + margin <= wa) return -1;
        return 0;
    }

    /** Cfg.BREAKOUT_TTL is on and this rat is breaking out. */
    static boolean breakingOut() { return Cfg.BREAKOUT_TTL > 0 && G.round < breakoutUntil; }

    /** A chase of walled-off cheese given up in round r: the second within 100 rounds of the last starts a breakout. */
    static void wallStrike(int r, int ttl) {
        wallStrikes = r - lastStrike > 100 ? 1 : wallStrikes + 1;
        lastStrike = r;
        if (wallStrikes >= 2) breakoutUntil = r + ttl;
    }

    /** Breakout exploring: follow a frontier path (to the never-seen tile nearest by path cost, digging dirt on the
     *  way), holding still while its forward search runs. False when none is found (nothing unseen is reachable). */
    static boolean exploreFrontier() throws GameActionException {
        if (!Flow.frontierLive()) Flow.requestFrontier();
        if (!Flow.frontierLive()) return false;
        if (Flow.building) return true;
        exploreTarget = Flow.tgt;
        Nav.goTo(exploreTarget);
        return true;
    }

    static void explore() throws GameActionException {
        if (Cfg.BREAKOUT_TTL > 0) {
            if (exploreTarget != null && Flow.failTgt != null && G.cheb(Flow.failTgt, exploreTarget) <= 1
                    && G.round - Flow.failRound < Cfg.FLOW_RETRY) breakoutUntil = G.round + Cfg.BREAKOUT_TTL;
            if (breakingOut() && exploreFrontier()) return;
        }
        if (exploreTarget == null || G.cheb(G.me, exploreTarget) <= 2 || Nav.stall > 20) {
            // target in our side frame: mirrored rats pick mirrored targets. Take the nearest never-seen
            // sample: once open ground is mapped, what is left unseen is buried in dirt, and Flow digs to it.
            // With no mine known by EXPLORE_CENTER_FROM, rank by closeness to the map centre instead: mines
            // that nobody has seen are buried between the kings, and shallow edge pockets do not reveal them.
            boolean centre = Comms.numMines == 0 && G.round >= Cfg.EXPLORE_CENTER_FROM;
            int ox = centre ? (G.W - 1) / 2 : G.me.x, oy = centre ? (G.H - 1) / 2 : G.me.y;
            int x = 0, y = 0, bd = Integer.MAX_VALUE;
            for (int k = 0; k < Cfg.EXPLORE_TRIES; k++) {
                int ax = G.rand(G.W), ay = G.rand(G.H);
                int cx = fu > 0 ? ax : G.W - 1 - ax, cy = fv > 0 ? ay : G.H - 1 - ay;
                int d = Sym.known(G.idx(cx, cy)) ? 1 << 20 : Cfg.SYM_CENTRE && centre ? G.centreCheb2(cx, cy)
                    : Math.max(Math.abs(cx - ox), Math.abs(cy - oy));
                if (d < bd) { bd = d; x = cx; y = cy; }
            }
            exploreTarget = new MapLocation(x, y);
        }
        Nav.goTo(exploreTarget);
    }

    static boolean scout;
    /** The most home mines this rat has known, and the round that count last grew (Cfg.SCOUT_DRY). */
    static int scoutMines = -1, scoutFound;

    /** A gatherer scout still scouting: until scoutUntil, or with Cfg.SCOUT_MINES while we know too few home mines. */
    static boolean scouting() {
        if (!scout || role != GATHER) return false;
        int until = scoutUntil(Cfg.SCOUT_AREA);
        if (G.round < until) return true;
        if (Cfg.SWEEP > 0 && G.round < Cfg.SWEEP_UNTIL && !sweepDone) return true;
        if (!moreScouting(Cfg.SCOUT_MINES, G.round)) return false;
        int h = homeMineCount();
        if (scoutMines < 0) { scoutMines = h; scoutFound = until; }
        else if (h > scoutMines) { scoutMines = h; scoutFound = G.round; }
        return h < Cfg.SCOUT_MINES && !scoutDry(Cfg.SCOUT_DRY, G.round, scoutFound);
    }

    /** Cfg.SCOUT_MINES is on and round r is before SCOUT_MINES_UNTIL. */
    static boolean moreScouting(int mines, int r) { return mines > 0 && r < Cfg.SCOUT_MINES_UNTIL; }

    /** No new home mine for dry rounds since round found (dry 0: never dry). */
    static boolean scoutDry(int dry, int r, int found) { return dry > 0 && r - found >= dry; }

    /** The round scouting ends: SCOUT_UNTIL, or W*H/area on a map big enough to make that later (area 0: off). */
    static int scoutUntil(int area) { return area > 0 ? Math.max(Cfg.SCOUT_UNTIL, G.W * G.H / area) : Cfg.SCOUT_UNTIL; }

    /** exploreHome's sample rank (lower wins) for a sample at Chebyshev distance d: Cfg.HOME_PICK 2 prefers far. */
    static int homeRank(int pick, int d) { return pick == 2 ? -d : d; }

    /** Scout movement: the never-seen tile of our half that Cfg.HOME_PICK takes from EXPLORE_TRIES samples; false
     *  (no move) when no sample is one. */
    static boolean exploreHome() throws GameActionException {
        if (Cfg.SWEEP > 0) return sweep(Cfg.SWEEP, 0);
        if (exploreTarget == null || G.cheb(G.me, exploreTarget) <= 2 || Nav.stall > 20 || Sym.known(G.idx(exploreTarget.x, exploreTarget.y))) {
            int x = -1, y = -1, bd = Integer.MAX_VALUE;
            for (int k = 0; k < Cfg.EXPLORE_TRIES; k++) {
                int ax = G.rand(G.W), ay = G.rand(G.H);
                int cx = fu > 0 ? ax : G.W - 1 - ax, cy = fv > 0 ? ay : G.H - 1 - ay;
                if (Sym.known(G.idx(cx, cy)) || !homeHalf(new MapLocation(cx, cy))) continue;
                int d = homeRank(Cfg.HOME_PICK, Math.max(Math.abs(cx - G.me.x), Math.abs(cy - G.me.y)));
                if (d < bd) { bd = d; x = cx; y = cy; }
                if (Cfg.HOME_PICK == 1) break;
            }
            if (x < 0) return false;
            exploreTarget = new MapLocation(x, y);
        }
        Nav.goTo(exploreTarget);
        return true;
    }

    /** Cfg.SWEEP: the cell being swept plus 1 (0 none), the round it is banned if still unseen, banned cells (lazy),
     *  and whether no cell of our half is left. A pick scans the cells over as many turns as the bytecodes need
     *  (sweepScan: the next cell plus 1, 0 = no scan under way; best score and cell so far; the scan's origin). */
    static int sweepCell, sweepBy, sweepScan, sweepBest, sweepBi, sweepOx, sweepOy;
    static boolean[] sweepBan;
    static boolean sweepDone;

    /** Cfg.SWEEP scout movement: toward the centre of the unseen, unbanned cell (c x c, in the side frame) of our half
     *  that minimises the walk there plus its distance on to the lane anchor; false (sweepDone) once there is none.
     *  cap > 0 (tests): scan at most cap cells a call; 0: scan while Cfg.SWEEP_LEFT bytecodes are left. */
    static boolean sweep(int c, int cap) throws GameActionException {
        int w = G.W, h = G.H, gw = (w + c - 1) / c, gh = (h + c - 1) / c, n = gw * gh;
        if (sweepBan == null || sweepBan.length != n) sweepBan = new boolean[n];
        if (sweepCell > 0 && (exploreTarget == null || Sym.known(G.idx(exploreTarget.x, exploreTarget.y)))) sweepCell = 0;
        else if (sweepCell > 0 && (Nav.stall > 20 || G.round > sweepBy)) { sweepBan[sweepCell - 1] = true; sweepCell = 0; }
        if (sweepCell == 0) {
            if (sweepScan == 0) {
                sweepScan = 1; sweepBest = Integer.MAX_VALUE; sweepBi = -1;
                sweepOx = fu > 0 ? G.me.x : w - 1 - G.me.x; sweepOy = fv > 0 ? G.me.y : h - 1 - G.me.y;
            }
            int lane = sweepLane(slot);
            int ax = sweepAnchor(lane, w, true), ay = sweepAnchor(lane, h, false);
            // the half test hoisted out of the loop when it is the plain nearer-our-start one
            boolean plain = !Cfg.PATH_ORACLE && Cfg.TRIP_HALF == 0 && !Cfg.HOME_ALL_SYM;
            MapLocation rf = ref(), tw = G.mirror(rf, firstSym());
            int k = sweepScan - 1, end = cap > 0 ? Math.min(n, k + cap) : n;
            for (; k < end; k++) {
                if (cap == 0 && Clock.getBytecodesLeft() + (G.debt += 8) - 6 < Cfg.SWEEP_LEFT) break;
                int j = k / gw, i = k - j * gw;
                int fx = Math.min(w - 1, i * c + c / 2), fy = Math.min(h - 1, j * c + c / 2);
                int sc = sweepScore(fx, fy, sweepOx, sweepOy, ax, ay);
                if (sc >= sweepBest || sweepBan[k]) continue;
                int x = fu > 0 ? fx : w - 1 - fx, y = fv > 0 ? fy : h - 1 - fy;
                if (Sym.known(G.idx(x, y))) continue;
                if (plain) {
                    int da = (x - rf.x) * (x - rf.x) + (y - rf.y) * (y - rf.y), db = (x - tw.x) * (x - tw.x) + (y - tw.y) * (y - tw.y);
                    if (!(da < db || Cfg.MINE_TIES && da == db)) continue;
                } else if (!homeHalf(new MapLocation(x, y))) continue;
                sweepBest = sc; sweepBi = k;
            }
            if (k < n) { sweepScan = k + 1; return true; } // the scan goes on next turn
            sweepScan = 0;
            if (sweepBi < 0) { sweepDone = true; return false; }
            int j = sweepBi / gw, i = sweepBi - j * gw;
            int fx = Math.min(w - 1, i * c + c / 2), fy = Math.min(h - 1, j * c + c / 2);
            sweepCell = sweepBi + 1;
            exploreTarget = new MapLocation(fu > 0 ? fx : w - 1 - fx, fv > 0 ? fy : h - 1 - fy);
            sweepBy = G.round + 2 * G.cheb(G.me, exploreTarget) + 20;
        }
        Nav.goTo(exploreTarget);
        return true;
    }

    /** Cfg.SWEEP: a scout's lane from its (mirror-invariant) slot: 0 the map centre, 1-4 the corners. */
    static int sweepLane(int slot) { return (slot >>> 10) % 5; }

    /** Cfg.SWEEP: lane's anchor coordinate along a side of length n, in the side frame (our king near (n-1, n-1)):
     *  lane 0 the centre, 1 (n-1, 0), 2 (0, n-1), 3 (n-1, n-1), 4 (0, 0). */
    static int sweepAnchor(int lane, int n, boolean x) {
        if (lane == 0) return (n - 1) / 2;
        return (x ? lane == 1 || lane == 3 : lane == 2 || lane == 3) ? n - 1 : 0;
    }

    /** Cfg.SWEEP: a cell centre's rank (lower first): Chebyshev from the rat plus on to the anchor, then nearer the
     *  rat, all in the frame. */
    static int sweepScore(int fx, int fy, int mx, int my, int ax, int ay) {
        int d = Math.max(Math.abs(fx - mx), Math.abs(fy - my));
        return (d + Math.max(Math.abs(fx - ax), Math.abs(fy - ay))) * 64 + d;
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
        if (Cfg.SEEK_CROWD <= 0 || G.round < seekNext || nearerAllies(mine) < Cfg.SEEK_CROWD) return false;
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
        } else if (Cfg.SEEK_FRONTIER && exploreFrontier()) return true;
        else if (exploreHome()) return true;
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
            case D.A_CASH_KING:
                if (cashSite == null || !me.equals(cashSite) || !rc.canBecomeRatKing()) return false;
                rc.becomeRatKing(); formed++; cashed++;
                return true;
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
                rc.dropRat(d); Nav.swapFrom = null;
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

    /** BABY_FACE: face option k (guardTurn's threat bookkeeping kept). */
    static boolean faceExec(int k) throws GameActionException {
        RobotController rc = G.rc;
        if (TrapRisk.nHeard > 0) threatRound = G.round;
        MapLocation me = G.me;
        switch (k) {
            case D.F_NONE: return true;
            case D.F_SEEN: {
                MapLocation t = null;
                int bd = Integer.MAX_VALUE;
                for (RobotInfo r : rc.senseNearbyRobots(Cfg.GUARD_DIST2, G.them)) {
                    if (r.type != UnitType.BABY_RAT) continue;
                    int d = me.distanceSquaredTo(r.location);
                    if (d < bd) { bd = d; t = r.location; }
                }
                return t != null && faceTo(t);
            }
            case D.F_MEMORY: {
                MapLocation t = null;
                int bd = Integer.MAX_VALUE;
                for (RobotInfo r : seen) {
                    if (r.team != G.them || r.type != UnitType.BABY_RAT) continue;
                    int d = me.distanceSquaredTo(r.location);
                    if (d <= Cfg.GUARD_DIST2 && d < bd) { bd = d; t = r.location; }
                }
                return t != null && faceTo(t);
            }
            case D.F_HEARD: {
                MapLocation t = null;
                int bd = Integer.MAX_VALUE;
                for (int i = 0; i < TrapRisk.nHeard; i++) {
                    int d = me.distanceSquaredTo(TrapRisk.heardE[i]);
                    if (d <= 8 && d < bd) { bd = d; t = TrapRisk.heardE[i]; }
                }
                return t != null && faceTo(t);
            }
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
            case D.C_MINE_GOSSIP: {
                if (myMineCount <= 0) return false;
                int key = myMines[G.round % myMineCount];
                return rc.squeak(Comms.squeak(Comms.SQ_MINE, key / 64, key % 64, 0));
            }
            case D.C_MINE_RELAY: {
                int key = unreported();
                return key >= 0 && rc.squeak(Comms.squeak(Comms.SQ_MINE, key / 64, key % 64, 0));
            }
            case D.C_SYM_KING:
            case D.C_SYM_GOSSIP: {
                int sh = Cfg.BURN ? Comms.read(Comms.SYM) & 7 : Comms.read(Comms.SYM);
                if (sh == 0 || (Sym.mask & sh) == sh) return false;
                if (!rc.squeak(Comms.squeak(Comms.SQ_SYM, 0, 0, Sym.mask))) return false;
                if (k == D.C_SYM_GOSSIP) symSqueaked = G.round;
                return true;
            }
            case D.C_MINE_KING:
                for (int i = 0; i < myMineCount; i++) {
                    int x = myMines[i] / 64, y = myMines[i] % 64;
                    if (!Comms.hasMine(x, y)) return rc.squeak(Comms.squeak(Comms.SQ_MINE, x, y, 0));
                }
                return false;
            case D.C_CAT:
                if (catSeen == null || G.round - catSeenRound > 2) return false;
                return rc.squeak(Comms.squeak(Comms.SQ_CAT, catSeen.x, catSeen.y, 0));
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
            case D.C_TRAP:
                if (!Cfg.TRAP_SHARE || nearestEnemy == null || nearestEnemy.direction == null) return false;
                return rc.squeak(Comms.squeak(Comms.SQ_TRAP, nearestEnemy.location.x, nearestEnemy.location.y, nearestEnemy.direction.ordinal()));
            default: return false;   // C_NONE, C_BURN, C_TRIP (masked)
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
        if (Cfg.GRAB_MICRO && GrabMicro.round == G.round) { moveDecision(); return; }
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
        boolean reach = catReach(), kc = knownCat() != null, mineOk = mineTarget != null || Comms.numMines + myMineCount > 0;
        int raw = rc.getRawCheese();
        a[D.M_GRAB] = false;
        a[D.M_CAT_DEFEND] = cat != null && reach;
        a[D.M_END_CAT] = reach && (cat != null || kc);
        a[D.M_FLEE_CAT] = cat != null;
        a[D.M_YIELD_KING] = cat != null;
        a[D.M_AIM_CARRY] = c != null && c.team == G.them;
        a[D.M_RESCUE] = rescueT != null && rc.isActionReady();
        a[D.M_AMMO_AWAY] = nearestEnemy != null && k != null;
        a[D.M_RETREAT] = k != null && nearEnemy(20);
        a[D.M_HUNGRY] = false;   // alias of M_DELIVER
        a[D.M_FORM_KING] = formSite() != null;
        a[D.M_CASH] = Cfg.CASH_BANK > 0 && cashSite != null;
        a[D.M_RAID] = true;
        a[D.M_FLANK] = nearestEnemy != null && rc.isActionReady() && c == null;
        a[D.M_PRESS] = eKing != null || lastEKing != null && G.round - lastEKingRound <= Cfg.PRESS_AGE;
        a[D.M_MICRO] = nearestEnemy != null;
        a[D.M_FACE] = nearEnemy(8);
        a[D.M_ATK_DELIVER] = false;   // alias of M_DELIVER
        a[D.M_STALL_CAT] = role == ATTACK;
        a[D.M_ATTACK] = true;
        a[D.M_WAR] = nearestEnemy != null || eKing != null || threatenedKing() != null || Comms.fresh(Comms.read(Comms.ALERT + 2), 3);
        a[D.M_FORT] = Cfg.FORT_R > 0 && k != null;
        a[D.M_FIGHT_CAT] = cat != null && reach;
        a[D.M_DELIVER] = raw > 0 && k != null;
        a[D.M_COURIER] = k != null;
        a[D.M_CHEESE_PICK] = cheeseTarget != null && G.cheb(G.me, cheeseTarget) == 1;
        a[D.M_CHEESE] = cheeseTarget != null;
        a[D.M_KNOWN_CAT] = reach && kc;
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
            case D.M_END_CAT: {
                boolean ok;
                if (cat != null) ok = approachCat(cat);
                else { MapLocation ct = knownCat(); if (ct == null) return false; Nav.goTo(ct); catStalled(ct); ok = true; }
                if (!ok) return false;
                why = 'e'; return true;
            }
            case D.M_FLEE_CAT:
                if (cat == null) return false;
                if (!(Cfg.SYM_CENTRE ? Nav.flee2(2 * cat.location.x + 1, 2 * cat.location.y + 1) : Nav.flee(cat.location))) return false;
                why = 'c'; return true;
            case D.M_YIELD_KING:
                if (cat == null || !yieldKing()) return false;
                why = 'y'; return true;
            case D.M_AIM_CARRY: {
                RobotInfo c = rc.getCarrying();
                if (c == null || c.team != G.them || !aimCarry()) return false;
                why = 'T'; return true;
            }
            case D.M_RESCUE:
                if (rescueT == null || !rescue()) return false;
                why = 'R'; return true;
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
                if (Cfg.CASH_BANK <= 0 || cashSite == null || !cashMove()) return false;
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
            case D.M_FORT:
                if (Cfg.FORT_R <= 0 || !fort()) return false;
                why = 'h'; return true;
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
            case D.M_CHEESE_PICK: {
                if (cheeseTarget == null || G.cheb(G.me, cheeseTarget) != 1) return false;
                Direction d = G.me.directionTo(cheeseTarget);
                if (rc.getDirection() != d && rc.canTurn(d)) { rc.turn(d); G.facing = d; }
                pickUp();
                if (!rc.canSenseLocation(cheeseTarget) || rc.senseMapInfo(cheeseTarget).getCheeseAmount() == 0) cheeseTarget = null;
                why = 'P'; return true;
            }
            case D.M_CHEESE:
                if (cheeseTarget == null) return false;
                why = 'C';
                Nav.goTo(cheeseTarget);
                if (Nav.stall > 12) {
                    if (Cfg.CHEESE_BAN > 0 && (!Cfg.CHEESE_BAN_WALL || Flow.detour(cheeseTarget))) banCheese(cheeseTarget);
                    if (Cfg.BREAKOUT_TTL > 0 && walledOff(G.me, cheeseTarget)) wallStrike(G.round, Cfg.BREAKOUT_TTL);
                    cheeseTarget = null;
                }
                return true;
            case D.M_KNOWN_CAT: {
                MapLocation ct = catReach() ? knownCat() : null;
                if (ct == null) return false;
                why = 'g';
                Nav.goTo(ct); catStalled(ct); return true;
            }
            case D.M_SCOUT_HOME: {
                MapLocation kl = Comms.nearestKing(G.me);
                if (kl == null) return false;
                if (Cfg.SWEEP > 0) sweepCell = sweepScan = 0;
                Nav.goTo(kl); why = 's'; return true;
            }
            case D.M_SCOUT:
                why = 'S';
                if (Cfg.SCOUT_FRONTIER && exploreFrontier()) return true;
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
                Direction t = G.facing.rotateRight().rotateRight();
                if (rc.canTurn(t)) { rc.turn(t); G.facing = t; }
                if (Cfg.MINE_TRAPS > 0 && mineTarget != null) mineTrap(mineTarget, Cfg.MINE_TRAPS);
                return true;
            }
            case D.M_MINE_WANDER:
                if (!netMine()) return false;
                if (Cfg.TRIP_HALF != 0 && atMine(mineTarget, Cfg.MINE_KING_R)) tripArrive(mineTarget, G.round);
                wanderAround(mineTarget);
                if (Cfg.MINE_TRAPS > 0) mineTrap(mineTarget, Cfg.MINE_TRAPS);
                why = 'W'; return true;
            case D.M_MINE_GO:
                if (!netMine()) return false;
                why = 'M';
                Nav.goTo(mineTarget);
                if (Nav.stall > 25) mineTarget = null;
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

    /** The cascade's mine refresh, copied verbatim (no shared helper with the cascade in pass 1); true with a mine. */
    static boolean netMine() throws GameActionException {
        if (Cfg.MINE_BALANCE && mineTarget != null && (Cfg.HOME_ALL_SYM ? Comms.numMines != minesAtPick || Sym.mask != maskAtPick : Comms.numMines != minesAtPick)) mineTarget = null;
        if (Cfg.MINE_ANY_COOP && !G.coop && mineTarget != null && !homeHalf(mineTarget)) mineTarget = null;
        if (postMargin > 0 && role == FIGHT && coopAtPick && !G.coop) mineTarget = null;
        if (Cfg.KMINE_SKIP > 0 && mineTarget != null && G.round % 10 == 0 && kingHeld(mineTarget)
                && homeMineCount() >= Cfg.KMINE_SKIP) mineTarget = null;
        if (Cfg.TRIP_HALF != 0 && mineTarget != null && Comms.tripSig != tripSigAtPick) mineTarget = null;
        if (mineTarget == null) { D.pickSite = 0; mineTarget = pickMine(); }
        // the cascade's idle shift (its idle-seek branch is the separate option M_IDLE_SEEK)
        if (mineTarget != null && Cfg.SEEK_IDLE > 0 && (role == GATHER || Cfg.FIGHT_SHIFT && role == FIGHT) && idle(mineTarget)) {
            if (Cfg.IDLE_SHIFT) { mineShift++; D.pickSite = 1; mineTarget = pickMine(); }
            if (mineTarget == null) { D.pickSite = 2; mineTarget = pickMine(); }
        }
        return mineTarget != null;
    }
}
