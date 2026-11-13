package alpha0.student;

import battlecode.common.*;

/**
 * Primitive action space (student-actions-v3). One index = one controller call (or one private-memory write).
 * A turn is a sequence of primitives terminated by END (at most MAX_STEPS).
 *
 * 0 END | 1..8 MOVE dir | 9..16 TURN dir | 17.. typed target actions (NT types x 7x7 offsets)
 * | THROW | DROP dir x8 | BECOME_KING | SQUEAK | WRITE_SHARED index x64 | WRITE_REG index x32 | DISINTEGRATE.
 * resign() is deliberately excluded: it only forfeits the game, so no policy should ever be able to choose it.
 *
 * Some actions take an integer parameter, chosen after the action (docs/capability-principle.md):
 *   AMOUNT (16 bits) for ATTACK (cheese spent on the bite; 0 = plain bite), PICKUP (0 = everything on the tile)
 *   and TRANSFER (0 = all raw cheese); out-of-range amounts are clamped to what the engine allows.
 *   SQUEAK (32 bits): the message content.  VALUE (10 bits): the shared-array or register value.
 * Every API argument value is reachable, so the space covers the full state-changing RobotController API
 * (moveForward() is exactly move(getDirection()) in engine 1.2.6; indicator/timeline calls have no game effect).
 * WRITE_REG writes the robot's private memory registers (Obs.Memory.reg), which it reads back in its observation.
 */
public final class Actions {
    public static final String VERSION = "student-actions-v3";
    public static final int TR = 3, TW = 2 * TR + 1, TC = TW * TW;
    public static final int ATTACK = 0, CARRY = 1, BUILD = 2, RAT_TRAP = 3, CAT_TRAP = 4, PLACE_DIRT = 5,
        REMOVE_DIRT = 6, PICKUP = 7, TRANSFER = 8, REMOVE_RAT_TRAP = 9, REMOVE_CAT_TRAP = 10, NT = 11;
    public static final int END = 0, MOVE0 = 1, TURN0 = 9, TARGET0 = 17;
    public static final int THROW = TARGET0 + NT * TC, DROP0 = THROW + 1, BECOME_KING = DROP0 + 8;
    public static final int SHARED_N = GameConstants.SHARED_ARRAY_SIZE, REGS = 32;
    public static final int SQUEAK = BECOME_KING + 1, WRITE_SHARED0 = SQUEAK + 1, WRITE_REG0 = WRITE_SHARED0 + SHARED_N,
        DISINTEGRATE = WRITE_REG0 + REGS;
    public static final int N = DISINTEGRATE + 1;
    public static final int MAX_STEPS = 128;
    /** Parameter kinds and their bit widths. */
    public static final int P_NONE = 0, P_AMOUNT = 1, P_SQUEAK = 2, P_VALUE = 3;
    public static final int[] PARAM_BITS = {0, 16, 32, 10};
    public static final int PARAM_MAX_BITS = 32;
    static final Direction[] DIRS = Obs.DIRS;

    public static int target(int type, int dx, int dy) { return TARGET0 + type * TC + (dy + TR) * TW + (dx + TR); }

    public static int paramKind(int a) {
        if (a >= TARGET0 && a < THROW) {
            int type = (a - TARGET0) / TC;
            return type == ATTACK || type == PICKUP || type == TRANSFER ? P_AMOUNT : P_NONE;
        }
        if (a == SQUEAK) return P_SQUEAK;
        if (a >= WRITE_SHARED0 && a < DISINTEGRATE) return P_VALUE;
        return P_NONE;
    }

    static int dirIndex(Direction d) {
        for (int k = 0; k < 8; k++) if (DIRS[k] == d) return k;
        return -1;
    }

    /** Max squared offset to scan per type (rough superset of engine ranges). */
    private static int reach(int type, boolean king) {
        if (type == TRANSFER) return 9;
        if (king) return 8;
        return 2;
    }

    /**
     * Fill mask[0..N). Returns number of legal actions (END included). endActions: allow DISINTEGRATE
     * (it is part of the space, but a sampled policy is only given it when the caller opts in).
     */
    public static int legal(RobotController rc, Obs.Memory m, boolean[] mask, boolean endActions) {
        java.util.Arrays.fill(mask, false);
        mask[END] = true;
        int n = 1;
        // calls that need no cooldown and are allowed in any state
        if (m == null || m.lastSqueakRound != rc.getRoundNum()) { mask[SQUEAK] = true; n++; }
        boolean king = rc.getType() == UnitType.RAT_KING;
        if (king) for (int i = 0; i < SHARED_N; i++) { mask[WRITE_SHARED0 + i] = true; n++; }
        for (int i = 0; i < REGS; i++) { mask[WRITE_REG0 + i] = true; n++; }
        if (endActions) { mask[DISINTEGRATE] = true; n++; }
        if (rc.isBeingCarried() || rc.isBeingThrown()) return n;
        Direction facing = rc.getDirection();
        if (rc.isMovementReady()) for (int k = 0; k < 8; k++) if (rc.canMove(DIRS[k])) { mask[MOVE0 + k] = true; n++; }
        if (rc.isTurningReady()) for (int k = 0; k < 8; k++) if (DIRS[k] != facing && rc.canTurn(DIRS[k])) { mask[TURN0 + k] = true; n++; }
        boolean ready = rc.isActionReady();
        MapLocation me = rc.getLocation();
        int raw = rc.getRawCheese();
        for (int type = 0; type < NT; type++) {
            boolean needsAction = type != REMOVE_RAT_TRAP && type != REMOVE_CAT_TRAP && type != PICKUP;
            if (needsAction && !ready) continue;
            if (type == BUILD && !king) continue;
            if ((type == CARRY || type == TRANSFER) && king) continue;
            if (type == TRANSFER && raw <= 0) continue;
            int r2 = reach(type, king);
            for (int dy = -TR; dy <= TR; dy++) for (int dx = -TR; dx <= TR; dx++) {
                if (dx * dx + dy * dy > r2) continue;
                MapLocation p = me.translate(dx, dy);
                if (!rc.onTheMap(p)) continue;
                boolean ok;
                try {
                    switch (type) {
                        case ATTACK: ok = rc.canAttack(p); break;
                        case CARRY: ok = rc.canCarryRat(p); break;
                        case BUILD: ok = rc.canBuildRat(p); break;
                        case RAT_TRAP: ok = rc.canPlaceRatTrap(p); break;
                        case CAT_TRAP: ok = rc.canPlaceCatTrap(p); break;
                        case PLACE_DIRT: ok = rc.canPlaceDirt(p); break;
                        case REMOVE_DIRT: ok = rc.canRemoveDirt(p); break;
                        case PICKUP: ok = rc.canPickUpCheese(p); break;
                        case TRANSFER: ok = rc.canTransferCheese(p, raw); break;
                        case REMOVE_RAT_TRAP: ok = rc.canRemoveRatTrap(p); break;
                        default: ok = rc.canRemoveCatTrap(p); break;
                    }
                } catch (RuntimeException e) { ok = false; }
                if (ok) { mask[target(type, dx, dy)] = true; n++; }
            }
        }
        if (!king) {
            try { if (rc.canThrowRat()) { mask[THROW] = true; n++; } } catch (RuntimeException e) {}
            for (int k = 0; k < 8; k++) try { if (rc.canDropRat(DIRS[k])) { mask[DROP0 + k] = true; n++; } } catch (RuntimeException e) {}
            try { if (rc.canBecomeRatKing()) { mask[BECOME_KING] = true; n++; } } catch (RuntimeException e) {}
        }
        return n;
    }

    /**
     * Execute a legal action with its parameter. Returns false for END. DISINTEGRATE throws the engine's
     * RobotDeathException (callers outside the sandbox must destroy the robot).
     */
    public static boolean execute(RobotController rc, Obs.Memory m, int a, int param) throws GameActionException {
        if (a == END) return false;
        if (a < TURN0) { rc.move(DIRS[a - MOVE0]); return true; }
        if (a < TARGET0) { rc.turn(DIRS[a - TURN0]); return true; }
        if (a < THROW) {
            int t = a - TARGET0, type = t / TC, c = t % TC;
            MapLocation p = rc.getLocation().translate(c % TW - TR, c / TW - TR);
            int v = param & 0xffff;
            switch (type) {
                case ATTACK: { int spend = Math.min(v, rc.getAllCheese()); if (spend > 0) rc.attack(p, spend); else rc.attack(p); break; }
                case CARRY: rc.carryRat(p); break;
                case BUILD: rc.buildRat(p); break;
                case RAT_TRAP: rc.placeRatTrap(p); break;
                case CAT_TRAP: rc.placeCatTrap(p); break;
                case PLACE_DIRT: rc.placeDirt(p); break;
                case REMOVE_DIRT: rc.removeDirt(p); break;
                case PICKUP: if (v > 0) rc.pickUpCheese(p, v); else rc.pickUpCheese(p); break;
                case TRANSFER: { int raw = rc.getRawCheese(); rc.transferCheese(p, v > 0 ? Math.min(v, raw) : raw); break; }
                case REMOVE_RAT_TRAP: rc.removeRatTrap(p); break;
                default: rc.removeCatTrap(p); break;
            }
            return true;
        }
        if (a == THROW) { rc.throwRat(); return true; }
        if (a < BECOME_KING) { rc.dropRat(DIRS[a - DROP0]); return true; }
        if (a == BECOME_KING) { rc.becomeRatKing(); return true; }
        if (a == SQUEAK) { rc.squeak(param); if (m != null) m.lastSqueakRound = rc.getRoundNum(); return true; }
        if (a < WRITE_REG0) { rc.writeSharedArray(a - WRITE_SHARED0, param & 1023); return true; }
        if (a < DISINTEGRATE) { if (m != null) m.reg[a - WRITE_REG0] = param & 1023; return true; }
        rc.disintegrate();
        return true;
    }

    /**
     * Map a teacher controller call (made from the robot's current state) to an index, or -1.
     * The call's parameter (in the encoding above) is stored in param[0].
     */
    public static int fromCall(RobotController rc, String op, Object[] args, int[] param) {
        param[0] = 0;
        MapLocation me = rc.getLocation();
        int type;
        switch (op) {
            case "moveForward": { int k = dirIndex(rc.getDirection()); return k < 0 ? -1 : MOVE0 + k; }
            case "move": { int k = dirIndex((Direction) args[0]); return k < 0 ? -1 : MOVE0 + k; }
            case "turn": { int k = dirIndex((Direction) args[0]); return k < 0 ? -1 : TURN0 + k; }
            case "throwRat": return THROW;
            case "dropRat": { int k = dirIndex((Direction) args[0]); return k < 0 ? -1 : DROP0 + k; }
            case "becomeRatKing": return BECOME_KING;
            case "squeak": param[0] = (Integer) args[0]; return SQUEAK;
            case "writeSharedArray": {
                int i = (Integer) args[0], v = (Integer) args[1];
                if (i < 0 || i >= SHARED_N || v < 0 || v > 1023) return -1;
                param[0] = v;
                return WRITE_SHARED0 + i;
            }
            case "disintegrate": return DISINTEGRATE;
            case "attack": type = ATTACK; break;
            case "carryRat": type = CARRY; break;
            case "buildRat": type = BUILD; break;
            case "placeRatTrap": type = RAT_TRAP; break;
            case "placeCatTrap": type = CAT_TRAP; break;
            case "placeDirt": type = PLACE_DIRT; break;
            case "removeDirt": type = REMOVE_DIRT; break;
            case "pickUpCheese": type = PICKUP; break;
            case "transferCheese": type = TRANSFER; break;
            case "removeRatTrap": type = REMOVE_RAT_TRAP; break;
            case "removeCatTrap": type = REMOVE_CAT_TRAP; break;
            default: return -1;
        }
        MapLocation p = (MapLocation) args[0];
        int dx = p.x - me.x, dy = p.y - me.y;
        if (dx < -TR || dx > TR || dy < -TR || dy > TR) return -1;
        if (args.length > 1) {
            int amount = Math.max(0, Math.min(0xffff, (Integer) args[1]));
            if (type == ATTACK) param[0] = amount;
            else if (type == PICKUP) {
                int avail = Integer.MAX_VALUE;
                try { avail = rc.senseMapInfo(p).getCheeseAmount(); } catch (GameActionException | RuntimeException e) {}
                param[0] = amount >= avail ? 0 : amount;
            } else if (type == TRANSFER) param[0] = amount == rc.getRawCheese() ? 0 : amount;
        }
        return target(type, dx, dy);
    }
}
