package alpha0.student;

import battlecode.common.*;

/**
 * Primitive action space (student-actions-v1). One index = one controller call.
 * A turn is a sequence of primitives terminated by END.
 *
 * 0 END | 1..8 MOVE dir | 9..16 TURN dir | 17.. typed target actions (T types x 7x7 offsets)
 * | THROW | DROP dir x8 | BECOME_KING.
 *
 * Approximations (explicit): attack(loc,amount) is treated as attack(loc); pickUpCheese
 * with an amount as a full pickup; transferCheese always transfers all raw cheese.
 * squeak/writeSharedArray/disintegrate/resign are not part of the student's action space.
 */
public final class Actions {
    public static final String VERSION = "student-actions-v1";
    public static final int TR = 3, TW = 2 * TR + 1, TC = TW * TW;
    public static final int ATTACK = 0, CARRY = 1, BUILD = 2, RAT_TRAP = 3, CAT_TRAP = 4, PLACE_DIRT = 5,
        REMOVE_DIRT = 6, PICKUP = 7, TRANSFER = 8, REMOVE_RAT_TRAP = 9, REMOVE_CAT_TRAP = 10, NT = 11;
    public static final int END = 0, MOVE0 = 1, TURN0 = 9, TARGET0 = 17;
    public static final int THROW = TARGET0 + NT * TC, DROP0 = THROW + 1, BECOME_KING = DROP0 + 8;
    public static final int N = BECOME_KING + 1;
    static final Direction[] DIRS = Obs.DIRS;

    public static int target(int type, int dx, int dy) { return TARGET0 + type * TC + (dy + TR) * TW + (dx + TR); }

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

    /** Fill mask[0..N). Returns number of legal actions (END included). */
    public static int legal(RobotController rc, boolean[] mask) {
        java.util.Arrays.fill(mask, false);
        mask[END] = true;
        int n = 1;
        if (rc.isBeingCarried() || rc.isBeingThrown()) return n;
        Direction facing = rc.getDirection();
        if (rc.isMovementReady()) for (int k = 0; k < 8; k++) if (rc.canMove(DIRS[k])) { mask[MOVE0 + k] = true; n++; }
        if (rc.isTurningReady()) for (int k = 0; k < 8; k++) if (DIRS[k] != facing && rc.canTurn(DIRS[k])) { mask[TURN0 + k] = true; n++; }
        boolean king = rc.getType() == UnitType.RAT_KING;
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

    /** Execute a legal action. Returns false for END. */
    public static boolean execute(RobotController rc, int a) throws GameActionException {
        if (a == END) return false;
        if (a < TURN0) { rc.move(DIRS[a - MOVE0]); return true; }
        if (a < TARGET0) { rc.turn(DIRS[a - TURN0]); return true; }
        if (a < THROW) {
            int t = a - TARGET0, type = t / TC, c = t % TC;
            MapLocation p = rc.getLocation().translate(c % TW - TR, c / TW - TR);
            switch (type) {
                case ATTACK: rc.attack(p); break;
                case CARRY: rc.carryRat(p); break;
                case BUILD: rc.buildRat(p); break;
                case RAT_TRAP: rc.placeRatTrap(p); break;
                case CAT_TRAP: rc.placeCatTrap(p); break;
                case PLACE_DIRT: rc.placeDirt(p); break;
                case REMOVE_DIRT: rc.removeDirt(p); break;
                case PICKUP: rc.pickUpCheese(p); break;
                case TRANSFER: rc.transferCheese(p, rc.getRawCheese()); break;
                case REMOVE_RAT_TRAP: rc.removeRatTrap(p); break;
                default: rc.removeCatTrap(p); break;
            }
            return true;
        }
        if (a == THROW) { rc.throwRat(); return true; }
        if (a < BECOME_KING) { rc.dropRat(DIRS[a - DROP0]); return true; }
        rc.becomeRatKing();
        return true;
    }

    /** Map a teacher controller call (made from the robot's current state) to an index, or -1. */
    public static int fromCall(RobotController rc, String op, Object[] args) {
        MapLocation me = rc.getLocation();
        int type;
        switch (op) {
            case "moveForward": { int k = dirIndex(rc.getDirection()); return k < 0 ? -1 : MOVE0 + k; }
            case "move": { int k = dirIndex((Direction) args[0]); return k < 0 ? -1 : MOVE0 + k; }
            case "turn": { int k = dirIndex((Direction) args[0]); return k < 0 ? -1 : TURN0 + k; }
            case "throwRat": return THROW;
            case "dropRat": { int k = dirIndex((Direction) args[0]); return k < 0 ? -1 : DROP0 + k; }
            case "becomeRatKing": return BECOME_KING;
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
        return target(type, dx, dy);
    }
}
