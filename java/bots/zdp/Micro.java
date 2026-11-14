package zdp;

import battlecode.common.*;

/**
 * Baby-rat combat movement once cooperation is over. Every tile we can reach this turn (stay or one of 8 steps) is
 * scored by what happens next turn: enemies that can reach and bite us there (within Chebyshev 2 of the tile),
 * enemies we can bite from there, and how many allies stand with us. We advance when the local fight is ours and
 * back off (strafing, so the enemy stays in our cone and cannot grab us) when it is not.
 */
public final class Micro {
    static final int[] eX = new int[24], eY = new int[24], eHp = new int[24];
    static int nE;

    /** Returns true when micro took the movement decision this turn. */
    public static boolean act(RobotInfo[] robots) throws GameActionException {
        RobotController rc = G.rc;
        if (!rc.isMovementReady()) return false;
        MapLocation me = rc.getLocation();
        nE = 0;
        int allies = 1, allyHp = rc.getHealth(), enemyHp = 0;
        MapLocation closest = null;
        int cd = Integer.MAX_VALUE;
        for (RobotInfo r : robots) {
            if (r.type != UnitType.BABY_RAT) continue;
            int d = me.distanceSquaredTo(r.location);
            if (r.team == G.them) {
                if (d > 20 || nE >= eX.length) continue;
                eX[nE] = r.location.x; eY[nE] = r.location.y; eHp[nE] = r.health; nE++;
                enemyHp += r.health;
                if (d < cd) { cd = d; closest = r.location; }
            } else if (r.team == G.us && d <= 13) {
                allies++;
                allyHp += r.health;
            }
        }
        if (nE == 0) return false;
        // the fight is ours when we bring more health to it (ties go to the defender near its king)
        boolean ours = allyHp > enemyHp || allyHp == enemyHp && Comms.nearestKing(me) != null
                && Comms.nearestKing(me).distanceSquaredTo(me) < closest.distanceSquaredTo(Comms.nearestKing(me));
        boolean canBite = rc.isActionReady();
        int hp = rc.getHealth();

        Direction best = Direction.CENTER;
        int bs = score(me, ours, canBite, hp) + 1; // small bias to staying (no cooldown spent)
        for (int i = 0; i < 8; i++) {
            Direction d = G.DIRS[i];
            if (!rc.canMove(d) || (Nav.avoid & (1 << i)) != 0) continue;
            MapLocation t = me.add(d);
            // pickup is free: a step onto cheese mid-fight takes it from them
            int c = rc.canSenseLocation(t) && rc.senseMapInfo(t).getCheeseAmount() > 0 ? Cfg.MICRO_CHEESE : 0;
            int s = (score(t, ours, canBite, hp) + c) * 2 + (G.hand(me, d) > 0 ? 1 : 0);
            if (s > bs * 2) { bs = s / 2; best = d; }
        }
        Direction toEnemy = me.directionTo(closest);
        if (best == Direction.CENTER) {
            if (rc.getDirection() != toEnemy && rc.canTurn(toEnemy)) { rc.turn(toEnemy); G.facing = toEnemy; }
            return true;
        }
        MapLocation t = me.add(best);
        Direction face = t.directionTo(closest);
        boolean advancing = t.distanceSquaredTo(closest) < me.distanceSquaredTo(closest);
        if (advancing) {
            // forward step toward them (cheap), ending facing the target as long as it is roughly ahead
            if (rc.getDirection() != best && rc.canTurn(best)) { rc.turn(best); G.facing = best; }
        } else if (rc.getDirection() != face && rc.canTurn(face)) {
            // backing off: turn to face them first, then strafe away
            rc.turn(face);
            G.facing = face;
        }
        if (rc.canMove(best)) {
            rc.move(best);
            G.me = rc.getLocation();
        }
        return true;
    }

    /** Next-turn value of standing on t. */
    static int score(MapLocation t, boolean ours, boolean canBite, int hp) {
        int threats = 0, targets = 0, weakest = 1000;
        for (int i = 0; i < nE; i++) {
            int dx = Math.abs(eX[i] - t.x), dy = Math.abs(eY[i] - t.y);
            int cheb = Math.max(dx, dy);
            if (cheb <= 2) threats++;
            if (dx * dx + dy * dy <= 2) { targets++; if (eHp[i] < weakest) weakest = eHp[i]; }
        }
        int s = 0;
        if (ours) {
            s += targets > 0 ? (canBite ? 40 : 15) + (100 - weakest) / 5 : 0;
            s -= 6 * Math.max(0, threats - 1);
            // stay in contact: closer to the nearest enemy is better
            int nd = Integer.MAX_VALUE;
            for (int i = 0; i < nE; i++) nd = Math.min(nd, Math.max(Math.abs(eX[i] - t.x), Math.abs(eY[i] - t.y)));
            s -= nd * 3;
        } else {
            s -= 20 * threats;
            s += targets > 0 && canBite && hp > 50 ? 10 : 0;
        }
        return s;
    }
}
