package alpha0.student;

import battlecode.common.*;
import java.util.*;

/**
 * Student observation encoder (student-obs-v1). Uses only the public RobotController
 * API plus per-robot memory of what that robot itself sensed. No teacher state, no
 * engine internals, no communication. One Memory instance per robot ID.
 *
 * Layout (uint8 each): local window L*L*CL, then coarse global G*G*CG, then NS scalars.
 */
public final class Obs {
    public static final String VERSION = "student-obs-v1";
    public static final int R = 5, L = 2 * R + 1, CL = 16;
    public static final int G = 16, CG = 8;
    public static final int NS = 32;
    public static final int LOCAL = L * L * CL, GLOBAL = G * G * CG;
    public static final int SIZE = LOCAL + GLOBAL + NS;

    // local channels
    static final int ON_MAP = 0, VISIBLE = 1, WALL = 2, DIRT = 3, CHEESE = 4, MINE = 5, OWN_RAT_TRAP = 6,
        OWN_CAT_TRAP = 7, OWN_BABY = 8, OWN_KING = 9, ENEMY_BABY = 10, ENEMY_KING = 11, CAT = 12,
        FACING_ME = 13, FLYING = 14, AGE = 15;
    // global channels
    static final int G_WALL = 0, G_DIRT = 1, G_MINE = 2, G_OWN_KING = 3, G_ENEMY_KING = 4, G_ENEMY_GUESS = 5,
        G_SELF = 6, G_SEEN = 7;

    static final Direction[] DIRS = {Direction.NORTH, Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST,
        Direction.SOUTH, Direction.SOUTHWEST, Direction.WEST, Direction.NORTHWEST};

    /** What one robot remembers about the map. */
    public static final class Memory {
        final int w, h;
        final byte[] flags;      // bit0 seen, bit1 wall, bit2 dirt, bit3 mine
        final int[] seenRound;
        MapLocation home;        // own location on first observed turn (king within 2 tiles of spawn)
        final HashMap<Integer, int[]> ownKings = new HashMap<>();   // id -> x,y,round
        final HashMap<Integer, int[]> enemyKings = new HashMap<>();
        Memory(int w, int h) { this.w = w; this.h = h; flags = new byte[w * h]; seenRound = new int[w * h]; }
    }

    private final HashMap<Integer, Memory> memories = new HashMap<>();

    public Memory memory(RobotController rc) {
        return memories.computeIfAbsent(rc.getID(), k -> new Memory(rc.getMapWidth(), rc.getMapHeight()));
    }

    public void forget(int id) { memories.remove(id); }

    private static int clamp(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

    /**
     * Sense, update memory, and encode into out[0..SIZE). step = primitives already executed this turn.
     */
    public void encode(RobotController rc, int step, byte[] out) {
        Arrays.fill(out, 0, SIZE, (byte) 0);
        Memory m = memory(rc);
        int round = rc.getRoundNum();
        MapLocation me = rc.getLocation();
        Team us = rc.getTeam();
        if (m.home == null) m.home = me;

        // ---- sense & update memory
        MapInfo[] infos = rc.senseNearbyMapInfos();
        RobotInfo[] robots = rc.senseNearbyRobots();
        for (MapInfo mi : infos) {
            MapLocation p = mi.getMapLocation();
            int i = p.y * m.w + p.x;
            int f = 1 | (mi.isWall() ? 2 : 0) | (mi.isDirt() ? 4 : 0) | (mi.hasCheeseMine() ? 8 : 0);
            m.flags[i] = (byte) f;
            m.seenRound[i] = round;
        }
        // forget kings whose remembered position is visible but no longer holds them
        HashSet<Integer> seenIds = new HashSet<>();
        for (RobotInfo r : robots) {
            seenIds.add(r.getID());
            if (r.getType() == UnitType.RAT_KING) {
                int[] v = {r.getLocation().x, r.getLocation().y, round};
                (r.getTeam() == us ? m.ownKings : m.enemyKings).put(r.getID(), v);
            }
        }
        if (rc.getType() == UnitType.RAT_KING) m.ownKings.put(rc.getID(), new int[]{me.x, me.y, round});
        for (HashMap<Integer, int[]> kings : List.of(m.ownKings, m.enemyKings)) {
            kings.entrySet().removeIf(e -> {
                if (seenIds.contains(e.getKey()) || e.getKey() == rc.getID()) return false;
                MapLocation p = new MapLocation(e.getValue()[0], e.getValue()[1]);
                return rc.canSenseLocation(p);   // should have seen it there
            });
        }

        // ---- local window
        int base;
        for (int dy = -R; dy <= R; dy++) for (int dx = -R; dx <= R; dx++) {
            int x = me.x + dx, y = me.y + dy;
            base = ((dy + R) * L + (dx + R)) * CL;
            if (x < 0 || y < 0 || x >= m.w || y >= m.h) continue;
            out[base + ON_MAP] = (byte) 255;
            int i = y * m.w + x;
            int f = m.flags[i];
            if ((f & 1) != 0) {
                if ((f & 2) != 0) out[base + WALL] = (byte) 255;
                if ((f & 4) != 0) out[base + DIRT] = (byte) 255;
                if ((f & 8) != 0) out[base + MINE] = (byte) 255;
                out[base + AGE] = (byte) clamp(round - m.seenRound[i]);
            } else out[base + AGE] = (byte) 255;
        }
        for (MapInfo mi : infos) {
            MapLocation p = mi.getMapLocation();
            int dx = p.x - me.x, dy = p.y - me.y;
            if (dx < -R || dx > R || dy < -R || dy > R) continue;
            base = ((dy + R) * L + (dx + R)) * CL;
            out[base + VISIBLE] = (byte) 255;
            out[base + CHEESE] = (byte) clamp(mi.getCheeseAmount() * 4);
            TrapType t = mi.getTrap();
            if (t == TrapType.RAT_TRAP) out[base + OWN_RAT_TRAP] = (byte) 255;
            else if (t == TrapType.CAT_TRAP) out[base + OWN_CAT_TRAP] = (byte) 255;
            if (mi.flyingRobot() != null) out[base + FLYING] = (byte) 255;
            RobotInfo r = null;
            try { if (rc.canSenseRobotAtLocation(p)) r = rc.senseRobotAtLocation(p); } catch (GameActionException e) { r = null; }
            if (r == null || r.getID() == rc.getID()) continue;
            int ch, hp;
            UnitType ut = r.getType();
            if (ut == UnitType.CAT) { ch = CAT; hp = clamp(55 + r.getHealth() * 200 / 4000); }
            else if (ut == UnitType.RAT_KING) { ch = r.getTeam() == us ? OWN_KING : ENEMY_KING; hp = clamp(55 + r.getHealth() * 200 / 600); }
            else { ch = r.getTeam() == us ? OWN_BABY : ENEMY_BABY; hp = clamp(55 + r.getHealth() * 2); }
            out[base + ch] = (byte) hp;
            Direction d = r.getDirection();
            if (d != null && d != Direction.CENTER) {
                int vx = me.x - r.getLocation().x, vy = me.y - r.getLocation().y;
                if (vx * d.dx + vy * d.dy > 0) out[base + FACING_ME] = (byte) 255;
            }
        }

        // ---- coarse global map from memory
        int[] cnt = new int[G * G], wall = new int[G * G], dirt = new int[G * G], mine = new int[G * G], seen = new int[G * G];
        for (int y = 0; y < m.h; y++) {
            int by = y * G / m.h;
            for (int x = 0; x < m.w; x++) {
                int b = by * G + x * G / m.w, f = m.flags[y * m.w + x];
                cnt[b]++;
                if ((f & 1) != 0) { seen[b]++; if ((f & 2) != 0) wall[b]++; if ((f & 4) != 0) dirt[b]++; if ((f & 8) != 0) mine[b]++; }
            }
        }
        int off = LOCAL;
        for (int b = 0; b < G * G; b++) {
            if (cnt[b] == 0) continue;
            int o = off + b * CG;
            out[o + G_WALL] = (byte) clamp(wall[b] * 255 / cnt[b]);
            out[o + G_DIRT] = (byte) clamp(dirt[b] * 255 / cnt[b]);
            out[o + G_MINE] = (byte) (mine[b] > 0 ? 255 : 0);
            out[o + G_SEEN] = (byte) clamp(seen[b] * 255 / cnt[b]);
        }
        for (int[] v : m.ownKings.values()) out[off + bin(m, v[0], v[1]) * CG + G_OWN_KING] = (byte) clamp(255 - (round - v[2]));
        for (int[] v : m.enemyKings.values()) out[off + bin(m, v[0], v[1]) * CG + G_ENEMY_KING] = (byte) clamp(255 - (round - v[2]) / 4);
        int hx = m.home.x, hy = m.home.y;
        out[off + bin(m, m.w - 1 - hx, m.h - 1 - hy) * CG + G_ENEMY_GUESS] = (byte) 255;
        out[off + bin(m, m.w - 1 - hx, hy) * CG + G_ENEMY_GUESS] = (byte) 255;
        out[off + bin(m, hx, m.h - 1 - hy) * CG + G_ENEMY_GUESS] = (byte) 255;
        out[off + bin(m, me.x, me.y) * CG + G_SELF] = (byte) 255;

        // ---- scalars
        int s = LOCAL + GLOBAL;
        boolean king = rc.getType() == UnitType.RAT_KING;
        out[s] = (byte) (king ? 0 : 255);
        out[s + 1] = (byte) (king ? 255 : 0);
        out[s + 2] = (byte) clamp(rc.getHealth() * 255 / (king ? 600 : 100));
        out[s + 3] = (byte) clamp(rc.getRawCheese() * 2);
        out[s + 4] = (byte) clamp((int) (Math.log1p(rc.getGlobalCheese()) * 28));
        out[s + 5] = (byte) clamp(rc.getDirt() * 8);
        out[s + 6] = (byte) clamp(round * 255 / 2000);
        out[s + 7] = (byte) (rc.isActionReady() ? 255 : 0);
        out[s + 8] = (byte) (rc.isMovementReady() ? 255 : 0);
        out[s + 9] = (byte) (rc.isTurningReady() ? 255 : 0);
        out[s + 10] = (byte) clamp(rc.getActionCooldownTurns() * 4);
        out[s + 11] = (byte) clamp(rc.getMovementCooldownTurns() * 4);
        Direction facing = rc.getDirection();
        for (int k = 0; k < 8; k++) if (DIRS[k] == facing) out[s + 12 + k] = (byte) 255;
        out[s + 20] = (byte) (rc.getCarrying() != null ? 255 : 0);
        out[s + 21] = (byte) (rc.isBeingCarried() ? 255 : 0);
        out[s + 22] = (byte) (rc.isCooperation() ? 255 : 0);
        Team bs = rc.getBackstabbingTeam();
        out[s + 23] = (byte) (bs == us ? 255 : 0);
        out[s + 24] = (byte) (bs != null && bs != us ? 255 : 0);
        out[s + 25] = (byte) clamp(rc.getNumberRatTraps() * 10);
        out[s + 26] = (byte) clamp(rc.getNumberCatTraps() * 25);
        out[s + 27] = (byte) clamp(m.w * 4);
        out[s + 28] = (byte) clamp(m.h * 4);
        out[s + 29] = (byte) clamp(step * 32);
        out[s + 30] = (byte) clamp(rc.getCurrentRatCost() * 2);
        out[s + 31] = (byte) (rc.canBecomeRatKing() ? 255 : 0);
    }

    private static int bin(Memory m, int x, int y) {
        x = Math.max(0, Math.min(m.w - 1, x)); y = Math.max(0, Math.min(m.h - 1, y));
        return (y * G / m.h) * G + x * G / m.w;
    }
}
