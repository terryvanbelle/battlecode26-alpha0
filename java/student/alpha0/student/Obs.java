package alpha0.student;

import battlecode.common.*;
import java.util.*;

/**
 * Student observation encoder (student-obs-v4). Uses only the public RobotController API plus per-robot memory of
 * what that robot itself sensed, and the obs-v2 team board. One Memory instance per robot ID.
 *
 * Layout: DENSE uint8 values (read as value/255): local window L*L*CL, coarse global G*G*CG, NS scalars,
 * SQ heard squeaks x SQM bytes of metadata; then BIT_BYTES of bits (LSB-first): full-resolution map memory
 * (MAP_MAX x MAP_MAX x MAP_PLANES, absolute coordinates), the raw shared array (64 x 10 bits), the robot's private
 * registers (32 x 10 bits) and the content of the heard squeaks (SQ x 32 bits). See docs/capability-principle.md.
 */
public final class Obs {
    public static final String VERSION = "student-obs-v4";
    public static final int R = 5, L = 2 * R + 1, CL = 25;
    public static final int G = 16, CG = 8;
    public static final int NS = 96, SQ = 16, SQM = 8;
    public static final int LOCAL = L * L * CL, GLOBAL = G * G * CG, SCALARS = LOCAL + GLOBAL, SQMETA = SCALARS + NS;
    public static final int DENSE = SQMETA + SQ * SQM;
    public static final int MAP_MAX = 60, MAP_PLANES = 8;
    public static final int B_MAP = 0, B_SHARED = MAP_MAX * MAP_MAX * MAP_PLANES, B_REG = B_SHARED + 64 * 10,
        B_SQ = B_REG + Actions.REGS * 10, BITS = B_SQ + SQ * 32, BIT_BYTES = (BITS + 7) / 8;
    public static final int SIZE = DENSE + BIT_BYTES;
    // map memory planes
    static final int P_SEEN = 0, P_WALL = 1, P_DIRT = 2, P_MINE = 3, P_CHEESE = 4, P_RAT_TRAP = 5, P_CAT_TRAP = 6, P_SELF = 7;

    // local channels
    static final int ON_MAP = 0, VISIBLE = 1, WALL = 2, DIRT = 3, CHEESE = 4, MINE = 5, OWN_RAT_TRAP = 6,
        OWN_CAT_TRAP = 7, OWN_BABY = 8, OWN_KING = 9, ENEMY_BABY = 10, ENEMY_KING = 11, CAT = 12,
        FACING_ME = 13, FLYING = 14, AGE = 15, ENEMY_MEM = 16,
        // obs-v4: the rest of RobotInfo
        DIR_X = 17, DIR_Y = 18, RAW_CHEESE = 19, CARRYING = 20, CHIRALITY = 21, ID_HI = 22, ID_LO = 23, HP_LO = 24;
    // global channels
    static final int G_WALL = 0, G_DIRT = 1, G_MINE = 2, G_OWN_KING = 3, G_ENEMY_KING = 4, G_ENEMY_GUESS = 5,
        G_SELF = 6, G_SEEN = 7;

    static final Direction[] DIRS = {Direction.NORTH, Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST,
        Direction.SOUTH, Direction.SOUTHWEST, Direction.WEST, Direction.NORTHWEST};

    /** What one robot remembers about the map. */
    public static final class Memory {
        final int w, h;
        final byte[] flags;      // bit0 seen, bit1 wall, bit2 dirt, bit3 mine, bit4 cheese, bit5 rat trap, bit6 cat trap
        final int[] seenRound;
        MapLocation home;        // own location on first observed turn (king within 2 tiles of spawn)
        final HashMap<Integer, int[]> ownKings = new HashMap<>();   // id -> x,y,round
        final HashMap<Integer, int[]> enemyKings = new HashMap<>();
        final HashMap<Integer, int[]> enemyRats = new HashMap<>();  // obs-v3: id -> x, y, round last seen (MEM_ROUNDS)
        int threatRound = -1000, hpPrev = -1, hpLoss, lastTurnRound = -1;
        int lastSqueakRound = -1;                                    // obs-v4: one squeak per turn (engine limit)
        final int[] reg = new int[Actions.REGS];                     // obs-v4: private registers (WRITE_REG)
        Memory(int w, int h) { this.w = w; this.h = h; flags = new byte[w * h]; seenRound = new int[w * h]; }
    }

    /**
     * Team board (obs-v2): a fixed protocol that fits the 64-slot shared array, which only kings write.
     * Each king publishes its centre and a heartbeat every turn; a king that sees enemies publishes an alert with
     * threat = enemy babies within THREAT_D2 of it (like zdb's Comms.ALERT); enemy-king sightings are published
     * (by kings directly, by babies through the squeak relay a tournament bot would use). Everyone reads it.
     */
    static final int THREAT_D2 = 18, DEFEND_D2 = 100, MEM_ROUNDS = 6;
    final HashMap<Integer, int[]> boardKings = new HashMap<>();   // id -> x, y, round
    int[] alert, eKing;                                           // x, y, round, threat / x, y, round

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
            TrapType tt = mi.getTrap();
            int f = 1 | (mi.isWall() ? 2 : 0) | (mi.isDirt() ? 4 : 0) | (mi.hasCheeseMine() ? 8 : 0)
                | (mi.getCheeseAmount() > 0 ? 16 : 0) | (tt == TrapType.RAT_TRAP ? 32 : 0) | (tt == TrapType.CAT_TRAP ? 64 : 0);
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
        // ---- obs-v3 short-term enemy memory (like zdb's start-of-turn `seen`, threatRound and blindHurt)
        if (round != m.lastTurnRound) {   // first encode of this turn
            int hp = rc.getHealth();
            m.hpLoss = m.hpPrev < 0 ? 0 : Math.max(0, m.hpPrev - hp);
            m.lastTurnRound = round;
        }
        m.hpPrev = rc.getHealth();
        for (RobotInfo r : robots)
            if (r.getTeam() != us && r.getType() == UnitType.BABY_RAT) {
                m.enemyRats.put(r.getID(), new int[]{r.getLocation().x, r.getLocation().y, round});
                m.threatRound = round;
            } else if (r.getTeam() != us && r.getType() == UnitType.RAT_KING) m.threatRound = round;
        m.enemyRats.entrySet().removeIf(e -> {
            int[] v = e.getValue();
            if (round - v[2] > MEM_ROUNDS) return true;
            if (seenIds.contains(e.getKey())) return false;
            return v[2] < round && rc.canSenseLocation(new MapLocation(v[0], v[1]));   // visible spot, no longer there
        });
        // ---- team board writes
        for (RobotInfo r : robots)
            if (r.getType() == UnitType.RAT_KING && r.getTeam() != us) eKing = new int[]{r.getLocation().x, r.getLocation().y, round};
        if (rc.getType() == UnitType.RAT_KING) {
            boardKings.put(rc.getID(), new int[]{me.x, me.y, round});
            int threat = 0, enemies = 0;
            for (RobotInfo r : robots) {
                if (r.getTeam() == us || r.getType() == UnitType.CAT) continue;
                enemies++;
                if (r.getType() != UnitType.RAT_KING && r.getLocation().distanceSquaredTo(me) <= THREAT_D2) threat++;
            }
            if (enemies > 0) alert = new int[]{me.x, me.y, round, Math.min(15, threat)};
        }
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
            out[base + CHEESE] = (byte) clamp(mi.getCheeseAmount());
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
                out[base + DIR_X] = (byte) (128 + 127 * d.dx);
                out[base + DIR_Y] = (byte) (128 + 127 * d.dy);
            }
            out[base + RAW_CHEESE] = (byte) clamp(r.getRawCheeseAmount());
            RobotInfo cr = r.getCarryingRobot();
            if (cr != null) out[base + CARRYING] = (byte) (cr.getTeam() == us ? 128 : 255);
            out[base + CHIRALITY] = (byte) clamp(128 + r.getChirality());
            out[base + ID_HI] = (byte) (r.getID() >> 8);
            out[base + ID_LO] = (byte) r.getID();
            out[base + HP_LO] = (byte) r.getHealth();
        }

        for (int[] v : m.enemyRats.values()) {
            int dx = v[0] - me.x, dy = v[1] - me.y;
            if (dx < -R || dx > R || dy < -R || dy > R) continue;
            int o = ((dy + R) * L + (dx + R)) * CL + ENEMY_MEM;
            out[o] = (byte) Math.max(out[o] & 255, 255 - (round - v[2]) * 32);
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
        for (int[] v : boardKings.values()) if (round - v[2] <= 2) out[off + bin(m, v[0], v[1]) * CG + G_OWN_KING] = (byte) 255;
        for (int[] v : m.enemyKings.values()) out[off + bin(m, v[0], v[1]) * CG + G_ENEMY_KING] = (byte) clamp(255 - (round - v[2]) / 4);
        if (eKing != null) { int o = off + bin(m, eKing[0], eKing[1]) * CG + G_ENEMY_KING; out[o] = (byte) Math.max(out[o] & 255, clamp(255 - (round - eKing[2]) / 4)); }
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

        // ---- team board reads (relative vectors; dx, dy as 128 + 4 * tiles)
        int[] near = null; int nd = Integer.MAX_VALUE, live = 0;
        for (int[] v : boardKings.values()) {
            if (round - v[2] > 2 || (king && v == boardKings.get(rc.getID()))) continue;
            live++;
            int d2 = (v[0] - me.x) * (v[0] - me.x) + (v[1] - me.y) * (v[1] - me.y);
            if (d2 < nd) { nd = d2; near = v; }
        }
        if (near != null) {
            out[s + 32] = (byte) 255;
            out[s + 33] = rel(near[0] - me.x); out[s + 34] = rel(near[1] - me.y);
            out[s + 35] = (byte) clamp((int) (Math.sqrt(nd) * 4));
        }
        out[s + 36] = (byte) clamp(live * 50);
        if (alert != null && round - alert[2] <= 3) {
            int d2 = (alert[0] - me.x) * (alert[0] - me.x) + (alert[1] - me.y) * (alert[1] - me.y);
            out[s + 37] = (byte) (round - alert[2] <= 2 && alert[3] >= 1 ? 255 : 0);
            out[s + 38] = (byte) clamp(alert[3] * 16);
            out[s + 39] = rel(alert[0] - me.x); out[s + 40] = rel(alert[1] - me.y);
            out[s + 41] = (byte) clamp(255 - (round - alert[2]) * 32);
            out[s + 45] = (byte) clamp((int) (Math.sqrt(d2) * 4));
            out[s + 47] = (byte) (out[s + 37] != 0 && d2 <= DEFEND_D2 ? 255 : 0);
        }
        if (eKing != null && round - eKing[2] <= 32) {
            out[s + 42] = (byte) clamp(255 - (round - eKing[2]) * 8);
            out[s + 43] = rel(eKing[0] - me.x); out[s + 44] = rel(eKing[1] - me.y);
            out[s + 46] = (byte) clamp((int) (Math.sqrt((eKing[0] - me.x) * (eKing[0] - me.x) + (eKing[1] - me.y) * (eKing[1] - me.y)) * 4));
        }
        out[s + 48] = (byte) clamp(255 - (round - m.threatRound) * 16);   // recency of any enemy rat sighting
        out[s + 49] = (byte) clamp(m.hpLoss * 4);                          // health lost since our last turn
        int remembered = 0;
        for (int[] v : m.enemyRats.values()) if (v[2] < round || !rc.canSenseLocation(new MapLocation(v[0], v[1]))) remembered++;
        out[s + 50] = (byte) clamp(remembered * 32);                        // remembered enemies not in view now

        // ---- obs-v4: exact getters (every value a bot can read about itself)
        int id = rc.getID();
        out[s + 52] = (byte) (id >> 8); out[s + 53] = (byte) id;
        out[s + 54] = (byte) clamp(rc.getTurningCooldownTurns() * 4);
        hilo(out, s + 55, rc.getAllCheese());
        hilo(out, s + 57, rc.getHealth());
        hilo(out, s + 59, rc.getRawCheese());
        hilo(out, s + 61, rc.getGlobalCheese());
        out[s + 63] = (byte) me.x; out[s + 64] = (byte) me.y;
        hilo(out, s + 65, round);
        out[s + 67] = (byte) m.w; out[s + 68] = (byte) m.h;
        RobotInfo carried = rc.getCarrying();
        if (carried != null) {
            out[s + 69] = (byte) (carried.getTeam() == us ? 128 : 255);
            out[s + 70] = (byte) (carried.getType() == UnitType.RAT_KING ? 255 : 128);
            hilo(out, s + 71, carried.getHealth());
            out[s + 73] = (byte) (carried.getID() >> 8); out[s + 74] = (byte) carried.getID();
            out[s + 75] = (byte) clamp(carried.getRawCheeseAmount());
        }
        out[s + 76] = (byte) (rc.isBeingThrown() ? 255 : 0);
        out[s + 77] = (byte) (m.lastSqueakRound == round ? 255 : 0);
        out[s + 79] = (byte) clamp(rc.getActionCooldownTurns() / 4);
        out[s + 80] = (byte) clamp(rc.getMovementCooldownTurns() / 4);
        out[s + 81] = (byte) clamp(rc.getCurrentRatCost() / 4);
        out[s + 84] = (byte) clamp(rc.getDirt());
        out[s + 85] = (byte) clamp(rc.getNumberRatTraps());
        out[s + 86] = (byte) clamp(rc.getNumberCatTraps());
        out[s + 87] = (byte) clamp(step * 2);

        // ---- obs-v4: heard squeaks (newest SQ of the last 5 rounds): metadata bytes + content bits
        Message[] msgs = rc.readSqueaks(-1);
        out[s + 78] = (byte) clamp(msgs.length * 16);
        int bitBase = DENSE * 8;
        for (int k = 0; k < SQ && k < msgs.length; k++) {
            Message msg = msgs[msgs.length - 1 - k];
            int o = SQMETA + k * SQM;
            MapLocation src = msg.getSource();
            out[o] = (byte) 255;
            out[o + 1] = rel(src.x - me.x); out[o + 2] = rel(src.y - me.y);
            out[o + 3] = (byte) clamp((round - msg.getRound()) * 50);
            out[o + 4] = (byte) (msg.getSenderID() >> 8); out[o + 5] = (byte) msg.getSenderID();
            out[o + 6] = (byte) src.x; out[o + 7] = (byte) src.y;
            int bytes = msg.getBytes();
            for (int b = 0; b < 32; b++) if ((bytes >>> b & 1) != 0) setBit(out, bitBase + B_SQ + k * 32 + b);
        }

        // ---- obs-v4: raw shared array and private registers
        for (int i = 0; i < 64; i++) {
            int v;
            try { v = rc.readSharedArray(i); } catch (GameActionException e) { v = 0; }
            for (int b = 0; b < 10; b++) if ((v >> b & 1) != 0) setBit(out, bitBase + B_SHARED + i * 10 + b);
        }
        for (int i = 0; i < Actions.REGS; i++) {
            int v = m.reg[i];
            for (int b = 0; b < 10; b++) if ((v >> b & 1) != 0) setBit(out, bitBase + B_REG + i * 10 + b);
        }

        // ---- obs-v4: full-resolution map memory, absolute coordinates
        for (int y = 0; y < m.h && y < MAP_MAX; y++) for (int x = 0; x < m.w && x < MAP_MAX; x++) {
            int f = m.flags[y * m.w + x];
            if (f == 0) continue;
            int t = bitBase + B_MAP + (y * MAP_MAX + x) * MAP_PLANES;
            for (int pl = 0; pl < 7; pl++) if ((f >> pl & 1) != 0) setBit(out, t + pl);
        }
        if (me.x < MAP_MAX && me.y < MAP_MAX) setBit(out, bitBase + B_MAP + (me.y * MAP_MAX + me.x) * MAP_PLANES + P_SELF);

    }

    private static byte rel(int d) { return (byte) clamp(128 + 4 * d); }

    private static void hilo(byte[] out, int o, int v) {
        v = Math.max(0, Math.min(65535, v));
        out[o] = (byte) (v >> 8); out[o + 1] = (byte) v;
    }

    private static void setBit(byte[] out, int bit) { out[bit >> 3] |= (byte) (1 << (bit & 7)); }

    private static int bin(Memory m, int x, int y) {
        x = Math.max(0, Math.min(m.w - 1, x)); y = Math.max(0, Math.min(m.h - 1, y));
        return (y * G / m.h) * G + x * G / m.w;
    }
}
