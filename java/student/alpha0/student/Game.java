package alpha0.student;

import battlecode.common.*;
import battlecode.crossplay.CrossPlayLanguage;
import battlecode.server.*;
import battlecode.world.*;
import battlecode.world.control.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.*;
import java.lang.reflect.Field;
import java.util.*;

/**
 * Runs one full game. Each team is either a sandboxed Java bot ("bot:PACKAGE:CLASSES_DIR")
 * or the student network ("net:WEIGHTS[:TEMPERATURE]"). Optionally records the teacher's
 * (bot team's) decisions, or the student's own decisions, as Records. On net teams --record-p/--record-p-king
 * sample whole turns (every step of a sampled turn is recorded) from a separate RNG, so recording never changes play.
 *
 * java ... alpha0.student.Game --map M --maps-dir D --seed S --a SPEC --b SPEC --out DIR
 *      [--record AB] [--record-p 0.1] [--record-p-king P (default: record-p)] [--max-rounds 2000]
 *      [--trace N]   write trace.tsv (babies, kings, cheese, transferred per team) every N rounds
 *      [--record-p-boost P]   babies that can become king or see an enemy are sampled with P (default: record-p)
 *      [--mix-net WEIGHTS --mix-p 0.5 [--mix-t 0]]   DAgger: on recorded bot teams, each robot turn is played by the
 *      student with probability mix-p (not recorded); the bot plays (and is sampled for recording) otherwise.
 *      [--decisions AB]   record bot teams' decision records (setIndicatorString channel, see Decisions) to
 *      decisions-A.bin.gz / decisions-B.bin.gz (every emit; net teams emit nothing)
 *      [--param-t T]   temperature for parameter bits (default 0 = most likely bits: values must be exact)
 *      [--end-actions 1]   let student teams choose disintegrate (off by default: it is in the action
 *      space, but a sampled imitation policy would occasionally pick them by accident).
 */
public final class Game {
    static final ObjectMapper JSON = new ObjectMapper();
    static boolean traceDeaths, endActions;
    static double paramT = 0;   // --param-t: temperature for parameter bits (default 0: most likely bits; values must be exact)
    static long trajHash = 17;   // per-round fingerprint of both teams' stats

    static final class Args {
        final Map<String, String> v = new HashMap<>();
        Args(String[] a) { for (int i = 0; i + 1 < a.length; i += 2) v.put(a[i].replaceFirst("^--", ""), a[i + 1]); }
        String get(String k, String d) { return v.getOrDefault(k, d); }
        String need(String k) { String s = v.get(k); if (s == null) throw new IllegalArgumentException("missing --" + k); return s; }
    }

    /** Bounded tail of bot console output. */
    static final class Tail extends OutputStream {
        final byte[] buf = new byte[16384]; int n;
        @Override public synchronized void write(int b) { if (n == buf.length) { System.arraycopy(buf, 1, buf, 0, n - 1); n--; } buf[n++] = (byte) b; }
        @Override public synchronized String toString() { return new String(buf, 0, n); }
    }

    /** Sandboxed teacher/opponent bot, optionally recording sampled turns. */
    static final class Bot extends PlayerControlProvider {
        final Obs obs = new Obs();
        final Records rec; final double p, pKing; double pBoost; final Random rng; final int teamIndex;
        final byte[] scratch = new byte[Obs.SIZE]; final boolean[] mask = new boolean[Actions.N];
        final List<String> failures = new ArrayList<>();
        GameWorld world;
        long turns, sampledTurns, ends, studentTurns, studentPrimitives;
        Net mixNet; double mixP, mixT; Random mixRng;
        Bot(Team team, String pkg, String classes, OutputStream out, Records rec, double p, double pKing, long seed) {
            super(team, pkg, CrossPlayLanguage.JAVA, null, classes, out, false);
            this.rec = rec; this.p = p; this.pKing = pKing; this.pBoost = p; this.rng = new Random(seed); this.teamIndex = team.ordinal();
        }
        @Override public void matchStarted(GameWorld w) { world = w; super.matchStarted(w); }
        /** Rare, decisive baby states (state-based, so the recorded P(action | state) stays unbiased). */
        boolean boosted(RobotController rc) {
            if (pBoost == p) return false;
            int[] al = obs.alert;   // obs-v2 board: a fresh threatened-king alert within defend range
            if (al != null && al[3] >= 1 && rc.getRoundNum() - al[2] <= 2
                && rc.getLocation().distanceSquaredTo(new MapLocation(al[0], al[1])) <= Obs.DEFEND_D2) return true;
            try { return rc.canBecomeRatKing() || rc.senseNearbyRobots(-1, rc.getTeam().opponent()).length > 0; }
            catch (GameActionException | RuntimeException e) { return false; }
        }
        @Override public void robotSpawned(InternalRobot robot) {
            super.robotSpawned(robot);
            if (getTerminated(robot)) failures.add("init_failed:" + robot.getID());
        }
        @Override public void robotKilled(InternalRobot robot) { super.robotKilled(robot); obs.forget(robot.getID()); }
        @Override public void runRobot(InternalRobot robot) {
            turns++;
            RobotController rc = robot.getController();
            if (mixNet != null && rec != null && world.getObjectInfo().getRobotByID(robot.getID()) != null && mixRng.nextDouble() < mixP) {
                studentTurns++;
                studentPrimitives += studentTurn(world, robot, obs, mixNet, mixT, mixRng, scratch, mask, null, teamIndex)[1];
                return;   // the bot's own player does not run this turn (like a skipped turn)
            }
            boolean sample = rec != null && world.getObjectInfo().getRobotByID(robot.getID()) != null && rng.nextDouble() < (rc.getType() == UnitType.RAT_KING ? pKing : boosted(rc) ? pBoost : p);
            if (rec != null && world.getObjectInfo().getRobotByID(robot.getID()) != null) {
                try { obs.encode(rc, 0, scratch); } catch (RuntimeException e) { /* memory update only */ }
            }
            if (sample) Hook.start(robot.getID(), teamIndex, obs, rec);
            try { super.runRobot(robot); }
            finally {
                if (sample) {
                    int step = Hook.stop();
                    sampledTurns++;
                    boolean alive = world.getObjectInfo().getRobotByID(robot.getID()) != null && !getTerminated(robot);
                    boolean exhausted = getBytecodesUsed(robot) >= robot.getType().bytecodeLimit - 50;
                    if (alive && !exhausted) {
                        try {
                            obs.encode(rc, step, scratch);
                            Actions.legal(rc, obs.memory(rc), mask, true);
                            rec.write(robot.getID(), world.getCurrentRound(), Actions.END, 0, teamIndex,
                                      rc.getType() == UnitType.RAT_KING, step, false, scratch, mask);
                            ends++;
                        } catch (Exception e) { if (Hook.error == null) Hook.error = e; }
                    }
                }
            }
            if (getTerminated(robot) && world.getObjectInfo().getRobotByID(robot.getID()) != null
                && !terminatedByRobotDeath(robot)) failures.add("sandbox_exit:" + robot.getID() + "@" + world.getCurrentRound());
        }
    }

    /** The student network controls this team directly through RobotControllerImpl. */
    static final class Student implements RobotControlProvider {
        final Obs obs = new Obs(); final Net net; final double temperature; final Random rng;
        final Records rec; final int teamIndex; final double p, pKing; final Random recRng;
        final byte[] buf = new byte[Obs.SIZE]; final boolean[] mask = new boolean[Actions.N];
        GameWorld world;
        long decisions, primitives, errors, sampledTurns;
        Student(Team team, Net net, double temperature, long seed, Records rec, double p, double pKing) {
            this.net = net; this.temperature = temperature; this.rng = new Random(seed); this.rec = rec; this.teamIndex = team.ordinal();
            this.p = p; this.pKing = pKing; this.recRng = new Random(seed + 31337);   // separate stream: sampling never changes play
        }
        @Override public void matchStarted(GameWorld w) { world = w; }
        @Override public void matchEnded() {}
        @Override public void roundStarted() {}
        @Override public void roundEnded() {}
        @Override public void robotSpawned(InternalRobot robot) {}
        final Map<Integer, Integer> lastHp = new HashMap<>();
        @Override public void robotKilled(InternalRobot robot) {
            obs.forget(robot.getID());
            if (traceDeaths) System.err.println("DEATH team=" + robot.getTeam() + " round=" + world.getCurrentRound() + " id=" + robot.getID()
                + " type=" + robot.getType() + " hpAtLastTurn=" + lastHp.get(robot.getID()) + " hp=" + robot.getHealth()
                + " thrown=" + robot.isBeingThrown() + " sinceThrown=" + robot.getTurnsSinceThrownOrDropped()
                + " loc=" + robot.getDiedLocation());
        }
        @Override public int getBytecodesUsed(InternalRobot robot) { return 0; }
        @Override public boolean getTerminated(InternalRobot robot) { return false; }
        @Override public void runRobot(InternalRobot robot) {
            lastHp.put(robot.getID(), robot.getHealth());
            boolean sample = rec != null && recRng.nextDouble() < (robot.getType() == UnitType.RAT_KING ? pKing : p);
            if (sample) sampledTurns++;
            long[] r = studentTurn(world, robot, obs, net, temperature, rng, buf, mask, sample ? rec : null, teamIndex);
            decisions += r[0]; primitives += r[1]; errors += r[2];
        }
    }

    /** One student-controlled turn: observe, pick, execute until END. Returns {decisions, primitives, errors}. */
    static long[] studentTurn(GameWorld world, InternalRobot robot, Obs obs, Net net, double temperature, Random rng,
                              byte[] buf, boolean[] mask, Records rec, int teamIndex) {
        long[] r = new long[3];
        if (world.getObjectInfo().getRobotByID(robot.getID()) == null) return r;
        RobotController rc = robot.getController();
        for (int step = 0; step < Actions.MAX_STEPS; step++) {
            if (world.getObjectInfo().getRobotByID(robot.getID()) == null || !world.isRunning()) return r;
            obs.encode(rc, step, buf);
            Obs.Memory m = obs.memory(rc);
            int n = Actions.legal(rc, m, mask, endActions);
            if (n <= 1) return r;
            net.forward(buf);
            int a = net.act(mask, temperature, rng);
            int param = net.param(a, paramT, rng);
            r[0]++;
            if (rec != null) {
                try { rec.write(robot.getID(), world.getCurrentRound(), a, param, teamIndex, rc.getType() == UnitType.RAT_KING, step, false, buf, mask); }
                catch (IOException e) { throw new UncheckedIOException(e); }
            }
            if (a == Actions.END) return r;
            try { Actions.execute(rc, m, a, param); r[1]++; }
            catch (GameActionException e) { r[2]++; return r; }
            catch (battlecode.instrumenter.RobotDeathException e) { world.destroyRobot(robot.getID()); r[1]++; return r; }
        }
        return r;
    }

    public static void main(String[] argv) throws Exception {
        Args args = new Args(argv);
        long t0 = System.nanoTime();
        File out = new File(args.need("out"));
        out.mkdirs();
        Config.getGlobalConfig().setBoolean("bc.server.robot-player-to-system-out", false);
        String mapName = args.need("map");
        int seed = Integer.parseInt(args.get("seed", "0"));
        LiveMap map = GameMapIO.loadMap(mapName, new File(args.get("maps-dir", "maps")), false);
        Field f = LiveMap.class.getDeclaredField("seed"); f.setAccessible(true); f.setInt(map, seed);
        InternalRobot.rand = new Random(1092L + 7919L * seed);
        String recordTeams = args.get("record", "");
        String decisionTeams = args.get("decisions", "");
        for (Team team : new Team[]{Team.A, Team.B})
            if (decisionTeams.contains(team.name())) Decisions.open(team.ordinal(), new File(out, "decisions-" + team.name() + ".bin.gz"));
        double recordP = Double.parseDouble(args.get("record-p", "0.1"));
        double recordPKing = Double.parseDouble(args.get("record-p-king", Double.toString(recordP)));
        Net mixNet = args.v.containsKey("mix-net") ? new Net(new File(args.need("mix-net"))) : null;
        double recordPBoost = Double.parseDouble(args.get("record-p-boost", Double.toString(recordP)));
        double mixP = Double.parseDouble(args.get("mix-p", "0.5")), mixT = Double.parseDouble(args.get("mix-t", "0"));
        int maxRounds = Integer.parseInt(args.get("max-rounds", "2000"));
        endActions = args.get("end-actions", "0").equals("1");
        if (args.get("param-t", null) != null) paramT = Double.parseDouble(args.get("param-t", null));

        TeamControlProvider controls = new TeamControlProvider();
        Map<Team, Object> providers = new LinkedHashMap<>();
        Map<Team, Records> records = new LinkedHashMap<>();
        Tail botOut = new Tail();
        String[] names = new String[2];
        for (Team team : new Team[]{Team.A, Team.B}) {
            String spec = args.need(team == Team.A ? "a" : "b");
            String[] parts = spec.split(":");
            names[team.ordinal()] = spec;
            Records rec = recordTeams.contains(team.name()) ? new Records(new File(out, "records-" + team.name() + ".bin.gz")) : null;
            if (rec != null) records.put(team, rec);
            RobotControlProvider prov;
            if (parts[0].equals("bot")) {
                Bot b = new Bot(team, parts[1], parts[2], botOut, rec, recordP, recordPKing, seed * 2L + team.ordinal());
                b.pBoost = recordPBoost;
                if (mixNet != null && rec != null) { b.mixNet = mixNet; b.mixP = mixP; b.mixT = mixT; b.mixRng = new Random(seed * 2L + team.ordinal() + 7777); }
                prov = b;
            }
            else if (parts[0].equals("net")) prov = new Student(team, new Net(new File(parts[1])), parts.length > 2 ? Double.parseDouble(parts[2]) : 0.0, seed * 2L + team.ordinal() + 99, rec, recordP, recordPKing);
            else throw new IllegalArgumentException("bad team spec " + spec);
            controls.registerControlProvider(team, prov);
            providers.put(team, prov);
        }
        controls.registerControlProvider(Team.NEUTRAL, new NullControlProvider());
        GameInfo info = new GameInfo("A", CrossPlayLanguage.JAVA, "A", null, "B", CrossPlayLanguage.JAVA, "B", null, new String[]{mapName}, null, false);
        GameWorld world = new GameWorld(map, controls, new alpha0.NoopGameMaker(info).getMatchMaker());
        String error = null;
        try {
            int trace = Integer.parseInt(args.get("trace", "0"));
            traceDeaths = trace > 0;
            PrintWriter tw = trace > 0 ? new PrintWriter(new FileWriter(new File(out, "trace.tsv"))) : null;
            if (tw != null) tw.println("round\tA_babies\tA_kings\tA_cheese\tA_xfer\tA_rattraps\tB_babies\tB_kings\tB_cheese\tB_xfer\tB_rattraps");
            while (world.isRunning() && world.getCurrentRound() < maxRounds + 1) {
                world.runRound();
                TeamInfo th = world.getTeamInfo();   // trajectory fingerprint: identical play gives identical hashes
                for (Team tm : new Team[]{Team.A, Team.B})
                    trajHash = trajHash * 1000003L + th.getCheese(tm) * 31L + th.getCheeseTransferred(tm) * 997L
                        + th.getNumBabyRats(tm) * 7919L + th.getNumRatKings(tm) * 104729L + th.getDamageToCats(tm)
                        + world.getTrapCount(TrapType.RAT_TRAP, tm) * 15485863L;
                if (tw != null && world.getCurrentRound() % trace == 0) {
                    TeamInfo t = world.getTeamInfo();
                    StringBuilder sb = new StringBuilder().append(world.getCurrentRound());
                    for (Team tm : new Team[]{Team.A, Team.B})
                        sb.append('\t').append(t.getNumBabyRats(tm)).append('\t').append(t.getNumRatKings(tm))
                          .append('\t').append(t.getCheese(tm)).append('\t').append(t.getCheeseTransferred(tm))
                          .append('\t').append(world.getTrapCount(TrapType.RAT_TRAP, tm));
                    tw.println(sb);
                }
            }
            if (tw != null) tw.close();
        } catch (Throwable t) { error = t.toString(); t.printStackTrace(); }
        if (world.isRunning()) controls.matchEnded();
        for (Records r : records.values()) r.close();
        Decisions.closeAll();

        Map<String, Object> res = new LinkedHashMap<>();
        res.put("map", mapName); res.put("seed", seed); res.put("a", names[0]); res.put("b", names[1]);
        res.put("obs_version", Obs.VERSION); res.put("action_version", Actions.VERSION);
        res.put("obs_size", Obs.SIZE); res.put("n_actions", Actions.N); res.put("record_size", Records.SIZE);
        res.put("tournament_limits", Boolean.getBoolean("bc.alpha0.tournamentLimits"));
        res.put("finished", !world.isRunning());
        res.put("rounds", world.getCurrentRound());
        res.put("traj_hash", Long.toHexString(trajHash));
        res.put("winner", world.getWinner() == null ? null : world.getWinner().name());
        res.put("reason", world.getGameStats().getDominationFactor() == null ? null : world.getGameStats().getDominationFactor().toString());
        TeamInfo ti = world.getTeamInfo();
        for (Team t : new Team[]{Team.A, Team.B}) {
            Map<String, Object> s = new LinkedHashMap<>();
            s.put("cheese", ti.getCheese(t)); s.put("transferred", ti.getCheeseTransferred(t)); s.put("cat_damage", ti.getDamageToCats(t));
            s.put("kings", ti.getNumRatKings(t)); s.put("babies", ti.getNumBabyRats(t)); s.put("points", ti.getPoints(t));
            Object p = providers.get(t);
            if (p instanceof Bot b) { s.put("turns", b.turns); s.put("sampled_turns", b.sampledTurns); s.put("end_records", b.ends); s.put("failures", b.failures); s.put("student_turns", b.studentTurns); s.put("student_primitives", b.studentPrimitives); }
            if (p instanceof Student st) { s.put("decisions", st.decisions); s.put("primitives", st.primitives); s.put("errors", st.errors); s.put("sampled_turns", st.sampledTurns); }
            if (records.containsKey(t)) s.put("records", records.get(t).count);
            if (decisionTeams.contains(t.name())) s.put("decisions_recorded", Decisions.count[t.ordinal()]);
            res.put(t.name(), s);
        }
        res.put("decisions_errors", Decisions.errors);
        res.put("hook_recorded", Hook.recorded); res.put("hook_label_not_legal", Hook.labelNotLegal); res.put("hook_failed_calls", Hook.failedCalls);
        res.put("hook_unmapped", Hook.unmapped); res.put("hook_ignored", Hook.ignored);
        res.put("hook_error", Hook.error == null ? null : Hook.error.toString());
        if (Hook.error != null) Hook.error.printStackTrace();
        res.put("error", error);
        res.put("bot_output_tail", botOut.toString().length() > 2000 ? botOut.toString().substring(botOut.toString().length() - 2000) : botOut.toString());
        res.put("seconds", (System.nanoTime() - t0) / 1e9);
        File tmp = new File(out, "result.json.tmp");
        JSON.writerWithDefaultPrettyPrinter().writeValue(tmp, res);
        tmp.renameTo(new File(out, "result.json"));
        System.exit(0);
    }
}
