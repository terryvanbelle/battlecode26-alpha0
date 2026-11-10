package alpha0;

import battlecode.common.*;
import battlecode.crossplay.CrossPlayLanguage;
import battlecode.server.*;
import battlecode.world.*;
import battlecode.world.control.*;
import com.fasterxml.jackson.databind.*;
import java.io.*;
import java.lang.reflect.Field;
import java.util.*;

/** JSONL bridge to the official 1.2.6 game physics. One action is one robot turn. */
public final class EngineBridge {
    private static final ObjectMapper JSON=new ObjectMapper();
    private static final Direction[] DIRS=Arrays.copyOf(Direction.values(),8);
    private static final Map<Integer,Saved> SNAPSHOTS=new HashMap<>();
    private static int nextSnapshot=1;
    private static Session game;
    private static JsonNode resetRequest;
    private static final DiagnosticLog ENGINE_DIAGNOSTICS=new DiagnosticLog();
    /** Bounded diagnostic tail; bot chatter cannot grow the worker's heap indefinitely. */
    private static final class DiagnosticLog extends OutputStream {
        private final byte[] tail=new byte[65536];
        private int size=0;
        @Override public synchronized void write(int value) { write(new byte[]{(byte)value},0,1); }
        @Override public synchronized void write(byte[] bytes,int offset,int count) {
            int take=Math.min(count,tail.length), keep=Math.min(size,tail.length-take);
            System.arraycopy(tail,size-keep,tail,0,keep);
            System.arraycopy(bytes,offset+count-take,tail,keep,take);size=keep+take;
        }
        synchronized void clear() {size=0;}
        @Override public synchronized String toString() {return new String(tail,0,size,java.nio.charset.StandardCharsets.UTF_8);}
    }
    private static final class MonitoredPlayer extends PlayerControlProvider {
        private final Session session;
        MonitoredPlayer(Session session,Team team,String pkg,String classes) {
            super(team,pkg,CrossPlayLanguage.JAVA,null,classes,session.botOutput,false);this.session=session;
        }
        @Override public void robotSpawned(InternalRobot robot) {
            super.robotSpawned(robot);
            if(getTerminated(robot)) session.recordFailure(robot,"initialization_failed");
        }
        @Override public void runRobot(InternalRobot robot) {
            super.runRobot(robot);
            // disintegrate() throws RobotDeathException before the scheduler removes the
            // robot. Use the sandbox's authoritative reason; a live entry alone is not
            // evidence of a crash. Resign/physical death may remove the robot first.
            if(getTerminated(robot) && !terminatedByRobotDeath(robot) && robot.getGameWorld().getObjectInfo().getRobotByID(robot.getID())!=null)
                session.recordFailure(robot,"unexpected_sandbox_exit");
        }
    }
    private static final class Saved {
        Session session; Random catRandom;
        Saved(Session s,Random r) {session=s;catRandom=r;}
    }
    public static final class Session {
        GameWorld world;
        TeamControlProvider controls;
        int[] order;
        int cursor;
        InternalRobot current;
        boolean benchmark;
        String actionVersion;
        int learnedTeam;
        long decisions;
        ArrayList<Map<String,Object>> benchmarkFailures=new ArrayList<>();
        DiagnosticLog botOutput;
        Session(JsonNode req) throws Exception {
            actionVersion=req.path("action_version").asText("macro_v1");
            if(!actionVersion.equals("macro_v1") && !actionVersion.equals("macro_v2")) throw new IllegalArgumentException("Unknown action_version: "+actionVersion);
            benchmark=req.hasNonNull("opponent_package"); learnedTeam=req.path("learned_team").asInt(0);
            if(benchmark) botOutput=new DiagnosticLog();
            String mapName=req.path("map").asText("tiny");
            LiveMap map=GameMapIO.loadMap(mapName,new File(req.path("maps_dir").asText("maps")),false);
            if(req.has("seed")) { Field f=LiveMap.class.getDeclaredField("seed"); f.setAccessible(true); f.setInt(map,req.get("seed").asInt()); }
            InternalRobot.rand=new Random(1092);
            controls=new TeamControlProvider();
            for(Team team:new Team[]{Team.A,Team.B,Team.NEUTRAL}) {
                RobotControlProvider provider=new NullControlProvider();
                if(benchmark && team!=Team.NEUTRAL && team.ordinal()!=learnedTeam) {
                    provider=new MonitoredPlayer(this,team,req.get("opponent_package").asText(),req.get("opponent_classes").asText());
                }
                controls.registerControlProvider(team,provider);
            }
            GameInfo info=new GameInfo("alpha0",CrossPlayLanguage.JAVA,"alpha0",null,"opponent",CrossPlayLanguage.JAVA,"opponent",null,new String[]{mapName},null,false);
            world=new GameWorld(map,controls,new NoopGameMaker(info).getMatchMaker());
            advance();
        }
        void recordFailure(InternalRobot robot,String reason) {
            benchmarkFailures.add(Map.of("robot_id",robot.getID(),"team",robot.getTeam().ordinal(),"round",robot.getGameWorld().getCurrentRound(),"reason",reason));
        }
        void finish(InternalRobot robot) {
            robot.setBytecodesUsed(controls.getBytecodesUsed(robot));
            robot.processEndOfTurn();
            if(controls.getTerminated(robot) && world.getObjectInfo().getRobotByID(robot.getID())!=null && robot.getLocation()!=null) world.destroyRobot(robot.getID());
        }
        void advance() {
            current=null;
            while(world.isRunning()) {
                if(order==null) {
                    world.processBeginningOfRound(); controls.roundStarted();
                    ArrayList<Integer> ids=new ArrayList<>();
                    world.getObjectInfo().eachDynamicBodyByExecOrder(r->{ids.add(r.getID());return true;});
                    order=ids.stream().mapToInt(Integer::intValue).toArray(); cursor=0;
                }
                while(cursor<order.length) {
                    InternalRobot r=world.getObjectInfo().getRobotByID(order[cursor++]);
                    if(r==null) continue;
                    r.processBeginningOfTurn();
                    // A flying rat can die on landing during beginning-of-turn processing.
                    // Its native sandbox would already be terminated; expose no dead actor.
                    if(world.getObjectInfo().getRobotByID(r.getID())==null) { finish(r); continue; }
                    if(r.getTeam()==Team.NEUTRAL || (benchmark && r.getTeam().ordinal()!=learnedTeam)) {
                        controls.runRobot(r); finish(r);
                    } else {
                        current=r;
                        // Forced HOLD turns carry no policy choice. Preserve all official turn effects.
                        if(legalActions().size()>1) return;
                        macro(0); finish(r); current=null;
                    }
                }
                controls.roundEnded(); world.processEndOfRound(); order=null;
                if(!world.isRunning()) controls.matchEnded();
            }
        }
        List<Integer> legalActions() {
            if("macro_v2".equals(actionVersion)) return MacroV2.legal(this);
            ArrayList<Integer> ids=new ArrayList<>();
            if(current==null) return ids;
            ids.add(0);
            RobotControllerImpl rc=current.getController();
            for(int i=0;i<8;i++) {
                // A move macro can clear adjacent dirt instead of moving this turn.
                if(rc.canMove(DIRS[i]) || rc.canRemoveDirt(rc.getLocation().add(DIRS[i]))) ids.add(i+1);
                if(buildLocation(rc,DIRS[i])!=null) ids.add(i+9);
            }
            return ids;
        }
        MapLocation buildLocation(RobotControllerImpl rc,Direction dir) {
            if(rc.getType()!=UnitType.RAT_KING || !rc.isActionReady()) return null;
            MapLocation here=rc.getLocation(), best=null; int bestCost=Integer.MAX_VALUE;
            // King's 3x3 footprint requires spawn offsets beyond the adjacent centre tile.
            for(int dy=-3;dy<=3;dy++) for(int dx=-3;dx<=3;dx++) {
                if(dx*dir.dx+dy*dir.dy<=0) continue;
                MapLocation p=here.translate(dx,dy);
                if(rc.canBuildRat(p)) {
                    int cost=(dx-2*dir.dx)*(dx-2*dir.dx)+(dy-2*dir.dy)*(dy-2*dir.dy);
                    if(cost<bestCost) {bestCost=cost;best=p;}
                }
            }
            return best;
        }
        void forage(RobotControllerImpl rc) throws GameActionException {
            MapLocation here=rc.getLocation();
            for(int dy=-2;dy<=2;dy++) for(int dx=-2;dx<=2;dx++) {
                MapLocation p=here.translate(dx,dy);
                if(rc.canPickUpCheese(p)) rc.pickUpCheese(p);
            }
            if(rc.getRawCheese()>0 && rc.getType()==UnitType.BABY_RAT) {
                for(InternalRobot friend:world.getAllRobots(rc.getTeam(),-1)) {
                    if(friend.getType()==UnitType.RAT_KING && rc.canTransferCheese(friend.getLocation(),rc.getRawCheese())) {
                        rc.transferCheese(friend.getLocation(),rc.getRawCheese()); break;
                    }
                }
            }
        }
        void attack(RobotControllerImpl rc) throws GameActionException {
            // Fixed tactical policy: prefer enemy rats, then cats; attacking can backstab.
            RobotInfo[] seen=rc.senseNearbyRobots();
            Arrays.sort(seen,Comparator.comparingInt((RobotInfo r)->r.team==rc.getTeam().opponent()?0:1).thenComparingInt(r->r.health).thenComparingInt(r->r.ID));
            for(RobotInfo r:seen) if(r.team!=rc.getTeam() && rc.canAttack(r.location)) {rc.attack(r.location);break;}
        }
        void macro(int action) {
            if("macro_v2".equals(actionVersion)) {MacroV2.execute(this,action);return;}
            try {
                RobotControllerImpl rc=current.getController();
                forage(rc);
                if(action>=9) {
                    Direction d=DIRS[action-9]; MapLocation p=buildLocation(rc,d);
                    if(rc.canTurn(d)) rc.turn(d);
                    if(p!=null && rc.canBuildRat(p)) rc.buildRat(p);
                } else if(action>0) {
                    Direction d=DIRS[action-1];
                    if(rc.canTurn(d)) rc.turn(d);
                    if(rc.canMove(d)) rc.move(d);
                    else if(rc.canRemoveDirt(rc.getLocation().add(d))) rc.removeDirt(rc.getLocation().add(d));
                }
                forage(rc); attack(rc);
            } catch(GameActionException e) {throw new IllegalStateException("Macro violated controller legality",e);}
        }
        void step(int action) {
            if(current==null || !legalActions().contains(action)) throw new IllegalArgumentException("Illegal action "+action);
            macro(action); finish(current); decisions++; advance();
        }
        Map<String,Object> state() {
            Map<String,Object> s=new LinkedHashMap<>();
            s.put("action_version",actionVersion==null?"macro_v1":actionVersion);
            s.put("action_dim","macro_v2".equals(actionVersion)?MacroV2.ACTION_DIM:17);
            int w=world.getGameMap().getWidth(),h=world.getGameMap().getHeight();
            s.put("width",w);s.put("height",h);s.put("round",world.getCurrentRound());s.put("seed",world.getMapSeed());s.put("map",world.getGameMap().getMapName());
            s.put("terminal",!world.isRunning());s.put("winner",world.getWinner()==null?-1:world.getWinner().ordinal());
            s.put("win_reason",world.getGameStats().getDominationFactor()==null?null:world.getGameStats().getDominationFactor().toString());
            s.put("current_id",current==null?-1:current.getID());s.put("current_team",current==null?-1:current.getTeam().ordinal());
            s.put("cooperation",world.isCooperation());s.put("legal_actions",legalActions());s.put("decisions",decisions);
            TeamInfo ti=world.getTeamInfo();
            s.put("bank",new int[]{ti.getCheese(Team.A),ti.getCheese(Team.B)});s.put("transferred",new int[]{ti.getCheeseTransferred(Team.A),ti.getCheeseTransferred(Team.B)});
            s.put("cat_damage",new int[]{ti.getDamageToCats(Team.A),ti.getDamageToCats(Team.B)});s.put("kings",new int[]{ti.getNumRatKings(Team.A),ti.getNumRatKings(Team.B)});
            s.put("dirt_bank",new int[]{ti.getDirt(Team.A),ti.getDirt(Team.B)});
            ArrayList<Map<String,Object>> robots=new ArrayList<>();
            world.getObjectInfo().eachDynamicBodyByExecOrder(r->{
                Map<String,Object> v=new LinkedHashMap<>();
                v.put("id",r.getID());v.put("team",r.getTeam().ordinal());v.put("type",r.getType().toString());
                v.put("x",r.getLocation().x);v.put("y",r.getLocation().y);v.put("hp",r.getHealth());v.put("cheese",r.getCheese());v.put("direction",r.getDirection().ordinal());
                v.put("action_cooldown",r.getActionCooldownTurns());v.put("movement_cooldown",r.getMovementCooldownTurns());v.put("turning_cooldown",r.getTurningCooldownTurns());
                v.put("carried",r.isGrabbedByRobot());v.put("thrown",r.isBeingThrown());
                v.put("carrying_id",r.getRobotBeingCarried()==null?-1:r.getRobotBeingCarried().getID());
                v.put("carried_by_id",r.getGrabbedByRobot()==null?-1:r.getGrabbedByRobot().getID());
                v.put("remaining_carried_duration",r.getRemainingCarriedDuration());v.put("chirality",r.getChirality());
                ArrayList<int[]> parts=new ArrayList<>();for(MapLocation p:r.getAllPartLocations()) parts.add(new int[]{p.x,p.y});v.put("parts",parts);
                robots.add(v);return true;});
            s.put("robots",robots);
            int n=w*h;int[] walls=new int[n],dirt=new int[n],cheese=new int[n],mines=new int[n],trapsA=new int[n],trapsB=new int[n];
            for(int y=0;y<h;y++)for(int x=0;x<w;x++) {int i=y*w+x;MapLocation p=new MapLocation(x,y);walls[i]=world.getWall(p)?1:0;dirt[i]=world.getDirt(p)?1:0;cheese[i]=world.getCheese(p);mines[i]=world.hasCheeseMine(p)?1:0;Trap a=world.getTrap(p,Team.A),b=world.getTrap(p,Team.B);trapsA[i]=a==null?0:a.getType()==TrapType.RAT_TRAP?1:2;trapsB[i]=b==null?0:b.getType()==TrapType.RAT_TRAP?1:2;}
            s.put("walls",walls);s.put("dirt",dirt);s.put("cheese",cheese);s.put("mines",mines);s.put("trapsA",trapsA);s.put("trapsB",trapsB);
            if(benchmark) {
                s.put("valid_game",benchmarkFailures.isEmpty());
                s.put("benchmark_failures",benchmarkFailures);
                if(!benchmarkFailures.isEmpty()) {
                    // Keep exception stacks separate: verbose bot chatter must not evict
                    // the actual engine error when the combined diagnostic is bounded.
                    String engineDiagnostics=ENGINE_DIAGNOSTICS.toString();
                    String botDiagnostics=botOutput.toString();
                    if(engineDiagnostics.length()>8000) engineDiagnostics=engineDiagnostics.substring(engineDiagnostics.length()-8000);
                    if(botDiagnostics.length()>4000) botDiagnostics=botDiagnostics.substring(botDiagnostics.length()-4000);
                    String diagnostics="ENGINE DIAGNOSTICS:\n"+engineDiagnostics+"\nBOT OUTPUT TAIL:\n"+botDiagnostics;
                    s.put("engine_diagnostics",engineDiagnostics);
                    s.put("bot_output_tail",botDiagnostics);
                    s.put("diagnostics",diagnostics);
                    s.put("error","Benchmark sandbox failure; game must not be scored: "+benchmarkFailures+"\n"+diagnostics);
                }
            }
            return s;
        }
    }
    private static int save() {
        if(game.benchmark) throw new IllegalStateException("Snapshots are supported for self-play only; benchmark sandboxes contain live threads");
        StateCopier c=new StateCopier(); int id=nextSnapshot++;
        SNAPSHOTS.put(id,new Saved(c.copy(game),c.copy(InternalRobot.rand)));return id;
    }
    public static void main(String[] args) throws Exception {
        BufferedReader in=new BufferedReader(new InputStreamReader(System.in));PrintWriter out=new PrintWriter(System.out,true);
        // stdout is the JSONL protocol, never a bot console. In particular, print()
        // without a newline must not prefix and corrupt the next JSON response.
        Config.getGlobalConfig().setBoolean("bc.server.robot-player-to-system-out",false);
        PrintStream originalError=System.err;
        System.setErr(new PrintStream(new OutputStream(){
            @Override public void write(int value) {originalError.write(value);ENGINE_DIAGNOSTICS.write(value);}
            @Override public void write(byte[] bytes,int offset,int count) {originalError.write(bytes,offset,count);ENGINE_DIAGNOSTICS.write(bytes,offset,count);}
        },true,java.nio.charset.StandardCharsets.UTF_8));
        String line;
        while((line=in.readLine())!=null) {
            try {
                JsonNode req=JSON.readTree(line);String cmd=req.path("cmd").asText();Object response;
                switch(cmd) {
                    case "reset":
                        if(game!=null && game.world.isRunning())game.controls.matchEnded();
                        ENGINE_DIAGNOSTICS.clear();SNAPSHOTS.clear();resetRequest=req.deepCopy();game=new Session(req);response=game.state();break;
                    case "step":game.step(req.get("action").asInt());response=game.state();break;
                    case "state":response=game.state();break;
                    case "snapshot":response=Map.of("snapshot",save());break;
                    case "restore":{
                        Saved saved=SNAPSHOTS.get(req.get("snapshot").asInt());if(saved==null)throw new IllegalArgumentException("Unknown snapshot");
                        StateCopier c=new StateCopier();game=c.copy(saved.session);InternalRobot.rand=c.copy(saved.catRandom);response=game.state();break;}
                    case "release":
                        if(req.has("snapshots")) for(JsonNode id:req.get("snapshots"))SNAPSHOTS.remove(id.asInt());
                        else SNAPSHOTS.remove(req.get("snapshot").asInt());response=Map.of("ok",true,"snapshots",SNAPSHOTS.size());break;
                    case "replay":
                        if(game!=null && game.world.isRunning())game.controls.matchEnded();
                        game=new Session(req.has("map")?req:resetRequest);for(JsonNode action:req.path("actions"))game.step(action.asInt());response=game.state();break;
                    case "close":if(game!=null && game.world.isRunning())game.controls.matchEnded();out.println("{\"ok\":true}");return;
                    default:throw new IllegalArgumentException("Unknown command: "+cmd);
                }
                if(req.path("snapshot").isBoolean() && req.path("snapshot").asBoolean() && response instanceof Map) ((Map<String,Object>)response).put("snapshot",save());
                out.println(JSON.writeValueAsString(response));
            } catch(Throwable error) {error.printStackTrace(System.err);out.println(JSON.writeValueAsString(Map.of("error",error.toString())));}
        }
        if(game!=null && game.world.isRunning())game.controls.matchEnded();
    }
}
