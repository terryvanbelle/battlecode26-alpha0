import alpha0.NoopGameMaker;
import com.fasterxml.jackson.databind.ObjectMapper;
import alpha0.imitation.*;
import battlecode.common.*;
import battlecode.crossplay.CrossPlayLanguage;
import battlecode.server.*;
import battlecode.world.*;
import battlecode.world.control.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Real GameWorld, synthetic bot, two rounds only. Deliberately never seals a game. */
public class WorldJournalTest {
 static GameWorld world;
 static long storedAuditBytes;
 static final long AUDIT_CAP=64L*1024*1024;
 static Path outputDirectory;
 static final List<Object> auditSequence=new ArrayList<>();
 static String saveAudit()throws IOException {
  var state=WorldAudit.capture(world);byte[] bytes=TraceJournal.canonical(state);
  if(!Arrays.equals(bytes,TraceJournal.canonical(WorldAudit.capture(world))))throw new AssertionError("audit mutated world");
  String sha=TraceJournal.digest(state);Path p=outputDirectory.resolve("world-audits").resolve(sha+".json");
  if(!Files.exists(p)){if(bytes.length>AUDIT_CAP-storedAuditBytes)throw new IOException("audit storage cap");Files.write(p,bytes,StandardOpenOption.CREATE_NEW);storedAuditBytes+=bytes.length;}
  return sha;
 }
 static final List<Object> roster=new ArrayList<>();
 static final List<Object> budgets=new ArrayList<>();
 static final List<Object> terminations=new ArrayList<>();
 static class Player extends PlayerControlProvider {
  Player(String classes,String bot){super(Team.A,bot,CrossPlayLanguage.JAVA,null,classes,new ByteArrayOutputStream(),false);}
  public void robotSpawned(InternalRobot r){roster.add(Map.of("robot",r.getID(),"team",r.getTeam().name()));super.robotSpawned(r);}
  public void runRobot(InternalRobot r){super.runRobot(r);budgets.add(List.of(r.getID(),world.getCurrentRound(),getBytecodesUsed(r)));
   try{auditSequence.add(List.of(r.getID(),world.getCurrentRound(),saveAudit()));}catch(IOException e){throw new UncheckedIOException(e);}}
 }
 public static void main(String[] args)throws Exception {
  Config.setGlobalConfig(new Config(new String[0]));
  var source=args.length>4?new ObjectMapper().readTree(Files.readString(Path.of(args[4]))):null;
  String gameId=source==null?"a".repeat(64):source.get("game_id").asText();TraceJournal.hash(gameId);
  boolean fullGame=source!=null && source.get("full_game").asBoolean();
  boolean recording=args[2].equals("recording");String bot=args.length>3?args[3]:"imitationfixture";Path output=Path.of(args[1]);Files.createDirectory(output);outputDirectory=output;Files.createDirectory(output.resolve("world-audits"));
  try(TraceJournal journal=new TraceJournal(output.resolve("trace.jsonl"),gameId,8*1024*1024)) {
   var store=new JournalObserver.ObservationStore(output.resolve("observations"),8*1024*1024);
   if(recording)RobotThreadRecorder.configure((rc,budget)->{
    var observer=new JournalObserver(journal,store,budget);
    return new RobotThreadRecorder.Session(){
     public PrimitiveRecorder.Snapshot before(RobotController r){return observer.before(r);}
     public int bytecodes(RobotController r){return observer.bytecodes(r);}
     public void accept(PrimitiveRecorder.Event e){observer.accept(e);}
     public void turnEnded(RobotController r,int n){throw new AssertionError("missing reason");}
     public void turnEnded(RobotController r,int n,String reason){
      String expected=bot.equals("imitationoverrunfixture")?"overrun":"yield";
      if(!reason.equals(expected))throw new IllegalStateException("unexpected fixture pause reason");
      // Minimal diagnostic hash only; not a complete physical-state proof.
      try{observer.endTurn(r,saveAudit(),reason,n);}
      catch(IOException e){throw new UncheckedIOException(e);}
     }
     public void terminated(RobotController r,int n){throw new AssertionError("missing exit cause");}
     public void terminated(RobotController r,int n,RobotThreadRecorder.Termination t){
      Map<String,Object> row=new TreeMap<>();row.put("robot",r.getID());row.put("team",r.getTeam().name());row.put("round",r.getRoundNum());row.put("budget",n);row.put("outcome",t.outcome());
      row.put("exception",t.exceptionClass());row.put("turn_open",t.turnOpen());terminations.add(row);
     }
    };
   });
   var map=GameMapIO.loadMap(source==null?"tiny":source.get("map").asText(),new File("maps"),false);
   if(source!=null){var seedField=LiveMap.class.getDeclaredField("seed");seedField.setAccessible(true);seedField.setInt(map,source.get("seed").asInt());}
   if(source!=null)Files.writeString(output.resolve("source-spec.json"),source.toString(),StandardOpenOption.CREATE_NEW);
   InternalRobot.rand=new Random(1092);
   var controls=new TeamControlProvider();controls.registerControlProvider(Team.A,new Player(args[0],bot));
   controls.registerControlProvider(Team.B,new NullControlProvider());controls.registerControlProvider(Team.NEUTRAL,new NullControlProvider());
   var info=new GameInfo("fixture",CrossPlayLanguage.JAVA,"fixture",null,"null",CrossPlayLanguage.JAVA,"null",null,new String[]{"tiny"},null,false);
   world=new GameWorld(map,controls,new NoopGameMaker(info).getMatchMaker());
   try{for(int i=0;i<(fullGame?GameConstants.GAME_MAX_NUMBER_OF_ROUNDS:2);i++){world.runRound();if(!world.isRunning())break;}}
   finally{controls.matchEnded();}
   RobotThreadRecorder.requireHealthy();
   if(fullGame && world.isRunning())throw new AssertionError("full-game source did not finish");
   String finalAudit=saveAudit();
   Files.write(output.resolve("world-audit.json"),TraceJournal.canonical(WorldAudit.capture(world)));
   Files.write(output.resolve("world-audit-sequence.json"),TraceJournal.canonical(auditSequence));
   if(!world.isRunning()) {
    if(world.getWinner()==null || world.getGameStats().getDominationFactor()==null)throw new AssertionError("missing terminal result");
    Files.write(output.resolve("terminal-proof.json"),TraceJournal.canonical(Map.of("schema","synthetic-world-terminal-v1","world_terminal",true,"winner",world.getWinner().name(),"reason",world.getGameStats().getDominationFactor().name(),"round",world.getCurrentRound(),"physical_state_sha256",finalAudit,"training_approved",false)));
   }
   // Check sensitivity to hidden shared state; restore it before accepting the fixture.
   var sharedField=GameWorld.class.getDeclaredField("sharedArray");sharedField.setAccessible(true);
   int[][] shared=(int[][])sharedField.get(world);int prior=shared[0][0];shared[0][0]=prior^1;
   if(TraceJournal.digest(WorldAudit.capture(world)).equals(finalAudit))throw new AssertionError("shared state omitted");
   shared[0][0]=prior;
   if(!TraceJournal.digest(WorldAudit.capture(world)).equals(finalAudit))throw new AssertionError("state restore failed");
   Files.write(output.resolve("recorder-roster.json"),TraceJournal.canonical(roster));
   Files.write(output.resolve("terminations.json"),TraceJournal.canonical(terminations));
   Files.write(output.resolve("budgets.json"),TraceJournal.canonical(budgets));
   Files.write(output.resolve("result.json"),TraceJournal.canonical(Map.of("round",world.getCurrentRound(),"complete",false,"teacher_games",0,"recording",recording,"world_terminal",!world.isRunning(),"winner",world.getWinner()==null?"none":world.getWinner().name())));
   System.out.println("PASS real-world synthetic fixture "+args[2]+" budgets="+budgets);
  }
 }
}
