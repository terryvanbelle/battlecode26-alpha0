import alpha0.NoopGameMaker;
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
 static final List<Object> budgets=new ArrayList<>();
 static class Player extends PlayerControlProvider {
  Player(String classes,String bot){super(Team.A,bot,CrossPlayLanguage.JAVA,null,classes,new ByteArrayOutputStream(),false);}
  public void runRobot(InternalRobot r){super.runRobot(r);budgets.add(List.of(r.getID(),world.getCurrentRound(),getBytecodesUsed(r)));}
 }
 public static void main(String[] args)throws Exception {
  Config.setGlobalConfig(new Config(new String[0]));
  boolean recording=args[2].equals("recording");String bot=args.length>3?args[3]:"imitationfixture";Path output=Path.of(args[1]);Files.createDirectory(output);
  try(TraceJournal journal=new TraceJournal(output.resolve("trace.jsonl"),"a".repeat(64),8*1024*1024)) {
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
      try{observer.endTurn(r,TraceJournal.digest(Map.of("round",world.getCurrentRound(),"robot",r.getID())),reason,n);}
      catch(IOException e){throw new UncheckedIOException(e);}
     }
     public void terminated(RobotController r,int n){} // externally stopped fixture: no invented death event
    };
   });
   var map=GameMapIO.loadMap("tiny",new File("maps"),false);
   InternalRobot.rand=new Random(1092);
   var controls=new TeamControlProvider();controls.registerControlProvider(Team.A,new Player(args[0],bot));
   controls.registerControlProvider(Team.B,new NullControlProvider());controls.registerControlProvider(Team.NEUTRAL,new NullControlProvider());
   var info=new GameInfo("fixture",CrossPlayLanguage.JAVA,"fixture",null,"null",CrossPlayLanguage.JAVA,"null",null,new String[]{"tiny"},null,false);
   world=new GameWorld(map,controls,new NoopGameMaker(info).getMatchMaker());
   try{for(int i=0;i<2;i++){world.runRound();if(!world.isRunning())throw new AssertionError("unexpected terminal");}}
   finally{controls.matchEnded();}
   RobotThreadRecorder.requireHealthy();
   Files.write(output.resolve("budgets.json"),TraceJournal.canonical(budgets));
   Files.write(output.resolve("result.json"),TraceJournal.canonical(Map.of("round",world.getCurrentRound(),"complete",false,"teacher_games",0,"recording",recording)));
   System.out.println("PASS real-world synthetic fixture "+args[2]+" budgets="+budgets);
  }
 }
}
