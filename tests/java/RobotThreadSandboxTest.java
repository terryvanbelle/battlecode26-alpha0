import alpha0.imitation.*;
import battlecode.common.*;
import battlecode.crossplay.CrossPlayLanguage;
import battlecode.instrumenter.*;
import battlecode.world.control.PlayerControlProvider;
import battlecode.server.Config;
import battlecode.util.*;
import java.io.*;
import java.lang.reflect.*;
import java.util.*;
public class RobotThreadSandboxTest {
 static int round=1;static List<Integer> budgets=new ArrayList<>();static List<String> callbacks=new ArrayList<>();static int events=0;
 static void check(boolean b){if(!b)throw new AssertionError();}
 public static void main(String[] args)throws Exception {
  Config.setGlobalConfig(new Config(new String[0]));
  boolean recording=args.length<2 || !args[1].equals("baseline");
  List<Integer> actualBudgets=new ArrayList<>();
  if(recording)RobotThreadRecorder.configure((rc,budget)->new RobotThreadRecorder.Session(){
   public PrimitiveRecorder.Snapshot before(RobotController r){callbacks.add(Thread.currentThread().getName());return new PrimitiveRecorder.Snapshot("a".repeat(64),budget.getAsInt());}
   public int bytecodes(RobotController r){return budget.getAsInt();}
   public void accept(PrimitiveRecorder.Event e){check(e.bytecodesAfter()==e.before().bytecodes());events++;}
   public void turnEnded(RobotController r,int n){budgets.add(n);}
   public void terminated(RobotController r,int n){}
  });
  RobotController rc=(RobotController)Proxy.newProxyInstance(RobotController.class.getClassLoader(),new Class[]{RobotController.class},(p,m,a)->{
   switch(m.getName()){
    case "getID":return 7;case "getTeam":return Team.A;case "getRoundNum":return round;
    case "getLocation":return new MapLocation(1,1);case "getType":return UnitType.RAT_KING;
    case "writeSharedArray":case "squeak":
     var f=PrimitiveRecorder.begin((RobotController)p,m.getName(),a);PrimitiveRecorder.finish(f,m.getName().equals("squeak")?Boolean.TRUE:null,null);return m.getName().equals("squeak")?Boolean.TRUE:null;
    default:throw new AssertionError(m.getName());
   }
  });
  String classes=args[0];var provider=new PlayerControlProvider(Team.A,"imitationfixture",CrossPlayLanguage.JAVA,null,classes,new ByteArrayOutputStream(),false);
  var player=new SandboxedRobotPlayer("imitationfixture",CrossPlayLanguage.JAVA,null,rc,123,new TeamClassLoaderFactory(classes).createLoader(false),new ByteArrayOutputStream(),null,provider);
  check(budgets.isEmpty());
  for(int i=0;i<2;i++){round=i+1;player.setBytecodeLimit(20000);player.step();check(!player.getTerminated());actualBudgets.add(player.getBytecodesUsed());if(recording)check(budgets.get(i)==player.getBytecodesUsed());}
  if(recording){check(events==4 && budgets.size()==2);check(callbacks.stream().allMatch(s->s.contains("imitationfixture")));}else check(events==0 && budgets.isEmpty());
  player.terminate();RobotThreadRecorder.requireHealthy();
  System.out.println("PASS actual instrumented robot thread; events="+events+" budgets="+actualBudgets);
 }
}
