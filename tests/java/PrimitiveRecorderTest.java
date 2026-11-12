import alpha0.imitation.*;
import battlecode.common.*;
import java.lang.reflect.*;
import java.util.*;
public class PrimitiveRecorderTest {
    static void check(boolean b){if(!b)throw new AssertionError();}
    public static void main(String[] args) throws Exception {
        List<PrimitiveRecorder.Event> events=new ArrayList<>();
        RobotController rc=(RobotController)Proxy.newProxyInstance(RobotController.class.getClassLoader(),new Class[]{RobotController.class},(p,m,a)->switch(m.getName()) {
            case "getID" -> 7;case "getTeam" -> Team.A;case "getRoundNum" -> 3;
            default -> throw new AssertionError("unexpected API read: "+m.getName());
        });
        PrimitiveRecorder.Observer o=new PrimitiveRecorder.Observer(){
            public PrimitiveRecorder.Snapshot before(RobotController r){return new PrimitiveRecorder.Snapshot("a".repeat(64),10);}
            public int bytecodes(RobotController r){return 20;}
            public void accept(PrimitiveRecorder.Event e){events.add(e);}
        };
        check(PrimitiveRecorder.begin(rc,"move",new Object[]{Direction.NORTH})==null);
        PrimitiveRecorder.finish(null,null,null);check(events.isEmpty());
        PrimitiveRecorder.bind(o);
        var outer=PrimitiveRecorder.begin(rc,"attack",new Object[]{new MapLocation(2,3)});
        var inner=PrimitiveRecorder.begin(rc,"attackAmount",new Object[]{new MapLocation(2,3),0});
        check(inner==null);PrimitiveRecorder.finish(inner,null,null);check(events.isEmpty());
        PrimitiveRecorder.finish(outer,null,null);check(events.size()==1 && events.get(0).operation().equals("attack"));
        check(events.get(0).arguments().equals(List.of(List.of(2,3))));
        var squeak=PrimitiveRecorder.begin(rc,"squeak",new Object[]{5});PrimitiveRecorder.finish(squeak,false,null);
        check(events.get(1).returnedBoolean().equals(false));
        var failed=PrimitiveRecorder.begin(rc,"move",new Object[]{Direction.NORTH});PrimitiveRecorder.finish(failed,null,new IllegalArgumentException("fixture"));
        check(events.get(2).exceptionClass().equals("java.lang.IllegalArgumentException"));
        PrimitiveRecorder.unbind();
        Thread t=new Thread(()->{check(PrimitiveRecorder.begin(rc,"move",new Object[]{Direction.SOUTH})==null);PrimitiveRecorder.finish(null,null,null);});t.start();t.join();
        check(events.size()==3);
        System.out.println("PASS: disabled path, nested overload, normalized arguments, false squeak, exception, thread isolation");
    }
}
