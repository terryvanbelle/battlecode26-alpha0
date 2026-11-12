package alpha0.imitation;

import battlecode.common.*;
import java.util.*;

/** Engine-side prototype. Never installed in the historical benchmark classpath. */
public final class PrimitiveRecorder {
    public interface Observer {
        /** Must use only public API inputs; budget source must be separately audited. */
        Snapshot before(RobotController rc);
        int bytecodes(RobotController rc);
        void accept(Event event);
    }
    public record Snapshot(String publicObservationSha256, int bytecodes) {
        public Snapshot {
            if (publicObservationSha256 == null || !publicObservationSha256.matches("[0-9a-f]{64}") || bytecodes < 0)
                throw new IllegalArgumentException("unverified observation or budget");
        }
    }
    public record Event(int robotId, Team team, int round, String operation,
                        List<Object> arguments, Snapshot before, int bytecodesAfter,
                        Boolean returnedBoolean, String exceptionClass) {}
    public static final class Frame {
        final Observer observer; final RobotController rc; final String operation;
        final List<Object> args; final Snapshot before; final int robotId, round;
        final Team team; boolean finished;
        Frame(Observer o, RobotController r, String op, Object[] values) {
            observer=o;rc=r;operation=op;args=normalize(values);
            robotId=r.getID();team=r.getTeam();round=r.getRoundNum();before=o.before(r);
        }
    }
    private static final class State { Observer observer; int depth; }
    private static final ThreadLocal<State> STATE=ThreadLocal.withInitial(State::new);
    public static void bind(Observer observer) {
        State s=STATE.get();
        if(s.depth!=0 || s.observer!=null || observer==null) throw new IllegalStateException("invalid binding");
        s.observer=observer;
    }
    public static void unbind() {
        State s=STATE.get();if(s.depth!=0)throw new IllegalStateException("unfinished primitive");STATE.remove();
    }
    private static List<Object> normalize(Object[] values) {
        ArrayList<Object> out=new ArrayList<>();
        for(Object v:values) {
            if(v instanceof MapLocation p) out.add(List.of(p.x,p.y));
            else if(v instanceof Direction d) out.add(d.name());
            else if(v instanceof Integer || v==null) out.add(v);
            else throw new IllegalArgumentException("unknown primitive argument type");
        }
        return Collections.unmodifiableList(out);
    }
    /** Nested public overloads retain physical behavior but emit only the outer API call. */
    public static Frame begin(RobotController rc,String operation,Object[] args) {
        State s=STATE.get();
        if(s.observer==null)return null;
        if(s.depth++!=0)return null;
        try{return new Frame(s.observer,rc,operation,args);}
        catch(RuntimeException|Error e){s.depth--;throw e;}
    }
    public static void finish(Frame frame,Boolean returned,Throwable error) {
        State s=STATE.get();if(s.observer==null)return;
        if(s.depth<=0)throw new IllegalStateException("unbalanced primitive");
        s.depth--;
        if(frame==null)return;
        if(frame.finished || s.depth!=0)throw new IllegalStateException("invalid frame lifecycle");
        frame.finished=true;
        int used=frame.observer.bytecodes(frame.rc);
        if(used<0)throw new IllegalStateException("missing charged budget");
        frame.observer.accept(new Event(frame.robotId,frame.team,frame.round,frame.operation,
            frame.args,frame.before,used,returned,error==null?null:error.getClass().getName()));
    }
    private PrimitiveRecorder(){}
}
