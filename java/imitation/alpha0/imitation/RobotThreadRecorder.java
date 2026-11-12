package alpha0.imitation;

import battlecode.common.RobotController;
import java.lang.reflect.Method;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.IntSupplier;
import java.util.function.BooleanSupplier;

/** Binding invoked only by the isolated, uninstrumented SandboxedRobotPlayer copy. */
public final class RobotThreadRecorder {
    public interface Session extends PrimitiveRecorder.Observer {
        void turnEnded(RobotController rc, int chargedBytecodes);
        void terminated(RobotController rc, int chargedBytecodes);
        default void turnEnded(RobotController rc,int chargedBytecodes,String reason){turnEnded(rc,chargedBytecodes);}
        default void terminated(RobotController rc,int chargedBytecodes,Termination termination){terminated(rc,chargedBytecodes);}
    }
    /** Raw sandbox outcome, not a claim that a physical death or game completion occurred. */
    public record Termination(String outcome,String exceptionClass,boolean turnOpen) {}
    public interface Factory { Session create(RobotController rc, IntSupplier chargedBytecodes); }
    private record Bound(RobotController rc, Session session, IntSupplier budget,BooleanSupplier dying) {}
    private static final ThreadLocal<Bound> BOUND=new ThreadLocal<>();
    private static final ThreadLocal<Boolean> STARTED=ThreadLocal.withInitial(()->false);
    private static volatile Factory factory;
    private static final ConcurrentLinkedQueue<String> failures=new ConcurrentLinkedQueue<>();
    public static synchronized void configure(Factory f) {
        if(factory!=null || f==null)throw new IllegalStateException("recorder already configured");
        factory=f;
    }
    public static String[] failures(){return failures.toArray(String[]::new);}
    public static void requireHealthy(){if(!failures.isEmpty())throw new IllegalStateException("invalid recorder run: "+failures);}
    private static void fail(Throwable t){failures.add(t.getClass().getName()+": "+t.getMessage());}
    /** Method belongs to this robot's isolated RobotMonitor, not the system loader's copy. */
    public static void attach(RobotController rc,Method counter) {
        if(factory==null)return;
        try {
            if(BOUND.get()!=null)throw new IllegalStateException("duplicate robot-thread binding");
            if(!counter.getName().equals("getBytecodeNum") || counter.getParameterCount()!=0 || counter.getReturnType()!=int.class)
                throw new IllegalArgumentException("wrong counter method");
            IntSupplier budget=()->{
                try { int value=(Integer)counter.invoke(null);if(value<0)throw new IllegalStateException("negative charged budget");return value; }
                catch(ReflectiveOperationException e){throw new IllegalStateException("cannot read robot budget",e);}
            };
            var deathFlag=counter.getDeclaringClass().getDeclaredField("shouldDie");
            deathFlag.setAccessible(true);
            BooleanSupplier dying=()->{try{return deathFlag.getBoolean(null);}catch(IllegalAccessException e){throw new IllegalStateException(e);}};
            Session session=factory.create(rc,budget);
            if(session==null)throw new IllegalStateException("missing session");
            PrimitiveRecorder.bind(session);BOUND.set(new Bound(rc,session,budget,dying));STARTED.set(false);
        } catch(Throwable t){fail(t);}
    }
    /** Initial scheduler pause has no completed turn and must not produce a fake record. */
    public static void pausing() {
        Bound b=BOUND.get();if(b==null || !STARTED.get())return;
        try {b.session.turnEnded(b.rc,b.budget.getAsInt(),PauseReason.current());STARTED.set(false);}catch(Throwable t){fail(t);}
    }
    public static void resumed(){
        Bound b=BOUND.get();if(b==null)return;
        try{STARTED.set(!b.dying.getAsBoolean());}catch(Throwable t){fail(t);}
    }
    /** Never prevent the sandbox's finally block from notifying its scheduler. */
    public static void detach(){detach(false,null);}
    public static void detach(boolean returned,Throwable failure) {
        Bound b=BOUND.get();
        if(b!=null) {
            try {String outcome=returned?"returned":failure==null?"unknown":
                failure.getClass().getName().equals("battlecode.instrumenter.RobotDeathException")?"robot_death_signal":"exception";
                b.session.terminated(b.rc,b.budget.getAsInt(),new Termination(outcome,failure==null?null:failure.getClass().getName(),STARTED.get()));}catch(Throwable t){fail(t);}
            try {PrimitiveRecorder.unbind();}catch(Throwable t){fail(t);}
        }
        BOUND.remove();STARTED.remove();
    }
    private RobotThreadRecorder(){}
}
