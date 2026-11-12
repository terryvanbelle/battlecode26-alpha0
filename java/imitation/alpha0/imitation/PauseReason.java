package alpha0.imitation;

/** Inspect the uninstrumented caller path; never infer exhaustion from a numeric threshold. */
public final class PauseReason {
    public static String current(){return classify(Thread.currentThread().getStackTrace());}
    public static String classify(StackTraceElement[] stack) {
        for(int i=0;i+1<stack.length;i++) {
            var frame=stack[i];var caller=stack[i+1];
            if(frame.getClassName().equals("battlecode.instrumenter.inject.RobotMonitor") && frame.getMethodName().equals("pause")) {
                if(caller.getClassName().equals("battlecode.common.Clock") && caller.getMethodName().equals("yield"))return "yield";
                if(caller.getClassName().equals("battlecode.instrumenter.inject.RobotMonitor") && caller.getMethodName().equals("incrementBytecodes"))return "overrun";
                throw new IllegalStateException("unknown monitor pause caller: "+caller);
            }
        }
        throw new IllegalStateException("missing monitor pause frame");
    }
    private PauseReason(){}
}
