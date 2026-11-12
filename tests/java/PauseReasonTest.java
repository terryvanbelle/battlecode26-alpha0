import alpha0.imitation.PauseReason;
public class PauseReasonTest {
 static StackTraceElement f(String c,String m){return new StackTraceElement(c,m,"fixture",1);}
 public static void main(String[] args){
  var pause=f("battlecode.instrumenter.inject.RobotMonitor","pause");
  if(!PauseReason.classify(new StackTraceElement[]{pause,f("battlecode.common.Clock","yield")}).equals("yield"))throw new AssertionError();
  if(!PauseReason.classify(new StackTraceElement[]{pause,f("battlecode.instrumenter.inject.RobotMonitor","incrementBytecodes"),f("battlecode.common.Clock","yield")}).equals("overrun"))throw new AssertionError("nearest caller must win");
  for(var stack:new StackTraceElement[][]{new StackTraceElement[]{},new StackTraceElement[]{pause,f("unknown","pause")},new StackTraceElement[]{f("battlecode.common.Clock","yield")}}){
   try{PauseReason.classify(stack);throw new AssertionError("unknown accepted");}catch(IllegalStateException expected){}
  }
 }
}
