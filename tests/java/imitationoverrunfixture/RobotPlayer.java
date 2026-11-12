package imitationoverrunfixture;
import battlecode.common.RobotController;
/** Deliberate exhaustion only; no teacher code or labels. */
public class RobotPlayer {
 public static volatile int counter;
 public static void run(RobotController rc){while(true)counter++;}
}
