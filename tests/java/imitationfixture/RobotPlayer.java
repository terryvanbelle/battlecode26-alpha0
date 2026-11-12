package imitationfixture;
import battlecode.common.*;
public class RobotPlayer {
 public static void run(RobotController rc) throws GameActionException {
  while(true){rc.writeSharedArray(0,17);rc.squeak(42);Clock.yield();}
 }
}
