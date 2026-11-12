package imitationboundaryfixture;
import battlecode.common.*;
/** Synthetic deliberately invalid calls and transient communication state. */
public class RobotPlayer {
 public static void run(RobotController rc) throws GameActionException {
  while(true){
   rc.writeSharedArray(0,19);rc.writeSharedArray(0,17);
   if(!rc.squeak(42)||rc.squeak(43))throw new AssertionError("message limit changed");
   MapLocation absent=new MapLocation(-100,-100);
   try{rc.attack(absent);throw new AssertionError("attack unexpectedly legal");}catch(GameActionException expected){}
   try{rc.attack(absent,1);throw new AssertionError("attack amount unexpectedly legal");}catch(GameActionException expected){}
   try{rc.pickUpCheese(absent);throw new AssertionError("pickup unexpectedly legal");}catch(GameActionException expected){}
   try{rc.pickUpCheese(absent,1);throw new AssertionError("pickup amount unexpectedly legal");}catch(GameActionException expected){}
   Clock.yield();
  }
 }
}
