package imitationmultifixture;
import battlecode.common.*;
public class RobotPlayer {
 public static void run(RobotController rc)throws GameActionException {
  boolean built=false;
  while(true){
   if(rc.getType().isRatKingType()){
    if(rc.getRoundNum()==10){rc.resign();return;}
    if(!built){
     MapLocation here=rc.getLocation();
     search:for(int dx=-4;dx<=4;dx++)for(int dy=-4;dy<=4;dy++){
      MapLocation target=new MapLocation(here.x+dx,here.y+dy);
      if(rc.canBuildRat(target)){rc.buildRat(target);built=true;break search;}
     }
    }
    rc.writeSharedArray(0,rc.getRoundNum());
   }
   rc.squeak(rc.getID());
   // Read public communication on every robot, exercising independent observation histories.
   rc.readSharedArray(0);rc.readSqueaks(-1);
   Clock.yield();
  }
 }
}
