package alpha0.imitation;

import battlecode.common.*;
import java.util.*;

/** Public API only. This is a raw capture, not a frozen network feature schema. */
public final class PublicObservation {
    static List<Integer> location(MapLocation p) {return p==null?null:List.of(p.x,p.y);}
    static Map<String,Object> robot(RobotInfo r,int depth) {
        if(r==null)return null;
        if(depth>64)throw new IllegalStateException("cyclic or excessive carrying chain");
        TreeMap<String,Object> out=new TreeMap<>();
        out.put("id",r.getID());out.put("team",r.getTeam().name());out.put("type",r.getType().name());
        out.put("health",r.getHealth());out.put("location",location(r.getLocation()));out.put("direction",r.getDirection().name());
        out.put("chirality",r.getChirality());out.put("raw_cheese",r.getRawCheeseAmount());out.put("carrying",robot(r.getCarryingRobot(),depth+1));
        return out;
    }
    public static Map<String,Object> capture(RobotController rc) throws GameActionException {
        TreeMap<String,Object> out=new TreeMap<>();
        out.put("schema","public-observation-draft-v1");out.put("robot_id",rc.getID());out.put("team",rc.getTeam().name());
        out.put("round",rc.getRoundNum());out.put("map_width",rc.getMapWidth());out.put("map_height",rc.getMapHeight());
        out.put("location",location(rc.getLocation()));out.put("direction",rc.getDirection().name());out.put("type",rc.getType().name());
        out.put("health",rc.getHealth());out.put("raw_cheese",rc.getRawCheese());out.put("global_cheese",rc.getGlobalCheese());out.put("all_cheese",rc.getAllCheese());out.put("dirt",rc.getDirt());
        out.put("cooperation",rc.isCooperation());Team back=rc.getBackstabbingTeam();out.put("backstabbing_team",back==null?null:back.name());
        out.put("rat_trap_count",rc.getNumberRatTraps());out.put("cat_trap_count",rc.getNumberCatTraps());
        out.put("action_cooldown",rc.getActionCooldownTurns());out.put("movement_cooldown",rc.getMovementCooldownTurns());out.put("turning_cooldown",rc.getTurningCooldownTurns());
        out.put("being_carried",rc.isBeingCarried());out.put("being_thrown",rc.isBeingThrown());out.put("carrying",robot(rc.getCarrying(),0));
        List<Object> parts=new ArrayList<>();for(MapLocation p:rc.getAllPartLocations())parts.add(location(p));out.put("parts",parts);
        List<Object> robots=new ArrayList<>();for(RobotInfo r:rc.senseNearbyRobots())robots.add(robot(r,0));out.put("visible_robots",robots);
        List<Object> tiles=new ArrayList<>();
        for(MapInfo m:rc.senseNearbyMapInfos()) {
            TreeMap<String,Object> tile=new TreeMap<>();tile.put("location",location(m.getMapLocation()));tile.put("passable",m.isPassable());tile.put("wall",m.isWall());tile.put("dirt",m.isDirt());tile.put("cheese",m.getCheeseAmount());tile.put("mine",m.hasCheeseMine());tile.put("trap",m.getTrap().name());tile.put("flying_robot",robot(m.flyingRobot(),0));tiles.add(tile);
        }
        out.put("visible_tiles",tiles);
        List<Object> messages=new ArrayList<>();
        for(Message m:rc.readSqueaks(-1)) {
            TreeMap<String,Object> message=new TreeMap<>();message.put("sender",m.getSenderID());message.put("round",m.getRound());message.put("content",m.getBytes());message.put("source",location(m.getSource()));messages.add(message);
        }
        out.put("messages",messages);
        List<Integer> shared=new ArrayList<>();for(int i=0;i<GameConstants.SHARED_ARRAY_SIZE;i++)shared.add(rc.readSharedArray(i));out.put("shared_array",shared);
        return Collections.unmodifiableMap(out);
    }
    private PublicObservation(){}
}
