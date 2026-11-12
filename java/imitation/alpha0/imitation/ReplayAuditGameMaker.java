package alpha0.imitation;

import battlecode.server.*;
import battlecode.common.*;
import battlecode.world.*;
import battlecode.instrumenter.profiler.*;
import java.io.*;
import java.util.*;

/** Bounded independent replay-callback audit. Not a complete primitive-call trace. */
public final class ReplayAuditGameMaker extends GameMaker {
    private final java.util.List<Object> events = new java.util.ArrayList<>();
    private long bytes;
    private boolean failed;
    private Object value(Object v) {
        if(v instanceof MapLocation l)return java.util.List.of(l.x,l.y);
        if(v instanceof Enum<?> e)return e.name();
        if(v==null || v instanceof String || v instanceof Integer || v instanceof Boolean)return v;
        throw new IllegalArgumentException("unsupported replay argument");
    }
    private synchronized void record(String method,Object... args) {
        if(failed)throw new IllegalStateException("replay audit failed");
        try {
            java.util.List<Object> values=new java.util.ArrayList<>();
            for(Object a:args)values.add(value(a));
            Object row=java.util.Map.of("sequence",events.size(),"method",method,"arguments",values);
            long size=TraceJournal.canonical(row).length;
            if(size>1024*1024 || size>8L*1024*1024-bytes)throw new IllegalStateException("replay audit cap");
            events.add(row);bytes+=size;
        }catch(IOException e){failed=true;throw new UncheckedIOException(e);}
        catch(RuntimeException e){failed=true;throw e;}
    }
    public synchronized void save(java.nio.file.Path path) throws IOException {
        if(failed)throw new IllegalStateException("replay audit failed");
        java.nio.file.Files.write(path,TraceJournal.canonical(events),java.nio.file.StandardOpenOption.CREATE_NEW);
    }
    private final MatchMaker sink = new Sink();
    public ReplayAuditGameMaker(GameInfo info) { super(info, null, false); }
    @Override public MatchMaker getMatchMaker() { return sink; }
    public final class Sink extends MatchMaker {
        @Override public void makeMatchHeader(LiveMap gameMap) {}
        @Override public void makeMatchFooter(Team winTeam, DominationFactor winType, int totalRounds,
                List<ProfilerCollection> profilerCollections) {}
        @Override public void startRound(int roundNum) { record("startRound" ,roundNum); }
        @Override public void endRound() { record("endRound"); }
        @Override public void startTurn(int robotID) { record("startTurn" ,robotID); }
        @Override public void endTurn(int robotID, int health, int cheese, int movementCooldown, int actionCooldown, int turningCooldown,
                int bytecodesUsed, MapLocation loc, Direction dir, boolean isCooperation) { record("endTurn" ,robotID,health,cheese,movementCooldown,actionCooldown,turningCooldown,bytecodesUsed,loc,dir,isCooperation); }
        @Override public void addDamageAction(int damagedRobotID, int damage) { record("addDamageAction" ,damagedRobotID,damage); }
        @Override public void addRatNapAction(int nappedID) { record("addRatNapAction" ,nappedID); }
        @Override public void addCatFeedAction(int sacrificedRatID) { record("addCatFeedAction" ,sacrificedRatID); }
        @Override public void addThrowAction(int thrownRobotID, MapLocation throwDirLocation) { record("addThrowAction" ,thrownRobotID,throwDirLocation); }
        @Override public void addBiteAction(int biterID) { record("addBiteAction" ,biterID); }
        @Override public void addScratchAction(int loc) { record("addScratchAction" ,loc); }
        @Override public void addStunAction(int robotID, int cooldown) { record("addStunAction" ,robotID,cooldown); }
        @Override public void addBecomeRatKingAction(int id) { record("addBecomeRatKingAction" ,id); }
        @Override public void addPlaceTrapAction(int trapID, MapLocation loc, Team team, TrapType type) { record("addPlaceTrapAction" ,trapID,loc,team,type); }
        @Override public void addRemoveTrapAction(MapLocation loc, Team team) { record("addRemoveTrapAction" ,loc,team); }
        @Override public void addTrapTriggerAction(int trapID, MapLocation loc, Team team, TrapType type) { record("addTrapTriggerAction" ,trapID,loc,team,type); }
        @Override public void addPlaceDirtAction(MapLocation loc) { record("addPlaceDirtAction" ,loc); }
        @Override public void addRemoveDirtAction(MapLocation loc) { record("addRemoveDirtAction" ,loc); }
        @Override public void addSqueakAction(MapLocation loc) { record("addSqueakAction" ,loc); }
        @Override public void addCheeseTransferAction(int toID, int amount) { record("addCheeseTransferAction" ,toID,amount); }
        @Override public void addCheeseSpawnAction(MapLocation loc, int amount) { record("addCheeseSpawnAction" ,loc,amount); }
        @Override public void addCheesePickUpAction(MapLocation loc) { record("addCheesePickUpAction" ,loc); }
        @Override public void addSpawnAction(int id, MapLocation loc, Direction dir, int chirality, Team team, UnitType type) { record("addSpawnAction" ,id,loc,dir,chirality,team,type); }
        @Override public void addDieAction(int id, boolean fromException) { record("addDieAction" ,id,fromException); }
        @Override public void addTeamInfo(Team team, int cheeseTransferred, int catDamage, int aliveRatKings, int aliveBabyRats, int amountDirtCollected, int ratTrapCount, int catTrapCount) { record("addTeamInfo" ,team,cheeseTransferred,catDamage,aliveRatKings,aliveBabyRats,amountDirtCollected,ratTrapCount,catTrapCount); }
        @Override public void addTimelineMarker(Team team, String label, int red, int green, int blue) { record("addTimelineMarker" ,team,label,red,green,blue); }
        @Override public void addIndicatorString(int id, String string) { record("addIndicatorString" ,id,string); }
        @Override public void addIndicatorDot(int id, MapLocation loc, int red, int green, int blue) { record("addIndicatorDot" ,id,loc,red,green,blue); }
        @Override public void addIndicatorLine(int id, MapLocation startLoc, MapLocation endLoc, int red, int green, int blue) { record("addIndicatorLine" ,id,startLoc,endLoc,red,green,blue); }
        @Override public void addDied(int id) { record("addDied" ,id); }
        @Override public OutputStream getOut() { return OutputStream.nullOutputStream(); }
    }
}
