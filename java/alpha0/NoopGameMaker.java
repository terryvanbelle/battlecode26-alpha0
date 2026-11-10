package alpha0;

import battlecode.server.*;
import battlecode.common.*;
import battlecode.world.*;
import battlecode.instrumenter.profiler.*;
import java.io.*;
import java.util.*;

/** Recording-free engine sink: gameplay never reads replay data. */
public final class NoopGameMaker extends GameMaker {
    private final MatchMaker sink = new Sink();
    public NoopGameMaker(GameInfo info) { super(info, null, false); }
    @Override public MatchMaker getMatchMaker() { return sink; }
    public final class Sink extends MatchMaker {
        @Override public void makeMatchHeader(LiveMap gameMap) {}
        @Override public void makeMatchFooter(Team winTeam, DominationFactor winType, int totalRounds,
                List<ProfilerCollection> profilerCollections) {}
        @Override public void startRound(int roundNum) {}
        @Override public void endRound() {}
        @Override public void startTurn(int robotID) {}
        @Override public void endTurn(int robotID, int health, int cheese, int movementCooldown, int actionCooldown, int turningCooldown,
                int bytecodesUsed, MapLocation loc, Direction dir, boolean isCooperation) {}
        @Override public void addDamageAction(int damagedRobotID, int damage) {}
        @Override public void addRatNapAction(int nappedID) {}
        @Override public void addCatFeedAction(int sacrificedRatID) {}
        @Override public void addThrowAction(int thrownRobotID, MapLocation throwDirLocation) {}
        @Override public void addBiteAction(int biterID) {}
        @Override public void addScratchAction(int loc) {}
        @Override public void addStunAction(int robotID, int cooldown) {}
        @Override public void addBecomeRatKingAction(int id) {}
        @Override public void addPlaceTrapAction(int trapID, MapLocation loc, Team team, TrapType type) {}
        @Override public void addRemoveTrapAction(MapLocation loc, Team team) {}
        @Override public void addTrapTriggerAction(int trapID, MapLocation loc, Team team, TrapType type) {}
        @Override public void addPlaceDirtAction(MapLocation loc) {}
        @Override public void addRemoveDirtAction(MapLocation loc) {}
        @Override public void addSqueakAction(MapLocation loc) {}
        @Override public void addCheeseTransferAction(int toID, int amount) {}
        @Override public void addCheeseSpawnAction(MapLocation loc, int amount) {}
        @Override public void addCheesePickUpAction(MapLocation loc) {}
        @Override public void addSpawnAction(int id, MapLocation loc, Direction dir, int chirality, Team team, UnitType type) {}
        @Override public void addDieAction(int id, boolean fromException) {}
        @Override public void addTeamInfo(Team team, int cheeseTransferred, int catDamage, int aliveRatKings, int aliveBabyRats, int amountDirtCollected, int ratTrapCount, int catTrapCount) {}
        @Override public void addTimelineMarker(Team team, String label, int red, int green, int blue) {}
        @Override public void addIndicatorString(int id, String string) {}
        @Override public void addIndicatorDot(int id, MapLocation loc, int red, int green, int blue) {}
        @Override public void addIndicatorLine(int id, MapLocation startLoc, MapLocation endLoc, int red, int green, int blue) {}
        @Override public void addDied(int id) {}
        @Override public OutputStream getOut() { return OutputStream.nullOutputStream(); }
    }
}
