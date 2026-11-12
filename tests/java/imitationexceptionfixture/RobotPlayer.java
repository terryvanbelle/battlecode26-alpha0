package imitationexceptionfixture;
import battlecode.common.RobotController;
/** Synthetic lifecycle fixture only. */
public class RobotPlayer {public static void run(RobotController rc){throw new IllegalStateException("synthetic lifecycle exception");}}
