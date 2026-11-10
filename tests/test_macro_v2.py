"""Macro-v2 integration regressions against real official-controller effects."""
from pathlib import Path
import json
import random
import subprocess
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENGINE_CP = f"{ROOT / 'build/java'}:{ROOT / 'vendor/engine/battlecode26-java-1.2.6.jar'}"
pytestmark = pytest.mark.skipif(
    not (ROOT / "build/java/alpha0/MacroV2.class").exists(),
    reason="compile the macro_v2 engine with scripts/setup_engine.py first",
)


FOOTPRINT_HARNESS = r'''
package alpha0;

import battlecode.common.*;
import battlecode.crossplay.CrossPlayLanguage;
import battlecode.server.*;
import battlecode.world.*;
import battlecode.world.control.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.*;

public final class MacroV2FootprintTest {
    private static RobotInfo info(int id, Team team, UnitType type, int x, int y) {
        return new RobotInfo(id,team,type,type.health,new MapLocation(x,y),Direction.EAST,0,0,null);
    }
    private static EngineBridge.Session fixture(boolean transfer) throws Exception {
        EngineBridge.Session s=new EngineBridge.Session(new ObjectMapper().readTree(
            "{\"map\":\"tiny\",\"maps_dir\":\"maps\",\"seed\":42,\"action_version\":\"macro_v2\"}"));
        RobotInfo[] initial = transfer
            ? new RobotInfo[]{info(10001,Team.A,UnitType.RAT_KING,10,10),info(10002,Team.B,UnitType.RAT_KING,16,16),info(10003,Team.A,UnitType.BABY_RAT,6,10)}
            : new RobotInfo[]{info(10001,Team.A,UnitType.RAT_KING,4,4),info(10002,Team.B,UnitType.RAT_KING,12,10),info(10003,Team.A,UnitType.BABY_RAT,10,10)};
        int w=20,h=20,n=w*h;
        LiveMap map=new LiveMap(w,h,new MapLocation(0,0),42,2000,"macro-fixture",MapSymmetry.ROTATIONAL,
            new boolean[n],new boolean[n],new boolean[n],new int[n],new ArrayList<>(),new ArrayList<>(),initial);
        TeamControlProvider controls=new TeamControlProvider();
        for(Team team:Team.values()) controls.registerControlProvider(team,new NullControlProvider());
        GameInfo gameInfo=new GameInfo("alpha0",CrossPlayLanguage.JAVA,"alpha0",null,"opponent",CrossPlayLanguage.JAVA,"opponent",null,new String[]{"macro-fixture"},null,false);
        s.world=new GameWorld(map,controls,new NoopGameMaker(gameInfo).getMatchMaker());
        s.controls=controls;
        s.current=s.world.getObjectInfo().getRobotByID(10003);
        s.world.processBeginningOfRound();
        s.current.processBeginningOfTurn();
        // The regression must exercise footprint targeting, not move closer until
        // the old incorrect centre-only target happens to become reachable.
        s.current.setMovementCooldownTurns(100);
        return s;
    }
    private static void attack() throws Exception {
        EngineBridge.Session s=fixture(false);
        InternalRobot king=s.world.getObjectInfo().getRobotByID(10002);
        RobotControllerImpl rc=s.current.getController();
        if(rc.canAttack(king.getLocation())) throw new AssertionError("Fixture centre must be out of bite range");
        if(!rc.canAttack(new MapLocation(11,10))) throw new AssertionError("Fixture king perimeter must be attackable");
        int before=king.getHealth();
        if(!MacroV2.legal(s).contains(21)) throw new AssertionError("Assault-king macro should be legal");
        MacroV2.execute(s,21);
        if(king.getHealth()>=before) throw new AssertionError("Baby failed to attack the king's legal perimeter tile");
        if(!s.current.getLocation().equals(new MapLocation(10,10))) throw new AssertionError("Attack fixture unexpectedly moved");
    }
    private static void transfer() throws Exception {
        EngineBridge.Session s=fixture(true);
        InternalRobot king=s.world.getObjectInfo().getRobotByID(10001);
        s.current.addCheese(20);
        RobotControllerImpl rc=s.current.getController();
        if(rc.canTransferCheese(king.getLocation(),20)) throw new AssertionError("Fixture king centre must be out of transfer range");
        if(!rc.canTransferCheese(new MapLocation(9,10),20)) throw new AssertionError("Fixture king perimeter must accept cheese");
        int before=s.world.getTeamInfo().getCheese(Team.A);
        if(!MacroV2.legal(s).contains(18)) throw new AssertionError("Return-cheese macro should be legal");
        MacroV2.execute(s,18);
        if(s.current.getCheese()!=0 || s.world.getTeamInfo().getCheese(Team.A)!=before+20)
            throw new AssertionError("Cheese was not transferred through the king's perimeter tile");
        if(!s.current.getLocation().equals(new MapLocation(6,10))) throw new AssertionError("Transfer fixture unexpectedly moved");
    }
    public static void main(String[] args) throws Exception {
        if(args[0].equals("attack")) attack(); else if(args[0].equals("transfer")) transfer(); else throw new AssertionError("Unknown case");
        System.out.println("PASS "+args[0]);
    }
}
'''


@pytest.fixture(scope="module")
def footprint_harness(tmp_path_factory):
    directory = tmp_path_factory.mktemp("macro-v2-footprints")
    source = directory / "MacroV2FootprintTest.java"
    source.write_text(FOOTPRINT_HARNESS)
    subprocess.run([
        str(ROOT / "vendor/jdk/bin/javac"), "--release", "21", "-cp", ENGINE_CP,
        "-d", str(directory), str(source),
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    return directory


@pytest.mark.parametrize("scenario", ["attack", "transfer"])
def test_macro_v2_targets_full_robot_footprints(footprint_harness, scenario):
    result = subprocess.run([
        str(ROOT / "vendor/jdk/bin/java"), "-cp", f"{footprint_harness}:{ENGINE_CP}",
        "alpha0.MacroV2FootprintTest", scenario,
    ], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS {scenario}" in result.stdout


def test_macro_v2_snapshot_branch_matches_full_replay():
    with tempfile.TemporaryFile(mode="w+b") as error_log:
        process = subprocess.Popen([
            str(ROOT / "vendor/jdk/bin/java"), "-cp", ENGINE_CP, "alpha0.EngineBridge",
        ], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error_log, text=True)

        def request(message):
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            assert line, "engine exited without a response"
            response = json.loads(line)
            assert "error" not in response, response
            return response

        try:
            reset = {"cmd": "reset", "map": "tiny", "maps_dir": "maps", "seed": 42, "action_version": "macro_v2"}
            state = request(reset)
            rng = random.Random(789)
            actions = []
            semantic_actions = 0
            for index in range(300):
                if state["terminal"]:
                    break
                assert all(0 <= action < 40 for action in state["legal_actions"])
                choices = state["legal_actions"]
                # Exercise the new target-based macros as well as directional ones.
                semantic_choices = [action for action in choices if action >= 17]
                action = rng.choice(semantic_choices if index % 2 == 0 and semantic_choices else choices)
                semantic_actions += action >= 17
                actions.append(action)
                if index % 30 == 0:
                    snapshot = request({"cmd": "snapshot"})["snapshot"]
                    expected = request({"cmd": "step", "action": action})
                    request({"cmd": "restore", "snapshot": snapshot})
                    state = request({"cmd": "step", "action": action})
                    assert state == expected, f"branch diverged at action {index}"
                    request({"cmd": "release", "snapshot": snapshot})
                else:
                    state = request({"cmd": "step", "action": action})
            assert semantic_actions >= 10, "trajectory did not exercise enough v2 semantic actions"
            replayed = request({**reset, "cmd": "replay", "actions": actions})
            assert replayed == state
        finally:
            if process.poll() is None:
                request({"cmd": "close"})
                process.wait(timeout=10)
            process.stdin.close()
            process.stdout.close()
