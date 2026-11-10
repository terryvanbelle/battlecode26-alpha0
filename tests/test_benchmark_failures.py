"""A crashed or unloadable opponent must never become a scored policy win."""
from pathlib import Path
import subprocess

import pytest

from alpha0.env import EngineEnv


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "build/java/alpha0/EngineBridge.class").exists(),
    reason="run scripts/setup_engine.py first",
)


@pytest.fixture
def fixture_bots(tmp_path):
    programs = {
        "badinit": "static int broken = 1 / Integer.parseInt(\"0\"); public static void run(RobotController rc) { while(true) Clock.yield(); }",
        "badreturn": "public static void run(RobotController rc) {}",
        "resigner": "public static void run(RobotController rc) { rc.resign(); while(true) Clock.yield(); }",
        "randomuser": "static java.util.Random rng = new java.util.Random(1); public static void run(RobotController rc) { while(true) { rng.nextInt(); Clock.yield(); } }",
    }
    sources = []
    for package, program in programs.items():
        path = tmp_path / package / "RobotPlayer.java"
        path.parent.mkdir()
        path.write_text(f"package {package}; import battlecode.common.*; public class RobotPlayer {{ {program} }}")
        sources.append(path)
    subprocess.run([
        str(ROOT / "vendor/jdk/bin/javac"), "--release", "21", "-cp",
        str(ROOT / "vendor/engine/battlecode26-java-1.2.6.jar"), "-d", str(tmp_path), *map(str, sources)
    ], check=True, capture_output=True, text=True)
    return tmp_path


@pytest.mark.parametrize("package", ["badinit", "badreturn"])
def test_invalid_opponent_is_error_not_a_win(fixture_bots, package):
    with EngineEnv(opponent_package=package, opponent_classes=fixture_bots, learned_team=1) as env:
        with pytest.raises(RuntimeError, match="Benchmark sandbox failure"):
            state = env.reset("tiny", 42)
            while not state.terminal:
                state = env.transition(state, 0)


def test_legitimate_resignation_remains_a_game_result(fixture_bots):
    with EngineEnv(opponent_package="resigner", opponent_classes=fixture_bots, learned_team=1) as env:
        state = env.reset("tiny", 42)
        while not state.terminal:
            state = env.transition(state, 0)
        assert state.raw["valid_game"] is True
        assert state.raw["benchmark_failures"] == []
        assert state.raw["winner"] == 1


def test_required_java_module_flags_and_missing_flag_detection(fixture_bots):
    command = [str(ROOT / "vendor/jdk/bin/java"), "-cp",
               f"{ROOT / 'build/java'}:{ROOT / 'vendor/engine/battlecode26-java-1.2.6.jar'}", "alpha0.EngineBridge"]
    with EngineEnv(command=command, opponent_package="randomuser", opponent_classes=fixture_bots, learned_team=1) as env:
        with pytest.raises(RuntimeError, match="Benchmark sandbox failure"):
            env.reset("tiny", 42)
    with EngineEnv(opponent_package="randomuser", opponent_classes=fixture_bots, learned_team=1) as env:
        state = env.reset("tiny", 42)
        assert not state.terminal
        assert state.raw["valid_game"] is True
        for _ in range(10):
            state = env.transition(state, 0)
        assert state.raw["round"] > 1
        assert state.raw["valid_game"] is True
