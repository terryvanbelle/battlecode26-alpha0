"""Pinned-engine integration checks; run after scripts/setup_engine.py."""
from pathlib import Path
import os
import json
import random
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "build/java/alpha0/EngineBridge.class").exists(), "run scripts/setup_engine.py first")
class EngineSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.process = subprocess.Popen(
            [str(Path(os.environ.get("JAVA_HOME", str(ROOT / "vendor/jdk"))) / "bin/java"), "-cp", "build/java:vendor/engine/battlecode26-java-1.2.6.jar", "alpha0.EngineBridge"],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    def tearDown(self):
        if self.process.poll() is None:
            self.request({"cmd": "close"})
            self.process.wait(timeout=10)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()

    def request(self, data):
        self.process.stdin.write(json.dumps(data) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        self.assertTrue(line, "engine exited without a response")
        result = json.loads(line)
        self.assertNotIn("error", result)
        return result

    def test_mutable_world_branches_equal_transcript_replay(self):
        reset = {"cmd": "reset", "map": "tiny", "seed": 42, "maps_dir": "maps"}
        state = self.request(reset)
        rng = random.Random(1)
        actions = []
        saw_spawn = False
        initial_ids = {robot["id"] for robot in state["robots"]}
        for index in range(400):
            if state["terminal"]:
                break
            action = rng.choice(state["legal_actions"])
            actions.append(action)
            if index % 40 == 0:
                snapshot = self.request({"cmd": "snapshot"})["snapshot"]
                branch = self.request({"cmd": "step", "action": action})
                self.request({"cmd": "restore", "snapshot": snapshot})
                state = self.request({"cmd": "step", "action": action})
                self.assertEqual(branch, state, f"restored branch differs at decision {index}")
                self.request({"cmd": "release", "snapshot": snapshot})
            else:
                state = self.request({"cmd": "step", "action": action})
            saw_spawn |= any(robot["id"] not in initial_ids for robot in state["robots"])
        replayed = self.request({**reset, "cmd": "replay", "actions": actions})
        self.assertEqual(state, replayed)
        self.assertTrue(saw_spawn, "trajectory must exercise new robot IDs and graph references")
        self.assertGreater(state["round"], 10, "trajectory must cross repeated round boundaries")


if __name__ == "__main__":
    unittest.main()
