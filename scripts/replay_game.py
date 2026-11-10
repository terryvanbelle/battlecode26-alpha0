#!/usr/bin/env python3
"""Replay a recorded action transcript and verify its terminal engine state."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpha0.env import EngineEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game", type=Path)
    args = parser.parse_args()
    record = json.loads(args.game.read_text())
    if record["status"] != "ok":
        parser.error("only a completed game has a verifiable terminal state")
    action_version = record.get("action_version", "macro_v1")
    with EngineEnv(action_version=action_version) as env:
        actual = env._request({"cmd": "replay", "map": record["map"], "seed": record["seed"],
                               "maps_dir": env.maps_dir, "actions": record["actions"],
                               "action_version": action_version})
    expected = record["terminal"]
    diagnostic_keys = {"snapshot", "valid_game", "benchmark_failures", "diagnostics"}
    differences = [key for key in expected if key not in diagnostic_keys and expected[key] != actual.get(key)]
    if differences or not actual["terminal"]:
        raise SystemExit("Replay mismatch in: " + ", ".join(differences or ["terminal"]))
    print(json.dumps({"verified": True, "map": record["map"], "seed": record["seed"],
                      "decisions": len(record["actions"]), "rounds": actual["round"],
                      "winner": actual["winner"]}))


if __name__ == "__main__":
    main()
