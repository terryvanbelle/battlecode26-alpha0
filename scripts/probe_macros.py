#!/usr/bin/env python3
"""Diagnostic rules-only macro controllers, NOT learned agents or training data.

Each cell runs in an isolated process group with a hard watchdog. Results live
under runs/macro-probe by default and are never merged into model progress.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONTROLLERS = ("assault", "economy", "balanced")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def distance(actor, other):
    parts = other.get("parts", [[other["x"], other["y"]]])
    return min((actor["x"] - x) ** 2 + (actor["y"] - y) ** 2 for x, y in parts)


def choose(raw, controller):
    """Fixed priorities from game rules; no benchmark identity or code is used."""
    actor = next(r for r in raw["robots"] if r["id"] == raw["current_id"])
    team = raw["current_team"]
    legal = set(raw["legal_actions"])
    friends = [r for r in raw["robots"] if r["team"] == team]
    babies = sum(r["type"] == "BABY_RAT" for r in friends)
    enemies = [r for r in raw["robots"] if r["team"] == 1 - team and not r.get("carried") and not r.get("thrown")]
    cats = [r for r in raw["robots"] if r["team"] == 2]
    enemy_distance = min((distance(actor, r) for r in enemies), default=10**9)
    cat_distance = min((distance(actor, r) for r in cats), default=10**9)
    bank = raw["bank"][team]
    if actor["type"] == "RAT_KING":
        if controller == "assault":
            priorities = [27, 26, 25, 17, 0] if bank >= 30 else [25, 17, 26, 0]
        elif controller == "economy":
            priorities = [26, 25, 17, 0] if babies < 12 and bank >= 50 else [25, 17, 0, 26]
        elif enemy_distance <= 8:
            priorities = [37, 28, 24, 27, 0] if bank >= 100 else [24, 28, 27, 0]
        else:
            priorities = [27, 26, 25, 17, 0] if babies < 24 and bank >= 50 else [25, 17, 0, 27]
    elif actor.get("carrying_id", -1) >= 0:
        priorities = [34, 35, 36, 21, 0]
    elif actor["cheese"] >= 10:
        priorities = [18, 17, 21, 0]
    elif controller == "assault":
        priorities = [21, 22, 17, 0]
    elif controller == "economy":
        priorities = [39, 19, 17, 0]
    elif actor["hp"] < 25 and min(enemy_distance, cat_distance) <= 8:
        priorities = [24, 18, 20, 0]
    elif enemy_distance <= 2:
        priorities = [37, 32, 28, 21, 22, 0] if bank >= 150 else [32, 28, 21, 22, 0]
    elif cat_distance <= 2:
        priorities = [29, 24, 23, 0]
    elif bank < 50 or actor["id"] % 4 == 0:
        priorities = [17, 19, 39, 21, 0]
    else:
        priorities = [21, 22, 17, 0]
    return next(action for action in priorities if action in legal)


def worker(spec):
    from alpha0.env import EngineEnv

    started = time.monotonic()
    env = None
    row = {**spec["identity"], "cell_id": spec["cell_id"], "status": "error", "score": None,
           "kind": "scripted-macro-diagnostic", "learned_model": False, "training_data": False}
    actions = []
    try:
        env = EngineEnv(opponent_package=spec["package"], opponent_classes=spec["classes"],
                        learned_team=0 if spec["identity"]["side"] == "A" else 1,
                        action_version="macro_v2", response_timeout=min(60, spec["timeout"]))
        state = env.reset(spec["identity"]["map"], spec["identity"]["seed"])
        while not state.terminal:
            remaining = spec["timeout"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("diagnostic game watchdog reached")
            env.response_timeout = min(60, max(1, remaining))
            action = choose(state.raw, spec["identity"]["controller"])
            actions.append(action)
            state = env.transition(state, action)
        if state.raw.get("valid_game") is not True or state.raw.get("benchmark_failures"):
            raise RuntimeError("engine did not certify a valid benchmark game")
        winner = state.raw["winner"]
        if winner not in (0, 1):
            raise RuntimeError(f"terminal engine response lacks a valid winner: {winner}")
        row.update(status="ok", score=float(winner == env.learned_team), winner=winner,
                   rounds=state.raw["round"], reason=state.raw.get("win_reason"),
                   kings=state.raw["kings"], bank=state.raw["bank"],
                   transferred=state.raw["transferred"], cat_damage=state.raw["cat_damage"],
                   final_king_health=[sum(r["hp"] for r in state.raw["robots"]
                                          if r["team"] == team and r["type"] == "RAT_KING") for team in (0, 1)])
    except Exception as exc:
        row.update(status="timeout" if isinstance(exc, TimeoutError) else "error", error=str(exc)[-14000:])
    finally:
        if env is not None:
            env.close()
    row.update(seconds=round(time.monotonic() - started, 3), decisions=len(actions),
               action_counts=dict(Counter(map(str, actions))), actions=actions)
    return row


def isolated(spec):
    started = time.monotonic()
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"],
                               cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(json.dumps(spec), timeout=spec["timeout"] + 5)
        if process.returncode:
            raise RuntimeError(f"worker exited {process.returncode}: {stderr[-6000:]}")
        return json.loads(stdout)
    except Exception as exc:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
        return {**spec["identity"], "cell_id": spec["cell_id"], "kind": "scripted-macro-diagnostic",
                "learned_model": False, "training_data": False,
                "status": "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "error",
                "score": None, "error": str(exc)[-8000:], "seconds": round(time.monotonic() - started, 3)}


def summary(rows, output):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["controller"], row["opponent"])].append(row)
    lines = ["# Scripted macro diagnostic", "", "These are fixed rules-only controllers, **not learned models**, and these games are **not training data**.",
             "One map/seed smoke results are diagnostic only; failures and timeouts are unscored.", "",
             "| Controller | Opponent | Wins / completed | Errors | Timeouts |", "|---|---|---:|---:|---:|"]
    aggregate = []
    for (controller, opponent), cells in sorted(groups.items()):
        ok = [r for r in cells if r["status"] == "ok"]
        wins = sum(r["score"] for r in ok)
        errors = sum(r["status"] == "error" for r in cells)
        timeouts = sum(r["status"] == "timeout" for r in cells)
        lines.append(f"| {controller} | {opponent} | {wins:g} / {len(ok)} | {errors} | {timeouts} |")
        aggregate.append(dict(controller=controller, opponent=opponent, wins=wins, completed=len(ok), errors=errors, timeouts=timeouts))
    (output / "README.md").write_text("\n".join(lines) + "\n")
    atomic_json(output / "summary.json", aggregate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=["assault", "balanced"])
    parser.add_argument("--opponents", nargs="+")
    parser.add_argument("--map", default="tiny")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/macro-probe")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin)), allow_nan=False))
        return
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    manifest = json.loads((ROOT / "configs/benchmarks.json").read_text())
    bots = [b for b in manifest["bots"] if args.opponents is None or b["id"] in args.opponents]
    if not bots or (args.opponents and set(args.opponents) - {b["id"] for b in bots}):
        parser.error("unknown or empty opponent selection")
    for bot in bots:
        class_file = ROOT / "build/benchmarks" / bot["id"] / Path(bot["package"].replace(".", "/")) / "RobotPlayer.class"
        if not class_file.exists():
            raise FileNotFoundError(f"Compile benchmarks first: missing {class_file}")
    classes = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "build/java").rglob("*.class"))}
    common = {"controller_script_sha256": digest(__file__), "bridge_classes_sha256": fingerprint(classes),
              "engine_jar_sha256": digest(ROOT / "vendor/engine/battlecode26-java-1.2.6.jar"),
              "map": args.map, "map_sha256": digest(ROOT / "maps" / f"{args.map}.map26"),
              "seed": args.seed, "action_version": "macro_v2", "benchmark_suite_id": manifest["suite_id"],
              "watchdog_seconds": args.timeout}
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "identity.json", {**common, "controllers": args.controllers,
                "purpose": "diagnostic fixed rules-only macro capability; not learned-policy evaluation or training"})
    specs, rows = [], []
    for controller in args.controllers:
        for bot in bots:
            for side in ("A", "B"):
                identity = {**common, "controller": controller, "opponent": bot["id"], "side": side,
                            "opponent_source_commit": bot["source_commit"], "opponent_java_sha256": bot["java_tree_sha256"]}
                cell_id = fingerprint(identity)
                path = args.output / "cells" / f"{cell_id}.json"
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                    continue
                specs.append({"cell_id": cell_id, "identity": identity, "timeout": args.timeout,
                              "package": bot["package"], "classes": str(ROOT / "build/benchmarks" / bot["id"])})
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(isolated, spec): spec for spec in specs}
        for future in as_completed(futures):
            row = future.result()
            atomic_json(args.output / "cells" / f"{row['cell_id']}.json", row)
            rows.append(row)
            with (args.output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            summary(rows, args.output)
            print(json.dumps({key: row.get(key) for key in ("controller", "opponent", "side", "status", "score", "rounds", "seconds")}), flush=True)
    summary(rows, args.output)


if __name__ == "__main__":
    main()
