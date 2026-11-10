#!/usr/bin/env python3
"""Export small, reviewable experiment artifacts without publishing replay buffers."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import shutil
import subprocess

import numpy
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/initial")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(run.glob("checkpoint-*.pt"))
    if not checkpoints:
        parser.error("no committed checkpoints in this run")
    exports = [*checkpoints, run / "config.json"]
    files = {}
    for source in dict.fromkeys(exports):
        destination = output / source.name
        shutil.copyfile(source, destination)
        files[destination.name] = sha(destination)
    games = []
    for path in sorted((run / "games").glob("*.json")):
        game = json.loads(path.read_text())
        summary = {key: value for key, value in game.items() if key not in ("actions", "terminal")}
        summary.update(file=path.name, sha256=sha(path), decisions=len(game.get("actions", [])))
        games.append(summary)
    # Action transcripts are compact and let the official bridge reproduce every game.
    transcripts = output / "games"
    transcripts.mkdir(exist_ok=True)
    for path in sorted((run / "games").glob("*.json")):
        shutil.copyfile(path, transcripts / path.name)
        files[f"games/{path.name}"] = sha(path)
    engine = json.loads((ROOT / "build/engine-manifest.json").read_text())
    engine.pop("jdk_used", None)
    metadata = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "config": json.loads((run / "config.json").read_text()),
                "python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
                "platform": platform.platform(), "engine": engine, "files": files, "games": games,
                "note": "Initial CPU experiment. All published checkpoints include optimizer/RNG metadata; full replay buffers stay in the local run directory. Engine metadata describes the validated final bridge; initialization/death diagnostic fixes during setup did not affect self-play macro mechanics."}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {len(games)} game transcripts and {len(checkpoints)} checkpoints to {output}")


if __name__ == "__main__":
    main()
