"""Bounded, resumable research cycles. A finished cycle is not a victory claim."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .evaluate import atomic_json, evaluate_checkpoint, fingerprint
from .gate_config import training_excluded_maps
from .learn_v2 import Learner
from .victory import evaluate_victory


def now():
    return datetime.now(timezone.utc).isoformat()


class Campaign:
    def __init__(self, root: Path, hours: float, *, config_path: Path | None = None):
        self.root = root.resolve()
        self.directory = self.root / "runs/campaign"
        self.run = self.directory / "v2"
        self.progress = self.root / "progress/campaign-v2"
        self.stop_file = self.directory / "STOP"
        self.config = json.loads((config_path or self.root / "configs/campaign-v2.json").read_text())
        gate = json.loads((self.root / "configs/victory_gate.json").read_text())
        if gate.get("schema_version") != 2:
            raise ValueError("Campaign requires the current all-map victory gate")
        self.gate_id, self.gate_hash = gate["gate_id"], fingerprint(gate)
        self.panel_ids = {bot["id"] for bot in gate["frozen_opponents"]}
        self.map_names = {entry["name"] for entry in gate["frozen_maps"]}
        if len(self.panel_ids) != gate["required_opponents"]:
            raise ValueError("Victory gate must contain exactly its declared frozen opponents")
        if len(self.map_names) != gate["required_maps"]:
            raise ValueError("Victory gate must contain exactly its declared frozen maps")
        overlap = set(self.config["maps"]) & training_excluded_maps(gate)
        if overlap:
            raise ValueError(f"Training maps overlap the reserved confirmation maps: {sorted(overlap)}")
        self.started = time.monotonic()
        self.deadline = self.started + hours * 3600
        self.cycle = {"started_at": now(), "budget_hours": hours, "pid": os.getpid(),
                      "experiment_id": self.config["experiment_id"], "all_opponents_beaten": False,
                      "gate_id": self.gate_id, "gate_config_sha256": self.gate_hash,
                      "required_maps": len(self.map_names),
                      "required_opponent_map_pairs": len(self.panel_ids) * len(self.map_names)}
        self.learner = Learner(self.run, self.config, self.progress)

    def status(self, phase: str, **extra):
        self.cycle.update(phase=phase, updated_at=now(), iteration=self.learner.iteration,
                          checkpoint=str(self.learner.checkpoint_path.relative_to(self.root)),
                          games_total=self.learner.games_total, **extra)
        atomic_json(self.directory / "status.json", self.cycle)
        # Public state does not depend on an ignored local run directory.
        atomic_json(self.progress / "status.json", self.cycle)
        print(json.dumps(self.cycle, allow_nan=False), flush=True)

    def stopped(self):
        return self.stop_file.exists() or time.monotonic() >= self.deadline

    def plots(self):
        subprocess.run([sys.executable, str(self.root / "scripts/plot_progress.py"),
                        "--progress-dir", str(self.progress)], cwd=self.root, check=True)

    def publish_recovery(self):
        """Publish only while no learner/evaluation workers are running.

        Publication failure leaves local progress intact and is visible in status;
        it must not turn a valid game into a loss or discard a checkpoint.
        """
        script = self.root / "scripts/campaign_snapshot.py"
        try:
            created = subprocess.run([sys.executable, str(script), "create", "--root", str(self.root),
                                      "--checkpoint-boundary"], check=True, capture_output=True, text=True,
                                     timeout=600)
            archive = json.loads(created.stdout)["archive"]
            published = subprocess.run([sys.executable, str(script), "publish", "--root", str(self.root),
                                        "--archive", archive], check=True, capture_output=True, text=True,
                                       timeout=600)
            pointer = json.loads(published.stdout)
            published_archive, sha256 = pointer["archive"], pointer["sha256"]
            if not isinstance(published_archive, str) or not isinstance(sha256, str):
                raise ValueError("Snapshot publication returned an invalid archive or SHA256")
            self.cycle["last_published_recovery"] = published_archive
            self.cycle.pop("recovery_publication_error", None)
            print(json.dumps({"recovery_published": published_archive, "sha256": sha256}), flush=True)
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as error:
            self.cycle["recovery_publication_error"] = str(error)
            print(json.dumps({"recovery_publication_error": str(error)}), flush=True)
        finally:
            # The final boundary has no subsequent phase transition to persist this.
            self.status(self.cycle.get("phase", "starting"))

    def evaluation(self, stage: str):
        self.status("evaluating", evaluation_stage=stage)
        arguments = dict(iteration=self.learner.iteration, progress_dir=self.progress,
                         jobs=self.config["workers"], timeout_seconds=self.config["game_timeout_seconds"],
                         deadline=self.deadline, stop_file=self.stop_file)
        if stage == "smoke":
            report = evaluate_checkpoint(self.learner.checkpoint_path, suite="smoke", **arguments)
        else:
            report = evaluate_victory(self.learner.checkpoint_path, stage=stage, root=self.root,
                manifest_path=self.root / "configs/benchmarks.json", maps_dir=self.root / "maps",
                gate_path=self.root / "configs/victory_gate.json",
                gate_ledger_dir=self.root / "runs/gate-ledgers", **arguments)
        # Keep a small index for human review; authoritative reports retain identities.
        summary = {key: report[key] for key in ("iteration", "checkpoint_sha256", "evaluation_config_sha256",
                   "suite", "stage", "expected", "completed", "unresolved_errors", "full_panel_complete", "opponents")}
        summary["passed"] = report.get("passed", False)
        for key in ("gate_id", "gate_config_sha256", "map_inventory_sha256", "all_maps_complete",
                    "victory_gate_passed", "confirmation_ledger"):
            if key in report:
                summary[key] = report[key]
        atomic_json(self.progress / f"latest-{stage}.json", summary)
        self.plots()
        return report

    def assessment_issue(self, report: dict, stage: str = "smoke") -> str | None:
        if not report["complete"]:
            return "evaluation_error" if report["unresolved_errors"] else "budget_or_stop"
        # A complete subset or malformed report must not advance the campaign.
        if not report.get("full_panel_complete") or set(report.get("opponents", {})) != self.panel_ids:
            return "evaluation_error"
        if report["unresolved_errors"]:
            return "evaluation_error"
        if stage != "smoke" and (
            report.get("gate_id") != self.gate_id or report.get("gate_config_sha256") != self.gate_hash or
            not report.get("all_maps_complete") or
            any(set(opponent.get("maps", {})) != self.map_names for opponent in report["opponents"].values())
        ):
            return "evaluation_error"
        return None

    def assess(self) -> str:
        """Resume the current checkpoint's required evaluation before training again."""
        iteration = self.learner.iteration
        if iteration % self.config["smoke_eval_every"] == 0:
            smoke = self.evaluation("smoke")
            issue = self.assessment_issue(smoke)
            if issue:
                return issue
            # A strong smoke result can justify an early standard evaluation, but is
            # never itself a victory criterion.
            promising = all(s["mean_score"] >= 0.5 for s in smoke["opponents"].values())
        else:
            promising = False
        # The expanded 3,552-game qualification is justified only after every
        # opponent passes the cheap screening floor. A periodic weak checkpoint
        # does not warrant thousands of games; smoke itself never grants victory.
        if promising:
            if self.stopped():
                return "budget_or_stop"
            standard = self.evaluation("standard")
            issue = self.assessment_issue(standard, "standard")
            if issue:
                return issue
            if standard["passed"]:
                if self.stopped():
                    return "budget_or_stop"
                confirmation = self.evaluation("confirmation")
                issue = self.assessment_issue(confirmation, "confirmation")
                if issue:
                    return issue
                same_candidate = all(standard.get(key) is not None and standard[key] == confirmation.get(key)
                    for key in ("checkpoint_sha256", "evaluation_config_sha256"))
                if not same_candidate:
                    return "evaluation_error"
                if confirmation["passed"] and confirmation["victory_gate_passed"]:
                    return "victory"
                # A failed confirmation consumes its seeds. Freeze a fresh unused
                # seed set on the same complete map inventory for a later candidate.
                return "confirmation_consumed_review_required"
        return "continue"

    def run_cycle(self):
        initial_iteration = self.learner.iteration
        self.status("starting", initial_iteration=initial_iteration)
        outcome = "budget_or_stop"
        while not self.stopped():
            outcome = self.assess()
            self.publish_recovery()
            if outcome != "continue":
                break
            if self.learner.iteration - initial_iteration >= self.config["max_iterations_per_cycle"]:
                outcome = "cycle_complete_review_required"
                break
            if self.stopped():
                outcome = "budget_or_stop"
                break
            self.status("self_play_and_training", evaluation_stage=None)
            summary = self.learner.train_iteration(self.deadline, self.stop_file)
            if summary is None:
                outcome = "budget_or_stop"
                break
            self.plots()
            self.status("checkpoint_ready", evaluation_stage=None)
        self.status("victory" if outcome == "victory" else "paused", reason=outcome,
                    all_opponents_beaten=outcome == "victory", ended_at=now(),
                    elapsed_seconds=time.monotonic() - self.started)
        self.plots()
        self.publish_recovery()
        return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if not 0 < args.hours <= 3:
        parser.error("A review cycle must be greater than zero and no longer than three hours")
    campaign = None
    try:
        campaign = Campaign(args.root, args.hours, config_path=args.config)
        campaign.run_cycle()
    except KeyboardInterrupt:
        if campaign:
            campaign.status("interrupted", reason="keyboard_interrupt")
        raise
    except Exception as error:
        if campaign:
            campaign.status("failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
