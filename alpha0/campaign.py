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

from .evaluate import atomic_json, evaluate_checkpoint
from .learn_v2 import Learner


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
        self.panel_ids = {bot["id"] for bot in gate["frozen_opponents"]}
        if len(self.panel_ids) != gate["required_opponents"]:
            raise ValueError("Victory gate must contain exactly its declared frozen opponents")
        overlap = set(self.config["maps"]) & set(gate["stages"]["holdout"]["maps"])
        if overlap:
            raise ValueError(f"Training maps overlap the reserved confirmation maps: {sorted(overlap)}")
        self.started = time.monotonic()
        self.deadline = self.started + hours * 3600
        self.cycle = {"started_at": now(), "budget_hours": hours, "pid": os.getpid(),
                      "experiment_id": self.config["experiment_id"], "all_opponents_beaten": False}
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
        report = evaluate_checkpoint(
            self.learner.checkpoint_path, iteration=self.learner.iteration,
            suite="smoke" if stage == "smoke" else "victory_gate", gate_stage=stage if stage != "smoke" else "standard",
            progress_dir=self.progress, jobs=self.config["workers"],
            timeout_seconds=self.config["game_timeout_seconds"],
            deadline=self.deadline, stop_file=self.stop_file)
        # Keep a small index for human review; authoritative reports retain identities.
        summary = {key: report[key] for key in ("iteration", "checkpoint_sha256", "evaluation_config_sha256",
                   "suite", "stage", "expected", "completed", "unresolved_errors", "full_panel_complete", "opponents")}
        summary["passed"] = report.get("passed", False)
        atomic_json(self.progress / f"latest-{stage}.json", summary)
        self.plots()
        return report

    def assessment_issue(self, report: dict) -> str | None:
        if not report["complete"]:
            return "evaluation_error" if report["unresolved_errors"] else "budget_or_stop"
        # A complete subset or malformed report must not advance the campaign.
        if not report.get("full_panel_complete") or set(report.get("opponents", {})) != self.panel_ids:
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
        if (iteration > 0 and iteration % self.config["standard_eval_every"] == 0) or promising:
            if self.stopped():
                return "budget_or_stop"
            standard = self.evaluation("standard")
            issue = self.assessment_issue(standard)
            if issue:
                return issue
            if standard["passed"]:
                if self.stopped():
                    return "budget_or_stop"
                holdout = self.evaluation("holdout")
                issue = self.assessment_issue(holdout)
                if issue:
                    return issue
                if holdout["passed"] and holdout["victory_gate_passed"]:
                    return "victory"
                # A failed held-out confirmation is consumed. An agent must design
                # and freeze a fresh disjoint confirmation before another candidate.
                return "holdout_consumed_review_required"
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
    if not 0 < args.hours <= 4:
        parser.error("A review cycle must be greater than zero and no longer than four hours")
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
