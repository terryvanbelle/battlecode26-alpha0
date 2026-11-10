"""Offline root-ranking report safeguards; all reports and images are fixtures.

These tests do not run the engine, score a policy, update parameters, or write to
the repository's production progress directory.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("root_ranking_plot", ROOT / "scripts/plot_root_ranking.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def root_fixture(index, delta=0, *, same_action=False):
    """Four terminal outcomes with differences in eighth-point increments."""
    steps = round(abs(delta) * 8)
    assert abs(steps / 8 - abs(delta)) < 1e-12 and steps <= 4
    rollout_scores = [.5] * 4
    for i in range(steps):
        rollout_scores[i] = 1 if delta > 0 else 0
    puct_scores = rollout_scores.copy() if same_action else [.5] * 4
    rollout_mean, puct_mean = sum(rollout_scores) / 4, sum(puct_scores) / 4
    puct_action = 4 if same_action else 21
    unique_cells = 4 if same_action else 8
    return {
        "root_id": f"root-{index:02d}", "map": "fixture-map", "role": "BABY_RAT", "round": 12,
        "legal_actions": [4, 21],
        "selection": {"complete": True, "expected": 4, "completed": 4, "errors": 0,
                      "action_scores": {"4": .75, "21": .25}, "selected_action": 4},
        "comparator": {"selected_action": puct_action},
        "validation": {
            "complete": True, "seeds": list(P.VALIDATION_SEEDS),
            "expected_unique_cells": unique_cells, "completed_unique_cells": unique_cells,
            "errors": 0, "rollout_action": 4, "puct_action": puct_action,
            "rollout_scores": rollout_scores, "puct_scores": puct_scores,
            "rollout_mean": rollout_mean, "puct_mean": puct_mean, "delta": rollout_mean - puct_mean,
        },
    }


def report_fixture(deltas=(.25, .25, 0, 0, 0), *, same_action=False):
    roots = [root_fixture(i, delta, same_action=same_action) for i, delta in enumerate(deltas)]
    report = {
        "schema_version": 1, "experiment_id": "terminal-root-ranking-v4",
        "identity_sha256": "a" * 64, "phase": "fixture", "roots_frozen": True,
        "gradient_updates": 0, "benchmark_data_used": False, "roots": roots,
        "totals": {"attempt_errors": 0},
    }
    record_decision(report)
    return report


def record_decision(report):
    """Supply independent expected metadata after changing a test fixture."""
    roots = report["roots"]
    complete = bool(roots) and report["roots_frozen"] and all(r["validation"]["complete"] for r in roots)
    deltas = [r["validation"]["delta"] for r in roots] if complete else []
    mean = sum(deltas) / len(deltas) if complete else None
    improved = sum(delta > 0 for delta in deltas) if complete else None
    worst = min(deltas) if complete else None
    criteria_pass = complete and mean >= .1 and improved >= 2 and worst >= -.25
    report["complete"] = complete
    report["decision"] = {
        "thresholds": dict(P.THRESHOLDS), "complete": complete,
        "criteria_pass": criteria_pass, "integrity_verified": True, "passes": criteria_pass,
        "mean_delta": mean, "improved_roots": improved, "worst_root_delta": worst,
    }


def make_unresolved(root, *, error=False):
    validation = root["validation"]
    validation.update(complete=False, completed_unique_cells=validation["expected_unique_cells"] - 1,
                      errors=int(error), rollout_scores=[], puct_scores=[],
                      rollout_mean=None, puct_mean=None, delta=None)


class DecisionTests(unittest.TestCase):
    def test_exact_aggregate_boundary_and_two_distinct_improved_roots_pass(self):
        view = P.report_view(report_fixture())
        self.assertEqual(view["state"], "PASS")
        self.assertEqual(view["mean_delta"], .1)
        self.assertEqual(view["improved_roots"], 2)
        self.assertEqual(view["worst_root_delta"], 0)
        self.assertEqual(view["complete_roots"], 5)

    def test_passing_scores_await_integrity_verification_before_any_pass_claim(self):
        report = report_fixture()
        report["decision"].update(integrity_verified=False, passes=False)
        view = P.report_view(report)
        self.assertTrue(view["complete"])
        self.assertTrue(view["criteria_pass"])
        self.assertFalse(view["integrity_verified"])
        self.assertFalse(view["passes"])
        self.assertEqual(view["state"], "AWAITING VERIFICATION")
        report["decision"]["passes"] = True
        with self.assertRaisesRegex(ValueError, "Recorded decision"):
            P.report_view(report)

    def test_verified_failing_scores_remain_fail(self):
        view = P.report_view(report_fixture((0, 0)))
        self.assertTrue(view["complete"])
        self.assertTrue(view["integrity_verified"])
        self.assertFalse(view["criteria_pass"])
        self.assertFalse(view["passes"])
        self.assertEqual(view["state"], "FAIL")

    def test_improved_root_count_is_not_winning_seed_count(self):
        view = P.report_view(report_fixture((.5, 0, 0, 0, 0)))
        self.assertEqual(view["mean_delta"], .1)
        self.assertEqual(view["improved_roots"], 1)
        self.assertEqual(view["state"], "FAIL")

    def test_worst_root_boundary_is_inclusive_and_independent_of_positive_mean(self):
        at_floor = P.report_view(report_fixture((-.25, .5, .5)))
        self.assertEqual(at_floor["worst_root_delta"], -.25)
        self.assertEqual(at_floor["state"], "PASS")
        below_floor = P.report_view(report_fixture((-.375, .5, .5)))
        self.assertGreater(below_floor["mean_delta"], .1)
        self.assertEqual(below_floor["improved_roots"], 2)
        self.assertEqual(below_floor["state"], "FAIL")

    def test_two_improved_roots_do_not_override_low_aggregate(self):
        view = P.report_view(report_fixture((.125, .125, 0)))
        self.assertEqual(view["improved_roots"], 2)
        self.assertLess(view["mean_delta"], .1)
        self.assertEqual(view["state"], "FAIL")

    def test_same_action_reuses_four_unique_runs_and_never_improves(self):
        report = report_fixture((.25, .5), same_action=True)
        view = P.report_view(report)
        self.assertEqual(view["validation_expected_unique"], 8)
        self.assertEqual(view["validation_completed_unique"], 8)
        self.assertEqual(view["mean_delta"], 0)
        self.assertEqual(view["improved_roots"], 0)
        self.assertEqual(view["state"], "FAIL")
        report["roots"][0]["validation"].update(expected_unique_cells=8, completed_unique_cells=8)
        with self.assertRaisesRegex(ValueError, "unique validation count"):
            P.report_view(report)

    def test_same_action_conflicting_outcomes_cannot_be_scored(self):
        report = report_fixture((0,), same_action=True)
        report["roots"][0]["validation"]["puct_scores"][0] = 0
        with self.assertRaisesRegex(ValueError, "same action.*conflicting"):
            P.report_view(report)

    def test_missing_or_error_root_has_no_mean_and_cannot_complete_gate(self):
        for error in (False, True):
            with self.subTest(error=error):
                report = report_fixture((.5, .5))
                make_unresolved(report["roots"][1], error=error)
                record_decision(report)
                view = P.report_view(report)
                self.assertEqual(view["state"], "INCOMPLETE")
                self.assertEqual(view["complete_roots"], 1)
                for key in ("mean_delta", "improved_roots", "worst_root_delta"):
                    self.assertIsNone(view[key])
                for key in ("rollout_mean", "puct_mean", "delta"):
                    self.assertIsNone(view["roots"][1][key])

    def test_incomplete_selection_cannot_lock_an_action(self):
        report = report_fixture((.5,))
        root = report["roots"][0]
        root["selection"].update(complete=False, completed=3, action_scores={"4": .75})
        with self.assertRaisesRegex(ValueError, "before complete selection"):
            P.report_view(report)
        root["selection"]["selected_action"] = None
        root["validation"].update(complete=False, expected_unique_cells=0, completed_unique_cells=0,
                                  rollout_action=None, rollout_scores=[], puct_scores=[],
                                  rollout_mean=None, puct_mean=None, delta=None)
        record_decision(report)
        view = P.report_view(report)
        self.assertEqual(view["state"], "INCOMPLETE")
        self.assertIsNone(view["roots"][0]["rollout_action"])
        self.assertEqual(view["validation_expected_unique"], 0)

    def test_duplicate_root_or_invalid_validation_seed_inventory_is_rejected(self):
        duplicate = report_fixture()
        duplicate["roots"][1]["root_id"] = duplicate["roots"][0]["root_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate root"):
            P.report_view(duplicate)
        for seeds in ([103021, 103021, 103067, 103091], [103003, 103043, 103067, 103091],
                      [103021, 103043, 103067], list(reversed(P.VALIDATION_SEEDS))):
            with self.subTest(seeds=seeds):
                report = report_fixture()
                report["roots"][0]["validation"]["seeds"] = seeds
                with self.assertRaisesRegex(ValueError, "validation seeds"):
                    P.report_view(report)

    def test_reported_gate_or_mean_cannot_override_measured_outcomes(self):
        report = report_fixture((0, 0))
        report["decision"]["passes"] = True
        with self.assertRaisesRegex(ValueError, "Recorded decision"):
            P.report_view(report)
        report = report_fixture()
        report["roots"][0]["validation"]["rollout_mean"] = 1
        with self.assertRaisesRegex(ValueError, "rollout_mean disagrees"):
            P.report_view(report)

    def test_invalid_terminal_scores_and_redefined_research_rule_are_rejected(self):
        for value in (.3, float("nan"), float("inf"), -1, 2, None, True):
            with self.subTest(value=value):
                report = report_fixture()
                report["roots"][0]["validation"]["rollout_scores"][0] = value
                with self.assertRaises(ValueError):
                    P.report_view(report)
        report = report_fixture()
        report["decision"]["thresholds"]["improved_roots_min"] = 1
        with self.assertRaisesRegex(ValueError, "precommitted rule"):
            P.report_view(report)

    def test_no_frozen_roots_is_incomplete_with_zero_coverage(self):
        report = report_fixture(())
        report["roots_frozen"] = False
        view = P.report_view(report)
        self.assertEqual(view["state"], "INCOMPLETE")
        self.assertEqual(view["roots"], [])
        for key in ("selection_expected", "selection_completed", "validation_expected_unique",
                    "validation_completed_unique", "complete_roots"):
            self.assertEqual(view[key], 0)
        self.assertIsNone(view["mean_delta"])


class ArtifactTests(unittest.TestCase):
    def test_measured_only_images_gray_missing_data_and_exact_source_snapshot(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.axes import Axes

        report = report_fixture((0, .5))
        # A completed loss is a measured zero, distinct from an unresolved cell.
        measured = report["roots"][0]["validation"]
        measured.update(rollout_scores=[0] * 4, puct_scores=[0] * 4, rollout_mean=0, puct_mean=0)
        make_unresolved(report["roots"][1], error=True)
        report["totals"]["attempt_errors"] = 1
        record_decision(report)
        original = deepcopy(report)
        source = ("\n" + json.dumps(report, indent=3) + "\n\n").encode()
        source_hash = hashlib.sha256(source).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(Axes, "axhspan", autospec=True, side_effect=Axes.axhspan) as bands:
                result = P.generate(report, output, source_bytes=source)
            self.assertEqual(bands.call_count, 2, "only the unresolved root gets two gray score bands")
            self.assertTrue(all(call.kwargs["color"] == "#e2e8f0" for call in bands.call_args_list))
            self.assertEqual(result["state"], "INCOMPLETE")
            self.assertEqual(result["report_sha256"], source_hash)
            self.assertEqual({p.name for p in output.iterdir()},
                             {"README.md", "comparison.png", "coverage.png", "report-snapshots"})
            snapshot = output / "report-snapshots" / f"{source_hash}.json"
            self.assertEqual(snapshot.read_bytes(), source)
            for name in ("comparison.png", "coverage.png"):
                self.assertEqual((output / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            readme = (output / "README.md").read_text()
            for text in ("Research decision: INCOMPLETE", "No SGD updates and no benchmark data",
                         "not new held-out roots", "Selection scores never count as validation",
                         "1/2 roots have complete paired validation", "Unresolved error",
                         "Gray denotes missing or incomplete evidence, never score zero", source_hash):
                self.assertIn(text, readme)
            self.assertIn("| 0.000 | 0.000 | +0.000 | Complete |", readme)
            self.assertIn("| — | — | — | Unresolved error |", readme)
            # Identical source reuse is allowed; damaged content-addressed evidence is not.
            P.generate(report, output, source_bytes=source)
            self.assertEqual(snapshot.read_bytes(), source)
            snapshot.write_bytes(b"tampered fixture")
            with self.assertRaisesRegex(ValueError, "immutable report snapshot changed"):
                P.generate(report, output, source_bytes=source)
        self.assertEqual(report, original)

    def test_mismatched_source_bytes_are_rejected(self):
        report = report_fixture()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Source bytes do not match"):
                P.generate(report, Path(directory), source_bytes=b"{}\n")
            self.assertFalse(list(Path(directory).rglob("*.png")))

    def test_no_roots_renders_pending_artifacts_without_inventing_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            result = P.generate(report_fixture(()), Path(directory))
            self.assertEqual(result["state"], "INCOMPLETE")
            readme = (Path(directory) / "README.md").read_text()
            self.assertIn("0/0 roots have complete paired validation", readme)
            self.assertIn("Validation: 0/0 valid scheduled unique cells", readme)
            self.assertNotIn("Research decision: PASS", readme)


if __name__ == "__main__":
    unittest.main()
