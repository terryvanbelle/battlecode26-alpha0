"""Checks for measured-only reporting and correct benchmark denominators."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('progress_report', Path(__file__).resolve().parents[1] / 'scripts/plot_progress.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def row(opponent='weak', score=1, status='ok', side='A', checkpoint='first'):
    return dict(iteration=1, checkpoint=checkpoint, opponent=opponent, map='tiny', seed=0, side=side, score=score, rounds=100, status=status)


class ProgressTest(unittest.TestCase):
    def test_wilson_extremes_retain_uncertainty(self):
        lo, hi = report.wilson(2, 2)
        self.assertAlmostEqual(lo, 0.3423802275)
        self.assertAlmostEqual(hi, 1)
        lo, hi = report.wilson(0, 2)
        self.assertAlmostEqual(lo, 0)
        self.assertAlmostEqual(hi, 0.6576197725)

    def test_errors_do_not_become_wins_or_ordinary_losses(self):
        rows = [row(score=1), row(score=0, status='error', side='B')]
        group = report.summarize(rows, ['weak', 'strong'])[0]
        self.assertEqual(group['completed'], 1)
        self.assertEqual(group['errors'], 1)
        self.assertEqual(group['macro_score'], 1)
        self.assertFalse(group['full_panel'])
        self.assertEqual(group['opponents']['strong']['n'], 0)

    def test_equal_opponent_weight_and_checkpoint_separation(self):
        rows = [row(), row(side='B'), row(opponent='strong', score=0), row(score=0, checkpoint='second')]
        groups = report.summarize(rows, ['weak', 'strong'])
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]['macro_score'], 0.5)
        self.assertEqual(groups[0]['completed'], 3)
        self.assertEqual(groups[1]['macro_score'], 0)

    def test_paired_bootstrap_keeps_both_sides_and_all_opponents(self):
        rows = []
        for map_index in range(4):
            for name in ('weak', 'strong'):
                for side in ('A', 'B'):
                    cell = row(name, score=map_index % 2, side=side)
                    cell['map'] = str(map_index)
                    rows.append(cell)
        stats = report.paired_panel_interval(rows, ['weak', 'strong'])
        self.assertEqual(stats['cells'], 4)
        self.assertEqual(stats['mean'], 0.5)
        self.assertLessEqual(stats['lower'], 0.5)
        self.assertGreaterEqual(stats['upper'], 0.5)
        rows[-1]['status'] = 'error'
        stats = report.paired_panel_interval(rows, ['weak', 'strong'])
        self.assertEqual(stats['cells'], 3)
        self.assertIsNone(stats['lower'])

    def test_iteration_zero_is_preserved(self):
        baseline = row(checkpoint='initial'); baseline['iteration'] = 0
        candidate = row(checkpoint='trained')
        _, rows, _ = report.clean_rows([], [baseline, candidate])
        groups = report.summarize(rows, ['weak'])
        self.assertEqual([g['iteration'] for g in groups], [0, 1])

    def test_repeated_cell_is_not_extra_evidence(self):
        training, rows, notices = report.clean_rows([], [row(status='error'), row()])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'ok')
        self.assertTrue(notices)

    def test_only_incomplete_final_line_is_tolerated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'live.jsonl'
            path.write_text(json.dumps(row()) + '\n{"iter')
            rows, notices = report.read_jsonl(path)
            self.assertEqual(len(rows), 1)
            self.assertTrue(notices)
            path.write_text('{"iter\n')
            with self.assertRaises(ValueError):
                report.read_jsonl(path)

    def test_invalid_completed_scores_fail_loudly(self):
        with self.assertRaises(ValueError):
            report.clean_rows([], [row(score=3)])


if __name__ == '__main__':
    unittest.main()
