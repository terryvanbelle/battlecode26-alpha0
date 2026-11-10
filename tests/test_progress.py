"""Checks for measured-only reporting and correct benchmark denominators."""
import importlib.util
import json
import math
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

    def test_wilson_endpoints_never_produce_negative_error_lengths(self):
        # n=14 and zero wins reproduced the real campaign plotting crash.
        for n in (1, 2, 14, 16, 296, 592, 3552):
            self.assertEqual(report.wilson(0, n)[0], 0.0)
            self.assertEqual(report.wilson(n, n)[1], 1.0)

    def test_stages_and_gate_configurations_remain_separate_evidence(self):
        rows = []
        for stage, config in (('smoke', 'old'), ('standard', 'old'),
                              ('field-twelve-v2-all-maps-v2:standard', 'new'),
                              ('field-twelve-v2-all-maps-v2:confirmation', 'new')):
            for side in ('A', 'B'):
                rows.append(dict(row(side=side), stage=stage, evaluation_config_sha256=config,
                                 checkpoint_sha256='same-checkpoint'))
        _, cleaned, _ = report.clean_rows([], rows + [dict(rows[-1])])
        self.assertEqual(len(cleaned), 8)
        groups = report.summarize(cleaned, ['weak'])
        self.assertEqual(len(groups), 4)
        self.assertTrue(all(group['completed'] == 2 for group in groups))
        self.assertTrue(all(group['paired']['cells'] == 1 for group in groups))
        old_identity = dict(rows[0], evaluation_config_sha256='different')
        _, cleaned, _ = report.clean_rows([], rows + [old_identity])
        self.assertEqual(len(report.summarize(cleaned, ['weak'])), 5)
        unbound = dict(row(), stage='standard')
        bound = dict(unbound, identity={'evaluation_config': {'all_map_gate': {'gate_id': 'current'}}})
        _, cleaned, _ = report.clean_rows([], [unbound, bound])
        self.assertEqual(len(report.summarize(cleaned, ['weak'])), 2)

    def test_legacy_and_current_identity_are_not_pooled(self):
        legacy = row()
        current = dict(legacy, stage='smoke', evaluation_config_sha256='versioned',
                       panel_fingerprint_sha256='panel')
        _, cleaned, _ = report.clean_rows([], [legacy, current])
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(len(report.summarize(cleaned, ['weak'])), 2)

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

    def test_legacy_value_loss_is_not_relabelled_as_bce(self):
        legacy = dict(iteration=1, value_loss=0.75, replay_positions=200)
        self.assertIsNone(report.learning_health_series([legacy]))
        measured = dict(iteration=2, value_loss=0.6, value_loss_type='BCEWithLogits',
                        value_mse=0.8, value_mean=0.0, value_saturated_fraction=0.0,
                        searched_positions=20, actual_decisions=100, replay_games=4)
        series = report.learning_health_series([legacy, measured])
        self.assertTrue(math.isnan(series['value_bce'][0]))
        self.assertEqual(series['value_bce'][1], 0.6)
        self.assertEqual(series['value_mean'][1], 0.0)
        self.assertEqual(series['value_saturated_fraction'][1], 0.0)
        self.assertTrue(math.isnan(series['replay_games'][0]))
        self.assertTrue(math.isnan(series['search_fraction'][0]))
        self.assertEqual(series['search_fraction'][1], 0.2)

    def test_zero_decisions_does_not_invent_search_coverage(self):
        series = report.learning_health_series([
            dict(iteration=1, value_bce=0.7, actual_decisions=0, searched_positions=0),
            dict(iteration=2, value_bce=0.5, searched_positions=10),
        ])
        self.assertTrue(all(math.isnan(value) for value in series['search_fraction']))

    def test_health_figure_is_conditional_and_labels_training_diagnostics(self):
        # Deliberate unit-test fixtures only, never written to repository progress.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = path / 'manifest.json'
            manifest.write_text(json.dumps({'bots': [{'id': 'weak'}]}))
            # Exercise the observed zero-score Wilson rendering failure and
            # independent stage series in temporary test-only measurements.
            evaluations = [dict(row(score=0, side='A' if i % 2 == 0 else 'B'), seed=i // 2,
                                stage='standard', evaluation_config_sha256='test-standard',
                                started_at='2026-11-10T04:06:00+00:00') for i in range(14)]
            evaluations.append(dict(row(), stage='smoke', evaluation_config_sha256='test-smoke',
                                    started_at='2026-11-10T03:50:00+00:00'))
            (path / 'evaluations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in evaluations))
            log = path / 'training.jsonl'
            log.write_text(json.dumps(dict(iteration=1, value_loss=0.8)) + '\n')
            result = report.generate(path, manifest)
            self.assertNotIn('learning-health.png', result['outputs'])
            self.assertFalse((path / 'learning-health.png').exists())
            self.assertNotIn('Learning health', (path / 'README.md').read_text())
            log.write_text(json.dumps(dict(iteration=2, value_loss=0.6,
                value_loss_type='BCEWithLogits', value_mse=0.8, value_mean=0.0,
                value_saturated_fraction=0.0, replay_games=4, replay_positions=200,
                searched_positions=20, actual_decisions=100)) + '\n')
            result = report.generate(path, manifest)
            self.assertIn('learning-health.png', result['outputs'])
            image = path / 'learning-health.png'
            self.assertTrue(image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
            text = (path / 'README.md').read_text()
            self.assertIn('training-batch diagnostics, not held-out calibration', text)
            self.assertIn('](learning-health.png)', text)
            self.assertIn(f'--progress-dir {path}', text)
            self.assertIn(f'--manifest {manifest}', text)
            self.assertIn('Evaluation series: `standard [test-sta]`', text)
            self.assertIn('do not establish the all-map victory gate', text)
            self.assertEqual(result['evaluation_cells'], 15)
            self.assertEqual(result['evaluation_groups'], 2)
            self.assertEqual(result['checkpoints'], 1)


if __name__ == '__main__':
    unittest.main()
