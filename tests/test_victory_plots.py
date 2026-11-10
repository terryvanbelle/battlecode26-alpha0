"""Visualization safeguards: incomplete evidence never acquires a fabricated score."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('victory_plot', ROOT/'scripts/plot_victory.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def pair(a=1.0, b=0.5, *, complete=True, errors=0, passed=True):
    return {'expected': 4, 'completed': 4 if complete else 3, 'errors': errors,
            'mean_score': (a+b)/2, 'passed': passed,
            'sides': {'A': {'expected': 2, 'completed': 2, 'mean_score': a},
                      'B': {'expected': 2, 'completed': 2 if complete else 1, 'mean_score': b}}}


def fixture():
    gate = {'schema_version': 2, 'gate_id': 'test-all-maps', 'frozen_opponents': [{'id': 'weak'}, {'id': 'strong'}],
            'stages': {stage: {'maps': ['tiny', 'large'], 'min_score_per_map': .6, 'min_score_each_side_per_map': .5}
                       for stage in ('standard', 'confirmation')}}
    report = {'stage': 'standard', 'gate_id': gate['gate_id'], 'checkpoint_sha256': 'ab'*32,
              'completed': 11, 'expected': 16, 'pending': 5, 'unresolved_errors': 1,
              'passed': False, 'victory_gate_passed': False, 'required_maps': 2,
              'expected_opponent_map_pairs': 4, 'completed_opponent_map_pairs': 2, 'passed_opponent_map_pairs': 1,
              'opponents': {'weak': {'mean_score': .75, 'passed': False, 'maps': {'tiny': pair(), 'large': pair(0,0,passed=False)}},
                            'strong': {'mean_score': 1, 'passed': False, 'maps': {'tiny': pair(1,1,complete=False,errors=1)}}}}
    return report, gate


class VictoryPlotsTest(unittest.TestCase):
    def test_incomplete_and_error_pairs_are_gray_not_zero(self):
        self.assertEqual(P.pair_values(pair(1,1,complete=False))['scores'], [None]*3)
        self.assertEqual(P.pair_values(pair(1,1,errors=1))['scores'], [None]*3)
        self.assertEqual(P.pair_values(None)['scores'], [None]*3)
        self.assertEqual(P.pair_values(pair(0,0,passed=False))['scores'], [0,0,0])

    def test_full_frozen_inventory_survives_missing_observations(self):
        report, gate = fixture()
        view = P.report_view(report, gate)
        self.assertEqual(len(view['values']), 4)
        self.assertEqual(view['complete_pairs_displayed'], 2)
        self.assertEqual(view['passed_pairs_displayed'], 1)
        self.assertEqual(view['values'][('strong','large')]['status'], 'Not recorded')

    def test_wrong_identity_and_out_of_inventory_rows_fail(self):
        report, gate = fixture()
        report['gate_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'identities'):
            P.report_view(report, gate)
        report['gate_id'] = gate['gate_id']
        report['opponents']['extra'] = {'maps': {}}
        with self.assertRaisesRegex(ValueError, 'outside'):
            P.report_view(report, gate)

    def test_invalid_scores_and_inconsistent_seat_means_fail(self):
        data = pair(); data['mean_score'] = float('nan')
        with self.assertRaises(ValueError): P.pair_values(data)
        data = pair(); data['mean_score'] = .1
        with self.assertRaisesRegex(ValueError, 'mean scores disagree'): P.pair_values(data)
        report, gate = fixture(); report['opponents']['weak']['maps']['tiny'] = pair(1,0,passed=True)
        with self.assertRaisesRegex(ValueError, 'threshold'): P.report_view(report,gate)

    def test_vacuous_or_partial_success_claims_fail(self):
        report, gate = fixture(); report['passed'] = True
        with self.assertRaisesRegex(ValueError, 'lacks complete'):
            P.report_view(report, gate)
        report['passed'] = False; report['victory_gate_passed'] = True
        with self.assertRaisesRegex(ValueError, 'passing confirmation'):
            P.report_view(report, gate)

    def test_generate_writes_only_requested_atomic_artifacts(self):
        report, gate = fixture()
        original = deepcopy(report)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            result = P.generate(report, output, gate=gate)
            self.assertEqual(set(p.name for p in output.iterdir()), {'all-maps-standard.png','all-maps-standard.md'})
            self.assertEqual((output/'all-maps-standard.png').read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            text = (output/'all-maps-standard.md').read_text()
            self.assertIn('11/16 valid games', text)
            self.assertIn('Unresolved error', text)
            self.assertIn('not a zero or a loss', text)
            self.assertIn('Full victory gate not established', text)
            self.assertIn('| `tiny` | 3/4 | — | — (2/2) | — (1/2) | 1 | Unresolved error |',text)
            self.assertEqual(result['complete_pairs_displayed'],2)
        self.assertEqual(report,original)

    def test_confirmation_names_and_empty_report_are_safe(self):
        report, gate = fixture(); report['stage'] = 'confirmation'
        report['opponents'] = {}; report.update(completed=0,pending=16,unresolved_errors=0)
        with tempfile.TemporaryDirectory() as folder:
            result=P.generate(report,Path(folder),gate=gate)
            self.assertTrue(Path(result['png']).name=='all-maps-confirmation.png')
            self.assertEqual(result['complete_pairs_displayed'],0)
            text=Path(result['markdown']).read_text()
            self.assertIn('0/16 valid games',text)
            self.assertNotIn('**Stage: PASSED',text)


if __name__=='__main__': unittest.main()
