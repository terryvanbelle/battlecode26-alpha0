"""Plot measured validation and deployment evidence without running actors."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from alpha0 import evaluate as E
from alpha0 import terminal_v5_report as R
from scripts import plot_terminal_v5 as P
from tests.test_evaluation_runner import successful
from tests import test_terminal_v5_report as fixtures


class TerminalV5PlotsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.config = json.loads((E.ROOT/'configs/terminal-target-head-v5.json').read_text())
        self.identity = {'config': self.config, 'panel_sha256': E.file_hash(E.ROOT/'configs/benchmarks.json')}
        self.empty = R.report(self.directory/'run', self.directory/'progress', self.config, self.identity)

    def tearDown(self): self.temp.cleanup()

    def populated(self):
        summary = deepcopy(self.empty)
        fixture = fixtures.TerminalV5ReportTest()
        fixture.setUp()
        try:
            panel, pins, specs, _, _ = fixture.benchmark_fixture(count=12)
            rows = [R.wrap_benchmark_result(spec, successful(spec['evaluation_spec']), 1) for spec in specs]
            summary['benchmarks'] = fixture.summary_bench(specs, rows, panel, pins)
        finally:
            fixture.tearDown()
        roots = []
        for index in range(16):
            root = deepcopy(fixtures.validation_roots()[index % 2])
            root.update(root_id=f'validation-{index:03d}', input_hash=f'input-{index}', source_json=f'source-{index}.json')
            roots.append(root)
        rows = [fixtures.continuation(root, action, seed, 1 if action == 2 else .5, self.config, self.identity)
                for root in roots for action in (0,1,2) for seed in R.VALIDATION_SEEDS]
        summary['validation'] = R.summarize_validation(roots, rows, self.config, self.identity)
        summary['decision'].update(complete=True, criteria_pass=True, integrity_verified=True, review_for_expansion=True)
        return summary

    def test_empty_phase_generates_coverage_without_inventing_scores(self):
        output = self.directory/'plots'
        result = P.generate(self.empty, output)
        for name in ('comparison.png', 'coverage.png'):
            self.assertEqual((output/name).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        text = (output/'README.md').read_text()
        self.assertIn('INCOMPLETE', text)
        self.assertIn('First-action validation', text)
        self.assertIn('Whole-policy benchmark', text)
        self.assertIn('Historical 20/24', text)
        self.assertIn('Not yet frozen', text)
        snapshot = output/'report-snapshots'/f"{result['report_sha256']}.json"
        self.assertEqual(json.loads(snapshot.read_text()), self.empty)
        self.assertIsNone(self.empty['validation']['mean_delta_vs_puct'])

    def test_complete_evidence_keeps_two_measurements_separate(self):
        summary = self.populated()
        view = P.report_view(summary)
        self.assertEqual(view['state'], 'READY FOR RESEARCH REVIEW')
        self.assertEqual(summary['validation']['mean_delta_vs_puct'], .5)
        self.assertEqual(summary['benchmarks']['panels']['cp2']['mean_score'], 1)
        self.assertFalse(summary['decision']['automatic_promotion'])
        self.assertFalse(summary['decision']['all_map_victory'])

    def test_historical_reference_cannot_supply_missing_fresh_panel(self):
        summary = deepcopy(self.empty)
        summary['decision'].update(complete=True, criteria_pass=True, integrity_verified=True, review_for_expansion=True)
        with self.assertRaisesRegex(ValueError, 'complete qualifying evidence'):
            P.report_view(summary)

    def test_missing_root_cannot_have_a_zero_score_or_final_mean(self):
        summary = self.populated()
        root = summary['validation']['roots'][0]
        root['complete'] = False
        root['means'] = {arm:0 for arm in R.ARMS}
        with self.assertRaisesRegex(ValueError, 'Incomplete roots'):
            P.report_view(summary)

    def test_identical_actions_cannot_have_separate_favorable_outcomes(self):
        summary = self.populated()
        summary['validation']['roots'][0]['actions']['cp2'] = 2
        with self.assertRaisesRegex(ValueError, 'Identical first actions'):
            P.report_view(summary)

    def test_rendered_means_must_match_recorded_scores(self):
        summary = self.populated()
        summary['benchmarks']['panels']['terminal_head']['mean_score'] = .9
        with self.assertRaisesRegex(ValueError, 'panel score'):
            P.report_view(summary)
        summary = self.populated()
        summary['validation']['mean_delta_vs_puct'] = .6
        with self.assertRaisesRegex(ValueError, 'equal-root'):
            P.report_view(summary)


if __name__ == '__main__': unittest.main()
