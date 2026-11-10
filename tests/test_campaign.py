"""Campaign state transitions must preserve assessment and full-panel gates."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from alpha0 import campaign as C


IDS = [f'bot-{i}' for i in range(12)]
PUBLISH_RECOVERY = C.Campaign.publish_recovery


def report(*, complete=True, full_panel=True, passed=False, score=0.0, errors=0, opponents=None):
    names = IDS if opponents is None else opponents
    return {'complete': complete, 'full_panel_complete': complete and full_panel,
            'unresolved_errors': errors, 'passed': passed, 'victory_gate_passed': passed,
            'opponents': {name: {'mean_score': score} for name in names}}


class FakeLearner:
    def __init__(self, run, config, progress):
        self.run = run; self.iteration = config.get('test_initial_iteration', 2)
        self.games_total = 0; self.training_calls = []
        self.checkpoint_path = run / f'checkpoint-{self.iteration:04d}.pt'

    def train_iteration(self, deadline, stop_file):
        self.training_calls.append(self.iteration)
        self.iteration += 1; self.games_total += 2
        self.checkpoint_path = self.run / f'checkpoint-{self.iteration:04d}.pt'
        return {'iteration': self.iteration}


class CampaignTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root/'configs').mkdir()
        self.config = {'experiment_id': 'test', 'maps': ['tiny'], 'workers': 2,
                       'game_timeout_seconds': 60, 'smoke_eval_every': 1,
                       'standard_eval_every': 99, 'max_iterations_per_cycle': 1}
        self.gate = {'frozen_opponents': [{'id': name} for name in IDS], 'required_opponents': 12,
                     'stages': {'holdout': {'maps': ['Stash']}}}
        self.fake_learner = mock.patch.object(C, 'Learner', FakeLearner); self.fake_learner.start()
        self.plotter = mock.patch.object(C.Campaign, 'plots'); self.plotter.start()
        # Ordinary state-machine tests must never invoke network publication.
        self.publisher = mock.patch.object(C.Campaign, 'publish_recovery'); self.publisher.start()
        self.output = redirect_stdout(io.StringIO()); self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None); self.publisher.stop(); self.plotter.stop(); self.fake_learner.stop(); self.temp.cleanup()

    def make(self):
        (self.root/'configs/campaign-v2.json').write_text(json.dumps(self.config))
        (self.root/'configs/victory_gate.json').write_text(json.dumps(self.gate))
        return C.Campaign(self.root, 1)

    def test_interrupted_assessment_resumes_before_any_training(self):
        campaign = self.make()
        with mock.patch.object(campaign, 'evaluation', return_value=report(complete=False)):
            self.assertEqual(campaign.run_cycle(), 'budget_or_stop')
        self.assertEqual(campaign.learner.training_calls, [])
        events = []
        def evaluate(stage):
            events.append(('evaluate', campaign.learner.iteration, stage))
            return report()
        original = campaign.learner.train_iteration
        def train(*args):
            events.append(('train', campaign.learner.iteration)); return original(*args)
        with mock.patch.object(campaign, 'evaluation', side_effect=evaluate), mock.patch.object(campaign.learner, 'train_iteration', side_effect=train):
            self.assertEqual(campaign.run_cycle(), 'cycle_complete_review_required')
        self.assertEqual(events[:2], [('evaluate', 2, 'smoke'), ('train', 2)])
        self.assertEqual(events[-1], ('evaluate', 3, 'smoke'))
        self.assertFalse(campaign.cycle['all_opponents_beaten'])

    def test_expired_budget_launches_neither_evaluation_nor_training(self):
        campaign = self.make(); campaign.deadline = time.monotonic()-1
        with mock.patch.object(campaign, 'evaluation', side_effect=AssertionError('expired budget')):
            self.assertEqual(campaign.run_cycle(), 'budget_or_stop')
        self.assertEqual(campaign.learner.training_calls, [])

    def test_stop_after_smoke_does_not_start_standard_or_training(self):
        campaign = self.make(); calls = []
        def evaluate(stage):
            calls.append(stage); campaign.stop_file.parent.mkdir(parents=True, exist_ok=True)
            campaign.stop_file.write_text('stop'); return report(score=1)
        with mock.patch.object(campaign, 'evaluation', side_effect=evaluate):
            self.assertEqual(campaign.run_cycle(), 'budget_or_stop')
        self.assertEqual(calls, ['smoke']); self.assertEqual(campaign.learner.training_calls, [])

    def test_failed_standard_cannot_be_replaced_by_high_panel_average(self):
        campaign = self.make(); calls = []
        standard = report(score=1, passed=False)
        standard['opponents'][IDS[-1]]['mean_score'] = 0.4
        standard['macro_score'] = 0.95
        def evaluate(stage):
            calls.append(stage); return report(score=1) if stage=='smoke' else standard
        with mock.patch.object(campaign, 'evaluation', side_effect=evaluate):
            self.assertEqual(campaign.assess(), 'continue')
        self.assertEqual(calls, ['smoke', 'standard'])

    def test_failed_holdout_pauses_without_training_another_candidate(self):
        campaign = self.make(); calls = []
        def evaluate(stage):
            calls.append(stage)
            return report(score=1, passed=True) if stage!='holdout' else report(score=0.4, passed=False)
        with mock.patch.object(campaign, 'evaluation', side_effect=evaluate):
            self.assertEqual(campaign.run_cycle(), 'holdout_consumed_review_required')
        self.assertEqual(calls, ['smoke', 'standard', 'holdout'])
        self.assertEqual(campaign.learner.training_calls, [])
        self.assertFalse(campaign.cycle['all_opponents_beaten'])

    def test_complete_passing_full_holdout_is_only_victory_path(self):
        campaign = self.make()
        with mock.patch.object(campaign, 'evaluation', return_value=report(score=1, passed=True)):
            self.assertEqual(campaign.run_cycle(), 'victory')
        self.assertTrue(campaign.cycle['all_opponents_beaten'])
        self.assertEqual(campaign.learner.training_calls, [])

    def test_partial_panel_marked_complete_cannot_advance(self):
        campaign = self.make()
        with mock.patch.object(campaign, 'evaluation', return_value=report(score=1, complete=True, full_panel=False)):
            self.assertEqual(campaign.assess(), 'evaluation_error')

    def test_missing_opponent_cannot_claim_victory(self):
        campaign = self.make(); calls = []
        def evaluate(stage):
            calls.append(stage)
            return report(score=1, passed=True, opponents=IDS[:-1] if stage=='holdout' else IDS)
        with mock.patch.object(campaign, 'evaluation', side_effect=evaluate):
            self.assertEqual(campaign.run_cycle(), 'evaluation_error')
        self.assertFalse(campaign.cycle['all_opponents_beaten'])

    def test_evaluation_error_pauses_without_training(self):
        campaign = self.make()
        with mock.patch.object(campaign, 'evaluation', return_value=report(complete=False, errors=1)):
            self.assertEqual(campaign.run_cycle(), 'evaluation_error')
        self.assertEqual(campaign.learner.training_calls, [])

    def test_recovery_failure_is_nonfatal_and_persisted(self):
        campaign = self.make(); campaign.status('paused', reason='budget_or_stop')
        error = C.subprocess.CalledProcessError(1, ['snapshot', 'publish'])
        with mock.patch.object(C.subprocess, 'run', side_effect=error):
            PUBLISH_RECOVERY(campaign)
        status = json.loads((campaign.progress/'status.json').read_text())
        self.assertIn('recovery_publication_error', status)
        self.assertEqual(status['phase'], 'paused')
        self.assertFalse(status['all_opponents_beaten'])

    def test_malformed_publication_response_does_not_abort(self):
        campaign = self.make(); campaign.status('paused')
        with mock.patch.object(C.subprocess, 'run', return_value=mock.Mock(stdout='{}')):
            PUBLISH_RECOVERY(campaign)
        self.assertIn('recovery_publication_error', json.loads((campaign.progress/'status.json').read_text()))

    def test_recovery_success_persists_pointer_and_clears_old_error(self):
        campaign = self.make(); campaign.status('paused', recovery_publication_error='previous failure')
        responses = [mock.Mock(stdout=json.dumps({'archive': '/tmp/checkpoint.tar.gz'})),
                     mock.Mock(stdout=json.dumps({'archive': 'checkpoint.tar.gz', 'sha256': 'a'*64}))]
        with mock.patch.object(C.subprocess, 'run', side_effect=responses) as run:
            PUBLISH_RECOVERY(campaign)
        self.assertEqual(run.call_count, 2)
        self.assertTrue(all(call.kwargs.get('timeout', 0) > 0 for call in run.call_args_list))
        status = json.loads((campaign.progress/'status.json').read_text())
        self.assertEqual(status['last_published_recovery'], 'checkpoint.tar.gz')
        self.assertNotIn('recovery_publication_error', status)

    def test_training_map_overlap_with_holdout_rejected(self):
        self.config['maps'].append('Stash')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.make()


if __name__ == '__main__':
    unittest.main()
