"""Behavioral checks for configuration-safe resume, retries, and frozen gates."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from alpha0 import evaluate as E


def fixture(root: Path):
    maps = root / 'maps'; maps.mkdir(exist_ok=True)
    (maps / 'tiny.map26').write_bytes(b'actual map bytes')
    manifest = {'suite_id': 'frozen12', 'bots': [
        {'id': name, 'source_commit': name + '-commit', 'java_tree_sha256': name + '-hash'}
        for name in ('weak', 'strong')]}
    schema = {'model_version': 'residual_v1', 'action_version': 'macro_v1', 'feature_version': 1,
              'action_dim': 17, 'network_config': {'action_dim': 17}}
    runtime = {'engine_version': '1.2.6', 'engine_jar_sha256': 'jar', 'engine_patch_sha256': 'patch',
               'bridge_sha256': 'bridge', 'physics': {'limit': 1000000000}}
    context = E.make_context('checkpoint-hash', schema, runtime, timeout_seconds=30)
    schedule = {'maps': ['tiny'], 'seeds': [7], 'sides': ['A', 'B']}
    specs = E.build_cell_specs(manifest, schedule, context, checkpoint=root / 'model.pt',
                 iteration=3, suite='smoke', stage='smoke', maps_dir=maps)
    return manifest, context, schedule, specs


def successful(spec, score=1, attempt=1):
    row = E.result_base(spec, attempt)
    identity = spec['identity']
    learned_team = 0 if identity['side'] == 'A' else 1
    winner = -1 if score == 0.5 else learned_team if score == 1 else 1-learned_team
    row.update(status='ok', score=score, rounds=100, winner=winner,
               terminal={'terminal': True, 'valid_game': True, 'benchmark_failures': [],
                         'winner': winner, 'round': 100, 'map': identity['map'], 'seed': identity['seed'],
                         'action_version': identity['evaluation_config']['policy_schema']['action_version']})
    return row


def failed(spec, kind='worker_process_error', attempt=1):
    row = E.result_base(spec, attempt)
    row.update(error='failure', error_kind=kind)
    return row


class EvaluationRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest, self.context, self.schedule, self.specs = fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, manifest=None, context=None):
        return E.build_cell_specs(manifest or self.manifest, self.schedule, context or self.context,
                    checkpoint=self.root / 'model.pt', iteration=3, suite='smoke', stage='smoke', maps_dir=self.root / 'maps')

    def test_resume_identity_changes_with_every_execution_dependency(self):
        original = self.specs[0]['evaluation_cell_id']
        context = deepcopy(self.context); context['checkpoint_sha256'] = 'new-checkpoint'
        self.assertNotEqual(self.build(context=context)[0]['evaluation_cell_id'], original)
        for key in ('engine_jar_sha256', 'engine_patch_sha256', 'bridge_sha256', 'physics'):
            context = deepcopy(self.context)
            context['evaluation_config']['runtime'][key] = {'changed': True}
            self.assertNotEqual(self.build(context=context)[0]['evaluation_cell_id'], original)
        for key in ('policy_schema', 'search_mode', 'search_config'):
            context = deepcopy(self.context); context['evaluation_config'][key] = {'changed': True}
            self.assertNotEqual(self.build(context=context)[0]['evaluation_cell_id'], original)
        manifest = deepcopy(self.manifest); manifest['bots'][0]['java_tree_sha256'] = 'new-source'
        self.assertNotEqual(self.build(manifest=manifest)[0]['evaluation_cell_id'], original)
        (self.root / 'maps/tiny.map26').write_bytes(b'changed map bytes')
        self.assertNotEqual(self.build()[0]['evaluation_cell_id'], original)

    def test_unknown_search_mode_fails_instead_of_greedy_fallback(self):
        with self.assertRaisesRegex(ValueError, 'Only greedy'):
            E.make_context('cp', {}, {}, timeout_seconds=10, search_mode='mcts')

    def test_v1_and_v2_schemas_are_explicit(self):
        self.assertEqual(E.policy_schema({'network_config': {'action_dim': 17}})['action_version'], 'macro_v1')
        payload = {'model_version': 'dual_scale_v2', 'feature_version': 2, 'network_config': {'action_dim': 40}}
        self.assertEqual(E.policy_schema(payload)['action_version'], 'macro_v2')
        payload['network_config']['action_dim'] = 17
        with self.assertRaises(ValueError):
            E.policy_schema(payload)

    def test_legacy_and_mismatched_rows_never_establish_completion(self):
        rows = [successful(spec) for spec in self.specs]
        rows[0].pop('identity')
        rows[1]['identity'] = {'different': 'configuration'}
        report = E.completion_report(self.specs, rows, required_opponents=2)
        self.assertFalse(report['complete'])
        self.assertEqual(report['completed'], 2)
        legacy = {'checkpoint_sha256': 'checkpoint-hash', 'opponent': 'weak', 'map': 'tiny', 'seed': 7,
                  'side': 'A', 'status': 'ok', 'score': 1, 'rounds': 100}
        self.assertEqual(E.completion_report(self.specs, [legacy])['completed'], 0)

    def test_missing_and_error_cells_cannot_pass_gate(self):
        rows = [successful(spec) for spec in self.specs[:-1]] + [failed(self.specs[-1], 'opponent_runtime_error')]
        report = E.gate_stage_report(self.specs, rows, {'min_score': 0.6, 'min_score_each_side': 0.5}, required_opponents=2)
        self.assertFalse(report['passed'])
        self.assertEqual(report['unresolved_errors'], 1)
        self.assertEqual(report['exhausted_cells'], 1)
        self.assertEqual(report['opponents']['strong']['losses'], 0)

    def test_every_opponent_and_each_seat_must_pass(self):
        rows = [successful(spec, 1 if spec['identity']['opponent'] == 'weak' else 0) for spec in self.specs]
        report = E.gate_stage_report(self.specs, rows, {'min_score': 0.6, 'min_score_each_side': 0.5}, required_opponents=2)
        self.assertTrue(report['complete']); self.assertFalse(report['passed'])
        rows = [successful(spec, 1 if spec['identity']['side'] == 'A' else 0) for spec in self.specs]
        report = E.gate_stage_report(self.specs, rows, {'min_score': 0.5, 'min_score_each_side': 0.5}, required_opponents=2)
        self.assertFalse(report['passed'])
        report = E.gate_stage_report(self.specs, [successful(s) for s in self.specs], {'min_score': 0.6, 'min_score_each_side': 0.5}, required_opponents=12)
        self.assertFalse(report['passed'])

    def test_first_success_is_immutable_not_best_of_replays(self):
        spec = self.specs[0]
        rows = [failed(spec, attempt=1), successful(spec, score=0, attempt=2), successful(spec, score=1, attempt=3)]
        self.assertEqual(E.selected_result(spec, rows)['score'], 0)

    def test_duplicate_schedule_fails(self):
        self.schedule['seeds'] = [7, 7]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.build()

    def test_retries_are_capped_and_bot_faults_do_not_retry(self):
        spec = self.specs[0]; store = self.root / 'cells'; directory = store / spec['evaluation_cell_id']
        E.atomic_json(directory / 'attempt-0001.json', failed(spec, 'opponent_runtime_error'))
        with mock.patch.object(E.subprocess, 'Popen', side_effect=AssertionError('bot failure must not retry')):
            self.assertEqual(E.run_cell(spec, store, 2)['attempt'], 1)
        (directory / 'attempt-0001.json').unlink()
        process = mock.Mock(); process.returncode = 9; process.wait.return_value = 9
        with mock.patch.object(E.subprocess, 'Popen', return_value=process) as launch:
            result = E.run_cell(spec, store, 2)
            self.assertEqual(launch.call_count, 2)
            self.assertEqual(result['attempt'], 2)
        with mock.patch.object(E.subprocess, 'Popen', side_effect=AssertionError('exhausted retry must not launch')):
            self.assertEqual(E.run_cell(spec, store, 2)['attempt'], 2)

    def test_successful_cell_resume_does_not_launch_worker(self):
        spec = self.specs[0]; store = self.root / 'cells'
        E.atomic_json(store / spec['evaluation_cell_id'] / 'attempt-0001.json', successful(spec, score=0))
        with mock.patch.object(E.subprocess, 'Popen', side_effect=AssertionError('completed cell must not replay')):
            self.assertEqual(E.run_cell(spec, store, 2)['score'], 0)

    def test_atomic_merge_preserves_legacy_and_separate_attempt_errors(self):
        progress = self.root / 'progress'; store = progress / 'evaluation_cells'; spec = self.specs[0]
        legacy = {'iteration': 0, 'checkpoint': 'old.pt', 'status': 'ok', 'score': 0}
        E.atomic_text(progress / 'evaluations.jsonl', json.dumps(legacy) + '\n')
        E.atomic_json(store / spec['evaluation_cell_id'] / 'request.json', spec)
        E.atomic_json(store / spec['evaluation_cell_id'] / 'attempt-0001.json', failed(spec))
        E.atomic_json(store / spec['evaluation_cell_id'] / 'attempt-0002.json', successful(spec, 0.5, 2))
        E.merge_progress(progress, store); E.merge_progress(progress, store)
        rows = E.read_jsonl(progress / 'evaluations.jsonl')
        self.assertEqual(len(rows), 2); self.assertEqual(rows[0], legacy)
        self.assertEqual(rows[1]['score'], 0.5)
        self.assertEqual(len(E.read_jsonl(progress / 'evaluation_errors.jsonl')), 1)

    def test_missing_artifacts_restore_published_success_without_replay(self):
        progress = self.root / 'progress'; store = progress / 'evaluation_cells'; spec = self.specs[0]
        E.atomic_text(progress / 'evaluations.jsonl', E.canonical(successful(spec, score=0)) + '\n')
        E.restore_known_attempts(self.specs, progress, store)
        with mock.patch.object(E.subprocess, 'Popen', side_effect=AssertionError('published success must not replay')):
            self.assertEqual(E.run_cell(spec, store, 2)['score'], 0)

    def test_holdout_requires_qualification_and_locks_one_candidate(self):
        gate = {'gate_id': 'gate1', 'panel_fingerprint_sha256': E.panel_fingerprint(self.manifest)}
        path = self.root / 'gate-ledger.json'
        with self.assertRaisesRegex(ValueError, 'Holdout requires'):
            E.lock_holdout(path, gate, self.context, standard_passed=False)
        ledger = E.lock_holdout(path, gate, self.context, standard_passed=True)
        self.assertEqual(ledger['status'], 'in_progress')
        self.assertEqual(E.lock_holdout(path, gate, self.context, standard_passed=True), ledger)
        other = deepcopy(self.context); other['checkpoint_sha256'] = 'new-candidate'
        with self.assertRaisesRegex(ValueError, 'already locked'):
            E.lock_holdout(path, gate, other, standard_passed=True)
        ledger['status'] = 'consumed_failed'; E.atomic_json(path, ledger)
        with self.assertRaises(ValueError):
            E.lock_holdout(path, gate, other, standard_passed=True)

    def test_invalid_terminal_and_disagreeing_winner_never_pass(self):
        spec = self.specs[0]
        for key, value in [('terminal', False), ('valid_game', False), ('benchmark_failures', ['crash']),
                           ('winner', 1), ('map', 'different'), ('seed', 8), ('action_version', 'macro_v2')]:
            row = successful(spec); row['terminal'][key] = value
            self.assertFalse(E.valid_result(spec, row), key)
        row = successful(spec); row['score'] = True
        self.assertFalse(E.valid_result(spec, row))

    def test_expired_deadline_launches_nothing(self):
        with mock.patch.object(E, 'run_cell', side_effect=AssertionError('deadline must stop launches')):
            reason = E.run_schedule(self.specs, self.root/'cells', jobs=2, max_attempts=2,
                                    on_result=lambda _: None, deadline=time.monotonic()-1)
        self.assertEqual(reason, 'deadline')

    def test_stop_file_limits_submission_and_active_cells_finish(self):
        stop = self.root/'STOP'; completed = []
        def fake_cell(spec, *_):
            stop.write_text('stop')
            return successful(spec)
        with mock.patch.object(E, 'run_cell', side_effect=fake_cell) as run:
            reason = E.run_schedule(self.specs, self.root/'cells', jobs=1, max_attempts=2,
                                    on_result=completed.append, stop_file=stop)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(len(completed), 1)
        self.assertEqual(reason, 'stop_file')

    def test_holdout_rejects_training_leakage_or_undeclared_training_maps(self):
        gate = {'stages': {'holdout': {'maps': ['unseen']}}}
        E.verify_holdout_isolation({'config': {'maps': ['tiny']}}, gate)
        with self.assertRaisesRegex(ValueError, 'used for training'):
            E.verify_holdout_isolation({'config': {'maps': ['tiny', 'unseen']}}, gate)
        with self.assertRaisesRegex(ValueError, 'metadata'):
            E.verify_holdout_isolation({}, gate)

    def test_gate_freezes_panel(self):
        gate = {'benchmark_suite_id': self.manifest['suite_id'], 'panel_fingerprint_sha256': E.panel_fingerprint(self.manifest), 'required_opponents': 2}
        E.verify_gate(self.manifest, gate)
        self.manifest['bots'][0]['source_commit'] = 'replacement'
        with self.assertRaisesRegex(ValueError, 'frozen'):
            E.verify_gate(self.manifest, gate)


if __name__ == '__main__':
    unittest.main()
