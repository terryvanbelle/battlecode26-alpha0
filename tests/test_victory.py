"""All-map victory cannot be replaced by attractive pooled scores or stale evidence."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from alpha0 import evaluate as E
from alpha0 import victory as V
from tests.test_evaluation_runner import successful, failed


class VictoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.root = Path(self.temporary.name)
        self.maps = self.root / 'maps'; self.maps.mkdir()
        for name in ('a', 'b'):
            (self.maps/f'{name}.map26').write_bytes(name.encode())
        self.manifest = {'suite_id': 'frozen', 'bots': [
            {'id': name, 'source_commit': name+'-commit', 'java_tree_sha256': name+'-hash'}
            for name in ('weak', 'strong')]}
        inventory = V.map_inventory(self.maps)
        self.gate = {'schema_version': 2, 'gate_id': 'all-maps-v2', 'benchmark_suite_id': 'frozen',
                     'frozen_opponents': self.manifest['bots'], 'required_opponents': 2,
                     'panel_fingerprint_sha256': E.panel_fingerprint(self.manifest),
                     'frozen_maps': inventory, 'required_maps': 2,
                     'map_inventory_sha256': E.fingerprint(inventory),
                     'training_excluded_maps': ['b'],
                     'stages': {stage: {'maps': ['a', 'b'], 'seeds': seeds, 'sides': ['A', 'B'],
                                       'min_score_per_map': .6, 'min_score_each_side_per_map': .5}
                                for stage, seeds in [('standard', [0, 2026]), ('confirmation', [90101, 90107, 90121, 90133])]}}
        self.schema = {'model_version': 'residual_v1', 'action_version': 'macro_v1', 'feature_version': 1,
                       'action_dim': 17, 'network_config': {'action_dim': 17}}
        self.runtime = {'engine_version': '1.2.6', 'engine_jar_sha256': 'jar', 'engine_patch_sha256': 'patch',
                        'bridge_sha256': 'bridge', 'physics': {'limit': 1000000000}}
        self.checkpoint = self.root/'model.pt'; self.checkpoint.write_bytes(b'checkpoint')
        self.context = V.gate_context(E.file_hash(self.checkpoint), self.schema, self.runtime, self.gate, timeout_seconds=30)
        self.progress = self.root/'progress/current'
        self.ledger_dir = self.root/'runs/gate-ledgers'
        self.manifest_path = self.root/'benchmarks.json'; self.manifest_path.write_text(json.dumps(self.manifest))
        self.gate_path = self.root/'gate.json'; self.gate_path.write_text(json.dumps(self.gate))
        self.payload = {'iteration': 3, 'games_total': 0, 'config': {'maps': ['a']}}
        self.load = mock.patch.object(E, 'load_policy', return_value=(None, None, self.payload, self.schema)); self.load.start()
        self.runtime_mock = mock.patch.object(E, 'runtime_identity', return_value=self.runtime); self.runtime_mock.start()
        self.writer = mock.patch.object(V, 'write_stage_report'); self.writer.start()
        self.output = redirect_stdout(io.StringIO()); self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None); self.writer.stop(); self.runtime_mock.stop(); self.load.stop(); self.temporary.cleanup()

    def specs(self, stage='standard'):
        return V.build_specs(self.manifest, self.gate, stage, self.context, checkpoint=self.checkpoint,
                             iteration=3, maps_dir=self.maps)

    def evaluate(self, **kwargs):
        return V.evaluate_victory(self.checkpoint, progress_dir=self.progress, maps_dir=self.maps,
            manifest_path=self.manifest_path, gate_path=self.gate_path, gate_ledger_dir=self.ledger_dir,
            root=self.root, timeout_seconds=30, **kwargs)

    def publish(self, specs, score=lambda spec: 1):
        store = self.progress/'evaluation_cells'
        for spec in specs:
            E.atomic_json(store/spec['evaluation_cell_id']/'request.json', spec)
            E.atomic_json(store/spec['evaluation_cell_id']/'attempt-0001.json', successful(spec, score(spec)))
        E.merge_progress(self.progress, store)

    def test_production_inventory_and_schedule_totals(self):
        manifest = json.loads((E.ROOT/'configs/benchmarks.json').read_text())
        gate = json.loads((E.ROOT/'configs/victory_gate.json').read_text())
        V.verify_gate(manifest, gate, E.ROOT/'maps')
        self.assertEqual(len(gate['frozen_maps']), 74)
        self.assertEqual(len(gate['frozen_opponents']), 12)
        for stage, expected in [('standard', 3552), ('confirmation', 7104)]:
            count = len(gate['frozen_maps'])*len(gate['frozen_opponents'])*len(gate['stages'][stage]['seeds'])*2
            self.assertEqual(count, expected)

    def test_unsupported_search_cannot_be_mislabeled_as_evaluation_policy(self):
        for arguments in ({'search_mode': 'mcts'}, {'search_config': {'simulations': 32}}):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, 'Only greedy'):
                self.evaluate(execute=False, **arguments)

    def test_inventory_add_remove_modify_and_duplicate_fail_closed(self):
        V.verify_gate(self.manifest, self.gate, self.maps)
        path = self.maps/'extra.map26'; path.write_bytes(b'added')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            V.verify_gate(self.manifest, self.gate, self.maps)
        path.unlink(); path = self.maps/'a.map26'; path.unlink()
        with self.assertRaises(ValueError): V.verify_gate(self.manifest, self.gate, self.maps)
        path.write_bytes(b'modified')
        with self.assertRaises(ValueError): V.verify_gate(self.manifest, self.gate, self.maps)
        path.write_bytes(b'a'); gate = deepcopy(self.gate); gate['frozen_maps'].append(gate['frozen_maps'][0])
        with self.assertRaises(ValueError): V.verify_gate(self.manifest, gate, self.maps)
        (self.maps/'A.map26').write_bytes(b'case-duplicate')
        with self.assertRaisesRegex(ValueError, 'case-insensitive duplicate'):
            V.map_inventory(self.maps)

    def test_empty_partial_duplicate_and_mixed_runtime_schedules_rejected(self):
        specs = self.specs()
        for wrong in ([], specs[:-1], specs[:-1]+[specs[0]]):
            with self.assertRaisesRegex(ValueError, 'exact full'):
                V.stage_report(wrong, [], self.gate, 'standard')
        mixed = deepcopy(specs); mixed[0]['identity']['checkpoint_sha256'] = 'another'
        with self.assertRaisesRegex(ValueError, 'Mixed'):
            V.stage_report(mixed, [], self.gate, 'standard')

    def test_pooled_opponent_win_rate_cannot_hide_failed_map(self):
        specs = self.specs()
        rows = [successful(spec, 0 if spec['identity']['opponent']=='strong' and spec['identity']['map']=='b'
                           and not (spec['identity']['side']=='A' and spec['identity']['seed']==0) else 1) for spec in specs]
        report = V.stage_report(specs, rows, self.gate, 'standard')
        self.assertTrue(report['complete']); self.assertFalse(report['passed'])
        self.assertEqual(report['opponents']['strong']['mean_score'], .625)
        self.assertFalse(report['opponents']['strong']['maps']['b']['passed'])
        self.assertEqual(report['passed_opponent_map_pairs'], 3)

    def test_each_map_seat_pass_is_required_even_when_overall_passes(self):
        specs = self.specs('confirmation')
        rows = [successful(spec, 1 if spec['identity']['side']=='A' or spec['identity']['seed']==90101 else 0) for spec in specs]
        report = V.stage_report(specs, rows, self.gate, 'confirmation')
        self.assertEqual(report['opponents']['strong']['mean_score'], .625)
        self.assertFalse(report['passed'])

    def test_actual_draws_count_and_all_draws_do_not_pass(self):
        specs = self.specs()
        rows = [successful(spec, 1 if spec['identity']['seed']==0 else 0 if spec['identity']['side']=='A' else .5) for spec in specs]
        self.assertTrue(V.stage_report(specs, rows, self.gate, 'standard')['passed'])
        self.assertFalse(V.stage_report(specs, [successful(s, .5) for s in specs], self.gate, 'standard')['passed'])

    def test_missing_errors_invalid_terminal_and_stale_rows_never_pass(self):
        specs = self.specs(); rows = [successful(s) for s in specs]
        for replacement in (None, failed(specs[-1]), successful(specs[-1])):
            current = deepcopy(rows[:-1])
            if replacement:
                if replacement['status']=='ok': replacement['terminal']['valid_game'] = False
                current.append(replacement)
            report = V.stage_report(specs, current, self.gate, 'standard')
            self.assertFalse(report['passed']); self.assertEqual(report['completed'], len(specs)-1)
            self.assertEqual(report['opponents']['strong']['losses'], 0)

    def test_conflicting_successes_rejected_instead_of_selecting_favorable_replay(self):
        specs = self.specs(); rows = [successful(s) for s in specs]
        rows.append(successful(specs[0], 0, attempt=2))
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            V.stage_report(specs, rows, self.gate, 'standard')
        rows[-1] = successful(specs[0], attempt=2)
        rows[-1]['rounds'] = 101; rows[-1]['terminal']['round'] = 101
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            V.stage_report(specs, rows, self.gate, 'standard')

    def test_confirmation_lock_requires_qualification_and_one_candidate(self):
        path = self.ledger_dir/'gate.json'
        with self.assertRaisesRegex(ValueError, 'every opponent/map'):
            V.lock_confirmation(path, self.gate, self.context, standard_passed=False, audit=lambda _: {})
        ledger = V.lock_confirmation(path, self.gate, self.context, standard_passed=True, audit=lambda _: {})
        self.assertEqual(ledger['status'], 'in_progress')
        changed = deepcopy(self.context); changed['checkpoint_sha256'] = 'other'
        with self.assertRaisesRegex(ValueError, 'another candidate'):
            V.lock_confirmation(path, self.gate, changed, standard_passed=True, audit=lambda _: {})
        changed = deepcopy(self.context); changed['evaluation_config_sha256'] = 'other-runtime'
        with self.assertRaises(ValueError): V.matching_ledger(path, self.gate, changed)

    def test_report_only_cannot_bypass_pre_game_confirmation_lock(self):
        self.publish(self.specs()); self.publish(self.specs('confirmation'))
        result = self.evaluate(stage='confirmation', execute=False)
        self.assertTrue(result['score_criteria_passed'])
        self.assertFalse(result['passed']); self.assertFalse(result['victory_gate_passed'])
        self.assertFalse(result['confirmation_locked'])
        with self.assertRaisesRegex(ValueError, 'previously exposed'):
            self.evaluate(stage='confirmation')

    def test_confirmation_seed_must_be_unused_in_all_recorded_history(self):
        prior = self.root/'progress/retired/games.jsonl'
        E.atomic_text(prior, json.dumps({'seed': 90101, 'status': 'error'})+'\n')
        with self.assertRaisesRegex(ValueError, 'previously exposed'):
            V.audit_fresh_seeds(self.root, self.checkpoint, self.payload, self.gate, self.context,
                                resume=False, progress=self.progress)
        prior.unlink()
        E.atomic_json(self.root/'runs/earlier/plan-0001.json', {'jobs': [{'seed': 90107, 'game_id': 'one'}]})
        with self.assertRaisesRegex(ValueError, 'training plan'):
            V.audit_fresh_seeds(self.root, self.checkpoint, self.payload, self.gate, self.context,
                                resume=False, progress=self.progress)

    def test_trained_candidate_requires_seed_history(self):
        for extra in ({}, {'training_seeds': []}, {'training_seeds': [10]}):
            with self.assertRaisesRegex(ValueError, 'Complete training seed history'):
                V.audit_fresh_seeds(self.root, self.checkpoint, {**self.payload, 'games_total': 8, **extra}, self.gate,
                                    self.context, resume=False, progress=self.progress)

    def test_future_or_unfinished_plans_cannot_supply_consumed_training_history(self):
        jobs = []
        for index in range(8):
            job = {'game_id': f'iter-0002-game-{index:03d}', 'map': 'a', 'seed': index+10}
            job['identity'] = dict(job); jobs.append(job)
        E.atomic_json(self.checkpoint.parent/'plan-0002.json', {'jobs': jobs})
        payload = {**self.payload, 'iteration': 1, 'games_total': 8}
        with self.assertRaisesRegex(ValueError, 'Complete training seed history'):
            V.audit_fresh_seeds(self.root, self.checkpoint, payload, self.gate, self.context,
                               resume=False, progress=self.progress)
        for job in jobs:
            job['game_id'] = job['game_id'].replace('0002', '0001'); job['identity'] = {k:job[k] for k in ('game_id','map','seed')}
        E.atomic_json(self.checkpoint.parent/'plan-0001.json', {'jobs': jobs})
        with self.assertRaisesRegex(ValueError, 'Complete training seed history'):
            V.audit_fresh_seeds(self.root, self.checkpoint, payload, self.gate, self.context,
                               resume=False, progress=self.progress)
        for job in jobs:
            E.atomic_json(self.checkpoint.parent/'games'/f"{job['game_id']}.json", {**job, 'status': 'ok'})
        audit = V.audit_fresh_seeds(self.root, self.checkpoint, payload, self.gate, self.context,
                                  resume=False, progress=self.progress)
        self.assertEqual(audit['consumed_training_games'], 8)
        jobs[0]['seed'] = 123; E.atomic_json(self.checkpoint.parent/'plan-0001.json', {'jobs': jobs})
        with self.assertRaisesRegex(ValueError, 'does not match its consumed plan'):
            V.audit_fresh_seeds(self.root, self.checkpoint, payload, self.gate, self.context,
                               resume=False, progress=self.progress)

    def test_reserved_training_maps_cannot_silently_disappear(self):
        for names in (None, [], ['not-a-frozen-map']):
            gate = deepcopy(self.gate)
            if names is None: del gate['training_excluded_maps']
            else: gate['training_excluded_maps'] = names
            with self.assertRaises(ValueError): V.verify_gate(self.manifest, gate, self.maps)

    def test_pinned_warmstart_ancestry_enforces_maps_and_fresh_seeds(self):
        import torch
        directory = self.root/'artifacts/prior'; directory.mkdir(parents=True)
        dataset = directory/'source_dataset.json'
        E.atomic_json(dataset, {'games': [{'map': 'a', 'seed': 90101}]})
        prior = {'bootstrap_config': {'source_curriculum': {'maps': ['a']}},
                 'source_dataset_sha256': E.file_hash(dataset), 'benchmark_data_used': False}
        artifact = directory/'warmstart.pt'; torch.save(prior, artifact)
        payload = deepcopy(self.payload)
        payload['config']['initialization'] = {'artifact': 'artifacts/prior/warmstart.pt',
            'artifact_sha256': E.file_hash(artifact), 'source_dataset_sha256': E.file_hash(dataset)}
        lineage = V.training_lineage(self.root, self.checkpoint, payload, self.gate)
        self.assertEqual(lineage['training_seeds'], [90101]); self.assertEqual(lineage['ancestors'], 1)
        with self.assertRaisesRegex(ValueError, 'initialization ancestry'):
            V.audit_fresh_seeds(self.root, self.checkpoint, payload, self.gate, self.context,
                               resume=False, progress=self.progress)
        gate = deepcopy(self.gate); gate['training_excluded_maps'] = ['a']
        with self.assertRaisesRegex(ValueError, 'ancestry violates'):
            V.training_lineage(self.root, self.checkpoint, payload, gate)
        dataset.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'dataset provenance changed'):
            V.training_lineage(self.root, self.checkpoint, payload, self.gate)

    def test_midrun_map_or_gate_change_blocks_final_report(self):
        def mutate(*args, **kwargs):
            (self.maps/'new.map26').write_bytes(b'new')
            return None
        with mock.patch.object(E, 'run_schedule', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'map inventory changed'):
                self.evaluate()
        (self.maps/'new.map26').unlink()
        def mutate_gate(*args, **kwargs):
            gate = deepcopy(self.gate); gate['gate_id'] = 'changed'
            self.gate_path.write_text(json.dumps(gate)); return None
        with mock.patch.object(E, 'run_schedule', side_effect=mutate_gate):
            with self.assertRaisesRegex(ValueError, 'Gate configuration changed'):
                self.evaluate()

    def test_deadline_launches_no_new_games_and_reports_incomplete(self):
        with mock.patch.object(E, 'run_cell', side_effect=AssertionError('must not launch')):
            result = self.evaluate(deadline=time.monotonic()-1)
        self.assertFalse(result['passed']); self.assertEqual(result['completed'], 0)
        self.assertEqual(result['stop_reason'], 'deadline')

    def test_stop_before_confirmation_never_reserves_candidate(self):
        self.publish(self.specs())
        stop = self.root/'STOP'; stop.write_text('stop')
        with mock.patch.object(E, 'run_cell', side_effect=AssertionError('must not launch')):
            result = self.evaluate(stage='confirmation', stop_file=stop)
        self.assertFalse(result['confirmation_locked']); self.assertFalse(result['victory_gate_passed'])
        self.assertFalse((self.ledger_dir/f"{self.gate['gate_id']}.json").exists())
        self.assertEqual(result['stop_reason'], 'stop_file')
        self.assertFalse(result['executed'])

    def test_cached_results_cannot_certify_changed_checkpoint_runtime_or_lineage(self):
        self.publish(self.specs())
        def mutate_checkpoint(*args, **kwargs): self.checkpoint.write_bytes(b'replaced'); return None
        with mock.patch.object(E, 'run_schedule', side_effect=mutate_checkpoint):
            with self.assertRaisesRegex(ValueError, 'Checkpoint changed during'):
                self.evaluate()
        self.checkpoint.write_bytes(b'checkpoint')
        def mutate_runtime(*args, **kwargs):
            self.runtime['engine_patch_sha256'] = 'replaced'; return None
        # The mocked runtime must return copies, like the real reader, rather than
        # mutating the already-captured context through the same dictionary object.
        with mock.patch.object(E, 'runtime_identity', side_effect=lambda _: deepcopy(self.runtime)):
            with mock.patch.object(E, 'run_schedule', side_effect=mutate_runtime):
                with self.assertRaisesRegex(ValueError, 'runtime changed'):
                    self.evaluate()
        self.runtime['engine_patch_sha256'] = 'patch'
        original = V.training_lineage(self.root, self.checkpoint, self.payload, self.gate)
        with mock.patch.object(V, 'training_lineage', side_effect=[original, {**original, 'source_inventory_sha256': 'different'}]):
            with self.assertRaisesRegex(ValueError, 'lineage changed'):
                self.evaluate(execute=False)

    def test_changed_raw_attempt_cannot_overwrite_published_loss(self):
        specs = self.specs(); self.publish(specs, score=lambda _: 0)
        before = (self.progress/'evaluations.jsonl').read_bytes()
        for spec in specs:
            E.atomic_json(self.progress/'evaluation_cells'/spec['evaluation_cell_id']/'attempt-0001.json', successful(spec))
        with mock.patch.object(E, 'run_cell', side_effect=AssertionError('conflicting evidence must not launch')):
            with self.assertRaisesRegex(ValueError, 'Conflicting successful'):
                self.evaluate()
        self.assertEqual((self.progress/'evaluations.jsonl').read_bytes(), before)

    def test_midrun_attempt_change_is_rejected_before_merging(self):
        specs = self.specs(); self.publish(specs, score=lambda _: 0)
        before = (self.progress/'evaluations.jsonl').read_bytes()
        def mutate(*args, **kwargs):
            E.atomic_json(self.progress/'evaluation_cells'/specs[0]['evaluation_cell_id']/'attempt-0001.json', successful(specs[0]))
            return None
        with mock.patch.object(E, 'run_schedule', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'Conflicting successful'):
                self.evaluate()
        self.assertEqual((self.progress/'evaluations.jsonl').read_bytes(), before)

    def test_cached_successes_skip_workers(self):
        self.publish(self.specs())
        with mock.patch.object(E, 'run_cell', side_effect=AssertionError('successful cell cannot replay')):
            result = self.evaluate()
        self.assertTrue(result['passed']); self.assertFalse(result['victory_gate_passed'])

    def test_failed_confirmation_consumed_and_missing_completed_evidence_not_replayed(self):
        self.publish(self.specs())
        def run(spec, store, max_attempts):
            row = successful(spec, 0)
            E.atomic_json(store/spec['evaluation_cell_id']/'request.json', spec)
            E.atomic_json(store/spec['evaluation_cell_id']/'attempt-0001.json', row)
            return row
        with mock.patch.object(E, 'run_cell', side_effect=run):
            result = self.evaluate(stage='confirmation')
        self.assertTrue(result['complete']); self.assertFalse(result['victory_gate_passed'])
        path = self.ledger_dir/f"{self.gate['gate_id']}.json"
        self.assertEqual(json.loads(path.read_text())['status'], 'consumed_failed')
        with mock.patch.object(E, 'run_cell', side_effect=AssertionError('completed cell cannot replay')):
            result = self.evaluate(stage='confirmation')
        self.assertFalse(result['passed'])
        spec = self.specs('confirmation')[0]
        (self.progress/'evaluation_cells'/spec['evaluation_cell_id']/'attempt-0001.json').unlink()
        rows = [row for row in E.read_jsonl(self.progress/'evaluations.jsonl') if row['evaluation_cell_id'] != spec['evaluation_cell_id']]
        E.atomic_text(self.progress/'evaluations.jsonl', ''.join(E.canonical(row)+'\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'evidence is missing'):
            self.evaluate(stage='confirmation')

    def test_fully_qualified_locked_confirmation_can_pass_and_resume_read_only(self):
        self.publish(self.specs())
        def run(spec, store, max_attempts):
            row = successful(spec)
            E.atomic_json(store/spec['evaluation_cell_id']/'request.json', spec)
            E.atomic_json(store/spec['evaluation_cell_id']/'attempt-0001.json', row)
            return row
        with mock.patch.object(E, 'run_cell', side_effect=run):
            result = self.evaluate(stage='confirmation')
        self.assertTrue(result['victory_gate_passed'])
        self.assertTrue(self.evaluate(stage='confirmation', execute=False)['victory_gate_passed'])


if __name__ == '__main__':
    unittest.main()
