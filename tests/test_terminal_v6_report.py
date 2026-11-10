"""Evidence-only checks for paired transfer, fresh panels and immutable attempts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from alpha0 import evaluate as E
from alpha0 import terminal_v6_report as R
from alpha0 import terminal_v6_common as C
from tests.test_evaluation_runner import successful


def validation_roots():
    return [{'root_id': f'validation-{i:03d}', 'map': 'tiny' if i==0 else 'closeup',
             'role': 'BABY_RAT' if i==0 else 'RAT_KING', 'player': i,
             'input_hash': f'input-{i}', 'state_sha256': f'state-{i}', 'prefix_sha256': f'prefix-{i}',
             'source_sha256': f'source-{i}', 'source_json': f'source-{i}.json',
             'legal_actions': [0,1,2,37], 'actions': {'cp2': 0, 'puct_head': 1, 'terminal_head': 2}}
            for i in range(2)]


def continuation(root, action, seed, score, config, identity, *, attempt=1, stage='validation'):
    spec = R.continuation_spec(root, stage, action, seed, config, identity)
    winner = -1 if score==.5 else root['player'] if score==1 else 1-root['player']
    raw = {'terminal': True, 'valid_game': True, 'benchmark_failures': [], 'winner': winner, 'round': 20}
    return {**spec, 'identity': C.fingerprint(spec), 'attempt': attempt, 'status': 'ok', 'score': score,
            'terminal': True, 'valid_game': True, 'benchmark_failures': [], 'winner': winner, 'terminal_round': 20,
            'terminal_state': raw, 'terminal_state_sha256': C.fingerprint(raw),
            'terminal_summary': {key:raw.get(key) for key in ('terminal','round','winner','win_reason','bank','kings','transferred','cat_damage')},
            'actions': [action], 'actions_sha256': C.fingerprint([action]), 'decisions': 1}


class TerminalV6ReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.config = {'validation_roots': 2, 'training_roots': 2, 'validation_seeds': R.VALIDATION_SEEDS,
                       'selection_seeds': [203003,203013], 'temperature': .35, 'selection_cell_ceiling': 1600}
        self.identity = {'experiment_id': 'unit-v6'}
        self.roots = validation_roots()

    def tearDown(self): self.temp.cleanup()

    def validation_rows(self, scores=None):
        scores = scores or [{0:.5, 1:.5, 2:1} for _ in self.roots]
        return [continuation(root, action, seed, scores[index][action], self.config, self.identity)
                for index, root in enumerate(self.roots) for action in sorted(set(root['actions'].values()))
                for seed in R.VALIDATION_SEEDS]

    def benchmark_fixture(self, count=2):
        maps = self.root/'maps'; maps.mkdir(exist_ok=True); (maps/'tiny.map26').write_bytes(b'map')
        panel = {'suite_id': 'frozen', 'bots': [{'id': f'bot-{i}', 'source_commit': f'commit-{i}',
                 'java_tree_sha256': f'java-{i}'} for i in range(count)],
                 'evaluation_protocol': {'smoke': {'maps':['tiny'], 'seeds':[0], 'sides':['A','B']}}}
        schema = {'model_version':'dual_scale_v2','action_version':'macro_v2','feature_version':2,
                  'action_dim':40,'network_config':{'action_dim':40}}
        runtime = {'engine_version':'1.2.6','engine_jar_sha256':'jar','engine_patch_sha256':'patch','bridge_sha256':'bridge'}
        self.identity.setdefault('evaluation_runtime', runtime)
        checkpoints = {arm:f'{arm}-sha' for arm in R.ARMS}; specs=[]
        self.identity['checkpoint_sha256'] = checkpoints['cp2']
        for arm in R.ARMS:
            context=E.make_context(checkpoints[arm],schema,runtime,timeout_seconds=1800)
            for evaluation in E.build_cell_specs(panel,panel['evaluation_protocol']['smoke'],context,
                    checkpoint=self.root/f'{arm}.pt',iteration=2,suite='smoke',stage='smoke',maps_dir=maps):
                key={'experiment_identity':E.fingerprint(self.identity),'arm':arm,'evaluation_cell_id':evaluation['evaluation_cell_id']}
                specs.append({'stage':'benchmark',**key,'cell_id':E.fingerprint(key),'evaluation_spec':evaluation})
        return panel,checkpoints,specs,schema,runtime

    def summary_bench(self,specs,rows,panel,pins):
        return R.summarize_benchmarks(specs,rows,self.identity,required_opponents=len(panel['bots']),
                                     expected_panel=panel,expected_checkpoints=pins)

    def test_complete_paired_validation_and_same_action_physical_dedup(self):
        self.roots[1]['actions']['cp2']=2
        rows=self.validation_rows()
        result=R.summarize_validation(self.roots,rows,self.config,self.identity)
        self.assertTrue(result['complete']); self.assertTrue(result['criteria_pass'])
        self.assertEqual(result['expected_unique_cells'],20)
        self.assertEqual(result['completed_unique_cells'],20)
        self.assertEqual(result['roots'][1]['scores']['cp2'],result['roots'][1]['scores']['terminal_head'])
        self.assertEqual(result['distinct_source_games_represented'],2)

    def test_partial_validation_never_supplies_zero_or_reduced_denominator(self):
        result=R.summarize_validation(self.roots,self.validation_rows()[:-1],self.config,self.identity)
        self.assertFalse(result['complete']); self.assertFalse(result['criteria_pass'])
        self.assertIsNone(result['mean_delta_vs_puct']); self.assertIsNone(result['roots'][-1]['means']['terminal_head'])
        self.assertEqual(result['completed_roots'],1)
        self.assertFalse(R.summarize_validation(self.roots[:1],self.validation_rows()[:12],self.config,self.identity)['complete'])

    def test_validation_requires_two_improved_roots_and_limits_each_regression(self):
        rows=self.validation_rows([{0:0,1:0,2:1},{0:0,1:1,2:.5}])
        result=R.summarize_validation(self.roots,rows,self.config,self.identity)
        self.assertEqual(result['mean_delta_vs_puct'],.25)
        self.assertFalse(result['criteria_pass']); self.assertEqual(result['worst_root_delta_vs_puct'],-.5)
        rows=self.validation_rows([{0:0,1:0,2:1},{0:.5,1:.5,2:.5}])
        self.assertFalse(R.summarize_validation(self.roots,rows,self.config,self.identity)['criteria_pass'])

    def test_cp2_nonnegative_rule_is_independent_of_puct_gain(self):
        rows=self.validation_rows([{0:1,1:0,2:.5},{0:1,1:0,2:.5}])
        result=R.summarize_validation(self.roots,rows,self.config,self.identity)
        self.assertEqual(result['mean_delta_vs_puct'],.5); self.assertEqual(result['mean_delta_vs_cp2'],-.5)
        self.assertFalse(result['criteria_pass'])

    def test_wrong_validation_seed_identity_winner_or_duplicate_success_rejected(self):
        for mutation in ('seed','identity','winner','score_bool','replay'):
            rows=self.validation_rows()
            if mutation=='seed': rows[0]['seed']=203003
            elif mutation=='identity': rows[0]['identity']='stale'
            elif mutation=='winner': rows[0]['winner']=0
            elif mutation=='score_bool': rows[0]['score']=True
            else:
                duplicate=deepcopy(rows[0]); duplicate['attempt']=2; rows.append(duplicate)
            with self.assertRaises(ValueError,msg=mutation): R.summarize_validation(self.roots,rows,self.config,self.identity)

    def test_identical_same_attempt_reference_is_not_extra_validation_evidence(self):
        rows=self.validation_rows(); rows.append(deepcopy(rows[0]))
        result=R.summarize_validation(self.roots,rows,self.config,self.identity)
        self.assertEqual(result['completed_unique_cells'],24)

    def test_selection_cannot_fit_on_partial_or_over_budget_root_coverage(self):
        rows=[continuation(root,a,seed,1,self.config,self.identity,stage='selection') for root in self.roots
              for a in root['legal_actions'] for seed in self.config['selection_seeds']]
        self.assertTrue(R.summarize_selection(self.roots,rows,self.config,self.identity)['complete'])
        self.assertFalse(R.summarize_selection(self.roots,rows[:-1],self.config,self.identity)['complete'])
        config={**self.config,'selection_cell_ceiling':1}
        self.assertFalse(R.summarize_selection(self.roots,rows,config,self.identity)['complete'])

    def test_fresh_benchmark_panels_and_each_opponent_nonregression(self):
        panel,pins,specs,_,_=self.benchmark_fixture()
        rows=[]
        for spec in specs:
            # Both cp2 and candidate aggregate 2/4, but candidate forgets bot-0.
            bot=spec['evaluation_spec']['identity']['opponent']
            score=1 if (spec['arm']=='cp2' and bot=='bot-0') or (spec['arm']=='terminal_head' and bot=='bot-1') else 0
            rows.append(R.wrap_benchmark_result(spec,successful(spec['evaluation_spec'],score),1))
        result=self.summary_bench(specs,rows,panel,pins)
        self.assertTrue(result['complete']); self.assertTrue(result['identity_verified'])
        self.assertEqual(result['panels']['cp2']['mean_score'],result['panels']['terminal_head']['mean_score'])
        self.assertFalse(result['per_opponent_nonregression'])
        self.assertEqual(result['opponent_comparisons']['bot-0']['delta'],-1)

    def test_error_then_success_counts_once_but_preserves_error_history(self):
        panel,pins,specs,_,_=self.benchmark_fixture()
        rows=[R.wrap_benchmark_result(spec,successful(spec['evaluation_spec']),1) for spec in specs[1:]]
        rows += [R.benchmark_error_record(specs[0],1,'worker_timeout','watchdog'),
                 R.wrap_benchmark_result(specs[0],successful(specs[0]['evaluation_spec'],attempt=2),2)]
        result=self.summary_bench(specs,rows,panel,pins)
        self.assertTrue(result['complete']); self.assertEqual(result['attempt_errors'],1)
        self.assertEqual(result['unresolved_errors'],0); self.assertEqual(result['completed'],len(specs))

    def test_missing_smoke_cell_is_not_a_loss_and_historical_baseline_cannot_fill_it(self):
        panel,pins,specs,_,_=self.benchmark_fixture()
        rows=[R.wrap_benchmark_result(spec,successful(spec['evaluation_spec']),1) for spec in specs[:-1]]
        result=self.summary_bench(specs,rows,panel,pins)
        self.assertFalse(result['complete']); self.assertIsNone(result['panels']['puct_head']['mean_score'])
        self.assertEqual(result['panels']['puct_head']['losses'],0)
        self.assertFalse(result['historical_cp2_score']['fresh'])

    def test_benchmark_replay_and_substituted_expected_pins_rejected(self):
        panel,pins,specs,_,_=self.benchmark_fixture()
        rows=[R.wrap_benchmark_result(spec,successful(spec['evaluation_spec']),1) for spec in specs]
        repeat=R.wrap_benchmark_result(specs[0],successful(specs[0]['evaluation_spec'],attempt=2),2)
        with self.assertRaisesRegex(ValueError,'replayed'): self.summary_bench(specs,rows+[repeat],panel,pins)
        with self.assertRaisesRegex(ValueError,'candidate differs'): self.summary_bench(specs,rows,panel,{**pins,'terminal_head':'replacement'})
        wrong=deepcopy(panel); wrong['bots'][0]['source_commit']='replacement'
        with self.assertRaisesRegex(ValueError,'opponent identities'): self.summary_bench(specs,rows,wrong,pins)
        with self.assertRaisesRegex(ValueError,'all three complete'): self.summary_bench(specs[:-1],rows,panel,pins)

    def test_atomic_attempt_helper_never_replays_success_and_caps_infrastructure_retry(self):
        _,_,specs,_,_=self.benchmark_fixture(); spec=specs[0]; directory=self.root/'cell'
        failure=R.benchmark_error_record(spec,1,'worker_timeout','watchdog')
        success=R.wrap_benchmark_result(spec,successful(spec['evaluation_spec'],attempt=2),2)
        with mock.patch.object(R,'benchmark_worker',side_effect=[failure,success]) as worker:
            self.assertEqual(R.record_benchmark_attempt(spec,directory,attempt=1)['status'],'error')
            self.assertEqual(R.record_benchmark_attempt(spec,directory,attempt=2)['status'],'ok')
            self.assertEqual(R.record_benchmark_attempt(spec,directory,attempt=2),success)
            self.assertEqual(worker.call_count,2)
        self.assertTrue((directory/'attempt-001.json').exists())
        with self.assertRaises(ValueError): R.record_benchmark_attempt(spec,directory,attempt=3)

    def test_benchmark_attempts_require_manifest_and_frozen_map(self):
        with self.assertRaisesRegex(ValueError, 'without the frozen'):
            R.summarize_benchmarks([], [{'status':'ok'}], self.identity)
        self.identity['maps'] = [{'name':'tiny','sha256':'substituted-map'}]
        panel,pins,specs,_,_=self.benchmark_fixture()
        with self.assertRaisesRegex(ValueError, 'frozen tiny map'):
            self.summary_bench(specs, [], panel, pins)

    def test_self_consistent_benchmark_runtime_substitution_is_rejected(self):
        self.identity['evaluation_runtime'] = {'engine_version':'substituted'}
        panel,pins,specs,_,_=self.benchmark_fixture()
        rows=[R.wrap_benchmark_result(spec,successful(spec['evaluation_spec']),1) for spec in specs]
        with self.assertRaisesRegex(ValueError, 'frozen experiment runtime'):
            self.summary_bench(specs, rows, panel, pins)

    def test_benchmark_start_journal_never_counts_as_finished_attempt(self):
        _,_,specs,_,_=self.benchmark_fixture(); spec=specs[0]; directory=self.root/'cell'
        E.atomic_json(directory/'attempt-001-start.json', {'attempt':1,'status':'incomplete'})
        success=R.wrap_benchmark_result(spec,successful(spec['evaluation_spec']),1)
        with mock.patch.object(R, 'benchmark_worker', return_value=success) as worker:
            self.assertEqual(R.record_benchmark_attempt(spec,directory,attempt=1), success)
            worker.assert_called_once()

    def test_builds_72_separate_wrappers_even_for_identical_checkpoint_bytes(self):
        panel,_,_,schema,runtime=self.benchmark_fixture(count=12)
        path=self.root/'manifest.json'; path.write_text(json.dumps(panel))
        checkpoint=self.root/'same.pt'; checkpoint.write_bytes(b'same')
        config={'benchmark_manifest':str(path),'maps_dir':str(self.root/'maps'),
                'checkpoint_sha256':E.file_hash(checkpoint),'evaluation_timeout_seconds':1800}
        identity={'panel_sha256':E.file_hash(path),'evaluation_runtime':runtime}
        with mock.patch.object(E,'load_policy',return_value=(None,None,{'iteration':2},schema)), mock.patch.object(E,'runtime_identity',return_value=runtime):
            specs=R.build_benchmark_specs({arm:checkpoint for arm in R.ARMS},config,identity)
        self.assertEqual(len(specs),72); self.assertEqual(len({spec['cell_id'] for spec in specs}),72)
        self.assertEqual(len({spec['evaluation_cell_id'] for spec in specs}),24)
        self.assertEqual(len({spec['evaluation_spec']['identity']['evaluation_config_sha256'] for spec in specs}),1)

    def test_empty_report_is_read_only_and_cannot_claim_expansion(self):
        config=json.loads((E.ROOT/'configs/terminal-target-head-v6.json').read_text())
        identity={'config':config,'panel_sha256':E.file_hash(E.ROOT/'configs/benchmarks.json')}
        result=R.report(self.root/'run',self.root/'progress',config,identity,integrity_verified=True)
        self.assertFalse(result['complete']); self.assertFalse(result['decision']['review_for_expansion'])
        self.assertFalse(result['decision']['all_map_victory']); self.assertFalse((self.root/'progress').exists())
        self.assertEqual(result['validation']['expected_roots'],16); self.assertEqual(result['benchmarks']['expected'],72)

    def test_source_start_without_end_is_counted_but_never_scored(self):
        run=self.root/'run'; path=run/'sources/training-m0-j1.attempts/attempt-001-start.json'
        row={'experiment_identity':E.fingerprint(self.identity),'split':'training','source_id':'training-m0-j1',
             'attempt':1,'status':'incomplete','terminal':False,'score':None,'source':'selfplay','temperature':1.0,'source_temperature':1.0}
        E.atomic_json(path,row)
        manifest={'roots':[],'sources':{},'assignments':[],'unavailable_slots':[],'complete':False}
        result=R.source_coverage(run,{'training':manifest,'validation':manifest},self.identity)
        self.assertEqual(result['training']['attempted_source_games'],1)
        self.assertEqual(result['training']['completed_source_games'],0)
        self.assertEqual(result['training']['start_only_attempts'],1)
        self.assertEqual(result['training']['unresolved_source_games'],1)

    def test_completed_source_cache_and_full_discovery_inventory_are_required(self):
        run=self.root/'run'; source_id='training-m0-j1'
        cache=run/'source-cache.npz'; cache.parent.mkdir(parents=True); cache.write_bytes(b'canonical input cache')
        row=continuation(self.roots[0],0,R.VALIDATION_SEEDS[0],1,self.config,self.identity)
        row.update(split='training',source_id=source_id,cache_npz=str(cache),cache_sha256=E.file_hash(cache),source='selfplay',temperature=1.0,source_temperature=1.0)
        source=run/'sources'/f'{source_id}.json'; E.atomic_json(source,row)
        manifest={'roots':[],'sources':{source_id:{'source_json':str(source),'source_sha256':E.file_hash(source)}},
                  'assignments':[],'unavailable_slots':[],'complete':True}
        self.assertEqual(R.source_coverage(run,{'training':manifest},self.identity)['training']['completed_source_games'],1)
        omitted={**manifest,'sources':{}}
        with self.assertRaisesRegex(ValueError,'omits completed discovery'):
            R.source_coverage(run,{'training':omitted},self.identity)
        cache.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'input cache changed'):
            R.source_coverage(run,{'training':manifest},self.identity)

    def test_source_cache_from_earlier_run_is_rejected_even_with_matching_hash(self):
        run=self.root/'run'; source_id='training-m0-j1'
        cache=self.root/'earlier-v5-cache.npz'; cache.write_bytes(b'old source cache')
        row=continuation(self.roots[0],0,R.VALIDATION_SEEDS[0],1,self.config,self.identity)
        row.update(split='training',source_id=source_id,cache_npz=str(cache),cache_sha256=E.file_hash(cache),source='selfplay',temperature=1.0,source_temperature=1.0)
        E.atomic_json(run/'sources'/f'{source_id}.json',row)
        manifest={'roots':[],'sources':{},'assignments':[],'unavailable_slots':[],'complete':False}
        with self.assertRaisesRegex(ValueError,'freshly generated run'):
            R.source_coverage(run,{'training':manifest},self.identity)

    def test_source_temperature_cannot_silently_remain_at_v5_setting(self):
        run=self.root/'run'
        row={'experiment_identity':E.fingerprint(self.identity),'split':'training','source_id':'training-m0-j1',
             'attempt':1,'status':'incomplete','terminal':False,'score':None,'source':'selfplay','temperature':.35,'source_temperature':.35}
        E.atomic_json(run/'sources/training-m0-j1.attempts/attempt-001-start.json',row)
        manifest={'roots':[],'sources':{},'assignments':[],'unavailable_slots':[],'complete':False}
        with self.assertRaisesRegex(ValueError,'temperature 1.0'):
            R.source_coverage(run,{'training':manifest},self.identity)


if __name__=='__main__': unittest.main()
