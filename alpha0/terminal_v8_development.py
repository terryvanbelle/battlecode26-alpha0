"""V8 full development-panel adapter, separate from historical smoke reporting."""
from pathlib import Path
import json
from collections import defaultdict
from . import evaluate as E
from . import terminal_v8_common as C
from .terminal_v8_report import ARMS, validate_benchmark_spec, select_benchmark_result, valid_benchmark_result
from scripts.compare_development import compare


def build_benchmark_specs(checkpoints, config, identity):
    if set(checkpoints) != set(ARMS):
        raise ValueError('all three fixed arms required')
    manifest_path = C.ROOT / config['benchmark_manifest']
    panel = json.loads(manifest_path.read_text())
    protocol = json.loads((C.ROOT / config['improvement_config']).read_text())
    if E.file_hash(manifest_path) != identity['panel_sha256']:
        raise ValueError('changed frozen panel')
    if E.file_hash(C.ROOT / config['improvement_config']) != identity['improvement_config_sha256']:
        raise ValueError('changed improvement rule')
    schedule = panel['evaluation_protocol']['standard']
    if len(panel['bots']) != 8 or schedule['maps'] != [m['name'] for m in protocol['maps']] or schedule['seeds'] != protocol['seeds']:
        raise ValueError('development schedule changed')
    specs = []
    for arm in ARMS:
        checkpoint = Path(checkpoints[arm]).resolve()
        sha = E.file_hash(checkpoint)
        if arm == 'cp2' and sha != protocol['baseline_sha256']:
            raise ValueError('wrong baseline')
        _, _, payload, schema = E.load_policy(checkpoint)
        runtime = E.runtime_identity(schema)
        if runtime != identity['evaluation_runtime']:
            raise ValueError('runtime changed')
        context = E.make_context(sha, schema, runtime, timeout_seconds=config['evaluation_timeout_seconds'])
        cells = E.build_cell_specs(panel, schedule, context, checkpoint=checkpoint,
                                  iteration=int(payload.get('iteration', 0)), suite='standard', stage='standard', maps_dir=C.ROOT/'maps')
        for cell in cells:
            key = dict(experiment_identity=E.fingerprint(identity), arm=arm, evaluation_cell_id=cell['evaluation_cell_id'])
            specs.append(dict(stage='benchmark', **key, cell_id=E.fingerprint(key), evaluation_spec=cell))
    if len(specs) != 768 or len({s['cell_id'] for s in specs}) != 768:
        raise ValueError('expected 768 unique fresh games')
    return specs


def summarize_benchmarks(specs, attempts, identity, *, expected_panel, expected_checkpoints=None):
    protocol = identity['improvement_config']
    schedule = expected_panel['evaluation_protocol']['standard']
    expected = {(b['id'], m, seed, side) for b in expected_panel['bots'] for m in schedule['maps']
                for seed in schedule['seeds'] for side in schedule.get('sides', ['A','B'])}
    if len(expected) != 256:
        raise ValueError('incomplete development schedule')
    histories = defaultdict(list)
    ids = {s['cell_id'] for s in specs}
    for row in attempts:
        if row['cell_id'] not in ids:
            raise ValueError('unexpected attempt')
        histories[row['cell_id']].append(row)
    if specs and (len(specs) != 768 or len(ids) != 768):
        raise ValueError('incomplete frozen spec manifest')
    panels = {arm:dict(expected=256, completed=0, unresolved_errors=0, wins=0, losses=0, draws=0,
                       score_sum=0., mean_score=None, complete=False, opponents={}, cells=[]) for arm in ARMS}
    seen = {arm:set() for arm in ARMS}; hashes=defaultdict(set); runtimes=set(); errors=0
    for spec in specs:
        validate_benchmark_spec(spec, E.fingerprint(identity))
        cell=spec['evaluation_spec']['identity'];arm=spec['arm'];p=panels[arm]
        key=(cell['opponent'],cell['map'],cell['seed'],cell['side'])
        if key not in expected or key in seen[arm] or cell['stage']!='standard':
            raise ValueError('changed development cell')
        seen[arm].add(key);hashes[arm].add(cell['checkpoint_sha256']);runtimes.add(cell['evaluation_config_sha256'])
        if cell['evaluation_config']['runtime'] != identity['evaluation_runtime']:
            raise ValueError('changed frozen runtime')
        if arm == 'cp2' and cell['checkpoint_sha256'] != protocol['baseline_sha256']:
            raise ValueError('changed baseline')
        bot=next(b for b in expected_panel['bots'] if b['id']==cell['opponent'])
        if (cell['opponent_java_sha256'],cell['opponent_source_commit']) != (bot['java_tree_sha256'],bot['source_commit']):
            raise ValueError('changed opponent source')
        if cell['panel_fingerprint_sha256']!=E.panel_fingerprint(expected_panel):
            raise ValueError('changed opponent panel')
        if cell['map_sha256']!={m['name']:m['sha256'] for m in protocol['maps']}[cell['map']]:
            raise ValueError('changed map')
        if expected_checkpoints and cell['checkpoint_sha256']!=expected_checkpoints[arm]:
            raise ValueError('changed candidate')
        history=histories[spec['cell_id']]
        errors+=sum(not valid_benchmark_result(spec,r) for r in history)
        row=select_benchmark_result(spec,history)
        if row is not None and valid_benchmark_result(spec,row):
            e=row['evaluation'];p['cells'].append(e);p['completed']+=1;p['score_sum']+=e['score']
            p[{1:'wins',0:'losses',.5:'draws'}[e['score']]]+=1
        elif row is not None:p['unresolved_errors']+=1
    if specs and (len(runtimes)!=1 or any(len(hashes[a])!=1 or seen[a]!=expected for a in ARMS)):
        raise ValueError('mixed runtime/checkpoint or missing schedule')
    for arm,p in panels.items():
        p['checkpoint_sha256']=next(iter(hashes[arm]),None)
        p['complete']=p['completed']==256 and p['unresolved_errors']==0
        p['mean_score']=p['score_sum']/256 if p['complete'] else None
    complete=all(p['complete'] for p in panels.values())
    comparisons={a:compare(panels['cp2']['cells'],panels[a]['cells'],protocol,expected_panel)
                 for a in ('terminal_head','puct_head')} if complete else {}
    return dict(expected=768,completed=sum(p['completed'] for p in panels.values()),
                unresolved_errors=sum(p['unresolved_errors'] for p in panels.values()),attempt_errors=errors,
                complete=complete,identity_verified=bool(complete and expected_checkpoints),panels=panels,
                comparisons=comparisons,development_improvement=bool(complete and comparisons['terminal_head']['passes_development_improvement_rule']),
                automatic_promotion=False)
