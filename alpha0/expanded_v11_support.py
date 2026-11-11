"""Unlaunched V11 downstream support: strict frozen-plan provenance and cache loading.

No actors, target scoring, fitting or benchmark execution on import.
"""
from pathlib import Path
import json
import types
import numpy as np
from alpha0 import terminal_v8_common as C
from alpha0.features_v2 import Observation
from alpha0.terminal_v8_collect import verify_terminal, CACHE_FIELDS


def read_pinned(pin):
    path=Path(pin['path']);path=path if path.is_absolute() else C.ROOT/path
    if C.sha256(path)!=pin['sha256']:raise ValueError('pinned plan changed')
    return json.loads(path.read_text())


def frozen_plans(identity):
    pins=identity['input_plans']
    if set(pins)!= {'training','validation','exclusions'}:raise ValueError('all input manifests required')
    training,validation,registry=(read_pinned(pins[k]) for k in ('training','validation','exclusions'))
    if not training['complete'] or not validation['complete']:raise ValueError('incomplete input plan')
    if len(training['roots'])!=64 or len(training['anchors'])!=1024 or len(validation['roots'])!=32:raise ValueError('fixed input denominator changed')
    if len({r['root_id'] for r in training['roots']})!=64 or len({r['root_id'] for r in validation['roots']})!=32:raise ValueError('duplicate root identifiers')
    for roots,n in ((training['roots'],2),(validation['roots'],1)):
        for m in range(8):
            for role in ('BABY_RAT','RAT_KING'):
                for player in (0,1):
                    if sum(r['map_index']==m and r['role']==role and r['player']==player for r in roots)!=n:raise ValueError('missing map/role/team stratum')
    gradient={r['input_hash'] for r in training['roots']+training['anchors']}
    excluded=set(registry['input_exclusions']);valid=[r['input_hash'] for r in validation['roots']]
    for name,digest in registry['pins'].items():
        if C.sha256(C.ROOT/name)!=digest:raise ValueError('exclusion source changed')
    if not gradient<=excluded or not set(C.historical_exclusions())<=excluded:raise ValueError('exclusion union incomplete')
    if len(set(valid))!=32 or set(valid)&excluded:raise ValueError('validation is not input-disjoint')
    # Original registry pins retain the complete V8 union, including its validation roots.
    for filename,field in [('progress/terminal-target-head-v8/roots-training.json','roots'),('progress/terminal-target-head-v8/roots-validation.json','roots'),('progress/terminal-target-head-v8/anchors.json','anchors')]:
        if filename not in registry['pins']:raise ValueError('missing V8 exclusion pin')
        rows=json.loads((C.ROOT/filename).read_text())[field]
        if not {r['input_hash'] for r in rows}<=excluded:raise ValueError('missing V8 exclusions')
    cells=2*sum(len(r['legal_actions']) for r in training['roots'])
    if cells>3200 or cells!=training['planned_selection_cells']:raise ValueError('target budget changed')
    return training,validation,registry


def verify_member(root, source, manifest, checkpoint_sha256):
    """Check exact manifest membership even when a worker adds task/action fields."""
    matches=[r for r in manifest['roots'] if r['root_id']==root.get('root_id')]
    if len(matches)!=1 or any(root.get(k)!=v for k,v in matches[0].items()):raise ValueError('root differs from frozen manifest')
    if source.get('experiment_identity')!=root.get('source_experiment_identity'):raise ValueError('source experiment mismatch')
    if source.get('checkpoint_sha256')!=checkpoint_sha256 or not source.get('complete'):raise ValueError('source checkpoint/completeness mismatch')
    if (source.get('map'),source.get('seed'),source.get('cache_sha256'))!=(root['map'],root['seed'],root['cache_sha256']):raise ValueError('source map/seed/cache mismatch')
    verify_terminal(source)
    return True


def load_root_backend(run_dir,progress_dir,check_stop,identity):
    """Reuse frozen physics/search code with a strictly stronger multi-source check."""
    training,validation,_=frozen_plans(identity)
    def provenance(root,source,actual_identity):
        if actual_identity!=identity:raise ValueError('downstream identity changed')
        manifests=[p for p in (training,validation) if any(r['root_id']==root.get('root_id') for r in p['roots'])]
        if len(manifests)!=1:raise ValueError('ambiguous or unknown root')
        return verify_member(root,source,manifests[0],identity['config']['checkpoint_sha256'])
    path=C.ROOT/'alpha0/terminal_v8_collect.py'
    if C.sha256(path)!=identity['collector_sha256']:raise ValueError('frozen backend changed')
    text=path.read_text();old='source.get("experiment_identity") != C.fingerprint(self.identity)'
    if text.count(old)!=1:raise ValueError('unexpected backend source guard')
    text=text.replace(old,'not verify_source_membership(root, source, self.identity)')
    seed_expression='"search_seed": 205201 + root["root_ordinal"]'
    if text.count(seed_expression)!=1:raise ValueError('unexpected comparator seed expression')
    text=text.replace(seed_expression,'"search_seed": config["search"]["root_seed_base"] + root["root_ordinal"]')
    module=types.ModuleType('alpha0._expanded_v11_root_backend');module.__file__=str(path)
    exec(compile(text,str(path),'exec'),module.__dict__)
    namespace=types.SimpleNamespace(**vars(C));namespace.RUN=Path(run_dir);namespace.PROGRESS=Path(progress_dir);namespace.check_stop=check_stop
    module.C=namespace;module.verify_source_membership=provenance
    return module


def cached_position(row):
    path=C.ROOT/row['cache_npz']
    if C.sha256(path)!=row['cache_sha256']:raise ValueError('cache changed')
    with np.load(path,allow_pickle=False) as cache:
        values={k:cache[k][row['feature_index']].copy() for k in CACHE_FIELDS}
    if C.input_hash(Observation(values['local'],values['global_board'],values['scalars']),values['mask'])!=row['input_hash']:raise ValueError('canonical input changed')
    if list(np.flatnonzero(values['mask']))!=sorted(row['legal_actions']):raise ValueError('legal mask changed')
    if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('nonfinite cache')
    return values


def build_complete_dataset(progress_dir,identity):
    """Assemble only the complete fixed 64-root target table; never fit a subset."""
    from alpha0.terminal_v8_collect import cell_spec,successful_cell
    training,validation,_=frozen_plans(identity);config=identity['config'];progress_dir=Path(progress_dir)
    if config['selection_seeds']!=[223003,223013] or config['selection_cell_ceiling']!=3200:raise ValueError('selection protocol changed')
    terminals=[];pucts=[];root_arrays=[];complete_cells=0
    for root in training['roots']:
        path=progress_dir/'comparators'/f"{root['root_id']}.json"
        if not path.exists():raise ValueError('comparator incomplete; no dataset')
        comparator=json.loads(path.read_text())
        spec={'experiment_identity':C.fingerprint(identity),'root_id':root['root_id'],'root_state_sha256':root['state_sha256'],'root_input_hash':root['input_hash'],'source_sha256':root['source_sha256'],'search':config['search'],'search_seed':config['search']['root_seed_base']+root['root_ordinal']}
        if comparator.get('identity')!=C.fingerprint(spec) or any(comparator.get(k)!=v for k,v in spec.items()):raise ValueError('comparator provenance changed')
        table={}
        for action in sorted(root['legal_actions']):
            scores=[]
            for seed in config['selection_seeds']:
                spec=cell_spec(root,'selection',action,seed,config,identity)
                directory=progress_dir/'cells'/f"selection-{root['root_id']}-a{action:02d}-s{seed}"
                attempts=[json.loads(p.read_text()) for p in sorted(directory.glob('attempt-[0-9][0-9][0-9].json'))]
                result=successful_cell(attempts,spec)
                if result is None:raise ValueError('target table incomplete; no dataset')
                scores.append(result['score']);complete_cells+=1
            table[action]=float(np.mean(scores))
        arrays=cached_position(root)
        visits={int(k):v for k,v in comparator['visits'].items()}
        if set(visits)!=set(root['legal_actions']) or any(type(v)!=int or v<0 for v in visits.values()) or sum(visits.values())!=32:raise ValueError('PUCT visit inventory changed')
        chosen=max(root['legal_actions'],key=lambda action:visits[action])
        if comparator['selected_action']!=chosen or not np.array_equal(np.asarray(comparator['policy_prior'],dtype=np.float32),arrays['prior']):raise ValueError('comparator prior/action changed')
        terminals.append(min(table,key=lambda a:(-table[a],-float(arrays['prior'][a]),a)));pucts.append(chosen);root_arrays.append(arrays)
    if complete_cells!=training['planned_selection_cells'] or complete_cells>3200:raise ValueError('complete selection denominator mismatch')
    anchor_arrays=[cached_position(r) for r in training['anchors']]
    games=sorted({r['source_id'] for r in training['anchors']})
    if len(games)!=64:raise ValueError('anchor source denominator changed')
    arrays={**{f'root_{key}':np.stack([r[key] for r in root_arrays]) for key in ('hidden','prior','mask')},**{f'anchor_{key}':np.stack([r[key] for r in anchor_arrays]) for key in ('hidden','prior','mask')},'terminal_actions':np.asarray(terminals,dtype=np.int64),'puct_actions':np.asarray(pucts,dtype=np.int64),'root_strata':np.asarray([[r['map_index'],int(r['role']=='RAT_KING'),r['player']] for r in training['roots']],dtype=np.int64),'anchor_game':np.asarray([games.index(r['source_id']) for r in training['anchors']],dtype=np.int64),'anchor_player':np.asarray([r['player'] for r in training['anchors']],dtype=np.int64)}
    return arrays,{'complete':True,'selection_cells':complete_cells,'input_plan_pins':identity['input_plans'],'identity_sha256':C.fingerprint(identity),'root_count':64,'anchor_count':1024,'validation_used_for_gradients':False,'benchmark_data_used':False}
