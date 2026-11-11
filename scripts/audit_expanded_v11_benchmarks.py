"""Read-only V11 benchmark provenance audit; complete panels required for decisions."""
import argparse,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import expanded_v11_run as S
from alpha0 import expanded_v11_benchmark_recovery as B
from alpha0.terminal_v8_report import validate_benchmark_spec
from scripts.compare_development import compare

def audit(require_complete=False):
    B.verify()
    identity=S.read(S.P/'identity.json');frozen=S.read(S.P/'validation-actions.json')
    saved=S.read(S.P/'benchmarks/specs.json')
    specs=S.build_benchmark_specs({a:S.C.ROOT/v['path'] for a,v in frozen['checkpoints'].items()},identity['config'],identity)
    if saved['specs']!=specs or len(specs)!=768:raise ValueError('changed benchmark schedule')
    rows={a:[] for a in frozen['checkpoints']};proof=[];pending=[]
    for spec in specs:
        validate_benchmark_spec(spec,S.C.fingerprint(identity))
        key=S.C.fingerprint({'kind':'benchmark','payload':spec,'identity':S.C.fingerprint(identity)})
        marker_path=S.P/'benchmark-recovery/jobs'/f'{key}.json'
        if not marker_path.exists():pending.append(spec['cell_id']);continue
        job_path=S.R/'benchmark-recovery/jobs'/f'{key}.json';job=S.read(job_path);marker=S.read(marker_path)
        if job!={'id':key,'kind':'benchmark','payload':spec}:raise ValueError('changed job descriptor')
        if marker['status']!='ok' or marker['job_sha256']!=S.C.sha256(job_path):raise ValueError('failed/changed job marker')
        if marker['output_hashes']!=S.job_outputs(job):raise ValueError('changed completed outputs')
        files=list((S.P/'benchmarks/cells'/spec['cell_id']).glob('attempt-*.json'))
        if len(files)!=1 or files[0].name!='attempt-001.json':raise ValueError('unexpected retry')
        row=S.read(files[0])
        if not S.valid_benchmark_result(spec,row):raise ValueError('invalid benchmark record')
        rows[spec['arm']].append(row['evaluation'])
        proof.append({'job_id':key,'marker_sha256':S.C.sha256(marker_path),'output_hashes':marker['output_hashes']})
    expected_ids={s['cell_id'] for s in specs}
    if any(p.parent.name not in expected_ids for p in (S.P/'benchmarks/cells').glob('*/attempt-*.json')):raise ValueError('extra benchmark cell')
    complete=not pending
    if require_complete and not complete:raise ValueError('incomplete benchmark panels; no decision')
    result={'created_at':S.C.now(),'identity_sha256':S.C.fingerprint(identity),'complete':complete,'expected':768,'verified_completed_jobs':len(proof),'pending_cell_ids':pending,'jobs':proof,'counts':{a:{'valid':len(rs),'wins':sum(r['score']==1 for r in rs),'losses':sum(r['score']==0 for r in rs),'draws':sum(r['score']==.5 for r in rs)} for a,rs in rows.items()},'automatic_promotion':False}
    if complete:
        cfg=S.read(S.C.ROOT/'configs/improvement-v4.json');manifest=S.read(S.C.ROOT/cfg['manifest'])
        original=[json.loads(line) for line in (S.C.ROOT/'progress/development-v4/baseline/evaluations.jsonl').read_text().splitlines()]
        result['comparisons']={}
        for arm in ('terminal_head','puct_head'):
            result['comparisons'][arm]={'versus_fresh_cp2':compare(rows['cp2'],rows[arm],cfg,manifest),'versus_original_cp2':compare(original,rows[arm],cfg,manifest)}
        result['tournament_priority_triggered']=any(sum(r['score']==1 for r in rs)>=180 for rs in rows.values())
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    result=audit(args.require_complete);S.C.immutable_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','pending_cell_ids')},indent=2))
