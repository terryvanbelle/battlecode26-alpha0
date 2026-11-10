"""Read-only audit of completed V8 panels against immutable terminal journals."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import evaluate as E
from scripts.compare_development import compare

def audit():
    progress=E.ROOT/'progress/terminal-target-head-v8'
    specs=json.loads((progress/'benchmarks/specs.json').read_text())
    if isinstance(specs,dict):specs=specs['specs']
    cfg=json.loads((E.ROOT/'configs/improvement-v4.json').read_text())
    manifest=json.loads((E.ROOT/cfg['manifest']).read_text())
    identity=json.loads((progress/'identity.json').read_text());panels={};proof={};errors=[]
    for arm in ('cp2','terminal_head','puct_head'):
        rows=[];hashes={};complete=True
        arm_specs=[s for s in specs if s['arm']==arm];assert len(arm_specs)==256
        for spec in arm_specs:
            successes=[]
            for path in sorted((progress/'benchmarks/cells'/spec['cell_id']).glob('attempt-[0-9][0-9][0-9].json')):
                r=json.loads(path.read_text());assert r['experiment_identity']==E.fingerprint(identity)
                assert r['cell_id']==spec['cell_id'] and r['arm']==arm and r['attempt'] in (1,2)
                hashes[str(path.relative_to(E.ROOT))]=E.file_hash(path)
                if r['status']=='ok':
                    assert E.valid_result(spec['evaluation_spec'],r['evaluation'])
                    assert r['attempt']==r['evaluation']['attempt']
                    successes.append(r['evaluation'])
                else:errors.append({'path':str(path.relative_to(E.ROOT)),'sha256':E.file_hash(path),'error_kind':r['error_kind'],'error':r['evaluation'].get('error')})
            assert len(successes)<=1,'successful cell repeated'
            if successes:rows+=successes
            else:complete=False
        if complete:
            panels[arm]=rows;proof[arm]={'games':256,'checkpoint_sha256':rows[0]['checkpoint_sha256'],
                'wins':sum(r['score']==1 for r in rows),'losses':sum(r['score']==0 for r in rows),'draws':sum(r['score']==.5 for r in rows),
                'terminal_journal_hashes':hashes}
    assert 'cp2' in panels and 'terminal_head' in panels
    original=E.read_jsonl(E.ROOT/'progress/development-v4/baseline/evaluations.jsonl')
    results={arm:{'vs_fresh_cp2':compare(panels['cp2'],rows,cfg,manifest),
                  'vs_original_baseline':compare(original,rows,cfg,manifest)} for arm,rows in panels.items() if arm!='cp2'}
    return {'verified_completed_panels':proof,'comparisons':results,'retained_failed_attempts':errors,
        'experiment_identity':E.fingerprint(identity),'all_three_panels_complete':len(panels)==3,
        'scope':'Completed-panel terminal evidence only; final whole-experiment/recovery audit remains separate.',
        'auditor_sha256':E.file_hash(Path(__file__))}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=audit()
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['comparisons'],indent=2))
