"""Derive development dashboard inputs from V11's immutable benchmark attempts.

No games, fitting, candidate selection or promotion. Canonical evidence remains
in progress/expanded-terminal-v11/benchmarks. Missing cells remain missing.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import evaluate as E
from alpha0.terminal_v8_report import select_benchmark_result,validate_benchmark_spec,valid_benchmark_result
ROOT=E.ROOT


def publish(progress=None, output=None):
    progress=Path(progress) if progress is not None else ROOT/'progress/expanded-terminal-v11'
    output=Path(output) if output is not None else ROOT/'progress/development-v4'
    if not (progress/'benchmarks/specs.json').exists():
        return {'status':'benchmark_not_started','playing_strength_claim':False}
    identity=json.loads((progress/'identity.json').read_text())
    frozen=json.loads((progress/'benchmarks/specs.json').read_text())
    assert frozen['identity_sha256']==E.fingerprint(identity)
    specs=frozen['specs'];assert len(specs)==768 and len({s['cell_id'] for s in specs})==768
    # Validate the entire frozen schedule before writing any derived views.
    from alpha0.terminal_v8_development import summarize_benchmarks
    actions=json.loads((progress/'validation-actions.json').read_text())
    assert actions['identity_sha256']==E.fingerprint(identity)
    panel_path=ROOT/identity['config']['benchmark_manifest']
    assert E.file_hash(panel_path)==identity['panel_sha256']
    attempts=[json.loads(p.read_text()) for p in (progress/'benchmarks/cells').glob('*/attempt-[0-9][0-9][0-9].json')]
    summarize_benchmarks(specs,attempts,identity,
        expected_panel=json.loads(panel_path.read_text()),
        expected_checkpoints={arm:row['sha256'] for arm,row in actions['checkpoints'].items()})
    report={}
    for arm in ('cp2','terminal_head','puct_head'):
        group=[s for s in specs if s['arm']==arm];assert len(group)==256
        rows=[];proof=[]
        for spec in group:
            validate_benchmark_spec(spec,E.fingerprint(identity))
            files=sorted((progress/'benchmarks/cells'/spec['cell_id']).glob('attempt-[0-9][0-9][0-9].json'))
            history=[json.loads(p.read_text()) for p in files]
            row=select_benchmark_result(spec,history)
            if row is None:continue
            if row['status']=='ok':assert valid_benchmark_result(spec,row)
            rows.append(row['evaluation'])
            proof.extend({'path':str(p.relative_to(ROOT)),'sha256':E.file_hash(p)} for p in files)
        if not rows:continue
        out=output/f'v11-{arm}'
        out.mkdir(parents=True,exist_ok=True)
        E.atomic_json(out/'source.json',{'experiment_identity':E.fingerprint(identity),'arm':arm,
                    'canonical_specs_sha256':E.file_hash(progress/'benchmarks/specs.json'),'attempt_files':proof,
                    'derived_dashboard_view':True,'automatic_promotion':False})
        tmp=out/'evaluations.jsonl.tmp'
        tmp.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows));tmp.replace(out/'evaluations.jsonl')
        valid=[r for r in rows if r['status']=='ok']
        counts={'valid':len(valid),'expected':256,'wins':sum(r['score']==1 for r in valid),
                'losses':sum(r['score']==0 for r in valid),'draws':sum(r['score']==.5 for r in valid),
                'unresolved_errors':len(rows)-len(valid),'complete':len(valid)==256,
                'checkpoint_sha256':rows[0]['checkpoint_sha256']}
        E.atomic_json(out/'coverage.json',counts);report[arm]=counts
    return report

if __name__=='__main__':print(json.dumps(publish(),indent=2))
