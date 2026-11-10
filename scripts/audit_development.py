"""Read-only audit of a complete frozen development panel and terminal evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import evaluate as E
from scripts.compare_development import compare


def audit(directory, config_path):
    cfg=json.loads(config_path.read_text());manifest=json.loads((E.ROOT/cfg['manifest']).read_text())
    rows=E.read_jsonl(directory/'evaluations.jsonl')
    self_comparison=compare(rows,rows,cfg,manifest)
    assert not self_comparison['passes_development_improvement_rule']
    if E.file_hash(E.ROOT/cfg['baseline_checkpoint']) != cfg['baseline_sha256']:
        raise ValueError('baseline checkpoint bytes changed')
    for m in cfg['maps']:
        if E.file_hash(E.ROOT/'maps'/f"{m['name']}.map26") != m['sha256']:
            raise ValueError('map bytes changed')
    attempts=E.read_jsonl(directory/'evaluation_errors.jsonl')
    requests={}
    for row in rows:
        p=directory/'evaluation_cells'/row['evaluation_cell_id']/'request.json'
        spec=json.loads(p.read_text())
        if not E.valid_result(spec,row):raise ValueError('invalid terminal cell '+row['evaluation_cell_id'])
        requests[row['evaluation_cell_id']]=E.file_hash(p)
    if attempts:raise ValueError('error journal requires individual review')
    counts=Counter({name:sum(r['score']==score for r in rows) for name,score in [('wins',1),('losses',0),('draws',.5)]})
    return {'verified':True,'suite_id':manifest['suite_id'],'checkpoint_sha256':cfg['baseline_sha256'],
            'games':len(rows),**counts,'win_rate':counts['wins']/len(rows),'unresolved_errors':0,
            'all_terminal_cells_verified':True,'config_sha256':E.file_hash(config_path),
            'manifest_sha256':E.file_hash(E.ROOT/cfg['manifest']),
            'evaluations_sha256':E.file_hash(directory/'evaluations.jsonl'),
            'request_hashes':requests,'self_comparison':self_comparison,
            'scope':'Baseline audit only; no learned improvement or promotion.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();result=audit(a.directory,a.config);E.atomic_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('request_hashes','self_comparison')},indent=2))
