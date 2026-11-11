"""Quiescent coverage review and deterministic root/anchor freeze; no scoring/SGD."""
import fcntl,json,sys
from contextlib import ExitStack
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_expanded_sources_v10r1 import C,R,P,LOCKS,read,identity,report,jobs,source_row

def select(source_rows):
    roots=[];missing=[];assignments=[];anchors=[];shortfalls=[]
    def full(candidate,row,m,j):
        return {**candidate,'source_json':row['source_json'],'source_sha256':C.sha256(C.ROOT/row['source_json']),
                'source_experiment_identity':row['experiment_identity'],'cache_npz':row['cache_npz'],'cache_sha256':row['cache_sha256'],'map':row['map'],'seed':row['seed'],'map_index':m,'slot_group_j':j}
    for m in range(8):
        used=set()
        for j in range(1,5):
            roles=[('BABY_RAT',0 if j%2 else 1),('RAT_KING',1 if j%2 else 0)];todo=list(roles);chosen={}
            for n in [j]+[x for x in range(5,9) if x%2==j%2]:
                if n!=j and n in used:continue
                if n!=j:used.add(n)
                row=source_rows[f'training-m{m}-j{n}'];filled=[]
                for role,player in list(todo):
                    candidate=row['root_candidates'].get(f'{role}:{player}')
                    if candidate:
                        chosen[role,player]=full(candidate,row,m,j);todo.remove((role,player));filled.append([role,player])
                assignments.append({'map_index':m,'group':j,'source_j':n,'filled':filled})
                if not todo:break
            for role,player in roles:
                if (role,player) in chosen:roots.append({**chosen[role,player],'root_id':f'training-{len(roots):03d}','root_ordinal':len(roots)})
                else:missing.append({'map_index':m,'group':j,'role':role,'player':player})
    rng=np.random.default_rng(223153)
    for m in range(8):
        for j in range(1,9):
            row=source_rows[f'training-m{m}-j{j}']
            for team in (0,1):
                candidates=row['anchor_candidates'][str(team)]
                if len(candidates)<8:shortfalls.append({'source':row['source_id'],'team':team,'available':len(candidates)});continue
                for idx in sorted(rng.choice(len(candidates),8,replace=False).tolist()):anchors.append(full(candidates[idx],row,m,j))
    assert len({(r['source_id'],r['decision_index']) for r in roots})==len(roots)
    return {'roots':roots,'anchors':anchors,'assignments':assignments,'missing_root_slots':missing,'anchor_shortfalls':shortfalls,
            'complete':len(roots)==64 and len(anchors)==1024 and not missing and not shortfalls,
            'unique_gradient_inputs':len({r['input_hash'] for r in roots+anchors}),
            'planned_selection_cells':2*sum(len(r['legal_actions']) for r in roots),
            'sgd_updates':0,'target_continuations':0}

def main():
    with ExitStack() as stack:
        for p in LOCKS:
            h=stack.enter_context(p.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        expected=read(P/'identity.json');assert identity()==expected
        measured=report(expected);assert measured['complete']
        rows={j['source_id']:source_row(j,expected)[0] for j in jobs(expected['config'])}
        result=select(rows);result.update(source_identity=C.fingerprint(expected),reviewer_sha256=C.sha256(Path(__file__)))
        C.immutable_json(P/'training-input-plan.json',result)
        C.atomic_json(P/'coverage-review.json',{k:v for k,v in result.items() if k not in ('roots','anchors','assignments')})
        print(json.dumps({k:v for k,v in result.items() if k not in ('roots','anchors','assignments')},indent=2))
if __name__=='__main__':main()
