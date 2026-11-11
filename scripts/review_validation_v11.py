"""Quiescent coverage review and deterministic root/anchor freeze; no scoring/SGD."""
import fcntl,json,sys
from contextlib import ExitStack
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_validation_v11 import C,R,P,LOCKS,read,identity,report,jobs,source_row

def select(source_rows):
    roots=[];missing=[];assignments=[];anchors=[];shortfalls=[]
    def full(candidate,row,m,j):
        return {**candidate,'source_json':row['source_json'],'source_sha256':C.sha256(C.ROOT/row['source_json']),
                'source_experiment_identity':row['experiment_identity'],'cache_npz':row['cache_npz'],'cache_sha256':row['cache_sha256'],'map':row['map'],'seed':row['seed'],'map_index':m,'slot_group_j':j}
    for m in range(8):
        used=set()
        for j in range(1,3):
            roles=[('BABY_RAT',0 if j%2 else 1),('RAT_KING',1 if j%2 else 0)];todo=list(roles);chosen={}
            for n in [j]+[x for x in range(3,7) if x%2==j%2]:
                if n!=j and n in used:continue
                if n!=j:used.add(n)
                row=source_rows[f'validation-m{m}-j{n}'];filled=[]
                for role,player in list(todo):
                    candidate=row['root_candidates'].get(f'{role}:{player}')
                    if candidate:
                        chosen[role,player]=full(candidate,row,m,j);todo.remove((role,player));filled.append([role,player])
                assignments.append({'map_index':m,'group':j,'source_j':n,'filled':filled})
                if not todo:break
            for role,player in roles:
                if (role,player) in chosen:roots.append({**chosen[role,player],'root_id':f'validation-{len(roots):03d}','root_ordinal':len(roots)})
                else:missing.append({'map_index':m,'group':j,'role':role,'player':player})
    from scripts.collect_validation_v11 import exclusions
    hashes=[r['input_hash'] for r in roots];overlap=sorted(set(hashes)&set(exclusions()))
    assert not overlap
    return {'roots':roots,'assignments':assignments,'missing_root_slots':missing,
            'complete':len(roots)==32 and len(set(hashes))==32 and not missing,
            'unique_validation_inputs':len(set(hashes)),'duplicate_inputs':sorted({h for h in hashes if hashes.count(h)>1}),
            'gradient_overlap':overlap,'exclusion_sha256':C.fingerprint(exclusions()),'sgd_updates':0,'target_continuations':0}

def main():
    with ExitStack() as stack:
        for p in LOCKS:
            h=stack.enter_context(p.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        expected=read(P/'identity.json');assert identity()==expected
        measured=report(expected);assert measured['complete']
        rows={j['source_id']:source_row(j,expected)[0] for j in jobs(expected['config'])}
        result=select(rows);result.update(source_identity=C.fingerprint(expected),reviewer_sha256=C.sha256(Path(__file__)))
        C.immutable_json(P/'validation-input-plan.json',result)
        C.atomic_json(P/'coverage-review.json',{k:v for k,v in result.items() if k not in ('roots','anchors','assignments')})
        print(json.dumps({k:v for k,v in result.items() if k not in ('roots','anchors','assignments')},indent=2))
if __name__=='__main__':main()
