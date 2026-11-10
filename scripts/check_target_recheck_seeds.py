"""Check proposed continuation seeds against local recorded JSON seed fields."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROPOSED={219003,219013,219029,219041}
def integers(value):
    if type(value) is int:yield value
    elif isinstance(value,list):
        for child in value:yield from integers(child)
    elif isinstance(value,dict):
        for child in value.values():yield from integers(child)
def seeds(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if 'seed' in k.lower():yield from integers(v)
            else:yield from seeds(v)
    elif isinstance(value,list):
        for v in value:yield from seeds(v)
def check():
    files={};matches=[];errors=[];observed=set()
    for directory in ('configs','runs','progress'):
        for p in sorted((ROOT/directory).rglob('*')):
            if p.suffix not in ('.json','.jsonl'):continue
            try:
                raw=p.read_bytes();rows=[json.loads(x) for x in raw.splitlines() if x.strip()] if p.suffix=='.jsonl' else [json.loads(raw)]
                used=set(v for r in rows for v in seeds(r));observed.update(used)
                files[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
                if used&PROPOSED:matches.append({'path':str(p.relative_to(ROOT)),'seeds':sorted(used&PROPOSED)})
            except (ValueError,OSError) as e:errors.append({'path':str(p.relative_to(ROOT)),'error':str(e)})
    return {'proposed_seeds':sorted(PROPOSED),'matches':matches,'read_errors':errors,'scanned_files':len(files),
        'unique_observed_seed_integers':len(observed),'inventory_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
        'scope':'Local recorded JSON/JSONL seed fields only. Active journals may advance. Rerun after V8 quiescence and review reserved formula ranges and historical archives before final preregistration.',
        'final_preregistration':False}
if __name__=='__main__':
    result=check();out=ROOT/'diagnostics/target-recheck-planning/seed-inventory.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
