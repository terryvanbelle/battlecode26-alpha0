"""Read-only local/archive seed inventory for a new teacher pilot. No games or SGD."""
import hashlib,json,os,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_target_recheck_seeds import seeds

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();observed=set();errors=[];counts={'local':0,'archive_members':0,'archives':0};current='';chain=hashlib.sha256()
    def status(phase):
        value={'phase':phase,'pid':os.getpid(),'counts':counts,'current':current,'elapsed_seconds':round(time.monotonic()-started,1),'errors':len(errors),'teacher_games':0,'sgd_started':False}
        tmp=out/'status.tmp';tmp.write_text(json.dumps(value)+'\n');tmp.replace(out/'status.json')
    def consume(raw,name):
        chain.update(name.encode());chain.update(hashlib.sha256(raw).digest())
        try:
            if name.endswith('.jsonl'):
                for line in raw.splitlines():
                    if line.strip():observed.update(seeds(json.loads(line)))
            else:observed.update(seeds(json.loads(raw)))
        except (ValueError,RecursionError) as e:errors.append({'path':name,'error':str(e)[:300]})
    status('scanning')
    roots=[ROOT/n for n in ('configs','runs','progress','diagnostics','artifacts','build/imitation')]
    for root in roots:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.is_relative_to(out.resolve()):continue
            if time.monotonic()-started>10800:raise TimeoutError('inventory deadline; partial status retained')
            current=str(p.relative_to(ROOT))
            if p.suffix in ('.json','.jsonl'):
                try:consume(p.read_bytes(),current);counts['local']+=1
                except OSError as e:errors.append({'path':current,'error':str(e)})
            elif p.name.endswith(('.tar.gz','.tgz')):
                status('scanning_archive')
                try:
                    with tarfile.open(p,'r|gz') as archive:
                        for member in archive:
                            if member.isfile() and member.name.endswith(('.json','.jsonl')):
                                consume(archive.extractfile(member).read(),current+'::'+member.name);counts['archive_members']+=1
                    counts['archives']+=1
                except (OSError,tarfile.TarError) as e:errors.append({'path':current,'error':str(e)})
            if sum(counts.values())%100==0:status('scanning')
    result={'counts':counts,'observed_seed_integers':sorted(observed),'read_errors':errors,'inventory_sha256':chain.hexdigest(),'final_preregistration':False,'scope':'JSON seed fields in local configs/runs/progress/diagnostics/artifacts/build-imitation and complete tar.gz archives; formulas and split archives require separate review'}
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2)
    current='';status('complete_review_required')
if __name__=='__main__':run(sys.argv[1])
