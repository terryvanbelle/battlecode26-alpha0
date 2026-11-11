"""Reviewed remaining selection in <=8-cell chunks, using frozen RootWorker.

No new scientific schedule, selection, retries, fit or benchmark execution.
"""
import argparse,ctypes,fcntl,os,signal,subprocess,sys,time
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import expanded_v11_run as S
from alpha0.terminal_v8_collect import cell_spec,successful_cell
REVIEW=S.P/'reviews/chunk-recovery-20261111.json'
D=S.R/'chunk-recovery-20261111'

def remaining(expected):
    roots=S.read(S.C.ROOT/expected['input_plans']['training']['path'])['roots'];chunks=[]
    for root in roots:
        cells=[]
        for action in sorted(root['legal_actions']):
            for seed in expected['config']['selection_seeds']:
                directory=S.P/'cells'/f"selection-{root['root_id']}-a{action:02d}-s{seed}"
                rows=[S.read(p) for p in sorted(directory.glob('attempt-[0-9][0-9][0-9].json'))]
                if successful_cell(rows,cell_spec(root,'selection',action,seed,expected['config'],expected)) is None:cells.append([action,seed])
        for start in range(0,len(cells),8):chunks.append({'root':root,'cells':cells[start:start+8]})
    return chunks

def verify(review):
    if S.identity()!=S.read(S.P/'identity.json') or S.C.fingerprint(S.identity())!=review['identity_sha256']:raise ValueError('identity changed')
    if S.C.sha256(Path(__file__))!=review['runner_sha256']:raise ValueError('chunk runner changed')
    for name,digest in review['preserved_files'].items():
        if S.C.sha256(S.C.ROOT/name)!=digest:raise ValueError('original evidence changed: '+name)
    for name in review['interrupted_cells']:
        row=S.read(S.C.ROOT/name)
        if row['status']!='incomplete' or row['terminal'] or row['score'] is not None or row['error']!='Interrupted: SIGTERM':raise ValueError('not unscored supervisor interruption')

def worker(index,deadline,parent):
    ctypes.CDLL(None).prctl(1,signal.SIGKILL)
    if os.getppid()!=parent:raise RuntimeError('parent changed')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(S.C.Interrupted('SIGTERM')))
    S.torch.set_num_threads(1);review=S.read(REVIEW);expected=S.read(S.P/'identity.json');verify(review)
    chunk=review['chunks'][index];task=S.R/'tasks'/f'chunk-recovery-{index:03d}.json'
    root={**chunk['root'],'worker_task_path':str(task)}
    backend=S.load_root_backend(S.R,S.P,S.check,expected)
    w=backend.RootWorker(root,expected['config'],expected,deadline)
    try:
        w.comparator()
        for action,seed in chunk['cells']:
            S.check(deadline)
            if w.cell('selection',action,seed) is None:raise RuntimeError('cell failed; no automatic retry')
    finally:w.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);parser.add_argument('--deadline',type=float);parser.add_argument('--parent',type=int);a=parser.parse_args()
    if a.worker is not None:return worker(a.worker,a.deadline,a.parent)
    D.mkdir(exist_ok=True);deadline=time.monotonic()+10800;live={}
    with ExitStack() as stack:
        for p in S.LOCKS:
            h=stack.enter_context(p.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        S.check(deadline)
        if S.V.active_actors():raise RuntimeError('actor or unknown cwd')
        review=S.read(REVIEW);verify(review)
        if remaining(S.read(S.P/'identity.json'))!=review['chunks']:raise ValueError('remaining schedule changed')
        with (D/'started.json').open('x') as f:S.json.dump({'pid':os.getpid(),'started_at':S.C.now(),'review_sha256':S.C.sha256(REVIEW)},f)
        pending=list(range(len(review['chunks'])))
        try:
            while pending or live:
                S.check(deadline)
                while pending and len(live)<2:
                    index=pending.pop(0);log=(D/f'chunk-{index:03d}.log').open('x')
                    proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(index),'--deadline',str(deadline),'--parent',str(os.getpid())],cwd=S.C.ROOT,stdout=log,stderr=log,start_new_session=True)
                    live[proc.pid]=(proc,log,index,time.monotonic())
                for pid,(proc,log,index,started) in list(live.items()):
                    task=S.R/'tasks'/f'chunk-recovery-{index:03d}.json'
                    if time.monotonic()>started+3600 or (task.exists() and time.monotonic()>S.read(task)['deadline_monotonic']+5):raise RuntimeError('unchanged watchdog expired')
                    if proc.poll() is None:continue
                    log.close();del live[pid]
                    if proc.returncode:raise RuntimeError('chunk failed; explicit review required')
                    S.C.immutable_json(D/f'chunk-{index:03d}.json',{'complete':True,'review_sha256':S.C.sha256(REVIEW)})
                S.C.atomic_json(D/'status.json',{'active':len(live),'pending':len(pending),'updated_at':S.C.now()});time.sleep(.5)
            verify(review)
            if remaining(S.read(S.P/'identity.json')):raise ValueError('selection incomplete')
            for key in review['jobs']:
                job=S.read(S.R/'jobs'/f'{key}.json')
                S.C.immutable_json(S.P/'jobs'/f'{key}.json',{'job_sha256':S.C.sha256(S.R/'jobs'/f'{key}.json'),'status':'ok','returncode':0,'completed_at':S.C.now(),'output_hashes':S.job_outputs(job),'reviewed_recovery_sha256':S.C.sha256(REVIEW)})
            S.C.immutable_json(D/'completed.json',{'review_sha256':S.C.sha256(REVIEW),'completed_at':S.C.now(),'selection_complete':True})
        finally:
            for pid,(proc,*_) in live.items():
                try:os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:pass
            for pid,(proc,log,*_) in live.items():
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(pid,signal.SIGKILL);proc.wait()
                log.close()
if __name__=='__main__':main()
