"""One reviewed recovery of two hash-pinned, unscored watchdog interruptions.

Preserves original job logs and cell attempts. Runs unchanged workers only for
these unfinished root jobs, whose successful cells are skipped by the worker.
No automatic retry of this recovery; no fitting or other jobs are launched.
"""
import fcntl,json,os,signal,subprocess,sys,time
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import expanded_v11_run as S

REVIEW=S.P/'reviews/watchdog-recovery-20261111.json'


def verify_review(review):
    if S.identity()!=S.read(S.P/'identity.json') or S.C.fingerprint(S.identity())!=review['identity_sha256']:
        raise ValueError('changed frozen identity')
    if set(review['roots'])!={'training-057','training-058'} or len(review['jobs'])!=2 or len(review['interrupted_cells'])!=2:
        raise ValueError('unexpected recovery scope')
    for path,sha in review['preserved_files'].items():
        if S.C.sha256(S.C.ROOT/path)!=sha:raise ValueError('changed original evidence: '+path)
    for path in review['interrupted_cells']:
        row=S.read(S.C.ROOT/path)
        if row['status']!='incomplete' or row['score'] is not None or row['terminal'] or row['error']!='Interrupted: SIGTERM':
            raise ValueError('not an unscored watchdog interruption')


def main():
    review=S.read(REVIEW);deadline=time.monotonic()+10800;live={}
    directory=S.R/'watchdog-recovery-20261111';directory.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for path in S.LOCKS:
            handle=stack.enter_context(path.open('a+'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        S.check(deadline)
        if S.V.active_actors():raise RuntimeError('another actor or unknown cwd')
        verify_review(review)
        # Exclusive journal prevents blind retry of this operational recovery.
        with (directory/'started.json').open('x') as handle:
            json.dump({'started_at':S.C.now(),'review_sha256':S.C.sha256(REVIEW),'pid':os.getpid()},handle)
        try:
            for key in review['jobs']:
                job=S.read(S.R/'jobs'/f'{key}.json')
                if job['kind']!='selection' or job['payload']['root_id'] not in review['roots'] or (S.P/'jobs'/f'{key}.json').exists():
                    raise ValueError('changed/closed job')
                log=(directory/f'{key}.log').open('x')
                proc=subprocess.Popen([sys.executable,'-u','-m','alpha0.expanded_v11_run','--worker',str(S.R/'jobs'/f'{key}.json'),'--deadline',str(deadline),'--parent',str(os.getpid())],cwd=S.C.ROOT,stdout=log,stderr=log,start_new_session=True)
                live[proc.pid]=(proc,log,job,time.monotonic())
            while live:
                S.check(deadline)
                for pid,(proc,log,job,started) in list(live.items()):
                    task=S.R/'tasks'/f"{job['id']}.json"
                    if time.monotonic()>started+3600 or (task.exists() and time.monotonic()>S.read(task)['deadline_monotonic']+5):raise RuntimeError('recovery watchdog expired')
                    if proc.poll() is None:continue
                    log.close();del live[pid]
                    if proc.returncode:raise RuntimeError('recovery worker failed; preserve and review')
                    verify_review(review)
                    S.C.immutable_json(S.P/'jobs'/f"{job['id']}.json",{'job_sha256':S.C.sha256(S.R/'jobs'/f"{job['id']}.json"),'status':'ok','returncode':0,'completed_at':S.C.now(),'output_hashes':S.job_outputs(job),'reviewed_recovery_sha256':S.C.sha256(REVIEW)})
                time.sleep(.5)
            verify_review(review)
            for original in review['interrupted_cells']:
                second=S.C.ROOT/Path(original).with_name('attempt-002.json')
                if not second.exists() or S.read(second)['status']!='ok':raise ValueError('reviewed cell did not finish on attempt2')
            S.C.immutable_json(directory/'completed.json',{'completed_at':S.C.now(),'review_sha256':S.C.sha256(REVIEW),'preserved_originals':True})
        finally:
            for pid,(proc,*_) in live.items():
                try:os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:pass
            for pid,(proc,log,*_) in live.items():
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(pid,signal.SIGKILL);proc.wait()
                log.close()

if __name__=='__main__':main()
