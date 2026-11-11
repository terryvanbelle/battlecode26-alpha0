"""Guarded one-cycle handoff after the specifically reviewed V11 watchdog recovery."""
import fcntl,json,os,subprocess,sys,time
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import expanded_v11_run as S
from scripts.recover_expanded_v11_watchdog import REVIEW,verify_review
from scripts.audit_expanded_v11_selection import audit

STATUS=S.R/'watchdog-handoff-status.json'

def status(phase,**extra):
    S.C.atomic_json(STATUS,dict(phase=phase,pid=os.getpid(),updated_at=S.C.now(),**extra))

def call(args):
    subprocess.run(args,cwd=S.C.ROOT,check=True)

def quiescent():
    with ExitStack() as stack:
        for path in S.LOCKS:
            handle=stack.enter_context(path.open('a+'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        S.check(time.monotonic()+60)
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            except (OSError,UnicodeDecodeError):continue
            if command.startswith(('/workspace/battlecode26-alpha0/.venv/bin/python ','.venv/bin/python ')) and any(x in command for x in ('alpha0.expanded_v11_run','scripts/recover_expanded_v11_watchdog.py')):
                raise RuntimeError('matching actor remains; cwd need not be readable')
        verify_review(S.read(REVIEW))

def main():
    with (S.R/'watchdog-handoff.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATUS.exists():raise RuntimeError('handoff already attempted; review required')
        deadline=time.monotonic()+3600;done=S.R/'watchdog-recovery-20261111/completed.json'
        try:
            status('waiting_reviewed_recovery')
            while not done.exists():
                S.check(deadline);time.sleep(5)
            # Recovery writes completion just before releasing locks.
            while True:
                S.check(deadline)
                try:quiescent();break
                except BlockingIOError:time.sleep(1)
            review=S.read(REVIEW)
            if S.read(done)['review_sha256']!=S.C.sha256(REVIEW):raise ValueError('wrong recovery completion')
            for key in review['jobs']:
                job=S.read(S.R/'jobs'/f'{key}.json');marker=S.read(S.P/'jobs'/f'{key}.json')
                if marker['status']!='ok' or marker.get('reviewed_recovery_sha256')!=S.C.sha256(REVIEW) or marker['output_hashes']!=S.job_outputs(job):raise ValueError('recovered root job proof mismatch')
            for original in review['interrupted_cells']:
                second=S.C.ROOT/Path(original).with_name('attempt-002.json')
                if S.read(second)['status']!='ok':raise ValueError('attempt2 incomplete')
            status('auditing_and_exporting')
            result=audit();audit_path=S.P/'reviews/recovered-watchdog-audit-20261111.json';S.C.immutable_json(audit_path,result)
            call([sys.executable,'scripts/report_expanded_v11.py'])
            call([sys.executable,'scripts/export_expanded_v11.py'])
            pointer=S.read(S.P/'recovery.json');manifest=S.read(S.C.ROOT/pointer['manifest'])
            if not manifest['isolated_restore_verified'] or manifest['identity_sha256']!=review['identity_sha256']:raise ValueError('export verification failed')
            for part in pointer['parts']:
                if S.C.sha256(S.C.ROOT/part['path'])!=part['sha256']:raise ValueError('recovery part hash mismatch')
            paths=[str(audit_path.relative_to(S.C.ROOT)),'progress/expanded-terminal-v11/recovery.json','progress/expanded-terminal-v11/derived-progress.json','progress/expanded-terminal-v11/coverage.png',pointer['manifest'],*[p['path'] for p in pointer['parts']]]
            call(['git','add','--',*paths]);call(['git','commit','-m','Verify recovered V11 watchdog cells and preserve continuation state']);call(['git','push'])
            quiescent()
            status('running_one_bounded_cycle',selection_valid=result['selection_valid'],recovery_manifest=pointer['manifest'])
            with (S.R/'watchdog-handoff-cycle.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-u','-m','alpha0.expanded_v11_run','--seconds','10800'],cwd=S.C.ROOT,stdout=log,stderr=log,start_new_session=True)
                status('running_one_bounded_cycle',owner_pid=proc.pid,recovery_manifest=pointer['manifest'])
                code=proc.wait()
            status('cycle_ended_review_required',returncode=code,actor_status=S.read(S.R/'status.json'))
        except BaseException as error:
            status('review_required',reason=str(error));raise

if __name__=='__main__':main()
