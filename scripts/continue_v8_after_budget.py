"""One guarded, reviewed continuation after a V8 budget boundary.

Never changes an experiment setting. Exports, verifies and restore-checks state
before one further bounded invocation; genuine failures and STOPs require review.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import terminal_v8_common as C
from alpha0.terminal_v8 import active_actors
from alpha0.terminal_v8_report import report
from scripts import export_terminal_v8 as X
from scripts.audit_v8_fits import audit as audit_fits
from scripts.audit_v8_inputs import audit as audit_inputs


def review_status(status, summary):
    if status.get('phase') not in ('paused','complete_review_required'):
        raise ValueError('not a reviewed pause or complete boundary')
    if status.get('phase')=='paused' and status.get('reason')!='deadline':
        raise ValueError('pause was not a budget deadline')
    if status.get('runtime_and_checkpoint_unchanged') is not True:
        raise ValueError('runtime integrity not certified')
    if not all(summary[k].get('complete') for k in ('selection','fit','validation')):
        raise ValueError('this continuation is only for unfinished benchmark evaluation')
    if summary['benchmarks'].get('complete'):
        return False
    if status['phase']!='paused':raise ValueError('inconsistent completed cycle')
    if summary['benchmarks'].get('unresolved_errors',0)>2:
        raise ValueError('more unfinished workers than the boundary permits')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-seconds',type=float,default=3600)
    p.add_argument('--cycle-seconds',type=float,default=10800);a=p.parse_args()
    if not 0<a.wait_seconds<=10800 or not 0<a.cycle_seconds<=10800:p.error('maximum three hours per wait/cycle')
    d=C.ROOT/'runs/research-sequence';d.mkdir(parents=True,exist_ok=True)
    def stopped():return (d/'STOP').exists() or any(p.exists() for p in C.STOPS)
    with (d/'boundary.lock').open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=C.load_config();expected=json.loads((C.PROGRESS/'identity.json').read_text())
        if C.identity(cfg)!=expected:raise ValueError('existing frozen runtime changed')
        initial=json.loads((C.RUN/'status.json').read_text())
        def state(phase,**fields):
            value=dict(phase=phase,pid=os.getpid(),updated_at=C.now(),identity_sha256=C.fingerprint(expected),**fields)
            C.atomic_json(d/'boundary-status.json',value);print(json.dumps(value),flush=True)
        state('waiting_for_budget_boundary',observed_owner=initial.get('pid'))
        deadline=time.monotonic()+a.wait_seconds
        try:
            while time.monotonic()<deadline:
                if stopped():state('stopped');return
                status=json.loads((C.RUN/'status.json').read_text())
                if status.get('phase') in ('paused','complete_review_required','review_required','invalidated'):
                    try:
                        with ExitStack() as stack:
                            for path in C.LOCKS:
                                h=stack.enter_context(path.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                            if active_actors():raise BlockingIOError('actors remain')
                    except BlockingIOError:time.sleep(5);continue
                    break
                time.sleep(5)
            else:state('wait_deadline_reached');return
            if C.identity(cfg)!=expected:raise ValueError('runtime changed during wait')
            summary=report(C.RUN,C.PROGRESS,cfg,expected,integrity_verified=True)
            resume=review_status(status,summary)
            # Preserve and permit only first-attempt watchdog interruptions at this boundary.
            failures=[]
            for path in (C.PROGRESS/'benchmarks/cells').glob('*/attempt-[0-9][0-9][0-9].json'):
                row=json.loads(path.read_text())
                if row['status']!='ok':
                    if row.get('error_kind')!='worker_timeout' or row.get('attempt')!=1:
                        raise ValueError('non-boundary benchmark failure requires review')
                    failures.append(str(path.relative_to(C.ROOT)))
            if len(failures)>2:raise ValueError('unexpected failure count')
            if audit_fits()!=json.loads((C.PROGRESS/'reviews/fit-audit-20261110.json').read_text()):
                raise ValueError('saved fit audit changed')
            if audit_inputs()!=json.loads((C.PROGRESS/'reviews/input-audit-20261110.json').read_text()):
                raise ValueError('saved input audit changed')
            if stopped():state('stopped');return
            state('exporting_verified_boundary',benchmark_completed=summary['benchmarks']['completed'])
            exported=X.create_export(C.ROOT);archive=Path(exported['archive'])
            verified=X.verify(archive,expected_sha256=exported['sha256'])
            with tempfile.TemporaryDirectory(prefix='v8-cycle-restore-') as tmp:
                restored=X.restore(archive,Path(tmp),expected_sha256=exported['sha256'])
            proof={'export':exported,'verification':verified,'isolated_restore':restored,
                   'boundary_status':status,'benchmark_completed':summary['benchmarks']['completed'],
                   'expected_benchmark_games':768,'watchdog_attempts_preserved':failures,'resume_eligible':resume}
            proof_path=archive.with_name(archive.name+'.verification.json');C.immutable_json(proof_path,proof)
            pointer=C.PROGRESS/'recovery.json';C.atomic_json(pointer,{'archive':str(archive.relative_to(C.ROOT)),
                   'sha256':exported['sha256'],'verification':str(proof_path.relative_to(C.ROOT)),
                   'partial':not summary['complete'],'benchmark_completed':summary['benchmarks']['completed']})
            readme=archive.parent/'README.md'
            previous=readme.read_text() if readme.exists() else '# V8 verified recovery generations\n\n'
            previous+=f"- [{archive.name}]({archive.name}), SHA256 `{exported['sha256']}`; {summary['benchmarks']['completed']}/768 benchmark games. [Verification and isolated restore]({proof_path.name}).\n"
            readme.write_text(previous)
            if archive.stat().st_size>=95*1024*1024:raise ValueError('archive needs chunked publication review')
            paths=[str(p.relative_to(C.ROOT)) for p in (archive,Path(exported['manifest']),proof_path,pointer,readme)]
            subprocess.run(['git','add','--',*paths],cwd=C.ROOT,check=True)
            subprocess.run(['git','commit','--quiet','--only','-m','Publish verified V8 budget-boundary recovery','--',*paths],cwd=C.ROOT,check=True)
            subprocess.run(['git','push'],cwd=C.ROOT,check=True)
            if not resume:state('complete_exported_review_required',archive=str(archive));return
            if stopped():state('stopped_after_export',archive=str(archive));return
            if C.identity(cfg)!=expected or active_actors():raise ValueError('runtime changed or another actor took ownership')
            state('resuming_unchanged_benchmark',archive=str(archive),cycle_seconds=a.cycle_seconds)
            result=subprocess.run([sys.executable,'-u','-m','alpha0.terminal_v8','--seconds',str(a.cycle_seconds)],cwd=C.ROOT)
            state('resumed_cycle_ended_review_required',returncode=result.returncode,archive=str(archive))
        except Exception as exc:
            state('review_required',reason=f'{type(exc).__name__}: {exc}')
            raise

if __name__=='__main__':main()
