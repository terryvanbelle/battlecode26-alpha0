"""One bounded baseline-to-V8 handoff. No duplicate actors or automatic promotion."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import evaluate as E
from alpha0 import terminal_v8_common as C
from alpha0.terminal_v8 import active_actors
from scripts.compare_development import compare


def inspect_baseline(directory, config, manifest, expected_identity):
    path=directory/'summary.json'
    if not path.exists():return None
    summary=json.loads(path.read_text())
    if not summary.get('complete') or summary.get('unresolved_errors'):
        raise ValueError('baseline ended incomplete or invalid; review required')
    rows=E.read_jsonl(directory/'evaluations.jsonl')
    compare(rows,rows,config,manifest)
    for row in rows:
        spec=json.loads((directory/'evaluation_cells'/row['evaluation_cell_id']/'request.json').read_text())
        if not E.valid_result(spec,row):raise ValueError('invalid terminal baseline evidence')
        if row['identity']['evaluation_config']['runtime']!=expected_identity['evaluation_runtime']:
            raise ValueError('baseline runtime differs from queued experiment')
    return {'games':len(rows),'wins':sum(r['score']==1 for r in rows),
            'losses':sum(r['score']==0 for r in rows),'draws':sum(r['score']==.5 for r in rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-seconds',type=float,default=10800)
    p.add_argument('--cycle-seconds',type=float,default=10800)
    args=p.parse_args()
    if not 0<args.wait_seconds<=10800 or not 0<args.cycle_seconds<=10800:
        p.error('each wait/cycle bound must be positive and at most three hours')
    directory=C.ROOT/'runs/research-sequence';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'sequence.lock').open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=C.load_config();identity=C.identity(cfg)
        protocol=json.loads((C.ROOT/cfg['improvement_config']).read_text())
        panel=json.loads((C.ROOT/cfg['benchmark_manifest']).read_text())
        baseline=C.ROOT/'progress/development-v4/baseline'
        deadline=time.monotonic()+args.wait_seconds
        def status(phase,**details):
            value=dict(phase=phase,pid=os.getpid(),updated_at=C.now(),identity_sha256=C.fingerprint(identity),**details)
            C.atomic_json(directory/'status.json',value);print(json.dumps(value),flush=True)
        status('waiting_for_baseline')
        while time.monotonic()<deadline:
            if (directory/'STOP').exists() or any(p.exists() for p in C.STOPS):
                status('stopped');return
            try:
                evidence=inspect_baseline(baseline,protocol,panel,identity)
            except Exception as exc:
                status('review_required',reason=str(exc));return
            if evidence is not None:
                try:
                    with ExitStack() as stack:
                        for path in C.LOCKS:
                            h=stack.enter_context(path.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        if active_actors():
                            raise BlockingIOError('actors still active')
                        if C.identity(cfg)!=identity:raise ValueError('queued runtime changed')
                        if (C.RUN/'status.json').exists():
                            status('review_required',reason='V8 already has state; inspect before resuming');return
                        if evidence['wins']/evidence['games']>=.70:
                            status('tournament_transition_required',baseline=evidence);return
                except BlockingIOError:
                    time.sleep(5);continue
                except Exception as exc:
                    status('review_required',reason=str(exc));return
                # Runner reacquires all locks and repeats STOP/actor/identity checks.
                status('starting_v8',baseline=evidence,cycle_seconds=args.cycle_seconds)
                result=subprocess.run([sys.executable,'-u','-m','alpha0.terminal_v8','--seconds',str(args.cycle_seconds)],cwd=C.ROOT)
                status('cycle_ended_review_required',returncode=result.returncode,baseline=evidence)
                return
            time.sleep(5)
        status('wait_deadline_reached')

if __name__=='__main__':main()
