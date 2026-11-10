"""Recover only independently reviewed SIGTERM cells; retain original journals."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
from contextlib import ExitStack
import fcntl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import terminal_v8_common as C
from alpha0.terminal_v8 import active_actors
from alpha0.terminal_v8_report import benchmark_worker, benchmark_error_record, valid_benchmark_result
MANIFEST=C.PROGRESS/'reviews/deadline-recovery-20261110.json'

def approved(row):
    return (row.get('attempt')==1 and row.get('status')=='error' and
            row.get('error_kind')=='evaluation_error' and
            row.get('evaluation',{}).get('error')=='SIGTERM' and
            row['evaluation'].get('score') is None and row['evaluation'].get('rounds') is None)

def main():
    m=json.loads(MANIFEST.read_text())
    if len(sys.argv)>1:
        cell=sys.argv[1];assert cell in m['cells']
        spec=json.loads((C.PROGRESS/'benchmarks/cells'/cell/'request.json').read_text())
        result=benchmark_worker(spec,2)
        C.immutable_json(C.PROGRESS/'benchmarks/cells'/cell/'attempt-002.json',result)
        return
    with ExitStack() as stack:
        for path in C.LOCKS:
            h=stack.enter_context(path.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not active_actors()
        expected=json.loads((C.PROGRESS/'identity.json').read_text())
        assert C.identity(C.load_config())==expected
        for cell,proof in m['cells'].items():
            d=C.PROGRESS/'benchmarks/cells'/cell
            first=d/'attempt-001.json'
            assert C.sha256(first)==proof['original_sha256'] and approved(json.loads(first.read_text()))
            spec=json.loads((d/'request.json').read_text())
            out=d/'attempt-002.json'
            if out.exists():
                assert valid_benchmark_result(spec,json.loads(out.read_text()))
                continue
            if any(p.exists() for p in C.STOPS):raise RuntimeError('STOP')
            limit=C.load_config()['evaluation_timeout_seconds']+30
            log=stack.enter_context((d/'recovery-attempt-002.log').open('x'))
            proc=subprocess.Popen([sys.executable,'-u',__file__,cell],cwd=C.ROOT,start_new_session=True,stdout=log,stderr=log)
            deadline=time.monotonic()+limit
            try:
                while proc.poll() is None:
                    if time.monotonic()>deadline or any(p.exists() for p in C.STOPS):raise RuntimeError('recovery deadline or STOP')
                    time.sleep(.5)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                if not out.exists():C.immutable_json(out,benchmark_error_record(spec,2,'worker_process_error','reviewed recovery worker ended without result'))
            assert valid_benchmark_result(spec,json.loads(out.read_text())), 'second attempt failed; review required'
        assert C.identity(C.load_config())==expected
        print('Both reviewed deadline cells recovered; original journals unchanged',flush=True)
if __name__=='__main__':main()
