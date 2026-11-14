"""Play an RL iteration's jobs.json (rl_schedule.py) in parallel; resumable. A fork of collect_teacher.py's runner.

rl_collect.py runs/rl/RUN/iter-NNNN/jobs.json [--workers K] [--slice LO:HI:N] [--deadline UNIX] [--min-done N] [--out DIR]

Each job plays package A vs package B (any two bot packages: java/bots, build/botsrc or 2026b bots, compiled on first
use as eval_student.py does) with --decisions on the job's recorded sides (none when record is '') and --trace N
(jobs.json trace; 0 or a missing key: no --trace, no trace.tsv; the result and traj_hash do not depend on it).
Games go to DIR/games/<job id>/ (DIR defaults to the jobs.json directory): result.json, trace.tsv, decisions-*.bin.gz.
  --slice LO:HI:N   only jobs whose index i has LO <= i % N < HI (box_split.sh)
  --deadline T      no job starts after unix time T (running games finish) ...
  --min-done N      ... unless fewer than N games are done or running (the loop's minimum games per iteration)
A job with result.json is done; a partial game directory is wiped and replayed. Failures go to DIR/failures.log.
"""
import argparse, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TZ', 'PDT+7')   # log in PDT (UTC-7, BC's permanent time; user preference)
time.tzset()
sys.path.insert(0, str(ROOT / 'scripts'))


def valid(res):
    return not (res.get('error') or not res.get('finished') or res['A'].get('failures') or res['B'].get('failures'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('jobs')
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--slice', help='LO:HI:N: only jobs with LO <= index %% N < HI')
    ap.add_argument('--deadline', type=float, default=0, help='unix time after which no job starts')
    ap.add_argument('--min-done', type=int, default=0, help='after the deadline, still start jobs while done + running < N')
    ap.add_argument('--out', default='', help='output dir (default: the jobs.json directory)')
    a = ap.parse_args()
    jp = Path(a.jobs)
    doc = json.loads(jp.read_text())
    out = Path(a.out) if a.out else jp.parent
    (out / 'games').mkdir(parents=True, exist_ok=True)
    jobs = doc['jobs']
    if a.slice:
        lo, hi, n = map(int, a.slice.split(':'))
        jobs = [j for i, j in enumerate(jobs) if lo <= i % n < hi]
    done_ids = {j['id'] for j in doc['jobs'] if (out / 'games' / j['id'] / 'result.json').exists()}
    todo = [j for j in jobs if j['id'] not in done_ids]
    print(f"{time.strftime('%H:%M:%S')} {len(doc['jobs'])} jobs, {len(jobs)} in slice, {len(todo)} to run", flush=True)
    if not todo:
        return

    from eval_student import resolve
    specs = {p: resolve(p, '') for p in sorted({j['a'] for j in todo} | {j['b'] for j in todo})}   # compile once
    flags = doc.get('java_flags', '-Xmx2g')
    trace = int(doc.get('trace') or 0)
    timeout = doc.get('game_timeout_seconds', 1200)
    lock = threading.Lock()
    state = dict(done=len(done_ids), running=0, skipped=0)

    def may_start():
        with lock:
            if a.deadline and time.time() > a.deadline and state['done'] + state['running'] >= a.min_done:
                state['skipped'] += 1
                return False
            state['running'] += 1
            return True

    def play(j):
        if not may_start():
            return j, None, 0.0
        g = out / 'games' / j['id']
        t0 = time.time()
        ok = False
        try:
            if g.exists():
                for f in g.iterdir():
                    f.unlink()   # incomplete previous attempt
            g.mkdir(parents=True, exist_ok=True)
            cmd = [str(ROOT / 'scripts/student_java.sh'), '--map', j['map'], '--maps-dir', str(ROOT / 'maps'),
                   '--seed', str(j['seed']), '--a', specs[j['a']], '--b', specs[j['b']], '--out', str(g)]
            if trace > 0:
                cmd += ['--trace', str(trace)]
            if j.get('record'):
                cmd += ['--decisions', j['record']]
            err = g / 'stderr.log'
            try:
                with open(err, 'w') as fe:
                    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=fe, timeout=timeout,
                                       env={**os.environ, 'JAVA_FLAGS': flags})
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = 'timeout'
            r = g / 'result.json'
            ok = r.exists() and valid(json.loads(r.read_text()))
            if ok:
                err.unlink(missing_ok=True)   # DEATH lines from --trace; not needed
            else:
                tail = err.read_text(errors='replace')[-2000:] if err.exists() else ''
                with lock:
                    (out / 'failures.log').open('a').write(f"{time.strftime('%H:%M:%S')} {j['id']} rc={rc} {tail}\n")
        finally:
            with lock:
                state['running'] -= 1
                state['done'] += (g / 'result.json').exists()
        return j, ok, time.time() - t0

    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(play, j) for j in todo]):
            j, ok, secs = fut.result()
            if ok is None:
                continue
            print(f"{time.strftime('%H:%M:%S')} {j['id']} {j['map']} {j['a']} vs {j['b']} rec={j.get('record')} ok={ok} "
                  f"{secs:.0f}s  [{state['done']}/{len(doc['jobs'])}]", flush=True)
            with lock:
                st = dict(done=state['done'], scheduled=len(doc['jobs']), updated=time.time())
            tmp = out / f'status.json.{os.getpid()}'
            tmp.write_text(json.dumps(st)); os.replace(tmp, out / 'status.json')
    if state['skipped']:
        print(f"{time.strftime('%H:%M:%S')} deadline: {state['skipped']} jobs not started", flush=True)


if __name__ == '__main__':
    main()
