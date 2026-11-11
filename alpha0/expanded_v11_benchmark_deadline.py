"""Explicit recovery of the reviewed V11 benchmark deadline, no automatic retry."""
import ctypes,os,signal,time
from pathlib import Path
from alpha0 import expanded_v11_run as S
from alpha0 import expanded_v11_benchmark_recovery as B
from alpha0.terminal_v8_report import benchmark_worker
from alpha0 import evaluate as E
REVIEW=S.P/'reviews/benchmark-deadline-recovery-20261111.json'

def verify():
    B.verify();review=S.read(REVIEW)
    if S.C.sha256(Path(__file__))!=review['runner_sha256']:raise ValueError('recovery runner changed')
    for name,digest in review['preserved_files'].items():
        if S.C.sha256(S.C.ROOT/name)!=digest:raise ValueError('preserved deadline evidence changed')
    return review

def attempt_number(spec,review):
    if spec['cell_id'] not in review['remaining_cell_ids']:raise ValueError('cell outside reviewed remaining schedule')
    directory=S.P/'benchmarks/cells'/spec['cell_id'];files=sorted(directory.glob('attempt-*.json'))
    old=review['interrupted'].get(spec['cell_id'])
    if not old:
        if files:raise ValueError('new cell already attempted; review required')
        return 1
    if len(files)!=1 or files[0].name!='attempt-001.json' or S.C.sha256(files[0])!=old:raise ValueError('interrupted record changed or retry already attempted')
    row=S.read(files[0]);ev=row['evaluation']
    if row['status']!='error' or ev.get('error')!='SIGTERM' or ev.get('score') is not None or ev.get('rounds') is not None:raise ValueError('not an unscored reviewed interruption')
    return 2

def worker(job,expected,deadline,parent):
    ctypes.CDLL(None).prctl(1,signal.SIGKILL)
    if os.getppid()!=parent:raise RuntimeError('parent disappeared')
    S.torch.set_num_threads(1)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(S.C.Interrupted('SIGTERM')))
    review=verify();assert S.identity()==expected
    spec=job['payload'];directory=S.P/'benchmarks/cells'/spec['cell_id']
    S.check(deadline)
    with E.locked(directory/'.lock'):
        attempt=attempt_number(spec,review)
        S.C.immutable_json(directory/'request.json',spec)
        task=S.R/'benchmark-deadline/tasks'/f"{job['id']}.json"
        S.C.set_task(task,'benchmark',min(expected['config']['evaluation_timeout_seconds']+30,max(.01,deadline-time.monotonic())))
        result=benchmark_worker(spec,attempt)
        S.C.immutable_json(directory/f'attempt-{attempt:03d}.json',result)
        S.C.clear_task(task)
        if not S.valid_benchmark_result(spec,result):raise RuntimeError('recovery failed; explicit review required')
    assert S.identity()==expected

def runtime():
    source=Path(S.__file__).read_text()
    for old,new in {"'alpha0.expanded_v11_run','--worker'":"'alpha0.expanded_v11_benchmark_deadline','--worker'","R/'jobs'":"R/'benchmark-deadline/jobs'","R/'logs'":"R/'benchmark-deadline/logs'","R/'tasks'":"R/'benchmark-deadline/tasks'","R/'status.json'":"R/'benchmark-deadline/status.json'","P/'jobs'":"P/'benchmark-deadline/jobs'"}.items():
        if old not in source:raise ValueError('source contract changed')
        source=source.replace(old,new)
    ns={'__name__':'v11_reviewed_benchmark_deadline'};exec(compile(source,S.__file__,'exec'),ns)
    ns.update(worker=worker,execute=execute);return ns

def execute(expected,deadline):
    review=verify();saved=S.read(S.P/'benchmarks/specs.json')['specs']
    pending=[s for s in saved if s['cell_id'] in review['remaining_cell_ids']]
    if len(pending)!=22 or any(s['arm']!='puct_head' for s in pending):raise ValueError('reviewed denominator changed')
    runtime()['run_jobs']('benchmark',pending,expected,deadline)
    frozen=S.read(S.P/'validation-actions.json');attempts=[S.read(p) for p in (S.P/'benchmarks/cells').glob('*/attempt-*.json')]
    result=S.summarize_benchmarks(saved,attempts,expected,expected_panel=S.read(S.C.ROOT/expected['config']['benchmark_manifest']),expected_checkpoints={a:v['sha256'] for a,v in frozen['checkpoints'].items()})
    S.C.immutable_json(S.P/'benchmark-summary.json',result)

if __name__=='__main__':
    verify();runtime()['main']()
