"""Reviewed benchmark-only API compatibility adapter; never retries a game."""
from pathlib import Path
from alpha0 import expanded_v11_run as S

REVIEW=S.P/'reviews/benchmark-adapter-recovery-20261111.json'

def runtime():
    source=Path(S.__file__).read_text()
    replacements={
        'attempt=1,max_attempts=1':'attempt=1,max_attempts=2',
        "'alpha0.expanded_v11_run','--worker'":"'alpha0.expanded_v11_benchmark_recovery','--worker'",
        "R/'jobs'":"R/'benchmark-recovery/jobs'",
        "R/'logs'":"R/'benchmark-recovery/logs'",
        "R/'tasks'":"R/'benchmark-recovery/tasks'",
        "R/'status.json'":"R/'benchmark-recovery/status.json'",
        "P/'jobs'":"P/'benchmark-recovery/jobs'",
    }
    for old,new in replacements.items():
        if old not in source:raise ValueError('adapter source contract changed')
        source=source.replace(old,new)
    ns={'__name__':'v11_reviewed_benchmark_runtime'}
    exec(compile(source,S.__file__,'exec'),ns)
    ns['execute']=execute
    return ns

def verify():
    review=S.read(REVIEW)
    if S.identity()!=S.read(S.P/'identity.json'):raise ValueError('frozen identity changed')
    if S.C.sha256(Path(__file__))!=review['adapter_sha256']:raise ValueError('adapter changed')
    for path,digest in review['preserved_files'].items():
        if S.C.sha256(S.C.ROOT/path)!=digest:raise ValueError('review evidence changed')
    if S.C.sha256(S.P/'benchmarks/specs.json')!=review['specs_sha256']:raise ValueError('benchmark schedule changed')
    return review

def execute(expected,deadline):
    verify()
    # Every prior stage must retain its completed immutable job proof.
    jobs=[S.read(p) for p in (S.R/'jobs').glob('*.json')]
    counts={kind:0 for kind in ('selection','comparator','fit','validation')}
    for job in jobs:
        if job['kind'] not in counts:continue
        marker=S.read(S.P/'jobs'/f"{job['id']}.json")
        if marker['status']!='ok' or marker['output_hashes']!=S.job_outputs(job):raise ValueError('prior stage proof changed')
        counts[job['kind']]+=1
    if counts!={'selection':64,'comparator':64,'fit':1,'validation':32}:raise ValueError('incomplete prior stages')
    frozen=S.read(S.P/'validation-actions.json');cfg=expected['config']
    specs=S.build_benchmark_specs({a:S.C.ROOT/v['path'] for a,v in frozen['checkpoints'].items()},cfg,expected)
    if len(specs)!=768 or S.read(S.P/'benchmarks/specs.json')['specs']!=specs:raise ValueError('frozen schedule mismatch')
    ns=runtime()
    ns['run_jobs']('benchmark',specs,expected,deadline)
    attempts=[S.read(p) for p in (S.P/'benchmarks/cells').glob('*/attempt-[0-9][0-9][0-9].json')]
    result=S.summarize_benchmarks(specs,attempts,expected,expected_panel=S.read(S.C.ROOT/cfg['benchmark_manifest']),expected_checkpoints={a:v['sha256'] for a,v in frozen['checkpoints'].items()})
    S.C.immutable_json(S.P/'benchmark-summary.json',result)

if __name__=='__main__':
    verify()
    runtime()['main']()
