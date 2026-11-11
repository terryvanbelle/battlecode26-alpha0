"""Bounded completion audit/export monitor; never starts games or training."""
import fcntl,os,re,subprocess,sys,time
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import expanded_v11_run as S
from scripts.audit_expanded_v11_benchmarks import audit
STATUS=S.R/'benchmark-deadline/finalizer-status.json'
def status(phase,**kw):S.C.atomic_json(STATUS,dict(phase=phase,pid=os.getpid(),updated_at=S.C.now(),**kw))
def call(args):subprocess.run(args,cwd=S.C.ROOT,check=True)
def main():
 with (S.R/'benchmark-deadline/finalizer.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if STATUS.exists():raise RuntimeError('monitor already attempted; review required')
  deadline=time.monotonic()+10800
  try:
   status('waiting')
   while True:
    S.check(deadline);st=S.read(S.R/'benchmark-deadline/status.json')
    if st['phase']=='complete_review_required':break
    if st['phase']=='review_required':raise RuntimeError('recovery stopped: '+str(st))
    time.sleep(5)
   with ExitStack() as stack:
    for p in S.LOCKS:
     h=stack.enter_context(p.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result=audit(require_complete=True)
    output=S.P/'reviews/completed-benchmark-audit-20261111.json';S.C.immutable_json(output,result)
   status('exporting')
   call([sys.executable,'scripts/publish_v11_development_views.py'])
   call([sys.executable,'scripts/report_development.py','progress/development-v4/v11-terminal_head','--manifest','configs/benchmarks-development-v4.json'])
   call([sys.executable,'scripts/report_expanded_v11.py'])
   counts=result['counts'];fmt=lambda a:f"{counts[a]['wins']}W / {counts[a]['losses']}L / {counts[a]['draws']}D"
   decision=result['comparisons']['terminal_head']['versus_fresh_cp2']['passes_development_improvement_rule']
   headline=f"V11 benchmarks complete: terminal {fmt('terminal_head')}; fresh cp2 {fmt('cp2')}; PUCT {fmt('puct_head')}, all 256/256 valid. Terminal development rule {'passed; fresh confirmation still required' if decision else 'failed'}. No promotion."
   readme=S.C.ROOT/'README.md';text=readme.read_text();text=re.sub(r'^\| \[V11\].*$',lambda _: '| [V11](progress/expanded-terminal-v11/README.md) | Complete, independently audited benchmark panels | '+headline+' |',text,flags=re.M);readme.write_text(text)
   page=S.P/'README.md';text=page.read_text();page.write_text(text.split('\n',1)[0]+'\n\n**'+headline+'** [Complete audit](reviews/completed-benchmark-audit-20261111.json). Earlier timestamped snapshots below are historical.\n'+text.split('\n',1)[1])

   call([sys.executable,'scripts/export_expanded_v11.py'])
   pointer=S.read(S.P/'recovery.json');manifest=S.read(S.C.ROOT/pointer['manifest'])
   if not manifest['isolated_restore_verified']:raise ValueError('restore not verified')
   for part in pointer['parts']:
    if S.C.sha256(S.C.ROOT/part['path'])!=part['sha256']:raise ValueError('archive hash mismatch')
   paths=[str(output.relative_to(S.C.ROOT)),'progress/expanded-terminal-v11/benchmark-summary.json','progress/expanded-terminal-v11/README.md','progress/expanded-terminal-v11/recovery.json','progress/expanded-terminal-v11/derived-progress.json','progress/expanded-terminal-v11/coverage.png','progress/development-v4/v11-cp2','progress/development-v4/v11-terminal_head','progress/development-v4/v11-puct_head','progress/overview','README.md',pointer['manifest'],*[p['path'] for p in pointer['parts']]]
   call(['git','add','--',*paths]);call(['git','commit','-m','Publish complete V11 benchmark audit and verified recovery']);call(['git','push'])
   status('published_review_required',audit=str(output.relative_to(S.C.ROOT)),manifest=pointer['manifest'],note='Archive retains historical original pipeline status and partial flag; complete benchmark audit is authoritative. No next experiment launched.')
  except BaseException as e:status('review_required',reason=str(e));raise
if __name__=='__main__':main()
