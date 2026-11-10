"""Run the new fixed broader-map development panel. No promotion or training."""
import sys,json,time,fcntl,argparse,threading
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import evaluate as E
from alpha0.constraint_v7 import active_actors
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--progress-dir',required=True,type=Path);p.add_argument('--hours',type=float,default=3);a=p.parse_args()
assert 0<a.hours<=3
with ExitStack() as stack:
 stops=[]
 for name in ['campaign','warmstart-v3','terminal-target-head-v5','terminal-target-head-v6','cached-head-constraint-v7','terminal-target-head-v8','development-v3']:
  d=E.ROOT/'runs'/name;d.mkdir(exist_ok=True,parents=True)
  stops.append(d/'STOP')
  assert not (d/'STOP').exists(),f'STOP: {name}'
  h=stack.enter_context((d/('campaign.lock' if name=='campaign' else 'experiment.lock')).open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not active_actors(),'active actors'
 cfg=json.loads((E.ROOT/'configs/improvement-v3.json').read_text())
 for m in cfg['maps']:assert E.file_hash(E.ROOT/'maps'/f"{m['name']}.map26")==m['sha256']
 def watch():
  while True:
   if any(p.exists() for p in stops):
    (E.ROOT/'runs/development-v3/STOP').touch();return
   time.sleep(.5)
 threading.Thread(target=watch,daemon=True).start()
 report=E.evaluate_checkpoint(a.checkpoint,suite='standard',manifest_path=E.ROOT/cfg['manifest'],progress_dir=a.progress_dir,jobs=2,deadline=time.monotonic()+3600*a.hours,stop_file=E.ROOT/'runs/development-v3/STOP')
 E.atomic_json(a.progress_dir/'summary.json',report)
 from scripts.report_development import generate
 generate(a.progress_dir)
 print(json.dumps(report['opponents'],indent=2))
