"""Bounded, resumable single-opponent crash qualification; never promotes a bot."""
import sys,json,fcntl,time,threading,argparse
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import evaluate as E
from alpha0.constraint_v7 import active_actors
parser=argparse.ArgumentParser()
parser.add_argument('--manifest',required=True,type=Path)
parser.add_argument('--progress-dir',required=True,type=Path)
a=parser.parse_args()
with ExitStack() as stack:
 for name in ['campaign','warmstart-v3','terminal-target-head-v5','terminal-target-head-v6','cached-head-constraint-v7','terminal-target-head-v8','development-v3']:
  d=E.ROOT/'runs'/name
  d.mkdir(parents=True,exist_ok=True)
  assert not (d/'STOP').exists()
  h=stack.enter_context((d/('campaign.lock' if name=='campaign' else 'experiment.lock')).open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def watch_stops():
  while True:
   names=['campaign','warmstart-v3','terminal-target-head-v5','terminal-target-head-v6','cached-head-constraint-v7','terminal-target-head-v8','development-v3']
   if any((E.ROOT/'runs'/n/'STOP').exists() for n in names):
    (E.ROOT/'runs/development-v3/STOP').touch();return
   time.sleep(.5)
 threading.Thread(target=watch_stops,daemon=True).start()
 assert not active_actors()
 p=a.progress_dir
 result=E.evaluate_checkpoint(E.ROOT/'runs/warmstart-v3/checkpoint-0002.pt',suite='standard',manifest_path=a.manifest,progress_dir=p,jobs=2,deadline=time.monotonic()+3600,stop_file=E.ROOT/'runs/development-v3/STOP')
 E.atomic_json(p/'summary.json',result)
