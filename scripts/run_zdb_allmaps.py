"""Requested frozen-cp2 versus frozen-zdb all-map diagnostic; no gradients."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,threading,fcntl
from contextlib import ExitStack
from alpha0 import evaluate as E
from alpha0.constraint_v7 import active_actors
ROOT=E.ROOT
P=ROOT/'progress/zdb-allmaps-cp2'
D=ROOT/'runs/zdb-allmaps-cp2'
names=['campaign','warmstart-v3','terminal-target-head-v5','terminal-target-head-v6','cached-head-constraint-v7','terminal-target-head-v8','zdb-allmaps-cp2']
with ExitStack() as stack:
 for name in names:
  directory=ROOT/'runs'/name;directory.mkdir(parents=True,exist_ok=True)
  h=stack.enter_context((directory/('campaign.lock' if name=='campaign' else 'experiment.lock')).open('a+'))
  fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
 stops=[ROOT/'runs'/n/'STOP' for n in names]
 assert not any(p.exists() for p in stops),'STOP exists'
 assert not active_actors(),'active actor'
 P.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((ROOT/'configs/benchmarks.json').read_text())
 gate=json.loads((ROOT/'configs/victory_gate.json').read_text())
 maps=sorted(gate['frozen_maps'],key=lambda x:x['name'])
 assert len(maps)==74
 assert {p.stem for p in (ROOT/'maps').glob('*.map26')}=={m['name'] for m in maps}
 for m in maps:assert E.file_hash(ROOT/'maps'/f"{m['name']}.map26")==m['sha256']
 manifest['evaluation_protocol']['standard']={'maps':[m['name'] for m in maps],'seeds':[0,2026],'games_per_checkpoint':296,'interpretation':'Requested all-map zdb head-to-head diagnostic, not full-panel victory.'}
 target=P/'schedule-manifest.json'
 if target.exists():assert json.loads(target.read_text())==manifest
 else:E.atomic_json(target,manifest)
 done=threading.Event()
 def watch():
  while not done.wait(.5):
   if any(p.exists() for p in stops):
    (D/'STOP').touch();return
 threading.Thread(target=watch,daemon=True).start()
 try:
  result=E.evaluate_checkpoint(ROOT/'runs/warmstart-v3/checkpoint-0002.pt',suite='standard',progress_dir=P,jobs=2,opponents=['vibe26b_zdb'],manifest_path=target,stop_file=D/'STOP',deadline=time.monotonic()+10800)
  E.atomic_json(P/'summary.json',result)
  print(json.dumps(result['opponents']),flush=True)
 finally:done.set()
