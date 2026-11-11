"""Bounded real-engine reconstruction of all frozen roots, no search or continuations."""
import fcntl,json,time
from contextlib import ExitStack
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts import collect_validation_v11 as V
from alpha0.expanded_v11_support import load_root_backend,frozen_plans


def main():
 torch.set_num_threads(1);out=V.C.ROOT/'diagnostics/expanded-v11-planning/root-reconstruction.json'
 if out.exists():raise RuntimeError('completed verification already exists; do not rerun')
 with ExitStack() as stack:
  for path in V.LOCKS:
   h=stack.enter_context(path.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if V.active_actors():raise RuntimeError('actors active')
  cfg=V.read(V.CFG);cfg.update(selection_seeds=[223003,223013],validation_seeds=[223021,223043,223067,223091],selection_cell_ceiling=3200)
  paths={'training':'progress/general-roots-v11/training-input-plan.json','validation':'progress/validation-v11/validation-input-plan.json','exclusions':'configs/validation-v11-exclusions.json'}
  identity={'config':cfg,'input_plans':{k:{'path':v,'sha256':V.C.sha256(V.C.ROOT/v)} for k,v in paths.items()},'collector_sha256':V.C.sha256(V.C.ROOT/'alpha0/terminal_v8_collect.py'),'adapter_sha256':V.C.sha256(V.C.ROOT/'alpha0/expanded_v11_support.py')}
  training,validation,_=frozen_plans(identity);deadline=time.monotonic()+1800
  run=V.C.ROOT/'runs/v11-root-reconstruction';run.mkdir(exist_ok=True)
  backend=load_root_backend(run,run/'progress',V.check,identity);results=[]
  for root in training['roots']+validation['roots']:
   V.check(deadline)
   worker=backend.RootWorker({**root,'worker_task_path':str(run/'current-task.json')},cfg,identity,deadline)
   try:
    state=worker.restore();results.append({'root_id':root['root_id'],'input_hash':root['input_hash'],'state_sha256':root['state_sha256'],'cache_fields_equal':True})
   finally:worker.close()
   V.C.atomic_json(run/'status.json',{'phase':'checking','verified':len(results),'expected':96})
  assert len(results)==96
  V.C.immutable_json(out,{'passed':True,'identity':identity,'roots':results,'verified':96,'search_simulations':0,'terminal_continuations':0,'sgd_updates':0})
  V.C.atomic_json(run/'status.json',{'phase':'complete','verified':96,'expected':96})
if __name__=='__main__':main()
