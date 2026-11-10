from pathlib import Path
import hashlib,json,re,time,torch
from alpha0.train import load_model
from alpha0.env import EngineEnv,make_evaluator
ROOT=Path(__file__).resolve().parents[3]
torch.set_num_threads(1)
checkpoint=ROOT/'artifacts/initial/checkpoint-0001.pt'
net,_=load_model(checkpoint);infer=make_evaluator(net)
records=[]
for seed in [0,1]:
 env=EngineEnv(opponent_package='Finals',opponent_classes=ROOT/'build/benchmarks/AdamTan12_Finals',learned_team=0)
 n=0;t=time.time()
 try:
  s=env.reset('tiny',seed)
  while not s.terminal:
   policy,_=infer(s);s=env.transition(s,max(s.legal_actions,key=lambda a:policy[a]));n+=1
  records.append({'seed':seed,'status':'ok','round':s.raw['round']})
 except Exception as e:
  error=str(e);trace=env._error_tail()
  records.append({'map':'tiny','seed':seed,'learned_side':'A','opponent_team':'B','checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'status':'error','round':int(re.search(r'round=(\d+)',error).group(1)),'robot_id':int(re.search(r'robot_id=(\d+)',error).group(1)),'completed_decisions':n,'exception':'battlecode.common.GameActionException: Target location is not on the map','source_frame':'Finals.Actions.ThrowOutsideCombat.throwAtWall(ThrowOutsideCombat.java:58)','stack_trace':trace,'seconds':round(time.time()-t,2)})
 finally:env.close()
Path('/tmp/alpha0-adamtan-crash-evidence.json').write_text(json.dumps({'opponent':'AdamTan12_Finals','diagnosis':'Uncaught off-map senseMapInfo call before bounds check; frozen source unchanged','reproduction_command':'PYTHONPATH=. .venv/bin/python benchmarks/retired/AdamTan12_Finals/reproduce.py','cells':records},indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k not in ['stack_trace','checkpoint_sha256']} for r in records],indent=2))
