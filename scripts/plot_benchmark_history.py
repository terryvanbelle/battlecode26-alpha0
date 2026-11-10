"""Build the README overview from complete, fixed-panel smoke measurements."""
from pathlib import Path
import json,collections,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'progress/overview';OUT.mkdir(exist_ok=True)
bots=[b['id'] for b in json.loads((ROOT/'configs/benchmarks.json').read_text())['bots']]
points=[];inputs={}
def add_series(path,family,prefix):
 inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 rows=[json.loads(l) for l in path.read_text().splitlines()]
 groups=collections.defaultdict(dict)
 for r in rows:
  if r.get('suite')!='smoke' or r.get('map')!='tiny' or r.get('seed')!=0 or r['opponent'] not in bots:continue
  key=(r['iteration'],r['checkpoint_sha256'],r.get('evaluation_config_sha256','legacy'))
  groups[key][(r['opponent'],r['side'])]=r
 for (iteration,sha,config),cells in sorted(groups.items()):
  if set(cells)!={(b,s) for b in bots for s in ['A','B']} or any(r['status']!='ok' for r in cells.values()):continue
  add_point(f'{prefix}\n{iteration}',family,list(cells.values()),sha,config)
def add_point(label,family,rows,sha,config):
 assert len(rows)==24
 values={b:sum(r['score']==1 for r in rows if r['opponent']==b)/2*100 for b in bots}
 points.append(dict(label=label,family=family,checkpoint_sha256=sha,evaluation_config_sha256=config,games=24,wins=sum(r['score']==1 for r in rows),draws=sum(r['score']==.5 for r in rows),win_percent=values))
for path,family,prefix in [('progress/evaluations.jsonl','V1: original controller','V1 cp'),('progress/campaign-v2/evaluations.jsonl','V2: neutral RL control','V2 cp'),('progress/bootstrap-v3/evaluations.jsonl','V3: own-teacher initialization','Mixed prior'),('progress/bootstrap-v3-winners/evaluations.jsonl','V3: own-teacher initialization','Winner prior'),('progress/warmstart-v3/evaluations.jsonl','V3: MCTS continuation','MCTS cp')]:
 add_series(ROOT/path,family,prefix)
groups=collections.defaultdict(list)
for path in sorted((ROOT/'progress/terminal-target-head-v6/benchmarks/cells').glob('*/attempt-*.json')):
 r=json.loads(path.read_text());inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 if r['status']=='ok':groups[r['arm']].append(r['evaluation'])
for arm,label in [('cp2','V6 fresh\ncp2'),('puct_head','V6 PUCT\nhead'),('terminal_head','V6 terminal\nhead')]:
 rows=list({r['evaluation_cell_id']:r for r in groups[arm]}.values());assert len(rows)==24 and all(r['status']=='ok' and r['map']=='tiny' and r['seed']==0 for r in rows)
 add_point(label,'V6: matched head experiment',rows,rows[0]['checkpoint_sha256'],rows[0]['evaluation_config_sha256'])
x=np.arange(len(points));mat=np.array([[p['win_percent'][b] for p in points] for b in bots]);total=np.array([p['wins']/24*100 for p in points])
fig,(ax,heat)=plt.subplots(2,1,figsize=(17,10),gridspec_kw={'height_ratios':[1,2.5]},sharex=True)
families=list(dict.fromkeys(p['family'] for p in points));colors=plt.get_cmap('tab10')
for k,f in enumerate(families):
 ids=[i for i,p in enumerate(points) if p['family']==f];ax.plot(ids,total[ids],'-o',label=f,color=colors(k),linewidth=2)
 for i in ids:ax.annotate(f"{points[i]['wins']}/24",(i,total[i]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
ax.set_ylim(-5,110);ax.set_ylabel('Overall win rate (%)');ax.grid(axis='y',alpha=.25);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.45),ncol=3,fontsize=9)
heat.imshow(mat,aspect='auto',vmin=0,vmax=100,cmap='RdYlGn');heat.set_yticks(range(12),bots);heat.set_xticks(x,[p['label'] for p in points],fontsize=9)
for j in range(12):
 for i in x:heat.text(i,j,f'{mat[j,i]:.0f}%',ha='center',va='center',fontsize=8,color='black' if mat[j,i]==50 else 'white')
for i in range(1,len(points)):
 if points[i]['family']!=points[i-1]['family']:
  for a in (ax,heat):a.axvline(i-.5,color='#333333',linestyle='--',alpha=.5)
heat.set_xlabel('Recorded checkpoint / experiment order (unequal elapsed time; branches are not one continuous training run)')
fig.suptitle('Benchmark win-rate history · all 12 frozen opponents',fontsize=18,y=.995)
fig.text(.5,.015,'Smoke only: tiny, seed 0, both seats. Each cell = 2 games; each column = 24. Wins / games (draws are not wins).\nController/runtime and initialization changes are separate experiments. No all-map victory is established; partial all-map zdb results are excluded.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.065,1,.96));fig.savefig(OUT/'benchmark-history.png',dpi=140);plt.close(fig)
(OUT/'benchmark-history.json').write_text(json.dumps({'scope':'complete fixed-panel tiny/seed0/both-seat smoke evaluations','points':points,'source_sha256':inputs},indent=2)+'\n')
(OUT/'README.md').write_text('''# Benchmark history

![Win-rate history](benchmark-history.png)

Each column is a complete 24-game smoke panel: 12 opponents, `tiny`, seed 0, both seats. The upper plot counts wins divided by games; the lower plot shows every opponent separately. Draws are not wins. Missing or partial panels are excluded rather than filled with zero. Dashed boundaries separate experiments; lines connect checkpoints only within a research branch. The x-axis is checkpoint order, not elapsed time.

The original V1 controller, corrected V2 controller, V3 own-teacher initialization, MCTS continuation and V6 matched head fits are separate experiments. Their differences must not all be attributed to reinforcement learning. Runtime/evaluation configuration identities and source checksums are retained in [the chart data](benchmark-history.json). Two games per opponent provide little statistical precision. Repeated deterministic smoke games are regression checks, not independent holdout replications.

The best measured smoke score remains 20/24. V7 is a training-only diagnostic and has no benchmark point; V8 has not run. The separate, user-stopped all-map zdb test is not pooled into this tiny-only history.

Regenerate with `.venv/bin/python scripts/plot_benchmark_history.py`.
''')
print([(p['label'],p['wins']) for p in points])
