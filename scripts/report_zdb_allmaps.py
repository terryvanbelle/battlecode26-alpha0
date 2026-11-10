from pathlib import Path
import json,collections,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parents[1]/'progress/zdb-allmaps-cp2'
s=json.loads((p/'summary.json').read_text())
assert s['complete'] and s['completed']==296 and s['unresolved_errors']==0
rows=[json.loads(l) for l in (p/'evaluations.jsonl').read_text().splitlines()]
assert len(rows)==296 and all(r['status']=='ok' for r in rows)
by=collections.defaultdict(collections.Counter);sides=collections.defaultdict(collections.Counter)
for r in rows:
 k={1:'wins',0:'losses',.5:'draws'}[r['score']];by[r['map']][k]+=1;sides[r['side']][k]+=1
assert len(by)==74 and all(sum(v.values())==4 for v in by.values())
with (p/'per-map.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['map','wins','losses','draws'])
 for name,c in sorted(by.items()):w.writerow([name,c['wins'],c['losses'],c['draws']])
names=sorted(by);fig,ax=plt.subplots(figsize=(10,19));left=[0]*74
for key,color in [('wins','#1b9e77'),('draws','#cccccc'),('losses','#d95f02')]:
 vals=[by[n][key] for n in names];ax.barh(names,vals,left=left,label=key,color=color);left=[a+b for a,b in zip(left,vals)]
ax.invert_yaxis();ax.set_xlim(0,4);ax.set_xticks(range(5));ax.legend();ax.set_title('Frozen warmstart cp2 versus vibe26b_zdb · all 74 maps');ax.set_xlabel('Games (two seeds × both seats), from learned player perspective');fig.tight_layout();fig.savefig(p/'per-map.png',dpi=120);plt.close(fig)
c=s['opponents']['vibe26b_zdb']
text=f'''# All-map head-to-head: cp2 versus vibe26b_zdb

From the learned player's perspective: **{c['wins']} wins, {c['losses']} losses, {c['draws']} draws**, in all 296 scheduled games. No unresolved errors.

74 checksummed maps × seeds 0 and 2026 × both seats. Greedy warmstart checkpoint 2, SHA `{s['checkpoint_sha256']}`. Opponent is frozen `vibe26b_zdb`, source commit `fc044c24c37d89d2581562e23439a2571bb2b300`; this does not include newer source-project changes. Local relaxed-resource engine settings apply. This one-opponent diagnostic is not the full victory gate, and none of its trajectories enter training.

| Learned seat | Wins | Losses | Draws |
|---|---:|---:|---:|
'''
for side in ['A','B']:
 c=sides[side];text+=f"| {side} | {c['wins']} | {c['losses']} | {c['draws']} |\n"
text+='\n[Per-map CSV](per-map.csv) · [Complete summary](summary.json) · [Raw game results](evaluations.jsonl)\n\n![Per-map results](per-map.png)\n'
(p/'README.md').write_text(text)
print(text.split('![Per-map')[0])
