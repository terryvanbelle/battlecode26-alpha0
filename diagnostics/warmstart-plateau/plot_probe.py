"""Plot only measured, completed own-policy counterfactual continuations."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
summary=json.loads((HERE/'counterfactual-v1/summary.json').read_text())
matrix=[];labels=[]
for root in summary['roots']:
    outcomes=defaultdict(list)
    for row in root['continuations']:
        if row['status']=='ok':outcomes[row['first_action']].append((row['terminal_value_root']+1)/2)
    matrix.append([sum(outcomes[action]) for action in (root['greedy_action'],37,24)])
    role='king' if root['actor']['type']=='RAT_KING' else 'baby'
    labels.append(f"{root['map']} · {role} · round {root['round']} (greedy {root['greedy_action']})")
values=np.asarray(matrix)
fig,ax=plt.subplots(figsize=(10,6.2),layout='constrained')
ax.imshow(values,cmap='RdYlGn',vmin=0,vmax=2,aspect='auto')
ax.set_yticks(range(len(labels)),labels)
ax.set_xticks(range(3),['Frozen cp2 greedy action','Powered bite (37)','Retreat (24)'])
for y in range(len(labels)):
    for x in range(3):ax.text(x,y,f'{values[y,x]:g}/2',ha='center',va='center',fontsize=13,fontweight='bold')
ax.set_title('Actual terminal continuation wins\nEight own-self-play roots; paired policy RNG seeds',fontsize=15,pad=15)
ax.set_xlabel('One forced first action, then the same frozen cp2 policy for both sides',labelpad=13)
fig.suptitle('All 48 searches chose the greedy action; all 48 continuations completed.\nNo SGD or benchmark data. Selected roots are not an independent strength test.',
             x=.5,y=-.01,fontsize=10)
fig.savefig(HERE/'counterfactual.png',dpi=170,bbox_inches='tight')
