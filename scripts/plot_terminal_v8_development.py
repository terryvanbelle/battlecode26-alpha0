"""Full-development V8 coverage, W/L/D and transfer evidence."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/battlecode-mpl')


def generate(summary, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    sampling=summary['sampling']
    if sampling['source_temperature']!=1 or sampling['continuation_temperature']!=.35 or sampling['earlier_trajectory_data_reused']:
        raise ValueError('wrong fresh transfer sampling')
    for p in summary['benchmarks']['panels'].values():
        cells=p['cells']
        if p['completed']!=len(cells) or p['complete']!=(len(cells)==256 and p['unresolved_errors']==0):
            raise ValueError('inconsistent panel coverage')
        if p['wins']!=sum(r['score']==1 for r in cells) or p['losses']!=sum(r['score']==0 for r in cells) or p['draws']!=sum(r['score']==.5 for r in cells):
            raise ValueError('inconsistent W/L/D')
    if summary['decision']['criteria_pass'] and not (summary['benchmarks']['complete'] and summary['benchmarks']['development_improvement'] and summary['validation']['complete']):
        raise ValueError('incomplete qualifying evidence')
    root=Path(__file__).resolve().parents[1]
    panel=json.loads((root/'configs/benchmarks-development-v4.json').read_text())
    bots=sorted(panel['bots'],key=lambda b:b['historical_rating']['elo'])
    maps=panel['evaluation_protocol']['standard']['maps']
    arms=('cp2','terminal_head','puct_head');bench=summary['benchmarks']
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(14,17))
    for ax,arm in zip(axes,arms):
        p=bench['panels'][arm];a=np.full((8,8),np.nan)
        for i,bot in enumerate(bots):
            for j,m in enumerate(maps):
                rows=[r for r in p['cells'] if r['opponent']==bot['id'] and r['map']==m]
                w,l,d=[sum(r['score']==v for r in rows) for v in (1,0,.5)]
                if len(rows)==4:a[i,j]=100*w/4
                ax.text(j,i,f'{w}/{l}/{d}\n{len(rows)}/4',ha='center',va='center',fontsize=8)
        ax.imshow(a,vmin=0,vmax=100,cmap='RdYlGn',aspect='auto')
        ax.set_xticks(range(8),maps,rotation=15);ax.set_yticks(range(8),[b['id'] for b in bots],fontsize=9)
        ax.set_title(f"{arm}: {p['completed']}/256 valid · {p['wins']}W/{p['losses']}L/{p['draws']}D · checkpoint {(p['checkpoint_sha256'] or 'pending')[:12]}")
    fig.suptitle('V8 broader-map transfer · each cell W/L/D and coverage; incomplete cells uncolored')
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(output_dir/'comparison.png',dpi=130);plt.close(fig)
    sel=summary['selection'];val=summary['validation']
    stages=[('Training roots',sel['frozen_roots'],32),('Anchors',summary['anchors']['completed'],512),
            ('Selection cells',sel['completed_cells'],sel['expected_cells']),
            ('Validation roots',val['completed_roots'],16),('Benchmark games',bench['completed'],768)]
    fig,ax=plt.subplots(figsize=(10,4))
    ax.barh([s[0] for s in stages],[s[1]/s[2] if s[2] else 0 for s in stages])
    for i,(_,n,total) in enumerate(stages):ax.text(.02,i,f'{n}/{total}',va='center')
    ax.set_xlim(0,1.1);ax.set_xlabel('Completed fraction');fig.tight_layout();fig.savefig(output_dir/'coverage.png',dpi=140);plt.close(fig)
    text=['# V8 development transfer','',f"Valid benchmark games: {bench['completed']}/768. No automatic promotion.",'','![Opponent-map W/L/D](comparison.png)','','![Coverage](coverage.png)','',
          'The terminal head is the preregistered candidate; the PUCT head is a matched control. Complete-panel improvement uses ≥5 percentage points score gain, positive paired-map bootstrap lower bound and aggregate zdb nonregression. Fresh-seed confirmation is required for promotion. First-action validation is a separately reported mechanism diagnostic, not a substitute for benchmark improvement.','',
          '## Comparisons','', '```json',json.dumps(bench['comparisons'],indent=2),'```','',
          '## Independent first-action validation','', '```json',json.dumps({k:val.get(k) for k in ('complete','mean_delta_vs_puct','mean_delta_vs_cp2','criteria_pass')},indent=2),'```','',
          '## Research decision','', '```json',json.dumps(summary['decision'],indent=2),'```','']
    (output_dir/'README.md').write_text('\n'.join(text))
    return summary
