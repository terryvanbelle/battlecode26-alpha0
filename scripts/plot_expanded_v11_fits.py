"""Read-only plots of complete V11 saved training history; not strength evidence."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    run=ROOT/'runs/expanded-terminal-v11/fit';out=ROOT/'progress/expanded-terminal-v11'
    audit=json.loads((out/'reviews/fit-audit-20261111.json').read_text());assert audit['verified']
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for arm in ('terminal_head','puct_head'):
        path=run/f'{arm}.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['arms'][arm]['checkpoint_sha256']
        payload=torch.load(path,map_location='cpu',weights_only=True);history=payload['fit_history']
        assert [r['step'] for r in history]==list(range(1,4097))
        for ax,key in zip(axes,('loss','target_ce','anchor_kl')):
            values=np.array([r[key] for r in history]);assert np.isfinite(values).all()
            ax.plot(np.arange(128,4097,128),values.reshape(32,128).mean(1),label=arm)
            ax.set_title(key);ax.set_xlabel('Optimizer update (128-update block means)');ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('V11 training diagnostics — not benchmark win rate');fig.tight_layout()
    fig.savefig(out/'fit-curves.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
