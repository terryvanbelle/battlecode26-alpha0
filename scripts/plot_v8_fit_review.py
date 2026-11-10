"""Plot saved training histories; no fitting and no strength inference."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/battlecode-mpl')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def generate(out):
    torch.set_num_threads(1)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for arm,color in [('terminal_head','#19766a'),('puct_head','#b16b17')]:
        p=torch.load(ROOT/'runs/terminal-target-head-v8/fit'/f'{arm}.pt',map_location='cpu',weights_only=True)
        assert p['updates']==2048 and len(p['fit_history'])==2048
        for ax,key in zip(axes,['target_ce','anchor_kl']):
            vals=np.array([r[key] for r in p['fit_history']]);ax.plot(range(1,2049),vals,color=color,alpha=.13,linewidth=.6)
            ax.plot(range(64,2049),np.convolve(vals,np.ones(64)/64,mode='valid'),color=color,label=arm)
            ax.set_xlabel('Optimizer updates');ax.set_ylabel(key);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('V8 fitting histories · 64-update moving means (raw batches faint)\nTraining diagnostics only; independent validation advantage was zero')
    fig.tight_layout();out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=150);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();generate(a.output)
