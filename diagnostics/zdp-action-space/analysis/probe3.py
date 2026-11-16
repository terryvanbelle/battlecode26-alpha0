"""Read-only: how many PRE/POST argmaxes change when the untrained nRatT input (col 32) is zeroed."""
import sys
import numpy as np
sys.path.insert(0, '/home/terry/rltest/bc26a0/scripts')
import dpnet_io
R = '/home/terry/rltest/bc26a0/runs/rl/cpi-x0'
nets = dpnet_io.load_net(R + '/nets/0/net.json')
A = ['PASS', 'DROP_ALLY', 'THROW', 'NAP', 'BITE_KING', 'BITE_RAT', 'BITE_CAT', 'CAT_TRAP', 'LAY_LANE', 'DELIVER', 'CASH_KING']
for p in (2, 3):
    z = np.load(f'{R}/iter-0000/data/p{p}.npz')
    x = z['x'].astype(np.float32); fb = (z['flags'] & 1).astype(bool); k = z['k']
    L = nets[p]
    m = dpnet_io.fit_mask(z['mask'], len(x), L[-1]['w'].shape[0])
    t0 = np.where(m, dpnet_io.forward(L, x, exact=False), -np.inf).argmax(1)
    x[:, 32] = 0
    t1 = np.where(m, dpnet_io.forward(L, x, exact=False), -np.inf).argmax(1)
    ch = t0 != t1
    pairs = {}
    for a, b in zip(t0[ch].tolist(), t1[ch].tolist()):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
    top = sorted(pairs.items(), key=lambda t: -t[1])[:10]
    print(f'point {p}: argmax changed in {int(ch.sum())} of {len(x)} rows ({ch.mean():.4f}); ' +
          ', '.join(f'{A[a] if a < len(A) else a}->{A[b] if b < len(A) else b}:{c}' for (a, b), c in top))
    # fallback that remains if the argmax picks followed t1 (rough): argmax-fallback rows whose argmax changes
    af = fb & (k == t0)
    print(f'   argmax-pick fallbacks {int(af.sum())}; of these on rows whose argmax changes {int((af & ch).sum())}')
