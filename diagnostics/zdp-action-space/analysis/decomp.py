"""Read-only: decompose cpi-x0 PRE/POST/FACE fallbacks; THROW-with-carry fallback by carry age."""
import sys
import numpy as np
sys.path.insert(0, '/home/terry/rltest/bc26a0/scripts')
import dpnet_io
R = '/home/terry/rltest/bc26a0/runs/rl/cpi-x0'
nets = dpnet_io.load_net(R + '/nets/0/net.json')
for p in (2, 3, 4):
    z = np.load(f'{R}/iter-0000/data/p{p}.npz')
    x = z['x'].astype(np.float32); k = z['k'].astype(np.int64); fb = (z['flags'] & 1).astype(bool)
    L = nets[p]
    lg = dpnet_io.forward(L, x, exact=False)
    m = dpnet_io.fit_mask(z['mask'], len(k), L[-1]['w'].shape[0])
    top = np.where(m, lg, -np.inf).argmax(1)
    lp = np.where(m, lg, -np.inf); lp = lp - lp.max(1, keepdims=True); pr = np.exp(lp); pr /= pr.sum(1, keepdims=True)
    pk = pr[np.arange(len(k)), np.minimum(k, pr.shape[1] - 1)]
    nall = m.sum(1)
    nfb = int(fb.sum())
    arg = k == top
    print(f'point {p}: rows {len(k)} fallback {nfb} ({fb.mean():.4f}); argmax picks fb {int((fb & arg).sum())}; non-argmax picks {int((~arg).sum())} fb {int((fb & ~arg).sum())}')
    # non-argmax picks: floor-like (net prob < 1e-3) vs natural (net prob >= 1e-3)
    lowp = pk < 1e-3
    print(f'   non-argmax with net p<1e-3 (floor): {int((~arg & lowp).sum())} fb {int((fb & ~arg & lowp).sum())}; net p>=1e-3: {int((~arg & ~lowp).sum())} fb {int((fb & ~arg & ~lowp).sum())}')
    if p == 2:
        c = x[:, 12]
        for cv in (0, 1, 2):
            s = (k == 2) & (c == cv)
            print(f'   picked THROW carry={cv}: {int(s.sum())} fb {int((fb & s).sum())}')
        s = (k == 2) & (c == 2) & arg
        age = x[:, 13]
        for lo, hi in ((0, 6), (6, 64)):
            ss = s & (age >= lo) & (age < hi)
            print(f'   argmax THROW, carry=2, age [{lo},{hi}): n {int(ss.sum())} fb {fb[ss].mean() if ss.any() else float("nan"):.3f}')
        oth = fb & ~(k == 2)
        v, cnt = np.unique(k[oth], return_counts=True)
        print('   non-THROW fallbacks by option', dict(zip(v.tolist(), cnt.tolist())))
