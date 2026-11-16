"""Read-only: features constant in zdb-mode imitation data (untrained input columns) and their cpi-x0 values; nRatT probe."""
import sys, glob
import numpy as np
sys.path.insert(0, '/home/terry/rltest/bc26a0/scripts')
import decisions, dpnet_io
files = sorted(glob.glob('/home/terry/projects/vibe/bc26a0/runs/teacher-dp2/games/g00[0-1]??/decisions-*.bin.gz'))[:40]
T = {2: [], 3: [], 4: []}
for f in files:
    d = decisions.read(f)
    for p in T:
        s = d['point'] == p
        T[p].append(d['feats'][s])
R = '/home/terry/rltest/bc26a0/runs/rl/cpi-x0'
nets = dpnet_io.load_net(R + '/nets/0/net.json')
for p in (2, 3, 4):
    t = np.concatenate(T[p])
    L = nets[p]
    nin = L[0]['w'].shape[1]
    nf_t = int(np.max([a.shape[1] for a in T[p]]))
    z = np.load(f'{R}/iter-0000/data/p{p}.npz')
    x = z['x']
    print(f'point {p}: teacher rows {len(t)} width {nf_t}; cpi-x0 rows {len(x)} width {x.shape[1]}; net n_in {nin}')
    w1 = np.abs(L[0]['w'])
    for j in range(min(nf_t, x.shape[1])):
        if t[:, j].std() < 1e-3:
            c = t[0, j]
            share = np.mean(x[:, j] != c)
            print(f'   const col {j} = {c}: cpi-x0 share != const {share:.4f}, cpi-x0 mean {x[:, j].mean():.2f} max {x[:, j].max()}; |W1| col mean {w1[:, j].mean():.3f} (all-col median {np.median(w1.mean(0)):.3f})' if j < nin else f'   const col {j} = {c} (beyond n_in)')
    if p == 2:
        xf = x.astype(np.float32)
        def top(xx):
            lg = dpnet_io.forward(L, xx, exact=False)
            m = dpnet_io.fit_mask(z['mask'], len(xx), L[-1]['w'].shape[0])
            return np.where(m, lg, -np.inf).argmax(1)
        t0 = top(xf)
        x2 = xf.copy(); x2[:, 32] = 0
        t1 = top(x2)
        s = (t0 == 2) & (xf[:, 12] == 0)
        print(f'   probe nRatT->0: argmax THROW rows {int((t0 == 2).sum())} -> {int((t1 == 2).sum())}; carry=0 THROW rows {int(s.sum())}, of which still THROW {int((t1[s] == 2).sum())}; their new argmax {np.unique(t1[s], return_counts=True)}')
        c2 = xf[:, 12] == 2
        print(f'   carry=2 rows {int(c2.sum())}: THROW argmax {int((t0[c2] == 2).sum())} -> {int((t1[c2] == 2).sum())} with nRatT 0')
        print(f'   nRatT hist cpi-x0: {np.unique(np.minimum(x[:, 32], 30) // 5 * 5, return_counts=True)}')
        for lo in (0, 1, 5, 10, 15, 20, 25):
            sel = (xf[:, 32] >= lo) & (xf[:, 32] < lo + 5) & (xf[:, 12] == 0)
            if lo == 0:
                sel = (xf[:, 32] == 0) & (xf[:, 12] == 0)
            if lo == 1:
                sel = (xf[:, 32] >= 1) & (xf[:, 32] < 5) & (xf[:, 12] == 0)
            if sel.any():
                print(f'     carry=0, nRatT bucket {lo}: rows {int(sel.sum())}, argmax THROW share {np.mean(t0[sel] == 2):.4f}')
