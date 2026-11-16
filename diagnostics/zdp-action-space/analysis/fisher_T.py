"""Per point, the softmax temperature T at which the v39 net's mean Fisher trace (floor 0) reaches FRAC of a uniform
policy's, on cpi-x0 states (box, read-only). Prints T, the deviation rate (1 - mean top probability) and the top-1
probability quantiles at that T.   fisher_T.py NET_JSON FRAC
"""
import sys, json
import numpy as np
sys.path.insert(0, 'scripts')
import dpnet_io

R = 'runs/rl/cpi-x0'
nets = dpnet_io.load_net(sys.argv[1])
frac = float(sys.argv[2])
rng = np.random.default_rng(0)
out = {}

def stats(lg, m, T):
    zz = np.where(m, lg / T, -np.inf)
    zz = zz - zz.max(1, keepdims=True)
    e = np.where(m, np.exp(zz), 0.0)
    s = e / e.sum(1, keepdims=True)
    s2 = (s * s).sum(1)
    F = (s * (1 - 2 * s + s2[:, None])).sum(1)           # floor 0: p = s, |g_k|^2 = |e_k - s|^2
    unif = 1 - 1 / m.sum(1)
    return F.mean() / unif.mean(), s.max(1)

for p in (1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 16):
    try:
        z = np.load(f'{R}/iter-0000/data/p{p}.npz')
    except FileNotFoundError:
        continue
    n = len(z['k'])
    idx = np.sort(rng.choice(n, size=min(n, 60000), replace=False))
    x = z['x'][idx].astype(np.float32)
    L = nets[p]
    n_out = L[-1]['w'].shape[0]
    m = dpnet_io.fit_mask(z['mask'][idx], len(idx), n_out)
    lg = dpnet_io.forward(L, x, exact=False).astype(np.float64)
    r1, _ = stats(lg, m, 1.0)
    lo, hi = 1.0, 1.0
    if r1 >= frac:
        T = 1.0
    else:
        while stats(lg, m, hi)[0] < frac and hi < 1e4: hi *= 2
        for _ in range(40):
            mid = (lo * hi) ** 0.5
            if stats(lg, m, mid)[0] < frac: lo = mid
            else: hi = mid
        T = hi
    r, top = stats(lg, m, T)
    out[p] = dict(T=T, rel_T1=float(r1), rel=float(r), dev=float(1 - top.mean()),
                  top_q=[float(np.quantile(top, q)) for q in (0.1, 0.5, 0.9)])
    print(f'p{p:<2} T={T:7.2f}  F/unif at T=1 {100 * r1:5.1f}% -> {100 * r:5.1f}%  dev {1 - top.mean():.3f}  '
          f'top q10/50/90 {np.quantile(top, 0.1):.2f}/{np.quantile(top, 0.5):.2f}/{np.quantile(top, 0.9):.2f}', flush=True)
json.dump(out, open(f'runs/aspace/fisher_T{int(round(100 * frac))}.json', 'w'), indent=1)
