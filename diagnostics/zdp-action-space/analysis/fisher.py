"""How much policy-gradient signal can the RL loss get from each record? (read-only; box)

For the floor-mixed policy p = (1-f) softmax(z/T) + f/m that rl_train differentiates, the score of option k is
g_k = d log p(k) / dz = (1-f) s_k (e_k - s) / p(k)  (s = softmax over the mask, T = 1). The Fisher trace
F = sum_k p(k) |g_k|^2 is the expected squared gradient a sampled record gives the logits. A uniform policy over m
options has F = 1 - 1/m; a saturated imitation net has F ~ 2 (1 - s_top). The floor adds sampled deviations but
almost no F, because g_k ~ s_k / p(k) -> 0 for floor-only options.

Per point, on cpi-x0 states with the v39 net: mean F per record at floor 0.005 and at the cpi-x0 floor, the share of
F from saturated rows (s_top > 0.999), the share of F carried by floor picks vs softmax picks, and the same for a
softened net (logits / 3) as a counterfactual.
"""
import sys, json
import numpy as np
sys.path.insert(0, 'scripts')
import dpnet_io

R = 'runs/rl/cpi-x0'
nets = dpnet_io.load_net(R + '/nets/0/net.json')
cfg = json.load(open('configs/cpi-x0.json'))
fpts = {int(k): v for k, v in (cfg.get('floor_points') or {}).items()}
rng = np.random.default_rng(0)
out = {}
for p in (1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 16):
    try:
        z = np.load(f'{R}/iter-0000/data/p{p}.npz')
    except FileNotFoundError:
        continue
    n = len(z['k'])
    idx = np.sort(rng.choice(n, size=min(n, 150000), replace=False))
    x = z['x'][idx].astype(np.float32)
    L = nets[p]
    n_out = L[-1]['w'].shape[0]
    m = dpnet_io.fit_mask(z['mask'][idx], len(idx), n_out)
    lg = dpnet_io.forward(L, x, exact=False)
    res = {}
    for tag, scale, f in (('f005', 1.0, 0.005), ('fcpi', 1.0, fpts.get(p, 0.005)), ('soft3_f005', 3.0, 0.005)):
        zz = np.where(m, lg / scale, -np.inf)
        zz = zz - zz.max(1, keepdims=True)
        e = np.where(m, np.exp(zz), 0.0)
        s = e / e.sum(1, keepdims=True)
        mm = m.sum(1, keepdims=True)
        pm = np.where(m, (1 - f) * s + f / mm, 0.0)
        # |e_k - s|^2 = 1 - 2 s_k + |s|^2
        s2 = (s * s).sum(1, keepdims=True)
        nk = np.where(m, 1 - 2 * s + s2, 0.0)
        ratio = np.where(m, (1 - f) * s / np.maximum(pm, 1e-300), 0.0)
        g2 = ratio ** 2 * nk                       # |g_k|^2
        F = (pm * g2).sum(1)                       # Fisher trace per row
        # split each row's F into the part carried by the floor's share of p(k) and the softmax's share
        Ff = ((f / mm) * g2).sum(1)
        top = s.max(1)
        sat = top > 0.999
        unif = (1 - 1 / mm[:, 0])
        res[tag] = dict(mean_F=float(F.mean()), mean_F_uniform=float(unif.mean()),
                        rel_to_uniform=float(F.mean() / unif.mean()),
                        sat_share_rows=float(sat.mean()), F_share_from_sat=float(F[sat].sum() / F.sum()),
                        F_share_floor=float(Ff.sum() / F.sum()),
                        dev_rate=float((1 - (pm * (s == top[:, None])).sum(1)).mean()),
                        F_p50=float(np.median(F)), F_p90=float(np.quantile(F, 0.9)))
    out[p] = res
    r0, r1, r2 = res['f005'], res['fcpi'], res['soft3_f005']
    print(f"p{p:<2} rows {len(idx):>6} m {float(m.sum(1).mean()):5.1f} | floor .005: F {r0['mean_F']:.4f} "
          f"({100 * r0['rel_to_uniform']:.1f}% of uniform), sat rows {r0['sat_share_rows']:.2f} carry "
          f"{r0['F_share_from_sat']:.2f} of F, floor carries {r0['F_share_floor']:.3f}, dev {r0['dev_rate']:.3f} | "
          f"cpi floor {fpts.get(p, 0.005)}: F {r1['mean_F']:.4f}, floor carries {r1['F_share_floor']:.3f}, dev "
          f"{r1['dev_rate']:.3f} | logits/3: F {r2['mean_F']:.4f} ({100 * r2['rel_to_uniform']:.1f}%), dev "
          f"{r2['dev_rate']:.3f}", flush=True)
json.dump(out, open('runs/fisher-cpi-x0.json', 'w'), indent=1)
