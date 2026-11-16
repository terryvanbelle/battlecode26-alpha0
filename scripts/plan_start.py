"""Start nets for design C's plan heads (DP.java, points 20-27; docs/rl-es-plan.md section 21).

  plan_start.py --out runs/plan/start [--sigma 0.6] [--uniform 0.05] [--hidden 64,64] [--heads 20-27] [--seed 0]

Each head is an MLP n_in-64-64-K whose last layer starts at zero, so the start policy is exactly its bias:
softmax(b) = (1 - U) N(k; zdb bin, SIGMA) / sum + U / K, a discretised Gaussian around zdb's bin (DP/Plan.ZDB) with
U spread uniformly. Hidden layers are Kaiming on inputs standardised by fixed per-feature scales (SCALE below, from
DP.feats' ranges), folded into the first layer; norm.npz carries the same (mu, sd), so rl_train unfolds onto those
scales (not onto one batch's statistics). Prints each head's p(zdb bin) and Fisher trace as a share of uniform's.
--uniform-start writes a uniform policy (b = 0) instead (design C's T3 check).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpnet_io  # noqa: E402
from dp_points import feature_counts, option_counts  # noqa: E402

ZDB = {20: 0, 21: 3, 22: 4, 23: 3, 24: 3, 25: 3, 26: 3, 27: 1}   # Plan.ZDB by point
# DP.feats: (mu, scale) per feature: round/10, coop, cause, since break, babies, spawned, kings, cheese/10,
# surplus/10 + 32768, rate/10, budget, W, H, area/10, sym mask, start-centre distance, mines, near mines, cats,
# eking age, alert, threat, enemies, rat cost, plan in force x 8 (bins), draws x 8 (bin + 1)
FEAT = ([(100, 60), (0.5, 0.5), (1, 1), (300, 500), (12, 10), (12, 10), (2, 1), (100, 150), (32768, 500), (20, 20),
         (100, 150), (40, 12), (40, 12), (160, 100), (4, 2), (15, 8), (8, 5), (4, 3), (0.5, 0.7), (500, 400),
         (0.2, 0.4), (1, 2), (1, 2), (60, 40)] + [(3, 2)] * 8 + [(2, 2)] * 8)


def start_bias(K, z, sigma, u):
    k = np.arange(K)
    g = np.exp(-0.5 * ((k - z) / sigma) ** 2)
    p = (1 - u) * g / g.sum() + u / K
    return np.log(p), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--sigma', type=float, default=0.6)
    ap.add_argument('--uniform', type=float, default=0.05, help='share of the start spread uniformly (U)')
    ap.add_argument('--uniform-start', action='store_true', help='uniform policy (b = 0): the T3 check')
    ap.add_argument('--hidden', default='64,64')
    ap.add_argument('--heads', default='20-27')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.heads.split('-')) if '-' in a.heads else (int(a.heads), int(a.heads))
    heads = list(range(lo, hi + 1))
    opts, feats = option_counts(), feature_counts()
    hidden = [int(h) for h in a.hidden.split(',')]
    rng = np.random.default_rng(a.seed)
    nets, norm, info = {}, {}, {}
    for p in heads:
        K, n_in = opts[p], feats[p]
        if n_in != len(FEAT):
            raise SystemExit(f'point {p}: {n_in} features in DP.java, FEAT has {len(FEAT)}')
        mu = np.array([m for m, _ in FEAT], np.float64); sd = np.array([s for _, s in FEAT], np.float64)
        layers, d = [], n_in
        for h in hidden:
            layers.append(dpnet_io.kaiming(rng, d, h)); d = h
        bias, pr = (np.zeros(K), np.full(K, 1.0 / K)) if a.uniform_start else start_bias(K, ZDB[p], a.sigma, a.uniform)
        layers.append(dict(w=np.zeros((K, d)), b=bias))
        nets[p] = dpnet_io.fold(layers, mu, sd)
        norm[p] = (mu.astype(np.float32), sd.astype(np.float32))
        fish = (1 - (pr ** 2).sum()) / (1 - 1 / K)
        info[p] = dict(K=K, zdb_bin=ZDB[p], p=[round(float(x), 4) for x in pr], p_zdb=round(float(pr[ZDB[p]]), 4),
                       fisher_share=round(float(fish), 4))
        print(f'point {p}: K {K}, zdb bin {ZDB[p]}, p(zdb) {pr[ZDB[p]]:.3f}, Fisher share {fish:.3f}, p {np.round(pr, 3).tolist()}')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = dict(meta=dict(plan_start=dict(sigma=a.sigma, uniform=a.uniform, uniform_start=a.uniform_start,
                                          hidden=hidden, seed=a.seed)))
    dpnet_io.save_net(out / 'net.json', nets, meta)
    dpnet_io.save_norm(out / 'norm.npz', norm)
    (out / 'plan_start.json').write_text(json.dumps({str(k): v for k, v in info.items()}, indent=1))
    # check: the folded net's logits on random inputs equal the bias (zero last layer)
    for p in heads:
        x = rng.uniform(0, 100, (4, feats[p])).astype(np.float32)
        z = dpnet_io.forward(dpnet_io.layers_from_raw(dpnet_io.load_raw(out / 'net.json')['points'][str(p)]), x)
        assert np.allclose(z - z[:, :1], (nets[p][-1]['b'] - nets[p][-1]['b'][0])[None, :], atol=1e-4), p
    print(f'wrote {out}/net.json, norm.npz, plan_start.json')


if __name__ == '__main__':
    main()
