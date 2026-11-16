"""Smoothed start: distil a net's own policy toward (1 - EPS) p + EPS w at chosen points, on recorded states.

  smooth_start.py --net NET.json --data DIR [DIR ...] --groups configs/groups-s2.json --smooth 5:0.2,7:0.2 --out OUT

p is the net's softmax at temperature 1 over the allowed options (no floor). w is the floor-group weights of --groups
for that row (dpnet_io.floor_weights: dormant options get none); a row with no weighted group allowed keeps p. Each point
starts from its own weights and is trained (Adam, full-softmax KL(target || q) over allowed options) on the p<P>.npz
states of --data (rl_data output), so the result plays the net at floor 0 while EPS of its mass sits inside the softmax,
where a sampled deviation carries gradient (a floor pick carries almost none, docs/rl-es-plan.md section 17).
Inputs are standardised with the data's column statistics while training and folded back on export; other points are
written back byte for byte. OUT gets net.json, norm.npz (copied from beside NET.json) and smooth.json (per point: rows,
on held-out rows: KL to target before/after, argmax agreement with the input net and with the target, p_top and
Fisher trace as a share of uniform's).
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpnet_io  # noqa: E402


def stats(z, mask, target):
    """(KL(target || q), p_top, Fisher share of uniform) averaged over rows; z float64 logits."""
    zm = np.where(mask, z, -np.inf)
    zm = zm - zm.max(1, keepdims=True)
    q = np.where(mask, np.exp(zm), 0.0); q /= q.sum(1, keepdims=True)
    lq = np.where(mask, np.log(np.maximum(q, 1e-300)), 0.0)
    lt = np.where(target > 0, np.log(np.maximum(target, 1e-300)), 0.0)
    kl = (target * (lt - lq)).sum(1)
    m = mask.sum(1)
    fu = np.where(m > 1, 1 - 1 / np.maximum(m, 1), np.nan)
    fish = (1 - (q * q).sum(1)) / fu
    return float(kl.mean()), float(q.max(1).mean()), float(np.nanmean(fish)), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--data', nargs='+', required=True, help='rl_data output dirs holding p<P>.npz')
    ap.add_argument('--groups', required=True)
    ap.add_argument('--smooth', required=True, help='P:EPS,...')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--val-frac', type=float, default=0.1, help='held-out rows for the reported statistics')
    ap.add_argument('--max-rows', type=int, default=400_000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    sm = {int(k): float(v) for k, v in (x.split(':') for x in a.smooth.split(',') if x)}
    groups = dpnet_io.load_groups(a.groups)
    raw = dpnet_io.load_raw(a.net)
    out_nets = {int(p): v for p, v in raw['points'].items()}   # raw JSON layers, written back unchanged
    report = {}
    t0 = time.time()
    for p, eps in sorted(sm.items()):
        if str(p) not in raw['points']:
            raise SystemExit(f'point {p}: not in {a.net}')
        if groups.get(p) is None:
            raise SystemExit(f'point {p}: no group table in {a.groups}')
        xs, ms = [], []
        for d in a.data:
            f = Path(d) / f'p{p}.npz'
            if f.exists():
                z = np.load(f); xs.append(z['x']); ms.append(z['mask'])
        if not xs:
            raise SystemExit(f'point {p}: no p{p}.npz in --data')
        layers = dpnet_io.layers_from_raw(raw['points'][str(p)])
        n_in, n_out = layers[0]['w'].shape[1], layers[-1]['w'].shape[0]
        nf = max(x.shape[1] for x in xs)
        X = np.concatenate([np.pad(x, ((0, 0), (0, nf - x.shape[1]))) for x in xs]).astype(np.float64)
        M = np.concatenate([dpnet_io.fit_mask(m, len(m), n_out) for m in ms])
        x = np.zeros((len(X), n_in)); k = min(n_in, nf); x[:, :k] = X[:, :k]
        if len(x) > a.max_rows:
            idx = rng.choice(len(x), a.max_rows, replace=False); x, M = x[idx], M[idx]
        keep = M.sum(1) >= 1
        x, M = x[keep], M[keep]
        N = len(x)
        val = rng.random(N) < a.val_frac
        z0 = dpnet_io.forward(layers, x.astype(np.float32)).astype(np.float64)
        zm = np.where(M, z0, -np.inf); zm -= zm.max(1, keepdims=True)
        p0 = np.where(M, np.exp(zm), 0.0); p0 /= p0.sum(1, keepdims=True)
        w = dpnet_io.floor_weights(M, groups[p], n_out).astype(np.float64)
        ws = w.sum(1, keepdims=True)
        has = ws[:, 0] > 0
        w = np.where(has[:, None], w / np.maximum(ws, 1e-30), 0.0)
        target = np.where(has[:, None], (1 - eps) * p0 + eps * w, p0)
        kl0, pt0, f0, _ = stats(z0[val], M[val], target[val])
        _, pt_in, f_in, _ = stats(z0[val], M[val], p0[val])
        mu, sd = dpnet_io.col_stats(x)
        un = dpnet_io.unfold(layers, mu, sd)
        mods = []
        for i, l in enumerate(un):
            lin = torch.nn.Linear(l['w'].shape[1], l['w'].shape[0]).double()
            with torch.no_grad():
                lin.weight.copy_(torch.tensor(l['w'])); lin.bias.copy_(torch.tensor(l['b']))
            mods.append(lin)
            if i < len(un) - 1:
                mods.append(torch.nn.ReLU())
        model = torch.nn.Sequential(*mods)
        tr = ~val
        tX = torch.tensor((x[tr] - mu) / sd); tM = torch.tensor(M[tr]); tT = torch.tensor(target[tr])
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        for ep in range(a.epochs):
            perm = torch.randperm(int(tr.sum()))
            for s in range(0, N, a.batch):
                b = perm[s:s + a.batch]
                lq = torch.log_softmax(model(tX[b]).masked_fill(~tM[b], -1e30), 1)
                loss = -(tT[b] * torch.where(tM[b], lq, torch.zeros_like(lq))).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        lin = [m for m in model if isinstance(m, torch.nn.Linear)]
        new = dpnet_io.fold([dict(w=l.weight.detach().numpy(), b=l.bias.detach().numpy()) for l in lin], mu, sd)
        out_nets[p] = [dpnet_io.layer_json(l) for l in new]
        z1 = dpnet_io.forward(dpnet_io.layers_from_raw(out_nets[p]), x.astype(np.float32)).astype(np.float64)
        kl1, pt1, f1, q1 = stats(z1[val], M[val], target[val])
        kl1t = stats(z1[~val], M[~val], target[~val])[0]
        am = lambda z: np.where(M[val], z[val], -np.inf).argmax(1)
        agree = float((am(z1) == am(z0)).mean())
        agree_t = float((am(z1) == target[val].argmax(1)).mean())
        report[p] = dict(eps=eps, rows=int(N), val_rows=int(val.sum()), rows_with_weight=float(has.mean()),
                         kl_before=kl0, kl_after=kl1, kl_after_train=kl1t, argmax_agree=agree, argmax_agree_target=agree_t,
                         p_top_in=pt_in, p_top_target=float(target[val].max(1).mean()), p_top_out=pt1, fisher_in=f_in,
                         fisher_out=f1)
        print(f'point {p}: eps {eps}, {N} rows ({has.mean():.3f} with weighted groups), held-out {val.sum()}: KL to target '
              f'{kl0:.4f} -> {kl1:.4f} (train {kl1t:.4f}); argmax agree {agree:.4f} (target argmax {agree_t:.4f}); p_top '
              f'{pt_in:.3f} -> {pt1:.3f} (target {target[val].max(1).mean():.3f}); Fisher share {f_in:.3f} -> {f1:.3f}  '
              f'[{time.time() - t0:.0f}s]', flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in raw.items() if k != 'points'}
    meta.setdefault('meta', {})
    if isinstance(meta['meta'], dict):
        meta['meta']['smooth_start'] = dict(net=a.net, groups=a.groups, smooth={str(k): v for k, v in sm.items()})
    dpnet_io.save_net(out / 'net.json', out_nets, meta)
    src_norm = Path(a.net).parent / 'norm.npz'
    if src_norm.exists():
        shutil.copyfile(src_norm, out / 'norm.npz')
    (out / 'smooth.json').write_text(json.dumps({str(k): v for k, v in report.items()}, indent=1))
    print(f'wrote {out}/net.json')


if __name__ == '__main__':
    main()
