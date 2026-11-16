"""Stage-B CPI step (docs/rl-es-plan.md sections 5 and 10): a last-layer update of the playing net from td_check.py's
state-dependent advantage fit, written as the two judged nets theta + alpha Delta and theta - alpha Delta.

cpi_step.py --net NET.json --fit FIT.json --data DIR [DIR ...] --config C --out OUT [--points 12,13 | --min-t 2]
            [--tv 0.1] [--alpha 1] [--build PREFIX]

FIT is td_check.py --cpi --save-fit on records played by NET. Per point p (default: the fit's points whose mean
per-fold out-of-fold t, t_fold_mean, is >= --min-t; a fit saved before t_fold_mean existed falls back to the pooled
t_oof / sqrt(2)), with h the net's last hidden layer after ReLU and pc_r(h) = ((h - mu) / sd) . vt_r / s_r the fit's
basis, Delta adds eta_p (W[k, 0] + sum_r W[k, 1 + r] pc_r(h)) to option k's logit: the last layer gets
    w[k] += eta_p sum_r W[k, 1 + r] vt_r / (sd s_r),    b[k] += eta_p (W[k, 0] - sum_r W[k, 1 + r] ((mu / sd) . vt_r) / s_r).
eta_p is set so that the mean total-variation distance between pi_{theta + Delta} and pi_theta over the point's
records in DIR (the config's floor, floor_points and temperature) is --tv. Writes OUT/delta.json (per point eta, tv,
top-choice flip rate), OUT/plus/net.json and OUT/minus/net.json (theta +- alpha Delta; other points keep their raw
values; norm.npz copied); --build PREFIX builds and compiles the packages PREFIXp (plus) and PREFIXm (minus).
"""
import argparse, copy, json, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dpnet_io  # noqa: E402
from sabotage import load_records  # noqa: E402


def affine(fit_p, version=None):
    """(dW [K, H], db [K]) of Delta at eta = 1 from one point's fit."""
    bases = fit_p['basis']
    if version is None:
        if len(bases) != 1:
            raise SystemExit(f'fit has bases for versions {sorted(bases)}: pass --version')
        version = next(iter(bases))
    B = bases[str(version)]
    mu, sd, vt, s = (np.asarray(B[k], dtype=np.float64) for k in ('mu', 'sd', 'vt', 's'))
    W = np.asarray(fit_p['W'], dtype=np.float64)
    A = (vt / s[:, None]) / sd[None, :]            # [P, H]: pc_r(h) = A[r] . h + a0[r]
    a0 = -(A * mu[None, :]).sum(1)
    dW = W[:, 1:] @ A
    db = W[:, 0] + W[:, 1:] @ a0
    return dW, db


def tv_of(z0, p0, off, eta, mask, T, floor):
    p1 = dpnet_io.probs((z0 + eta * off).astype(np.float32), mask, T, floor).astype(np.float64)
    return float(0.5 * np.abs(p1 - p0).sum(1).mean()), float((p1.argmax(1) != p0.argmax(1)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--net', required=True)
    ap.add_argument('--fit', required=True)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--points', default='')
    ap.add_argument('--min-t', type=float, default=2.0)
    ap.add_argument('--tv', type=float, default=0.1)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--version')
    ap.add_argument('--max-rows', type=int, default=200000)
    ap.add_argument('--build')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    floor, T = cfg.get('floor', 0.005), cfg.get('temperature', 1.0)
    fpts = {int(k): float(v) for k, v in (cfg.get('floor_points') or {}).items()}
    fit = json.loads(Path(a.fit).read_text())['points']
    if a.points:
        points = [int(p) for p in a.points.split(',')]
    else:
        tsel = {int(p): f['t_fold_mean'] if f.get('t_fold_mean') is not None else (f.get('t_oof') or 0) / 2 ** 0.5
                for p, f in fit.items()}
        points = sorted(p for p, t in tsel.items() if t >= a.min_t)
    if not points:
        raise SystemExit(f'no point with out-of-fold t >= {a.min_t}')
    raw = dpnet_io.load_raw(a.net)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(net=a.net, fit=a.fit, tv_target=a.tv, alpha=a.alpha, points={})
    nets = {s: copy.deepcopy(raw) for s in ('plus', 'minus')}
    for p in points:
        layers = dpnet_io.layers_from_raw(raw['points'][str(p)])
        dW, db = affine(fit[str(p)], a.version)
        K, H = layers[-1]['w'].shape
        if dW.shape != (K, H):
            raise SystemExit(f'point {p}: fit gives {dW.shape}, the net {K, H}')
        x, mask = load_records(a.data, p, layers[0]['w'].shape[1], K, a.max_rows)
        h = np.maximum(dpnet_io.forward(layers[:-1], x, exact=False), 0).astype(np.float64)
        z0 = dpnet_io.forward(layers, x, exact=False)
        off = (h @ dW.T + db).astype(np.float64)
        fl = fpts.get(p, floor)
        p0 = dpnet_io.probs(z0, mask, T, fl).astype(np.float64)
        lo, hi = 0.0, 1.0
        while tv_of(z0, p0, off, hi, mask, T, fl)[0] < a.tv and hi < 1e6:
            lo, hi = hi, hi * 2
        for _ in range(30):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if tv_of(z0, p0, off, m, mask, T, fl)[0] < a.tv else (lo, m)
        eta = hi
        st = dict(records=int(len(x)), eta=eta, offset_sd=float((eta * off).std()))
        for sgn, name in ((1, 'plus'), (-1, 'minus')):
            st[f'tv_{name}'], st[f'flip_{name}'] = tv_of(z0, p0, off, sgn * a.alpha * eta, mask, T, fl)
            last = dict(nets[name]['points'][str(p)][-1])
            w = np.asarray(last['w'], dtype=np.float64).reshape(K, H) + sgn * a.alpha * eta * dW
            b = np.asarray(last['b'], dtype=np.float64) + sgn * a.alpha * eta * db
            last['w'] = [float(v) for v in w.astype(np.float32).ravel()]
            last['b'] = [float(v) for v in b.astype(np.float32)]
            ls = list(nets[name]['points'][str(p)])
            ls[-1] = last
            nets[name]['points'][str(p)] = ls
        report['points'][p] = st
        print(f"p{p}: {st['records']} records, eta {eta:.3g}, logit offset sd {st['offset_sd']:.3f} | plus tv "
              f"{st['tv_plus']:.4f} flip {st['flip_plus']:.4f} | minus tv {st['tv_minus']:.4f} flip {st['flip_minus']:.4f}")
    norm = Path(a.net).with_name('norm.npz')
    for name, doc in nets.items():
        doc['cpi_step'] = dict(sign=name, alpha=a.alpha, tv=a.tv, points=points, fit=a.fit)
        d = out / name
        d.mkdir(exist_ok=True)
        (d / 'net.json').write_text(json.dumps(doc))
        if norm.exists():
            shutil.copy(norm, d / 'norm.npz')
    (out / 'delta.json').write_text(json.dumps(report, indent=1))
    if a.build:
        from es_schedule import build
        for name, suf in (('plus', 'p'), ('minus', 'm')):
            build(cfg, out / name / 'net.json', a.build + suf)
            print(f'built {a.build + suf}')


if __name__ == '__main__':
    main()
