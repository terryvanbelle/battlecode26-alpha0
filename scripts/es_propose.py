"""Stage-B proposal from ES data (docs/rl-es-plan.md section 5): the pooled ES gradient over several iterations of
a run, turned into one direction Delta for the 1-D judge (theta + alpha Delta vs theta - alpha Delta).

es_propose.py --config C --run RUN --iters 1-8 --data DIR [DIR ...] --out OUT [--tv 0.1] [--alpha 1] [--theta K]
              [--build PREFIX]

Per coordinate j, every direction d of the listed iterations (dirs.json eps and sigma, update.json unit scores s)
gives x_dj = eps_dj sbar_d / sigma_dj, sbar_d the mean score of the direction's units: E x_j = grad_j f (the units
score theta+ - 0.5 with theta+- = phi +- sigma eps). The proposal is the natural step Delta_j = sigma_ref_j^2 mean_d
x_dj (sigma_ref: the last listed iteration's sigma); the per-coordinate t and the split-half cosine of the pooled
gradient (even vs odd directions) are reported. Delta is scaled by eta so that the mean over the points of the
total-variation distance between pi_{theta + eta Delta} and pi_theta on the records in DIR (the config's floors) is
--tv. theta is the run's base with phi/<K>.json (--theta, default 0: the base itself). Writes OUT/delta.json and
OUT/plus|minus/net.json (theta +- alpha eta Delta); --build PREFIX builds PREFIXp and PREFIXm.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dpnet_io, es_net  # noqa: E402
from es_schedule import run_dir, iter_dir  # noqa: E402
from sabotage import load_records  # noqa: E402


def ints(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def pooled_gradient(rd, iters):
    """{coord: list of x_dj}, sigma of the last iteration, directions used."""
    xs, sig_ref, nd = {}, None, 0
    for k in iters:
        I = iter_dir(rd, k)
        dd = json.loads((I / 'dirs.json').read_text())
        units = json.loads((I / 'update.json').read_text())['units']
        by = {}
        for u in units:
            if u.get('s') is not None:
                by.setdefault(u['dir'], []).append(u['s'])
        sig = dd['sigma']
        for d in dd['dirs']:
            ss = by.get(d['i'])
            if not ss:
                continue
            sb = sum(ss) / len(ss)
            nd += 1
            for c, e in d['eps'].items():
                xs.setdefault(c, []).append(e * sb / sig[c])
        sig_ref = sig
    return xs, sig_ref, nd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--iters', required=True)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tv', type=float, default=0.1)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--theta', type=int, default=0)
    ap.add_argument('--max-rows', type=int, default=100000)
    ap.add_argument('--build')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    rd = run_dir(cfg, a.run)
    base_raw = json.loads((rd / 'base' / 'net.json').read_text())
    phi = es_net.load_phi(rd / 'phi' / f'{a.theta}.json') if a.theta else es_net.zero_phi(base_raw, cfg['points'])
    for p in cfg['points']:   # a phi saved before low-rank coordinates existed
        phi.setdefault(int(p), es_net.zero_phi(base_raw, [p])[int(p)])
    equiv = es_net.load_equiv(cfg.get('equiv', ''))
    xs, sig, nd = pooled_gradient(rd, ints(a.iters))
    coords = sorted(xs)
    grad, tstat = {}, {}
    for c in coords:
        x = np.asarray(xs[c])
        grad[c] = float(x.mean())
        se = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
        tstat[c] = grad[c] / se if se > 0 else 0.0
    ev = {c: np.asarray(xs[c])[0::2].mean() for c in coords}
    od = {c: np.asarray(xs[c])[1::2].mean() for c in coords}
    va, vb = np.array([ev[c] for c in coords]), np.array([od[c] for c in coords])
    cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-30))
    delta = {c: sig[c] ** 2 * grad[c] for c in coords}
    ts = np.abs(np.array([tstat[c] for c in coords]))
    print(f'{len(coords)} coordinates over {nd} directions (iterations {a.iters}): |t|>2 share {float((ts > 2).mean()):.3f} '
          f'(chance 0.046), max |t| {ts.max():.2f}, split-half cosine {cos:+.3f}')
    floor, T = cfg.get('floor', 0.005), cfg.get('temperature', 1.0)
    fpts = {int(k): float(v) for k, v in (cfg.get('floor_points') or {}).items()}
    pts = sorted({es_net.coord_point(c) for c in coords})
    centre = es_net.apply_phi(base_raw, phi)
    recs, z0, p0 = {}, {}, {}
    for p in pts:
        ls = dpnet_io.layers_from_raw(centre['points'][str(p)])
        x, mask = load_records(a.data, p, ls[0]['w'].shape[1], ls[-1]['w'].shape[0], a.max_rows)
        recs[p] = (x, mask)
        z0[p] = dpnet_io.forward(ls, x, exact=False)
        p0[p] = dpnet_io.probs(z0[p], mask, T, fpts.get(p, floor)).astype(np.float64)

    def tv(eta):
        th = es_net.phi_add(phi, {c: eta * v for c, v in delta.items()}, equiv)
        net = es_net.apply_phi(base_raw, th)
        out = {}
        for p in pts:
            x, mask = recs[p]
            ls = dpnet_io.layers_from_raw(net['points'][str(p)])
            pr = dpnet_io.probs(dpnet_io.forward(ls, x, exact=False), mask, T, fpts.get(p, floor)).astype(np.float64)
            out[p] = (float(0.5 * np.abs(pr - p0[p]).sum(1).mean()), float((pr.argmax(1) != p0[p].argmax(1)).mean()))
        return out, th, net

    def mean_tv(eta):
        return float(np.mean([v[0] for v in tv(eta)[0].values()]))
    lo, hi = 0.0, 1.0
    while mean_tv(hi) < a.tv and hi < 1e6:
        lo, hi = hi, hi * 2
    for _ in range(25):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if mean_tv(m) < a.tv else (lo, m)
    eta = hi
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = dict(run=str(rd), iters=a.iters, theta=a.theta, directions=nd, coords=len(coords), split_half_cos=cos,
               t_gt2_share=float((ts > 2).mean()), tv_target=a.tv, alpha=a.alpha, eta=eta, points={},
               top=sorted(([c, round(delta[c] * eta, 3), round(tstat[c], 2)] for c in coords), key=lambda r: -abs(r[2]))[:12])
    for sgn, name in ((1, 'plus'), (-1, 'minus')):
        st, th, net = tv(sgn * a.alpha * eta)
        for p, (t_, f_) in st.items():
            rep['points'].setdefault(str(p), {})[f'tv_{name}'] = round(t_, 4)
            rep['points'][str(p)][f'flip_{name}'] = round(f_, 4)
        net['es_proposal'] = dict(sign=name, alpha=a.alpha, eta=eta, iters=a.iters, run=str(rd))
        (out / name).mkdir(exist_ok=True)
        (out / name / 'net.json').write_text(json.dumps(net))
        es_net.save_phi(out / name / 'phi.json', th, sign=name, alpha=a.alpha, eta=eta)
    (out / 'delta.json').write_text(json.dumps(dict(rep, delta=delta), indent=1))
    for p, st in rep['points'].items():
        print(f"p{p}: plus tv {st['tv_plus']:.4f} flip {st['flip_plus']:.4f} | minus tv {st['tv_minus']:.4f} flip {st['flip_minus']:.4f}")
    print(f'eta {eta:.4g}; top coordinates (coord, step, t): {rep["top"][:6]}')
    if a.build:
        from es_schedule import build
        for name, suf in (('plus', 'p'), ('minus', 'm')):
            build(cfg, out / name / 'net.json', a.build + suf)
            print(f'built {a.build + suf}')


if __name__ == '__main__':
    main()
