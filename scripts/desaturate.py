"""Desaturate decision nets cheaply (docs/rl-plan.md section 7): scale the last layer (weights and bias) of the chosen
points by 1/tau, i.e. softmax(z / tau) -- the same as sampling at temperature tau, but baked into net.json so the build
keeps T = 1. Other points are copied value-for-value (their W.java entries stay byte-identical). norm.npz beside the
input is copied (the scale does not touch layer 1's folding).

desaturate.py --net runs/dp/net-v2/net.json --out runs/rl/X/start/net.json --points 5,7 --tau 2
desaturate.py --report DATA_DIR [DATA_DIR...] --net NET.json [--net2 NET2.json] [--points 5,7,...] [--floor 0.005]
    saturation per point on rl_data.py p<point>.npz records: the fraction of records (k picked, >= 2 options allowed)
    whose softmax part (before the floor) gives the sampled k more than 0.999 (sat_k), and gives any option more than
    0.999 (sat_max), under NET (and NET2), recomputed from the recorded features and masks; plus mean KL(NET2 || NET).
desaturate.py --self-test
"""
import argparse, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dpnet_io  # noqa: E402

NONE = 0xFFFF


def desaturate(raw, points, tau):
    """A new net.json dict with the last layer of each point in points scaled by 1/tau (others untouched)."""
    doc = dict(raw)
    pts = {}
    for ps, layers in raw['points'].items():
        if int(ps) in points:
            last = dict(layers[-1])
            last['w'] = [v / tau for v in last['w']]
            last['b'] = [v / tau for v in last['b']]
            pts[ps] = list(layers[:-1]) + [last]
        else:
            pts[ps] = layers
    doc['points'] = pts
    doc['desaturated'] = dict(points=sorted(points), tau=tau, **({'from': raw['desaturated']} if 'desaturated' in raw else {}))
    return doc


def soft_probs(layers, x, mask):
    """Softmax part (floor 0) as the bot computes it."""
    return dpnet_io.probs(dpnet_io.forward(layers, x), mask, 1.0, 0.0).astype(np.float64)


def report(dirs, net, net2, points, floor, max_rows=200000, seed=0, floor_points=None):
    """floor_points: {point: floor} overrides (the build's W.FL)."""
    rng = np.random.default_rng(seed)
    floor0 = floor
    a = dpnet_io.load_net(net)
    b = dpnet_io.load_net(net2) if net2 else None
    out = {}
    for p in points:
        parts = [np.load(Path(d) / f'p{p}.npz') for d in dirs if (Path(d) / f'p{p}.npz').exists()]
        if not parts or p not in a:
            continue
        floor = (floor_points or {}).get(p, floor0)
        n_out = a[p][-1]['w'].shape[0]
        x = np.concatenate([np.pad(z['x'], ((0, 0), (0, max(0, a[p][0]['w'].shape[1] - z['x'].shape[1])))) for z in parts])
        k = np.concatenate([z['k'] for z in parts]).astype(np.int64)
        mask = np.concatenate([dpnet_io.fit_mask(z['mask'], len(z['k']), n_out) for z in parts])
        keep = (k != NONE) & (k < n_out) & (mask.sum(1) > 1)
        keep &= np.where(keep, mask[np.arange(len(k)), np.minimum(k, n_out - 1)], False)
        idx = np.nonzero(keep)[0]
        if len(idx) > max_rows:
            idx = np.sort(rng.choice(idx, max_rows, replace=False))
        x, k, mask = x[idx], k[idx], mask[idx]
        row = dict(records=int(len(idx)), options=int(n_out), mean_allowed=round(float(mask.sum(1).mean()), 3), floor=floor)
        for tag, nets in (('net', a), ('net2', b)):
            if nets is None:
                continue
            ps = soft_probs(nets[p], x, mask)
            row[f'sat_k_{tag}'] = round(float((ps[np.arange(len(k)), k] > 0.999).mean()), 4)
            row[f'sat_max_{tag}'] = round(float((ps.max(1) > 0.999).mean()), 4)
            pf = dpnet_io.probs(dpnet_io.forward(nets[p], x), mask, 1.0, floor).astype(np.float64)
            row[f'mean_pk_{tag}'] = round(float(pf[np.arange(len(k)), k].mean()), 4)
            row[f'entropy_{tag}'] = round(float(-(np.where(mask, pf * np.log(np.maximum(pf, 1e-30)), 0)).sum(1).mean()), 4)
            if tag == 'net2':
                pa = dpnet_io.probs(dpnet_io.forward(a[p], x), mask, 1.0, floor).astype(np.float64)
                kl = np.where(mask, pf * (np.log(np.maximum(pf, 1e-30)) - np.log(np.maximum(pa, 1e-30))), 0).sum(1)
                row['kl_net2_net'] = round(float(kl.mean()), 4)
        out[p] = row
    return out


def self_test():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        nets = {}
        for p, (nin, nout) in {5: (12, 5), 6: (10, 13)}.items():
            dims = [nin, 16, 16, nout]
            nets[p] = [dict(w=rng.normal(0, 3, (o, i)), b=rng.normal(0, 1, o)) for i, o in zip(dims, dims[1:])]
        src = td / 'a' / 'net.json'; src.parent.mkdir()
        dpnet_io.save_net(src, nets, dict(hidden=[16, 16]))
        dpnet_io.save_norm(src.parent / 'norm.npz', {5: (np.zeros(12, np.float32), np.ones(12, np.float32))})
        dst = td / 'b' / 'net.json'
        for tau in (2.0, 3.0):
            main(['--net', str(src), '--out', str(dst), '--points', '5', '--tau', str(tau)])
            ra, rb = dpnet_io.load_raw(src), dpnet_io.load_raw(dst)
            assert ra['points']['6'] == rb['points']['6'], 'untouched point changed'
            assert ra['points']['5'][:-1] == rb['points']['5'][:-1], 'hidden layers changed'
            assert (dst.parent / 'norm.npz').exists(), 'norm.npz not copied'
            a, b = dpnet_io.load_net(src), dpnet_io.load_net(dst)
            x = rng.integers(0, 40, (500, 12))
            za, zb = dpnet_io.forward(a[5], x, exact=False).astype(np.float64), dpnet_io.forward(b[5], x, exact=False).astype(np.float64)
            assert np.abs(zb - za / tau).max() <= 1e-5 * np.abs(za).max(), 'logits not scaled by 1/tau'
            mask = rng.random((500, 5)) < 0.8; mask[:, 0] = True
            pa = dpnet_io.probs(dpnet_io.forward(a[5], x), mask, tau, 0.005)
            pb = dpnet_io.probs(dpnet_io.forward(b[5], x), mask, 1.0, 0.005)
            assert np.abs(pa - pb).max() < 1e-3, 'not equal to sampling at temperature tau'
            # saturation report: synthetic p5.npz drawn from the source net
            dd = td / 'data'; dd.mkdir(exist_ok=True)
            pr = dpnet_io.probs(dpnet_io.forward(a[5], x), mask, 1.0, 0.005).astype(np.float64)
            k = np.array([rng.choice(5, p=r / r.sum()) for r in pr])
            np.savez(dd / 'p5.npz', x=x.astype(np.uint16), k=k.astype(np.int32), mask=mask)
            r = report([dd], src, dst, [5], 0.005)[5]
            assert r['sat_max_net2'] <= r['sat_max_net'] and r['entropy_net2'] >= r['entropy_net'], r
    print('desaturate self-test OK')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--net')
    ap.add_argument('--out', help='output net.json')
    ap.add_argument('--points', default='5,7,10,11,12,13')
    ap.add_argument('--tau', type=float, default=2.0)
    ap.add_argument('--report', nargs='+', help='rl_data.py data dirs: print the saturation report')
    ap.add_argument('--net2', default='')
    ap.add_argument('--floor', type=float, default=0.005)
    ap.add_argument('--floor-points', default='', help='per-point floors P:F,... (the build\'s W.FL)')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    points = {int(v) for v in a.points.split(',') if v}
    if a.report:
        fp = {int(k): float(v) for k, v in (x.split(':') for x in a.floor_points.split(',') if x)}
        r = report(a.report, a.net, a.net2, sorted(points), a.floor, floor_points=fp)
        print(json.dumps({str(k): v for k, v in r.items()}, indent=1))
        return
    if not (a.net and a.out) or a.tau <= 0:
        raise SystemExit('--net, --out and tau > 0 required')
    raw = dpnet_io.load_raw(a.net)
    missing = points - {int(p) for p in raw['points']}
    if missing:
        raise SystemExit(f'points {sorted(missing)} not in {a.net}')
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(desaturate(raw, points, a.tau)))
    nsrc = Path(a.net).parent / 'norm.npz'
    if nsrc.exists():
        shutil.copyfile(nsrc, out.parent / 'norm.npz')
    print(f'wrote {out} (points {sorted(points)} last layer / {a.tau})')


if __name__ == '__main__':
    main()
