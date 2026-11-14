"""Sabotage a decision net for the M1 known-answer test (docs/rl-plan.md section 5, P1): add DELTA to one option's
output bias of one point. Other points and all other weights are copied value-for-value (their W.java entries stay
byte-identical); norm.npz beside the input is copied (the output bias is not touched by the layer-1 folding).

sabotage.py --net runs/m0/desat-b5/net.json --out runs/m1/sab3/net.json --bias 5:1:3      (+3 on BABY_ROLE R_SCOUT)
sabotage.py --probs DATA_DIR [DATA_DIR...] --net A.json [--net B.json ...] [--point 5] [--floor 0.005]
    mean floor-mixed probability of each option of the point under each net, over the rl_data.py p<point>.npz records
    (k picked, >= 2 options allowed; up to --max-rows), plus each net's total variation distance to the first net.
sabotage.py --self-test
"""
import argparse, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dpnet_io  # noqa: E402

NONE = 0xFFFF


def parse_bias(s):
    p, o, d = s.split(':')
    return int(p), int(o), float(d)


def sabotage(raw, biases):
    """A new net.json dict with DELTA added to the last-layer bias of (point, option) for each (point, option, delta)."""
    doc = dict(raw)
    pts = dict(raw['points'])
    for p, o, d in biases:
        layers = list(pts[str(p)])
        last = dict(layers[-1])
        b = list(last['b'])
        if not 0 <= o < len(b):
            raise ValueError(f'point {p} has {len(b)} options, not {o}')
        b[o] = b[o] + d
        last['b'] = b
        layers[-1] = last
        pts[str(p)] = layers
    doc['points'] = pts
    doc['sabotaged'] = dict(bias=[[p, o, d] for p, o, d in biases], **({'from': raw['sabotaged']} if 'sabotaged' in raw else {}))
    return doc


def load_records(dirs, point, n_in, n_out, max_rows=400000, seed=0):
    parts = [np.load(Path(d) / f'p{point}.npz') for d in dirs if (Path(d) / f'p{point}.npz').exists()]
    if not parts:
        raise SystemExit(f'no p{point}.npz in {dirs}')
    x = np.concatenate([np.pad(z['x'], ((0, 0), (0, max(0, n_in - z['x'].shape[1]))))[:, :n_in] for z in parts])
    k = np.concatenate([z['k'] for z in parts]).astype(np.int64)
    mask = np.concatenate([dpnet_io.fit_mask(z['mask'], len(z['k']), n_out) for z in parts])
    keep = (k != NONE) & (mask.sum(1) > 1)
    idx = np.nonzero(keep)[0]
    if len(idx) > max_rows:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_rows, replace=False))
    return x[idx], mask[idx]


def mean_probs(dirs, nets, point, floor, max_rows=400000):
    """{net path: dict(mean_probs, tv_to_first)} on the same records."""
    first = dpnet_io.load_net(nets[0])[point]
    x, mask = load_records(dirs, point, first[0]['w'].shape[1], first[-1]['w'].shape[0], max_rows)
    out, p0 = {}, None
    for n in nets:
        layers = dpnet_io.load_net(n)[point]
        pr = dpnet_io.probs(dpnet_io.forward(layers, x), mask, 1.0, floor).astype(np.float64)
        if p0 is None:
            p0 = pr
        out[n] = dict(records=int(len(x)), mean_probs=[round(float(v), 5) for v in pr.mean(0)],
                      tv_to_first=round(float(0.5 * np.abs(pr - p0).sum(1).mean()), 5))
    return out


def self_test():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        nets = {}
        for p, (nin, nout) in {5: (12, 5), 11: (10, 7)}.items():
            dims = [nin, 16, 16, nout]
            nets[p] = [dict(w=rng.normal(0, 1, (o, i)), b=rng.normal(0, 1, o)) for i, o in zip(dims, dims[1:])]
        src = td / 'a' / 'net.json'; src.parent.mkdir()
        dpnet_io.save_net(src, nets, dict(hidden=[16, 16]))
        dpnet_io.save_norm(src.parent / 'norm.npz', {5: (np.zeros(12, np.float32), np.ones(12, np.float32))})
        dst = td / 'b' / 'net.json'
        main(['--net', str(src), '--out', str(dst), '--bias', '5:1:3'])
        ra, rb = dpnet_io.load_raw(src), dpnet_io.load_raw(dst)
        assert ra['points']['11'] == rb['points']['11'], 'untouched point changed'
        assert ra['points']['5'][:-1] == rb['points']['5'][:-1], 'hidden layers changed'
        assert ra['points']['5'][-1]['w'] == rb['points']['5'][-1]['w'], 'last-layer weights changed'
        ba, bb = ra['points']['5'][-1]['b'], rb['points']['5'][-1]['b']
        assert [i for i in range(len(ba)) if ba[i] != bb[i]] == [1] and abs(bb[1] - ba[1] - 3) < 1e-12, 'bias not +3 on option 1'
        assert rb['sabotaged'] == {'bias': [[5, 1, 3.0]]}
        assert (dst.parent / 'norm.npz').exists(), 'norm.npz not copied'
        a, b = dpnet_io.load_net(src), dpnet_io.load_net(dst)
        x = rng.integers(0, 40, (400, 12))
        za, zb = dpnet_io.forward(a[5], x).astype(np.float64), dpnet_io.forward(b[5], x).astype(np.float64)
        d = zb - za
        assert np.abs(d[:, 1] - 3).max() < 1e-4 and np.abs(np.delete(d, 1, 1)).max() == 0, 'logit shift not exactly +3 on option 1'
        dd = td / 'data'; dd.mkdir()
        mask = rng.random((400, 5)) < 0.8; mask[:, :2] = True
        np.savez(dd / 'p5.npz', x=x.astype(np.uint16), k=np.zeros(400, np.int32), mask=mask)
        r = mean_probs([dd], [str(src), str(dst)], 5, 0.005)
        assert r[str(dst)]['mean_probs'][1] > r[str(src)]['mean_probs'][1] and r[str(src)]['tv_to_first'] == 0, r
        try:
            sabotage(ra, [(5, 9, 1.0)])
            raise AssertionError('out-of-range option accepted')
        except ValueError:
            pass
    print('sabotage self-test OK')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--net', action='append', default=[])
    ap.add_argument('--out', help='output net.json')
    ap.add_argument('--bias', action='append', default=[], help='POINT:OPTION:DELTA (repeatable)')
    ap.add_argument('--probs', nargs='+', help='rl_data.py data dirs: print mean option probabilities per net')
    ap.add_argument('--point', type=int, default=5)
    ap.add_argument('--floor', type=float, default=0.005)
    ap.add_argument('--max-rows', type=int, default=400000)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.probs:
        print(json.dumps(mean_probs(a.probs, a.net, a.point, a.floor, a.max_rows), indent=1))
        return
    if len(a.net) != 1 or not a.out or not a.bias:
        raise SystemExit('--net, --out and --bias required')
    raw = dpnet_io.load_raw(a.net[0])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sabotage(raw, [parse_bias(s) for s in a.bias])))
    nsrc = Path(a.net[0]).parent / 'norm.npz'
    if nsrc.exists():
        shutil.copyfile(nsrc, out.parent / 'norm.npz')
    print(f'wrote {out} (bias {a.bias})')


if __name__ == '__main__':
    main()
