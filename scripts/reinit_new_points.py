#!/usr/bin/env python3
"""Rescale the first layer of zero-output decision-point nets so their hidden activations are O(1).

widen_dpnet.py --init-new calibrated gives a new point (8, 9, 14, 15, 16) random hidden layers acting on RAW features
(shared-array values, coordinates: up to ~65k) and an all-zero last layer, so the policy is the last-layer bias. The
function is fine, but hidden activations are ~1e3-1e4, so one Adam step on the last layer moves the logits by hundreds
(rl-m2 iteration 0: KL 6-9 after one step on points 14 and 16). Because the last layer is zero, the first layer can be
rescaled freely without changing any logit: W1 <- W1 / sd, b1 <- b1 - (W1 / sd) @ mu, with (mu, sd) from recorded
features of that point (rl_data p<point>.npz). The policy is unchanged exactly (logits = 0 . h + b).

  scripts/reinit_new_points.py --net IN.json --data DIR [DIR ...] --out OUT.json [--points 8,9,14,15,16]
  scripts/reinit_new_points.py --self-test
Only points whose last layer is all zero are touched; every other point is copied value for value.
"""
import argparse, json, sys
import numpy as np


def stats(dirs, p, n_in):
    xs = []
    for d in dirs:
        try:
            xs.append(np.load(f'{d}/p{p}.npz')['x'].astype(np.float64))
        except FileNotFoundError:
            pass
    if not xs:
        return None
    nf = max(x.shape[1] for x in xs)
    x = np.concatenate([np.pad(x, ((0, 0), (0, nf - x.shape[1]))) for x in xs])
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-3, 1.0, sd)
    mu = np.pad(mu, (0, max(0, n_in - len(mu))))[:n_in]
    sd = np.pad(sd, (0, max(0, n_in - len(sd))), constant_values=1.0)[:n_in]
    return mu, sd


def rescale(layers, mu, sd):
    """New first layer (w, b) acting on raw x that equals the old first layer acting on (x - mu) / sd."""
    l0 = layers[0]
    W = np.asarray(l0['w'], np.float64).reshape(l0['n_out'], l0['n_in'])
    b = np.asarray(l0['b'], np.float64)
    Wn = W / sd[None, :]
    bn = b - Wn @ mu
    return dict(l0, w=Wn.ravel().tolist(), b=bn.tolist())


def zero_last(layers):
    return not np.any(np.asarray(layers[-1]['w'], np.float64))


def process(net, dirs, points):
    pts = net['points'] if 'points' in net else net
    done = {}
    for p in points:
        key = str(p)
        if key not in pts or not isinstance(pts[key], list) or not zero_last(pts[key]):
            continue
        s = stats(dirs, p, pts[key][0]['n_in'])
        if s is None:
            continue
        pts[key] = [rescale(pts[key], *s)] + pts[key][1:]
        done[p] = int(len(s[0]))
    return done


def forward(layers, x):
    for i, l in enumerate(layers):
        W = np.asarray(l['w'], np.float64).reshape(l['n_out'], l['n_in'])
        x = x @ W.T + np.asarray(l['b'], np.float64)
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    return x


def self_test():
    import tempfile, os
    rng = np.random.default_rng(0)
    n_in, h, n_out = 6, 8, 5
    def lay(a, b, w):
        return dict(n_in=a, n_out=b, w=w.ravel().tolist(), b=np.zeros(b).tolist())
    L = [lay(n_in, h, rng.uniform(-.25, .25, (h, n_in))), lay(h, h, rng.uniform(-.3, .3, (h, h))),
         dict(n_in=h, n_out=n_out, w=[0.0] * (h * n_out), b=[0.0, -12, -12, -3, -12])]
    other = [lay(n_in, h, rng.normal(size=(h, n_in))), lay(h, h, rng.normal(size=(h, h))), lay(h, n_out, rng.normal(size=(n_out, h)))]
    net = dict(version=1, points={'14': L, '5': other})
    x = rng.uniform(0, 60000, (500, n_in))
    with tempfile.TemporaryDirectory() as t:
        np.savez(os.path.join(t, 'p14.npz'), x=x.astype(np.uint16))
        before = forward(L, x)
        hb = np.maximum(x @ np.asarray(L[0]['w']).reshape(h, n_in).T, 0)
        out = json.loads(json.dumps(net))
        done = process(out, [t], [14, 5])
    assert done == {14: n_in}, done
    after = forward(out['points']['14'], x)
    assert np.array_equal(before, after), 'logits changed'
    ha = np.maximum(x.astype(np.uint16).astype(np.float64) @ np.asarray(out['points']['14'][0]['w']).reshape(h, n_in).T
                    + np.asarray(out['points']['14'][0]['b']), 0)
    assert abs(ha).mean() < 5 and abs(hb).mean() > 100, (abs(ha).mean(), abs(hb).mean())
    assert out['points']['5'] == net['points']['5'], 'non-zero-output point touched'
    print('reinit_new_points self-test OK (hidden |mean| %.3g -> %.3g, logits identical)' % (abs(hb).mean(), abs(ha).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--net'); ap.add_argument('--out'); ap.add_argument('--data', nargs='+', default=[])
    ap.add_argument('--points', default='8,9,14,15,16')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    net = json.load(open(a.net))
    done = process(net, a.data, [int(p) for p in a.points.split(',') if p])
    json.dump(net, open(a.out, 'w'))
    print(f'wrote {a.out}: rescaled first layer of points {sorted(done)} (zero last layer, data from {len(a.data)} dirs)')


if __name__ == '__main__':
    sys.exit(main())
