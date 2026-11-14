"""ES policy offsets over a fixed base net (docs/rl-es-plan.md section 1): phi = {point: {"b": [offset per option],
"logT": x}}. A point in phi gets last-layer logits (W h + b_base) * exp(-logT) + b_offset: the scale first (as
desaturate.py), then the bias (as sabotage.py). Points not in phi, hidden layers, zero offsets and logT = 0 are copied
value-for-value, so their W.java entries stay byte-identical.

    from es_net import load_phi, save_phi, zero_phi, apply_phi, blocks, coords, calibrate_sigma
    blocks(base_raw, [11]) -> [Block(name='11', point=11, coords=['11:b0', ..., '11:b21', '11:logT'])]

Coordinates are named '<point>:b<option>' and '<point>:logT'; a flat {coord: value} dict maps onto phi with
phi_get / phi_add. Equivalent options (an equiv file, {point: [[o1, o2, ...], ...]}) share one coordinate: the
coordinate is named after the first option and moves all of them together.

es_net.py --base NET.json --phi PHI.json --out OUT/net.json        write the centre net (norm.npz copied)
es_net.py --base NET.json --blocks 11 [--exclude 11:3] [--equiv F]  print the blocks
es_net.py --calibrate DATA_DIR... --base NET.json --phi PHI.json --points 5,7 [--target 0.25] [--cap 15] [--floor 0.005]
es_net.py --diff NET_A NET_B                                    points whose packed W.java entry differs (M1 criterion 3)
es_net.py --self-test
"""
import argparse, copy, json, math, shutil, sys, tempfile
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

VALUE_POINTS = (9, 15)       # BABY_VAL, KING_VAL: a global bias over digits is meaningless
SUB_BLOCK = 24               # max options per sub-block ...
SPLIT_OVER = 48              # ... for points with more coordinates than this (KING_COMMS 81, BABY_MOVE 72)
Block = namedtuple('Block', 'name point coords')


# ---- phi
def load_phi(path):
    return {int(p): {'b': [float(v) for v in d['b']], 'logT': float(d.get('logT', 0.0))}
            for p, d in json.loads(Path(path).read_text()).get('phi', {}).items()}


def save_phi(path, phi, **meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(dict(meta, phi={str(p): d for p, d in sorted(phi.items())}), indent=1))
    tmp.replace(path)


def n_options(base_raw, p):
    """Live options of point p: the net's outputs, or its "active" count (widen_dpnet.py --active old)."""
    n = base_raw['points'][str(p)][-1]['n_out']
    a = int((base_raw.get('active') or {}).get(str(p), 0))
    return a if 0 < a < n else n


def zero_phi(base_raw, points):
    return {int(p): {'b': [0.0] * base_raw['points'][str(p)][-1]['n_out'], 'logT': 0.0} for p in points}


def apply_phi(base_raw, phi):
    """A new net.json dict: the base with phi applied (points not in phi untouched, objects shared)."""
    doc = dict(base_raw)
    pts = dict(base_raw['points'])
    for p, d in phi.items():
        layers = list(pts[str(p)])
        last = dict(layers[-1])
        if len(d['b']) != last['n_out']:
            raise ValueError(f"point {p}: phi has {len(d['b'])} offsets, the net {last['n_out']} outputs")
        lt = float(d.get('logT', 0.0))
        if lt != 0.0:
            s = math.exp(-lt)
            last['w'] = [v * s for v in last['w']]
            last['b'] = [v * s for v in last['b']]
        if any(o != 0.0 for o in d['b']):
            last['b'] = [v + o if o != 0.0 else v for v, o in zip(last['b'], d['b'])]
        layers[-1] = last
        pts[str(p)] = layers
    doc['points'] = pts
    doc['es_phi'] = {str(p): d for p, d in sorted(phi.items()) if d.get('logT') or any(d['b'])}
    if not doc['es_phi']:
        del doc['es_phi']
    return doc


# ---- coordinates
def coord_point(c):
    return int(c.split(':')[0])


def coord_options(c, equiv=None):
    """Options a coordinate moves (one, or an equivalence class); [] for logT."""
    p, k = c.split(':')
    if k == 'logT':
        return []
    o = int(k[1:])
    for cls in (equiv or {}).get(int(p), []):
        if o in cls:
            return sorted(cls)
    return [o]


def phi_get(phi, c):
    p, k = coord_point(c), c.split(':')[1]
    return phi[p]['logT'] if k == 'logT' else phi[p]['b'][int(k[1:])]


def phi_add(phi, delta, equiv=None):
    """A copy of phi with {coord: value} added (an equivalence coordinate adds to all its options)."""
    out = copy.deepcopy(phi)
    for c, v in delta.items():
        p = coord_point(c)
        if c.endswith(':logT'):
            out[p]['logT'] += v
        else:
            for o in coord_options(c, equiv):
                out[p]['b'][o] += v
    return out


def load_equiv(path):
    if not path:
        return {}
    return {int(p): [sorted(int(o) for o in cls) for cls in v] for p, v in json.loads(Path(path).read_text()).items()}


def parse_exclude(s):
    """'11:3,11:4' -> {11: {3, 4}}"""
    out = {}
    for x in (s or '').split(',') if isinstance(s, str) else (s or []):
        if x:
            p, o = str(x).split(':')
            out.setdefault(int(p), set()).add(int(o))
    return out


def blocks(base_raw, points, exclude=None, equiv=None, logT=True, sub=SUB_BLOCK, split_over=SPLIT_OVER):
    """[Block] for the points (value points refused). Options beyond the active count, excluded options and all but the
    first of each equivalence class get no coordinate. A point with more than `split_over` option coordinates is split into
    sub-blocks of at most `sub` contiguous options (the logT coordinate goes with the first)."""
    exclude = exclude or {}
    equiv = equiv or {}
    out = []
    for p in points:
        p = int(p)
        if p in VALUE_POINTS:
            raise SystemExit(f'point {p} is a value-digit point: no ES block')
        n = n_options(base_raw, p)
        follow = {o for cls in equiv.get(p, []) for o in cls[1:]}
        opts = [o for o in range(n) if o not in exclude.get(p, set()) and o not in follow]
        chunks = ([opts[i:i + sub] for i in range(0, len(opts), sub)] if len(opts) > split_over else [opts]) or [[]]
        for ci, ch in enumerate(chunks):
            cs = [f'{p}:b{o}' for o in ch] + ([f'{p}:logT'] if logT and ci == 0 else [])
            out.append(Block(f'{p}' if len(chunks) == 1 else f'{p}.{ci}', p, cs))
    return out


# ---- calibration (M2): sigma_j = offset at which mean |dP_j| over own records reaches target
def calibrate_sigma(base_raw, phi, rec_dirs, target=0.25, cap=15.0, lo=3.0, floor=0.005, logT_sigma=0.3,
                    max_rows=100000, coords_=None):
    import numpy as np
    import dpnet_io
    from sabotage import load_records
    centre = apply_phi(base_raw, phi)
    out = {}
    for p in sorted({coord_point(c) for c in coords_} if coords_ else phi):
        layers = dpnet_io.layers_from_raw(centre['points'][str(p)])
        x, mask = load_records(rec_dirs, p, layers[0]['w'].shape[1], layers[-1]['w'].shape[0], max_rows)
        z = dpnet_io.forward(layers, x, exact=False).astype(np.float64)
        p0 = dpnet_io.probs(z, mask, 1.0, floor).astype(np.float64)
        cs = [c for c in (coords_ or []) if coord_point(c) == p] or [f'{p}:b{o}' for o in range(n_options(base_raw, p))] + [f'{p}:logT']
        for c in cs:
            if c.endswith(':logT'):
                out[c] = logT_sigma
                continue
            o = int(c.split(':b')[1])
            rows = mask[:, o]
            if not rows.any():
                out[c] = cap
                continue

            def f(d):
                zz = z[rows].copy(); zz[:, o] += d
                pp = dpnet_io.probs(zz, mask[rows], 1.0, floor).astype(np.float64)[:, o]
                zm = z[rows].copy(); zm[:, o] -= d
                pm = dpnet_io.probs(zm, mask[rows], 1.0, floor).astype(np.float64)[:, o]
                return 0.5 * (np.abs(pp - p0[rows, o]).mean() + np.abs(pm - p0[rows, o]).mean())
            if f(cap) < target:
                out[c] = cap
                continue
            a, b = 0.0, cap
            for _ in range(30):
                m = 0.5 * (a + b)
                a, b = (m, b) if f(m) < target else (a, m)
            out[c] = float(min(cap, max(lo, b)))
    return out


# ---- self-test
def self_test():
    import numpy as np
    import dpnet_io, build_dpnet
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        nets = {}
        for p, (nin, nout) in {5: (12, 6), 11: (10, 22), 14: (9, 81)}.items():
            dims = [nin, 16, 16, nout]
            nets[p] = [dict(w=rng.normal(0, 2, (o, i)), b=rng.normal(0, 3, o)) for i, o in zip(dims, dims[1:])]
        src = td / 'net.json'
        dpnet_io.save_net(src, nets, dict(hidden=[16, 16]))
        raw = dpnet_io.load_raw(src)
        raw['points']['11'][-1]['b'][3] = -0.0   # a signed zero must survive phi = 0
        # 1. phi = 0: every point's packed W.java entry byte-identical
        z = apply_phi(raw, zero_phi(raw, [5, 11, 14]))
        assert 'es_phi' not in z
        for p in raw['points']:
            assert build_dpnet.pack(z['points'][p]) == build_dpnet.pack(raw['points'][p]), f'phi=0 changed point {p}'
            assert json.dumps(z['points'][p]) == json.dumps(raw['points'][p]), f'phi=0 changed point {p} json'
        # 2. bias + logT: exact logit relation, untouched points/hidden layers/other options identical
        phi = zero_phi(raw, [11])
        phi[11]['b'][21] = 7.5; phi[11]['logT'] = 0.4
        n = apply_phi(raw, phi)
        assert n['points']['5'] is raw['points']['5'] and n['points']['14'] is raw['points']['14']
        assert n['points']['11'][:-1] == raw['points']['11'][:-1]
        a, b = dpnet_io.layers_from_raw(raw['points']['11']), dpnet_io.layers_from_raw(n['points']['11'])
        x = rng.integers(0, 30, (300, 10))
        za = dpnet_io.forward(a, x, exact=False).astype(np.float64); zb = dpnet_io.forward(b, x, exact=False).astype(np.float64)
        want = za * math.exp(-0.4); want[:, 21] += 7.5
        assert np.abs(zb - want).max() < 1e-4 * max(1, np.abs(za).max()), 'logits != scale then bias'
        phi2 = zero_phi(raw, [11]); phi2[11]['b'][21] = 7.5
        n2 = apply_phi(raw, phi2)
        bb, b0 = n2['points']['11'][-1]['b'], raw['points']['11'][-1]['b']
        assert [i for i in range(22) if json.dumps(bb[i]) != json.dumps(b0[i])] == [21], 'other biases changed'
        assert n2['points']['11'][-1]['w'] == raw['points']['11'][-1]['w']
        # 3. phi round trip, phi_add with equivalence
        save_phi(td / 'phi.json', phi, iter=0)
        assert load_phi(td / 'phi.json') == phi
        eq = {11: [[2, 5]]}
        ph = phi_add(zero_phi(raw, [11]), {'11:b2': 1.0, '11:logT': 0.1}, eq)
        assert ph[11]['b'][2] == ph[11]['b'][5] == 1.0 and ph[11]['logT'] == 0.1 and sum(ph[11]['b']) == 2.0
        # 4. blocks: value points refused, active/exclude/equiv, KING_COMMS sub-blocks <= 24
        bl = blocks(raw, [11], exclude={11: {0}}, equiv=eq)
        assert len(bl) == 1 and len(bl[0].coords) == 22 - 1 - 1 + 1 and '11:b5' not in bl[0].coords and '11:b0' not in bl[0].coords
        bl14 = blocks(raw, [14])
        assert len(bl14) == 4 and all(len(b.coords) <= 25 for b in bl14) and sum(len(b.coords) for b in bl14) == 82
        raw2 = dict(raw, active={'11': 18})
        assert len(blocks(raw2, [11])[0].coords) == 19
        try:
            blocks(raw, [15] if '15' in raw['points'] else [9])
            raise AssertionError('value point accepted')
        except SystemExit:
            pass
        # 5. calibration: sigma within [lo, cap], larger for a deeply buried option
        dd = td / 'rec'; dd.mkdir()
        mask = np.ones((300, 22), bool)
        np.savez(dd / 'p11.npz', x=x.astype(np.uint16), k=np.zeros(300, np.int32), mask=mask)
        sg = calibrate_sigma(raw, zero_phi(raw, [11]), [dd], coords_=['11:b0', '11:b21', '11:logT'])
        assert all(3.0 <= v <= 15.0 for c, v in sg.items() if c != '11:logT') and sg['11:logT'] == 0.3, sg
    print('es_net self-test OK')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base')
    ap.add_argument('--phi')
    ap.add_argument('--out')
    ap.add_argument('--blocks', default='')
    ap.add_argument('--exclude', default='')
    ap.add_argument('--equiv', default='')
    ap.add_argument('--calibrate', nargs='+')
    ap.add_argument('--points', default='')
    ap.add_argument('--target', type=float, default=0.25)
    ap.add_argument('--cap', type=float, default=15.0)
    ap.add_argument('--floor', type=float, default=0.005)
    ap.add_argument('--diff', nargs=2)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.diff:
        import build_dpnet
        ra, rb = (json.loads(Path(f).read_text()) for f in a.diff)
        pts = sorted(set(ra['points']) | set(rb['points']), key=int)
        diff = [p for p in pts if p not in ra['points'] or p not in rb['points']
                or build_dpnet.pack(ra['points'][p]) != build_dpnet.pack(rb['points'][p])]
        print(json.dumps(dict(points=len(pts), differ=diff, active_equal=ra.get('active') == rb.get('active'))))
        return
    raw = json.loads(Path(a.base).read_text())
    if a.blocks:
        for b in blocks(raw, [int(p) for p in a.blocks.split(',')], parse_exclude(a.exclude), load_equiv(a.equiv)):
            print(b.name, len(b.coords), ' '.join(b.coords))
        return
    phi = load_phi(a.phi) if a.phi else {}
    if a.calibrate:
        pts = [int(p) for p in a.points.split(',') if p]
        phi = {**zero_phi(raw, pts), **phi}
        print(json.dumps(calibrate_sigma(raw, phi, a.calibrate, a.target, a.cap, floor=a.floor,
                                         coords_=[c for b in blocks(raw, pts) for c in b.coords]), indent=1))
        return
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(apply_phi(raw, phi)))
    if (Path(a.base).parent / 'norm.npz').exists():
        shutil.copyfile(Path(a.base).parent / 'norm.npz', out.parent / 'norm.npz')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
