"""Pass 3 tests (numpy only, no games):

  * dp_points: the Java catalogue matches the pass 3 design table;
  * widen: old logits bit-identical on the old columns / rows, new rows at the calibrated bias, R_KEEP dominant,
    active table, new points (zdb-hard picks option 0 at prob ~1, value points uniform), norm padding;
  * build_dpnet.compat: refuses oversize nets, reports reach.

    python3 tests/test_widen.py      (also runs under pytest)
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dp_points, dpnet_io  # noqa: E402

WANT = {1: (72, 76), 2: (39, 74), 3: (39, 74), 4: (13, 32), 5: (6, 112), 6: (15, 53), 7: (40, 45), 8: (16, 36),
        9: (32, 30), 10: (37, 48), 11: (22, 41), 12: (17, 38), 13: (30, 38), 14: (81, 94), 15: (32, 22),
        16: (9, 24)}


def test_dp_points():
    o, f = dp_points.option_counts(), dp_points.feature_counts()
    for p, (no, nf) in WANT.items():
        assert f.get(p) == nf, (p, f.get(p), nf)
        if no is not None:
            assert o.get(p) == no, (p, o.get(p), no)
    assert set(dp_points.SECONDARY) == {6, 7, 9, 15}


def toy_net(rng, n_in, n_out, hidden=(8, 8)):
    dims = [n_in, *hidden, n_out]
    return [dpnet_io.layer_json(dict(w=rng.normal(0, 1, (o, i)).astype(np.float32), b=rng.normal(0, 1, o).astype(np.float32)))
            for i, o in zip(dims, dims[1:])]


def test_widen():
    rng = np.random.default_rng(1)
    raw = {'version': 'dpnet-v1', 'hidden': [8, 8], 'points': {'1': toy_net(rng, 10, 56), '5': toy_net(rng, 32, 5),
                                                               '4': toy_net(rng, 32, 9)}}
    opts = {1: 72, 5: 6, 4: 9, 8: 16, 9: 32}
    feats = {1: 12, 5: 40, 4: 32, 8: 6, 9: 5}
    calib = {1: rng.integers(0, 5, (300, 12)), 5: np.c_[rng.integers(0, 2, 300), rng.integers(0, 5, (300, 39))],
             8: rng.integers(0, 5, (300, 6))}
    norm = {1: (np.zeros(10, np.float32), np.ones(10, np.float32))}
    doc, nrm, info = dpnet_io.widen(raw, opts, feats, calib=calib, floor=0.005, init_new='zdb-hard', norm=norm)
    old = {int(p): dpnet_io.layers_from_raw(v) for p, v in raw["points"].items()}
    new = {int(p): dpnet_io.layers_from_raw(v) for p, v in doc['points'].items()}
    # old points unchanged where nothing to widen; widened ones replay exactly on the old outputs
    assert doc['points']['4'] == raw['points']['4']
    for p, n_in, n_out in ((1, 10, 56), (5, 32, 5)):
        X = calib[p]
        z0 = dpnet_io.forward(old[p], X[:, :n_in])
        z1 = dpnet_io.forward(new[p], X)
        assert np.array_equal(z0, z1[:, :n_out]), p
        assert new[p][0]['w'].shape == (8, feats[p]) and new[p][-1]['w'].shape == (opts[p], 8)
    z = dpnet_io.forward(old[1], calib[1][:, :10]).astype(np.float64)
    b = float(np.median(z.max(1)) - np.log(10 * 72 / 0.005))
    assert np.allclose(new[1][-1]['b'][56:], b, atol=1e-4) and not new[1][-1]['w'][56:].any()
    zr = dpnet_io.forward(new[5], calib[5])
    rev = calib[5][:, 0] == 1
    assert (zr[rev].argmax(1) == 5).all()
    assert doc['active'] == {'1': 56, '5': 5}
    assert set(doc['points']) == {'1', '4', '5', '8', '9'}
    z8 = dpnet_io.forward(new[8], calib[8])
    p8 = dpnet_io.probs(z8, None, 1.0, 0.0)
    assert (p8[:, 0] > 0.999).all()
    z9 = dpnet_io.forward(new[9], rng.integers(0, 5, (20, 5)))
    assert np.allclose(z9, 0)
    assert len(nrm[1][0]) == 12 and len(nrm[8][0]) == 6
    # active full: no table
    doc2, _, _ = dpnet_io.widen(raw, opts, feats, active='full')
    assert 'active' not in doc2


def test_compat():
    import build_dpnet
    rng = np.random.default_rng(2)
    o, f = dp_points.option_counts(), dp_points.feature_counts()
    net = {'points': {'4': toy_net(rng, f[4], o[4]), '1': toy_net(rng, 10, 56)}, 'active': {'1': 56}}
    r = build_dpnet.compat(net, {1, 4})
    assert r[1] == (56, 56, 72) and r[4] == (o[4], o[4], o[4])
    bad = {'points': {'4': toy_net(rng, f[4], o[4] + 1)}}
    try:
        build_dpnet.compat(bad, {4})
        raise AssertionError('oversize net accepted')
    except SystemExit:
        pass


if __name__ == '__main__':
    test_dp_points(); test_widen(); test_compat()
    print('ok')
