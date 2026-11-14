"""Synthetic checks of scripts/rl_train.py and scripts/dpnet_io.py (numpy + torch, CPU; no games needed).

    python3 tests/test_rl_train.py [--work DIR] [--net NET.json]     (also runs under pytest)

1. dpnet_io: unfold/fold round trip, exact forward vs BLAS, probs (numpy) vs probs (torch), probs sum to 1.
2. lr 0: the exported net.json is byte-identical to the input (a synthetic net; plus --net, e.g. a real net-v1).
3. Planted reward (--fake-reward 6:0, 2 iterations, each re-sampled from the newly exported net): P(option 0) rises on
   point 6 and the other point's layers are byte-identical.
4. Outcome reward through V and the paired baseline: z favours team-games that chose option 2 more often on point 5;
   P(option 2) rises.
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dpnet_io  # noqa: E402

FLOOR = 0.005
NOPT = {5: 5, 6: 13}
NF = {5: 12, 6: 10}


def make_net(path, seed=0):
    rng = np.random.default_rng(seed)
    nets, norm = {}, {}
    for p, n_out in NOPT.items():
        dims = [NF[p], 16, 16, n_out]
        layers = [dict(w=rng.normal(0, 1 / np.sqrt(i), (o, i)), b=rng.normal(0, 0.1, o)) for i, o in zip(dims, dims[1:])]
        mu = rng.uniform(0, 20, NF[p]).astype(np.float32); sd = rng.uniform(1, 5, NF[p]).astype(np.float32)
        nets[p] = dpnet_io.fold(layers, mu, sd); norm[p] = (mu, sd)
    dpnet_io.save_net(path, nets, dict(hidden=[16, 16]))
    dpnet_io.save_norm(Path(path).parent / 'norm.npz', norm)


def sample_data(net_path, out_dir, seed, games=40, per=150, reward=None, points=(5, 6)):
    """Play nothing: per (game, team) draw records from the net exactly as DNet.sample would, quantise -ln p as the
    trailer does, and give the team-game an outcome. Games come in seat-swapped pairs (pair = game // 2)."""
    rng = np.random.default_rng(seed)
    nets = {p: v for p, v in dpnet_io.load_net(net_path).items() if p in points}
    NF = {p: v[0]['w'].shape[1] for p, v in nets.items()}
    NOPT = {p: v[-1]['w'].shape[0] for p, v in nets.items()}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    recs = {p: [] for p in nets}
    trace_rows = []
    for g in range(games):
        zg = None
        team_recs = {}
        for team in (0, 1):
            for p in nets:
                N = per
                x = rng.integers(0, 40, (N, NF[p])).astype(np.int32)
                mask = rng.random((N, NOPT[p])) < 0.8
                mask[:, 0] = True
                pr = dpnet_io.probs(dpnet_io.forward(nets[p], x), mask, 1.0, FLOOR).astype(np.float64)
                k = np.array([rng.choice(NOPT[p], p=row / row.sum()) for row in pr])
                q = np.minimum(65535, np.round(-np.log(pr[np.arange(N), k]) * 4096))
                rnd = np.sort(rng.integers(1, 2000, N))
                team_recs[(team, p)] = dict(x=x.astype(np.uint16), k=k.astype(np.int32), logq=(-q / 4096).astype(np.float32),
                                            mask=mask, flags=(rng.random(N) < 0.1).astype(np.uint8), round=rnd.astype(np.int32))
        # outcome: random, or (reward) the team whose point-5 records chose option 2 more often wins
        if reward is None:
            w = int(rng.integers(0, 2))
        else:
            f = [np.mean(team_recs[(t, 5)]['k'] == reward) for t in (0, 1)]
            w = int(f[1] > f[0]) if f[0] != f[1] else int(rng.integers(0, 2))
        for team in (0, 1):
            z = 1.0 if team == w else -1.0
            bins = np.arange(0, 2000, 10)
            tr = np.stack([bins / 2000, rng.normal(0.3 * z, 1, len(bins))], 1).astype(np.float32)
            trace_rows.append(dict(game=np.full(len(bins), g), team=np.full(len(bins), team), round=bins,
                                   pair=np.full(len(bins), g // 2), z=np.full(len(bins), z, np.float32), trace=tr))
            for p in nets:
                r = team_recs[(team, p)]
                r.update(team=np.full(per, team, np.int32), game=np.full(per, g, np.int64), pair=np.full(per, g // 2, np.int64),
                         z=np.full(per, z, np.float32), trace=tr[r['round'] // 10])
                recs[p].append(r)
    for p, parts in recs.items():
        np.savez(out_dir / f'p{p}.npz', **{k: np.concatenate([r[k] for r in parts]) for k in parts[0]})
    np.savez(out_dir / 'trace.npz', **{k: np.concatenate([r[k] for r in trace_rows]) for k in trace_rows[0]})


def run_train(*args):
    cmd = [sys.executable, str(ROOT / 'scripts/rl_train.py'), '--floor', str(FLOOR), '--threads', '2', *map(str, args)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    return r.returncode


def check_io():
    import torch
    rng = np.random.default_rng(3)
    layers = [dict(w=rng.normal(0, 0.3, (16, 12)), b=rng.normal(0, 0.1, 16)), dict(w=rng.normal(0, 0.3, (5, 16)), b=rng.normal(0, 0.1, 5))]
    mu, sd = rng.uniform(0, 9, 12).astype(np.float32), rng.uniform(1, 3, 12).astype(np.float32)
    back = dpnet_io.unfold(dpnet_io.fold(layers, mu, sd), mu, sd)
    assert all(np.allclose(a['w'], b['w'], atol=1e-12) and np.allclose(a['b'], b['b'], atol=1e-10) for a, b in zip(layers, back))
    f32 = [dict(w=l['w'].astype(np.float32), b=l['b'].astype(np.float32)) for l in dpnet_io.fold(layers, mu, sd)]
    X = rng.integers(0, 30, (500, 12))
    ze, zf = dpnet_io.forward(f32, X), dpnet_io.forward(f32, X, exact=False)
    assert np.abs(ze - zf).max() < 1e-3, np.abs(ze - zf).max()
    mask = rng.random((500, 5)) < 0.6; mask[:, 1] = True
    pn = dpnet_io.probs(ze, mask, 1.0, FLOOR)
    pt = dpnet_io.probs(torch.tensor(ze), torch.tensor(mask), 1.0, FLOOR).numpy()
    assert np.abs(pn - pt).max() < 1e-6 and np.allclose(pn.sum(1), 1, atol=1e-5) and (pn[~mask] == 0).all()
    assert (pn[mask] >= FLOOR / 5 - 1e-9).all()
    print('dpnet_io ok: fold round trip, forward exact vs BLAS max', float(np.abs(ze - zf).max()))


def main(work=None, real_net=None):
    work = Path(work or tempfile.mkdtemp(prefix='rltest-'))
    work.mkdir(parents=True, exist_ok=True)
    check_io()
    # 2. identity at lr 0
    n0 = work / 'nets/0'; n0.mkdir(parents=True, exist_ok=True)
    make_net(n0 / 'net.json')
    sample_data(n0 / 'net.json', work / 'iter-0000/data', seed=1)
    assert run_train('--net', n0 / 'net.json', '--data', work / 'iter-0000/data', '--out', work / 'lr0', '--points', '5,6',
                     '--lr', 0) == 0
    assert (work / 'lr0/net.json').read_bytes() == (n0 / 'net.json').read_bytes(), 'lr 0 export differs'
    m = json.loads((work / 'lr0/metrics.json').read_text())
    for p, mt in m['points'].items():
        assert mt['parity_median'] < 1e-3 and mt['ratio_start_maxabslog'] < 2e-3, (p, mt['parity_median'])
    print('lr 0 identity ok; value', m['value'])
    if real_net:
        rn = work / 'real'; rn.mkdir(exist_ok=True)
        assert run_train('--net', real_net, '--data', work / 'iter-0000/data', '--out', rn, '--points', '5,6', '--lr', 0) == 0
        assert (rn / 'net.json').read_bytes() == Path(real_net).read_bytes()
        text = Path(real_net).read_text()
        assert json.dumps(json.loads(text)) == text, 'net.json does not round-trip through json'
        rf = work / 'realfake'
        sample_data(real_net, work / 'realdata', seed=5, games=10, per=100)
        assert run_train('--net', real_net, '--data', work / 'realdata', '--out', rf, '--fake-reward', '6:0',
                         '--lr', 3e-3) == 0
        ra, rb = dpnet_io.load_raw(real_net)['points'], dpnet_io.load_raw(rf / 'net.json')['points']
        assert ra['6'] != rb['6'] and all(ra[q] == rb[q] for q in ra if q != '6')
        print('lr 0 identity ok on', real_net)
    # 3. planted reward, 2 iterations, data re-sampled from each exported net
    prev = n0
    probs = []
    for it in (1, 2):
        d = work / f'fake-{it}/data'
        sample_data(prev / 'net.json', d, seed=10 + it)
        nxt = work / f'nets/fake{it}'
        assert run_train('--net', prev / 'net.json', '--anchor', n0 / 'net.json', '--data', d, '--out', nxt,
                         '--fake-reward', '6:0', '--lr', 3e-3) == 0
        mt = json.loads((nxt / 'metrics.json').read_text())['points']['6']
        probs.append((mt['fake_option_prob_start'], mt['fake_option_prob_end']))
        assert dpnet_io.load_raw(nxt / 'net.json')['points']['5'] == dpnet_io.load_raw(n0 / 'net.json')['points']['5']
        prev = nxt
    print('planted reward P(option 0) per iteration (start, end):', probs)
    assert probs[0][1] > probs[0][0] and probs[1][1] > probs[1][0] and probs[1][0] > probs[0][0] + 1e-4
    # the shift shows in the records sampled from the exported net
    a, b = np.load(work / 'fake-1/data/p6.npz'), np.load(work / 'fake-2/data/p6.npz')
    print('sampled frequency of option 0:', float((a['k'] == 0).mean()), '->', float((b['k'] == 0).mean()))
    # 4. outcome reward through V + paired baseline
    d = work / 'reward/data'
    sample_data(n0 / 'net.json', d, seed=99, games=80, reward=2)
    assert run_train('--net', n0 / 'net.json', '--data', d, '--out', work / 'reward/net', '--points', '5', '--lr', 3e-3,
                     '--epochs', 4) == 0
    mt = json.loads((work / 'reward/net/metrics.json').read_text())['points']['5']
    print('reward on option 2: mean P', mt['mean_probs_start'][2], '->', mt['mean_probs_end'][2], 'snr', mt.get('grad_snr_cos'))
    assert mt['mean_probs_end'][2] > mt['mean_probs_start'][2]
    print('ALL OK', work)


def test_rl_train(tmp_path):
    main(tmp_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=None)
    ap.add_argument('--net', default=None, help='a real net.json for the lr-0 identity check')
    a = ap.parse_args()
    main(a.work, a.net)
