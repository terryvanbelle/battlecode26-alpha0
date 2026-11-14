"""Synthetic checks of scripts/rl_train.py and scripts/dpnet_io.py (numpy + torch, CPU; no games needed).

    python3 tests/test_rl_train.py [--work DIR] [--net NET.json]     (also runs under pytest)

1. dpnet_io: unfold/fold round trip, exact forward vs BLAS, probs (numpy) vs probs (torch), probs sum to 1.
2. lr 0: the exported net.json is byte-identical to the input (a synthetic net; plus --net, e.g. a real net-v1).
3. Planted reward (--fake-reward 6:0, 2 iterations, each re-sampled from the newly exported net): P(option 0) rises on
   point 6 and the other point's layers are byte-identical.
4. Outcome reward through V and the paired baseline: z favours team-games that chose option 2 more often on point 5;
   P(option 2) rises.
5. --no-kl: a far anchor neither refuses the export nor changes it.
6. rl_critic: GAE (lambda 1 = z - V, lambda 0 = one-step TD), record -> bin mapping, the critic's antisymmetry, AUC.
7. Critic + GAE in rl_train (--critic trace|rich --gae-lambda 0.8): each option-2 pick on point 5 adds a baby to the
   team's next trace bin and the outcome is a noisy function of the final babies difference; P(option 2) rises.
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dpnet_io, rl_critic  # noqa: E402

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


def sample_gae_data(net_path, out_dir, seed, games=40, bins=40, per_bin=6, opt=2):
    """Point-5 records whose option `opt` adds a baby to the team's trace from the next 10-round bin on; z is a noisy
    function of the final babies difference. 20-column traces (rl_train.TRACE_COLS order), seat-swapped pairs."""
    rng = np.random.default_rng(seed)
    net = dpnet_io.load_net(net_path)[5]
    nf, no = net[0]['w'].shape[1], net[-1]['w'].shape[0]
    recs, trace_rows = [], []
    for g in range(games):
        babies, team_recs = {}, {}
        for team in (0, 1):
            b = np.zeros(bins + 1)
            parts = []
            for t in range(bins):
                x = rng.integers(0, 40, (per_bin, nf)).astype(np.int32)
                mask = rng.random((per_bin, no)) < 0.8
                mask[:, 0] = True; mask[:, opt] = True
                pr = dpnet_io.probs(dpnet_io.forward(net, x), mask, 1.0, FLOOR).astype(np.float64)
                k = np.array([rng.choice(no, p=row / row.sum()) for row in pr])
                q = np.minimum(65535, np.round(-np.log(pr[np.arange(per_bin), k]) * 4096))
                b[t + 1] = b[t] + (k == opt).sum() + rng.poisson(1.0)
                parts.append(dict(x=x.astype(np.uint16), k=k.astype(np.int32), logq=(-q / 4096).astype(np.float32), mask=mask,
                                  flags=np.zeros(per_bin, np.uint8), round=(10 * t + rng.integers(0, 10, per_bin)).astype(np.int32)))
            babies[team], team_recs[team] = b, {key: np.concatenate([pt[key] for pt in parts]) for key in parts[0]}
        w = int(rng.random() < 1 / (1 + np.exp(-(babies[0][bins] - babies[1][bins]) / 4)))   # 1: team 0 wins
        for team in (0, 1):
            z = 1.0 if (team == 0) == bool(w) else -1.0
            rounds = 10 * np.arange(bins)

            def side(s_):
                bb = babies[s_][:bins]
                return np.stack([bb, np.ones(bins), 500 + 5 * rounds / 10 + rng.normal(0, 20, bins), np.zeros(bins),
                                 np.zeros(bins), np.zeros(bins)], 1)
            us, them = side(team), side(1 - team)
            tr = np.concatenate([us, them, us - them, (rounds / 2000)[:, None], np.full((bins, 1), 0.5)], 1).astype(np.float32)
            trace_rows.append(dict(game=np.full(bins, g), team=np.full(bins, team), round=rounds, pair=np.full(bins, g // 2),
                                   z=np.full(bins, z, np.float32), trace=tr))
            r = dict(team_recs[team])
            n = len(r['k'])
            r.update(team=np.full(n, team, np.int32), game=np.full(n, g, np.int64), pair=np.full(n, g // 2, np.int64),
                     z=np.full(n, z, np.float32), trace=tr[r['round'] // 10])
            recs.append(r)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'p5.npz', **{k: np.concatenate([r[k] for r in recs]) for k in recs[0]})
    np.savez(out_dir / 'trace.npz', **{k: np.concatenate([r[k] for r in trace_rows]) for k in trace_rows[0]})


def check_critic():
    rng = np.random.default_rng(7)
    # two team-games (game 3 team 0: 4 bins; game 3 team 1: 3 bins), rows shuffled
    rows = dict(game=np.array([3, 3, 3, 3, 3, 3, 3]), team=np.array([0, 0, 0, 0, 1, 1, 1]),
                round=np.array([0, 10, 20, 30, 10, 20, 30]), z=np.array([1, 1, 1, 1, -1, -1, -1], np.float32))
    v = np.array([0.1, 0.3, -0.2, 0.5, 0.0, -0.4, -0.6])
    perm = rng.permutation(7)
    rp = {k: a[perm] for k, a in rows.items()}
    for lam in (1.0, 0.0, 0.7):
        A = rl_critic.gae(rp, v[perm], lam)[np.argsort(perm)]
        exp = []
        for idx in ([0, 1, 2, 3], [4, 5, 6]):
            vs, z = v[idx], rows['z'][idx[0]]
            dl = np.r_[vs[1:], z] - vs
            exp += [sum(lam ** l * dl[t + l] for l in range(len(dl) - t)) for t in range(len(dl))]
        assert np.allclose(A, exp), (lam, A, exp)
    assert np.allclose(rl_critic.gae(rows, v, 1.0), rows['z'] - v)
    # record -> bin: before the first row -> first row; between rows -> latest row <= round; unknown team-game -> -1
    ri = rl_critic.record_rows(rp, np.array([3, 3, 3, 3, 4]), np.array([0, 1, 1, 0, 0]), np.array([25, 5, 30, 999, 10]))
    got = [None if i < 0 else (int(rp['team'][i]), int(rp['round'][i])) for i in ri]
    assert got == [(0, 20), (1, 10), (1, 30), (0, 30), None], got
    # antisymmetry: swapping us/them and the seat negates V
    n = 600
    tr = rng.normal(0, 1, (n, 20)).astype(np.float32); tr[:, 18] = rng.random(n); tr[:, 19] = 0.5
    tr[:, 12:18] = tr[:, :6] - tr[:, 6:12]
    rr = dict(game=np.repeat(np.arange(n // 20), 20), team=np.tile([0, 1], n // 2), round=np.tile(np.arange(20) * 10, n // 20),
              z=np.sign(tr[:, 0] - tr[:, 6] + rng.normal(0, 1, n)).astype(np.float32), trace=tr)
    c = rl_critic.Critic('rich', epochs=3, seed=1).fit(rl_critic.features(rr), (rr['z'] + 1) / 2, rr['game'])
    sw = dict(rr); sw['trace'] = np.concatenate([tr[:, 6:12], tr[:, :6], -tr[:, 12:18], tr[:, 18:]], 1); sw['team'] = 1 - rr['team']
    va, vb = c.predict(rl_critic.features(rr)), c.predict(rl_critic.features(sw))
    assert np.abs(va + vb).max() < 1e-5, np.abs(va + vb).max()
    # AUC with ties matches the trainer's
    sc = rng.integers(0, 5, 300).astype(float); y = (rng.random(300) < 0.4).astype(float)
    assert abs(rl_critic.auc(sc, y) - auc_ref(sc, y)) < 1e-12
    print('rl_critic ok: gae, record_rows, antisymmetry max', float(np.abs(va + vb).max()))


def auc_ref(score, y):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import rl_train
    return rl_train.auc(score, y)


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
    # 5. --no-kl (M1): a far anchor (KL far above --max-anchor-kl) neither refuses the export nor leaks into the gradient
    far = work / 'far'; far.mkdir(exist_ok=True)
    raw = dpnet_io.load_raw(n0 / 'net.json')
    raw['points']['5'][-1]['b'][1] += 8.0
    (far / 'net.json').write_text(json.dumps(raw))
    outs = []
    for tag, anc in (('self', n0 / 'net.json'), ('far', far / 'net.json')):
        o = work / f'nokl-{tag}'
        assert run_train('--net', n0 / 'net.json', '--anchor', anc, '--data', d, '--out', o, '--points', '5', '--lr', 3e-3,
                         '--no-kl', '--max-anchor-kl', 0.01, '--collapse-kl', 0.01, '--no-snr') == 0, f'--no-kl refused ({tag})'
        outs.append((o / 'net.json').read_bytes())
        mt = json.loads((o / 'metrics.json').read_text())['points']['5']
        assert mt['beta'] == 0.0 and mt['changed']
    assert mt['kl_anchor_start'] > 0.5, mt['kl_anchor_start']
    assert outs[0] == outs[1], '--no-kl: the anchor changed the trained net'
    print('--no-kl ok: far anchor KL', mt['kl_anchor_start'], 'no refusal, identical export')
    # 6. rl_critic units
    check_critic()
    # 7. critic + GAE through rl_train: option 2 adds babies; P(option 2) must rise
    d = work / 'gae/data'
    sample_gae_data(n0 / 'net.json', d, seed=21)
    for crit in ('trace', 'rich'):
        o = work / f'gae/{crit}'
        assert run_train('--net', n0 / 'net.json', '--data', d, '--out', o, '--points', '5', '--lr', 3e-3, '--epochs', 4,
                         '--critic', crit, '--gae-lambda', 0.8) == 0, crit
        m = json.loads((o / 'metrics.json').read_text())
        mt = m['points']['5']
        print(f'GAE {crit} lambda 0.8: P(option 2)', mt['mean_probs_start'][2], '->', mt['mean_probs_end'][2], 'snr',
              mt.get('grad_snr_cos'), 'critic', {k: v for k, v in m['critic'].items() if k != 'heldout'})
        assert m['critic']['gae_lambda'] == 0.8 and mt['no_bin'] == 0 and m['pairs_with_baseline'] == 0
        assert mt['mean_probs_end'][2] > mt['mean_probs_start'][2]
        if crit == 'rich':
            assert m['critic']['oof_frac'] == 1.0 and m['critic']['heldout']['all']['auc'] is not None
    print('ALL OK', work)


def test_rl_train(tmp_path):
    main(tmp_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=None)
    ap.add_argument('--net', default=None, help='a real net.json for the lr-0 identity check')
    a = ap.parse_args()
    main(a.work, a.net)
