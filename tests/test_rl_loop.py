"""RL loop plumbing without games (docs/rl-plan.md section 4): rl_schedule.py's pool/PFSP/jobs and rl_data.py's
trailer parsing, parity check, z/trace join, npz layout and raw-file deletion, on synthetic games whose decision files
carry Java-format trailers computed with dpnet_io (the bot's sampling math). Needs numpy.

    python3 tests/test_rl_loop.py [WORKDIR]       (or pytest tests/test_rl_loop.py)
"""
import gzip, json, math, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import dpnet_io  # noqa: E402
from test_trailer import java_trailer  # noqa: E402

PY = sys.executable
CFG = dict(json.loads((ROOT / 'configs/rl-v1.json').read_text()), name='t', games_per_iter=20, mirror_fraction=0.6,
           trainable_points=[5], frozen_sample=1.0, training_maps=['mincemeat', 'jail'])


def make_net(path, nin=(32, 49), nout=(5, 39), seed=0):
    rng = np.random.default_rng(seed)
    pts = {}
    for p, ni, no in ((5, nin[0], nout[0]), (1, nin[1], nout[1])):
        ls = [dict(w=rng.normal(0, 0.3, (16, ni)), b=rng.normal(0, 0.1, 16)), dict(w=rng.normal(0, 1, (no, 16)), b=rng.normal(0, 1, no))]
        pts[p] = ls
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dpnet_io.save_net(path, pts)
    return dpnet_io.load_net(path)


def records(net, p, N, rng, team, floor, version, rounds=1500):
    """Decision file bytes for N records of point p sampled like DNet.sample."""
    ls = net[p]
    ni, no = ls[0]['w'].shape[1], ls[-1]['w'].shape[0]
    x = rng.integers(0, 30, (N, ni))
    mask = rng.random((N, no)) < 0.7
    mask[:, 0] = True
    pr = dpnet_io.probs(dpnet_io.forward(ls, x), mask, 1.0, floor)
    out = b''
    for i in range(N):
        q = pr[i].astype(np.float64)
        k = int(rng.choice(no, p=q / q.sum()))
        tr = java_trailer(k, float(pr[i, k]), no, mask[i].tolist(), 0, version)
        feats = list(map(int, x[i])) + tr
        out += struct.pack('>ihBBHHH', 1000 + i, int(rng.integers(1, rounds)), team, 0, p, k, len(feats))
        out += struct.pack(f'>{len(feats)}H', *feats)
    return out


def make_games(idir, net, floor, version, rng):
    """3 games: j0000 mirror (A wins), j0001/j0002 a pool pair (learner A loses, learner B draws)."""
    jobs = [dict(id='j0000', unit=0, kind='mirror', map='mincemeat', seed=1, a='zz', b='zz', opponent='self', record='AB', learner_sides='AB', pair=-1),
            dict(id='j0001', unit=1, kind='pool', map='jail', seed=2, a='zz', b='yy', opponent='v0', record='A', learner_sides='A', pair=1),
            dict(id='j0002', unit=1, kind='pool', map='jail', seed=2, a='yy', b='zz', opponent='v0', record='B', learner_sides='B', pair=1)]
    (idir / 'jobs.json').write_text(json.dumps(dict(iter=version, learner=dict(version=version, package='zz_nonexistent'), jobs=jobs)))
    for j, win in zip(jobs, ('A', 'B', None)):
        g = idir / 'games' / j['id']; g.mkdir(parents=True)
        side = {s: dict(transferred=10, kings=1, babies=5, cheese=3, cat_damage=0, failures=0) for s in 'AB'}
        (g / 'result.json').write_text(json.dumps(dict(finished=True, error=None, winner=win, reason='x', rounds=2000, **side)))
        rows = ['round\tA_babies\tA_kings\tA_cheese\tA_xfer\tA_rattraps\tB_babies\tB_kings\tB_cheese\tB_xfer\tB_rattraps']
        for r in range(10, 2001, 10):
            rows.append('\t'.join(map(str, [r, r // 100, 1, r, r * 2, 0, r // 50, 1, r // 2, r, 1])))
        (g / 'trace.tsv').write_text('\n'.join(rows) + '\n')
        for s in j['learner_sides']:
            t = 'AB'.index(s)
            (g / f'decisions-{s}.bin.gz').write_bytes(gzip.compress(records(net, 5, 200, rng, t, floor, version) + records(net, 1, 300, rng, t, floor, version)))


def run(*args):
    return subprocess.run([PY, *map(str, args)], capture_output=True, text=True)


def check_data(work):
    work = Path(work)
    cfgp = work / 'cfg.json'; cfgp.write_text(json.dumps(CFG))
    rd = work / 'runs' / 't'
    idir = rd / 'iter-0003'
    idir.mkdir(parents=True)
    net = make_net(rd / 'nets' / '3' / 'net.json')
    make_games(idir, net, CFG['floor'], 3, np.random.default_rng(1))
    r = run(ROOT / 'scripts/rl_data.py', '--config', cfgp, '--runs', work / 'runs', '--iter', 3)
    assert r.returncode == 0, r.stdout + r.stderr
    s = json.loads((idir / 'data/summary.json').read_text())
    assert s['parity']['passed'] and s['parity']['median'] < 3e-4 and s['parity']['n'] == 4 * 500, s['parity']
    assert s['score'] == {'v0': dict(points=0.5, games=2, score=0.25)}, s['score']
    d = np.load(idir / 'data/p5.npz')
    assert len(d['k']) == 800 and d['x'].shape == (800, 32) and d['mask'].shape == (800, 5) and d['trace'].shape == (800, 20)
    # z per team: mirror A +1 / B -1, pool learner A -1, pool learner B (draw) 0; pairs: mirror -1, pool 1
    for gi, team, z, pair in ((0, 0, 1, -1), (0, 1, -1, -1), (1, 0, -1, 1), (2, 1, 0, 1)):
        sel = (d['game'] == gi) & (d['team'] == team)
        assert sel.sum() == 200 and (d['z'][sel] == z).all() and (d['pair'][sel] == pair).all(), (gi, team)
    # trace join: latest row with round <= the record's round; us/them by team
    i = int(np.nonzero((d['game'] == 0) & (d['team'] == 1))[0][0])
    rb = max(10, d['round'][i] // 10 * 10)
    exp_us = [rb // 50, 1, rb // 2, rb]
    assert np.allclose(d['trace'][i, :4], exp_us) and np.isclose(d['trace'][i, 18], rb / 2000) \
        and np.isclose(d['trace'][i, 19], 31 * 31 / 3600), (d['trace'][i], d['round'][i])
    assert np.isclose(d['trace'][i, 4], rb - max(0, rb - 50) if rb >= 60 else rb)   # xfer_d50
    t = np.load(idir / 'data/trace.npz')
    assert len(t['game']) == 4 * 200 and t['trace'].shape == (800, 20)
    assert len(np.load(idir / 'data/p1.npz')['k']) == 1200   # frozen_sample 1.0
    assert not list(idir.glob('games/*/decisions-*.bin.gz')), 'raw files not deleted'

    # parity failure: a net that differs from the one that played -> exit 2, raw kept
    idir2 = rd / 'iter-0004'; idir2.mkdir()
    make_games(idir2, net, CFG['floor'], 4, np.random.default_rng(2))
    make_net(rd / 'nets' / '4' / 'net.json', seed=9)
    r = run(ROOT / 'scripts/rl_data.py', '--config', cfgp, '--runs', work / 'runs', '--iter', 4)
    assert r.returncode == 2 and 'PARITY FAILED' in r.stdout, r.stdout + r.stderr
    assert len(list(idir2.glob('games/*/decisions-*.bin.gz'))) == 4
    print('rl_data ok')


def check_schedule(work):
    work = Path(work)
    cfgp = work / 'cfg.json'; cfgp.write_text(json.dumps(CFG))
    rs = work / 'runs'
    for k in range(0, 14):
        r = run(ROOT / 'scripts/rl_schedule.py', '--config', cfgp, '--runs', rs, '--iter', k)
        assert r.returncode == 0, r.stderr
        doc = json.loads((rs / 't' / f'iter-{k:04d}' / 'jobs.json').read_text())
        jobs = doc['jobs']
        assert len(jobs) == 20 and all(j['seed'] == CFG['seed_base'] + 1000 * k + j['unit'] for j in jobs)
        if k == 0:
            assert all(j['kind'] == 'mirror' for j in jobs)
        else:
            pool = [j for j in jobs if j['kind'] == 'pool']
            assert len(pool) == 8 and sum(j['kind'] == 'mirror' for j in jobs) == 12
            for u in {j['unit'] for j in pool}:
                a, b = [j for j in pool if j['unit'] == u]
                assert (a['map'], a['seed']) == (b['map'], b['seed']) and a['record'] == 'A' and b['record'] == 'B' \
                    and a['a'] == b['b'] == f'zdpr{k}' and a['pair'] == u
            assert doc['counts'].get('v0', 0) >= 1, doc['counts']
        # fake results: the learner beats v0 always, loses to every snapshot
        for j in jobs:
            g = rs / 't' / f'iter-{k:04d}' / 'games' / j['id']; g.mkdir(parents=True)
            lw = j['opponent'] == 'v0'
            win = j['learner_sides'][0] if lw else ('B' if j['learner_sides'] == 'A' else 'A')
            (g / 'result.json').write_text(json.dumps(dict(finished=True, winner=win, A={}, B={})))
    pool = json.loads((rs / 't' / 'pool.json').read_text())
    names = [m['name'] for m in pool['members']]
    assert names == ['v0', 'v2', 'v4', 'v6', 'v8', 'v10', 'v12'], names
    assert pool['winrate']['v0'] == 1.0 and pool['winrate']['v2'] == 0.0
    w = pool['weights']   # PFSP from iter 10: v0 (winrate 1) at the 10% floor
    assert abs(w['v0'] - 0.10) < 1e-9 and abs(sum(w.values()) - 1) < 1e-9, w
    r = run(ROOT / 'scripts/rl_schedule.py', '--config', cfgp, '--runs', rs, '--promote', 4)
    pool = json.loads((rs / 't' / 'pool.json').read_text())
    assert pool['best'] == 4 and [m for m in pool['members'] if m['name'] == 'v4'][0]['promoted']
    print('rl_schedule ok')


def test_rl_loop(tmp_path):
    (tmp_path / 'a').mkdir(); (tmp_path / 'b').mkdir()
    check_data(tmp_path / 'a')
    check_schedule(tmp_path / 'b')


if __name__ == '__main__':
    w = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    (w / 'a').mkdir(parents=True, exist_ok=True); (w / 'b').mkdir(parents=True, exist_ok=True)
    check_data(w / 'a')
    check_schedule(w / 'b')
