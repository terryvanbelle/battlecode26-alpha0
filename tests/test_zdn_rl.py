"""zdn RL pipeline tests (scripts/zdn_rl.py, zdn_rl_data.py, zdn_rl_train.py) on synthetic records and games.

  - The decision table holds every recorded choice, grouped by turn in order, with its options; subsets keep them.
  - net_from_json gives ZNet.java's logits from raw inputs; unfold / export keep the function for any standardisation.
  - zdn_rl_data passes parity when the records' q came from the nets that played, fails (exit 2) when they did not.
  - zdn_rl_train: --lr 0 leaves the nets byte-identical; a planted reward moves the policy toward the rewarded choice.
"""
import json, subprocess, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
np = pytest.importorskip('numpy')
torch = pytest.importorskip('torch')
import zdn_data, zdn_rl  # noqa: E402
import zdn_helpers as zh  # noqa: E402

REGS = ((3, 2), (4, 1), (2, 5))


def _game(tmp_path, seed=0, policy=None, name='decisions-A.bin.gz', **kw):
    recs, turns = zh.synth_records(np.random.default_rng(seed), policy=policy, **kw)
    f = tmp_path / name
    f.parent.mkdir(parents=True, exist_ok=True)
    zh.write(f, recs)
    return f, turns


def test_table_holds_every_choice_in_turn_order(tmp_path):
    f, turns = _game(tmp_path, turns=25)
    tab = zdn_rl.load_table([f], 'baby')
    assert len(tab['SL']) == 25 and tab['meta']['nR'] == 3 and tab['meta']['nk'] == [6, 3, 4, 2]
    for t, tu in enumerate(turns):
        rows = np.arange(tab['d0'][t], tab['d0'][t + 1])
        assert tab['SL'][t].tolist() == list(tu['S']) + list(tu['L'])
        assert tab['head'][rows].tolist() == [0, 1, 2] + [-1] * len(tu['steps'])
        assert tab['step'][rows[3:]].tolist() == list(range(len(tu['steps'])))
        kind, x = zdn_rl.gather(tab, rows)
        for i, (k, xs, lab) in enumerate(tu['regs']):
            assert kind[i, :len(k)].tolist() == list(k) and (kind[i, len(k):] == -1).all()
            assert x[i, :len(k), :xs.shape[1]].tolist() == xs.tolist() and tab['label'][rows[i]] == lab
            assert tab['rkind'][t, i] == k[lab]
        for i, (ctx, k, xs, lab) in enumerate(tu['steps']):
            r = 3 + i
            assert kind[r, :len(k)].tolist() == list(k) and tab['label'][rows[r]] == lab
            assert x[r, :len(k), :xs.shape[1]].tolist() == xs.tolist() and tab['ctx'][rows[r]].tolist() == list(ctx)
    assert (tab['q'][tab['n'] >= 2] > 0).all() and (tab['q'][tab['n'] == 1] == 0).all()
    sub = zdn_rl.subset(tab, np.array([3, 7, 20]))
    for i, t in enumerate([3, 7, 20]):
        a = np.arange(tab['d0'][t], tab['d0'][t + 1]); b = np.arange(sub['d0'][i], sub['d0'][i + 1])
        for k in ('head', 'label', 'q', 'ctx'):
            assert (tab[k][a] == sub[k][b]).all()
        assert all((u == v).all() for u, v in zip(zdn_rl.gather(tab, a), zdn_rl.gather(sub, b)))


def _ref_logp(pk, tab, rows, T=1.0):
    """log p of each row's label from the packed net's float64 math (zdn_helpers.ref_logits)."""
    out = []
    off = np.concatenate([[0], np.cumsum([k for k, _ in REGS])])
    kind, x = zdn_rl.gather(tab, rows)
    for i, r in enumerate(rows):
        t, h = tab['turn'][r], int(tab['head'][r])
        n = tab['n'][r]
        if h < 0:
            ctx = tab['ctx'][r]
            xs = x[i, :n, :3]
        else:
            ctx = np.zeros(off[-1])
            for j in range(h):
                ctx[off[j] + tab['rkind'][t, j]] = 1
            xs = x[i, :n, :REGS[h][1]]
        z = zh.ref_logits(pk, tab['SL'][t], ctx, kind[i, :n], xs, h) / T
        z = z - z.max()
        out.append(z[tab['label'][r]] - np.log(np.exp(z).sum()))
    return np.array(out)


def test_net_from_json_unfold_and_export_keep_the_function(tmp_path):
    net, st = zh.random_net(seed=2)
    doc, chars = zh.export_packed(net, st, tmp_path / 'net.json')
    pk = zh.unpack(chars)
    f, _ = _game(tmp_path, seed=3, turns=12)
    tab = zdn_rl.load_table([f], 'baby')
    turns = np.arange(len(tab['SL']))
    raw = zdn_rl.net_from_json(doc)
    rows, lp, _ = zdn_rl.logp_rows(raw, zdn_rl.Norm.identity(raw), tab, turns, 0.7, 'cpu')
    assert np.allclose(lp.detach().numpy(), _ref_logp(pk, tab, rows, 0.7), atol=2e-4)
    norm = zdn_rl.Norm.fit(tab, raw)
    unf = zdn_rl.unfold(raw, norm)
    _, lp2, _ = zdn_rl.logp_rows(unf, norm, tab, turns, 0.7, 'cpu')
    assert np.allclose(lp2.detach().numpy(), lp.detach().numpy(), atol=2e-4)
    zdn_rl.export(unf, norm, tmp_path / 'again.json', doc['meta'])
    again = zdn_rl.net_from_json(json.loads((tmp_path / 'again.json').read_text()))
    _, lp3, _ = zdn_rl.logp_rows(again, zdn_rl.Norm.identity(again), tab, turns, 0.7, 'cpu')
    assert np.allclose(lp3.detach().numpy(), lp.detach().numpy(), atol=2e-4)
    norm.save(tmp_path / 'n.npz')
    n2 = zdn_rl.Norm.load(tmp_path / 'n.npz')
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(norm.kst, n2.kst))


# ---------------------------------------------------------------- the data and train stages

def _iteration(tmp_path, n_games=16, turns=60, seed=0, reward_kind=None, corrupt_q=False):
    """A fake RL iteration: nets/0/{baby,king}.json (random), iter dir with jobs.json and n_games games of learner
    (side A) records sampled from nets/0's baby net. reward_kind: A wins the games where its steps chose that kind most
    often (above the median), else B."""
    nets = tmp_path / 'nets/0'
    nets.mkdir(parents=True)
    net, st = zh.random_net(seed=7)
    doc, chars = zh.export_packed(net, st, nets / 'baby.json')
    kn, kst = zh.random_net(seed=8, regs=((2, 2), (5, 2)))
    zh.export_packed(kn, kst, nets / 'king.json')
    pk = zh.unpack(chars)
    idir = tmp_path / 'iter'
    jobs, frac = [], []
    rng = np.random.default_rng(seed)
    for g in range(n_games):
        gid = f'g{g:03d}'
        f, tu = _game(idir / 'games' / gid, seed=seed * 1000 + g, policy=pk, turns=turns, max_steps=4)
        if corrupt_q:
            recs, _ = zh.synth_records(np.random.default_rng(seed * 1000 + g), policy=pk, turns=turns, max_steps=4, version=0)
            zh.write(f, [r[:-2] + bytes([0, 9]) if r[9] in (201,) else r for r in recs])   # every step's q -> 9
        frac.append(np.mean([s[1][s[3]] == reward_kind for t in tu for s in t['steps']]) if reward_kind is not None else rng.random())
        jobs.append(dict(id=gid, map='DefaultSmall', seed=g, kind='pool', opponent='zdnr0', learner_sides='A', pair=-1, a='zdnr0', b='zdnr0'))
    med = np.median(frac)
    for j, fr in zip(jobs, frac):
        g = idir / 'games' / j['id']
        res = dict(finished=True, error=None, winner='A' if fr > med else 'B', reason='x', rounds=200, seconds=1,
                   A={'failures': []}, B={'failures': []})
        (g / 'result.json').write_text(json.dumps(res))
        rows = ['round\tAb\tAk\tAc\tAx\tAt\tBb\tBk\tBc\tBx\tBt'] + [f'{r}\t5\t1\t100\t0\t0\t5\t1\t100\t0\t0' for r in range(0, 200, 10)]
        (g / 'trace.tsv').write_text('\n'.join(rows) + '\n')
    (idir / 'jobs.json').write_text(json.dumps(dict(learner=dict(version=0, package='zdnr0'), jobs=jobs)))
    cfg = tmp_path / 'cfg.json'
    cfg.write_text(json.dumps(dict(name='t', seed_base=1, temperature=1.0, parity=dict(median_max=1e-3, p999_max=2e-3))))
    return cfg, idir, tmp_path / 'nets', pk


def _data(cfg, idir, nets):
    return subprocess.run([sys.executable, str(ROOT / 'scripts/zdn_rl_data.py'), '--config', str(cfg), '--iter', '0',
                           '--iter-dir', str(idir), '--nets', str(nets), '--keep-raw'], capture_output=True, text=True)


need_maps = pytest.mark.skipif(not (ROOT / 'maps/DefaultSmall.map26').exists(), reason='needs maps/DefaultSmall.map26')


@need_maps
def test_data_stage_parity(tmp_path):
    cfg, idir, nets, _ = _iteration(tmp_path / 'ok', n_games=3, turns=20)
    p = _data(cfg, idir, nets)
    assert p.returncode == 0, p.stdout + p.stderr
    s = json.loads((idir / 'data/summary.json').read_text())
    u = s['units']['baby']
    assert s['parity']['passed'] and u['turns'] == 60 and u['parity_n'] > 50 and u['parity_median'] < 2e-4
    assert u['learner_version_frac'] == 1.0 and s['units']['king']['turns'] == 0
    tab = __import__('zdn_rl_data').load(idir / 'data/baby.npz')
    assert set(tab['game'].tolist()) == {0, 1, 2} and (tab['team'] == 0).all() and tab['meta']['nR'] == 3
    cfg, idir, nets, _ = _iteration(tmp_path / 'bad', n_games=2, turns=20, corrupt_q=True)
    p = _data(cfg, idir, nets)
    assert p.returncode == 2 and 'PARITY FAILED' in p.stdout


def _train(nets, data, out, *extra):
    return subprocess.run([sys.executable, str(ROOT / 'scripts/zdn_rl_train.py'), '--nets', str(nets / '0'), '--anchor', str(nets / '0'),
                           '--data', str(data), '--out', str(out), '--version', '0', '--threads', '2', *extra],
                          capture_output=True, text=True)


@need_maps
def test_train_identity_and_planted_reward(tmp_path):
    # many short games: each choice moves its game's outcome more (the planted signal per decision ~ 1/sqrt(steps))
    cfg, idir, nets, pk = _iteration(tmp_path, n_games=192, turns=2, reward_kind=2)
    assert _data(cfg, idir, nets).returncode == 0
    p = _train(nets, idir / 'data', tmp_path / 'same', '--lr', '0')
    assert p.returncode == 0, p.stderr[-3000:]
    for u in ('baby', 'king'):
        assert (tmp_path / f'same/{u}.json').read_bytes() == (nets / f'0/{u}.json').read_bytes()
    p = _train(nets, idir / 'data', tmp_path / 'new', '--lr', '0.01', '--epochs', '6', '--batch-turns', '64',
               '--target-kl', '10', '--clip', '0.5', '--beta', '0', '--c-ent', '0', '--max-anchor-kl', '100')
    assert p.returncode == 0, p.stderr[-3000:]
    m = json.loads((tmp_path / 'new/metrics.json').read_text())
    assert m['changed_units'] == ['baby'] and m['units']['baby']['steps'] > 10
    assert abs(m['units']['baby']['kl_old_start']) < 1e-4    # the unfolded learner reproduces the recorded q
    assert (tmp_path / 'new/king.json').read_bytes() == (nets / '0/king.json').read_bytes()

    # the rewarded kind's probability went up, over the steps where it competed with another kind
    tab = __import__('zdn_rl_data').load(idir / 'data/baby.npz')
    rows = np.nonzero((tab['head'] < 0) & (tab['n'] >= 2))[0]
    kind, _ = zdn_rl.gather(tab, rows)
    some = (kind == 2).any(1) & ((kind >= 0) & (kind != 2)).any(1)
    rows, kind = rows[some], kind[some]

    def p_kind(path):
        net = zdn_rl.net_from_json(json.loads(Path(path).read_text()))
        turns = np.unique(tab['turn'][rows])
        _, _, full = zdn_rl.logp_rows(net, zdn_rl.Norm.identity(net), tab, turns, 1.0, 'cpu', rows=rows)
        pos, lsm = full[-1]
        pr = lsm.exp().detach().numpy()
        return float((pr * (kind[pos, :pr.shape[1]] == 2)).sum(1).mean())

    before, after = p_kind(nets / '0/baby.json'), p_kind(tmp_path / 'new/baby.json')
    assert after > before + 0.03, (before, after, m['units']['baby'])
