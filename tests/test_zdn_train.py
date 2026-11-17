"""zdn trainer tests (scripts/zdn_train.py): the register context, the export's folding of the standardisation into the
weights (checked against the packed net's math), and a small end-to-end training run on synthetic records."""
import json, re, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
np = pytest.importorskip('numpy')
torch = pytest.importorskip('torch')
import zdn_pack, zdn_train  # noqa: E402
import zdn_helpers as zh  # noqa: E402


def test_reg_context():
    R = [dict(nk=2, kind=np.array([[0, 1], [0, 1], [0, 1]]), lab=np.array([1, 0, -1])),
         dict(nk=3, kind=np.array([[2, 0], [1, 1], [0, 2]]), lab=np.array([0, 1, 1])),
         dict(nk=2, kind=np.array([[0, 1], [0, 1], [0, 1]]), lab=np.array([0, 0, 0]))]
    rc = zdn_train.reg_context(R)
    assert [a.shape for a in rc] == [(3, 7)] * 3
    assert not rc[0].any()
    assert rc[1].tolist() == [[0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    assert rc[2].tolist() == [[0, 1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]]


def test_export_folds_standardisation_exactly(tmp_path):
    """The packed net on raw inputs (ZNet.java's math) gives the trained net's logits on standardised inputs."""
    net, st = zh.random_net()
    _, chars = zh.export_packed(net, st, tmp_path / 'net.json')
    pk = zh.unpack(chars)
    rng = np.random.default_rng(1)
    x = rng.integers(-20, 20, 12)
    for head, (nK, D) in [(-1, (6, 3)), (0, (3, 2)), (1, (4, 1)), (2, (2, 5))]:
        nC = 4 if head < 0 else 9
        ctx = rng.integers(-3, 4, nC) if head < 0 else (rng.random(nC) < 0.3).astype(int)
        kind = rng.integers(0, nK, 7)
        xs = rng.integers(-30, 30, (7, D))
        want = zh.torch_logits(net, st, x, ctx, kind, xs, head)
        got = zh.ref_logits(pk, x, ctx, kind, xs, head)
        assert np.allclose(got, want, rtol=1e-4, atol=1e-4), head
    assert zh.ref_logits(pk, x, np.zeros(4), [6, -1, 0], np.zeros((3, 3)), -1)[:2].tolist() == [-np.inf, -np.inf]


def _synthetic_games(root, games=8, turns=80, steps=3, seed=0):
    """Baby records with learnable labels: every head's label is the option with the largest first feature."""
    rng = np.random.default_rng(seed)
    for g in range(games):
        recs = []
        for t in range(turns):
            robot, rnd = 1000 + t % 3, t + 1
            regs = []
            for nk, D, K in ((3, 2, 3), (2, 1, 3)):
                kind = rng.integers(0, nk, K).tolist()
                x = rng.integers(-40, 40, (K, D))
                regs.append((kind, x.tolist(), int(np.argmax(x[:, 0])), D, nk))
            recs.append(zh.record(robot, rnd, 200, None, zh.reg_feats(rng.integers(-9, 9, 4).tolist(), rng.integers(0, 3, 6).tolist(), regs)))
            for s in range(steps):
                kind = rng.integers(0, 3, 4).tolist()          # declared nk 4: kind 3 never occurs
                x = rng.integers(-40, 40, (4, 2))
                lab = int(np.argmax(x[:, 0])) if s < steps - 1 or t % 10 else -1
                recs.append(zh.record(robot, rnd, 201, lab, zh.step_feats(s, rng.integers(0, 3, 3).tolist(), kind, x.tolist(), 2, 4)))
        d = root / f'g{g:05d}'
        d.mkdir(parents=True)
        zh.write(d / 'decisions-A.bin.gz', recs)


def test_train_end_to_end(tmp_path, monkeypatch):
    _synthetic_games(tmp_path / 'games')
    out = tmp_path / 'net'
    monkeypatch.setattr(sys, 'argv', ['zdn_train.py', '--games', str(tmp_path / 'games/*'), '--unit', 'baby', '--out', str(out),
                                      '--epochs', '8', '--device', 'cpu', '--batch', '64', '--val-every', '4', '--lr', '3e-3'])
    zdn_train.main()
    log = (out / 'train.log').read_text().splitlines()
    assert log[0].startswith('baby: 8 files, 640 turns, 1920 steps, 2 registers')
    m = re.match(r'epoch 8: val step loss [\d.]+ acc ([\d.]+); register acc \[([\d.]+), ([\d.]+)\]', log[-2])
    assert m, log[-2]
    step_acc, reg0, reg1 = map(float, m.groups())
    assert step_acc > 0.6                     # chance about 0.3
    assert reg0 > 0.5 and reg1 > 0.5          # chance about 0.33
    j = json.loads((out / 'net.json').read_text())
    assert len(j['step']['B']) == 4           # the declared kind count, not the largest kind seen
    assert [len(r['B']) for r in j['regs']] == [3, 2]
    assert [len(r['C'][0]) for r in j['regs']] == [5, 5]
    chars = zdn_pack.pack(j)
    assert chars[10:16] == [3, 2, 5, 2, 1, 5]
    zh.unpack(chars)
