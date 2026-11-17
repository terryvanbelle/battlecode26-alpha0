"""zdn tests that need the JDK (and, for the games, the engine runtime under build/student and the maps).

  - ZNet.java computes the same logits as the trained torch net (through export + pack + the Java parser).
  - The teacher build (zdb + recorder) plays exactly like plain zdb (same trajectory hash), raises nothing, and labels
    its records: every register value and primitive of zdb's is one of zdn's options.
  - The zdn player runs without errors and records, with no net (uniform choices) and with a net trained on the
    teacher's records.
"""
import json, os, struct, subprocess, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
np = pytest.importorskip('numpy')
import build_zdn, zdn_data  # noqa: E402
import zdn_helpers as zh  # noqa: E402

JDK = ROOT / 'vendor/jdk/bin'
JAR = ROOT / 'vendor/engine/battlecode26-java-1.2.6.jar'
MAP = 'DefaultSmall'
need_jdk = pytest.mark.skipif(not (JDK / 'javac').exists() or not JAR.exists(), reason='needs vendor/jdk and the engine jar')
need_engine = pytest.mark.skipif(not (JDK / 'javac').exists() or not (ROOT / 'build/student/classes').exists()
                                 or not (ROOT / f'maps/{MAP}.map26').exists(), reason='needs the student runtime build and maps')


def javac(files, out):
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([str(JDK / 'javac'), '--release', '21', '-nowarn', '-proc:none', '-cp', str(JAR), '-d', str(out),
                        *map(str, files)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    return out


def game(out, a, b, decisions=None, seed=3):
    cmd = [str(ROOT / 'scripts/student_java.sh'), '--map', MAP, '--maps-dir', str(ROOT / 'maps'), '--seed', str(seed),
           '--a', a, '--b', b, '--out', str(out)]
    if decisions:
        cmd += ['--decisions', decisions]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env={**os.environ, 'JAVA_FLAGS': '-Xmx2g'})
    assert (out / 'result.json').exists(), p.stderr[-3000:]
    res = json.loads((out / 'result.json').read_text())
    assert res['finished'] and res['error'] is None and res['decisions_errors'] == 0, res
    return res


# ---------------------------------------------------------------- ZNet.java parity

@need_jdk
def test_znet_java_matches_torch(tmp_path):
    pytest.importorskip('torch')
    net, st = zh.random_net(seed=4)
    _, chars = zh.export_packed(net, st, tmp_path / 'net.json')
    (tmp_path / 'net.bin').write_bytes(struct.pack(f'>{len(chars)}H', *chars))
    pk = zh.unpack(chars)
    rng = np.random.default_rng(5)
    cases, lines = [], []
    for head, (nK, D) in [(-1, (6, 3)), (0, (3, 2)), (1, (4, 1)), (2, (2, 5))] * 3:
        x = rng.integers(-20, 20, 12)
        x[rng.random(12) < 0.3] = 0                      # ZNet.java skips zero inputs
        ctx = rng.integers(-3, 4, 4) if head < 0 else (rng.random(9) < 0.3).astype(int)
        kind = rng.integers(0, nK, 6)
        kind[0] = nK                                    # a kind the net does not know
        xs = rng.integers(-30, 30, (6, D))
        cases.append((head, x, ctx, kind, xs))
        lines.append(' '.join(map(str, [head, len(x), *x, len(ctx), *ctx, len(kind), D,
                                        *[v for k, row in zip(kind, xs) for v in (k, *row)]])))
    (tmp_path / 'cases.txt').write_text('\n'.join(lines) + '\n')
    cls = javac([*sorted((ROOT / 'java/bots/zdn').glob('*.java')), ROOT / 'tests/java/zdn/ZNetCheck.java'], tmp_path / 'classes')
    p = subprocess.run([str(JDK / 'java'), '-cp', f'{cls}:{JAR}', 'zdn.ZNetCheck', str(tmp_path / 'net.bin'), str(tmp_path / 'cases.txt')],
                       capture_output=True, text=True, check=True)
    got = [np.array([float('-inf') if v == '-inf' else float(v) for v in line.split()]) for line in p.stdout.splitlines()]
    assert len(got) == len(cases)
    for (head, x, ctx, kind, xs), z in zip(cases, got):
        assert np.isneginf(z[0])
        ref = zh.ref_logits(pk, x, ctx, kind, xs, head)
        want = zh.torch_logits(net, st, x, ctx, kind[1:], xs[1:], head)
        assert np.allclose(z[1:], ref[1:], rtol=1e-4, atol=1e-3), head
        assert np.allclose(z[1:], want, rtol=1e-3, atol=1e-3), head


# ---------------------------------------------------------------- games

@pytest.fixture(scope='module')
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp('zdn')
    javac(sorted((ROOT / 'java/bots/zdp').glob('*.java')), d / 'classes/zdp')
    build_zdn.build_teacher('zdntx', 300, 1000, out=d / 'src')
    javac(sorted((d / 'src/zdntx').glob('*.java')), d / 'classes/zdntx')
    return d


@pytest.fixture(scope='module')
def teacher_games(built):
    zdp, zdt = f"bot:zdp:{built / 'classes/zdp'}", f"bot:zdntx:{built / 'classes/zdntx'}"
    plain = game(built / 'g-plain', zdp, zdp)
    rec = game(built / 'g-teacher', zdt, zdt, decisions='AB')
    return plain, rec, built / 'g-teacher'


@need_engine
def test_recorder_leaves_zdb_play_unchanged(teacher_games):
    plain, rec, _ = teacher_games
    assert (rec['traj_hash'], rec['winner'], rec['rounds']) == (plain['traj_hash'], plain['winner'], plain['rounds'])


@need_engine
def test_recorder_labels_zdb_choices(teacher_games):
    _, rec, g = teacher_games
    for side in 'AB':
        f = g / f'decisions-{side}.bin.gz'
        assert zdn_data.errors(f) == []
        d = zdn_data.read(f)
        for unit in ('baby', 'king'):
            r, s = d[unit]['reg'], d[unit]['step']
            assert r is not None and s is not None, (side, unit)
            for i, reg in enumerate(r['regs']):
                assert np.mean(reg['lab'] >= 0) >= 0.98, (side, unit, i)
            assert np.mean(s['label'] >= 0) >= 0.98, (side, unit)
            chosen = np.where(s['label'] >= 0, s['kind'][np.arange(len(s['label'])), np.maximum(s['label'], 0)], -1)
            ends = int((chosen == 0).sum())
            assert 0.98 * len(r['robot']) <= ends <= len(r['robot']), (side, unit)   # one END per recorded turn
        assert (d['baby']['step']['kind'][np.arange(len(d['baby']['step']['label'])), np.maximum(d['baby']['step']['label'], 0)] == 1).any()


def _check_player_records(f):
    assert zdn_data.errors(f) == []
    d = zdn_data.read(f)
    r, s = d['baby']['reg'], d['baby']['step']
    assert r is not None and len(r['robot']) > 50
    for reg in r['regs']:
        n = (reg['kind'] >= 0).sum(1)
        assert ((reg['lab'] >= 0) & (reg['lab'] < n)).all()
    n = (s['kind'] >= 0).sum(1)
    assert ((s['label'] >= 0) & (s['label'] < n)).all()
    turns = {}
    for robot, rnd, st in zip(s['robot'], s['round'], s['step']):
        turns.setdefault((int(robot), int(rnd)), []).append(int(st))
    assert all(v == list(range(len(v))) and len(v) <= 16 for v in turns.values())
    return d


@need_engine
def test_player_without_net_runs_and_records(built):
    build_zdn.build_player('zdnx', record=True, out=built / 'src')
    javac(sorted((built / 'src/zdnx').glob('*.java')), built / 'classes/zdnx')
    game(built / 'g-player', f"bot:zdnx:{built / 'classes/zdnx'}", f"bot:zdp:{built / 'classes/zdp'}", decisions='A')
    _check_player_records(built / 'g-player/decisions-A.bin.gz')


@need_engine
def test_player_with_trained_net(built, teacher_games):
    pytest.importorskip('torch')
    nets = {}
    for unit in ('baby', 'king'):
        out = built / f'net-{unit}'
        subprocess.run([sys.executable, str(ROOT / 'scripts/zdn_train.py'), '--games', str(teacher_games[2]), '--unit', unit,
                        '--out', str(out), '--epochs', '1', '--device', 'cpu', '--val-every', '2'],
                       check=True, capture_output=True, text=True, timeout=900)
        nets[unit] = str(out / 'net.json')
    build_zdn.build_player('zdny', baby=nets['baby'], king=nets['king'], record=True, out=built / 'src')
    javac(sorted((built / 'src/zdny').glob('*.java')), built / 'classes/zdny')
    game(built / 'g-net', f"bot:zdny:{built / 'classes/zdny'}", f"bot:zdp:{built / 'classes/zdp'}", decisions='A')
    _check_player_records(built / 'g-net/decisions-A.bin.gz')
