"""zdn pipeline tests without the engine (docs/zdn.md): the teacher/player builds, the record reader, the packed-net
layout, the trainer's register context and export folding, and a small end-to-end training run."""
import json, re, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_zdn  # noqa: E402
import zdn_pack  # noqa: E402

np = pytest.importorskip('numpy')
import zdn_helpers as zh  # noqa: E402


# ---------------------------------------------------------------- builds

def test_zdp_and_zdn_share_no_class_but_robotplayer():
    """build_teacher puts zdp, zdn and the recorder in one package; a shared file name would silently drop a class."""
    zdp = {f.name for f in build_zdn.ZDP.glob('*.java')}
    zdn = {f.name for f in build_zdn.ZDN.glob('*.java')}
    zdt = {f.name for f in build_zdn.ZDT.glob('*.java')}
    assert zdp & zdn == {'RobotPlayer.java'}
    assert not zdt & (zdp | zdn)


def test_teacher_build_applies_every_patch_once(tmp_path):
    build_zdn.build_teacher('zdntx', 7, 9, out=tmp_path)
    d = tmp_path / 'zdntx'
    files = {f.name: f.read_text() for f in d.glob('*.java')}
    for name, text in files.items():
        assert re.search(r'^package zdntx;', text, re.M), name
        assert not re.search(r'^package (zdp|zdn);', text, re.M), name
    for name, patches in build_zdn.TEACHER_PATCHES.items():
        for old, new in patches:
            assert files[name].count(new) == 1, (name, new[:60])
    assert 'G.startTurn' in files['RobotPlayer.java']          # zdb's player, not zdn's
    assert 'ZPlayer.java' not in files
    assert 'new RcLog(rc)' in files['RobotPlayer.java']
    assert 'static final int BABY_PER_MILLE = 7, KING_PER_MILLE = 9;' in files['ZT.java']
    hooks = {n: t.count('ZT.') for n, t in files.items() if n not in ('ZT.java', 'RcLog.java') and 'ZT.' in t}
    assert hooks == {'RobotPlayer.java': 3, 'Nav.java': 1, 'Baby.java': 4}   # init/start/end, goal, mode + 3 samples
    assert files['RcLog.java'].count('ZT.pre(') >= 20


def test_teacher_build_rejects_a_drifted_patch(tmp_path, monkeypatch):
    patches = dict(build_zdn.TEACHER_PATCHES)
    patches['Nav.java'] = [('this text is not in Nav.java', 'x')]
    monkeypatch.setattr(build_zdn, 'TEACHER_PATCHES', patches)
    with pytest.raises(SystemExit, match='Nav.java'):
        build_zdn.build_teacher('zdntx', out=tmp_path)


def test_player_build(tmp_path):
    build_zdn.build_player('zdnx', record=True, out=tmp_path)
    d = tmp_path / 'zdnx'
    assert 'public static int RECORD_PER_MILLE = 1000;' in (d / 'ZPlayer.java').read_text()
    zw = (d / 'ZW.java').read_text()
    assert zw.startswith('package zdnx;')
    assert 'static final String[] BABY = {};' in zw and 'static final String[] KING = {};' in zw
    build_zdn.build_player('zdnx', out=tmp_path)
    assert 'public static int RECORD_PER_MILLE = 0;' in (d / 'ZPlayer.java').read_text()
    build_zdn.build_player('zdnx', record=25, out=tmp_path)
    assert 'public static int RECORD_PER_MILLE = 25;' in (d / 'ZPlayer.java').read_text()
    assert {f.name for f in d.glob('*.java')} == {f.name for f in build_zdn.ZDN.glob('*.java')}


# ---------------------------------------------------------------- packing

def _java_literal(lit):
    """Decode one Java string literal the way javac does (unicode escapes first, then the simple escapes)."""
    assert lit[0] == lit[-1] == '"'
    body = lit[1:-1]
    assert '\n' not in body and '\r' not in body
    out, i = [], 0
    while i < len(body):
        if body.startswith('\\u', i):
            out.append(int(body[i + 2:i + 6], 16)); i += 6
        elif body[i] == '\\':
            out.append({'n': 10, 'r': 13, '"': 34, '\\': 92}[body[i + 1]]); i += 2
        else:
            assert body[i] != '"'
            out.append(ord(body[i])); i += 1
    return out


def test_java_string_escapes_chars_that_break_source():
    chars = [0x0A, 0x0D, 0x22, 0x5C, 0x41, 0x0000, 0xD800, 0xFFFF, 0x5C, 0x75]
    parts = zdn_pack.java_string(chars, chunk=3)
    assert len(parts) == 4
    for bad in ('\\u000a', '\\u000d', '\\u0022', '\\u005c'):
        assert bad not in ''.join(parts)
    assert [c for p in parts for c in _java_literal(p)] == chars


def _tiny_net_json():
    rng = np.random.default_rng(3)
    r = lambda *s: rng.normal(size=s).tolist()
    sc = lambda nC, nK, D: dict(A=r(4, 3), a=r(4), **({'C': r(4, nC)} if nC else {}), B=r(nK, 4, D), E=r(nK, 4), w=r(4), w0=0.25)
    return dict(meta=dict(nS=2, nL=3, nC=2), W1=r(5, 5), b1=r(5), W2=r(3, 5), b2=r(3),
                step=sc(2, 3, 2), regs=[sc(5, 2, 1), sc(0, 3, 4)])


def test_pack_header_and_round_trip():
    j = _tiny_net_json()
    chars = zdn_pack.pack(j)
    assert chars[:10] == [0x5A4E, 2, 3, 5, 3, 4, 2, 3, 2, 2]
    assert chars[10:16] == [2, 1, 5, 3, 4, 0]                 # (nK, D, nC) per register
    net = zh.unpack(chars)
    f32 = lambda a: np.asarray(a, np.float32).astype(np.float64)
    assert np.array_equal(net['W1'], f32(j['W1'])) and np.array_equal(net['b2'], f32(j['b2']))
    for got, want in [(net['step'], j['step'])] + list(zip(net['regs'], j['regs'])):
        for k in ('A', 'a', 'B', 'E', 'w'):
            assert np.array_equal(got[k], f32(want[k])), k
        assert (got['C'] is None) == ('C' not in want)
        if 'C' in want:
            assert np.array_equal(got['C'], f32(want['C']))
        assert got['w0'] == np.float32(want['w0'])


def test_write_zw(tmp_path):
    p = tmp_path / 'net.json'
    p.write_text(json.dumps(_tiny_net_json()))
    out = tmp_path / 'ZW.java'
    zdn_pack.write_zw(out, 'zdnq', baby=str(p), temperature=0.5)
    s = out.read_text()
    assert s.startswith('package zdnq;') and 'static final float TEMPERATURE = 0.5f;' in s
    assert 'static final String[] KING = {};' in s
    body = s[s.index('BABY = {') + 8:s.index('};')]
    lits = re.findall(r'"(?:[^"\\]|\\.)*"', body)
    assert [c for lit in lits for c in _java_literal(lit)] == zdn_pack.pack(_tiny_net_json())


# ---------------------------------------------------------------- records

def test_reader_round_trip(tmp_path):
    import zdn_data
    regs_a = [([0, 1, 1], [[5, -3], [0, 0], [-7, 9]], 2, 2, 2), ([0], [[1]], -1, 1, 4)]
    regs_b = [([1], [[-1, 2]], 0, 2, 2), ([0, 3], [[4], [-4]], 1, 1, 4)]
    recs = [
        zh.record(11, 5, 200, None, zh.reg_feats([-2, 3, 40000 - 32768], [0, 7, 200], regs_a)),
        zh.record(11, 5, 201, 1, zh.step_feats(0, [1, -1], [0, 1], [[3, -3], [2, 2]], 2, 21)),
        zh.record(11, 5, 201, -1, zh.step_feats(1, [0, 0], [0, 1, 2], [[0, 0], [1, 1], [-5, 6]], 2, 21)),
        zh.record(12, 6, 200, None, zh.reg_feats([1, 1, 1], [1, 2, 3], regs_b, version=7, q=[1234, 0])),
        zh.record(12, 6, 201, 0, zh.step_feats(0, [9, 9], [0], [[0, 1]], 2, 21, q=65534)),
        zh.record(13, 6, 200, None, zh.reg_feats([4, 4, 4], [0, 0, 0], regs_b, ver=1)),    # the first collection's VER 1
        zh.record(13, 6, 201, 0, zh.step_feats(0, [8, 8], [0], [[0, 1]], 2, 21, ver=1)),
        zh.record(40, 6, 210, None, zh.reg_feats([1], [1], [([0, 1], [[1, 2, 3], [4, 5, 6]], 1, 3, 2)]), king=True),
        zh.record(40, 6, 299, 5, zh.error_feats('java.lang.RuntimeException: boom')),
    ]
    f = tmp_path / 'decisions-A.bin.gz'
    zh.write(f, recs)
    d = zdn_data.read(f)
    r, s = d['baby']['reg'], d['baby']['step']
    assert r['robot'].tolist() == [11, 12, 13] and r['round'].tolist() == [5, 6, 6]
    assert r['S'].tolist() == [[-2, 3, 40000 - 32768], [1, 1, 1], [4, 4, 4]]
    assert r['L'].tolist() == [[0, 7, 200], [1, 2, 3], [0, 0, 0]]
    assert r['version'].tolist() == [-1, 7, -1]
    g0, g1 = r['regs']
    assert g0['nk'] == 2 and g1['nk'] == 4
    assert g0['kind'].tolist() == [[0, 1, 1], [1, -1, -1], [1, -1, -1]]
    assert g0['x'][0].tolist() == [[5, -3], [0, 0], [-7, 9]] and g0['x'][1, 0].tolist() == [-1, 2]
    assert g0['lab'].tolist() == [2, 0, 0] and g1['lab'].tolist() == [-1, 1, 1]
    assert g0['q'].tolist() == [-1, 1234, -1] and g1['q'].tolist() == [-1, 0, -1]
    assert s['label'].tolist() == [1, -1, 0, 0] and s['step'].tolist() == [0, 1, 0, 0] and s['nk'] == 21
    assert s['q'].tolist() == [-1, -1, 65534, -1]
    assert s['ctx'].tolist() == [[1, -1], [0, 0], [9, 9], [8, 8]]
    assert s['kind'].tolist() == [[0, 1, -1], [0, 1, 2], [0, -1, -1], [0, -1, -1]]
    assert s['x'][1, 2].tolist() == [-5, 6]
    k = d['king']['reg']
    assert k['robot'].tolist() == [40] and k['regs'][0]['x'][0].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert d['king']['step'] is None
    assert zdn_data.errors(f) == [(40, 6, 5, 'java.lang.RuntimeException: boom')]


def test_reader_stops_at_a_truncated_record(tmp_path):
    import gzip, zdn_data
    good = zh.record(11, 5, 200, None, zh.reg_feats([1], [2], [([0], [[1]], 0, 1, 2)]))
    f = tmp_path / 'decisions-A.bin.gz'
    f.write_bytes(gzip.compress(good + good[:20]))
    assert zdn_data.read(f)['baby']['reg']['robot'].tolist() == [11]
