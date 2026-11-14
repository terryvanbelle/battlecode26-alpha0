"""RL trailer round trip: a record built in zdp's wire format (java/bots/zdp/D.java emit + trailer) goes through the
harness's file format (Decisions.take) and back through scripts/decisions.py (split_trailer; read_rl when numpy is
installed). Runs without numpy (the read_rl cases are then skipped).

    python3 tests/test_trailer.py                      # the synthetic round trip
    python3 tests/test_trailer.py decisions-A.bin.gz   # also: every record of a real RL-build file parses
"""
import gzip, math, os, random, struct, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import decisions as DEC  # noqa: E402

MAGIC = 0xDEC0
FILES = []   # real decisions files from the command line


def f32(x):
    return struct.unpack('>f', struct.pack('>f', x))[0]


def java_trailer(k, p, n_opt, mask, flags, version):
    """D.trailer's chars (k < 0: none; mask None: all allowed; p a float32 value)."""
    words = (n_opt + 15) >> 4
    out = [0xFFFF if k < 0 else k]
    q = 0 if k < 0 else (65535 if p <= 0 else math.floor(-math.log(p) * 4096 + 0.5))   # Math.round
    out.append(max(0, min(65535, q)))
    out.append(flags)
    for w in range(words):
        v = 0
        if mask is None:
            v = 0xFFFF
        else:
            for i in range(16):
                o = 16 * w + i
                if o < n_opt and o < len(mask) and mask[o]:
                    v |= 1 << i
        out.append(v)
    out += [version, 5 + words]
    return out


def java_string(point, label, feats, trailer):
    """The indicator string D.emit sends (as a list of chars): MAGIC, point, label, n, features + trailer."""
    body = list(feats) + list(trailer)
    return [MAGIC, point, 0xFFFF if label < 0 else label, len(body)] + body


def harness_record(s, robot=7, rnd=123, team=0, king=0):
    """Decisions.take: the file record for indicator string s (None when the harness would reject it)."""
    if len(s) < 4 or len(s) != 4 + s[3]:
        return None
    n = s[3]
    return struct.pack('>ihBBHHH', robot, rnd, team, king, s[1], s[2], n) + struct.pack(f'>{n}H', *s[4:4 + n])


CASES = [
    # point, label, n features, k, p, nOpt, mask, flags, version
    (1, 0x8000 | (40 << 6) | 23, 64, 40, 0.0123, 56, [i % 3 != 0 for i in range(56)], 1, 7),   # BABY_MOVE fallback
    (1, 12, 64, 12, 0.9991, 56, [True] * 56, 0, 7),
    (5, 2, 32, 2, 1.0, 5, None, 0, 0),                     # BABY_ROLE, no mask, p = 1
    (6, 0, 37, 0, 0.5, 13, [True] * 13, 2, 65535),          # BABY_COMMS (secondary slot)
    (7, 3, 45, 3, 0.004, 40, [i < 9 for i in range(40)], 2, 3),   # MINE_PICK (secondary)
    (11, 21, 33, 21, 0.25, 22, None, 0, 1),                 # KING_SPAWN
    (13, 18, 29, -1, 1.0, 30, [False] * 30, 0, 1),          # KING_FORM: nothing allowed (k none, mask words kept)
    (12, 0, 30, -1, 1.0, 0, None, 0, 1),                    # KING_ACT: no sample at all (no mask words)
    (10, 37, 40, 36, 1e-30, 37, [True] * 37, 1, 2),         # KING_MOVE: tiny p -> q clamps to 65535
]


def build(case, rng):
    point, label, nf, k, p, n_opt, mask, flags, version = case
    feats = [rng.randrange(0, 65536) for _ in range(nf - 1)] + [rng.choice([20, 30, 60])]   # G.H last, like zdp
    p = f32(p)
    tr = java_trailer(k, p, n_opt, mask, flags, version)
    return feats, tr, java_string(point, label, feats, tr)


class TrailerRoundTrip(unittest.TestCase):
    def test_split(self):
        rng = random.Random(1)
        for case in CASES:
            point, label, nf, k, p, n_opt, mask, flags, version = case
            feats, tr, s = build(case, rng)
            self.assertLessEqual(len(s), 96, 'record must fit D.F (96 chars)')
            self.assertIsNotNone(harness_record(s))
            got, t = DEC.split_trailer(s[4:], point)
            self.assertEqual(got, feats)
            self.assertIsNotNone(t)
            self.assertEqual(t['k'], 0xFFFF if k < 0 else k)
            self.assertEqual(t['flags'], flags)
            self.assertEqual(t['version'], version)
            self.assertEqual(t['length'], 5 + (n_opt + 15) // 16)
            if k >= 0:
                if p > 1e-20:
                    self.assertLess(abs(t['logq'] - math.log(f32(p))), 0.5 / 4096 + 1e-9)
                else:
                    self.assertEqual(t['q'], 65535)
            else:
                self.assertEqual(t['q'], 0)
            want = [True] * n_opt if mask is None else list(mask)
            self.assertEqual(t['mask'][:n_opt], want)

    def test_no_trailer(self):
        """v1 records (no trailer) come back unchanged."""
        rng = random.Random(2)
        for point, nf in ((1, 64), (2, 53), (5, 32), (6, 37), (10, 40)):
            feats = [rng.randrange(0, 2) for _ in range(nf - 1)] + [rng.choice([20, 30, 60])]
            got, t = DEC.split_trailer(feats, point)
            self.assertIsNone(t)
            self.assertEqual(got, feats)

    @unittest.skipIf(DEC.np is None, 'numpy not installed')
    def test_read_rl(self):
        np = DEC.np
        rng = random.Random(3)
        recs, want = [], []
        for rep in range(5):
            for case in CASES:
                feats, tr, s = build(case, rng)
                recs.append(harness_record(s, robot=100 + rep, rnd=rep, team=rep & 1))
                want.append((case, feats, rep))
        # a v1 (no trailer) record of another point in the same file is skipped
        recs.append(harness_record([MAGIC, 4, 0, 31] + [1] * 30 + [30]))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'decisions-A.bin.gz')
            Path(path).write_bytes(gzip.compress(b''.join(recs)))
            got = DEC.read_rl(path)
            got_n = DEC.read_rl(path, nopt={1: 56, 7: 40})
        self.assertNotIn(4, got)
        for point in {c[0] for c in CASES}:
            rows = [w for w in want if w[0][0] == point]
            g = got[point]
            self.assertEqual(len(g['k']), len(rows))
            for i, (case, feats, rep) in enumerate(rows):
                _, label, nf, k, p, n_opt, mask, flags, version = case
                self.assertEqual(g['x'][i, :nf].tolist(), feats)
                self.assertEqual(int(g['k'][i]), 0xFFFF if k < 0 else k)
                self.assertEqual(int(g['flags'][i]), flags)
                self.assertEqual(int(g['version'][i]), version)
                self.assertEqual(int(g['robot'][i]), 100 + rep)
                self.assertEqual(int(g['round'][i]), rep)
                self.assertEqual(int(g['team'][i]), rep & 1)
                self.assertEqual(int(g['label'][i]), label)
                one = DEC.split_trailer(g['x'][i, :nf].tolist() + java_trailer(k, f32(p), n_opt, mask, flags, version), point)[1]
                self.assertAlmostEqual(float(g['logq'][i]), one['logq'], places=6)
                if n_opt:
                    exp = [True] * n_opt if mask is None else list(mask)
                    self.assertEqual(g['mask'][i, :n_opt].tolist(), exp)
        self.assertEqual(got_n[1]['mask'].shape[1], 56)
        self.assertEqual(got_n[7]['mask'].shape[1], 40)

    def test_real_files(self):
        """Every record of an RL-build file carries a trailer that split_trailer and read_rl agree on."""
        if DEC.np is None or not FILES:
            self.skipTest('no decisions files given (or no numpy)')
        for path in FILES:
            v1 = DEC.read(path)
            rl = DEC.read_rl(path)
            total = sum(len(d['k']) for d in rl.values())
            self.assertEqual(total, len(v1['point']), f'{path}: {total} of {len(v1["point"])} records trailed')
            for i in range(min(2000, len(v1['point']))):
                pt, n = int(v1['point'][i]), int(v1['n'][i])
                _, t = DEC.split_trailer(v1['feats'][i, :n].tolist(), pt)
                self.assertIsNotNone(t, f'{path} record {i} point {pt}')


if __name__ == '__main__':
    FILES[:] = [a for a in sys.argv[1:] if not a.startswith('-')]
    unittest.main(argv=[sys.argv[0]] + [a for a in sys.argv[1:] if a.startswith('-')])
