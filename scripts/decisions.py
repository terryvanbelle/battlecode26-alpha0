"""Read decision records (decisions-v1, see java/student/alpha0/student/Decisions.java).

Record (gzip, big-endian): int32 robot, uint16 round, uint8 team, uint8 king, uint16 point, uint16 label,
uint16 n, then n uint16 features.

    python3 scripts/decisions.py FILE [FILE ...]   # per-point counts and label histograms
"""
import gzip, sys
from collections import Counter
from pathlib import Path
import numpy as np

HEAD = 14


def read(path) -> dict:
    buf = gzip.decompress(Path(path).read_bytes())
    off, starts, ns = 0, [], []
    while off + HEAD <= len(buf):
        n = int.from_bytes(buf[off + 12:off + 14], 'big')
        if off + HEAD + 2 * n > len(buf):
            break   # truncated tail
        starts.append(off); ns.append(n)
        off += HEAD + 2 * n
    N, maxn = len(starts), max(ns, default=0)
    starts, n = np.array(starts, dtype=np.int64), np.array(ns, dtype=np.int32)
    raw = np.frombuffer(buf, dtype=np.uint8)

    def field(o, k, dt):
        idx = starts[:, None] + o + np.arange(k)
        return raw[idx].copy().view(dt).reshape(N).astype(np.int32) if N else np.zeros(0, np.int32)

    feats = np.zeros((N, maxn), dtype=np.int32)
    for i, (s, k) in enumerate(zip(starts, ns)):
        if k:
            feats[i, :k] = np.frombuffer(buf, dtype='>u2', count=k, offset=int(s) + HEAD)
    return dict(robot=field(0, 4, '>i4'), round=field(4, 2, '>u2'), team=field(6, 1, 'u1'), king=field(7, 1, 'u1'),
                point=field(8, 2, '>u2'), label=field(10, 2, '>u2'), n=n, feats=feats)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    d = [read(p) for p in sys.argv[1:]]
    cat = {k: np.concatenate([x[k] for x in d]) for k in ('robot', 'round', 'king', 'point', 'label', 'n')}
    print(f'{len(cat["point"])} records from {len(d)} file(s)')
    for pt in sorted(set(cat['point'].tolist())):
        sel = cat['point'] == pt
        hist = Counter(cat['label'][sel].tolist())
        ns = cat['n'][sel]
        print(f'point {pt}: {sel.sum()} records (king {int(cat["king"][sel].sum())}), n {ns.min()}..{ns.max()}, '
              f'robots {len(set(cat["robot"][sel].tolist()))}')
        print('   labels: ' + ', '.join(f'{l}:{c}' for l, c in sorted(hist.items())))


if __name__ == '__main__':
    main()
