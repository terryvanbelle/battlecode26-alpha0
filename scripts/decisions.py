"""Read decision records (decisions-v1, see java/student/alpha0/student/Decisions.java).

Record (gzip, big-endian): int32 robot, uint16 round, uint8 team, uint8 king, uint16 point, uint16 label,
uint16 n, then n uint16 features.

RL builds (zdp D.RL, docs/rl-plan.md section 4) append a sampling trailer to the features, counted in n:
    k (0xFFFF: no pick), q = min(65535, round(-ln p_k * 4096)), flags (bit0 fallback, bit1 secondary slot),
    ceil(nOpt/16) mask words (bit i of word w: option 16w + i allowed), NET_VERSION, trailerLen = 5 + ceil(nOpt/16).
split_trailer() splits one record's features; read_rl() reads a whole file's trailer records (vectorised).

    python3 scripts/decisions.py FILE [FILE ...]   # per-point counts and label histograms (+ trailer stats)
"""
import gzip, sys
from collections import Counter
from pathlib import Path
try:
    import numpy as np
except ImportError:   # split_trailer works without numpy (tests/test_trailer.py)
    np = None

HEAD = 14
NONE = 0xFFFF
SECONDARY = (6, 7)    # BABY_COMMS, MINE_PICK: recorded from D's secondary slot (flags bit1)
MAX_WORDS = 8         # sanity bound on mask words (BABY_MOVE's 56 options need 4)


def _trailer_ok(L, n, flags, k, point):
    """True when a candidate trailer of length L (the record's last char) is self-consistent."""
    words = L - 5
    if not 0 <= words <= MAX_WORDS or L > n:
        return False
    if flags > 3 or bool(flags & 2) != (point in SECONDARY):
        return False
    return k == NONE or k < 16 * words


def split_trailer(feats_row, point):
    """(features, trailer) for one record's n features (any int sequence). trailer is None when the record has no
    valid RL trailer (a non-RL build), else dict(k (0xFFFF: none), q, logq = -q/4096, flags, fallback, secondary,
    words (mask words), mask (list of bool, 16 * words long: option i allowed), version, length)."""
    row = [int(v) for v in feats_row]
    n = len(row)
    if n < 5:
        return row, None
    L = row[-1]
    if L < 5 or L > n:
        return row, None
    t = row[n - L:]
    k, q, flags = t[0], t[1], t[2]
    if not _trailer_ok(L, n, flags, k, point):
        return row, None
    words = t[3:L - 2]
    mask = [bool(w >> i & 1) for w in words for i in range(16)]
    return row[:n - L], dict(k=k, q=q, logq=-q / 4096.0, flags=flags, fallback=bool(flags & 1),
                             secondary=bool(flags & 2), words=len(words), mask=mask, version=t[L - 2], length=L)


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


def _scan(buf):
    """Record offsets grouped by (point, n)."""
    off, L, groups = 0, len(buf), {}
    while off + HEAD <= L:
        n = (buf[off + 12] << 8) | buf[off + 13]
        if off + HEAD + 2 * n > L:
            break   # truncated tail
        pt = (buf[off + 8] << 8) | buf[off + 9]
        groups.setdefault((pt, n), []).append(off)
        off += HEAD + 2 * n
    return groups


def read_rl(path, nopt=None):
    """{point: dict(x int32 [N, nf], k int32 (0xFFFF: none), logq float32 (-q/4096 = log p_k, 0 when none),
    flags int32, mask bool [N, nOpt], version int32, robot, round, team, label (int32 [N]), off int64 [N] (byte
    offset: file order))} for one decisions file. Only records with a valid trailer are kept (a non-RL file gives {};
    a (point, n) group counts as trailed only when most of its records pass the trailer checks).
    nOpt is nopt[point] when given (the net's option count: extra columns dropped, missing ones False), else 16 x
    the most mask words seen. A null Java mask is all ones in every word, so pass nopt (or trim to the net's
    n_out) before using the mask. Records with no sample (k none) have no mask words: an all-False row."""
    buf = gzip.decompress(Path(path).read_bytes())
    raw = np.frombuffer(buf, dtype=np.uint8)
    per = {}
    for (pt, n), offs in _scan(buf).items():
        if n < 5:
            continue
        o = np.asarray(offs, dtype=np.int64)

        j = o + HEAD + 2 * (n - 1)
        L = (raw[j].astype(np.int32) << 8) | raw[j + 1]   # the last char: trailerLen in an RL record
        okL = (L >= 5) & (L <= min(n, 5 + MAX_WORDS))
        found = []
        for Lv in np.unique(L[okL]).tolist():
            sel = o[L == Lv]
            if not len(sel):
                continue
            nf, words = n - Lv, Lv - 5
            idx = sel[:, None] + HEAD + 2 * np.arange(n)[None, :]
            row = (raw[idx].astype(np.int32) << 8) | raw[idx + 1]
            t = row[:, nf:]
            k, q, flags = t[:, 0], t[:, 1], t[:, 2]
            good = (flags <= 3) & (((flags & 2) != 0) == (pt in SECONDARY)) & ((k == NONE) | (k < 16 * words))
            if not good.any():
                continue
            row, t, sel = row[good], t[good], sel[good]
            mw = t[:, 3:3 + words]
            mask = ((mw[:, :, None] >> np.arange(16)[None, None, :]) & 1).astype(bool).reshape(len(sel), 16 * words)
            head = lambda a, b, dt: raw[sel[:, None] + np.arange(a, b)].copy().view(dt).reshape(len(sel)).astype(np.int32)
            found.append(dict(
                x=row[:, :nf], k=t[:, 0], logq=(-t[:, 1] / 4096.0).astype(np.float32), flags=t[:, 2], mask=mask,
                version=t[:, Lv - 2], robot=head(0, 4, '>i4'), round=head(4, 6, '>u2'), team=head(6, 7, 'u1'),
                label=head(10, 12, '>u2'), off=sel))
        # an RL build trails every record; in a non-RL file a last feature can pass the checks only by chance
        if sum(len(f['k']) for f in found) * 2 > len(o):
            per.setdefault(pt, []).extend(found)
    out = {}
    for pt, parts in per.items():
        nf = max(p['x'].shape[1] for p in parts)
        no = nopt[pt] if nopt and pt in nopt else max(p['mask'].shape[1] for p in parts)

        def fitw(a, w):
            return a[:, :w] if a.shape[1] >= w else np.pad(a, ((0, 0), (0, w - a.shape[1])))

        d = {key: np.concatenate([p[key] for p in parts]) for key in ('k', 'logq', 'flags', 'version', 'robot',
                                                                       'round', 'team', 'label', 'off')}
        d['x'] = np.concatenate([fitw(p['x'], nf) for p in parts]).astype(np.int32)
        d['mask'] = np.concatenate([fitw(p['mask'], no) for p in parts])
        order = np.argsort(d['off'], kind='stable')
        out[pt] = {key: v[order] for key, v in d.items()}
    return out


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
    rl = [read_rl(p) for p in sys.argv[1:]]
    for pt in sorted({pt for r in rl for pt in r}):
        parts = [r[pt] for r in rl if pt in r]
        k = np.concatenate([p['k'] for p in parts]); fl = np.concatenate([p['flags'] for p in parts])
        lq = np.concatenate([p['logq'] for p in parts]); ver = np.concatenate([p['version'] for p in parts])
        picked = k != NONE
        print(f'trailer point {pt}: {len(k)} records, {int(picked.sum())} picked, fallback '
              f'{float((fl[picked] & 1).mean()) if picked.any() else 0:.3f}, mean log p '
              f'{float(lq[picked].mean()) if picked.any() else 0:.4f}, versions {sorted(set(ver.tolist()))}')


if __name__ == '__main__':
    main()
