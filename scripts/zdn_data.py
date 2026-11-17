"""Read zdn records (docs/zdn.md, "Records") from decisions-{A,B}.bin.gz files into numpy arrays.

The file format is decisions-v1 (scripts/decisions.py): a 14-byte header (int32 robot, uint16 round, uint8 team,
uint8 king, uint16 point, uint16 label, uint16 n) and n uint16 features. zdn uses these points:

  REG  (200 baby, 210 king): one per sampled turn, before the robot acts. label unused. Features:
       VER, nS, nL, nR, S[nS], L[nL], then per register r: K_r, lab_r, D_r, NK_r, K_r rows of (kind, D_r features)
  STEP (201 baby, 211 king): one per primitive call of that turn, and one for END. label = index of the chosen
       option among the K (0xFFFF: the call matched no option). Features:
       VER, step, nC, D, K, NK, ctx[nC], K rows of (kind, D features)

  ERROR (299): an exception the bot caught (label = where: 1 player, 2-5 recorder start/pre/post/end); features are
       the exception's text.

S, ctx and row features are signed (stored + 32768); L is unsigned. lab_r is 0xFFFF when zdb's value matched no
option of register r. NK is the number of option kinds (the scorer's kind table size).

  zdn_data.py FILE [FILE ...]      # counts, option counts, label coverage per point and register
"""
import gzip, sys
from pathlib import Path
import numpy as np

HEAD = 14
NONE = 0xFFFF
OFF = 32768
REG = {200: 'baby', 210: 'king'}
STEP = {201: 'baby', 211: 'king'}
ERROR = 299
VER = 1


def _scan(buf):
    off, L, out = 0, len(buf), []
    while off + HEAD <= L:
        n = (buf[off + 12] << 8) | buf[off + 13]
        if off + HEAD + 2 * n > L:
            break
        out.append((off, (buf[off + 8] << 8) | buf[off + 9], n))
        off += HEAD + 2 * n
    return out


def read(path, kinds=('baby', 'king')):
    """{unit: dict(reg=..., step=...)} for one file. reg: robot, round, S [N, nS] int32 (signed), L [N, nL] int32,
    regs: list per register of dict(kind [N, Kmax] int32 (-1 pad), x [N, Kmax, D] int32, lab [N] int32 (-1 none), nk).
    step: robot, round, step, label (-1 none), ctx [M, nC], kind [M, Kmax] (-1 pad), x [M, Kmax, D], nk."""
    buf = gzip.decompress(Path(path).read_bytes())
    recs = _scan(buf)
    out = {}
    for unit in kinds:
        rp = [p for p, u in REG.items() if u == unit][0]
        sp = [p for p, u in STEP.items() if u == unit][0]
        regs = [(o, n) for o, p, n in recs if p == rp]
        steps = [(o, n) for o, p, n in recs if p == sp]
        out[unit] = dict(reg=_read_reg(buf, regs), step=_read_step(buf, steps))
    return out


def errors(path):
    """[(robot, round, where, text)] of the file's ERROR records."""
    buf = gzip.decompress(Path(path).read_bytes())
    out = []
    for o, p, n in _scan(buf):
        if p == ERROR:
            robot, rnd, where = _hdr(buf, o)
            f = np.frombuffer(buf, dtype='>u2', count=n, offset=o + HEAD)
            out.append((robot, rnd, where, ''.join(map(chr, f))))
    return out


def _hdr(buf, o):
    robot = int.from_bytes(buf[o:o + 4], 'big', signed=True)
    rnd = int.from_bytes(buf[o + 4:o + 6], 'big')
    label = int.from_bytes(buf[o + 10:o + 12], 'big')
    return robot, rnd, label


def _read_reg(buf, regs):
    if not regs:
        return None
    robots, rounds, S, L, R = [], [], [], [], None
    for o, n in regs:
        f = np.frombuffer(buf, dtype='>u2', count=n, offset=o + HEAD).astype(np.int32)
        if f[0] != VER:
            continue
        nS, nL, nR = int(f[1]), int(f[2]), int(f[3])
        p = 4
        s = f[p:p + nS] - OFF; p += nS
        l = f[p:p + nL]; p += nL
        if R is None:
            R = [dict(kind=[], x=[], lab=[], nk=0) for _ in range(nR)]
        for r in range(nR):
            K, lab, D, nk = int(f[p]), int(f[p + 1]), int(f[p + 2]), int(f[p + 3]); p += 4
            R[r]['nk'] = nk
            rows = f[p:p + K * (1 + D)].reshape(K, 1 + D); p += K * (1 + D)
            R[r]['kind'].append(rows[:, 0])
            R[r]['x'].append(rows[:, 1:] - OFF)
            R[r]['lab'].append(-1 if lab == NONE else lab)
        robot, rnd, _ = _hdr(buf, o)
        robots.append(robot); rounds.append(rnd); S.append(s); L.append(l)
    regs_out = []
    for r in R or []:
        Kmax = max(len(k) for k in r['kind'])
        D = r['x'][0].shape[1]
        N = len(r['kind'])
        kind = np.full((N, Kmax), -1, np.int32)
        x = np.zeros((N, Kmax, D), np.int32)
        for i, (k, xx) in enumerate(zip(r['kind'], r['x'])):
            kind[i, :len(k)] = k
            x[i, :len(k)] = xx
        regs_out.append(dict(kind=kind, x=x, lab=np.array(r['lab'], np.int32), nk=r['nk']))
    return dict(robot=np.array(robots, np.int32), round=np.array(rounds, np.int32), S=np.array(S, np.int32),
                L=np.array(L, np.int32), regs=regs_out)


def _read_step(buf, steps):
    if not steps:
        return None
    robots, rounds, stepi, labels, ctx, kinds, xs, nk = [], [], [], [], [], [], [], 0
    for o, n in steps:
        f = np.frombuffer(buf, dtype='>u2', count=n, offset=o + HEAD).astype(np.int32)
        if f[0] != VER:
            continue
        st, nC, D, K, nk = int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5])
        p = 6
        c = f[p:p + nC] - OFF; p += nC
        rows = f[p:p + K * (1 + D)].reshape(K, 1 + D)
        robot, rnd, label = _hdr(buf, o)
        robots.append(robot); rounds.append(rnd); stepi.append(st); labels.append(-1 if label == NONE else label)
        ctx.append(c); kinds.append(rows[:, 0]); xs.append(rows[:, 1:] - OFF)
    M = len(kinds)
    Kmax = max(len(k) for k in kinds)
    D = xs[0].shape[1]
    kind = np.full((M, Kmax), -1, np.int32)
    x = np.zeros((M, Kmax, D), np.int32)
    for i, (k, xx) in enumerate(zip(kinds, xs)):
        kind[i, :len(k)] = k
        x[i, :len(k)] = xx
    return dict(robot=np.array(robots, np.int32), round=np.array(rounds, np.int32), step=np.array(stepi, np.int32),
                label=np.array(labels, np.int32), ctx=np.array(ctx, np.int32), kind=kind, x=x, nk=nk)


def main():
    for f in sys.argv[1:]:
        err = errors(f)
        if err:
            print(f'{f}: {len(err)} error records; first: {err[0]}')
        d = read(f)
        for unit, v in d.items():
            r, s = v['reg'], v['step']
            if r is None and s is None:
                continue
            print(f'{f} {unit}: turns {0 if r is None else len(r["robot"])}, steps {0 if s is None else len(s["robot"])}')
            if r is not None:
                for i, g in enumerate(r['regs']):
                    nopt = (g['kind'] >= 0).sum(1)
                    lab = g['lab']
                    k = np.where(lab >= 0, g['kind'][np.arange(len(lab)), np.maximum(lab, 0)], -1)
                    kc = np.bincount(k[k >= 0], minlength=1)
                    print(f'  register {i}: options mean {nopt.mean():.1f} max {nopt.max()}, labelled {np.mean(lab >= 0):.3f}; '
                          f'chosen kinds {dict((j, int(c)) for j, c in enumerate(kc) if c)}')
            if s is not None:
                nopt = (s['kind'] >= 0).sum(1)
                lab = s['label']
                k = np.where(lab >= 0, s['kind'][np.arange(len(lab)), np.maximum(lab, 0)], -1)
                kc = np.bincount(k[k >= 0], minlength=1)
                print(f'  steps: options mean {nopt.mean():.1f} max {nopt.max()}, labelled {np.mean(lab >= 0):.3f}; '
                      f'chosen kinds {dict((i, int(c)) for i, c in enumerate(kc) if c)}')


if __name__ == '__main__':
    main()
