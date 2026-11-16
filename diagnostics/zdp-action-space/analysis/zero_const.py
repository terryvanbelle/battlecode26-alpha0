"""Zero the first-layer weights of input columns that were constant in a point's imitation data (box).

train_dp gives a constant column sd = 1 (mu = the constant), so its first-layer weights never received a gradient and
keep their random init; widen's added columns are (mu 0, sd 1) with zero weights. At the listed (frozen) points every
column with sd == 1 exactly is zeroed, and its contribution at x = mu moves into the bias, so the function is unchanged
on imitation-like inputs and no longer reacts to values the net never saw (e.g. the team rat-trap count, raider).
  zero_const.py NET_DIR OUT_DIR POINTS [CHECK_DATA_DIR]
"""
import sys, json, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, 'scripts')
import dpnet_io

src, dst, pts = Path(sys.argv[1]), Path(sys.argv[2]), [int(p) for p in sys.argv[3].split(',')]
chk = Path(sys.argv[4]) if len(sys.argv) > 4 else None
raw = json.loads((src / 'net.json').read_text())
import glob, decisions
files = sorted(glob.glob('/home/terry/projects/vibe/bc26a0/runs/teacher-dp2/games/g00[0-1]??/decisions-*.bin.gz'))[:40]
T = {p: [] for p in pts}
W = {p: 0 for p in pts}
for f in files:
    d = decisions.read(f)
    for p in pts:
        s_ = d['point'] == p
        if s_.any():
            T[p].append(d['feats'][s_]); W[p] = max(W[p], int(d['n'][s_].max()))
report = {}
for p in pts:
    layers = raw['points'][str(p)]
    l1 = layers[0]
    n_in, n_out = l1['n_in'], l1['n_out']
    w = np.array(l1['w'], np.float64).reshape(n_out, n_in)
    b = np.array(l1['b'], np.float64)
    t = np.concatenate([a[:, :W[p]] for a in T[p]]).astype(np.float64)
    tsd = t.std(0)
    mu = np.zeros(n_in); mu[:W[p]] = t.mean(0)
    cols = [j for j in range(min(W[p], n_in)) if tsd[j] < 1e-3 and np.abs(w[:, j]).max() > 0]
    wid = [j for j in range(W[p], n_in) if np.abs(w[:, j]).max() > 0]
    print(f'point {p}: teacher rows {len(t)} width {W[p]}, net n_in {n_in}; widened columns with nonzero weights: {wid}')
    x = None
    if chk is not None and (chk / f'p{p}.npz').exists():
        x = np.load(chk / f'p{p}.npz')['x']
    info = []
    for j in cols:
        b += w[:, j] * mu[j]
        share = float(np.mean(x[:, j] != mu[j])) if x is not None and j < x.shape[1] else None
        info.append(dict(col=j, mu=float(mu[j]), w_abs_mean=float(np.abs(w[:, j]).mean()), share_off=share))
        w[:, j] = 0.0
    l1['w'] = w.reshape(-1).tolist()
    l1['b'] = b.tolist()
    report[p] = info
    print(f'point {p}: zeroed {len(cols)} columns: ' + ', '.join(
        f"{d['col']}(mu {d['mu']:g}, |w| {d['w_abs_mean']:.3f}, off {d['share_off'] if d['share_off'] is None else round(d['share_off'], 4)})"
        for d in info))
raw.setdefault('meta', {})['zero_const'] = {str(k): [d['col'] for d in v] for k, v in report.items()}
dst.mkdir(parents=True, exist_ok=True)
(dst / 'net.json').write_text(json.dumps(raw))
shutil.copyfile(src / 'norm.npz', dst / 'norm.npz')  # unchanged (covers the trainable points only)
json.dump(report, open(dst / 'zero_const.json', 'w'), indent=1)
