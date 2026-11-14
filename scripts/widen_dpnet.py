"""Widen a decision net to pass 3's Java points (docs/decision-points.md, "Pass 3: capability completion").

widen_dpnet.py --net runs/dp/net-v2/net.json --out OUT/net.json [--calib DIR|FILE ...] [--floor 0.005]
               [--active old|full] [--init-new zdb-hard|calibrated] [--points 1,5,...] [--seed 0]

For every point of the net with fewer outputs / inputs than the Java point (scripts/dp_points.py):
  * zero input columns are appended (old logits unchanged bit for bit);
  * new output rows get zero weights and bias median(max old logit) - ln(10 m / floor) over the calib records
    (-20 without), so a new option starts at ~0.1 floor / m of the probability but keeps a live gradient;
    BABY_ROLE's R_KEEP gets p99.9 + 8 over the review records (60 without), so reviews keep the role as zdb does.
  * --active old (default) writes "active" {point: old n_out}: build_dpnet.py then sets W.A and the bot samples only
    the old options -- the widened build replays the unwidened one exactly. --active full: all options live.
--init-new adds nets for the new points (8, 9, 14, 15, 16) that the net lacks: zdb-hard (option 0 = zdb's choice at
bias 0, the rest -30: replays zdb) or calibrated (rest -ln(10 m / floor)); the value points 9, 15 start uniform.
norm.npz beside the net is padded (and extended for new points) next to OUT. --calib: decisions directories
(*.dec.gz, recursively) for calibration and new-point normalisation.
"""
import argparse, gzip, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp_points, dpnet_io, decisions  # noqa: E402


def load_calib(paths, feats, max_rows=50000, max_files=12, seed=0):
    """{point: int32 features [N, n_feats]} from decision files (decisions-*.bin.gz; dirs are searched recursively, at
    most max_files files). Labels are ignored; records longer than the point's features (an RL trailer) are cut."""
    rng = np.random.default_rng(seed)
    files = []
    for d in paths:
        files += [Path(d)] if Path(d).is_file() else sorted(Path(d).rglob('decisions-*.bin.gz'))
    parts = {}
    for f in files[:max_files]:
        buf = gzip.decompress(f.read_bytes())
        raw = np.frombuffer(buf, dtype=np.uint8)
        for (pt, n), offs in decisions._scan(buf).items():
            nf = feats.get(pt)
            if nf is None or n < nf:
                continue
            o = np.asarray(offs, np.int64)
            idx = o[:, None] + decisions.HEAD + 2 * np.arange(nf)[None, :]
            x = (raw[idx].astype(np.int32) << 8) | raw[idx + 1]   # unsigned 16-bit chars, as DNet reads them
            parts.setdefault(pt, []).append(x)
    out = {}
    for p, xs in parts.items():
        X = np.concatenate(xs)
        if len(X) > max_rows:
            X = X[rng.choice(len(X), max_rows, replace=False)]
        out[p] = X
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--calib', nargs='*', default=[])
    ap.add_argument('--floor', type=float, default=0.005, help='the floor the widened net will run at')
    ap.add_argument('--active', default='old', help="old | full | P:N,...")
    ap.add_argument('--init-new', default='', choices=['', 'zdb-hard', 'calibrated'])
    ap.add_argument('--points', default='', help='only widen / add these points')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    opts, feats = dp_points.option_counts(), dp_points.feature_counts()
    raw = dpnet_io.load_raw(a.net)
    calib = load_calib(a.calib, feats, seed=a.seed) if a.calib else {}
    active = a.active if a.active in ('old', 'full') else {int(k): int(v) for k, v in (x.split(':') for x in a.active.split(',') if x)}
    points = [int(p) for p in a.points.split(',') if p] or None
    doc, norm, info = dpnet_io.widen(raw, opts, feats, active=active, calib=calib, floor=a.floor,
                                     init_new=a.init_new or None, seed=a.seed, points=points,
                                     norm=dpnet_io.load_norm(a.net))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))
    if norm:
        dpnet_io.save_norm(out.parent / 'norm.npz', norm)
    for p, i in sorted(info.items(), key=lambda kv: int(kv[0])):
        print(f'point {p}: {json.dumps(i)}' + (f' (calib {len(calib[int(p)])})' if int(p) in calib else ''))
    print(f'active {doc.get("active")}; wrote {out}')


if __name__ == '__main__':
    main()
