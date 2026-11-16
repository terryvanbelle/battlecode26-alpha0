"""Plan-head drift (docs/rl-es-plan.md section 21): each net version's mean policy per plan head on a fixed set of states.

  plan_drift.py --run runs/rl/rl-p1 [--states runs/plan/states] [--from-iter 0] [--json OUT]

The states are the plan heads' records (p20..p27.npz, rl_data output) of one early iteration, copied once to --states
(from runs/rl/RUN/iter-NNNN/data when --states is empty), so every version is read on the same states. For each
nets/<v>/net.json the script prints, per head, the mean softmax over all options (temperature 1, no floor: the
policy rl-p1 samples), its mean bin and the share on zdb's bin. CORE (point 22) moving its mass toward bin 2 (12/8)
is the known answer rl-p1 is checked against.
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpnet_io  # noqa: E402

HEADS = {20: 'POSTURE', 21: 'ARMY', 22: 'CORE', 23: 'THRIFT', 24: 'ROLES', 25: 'RUSH', 26: 'LOAD', 27: 'KINGS'}
ZDB = {20: 0, 21: 3, 22: 4, 23: 3, 24: 3, 25: 3, 26: 3, 27: 1}   # Plan.ZDB by point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--states', default='runs/plan/states')
    ap.add_argument('--from-iter', type=int, default=0, help='iteration whose records seed --states')
    ap.add_argument('--json', help='write the table here as JSON lines (one per version)')
    a = ap.parse_args()
    run, st = Path(a.run), Path(a.states)
    if not any(st.glob('p2*.npz')):
        src = run / f'iter-{a.from_iter:04d}' / 'data'
        files = sorted(src.glob('p2[0-7].npz'))
        if not files:
            raise SystemExit(f'no plan-head records in {src}')
        st.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, st / f.name)
    X = {}
    for p in HEADS:
        f = st / f'p{p}.npz'
        if f.exists():
            X[p] = np.load(f)['x'].astype(np.float32)
    vers = sorted(int(d.name) for d in (run / 'nets').iterdir() if d.name.isdigit() and (d / 'net.json').exists())
    rows = []
    for v in vers:
        net = dpnet_io.load_net(run / 'nets' / str(v) / 'net.json')
        row = dict(version=v, heads={})
        for p, x in X.items():
            if p not in net:
                continue
            z = dpnet_io.forward(net[p], x).astype(np.float64)
            z -= z.max(1, keepdims=True)
            q = np.exp(z); q /= q.sum(1, keepdims=True)
            m = q.mean(0)
            row['heads'][HEADS[p]] = dict(p=[round(float(t), 3) for t in m], mean_bin=round(float((m * np.arange(len(m))).sum()), 3),
                                          p_zdb=round(float(m[ZDB[p]]), 3), rows=len(x))
        rows.append(row)
    names = [HEADS[p] for p in X]
    print('version  ' + '  '.join(f'{n:>14}' for n in names) + '   (mean bin / p at zdb bin)')
    for r in rows:
        print(f"v{r['version']:<7} " + '  '.join(f"{r['heads'][n]['mean_bin']:6.2f} / {r['heads'][n]['p_zdb']:.2f}" if n in r['heads'] else f"{'':>14}" for n in names))
    last = rows[-1] if rows else None
    if last and 'CORE' in last['heads']:
        print(f"CORE v{last['version']} p by bin (8/4 10/6 12/8 15/10 18/12* 21/15 24/18): {last['heads']['CORE']['p']}")
    if a.json:
        Path(a.json).write_text(''.join(json.dumps(r) + '\n' for r in rows))


if __name__ == '__main__':
    main()
