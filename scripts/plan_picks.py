"""Plan-head picks per team-game, and their effect on the result (docs/rl-es-plan.md section 21, rl-p2's readings).

  plan_picks.py --run runs/rl/rl-p2 [--out runs/rl/rl-p2/plan_picks.jsonl] [--report] [--iters 1-10] [--summary]

Collect (always): for every iter-NNNN/data holding plan-head records (p20..p27.npz) whose iteration is not yet in
--out, append one row per recorded team-game: iteration, game, team, version, map, game rounds, result z (+1/-1) and,
per head, its picks as [round, bin]. Runs before the loop prunes an iteration's data.

--report: per head, pooled over --iters, a least-squares fit of z on the share of the game spent under each bin
(each pick holds from its round to the head's next pick or the game's end), zdb's bin the baseline, so a bin's
coefficient is the result difference between a whole game under that bin and one under zdb's (in brackets: t and
the bin's exposure in games' worth; bins under 3 games are left out). Also the pre-set
reading for CORE: z on the share of the game spent below zdb's bin (fewer core babies), slope and t.
--summary prints that CORE reading as one line (for rl_chart_sync.sh's EXTRA).
"""
import argparse, json
from pathlib import Path
import numpy as np

HEADS = {20: 'POSTURE', 21: 'ARMY', 22: 'CORE', 23: 'THRIFT', 24: 'ROLES', 25: 'RUSH', 26: 'LOAD', 27: 'KINGS'}
ZDB = {20: 0, 21: 3, 22: 4, 23: 3, 24: 3, 25: 3, 26: 3, 27: 1}   # Plan.ZDB by point
K = {20: 7, 21: 7, 22: 7, 23: 7, 24: 7, 25: 6, 26: 7, 27: 4}


def collect(run, out):
    done = set()
    if out.exists():
        done = {json.loads(l)['iter'] for l in out.open()}
    added = 0
    for d in sorted(run.glob('iter-*/data')):
        it = int(d.parent.name.split('-')[1])
        if it in done or not (d / 'p22.npz').exists() or not (d / 'games.jsonl').exists():
            continue
        games = {}
        for l in (d / 'games.jsonl').open():
            g = json.loads(l)
            games[int(g['game'][1:])] = g
        rows = {}
        for p in HEADS:
            f = d / f'p{p}.npz'
            if not f.exists():
                continue
            z = np.load(f)
            for i in range(len(z['k'])):
                key = (int(z['game'][i]), int(z['team'][i]))
                r = rows.setdefault(key, dict(iter=it, game=key[0], team=key[1], version=int(z['version'][i]),
                                              z=float(z['z'][i]), picks={}))
                r['picks'].setdefault(str(p), []).append([int(z['round'][i]), int(z['k'][i])])
        with out.open('a') as fo:
            for key in sorted(rows):
                r = rows[key]
                g = games.get(key[0], {})
                r['map'], r['rounds'] = g.get('map'), g.get('rounds')
                for v in r['picks'].values():
                    v.sort()
                fo.write(json.dumps(r) + '\n')
        added += 1
    return added


def shares(picks, rounds, k):
    """Share of the game (from the first pick to the end) spent under each bin."""
    s = np.zeros(k)
    end = max(rounds or 0, picks[-1][0] + 1)
    for i, (r, b) in enumerate(picks):
        nxt = picks[i + 1][0] if i + 1 < len(picks) else end
        s[b] += max(nxt - r, 0)
    t = s.sum()
    return s / t if t > 0 else s


def fit(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ b
    dof = max(len(y) - X1.shape[1], 1)
    cov = np.linalg.pinv(X1.T @ X1) * (res @ res) / dof
    return b[1:], np.sqrt(np.clip(np.diag(cov)[1:], 1e-300, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--iters', help='A-B: pool these iterations (default: all)')
    a = ap.parse_args()
    run = Path(a.run)
    out = Path(a.out) if a.out else run / 'plan_picks.jsonl'
    n_new = collect(run, out)
    if not (a.report or a.summary):
        print(f'{out}: {n_new} iterations added')
        return
    rows = [json.loads(l) for l in out.open()]
    if a.iters:
        lo, hi = (int(x) for x in a.iters.split('-'))
        rows = [r for r in rows if lo <= r['iter'] <= hi]
    its = sorted({r['iter'] for r in rows})
    span = f'iterations {its[0]}-{its[-1]}' if its else 'no iterations'
    for p in HEADS:
        if a.summary and p != 22:
            continue
        rr = [r for r in rows if str(p) in r['picks']]
        if len(rr) < 20:
            continue
        S = np.array([shares(r['picks'][str(p)], r['rounds'], K[p]) for r in rr])
        y = np.array([r['z'] for r in rr])
        below = S[:, :ZDB[p]].sum(1)
        if p == 22:
            if below.std() > 0:
                (b1,), (se1,) = fit(below[:, None], y)
            else:
                b1, se1 = 0.0, float('inf')
            line = (f"CORE pick vs result, {span} ({len(rr)} team-games): a game under fewer core babies than zdb's "
                    f"18/12 scores {b1:+.2f} (t {b1 / se1:.2f}) in result units (±1)")
            if a.summary:
                print(line)
                return
            print(line)
        expo = S.sum(0)   # games' worth of exposure per bin
        keep = [b for b in range(K[p]) if b != ZDB[p] and expo[b] >= 3]
        coef, se = fit(S[:, keep], y)
        print(f"{HEADS[p]:<8} n {len(rr):5d}  zdb bin {ZDB[p]} ({expo[ZDB[p]]:.0f} games): " +
              '  '.join(f'b{b} {c:+.2f} (t {c / s:+.1f}, {expo[b]:.0f})' for b, c, s in zip(keep, coef, se)))


if __name__ == '__main__':
    main()
