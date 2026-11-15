"""Paired score of an eval_student.py head-to-head (es_eval.sh output dirs), pooled over several dirs (e.g. one per
host with disjoint seeds; a game played in two of the dirs is an error): the stage-B judge of docs/rl-es-plan.md section 5.

pair_score.py OUT [OUT ...] [--jsonl FILE]

A unit is (opponent, map, seed) with both seats played and valid; its score is the player's mean over the two games
(win 1, draw 0.5, loss 0). Prints one json line: units, games, points, score, the paired t of the unit scores against
0.5 (sd over units), the Wilson 95% interval of the game score, and the exact-zero units (both seats gave the same
traj_hash). Exit 0, or 2 when no unit is complete.
"""
import argparse, json, math, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outs', nargs='+')
    ap.add_argument('--jsonl')
    a = ap.parse_args()
    units, invalid, player = {}, 0, None
    for out in map(Path, a.outs):
        doc = json.loads((out / 'jobs.json').read_text())
        player = player or doc.get('player')
        for j in doc['jobs']:
            r = out / 'games' / j['id'] / 'result.json'
            res = json.loads(r.read_text()) if r.exists() else None
            if res is None or res.get('error') or not res.get('finished') or res['A'].get('failures') or res['B'].get('failures'):
                invalid += 1
                continue
            pts = 0.5 if res['winner'] is None else float(res['winner'] == j['side'])
            u = units.setdefault((j['opponent'], j['map'], j['seed']), {})
            if j['side'] in u:
                sys.exit(f"{out}: {j['id']} also played in another dir (pooled dirs need disjoint seeds or maps)")
            u[j['side']] = (pts, res.get('traj_hash'))
    full = [u for u in units.values() if len(u) == 2]
    if not full:
        print(json.dumps(dict(outs=a.outs, units=0, invalid=invalid)))
        sys.exit(2)
    s = [(u['A'][0] + u['B'][0]) / 2 for u in full]
    n = len(s)
    m = sum(s) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in s) / (n - 1)) if n > 1 else 0.0
    t = (m - 0.5) / (sd / math.sqrt(n)) if sd > 0 else (0.0 if m == 0.5 else math.copysign(math.inf, m - 0.5))
    g, pts = 2 * n, sum(s) * 2
    z = 1.96
    ph = pts / g
    c = (ph + z * z / (2 * g)) / (1 + z * z / g)
    w = z * math.sqrt(ph * (1 - ph) / g + z * z / (4 * g * g)) / (1 + z * z / g)
    ez = sum(1 for u in full if u['A'][1] is not None and u['A'][1] == u['B'][1])
    row = dict(outs=a.outs, player=player, units=n, games=g, points=pts, score=round(m, 4), t=round(t, 3),
               lo=round(c - w, 4), hi=round(c + w, 4), exact_zero_units=ez, incomplete_units=len(units) - n,
               invalid=invalid)
    line = json.dumps(row)
    print(line)
    if a.jsonl:
        with open(a.jsonl, 'a') as f:
            f.write(line + '\n')


if __name__ == '__main__':
    main()
