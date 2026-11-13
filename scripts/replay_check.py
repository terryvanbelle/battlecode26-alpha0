"""Check that bot X plays exactly the games bot REF plays (same opponents, maps, seeds, sides).

replay_check.py --bot zdp --ref zdb --opponents zdb,examplefuncsplayer --maps jail,pipes --seeds 21 --out runs/rc/NAME
Plays both through eval_student.py (resumable) and compares each game's trajectory hash and final stats.
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('rounds', 'winner', 'reason', 'traj_hash')
TEAM = ('cheese', 'transferred', 'cat_damage', 'kings', 'babies', 'points')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bot', required=True); ap.add_argument('--ref', required=True)
    ap.add_argument('--opponents', required=True); ap.add_argument('--maps', required=True)
    ap.add_argument('--seeds', default='21'); ap.add_argument('--sides', default='AB')
    ap.add_argument('--out', required=True); ap.add_argument('--workers', default='7')
    a = ap.parse_args()
    out = Path(a.out)
    for who in (a.ref, a.bot):
        subprocess.run([sys.executable, str(ROOT / 'scripts/eval_student.py'), '--player', who, '--opponents', a.opponents,
                        '--maps', a.maps, '--seeds', a.seeds, '--sides', a.sides, '--out', str(out / who),
                        '--workers', a.workers, '--limits'], check=True)
    same = diff = missing = 0
    for g in sorted((out / a.ref / 'games').iterdir()):
        r1, r2 = g / 'result.json', out / a.bot / 'games' / g.name / 'result.json'
        if not (r1.exists() and r2.exists()):
            missing += 1; continue
        x, y = json.loads(r1.read_text()), json.loads(r2.read_text())
        fx = [x.get(k) for k in KEYS] + [x[t][k] for t in 'AB' for k in TEAM]
        fy = [y.get(k) for k in KEYS] + [y[t][k] for t in 'AB' for k in TEAM]
        if fx == fy and x.get('traj_hash'):
            same += 1
        else:
            diff += 1
            print(f'DIFF {g.name}: {a.ref} r{x["rounds"]} {x["winner"]} xfer {x["A"]["transferred"]}/{x["B"]["transferred"]}'
                  f'  {a.bot} r{y["rounds"]} {y["winner"]} xfer {y["A"]["transferred"]}/{y["B"]["transferred"]}')
        for r in (x, y):
            if r.get('error') or r.get('hook_error') or r['A'].get('failures') or r['B'].get('failures'):
                print(f'  problem in {g.name}: {r.get("error")} {r.get("hook_error")} {r["A"].get("failures")} {r["B"].get("failures")}')
    print(f'REPLAY {a.bot} vs {a.ref}: identical {same}, different {diff}, missing {missing}')
    sys.exit(0 if diff == 0 and missing == 0 else 1)


if __name__ == '__main__':
    main()
