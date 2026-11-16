"""Stage-0 report (docs/rl-es-plan.md section 19): each variant's score vs zdb overall and on the small and large halves
of the row's maps (split at the median area), from the judge's game results on the box.

  python3 diagnostics/zdp-action-space/stage0/s0_report.py [--json OUT]     (run in ~/rltest/bc26a0 on the box)

A game scores 1 for a win by the variant, 0 for a loss, 0.5 otherwise. Units are (map, seed) pairs of both seats, as
pair_score.py counts them; t is the mean unit score's distance from 0.5 over its standard error.
"""
import argparse, glob, json, math, sys
from pathlib import Path

sys.path.insert(0, 'scripts')
from rl_data import map_area  # noqa: E402

HERE = Path(__file__).resolve().parent


def unit_stats(units):
    n = len(units)
    if n < 2:
        return dict(n=n, score=None, t=None)
    m = sum(units) / n
    var = sum((u - m) ** 2 for u in units) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else float('nan')
    return dict(n=n, score=round(m, 4), t=round((m - 0.5) / se, 2) if se == se and se > 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    variants = [l.rstrip('\n').split('\t') for l in (HERE / 'variants.tsv').read_text().splitlines()[1:] if l.strip()]
    rows, areas = [], {}
    for name, grp, label, cfg in variants:
        run = f's0{name[4:]}'
        games = {}
        for f in glob.glob(f'runs/cpi/{run}/*/games/*/result.json'):
            j = json.loads(Path(f).read_text())
            side = Path(f).parent.name.rsplit('__', 1)[1]
            g = 1.0 if j.get('winner') == side else 0.0 if j.get('winner') in ('A', 'B') else 0.5
            games.setdefault((j['map'], j['seed']), []).append(g)
            if j['map'] not in areas:
                areas[j['map']] = map_area(j['map'])
        rows.append((name, grp, label, games))
    if not areas:
        raise SystemExit('no results found (run in ~/rltest/bc26a0 on the box)')
    med = sorted(areas.values())[len(areas) // 2]
    out = []
    for name, grp, label, games in rows:
        units = {k: sum(v) / len(v) for k, v in games.items() if len(v) == 2}
        allu = list(units.values())
        small = [u for (m, s), u in units.items() if areas[m] < med]
        large = [u for (m, s), u in units.items() if areas[m] >= med]
        r = dict(name=name, group=grp, label=label, all=unit_stats(allu), small=unit_stats(small), large=unit_stats(large))
        if r['small']['score'] is not None and r['large']['score'] is not None:
            r['split'] = round(r['large']['score'] - r['small']['score'], 4)
        out.append(r)
    print(f'median map area {med} ({len(areas)} maps)')
    print(f"{'variant':8} {'group':8} {'label':22} {'n':>3} {'score':>6} {'t':>6} | {'small':>6} {'t':>6} | {'large':>6} {'t':>6} | split")
    for r in out:
        A, S, L = r['all'], r['small'], r['large']
        f = lambda v: '' if v is None else v
        print(f"{r['name']:8} {r['group']:8} {r['label'][:22]:22} {A['n']:>3} {f(A['score']):>6} {f(A['t']):>6} | "
              f"{f(S['score']):>6} {f(S['t']):>6} | {f(L['score']):>6} {f(L['t']):>6} | {r.get('split', '')}")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(median_area=med, areas=areas, variants=out), indent=1))


if __name__ == '__main__':
    main()
