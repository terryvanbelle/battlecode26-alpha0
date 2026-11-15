#!/usr/bin/env python3
"""Turn an es_loop run into the inputs of scripts/plot_rl_vs.py (no dependencies; runs where the run lives).

  scripts/es_progress.py runs/es/RUN OUTDIR [--start-gates runs/rl/R/gates.jsonl --start-version N]

OUTDIR/progress.jsonl: one row per finished iteration k, {"iter": k, "score": {opp: {"points", "games"}}}, from the
iteration's pool games (the candidates theta_k +- sigma*eps against each pool package; their average is a first-order
estimate of the centre theta_k). OUTDIR/gates.jsonl: the run's gates as {"mode": "anchor-<opponent>", "version",
"points", "games"}, version = the centre gated (gates/anchor-vV). --start-gates adds the start net's own gates as
version 0: the rows of that gates.jsonl at --start-version (rl_loop mode "anchor" is the config's anchor_package,
named by --start-anchor, default zdpf005; "anchor-<opp>" rows keep their opponent).
Invalid games (error, unfinished, robot failures) are skipped, as in es_eval.sh. Score: win 1, draw 0.5.
"""
import argparse, json, re
from pathlib import Path


def ok(res):
    return res and not res.get('error') and res.get('finished') and not res['A'].get('failures') \
        and not res['B'].get('failures')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run'); ap.add_argument('out')
    ap.add_argument('--start-gates'); ap.add_argument('--start-version', type=int)
    ap.add_argument('--start-anchor', default='zdpf005')
    a = ap.parse_args()
    rd, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in sorted(rd.glob('iter-[0-9][0-9][0-9][0-9]')):
        if not (it / '.done').exists() and not (it / 'update.json').exists():
            continue   # still collecting
        score = {}
        for j in json.loads((it / 'jobs.json').read_text())['jobs']:
            if j.get('kind') != 'pool':
                continue
            r = it / 'games' / j['id'] / 'result.json'
            res = json.loads(r.read_text()) if r.exists() else None
            if not ok(res):
                continue
            s = score.setdefault(j['opponent'], {'points': 0.0, 'games': 0})
            s['games'] += 1
            s['points'] += 0.5 if res['winner'] is None else float(res['winner'] == j['theta_side'])
        rows.append({'iter': int(it.name[5:]), 'score': score})
    (out / 'progress.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    gates = []
    if a.start_gates:
        for line in open(a.start_gates):
            r = json.loads(line)
            if r.get('version') == a.start_version and r.get('games'):
                mode = r['mode'] if r['mode'] != 'anchor' else 'anchor-' + a.start_anchor
                gates.append({'mode': mode, 'version': 0, 'points': r['points'], 'games': r['games']})
    if (rd / 'gates.jsonl').exists():
        for line in open(rd / 'gates.jsonl'):
            r = json.loads(line)
            m = re.search(r'anchor(?:-[^/]*)?-v(\d+)$', r.get('out', ''))
            if m and r.get('games'):
                gates.append({'mode': 'anchor-' + r['opponent'], 'version': int(m.group(1)),
                              'points': r['points'], 'games': r['games']})
    (out / 'gates.jsonl').write_text(''.join(json.dumps(g) + '\n' for g in gates))
    print(f'{out}: {len(rows)} iterations, {len(gates)} gates')


if __name__ == '__main__':
    main()
