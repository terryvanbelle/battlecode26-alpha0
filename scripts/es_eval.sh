#!/bin/bash
# ES gates (docs/rl-es-plan.md section 2, item 6): a paired head-to-head of PLAYER vs OPPONENT on MAPS x SEEDS x AB
# (eval_student.py, resumable), then the score, its Wilson 95% CI and the exact-zero pair count (both seats gave the
# same traj_hash: the two bots played identically on that map and seed).
#   scripts/es_eval.sh PLAYER OPPONENT MAPS SEEDS OUT [WORKERS] [JSONL]
# PLAYER / OPPONENT: any eval_student.py name (build/botsrc packages compile on first use). MAPS, SEEDS comma lists.
# Prints one json line (also appended to JSONL when given). Exit 0 ok, 2 error. Never pass --limits.
export TZ='PDT+7'
cd "$(dirname "$0")/.." || exit 2
PY=${PY:-.venv/bin/python}
P=$1 O=$2 MAPS=$3 SEEDS=$4 OUT=$5 W=${6:-8} JL=$7
[ -n "$OUT" ] || { sed -n 2,7p "$0"; exit 2; }
$PY -u scripts/eval_student.py --player $P --opponents $O --maps $MAPS --seeds $SEEDS --sides AB --out $OUT --workers $W > $OUT.log 2>&1 || exit 2
J=$($PY - "$OUT" "$P" "$O" <<'EOF'
import json, math, sys, time
from pathlib import Path
out, p, o = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
pts = g = bad = 0
hashes = {}
for j in json.loads((out / 'jobs.json').read_text())['jobs']:
    r = out / 'games' / j['id'] / 'result.json'
    res = json.loads(r.read_text()) if r.exists() else None
    if res is None or res.get('error') or not res.get('finished') or res['A'].get('failures') or res['B'].get('failures'):
        bad += 1
        continue
    g += 1
    pts += 0.5 if res['winner'] is None else float(res['winner'] == j['side'])
    hashes.setdefault((j['opponent'], j['map'], j['seed']), {})[j['side']] = res.get('traj_hash')
ez = sum(1 for h in hashes.values() if len(h) == 2 and h['A'] is not None and h['A'] == h['B'])
s = pts / g if g else 0.0
z = 1.96
if g:
    den = 1 + z * z / g; c = (s + z * z / (2 * g)) / den; h = z * math.sqrt(s * (1 - s) / g + z * z / (4 * g * g)) / den
    lo, hi = c - h, c + h
else:
    lo = hi = 0.0
print(json.dumps(dict(player=p, opponent=o, out=str(out), points=pts, games=g, invalid=bad, score=round(s, 4),
                      lo=round(lo, 4), hi=round(hi, 4), exact_zero_pairs=ez, pairs=len(hashes), time=time.time())))
EOF
) || exit 2
echo "$J"
[ -n "$JL" ] && echo "$J" >> $JL
exit 0
