#!/bin/bash
# Strength check for a zdp variant (pass-2 behaviour changes must play equal or better than zdb; docs/decision-points.md).
#   scripts/strength_check.sh NAME "PUSH PATHS" [TAG]      (on the VM; games split with the Windows box)
# NAME vs zdb: 8 maps x 4 seeds x 2 sides = 64 games; NAME vs the held-out field + examplefuncsplayer: 8 maps x 2 seeds
# x 2 sides = 192 games. Reference: the same field jobs with zdp (= zdb), run once and reused. No bytecode limits.
export TZ='PDT+7'
cd ~/projects/vibe/bc26a0
PY=.venv/bin/python
source scripts/box_split.sh
BOXPY=.venv/bin/python
NAME=$1 PUSH=$2 TAG=${3:-$1}
ZMAPS=mincemeat,jail,rift,pipes,gridlock,thunderdome,sunandmoon,ZeroDay
GMAPS=tiny,DefaultSmall,DefaultMedium,DefaultLarge,closeup,knifefight,cheesefarm,wallsofparadis
HELD=VarunVejalla_final4_run_away10,GuzziFlipFlops_Hermesv8,quesswho_quals,daannte_v47_Skadoosh,AlexT101_usqualsbot,examplefuncsplayer
split_eval() {   # tag push out args...
  local tag=$1 push=$2 out=$3; shift 3
  box_request $tag $out "$push" $BOXPY -u scripts/eval_student.py "$@" --out $out --slice 1:4:4 --workers 10
  $PY -u scripts/eval_student.py "$@" --out $out --slice 0:1:4 > $out.vmpart.log 2>&1
  box_wait $tag
  $PY -u scripts/eval_student.py "$@" --out $out > $out.log 2>&1
}
mkdir -p runs/sc
echo "START $NAME $(date +%H:%M)"
split_eval sc-$TAG-zdb "$PUSH" runs/sc/$TAG-zdb --player $NAME --opponents zdb --maps $ZMAPS --seeds 31,32,33,34 --sides AB
[ -f runs/sc/ref-field.log ] || split_eval sc-ref-field "java" runs/sc/ref-field --player zdp --opponents $HELD --maps $GMAPS --seeds 11,12 --sides AB
split_eval sc-$TAG-field "$PUSH" runs/sc/$TAG-field --player $NAME --opponents $HELD --maps $GMAPS --seeds 11,12 --sides AB
$PY - "$TAG" <<'EOF'
import json, sys, glob, math
tag = sys.argv[1]
def score(d):
    w = n = 0; per = {}
    for g in glob.glob(f'runs/sc/{d}/games/*/result.json'):
        r = json.load(open(g)); name = g.split('/')[-2]
        opp = name.split('__')[0]; side = name.split('__')[-1]
        if r.get('winner') is None: continue
        win = r['winner'] == side
        w += win; n += 1; a = per.setdefault(opp, [0, 0]); a[0] += win; a[1] += 1
    return w, n, per
def ci(w, n):
    if not n: return (0, 0)
    p, z = w / n, 1.96; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
for d in (f'{tag}-zdb', f'{tag}-field', 'ref-field'):
    w, n, per = score(d); lo, hi = ci(w, n)
    print(f'{d:24s} {w:4d}/{n:<4d} {w / max(n, 1):.3f}  95% CI [{lo:.2f}, {hi:.2f}]  ' + ' '.join(f'{k[:8]}:{v[0]}/{v[1]}' for k, v in sorted(per.items())))
EOF
echo "DONE $NAME $(date +%H:%M)"
