#!/bin/bash
# RL gates (docs/rl-plan.md section 5): a thin wrapper over eval_student.py / strength_check.sh for zdpr<V>.
#   scripts/rl_gate.sh CONFIG RUN anchor   V   zdpr<V> vs the reference (zdpn2): ZMAPS x anchor_seeds x 2 sides (32), VM only
#   scripts/rl_gate.sh CONFIG RUN promote  V   ZMAPS x promote_seeds x 2 sides (128), split with the box; pass = Wilson lo > 0.50
#   scripts/rl_gate.sh CONFIG RUN strength V   strength_check.sh (64 vs zdb + 192 held-out field) and the field rule:
#                                              pass = field score >= the reference's field score - field_margin
# Run on the VM from anywhere. Games: runs/rl/RUN/gates/<mode>-v<V>/; verdict: runs/rl/RUN/gates/<mode>-v<V>.json
# (score, games, Wilson 95% CI, pass) plus a row in runs/rl/RUN/gates.jsonl. Exit 0 = pass, 1 = fail, 2 = error.
# The field never feeds back into training (PFSP, beta, staging); it can only stop a promotion (rule in rl_loop.sh).
export TZ='PDT+7'
cd "$(dirname "$0")/.." || exit 2
PY=.venv/bin/python
BOXPY=.venv/bin/python
source scripts/box_split.sh
CFG=$1 RUN=$2 MODE=$3 V=$4
[ -n "$V" ] || { sed -n 2,9p "$0"; exit 2; }
cfgv() { $PY -c "import json,sys; c=json.load(open('$CFG')); v=eval('c'+sys.argv[1]); print(','.join(map(str,v)) if isinstance(v,list) else v)" "$1"; }
PREFIX=$(cfgv "['package_prefix']") REF=$(cfgv "['anchor_package']")
ZMAPS=$(cfgv "['zmaps']")
NAME=$PREFIX$V
GD=runs/rl/$RUN/gates
mkdir -p $GD
OUT=$GD/$MODE-v$V

verdict() {   # dirs... -> json verdict on stdout (score = points / games over the given eval dirs' summary.json)
  $PY - "$MODE" "$V" "$@" <<'EOF'
import json, math, sys
mode, v, dirs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
pts = g = bad = 0
for d in dirs:
    s = json.load(open(f'{d}/summary.json'))['summary']
    for o in s.values():
        pts += o['points']; g += o['games']; bad += o['invalid']
p = pts / g if g else 0.0
z = 1.96
if g:
    den = 1 + z * z / g; c = (p + z * z / (2 * g)) / den; h = z * math.sqrt(p * (1 - p) / g + z * z / (4 * g * g)) / den
    lo, hi = c - h, c + h
else:
    lo = hi = 0.0
print(json.dumps(dict(mode=mode, version=v, points=pts, games=g, invalid=bad, score=round(p, 4), lo=round(lo, 4), hi=round(hi, 4))))
EOF
}

record() {   # json -> gate file + gates.jsonl
  echo "$1" > $OUT.json
  echo "$1" >> runs/rl/$RUN/gates.jsonl
  echo "GATE $(date +%H:%M) $1"
}

case $MODE in
  anchor)
    $PY -u scripts/eval_student.py --player $NAME --opponents $REF --maps $ZMAPS --seeds $(cfgv "['gates']['anchor_seeds']") \
      --sides AB --out $OUT --workers $(cfgv "['workers_vm']") > $OUT.log 2>&1 || exit 2
    J=$(verdict $OUT) || exit 2
    record "$($PY -c "import json,sys; d=json.loads(sys.argv[1]); d['pass']=d['score']>=$(cfgv "['gates']['promote_if_anchor']"); print(json.dumps(d))" "$J")"
    ;;
  promote)
    ARGS="--player $NAME --opponents $REF --maps $ZMAPS --seeds $(cfgv "['gates']['promote_seeds']") --sides AB"
    box_request rlgate-$RUN-$V $OUT "build/botsrc/$NAME" $BOXPY -u scripts/eval_student.py $ARGS --out $OUT --slice 1:4:4 --workers $(cfgv "['workers_box']")
    $PY -u scripts/eval_student.py $ARGS --out $OUT --slice 0:1:4 --workers $(cfgv "['workers_vm']") > $OUT.vmpart.log 2>&1
    box_wait rlgate-$RUN-$V
    $PY -u scripts/eval_student.py $ARGS --out $OUT --workers $(cfgv "['workers_vm']") > $OUT.log 2>&1 || exit 2
    J=$(verdict $OUT) || exit 2
    record "$($PY -c "import json,sys; d=json.loads(sys.argv[1]); d['pass']=d['lo']>$(cfgv "['gates']['promote_wilson_lo']"); print(json.dumps(d))" "$J")"
    ;;
  strength)
    TAG=rl-$RUN-v$V
    bash scripts/strength_check.sh $NAME "build/botsrc/$NAME" $TAG > $OUT.log 2>&1 || exit 2
    # the reference's field score on the same jobs (zdpn2's own strength check, else run it once)
    if [ ! -f runs/sc/$REF-field/summary.json ]; then
      HELD=$(cfgv "['heldout_field']") GMAPS=$(cfgv "['gmaps']")
      split_eval_ref() {
        local out=runs/sc/$REF-field
        box_request sc-$REF-field $out "build/botsrc/$REF" $BOXPY -u scripts/eval_student.py --player $REF --opponents $HELD --maps $GMAPS --seeds 11,12 --sides AB --out $out --slice 1:4:4 --workers 10
        $PY -u scripts/eval_student.py --player $REF --opponents $HELD --maps $GMAPS --seeds 11,12 --sides AB --out $out --slice 0:1:4 > $out.vmpart.log 2>&1
        box_wait sc-$REF-field
        $PY -u scripts/eval_student.py --player $REF --opponents $HELD --maps $GMAPS --seeds 11,12 --sides AB --out $out > $out.log 2>&1
      }
      split_eval_ref || exit 2
    fi
    F=$(verdict runs/sc/$TAG-field) || exit 2
    R=$(verdict runs/sc/$REF-field) || exit 2
    Z=$(verdict runs/sc/$TAG-zdb) || exit 2
    record "$($PY -c "
import json, sys
f, r, z = (json.loads(a) for a in sys.argv[1:4])
print(json.dumps(dict(mode='strength', version=f['version'], field=f, ref_field=r, zdb=z,
                      **{'pass': f['score'] >= r['score'] - $(cfgv "['gates']['field_margin']")})))" "$F" "$R" "$Z")"
    ;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
$PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT.json'))['pass'] else 1)"
