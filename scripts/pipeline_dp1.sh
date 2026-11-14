#!/bin/bash
# Decision-point imitation, round 1 (docs/decision-points.md). No bytecode limits (user, 2026-11-13).
# teacher-dp1 (600 zdp games, decision records only; box 3/4, VM 1/4) -> train_dp (one MLP per point) -> zdpn1
# (net-mode zdp) -> 32 games vs zdb (same jobs as the primitive students' zdb check) + held-out field check, with
# zdp (= zdb) as the reference on the field jobs.
export TZ='PDT+7'
cd ~/projects/vibe/bc26a0
PY=.venv/bin/python
source scripts/box_split.sh
BOXPY=.venv/bin/python
ZMAPS=mincemeat,jail,rift,pipes,gridlock,thunderdome,sunandmoon,ZeroDay
GMAPS=tiny,DefaultSmall,DefaultMedium,DefaultLarge,closeup,knifefight,cheesefarm,wallsofparadis
HELD=VarunVejalla_final4_run_away10,GuzziFlipFlops_Hermesv8,quesswho_quals,daannte_v47_Skadoosh,AlexT101_usqualsbot
KEEP="--keep 1:0.25,4:0.4,5:1 --keep0 2:0.03,3:0.03,6:0.03,12:0.2"
collect() { mkdir -p runs/$1; for i in 1 2; do $PY -u scripts/collect_teacher.py configs/$1.json >> runs/$1/collect.log 2>&1; done; echo COLLECTED $1 $(date +%H:%M); }
collect_split() {   # name, extra push paths
  mkdir -p runs/$1
  box_request $1 runs/$1 "$2" $BOXPY -u scripts/collect_teacher.py configs/$1.json --slice 1:4:4 --workers 10
  $PY -u scripts/collect_teacher.py configs/$1.json --slice 0:1:4 >> runs/$1/collect.log 2>&1
  box_wait $1
  collect $1
}
split_eval() {   # tag push out args...
  local tag=$1 push=$2 out=$3; shift 3
  box_request $tag $out "$push" $BOXPY -u scripts/eval_student.py "$@" --out $out --slice 1:4:4 --workers 10
  $PY -u scripts/eval_student.py "$@" --out $out --slice 0:1:4 > $out.vmpart.log 2>&1
  box_wait $tag
  $PY -u scripts/eval_student.py "$@" --out $out > $out.log 2>&1
}
echo START $(date +%H:%M)
collect_split teacher-dp1 "java"
mkdir -p runs/dp/net-v1
$PY -u scripts/train_dp.py --games 'runs/teacher-dp1/games/*' --out runs/dp/net-v1 --epochs 15 $KEEP > runs/dp/net-v1/train.log 2>&1 || { echo TRAINFAILED; exit 1; }
echo TRAINED $(date +%H:%M)
$PY scripts/build_dpnet.py --net runs/dp/net-v1/net.json --name zdpn1 --temperature 1.0 --floor 0.001
split_eval vszdb-zdpn1 "java build/botsrc/zdpn1" runs/eval/vszdb-zdpn1 --player zdpn1 --opponents zdb --maps $ZMAPS --seeds 21,22 --sides AB
echo VSZDB $(date +%H:%M); tail -3 runs/eval/vszdb-zdpn1.log
split_eval field-zdpn1 "java build/botsrc/zdpn1" runs/eval/field-zdpn1 --player zdpn1 --opponents $HELD,examplefuncsplayer --maps $GMAPS --seeds 11 --sides AB
split_eval field-zdp "java" runs/eval/field-zdp --player zdp --opponents $HELD,examplefuncsplayer --maps $GMAPS --seeds 11 --sides AB
echo FIELD $(date +%H:%M); tail -8 runs/eval/field-zdpn1.log; tail -8 runs/eval/field-zdp.log
echo ALLDONE $(date +%H:%M)
