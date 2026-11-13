#!/bin/bash
# Capability-complete restart (student-actions-v3 / student-obs-v4, docs/capability-principle.md).
# Takes over from pipeline_obs_v4b.sh (same steps) to split game playing with the Windows box (scripts/box_split.sh;
# the driver runs scripts/box_helper.sh). The box takes 3/4 of each run's games (it plays ~4x as fast as the VM),
# the VM 1/4, then the VM reruns the full (resumable) job to fill gaps.
# teacher-v9 (pure zdb) -> student-v9 -> vs zdb -> teacher-v10-dagger -> student-v10 -> vs zdb + frozen gate
export TZ='PDT+7'
cd ~/projects/vibe/bc26a0
PY=.venv/bin/python
source scripts/box_split.sh
BOXPY=.venv/bin/python
ZMAPS=mincemeat,jail,rift,pipes,gridlock,thunderdome,sunandmoon,ZeroDay
GMAPS=tiny,DefaultSmall,DefaultMedium,DefaultLarge,closeup,knifefight,cheesefarm,wallsofparadis
GOPP=@untrained,examplefuncsplayer,lectureplayer,VarunVejalla_final4_run_away10,GuzziFlipFlops_Hermesv8,quesswho_quals,daannte_v47_Skadoosh,AlexT101_usqualsbot
collect() { mkdir -p runs/$1; for i in 1 2; do $PY -u scripts/collect_teacher.py configs/$1.json >> runs/$1/collect.log 2>&1; done; echo COLLECTED $1 $(date +%H:%M); }
train() { mkdir -p runs/$1; $PY -u -m alpha0.student.train --run $2 --out runs/$1 --epochs 6 --threads 8 --king-weight 2 $3 > runs/$1/train.log 2>&1 || { echo TRAINFAILED $1; exit 1; }; echo TRAINED $1 $(date +%H:%M); }
split_eval() {   # tag push out args...
  local tag=$1 push=$2 out=$3; shift 3
  box_request $tag $out "$push" $BOXPY -u scripts/eval_student.py "$@" --out $out --slice 1:4:4 --workers 10
  $PY -u scripts/eval_student.py "$@" --out $out --slice 0:1:4 > $out.vmpart.log 2>&1
  box_wait $tag
  $PY -u scripts/eval_student.py "$@" --out $out > $out.log 2>&1
}
vszdb() { split_eval vszdb-$1 "runs/$1/best.bin" runs/eval/vszdb-$1-T1.0 --player net:runs/$1/best.bin:1.0 --opponents zdb --maps $ZMAPS --seeds 21,22 --sides AB --limits; echo VSZDB $1 $(date +%H:%M); }
gate() { split_eval gate-$1 "runs/$1/best.bin runs/$1/untrained.bin" runs/eval/gate-$1-T1.0 --player net:runs/$1/best.bin:1.0 --opponents $GOPP --maps $GMAPS --seeds 11,12 --sides AB --limits \
  --untrained runs/$1/untrained.bin; echo GATE $1 $(date +%H:%M); }
collect_split() {   # name, extra push paths (e.g. the DAgger mix net)
  mkdir -p runs/$1
  box_request $1 runs/$1 "$2" $BOXPY -u scripts/collect_teacher.py configs/$1.json --slice 1:4:4 --workers 10
  $PY -u scripts/collect_teacher.py configs/$1.json --slice 0:1:4 >> runs/$1/collect.log 2>&1
  box_wait $1
  collect $1
}
echo START $(date +%H:%M)
while pgrep -f "[c]ollect_teacher.py configs/teacher-v9" >/dev/null; do sleep 60; done
echo OLDCOLLECTDONE $(date +%H:%M)
$PY scripts/build_student.py || { echo BUILDFAILED; exit 1; }
collect teacher-v9
train student-v9 runs/teacher-v9 ""
vszdb student-v9
collect_split teacher-v10-dagger runs/student-v9/best.bin
train student-v10 runs/teacher-v9,runs/teacher-v10-dagger ""
vszdb student-v10
gate student-v10
echo ALLDONE $(date +%H:%M)
