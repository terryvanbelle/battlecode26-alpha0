#!/bin/bash
# Capability-complete restart (student-actions-v2 / student-obs-v4, docs/capability-principle.md):
# teacher-v9 (pure zdb) -> student-v9 -> vs zdb -> teacher-v10-dagger -> student-v10 -> vs zdb + frozen gate
export TZ='PDT+7'
cd ~/projects/vibe/bc26a0
PY=.venv/bin/python
ZMAPS=mincemeat,jail,rift,pipes,gridlock,thunderdome,sunandmoon,ZeroDay
GMAPS=tiny,DefaultSmall,DefaultMedium,DefaultLarge,closeup,knifefight,cheesefarm,wallsofparadis
GOPP=@untrained,examplefuncsplayer,lectureplayer,VarunVejalla_final4_run_away10,GuzziFlipFlops_Hermesv8,quesswho_quals,daannte_v47_Skadoosh,AlexT101_usqualsbot
collect() { mkdir -p runs/$1; for i in 1 2; do $PY -u scripts/collect_teacher.py configs/$1.json >> runs/$1/collect.log 2>&1; done; echo COLLECTED $1 $(date +%H:%M); }
train() { mkdir -p runs/$1; $PY -u -m alpha0.student.train --run $2 --out runs/$1 --epochs 6 --threads 8 --king-weight 2 $3 > runs/$1/train.log 2>&1 || { echo TRAINFAILED $1; exit 1; }; echo TRAINED $1 $(date +%H:%M); }
vszdb() { $PY -u scripts/eval_student.py --player net:runs/$1/best.bin:1.0 --opponents zdb --maps $ZMAPS --seeds 21,22 --sides AB --limits --out runs/eval/vszdb-$1-T1.0 > runs/eval/vszdb-$1-T1.0.log 2>&1; echo VSZDB $1 $(date +%H:%M); }
gate() { $PY -u scripts/eval_student.py --player net:runs/$1/best.bin:1.0 --opponents $GOPP --maps $GMAPS --seeds 11,12 --sides AB --limits \
  --untrained runs/$1/untrained.bin --out runs/eval/gate-$1-T1.0 > runs/eval/gate-$1-T1.0.log 2>&1; echo GATE $1 $(date +%H:%M); }
echo START $(date +%H:%M)
$PY scripts/build_student.py || { echo BUILDFAILED; exit 1; }
collect teacher-v9
train student-v9 runs/teacher-v9 ""
vszdb student-v9
collect teacher-v10-dagger
train student-v10 runs/teacher-v9,runs/teacher-v10-dagger ""
vszdb student-v10
gate student-v10
echo ALLDONE $(date +%H:%M)
