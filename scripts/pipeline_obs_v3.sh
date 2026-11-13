#!/bin/bash
# obs-v3 restart: teacher-v7 (pure zdb) -> student-v7 -> teacher-v8-dagger -> student-v8 -> vs zdb + frozen gate
export TZ='PDT+7'
cd ~/projects/vibe/bc26a0
PY=.venv/bin/python
ZMAPS=mincemeat,jail,rift,pipes,gridlock,thunderdome,sunandmoon,ZeroDay
GMAPS=tiny,DefaultSmall,DefaultMedium,DefaultLarge,closeup,knifefight,cheesefarm,wallsofparadis
GOPP=@untrained,examplefuncsplayer,lectureplayer,VarunVejalla_final4_run_away10,GuzziFlipFlops_Hermesv8,quesswho_quals,daannte_v47_Skadoosh,AlexT101_usqualsbot
collect() { mkdir -p runs/$1; for i in 1 2; do $PY -u scripts/collect_teacher.py configs/$1.json >> runs/$1/collect.log 2>&1; done; echo COLLECTED $1 $(date +%H:%M); }
train() { mkdir -p runs/$1; $PY -u -m alpha0.student.train --run $2 --out runs/$1 --epochs 8 --threads 8 --king-weight 2 $3 > runs/$1/train.log 2>&1 || { echo TRAINFAILED $1; exit 1; }; echo TRAINED $1 $(date +%H:%M); }
vszdb() { $PY -u scripts/eval_student.py --player net:runs/$1/best.bin:1.0 --opponents zdb --maps $ZMAPS --seeds 21,22 --sides AB --limits --out runs/eval/vszdb-$1-T1.0 > runs/eval/vszdb-$1-T1.0.log 2>&1; echo VSZDB $1 $(date +%H:%M); }
echo START $(date +%H:%M)
collect teacher-v7
train student-v7 runs/teacher-v7 ""
vszdb student-v7
collect teacher-v8-dagger
train student-v8 runs/teacher-v7,runs/teacher-v8-dagger "--baby-keep 0.4"
vszdb student-v8
$PY -u scripts/eval_student.py --player net:runs/student-v8/best.bin:1.0 --opponents $GOPP --maps $GMAPS --seeds 11,12 --sides AB --limits \
  --untrained runs/student-v8/untrained.bin --out runs/eval/gate-student-v8-T1.0 > runs/eval/gate-student-v8-T1.0.log 2>&1
echo ALLDONE $(date +%H:%M)
