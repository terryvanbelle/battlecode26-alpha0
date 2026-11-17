#!/bin/bash
# zdn imitation, round 1 (docs/zdn.md): build the teacher (zdb + zdn recorder) -> collect teacher-zdn1 on this host.
# Training runs on the box GPU (zdn_train.py), after the games are copied there.
export TZ='PDT+7'
cd ~/projects/vibe/bc26a0
PY=.venv/bin/python
echo START $(date +%H:%M)
$PY scripts/build_zdn.py --teacher zdnt --baby-pm 40 --king-pm 150 || { echo BUILDFAILED; exit 1; }
mkdir -p runs/teacher-zdn1
for i in 1 2; do $PY -u scripts/collect_teacher.py configs/teacher-zdn1.json "$@" >> runs/teacher-zdn1/collect.log 2>&1; done
echo COLLECTED $(date +%H:%M) $(ls runs/teacher-zdn1/games | wc -l) games
