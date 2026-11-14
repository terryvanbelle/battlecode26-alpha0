#!/bin/bash
# Decision-point imitation, round 2 (docs/decision-points.md): zdp after pass 2 -> teacher-dp2 (1000 games, box 3/4)
# -> train_dp -> zdpn2 -> strength_check (64 vs zdb, 192 vs the held-out field). No bytecode limits.
export TZ='PDT+7'
cd ~/projects/vibe/bc26a0
PY=.venv/bin/python
source scripts/box_split.sh
BOXPY=.venv/bin/python
KEEP="--keep 1:0.25,4:0.4,5:1 --keep0 2:0.03,3:0.03,6:0.03,12:0.2"
collect() { mkdir -p runs/$1; for i in 1 2; do $PY -u scripts/collect_teacher.py configs/$1.json >> runs/$1/collect.log 2>&1; done; echo COLLECTED $1 $(date +%H:%M); }
echo START $(date +%H:%M)
mkdir -p runs/teacher-dp2
box_request teacher-dp2 runs/teacher-dp2 "java" $BOXPY -u scripts/collect_teacher.py configs/teacher-dp2.json --slice 1:4:4 --workers 10
$PY -u scripts/collect_teacher.py configs/teacher-dp2.json --slice 0:1:4 >> runs/teacher-dp2/collect.log 2>&1
box_wait teacher-dp2
collect teacher-dp2
mkdir -p runs/dp/net-v2
$PY -u scripts/train_dp.py --games 'runs/teacher-dp2/games/*' --out runs/dp/net-v2 --epochs 15 $KEEP > runs/dp/net-v2/train.log 2>&1 || { echo TRAINFAILED; exit 1; }
echo TRAINED $(date +%H:%M); grep "^point" runs/dp/net-v2/train.log | cut -c1-150
$PY scripts/build_dpnet.py --net runs/dp/net-v2/net.json --name zdpn2 --temperature 1.0 --floor 0.001
bash scripts/strength_check.sh zdpn2 "java build/botsrc/zdpn2"
echo ALLDONE $(date +%H:%M)
