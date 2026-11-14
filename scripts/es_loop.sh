#!/bin/bash
# ES driver (docs/rl-es-plan.md): head-to-head antithetic ES over per-point output-bias offsets + log-temperature.
#   scripts/es_loop.sh CONFIG [RUN] [MAX_ITER] [END_UNIX]
# Run state: runs/es/RUN/ (RUN defaults to the config name): base/net.json (the fixed base = start_net), phi/<k>.json,
# centre/<k>/net.json, iter-NNNN/{dirs.json, jobs.json, games/, update.json, phi.json, stages.tsv, .<stage>},
# metrics.jsonl, gates.jsonl, gates/, loop.log. Resumable per stage. Stop: touch runs/es/STOP or runs/es/RUN/STOP.
# Per iteration k:
#   1 disk guard   2 schedule: draw directions, build + compile the 2 n_dir candidate packages (es_schedule.py)
#   3 collect: rl_collect.py over jobs.json; with a non-empty box_slice (run on the VM) the box gets its slice through
#     box_split.sh (box_request pushes every candidate package of jobs.json), the VM plays vm_slice, then fills in;
#     no job starts after T + collect_deadline_min unless fewer than min_games are done or running
#   4 update (es_update.py): phi/<k+1>.json, centre/<k+1>/net.json, a metrics.jsonl row
#   5 anchor after the iterations in gates.anchor_after (or every gates.anchor_every): centre <prefix><k+1>c vs
#     anchor_package on zmaps x anchor_seeds x AB (es_eval.sh); gates.anchor_background runs it detached
#   6 prune: the candidate packages of iteration k
export TZ='PDT+7'
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-.venv/bin/python}
BOXPY=.venv/bin/python
source scripts/box_split.sh
CFG=$1
[ -f "$CFG" ] || { sed -n 2,16p "$0"; exit 1; }
cfgv() { $PY -c "import json,sys; c=json.load(open('$CFG')); v=eval('c'+sys.argv[1]); print(','.join(map(str,v)) if isinstance(v,list) else v)" "$1"; }
RUN=${2:-$(cfgv "['name']")}
MAX_ITER=${3:-1000000}
END=${4:-0}
RD=runs/es/$RUN
mkdir -p $RD
LOG=$RD/loop.log
PREFIX=$(cfgv "['package_prefix']")
log() { echo "$(date +%m-%d\ %H:%M:%S) $*" | tee -a $LOG; }
get() { $PY -c "import json,sys; c=json.load(open('$CFG'));
v=c
for k in sys.argv[1].split('.'):
    v=v.get(k) if isinstance(v,dict) else None
print(sys.argv[2] if v is None else (','.join(map(str,v)) if isinstance(v,list) else v))" "$1" "$2"; }
stopped() { [ -f runs/es/STOP ] || [ -f $RD/STOP ]; }
over() { stopped && { log "STOP file"; return 0; }; [ "$END" -gt 0 ] && [ "$(date +%s)" -ge "$END" ] && { log "end time reached"; return 0; }; return 1; }
stage() {   # stage DIR NAME CMD...: run unless DIR/.NAME exists; time it into DIR/stages.tsv
  local d=$1 n=$2; shift 2
  [ -f $d/.$n ] && return 0
  local t0=$(date +%s)
  "$@" || return $?
  echo -e "$n\t$t0\t$(date +%s)" >> $d/stages.tsv
  touch $d/.$n
}

collect() {   # k
  local k=$1 I=$RD/iter-$(printf %04d $1)
  local D=$(( $(date +%s) + $(get collect_deadline_min 60) * 60 )) MIN=$(get min_games 0) tag=es-$RUN-$1 n
  local WV=$(get workers_vm 7) BS=$(get box_slice 0:0:4)
  local box=$(echo $BS | awk -F: '{print ($1 < $2) ? 1 : 0}')   # empty box slice (LO == HI): single host
  if [ $box -eq 1 ]; then
    local push=$($PY -c "import json; print(' '.join(sorted({'build/botsrc/' + p for j in json.load(open('$I/jobs.json'))['jobs'] for p in (j['a'], j['b']) if p.startswith('$PREFIX')})))")
    box_request $tag $I "$I/jobs.json $push" $BOXPY -u scripts/rl_collect.py $I/jobs.json --slice $BS --workers $(get workers_box 10) --deadline $D
    $PY -u scripts/rl_collect.py $I/jobs.json --slice $(get vm_slice 0:1:4) --workers $WV --deadline $D >> $I/collect.log 2>&1
    box_wait $tag | tee -a $LOG
  fi
  for attempt in 1 2 3; do
    $PY -u scripts/rl_collect.py $I/jobs.json --workers $WV --deadline $D --min-done $MIN >> $I/collect.log 2>&1
    n=$(ls $I/games/*/result.json 2>/dev/null | wc -l)
    [ $n -ge $MIN ] && break
  done
  log "collected iter $k: $n games"
  [ $n -ge $MIN ]
}

anchor() {   # v: centre version (after v updates)
  local v=$1 E=$(get gates.anchor_every 0) AA=",$(get gates.anchor_after ''),"
  local due=0
  [ "$E" -gt 0 ] && [ $((v % E)) -eq 0 ] && due=1
  [[ $AA == *",$((v - 1)),"* ]] && due=1
  [ $due -eq 1 ] || return 0
  local REF=$(get anchor_package '') name=${PREFIX}${v}c
  [ -n "$REF" ] || return 0
  $PY scripts/es_schedule.py --config $CFG --run $RUN --centre $v >> $LOG 2>&1 || { log "centre build v$v failed"; return 0; }
  mkdir -p $RD/gates
  local cmd="bash scripts/es_eval.sh $name $REF $(get zmaps '') $(get gates.anchor_seeds 41,42) $RD/gates/anchor-v$v $(get gates.anchor_workers 4) $RD/gates.jsonl"
  if [ "$(get gates.anchor_background False)" = "True" ]; then
    (setsid bash -c "$cmd >> $LOG 2>&1" > /dev/null 2>&1 < /dev/null &)
    log "anchor v$v started in the background ($name vs $REF)"
  else
    log "anchor v$v: $($cmd 2>&1 | tail -1)"
  fi
  return 0
}

# ---- start: base/net.json is the start net (fixed for the run)
if [ ! -f $RD/base/net.json ]; then
  SN=$(cfgv "['start_net']")
  mkdir -p $RD/base && cp $SN $RD/base/net.json
  [ -f $(dirname $SN)/norm.npz ] && cp $(dirname $SN)/norm.npz $RD/base/norm.npz
  cp $CFG $RD/config.json
  log "START run $RUN from $SN"
fi
k=0
while [ -f $RD/iter-$(printf %04d $k)/.done ]; do k=$((k + 1)); done
log "loop: run $RUN at iteration $k (max $MAX_ITER, end ${END})"
while [ $k -lt $MAX_ITER ]; do
  over && exit 0
  I=$RD/iter-$(printf %04d $k)
  mkdir -p $I
  free=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
  if [ $free -lt $(get disk_guard_gb 2 | cut -d. -f1) ] || [ $free -lt 1 ]; then log "DISK GUARD: ${free}G free"; exit 1; fi
  stage $I scheduled $PY scripts/es_schedule.py --config $CFG --run $RUN --iter $k >> $LOG 2>&1 || { log "schedule failed"; exit 1; }
  over && exit 0
  stage $I collected collect $k || { log "collect iter $k: too few games"; exit 1; }
  over && exit 0
  stage $I updated $PY -u scripts/es_update.py --config $CFG --run $RUN --iter $k >> $LOG 2>&1 || { log "update iter $k failed"; exit 1; }
  stage $I gated anchor $((k + 1))
  $PY scripts/es_schedule.py --config $CFG --run $RUN --prune $k >> $LOG 2>&1
  touch $I/.done
  log "iteration $k done"
  k=$((k + 1))
done
