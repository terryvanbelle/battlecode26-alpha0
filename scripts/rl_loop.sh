#!/bin/bash
# RL self-play driver (docs/rl-plan.md section 4): runs iterations until STOP, the iteration limit or the end time.
#   scripts/rl_loop.sh CONFIG [RUN] [MAX_ITER] [END_UNIX]          (on the VM; collection split with the Windows box)
# Run state: runs/rl/RUN/ (RUN defaults to the config name): nets/<v>/{net.json,norm.npz}, pool.json, state.json
# (lr, best, field_fails), beta.json (rl_train's per-point beta multipliers), iter-NNNN/{jobs.json, games/, data/,
# stages.tsv, .<stage> markers}, gates/, gates.jsonl, progress.jsonl, loop.log. Resumable: each stage of an iteration
# is skipped when its marker exists. Stop: touch runs/rl/STOP (all runs) or runs/rl/RUN/STOP; checked between stages.
# Per iteration k (learner = net version k, package zdpr<k>):
#   1 disk guard (abort under disk_guard_gb free)   2 build zdpr<k>   3 schedule (rl_schedule.py)
#   4 collect: box_request the box slice, the VM slice, box_wait, VM fill-in; no job starts after T+collect_deadline_min
#     unless fewer than min_games are done/running (rl_collect.py --deadline/--min-done)
#   5 rl_data (parity failure stops the loop)   6 rl_train -> nets/<k+1> (KL refusal: roll back to the best net)
#   7 build zdpr<k+1>   8 gates on schedule (anchor every anchor_every, promotion every promote_every or when the
#     anchor reaches promote_if_anchor; a promoted net gets strength_check; field rule: fail -> roll back to best, lr/2,
#     field_fail_stop fails in a row -> STOP)   9 prune   10 a row in progress.jsonl
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
RD=runs/rl/$RUN
mkdir -p $RD/nets
LOG=$RD/loop.log
PREFIX=$(cfgv "['package_prefix']")
FLOOR=$(cfgv "['floor']") TEMP=$(cfgv "['temperature']")
POINTS=$(cfgv "['trainable_points']")
log() { echo "$(date +%m-%d\ %H:%M:%S) $*" | tee -a $LOG; }
st() {   # st get KEY | st set KEY VALUE (json value)
  $PY - "$RD/state.json" "$@" <<'EOF'
import json, os, sys
p, op, key = sys.argv[1:4]
s = json.load(open(p)) if os.path.exists(p) else {}
if op == 'get':
    print(s.get(key, ''))
else:
    s[key] = json.loads(sys.argv[4]); json.dump(s, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p)
EOF
}
stopped() { [ -f runs/rl/STOP ] || [ -f $RD/STOP ]; }
over() { stopped && { log "STOP file"; return 0; }; [ "$END" -gt 0 ] && [ "$(date +%s)" -ge "$END" ] && { log "end time reached"; return 0; }; return 1; }
stage() {   # stage DIR NAME CMD...: run unless DIR/.NAME exists; time it into DIR/stages.tsv
  local d=$1 n=$2; shift 2
  [ -f $d/.$n ] && return 0
  local t0=$(date +%s)
  "$@" || return $?
  echo -e "$n\t$t0\t$(date +%s)" >> $d/stages.tsv
  touch $d/.$n
}
build_pkg() {   # version: zdpr<v> from nets/<v>, RL build, compiled; marker nets/<v>/.built
  local v=$1
  [ -f $RD/nets/$v/.built ] && [ -d build/bots/$PREFIX$v ] && return 0
  $PY scripts/build_dpnet.py --net $RD/nets/$v/net.json --name $PREFIX$v --temperature $TEMP --floor $FLOOR --rl --version $v --run $RUN >> $LOG 2>&1 || return 1
  $PY -c "import sys; sys.path.insert(0, 'scripts'); from eval_student import compile_local; compile_local('$PREFIX$v')" >> $LOG 2>&1 || return 1
  touch $RD/nets/$v/.built
}
rollback() {   # rollback V REASON: nets/V := nets/best (net.json, norm.npz), rebuild zdpr<V>
  local v=$1 best=$(st get best)
  best=${best:-0}
  log "ROLLBACK v$v -> v$best ($2)"
  mkdir -p $RD/nets/$v
  cp $RD/nets/$best/net.json $RD/nets/$v/net.json
  [ -f $RD/nets/$best/norm.npz ] && cp $RD/nets/$best/norm.npz $RD/nets/$v/norm.npz
  echo "{\"rolled_back_to\": $best, \"reason\": \"$2\"}" > $RD/nets/$v/ROLLBACK.json
  rm -f $RD/nets/$v/.built
  build_pkg $v
}

collect() {   # k
  local k=$1 I=$RD/iter-$(printf %04d $1)
  local D=$(( $(date +%s) + $(cfgv "['collect_deadline_min']") * 60 )) MIN=$(cfgv "['min_games']") tag=rl-$RUN-$1 n
  local WV=$(cfgv "['workers_vm']") v
  for v in $($PY -c "import json; print(' '.join(sorted({p[len('$PREFIX'):] for j in json.load(open('$I/jobs.json'))['jobs'] for p in (j['a'], j['b']) if p.startswith('$PREFIX')})))"); do
    build_pkg $v || { log "build v$v (pool member) failed"; return 1; }   # a pruned or rolled-back member
  done
  box_request $tag $I "$I/jobs.json build/botsrc/$PREFIX$k" $BOXPY -u scripts/rl_collect.py $I/jobs.json --slice $(cfgv "['box_slice']") --workers $(cfgv "['workers_box']") --deadline $D
  $PY -u scripts/rl_collect.py $I/jobs.json --slice $(cfgv "['vm_slice']") --workers $WV --deadline $D >> $I/collect.log 2>&1
  box_wait $tag | tee -a $LOG
  for attempt in 1 2 3; do
    $PY -u scripts/rl_collect.py $I/jobs.json --workers $WV --deadline $D --min-done $MIN >> $I/collect.log 2>&1
    n=$(ls $I/games/*/result.json 2>/dev/null | wc -l)
    [ $n -ge $MIN ] && break
  done
  log "collected iter $k: $n games"
  [ $n -ge $MIN ]
}

train() {   # k
  local k=$1 I=$RD/iter-$(printf %04d $1) rc
  local W=$(cfgv "['train']['value_window']") vd="" j
  for ((j = k - W + 1; j <= k; j++)); do
    [ $j -ge 0 ] && [ -f $RD/iter-$(printf %04d $j)/data/trace.npz ] && vd="$vd $RD/iter-$(printf %04d $j)/data"
  done
  local beta=$($PY -c "
import json; t = json.load(open('$CFG'))['train']
print(t['beta_end'] + (t['beta_start'] - t['beta_end']) * max(0.0, 1 - $k / t['beta_anneal_iters']))")
  local lr=$(st get lr)
  [ -n "$lr" ] || { lr=$(cfgv "['train']['lr']"); st set lr $lr; }
  local T="['train']"
  $PY -u scripts/rl_train.py --net $RD/nets/$k/net.json --anchor $(cfgv "['anchor_net']") --data $I/data --value-data $vd \
    --out $RD/nets/$((k + 1)) --points $POINTS --floor $FLOOR --temperature $TEMP --version $k \
    --beta $beta --beta-state $RD/beta.json --lr $lr --epochs $(cfgv "$T['epochs']") --batch $(cfgv "$T['batch']") \
    --clip $(cfgv "$T['ppo_clip']") --target-kl $(cfgv "$T['target_kl']") --grad-clip $(cfgv "$T['grad_clip']") \
    --c-ent $(cfgv "$T['c_ent']") --collapse-kl $(cfgv "$T['collapse_kl']") --max-anchor-kl $(cfgv "$T['max_anchor_kl']") \
    --adv-clip $(cfgv "$T['adv_clip']") --val-frac $(cfgv "$T['value_val_frac']") --threads $(cfgv "$T['threads']") \
    --seed $k > $I/train.log 2>&1
  rc=$?
  if [ $rc -eq 3 ]; then rollback $((k + 1)) "KL collapse refusal at iter $k"; return $?; fi
  return $rc
}

gates() {   # v (the new version)
  local v=$1 E=$(cfgv "['gates']['anchor_every']") P=$(cfgv "['gates']['promote_every']") prom=0 rc best
  if [ $((v % E)) -eq 0 ]; then
    bash scripts/rl_gate.sh $CFG $RUN anchor $v >> $LOG 2>&1; rc=$?
    [ $rc -eq 2 ] && log "anchor gate error v$v"
    [ $rc -eq 0 ] && prom=1
  fi
  [ $((v % P)) -eq 0 ] && prom=1
  [ $prom -eq 1 ] || return 0
  bash scripts/rl_gate.sh $CFG $RUN promote $v >> $LOG 2>&1; rc=$?
  [ $rc -eq 0 ] || { log "promotion gate v$v: not promoted (rc $rc)"; return 0; }
  bash scripts/rl_gate.sh $CFG $RUN strength $v >> $LOG 2>&1; rc=$?
  if [ $rc -eq 0 ]; then
    $PY scripts/rl_schedule.py --config $CFG --run $RUN --promote $v >> $LOG 2>&1
    st set best $v; st set field_fails 0
    local lr2=$(cfgv "['train']['lr_after_promotion']")
    $PY -c "import sys; sys.exit(0 if float('$(st get lr)') > $lr2 else 1)" && st set lr $lr2
    log "PROMOTED v$v"
  elif [ $rc -eq 1 ]; then
    local ff=$(st get field_fails); ff=$(( ${ff:-0} + 1 ))
    st set field_fails $ff
    st set lr $($PY -c "print(float('$(st get lr)') / 2)")
    rollback $v "field rule at promotion (fail $ff)"
    if [ $ff -ge $(cfgv "['gates']['field_fail_stop']") ]; then log "field rule failed $ff times in a row: STOP"; touch $RD/STOP; fi
  else
    log "strength gate error v$v"
  fi
  return 0
}

prune() {   # k: packages for pool members, the learner and the best only; npz for the last keep_npz_iters iterations
  local k=$1 keep
  keep=$($PY -c "
import json; p = json.load(open('$RD/pool.json')); s = json.load(open('$RD/state.json')) if __import__('os').path.exists('$RD/state.json') else {}
print(' '.join(sorted({'$PREFIX%d' % m['version'] for m in p['members']} | {'$PREFIX%d' % v for v in ($k, $k + 1)} | {'$PREFIX%s' % s.get('best', 0)})))")
  for d in build/botsrc/$PREFIX[0-9]* build/bots/$PREFIX[0-9]*; do
    [ -d "$d" ] || continue
    local b=$(basename $d)
    [[ $b =~ ^$PREFIX[0-9]+$ ]] || continue
    [[ " $keep " == *" $b "* ]] || { rm -rf "$d"; rm -f $RD/nets/${b#$PREFIX}/.built; }
  done
  local N=$(cfgv "['keep_npz_iters']") j
  for ((j = 0; j <= k - N; j++)); do
    rm -f $RD/iter-$(printf %04d $j)/data/p*.npz $RD/iter-$(printf %04d $j)/games/*/decisions-*.bin.gz
  done
}

progress() {   # k
  $PY - "$RD" "$1" <<'EOF' >> $RD/progress.jsonl
import json, os, sys, time
rd, k = sys.argv[1], int(sys.argv[2])
I = f'{rd}/iter-{k:04d}'
row = dict(iter=k, time=time.time())
def js(p):
    return json.load(open(p)) if os.path.exists(p) else None
s = js(f'{I}/data/summary.json') or {}
row.update(games=s.get('valid'), score=s.get('score'), parity=s.get('parity'),
           points={p: {k2: v.get(k2) for k2 in ('rows', 'picked', 'fallback', 'saturated', 'parity_median')} for p, v in s.get('points', {}).items()})
m = js(f'{rd}/nets/{k + 1}/metrics.json') or {}
row.update(value=m.get('value'), exported=m.get('exported'), refused=m.get('refused'),
           train={p: {k2: v.get(k2) for k2 in ('used', 'kl_anchor_end', 'kl_old_new', 'entropy_end', 'clip_frac', 'early_stop', 'grad_snr_cos', 'alarms')}
                  for p, v in (m.get('points') or {}).items()})
row['rollback'] = js(f'{rd}/nets/{k + 1}/ROLLBACK.json')
row['state'] = js(f'{rd}/state.json')
st = {}
if os.path.exists(f'{I}/stages.tsv'):
    for l in open(f'{I}/stages.tsv'):
        n, a, b = l.split()
        st[n] = int(b) - int(a)
row['stage_seconds'] = st
g = os.path.join(rd, 'gates.jsonl')
row['gates'] = [json.loads(l) for l in open(g) if json.loads(l).get('version') == k + 1] if os.path.exists(g) else []
print(json.dumps(row))
EOF
}

# ---- start: nets/0 is the start net
if [ ! -f $RD/nets/0/net.json ]; then
  SN=$(cfgv "['start_net']")
  mkdir -p $RD/nets/0 && cp $SN $RD/nets/0/net.json
  [ -f $(dirname $SN)/norm.npz ] && cp $(dirname $SN)/norm.npz $RD/nets/0/norm.npz
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
  if [ $free -lt $(cfgv "['disk_guard_gb']" | cut -d. -f1) ] || [ $free -lt 2 ]; then log "DISK GUARD: ${free}G free"; exit 1; fi
  build_pkg $k || { log "build v$k failed"; exit 1; }
  stage $I scheduled $PY scripts/rl_schedule.py --config $CFG --run $RUN --iter $k >> $LOG 2>&1 || { log "schedule failed"; exit 1; }
  over && exit 0
  stage $I collected collect $k || { log "collect iter $k: too few games"; exit 1; }
  over && exit 0
  stage $I data $PY -u scripts/rl_data.py --config $CFG --run $RUN --iter $k >> $I/data.log 2>&1
  rc=$?; [ $rc -eq 0 ] || { log "rl_data iter $k failed (rc $rc$([ $rc -eq 2 ] && echo ', PARITY')); see $I/data.log"; exit 1; }
  tail -1 $I/data.log | tee -a $LOG
  over && exit 0
  stage $I trained train $k || { log "train iter $k failed; see $I/train.log"; exit 1; }
  grep '^point\|^value\|^wrote\|^REFUSED' $I/train.log | cut -c1-200 >> $LOG
  build_pkg $((k + 1)) || { log "build v$((k + 1)) failed"; exit 1; }
  stage $I gated gates $((k + 1))
  prune $k
  stage $I logged progress $k
  touch $I/.done
  log "iteration $k done"
  k=$((k + 1))
done
