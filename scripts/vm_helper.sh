#!/bin/bash
# Driver-side helper, the reverse of scripts/box_helper.sh: the loop runs on the Windows box (sandbox ~/rltest/bc26a0)
# and the GCloud VM plays the box_slice. Polls the box's runs/box/*.req (box_split.sh box_request), pushes jobs.json,
# the collector scripts and every bot package of jobs.json (build/bots/<p> compiled + build/botsrc/<p>) to the VM
# sandbox (~/rlvm/bc26a0), runs the request's command there, copies the finished game dirs of RESULT_DIR back into the
# box (never overwriting a file the box already has) and writes the box's runs/box/<tag>.status. Run it in the
# background on the driver:
#   VM_JAVA_FLAGS='-Xmx1200m -XX:+UseSerialGC' nohup scripts/vm_helper.sh > runs/vm_helper.log 2>&1 &
#   scripts/vm_helper.sh setup     (one-off, idempotent: sandbox skeleton, engine jar, maps, build/java, build/student
#                                   copied from the box; JDK and .venv symlinked to the VM's own; every request runs it)
# Env: VM_JAVA_FLAGS overrides jobs.json java_flags (the config's java_flags) for the VM games only (-XX:+UseSerialGC
# cut VM game time ~20% with identical traj_hash); VM_GAME_TIMEOUT (default 3600) raises the VM's game_timeout_seconds;
# BOX_R / VM_R: the sandboxes (relative to the homes). The VM sandbox keeps only the current request's packages and
# run dir. Config on the box: box_slice = the VM's share, workers_box = the VM's workers, workers_vm = the box's own.
source ~/projects/vibe/2026b/tools/vm.sh >/dev/null 2>&1
source "$(dirname "$0")/helper_lib.sh"
set -o pipefail   # a failed tar on either end fails the copy
BOX=(ssh -i ~/.ssh/wsl_box -o ConnectTimeout=20 -o ServerAliveInterval=30 -o BatchMode=yes terry@100.93.26.126)
BR=${BOX_R:-rltest/bc26a0}
VR=${VM_R:-rlvm/bc26a0}
COLLECT="scripts/rl_collect.py scripts/eval_student.py scripts/student_java.sh"
vm() { ensure_vm >/dev/null 2>&1; timeout "${T:-120}" ssh "${SSHO[@]}" "$USER_NAME@$IP" "$@"; }
box() { timeout "${T:-120}" "${BOX[@]}" "$@"; }
now() { TZ=PDT+7 date +%H:%M; }
status() { local m=${3//[\'\"\$\`]/}; box "mkdir -p $BR/runs/box && echo '$2 $(now) $m' > $BR/runs/box/$1.status"; echo "$(now) $1 $2 $m"; }

setup() {
  vm "mkdir -p $VR/vendor/engine $VR/build/bots $VR/build/botsrc $VR/scripts $VR/runs/box && cd $VR &&
      ln -sfn \$(readlink -f ~/projects/vibe/bc26a0/vendor/jdk) vendor/jdk && ln -sfn ~/projects/vibe/bc26a0/.venv .venv" || return 1
  [ "$1" = force ] || ! vm "cd $VR && test -f vendor/engine/battlecode26-java-1.2.6.jar -a -d maps -a -d build/java -a -d build/student" || return 0
  T=900 box "cd $BR && tar czhf - vendor/engine/battlecode26-java-1.2.6.jar maps build/java build/student $COLLECT" |
    T=900 vm "cd $VR && rm -rf maps build/java build/student && tar xzf -"
}
[ "$1" = setup ] && { setup force && echo "setup ok: $VR"; exit; }

while true; do
  for req in $(box "cd $BR/runs/box 2>/dev/null && ls *.req 2>/dev/null"); do
    tag=${req%.req}
    box "cd $BR/runs/box && mv $req $tag.taken" || continue
    mapfile -t L < <(box "cat $BR/runs/box/$tag.taken")
    res=${L[0]} push=${L[1]} cmd=${L[2]}
    status "$tag" running push
    jobs=$(echo "$push" | tr ' ' '\n' | grep 'jobs[.]json$' | head -1)
    pk=$(box "cd $BR && python3 -c \"import json; print(' '.join(sorted({p for j in json.load(open('$jobs'))['jobs'] for p in (j['a'], j['b'])})))\"")
    paths=$(box "cd $BR && for p in $pk; do for d in build/bots/\$p build/botsrc/\$p; do [ -e \$d ] && echo \$d; done; done" | tr '\n' ' ')
    if [ -z "$pk" ] || ! setup || ! T=900 box "cd $BR && tar czhf - $COLLECT $push $paths" |
         T=900 vm "cd $VR && keep='$pk' && for d in build/bots build/botsrc; do for p in \$(ls \$d); do case \" \$keep \" in *\" \$p \"*) ;; *) rm -rf \$d/\$p;; esac; done; done &&
                   for p in $pk; do rm -rf build/bots/\$p build/botsrc/\$p; done &&
                   find runs -mindepth 3 -maxdepth 3 -name 'iter-*' ! -path '$res' -exec rm -rf {} + ; tar xzf -"; then
      status "$tag" failed push; continue
    fi
    # the VM's copy of jobs.json: VM_JAVA_FLAGS, and a game timeout of at least VM_GAME_TIMEOUT (games run ~4x slower)
    vm "cd $VR && python3 -c \"import json; p='$jobs'; d=json.load(open(p)); f='${VM_JAVA_FLAGS:-}'; d['java_flags']=f or d.get('java_flags', '-Xmx2g')
d['game_timeout_seconds']=max(d.get('game_timeout_seconds', 1200), ${VM_GAME_TIMEOUT:-3600}); json.dump(d, open(p, 'w'))\""
    vm "cd $VR && rm -f runs/box/$tag.rc && (setsid bash -c '$cmd > runs/box/$tag.log 2>&1; echo \$? > runs/box/$tag.rc' > /dev/null 2>&1 < /dev/null &)"
    misses=0 state=failed
    while true; do
      sleep 60
      if out=$(T=30 vm "$(job_state $VR $tag)"); then
        misses=0
        case $out in
          RUNNING*) status "$tag" running "${out#RUNNING }";;
          rc=0) state=done; break;;
          *) state=failed; break;;
        esac
      else
        misses=$((misses + 1)); [ $misses -ge 10 ] && { out=unreachable; break; }
      fi
    done
    # finished games only; files the box already has are kept (a late copy never clobbers the box's own fill-in)
    if T=1800 vm "cd $VR && find $res/games -name result.json -printf '%h\n' 2>/dev/null | tar czf - -T -" |
       T=1800 box "cd $BR && tar xzf - --skip-old-files"; then
      n=$(box "cd $BR && ls $res/games/*/result.json 2>/dev/null | wc -l")
      out="$out, box has $n games"
    else
      state=failed out="$out, copy-back failed"
    fi
    vm "cat $VR/runs/box/$tag.log" | box "cat > $BR/runs/box/$tag.vm.log"
    status "$tag" $state "$out"
  done
  sleep 30
done
