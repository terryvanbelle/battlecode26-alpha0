#!/bin/bash
# Driver-side helper for scripts/box_split.sh: runs requests the VM pipeline posts in runs/box/ on the Windows box
# (WSL2, reached over Tailscale) and copies finished games back to the VM. Run it in the background on the driver.
source ~/projects/vibe/2026b/tools/vm.sh >/dev/null 2>&1
BOX=(ssh -i ~/.ssh/wsl_box -o ConnectTimeout=20 -o ServerAliveInterval=30 -o BatchMode=yes terry@100.93.26.126)
R=projects/vibe/bc26a0
vm() { ensure_vm >/dev/null 2>&1; timeout "${T:-120}" ssh "${SSHO[@]}" "$USER_NAME@$IP" "$@"; }
box() { timeout "${T:-120}" "${BOX[@]}" "$@"; }
status() { local m=${3//[\'\"\$\`]/}; vm "echo '$2 $(TZ=PDT+7 date +%H:%M) $m' > $R/runs/box/$1.status"; echo "$(TZ=PDT+7 date +%H:%M) $1 $2 $m"; }

while true; do
  for req in $(vm "cd $R/runs/box 2>/dev/null && ls *.req 2>/dev/null"); do
    tag=${req%.req}
    vm "cd $R/runs/box && mv $req $tag.taken" || continue
    mapfile -t L < <(vm "cat $R/runs/box/$tag.taken")
    res=${L[0]} push=${L[1]} cmd=${L[2]}
    status "$tag" running push
    if ! T=900 vm "cd $R && tar czf - build scripts configs alpha0 $push" | T=900 box "cd $R && tar xzf -"; then
      status "$tag" failed push; continue
    fi
    box "cd $R && mkdir -p runs/box && rm -f runs/box/$tag.rc && (setsid bash -c '$cmd > runs/box/$tag.log 2>&1; echo \$? > runs/box/$tag.rc' > /dev/null 2>&1 < /dev/null &)"
    misses=0 state=failed
    while true; do
      sleep 60
      if out=$(T=30 box "cd $R; if [ -f runs/box/$tag.rc ]; then echo rc=\$(cat runs/box/$tag.rc); elif pgrep -f 'box/$tag[.]log' >/dev/null; then echo RUNNING \$(tail -1 runs/box/$tag.log | cut -c1-80); else echo DIED; fi"); then
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
    # game runs: only finished games; anything else (e.g. a training run): the whole directory
    T=1800 box "cd $R && if [ -d $res/games ]; then find $res/games -name result.json -printf '%h\n'; else echo $res; fi | tar czf - -T -" | T=1800 vm "cd $R && tar xzf -"
    status "$tag" $state "$out"
  done
  sleep 30
done
