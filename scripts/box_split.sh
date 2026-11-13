# Split resumable runs between the VM and the Windows box (sourced by pipeline scripts on the VM).
# The VM cannot reach the box; the driver's scripts/box_helper.sh polls runs/box/ for requests, copies the code,
# build and PUSH paths to the box, runs CMD there and copies the finished games of RESULT_DIR back.
# The caller then reruns the whole (resumable) job on the VM, which fills in whatever the box did not finish,
# so a box that is off, busy or unreachable only costs time.
#   box_request TAG RESULT_DIR "PUSH PATHS" CMD...
#   box_wait TAG      returns when the box is done/failed, silent for 15 min, or the request was not taken in 5 min
BOXQ=runs/box
box_request() {
  local tag=$1 res=$2 push=$3; shift 3
  mkdir -p $BOXQ; rm -f $BOXQ/$tag.*
  printf '%s\n%s\n%s\n' "$res" "$push" "$*" > $BOXQ/$tag.tmp && mv $BOXQ/$tag.tmp $BOXQ/$tag.req
}
box_wait() {
  local tag=$1 t0=$(date +%s) st age
  while true; do
    if [ -f $BOXQ/$tag.status ]; then
      st=$(cut -d' ' -f1 $BOXQ/$tag.status); age=$(( $(date +%s) - $(stat -c %Y $BOXQ/$tag.status) ))
      case $st in done|failed) echo "BOX $tag $(cat $BOXQ/$tag.status) $(date +%H:%M)"; return;; esac
      [ $age -gt 900 ] && { echo "BOX $tag silent $(date +%H:%M)"; return; }
    elif [ $(( $(date +%s) - t0 )) -gt 300 ] && [ -f $BOXQ/$tag.req ]; then
      rm -f $BOXQ/$tag.req; echo "BOX $tag not picked up $(date +%H:%M)"; return
    fi
    sleep 30
  done
}
