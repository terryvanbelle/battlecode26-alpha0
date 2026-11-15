#!/bin/bash
# Keep an ES run's PROGRESS.md chart and status line current. After every finished iteration (or new gate) of
# runs/es/RUN in the box sandbox: scripts/es_progress.py there (piped in, nothing deployed), scripts/plot_rl_vs.py
# here -> docs/img/RUN-score.svg, the one-line status between <!-- ES_STATUS_START --> and <!-- ES_STATUS_END --> in
# PROGRESS.md rewritten, and a commit of only those two paths, pushed to the current branch. Run on the driver:
#   setsid nohup scripts/es_chart_sync.sh es-m2 > runs/es_chart_sync.log 2>&1 < /dev/null &
# Env: START_GATES / START_VERSION (the start net's gates as version 0; default rl-m2 v39), START_LABEL, POOL_OPP
# (pool package scored as "the start net", default zdpg39), ANCHOR (gate opponent, default zdpf005), EVERY (poll
# seconds, default 120). Exits when runs/es/RUN/STOP exists on the box (after a last refresh).
export TZ='PDT+7'
cd "$(dirname "$0")/.." || exit 1
RUN=${1:?usage: es_chart_sync.sh RUN}
BOX=(ssh -i ~/.ssh/wsl_box -o ConnectTimeout=20 -o BatchMode=yes terry@100.93.26.126)
BR=rltest/bc26a0
START_GATES=${START_GATES:-runs/rl/rl-m2/gates.jsonl} START_VERSION=${START_VERSION:-39} START_LABEL=${START_LABEL:-rl-m2 v39}
POOL_OPP=${POOL_OPP:-zdpg39} ANCHOR=${ANCHOR:-zdpf005} EVERY=${EVERY:-120}
SVG=docs/img/$RUN-score.svg
TMP=$(mktemp -d)
trap 'rm -rf $TMP' EXIT
log() { echo "$(date +%m-%d\ %H:%M:%S) $*"; }

refresh() {
  timeout 120 "${BOX[@]}" "cd $BR && python3 - runs/es/$RUN /tmp/esprog-$RUN --start-gates $START_GATES --start-version $START_VERSION >&2 && tar czf - -C /tmp/esprog-$RUN . && rm -rf /tmp/esprog-$RUN" \
    < scripts/es_progress.py > $TMP/p.tgz 2> $TMP/err || { log "extract failed: $(tail -1 $TMP/err)"; return 1; }
  rm -rf $TMP/p && mkdir $TMP/p && tar xzf $TMP/p.tgz -C $TMP/p || return 1
  python3 scripts/plot_rl_vs.py $TMP/p/progress.jsonl --opp $POOL_OPP --pool-label "vs the start net (pool games)" \
    --gates $TMP/p/gates.jsonl --gate-mode anchor-$ANCHOR --gate-label "vs $ANCHOR (32-game gate)" --out $SVG \
    --title "$RUN: ES centre vs its start net ($START_LABEL) and vs $ANCHOR" \
    --xlabel "ES version (centre after k updates)" > /dev/null || return 1
  python3 - $TMP/p "$POOL_OPP" "$ANCHOR" "$START_LABEL" <<'EOF' || return 1
import json, re, sys, time
from pathlib import Path
d, opp, anchor, start = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
rows = [json.loads(l) for l in open(d / 'progress.jsonl')]
gates = sorted((g['version'], g['points'], g['games']) for g in map(json.loads, open(d / 'gates.jsonl'))
               if g['mode'] == 'anchor-' + anchor)
p = sum(r['score'].get(opp, {}).get('points', 0) for r in rows)
n = sum(r['score'].get(opp, {}).get('games', 0) for r in rows)
gl = ', '.join(f"v{v}{f' ({start})' if v == 0 else ''} {pt:g}/{g}" for v, pt, g in gates)
last = max((r['iter'] for r in rows), default=-1)
text = (f"Updated automatically after each iteration ([es_chart_sync.sh](scripts/es_chart_sync.sh)); last "
        f"{time.strftime('%B %-d, %H:%M PDT')}, " + (f"after iteration {last}" if last >= 0 else "no iteration finished yet")
        + f". Gates against {anchor}: {gl}." + (f" Pool games against the start net, iterations 0–{last}: pooled "
                                                f"{p:g}/{n} = {p / max(n, 1):.2f}." if n else ""))
f = Path('PROGRESS.md'); s = f.read_text()
s2, k = re.subn(r'<!-- ES_STATUS_START -->.*?<!-- ES_STATUS_END -->',
                lambda _: f'<!-- ES_STATUS_START -->{text}<!-- ES_STATUS_END -->', s, flags=re.S)
if k != 1:
    sys.exit('PROGRESS.md: no ES_STATUS block')
f.write_text(s2)
print(text)
EOF
  git add -- $SVG PROGRESS.md || return 1   # a new run's SVG is untracked until its first commit
  if git diff --cached --quiet -- $SVG PROGRESS.md; then log "no change"; return 0; fi
  git commit -q -m "PROGRESS: $RUN chart and status $(n=$(ls_done); [ $n -ge 0 ] && echo after iteration $n || echo at the start)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SzziPTzb6YbDMqwPrZRdy9" -- $SVG PROGRESS.md || return 1
  git push -q origin HEAD 2>/dev/null || { git pull -q --rebase --autostash && git push -q origin HEAD; } ||
    { log "push failed"; return 1; }
  log "pushed $(git log --oneline -1)"
}

state() { timeout 60 "${BOX[@]}" "cd $BR/runs/es/$RUN && echo \$(ls -d iter-*/.done 2>/dev/null | wc -l) \$(cat gates.jsonl 2>/dev/null | wc -l) \$(test -f STOP && echo stop)"; }
ls_done() { echo $(( ${cur%% *} - 1 )); }

last=""
while true; do
  cur=$(state) || { log "box unreachable"; sleep $EVERY; continue; }
  if [ "$cur" != "$last" ]; then
    refresh && last=$cur
  fi
  [[ $cur == *stop* ]] && { log "STOP on the box: exiting"; exit 0; }
  sleep $EVERY
done
