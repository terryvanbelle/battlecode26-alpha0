#!/bin/bash
# Keep an RL run's PROGRESS.md chart and status line current. After every finished iteration (or new gate) of
# runs/rl/RUN in the box sandbox: its progress.jsonl and gates.jsonl are copied here, scripts/plot_rl_vs.py draws
# docs/img/RUN-score.svg (pool games against v0, plus one gate series), the one-line status between
# <!-- RL_STATUS_START --> and <!-- RL_STATUS_END --> in PROGRESS.md is rewritten, and only those two paths are
# committed and pushed to the current branch. Run on the driver:
#   setsid nohup scripts/rl_chart_sync.sh rl-s1 > runs/rl_chart_sync.log 2>&1 < /dev/null &
# Env: GATE_MODE (the charted gate series, default anchor-zdpg39), GATE_LABEL, ANCHOR_LABEL (the plain anchor gate's
# opponent in the status line, default "the start"), EVERY (poll seconds, default 120). Exits when runs/rl/RUN/STOP
# exists on the box (after a last refresh).
export TZ='PDT+7'
cd "$(dirname "$0")/.." || exit 1
RUN=${1:?usage: rl_chart_sync.sh RUN}
BOX=(ssh -i ~/.ssh/wsl_box -o ConnectTimeout=20 -o BatchMode=yes terry@100.93.26.126)
BR=rltest/bc26a0
GATE_MODE=${GATE_MODE:-anchor-zdpg39} GATE_LABEL=${GATE_LABEL:-vs zdpg39 = rl-m2 v39 (32-game gate)}
ANCHOR_LABEL=${ANCHOR_LABEL:-the start} EVERY=${EVERY:-120}
SVG=docs/img/$RUN-score.svg
TMP=$(mktemp -d)
trap 'rm -rf $TMP' EXIT
log() { echo "$(date +%m-%d\ %H:%M:%S) $*"; }

refresh() {
  timeout 120 "${BOX[@]}" "cd $BR/runs/rl/$RUN && tar czf - progress.jsonl \$(test -f gates.jsonl && echo gates.jsonl)" \
    > $TMP/p.tgz 2> $TMP/err || { log "fetch failed: $(tail -1 $TMP/err)"; return 1; }
  rm -rf $TMP/p && mkdir $TMP/p && tar xzf $TMP/p.tgz -C $TMP/p || return 1
  touch $TMP/p/gates.jsonl
  python3 scripts/plot_rl_vs.py $TMP/p/progress.jsonl --opp v0 --pool-label "vs the start net v0 (pool games)" \
    --gates $TMP/p/gates.jsonl --gate-mode "$GATE_MODE" --gate-label "$GATE_LABEL" --out $SVG \
    --title "$RUN: learner vs its start net and vs rl-m2 v39" > /dev/null || return 1
  python3 - $TMP/p "$GATE_MODE" "$ANCHOR_LABEL" <<'EOF' || return 1
import json, re, sys, time
from pathlib import Path
d, gmode, alabel = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(d / 'progress.jsonl') if l.strip()]
gates = [json.loads(l) for l in open(d / 'gates.jsonl') if l.strip()]
fmt = lambda mode: ', '.join(f"v{g['version']} {g['points']:g}/{g['games']}" for g in sorted(
    (g for g in gates if g['mode'] == mode), key=lambda g: g['version']))
p = sum(r['score'].get('v0', {}).get('points', 0) for r in rows)
n = sum(r['score'].get('v0', {}).get('games', 0) for r in rows)
last = max((r['iter'] for r in rows), default=-1)
opp = gmode.split('-', 1)[1] if '-' in gmode else gmode
text = (f"Updated automatically after each iteration ([rl_chart_sync.sh](scripts/rl_chart_sync.sh)); last "
        f"{time.strftime('%B %-d, %H:%M PDT')}, " + (f"after iteration {last}" if last >= 0 else "no iteration finished yet")
        + (f". Gates against {alabel}: {fmt('anchor')}" if fmt('anchor') else '')
        + (f". Against {opp}: {fmt(gmode)}" if fmt(gmode) else '')
        + (f". Pool games against v0, iterations 0–{last}: {p:g}/{n} = {p / max(n, 1):.2f}" if n else '') + '.')
f = Path('PROGRESS.md'); s = f.read_text()
s2, k = re.subn(r'<!-- RL_STATUS_START -->.*?<!-- RL_STATUS_END -->',
                lambda _: f'<!-- RL_STATUS_START -->{text}<!-- RL_STATUS_END -->', s, flags=re.S)
if k != 1:
    sys.exit('PROGRESS.md: no RL_STATUS block')
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

state() { timeout 60 "${BOX[@]}" "cd $BR/runs/rl/$RUN 2>/dev/null && echo \$(ls -d iter-*/.done 2>/dev/null | wc -l) \$(cat gates.jsonl 2>/dev/null | wc -l) \$(test -f STOP -o -f ../STOP && echo stop)"; }
ls_done() { echo $(( ${cur%% *} - 1 )); }

last=""
while true; do
  cur=$(state) || { log "box unreachable or no run dir yet"; sleep $EVERY; continue; }
  if [ "$cur" != "$last" ] && [ "${cur%% *}" -gt 0 ]; then
    refresh && last=$cur
  fi
  [[ $cur == *stop* ]] && { log "STOP on the box: exiting"; exit 0; }
  sleep $EVERY
done
