#!/usr/bin/env python3
"""Plot an rl_loop run's learner score by version as a standalone SVG: against one pool opponent (default v0) from
progress.jsonl, plus optional gate series from gates.jsonl (e.g. the anchor-zdb tracking gate).

  scripts/plot_rl_vs.py runs/rl/RUN/progress.jsonl --out docs/img/RUN.svg [--opp v0]
      [--gates runs/rl/RUN/gates.jsonl --gate-mode anchor-zdb --gate-label "vs zdb (32-game gate)"]
      [--title T] [--mark K:TEXT ...] [--pool-label L] [--xlabel L] [--newest-after-perfect [--newest-label L]]
  scripts/plot_rl_vs.py OUT/progress.jsonl --opp zdpg0,zdpg39 ...   (an ES run, after scripts/es_progress.py RUN OUT)

Pool series: iteration k's pool games, the learner (version k) against --opp (a comma list is pooled). Gate series: one point per gated version.
With --newest-after-perfect (one --opp): once the learner wins every one of its latest pool games against --opp (the
latest iterations back to at least MIN_PERFECT_GAMES games; after PFSP starts v0 gets about 6 games an iteration), the
comparison against --opp has nothing left to show, so from the next iteration on the chart follows the learner
against the newest pool version it played instead (a third series; a mark at the switch).
Score = points / games (win 1, draw 0.5) with a Wilson 95% interval; the dashed line is 0.5 (even). No dependencies
(plain Python, hand-written SVG that follows the viewer's light/dark theme; hover a point for its numbers).
"""
import argparse, json, math, re

# categorical slots 1-3 of the dataviz reference palette (light, dark); each series keeps its slot whether or not the
# others are drawn: the pool series 0, the gate series 1, the newest-version series 2
COLORS = [('#2a78d6', '#3987e5'), ('#eb6834', '#d95926'), ('#1baf7a', '#199e70')]
MIN_PERFECT_GAMES = 8


def wilson(p, n, z=1.96):
    if not n:
        return 0.0, 1.0
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def pool_scores(path):
    """Each iteration's pool score ({opponent: {points, games}}), the last row per iteration, in iteration order."""
    rows = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows[r['iter']] = r.get('score') or {}
    return dict(sorted(rows.items()))


def pool_series(path, opp):
    rows = []
    for k, sc in pool_scores(path).items():
        p = sum(sc[o]['points'] for o in opp.split(',') if o in sc)
        g = sum(sc[o]['games'] for o in opp.split(',') if o in sc)
        if g:
            rows.append((k, p, g))
    return rows


def newest_after_perfect(path, opp, min_games=MIN_PERFECT_GAMES):
    """The pool series against opp up to and including the first iteration k by which the learner won every one of
    its latest games against opp (iterations k, k-1, ... back to at least min_games games), then, from the next
    iteration on, against the newest version (vN) that iteration's pool games played. Returns (rows against opp
    [(k, points, games)], rows after the switch [(k, points, games, opponent)], the switch iteration or None)."""
    before, after, switch = [], [], None
    for k, sc in pool_scores(path).items():
        if switch is None:
            p, g = sc.get(opp, {}).get('points', 0), sc.get(opp, {}).get('games', 0)
            if g:
                before.append((k, p, g))
                if perfect_run(before, min_games):
                    switch = k
            continue
        vs = [o for o in sc if re.fullmatch(r'v\d+', o) and sc[o].get('games')]
        if vs:
            o = max(vs, key=lambda o: int(o[1:]))
            after.append((k, sc[o]['points'], sc[o]['games'], o))
    return before, after, switch


def perfect_run(rows, min_games):
    """The latest rows [(k, points, games)] that hold at least min_games games, if the learner won all of them."""
    n = 0
    for i in range(len(rows) - 1, -1, -1):
        if rows[i][1] != rows[i][2]:
            return None
        n += rows[i][2]
        if n >= min_games:
            return rows[i:]
    return None


def pool_status(path, opp='v0', min_games=MIN_PERFECT_GAMES):
    """The status sentence for scripts/rl_chart_sync.sh: pooled score against opp, and after the switch
    (newest_after_perfect) the pooled score against the newest version with the latest iteration's game."""
    before, after, switch = newest_after_perfect(path, opp, min_games)
    if not before:
        return ''
    p, n = sum(r[1] for r in before), sum(r[2] for r in before)
    s = f"Pool games against {opp}, iterations {before[0][0]}–{before[-1][0]}: {p:g}/{n} = {p / n:.2f}"
    if switch is None:
        return s
    run = perfect_run(before, min_games)
    g = sum(r[2] for r in run)
    s += (f"; the learner won all {g} in iteration{'s' if len(run) > 1 else ''} "
          + (f"{run[0][0]}–{switch}" if len(run) > 1 else f"{switch}")
          + ", so the chart now compares against the newest pool version")
    if after:
        P, N = sum(r[1] for r in after), sum(r[2] for r in after)
        k, pk, gk, o = after[-1]
        s += (f": iteration {k} against {o}: {pk:g}/{gk}" if len(after) == 1 else
              f": iterations {after[0][0]}–{k} {P:g}/{N} = {P / N:.2f} (iteration {k} against {o}: {pk:g}/{gk})")
    return s


def gate_series(path, mode):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r.get('mode') == mode and r.get('games'):
            rows[r['version']] = (r['version'], r['points'], r['games'])
    return sorted(rows.values())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('progress'); ap.add_argument('--out', required=True); ap.add_argument('--opp', default='v0')
    ap.add_argument('--gates'); ap.add_argument('--gate-mode', default='anchor-zdb')
    ap.add_argument('--gate-label', default=None)
    ap.add_argument('--title', default=None); ap.add_argument('--mark', action='append', default=[])
    ap.add_argument('--pool-label', default=None); ap.add_argument('--xlabel', default='learner version (iteration)')
    ap.add_argument('--newest-after-perfect', action='store_true')
    ap.add_argument('--newest-label', default='vs the newest pool version (pool games)')
    a = ap.parse_args(argv)
    pool_label = a.pool_label or f'vs {a.opp} (pool games)'
    # series: (name, rows [(k, points, games, opponent or None)], x offset, colour slot)
    if a.newest_after_perfect:
        if ',' in a.opp:
            raise SystemExit('--newest-after-perfect takes one --opp')
        before, after, switch = newest_after_perfect(a.progress, a.opp)
        series = [(pool_label, [r + (None,) for r in before], 0.0, 0), (a.newest_label, after, 0.0, 2)]
        if switch is not None:
            a.mark.append(f'{switch + 0.5}:{a.opp} 100% at {switch}; then the newest version')
    else:
        series = [(pool_label, [r + (None,) for r in pool_series(a.progress, a.opp)], 0.0, 0)]
    if a.gates:
        series.append((a.gate_label or f'{a.gate_mode} gate',
                       [r + (None,) for r in gate_series(a.gates, a.gate_mode)], 0.12, 1))
    series = [s for s in series if s[1]]
    if not series:
        raise SystemExit('nothing to plot')

    W, L, R, B = 720, 64, 24, 56
    T = 84 + 18 * (len(series) - 1)       # one legend row per series
    H = T + 240 + B
    pw, ph = W - L - R, H - T - B
    kmax = max(r[0] for _, rows, _, _ in series for r in rows) + 1
    X = lambda k: L + pw * k / kmax
    Y = lambda v: T + ph * (1 - v)
    title = a.title or 'Learner score by version'
    css_l = ''.join(f'--s{i}:{c[0]};' for i, c in enumerate(COLORS))
    css_d = ''.join(f'--s{i}:{c[1]};' for i, c in enumerate(COLORS))
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="{title}">',
         '<style>',
         f'svg{{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;{css_l}font-family:'
         '-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}',
         f'@media (prefers-color-scheme: dark){{svg{{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#3a3936;{css_d}}}}}',
         '.t{fill:var(--ink);font-size:15px;font-weight:600}.st{fill:var(--ink2);font-size:12px}'
         '.ax{fill:var(--ink2);font-size:11px}.lg{fill:var(--ink);font-size:12px}.gl{stroke:var(--grid);stroke-width:1}'
         '.ref{stroke:var(--ink2);stroke-width:1;stroke-dasharray:4 4}.mk{stroke:var(--ink2);stroke-width:1;stroke-dasharray:2 3}'
         + ''.join(f'.wh{i}{{stroke:var(--s{i});stroke-width:2;stroke-linecap:round;opacity:.5}}'
                   f'.dot{i}{{fill:var(--s{i});stroke:var(--surface);stroke-width:2}}'
                   f'.ln{i}{{stroke:var(--s{i});stroke-width:2;fill:none;opacity:.45}}' for i in range(len(COLORS))),
         '</style>',
         f'<rect width="{W}" height="{H}" fill="var(--surface)"/>',
         f'<text class="t" x="{L}" y="24">{title}</text>',
         f'<text class="st" x="{L}" y="43">Score (win 1, draw 0.5) with Wilson 95% interval; hover a point for its numbers.</text>']
    # legend (always present for 2+ series; text in ink, the swatch carries identity)
    for j, (name, rows, _, c) in enumerate(series):
        tp, tg = sum(r[1] for r in rows), sum(r[2] for r in rows)
        label = f'{name}: pooled {tp:g}/{tg} = {tp / tg:.2f}'
        y = 62 + 18 * j
        o.append(f'<circle class="dot{c}" cx="{L + 5}" cy="{y}" r="5"/><text class="lg" x="{L + 15}" y="{y + 4}">{label}</text>')
    for v in (0, .25, .5, .75, 1):
        o.append(f'<line class="gl" x1="{L}" x2="{L + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        o.append(f'<text class="ax" x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    o.append(f'<line class="ref" x1="{L}" x2="{L + pw}" y1="{Y(.5):.1f}" y2="{Y(.5):.1f}"/>')
    o.append(f'<text class="ax" x="{L + pw}" y="{Y(.5) - 6:.1f}" text-anchor="end">even</text>')
    step = max(1, round(kmax / 12))
    for k in range(0, kmax + 1, step):
        o.append(f'<text class="ax" x="{X(k):.1f}" y="{T + ph + 18}" text-anchor="middle">{k}</text>')
    o.append(f'<text class="ax" x="{L + pw / 2}" y="{H - 14}" text-anchor="middle">{a.xlabel}</text>')
    o.append(f'<text class="ax" transform="translate(16 {T + ph / 2}) rotate(-90)" text-anchor="middle">score</text>')
    for m in a.mark:
        k, txt = m.split(':', 1)
        x = X(float(k))
        o.append(f'<line class="mk" x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{T + ph}"/>')
        o.append(f'<text class="ax" x="{x + 4:.1f}" y="{T + 12}">{txt}</text>')
    for name, rows, dx, c in series:
        o.append(f'<polyline class="ln{c}" points="' + ' '.join(f'{X(r[0] + dx):.1f},{Y(r[1] / r[2]):.1f}' for r in rows) + '"/>')
        for k, p, g, opp in rows:
            lo, hi = wilson(p / g, g)
            x = X(k + dx)
            o.append(f'<g><title>v{k} {f"vs {opp}" if opp else name}: {p:g}/{g} = {p / g:.2f} (95% {lo:.2f}-{hi:.2f})</title>'
                     f'<line class="wh{c}" x1="{x:.1f}" x2="{x:.1f}" y1="{Y(lo):.1f}" y2="{Y(hi):.1f}"/>'
                     f'<circle class="dot{c}" cx="{x:.1f}" cy="{Y(p / g):.1f}" r="5"/></g>')
    o.append('</svg>')
    open(a.out, 'w').write('\n'.join(o) + '\n')
    print(f'wrote {a.out}: ' + '; '.join(f'{n} {len(r)} points' for n, r, _, _ in series))


if __name__ == '__main__':
    main()
