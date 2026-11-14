#!/usr/bin/env python3
"""Plot an rl_loop run's learner score against one pool opponent (default v0) per iteration, as a standalone SVG.

  scripts/plot_rl_vs.py runs/rl/RUN/progress.jsonl --out docs/img/RUN-vs-v0.svg [--opp v0] [--title T] [--mark K:TEXT ...]

Each point is iteration k's pool games: the learner (version k) against the opponent, score = points / games (win 1,
draw 0.5), with a Wilson 95% interval. The dashed line is 0.5 (even). A pooled band over the last --window iterations
is drawn as a light step. No dependencies (plain Python, hand-written SVG that follows the viewer's light/dark theme).
"""
import argparse, json, math


def wilson(p, n, z=1.96):
    if not n:
        return 0.0, 1.0
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('progress'); ap.add_argument('--out', required=True); ap.add_argument('--opp', default='v0')
    ap.add_argument('--title', default=None); ap.add_argument('--mark', action='append', default=[])
    a = ap.parse_args()
    rows = []
    for line in open(a.progress):
        r = json.loads(line)
        s = (r.get('score') or {}).get(a.opp)
        if s and s.get('games'):
            rows.append((r['iter'], s['points'], s['games']))
    rows = sorted({k: (k, p, g) for k, p, g in rows}.values())
    if not rows:
        raise SystemExit(f'no games against {a.opp} in {a.progress}')
    tot_p, tot_g = sum(r[1] for r in rows), sum(r[2] for r in rows)

    W, H = 720, 360
    L, R, T, B = 64, 24, 64, 56
    pw, ph = W - L - R, H - T - B
    kmax = max(r[0] for r in rows) + 1
    X = lambda k: L + pw * k / kmax
    Y = lambda v: T + ph * (1 - v)
    title = a.title or f'Learner vs {a.opp}, by iteration'
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="{title}">',
         '<style>',
         'svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;--s1:#2a78d6;--s1band:#2a78d633;font-family:'
         '-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}',
         '@media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#3a3936;'
         '--s1:#3987e5;--s1band:#3987e540}}',
         '.t{fill:var(--ink);font-size:15px;font-weight:600}.st{fill:var(--ink2);font-size:12px}'
         '.ax{fill:var(--ink2);font-size:11px}.gl{stroke:var(--grid);stroke-width:1}'
         '.ref{stroke:var(--ink2);stroke-width:1;stroke-dasharray:4 4}.wh{stroke:var(--s1);stroke-width:2;stroke-linecap:round}'
         '.dot{fill:var(--s1);stroke:var(--surface);stroke-width:2}.ln{stroke:var(--s1);stroke-width:2;fill:none;opacity:.45}'
         '.mk{stroke:var(--ink2);stroke-width:1;stroke-dasharray:2 3}',
         '</style>',
         f'<rect width="{W}" height="{H}" fill="var(--surface)"/>',
         f'<text class="t" x="{L}" y="24">{title}</text>',
         f'<text class="st" x="{L}" y="43">Score per iteration (win 1, draw 0.5), Wilson 95% interval. '
         f'All iterations pooled: {tot_p:g}/{tot_g} = {tot_p / tot_g:.2f}</text>']
    for v in (0, .25, .5, .75, 1):
        o.append(f'<line class="gl" x1="{L}" x2="{L + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        o.append(f'<text class="ax" x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    o.append(f'<line class="ref" x1="{L}" x2="{L + pw}" y1="{Y(.5):.1f}" y2="{Y(.5):.1f}"/>')
    o.append(f'<text class="ax" x="{L + pw}" y="{Y(.5) - 6:.1f}" text-anchor="end">even</text>')
    step = max(1, round(kmax / 12))
    for k in range(0, kmax + 1, step):
        o.append(f'<text class="ax" x="{X(k):.1f}" y="{T + ph + 18}" text-anchor="middle">{k}</text>')
    o.append(f'<text class="ax" x="{L + pw / 2}" y="{H - 14}" text-anchor="middle">iteration (learner version)</text>')
    o.append(f'<text class="ax" transform="translate(16 {T + ph / 2}) rotate(-90)" text-anchor="middle">'
             f'score vs {a.opp}</text>')
    for m in a.mark:
        k, txt = m.split(':', 1)
        x = X(float(k))
        o.append(f'<line class="mk" x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{T + ph}"/>')
        o.append(f'<text class="ax" x="{x + 4:.1f}" y="{T + 12}">{txt}</text>')
    o.append('<polyline class="ln" points="' + ' '.join(f'{X(k):.1f},{Y(p / g):.1f}' for k, p, g in rows) + '"/>')
    for k, p, g in rows:
        lo, hi = wilson(p / g, g)
        o.append(f'<g><title>v{k} vs {a.opp}: {p:g}/{g} = {p / g:.2f} (95% {lo:.2f}-{hi:.2f})</title>'
                 f'<line class="wh" x1="{X(k):.1f}" x2="{X(k):.1f}" y1="{Y(lo):.1f}" y2="{Y(hi):.1f}" opacity=".5"/>'
                 f'<circle class="dot" cx="{X(k):.1f}" cy="{Y(p / g):.1f}" r="5"/></g>')
    o.append('</svg>')
    open(a.out, 'w').write('\n'.join(o) + '\n')
    print(f'wrote {a.out}: {len(rows)} iterations, pooled {tot_p:g}/{tot_g}')


if __name__ == '__main__':
    main()
