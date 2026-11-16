#!/usr/bin/env python3
"""Plot head-to-head scores, one row per test, as a standalone SVG dot plot with 95% intervals (sorted by score).

  scripts/plot_points.py ROWS.json --out docs/img/NAME.svg [--title T] [--subtitle S] [--groups G1,G2]

ROWS.json holds {"rows": [{label, points, games, score, t, lo, hi, group?}, ...]} (e.g. the judge's pair_score fields).
A row's group picks its colour (the --groups order fixes the colour order; a legend shows each group). The dashed line
is 0.5 (even). No dependencies (plain Python, hand-written SVG that follows the viewer's light/dark theme; hover a row
for its numbers).
"""
import argparse, json

# categorical slots 1 and 2 of the dataviz reference palette (light, dark), as in plot_rl_vs.py
COLORS = [('#2a78d6', '#3987e5'), ('#eb6834', '#d95926')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rows')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='Score by test')
    ap.add_argument('--subtitle', default='Score (win 1, draw 0.5) with 95% interval; hover a row for its numbers.')
    ap.add_argument('--groups', default='')
    a = ap.parse_args()
    rows = sorted(json.load(open(a.rows))['rows'], key=lambda r: (r['score'], r['label']))
    groups = [g for g in a.groups.split(',') if g] or sorted({r.get('group', '') for r in rows})
    gi = {g: i % len(COLORS) for i, g in enumerate(groups)}
    legend = len([g for g in groups if g]) >= 2

    rh = 26
    W = 760
    L, R, T, B = 190, 200, 96 if legend else 72, 44
    pw = W - L - R
    H = T + rh * len(rows) + B
    X = lambda v: L + pw * v
    css_l = ''.join(f'--s{i}:{c[0]};' for i, c in enumerate(COLORS))
    css_d = ''.join(f'--s{i}:{c[1]};' for i, c in enumerate(COLORS))
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="{a.title}">',
         '<style>',
         f'svg{{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;{css_l}font-family:'
         '-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}',
         f'@media (prefers-color-scheme: dark){{svg{{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#3a3936;{css_d}}}}}',
         '.t{fill:var(--ink);font-size:15px;font-weight:600}.st{fill:var(--ink2);font-size:12px}'
         '.ax{fill:var(--ink2);font-size:11px}.lb{fill:var(--ink);font-size:12px}.lg{fill:var(--ink);font-size:12px}'
         '.gl{stroke:var(--grid);stroke-width:1}.ref{stroke:var(--ink2);stroke-width:1;stroke-dasharray:4 4}'
         + ''.join(f'.wh{i}{{stroke:var(--s{i});stroke-width:2;stroke-linecap:round;opacity:.5}}'
                   f'.dot{i}{{fill:var(--s{i});stroke:var(--surface);stroke-width:2}}' for i in range(len(COLORS))),
         '</style>',
         f'<rect width="{W}" height="{H}" fill="var(--surface)"/>',
         f'<text class="t" x="16" y="24">{a.title}</text>',
         f'<text class="st" x="16" y="43">{a.subtitle}</text>']
    if legend:
        lx = 16
        for g in groups:
            if not g:
                continue
            o.append(f'<circle class="dot{gi[g]}" cx="{lx + 5}" cy="66" r="5"/><text class="lg" x="{lx + 15}" y="70">{g}</text>')
            lx += 40 + 6.6 * len(g)
    y0, y1 = T - 8, T + rh * len(rows)
    for v in (0, .25, .5, .75, 1):
        o.append(f'<line class="gl" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{y0}" y2="{y1}"/>')
        o.append(f'<text class="ax" x="{X(v):.1f}" y="{y1 + 16}" text-anchor="middle">{v:.2f}</text>')
    o.append(f'<line class="ref" x1="{X(.5):.1f}" x2="{X(.5):.1f}" y1="{y0}" y2="{y1}"/>')
    o.append(f'<text class="ax" x="{X(.5) + 4:.1f}" y="{y0 + 2}">even</text>')
    o.append(f'<text class="ax" x="{L + pw / 2}" y="{H - 10}" text-anchor="middle">score</text>')
    for j, r in enumerate(rows):
        y = T + rh * j + rh / 2
        i = gi.get(r.get('group', ''), 0)
        g = r['games']
        num = f"{r['points']:g}/{g} = {r['score']:.2f}" + (f", t {r['t']:+.1f}" if 't' in r else '')
        o.append(f'<g><title>{r["label"]}: {num} (95% {r["lo"]:.2f}-{r["hi"]:.2f})</title>'
                 f'<text class="lb" x="{L - 12}" y="{y + 4:.1f}" text-anchor="end">{r["label"]}</text>'
                 f'<line class="wh{i}" x1="{X(r["lo"]):.1f}" x2="{X(r["hi"]):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
                 f'<circle class="dot{i}" cx="{X(r["score"]):.1f}" cy="{y:.1f}" r="5"/>'
                 f'<text class="ax" x="{L + pw + 12}" y="{y + 4:.1f}">{num}</text></g>')
    o.append('</svg>')
    open(a.out, 'w').write('\n'.join(o) + '\n')
    print(f'wrote {a.out}: {len(rows)} rows')


if __name__ == '__main__':
    main()
