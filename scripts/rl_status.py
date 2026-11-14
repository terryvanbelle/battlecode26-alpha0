"""Read-only status of an RL run (scripts/rl_loop.sh): one block per iteration, then the anchor/promotion gates.

    rl_status.py RUN [--runs runs/rl] [--last N] [--top 3]

Per iteration (iter-NNNN/data/summary.json, nets/<k+1>/metrics.json, stages.tsv):
  games, learner score vs each pool member (mirror games are 0.5 by construction and are not listed), wall time;
  per trainable point: records used, KL to the anchor (start -> end), KL old -> new, clip fraction, entropy
  (start -> end), gradient SNR cosine, early stop, alarms, and the top options by mean P (end of training);
  the value model's Brier (train / val, against the constant predictor) and AUC.
Gates (gates.jsonl): mode, version, points / games, score with its Wilson 95% CI, pass.
An iteration still in progress shows its collected-game count and current stage.
"""
import argparse, json, math, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def js(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def wilson(pts, g, z=1.96):
    if not g:
        return 0.0, 0.0
    s = pts / g
    den = 1 + z * z / g
    c = (s + z * z / (2 * g)) / den
    h = z * math.sqrt(s * (1 - s) / g + z * z / (4 * g * g)) / den
    return c - h, c + h


def f(v, fmt='.3f'):
    return '-' if v is None else format(v, fmt)


def stages(I):
    out = {}
    p = I / 'stages.tsv'
    if p.exists():
        for line in p.read_text().split('\n'):
            if line.strip():
                n, a, b = line.split()
                out[n] = (int(a), int(b))
    return out


def iteration(rd, I, k, top):
    lines = []
    st = stages(I)
    wall = (max(b for _, b in st.values()) - min(a for a, _ in st.values())) if st else None
    s = js(I / 'data' / 'summary.json')
    if s is None:
        jobs = js(I / 'jobs.json')
        n = len(list((I / 'games').glob('*/result.json'))) if (I / 'games').exists() else 0
        stage = next((m for m in ('collected', 'scheduled') if (I / f'.{m}').exists()), 'building')
        lines.append(f'iter {k}: in progress (last stage done: {stage}), {n}/{len(jobs["jobs"]) if jobs else "?"} games')
        return lines
    sc = s.get('score') or {}
    vs = ', '.join(f"{o} {v['points']:g}/{v['games']} = {v['score']:.3f}" for o, v in sorted(sc.items(), key=lambda kv: kv[0]))
    secs = ' '.join(f'{n} {b - a}s' for n, (a, b) in st.items())
    lines.append(f"iter {k}: {s.get('valid')} valid games ({s.get('invalid', 0)} invalid); learner vs pool: {vs or 'mirror only'}; "
                 f"wall {f(wall / 60 if wall else None, '.1f')} min [{secs}]")
    m = js(rd / 'nets' / str(k + 1) / 'metrics.json')
    rb = js(rd / 'nets' / str(k + 1) / 'ROLLBACK.json')
    if rb:
        lines.append(f'  ROLLBACK: {rb}')
    if m is None:
        lines.append('  (not trained yet)')
        return lines
    v = m.get('value') or {}
    lines.append(f"  value: Brier train {f(v.get('train_brier'))} val {f(v.get('val_brier'))} (const {f(v.get('val_brier_const'))}), "
                 f"AUC {f(v.get('val_auc'))}, rows {v.get('rows')}; exported {m.get('exported')}, refused {m.get('refused')}")
    for p, d in sorted((m.get('points') or {}).items(), key=lambda kv: int(kv[0])):
        mp = d.get('mean_probs_end') or d.get('mean_probs_start') or []
        ps = d.get('mean_probs_start') or []
        best = sorted(range(len(mp)), key=lambda i: -mp[i])[:top]
        tops = ' '.join(f"{i}:{mp[i]:.3f}" + (f"({ps[i]:.3f})" if i < len(ps) else '') for i in best)
        lines.append(f"  p{p:>2}: used {d.get('used')}, KL anchor {f(d.get('kl_anchor_start'), '.4f')}->{f(d.get('kl_anchor_end'), '.4f')}, "
                     f"KL old->new {f(d.get('kl_old_new'), '.4f')}, clip {f(d.get('clip_frac'))}, entropy "
                     f"{f(d.get('entropy_start'))}->{f(d.get('entropy_end'))}, SNR cos {f(d.get('grad_snr_cos'))}, steps {d.get('steps')}, "
                     f"early stop {'yes' if d.get('early_stop') else 'no'}, alarms {d.get('alarms') or []}")
        lines.append(f'       top P end(start): {tops}')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('--runs', default=str(ROOT / 'runs' / 'rl'))
    ap.add_argument('--last', type=int, default=0, help='only the last N iterations')
    ap.add_argument('--top', type=int, default=3, help='top options shown per point')
    a = ap.parse_args()
    rd = Path(a.runs) / a.run
    its = sorted(rd.glob('iter-[0-9]*'))
    if a.last:
        its = its[-a.last:]
    state = js(rd / 'state.json') or {}
    cfg = js(rd / 'config.json') or {}
    stop = (rd / 'STOP').exists() or (rd.parent / 'STOP').exists()
    print(f"run {a.run}: {len(list(rd.glob('iter-*/.done')))} iterations done, trainable {cfg.get('trainable_points')}, "
          f"state {state}{', STOP file present' if stop else ''}; now {time.strftime('%m-%d %H:%M', time.gmtime(time.time() - 7 * 3600))} PDT")
    for I in its:
        for line in iteration(rd, I, int(I.name.split('-')[1]), a.top):
            print(line)
    g = rd / 'gates.jsonl'
    rows = [json.loads(l) for l in g.read_text().split('\n') if l.strip()] if g.exists() else []
    print(f'gates ({len(rows)}):' if rows else 'gates: none yet')
    for r in rows:
        if r.get('mode') == 'strength':
            print(f"  strength v{r.get('version')}: field {r['field'].get('score')} vs ref {r['ref_field'].get('score')}, "
                  f"zdb {r['zdb'].get('score')}, pass {r.get('pass')}")
            continue
        lo, hi = (r['lo'], r['hi']) if 'lo' in r else wilson(r.get('points', 0), r.get('games', 0))
        print(f"  {r.get('mode')} v{r.get('version')}: {r.get('points'):g}/{r.get('games')} = {r.get('score', 0):.3f} "
              f"[{lo:.3f}, {hi:.3f}], invalid {r.get('invalid', 0)}, pass {r.get('pass')}")


if __name__ == '__main__':
    main()
