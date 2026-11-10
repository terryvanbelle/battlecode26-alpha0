"""Per-bot win-rate history on one frozen development suite."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/battlecode-mpl')
ROOT = Path(__file__).resolve().parents[1]


def summarize(rows, manifest):
    schedule = manifest['evaluation_protocol']['standard']
    expected = {(m, s, t) for m in schedule['maps'] for s in schedule['seeds']
                for t in schedule.get('sides', ['A', 'B'])}
    bots = {b['id']: b for b in manifest['bots']}
    if not rows:
        return None
    if {r['benchmark_suite_id'] for r in rows} != {manifest['suite_id']}:
        raise ValueError('mixed or incompatible suite')
    for field in ['checkpoint_sha256', 'evaluation_config_sha256', 'panel_fingerprint_sha256']:
        if len({r[field] for r in rows}) != 1:
            raise ValueError('mixed ' + field)
    cells = {}
    for r in rows:
        key = (r['opponent'], r['map'], r['seed'], r['side'])
        if key in cells or key[0] not in bots or key[1:] not in expected:
            raise ValueError('duplicate or unexpected cell')
        b = bots[r['opponent']]
        if (r['opponent_java_sha256'], r['opponent_source_commit']) != (b['java_tree_sha256'], b['source_commit']):
            raise ValueError('changed opponent')
        cells[key] = r
    result = {}
    for bot in bots:
        rr = [r for r in rows if r['opponent'] == bot]
        valid = [r for r in rr if r['status'] == 'ok' and r['score'] in (0, .5, 1)]
        complete = len(valid) == len(expected)
        counts = {name: sum(r['score'] == score for r in valid)
                  for name, score in [('wins', 1), ('losses', 0), ('draws', .5)]}
        crash_exclusion = (bot == 'thewaxmango_collegejunior' and len(rr) == len(expected)
                           and all(r['status'] == 'ok' or r.get('error_kind') == 'opponent_runtime_error' for r in rr))
        rate_available = bool(valid) and (complete or crash_exclusion)
        result[bot] = {**counts, 'valid': len(valid), 'expected': len(expected),
                       'errors': len(rr)-len(valid), 'complete': complete,
                       'excluded_crashes': len(rr)-len(valid) if crash_exclusion else 0,
                       'win_rate': counts['wins']/len(valid) if rate_available else None}
    overall = {key: sum(bot[key] for bot in result.values())
               for key in ('wins', 'losses', 'draws', 'valid', 'expected', 'errors', 'excluded_crashes')}
    overall['complete'] = all(bot['complete'] for bot in result.values())
    overall['win_rate'] = (overall['wins']/overall['valid']
                           if overall['valid'] and all(bot['win_rate'] is not None for bot in result.values()) else None)
    return {'overall': overall, 'checkpoint_sha256': rows[0]['checkpoint_sha256'],
            'evaluation_config_sha256': rows[0]['evaluation_config_sha256'],
            'started_at': min(r['started_at'] for r in rows), 'bots': result}


def generate(current, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime
    from zoneinfo import ZoneInfo
    paths = set((ROOT/'progress').glob('development-*/**/evaluations.jsonl'))
    paths.add(Path(current).resolve()/'evaluations.jsonl')
    history = []
    for path in sorted(paths):
        if not path.exists():
            continue
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        if not rows or rows[0]['benchmark_suite_id'] != manifest['suite_id']:
            continue
        item = summarize(rows, manifest)
        item.update(source=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(raw).hexdigest())
        history.append(item)
    history.sort(key=lambda x: (x['started_at'], x['source']))
    ordered = sorted(manifest['bots'], key=lambda b: (b['historical_rating']['elo'], b['id']))
    fig, ax = plt.subplots(figsize=(15, 6.5))
    labels = [datetime.fromisoformat(h['started_at'].replace('Z','+00:00')).astimezone(ZoneInfo('America/Vancouver')).strftime('%b %d %H:%M')+'\n'+h['checkpoint_sha256'][:8] for h in history]
    colors = plt.get_cmap('tab10').colors
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    for index, bot in enumerate(ordered):
        latest = history[-1]['bots'][bot['id']] if history else None
        label = f"{bot['id']} · Elo {bot['historical_rating']['elo']:.1f}"
        if latest and latest['win_rate'] is not None:
            label += f" · {100*latest['win_rate']:.1f}% ({latest['wins']}/{latest['valid']})"
            if latest['excluded_crashes']:
                label += f" · {latest['excluded_crashes']} crashes excluded"
        else:
            label += ' · pending'
        # Never connect measurements across different runtimes.
        for j, runtime in enumerate(sorted({h['evaluation_config_sha256'] for h in history})):
            ys = [100*h['bots'][bot['id']]['win_rate'] if h['evaluation_config_sha256']==runtime and h['bots'][bot['id']]['win_rate'] is not None else float('nan') for h in history]
            ax.plot(range(len(history)), ys, marker=markers[index % len(markers)],
                    color=colors[index % len(colors)], linewidth=2, markersize=8,
                    label=label if j == 0 else None)
    latest_overall = history[-1]['overall'] if history else None
    overall_label = 'Overall'
    if latest_overall and latest_overall['win_rate'] is not None:
        overall_label += f" · {100*latest_overall['win_rate']:.1f}% ({latest_overall['wins']}/{latest_overall['valid']})"
        if latest_overall['excluded_crashes']:
            overall_label += f" · {latest_overall['excluded_crashes']} crashes excluded"
    else:
        overall_label += ' · pending'
    for j, runtime in enumerate(sorted({h['evaluation_config_sha256'] for h in history})):
        ys = [100*h['overall']['win_rate'] if h['evaluation_config_sha256']==runtime and h['overall']['win_rate'] is not None else float('nan') for h in history]
        ax.plot(range(len(history)), ys, color='#111111', marker='o', linewidth=3.5,
                markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=10,
                label=overall_label if j == 0 else None)
    ax.set_ylim(-3,103);ax.set_xlim(-.5,max(.5,len(history)-.5))
    ax.set_yticks(range(0,101,10));ax.grid(alpha=.2)
    ax.set_xticks(range(len(history)),labels,fontsize=9)
    ax.set_ylabel('Win rate (%)');ax.set_xlabel('Evaluation start (America/Vancouver) · checkpoint SHA')
    denominator_note = 'thewaxmango crashes excluded from bot and overall rates' if any(h['overall']['excluded_crashes'] for h in history) else 'complete valid games; incomplete panels remain pending'
    ax.set_title(f"Overall and per-bot win rate over time · {manifest['suite_id']}\nEight-map schedule · {denominator_note}",fontsize=13)
    ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=9,
              title='Opponents by prior Elo · latest wins / valid games')
    fig.tight_layout()
    out=ROOT/'progress/overview';out.mkdir(exist_ok=True)
    fig.savefig(out/'development-history.png',dpi=160);plt.close(fig)
    (out/'development-history.json').write_text(json.dumps({'suite_id':manifest['suite_id'],'aggregation':'wins / valid games; thewaxmango opponent crashes excluded; other incomplete panels have no rate; draws are not wins','evaluations':history},indent=2)+'\n')
