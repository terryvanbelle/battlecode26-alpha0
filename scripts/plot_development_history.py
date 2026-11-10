"""Per-bot, equal-map win-rate history on one frozen development suite."""
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
        result[bot] = {**counts, 'valid': len(valid), 'expected': len(expected),
                       'errors': len(rr)-len(valid), 'complete': complete,
                       'win_rate': counts['wins']/len(expected) if complete else None}
    return {'checkpoint_sha256': rows[0]['checkpoint_sha256'],
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
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    labels = [datetime.fromisoformat(h['started_at'].replace('Z','+00:00')).astimezone(ZoneInfo('America/Vancouver')).strftime('%b %d %H:%M')+'\n'+h['checkpoint_sha256'][:8] for h in history]
    for ax, bot in zip(axes.flat, ordered):
        # Separate lines per runtime; gaps remain gaps, never estimated rates.
        for runtime in sorted({h['evaluation_config_sha256'] for h in history}):
            ys = [100*h['bots'][bot['id']]['win_rate'] if h['evaluation_config_sha256']==runtime and h['bots'][bot['id']]['complete'] else float('nan') for h in history]
            ax.plot(range(len(history)), ys, 'o-', color='#2374AB', linewidth=2)
        for i,h in enumerate(history):
            c=h['bots'][bot['id']]
            if c['complete']:
                y=100*c['win_rate'];ax.annotate(f"{y:.1f}%\n{c['wins']}W/{c['losses']}L/{c['draws']}D",(i,y),xytext=(0,10 if y<80 else -30),textcoords='offset points',ha='center',fontsize=8)
            else:
                ax.text(i,50,f"Pending\n{c['valid']}/{c['expected']} valid\n{c['errors']} errors",ha='center',va='center',fontsize=9,color='#8c5318')
        ax.set_title(f"{bot['id']}\nPrior Elo {bot['historical_rating']['elo']:.1f}",fontsize=10)
        ax.set_ylim(-5,110);ax.set_xlim(-.6,max(.6,len(history)-.4));ax.set_yticks([0,25,50,75,100]);ax.grid(alpha=.2)
        ax.set_xticks(range(len(history)),labels,fontsize=8)
        ax.tick_params(axis='x',labelbottom=True)
    fig.supylabel('Win rate (%) · all eight maps');fig.supxlabel('Evaluation start (America/Vancouver) · checkpoint SHA')
    fig.suptitle(f"Per-bot win rate over time · {manifest['suite_id']}\n32 games per point: 8 maps × 2 seeds × both seats. Incomplete bot panels have no rate.",fontsize=13)
    fig.tight_layout(rect=(.015,.03,1,.92))
    out=ROOT/'progress/overview';out.mkdir(exist_ok=True)
    fig.savefig(out/'development-history.png',dpi=160);plt.close(fig)
    (out/'development-history.json').write_text(json.dumps({'suite_id':manifest['suite_id'],'aggregation':'wins / all scheduled games; equal map weights; draws are not wins','evaluations':history},indent=2)+'\n')
