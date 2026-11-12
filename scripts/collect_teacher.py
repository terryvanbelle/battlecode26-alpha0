"""Collect zdb demonstration games in parallel (resumable). Usage: collect_teacher.py CONFIG [--games N] [--workers K]"""
import argparse, json, os, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TZ', 'PDT+7')   # log in PDT (UTC-7, BC's permanent time; user preference)
time.tzset()


def field(path):
    out = {}
    for line in Path(os.path.expanduser(path)).read_text().splitlines():
        parts = line.split('\t')
        if len(parts) >= 3:
            out[parts[0]] = (parts[2], parts[1])   # package, classes dir
    return out


def schedule(cfg):
    maps = sorted(p.stem for p in (ROOT / 'maps').glob('*.map26') if p.stem not in cfg['excluded_maps'])
    games = []
    for g in range(cfg['games']):
        rng = random.Random(cfg['seed_base'] + g)
        mirror = rng.random() < cfg['mirror_fraction']
        opp = 'zdb' if mirror else rng.choice(cfg['train_opponents'])
        zside = 'AB' if mirror else rng.choice('AB')
        games.append({'id': f'g{g:05d}', 'map': rng.choice(maps), 'seed': cfg['seed_base'] + g, 'opponent': opp,
                      'teacher_sides': zside, 'split': 'val' if g % cfg['validation_every'] == 0 else 'train'})
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--games', type=int)
    ap.add_argument('--workers', type=int)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.games:
        cfg['games'] = a.games
    run = ROOT / 'runs' / cfg['name']
    (run / 'games').mkdir(parents=True, exist_ok=True)
    fld = field(cfg['field_tsv'])
    t = cfg['teacher']
    bots = {'zdb': (t['package'], str(ROOT / t['classes']))}
    for name in cfg['train_opponents']:
        bots[name] = fld[name]
    games = schedule(cfg)
    (run / 'schedule.json').write_text(json.dumps(games, indent=1))
    todo = [g for g in games if not (run / 'games' / g['id'] / 'result.json').exists()]
    print(f'{len(games)} scheduled, {len(todo)} to run', flush=True)
    flags = '-Xmx2g' + (' -Dbc.alpha0.tournamentLimits=true' if cfg['tournament_limits'] else '')

    def play(g):
        out = run / 'games' / g['id']
        if out.exists():
            for f in out.iterdir():
                f.unlink()   # incomplete previous attempt
        zdb, opp = bots['zdb'], bots[g['opponent']]
        a, b = (zdb, opp) if 'A' in g['teacher_sides'] else (opp, zdb)
        cmd = [str(ROOT / 'scripts/student_java.sh'), '--map', g['map'], '--maps-dir', str(ROOT / 'maps'), '--seed', str(g['seed']),
               '--a', f'bot:{a[0]}:{a[1]}', '--b', f'bot:{b[0]}:{b[1]}', '--out', str(out),
               '--record', g['teacher_sides'], '--record-p', str(cfg['record_p']),
               '--record-p-king', str(cfg.get('record_p_king', cfg['record_p']))]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg['game_timeout_seconds'], env={**os.environ, 'JAVA_FLAGS': flags})
            ok = (out / 'result.json').exists()
            if not ok:
                (run / 'failures.log').open('a').write(f"{g['id']} rc={p.returncode} {p.stderr[-2000:]}\n")
        except subprocess.TimeoutExpired:
            ok = False
            (run / 'failures.log').open('a').write(f"{g['id']} timeout\n")
        return g, ok, time.time() - t0

    done = len(games) - len(todo)
    with ThreadPoolExecutor(a.workers or cfg['workers']) as ex:
        for fut in as_completed([ex.submit(play, g) for g in todo]):
            g, ok, secs = fut.result()
            done += ok
            print(f"{time.strftime('%H:%M:%S')} {g['id']} {g['map']} vs {g['opponent']} ok={ok} {secs:.0f}s  [{done}/{len(games)}]", flush=True)
            (run / 'status.json').write_text(json.dumps({'done': done, 'scheduled': len(games), 'updated': time.time()}))


if __name__ == '__main__':
    main()
