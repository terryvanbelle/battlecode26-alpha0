"""Play a player against opponent groups on maps x seeds x sides; resumable; prints a summary.

eval_student.py --player net:runs/student-v1/best.bin:0 --opponents a,b,c --maps m1,m2 --seeds 1,2 --out runs/eval/NAME
Opponent names: field bot ids (FIELD.tsv), examplefuncsplayer, lectureplayer, zdb, or @untrained / net:PATH:T.
"""
import argparse, json, os, subprocess, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TZ', 'PDT+7')   # log in PDT (UTC-7, BC's permanent time; user preference)
time.tzset()
BC26B = Path(os.path.expanduser('~/projects/vibe/2026b'))


def compile_local(name):
    """Compile a local bot (java/bots/NAME in this repo, else bots/NAME in 2026b); rebuilds when a source changed."""
    dest = ROOT / 'build/bots' / name
    src = ROOT / 'java/bots' / name
    if not src.is_dir():
        src = BC26B / 'bots' / name
    files = sorted(src.glob('*.java'))
    stamp = dest / '.stamp'
    newest = max(f.stat().st_mtime for f in files)
    if not (dest / name).exists() or (src.parent == ROOT / 'java/bots' and (not stamp.exists() or stamp.stat().st_mtime < newest)):
        tmp = dest.with_name(name + '.tmp')
        subprocess.run(['rm', '-rf', str(tmp)], check=True)
        tmp.mkdir(parents=True)
        subprocess.run([str(ROOT / 'vendor/jdk/bin/javac'), '--release', '21', '-nowarn', '-proc:none', '-cp',
                        str(ROOT / 'vendor/engine/battlecode26-java-1.2.6.jar'), '-d', str(tmp), *map(str, files)], check=True)
        (tmp / '.stamp').touch()
        subprocess.run(['rm', '-rf', str(dest)], check=True)
        tmp.rename(dest)
    return f'bot:{name}:{dest}'


def resolve(name, untrained):
    if name.startswith(('net:', 'bot:')):
        return name
    if name == '@untrained':
        return f'net:{untrained}:0'
    if name in ('examplefuncsplayer', 'lectureplayer', 'zdb') or (ROOT / 'java/bots' / name).is_dir():
        return compile_local(name)
    for line in (BC26B / 'research/field/FIELD.tsv').read_text().splitlines():
        p = line.split('\t')
        if p[0] == name:
            return f'bot:{p[2]}:{p[1]}'
    raise SystemExit(f'unknown opponent {name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--player', required=True)
    ap.add_argument('--opponents', required=True)
    ap.add_argument('--maps', required=True)
    ap.add_argument('--seeds', default='11,12')
    ap.add_argument('--sides', default='AB')
    ap.add_argument('--out', required=True)
    ap.add_argument('--untrained', default='')
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--limits', action='store_true', help='official bytecode limits for bot teams')
    ap.add_argument('--slice', help='LO:HI:N: only jobs with LO <= index %% N < HI (split across machines; summary covers those)')
    a = ap.parse_args()
    out = Path(a.out); (out / 'games').mkdir(parents=True, exist_ok=True)
    player = resolve(a.player, a.untrained)
    opps = {o: resolve(o, a.untrained) for o in a.opponents.split(',')}
    jobs = []
    for o, spec in opps.items():
        for m in a.maps.split(','):
            for s in a.seeds.split(','):
                for side in a.sides:
                    jobs.append(dict(id=f'{o}__{m}__{s}__{side}', opponent=o, map=m, seed=int(s), side=side, opp_spec=spec))
    (out / 'jobs.json').write_text(json.dumps({'player': player, 'jobs': jobs}, indent=1))
    if a.slice:
        lo, hi, n = map(int, a.slice.split(':'))
        jobs = [j for i, j in enumerate(jobs) if lo <= i % n < hi]
    flags = '-Xmx2g' + (' -Dbc.alpha0.tournamentLimits=true' if a.limits else '')

    def play(j):
        d = out / 'games' / j['id']
        if (d / 'result.json').exists():
            return j, json.loads((d / 'result.json').read_text())
        A, B = (player, j['opp_spec']) if j['side'] == 'A' else (j['opp_spec'], player)
        cmd = [str(ROOT / 'scripts/student_java.sh'), '--map', j['map'], '--maps-dir', str(ROOT / 'maps'), '--seed', str(j['seed']),
               '--a', A, '--b', B, '--out', str(d)]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env={**os.environ, 'JAVA_FLAGS': flags})
        except subprocess.TimeoutExpired:
            pass
        r = d / 'result.json'
        return j, (json.loads(r.read_text()) if r.exists() else None)

    score = defaultdict(lambda: [0.0, 0, 0])  # points, games, invalid
    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(play, j) for j in jobs]):
            j, r = fut.result()
            s = score[j['opponent']]
            if r is None or r.get('error') or not r.get('finished') or r['A'].get('failures') or r['B'].get('failures'):
                s[2] += 1
                continue
            s[1] += 1
            s[0] += 0.5 if r['winner'] is None else float(r['winner'] == j['side'])
    summary = {o: {'score': round(p / g, 3) if g else None, 'points': p, 'games': g, 'invalid': bad} for o, (p, g, bad) in score.items()}
    (out / 'summary.json').write_text(json.dumps({'player': player, 'summary': summary}, indent=1))
    for o, v in sorted(summary.items()):
        print(f"{o:34s} {v['points']:5.1f}/{v['games']:3d}  {v['score']}  invalid={v['invalid']}")


if __name__ == '__main__':
    main()
