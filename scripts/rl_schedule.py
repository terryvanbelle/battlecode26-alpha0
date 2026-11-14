"""Schedule one RL self-play iteration (docs/rl-plan.md sections 3-4): update the opponent pool and its PFSP win rates from
the previous iteration's games, then write runs/rl/RUN/iter-NNNN/jobs.json.

rl_schedule.py --config configs/rl-v1.json --iter K [--run RUN] [--games N]     schedule iteration K (learner = version K)
rl_schedule.py --config configs/rl-v1.json --promote V [--run RUN]              mark version V as a promoted best
rl_schedule.py --config configs/rl-v1.json --show [--run RUN]                   print the pool

Iteration K's learner is net version K, built as package <package_prefix>K (zdpr<K>). Games come in units with seeds
seed_base + seed_iter_stride * K + unit:
  - mirror units (mirror_fraction of the games): one learner-vs-learner game, both sides recorded, pair -1;
  - pool units: a seat-swapped pair (same map and seed, learner as A then as B) against one pool member, the learner
    side recorded, pair = unit.
Pool (pool.json): version 0 (the start net, zdpn2) is permanent; after iteration K-1 its snapshot joins when
(K-1) % snapshot_every == 0; the last keep_last snapshots stay, plus version 0 and every promoted best. Opponents are
uniform until pfsp_from_iter, then PFSP-lite: weight ~ (1 - winrate)^pfsp_power, winrate an EMA (alpha = 2/(n+1),
n = winrate_ema_iters) of the learner's score against the member; version 0 gets at least base_min_share of the pool
games. Opponent counts are allocated by largest remainder (stratified, not iid). bootstrap_zdb > 0 (README-flagged
fallback, section 7) gives that fraction of the pool pairs to an unrecorded zdb.
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_dir(cfg, run, runs=''):
    return (Path(runs) if runs else ROOT / 'runs' / 'rl') / (run or cfg['name'])


def iter_dir(rd, k):
    return rd / f'iter-{k:04d}'


def write_json(path, obj):
    """Atomic json write."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def package(cfg, v):
    return f"{cfg.get('package_prefix', 'zdpr')}{v}"


def load_pool(cfg, rd):
    p = rd / 'pool.json'
    if p.exists():
        return json.loads(p.read_text())
    base = dict(name='v0', version=0, package=package(cfg, 0), added_iter=0, permanent=True, promoted=False, base=True)
    return dict(members=[base], winrate={}, history={}, scored_iters=[], best=0, promotions=[])


def learner_score(jobs_doc, games_dir):
    """{opponent name: [points, games]} for the learner in pool games with a valid result (mirror games skipped)."""
    out = {}
    for j in jobs_doc['jobs']:
        if j['kind'] != 'pool':
            continue
        r = games_dir / j['id'] / 'result.json'
        if not r.exists():
            continue
        res = json.loads(r.read_text())
        if res.get('error') or not res.get('finished') or res['A'].get('failures') or res['B'].get('failures'):
            continue
        side = j['learner_sides']
        pts = 0.5 if res.get('winner') is None else float(res['winner'] == side)
        s = out.setdefault(j['opponent'], [0.0, 0])
        s[0] += pts; s[1] += 1
    return out


def update_winrates(cfg, rd, pool, k):
    """Fold iteration k-1's pool results into the EMA win rates (once per iteration)."""
    prev = k - 1
    if prev < 0 or prev in pool['scored_iters']:
        return
    jp = iter_dir(rd, prev) / 'jobs.json'
    if not jp.exists():
        return
    sc = learner_score(json.loads(jp.read_text()), iter_dir(rd, prev) / 'games')
    n = cfg['pool'].get('winrate_ema_iters', 2)
    alpha = 2.0 / (n + 1)
    for name, (pts, g) in sc.items():
        if not g:
            continue
        s = pts / g
        old = pool['winrate'].get(name)
        pool['winrate'][name] = s if old is None else alpha * s + (1 - alpha) * old
        pool['history'].setdefault(name, []).append([prev, pts, g])
    pool['scored_iters'].append(prev)


def add_snapshot(cfg, pool, k):
    """After iteration k-1 finished, its learner joins the pool when (k-1) % snapshot_every == 0; trim old snapshots."""
    v = k - 1
    pc = cfg['pool']
    if v > 0 and v % pc.get('snapshot_every', 2) == 0 and not any(m['version'] == v for m in pool['members']):
        pool['members'].append(dict(name=f'v{v}', version=v, package=package(cfg, v), added_iter=k, permanent=False,
                                    promoted=False, base=False))
    snaps = [m for m in pool['members'] if not m['permanent'] and not m['promoted']]
    drop = {m['name'] for m in sorted(snaps, key=lambda m: m['version'])[:max(0, len(snaps) - pc.get('keep_last', 8))]}
    pool['members'] = [m for m in pool['members'] if m['name'] not in drop]


def weights(cfg, pool, k):
    """{member name: share of the pool games}."""
    pc = cfg['pool']
    mem = [m for m in pool['members'] if m['version'] < k]
    if not mem:
        return {}
    if k < pc.get('pfsp_from_iter', 10):
        w = {m['name']: 1.0 for m in mem}
    else:
        w = {m['name']: max(1e-3, (1.0 - pool['winrate'].get(m['name'], 0.5))) ** pc.get('pfsp_power', 1.0) for m in mem}
    tot = sum(w.values())
    w = {n: v / tot for n, v in w.items()}
    base = [m['name'] for m in mem if m.get('base')]
    lo = pc.get('base_min_share', 0.10)
    if base and len(w) > 1 and w[base[0]] < lo:
        rest = 1 - w[base[0]]
        w = {n: (lo if n == base[0] else v * (1 - lo) / rest) for n, v in w.items()}
    return w


def allocate(w, n):
    """Largest-remainder counts summing to n."""
    raw = {k: v * n for k, v in w.items()}
    cnt = {k: int(math.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - cnt[k]), k))[:n - sum(cnt.values())]:
        cnt[k] += 1
    return cnt


def schedule(cfg, rd, k, games):
    pool = load_pool(cfg, rd)
    update_winrates(cfg, rd, pool, k)
    add_snapshot(cfg, pool, k)
    learner = package(cfg, k)
    rng = random.Random(cfg['seed_base'] * 7 + k)
    n_pool_pairs = int(round(games * (1 - cfg['mirror_fraction']) / 2))
    w = weights(cfg, pool, k)
    if not w:   # iteration 0: nothing older than the learner; all mirror
        n_pool_pairs = 0
    n_mirror = games - 2 * n_pool_pairs
    zdb_pairs = int(round(n_pool_pairs * cfg.get('bootstrap_zdb', 0.0)))
    cnt = allocate(w, n_pool_pairs - zdb_pairs) if w else {}
    base = next((m['name'] for m in pool['members'] if m.get('base')), None)
    need = math.ceil(cfg['pool'].get('base_min_share', 0.0) * (n_pool_pairs - zdb_pairs) - 1e-9)
    while base in cnt and cnt[base] < need:   # version 0's minimum share holds in whole pairs too
        donor = max((n for n in cnt if n != base), key=lambda n: (cnt[n], n), default=None)
        if donor is None or cnt[donor] == 0:
            break
        cnt[donor] -= 1; cnt[base] += 1
    opps = [o for o, c in sorted(cnt.items()) for _ in range(c)] + ['zdb'] * zdb_pairs
    pkg = {m['name']: m['package'] for m in pool['members']}
    pkg['zdb'] = 'zdb'
    units = ['mirror'] * n_mirror + opps
    rng.shuffle(units)
    maps = cfg['training_maps']
    base_seed = cfg['seed_base'] + cfg.get('seed_iter_stride', 1000) * k
    jobs = []
    for u, kind in enumerate(units):
        m, seed = rng.choice(maps), base_seed + u
        if kind == 'mirror':
            jobs.append(dict(id=f'j{len(jobs):04d}', unit=u, kind='mirror', map=m, seed=seed, a=learner, b=learner,
                             opponent='self', record='AB', learner_sides='AB', pair=-1))
            continue
        for side in 'AB':
            a, b = (learner, pkg[kind]) if side == 'A' else (pkg[kind], learner)
            jobs.append(dict(id=f'j{len(jobs):04d}', unit=u, kind='pool', map=m, seed=seed, a=a, b=b, opponent=kind,
                             record=side, learner_sides=side, pair=u))
    doc = dict(run=rd.name, iter=k, learner=dict(version=k, package=learner), created=time.time(),
               games=len(jobs), mirror_games=n_mirror, pool_pairs=n_pool_pairs, weights=w, counts=cnt,
               winrate=dict(pool['winrate']), trace=cfg.get('trace', 10), java_flags=cfg.get('java_flags', '-Xmx2g'),
               game_timeout_seconds=cfg.get('game_timeout_seconds', 1200), jobs=jobs)
    idir = iter_dir(rd, k)
    idir.mkdir(parents=True, exist_ok=True)
    write_json(idir / 'jobs.json', doc)
    pool['updated_iter'] = k
    pool['weights'] = w
    write_json(rd / 'pool.json', pool)
    return doc


def promote(cfg, rd, v):
    pool = load_pool(cfg, rd)
    m = next((m for m in pool['members'] if m['version'] == v), None)
    if m is None:
        m = dict(name=f'v{v}', version=v, package=package(cfg, v), added_iter=v + 1, permanent=False, base=False)
        pool['members'].append(m)
    m['promoted'] = True
    pool['best'] = v
    pool['promotions'].append([v, time.time()])
    write_json(rd / 'pool.json', pool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', default='', help='run name (default: the config name); state in runs/rl/RUN')
    ap.add_argument('--iter', type=int, help='schedule this iteration')
    ap.add_argument('--runs', default='', help='runs root (default runs/rl; tests)')
    ap.add_argument('--games', type=int, help='games this iteration (default: config games_per_iter)')
    ap.add_argument('--promote', type=int, help='mark this version a promoted best (kept in the pool)')
    ap.add_argument('--show', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    rd = run_dir(cfg, a.run, a.runs)
    rd.mkdir(parents=True, exist_ok=True)
    if a.promote is not None:
        promote(cfg, rd, a.promote)
        print(f'promoted v{a.promote}')
    if a.iter is not None:
        doc = schedule(cfg, rd, a.iter, a.games or cfg['games_per_iter'])
        print(f"iter {a.iter}: {doc['games']} games ({doc['mirror_games']} mirror, {doc['pool_pairs']} pool pairs) "
              f"counts {doc['counts']} -> {iter_dir(rd, a.iter) / 'jobs.json'}")
    if a.show or (a.iter is None and a.promote is None):
        pool = load_pool(cfg, rd)
        for m in pool['members']:
            print(f"{m['name']:6s} {m['package']:10s} added {m['added_iter']:4d} perm {m['permanent']!s:5s} "
                  f"promoted {m['promoted']!s:5s} winrate {pool['winrate'].get(m['name'])}")
        print(f"best v{pool['best']}, weights {pool.get('weights')}")


if __name__ == '__main__':
    main()
