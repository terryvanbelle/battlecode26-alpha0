"""Schedule one ES iteration (docs/rl-es-plan.md section 1): draw the antithetic directions, write and build the
candidate packages, write jobs.json. Run state lives in runs/es/RUN (es_loop.sh): base/net.json (the fixed base),
phi/<k>.json (the centre's offsets at the start of iteration k), iter-NNNN/{dirs.json, jobs.json, games/, cand/}.

es_schedule.py --config C --run RUN --iter K [--no-build]      draw + build + jobs.json for iteration K
es_schedule.py --config C --run RUN --prune K                  remove the build/botsrc and build/bots packages of K
es_schedule.py --config C --run RUN --centre K                 build the centre package <prefix><K>c (gates)

Directions: eps_i ~ N(0, I) over block (i + k) mod n_blocks, rng seed 2_000_000 + 1000 k + i; candidates
theta+-_i = phi_k +- sigma * eps_i, packages <prefix><k>_<i>p / <prefix><k>_<i>m (build_dpnet.py, no --rl, floor and
floor_points from the config). Each direction gets units_per_dir units; a unit is (map, seed) played seat-swapped:
(A = theta+, B = theta-) then (A = theta-, B = theta+). Maps: a per-iteration permutation of training_maps (never
chosen by score), unit u takes perm[u % len]; seed = seed_base + 1000 k + u. A pool unit (pool_fraction of the units,
M2) plays theta+ and theta- each against one pool package, seat-swapped (4 games). Jobs record nothing and trace 0.
"""
import argparse, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import es_net  # noqa: E402

DIR_SEED = 2_000_000


def run_dir(cfg, run, runs=''):
    return (Path(runs) if runs else ROOT / 'runs' / 'es') / (run or cfg['name'])


def iter_dir(rd, k):
    return rd / f'iter-{k:04d}'


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def cfg_blocks(cfg, base_raw):
    return es_net.blocks(base_raw, cfg['points'], es_net.parse_exclude(cfg.get('exclude', [])),
                         es_net.load_equiv(cfg.get('equiv', '')), logT=cfg.get('logT', True))


def sigma_of(cfg, rd, coords):
    """{coord: sigma}: calibrated values (rd/sigma.json, es_update --calibrate) over the config's sigma_bias /
    sigma_logT, with per-coordinate overrides cfg['sigma'] {coord or point: value}."""
    cal = json.loads((rd / 'sigma.json').read_text()).get('sigma', {}) if (rd / 'sigma.json').exists() else {}
    ov = cfg.get('sigma', {})
    out = {}
    for c in coords:
        p = str(es_net.coord_point(c))
        if c in ov:
            out[c] = float(ov[c])
        elif c.endswith(':logT'):
            out[c] = float(ov.get(p + ':logT', cfg.get('sigma_logT', 0.3)))
        elif p in ov:
            out[c] = float(ov[p])
        else:
            out[c] = float(cal.get(c, cfg.get('sigma_bias', 10.0)))
    return out


def draw(k, n_dir, blocks, sigma):
    """[{'i', 'block', 'eps': {coord: e}, 'delta': {coord: sigma * e}}]"""
    import numpy as np
    out = []
    for i in range(n_dir):
        b = blocks[(i + k) % len(blocks)]
        e = np.random.default_rng(DIR_SEED + 1000 * k + i).standard_normal(len(b.coords))
        eps = {c: float(v) for c, v in zip(b.coords, e)}
        out.append(dict(i=i, block=b.name, eps=eps, delta={c: sigma[c] * v for c, v in eps.items()}))
    return out


def cand_name(cfg, k, i, sign):
    return f"{cfg['package_prefix']}{k}_{i}{'p' if sign > 0 else 'm'}"


def build(cfg, net_path, name, compile_=True):
    fl = ','.join(f'{p}:{f}' for p, f in sorted((cfg.get('floor_points') or {}).items()))
    cmd = [sys.executable, str(ROOT / 'scripts/build_dpnet.py'), '--net', str(net_path), '--name', name,
           '--floor', str(cfg.get('floor', 0.005)), '--temperature', str(cfg.get('temperature', 1.0))]
    if fl:
        cmd += ['--floor-points', fl]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    if compile_:
        from eval_student import compile_local
        compile_local(name)


def phi_k(cfg, rd, base_raw, k):
    p = rd / 'phi' / f'{k}.json'
    if p.exists():
        return es_net.load_phi(p)
    if k == 0:
        phi = es_net.zero_phi(base_raw, cfg['points'])
        es_net.save_phi(p, phi, iter=0)
        return phi
    raise SystemExit(f'{p} missing (es_update of iteration {k - 1} not done)')


def write_candidates(cfg, rd, k, dirs, base_raw, phi, do_build=True, workers=8):
    """I/cand/<i><p|m>/net.json, then build_dpnet + compile each (parallel); the candidate nets are deleted after a
    successful build unless keep_cand (dirs.json can regenerate them)."""
    I = iter_dir(rd, k)
    equiv = es_net.load_equiv(cfg.get('equiv', ''))
    todo = []
    for d in dirs:
        for sign in (1, -1):
            name = cand_name(cfg, k, d['i'], sign)
            net = I / 'cand' / f"{d['i']}{'p' if sign > 0 else 'm'}" / 'net.json'
            if (ROOT / 'build/bots' / name / name).exists() and (ROOT / 'build/botsrc' / name / 'W.java').exists():
                continue
            net.parent.mkdir(parents=True, exist_ok=True)
            th = es_net.phi_add(phi, {c: sign * v for c, v in d['delta'].items()}, equiv)
            net.write_text(json.dumps(es_net.apply_phi(base_raw, th)))
            todo.append((net, name))
    if do_build and todo:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda t: build(cfg, *t), todo))
        if not cfg.get('keep_cand'):
            shutil.rmtree(I / 'cand', ignore_errors=True)
    return [n for _, n in todo]


def jobs(cfg, k, dirs):
    import numpy as np
    maps = list(cfg['training_maps'])
    perm = [maps[j] for j in np.random.default_rng(DIR_SEED + 1000 * k + 999).permutation(len(maps))]
    upd = cfg.get('units_per_dir', 2)
    pool = cfg.get('pool_packages') or []
    pf = cfg.get('pool_fraction', 0.0) if pool else 0.0
    out = []
    u = 0
    for d in dirs:
        P, M = cand_name(cfg, k, d['i'], 1), cand_name(cfg, k, d['i'], -1)
        for r in range(upd):
            m, seed = perm[u % len(perm)], cfg['seed_base'] + 1000 * k + u
            if pf and int((u + 1) * pf) > int(u * pf):   # pool units spread evenly
                opp = pool[u % len(pool)]
                for th, sgn in ((P, 1), (M, -1)):
                    for side in 'AB':
                        a, b = (th, opp) if side == 'A' else (opp, th)
                        out.append(dict(unit=u, dir=d['i'], kind='pool', map=m, seed=seed, a=a, b=b, opponent=opp,
                                        theta=sgn, theta_side=side, record=''))
            else:
                for a, b, sa in ((P, M, 1), (M, P, -1)):
                    out.append(dict(unit=u, dir=d['i'], kind='h2h', map=m, seed=seed, a=a, b=b, sign_a=sa, record=''))
            u += 1
    for n, j in enumerate(out):
        j['id'] = f'j{n:04d}'
    return out


def schedule(cfg, rd, k, do_build=True):
    base_raw = json.loads((rd / 'base' / 'net.json').read_text())
    bl = cfg_blocks(cfg, base_raw)
    phi = phi_k(cfg, rd, base_raw, k)
    sigma = sigma_of(cfg, rd, [c for b in bl for c in b.coords])
    dirs = draw(k, cfg['n_dir'], bl, sigma)
    I = iter_dir(rd, k)
    I.mkdir(parents=True, exist_ok=True)
    write_json(I / 'dirs.json', dict(iter=k, sigma=sigma, blocks=[b._asdict() for b in bl], dirs=dirs))
    t0 = time.time()
    built = write_candidates(cfg, rd, k, dirs, base_raw, phi, do_build, cfg.get('build_workers', 8))
    js = jobs(cfg, k, dirs)
    doc = dict(run=rd.name, iter=k, created=time.time(), games=len(js), trace=0, java_flags=cfg.get('java_flags', '-Xmx2g'),
               game_timeout_seconds=cfg.get('game_timeout_seconds', 1200), jobs=js)
    write_json(I / 'jobs.json', doc)
    print(f'iter {k}: {len(dirs)} directions over blocks {sorted({d["block"] for d in dirs})}, {len(js)} games, '
          f'built {len(built)} packages in {time.time() - t0:.0f}s -> {I / "jobs.json"}')
    return doc


def prune(cfg, k):
    pre = f"{cfg['package_prefix']}{k}_"
    n = 0
    for d in list((ROOT / 'build/botsrc').glob(pre + '*')) + list((ROOT / 'build/bots').glob(pre + '*')):
        shutil.rmtree(d, ignore_errors=True); n += 1
    print(f'pruned {n} package dirs {pre}*')


def centre(cfg, rd, k):
    """Build <prefix><k>c from phi/<k>.json (the centre after k updates)."""
    base_raw = json.loads((rd / 'base' / 'net.json').read_text())
    phi = phi_k(cfg, rd, base_raw, k)
    net = rd / 'centre' / str(k) / 'net.json'
    net.parent.mkdir(parents=True, exist_ok=True)
    net.write_text(json.dumps(es_net.apply_phi(base_raw, phi)))
    name = f"{cfg['package_prefix']}{k}c"
    build(cfg, net, name)
    print(f'built {name} from {net}')
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', default='')
    ap.add_argument('--runs', default='', help='runs root (default runs/es; tests)')
    ap.add_argument('--iter', type=int)
    ap.add_argument('--prune', type=int)
    ap.add_argument('--centre', type=int)
    ap.add_argument('--no-build', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    rd = run_dir(cfg, a.run, a.runs)
    if a.iter is not None:
        schedule(cfg, rd, a.iter, not a.no_build)
    if a.prune is not None:
        prune(cfg, a.prune)
    if a.centre is not None:
        centre(cfg, rd, a.centre)


if __name__ == '__main__':
    main()
