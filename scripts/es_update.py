"""ES update for one iteration (docs/rl-es-plan.md section 1): unit fitness from the head-to-head games, the
thresholded-t sign step, the new centre, one metrics row.

es_update.py --config C --run RUN --iter K      reads iter-K/{dirs.json, jobs.json, games/}, phi/<K>.json; writes
    phi/<K+1>.json, iter-K/phi.json (same), iter-K/update.json (per-unit scores, per-coordinate stats),
    centre/<K+1>/net.json and a row in metrics.jsonl
es_update.py --config C --run RUN --calibrate DATA_DIR...   sigma.json from own records (p<point>.npz, rl_data.py)
es_update.py --self-test

Fitness of unit u: h2h unit (two games, theta+ as A then as B): s_u = mean score of theta+ - 0.5 (win 1, draw 0.5);
pool unit (theta+ and theta- each seat-swapped vs one pool package): s_u = (score theta+ - score theta-) / 2. A unit
with an invalid or missing game is dropped (counted); a h2h unit whose two games have the same traj_hash (the
perturbation changed nothing) is an exact zero: kept, counted.
Step per coordinate j over the directions whose block holds j: x_d = eps_dj mean_{u in d} s_u (clustered by direction:
units of a direction share eps), g = mean x, se = sd(x)/sqrt(n),
t = g/se, dphi_j = step_frac sigma_j sign(t) min(t_cap, max(0, |t| - t_dead)) (plan: 0.25, 2, 1). Then the leash
phi <- (1 - leash) phi is applied before the step, and the caps |b| <= bias_cap, logT in [-logT_cap, logT_cap].
"""
import argparse, json, math, os, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import es_net  # noqa: E402
from es_schedule import run_dir, iter_dir, write_json  # noqa: E402

NONE = 0xFFFF


def _valid(res):
    return res is not None and not (res.get('error') or not res.get('finished') or res['A'].get('failures') or res['B'].get('failures'))


def _pts(res, side):
    w = res.get('winner')
    return 0.5 if w is None else float(w == side)


def unit_scores(I):
    """[{unit, dir, kind, map, s, exact_zero, rounds}], dropped unit count."""
    I = Path(I)
    doc = json.loads((I / 'jobs.json').read_text())
    by = {}
    for j in doc['jobs']:
        by.setdefault(j['unit'], []).append(j)
    out, dropped = [], 0
    for u, js in sorted(by.items()):
        res = []
        for j in js:
            r = I / 'games' / j['id'] / 'result.json'
            res.append(json.loads(r.read_text()) if r.exists() else None)
        if not all(_valid(r) for r in res):
            dropped += 1
            continue
        j0 = js[0]
        rounds = sum(r.get('rounds') or 0 for r in res) / len(res)
        if j0['kind'] == 'h2h':
            plus = [_pts(r, 'A' if j['sign_a'] > 0 else 'B') for j, r in zip(js, res)]
            s = sum(plus) / len(plus) - 0.5
            ez = len(res) == 2 and res[0].get('traj_hash') is not None and res[0]['traj_hash'] == res[1]['traj_hash']
        else:
            sp = [_pts(r, j['theta_side']) for j, r in zip(js, res) if j['theta'] > 0]
            sm = [_pts(r, j['theta_side']) for j, r in zip(js, res) if j['theta'] < 0]
            s = 0.5 * (sum(sp) / len(sp) - sum(sm) / len(sm))
            ez = False
        out.append(dict(unit=u, dir=j0['dir'], kind=j0['kind'], map=j0['map'], s=s, exact_zero=bool(ez), rounds=rounds))
    return out, dropped


def coord_stats(units, dirs, coords):
    """{coord: (g, se, t, n)} over the directions that perturb the coordinate. The units of one direction share eps,
    so under a real effect their eps_j s_u are correlated even for a null coordinate j: the sample is clustered by
    direction (x_d = eps_dj * mean s over d's units; n = directions), else null |t| is inflated (~1.4x at 2 units)."""
    eps = {d['i']: d['eps'] for d in dirs}
    by = {}
    for u in units:
        by.setdefault(u['dir'], []).append(u['s'])
    out = {}
    for c in coords:
        xs = [eps[i][c] * sum(ss) / len(ss) for i, ss in sorted(by.items()) if c in eps[i]]
        n = len(xs)
        if n < 2:
            out[c] = (0.0, 0.0, 0.0, n)
            continue
        g = sum(xs) / n
        sd = math.sqrt(sum((x - g) ** 2 for x in xs) / (n - 1))
        se = sd / math.sqrt(n)
        out[c] = (g, se, g / se if se > 0 else 0.0, n)
    return out


def step(phi, dirs, units, sigma, cfg, equiv=None):
    """(phi', {coord: dict(g, se, t, n, step, phi)})"""
    coords = sorted({c for d in dirs for c in d['eps']})
    st = coord_stats(units, dirs, coords)
    sf, tc, td = cfg.get('step_frac', 0.25), cfg.get('t_cap', 2.0), cfg.get('t_dead', 1.0)
    leash = cfg.get('leash', 0.0)
    new = {p: {'b': [v * (1 - leash) for v in d['b']], 'logT': d['logT'] * (1 - leash)} for p, d in phi.items()}
    delta = {}
    for c, (g, se, t, n) in st.items():
        m = min(tc, max(0.0, abs(t) - td))
        if m > 0:
            delta[c] = sf * sigma[c] * math.copysign(m, t)
    new = es_net.phi_add(new, delta, equiv)
    bc, lc = cfg.get('bias_cap', 30.0), cfg.get('logT_cap', 1.0)
    for d in new.values():
        d['b'] = [max(-bc, min(bc, v)) for v in d['b']]
        d['logT'] = max(-lc, min(lc, d['logT']))
    stats = {c: dict(g=g, se=se, t=t, n=n, step=delta.get(c, 0.0), phi=es_net.phi_get(new, c)) for c, (g, se, t, n) in st.items()}
    return new, stats


def split_half_cos(units, dirs, coords):
    a = coord_stats([u for u in units if u['dir'] % 2 == 0], dirs, coords)   # halves by direction (independent eps)
    b = coord_stats([u for u in units if u['dir'] % 2 == 1], dirs, coords)
    va, vb = [a[c][0] for c in coords], [b[c][0] for c in coords]
    na, nb = math.sqrt(sum(v * v for v in va)), math.sqrt(sum(v * v for v in vb))
    return sum(x * y for x, y in zip(va, vb)) / (na * nb) if na and nb else None


def record_metrics(cfg, base_raw, phi, points):
    """Per record set (cfg fixed_records: [{name, dirs, max_round?}]): per point the centre's mean P(option)
    (floor-mixed), KL(centre || base), and P(watch option) by round bin."""
    import numpy as np
    import dpnet_io
    sets = cfg.get('fixed_records') or []
    if not sets:
        return {}
    floor = cfg.get('floor', 0.005)
    fpts = {int(k): float(v) for k, v in (cfg.get('floor_points') or {}).items()}
    bins = cfg.get('round_bins', [0, 200, 500, 1000, 2001])
    watch = cfg.get('watch', [])
    centre = es_net.apply_phi(base_raw, phi)
    out = {}
    for rs in sets:
        dirs = [ROOT / d for d in rs['dirs']]
        row = {}
        for p in points:
            parts = [np.load(d / f'p{p}.npz') for d in dirs if (d / f'p{p}.npz').exists()]
            if not parts:
                continue
            lc = dpnet_io.layers_from_raw(centre['points'][str(p)])
            lb = dpnet_io.layers_from_raw(base_raw['points'][str(p)])
            n_in, n_out = lc[0]['w'].shape[1], lc[-1]['w'].shape[0]
            x = np.concatenate([np.pad(z['x'], ((0, 0), (0, max(0, n_in - z['x'].shape[1]))))[:, :n_in] for z in parts])
            k = np.concatenate([z['k'] for z in parts]).astype(np.int64)
            rnd = np.concatenate([z['round'] for z in parts]) if all('round' in z for z in parts) else np.zeros(len(k), int)
            mask = np.concatenate([dpnet_io.fit_mask(z['mask'], len(z['k']), n_out) for z in parts])
            keep = (k != NONE) & (mask.sum(1) > 1)
            if rs.get('max_round'):
                keep &= rnd < rs['max_round']
            idx = np.nonzero(keep)[0]
            mr = rs.get('max_rows', 200000)
            if len(idx) > mr:
                idx = np.sort(np.random.default_rng(0).choice(idx, mr, replace=False))
            x, mask, rnd = x[idx], mask[idx], rnd[idx]
            fl = fpts.get(p, floor)
            pc = dpnet_io.probs(dpnet_io.forward(lc, x, exact=False), mask, 1.0, fl).astype(np.float64)
            pb = dpnet_io.probs(dpnet_io.forward(lb, x, exact=False), mask, 1.0, fl).astype(np.float64)
            kl = np.where(mask, pc * (np.log(np.maximum(pc, 1e-30)) - np.log(np.maximum(pb, 1e-30))), 0).sum(1)
            r = dict(records=int(len(idx)), mean_p=[round(float(v), 5) for v in pc.mean(0)], kl_base=round(float(kl.mean()), 5))
            for c in watch:
                if es_net.coord_point(c) != p:
                    continue
                o = int(c.split(':b')[1])
                r[f'P_{c}'] = round(float(pc[:, o].mean()), 5)
                r[f'P_{c}_by_round'] = {f'{lo}-{hi}': (round(float(pc[(rnd >= lo) & (rnd < hi), o].mean()), 5) if ((rnd >= lo) & (rnd < hi)).any() else None)
                                        for lo, hi in zip(bins, bins[1:])}
            row[str(p)] = r
        out[rs['name']] = row
    return out


def update(cfg, rd, k):
    I = iter_dir(rd, k)
    base_raw = json.loads((rd / 'base' / 'net.json').read_text())
    dd = json.loads((I / 'dirs.json').read_text())
    dirs, sigma = dd['dirs'], dd['sigma']
    phi = es_net.load_phi(rd / 'phi' / f'{k}.json')
    equiv = es_net.load_equiv(cfg.get('equiv', ''))
    units, dropped = unit_scores(I)
    if len(units) < cfg.get('min_units', 2):
        raise SystemExit(f'iter {k}: only {len(units)} valid units ({dropped} dropped)')
    new, stats = step(phi, dirs, units, sigma, cfg, equiv)
    coords = sorted(stats)
    es_net.save_phi(rd / 'phi' / f'{k + 1}.json', new, iter=k + 1, from_iter=k)
    es_net.save_phi(I / 'phi.json', new, iter=k + 1, from_iter=k)
    cn = rd / 'centre' / str(k + 1) / 'net.json'
    cn.parent.mkdir(parents=True, exist_ok=True)
    cn.write_text(json.dumps(es_net.apply_phi(base_raw, new)))
    # map-length classes: unit mean rounds <= the iteration median = short
    rs = sorted(u['rounds'] for u in units)
    med = rs[len(rs) // 2]
    cls = {'short': [u for u in units if u['rounds'] <= med], 'long': [u for u in units if u['rounds'] > med]}
    by_cls = {n: {c: dict(g=round(g, 5), t=round(t, 3), n=m) for c, (g, se, t, m) in coord_stats(us, dirs, coords).items()}
              for n, us in cls.items()}
    planted = set(cfg.get('planted', []))
    null = [abs(es_net.phi_get(new, c)) for c in coords if c not in planted and not c.endswith(':logT')]
    ts = [abs(stats[c]['t']) for c in coords]
    row = dict(iter=k, time=time.time(), units=len(units), dropped=dropped, games=2 * len(units),
               exact_zero=sum(u['exact_zero'] for u in units), exact_zero_share=round(sum(u['exact_zero'] for u in units) / len(units), 4),
               mean_abs_s=round(sum(abs(u['s']) for u in units) / len(units), 4),
               t_gt2_share=round(sum(t > 2 for t in ts) / len(ts), 4) if ts else None,
               split_half_cos=split_half_cos(units, dirs, coords), round_median=med,
               coords={c: {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in s.items()} for c, s in stats.items()},
               by_class=by_cls,
               null_drift=dict(linf=round(max(null), 4) if null else 0.0, median=round(sorted(null)[len(null) // 2], 4) if null else 0.0),
               logT={str(p): round(d['logT'], 4) for p, d in new.items()},
               records=record_metrics(cfg, base_raw, new, sorted(new)))
    write_json(I / 'update.json', dict(units=units, **{kk: row[kk] for kk in ('coords', 'by_class')}))
    with open(rd / 'metrics.jsonl', 'a') as f:
        f.write(json.dumps(row) + '\n')
    top = sorted(coords, key=lambda c: -abs(stats[c]['t']))[:5]
    print(f"iter {k}: {len(units)} units ({dropped} dropped, {row['exact_zero']} exact-zero), |t|>2 share {row['t_gt2_share']}, "
          f"split-half cos {row['split_half_cos'] if row['split_half_cos'] is None else round(row['split_half_cos'], 3)}; top "
          + ', '.join(f"{c} t={stats[c]['t']:+.2f} step={stats[c]['step']:+.2f} phi={stats[c]['phi']:+.2f}" for c in top))
    for n, r in row['records'].items():
        for p, v in r.items():
            print(f"  records {n} point {p}: kl_base {v['kl_base']} " + ' '.join(f"{kk}={vv}" for kk, vv in v.items() if kk.startswith('P_')))
    return row


# ---- self-test
def self_test():
    import numpy as np
    import es_schedule
    rng = np.random.default_rng(5)
    base = {'points': {'11': [dict(n_in=4, n_out=4, w=[0.0] * 16, b=[0.0] * 4), dict(n_in=4, n_out=22, w=[0.0] * 88, b=[0.0] * 22)]}}
    cfg = dict(points=[11], step_frac=0.25, t_cap=2.0, t_dead=1.0, bias_cap=30.0, logT_cap=1.0)
    bl = es_net.blocks(base, [11])
    coords = bl[0].coords
    sigma = {c: (0.3 if c.endswith('logT') else 10.0) for c in coords}
    planted = '11:b21'
    # 1. planted linear effect: s = clip(beta * eps_planted + noise); t has the right sign, nulls move only if |t| > 1
    for beta, sign in ((0.25, -1), (0.25, 1)):
        moved_null = big_null = 0
        for k in range(6):
            dirs = es_schedule.draw(k, 32, bl, sigma)
            units = []
            for d in dirs:
                for r in range(2):
                    s = float(np.clip(sign * beta * d['eps'][planted] + rng.normal(0, 0.25), -0.5, 0.5))
                    units.append(dict(unit=2 * d['i'] + r, dir=d['i'], s=s))
            new, st = step(es_net.zero_phi(base, [11]), dirs, units, sigma, cfg)
            assert np.sign(st[planted]['t']) == sign and abs(st[planted]['t']) > 2, st[planted]
            assert abs(new[11]['b'][21] - sign * min(2, abs(st[planted]['t']) - 1) * 2.5) < 1e-9
            for c in coords:
                if c == planted:
                    continue
                if abs(st[c]['t']) <= 1:
                    assert st[c]['step'] == 0 and es_net.phi_get(new, c) == 0, (c, st[c])
                else:
                    moved_null += 1
                big_null += abs(st[c]['t']) > 2
        assert moved_null < 0.45 * 6 * 22 and big_null < 0.15 * 6 * 22, (moved_null, big_null)
    # 2. pure noise: |t| > 2 share near chance; leash and caps
    dirs = es_schedule.draw(0, 32, bl, sigma)
    units = [dict(unit=2 * d['i'] + r, dir=d['i'], s=float(rng.choice([-0.5, 0, 0, 0.5]))) for d in dirs for r in range(2)]
    phi = es_net.zero_phi(base, [11]); phi[11]['b'][3] = 29.9; phi[11]['logT'] = 0.99
    new, st = step(phi, dirs, units, sigma, dict(cfg, leash=0.5))
    assert all(abs(v) <= 30 for v in new[11]['b']) and -1 <= new[11]['logT'] <= 1
    assert abs(new[11]['b'][3] - (14.95 + st['11:b3']['step'])) < 1e-9
    # 3. unit_scores: pairing, draws, exact zeros, dropped units, pool units
    with tempfile.TemporaryDirectory() as td:
        I = Path(td)
        jobs = [dict(id='j0000', unit=0, dir=0, kind='h2h', map='m', seed=1, a='P', b='M', sign_a=1),
                dict(id='j0001', unit=0, dir=0, kind='h2h', map='m', seed=1, a='M', b='P', sign_a=-1),
                dict(id='j0002', unit=1, dir=0, kind='h2h', map='m', seed=2, a='P', b='M', sign_a=1),
                dict(id='j0003', unit=1, dir=0, kind='h2h', map='m', seed=2, a='M', b='P', sign_a=-1),
                dict(id='j0004', unit=2, dir=1, kind='h2h', map='m', seed=3, a='P', b='M', sign_a=1),
                dict(id='j0005', unit=2, dir=1, kind='h2h', map='m', seed=3, a='M', b='P', sign_a=-1),
                dict(id='j0006', unit=3, dir=1, kind='h2h', map='m', seed=4, a='P', b='M', sign_a=1),
                dict(id='j0007', unit=3, dir=1, kind='h2h', map='m', seed=4, a='M', b='P', sign_a=-1)]
        for th, sg in (('P', 1), ('M', -1)):
            for side in 'AB':
                a, b = (th, 'Z') if side == 'A' else ('Z', th)
                jobs.append(dict(id=f'j{len(jobs):04d}', unit=4, dir=1, kind='pool', map='m', seed=5, a=a, b=b, opponent='Z',
                                 theta=sg, theta_side=side))
        (I / 'jobs.json').write_text(json.dumps(dict(jobs=jobs)))
        ok = dict(finished=True, error=None, A=dict(failures=[]), B=dict(failures=[]), rounds=100)
        res = {'j0000': dict(ok, winner='A', traj_hash='1'), 'j0001': dict(ok, winner='B', traj_hash='2'),   # P wins both
               'j0002': dict(ok, winner='A', traj_hash='3'), 'j0003': dict(ok, winner='A', traj_hash='3'),   # identical: 0
               'j0004': dict(ok, winner=None, traj_hash='4'), 'j0005': dict(ok, winner='A', traj_hash='5'),  # P 0.5 + 0
               'j0006': dict(ok, winner='A', traj_hash='6'),                                                 # j0007 missing
               'j0008': dict(ok, winner='A'), 'j0009': dict(ok, winner='A'),   # P vs Z: 1, 0
               'j0010': dict(ok, winner='B'), 'j0011': dict(ok, winner='B')}   # M vs Z: 0, 1 -> s = 0
        for j, r in res.items():
            (I / 'games' / j).mkdir(parents=True)
            (I / 'games' / j / 'result.json').write_text(json.dumps(r))
        us, dropped = unit_scores(I)
        s = {u['unit']: u for u in us}
        assert dropped == 1 and set(s) == {0, 1, 2, 4}, (dropped, s)
        assert s[0]['s'] == 0.5 and not s[0]['exact_zero']
        assert s[1]['s'] == 0.0 and s[1]['exact_zero']
        assert s[2]['s'] == -0.25 and s[4]['s'] == 0.0, s
    print('es_update self-test OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config')
    ap.add_argument('--run', default='')
    ap.add_argument('--runs', default='')
    ap.add_argument('--iter', type=int)
    ap.add_argument('--calibrate', nargs='+')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    cfg = json.loads(Path(a.config).read_text())
    rd = run_dir(cfg, a.run, a.runs)
    if a.calibrate:
        base_raw = json.loads((rd / 'base' / 'net.json').read_text())
        last = max([int(p.stem) for p in (rd / 'phi').glob('*.json')], default=0)
        phi = es_net.load_phi(rd / 'phi' / f'{last}.json') if last or (rd / 'phi' / '0.json').exists() else es_net.zero_phi(base_raw, cfg['points'])
        from es_schedule import cfg_blocks
        coords = [c for b in cfg_blocks(cfg, base_raw) for c in b.coords]
        sg = es_net.calibrate_sigma(base_raw, phi, a.calibrate, cfg.get('calib_target', 0.25), cfg.get('calib_cap', 15.0),
                                    floor=cfg.get('floor', 0.005), coords_=coords)
        write_json(rd / 'sigma.json', dict(phi_iter=last, data=a.calibrate, sigma=sg, time=time.time()))
        print(json.dumps(sg))
        return
    update(cfg, rd, a.iter)


if __name__ == '__main__':
    main()
