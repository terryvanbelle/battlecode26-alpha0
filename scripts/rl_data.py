"""Turn an RL iteration's games into training data (docs/rl-plan.md section 4): parse the trailers, run the parity check
against the net that played, join z and the trace, write per-point npz, delete the raw decision files.

rl_data.py --config configs/rl-v1.json --iter K [--run RUN] [--points 5,7,10,11,12,13] [--frozen-sample 0.05]
    [--floor F] [--temperature T] [--keep-raw] [--max-rows N] [--iter-dir DIR --nets DIR]

Reads DIR/jobs.json (default runs/rl/RUN/iter-NNNN) and each valid game's result.json, trace.tsv and the
decisions-<side>.bin.gz of the learner's recorded sides. Writes DIR/data/:
  p<point>.npz   one row per trailed record (trainable points: all; other points: a frozen_sample fraction, for
                 diagnostics): x uint16 [N, nf], k int32 (0xFFFF none), logq float32, mask bool [N, nOpt] (nOpt = the
                 net's n_out), flags uint8, version int32, team int8 (0 A, 1 B), round int32, game int32 (job index),
                 pair int32 (-1: unpaired), z float32 (+1 win, -1 loss, 0 draw, for the record's team),
                 trace float32 [N, 20] (rl_train.TRACE_COLS, at the latest trace row with round <= the record's round;
                 the first row for records before it), robot int32
  trace.npz      one row per learner team-game and trace row: game, team, round, pair, z, trace (fits V)
  summary.json   games, learner score by opponent, per point: counts, fallback per option, saturation, parity
  games.jsonl    one row per game (winner, reason, rounds, transferred, kings, babies at 500/1000/1500, ...)
Parity: log p_k recomputed with dpnet_io (nets/<NET_VERSION>/net.json, the recorded mask, floor, T) against the
trailer's -q/4096 on every kept record with a sample (q = 65535, the cap, is skipped). Pass: median |dlogp| < 1e-3 and
the 99.9th percentile < 2e-3 overall, and every point's median < 1e-3. On failure nothing is deleted and the exit
code is 2. --max-rows caps a point's rows by uniform bottom-k sampling (per-record random keys).
"""
import argparse, json, math, os, struct, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import decisions, dpnet_io  # noqa: E402

NONE = 0xFFFF
QMAX = 65535
NCOL = 6   # babies, kings, cheese, xfer, xfer_d50, traps
PARITY_FAIL = 2


def map_area(name, cache={}):
    """Width x height from the .map26 flatbuffer (root table GameMap: field 0 name, field 1 size struct (x, y))."""
    if name not in cache:
        b = (ROOT / 'maps' / f'{name}.map26').read_bytes()
        root = struct.unpack_from('<I', b, 0)[0]
        vt = root - struct.unpack_from('<i', b, root)[0]
        off = struct.unpack_from('<H', b, vt + 6)[0]
        w, h = struct.unpack_from('<ii', b, root + off)
        cache[name] = w * h
    return cache[name]


def read_trace(path):
    """trace.tsv -> int64 [R, 11] (round, A_babies, A_kings, A_cheese, A_xfer, A_rattraps, B_...)."""
    rows = [list(map(int, l.split('\t'))) for l in Path(path).read_text().splitlines()[1:] if l.strip()]
    return np.array(rows, dtype=np.int64).reshape(-1, 11)


def trace_features(tr, team, area):
    """float32 [R, 20] per trace row for team (0 A, 1 B): us_*, them_*, diff_* of (babies, kings, cheese, xfer,
    xfer_d50, traps), round / 2000, map_area / 3600. xfer_d50 = xfer now - xfer at the latest row <= round - 50 (0 before)."""
    R = len(tr)
    rounds = tr[:, 0]
    back = np.searchsorted(rounds, rounds - 50, side='right') - 1

    def side(s):
        c = tr[:, 1 + 5 * s: 6 + 5 * s].astype(np.float64)   # babies, kings, cheese, xfer, traps
        x0 = np.where(back >= 0, c[np.maximum(back, 0), 3], 0.0)
        return np.stack([c[:, 0], c[:, 1], c[:, 2], c[:, 3], c[:, 3] - x0, c[:, 4]], 1)

    us, them = side(team), side(1 - team)
    f = np.concatenate([us, them, us - them, (rounds / 2000.0)[:, None], np.full((R, 1), area / 3600.0)], 1)
    return f.astype(np.float32)


def z_of(res, side):
    w = res.get('winner')
    return 0.0 if w is None else (1.0 if w == side else -1.0)


def game_row(j, res, tr):
    row = dict(game=j['id'], map=j['map'], seed=j['seed'], kind=j['kind'], opponent=j['opponent'],
               learner=j['learner_sides'], winner=res.get('winner'), reason=res.get('reason'), rounds=res.get('rounds'),
               seconds=res.get('seconds'))
    for s in 'AB':
        row[s] = {k: res[s].get(k) for k in ('transferred', 'kings', 'babies', 'cheese', 'cat_damage', 'decisions_recorded')}
    if tr is not None and len(tr):
        for s, o in (('A', 0), ('B', 5)):
            row[s]['babies_at'] = {str(r): int(tr[min(np.searchsorted(tr[:, 0], r), len(tr) - 1), 1 + o]) for r in (500, 1000, 1500)}
            dead = np.nonzero(tr[:, 2 + o] < tr[0, 2 + o])[0]   # trace kings below the first row's: first king death
            row[s]['first_king_death'] = int(tr[dead[0], 0]) if len(dead) else None
    return row


class Acc:
    """Per-point row accumulator with an optional uniform cap (bottom-k on random keys)."""

    def __init__(self, cap, rng):
        self.parts, self.n, self.cap, self.rng, self.thr = [], 0, cap, rng, 1.0

    def add(self, d):
        key = self.rng.random(len(d['k']))
        sel = key < self.thr
        if not sel.all():
            d = {k: v[sel] for k, v in d.items()}; key = key[sel]
        d['_key'] = key
        self.parts.append(d); self.n += len(key)
        if self.cap and self.n > 2 * self.cap:
            self.shrink()

    def shrink(self):
        if not self.cap or self.n <= self.cap:
            return
        d = self.cat()
        self.thr = float(np.partition(d['_key'], self.cap - 1)[self.cap - 1])
        sel = d['_key'] <= self.thr
        self.parts = [{k: v[sel] for k, v in d.items()}]; self.n = int(sel.sum())

    def cat(self):
        nf = max(p['x'].shape[1] for p in self.parts)
        out = {}
        for k in self.parts[0]:
            if k == 'x':
                out[k] = np.concatenate([np.pad(p['x'], ((0, 0), (0, nf - p['x'].shape[1]))) for p in self.parts])
            else:
                out[k] = np.concatenate([p[k] for p in self.parts])
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', default='')
    ap.add_argument('--iter', type=int, required=True)
    ap.add_argument('--runs', default='', help='runs root (default runs/rl; tests)')
    ap.add_argument('--iter-dir', default='', help='default runs/rl/RUN/iter-NNNN')
    ap.add_argument('--nets', default='', help='nets dir holding <version>/net.json (default runs/rl/RUN/nets)')
    ap.add_argument('--points', default='', help='trainable points (default: config trainable_points)')
    ap.add_argument('--frozen-sample', type=float, help='fraction of other points kept (default: config, 0.05)')
    ap.add_argument('--floor', type=float, help="sampling floor (default: the learner build's BUILD.json, else config)")
    ap.add_argument('--temperature', type=float)
    ap.add_argument('--max-rows', type=int, help='per-point row cap (default: config max_rows_per_point)')
    ap.add_argument('--keep-raw', action='store_true', help='do not delete decisions-*.bin.gz')
    a = ap.parse_args()
    t0 = time.time()
    cfg = json.loads(Path(a.config).read_text())
    rd = (Path(a.runs) if a.runs else ROOT / 'runs' / 'rl') / (a.run or cfg['name'])
    idir = Path(a.iter_dir) if a.iter_dir else rd / f'iter-{a.iter:04d}'
    nets_dir = Path(a.nets) if a.nets else rd / 'nets'
    doc = json.loads((idir / 'jobs.json').read_text())
    points = {int(p) for p in (a.points.split(',') if a.points else cfg['trainable_points']) if str(p)}
    fs = cfg.get('frozen_sample', 0.05) if a.frozen_sample is None else a.frozen_sample
    cap = cfg.get('max_rows_per_point', 0) if a.max_rows is None else a.max_rows
    build = ROOT / 'build/botsrc' / doc['learner']['package'] / 'BUILD.json'
    bj = json.loads(build.read_text()) if build.exists() else {}
    floor = a.floor if a.floor is not None else bj.get('floor', cfg['floor'])
    # per-point floors (build_dpnet.py --floor-points; W.FL): the learner build's, else the config's
    fpts = {int(k): float(v) for k, v in (bj.get('floor_points') or cfg.get('floor_points') or {}).items()}
    T = a.temperature if a.temperature is not None else bj.get('temperature', cfg.get('temperature', 1.0))
    pcfg = cfg.get('parity', {})
    rng = np.random.default_rng(cfg['seed_base'] + a.iter)

    nets = {}

    def net(v):
        if v not in nets:
            f = nets_dir / str(v) / 'net.json'
            nets[v] = dpnet_io.load_net(f) if f.exists() else None
        return nets[v]

    lv = doc['learner']['version']
    if net(lv) is None:
        raise SystemExit(f'no net for the learner version {lv} in {nets_dir}')
    nopt = {p: ls[-1]['w'].shape[0] for p, ls in net(lv).items()}

    acc, tracerows, games, raws = {}, [], [], []
    score, nvalid, nbad, missing = {}, 0, 0, 0
    for gi, j in enumerate(doc['jobs']):
        g = idir / 'games' / j['id']
        rp = g / 'result.json'
        if not rp.exists():
            continue
        res = json.loads(rp.read_text())
        ok = not (res.get('error') or not res.get('finished') or res['A'].get('failures') or res['B'].get('failures'))
        tp = g / 'trace.tsv'
        tr = read_trace(tp) if tp.exists() else None
        if not ok or tr is None or not len(tr):
            nbad += 1
            continue
        files = {s: g / f'decisions-{s}.bin.gz' for s in j['learner_sides']}
        if not all(f.exists() for f in files.values()):
            missing += 1
            continue
        nvalid += 1
        games.append(game_row(j, res, tr))
        if j['kind'] == 'pool':
            s = score.setdefault(j['opponent'], [0.0, 0])
            s[0] += (z_of(res, j['learner_sides']) + 1) / 2; s[1] += 1
        area = map_area(j['map'])
        for side, f in files.items():
            raws.append(f)
            team = 'AB'.index(side)
            z = z_of(res, side)
            # the partner game of a seat-swapped pair must be valid too, else the pair baseline would cancel A
            pair = j['pair']
            if pair >= 0:
                mates = [m for m in doc['jobs'] if m['pair'] == pair and m['id'] != j['id']]
                if not all(_valid_game(idir / 'games' / m['id']) for m in mates):
                    pair = -1
            tf = trace_features(tr, team, area)
            tracerows.append(dict(game=np.full(len(tr), gi, np.int32), team=np.full(len(tr), team, np.int8),
                                  round=tr[:, 0].astype(np.int32), pair=np.full(len(tr), pair, np.int32),
                                  z=np.full(len(tr), z, np.float32), trace=tf))
            for p, d in decisions.read_rl(f, nopt).items():
                N = len(d['k'])
                if p not in points:
                    sel = rng.random(N) < fs
                    if not sel.any():
                        continue
                    d = {k: v[sel] for k, v in d.items()}; N = int(sel.sum())
                idx = np.clip(np.searchsorted(tr[:, 0], d['round'], side='right') - 1, 0, len(tr) - 1)
                row = dict(x=d['x'].astype(np.uint16), k=d['k'].astype(np.int32), logq=d['logq'].astype(np.float32),
                           mask=d['mask'], flags=d['flags'].astype(np.uint8), version=d['version'].astype(np.int32),
                           team=np.full(N, team, np.int8), round=d['round'].astype(np.int32),
                           game=np.full(N, gi, np.int32), pair=np.full(N, pair, np.int32),
                           z=np.full(N, z, np.float32), trace=tf[idx], robot=d['robot'].astype(np.int32))
                acc.setdefault(p, Acc(cap if p in points else 0, rng)).add(row)

    # parity, per point
    summary = dict(iter=a.iter, run=rd.name, learner=doc['learner'], floor=floor, floor_points=fpts, temperature=T,
                   games=len(doc['jobs']),
                   valid=nvalid, invalid=nbad, missing_decisions=missing, trainable=sorted(points), frozen_sample=fs,
                   score={o: dict(points=v[0], games=v[1], score=round(v[0] / v[1], 4) if v[1] else None)
                          for o, v in sorted(score.items())}, points={})
    data, alld, fail = {}, [], []
    for p in sorted(acc):
        acc[p].shrink(); d = acc[p].cat(); d.pop('_key')
        data[p] = d
        N = len(d['k'])
        picked = d['k'] != NONE
        dl = np.full(N, np.nan)
        for v in np.unique(d['version'][picked]).tolist():
            nv = net(v)
            sel = picked & (d['version'] == v)
            if nv is None or p not in nv:
                dl[sel] = np.inf
                continue
            ls = nv[p]
            n_out = ls[-1]['w'].shape[0]
            pr = dpnet_io.probs(dpnet_io.forward(ls, d['x'][sel]), dpnet_io.fit_mask(d['mask'][sel], int(sel.sum()), n_out), T,
                                fpts.get(p, floor))
            k = d['k'][sel]
            inr = k < n_out
            pk = np.zeros(len(k), np.float32)
            pk[inr] = pr[np.nonzero(inr)[0], k[inr]]
            with np.errstate(divide='ignore'):
                dl[sel] = np.abs(np.log(pk.astype(np.float64)) - d['logq'][sel])
        chk = picked & (np.round(-d['logq'] * 4096) < QMAX)
        dv = dl[chk]
        alld.append(dv)
        fl = d['flags'][picked].astype(np.int64) & 1
        ks = d['k'][picked].astype(np.int64)
        no = d['mask'].shape[1]
        hist = np.bincount(ks, minlength=no)
        st = dict(rows=N, trainable=p in points, picked=int(picked.sum()),
                  fallback=float(fl.mean()) if len(fl) else None,
                  fallback_by_option={str(o): round(float(r), 4) for o, (r, c) in
                                      enumerate(zip(np.bincount(ks, weights=fl, minlength=no) / np.maximum(hist, 1), hist)) if c},
                  option_hist={str(o): int(c) for o, c in enumerate(hist) if c},
                  saturated=saturation(d, picked, fpts.get(p, floor)), floor=fpts.get(p, floor),
                  single_option=int((picked & (d['mask'].sum(1) <= 1)).sum()),
                  versions=sorted(set(d['version'].tolist())),
                  parity_n=int(len(dv)), parity_median=float(np.median(dv)) if len(dv) else None,
                  parity_p999=float(np.quantile(dv, 0.999)) if len(dv) else None,
                  parity_max=float(dv.max()) if len(dv) else None)
        if len(dv) and not st['parity_median'] < pcfg.get('median_max', 1e-3):
            fail.append(p)
        summary['points'][str(p)] = st
    allv = np.concatenate(alld) if alld else np.zeros(0)
    summary['parity'] = dict(n=int(len(allv)), median=float(np.median(allv)) if len(allv) else None,
                             p999=float(np.quantile(allv, 0.999)) if len(allv) else None,
                             max=float(allv.max()) if len(allv) else None, failed_points=fail)
    passed = bool(len(allv)) and not fail and summary['parity']['median'] < pcfg.get('median_max', 1e-3) \
        and summary['parity']['p999'] < pcfg.get('p999_max', 2e-3)
    summary['parity']['passed'] = passed

    out = idir / 'data'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'games.jsonl', 'w') as fh:
        for r in games:
            fh.write(json.dumps(r) + '\n')
    if not passed:
        summary['seconds'] = round(time.time() - t0, 1)
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print(f"PARITY FAILED: {summary['parity']}; raw data kept", flush=True)
        sys.exit(PARITY_FAIL)
    expect = {}
    for p, d in data.items():
        with open(out / f'p{p}.npz.tmp', 'wb') as fh:
            np.savez_compressed(fh, **d)
        os.replace(out / f'p{p}.npz.tmp', out / f'p{p}.npz')
        expect[f'p{p}.npz'] = len(d['k'])
    if tracerows:
        tdat = {k: np.concatenate([r[k] for r in tracerows]) for k in tracerows[0]}
        with open(out / 'trace.npz.tmp', 'wb') as fh:
            np.savez_compressed(fh, **tdat)
        os.replace(out / 'trace.npz.tmp', out / 'trace.npz')
        expect['trace.npz'] = len(tdat['game'])
    # verify, then delete the raw decision files
    for name, n in expect.items():
        with np.load(out / name) as z:
            if any(len(z[k]) != n for k in z.files):
                raise SystemExit(f'verify failed: {name}')
    summary['files'] = expect
    deleted = 0
    if not a.keep_raw:
        for f in raws:
            f.unlink(missing_ok=True); deleted += 1
    summary['raw_deleted'] = deleted
    summary['seconds'] = round(time.time() - t0, 1)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    pr = summary['parity']
    print(f"iter {a.iter}: {nvalid} valid games ({nbad} invalid, {missing} missing decisions), score {summary['score']}, "
          f"parity median {pr['median']:.2e} p999 {pr['p999']:.2e} n {pr['n']}; "
          + ', '.join(f"p{p} {data[p]['k'].shape[0]}" for p in sorted(data)) + f"; deleted {deleted} raw files", flush=True)


def saturation(d, picked, floor):
    """Fraction of sampled records whose softmax part (before the floor mix) gave k more than 0.999."""
    if not picked.any():
        return None
    m = np.maximum(d['mask'][picked].sum(1), 1)
    soft = (np.exp(d['logq'][picked].astype(np.float64)) - floor / m) / (1 - floor)
    return float((soft > 0.999).mean())


def _valid_game(g):
    r = g / 'result.json'
    if not r.exists():
        return False
    res = json.loads(r.read_text())
    return not (res.get('error') or not res.get('finished') or res['A'].get('failures') or res['B'].get('failures')) \
        and (g / 'trace.tsv').exists()


if __name__ == '__main__':
    main()
