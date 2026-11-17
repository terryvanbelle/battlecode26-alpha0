"""Turn a zdn RL iteration's games into training data (docs/zdn.md, "RL"): decision tables of the learner's recorded
turns, the parity check against the nets that played, outcomes and traces.

zdn_rl_data.py --config configs/rl-zdn1.json --iter K [--run RUN] [--keep-raw] [--max-turns N]
    [--iter-dir DIR --nets DIR]          (tests)

Reads DIR/jobs.json (default runs/rl/RUN/iter-NNNN), each valid game's result.json and trace.tsv, and the learner
sides' decisions-<side>.bin.gz (zdn REG/STEP records of a sample of turns, VER 2: every choice carries
q = round(-4096 log p)). Writes DIR/data/:
  <unit>.npz    (baby, king) the decision table (zdn_rl.load_table) plus, per turn: game int32 (job index),
                team int8 (0 A, 1 B), pair int32 (-1 unpaired), z float32 (+1 win, -1 loss, 0 draw for the turn's team)
  trace.npz     one row per learner team-game and trace row: game, team, round, pair, z, trace (rl_data's; fits V)
  summary.json  games, learner score by opponent, per unit: turns, rows by head, parity
  games.jsonl   one row per game (rl_data.game_row)
Parity: log p of every sampled choice with at least two options is recomputed from the record (zdn_rl.logp_rows,
nets/<version>/<unit>.json, the config temperature) and compared with -q / 4096. Pass: median |d| < 1e-3 and
99.9th percentile < 2e-3 per unit (config "parity"). On failure nothing is deleted and the exit code is 2.
--max-turns caps a unit's turns by uniform sampling of turns.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import rl_data  # noqa: E402
import zdn_rl  # noqa: E402

PARITY_FAIL = 2
UNITS = ('baby', 'king')


def parity(tab, nets_dir, unit, T, cache):
    """|log p recomputed - (-q / 4096)| of every sampled multi-option row (np.inf where the net is missing)."""
    import torch
    sel_rows = np.nonzero((tab['q'] >= 0) & (tab['label'] >= 0) & (tab['n'] >= 2))[0]
    out = np.full(len(sel_rows), np.nan)
    if not len(sel_rows):
        return out
    ver = tab['version'][tab['turn'][sel_rows]]
    for v in np.unique(ver).tolist():
        key = (v, unit)
        if key not in cache:
            f = Path(nets_dir) / str(v) / f'{unit}.json'
            if v < 0 or not f.exists():
                cache[key] = None
            else:
                net = zdn_rl.net_from_json(json.loads(f.read_text()))
                cache[key] = (net.eval(), zdn_rl.Norm.identity(net))
        m = ver == v
        if cache[key] is None:
            out[m] = np.inf
            continue
        net, norm = cache[key]
        rows = sel_rows[m]
        turns = np.unique(tab['turn'][rows])
        with torch.no_grad():
            for i in range(0, len(turns), 4096):
                tb = turns[i:i + 4096]
                rr = zdn_rl.turn_rows(tab, tb)
                rr = rr[np.isin(rr, rows)]
                _, lp, _ = zdn_rl.logp_rows(net, norm, tab, tb, T, 'cpu', rows=rr)
                d = np.abs(lp.numpy().astype(np.float64) + tab['q'][rr] / zdn_rl.LOGP_SCALE)
                out[np.searchsorted(sel_rows, rr)] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', default='')
    ap.add_argument('--iter', type=int, required=True)
    ap.add_argument('--iter-dir', default='')
    ap.add_argument('--nets', default='')
    ap.add_argument('--max-turns', type=int, help='per-unit turn cap (default: config max_turns_per_unit, 0 = none)')
    ap.add_argument('--keep-raw', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    cfg = json.loads(Path(a.config).read_text())
    rd = ROOT / 'runs' / 'rl' / (a.run or cfg['name'])
    idir = Path(a.iter_dir) if a.iter_dir else rd / f'iter-{a.iter:04d}'
    nets_dir = Path(a.nets) if a.nets else rd / 'nets'
    doc = json.loads((idir / 'jobs.json').read_text())
    T = cfg.get('temperature', 1.0)
    cap = cfg.get('max_turns_per_unit', 0) if a.max_turns is None else a.max_turns
    pcfg = cfg.get('parity', {})
    rng = np.random.default_rng(cfg['seed_base'] + a.iter)

    files, fmeta, tracerows, games, score = [], [], [], [], {}
    nvalid = nbad = missing = 0
    for gi, j in enumerate(doc['jobs']):
        g = idir / 'games' / j['id']
        if not rl_data._valid_game(g):
            nbad += (g / 'result.json').exists()
            continue
        res = json.loads((g / 'result.json').read_text())
        tr = rl_data.read_trace(g / 'trace.tsv')
        if not len(tr):
            nbad += 1
            continue
        fs = {s: g / f'decisions-{s}.bin.gz' for s in j['learner_sides']}
        if not all(f.exists() for f in fs.values()):
            missing += 1
            continue
        nvalid += 1
        games.append(rl_data.game_row(j, res, tr))
        if j['kind'] == 'pool':
            sc = score.setdefault(j['opponent'], [0.0, 0])
            sc[0] += (rl_data.z_of(res, j['learner_sides']) + 1) / 2; sc[1] += 1
        pair = j['pair']
        if pair >= 0 and not all(rl_data._valid_game(idir / 'games' / m['id']) for m in doc['jobs']
                                 if m['pair'] == pair and m['id'] != j['id']):
            pair = -1
        area = rl_data.map_area(j['map'])
        for side, f in fs.items():
            team = 'AB'.index(side)
            z = rl_data.z_of(res, side)
            files.append(f); fmeta.append((gi, team, pair, z))
            tf = rl_data.trace_features(tr, team, area)
            tracerows.append(dict(game=np.full(len(tr), gi, np.int32), team=np.full(len(tr), team, np.int8),
                                  round=tr[:, 0].astype(np.int32), pair=np.full(len(tr), pair, np.int32),
                                  z=np.full(len(tr), z, np.float32), trace=tf))

    summary = dict(iter=a.iter, run=rd.name, learner=doc['learner'], temperature=T, games=len(doc['jobs']), valid=nvalid,
                   invalid=nbad, missing_decisions=missing,
                   score={o: dict(points=v[0], games=v[1], score=round(v[0] / v[1], 4) if v[1] else None)
                          for o, v in sorted(score.items())}, units={})
    out = idir / 'data'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'games.jsonl', 'w') as fh:
        for r in games:
            fh.write(json.dumps(r) + '\n')
    tables, failed, cache = {}, [], {}
    fm = np.array(fmeta, dtype=np.float64).reshape(-1, 4)
    for unit in UNITS:
        tab = zdn_rl.load_table(files, unit)
        if tab is None:
            summary['units'][unit] = dict(turns=0)
            continue
        N0 = len(tab['SL'])
        if cap and N0 > cap:
            tab = zdn_rl.subset(tab, np.sort(rng.choice(N0, cap, replace=False)))
        f = tab['src']
        tab['game'] = fm[f, 0].astype(np.int32); tab['team'] = fm[f, 1].astype(np.int8)
        tab['pair'] = fm[f, 2].astype(np.int32); tab['z'] = fm[f, 3].astype(np.float32)
        lv = doc['learner']['version']
        d = parity(tab, nets_dir, unit, T, cache)
        st = dict(turns=int(len(tab['SL'])), turns_before_cap=int(N0), rows=int(len(tab['head'])),
                  rows_by_head={str(h): int((tab['head'] == h).sum()) for h in np.unique(tab['head']).tolist()},
                  sampled_multi=int(len(d)), versions=sorted(set(tab['version'].tolist())),
                  learner_version_frac=float((tab['version'] == lv).mean()),
                  parity_n=int(len(d)), parity_missing_net=int(np.isinf(d).sum()),
                  parity_median=float(np.median(d)) if len(d) else None,
                  parity_p999=float(np.quantile(d, 0.999)) if len(d) else None,
                  parity_max=float(d.max()) if len(d) else None,
                  steps_per_turn=float((tab['head'] < 0).sum() / max(len(tab['SL']), 1)))
        if len(d) and not (st['parity_median'] < pcfg.get('median_max', 1e-3) and st['parity_p999'] < pcfg.get('p999_max', 2e-3)):
            failed.append(unit)
        summary['units'][unit] = st
        tables[unit] = tab
    summary['parity'] = dict(failed_units=failed, passed=not failed and bool(tables))
    if failed or not tables:
        summary['seconds'] = round(time.time() - t0, 1)
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print(f'PARITY FAILED ({failed or "no data"}): ' + json.dumps(summary['units']) + '; raw data kept', flush=True)
        sys.exit(PARITY_FAIL)
    expect = {}
    for unit, tab in tables.items():
        meta = tab.pop('meta')
        with open(out / f'{unit}.npz.tmp', 'wb') as fh:
            np.savez_compressed(fh, meta=np.array(json.dumps(meta)), **tab)
        os.replace(out / f'{unit}.npz.tmp', out / f'{unit}.npz')
        expect[f'{unit}.npz'] = len(tab['SL'])
    if tracerows:
        tdat = {k: np.concatenate([r[k] for r in tracerows]) for k in tracerows[0]}
        with open(out / 'trace.npz.tmp', 'wb') as fh:
            np.savez_compressed(fh, **tdat)
        os.replace(out / 'trace.npz.tmp', out / 'trace.npz')
        expect['trace.npz'] = len(tdat['game'])
    for name, n in expect.items():
        with np.load(out / name) as zf:
            k0 = 'SL' if name != 'trace.npz' else 'game'
            if len(zf[k0]) != n:
                raise SystemExit(f'verify failed: {name}')
    summary['files'] = expect
    deleted = 0
    if not a.keep_raw:
        for f in files:
            f.unlink(missing_ok=True); deleted += 1
    summary['raw_deleted'] = deleted
    summary['seconds'] = round(time.time() - t0, 1)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    u = summary['units']
    print(f"iter {a.iter}: {nvalid} valid games ({nbad} invalid, {missing} missing decisions), score {summary['score']}, "
          + '; '.join(f"{k} {v['turns']} turns {v.get('rows', 0)} rows parity median {v.get('parity_median') or 0:.2e} "
                      f"p999 {v.get('parity_p999') or 0:.2e}" for k, v in u.items())
          + f'; deleted {deleted} raw files', flush=True)


def load(path):
    """A <unit>.npz back into a decision table (meta restored)."""
    with np.load(path) as z:
        tab = {k: z[k] for k in z.files if k != 'meta'}
        tab['meta'] = json.loads(str(z['meta']))
    return tab


if __name__ == '__main__':
    main()
