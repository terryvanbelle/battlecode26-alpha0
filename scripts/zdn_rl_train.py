"""One PPO iteration over every head of the zdn nets (docs/zdn.md, "RL"): advantages from the critic, a clipped
policy-gradient update of all registers and steps with a KL leash to the anchor (the imitation nets), export.

zdn_rl_train.py --nets runs/rl/R/nets/3 --anchor runs/zdn/zdn1 --data runs/rl/R/iter-0003/data \\
    --value-data runs/rl/R/iter-000{1,2,3}/data --out runs/rl/R/nets/4 [--version 3] [--units baby,king]
    [--temperature 1] [--lr 3e-4] [--epochs 2] [--batch-turns 512] [--clip 0.2] [--target-kl 0.05] [--grad-clip 0.5]
    [--beta 0.05] [--c-ent 0.003] [--max-anchor-kl 0.5] [--adv-clip 5] [--critic trace|richlin|rich]
    [--gae-lambda 1] [--critic-folds 4] [--device auto]

--nets holds the learner theta_k (the nets that played --data): <unit>.json, and <unit>.norm.npz (the training
standardisation; else fitted on the data and written to --out). --anchor holds the KL anchor's <unit>.json.
Data: zdn_rl_data.py's <unit>.npz (decision tables) and trace.npz.

Advantage of a turn (every row of the turn shares it): as rl_train.py, from the turn's team-game and 10-round bin:
A = z - V(trace) (--critic trace, --gae-lambda 1, minus the seat-swapped pair baseline), or GAE over the bins under
rl_critic's critic; then standardised per unit and clipped at --adv-clip.
Loss over the rows of a batch of turns (a row: one sampled choice with at least two options, of any head):
  -mean min(r A, clip(r, 1 - clip, 1 + clip) A) - c_ent mean H(pi) + beta mean KL(pi || anchor),
  r = pi(label) / exp(-q / 4096) (q: the behaviour log-probability the player recorded; zdn_rl_data checked it).
A batch whose approximate KL(theta_k -> theta) exceeds --target-kl stops the unit's training (before its update).
Outputs in --out: <unit>.json and <unit>.norm.npz per unit (inputs copied unchanged when a unit took no step),
metrics.json. When a unit's KL to the anchor ends above --max-anchor-kl, only metrics.json is written; exit code 3.
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import rl_critic, rl_train, zdn_rl, zdn_rl_data  # noqa: E402

REFUSED = 3
UNITS = ('baby', 'king')


def advantages(a, tabs, dir_index):
    """{unit: raw A per turn} and metrics."""
    vdirs = a.value_data if a.value_data else a.data
    use_gae = a.critic != 'trace' or a.gae_lambda < 1
    V = rl_train.Value(None if use_gae and a.critic != 'trace' else rl_train.value_rows(vdirs, dir_index), a.val_frac, a.seed)
    m = dict(value=V.metrics)
    if use_gae:
        ns = argparse.Namespace(critic=a.critic, gae_lambda=a.gae_lambda, critic_folds=a.critic_folds, seed=a.seed, data=a.data)
        rows, A_rows, gm, _ = rl_train.gae_rows(ns, vdirs, dir_index, V)
        base = {} if a.gae_lambda < 1 else rl_train.pair_baseline_rows(rows, A_rows)
        m['critic'] = gm
    else:
        rows = rl_train.value_rows(a.data, dir_index)
        A_rows = rows['z'].astype(np.float64) - V(rows['trace'])
        base = rl_train.pair_baseline(rows, V)
    m['pairs_with_baseline'] = len(base)
    out = {}
    for unit, tab in tabs.items():
        ri = rl_critic.record_rows(rows, tab['game'], tab['team'], tab['round'])
        A = np.where(ri >= 0, A_rows[np.maximum(ri, 0)], 0.0)
        if base:
            b = np.array([base.get(int(g), np.nan) for g in tab['game'] * 2 + tab['team'].astype(np.int64)])
            A = A - np.where(np.isfinite(b), b, 0.0)
        m[f'{unit}_no_bin'] = int((ri < 0).sum())
        out[unit] = A
    return out, m


def train_unit(a, unit, tab, A_turn, dev, mt):
    """PPO on one unit's table. Returns (net, norm) to export, or None when no step was taken."""
    import torch
    raw = zdn_rl.net_from_json(json.loads((Path(a.nets) / f'{unit}.json').read_text()))
    nf = Path(a.nets) / f'{unit}.norm.npz'
    norm = zdn_rl.Norm.load(nf) if nf.exists() else zdn_rl.Norm.fit(tab, raw)
    mt['norm_from_data'] = not nf.exists()
    net = zdn_rl.unfold(raw, norm).to(dev)
    anc = zdn_rl.net_from_json(json.loads((Path(a.anchor) / f'{unit}.json').read_text())).to(dev).eval()
    anc_norm = zdn_rl.Norm.identity(anc)
    for p in anc.parameters():
        p.requires_grad_(False)

    usable = (tab['q'] >= 0) & (tab['label'] >= 0) & (tab['n'] >= 2)
    if a.version is not None:
        tv = tab['version'][tab['turn']]
        mt['version_mismatch_rows'] = int((usable & (tv != a.version)).sum())
        usable &= tv == a.version
    mt['rows'] = int(len(usable)); mt['usable_rows'] = int(usable.sum())
    turns = np.unique(tab['turn'][usable])
    mt['turns'] = int(len(turns))
    if len(turns) < 2 or a.lr == 0:
        return None
    A = A_turn.astype(np.float64)
    mt['adv_raw_mean'], mt['adv_raw_sd'] = float(A[turns].mean()), float(A[turns].std())
    A = np.clip((A - A[turns].mean()) / (A[turns].std() + 1e-8), -a.adv_clip, a.adv_clip)
    st_rows = np.nonzero(usable & (tab['head'] < 0))[0]
    if len(st_rows):
        ck = tab['kind'][tab['o0'][st_rows] + tab['label'][st_rows]]
        Araw = A_turn[tab['turn'][st_rows]]
        mt['adv_by_step_kind'] = {int(k): [int((ck == k).sum()), round(float(Araw[ck == k].mean()), 4)] for k in np.unique(ck)}

    rng = np.random.default_rng(a.seed)
    ev_turns = np.sort(rng.choice(turns, min(len(turns), a.eval_turns), replace=False))

    def terms(tb, grad):
        rows = zdn_rl.turn_rows(tab, tb)
        rows = rows[usable[rows]]
        with torch.set_grad_enabled(grad):
            _, lp, full = zdn_rl.logp_rows(net, norm, tab, tb, a.temperature, dev, rows=rows)
        with torch.no_grad():
            _, _, afull = zdn_rl.logp_rows(anc, anc_norm, tab, tb, a.temperature, dev, rows=rows)
        ent = torch.zeros(len(rows), device=dev); kla = torch.zeros(len(rows), device=dev)
        for h, (pos, lsm) in full.items():
            p = lsm.exp()
            alsm = afull[h][1]
            K = min(lsm.shape[1], alsm.shape[1])
            ip = torch.from_numpy(pos).to(dev)
            ent = ent.index_put((ip,), -(p * lsm).sum(-1))
            kla = kla.index_put((ip,), (p[:, :K] * (lsm[:, :K] - alsm[:, :K])).sum(-1))
        lq = torch.from_numpy(-tab['q'][rows].astype(np.float32) / zdn_rl.LOGP_SCALE).to(dev)
        At = torch.from_numpy(A[tab['turn'][rows]].astype(np.float32)).to(dev)
        return rows, lp, lq, At, ent, kla

    def summary(tag):
        net.eval()
        tot = dict(ent=0.0, kla=0.0, klold=0.0, n=0)
        byh = {}
        with torch.no_grad():
            for i in range(0, len(ev_turns), a.batch_turns):
                rows, lp, lq, _, ent, kla = terms(ev_turns[i:i + a.batch_turns], False)
                lr_ = (lp - lq)
                klo = (lr_.exp() - 1 - lr_)          # approx KL(theta_k -> theta), per row
                tot['ent'] += float(ent.sum()); tot['kla'] += float(kla.sum()); tot['klold'] += float(klo.sum()); tot['n'] += len(rows)
                hd = tab['head'][rows]
                for h in np.unique(hd):
                    s = torch.from_numpy(hd == h).to(dev)
                    b = byh.setdefault(int(h), [0.0, 0.0, 0])
                    b[0] += float(ent[s].sum()); b[1] += float(kla[s].sum()); b[2] += int(s.sum())
        net.train()
        n = max(tot['n'], 1)
        mt[f'entropy_{tag}'] = tot['ent'] / n; mt[f'kl_anchor_{tag}'] = tot['kla'] / n; mt[f'kl_old_{tag}'] = tot['klold'] / n
        mt[f'by_head_{tag}'] = {str(h): dict(rows=v[2], entropy=round(v[0] / v[2], 4), kl_anchor=round(v[1] / v[2], 4))
                                for h, v in sorted(byh.items())}

    summary('start')
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    steps = clipped = seen = 0
    stopped = None
    for ep in range(a.epochs):
        perm = rng.permutation(turns)
        for i in range(0, len(perm), a.batch_turns):
            rows, lp, lq, At, ent, kla = terms(perm[i:i + a.batch_turns], True)
            if not len(rows):
                continue
            logr = lp - lq
            r = logr.exp()
            approx_kl = float(((r - 1) - logr).mean())
            if approx_kl > a.target_kl:
                stopped = dict(epoch=ep, step=steps, approx_kl=approx_kl)
                break
            surr = torch.min(r * At, torch.clamp(r, 1 - a.clip, 1 + a.clip) * At)
            loss = -surr.mean() - a.c_ent * ent.mean() + a.beta * kla.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.grad_clip)
            opt.step()
            steps += 1
            clipped += int(((r.detach() - 1).abs() > a.clip).sum()); seen += len(rows)
        if stopped:
            break
    mt.update(steps=steps, clip_frac=clipped / seen if seen else 0.0, early_stop=stopped)
    summary('end')
    if steps == 0:
        return None
    return net.cpu(), norm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nets', required=True, help='dir of the learner: <unit>.json [+ <unit>.norm.npz]')
    ap.add_argument('--anchor', required=True, help='dir of the KL anchor: <unit>.json')
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--value-data', nargs='*', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--version', type=int, default=None, help="keep only turns recorded by this net version")
    ap.add_argument('--units', default='baby,king')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-turns', type=int, default=512)
    ap.add_argument('--eval-turns', type=int, default=20000)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--target-kl', type=float, default=0.05)
    ap.add_argument('--grad-clip', type=float, default=0.5)
    ap.add_argument('--beta', type=float, default=0.05)
    ap.add_argument('--c-ent', type=float, default=0.003)
    ap.add_argument('--max-anchor-kl', type=float, default=0.5)
    ap.add_argument('--adv-clip', type=float, default=5.0)
    ap.add_argument('--critic', default='trace', choices=('trace', 'richlin', 'rich'))
    ap.add_argument('--gae-lambda', type=float, default=1.0)
    ap.add_argument('--critic-folds', type=int, default=4)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--device', default='auto', help='auto: cuda when available, else cpu')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    if a.device == 'auto':
        a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    units = [u for u in a.units.split(',') if u]
    vdirs = a.value_data if a.value_data else a.data
    dir_index = {str(d): i for i, d in enumerate(dict.fromkeys(list(a.data) + list(vdirs)))}
    tabs = {}
    for unit in units:
        parts = []
        for d in a.data:
            f = Path(d) / f'{unit}.npz'
            if f.exists():
                t = zdn_rl_data.load(f)
                t['game'] = t['game'].astype(np.int64) + (dir_index[str(d)] << 32)
                t['pair'] = np.where(t['pair'] >= 0, t['pair'].astype(np.int64) + (dir_index[str(d)] << 32), -1)
                parts.append(t)
        if len(parts) > 1:
            raise SystemExit('one --data dir per unit is supported')   # tables carry per-dir option offsets
        if parts:
            tabs[unit] = parts[0]
    A, m = advantages(a, tabs, dir_index)
    print(f"value model: {m['value']}" + (f"; critic: {m['critic']}" if 'critic' in m else ''), flush=True)
    metrics = dict(args=vars(a), **m, units={})
    results, refused = {}, []
    for unit in units:
        mt = {}
        metrics['units'][unit] = mt
        if unit not in tabs:
            print(f'{unit}: no data, unchanged', flush=True)
            continue
        tu = time.time()
        res = train_unit(a, unit, tabs[unit], A[unit], torch.device(a.device), mt)
        mt['seconds'] = round(time.time() - tu, 1)
        if res is not None:
            results[unit] = res
        if mt.get('kl_anchor_end', 0) > a.max_anchor_kl:
            refused.append(unit)
        print(f"{unit}: {mt.get('usable_rows', 0)}/{mt.get('rows', 0)} rows in {mt.get('turns', 0)} turns, steps "
              f"{mt.get('steps', 0)}, early stop {mt.get('early_stop')}, KL anchor {mt.get('kl_anchor_start', 0):.4f} -> "
              f"{mt.get('kl_anchor_end', 0):.4f}, entropy {mt.get('entropy_start', 0):.4f} -> {mt.get('entropy_end', 0):.4f}, "
              f"KL old {mt.get('kl_old_start', 0):.2e} -> {mt.get('kl_old_end', 0):.4f}, clip {mt.get('clip_frac', 0):.3f}", flush=True)
        if 'adv_by_step_kind' in mt:
            print(f"{unit} advantage by chosen step kind (rows, mean raw A): {mt['adv_by_step_kind']}", flush=True)
    metrics['refused'] = refused
    if refused:
        metrics['exported'] = False
        metrics['seconds'] = round(time.time() - t0, 1)
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=1, default=str))
        print(f'REFUSED: KL to anchor > {a.max_anchor_kl} on {refused}; nothing exported', flush=True)
        sys.exit(REFUSED)
    for unit in UNITS:
        src = Path(a.nets) / f'{unit}.json'
        if unit in results:
            net, norm = results[unit]
            meta = json.loads(src.read_text())['meta']
            zdn_rl.export(net, norm, out / f'{unit}.json', meta)
            norm.save(out / f'{unit}.norm.npz')
        elif src.exists():
            shutil.copyfile(src, out / f'{unit}.json')
            if (Path(a.nets) / f'{unit}.norm.npz').exists():
                shutil.copyfile(Path(a.nets) / f'{unit}.norm.npz', out / f'{unit}.norm.npz')
    metrics['changed_units'] = sorted(results)
    metrics['exported'] = True
    metrics['seconds'] = round(time.time() - t0, 1)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1, default=str))
    print(f'wrote {out} (changed {sorted(results)}) in {metrics["seconds"]}s', flush=True)


if __name__ == '__main__':
    main()
