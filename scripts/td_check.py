"""C0 (docs/rl-es-plan.md section 3): how much per-decision TD signal do recorded self-play games carry?

td_check.py --run-dir runs/rl/rl-m2 --config configs/rl-m2.json --iters 36-39 [--window 32-35]
    [--points 5,7,10,11,12,13,16] [--horizon 100] [--critic rich] [--folds 4] [--out FILE]

Every record of a listed point with at least two allowed options gets pi(.|x) from the net that played it
(nets/<version>/net.json, the config's floor / floor_points / temperature; the parity median |log pi_k - logq| is
reported). A natural deviation is a sampled option other than pi's top choice: D = 1[k != argmax pi], with propensity
e = 1 - max pi. The R-learner (Robinson) estimate of a deviation's mean effect on a target Y, clustered by game:
    tau = sum Y (D - e) / sum (D - e)^2,   se from the per-game sums of (Y - tau (D - e)) (D - e),   t = tau / se.
Targets use the critic V (rl_critic.fit_crossed on the window + target team-game bins; target games out of fold) at
the record's 10-round bin b:
    td   V(b + horizon) - V(b), or z - V(b) when the team-game ends sooner (the plan's TD-H100 at horizon 100)
    mc   z - V(b)
    pre  V(b) - V(b - horizon): a placebo fixed before the decision, so its t is N(0, 1) if the clustering is right
Per option k the same estimate with D_k = 1[k_i = k] - pi_k(x_i) gives option k's advantage over pi's average.
Plan rule: median |t_td| over the Phase-A points (5, 7, 10, 11, 12, 13) below 1 -> stage B proposes by low-rank ES,
otherwise by CPI.

--cpi: the state-dependent version CPI would use. Option k's advantage is linear in phi(x) = [1, the top --pcs
principal components of the playing net's last hidden layer]: Y_td ~ sum_k (1[k_i = k] - pi_k(x_i)) w_k . phi(x_i)
(ridge, cross-fitted over two halves of the games). Out of fold, the regression of Y_td on the predicted advantage
of the sampled option (game-clustered t) says whether the fitted advantages carry over to unseen games.
Use t_fold_mean, the mean of the two folds' own t (fit on one half, test on the other; each is N(0, 1) under no
signal). t_oof pools both halves' predictions into one clustered t, but the two halves' terms are nearly the same
bilinear form of the two halves' scores (the folds' t correlate 0.96-0.99 under the sham null), so t_oof is about
sqrt(2) times too large: kept only for comparison with earlier runs.
The same check on the mc target (z - V) is reported as cpi_mc (a secondary diagnostic: the plan's rule uses td).
Leverage: top_game_share is the largest share of one game in the out-of-fold sum of squared predicted advantages,
eff_games the effective number of games (sum s_g)^2 / sum s_g^2; a clustered t leans on few games when it is small.
--sham N (with --cpi): the same out-of-fold check N times with each record's option redrawn from pi(.|x) (choices
that cannot affect Y), giving the statistic's own null distribution at each point (cpi_sham: the t values, their
mean and sd, and the empirical p of the real t_oof, (1 + #{sham t >= t_oof}) / (1 + N)).
--save-fit FILE (with --cpi): per point, each playing version's PC basis (mu, sd, vt, s: pc_r = ((h - mu) / sd) . vt_r
/ s_r, h the last hidden layer after ReLU) and W [options, 1 + pcs] fitted on all target games, for cpi_step.py.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dpnet_io, rl_critic, rl_train  # noqa: E402

NONE = 0xFFFF
PHASE_A = (5, 7, 10, 11, 12, 13)


def ints(s):
    out = []
    for part in (s or '').split(','):
        if '-' in part:
            lo, hi = map(int, part.split('-'))
            out += list(range(lo, hi + 1))
        elif part:
            out.append(int(part))
    return out


def shifted(rows, v, L):
    """(ahead, behind) per team-game bin row: v L bins later in its team-game (z when it ends sooner) and v L bins
    earlier (NaN before the team-game's first bin)."""
    order, seg, first = rl_critic.seq_index(rows)
    n = len(order)
    vs, zs = v[order].astype(np.float64), rows['z'][order].astype(np.float64)
    pos = np.arange(n)
    fwd = pos + L
    okf = fwd < n
    okf[okf] = seg[fwd[okf]] == seg[pos[okf]]
    back = pos - L
    okb = back >= first
    ahead, behind = np.empty(n), np.empty(n)
    ahead[order] = np.where(okf, vs[np.minimum(fwd, n - 1)], zs)
    behind[order] = np.where(okb, vs[np.maximum(back, 0)], np.nan)
    return ahead, behind


def rlearn(Y, R, g):
    """dict(tau, se, t, n) of sum Y R / sum R^2, standard error clustered by g; None without variation."""
    ok = np.isfinite(Y)
    Y, R, g = Y[ok], R[ok], g[ok]
    Q = R * R
    den = float(Q.sum())
    if den <= 1e-12:
        return None
    tau = float((Y * R).sum() / den)
    _, inv = np.unique(g, return_inverse=True)
    S = np.bincount(inv, weights=(Y - tau * R) * R)
    se = float(np.sqrt((S * S).sum()) / den)
    return dict(tau=round(tau, 5), se=round(se, 5), t=round(tau / se, 2) if se > 0 else 0.0, n=int(ok.sum()))


def cpi_check(R, Phi, Y, g, ridge, seed=0, full=True):
    """Out-of-fold check of a linear state-dependent advantage model. R [N, K] treatment residuals (0 where an option
    is not allowed), Phi [N, P] features, Y [N] target, g [N] game ids. Returns (dict(t_oof, slope, top_game_share,
    eff_games, in_sample_t), W [K, P] fitted on all games); full=False skips the all-games fit (W None)."""
    N, K = R.shape
    P = Phi.shape[1]
    games = np.unique(g)
    half = np.isin(g, np.random.default_rng(seed).permutation(games)[:len(games) // 2])

    def fit(sel):
        A = np.zeros((K * P, K * P)); b = np.zeros(K * P)
        idx = np.nonzero(sel)[0]
        for s0 in range(0, len(idx), 20000):
            ii = idx[s0:s0 + 20000]
            Z = (R[ii][:, :, None] * Phi[ii][:, None, :]).reshape(len(ii), K * P)
            A += Z.T @ Z; b += Z.T @ Y[ii]
        lam = ridge * max(float(np.trace(A)) / (K * P), 1e-12)
        return np.linalg.solve(A + lam * np.eye(K * P), b).reshape(K, P)

    def pred(W, sel):
        idx = np.nonzero(sel)[0]
        return (R[idx] * (Phi[idx] @ W.T)).sum(1), idx

    out = {}
    ahat = np.zeros(N)
    tf = []
    for tr in (half, ~half):
        W = fit(tr)
        a_, idx = pred(W, ~tr)
        ahat[idx] = a_
        rf = rlearn(Y[idx], a_, g[idx])
        tf.append(None if rf is None else rf['t'])
    out['t_folds'] = tf
    out['t_fold_mean'] = None if None in tf else round(float(np.mean(tf)), 2)
    r = rlearn(Y, ahat, g)
    out['t_oof'] = None if r is None else r['t']
    out['slope'] = None if r is None else r['tau']
    okf = np.isfinite(Y)
    _, inv = np.unique(g[okf], return_inverse=True)
    sg = np.bincount(inv, weights=ahat[okf] ** 2)
    out['top_game_share'] = round(float(sg.max() / max(sg.sum(), 1e-30)), 4)
    out['eff_games'] = round(float(sg.sum() ** 2 / max((sg * sg).sum(), 1e-30)), 1)
    if not full:
        return out, None
    Wall = fit(np.ones(N, bool))
    a_, idx = pred(Wall, np.ones(N, bool))
    ri = rlearn(Y, a_, g)
    out['in_sample_t'] = None if ri is None else ri['t']
    out['features'] = P; out['options'] = K
    return out, Wall

def sham_check(P, U, Phi, Y, g, ridge, n, seed, t_real):
    """cpi_check's out-of-fold t with each record's option redrawn from P (the policy, over the allowed options U):
    the statistic's null distribution on these states and targets."""
    rng = np.random.default_rng(seed)
    Q = P * U
    c = np.cumsum(Q, 1)
    tot = c[:, -1:]
    has = tot[:, 0] > 0
    ar = np.arange(Q.shape[1])[None, :]
    ts, shares, folds, fm, t0 = [], [], [], [], time.time()
    for _ in range(n):
        ks = np.minimum((c < rng.random(len(Q))[:, None] * tot).sum(1), Q.shape[1] - 1)
        Rs = np.where(U, (ks[:, None] == ar).astype(np.float64) - P, 0.0)
        Rs[~has] = 0
        r = cpi_check(Rs, Phi, Y, g, ridge, full=False)[0]
        if r['t_oof'] is not None:
            ts.append(float(r['t_oof'])); shares.append(r['top_game_share'])
            if None not in r['t_folds']:
                folds.append(r['t_folds']); fm.append(r['t_fold_mean'])
    tt, fmm = np.array(ts), np.array(fm)
    fa = np.array(folds) if folds else np.zeros((0, 2))
    return dict(n=len(tt), t=ts, mean=round(float(tt.mean()), 2), sd=round(float(tt.std(ddof=1)), 2),
                fold_sd=[round(float(x), 2) for x in fa.std(0, ddof=1)] if len(fa) > 2 else None,
                fold_corr=round(float(np.corrcoef(fa.T)[0, 1]), 2) if len(fa) > 2 else None,
                fold_mean_sd=round(float(fmm.std(ddof=1)), 2) if len(fmm) > 2 else None,
                ge2=int((tt >= 2).sum()), abs_gt2=int((np.abs(tt) > 2).sum()), max_abs=round(float(np.abs(tt).max()), 2),
                p_real=None if t_real is None else round(float((1 + (tt >= t_real).sum()) / (1 + len(tt))), 3),
                median_top_game_share=round(float(np.median(shares)), 4), seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--iters', required=True)
    ap.add_argument('--window', default='')
    ap.add_argument('--points', default='5,7,10,11,12,13,16')
    ap.add_argument('--horizon', type=int, default=100)
    ap.add_argument('--critic', default='rich')
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out')
    ap.add_argument('--cpi', action='store_true')
    ap.add_argument('--pcs', type=int, default=8)
    ap.add_argument('--ridge', type=float, default=0.1)
    ap.add_argument('--save-fit')
    ap.add_argument('--sham', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    cfg = json.loads(Path(a.config).read_text())
    floor = cfg['floor']
    fpts = {int(k): float(v) for k, v in (cfg.get('floor_points') or {}).items()}
    T = cfg.get('temperature', 1.0)
    rd = Path(a.run_dir)
    tdirs = [rd / f'iter-{k:04d}' / 'data' for k in ints(a.iters)]
    wdirs = [rd / f'iter-{k:04d}' / 'data' for k in ints(a.window)]
    alld = list(dict.fromkeys(wdirs + tdirs))
    dir_index = {str(d): i for i, d in enumerate(alld)}
    rows = rl_train.value_rows(alld, dir_index)
    tgt = np.isin(rows['game'] >> 32, [dir_index[str(d)] for d in tdirs])
    v, oof = rl_critic.fit_crossed(rows, tgt, a.critic, folds=a.folds, seed=a.seed)
    ahead, behind = shifted(rows, v, a.horizon // 10)
    out = dict(run=str(rd), iters=a.iters, window=a.window, horizon=a.horizon, critic=a.critic,
               critic_seconds=round(time.time() - t0, 1), heldout=rl_critic.metrics(v, rows, tgt & oof), points={})
    print(f"critic {a.critic}: {int(tgt.sum())} target bins, held-out {out['heldout']['all']} "
          f"({out['critic_seconds']} s)", flush=True)

    nets = {}
    fits = {}

    def net(ver):
        if ver not in nets:
            f = rd / 'nets' / str(ver) / 'net.json'
            nets[ver] = dpnet_io.load_net(f) if f.exists() else None
        return nets[ver]

    for p in ints(a.points):
        d = rl_train.load_dirs(tdirs, [p], dir_index).get(p)
        if d is None:
            continue
        fl = fpts.get(p, floor)
        N = len(d['k'])
        pk_all = np.full(N, np.nan)
        e, D, ok = np.zeros(N), np.zeros(N), np.zeros(N, bool)
        opt_R = {}   # option -> (rows, residual 1[k = o] - pi_o)
        picked = d['k'] != NONE
        locals_cpi = {}
        for ver in np.unique(d['version'][picked]).tolist():
            nv = net(ver)
            sel = np.nonzero(picked & (d['version'] == ver))[0]
            if nv is None or p not in nv:
                continue
            ls = nv[p]
            n_out = ls[-1]['w'].shape[0]
            m = dpnet_io.fit_mask(d['mask'][sel], len(sel), n_out)
            pr = dpnet_io.probs(dpnet_io.forward(ls, d['x'][sel], exact=False), m, T, fl).astype(np.float64)
            k = d['k'][sel].astype(np.int64)
            inr = k < n_out
            two = m.sum(1) >= 2
            use = inr & two
            top = pr.argmax(1)
            if a.cpi:
                h = np.maximum(dpnet_io.forward(ls[:-1], d['x'][sel], exact=False), 0).astype(np.float64)
                mu, sd = h.mean(0), h.std(0) + 1e-6
                hz = (h - mu) / sd
                _, _, vt = np.linalg.svd(hz[np.random.default_rng(0).permutation(len(hz))[:20000]], full_matrices=False)
                pcs = hz @ vt[:a.pcs].T
                s_pc = pcs.std(0) + 1e-6
                pcs /= s_pc
                locals_cpi.setdefault('basis', {})[int(ver)] = dict(mu=mu.tolist(), sd=sd.tolist(),
                                                                    vt=vt[:a.pcs].tolist(), s=s_pc.tolist())
                if 'Phi' not in locals_cpi:
                    locals_cpi['Phi'] = np.zeros((N, a.pcs + 1)); locals_cpi['R'] = np.zeros((N, n_out))
                    if a.sham:
                        locals_cpi['P'] = np.zeros((N, n_out)); locals_cpi['U'] = np.zeros((N, n_out), bool)
                locals_cpi['Phi'][sel] = np.c_[np.ones(len(sel)), pcs]
                Rk = np.where(m, (k[:, None] == np.arange(n_out)[None, :]).astype(np.float64) - pr, 0.0)
                Rk[~use] = 0
                locals_cpi['R'][sel, :Rk.shape[1]] = Rk
                if a.sham:
                    locals_cpi['P'][sel, :pr.shape[1]] = pr
                    locals_cpi['U'][sel, :m.shape[1]] = m & use[:, None]
            pk_all[sel[inr]] = pr[np.nonzero(inr)[0], k[inr]]
            s2 = sel[use]
            e[s2] = 1 - pr[use, top[use]]
            D[s2] = (k[use] != top[use]).astype(np.float64)
            ok[s2] = True
            for o in range(n_out):
                al = use & m[:, o]
                if not al.any():
                    continue
                rr = (k[al] == o).astype(np.float64) - pr[al, o]
                r0, q0 = opt_R.get(o, (np.zeros(0, np.int64), np.zeros(0)))
                opt_R[o] = (np.r_[r0, sel[al]], np.r_[q0, rr])
        par = np.abs(np.log(np.maximum(pk_all[picked], 1e-30)) - d['logq'][picked].astype(np.float64))
        bi = rl_critic.record_rows(rows, d['game'], d['team'], d['round'])
        good = ok & (bi >= 0)
        good[good] = tgt[bi[good]] & np.isfinite(v[bi[good]])
        b = np.where(good, bi, 0)
        Y = dict(td=ahead[b] - v[b], mc=rows['z'][b].astype(np.float64) - v[b], pre=v[b] - behind[b])
        g = d['game']
        R = D - e
        rnd = d['round']
        st = dict(records=int(N), used=int(good.sum()), deviations=int(D[good].sum()),
                  deviation_rate=round(float(D[good].mean()), 4) if good.any() else None,
                  mean_e=round(float(e[good].mean()), 4) if good.any() else None,
                  parity_median=float(np.nanmedian(par)) if len(par) else None)
        for name, y in Y.items():
            st[name] = rlearn(y[good], R[good], g[good])
        for lo, hi in ((0, 500), (500, 1000), (1000, 2001)):
            sb = good & (rnd >= lo) & (rnd < hi)
            st[f'td_r{lo}'] = rlearn(Y['td'][sb], R[sb], g[sb])
        per = {}
        for o, (ix, rr) in sorted(opt_R.items()):
            gm = good[ix]
            if gm.sum() < 100:
                continue
            ix, rr = ix[gm], rr[gm]
            chosen = int((rr > 0).sum())
            r = rlearn(Y['td'][ix], rr, g[ix])
            if r is not None and chosen >= 20:
                per[o] = dict(r, chosen=chosen, rate=round(chosen / len(ix), 4))
        ts = [abs(r['t']) for r in per.values()]
        st['options'] = dict(tested=len(per), t_gt2=int(sum(t > 2 for t in ts)),
                             chance_t_gt2=round(0.0455 * len(per), 1),
                             top=sorted(([o, r['tau'], r['t'], r['chosen']] for o, r in per.items()),
                                        key=lambda x: -abs(x[2]))[:5])
        if a.cpi and 'Phi' in locals_cpi:
            st['cpi'], Wall = cpi_check(locals_cpi['R'][good], locals_cpi['Phi'][good], Y['td'][good], g[good], a.ridge)
            st['cpi_mc'] = cpi_check(locals_cpi['R'][good], locals_cpi['Phi'][good], Y['mc'][good], g[good], a.ridge)[0]
            fits[p] = dict(basis=locals_cpi['basis'], W=Wall.tolist(), t_oof=st['cpi']['t_oof'],
                           t_fold_mean=st['cpi']['t_fold_mean'], t_folds=st['cpi']['t_folds'],
                           slope=st['cpi']['slope'], floor=fl, pcs=a.pcs, ridge=a.ridge)
            if a.sham:
                st['cpi_sham'] = sham_check(locals_cpi['P'][good], locals_cpi['U'][good], locals_cpi['Phi'][good],
                                            Y['td'][good], g[good], a.ridge, a.sham, 1000 + p, st['cpi']['t_oof'])
        out['points'][p] = st
        f = lambda r: 'n/a' if r is None else f"{r['tau']:+.4f} t {r['t']:+.2f}"  # noqa: E731
        print(f"p{p:<2} used {st['used']:8d} dev {st['deviations']:6d} ({st['deviation_rate']}) parity "
              f"{st['parity_median']:.1e} | td {f(st['td'])} | mc {f(st['mc'])} | pre {f(st['pre'])} | "
              f"by round td {', '.join(f(st[f'td_r{lo}']) for lo in (0, 500, 1000))} | options |t|>2 "
              f"{st['options']['t_gt2']}/{st['options']['tested']} (chance {st['options']['chance_t_gt2']})" + (f" | cpi {st['cpi']} | cpi_mc t_oof {st['cpi_mc']['t_oof']}" if 'cpi' in st else '')
              + (f" | sham {({k_: v_ for k_, v_ in st['cpi_sham'].items() if k_ != 't'})}" if 'cpi_sham' in st else ''), flush=True)
    ta = [abs(out['points'][p]['td']['t']) for p in PHASE_A if p in out['points'] and out['points'][p]['td']]
    out['phase_a_median_abs_t_td'] = float(np.median(ta)) if ta else None
    out['seconds'] = round(time.time() - t0, 1)
    print(f"Phase-A median |t_td| = {out['phase_a_median_abs_t_td']} -> stage B proposals by "
          f"{'CPI' if (out['phase_a_median_abs_t_td'] or 0) >= 1 else 'low-rank ES'} ({out['seconds']} s)")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    if a.save_fit:
        Path(a.save_fit).write_text(json.dumps(dict(run=str(rd), iters=a.iters, window=a.window, horizon=a.horizon,
                                                    critic=a.critic, points=fits)))


if __name__ == '__main__':
    main()
