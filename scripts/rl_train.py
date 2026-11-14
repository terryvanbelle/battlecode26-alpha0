"""One PPO-lite iteration over the zdp decision nets (docs/rl-plan.md sections 2-3): fit V, compute advantages, update
each trainable point's net with a KL leash to the anchor (zdpn2), export the next net.json.

rl_train.py --net runs/rl/R/nets/3/net.json --anchor runs/dp/net-v2/net.json --data runs/rl/R/iter-0003/data \\
    --value-data runs/rl/R/iter-000{1,2,3}/data --out runs/rl/R/nets/4 [--points 5,7,10,11,12,13] [--floor 0.005]
    [--beta 0.03] [--beta-state runs/rl/R/beta.json] [--lr 3e-4] [--epochs 2] [--batch 4096]
rl_train.py ... --lr 0                      identity: the exported net.json (and so W.java) is byte-identical
rl_train.py ... --points 6 --fake-reward 6:0 --no-kl     planted reward: A = 1 if k == 0 else 0 on point 6

Inputs. --net is the learner theta_k (the net that played the data); its fold normalisation comes from norm.npz beside
it (train_dp.py / rl_train.py write one), else from the data (any mu/sd unfolds exactly). Each --data / --value-data
directory holds, written by rl_data.py, only the learner's recorded sides:

  p<point>.npz, one row per decision record (records with k none may be present; the trainer drops them):
    x      uint16/int32 [N, nf]  the point's raw features (trailer stripped; padded/truncated to the net's n_in)
    k      int32 [N]             sampled option (0xFFFF: none)
    logq   float32 [N]           -q / 4096: log of the behaviour probability of k (the trailer's q)
    mask   bool [N, nOpt]        allowed options from the trailer (sized to the net's n_out like D.fit)
    flags  uint8/int32 [N]       trailer flags: bit0 fallback (zdb's cascade ran), bit1 secondary slot
    team   int32 [N]             0 = A, 1 = B
    round  int32 [N]             the record's round
    game   int64 [N]             game id, unique within the directory
    pair   int64 [N]             seat-swapped pair id, unique within the directory; -1 when not paired (mirror games)
    z      float32 [N]           outcome for the record's team: +1 win, -1 loss, 0 draw
    trace  float32 [N, Ftrace]   the team's trace at the record's 10-round bin (columns: TRACE_COLS below)
    version int32 [N] (optional) trailer NET_VERSION; with --version, other versions are dropped
  trace.npz (optional, recommended): one row per learner team-game and 10-round bin, keys game, team, round, pair, z,
    trace (as above). V is fitted on these rows; without the file they are the distinct (game, team, bin) rows of the
    p*.npz files.

TRACE_COLS (the writer's order; the trainer only needs the same Ftrace everywhere, it standardises each column):
  us_{babies, kings, cheese, xfer, xfer_d50, traps}, them_{same 6}, (us - them) of the 6, round / 2000, map_area / 3600.

Outputs in --out: net.json (frozen and unchanged points copied value-for-value, so W.java is byte-identical; the whole
file is copied byte-for-byte when nothing changed), norm.npz, metrics.json. With the collapse rule triggered (a point's
KL to the anchor > --max-anchor-kl) nothing but metrics.json is written and the exit code is 3.
"""
import argparse, json, math, shutil, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dpnet_io  # noqa: E402

NONE = 0xFFFF
TRACE_COLS = [f'{s}_{c}' for s in ('us', 'them', 'diff') for c in ('babies', 'kings', 'cheese', 'xfer', 'xfer_d50', 'traps')] \
    + ['round_2000', 'map_area_3600']
REFUSED = 3


# ---------------------------------------------------------------------------------------------------- data

def load_dirs(dirs, points, dir_index):
    """{point: dict of concatenated arrays} for the given points; game/pair ids made global as dir_index << 32 | id."""
    out = {}
    for p in points:
        parts = []
        for d in dirs:
            f = Path(d) / f'p{p}.npz'
            if not f.exists():
                continue
            z = np.load(f)
            part = {key: z[key] for key in z.files}
            g = dir_index[str(d)] << 32
            part['game'] = part['game'].astype(np.int64) + g
            part['pair'] = np.where(part['pair'] >= 0, part['pair'].astype(np.int64) + g, -1)
            parts.append(part)
        if not parts:
            continue
        nf = max(pt['x'].shape[1] for pt in parts)
        no = max(pt['mask'].shape[1] for pt in parts)
        d = {}
        for key in parts[0]:
            if key == 'x':
                d[key] = np.concatenate([np.pad(pt['x'], ((0, 0), (0, nf - pt['x'].shape[1]))) for pt in parts])
            elif key == 'mask':
                d[key] = np.concatenate([np.pad(pt['mask'], ((0, 0), (0, no - pt['mask'].shape[1]))) for pt in parts])
            elif all(key in pt for pt in parts):
                d[key] = np.concatenate([pt[key] for pt in parts])
        out[p] = d
    return out


def value_rows(dirs, dir_index):
    """V rows: dict(game, team, round, pair, z, trace) from trace.npz, or the distinct (game, team, bin) rows of p*.npz."""
    parts = []
    for d in dirs:
        g = dir_index[str(d)] << 32
        f = Path(d) / 'trace.npz'
        if f.exists():
            z = np.load(f)
            r = {k: z[k] for k in ('game', 'team', 'round', 'z', 'trace')}
            r['pair'] = z['pair'] if 'pair' in z.files else np.full(len(r['game']), -1)
        else:
            acc = {k: [] for k in ('game', 'team', 'round', 'z', 'trace', 'pair')}
            for f in sorted(Path(d).glob('p*.npz')):
                z = np.load(f)
                for k in acc:
                    acc[k].append(z[k])
            if not acc['game']:
                continue
            r = {k: np.concatenate(v) for k, v in acc.items()}
            r['round'] = r['round'] // 10 * 10
            key = np.stack([r['game'].astype(np.int64), r['team'].astype(np.int64), r['round'].astype(np.int64)], 1)
            _, first = np.unique(key, axis=0, return_index=True)
            r = {k: v[first] for k, v in r.items()}
        r['game'] = r['game'].astype(np.int64) + g
        r['pair'] = np.where(r['pair'] >= 0, r['pair'].astype(np.int64) + g, -1)
        parts.append(r)
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---------------------------------------------------------------------------------------------------- value model

def sigmoid(v):
    return 1 / (1 + np.exp(-np.clip(v, -40, 40)))


def auc(score, y):
    """ROC AUC of score for binary y (ties averaged); None without both classes."""
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]
    i = 0
    while i < len(s):   # average ranks of ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


class Value:
    """V(trace) = 2 sigmoid(w . [standardised trace, 1]) - 1, logistic regression on y = (z + 1) / 2 (IRLS, ridge)."""

    def __init__(self, rows, val_frac, seed, ridge=1e-2):
        self.w = None
        self.metrics = dict(rows=0)
        if rows is None or rows['trace'].ndim != 2 or rows['trace'].shape[1] == 0:
            return
        T, y, games = rows['trace'].astype(np.float64), (rows['z'].astype(np.float64) + 1) / 2, rows['game']
        ug = np.unique(games)
        if len(ug) < 2:
            return
        rng = np.random.default_rng(seed)
        val = np.isin(games, rng.choice(ug, max(1, int(round(len(ug) * val_frac))), replace=False))
        if val.all():
            val[:] = False
        self.mu = T[~val].mean(0); sd = T[~val].std(0); self.sd = np.where(sd < 1e-9, 1.0, sd)
        Phi = self._phi(T)
        A, yt = Phi[~val], y[~val]
        w = np.zeros(Phi.shape[1])
        reg = ridge * np.eye(len(w)); reg[-1, -1] = 0
        for _ in range(50):
            p = sigmoid(A @ w)
            g = A.T @ (p - yt) + reg @ w
            H = (A * (p * (1 - p) + 1e-9)[:, None]).T @ A + reg
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        pv, yv = sigmoid(Phi[val] @ w), y[val]
        m = dict(rows=int(len(y)), games=int(len(ug)), val_games=int(len(np.unique(games[val]))), ftrace=int(T.shape[1]),
                 train_brier=float(((sigmoid(A @ w) - yt) ** 2).mean()))
        if val.any():
            m['val_brier'] = float(((pv - yv) ** 2).mean())
            m['val_brier_const'] = float(((yt.mean() - yv) ** 2).mean())
            dec = yv != 0.5
            m['val_auc'] = auc(pv[dec], yv[dec])
            rv = rows['round'][val]
            for r in (500, 1000, 1500):
                s = dec & (np.abs(rv - r) < 50)
                m[f'val_auc_r{r}'] = auc(pv[s], yv[s]) if s.any() else None
        self.metrics = m

    def _phi(self, T):
        return np.concatenate([(T - self.mu) / self.sd, np.ones((len(T), 1))], 1)

    def __call__(self, trace):
        if self.w is None:
            return np.zeros(len(trace))
        return 2 * sigmoid(self._phi(trace.astype(np.float64)) @ self.w) - 1


def pair_baseline(vrows, V):
    """{(game, team): pair baseline}: the mean over the pair's learner team-games of mean_bins(z - V); only for pairs
    with at least two learner team-games."""
    if vrows is None:
        return {}
    res = vrows['z'].astype(np.float64) - V(vrows['trace'])
    key = vrows['game'] * 2 + vrows['team'].astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    rbar = np.bincount(inv, weights=res) / np.bincount(inv)
    kpair = np.full(len(uk), -1, np.int64); kpair[inv] = vrows['pair']
    out = {}
    for pr in np.unique(kpair[kpair >= 0]):
        sel = np.nonzero(kpair == pr)[0]
        if len(sel) >= 2:
            b = float(rbar[sel].mean())
            for i in sel:
                out[int(uk[i])] = b
    return out


# ---------------------------------------------------------------------------------------------------- policy

def build_model(layers64):
    import torch, torch.nn as nn
    mods = []
    for i, l in enumerate(layers64):
        lin = nn.Linear(l['w'].shape[1], l['w'].shape[0])
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(l['w'], dtype=torch.float32))
            lin.bias.copy_(torch.tensor(l['b'], dtype=torch.float32))
        mods.append(lin)
        if i < len(layers64) - 1:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


def policy_terms(model, X, mask, T, floor, chunk=1 << 16):
    """(probabilities, log probabilities with 0 on masked options) over all rows, no grad."""
    import torch
    ps, lps = [], []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            p = dpnet_io.probs(model(X[s:s + chunk]), mask[s:s + chunk], T, floor)
            ps.append(p); lps.append(torch.where(mask[s:s + chunk], torch.log(p.clamp_min(1e-30)), torch.zeros_like(p)))
    return torch.cat(ps), torch.cat(lps)


def kl_rows(p, lp, lq, mask):
    import torch
    return torch.where(mask, p * (lp - lq), torch.zeros_like(p)).sum(1)


def entropy_rows(p, lp, mask):
    import torch
    return -torch.where(mask, p * lp, torch.zeros_like(p)).sum(1)


def pg_grad(model, X, mask, k, logq, A, T, floor):
    """Flattened gradient of the unclipped surrogate mean(r A) (gradient-SNR diagnostic)."""
    import torch
    model.zero_grad()
    p = dpnet_io.probs(model(X), mask, T, floor)
    lp = torch.log(p.gather(1, k[:, None]).squeeze(1).clamp_min(1e-30))
    (-(torch.exp(lp - logq) * A).mean()).backward()
    g = torch.cat([q.grad.flatten() for q in model.parameters()])
    model.zero_grad()
    return g


def train_point(p, d, layers, norm, anchor_layers, A, a, beta, rng, metrics):
    """PPO-lite on one point. Returns the trained float32 layer list (or None if unchanged)."""
    import torch
    n_out = layers[-1]['w'].shape[0]
    n_in = layers[0]['w'].shape[1]
    N = len(d['k'])
    x = np.zeros((N, n_in), np.float32)
    m = min(n_in, d['x'].shape[1]); x[:, :m] = d['x'][:, :m]
    mask = dpnet_io.fit_mask(d['mask'], N, n_out)
    mu, sd = norm
    # behaviour (theta_k, exactly as the bot) and anchor probabilities from the folded nets
    pk_bot = dpnet_io.probs(dpnet_io.forward(layers, x), mask, a.temperature, a.floor)
    p_k = pk_bot[np.arange(N), d['k']]
    dlog = np.abs(np.log(np.maximum(p_k, 1e-30)) - d['logq'])
    soft_k = (p_k.astype(np.float64) - a.floor / np.maximum(mask.sum(1), 1)) / (1 - a.floor)   # softmax part, before the floor
    metrics.update(parity_median=float(np.median(dlog)), parity_p999=float(np.quantile(dlog, 0.999)),
                   parity_max=float(dlog.max()), sat_frac=float((soft_k > 0.999).mean()))
    if anchor_layers is not None:
        pa = dpnet_io.probs(dpnet_io.forward(anchor_layers, x), mask, a.temperature, a.floor)
        la = np.where(mask, np.log(np.maximum(pa, 1e-30)), 0).astype(np.float32)
    else:
        pa = la = None

    model = build_model([dict(w=l['w'].astype(np.float32), b=l['b'].astype(np.float32))
                         for l in dpnet_io.unfold(layers, mu, sd)])
    init = [q.detach().clone() for q in model.parameters()]
    tX = torch.tensor(((x - mu) / sd).astype(np.float32))
    tm = torch.tensor(mask)
    tk = torch.tensor(d['k'].astype(np.int64))
    tq = torch.tensor(d['logq'].astype(np.float32))
    tA = torch.tensor(A.astype(np.float32))
    tla = torch.tensor(la) if la is not None else None
    T, f = a.temperature, a.floor

    def summary(tag):
        pp, lp = policy_terms(model, tX, tm, T, f)
        out = {f'entropy_{tag}': float(entropy_rows(pp, lp, tm).mean()),
               f'mean_probs_{tag}': [round(v, 6) for v in pp.mean(0).tolist()]}
        if tla is not None:
            out[f'kl_anchor_{tag}'] = float(kl_rows(pp, lp, tla, tm).mean())
        return out, pp, lp

    s0, p_old, lp_old = summary('start')
    metrics.update(s0)
    lpk0 = lp_old.gather(1, tk[:, None]).squeeze(1)
    metrics['ratio_start_maxabslog'] = float((lpk0 - tq).abs().max())
    if pa is not None:
        metrics['entropy_anchor'] = float(-(np.where(mask, pa * la, 0)).sum(1).mean())
        metrics['mean_probs_anchor'] = [round(v, 6) for v in pa.mean(0).astype(np.float64).tolist()]

    if a.snr and N >= 64:
        games = np.unique(d['game'])
        half = np.isin(d['game'], rng.choice(games, len(games) // 2, replace=False)) if len(games) >= 2 else \
            rng.random(N) < 0.5
        cos = []
        for sel in (half, ~half):
            idx = np.nonzero(sel)[0]
            if len(idx) > 200000:
                idx = rng.choice(idx, 200000, replace=False)
            it = torch.tensor(idx)
            cos.append(pg_grad(model, tX[it], tm[it], tk[it], tq[it], tA[it], T, f))
        den = float(cos[0].norm() * cos[1].norm())
        metrics['grad_snr_cos'] = float(cos[0] @ cos[1]) / den if den > 0 else None

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps = clipped = seen = 0
    stopped = None
    for ep in range(a.epochs):
        perm = torch.tensor(rng.permutation(N))
        for s in range(0, N, a.batch):
            idx = perm[s:s + a.batch]
            mk = tm[idx]
            pp = dpnet_io.probs(model(tX[idx]), mk, T, f)
            lp = torch.where(mk, torch.log(pp.clamp_min(1e-30)), torch.zeros_like(pp))
            lpk = lp.gather(1, tk[idx][:, None]).squeeze(1)
            logr = lpk - tq[idx]
            r = torch.exp(logr)
            with torch.no_grad():
                approx_kl = float(((r - 1) - logr).mean())
            if approx_kl > a.target_kl:
                stopped = dict(epoch=ep, step=steps, approx_kl=approx_kl)
                break
            Ai = tA[idx]
            surr = torch.min(r * Ai, torch.clamp(r, 1 - a.clip, 1 + a.clip) * Ai)
            loss = -surr.mean() - a.c_ent * entropy_rows(pp, lp, mk).mean()
            if tla is not None and beta > 0:
                loss = loss + beta * kl_rows(pp, lp, tla[idx], mk).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            steps += 1
            with torch.no_grad():
                clipped += int(((r - 1).abs() > a.clip).sum()); seen += len(idx)
        if stopped:
            break
    metrics.update(steps=steps, clip_frac=clipped / seen if seen else 0.0, early_stop=stopped)
    s1, p_new, lp_new = summary('end')
    metrics.update(s1)
    metrics['kl_old_new'] = float(kl_rows(p_old, lp_old, lp_new, tm).mean())
    if a.fake_reward:
        fo = a.fake_reward[1]
        metrics['fake_option_prob_start'] = float(p_old[:, fo].mean()) if fo < n_out else None
        metrics['fake_option_prob_end'] = float(p_new[:, fo].mean()) if fo < n_out else None
    changed = any(not torch.equal(q0, q) for q0, q in zip(init, model.parameters()))
    metrics['changed'] = changed
    if not changed:
        return None, None
    trained = [dict(w=lin.weight.detach().double().numpy(), b=lin.bias.detach().double().numpy())
               for lin in model if isinstance(lin, torch.nn.Linear)]
    folded = dpnet_io.fold(trained, mu, sd)
    # export parity: the folded net as the bot will run it (float32) vs the trained model
    f32 = [dict(w=l['w'].astype(np.float32), b=l['b'].astype(np.float32)) for l in folded]
    pb = dpnet_io.probs(dpnet_io.forward(f32, x, exact=False), mask, T, f)
    pn = p_new.numpy()
    with np.errstate(divide='ignore'):
        dl = np.abs(np.log(np.maximum(pb, 1e-30)) - np.log(np.maximum(pn, 1e-30)))[mask]
    metrics['export_parity_max'] = float(dl.max()) if dl.size else 0.0
    return folded, p_new


# ---------------------------------------------------------------------------------------------------- main

def parse_points(s):
    return [int(v) for v in s.split(',') if v]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--net', required=True, help="the learner's net.json (theta_k, the net that played --data)")
    ap.add_argument('--norm', default='', help='norm.npz for --net (default: beside it; else from the data)')
    ap.add_argument('--anchor', default='', help='KL anchor net.json (zdpn2; default --net)')
    ap.add_argument('--data', nargs='+', required=True, help="this iteration's data dirs (p<point>.npz)")
    ap.add_argument('--value-data', nargs='*', default=None, help='data dirs for fitting V (default --data)')
    ap.add_argument('--out', required=True, help='output dir: net.json, norm.npz, metrics.json')
    ap.add_argument('--points', default='5,7,10,11,12,13', help='trainable points (others are copied unchanged)')
    ap.add_argument('--floor', type=float, default=0.005, help="the build's NET_FLOOR (behaviour and trained policy)")
    ap.add_argument('--floor-points', default='', help='per-point floors P:F,... (the build\'s W.FL; others --floor)')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--version', type=int, default=None, help='keep only records with this trailer NET_VERSION')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--target-kl', type=float, default=0.02, help='early stop when approx KL(theta_k -> theta) exceeds')
    ap.add_argument('--grad-clip', type=float, default=0.5)
    ap.add_argument('--beta', type=float, default=0.03, help='KL-to-anchor weight (times the beta-state multiplier)')
    ap.add_argument('--beta-state', default='', help='json {point: beta multiplier}, read and updated by the alarms')
    ap.add_argument('--no-kl', action='store_true', help='no KL leash (M1 only); also disables the collapse refusal')
    ap.add_argument('--c-ent', type=float, default=0.003)
    ap.add_argument('--collapse-kl', type=float, default=0.3, help='KL to anchor above this doubles beta_p')
    ap.add_argument('--max-anchor-kl', type=float, default=0.5, help='KL to anchor above this refuses the export')
    ap.add_argument('--adv-clip', type=float, default=5.0)
    ap.add_argument('--no-pair', action='store_true', help='no paired-game baseline')
    ap.add_argument('--no-value', action='store_true', help='no value baseline: V = 0, A = z (then normalised)')
    ap.add_argument('--fake-reward', default='', help='point:option planted test: A = 1 if k == option else 0')
    ap.add_argument('--max-rows', type=int, default=2_000_000, help='per-point uniform subsample cap')
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--no-snr', dest='snr', action='store_false')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    points = parse_points(a.points)
    fpts = {int(k): float(v) for k, v in (x.split(':') for x in a.floor_points.split(',') if x)}
    if a.fake_reward:
        fp, fo = (int(v) for v in a.fake_reward.split(':'))
        a.fake_reward = (fp, fo)
        points = [fp]   # the planted test trains its point only
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    raw = dpnet_io.load_raw(a.net)
    nets = {int(p): dpnet_io.layers_from_raw(v) for p, v in raw['points'].items()}
    norm = dpnet_io.load_norm(a.norm or a.net)
    anchor = dpnet_io.load_net(a.anchor) if a.anchor else nets
    beta_mult = {}
    if a.beta_state and Path(a.beta_state).exists():
        beta_mult = {int(k): float(v) for k, v in json.loads(Path(a.beta_state).read_text()).items()}

    vdirs = a.value_data if a.value_data else a.data
    dir_index = {str(d): i for i, d in enumerate(dict.fromkeys(list(a.data) + list(vdirs)))}
    data = load_dirs(a.data, [p for p in points if p in nets], dir_index)
    V = Value(None if a.no_value else value_rows(vdirs, dir_index), a.val_frac, a.seed)   # no rows: V = 0
    base = {} if a.no_pair else pair_baseline(value_rows(a.data, dir_index), V)
    print(f'value model: {V.metrics}', flush=True)
    metrics = dict(args={k: v for k, v in vars(a).items()}, value=V.metrics, points={}, pairs_with_baseline=len(base))

    new_layers, new_norm, refused = {}, dict(norm), []
    for p in points:
        if p not in nets or p not in data:
            print(f'point {p}: no net or no data, unchanged', flush=True)
            continue
        d = data[p]
        mt = dict(records=int(len(d['k'])))
        n_out = nets[p][-1]['w'].shape[0]
        keep = d['k'] != NONE
        if a.version is not None and 'version' in d:
            mt['version_mismatch'] = int((keep & (d['version'] != a.version)).sum())
            keep &= d['version'] == a.version
        maskf = dpnet_io.fit_mask(d['mask'], len(keep), n_out)
        kk = np.where(keep, d['k'], 0)
        bad = keep & ((kk >= n_out) | ~maskf[np.arange(len(kk)), np.minimum(kk, n_out - 1)])
        mt['k_not_allowed'] = int(bad.sum())
        keep &= ~bad
        mt['single_option'] = int((keep & (maskf.sum(1) <= 1)).sum())
        keep &= maskf.sum(1) > 1
        # diagnostics on all usable-k records: option histogram and fallback rate per option
        ks = d['k'][keep].astype(np.int64)
        fb = (d['flags'][keep].astype(np.int64) & 1)
        hist = np.bincount(ks, minlength=n_out)
        mt['option_hist'] = hist.tolist()
        mt['fallback_rate'] = [float(v) if c else None for v, c in
                               zip(np.bincount(ks, weights=fb, minlength=n_out) / np.maximum(hist, 1), hist)]
        idx = np.nonzero(keep)[0]
        if len(idx) > a.max_rows:
            idx = np.sort(rng.choice(idx, a.max_rows, replace=False))
        d = {k: v[idx] for k, v in d.items()}
        d['mask'] = maskf[idx]
        mt['used'] = int(len(idx))
        if len(idx) < 2:
            print(f'point {p}: {len(idx)} usable records, unchanged', flush=True)
            metrics['points'][p] = mt
            continue
        if a.fake_reward:
            A = (d['k'] == a.fake_reward[1]).astype(np.float64)
        else:
            A = d['z'].astype(np.float64) - V(d['trace'])
            if base:
                gk = d['game'] * 2 + d['team'].astype(np.int64)
                b = np.array([base.get(int(g), np.nan) for g in gk])
                mt['paired_frac'] = float(np.isfinite(b).mean())
                A = A - np.where(np.isfinite(b), b, 0.0)
            mt['adv_raw_mean'], mt['adv_raw_sd'] = float(A.mean()), float(A.std())
            A = np.clip((A - A.mean()) / (A.std() + 1e-8), -a.adv_clip, a.adv_clip)
        if p in norm:
            mu, sd = (np.asarray(v) for v in norm[p])
        else:
            mu = d['x'].astype(np.float32).mean(0); sd = d['x'].astype(np.float32).std(0)
            sd = np.where(sd < 1e-3, 1.0, sd).astype(np.float32)
            n_in = nets[p][0]['w'].shape[1]
            mu = np.pad(mu, (0, max(0, n_in - len(mu))))[:n_in]; sd = np.pad(sd, (0, max(0, n_in - len(sd))), constant_values=1)[:n_in]
            mt['norm_from_data'] = True
        beta = 0.0 if a.no_kl else a.beta * beta_mult.get(p, 1.0)
        mt['beta'] = beta
        tp = time.time()
        if anchor.get(p) is not None and anchor[p][-1]['w'].shape[0] != n_out:
            raise SystemExit(f'point {p}: the anchor has {anchor[p][-1]["w"].shape[0]} outputs, the net {n_out} '
                             '(widen the anchor with widen_dpnet.py too)')
        ap_ = argparse.Namespace(**{**vars(a), 'floor': fpts.get(p, a.floor)})   # this point's floor
        mt['floor'] = ap_.floor
        folded, _ = train_point(p, d, nets[p], (mu, sd), anchor.get(p), A, ap_, beta, rng, mt)
        mt['seconds'] = round(time.time() - tp, 1)
        new_norm[p] = (mu, sd)
        if folded is not None:
            new_layers[p] = folded
        # alarms (section 6) -> beta_p x2 next iteration; collapse refusal
        alarms = []
        kla = mt.get('kl_anchor_end')
        if kla is not None and kla > a.collapse_kl:
            alarms.append('kl_anchor')
        if 'entropy_anchor' in mt and mt['entropy_end'] < 0.5 * mt['entropy_anchor']:
            alarms.append('entropy')
        if 'mean_probs_anchor' in mt and any(pe > 0.95 and pa <= 0.70 for pe, pa in zip(mt['mean_probs_end'], mt['mean_probs_anchor'])):
            alarms.append('option_hist')
        mt['alarms'] = alarms
        if alarms and not a.no_kl:
            beta_mult[p] = beta_mult.get(p, 1.0) * 2
        if kla is not None and kla > a.max_anchor_kl and not a.no_kl:
            refused.append(p)
        metrics['points'][p] = mt
        print(f'point {p}: {mt["used"]}/{mt["records"]} used, steps {mt["steps"]}, early stop {mt["early_stop"]}, '
              f'KL anchor {mt.get("kl_anchor_start", 0):.4f} -> {mt.get("kl_anchor_end", 0):.4f}, entropy '
              f'{mt["entropy_start"]:.4f} -> {mt["entropy_end"]:.4f}, clip {mt["clip_frac"]:.3f}, parity median '
              f'{mt["parity_median"]:.2e}, changed {mt["changed"]}, alarms {alarms}', flush=True)

    metrics['refused'] = refused
    metrics['changed_points'] = sorted(new_layers)
    if a.beta_state:
        Path(a.beta_state).write_text(json.dumps({str(k): v for k, v in sorted(beta_mult.items())}))
    metrics['beta_mult'] = {str(k): v for k, v in sorted(beta_mult.items())}
    if refused:
        metrics['exported'] = False
        metrics['seconds'] = round(time.time() - t0, 1)
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=1))
        print(f'REFUSED: KL to anchor > {a.max_anchor_kl} on points {refused}; nothing exported')
        sys.exit(REFUSED)
    if new_layers:
        pts = {}
        for ps, v in raw['points'].items():
            pts[ps] = [dpnet_io.layer_json(l) for l in new_layers[int(ps)]] if int(ps) in new_layers else v
        doc = dict(raw); doc['points'] = pts
        (out / 'net.json').write_text(json.dumps(doc))
    else:
        shutil.copyfile(a.net, out / 'net.json')
    dpnet_io.save_norm(out / 'norm.npz', {p: v for p, v in new_norm.items()})
    metrics['exported'] = True
    metrics['seconds'] = round(time.time() - t0, 1)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1, default=str))
    print(f'wrote {out}/net.json (changed points {sorted(new_layers)}) in {metrics["seconds"]}s')


if __name__ == '__main__':
    main()
