"""Learned team-state critic and GAE / TD(lambda) advantages for rl_train.py (docs/rl-plan.md section 2, Phase B
variance reducers; docs/rl-plan.md "Critic and GAE" has the offline evaluation).

The critic V(s_t) predicts the team's outcome z from the team trace up to the record's 10-round bin t:
  * per side (us, them; the same function for both): babies, kings, cheese/1000, xfer/1000, log1p(cheese), log1p(xfer),
    xfer_d50/100, traps, the change over the last 1 / 5 / 20 bins of babies, kings, log1p(cheese), log1p(xfer),
    babies and kings lost since their maximum so far, and those maxima;
  * shared: round/2000, (round/2000)^2, map_area/3600;
  * seat: +1 for team A, -1 for team B.
The model is antisymmetric: logit = f(us, them, shared, seat) - f(them, us, shared, -seat), so in mirror games
V_A = -V_B exactly and V = 0 at an even state. kind 'rich' is a (64, 64) MLP f; 'richlin' is f linear in
[us, us*r, us*r^2] (a logistic regression on the antisymmetrised features).

Cross-fitting: rows of the current iteration are predicted out of fold (their games split into --critic-folds groups,
each predicted by a model trained on every other row of the window), so a game's own outcome never leaks into its
baseline. Those out-of-fold predictions also give the honest held-out Brier / AUC that metrics() reports.

GAE (gamma = 1) per team-game, over its trace bins in round order:
  delta_t = V(s_{t+1}) - V(s_t), delta_T = z - V(s_T) (last bin),  A_t = sum_l lambda^l delta_{t+l}.
lambda = 1 gives A_t = z - V(s_t), the original estimator. A record gets the A of its bin (the latest trace row with
round <= its round; the first row before that), i.e. the change in V from the state it acted in onwards.
"""
import numpy as np

SIDE = 6            # per-side raw trace columns: babies, kings, cheese, xfer, xfer_d50, traps (rl_train.TRACE_COLS)
LAGS = (1, 5, 20)   # bins
ROUND_COL, AREA_COL = 18, 19
BINS = ((0, 200), (200, 500), (500, 1000), (1000, 1500), (1500, 2001))


# ---------------------------------------------------------------------------------------------------- sequences

def seq_index(rows):
    """(order, seg, first): rows sorted by (team-game, round); seg = team-game number per sorted row; first = the sorted
    index of each row's sequence start."""
    key = rows['game'].astype(np.int64) * 2 + rows['team'].astype(np.int64)
    order = np.lexsort((rows['round'], key))
    ks = key[order]
    start = np.r_[True, ks[1:] != ks[:-1]]
    seg = np.cumsum(start) - 1
    first = np.nonzero(start)[0][seg]
    return order, seg, first


def _lag(a, first, L):
    idx = np.maximum(np.arange(len(a)) - L, first)
    return a[idx]


def _cummax(a, seg):
    big = float(np.abs(a).max() + 1) * 4
    return np.maximum.accumulate(a + seg * big) - seg * big


def _side(c, seg, first):
    """Features of one side from its 6 raw columns (rows sorted by sequence)."""
    b, k, ch, x, xd, tr = (c[:, i].astype(np.float64) for i in range(SIDE))
    lc, lx = np.log1p(np.maximum(ch, 0)), np.log1p(np.maximum(x, 0))
    f = [b, k, ch / 1000, x / 1000, lc, lx, xd / 100, tr]
    for L in LAGS:
        for v in (b, k, lc, lx):
            f.append(v - _lag(v, first, L))
    mb, mk = _cummax(b, seg), _cummax(k, seg)
    f += [mb - b, mk - k, mb, mk]
    return np.stack(f, 1)


def features(rows):
    """(U, Th, S, seat) in the rows' own order: per-side features of us and them, shared features, seat (+1 A, -1 B)."""
    order, seg, first = seq_index(rows)
    T = rows['trace'][order].astype(np.float64)
    U = _side(T[:, :SIDE], seg, first)
    Th = _side(T[:, SIDE:2 * SIDE], seg, first)
    r = T[:, ROUND_COL]
    S = np.stack([r, r * r, T[:, AREA_COL]], 1)
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    seat = 1.0 - 2.0 * rows['team'].astype(np.float64)
    return U[inv], Th[inv], S[inv], seat


# ---------------------------------------------------------------------------------------------------- models

def _model(kind, cu, cs, hidden):
    import torch, torch.nn as nn

    class Anti(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == 'rich':
                self.f = nn.Sequential(nn.Linear(2 * cu + cs + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, 1))
            else:   # richlin: w . [u, u r, u r^2] + w_s seat, antisymmetrised -> w . (phi(us) - phi(them)) + 2 w_s seat
                self.f = nn.Linear(3 * cu + 1, 1, bias=False)

        def g(self, U, Th, S, seat):
            if kind == 'rich':
                return self.f(torch.cat([U, Th, S, seat[:, None]], 1)).squeeze(1)
            r = S[:, :1]
            return self.f(torch.cat([U, U * r, U * r * r, seat[:, None]], 1)).squeeze(1)

        def forward(self, U, Th, S, seat):
            return self.g(U, Th, S, seat) - self.g(Th, U, S, -seat)
    return Anti()


class Critic:
    """One fitted antisymmetric critic (kind 'rich' or 'richlin'); predict() returns V in [-1, 1]. With games given,
    a game-level inner split (val_frac of the games) picks the best epoch and fits a logit temperature on it, so the
    MLP does not get overconfident on a few hundred outcomes."""

    def __init__(self, kind='rich', hidden=32, epochs=20, lr=3e-3, wd=1e-3, batch=2048, seed=0, val_frac=0.2):
        self.kind, self.hidden, self.epochs, self.lr, self.wd, self.batch, self.seed = kind, hidden, epochs, lr, wd, batch, seed
        self.val_frac = val_frac
        self.temp = 1.0

    def fit(self, F, y, games=None):
        import copy, torch
        U, Th, S, seat = F
        both = np.concatenate([U, Th])
        self.mu = both.mean(0); sd = both.std(0); self.sd = np.where(sd < 1e-9, 1.0, sd)
        self.smu = S.mean(0); ssd = S.std(0); self.ssd = np.where(ssd < 1e-9, 1.0, ssd)
        rng = np.random.default_rng(self.seed)
        val = np.zeros(len(y), bool)
        if games is not None and self.val_frac > 0:
            ug = np.unique(games)
            if len(ug) >= 10:
                val = np.isin(games, rng.choice(ug, max(2, int(round(len(ug) * self.val_frac))), replace=False))
        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        self.m = _model(self.kind, U.shape[1], S.shape[1], self.hidden)
        tU, tT, tS, ts = self._t(F)
        ty = torch.tensor(y, dtype=torch.float32)
        tr_idx = torch.tensor(np.nonzero(~val)[0]); va_idx = torch.tensor(np.nonzero(val)[0])
        opt = torch.optim.Adam(self.m.parameters(), lr=self.lr, weight_decay=self.wd)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        n = len(tr_idx)
        best, best_state, self.best_epoch = np.inf, None, self.epochs
        for ep in range(self.epochs):
            perm = tr_idx[torch.randperm(n, generator=g)]
            for s in range(0, n, self.batch):
                i = perm[s:s + self.batch]
                loss = bce(self.m(tU[i], tT[i], tS[i], ts[i]), ty[i])
                opt.zero_grad(); loss.backward(); opt.step()
            if len(va_idx):
                with torch.no_grad():
                    vl = float(bce(self.m(tU[va_idx], tT[va_idx], tS[va_idx], ts[va_idx]), ty[va_idx]))
                if vl < best:
                    best, best_state, self.best_epoch = vl, copy.deepcopy(self.m.state_dict()), ep + 1
        if best_state is not None:
            self.m.load_state_dict(best_state)
            with torch.no_grad():
                lv = self.m(tU[va_idx], tT[va_idx], tS[va_idx], ts[va_idx]).double().numpy()
            self.temp = _temperature(lv, y[val])
        return self

    def _t(self, F):
        import torch
        U, Th, S, seat = F
        f = lambda a: torch.tensor(a, dtype=torch.float32)
        return f((U - self.mu) / self.sd), f((Th - self.mu) / self.sd), f((S - self.smu) / self.ssd), f(seat)

    def predict(self, F):
        import torch
        with torch.no_grad():
            lg = self.m(*self._t(F)).double().numpy()
        return 2 / (1 + np.exp(-np.clip(lg * self.temp, -40, 40))) - 1


def _temperature(logit, y):
    """Scale a in [0, 1.5] minimising the log loss of sigmoid(a * logit) (golden-section search; a = 0 is V = 0)."""
    def nll(a):
        p = np.clip(1 / (1 + np.exp(-np.clip(a * logit, -40, 40))), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    lo, hi = 0.0, 1.5
    r = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        a, b = hi - r * (hi - lo), lo + r * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


def _take(F, idx):
    return tuple(a[idx] for a in F)


def fit_crossed(rows, target_mask, kind='rich', folds=4, seed=0, fit_rest=False, **kw):
    """V for rows in target_mask out of fold (their games split into `folds` groups; each group predicted by a model
    trained on all other rows); with fit_rest, the other rows in sample by a model fitted on all rows, else NaN. Rows
    are team-game bins (rl_train.value_rows). Returns (v, oof) with oof marking the out-of-fold predictions."""
    F = features(rows)
    y = (rows['z'].astype(np.float64) + 1) / 2
    games = rows['game'].astype(np.int64)
    v = np.full(len(y), np.nan)
    oof = np.zeros(len(y), bool)
    tg = np.unique(games[target_mask])
    rng = np.random.default_rng(seed)
    grp = dict(zip(rng.permutation(tg).tolist(), (np.arange(len(tg)) % max(1, folds)).tolist()))
    gid = np.array([grp.get(int(g), -1) for g in games])
    if len(tg) >= 2 and folds >= 2:
        for f in range(folds):
            te = gid == f
            if not te.any():
                continue
            c = Critic(kind, seed=seed + f, **kw).fit(_take(F, ~te), y[~te], games[~te])
            v[te] = c.predict(_take(F, te)); oof[te] = True
    need = ~oof if fit_rest else (target_mask & ~oof)   # rest, or target rows left in sample (too few games)
    if need.any():
        c = Critic(kind, seed=seed + 99, **kw).fit(F, y, games)
        v[need] = c.predict(_take(F, need))
    return v, oof


# ---------------------------------------------------------------------------------------------------- metrics

def auc(score, y):
    """ROC AUC of score for binary y (ties averaged); None without both classes."""
    pos = y == 1
    npos, nneg = int(pos.sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(score, kind='mergesort')
    s = score[order]
    ranks = np.empty(len(s))
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    ranks[order] = (cum - (cnt - 1) / 2)[inv]
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(v, rows, sel=None):
    """Brier (on p = (v+1)/2 vs y = (z+1)/2), the constant-0.5 Brier, and AUC on decided rows, overall and per round
    bin (BINS), over rows[sel]."""
    sel = np.ones(len(v), bool) if sel is None else sel
    p, y, r = (v[sel] + 1) / 2, (rows['z'][sel].astype(np.float64) + 1) / 2, rows['round'][sel]
    out = dict(rows=int(sel.sum()), games=int(len(np.unique(rows['game'][sel]))))

    def one(m):
        if not m.any():
            return None
        dec = m & (y != 0.5)
        a = auc(p[dec], y[dec])
        return dict(n=int(m.sum()), brier=round(float(((p[m] - y[m]) ** 2).mean()), 4),
                    brier_half=round(float(((0.5 - y[m]) ** 2).mean()), 4), auc=None if a is None else round(a, 3))
    out['all'] = one(np.ones(len(p), bool))
    for lo, hi in BINS:
        out[f'r{lo}'] = one((r >= lo) & (r < hi))
    return out


# ---------------------------------------------------------------------------------------------------- GAE

def gae(rows, v, lam):
    """GAE(gamma = 1, lambda) advantage per row (team-game bin): delta_t = v_{t+1} - v_t, delta_last = z - v_last."""
    order, seg, first = seq_index(rows)
    vs, zs = v[order].astype(np.float64), rows['z'][order].astype(np.float64)
    last = np.r_[seg[1:] != seg[:-1], True]
    nxt = np.where(last, zs, np.r_[vs[1:], 0.0])
    delta = nxt - vs
    A = np.empty(len(vs))
    acc = 0.0
    for i in range(len(vs) - 1, -1, -1):
        acc = delta[i] + (0.0 if last[i] else lam * acc)
        A[i] = acc
    out = np.empty_like(A); out[order] = A
    return out


def event_gae(rows, v, ri, key, lam):
    """Semi-MDP GAE (gamma = 1, lambda) over one point's own decision times, for committed decisions.

    ri: each record's bin (record_rows; -1 = none, A = 0). key: the sequence each record belongs to (team-game, or
    team-game and robot). A sequence's events are its distinct bins in round order (records of one sequence sharing a
    bin share one event and one A). delta_i = v(event i+1) - v(event i), and z - v(last event); A_i = sum_j lambda^(j-i)
    delta_j. lambda = 1 gives z - V(bin), as gae(lambda = 1) does; lambda = 0 gives the change in V over the decision's
    commitment (to the sequence's next decision, or to the outcome)."""
    ri = np.asarray(ri, np.int64); key = np.asarray(key, np.int64)
    A = np.zeros(len(ri))
    ok = ri >= 0
    if not ok.any():
        return A
    k, b = key[ok], ri[ok]
    rr = rows['round'].astype(np.int64)[b]
    order = np.lexsort((b, rr, k))
    ks, bs = k[order], b[order]
    new = np.r_[True, (ks[1:] != ks[:-1]) | (bs[1:] != bs[:-1])]
    ev = np.cumsum(new) - 1                       # event number of each sorted record
    ek, eb = ks[new], bs[new]
    vs = v[eb].astype(np.float64); zs = rows['z'][eb].astype(np.float64)
    last = np.r_[ek[1:] != ek[:-1], True]
    delta = np.where(last, zs, np.r_[vs[1:], 0.0]) - vs
    Ae = np.empty(len(vs)); acc = 0.0
    for i in range(len(vs) - 1, -1, -1):
        acc = delta[i] + (0.0 if last[i] else lam * acc)
        Ae[i] = acc
    Ao = np.empty(len(order)); Ao[order] = Ae[ev]
    A[ok] = Ao
    return A


def record_rows(rows, game, team, rnd):
    """Index into rows (team-game bins) of each record's bin: the latest row of its team-game with round <= rnd, the
    first row before that; -1 when its team-game has no rows."""
    key = rows['game'].astype(np.int64) * 2 + rows['team'].astype(np.int64)
    comb = key * 8192 + rows['round'].astype(np.int64)
    order = np.argsort(comb, kind='mergesort')
    cs = comb[order]
    rk = game.astype(np.int64) * 2 + team.astype(np.int64)
    pos = np.searchsorted(cs, rk * 8192 + rnd.astype(np.int64), side='right') - 1
    ok = (pos >= 0) & (cs[np.maximum(pos, 0)] // 8192 == rk)
    nxt = np.minimum(pos + 1, len(cs) - 1)
    early = ~ok & (cs[nxt] // 8192 == rk)          # before the sequence's first row
    pos = np.where(early, nxt, pos)
    ok |= early
    return np.where(ok, order[np.clip(pos, 0, len(cs) - 1)], -1)
