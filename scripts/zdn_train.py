"""Imitation training for zdn (docs/zdn.md): one net per unit type (baby, king), trained on teacher records.

  zdn_train.py --games 'runs/teacher-zdn1/games/*' --unit baby --out runs/zdn/net-b1 [--epochs 8] [--smooth 0.1]
               [--max-turns 600000] [--keep-frac 1.0] [--val-every 10] [--device cuda]

Net (mirrored by java/bots/zdn/ZNet.java):
  trunk  h = relu(W2 relu(W1 [S; L]* + b1) + b2)              (* standardised; the standardisation is folded in)
  scorer for one head (the step head, and one per register), over its K options:
         u   = A h + C ctx + a                                 (ctx: the step context; for register r, one-hots of
                                                                the option kinds chosen for registers 0..r-1)
         v_c = B[kind_c] x_c* + E[kind_c]                      (x_c*: option c's features, standardised per kind)
         logit_c = w . relu(u + v_c) + w0
  softmax over the record's options (padding masked).
Loss: cross-entropy with label smoothing over the record's options; records whose label matched no option are
skipped. Validation: every --val-every'th game (by sorted path) is held out. --keep-frac keeps a uniform random
fraction of each file's turns (with their steps): more games for the same memory than --max-turns, which stops after
the first files.
Writes OUT/net.json (weights with the standardisation folded in, plus dims and kind counts), OUT/norm.npz (that
standardisation, for RL: zdn_rl.Norm), OUT/train.log lines.
"""
import argparse, glob, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zdn_data  # noqa: E402


def low_fragmentation_malloc():
    """glibc: serve every allocation over 64 KB by mmap (returned to the OS when freed) instead of letting the dynamic
    threshold climb to 32 MB, where the per-file arrays of load() fragment the heap (peak RSS 975 -> 722 MB on 31k
    baby turns). No-op elsewhere."""
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6')
        libc.mallopt(-3, 1 << 16)    # M_MMAP_THRESHOLD
        libc.mallopt(-1, 0)          # M_TRIM_THRESHOLD
    except (OSError, AttributeError):
        pass


def load(paths, unit, max_turns, keep_frac=1.0, rng=None):
    """Concatenate one unit's records over files, keeping each turn (and its steps) with probability keep_frac.
    Returns (turn arrays, step arrays, register arrays)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    T = dict(S=[], L=[], game=[])
    R = None
    St = dict(turn=[], step=[], label=[], ctx=[], kind=[], x=[])
    nturn, nk_step = 0, 0
    for gi, p in enumerate(paths):
        try:
            d = zdn_data.read(p, kinds=(unit,))[unit]
        except Exception as e:   # a truncated file from a crashed game
            print(f'skip {p}: {e}', flush=True)
            continue
        r, s = d['reg'], d['step']
        if r is None or s is None:
            continue
        if keep_frac < 1:
            keep = np.nonzero(rng.random(len(r['robot'])) < keep_frac)[0]
            r = dict(r, regs=[{k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in g.items()} for g in r['regs']],
                     **{k: v[keep] for k, v in r.items() if k != 'regs' and isinstance(v, np.ndarray)})
        key = {(int(a), int(b)): nturn + i for i, (a, b) in enumerate(zip(r['robot'], r['round']))}
        T['S'].append(r['S'].astype(np.int16)); T['L'].append(r['L'].astype(np.uint8)); T['game'].append(np.full(len(r['robot']), gi, np.int32))
        if R is None:
            R = [dict(kind=[], x=[], lab=[], nk=g['nk']) for g in r['regs']]
        for rr, g in zip(R, r['regs']):
            rr['kind'].append(g['kind'].astype(np.int16)); rr['x'].append(g['x'].astype(np.int16)); rr['lab'].append(g['lab'])
        ti = np.array([key.get((int(a), int(b)), -1) for a, b in zip(s['robot'], s['round'])], np.int64)
        ok = ti >= 0
        nk_step = s['nk']
        St['turn'].append(ti[ok]); St['step'].append(s['step'][ok]); St['label'].append(s['label'][ok])
        St['ctx'].append(s['ctx'][ok].astype(np.int16)); St['kind'].append(s['kind'][ok].astype(np.int16)); St['x'].append(s['x'][ok].astype(np.int16))
        nturn += len(r['robot'])
        if nturn >= max_turns:
            break
    T = {k: _cat(v) for k, v in T.items()}
    St = {k: (_padcat(v) if k in ('kind', 'x') else _cat(v)) for k, v in St.items()}
    St['nk'] = nk_step
    R = [{k: (v if k == 'nk' else _padcat(v) if k in ('kind', 'x') else _cat(v)) for k, v in rr.items()} for rr in (R or [])]
    return T, St, R


def reg_context(R):
    """Per register r: [N, sum NK] one-hots of the labelled kinds of registers 0..r-1 (zeros for r.. and unlabelled)."""
    off = np.cumsum([0] + [r['nk'] for r in R])
    N = len(R[0]['lab']) if R else 0
    full = np.zeros((N, off[-1]), np.int8)
    out = []
    for j, r in enumerate(R):
        out.append(full.copy())
        lab = r['lab']
        ok = lab >= 0
        k = r['kind'][np.arange(N), np.maximum(lab, 0)]
        ok &= (k >= 0) & (k < r['nk'])
        full[np.where(ok)[0], off[j] + k[ok]] = 1
    return out


def _cat(arrs):
    """np.concatenate along axis 0 into one preallocated array, freeing each piece once copied (arrs is emptied), so
    the peak is the result plus one piece rather than twice the data. None for no pieces."""
    if not arrs:
        return None
    out = np.empty((sum(len(a) for a in arrs),) + arrs[0].shape[1:], np.result_type(*arrs))
    o = 0
    for i in range(len(arrs)):
        a, arrs[i] = arrs[i], None
        out[o:o + len(a)] = a
        o += len(a)
    arrs.clear()
    return out


def _padcat(arrs):
    """_cat with axis 1 padded to the longest piece: -1 for 2-d pieces (option kinds), 0 for 3-d (option features)."""
    K = max(a.shape[1] for a in arrs)
    out = np.empty((sum(len(a) for a in arrs), K) + arrs[0].shape[2:], np.result_type(*arrs))
    o = 0
    for i in range(len(arrs)):
        a, arrs[i] = arrs[i], None
        n, k = a.shape[:2]
        out[o:o + n, :k] = a
        out[o:o + n, k:] = -1 if a.ndim == 2 else 0
        o += n
    arrs.clear()
    return out


def stats(X, axis=0, rows=None, chunk=1 << 22):
    """Per-column mean and sd over the rows of X (all, or the indices `rows`), accumulated in float64 over chunks of
    at most `chunk` elements: no float64 copy of X (the trunk input has over a thousand columns). Constant columns get
    sd 1."""
    assert axis == 0
    rows = np.arange(len(X)) if rows is None else np.asarray(rows)
    chunk = max(1, chunk // max(1, int(np.prod(X.shape[1:]))))      # rows per chunk
    N = max(len(rows), 1)
    s = np.zeros(X.shape[1:], np.float64)
    for i in range(0, len(rows), chunk):
        s += X[rows[i:i + chunk]].sum(0, dtype=np.float64)
    mu = s / N
    v = np.zeros_like(s)
    for i in range(0, len(rows), chunk):
        v += ((X[rows[i:i + chunk]].astype(np.float64) - mu) ** 2).sum(0)
    sd = np.sqrt(v / N)
    sd = np.where(sd < 1e-6, 1.0, sd)   # constant columns: centred, not scaled
    return mu.astype(np.float32), sd.astype(np.float32)


def kind_stats(kind, x, nk):
    D = x.shape[-1]
    mu, sd = np.zeros((nk, D), np.float32), np.ones((nk, D), np.float32)
    for k in range(nk):
        sel = kind == k
        if sel.sum() > 1:
            mu[k], sd[k] = stats(x[sel])
    return mu, sd


class Scorer(nn.Module):
    def __init__(self, H, nC, nK, D, Hh):
        super().__init__()
        self.A = nn.Linear(H, Hh)
        self.C = nn.Linear(nC, Hh, bias=False) if nC else None
        self.B = nn.Parameter(torch.randn(nK, Hh, D) / math.sqrt(max(D, 1)))
        self.E = nn.Parameter(torch.zeros(nK, Hh))
        self.w = nn.Linear(Hh, 1)
        self.nK = nK

    def forward(self, h, ctx, kind, x):
        u = self.A(h)
        if self.C is not None:
            u = u + self.C(ctx)
        k = kind.clamp(min=0)
        # B[k] materialises [batch, K, Hh, D]: only while that stays small (64 MB); else one matmul per kind
        v = torch.einsum('bkd,bkhd->bkh', x, self.B[k]) if x.shape[0] * x.shape[1] * self.B[0].numel() <= (1 << 24) \
            else self._v_loop(k, x)
        v = v + self.E[k]
        z = self.w(torch.relu(u[:, None, :] + v)).squeeze(-1)
        return z.masked_fill(kind < 0, -1e9)

    def _v_loop(self, k, x):
        v = torch.zeros(x.shape[0], x.shape[1], self.B.shape[1], device=x.device)
        for j in range(self.nK):
            sel = k == j
            if sel.any():
                v[sel] = x[sel] @ self.B[j].T
        return v


class ZNet(nn.Module):
    def __init__(self, nS, nL, nC, step_dims, reg_dims, H1=256, H=128, Hh=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(nS + nL, H1), nn.ReLU(), nn.Linear(H1, H), nn.ReLU())
        self.step = Scorer(H, nC, *step_dims, Hh)
        nCr = sum(nK for nK, D in reg_dims)
        self.regs = nn.ModuleList([Scorer(H, nCr, nK, D, Hh) for nK, D in reg_dims])

    def h(self, SL):
        return self.trunk(SL)


def smooth_ce(z, lab, eps):
    """Cross-entropy with label smoothing over the unmasked options (z masked with -1e9)."""
    logp = torch.log_softmax(z, -1)
    legal = (z > -1e8).float()
    nll = -logp.gather(1, lab[:, None]).squeeze(1)
    uni = -(logp * legal).sum(-1) / legal.sum(-1).clamp(min=1)
    return ((1 - eps) * nll + eps * uni).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', required=True, help="glob of game directories (each with decisions-*.bin.gz)")
    ap.add_argument('--unit', choices=['baby', 'king'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--smooth', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--max-turns', type=int, default=600000)
    ap.add_argument('--keep-frac', type=float, default=1.0, help='keep each turn with this probability')
    ap.add_argument('--val-every', type=int, default=10)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'train.log', 'a')
    say = lambda s: (print(s, flush=True), log.write(s + '\n'), log.flush())

    files = sorted(f for g in sorted(glob.glob(a.games)) for f in glob.glob(g + '/decisions-*.bin.gz'))
    t0 = time.time()
    low_fragmentation_malloc()
    T, St, R = load(files, a.unit, a.max_turns, a.keep_frac, np.random.default_rng(a.seed + 1))
    nT, nM = len(T['S']), len(St['turn'])
    say(f'{a.unit}: {len(files)} files, {nT} turns, {nM} steps, {len(R)} registers ({time.time() - t0:.0f} s)')

    game = T['game']
    val_turn = (game % a.val_every) == 0
    nS, nL, nC = T['S'].shape[1], T['L'].shape[1], St['ctx'].shape[1]
    nK_step = max(St['nk'], int(St['kind'].max()) + 1)
    D_step = St['x'].shape[2]
    reg_dims = [(max(r['nk'], int(r['kind'].max()) + 1), r['x'].shape[2]) for r in R]
    RC = reg_context(R)
    tr_t = ~val_turn
    (mu_S, sd_S), (mu_L, sd_L) = (stats(T[k], rows=np.nonzero(tr_t)[0]) for k in ('S', 'L'))
    mu_SL, sd_SL = np.concatenate([mu_S, mu_L]), np.concatenate([sd_S, sd_L])
    mu_c, sd_c = stats(St['ctx'])
    kst_step = kind_stats(St['kind'], St['x'], nK_step)
    kst_reg = [kind_stats(r['kind'], r['x'], nK) for r, (nK, D) in zip(R, reg_dims)]

    dev = torch.device(a.device)
    net = ZNet(nS, nL, nC, (nK_step, D_step), reg_dims).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    tS = lambda idx: torch.from_numpy((np.concatenate([T['S'][idx], T['L'][idx]], 1) - mu_SL) / sd_SL).float().to(dev)

    def kx(kind, x, ks):
        mu, sd = ks
        k = np.maximum(kind, 0)
        xs = (x - mu[k]) / sd[k]
        xs[kind < 0] = 0
        return torch.from_numpy(kind).long().to(dev), torch.from_numpy(xs.astype(np.float32)).to(dev)

    step_tr = np.where(~val_turn[St['turn']] & (St['label'] >= 0))[0]
    step_va = np.where(val_turn[St['turn']] & (St['label'] >= 0))[0]
    say(f'step records: train {len(step_tr)}, val {len(step_va)}, unlabelled {(St["label"] < 0).mean():.4f}')
    turn_tr, turn_va = np.where(~val_turn)[0], np.where(val_turn)[0]

    def step_loss(idx, train):
        h = net.h(tS(St['turn'][idx]))
        ctx = torch.from_numpy((St['ctx'][idx] - mu_c) / sd_c).float().to(dev)
        kind, x = kx(St['kind'][idx], St['x'][idx].astype(np.float32), kst_step)
        z = net.step(h, ctx, kind, x)
        lab = torch.from_numpy(St['label'][idx]).long().to(dev)
        loss = smooth_ce(z, lab, a.smooth if train else 0.0)
        acc = (z.argmax(-1) == lab).float().mean()
        return loss, acc, z, lab

    def reg_loss(idx, train):
        h = net.h(tS(idx))
        losses, accs = [], []
        for j, (r, sc) in enumerate(zip(R, net.regs)):
            ok = r['lab'][idx] >= 0
            if not ok.any():
                continue
            ii = idx[ok]
            kind, x = kx(r['kind'][ii], r['x'][ii].astype(np.float32), kst_reg[j])
            rc = torch.from_numpy(RC[j][ii].astype(np.float32)).to(dev)
            z = sc(h[torch.from_numpy(ok).to(dev)], rc, kind, x)
            lab = torch.from_numpy(r['lab'][ii]).long().to(dev)
            losses.append(smooth_ce(z, lab, a.smooth if train else 0.0))
            accs.append((z.argmax(-1) == lab).float().mean())
        return (sum(losses) if losses else torch.zeros((), device=dev)), accs

    def evaluate():
        net.eval()
        with torch.no_grad():
            ls, ac, n = 0.0, 0.0, 0
            kinds_ok = np.zeros(nK_step); kinds_n = np.zeros(nK_step)
            for i in range(0, len(step_va), 4096):
                idx = step_va[i:i + 4096]
                l, acc, z, lab = step_loss(idx, False)
                ls += float(l) * len(idx); ac += float(acc) * len(idx); n += len(idx)
                k = St['kind'][idx, St['label'][idx]]
                hit = (z.argmax(-1) == lab).cpu().numpy()
                np.add.at(kinds_n, k, 1); np.add.at(kinds_ok, k, hit)
            racc = [0.0] * len(R); rn = 0
            for i in range(0, len(turn_va), 4096):
                idx = turn_va[i:i + 4096]
                _, accs = reg_loss(idx, False)
                for j, v in enumerate(accs):
                    racc[j] += float(v) * len(idx)
                rn += len(idx)
        net.train()
        per_kind = {int(k): round(float(kinds_ok[k] / kinds_n[k]), 3) for k in range(nK_step) if kinds_n[k] >= 20}
        return ls / max(n, 1), ac / max(n, 1), [round(v / max(rn, 1), 3) for v in racc], per_kind

    for ep in range(a.epochs):
        rng.shuffle(step_tr); rng.shuffle(turn_tr)
        nb = math.ceil(len(step_tr) / a.batch)
        tb = max(1, math.ceil(len(turn_tr) / nb))
        for b in range(nb):
            sidx = step_tr[b * a.batch:(b + 1) * a.batch]
            tidx = turn_tr[(b * tb) % max(len(turn_tr), 1):][:tb]
            l1, _, _, _ = step_loss(sidx, True)
            l2, _ = reg_loss(tidx, True) if len(tidx) else (torch.zeros((), device=dev), [])
            loss = l1 + l2
            opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups:
            g['lr'] *= 0.7
        vl, va, ra, pk = evaluate()
        say(f'epoch {ep + 1}: val step loss {vl:.4f} acc {va:.4f}; register acc {ra}; step acc by chosen kind {pk}')

    export(net, out / 'net.json', mu_SL, sd_SL, mu_c, sd_c, kst_step, kst_reg, dict(unit=a.unit, nS=nS, nL=nL, nC=nC))
    save_norm(out / 'norm.npz', mu_SL, sd_SL, mu_c, sd_c, [kst_step] + kst_reg)
    say(f'wrote {out / "net.json"}, {out / "norm.npz"}')


def save_norm(path, mu_SL, sd_SL, mu_c, sd_c, kst):
    """The standardisation the net was trained with (zdn_rl.Norm's file: RL keeps training in the same scale)."""
    d = dict(mu_SL=mu_SL, sd_SL=sd_SL, mu_c=mu_c, sd_c=sd_c, nhead=np.int64(len(kst)))
    for i, (mu, sd) in enumerate(kst):
        d[f'kmu{i}'], d[f'ksd{i}'] = mu, sd
    np.savez(path, **d)


def export(net, path, mu_SL, sd_SL, mu_c, sd_c, kst_step, kst_reg, meta):
    """Fold every standardisation into the weights; ZNet.java reads this layout."""
    P = lambda t: t.detach().cpu().double().numpy()
    W1, b1 = P(net.trunk[0].weight), P(net.trunk[0].bias)
    W1f = W1 / sd_SL[None, :]
    b1f = b1 - W1f @ mu_SL
    W2, b2 = P(net.trunk[2].weight), P(net.trunk[2].bias)

    def scorer(sc, ks, ctx_norm):
        A, a = P(sc.A.weight), P(sc.A.bias)
        d = dict(A=A.tolist(), a=a.tolist(), w=P(sc.w.weight)[0].tolist(), w0=float(P(sc.w.bias)[0]))
        if sc.C is not None and ctx_norm is None:   # register context: raw one-hots
            d['C'] = P(sc.C.weight).tolist()
        elif sc.C is not None:
            C = P(sc.C.weight) / ctx_norm[1][None, :]
            d['C'] = C.tolist()
            d['a'] = (a - C @ ctx_norm[0]).tolist()
        B, E = P(sc.B), P(sc.E)
        mu, sd = ks
        Bf = B / sd[:, None, :]
        Ef = E - np.einsum('khd,kd->kh', Bf, mu)
        d['B'] = Bf.tolist(); d['E'] = Ef.tolist()
        return d

    net_json = dict(meta=meta, W1=W1f.tolist(), b1=b1f.tolist(), W2=W2.tolist(), b2=b2.tolist(),
                    step=scorer(net.step, kst_step, (mu_c, sd_c)),
                    regs=[scorer(sc, ks, None) for sc, ks in zip(net.regs, kst_reg)])
    Path(path).write_text(json.dumps(net_json))


if __name__ == '__main__':
    main()
