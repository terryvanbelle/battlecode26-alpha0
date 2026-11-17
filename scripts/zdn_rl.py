"""zdn RL utilities (docs/zdn.md, "RL"): a zdn net.json as a torch module, the standardisation used for training, and
the decision table of recorded turns.

  net_from_json(doc)   zdn_train.ZNet holding the folded weights: raw inputs give ZNet.java's logits
  Norm                 the training standardisation (fit on a table, save / load as npz); unfold(net, norm) and
                       export(net, norm, path, meta) are exact inverses (the function is unchanged)
  load_table(paths)    every sampled register choice and primitive step of one unit's recorded turns, one row each
  head_logits          masked logits of a batch of rows (all of one head); turn_rows lists a set of turns' rows

Decision table (dict of numpy arrays) of one unit (baby or king):
  turns, N:   SL [N, nS + nL] int16 (raw S, then L), robot, round, version (-1 none), src (index into paths),
              rkind [N, nR] int16 (kind of the chosen option per register, -1 none), d0 [N + 1] int64 (row ranges)
  rows, M:    turn int32, head int8 (-1 step, r register r), step int16, label int16 (-1 none), q int32 (-1 none),
              n int16 (options), o0 int64 (first option), ctx [M, nC] int16 (the step context; zeros for registers)
  options, P: kind int16, x [P, Dmax] int16 (each head's D features, zero-padded to the widest head's)
  meta:       nS, nL, nC, nR, nk [1 + nR] (step first, the records' NK), D [1 + nR]
A turn's rows are d0[t] .. d0[t+1] - 1: its registers in order, then its steps in order.
"""
import copy
import numpy as np
import torch

import zdn_data
import zdn_train

LOGP_SCALE = 4096.0   # q = round(-4096 log p) (ZE.q)


# ---------------------------------------------------------------------------------------------------- nets

def net_from_json(doc):
    """zdn_train.ZNet with a net.json's folded weights (raw inputs in, ZNet.java's logits out)."""
    m = doc['meta']
    W1, W2 = np.asarray(doc['W1'], np.float32), np.asarray(doc['W2'], np.float32)
    H1, H = W1.shape[0], W2.shape[0]
    st = doc['step']
    Hh = len(st['a'])
    dims = lambda sc: (len(sc['B']), np.asarray(sc['B'], np.float32).reshape(len(sc['B']), Hh, -1).shape[2])
    net = zdn_train.ZNet(m['nS'], m['nL'], m['nC'], dims(st), [dims(r) for r in doc['regs']], H1=H1, H=H, Hh=Hh)
    t = lambda v: torch.as_tensor(np.asarray(v, np.float32))
    with torch.no_grad():
        net.trunk[0].weight.copy_(t(W1)); net.trunk[0].bias.copy_(t(doc['b1']))
        net.trunk[2].weight.copy_(t(W2)); net.trunk[2].bias.copy_(t(doc['b2']))
        for sc, d in [(net.step, st)] + list(zip(net.regs, doc['regs'])):
            sc.A.weight.copy_(t(d['A'])); sc.A.bias.copy_(t(d['a']))
            if sc.C is not None:
                if 'C' in d:
                    sc.C.weight.copy_(t(d['C']))
                else:
                    sc.C.weight.zero_()
            sc.B.copy_(t(d['B']).reshape(sc.B.shape)); sc.E.copy_(t(d['E']))
            sc.w.weight.copy_(t(d['w'])[None, :]); sc.w.bias.copy_(t([d['w0']]))
    return net


def head_dims(net):
    """[(nK, D)] per head, step first."""
    return [(sc.B.shape[0], sc.B.shape[2]) for sc in [net.step] + list(net.regs)]


class Norm:
    """Training standardisation: SL (trunk input), c (step context), kst[i] = (mu [nK, D], sd [nK, D]) per head i
    (0 = step, 1 + r = register r). Registers' context is raw one-hots (as in zdn_train)."""

    def __init__(self, mu_SL, sd_SL, mu_c, sd_c, kst):
        self.mu_SL, self.sd_SL, self.mu_c, self.sd_c, self.kst = mu_SL, sd_SL, mu_c, sd_c, kst

    @classmethod
    def identity(cls, net):
        nIn = net.trunk[0].weight.shape[1]
        nC = net.step.C.weight.shape[1] if net.step.C is not None else 0
        f = lambda n: (np.zeros(n, np.float32), np.ones(n, np.float32))
        return cls(*f(nIn), *f(nC), [(np.zeros((k, d), np.float32), np.ones((k, d), np.float32)) for k, d in head_dims(net)])

    @classmethod
    def fit(cls, tab, net, max_rows=50000, seed=0):
        """Statistics of a decision table (at most max_rows sampled rows per head), for a net's head sizes."""
        rng = np.random.default_rng(seed)
        mu_SL, sd_SL = zdn_train.stats(tab['SL'])
        steps = tab['head'] < 0
        mu_c, sd_c = zdn_train.stats(tab['ctx'][steps]) if steps.any() else (np.zeros(tab['ctx'].shape[1], np.float32),
                                                                              np.ones(tab['ctx'].shape[1], np.float32))
        kst = []
        for i, (nK, D) in enumerate(head_dims(net)):
            rows = np.nonzero(tab['head'] == i - 1)[0]
            if len(rows) > max_rows:
                rows = np.sort(rng.choice(rows, max_rows, replace=False))
            kind, x = gather(tab, rows)
            kst.append(zdn_train.kind_stats(kind, x[..., :D], nK))
        return cls(mu_SL, sd_SL, mu_c, sd_c, kst)

    def save(self, path):
        zdn_train.save_norm(path, self.mu_SL, self.sd_SL, self.mu_c, self.sd_c, self.kst)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z['mu_SL'], z['sd_SL'], z['mu_c'], z['sd_c'], [(z[f'kmu{i}'], z[f'ksd{i}']) for i in range(int(z['nhead']))])


def unfold(raw, norm):
    """A copy of a folded net (net_from_json) taking norm-standardised inputs instead of raw ones (same function)."""
    net = copy.deepcopy(raw)
    t = lambda v: torch.as_tensor(np.asarray(v, np.float32))
    with torch.no_grad():
        W1f = raw.trunk[0].weight.detach().double()
        net.trunk[0].weight.copy_((W1f * t(norm.sd_SL).double()[None, :]).float())
        net.trunk[0].bias.copy_((raw.trunk[0].bias.double() + W1f @ t(norm.mu_SL).double()).float())
        if raw.step.C is not None:
            Cf = raw.step.C.weight.detach().double()
            net.step.C.weight.copy_((Cf * t(norm.sd_c).double()[None, :]).float())
            net.step.A.bias.copy_((raw.step.A.bias.double() + Cf @ t(norm.mu_c).double()).float())
        for sc_raw, sc, (mu, sd) in zip([raw.step] + list(raw.regs), [net.step] + list(net.regs), norm.kst):
            Bf = sc_raw.B.detach().double()
            sc.B.copy_((Bf * t(sd).double()[:, None, :]).float())
            sc.E.copy_((sc_raw.E.double() + torch.einsum('khd,kd->kh', Bf, t(mu).double())).float())
    return net


def export(net, norm, path, meta):
    """Fold norm into the weights and write net.json (zdn_train.export)."""
    zdn_train.export(net, path, norm.mu_SL, norm.sd_SL, norm.mu_c, norm.sd_c, norm.kst[0], norm.kst[1:], meta)


# ---------------------------------------------------------------------------------------------------- the table

def _flat(kind, x, Dmax):
    """Ragged options of padded per-row arrays: (n [R], kind_flat, x_flat [P, Dmax])."""
    ok = kind >= 0
    n = ok.sum(1).astype(np.int16)
    xf = x[ok]
    if xf.shape[1] < Dmax:
        xf = np.pad(xf, ((0, 0), (0, Dmax - xf.shape[1])))
    return n, kind[ok].astype(np.int16), xf.astype(np.int16)


def load_table(paths, unit, files=None):
    """The decision table of one unit over decisions files (see the module doc). files: the paths' indices to store in
    turns' src column (default 0..len-1). Unreadable files are skipped with a message."""
    files = list(range(len(paths))) if files is None else list(files)
    parts, meta = [], None
    for fi, p in zip(files, paths):
        try:
            d = zdn_data.read(p, kinds=(unit,))[unit]
        except Exception as e:   # a truncated file from a crashed game
            print(f'skip {p}: {e}', flush=True)
            continue
        r, s = d['reg'], d['step']
        if r is None or not len(r['robot']):
            continue
        m = dict(nS=r['S'].shape[1], nL=r['L'].shape[1], nC=0 if s is None else s['ctx'].shape[1], nR=len(r['regs']),
                 nk=[0 if s is None else s['nk']] + [g['nk'] for g in r['regs']],
                 D=[0 if s is None else s['x'].shape[2]] + [g['x'].shape[2] for g in r['regs']])
        if meta is None:
            meta = m
        else:
            for k in ('nS', 'nL', 'nR'):
                if m[k] != meta[k]:
                    raise SystemExit(f'{p}: {k} {m[k]} differs from {meta[k]}')
            meta['nC'] = max(meta['nC'], m['nC'])
            meta['nk'] = [max(a, b) for a, b in zip(meta['nk'], m['nk'])]
            meta['D'] = [max(a, b) for a, b in zip(meta['D'], m['D'])]
        parts.append((fi, r, s))
    if meta is None:
        return None
    Dmax = max(meta['D'])
    nR, nC = meta['nR'], meta['nC']
    T = dict(SL=[], robot=[], round=[], version=[], src=[], rkind=[])
    Rw = dict(turn=[], head=[], step=[], label=[], q=[], n=[], o0=[], ctx=[], order=[])
    O = dict(kind=[], x=[])
    nt, npo = 0, 0
    for fi, r, s in parts:
        N = len(r['robot'])
        T['SL'].append(np.concatenate([r['S'], r['L']], 1).astype(np.int16))
        T['robot'].append(r['robot']); T['round'].append(r['round']); T['version'].append(r['version'])
        T['src'].append(np.full(N, fi, np.int32))
        rk = np.full((N, nR), -1, np.int16)
        for j, g in enumerate(r['regs']):
            lab = g['lab']
            rk[:, j] = np.where(lab >= 0, g['kind'][np.arange(N), np.maximum(lab, 0)], -1)
            n, kf, xf = _flat(g['kind'], g['x'], Dmax)
            o0 = npo + np.concatenate([[0], np.cumsum(n[:-1], dtype=np.int64)])
            npo += int(n.sum())
            O['kind'].append(kf); O['x'].append(xf)
            Rw['turn'].append(nt + np.arange(N, dtype=np.int64)); Rw['head'].append(np.full(N, j, np.int8))
            Rw['step'].append(np.zeros(N, np.int16)); Rw['label'].append(lab.astype(np.int16)); Rw['q'].append(g['q'])
            Rw['n'].append(n); Rw['o0'].append(o0); Rw['ctx'].append(np.zeros((N, nC), np.int16))
            Rw['order'].append(np.full(N, j, np.int32))
        T['rkind'].append(rk)
        if s is not None and len(s['robot']):
            key = {(int(a), int(b)): nt + i for i, (a, b) in enumerate(zip(r['robot'], r['round']))}
            ti = np.array([key.get((int(a), int(b)), -1) for a, b in zip(s['robot'], s['round'])], np.int64)
            ok = ti >= 0
            if (~ok).any():
                print(f'file {fi}: {int((~ok).sum())} step records without their turn record dropped', flush=True)
            M = int(ok.sum())
            n, kf, xf = _flat(s['kind'][ok], s['x'][ok], Dmax)
            o0 = npo + np.concatenate([[0], np.cumsum(n[:-1], dtype=np.int64)])[:M]
            npo += int(n.sum())
            O['kind'].append(kf); O['x'].append(xf)
            c = s['ctx'][ok].astype(np.int16)
            if c.shape[1] < nC:
                c = np.pad(c, ((0, 0), (0, nC - c.shape[1])))
            Rw['turn'].append(ti[ok]); Rw['head'].append(np.full(M, -1, np.int8)); Rw['step'].append(s['step'][ok].astype(np.int16))
            Rw['label'].append(s['label'][ok].astype(np.int16)); Rw['q'].append(s['q'][ok]); Rw['n'].append(n)
            Rw['o0'].append(o0); Rw['ctx'].append(c); Rw['order'].append(nR + s['step'][ok].astype(np.int32))
        nt += N
    tab = {k: np.concatenate(v) for k, v in T.items()}
    rows = {k: np.concatenate(v) for k, v in Rw.items()}
    order = np.lexsort((rows.pop('order'), rows['turn']))
    rows = {k: v[order] for k, v in rows.items()}
    rows['turn'] = rows['turn'].astype(np.int32)
    rows['q'] = rows['q'].astype(np.int32)
    tab.update(rows)
    tab['kind'] = np.concatenate(O['kind']); tab['x'] = np.concatenate(O['x'])
    tab['d0'] = np.searchsorted(tab['turn'], np.arange(nt + 1)).astype(np.int64)
    tab['meta'] = meta
    return tab


def subset(tab, turns):
    """The table of a subset of turns (sorted indices); options are re-packed."""
    turns = np.asarray(turns, np.int64)
    rows = turn_rows(tab, turns)
    out = {k: tab[k][turns] for k in ('SL', 'robot', 'round', 'version', 'src', 'rkind')}
    for k in ('head', 'step', 'label', 'q', 'n', 'ctx'):
        out[k] = tab[k][rows]
    remap = np.full(len(tab['SL']), -1, np.int64); remap[turns] = np.arange(len(turns))
    out['turn'] = remap[tab['turn'][rows]].astype(np.int32)
    kind, x = gather(tab, rows)
    ok = kind >= 0
    out['kind'], out['x'] = kind[ok], x[ok]
    out['o0'] = np.concatenate([[0], np.cumsum(out['n'][:-1], dtype=np.int64)]).astype(np.int64) if len(rows) else np.zeros(0, np.int64)
    out['d0'] = np.searchsorted(out['turn'], np.arange(len(turns) + 1)).astype(np.int64)
    out['meta'] = tab['meta']
    return out


def turn_rows(tab, turns):
    """Row indices of the given turns (in turn order)."""
    a, b = tab['d0'][turns], tab['d0'][np.asarray(turns) + 1]
    if not len(a):
        return np.zeros(0, np.int64)
    n = b - a
    return np.repeat(a - np.concatenate([[0], np.cumsum(n[:-1])]), n) + np.arange(n.sum())


def gather(tab, rows):
    """Padded options of rows: kind [B, K] (-1 pad), x [B, K, Dmax]."""
    n = tab['n'][rows].astype(np.int64)
    K = int(n.max()) if len(n) else 1
    ar = np.arange(K)[None, :]
    valid = ar < n[:, None]
    idx = np.where(valid, tab['o0'][rows][:, None] + ar, 0)
    if not len(tab['kind']):
        return np.full((len(rows), K), -1, np.int16), np.zeros((len(rows), K, tab['x'].shape[1]), np.int16)
    kind = np.where(valid, tab['kind'][idx], -1).astype(np.int16)
    x = tab['x'][idx] * valid[..., None]
    return kind, x


# ---------------------------------------------------------------------------------------------------- logits

def trunk_input(norm, SL, dev):
    return torch.from_numpy(((SL - norm.mu_SL) / norm.sd_SL).astype(np.float32)).to(dev)


def reg_ctx(tab, rows, head, nk_net):
    """Register context of rows (all of register `head`): one-hots of the kinds chosen for registers before it."""
    off = np.concatenate([[0], np.cumsum(nk_net)])
    c = np.zeros((len(rows), off[-1]), np.float32)
    rk = tab['rkind'][tab['turn'][rows]]
    for j in range(head):
        k = rk[:, j].astype(np.int64)
        ok = (k >= 0) & (k < nk_net[j])
        c[np.nonzero(ok)[0], off[j] + k[ok]] = 1
    return c


def head_logits(net, norm, tab, rows, head, h, dev):
    """Logits [B, K] of rows (all of head: -1 step, r register), the unknown kinds (>= the net's nK) and padding masked
    to -1e9, as ZNet.java (-inf). h: the trunk output of each row's turn ([B, H])."""
    sc = net.step if head < 0 else net.regs[head]
    nK, D = sc.B.shape[0], sc.B.shape[2]
    kind, x = gather(tab, rows)
    kind = np.where(kind >= nK, -1, kind).astype(np.int64)
    mu, sd = norm.kst[head + 1]
    k0 = np.maximum(kind, 0)
    xs = (x[..., :D] - mu[k0]) / sd[k0]
    xs[kind < 0] = 0
    if head < 0:
        ctx = (tab['ctx'][rows] - norm.mu_c) / norm.sd_c
    else:
        ctx = reg_ctx(tab, rows, head, [r.B.shape[0] for r in net.regs])
    ctx = torch.from_numpy(np.asarray(ctx, np.float32)).to(dev)
    return sc(h, ctx, torch.from_numpy(kind).to(dev), torch.from_numpy(xs.astype(np.float32)).to(dev))


def heads(tab, rows):
    """{head: positions within rows} for the heads present."""
    hd = tab['head'][rows]
    return {int(k): np.nonzero(hd == k)[0] for k in np.unique(hd)}


def logp_rows(net, norm, tab, turns, T, dev, rows=None):
    """(rows, logp of the recorded label [len(rows)], log-softmax [len(rows), K] per head as {head: (pos, lsm)}) for a set
    of turns, at temperature T. Rows without a label get logp 0."""
    turns = np.asarray(turns, np.int64)
    rows = turn_rows(tab, turns) if rows is None else rows
    pos_t = np.full(len(tab['SL']), -1, np.int64); pos_t[turns] = np.arange(len(turns))
    H = net.h(trunk_input(norm, tab['SL'][turns], dev))
    lp = torch.zeros(len(rows), device=dev)
    full = {}
    for head, pos in heads(tab, rows).items():
        r = rows[pos]
        z = head_logits(net, norm, tab, r, head, H[torch.from_numpy(pos_t[tab['turn'][r]]).to(dev)], dev)
        lsm = torch.log_softmax(z / T, -1)
        lab = torch.from_numpy(np.maximum(tab['label'][r], 0).astype(np.int64)).to(dev)
        v = lsm.gather(1, lab[:, None]).squeeze(1)
        lp = lp.index_put((torch.from_numpy(pos).to(dev),), torch.where(torch.from_numpy(tab['label'][r] >= 0).to(dev), v, torch.zeros_like(v)))
        full[head] = (pos, lsm)
    return rows, lp, full
