"""Read-only query on cpi-x0 iteration-0 records: why the PRE/POST/FACE argmax falls back. Prints only."""
import sys, json, glob
import numpy as np
sys.path.insert(0, '/home/terry/rltest/bc26a0/scripts')
import dpnet_io, decisions

R = '/home/terry/rltest/bc26a0/runs/rl/cpi-x0'
nets = dpnet_io.load_net(R + '/nets/0/net.json')
A = (['PASS', 'DROP_ALLY', 'THROW', 'NAP', 'BITE_KING', 'BITE_RAT', 'BITE_CAT', 'CAT_TRAP', 'LAY_LANE', 'DELIVER',
      'CASH_KING', 'RAT_TRAP', 'X_BITE_FWD', 'X_CARRY_FWD', 'X_THROW_NOW', 'X_DROP_FWD', 'X_RTRAP_FWD', 'X_CTRAP_FWD',
      'X_DPLACE_FWD', 'X_DREM_FWD', 'X_KING'] + [f'X_BITE_C{i}' for i in range(6)] + [f'X_DROP_D{i}' for i in range(8)]
     + ['X_XFER10', 'X_XFER50', 'X_XFER_HALF', 'X_DISINT'])
F = ['FNONE', 'SEEN', 'MEMORY', 'HEARD', 'GUARD'] + [f'DIR{i}' for i in range(8)]
FN = ['post', 'role', 'raw', 'all10', 'coop', 'r10', 'hp', 'hostile', 'movR', 'turnR', 'moved', 'why', 'carry', 'cage',
      'canThrow', 'foes', 'friends', 'eAdj', 'eAtk', 'eCarry', 'eHold', 'minHp', 'kAdj', 'kAtk', 'kHp', 'cat', 'catD4',
      'cAdj', 'cAtk', 'kingD', 'kingCatD', 'nCatT', 'nRatT', 'budget', 'lane', 'give', 'cash', 'onCash', 'canKing', 'ne',
      'neD', 'hunting', 'fd', 'front', 'rtF', 'ctF', 'dpF', 'drF', 'dirt', 'x', 'y', 'W', 'H', 'fhp', 'dropm', 'atkF',
      'kd', 'trap']


def top_of(p, z):
    x = z['x'].astype(np.float32)
    L = nets[p]
    n_out = L[-1]['w'].shape[0]
    lg = dpnet_io.forward(L, x, exact=False)
    m = dpnet_io.fit_mask(z['mask'], len(z['k']), n_out)
    lg = np.where(m, lg, -np.inf)
    return lg.argmax(1), x


def hist(a, n=12):
    v, c = np.unique(a, return_counts=True)
    o = np.argsort(-c)[:n]
    return ', '.join(f'{int(v[i])}:{int(c[i])}' for i in o)


for p, names in ((2, A), (3, A)):
    z = np.load(f'{R}/iter-0000/data/p{p}.npz')
    top, x = top_of(p, z)
    k, fb = z['k'], (z['flags'] & 1).astype(bool)
    N = len(k)
    print(f'== point {p}: rows {N}; argmax hist', ', '.join(f'{names[v]}:{c}' for v, c in zip(*np.unique(top, return_counts=True))))
    carry = x[:, 12]
    print(f'  carry hist all rows: {hist(carry)}; role hist: {hist(x[:, 1])}; coop: {hist(x[:, 4])}; hostile: {hist(x[:, 7])}')
    for t in (2, 3, 5, 6, 7, 9):
        s = top == t
        if not s.any():
            continue
        tk = s & (k == t)
        print(f'  argmax {names[t]}: {int(s.sum())} rows ({s.mean():.4f}); sampled==argmax {int(tk.sum())}, fb {fb[tk].mean() if tk.any() else float("nan"):.3f}')
        print(f'    carry {hist(carry[s])} | canThrow {hist(x[s, 14])} | cage {hist(x[s, 13], 10)} | hostile {hist(x[s, 7])} | coop {hist(x[s, 4])}')
        print(f'    role {hist(x[s, 1])} | eAdj {hist(x[s, 17])} | eAtk {hist(x[s, 18])} | eCarry {hist(x[s, 19])} | cat {hist(x[s, 25])} cAdj {hist(x[s, 27])}')
        if t == 2:
            for cv in (0, 1, 2):
                c = tk & (carry == cv)
                if c.any():
                    print(f'    carry={cv}: n {int(c.sum())} fb {fb[c].mean():.3f}; canThrow=1 share {(x[c, 14] == 1).mean():.3f}; '
                          f'fb|canThrow=1 {fb[c & (x[:, 14] == 1)].mean() if (c & (x[:, 14] == 1)).any() else float("nan"):.3f}; '
                          f'fb|canThrow=0 {fb[c & (x[:, 14] == 0)].mean() if (c & (x[:, 14] == 0)).any() else float("nan"):.3f}')
            # concentration
            g = z['game'][s].astype(np.int64) * 4 + z['team'][s]
            ug, cg = np.unique(g, return_counts=True)
            cg = np.sort(cg)[::-1]
            print(f'    game-sides with argmax THROW: {len(ug)} of {len(np.unique(z["game"].astype(np.int64) * 4 + z["team"]))}; '
                  f'top-10 share {cg[:10].sum() / cg.sum():.3f}; top counts {cg[:10].tolist()}')
            rid = (z['game'][s].astype(np.int64) * 4 + z['team'][s]) * 100000 + z['robot'][s]
            ur, cr = np.unique(rid, return_counts=True)
            print(f'    distinct robots {len(ur)}; per-robot count quantiles 50/90/99/max {np.percentile(cr, [50, 90, 99]).tolist()} {cr.max()}')
            # all-carry=2 rows: how often is THROW the argmax
            c2 = carry == 2
            print(f'    rows with carry=2: {int(c2.sum())}; argmax THROW share {np.mean(top[c2] == 2):.3f}; argmax hist {hist(top[c2])}')
            c0 = s & (carry == 0)
            if c0.any():
                print('    carry=0 & argmax THROW feature means vs all rows:')
                for j, nm in enumerate(FN):
                    a, b = x[c0, j].mean(), x[:, j].mean()
                    if abs(a - b) > 0.25 * (x[:, j].std() + 1e-6):
                        print(f'      {nm}: {a:.2f} vs {b:.2f} (sd {x[:, j].std():.2f})')
                print(f'    carry=0 & THROW rounds q10/50/90 {np.percentile(z["round"][c0], [10, 50, 90]).tolist()}; all {np.percentile(z["round"], [10, 50, 90]).tolist()}')
                gg = z['game'][c0]
                print(f'    carry=0 & THROW games: {len(np.unique(gg))}, top-5 games {np.sort(np.unique(gg, return_counts=True)[1])[::-1][:5].tolist()}')

# FACE
z = np.load(f'{R}/iter-0000/data/p4.npz')
top, x = top_of(4, z)
k, fb = z['k'], (z['flags'] & 1).astype(bool)
print('== point 4 argmax hist', ', '.join(f'{F[v]}:{c}' for v, c in zip(*np.unique(top, return_counts=True))))
for t in range(13):
    s = (top == t) & (k == t)
    if s.any():
        print(f'  argmax==sampled {F[t]}: n {int(s.sum())} fb {fb[s].mean():.4f}')

# labels from raw decision files of 24 games: what ran instead when the pick fell back
nopt = {p: ls[-1]['w'].shape[0] for p, ls in nets.items()}
files = sorted(glob.glob(f'{R}/iter-0000/games/j00[0-2]?/decisions-*.bin.gz'))[:48]
pairs = {2: {}, 3: {}, 4: {}}
tot = {2: 0, 3: 0, 4: 0}
for f in files:
    d = decisions.read_rl(f, nopt)
    for p in (2, 3, 4):
        if p not in d:
            continue
        q = d[p]
        tot[p] += len(q['k'])
        fbm = (q['flags'] & 1).astype(bool)
        for kk, ll in zip(q['k'][fbm].tolist(), q['label'][fbm].tolist()):
            pairs[p][(kk, ll)] = pairs[p].get((kk, ll), 0) + 1
print(f'== raw files {len(files)}')
for p in (2, 3, 4):
    names = F if p == 4 else A
    items = sorted(pairs[p].items(), key=lambda t: -t[1])[:25]
    nf = sum(pairs[p].values())
    print(f'  point {p}: records {tot[p]}, fallback {nf}; (picked -> label) ' +
          ', '.join(f'{names[a] if a < len(names) else a}->{names[b] if b < len(names) else b}:{c}' for (a, b), c in items))
