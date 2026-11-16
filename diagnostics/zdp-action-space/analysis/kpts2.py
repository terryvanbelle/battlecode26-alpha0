"""Read-only follow-ups for KING_SPAWN / KING_ACT / KING_FORM: label drift, trap build-up and formation outcomes,
for one data dir (argv[1])."""
import sys
import numpy as np
D = sys.argv[1]
NONE = 0xFFFF


def load(p):
    z = np.load(f'{D}/p{p}.npz')
    k = z['k'].astype(np.int64); fl = z['flags'].astype(np.int64)
    keep = (k != NONE) & ((fl & 2) == 0)
    out = {n: z[n][keep] for n in ('x', 'k', 'flags', 'game', 'team', 'round', 'robot')}
    out['k'] = out['k'].astype(np.int64); out['fb'] = (out['flags'] & 1) != 0
    out['x'] = out['x'].astype(np.int64)
    out['tg'] = out['game'].astype(np.int64) * 2 + out['team'].astype(np.int64)
    return out


def pct(a):
    return f'{100 * float(np.mean(a)):.1f}%' if len(a) else 'n/a'


print('DATA', D)
d = load(11); x, k, fb = d['x'], d['k'], d['fb']
ntg = len(np.unique(d['tg']))
nosur = (x[:, 9] == 0) & (x[:, 10] == 0) & (x[:, 14] < 32768)
flo = ~nosur & (x[:, 15] < 32768)
capd = ~nosur & ~flo & (x[:, 6] >= x[:, 13])
zsp = ~nosur & ~flo & ~capd
m = flo & (k == 2)
print('P11 team-games', ntg, '| floor-state records picking S_CAP:', int(m.sum()), 'of which babies>=cap', pct(x[m, 6] >= x[m, 13]))
m = flo & (k == 1)
print('   floor-state records picking S_FLOOR:', int(m.sum()), 'of which babies>=cap', pct(x[m, 6] >= x[m, 13]))
bld = (k >= 3) & (k <= 18)
print('   zdb-spawn states per team-game', round(float(zsp.sum()) / ntg, 1), '; executed builds per team-game', round(float((bld & ~fb).sum()) / ntg, 1),
      '; executed builds in non-spawn states per team-game', round(float((bld & ~fb & ~zsp).sum()) / ntg, 2),
      '; no-spawn picks in zdb-spawn states per team-game', round(float((zsp & ((k <= 2) | (k >= 20))).sum()) / ntg, 2))

d = load(12); x, k, fb = d['x'], d['k'], d['fb']
ntg = len(np.unique(d['tg']))
rtr = x[:, 16]; r = d['round'].astype(np.int64)
print('P12 team-games', ntg, '| executed per team-game:', {nm: round(float(((k == kk) & ~fb).sum()) / ntg, 2) for kk, nm in ((4, 'TRAP_FOE'), (5, 'TRAP_FORT'), (6, 'MOAT'), (9, 'CAT_TRAP'), (10, 'DIG'))})
for lo, hi in ((0, 200), (200, 500), (500, 1000), (1000, 1500), (1500, 2001)):
    mm = (r >= lo) & (r < hi)
    if mm.any(): print(f'   rounds {lo}-{hi}: mean team rat traps {rtr[mm].mean():.2f}, share>=10 {pct(rtr[mm] >= 10)}, mean ring dirt-placeable {x[mm, 19].mean():.2f}, ring dirt-removable {x[mm, 20].mean():.2f}')

d = load(13); x, k, fb = d['x'], d['k'], d['fb']
ntg = len(np.unique(d['tg']))
order = np.lexsort((d['round'], d['robot'], d['tg']))
k2, fb2, x2, tg2, rb2 = k[order], fb[order], x[order], d['tg'][order], d['robot'][order]
nk2, op2 = x2[:, 2], x2[:, 3]
opn = (k2 >= 13) & (k2 <= 29) & ~fb2
print('P13 team-games', ntg, '| executed opens per team-game', round(float(opn.sum()) / ntg, 2))
H = 15


def rise(j):
    e = j + 1
    while e < len(k2) and e <= j + H and tg2[e] == tg2[j] and rb2[e] == rb2[j]:
        if nk2[e] > nk2[j]: return 1
        e += 1
    return 0


one = (nk2 == 1)
idx = np.nonzero(opn & one)[0]
base = np.nonzero(one & (op2 == 0) & ~((k2 >= 13) & (k2 <= 29)))[0]
rng = np.random.default_rng(0)
bs = rng.choice(base, min(len(base), 20000), replace=False) if len(base) else base
print(f'   1-king states: net opens {len(idx)}, a new king within {H} turns after an open: {sum(rise(j) for j in idx)}; '
      f'baseline (random 1-king record, no site, no open pick) rate {np.mean([rise(j) for j in bs]) if len(bs) else float("nan"):.4f}')
for kk in range(13, 30):
    j = np.nonzero(opn & one & (k2 == kk))[0]
    if len(j): print(f'     option {kk}: opens {len(j)}, new king within {H}: {sum(rise(t) for t in j)}, median own babies in sight {int(np.median(x2[j, 25]))}, median round {int(np.median(x2[j, 1]))}')
# zdb's own opens (F_START_* labels picked as argmax-like in gate-pass state) for reference
zd = np.nonzero(one & (k2 >= 13) & (k2 <= 17) & ~fb2 & (x2[:, 15] == 1))[0]
print(f'   opens in gate-pass 1-king states (zdb would open): {len(zd)}, new king within {H}: {sum(rise(j) for j in zd)}')
