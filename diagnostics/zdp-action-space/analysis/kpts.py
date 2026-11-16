"""Read-only: explain KING_SPAWN (11), KING_ACT (12), KING_FORM (13) fallbacks from the recorded features (cpi-x0)."""
import sys
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else '/home/terry/rltest/bc26a0/runs/rl/cpi-x0/iter-0000/data'
NONE = 0xFFFF


def load(p):
    z = np.load(f'{D}/p{p}.npz')
    k = z['k'].astype(np.int64); fl = z['flags'].astype(np.int64)
    keep = (k != NONE) & ((fl & 2) == 0)
    out = {n: z[n][keep] for n in ('x', 'k', 'flags', 'game', 'team', 'round', 'robot')}
    out['k'] = out['k'].astype(np.int64); out['fb'] = (out['flags'] & 1) != 0
    out['x'] = out['x'].astype(np.int64)
    return out


def pct(a):
    return f'{100 * float(np.mean(a)):.1f}%' if len(a) else 'n/a'


# ---------------- KING_SPAWN
d = load(11); x, k, fb = d['x'], d['k'], d['fb']
print('P11 rows', len(k), 'x shape', x.shape)
bm, dm, cost, g10 = x[:, 22], x[:, 23], x[:, 5], x[:, 2]
boot, war, babies, cap = x[:, 9], x[:, 10], x[:, 6], x[:, 13]
nosur = (boot == 0) & (war == 0) & (x[:, 14] < 32768)
flo = ~nosur & (x[:, 15] < 32768)
capd = ~nosur & ~flo & (babies >= cap)
zspawn = ~nosur & ~flo & ~capd
zst = np.where(nosur, 0, np.where(flo, 1, np.where(capd, 2, 3)))
print('zdb state share: nosurplus %s floor %s cap %s spawn %s' % tuple(pct(zst == s) for s in range(4)))
print('bm==0 (nothing buildable) share of all records', pct(bm == 0), '; g<cost (g10*10+9<cost)', pct(g10 * 10 + 9 < cost))
bld = (k >= 3) & (k <= 18)
i = np.clip(k - 3, 0, 15)
bit = (bm >> i) & 1
print('builds picked', int(bld.sum()), 'fallback', pct(fb[bld]), 'agreement fb == (bit i==0):', pct(fb[bld] == (bit[bld] == 0)))
print('  of build picks: bm==0 (unaffordable/all blocked)', pct(bm[bld] == 0), '; bm!=0 & tile i blocked', pct((bm[bld] != 0) & (bit[bld] == 0)))
for s, nm in enumerate(['nosurplus', 'floor', 'cap', 'zdb-spawn']):
    m = bld & (zst == s)
    print(f'  zdb state {nm}: build picks {int(m.sum())}, executed (no fallback) {pct(~fb[m])}')
ok = bld & ~fb
print('  executed builds by zdb state (extra spawns unless zdb-spawn):', {nm: int((ok & (zst == s)).sum()) for s, nm in enumerate(['nosurplus', 'floor', 'cap', 'zdb-spawn'])})
# label after a failed build pick
fbb = bld & fb
print('  failed build -> zdb label dist:', {int(a): int(c) for a, c in zip(*np.unique(k[fbb] * 0 + 0, return_counts=True))} if False else '')
# per team-game extra spawns
tg = d['game'].astype(np.int64) * 2 + d['team'].astype(np.int64)
ntg = len(np.unique(tg))
print('  team-games', ntg, 'extra spawns per team-game (executed build picks in non-spawn zdb state):', round(float((ok & (zst != 3)).sum()) / ntg, 2),
      '; zdb-spawn-state records per team-game', round(float((zst == 3).sum()) / ntg, 1))
for s, nm in enumerate(['nosurplus', 'floor', 'cap']):
    m = ok & (zst == s)
    if m.any():
        print(f'    extra spawn in {nm}: median cost {int(np.median(cost[m]))}, median global {int(np.median(g10[m]) * 10)}, median babies {int(np.median(babies[m]))}')
dig = k == 19
print('S_DIG picks', int(dig.sum()), 'fb', pct(fb[dig]), '; dm==0 among them', pct(dm[dig] == 0), '; g<45', pct(g10[dig] * 10 + 9 < 45))
for kk in (0, 1, 2, 20, 21):
    m = k == kk
    print(f'  option {kk}: picks {int(m.sum())}, zdb state dist', {nm: pct(zst[m] == s) for s, nm in enumerate(['nos', 'flo', 'cap', 'spawn'])})

# ---------------- KING_ACT
d = load(12); x, k, fb = d['x'], d['k'], d['fb']
print('\nP12 rows', len(k), 'x shape', x.shape)
coop, gg, reach, catr, nt, np_, nr = x[:, 0], x[:, 3], x[:, 8], x[:, 15], x[:, 18], x[:, 19], x[:, 20]
rtr = x[:, 16]
print('records: coop', pct(coop == 1), '; reach>0', pct(reach > 0), '; catReach', pct(catr > 0), '; nt>0', pct(nt > 0), '; np>0', pct(np_ > 0), '; nr>0', pct(nr > 0))
names = ['NONE', 'BITE_KILL', 'BITE', 'BITE_CAT', 'TRAP_FOE', 'TRAP_FORT', 'MOAT_FILL', 'BITE_NEAREST', 'BITE_KING', 'CAT_TRAP', 'DIG_RING']
for kk in range(11):
    m = k == kk
    s = f'  {kk} {names[kk]}: picks {int(m.sum())} fb {pct(fb[m])}'
    if kk == 5: s += f' | nt==0 among picks {pct(nt[m] == 0)} agree {pct(fb[m] == (nt[m] == 0))}'
    if kk == 6: s += f' | np==0 among picks {pct(np_[m] == 0)}'
    if kk == 10: s += f' | nr==0 or g<45 {pct((nr[m] == 0) | (gg[m] < 45))} agree {pct(fb[m] == ((nr[m] == 0) | (gg[m] < 45)))}'
    if kk in (1, 2, 7, 8): s += f' | reach==0 {pct(reach[m] == 0)}; executed in coop {int((m & ~fb & (coop == 1)).sum())}'
    if kk == 3: s += f' | !catReach {pct(catr[m] == 0)}'
    if kk == 9: s += f' | executed coop/war {int((m & ~fb & (coop == 1)).sum())}/{int((m & ~fb & (coop == 0)).sum())}; fb in war {pct(fb[m & (coop == 0)])}, in coop {pct(fb[m & (coop == 1)])}'
    if kk == 4: s += f' | no enemy baby in sight {pct(x[m, 4] == 0)}'
    print(s)
tg = d['game'].astype(np.int64) * 2 + d['team'].astype(np.int64)
ntg = len(np.unique(tg))
for kk in (4, 5, 6, 9, 10):
    print(f'  executed {names[kk]} per team-game: {round(float(((k == kk) & ~fb).sum()) / ntg, 2)}')
print('  rat traps (team) at KING_ACT records: share >= 10 (babies stop combat traps):', pct(rtr >= 10), '; mean', round(float(rtr.mean()), 2))
# rat trap count by round bucket
r = d['round'].astype(np.int64)
for lo, hi in ((0, 200), (200, 500), (500, 1000), (1000, 1500), (1500, 2001)):
    m = (r >= lo) & (r < hi)
    if m.any(): print(f'    rounds {lo}-{hi}: mean rat traps {rtr[m].mean():.2f}, share>=10 {pct(rtr[m] >= 10)}')

# ---------------- KING_FORM
d = load(13); x, k, fb = d['x'], d['k'], d['fb']
print('\nP13 rows', len(k), 'x shape', x.shape)
op, warf, capf, rich, endk, inc, gate = x[:, 3], x[:, 10], x[:, 11], x[:, 12], x[:, 13], x[:, 14], x[:, 15]
kind, rm, nk = x[:, 18], x[:, 20], x[:, 2]
print('records: site open', pct(op == 1), '; numKings dist', {int(a): int(c) for a, c in zip(*np.unique(nk, return_counts=True))})
zst = np.where(warf == 1, 0, np.where((capf == 1) & (rich == 0), 1, np.where(op == 1, 2, np.where(endk == 1, 3, np.where(gate == 0, 4, 5)))))
zn = ['WAR_OFF', 'CAP_OFF', 'site-open', 'endKing', 'IDLE(gate)', 'gate-pass(rich/income)']
print('zdb state share:', {zn[s]: pct(zst == s) for s in range(6)})
opn = (k >= 13) & (k <= 29)
for kk in range(13, 30):
    m = k == kk
    s = f'  {kk}: picks {int(m.sum())} fb {pct(fb[m])}'
    if kk >= 18:
        b = (rm[m] >> (kk - 18)) & 1
        s += f' | rm bit clear {pct(b == 0)} agree {pct(fb[m] == (b == 0))}'
    if kk in (15, 16): s += f' | rm==0 {pct(rm[m] == 0)} agree {pct(fb[m] == (rm[m] == 0))}'
    if kk == 13: s += f' | pickFormSite kind!=1 {pct(kind[m] != 1)}'
    print(s)
ok = opn & ~fb
print('executed site opens:', int(ok.sum()), 'by zdb state', {zn[s]: int((ok & (zst == s)).sum()) for s in range(6)})
tg = d['game'].astype(np.int64) * 2 + d['team'].astype(np.int64)
ntg = len(np.unique(tg))
print('  per team-game', round(float(ok.sum()) / ntg, 2))
# how long net-opened sites stay open: walk each team-game's records in round order
order = np.lexsort((d['round'], d['robot'], tg))
k2, fb2, op2, tg2, rb2, rd2, nk2, zs2 = k[order], fb[order], op[order], tg[order], d['robot'][order], d['round'][order], nk[order], zst[order]
lens, grew = [], 0
idx = np.nonzero(((k2 >= 13) & (k2 <= 29) & ~fb2))[0]
for j in idx:
    n = 0; t = j + 1
    while t < len(k2) and tg2[t] == tg2[j] and rb2[t] == rb2[j] and op2[t] == 1:
        n += 1; t += 1
    lens.append(n)
    # did numKings rise within 80 records of the open?
    e = j + 1
    while e < len(k2) and e < j + 80 and tg2[e] == tg2[j] and rb2[e] == rb2[j]:
        if nk2[e] > nk2[j]: grew += 1; break
        e += 1
lens = np.array(lens)
if len(lens):
    print('  net-opened site stays open (records): median', int(np.median(lens)), 'p90', int(np.quantile(lens, 0.9)), 'share closed next turn', pct(lens == 0))
    print('  numKings rose within 80 turns after a net open:', grew, 'of', len(lens))
    for s in range(6):
        m = zs2[idx] == s
        if m.any(): print(f'    opened in zdb state {zn[s]}: n {int(m.sum())}, median open {int(np.median(lens[m]))}, closed next turn {pct(lens[m] == 0)}')
for kk in range(0, 13):
    m = k == kk
    print(f'  {kk}: picks {int(m.sum())} fb {pct(fb[m])} | with site open {pct(op[m] == 1)}')
