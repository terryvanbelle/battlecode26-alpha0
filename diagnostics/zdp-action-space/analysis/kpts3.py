"""Read-only: per-option picks, share, fallback rate and executed-per-team-game for points 11, 12, 13 of one data dir."""
import sys
import numpy as np
D = sys.argv[1]
NONE = 0xFFFF
print('DATA', D)
for p in (11, 12, 13):
    z = np.load(f'{D}/p{p}.npz')
    k = z['k'].astype(np.int64); fl = z['flags'].astype(np.int64)
    keep = (k != NONE) & ((fl & 2) == 0)
    k = k[keep]; fb = (fl[keep] & 1) != 0
    tg = z['game'][keep].astype(np.int64) * 2 + z['team'][keep].astype(np.int64)
    ntg = len(np.unique(tg)); n = len(k)
    print(f'P{p} records {n} team-games {ntg} records/team-game {n / ntg:.1f} overall fallback {100 * fb.mean():.2f}%')
    rows = []
    for o in range(int(k.max()) + 1):
        m = k == o
        c = int(m.sum())
        if c == 0: continue
        rows.append(f'{o}:{c}/{100 * c / n:.2f}%/fb{100 * fb[m].mean():.1f}%/ex{(m & ~fb).sum() / ntg:.2f}')
    print('  ' + ' '.join(rows))
