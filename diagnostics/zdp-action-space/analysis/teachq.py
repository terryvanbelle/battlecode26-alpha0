"""Read-only: zdb-mode (teacher-dp2) PRE/POST records: THROW labels vs carry, and the late-game feature ranges."""
import sys, glob
import numpy as np
sys.path.insert(0, '/home/terry/projects/vibe/bc26a0/scripts')
import decisions
files = sorted(glob.glob('/home/terry/projects/vibe/bc26a0/runs/teacher-dp2/games/g00[0-1]??/decisions-*.bin.gz'))[:60]
acc = {2: [], 3: []}
lab = {2: [], 3: []}
rnd = {2: [], 3: []}
for f in files:
    d = decisions.read(f)
    pt = d['point']
    for p in (2, 3):
        s = pt == p
        acc[p].append(d['feats'][s][:, :53]); lab[p].append(d['label'][s]); rnd[p].append(d['round'][s])
print('files', len(files))
for p in (2, 3):
    x = np.concatenate(acc[p]); l = np.concatenate(lab[p]); r = np.concatenate(rnd[p])
    print(f'point {p}: n {len(l)}; label THROW {int((l == 2).sum())}, THROW with carry=0 {int(((l == 2) & (x[:, 12] == 0)).sum())}, carry=2 rows {int((x[:, 12] == 2).sum())}, carry=1 rows {int((x[:, 12] == 1).sum())}')
    for j, nm in ((32, 'nRatT'), (31, 'nCatT'), (48, 'dirt'), (33, 'budget'), (5, 'r10'), (16, 'friends'), (43, 'front')):
        print(f'   {nm}: q50/90/99/max {np.percentile(x[:, j], [50, 90, 99]).tolist()} {x[:, j].max()}; share>=20 {np.mean(x[:, j] >= 20):.4f}')
    late = r >= 1000
    print(f'   rounds>=1000 share {late.mean():.3f}; nRatT>=20 & r>=1000 share {np.mean(late & (x[:, 32] >= 20)):.5f}')
