"""Per point: deviation from the net's argmax, and how often a deviation falls back to zdb's cascade."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'scripts')
import dpnet_io
NONE = 0xFFFF
data = Path(sys.argv[1]); net = json.loads(Path(sys.argv[2]).read_text())
summ = json.loads((data / 'summary.json').read_text())
fpts = {int(k): v for k, v in summ['floor_points'].items()}
out = {}
for f in sorted(data.glob('p*.npz'), key=lambda p: int(p.stem[1:])):
    p = int(f.stem[1:])
    if str(p) not in net['points']:
        continue
    z = np.load(f)
    k = z['k'].astype(np.int64); fl = z['flags']
    sec = (fl & 2) != 0
    ls = dpnet_io.layers_from_raw(net['points'][str(p)])
    n_in, n_out = ls[0]['w'].shape[1], ls[-1]['w'].shape[0]
    keep = (k != NONE) & ~sec
    idx = np.nonzero(keep)[0]
    if len(idx) > 300000:
        idx = np.sort(np.random.default_rng(0).choice(idx, 300000, replace=False))
    x = np.pad(z['x'][idx], ((0, 0), (0, max(0, n_in - z['x'].shape[1]))))[:, :n_in]
    mask = dpnet_io.fit_mask(z['mask'][idx], len(idx), n_out)
    kk = k[idx]; fb = (fl[idx] & 1) != 0
    multi = mask.sum(1) > 1
    lg = dpnet_io.forward(ls, x, exact=False)
    pr = dpnet_io.probs(lg, mask, 1.0, 0.0).astype(np.float64)
    am = pr.argmax(1)
    dev = (kk != am) & multi
    nopt = mask.sum(1)
    r = dict(records=int(len(idx)), multi_share=round(float(multi.mean()), 4), mean_allowed=round(float(nopt[multi].mean()), 2) if multi.any() else 0,
             p_top_mean=round(float(pr.max(1)[multi].mean()), 4) if multi.any() else None,
             dev_rate=round(float(dev[multi].mean()), 4) if multi.any() else None,
             fb_given_dev=round(float(fb[dev].mean()), 4) if dev.any() else None,
             fb_given_top=round(float(fb[~dev & multi].mean()), 4) if (~dev & multi).any() else None,
             real_dev_rate=round(float((dev & ~fb)[multi].mean()), 4) if multi.any() else None,
             floor=fpts.get(p, summ['floor']))
    # per deviated option: count, fallback share
    per = {}
    for o in np.unique(kk[dev]):
        m = dev & (kk == o)
        per[int(o)] = [int(m.sum()), round(float(fb[m].mean()), 3)]
    r['dev_by_option'] = per
    out[p] = r
    print(p, {a: b for a, b in r.items() if a != 'dev_by_option'}, flush=True)
json.dump(out, open('/tmp/devfb.json', 'w'))
