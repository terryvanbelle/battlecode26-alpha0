"""Train the decision nets (one small MLP per decision point) on zdp decision records.

train_dp.py --games 'runs/dp/teacher-v1/games/*' --out runs/dp/net-v1 [--hidden 64,64] [--epochs 20] [--device cpu]

Reads decisions-*.bin.gz from each game dir, splits train/val by game, trains a classifier per point
(label = the option zdb took) and writes net.json (float weights, input normalisation folded into layer 1)
and metrics.json. scripts/build_dpnet.py packs net.json into a net-mode bot.
"""
import argparse, glob, json, re, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

NONE = 0xFFFF


def option_counts(src=ROOT / 'java/bots/zdp'):
    """{point id: option count} from the D*.java constants (NAME = id and NAME_OPTIONS = count)."""
    consts = {}
    for f in sorted(Path(src).glob('D*.java')):
        for m in re.finditer(r'\b([A-Z][A-Z0-9_]*)\s*=\s*(\d+)\b', f.read_text()):
            consts.setdefault(m.group(1), int(m.group(2)))
    return {consts[k[:-8]]: v for k, v in consts.items() if k.endswith('_OPTIONS') and k[:-8] in consts}


def read_fast(path, keep, rng, keep0={}):
    """{point: (feats [N, n] int32, labels)} from one decisions file, keeping each labelled record of point p with
    probability keep.get(p, 1). Assumes a point's records share one feature count (true for zdp)."""
    import gzip
    buf = gzip.decompress(Path(path).read_bytes())
    mv, off, L = memoryview(buf), 0, len(buf)
    offs = {}
    while off + 14 <= L:
        n = (buf[off + 12] << 8) | buf[off + 13]
        if off + 14 + 2 * n > L:
            break
        pt = (buf[off + 8] << 8) | buf[off + 9]
        offs.setdefault((pt, n), []).append(off)
        off += 14 + 2 * n
    raw = np.frombuffer(buf, dtype=np.uint8)
    out = {}
    for (pt, n), o in offs.items():
        o = np.asarray(o, dtype=np.int64)
        lab = (raw[o + 10].astype(np.int64) << 8) | raw[o + 11]
        sel = lab != NONE
        if keep.get(pt, 1.0) < 1.0:
            sel &= rng.random(len(o)) < keep[pt]
        if keep0.get(pt, 1.0) < 1.0:   # label 0 only (corrected at export: label-0 logit minus log(keep0))
            sel &= (lab != 0) | (rng.random(len(o)) < keep0[pt])
        o, lab = o[sel], lab[sel]
        idx = o[:, None] + 14 + 2 * np.arange(n)[None, :]
        feats = (raw[idx].astype(np.int32) << 8) | raw[idx + 1]
        out.setdefault(pt, []).append((feats, lab))
    return out


def load(dirs, keep, keep0={}):
    rng = np.random.default_rng(1)
    per = {}   # point -> list of (feats, labels, game index)
    for gi, d in enumerate(dirs):
        for f in sorted(Path(d).glob('decisions-*.bin.gz')):
            for pt, parts in read_fast(f, keep, rng, keep0).items():
                for X, y in parts:
                    per.setdefault(pt, []).append((X, y, np.full(len(y), gi)))
    out = {}
    for pt, parts in per.items():
        n = max(p[0].shape[1] for p in parts)
        X = np.concatenate([np.pad(p[0], ((0, 0), (0, n - p[0].shape[1]))) for p in parts]).astype(np.float32)
        out[pt] = (X, np.concatenate([p[1] for p in parts]).astype(np.int64), np.concatenate([p[2] for p in parts]))
    return out


def mlp(n_in, hidden, n_out):
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', nargs='+', required=True, help='game dir globs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--hidden', default='64,64')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--keep0', default='', help='per-point keep probability for label-0 records only, e.g. 2:0.05')
    ap.add_argument('--keep', default='', help='per-point subsampling, e.g. 1:0.3,2:0.1 (keep probability)')
    a = ap.parse_args()
    torch.manual_seed(0)
    dirs = sorted({d for g in a.games for d in glob.glob(g)})
    t0 = time.time()
    keep = {int(k): float(v) for k, v in (x.split(':') for x in a.keep.split(',') if x)}
    keep0 = {int(k): float(v) for k, v in (x.split(':') for x in a.keep0.split(',') if x)}
    data = load(dirs, keep, keep0)
    counts = option_counts()
    hidden = [int(h) for h in a.hidden.split(',') if h]
    rng = np.random.default_rng(0)
    val_games = set(rng.choice(len(dirs), max(1, int(len(dirs) * a.val_frac)), replace=False).tolist())
    print(f'{len(dirs)} games ({len(val_games)} val), points {sorted(data)}, loaded in {time.time() - t0:.0f}s', flush=True)
    net_out, metrics = {}, {}
    for pt, (X, y, g) in sorted(data.items()):
        n_out = max(counts.get(pt, 0), int(y.max()) + 1)
        isval = np.isin(g, list(val_games))
        Xt, yt, Xv, yv = X[~isval], y[~isval], X[isval], y[isval]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-3
        model = mlp(X.shape[1], hidden, n_out).to(a.device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        tX = torch.tensor((Xt - mu) / sd, device=a.device); ty = torch.tensor(yt, device=a.device)
        vX = torch.tensor((Xv - mu) / sd, device=a.device); vy = torch.tensor(yv, device=a.device)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        for ep in range(a.epochs):
            model.train()
            perm = torch.randperm(len(tX), device=a.device)
            for i in range(0, len(perm), a.batch):
                idx = perm[i:i + a.batch]
                loss = nn.functional.cross_entropy(model(tX[idx]), ty[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            lv = model(vX) if len(vX) else None
            acc = float((lv.argmax(1) == vy).float().mean()) if lv is not None else None
            vloss = float(nn.functional.cross_entropy(lv, vy)) if lv is not None else None
            tacc = float((model(tX).argmax(1) == ty).float().mean())
        base = float((yv == np.bincount(yt).argmax()).mean()) if len(yv) else None   # on the (subsampled) val set
        hist = np.bincount(y, minlength=n_out).tolist()
        metrics[pt] = dict(train=int(len(yt)), val=int(len(yv)), n_in=int(X.shape[1]), n_out=n_out, val_acc=acc,
                           val_loss=vloss, train_acc=tacc, majority_baseline=base, label_hist=hist)
        print(f'point {pt}: {len(yt)} train / {len(yv)} val, {X.shape[1]} features, {n_out} options; '
              f'val acc {acc} (majority {base}), train acc {tacc:.3f}', flush=True)
        # export with normalisation folded into layer 1: W1 (x - mu) / sd + b1 = (W1 / sd) x + (b1 - W1 mu / sd)
        lins = [m for m in model if isinstance(m, nn.Linear)]
        layers = []
        for i, l in enumerate(lins):
            Wm = l.weight.detach().cpu().double().numpy(); bv = l.bias.detach().cpu().double().numpy()
            if i == 0:
                bv = bv - (Wm * (mu / sd)).sum(1); Wm = Wm / sd
            if i == len(lins) - 1 and keep0.get(pt, 1.0) < 1.0:
                bv = bv.copy(); bv[0] -= np.log(keep0[pt])   # undo label-0 subsampling: trained odds = true odds * keep0
            layers.append(dict(n_in=int(Wm.shape[1]), n_out=int(Wm.shape[0]), w=Wm.ravel().tolist(), b=bv.tolist()))
        net_out[pt] = layers
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'net.json').write_text(json.dumps({'version': 'dpnet-v1', 'hidden': hidden, 'points': {str(k): v for k, v in net_out.items()}}))
    (out / 'metrics.json').write_text(json.dumps({'games': len(dirs), 'val_games': len(val_games), 'points': metrics}, indent=1))
    print(f'wrote {out}/net.json in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
