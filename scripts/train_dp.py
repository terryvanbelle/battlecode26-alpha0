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
import decisions  # noqa: E402

NONE = 0xFFFF


def option_counts(src=ROOT / 'java/bots/zdp'):
    """{point id: option count} from the D*.java constants (NAME = id and NAME_OPTIONS = count)."""
    consts = {}
    for f in sorted(Path(src).glob('D*.java')):
        for m in re.finditer(r'\b([A-Z][A-Z0-9_]*)\s*=\s*(\d+)\b', f.read_text()):
            consts.setdefault(m.group(1), int(m.group(2)))
    return {consts[k[:-8]]: v for k, v in consts.items() if k.endswith('_OPTIONS') and k[:-8] in consts}


def load(dirs):
    per = {}   # point -> list of (feats, labels, game index)
    for gi, d in enumerate(dirs):
        for f in sorted(Path(d).glob('decisions-*.bin.gz')):
            r = decisions.read(f)
            for pt in np.unique(r['point']):
                sel = (r['point'] == pt) & (r['label'] != NONE)
                if not sel.any():
                    continue
                n = int(r['n'][sel].max())
                per.setdefault(int(pt), []).append((r['feats'][sel, :n], r['label'][sel], np.full(sel.sum(), gi)))
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
    a = ap.parse_args()
    torch.manual_seed(0)
    dirs = sorted({d for g in a.games for d in glob.glob(g)})
    t0 = time.time()
    data = load(dirs)
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
        base = float((yv == np.bincount(yt).argmax()).mean()) if len(yv) else None
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
            layers.append(dict(n_in=int(Wm.shape[1]), n_out=int(Wm.shape[0]), w=Wm.ravel().tolist(), b=bv.tolist()))
        net_out[pt] = layers
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'net.json').write_text(json.dumps({'version': 'dpnet-v1', 'hidden': hidden, 'points': {str(k): v for k, v in net_out.items()}}))
    (out / 'metrics.json').write_text(json.dumps({'games': len(dirs), 'val_games': len(val_games), 'points': metrics}, indent=1))
    print(f'wrote {out}/net.json in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
