"""Behaviour-cloning fit of the student MLP on teacher records; exports Java weights.

python -m alpha0.student.train --run runs/teacher-v1[,runs/teacher-v2] --out runs/student-v1 [--epochs 4]
"""
import argparse, json, struct, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import read_records, unpack_mask
from .spec import OBS_VERSION, OBS, MASK_BYTES, RECORD, N_ACTIONS, LOCAL, GLOBAL, L, CL, action_group

MAGIC = 0x53544E31


class StudentNet(nn.Module):
    def __init__(self, h1=384, h2=256):
        super().__init__()
        self.l1, self.l2 = nn.Linear(OBS, h1), nn.Linear(h1, h2)
        self.policy, self.value = nn.Linear(h2, N_ACTIONS), nn.Linear(h2, 1)

    def forward(self, x_uint8):
        x = x_uint8.float() * (1 / 255)
        h = F.relu(self.l2(F.relu(self.l1(x))))
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1)


def export(net: StudentNet, path: Path):
    def arr(t):
        return t.detach().cpu().float().numpy().astype('>f4').tobytes()
    with open(path, 'wb') as f:
        f.write(struct.pack('>5i', MAGIC, OBS, net.l1.out_features, net.l2.out_features, N_ACTIONS))
        f.write(arr(net.l1.weight.t().contiguous())); f.write(arr(net.l1.bias))
        f.write(arr(net.l2.weight)); f.write(arr(net.l2.bias))
        f.write(arr(net.policy.weight)); f.write(arr(net.policy.bias))
        f.write(arr(net.value.weight.reshape(-1))); f.write(arr(net.value.bias))


def _gz_records(f: Path) -> int:
    with open(f, 'rb') as fh:  # gzip trailer holds uncompressed size mod 2**32 (files are far smaller)
        fh.seek(-4, 2)
        return int.from_bytes(fh.read(4), 'little') // RECORD


def rare_state(o):
    """State-based (label-free) test for rare decisive baby states: can become king, or an enemy is in view."""
    loc = o[:, :LOCAL].reshape(len(o), L * L, CL)
    return ((o[:, LOCAL + GLOBAL + 31] > 0) | (o[:, LOCAL + GLOBAL + 47] > 0)   # can become king / defend alert (obs-v2)
            | (loc[:, :, 10] > 0).any(1) | (loc[:, :, 11] > 0).any(1))


def load_split(runs, split: str, max_games=None, baby_keep=1.0, seed=0):
    """Two-pass load into preallocated arrays, so peak RAM is ~1x the dataset (~25 GB for all of teacher-v1).
    baby_keep < 1 subsamples ordinary baby records (kings and rare_state babies are always kept) to fit RAM."""
    runs = [runs] if isinstance(runs, (str, Path)) else runs
    sched = [(Path(r), g) for r in runs for g in json.loads((Path(r) / 'schedule.json').read_text())]
    files, games = [], 0
    for run, g in sched:
        if g['split'] != split:
            continue
        d = run / 'games' / g['id']
        res = d / 'result.json'
        if not res.exists():
            continue
        r = json.loads(res.read_text())
        if r.get('error') or r.get('hook_error') or not r.get('finished'):
            continue
        if r.get('obs_version') != OBS_VERSION:
            raise SystemExit(f'{res}: obs {r.get("obs_version")} != {OBS_VERSION}')
        for side in g['teacher_sides']:
            f = d / f'records-{side}.bin.gz'
            if f.exists():
                v = 0.0 if r['winner'] is None else (1.0 if r['winner'] == side else -1.0)
                files.append((f, v))
        games += 1
        if max_games and games >= max_games:
            break
    def kept(i, f):
        h, o, m = read_records(f)
        keep = unpack_mask(m, N_ACTIONS).sum(1) > 1
        if baby_keep < 1:   # per-file seed, so the counting pass and the fill pass agree exactly
            rng = np.random.default_rng([seed, i])
            keep &= (h['king'] != 0) | rare_state(o) | (rng.random(len(keep)) < baby_keep)
        return h, o, m, keep
    if baby_keep < 1:
        cap = sum(int(kept(i, f)[3].sum()) for i, (f, _) in enumerate(files))
    else:
        cap = sum(_gz_records(f) for f, _ in files)
    obs = np.empty((cap, OBS), np.uint8); mask = np.empty((cap, MASK_BYTES), np.uint8)
    label = np.empty(cap, np.int64); value = np.empty(cap, np.float32); king = np.empty(cap, bool)
    n = 0
    for i, (f, v) in enumerate(files):
        h, o, m, keep = kept(i, f)
        k = int(keep.sum())
        obs[n:n + k] = o[keep]; mask[n:n + k] = m[keep]
        label[n:n + k] = h['label'][keep]; king[n:n + k] = h['king'][keep].astype(bool); value[n:n + k] = v
        n += k
    return dict(obs=obs[:n], mask=mask[:n], label=label[:n], value=value[:n], king=king[:n], games=games)


def batches(n, bs, rng):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def run_batch(net, data, idx, device):
    x = torch.from_numpy(data['obs'][idx]).to(device)
    legal = torch.from_numpy(unpack_mask(data['mask'][idx], N_ACTIONS)).to(device)
    y = torch.from_numpy(data['label'][idx]).to(device)
    v = torch.from_numpy(data['value'][idx]).to(device)
    k = torch.from_numpy(data['king'][idx]).to(device)
    logits, value = net(x)
    logits = logits.masked_fill(~legal, -1e9)
    return logits, value, y, v, k


@torch.no_grad()
def evaluate(net, data, device, bs=4096):
    net.eval()
    n = len(data['label'])
    tot = Counter(); correct = Counter(); loss = 0.0; vloss = 0.0
    for i in range(0, n, bs):
        idx = np.arange(i, min(n, i + bs))
        logits, value, y, v, _ = run_batch(net, data, idx, device)
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        vloss += F.mse_loss(value, v, reduction='sum').item()
        pred = logits.argmax(1).cpu().numpy()
        for yy, pp, kk in zip(data['label'][idx], pred, data['king'][idx]):
            g = ('king:' if kk else 'baby:') + action_group(int(yy))
            tot[g] += 1; correct[g] += int(yy == pp)
            tot['all'] += 1; correct['all'] += int(yy == pp)
    net.train()
    return {'loss': loss / max(n, 1), 'value_mse': vloss / max(n, 1), 'acc': correct['all'] / max(n, 1),
            'by_group': {g: [round(correct[g] / tot[g], 3), tot[g]] for g in sorted(tot) if g != 'all'}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='teacher run dir, or several separated by commas')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--value-weight', type=float, default=0.25)
    ap.add_argument('--h1', type=int, default=384)
    ap.add_argument('--h2', type=int, default=256)
    ap.add_argument('--king-weight', type=float, default=4.0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--max-games', type=int)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--baby-keep', type=float, default=1.0, help='fraction of non-king train records kept (RAM)')
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    run, out = [Path(r) for r in a.run.split(',')], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train, val = load_split(run, 'train', a.max_games, a.baby_keep, a.seed), load_split(run, 'val', a.max_games and max(1, a.max_games // 9))
    print(f"loaded train {len(train['label'])} recs/{train['games']} games, val {len(val['label'])}/{val['games']} in {time.time()-t0:.0f}s", flush=True)
    device = 'cpu'
    net = StudentNet(a.h1, a.h2)
    export(net, out / 'untrained.bin')
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(train['label']) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.05)
    rng = np.random.default_rng(a.seed)
    log = open(out / 'train.jsonl', 'a')
    best = None
    step = 0
    for ep in range(a.epochs):
        run_loss = 0.0
        for idx in batches(len(train['label']), a.bs, rng):
            logits, value, y, v, k = run_batch(net, train, np.sort(idx), device)
            w = 1.0 + (a.king_weight - 1.0) * k.float()
            loss = (F.cross_entropy(logits, y, reduction='none') * w).sum() / w.sum() + a.value_weight * F.mse_loss(value, v)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            run_loss += loss.item()
            if step % 500 == 0:
                print(f'ep {ep} step {step}/{steps} loss {run_loss/500:.4f} {time.time()-t0:.0f}s', flush=True); run_loss = 0.0
        ev = evaluate(net, val, device)
        ev.update(epoch=ep, step=step, seconds=round(time.time() - t0))
        log.write(json.dumps(ev) + '\n'); log.flush()
        print(f"epoch {ep}: val loss {ev['loss']:.4f} acc {ev['acc']:.4f} value_mse {ev['value_mse']:.3f}", flush=True)
        export(net, out / f'epoch{ep}.bin')
        if best is None or ev['loss'] < best:
            best = ev['loss']
            export(net, out / 'best.bin'); torch.save(net.state_dict(), out / 'best.pt')
            (out / 'best.json').write_text(json.dumps(ev, indent=1))
    print('done', flush=True)


if __name__ == '__main__':
    main()
