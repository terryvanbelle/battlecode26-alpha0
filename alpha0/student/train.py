"""Behaviour-cloning fit of the student network (student-mlp-v2) on teacher records; exports Java weights.

python -m alpha0.student.train --run runs/teacher-v9[,runs/teacher-v10] --out runs/student-v9 [--epochs 6]

Records are streamed: each epoch visits the training games in random order, CHUNK games at a time, shuffled within
the chunk (obs-v4 records are ~9 KB, so the whole set does not fit in RAM).
"""
import argparse, json, math, struct, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import read_records, unpack_mask
from .spec import (OBS_VERSION, OBS, RECORD, N_ACTIONS, LEGACY_N_ACTIONS, LOCAL, GLOBAL, L, CL, DENSE, MAP_BITS, OTHER_BITS, BITS,
                   PARAM_MAX_BITS, PARAM_BITS, PARAM_KIND_OF, P_NONE, action_group)

MAGIC = 0x53544E32
KIND_OF = torch.tensor(PARAM_KIND_OF)
# bits used by each param kind: [kinds, 32] 0/1
KIND_BITS = torch.tensor([[1.0 if b < PARAM_BITS[k] else 0.0 for b in range(PARAM_MAX_BITS)] for k in range(4)])
SHIFTS = torch.arange(8, dtype=torch.uint8)


class StudentNet(nn.Module):
    def __init__(self, h1=384, h2=256, hm=64, hq=64):
        super().__init__()
        self.map = nn.Linear(MAP_BITS, hm)
        self.l1, self.l2 = nn.Linear(DENSE + OTHER_BITS + hm, h1), nn.Linear(h1, h2)
        self.policy, self.value = nn.Linear(h2, N_ACTIONS), nn.Linear(h2, 1)
        self.pq, self.emb, self.pb = nn.Linear(h2, hq), nn.Embedding(N_ACTIONS, hq), nn.Linear(hq, PARAM_MAX_BITS)
        nn.init.normal_(self.emb.weight, std=0.1)

    def trunk(self, x_uint8):
        dense = x_uint8[:, :DENSE].float() * (1 / 255)
        bits = ((x_uint8[:, DENSE:].unsqueeze(-1) >> SHIFTS.to(x_uint8.device)) & 1).reshape(len(x_uint8), -1)[:, :BITS].float()
        m = F.relu(self.map(bits[:, :MAP_BITS]))
        return F.relu(self.l2(F.relu(self.l1(torch.cat([dense, bits[:, MAP_BITS:], m], 1)))))

    def forward(self, x_uint8):
        h = self.trunk(x_uint8)
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1), h

    def param_logits(self, h, a):
        return self.pb(F.relu(self.pq(h) + self.emb(a)))


def export(net: StudentNet, path: Path):
    def arr(t):
        return t.detach().cpu().float().numpy().astype('>f4').tobytes()
    with open(path, 'wb') as f:
        f.write(struct.pack('>10i', MAGIC, DENSE, MAP_BITS, OTHER_BITS, net.map.out_features, net.l1.out_features,
                            net.l2.out_features, net.pq.out_features, N_ACTIONS, PARAM_MAX_BITS))
        f.write(arr(net.map.weight.t().contiguous())); f.write(arr(net.map.bias))
        f.write(arr(net.l1.weight.t().contiguous())); f.write(arr(net.l1.bias))
        f.write(arr(net.l2.weight)); f.write(arr(net.l2.bias))
        f.write(arr(net.policy.weight)); f.write(arr(net.policy.bias))
        f.write(arr(net.value.weight.reshape(-1))); f.write(arr(net.value.bias))
        f.write(arr(net.pq.weight)); f.write(arr(net.emb.weight)); f.write(arr(net.pq.bias))
        f.write(arr(net.pb.weight)); f.write(arr(net.pb.bias))


def _gz_records(f: Path) -> int:
    with open(f, 'rb') as fh:  # gzip trailer holds uncompressed size mod 2**32
        fh.seek(-4, 2)
        return int.from_bytes(fh.read(4), 'little') // RECORD


def rare_state(o):
    """State-based (label-free) test for rare decisive baby states: can become king, defend alert, or an enemy in view."""
    loc = o[:, :LOCAL].reshape(len(o), L * L, CL)
    return ((o[:, LOCAL + GLOBAL + 31] > 0) | (o[:, LOCAL + GLOBAL + 47] > 0)
            | (loc[:, :, 10] > 0).any(1) | (loc[:, :, 11] > 0).any(1))


def game_files(runs, split: str, max_games=None):
    """[(records file, outcome for that side)] for finished, error-free games of the split."""
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
        if r.get('obs_version') != OBS_VERSION or r.get('obs_size') != OBS or r.get('n_actions') not in (N_ACTIONS, *LEGACY_N_ACTIONS):
            raise SystemExit(f'{res}: obs {r.get("obs_version")}/{r.get("obs_size")}/{r.get("n_actions")} != {OBS_VERSION}/{OBS}/{N_ACTIONS}')
        for side in g['teacher_sides']:
            f = d / f'records-{side}.bin.gz'
            if f.exists():
                files.append((f, 0.0 if r['winner'] is None else (1.0 if r['winner'] == side else -1.0)))
        games += 1
        if max_games and games >= max_games:
            break
    return files, games


def load_files(files, baby_keep=1.0, seed=0, tag=0):
    """Load records of several files into arrays (drops records with a single legal action)."""
    parts = []
    for i, (f, v) in enumerate(files):
        h, o, m = read_records(f)
        keep = (unpack_mask(m, N_ACTIONS).sum(1) > 1) & (h['label'] < N_ACTIONS)   # legacy-index labels (resign) dropped
        if baby_keep < 1:
            rng = np.random.default_rng([seed, tag, i])
            keep &= (h['king'] != 0) | rare_state(o) | (rng.random(len(keep)) < baby_keep)
        parts.append(dict(obs=o[keep], mask=m[keep], label=h['label'][keep].astype(np.int64),
                          param=h['param'][keep].astype(np.int64) & 0xffffffff,
                          value=np.full(int(keep.sum()), v, np.float32), king=h['king'][keep].astype(bool)))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None


DEV = torch.device('cpu')


def run_batch(net, data, idx):
    x = torch.from_numpy(data['obs'][idx]).to(DEV, non_blocking=True)
    legal = torch.from_numpy(unpack_mask(data['mask'][idx], N_ACTIONS)).to(DEV, non_blocking=True)
    y = torch.from_numpy(data['label'][idx]).to(DEV, non_blocking=True)
    logits, value, h = net(x)
    return logits.masked_fill(~legal, -1e9), value, h, y


def param_loss(net, h, y, param):
    """Mean BCE over the bits each labelled parameter uses (0 if no record in the batch takes a parameter)."""
    kind = KIND_OF.to(y.device)[y]
    sel = kind != P_NONE
    if not sel.any():
        return h.new_zeros(()), None
    pl = net.param_logits(h[sel], y[sel])
    target = ((torch.as_tensor(param).to(y.device)[sel].unsqueeze(1) >> torch.arange(PARAM_MAX_BITS, device=y.device)) & 1).float()
    used = KIND_BITS.to(y.device)[kind[sel]]
    bce = F.binary_cross_entropy_with_logits(pl, target, reduction='none')
    return (bce * used).sum() / used.sum(), (pl, target, used, kind[sel])


@torch.no_grad()
def evaluate(net, data, bs=2048):
    net.eval()
    n = len(data['label'])
    tot = Counter(); correct = Counter(); loss = 0.0; vloss = 0.0
    ptot = Counter(); pexact = Counter(); pbits = Counter(); pbtot = Counter()
    for i in range(0, n, bs):
        idx = np.arange(i, min(n, i + bs))
        logits, value, h, y = run_batch(net, data, idx)
        v = torch.from_numpy(data['value'][idx]).to(DEV)
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        vloss += F.mse_loss(value, v, reduction='sum').item()
        pred = logits.argmax(1).cpu().numpy()
        for yy, pp, kk in zip(data['label'][idx], pred, data['king'][idx]):
            g = ('king:' if kk else 'baby:') + action_group(int(yy))
            tot[g] += 1; correct[g] += int(yy == pp)
            tot['all'] += 1; correct['all'] += int(yy == pp)
        _, extra = param_loss(net, h, y, data['param'][idx])
        if extra is not None:
            pl, target, used, kind = extra
            ok = ((pl > 0).float() == target).float() * used
            ok, used = ok.cpu(), used.cpu()
            for g, (o, u, k) in zip(range(len(kind)), zip(ok, used, kind.tolist())):
                name = ['', 'amount', 'squeak', 'value'][k]
                ptot[name] += 1; pexact[name] += int(o.sum().item() == u.sum().item())
                pbits[name] += o.sum().item(); pbtot[name] += u.sum().item()
    net.train()
    return {'loss': loss / max(n, 1), 'value_mse': vloss / max(n, 1), 'acc': correct['all'] / max(n, 1),
            'by_group': {g: [round(correct[g] / tot[g], 3), tot[g]] for g in sorted(tot) if g != 'all'},
            'param_exact': {k: [round(pexact[k] / ptot[k], 3), ptot[k]] for k in sorted(ptot)},
            'param_bit_acc': {k: round(pbits[k] / pbtot[k], 4) for k in sorted(pbtot)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='teacher run dir, or several separated by commas')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--value-weight', type=float, default=0.25)
    ap.add_argument('--param-weight', type=float, default=1.0)
    ap.add_argument('--h1', type=int, default=384)
    ap.add_argument('--h2', type=int, default=256)
    ap.add_argument('--hm', type=int, default=64)
    ap.add_argument('--hq', type=int, default=64)
    ap.add_argument('--king-weight', type=float, default=4.0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--max-games', type=int)
    ap.add_argument('--val-games', type=int, default=25)
    ap.add_argument('--chunk-files', type=int, default=80, help='record files loaded and shuffled together')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', help="'cpu' or 'cuda'")
    ap.add_argument('--baby-keep', type=float, default=1.0, help='fraction of ordinary baby train records kept')
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    global DEV
    DEV = torch.device(a.device)
    runs, out = [Path(r) for r in a.run.split(',')], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train_files, train_games = game_files(runs, 'train', a.max_games)
    val_files, val_games = game_files(runs, 'val', a.val_games)
    val = load_files(val_files)
    est = int(sum(_gz_records(f) for f, _ in train_files) * (1 if a.baby_keep >= 1 else min(1, a.baby_keep * 1.6)))
    print(f"train {len(train_files)} files/{train_games} games (~{est} recs), val {len(val['label'])}/{val_games} games, "
          f"loaded in {time.time()-t0:.0f}s", flush=True)
    net = StudentNet(a.h1, a.h2, a.hm, a.hq)
    export(net, out / 'untrained.bin')
    net.to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    total = max(1, a.epochs * est // a.bs)
    def lr_at(step):   # 5% linear warmup, then cosine to 2% of max
        p = min(1.0, step / total)
        return a.lr * (p / 0.05 if p < 0.05 else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (p - 0.05) / 0.95)))
    rng = np.random.default_rng(a.seed)
    log = open(out / 'train.jsonl', 'a')
    best, step = None, 0
    for ep in range(a.epochs):
        run_loss, nb = 0.0, 0
        order = rng.permutation(len(train_files))
        for c in range(0, len(order), a.chunk_files):
            data = load_files([train_files[i] for i in order[c:c + a.chunk_files]], a.baby_keep, a.seed, ep * 100000 + c)
            if data is None:
                continue
            n = len(data['label'])
            perm = rng.permutation(n)
            for i in range(0, n, a.bs):
                idx = np.sort(perm[i:i + a.bs])
                logits, value, h, y = run_batch(net, data, idx)
                v = torch.from_numpy(data['value'][idx]).to(DEV); k = torch.from_numpy(data['king'][idx]).to(DEV)
                w = 1.0 + (a.king_weight - 1.0) * k.float()
                pl, _ = param_loss(net, h, y, data['param'][idx])
                loss = ((F.cross_entropy(logits, y, reduction='none') * w).sum() / w.sum()
                        + a.value_weight * F.mse_loss(value, v) + a.param_weight * pl)
                for g in opt.param_groups:
                    g['lr'] = lr_at(step)
                opt.zero_grad(); loss.backward(); opt.step(); step += 1
                run_loss += loss.detach(); nb += 1
                if step % 500 == 0:
                    print(f'ep {ep} step {step}/{total} loss {float(run_loss)/nb:.4f} {time.time()-t0:.0f}s', flush=True)
                    run_loss, nb = 0.0, 0
            del data
        ev = evaluate(net, val)
        ev.update(epoch=ep, step=step, seconds=round(time.time() - t0))
        log.write(json.dumps(ev) + '\n'); log.flush()
        print(f"epoch {ep}: val loss {ev['loss']:.4f} acc {ev['acc']:.4f} value_mse {ev['value_mse']:.3f} "
              f"param_exact {ev['param_exact']}", flush=True)
        export(net, out / f'epoch{ep}.bin')
        if best is None or ev['loss'] < best:
            best = ev['loss']
            export(net, out / 'best.bin'); torch.save(net.state_dict(), out / 'best.pt')
            (out / 'best.json').write_text(json.dumps(ev, indent=1))
    print('done', flush=True)


if __name__ == '__main__':
    main()
