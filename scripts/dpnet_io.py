"""Decision-net I/O and the bot's exact sampling math (shared by train_dp.py, rl_data.py and rl_train.py).

    from dpnet_io import load_net, forward, probs, unfold, fold, save_net, load_norm
    nets = load_net('runs/dp/net-v2/net.json')            # {point: [dict(w=float32[out,in], b=float32[out]), ...]}
    z = forward(nets[5], X)                                  # logits, float32, same arithmetic as DNet.logits
    p = probs(z, mask, T=1.0, floor=0.005)                   # p[] exactly as DNet.sample computes it

    python3 scripts/dpnet_io.py NET.json [NET2.json]         # summary; with two nets: which points differ

net.json (train_dp.py / rl_train.py): {"version", "hidden", "points": {"<point>": [layer, ...]}}, layer =
{n_in, n_out, w: row-major [out][in] list, b: list}; input normalisation is folded into layer 1 and build_dpnet.py
packs the values as float32. norm.npz next to net.json holds the (mu, sd) used for folding (keys mu_<p>, sd_<p>).
Folding: W1f = W1 / sd, b1f = b1 - W1 @ (mu / sd). It is exact for any (mu, sd), so unfold/fold round-trips.
"""
import json, sys
from pathlib import Path
import numpy as np

F32 = np.float32


def load_raw(path):
    """The parsed net.json dict (values untouched, so frozen points can be written back byte-identically)."""
    return json.loads(Path(path).read_text())


def layers_from_raw(raw_layers):
    """JSON layers -> [dict(w=float32[out,in], b=float32[out])], cast exactly as build_dpnet.pack does."""
    out = []
    for l in raw_layers:
        w = np.asarray(l['w'], dtype=np.float64).astype(F32).reshape(l['n_out'], l['n_in'])
        b = np.asarray(l['b'], dtype=np.float64).astype(F32)
        out.append(dict(w=w, b=b))
    return out


def load_net(path):
    """{point (int): [dict(w=float32[out, in], b=float32[out]), ...]} from a net.json."""
    raw = load_raw(path)
    return {int(p): layers_from_raw(v) for p, v in raw['points'].items()}


def load_norm(path):
    """{point: (mu float32[in], sd float32[in])} from norm.npz (path to the file or to the net.json beside it);
    {} if missing."""
    path = Path(path)
    if path.suffix == '.json':
        path = path.parent / 'norm.npz'
    if not path.exists():
        return {}
    z = np.load(path)
    return {int(k[3:]): (z[k], z['sd_' + k[3:]]) for k in z.files if k.startswith('mu_')}


def save_norm(path, norm):
    """Write norm.npz from {point: (mu, sd)}."""
    arrs = {}
    for p, (mu, sd) in sorted(norm.items()):
        arrs[f'mu_{p}'] = np.asarray(mu); arrs[f'sd_{p}'] = np.asarray(sd)
    with open(path, 'wb') as fh:
        np.savez(fh, **arrs)


def forward(layers, X, exact=True, chunk=65536):
    """Logits float32 [N, out] for raw features X [N, nf] (any numeric dtype). Same math as DNet.logits: x[k] = F[4+k]
    for k < min(n_in, nf), else 0; per output s = b[o], then s += w[o,k] * x[k] for k ascending, each product and sum
    rounded to float32 (Java has no FMA here); ReLU on hidden layers. exact=False uses BLAS (faster, ~1e-6 off)."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[None, :]
    n_in = layers[0]['w'].shape[1]
    N = X.shape[0]
    x0 = np.zeros((N, n_in), dtype=F32)
    m = min(n_in, X.shape[1])
    x0[:, :m] = X[:, :m].astype(F32)
    out = np.empty((N, layers[-1]['w'].shape[0]), dtype=F32)
    for s in range(0, max(N, 1), chunk):
        x = x0[s:s + chunk]
        for li, l in enumerate(layers):
            w, b = l['w'], l['b']
            if exact:
                y = np.broadcast_to(b, (x.shape[0], w.shape[0])).copy()
                wt = np.ascontiguousarray(w.T)
                for k in range(w.shape[1]):
                    y += x[:, k:k + 1] * wt[k][None, :]
            else:
                y = (x @ w.T + b).astype(F32)
            if li < len(layers) - 1:
                y[y < 0] = 0
            x = y
        out[s:s + chunk] = x
    return out


def fit_mask(mask, N, n):
    """Allowed mask bool [N, n] as D.fit sizes it (None: all allowed; extra options not allowed; extra bits dropped)."""
    if mask is None:
        return np.ones((N, n), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 1:
        mask = np.broadcast_to(mask, (N, mask.shape[0]))
    if mask.shape[1] == n:
        return mask
    out = np.zeros((N, n), dtype=bool)
    m = min(n, mask.shape[1])
    out[:, :m] = mask[:, :m]
    return out


def probs(logits, mask=None, T=1.0, floor=0.0):
    """Probabilities [N, n] exactly as DNet.sample fills p[]: masked softmax of logits / T (relative to the best allowed
    logit, exp in double then cast to float, float32 running total), times (1 - floor), plus floor / m on each of the m
    allowed options. T <= 0: the argmax gets 1 - floor. Rows with nothing allowed are all zero.
    numpy input: float32 result, bit-faithful to Java's arithmetic order. torch input (logits a tensor, mask a bool
    tensor or None): a differentiable float32 version of the same formula (for training; equal to ~1 ulp)."""
    try:
        import torch
        if isinstance(logits, torch.Tensor):
            return _probs_torch(logits, mask, T, floor)
    except ImportError:
        pass
    z = np.asarray(logits, dtype=F32)
    if z.ndim == 1:
        z = z[None, :]
    N, n = z.shape
    allowed = fit_mask(mask, N, n)
    m = allowed.sum(1)
    zm = np.where(allowed, z, -np.inf)
    best = np.argmax(zm, axis=1)   # first maximum, like Java's strict '>'
    zb = z[np.arange(N), best]
    one_f = F32(1) - F32(floor)
    if T > 0:
        d = np.where(allowed, (z - zb[:, None]) / F32(T), -np.inf)   # float arithmetic; exp of allowed options only
        e = np.exp(d.astype(np.float64)).astype(F32)            # (float) Math.exp(double)
        tot = np.cumsum(e, axis=1, dtype=F32)[:, -1]            # sequential float32 sum, k ascending
        with np.errstate(invalid='ignore', divide='ignore'):
            p = (e / tot[:, None]) * one_f
    else:
        p = np.zeros((N, n), dtype=F32)
        p[np.arange(N), best] = one_f
    with np.errstate(invalid='ignore', divide='ignore'):
        add = F32(floor) / m.astype(F32)                        # floor / m in float
    p = np.where(allowed, p + add[:, None], F32(0)).astype(F32)
    p[m == 0] = 0
    return p


def _probs_torch(z, mask, T, floor):
    import torch
    if mask is None:
        mask = torch.ones_like(z, dtype=torch.bool)
    m = mask.sum(1, keepdim=True).clamp(min=1).to(z.dtype)
    zm = torch.where(mask, z, torch.full_like(z, -float('inf')))
    zb = zm.max(1, keepdim=True).values.detach()
    if T > 0:
        d = torch.where(mask, (z - zb) / T, torch.full_like(z, -1e4))   # masked options never reach exp (no inf/nan grads)
        e = torch.where(mask, torch.exp(d.double()).to(z.dtype), torch.zeros_like(z))
        p = e / e.sum(1, keepdim=True) * (1 - floor)
    else:
        p = torch.zeros_like(z)
        p.scatter_(1, zm.argmax(1, keepdim=True), 1 - floor)
    return torch.where(mask, p + floor / m, torch.zeros_like(p))


def unfold(layers, mu, sd):
    """Folded layers -> layers acting on (x - mu) / sd (float64): W1 = W1f * sd, b1 = b1f + W1f @ mu."""
    out = [dict(w=np.asarray(l['w'], np.float64).copy(), b=np.asarray(l['b'], np.float64).copy()) for l in layers]
    w1f = out[0]['w']
    mu = np.asarray(mu, np.float64); sd = np.asarray(sd, np.float64)
    out[0]['b'] = out[0]['b'] + w1f @ mu
    out[0]['w'] = w1f * sd[None, :]
    return out


def fold(layers, mu, sd):
    """Layers acting on (x - mu) / sd -> folded layers (train_dp.py's export math): W1 / sd, b1 - W1 @ (mu / sd)."""
    out = [dict(w=np.asarray(l['w'], np.float64).copy(), b=np.asarray(l['b'], np.float64).copy()) for l in layers]
    W, b = out[0]['w'], out[0]['b']
    out[0]['b'] = b - (W * (mu / sd)).sum(1)
    out[0]['w'] = W / sd
    return out


def layer_json(l):
    """A layer as net.json stores it (floats written at full float64 precision, as train_dp.py does)."""
    W = np.asarray(l['w'], np.float64); b = np.asarray(l['b'], np.float64)
    return dict(n_in=int(W.shape[1]), n_out=int(W.shape[0]), w=W.ravel().tolist(), b=b.tolist())


def save_net(path, nets, meta=None):
    """Write net.json. nets: {point: layers}, layers either array dicts (w [out,in], b) or raw JSON layer dicts
    (with n_in/n_out, written back unchanged). meta: the other top-level keys (default version dpnet-v1, hidden)."""
    meta = dict(meta or {})
    pts = {}
    for p, layers in nets.items():
        pts[str(p)] = [l if 'n_in' in l else layer_json(l) for l in layers]
    doc = {'version': meta.pop('version', 'dpnet-v1'), 'hidden': meta.pop('hidden', None), **meta, 'points': pts}
    if doc['hidden'] is None:
        any_l = next(iter(nets.values()), [])
        doc['hidden'] = [int((l['n_out'] if 'n_in' in l else np.shape(l['w'])[0])) for l in any_l[:-1]]
    Path(path).write_text(json.dumps(doc))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    a = load_net(sys.argv[1])
    for p, ls in sorted(a.items()):
        print(f'point {p}: ' + ' -> '.join(str(l['w'].shape[1]) for l in ls) + f' -> {ls[-1]["w"].shape[0]}')
    if len(sys.argv) > 2:
        ra, rb = load_raw(sys.argv[1])['points'], load_raw(sys.argv[2])['points']
        for p in sorted(set(ra) | set(rb), key=int):
            print(f'point {p}: {"same" if ra.get(p) == rb.get(p) else "DIFFERENT"}')


if __name__ == '__main__':
    main()
