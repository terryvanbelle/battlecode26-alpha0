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


NEW_POINTS = (8, 9, 14, 15, 16)   # pass 3's new points (BABY_FREE, BABY_VAL, KING_COMMS, KING_VAL, KING_TURN)
VALUE_POINTS = (9, 15)            # value values: no zdb label, uniform start
R_KEEP_POINT, R_KEEP = 5, 5


def kaiming(rng, n_in, n_out):
    """A ReLU hidden layer (Kaiming-uniform weights, zero bias), float64."""
    bound = np.sqrt(6.0 / n_in)
    return dict(w=rng.uniform(-bound, bound, (n_out, n_in)), b=np.zeros(n_out))


def widen(raw, opts, feats, active='old', calib=None, floor=0.005, init_new=None, keep_bias=None, hidden=(64, 64),
          seed=0, points=None, norm=None, delta=None):
    """Widen a net.json dict to the Java points (pass 3, scripts/widen_dpnet.py). Returns (doc, norm, info).

    opts / feats: {point: Java option / feature count} (dp_points). For each point of raw whose net is smaller:
    zero input columns appended to layer 1 (old logits bit-identical: features are >= 0 and 0 * x adds +0), output
    rows appended with zero weights and bias b_new = median(max old logit on calib) - ln(10 m / floor) (m = the new
    option count: a new option's softmax share is about 0.1 floor / m, so its policy gradient stays alive); without
    calib records b_new = -20. BABY_ROLE's R_KEEP row instead gets p99.9 (max old logit on review records) + 8 (no
    calib: keep_bias, default 60), so KEEP dominates reviews. active: 'old' writes doc['active'][p] = the old output
    count (the build samples only those: exact replay), 'full' none, or {point: n}.
    init_new 'zdb-hard' / 'calibrated' adds nets for the new points missing from raw: (feats, *hidden, opts) MLPs,
    Kaiming-uniform hidden layers (seed), input normalisation from calib (else mu 0, sd 1) folded in, last layer zero;
    biases 0 on option 0 (zdb's choice) and -30 (zdb-hard: replays zdb) or -delta (calibrated, default
    ln(10 m / floor)) on the rest; the value points (9, 15) get all-zero biases (uniform).
    calib: {point: raw features [N, n] (any width)}. norm: {point: (mu, sd)} from norm.npz (padded here)."""
    rng = np.random.default_rng(seed)
    calib = calib or {}
    norm = dict(norm or {})
    doc = {k: v for k, v in raw.items() if k != 'points'}
    pts, info, act = {}, {}, dict(doc.get('active') or {})
    sel = set(points) if points else None
    for ps, layers in raw['points'].items():
        p = int(ps)
        n_in, n_out = layers[0]['n_in'], layers[-1]['n_out']
        tin, tout = max(n_in, feats.get(p, n_in)), max(n_out, opts.get(p, n_out))
        if (sel is not None and p not in sel) or (tin == n_in and tout == n_out):
            pts[ps] = layers
            continue
        ls = layers_from_raw(layers)
        old = [dict(w=l['w'].copy(), b=l['b'].copy()) for l in ls]
        w0 = np.zeros((ls[0]['w'].shape[0], tin), F32)
        w0[:, :n_in] = ls[0]['w']
        ls[0]['w'] = w0
        b_new = None
        if tout > n_out:
            X = calib.get(p)
            if X is not None and len(X):
                z = forward(old, np.asarray(X)[:, :n_in], exact=False).astype(np.float64)
                b_new = float(np.median(z.max(1)) - np.log(10.0 * tout / max(floor, 1e-9)))
            else:
                b_new = -20.0
            last = ls[-1]
            w = np.zeros((tout, last['w'].shape[1]), F32)
            w[:n_out] = last['w']
            b = np.full(tout, b_new, F32)
            b[:n_out] = last['b']
            if p == R_KEEP_POINT and n_out <= R_KEEP < tout:
                if X is not None and len(X) and (np.asarray(X)[:, 0] == 1).any():
                    zr = z[np.asarray(X)[:, 0] == 1]
                    b[R_KEEP] = float(np.quantile(zr.max(1), 0.999) + 8.0)
                else:
                    b[R_KEEP] = 60.0 if keep_bias is None else keep_bias
            last['w'], last['b'] = w, b
            if active == 'old':
                act[ps] = n_out
            elif isinstance(active, dict) and p in active:
                act[ps] = int(active[p])
        pts[ps] = [layer_json(l) for l in ls]
        if p in norm:
            mu, sd = (np.asarray(v, np.float64) for v in norm[p])
            X = calib.get(p)
            mu2, sd2 = np.zeros(tin), np.ones(tin)
            if X is not None and len(X) and np.asarray(X).shape[1] >= tin:
                xs = np.asarray(X, np.float64)[:, :tin]
                mu2, sd2 = xs.mean(0), np.where(xs.std(0) < 1e-3, 1.0, xs.std(0))
            mu2[:len(mu)], sd2[:len(sd)] = mu[:tin], sd[:tin]
            norm[p] = (mu2.astype(np.float32), sd2.astype(np.float32))
        info[ps] = dict(cols_added=int(tin - n_in), rows_added=int(tout - n_out), b_new=b_new,
                        r_keep_bias=float(ls[-1]['b'][R_KEEP]) if p == R_KEEP_POINT and tout > R_KEEP >= n_out else None)
    if active == 'full':
        act = {k: v for k, v in act.items() if k not in info}
    if init_new:
        for p in NEW_POINTS:
            if str(p) in pts or (sel is not None and p not in sel) or p not in opts or p not in feats:
                continue
            tin, tout = feats[p], opts[p]
            X = calib.get(p)
            if X is not None and len(X) and np.asarray(X).shape[1] >= tin:
                xs = np.asarray(X, np.float64)[:, :tin]
                mu, sd = xs.mean(0), np.where(xs.std(0) < 1e-3, 1.0, xs.std(0))
            else:
                mu, sd = np.zeros(tin), np.ones(tin)
            dims = [tin, *hidden]
            ls = [kaiming(rng, i, o) for i, o in zip(dims, dims[1:])]
            b = np.zeros(tout)
            d = None
            if p not in VALUE_POINTS:
                d = 30.0 if init_new == 'zdb-hard' else (delta if delta is not None else float(np.log(10.0 * tout / max(floor, 1e-9))))
                b[1:] = -d
            ls.append(dict(w=np.zeros((tout, dims[-1])), b=b))
            ls = fold(ls, mu, sd)
            pts[str(p)] = [layer_json(l) for l in ls]
            norm[p] = (mu.astype(np.float32), sd.astype(np.float32))
            info[str(p)] = dict(new=init_new, n_in=tin, n_out=tout, delta=d)
    doc['points'] = {k: pts[k] for k in sorted(pts, key=int)}
    if act:
        doc['active'] = {k: act[k] for k in sorted(act, key=int)}
    else:
        doc.pop('active', None)
    doc['widened'] = dict(points=info, active=active if isinstance(active, str) else {str(k): v for k, v in active.items()},
                          floor=floor, init_new=init_new, **({'from': raw['widened']} if 'widened' in raw else {}))
    return doc, norm, info


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
