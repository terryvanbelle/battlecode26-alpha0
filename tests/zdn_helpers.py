"""Shared helpers for the zdn tests: a record encoder (mirror of java/bots/zdn/ZE.java), a packed-net reader and the
net's math (mirrors of java/bots/zdn/ZNet.java), and a small random net with its standardisation."""
import gzip, struct
import numpy as np

NONE = 0xFFFF
OFF = 32768


def record(robot, rnd, point, label, feats, team=0, king=False):
    """One decisions-v1 record: the 14-byte header and the uint16 features."""
    feats = [int(v) for v in feats]
    assert all(0 <= v <= 0xFFFF for v in feats)
    head = struct.pack('>iHBBHHH', robot, rnd, team, int(king), point, NONE if label is None or label < 0 else label, len(feats))
    return head + struct.pack(f'>{len(feats)}H', *feats)


def reg_feats(S, L, regs):
    """ZE.reg's features. regs: [(kind [K], x [K][D], lab or -1, D, nk)]."""
    f = [1, len(S), len(L), len(regs)] + [v + OFF for v in S] + list(L)
    for kind, x, lab, D, nk in regs:
        f += [len(kind), NONE if lab < 0 else lab, D, nk]
        for k, row in zip(kind, x):
            f += [k] + [v + OFF for v in row]
    return f


def step_feats(step, ctx, kind, x, D, nk):
    """ZE.step's features (the label goes in the header)."""
    f = [1, step, len(ctx), D, len(kind), nk] + [v + OFF for v in ctx]
    for k, row in zip(kind, x):
        f += [k] + [v + OFF for v in row]
    return f


def error_feats(text):
    return [ord(c) for c in text]


def write(path, records):
    path.write_bytes(gzip.compress(b''.join(records)))


def unpack(chars):
    """Read a packed net (zdn_pack.pack's chars) exactly the way ZNet.java's constructor does."""
    pos = 0

    def ch():
        nonlocal pos
        pos += 1
        return chars[pos - 1]

    def fl(n):
        nonlocal pos
        a = np.array(chars[pos:pos + 2 * n], np.uint32)
        pos += 2 * n
        return ((a[0::2] << 16) | a[1::2]).astype(np.uint32).view(np.float32).astype(np.float64)

    assert ch() == 0x5A4E
    nS, nL, H1, H, Hh = ch(), ch(), ch(), ch(), ch()
    nC, nK, D, nR = ch(), ch(), ch(), ch()
    rdims = [(ch(), ch(), ch()) for _ in range(nR)]
    net = dict(nS=nS, nL=nL, H1=H1, H=H, Hh=Hh, rdims=rdims)
    nIn = nS + nL
    net['W1'] = fl(H1 * nIn).reshape(H1, nIn)
    net['b1'] = fl(H1)
    net['W2'] = fl(H * H1).reshape(H, H1)
    net['b2'] = fl(H)

    def scorer(nC, nK, D):
        s = dict(nC=nC, nK=nK, D=D)
        s['A'] = fl(Hh * H).reshape(Hh, H)
        s['a'] = fl(Hh)
        s['C'] = fl(Hh * nC).reshape(Hh, nC) if nC else None
        s['B'] = fl(nK * Hh * D).reshape(nK, Hh, D)
        s['E'] = fl(nK * Hh).reshape(nK, Hh)
        s['w'] = fl(Hh)
        s['w0'] = fl(1)[0]
        return s

    net['step'] = scorer(nC, nK, D)
    net['regs'] = [scorer(c, k, d) for k, d, c in rdims]
    assert pos == len(chars), 'trailing chars'
    return net


def ref_logits(net, x, ctx, kind, xs, head):
    """ZNet.java's trunk + logits on raw inputs (float64). head -1: the step scorer; r: register r."""
    h1 = np.maximum(net['W1'] @ np.asarray(x, np.float64) + net['b1'], 0)
    h = np.maximum(net['W2'] @ h1 + net['b2'], 0)
    s = net['step'] if head < 0 else net['regs'][head]
    u = s['A'] @ h + s['a']
    if s['C'] is not None:
        u = u + s['C'] @ np.asarray(ctx, np.float64)
    z = []
    for k, xx in zip(kind, xs):
        if k < 0 or k >= s['nK']:
            z.append(-np.inf)
            continue
        t = u + s['E'][k] + s['B'][k] @ np.asarray(xx, np.float64)
        z.append(float(s['w'] @ np.maximum(t, 0) + s['w0']))
    return np.array(z)


def random_net(seed=0, nS=5, nL=7, nC=4, step=(6, 3), regs=((3, 2), (4, 1), (2, 5))):
    """A small zdn_train.ZNet with every parameter random, plus random standardisation statistics."""
    import torch
    import zdn_train
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = zdn_train.ZNet(nS, nL, nC, step, list(regs), H1=9, H=8, Hh=6)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn_like(p) * 0.6)
    f32 = lambda a: np.asarray(a, np.float32)
    st = dict(nS=nS, nL=nL, nC=nC, mu_SL=f32(rng.normal(0, 3, nS + nL)), sd_SL=f32(rng.uniform(0.5, 3, nS + nL)),
              mu_c=f32(rng.normal(0, 1, nC)), sd_c=f32(rng.uniform(0.5, 2, nC)),
              kst_step=(f32(rng.normal(0, 5, (step[0], step[1]))), f32(rng.uniform(0.5, 4, (step[0], step[1])))),
              kst_reg=[(f32(rng.normal(0, 5, (k, d))), f32(rng.uniform(0.5, 4, (k, d)))) for k, d in regs])
    return net, st


def torch_logits(net, st, x, ctx, kind, xs, head):
    """The trained net's own forward pass on standardised inputs (what training optimises)."""
    import torch
    x = (np.asarray(x, np.float32) - st['mu_SL']) / st['sd_SL']
    kind = np.asarray(kind)
    mu, sd = st['kst_step'] if head < 0 else st['kst_reg'][head]
    k = np.maximum(kind, 0)
    xs = (np.asarray(xs, np.float32) - mu[k]) / sd[k]
    with torch.no_grad():
        h = net.h(torch.from_numpy(x[None]))
        if head < 0:
            c = torch.from_numpy(((np.asarray(ctx, np.float32) - st['mu_c']) / st['sd_c'])[None])
            sc = net.step
        else:
            c = torch.from_numpy(np.asarray(ctx, np.float32)[None])
            sc = net.regs[head]
        z = sc(h, c, torch.from_numpy(kind[None]).long(), torch.from_numpy(xs[None].astype(np.float32)))
    return z[0].double().numpy()


def export_packed(net, st, path):
    """zdn_train.export the net, then zdn_pack.pack it: (net.json dict, chars)."""
    import json
    import zdn_pack, zdn_train
    zdn_train.export(net, path, st['mu_SL'], st['sd_SL'], st['mu_c'], st['sd_c'], st['kst_step'], st['kst_reg'],
                     dict(unit='baby', nS=st['nS'], nL=st['nL'], nC=st['nC']))
    j = json.loads(path.read_text())
    return j, zdn_pack.pack(j)
