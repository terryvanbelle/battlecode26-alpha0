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


def reg_feats(S, L, regs, ver=2, version=-1, q=None):
    """ZE.reg's features. regs: [(kind [K], x [K][D], lab or -1, D, nk)]; VER 2 adds the trailer (version, q per
    register; -1 / None: none)."""
    f = [ver, len(S), len(L), len(regs)] + [v + OFF for v in S] + list(L)
    for kind, x, lab, D, nk in regs:
        f += [len(kind), NONE if lab < 0 else lab, D, nk]
        for k, row in zip(kind, x):
            f += [k] + [v + OFF for v in row]
    if ver >= 2:
        f += [NONE if version < 0 else version] + [NONE if q is None or q[r] < 0 else q[r] for r in range(len(regs))]
    return f


def step_feats(step, ctx, kind, x, D, nk, ver=2, q=-1):
    """ZE.step's features (the label goes in the header); VER 2 adds q (-1: none)."""
    f = [ver, step, len(ctx), D, len(kind), nk] + [v + OFF for v in ctx]
    for k, row in zip(kind, x):
        f += [k] + [v + OFF for v in row]
    if ver >= 2:
        f.append(NONE if q < 0 else q)
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


def q_of(logp):
    """ZE.q: round(-4096 log p), capped."""
    return int(min(65534, round(-4096 * logp)))


def synth_records(rng, nS=5, nL=7, nC=4, step=(6, 3), regs=((3, 2), (4, 1), (2, 5)), turns=30, robots=4, king=False,
                  policy=None, version=0, T=1.0, max_steps=3):
    """Player-like REG/STEP records of random turns (VER 2). Choices are sampled from policy (an unpack()ed net: its
    softmax at T over ref_logits, with the register context of the earlier choices) or uniformly; q is the sampled
    choice's log-probability as ZE writes it. Returns (records, turns): turns = [dict(robot, round, S, L, regs=[(kind,
    x, lab)], steps=[(ctx, kind, x, lab)])]."""
    pr, ps = (210, 211) if king else (200, 201)
    nks = [k for k, _ in regs]
    off = np.concatenate([[0], np.cumsum(nks)])

    def pick(head, x, ctx, kind, xs):
        if policy is None:
            return int(rng.integers(len(kind))), -np.log(len(kind))
        z = ref_logits(policy, x, ctx, kind, xs, head) / T
        p = np.exp(z - z.max()); p /= p.sum()
        i = int(rng.choice(len(kind), p=p))
        return i, float(np.log(p[i]))

    recs, out = [], []
    for t in range(turns):
        robot, rnd = 10 + t % robots, 1 + t // robots
        S, L = rng.integers(-20, 20, nS), rng.integers(0, 50, nL)
        x = np.concatenate([S, L])
        rctx = np.zeros(off[-1])
        rg, qs = [], []
        for j, (nK, D) in enumerate(regs):
            K = int(rng.integers(1, 5))
            kind, xs = rng.integers(0, nK, K), rng.integers(-9, 9, (K, D))
            lab, lp = pick(j, x, rctx, kind, xs) if K > 1 else (0, 0.0)
            rctx[off[j] + kind[lab]] = 1
            rg.append((kind, xs, lab)); qs.append(q_of(lp))
        recs.append(record(robot, rnd, pr, None, reg_feats(S, L, [(k, xs, lab, xs.shape[1], nK) for (k, xs, lab), (nK, _) in zip(rg, regs)],
                                                             version=version, q=qs), king=king))
        st = []
        for s in range(int(rng.integers(1, max_steps + 1))):
            K = int(rng.integers(1, 6))
            ctx, kind, xs = rng.integers(-3, 4, nC), rng.integers(0, step[0], K), rng.integers(-9, 9, (K, step[1]))
            lab, lp = pick(-1, x, ctx, kind, xs) if K > 1 else (0, 0.0)
            recs.append(record(robot, rnd, ps, lab, step_feats(s, ctx, kind, xs, step[1], step[0], q=q_of(lp)), king=king))
            st.append((ctx, kind, xs, lab))
        out.append(dict(robot=robot, round=rnd, S=S, L=L, regs=rg, steps=st))
    return recs, out
