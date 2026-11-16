"""T0: Java DNet.sample / floorWeights against dpnet_io.probs / floor_weights (numpy bit-exact, torch ~1e-6).
  t0.py gen OUT_ROWS GROUPS_JSON   |   t0.py check ROWS JAVA_OUT GROUPS_JSON"""
import sys, json
import numpy as np
sys.path.insert(0, 'scripts')
import dpnet_io
F32 = np.float32

def gen(out, gpath):
    rng = np.random.default_rng(7)
    groups = dpnet_io.load_groups(gpath)
    lines = []
    for i in range(100000):
        p = int(rng.integers(0, 5))                    # points 0-3 have groups, 4 has none
        n = len(groups[p][0]) if p in groups else 22
        n = n + int(rng.integers(0, 3)) if p == 3 else n   # point 3: options beyond len(g) are dormant
        T = float(rng.choice([1.0, 1.0, 0.5, 2.0, 0.0]))
        fl = float(rng.choice([0.0, 0.005, 0.2, 0.2, 0.5]))
        m = rng.random(n) < rng.choice([0.2, 0.6, 1.0])
        if rng.random() < 0.01:
            m[:] = False
        z = (rng.standard_normal(n) * rng.choice([1.0, 5.0, 30.0])).astype(F32)
        if rng.random() < 0.1:
            z[:] = z[0]                                # ties
        bits = ' '.join(str(int(v)) for v in z.view(np.int32))
        lines.append(f"{p} {T} {fl} {''.join('1' if b else '0' for b in m)} {bits}")
    open(out, 'w').write('\n'.join(lines) + '\n')

def check(rows, jout, gpath):
    groups = dpnet_io.load_groups(gpath)
    nbad = nfw = nrow = 0
    worst_t = 0.0
    import torch
    for line, res in zip(open(rows), open(jout)):
        t = line.split(); r = res.split()
        p, T, fl, ms = int(t[0]), float(t[1]), float(t[2]), t[3]
        n = len(ms)
        m = np.array([c == '1' for c in ms])[None, :]
        z = np.array([int(v) for v in t[4:4 + n]], np.int32).view(F32)[None, :]
        g = groups.get(p)
        k, pbits = int(r[0]), int(r[1])
        fwj = None if r[2] == 'null' else np.array([int(v) for v in r[2:2 + n]], np.int32).view(F32)
        fwn = dpnet_io.floor_weights(m, g, n)
        nrow += 1
        if (fwj is None) != (fwn is None) or (fwj is not None and not np.array_equal(fwj, fwn[0])):
            nfw += 1
            if nfw <= 5: print('FW MISMATCH', line[:80], fwj, None if fwn is None else fwn[0])
        if k < 0:
            if m.any(): nbad += 1; print('k<0 with allowed options', line[:80])
            continue
        if T <= 0 and (fl <= 0 or (fwn is not None and fwn.sum() <= 0)):
            continue                                   # Java returns the argmax without computing p
        pn = dpnet_io.probs(z, m, T, fl, g)[0]
        if pn[k].view(np.int32) != np.int32(pbits).astype(np.int32) and np.array(pbits, np.int64).astype(np.uint32).view(np.int32) != pn[k].view(np.int32):
            nbad += 1
            if nbad <= 5: print('P MISMATCH', line[:80], k, np.int32(pbits).view(F32) if -2**31 <= pbits < 2**31 else pbits, pn[k])
        pt = dpnet_io.probs(torch.tensor(z), torch.tensor(m), T, fl, g).numpy()[0]
        worst_t = max(worst_t, float(np.abs(pt - pn).max()))
    print(f'rows {nrow}: floorWeights mismatches {nfw}, p mismatches {nbad}, torch max |dp| {worst_t:.2e}')

if __name__ == '__main__':
    {'gen': gen, 'check': check}[sys.argv[1]](*sys.argv[2:])
