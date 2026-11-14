"""Make a net-mode zdp bot from trained decision nets.

build_dpnet.py --net runs/dp/net-v1/net.json --name zdpn1 [--temperature 1.0] [--floor 0.001] [--points 1,2] [--cfg A=x,B=y]
build_dpnet.py --name zdpoff --cfg FIX_X=false,...      (no --net: a plain zdp variant, D.NET stays false)
build_dpnet.py --net runs/rl/r1/nets/3/net.json --name zdpr3 --floor 0.005 --rl --version 3 [--run r1]
    (RL build: D.RL on, every record carries the sampling trailer with NET_VERSION 3; --run also copies net.json to
    runs/rl/<run>/nets/<version>/net.json, which the parity check reads)

Writes build/botsrc/NAME: zdp's sources in package NAME with D.NET on, the sampling settings, and W.java holding
the packed nets (see DNet). eval_student.py and replay_check.py then accept NAME as a player or opponent.
"""
import argparse, json, re, shutil, struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'java/bots/zdp'


def pack(layers):
    import numpy as np
    """DNet's char layout: nLayers, (in, out) per layer, then per layer the weights ([out][in]) and biases as float32
    bits, two chars each (high first)."""
    chars = [len(layers)]
    for l in layers:
        chars += [l['n_in'], l['n_out']]
    for l in layers:
        for key in ('w', 'b'):
            bits = np.asarray(l[key], dtype='>f4').view('>u4').astype(np.int64)
            chars += np.stack([bits >> 16, bits & 0xFFFF], 1).ravel().tolist()
    return chars


# javac expands \\uXXXX before lexing, so these four must use ordinary escapes inside a literal
SAFE = {0x0a: '\\n', 0x0d: '\\r', 0x22: '\\"', 0x5c: '\\\\'}


def java_string(chars, chunk=8000):
    parts = []
    for i in range(0, len(chars), chunk):
        parts.append('"' + ''.join(SAFE.get(c, f'\\u{c:04x}') for c in chars[i:i + chunk]) + '"')
    return 'j(' + ', '.join(parts) + ')'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', default='')
    ap.add_argument('--cfg', default='', help='Cfg constant overrides NAME=value,... (value as Java source)')
    ap.add_argument('--name', required=True)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--floor', type=float, default=0.001)
    ap.add_argument('--points', default='', help='only these point ids get a net (default: all in net.json)')
    ap.add_argument('--rl', action='store_true', help='RL build: D.RL on (each record carries the sampling trailer)')
    ap.add_argument('--version', type=int, default=0, help='D.NET_VERSION written into each trailer (0..65535)')
    ap.add_argument('--run', default='', help='RL run name: copy net.json to runs/rl/RUN/nets/VERSION/net.json')
    a = ap.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]*', a.name):
        raise SystemExit('name must be a lowercase Java package name')
    if (a.rl or a.run) and not a.net:
        raise SystemExit('--rl/--run need --net')
    if not 0 <= a.version <= 65535:
        raise SystemExit('--version must fit in 16 bits')
    net = json.loads(Path(a.net).read_text()) if a.net else None
    keep = ({int(p) for p in a.points.split(',') if p} or {int(p) for p in net['points']}) if net else set()
    cfg = dict(x.split('=', 1) for x in a.cfg.split(',') if x)
    dest = ROOT / 'build/botsrc' / a.name
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for f in SRC.glob('*.java'):
        s = f.read_text().replace('package zdp;', f'package {a.name};', 1)
        if f.name == 'Cfg.java':
            for k, v in cfg.items():
                s, n = re.subn(r'(static final \w+ ' + k + r' = )[^;,]*', lambda m: m.group(1) + v, s)
                if n != 1:
                    raise SystemExit(f'Cfg.{k}: {n} matches')
        if f.name == 'D.java' and net:
            s, k = re.subn(r'static final boolean NET = false;', 'static final boolean NET = true;', s)
            if not k:
                raise SystemExit('D.java has no "static final boolean NET = false;"')
            s = re.sub(r'static final float NET_T = [^;]*;', f'static final float NET_T = {a.temperature}f;', s)
            s = re.sub(r'static final float NET_FLOOR = [^;]*;', f'static final float NET_FLOOR = {a.floor}f;', s)
            for pat, val in ((r'static final boolean RL = false;', f'static final boolean RL = {str(a.rl).lower()};'),
                             (r'static final int NET_VERSION = 0;', f'static final int NET_VERSION = {a.version};')):
                s, k = re.subn(pat, val, s)
                if k != 1:
                    raise SystemExit(f'D.java: {k} matches for "{pat}"')
        if f.name != 'W.java' or not net:
            (dest / f.name).write_text(s)
    if not net:
        (dest / 'BUILD.json').write_text(json.dumps({'cfg': cfg}, indent=1))
        print(f'wrote {dest} (no net; cfg {cfg})')
        return
    n = max(keep) + 1
    entries = ['""'] * n
    for p in keep:
        entries[p] = java_string(pack(net['points'][str(p)]))
    (dest / 'W.java').write_text(
        f'package {a.name};\n\n/** Generated by scripts/build_dpnet.py from {a.net}. */\nfinal class W {{\n'
        f'    static String j(String... s) {{ StringBuilder b = new StringBuilder(); for (String x : s) b.append(x); return b.toString(); }}\n'
        f'    static final String[] P = {{\n        ' + ',\n        '.join(entries) + '\n    };\n\n    private W() {}\n}\n')
    copy = ''
    if a.run:
        cp = ROOT / 'runs/rl' / a.run / 'nets' / str(a.version) / 'net.json'
        cp.parent.mkdir(parents=True, exist_ok=True)
        if cp.resolve() != Path(a.net).resolve():
            shutil.copyfile(a.net, cp)
        copy = str(cp.relative_to(ROOT))
    (dest / 'BUILD.json').write_text(json.dumps({'net': a.net, 'temperature': a.temperature, 'floor': a.floor, 'cfg': cfg,
                                                 'points': sorted(keep), 'rl': a.rl, 'version': a.version,
                                                 'run': a.run or None, 'net_copy': copy or None}, indent=1))
    print(f'wrote {dest} (points {sorted(keep)}{", RL version %d" % a.version if a.rl else ""})')


if __name__ == '__main__':
    main()
