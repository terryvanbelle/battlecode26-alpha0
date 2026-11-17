"""Build zdn bots (docs/zdn.md) into build/botsrc/NAME, where eval_student.py / collect_teacher.py compile them.

  build_zdn.py --teacher zdnt [--baby-pm 15] [--king-pm 100]
                                       # zdb (java/bots/zdp) + zdn perception + the recorder (zdn-teacher), one package;
                                       # records sampled turns per mille of baby / king turns
  build_zdn.py --player zdn0 [--baby runs/zdn/b1/net.json] [--king runs/zdn/k1/net.json] [--temperature 1.0]
               [--record]              # zdn alone; no net = uniform choices; --record: REG/STEP records every turn

The teacher build patches zdp's sources (each patch must match exactly once):
  RobotPlayer  the engine's RobotController is wrapped in RcLog (every primitive reaches ZT.pre/post); ZT.start after
               G.startTurn, ZT.end after the turn (and Flow.work)
  D            EMIT off (zdb's own decision records would mix with zdn's)
  Nav.goTo     ZT.goal(t): the GOAL register's label
  Baby         the movement branch (ZT.mode after moveDecision), and every exploration draw (explore, exploreHome,
               wanderAround) as an EXPL register candidate
"""
import argparse, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZDP, ZDN, ZDT = (ROOT / 'java/bots' / n for n in ('zdp', 'zdn', 'zdn-teacher'))
sys.path.insert(0, str(ROOT / 'scripts'))
import zdn_pack  # noqa: E402

TEACHER_PATCHES = {
    'RobotPlayer.java': [
        ('        G.init(rc);\n', '        ZT.init(rc);\n        G.init(new RcLog(rc));\n'),
        ('                G.startTurn();\n', '                G.startTurn();\n                ZT.start();\n'),
        ('            int bc = Clock.getBytecodeNum();\n', '            ZT.end();\n            int bc = Clock.getBytecodeNum();\n'),
    ],
    'D.java': [('static final boolean EMIT = true;', 'static final boolean EMIT = false;')],
    'Nav.java': [('    public static boolean goTo(MapLocation t) throws GameActionException {\n',
                  '    public static boolean goTo(MapLocation t) throws GameActionException {\n        ZT.goal(t);\n')],
    'Baby.java': [
        ('if (D.net(D.BABY_MOVE)) netMove(); else moveDecision();',
         'if (D.net(D.BABY_MOVE)) netMove(); else { D.lab = -1; moveDecision(); ZT.mode(D.lab); }'),
        ('                int d = Sym.known(G.idx(cx, cy)) ? 1 << 20',
         '                ZT.sample(2, cx, cy);\n                int d = Sym.known(G.idx(cx, cy)) ? 1 << 20'),
        ('                if (Sym.known(G.idx(cx, cy)) || !homeHalf(new MapLocation(cx, cy))) continue;',
         '                ZT.sample(3, cx, cy);\n                if (Sym.known(G.idx(cx, cy)) || !homeHalf(new MapLocation(cx, cy))) continue;'),
        ('            exploreTarget = new MapLocation(clampX(m.x + fu * (G.rand(2 * b + 1) - b)), clampY(m.y + fv * (G.rand(2 * b + 1) - b)));\n',
         '            exploreTarget = new MapLocation(clampX(m.x + fu * (G.rand(2 * b + 1) - b)), clampY(m.y + fv * (G.rand(2 * b + 1) - b)));\n'
         '            ZT.sample(4, exploreTarget.x, exploreTarget.y);\n'),
    ],
}


def renamed(text, src_pkg, name):
    out, n = re.subn(rf'^package {src_pkg};', f'package {name};', text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f'no "package {src_pkg};" line')
    return out


def build_teacher(name, baby_pm=15, king_pm=100, out=None):
    dest = Path(out or ROOT / 'build/botsrc') / name
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for f in sorted(ZDP.glob('*.java')):
        s = renamed(f.read_text(), 'zdp', name)
        for old, new in TEACHER_PATCHES.get(f.name, []):
            k = s.count(old)
            if k != 1:
                raise SystemExit(f'{f.name}: patch matches {k} times: {old.strip()[:80]}')
            s = s.replace(old, new)
        (dest / f.name).write_text(s)
    for f in sorted(ZDN.glob('*.java')):
        if f.name in ('RobotPlayer.java', 'ZPlayer.java') or (dest / f.name).exists():
            continue
        (dest / f.name).write_text(renamed(f.read_text(), 'zdn', name))
    for f in sorted(ZDT.glob('*.java')):
        s = renamed(f.read_text(), 'zdn', name)
        if f.name == 'ZT.java':
            old = 'static final int BABY_PER_MILLE = 15, KING_PER_MILLE = 100;'
            if s.count(old) != 1:
                raise SystemExit('ZT.java: no sampling constants')
            s = s.replace(old, f'static final int BABY_PER_MILLE = {baby_pm}, KING_PER_MILLE = {king_pm};')
        (dest / f.name).write_text(s)
    print(f'wrote {dest} (teacher: zdb + zdn recorder; sampled per mille: baby {baby_pm}, king {king_pm})')


def build_player(name, baby=None, king=None, temperature=1.0, record=False, out=None):
    dest = Path(out or ROOT / 'build/botsrc') / name
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for f in sorted(ZDN.glob('*.java')):
        if f.name == 'ZW.java':
            continue
        s = renamed(f.read_text(), 'zdn', name)
        if f.name == 'ZPlayer.java' and record:
            old = 'public static boolean RECORD = false;'
            if s.count(old) != 1:
                raise SystemExit('ZPlayer.java: no RECORD flag')
            s = s.replace(old, 'public static boolean RECORD = true;')
        (dest / f.name).write_text(s)
    zdn_pack.write_zw(dest / 'ZW.java', name, baby, king, temperature)
    print(f'wrote {dest} (player; baby net {baby or "none"}, king net {king or "none"}, T {temperature}, record {record})')


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--teacher', metavar='NAME')
    g.add_argument('--player', metavar='NAME')
    ap.add_argument('--baby')
    ap.add_argument('--king')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--baby-pm', type=int, default=15)
    ap.add_argument('--king-pm', type=int, default=100)
    ap.add_argument('--out', help='parent directory of the package (default build/botsrc)')
    a = ap.parse_args()
    name = a.teacher or a.player
    if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
        raise SystemExit('name must be a lowercase Java package name')
    if a.teacher:
        build_teacher(name, a.baby_pm, a.king_pm, a.out)
    else:
        build_player(name, a.baby, a.king, a.temperature, a.record, a.out)


if __name__ == '__main__':
    main()
