"""Build the student harness: hooked RobotControllerImpl + alpha0.student classes -> build/student/classes.

Also injects the decision-record channel: setIndicatorString(s) with s.charAt(0) == '\\uDEC0' goes to
alpha0.student.Decisions.take (never to the robot's indicator string).

Runtime classpath: build/student/classes:build/java:vendor/engine/battlecode26-java-1.2.6.jar
(requires scripts/setup_engine.py to have run).
"""
import re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vendor/engine/src/battlecode/world/RobotControllerImpl.java'
OUT = ROOT / 'build/student'
JAR = ROOT / 'vendor/engine/battlecode26-java-1.2.6.jar'
PRIMITIVES = {'moveForward', 'move', 'turn', 'buildRat', 'becomeRatKing', 'placeDirt', 'removeDirt', 'placeRatTrap',
              'removeRatTrap', 'placeCatTrap', 'removeCatTrap', 'pickUpCheese', 'attack', 'squeak', 'writeSharedArray',
              'transferCheese', 'throwRat', 'dropRat', 'carryRat', 'disintegrate', 'resign'}


def transform(source: str) -> tuple[str, list[str]]:
    pattern = re.compile(r'public\s+(void|boolean)\s+(\w+)\s*\(([^)]*)\)\s*(throws\s+GameActionException\s*)?\{')
    edits, found = [], []
    for m in pattern.finditer(source):
        result, name, params, throws = m.groups()
        if name not in PRIMITIVES:
            continue
        args = [p.strip().split()[-1] for p in params.split(',') if p.strip()]
        start, depth, i = m.end(), 1, m.end()
        while depth:
            c = source[i]
            depth += (c == '{') - (c == '}')
            i += 1
        body = source[start:i - 1]
        helper = f'_student_{name}_{len(args)}'
        sig = f'public {result} {name}({params})' + (' throws GameActionException' if throws else '')
        priv = f'private {result} {helper}({params})' + (' throws GameActionException' if throws else '')
        call = f'{helper}({", ".join(args)})'
        run = f'{result} r = {call}; return r;' if result == 'boolean' else f'{call};'
        catch = 'GameActionException | RuntimeException | Error' if throws else 'RuntimeException | Error'
        wrapper = (f'{sig} {{\n'
                   f'        Object tok = alpha0.student.Hook.begin(this, "{name}", new Object[]{{{", ".join(args)}}});\n'
                   f'        Throwable failure = null;\n'
                   f'        try {{ {run} }}\n'
                   f'        catch ({catch} e) {{ failure = e; throw e; }}\n'
                   f'        finally {{ alpha0.student.Hook.finish(this, tok, failure); }}\n'
                   f'    }}\n    {priv} {{{body}}}')
        edits.append((m.start(), i, wrapper))
        found.append(name)
    m = re.search(r'public\s+void\s+setIndicatorString\s*\(\s*String\s+(\w+)\s*\)\s*\{', source)
    if not m:
        raise SystemExit('setIndicatorString(String) not found in engine source')
    v = m.group(1)
    edits.append((m.end(), m.end(), f"\n        if ({v} != null && !{v}.isEmpty() && {v}.charAt(0) == '\\uDEC0') "
                                     f"{{ alpha0.student.Decisions.take(this, {v}); return; }}"))
    missing = PRIMITIVES - set(found)
    if missing:
        raise SystemExit(f'primitive methods not found in engine source: {sorted(missing)}')
    edits.sort(key=lambda e: e[0])
    for a, b, text in reversed(edits):
        source = source[:a] + text + source[b:]
    return source, found


def main():
    javac = ROOT / 'vendor/jdk/bin/javac'
    if not SOURCE.exists() or not (ROOT / 'build/java').exists():
        raise SystemExit('run scripts/setup_engine.py first')
    # Build into a temp dir and swap, so running games never see a half-written class tree.
    final, OUT_TMP = OUT, OUT.with_name(OUT.name + '.tmp')
    if OUT_TMP.exists():
        shutil.rmtree(OUT_TMP)
    out = OUT_TMP
    src = out / 'src/battlecode/world'
    src.mkdir(parents=True)
    text, found = transform(SOURCE.read_text())
    (src / 'RobotControllerImpl.java').write_text(text)
    classes = out / 'classes'
    classes.mkdir()
    java = sorted((ROOT / 'java/student').rglob('*.java'))
    subprocess.run([str(javac), '--release', '21', '-nowarn', '-cp', f'{ROOT / "build/java"}:{JAR}', '-d', str(classes),
                    str(src / 'RobotControllerImpl.java'), *map(str, java)], check=True)
    old = final.with_name(final.name + '.old')
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        final.rename(old)
    out.rename(final)
    print(f'built {final / "classes"} ({len(found)} hooked primitives: {sorted(set(found))})')


if __name__ == '__main__':
    main()
