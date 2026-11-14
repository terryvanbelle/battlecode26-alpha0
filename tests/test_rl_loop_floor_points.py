"""rl_loop.sh passes the config's per-point floors (floor_points -> W.FL) to build_dpnet.py and rl_train.py.

Runs a copy of rl_loop.sh in a temp dir for one iteration whose schedule/collect/data stages are pre-marked done, with
PY pointing at a stub that logs build_dpnet.py / rl_train.py / compile calls instead of running them (other Python,
the loop's own json helpers, runs for real). No games, no numpy, no JDK.

    python3 tests/test_rl_loop_floor_points.py      (or pytest)
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STUB = r'''#!/bin/bash
LOG="$(dirname "$0")/calls.log"
a="$*"
case "$a" in
  *build_dpnet.py*) echo "BUILD $a" >> "$LOG"; n=$(echo "$a" | sed -n 's/.*--name \([^ ]*\).*/\1/p'); mkdir -p build/bots/$n; exit 0;;
  *rl_train.py*) echo "TRAIN $a" >> "$LOG"; o=$(echo "$a" | sed -n 's/.*--out \([^ ]*\).*/\1/p'); mkdir -p $o; cp runs/rl/t/nets/0/net.json $o/net.json; exit 0;;
  *compile_local*) echo "COMPILE" >> "$LOG"; exit 0;;
esac
exec REALPY "$@"
'''


def run_loop(work, floor_points):
    work = Path(work)
    (work / 'scripts').mkdir(parents=True)
    for f in ('rl_loop.sh', 'box_split.sh'):
        shutil.copy(ROOT / 'scripts' / f, work / 'scripts' / f)
    stub = work / 'pystub'
    stub.write_text(STUB.replace('REALPY', sys.executable))
    stub.chmod(0o755)
    cfg = dict(json.loads((ROOT / 'configs/rl-v1.json').read_text()), name='t', anchor_net='anchor.json')
    if floor_points is not None:
        cfg['floor_points'] = floor_points
    cfg['gates'] = dict(cfg['gates'], anchor_every=1000, promote_every=1000)
    (work / 'cfg.json').write_text(json.dumps(cfg))
    rd = work / 'runs/rl/t'
    (rd / 'nets/0').mkdir(parents=True)
    (rd / 'nets/0/net.json').write_text('{}')
    (rd / 'pool.json').write_text(json.dumps(dict(members=[dict(version=0)])))
    I = rd / 'iter-0000'
    (I / 'data').mkdir(parents=True)
    for m in ('scheduled', 'collected', 'data'):
        (I / f'.{m}').touch()
    env = dict(os.environ, PY=str(stub))
    r = subprocess.run(['bash', str(work / 'scripts/rl_loop.sh'), 'cfg.json', 't', '1'], cwd=work, env=env,
                       capture_output=True, text=True, timeout=120)
    assert (I / '.done').exists(), (r.stdout, r.stderr, (rd / 'loop.log').read_text() if (rd / 'loop.log').exists() else '')
    return (work / 'calls.log').read_text().splitlines()


def test_floor_points_passed():
    with tempfile.TemporaryDirectory() as td:
        calls = run_loop(td, {'14': 0.002, '5': 0.001})
    builds = [c for c in calls if c.startswith('BUILD')]
    trains = [c for c in calls if c.startswith('TRAIN')]
    assert len(builds) == 2 and len(trains) == 1, calls
    for c in builds + trains:
        assert '--floor-points 5:0.001,14:0.002' in c, c
        assert '--floor 0.005' in c, c


def test_no_floor_points():
    with tempfile.TemporaryDirectory() as td:
        calls = run_loop(td, None)
    assert calls and not any('--floor-points' in c for c in calls), calls


if __name__ == '__main__':
    test_floor_points_passed()
    test_no_floor_points()
    print('test_rl_loop_floor_points OK')
