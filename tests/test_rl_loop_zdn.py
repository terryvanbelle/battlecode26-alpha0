"""rl_loop.sh with config "kind": "zdn" (docs/zdn.md, "RL"): the start copies the start nets, their norm.npz and the
anchor nets; the build, data and train stages call build_zdn.py / zdn_rl_data.py / zdn_rl_train.py with the config's
values; a KL refusal (exit 3) rolls the new version back to the best nets, baby and king with their norms.

Runs a copy of rl_loop.sh in a temp dir for one iteration whose schedule/collect stages are pre-marked done, with PY
pointing at a stub that logs the build / data / train / compile calls instead of running them (the loop's own json
helpers run for real). No games, no numpy, no JDK. Same harness as test_rl_loop_floor_points.py.

    python3 tests/test_rl_loop_zdn.py      (or pytest)
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STUB = r'''#!/bin/bash
LOG="$(dirname "$0")/calls.log"
a="$*"
arg() { echo "$a" | sed -n "s/.*--$1 \([^ ]*\).*/\1/p"; }
case "$a" in
  *build_zdn.py*) echo "BUILD $a" >> "$LOG"; mkdir -p build/bots/$(arg player); exit 0;;
  *build_dpnet.py*) echo "DPBUILD $a" >> "$LOG"; exit 1;;
  *zdn_rl_data.py*) echo "DATA $a" >> "$LOG"; echo "iter: stub data"; exit 0;;
  *rl_data.py*) echo "DPDATA $a" >> "$LOG"; exit 1;;
  *zdn_rl_train.py*) echo "TRAIN $a" >> "$LOG"
    [ "${STUB_TRAIN_RC:-0}" = 3 ] && { echo "REFUSED stub"; exit 3; }
    o=$(arg out); n=$(arg nets); mkdir -p $o; echo '{"trained": 1}' > $o/baby.json; echo '{"trained": 1}' > $o/king.json
    cp $n/baby.norm.npz $o/; echo "baby: stub"; exit 0;;
  *rl_train.py*) echo "DPTRAIN $a" >> "$LOG"; exit 1;;
  *compile_local*) echo "COMPILE $a" >> "$LOG"; exit 0;;
esac
exec REALPY "$@"
'''


def run_loop(work, train_rc=0):
    work = Path(work)
    (work / 'scripts').mkdir(parents=True)
    for f in ('rl_loop.sh', 'box_split.sh'):
        shutil.copy(ROOT / 'scripts' / f, work / 'scripts' / f)
    stub = work / 'pystub'
    stub.write_text(STUB.replace('REALPY', sys.executable))
    stub.chmod(0o755)
    for u in ('baby', 'king'):
        (work / 'start' / u).mkdir(parents=True)
        (work / 'start' / u / 'net.json').write_text(json.dumps({'start': u}))
        (work / 'anc' / u).mkdir(parents=True)
        (work / 'anc' / u / 'net.json').write_text(json.dumps({'anchor': u}))
    (work / 'start/baby/norm.npz').write_bytes(b'baby norm')          # the king's start has no norm.npz
    cfg = json.loads((ROOT / 'configs/rl-zdn1.json').read_text())
    cfg.update(name='t', start_nets={u: f'start/{u}/net.json' for u in ('baby', 'king')},
               anchor_nets={u: f'anc/{u}/net.json' for u in ('baby', 'king')}, disk_guard_gb=0)
    cfg['gates'] = dict(cfg['gates'], anchor_every=1000, promote_every=1000)
    (work / 'cfg.json').write_text(json.dumps(cfg))
    rd = work / 'runs/rl/t'
    I = rd / 'iter-0000'
    (I / 'data').mkdir(parents=True)
    (I / 'data/trace.npz').write_bytes(b'')
    for m in ('scheduled', 'collected'):
        (I / f'.{m}').touch()
    (rd / 'pool.json').write_text(json.dumps(dict(members=[dict(version=0)])))
    env = dict(os.environ, PY=str(stub), STUB_TRAIN_RC=str(train_rc))
    r = subprocess.run(['bash', str(work / 'scripts/rl_loop.sh'), 'cfg.json', 't', '1'], cwd=work, env=env,
                       capture_output=True, text=True, timeout=120)
    log = (rd / 'loop.log').read_text() if (rd / 'loop.log').exists() else ''
    assert (I / '.done').exists(), (r.stdout, r.stderr, log)
    return cfg, rd, (work / 'calls.log').read_text().splitlines(), log


def flags(call):
    """--key value pairs of a logged call (flags without a value map to '')."""
    t = call.split()
    out = {}
    for i, w in enumerate(t):
        if w.startswith('--'):
            out[w[2:]] = t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith('--') else ''
    return out


def test_zdn_iteration_calls_the_zdn_stages():
    with tempfile.TemporaryDirectory() as td:
        cfg, rd, calls, log = run_loop(td)
        n0 = rd / 'nets/0'
        assert json.loads((n0 / 'baby.json').read_text()) == {'start': 'baby'}
        assert json.loads((n0 / 'king.json').read_text()) == {'start': 'king'}
        assert (n0 / 'baby.norm.npz').read_bytes() == b'baby norm' and not (n0 / 'king.norm.npz').exists()
        assert json.loads((rd / 'anchor/king.json').read_text()) == {'anchor': 'king'}
        assert (rd / 'nets/1/baby.norm.npz').read_bytes() == b'baby norm'
        assert not any(c.startswith(('DPBUILD', 'DPDATA', 'DPTRAIN')) for c in calls), calls
        builds = [flags(c) for c in calls if c.startswith('BUILD')]
        assert [b['player'] for b in builds] == ['zdnr0', 'zdnr1'], calls
        for v, b in enumerate(builds):
            assert b['baby'] == f'runs/rl/t/nets/{v}/baby.json' and b['king'] == f'runs/rl/t/nets/{v}/king.json', b
            assert (b['record'], b['record-king'], b['version']) == ('20', '250', str(v)), b
            assert float(b['temperature']) == cfg['temperature'], b
        assert sum(c.startswith('COMPILE') for c in calls) == 2, calls
        data = [flags(c) for c in calls if c.startswith('DATA')]
        assert data == [{'config': 'cfg.json', 'run': 't', 'iter': '0'}], calls
        train = [flags(c) for c in calls if c.startswith('TRAIN')]
        assert len(train) == 1, calls
        t, tc = train[0], cfg['train']
        assert (t['nets'], t['anchor'], t['out'], t['version']) == ('runs/rl/t/nets/0', 'runs/rl/t/anchor', 'runs/rl/t/nets/1', '0'), t
        assert t['data'] == 'runs/rl/t/iter-0000/data' and t['value-data'] == 'runs/rl/t/iter-0000/data', t
        assert abs(float(t['beta']) - tc['beta_start']) < 1e-9 and float(t['lr']) == tc['lr'], t
        for key, flag in (('epochs', 'epochs'), ('batch_turns', 'batch-turns'), ('ppo_clip', 'clip'), ('target_kl', 'target-kl'),
                          ('grad_clip', 'grad-clip'), ('c_ent', 'c-ent'), ('max_anchor_kl', 'max-anchor-kl'), ('adv_clip', 'adv-clip'),
                          ('value_val_frac', 'val-frac'), ('threads', 'threads'), ('critic', 'critic'), ('gae_lambda', 'gae-lambda'),
                          ('critic_folds', 'critic-folds'), ('device', 'device')):
            assert t[flag] == str(tc[key]), (flag, t[flag], tc[key])
        assert 'baby: stub' in log and 'stub data' in log
        assert not (rd / 'nets/1/ROLLBACK.json').exists()
        rows = [json.loads(l) for l in (rd / 'progress.jsonl').read_text().splitlines()]
        assert len(rows) == 1 and rows[0]['iter'] == 0 and 'train_units' in rows[0]


def test_zdn_refusal_rolls_back_both_units():
    with tempfile.TemporaryDirectory() as td:
        cfg, rd, calls, log = run_loop(td, train_rc=3)
        n1 = rd / 'nets/1'
        assert json.loads((n1 / 'ROLLBACK.json').read_text())['rolled_back_to'] == 0
        for u in ('baby', 'king'):
            assert (n1 / f'{u}.json').read_text() == (rd / f'nets/0/{u}.json').read_text(), u
        assert (n1 / 'baby.norm.npz').read_bytes() == b'baby norm' and not (n1 / 'king.norm.npz').exists()
        assert [flags(c)['player'] for c in calls if c.startswith('BUILD')] == ['zdnr0', 'zdnr1'], calls
        assert 'ROLLBACK v1 -> v0' in log


if __name__ == '__main__':
    test_zdn_iteration_calls_the_zdn_stages()
    test_zdn_refusal_rolls_back_both_units()
    print('test_rl_loop_zdn OK')
