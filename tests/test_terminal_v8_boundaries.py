"""Fresh experiment allocation, prior-input exclusion and real process isolation."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import pytest

from alpha0 import terminal_v8 as S, terminal_v8_common as C
from alpha0.terminal_v8_collect import source_spec


def test_fresh_seed_inventory_retains_exact_allocation_and_temperatures():
    c=C.load_config()
    assert c['fit']['updates']==2048 and c['fit']['anchor_kl_weight']==1
    assert c['source_temperature']==1 and c['temperature']==.35
    new=set(); old=set()
    for split,maximum,initial in [('training',8,4),('validation',6,2)]:
        for m in range(4):
            for j in range(1,maximum+1):
                job=C.source_job(split,m,j,c)
                base,pol=(20263200,330000) if split=='training' else (20264200,340000)
                assert job['seed']==base+100*m+j
                assert job['policy_seeds']==[2*(pol+100*m+j)+t for t in (0,1)]
                checked_job={**job, 'exclusion_map': {'a'*64: [{'kind':'fixture','id':0}]}} if split=='validation' else job
                assert source_spec(checked_job,c,{})['temperature']==1
                new.update([job['seed'],*job['policy_seeds']])
                old.update([base-2000+100*m+j,*[2*(pol-20000+100*m+j)+t for t in (0,1)]])
    new.update(c['selection_seeds']+c['validation_seeds']+[c['anchor_seed'],c['fit']['seed']])
    new.update(c['search']['root_seed_base']+i for i in range(32))
    previous=json.loads((C.ROOT/'configs/terminal-target-head-v6.json').read_text())
    old.update(previous['selection_seeds']+previous['validation_seeds']+[previous['anchor_seed'],previous['fit']['seed']])
    old.update(previous['search']['root_seed_base']+i for i in range(32))
    assert not new&old


def test_historical_hash_registry_is_exact_and_pinned():
    registry=C.historical_exclusions()
    meta=json.loads((C.ROOT/'runs/terminal-target-head-v6/dataset/training.json').read_text())
    val=json.loads((C.ROOT/'progress/terminal-target-head-v6/roots-validation.json').read_text())
    expected=set(meta['root_input_hashes']+meta['anchor_input_hashes']+[r['input_hash'] for r in val['roots']])
    assert len(expected)==560 and set(registry)==expected
    with patch.object(C,'sha256',return_value='bad'):
        with pytest.raises(ValueError,match='historical exclusion input changed'):
            C.historical_exclusions()


def test_all_seven_stops_and_unreadable_actor_cwd_block():
    assert len(C.LOCKS)==len(C.STOPS)==7
    assert {p.parent.name for p in C.STOPS}=={'campaign','warmstart-v3','terminal-target-head-v5','terminal-target-head-v6','cached-head-constraint-v7','terminal-target-head-v8','development-v3'}
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); stops=tuple(root/f'STOP{i}' for i in range(7))
        with patch.object(C,'STOPS',stops):
            for stop in stops:
                stop.touch()
                with pytest.raises(C.Interrupted,match='STOP'):
                    C.check_stop(float('inf'))
                stop.unlink()
        process=root/'123456';process.mkdir()
        (process/'cmdline').write_bytes(b'python\0-u\0-m\0alpha0.terminal_v8\0')
        original=Path.resolve
        def resolve(path,*a,**kw):
            if path.name=='cwd':raise PermissionError('managed namespace')
            return original(path,*a,**kw)
        with patch.object(Path,'resolve',resolve):
            found=S.active_actors(root)
        assert len(found)==1 and found[0]['reason']=='unreadable_cwd'


def test_parent_death_kills_worker_and_descendant_group():
    with tempfile.TemporaryDirectory() as d:
        ready=Path(d)/'ready.json'
        child='''import os,sys,json,subprocess,time
from pathlib import Path
from alpha0.terminal_v8 import protect_worker_lifecycle
protect_worker_lifecycle(int(sys.argv[1]))
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
Path(sys.argv[2]).write_text(json.dumps([os.getpid(),p.pid]))
time.sleep(30)
'''
        parent='''import os,sys,subprocess,time
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c',sys.argv[1],str(os.getpid()),sys.argv[2]],start_new_session=True)
end=time.monotonic()+5
while not Path(sys.argv[2]).exists():
 if time.monotonic()>end: raise RuntimeError('child not ready')
 time.sleep(.01)
'''
        p=subprocess.Popen([sys.executable,'-c',parent,child,str(ready)],cwd=C.ROOT)
        try:
            p.wait(timeout=8)
            assert p.returncode==0
            ids=json.loads(ready.read_text())
            end=time.monotonic()+3
            def live(pid):
                try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
                except FileNotFoundError:return False
            while any(live(pid) for pid in ids) and time.monotonic()<end:time.sleep(.02)
            assert not any(live(pid) for pid in ids)
        finally:
            if p.poll() is None:p.kill();p.wait()
            if ready.exists():
                for pid in json.loads(ready.read_text()):
                    try:os.kill(pid,9)
                    except ProcessLookupError:pass
