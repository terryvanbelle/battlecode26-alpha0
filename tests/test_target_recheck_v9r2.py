from pathlib import Path
from copy import deepcopy
import pytest
from scripts.run_target_recheck_v9r2 import valid_records,P,read,C
from alpha0.terminal_v8_collect import verify_terminal

def test_real_inherited_transcript_validates_without_changing_bytes():
    pins=read(C.ROOT/'configs/target-recheck-v9r2-inherited.json')
    assert len(pins)==8
    for name,pin in pins.items():
        original=(C.ROOT/pin['path']).read_bytes()
        _,ok=valid_records(P/'cells'/name)
        assert len(ok)==1 and ok[0]['decisions']==len(ok[0]['actions'])
        assert (C.ROOT/pin['path']).read_bytes()==original
        corrupted=deepcopy(ok[0]);corrupted['actions'][0]=(corrupted['actions'][0]+1)%40
        with pytest.raises(ValueError):verify_terminal(corrupted,corrupted['root_player'])
