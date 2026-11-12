import json
import pytest
from alpha0.imitation.replay import compare_replay

def pair(tmp_path):
    paths=[tmp_path/'baseline.json',tmp_path/'recording.json']
    rows=[{'sequence':0,'method':'addSqueakAction','arguments':[[1,2]]},{'sequence':1,'method':'endRound','arguments':[]}]
    for p in paths:p.write_text(json.dumps(rows))
    return paths,rows

def test_scope(tmp_path):
    paths,_=pair(tmp_path);r=compare_replay(*paths)
    assert r['callbacks']==2 and r['equal']
    assert not r['complete_primitive_trace'] and not r['training_approved']

@pytest.mark.parametrize('change',['argument','order','missing','empty','sequence'])
def test_mismatch(tmp_path,change):
    paths,rows=pair(tmp_path)
    if change=='argument':rows[0]['arguments']=[[2,1]]
    if change=='order':rows.reverse()
    if change=='missing':rows.pop()
    if change=='empty':rows=[]
    if change=='sequence':rows[0]['sequence']=True
    paths[1].write_text(json.dumps(rows))
    with pytest.raises(ValueError):compare_replay(*paths)
