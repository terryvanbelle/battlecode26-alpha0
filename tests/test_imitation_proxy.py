import copy
import pytest
from scripts.check_imitation_proxy_fixture import compare

def records():
    payload=dict(operation='squeak',arguments=[42],return_boolean=False,exception=None)
    trace=[dict(kind='action',robot_id=5,team='A',round=1,payload=payload)]
    rows=[dict(sequence=0,robot=5,team='A',round=1,**payload)]
    return rows,trace

def test_false_and_exception_preserved():
    rows,trace=records();assert compare(rows,trace)==1
    rows[0]['exception']=trace[0]['payload']['exception']='battlecode.common.GameActionException'
    rows[0]['return_boolean']=trace[0]['payload']['return_boolean']=None
    assert compare(rows,trace)==1

@pytest.mark.parametrize('field,value',[('arguments',[43]),('return_boolean',True),('exception','Unexpected'),('robot',6),('round',2),('sequence',1)])
def test_disagreement_rejected(field,value):
    rows,trace=records();rows[0][field]=value
    with pytest.raises(ValueError):compare(rows,trace)

def test_missing_duplicate_rejected():
    rows,trace=records()
    for bad in [[],rows+copy.deepcopy(rows)]:
        with pytest.raises(ValueError):compare(bad,trace)
