from copy import deepcopy
from scripts.recover_v8_deadline_cells import approved

def test_recovery_is_only_unscored_first_sigterm():
    row={'attempt':1,'status':'error','error_kind':'evaluation_error',
         'evaluation':{'error':'SIGTERM','score':None,'rounds':None}}
    assert approved(row)
    for key,value in [('attempt',2),('status','ok'),('error_kind','invalid_game')]:
        other={**row,key:value};assert not approved(other)
    for key,value in [('error','AssertionError'),('score',0),('score',1),('rounds',200)]:
        other=deepcopy(row);other['evaluation'][key]=value;assert not approved(other)
