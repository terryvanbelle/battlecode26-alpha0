from copy import deepcopy
import pytest
from scripts import audit_expanded_v11_selection as A


def fixture():
    progress=A.ROOT/'progress/expanded-terminal-v11'
    identity=A.read(progress/'identity.json')
    roots=A.read(A.ROOT/identity['input_plans']['training']['path'])['roots']
    path=next((progress/'cells').glob('selection-*/attempt-001.json'))
    row=A.read(path);root=next(r for r in roots if r['root_id']==row['root_id'])
    return row,root,identity


def test_actual_saved_terminal():
    row,root,identity=fixture()
    A.verify_cell(row,root,identity,row['action'],row['seed'])


@pytest.mark.parametrize('change', ['score','transcript','terminal','seed','root','failed'])
def test_tampering_rejected(change):
    original,root,identity=fixture();row=deepcopy(original)
    if change=='score':row['score']=.25
    elif change=='transcript':row['actions'].append(0)
    elif change=='terminal':row['terminal_state']['round']+=1
    elif change=='seed':row['seed']+=1
    elif change=='root':row['root_input_hash']='0'*64
    elif change=='failed':row['status']='error'
    with pytest.raises(ValueError):
        A.verify_cell(row,root,identity,original['action'],original['seed'])
