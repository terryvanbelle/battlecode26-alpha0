import copy
import pytest
from alpha0.imitation.provenance import source_spec,validate,invocation_id

def setup(tmp_path):
    paths={}
    for name in ('map_file','engine_jar','harness_source'):
        p=tmp_path/name;p.write_text(name);paths[name]=p
    for name in ('bot_classes','hook_classes'):
        p=tmp_path/name;p.mkdir();(p/'Robot.class').write_bytes(b'fixture');paths[name]=p
    spec=source_spec(slot='slot-1',map_name='tiny',seed=314159,bot='imitationresignfixture',full_game=True,**paths)
    return paths,spec

def test_identity_and_modes(tmp_path):
    paths,spec=setup(tmp_path);assert validate(spec,**paths)==spec['game_id']
    assert invocation_id(spec,'baseline')!=invocation_id(spec,'recording')
    other=source_spec(slot='slot-2',map_name='tiny',seed=314159,bot='imitationresignfixture',full_game=True,**paths)
    assert other['game_id']!=spec['game_id']

@pytest.mark.parametrize('mutation',['class_change','class_add','map','seed','extra'])
def test_changed_identity_rejected(tmp_path,mutation):
    paths,spec=setup(tmp_path)
    if mutation=='class_change':(paths['bot_classes']/'Robot.class').write_bytes(b'changed')
    if mutation=='class_add':(paths['hook_classes']/'New.class').write_bytes(b'new')
    if mutation=='map':paths['map_file'].write_bytes(b'changed')
    if mutation=='seed':spec['seed']+=1
    if mutation=='extra':spec['training_approved']=True
    with pytest.raises(ValueError,match='identity'):validate(spec,**paths)

def test_teacher_and_excluded_maps_blocked(tmp_path):
    paths,_=setup(tmp_path)
    for bot,map_name in [('vibe26b_zdb','tiny'),('imitationresignfixture','Stash')]:
        with pytest.raises(ValueError):source_spec(slot='x',map_name=map_name,seed=1,bot=bot,full_game=True,**paths)

def test_symlink_blocked(tmp_path):
    paths,spec=setup(tmp_path);(paths['bot_classes']/'Alias.class').symlink_to(paths['map_file'])
    with pytest.raises(ValueError,match='symlink'):validate(spec,**paths)
