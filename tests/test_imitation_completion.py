import json
from pathlib import Path
import pytest
from alpha0.imitation.trace import TraceWriter,digest
from alpha0.imitation.completion import completion,seal,verify

ROSTER=[{'robot':5,'team':'A'}]
GAME='a'*64

def fixture(tmp_path,action=True):
    source=Path(__file__).resolve().parents[1]/'diagnostics/zdb-imitation-planning/world-audit/terminal-example'
    for name in ('world-audit.json','terminal-proof.json'):(tmp_path/name).write_bytes((source/name).read_bytes())
    (tmp_path/'observations').mkdir()
    obs={'robot_id':5,'team':'A','round':1};sha=digest(obs)
    (tmp_path/'observations'/f'{sha}.json').write_text(json.dumps(obs))
    w=TraceWriter(tmp_path/'trace.jsonl',GAME)
    if action:w.append(5,'A',1,0,'action',{'operation':'resign','arguments':[],'outcome':'exception','exception':'battlecode.instrumenter.RobotDeathException','return_boolean':None,'public_observation_sha256':sha,'bytecodes_before':1,'bytecodes_after':1})
    w.close()
    end={'robot':5,'team':'A','round':1,'budget':1,'outcome':'robot_death_signal','exception':'battlecode.instrumenter.RobotDeathException','turn_open':True}
    (tmp_path/'terminations.json').write_text(json.dumps([end]))
    return tmp_path

def test_last_observation_explicitly_not_post_death(tmp_path):
    root=fixture(tmp_path);s=seal(root,root/'completion.json',expected_roster=ROSTER,game_id=GAME)
    assert s['closures'][0]['observation_status']=='last_recorded_not_post_exit'
    assert not s['training_approved'] and not s['teacher_equivalence_proven']
    assert verify(root,root/'completion.json',expected_roster=ROSTER,game_id=GAME)==s
    with pytest.raises(FileExistsError):seal(root,root/'completion.json',expected_roster=ROSTER,game_id=GAME)

def test_no_observation_is_not_fabricated(tmp_path):
    root=fixture(tmp_path,False);r=completion(root,expected_roster=ROSTER,game_id=GAME)
    assert r['closures'][0]['last_observation'] is None
    assert r['closures'][0]['observation_status']=='not_captured'

@pytest.mark.parametrize('change',['missing','duplicate','unknown','closed_action'])
def test_bad_exits_rejected(tmp_path,change):
    root=fixture(tmp_path);p=root/'terminations.json';exits=json.loads(p.read_text())
    if change=='missing':exits=[]
    if change=='duplicate':exits*=2
    if change=='unknown':exits[0]['outcome']='unknown'
    if change=='closed_action':exits[0]['turn_open']=False
    p.write_text(json.dumps(exits))
    with pytest.raises(ValueError):seal(root,root/'completion.json',expected_roster=ROSTER,game_id=GAME)
    assert not (root/'completion.json').exists()

def test_seal_tampering_rejected(tmp_path):
    root=fixture(tmp_path);p=root/'completion.json';s=seal(root,p,expected_roster=ROSTER,game_id=GAME)
    s['closures'][0]['observation_status']='post_death';s['sha256']=digest({k:v for k,v in s.items() if k!='sha256'});p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='mismatch'):verify(root,p,expected_roster=ROSTER,game_id=GAME)

def test_wrong_identity_rejected(tmp_path):
    root=fixture(tmp_path)
    with pytest.raises(ValueError,match='identity'):completion(root,expected_roster=ROSTER,game_id='b'*64)
