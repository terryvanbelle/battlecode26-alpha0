"""Versioned audit-only closure/seal for terminal recorder runs, not training approval."""
import hashlib
import json
from pathlib import Path
from .trace import read_trace, digest, exact, hash_value, integer
from .terminal import validate_terminal

def load(path, limit=64*1024*1024):
    if path.stat().st_size>limit:raise ValueError('audit input exceeds cap')
    return json.loads(path.read_text())

def completion(root, *, expected_roster, game_id):
    root=Path(root);hash_value(game_id)
    if not expected_roster:raise ValueError('empty recorder roster')
    def identity(row):
        exact(row,('robot','team'));integer(row['robot'])
        if row['team'] not in ('A','B'):raise ValueError('invalid roster team')
        return row['team'],row['robot']
    keys=[identity(r) for r in expected_roster]
    if len(set(keys))!=len(keys):raise ValueError('duplicate roster')
    proof=load(root/'terminal-proof.json');world=load(root/'world-audit.json')
    validate_terminal(proof,world)
    rows=read_trace(root/'trace.jsonl');last={}
    for row in rows:
        key=(row['team'],row['robot_id'])
        if row['game_id']!=game_id or key not in keys:raise ValueError('unexpected journal identity')
        if row['round']>proof['round']:raise ValueError('record after terminal round')
        sha=row['payload']['public_observation_sha256'];hash_value(sha)
        obs=load(root/'observations'/f'{sha}.json',1024*1024)
        if digest(obs)!=sha:raise ValueError('observation hash mismatch')
        if obs.get('robot_id')!=row['robot_id'] or obs.get('team')!=row['team'] or obs.get('round')!=row['round']:
            raise ValueError('observation identity mismatch')
        last[key]=row
    exits=load(root/'terminations.json');seen=set();closures=[]
    for end in exits:
        exact(end,('robot','team','round','budget','outcome','exception','turn_open'))
        key=(end['team'],end['robot']);integer(end['robot']);integer(end['round']);integer(end['budget'])
        if key not in keys or key in seen:raise ValueError('unexpected or duplicate exit')
        seen.add(key)
        if type(end['turn_open']) is not bool or end['round']>proof['round']:raise ValueError('invalid exit boundary')
        if end['outcome'] not in ('returned','exception','robot_death_signal'):raise ValueError('unknown exit cause')
        if end['outcome']=='returned':
            if end['exception'] is not None:raise ValueError('return with exception')
        elif not isinstance(end['exception'],str) or not end['exception']:raise ValueError('missing exception')
        if end['outcome']=='robot_death_signal' and end['exception']!='battlecode.instrumenter.RobotDeathException':raise ValueError('wrong death signal')
        row=last.get(key)
        if row and row['round']>end['round']:raise ValueError('record after exit')
        if row and row['kind']=='action' and not end['turn_open']:raise ValueError('unclosed action turn')
        observation=None if row is None else {'sha256':row['payload']['public_observation_sha256'],'record_sequence':row['sequence'],'record_sha256':row['sha256'],'round':row['round'],'turn':row['turn']}
        closures.append({'exit':end,'observation_status':'not_captured' if row is None else 'last_recorded_not_post_exit','last_observation':observation})
    if seen!=set(keys):raise ValueError('missing robot exit')
    closures.sort(key=lambda c:(c['exit']['team'],c['exit']['robot']))
    files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('trace.jsonl','terminations.json','terminal-proof.json','world-audit.json')}
    return {'schema':'terminal-recorder-completion-v1','game_id':game_id,'scope':'audit-only synthetic recorder closure','complete':True,'record_count':len(rows),'expected_roster':sorted(expected_roster,key=lambda r:(r['team'],r['robot'])),'closures':closures,'files':files,'terminal_proof_sha256':digest(proof),'training_approved':False,'teacher_equivalence_proven':False}

def seal(root, destination, *, expected_roster, game_id):
    result=completion(root,expected_roster=expected_roster,game_id=game_id)
    result['sha256']=digest(result)
    # Exclusive creation; a crash cannot leave a valid JSON completion marker.
    with Path(destination).open('x') as stream:
        json.dump(result,stream,sort_keys=True,separators=(',',':'));stream.write('\n');stream.flush()
    return result

def verify(root, path, *, expected_roster, game_id):
    saved=load(Path(path));expected=completion(root,expected_roster=expected_roster,game_id=game_id)
    expected['sha256']=digest(expected)
    if saved!=expected:raise ValueError('completion seal mismatch')
    return saved
