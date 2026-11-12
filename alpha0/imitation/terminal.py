"""Audit-only terminal proof checks. Does not approve a journal or training data."""
from .trace import exact, hash_value, integer, digest

def validate_terminal(proof, world_audit):
    exact(proof, ('schema','world_terminal','winner','reason','round','physical_state_sha256','training_approved'))
    if proof['schema']!='synthetic-world-terminal-v1' or proof['world_terminal'] is not True or proof['training_approved'] is not False:
        raise ValueError('invalid terminal proof scope')
    hash_value(proof['physical_state_sha256']);integer(proof['round'])
    if proof['physical_state_sha256']!=digest(world_audit):raise ValueError('world audit hash mismatch')
    if world_audit.get('schema')!='engine-world-audit-v1':raise ValueError('unknown world audit')
    world=world_audit['world']
    if world.get('class')!='battlecode.world.GameWorld':raise ValueError('wrong world class')
    fields=world['fields'];prefix='battlecode.world.GameWorld.'
    if fields[prefix+'running'] is not False or fields[prefix+'currentRound']!=proof['round']:
        raise ValueError('world is not terminal at declared round')
    stats=fields[prefix+'gameStats']
    if stats.get('class')!='battlecode.world.GameStats':raise ValueError('missing game stats')
    stats=stats['fields']
    for field,key,enum in (('winner','winner','battlecode.common.Team'),('dominationFactor','reason','battlecode.world.DominationFactor')):
        value=stats['battlecode.world.GameStats.'+field]
        if not isinstance(value,dict) or value.get('enum')!=enum or value.get('name')!=proof[key]:raise ValueError('terminal result mismatch')
    if proof['winner'] not in ('A','B'):raise ValueError('missing player winner')
    return {'terminal_verified':True,'training_approved':False,'journal_complete':False}
