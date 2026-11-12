"""Pinned diagnostic source identities. Does not register seeds or authorize teacher play."""
import hashlib
from pathlib import Path
from .trace import digest,hash_value,integer
EXCLUDED_MAPS={'Stash','Hike','heart','ratsareants'}

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def tree_hashes(path):
    path=Path(path)
    if path.is_symlink():raise ValueError('symlink class root')
    entries={}
    for item in sorted(path.rglob('*')):
        if item.is_symlink():raise ValueError('symlink in class tree')
        if item.is_file():entries[str(item.relative_to(path))]=file_hash(item)
    if not entries:raise ValueError('empty class tree')
    return entries

def source_spec(*,slot,map_name,seed,bot,map_file,engine_jar,bot_classes,hook_classes,harness_source,full_game):
    if not isinstance(slot,str) or not slot or map_name in EXCLUDED_MAPS:raise ValueError('invalid or excluded source')
    integer(seed)
    if seed>=2**31 or type(full_game) is not bool:raise ValueError('invalid source settings')
    # This API deliberately cannot silently graduate synthetic fixtures into teacher collection.
    if bot not in ('imitationboundaryfixture','imitationdeathfixture','imitationresignfixture','imitationexceptionfixture','imitationreturnfixture','imitationfixture','imitationoverrunfixture'):raise ValueError('teacher protocol not registered')
    core={'schema':'synthetic-game-source-v1','slot':slot,'map':map_name,'seed':seed,'bot':bot,'full_game':full_game,
          'map_sha256':file_hash(map_file),'engine_sha256':file_hash(engine_jar),
          'bot_classes':tree_hashes(bot_classes),'hook_classes':tree_hashes(hook_classes),
          'harness_sha256':file_hash(harness_source),'engine_limits':'unmodified','teacher_games':0,'training_approved':False}
    return {**core,'game_id':digest(core)}

def validate(spec,*,map_file,engine_jar,bot_classes,hook_classes,harness_source):
    actual=source_spec(slot=spec['slot'],map_name=spec['map'],seed=spec['seed'],bot=spec['bot'],full_game=spec['full_game'],map_file=map_file,engine_jar=engine_jar,bot_classes=bot_classes,hook_classes=hook_classes,harness_source=harness_source)
    if actual!=spec:raise ValueError('source identity changed')
    return actual['game_id']

def invocation_id(spec,mode):
    hash_value(spec['game_id'])
    if mode not in ('baseline','recording'):raise ValueError('unknown mode')
    return digest({'game_id':spec['game_id'],'mode':mode,'classpath_order':'original-engine-first' if mode=='baseline' else 'isolated-hooks-first'})
