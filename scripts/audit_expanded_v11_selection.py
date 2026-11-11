"""Independent saved-record selection audit; no collector imports or engine runs.

A live snapshot certifies only the captured immutable records, never recovery,
complete data, fitting correctness, or playing strength.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_cell(row, root, identity, action, seed, attempt=1):
    expected = dict(experiment_identity=digest(identity), root_id=root['root_id'],
        root_player=root['player'], root_state_sha256=root['state_sha256'],
        root_prefix_sha256=root['prefix_sha256'], root_input_hash=root['input_hash'],
        source_sha256=root['source_sha256'], stage='selection', action=action,
        seed=seed, temperature=identity['config']['temperature'])
    require(row.get('identity') == digest(expected) and all(row.get(k) == v for k,v in expected.items()), 'cell identity/schedule mismatch')
    require(row.get('attempt') == attempt and row.get('status') == 'ok', 'failed or repeated attempt requires review')
    require(row.get('terminal') is True and row.get('valid_game') is True and row.get('benchmark_failures') == [], 'invalid terminal record')
    raw = row['terminal_state']; winner = row['winner']
    require(type(winner) is int and winner in (-1,0,1) and type(raw.get('winner')) is int and raw['winner'] == winner, 'winner mismatch')
    require(raw.get('terminal') is True and raw.get('valid_game',True) is True and raw.get('benchmark_failures',[]) == [], 'invalid terminal state')
    require(type(raw.get('round')) is int and raw['round'] > 0 and type(row.get('terminal_round')) is int and row['terminal_round'] == raw['round'], 'round mismatch')
    require(digest({k:v for k,v in raw.items() if k != 'snapshot'}) == row['terminal_state_sha256'], 'terminal hash mismatch')
    require(row['terminal_summary'] == {k:raw.get(k) for k in ('terminal','round','winner','win_reason','bank','kings','transferred','cat_damage')}, 'terminal summary mismatch')
    require(type(row.get('score')) in (int,float) and row['score'] == (.5 if winner == -1 else float(winner == root['player'])), 'score mismatch')
    actions = row['actions']
    require(isinstance(actions,list) and actions and actions[0] == action and all(type(a) is int and 0 <= a < 40 for a in actions), 'action transcript mismatch')
    require(type(row.get('decisions')) is int and row['decisions'] == len(actions) and digest(actions) == row['actions_sha256'], 'action transcript hash/count mismatch')


def audit(base=ROOT):
    progress=base/'progress/expanded-terminal-v11'; run=base/'runs/expanded-terminal-v11'
    identity=read(progress/'identity.json')
    for path, expected in identity['sources'].items():
        require(sha(base/path)==expected, f'changed pinned source: {path}')
    for row in identity['input_plans'].values():
        require(sha(base/row['path'])==row['sha256'], 'changed input plan')
    training=read(base/identity['input_plans']['training']['path'])
    roots=training['roots']; require(len(roots)==64 and len({r['root_id'] for r in roots})==64, 'missing root slots')
    schedule={f"selection-{r['root_id']}-a{a:02d}-s{s}":(r,a,s) for r in roots for a in r['legal_actions'] for s in identity['config']['selection_seeds']}
    require(len(schedule)==2974,'changed selection schedule')
    files=sorted((progress/'cells').glob('selection-*/attempt-[0-9][0-9][0-9].json'))
    proofs=[]; seen=set(); counts=Counter(); reviewed_interruptions=[]
    review_path=progress/'reviews/deadline-recovery-20261111.json'
    review=read(review_path) if review_path.exists() else None
    allowed=set(review['interrupted_cells']) if review else set()
    if review:
        require(review['identity_sha256']==digest(identity), 'foreign recovery review')
        # Verify original cells/logs/descriptors; status is allowed to advance later.
        for name, expected in review['preserved_files'].items():
            if name.endswith('/status.json'):continue
            require(sha(base/name)==expected, 'changed pre-recovery evidence')
    for path in files:
        name=path.parent.name
        require(name in schedule and name not in seen, 'unexpected or repeated scheduled cell')
        root,action,seed=schedule[name]; row=read(path)
        original=str(path.with_name('attempt-001.json').relative_to(base))
        if str(path.relative_to(base)) in allowed:
            require(row['status']=='incomplete' and row['score'] is None and row['terminal'] is False and row['error']=='Interrupted: STOP or deadline', 'invalid reviewed interruption')
            reviewed_interruptions.append({'path':str(path.relative_to(base)),'sha256':sha(path)})
            continue
        verify_cell(row,root,identity,action,seed,attempt=2 if original in allowed else 1)
        seen.add(name); counts[root['map']]+=1
        proofs.append({'path':str(path.relative_to(base)),'sha256':sha(path)})
    comparators=0
    for root in roots:
        path=progress/'comparators'/f"{root['root_id']}.json"
        if not path.exists():continue
        row=read(path)
        spec=dict(experiment_identity=digest(identity),root_id=root['root_id'],root_state_sha256=root['state_sha256'],root_input_hash=root['input_hash'],source_sha256=root['source_sha256'],search=identity['config']['search'],search_seed=223201+root['root_ordinal'])
        require(row['identity']==digest(spec) and all(row[k]==v for k,v in spec.items()),'comparator identity mismatch')
        visits={int(k):v for k,v in row['visits'].items()}
        require(set(visits)<=set(root['legal_actions']) and all(type(v) is int and v>=0 for v in visits.values()) and sum(visits.values())==32, 'comparator visits mismatch')
        require(row['selected_action'] in visits and visits[row['selected_action']]==max(visits.values()), 'comparator action mismatch')
        comparators+=1
    jobs=Counter(); output_files=0
    for path in sorted((progress/'jobs').glob('*.json')):
        marker=read(path); descriptor=run/'jobs'/path.name; job=read(descriptor)
        require(marker['status']=='ok' and marker['job_sha256']==sha(descriptor), 'failed/changed completed job')
        require(job['id']==path.stem==digest({'kind':job['kind'],'payload':job['payload'],'identity':digest(identity)}), 'job identity mismatch')
        require(bool(marker['output_hashes']), 'missing output hashes')
        for name, expected in marker['output_hashes'].items():
            require(sha(base/name)==expected, f'changed job output: {name}'); output_files+=1
        jobs[job['kind']]+=1
    return dict(updated_at=datetime.now(timezone.utc).isoformat(), passed=True,
        experiment_identity=digest(identity), scope='saved-record snapshot; no independent engine replay or recovery certification',
        selection_valid=len(seen),selection_expected=len(schedule),selection_complete=len(seen)==len(schedule),
        comparators_valid=comparators,completed_jobs=dict(jobs),hashed_job_outputs=output_files,
        per_map_valid=dict(counts),terminal_records=proofs,reviewed_interruptions=reviewed_interruptions,playing_strength_claim=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    result=audit();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as handle:json.dump(result,handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='terminal_records'},indent=2))
