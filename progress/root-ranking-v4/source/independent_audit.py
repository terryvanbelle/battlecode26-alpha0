#!/usr/bin/env python3
"""Independent stdlib-only audit. Provisional mode never writes evidence."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import tempfile

ROOT = Path('/workspace/battlecode26-alpha0')
PROGRESS = ROOT / 'progress/root-ranking-v4'
PREFLIGHT = PROGRESS / 'independent-root-preflight.json'
OUTPUT = PROGRESS / 'independent-final-audit.json'
LOCK_NAMES = ('runs/root-ranking-v4/experiment.lock', 'runs/campaign/campaign.lock', 'runs/warmstart-v3/experiment.lock')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stamp(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, 'timestamp lacks timezone')
    return parsed.timestamp()


def actor_state(auditor_pid=None):
    processes = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = [item.decode(errors='replace') for item in (path / 'cmdline').read_bytes().split(b'\0') if item]
            if 'alpha0.root_ranking_v4' in args or 'alpha0.EngineBridge' in args:
                processes.append({'pid': int(path.name), 'worker': '--worker' in args,
                                  'kind': 'engine' if 'alpha0.EngineBridge' in args else 'v4_runner'})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    lock_rows = Path('/proc/locks').read_text().splitlines()
    locks = {}
    for name in LOCK_NAMES:
        path = ROOT / name
        if path.exists():
            inode = str(path.stat().st_ino)
            locks[name] = [line for line in lock_rows if line.split()[5].split(':')[-1] == inode]
        else:
            locks[name] = []
    conflicting = {name: [row for row in rows if int(row.split()[4]) != auditor_pid] for name, rows in locks.items()}
    owns_all = auditor_pid is not None and all(any(int(row.split()[4]) == auditor_pid for row in rows) for rows in locks.values())
    return {'processes': processes, 'locks': locks,
            'auditor_pid': auditor_pid, 'auditor_owns_all_exclusion_locks': owns_all,
            'quiescent': not processes and not any(conflicting.values())}


def audit(final=False):
    began = datetime.now(timezone.utc).isoformat()
    tracked = {}
    saved_times_path = OUTPUT if OUTPUT.exists() else Path('/tmp/bc26-root-ranking-original-times.json')
    saved_times = read(saved_times_path) if saved_times_path.exists() else {}
    if saved_times_path == OUTPUT:
        saved_times = saved_times['chronology']['observations']
    observed_times = {}

    def track(path):
        path = Path(path)
        value = sha(path)
        tracked[str(path)] = value
        return value

    def artifact_time(path):
        """Use hash-bound original observations, never Git-checkout mtimes."""
        path = Path(path)
        name = str(path.relative_to(ROOT))
        content_hash = track(path)
        if name in saved_times:
            observation = saved_times[name]
            require(observation['sha256'] == content_hash, 'chronology observation belongs to different artifact bytes')
        else:
            require(saved_times_path != OUTPUT, 'restored audit lacks original chronology evidence for an artifact')
            observation = {'sha256': content_hash, 'mtime_ns': path.stat().st_mtime_ns,
                           'observed_at': datetime.now(timezone.utc).isoformat()}
        observed_times[name] = observation
        return observation['mtime_ns'] / 1_000_000_000

    identity = read(PROGRESS / 'identity.json')
    cfg = identity['config']
    identity_hash = fp(identity)
    require(read(PROGRESS / 'config.json') == cfg, 'published config differs from frozen identity')
    require(read(ROOT / 'configs/root-ranking-v4.json') == cfg, 'current config differs')
    require(fp(cfg) == identity['config_sha256'], 'semantic config hash differs')
    require(cfg['checkpoint_sha256'] == identity['checkpoint_sha256'] == track(ROOT / cfg['checkpoint']), 'checkpoint hash differs')
    require(track(ROOT / 'configs/root-ranking-v4.json') == identity['config_file_sha256'] == track(PROGRESS / 'source/root-ranking-v4.json'), 'config bytes differ')
    require(track(ROOT / 'alpha0/root_ranking_v4.py') == identity['runner_source_sha256'] == track(PROGRESS / 'source/root_ranking_v4.py'), 'runner source differs')
    require(cfg['source_games'] == [f'iter-0001-game-{i:03d}' for i in range(4, 8)], 'unexpected source games')
    require(cfg['root_roles'] == ['BABY_RAT', 'RAT_KING'] and cfg['required_legal_action'] == 37, 'root selection rule differs')
    require(cfg['selection_seeds'] == [103003, 103013] and cfg['validation_seeds'] == [103021, 103043, 103067, 103091], 'seed sets differ')
    require(cfg['temperature'] == .35 and cfg['workers'] == cfg['torch_threads'] == 1, 'policy or resource recipe differs')
    require(cfg['thresholds'] == {'mean_delta_min': .1, 'improved_roots_min': 2, 'worst_root_delta_min': -.25}, 'thresholds differ')
    require(cfg['search'] == {'simulations': 32, 'max_depth': 12, 'cpuct': 1.5, 'dirichlet_alpha': .3, 'exploration_fraction': .25, 'root_noise': True, 'root_seed_base': 20261114}, 'PUCT recipe differs')
    require(cfg['gradient_updates'] == 0 and cfg['benchmark_data_used'] is False and cfg['wall_budget_seconds'] == 7200, 'diagnostic scope differs')
    gate = read(ROOT / 'configs/victory_gate.json')
    reserved = gate['stages']['confirmation']['seeds']
    require(identity['reserved_confirmation_seeds'] == reserved, 'reserved seed set changed')
    require(not set(reserved) & set(cfg['selection_seeds'] + cfg['validation_seeds']), 'reserved seed contamination')
    runtime = identity['runtime']
    for group in ('files', 'compiled_classes'):
        for name, expected in runtime[group].items():
            require(track(ROOT / name) == expected, f'changed runtime {group}: {name}')
    present_classes = {str(path.relative_to(ROOT)) for path in (ROOT / 'build/java').rglob('*.class')}
    require(present_classes == set(runtime['compiled_classes']), 'compiled class inventory changed')
    require(track(ROOT / 'vendor/jdk/release') == runtime['jdk_release_sha256'], 'JDK release differs')
    for name in ('numpy', 'torch'):
        require(metadata.version(name) == runtime[f'{name}_version'], f'{name} version differs')

    preflight = read(PREFLIGHT)
    preflight_hash = track(PREFLIGHT)
    source_paths = {str(Path(cfg['source_run']) / 'games' / f'{game}.json') for game in cfg['source_games']}
    require(set(identity['source_games']) == source_paths, 'source-game inventory differs')
    sources = {}
    for name in sorted(source_paths):
        source = read(ROOT / name)
        game = source['game_id']
        require(track(ROOT / name) == identity['source_games'][name] == preflight['source_games'][game], 'source game hash differs')
        require(source['source'] == 'selfplay' and source['status'] == 'ok' and source['opponent_checkpoint'] is None, 'non-own source game')
        require(not source.get('benchmark') and not source.get('evaluation') and source['action_version'] == 'macro_v2', 'source-game regime differs')
        require(source['training_runtime'] == runtime, 'source trajectory runtime differs')
        sources[game] = source
    for name, expected in identity['maps'].items():
        require(track(ROOT / name) == expected, 'map bytes differ')

    manifest = read(PROGRESS / 'roots.json')
    track(PROGRESS / 'identity.json')
    track(PROGRESS / 'config.json')
    track(PROGRESS / 'roots.json')
    roots = manifest['roots']
    roots_hash = fp(manifest)
    require(manifest['identity'] == identity_hash and manifest['all_sources_scanned'] is True, 'root manifest identity/inventory incomplete')
    require(len(roots) == len(preflight['roots']) == 8, 'actual root count differs from preflight')
    require({r['game_id'] for r in roots} == set(cfg['source_games']), 'missing source-game roots')
    for n, (root, earlier) in enumerate(zip(roots, preflight['roots'])):
        require(root['ordinal'] == n and root['root_id'] == f'root-{n:02d}', 'root ordinal differs')
        require(all(root.get(key) == value for key, value in earlier.items()), f'root differs from preflight: {root["root_id"]}')
        require(fp(sources[root['game_id']]['actions'][:root['decision_index']]) == root['prefix_sha256'], 'archived prefix differs')
        require(root['player'] in (0, 1) and len(set(root['legal_actions'])) == len(root['legal_actions']) and 37 in root['legal_actions'], 'invalid root team/legal actions')
    require(all(not x['unavailable_roles'] for x in manifest['source_availability']), 'unexpected unavailable role')
    require(2 * sum(len(r['legal_actions']) for r in roots) == preflight['selection_cells'] == 374, 'selection coverage changed')
    by_root = {r['root_id']: r for r in roots}

    comparators = {}
    for root in roots:
        rid = root['root_id']
        path = PROGRESS / 'comparators' / f'{rid}.json'
        record = read(path)
        track(path)
        require(record['identity'] == identity_hash and record['roots_sha256'] == roots_hash and record['root_id'] == rid, 'comparator identity differs')
        require(record['search'] == cfg['search'] and record['search_seed'] == 20261114 + root['ordinal'], 'comparator search differs')
        priors, visits = record['policy_prior'], record['visits']
        require(len(priors) == 40 and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in priors), 'invalid prior')
        require(abs(sum(priors) - 1) < 5e-6 and all(priors[a] == 0 for a in set(range(40)) - set(root['legal_actions'])), 'prior mask/normalization differs')
        require(set(visits) == {str(a) for a in root['legal_actions']} and all(type(v) is int and v >= 0 for v in visits.values()) and sum(visits.values()) == 32, 'invalid comparator visit coverage')
        chosen = max(root['legal_actions'], key=lambda a: visits[str(a)])
        require(record['selected_action'] == chosen, 'comparator is not visit argmax with original legal order')
        require(record['visit_policy'] == [visits.get(str(a), 0) / 32 for a in range(40)], 'comparator visit policy differs')
        require(0 <= record['max_depth'] <= 12 and record['nodes'] >= 1, 'comparator search depth differs')
        comparators[rid] = record

    successes, attempts, pending_starts = {}, [], []
    all_starts = []
    semantic_fields = ('experiment_identity', 'roots_sha256', 'root_id', 'root_state_sha256', 'root_prefix_sha256', 'root_player', 'stage', 'action', 'seed', 'temperature')
    directories = sorted(path for path in (PROGRESS / 'cells').glob('*') if path.is_dir())
    for directory in directories:
        starts = sorted(directory.glob('attempt-[0-9][0-9][0-9]-start.json'))
        results = sorted(directory.glob('attempt-[0-9][0-9][0-9].json'))
        require(starts, f'cell lacks a start journal: {directory.name}')
        base = read(starts[0])
        root = by_root[base['root_id']]
        rid, stage, action, seed = base['root_id'], base['stage'], base['action'], base['seed']
        require(stage in ('selection', 'validation') and action in root['legal_actions'] and seed in cfg[stage + '_seeds'], 'cell outside root/action/seed schedule')
        key = (rid, stage, action, seed)
        require(directory.name == f'{stage}-{rid}-a{action:02d}-s{seed}', 'cell directory differs from identity')
        spec = {'experiment_identity': identity_hash, 'roots_sha256': roots_hash, 'root_id': rid,
                'root_state_sha256': root['state_sha256'], 'root_prefix_sha256': root['prefix_sha256'],
                'root_player': root['player'], 'stage': stage, 'action': action, 'seed': seed, 'temperature': .35}
        start_by_number = {}
        for path in starts:
            value = read(path)
            number = int(path.name.split('-')[1])
            require(value['attempt'] == number and value['identity'] == fp(spec), 'start attempt identity differs')
            require({k: value[k] for k in semantic_fields} == spec, 'start spec differs')
            require(value['status'] == 'incomplete' and value['score'] is None and value['terminal'] is False, 'start journal contains outcome')
            all_starts.append((stamp(value['started_at']), key))
            start_by_number[number] = value
            track(path)
        successful_numbers = []
        for path in results:
            value = read(path)
            number = int(path.stem.split('-')[1])
            require(number in start_by_number and number == value['attempt'], 'result lacks matching start')
            require(value['identity'] == fp(spec) and {k: value[k] for k in semantic_fields} == spec, 'result identity differs')
            require(value['started_at'] == start_by_number[number]['started_at'], 'start time changed')
            start, end = stamp(value['started_at']), stamp(value['ended_at'])
            require(end >= start and value['status'] in ('ok', 'incomplete', 'error'), 'invalid result chronology/status')
            if 'elapsed_seconds' in value:
                require(math.isfinite(value['elapsed_seconds']) and value['elapsed_seconds'] >= 0 and abs((end - start) - value['elapsed_seconds']) < 2, 'wall/monotonic elapsed times disagree')
            if value['status'] == 'ok':
                require(key not in successes, 'duplicate successful cell')
                require(value['terminal'] is True and value['valid_game'] is True and value['benchmark_failures'] == [], 'invalid terminal game')
                winner = value['winner']
                require(type(winner) is int and winner in (-1, 0, 1), 'invalid winner')
                expected_score = Fraction(1, 2) if winner == -1 else Fraction(int(winner == root['player']))
                require(type(value['score']) in (int, float) and Fraction(str(value['score'])) == expected_score, 'wrong player-relative terminal score')
                terminal = value['terminal_summary']
                require(terminal['terminal'] is True and terminal['winner'] == winner and terminal['round'] == value['terminal_round'] >= root['round'], 'terminal proof inconsistent')
                require(type(value['terminal_round']) is int and isinstance(terminal['win_reason'], str) and terminal['win_reason'], 'missing terminal reason/round')
                require(0 < value['decisions'] <= cfg['max_decisions'], 'invalid decision count')
                require(all(re.fullmatch('[0-9a-f]{64}', value[x]) for x in ('terminal_state_sha256', 'actions_sha256')), 'invalid transcript/state hash')
                successes[key] = value
                successful_numbers.append(number)
            else:
                require(value['score'] is None and value['terminal'] is False, 'incomplete attempt has terminal label')
            attempts.append({'cell': list(key), 'attempt': number, 'status': value['status'], 'start': start, 'end': end})
            track(path)
        require(len(successful_numbers) <= 1, 'more than one success')
        if successful_numbers:
            require(max(start_by_number) == successful_numbers[0], 'a completed cell was scheduled again')
        result_numbers = {int(path.stem.split('-')[1]) for path in results}
        for number in set(start_by_number) - result_numbers:
            pending_starts.append({'cell': list(key), 'attempt': number})

    if all_starts:
        first_start = min(x[0] for x in all_starts)
        require(artifact_time(PROGRESS / 'roots.json') <= first_start, 'root freeze postdates rollout start')
        require(all(artifact_time(PROGRESS / 'comparators' / f'{r["root_id"]}.json') <= first_start for r in roots), 'comparator freeze postdates rollout start')
    ordered_attempts = sorted(attempts, key=lambda x: x['start'])
    for earlier, later in zip(ordered_attempts, ordered_attempts[1:]):
        require(later['start'] >= earlier['end'], 'terminal rollout attempts overlap despite one actor')

    expected_selection = {(r['root_id'], 'selection', action, seed) for r in roots for action in r['legal_actions'] for seed in cfg['selection_seeds']}
    expected_validation = set()
    root_reports, deltas = [], []
    all_selection_ready = True
    for root in roots:
        rid = root['root_id']
        expected = {key for key in expected_selection if key[0] == rid}
        ready = expected <= set(successes)
        all_selection_ready &= ready
        means = {}
        selected = None
        selection_path = PROGRESS / 'selections' / f'{rid}.json'
        if ready:
            means = {action: sum(Fraction(str(successes[rid, 'selection', action, seed]['score'])) for seed in cfg['selection_seeds']) / 2 for action in root['legal_actions']}
            selected = max(root['legal_actions'], key=lambda action: (means[action], comparators[rid]['policy_prior'][action], -action))
            if selection_path.exists():
                published = read(selection_path)
                require(published['identity'] == identity_hash and published['roots_sha256'] == roots_hash and published['root_id'] == rid and published['complete'] is True, 'selection provenance differs')
                require(published['selected_action'] == selected and published['action_scores'] == {str(a): float(v) for a, v in means.items()}, 'published selection differs from independent ranking')
                require(published['completed'] == published['expected'] == len(expected) and published['errors'] == 0, 'published selection coverage differs')
                require(max(stamp(successes[key]['ended_at']) for key in expected) <= artifact_time(selection_path), 'action selection predates completed selection outcomes')
                track(selection_path)
            elif final:
                raise ValueError('complete selection has no frozen selection artifact')
        else:
            require(not selection_path.exists(), 'action frozen before all legal-action selection cells completed')
        comparator_action = comparators[rid]['selected_action']
        required_validation = {(rid, 'validation', action, seed) for action in {selected, comparator_action} for seed in cfg['validation_seeds']} if ready else set()
        expected_validation |= required_validation
        existing_validation = {key for key in successes if key[0] == rid and key[1] == 'validation'}
        require(existing_validation <= required_validation, 'validation outcome belongs to an unselected action')
        for start, key in all_starts:
            if key[0] == rid and key[1] == 'validation':
                require(key in required_validation and selection_path.exists(), 'validation attempted before complete selection')
                require(artifact_time(selection_path) <= start, 'validation started before selection was frozen')
        validation_ready = ready and required_validation <= set(successes)
        scores_a = [Fraction(str(successes[rid, 'validation', selected, seed]['score'])) for seed in cfg['validation_seeds']] if validation_ready else []
        scores_b = [Fraction(str(successes[rid, 'validation', comparator_action, seed]['score'])) for seed in cfg['validation_seeds']] if validation_ready else []
        delta = sum((a - b for a, b in zip(scores_a, scores_b)), Fraction(0)) / 4 if validation_ready else None
        if delta is not None:
            deltas.append(delta)
        root_reports.append({'root_id': rid, 'game_id': root['game_id'], 'map': root['map'], 'role': root['role'],
            'selection_complete': ready, 'selection_expected': len(expected), 'selection_completed': len(expected & set(successes)),
            'action_scores': {str(a): float(v) for a, v in means.items()}, 'independent_selected_action': selected,
            'puct_action': comparator_action, 'validation_expected_unique': len(required_validation),
            'validation_completed_unique': len(required_validation & set(successes)), 'validation_complete': validation_ready,
            'rollout_scores': list(map(float, scores_a)), 'puct_scores': list(map(float, scores_b)),
            'paired_delta': None if delta is None else float(delta)})

    require(set(successes) <= expected_selection | expected_validation, 'extra successful cells outside frozen selection/derived validation')
    complete = all_selection_ready and len(deltas) == len(roots) == 8 and not pending_starts
    mean = sum(deltas, Fraction(0)) / 8 if complete else None
    improved = sum(delta > 0 for delta in deltas) if complete else None
    worst = min(deltas) if complete else None
    passed = bool(complete and mean >= Fraction(1, 10) and improved >= 2 and worst >= Fraction(-1, 4))
    process_state = actor_state(os.getpid() if final else None)
    report = {'schema_version': 1, 'audit_id': 'terminal-root-ranking-v4-independent-v1',
        'snapshot_started_at': began, 'generated_at': datetime.now(timezone.utc).isoformat(), 'final': final,
        'identity_sha256': identity_hash, 'roots_sha256': roots_hash, 'completed': complete,
        'preflight_source_path': str(PREFLIGHT.relative_to(ROOT)) if PREFLIGHT.is_relative_to(ROOT) else str(PREFLIGHT),
        'preflight_source_sha256': preflight_hash,
        'selection_expected': len(expected_selection), 'selection_completed': len(expected_selection & set(successes)),
        'validation_expected_unique': len(expected_validation), 'validation_completed_unique': len(expected_validation & set(successes)),
        'terminal_games_completed': len(successes), 'error_attempts': sum(a['status'] == 'error' for a in attempts),
        'incomplete_attempts': sum(a['status'] == 'incomplete' for a in attempts), 'pending_start_records': pending_starts,
        'independent_decision': {'mean_delta': None if mean is None else float(mean), 'mean_delta_exact': None if mean is None else str(mean),
            'improved_roots': improved, 'worst_root_delta': None if worst is None else float(worst),
            'passes_mechanism_rule': passed if final else None},
        'roots': root_reports, 'actor_state': process_state,
        'chronology': {'mode': 'archived_original_observations' if saved_times_path == OUTPUT else 'original_run_observations',
            'observations': observed_times, 'prior_observations_source_sha256': sha(saved_times_path) if saved_times_path.exists() else None,
            'interpretation': 'Root/comparator/selection files lack embedded creation timestamps. Their before-scoring chronology uses original filesystem mtime observations bound to exact artifact SHA-256s. A restored checkout reuses these observations; its new filesystem mtimes are not treated as original chronology. Cell start/end order and complete-selection dependencies are recomputed from the immutable records.'},
        'checks': ['frozen checkpoint/config/runner/compiled classes/JDK/package versions', 'own source games/maps and independent preflight roots',
            'all legal actions and both selection seeds', 'PUCT 32 visits/depth12 and fixed noise seed', 'root-relative terminal proof and unique successful cells',
            'start/end, one-actor, root/comparator freeze and selection-before-validation chronology',
            'independent prior/action-ID tie-break', 'separate four-seed paired validation with equal root weight', 'exact rational decision thresholds'],
        'limitations': ['No engine replay or model inference was performed by this audit; terminal state/transcript hashes and frozen prior/search outputs are checked for provenance and internal consistency.',
            'Artifact publication chronology relies on recorded original filesystem observations, not timestamps regenerated by Git restore.',
            'This is an eight-root own-self-play mechanism diagnostic, not a benchmark or all-map victory claim.'],
        'evidence_sha256': {str(Path(k).relative_to(ROOT)) if Path(k).is_relative_to(ROOT) else k: v for k, v in tracked.items()}}
    if final:
        require(complete and process_state['quiescent'] and process_state['auditor_owns_all_exclusion_locks'],
                'final audit requires complete evidence, no actors, and ownership of all exclusion locks')
        status = read(ROOT / 'runs/root-ranking-v4/status.json')
        summary = read(PROGRESS / 'summary.json')
        require(status['phase'] == 'complete_review_required' and status['runtime_and_checkpoint_unchanged'] is True, 'runner did not publish completed integrity check')
        require(status['identity'] == identity_hash and summary['identity_sha256'] == identity_hash and summary['complete'] is True, 'final runner identity/completeness differs')
        for original in (status['decision'], summary['decision']):
            require(original['complete'] is True and original['integrity_verified'] is True, 'final integrity not verified')
            require(original['mean_delta'] == float(mean) and original['improved_roots'] == improved and original['worst_root_delta'] == float(worst), 'published decision differs from independent arithmetic')
            require(original['criteria_pass'] == original['passes'] == passed, 'published pass/fail differs')
        require(summary['totals']['selection_expected'] == len(expected_selection) and summary['totals']['selection_completed'] == len(expected_selection), 'summary selection counts differ')
        require(summary['totals']['validation_expected_unique'] == len(expected_validation) == summary['totals']['validation_completed_unique'], 'summary validation counts differ')
        report['published_decision_matches_independent_audit'] = True
        report['runner_status_sha256'] = track(ROOT / 'runs/root-ranking-v4/status.json')
        report['runner_summary_sha256'] = track(PROGRESS / 'summary.json')
    # Recheck all immutable input/evidence files after reading; final mode also
    # checks that no actor appeared during this audit window.
    for name, expected in tracked.items():
        require(sha(name) == expected, f'file changed while audited: {name}')
    if final:
        final_state = actor_state(os.getpid())
        require(final_state['quiescent'] and final_state['auditor_owns_all_exclusion_locks'], 'actor or conflicting lock appeared during final audit')
    report['auditor_source_sha256'] = sha(__file__)
    return report


def main():
    global ROOT, PROGRESS, PREFLIGHT, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--preflight', type=Path)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    PROGRESS = ROOT / 'progress/root-ranking-v4'
    PREFLIGHT = args.preflight.resolve() if args.preflight else PROGRESS / 'independent-root-preflight.json'
    OUTPUT = PROGRESS / 'independent-final-audit.json'
    with ExitStack() as stack:
        if args.final:
            for name in LOCK_NAMES:
                path = ROOT / name
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(path.open('a+'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = actor_state(os.getpid())
            require(before['quiescent'] and before['auditor_owns_all_exclusion_locks'], 'actor not quiescent under audit locks')
        report = audit(args.final)
        if args.final:
            require(not OUTPUT.exists(), 'independent final evidence already exists; never overwrite it')
            auditor_destination = PROGRESS / 'source/independent_audit.py'
            auditor_bytes = Path(__file__).read_bytes()
            if auditor_destination.exists():
                require(auditor_destination.read_bytes() == auditor_bytes, 'archived independent auditor differs')
            else:
                fd, temporary = tempfile.mkstemp(dir=auditor_destination.parent, prefix='.independent-auditor-')
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(auditor_bytes)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.link(temporary, auditor_destination)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            require(sha(auditor_destination) == report['auditor_source_sha256'], 'auditor changed during final audit')
            report['auditor_source_path'] = str(auditor_destination.relative_to(ROOT))
            fd, temporary = tempfile.mkstemp(dir=OUTPUT.parent, prefix='.independent-audit-')
            try:
                with os.fdopen(fd, 'w') as stream:
                    json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
                    stream.write('\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, OUTPUT)
            finally:
                Path(temporary).unlink(missing_ok=True)
    print(json.dumps({key: report[key] for key in ('final', 'completed', 'selection_expected', 'selection_completed',
        'validation_expected_unique', 'validation_completed_unique', 'terminal_games_completed', 'error_attempts',
        'incomplete_attempts', 'pending_start_records', 'independent_decision')}, indent=2))
    print(json.dumps([{'root_id': root['root_id'], 'selected': root['independent_selected_action'], 'puct': root['puct_action'],
        'validation_complete': root['validation_complete'], 'delta': root['paired_delta']} for root in report['roots']], indent=2))


if __name__ == '__main__':
    main()
