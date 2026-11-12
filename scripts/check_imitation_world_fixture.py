"""Bounded synthetic two-round GameWorld check. No teacher and no completion seal."""
import argparse, contextlib, fcntl, hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from alpha0.env import JAVA_OPENS
from alpha0.expanded_v11_run import LOCKS, STOPS
from alpha0.imitation.trace import read_trace, digest
from alpha0.imitation.terminal import validate_terminal
from alpha0.imitation.completion import seal, verify
from alpha0.imitation.provenance import source_spec,validate,invocation_id
from alpha0.imitation.audit_store import read_snapshot
from alpha0.imitation.replay import compare_replay
from alpha0.imitation.runtime import runtime_spec, validate_runtime, launch_command, clean_environment
from scripts.build_imitation_hooks import build

def check(output,bot="imitationfixture",full_game=False):
    output=Path(output).resolve()
    with contextlib.ExitStack() as stack:
        for path in LOCKS:
            stream=stack.enter_context(open(path,'a'))
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any(Path(p).exists() for p in STOPS):raise RuntimeError('research STOP present')
        environment=clean_environment()
        runtime=runtime_spec(ROOT/'vendor/jdk',JAVA_OPENS)
        manifest=build(output)
        (output/'runtime-spec.json').write_text(json.dumps(runtime,indent=2)+'\n')
        classes=output/'test-classes';classes.mkdir()
        jar=ROOT/'vendor/engine/battlecode26-java-1.2.6.jar'
        sources=[ROOT/p for p in ('java/alpha0/NoopGameMaker.java','tests/java/WorldJournalTest.java',f'tests/java/{bot}/RobotPlayer.java')]
        subprocess.run([str(ROOT/'vendor/jdk/bin/javac'),'-cp',f'{output}/classes:{jar}','-d',str(classes),*map(str,sources)],check=True,timeout=30)
        artifacts=dict(map_file=ROOT/'maps/tiny.map26',engine_jar=jar,bot_classes=classes,hook_classes=output/'classes',harness_source=ROOT/'tests/java/WorldJournalTest.java')
        spec=source_spec(slot=output.name,map_name='tiny',seed=314159,bot=bot,full_game=full_game,**artifacts)
        (output/'source-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
        logs={};timings={};launches={}
        import time
        for mode in ('recording','baseline'):
            # Baseline resolves original engine classes first. Shared fixture bytecode is identical.
            cp=f'{classes}:{output}/classes:{jar}' if mode=='recording' else f'{classes}:{jar}:{output}/classes'
            validate(spec,**artifacts)
            validate_runtime(runtime,ROOT/'vendor/jdk',JAVA_OPENS)
            command=launch_command(runtime,classpath=cp,arguments=[str(classes),str(output/mode),mode,bot,str(output/'source-spec.json')])
            launches[mode]={'argv':command,'runtime_id':runtime['runtime_id'],'game_id':spec['game_id']}
            (output/'launches.json').write_text(json.dumps(launches,indent=2)+'\n')
            started=time.monotonic()
            result=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True,timeout=45)
            timings[mode]=time.monotonic()-started
            validate_runtime(runtime,ROOT/'vendor/jdk',JAVA_OPENS)
            validate(spec,**artifacts)
            if result.returncode==0 and json.loads((output/mode/'source-spec.json').read_text())!=spec:raise ValueError('engine source echo differs')
            (output/f'{mode}.stdout').write_text(result.stdout);(output/f'{mode}.stderr').write_text(result.stderr)
            if result.returncode:raise RuntimeError(f'{mode} failed; retained logs')
            logs[mode]=result.stdout.strip()
        replay_comparison=compare_replay(output/'baseline/replay-audit.json',output/'recording/replay-audit.json')
        (output/'replay-comparison.json').write_text(json.dumps(replay_comparison,indent=2)+'\n')
        assert (output/'recording/world-audit.json').read_bytes()==(output/'baseline/world-audit.json').read_bytes()
        assert (output/'recording/world-audit-sequence.json').read_bytes()==(output/'baseline/world-audit-sequence.json').read_bytes()
        for mode in ('recording','baseline'):
            for path in (output/mode/'world-audits').glob('*.index.json'):
                sha=path.name.removesuffix('.index.json')
                assert digest(json.loads(read_snapshot(path.parent,sha)))==sha
        terminal=(output/'recording/terminal-proof.json').exists()
        assert terminal==(output/'baseline/terminal-proof.json').exists()
        if terminal:
            assert (output/'recording/terminal-proof.json').read_bytes()==(output/'baseline/terminal-proof.json').read_bytes()
            for mode in ('recording','baseline'):
                validate_terminal(json.loads((output/mode/'terminal-proof.json').read_text()),json.loads((output/mode/'world-audit.json').read_text()))
        if terminal:
            roster=json.loads((output/'recording/recorder-roster.json').read_text())
            assert roster==json.loads((output/'baseline/recorder-roster.json').read_text())
            seal(output/'recording',output/'recording/completion.json',expected_roster=roster,game_id=spec['game_id'])
            verify(output/'recording',output/'recording/completion.json',expected_roster=roster,game_id=spec['game_id'])
        rows=read_trace(output/'recording/trace.jsonl')
        lifecycle=bot in ('imitationdeathfixture','imitationresignfixture','imitationexceptionfixture','imitationreturnfixture')
        if bot=='imitationmultifixture':
            exits=json.loads((output/'recording/terminations.json').read_text())
            robot_ids={r['robot_id'] for r in rows}
            if len(robot_ids)<2:raise ValueError('multi-robot fixture failed to spawn/execute')
            if not terminal:raise ValueError('multi-robot terminal fixture incomplete')
            if not any(r['kind']=='action' and r['payload']['operation']=='buildRat' for r in rows):raise ValueError('spawn action missing')
            if {x['robot'] for x in exits}!=robot_ids:raise ValueError('multi-robot exits differ')
        elif lifecycle:
            expected=['action'] if bot in ('imitationdeathfixture','imitationresignfixture') else []
            assert [r['kind'] for r in rows]==expected
            exits=json.loads((output/'recording/terminations.json').read_text())
            expected_exit={'imitationdeathfixture':'robot_death_signal','imitationresignfixture':'robot_death_signal','imitationexceptionfixture':'exception','imitationreturnfixture':'returned'}[bot]
            assert len(exits)==1 and exits[0]['outcome']==expected_exit and exits[0]['turn_open']
        else:
            expected=(['action']*8+['turn_end'])*2 if bot=='imitationboundaryfixture' else (['turn_end']*2 if bot=='imitationoverrunfixture' else ['action','action','turn_end']*2)
            assert [r['kind'] for r in rows]==expected
            reason='overrun' if bot=='imitationoverrunfixture' else 'yield'
            assert all(r['payload']['termination']==reason for r in rows if r['kind']=='turn_end')
            exits=json.loads((output/'recording/terminations.json').read_text())
            assert len(exits)==1 and not exits[0]['turn_open']
        if bot=='imitationboundaryfixture':
            actions=[r['payload'] for r in rows if r['kind']=='action']
            operations=['writeSharedArray','writeSharedArray','squeak','squeak','attack','attackAmount','pickUpCheese','pickUpCheeseAmount']
            for offset in (0,8):
                chunk=actions[offset:offset+8]
                if [p['operation'] for p in chunk]!=operations:raise ValueError('boundary operation coverage')
                if [p['return_boolean'] for p in chunk[2:4]]!=[True,False]:raise ValueError('boundary boolean coverage')
                if any(p['exception']!='battlecode.common.GameActionException' for p in chunk[4:]):raise ValueError('boundary exception coverage')
        for key in ('world_terminal','winner','round'):
            assert json.loads((output/'recording/result.json').read_text())[key]==json.loads((output/'baseline/result.json').read_text())[key]
        assert not read_trace(output/'baseline/trace.jsonl')
        budgets=json.loads((output/'recording/budgets.json').read_text())
        assert budgets==json.loads((output/'baseline/budgets.json').read_text())
        observations=list((output/'recording/observations').glob('*.json'))
        for path in observations:
            assert digest(json.loads(path.read_text()))==path.stem
        for row in rows:
            if row['kind']=='turn_end':assert read_snapshot(output/'recording/world-audits',row['payload']['physical_state_sha256'])
            assert (output/'recording/observations'/f"{row['payload']['public_observation_sha256']}.json").is_file()
        result={'scope':'real GameWorld, synthetic bot, two nonterminal rounds; not teacher equivalence',
                'runtime_id':runtime['runtime_id'],'source_game_id':spec['game_id'],'invocation_ids':{m:invocation_id(spec,m) for m in ('baseline','recording')},'wall_seconds':timings,'full_game_requested':full_game,'replay_callbacks_equal':True,'replay_is_complete_primitive_trace':False,'world_audit_equal':True,'terminal_proof_equal':terminal,'bot':bot,'teacher_games':0,'complete_game':False,'training_approved':False,'records':len(rows),
                'sandbox_exits':exits,'observations':len(observations),'matched_bytecode_budgets':budgets,'logs':logs,
                'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()}
        (output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bot',choices=['imitationmultifixture','imitationboundaryfixture','imitationfixture','imitationoverrunfixture','imitationdeathfixture','imitationresignfixture','imitationexceptionfixture','imitationreturnfixture'],default='imitationfixture')
    parser.add_argument('--full-game',action='store_true')
    args=parser.parse_args();print(json.dumps(check(args.output,args.bot,args.full_game),indent=2))
