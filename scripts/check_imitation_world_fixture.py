"""Bounded synthetic two-round GameWorld check. No teacher and no completion seal."""
import argparse, contextlib, fcntl, hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from alpha0.env import JAVA_OPENS
from alpha0.expanded_v11_run import LOCKS, STOPS
from alpha0.imitation.trace import read_trace, digest
from alpha0.imitation.terminal import validate_terminal
from scripts.build_imitation_hooks import build

def check(output,bot="imitationfixture"):
    output=Path(output).resolve()
    with contextlib.ExitStack() as stack:
        for path in LOCKS:
            stream=stack.enter_context(open(path,'a'))
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any(Path(p).exists() for p in STOPS):raise RuntimeError('research STOP present')
        manifest=build(output)
        classes=output/'test-classes';classes.mkdir()
        jar=ROOT/'vendor/engine/battlecode26-java-1.2.6.jar'
        sources=[ROOT/p for p in ('java/alpha0/NoopGameMaker.java','tests/java/WorldJournalTest.java',f'tests/java/{bot}/RobotPlayer.java')]
        subprocess.run([str(ROOT/'vendor/jdk/bin/javac'),'-cp',f'{output}/classes:{jar}','-d',str(classes),*map(str,sources)],check=True,timeout=30)
        logs={}
        for mode in ('recording','baseline'):
            # Baseline resolves original engine classes first. Shared fixture bytecode is identical.
            cp=f'{classes}:{output}/classes:{jar}' if mode=='recording' else f'{classes}:{jar}:{output}/classes'
            result=subprocess.run([str(ROOT/'vendor/jdk/bin/java'),*JAVA_OPENS,'-cp',cp,'WorldJournalTest',str(classes),str(output/mode),mode,bot],cwd=ROOT,capture_output=True,text=True,timeout=45)
            (output/f'{mode}.stdout').write_text(result.stdout);(output/f'{mode}.stderr').write_text(result.stderr)
            if result.returncode:raise RuntimeError(f'{mode} failed; retained logs')
            logs[mode]=result.stdout.strip()
        assert (output/'recording/world-audit.json').read_bytes()==(output/'baseline/world-audit.json').read_bytes()
        assert (output/'recording/world-audit-sequence.json').read_bytes()==(output/'baseline/world-audit-sequence.json').read_bytes()
        for mode in ('recording','baseline'):
            for path in (output/mode/'world-audits').glob('*.json'):
                assert digest(json.loads(path.read_text()))==path.stem
        terminal=(output/'recording/terminal-proof.json').exists()
        assert terminal==(output/'baseline/terminal-proof.json').exists()
        if terminal:
            assert (output/'recording/terminal-proof.json').read_bytes()==(output/'baseline/terminal-proof.json').read_bytes()
            for mode in ('recording','baseline'):
                validate_terminal(json.loads((output/mode/'terminal-proof.json').read_text()),json.loads((output/mode/'world-audit.json').read_text()))
        rows=read_trace(output/'recording/trace.jsonl')
        lifecycle=bot in ('imitationdeathfixture','imitationresignfixture','imitationexceptionfixture','imitationreturnfixture')
        if lifecycle:
            expected=['action'] if bot in ('imitationdeathfixture','imitationresignfixture') else []
            assert [r['kind'] for r in rows]==expected
            exits=json.loads((output/'recording/terminations.json').read_text())
            expected_exit={'imitationdeathfixture':'robot_death_signal','imitationresignfixture':'robot_death_signal','imitationexceptionfixture':'exception','imitationreturnfixture':'returned'}[bot]
            assert len(exits)==1 and exits[0]['outcome']==expected_exit and exits[0]['turn_open']
        else:
            expected=['turn_end']*2 if bot=='imitationoverrunfixture' else ['action','action','turn_end']*2
            assert [r['kind'] for r in rows]==expected
            reason='overrun' if bot=='imitationoverrunfixture' else 'yield'
            assert all(r['payload']['termination']==reason for r in rows if r['kind']=='turn_end')
            exits=json.loads((output/'recording/terminations.json').read_text())
            assert len(exits)==1 and not exits[0]['turn_open']
        for key in ('world_terminal','winner','round'):
            assert json.loads((output/'recording/result.json').read_text())[key]==json.loads((output/'baseline/result.json').read_text())[key]
        assert not read_trace(output/'baseline/trace.jsonl')
        budgets=json.loads((output/'recording/budgets.json').read_text())
        assert budgets==json.loads((output/'baseline/budgets.json').read_text())
        observations=list((output/'recording/observations').glob('*.json'))
        for path in observations:
            assert digest(json.loads(path.read_text()))==path.stem
        for row in rows:
            if row['kind']=='turn_end':assert (output/'recording/world-audits'/f"{row['payload']['physical_state_sha256']}.json").is_file()
            assert (output/'recording/observations'/f"{row['payload']['public_observation_sha256']}.json").is_file()
        result={'scope':'real GameWorld, synthetic bot, two nonterminal rounds; not teacher equivalence',
                'world_audit_equal':True,'terminal_proof_equal':terminal,'bot':bot,'teacher_games':0,'complete_game':False,'training_approved':False,'records':len(rows),
                'sandbox_exits':exits,'observations':len(observations),'matched_bytecode_budgets':budgets,'logs':logs,
                'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()}
        (output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bot',choices=['imitationfixture','imitationoverrunfixture','imitationdeathfixture','imitationresignfixture','imitationexceptionfixture','imitationreturnfixture'],default='imitationfixture')
    args=parser.parse_args();print(json.dumps(check(args.output,args.bot),indent=2))
