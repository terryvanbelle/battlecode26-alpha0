"""Bounded synthetic two-round GameWorld check. No teacher and no completion seal."""
import argparse, contextlib, fcntl, hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from alpha0.env import JAVA_OPENS
from alpha0.expanded_v11_run import LOCKS, STOPS
from alpha0.imitation.trace import read_trace, digest
from scripts.build_imitation_hooks import build

def check(output):
    output=Path(output).resolve()
    with contextlib.ExitStack() as stack:
        for path in LOCKS:
            stream=stack.enter_context(open(path,'a'))
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any(Path(p).exists() for p in STOPS):raise RuntimeError('research STOP present')
        manifest=build(output)
        classes=output/'test-classes';classes.mkdir()
        jar=ROOT/'vendor/engine/battlecode26-java-1.2.6.jar'
        sources=[ROOT/p for p in ('java/alpha0/NoopGameMaker.java','tests/java/WorldJournalTest.java','tests/java/imitationfixture/RobotPlayer.java')]
        subprocess.run([str(ROOT/'vendor/jdk/bin/javac'),'-cp',f'{output}/classes:{jar}','-d',str(classes),*map(str,sources)],check=True,timeout=30)
        logs={}
        for mode in ('recording','baseline'):
            # Baseline resolves original engine classes first. Shared fixture bytecode is identical.
            cp=f'{classes}:{output}/classes:{jar}' if mode=='recording' else f'{classes}:{jar}:{output}/classes'
            result=subprocess.run([str(ROOT/'vendor/jdk/bin/java'),*JAVA_OPENS,'-cp',cp,'WorldJournalTest',str(classes),str(output/mode),mode],cwd=ROOT,capture_output=True,text=True,timeout=45)
            (output/f'{mode}.stdout').write_text(result.stdout);(output/f'{mode}.stderr').write_text(result.stderr)
            if result.returncode:raise RuntimeError(f'{mode} failed; retained logs')
            logs[mode]=result.stdout.strip()
        rows=read_trace(output/'recording/trace.jsonl')
        assert len(rows)==6 and [r['kind'] for r in rows]==['action','action','turn_end']*2
        assert not read_trace(output/'baseline/trace.jsonl')
        budgets=json.loads((output/'recording/budgets.json').read_text())
        assert budgets==json.loads((output/'baseline/budgets.json').read_text())
        observations=list((output/'recording/observations').glob('*.json'))
        for path in observations:
            assert digest(json.loads(path.read_text()))==path.stem
        for row in rows:
            assert (output/'recording/observations'/f"{row['payload']['public_observation_sha256']}.json").is_file()
        result={'scope':'real GameWorld, synthetic bot, two nonterminal rounds; not teacher equivalence',
                'teacher_games':0,'complete_game':False,'training_approved':False,'records':len(rows),
                'observations':len(observations),'matched_bytecode_budgets':budgets,'logs':logs,
                'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()}
        (output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(check(parser.parse_args().output),indent=2))
