"""Third-arm synthetic interface-proxy diagnostic; never teacher authorization."""
import contextlib,fcntl,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from alpha0.expanded_v11_run import LOCKS,STOPS
from alpha0.imitation.runtime import validate_runtime,launch_command,clean_environment
from alpha0.imitation.provenance import validate
from alpha0.imitation.trace import read_trace
from alpha0.env import JAVA_OPENS

def compare(rows,trace):
    expected=[]
    for r in trace:
        if r['kind']!='action':continue
        p=r['payload']
        expected.append(dict(sequence=len(expected),robot=r['robot_id'],team=r['team'],round=r['round'],operation=p['operation'],arguments=p['arguments'],return_boolean=p['return_boolean'],exception=p['exception']))
    if rows!=expected:raise ValueError('independent controller calls differ')
    return len(expected)

def check(output):
    output=Path(output).resolve()
    with contextlib.ExitStack() as stack:
        for path in LOCKS:fcntl.flock(stack.enter_context(open(path,'a')),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any(Path(p).exists() for p in STOPS):raise ValueError('research STOP')
        spec=json.loads((output/'source-spec.json').read_text());runtime=json.loads((output/'runtime-spec.json').read_text())
        artifacts=dict(map_file=ROOT/'maps/tiny.map26',engine_jar=ROOT/'vendor/engine/battlecode26-java-1.2.6.jar',bot_classes=output/'test-classes',hook_classes=output/'classes',harness_source=ROOT/'tests/java/WorldJournalTest.java')
        validate(spec,**artifacts);validate_runtime(runtime,ROOT/'vendor/jdk',JAVA_OPENS)
        if (output/'proxy').exists():raise ValueError('exclusive proxy attempt')
        cp=f"{output}/test-classes:{artifacts['engine_jar']}:{output}/classes"
        command=launch_command(runtime,classpath=cp,arguments=[str(output/'test-classes'),str(output/'proxy'),'proxy',spec['bot'],str(output/'source-spec.json')])
        with (output/'proxy-launch.json').open('x') as f:json.dump(command,f)
        result=subprocess.run(command,cwd=ROOT,env=clean_environment(),capture_output=True,text=True,timeout=45)
        (output/'proxy.stdout').write_text(result.stdout);(output/'proxy.stderr').write_text(result.stderr)
        if result.returncode:raise ValueError('proxy failed; logs retained')
        validate(spec,**artifacts);validate_runtime(runtime,ROOT/'vendor/jdk',JAVA_OPENS)
        for name in ['budgets.json','world-audit.json','world-audit-sequence.json','replay-audit.json','recorder-roster.json','source-spec.json']:
            if (output/'proxy'/name).read_bytes()!=(output/'baseline'/name).read_bytes():raise ValueError('proxy differs: '+name)
        terminal=(output/'baseline/terminal-proof.json').exists()
        if terminal != (output/'proxy/terminal-proof.json').exists():raise ValueError('terminal presence differs')
        if terminal and (output/'baseline/terminal-proof.json').read_bytes()!=(output/'proxy/terminal-proof.json').read_bytes():raise ValueError('terminal proof differs')
        count=compare(json.loads((output/'proxy/controller-calls.json').read_text()),read_trace(output/'recording/trace.jsonl'))
        review={'matched_primitive_calls':count,'budgets_world_replay_equal':True,'teacher_equivalence':False,'training_approved':False,'scope':'synthetic proxy diagnostic; wall-clock equivalence unproven'}
        (output/'proxy-review.json').write_text(json.dumps(review,indent=2)+'\n');return review
if __name__=='__main__':print(json.dumps(check(sys.argv[1]),indent=2))
