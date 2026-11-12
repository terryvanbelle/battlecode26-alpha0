"""Generate/compile isolated primitive controller hooks; never runs a teacher game."""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from alpha0.imitation.trace import OPERATIONS
from scripts.inventory_zdb_imitation_interface import code_only
SOURCE=ROOT/'vendor/engine/src/battlecode/world/RobotControllerImpl.java'

def transform(source):
    masked=code_only(source);edits=[];found=[]
    pattern=r'public\s+(void|boolean)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+GameActionException\s*)?\{'
    for m in re.finditer(pattern,masked):
        result,name,params=m.groups();args=[s.strip().split()[-1] for s in params.split(',') if s.strip()]
        op=name+('Amount' if name in ('attack','pickUpCheese') and len(args)==2 else '')
        if op not in OPERATIONS:continue
        if len(args)!=len(OPERATIONS[op]):raise ValueError('unexpected overload')
        start=m.end();depth=1;i=start
        while depth and i<len(masked):
            depth+=(masked[i]=='{')-(masked[i]=='}');i+=1
        if depth:raise ValueError('unbalanced source')
        body=source[start:i-1];signature=source[m.start():start-1].strip()
        helper='_imitationOriginal_'+op
        private_signature=signature.replace('public ','private ',1).replace(name+'(',helper+'(',1)
        # Source uses space before '(' only in signatures matched above; fail closed.
        if helper not in private_signature:raise ValueError('unexpected method formatting')
        throws=' throws GameActionException' if 'throws GameActionException' in signature else ''
        invoke=f'{helper}({", ".join(args)})'
        call=f'Boolean result = {invoke}; returned = result; return result;' if result=='boolean' else f'{invoke};'
        wrapper=signature+' {\n'+f'        alpha0.imitation.PrimitiveRecorder.Frame frame = alpha0.imitation.PrimitiveRecorder.begin(this, "{op}", new Object[]{{{", ".join(args)}}});\n'+'''        Throwable failure = null; Boolean returned = null;
        try { '''+call+''' }
        catch ('''+('GameActionException | ' if throws else '')+'''RuntimeException | Error e) { failure=e; throw e; }
        finally { alpha0.imitation.PrimitiveRecorder.finish(frame, returned, failure); }
    }
    '''+private_signature+' {'+body+'}'
        edits.append((m.start(),i,wrapper));found.append(op)
    if sorted(found)!=sorted(OPERATIONS):raise ValueError(f'primitive coverage differs: {set(OPERATIONS)-set(found)}')
    for a,b,text in reversed(edits):source=source[:a]+text+source[b:]
    return source,found

def transform_sandbox(source):
    begin=source.index('final Pauser pauser = () -> {')
    end=source.index('final Killer killer = () -> {',begin)
    pause=source[begin:end]
    for old,new in {
        'synchronized (notifier) {':'alpha0.imitation.RobotThreadRecorder.pausing();\n                synchronized (notifier) {',
        'notifier.wait();':'notifier.wait();\n                    alpha0.imitation.RobotThreadRecorder.resumed();',
    }.items():
        if pause.count(old)!=1:raise ValueError('pauser lifecycle contract changed')
        pause=pause.replace(old,new)
    source=source[:begin]+pause+source[end:]
    for old,new in {
        'initMethod.invoke(null, pauser, killer, this.seed, profiler, provider);':'initMethod.invoke(null, pauser, killer, this.seed, profiler, provider);\n                alpha0.imitation.RobotThreadRecorder.attach(robotController, getBytecodeNumMethod);',
        "// Ensure that we know we're terminated.":"alpha0.imitation.RobotThreadRecorder.detach(imitationReturned, imitationFailure);\n                // Ensure that we know we're terminated.",
    }.items():
        if source.count(old)!=1:raise ValueError('sandbox lifecycle contract changed')
        source=source.replace(old,new)
    # Capture the actual sandbox exit, preserving the original catch/return behavior.
    old='mainThread = new Thread(() -> {\n            try {'
    if source.count(old)!=1:raise ValueError('robot main-thread scope changed')
    source=source.replace(old,'mainThread = new Thread(() -> {\n            Throwable imitationFailure = null; boolean imitationReturned = false;\n            try {')
    old='loadAndRunPlayer(teamName, teamLanguage, PLAYER_CLASS_NAME);'
    if source.count(old)!=1:raise ValueError('player entry contract changed')
    source=source.replace(old,old+'\n                imitationReturned = true;')
    for kind in ('NonJavaBotException','IllegalAccessException','InvocationTargetException','InstrumentationException','RobotDeathException'):
        old='} catch (final '+kind+' e) {'
        if source.count(old)!=1:raise ValueError('sandbox exception contract changed: '+kind)
        cause='e.getCause()' if kind=='InvocationTargetException' else 'e'
        source=source.replace(old,old+'\n                imitationFailure = '+cause+';')
    return source

def build(output):
    output=Path(output).resolve()
    if output.exists():raise ValueError('exclusive build directory already exists')
    if not output.is_relative_to(ROOT/'build/imitation'):raise ValueError('isolated namespace required')
    source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest();text,ops=transform(SOURCE.read_text())
    generated=output/'src/battlecode/world/RobotControllerImpl.java';generated.parent.mkdir(parents=True);generated.write_text(text)
    sandbox_source=ROOT/'vendor/engine/src/battlecode/instrumenter/SandboxedRobotPlayer.java'
    sandbox=output/'src/battlecode/instrumenter/SandboxedRobotPlayer.java';sandbox.parent.mkdir(parents=True);sandbox.write_text(transform_sandbox(sandbox_source.read_text()))
    classes=output/'classes';classes.mkdir()
    hook=ROOT/'java/imitation/alpha0/imitation/PrimitiveRecorder.java'
    jar=ROOT/'vendor/engine/battlecode26-java-1.2.6.jar'
    subprocess.run([str(ROOT/'vendor/jdk/bin/javac'),'-cp',str(jar),'-d',str(classes),*[str(p) for p in sorted((ROOT/'java/imitation/alpha0/imitation').glob('*.java'))],str(generated),str(sandbox)],check=True,timeout=120)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
    record={'isolated_build':str(output.relative_to(ROOT)),'primitive_operations':ops,'source_sha256':source_hash,'hook_sha256':hashlib.sha256(hook.read_bytes()).hexdigest(),'observation_sha256':hashlib.sha256((ROOT/'java/imitation/alpha0/imitation/PublicObservation.java').read_bytes()).hexdigest(),'sandbox_source_sha256':hashlib.sha256(sandbox_source.read_bytes()).hexdigest(),'sandbox_generated_sha256':hashlib.sha256(sandbox.read_bytes()).hexdigest(),'thread_recorder_sha256':hashlib.sha256((ROOT/'java/imitation/alpha0/imitation/RobotThreadRecorder.java').read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(generated.read_bytes()).hexdigest(),'classes':{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in classes.rglob('*.class')},'engine_equivalence_proven':False,'teacher_games':0,'training_enabled':False}
    (output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));return record
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);build(p.parse_args().output)
